"""Check cited claims against the text associated with each citation label.

Unknown labels and unsupported claims receive warnings. Model failures produce
a verification notice so the answer pipeline can continue.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from llm_client import get_llm_client, model_names, output_token_limit


# ---------------------------------------------------------------------------
# LOAD API KEY
# ---------------------------------------------------------------------------

# Relative to this file's location, same pattern as agents.py / long_term.py —
# a hardcoded absolute path here would break on any machine but the one it
# was written on.
load_dotenv(Path(__file__).parent.parent / "Apikey.env")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Model name comes from llm_client.py's "check" role (same provider switch
# used everywhere else — see that module). Historically this needed to be
# a full-size model, not the cheapest tier: an earlier small-model check
# produced false-positive "unsupported" verdicts on genuinely well-grounded
# claims even with added context.
CHECK_MODEL = model_names()["check"]
CORRECTION_MODEL = model_names()["main"]

# Upper bound on how many individual citation checks to run per answer.
# Citation checks run one LLM call each, so this caps worst-case latency
# and cost on answers with many citations (mirrors the old code's
# `retrieved_chunks[:6]` cap, applied per-citation instead of per-answer).
MAX_CITATIONS_TO_CHECK = 10

# Prevent a correction request from growing without bound when many large
# retrieved chunks are available. The most useful context is kept in the same
# insertion order used to build chunk_by_label.
MAX_CORRECTION_CONTEXT_CHARS = 30_000

CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")

# Repairs the model's most common malformed-citation pattern: gluing its
# own "Art." citation prefix onto an "Article N" / "ARTICLE N" mention
# copied from prose or from a chunk's body text, e.g. "[Art. Article 210]"
# -> "[Art. 210]". This is a deterministic regex fix, not another LLM
# call — prompting reduces this behaviour but an LLM never follows
# formatting instructions with 100% reliability, so this is a backstop,
# not a replacement for the prompt-side fix in agents.py.
_MALFORMED_CITATION = re.compile(
    r"\[(?P<prefix>[^\[\]]*?\|\s*)?Art\.\s*(?:Article|ARTICLE)\s+"
    r"(?P<number>\d+[a-zA-Z\-]*)\]",
    re.IGNORECASE,
)


def _normalize_malformed_citations(answer: str) -> str:
    return _MALFORMED_CITATION.sub(
        lambda m: f"[{m.group('prefix') or ''}Art. {m.group('number')}]",
        answer,
    )


# Splits on sentence-ending punctuation followed by whitespace and then
# either a capital letter or an opening bracket (citations often sit right
# at the end of a sentence, e.g. "...atto pubblico [Art. 162]."). This is
# a heuristic, not a real sentence tokenizer — good enough for a
# best-effort guardrail, and specifically avoids splitting on things like
# "Art. 162" (the character after "Art." is a digit, which the lookahead
# does not match).
#
# Also splits at markdown list/heading boundaries — a numbered list item
# ("\n2.  **Timing:** ..."), a bullet ("\n- ..."), or a bold header
# ("\n**When it binds third parties:**") never follows a "[.!?] " pattern,
# so without these, a citation trailing one list item gets silently
# merged with the start of the NEXT, unrelated item into one oversized
# "sentence" (observed: a citation for item 1 checked against item 1's
# tail + all of item 2's text — a guaranteed false "doesn't support this
# claim" once the model style leans on markdown structure instead of
# plain prose).
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\[])"
    r"|(?<=\S)\n+(?=\d{1,3}\.\s)"
    r"|(?<=\S)\n+(?=[-*•]\s)"
    r"|(?<=\S)\n+(?=\*\*)"
)

# A fragment that's nothing but bracketed citations (e.g. a trailing
# "[Art. 162], [Art. 163], [Art. 210]" cluster at the end of a bullet
# point) has no claim of its own — the "-> [" branch of _SENTENCE_SPLIT
# above (needed for citations that genuinely open a new sentence) ends up
# cutting it away from the very prose it supports. Re-merging it onto the
# previous fragment keeps each citation attached to what it's actually
# backing, instead of floating alone with nothing to verify.
_CITATION_ONLY = re.compile(r"^(?:\[[^\[\]]+\]\s*,?\s*)+$")


# ---------------------------------------------------------------------------
# PARSING HELPERS
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> List[str]:
    raw = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    merged: List[str] = []
    for s in raw:
        if merged and _CITATION_ONLY.match(s):
            merged[-1] = f"{merged[-1]} {s}"
        else:
            merged.append(s)
    return merged


def _citations_in(sentence: str) -> List[str]:
    return CITATION_PATTERN.findall(sentence)


# A more precise citation than the corpus's labels support: source articles
# are often internally divided into numbered subsections (e.g. Slovenia's
# Art. 15 has "(1)... (2)... (3)..."), and a model citing "[Art. 15(1)]" is
# being MORE accurate, not inventing anything — but chunk_by_label's key is
# just "Art. 15" (the whole article is one chunk, not split by subsection),
# so an exact-match lookup on "Art. 15(1)" fails and gets reported as
# "label not found", a false positive. This strips a trailing "(N)"-style
# subsection reference for LOOKUP purposes only — the citation as displayed
# in the answer is untouched, only the guardrail's internal matching gets
# the fallback.
_SUBSECTION_SUFFIX = re.compile(
    r"^(?P<prefix>[^\[\]]*?\|\s*)?"
    r"(?P<article>Art\.?\s*\d+[a-zA-Z\-]*)\s*"
    r"\(\s*\d+[a-zA-Z]*\s*\)$",
    re.IGNORECASE,
)


def _base_article_label(label: str) -> Optional[str]:
    """ "Art. 15(1)" -> "Art. 15" for lookup purposes; None if `label`
    doesn't have a trailing subsection reference to strip."""
    m = _SUBSECTION_SUFFIX.match(label.strip())
    return f"{m.group('prefix') or ''}{m.group('article')}" if m else None


def _label_core(label: str) -> str:
    """Identifying part of a label, for cross-reference matching (e.g. "Art. 162" -> "162")."""
    m = re.search(r"\d+[a-zA-Z\-]*", label)
    return m.group(0) if m else ""


# Matches labels that are ENTIRELY made of one or more "Art. N" style
# tokens (e.g. "Art. 162", "Art. 73, Art. 74") — i.e. genuine article
# citations, as opposed to case-law labels (CASE_IDs), which are free-form
# strings that often contain digits of their own for unrelated reasons
# (dates, docket numbers, section numbers — e.g. "Court of Appeal of
# Salerno sec. II, 29/12/2022" contains "29" from the date). Used to gate
# _cross_referenced_source_text below: matching a bare number against retrieved
# text only makes sense for article numbers, never for a case citation's
# incidental digits.
_ARTICLE_LABEL = re.compile(
    r"^(?:[^\[\]]*?\|\s*)?Art(?:icle)?\.?\s*\d+[a-zA-Z\-]*"
    r"(?:\s*,\s*Art(?:icle)?\.?\s*\d+[a-zA-Z\-]*)*$",
    re.IGNORECASE,
)


def _sibling_countries(sentence: str, label: str, label_country: Dict[str, str]) -> set:
    """
    Countries of the OTHER labels cited in the same sentence as `label`,
    used as a proxy for which jurisdiction the sentence is actually
    about — see _cross_referenced_source_text. If the sentence cites a Slovenian
    case alongside the uncorroborated label, that's a strong signal the
    uncorroborated label should also be Slovenian.
    """
    countries = set()
    for other in _citations_in(sentence):
        if other == label:
            continue
        country = label_country.get(other)
        if country:
            countries.add(country)
    return countries


def _cross_referenced_source_text(
    label: str,
    chunk_by_label: Dict[str, str],
    label_country: Optional[Dict[str, str]] = None,
    sentence: str = "",
) -> Optional[str]:
    """
    Return retrieved source text that mentions `label`, when `label` was not
    itself retrieved — e.g. "Art. 162" is
    not in chunk_by_label, but the retrieved text of "Art. 210" literally
    says "...in accordance with Article 162...". In that case the model
    isn't inventing Article 162 out of nowhere: it's reporting what an
    actually-retrieved source says about it. Flagging that as "invented"
    produces exactly the kind of false alarm that erodes trust in the
    guardrail, so this is treated as acceptable rather than unknown.

    Countries matter here: article numbers are small integers that can
    coincidentally exist in more than one country's civil code with
    unrelated content (e.g. both Italy and Slovenia could have their own,
    completely different, "Article 162"). On a comparative multi-country
    answer, a bare number match against ANY retrieved chunk — regardless
    of which country it came from — can "corroborate" a citation that is
    actually a genuine attribution error. When `label_country` is
    supplied and at least one sibling citation in the same sentence has a
    known country, the match is restricted to chunks from that same
    country. If no sibling has a known country, there is no signal to
    scope by, so it falls back to the old unscoped behaviour rather than
    getting stricter with nothing to go on.
    """
    # Restrict this fallback to genuine article-style labels — see
    # _ARTICLE_LABEL above. A case-law label's incidental digits (dates,
    # docket numbers) matching some unrelated "Art. N" mention elsewhere
    # is a coincidence, not corroboration.
    if not _ARTICLE_LABEL.match(label.strip()):
        return None

    core = _label_core(label)
    if not core:
        return None
    pattern = re.compile(rf"\bArt(?:icle)?\.?\s*{re.escape(core)}\b", re.IGNORECASE)

    label_country = label_country or {}
    sibling_countries = (
        _sibling_countries(sentence, label, label_country) if sentence else set()
    )

    for other_label, text in chunk_by_label.items():
        if not pattern.search(text):
            continue
        if sibling_countries:
            other_country = label_country.get(other_label)
            if other_country and other_country not in sibling_countries:
                # Same article number, but from a different country's
                # source than this sentence appears to be about — likely
                # a numeric coincidence, not real corroboration. Keep
                # looking rather than accepting this match.
                continue
        return text
    return None


def _extract_citation_claims(answer: str) -> List[Tuple[str, str]]:
    """
    Returns a de-duplicated, order-preserving list of (label, sentence)
    pairs: every distinct combination of a cited label and the sentence
    that cites it.
    """
    seen = set()
    pairs: List[Tuple[str, str]] = []
    for sentence in _split_sentences(answer):
        for label in _citations_in(sentence):
            key = (label, sentence)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return pairs


def _prioritize_claims(claims: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Reorders claims so the FIRST occurrence of every distinct label comes
    before any repeat citation of a label already seen. Order within each
    group is preserved.

    MAX_CITATIONS_TO_CHECK caps how many claims actually get checked per
    answer — without this, a label cited in five different sentences
    could burn through the whole budget on repeats of one already-verified
    source, leaving a different (and possibly wrong) citation elsewhere in
    the same answer completely unchecked. Checking each distinct label at
    least once first makes better use of a limited budget than a naive
    first-N-in-order slice.
    """
    seen_labels = set()
    first_pass: List[Tuple[str, str]] = []
    repeats: List[Tuple[str, str]] = []
    for label, sentence in claims:
        if label not in seen_labels:
            seen_labels.add(label)
            first_pass.append((label, sentence))
        else:
            repeats.append((label, sentence))
    return first_pass + repeats


# ---------------------------------------------------------------------------
# PER-CITATION CHECK
# ---------------------------------------------------------------------------


def _check_single_citation(
    sentence: str,
    label: str,
    source_text: str,
    llm_client: OpenAI,
    full_answer: str = "",
) -> Optional[bool]:
    """
    Returns True if the source supports the claim, False if it
    contradicts/doesn't support it, or None if the check itself could not
    be completed (transient error) — None is treated as "unverifiable",
    not as a failure, by the caller.

    `full_answer` is passed purely as context for resolving pronouns/
    references in `sentence` (e.g. "This agreement...", "It also...") —
    checking a sentence in total isolation can make an otherwise-grounded
    claim look unsupported simply because its antecedent is missing, not
    because the source doesn't back it up. The verdict must still be
    about the SENTENCE specifically, not the answer as a whole.
    """
    try:
        context_block = (
            f"FULL ANSWER (context only, to resolve pronouns/references — "
            f"do not judge this as a whole):\n{full_answer}\n\n"
            if full_answer
            else ""
        )
        # The visible verdict is deliberately limited to YES/NO.
        response = llm_client.chat.completions.create(
            model=CHECK_MODEL,
            max_tokens=output_token_limit(10, gemini=4096),
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a citation checker. Given a SOURCE and a "
                        "SENTENCE that cites it, decide if the SOURCE actually "
                        "supports the SENTENCE. Paraphrasing, summarizing, or "
                        "combining information from different parts of the "
                        "SOURCE still counts as supported. If the SENTENCE "
                        "contains a pronoun or reference (e.g. 'this "
                        "agreement', 'it'), use the FULL ANSWER below only to "
                        "figure out what it refers to. If the SENTENCE cites "
                        "this SOURCE together with another source, and this "
                        "SOURCE is only referenced incidentally (e.g. as the "
                        "place where a specific amount, share, or rule is "
                        "defined, rather than as the main subject of the "
                        "sentence), it is enough for the SOURCE to be "
                        "consistent with that specific incidental point — it "
                        "does not need to justify the entire sentence on its "
                        "own. Only flag NO if, once all that is accounted "
                        "for, the SOURCE contradicts the SENTENCE or says "
                        "nothing relevant to it at all.\n\n"
                        "Reply with exactly one word: YES or NO.\n\n"
                        f"{context_block}"
                        f"SOURCE [{label}]:\n{source_text}\n\n"
                        f"SENTENCE TO CHECK:\n{sentence}"
                    ),
                }
            ],
        )
        verdict = response.choices[0].message.content.strip().upper()
    except Exception as e:
        logger.warning("Citation check failed for [%s]: %s", label, e)
        return None

    if verdict.startswith("YES"):
        return True
    if verdict.startswith("NO"):
        return False

    logger.warning("Unrecognized citation-check verdict for [%s]: %r", label, verdict)
    return None


def _rewrite_with_grounding_feedback(
    answer: str,
    chunk_by_label: Dict[str, str],
    issues: List[Tuple[str, str, str]],
    llm_client: OpenAI,
) -> Optional[str]:
    """Rewrite an answer once, using the detected issues and retrieved text.

    Returns None when the correction call fails or produces an empty response.
    The caller is responsible for running the corrected answer through the
    guardrail again; this function deliberately does not decide that its own
    rewrite is safe.
    """
    issue_lines = "\n".join(
        f'- [{label}] in "{sentence}" — {reason}' for label, sentence, reason in issues
    )

    source_parts: List[str] = []
    chars_used = 0
    issue_labels = [label for label, _, _ in issues]
    prioritized_labels = list(
        dict.fromkeys(
            [
                base
                if (base := _base_article_label(label)) in chunk_by_label
                else label
                for label in issue_labels
            ]
            + list(chunk_by_label)
        )
    )
    for label in prioritized_labels:
        text = chunk_by_label.get(label)
        if text is None:
            continue
        block = f"SOURCE [{label}]:\n{text}\n"
        remaining = MAX_CORRECTION_CONTEXT_CHARS - chars_used
        if remaining <= 0:
            break
        source_parts.append(block[:remaining])
        chars_used += min(len(block), remaining)

    try:
        response = llm_client.chat.completions.create(
            model=CORRECTION_MODEL,
            max_tokens=output_token_limit(2048, gemini=8192),
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "The response below has citation-grounding problems. "
                        "Rewrite it before it is shown to the user, using only "
                        "information supported by the supplied sources. Correct "
                        "or remove unsupported claims and unknown/misattributed "
                        "citations. Preserve useful supported content and cite "
                        "sources only with their exact bracketed labels. Do not "
                        "mention this correction process and do not add a preamble.\n\n"
                        f"DETECTED PROBLEMS:\n{issue_lines}\n\n"
                        f"ORIGINAL RESPONSE:\n{answer}\n\n"
                        "RETRIEVED SOURCES:\n" + "\n".join(source_parts)
                    ),
                }
            ],
        )
        corrected = response.choices[0].message.content.strip()
        return corrected or None
    except Exception as e:
        logger.warning("Automatic grounding correction failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# GROUNDING CHECK (public entry point — name kept for backward compatibility)
# ---------------------------------------------------------------------------


def check_grounding(
    answer: str,
    chunk_by_label: Dict[str, str],
    llm_client: OpenAI = None,
    label_country: Optional[Dict[str, str]] = None,
    _correction_attempted: bool = False,
) -> str:
    """
    Verify that every "[label]" citation in `answer` is supported by its
    own labelled source text.

    Parameters
    ----------
    answer : str
        The answer generated by the agents (after aggregation), containing
        zero or more "[label]" citations.
    chunk_by_label : Dict[str, str]
        Map from citation label (e.g. "Art. 162", or a CASE_ID) to the raw
        source text retrieved for it. Built by the caller from every
        agent's PartialAnswer.labeled_chunks.
    llm_client : OpenAI
        OpenAI-compatible client (see llm_client.py). Created automatically
        if not provided.
    label_country : Dict[str, str], optional
        Map from citation label to the country of the source it came
        from. Used only to scope the indirect-reference fallback check
        (an uncited-but-mentioned label) by jurisdiction, so a same-
        numbered article from an unrelated country's retrieved chunk
        isn't mistaken for corroboration. Safe to omit — falls back to
        the old unscoped behaviour.

    Returns
    -------
    str
        The original answer if every citation checks out (or if the
        answer has no citations to check), or the answer with a warning
        prepended listing exactly which citations failed, could not be
        verified, or were not checked at all (per-answer check limit).
    """
    label_country = label_country or {}
    if not chunk_by_label:
        return (
            "[WARNING: no source documents were retrieved for this answer.]\n\n"
            + answer
        )

    # Fix known formatting slips (e.g. "[Art. Article 210]" -> "[Art. 210]")
    # before extracting citations, so a malformed-but-fixable label isn't
    # misread as pointing at a source that doesn't exist. The repaired
    # text is also what gets returned to the user, so they see the clean
    # label too, not the mangled one.
    answer = _normalize_malformed_citations(answer)

    logger.debug("chunk_by_label keys: %s", list(chunk_by_label.keys()))

    try:
        claims = _prioritize_claims(_extract_citation_claims(answer))
    except Exception as e:
        logger.warning("Citation extraction failed, returning answer unchecked: %s", e)
        return (
            "[NOTE: the grounding check could not be completed for this "
            "answer due to a temporary error.]\n\n" + answer
        )

    if not claims:
        # No bracketed citations found at all — nothing to verify against
        # a specific source. This does not necessarily mean the answer is
        # wrong (e.g. a short "no relevant documents" reply), so no
        # warning is added; this differs from the "no chunks retrieved"
        # case above, which IS worth flagging.
        if _correction_attempted:
            return (
                "[NOTE: the guardrail corrected citation-grounding issues and "
                "the revised answer passed the second check.]\n\n" + answer
            )
        return answer

    logger.debug("Extracted citation claims: %s", claims)

    if llm_client is None:
        llm_client = get_llm_client()

    # Each entry is (label, sentence) — keeping the specific sentence, not
    # just the label, means the warning shown to the user (and to whoever
    # is debugging it) points straight at the exact claim that failed
    # instead of requiring a full log trace to find it, which is what
    # made diagnosing the last few false positives in this session slower
    # than it needed to be.
    unknown_label: List[
        Tuple[str, str]
    ] = []  # cited a label that isn't in chunk_by_label at all
    unsupported: List[
        Tuple[str, str]
    ] = []  # label exists, but source doesn't support the claim
    unverifiable: List[Tuple[str, str]] = []  # check itself failed (transient error)

    # Claims beyond the per-answer cap: not wrong, just never checked —
    # reported separately (below) so that stays visible instead of
    # silently looking identical to "everything passed".
    checked_claims = claims[:MAX_CITATIONS_TO_CHECK]
    skipped_claims = claims[MAX_CITATIONS_TO_CHECK:]

    for label, sentence in checked_claims:
        source_text = chunk_by_label.get(label)

        if source_text is None:
            base_label = _base_article_label(label)
            if base_label and base_label in chunk_by_label:
                source_text = chunk_by_label[base_label]
                logger.debug(
                    "Citation [%s] matched base article [%s] (subsection "
                    "reference stripped for lookup).",
                    label,
                    base_label,
                )

        if source_text is None:
            source_text = _cross_referenced_source_text(
                label, chunk_by_label, label_country, sentence
            )
            if source_text is not None:
                logger.debug(
                    "Citation [%s] not directly retrieved, but mentioned by "
                    "another source; verifying the claim against that source.",
                    label,
                )
            else:
                unknown_label.append((label, sentence))
                continue

        try:
            verdict = _check_single_citation(
                sentence, label, source_text, llm_client, full_answer=answer
            )
        except Exception as e:
            # Belt-and-braces: _check_single_citation already catches its
            # own exceptions, but nothing here should ever be able to take
            # the whole pipeline down.
            logger.warning("Unexpected error checking citation [%s]: %s", label, e)
            verdict = None

        if verdict is True:
            continue
        elif verdict is False:
            unsupported.append((label, sentence))
        else:
            unverifiable.append((label, sentence))

    # Every problem category found gets reported together — earlier
    # versions of this check returned as soon as the first non-empty
    # category was found, which silently dropped any other real issues
    # (e.g. a second, differently-broken citation in the same answer)
    # from the warning shown to the user.
    if not (unknown_label or unsupported or unverifiable or skipped_claims):
        if _correction_attempted:
            return (
                "[NOTE: the guardrail corrected citation-grounding issues and "
                "the revised answer passed the second check.]\n\n" + answer
            )
        return answer

    # Rewrite only for concrete grounding failures. A timeout/unrecognized
    # checker verdict and claims skipped due to the cap do not prove that the
    # answer itself is wrong, so rewriting for those cases could make it worse.
    concrete_issues: List[Tuple[str, str, str]] = [
        (label, sentence, "citation label was not found in the retrieved sources")
        for label, sentence in unknown_label
    ] + [
        (label, sentence, "the cited source does not support this claim")
        for label, sentence in unsupported
    ]

    correction_failed = False
    if concrete_issues and not _correction_attempted:
        corrected = _rewrite_with_grounding_feedback(
            answer, chunk_by_label, concrete_issues, llm_client
        )
        if corrected:
            # Exactly one retry: the private flag prevents an infinite
            # correction loop if the rewritten answer still has problems.
            return check_grounding(
                corrected,
                chunk_by_label,
                llm_client,
                label_country=label_country,
                _correction_attempted=True,
            )
        correction_failed = True

    def _snippet(sentence: str, max_len: int = 100) -> str:
        s = sentence.strip()
        return s if len(s) <= max_len else s[:max_len].rstrip() + "..."

    def _format_group(entries: List[Tuple[str, str]], reason: str) -> List[str]:
        """One line per distinct (label, sentence) pair, so the exact
        claim that failed is visible without needing the debug log."""
        out = []
        for label, sentence in dict.fromkeys(entries):
            out.append(f'  - [{label}] "{_snippet(sentence)}" — {reason}')
        return out

    if _correction_attempted:
        lines = [
            "[WARNING: the guardrail corrected the answer, but the second "
            "check still found possible citation issue(s).]"
        ]
    else:
        lines = ["[WARNING: possible citation issue(s) in this answer]"]

    if correction_failed:
        lines.append(
            "  - An automatic correction was attempted but could not be completed."
        )

    lines += _format_group(
        unknown_label,
        "label not found among the retrieved sources; may be invented or mislabeled.",
    )
    lines += _format_group(
        unsupported,
        "source found, but does not appear to support this specific claim "
        "— may be correct but misattributed, or incorrect.",
    )
    lines += _format_group(
        unverifiable,
        "could not be verified due to a temporary error.",
    )

    if skipped_claims:
        skipped_labels = list(dict.fromkeys(label for label, _ in skipped_claims))
        lines.append(
            f"  - {len(skipped_claims)} additional citation(s) not checked "
            f"(labels: {', '.join(skipped_labels)}) — per-answer "
            f"verification limit of {MAX_CITATIONS_TO_CHECK} reached."
        )

    return "\n".join(lines) + "\n\n" + answer
