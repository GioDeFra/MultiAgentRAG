"""
agents.py — Core agent classes for the multi-agent legal RAG system.

Classes
-------
SpecializedAgent
    Queries its slice of the single Pinecone index ("legal-rag") via a
    metadata filter (see config.py), reranks the candidates, and
    generates a partial answer. Can also receive relevant long-term-memory
    context (similar past Q&A) as supporting background.

SupervisorAgent
    Entry point for user questions. Triages (direct vs retrieve),
    routes to the right specialized agents, aggregates partial answers,
    and runs the output guardrail. Also manages short-term, long-term,
    and chat-history memory.
"""
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Environment variables, including HF_TOKEN, must be loaded before importing
# sentence-transformers/Hugging Face. Otherwise a token present only in
# Apikey.env is invisible while those libraries initialise.
load_dotenv(Path(__file__).parent / "Apikey.env")
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import AGENT_MAP, AGENT_REGISTRY, AgentDescription
from guardrails.output_guard import check_grounding
from llm_client import get_llm_client, model_names
from memory.chat_history import ChatHistoryStore
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory, Turn


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Model names now come from llm_client.py, so they can be provider-specific and centrally managed.
_MODELS = model_names()

EMBEDDING_MODEL  = "BAAI/bge-m3"  # MUST match the model used in Ingestion.ipynb.

RERANKER_MODEL   = "BAAI/bge-reranker-v2-m3"  # Same family as EMBEDDING_MODEL
                                              # (BGE-M3).

# Long-term memory lives in its own ChromaDB store (long_term.py).
# Keeping LTM on its original model avoids re-embedding existing Q&A.
LTM_EMBEDDING_MODEL = "all-mpnet-base-v2"


PINECONE_INDEX_NAME = "legal-rag"
DB_DIR = "./chroma_db"   # long-term memory store only, not the RAG corpus
MAX_PARALLEL_AGENTS = 4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class PartialAnswer:
    agent_id: str
    agent_name: str
    answer: str

    # Sources shown to the LLM as:
    # (citation label, text and metadata, country, source identifier).
    # Used by the output guardrail to verify citations against the correct source.

    labeled_chunks: List[Tuple[str, str, str, str]]
    retrieved_documents: List[dict]
    relevance_score: float


# ---------------------------------------------------------------------------
# CONTEXT / CITATION HELPERS
# ---------------------------------------------------------------------------

_ARTICLE_HEADING = re.compile(
    r"^\s*(?:(?:ARTICLE|ART\.?)\s+)?(?P<number>\d+[A-Za-z]*(?:[-./][A-Za-z0-9]+)?)\.?"
    r"[ \t]*(?:\r?\n|$)",
    re.IGNORECASE,
)


def _citation_label(meta: dict) -> str:
    """Read the canonical label written by ingestion; never invent one."""
    return str(meta.get("citation_label") or "").strip()


def _strip_redundant_heading(text: str, label: str) -> str:
    """Remove a repeated leading article number from statute text."""
    if "Art." not in label:
        return text
    return _ARTICLE_HEADING.sub("", text, count=1)


def _defang_bracketed_labels(text: str) -> str:
    """Prevent labels recalled from long-term memory becoming citations."""
    return text.replace("[", "(").replace("]", ")")


_CONTEXT_LINE_SKIP_KEYS = {
    "CASE_ID", "citation_label", "country", "doc_type", "law", "source",
    "text", "chunk_index", "n_chunks",
}
_CONTEXT_LINE_PLACEHOLDER_VALUES = {
    "", "no data", "not specified", "n/a", "unknown",
}

SOURCE_EXCERPT_CHARS = 320


def _source_name(source: str) -> str:
    """Return the final path component for either POSIX or Windows paths."""
    return re.split(r"[/\\]", source)[-1] or source


def _article_references(meta: dict, label: str) -> List[str]:
    """Collect article references from metadata, falling back to the label."""
    values = []
    for key in ("article", "articles", "civil_codes_used"):
        value = meta.get(key)
        if isinstance(value, list):
            values.extend(str(item).strip() for item in value if str(item).strip())
        elif value not in (None, ""):
            values.append(str(value).strip())
    if not values:
        values = re.findall(r"(?:Art\.?|Article|§)\s*\d+[A-Za-z]*(?:[-./]\w+)?", label, re.IGNORECASE)
    return list(dict.fromkeys(values))


def _source_appendix(documents: List[dict]) -> str:
    """Render compact retrieval evidence for display below the answer."""
    if not documents:
        return (
            "\n\n---\n\n### Sources used\n\n"
            "No documentary sources were retrieved for this response."
        )
    lines = ["\n\n---\n\n### Sources used"]
    for index, document in enumerate(documents, start=1):
        articles = ", ".join(document["articles"]) or "Not specified"
        lines.extend([
            f"\n**{index}. {document['document_name']}**",
            f"- Country: {document['country'] or 'Not specified'}",
            f"- Document type: {document['document_type'] or 'Not specified'}",
            f"- Citation label: `{document['citation_label']}`",
            f"- Articles mentioned: {articles}",
            f"- Source: `{document['source']}`",
            f"- Relevant excerpt: “{document['excerpt']}”",
        ])
    return "\n".join(lines)


def _deduplicate_documents(documents: List[dict]) -> List[dict]:
    """Keep one display/log entry per source and citation label."""
    unique = {}
    for document in documents:
        key = (document["source"], document["citation_label"])
        existing = unique.get(key)
        if existing is None or document["relevance_score"] > existing["relevance_score"]:
            unique[key] = document
    return list(unique.values())


def _build_metadata_line(meta: dict) -> str:
    """Render useful source metadata without repeating internal fields."""
    parts = []
    for key, value in meta.items():
        if key in _CONTEXT_LINE_SKIP_KEYS or value in (None, [], {}):
            continue
        if isinstance(value, str) and value.strip().lower() in _CONTEXT_LINE_PLACEHOLDER_VALUES:
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        parts.append(f"{key}: {value}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# SPECIALIZED AGENT
# ---------------------------------------------------------------------------

class SpecializedAgent:
    """
    One node in the multi-agent system.
    Responsible for retrieval (Pinecone, filtered) + reranking + answer
    generation over its own slice of the knowledge base.
    """

    def __init__(
        self,
        description: AgentDescription,
        pinecone_index,
        embed_model: SentenceTransformer,
        llm_client: OpenAI,
        reranker: CrossEncoder,
        n_retrieve: int = 20,   # candidates from Pinecone before reranking
        # Chunks passed to LLM after reranking. Originally raised from 5 to
        # 8 (see git history) because Art. 162 had ranked #8 post-rerank
        # with the old all-mpnet-base-v2 + ms-marco-MiniLM-L-6-v2 pipeline
        # and got cut at top_k=5. Re-tested after switching to BGE-M3 +
        # bge-reranker-v2-m3 (discussed in chat): on that same query,
        # Art. 162 now ranks #3 post-rerank (chunk_by_label keys observed:
        # ['Art. 159', 'Art. 163', 'Art. 162', ...]), comfortably inside a
        # 5-slot window — the new pipeline retrieves/ranks this corpus
        # better. Lowered back to 5 on that evidence. If a similarly
        # relevant article ever gets cut again, check the DEBUG-level
        # `chunk_by_label keys` log line (guardrails/output_guard.py) for
        # this agent's query — it reflects the actual post-rerank order —
        # before raising this back up.
        top_k: int = 5,
    ) -> None:
        self.description    = description
        self.pinecone_index = pinecone_index
        self.embed_model    = embed_model
        self.llm_client    = llm_client
        self.reranker       = reranker
        self.n_retrieve     = n_retrieve
        self.top_k          = top_k

        if not description.pinecone_filter:
            print(f"  [WARN] Agent '{description.agent_id}' has no pinecone_filter "
                  f"configured — it will match ANY vector in the index.")

    # ── Retrieval ────────────────────────────────────────────────────────────

    def _retrieve(self, query: str) -> List[Tuple[str, dict]]:
        """
        Embed the query and search this agent's slice of the Pinecone
        index (via its metadata filter). Returns (text, metadata) pairs,
        best match first (Pinecone already returns matches sorted by
        descending similarity score).
        """
        try:
            # normalize_embeddings=True to match how vectors were written
            # at ingestion time (Ingestion.ipynb, upsert_records) —
            # BGE-M3's own docs recommend normalised embeddings for
            # cosine-similarity retrieval.
            query_vector = self.embed_model.encode(
                [query], normalize_embeddings=True
            )[0].tolist()
            result = self.pinecone_index.query(
                vector=query_vector,
                filter=self.description.pinecone_filter or None,
                top_k=self.n_retrieve,
                include_metadata=True,
            )
        except Exception as e:
            print(f"  [WARN] Pinecone retrieval error for agent "
                  f"'{self.description.agent_id}': {e}")
            return []

        candidates = []
        for match in result.get("matches", []):
            meta = match.get("metadata", {}) or {}
            text = meta.get("text", "")
            if text:
                candidates.append((text, meta))

        # Debug aid: raw Pinecone ranking (pre-rerank) for this agent's
        # query, so a document that never shows up in the final answer
        # can be checked against — did it fail to make even the top-20
        # nearest neighbours (embedding/similarity issue), or did it get
        # demoted by the cross-encoder reranker afterwards (see _rerank)?
        logger.debug(
            "Raw Pinecone candidates for '%s...': %s",
            query[:60],
            [
                (m.get("metadata", {}).get("civil_codes_used")
                 or m.get("metadata", {}).get("CASE_ID")
                 or m.get("metadata", {}).get("source"),
                 round(m.get("score", 0), 4))
                for m in result.get("matches", [])
            ],
        )
        return candidates

    # ── Reranking ────────────────────────────────────────────────────────────

    def _rerank(
        self, query: str, candidates: List[Tuple[str, dict]]
    ) -> List[Tuple[float, str, dict]]:
        """Use a cross-encoder to rerank candidates by true relevance."""
        if not candidates:
            return []

        # Score against the same enriched text later shown to the LLM
        # (metadata line + raw chunk), not the raw chunk alone. Some facts
        # (a cost figure, succession_type, marital_regime, etc.) live only
        # in metadata and are never spelled out in the chunk's prose (see
        # _build_metadata_line's docstring) — grounding_text downstream
        # already includes meta_line for exactly this reason. Scoring on
        # raw text only meant a chunk that's genuinely the right answer to
        # a metadata-driven question could get pushed out of top_k by the
        # cross-encoder before the LLM ever had a chance to see it.
        pairs = []
        for doc, meta in candidates:
            meta_line = _build_metadata_line(meta)
            scoring_text = f"{meta_line}\n{doc}" if meta_line else doc
            pairs.append((query, scoring_text))

        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            (float(score), document, metadata)
            for score, (document, metadata) in ranked[: self.top_k]
        ]

    # ── Answer generation ────────────────────────────────────────────────────

    def answer(self, query: str, ltm_context: str = "") -> PartialAnswer:
        """
        Full pipeline: retrieve (Pinecone) → rerank → generate answer.

        Parameters
        ----------
        ltm_context : str, optional
            Pre-formatted block of relevant past Q&A pulled from long-term
            memory (see SupervisorAgent._format_ltm_context). Passed as
            supporting background only — the retrieved documents remain
            the primary and authoritative source.
        """
        candidates = self._retrieve(query)
        top_chunks = self._rerank(query, candidates)

        if not top_chunks:
            return PartialAnswer(
                agent_id=self.description.agent_id,
                agent_name=self.description.agent_id,
                answer="No relevant documents found in this agent's knowledge base.",
                labeled_chunks=[],
                retrieved_documents=[],
                relevance_score=0.0,
            )

        # Build context block. Field names match what Ingestion.ipynb
        # actually writes to Pinecone metadata: "text" (the raw chunk) and
        # "source" (the source file path) — NOT "raw_chunk"/"source_file".
        #
        # Each chunk is labelled during ingestion and the canonical
        # citation_label metadata is used here verbatim. Query-time code must
        # never derive a different label from article references or headings.
        prepared_chunks = []
        sources_by_label: Dict[str, str] = {}
        ambiguous_labels = set()
        for _score, doc, meta in top_chunks:
            label = _citation_label(meta)
            source = str(meta.get("source") or "unknown source")
            if not label:
                logger.warning(
                    "Skipping retrieved chunk without citation_label (source=%s). "
                    "Re-run ingestion if this is an old Pinecone record.",
                    source,
                )
                continue
            previous_source = sources_by_label.get(label)
            if previous_source is not None and previous_source != source:
                ambiguous_labels.add(label)
            else:
                sources_by_label[label] = source
            prepared_chunks.append((_score, doc, meta, label, source))

        if ambiguous_labels:
            logger.warning(
                "Ignoring non-unique citation labels returned to agent %s: %s",
                self.description.agent_id,
                sorted(ambiguous_labels),
            )

        context_parts = []
        labeled_chunks: List[Tuple[str, str, str, str]] = []
        retrieved_documents: List[dict] = []
        retained_scores: List[float] = []
        for score, doc, meta, label, source in prepared_chunks:
            # A label shared by different source documents cannot be verified
            # safely. Do not rename it at query time: ingestion owns labels.
            if label in ambiguous_labels:
                continue
            meta_line = _build_metadata_line(meta)
            source_text = str(meta.get("text") or doc or "")
            raw_chunk = _strip_redundant_heading(source_text, label)
            country = str(meta.get("country") or "")
            # Include meta_line in what the guardrail checks against, not
            # just raw_chunk — see PartialAnswer.labeled_chunks docstring.
            grounding_text = f"{meta_line}\n{raw_chunk}" if meta_line else raw_chunk
            labeled_chunks.append((label, grounding_text, country, source))
            excerpt = re.sub(r"\s+", " ", raw_chunk).strip()
            if len(excerpt) > SOURCE_EXCERPT_CHARS:
                excerpt = excerpt[:SOURCE_EXCERPT_CHARS].rsplit(" ", 1)[0] + "…"
            retrieved_documents.append({
                "agent_id": self.description.agent_id,
                "citation_label": label,
                "document_name": _source_name(source),
                "source": source,
                "country": country,
                "document_type": str(meta.get("doc_type") or ""),
                "legal_area": str(meta.get("law") or ""),
                "articles": _article_references(meta, label),
                "excerpt": excerpt,
                "relevance_score": round(float(score), 4),
            })
            retained_scores.append(score)
            context_parts.append(
                f"[{label}] {meta_line} | Source: {source}\n{raw_chunk}"
            )

        if not context_parts:
            return PartialAnswer(
                agent_id=self.description.agent_id,
                agent_name=self.description.agent_id,
                answer="No unambiguous, citable documents were found in this agent's knowledge base.",
                labeled_chunks=[],
                retrieved_documents=[],
                relevance_score=0.0,
            )
        context = "\n\n---\n\n".join(context_parts)

        user_content = f"Question: {query}\n\nContext:\n{context}"
        if ltm_context:
            user_content += f"\n\n{ltm_context}"

        # Generate a citation-grounded answer.
        response = self.llm_client.chat.completions.create(
            model=_MODELS["main"],
            max_tokens=1024,
            # Low, not zero: still needs to write fluent prose, but a high
            # default temperature gives the model more room to "fill gaps"
            # with plausible-sounding content not actually in the
            # retrieved context (observed: fabricated article numbers,
            # a case citation never retrieved this turn) — keeping it low
            # biases toward sticking to what's actually in the prompt.
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a specialized legal assistant covering:\n"
                        f"  Countries   : {', '.join(self.description.countries)}\n"
                        f"  Legal areas : {', '.join(self.description.legal_areas)}\n"
                        f"  Doc types   : {', '.join(self.description.content_types)}\n\n"
                        "Answer using ONLY the provided context documents.\n"
                        "Each document below is preceded by its citation label in "
                        "square brackets, e.g. [Italy | Divorce | Art. 162] or "
                        "[Italy | Divorce | Court of Appeal of Salerno sec. II, "
                        "29/12/2022]. When you reference a document, "
                        "copy that exact bracketed label character-for-character — "
                        "never renumber it, never invent a label, never attribute a "
                        "claim to a label other than the one attached to the document "
                        "that actually supports it, and never build a new citation by "
                        "combining the word 'Art.' with any article number or heading "
                        "you find inside a document's body text. The ONLY valid "
                        "citation for a document is the exact bracketed label shown "
                        "immediately before it.\n"
                        "If a fact is not supported by any of the provided documents, "
                        "say so instead of citing your own general knowledge.\n"
                        "If you cite two or more labels together for a single "
                        "specific claim (e.g. '[Art. 71] and [Art. 75]'), make sure "
                        "EVERY one of those labels individually supports that exact "
                        "claim — not just the general topic. Two documents about the "
                        "same general subject are not automatically both citable for "
                        "the same specific sentence: if only one of them actually "
                        "supports the point being made, cite only that one, and give "
                        "the other document its own separate sentence for whatever "
                        "specific point it does support.\n"
                        "Distinguish clearly between case-law and legislation.\n"
                        "If the context is insufficient, say so explicitly.\n\n"
                        "You may also receive a 'Relevant past Q&A' block with "
                        "similar questions answered in previous sessions. Treat "
                        "it only as supporting background, never as authoritative "
                        "over the retrieved documents — if it conflicts with the "
                        "context documents, the documents win. Any article or "
                        "case identifier mentioned inside that block is NOT a "
                        "valid citation for this answer — the only valid "
                        "citations are the exact bracketed labels shown in the "
                        "Context section above. If something in the background "
                        "block seems relevant, find the matching document in "
                        "the Context section and cite that label instead; if "
                        "there is no matching document there, do not cite "
                        "anything for that point."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        )

        answer_text = response.choices[0].message.content

        # Every agent uses the same cross-encoder, so its best reranker score
        # is meaningful for ordering partial answers during aggregation.
        relevance = round(max(retained_scores), 4)

        return PartialAnswer(
            agent_id=self.description.agent_id,
            agent_name=self.description.agent_id,
            answer=answer_text,
            labeled_chunks=labeled_chunks,
            retrieved_documents=retrieved_documents,
            relevance_score=relevance,
        )


# ---------------------------------------------------------------------------
# SUPERVISOR AGENT
# ---------------------------------------------------------------------------

class SupervisorAgent:
    """
    Orchestrator that:
      1. Triages the question (direct LLM answer vs retrieval)
      2. Routes to the right specialized agents
      3. Aggregates partial answers
      4. Runs the output guardrail
      5. Updates short-term, long-term, and chat-history memory
    """

    def __init__(
        self,
        pinecone_index_name: str = PINECONE_INDEX_NAME,
        db_dir: str = DB_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        chat_history_db_path: str = "./chat_history.db",
    ) -> None:
        self.llm_client = get_llm_client()

        # Shared infrastructure
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise RuntimeError("Missing PINECONE_API_KEY — add it to Apikey.env")
        pc = Pinecone(api_key=pinecone_api_key)
        pinecone_index = pc.Index(pinecone_index_name)

        # Query-time embedding model — MUST match whatever ingestion wrote
        # to Pinecone (see EMBED_MODEL_NAME in Ingestion.ipynb).
        # Shared across all agents since it's only used to embed the
        # user's question, not documents.
        embed_model = SentenceTransformer(embedding_model)

        reranker = CrossEncoder(RERANKER_MODEL)

        # Memory
        # Short-term memory is a pure 10-turn RAM window. Appending turn 11
        # discards turn 1 without making an additional LLM call.
        self.stm = ShortTermMemory(max_turns=10)
        # NOTE: long-term Q&A summary memory still uses ChromaDB — a
        # separate concern from the Pinecone-backed RAG document corpus,
        # with its own embedding model (LTM_EMBEDDING_MODEL, see above) —
        # NOT `embedding_model`, deliberately, so LTM is unaffected by
        # changes to the RAG pipeline's embedding model.
        self.ltm = LongTermMemory(db_dir=db_dir, embedding_model=LTM_EMBEDDING_MODEL)

        # Full-fidelity chat history (for the "previous chats" UI list —
        # separate concern from stm/ltm, see memory/chat_history.py).
        # One session per SupervisorAgent instance; call new_session() to
        # start a fresh one (e.g. user clicks "New chat" in the UI).
        self.history = ChatHistoryStore(db_path=chat_history_db_path)
        self.session_id = self.history.start_session()

        # Specialized agents — each queries the SAME Pinecone index, but
        # with a different metadata filter (see config.py).
        self._agents: Dict[str, SpecializedAgent] = {
            desc.agent_id: SpecializedAgent(
                description=desc,
                pinecone_index=pinecone_index,
                embed_model=embed_model,
                llm_client=self.llm_client,
                reranker=reranker,
            )
            for desc in AGENT_REGISTRY
        }

        # Pre-render agent registry for routing prompt
        self._registry = self._build_registry()

    # ── Registry description ─────────────────────────────────────────────────

    def _build_registry(self) -> str:
        lines = []
        for desc in AGENT_REGISTRY:
            lines.append(
                f"- agent_id: {desc.agent_id}\n"
                f"  covers  : {', '.join(desc.countries)} | "
                f"{', '.join(desc.legal_areas)} | "
                f"{', '.join(desc.content_types)}\n"
                f"  summary : {desc.description}"
            )
        return "\n\n".join(lines)

    # ── LTM context formatting ───────────────────────────────────────────────

    @staticmethod
    def _format_ltm_context(hits: List[Tuple[str, str, float, str]]) -> str:
        """
        Format long-term-memory hits (similar past Q&A) into a block to
        pass to specialized agents as supporting background. Empty string
        if there are no hits, so callers can append it unconditionally.
        """
        if not hits:
            return ""

        lines = [
            "Relevant past Q&A from long-term memory (background only — "
            "any article/case identifier mentioned below is NOT a valid "
            "citation; see the system instructions for how to handle this):"
        ]
        for q, a, _dist, country_scope in hits:
            safe_answer = _defang_bracketed_labels(a)
            jurisdiction = country_scope.replace("|", ", ")
            lines.append(
                f'- Jurisdiction: {jurisdiction}\n'
                f'  Previously asked: "{q}"\n'
                f'  Previous answer: {safe_answer}'
            )
        return "\n".join(lines)

    # ── Triage + Routing (single LLM call) ──────────────────────────────────

    def _fallback_route(self, query: str) -> List[str]:
        """Deterministic, bounded fallback when the router output is invalid.

        It narrows agents using explicit country, legal-area and source-type
        words in the question. With no usable signal it returns no agents
        instead of launching the entire registry.
        """
        text = query.casefold()
        country_terms = {
            "Italy": ("italy", "italia", "italian", "italiano", "italiana"),
            "Slovenia": ("slovenia", "slovenian", "sloveno", "slovena"),
            "Estonia": ("estonia", "estonian", "estone", "estoni"),
        }
        area_terms = {
            "Divorce": (
                "divorce", "divorz", "separation", "separazione", "marriage",
                "matrimonio", "spouse", "coniuge", "custody", "affidamento",
            ),
            "Inheritance": (
                "inherit", "succession", "eredit", "successione", "will",
                "testament", "heir", "erede", "legittima",
            ),
        }
        case_terms = (
            "case law", "giurisprud", "court", "tribunal", "cassazione",
            "sentenza", "decisione giudiziaria",
        )
        law_terms = (
            "article", "articolo", "statute", "legge", "codice", "provision",
            "norma", "legislation", "legislazione",
        )

        countries = {
            country
            for country, terms in country_terms.items()
            if any(term in text for term in terms)
        }
        areas = {
            area
            for area, terms in area_terms.items()
            if any(term in text for term in terms)
        }
        wants_cases = any(term in text for term in case_terms)
        wants_law = any(term in text for term in law_terms)

        if not countries and not areas:
            return []

        selected = []
        for desc in AGENT_REGISTRY:
            if countries and not countries.intersection(desc.countries):
                continue
            if areas and not areas.intersection(desc.legal_areas):
                continue
            if wants_cases != wants_law:
                wanted_type = "case" if wants_cases else "civil_code"
                if wanted_type not in desc.content_types:
                    continue
            selected.append(desc.agent_id)
        return selected

    def _triage_and_route(
        self, query: str, session_context: str
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Single LLM call that decides:
          - retrieval: true/false
          - direct_answer: if retrieval=false, the answer itself
          - selected_agents: if retrieval=true, which agents to activate

        Returns (needs_retrieval, direct_answer, agent_ids)
        """
        try:
            response = self.llm_client.chat.completions.create(
                model=_MODELS["main"],
                max_tokens=600,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the routing layer of a multi-agent legal RAG system.\n\n"
                            "When you receive a question:\n"
                            "1. If it is conversational or asks a stable, general fact that can "
                            "be answered reliably without consulting the corpus (for example, "
                            "when divorce was introduced in Italy), set retrieval=false and "
                            "provide a direct answer.\n"
                            "2. If it asks for current or detailed legal rules, exact articles, "
                            "case law, source-backed analysis, or a comparison between countries, "
                            "set retrieval=true and select the relevant agents.\n"
                            "3. When uncertain whether a legal claim requires sources, prefer "
                            "retrieval=true.\n\n"
                            "Available agents:\n"
                            f"{self._registry}\n\n"
                            "Recent conversation context:\n"
                            f"{session_context}\n\n"
                            "Respond ONLY with valid JSON, no markdown fences:\n"
                            "{\n"
                            '  "retrieval": true or false,\n'
                            '  "direct_answer": "..." (only if retrieval=false),\n'
                            '  "selected_agents": ["id1", "id2"] (only if retrieval=true),\n'
                            '  "reasoning": "..."\n'
                            "}"
                        ),
                    },
                    {"role": "user", "content": query},
                ],
            )

            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("router response must be a JSON object")

            needs_retrieval = data.get("retrieval")
            if type(needs_retrieval) is not bool:
                raise ValueError("'retrieval' must be a JSON boolean")

            reasoning = data.get("reasoning", "")
            if not isinstance(reasoning, str):
                reasoning = ""

            print(f"\n[Supervisor] Retrieval: {needs_retrieval}")
            print(f"[Supervisor] Reasoning: {reasoning}")

            if not needs_retrieval:
                direct_answer = data.get("direct_answer")
                if not isinstance(direct_answer, str) or not direct_answer.strip():
                    raise ValueError("direct route requires a non-empty 'direct_answer'")
                return False, direct_answer.strip(), []

            selected = data.get("selected_agents")
            if not isinstance(selected, list) or not all(
                isinstance(agent_id, str) for agent_id in selected
            ):
                raise ValueError("'selected_agents' must be a list of strings")

            valid_agents = list(dict.fromkeys(
                agent_id for agent_id in selected if agent_id in self._agents
            ))
            if not valid_agents:
                raise ValueError("router selected no valid agents")
            return True, None, valid_agents

        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, IndexError) as exc:
            fallback_agents = self._fallback_route(query)
            logger.warning(
                "Router response invalid (%s); deterministic fallback selected: %s",
                exc,
                fallback_agents,
            )
            return True, None, fallback_agents
        except Exception as exc:
            fallback_agents = self._fallback_route(query)
            logger.warning(
                "Router request failed (%s); deterministic fallback selected: %s",
                exc,
                fallback_agents,
            )
            return True, None, fallback_agents

    # ── Aggregation ──────────────────────────────────────────────────────────

    def _aggregate(self, query: str, partials: List[PartialAnswer]) -> str:
        """Synthesise multiple partial answers into one coherent reply."""
        if len(partials) == 1:
            return partials[0].answer

        sections = "\n\n".join(
            f"### {p.agent_name}\n{p.answer}"
            for p in sorted(partials, key=lambda x: x.relevance_score, reverse=True)
        )

        # More than one partial does NOT necessarily mean more than one
        # country — the common case is one country split across two
        # agents (case law + legislation, e.g. slovenia_divorce_cases +
        # slovenia_divorce_law). Telling the model to act as a
        # "comparative law expert" and "highlight differences between
        # jurisdictions" in that situation invites it to manufacture a
        # comparison against a country nobody asked about and no agent
        # retrieved anything for (observed: a Slovenia-only answer citing
        # an Italian case that was never in context). Only use the
        # comparative framing when the partials genuinely span more than
        # one country.
        countries_involved = set()
        for p in partials:
            agent = self._agents.get(p.agent_id)
            if agent:
                countries_involved.update(agent.description.countries)
        is_comparative = len(countries_involved) > 1

        if is_comparative:
            system_content = (
                "You are a senior comparative law expert.\n"
                "Synthesise the partial answers below into one clear, "
                "well-structured response.\n"
                "Highlight similarities and differences between jurisdictions.\n"
                "Preserve all document references exactly as they appear in "
                "the partial answers (e.g. [Italy | Divorce | Art. 162]) — "
                "do not shorten, renumber, merge, or "
                "invent new labels.\n"
                "Do not reveal the multi-agent architecture."
            )
        else:
            system_content = (
                "You are a senior legal expert.\n"
                "The partial answers below all concern the SAME single "
                "jurisdiction (case law and legislation, retrieved and "
                "answered separately) — synthesise them into one clear, "
                "coherent response about that one jurisdiction only.\n"
                "Do not compare against, or mention, any other country's "
                "law — none was retrieved for this question.\n"
                "Preserve all document references exactly as they appear in "
                "the partial answers (e.g. [Italy | Divorce | Art. 162]) — "
                "do not shorten, renumber, merge, or "
                "invent new labels.\n"
                "Do not reveal the multi-agent architecture."
            )

        # Synthesize partial answers while preserving exact citation labels.
        response = self.llm_client.chat.completions.create(
            model=_MODELS["main"],
            max_tokens=2048,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nPartial answers:\n{sections}",
                },
            ],
        )
        return response.choices[0].message.content

    # ── Background LTM store ────────────────────────────────────────────────

    def _store_in_ltm_background(
        self,
        query: str,
        answer: str,
        agents_used: List[str],
        countries_used: List[str],
    ) -> None:
        """
        Runs LongTermMemory.store() (which makes an LLM call to produce the
        summary) in a daemon thread, so the user gets their answer
        immediately instead of waiting for this second, non-critical LLM call.

        Daemon thread: if the process exits before this finishes, it's
        simply dropped — acceptable here since it's best-effort background
        indexing, not something the user is waiting on or that must be
        transactionally guaranteed.
        """
        def _run():
            try:
                self.ltm.store(
                    query=query,
                    answer=answer,
                    agents_used=agents_used,
                    countries_used=countries_used,
                )
            except Exception as e:
                print(f"  [WARN] Background LTM store failed: {e}")

        threading.Thread(target=_run, daemon=True).start()

    # ── Session management ───────────────────────────────────────────────────

    def new_session(self) -> str:
        """
        Start a fresh chat history session and reset short-term memory.
        Call this when the user clicks "New chat" in the UI. Long-term
        memory (ltm) is untouched — it's cross-session by design.
        """
        self.stm.reset()
        self.session_id = self.history.start_session()
        return self.session_id

    def load_session(self, session_id: str) -> List[Dict]:
        """
        Reload a past session: makes it the active session (further turns
        append to it) and restores short-term memory context from it, so
        follow-up questions after reload have the right context.

        Returns the list of turns (as dicts) for the UI to render.
        """
        turns_data = self.history.load_session(session_id)
        self.session_id = session_id

        stm_turns = [
            Turn(
                turn_id=t["turn_id"],
                query=t["query"],
                agents_activated=t["agents_activated"],
                answer=t["answer"],
            )
            for t in turns_data
        ]
        self.stm.load_history(stm_turns)

        return turns_data

    # ── Public entry point ───────────────────────────────────────────────────

    def ask(self, query: str) -> str:
        """
        Full pipeline:
        triage → (route → agents → aggregate) → guardrail → memory update
        """
        print(f"\n{'-'*60}")
        print(f"[Supervisor] Query: {query}")

        # Session context from short-term memory
        session_context = self.stm.as_context_string(n_turns=3)

        # NOTE: long-term memory recall happens per-agent, below, once we
        # know which agent(s) are handling this query — see the retrieval
        # loop. It's intentionally NOT computed here / injected into the
        # triage/routing prompt.

        # Triage + routing
        needs_retrieval, direct_answer, agent_ids = self._triage_and_route(
            query, session_context
        )

        # ── Direct answer path ───────────────────────────────────────────────
        if not needs_retrieval and direct_answer:
            print("[Supervisor] Answering directly (no retrieval needed).")
            direct_answer += _source_appendix([])
            turn = self.stm.add_turn(
                query=query,
                agents_activated=[],
                answer=direct_answer,
            )
            # Direct (non-retrieval) answers are intentionally NOT persisted
            # to long-term memory — LTM is reserved for retrieval-grounded
            # answers backed by actual sources. Chat history, however, is a
            # full transcript of the session regardless of path, so it IS
            # saved here.
            self.history.save_turn(
                session_id=self.session_id,
                turn_id=turn.turn_id,
                query=query,
                answer=direct_answer,
                agents_activated=[],
            )
            return direct_answer

        # ── Retrieval path ───────────────────────────────────────────────────
        if not agent_ids:
            answer = (
                "I could not identify a sufficiently specific jurisdiction or "
                "legal area for retrieval. Please mention the country and whether "
                "the question concerns divorce or inheritance."
            ) + _source_appendix([])
            turn = self.stm.add_turn(query, [], answer)
            self.history.save_turn(
                self.session_id, turn.turn_id, query, answer, []
            )
            return answer

        # Dispatch to selected agents, passing along relevant LTM background.
        # Recall is scoped to each agent individually (agent_id filter) so
        # a query answered by e.g. the Slovenia agent only ever sees past
        # Q&A that Slovenia (or another matching agent) actually answered —
        # not a topically-similar but unrelated answer from a different
        # country's agent (observed: an Italy Q&A about matrimonial
        # regimes, containing "[Art. 162]", was being recalled as
        # background for an unrelated Slovenia question and its citation
        # label got copied into the new answer).
        def _run_agent(aid: str) -> PartialAnswer:
            print(f"[{aid}] Generating partial answer ...")
            ltm_hits = self.ltm.recall_similar(query, n=2, agent_id=aid)
            if ltm_hits:
                print(
                    f"[Supervisor] Found {len(ltm_hits)} similar past Q&A "
                    f"in LTM for agent '{aid}'."
                )
            ltm_context = self._format_ltm_context(ltm_hits)
            return self._agents[aid].answer(query, ltm_context=ltm_context)

        partials_by_agent: Dict[str, PartialAnswer] = {}
        failed_agents: Dict[str, str] = {}
        no_document_agents: List[str] = []
        worker_count = min(MAX_PARALLEL_AGENTS, len(agent_ids))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_agent = {
                executor.submit(_run_agent, aid): aid for aid in agent_ids
            }
            for future in as_completed(future_to_agent):
                aid = future_to_agent[future]
                try:
                    partial = future.result()
                    if partial.labeled_chunks:
                        partials_by_agent[aid] = partial
                    else:
                        no_document_agents.append(aid)
                except Exception as exc:
                    failed_agents[aid] = str(exc)
                    logger.exception("Agent %s failed", aid)

        # Restore router order after futures complete so equal reranker scores
        # and persisted agent lists remain deterministic.
        partials = [
            partials_by_agent[aid]
            for aid in agent_ids
            if aid in partials_by_agent
        ]

        if not partials:
            if failed_agents:
                answer = (
                    "The relevant sources could not be consulted because of a "
                    "temporary retrieval or generation error. Please try again."
                )
            else:
                answer = "No relevant, unambiguous documents were found for this question."
            answer += _source_appendix([])
            turn = self.stm.add_turn(query, [], answer)
            self.history.save_turn(
                self.session_id, turn.turn_id, query, answer, []
            )
            return answer

        # Aggregate
        try:
            final_answer = self._aggregate(query, partials)
        except Exception as exc:
            logger.exception("Aggregation failed; returning successful partial answers")
            final_answer = "\n\n".join(partial.answer for partial in partials)

        # Build a label -> source-text map across every agent's retrieved
        # chunks, so the output guardrail can look up the exact text behind
        # any "[label]" citation the model used in final_answer, instead of
        # only comparing the whole answer against an unlabelled bag of
        # chunks.
        #
        # A single source document is often split into multiple chunks
        # (see chunk_text in Ingestion.ipynb) that all carry the SAME label —
        # e.g. a long civil-code article retrieved as two separate top-k
        # hits. A plain dict comprehension would let the later chunk
        # silently overwrite the earlier one, leaving only a fragment of
        # the real document behind the label. That fragment is then all
        # the citation checker ever sees — so a claim genuinely supported
        # by the FULL article can get flagged as unsupported just because
        # it happened to land in the chunk that got overwritten (observed:
        # Art. 105's "one-half" rule and Art. 106's "gifts within three
        # years" rule were both in the first of two chunks, but only the
        # second chunk survived into chunk_by_label). Concatenating chunks
        # that share a label, instead of overwriting, keeps the full
        # document available for verification.
        chunk_by_label: Dict[str, str] = {}
        label_source: Dict[str, str] = {}
        # label -> country, built alongside chunk_by_label so the guardrail
        # can scope its cross-reference heuristic by jurisdiction (see
        # _is_cross_referenced in guardrails/output_guard.py) instead of
        # matching an article number against ANY retrieved country's text.
        label_country: Dict[str, str] = {}
        # Labels seen with two DIFFERENT countries in the same turn (e.g.
        # the same article number happens to exist in two countries'
        # corpora and both got retrieved in one comparative query) — for
        # these, no single country is reliable, so they're excluded from
        # label_country entirely rather than silently keeping whichever
        # country happened to be processed first.
        _ambiguous_country_labels: set = set()
        for p in partials:
            for label, text, country, source in p.labeled_chunks:
                if label in _ambiguous_country_labels:
                    continue

                previous_source = label_source.get(label)
                if previous_source is not None and previous_source != source:
                    # Never concatenate different documents behind one label:
                    # doing so could make an incorrectly-attributed citation pass.
                    chunk_by_label.pop(label, None)
                    label_country.pop(label, None)
                    _ambiguous_country_labels.add(label)
                    logger.warning(
                        "Citation label %r is shared by different sources (%s, %s); "
                        "excluding it from grounding.",
                        label,
                        previous_source,
                        source,
                    )
                    continue

                label_source[label] = source
                if label not in chunk_by_label:
                    chunk_by_label[label] = text
                elif text not in chunk_by_label[label]:
                    chunk_by_label[label] += "\n" + text
                if not country or label in _ambiguous_country_labels:
                    continue
                if label not in label_country:
                    label_country[label] = country
                elif label_country[label] != country:
                    label_country.pop(label, None)
                    _ambiguous_country_labels.add(label)

        # Output guardrail — verifies each citation individually against
        # its own source, not the answer as a whole against everything.
        try:
            final_answer = check_grounding(
                final_answer,
                chunk_by_label,
                self.llm_client,
                label_country=label_country,
            )
        except Exception:
            logger.exception("Output guardrail failed unexpectedly")
            final_answer = (
                "[NOTE: citation verification could not be completed due to a "
                "temporary error.]\n\n" + final_answer
            )

        if failed_agents or no_document_agents:
            final_answer = (
                "[NOTE: some relevant sources could not be consulted; the answer "
                "uses only the sources retrieved successfully.]\n\n" + final_answer
            )

        # Update memory
        successful_agent_ids = [partial.agent_id for partial in partials]
        retrieved_documents = _deduplicate_documents([
            document
            for partial in partials
            for document in partial.retrieved_documents
        ])
        display_answer = final_answer + _source_appendix(retrieved_documents)
        turn = self.stm.add_turn(
            query=query,
            agents_activated=successful_agent_ids,
            answer=display_answer,
        )
        self.history.save_turn(
            session_id=self.session_id,
            turn_id=turn.turn_id,
            query=query,
            answer=display_answer,
            agents_activated=successful_agent_ids,
            retrieved_documents=retrieved_documents,
        )
        # Runs in the background — the user gets final_answer immediately,
        # the summary LLM call for LTM happens after the fact.
        countries_used = sorted({
            country
            for aid in successful_agent_ids
            for country in self._agents[aid].description.countries
        })
        self._store_in_ltm_background(
            query=query,
            answer=final_answer,
            agents_used=successful_agent_ids,
            countries_used=countries_used,
        )

        return display_answer
