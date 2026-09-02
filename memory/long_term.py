"""
memory/long_term.py — Long-term (cross-session) persistent memory.

Stores two things in ChromaDB (on disk, survives restarts):

1. Past Q&A pairs
   Embedded and indexed so the supervisor can find semantically
   similar questions that were already answered. If a good match
   is found, the past answer is injected as extra context —
   avoiding redundant retrieval and improving consistency.

   The stored "answer" is an LLM-written summary (concise, preserves
   figures/article numbers) rather than a raw mid-sentence truncation.
   If summarization fails for any reason, a word-boundary truncation
   is used as a safe fallback — never a mid-word cut.

2. Extracted facts
   Short atomic statements extracted from each answer by the LLM
   (e.g. "In Italy the forced heirship share for one child is 1/2
   of the estate"). Retrieved as background knowledge to enrich
   future answers on related topics.

Summary and facts are produced in a SINGLE LLM call (not two), to
avoid doubling latency/cost per store().
"""

import json
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from llm_client import get_llm_client, model_names, extra_kwargs


# ---------------------------------------------------------------------------
# LOAD API KEY
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent.parent / "Apikey.env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COLLECTION NAMES
# ---------------------------------------------------------------------------

QA_COLLECTION   = "ltm_qa_pairs"
FACT_COLLECTION = "ltm_facts"

# Hard ceiling only used as a fallback if LLM summarization fails.
# Never used to cut a sentence in half — see _truncate().
ANSWER_FALLBACK_MAX_CHARS = 2000


# ---------------------------------------------------------------------------
# LONG-TERM MEMORY
# ---------------------------------------------------------------------------

class LongTermMemory:
    """
    Persistent semantic memory backed by two ChromaDB collections.

    Parameters
    ----------
    db_dir : str
        Path to the ChromaDB folder. Separate concern from the RAG
        document corpus, which lives in Pinecone (see agents.py) — this
        is only for cross-session Q&A/fact memory.
    embedding_model : str
        SentenceTransformer model name. Deliberately independent from the
        RAG corpus's embedding model (currently BGE-M3, see agents.py
        EMBEDDING_MODEL): unlike Pinecone, where query and document
        vectors must share the same dimensionality, this ChromaDB
        instance only ever compares vectors against other vectors it
        wrote itself — there's no cross-store compatibility requirement.
        Keeping it on all-mpnet-base-v2 avoids re-embedding existing
        history and re-tuning the thresholds below every time the main
        RAG pipeline's embedding model changes.
    qa_similarity_threshold : float
        Cosine distance below which a past Q&A is considered a match.
        Lower = stricter. 0.25 works well for legal questions.
    fact_similarity_threshold : float
        Cosine distance below which a fact is considered relevant.
        Looser than qa_similarity_threshold since facts are atomic
        and meant to be recalled from many different angles.
    dedup_similarity_threshold : float
        Cosine distance below which an incoming question is treated as
        "the same question" as one already stored, and updated in place
        instead of creating a new record. Much stricter than
        qa_similarity_threshold (which is for *recall*, not identity).

    Thread-safety
    -------------
    ChromaDB's PersistentClient is not guaranteed safe for concurrent
    reads/writes across threads. Since store() can be called from a
    background thread (e.g. supervisor answers the user immediately and
    persists to LTM afterwards) while another call to recall_similar()/
    recall_facts() may be in flight on the main thread, all collection
    access is serialized behind a single lock.
    """

    def __init__(
        self,
        db_dir: str = "./chroma_db",
        embedding_model: str = "all-mpnet-base-v2",
        qa_similarity_threshold: float = 0.25,
        fact_similarity_threshold: float = 0.40,
        dedup_similarity_threshold: float = 0.05,
    ) -> None:
        self.qa_threshold = qa_similarity_threshold
        self.fact_threshold = fact_similarity_threshold
        self.dedup_threshold = dedup_similarity_threshold
        self._llm = get_llm_client()
        self._light_model = model_names()["light"]

        # Guards all reads/writes to _qa_col and _fact_col. The LLM calls
        # (summary/facts extraction) happen OUTSIDE this lock — they're slow
        # network calls that don't touch the DB, so there's no reason to
        # block other readers/writers while waiting on them.
        self._db_lock = threading.Lock()

        client = chromadb.PersistentClient(path=db_dir)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        self._qa_col = client.get_or_create_collection(
            name=QA_COLLECTION,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._fact_col = client.get_or_create_collection(
            name=FACT_COLLECTION,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Store a Q&A pair ─────────────────────────────────────────────────────

    def store(
        self,
        query: str,
        answer: str,
        agents_used: List[str],
    ) -> None:
        """
        Save a Q&A pair (as an LLM summary) and its extracted atomic facts.
        Called at the end of every successful supervisor.ask() call.

        Dedup behaviour: if a near-identical question already exists
        (distance <= dedup_similarity_threshold), that record is UPDATED
        in place (same qa_id, new answer/facts) instead of creating a new
        one — avoids the QA collection filling up with many entries for
        essentially the same question asked with different wording.
        """
        # Summarization is a slow network call — do it BEFORE taking the
        # lock so other threads aren't blocked waiting on the LLM API.
        summary, facts = self._summarize_and_extract(answer)
        stored_answer = summary if summary is not None else self._truncate(
            answer, ANSWER_FALLBACK_MAX_CHARS
        )

        with self._db_lock:
            existing_qa_id = self._find_duplicate_locked(query)
            qa_id = existing_qa_id or uuid.uuid4().hex

            # ChromaDB's `where` filter only matches flat scalar metadata
            # fields (no "list contains" queries), so alongside the
            # human-readable "agents_used" JSON string, one boolean flag
            # per agent is also stored (e.g. "agent_italy_family": True) —
            # that's what recall_similar(agent_id=...) actually filters on,
            # so a query answered partly by agent X can be recalled by
            # agent X later, without pulling in every other agent's answers
            # too (see recall_similar docstring for why this matters).
            agent_flags = {f"agent_{aid}": True for aid in agents_used}

            self._qa_col.upsert(
                ids=[qa_id],
                documents=[query],
                metadatas=[{
                    "answer": stored_answer,
                    "answer_is_summary": summary is not None,
                    "agents_used": json.dumps(agents_used),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **agent_flags,
                }],
            )

            if existing_qa_id:
                logger.info("Updated existing QA record qa_id=%s (near-duplicate question)", qa_id)
                # Drop the old facts tied to this qa_id — they belonged to
                # the previous answer and would otherwise linger stale
                # alongside the new ones.
                self._delete_facts_for_qa_locked(qa_id)

            if facts:
                self._store_facts_locked(facts, qa_id)

    def _find_duplicate_locked(self, query: str) -> Optional[str]:
        """
        Return the qa_id of an existing near-identical question, or None.
        Caller must hold self._db_lock.
        """
        if self._qa_col.count() == 0:
            return None

        results = self._qa_col.query(
            query_texts=[query],
            n_results=1,
            include=["distances"],
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        if ids and distances and distances[0] <= self.dedup_threshold:
            return ids[0]
        return None

    def _delete_facts_for_qa_locked(self, qa_id: str) -> None:
        """Delete all facts previously linked to this qa_id. Caller must hold self._db_lock."""
        try:
            existing = self._fact_col.get(where={"qa_id": qa_id}, include=[])
            stale_ids = existing.get("ids", [])
            if stale_ids:
                self._fact_col.delete(ids=stale_ids)
        except Exception as e:
            logger.warning("Failed to clear stale facts for qa_id=%s: %s", qa_id, e)

    # ── Recall similar past Q&A ──────────────────────────────────────────────

    def recall_similar(
        self, query: str, n: int = 2, agent_id: Optional[str] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Find past Q&A pairs semantically similar to the current query.

        Parameters
        ----------
        agent_id : str, optional
            If given, only recall past Q&A that this same agent previously
            helped answer (see the `agent_{id}` flags written in store()).
            Without this, semantic similarity alone can match a
            topically-related but jurisdictionally-unrelated past answer
            (e.g. an Italy matrimonial-regime answer surfacing as
            "background" for an unrelated Slovenia question) — and its
            citation labels can leak into the new answer even though no
            document with that label was actually retrieved this turn.
            Passing the requesting agent's id keeps recall scoped to
            Q&A that agent (or another agent covering the same
            country/area) actually contributed to.

        Returns
        -------
        List of (past_question, past_answer, distance) tuples,
        only those within qa_similarity_threshold.
        Empty list if nothing relevant found.
        """
        where = {f"agent_{agent_id}": True} if agent_id else None

        with self._db_lock:
            if self._qa_col.count() == 0:
                return []

            query_kwargs = dict(
                query_texts=[query],
                n_results=min(n, self._qa_col.count()),
                include=["documents", "metadatas", "distances"],
            )
            if where:
                query_kwargs["where"] = where

            results = self._qa_col.query(**query_kwargs)

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist <= self.qa_threshold:
                hits.append((doc, meta["answer"], dist))

        return hits

    # ── Recall relevant facts ────────────────────────────────────────────────

    def recall_facts(self, query: str, n: int = 4) -> List[str]:
        """
        Return the most relevant stored facts for the current query.
        Uses a slightly looser threshold than Q&A recall.
        """
        with self._db_lock:
            if self._fact_col.count() == 0:
                return []

            results = self._fact_col.query(
                query_texts=[query],
                n_results=min(n, self._fact_col.count()),
                include=["documents", "distances"],
            )

        return [
            doc
            for doc, dist in zip(
                results["documents"][0],
                results["distances"][0],
            )
            if dist <= self.fact_threshold
        ]

    # ── Summary + fact extraction (single LLM call) ─────────────────────────

    def _summarize_and_extract(
        self, answer: str
    ) -> Tuple[Optional[str], List[str]]:
        """
        Ask the LLM, in one call, to:
          1. Write a concise summary of the answer (preserving figures,
             fractions, article numbers, named laws).
          2. Extract 2-5 atomic, context-free facts.

        Returns (summary, facts). On any failure returns (None, []) so
        callers can fall back to truncation without crashing.
        """
        try:
            # thinking=False: cheap high-volume summarization, not worth the
            # extra latency of a reasoning pass.
            response = self._llm.chat.completions.create(
                model=self._light_model,
                max_tokens=500,
                temperature=0.2,
                **extra_kwargs(thinking=False),
                messages=[{
                    "role": "user",
                    "content": (
                        "You will receive a legal answer. Do two things:\n"
                        "1. Write a concise summary (max ~80 words) that preserves "
                        "every specific figure, fraction, article number, or named "
                        "law mentioned in the text.\n"
                        "2. Extract 2 to 5 short, self-contained atomic facts. Each "
                        "fact must be one sentence understandable with no other "
                        "context.\n\n"
                        "Return ONLY a JSON object, no other text, no markdown "
                        "fences:\n"
                        '{"summary": "...", "facts": ["fact 1", "fact 2"]}\n\n'
                        f"Text:\n{answer[:3000]}"
                    ),
                }],
            )

            raw = self._strip_fences(response.choices[0].message.content)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("no JSON object found in model output")

            data = json.loads(match.group(0))

            summary = data.get("summary")
            summary = (
                summary.strip()
                if isinstance(summary, str) and summary.strip()
                else None
            )

            raw_facts = data.get("facts", [])
            facts = [
                f.strip() for f in raw_facts if isinstance(f, str) and f.strip()
            ]

            return summary, facts

        except Exception as e:
            logger.warning(
                "Summary/fact extraction failed, falling back to truncation: %s",
                e,
            )
            return None, []

    @staticmethod
    def _strip_fences(raw: str) -> str:
        """Remove markdown code fences if the model added them."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        return raw.strip()

    def _store_facts_locked(self, facts: List[str], qa_id: str) -> None:
        """Caller must hold self._db_lock."""
        self._fact_col.upsert(
            ids=[uuid.uuid4().hex for _ in facts],
            documents=facts,
            metadatas=[{
                "qa_id": qa_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            } for _ in facts],
        )
        logger.info("Stored %d facts for qa_id=%s", len(facts), qa_id)

    # ── Fallback truncation (word-boundary, never mid-word) ─────────────────

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        cut = text[:limit].rsplit(" ", 1)[0]
        return cut + " ..."

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        with self._db_lock:
            return {
                "qa_pairs": self._qa_col.count(),
                "facts":    self._fact_col.count(),
            }

    def __repr__(self) -> str:
        s = self.stats()
        return f"LongTermMemory(qa={s['qa_pairs']}, facts={s['facts']})"