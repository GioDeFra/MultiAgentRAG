"""
memory/long_term.py — Long-term (cross-session) persistent memory.

Stores concise summaries of retrieval-grounded Q&A in ChromaDB (on disk,
survives restarts).

Past Q&A pairs are embedded and indexed so the supervisor can find
semantically similar questions that were already answered. If a good match
is found, the past summary is injected as supporting background to improve
continuity and consistency. It does NOT replace the normal Pinecone
retrieval: current retrieved documents remain the authoritative sources.

The stored "answer" is an LLM-written summary (concise, preserving
figures/article numbers) rather than the full response. If summarization
fails, a word-boundary truncation is used as a safe fallback.
"""

import json
import logging
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

QA_COLLECTION = "ltm_qa_pairs"
LEGACY_FACT_COLLECTION = "ltm_facts"
DEFAULT_MAX_QA_PAIRS = 150

# Hard ceiling only used as a fallback if LLM summarization fails.
# Never used to cut a sentence in half — see _truncate().
ANSWER_FALLBACK_MAX_CHARS = 2000


# ---------------------------------------------------------------------------
# LONG-TERM MEMORY
# ---------------------------------------------------------------------------

class LongTermMemory:
    """
    Persistent semantic memory backed by one ChromaDB collection.

    Parameters
    ----------
    db_dir : str
        Path to the ChromaDB folder. Separate concern from the RAG
        document corpus, which lives in Pinecone (see agents.py) — this
        is only for cross-session Q&A summary memory.
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
    persists to LTM afterwards) while recall_similar() may be in flight on
    the main thread, all collection access is serialized behind one lock.
    """

    def __init__(
        self,
        db_dir: str = "./chroma_db",
        embedding_model: str = "all-mpnet-base-v2",
        qa_similarity_threshold: float = 0.25,
        dedup_similarity_threshold: float = 0.05,
        max_qa_pairs: int = DEFAULT_MAX_QA_PAIRS,
    ) -> None:
        if max_qa_pairs < 1:
            raise ValueError("max_qa_pairs must be at least 1")

        self.qa_threshold = qa_similarity_threshold
        self.dedup_threshold = dedup_similarity_threshold
        self.max_qa_pairs = max_qa_pairs
        self._llm = get_llm_client()
        self._light_model = model_names()["light"]

        # Guards all reads/writes to _qa_col. The slow LLM summary call happens
        # OUTSIDE this lock because it does not touch the database.
        self._db_lock = threading.Lock()

        client = chromadb.PersistentClient(path=db_dir)

        # One-time cleanup of the facts collection used by older versions.
        # Facts were written but never consumed by the answer pipeline.
        collection_names = {
            c.name if hasattr(c, "name") else str(c)
            for c in client.list_collections()
        }
        if LEGACY_FACT_COLLECTION in collection_names:
            client.delete_collection(name=LEGACY_FACT_COLLECTION)
            logger.info("Deleted unused legacy collection %s", LEGACY_FACT_COLLECTION)

        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        self._qa_col = client.get_or_create_collection(
            name=QA_COLLECTION,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Store a Q&A pair ─────────────────────────────────────────────────────

    def store(
        self,
        query: str,
        answer: str,
        agents_used: List[str],
        countries_used: Optional[List[str]] = None,
    ) -> None:
        """
        Save a Q&A pair using an LLM summary of the answer.
        Called at the end of every successful supervisor.ask() call.

        Dedup behaviour: if a near-identical question already exists
        (distance <= dedup_similarity_threshold), that record is UPDATED
        in place (same qa_id, new answer) instead of creating a new
        one — avoids the QA collection filling up with many entries for
        essentially the same question asked with different wording. Duplicate
        detection is scoped to the same agent set, so equal questions handled
        by different jurisdictions cannot overwrite one another.
        """
        clean_answer = self._strip_guardrail_notice(answer)
        countries_used = sorted(set(countries_used or []))
        country_scope = "|".join(countries_used) or "not specified"

        # Summarization is a slow network call — do it BEFORE taking the
        # lock so other threads aren't blocked waiting on the LLM API.
        summary = self._summarize(clean_answer, countries_used)
        stored_answer = summary if summary is not None else self._truncate(
            clean_answer, ANSWER_FALLBACK_MAX_CHARS
        )
        agent_scope = self._agent_scope(agents_used)

        with self._db_lock:
            existing_qa_id = self._find_duplicate_locked(query, agent_scope)
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
                    "agent_scope": agent_scope,
                    "countries_used": json.dumps(countries_used),
                    "country_scope": country_scope,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **agent_flags,
                }],
            )

            if existing_qa_id:
                logger.info("Updated existing QA record qa_id=%s (near-duplicate question)", qa_id)
            self._enforce_retention_locked(protected_qa_id=qa_id)

    @staticmethod
    def _agent_scope(agents_used: List[str]) -> str:
        """Stable metadata key representing the exact set of agents used."""
        return "|".join(sorted(set(agents_used)))

    def _find_duplicate_locked(
        self, query: str, agent_scope: str
    ) -> Optional[str]:
        """
        Return the qa_id of an existing near-identical question, or None.
        Caller must hold self._db_lock.
        """
        if self._qa_col.count() == 0:
            return None

        results = self._qa_col.query(
            query_texts=[query],
            n_results=1,
            where={"agent_scope": agent_scope},
            include=["distances"],
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        if ids and distances and distances[0] <= self.dedup_threshold:
            return ids[0]
        return None

    def _enforce_retention_locked(self, protected_qa_id: str) -> None:
        """Delete oldest Q&A one at a time until the configured cap is met."""
        while self._qa_col.count() > self.max_qa_pairs:
            records = self._qa_col.get(include=["metadatas"])
            candidates = [
                (qa_id, metadata.get("timestamp", ""))
                for qa_id, metadata in zip(records["ids"], records["metadatas"])
                if qa_id != protected_qa_id
            ]
            if not candidates:
                return

            oldest_qa_id, _ = min(candidates, key=lambda item: item[1])
            self._qa_col.delete(ids=[oldest_qa_id])
            logger.info("Deleted oldest LTM Q&A record qa_id=%s", oldest_qa_id)

    # ── Recall similar past Q&A ──────────────────────────────────────────────

    def recall_similar(
        self, query: str, n: int = 2, agent_id: Optional[str] = None
    ) -> List[Tuple[str, str, float, str]]:
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
        List of (past_question, past_answer, distance, country_scope) tuples,
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
                hits.append((
                    doc,
                    meta["answer"],
                    dist,
                    meta.get("country_scope", "not specified"),
                ))

        return hits

    # ── Summary generation ──────────────────────────────────────────────────

    def _summarize(
        self, answer: str, countries_used: List[str]
    ) -> Optional[str]:
        """
        Ask the LLM for a concise summary of the complete answer, preserving
        figures, fractions, article numbers, and named laws. Returns None on
        failure so the caller can fall back to safe truncation.
        """
        jurisdictions = ", ".join(countries_used) or "not specified"
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
                        "Summarize the following legal answer in at most about "
                        "80 words. Preserve "
                        "every specific figure, fraction, article number, or named "
                        "law needed to understand the result. Return only the "
                        "summary, with no JSON, heading, or preamble. Make the "
                        "applicable jurisdiction explicit in the summary.\n\n"
                        f"Jurisdiction(s): {jurisdictions}\n\n"
                        f"Text:\n{answer}"
                    ),
                }],
            )
            summary = response.choices[0].message.content.strip()
            return summary or None

        except Exception as e:
            logger.warning(
                "Summary generation failed, falling back to truncation: %s",
                e,
            )
            return None

    @staticmethod
    def _strip_guardrail_notice(answer: str) -> str:
        """Remove a leading technical guardrail notice before memorization."""
        guardrail_prefixes = (
            "[NOTE: the guardrail",
            "[NOTE: the grounding check",
            "[WARNING: possible citation issue",
            "[WARNING: the guardrail",
            "[WARNING: no source documents",
        )
        if answer.startswith(guardrail_prefixes):
            _notice, separator, clean_answer = answer.partition("\n\n")
            if separator and clean_answer.strip():
                return clean_answer.strip()
        return answer

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
            return {"qa_pairs": self._qa_col.count()}

    def __repr__(self) -> str:
        s = self.stats()
        return f"LongTermMemory(qa={s['qa_pairs']}/{self.max_qa_pairs})"
