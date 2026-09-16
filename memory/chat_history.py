"""
memory/chat_history.py — Full-fidelity chat session history.

Powers a "previous chats" sidebar: a chronological list of past sessions, each with its full transcript of turns. The user can click a session to reload it verbatim.

This is deliberately separate from:
  - short_term.py  (in-RAM only, current session's sliding window)
  - long_term.py   (semantic recall of similar Q&A,
                     lossy by design — summaries, not full transcripts)

Chat history stores the FULL, UNMODIFIED text of every turn, indexed by
session, so the user can reopen and re-read a past conversation exactly
as it happened. No embeddings, no similarity search, no summarization —
just persistent, chronological storage.

Backed by SQLite (stdlib only, no extra dependency) since access here is
"list sessions" / "load this session by id", not semantic search.
"""

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# SCHEMA
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    session_id        TEXT NOT NULL,
    turn_id           INTEGER NOT NULL,
    query             TEXT NOT NULL,
    answer            TEXT NOT NULL,
    agents_activated  TEXT NOT NULL,   -- JSON list
    retrieved_documents TEXT NOT NULL DEFAULT '[]', -- JSON list
    timestamp         TEXT NOT NULL,
    PRIMARY KEY (session_id, turn_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
"""

TITLE_MAX_CHARS = 60
DEFAULT_MAX_SESSIONS = 10
DEFAULT_MAX_DB_SIZE_MB = 30
SOURCE_APPENDIX_MARKER = "\n\n---\n\n### Sources used"


def _answer_without_source_appendix(answer: str) -> str:
    """Exclude the chat-only source display from the compact JSON export."""
    return answer.split(SOURCE_APPENDIX_MARKER, 1)[0]


# ---------------------------------------------------------------------------
# CHAT HISTORY STORE
# ---------------------------------------------------------------------------


class ChatHistoryStore:
    """
    Persistent, chronological store of full chat sessions.

    Parameters
    ----------
    db_path : str
        Path to the SQLite file (separate from the ChromaDB directory
        used by long_term.py — this is a different kind of data).
    """

    def __init__(
        self,
        db_path: str = "./chat_history.db",
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_db_size_mb: int = DEFAULT_MAX_DB_SIZE_MB,
    ) -> None:
        self._path = db_path
        self._json_path = str(Path(db_path).with_suffix(".json"))
        self.max_sessions = max_sessions
        self.max_db_size_bytes = max_db_size_mb * 1024 * 1024
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── Session lifecycle ────────────────────────────────────────────────────

    def start_session(self) -> str:
        """Create a new empty session and return its id."""
        session_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, "New chat", now, now),
            )
        return session_id

    def save_turn(
        self,
        session_id: str,
        turn_id: int,
        query: str,
        answer: str,
        agents_activated: List[str],
        retrieved_documents: Optional[List[Dict]] = None,
    ) -> None:
        """
        Append one turn to a session. Call this after every supervisor.ask(),
        alongside stm.add_turn() — same data, different destination.

        The session title is set from the first turn's question (truncated).
        """
        now = datetime.now(timezone.utc).isoformat()
        retrieved_documents = retrieved_documents or []
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO turns "
                "(session_id, turn_id, query, answer, agents_activated, "
                "retrieved_documents, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    turn_id,
                    query,
                    answer,
                    json.dumps(agents_activated, ensure_ascii=False),
                    json.dumps(retrieved_documents, ensure_ascii=False),
                    now,
                ),
            )

            row = conn.execute(
                "SELECT title FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            is_first_turn = turn_id == 0
            if row and is_first_turn:
                title = query.strip().replace("\n", " ")
                if len(title) > TITLE_MAX_CHARS:
                    title = title[:TITLE_MAX_CHARS].rsplit(" ", 1)[0] + "..."
                conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (title, now, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )

        # Keep the active session, but progressively remove the oldest chats
        # when either retention limit is exceeded. This runs after saving a
        # real turn, rather than after start_session(), so repeatedly opening
        # the application cannot evict useful chats by creating empty ones.
        self._enforce_retention(protected_session_id=session_id)

        self._export_json()

    @staticmethod
    def _export_turn(turn) -> dict:
        """Format one history turn without source excerpts."""
        documents = json.loads(turn["retrieved_documents"])
        return {
            "turn_id": turn["turn_id"],
            "timestamp": turn["timestamp"],
            "user_question": turn["query"],
            "system_answer": _answer_without_source_appendix(turn["answer"]),
            "agents_activated": json.loads(turn["agents_activated"]),
            "retrieved_documents": [
                {key: value for key, value in document.items() if key != "excerpt"}
                for document in documents
            ],
        }

    def _export_json(self) -> None:
        """Write a complete JSON history while excluding source excerpts."""
        try:
            with self._lock, self._connect() as conn:
                session_rows = conn.execute(
                    "SELECT session_id, title, created_at, updated_at "
                    "FROM sessions ORDER BY updated_at DESC"
                ).fetchall()
                sessions = []
                for session in session_rows:
                    turn_rows = conn.execute(
                        "SELECT turn_id, query, answer, agents_activated, "
                        "retrieved_documents, timestamp FROM turns "
                        "WHERE session_id = ? ORDER BY turn_id ASC",
                        (session["session_id"],),
                    ).fetchall()
                    turns = []
                    for turn in turn_rows:
                        turns.append(self._export_turn(turn))
                    sessions.append(
                        {
                            "session_id": session["session_id"],
                            "title": session["title"],
                            "created_at": session["created_at"],
                            "updated_at": session["updated_at"],
                            "turns": turns,
                        }
                    )

            destination = Path(self._json_path)
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {"sessions": sessions},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary.replace(destination)
        except (OSError, sqlite3.Error, json.JSONDecodeError):
            # SQLite remains authoritative if the readable export cannot be
            # refreshed because of a transient filesystem or data problem.
            pass

    def _enforce_retention(self, protected_session_id: str) -> int:
        """Apply count and on-disk size limits, oldest session first.

        The protected session is never deleted. SQLite keeps freed pages in
        its file, so ``VACUUM`` is required after deletions for the 30 MB
        limit to reflect the real disk usage.

        Returns the number of sessions deleted.
        """
        deleted = 0
        with self._lock:
            # First enforce the cheap, deterministic session-count limit.
            while self._session_count_unlocked() > self.max_sessions:
                if not self._delete_oldest_unlocked(protected_session_id):
                    break
                deleted += 1

            if deleted:
                self._vacuum_unlocked()

            # Then remove one old session at a time until the physical SQLite
            # file is back under the configured disk threshold.
            while self._db_size_unlocked() > self.max_db_size_bytes:
                if not self._delete_oldest_unlocked(protected_session_id):
                    break
                deleted += 1
                self._vacuum_unlocked()

        return deleted

    def _session_count_unlocked(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    def _delete_oldest_unlocked(self, protected_session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id != ? "
                "ORDER BY updated_at ASC LIMIT 1",
                (protected_session_id,),
            ).fetchone()

            if row is None:
                return False

            session_id = row["session_id"]
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        return True

    def _vacuum_unlocked(self) -> None:
        # VACUUM cannot run inside the transaction managed by _connect().
        conn = sqlite3.connect(self._path)
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()

    def _db_size_unlocked(self) -> int:
        path = Path(self._path)
        return path.stat().st_size if path.exists() else 0

    # ── Listing / loading (for the UI sidebar) ──────────────────────────────

    def list_sessions(self, limit: int = 50) -> List[Dict]:
        """
        Return sessions most-recently-updated first — exactly what a
        sidebar needs: session_id, title, timestamps, turn count.
        """
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.updated_at,
                       COUNT(t.turn_id) AS turn_count
                FROM sessions s
                LEFT JOIN turns t ON t.session_id = s.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def load_session(self, session_id: str) -> List[Dict]:
        """
        Return all turns for a session, oldest first — call this when the
        user clicks a session in the sidebar to reload it verbatim.
        """
        with self._lock, self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if exists is None:
                raise ValueError(f"Unknown chat session: {session_id}")
            rows = conn.execute(
                "SELECT turn_id, query, answer, agents_activated, "
                "retrieved_documents, timestamp "
                "FROM turns WHERE session_id = ? ORDER BY turn_id ASC",
                (session_id,),
            ).fetchall()
        return [
            {
                "turn_id": r["turn_id"],
                "query": r["query"],
                "answer": r["answer"],
                "agents_activated": json.loads(r["agents_activated"]),
                "retrieved_documents": json.loads(r["retrieved_documents"]),
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    def delete_session(self, session_id: str) -> None:
        """Permanently remove a session and its turns."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def rename_session(self, session_id: str, new_title: str) -> None:
        """Let the user manually rename a session."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (new_title.strip()[:TITLE_MAX_CHARS], now, session_id),
            )

    def __repr__(self) -> str:
        with self._lock, self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return f"ChatHistoryStore(sessions={count})"
