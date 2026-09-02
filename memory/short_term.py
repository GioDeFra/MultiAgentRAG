"""
memory/short_term.py — Short-term (session) memory.

Keeps track of the current conversation:
  - All question/answer turns
  - Which agents were activated at each turn
  - Most recently mentioned concepts (country, legal area, etc.)

Everything lives in RAM — nothing is saved to disk.
When the user closes the program, this memory is lost.
"""

from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Deque, List, Optional


# ---------------------------------------------------------------------------
# ONE CONVERSATION TURN
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    turn_id: int
    query: str
    agents_activated: List[str]
    answer: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# SHORT-TERM MEMORY
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """
    Sliding window of the last N conversation turns.

    Parameters
    ----------
    max_turns : int
        How many turns to keep IN FULL. Once this limit is reached, the
        oldest turn is folded into `rolling_summary` instead of being
        discarded outright (see `summarizer` below).
    summarizer : Callable[[str, Turn], str], optional
        Called as `summarizer(existing_summary, evicted_turn) -> new_summary`
        whenever a turn is about to fall out of the window. Plug in your
        own LLM call here, e.g.:

            def my_summarizer(existing_summary: str, turn: Turn) -> str:
                prompt = (
                    f"Existing summary:\n{existing_summary}\n\n"
                    f"New turn to fold in:\nQ: {turn.query}\nA: {turn.answer}\n\n"
                    "Return an updated, still-concise summary."
                )
                return call_my_llm(prompt)

            stm = ShortTermMemory(max_turns=10, summarizer=my_summarizer)

        If omitted, a naive fallback just appends a truncated one-liner
        per evicted turn — good enough to not lose information silently,
        but a real LLM summarizer will compress much better over time.

    Note
    ----
    `turn_id` is a monotonically increasing counter across the whole
    session — it does NOT reset when old turns are evicted from the
    window. It's a stable, unique identifier (e.g. for logging), not
    a positional index into `_turns`.
    """

    def __init__(
        self,
        max_turns: int = 10,
        summarizer: Optional[Callable[[str, "Turn"], str]] = None,
    ) -> None:
        self.max_turns = max_turns
        self.summarizer = summarizer or self._default_summarizer
        self._turns: Deque[Turn] = deque(maxlen=max_turns)
        self._counter: int = 0
        self.rolling_summary: str = ""

    @staticmethod
    def _default_summarizer(existing_summary: str, turn: "Turn") -> str:
        """
        Naive fallback used when no `summarizer` is supplied to __init__:
        appends a truncated one-liner for the evicted turn instead of
        losing it outright. Not real compression — an LLM-backed
        summarizer does much better over a long session — but keeps some
        trace of every evicted turn even if the caller doesn't wire one
        up (see the `summarizer` param docstring above).
        """
        agents = ", ".join(turn.agents_activated) or "none"
        query_preview = ShortTermMemory._truncate(turn.query, 80)
        line = f"- {query_preview} (agents: {agents})"
        return (existing_summary + "\n" + line).strip() if existing_summary else line

    # ── Write ────────────────────────────────────────────────────────────────

    def add_turn(
        self,
        query: str,
        agents_activated: List[str],
        answer: str,
    ) -> Turn:
        """Save a completed conversation turn.

        If the window is already full, the oldest turn is folded into
        `rolling_summary` before being evicted, so it isn't lost outright.
        """
        if len(self._turns) == self.max_turns and self._turns:
            evicted = self._turns[0]  # about to be dropped by deque's maxlen
            self.rolling_summary = self.summarizer(self.rolling_summary, evicted)

        turn = Turn(
            turn_id=self._counter,
            query=query,
            agents_activated=agents_activated,
            answer=answer,
        )
        self._turns.append(turn)
        self._counter += 1
        return turn

    def load_history(self, turns: List["Turn"]) -> None:
        """
        Bulk-restore turns from a previously saved session (e.g. reloaded
        from ChatHistoryStore when the user picks a past chat in the UI).

        Unlike add_turn(), this does NOT call the summarizer — replaying a
        long session turn-by-turn through add_turn() would trigger the LLM
        summarizer repeatedly just to reconstruct history that's about to
        be overwritten anyway. Only the most recent `max_turns` are kept in
        the live window; anything older is dropped (rolling_summary is
        reset, since we don't persist it in chat history — the full
        transcript itself is the source of truth there).

        `turns` must be oldest-first, matching ChatHistoryStore.load_session().
        """
        self._turns.clear()
        self.rolling_summary = ""

        kept = turns[-self.max_turns:]
        for t in kept:
            self._turns.append(t)

        self._counter = (max((t.turn_id for t in turns), default=-1) + 1)

    # ── Read ─────────────────────────────────────────────────────────────────

    @property
    def turns(self) -> List[Turn]:
        """All turns currently in the window, oldest first."""
        return list(self._turns)

    @property
    def last_turn(self) -> Optional[Turn]:
        """The most recent turn, or None if empty."""
        return self._turns[-1] if self._turns else None

    def recent_queries(self, n: int = 3) -> List[str]:
        """Return the last n user questions."""
        return [t.query for t in list(self._turns)[-n:]]

    def recent_agents(self, n: int = 3) -> List[str]:
        """
        Return unique agent IDs used in the last n turns, most relevant first.

        Relevance = frequency weighted by recency: an agent used once in
        the very last turn can outrank one used more often several turns
        ago, so the ranking reflects what the conversation is about *now*,
        not just what came up most in the window overall.
        """
        turns = list(self._turns)[-n:]
        scores = {}
        for position, t in enumerate(turns, start=1):  # position=1 is oldest in window
            for aid in t.agents_activated:
                scores[aid] = scores.get(aid, 0) + position
        return sorted(scores, key=scores.get, reverse=True)

    def as_context_string(self, n_turns: int = 3) -> str:
        """
        Render the rolling summary (older turns, compressed) plus the last
        n turns in full, as plain text to inject into a prompt.
        """
        turns = list(self._turns)[-n_turns:]
        if not turns and not self.rolling_summary:
            return "(no previous turns in this session)"

        lines = []
        if self.rolling_summary:
            lines.append("Summary of earlier turns:")
            lines.append(self.rolling_summary)
            lines.append("")  # blank line separating summary from recent turns

        for t in turns:
            lines.append(f"Q: {t.query}")
            answer = t.answer.replace("\n", " ")
            preview = self._truncate(answer, 200)
            lines.append(f"A: {preview}")
        return "\n".join(lines)

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        """Truncate on a word boundary instead of mid-word."""
        if len(text) <= limit:
            return text
        cut = text[:limit].rsplit(" ", 1)[0]
        return cut + " ..."

    def reset(self) -> None:
        """
        Clear the session: turns, rolling summary, and the turn_id
        counter. Called when starting a brand new session (e.g.
        SupervisorAgent.new_session()) — turn_id is meant to be a stable
        counter within one session (see the class docstring), not across
        separate sessions, so a fresh session starts numbering from 0
        again.
        """
        self._turns.clear()
        self.rolling_summary = ""
        self._counter = 0

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return f"ShortTermMemory(turns={len(self._turns)}/{self.max_turns})"