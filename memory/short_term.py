"""In-memory sliding window for the current conversation.

Only the most recent ``max_turns`` question/answer pairs are retained. When
the window is full, appending a new turn discards the oldest one. There is no
LLM summarization and no persistence here; full chat persistence belongs to
``memory/chat_history.py``.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List


@dataclass
class Turn:
    """One completed question/answer exchange in the current session."""

    turn_id: int
    query: str
    agents_activated: List[str]
    answer: str


class ShortTermMemory:
    """Keep the last ``max_turns`` completed turns in RAM, oldest first.

    ``turn_id`` increases throughout one session and does not change when an
    old turn leaves the window. Calling :meth:`reset` starts numbering again.
    """

    def __init__(self, max_turns: int = 10) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        self.max_turns = max_turns
        self._turns: Deque[Turn] = deque(maxlen=max_turns)
        self._counter = 0

    def add_turn(
        self,
        query: str,
        agents_activated: List[str],
        answer: str,
    ) -> Turn:
        """Append a completed turn; ``deque`` evicts the oldest if full."""
        turn = Turn(
            turn_id=self._counter,
            query=query,
            agents_activated=agents_activated,
            answer=answer,
        )
        self._turns.append(turn)
        self._counter += 1
        return turn

    def load_history(self, turns: List[Turn]) -> None:
        """Restore only the newest turns from a full persisted transcript.

        The transcript must be oldest-first, matching
        ``ChatHistoryStore.load_session()``. Older turns remain available to
        the UI from chat history but are not loaded into prompt memory.
        """
        self._turns.clear()
        self._turns.extend(turns[-self.max_turns:])
        self._counter = max((turn.turn_id for turn in turns), default=-1) + 1

    def as_context_string(self, n_turns: int = 3) -> str:
        """Render the last ``n_turns`` questions and full answers for a prompt."""
        if n_turns < 1:
            raise ValueError("n_turns must be at least 1")

        turns = list(self._turns)[-n_turns:]
        if not turns:
            return "(no previous turns in this session)"

        lines = []
        for turn in turns:
            lines.append(f"Q: {turn.query}")
            lines.append(f"A: {turn.answer}")
        return "\n".join(lines)

    def __len__(self) -> int:
        """Return the number of turns currently retained in memory."""
        return len(self._turns)

    def as_routing_context(self, n_turns: int = 3) -> List[dict]:
        """Keep user choices distinct from assistant suggestions and source cards."""
        if n_turns < 1:
            raise ValueError("n_turns must be at least 1")
        return [
            {"query": turn.query, "answer": turn.answer.split("\n\n---\n\n### Sources used", 1)[0]}
            for turn in list(self._turns)[-n_turns:]
        ]

    def reset(self) -> None:
        """Clear the in-memory window and restart the session turn counter."""
        self._turns.clear()
        self._counter = 0
