"""Ordered, append-only trace of everything that happens in one attempt."""

from __future__ import annotations

import copy
import time
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

EventKind = Literal[
    "lifecycle",
    "model_request",
    "model_response",
    "tool_attempt",  # what the agent asked for, verbatim
    "tool_decision",  # what the gateway allowed or denied, and why
    "tool_effect",  # what actually changed in the environment
    "agent_response",
]


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int
    kind: EventKind
    name: str
    payload: dict[str, Any]
    recorded_at: datetime  # UTC wall clock, for display
    monotonic_ns: int  # for durations; never compare across processes


class TraceRecorder:
    """Assigns gap-free sequence numbers starting at 1, in recording order."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def record(self, kind: EventKind, name: str, payload: dict[str, Any] | None = None) -> TraceEvent:
        event = TraceEvent(
            sequence=len(self._events) + 1,
            kind=kind,
            name=name,
            # Copy so later mutation by the caller cannot rewrite history.
            payload=copy.deepcopy(payload or {}),
            recorded_at=datetime.now(UTC),
            monotonic_ns=time.monotonic_ns(),
        )
        self._events.append(event)
        return event.model_copy(deep=True)

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(event.model_copy(deep=True) for event in self._events)
