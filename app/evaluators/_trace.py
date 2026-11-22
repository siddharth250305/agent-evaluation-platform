"""Read-only helpers for evaluators that inspect the trace."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from app.tools.gateway import SCHEMA_REJECTIONS
from app.trace import TraceEvent

_DECIMAL_STRING = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool attempt joined with its gateway decision and environment effect."""

    sequence: int
    tool: str
    args: Any  # raw, exactly as the agent sent it
    decision: str | None  # "allowed" | "denied" | None if the trace has no decision
    effect: dict[str, Any] | None
    reason_code: str | None = None  # the gateway's internal reason for its decision

    @property
    def schema_valid(self) -> bool | None:
        """Matched an offered tool schema (it passed admission). None: the trace has no decision."""
        if self.decision is None:
            return None
        return self.reason_code not in SCHEMA_REJECTIONS

    @property
    def executed(self) -> bool:
        return self.effect is not None and self.effect.get("status") in ("applied", "no_change")

    @property
    def applied(self) -> bool:
        return self.effect is not None and self.effect.get("status") == "applied"


def tool_call_records(events: tuple[TraceEvent, ...]) -> list[ToolCallRecord]:
    decisions = {e.payload["attempt_sequence"]: e.payload for e in events if e.kind == "tool_decision"}
    effects = {e.payload["attempt_sequence"]: e.payload for e in events if e.kind == "tool_effect"}
    return [
        ToolCallRecord(
            sequence=e.sequence,
            tool=e.name,
            args=e.payload["args"],
            decision=decisions.get(e.sequence, {}).get("decision"),
            effect=effects.get(e.sequence),
            reason_code=decisions.get(e.sequence, {}).get("reason_code"),
        )
        for e in events
        if e.kind == "tool_attempt"
    ]


def final_response_event(events: tuple[TraceEvent, ...]) -> TraceEvent:
    for event in reversed(events):
        if event.kind == "agent_response":
            return event
    raise LookupError("trace has no agent_response event")


def to_decimal(value: Any) -> Decimal | None:
    """Numbers and numeric strings as Decimal; anything else (including bools and NaN) as None."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    if isinstance(value, str) and not _DECIMAL_STRING.fullmatch(value.strip()):
        return None  # rejects "1_000", "1e3", non-ASCII digits
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    return number if number.is_finite() else None
