"""Evaluator-only task sections: expected outcome, trajectory constraints, reference solution."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.tools.schemas import TOOLS

Category = Literal["read_only_lookup", "eligible_action", "ineligible_action", "missing_info",
                   "access_boundary", "tool_failure", "duplicate_operation", "untrusted_content"]
TicketStatus = Literal["open", "pending", "resolved", "escalated"]

_ID = r"^[a-z][a-z0-9_]*$"
_REASON_CODE = r"^[A-Z][A-Z0-9_]*$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _known_tool(tool: str) -> str:
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}")
    return tool


class ExpectedOutcome(_Strict):
    outcome: Literal["refund_issued", "no_refund"]
    order_id: str | None = None
    refund_amount: Decimal | None = Field(default=None, gt=0)
    ticket_status: TicketStatus | None = None
    response_required_any: tuple[str, ...] = ()  # clarification/completeness anchors, evaluator-only

    @model_validator(mode="after")
    def _amount_matches_outcome(self) -> ExpectedOutcome:
        if self.outcome == "refund_issued" and (self.order_id is None or self.refund_amount is None):
            raise ValueError("refund_issued needs order_id and refund_amount")
        if self.outcome == "no_refund" and self.refund_amount is not None:
            raise ValueError("no_refund must not set refund_amount")
        return self


class RequiredCall(_Strict):
    id: str = Field(pattern=_ID)
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)  # subset that must match; numbers compare as decimals
    missing_code: str = Field(default="MISSING_REQUIRED_CALL", pattern=_REASON_CODE)
    # A denied attempt also satisfies this call (e.g. looking up an order that is not on the account).
    allow_denied: bool = False
    allow_failed: bool = False  # explicit attempted operation in a fault-handling scenario

    _check_tool = field_validator("tool")(_known_tool)


class TrajectorySpec(_Strict):
    tools_needed: bool
    required_calls: tuple[RequiredCall, ...] = ()
    order: tuple[tuple[str, str], ...] = ()  # (before_id, after_id) partial-order constraints
    max_repeats: int = Field(default=2, ge=1)  # identical calls allowed before EXCESS_REPETITION

    @model_validator(mode="after")
    def _consistent(self) -> TrajectorySpec:
        ids = [c.id for c in self.required_calls]
        if not self.tools_needed:
            if self.required_calls or self.order:
                raise ValueError("a task that needs no tools cannot require calls")
            return self
        if not ids:
            raise ValueError("a task that needs tools must require at least one call")
        if len(ids) != len(set(ids)):
            raise ValueError("required call ids must be unique")
        unknown = {i for pair in self.order for i in pair} - set(ids)
        if unknown:
            raise ValueError(f"order references unknown call ids: {sorted(unknown)}")
        _assert_acyclic(ids, self.order)
        return self


def _assert_acyclic(ids: list[str], edges: tuple[tuple[str, str], ...]) -> None:
    incoming = {i: 0 for i in ids}
    for _, after in edges:
        incoming[after] += 1
    ready = [i for i, n in incoming.items() if n == 0]
    seen = 0
    while ready:
        node = ready.pop()
        seen += 1
        for before, after in edges:
            if before == node:
                incoming[after] -= 1
                if incoming[after] == 0:
                    ready.append(after)
    if seen != len(ids):
        raise ValueError("order constraints contain a cycle")


class ReferenceCall(_Strict):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)

    _check_tool = field_validator("tool")(_known_tool)


class ReferenceSolution(_Strict):
    """A known-good run used to check the task mechanically. Never shown to the agent."""

    calls: tuple[ReferenceCall, ...] = ()
    final_response: str = Field(min_length=1)


class Review(_Strict):
    status: Literal["draft", "machine_checked", "human_reviewed"]
    notes: str = ""
