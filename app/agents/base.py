"""The common adapter interface every agent implements."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.accounting import UsageLedger
from app.providers.base import ToolSpec
from app.tasks import AgentTaskView
from app.trace import TraceRecorder

StopReason = Literal[
    "completed",
    "max_turns",
    "max_tool_calls",
    "timeout",
    "budget_exceeded",
    "provider_error",
    "output_truncated",
    "refused",
]


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    final_response: str
    stop_reason: StopReason
    metadata: dict[str, Any] = Field(default_factory=dict)


# (tool name, raw arguments, optional call id) -> {"ok": True, "result": ...} | {"ok": False, "error": {...}}
ToolCaller = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class AgentRuntime:
    """What the platform hands an adapter for one attempt. No fixtures, no expected outcomes."""

    call_tool: ToolCaller
    tool_specs: tuple[ToolSpec, ...]
    trace: TraceRecorder
    ledger: UsageLedger


class AgentAdapter(Protocol):
    name: str

    def run(self, task: AgentTaskView, runtime: AgentRuntime) -> AgentResult: ...
