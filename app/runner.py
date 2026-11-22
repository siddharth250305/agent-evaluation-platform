"""Runs one attempt: fresh environment, gateway, agent, trace, then evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.accounting import UsageLedger, UsageSummary
from app.agents.base import AgentAdapter, AgentResult, AgentRuntime, StopReason
from app.environment import SupportEnvironment
from app.evaluators import EvaluationContext, Score, evaluate_all
from app.tasks import AgentTaskView, EvaluatorTaskView, load_agent_view, load_evaluator_view
from app.tools.gateway import CustomerContext, ToolGateway
from app.trace import TraceEvent, TraceRecorder

ExecutionStatus = Literal[
    "queued", "running", "completed", "timed_out", "budget_exceeded", "infrastructure_error", "cancelled"
]

# Hitting a turn or tool-call limit is a finished (unsuccessful) execution, not an infra failure.
_EXECUTION_STATUS: dict[StopReason, ExecutionStatus] = {
    "completed": "completed",
    "max_turns": "completed",
    "max_tool_calls": "completed",
    "output_truncated": "completed",
    "refused": "completed",
    "timeout": "timed_out",
    "budget_exceeded": "budget_exceeded",
    "provider_error": "infrastructure_error",
}


_SUCCESS_EVALUATORS = ("outcome", "policy", "response")


class AttemptCancelled(BaseException):
    """Raised at the next tool call once cancellation is requested.

    A ``BaseException`` (like ``KeyboardInterrupt``) so an adapter's broad
    ``except Exception`` cannot swallow it and keep executing tools.
    """


@dataclass(frozen=True)
class AttemptResult:
    task_id: str
    environment_id: str
    execution_status: ExecutionStatus
    agent_result: AgentResult | None  # None when the adapter crashed
    events: tuple[TraceEvent, ...]
    initial_snapshot: dict[str, Any]
    final_snapshot: dict[str, Any]
    usage: UsageSummary
    score: Score | None  # outcome score; None for infrastructure errors and cancellations (not graded)
    scores: tuple[Score, ...]  # every evaluator, in DEFAULT_EVALUATORS order; empty when not graded
    # outcome and policy and response all pass; None if any of them could not be evaluated.
    success: bool | None


def run_attempt(path: Path, agent: AgentAdapter) -> AttemptResult:
    return run_task_attempt(load_agent_view(path), load_evaluator_view(path), agent)


def run_task_attempt(
    agent_view: AgentTaskView,
    evaluator_view: EvaluatorTaskView,
    agent: AgentAdapter,
    *,
    environment_id: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    allowed_tools: Iterable[str] | None = None,
) -> AttemptResult:
    """One attempt in a fresh environment.

    ``cancel_check`` is consulted before the agent starts and before every tool
    call; once it returns True no further tool runs and the attempt ends
    ``cancelled``. ``allowed_tools`` (an agent configuration's allowlist)
    narrows the task's tools; it can never add tools the task does not offer.
    """
    if allowed_tools is not None:
        allowed = set(allowed_tools)
        agent_view = agent_view.model_copy(update={"tools": tuple(t for t in agent_view.tools if t in allowed)})
    env = SupportEnvironment(evaluator_view.fixture, environment_id=environment_id)
    trace = TraceRecorder()
    ledger = UsageLedger()
    gateway = ToolGateway(
        env,
        CustomerContext(customer_id=env.session.customer_id, ticket_id=env.session.ticket_id),
        trace,
        agent_view.tools,
    )

    def call_tool(name: str, raw_args: object, call_id: str | None = None) -> dict:
        if cancel_check is not None and cancel_check():
            trace.record("lifecycle", "cancellation_observed", {"blocked_tool": name})
            raise AttemptCancelled
        return gateway.call(name, raw_args, call_id)

    runtime = AgentRuntime(call_tool=call_tool, tool_specs=gateway.tool_specs, trace=trace, ledger=ledger)

    trace.record(
        "lifecycle",
        "attempt_started",
        {"task_id": evaluator_view.task_id, "environment_id": env.environment_id, "agent": agent.name},
    )
    agent_result: AgentResult | None
    execution_status: ExecutionStatus
    try:
        if cancel_check is not None and cancel_check():
            trace.record("lifecycle", "cancellation_observed", {"blocked_tool": None})
            raise AttemptCancelled
        agent_result = agent.run(agent_view, runtime)
        if not isinstance(agent_result, AgentResult):
            raise TypeError(f"adapter returned {type(agent_result).__name__}, not AgentResult")
        execution_status = _EXECUTION_STATUS[agent_result.stop_reason]
    except AttemptCancelled:
        # Partial work is not graded: scores would describe an attempt nobody let finish.
        agent_result = None
        execution_status = "cancelled"
    except Exception as exc:  # noqa: BLE001 - an adapter crash is an infrastructure error, recorded not raised
        agent_result = None
        execution_status = "infrastructure_error"
        trace.record("lifecycle", "attempt_failed", {"error_type": type(exc).__name__, "message": str(exc)})
    else:
        trace.record(
            "agent_response",
            "final_response",
            {"text": agent_result.final_response, "stop_reason": agent_result.stop_reason},
        )
    trace.record("lifecycle", "attempt_completed", {"execution_status": execution_status})

    final_snapshot = env.snapshot()
    scores: tuple[Score, ...] = ()
    if execution_status not in ("infrastructure_error", "cancelled"):
        ctx = EvaluationContext(
            task=evaluator_view, initial_state=env.initial_snapshot, final_state=final_snapshot, events=trace.events
        )
        scores = evaluate_all(ctx)
    by_name = {s.evaluator_name: s for s in scores}
    return AttemptResult(
        task_id=evaluator_view.task_id,
        environment_id=env.environment_id,
        execution_status=execution_status,
        agent_result=agent_result,
        events=trace.events,
        initial_snapshot=env.initial_snapshot,
        final_snapshot=final_snapshot,
        usage=ledger.summary(),
        score=by_name.get("outcome"),
        scores=scores,
        success=_task_success(by_name),
    )


def _task_success(scores: dict[str, Score]) -> bool | None:
    """success = outcome satisfied AND no hard policy violation AND the message matches reality.

    Any definite failure makes the run unsuccessful, even if another evaluator errored;
    only when nothing failed but something could not be evaluated is success unknown.
    Trajectory is diagnostic and does not gate success.
    """
    required = [scores.get(name) for name in _SUCCESS_EVALUATORS]
    if any(s is not None and s.status == "fail" for s in required):
        return False
    if any(s is None or s.status != "pass" for s in required):
        return None
    return True
