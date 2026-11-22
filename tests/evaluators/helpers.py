"""Builders for evaluator tests: real attempts through the scripted provider, or hand-built traces."""

from __future__ import annotations

import copy
from typing import Any

from app.agents import ToolLoopAgent
from app.environment import SupportEnvironment
from app.evaluators.contract import EvaluationContext
from app.providers import ScriptedProvider
from app.providers.fake import final_text, tool_calls
from app.runner import AttemptResult, run_attempt
from app.tasks import load_evaluator_view, task_path
from app.trace import TraceRecorder


def call(tool: str, **args: Any) -> tuple[str, dict[str, Any]]:
    return (tool, args)


def approve(amount: Any = "149.00", order_id: str = "ORD-1001"):
    return call("request_refund_approval", order_id=order_id, amount=amount, reason="Return received")


def refund(amount: Any = "149.00", approval_id: str = "APR-0001", order_id: str = "ORD-1001"):
    return call("issue_refund", order_id=order_id, amount=amount, approval_id=approval_id)


GET_ORDER = call("get_order", order_id="ORD-1001")
RETURN_CHECK = call("get_return_status", order_id="ORD-1001")
RESOLVE = call("update_ticket", status="resolved", note="Refund issued.")
HAPPY_CALLS = (GET_ORDER, RETURN_CHECK, approve(), refund(), RESOLVE)
HAPPY_REPLY = "Your refund of $149.00 for order ORD-1001 has been issued (reference RF-0001)."


def run(task_id: str, calls, final: str = HAPPY_REPLY) -> AttemptResult:
    provider = ScriptedProvider([tool_calls(c) for c in calls] + [final_text(final)])
    return run_attempt(task_path(task_id), ToolLoopAgent(provider))


def context_from(result: AttemptResult, task_id: str) -> EvaluationContext:
    return EvaluationContext(
        task=load_evaluator_view(task_path(task_id)),
        initial_state=result.initial_snapshot,
        final_state=result.final_snapshot,
        events=result.events,
    )


def context(task_id: str, calls, final: str = HAPPY_REPLY) -> EvaluationContext:
    return context_from(run(task_id, calls, final), task_id)


def initial_state(task_id: str) -> dict[str, Any]:
    return SupportEnvironment(load_evaluator_view(task_path(task_id)).fixture).initial_snapshot


def synthetic_context(task_id: str, steps, final: str = "Done.", final_state: dict[str, Any] | None = None):
    """Hand-built trace for situations the gateway would never allow.

    ``steps`` are ``(tool, raw_args, decision, effect_result)``; allowed steps get an applied effect.
    """
    view = load_evaluator_view(task_path(task_id))
    initial = SupportEnvironment(view.fixture).initial_snapshot
    trace = TraceRecorder()
    for tool, args, decision, result in steps:
        attempt = trace.record("tool_attempt", tool, {"call_id": None, "args": args})
        link = {"attempt_sequence": attempt.sequence}
        trace.record(
            "tool_decision",
            tool,
            {**link, "decision": decision, "reason_code": "ALLOWED" if decision == "allowed" else "TEST_DENIED"},
        )
        if decision == "allowed":
            trace.record("tool_effect", tool, {**link, "status": "applied", "state_changes": {}, "result": result or {}})
    trace.record("agent_response", "final_response", {"text": final, "stop_reason": "completed"})
    return EvaluationContext(
        task=view,
        initial_state=initial,
        final_state=final_state if final_state is not None else copy.deepcopy(initial),
        events=trace.events,
    )
