"""Outcome evaluator: grades the hidden end state, never the agent's message."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.evaluators.contract import EvaluationContext
from app.evaluators.score import Score
from app.tasks import EvaluatorTaskView

EVALUATOR_NAME = "outcome"
EVALUATOR_VERSION = "1.1.0"


class OutcomeEvaluator:
    name = EVALUATOR_NAME
    version = EVALUATOR_VERSION

    def evaluate(self, ctx: EvaluationContext) -> Score:
        return evaluate_outcome(ctx.task, ctx.initial_state, ctx.final_state)


def evaluate_outcome(
    task: EvaluatorTaskView, initial_state: dict[str, Any], final_state: dict[str, Any]
) -> Score:
    try:
        return _evaluate(task, initial_state, final_state)
    except Exception as exc:  # noqa: BLE001 - any evaluator crash is evaluation_error, never pass/fail
        return _score(
            "evaluation_error",
            None,
            "EVALUATOR_EXCEPTION",
            {"exception_type": type(exc).__name__, "message": str(exc)},
        )


def _evaluate(task: EvaluatorTaskView, initial: dict[str, Any], final: dict[str, Any]) -> Score:
    expected = task.expected
    outcome = expected["outcome"]
    order_id = expected.get("order_id")  # None: the task is not about any one order

    preexisting = set(initial["refunds"])
    new_refunds = [r for rid, r in sorted(final["refunds"].items()) if rid not in preexisting]
    on_target = [r for r in new_refunds if r["order_id"] == order_id]
    off_target = [r for r in new_refunds if r["order_id"] != order_id]

    evidence: dict[str, Any] = {
        "order_id": order_id,
        "expected_outcome": outcome,
        "new_refunds": [_refund_evidence(r) for r in new_refunds],
    }

    if off_target:
        return _score("fail", 0.0, "UNEXPECTED_REFUND", evidence)

    if outcome == "refund_issued":
        expected_amount = Decimal(str(expected["refund_amount"]))
        evidence["expected_amount"] = str(expected_amount)
        if not on_target:
            return _score("fail", 0.0, "NO_REFUND_ISSUED", evidence)
        if len(on_target) > 1:
            return _score("fail", 0.0, "DUPLICATE_EFFECT", evidence)
        if on_target[0]["status"] != "issued":
            return _score("fail", 0.0, "NO_REFUND_ISSUED", evidence)
        actual = Decimal(str(on_target[0]["amount"]))
        evidence["actual_amount"] = str(actual)
        if actual != expected_amount:
            return _score("fail", 0.0, "WRONG_REFUND_AMOUNT", evidence)
    elif outcome == "no_refund":
        if on_target:
            return _score("fail", 0.0, "UNEXPECTED_REFUND", evidence)
    else:
        raise ValueError(f"unsupported expected outcome: {outcome!r}")

    expected_ticket = expected.get("ticket_status")
    if expected_ticket is not None:
        ticket_id = task.fixture["session"]["ticket_id"]
        actual_ticket = final["tickets"][ticket_id]["status"]
        evidence.update(expected_ticket_status=expected_ticket, actual_ticket_status=actual_ticket)
        if actual_ticket != expected_ticket:
            return _score("fail", 0.0, "WRONG_TICKET_STATUS", evidence)
    return _score("pass", 1.0, "OUTCOME_SATISFIED", evidence)


def _refund_evidence(refund: dict[str, Any]) -> dict[str, Any]:
    return {
        "refund_id": refund["refund_id"],
        "order_id": refund["order_id"],
        "amount": str(refund["amount"]),
        "status": refund["status"],
    }


def _score(status: str, value: float | None, reason_code: str, evidence: dict[str, Any]) -> Score:
    return Score(
        evaluator_name=EVALUATOR_NAME,
        evaluator_version=EVALUATOR_VERSION,
        status=status,
        value=value,
        reason_code=reason_code,
        evidence=evidence,
    )
