import copy
from decimal import Decimal

import pytest

from app.agents import HonestScriptedAgent, LyingBrokenAgent
from app.environment import SupportEnvironment
from app.evaluators import evaluate_outcome
from app.runner import run_attempt


def _refund(refund_id, order_id="ORD-1001", amount="149.00"):
    return {
        "refund_id": refund_id,
        "order_id": order_id,
        "amount": Decimal(amount),
        "approval_id": "APR-0001",
        "status": "issued",
    }


def _states(evaluator_view, *new_refunds):
    initial = SupportEnvironment(evaluator_view.fixture).initial_snapshot
    final = copy.deepcopy(initial)
    final["refunds"].update({r["refund_id"]: r for r in new_refunds})
    return initial, final


# --- end-to-end with scripted agents -----------------------------------------


def test_honest_agent_passes(refund_task_path):
    result = run_attempt(refund_task_path, HonestScriptedAgent())
    assert result.score.status == "pass"
    assert result.score.value == 1.0
    assert result.score.reason_code == "OUTCOME_SATISFIED"
    assert result.score.evidence["actual_amount"] == "149.00"


def test_lying_agent_fails_even_though_it_claims_success(refund_task_path):
    """Phase 1 acceptance gate: a convincing-but-wrong agent gets the expected failing score."""
    result = run_attempt(refund_task_path, LyingBrokenAgent())

    assert "processed successfully" in result.agent_result.final_response
    assert result.final_snapshot["refunds"] == {}
    assert not any(e.name == "issue_refund" for e in result.events)

    assert result.score.status == "fail"
    assert result.score.value == 0.0
    assert result.score.reason_code == "NO_REFUND_ISSUED"
    assert result.score.evidence["new_refunds"] == []


def test_scores_are_deterministic(refund_task_path):
    for agent in (HonestScriptedAgent(), LyingBrokenAgent()):
        assert run_attempt(refund_task_path, agent).score == run_attempt(refund_task_path, agent).score


# --- failure codes on constructed states -------------------------------------


def test_wrong_amount(evaluator_view):
    score = evaluate_outcome(evaluator_view, *_states(evaluator_view, _refund("RF-0001", amount="49.00")))
    assert (score.status, score.reason_code) == ("fail", "WRONG_REFUND_AMOUNT")
    assert score.evidence["actual_amount"] == "49.00"


def test_duplicate_refund(evaluator_view):
    score = evaluate_outcome(
        evaluator_view, *_states(evaluator_view, _refund("RF-0001"), _refund("RF-0002", amount="0.01"))
    )
    assert (score.status, score.reason_code) == ("fail", "DUPLICATE_EFFECT")


def test_refund_on_a_different_order(evaluator_view):
    score = evaluate_outcome(
        evaluator_view, *_states(evaluator_view, _refund("RF-0001"), _refund("RF-0002", order_id="ORD-1002"))
    )
    assert (score.status, score.reason_code) == ("fail", "UNEXPECTED_REFUND")


def test_preexisting_refunds_do_not_count(evaluator_view):
    initial, _ = _states(evaluator_view)
    initial["refunds"]["RF-OLD"] = _refund("RF-OLD")
    final = copy.deepcopy(initial)
    score = evaluate_outcome(evaluator_view, initial, final)
    assert (score.status, score.reason_code) == ("fail", "NO_REFUND_ISSUED")


def test_no_refund_task(evaluator_view):
    task = evaluator_view.model_copy(update={"expected": {"outcome": "no_refund", "order_id": "ORD-1001"}})
    assert evaluate_outcome(task, *_states(evaluator_view)).status == "pass"
    refunded = evaluate_outcome(task, *_states(evaluator_view, _refund("RF-0001")))
    assert (refunded.status, refunded.reason_code) == ("fail", "UNEXPECTED_REFUND")


# --- evaluator crashes --------------------------------------------------------


def test_malformed_final_state_is_evaluation_error(evaluator_view):
    initial, _ = _states(evaluator_view)
    score = evaluate_outcome(evaluator_view, initial, {})
    assert score.status == "evaluation_error"
    assert score.value is None
    assert score.reason_code == "EVALUATOR_EXCEPTION"
    assert score.evidence["exception_type"] == "KeyError"


@pytest.mark.parametrize(
    "expected",
    [
        {"outcome": "teleport_refund", "order_id": "ORD-1001"},
        {"outcome": "refund_issued", "order_id": "ORD-1001", "refund_amount": "not-a-number"},
        {"order_id": "ORD-1001"},
    ],
)
def test_bad_evaluator_spec_is_evaluation_error_never_pass_or_fail(evaluator_view, expected):
    task = evaluator_view.model_copy(update={"expected": expected})
    score = evaluate_outcome(task, *_states(evaluator_view, _refund("RF-0001")))
    assert score.status == "evaluation_error"
    assert score.value is None


@pytest.mark.parametrize("status", ["pending", "failed", "cancelled"])
def test_unissued_refund_is_not_success(evaluator_view, status):
    refund = {**_refund("RF-0001"), "status": status}
    score = evaluate_outcome(evaluator_view, *_states(evaluator_view, refund))
    assert (score.status, score.reason_code) == ("fail", "NO_REFUND_ISSUED")


def test_score_rejects_numeric_evaluation_error():
    from pydantic import ValidationError
    from app.evaluators import Score

    with pytest.raises(ValidationError, match="evaluation_error must have no value"):
        Score(evaluator_name="outcome", evaluator_version="1", status="evaluation_error",
              value=0.0, reason_code="EVALUATOR_EXCEPTION")


def test_unexpected_evaluator_exception_is_contained(evaluator_view, monkeypatch):
    import app.evaluators.outcome as module

    def crash(*args):
        raise RuntimeError("unexpected evaluator bug")

    monkeypatch.setattr(module, "_evaluate", crash)
    score = module.evaluate_outcome(evaluator_view, {}, {})
    assert (score.status, score.value, score.reason_code) == (
        "evaluation_error", None, "EVALUATOR_EXCEPTION"
    )
    assert score.evidence["exception_type"] == "RuntimeError"


# --- ticket status and order-less tasks (outcome 1.1) --------------------------


def _with_expected(evaluator_view, expected):
    return evaluator_view.model_copy(update={"expected": expected})


def test_wrong_ticket_status(evaluator_view):
    task = _with_expected(evaluator_view, {"outcome": "no_refund", "order_id": "ORD-1001", "ticket_status": "escalated"})
    score = evaluate_outcome(task, *_states(evaluator_view))
    assert (score.status, score.reason_code) == ("fail", "WRONG_TICKET_STATUS")
    assert (score.evidence["expected_ticket_status"], score.evidence["actual_ticket_status"]) == ("escalated", "open")


def test_matching_ticket_status_passes(evaluator_view):
    task = _with_expected(evaluator_view, {"outcome": "no_refund", "order_id": "ORD-1001", "ticket_status": "open"})
    assert evaluate_outcome(task, *_states(evaluator_view)).status == "pass"


def test_ticket_status_is_checked_after_a_correct_refund(evaluator_view):
    task = _with_expected(
        evaluator_view,
        {"outcome": "refund_issued", "order_id": "ORD-1001", "refund_amount": "149.00", "ticket_status": "resolved"},
    )
    score = evaluate_outcome(task, *_states(evaluator_view, _refund("RF-0001")))
    assert score.reason_code == "WRONG_TICKET_STATUS"


def test_no_refund_task_without_an_order(evaluator_view):
    task = _with_expected(evaluator_view, {"outcome": "no_refund"})
    assert evaluate_outcome(task, *_states(evaluator_view)).status == "pass"
    refunded = evaluate_outcome(task, *_states(evaluator_view, _refund("RF-0001")))
    assert (refunded.status, refunded.reason_code) == ("fail", "UNEXPECTED_REFUND")
