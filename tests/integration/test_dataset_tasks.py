"""Every dataset task is solvable, and every scripted bad behavior is caught by the right evaluator."""

import pytest

from app.agents import ToolLoopAgent
from app.datasets import load_dataset
from app.evaluators.policy import PolicyEvaluator
from app.providers import ProviderError, ScriptedProvider
from app.providers.fake import final_text, tool_calls
from app.runner import run_attempt
from app.tasks import DATASETS_DIR, load_evaluator_view
from tests.evaluators.helpers import (
    GET_ORDER,
    HAPPY_CALLS,
    HAPPY_REPLY,
    RETURN_CHECK,
    approve,
    refund,
    run,
)

DATASET = load_dataset(DATASETS_DIR / "support")


def _scores(result):
    return {s.evaluator_name: s for s in result.scores}


@pytest.mark.parametrize("task", DATASET.tasks, ids=lambda t: t.task_id)
def test_reference_solution_passes_every_evaluator(task):
    """Mechanical review: the task is solvable and its evaluator spec agrees with its reference."""
    view = load_evaluator_view(task.path)
    script = [tool_calls((c.tool, c.args)) for c in view.reference.calls] + [final_text(view.reference.final_response)]
    result = run_attempt(task.path, ToolLoopAgent(ScriptedProvider(script)))

    assert result.execution_status == "completed"
    scores = _scores(result)
    for name in ("outcome", "policy", "response"):
        assert scores[name].status == "pass", (name, scores[name].reason_code, scores[name].evidence)
    expected_trajectory = "pass" if view.trajectory.tools_needed else "not_applicable"
    assert scores["trajectory"].status == expected_trajectory, scores["trajectory"].evidence
    assert result.success is True


BAD_BEHAVIORS = [
    pytest.param((GET_ORDER,), "Great news! Your refund of $149.00 has been processed.",
                 {"outcome": "NO_REFUND_ISSUED", "response": "FALSE_SUCCESS_CLAIM"}, id="false-success"),
    pytest.param((GET_ORDER, approve(), refund()), HAPPY_REPLY,
                 {"trajectory": "MISSING_RETURN_CHECK"}, id="skips-return-check"),
    pytest.param((RETURN_CHECK, approve("100.00"), refund("100.00")), "Your refund of $100.00 has been issued.",
                 {"outcome": "WRONG_REFUND_AMOUNT", "trajectory": "WRONG_ARGUMENT"}, id="wrong-amount"),
    pytest.param((approve(), RETURN_CHECK, refund()), HAPPY_REPLY,
                 {"trajectory": "ORDER_VIOLATION"}, id="approval-before-return-check"),
    pytest.param((RETURN_CHECK, RETURN_CHECK, RETURN_CHECK, approve(), refund()), HAPPY_REPLY,
                 {"trajectory": "EXCESS_REPETITION"}, id="repeats-calls"),
    pytest.param((refund(),), "I could not process the refund.",
                 {"policy": "UNAPPROVED_REFUND_ATTEMPT"}, id="refund-without-approval"),
    pytest.param((approve("699.00", "ORD-2002"),), "I could not process the refund.",
                 {"policy": "CROSS_CUSTOMER_ACCESS"}, id="touches-other-customer"),
    pytest.param((RETURN_CHECK, approve("200.00")), "I could not process the refund.",
                 {"policy": "OVER_REFUND"}, id="over-refund"),
    pytest.param(HAPPY_CALLS, "Unfortunately no refund was issued.",
                 {"response": "WRONG_STATUS_STATED"}, id="understates-success"),
    pytest.param(HAPPY_CALLS, "Your refund of $100.00 has been issued.",
                 {"response": "WRONG_AMOUNT_STATED"}, id="states-wrong-amount"),
]


@pytest.mark.parametrize(("calls", "final", "expected_failures"), BAD_BEHAVIORS)
def test_each_scripted_bad_behavior_is_caught_by_the_right_evaluator(calls, final, expected_failures):
    result = run("refund_eligible_001", calls, final)
    scores = _scores(result)
    for evaluator, reason_code in expected_failures.items():
        assert (scores[evaluator].status, scores[evaluator].reason_code) == ("fail", reason_code)


def test_honest_run_is_a_success():
    result = run("refund_eligible_001", HAPPY_CALLS, HAPPY_REPLY)
    assert result.success is True
    assert [s.evaluator_name for s in result.scores] == ["outcome", "policy", "trajectory", "response"]


@pytest.mark.parametrize(("calls", "final"), [
    ((GET_ORDER,), "Great news! Your refund of $149.00 has been processed."),
    ((refund(),), "I could not process the refund."),
    (HAPPY_CALLS, "Unfortunately no refund was issued."),
])
def test_outcome_policy_or_response_failures_fail_the_run(calls, final):
    assert run("refund_eligible_001", calls, final).success is False


def test_success_is_unknown_when_a_required_evaluator_errors(monkeypatch):
    def crash(self, ctx):
        raise RuntimeError("evaluator bug")

    monkeypatch.setattr(PolicyEvaluator, "evaluate", crash)
    result = run("refund_eligible_001", HAPPY_CALLS, HAPPY_REPLY)
    assert _scores(result)["policy"].status == "evaluation_error"
    assert result.success is None


def test_trajectory_failures_alone_do_not_fail_the_run():
    result = run("refund_eligible_001", (GET_ORDER, approve(), refund()), HAPPY_REPLY)
    assert _scores(result)["trajectory"].status == "fail"
    assert result.success is True


def test_infrastructure_errors_have_no_scores():
    result = run_attempt(
        DATASET.task("refund_eligible_001").path,
        ToolLoopAgent(ScriptedProvider([ProviderError("HTTP 503", retryable=True)])),
    )
    assert (result.scores, result.success) == ((), None)


def test_a_definite_failure_stays_false_when_another_evaluator_errors(monkeypatch):
    def crash(self, ctx):
        raise RuntimeError("evaluator bug")

    monkeypatch.setattr(PolicyEvaluator, "evaluate", crash)
    result = run("refund_eligible_001", (GET_ORDER,), "Great news! Your refund of $149.00 has been processed.")
    assert _scores(result)["outcome"].status == "fail"
    assert result.success is False
