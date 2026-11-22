import re
from datetime import UTC, datetime

import pytest

from app.evaluators import DEFAULT_EVALUATORS, Score, evaluate_all, run_evaluator
from app.evaluators.contract import EvaluationContext
from app.trace import TraceEvent
from tests.evaluators.helpers import HAPPY_CALLS, context

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def _broken_context(evaluator_view):
    malformed = TraceEvent(
        sequence=1, kind="tool_attempt", name="issue_refund", payload={},
        recorded_at=datetime.now(UTC), monotonic_ns=0,
    )
    return EvaluationContext(task=evaluator_view, initial_state={}, final_state={}, events=(malformed,))


def test_evaluators_have_unique_names_and_semver_versions():
    assert [e.name for e in DEFAULT_EVALUATORS] == ["outcome", "policy", "trajectory", "response"]
    assert all(SEMVER.fullmatch(e.version) for e in DEFAULT_EVALUATORS)


@pytest.mark.parametrize("evaluator", DEFAULT_EVALUATORS, ids=lambda e: e.name)
def test_any_evaluator_exception_becomes_evaluation_error(evaluator, evaluator_view):
    result = run_evaluator(evaluator, _broken_context(evaluator_view))
    assert (result.status, result.value, result.reason_code) == ("evaluation_error", None, "EVALUATOR_EXCEPTION")
    assert (result.evaluator_name, result.evaluator_version) == (evaluator.name, evaluator.version)
    assert result.evidence["exception_type"]


def test_crashes_in_unknown_evaluators_are_contained(evaluator_view):
    class Exploding:
        name, version = "exploding", "0.1.0"

        def evaluate(self, ctx):
            raise ZeroDivisionError("bug")

    result = run_evaluator(Exploding(), _broken_context(evaluator_view))
    assert (result.status, result.value, result.evaluator_name) == ("evaluation_error", None, "exploding")
    assert result.evidence["exception_type"] == "ZeroDivisionError"


def test_a_score_claiming_another_evaluators_identity_is_an_error(evaluator_view):
    class Impostor:
        name, version = "impostor", "1.0.0"

        def evaluate(self, ctx):
            return Score(evaluator_name="outcome", evaluator_version="1.0.0", status="pass", value=1.0, reason_code="OK")

    result = run_evaluator(Impostor(), _broken_context(evaluator_view))
    assert (result.status, result.evaluator_name) == ("evaluation_error", "impostor")


def test_evaluate_all_returns_one_score_per_evaluator_in_order():
    scores = evaluate_all(context("refund_eligible_001", HAPPY_CALLS))
    assert [s.evaluator_name for s in scores] == ["outcome", "policy", "trajectory", "response"]
    assert [s.status for s in scores] == ["pass", "pass", "pass", "pass"]


@pytest.mark.parametrize(
    ("task_id", "calls", "final"),
    [
        ("refund_eligible_001", HAPPY_CALLS, "Your refund of $149.00 has been issued."),
        ("refund_eligible_001", (), "Your refund has been processed."),
        ("no_tools_thanks_001", (), "You're welcome!"),
    ],
)
def test_every_score_obeys_the_contract(task_id, calls, final):
    for result in evaluate_all(context(task_id, calls, final)):
        assert SEMVER.fullmatch(result.evaluator_version)
        if result.status in ("pass", "fail"):
            assert 0.0 <= result.value <= 1.0
            assert result.applicability == "applicable"
        if result.status == "not_applicable":
            assert (result.value, result.applicability) == (None, "not_applicable")
        assert result.reason_code.isupper()
