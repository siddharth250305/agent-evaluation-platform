import dataclasses

import pytest

from app.datasets import load_dataset
from app.evaluators.contract import run_evaluator
from app.evaluators.trajectory import TrajectoryEvaluator
from app.task_spec import RequiredCall, TrajectorySpec
from app.tasks import DATASETS_DIR, load_evaluator_view
from tests.evaluators.helpers import GET_ORDER, HAPPY_CALLS, RETURN_CHECK, approve, call, context, refund

TRAJECTORY = TrajectoryEvaluator()


def score(ctx):
    return run_evaluator(TRAJECTORY, ctx)


def test_reference_flow_passes():
    result = score(context("refund_eligible_001", HAPPY_CALLS))
    assert (result.status, result.value, result.reason_code) == ("pass", 1.0, "TRAJECTORY_SATISFIED")
    assert set(result.evidence["matches"]) == {"check_return", "approval", "refund"}
    assert all(isinstance(seq, int) for seq in result.evidence["matches"].values())
    assert result.evidence["required_call_recall"] == 1.0
    assert result.evidence["tool_call_precision"] == pytest.approx(3 / 5)


@pytest.mark.parametrize(
    "calls",
    [
        (RETURN_CHECK, GET_ORDER, approve(), refund()),
        (GET_ORDER, RETURN_CHECK, approve(), refund()),
        (RETURN_CHECK, approve(), refund()),  # get_order is optional
    ],
)
def test_alternative_valid_sequences_pass(calls):
    assert score(context("refund_eligible_001", calls)).status == "pass"


@pytest.mark.parametrize("checks_first", ["refund_status", "return"])
def test_partial_order_allows_unordered_checks(checks_first):
    refund_status = call("get_refund_status", order_id="ORD-4002")
    return_check = call("get_return_status", order_id="ORD-4002")
    checks = (refund_status, return_check) if checks_first == "refund_status" else (return_check, refund_status)
    calls = (*checks, approve("80.00", "ORD-4002"), refund("80.00", "APR-0001", "ORD-4002"))
    assert score(context("partial_refund_remaining_001", calls)).status == "pass"


# --- failure codes -------------------------------------------------------------------


def test_missing_return_check_has_its_own_code():
    result = score(context("refund_eligible_001", (GET_ORDER, approve(), refund())))
    assert (result.status, result.reason_code) == ("fail", "MISSING_RETURN_CHECK")
    assert result.value == pytest.approx(2 / 3)
    assert result.evidence["unmatched"] == ["check_return"]


def test_missing_call_without_a_custom_code_uses_the_default():
    result = score(context("approval_rejected_001", (call("get_return_status", order_id="ORD-3002"),)))
    assert result.reason_code == "MISSING_REQUIRED_CALL"


def test_wrong_argument():
    result = score(context("refund_eligible_001", (RETURN_CHECK, approve("100.00"), refund("100.00"))))
    assert result.reason_code == "WRONG_ARGUMENT"


def test_order_violation():
    result = score(context("refund_eligible_001", (approve(), RETURN_CHECK, refund())))
    assert result.reason_code == "ORDER_VIOLATION"
    assert result.evidence["order_violations"] == [["check_return", "approval"]]


def test_excess_repetition():
    result = score(context("refund_eligible_001", (RETURN_CHECK, RETURN_CHECK, RETURN_CHECK, approve(), refund())))
    assert (result.status, result.reason_code) == ("fail", "EXCESS_REPETITION")
    assert result.evidence["repeated_calls"][0]["tool"] == "get_return_status"
    assert result.evidence["repeated_calls"][0]["count"] == 3


# --- matching semantics ---------------------------------------------------------------


def test_denied_calls_do_not_satisfy_requirements():
    result = score(context("refund_eligible_001", (RETURN_CHECK, refund())))  # refund denied: no approval
    assert result.status == "fail"
    assert result.evidence["matches"]["refund"] is None


def test_allow_denied_requirement_is_satisfied_by_a_denied_lookup():
    calls = (call("get_order", order_id="ORD-2002"),)
    assert score(context("cross_customer_request_001", calls, final="Not found.")).status == "pass"


def test_required_calls_are_matched_one_to_one():
    spec = TrajectorySpec(
        tools_needed=True,
        required_calls=(
            RequiredCall(id="first", tool="get_order", args={"order_id": "ORD-1001"}),
            RequiredCall(id="second", tool="get_order", args={"order_id": "ORD-1001"}),
        ),
    )

    def with_spec(ctx):
        return dataclasses.replace(ctx, task=ctx.task.model_copy(update={"trajectory": spec}))

    once = score(with_spec(context("refund_eligible_001", (GET_ORDER,))))
    twice = score(with_spec(context("refund_eligible_001", (GET_ORDER, GET_ORDER))))
    assert (once.status, once.reason_code) == ("fail", "MISSING_REQUIRED_CALL")
    assert twice.status == "pass"


def test_numeric_arguments_are_compared_as_decimals():
    calls = (RETURN_CHECK, approve(149), refund(149.0))
    assert score(context("refund_eligible_001", calls)).status == "pass"


def test_tasks_that_need_no_tools_are_not_applicable():
    result = score(context("no_tools_thanks_001", (), final="You're welcome!"))
    assert (result.status, result.value, result.applicability, result.reason_code) == (
        "not_applicable",
        None,
        "not_applicable",
        "TOOLS_NOT_NEEDED",
    )


# --- every missing-call code declared in the dataset is reachable -------------------------


def _drop_cases():
    dataset = load_dataset(DATASETS_DIR / "support")
    for task in dataset.tasks:
        view = load_evaluator_view(task.path)
        for required in view.trajectory.required_calls:
            yield pytest.param(task.task_id, required.id, required.missing_code, id=f"{task.task_id}-{required.id}")


@pytest.mark.parametrize(("task_id", "required_id", "missing_code"), list(_drop_cases()))
def test_dropping_a_required_call_reports_its_missing_code(task_id, required_id, missing_code):
    from app.tasks import task_path

    view = load_evaluator_view(task_path(task_id))
    required = next(r for r in view.trajectory.required_calls if r.id == required_id)
    calls = [(c.tool, c.args) for c in view.reference.calls]
    drop = next(
        i
        for i, (tool, args) in enumerate(calls)
        if tool == required.tool and all(str(args.get(k)) == str(v) for k, v in required.args.items())
    )
    del calls[drop]
    result = score(context(task_id, tuple(calls), final=view.reference.final_response))
    assert (result.status, result.reason_code) == ("fail", missing_code)


# --- regressions from the Phase 3 review: matching cost and strictness -------------------


def _spec_ctx(ctx, spec):
    return dataclasses.replace(ctx, task=ctx.task.model_copy(update={"trajectory": spec}))


def test_many_identical_candidates_match_quickly():
    import time

    from tests.evaluators.helpers import synthetic_context

    steps = [("get_order", {"order_id": "ORD-1001"}, "allowed", {}) for _ in range(20)]
    spec = TrajectorySpec(
        tools_needed=True,
        max_repeats=50,
        required_calls=tuple(RequiredCall(id=f"r{i}", tool="get_order", args={"order_id": "ORD-1001"}) for i in range(6)),
    )
    started = time.perf_counter()
    result = score(_spec_ctx(synthetic_context("refund_eligible_001", steps), spec))
    assert result.status == "pass"
    assert time.perf_counter() - started < 1.0


def test_impossible_order_with_many_candidates_is_reported_quickly():
    import time

    from tests.evaluators.helpers import synthetic_context

    steps = [("get_order", {"order_id": "ORD-1001"}, "allowed", {}) for _ in range(10)]
    steps += [("get_return_status", {"order_id": "ORD-1001"}, "allowed", {}) for _ in range(10)]
    spec = TrajectorySpec(
        tools_needed=True,
        max_repeats=50,
        required_calls=(
            RequiredCall(id="check", tool="get_return_status"),
            *(RequiredCall(id=f"look{i}", tool="get_order") for i in range(5)),
        ),
        order=(("check", "look0"),),
    )
    started = time.perf_counter()
    result = score(_spec_ctx(synthetic_context("refund_eligible_001", steps), spec))
    assert result.reason_code == "ORDER_VIOLATION"
    assert time.perf_counter() - started < 1.0


def test_search_budget_overrun_is_an_evaluation_error(monkeypatch):
    import app.evaluators.trajectory as module

    monkeypatch.setattr(module, "_SEARCH_BUDGET", 3)
    result = score(context("refund_eligible_001", (approve(), RETURN_CHECK, refund())))
    assert (result.status, result.value) == ("evaluation_error", None)


def test_booleans_and_loose_numbers_do_not_match_numeric_arguments():
    from app.evaluators.trajectory import _args_match

    assert not _args_match({"amount": 1}, {"amount": True})
    assert not _args_match({"amount": "1000"}, {"amount": "1_000"})
    assert _args_match({"amount": "149.00"}, {"amount": 149})
