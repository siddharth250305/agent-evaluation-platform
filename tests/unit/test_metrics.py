"""Scorecard and comparison math: denominators, unknown cost, retries, regressions, task-level bootstrap."""

from __future__ import annotations

import itertools
from decimal import Decimal

import pytest

from app.metrics import (
    AttemptFacts,
    Ratio,
    RunFacts,
    ScoreFacts,
    by_task,
    cluster_bootstrap,
    compare,
    distribution,
    failure_reasons,
    hard_violations,
    money,
    quantile,
    scorecard,
)

_ids = itertools.count(1)


def attempt(number: int = 1, *, cost: str | None = "0.01", status: str = "completed", started: bool = True,
            duration: float | None = 100.0, tokens: int | None = 10) -> AttemptFacts:
    return AttemptFacts(
        attempt_number=number, status=status, started=started,
        cost_usd=Decimal(cost) if cost is not None else None,
        input_tokens=tokens, output_tokens=tokens, duration_ms=duration,
    )


def policy(status: str = "pass", *, executed: bool = False, codes: tuple[str, ...] = ()) -> ScoreFacts:
    violations = [{"code": c, "attempt_sequence": 3, "executed": executed} for c in codes]
    return ScoreFacts("policy", "1.1.0", status, 1.0 if status == "pass" else 0.0,
                      codes[0] if codes else "POLICY_COMPLIANT", "applicable",
                      {"violations": violations, "executed_violation": executed})


def run(task: str = "t1", evaluation: str = "pass", *, rep: int = 1, status: str = "completed",
        attempts: tuple[AttemptFacts, ...] | None = None, scores: tuple[ScoreFacts, ...] | None = None,
        queue_ms: float | None = 5.0, category: str = "refund", config: str = "cfg") -> RunFacts:
    if scores is None:
        scores = (policy(),) if evaluation in ("pass", "fail") else ()
    return RunFacts(
        run_id=f"run-{next(_ids)}", agent_config_id=config, agent_config_name=config, task_id=task,
        task_version_id=f"tv-{task}", category=category, repetition=rep, status=status,
        evaluation_status=evaluation, success={"pass": True, "fail": False}.get(evaluation),
        queue_ms=queue_ms, attempts=(attempt(),) if attempts is None else attempts, scores=scores,
    )


def runs_for(results: dict[str, list[str]]) -> list[RunFacts]:
    """``{"task": ["pass", "fail"]}`` -> one run per repetition."""
    return [run(task, evaluation, rep=i) for task, evaluations in results.items()
            for i, evaluation in enumerate(evaluations, start=1)]


# --- primitives --------------------------------------------------------------------------


def test_ratio_with_an_empty_denominator_is_unknown_not_zero() -> None:
    assert Ratio(0, 0).value is None
    assert Ratio(0, 0).as_dict() == {"numerator": 0, "denominator": 0, "value": None}
    assert Ratio(1, 4).value == 0.25


def test_money_is_a_plain_decimal_string_and_unknown_stays_none() -> None:
    assert money(Decimal("0E-8")) == "0"
    assert money(Decimal("0.00240000")) == "0.0024"
    assert money(Decimal("1E+1")) == "10"
    assert money(None) is None


def test_quantile_interpolates_and_distribution_counts_unknowns() -> None:
    assert quantile([], 0.5) is None
    assert quantile([1.0, 3.0], 0.5) == 2.0
    assert quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) == pytest.approx(4.8)
    summary = distribution([10.0, None, 30.0])
    assert summary == {"n": 2, "unknown": 1, "mean": 20.0, "median": 20.0, "p95": pytest.approx(29.0)}
    assert distribution([None])["mean"] is None


# --- scorecard denominators --------------------------------------------------------------


def test_scorecard_reports_every_rate_with_its_own_denominator() -> None:
    runs = [
        run("a", "pass"),
        run("b", "fail"),
        run("c", "evaluation_error"),
        run("d", "not_applicable", status="infrastructure_error", attempts=(attempt(cost=None),)),
    ]
    card = scorecard(runs)
    assert card["planned_runs"] == 4 and card["scored_runs"] == 2 and card["evaluation_errors"] == 1
    assert card["verified_success_rate"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert card["strict_success_rate"] == {"numerator": 1, "denominator": 4, "value": 0.25}
    assert card["evaluation_coverage"] == {"numerator": 2, "denominator": 4, "value": 0.5}
    assert "accuracy" not in card and "score" not in card  # no blended number


def test_unsafe_and_executed_rates_use_policy_scored_runs_as_denominator() -> None:
    runs = [
        run("a", "fail", scores=(policy("fail", executed=True, codes=("OVER_REFUND",)),)),
        run("b", "fail", scores=(policy("fail", executed=False, codes=("UNAPPROVED_REFUND_ATTEMPT",)),)),
        run("c", "pass", scores=(policy("pass"),)),
        run("d", "not_applicable", status="infrastructure_error", scores=()),  # never policy-scored
    ]
    card = scorecard(runs)
    assert card["unsafe_attempt_rate"] == {"numerator": 2, "denominator": 3, "value": pytest.approx(2 / 3)}
    assert card["executed_violation_rate"] == {"numerator": 1, "denominator": 3, "value": pytest.approx(1 / 3)}


def test_policy_evaluation_error_is_excluded_from_the_safety_denominator() -> None:
    errored = ScoreFacts("policy", "1.1.0", "evaluation_error", None, "EVALUATOR_EXCEPTION", "applicable", {})
    card = scorecard([run("a", "evaluation_error", scores=(errored,)), run("b", "pass")])
    assert card["unsafe_attempt_rate"] == {"numerator": 0, "denominator": 1, "value": 0.0}


def test_passed_all_repetitions_groups_runs_by_task() -> None:
    card = scorecard(runs_for({"a": ["pass", "pass", "pass"], "b": ["pass", "fail", "pass"], "c": ["pass"]}))
    assert card["tasks"] == 3
    assert card["passed_all_repetitions"] == {"numerator": 2, "denominator": 3, "value": pytest.approx(2 / 3)}


def test_task_outcomes() -> None:
    tasks = by_task(runs_for({
        "all": ["pass", "pass"], "some_fail": ["pass", "fail"], "unscored": ["pass", "evaluation_error"],
    }))
    assert tasks["all"].outcome == "passed_all"
    assert tasks["some_fail"].outcome == "failed_any"
    assert tasks["unscored"].outcome == "inconclusive"
    assert tasks["unscored"].pass_rate.as_dict() == {"numerator": 1, "denominator": 2, "value": 0.5}


# --- cost and usage ----------------------------------------------------------------------


def test_one_unknown_cost_makes_the_total_unknown_never_zero() -> None:
    runs = [run("a", "pass"), run("b", "pass", attempts=(attempt(cost=None, tokens=None),))]
    usage = scorecard(runs)["usage"]
    assert usage["cost_usd"] is None
    assert usage["attempts_with_unknown_cost"] == 1
    assert usage["cost_per_successful_run_usd"]["value"] is None
    assert usage["cost_per_successful_run_usd"]["denominator"] == 2
    assert usage["input_tokens"] is None and usage["attempts_with_unknown_tokens"] == 1


def test_retries_count_toward_cost_and_unstarted_or_running_attempts_do_not() -> None:
    retried = run("a", "pass", attempts=(
        attempt(1, cost="0.01", status="infrastructure_error"),
        attempt(2, cost="0.02"),
    ))
    pending = run("b", "pending", status="running", attempts=(
        attempt(1, cost="0.04", status="infrastructure_error"),
        attempt(2, cost=None, status="running"),  # still running: its usage is not final yet
    ))
    never = run("c", "not_applicable", status="cancelled", attempts=(attempt(1, cost=None, status="cancelled", started=False),))
    usage = scorecard([retried, pending, never])["usage"]
    assert usage["cost_usd"] == "0.07"
    assert usage["attempts"] == 3 and usage["retry_attempts"] == 1
    assert usage["cost_per_successful_run_usd"] == {"numerator_usd": "0.07", "denominator": 1, "value": "0.07"}
    assert retried.cost_usd == Decimal("0.03")


def test_cost_per_successful_run_is_unknown_without_successes() -> None:
    usage = scorecard([run("a", "fail")])["usage"]
    assert usage["cost_usd"] == "0.01"
    assert usage["cost_per_successful_run_usd"] == {"numerator_usd": "0.01", "denominator": 0, "value": None}


def test_execution_latency_comes_from_the_result_attempt_and_queue_latency_is_separate() -> None:
    r = run("a", "pass", queue_ms=250.0, attempts=(attempt(1, duration=900.0, status="infrastructure_error"),
                                                  attempt(2, duration=120.0)))
    assert r.execution_ms == 120.0
    latency = scorecard([r, run("b", "pass", queue_ms=None)])["latency_ms"]
    assert latency["queue"]["n"] == 1 and latency["queue"]["unknown"] == 1
    assert latency["execution"]["median"] == pytest.approx(110.0)


# --- comparison --------------------------------------------------------------------------


def test_wins_losses_ties_regressions_and_improvements() -> None:
    baseline = runs_for({
        "regressed": ["pass", "pass"],
        "improved": ["fail", "fail"],
        "tie_pass": ["pass", "pass"],
        "tie_mixed": ["pass", "fail"],
        "partial_win": ["fail", "fail"],
    })
    candidate = runs_for({
        "regressed": ["pass", "fail"],
        "improved": ["pass", "pass"],
        "tie_pass": ["pass", "pass"],
        "tie_mixed": ["fail", "pass"],
        "partial_win": ["pass", "fail"],
    })
    result = compare(baseline, candidate, resamples=200)
    s = result["summary"]
    assert (s["tasks_compared"], s["wins"], s["losses"], s["ties"]) == (5, 2, 1, 2)
    assert s["regressed_tasks"] == ["regressed"] and s["regressions"] == 1
    assert s["improved_tasks"] == ["improved"] and s["improvements"] == 1
    by_id = {t["task_id"]: t for t in result["tasks"]}
    assert by_id["partial_win"]["result"] == "win" and not by_id["partial_win"]["improvement"]
    assert by_id["regressed"]["baseline"]["pass_rate"] == {"numerator": 2, "denominator": 2, "value": 1.0}


def test_an_unscored_candidate_repetition_is_a_loss_but_not_a_regression() -> None:
    result = compare(runs_for({"t": ["pass", "pass"]}), runs_for({"t": ["pass", "evaluation_error"]}), resamples=0)
    [task] = result["tasks"]
    assert task["result"] == "loss" and task["candidate"]["outcome"] == "inconclusive"
    assert task["regression"] is False


def test_pass_rates_compare_across_different_repetition_counts() -> None:
    result = compare(runs_for({"t": ["pass", "fail"]}), runs_for({"t": ["pass", "pass", "fail", "fail"]}), resamples=0)
    assert result["tasks"][0]["result"] == "tie"  # 1/2 == 2/4


def test_metric_deltas_are_candidate_minus_baseline_with_both_denominators() -> None:
    baseline = runs_for({"a": ["pass"], "b": ["fail"], "c": ["fail"], "d": ["evaluation_error"]})
    candidate = runs_for({"a": ["pass"], "b": ["pass"], "c": ["pass"], "d": ["fail"]})
    metrics = compare(baseline, candidate, resamples=0)["metrics"]
    verified = metrics["verified_success_rate"]
    assert verified["baseline"] == {"numerator": 1, "denominator": 3, "value": pytest.approx(1 / 3)}
    assert verified["candidate"] == {"numerator": 3, "denominator": 4, "value": 0.75}
    assert verified["delta"] == pytest.approx(0.75 - 1 / 3)
    assert metrics["strict_success_rate"]["delta"] == pytest.approx(0.5)
    assert metrics["evaluation_coverage"]["delta"] == pytest.approx(0.25)
    assert set(metrics) == {
        "verified_success_rate", "strict_success_rate", "evaluation_coverage",
        "unsafe_attempt_rate", "executed_violation_rate", "passed_all_repetitions",
    }


def test_an_undefined_rate_gives_an_undefined_delta() -> None:
    baseline = [run("a", "not_applicable", status="infrastructure_error", scores=())]
    candidate = [run("a", "pass")]
    metrics = compare(baseline, candidate, resamples=0)["metrics"]
    assert metrics["verified_success_rate"]["baseline"]["value"] is None
    assert metrics["verified_success_rate"]["delta"] is None
    assert metrics["strict_success_rate"]["delta"] == 1.0


def test_cost_delta_is_unknown_when_either_side_is_unknown() -> None:
    known = [run("a", "pass", attempts=(attempt(cost="0.10"),))]
    retried = [run("a", "pass", attempts=(attempt(1, cost="0.10", status="infrastructure_error"), attempt(2, cost="0.15")))]
    unknown = [run("a", "pass", attempts=(attempt(cost=None),))]
    assert compare(known, retried, resamples=0)["cost"]["delta_cost_usd"] == "0.15"
    assert compare(retried, known, resamples=0)["cost"]["delta_cost_usd"] == "-0.15"
    assert compare(known, unknown, resamples=0)["cost"]["delta_cost_usd"] is None
    assert compare(known, unknown, resamples=0)["cost"]["candidate"]["cost_usd"] is None
    assert compare(unknown, known, resamples=0)["cost"]["delta_cost_per_successful_run_usd"] is None


def test_latency_deltas_are_reported_separately_for_queue_and_execution() -> None:
    baseline = [run("a", "pass", queue_ms=100.0, attempts=(attempt(duration=1000.0),))]
    candidate = [run("a", "pass", queue_ms=40.0, attempts=(attempt(duration=1500.0),))]
    latency = compare(baseline, candidate, resamples=0)["latency_ms"]
    assert latency["queue"]["delta_mean"] == -60.0 and latency["queue"]["delta_median"] == -60.0
    assert latency["execution"]["delta_mean"] == 500.0


def test_tasks_present_on_one_side_only_are_reported_not_compared() -> None:
    result = compare(runs_for({"a": ["pass"], "b": ["pass"]}), runs_for({"a": ["fail"], "c": ["pass"]}), resamples=0)
    assert result["summary"]["tasks_compared"] == 1
    assert result["summary"]["baseline_only_tasks"] == ["b"] and result["summary"]["candidate_only_tasks"] == ["c"]


# --- bootstrap ---------------------------------------------------------------------------


def test_bootstrap_resamples_tasks_so_repetitions_stay_together() -> None:
    # Two tasks x 10 repetitions: one always passes, one always fails. Resampling tasks
    # can only give a strict rate of 0, 0.5 or 1, so the 95% interval is [0, 1].
    # Resampling the 20 runs independently would give a much narrower ~[0.3, 0.7].
    runs = runs_for({"always": ["pass"] * 10, "never": ["fail"] * 10})
    ci = compare(runs, runs, resamples=2000, seed=7)["metrics"]["strict_success_rate"]["baseline_ci"]
    assert ci["unit"] == "task" and ci["clusters"] == 2
    assert ci["low"] == 0.0 and ci["high"] == 1.0


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    baseline = runs_for({f"t{i}": ["pass" if i % 3 else "fail"] * 3 for i in range(12)})
    candidate = runs_for({f"t{i}": ["pass" if i % 4 else "fail"] * 3 for i in range(12)})
    first = compare(baseline, candidate, resamples=500, seed=42)
    again = compare(baseline, candidate, resamples=500, seed=42)
    other = compare(baseline, candidate, resamples=500, seed=43)
    assert first == again
    assert first["metrics"]["strict_success_rate"]["delta_ci"] != other["metrics"]["strict_success_rate"]["delta_ci"]
    delta = first["metrics"]["strict_success_rate"]
    assert delta["delta_ci"]["low"] <= delta["delta"] <= delta["delta_ci"]["high"]


def test_identical_per_task_deltas_give_a_degenerate_interval() -> None:
    baseline = runs_for({f"t{i}": ["fail", "fail"] for i in range(5)})
    candidate = runs_for({f"t{i}": ["pass", "fail"] for i in range(5)})
    ci = compare(baseline, candidate, resamples=300)["metrics"]["strict_success_rate"]["delta_ci"]
    assert ci["low"] == pytest.approx(0.5) and ci["high"] == pytest.approx(0.5)


def test_bootstrap_drops_undefined_resamples_and_handles_no_data() -> None:
    clusters = [(1, 0), (0, 0)]  # (successes, scored); a resample of only the second has no denominator
    ci = cluster_bootstrap(clusters, lambda s: Ratio(sum(c[0] for c in s), sum(c[0] for c in s)).value,
                           resamples=400, seed=1, confidence=0.9)
    assert 0 < ci["valid_resamples"] < 400
    empty = cluster_bootstrap([], lambda s: 1.0, resamples=100, seed=0, confidence=0.95)
    assert empty["low"] is None and empty["valid_resamples"] == 0
    assert cluster_bootstrap([1], lambda s: 1.0, resamples=0, seed=0, confidence=0.95)["low"] is None


# --- breakdowns --------------------------------------------------------------------------


def test_failure_reasons_and_hard_violations() -> None:
    response_fail = ScoreFacts("response", "1.2.0", "fail", 0.0, "FALSE_SUCCESS_CLAIM", "applicable", {})
    runs = [
        run("a", "fail", scores=(policy(), response_fail)),
        run("b", "fail", scores=(policy("fail", codes=("UNAPPROVED_REFUND_ATTEMPT",)), response_fail)),
        run("c", "fail", scores=(policy("fail", executed=True, codes=("OVER_REFUND",)),)),
    ]
    reasons = failure_reasons(runs)
    assert reasons[0] == {"evaluator": "response", "status": "fail", "reason_code": "FALSE_SUCCESS_CLAIM", "runs": 2}
    violations = hard_violations(runs)
    assert [v["task_id"] for v in violations] == ["c", "b"]  # executed first
    assert violations[0]["executed"] is True and violations[0]["codes"] == ["OVER_REFUND"]


def test_bootstrap_does_not_depend_on_run_order_or_mutate_global_rng() -> None:
    import random

    baseline = runs_for({"a": ["pass"] * 3, "b": ["fail", "pass"], "c": ["evaluation_error"]})
    candidate = runs_for({"a": ["fail"] * 3, "b": ["pass", "pass"], "c": ["pass"]})
    state = random.getstate()
    first = compare(baseline, candidate, resamples=127, seed=19)
    second = compare(list(reversed(baseline)), list(reversed(candidate)), resamples=127, seed=19)
    assert first == second
    assert random.getstate() == state


def test_baseline_error_prevents_regression_even_if_other_repetitions_pass() -> None:
    result = compare(runs_for({"t": ["pass", "evaluation_error"]}),
                     runs_for({"t": ["fail", "fail"]}), resamples=0)
    assert result["tasks"][0]["result"] == "loss"
    assert result["summary"]["regressed_tasks"] == []


def test_all_unscored_verified_bootstrap_is_null_but_strict_is_zero() -> None:
    runs = runs_for({"a": ["evaluation_error"] * 2, "b": ["pending"] * 2})
    result = compare(runs, runs, resamples=50)
    verified = result["metrics"]["verified_success_rate"]
    assert verified["delta"] is None
    assert verified["delta_ci"]["valid_resamples"] == 0
    assert verified["baseline_ci"]["low"] is None
    assert result["metrics"]["strict_success_rate"]["delta_ci"]["low"] == 0
