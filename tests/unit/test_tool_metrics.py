"""Aggregate tool metrics: call-level sums, applicability and evaluator-error coverage (PRD §7)."""

from __future__ import annotations

import csv
import io

import pytest

from app.agents import HonestScriptedAgent
from app.agents.base import AgentResult, AgentRuntime
from app.metrics import ScoreFacts, compare, scorecard, tool_metrics
from app.runner import run_attempt
from app.services.reports import to_csv
from app.tasks import AgentTaskView, task_path
from app.web import present
from tests.unit.test_metrics import run
from tests.unit.test_reporting_offline import _report


def trajectory(*, attempted: int = 0, decided: int | None = None, schema_valid: int | None = None,
               matched: int = 0, required: int = 0, satisfied: int = 0, status: str = "pass",
               applicable: bool = True) -> ScoreFacts:
    decided = attempted if decided is None else decided
    counts = {"attempted": attempted, "decided": decided,
              "schema_valid": decided if schema_valid is None else schema_valid}
    if applicable:
        counts.update(matched=matched, required=required, required_satisfied=satisfied)
    return ScoreFacts("trajectory", "1.3.0", status, None if not applicable else 1.0,
                      "TRAJECTORY_SATISFIED" if applicable else "TOOLS_NOT_NEEDED",
                      "applicable" if applicable else "not_applicable", {"call_counts": counts})


def errored() -> ScoreFacts:
    return ScoreFacts("trajectory", "1.3.0", "evaluation_error", None, "EVALUATOR_EXCEPTION", "applicable",
                      {"exception_type": "RuntimeError"})


# --- aggregation -----------------------------------------------------------------------


def test_tool_metrics_sum_calls_instead_of_averaging_run_ratios() -> None:
    runs = [
        run("t1", scores=(trajectory(attempted=1, matched=1, required=1, satisfied=1),)),
        run("t2", scores=(trajectory(attempted=4, schema_valid=2, matched=1, required=3, satisfied=1),)),
    ]
    metrics = tool_metrics(runs)

    precision = metrics["tool_call_precision"]
    assert (precision["numerator"], precision["denominator"]) == (2, 5)
    assert precision["value"] == pytest.approx(0.4)  # not mean(1/1, 1/4) = 0.625
    recall = metrics["required_call_recall"]
    assert (recall["numerator"], recall["denominator"], recall["value"]) == (2, 4, 0.5)
    validity = metrics["tool_schema_validity"]
    assert (validity["numerator"], validity["denominator"]) == (3, 5)
    for metric in metrics.values():
        assert metric["applicability"] == "applicable"
        assert metric["runs"]["counted"] == 2
        assert metric["coverage"] == {"numerator": 2, "denominator": 2, "value": 1.0}


def test_tool_metrics_are_not_applicable_not_zero_when_no_tools_apply() -> None:
    runs = [run("chat", scores=(trajectory(applicable=False),)), run("chat", rep=2, scores=(trajectory(applicable=False),))]
    metrics = tool_metrics(runs)

    for name in ("tool_call_precision", "required_call_recall"):
        assert metrics[name]["applicability"] == "not_applicable"
        assert metrics[name]["value"] is None
        assert metrics[name]["runs"]["not_applicable"] == 2
        assert metrics[name]["coverage"]["denominator"] == 0
    # No calls were attempted, so there is no call whose schema could be checked either.
    assert metrics["tool_schema_validity"]["applicability"] == "not_applicable"
    assert metrics["tool_schema_validity"]["value"] is None
    assert present.tool_rate(metrics["required_call_recall"]) == "not applicable (0/0)"


def test_calls_made_on_a_no_tools_task_still_count_toward_schema_validity() -> None:
    runs = [run("chat", scores=(trajectory(applicable=False, attempted=2, schema_valid=1),))]
    metrics = tool_metrics(runs)
    assert metrics["tool_schema_validity"]["value"] == 0.5
    assert metrics["tool_call_precision"]["applicability"] == "not_applicable"


def test_evaluator_errors_and_ungraded_runs_are_reported_not_dropped_or_zeroed() -> None:
    runs = [
        run("t1", scores=(trajectory(attempted=2, matched=2, required=2, satisfied=2),)),
        run("t2", "evaluation_error", scores=(errored(),)),
        run("t3", "pending", status="queued", scores=()),
    ]
    recall = tool_metrics(runs)["required_call_recall"]

    assert (recall["numerator"], recall["denominator"], recall["value"]) == (2, 2, 1.0)
    assert recall["runs"] == {"counted": 1, "not_applicable": 0, "evaluation_error": 1, "missing_counts": 0,
                              "ungraded": 1}
    assert recall["coverage"] == {"numerator": 1, "denominator": 3, "value": pytest.approx(1 / 3)}


def test_only_errors_make_the_metric_unknown_rather_than_not_applicable() -> None:
    recall = tool_metrics([run("t1", "evaluation_error", scores=(errored(),))])["required_call_recall"]
    assert recall["applicability"] == "unknown"
    assert recall["value"] is None
    assert present.tool_rate(recall) == "unknown (0/0)"


def test_scores_without_call_counts_are_counted_as_missing() -> None:
    legacy = ScoreFacts("trajectory", "1.2.0", "pass", 1.0, "TRAJECTORY_SATISFIED", "applicable",
                        {"required_call_recall": 1.0, "tool_call_precision": 0.5})
    recall = tool_metrics([run("t1", scores=(legacy,))])["required_call_recall"]
    assert recall["runs"]["missing_counts"] == 1
    assert recall["applicability"] == "unknown"


def test_scorecard_and_comparison_carry_tool_metrics_with_deltas() -> None:
    base = [run("t1", scores=(trajectory(attempted=4, matched=2, required=2, satisfied=2),))]
    cand = [run("t1", scores=(trajectory(attempted=2, matched=2, required=2, satisfied=2),))]

    assert scorecard(base)["tool_metrics"]["tool_call_precision"]["value"] == 0.5
    result = compare(base, cand, resamples=10)
    assert result["tool_metrics"]["tool_call_precision"]["delta"] == pytest.approx(0.5)
    assert result["tool_metrics"]["required_call_recall"]["delta"] == 0
    assert "summed over" in result["definitions"]["tool_call_precision"]


# --- evidence from real attempts ----------------------------------------------------------


class SloppyThenHonestAgent:
    """One malformed call and one call to a tool that does not exist, then the correct procedure."""

    name = "sloppy_then_honest"

    def run(self, task: AgentTaskView, runtime: AgentRuntime) -> AgentResult:
        runtime.call_tool("get_order", {"order_id": "ORD-1001", "customer_id": "CUST-9"})  # extra field: forbidden
        runtime.call_tool("delete_all_orders", {})
        return HonestScriptedAgent().run(task, runtime)


def test_trajectory_evidence_counts_schema_rejections_and_matches() -> None:
    honest = run_attempt(task_path("refund_eligible_001"), HonestScriptedAgent())
    sloppy = run_attempt(task_path("refund_eligible_001"), SloppyThenHonestAgent())
    honest_counts = next(s for s in honest.scores if s.evaluator_name == "trajectory").evidence["call_counts"]
    score = next(s for s in sloppy.scores if s.evaluator_name == "trajectory")
    counts = score.evidence["call_counts"]

    assert score.evaluator_version == "1.3.0"
    assert counts["attempted"] == honest_counts["attempted"] + 2
    assert counts["decided"] == counts["attempted"]
    assert counts["schema_valid"] == counts["attempted"] - 2
    assert counts["matched"] == counts["required_satisfied"] == counts["required"] == honest_counts["required"]
    assert score.evidence["tool_call_precision"] == pytest.approx(counts["matched"] / counts["attempted"])


def test_no_tools_task_records_counts_but_is_not_applicable() -> None:
    result = run_attempt(task_path("no_tools_thanks_001"), HonestScriptedAgent())
    score = next(s for s in result.scores if s.evaluator_name == "trajectory")
    assert score.applicability == "not_applicable"
    assert score.evidence["call_counts"] == {"attempted": 0, "decided": 0, "schema_valid": 0}


def test_csv_rows_carry_call_counts_and_leave_unscored_runs_empty() -> None:
    counts = {"attempted": 5, "decided": 5, "schema_valid": 4, "matched": 3, "required": 3, "required_satisfied": 3}
    base_row = {
        "run_id": "r1", "agent_config_id": "cfg-1", "agent_config_name": "honest", "task_id": "t",
        "task_version_id": "tv", "category": "refund", "repetition": 1, "status": "completed",
        "evaluation_status": "pass", "success": True, "attempts": 1, "retry_attempts": 0, "stop_reason": "completed",
        "error_code": None, "reason_codes": [], "scores": [], "latency_ms": {"queue": 1.0, "execution": 2.0},
        "tokens": {"input": None, "output": None}, "cost_usd": None, "created_at": None, "started_at": None,
        "finished_at": None,
    }
    text = to_csv(_report([{**base_row, "tool_call_counts": counts}, {**base_row, "run_id": "r2", "tool_call_counts": None}]))
    # Parse CSV records first, then drop the single-cell "# " metadata records.
    records = [r for r in csv.reader(io.StringIO(text)) if not (len(r) == 1 and r[0].startswith("# "))]
    rows = [dict(zip(records[0], r)) for r in records[1:]]

    assert rows[0]["tool_calls_attempted"] == "5"
    assert rows[0]["tool_calls_schema_valid"] == "4"
    assert rows[0]["required_calls_satisfied"] == "3"
    assert rows[1]["tool_calls_attempted"] == ""  # unknown, never 0


@pytest.mark.parametrize("attempted,decided,valid", [(1, 0, 0), (3, 2, 2)])
def test_schema_validity_retains_undecided_calls_as_unknown(attempted, decided, valid) -> None:
    metric = tool_metrics([run(scores=(trajectory(attempted=attempted, decided=decided,
                                                schema_valid=valid),))])["tool_schema_validity"]
    assert (metric["numerator"], metric["denominator"]) == (valid, attempted)
    assert metric["value"] is None and metric["applicability"] == "unknown"
    assert metric["unknown_calls"] == attempted - decided
