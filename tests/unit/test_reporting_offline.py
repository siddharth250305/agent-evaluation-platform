"""Export CSV format, dashboard helpers and template rendering, without a database."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.metrics import compare
from app.services.reports import to_csv
from app.web import present
from app.web.routes import templates
from tests.unit.test_metrics import attempt, run, runs_for

MANIFEST = {
    "schema_version": 1,
    "dataset": {"name": "support", "content_hash": "d" * 64},
    "environment": {"version": "1.0.0"},
    "evaluators": [{"name": "outcome", "version": "1.0.0"}, {"name": "response", "version": "1.2.0"}],
    "agent_configs": [{"id": "cfg-1", "name": "honest", "model": None, "config_hash": "c" * 64,
                       "prompt_hash": None, "tool_schema_hash": "t" * 64, "limits": {}, "settings": {}}],
    "fault_config": {"refund_service_timeout_001": {"issue_refund:ORD-1001": "timeout"}},
    "code": {"git_commit": None, "lockfile_sha256": None},
}


def _report(runs: list[dict]) -> dict:
    return {
        "export_version": 1,
        "generated_at": "2026-11-21T00:00:00+00:00",
        "experiment": {"id": "exp-1", "split": "validation"},
        "manifest": MANIFEST,
        "manifest_sha256": "m" * 64,
        "manifest_supplement": {"resolved_models": {}},
        "runs": runs,
    }


def _run_row(**overrides) -> dict:
    row = {
        "run_id": "run-1", "agent_config_id": "cfg-1", "agent_config_name": "honest", "task_id": "refund_eligible_001",
        "task_version_id": "tv-1", "category": "refund", "repetition": 1, "status": "completed",
        "evaluation_status": "fail", "success": False, "attempts": 2, "retry_attempts": 1, "stop_reason": "completed",
        "error_code": None, "reason_codes": ["response:FALSE_SUCCESS_CLAIM"],
        "scores": [
            {"evaluator_name": "outcome", "status": "fail", "reason_code": "NO_REFUND_ISSUED", "value": 0.0},
            {"evaluator_name": "response", "status": "fail", "reason_code": "FALSE_SUCCESS_CLAIM", "value": 0.0},
        ],
        "latency_ms": {"queue": 12.5, "execution": 40.0}, "tokens": {"input": None, "output": None},
        "cost_usd": None, "created_at": "2026-11-21T00:00:00+00:00", "started_at": None, "finished_at": None,
    }
    return {**row, **overrides}


def _parse(text: str) -> tuple[list[str], list[dict[str, str]]]:
    records = list(csv.reader(io.StringIO(text)))
    comments = [row[0][2:] for row in records if len(row) == 1 and row[0].startswith("# ")]
    body = [row for row in records if not (len(row) == 1 and row[0].startswith("# "))]
    return comments, [dict(zip(body[0], row)) for row in body[1:]]



# --- CSV -------------------------------------------------------------------------------


def test_csv_carries_the_full_manifest_and_one_row_per_run() -> None:
    comments, rows = _parse(to_csv(_report([_run_row(), _run_row(run_id="run-2", repetition=2)])))
    manifest_line = next(c for c in comments if c.startswith("manifest: "))
    assert json.loads(manifest_line.removeprefix("manifest: ")) == MANIFEST
    assert f"manifest_sha256: {'m' * 64}" in comments
    assert [r["run_id"] for r in rows] == ["run-1", "run-2"]


def test_csv_rows_repeat_identifying_hashes_and_flatten_scores() -> None:
    _, [row] = _parse(to_csv(_report([_run_row()])))
    assert row["manifest_sha256"] == "m" * 64
    assert row["dataset_content_hash"] == "d" * 64
    assert row["evaluator_versions"] == "outcome@1.0.0;response@1.2.0"
    assert row["agent_config_hash"] == "c" * 64 and row["tool_schema_hash"] == "t" * 64
    assert row["response_status"] == "fail" and row["response_reason_code"] == "FALSE_SUCCESS_CLAIM"
    assert row["outcome_reason_code"] == "NO_REFUND_ISSUED"
    assert row["reason_codes"] == "response:FALSE_SUCCESS_CLAIM"
    assert row["success"] == "false" and row["retry_attempts"] == "1" and row["split"] == "validation"


def test_csv_unknown_cost_and_tokens_are_empty_never_zero() -> None:
    _, [unknown, known] = _parse(to_csv(_report([_run_row(), _run_row(run_id="run-2", cost_usd="0", tokens={"input": 0, "output": 0})])))
    assert unknown["cost_usd"] == "" and unknown["input_tokens"] == ""
    assert known["cost_usd"] == "0" and known["input_tokens"] == "0"


def test_csv_neutralises_spreadsheet_formulas() -> None:
    _, [row] = _parse(to_csv(_report([_run_row(task_id="=HYPERLINK(\"http://x\")")])))
    assert row["task_id"].startswith("'=")


# --- presentation helpers ----------------------------------------------------------------


def test_state_diff_reports_leaf_changes_with_paths() -> None:
    before = {"refunds": {}, "tickets": {"T-1": {"status": "open", "note": None}}, "orders": {"O": {"total": "1"}}}
    after = {"refunds": {"R-1": {"amount": "149.00"}}, "tickets": {"T-1": {"status": "resolved", "note": None}},
             "orders": {"O": {"total": "1"}}}
    diff = present.state_diff(before, after)
    assert diff == [
        {"path": "refunds.R-1", "change": "added", "before": None, "after": {"amount": "149.00"}},
        {"path": "tickets.T-1.status", "change": "changed", "before": "open", "after": "resolved"},
    ]
    assert present.state_diff(after, before)[0]["change"] == "removed"
    assert present.state_diff(before, before) == []


def test_evidence_sequences_find_cited_trace_events_at_any_depth() -> None:
    policy = {"violations": [{"code": "OVER_REFUND", "attempt_sequence": 7}, {"attempt_sequence": None}],
              "blocked_read_attempts": [3]}
    response = {"final_response_sequence": 12, "proof": {"issue_refund_events": [{"sequence": 9, "kind": "tool_attempt"}],
                                                         "final_response_sequence": 12}}
    trajectory = {"wrong_arguments": {"attempted": [{"sequence": 4, "args": {}}]}, "required_call_recall": 0.5}
    assert present.evidence_sequences(policy) == {7}
    assert present.evidence_sequences(response) == {9, 12}
    assert present.evidence_sequences(trajectory) == {4}
    assert present.evidence_sequences({"flag_sequence": True}) == set()


def test_filters_never_show_unknown_as_zero() -> None:
    assert present.pct({"numerator": 0, "denominator": 0, "value": None}) == "n/a (0/0)"
    assert present.pct({"numerator": 5, "denominator": 8, "value": 0.625}) == "62.5% (5/8)"
    assert present.usd(None) == "unknown"
    assert present.usd("0.0024") == "$0.0024"
    assert present.usd(Decimal("0E-8")) == "$0"
    assert present.usd_delta("-0.15") == "-$0.15" and present.usd_delta("0.15") == "+$0.15"
    assert present.pp(0.125) == "+12.5 pp" and present.pp(None) == "n/a"
    assert present.ms(None) == "unknown" and present.ms(1500.0) == "1.50 s" and present.ms_delta(-40.0) == "-40 ms"


def test_timestamps_are_shown_in_utc() -> None:
    local = datetime(2026, 11, 21, 18, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert present.when(local) == "2026-11-21 13:00:00 UTC"
    assert present.when("2026-11-21T13:00:00+00:00") == "2026-11-21 13:00:00 UTC"
    assert present.when(None) == "-"


# --- templates render from plain data ---------------------------------------------------


def _render(name: str, **context) -> str:
    return templates.env.get_template(name).render(**context)


def test_compare_template_renders_regressions_deltas_and_definitions() -> None:
    baseline = runs_for({"kept": ["pass"], "broke": ["pass"], "fixed": ["fail"]})
    candidate = runs_for({"kept": ["pass"], "broke": ["fail"], "fixed": ["pass"]})
    candidate[0] = run("kept", "pass", attempts=(attempt(cost=None),))
    result = compare(baseline, candidate, resamples=100)
    side = {"experiment_id": "e", "experiment_name": "exp", "agent_config_name": "cfg", "adapter": "scripted",
            "model": None, "prompt_hash": None, "repetitions": 1, "planned_runs": 3, "complete": True,
            "unfinished_runs": 0}
    result.update(baseline=side, candidate=side, warnings=[], config_differences={"adapter": {}})
    html = _render("compare.html", options=[], baseline="", candidate="", result=result, error=None, active="compare")
    assert "Regressed" in html and "broke" in html
    assert result["definitions"]["strict_success_rate"] in html
    assert "unknown" in html  # candidate cost is unknown, not $0
    assert 'data-kind="all loss regression"' in html


def test_benchmarks_template_has_an_empty_state_and_separation_notice() -> None:
    html = _render("benchmarks.html", results=[], active="benchmarks")
    assert 'id="no-benchmarks"' in html and "separately" in html


def test_templates_escape_untrusted_text() -> None:
    html = _render("error.html", title="Not found", message="<script>alert(1)</script>", active=None)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


@pytest.mark.parametrize("name", ["experiments.html", "experiment.html", "run.html", "compare.html",
                                  "benchmarks.html", "error.html"])
def test_every_template_compiles(name: str) -> None:
    templates.env.get_template(name)


@pytest.mark.parametrize("payload", ["=1+1", "+1+1", "-1+1", "@SUM(1)", "\t=1", "\r=1", "\n=1", "  =1", " \t@SUM(1)"])
def test_csv_protects_formula_prefixes_and_preserves_multiline_cells(payload) -> None:
    _, [row] = _parse(to_csv(_report([_run_row(agent_config_name=payload)])))
    assert row["agent_config_name"] == "'" + payload


def test_csv_metadata_is_one_cell_and_round_trips_hostile_manifest() -> None:
    from copy import deepcopy

    report = deepcopy(_report([_run_row()]))
    payload = 'text,=HYPERLINK("https://example.invalid"),\r\n@SUM(1)'
    report["manifest"]["agent_configs"][0]["settings"] = {"=1": payload}
    report["manifest_supplement"] = {"resolved_models": {"cfg-1": [payload]}}
    exported = to_csv(report)
    records = list(csv.reader(io.StringIO(exported)))
    # Spreadsheet import must never split metadata into unprotected cells.
    assert all(len(record) == 1 for record in records[:7])
    comments, _ = _parse(exported)
    stored = next(c.removeprefix("manifest: ") for c in comments if c.startswith("manifest: "))
    assert json.loads(stored) == report["manifest"]


def test_csv_not_applicable_and_evaluation_error_values_stay_empty() -> None:
    row = _run_row(scores=[
        {"evaluator_name": "outcome", "status": "evaluation_error", "reason_code": "EVALUATOR_EXCEPTION", "value": None},
        {"evaluator_name": "response", "status": "not_applicable", "reason_code": "NOT_REQUIRED", "value": None},
    ])
    _, [parsed] = _parse(to_csv(_report([row])))
    assert parsed["outcome_status"] == "evaluation_error" and parsed["outcome_value"] == ""
    assert parsed["response_status"] == "not_applicable" and parsed["response_value"] == ""
