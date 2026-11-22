"""The five dashboard screens, rendered against a real database. Scripted agents only."""

from __future__ import annotations

import re
import uuid

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import BenchmarkResult
from tests.postgres.helpers import client_for, launch, worker


@pytest.fixture
def finished(sm: sessionmaker[Session]):
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted"), ("lying", "lying_broken")])
    worker(sm).run_until_idle()
    ids = {c["name"]: c["id"] for c in experiment["manifest"]["agent_configs"]}
    runs = client.get(f"/experiments/{experiment['id']}/runs", params={"limit": 500}).json()["items"]
    return client, experiment, ids, runs


def _section(html: str, start: str, end: str) -> str:
    return html[html.index(start):html.index(end, html.index(start))]


def _table(html: str, table_id: str) -> str:
    return _section(html, f'id="{table_id}"', "</table>")


def test_root_redirects_to_the_experiment_list(sm) -> None:
    response = client_for(sm).get("/", follow_redirects=False)
    assert response.status_code == 307 and response.headers["location"] == "/ui/experiments"


# --- 1. experiment list ----------------------------------------------------------------


def test_experiment_list_shows_configs_dataset_progress_rates_spend_and_start(finished) -> None:
    client, experiment, _, runs = finished
    response = client.get("/ui/experiments")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    row = _table(response.text, "experiments")
    planned = len(runs)
    scored = sum(r["evaluation_status"] in ("pass", "fail") for r in runs)
    passed = sum(r["evaluation_status"] == "pass" for r in runs)
    assert f'/ui/experiments/{experiment["id"]}' in row
    assert "honest" in row and "lying" in row and "support" in row
    assert f"100.0% ({planned}/{planned})" in row  # progress
    assert f"({passed}/{scored})" in row  # verified success with its denominator
    assert "$0" in row and "not started" not in row


def test_empty_experiment_list(sm) -> None:
    assert "No experiments yet" in client_for(sm).get("/ui/experiments").text


# --- 2. experiment detail --------------------------------------------------------------


def test_experiment_detail_shows_scorecard_categories_failure_reasons_and_runs(finished) -> None:
    client, experiment, ids, runs = finished
    html = client.get(f"/ui/experiments/{experiment['id']}").text

    scorecard = _table(html, "scorecard")
    assert "Verified success" in scorecard and "Executed violation rate" in scorecard and "Cost per successful run" in scorecard
    assert "FALSE_SUCCESS_CLAIM" in _table(html, "reasons")
    assert "Hard violations" in html
    for category in {r["category"] for r in runs}:
        assert category in html
    assert html.count('href="/ui/runs/') >= len(runs)
    assert f'/experiments/{experiment["id"]}/export?format=csv' in html
    assert f'/ui/compare?baseline={experiment["id"]}:{ids["honest"]}&amp;candidate={experiment["id"]}:{ids["lying"]}' in html


def test_experiment_detail_run_list_is_filterable(finished) -> None:
    client, experiment, ids, runs = finished
    lying_only = _table(client.get(f"/ui/experiments/{experiment['id']}", params={"config": ids["lying"]}).text, "runs")
    assert "<td>lying</td>" in lying_only and "<td>honest</td>" not in lying_only

    failed = client.get(f"/ui/experiments/{experiment['id']}", params={"success": "false"}).text
    expected = [r["id"] for r in runs if r["success"] is False]
    assert sorted(re.findall(r'href="/ui/runs/([0-9a-f-]+)"', _table(failed, "runs"))) == sorted(expected)

    by_reason = client.get(f"/ui/experiments/{experiment['id']}", params={"q": "response:FALSE_SUCCESS_CLAIM"}).text
    assert "<td>honest</td>" not in _table(by_reason, "runs")
    nothing = client.get(f"/ui/experiments/{experiment['id']}", params={"category": "no-such-category"}).text
    assert "No runs match these filters." in nothing


def test_unknown_experiment_page_is_404(sm) -> None:
    response = client_for(sm).get(f"/ui/experiments/{uuid.uuid4()}")
    assert response.status_code == 404 and "not found" in response.text


# --- 3. run detail ---------------------------------------------------------------------


def _lying_refund(runs: list[dict]) -> dict:
    return next(r for r in runs if r["task_id"] == "refund_eligible_001" and r["agent_config_name"] == "lying")


def test_run_detail_separates_agent_visible_input_from_hidden_ground_truth(finished) -> None:
    client, _, _, runs = finished
    html = client.get(f"/ui/runs/{_lying_refund(runs)['id']}").text

    visible = _section(html, 'class="panel visible"', 'class="panel hidden-truth"')
    hidden = _section(html, 'class="panel hidden-truth"', "</section>")
    assert "AGENT-VISIBLE" in visible and "ORD-1001" in visible and "issue_refund" in visible
    # Ground truth never appears in the agent-visible panel.
    for secret in ("refund_issued", "refund_amount", "required_calls", "CUST-001"):
        assert secret not in visible
        assert secret in hidden
    assert "HIDDEN FROM THE AGENT" in hidden


def test_run_detail_shows_findings_state_diff_and_flags_failing_trace_events(finished) -> None:
    client, _, _, runs = finished
    run = _lying_refund(runs)
    detail = client.get(f"/runs/{run['id']}").json()
    response_score = next(s for s in detail["scores"] if s["evaluator_name"] == "response")
    final_sequence = response_score["evidence"]["final_response_sequence"]

    html = client.get(f"/ui/runs/{run['id']}").text

    assert "FALSE_SUCCESS_CLAIM" in _table(html, "scores")
    assert "has been processed successfully" in html  # the convincing message is shown as evidence
    diff = _table(html, "state-diff")
    assert "tickets." in diff and "resolved" in diff and "refunds." not in diff  # no refund was ever created
    event = re.search(rf'<li class="([^"]*)" data-kind="[^"]*" id="event-{final_sequence}">', html)
    assert event and "flagged" in event.group(1).split()
    assert "response: FALSE_SUCCESS_CLAIM" in html[event.start():html.index("</li>", event.start())]
    events = client.get(f"/attempts/{detail['attempts'][0]['id']}/events").json()["items"]
    assert html.count('<li class="') == len(events)
    sequences = [int(s) for s in re.findall(r'id="event-(\d+)"', html)]
    assert sequences == sorted(sequences) == [e["sequence"] for e in events]


def test_run_detail_attempt_selection_and_missing_runs(finished) -> None:
    client, _, _, runs = finished
    run_id = _lying_refund(runs)["id"]
    assert client.get(f"/ui/runs/{run_id}", params={"attempt": 1}).status_code == 200
    assert client.get(f"/ui/runs/{run_id}", params={"attempt": 9}).status_code == 404
    assert client.get(f"/ui/runs/{uuid.uuid4()}").status_code == 404


def test_a_queued_run_renders_without_attempts(sm) -> None:
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted")])
    run = client.get(f"/experiments/{experiment['id']}/runs").json()["items"][0]
    html = client.get(f"/ui/runs/{run['id']}").text
    assert "No attempt has started yet." in html and "AGENT-VISIBLE" in html


# --- 4. comparison ---------------------------------------------------------------------


def test_comparison_view_lists_wins_losses_regressions_and_deltas(finished) -> None:
    client, experiment, ids, _ = finished
    baseline, candidate = f"{experiment['id']}:{ids['honest']}", f"{experiment['id']}:{ids['lying']}"
    api = client.post("/comparisons", json={
        "baseline": {"experiment_id": experiment["id"], "agent_config_id": ids["honest"]},
        "candidate": {"experiment_id": experiment["id"], "agent_config_id": ids["lying"]},
    }).json()

    response = client.get("/ui/compare", params={"baseline": baseline, "candidate": candidate})

    assert response.status_code == 200
    html = response.text
    summary = _section(html, 'id="summary"', "<h2>")
    assert f'<div class="label">Regressions</div><div class="value">{api["summary"]["regressions"]}</div>' in summary
    assert f'<div class="label">Losses</div><div class="value">{api["summary"]["losses"]}</div>' in summary
    for task_id in api["summary"]["regressed_tasks"]:
        assert task_id in _section(html, "Regressed", "</div>")
    metrics = _table(html, "metrics")
    strict = api["metrics"]["strict_success_rate"]
    assert f'({strict["baseline"]["numerator"]}/{strict["baseline"]["denominator"]})' in metrics
    assert f'{strict["delta"] * 100:+.1f} pp' in metrics
    assert "Queue" in _table(html, "latency") and "retries included" in _table(html, "cost")
    assert html.count("data-kind=\"all ") == api["summary"]["tasks_compared"]


def test_comparison_view_form_and_errors(finished) -> None:
    client, experiment, ids, _ = finished
    empty = client.get("/ui/compare")
    assert empty.status_code == 200 and f'value="{experiment["id"]}:{ids["lying"]}"' in empty.text
    same = f"{experiment['id']}:{ids['honest']}"
    error = client.get("/ui/compare", params={"baseline": same, "candidate": same})
    assert error.status_code == 422 and "same configuration" in error.text
    garbage = client.get("/ui/compare", params={"baseline": "nope", "candidate": "nope"})
    assert garbage.status_code == 422 and "Pick a baseline" in garbage.text


# --- 5. benchmarks ---------------------------------------------------------------------


def test_benchmarks_view_is_separate_from_custom_scores(sm, finished) -> None:
    client, experiment, _, _ = finished
    assert 'id="no-benchmarks"' in client.get("/ui/benchmarks").text
    with sm.begin() as session:
        session.add(BenchmarkResult(
            benchmark="bfcl", benchmark_version="v3-pinned-abc123", category="simple", model="claude-sonnet-5-5",
            task_count=400, scoring_method="AST match", metrics={"accuracy": 0.9}, raw_artifact_uri="artifacts/bfcl.jsonl",
            manifest={"commit": "abc123"},
        ))

    html = client.get("/ui/benchmarks").text
    table = _table(html, "benchmarks")
    assert "bfcl" in table and "v3-pinned-abc123" in table and "AST match" in table and "400" in table
    for page in ("/ui/experiments", f"/ui/experiments/{experiment['id']}"):
        assert "bfcl" not in client.get(page).text  # never mixed into custom support scores


# --- plumbing --------------------------------------------------------------------------


def test_static_assets_are_served_and_dashboard_routes_stay_out_of_the_api_schema(sm) -> None:
    client = client_for(sm)
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert "/comparisons" in paths and "/experiments/{experiment_id}/export" in paths
    assert not any(p.startswith("/ui") for p in paths)


def test_hidden_markup_cannot_escape_into_agent_visible_panel(sm, finished) -> None:
    from sqlalchemy import select
    from app.db.models import Run, TaskVersion

    client, _, _, runs = finished
    run_id = _lying_refund(runs)["id"]
    sentinel = '</pre></section><script>hidden_ground_truth_marker()</script>'
    with sm.begin() as session:
        task = session.scalar(select(TaskVersion).join(Run, Run.task_version_id == TaskVersion.id)
                              .where(Run.id == uuid.UUID(run_id)))
        task.evaluator_view = {**task.evaluator_view, "expected": {"review_sentinel": sentinel}}
    response = client.get(f"/ui/runs/{run_id}")
    assert response.status_code == 200
    visible = _section(response.text, 'class="panel visible"', 'class="panel hidden-truth"')
    assert "hidden_ground_truth_marker" not in visible
    assert "hidden_ground_truth_marker" in response.text
    assert "<script>hidden_ground_truth_marker" not in response.text
    assert set(response.context["agent_visible"]) == {"user_request", "system_prompt", "tools"}
