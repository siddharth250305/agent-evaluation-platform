"""GET /experiments/{id}/export: per-run results bundled with the immutable manifest, as JSON and CSV."""

from __future__ import annotations

import csv
import io
import json
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Attempt, Experiment
from app.services.hashing import sha256_json
from tests.postgres.helpers import client_for, launch, worker

FAULT_TASKS = {
    "approval_service_unavailable_001", "approval_service_unavailable_002",
    "order_service_unavailable_001", "order_service_unavailable_002",
    "refund_service_timeout_001", "refund_service_timeout_002",
}


@pytest.fixture
def finished(sm: sessionmaker[Session]):
    """Honest and lying configurations on the dev split (which holds every fault-injected task), fully run."""
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted"), ("lying", "lying_broken")], split="dev")
    worker(sm).run_until_idle()
    return client, experiment


def _csv(text: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    records = list(csv.reader(io.StringIO(text)))
    comments = {}
    body = []
    for row in records:
        if len(row) == 1 and row[0].startswith("# "):
            key, _, value = row[0][2:].partition(": ")
            comments[key] = value
        else:
            body.append(row)
    return comments, [dict(zip(body[0], row)) for row in body[1:]]



def test_json_export_bundles_the_manifest_with_per_run_results(finished) -> None:
    client, experiment = finished
    response = client.get(f"/experiments/{experiment['id']}/export", params={"format": "json"})

    assert response.status_code == 200
    assert response.headers["content-disposition"] == f'attachment; filename="experiment-{experiment["id"]}.json"'
    report = response.json()
    manifest = report["manifest"]
    assert manifest == client.get(f"/experiments/{experiment['id']}").json()["manifest"]
    assert report["manifest_sha256"] == sha256_json(manifest)

    # Everything PRD §10 asks a manifest to pin.
    assert manifest["dataset"]["content_hash"] and manifest["environment"]["version"]
    assert {e["name"] for e in manifest["evaluators"]} == {"outcome", "policy", "trajectory", "response"}
    assert all(len(c["tool_schema_hash"]) == 64 and "prompt_hash" in c and "settings" in c and "limits" in c
               for c in manifest["agent_configs"])
    assert {"git_commit", "lockfile_sha256", "pyproject_sha256"} <= set(manifest["code"])
    assert set(manifest["fault_config"]) == FAULT_TASKS
    assert manifest["fault_config"]["refund_service_timeout_001"] == {"issue_refund:ORD-1001": "timeout"}
    assert "fault_config" not in report["manifest_supplement"]  # already in the manifest
    assert report["manifest_supplement"]["resolved_models"] == {}  # scripted agents call no model

    planned = experiment["progress"]["planned_runs"]
    assert len(report["runs"]) == planned
    assert len({r["run_id"] for r in report["runs"]}) == planned
    lying_refund = next(r for r in report["runs"]  # the convincing-but-wrong demo, on a dev-split task
                        if r["agent_config_name"] == "lying" and "response:FALSE_SUCCESS_CLAIM" in r["reason_codes"])
    assert lying_refund["success"] is False and lying_refund["evaluation_status"] == "fail"
    assert "response:FALSE_SUCCESS_CLAIM" in lying_refund["reason_codes"]
    assert {s["evaluator_name"] for s in lying_refund["scores"]} == {"outcome", "policy", "trajectory", "response"}
    assert all("evidence" in s and "evaluator_version" in s for s in lying_refund["scores"])
    assert lying_refund["latency_ms"]["queue"] is not None and lying_refund["latency_ms"]["execution"] is not None
    assert lying_refund["cost_usd"] == "0" and lying_refund["attempts"] == 1

    cards = {c["agent_config_name"]: c["scorecard"] for c in report["scorecards"]}
    assert cards["lying"]["verified_success_rate"]["numerator"] == 0
    assert cards["honest"]["strict_success_rate"]["denominator"] == planned // 2


def test_csv_export_has_one_row_per_run_and_the_manifest(finished) -> None:
    client, experiment = finished
    json_report = client.get(f"/experiments/{experiment['id']}/export").json()
    response = client.get(f"/experiments/{experiment['id']}/export", params={"format": "csv"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["content-disposition"].endswith(f'experiment-{experiment["id"]}.csv"')
    comments, rows = _csv(response.text)
    assert json.loads(comments["manifest"]) == json_report["manifest"]
    assert comments["manifest_sha256"] == json_report["manifest_sha256"]
    assert len(rows) == experiment["progress"]["planned_runs"]
    assert {r["run_id"] for r in rows} == {r["run_id"] for r in json_report["runs"]}
    assert all(r["manifest_sha256"] == json_report["manifest_sha256"] for r in rows)
    lying_refund = next(r for r in rows
                        if r["agent_config_name"] == "lying" and r["response_reason_code"] == "FALSE_SUCCESS_CLAIM")
    assert lying_refund["response_status"] == "fail" and lying_refund["response_reason_code"] == "FALSE_SUCCESS_CLAIM"
    assert lying_refund["success"] == "false" and lying_refund["cost_usd"] == "0"
    assert lying_refund["split"] == "dev" and lying_refund["dataset_content_hash"]
    assert {"queue_ms", "execution_ms", "input_tokens", "output_tokens", "tool_schema_hash", "prompt_hash",
            "git_commit", "lockfile_sha256", "evaluator_versions"} <= set(lying_refund)


def test_unknown_cost_is_exported_as_null_never_zero(sm, finished) -> None:
    client, experiment = finished
    with sm.begin() as session:
        attempt = session.scalars(select(Attempt).limit(1)).one()
        attempt.cost_usd = None
        attempt.input_tokens = None
        run_id = str(attempt.run_id)

    report = client.get(f"/experiments/{experiment['id']}/export").json()
    run = next(r for r in report["runs"] if r["run_id"] == run_id)
    assert run["cost_usd"] is None and run["tokens"]["input"] is None
    affected = next(c["scorecard"] for c in report["scorecards"] if c["agent_config_id"] == run["agent_config_id"])
    assert affected["usage"]["cost_usd"] is None and affected["usage"]["attempts_with_unknown_cost"] == 1

    _, rows = _csv(client.get(f"/experiments/{experiment['id']}/export", params={"format": "csv"}).text)
    row = next(r for r in rows if r["run_id"] == run_id)
    assert row["cost_usd"] == "" and row["input_tokens"] == ""


def test_manifests_from_before_phase_5_get_their_fault_config_from_task_versions(sm) -> None:
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted")], split="dev")
    with sm.begin() as session:
        stored = session.get(Experiment, uuid.UUID(experiment["id"]))
        original = stored.manifest["fault_config"]
        stored.manifest = {k: v for k, v in stored.manifest.items() if k != "fault_config"}

    report = client.get(f"/experiments/{experiment['id']}/export").json()

    assert "fault_config" not in report["manifest"]
    assert report["manifest_supplement"]["fault_config"] == original
    assert report["manifest_sha256"] == sha256_json(report["manifest"])  # the stored manifest, untouched


def test_an_unfinished_experiment_exports_every_planned_run(sm) -> None:
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted")])
    report = client.get(f"/experiments/{experiment['id']}/export").json()
    assert len(report["runs"]) == experiment["progress"]["planned_runs"]
    assert all(r["status"] == "queued" and r["cost_usd"] == "0" and r["attempts"] == 0 for r in report["runs"])
    assert report["scorecards"][0]["scorecard"]["verified_success_rate"]["value"] is None


def test_export_errors(sm) -> None:
    client = client_for(sm)
    assert client.get(f"/experiments/{uuid.uuid4()}/export").status_code == 404
    experiment = launch(client, [("honest", "honest_scripted")])
    assert client.get(f"/experiments/{experiment['id']}/export", params={"format": "xml"}).status_code == 422


def test_no_tool_scores_remain_not_applicable_in_json_and_csv(finished) -> None:
    client, experiment = finished
    report = client.get(f"/experiments/{experiment['id']}/export").json()
    no_tools = {r["run_id"] for r in report["runs"] for s in r["scores"]
                if s["evaluator_name"] == "trajectory" and s["status"] == "not_applicable"}
    assert no_tools
    for run in report["runs"]:
        if run["run_id"] in no_tools:
            score = next(s for s in run["scores"] if s["evaluator_name"] == "trajectory")
            assert score["value"] is None and score["applicability"] == "not_applicable"
    _, rows = _csv(client.get(f"/experiments/{experiment['id']}/export", params={"format": "csv"}).text)
    for row in rows:
        if row["run_id"] in no_tools:
            assert row["trajectory_status"] == "not_applicable" and row["trajectory_value"] == ""
