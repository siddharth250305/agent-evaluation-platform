"""POST /experiments: one transaction, idempotent, plus the read and cancel endpoints."""

from __future__ import annotations

import threading
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Experiment, Job, Run
from app.services import experiments as experiment_service
from app.services.experiments import ExperimentCreate, create_experiment
from tests.postgres.helpers import client_for, experiment_payload, import_support, launch, register, worker


def _count(sm: sessionmaker[Session], model: type) -> int:
    with sm() as session:
        return session.scalar(select(func.count()).select_from(model))


@pytest.fixture
def setup(sm: sessionmaker[Session]):
    client = client_for(sm)
    version = import_support(client)
    honest = register(client, "honest", "honest_scripted")
    lying = register(client, "lying", "lying_broken")
    return client, version, [honest["id"], lying["id"]]


def test_creates_experiment_runs_and_jobs_together(sm: sessionmaker[Session], setup) -> None:
    client, version, config_ids = setup
    payload = experiment_payload(version["id"], config_ids, split="validation", repetitions=3)

    response = client.post("/experiments", json=payload, headers={"Idempotency-Key": "k-1"})

    assert response.status_code == 201
    body = response.json()
    planned = version["split_counts"]["validation"] * 2 * 3
    assert body["status"] == "queued"
    assert body["progress"]["planned_runs"] == planned
    assert body["progress"]["by_status"] == {"queued": planned}
    assert _count(sm, Run) == planned and _count(sm, Job) == planned
    with sm() as session:
        assert set(session.scalars(select(Job.status)).all()) == {"queued"}
        repetitions = session.scalars(select(Run.repetition).distinct()).all()
    assert sorted(repetitions) == [1, 2, 3]


def test_manifest_pins_dataset_configs_and_evaluators(setup) -> None:
    client, version, config_ids = setup
    body = client.post(
        "/experiments", json=experiment_payload(version["id"], config_ids), headers={"Idempotency-Key": "k-m"}
    ).json()
    manifest = body["manifest"]
    assert manifest["dataset"]["content_hash"] == version["content_hash"]
    assert manifest["split"] == "validation"
    assert {c["id"] for c in manifest["agent_configs"]} == set(config_ids)
    assert all(len(c["config_hash"]) == 64 and len(c["tool_schema_hash"]) == 64 for c in manifest["agent_configs"])
    assert {e["name"] for e in manifest["evaluators"]} == {"outcome", "policy", "trajectory", "response"}
    assert len(manifest["tasks"]) == version["split_counts"]["validation"]
    assert manifest["environment"]["version"]


def test_idempotency_key_is_required(setup) -> None:
    client, version, config_ids = setup
    response = client.post("/experiments", json=experiment_payload(version["id"], config_ids))
    assert response.status_code == 400
    assert "Idempotency-Key" in response.json()["detail"]


def test_same_key_and_payload_returns_the_same_experiment(sm: sessionmaker[Session], setup) -> None:
    client, version, config_ids = setup
    payload = experiment_payload(version["id"], config_ids)

    first = client.post("/experiments", json=payload, headers={"Idempotency-Key": "same"})
    second = client.post("/experiments", json=payload, headers={"Idempotency-Key": "same"})

    assert first.status_code == 201 and second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert _count(sm, Experiment) == 1
    assert _count(sm, Run) == first.json()["progress"]["planned_runs"]


def test_same_key_with_a_different_payload_is_409(sm: sessionmaker[Session], setup) -> None:
    client, version, config_ids = setup
    client.post("/experiments", json=experiment_payload(version["id"], config_ids), headers={"Idempotency-Key": "k"})
    runs_before = _count(sm, Run)

    conflict = client.post(
        "/experiments",
        json=experiment_payload(version["id"], config_ids, repetitions=2),
        headers={"Idempotency-Key": "k"},
    )

    assert conflict.status_code == 409
    assert _count(sm, Experiment) == 1 and _count(sm, Run) == runs_before


def test_concurrent_requests_with_one_key_create_one_experiment(sm: sessionmaker[Session], setup) -> None:
    _, version, config_ids = setup
    payload = ExperimentCreate.model_validate(experiment_payload(version["id"], config_ids))
    results: list = []
    barrier = threading.Barrier(4)

    def submit() -> None:
        barrier.wait()
        results.append(create_experiment(sm, payload, "race"))

    threads = [threading.Thread(target=submit) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len({experiment.id for experiment, _ in results}) == 1
    assert sum(created for _, created in results) == 1
    assert _count(sm, Experiment) == 1


def test_creation_is_all_or_nothing(sm: sessionmaker[Session], setup, monkeypatch: pytest.MonkeyPatch) -> None:
    client, version, config_ids = setup

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("database fell over while creating jobs")

    monkeypatch.setattr(experiment_service, "_job_for", explode)
    with pytest.raises(RuntimeError):
        client.post(
            "/experiments", json=experiment_payload(version["id"], config_ids), headers={"Idempotency-Key": "boom"}
        )
    assert _count(sm, Experiment) == 0 and _count(sm, Run) == 0 and _count(sm, Job) == 0


@pytest.mark.parametrize(
    "change",
    [
        {"agent_config_ids": [str(uuid.uuid4())]},
        {"dataset_version_id": str(uuid.uuid4())},
        {"split": "train"},
        {"repetitions": 0},
        {"agent_config_ids": []},
    ],
)
def test_invalid_requests_create_nothing(sm: sessionmaker[Session], setup, change: dict) -> None:
    client, version, config_ids = setup
    payload = {**experiment_payload(version["id"], config_ids), **change}
    response = client.post("/experiments", json=payload, headers={"Idempotency-Key": uuid.uuid4().hex})
    assert response.status_code == 422
    assert _count(sm, Experiment) == 0 and _count(sm, Run) == 0


def test_duplicate_config_ids_are_rejected(setup) -> None:
    client, version, config_ids = setup
    payload = experiment_payload(version["id"], [config_ids[0], config_ids[0]])
    assert client.post("/experiments", json=payload, headers={"Idempotency-Key": "dup"}).status_code == 422


def test_unknown_ids_are_404(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    missing = uuid.uuid4()
    assert client.get(f"/experiments/{missing}").status_code == 404
    assert client.get(f"/experiments/{missing}/runs").status_code == 404
    assert client.post(f"/experiments/{missing}/cancel").status_code == 404
    assert client.get(f"/runs/{missing}").status_code == 404
    assert client.get(f"/attempts/{missing}/events").status_code == 404
    assert client.get("/experiments/not-a-uuid").status_code == 422


def test_runs_are_paginated_and_filterable(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted"), ("lying", "lying_broken")])
    planned = experiment["progress"]["planned_runs"]

    page = client.get(f"/experiments/{experiment['id']}/runs", params={"limit": 3, "offset": 0}).json()
    rest = client.get(f"/experiments/{experiment['id']}/runs", params={"limit": 100, "offset": 3}).json()
    assert page["total"] == planned and len(page["items"]) == 3
    assert len({r["id"] for r in page["items"] + rest["items"]}) == planned

    worker(sm).run_until_idle()
    failed = client.get(f"/experiments/{experiment['id']}/runs", params={"success": "false"}).json()
    lying_id = next(c["id"] for c in experiment["manifest"]["agent_configs"] if c["name"] == "lying")
    by_config = client.get(f"/experiments/{experiment['id']}/runs", params={"agent_config_id": lying_id}).json()
    assert failed["total"] >= 1 and all(r["success"] is False for r in failed["items"])
    assert by_config["total"] == planned // 2
    assert all(r["agent_config_id"] == lying_id for r in by_config["items"])
    assert client.get(f"/experiments/{experiment['id']}/runs", params={"limit": 0}).status_code == 422


def test_cancel_stops_queued_runs(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted")])

    response = client.post(f"/experiments/{experiment['id']}/cancel")
    again = client.post(f"/experiments/{experiment['id']}/cancel")

    assert response.status_code == 202 and again.status_code == 202
    body = client.get(f"/experiments/{experiment['id']}").json()
    assert body["status"] == "cancelled"
    assert body["progress"]["by_status"] == {"cancelled": experiment["progress"]["planned_runs"]}
    with sm() as session:
        assert set(session.scalars(select(Job.status)).all()) == {"cancelled"}
    assert worker(sm).process_one() is False  # nothing left to claim


def test_cancelling_a_completed_experiment_is_409(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted")])
    worker(sm).run_until_idle()
    assert client.post(f"/experiments/{experiment['id']}/cancel").status_code == 409


def test_list_experiments(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted")])
    assert [e["id"] for e in client.get("/experiments").json()] == [experiment["id"]]
