"""Paid-pilot safeguards exercised with local PostgreSQL and fake HTTP only."""
import json
import uuid
from decimal import Decimal

import httpx
import pytest
from sqlalchemy import select

from app.agents import Limits, ToolLoopAgent
from app.db.models import Attempt, TraceEvent, Run
from app.providers.openai_http import OpenAIChatProvider
from app.worker.worker import Worker
from tests.postgres.helpers import client_for, import_support, register, experiment_payload, launch

KEY = "sk-offline-export-DO-NOT-LEAK-987654321"


@pytest.mark.parametrize("status,code,expected", [(401, "invalid_api_key", 1),
                                                  (429, "insufficient_quota", 1),
                                                  (429, "rate_limit_exceeded", 3)])
def test_retry_classification_and_secret_redaction_survive_persistence_and_export(sm, status, code, expected) -> None:
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"type": code, "code": None,
                             "message": f"Authorization: Bearer {KEY}"}})
    def adapter(spec):
        return ToolLoopAgent(OpenAIChatProvider(api_key=KEY, client=httpx.Client(transport=httpx.MockTransport(handler))),
                             limits=Limits(max_cost_usd=Decimal("0.50")))
    adapters = {"tool_loop": adapter}
    client = client_for(sm, adapters=adapters)
    dataset = import_support(client)
    config = register(client, "pilot", "tool_loop", model="gpt-4.1-mini", limits={"max_cost_usd": "0.50"})
    payload = experiment_payload(dataset["id"], [config["id"]])
    payload["task_ids"] = ["refund_eligible_001"]
    experiment = client.post("/experiments", json=payload, headers={"Idempotency-Key": uuid.uuid4().hex}).json()
    assert "id" in experiment, experiment
    Worker(sm, adapters=adapters, experiment_id=uuid.UUID(experiment["id"]), poll_seconds=0.01,
           retry_backoff_seconds=0).run_until_idle()
    assert len(calls) == expected
    with sm() as session:
        attempts = session.scalars(select(Attempt)).all()
        assert len(attempts) == expected and all(a.cost_usd is None for a in attempts)
        serialized = json.dumps([e.payload for e in session.scalars(select(TraceEvent))], default=str)
    report = client.get(f"/experiments/{experiment['id']}/export").json()
    csv = client.get(f"/experiments/{experiment['id']}/export?format=csv").text
    assert report["scorecards"][0]["scorecard"]["usage"]["cost_usd"] is None
    for output in (serialized, json.dumps(report), csv):
        assert KEY not in output and "Bearer " not in output and "Authorization" not in output


def test_scoped_pilot_worker_does_not_drain_another_experiment(sm) -> None:
    client = client_for(sm)
    other = launch(client, [("other", "honest_scripted")])
    own = launch(client, [("own", "honest_scripted")])
    Worker(sm, experiment_id=uuid.UUID(own["id"]), poll_seconds=0.01).run_until_idle()
    with sm() as session:
        assert set(session.scalars(select(Run.status).where(Run.experiment_id == uuid.UUID(other["id"])))) == {"queued"}
        assert set(session.scalars(select(Run.status).where(Run.experiment_id == uuid.UUID(own["id"])))) == {"completed"}


def test_worker_without_the_pinned_price_version_leaves_the_job_for_an_up_to_date_worker(sm, monkeypatch) -> None:
    # Reproduces experiment 6d50a610: a long-running worker started before the gpt-4.1-mini price row
    # existed leased 18 of 20 pilot jobs and failed each one ADAPTER_UNAVAILABLE in ~13 ms.
    from app.worker import queue

    client = client_for(sm)
    dataset = import_support(client)
    config = register(client, "pilot", "tool_loop", model="gpt-4.1-mini", limits={"max_cost_usd": "0.50"})
    payload = experiment_payload(dataset["id"], [config["id"]])
    payload["task_ids"] = ["refund_eligible_001"]
    pilot = client.post("/experiments", json=payload, headers={"Idempotency-Key": uuid.uuid4().hex}).json()
    assert pilot["manifest"]["price_versions"] == [{"model": "gpt-4.1-mini", "price_version_id": "openai-2026-11-22"}]
    scripted = launch(client, [("honest", "honest_scripted")])  # queued later; pins no price version

    up_to_date = queue.KNOWN_PRICE_VERSION_IDS
    monkeypatch.setattr(queue, "KNOWN_PRICE_VERSION_IDS", up_to_date - {"openai-2026-11-22"})
    stale = queue.claim_job(sm, "stale-worker", lease_seconds=30)
    assert stale is not None and str(stale.experiment_id) == scripted["id"]  # skipped the older pilot job
    with sm() as session:
        run = session.scalars(select(Run).where(Run.experiment_id == uuid.UUID(pilot["id"]))).one()
        assert run.status == "queued"
        assert session.scalars(select(Attempt).where(Attempt.run_id == run.id, Attempt.status != "queued")).all() == []

    monkeypatch.setattr(queue, "KNOWN_PRICE_VERSION_IDS", up_to_date)
    fresh = queue.claim_job(sm, "fresh-worker", lease_seconds=30, experiment_id=uuid.UUID(pilot["id"]))
    assert fresh is not None and fresh.attempt_number == 1


def test_evaluate_preflight_finds_a_running_background_worker_from_its_claimed_attempts(sm) -> None:
    import socket
    import subprocess
    import sys

    from app import cli

    # A stand-in background worker: a live process whose command line names app.worker.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "-m", "app.worker"])
    try:
        launch(client_for(sm), [("honest", "honest_scripted")])
        Worker(sm, worker_id=f"{socket.gethostname()}:{proc.pid}:bg0001", poll_seconds=0.01).run_until_idle()
        assert cli._background_workers(sm) == [proc.pid]
    finally:
        proc.kill()
        proc.wait()
    assert cli._background_workers(sm) == []  # an exited worker no longer blocks `make evaluate`
