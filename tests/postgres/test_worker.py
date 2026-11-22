"""The worker: SKIP LOCKED claims, leases, heartbeats, end-to-end results, cancellation, concurrency."""

from __future__ import annotations

import threading
import time
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.agents import ToolLoopAgent
from app.agents.base import AgentResult, AgentRuntime
from app.agents.registry import AgentConfigSpec
from app.db.models import Attempt, Job, ModelCall, Run, Score, TaskVersion, ToolCall, TraceEvent
from app.providers.base import ProviderError
from app.providers.fake import ScriptedProvider
from app.worker import queue
from app.services.experiments import cancel_experiment
from app.tasks import AgentTaskView
from app.worker.executor import execute
from app.worker.queue import claim_job, heartbeat, sweep_expired_leases
from app.worker.worker import DEFAULT_CONCURRENCY
from tests.postgres.helpers import PROBE, TEST_ADAPTERS, client_for, launch, worker


def _runs(sm: sessionmaker[Session]) -> list[Run]:
    with sm() as session:
        return list(session.scalars(select(Run)).all())


def _attempts(sm: sessionmaker[Session], run_id: uuid.UUID) -> list[Attempt]:
    with sm() as session:
        return list(session.scalars(select(Attempt).where(Attempt.run_id == run_id).order_by(Attempt.attempt_number)))


def test_experiment_runs_end_to_end(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted"), ("lying", "lying_broken")])

    worker(sm).run_until_idle()

    body = client.get(f"/experiments/{experiment['id']}").json()
    planned = experiment["progress"]["planned_runs"]
    assert body["status"] == "completed"
    assert body["progress"]["by_status"] == {"completed": planned}
    card = body["scorecard"]
    assert card["scored_runs"] == planned and card["evaluation_coverage"] == 1.0
    assert card["successful_runs"] + card["failed_runs"] + card["evaluation_errors"] == planned
    assert set(card["evaluators"]) == {"outcome", "policy", "trajectory", "response"}
    assert card["cost_usd"] == "0"  # scripted agents make no model calls

    runs = client.get(f"/experiments/{experiment['id']}/runs", params={"limit": 100}).json()["items"]
    lying_refund = next(r for r in runs if r["task_id"] == "refund_eligible_001" and r["agent_config_name"] == "lying")
    detail = client.get(f"/runs/{lying_refund['id']}").json()
    assert detail["success"] is False and detail["evaluation_status"] == "fail"
    response_score = next(s for s in detail["scores"] if s["evaluator_name"] == "response")
    assert response_score["reason_code"] == "FALSE_SUCCESS_CLAIM"
    [attempt] = detail["attempts"]
    assert attempt["status"] == "completed" and attempt["environment_id"]

    events = client.get(f"/attempts/{attempt['id']}/events").json()["items"]
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    assert events[0]["name"] == "attempt_started" and events[-1]["name"] == "attempt_completed"
    assert events[0]["payload"]["environment_id"] == attempt["environment_id"]
    proof = response_score["evidence"]["proof"]["final_response_sequence"]
    assert any(e["sequence"] == proof and e["kind"] == "agent_response" for e in events)


def test_results_are_persisted_relationally(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    launch(client, [("honest", "honest_scripted")])
    worker(sm).run_until_idle()
    with sm() as session:
        attempts = session.scalars(select(Attempt)).all()
        assert all(a.initial_snapshot and a.final_snapshot and a.finished_at for a in attempts)
        tool_calls = session.scalars(select(ToolCall)).all()
        assert tool_calls and all(c.decision in ("allowed", "denied") for c in tool_calls)
        assert session.scalar(select(func.count()).select_from(Score)) == 4 * len(attempts)
        categories = set(session.scalars(select(Score.task_category)).all())
        expected = set(
            session.scalars(select(TaskVersion.category).join(Run, Run.task_version_id == TaskVersion.id)).all()
        )
    assert categories == expected


def test_model_calls_record_usage_cost_and_price_version(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    launch(client, [("loop", "tool_loop_scripted")])
    worker(sm).run_until_idle()
    with sm() as session:
        calls = session.scalars(select(ModelCall)).all()
        assert calls and all(c.input_tokens == 100 and c.output_tokens == 20 for c in calls)
        assert all(c.cost_usd is not None and c.price_version_id is not None for c in calls)
        attempt = session.scalars(select(Attempt)).first()
    assert attempt.cost_usd is not None and attempt.cost_usd > 0
    assert attempt.usage["model_calls"] == 2


def test_claim_skips_rows_locked_by_another_transaction(sm: sessionmaker[Session]) -> None:
    launch(client_for(sm), [("honest", "honest_scripted")])
    with sm() as locker:
        locked_id = locker.execute(
            select(Job.id).order_by(Job.available_at, Job.id).limit(1).with_for_update()
        ).scalar_one()
        # Without SKIP LOCKED the claim would block behind the lock; the join timeout catches that.
        leases: list = []
        claimer = threading.Thread(target=lambda: leases.append(claim_job(sm, "w-test", lease_seconds=30)))
        claimer.start()
        claimer.join(timeout=5)
        blocked = claimer.is_alive()
        locker.rollback()
        claimer.join()
    assert not blocked
    assert leases[0] is not None and leases[0].job_id != locked_id


def test_concurrent_claims_never_share_a_job(sm: sessionmaker[Session]) -> None:
    experiment = launch(client_for(sm), [("honest", "honest_scripted")], split="dev")
    planned = experiment["progress"]["planned_runs"]
    leases: list = []
    barrier = threading.Barrier(planned)

    def grab() -> None:
        barrier.wait()
        leases.append(claim_job(sm, f"w-{threading.get_ident()}", lease_seconds=30))

    threads = [threading.Thread(target=grab) for _ in range(planned)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    claimed = [lease.job_id for lease in leases if lease is not None]
    assert len(claimed) == len(set(claimed)) == planned


def test_claim_creates_a_running_attempt_with_a_fresh_environment(sm: sessionmaker[Session]) -> None:
    launch(client_for(sm), [("honest", "honest_scripted")])
    first = claim_job(sm, "w1", lease_seconds=30)
    second = claim_job(sm, "w1", lease_seconds=30)
    assert first.environment_id != second.environment_id
    with sm() as session:
        attempt = session.get(Attempt, first.attempt_id)
        job = session.get(Job, first.job_id)
        run = session.get(Run, first.run_id)
    assert attempt.status == "running" and attempt.attempt_number == 1 and attempt.worker_id == "w1"
    assert job.status == "leased" and job.lease_token == first.token and job.attempts_started == 1
    assert run.status == "running"


def test_heartbeat_extends_only_the_current_lease(sm: sessionmaker[Session]) -> None:
    launch(client_for(sm), [("honest", "honest_scripted")])
    lease = claim_job(sm, "w1", lease_seconds=5)
    with sm() as session:
        before = session.get(Job, lease.job_id).lease_expires_at

    assert heartbeat(sm, lease, lease_seconds=60) is True
    with sm() as session:
        after = session.get(Job, lease.job_id).lease_expires_at
    assert after - before > timedelta(seconds=50)

    stale = lease.__class__(**{**lease.__dict__, "token": uuid.uuid4()})
    assert heartbeat(sm, stale, lease_seconds=60) is False


def test_agent_failures_are_never_retried(sm: sessionmaker[Session]) -> None:
    launch(client_for(sm), [("lying", "lying_broken")])
    worker(sm).run_until_idle()
    for run in _runs(sm):
        assert run.status == "completed"
        assert len(_attempts(sm, run.id)) == 1


def test_adapter_crashes_fail_explicitly_without_rerunning_the_agent(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    experiment = launch(client, [("crash", "crashing")])
    worker(sm, retry_backoff_seconds=0).run_until_idle()

    for run in _runs(sm):
        [attempt] = _attempts(sm, run.id)
        assert attempt.status == "infrastructure_error" and attempt.error_code == "ADAPTER_EXCEPTION"
        assert run.status == "infrastructure_error" and run.evaluation_status == "not_applicable"
    body = client.get(f"/experiments/{experiment['id']}").json()
    assert body["status"] == "completed"  # every planned run ended with a result or an explicit error
    assert body["scorecard"]["evaluation_coverage"] == 0.0
    with sm() as session:
        assert set(session.scalars(select(Job.status)).all()) == {"failed"}
        assert session.scalar(select(func.count()).select_from(Score)) == 0


def test_cancellation_stops_new_tool_execution(sm: sessionmaker[Session]) -> None:
    observed: dict = {}

    class CancelsMidway:
        name = "cancels_midway"

        def run(self, task: AgentTaskView, runtime: AgentRuntime) -> AgentResult:
            runtime.call_tool("get_order", {"order_id": "ORD-1001"})
            with sm() as session:
                experiment_id = session.scalar(select(Run.experiment_id).limit(1))
            cancel_experiment(sm, experiment_id)
            observed["after_cancel"] = runtime.call_tool("get_return_status", {"order_id": "ORD-1001"})
            return AgentResult(final_response="unreachable", stop_reason="completed")

    adapters = {**TEST_ADAPTERS, "cancels_midway": lambda spec: CancelsMidway()}
    client = client_for(sm, adapters)
    experiment = launch(client, [("midway", "cancels_midway")])
    worker(sm, adapters, concurrency=1).run_until_idle()

    assert "after_cancel" not in observed  # the second tool call never returned
    body = client.get(f"/experiments/{experiment['id']}").json()
    assert body["status"] == "cancelled"
    assert body["progress"]["by_status"] == {"cancelled": experiment["progress"]["planned_runs"]}
    with sm() as session:
        [executed] = session.scalars(select(Attempt).where(Attempt.status == "cancelled")).all()
        tools = session.scalars(
            select(TraceEvent.name).where(TraceEvent.attempt_id == executed.id, TraceEvent.kind == "tool_attempt")
        ).all()
        names = session.scalars(select(TraceEvent.name).where(TraceEvent.attempt_id == executed.id)).all()
        assert session.scalar(select(func.count()).select_from(Score)) == 0
    assert tools == ["get_order"]
    assert "cancellation_observed" in names


def test_default_concurrency_is_two_and_is_never_exceeded(sm: sessionmaker[Session]) -> None:
    assert DEFAULT_CONCURRENCY == 2
    launch(client_for(sm), [("slow", "slow_honest")], split="dev")
    PROBE.peak = 0

    worker(sm).run_until_idle()

    assert PROBE.peak == 2
    assert all(r.status == "completed" for r in _runs(sm))


def test_two_workers_process_each_run_exactly_once(sm: sessionmaker[Session]) -> None:
    experiment = launch(client_for(sm), [("slow", "slow_honest")], split="dev")
    workers = [worker(sm, worker_id=f"w{i}") for i in range(2)]
    threads = [threading.Thread(target=w.run_until_idle) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    runs = _runs(sm)
    assert len(runs) == experiment["progress"]["planned_runs"]
    assert all(r.status == "completed" for r in runs)
    assert all(len(_attempts(sm, r.id)) == 1 for r in runs)
    with sm() as session:
        assert set(session.scalars(select(Attempt.worker_id)).all()) <= {"w0", "w1"}


def test_unknown_adapter_fails_without_retrying(sm: sessionmaker[Session]) -> None:
    launch(client_for(sm), [("honest", "honest_scripted")])
    worker(sm, {"lying_broken": TEST_ADAPTERS["lying_broken"]}).run_until_idle()
    for run in _runs(sm):
        [attempt] = _attempts(sm, run.id)
        assert attempt.status == "infrastructure_error" and attempt.error_code == "ADAPTER_UNAVAILABLE"
        assert run.status == "infrastructure_error"


def test_config_tools_narrow_the_task_tools(sm: sessionmaker[Session]) -> None:
    seen: list[tuple[str, ...]] = []

    def factory(spec: AgentConfigSpec):
        class Recorder:
            name = "recorder"

            def run(self, task: AgentTaskView, runtime: AgentRuntime) -> AgentResult:
                seen.append(tuple(t.name for t in runtime.tool_specs))
                return AgentResult(final_response="", stop_reason="completed")

        return Recorder()

    adapters = {**TEST_ADAPTERS, "recorder": factory}
    client = client_for(sm, adapters)
    version = client.post("/datasets/import", json={"dataset": "support"}).json()
    config = client.post(
        "/agent-configs", json={"name": "readonly", "adapter": "recorder", "tools": ["get_order", "search_policy"]}
    ).json()
    client.post(
        "/experiments",
        json={"dataset_version_id": version["id"], "agent_config_ids": [config["id"]], "split": "validation"},
        headers={"Idempotency-Key": "tools"},
    )
    worker(sm, adapters).run_until_idle()
    assert seen and all(set(tools) <= {"get_order", "search_policy"} for tools in seen)


def _provider_failure_adapters(retryable: bool) -> dict:
    def factory(spec: AgentConfigSpec) -> ToolLoopAgent:
        provider = ScriptedProvider(
            [ProviderError("upstream said no", retryable=retryable, status_code=503 if retryable else 400)],
            model="claude-opus-5-5",
        )
        return ToolLoopAgent(provider)

    return {**TEST_ADAPTERS, "failing_provider": factory}


def test_transient_provider_errors_get_new_attempts_with_unknown_usage(sm: sessionmaker[Session]) -> None:
    adapters = _provider_failure_adapters(retryable=True)
    launch(client_for(sm, adapters), [("flaky", "failing_provider")])
    worker(sm, adapters, retry_backoff_seconds=0).run_until_idle()
    for run in _runs(sm):
        attempts = _attempts(sm, run.id)
        assert [a.error_code for a in attempts] == ["PROVIDER_ERROR"] * 3
        assert len({a.environment_id for a in attempts}) == 3
        assert all(a.cost_usd is None and a.input_tokens is None for a in attempts)  # unknown, never zero
        assert run.status == "infrastructure_error"


def test_permanent_provider_errors_are_not_retried(sm: sessionmaker[Session]) -> None:
    adapters = _provider_failure_adapters(retryable=False)
    launch(client_for(sm, adapters), [("broken", "failing_provider")])
    worker(sm, adapters, retry_backoff_seconds=0).run_until_idle()
    for run in _runs(sm):
        [attempt] = _attempts(sm, run.id)
        assert attempt.error_code == "PROVIDER_ERROR"
        assert run.status == "infrastructure_error"


def test_hostile_output_is_stored_safely(sm: sessionmaker[Session]) -> None:
    class Hostile:
        name = "hostile"

        def run(self, task: AgentTaskView, runtime: AgentRuntime) -> AgentResult:
            runtime.call_tool("get_order\x00" + "x" * 300, {"order_id": float("nan")})
            return AgentResult(final_response="refund \x00 issued", stop_reason="completed")

    adapters = {**TEST_ADAPTERS, "hostile": lambda spec: Hostile()}
    launch(client_for(sm, adapters), [("hostile", "hostile")])
    worker(sm, adapters).run_until_idle()
    for run in _runs(sm):
        [attempt] = _attempts(sm, run.id)
        assert attempt.status == "completed" and "\x00" not in attempt.final_response
    with sm() as session:
        call = session.scalars(select(ToolCall)).first()
    assert len(call.tool) <= 100 and call.args == {"order_id": None}


def test_a_result_that_cannot_be_saved_fails_once_and_keeps_its_usage(
    sm: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_persist(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(queue, "_persist", broken_persist)
    launch(client_for(sm), [("loop", "tool_loop_scripted")])
    worker(sm).run_until_idle()
    for run in _runs(sm):
        [attempt] = _attempts(sm, run.id)  # the agent ran once and is never re-run
        assert attempt.status == "infrastructure_error" and attempt.error_code == "PERSIST_FAILED"
        assert attempt.input_tokens == 200 and attempt.cost_usd is not None and attempt.cost_usd > 0
        assert run.status == "infrastructure_error"


def test_a_worker_that_loses_its_lease_stops_and_writes_nothing(sm: sessionmaker[Session]) -> None:
    observed: dict = {}

    class LosesLease:
        name = "loses_lease"

        def run(self, task: AgentTaskView, runtime: AgentRuntime) -> AgentResult:
            runtime.call_tool("get_order", {"order_id": "ORD-1001"})
            # Simulate a stall long enough for the lease to lapse and the run to be handed on.
            with sm.begin() as session:
                session.execute(update(Job).values(lease_expires_at=func.now() - timedelta(seconds=1)))
            observed["swept"] = len(sweep_expired_leases(sm))
            time.sleep(0.8)  # > heartbeat interval: the heartbeat discovers the lease is gone
            observed["after_loss"] = runtime.call_tool("get_return_status", {"order_id": "ORD-1001"})
            return AgentResult(final_response="done", stop_reason="completed")

    adapters = {**TEST_ADAPTERS, "loses_lease": lambda spec: LosesLease()}
    client = client_for(sm, adapters)
    version = client.post("/datasets/import", json={"dataset": "support"}).json()
    config = client.post("/agent-configs", json={"name": "loser", "adapter": "loses_lease"}).json()
    client.post(
        "/experiments",
        json={"dataset_version_id": version["id"], "agent_config_ids": [config["id"]], "split": "validation"},
        headers={"Idempotency-Key": "lease-loss"},
    )
    lease = claim_job(sm, "stalled", lease_seconds=30)
    # First heartbeat at 0.3 s: after the forced expiry + sweep, so it cannot renew the lease in between.
    execute(sm, lease, adapters=adapters, lease_seconds=30, heartbeat_seconds=0.3, retry_backoff_seconds=0)

    assert observed["swept"] == 1
    assert "after_loss" not in observed  # the next tool call was refused
    first, replacement = _attempts(sm, lease.run_id)
    assert first.error_code == "WORKER_LEASE_EXPIRED" and first.final_snapshot is None
    assert replacement.status == "queued" and replacement.environment_id != first.environment_id
    with sm() as session:
        assert session.scalar(select(func.count()).select_from(Score).where(Score.run_id == lease.run_id)) == 0
        names = session.scalars(select(TraceEvent.name).where(TraceEvent.attempt_id == first.id)).all()
    assert names == ["attempt_interrupted"]


@pytest.mark.parametrize("bad", [0, -1])
def test_worker_rejects_nonpositive_concurrency(sm: sessionmaker[Session], bad: int) -> None:
    with pytest.raises(ValueError):
        worker(sm, concurrency=bad)
