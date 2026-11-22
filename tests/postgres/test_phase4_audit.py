"""Deterministic concurrency and persistence regressions from the Phase 4 audit."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import threading

import pytest
from sqlalchemy import event, func, select, text, update
from sqlalchemy.orm import Session, sessionmaker

from app.agents import ToolLoopAgent
from app.db.models import Attempt, Experiment, Job, Run, Score, TraceEvent
from app.providers import ProviderError, ScriptedProvider
from app.services.experiments import cancel_experiment
from app.worker import queue, executor
from tests.postgres.helpers import TEST_ADAPTERS, client_for, launch, worker
from tests.postgres.test_recovery import _execute_locally, _expire


@pytest.mark.parametrize("operation", ["heartbeat", "finalize", "fail"])
def test_expired_token_cannot_write_before_sweeper_runs(sm, operation):
    launch(client_for(sm), [("honest", "honest_scripted")])
    lease = queue.claim_job(sm, "expired", lease_seconds=30)
    result = _execute_locally(sm, lease)
    _expire(sm, lease.job_id)
    if operation == "heartbeat":
        assert queue.heartbeat(sm, lease, lease_seconds=30) is False
    else:
        with pytest.raises(queue.LeaseLost):
            if operation == "finalize":
                queue.finalize_attempt(sm, lease, result, duration_ms=1)
            else:
                queue.fail_attempt(sm, lease, error_code="FAILED", message="late", retryable=False)
    assert queue.sweep_expired_leases(sm) == [lease.job_id]
    with sm() as session:
        assert session.get(Attempt, lease.attempt_id).error_code == "WORKER_LEASE_EXPIRED"
        assert session.scalar(select(func.count()).select_from(Score)) == 0


def _bounded_sessions(sm):
    """Test failures must time out instead of leaving a blocked worker thread behind."""
    class BoundedSession(Session):
        pass

    @event.listens_for(BoundedSession, "after_begin")
    def set_timeout(session, transaction, connection):
        connection.execute(text("SET LOCAL statement_timeout = '8s'"))
        connection.execute(text("SET LOCAL lock_timeout = '5s'"))

    return sessionmaker(sm.kw["bind"], class_=BoundedSession, expire_on_commit=False)


def test_two_finalizers_with_fk_locks_do_not_deadlock(sm, monkeypatch):
    launch(client_for(sm), [("honest", "honest_scripted")])
    leases = [queue.claim_job(sm, f"w{i}", lease_seconds=30) for i in range(2)]
    results = [_execute_locally(sm, lease) for lease in leases]
    flushed = threading.Barrier(2, timeout=5)
    persist = queue._persist

    def synchronized_persist(session, *args):
        persist(session, *args)
        session.flush()  # both finalizers hold FK KEY SHARE on the same experiment
        flushed.wait()

    monkeypatch.setattr(queue, "_persist", synchronized_persist)
    bounded = _bounded_sessions(sm)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue.finalize_attempt, bounded, lease, result, duration_ms=1)
                   for lease, result in zip(leases, results)]
        for future in futures:
            future.result(timeout=12)
    with sm() as session:
        assert all(session.get(Job, lease.job_id).status == "done" for lease in leases)
        assert session.scalar(select(func.count()).select_from(Score)) == 8


def test_cancellation_can_commit_while_finalizer_holds_score_fk_locks(sm, monkeypatch):
    exp = launch(client_for(sm), [("honest", "honest_scripted")])
    lease = queue.claim_job(sm, "w", lease_seconds=30)
    result = _execute_locally(sm, lease)
    flushed, cancelled = threading.Event(), threading.Event()
    persist = queue._persist

    def paused_persist(session, *args):
        persist(session, *args)
        session.flush()
        flushed.set()
        assert cancelled.wait(6)

    monkeypatch.setattr(queue, "_persist", paused_persist)
    with ThreadPoolExecutor(max_workers=2) as pool:
        finalizer = pool.submit(queue.finalize_attempt, _bounded_sessions(sm), lease, result, duration_ms=1)
        try:
            assert flushed.wait(5)
            cancel_experiment(_bounded_sessions(sm), exp["id"])
        finally:
            cancelled.set()
        finalizer.result(timeout=12)
    with sm() as session:
        assert session.get(Experiment, exp["id"]).status == "cancelled"
        # An execution already finished before cancellation keeps its actual result.
        assert session.get(Job, lease.job_id).status == "done"
        assert set(session.scalars(select(Job.status))) <= {"done", "cancelled"}


def test_sweeper_and_cancellation_do_not_deadlock_or_leave_a_retry(sm, monkeypatch):
    exp = launch(client_for(sm), [("honest", "honest_scripted")])
    lease = queue.claim_job(sm, "expired", lease_seconds=30)
    _expire(sm, lease.job_id)
    started = threading.Barrier(2, timeout=5)
    bounded = _bounded_sessions(sm)

    def sweep():
        started.wait()
        return queue.sweep_expired_leases(bounded)

    def cancel():
        started.wait()
        return cancel_experiment(bounded, exp["id"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(sweep), pool.submit(cancel)]
        for future in futures:
            future.result(timeout=12)
    with sm() as session:
        assert set(session.scalars(select(Job.status))) == {"cancelled"}
        assert session.get(Experiment, exp["id"]).status == "cancelled"


@pytest.mark.parametrize("retryable,attempt_count", [(False, 1), (True, 3)])
def test_nul_in_provider_error_does_not_change_retry_classification(sm, retryable, attempt_count):
    adapters = {**TEST_ADAPTERS, "nul_provider": lambda spec: ToolLoopAgent(ScriptedProvider([
        ProviderError("upstream\x00failure", retryable=retryable, status_code=503 if retryable else 400)
    ]))}
    launch(client_for(sm, adapters), [("nul", "nul_provider")])
    worker(sm, adapters, retry_backoff_seconds=0).run_until_idle()
    with sm() as session:
        for run in session.scalars(select(Run)):
            attempts = session.scalars(select(Attempt).where(Attempt.run_id == run.id)).all()
            assert len(attempts) == attempt_count
            assert all(a.error_code == "PROVIDER_ERROR" for a in attempts)
            assert all(a.cost_usd is None and a.input_tokens is None for a in attempts)
            assert all(a.error["message"] == "upstream\ufffdfailure" for a in attempts)


@pytest.mark.parametrize("persist_failure", [False, True])
def test_dirty_usage_metadata_preserves_known_accounting(sm, monkeypatch, persist_failure):
    launch(client_for(sm), [("honest", "honest_scripted")])
    lease = queue.claim_job(sm, "w", lease_seconds=30)
    result = _execute_locally(sm, lease)
    usage = result.usage.model_copy(update={"input_tokens": 123, "output_tokens": 45,
                                          "cost_usd": Decimal("0.0123"), "price_version_ids": ("rate\x00v1",)})
    if persist_failure:
        queue.fail_attempt(sm, lease, error_code="PERSIST_FAILED", message="bad\x00storage", retryable=False, usage=usage)
    else:
        queue.finalize_attempt(sm, lease, replace(result, usage=usage), duration_ms=1)
    with sm() as session:
        attempt = session.get(Attempt, lease.attempt_id)
        assert (attempt.input_tokens, attempt.output_tokens, attempt.cost_usd) == (123, 45, Decimal("0.0123"))
        assert attempt.usage["price_version_ids"] == ["rate\ufffdv1"]
        assert session.get(Job, lease.job_id).status == ("failed" if persist_failure else "done")


def test_unexpected_provider_crash_is_not_retried_or_reported_free(sm):
    def crash(request):
        raise RuntimeError("decoder failed")

    adapters = {**TEST_ADAPTERS, "decoder_crash": lambda spec: ToolLoopAgent(ScriptedProvider([crash]))}
    exp = launch(client_for(sm, adapters), [("decoder", "decoder_crash")])
    worker(sm, adapters).run_until_idle()
    with sm() as session:
        for run in session.scalars(select(Run)):
            [attempt] = session.scalars(select(Attempt).where(Attempt.run_id == run.id)).all()
            assert attempt.error_code == "ADAPTER_EXCEPTION"
            assert attempt.usage["model_calls"] == 1
            assert attempt.cost_usd is None
    card = client_for(sm).get(f"/experiments/{exp['id']}").json()["scorecard"]
    assert card["cost_usd"] is None


@pytest.mark.xfail(strict=True, reason="Open durability gap: no durable execution-complete checkpoint before both persistence writes fail")
def test_double_persistence_failure_does_not_rerun_a_completed_agent(sm, monkeypatch):
    launch(client_for(sm), [("loop", "tool_loop_scripted")])
    lease = queue.claim_job(sm, "w", lease_seconds=30)

    def unavailable(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(queue, "_persist", unavailable)
    monkeypatch.setattr(executor, "fail_attempt", unavailable)
    with pytest.raises(RuntimeError, match="database unavailable"):
        executor.execute(sm, lease, adapters=TEST_ADAPTERS, lease_seconds=30,
                         heartbeat_seconds=.2, retry_backoff_seconds=0)
    _expire(sm, lease.job_id)
    queue.sweep_expired_leases(sm)
    with sm() as session:
        job = session.get(Job, lease.job_id)
        attempt = session.get(Attempt, lease.attempt_id)
        assert job.status == "failed"  # currently queued: the finished agent will be rerun
        assert attempt.input_tokens == 200  # currently lost with both rolled-back writes


def test_claim_does_not_wait_for_experiment_while_holding_queued_job(sm, monkeypatch):
    exp = launch(client_for(sm), [("honest", "honest_scripted")])
    claimed, release = threading.Event(), threading.Event()
    original = queue._new_attempt

    def paused_new_attempt(session, run_id):
        attempt = original(session, run_id)
        claimed.set()  # claim holds Job's row lock; its committed status is still queued
        assert release.wait(6)
        return attempt

    monkeypatch.setattr(queue, "_new_attempt", paused_new_attempt)
    bounded = _bounded_sessions(sm)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claiming = pool.submit(queue.claim_job, bounded, "claiming", lease_seconds=30)
        assert claimed.wait(5)
        # Hold the experiment exactly as cancellation does. Claim must commit its
        # job first; only its subsequent cosmetic transaction may wait here.
        try:
            with bounded.begin() as session:
                session.get(Experiment, exp["id"], with_for_update=queue._EXPERIMENT_LOCK)
                release.set()
                # Taking the queued-job locks would deadlock if claim still needed
                # the experiment inside its first transaction.
                session.scalars(select(Job).join(Run, Run.id == Job.run_id)
                                .where(Run.experiment_id == exp["id"], Job.status == "queued")
                                .with_for_update(of=Job)).all()
        finally:
            release.set()
        lease = claiming.result(timeout=12)
    cancel_experiment(sm, exp["id"])
    _expire(sm, lease.job_id)
    queue.sweep_expired_leases(sm)
    with sm() as session:
        assert set(session.scalars(select(Job.status))) == {"cancelled"}


def test_invalid_adapter_return_is_not_retried_and_retains_spend(sm):
    from app.providers.base import Usage

    class InvalidAdapter:
        name = "invalid"

        def run(self, task, runtime):
            runtime.ledger.record("model", Usage(input_tokens=100, output_tokens=20), Decimal("0.01"), "test")
            return None

    adapters = {**TEST_ADAPTERS, "invalid_return": lambda spec: InvalidAdapter()}
    launch(client_for(sm, adapters), [("invalid", "invalid_return")])
    worker(sm, adapters, retry_backoff_seconds=0).run_until_idle()
    with sm() as session:
        for run in session.scalars(select(Run)):
            [attempt] = session.scalars(select(Attempt).where(Attempt.run_id == run.id)).all()
            assert attempt.error_code == "ADAPTER_EXCEPTION"
            assert (attempt.input_tokens, attempt.cost_usd) == (100, Decimal("0.01"))
