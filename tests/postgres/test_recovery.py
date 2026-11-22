"""Recovery: expired leases, stale workers, a killed worker process, and an API restart."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

from pydantic_core import to_jsonable_python
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.agents import HonestScriptedAgent
from app.db.models import Attempt, Experiment, Job, Run, Score, TaskVersion, TraceEvent
from app.environment import SupportEnvironment
from app.runner import run_task_attempt
from app.tasks import AgentTaskView, EvaluatorTaskView
from app.worker.queue import LeaseLost, claim_job, finalize_attempt, sweep_expired_leases
from tests.postgres.helpers import client_for, launch, worker

ROOT = Path(__file__).resolve().parents[2]


def _expire(sm: sessionmaker[Session], job_id) -> None:
    with sm.begin() as session:
        session.execute(
            update(Job).where(Job.id == job_id).values(lease_expires_at=func.now() - timedelta(seconds=1))
        )


def _attempts(sm: sessionmaker[Session], run_id) -> list[Attempt]:
    with sm() as session:
        return list(session.scalars(select(Attempt).where(Attempt.run_id == run_id).order_by(Attempt.attempt_number)))


def _execute_locally(sm: sessionmaker[Session], lease):
    """Run the leased attempt in-process, as a worker would, without finalizing it."""
    with sm() as session:
        task = session.get(TaskVersion, session.get(Run, lease.run_id).task_version_id)
        agent_view = AgentTaskView.model_validate(task.agent_view)
        evaluator_view = EvaluatorTaskView.model_validate(task.evaluator_view)
    return run_task_attempt(agent_view, evaluator_view, HonestScriptedAgent(), environment_id=lease.environment_id)


def test_sweeper_interrupts_the_attempt_and_queues_a_fresh_replacement(sm: sessionmaker[Session]) -> None:
    launch(client_for(sm), [("honest", "honest_scripted")])
    lease = claim_job(sm, "doomed-worker", lease_seconds=30)
    _expire(sm, lease.job_id)

    swept = sweep_expired_leases(sm)

    assert swept == [lease.job_id]
    first, replacement = _attempts(sm, lease.run_id)
    assert first.status == "infrastructure_error" and first.error_code == "WORKER_LEASE_EXPIRED"
    assert first.error["worker_id"] == "doomed-worker" and first.finished_at is not None
    assert replacement.status == "queued" and replacement.attempt_number == 2
    assert replacement.environment_id != first.environment_id
    with sm() as session:
        job = session.get(Job, lease.job_id)
        run = session.get(Run, lease.run_id)
        events = session.scalars(select(TraceEvent).where(TraceEvent.attempt_id == first.id)).all()
    assert job.status == "queued" and job.lease_token is None and job.current_attempt_id == replacement.id
    assert run.status == "queued"
    assert [(e.kind, e.name) for e in events] == [("lifecycle", "attempt_interrupted")]
    assert sweep_expired_leases(sm) == []  # idempotent


def test_replacement_attempt_runs_in_a_clean_environment(sm: sessionmaker[Session]) -> None:
    launch(client_for(sm), [("honest", "honest_scripted")])
    lease = claim_job(sm, "doomed-worker", lease_seconds=30)
    _expire(sm, lease.job_id)
    sweep_expired_leases(sm)

    worker(sm).run_until_idle()

    first, second = _attempts(sm, lease.run_id)
    assert second.status == "completed" and second.attempt_number == 2
    with sm() as session:
        task = session.get(TaskVersion, session.get(Run, lease.run_id).task_version_id)
    clean = SupportEnvironment(task.evaluator_view["fixture"]).initial_snapshot
    assert second.initial_snapshot == to_jsonable_python(clean)


def test_a_stale_worker_cannot_write_results(sm: sessionmaker[Session]) -> None:
    launch(client_for(sm), [("honest", "honest_scripted")])
    stale = claim_job(sm, "slow-worker", lease_seconds=30)
    result = _execute_locally(sm, stale)  # the slow worker finishes its work...
    _expire(sm, stale.job_id)
    sweep_expired_leases(sm)  # ...but its lease expired and the run was handed on
    worker(sm).run_until_idle()

    try:
        finalize_attempt(sm, stale, result, duration_ms=1.0)
    except LeaseLost:
        pass
    else:
        raise AssertionError("a stale lease must not finalize")

    first, second = _attempts(sm, stale.run_id)
    assert first.status == "infrastructure_error" and first.final_snapshot is None
    with sm() as session:
        scored_attempts = set(session.scalars(select(Score.attempt_id).where(Score.run_id == stale.run_id)))
        stale_events = session.scalar(
            select(func.count()).select_from(TraceEvent).where(TraceEvent.attempt_id == first.id)
        )
    assert scored_attempts == {second.id}  # exactly one result per run
    assert stale_events == 1  # only the sweeper's interruption marker


def test_expired_leases_are_given_up_after_max_attempts(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted")])
    with sm.begin() as session:
        session.execute(update(Job).values(max_attempts=1))
    while (lease := claim_job(sm, "doomed", lease_seconds=30)) is not None:
        _expire(sm, lease.job_id)
    sweep_expired_leases(sm)

    body = client.get(f"/experiments/{experiment['id']}").json()
    assert body["status"] == "completed"
    assert body["progress"]["by_status"] == {"infrastructure_error": experiment["progress"]["planned_runs"]}
    with sm() as session:
        assert set(session.scalars(select(Job.status))) == {"failed"}


def test_sweeping_a_cancelled_experiment_cancels_instead_of_retrying(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    experiment = launch(client, [("honest", "honest_scripted")])
    lease = claim_job(sm, "doomed", lease_seconds=30)
    client.post(f"/experiments/{experiment['id']}/cancel")
    assert client.get(f"/experiments/{experiment['id']}").json()["status"] == "cancelling"
    _expire(sm, lease.job_id)

    sweep_expired_leases(sm)

    [attempt] = _attempts(sm, lease.run_id)
    assert attempt.status == "infrastructure_error"
    body = client.get(f"/experiments/{experiment['id']}").json()
    assert body["status"] == "cancelled"
    assert body["progress"]["by_status"] == {"cancelled": experiment["progress"]["planned_runs"]}


def test_api_restart_loses_no_queued_work(sm: sessionmaker[Session]) -> None:
    with client_for(sm) as first_api:
        experiment = launch(first_api, [("honest", "honest_scripted")])
    # The first API process is gone; a new one sees the same plan, and a worker finishes it.
    with client_for(sm) as second_api:
        body = second_api.get(f"/experiments/{experiment['id']}").json()
        assert body["progress"]["by_status"] == {"queued": experiment["progress"]["planned_runs"]}
        worker(sm).run_until_idle()
        assert second_api.get(f"/experiments/{experiment['id']}").json()["status"] == "completed"


def test_killed_worker_process_yields_interrupted_attempt_and_clean_recovery(
    sm: sessionmaker[Session], database_url: str
) -> None:
    experiment = launch(client_for(sm), [("slow", "slow_honest")])
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tests" / "postgres" / "worker_process.py")],
        env={**os.environ, "DATABASE_URL": database_url, "PYTHONPATH": str(ROOT)},
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            with sm() as session:
                running = session.scalar(select(func.count()).select_from(Attempt).where(Attempt.status == "running"))
            if running:
                break
            assert proc.poll() is None, proc.stderr.read().decode()
            time.sleep(0.05)
        else:
            raise AssertionError("worker never started an attempt")
        proc.send_signal(signal.SIGKILL)  # no cleanup, no goodbye
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()

    with sm() as session:
        interrupted = session.scalars(select(Attempt).where(Attempt.status == "running")).all()
    assert interrupted, "the killed worker left running attempts behind"
    time.sleep(1.5)  # the subprocess used a 1s lease and can no longer heartbeat
    assert len(sweep_expired_leases(sm)) == len(interrupted)

    worker(sm).run_until_idle()

    with sm() as session:
        exp = session.get(Experiment, experiment["id"])
        runs = session.scalars(select(Run)).all()
        scores_per_run = dict(
            session.execute(select(Score.run_id, func.count(func.distinct(Score.attempt_id))).group_by(Score.run_id)).all()
        )
    assert exp.status == "completed"
    assert all(r.status == "completed" for r in runs)
    assert all(scores_per_run[r.id] == 1 for r in runs)  # no duplicate results
    for attempt in interrupted:
        attempts = _attempts(sm, attempt.run_id)
        assert attempts[0].error_code == "WORKER_LEASE_EXPIRED"
        assert attempts[-1].status == "completed"
        assert attempts[-1].environment_id != attempts[0].environment_id

