"""PostgreSQL job queue: SKIP LOCKED claims, lease tokens, heartbeats and an expired-lease sweeper.

Execution is at-least-once; results are written at most once. Every write
that finishes an attempt first re-locks the job and checks the lease token,
so a worker whose lease expired (and whose run was handed to a replacement
attempt) can never record a result.

Lock order is always job -> experiment. Cancellation locks the experiment and
then only *queued* jobs, which a worker never holds while waiting for the
experiment, so the two cannot deadlock.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from pydantic_core import to_jsonable_python
from sqlalchemy import ColumnElement, column, exists, func, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Attempt, Experiment, Job, ModelCall, Run, Score, TaskVersion, ToolCall, TraceEvent
from app.accounting import PRICE_VERSIONS, UsageSummary
from app.runner import AttemptResult
from app.services.catalog import price_version_pk

log = logging.getLogger(__name__)

INTERRUPTED = "WORKER_LEASE_EXPIRED"

# FOR NO KEY UPDATE: serialises writers of the experiment row (status, completion) but, unlike
# FOR UPDATE, does not conflict with the KEY SHARE locks that inserting scores (FK -> experiments)
# takes. With FOR UPDATE, two finalizers that had each inserted scores deadlocked on the row.
_EXPERIMENT_LOCK = {"key_share": True}


class LeaseLost(Exception):
    """The lease expired or was taken over; this worker must not write results."""


@dataclass(frozen=True)
class Lease:
    job_id: uuid.UUID
    run_id: uuid.UUID
    experiment_id: uuid.UUID
    attempt_id: uuid.UUID
    attempt_number: int
    environment_id: str
    token: uuid.UUID
    worker_id: str


def new_environment_id() -> str:
    return uuid.uuid4().hex


def _clean(value: Any) -> Any:
    """PostgreSQL rejects NUL in text and JSONB; agent and model output can contain anything."""
    if isinstance(value, str):
        return value.replace("\x00", "\ufffd")
    if isinstance(value, dict):
        return {_clean(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    # NaN/Infinity are not valid JSONB numbers.
    return _clean(to_jsonable_python(value, fallback=repr, inf_nan_mode="null"))


def _text(value: str | None, limit: int | None = None) -> str | None:
    if value is None:
        return None
    cleaned = value.replace("\x00", "\ufffd")
    return cleaned[:limit] if limit is not None else cleaned


# --- claiming ------------------------------------------------------------------------

# Price versions are code, not data: a worker started before a new row was added cannot price
# (and so cannot budget) that model. It must leave such jobs to an up-to-date worker instead of
# failing them, or silently pricing them with an older version than the manifest pinned.
KNOWN_PRICE_VERSION_IDS: frozenset[str] = frozenset(p.price_version_id for p in PRICE_VERSIONS)


def _priced_here(known: frozenset[str]) -> ColumnElement[bool]:
    """True when every price version pinned in the experiment's manifest is one this worker has."""
    pinned = (
        func.jsonb_array_elements(func.coalesce(Experiment.manifest["price_versions"], func.jsonb_build_array()))
        .table_valued(column("value", JSONB))
        .render_derived(name="pinned")
    )
    return ~exists().select_from(pinned).where(pinned.c.value["price_version_id"].astext.not_in(sorted(known)))



def claim_job(sm: sessionmaker[Session], worker_id: str, *, lease_seconds: float, experiment_id: uuid.UUID | None = None) -> Lease | None:
    """Lease the oldest available job, skipping rows other workers hold and experiments this worker
    cannot price. None if nothing is available."""
    with sm.begin() as session:
        row = session.execute(
            select(Job, Run)
            .join(Run, Run.id == Job.run_id)
            .join(Experiment, Experiment.id == Run.experiment_id)
            .where(Job.status == "queued", Job.available_at <= func.now(), Experiment.cancel_requested_at.is_(None))
            .where(Run.experiment_id == experiment_id if experiment_id is not None else True)
            .where(_priced_here(KNOWN_PRICE_VERSION_IDS))
            .order_by(Job.available_at, Job.id)
            .limit(1)
            .with_for_update(of=Job, skip_locked=True)
        ).first()
        if row is None:
            return None
        job, run = row
        attempt = session.get(Attempt, job.current_attempt_id) if job.current_attempt_id else None
        if attempt is None or attempt.status != "queued":
            attempt = _new_attempt(session, run.id)
        attempt.status = "running"
        attempt.worker_id = worker_id
        attempt.started_at = func.now()

        token = uuid.uuid4()
        job.status = "leased"
        job.lease_token = token
        job.lease_expires_at = func.now() + timedelta(seconds=lease_seconds)
        job.leased_by = worker_id
        job.heartbeat_at = func.now()
        job.attempts_started += 1
        job.current_attempt_id = attempt.id
        run.status = "running"
        if run.started_at is None:
            run.started_at = func.now()
        lease = Lease(
            job_id=job.id,
            run_id=run.id,
            experiment_id=run.experiment_id,
            attempt_id=attempt.id,
            attempt_number=attempt.attempt_number,
            environment_id=attempt.environment_id,
            token=token,
            worker_id=worker_id,
        )
    # Separate transaction: never wait on the experiment row while holding a job lock.
    # Cosmetic status only, so a failure here must not strand the lease we already hold.
    try:
        with sm.begin() as session:
            session.execute(
                update(Experiment)
                .where(Experiment.id == lease.experiment_id, Experiment.status == "queued")
                .values(status="running")
                .execution_options(synchronize_session=False)
            )
    except Exception:  # noqa: BLE001
        log.exception("could not mark experiment %s running", lease.experiment_id)
    return lease


def _new_attempt(session: Session, run_id: uuid.UUID) -> Attempt:
    """A queued attempt with a brand-new environment id; the previous attempt's environment is never reused."""
    number = (session.scalar(select(func.max(Attempt.attempt_number)).where(Attempt.run_id == run_id)) or 0) + 1
    attempt = Attempt(
        id=uuid.uuid4(), run_id=run_id, attempt_number=number, status="queued", environment_id=new_environment_id()
    )
    session.add(attempt)
    session.flush()
    return attempt


# --- while running -------------------------------------------------------------------


def heartbeat(sm: sessionmaker[Session], lease: Lease, *, lease_seconds: float) -> bool:
    """Extend the lease. False means it is no longer ours and the attempt must stop."""
    with sm.begin() as session:
        try:
            job = _locked_job(session, lease)
        except LeaseLost:
            return False
        job.lease_expires_at = func.clock_timestamp() + timedelta(seconds=lease_seconds)
        job.heartbeat_at = func.clock_timestamp()
        return True



def cancel_requested(sm: sessionmaker[Session], experiment_id: uuid.UUID) -> bool:
    with sm() as session:
        return session.scalar(
            select(Experiment.cancel_requested_at.is_not(None)).where(Experiment.id == experiment_id)
        ) is True


# --- finishing -----------------------------------------------------------------------


def finalize_attempt(
    sm: sessionmaker[Session],
    lease: Lease,
    result: AttemptResult,
    *,
    duration_ms: float,
    retry_backoff_seconds: float = 2.0,
) -> None:
    """Persist the attempt (trace, model and tool calls, scores) and settle the run, under the lease."""
    with sm.begin() as session:
        job = _locked_job(session, lease)
        run = session.get(Run, lease.run_id)
        attempt = session.get(Attempt, lease.attempt_id)
        task = session.get(TaskVersion, run.task_version_id)
        _persist(session, attempt, run, task, result, duration_ms)
        # Experiment lock only now (still job -> experiment), so finalizers don't serialise on the inserts.
        experiment = session.get(Experiment, lease.experiment_id, with_for_update=_EXPERIMENT_LOCK)

        if result.execution_status == "infrastructure_error":
            _after_infrastructure_error(
                session, job, run, experiment,
                message=(attempt.error or {}).get("message", "infrastructure error"),
                retryable=_retryable(result),
                backoff_seconds=retry_backoff_seconds,
            )
        elif result.execution_status == "cancelled":
            _close(job, run, run_status="cancelled", job_status="cancelled", evaluation_status="not_applicable")
        else:
            _close(
                job, run,
                run_status=result.execution_status,
                job_status="done",
                evaluation_status=evaluation_status_for(result),
                success=result.success,
            )
        _settle_experiment(session, experiment)


def fail_attempt(
    sm: sessionmaker[Session],
    lease: Lease,
    *,
    error_code: str,
    message: str,
    retryable: bool,
    retry_backoff_seconds: float = 2.0,
    usage: UsageSummary | None = None,
) -> None:
    """Record an attempt that could not run or be saved as an explicit infrastructure error.

    ``usage`` keeps the spend of an attempt that ran but whose result could not be persisted.
    """
    with sm.begin() as session:
        job = _locked_job(session, lease)
        run = session.get(Run, lease.run_id)
        attempt = session.get(Attempt, lease.attempt_id)
        attempt.status = "infrastructure_error"
        attempt.error_code = error_code
        attempt.error = {"message": _text(message, 2000)}
        attempt.finished_at = func.now()
        if usage is not None:
            attempt.usage = _jsonable(usage.model_dump(mode="json"))
            attempt.input_tokens = usage.input_tokens
            attempt.output_tokens = usage.output_tokens
            attempt.cost_usd = usage.cost_usd
        experiment = session.get(Experiment, lease.experiment_id, with_for_update=_EXPERIMENT_LOCK)
        _append_lifecycle(session, attempt.id, "attempt_failed", {"error_code": error_code, "message": message})
        _after_infrastructure_error(
            session, job, run, experiment, message=message, retryable=retryable, backoff_seconds=retry_backoff_seconds
        )
        _settle_experiment(session, experiment)


def sweep_expired_leases(sm: sessionmaker[Session], *, retry_backoff_seconds: float = 0.0, experiment_id: uuid.UUID | None = None) -> list[uuid.UUID]:
    """Mark attempts whose worker stopped heartbeating as interrupted and hand their runs on.

    One job per transaction, so two sweepers never lock experiments in different orders.
    """
    swept: list[uuid.UUID] = []
    while True:
        with sm.begin() as session:
            job = session.scalar(
                select(Job)
                .where(Job.status == "leased", Job.lease_expires_at < func.now())
                .where(Job.run_id.in_(select(Run.id).where(Run.experiment_id == experiment_id))
                       if experiment_id is not None else True)
                .order_by(Job.lease_expires_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if job is None:
                return swept
            run = session.get(Run, job.run_id)
            experiment = session.get(Experiment, run.experiment_id, with_for_update=_EXPERIMENT_LOCK)
            attempt = session.get(Attempt, job.current_attempt_id)
            details = {
                "worker_id": job.leased_by,
                "lease_expires_at": job.lease_expires_at.isoformat(),
                "last_heartbeat_at": job.heartbeat_at.isoformat() if job.heartbeat_at else None,
            }
            attempt.status = "infrastructure_error"
            attempt.error_code = INTERRUPTED
            attempt.error = details
            attempt.finished_at = func.now()
            _append_lifecycle(session, attempt.id, "attempt_interrupted", {"reason": "lease_expired", **details})
            _after_infrastructure_error(
                session, job, run, experiment,
                message=f"worker {job.leased_by} stopped heartbeating",
                retryable=True,
                backoff_seconds=retry_backoff_seconds,
            )
            _settle_experiment(session, experiment)
            swept.append(job.id)


# --- shared steps --------------------------------------------------------------------


def _locked_job(session: Session, lease: Lease) -> Job:
    job = session.scalar(
        select(Job)
        .where(Job.id == lease.job_id, Job.lease_token == lease.token, Job.status == "leased")
        .with_for_update()
    )
    # Check the wall clock *after* obtaining the row lock: transaction-start now()
    # could predate a long lock wait. An expired token must never be resurrected.
    now = session.scalar(select(func.clock_timestamp()))
    if job is None or job.lease_expires_at is None or job.lease_expires_at <= now:
        raise LeaseLost(f"lease on job {lease.job_id} is no longer held by {lease.worker_id}")
    return job


def _after_infrastructure_error(
    session: Session,
    job: Job,
    run: Run,
    experiment: Experiment,
    *,
    message: str,
    retryable: bool,
    backoff_seconds: float,
) -> None:
    """Infra failures (never agent behaviour) get a new attempt in a fresh environment, up to max_attempts."""
    job.last_error = _text(message, 2000)
    if experiment.cancel_requested_at is not None:
        _close(job, run, run_status="cancelled", job_status="cancelled", evaluation_status="not_applicable")
    elif retryable and job.attempts_started < job.max_attempts:
        replacement = _new_attempt(session, run.id)
        job.status = "queued"
        job.current_attempt_id = replacement.id
        job.available_at = func.now() + timedelta(seconds=backoff_seconds)
        _release(job)
        run.status = "queued"
    else:
        _close(job, run, run_status="infrastructure_error", job_status="failed", evaluation_status="not_applicable")


def _close(
    job: Job,
    run: Run,
    *,
    run_status: str,
    job_status: str,
    evaluation_status: str,
    success: bool | None = None,
) -> None:
    job.status = job_status
    _release(job)
    run.status = run_status
    run.evaluation_status = evaluation_status
    run.success = success
    run.finished_at = func.now()


def _release(job: Job) -> None:
    job.lease_token = None
    job.lease_expires_at = None
    job.leased_by = None


def _settle_experiment(session: Session, experiment: Experiment) -> None:
    """Close the experiment once no run is queued or running. Caller holds the experiment lock."""
    session.flush()
    active = session.scalar(
        select(func.count())
        .select_from(Run)
        .where(Run.experiment_id == experiment.id, Run.status.in_(("queued", "running")))
    )
    if active == 0 and experiment.status not in ("completed", "cancelled"):
        experiment.status = "cancelled" if experiment.cancel_requested_at is not None else "completed"
        experiment.completed_at = func.now()


def _retryable(result: AttemptResult) -> bool:
    """Only a provider error the provider itself calls transient (429, 5xx, timeouts) gets a new attempt.

    An adapter exception re-runs agent code that most likely fails the same way
    (and may bill again), and a 4xx or a missing API key never fixes itself.
    """
    agent = result.agent_result
    return agent is not None and agent.stop_reason == "provider_error" and agent.metadata.get("retryable") is True


def evaluation_status_for(result: AttemptResult) -> str:
    """A graded attempt's run status: pass, fail, or evaluation_error when an evaluator could not decide."""
    if result.success is True:
        return "pass"
    if result.success is False:
        return "fail"
    return "evaluation_error"


def _append_lifecycle(session: Session, attempt_id: uuid.UUID, name: str, payload: dict[str, Any]) -> None:
    last = session.scalar(select(func.max(TraceEvent.event_sequence)).where(TraceEvent.attempt_id == attempt_id)) or 0
    session.add(
        TraceEvent(
            attempt_id=attempt_id,
            event_sequence=last + 1,
            kind="lifecycle",
            name=name,
            payload=_jsonable(payload),
            recorded_at=datetime.now(UTC),
        )
    )


def _persist(
    session: Session, attempt: Attempt, run: Run, task: TaskVersion, result: AttemptResult, duration_ms: float
) -> None:
    agent = result.agent_result
    usage = result.usage
    attempt.status = result.execution_status
    attempt.stop_reason = agent.stop_reason if agent else None
    attempt.final_response = _text(agent.final_response) if agent else None
    attempt.agent_metadata = _jsonable(agent.metadata) if agent else None
    attempt.initial_snapshot = _jsonable(result.initial_snapshot)
    attempt.final_snapshot = _jsonable(result.final_snapshot)
    attempt.usage = _jsonable(usage.model_dump(mode="json"))
    attempt.input_tokens = usage.input_tokens
    attempt.output_tokens = usage.output_tokens
    attempt.cost_usd = usage.cost_usd
    attempt.duration_ms = duration_ms
    attempt.finished_at = func.now()
    if result.execution_status == "infrastructure_error":
        failure = next((e.payload for e in result.events if e.name == "attempt_failed"), None)
        if failure is not None:
            attempt.error_code = "ADAPTER_EXCEPTION"
            attempt.error = _jsonable(failure)
        else:
            attempt.error_code = "PROVIDER_ERROR"
            attempt.error = _jsonable({"message": (agent.metadata.get("error") if agent else None) or "provider error"})

    session.add_all(
        TraceEvent(
            attempt_id=attempt.id,
            event_sequence=e.sequence,
            kind=e.kind,
            name=_text(e.name, 200),
            payload=_jsonable(e.payload),
            recorded_at=e.recorded_at,
            monotonic_ns=e.monotonic_ns,
        )
        for e in result.events
    )
    session.add_all(_model_calls(session, attempt.id, result))
    session.add_all(_tool_calls(attempt.id, result))
    session.add_all(
        Score(
            attempt_id=attempt.id,
            run_id=run.id,
            experiment_id=run.experiment_id,
            task_category=task.category,
            evaluator_name=s.evaluator_name,
            evaluator_version=s.evaluator_version,
            dimension="overall",
            status=s.status,
            value=s.value,
            reason_code=s.reason_code,
            evidence=_jsonable(s.evidence),
            applicability=s.applicability,
        )
        for s in result.scores
    )


def _model_calls(session: Session, attempt_id: uuid.UUID, result: AttemptResult) -> list[ModelCall]:
    calls = []
    for e in result.events:
        if e.kind != "model_response":
            continue
        p = e.payload
        usage = p.get("usage") or {}
        calls.append(
            ModelCall(
                attempt_id=attempt_id,
                event_sequence=e.sequence,
                turn=p.get("turn"),
                model=_text(e.name, 200),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cache_read_input_tokens=usage.get("cache_read_input_tokens"),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
                cost_usd=Decimal(p["cost_usd"]) if p.get("cost_usd") is not None else None,
                price_version_id=price_version_pk(session, p.get("price_version_id"), p.get("priced_model")),
                latency_ms=p.get("latency_ms"),
                stop_reason=p.get("stop_reason"),
                error=_text(p.get("error")),
            )
        )
    return calls


def _tool_calls(attempt_id: uuid.UUID, result: AttemptResult) -> list[ToolCall]:
    decisions = {e.payload["attempt_sequence"]: e.payload for e in result.events if e.kind == "tool_decision"}
    effects = {e.payload["attempt_sequence"]: e.payload for e in result.events if e.kind == "tool_effect"}
    calls = []
    for e in result.events:
        if e.kind != "tool_attempt":
            continue
        decision = decisions.get(e.sequence, {})
        effect = effects.get(e.sequence)
        calls.append(
            ToolCall(
                attempt_id=attempt_id,
                attempt_sequence=e.sequence,
                call_id=_text(e.payload.get("call_id"), 200),
                tool=_text(e.name, 100),
                args=_jsonable(e.payload.get("args")),
                decision=decision.get("decision"),
                reason_code=decision.get("reason_code"),
                effect_status=effect.get("status") if effect else None,
                state_changes=_jsonable(effect.get("state_changes")) if effect else None,
            )
        )
    return calls
