"""Experiments: planned in one transaction, idempotent by key, cancellable, summarised as a scorecard."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.accounting import PRICE_VERSIONS
from app.db.models import (
    AgentConfig,
    Attempt,
    Dataset,
    DatasetVersion,
    Experiment,
    Job,
    Run,
    Score,
    TaskVersion,
    TraceEvent,
)
from app.environment import ENVIRONMENT_VERSION
from app.evaluators import DEFAULT_EVALUATORS
from app.services.catalog import tool_schema_hash
from app.services.errors import Conflict, InvalidRequest, NotFound
from app.services.hashing import sha256_json, sha256_text

_ROOT = Path(__file__).resolve().parents[2]
_ACTIVE = ("queued", "running")
_EXPERIMENT_LOCK = {"key_share": True}  # FOR NO KEY UPDATE; see app/worker/queue.py


class ExperimentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=200)
    dataset_version_id: uuid.UUID
    agent_config_ids: list[uuid.UUID] = Field(min_length=1, max_length=20)
    split: Literal["dev", "validation", "test"]
    repetitions: int = Field(default=1, ge=1, le=10)

    @field_validator("agent_config_ids")
    @classmethod
    def _unique(cls, ids: list[uuid.UUID]) -> list[uuid.UUID]:
        if len(set(ids)) != len(ids):
            raise ValueError("agent_config_ids must be unique")
        return ids

    def request_hash(self) -> str:
        body = self.model_dump(mode="json")
        body["agent_config_ids"] = sorted(body["agent_config_ids"])  # the plan, not the listing order
        return sha256_json(body)


# --- creation ------------------------------------------------------------------------


def create_experiment(
    sm: sessionmaker[Session], payload: ExperimentCreate, idempotency_key: str
) -> tuple[Experiment, bool]:
    """Create experiment, runs and jobs in one transaction.

    The same key with the same payload returns the original experiment
    (``created`` is False); with a different payload it is a ``Conflict``.
    Concurrent requests with one key race on the unique index; losers replay.
    """
    if not idempotency_key or len(idempotency_key) > 200:
        raise InvalidRequest("Idempotency-Key must be 1-200 characters")
    request_hash = payload.request_hash()
    with sm() as session:
        existing = session.scalar(select(Experiment).where(Experiment.idempotency_key == idempotency_key))
    if existing is not None:
        return _replay(existing, request_hash), False
    try:
        with sm.begin() as session:
            experiment = _plan(session, payload, idempotency_key, request_hash)
    except IntegrityError:
        with sm() as session:
            existing = session.scalar(select(Experiment).where(Experiment.idempotency_key == idempotency_key))
        if existing is None:
            raise
        return _replay(existing, request_hash), False
    return experiment, True


def _replay(existing: Experiment, request_hash: str) -> Experiment:
    if existing.request_hash != request_hash:
        raise Conflict("this Idempotency-Key was already used with a different request")
    return existing


def _plan(session: Session, payload: ExperimentCreate, key: str, request_hash: str) -> Experiment:
    version = session.get(DatasetVersion, payload.dataset_version_id)
    if version is None:
        raise InvalidRequest(f"dataset version {payload.dataset_version_id} does not exist")
    found = {c.id: c for c in session.scalars(select(AgentConfig).where(AgentConfig.id.in_(payload.agent_config_ids)))}
    missing = [str(i) for i in payload.agent_config_ids if i not in found]
    if missing:
        raise InvalidRequest(f"agent configurations do not exist: {missing}")
    configs = [found[i] for i in payload.agent_config_ids]
    tasks = session.scalars(
        select(TaskVersion)
        .where(TaskVersion.dataset_version_id == version.id, TaskVersion.split == payload.split)
        .order_by(TaskVersion.task_id)
    ).all()
    if not tasks:
        raise InvalidRequest(f"split {payload.split!r} has no tasks in this dataset version")
    dataset = session.get(Dataset, version.dataset_id)

    experiment = Experiment(
        id=uuid.uuid4(),
        name=payload.name,
        dataset_version_id=version.id,
        split=payload.split,
        repetitions=payload.repetitions,
        status="queued",
        idempotency_key=key,
        request_hash=request_hash,
        manifest=_manifest(dataset, version, payload, tasks, configs),
    )
    session.add(experiment)
    session.flush()  # a duplicate key surfaces here, before any runs are written

    runs = [
        Run(
            id=uuid.uuid4(),
            experiment_id=experiment.id,
            agent_config_id=config.id,
            task_version_id=task.id,
            repetition=repetition,
            status="queued",
            evaluation_status="pending",
        )
        for config in configs
        for task in tasks
        for repetition in range(1, payload.repetitions + 1)
    ]
    session.add_all(runs)
    session.flush()
    session.add_all([_job_for(run) for run in runs])
    session.flush()
    return experiment


def _job_for(run: Run) -> Job:
    return Job(id=uuid.uuid4(), run_id=run.id, status="queued")


def _manifest(
    dataset: Dataset,
    version: DatasetVersion,
    payload: ExperimentCreate,
    tasks: list[TaskVersion],
    configs: list[AgentConfig],
) -> dict[str, Any]:
    """Everything needed to reproduce the experiment, frozen at creation."""
    models = {c.model for c in configs if c.model}
    return {
        "schema_version": 1,
        "dataset": {
            "name": dataset.name,
            "dataset_version_id": str(version.id),
            "content_hash": version.content_hash,
            "split_seed": version.split_seed,
        },
        "split": payload.split,
        "repetitions": payload.repetitions,
        "tasks": [
            {"task_version_id": str(t.id), "task_id": t.task_id, "content_hash": t.content_hash} for t in tasks
        ],
        "agent_configs": [
            {
                "id": str(c.id),
                "name": c.name,
                "adapter": c.adapter,
                "model": c.model,
                "config_hash": c.config_hash,
                "prompt_hash": c.prompt_hash,
                "tools": c.tools,
                "tool_schema_hash": tool_schema_hash(c.tools),
                "limits": c.limits,
                "settings": c.settings,
            }
            for c in configs
        ],
        "evaluators": [{"name": e.name, "version": e.version} for e in DEFAULT_EVALUATORS],
        "environment": {"name": "support_environment", "version": ENVIRONMENT_VERSION},
        "fault_config": fault_config(tasks),
        "price_versions": [
            {"price_version_id": p.price_version_id, "model": p.model} for p in PRICE_VERSIONS if p.model in models
        ],
        "code": _code_identity(),
    }


def fault_config(tasks: Iterable[TaskVersion]) -> dict[str, dict[str, str]]:
    """Injected faults per task (``{task_id: {"tool:resource": "timeout"}}``); tasks without faults are omitted."""
    faults = {t.task_id: (t.evaluator_view.get("fixture") or {}).get("faults") or {} for t in tasks}
    return {task_id: f for task_id, f in sorted(faults.items()) if f}


def _code_identity() -> dict[str, Any]:
    try:
        platform_version = metadata.version("agent-eval-platform")
    except metadata.PackageNotFoundError:
        platform_version = None
    pyproject = _ROOT / "pyproject.toml"
    lockfile = _ROOT / "uv.lock"
    return {
        "platform_version": platform_version,
        "git_commit": os.environ.get("GIT_COMMIT"),  # the repo may not be under git; CI sets this
        "pyproject_sha256": sha256_text(pyproject.read_text()) if pyproject.is_file() else None,
        "lockfile_sha256": sha256_text(lockfile.read_text()) if lockfile.is_file() else None,
    }


# --- cancellation --------------------------------------------------------------------


def cancel_experiment(sm: sessionmaker[Session], experiment_id: uuid.UUID) -> Experiment:
    """Cancel queued runs now; running attempts stop at their next tool call.

    Lock order (experiment, then queued jobs) never waits on a leased job, so it
    cannot deadlock with workers, which lock their job first and the experiment last.
    """
    now = func.now()  # the database clock, like every other queue timestamp
    with sm.begin() as session:
        experiment = session.get(Experiment, experiment_id, with_for_update=_EXPERIMENT_LOCK)
        if experiment is None:
            raise NotFound(f"experiment {experiment_id} not found")
        if experiment.status == "completed":
            raise Conflict("experiment already completed")
        if experiment.cancel_requested_at is None:
            experiment.cancel_requested_at = now
        queued = session.execute(
            select(Job, Run)
            .join(Run, Run.id == Job.run_id)
            .where(Run.experiment_id == experiment_id, Job.status == "queued")
            .with_for_update(of=Job)
        ).all()
        for job, run in queued:
            job.status = "cancelled"
            run.status = "cancelled"
            run.evaluation_status = "not_applicable"
            run.finished_at = now
            if job.current_attempt_id is not None:
                attempt = session.get(Attempt, job.current_attempt_id)
                if attempt is not None and attempt.status == "queued":
                    attempt.status = "cancelled"
                    attempt.finished_at = now
        session.flush()
        running = session.scalar(
            select(func.count()).select_from(Run).where(Run.experiment_id == experiment_id, Run.status == "running")
        )
        if running:
            experiment.status = "cancelling"
        elif experiment.status != "cancelled":
            experiment.status = "cancelled"
            experiment.completed_at = now
    return experiment


# --- read models ---------------------------------------------------------------------


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _money(value: Decimal | None) -> str | None:
    """Plain decimal string ("0", "0.0024"); NUMERIC columns read back as e.g. Decimal('0E-8')."""
    return format(value.normalize(), "f") if value is not None else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def experiment_summary(experiment: Experiment, planned: int) -> dict[str, Any]:
    return {
        "id": str(experiment.id),
        "name": experiment.name,
        "status": experiment.status,
        "dataset_version_id": str(experiment.dataset_version_id),
        "split": experiment.split,
        "repetitions": experiment.repetitions,
        "planned_runs": planned,
        "created_at": _iso(experiment.created_at),
        "cancel_requested_at": _iso(experiment.cancel_requested_at),
        "completed_at": _iso(experiment.completed_at),
    }


def list_experiments(session: Session, *, limit: int, offset: int) -> list[dict[str, Any]]:
    planned = select(func.count()).where(Run.experiment_id == Experiment.id).correlate(Experiment).scalar_subquery()
    rows = session.execute(
        select(Experiment, planned)
        .order_by(Experiment.created_at.desc(), Experiment.id)
        .limit(limit)
        .offset(offset)
    ).all()
    return [experiment_summary(experiment, n) for experiment, n in rows]


def experiment_detail(session: Session, experiment_id: uuid.UUID) -> dict[str, Any]:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise NotFound(f"experiment {experiment_id} not found")
    in_experiment = Run.experiment_id == experiment_id

    by_status = dict(session.execute(select(Run.status, func.count()).where(in_experiment).group_by(Run.status)).all())
    by_evaluation = dict(
        session.execute(
            select(Run.evaluation_status, func.count()).where(in_experiment).group_by(Run.evaluation_status)
        ).all()
    )
    planned = sum(by_status.values())
    successful = by_evaluation.get("pass", 0)
    failed = by_evaluation.get("fail", 0)
    scored = successful + failed

    evaluators: dict[str, dict[str, int]] = {}
    for name, status, count in session.execute(
        select(Score.evaluator_name, Score.status, func.count())
        .where(Score.experiment_id == experiment_id)
        .group_by(Score.evaluator_name, Score.status)
    ):
        evaluators.setdefault(name, {})[status] = count

    costs = session.execute(
        select(Attempt.cost_usd)
        .join(Run, Run.id == Attempt.run_id)
        # Attempts that never started (e.g. a queued retry that was cancelled) spent nothing.
        .where(in_experiment, Attempt.status.not_in(_ACTIVE), Attempt.started_at.is_not(None))
    ).scalars().all()
    unknown_cost = sum(1 for c in costs if c is None)
    total_cost = None if unknown_cost else sum(costs, start=Decimal("0"))

    per_config = session.execute(
        select(
            AgentConfig.id,
            AgentConfig.name,
            func.count(),
            func.count().filter(Run.evaluation_status.in_(("pass", "fail"))),
            func.count().filter(Run.evaluation_status == "pass"),
        )
        .join(Run, Run.agent_config_id == AgentConfig.id)
        .where(in_experiment)
        .group_by(AgentConfig.id, AgentConfig.name)
        .order_by(AgentConfig.name)
    ).all()

    return {
        **experiment_summary(experiment, planned),
        "manifest": experiment.manifest,
        "progress": {
            "planned_runs": planned,
            "by_status": by_status,
            "by_evaluation_status": by_evaluation,
        },
        "scorecard": {
            "planned_runs": planned,
            "scored_runs": scored,
            "successful_runs": successful,
            "failed_runs": failed,
            "evaluation_errors": by_evaluation.get("evaluation_error", 0),
            "verified_success_rate": _rate(successful, scored),
            "strict_success_rate": _rate(successful, planned),
            "evaluation_coverage": _rate(scored, planned),
            "evaluators": evaluators,
            # Unknown, never zero, if any finished attempt's usage is unknown (e.g. an interrupted one).
            "cost_usd": _money(total_cost),
            "attempts_with_unknown_cost": unknown_cost,
            "by_agent_config": [
                {
                    "agent_config_id": str(config_id),
                    "name": name,
                    "planned_runs": n,
                    "scored_runs": n_scored,
                    "successful_runs": n_pass,
                    "verified_success_rate": _rate(n_pass, n_scored),
                }
                for config_id, name, n, n_scored, n_pass in per_config
            ],
        },
    }


def list_runs(
    session: Session,
    experiment_id: uuid.UUID,
    *,
    filters: dict[str, Any],
    limit: int,
    offset: int,
) -> dict[str, Any]:
    if session.get(Experiment, experiment_id) is None:
        raise NotFound(f"experiment {experiment_id} not found")
    attempts = select(func.count()).where(Attempt.run_id == Run.id).correlate(Run).scalar_subquery()
    conditions = [Run.experiment_id == experiment_id]
    column_filters = {
        "status": Run.status,
        "evaluation_status": Run.evaluation_status,
        "success": Run.success,
        "agent_config_id": Run.agent_config_id,
        "category": TaskVersion.category,
        "task_id": TaskVersion.task_id,
    }
    for key, column in column_filters.items():
        if filters.get(key) is not None:
            conditions.append(column == filters[key])
    base = (
        select(Run, TaskVersion.task_id, TaskVersion.category, AgentConfig.name, attempts)
        .join(TaskVersion, TaskVersion.id == Run.task_version_id)
        .join(AgentConfig, AgentConfig.id == Run.agent_config_id)
        .where(*conditions)
    )
    total = session.scalar(select(func.count()).select_from(base.subquery()))
    rows = session.execute(
        base.order_by(AgentConfig.name, TaskVersion.task_id, Run.repetition).limit(limit).offset(offset)
    ).all()
    return {
        "items": [_run_row(run, task_id, category, config_name, n) for run, task_id, category, config_name, n in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def _run_row(run: Run, task_id: str, category: str, config_name: str, attempt_count: int) -> dict[str, Any]:
    return {
        "id": str(run.id),
        "experiment_id": str(run.experiment_id),
        "agent_config_id": str(run.agent_config_id),
        "agent_config_name": config_name,
        "task_version_id": str(run.task_version_id),
        "task_id": task_id,
        "category": category,
        "repetition": run.repetition,
        "status": run.status,
        "evaluation_status": run.evaluation_status,
        "success": run.success,
        "attempt_count": attempt_count,
        "created_at": _iso(run.created_at),
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
    }


def run_detail(session: Session, run_id: uuid.UUID) -> dict[str, Any]:
    run = session.get(Run, run_id)
    if run is None:
        raise NotFound(f"run {run_id} not found")
    task = session.get(TaskVersion, run.task_version_id)
    config = session.get(AgentConfig, run.agent_config_id)
    attempts = session.scalars(
        select(Attempt).where(Attempt.run_id == run_id).order_by(Attempt.attempt_number)
    ).all()
    scores = session.scalars(select(Score).where(Score.run_id == run_id).order_by(Score.id)).all()
    return {
        **_run_row(run, task.task_id, task.category, config.name, len(attempts)),
        "task": {
            "task_version_id": str(task.id),
            "task_id": task.task_id,
            "family": task.family,
            "category": task.category,
            "split": task.split,
            "user_request": task.agent_view["user_request"],
        },
        "attempts": [
            {
                "id": str(a.id),
                "attempt_number": a.attempt_number,
                "status": a.status,
                "environment_id": a.environment_id,
                "worker_id": a.worker_id,
                "error_code": a.error_code,
                "error": a.error,
                "stop_reason": a.stop_reason,
                "final_response": a.final_response,
                "usage": a.usage,
                "cost_usd": _money(a.cost_usd),
                "duration_ms": a.duration_ms,
                "created_at": _iso(a.created_at),
                "started_at": _iso(a.started_at),
                "finished_at": _iso(a.finished_at),
            }
            for a in attempts
        ],
        "scores": [
            {
                "attempt_id": str(s.attempt_id),
                "evaluator_name": s.evaluator_name,
                "evaluator_version": s.evaluator_version,
                "dimension": s.dimension,
                "status": s.status,
                "value": s.value,
                "reason_code": s.reason_code,
                "evidence": s.evidence,
                "applicability": s.applicability,
            }
            for s in scores
        ],
    }


def attempt_events(session: Session, attempt_id: uuid.UUID, *, after_sequence: int, limit: int) -> dict[str, Any]:
    if session.get(Attempt, attempt_id) is None:
        raise NotFound(f"attempt {attempt_id} not found")
    events = session.scalars(
        select(TraceEvent)
        .where(TraceEvent.attempt_id == attempt_id, TraceEvent.event_sequence > after_sequence)
        .order_by(TraceEvent.event_sequence)
        .limit(limit)
    ).all()
    return {
        "attempt_id": str(attempt_id),
        "items": [
            {
                "sequence": e.event_sequence,
                "kind": e.kind,
                "name": e.name,
                "payload": e.payload,
                "recorded_at": _iso(e.recorded_at),
            }
            for e in events
        ],
        "next_after_sequence": events[-1].event_sequence if len(events) == limit else None,
    }
