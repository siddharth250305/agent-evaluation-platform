"""Read runs, attempts and scores into ``app.metrics`` facts. Shared by comparisons, exports and the dashboard."""

from __future__ import annotations

import uuid
from collections import defaultdict
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import AgentConfig, Attempt, Experiment, Run, Score, TaskVersion
from app.metrics import AttemptFacts, Ratio, RunFacts, ScoreFacts, money

_ACTIVE = ("queued", "running")
_TERMINAL = ("completed", "timed_out", "budget_exceeded", "infrastructure_error", "cancelled")


def load_run_facts(
    session: Session, experiment_id: uuid.UUID, agent_config_id: uuid.UUID | None = None
) -> list[RunFacts]:
    """Every planned run of the experiment (optionally one configuration), ordered by config, task, repetition."""
    conditions = [Run.experiment_id == experiment_id]
    if agent_config_id is not None:
        conditions.append(Run.agent_config_id == agent_config_id)

    attempts: dict[uuid.UUID, list[tuple[Any, AttemptFacts]]] = defaultdict(list)
    for row in session.execute(
        select(
            Attempt.run_id, Attempt.attempt_number, Attempt.status, Attempt.started_at, Attempt.cost_usd,
            Attempt.input_tokens, Attempt.output_tokens, Attempt.duration_ms, Attempt.stop_reason, Attempt.error_code,
        )
        .join(Run, Run.id == Attempt.run_id)
        .where(*conditions)
        .order_by(Attempt.run_id, Attempt.attempt_number)
    ):
        attempts[row.run_id].append((row.started_at, AttemptFacts(
            attempt_number=row.attempt_number,
            status=row.status,
            started=row.started_at is not None,
            cost_usd=row.cost_usd,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            duration_ms=row.duration_ms,
            stop_reason=row.stop_reason,
            error_code=row.error_code,
        )))

    scores: dict[uuid.UUID, list[ScoreFacts]] = defaultdict(list)
    for row in session.execute(
        select(
            Score.run_id, Score.evaluator_name, Score.evaluator_version, Score.status, Score.value,
            Score.reason_code, Score.applicability, Score.evidence,
        )
        .join(Run, Run.id == Score.run_id)
        .where(*conditions)
        .order_by(Score.id)
    ):
        scores[row.run_id].append(ScoreFacts(
            evaluator_name=row.evaluator_name,
            evaluator_version=row.evaluator_version,
            status=row.status,
            value=row.value,
            reason_code=row.reason_code,
            applicability=row.applicability,
            evidence=row.evidence,
        ))

    facts = []
    for run, task_id, category, config_name in session.execute(
        select(Run, TaskVersion.task_id, TaskVersion.category, AgentConfig.name)
        .join(TaskVersion, TaskVersion.id == Run.task_version_id)
        .join(AgentConfig, AgentConfig.id == Run.agent_config_id)
        .where(*conditions)
        .order_by(AgentConfig.name, TaskVersion.task_id, Run.repetition)
    ):
        run_attempts = attempts.get(run.id, [])
        first_start = next((started for started, _ in run_attempts if started is not None), None)
        queue_ms = (first_start - run.created_at).total_seconds() * 1000 if first_start is not None else None
        facts.append(RunFacts(
            run_id=str(run.id),
            agent_config_id=str(run.agent_config_id),
            agent_config_name=config_name,
            task_id=task_id,
            task_version_id=str(run.task_version_id),
            category=category,
            repetition=run.repetition,
            status=run.status,
            evaluation_status=run.evaluation_status,
            success=run.success,
            queue_ms=queue_ms,
            attempts=tuple(a for _, a in run_attempts),
            scores=tuple(scores.get(run.id, ())),
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
        ))
    return facts


def experiment_rows(session: Session, *, limit: int = 100) -> list[dict[str, Any]]:
    """One summary row per experiment for the list screen, in a fixed number of queries."""
    experiments = session.scalars(
        select(Experiment).order_by(Experiment.created_at.desc(), Experiment.id).limit(limit)
    ).all()
    ids = [e.id for e in experiments]
    if not ids:
        return []
    counts = {
        row.experiment_id: row
        for row in session.execute(
            select(
                Run.experiment_id,
                func.count().label("planned"),
                func.count().filter(Run.status.in_(_TERMINAL)).label("finished"),
                func.count().filter(Run.evaluation_status.in_(("pass", "fail"))).label("scored"),
                func.count().filter(Run.evaluation_status == "pass").label("successful"),
                func.min(Run.started_at).label("started_at"),
            )
            .where(Run.experiment_id.in_(ids))
            .group_by(Run.experiment_id)
        )
    }
    costs: dict[uuid.UUID, list[Decimal | None]] = defaultdict(list)
    for experiment_id, cost in session.execute(
        select(Run.experiment_id, Attempt.cost_usd)
        .join(Run, Run.id == Attempt.run_id)
        .where(Run.experiment_id.in_(ids), Attempt.status.not_in(_ACTIVE), Attempt.started_at.is_not(None))
    ):
        costs[experiment_id].append(cost)

    rows = []
    for experiment in experiments:
        c = counts.get(experiment.id)
        planned, finished, scored, successful = (c.planned, c.finished, c.scored, c.successful) if c else (0, 0, 0, 0)
        spent = costs.get(experiment.id, [])
        unknown = sum(1 for v in spent if v is None)
        manifest = experiment.manifest
        rows.append({
            "id": str(experiment.id),
            "name": experiment.name,
            "status": experiment.status,
            "dataset": manifest.get("dataset", {}),
            "split": experiment.split,
            "repetitions": experiment.repetitions,
            "agent_configs": [
                {"id": cfg["id"], "name": cfg["name"], "model": cfg.get("model")}
                for cfg in manifest.get("agent_configs", [])
            ],
            "progress": Ratio(finished, planned).as_dict(),
            "verified_success_rate": Ratio(successful, scored).as_dict(),
            "strict_success_rate": Ratio(successful, planned).as_dict(),
            "evaluation_coverage": Ratio(scored, planned).as_dict(),
            "cost_usd": None if unknown else money(sum((v for v in spent if v is not None), Decimal("0"))),
            "attempts_with_unknown_cost": unknown,
            "created_at": experiment.created_at,
            "started_at": c.started_at if c else None,
        })
    return rows
