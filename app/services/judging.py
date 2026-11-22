"""Run the optional communication judge over an experiment's graded attempts, and summarise it apart.

Only attempts the deterministic evaluators graded and that produced a final
message are judged, once per judge version (re-running skips them). Judge rows
never touch runs, attempts or scores, so run success, evaluation status,
scorecards, comparisons and candidate cost are identical with or without it.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Attempt, Experiment, JudgeScore, Run, Score, TaskVersion
from app.judges.communication import CommunicationJudge, JudgeInput, JudgeRow, summarize
from app.services.errors import NotFound


@dataclass(frozen=True)
class _Item:
    attempt_id: uuid.UUID
    run_id: uuid.UUID
    agent_config_id: uuid.UUID
    judge_input: JudgeInput


def _judgeable(session: Session, experiment_id: uuid.UUID) -> list[_Item]:
    """Graded attempts with a final message, in a stable order."""
    graded = select(Score.attempt_id).where(Score.experiment_id == experiment_id).distinct()
    rows = session.execute(
        select(Attempt.id, Attempt.run_id, Run.agent_config_id, Attempt.final_response, TaskVersion.agent_view)
        .join(Run, Run.id == Attempt.run_id)
        .join(TaskVersion, TaskVersion.id == Run.task_version_id)
        .where(Run.experiment_id == experiment_id, Attempt.final_response.is_not(None), Attempt.id.in_(graded))
        .order_by(Run.agent_config_id, TaskVersion.task_id, Run.repetition, Attempt.attempt_number)
    ).all()
    outcomes = {
        attempt_id: evidence.get("actual")
        for attempt_id, evidence in session.execute(
            select(Score.attempt_id, Score.evidence)
            .where(Score.experiment_id == experiment_id, Score.evaluator_name == "response")
        )
    }
    return [
        _Item(
            attempt_id=row.id,
            run_id=row.run_id,
            agent_config_id=row.agent_config_id,
            judge_input=JudgeInput(
                customer_request=row.agent_view.get("user_request", ""),
                agent_response=row.final_response,
                verified_outcome=outcomes.get(row.id),
            ),
        )
        for row in rows
    ]


def judge_experiment(
    sm: sessionmaker[Session], experiment_id: uuid.UUID, judge: CommunicationJudge, *, limit: int | None = None
) -> dict[str, int]:
    """Judge every not-yet-judged attempt. Each verdict commits on its own, so an interruption keeps the rest."""
    if limit is not None and limit < 1:
        raise ValueError("judge limit must be at least 1")
    with sm() as session:
        if session.get(Experiment, experiment_id) is None:
            raise NotFound(f"experiment {experiment_id} not found")
        items = _judgeable(session, experiment_id)
        done = set(session.scalars(
            select(JudgeScore.attempt_id).where(
                JudgeScore.experiment_id == experiment_id,
                JudgeScore.judge_name == judge.name,
                JudgeScore.judge_version == judge.version,
            )
        ))
    pending = [item for item in items if item.attempt_id not in done]
    if limit is not None:
        pending = pending[:limit]

    counts = {"eligible": len(items), "already_judged": len(done), "scored": 0, "evaluation_errors": 0}
    for item in pending:
        # The lock is scoped to the same identity as the unique constraint. Recheck
        # under it before calling the provider, so a concurrent CLI cannot spend twice.
        identity = f"{item.attempt_id}:{judge.name}:{judge.version}"
        lock_id = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big", signed=True)
        with sm.begin() as session:
            session.execute(select(func.pg_advisory_xact_lock(lock_id)))
            existing = session.scalar(select(JudgeScore.id).where(
                JudgeScore.attempt_id == item.attempt_id,
                JudgeScore.judge_name == judge.name,
                JudgeScore.judge_version == judge.version,
            ))
            if existing is not None:
                counts["already_judged"] += 1
                continue
            verdict = judge.judge(item.judge_input)
            usage = verdict.usage
            session.add(JudgeScore(
                attempt_id=item.attempt_id,
                run_id=item.run_id,
                experiment_id=experiment_id,
                judge_name=judge.name,
                judge_version=judge.version,
                rubric_sha256=judge.rubric_sha256,
                model=judge.model,
                resolved_model=verdict.resolved_model,
                status=verdict.status,
                scores={d: {**v, "reason": v["reason"].replace("\x00", "\ufffd")}
                        for d, v in verdict.scores.items()} if verdict.scores else None,
                reason_code=verdict.reason_code,
                error=verdict.error.replace("\x00", "\ufffd") if verdict.error else None,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                cache_read_input_tokens=usage.cache_read_input_tokens if usage else None,
                cache_creation_input_tokens=usage.cache_creation_input_tokens if usage else None,
                cost_usd=verdict.cost_usd,
                price_version_id=verdict.price_version_id,
                latency_ms=verdict.latency_ms,
            ))
        counts["scored" if verdict.status == "scored" else "evaluation_errors"] += 1
    return counts


def judge_summaries(session: Session, experiment_id: uuid.UUID) -> list[dict[str, Any]]:
    """One block per (configuration, judge version), in manifest order. Empty when never judged."""
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise NotFound(f"experiment {experiment_id} not found")
    rows = session.execute(
        select(JudgeScore, Run.agent_config_id)
        .join(Run, Run.id == JudgeScore.run_id)
        .where(JudgeScore.experiment_id == experiment_id)
        .order_by(JudgeScore.id)
    ).all()
    if not rows:
        return []
    eligible = Counter(str(item.agent_config_id) for item in _judgeable(session, experiment_id))
    grouped: dict[tuple[str, str, str], list[JudgeScore]] = defaultdict(list)
    for score, config_id in rows:
        grouped[(str(config_id), score.judge_name, score.judge_version)].append(score)

    configs = experiment.manifest.get("agent_configs", [])
    position = {c["id"]: i for i, c in enumerate(configs)}
    names = {c["id"]: c["name"] for c in configs}
    out = []
    for (config_id, judge_name, judge_version), scores in sorted(
        grouped.items(), key=lambda kv: (position.get(kv[0][0], len(configs)), kv[0])
    ):
        out.append({
            "agent_config_id": config_id,
            "agent_config_name": names.get(config_id),
            "judge": f"{judge_name}@{judge_version}",
            "models": sorted({s.model for s in scores}),
            "resolved_models": sorted({s.resolved_model for s in scores if s.resolved_model}),
            "rubric_sha256": sorted({s.rubric_sha256 for s in scores}),
            "error_reasons": dict(sorted(Counter(s.reason_code for s in scores if s.status == "evaluation_error").items())),
            **summarize(
                [JudgeRow(s.status, s.scores, s.input_tokens, s.output_tokens, s.cost_usd) for s in scores],
                eligible[config_id],
            ),
        })
    return out
