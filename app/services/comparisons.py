"""Compare two (experiment, agent configuration) pairs task by task.

Only compatible pairs are compared: same dataset version and split, same
environment version, same evaluator versions and the same limits. Anything
else would compare different problems, different graders or different budgets.
The comparison is computed on request and not stored; it is deterministic for
a given seed, so the same request over the same finished runs gives the same answer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.db.models import Experiment
from app.metrics import RunFacts, compare
from app.services.errors import InvalidRequest, NotFound
from app.services.results import load_run_facts

_CONFIG_FIELDS = ("adapter", "model", "prompt_hash", "tools", "tool_schema_hash", "settings", "config_hash")


class ComparisonSide(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: uuid.UUID
    agent_config_id: uuid.UUID | None = Field(
        default=None, description="Required when the experiment has more than one agent configuration"
    )


class ComparisonCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline: ComparisonSide
    candidate: ComparisonSide
    bootstrap_resamples: int = Field(default=2000, ge=0, le=20000)
    seed: int = 0
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)


@dataclass(frozen=True)
class _Side:
    role: str
    experiment: Experiment
    config: dict[str, Any]  # the configuration as frozen in the experiment manifest

    @property
    def evaluators(self) -> list[tuple[str, str]]:
        return sorted((e["name"], e["version"]) for e in self.experiment.manifest.get("evaluators", []))

    @property
    def environment_version(self) -> str | None:
        return self.experiment.manifest.get("environment", {}).get("version")


def _resolve(session: Session, side: ComparisonSide, role: str) -> _Side:
    experiment = session.get(Experiment, side.experiment_id)
    if experiment is None:
        raise NotFound(f"{role} experiment {side.experiment_id} not found")
    configs = experiment.manifest.get("agent_configs", [])
    if side.agent_config_id is None:
        if len(configs) != 1:
            raise InvalidRequest(
                f"{role} experiment {experiment.id} has {len(configs)} agent configurations; "
                f"choose one with {role}.agent_config_id"
            )
        return _Side(role, experiment, configs[0])
    config = next((c for c in configs if c["id"] == str(side.agent_config_id)), None)
    if config is None:
        raise InvalidRequest(f"agent configuration {side.agent_config_id} is not part of {role} experiment {experiment.id}")
    return _Side(role, experiment, config)


def incompatibilities(baseline: _Side, candidate: _Side) -> list[str]:
    b, c = baseline, candidate
    checks = (
        ("dataset version", str(b.experiment.dataset_version_id), str(c.experiment.dataset_version_id)),
        ("split", b.experiment.split, c.experiment.split),
        ("environment version", b.environment_version, c.environment_version),
        ("evaluator versions", b.evaluators, c.evaluators),
        ("limits", b.config.get("limits"), c.config.get("limits")),
    )
    return [f"{name} differs (baseline {bv!r}, candidate {cv!r})" for name, bv, cv in checks if bv != cv]


def _side_out(side: _Side, runs: list[RunFacts]) -> dict[str, Any]:
    unfinished = sum(not r.finished for r in runs)
    return {
        "experiment_id": str(side.experiment.id),
        "experiment_name": side.experiment.name,
        "experiment_status": side.experiment.status,
        "agent_config_id": side.config["id"],
        "agent_config_name": side.config["name"],
        "adapter": side.config.get("adapter"),
        "model": side.config.get("model"),
        "prompt_hash": side.config.get("prompt_hash"),
        "repetitions": side.experiment.repetitions,
        "planned_runs": len(runs),
        "unfinished_runs": unfinished,
        "complete": unfinished == 0,
    }


def compare_experiments(session: Session, payload: ComparisonCreate) -> dict[str, Any]:
    baseline = _resolve(session, payload.baseline, "baseline")
    candidate = _resolve(session, payload.candidate, "candidate")
    if baseline.experiment.id == candidate.experiment.id and baseline.config["id"] == candidate.config["id"]:
        raise InvalidRequest("baseline and candidate are the same configuration in the same experiment")
    problems = incompatibilities(baseline, candidate)
    if problems:
        raise InvalidRequest("experiments are not comparable: " + "; ".join(problems))

    base_runs = load_run_facts(session, baseline.experiment.id, uuid.UUID(baseline.config["id"]))
    cand_runs = load_run_facts(session, candidate.experiment.id, uuid.UUID(candidate.config["id"]))
    result = compare(
        base_runs,
        cand_runs,
        resamples=payload.bootstrap_resamples,
        seed=payload.seed,
        confidence=payload.confidence,
    )

    warnings = [
        f"{side.role} has {out['unfinished_runs']} unfinished runs; strict metrics count them as unsuccessful"
        for side, out in ((baseline, _side_out(baseline, base_runs)), (candidate, _side_out(candidate, cand_runs)))
        if out["unfinished_runs"]
    ]
    if baseline.experiment.repetitions != candidate.experiment.repetitions:
        warnings.append("repetitions differ; per-task pass rates are compared as fractions of each side's repetitions")

    return {
        "baseline": _side_out(baseline, base_runs),
        "candidate": _side_out(candidate, cand_runs),
        "compatibility": {
            "dataset_version_id": str(baseline.experiment.dataset_version_id),
            "split": baseline.experiment.split,
            "environment_version": baseline.environment_version,
            "evaluator_versions": [f"{name}@{version}" for name, version in baseline.evaluators],
            "limits": baseline.config.get("limits"),
        },
        "config_differences": {
            name: {"baseline": baseline.config.get(name), "candidate": candidate.config.get(name)}
            for name in _CONFIG_FIELDS
            if baseline.config.get(name) != candidate.config.get(name)
        },
        "warnings": warnings,
        **result,
    }
