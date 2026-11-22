"""Experiment export: per-run results bundled with the immutable manifest, as JSON or CSV.

Anyone holding an export can tell exactly what produced it: the stored manifest
(dataset, environment and evaluator versions, prompt and tool-schema hashes,
code identity, model settings, fault config) travels with every file, and every
CSV row repeats the identifying hashes. Unknown values are null / empty, never zero.
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Attempt, Experiment, ModelCall, Run, TaskVersion
from app.metrics import RunFacts, money, scorecard
from app.services.errors import NotFound
from app.services.experiments import experiment_summary, fault_config
from app.services.hashing import sha256_json
from app.services.judging import judge_summaries
from app.services.results import load_run_facts

EXPORT_VERSION = 1

_RUN_COLUMNS = (
    "run_id", "experiment_id", "agent_config_id", "agent_config_name", "model", "task_id", "task_version_id",
    "category", "split", "repetition", "status", "evaluation_status", "success", "attempts", "retry_attempts",
    "stop_reason", "error_code", "reason_codes",
)
_TOOL_COLUMNS = (
    ("tool_calls_attempted", "attempted"), ("tool_calls_decided", "decided"),
    ("tool_calls_schema_valid", "schema_valid"), ("tool_calls_matched", "matched"),
    ("required_calls", "required"), ("required_calls_satisfied", "required_satisfied"),
)
_USAGE_COLUMNS = (
    "queue_ms", "execution_ms", "input_tokens", "output_tokens", "cost_usd", "created_at", "started_at", "finished_at",
)
_MANIFEST_COLUMNS = (
    "manifest_sha256", "dataset_name", "dataset_content_hash", "environment_version", "evaluator_versions",
    "agent_config_hash", "prompt_hash", "tool_schema_hash", "git_commit", "lockfile_sha256",
)
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _call_counts(run: RunFacts) -> dict[str, int] | None:
    trajectory = run.score("trajectory")
    counts = trajectory.evidence.get("call_counts") if trajectory is not None else None
    return counts if isinstance(counts, dict) else None


def _run_out(run: RunFacts) -> dict[str, Any]:
    result = run.result_attempt
    counted = run.counted_attempts
    return {
        "run_id": run.run_id,
        "agent_config_id": run.agent_config_id,
        "agent_config_name": run.agent_config_name,
        "task_id": run.task_id,
        "task_version_id": run.task_version_id,
        "category": run.category,
        "repetition": run.repetition,
        "status": run.status,
        "evaluation_status": run.evaluation_status,
        "success": run.success,
        "attempts": len(counted),
        "retry_attempts": max(0, len(counted) - 1),
        "stop_reason": result.stop_reason if result else None,
        "error_code": result.error_code if result else None,
        "reason_codes": [f"{s.evaluator_name}:{s.reason_code}" for s in run.failing_scores],
        "scores": [
            {
                "evaluator_name": s.evaluator_name,
                "evaluator_version": s.evaluator_version,
                "status": s.status,
                "value": s.value,
                "reason_code": s.reason_code,
                "applicability": s.applicability,
                "evidence": s.evidence,
            }
            for s in run.scores
        ],
        "tool_call_counts": _call_counts(run),  # None: trajectory did not score this run
        "latency_ms": {"queue": run.queue_ms, "execution": run.execution_ms},
        "tokens": {"input": run.input_tokens, "output": run.output_tokens},
        "cost_usd": money(run.cost_usd),
        "created_at": _iso(run.created_at),
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
    }


def _supplement(session: Session, experiment: Experiment) -> dict[str, Any]:
    """Facts known only after running (resolved model ids), plus fault config for pre-Phase-5 manifests."""
    resolved: dict[str, set[str]] = defaultdict(set)
    for config_id, model in session.execute(
        select(Run.agent_config_id, ModelCall.model)
        .select_from(Run)
        .join(Attempt, Attempt.run_id == Run.id)
        .join(ModelCall, ModelCall.attempt_id == Attempt.id)
        .where(Run.experiment_id == experiment.id)
        .distinct()
    ):
        resolved[str(config_id)].add(model)
    supplement: dict[str, Any] = {
        "resolved_models": {config_id: sorted(models) for config_id, models in sorted(resolved.items())},
    }
    if "fault_config" not in experiment.manifest:
        # Task versions are immutable, so this is what the manifest would have recorded.
        ids = [uuid.UUID(t["task_version_id"]) for t in experiment.manifest.get("tasks", [])]
        tasks = session.scalars(select(TaskVersion).where(TaskVersion.id.in_(ids))).all()
        supplement["fault_config"] = fault_config(tasks)
    return supplement


def export_experiment(session: Session, experiment_id: uuid.UUID) -> dict[str, Any]:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise NotFound(f"experiment {experiment_id} not found")
    runs = load_run_facts(session, experiment_id)
    by_config: dict[str, list[RunFacts]] = defaultdict(list)
    for run in runs:
        by_config[run.agent_config_id].append(run)
    return {
        "export_version": EXPORT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment": experiment_summary(experiment, len(runs)),
        "manifest": experiment.manifest,
        "manifest_sha256": sha256_json(experiment.manifest),
        "manifest_supplement": _supplement(session, experiment),
        "scorecards": [
            {
                "agent_config_id": config["id"],
                "agent_config_name": config["name"],
                "scorecard": scorecard(by_config.get(config["id"], [])),
            }
            for config in experiment.manifest.get("agent_configs", [])
        ],
        "runs": [_run_out(run) for run in runs],
        # Optional LLM judge: secondary, uncalibrated, never part of a scorecard; its cost is not candidate cost.
        "secondary_judges": judge_summaries(session, experiment_id),
    }


# --- CSV -------------------------------------------------------------------------------


def _cell(value: Any) -> Any:
    """Empty for unknown; strings that a spreadsheet would run as a formula are quoted."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str) and (value.startswith(_FORMULA_PREFIXES) or value.lstrip().startswith(_FORMULA_PREFIXES)):
        return "'" + value
    return value


def to_csv(report: dict[str, Any]) -> str:
    """One row per run. Single-cell ``#`` CSV records on top carry the full manifest; each row repeats its identifying hashes."""
    manifest = report["manifest"]
    evaluators = [e["name"] for e in manifest.get("evaluators", [])]
    configs = {c["id"]: c for c in manifest.get("agent_configs", [])}
    code = manifest.get("code", {})
    shared = {
        "manifest_sha256": report["manifest_sha256"],
        "dataset_name": manifest.get("dataset", {}).get("name"),
        "dataset_content_hash": manifest.get("dataset", {}).get("content_hash"),
        "environment_version": manifest.get("environment", {}).get("version"),
        "evaluator_versions": ";".join(f"{e['name']}@{e['version']}" for e in manifest.get("evaluators", [])),
        "git_commit": code.get("git_commit"),
        "lockfile_sha256": code.get("lockfile_sha256"),
    }
    score_columns = [f"{name}_{field}" for name in evaluators for field in ("status", "reason_code", "value")]
    tool_columns = [column for column, _ in _TOOL_COLUMNS]
    header = [*_RUN_COLUMNS, *score_columns, *tool_columns, *_USAGE_COLUMNS, *_MANIFEST_COLUMNS]

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for line in (
        f"agent-eval-platform experiment export v{report['export_version']}",
        f"experiment_id: {report['experiment']['id']}",
        f"generated_at: {report['generated_at']}",
        "empty cells are unknown or not applicable; an unknown cost is never zero",
        f"manifest_sha256: {report['manifest_sha256']}",
        "manifest: " + json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        "manifest_supplement: "
        + json.dumps(report["manifest_supplement"], sort_keys=True, separators=(",", ":"), ensure_ascii=False),
    ):
        # Metadata is CSV too: raw JSON commas could expose formula cells.
        writer.writerow([f"# {line}"])
    writer.writerow([_cell(column) for column in header])
    split = report["experiment"]["split"]
    for run in report["runs"]:
        config = configs.get(run["agent_config_id"], {})
        scores = {s["evaluator_name"]: s for s in run["scores"]}
        row: dict[str, Any] = {
            **{column: run.get(column) for column in _RUN_COLUMNS},
            "experiment_id": report["experiment"]["id"],
            "model": config.get("model"),
            "split": split,
            "reason_codes": ";".join(run["reason_codes"]),
            "queue_ms": run["latency_ms"]["queue"],
            "execution_ms": run["latency_ms"]["execution"],
            "input_tokens": run["tokens"]["input"],
            "output_tokens": run["tokens"]["output"],
            "cost_usd": run["cost_usd"],
            "created_at": run["created_at"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            **shared,
            "agent_config_hash": config.get("config_hash"),
            "prompt_hash": config.get("prompt_hash"),
            "tool_schema_hash": config.get("tool_schema_hash"),
        }
        counts = run.get("tool_call_counts") or {}
        for column, key in _TOOL_COLUMNS:
            row[column] = counts.get(key)
        for name in evaluators:
            score = scores.get(name, {})
            row[f"{name}_status"] = score.get("status")
            row[f"{name}_reason_code"] = score.get("reason_code")
            row[f"{name}_value"] = score.get("value")
        writer.writerow([_cell(row.get(column)) for column in header])
    return out.getvalue()
