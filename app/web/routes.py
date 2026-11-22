"""Dashboard: five server-rendered screens over the same services as the JSON API.

Experiment list, experiment detail, run detail, comparison and benchmarks.
Pages work without JavaScript; the little there is only filters tables in place.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.benchmarks.bfcl import ARTIFACTS_DIR, ROOT
from app.db.models import (
    AgentConfig,
    Attempt,
    BenchmarkResult,
    Experiment,
    JudgeScore,
    Run,
    Score,
    TaskVersion,
    TraceEvent,
)
from app.metrics import RunFacts, by_category, failure_reasons, hard_violations, scorecard
from app.services.comparisons import ComparisonCreate, ComparisonSide, compare_experiments
from app.services.errors import ServiceError
from app.services.judging import judge_summaries
from app.services.results import experiment_rows, load_run_facts
from app.web.present import FILTERS, evidence_sequences, state_diff

_HERE = Path(__file__).parent
STATIC_DIR = _HERE / "static"
templates = Jinja2Templates(directory=str(_HERE / "templates"))
templates.env.filters.update(FILTERS)

router = APIRouter(include_in_schema=False)

_MAX_RUN_ROWS = 500
_FAILING = ("fail", "evaluation_error")


def _sm(request: Request) -> sessionmaker[Session]:
    return request.app.state.sessionmaker


def _page(request: Request, name: str, status_code: int = 200, **context: Any) -> HTMLResponse:
    return templates.TemplateResponse(request, name, context, status_code=status_code)


def _not_found(request: Request, message: str) -> HTMLResponse:
    return _page(request, "error.html", 404, title="Not found", message=message)


@router.get("/")
def home() -> Response:
    return RedirectResponse("/ui/experiments", status_code=307)


# --- 1. experiment list ----------------------------------------------------------------


@router.get("/ui/experiments", response_class=HTMLResponse)
def experiments_page(request: Request) -> HTMLResponse:
    with _sm(request)() as session:
        rows = experiment_rows(session)
    return _page(request, "experiments.html", active="experiments", experiments=rows)


# --- 2. experiment detail --------------------------------------------------------------


def _matches(run: RunFacts, filters: dict[str, str]) -> bool:
    checks = {
        "config": run.agent_config_id,
        "status": run.status,
        "evaluation_status": run.evaluation_status,
        "category": run.category,
        "success": {True: "true", False: "false", None: "unknown"}[run.success],
    }
    if any(filters.get(key) and filters[key] != value for key, value in checks.items()):
        return False
    needle = filters.get("q", "").strip().lower()
    haystack = " ".join([run.task_id, *(f"{s.evaluator_name}:{s.reason_code}" for s in run.failing_scores)]).lower()
    return not needle or needle in haystack


@router.get("/ui/experiments/{experiment_id}", response_class=HTMLResponse)
def experiment_page(
    experiment_id: uuid.UUID,
    request: Request,
    config: str = "",
    status: str = "",
    evaluation_status: str = "",
    success: str = "",
    category: str = "",
    q: str = "",
) -> HTMLResponse:
    with _sm(request)() as session:
        experiment = session.get(Experiment, experiment_id)
        if experiment is None:
            return _not_found(request, f"experiment {experiment_id} not found")
        runs = load_run_facts(session, experiment_id)
        judges = judge_summaries(session, experiment_id)

    configs = experiment.manifest.get("agent_configs", [])
    per_config = []
    for cfg in configs:
        own = [r for r in runs if r.agent_config_id == cfg["id"]]
        per_config.append({"config": cfg, "scorecard": scorecard(own), "categories": by_category(own)})
    filters = {
        "config": config, "status": status, "evaluation_status": evaluation_status,
        "success": success, "category": category, "q": q,
    }
    filtered = [r for r in runs if _matches(r, filters)]
    return _page(
        request,
        "experiment.html",
        active="experiments",
        experiment=experiment,
        configs=configs,
        per_config=per_config,
        judges=judges,
        reasons=failure_reasons(runs),
        violations=hard_violations(runs),
        runs=filtered[:_MAX_RUN_ROWS],
        matching_runs=len(filtered),
        total_runs=len(runs),
        max_rows=_MAX_RUN_ROWS,
        filters=filters,
        options={
            "status": sorted({r.status for r in runs}),
            "evaluation_status": sorted({r.evaluation_status for r in runs}),
            "category": sorted({r.category for r in runs}),
        },
    )


# --- 3. run detail ---------------------------------------------------------------------


def _selected_attempt(attempts: list[Attempt], scores: list[Score], number: int | None) -> Attempt | None:
    if number is not None:
        return next((a for a in attempts if a.attempt_number == number), None)
    scored = {s.attempt_id for s in scores}
    graded = [a for a in attempts if a.id in scored]
    return (graded or attempts or [None])[-1]


def _event_flags(event: TraceEvent, failing: dict[int, list[str]]) -> list[str]:
    flags = []
    if event.event_sequence in failing:
        flags.append("flagged")
    if event.kind == "tool_decision" and event.payload.get("decision") == "denied":
        flags.append("denied")
    if event.kind == "tool_effect" and event.payload.get("status") == "failed":
        flags.append("failed-effect")
    return flags


@router.get("/ui/runs/{run_id}", response_class=HTMLResponse)
def run_page(run_id: uuid.UUID, request: Request, attempt: int | None = None) -> HTMLResponse:
    with _sm(request)() as session:
        run = session.get(Run, run_id)
        if run is None:
            return _not_found(request, f"run {run_id} not found")
        task = session.get(TaskVersion, run.task_version_id)
        config = session.get(AgentConfig, run.agent_config_id)
        experiment = session.get(Experiment, run.experiment_id)
        attempts = list(session.scalars(select(Attempt).where(Attempt.run_id == run_id).order_by(Attempt.attempt_number)))
        scores = list(session.scalars(select(Score).where(Score.run_id == run_id).order_by(Score.id)))
        selected = _selected_attempt(attempts, scores, attempt)
        if attempt is not None and selected is None:
            return _not_found(request, f"run {run_id} has no attempt {attempt}")
        events = list(session.scalars(
            select(TraceEvent).where(TraceEvent.attempt_id == selected.id).order_by(TraceEvent.event_sequence)
        )) if selected else []
        judged = list(session.scalars(
            select(JudgeScore).where(JudgeScore.attempt_id == selected.id).order_by(JudgeScore.id)
        )) if selected else []

    shown_scores = [s for s in scores if selected is not None and s.attempt_id == selected.id]
    failing: dict[int, list[str]] = {}
    for score in shown_scores:
        if score.status in _FAILING:
            for sequence in evidence_sequences(score.evidence):
                failing.setdefault(sequence, []).append(f"{score.evaluator_name}: {score.reason_code}")
    timeline = [{"event": e, "flags": _event_flags(e, failing), "cited_by": failing.get(e.event_sequence, [])}
                for e in events]
    diff = None
    if selected is not None and selected.initial_snapshot is not None and selected.final_snapshot is not None:
        diff = state_diff(selected.initial_snapshot, selected.final_snapshot)

    offered = task.agent_view.get("tools", [])
    return _page(
        request,
        "run.html",
        active="experiments",
        run=run,
        task=task,
        config=config,
        experiment=experiment,
        attempts=attempts,
        selected=selected,
        scores=shown_scores,
        judged=judged,
        timeline=timeline,
        flagged_count=sum(1 for item in timeline if "flagged" in item["flags"]),
        diff=diff,
        agent_visible={
            "system_prompt": config.prompt,
            "user_request": task.agent_view.get("user_request"),
            "tools": [t for t in offered if config.tools is None or t in config.tools],
        },
        hidden={
            "expected": task.evaluator_view.get("expected"),
            "trajectory": task.evaluator_view.get("trajectory"),
            "reference": task.evaluator_view.get("reference"),
            "fixture": task.evaluator_view.get("fixture"),
        },
    )


# --- 4. comparison ---------------------------------------------------------------------


def _side(value: str) -> ComparisonSide:
    experiment_id, _, config_id = value.partition(":")
    return ComparisonSide(experiment_id=experiment_id, agent_config_id=config_id or None)


@router.get("/ui/compare", response_class=HTMLResponse)
def compare_page(request: Request, baseline: str = "", candidate: str = "") -> HTMLResponse:
    result, error, status_code = None, None, 200
    with _sm(request)() as session:
        rows = experiment_rows(session)
        if baseline and candidate:
            try:
                payload = ComparisonCreate(baseline=_side(baseline), candidate=_side(candidate))
                result = compare_experiments(session, payload)
            except ValidationError:
                error, status_code = "Pick a baseline and a candidate from the lists.", 422
            except ServiceError as exc:
                error, status_code = str(exc), exc.status_code
    options = [
        {
            "value": f"{row['id']}:{cfg['id']}",
            "label": f"{row['name'] or row['id'][:8]} · {cfg['name']} · {row['split']} · ×{row['repetitions']}",
        }
        for row in rows
        for cfg in row["agent_configs"]
    ]
    return _page(
        request,
        "compare.html",
        status_code,
        active="compare",
        options=options,
        baseline=baseline,
        candidate=candidate,
        result=result,
        error=error,
    )


# --- 5. benchmarks ---------------------------------------------------------------------


@router.get("/ui/benchmarks", response_class=HTMLResponse)
def benchmarks_page(request: Request) -> HTMLResponse:
    with _sm(request)() as session:
        results = list(session.scalars(
            select(BenchmarkResult).order_by(
                BenchmarkResult.benchmark, BenchmarkResult.benchmark_version, BenchmarkResult.category,
                BenchmarkResult.model,
            )
        ))
    return _page(request, "benchmarks.html", active="benchmarks", results=results)


def _artifact_path(uri: str | None) -> Path | None:
    """Only files under the artifacts directory are served, whatever the stored URI says."""
    if not uri:
        return None
    path = (ROOT / uri).resolve()
    if not path.is_relative_to(ARTIFACTS_DIR.resolve()) or not path.is_file():
        return None
    return path


@router.get("/ui/benchmarks/{result_id}/raw")
def benchmark_raw_outputs(result_id: uuid.UUID, request: Request) -> Response:
    """Raw model outputs as stored: served as a download, never rendered as HTML."""
    with _sm(request)() as session:
        result = session.get(BenchmarkResult, result_id)
    path = _artifact_path(result.raw_artifact_uri) if result is not None else None
    if path is None:
        return _not_found(request, f"no raw output artifact for benchmark result {result_id}")
    return FileResponse(
        path,
        media_type="application/x-ndjson",
        filename=f"bfcl-{result_id}-raw_outputs.jsonl",
        headers={"X-Content-Type-Options": "nosniff"},
    )
