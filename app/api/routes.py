"""HTTP endpoints. Thin: validation here, transactions and rules in ``app.services``."""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.db.migrate import current_revision, head_revision
from app.db.models import AgentConfig, Dataset, DatasetVersion
from app.services import experiments as experiment_service
from app.services import reports
from app.services.comparisons import ComparisonCreate, compare_experiments
from app.services.catalog import AgentConfigIn, import_dataset, register_agent_config, split_counts
from app.services.experiments import ExperimentCreate

router = APIRouter()

ExecutionStatus = Literal[
    "queued", "running", "completed", "timed_out", "budget_exceeded", "infrastructure_error", "cancelled"
]
EvaluationStatus = Literal["pending", "pass", "fail", "partial", "not_applicable", "evaluation_error"]


def _sm(request: Request) -> sessionmaker[Session]:
    return request.app.state.sessionmaker


# --- health --------------------------------------------------------------------------


@router.get("/health/live", tags=["health"])
def live() -> dict[str, str]:
    """The process is up. Says nothing about dependencies."""
    return {"status": "ok"}


@router.get("/health/ready", tags=["health"])
def ready(request: Request, response: Response) -> dict[str, Any]:
    """Ready only when PostgreSQL answers and the schema is at the latest migration."""
    try:
        with _sm(request)() as session:
            connection = session.connection()
            connection.execute(text("SELECT 1"))
            revision = current_revision(connection)
    except SQLAlchemyError:
        response.status_code = 503
        return {"status": "unavailable", "database": "unavailable", "migrations": "unknown"}
    migrations = "head" if revision == head_revision() else "behind"
    if migrations != "head":
        response.status_code = 503
    return {"status": "ready" if migrations == "head" else "unavailable", "database": "ok",
            "migrations": migrations, "revision": revision}


# --- datasets ------------------------------------------------------------------------


class DatasetImport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(pattern=r"^[a-z0-9_]+$", max_length=100, description="Directory name under datasets/")


def _version_out(version: DatasetVersion) -> dict[str, Any]:
    return {
        "id": str(version.id),
        "dataset_id": str(version.dataset_id),
        "content_hash": version.content_hash,
        "split_seed": version.split_seed,
        "task_count": version.task_count,
        "split_counts": split_counts(version),
        "created_at": version.created_at.isoformat(),
    }


@router.post("/datasets/import", status_code=201, tags=["datasets"])
def import_dataset_route(body: DatasetImport, request: Request, response: Response) -> dict[str, Any]:
    """Validate a dataset directory and store it as an immutable version (200 if this content exists)."""
    with _sm(request).begin() as session:
        version, created = import_dataset(session, body.dataset)
        out = _version_out(version)
    if not created:
        response.status_code = 200
    return out


@router.get("/datasets", tags=["datasets"])
def list_datasets(request: Request) -> list[dict[str, Any]]:
    with _sm(request)() as session:
        datasets = session.scalars(select(Dataset).order_by(Dataset.name)).all()
        versions = session.scalars(select(DatasetVersion).order_by(DatasetVersion.created_at)).all()
        return [
            {
                "id": str(d.id),
                "name": d.name,
                "versions": [_version_out(v) for v in versions if v.dataset_id == d.id],
            }
            for d in datasets
        ]


# --- agent configurations ------------------------------------------------------------


def _config_out(config: AgentConfig) -> dict[str, Any]:
    return {
        "id": str(config.id),
        "name": config.name,
        "adapter": config.adapter,
        "model": config.model,
        "prompt_hash": config.prompt_hash,
        "tools": config.tools,
        "limits": config.limits,
        "settings": config.settings,
        "config_hash": config.config_hash,
        "created_at": config.created_at.isoformat(),
    }


@router.post("/agent-configs", status_code=201, tags=["agent configurations"])
def register_config(body: AgentConfigIn, request: Request, response: Response) -> dict[str, Any]:
    """Register an immutable configuration (200 if identical, 409 if the name exists with other content)."""
    with _sm(request).begin() as session:
        config, created = register_agent_config(session, body, known_adapters=request.app.state.adapter_names)
        out = _config_out(config)
    if not created:
        response.status_code = 200
    return out


@router.get("/agent-configs", tags=["agent configurations"])
def list_configs(request: Request) -> list[dict[str, Any]]:
    with _sm(request)() as session:
        configs = session.scalars(select(AgentConfig).order_by(AgentConfig.created_at, AgentConfig.name)).all()
        return [_config_out(c) for c in configs]


# --- experiments ---------------------------------------------------------------------


@router.post("/experiments", status_code=201, tags=["experiments"])
def create_experiment_route(
    body: ExperimentCreate,
    request: Request,
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> dict[str, Any]:
    """Create the experiment, its runs and their jobs in one transaction.

    Requires ``Idempotency-Key``: repeating a request with the same key and body
    returns the same experiment (200); the same key with a different body is 409.
    """
    if not idempotency_key or not idempotency_key.strip():
        raise HTTPException(status_code=400, detail="the Idempotency-Key header is required")
    experiment, created = experiment_service.create_experiment(_sm(request), body, idempotency_key.strip())
    if not created:
        response.status_code = 200
        response.headers["Idempotent-Replayed"] = "true"
    with _sm(request)() as session:
        return experiment_service.experiment_detail(session, experiment.id)


@router.get("/experiments", tags=["experiments"])
def list_experiments_route(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[dict[str, Any]]:
    with _sm(request)() as session:
        return experiment_service.list_experiments(session, limit=limit, offset=offset)


@router.get("/experiments/{experiment_id}", tags=["experiments"])
def get_experiment(experiment_id: uuid.UUID, request: Request) -> dict[str, Any]:
    """Progress and scorecard."""
    with _sm(request)() as session:
        return experiment_service.experiment_detail(session, experiment_id)


@router.get("/experiments/{experiment_id}/runs", tags=["experiments"])
def list_runs(
    experiment_id: uuid.UUID,
    request: Request,
    status: ExecutionStatus | None = None,
    evaluation_status: EvaluationStatus | None = None,
    success: bool | None = None,
    agent_config_id: uuid.UUID | None = None,
    category: str | None = None,
    task_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    filters = {
        "status": status,
        "evaluation_status": evaluation_status,
        "success": success,
        "agent_config_id": agent_config_id,
        "category": category,
        "task_id": task_id,
    }
    with _sm(request)() as session:
        return experiment_service.list_runs(session, experiment_id, filters=filters, limit=limit, offset=offset)


@router.post("/experiments/{experiment_id}/cancel", status_code=202, tags=["experiments"])
def cancel(experiment_id: uuid.UUID, request: Request) -> dict[str, Any]:
    """Queued runs are cancelled now; running attempts stop before their next tool call."""
    experiment_service.cancel_experiment(_sm(request), experiment_id)
    with _sm(request)() as session:
        return experiment_service.experiment_detail(session, experiment_id)


@router.get("/experiments/{experiment_id}/export", tags=["experiments"])
def export_experiment_route(
    experiment_id: uuid.UUID,
    request: Request,
    format_: Annotated[Literal["json", "csv"], Query(alias="format")] = "json",
) -> Response:
    """Per-run results, scores, reason codes, latency, tokens and cost, bundled with the experiment manifest."""
    with _sm(request)() as session:
        report = reports.export_experiment(session, experiment_id)
    disposition = {"Content-Disposition": f'attachment; filename="experiment-{experiment_id}.{format_}"'}
    if format_ == "csv":
        return Response(reports.to_csv(report), media_type="text/csv; charset=utf-8", headers=disposition)
    return JSONResponse(report, headers=disposition)


# --- comparisons ---------------------------------------------------------------------


@router.post("/comparisons", tags=["comparisons"])
def compare_route(body: ComparisonCreate, request: Request) -> dict[str, Any]:
    """Compare two compatible (experiment, configuration) pairs task by task.

    Compatible means the same dataset version and split, environment version,
    evaluator versions and limits; anything else is 422. Computed on request
    (nothing is stored) and deterministic for a given ``seed``.
    """
    with _sm(request)() as session:
        return compare_experiments(session, body)


# --- runs and traces -----------------------------------------------------------------


@router.get("/runs/{run_id}", tags=["runs"])
def get_run(run_id: uuid.UUID, request: Request) -> dict[str, Any]:
    """Outcome, every attempt (including interrupted ones) and the scores with their evidence."""
    with _sm(request)() as session:
        return experiment_service.run_detail(session, run_id)


@router.get("/attempts/{attempt_id}/events", tags=["runs"])
def get_events(
    attempt_id: uuid.UUID,
    request: Request,
    after_sequence: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=5000)] = 1000,
) -> dict[str, Any]:
    """The ordered trace of one attempt."""
    with _sm(request)() as session:
        return experiment_service.attempt_events(session, attempt_id, after_sequence=after_sequence, limit=limit)


