"""PostgreSQL schema.

Relational columns for anything filtered, joined or compared; JSONB for
flexible payloads (views, manifests, evidence, events). Timestamps are UTC
(``timestamptz``); durations come from monotonic clocks. Status columns are
plain strings guarded by CHECK constraints so the enums stay easy to migrate.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EXECUTION_STATUSES = (
    "queued", "running", "completed", "timed_out", "budget_exceeded", "infrastructure_error", "cancelled",
)
EVALUATION_STATUSES = ("pending", "pass", "fail", "partial", "not_applicable", "evaluation_error")
EXPERIMENT_STATUSES = ("queued", "running", "cancelling", "cancelled", "completed")
JOB_STATUSES = ("queued", "leased", "done", "failed", "cancelled")
SPLITS = ("dev", "validation", "test")

NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


# Python None -> SQL NULL (not the JSON literal null), so IS NULL filters mean what they say.
_NullableJSONB = JSONB(none_as_null=True)


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)
    # Fetch server defaults (created_at, ...) with RETURNING at flush, so objects stay usable after commit.
    __mapper_args__ = {"eager_defaults": True}


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _created() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    created_at: Mapped[datetime] = _created()


class DatasetVersion(Base):
    """Immutable: identified by its content hash within a dataset."""

    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "content_hash"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    split_seed: Mapped[str] = mapped_column(String(200), nullable=False)
    task_count: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _created()


class TaskVersion(Base):
    """One task as frozen in a dataset version. ``agent_view`` is the only part an agent may see."""

    __tablename__ = "task_versions"
    __table_args__ = (
        UniqueConstraint("dataset_version_id", "task_id"),
        CheckConstraint(_in("split", SPLITS), name="split"),
        Index("ix_task_versions_version_split", "dataset_version_id", "split"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dataset_versions.id"), nullable=False)
    task_id: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    family: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    split: Mapped[str] = mapped_column(String(20), nullable=False)
    paraphrase_of: Mapped[str | None] = mapped_column(String(200))
    review_status: Mapped[str] = mapped_column(String(30), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    agent_view: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    evaluator_view: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class AgentConfig(Base):
    """Immutable: a name always means the same model, prompt, tools, adapter and limits."""

    __tablename__ = "agent_configs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    adapter: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str | None] = mapped_column(String(200))
    prompt: Mapped[str | None] = mapped_column(Text)
    prompt_hash: Mapped[str | None] = mapped_column(String(64))
    tools: Mapped[list[str] | None] = mapped_column(_NullableJSONB)
    limits: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = _created()


class Experiment(Base):
    __tablename__ = "experiments"
    __table_args__ = (
        CheckConstraint(_in("status", EXPERIMENT_STATUSES), name="status"),
        CheckConstraint(_in("split", SPLITS), name="split"),
        CheckConstraint("repetitions >= 1", name="repetitions"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str | None] = mapped_column(String(200))
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dataset_versions.id"), nullable=False)
    split: Mapped[str] = mapped_column(String(20), nullable=False)
    repetitions: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="queued")
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _created()
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Run(Base):
    """One configuration x one task x one repetition."""

    __tablename__ = "runs"
    __table_args__ = (
        UniqueConstraint("experiment_id", "agent_config_id", "task_version_id", "repetition", name="uq_runs_plan"),
        CheckConstraint(_in("status", EXECUTION_STATUSES), name="status"),
        CheckConstraint(_in("evaluation_status", EVALUATION_STATUSES), name="evaluation_status"),
        CheckConstraint("repetition >= 1", name="repetition"),
        Index("ix_runs_experiment_status", "experiment_id", "status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    experiment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    agent_config_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_configs.id"), nullable=False, index=True)
    task_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("task_versions.id"), nullable=False, index=True)
    repetition: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, server_default="queued")
    evaluation_status: Mapped[str] = mapped_column(String(30), nullable=False, server_default="pending")
    success: Mapped[bool | None] = mapped_column(Boolean)
    created_at: Mapped[datetime] = _created()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Attempt(Base):
    """One concrete execution of a run, in its own environment. Infra retries add attempts."""

    __tablename__ = "attempts"
    __table_args__ = (
        UniqueConstraint("run_id", "attempt_number"),
        CheckConstraint(_in("status", EXECUTION_STATUSES), name="status"),
        CheckConstraint("attempt_number >= 1", name="attempt_number"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id"), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    environment_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    worker_id: Mapped[str | None] = mapped_column(String(200))
    error_code: Mapped[str | None] = mapped_column(String(100))
    error: Mapped[dict[str, Any] | None] = mapped_column(_NullableJSONB)
    stop_reason: Mapped[str | None] = mapped_column(String(50))
    final_response: Mapped[str | None] = mapped_column(Text)
    agent_metadata: Mapped[dict[str, Any] | None] = mapped_column(_NullableJSONB)
    initial_snapshot: Mapped[dict[str, Any] | None] = mapped_column(_NullableJSONB)
    final_snapshot: Mapped[dict[str, Any] | None] = mapped_column(_NullableJSONB)
    usage: Mapped[dict[str, Any] | None] = mapped_column(_NullableJSONB)
    input_tokens: Mapped[int | None] = mapped_column(Integer)  # None: unknown, never zero
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    duration_ms: Mapped[float | None] = mapped_column(Float)  # monotonic
    created_at: Mapped[datetime] = _created()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Job(Base):
    """Queue entry for one run. Only the holder of ``lease_token`` may write the run's results."""

    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(_in("status", JOB_STATUSES), name="status"),
        CheckConstraint("(status = 'leased') = (lease_token IS NOT NULL)", name="lease_token_iff_leased"),
        Index("ix_jobs_available", "available_at", postgresql_where=text("status = 'queued'")),
        Index("ix_jobs_lease_expiry", "lease_expires_at", postgresql_where=text("status = 'leased'")),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id"), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="queued")
    current_attempt_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("attempts.id"), index=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    lease_token: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    leased_by: Mapped[str | None] = mapped_column(String(200))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempts_started: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="3")
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class TraceEvent(Base):
    __tablename__ = "trace_events"
    __table_args__ = (UniqueConstraint("attempt_id", "event_sequence"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    attempt_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("attempts.id"), nullable=False)
    event_sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(30), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    monotonic_ns: Mapped[int | None] = mapped_column(BigInteger)


class PriceVersion(Base):
    """Dated, immutable per-model rates (USD per million tokens). NULL rate = unknown."""

    __tablename__ = "price_versions"
    __table_args__ = (UniqueConstraint("price_version_id", "model"),)

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    price_version_id: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    effective_date: Mapped[date] = mapped_column(Date, nullable=False)
    input_per_mtok: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    output_per_mtok: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    cache_read_per_mtok: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    cache_write_per_mtok: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    created_at: Mapped[datetime] = _created()


class ModelCall(Base):
    __tablename__ = "model_calls"
    __table_args__ = (UniqueConstraint("attempt_id", "event_sequence"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    attempt_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("attempts.id"), nullable=False)
    event_sequence: Mapped[int] = mapped_column(Integer, nullable=False)  # the model_response event
    turn: Mapped[int | None] = mapped_column(Integer)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_read_input_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_creation_input_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    price_version_id: Mapped[int | None] = mapped_column(ForeignKey("price_versions.id"))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    stop_reason: Mapped[str | None] = mapped_column(String(50))
    error: Mapped[str | None] = mapped_column(Text)


class ToolCall(Base):
    """A tool attempt joined with the gateway decision and the actual effect."""

    __tablename__ = "tool_calls"
    __table_args__ = (UniqueConstraint("attempt_id", "attempt_sequence"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    attempt_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("attempts.id"), nullable=False)
    attempt_sequence: Mapped[int] = mapped_column(Integer, nullable=False)  # the tool_attempt event
    call_id: Mapped[str | None] = mapped_column(String(200))
    tool: Mapped[str] = mapped_column(String(100), nullable=False)
    args: Mapped[Any] = mapped_column(_NullableJSONB, nullable=True)
    decision: Mapped[str | None] = mapped_column(String(20))
    reason_code: Mapped[str | None] = mapped_column(String(100))
    effect_status: Mapped[str | None] = mapped_column(String(20))
    state_changes: Mapped[dict[str, Any] | None] = mapped_column(_NullableJSONB)


class Score(Base):
    __tablename__ = "scores"
    __table_args__ = (
        UniqueConstraint(
            "attempt_id", "evaluator_name", "evaluator_version", "dimension", name="uq_scores_attempt_evaluator"
        ),
        CheckConstraint(_in("status", EVALUATION_STATUSES), name="status"),
        CheckConstraint("applicability IN ('applicable', 'not_applicable')", name="applicability"),
        CheckConstraint("status <> 'evaluation_error' OR value IS NULL", name="error_has_no_value"),
        Index("ix_scores_experiment_category", "experiment_id", "task_category"),
        Index("ix_scores_experiment_evaluator", "experiment_id", "evaluator_name", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    attempt_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("attempts.id"), nullable=False)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id"), nullable=False, index=True)
    experiment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    task_category: Mapped[str] = mapped_column(String(50), nullable=False)
    evaluator_name: Mapped[str] = mapped_column(String(100), nullable=False)
    evaluator_version: Mapped[str] = mapped_column(String(30), nullable=False)
    dimension: Mapped[str] = mapped_column(String(100), nullable=False, server_default="overall")
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    reason_code: Mapped[str] = mapped_column(String(100), nullable=False)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    applicability: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = _created()


class JudgeScore(Base):
    """Optional LLM judge of communication quality. Secondary: kept out of ``scores`` on purpose.

    Nothing here feeds run success, evaluation status, scorecards or comparisons.
    The judge's own tokens and cost live here, never in the candidate agent's usage.
    """

    __tablename__ = "judge_scores"
    __table_args__ = (
        UniqueConstraint("attempt_id", "judge_name", "judge_version", name="uq_judge_scores_attempt_judge"),
        CheckConstraint("status IN ('scored', 'evaluation_error')", name="status"),
        CheckConstraint("status <> 'evaluation_error' OR scores IS NULL", name="error_has_no_scores"),
        Index("ix_judge_scores_experiment_judge", "experiment_id", "judge_name"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    attempt_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("attempts.id"), nullable=False)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id"), nullable=False, index=True)
    experiment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    judge_name: Mapped[str] = mapped_column(String(100), nullable=False)
    judge_version: Mapped[str] = mapped_column(String(30), nullable=False)
    rubric_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)  # as configured
    resolved_model: Mapped[str | None] = mapped_column(String(200))  # as reported by the provider
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    scores: Mapped[dict[str, Any] | None] = mapped_column(_NullableJSONB)  # {dimension: {"score": 1-5, "reason"}}
    reason_code: Mapped[str] = mapped_column(String(100), nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int | None] = mapped_column(Integer)  # None: unknown, never zero
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_read_input_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_creation_input_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    price_version_id: Mapped[str | None] = mapped_column(String(100))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = _created()


BENCHMARK_RESULT_KINDS = ("model_run", "scorer_self_check")


class BenchmarkResult(Base):
    """A pinned public benchmark result (e.g. one BFCL category). Never mixed with custom support scores.

    ``task_count`` < ``category_task_count`` means a subset, and ``subset_method``
    says how it was chosen. ``scorer_self_check`` rows replay ground truth to
    validate the scorer; they are not model results.
    """

    __tablename__ = "benchmark_results"
    __table_args__ = (
        CheckConstraint("task_count >= 0", name="task_count"),
        CheckConstraint(_in("result_kind", BENCHMARK_RESULT_KINDS), name="result_kind"),
        CheckConstraint(
            "category_task_count IS NULL OR category_task_count >= task_count", name="subset_within_category"
        ),
        CheckConstraint(
            "category_task_count IS NULL OR task_count = category_task_count OR subset_method IS NOT NULL",
            name="subset_is_labeled",
        ),
        Index("ix_benchmark_results_benchmark_version", "benchmark", "benchmark_version"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    benchmark: Mapped[str] = mapped_column(String(100), nullable=False)
    benchmark_version: Mapped[str] = mapped_column(String(100), nullable=False)  # pinned release or commit
    category: Mapped[str] = mapped_column(String(200), nullable=False)
    model: Mapped[str | None] = mapped_column(String(200))
    task_count: Mapped[int] = mapped_column(Integer, nullable=False)  # entries actually run
    category_task_count: Mapped[int | None] = mapped_column(Integer)  # entries in the whole category
    subset_method: Mapped[str | None] = mapped_column(Text)
    result_kind: Mapped[str] = mapped_column(String(30), nullable=False, server_default="model_run")
    scoring_method: Mapped[str] = mapped_column(String(200), nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    raw_artifact_uri: Mapped[str | None] = mapped_column(Text)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
