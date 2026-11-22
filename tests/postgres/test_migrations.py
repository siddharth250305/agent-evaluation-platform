"""The Alembic migration is the schema: it creates every table, constraint and index, and matches the models."""

from __future__ import annotations

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
import re

import pytest
from sqlalchemy import CheckConstraint, create_engine, inspect, text
from sqlalchemy.engine import Engine

from app.db.migrate import downgrade, upgrade
from app.db.models import Base
from tests.postgres.conftest import create_database, drop_database

TABLES = {
    "datasets", "dataset_versions", "task_versions", "agent_configs", "experiments", "runs", "attempts",
    "jobs", "trace_events", "model_calls", "tool_calls", "scores", "price_versions", "benchmark_results",
    "judge_scores",
}


def _unique_sets(engine: Engine, table: str) -> set[tuple[str, ...]]:
    return {tuple(c["column_names"]) for c in inspect(engine).get_unique_constraints(table)}


def _indexes(engine: Engine, table: str) -> dict[str, dict]:
    return {i["name"]: i for i in inspect(engine).get_indexes(table)}


def test_migration_creates_every_table(engine: Engine) -> None:
    assert TABLES <= set(inspect(engine).get_table_names())


def test_models_and_migration_agree(engine: Engine) -> None:
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True, "compare_server_default": True})
        diff = compare_metadata(context, Base.metadata)
    assert diff == []


def _assert_checks_match(conn) -> None:
    """Have PostgreSQL parse model expressions, then compare canonical definitions by table/name."""
    live = {(table, name): definition for table, name, definition in conn.execute(text(
        "SELECT c.relname, k.conname, pg_get_constraintdef(k.oid) "
        "FROM pg_constraint k JOIN pg_class c ON c.oid = k.conrelid "
        "WHERE k.contype = 'c' AND k.connamespace = 'public'::regnamespace"
    )) if table in Base.metadata.tables}
    declared = {}
    quote = conn.dialect.identifier_preparer.quote
    for table in Base.metadata.sorted_tables:
        checks = [c for c in table.constraints if isinstance(c, CheckConstraint)]
        if not checks:
            continue
        shadow = f"audit_{table.name}"
        conn.execute(text(f"CREATE TEMP TABLE {quote(shadow)} (LIKE public.{quote(table.name)}) ON COMMIT DROP"))
        for check in checks:
            conn.execute(text(f"ALTER TABLE {quote(shadow)} ADD CONSTRAINT {quote(check.name)} CHECK ({check.sqltext})"))
        for name, definition in conn.execute(text(
            "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid = to_regclass(:table) AND contype = 'c'"
        ), {"table": shadow}):
            declared[(table.name, name)] = definition
    assert live == declared


def test_check_constraints_match_the_models(engine: Engine) -> None:
    # Roll back temp DDL so this check never mutates the migrated public schema.
    with engine.connect() as conn, conn.begin():
        _assert_checks_match(conn)


def test_check_comparison_detects_same_name_but_weakened_definition(engine: Engine) -> None:
    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            conn.execute(text("ALTER TABLE public.scores DROP CONSTRAINT ck_scores_error_has_no_value"))
            conn.execute(text("ALTER TABLE public.scores ADD CONSTRAINT ck_scores_error_has_no_value CHECK (true)"))
            with pytest.raises(AssertionError):
                _assert_checks_match(conn)
        finally:
            transaction.rollback()


def test_check_comparison_ignores_other_schemas(engine: Engine) -> None:
    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            conn.execute(text("CREATE SCHEMA audit_other"))
            conn.execute(text("CREATE TABLE audit_other.scores (value float, "
                              "CONSTRAINT ck_scores_error_has_no_value CHECK (value > 100))"))
            _assert_checks_match(conn)
        finally:
            transaction.rollback()


def test_required_unique_constraints(engine: Engine) -> None:
    assert ("experiment_id", "agent_config_id", "task_version_id", "repetition") in _unique_sets(engine, "runs")
    assert ("run_id", "attempt_number") in _unique_sets(engine, "attempts")
    assert ("attempt_id", "event_sequence") in _unique_sets(engine, "trace_events")
    assert ("attempt_id", "evaluator_name", "evaluator_version", "dimension") in _unique_sets(engine, "scores")
    assert ("run_id",) in _unique_sets(engine, "jobs")
    assert ("idempotency_key",) in _unique_sets(engine, "experiments")
    assert ("dataset_id", "content_hash") in _unique_sets(engine, "dataset_versions")
    assert ("dataset_version_id", "task_id") in _unique_sets(engine, "task_versions")
    assert ("price_version_id", "model") in _unique_sets(engine, "price_versions")


def test_queue_and_query_indexes(engine: Engine) -> None:
    jobs = _indexes(engine, "jobs")
    assert jobs["ix_jobs_available"]["column_names"] == ["available_at"]
    assert _predicate(jobs["ix_jobs_available"]) == "status = 'queued'"
    assert jobs["ix_jobs_lease_expiry"]["column_names"] == ["lease_expires_at"]
    assert _predicate(jobs["ix_jobs_lease_expiry"]) == "status = 'leased'"
    assert _indexes(engine, "scores")["ix_scores_run_id"]["column_names"] == ["run_id"]
    assert _indexes(engine, "runs")["ix_runs_experiment_status"]["column_names"] == ["experiment_id", "status"]
    scores = _indexes(engine, "scores")
    assert scores["ix_scores_experiment_category"]["column_names"] == ["experiment_id", "task_category"]


def test_downgrade_and_upgrade_round_trip(pg_available: None) -> None:
    url = create_database("agent_eval_migration")
    try:
        upgrade(url)
        downgrade(url, "base")
        engine = create_engine(url)
        assert set(inspect(engine).get_table_names()) <= {"alembic_version"}
        engine.dispose()
        upgrade(url)
        engine = create_engine(url)
        assert TABLES <= set(inspect(engine).get_table_names())
        engine.dispose()
    finally:
        drop_database(url)


def _predicate(index: dict) -> str:
    expression = index["dialect_options"]["postgresql_where"]
    return " ".join(re.sub(r"::text|[()]", "", expression).split())


def test_partial_indexes_are_ready_valid_and_nonunique(engine: Engine) -> None:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT c.relname, i.indisvalid, i.indisready, i.indisunique FROM pg_index i "
            "JOIN pg_class c ON c.oid = i.indexrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relname IN ('ix_jobs_available', 'ix_jobs_lease_expiry')"
        )).all()
    assert len(rows) == 2
    assert all(valid and ready and not unique for _, valid, ready, unique in rows)


def test_metadata_comparison_detects_default_and_nullability_drift(engine: Engine) -> None:
    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            conn.execute(text("ALTER TABLE public.jobs ALTER COLUMN max_attempts SET DEFAULT 99"))
            conn.execute(text("ALTER TABLE public.jobs ALTER COLUMN available_at DROP NOT NULL"))
            context = MigrationContext.configure(conn, opts={"compare_type": True, "compare_server_default": True})
            differences = compare_metadata(context, Base.metadata)
            kinds = {change[0] for diff in differences for change in (diff if isinstance(diff, list) else [diff])}
            assert {"modify_default", "modify_nullable"} <= kinds
        finally:
            transaction.rollback()
