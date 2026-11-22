"""Initial schema: datasets, configs, experiments, queue, traces, scores, prices.

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-21

Generated from app.db.models with Alembic's renderer, then reviewed.
tests/postgres/test_migrations.py checks that models and migration agree.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('agent_configs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('adapter', sa.String(length=100), nullable=False),
        sa.Column('model', sa.String(length=200), nullable=True),
        sa.Column('prompt', sa.Text(), nullable=True),
        sa.Column('prompt_hash', sa.String(length=64), nullable=True),
        sa.Column('tools', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=True),
        sa.Column('limits', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('settings', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('config_hash', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_agent_configs')),
        sa.UniqueConstraint('name', name=op.f('uq_agent_configs_name'))
    )
    op.create_table('datasets',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_datasets')),
        sa.UniqueConstraint('name', name=op.f('uq_datasets_name'))
    )
    op.create_table('price_versions',
        sa.Column('id', sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column('price_version_id', sa.String(length=100), nullable=False),
        sa.Column('model', sa.String(length=200), nullable=False),
        sa.Column('effective_date', sa.Date(), nullable=False),
        sa.Column('input_per_mtok', sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column('output_per_mtok', sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column('cache_read_per_mtok', sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column('cache_write_per_mtok', sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_price_versions')),
        sa.UniqueConstraint('price_version_id', 'model', name=op.f('uq_price_versions_price_version_id_model'))
    )
    op.create_table('dataset_versions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('dataset_id', sa.UUID(), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('split_seed', sa.String(length=200), nullable=False),
        sa.Column('task_count', sa.Integer(), nullable=False),
        sa.Column('manifest', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], name=op.f('fk_dataset_versions_dataset_id_datasets')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_dataset_versions')),
        sa.UniqueConstraint('dataset_id', 'content_hash', name=op.f('uq_dataset_versions_dataset_id_content_hash'))
    )
    op.create_table('experiments',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=True),
        sa.Column('dataset_version_id', sa.UUID(), nullable=False),
        sa.Column('split', sa.String(length=20), nullable=False),
        sa.Column('repetitions', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=20), server_default='queued', nullable=False),
        sa.Column('idempotency_key', sa.String(length=200), nullable=False),
        sa.Column('request_hash', sa.String(length=64), nullable=False),
        sa.Column('manifest', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('cancel_requested_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("split IN ('dev', 'validation', 'test')", name=op.f('ck_experiments_split')),
        sa.CheckConstraint("status IN ('queued', 'running', 'cancelling', 'cancelled', 'completed')", name=op.f('ck_experiments_status')),
        sa.CheckConstraint('repetitions >= 1', name=op.f('ck_experiments_repetitions')),
        sa.ForeignKeyConstraint(['dataset_version_id'], ['dataset_versions.id'], name=op.f('fk_experiments_dataset_version_id_dataset_versions')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_experiments')),
        sa.UniqueConstraint('idempotency_key', name=op.f('uq_experiments_idempotency_key'))
    )
    op.create_table('task_versions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('dataset_version_id', sa.UUID(), nullable=False),
        sa.Column('task_id', sa.String(length=200), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('family', sa.String(length=200), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('split', sa.String(length=20), nullable=False),
        sa.Column('paraphrase_of', sa.String(length=200), nullable=True),
        sa.Column('review_status', sa.String(length=30), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('agent_view', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('evaluator_view', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.CheckConstraint("split IN ('dev', 'validation', 'test')", name=op.f('ck_task_versions_split')),
        sa.ForeignKeyConstraint(['dataset_version_id'], ['dataset_versions.id'], name=op.f('fk_task_versions_dataset_version_id_dataset_versions')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_task_versions')),
        sa.UniqueConstraint('dataset_version_id', 'task_id', name=op.f('uq_task_versions_dataset_version_id_task_id'))
    )
    op.create_index('ix_task_versions_version_split', 'task_versions', ['dataset_version_id', 'split'], unique=False)
    op.create_table('runs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('experiment_id', sa.UUID(), nullable=False),
        sa.Column('agent_config_id', sa.UUID(), nullable=False),
        sa.Column('task_version_id', sa.UUID(), nullable=False),
        sa.Column('repetition', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=30), server_default='queued', nullable=False),
        sa.Column('evaluation_status', sa.String(length=30), server_default='pending', nullable=False),
        sa.Column('success', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("evaluation_status IN ('pending', 'pass', 'fail', 'partial', 'not_applicable', 'evaluation_error')", name=op.f('ck_runs_evaluation_status')),
        sa.CheckConstraint("status IN ('queued', 'running', 'completed', 'timed_out', 'budget_exceeded', 'infrastructure_error', 'cancelled')", name=op.f('ck_runs_status')),
        sa.CheckConstraint('repetition >= 1', name=op.f('ck_runs_repetition')),
        sa.ForeignKeyConstraint(['agent_config_id'], ['agent_configs.id'], name=op.f('fk_runs_agent_config_id_agent_configs')),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], name=op.f('fk_runs_experiment_id_experiments')),
        sa.ForeignKeyConstraint(['task_version_id'], ['task_versions.id'], name=op.f('fk_runs_task_version_id_task_versions')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_runs')),
        sa.UniqueConstraint('experiment_id', 'agent_config_id', 'task_version_id', 'repetition', name='uq_runs_plan')
    )
    op.create_index(op.f('ix_runs_agent_config_id'), 'runs', ['agent_config_id'], unique=False)
    op.create_index('ix_runs_experiment_status', 'runs', ['experiment_id', 'status'], unique=False)
    op.create_index(op.f('ix_runs_task_version_id'), 'runs', ['task_version_id'], unique=False)
    op.create_table('attempts',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('run_id', sa.UUID(), nullable=False),
        sa.Column('attempt_number', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=30), nullable=False),
        sa.Column('environment_id', sa.String(length=64), nullable=False),
        sa.Column('worker_id', sa.String(length=200), nullable=True),
        sa.Column('error_code', sa.String(length=100), nullable=True),
        sa.Column('error', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=True),
        sa.Column('stop_reason', sa.String(length=50), nullable=True),
        sa.Column('final_response', sa.Text(), nullable=True),
        sa.Column('agent_metadata', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=True),
        sa.Column('initial_snapshot', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=True),
        sa.Column('final_snapshot', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=True),
        sa.Column('usage', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('cost_usd', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('duration_ms', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('queued', 'running', 'completed', 'timed_out', 'budget_exceeded', 'infrastructure_error', 'cancelled')", name=op.f('ck_attempts_status')),
        sa.CheckConstraint('attempt_number >= 1', name=op.f('ck_attempts_attempt_number')),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], name=op.f('fk_attempts_run_id_runs')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_attempts')),
        sa.UniqueConstraint('environment_id', name=op.f('uq_attempts_environment_id')),
        sa.UniqueConstraint('run_id', 'attempt_number', name=op.f('uq_attempts_run_id_attempt_number'))
    )
    op.create_table('jobs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('run_id', sa.UUID(), nullable=False),
        sa.Column('status', sa.String(length=20), server_default='queued', nullable=False),
        sa.Column('current_attempt_id', sa.UUID(), nullable=True),
        sa.Column('available_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('lease_token', sa.UUID(), nullable=True),
        sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('leased_by', sa.String(length=200), nullable=True),
        sa.Column('heartbeat_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('attempts_started', sa.Integer(), server_default='0', nullable=False),
        sa.Column('max_attempts', sa.Integer(), server_default='3', nullable=False),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("(status = 'leased') = (lease_token IS NOT NULL)", name=op.f('ck_jobs_lease_token_iff_leased')),
        sa.CheckConstraint("status IN ('queued', 'leased', 'done', 'failed', 'cancelled')", name=op.f('ck_jobs_status')),
        sa.ForeignKeyConstraint(['current_attempt_id'], ['attempts.id'], name=op.f('fk_jobs_current_attempt_id_attempts')),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], name=op.f('fk_jobs_run_id_runs')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_jobs')),
        sa.UniqueConstraint('run_id', name=op.f('uq_jobs_run_id'))
    )
    op.create_index('ix_jobs_available', 'jobs', ['available_at'], unique=False, postgresql_where=sa.text("status = 'queued'"))
    op.create_index(op.f('ix_jobs_current_attempt_id'), 'jobs', ['current_attempt_id'], unique=False)
    op.create_index('ix_jobs_lease_expiry', 'jobs', ['lease_expires_at'], unique=False, postgresql_where=sa.text("status = 'leased'"))
    op.create_table('model_calls',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column('attempt_id', sa.UUID(), nullable=False),
        sa.Column('event_sequence', sa.Integer(), nullable=False),
        sa.Column('turn', sa.Integer(), nullable=True),
        sa.Column('model', sa.String(length=200), nullable=False),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('cache_read_input_tokens', sa.Integer(), nullable=True),
        sa.Column('cache_creation_input_tokens', sa.Integer(), nullable=True),
        sa.Column('cost_usd', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('price_version_id', sa.Integer(), nullable=True),
        sa.Column('latency_ms', sa.Float(), nullable=True),
        sa.Column('stop_reason', sa.String(length=50), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['attempt_id'], ['attempts.id'], name=op.f('fk_model_calls_attempt_id_attempts')),
        sa.ForeignKeyConstraint(['price_version_id'], ['price_versions.id'], name=op.f('fk_model_calls_price_version_id_price_versions')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_model_calls')),
        sa.UniqueConstraint('attempt_id', 'event_sequence', name=op.f('uq_model_calls_attempt_id_event_sequence'))
    )
    op.create_table('scores',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column('attempt_id', sa.UUID(), nullable=False),
        sa.Column('run_id', sa.UUID(), nullable=False),
        sa.Column('experiment_id', sa.UUID(), nullable=False),
        sa.Column('task_category', sa.String(length=50), nullable=False),
        sa.Column('evaluator_name', sa.String(length=100), nullable=False),
        sa.Column('evaluator_version', sa.String(length=30), nullable=False),
        sa.Column('dimension', sa.String(length=100), server_default='overall', nullable=False),
        sa.Column('status', sa.String(length=30), nullable=False),
        sa.Column('value', sa.Float(), nullable=True),
        sa.Column('reason_code', sa.String(length=100), nullable=False),
        sa.Column('evidence', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('applicability', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("applicability IN ('applicable', 'not_applicable')", name=op.f('ck_scores_applicability')),
        sa.CheckConstraint("status <> 'evaluation_error' OR value IS NULL", name=op.f('ck_scores_error_has_no_value')),
        sa.CheckConstraint("status IN ('pending', 'pass', 'fail', 'partial', 'not_applicable', 'evaluation_error')", name=op.f('ck_scores_status')),
        sa.ForeignKeyConstraint(['attempt_id'], ['attempts.id'], name=op.f('fk_scores_attempt_id_attempts')),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], name=op.f('fk_scores_experiment_id_experiments')),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], name=op.f('fk_scores_run_id_runs')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_scores')),
        sa.UniqueConstraint('attempt_id', 'evaluator_name', 'evaluator_version', 'dimension', name='uq_scores_attempt_evaluator')
    )
    op.create_index('ix_scores_experiment_category', 'scores', ['experiment_id', 'task_category'], unique=False)
    op.create_index('ix_scores_experiment_evaluator', 'scores', ['experiment_id', 'evaluator_name', 'status'], unique=False)
    op.create_index(op.f('ix_scores_run_id'), 'scores', ['run_id'], unique=False)
    op.create_table('tool_calls',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column('attempt_id', sa.UUID(), nullable=False),
        sa.Column('attempt_sequence', sa.Integer(), nullable=False),
        sa.Column('call_id', sa.String(length=200), nullable=True),
        sa.Column('tool', sa.String(length=100), nullable=False),
        sa.Column('args', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=True),
        sa.Column('decision', sa.String(length=20), nullable=True),
        sa.Column('reason_code', sa.String(length=100), nullable=True),
        sa.Column('effect_status', sa.String(length=20), nullable=True),
        sa.Column('state_changes', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['attempt_id'], ['attempts.id'], name=op.f('fk_tool_calls_attempt_id_attempts')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_tool_calls')),
        sa.UniqueConstraint('attempt_id', 'attempt_sequence', name=op.f('uq_tool_calls_attempt_id_attempt_sequence'))
    )
    op.create_table('trace_events',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column('attempt_id', sa.UUID(), nullable=False),
        sa.Column('event_sequence', sa.Integer(), nullable=False),
        sa.Column('kind', sa.String(length=30), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('monotonic_ns', sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(['attempt_id'], ['attempts.id'], name=op.f('fk_trace_events_attempt_id_attempts')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_trace_events')),
        sa.UniqueConstraint('attempt_id', 'event_sequence', name=op.f('uq_trace_events_attempt_id_event_sequence'))
    )


def downgrade() -> None:
    op.drop_table('trace_events')
    op.drop_table('tool_calls')
    op.drop_index('ix_scores_experiment_category', table_name='scores')
    op.drop_index('ix_scores_experiment_evaluator', table_name='scores')
    op.drop_index(op.f('ix_scores_run_id'), table_name='scores')
    op.drop_table('scores')
    op.drop_table('model_calls')
    op.drop_index('ix_jobs_available', table_name='jobs', postgresql_where=sa.text("status = 'queued'"))
    op.drop_index(op.f('ix_jobs_current_attempt_id'), table_name='jobs')
    op.drop_index('ix_jobs_lease_expiry', table_name='jobs', postgresql_where=sa.text("status = 'leased'"))
    op.drop_table('jobs')
    op.drop_table('attempts')
    op.drop_index(op.f('ix_runs_agent_config_id'), table_name='runs')
    op.drop_index('ix_runs_experiment_status', table_name='runs')
    op.drop_index(op.f('ix_runs_task_version_id'), table_name='runs')
    op.drop_table('runs')
    op.drop_index('ix_task_versions_version_split', table_name='task_versions')
    op.drop_table('task_versions')
    op.drop_table('experiments')
    op.drop_table('dataset_versions')
    op.drop_table('price_versions')
    op.drop_table('datasets')
    op.drop_table('agent_configs')
