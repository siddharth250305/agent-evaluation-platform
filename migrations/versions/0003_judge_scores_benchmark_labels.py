"""Optional LLM judge scores (kept apart from evaluator scores) and honest benchmark subset labels.

Revision ID: 0003_judge_benchmark_labels
Revises: 0002_benchmark_results
Create Date: 2026-11-21

tests/postgres/test_migrations.py checks that models and migration agree.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_judge_benchmark_labels"
down_revision: str | None = "0002_benchmark_results"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('judge_scores',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column('attempt_id', sa.UUID(), nullable=False),
        sa.Column('run_id', sa.UUID(), nullable=False),
        sa.Column('experiment_id', sa.UUID(), nullable=False),
        sa.Column('judge_name', sa.String(length=100), nullable=False),
        sa.Column('judge_version', sa.String(length=30), nullable=False),
        sa.Column('rubric_sha256', sa.String(length=64), nullable=False),
        sa.Column('model', sa.String(length=200), nullable=False),
        sa.Column('resolved_model', sa.String(length=200), nullable=True),
        sa.Column('status', sa.String(length=30), nullable=False),
        sa.Column('scores', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('reason_code', sa.String(length=100), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('cache_read_input_tokens', sa.Integer(), nullable=True),
        sa.Column('cache_creation_input_tokens', sa.Integer(), nullable=True),
        sa.Column('cost_usd', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('price_version_id', sa.String(length=100), nullable=True),
        sa.Column('latency_ms', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("status IN ('scored', 'evaluation_error')", name=op.f('ck_judge_scores_status')),
        sa.CheckConstraint("status <> 'evaluation_error' OR scores IS NULL", name=op.f('ck_judge_scores_error_has_no_scores')),
        sa.ForeignKeyConstraint(['attempt_id'], ['attempts.id'], name=op.f('fk_judge_scores_attempt_id_attempts')),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], name=op.f('fk_judge_scores_experiment_id_experiments')),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], name=op.f('fk_judge_scores_run_id_runs')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_judge_scores')),
        sa.UniqueConstraint('attempt_id', 'judge_name', 'judge_version', name='uq_judge_scores_attempt_judge'),
    )
    op.create_index('ix_judge_scores_experiment_judge', 'judge_scores', ['experiment_id', 'judge_name'], unique=False)
    op.create_index(op.f('ix_judge_scores_run_id'), 'judge_scores', ['run_id'], unique=False)

    op.add_column('benchmark_results', sa.Column('category_task_count', sa.Integer(), nullable=True))
    op.add_column('benchmark_results', sa.Column('subset_method', sa.Text(), nullable=True))
    op.add_column('benchmark_results', sa.Column('result_kind', sa.String(length=30), server_default='model_run', nullable=False))
    op.create_check_constraint(
        op.f('ck_benchmark_results_result_kind'), 'benchmark_results',
        "result_kind IN ('model_run', 'scorer_self_check')",
    )
    op.create_check_constraint(
        op.f('ck_benchmark_results_subset_within_category'), 'benchmark_results',
        'category_task_count IS NULL OR category_task_count >= task_count',
    )
    op.create_check_constraint(
        op.f('ck_benchmark_results_subset_is_labeled'), 'benchmark_results',
        'category_task_count IS NULL OR task_count = category_task_count OR subset_method IS NOT NULL',
    )


def downgrade() -> None:
    op.drop_constraint(op.f('ck_benchmark_results_subset_is_labeled'), 'benchmark_results', type_='check')
    op.drop_constraint(op.f('ck_benchmark_results_subset_within_category'), 'benchmark_results', type_='check')
    op.drop_constraint(op.f('ck_benchmark_results_result_kind'), 'benchmark_results', type_='check')
    op.drop_column('benchmark_results', 'result_kind')
    op.drop_column('benchmark_results', 'subset_method')
    op.drop_column('benchmark_results', 'category_task_count')
    op.drop_index(op.f('ix_judge_scores_run_id'), table_name='judge_scores')
    op.drop_index('ix_judge_scores_experiment_judge', table_name='judge_scores')
    op.drop_table('judge_scores')
