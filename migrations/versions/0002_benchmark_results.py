"""Benchmark results: pinned public benchmark numbers, kept apart from custom support scores.

Revision ID: 0002_benchmark_results
Revises: 0001_initial
Create Date: 2026-11-21

tests/postgres/test_migrations.py checks that models and migration agree.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_benchmark_results"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('benchmark_results',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('benchmark', sa.String(length=100), nullable=False),
        sa.Column('benchmark_version', sa.String(length=100), nullable=False),
        sa.Column('category', sa.String(length=200), nullable=False),
        sa.Column('model', sa.String(length=200), nullable=True),
        sa.Column('task_count', sa.Integer(), nullable=False),
        sa.Column('scoring_method', sa.String(length=200), nullable=False),
        sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('raw_artifact_uri', sa.Text(), nullable=True),
        sa.Column('manifest', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('imported_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('task_count >= 0', name=op.f('ck_benchmark_results_task_count')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_benchmark_results'))
    )
    op.create_index('ix_benchmark_results_benchmark_version', 'benchmark_results', ['benchmark', 'benchmark_version'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_benchmark_results_benchmark_version', table_name='benchmark_results')
    op.drop_table('benchmark_results')
