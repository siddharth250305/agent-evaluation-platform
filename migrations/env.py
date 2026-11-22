"""Alembic environment: online migrations against DATABASE_URL (or the URL passed by app.db.migrate)."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool

from app.db.models import Base
from app.db.session import database_url

config = context.config
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.attributes.get("database_url") or database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(config.attributes.get("database_url") or database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
