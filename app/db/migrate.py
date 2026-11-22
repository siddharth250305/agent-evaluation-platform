"""Run Alembic programmatically (``make migrate``, tests, readiness checks)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Connection

from app.db.session import database_url

ROOT = Path(__file__).resolve().parents[2]


def alembic_config(url: str | None = None) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    # Passed as an attribute, not an ini option, so a '%' in a password is never interpolated.
    config.attributes["database_url"] = url or database_url()
    return config


def upgrade(url: str | None = None, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str | None = None, revision: str = "-1") -> None:
    command.downgrade(alembic_config(url), revision)


def head_revision() -> str | None:
    return ScriptDirectory.from_config(alembic_config("postgresql://unused")).get_current_head()


def current_revision(connection: Connection) -> str | None:
    return MigrationContext.configure(connection).get_current_revision()


if __name__ == "__main__":
    upgrade()
    print(f"database at revision {head_revision()}")
