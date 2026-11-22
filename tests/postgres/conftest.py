"""PostgreSQL fixtures: one throwaway database per test session, migrated with Alembic.

The server must be local (``make db-up``). If it is not reachable these tests
skip with the reason, unless ``REQUIRE_POSTGRES=1`` (``make test-db``), where
an unreachable database is a failure. Nothing here talks to a paid API.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.db.migrate import upgrade
from app.db.models import Base
from app.db.session import make_sessionmaker
from app.services.catalog import sync_price_versions

ADMIN_URL = os.environ.get(
    "TEST_DATABASE_ADMIN_URL", "postgresql+psycopg://eval:eval_local_only@127.0.0.1:5432/postgres"
)
_LOOPBACK = {"127.0.0.1", "localhost", "::1", None}
_KEEP = {"price_versions"}


def _admin_engine() -> Engine:
    url = make_url(ADMIN_URL)
    if url.host not in _LOOPBACK:
        raise RuntimeError(f"test database must be local, got host {url.host!r}")
    return create_engine(url, isolation_level="AUTOCOMMIT", connect_args={"connect_timeout": 3})


def create_database(prefix: str = "agent_eval_test") -> str:
    """Create an empty database and return its URL (as a string with the password)."""
    name = f"{prefix}_{uuid.uuid4().hex[:10]}"
    admin = _admin_engine()
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    return make_url(ADMIN_URL).set(database=name).render_as_string(hide_password=False)


def drop_database(url: str) -> None:
    admin = _admin_engine()
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{make_url(url).database}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture(scope="session")
def pg_available() -> None:
    try:
        engine = _admin_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:  # noqa: BLE001 - any connection failure means "no local Postgres"
        message = f"PostgreSQL not reachable at {make_url(ADMIN_URL).render_as_string()} ({type(exc).__name__}); run `make db-up`"
        if os.environ.get("REQUIRE_POSTGRES") == "1":
            pytest.fail(message)
        pytest.skip(message)


@pytest.fixture(scope="session")
def database_url(pg_available: None) -> Iterator[str]:
    url = create_database()
    upgrade(url)
    yield url
    drop_database(url)


@pytest.fixture(scope="session")
def engine(database_url: str) -> Iterator[Engine]:
    engine = create_engine(database_url, pool_size=10, max_overflow=10)
    yield engine
    engine.dispose()


@pytest.fixture
def sm(engine: Engine) -> sessionmaker[Session]:
    """A clean database for every test (price versions are reference data and stay)."""
    tables = ", ".join(t.name for t in Base.metadata.sorted_tables if t.name not in _KEEP)
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    maker = make_sessionmaker(engine)
    with maker.begin() as session:
        sync_price_versions(session)
    return maker
