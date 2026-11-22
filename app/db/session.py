"""Engine and session factories. The URL comes from ``DATABASE_URL``."""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

# Matches compose.yaml's local-only defaults; real deployments set DATABASE_URL.
DEFAULT_DATABASE_URL = "postgresql+psycopg://eval:eval_local_only@127.0.0.1:5432/agent_eval"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def make_engine(url: str | None = None, **kwargs: Any) -> Engine:
    kwargs.setdefault("pool_pre_ping", True)
    return create_engine(url or database_url(), **kwargs)


def make_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    # expire_on_commit=False: objects stay readable after the transaction that loaded them.
    return sessionmaker(engine, expire_on_commit=False)
