"""FastAPI application factory. Run with ``uvicorn --factory app.api.app:create_app``."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session, sessionmaker

from app.agents.registry import DEFAULT_ADAPTERS, AdapterFactory
from app.api.routes import router
from app.db.session import make_engine, make_sessionmaker
from app.services.errors import ServiceError
from app.web.routes import STATIC_DIR
from app.web.routes import router as dashboard_router


def create_app(
    sessionmaker: sessionmaker[Session] | None = None,
    *,
    adapters: Mapping[str, AdapterFactory] = DEFAULT_ADAPTERS,
) -> FastAPI:
    """The API holds no work in memory: everything lives in PostgreSQL, so restarts lose nothing."""
    app = FastAPI(
        title="Agent Evaluation Platform",
        version="0.1.0",
        summary="Run agent configurations against versioned tasks and score the verified end state.",
    )
    # Created lazily: the engine connects on first use, so the API starts (and reports not-ready) without a DB.
    app.state.sessionmaker = sessionmaker or make_sessionmaker(make_engine())
    app.state.adapter_names = frozenset(adapters)
    app.include_router(router)
    app.include_router(dashboard_router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    return app
