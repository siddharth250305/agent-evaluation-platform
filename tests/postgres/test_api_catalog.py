"""Dataset import, agent configuration registration and health endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.app import create_app
from app.datasets import load_dataset
from app.db.models import TaskVersion
from app.db.session import make_engine, make_sessionmaker
from app.tasks import DATASETS_DIR
from tests.postgres.helpers import client_for, import_support, register


def test_import_dataset_stores_an_immutable_version(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    on_disk = load_dataset(DATASETS_DIR / "support")

    first = client.post("/datasets/import", json={"dataset": "support"})
    again = client.post("/datasets/import", json={"dataset": "support"})

    assert first.status_code == 201 and again.status_code == 200
    assert first.json()["id"] == again.json()["id"]
    body = first.json()
    assert body["content_hash"] == on_disk.content_hash
    assert body["task_count"] == len(on_disk.tasks)
    assert body["split_counts"] == {s: len(on_disk.split(s)) for s in ("dev", "validation", "test")}
    with sm() as session:
        assert session.scalar(select(func.count()).select_from(TaskVersion)) == len(on_disk.tasks)


def test_stored_agent_view_holds_only_agent_visible_fields(sm: sessionmaker[Session]) -> None:
    import_support(client_for(sm))
    with sm() as session:
        views = session.scalars(select(TaskVersion.agent_view)).all()
    assert views and all(set(view) == {"user_request", "tools"} for view in views)


def test_import_rejects_unknown_or_unsafe_dataset_names(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    assert client.post("/datasets/import", json={"dataset": "nope"}).status_code == 404
    assert client.post("/datasets/import", json={"dataset": "../support"}).status_code == 422
    assert client.post("/datasets/import", json={"dataset": "support/fixtures"}).status_code == 422


def test_list_datasets_with_versions(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    version = import_support(client)
    listed = client.get("/datasets").json()
    assert [d["name"] for d in listed] == ["support"]
    assert [v["id"] for v in listed[0]["versions"]] == [version["id"]]


def test_agent_configs_are_immutable(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    body = {"name": "honest-v1", "adapter": "honest_scripted"}

    created = client.post("/agent-configs", json=body)
    same = client.post("/agent-configs", json=body)
    changed = client.post("/agent-configs", json={**body, "limits": {"max_tool_calls": 3}})

    assert created.status_code == 201 and same.status_code == 200
    assert created.json()["id"] == same.json()["id"]
    assert len(created.json()["config_hash"]) == 64
    assert changed.status_code == 409
    assert [c["name"] for c in client.get("/agent-configs").json()] == ["honest-v1"]


def test_agent_config_validation(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    assert client.post("/agent-configs", json={"name": "x", "adapter": "no_such_adapter"}).status_code == 422
    assert client.post("/agent-configs", json={"name": "Bad Name", "adapter": "honest_scripted"}).status_code == 422
    assert client.post("/agent-configs", json={"name": "t", "adapter": "tool_loop"}).status_code == 422  # needs a model
    bad_tool = {"name": "t2", "adapter": "honest_scripted", "tools": ["delete_everything"]}
    assert client.post("/agent-configs", json=bad_tool).status_code == 422
    bad_limit = {"name": "t3", "adapter": "honest_scripted", "limits": {"max_model_turns": 0}}
    assert client.post("/agent-configs", json=bad_limit).status_code == 422


def test_tool_loop_config_records_prompt_hash(sm: sessionmaker[Session]) -> None:
    config = register(
        client_for(sm), "opus-prompt-v1", "tool_loop", model="claude-opus-5-5", prompt="Be careful.",
        limits={"max_cost_usd": "0.50"},
    )
    assert config["model"] == "claude-opus-5-5"
    assert len(config["prompt_hash"]) == 64
    assert config["limits"]["max_cost_usd"] == "0.5"
    same_budget = client_for(sm).post(
        "/agent-configs",
        json={"name": "opus-prompt-v1", "adapter": "tool_loop", "model": "claude-opus-5-5", "prompt": "Be careful.",
              "limits": {"max_cost_usd": 0.5}},
    )
    assert same_budget.status_code == 200 and same_budget.json()["config_hash"] == config["config_hash"]


def test_health_live_and_ready(sm: sessionmaker[Session]) -> None:
    client = client_for(sm)
    assert client.get("/health/live").json() == {"status": "ok"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["database"] == "ok" and ready.json()["migrations"] == "head"


def test_ready_is_503_when_the_database_is_unreachable() -> None:
    # Port 1 on loopback: refused immediately, no network involved.
    engine = make_engine("postgresql+psycopg://eval:x@127.0.0.1:1/none", connect_args={"connect_timeout": 1})
    client = TestClient(create_app(sessionmaker=make_sessionmaker(engine)))
    assert client.get("/health/live").status_code == 200
    ready = client.get("/health/ready")
    assert ready.status_code == 503 and ready.json()["database"] == "unavailable"
    engine.dispose()
