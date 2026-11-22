"""Scripted adapters and API shortcuts for the PostgreSQL tests. Zero paid API calls."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.agents import HonestScriptedAgent, Limits, ToolLoopAgent
from app.agents.base import AgentResult, AgentRuntime
from app.agents.registry import DEFAULT_ADAPTERS, AdapterFactory, AgentConfigSpec
from app.api.app import create_app
from app.providers.fake import ScriptedProvider, final_text, tool_calls
from app.tasks import AgentTaskView
from app.worker.worker import Worker


class CrashingAgent:
    name = "crashing"

    def run(self, task: AgentTaskView, runtime: AgentRuntime) -> AgentResult:
        raise RuntimeError("adapter crashed")


class ConcurrencyProbe:
    """Counts how many probed agents are executing at the same time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def __enter__(self) -> None:
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self.current -= 1


class SlowHonestAgent:
    name = "slow_honest"

    def __init__(self, delay: float, probe: ConcurrencyProbe | None = None) -> None:
        self.delay = delay
        self.probe = probe or ConcurrencyProbe()

    def run(self, task: AgentTaskView, runtime: AgentRuntime) -> AgentResult:
        with self.probe:
            time.sleep(self.delay)
            return HonestScriptedAgent().run(task, runtime)


def _scripted_tool_loop(spec: AgentConfigSpec) -> ToolLoopAgent:
    provider = ScriptedProvider(
        [
            tool_calls(("get_order", {"order_id": "ORD-1001"})),
            final_text("I looked up ORD-1001 for you."),
        ],
        model="claude-opus-5-5",
    )
    return ToolLoopAgent(provider, limits=Limits(max_model_turns=3))


PROBE = ConcurrencyProbe()

TEST_ADAPTERS: dict[str, AdapterFactory] = {
    **DEFAULT_ADAPTERS,
    "crashing": lambda spec: CrashingAgent(),
    "slow_honest": lambda spec: SlowHonestAgent(0.3, PROBE),
    "tool_loop_scripted": _scripted_tool_loop,
}


def client_for(sm: sessionmaker[Session], adapters: dict[str, AdapterFactory] | None = None) -> TestClient:
    return TestClient(create_app(sessionmaker=sm, adapters=adapters or TEST_ADAPTERS))


def import_support(client: TestClient) -> dict[str, Any]:
    response = client.post("/datasets/import", json={"dataset": "support"})
    assert response.status_code in (200, 201), response.text
    return response.json()


def register(client: TestClient, name: str, adapter: str, **extra: Any) -> dict[str, Any]:
    response = client.post("/agent-configs", json={"name": name, "adapter": adapter, **extra})
    assert response.status_code in (200, 201), response.text
    return response.json()


def experiment_payload(
    dataset_version_id: str, config_ids: list[str], *, split: str = "validation", repetitions: int = 1
) -> dict[str, Any]:
    return {
        "name": "phase-4 test",
        "dataset_version_id": dataset_version_id,
        "agent_config_ids": config_ids,
        "split": split,
        "repetitions": repetitions,
    }


def launch(
    client: TestClient,
    configs: list[tuple[str, str]],
    *,
    split: str = "validation",
    repetitions: int = 1,
) -> dict[str, Any]:
    """Import the support dataset, register ``(name, adapter)`` configs and create an experiment."""
    version = import_support(client)
    ids = [register(client, name, adapter)["id"] for name, adapter in configs]
    response = client.post(
        "/experiments",
        json=experiment_payload(version["id"], ids, split=split, repetitions=repetitions),
        headers={"Idempotency-Key": uuid.uuid4().hex},
    )
    assert response.status_code == 201, response.text
    return response.json()


def worker(sm: sessionmaker[Session], adapters: dict[str, AdapterFactory] | None = None, **kwargs: Any) -> Worker:
    kwargs.setdefault("lease_seconds", 30.0)
    kwargs.setdefault("heartbeat_seconds", 0.2)
    kwargs.setdefault("poll_seconds", 0.05)
    return Worker(sm, adapters=adapters or TEST_ADAPTERS, **kwargs)
