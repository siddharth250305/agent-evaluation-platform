import socket

import pytest

from app.tasks import AgentTaskView, EvaluatorTaskView, load_agent_view, load_evaluator_view, task_path

REFUND_TASK = task_path("refund_eligible_001")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee zero paid API calls: any outbound connection fails the test."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise RuntimeError("network access is disabled in the deterministic test suite")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def refund_task_path():
    return REFUND_TASK


@pytest.fixture
def agent_view() -> AgentTaskView:
    return load_agent_view(REFUND_TASK)


@pytest.fixture
def evaluator_view() -> EvaluatorTaskView:
    return load_evaluator_view(REFUND_TASK)
