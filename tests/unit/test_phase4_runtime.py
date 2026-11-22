"""Phase 4 runtime pieces that need no database: cancellation, pinned environment ids, adapter registry."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.agents import HonestScriptedAgent, ToolLoopAgent
from app.agents.base import AgentResult, AgentRuntime
from app.agents.registry import DEFAULT_ADAPTERS, AgentConfigSpec, build_agent, limits_from
from app.environment import SupportEnvironment
from app.providers.fake import ScriptedProvider, final_text, tool_calls
from app.runner import AttemptCancelled, run_task_attempt
from app.tasks import AgentTaskView, load_task, task_path

REFUND = load_task(task_path("refund_eligible_001"))


class TwoLookups:
    name = "two_lookups"

    def __init__(self, on_first=lambda: None) -> None:
        self.on_first = on_first
        self.second = None

    def run(self, task: AgentTaskView, runtime: AgentRuntime) -> AgentResult:
        runtime.call_tool("get_order", {"order_id": "ORD-1001"})
        self.on_first()
        self.second = runtime.call_tool("get_return_status", {"order_id": "ORD-1001"})
        return AgentResult(final_response="done", stop_reason="completed")


def test_cancellation_stops_the_next_tool_call() -> None:
    flag = {"cancel": False}
    agent = TwoLookups(on_first=lambda: flag.update(cancel=True))

    result = run_task_attempt(REFUND.agent_view, REFUND.evaluator_view, agent, cancel_check=lambda: flag["cancel"])

    assert result.execution_status == "cancelled"
    assert agent.second is None
    assert [e.name for e in result.events if e.kind == "tool_attempt"] == ["get_order"]
    observed = next(e for e in result.events if e.name == "cancellation_observed")
    assert observed.payload == {"blocked_tool": "get_return_status"}
    assert result.scores == () and result.success is None and result.agent_result is None


def test_cancellation_before_start_never_runs_the_agent() -> None:
    agent = TwoLookups()
    result = run_task_attempt(REFUND.agent_view, REFUND.evaluator_view, agent, cancel_check=lambda: True)
    assert result.execution_status == "cancelled"
    assert not [e for e in result.events if e.kind == "tool_attempt"]
    assert result.final_snapshot == result.initial_snapshot


def test_an_adapter_cannot_swallow_cancellation() -> None:
    class Swallower:
        name = "swallower"

        def run(self, task: AgentTaskView, runtime: AgentRuntime) -> AgentResult:
            try:
                runtime.call_tool("get_order", {"order_id": "ORD-1001"})
            except Exception:  # noqa: BLE001 - the point of the test
                pass
            return AgentResult(final_response="refund issued", stop_reason="completed")

    assert not issubclass(AttemptCancelled, Exception)
    calls = iter([False, True])
    result = run_task_attempt(
        REFUND.agent_view, REFUND.evaluator_view, Swallower(), cancel_check=lambda: next(calls, True)
    )
    assert result.execution_status == "cancelled"


def test_environment_id_can_be_assigned_by_the_platform() -> None:
    result = run_task_attempt(REFUND.agent_view, REFUND.evaluator_view, HonestScriptedAgent(), environment_id="env-abc")
    assert result.environment_id == "env-abc"
    assert result.events[0].payload["environment_id"] == "env-abc"
    assert SupportEnvironment(REFUND.evaluator_view.fixture, environment_id="env-x").environment_id == "env-x"


def test_tool_allowlist_narrows_the_task_tools() -> None:
    seen = []

    class Recorder:
        name = "recorder"

        def run(self, task: AgentTaskView, runtime: AgentRuntime) -> AgentResult:
            seen.append((task.tools, tuple(t.name for t in runtime.tool_specs)))
            denied = runtime.call_tool("issue_refund", {"order_id": "ORD-1001", "amount": 1, "approval_id": "A"})
            seen.append(denied["error"]["code"])
            return AgentResult(final_response="", stop_reason="completed")

    run_task_attempt(REFUND.agent_view, REFUND.evaluator_view, Recorder(), allowed_tools=("get_order",))
    assert seen[0] == (("get_order",), ("get_order",))
    assert seen[1] == "tool_not_allowed"


def test_registry_builds_scripted_and_tool_loop_agents_without_network() -> None:
    honest = build_agent(AgentConfigSpec(name="h", adapter="honest_scripted"), DEFAULT_ADAPTERS)
    loop = build_agent(
        AgentConfigSpec(
            name="l", adapter="tool_loop", model="claude-opus-5-5", prompt="Be careful.",
            limits={"max_model_turns": 4, "max_cost_usd": "0.25"},
        ),
        DEFAULT_ADAPTERS,
    )
    assert isinstance(honest, HonestScriptedAgent)
    assert isinstance(loop, ToolLoopAgent)
    assert loop.system_prompt == "Be careful."
    assert loop.limits.max_model_turns == 4 and loop.limits.max_cost_usd == Decimal("0.25")
    assert loop.provider.model == "claude-opus-5-5"
    with pytest.raises(KeyError):
        build_agent(AgentConfigSpec(name="x", adapter="nope"), DEFAULT_ADAPTERS)


def test_limits_from_rejects_unknown_or_invalid_fields() -> None:
    assert limits_from({}).max_tool_calls == 20
    with pytest.raises(ValueError):
        limits_from({"max_turns": 3})
    with pytest.raises(ValueError):
        limits_from({"max_model_turns": 0})


def test_model_responses_name_their_price_version() -> None:
    provider = ScriptedProvider(
        [tool_calls(("get_order", {"order_id": "ORD-1001"})), final_text("ok")], model="claude-opus-5-5"
    )
    result = run_task_attempt(REFUND.agent_view, REFUND.evaluator_view, ToolLoopAgent(provider))
    responses = [e for e in result.events if e.kind == "model_response"]
    assert responses and all(e.payload["price_version_id"] == "anthropic-2026-10-06" for e in responses)
    assert all(e.payload["priced_model"] == "claude-opus-5-5" for e in responses)


def test_unexpected_provider_crash_is_unknown_usage_not_free() -> None:
    def crash(request):
        raise RuntimeError("response decoder crashed after sending request")

    provider = ScriptedProvider([crash], model="claude-opus-5-5")
    result = run_task_attempt(REFUND.agent_view, REFUND.evaluator_view, ToolLoopAgent(provider))
    assert result.execution_status == "infrastructure_error"
    assert result.agent_result is None
    assert result.usage.model_calls == 1
    assert result.usage.unknown_usage_calls == 1
    assert result.usage.cost_usd is None and result.usage.input_tokens is None
    responses = [e for e in result.events if e.kind == "model_response"]
    assert len(responses) == 1 and responses[0].payload["retryable"] is False


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_scores_become_evaluation_errors(number) -> None:
    from app.evaluators import Score
    from app.evaluators.contract import run_evaluator

    class BrokenEvaluator:
        name = "broken"
        version = "1"

        def evaluate(self, ctx):
            return Score(evaluator_name=self.name, evaluator_version=self.version,
                         status="pass", value=number, reason_code="BAD")

    score = run_evaluator(BrokenEvaluator(), None)
    assert score.status == "evaluation_error" and score.value is None


def test_json_sanitization_covers_nested_keys_values_and_nonfinite_numbers() -> None:
    from app.worker.queue import _jsonable

    raw = {"a\x00": ["b\x00", float("nan"), float("inf"), {"cost": Decimal("0.1234")} ]}
    assert _jsonable(raw) == {"a\ufffd": ["b\ufffd", None, None, {"cost": "0.1234"}]}
    assert "a\x00" in raw  # persistence sanitization cannot mutate evaluator evidence


def test_invalid_adapter_return_is_a_contained_adapter_failure():
    class InvalidAdapter:
        name = "invalid"

        def run(self, task, runtime):
            return None

    result = run_task_attempt(REFUND.agent_view, REFUND.evaluator_view, InvalidAdapter())
    assert result.execution_status == "infrastructure_error"
    assert result.agent_result is None
    assert next(e for e in result.events if e.name == "attempt_failed").payload["error_type"] == "TypeError"
