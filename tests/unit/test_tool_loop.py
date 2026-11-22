import json
from datetime import date
from decimal import Decimal

import pytest

from app.accounting import PriceVersion, UsageLedger
from app.agents import AgentRuntime, Limits, ToolLoopAgent
from app.environment import SupportEnvironment
from app.providers import ModelResponse, ProviderError, ScriptedProvider, ToolCall
from app.providers.fake import final_text, known_usage, tool_calls
from app.tools import CustomerContext, ToolGateway
from app.trace import TraceRecorder

TEST_PRICE = PriceVersion(
    "test-v1", date(2026, 1, 1), "scripted-model", Decimal("1.00"), Decimal("1.00"), Decimal("0.10"), Decimal("1.25")
)
GET_ORDER = ("get_order", {"order_id": "ORD-1001"})


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def runtime(evaluator_view, agent_view):
    env = SupportEnvironment(evaluator_view.fixture)
    trace = TraceRecorder()
    gateway = ToolGateway(env, CustomerContext("CUST-001", "T-5001"), trace, agent_view.tools)
    return AgentRuntime(call_tool=gateway.call, tool_specs=gateway.tool_specs, trace=trace, ledger=UsageLedger())


def _kinds(runtime, kind):
    return [e for e in runtime.trace.events if e.kind == kind]


def _unknown_usage_tool_call(call_id):
    return ModelResponse(
        text="",
        tool_calls=(ToolCall(id=call_id, name="get_order", arguments={"order_id": "ORD-1001"}),),
        stop_reason="tool_use",
        usage=None,
        model="scripted-model",
    )


# --- normal operation ----------------------------------------------------------


def test_completes_when_the_model_stops_calling_tools(runtime, agent_view):
    provider = ScriptedProvider([tool_calls(GET_ORDER), final_text("Your order total is $149.00.")])
    result = ToolLoopAgent(provider).run(agent_view, runtime)
    assert result.stop_reason == "completed"
    assert result.final_response == "Your order total is $149.00."
    assert (result.metadata["model_turns"], result.metadata["tool_calls"]) == (2, 1)


def test_first_request_is_only_the_customer_message(runtime, agent_view):
    provider = ScriptedProvider([final_text("ok")])
    ToolLoopAgent(provider).run(agent_view, runtime)
    first = provider.requests[0]
    assert first.messages == (first.messages[0],)
    assert first.messages[0].content == agent_view.user_request
    assert [t.name for t in first.tools] == list(agent_view.tools)


def test_tool_results_are_threaded_back_to_the_model(runtime, agent_view):
    provider = ScriptedProvider([tool_calls(GET_ORDER), final_text("done")])
    ToolLoopAgent(provider).run(agent_view, runtime)
    second = provider.requests[1].messages
    assert [m.role for m in second] == ["user", "assistant", "tool"]
    assert second[2].tool_call_id == second[1].tool_calls[0].id
    outcome = json.loads(second[2].content)
    assert outcome["ok"] and outcome["result"]["total"] == "149.00"


def test_invalid_tool_arguments_are_returned_to_the_model_as_errors(runtime, agent_view):
    provider = ScriptedProvider([tool_calls(("get_order", {"order_id": 1001})), final_text("sorry")])
    result = ToolLoopAgent(provider).run(agent_view, runtime)
    tool_message = provider.requests[1].messages[2]
    assert tool_message.is_error
    assert json.loads(tool_message.content)["error"]["code"] == "invalid_arguments"
    assert result.stop_reason == "completed"


def test_every_model_call_is_traced(runtime, agent_view):
    provider = ScriptedProvider([tool_calls(GET_ORDER), final_text("done")])
    ToolLoopAgent(provider).run(agent_view, runtime)
    requests, responses = _kinds(runtime, "model_request"), _kinds(runtime, "model_response")
    assert len(requests) == len(responses) == len(provider.requests) == 2
    assert [len(r.payload["new_messages"]) for r in requests] == [1, 2]  # user; then assistant + tool


def test_costs_are_recorded_per_call(runtime, agent_view):
    provider = ScriptedProvider([tool_calls(GET_ORDER), final_text("done")])
    ToolLoopAgent(provider, price=TEST_PRICE).run(agent_view, runtime)
    summary = runtime.ledger.summary()
    assert summary.model_calls == 2
    assert summary.cost_usd == Decimal("0.00024")  # 2 x (100 in + 20 out) at $1/MTok
    assert summary.price_version_ids == ("test-v1",)


# --- limits ----------------------------------------------------------------------


def test_max_model_turns(runtime, agent_view):
    provider = ScriptedProvider([tool_calls(GET_ORDER) for _ in range(5)])
    result = ToolLoopAgent(provider, limits=Limits(max_model_turns=2)).run(agent_view, runtime)
    assert result.stop_reason == "max_turns"
    assert len(provider.requests) == 2
    assert runtime.trace.events[-1].payload["stop_reason"] == "max_turns"


def test_max_tool_calls(runtime, agent_view):
    provider = ScriptedProvider([tool_calls(GET_ORDER, GET_ORDER, GET_ORDER)])
    result = ToolLoopAgent(provider, limits=Limits(max_tool_calls=2)).run(agent_view, runtime)
    assert result.stop_reason == "max_tool_calls"
    assert len(_kinds(runtime, "tool_attempt")) == 2


def test_timeout_stops_before_running_tools(runtime, agent_view):
    clock = FakeClock()
    provider = ScriptedProvider([tool_calls(GET_ORDER)], on_call=lambda: clock.advance(61))
    agent = ToolLoopAgent(provider, limits=Limits(timeout_seconds=60), clock=clock)
    result = agent.run(agent_view, runtime)
    assert result.stop_reason == "timeout"
    assert _kinds(runtime, "tool_attempt") == []


def test_late_final_response_is_timeout_with_usage_preserved(runtime, agent_view):
    clock = FakeClock()
    provider = ScriptedProvider([final_text("late")], on_call=lambda: clock.advance(61))
    result = ToolLoopAgent(provider, limits=Limits(timeout_seconds=60), clock=clock).run(agent_view, runtime)
    assert result.stop_reason == "timeout"
    assert runtime.ledger.summary().model_calls == 1


def test_budget_blocks_a_call_it_cannot_afford(runtime, agent_view):
    provider = ScriptedProvider([final_text("never sent")])
    limits = Limits(max_cost_usd=Decimal("0.001"), max_output_tokens=2000)  # output alone reserves $0.002
    result = ToolLoopAgent(provider, limits=limits, price=TEST_PRICE).run(agent_view, runtime)
    assert result.stop_reason == "budget_exceeded"
    assert provider.requests == []


def test_actual_usage_over_budget_stops_before_tools(runtime, agent_view):
    expensive = tool_calls(GET_ORDER, usage=known_usage(input_tokens=100_000, output_tokens=0))  # $0.10
    provider = ScriptedProvider([expensive])
    limits = Limits(max_cost_usd=Decimal("0.05"), max_output_tokens=1000)
    result = ToolLoopAgent(provider, limits=limits, price=TEST_PRICE).run(agent_view, runtime)
    assert result.stop_reason == "budget_exceeded"
    assert _kinds(runtime, "tool_attempt") == []
    assert runtime.ledger.summary().cost_usd == Decimal("0.1")


def test_unknown_usage_counts_its_reservation_against_the_budget(runtime, agent_view):
    # Output reservation is $0.10 per call; the budget fits one reservation but not two.
    limits = Limits(max_cost_usd=Decimal("0.15"), max_output_tokens=100_000)
    provider = ScriptedProvider([_unknown_usage_tool_call("u1"), final_text("done")])
    result = ToolLoopAgent(provider, limits=limits, price=TEST_PRICE).run(agent_view, runtime)
    assert result.stop_reason == "budget_exceeded"
    assert len(provider.requests) == 1
    summary = runtime.ledger.summary()
    assert (summary.unknown_usage_calls, summary.cost_usd) == (1, None)


def test_known_small_usage_does_not_exhaust_the_same_budget(runtime, agent_view):
    limits = Limits(max_cost_usd=Decimal("0.15"), max_output_tokens=100_000)
    provider = ScriptedProvider([tool_calls(GET_ORDER), final_text("done")])
    result = ToolLoopAgent(provider, limits=limits, price=TEST_PRICE).run(agent_view, runtime)
    assert result.stop_reason == "completed"
    assert len(provider.requests) == 2


def test_a_budget_without_a_price_is_rejected_up_front():
    with pytest.raises(ValueError):
        ToolLoopAgent(ScriptedProvider([]), limits=Limits(max_cost_usd=Decimal("1")))


def test_provider_error_stops_the_loop_with_unknown_usage(runtime, agent_view):
    provider = ScriptedProvider([ProviderError("HTTP 529", retryable=True, status_code=529)])
    result = ToolLoopAgent(provider).run(agent_view, runtime)
    assert result.stop_reason == "provider_error"
    assert result.metadata["retryable"] is True
    summary = runtime.ledger.summary()
    assert (summary.model_calls, summary.unknown_usage_calls, summary.cost_usd) == (1, 1, None)


@pytest.mark.parametrize(("provider_stop", "stop_reason"), [("refusal", "refused"), ("max_tokens", "output_truncated")])
def test_refused_or_truncated_turns_never_run_tools(runtime, agent_view, provider_stop, stop_reason):
    response = tool_calls(GET_ORDER).model_copy(update={"stop_reason": provider_stop})
    result = ToolLoopAgent(ScriptedProvider([response])).run(agent_view, runtime)
    assert result.stop_reason == stop_reason
    assert _kinds(runtime, "tool_attempt") == []


@pytest.mark.parametrize(
    "kwargs", [{"max_model_turns": 0}, {"timeout_seconds": 0}, {"max_cost_usd": Decimal("0")}]
)
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Limits(**kwargs)
