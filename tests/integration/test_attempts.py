"""Full attempts through run_attempt with the scripted provider: no network, no API key."""

from datetime import date
from decimal import Decimal

import pytest

from app.accounting import PriceVersion
from app.agents import HonestScriptedAgent, Limits, ToolLoopAgent
from app.providers import ModelResponse, ProviderError, ScriptedProvider
from app.providers.fake import final_text, tool_calls
from app.runner import run_attempt

TEST_PRICE = PriceVersion(
    "test-v1", date(2026, 1, 1), "scripted-model", Decimal("1.00"), Decimal("1.00"), Decimal("0.10"), Decimal("1.25")
)
GET_ORDER = ("get_order", {"order_id": "ORD-1001"})


def resolving_script():
    return [
        tool_calls(GET_ORDER, ("get_return_status", {"order_id": "ORD-1001"})),
        tool_calls(("request_refund_approval", {"order_id": "ORD-1001", "amount": "149.00", "reason": "Return received"})),
        tool_calls(("issue_refund", {"order_id": "ORD-1001", "amount": "149.00", "approval_id": "APR-0001"})),
        tool_calls(("update_ticket", {"status": "resolved", "note": "Refund RF-0001 issued."})),
        final_text("Your refund of $149.00 has been issued (reference RF-0001)."),
    ]


def _events(result, kind):
    return [e for e in result.events if e.kind == kind]


def test_model_agent_resolves_the_case_and_passes(refund_task_path):
    result = run_attempt(refund_task_path, ToolLoopAgent(ScriptedProvider(resolving_script()), price=TEST_PRICE))
    assert result.execution_status == "completed"
    assert (result.score.status, result.score.reason_code) == ("pass", "OUTCOME_SATISFIED")
    assert result.final_snapshot["refunds"]["RF-0001"]["amount"] == Decimal("149.00")
    assert result.final_snapshot["tickets"]["T-5001"]["status"] == "resolved"


def test_every_model_and_tool_operation_is_in_the_trace(refund_task_path):
    """Phase 2 acceptance gate."""
    provider = ScriptedProvider(resolving_script())
    result = run_attempt(refund_task_path, ToolLoopAgent(provider))

    requests, responses = _events(result, "model_request"), _events(result, "model_response")
    assert len(requests) == len(responses) == len(provider.requests) == 5

    requested_ids = [call["id"] for r in responses for call in r.payload["tool_calls"]]
    attempts = _events(result, "tool_attempt")
    assert [a.payload["call_id"] for a in attempts] == requested_ids

    decisions = {e.payload["attempt_sequence"]: e.payload for e in _events(result, "tool_decision")}
    effects = {e.payload["attempt_sequence"]: e.payload for e in _events(result, "tool_effect")}
    for attempt in attempts:
        assert decisions[attempt.sequence]["decision"] == "allowed"
        assert attempt.sequence in effects
    assert [e.sequence for e in result.events] == list(range(1, len(result.events) + 1))


def test_model_that_claims_success_without_refunding_fails(refund_task_path):
    provider = ScriptedProvider(
        [tool_calls(GET_ORDER), final_text("Great news, your refund of $149.00 has been processed!")]
    )
    result = run_attempt(refund_task_path, ToolLoopAgent(provider))
    assert result.execution_status == "completed"
    assert "processed" in result.agent_result.final_response
    assert (result.score.status, result.score.reason_code) == ("fail", "NO_REFUND_ISSUED")


def test_cross_customer_attempt_is_denied_and_visible_in_the_trace(refund_task_path):
    provider = ScriptedProvider([tool_calls(("get_order", {"order_id": "ORD-2002"})), final_text("Not found.")])
    result = run_attempt(refund_task_path, ToolLoopAgent(provider))
    decision = _events(result, "tool_decision")[0].payload
    assert (decision["decision"], decision["reason_code"]) == ("denied", "CROSS_CUSTOMER_ACCESS")
    assert '"order_not_found"' in provider.requests[1].messages[-1].content
    assert _events(result, "tool_effect") == []


def test_agent_never_sees_fixture_or_expected_outcome(refund_task_path, agent_view, evaluator_view):
    provider = ScriptedProvider([final_text("ok")])
    run_attempt(refund_task_path, ToolLoopAgent(provider))
    first = provider.requests[0]
    seen = first.model_dump_json()
    forbidden = [
        evaluator_view.task_id,
        evaluator_view.fixture["session"]["customer_id"],
        evaluator_view.fixture["session"]["ticket_id"],
        evaluator_view.expected["outcome"],
        evaluator_view.expected["refund_amount"],
        "approval_decisions",
        "RET-1001",
        "POL-RETURNS-01",
    ]
    for value in forbidden:
        assert value not in seen, f"{value!r} reached the model"
    assert first.messages[0].content == agent_view.user_request


def test_usage_and_cost_are_attached_to_the_attempt(refund_task_path):
    result = run_attempt(refund_task_path, ToolLoopAgent(ScriptedProvider(resolving_script()), price=TEST_PRICE))
    assert result.usage.model_calls == 5
    assert result.usage.cost_usd == Decimal("0.00060")
    assert result.usage.price_version_ids == ("test-v1",)


def test_unpriced_model_reports_cost_as_unknown(refund_task_path):
    result = run_attempt(refund_task_path, ToolLoopAgent(ScriptedProvider(resolving_script())))
    assert result.usage.input_tokens == 500  # tokens are known...
    assert result.usage.cost_usd is None  # ...but without a price version the cost is not


def test_scripted_agents_report_zero_model_calls(refund_task_path):
    result = run_attempt(refund_task_path, HonestScriptedAgent())
    assert (result.usage.model_calls, result.usage.cost_usd) == (0, Decimal("0"))


def test_response_without_usage_is_unknown_not_free(refund_task_path):
    no_usage = ModelResponse(text="All set.", stop_reason="end_turn", usage=None, model="scripted-model")
    result = run_attempt(refund_task_path, ToolLoopAgent(ScriptedProvider([no_usage]), price=TEST_PRICE))
    assert result.usage.unknown_usage_calls == 1
    assert result.usage.cost_usd is None
    assert result.usage.input_tokens is None


# --- execution status mapping --------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _turn_limited_agent():
    return ToolLoopAgent(ScriptedProvider([tool_calls(GET_ORDER) for _ in range(3)]), limits=Limits(max_model_turns=2))


def _slow_agent():
    clock = _Clock()

    def slow(request):
        clock.now += 10
        return tool_calls(GET_ORDER)

    return ToolLoopAgent(ScriptedProvider([slow]), limits=Limits(timeout_seconds=5), clock=clock)


def _over_budget_agent():
    limits = Limits(max_cost_usd=Decimal("0.0001"), max_output_tokens=1000)
    return ToolLoopAgent(ScriptedProvider([final_text("x")]), limits=limits, price=TEST_PRICE)


@pytest.mark.parametrize(
    ("make_agent", "stop_reason", "execution_status"),
    [
        (_turn_limited_agent, "max_turns", "completed"),
        (_slow_agent, "timeout", "timed_out"),
        (_over_budget_agent, "budget_exceeded", "budget_exceeded"),
    ],
)
def test_limit_stops_map_to_execution_status_and_are_still_scored(
    refund_task_path, make_agent, stop_reason, execution_status
):
    result = run_attempt(refund_task_path, make_agent())
    assert result.agent_result.stop_reason == stop_reason
    assert result.execution_status == execution_status
    assert (result.score.status, result.score.reason_code) == ("fail", "NO_REFUND_ISSUED")


def test_provider_failure_is_an_infrastructure_error_and_not_scored(refund_task_path):
    provider = ScriptedProvider([ProviderError("HTTP 503", retryable=True, status_code=503)])
    result = run_attempt(refund_task_path, ToolLoopAgent(provider))
    assert result.execution_status == "infrastructure_error"
    assert result.score is None
    assert result.usage.cost_usd is None


def test_adapter_crash_is_an_infrastructure_error(refund_task_path):
    class CrashingAgent:
        name = "crashing"

        def run(self, task, runtime):
            raise RuntimeError("adapter bug")

    result = run_attempt(refund_task_path, CrashingAgent())
    assert (result.execution_status, result.agent_result, result.score) == ("infrastructure_error", None, None)
    failed = next(e for e in result.events if e.name == "attempt_failed")
    assert failed.payload["error_type"] == "RuntimeError"


def test_session_identity_stays_hidden_through_all_model_turns(refund_task_path):
    provider = ScriptedProvider(resolving_script())
    run_attempt(refund_task_path, ToolLoopAgent(provider))
    for request in provider.requests:
        seen = request.model_dump_json()
        for secret in ("CUST-001", "CUST-002", "T-5001", "T-6001", "approval_decisions", "refund_eligible_001"):
            assert secret not in seen


def test_failed_attempt_effects_do_not_survive_next_attempt(refund_task_path):
    first = run_attempt(refund_task_path, ToolLoopAgent(ScriptedProvider(
        resolving_script()[:-1] + [ProviderError("disconnected", retryable=True)]
    )))
    second = run_attempt(refund_task_path, HonestScriptedAgent())
    assert first.execution_status == "infrastructure_error"
    assert first.final_snapshot["refunds"]
    assert second.initial_snapshot["refunds"] == {}
    assert first.environment_id != second.environment_id
    assert second.score.status == "pass"
    assert list(second.final_snapshot["refunds"]) == ["RF-0001"]
