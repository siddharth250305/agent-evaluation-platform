"""OpenAI Chat Completions adapter over a mock transport: no network, no key, $0."""

from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from app.accounting import cost_usd, price_for
from app.agents import ToolLoopAgent
from app.agents.registry import DEFAULT_ADAPTERS, AgentConfigSpec, build_agent
from app.providers import AnthropicMessagesProvider, Message, ModelRequest, OpenAIChatProvider, ProviderError, ToolCall
from app.providers import ToolSpec, Usage, factory
from app.providers.openai_http import build_request_body, parse_response
from app.runner import run_task_attempt
from app.tasks import load_task, task_path

KEY = "sk-test-DO-NOT-LEAK-0123456789"
REQUEST = ModelRequest(
    system="sys",
    messages=(Message(role="user", content="hi"),),
    tools=(ToolSpec(name="get_order", description="Look up an order", input_schema={"type": "object"}),),
    max_output_tokens=1000,
)


def _completion(message: dict, finish: str | None, usage: dict | None = None, **extra) -> dict:
    body = {"id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-4.1-mini-2025-04-14",
            "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}], **extra}
    if usage is not None:
        body["usage"] = usage
    return body


USAGE = {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165,
         "prompt_tokens_details": {"cached_tokens": 20}, "completion_tokens_details": {"reasoning_tokens": 0}}
TOOL_CALL = _completion({"content": None, "refusal": None, "annotations": [], "tool_calls": [
    {"id": "call_1", "type": "function", "function": {"name": "get_order", "arguments": '{"order_id": "ORD-1001"}'}},
]}, "tool_calls", USAGE)
FINAL = _completion({"content": "Your order shipped.", "refusal": None}, "stop", USAGE)


def _provider(handler, **kwargs) -> OpenAIChatProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAIChatProvider(client=client, **{"api_key": KEY, **kwargs})


def _reply(body: dict, status: int = 200):
    return lambda request: httpx.Response(status, json=body)


# --- request shape --------------------------------------------------------------


def test_request_body_shape() -> None:
    assert build_request_body("gpt-4.1-mini", REQUEST, reasoning_effort=None) == {
        "model": "gpt-4.1-mini",
        "max_completion_tokens": 1000,
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        "store": False,
        "service_tier": "default",
        "tools": [{"type": "function", "function": {
            "name": "get_order", "description": "Look up an order", "parameters": {"type": "object"}}}],
    }
    assert build_request_body("o4-mini", REQUEST, reasoning_effort="low")["reasoning_effort"] == "low"
    empty = ModelRequest(system="", messages=REQUEST.messages, tools=(), max_output_tokens=10)
    assert "tools" not in build_request_body("gpt-4.1-mini", empty, reasoning_effort=None)
    assert build_request_body("gpt-4.1-mini", empty, reasoning_effort=None)["messages"][0]["role"] == "user"


def test_assistant_turns_are_echoed_and_tool_results_use_tool_call_ids() -> None:
    echoed = parse_response(TOOL_CALL, fallback_model="gpt-4.1-mini")
    rebuilt = Message(role="assistant", content="", tool_calls=(ToolCall(id="call_2", name="get_order",
                                                                         arguments={"order_id": "ORD-2"}),))
    request = REQUEST.model_copy(update={"messages": (
        REQUEST.messages[0],
        Message(role="assistant", content="", tool_calls=echoed.tool_calls, provider_content=echoed.provider_content),
        Message(role="tool", tool_call_id="call_1", content='{"ok": true}'),
        rebuilt,
        Message(role="tool", tool_call_id="call_2", content='{"ok": false}', is_error=True),
    )})
    wire = build_request_body("gpt-4.1-mini", request, reasoning_effort=None)["messages"]
    assert wire[2] == {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                       "function": {"name": "get_order", "arguments": '{"order_id": "ORD-1001"}'}}]}
    assert "annotations" not in wire[2]  # response-only fields are never sent back
    assert wire[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}
    assert json.loads(wire[4]["tool_calls"][0]["function"]["arguments"]) == {"order_id": "ORD-2"}
    assert wire[5]["tool_call_id"] == "call_2"


# --- responses ------------------------------------------------------------------


def test_tool_call_response_sends_bearer_auth_and_parses_arguments() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(url=str(request.url), auth=request.headers["authorization"], body=json.loads(request.content))
        return httpx.Response(200, json=TOOL_CALL)

    response = _provider(handler).complete(REQUEST, timeout_seconds=5)
    assert seen["url"] == "https://api.openai.com/v1/chat/completions" and seen["auth"] == f"Bearer {KEY}"
    assert seen["body"]["store"] is False
    assert response.stop_reason == "tool_use" and response.text == ""
    assert response.tool_calls == (ToolCall(id="call_1", name="get_order", arguments={"order_id": "ORD-1001"}),)
    assert response.model == "gpt-4.1-mini-2025-04-14" and response.response_id == "chatcmpl-1"
    # prompt_tokens includes the cached part; Usage.input_tokens is the uncached part.
    assert response.usage == Usage(input_tokens=100, output_tokens=45, cache_read_input_tokens=20,
                                   cache_creation_input_tokens=0)
    expected = (100 * Decimal("0.40") + 20 * Decimal("0.10") + 45 * Decimal("1.60")) / 1_000_000
    assert cost_usd(response.usage, price_for("gpt-4.1-mini")) == expected


def test_final_text_response() -> None:
    response = _provider(_reply(FINAL)).complete(REQUEST, timeout_seconds=5)
    assert (response.text, response.tool_calls, response.stop_reason) == ("Your order shipped.", (), "end_turn")


def test_invalid_argument_json_is_passed_through_raw_for_the_gateway_to_reject() -> None:
    broken = _completion({"content": None, "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "get_order", "arguments": '{"order_id": '}}]}, "tool_calls")
    response = parse_response(broken, fallback_model="gpt-4.1-mini")
    assert response.tool_calls[0].arguments == '{"order_id": '
    assert response.provider_content[0]["tool_calls"][0]["function"]["arguments"] == '{"order_id": '


@pytest.mark.parametrize(("finish", "message", "expected"), [
    ("stop", {"content": "ok"}, "end_turn"),
    ("tool_calls", {"content": None}, "tool_use"),
    ("length", {"content": "trunc"}, "max_tokens"),
    ("content_filter", {"content": None}, "refusal"),
    ("stop", {"content": None, "refusal": "I can't help with that."}, "refusal"),
    (None, {"content": "?"}, "other"),
])
def test_stop_reasons_are_normalized(finish, message, expected) -> None:
    response = parse_response(_completion(message, finish), fallback_model="gpt-4.1-mini")
    assert response.stop_reason == expected
    if message.get("refusal"):
        assert response.text == "I can't help with that."


@pytest.mark.parametrize(("raw", "expected"), [
    (None, None),  # no usage object at all
    ({}, Usage(cache_creation_input_tokens=0)),
    ({"prompt_tokens": 50, "completion_tokens": 5}, Usage(output_tokens=5, cache_creation_input_tokens=0)),
    ({"prompt_tokens": 50, "prompt_tokens_details": {}}, Usage(cache_creation_input_tokens=0)),
])
def test_missing_usage_stays_unknown_and_costs_unknown(raw, expected) -> None:
    usage = parse_response(_completion({"content": "x"}, "stop", raw), fallback_model="gpt-4.1-mini").usage
    assert usage == expected
    assert cost_usd(usage, price_for("gpt-4.1-mini")) is None  # never zero, never free


# --- errors: retryable means "a new attempt may help", never an in-provider retry -----


@pytest.mark.parametrize(("status", "code", "retryable"), [
    (429, "rate_limit_exceeded", True),
    (429, "insufficient_quota", False),  # out of credit: waiting will not fix it
    (500, None, True), (502, None, True), (503, None, True), (408, None, True), (409, None, True),
    (400, "invalid_request_error", False), (401, "invalid_api_key", False), (403, None, False), (404, None, False),
])
def test_http_errors_become_provider_errors(status, code, retryable) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "nope", "type": "x", "code": code}})

    with pytest.raises(ProviderError) as raised:
        _provider(handler).complete(REQUEST, timeout_seconds=5)
    assert (raised.value.retryable, raised.value.status_code) == (retryable, status)
    assert len(calls) == 1  # exactly one request: the adapter never retries on its own


def test_non_json_error_bodies_are_reported() -> None:
    with pytest.raises(ProviderError, match="HTTP 502: provider request failed") as raised:
        _provider(lambda r: httpx.Response(502, text="bad gateway")).complete(REQUEST, timeout_seconds=5)
    assert raised.value.retryable


@pytest.mark.parametrize("exc", [httpx.ReadTimeout("slow"), httpx.ConnectError("refused")])
def test_timeouts_and_transport_errors_are_retryable(exc) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    with pytest.raises(ProviderError) as raised:
        _provider(handler).complete(REQUEST, timeout_seconds=0.01)
    assert raised.value.retryable and raised.value.status_code is None


def test_the_caller_timeout_is_passed_to_every_request() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.extensions["timeout"])
        return httpx.Response(200, json=FINAL)

    _provider(handler).complete(REQUEST, timeout_seconds=7.5)
    assert set(seen.values()) == {7.5}


def test_missing_api_key_fails_before_any_request(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request may be sent without a key")

    provider = OpenAIChatProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ProviderError, match="OPENAI_API_KEY is not set") as raised:
        provider.complete(REQUEST, timeout_seconds=1)
    assert not raised.value.retryable


# --- key and header redaction ---------------------------------------------------------


def test_keys_echoed_in_error_messages_are_redacted() -> None:
    body = {"error": {"message": f"Incorrect API key provided: sk-test-***-6789. Also {KEY}.", "code": "invalid_api_key"}}
    with pytest.raises(ProviderError) as raised:
        _provider(_reply(body, 401)).complete(REQUEST, timeout_seconds=5)
    message = str(raised.value)
    assert KEY not in message and "sk-test" not in message and "sk-[REDACTED]" in message
    assert raised.value.__cause__ is None


def test_transport_error_text_is_redacted_and_not_chained() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"proxy rejected Bearer {KEY}")

    with pytest.raises(ProviderError) as raised:
        _provider(handler).complete(REQUEST, timeout_seconds=1)
    assert KEY not in str(raised.value) and raised.value.__cause__ is None


def test_the_key_is_never_in_repr_trace_or_stored_assistant_content() -> None:
    refund = load_task(task_path("refund_eligible_001"))
    replies = iter([
        httpx.Response(200, json=TOOL_CALL),
        httpx.Response(401, json={"error": {"message": f"Incorrect API key provided: {KEY}", "code": "invalid_api_key"}}),
    ])
    provider = _provider(lambda request: next(replies))
    assert KEY not in repr(provider)
    result = run_task_attempt(refund.agent_view, refund.evaluator_view, ToolLoopAgent(provider))
    stored = json.dumps([e.payload for e in result.events], default=str)
    assert "get_order" in stored and KEY not in stored and "Bearer" not in stored
    assert result.agent_result.stop_reason == "provider_error"


# --- the full tool loop and the provider choice ------------------------------------------


def test_tool_loop_round_trip_records_real_cost_with_the_openai_price_version() -> None:
    refund = load_task(task_path("refund_eligible_001"))
    requests = []
    replies = iter([TOOL_CALL, FINAL])

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=next(replies))

    result = run_task_attempt(refund.agent_view, refund.evaluator_view, ToolLoopAgent(_provider(handler)))
    assert [m["role"] for m in requests[1]["messages"]] == ["system", "user", "assistant", "tool"]
    assert requests[1]["messages"][3]["tool_call_id"] == "call_1"
    assert result.agent_result.final_response == "Your order shipped."
    responses = [e.payload for e in result.events if e.kind == "model_response"]
    assert {r["price_version_id"] for r in responses} == {"openai-2026-11-22"}
    assert result.usage.cost_usd is not None and result.usage.unknown_usage_calls == 0


def test_factory_picks_the_adapter_and_key_from_the_model_name(monkeypatch) -> None:
    assert factory.provider_for("claude-sonnet-5-5") == "anthropic"
    assert {factory.provider_for(m) for m in ("gpt-4.1-mini", "o4-mini", "chatgpt-4o-latest")} == {"openai"}
    with pytest.raises(ValueError, match="no provider adapter"):
        factory.provider_for("llama-3")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    assert factory.missing_api_keys(["claude-opus-5-5", "gpt-4.1-mini", "gpt-4.1-mini"]) == ["OPENAI_API_KEY"]
    assert factory.resolve_effort("claude-opus-5-5", None) == "medium"
    assert factory.resolve_effort("gpt-4.1-mini", None) is None


def test_registry_builds_an_openai_tool_loop_for_gpt_configs() -> None:
    agent = build_agent(AgentConfigSpec(name="g", adapter="tool_loop", model="gpt-4.1-mini",
                                        limits={"max_cost_usd": "0.50"}), DEFAULT_ADAPTERS)
    assert isinstance(agent.provider, OpenAIChatProvider) and agent.price.price_version_id == "openai-2026-11-22"
    assert agent.name == "tool_loop:openai:gpt-4.1-mini"
    claude = build_agent(AgentConfigSpec(name="c", adapter="tool_loop", model="claude-opus-5-5"), DEFAULT_ADAPTERS)
    assert isinstance(claude.provider, AnthropicMessagesProvider)


@pytest.mark.parametrize("source", ["success", "unexpected", "validation", "headers", "masked_401", "truncated"])
def test_provider_boundary_redacts_all_error_and_response_paths(source, caplog) -> None:
    import logging

    def handler(request):
        if source == "unexpected":
            raise RuntimeError(f"Authorization: Bearer {KEY}")
        if source == "headers":
            logging.getLogger("httpcore.http11").warning("headers=%r", {"Authorization": f"Bearer {KEY}"})
        if source == "validation":
            return httpx.Response(200, json=_completion({"content": {"secret": KEY}}, "stop", USAGE))
        if source == "masked_401":
            return httpx.Response(401, json={"error": {"message": "Incorrect key sk-proj-prefix…TAILSECRET"}})
        if source == "truncated":
            return httpx.Response(502, text="x" * 190 + KEY)
        return httpx.Response(200, json=_completion({"content": f"Authorization: Bearer {KEY}"}, "stop", USAGE))

    task = load_task(task_path("refund_eligible_001"))
    result = run_task_attempt(task.agent_view, task.evaluator_view, ToolLoopAgent(_provider(handler)))
    recorded = json.dumps([e.payload for e in result.events], default=str) + caplog.text
    assert KEY not in recorded and "TAILSECRET" not in recorded
    assert "Bearer " not in recorded and "sk-test-DO" not in recorded


@pytest.mark.parametrize("raw", [
    {"prompt_tokens": 10, "completion_tokens": -2, "prompt_tokens_details": {"cached_tokens": 0}},
    {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 11}},
    {"prompt_tokens": True, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 0}},
    {"prompt_tokens": 10, "completion_tokens": "2", "prompt_tokens_details": {"cached_tokens": 0}},
])
def test_invalid_usage_cannot_create_negative_or_coerced_cost(raw) -> None:
    usage = parse_response(_completion({"content": "ok"}, "stop", raw), fallback_model="gpt-4.1-mini").usage
    assert cost_usd(usage, price_for("gpt-4.1-mini")) is None


def test_quota_type_is_nonretryable_even_without_a_code() -> None:
    with pytest.raises(ProviderError) as raised:
        _provider(_reply({"error": {"type": "insufficient_quota", "code": None}}, 429)).complete(REQUEST, timeout_seconds=1)
    assert raised.value.retryable is False


def test_nonstandard_nan_arguments_remain_invalid_json() -> None:
    body = _completion({"tool_calls": [{"id": "c", "function": {"name": "get_order", "arguments": '{"order_id": NaN}'}}]}, "tool_calls")
    assert isinstance(parse_response(body, fallback_model="gpt-4.1-mini").tool_calls[0].arguments, str)


def test_pilot_budget_reserves_full_context_and_stops_unknown_usage() -> None:
    from app.agents import Limits
    from app.accounting import Budget

    price = price_for("gpt-4.1-mini")
    budget = Budget(Decimal("0.50"), price)
    assert budget.reserve(REQUEST) >= (Decimal(1047576) * price.input_per_mtok + 1000 * price.output_per_mtok) / 1000000
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={k: v for k, v in TOOL_CALL.items() if k != "usage"})
    task = load_task(task_path("refund_eligible_001"))
    result = run_task_attempt(task.agent_view, task.evaluator_view,
                              ToolLoopAgent(_provider(handler), limits=Limits(max_cost_usd=Decimal("0.50"))))
    assert result.agent_result.stop_reason == "budget_exceeded" and len(calls) == 1
    assert result.usage.cost_usd is None


def test_openai_repeating_calls_stop_at_hard_tool_and_turn_caps() -> None:
    from app.agents import Limits

    task = load_task(task_path("refund_eligible_001"))
    for limits, stop in [(Limits(max_model_turns=2, max_cost_usd=Decimal("0.50")), "max_turns"),
                         (Limits(max_tool_calls=1, max_cost_usd=Decimal("0.50")), "max_tool_calls")]:
        seen = []
        def handler(request):
            seen.append(request)
            assert 0 < request.extensions["timeout"]["read"] <= limits.timeout_seconds
            return httpx.Response(200, json=TOOL_CALL)
        result = run_task_attempt(task.agent_view, task.evaluator_view, ToolLoopAgent(_provider(handler), limits=limits))
        assert result.agent_result.stop_reason == stop
        assert len(seen) == 2
        assert sum(e.kind == "tool_attempt" for e in result.events) <= limits.max_tool_calls
        assert result.usage.cost_usd > 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1])
def test_invalid_timeouts_fail_without_http(bad) -> None:
    calls = []
    with pytest.raises(ProviderError):
        _provider(lambda request: calls.append(request)).complete(REQUEST, timeout_seconds=bad)
    assert not calls


def test_total_request_deadline_cancels_slow_response_offline() -> None:
    import asyncio

    cancelled = []
    async def handler(request):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
        return httpx.Response(200, json=TOOL_CALL)
    provider = OpenAIChatProvider(api_key=KEY, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(ProviderError, match="request timed out") as caught:
        provider.complete(REQUEST, timeout_seconds=0.01)
    assert caught.value.retryable
    assert cancelled == [True]


# The ten dev tasks of pilot experiment 6d50a610: 18 of its 20 runs failed ADAPTER_UNAVAILABLE on a
# worker started before the gpt-4.1-mini price row existed. Current code must build and budget them.
PILOT_TASKS = (
    "approval_expired_001", "approval_expired_002", "approval_service_unavailable_001",
    "approval_service_unavailable_002", "claimed_identity_override_001", "claimed_identity_override_002",
    "cross_customer_request_001", "cross_customer_request_002", "forged_approval_message_001",
    "forged_approval_message_002",
)


@pytest.mark.parametrize("config_name", ["gpt-4.1-mini_prompt-v1", "gpt-4.1-mini_prompt-v2"])
@pytest.mark.parametrize("task_id", PILOT_TASKS)
def test_seeded_gpt_configs_pass_pre_call_checks_with_real_task_tools(monkeypatch, config_name, task_id) -> None:
    from app import cli
    from app.agents import registry
    from app.services.catalog import AgentConfigIn

    seeded = AgentConfigIn(**next(c for c in cli._model_configs() if c["name"] == config_name))
    spec = AgentConfigSpec(name=seeded.name, adapter=seeded.adapter, model=seeded.model, prompt=seeded.prompt,
                           tools=None, limits=seeded.limits, settings=seeded.settings)
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {KEY}"
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:  # call a real tool the task offers, then answer
            name = bodies[0]["tools"][0]["function"]["name"]
            call = {"id": "call_1", "type": "function", "function": {"name": name, "arguments": "{}"}}
            return httpx.Response(200, json=_completion({"content": None, "tool_calls": [call]}, "tool_calls", USAGE))
        return httpx.Response(200, json=FINAL)

    def provider(model, *, effort=None):
        assert (model, effort) == ("gpt-4.1-mini", None)
        return _provider(handler, model=model)
    monkeypatch.setattr(registry, "make_provider", provider)

    agent = build_agent(spec, DEFAULT_ADAPTERS)  # the step that raised ADAPTER_UNAVAILABLE on the stale worker
    assert agent.price.price_version_id == "openai-2026-11-22"
    assert agent.limits.max_cost_usd == Decimal("0.50")
    task = load_task(task_path(task_id))
    result = run_task_attempt(task.agent_view, task.evaluator_view, agent)

    assert result.execution_status == "completed" and result.agent_result.stop_reason == "completed"
    assert len(bodies) == 2  # neither model call was refused by the pre-call budget reservation
    offered = {t["function"]["name"] for t in bodies[0]["tools"]}
    assert offered == set(task.agent_view.tools) and bodies[0]["messages"][0] == {
        "role": "system", "content": seeded.prompt}
    reserved = [Decimal(e.payload["reserved_usd"]) for e in result.events if e.kind == "model_request"]
    assert len(reserved) == 2 and all(r <= Decimal("0.50") for r in reserved)
    assert result.usage.cost_usd == 2 * cost_usd(parse_response(FINAL, fallback_model="gpt-4.1-mini").usage,
                                                 agent.price)
    assert result.usage.price_version_ids == ("openai-2026-11-22",)
