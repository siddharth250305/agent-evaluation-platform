import copy
import json

import httpx
import pytest

from app.providers import (
    AnthropicMessagesProvider,
    Message,
    ModelRequest,
    ProviderError,
    ScriptedProvider,
    ToolCall,
    ToolSpec,
    Usage,
)
from app.providers.anthropic_http import build_request_body, parse_response
from app.providers.fake import final_text, tool_calls

REQUEST = ModelRequest(
    system="sys",
    messages=(Message(role="user", content="hi"),),
    tools=(ToolSpec(name="get_order", description="Look up an order", input_schema={"type": "object"}),),
    max_output_tokens=1000,
)
API_RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_order", "input": {"order_id": "ORD-1001"}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 120, "output_tokens": 45, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
}


def _provider(handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return AnthropicMessagesProvider(client=client, **{"api_key": "test-key", **kwargs})


# --- scripted provider ---------------------------------------------------------


def test_scripted_provider_replays_steps_and_records_requests():
    first, second = tool_calls(("get_order", {"order_id": "ORD-1001"})), final_text("done")
    provider = ScriptedProvider([first, lambda request: second])
    assert provider.complete(REQUEST, timeout_seconds=1) is first
    assert provider.complete(REQUEST, timeout_seconds=1) is second
    assert provider.requests == [REQUEST, REQUEST]


def test_scripted_provider_raises_scripted_errors_and_when_exhausted():
    provider = ScriptedProvider([ProviderError("HTTP 529", retryable=True, status_code=529)])
    with pytest.raises(ProviderError) as scripted:
        provider.complete(REQUEST, timeout_seconds=1)
    assert scripted.value.retryable
    with pytest.raises(ProviderError) as exhausted:
        provider.complete(REQUEST, timeout_seconds=1)
    assert not exhausted.value.retryable


# --- Anthropic Messages adapter (mock transport, no network) ------------------


def test_request_body_shape():
    body = build_request_body("claude-opus-5-5", REQUEST, effort="medium")
    assert body == {
        "model": "claude-opus-5-5",
        "max_tokens": 1000,
        "system": "sys",
        "messages": [{"role": "user", "content": "hi"}],
        "output_config": {"effort": "medium"},
        "tools": [{"name": "get_order", "description": "Look up an order", "input_schema": {"type": "object"}}],
    }


def test_assistant_blocks_are_echoed_verbatim_and_tool_results_grouped():
    blocks = tuple(copy.deepcopy(API_RESPONSE["content"]))
    messages = (
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="Let me check.",
            tool_calls=(
                ToolCall(id="toolu_1", name="get_order", arguments={"order_id": "ORD-1001"}),
                ToolCall(id="toolu_2", name="get_order", arguments={"order_id": "ORD-9"}),
            ),
            provider_content=blocks,
        ),
        Message(role="tool", tool_call_id="toolu_1", content='{"ok": true}'),
        Message(role="tool", tool_call_id="toolu_2", content='{"ok": false}', is_error=True),
    )
    wire = build_request_body("m", REQUEST.model_copy(update={"messages": messages}), effort="medium")["messages"]

    assert wire[1] == {"role": "assistant", "content": list(blocks)}  # thinking block preserved unchanged
    assert wire[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"ok": true}'},
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": '{"ok": false}', "is_error": True},
        ],
    }
    assert len(wire) == 3


def test_assistant_without_provider_content_is_rebuilt():
    message = Message(
        role="assistant", content="Checking.", tool_calls=(ToolCall(id="t1", name="get_order", arguments={"x": 1}),)
    )
    wire = build_request_body("m", REQUEST.model_copy(update={"messages": (message,)}), effort="low")["messages"]
    assert wire[0]["content"] == [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "t1", "name": "get_order", "input": {"x": 1}},
    ]


def test_complete_sends_auth_headers_and_parses_the_response():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=API_RESPONSE)

    response = _provider(handler).complete(REQUEST, timeout_seconds=30)

    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "test-key"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["model"] == "claude-opus-5-5"
    assert response.text == "Let me check."
    assert response.tool_calls == (ToolCall(id="toolu_1", name="get_order", arguments={"order_id": "ORD-1001"}),)
    assert response.stop_reason == "tool_use"
    assert response.usage == Usage(
        input_tokens=120, output_tokens=45, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    assert (response.model, response.response_id) == ("claude-opus-5-5", "msg_1")
    assert response.provider_content[0]["type"] == "thinking"


@pytest.mark.parametrize(
    ("raw_usage", "expected"),
    [
        ("absent", None),
        ({"input_tokens": 5}, Usage(input_tokens=5)),
    ],
)
def test_missing_usage_stays_unknown(raw_usage, expected):
    data = {k: v for k, v in API_RESPONSE.items() if k != "usage"}
    if raw_usage != "absent":
        data["usage"] = raw_usage
    assert parse_response(data, fallback_model="m").usage == expected


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("end_turn", "end_turn"),
        ("stop_sequence", "end_turn"),
        ("tool_use", "tool_use"),
        ("max_tokens", "max_tokens"),
        ("refusal", "refusal"),
        ("pause_turn", "other"),
    ],
)
def test_stop_reasons_are_normalized(raw, normalized):
    assert parse_response({**API_RESPONSE, "stop_reason": raw}, fallback_model="m").stop_reason == normalized


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (500, True), (529, True), (400, False), (401, False)])
def test_http_errors_become_provider_errors(status, retryable):
    def handler(request):
        return httpx.Response(status, json={"type": "error", "error": {"type": "x", "message": "boom"}})

    with pytest.raises(ProviderError) as exc:
        _provider(handler).complete(REQUEST, timeout_seconds=30)
    assert (exc.value.retryable, exc.value.status_code) == (retryable, status)
    assert "boom" in str(exc.value)


def test_timeouts_are_retryable_provider_errors():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError) as exc:
        _provider(handler).complete(REQUEST, timeout_seconds=0.1)
    assert exc.value.retryable


def test_missing_api_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=API_RESPONSE)

    provider = AnthropicMessagesProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ProviderError):
        provider.complete(REQUEST, timeout_seconds=1)
    assert calls == []
