"""Claude Messages API adapter over raw HTTPX (POST /v1/messages, manual tool loop)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from app.providers.base import (
    Message,
    ModelRequest,
    ModelResponse,
    ProviderError,
    StopReason,
    ToolCall,
    Usage,
)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-opus-5-5"

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}
_RETRYABLE_STATUS = {408, 409, 429}


class AnthropicMessagesProvider:
    """One request per model turn. No automatic retries: infra retries are new attempts.

    Server-side refusal fallbacks are deliberately not enabled: they would switch
    models mid-run, and an agent configuration must stay a fixed model.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        api_key: str | None = None,
        client: httpx.Client | None = None,
        url: str = API_URL,
    ) -> None:
        self.model = model
        self._effort = effort  # pinned explicitly so the configuration is reproducible
        self._api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        self._client = client or httpx.Client()
        self._url = url

    def complete(self, request: ModelRequest, *, timeout_seconds: float) -> ModelResponse:
        if not self._api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set", retryable=False)
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        body = build_request_body(self.model, request, effort=self._effort)
        try:
            response = self._client.post(self._url, json=body, headers=headers, timeout=timeout_seconds)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"request timed out: {exc}", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(f"transport error: {exc}", retryable=True) from exc

        if response.status_code != 200:
            status = response.status_code
            raise ProviderError(
                f"HTTP {status}: {_error_message(response)}",
                retryable=status in _RETRYABLE_STATUS or status >= 500,
                status_code=status,
            )
        return parse_response(response.json(), fallback_model=self.model)


def build_request_body(model: str, request: ModelRequest, *, effort: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": request.max_output_tokens,
        "messages": _wire_messages(request.messages),
        "output_config": {"effort": effort},
    }
    if request.system:  # an empty system prompt is omitted rather than sent as empty text
        body["system"] = request.system
    if request.tools:
        body["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in request.tools
        ]
    return body


def _wire_messages(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    previous_role: str | None = None
    for message in messages:
        if message.role == "user":
            wire.append({"role": "user", "content": message.content})
        elif message.role == "assistant":
            if message.provider_content is not None:
                content = [dict(block) for block in message.provider_content]
            else:
                content = [{"type": "text", "text": message.content}] if message.content else []
                content += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in message.tool_calls
                ]
            wire.append({"role": "assistant", "content": content})
        else:
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
            }
            if message.is_error:
                block["is_error"] = True
            # All results for one assistant turn go back in a single user message.
            if previous_role == "tool":
                wire[-1]["content"].append(block)
            else:
                wire.append({"role": "user", "content": [block]})
        previous_role = message.role
    return wire


def parse_response(data: dict[str, Any], *, fallback_model: str) -> ModelResponse:
    blocks = data.get("content") or []
    return ModelResponse(
        text="".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
        tool_calls=tuple(
            ToolCall(id=b["id"], name=b["name"], arguments=b.get("input")) for b in blocks if b.get("type") == "tool_use"
        ),
        stop_reason=_STOP_REASONS.get(data.get("stop_reason") or "", "other"),
        usage=_usage(data.get("usage")),
        model=data.get("model") or fallback_model,
        response_id=data.get("id"),
        provider_content=tuple(blocks),
    )


def _usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    # Missing fields stay None (unknown); they are never defaulted to zero.
    return Usage(
        input_tokens=raw.get("input_tokens"),
        output_tokens=raw.get("output_tokens"),
        cache_read_input_tokens=raw.get("cache_read_input_tokens"),
        cache_creation_input_tokens=raw.get("cache_creation_input_tokens"),
    )


def _error_message(response: httpx.Response) -> str:
    try:
        return response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return response.text[:200]
