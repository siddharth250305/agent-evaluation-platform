"""OpenAI Chat Completions adapter over raw HTTPX (POST /v1/chat/completions, manual tool loop)."""

from __future__ import annotations

import asyncio
import json
import math
import logging
import os
import re
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

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4.1-mini"

_STOP_REASONS: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}
_RETRYABLE_STATUS = {408, 409, 429}
# A 429 that waiting cannot fix: the account is out of credit.
_NON_RETRYABLE_ERROR_CODES = {"insufficient_quota"}
# OpenAI echoes a masked key in 401 messages ("Incorrect API key provided: sk-ab***yz").
_KEY_LIKE = re.compile(r"sk-[A-Za-z0-9_*.\-]{3,}")


class _HTTPRedactionFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.secrets: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for secret in tuple(self.secrets):
            message = message.replace(secret, "[REDACTED]")
        if re.search(r"authorization|bearer\s", message, re.IGNORECASE):
            message = "[REDACTED HTTP diagnostic]"
        record.msg, record.args = _KEY_LIKE.sub("sk-[REDACTED]", message), ()
        record.exc_info = record.exc_text = None
        return True


class OpenAIChatProvider:
    """One request per model turn. No automatic retries: infra retries are new attempts.

    The API key is held in memory only: it never appears in errors, traces, ``repr`` or
    ``provider_content``, and requests set ``store: false``.
    """

    name = "openai"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
        client: httpx.Client | httpx.AsyncClient | None = None,
        url: str = API_URL,
    ) -> None:
        self.model = model
        # Only reasoning models accept reasoning_effort; None leaves it out of the request.
        self._reasoning_effort = reasoning_effort
        self._api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self._client = client
        self._url = url
        # HTTPX/httpcore debug logs can include response headers and exception
        # reprs. Scrub at the originating loggers before any handler sees them.
        for name in ("httpx", "httpcore.connection", "httpcore.http11", "httpcore.http2",
                     "httpcore.proxy", "httpcore.socks"):
            logger = logging.getLogger(name)
            filters = [f for f in logger.filters if isinstance(f, _HTTPRedactionFilter)]
            guard = filters[0] if filters else _HTTPRedactionFilter()
            if self._api_key:
                guard.secrets.add(self._api_key)
            if not filters:
                logger.addFilter(guard)

    def __repr__(self) -> str:
        return f"OpenAIChatProvider(model={self.model!r}, reasoning_effort={self._reasoning_effort!r})"

    def complete(self, request: ModelRequest, *, timeout_seconds: float) -> ModelResponse:
        # No vendor/HTTPX/validation exception containing response data may escape
        # into the runner's generic exception trace or worker logs.
        try:
            return self._complete(request, timeout_seconds=timeout_seconds)
        except ProviderError:
            raise
        except Exception:
            raise ProviderError("invalid provider response or request", retryable=False) from None

    def _complete(self, request: ModelRequest, *, timeout_seconds: float) -> ModelResponse:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ProviderError("request timeout must be finite and positive", retryable=False)
        if not self._api_key:
            raise ProviderError("OPENAI_API_KEY is not set", retryable=False)
        headers = {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }
        body = build_request_body(self.model, request, reasoning_effort=self._reasoning_effort)
        try:
            if isinstance(self._client, httpx.Client):
                # Synchronous injected clients are an offline-test seam only.
                response = self._client.post(self._url, json=body, headers=headers, timeout=timeout_seconds)
            else:
                response = asyncio.run(self._post_with_deadline(body, headers, timeout_seconds))
        except (httpx.TimeoutException, TimeoutError):
            raise ProviderError("request timed out", retryable=True) from None
        except httpx.TransportError:
            raise ProviderError("transport error", retryable=True) from None

        if response.status_code != 200:
            status = response.status_code
            message, code = _error_details(response)
            # Vendor diagnostics can contain masked, escaped or truncated keys.
            # Persist only our own descriptions, never the provider's error body.
            message = {
                400: "invalid request", 401: "authentication failed: sk-[REDACTED]",
                403: "access denied", 404: "not found", 408: "request timed out",
                409: "request conflict", 429: "quota exhausted" if code == "insufficient_quota" else "rate limited",
            }.get(status, "provider request failed")
            raise ProviderError(
                f"HTTP {status}: {self._redact(message)[:200]}",
                retryable=(status in _RETRYABLE_STATUS or 500 <= status < 600) and code not in _NON_RETRYABLE_ERROR_CODES,
                status_code=status,
            )
        return parse_response(self._clean(response.json()), fallback_model=self.model)

    async def _post_with_deadline(self, body: dict, headers: dict, timeout: float) -> httpx.Response:
        # Bound the whole exchange, including a response that continuously drips
        # bytes faster than the per-read timeout. Cancellation closes the request;
        # no background thread continues issuing requests after the deadline.
        async with asyncio.timeout(timeout):
            if isinstance(self._client, httpx.AsyncClient):
                return await self._client.post(self._url, json=body, headers=headers, timeout=timeout)
            async with httpx.AsyncClient() as client:
                return await client.post(self._url, json=body, headers=headers, timeout=timeout)

    def _clean(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, list):
            return [self._clean(v) for v in value]
        if isinstance(value, dict):
            return {self._redact(k): "[REDACTED]" if k.lower() == "authorization" else self._clean(v)
                    for k, v in value.items()}
        return value

    def _redact(self, text: str) -> str:
        if re.search(r"authorization|bearer\s", text, re.IGNORECASE):
            return "[REDACTED provider diagnostic]"
        if self._api_key:
            text = text.replace(self._api_key, "[REDACTED]")
        return _KEY_LIKE.sub("sk-[REDACTED]", text)


def build_request_body(model: str, request: ModelRequest, *, reasoning_effort: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "max_completion_tokens": request.max_output_tokens,
        "messages": _wire_messages(request.system, request.messages),
        "service_tier": "default",  # pin the Standard-tier price used by the ledger
        "store": False,  # do not keep completions on OpenAI's side
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    if request.tools:
        body["tools"] = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
            for t in request.tools
        ]
    return body


def _wire_messages(system: str, messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    # An empty system prompt is omitted rather than sent as empty text.
    wire: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for message in messages:
        if message.role == "user":
            wire.append({"role": "user", "content": message.content})
        elif message.role == "assistant":
            if message.provider_content:
                wire.append(dict(message.provider_content[0]))
            else:
                wire.append(_assistant_message(
                    message.content or None,
                    [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls],
                ))
        else:
            # The tool loop's result content already carries "ok": false on errors; Chat
            # Completions has no separate is_error flag.
            wire.append({"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content})
    return wire


def _assistant_message(content: str | None, calls: list[dict[str, Any]], refusal: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        out["tool_calls"] = [
            {
                "id": c["id"],
                "type": "function",
                "function": {
                    "name": c["name"],
                    # Echo the model's raw argument string unchanged; re-serialize only parsed values.
                    "arguments": c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"]),
                },
            }
            for c in calls
        ]
    if refusal:
        out["refusal"] = refusal
    return out


def parse_response(data: dict[str, Any], *, fallback_model: str) -> ModelResponse:
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    raw_calls = [c for c in message.get("tool_calls") or [] if c.get("type", "function") == "function"]
    refusal = message.get("refusal")
    stop_reason: StopReason = "refusal" if refusal else _STOP_REASONS.get(choice.get("finish_reason") or "", "other")
    return ModelResponse(
        text=message.get("content") or refusal or "",
        tool_calls=tuple(
            ToolCall(id=c["id"], name=c["function"]["name"], arguments=_arguments(c["function"].get("arguments")))
            for c in raw_calls
        ),
        stop_reason=stop_reason,
        usage=_usage(data.get("usage")),
        model=data.get("model") or fallback_model,
        response_id=data.get("id"),
        # Only fields Chat Completions accepts back on an assistant message.
        provider_content=(_assistant_message(
            message.get("content"),
            [{"id": c["id"], "name": c["function"]["name"], "arguments": c["function"].get("arguments", "")}
             for c in raw_calls],
            refusal,
        ),),
    )


def _arguments(raw: Any) -> Any:
    """Arguments arrive as a JSON string. Invalid JSON is passed through raw for the tool gateway to reject."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("invalid JSON constant")))
    except ValueError:
        return raw


def _usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    def count(value: Any) -> int | None:
        return value if type(value) is int and value >= 0 else None

    prompt, completion = count(raw.get("prompt_tokens")), count(raw.get("completion_tokens"))
    details = raw.get("prompt_tokens_details")
    cached = count(details.get("cached_tokens")) if isinstance(details, dict) else None
    # prompt_tokens includes cached tokens; Usage.input_tokens is the uncached part. If the
    # split is not reported, both stay None (unknown), never zero. Reasoning tokens are part
    # of completion_tokens and billed as output.
    both = prompt is not None and cached is not None and cached <= prompt
    return Usage(
        input_tokens=prompt - cached if both else None,
        output_tokens=completion,
        cache_read_input_tokens=cached if both else None,
        cache_creation_input_tokens=0,  # OpenAI caching has no separately billed cache writes
    )


def _error_details(response: httpx.Response) -> tuple[str, str | None]:
    try:
        error = response.json()["error"]
        code = error.get("code") or error.get("type")
        if error.get("type") == "insufficient_quota":
            code = "insufficient_quota"
        return str(error.get("message") or ""), code
    except (ValueError, KeyError, TypeError, AttributeError):
        return response.text, None
