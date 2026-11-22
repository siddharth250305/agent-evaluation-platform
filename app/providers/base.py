"""Provider-neutral model interface. Adapters translate to and from a vendor's wire format."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolSpec(_Frozen):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(_Frozen):
    id: str
    name: str
    arguments: Any  # raw model output; the tool gateway validates it


class Message(_Frozen):
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    is_error: bool = False
    # Opaque assistant content to echo back verbatim on the next request
    # (e.g. thinking blocks, which must be returned unchanged).
    provider_content: tuple[dict[str, Any], ...] | None = None


class Usage(_Frozen):
    """Token usage as reported by the provider. ``None`` means not reported (unknown), never zero."""

    input_tokens: int | None = None  # uncached input
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]


class ModelRequest(_Frozen):
    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    max_output_tokens: int


class ModelResponse(_Frozen):
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: StopReason
    usage: Usage | None  # None when the provider sent no usage at all
    model: str  # resolved model id as reported by the provider
    response_id: str | None = None
    provider_content: tuple[dict[str, Any], ...] | None = None


class ProviderError(Exception):
    """The provider call failed. ``retryable`` marks infrastructure failures (429, 5xx, network)."""

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class ModelProvider(Protocol):
    name: str
    model: str

    def complete(self, request: ModelRequest, *, timeout_seconds: float) -> ModelResponse: ...
