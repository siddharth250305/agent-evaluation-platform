"""Deterministic scripted provider for offline tests. Never touches the network."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from typing import Any

from app.providers.base import ModelRequest, ModelResponse, ProviderError, ToolCall, Usage

Step = ModelResponse | ProviderError | Callable[[ModelRequest], ModelResponse]

_call_ids = itertools.count(1)


def known_usage(input_tokens: int = 100, output_tokens: int = 20) -> Usage:
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def tool_calls(*calls: tuple[str, Any], text: str = "", usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=tuple(ToolCall(id=f"call_{next(_call_ids)}", name=name, arguments=args) for name, args in calls),
        stop_reason="tool_use",
        usage=usage if usage is not None else known_usage(),
        model="scripted-model",
    )


def final_text(text: str, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        text=text,
        stop_reason="end_turn",
        usage=usage if usage is not None else known_usage(),
        model="scripted-model",
    )


class ScriptedProvider:
    """Replays ``steps`` in order and records every request it receives."""

    name = "scripted"

    def __init__(
        self,
        steps: Iterable[Step],
        *,
        model: str = "scripted-model",
        on_call: Callable[[], None] | None = None,
    ) -> None:
        self.model = model
        self.requests: list[ModelRequest] = []
        self._steps = list(steps)
        self._on_call = on_call

    def complete(self, request: ModelRequest, *, timeout_seconds: float) -> ModelResponse:
        self.requests.append(request)
        if self._on_call is not None:
            self._on_call()
        if not self._steps:
            raise ProviderError("scripted provider has no steps left", retryable=False)
        step = self._steps.pop(0)
        if isinstance(step, ProviderError):
            raise step
        if isinstance(step, ModelResponse):
            return step
        return step(request)
