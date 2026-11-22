"""Choose the provider adapter for a model name, and the environment variable that holds its key.

The model name alone decides the vendor, so a configuration (which pins the model and is
hashed) fully determines the provider. Keys stay in the environment; this module only
reports whether one is set, never its value.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

from app.providers.anthropic_http import AnthropicMessagesProvider
from app.providers.base import ModelProvider
from app.providers.openai_http import OpenAIChatProvider

API_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
WIRE_API = {"anthropic": "anthropic_messages", "openai": "openai_chat_completions"}
# Anthropic configurations pin effort; OpenAI non-reasoning models reject reasoning_effort.
DEFAULT_EFFORT: dict[str, str | None] = {"anthropic": "medium", "openai": None}

_OPENAI_MODEL = re.compile(r"^(gpt-|chatgpt-|o\d)")


def provider_for(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if _OPENAI_MODEL.match(model):
        return "openai"
    raise ValueError(f"no provider adapter for model {model!r}")


def resolve_effort(model: str, effort: str | None) -> str | None:
    return effort if effort is not None else DEFAULT_EFFORT[provider_for(model)]


def missing_api_keys(models: Iterable[str]) -> list[str]:
    """Environment variables, in sorted order, that the given models need and that are unset or empty."""
    needed = {API_KEY_ENV[provider_for(m)] for m in models}
    return sorted(name for name in needed if not os.environ.get(name))


def make_provider(model: str, *, effort: str | None = None) -> ModelProvider:
    provider, effort = provider_for(model), resolve_effort(model, effort)
    if provider == "anthropic":
        return AnthropicMessagesProvider(model=model, effort=effort or DEFAULT_EFFORT["anthropic"])
    return OpenAIChatProvider(model=model, reasoning_effort=effort)
