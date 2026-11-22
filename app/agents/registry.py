"""Build an agent from a stored agent configuration.

An agent configuration names an adapter; the worker looks the adapter up here
and hands it the configuration. Secrets (API keys) never live in a
configuration: the provider reads them from the worker's environment.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from decimal import Decimal, InvalidOperation
from typing import Any

from app.agents.base import AgentAdapter
from app.agents.scripted import HonestScriptedAgent, LyingBrokenAgent
from app.agents.tool_loop import DEFAULT_SYSTEM_PROMPT, Limits, ToolLoopAgent
from app.providers.anthropic_http import AnthropicMessagesProvider

# Adapters that call a model and therefore need one named in the configuration.
MODEL_ADAPTERS = frozenset({"tool_loop"})


@dataclass(frozen=True)
class AgentConfigSpec:
    name: str
    adapter: str
    model: str | None = None
    prompt: str | None = None
    tools: tuple[str, ...] | None = None  # None: every tool the task offers
    limits: Mapping[str, Any] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)


AdapterFactory = Callable[[AgentConfigSpec], AgentAdapter]

_LIMIT_FIELDS = {f.name for f in fields(Limits)}


def limits_from(raw: Mapping[str, Any]) -> Limits:
    """Validated ``Limits``; ``max_cost_usd`` arrives as a decimal string so it round-trips through JSON."""
    unknown = set(raw) - _LIMIT_FIELDS
    if unknown:
        raise ValueError(f"unknown limits: {sorted(unknown)}")
    values = dict(raw)
    if values.get("max_cost_usd") is not None:
        try:
            values["max_cost_usd"] = Decimal(str(values["max_cost_usd"]))
        except InvalidOperation:
            raise ValueError("max_cost_usd must be a decimal number") from None
    try:
        return Limits(**values)
    except TypeError as exc:
        raise ValueError(str(exc)) from None


def _tool_loop(spec: AgentConfigSpec) -> AgentAdapter:
    if not spec.model:
        raise ValueError("the tool_loop adapter needs a model")
    provider = AnthropicMessagesProvider(model=spec.model, effort=spec.settings.get("effort", "medium"))
    return ToolLoopAgent(provider, system_prompt=spec.prompt or DEFAULT_SYSTEM_PROMPT, limits=limits_from(spec.limits))


DEFAULT_ADAPTERS: Mapping[str, AdapterFactory] = {
    "honest_scripted": lambda spec: HonestScriptedAgent(),
    "lying_broken": lambda spec: LyingBrokenAgent(),
    "tool_loop": _tool_loop,
}


def build_agent(spec: AgentConfigSpec, adapters: Mapping[str, AdapterFactory]) -> AgentAdapter:
    return adapters[spec.adapter](spec)
