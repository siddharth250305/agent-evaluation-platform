"""Hand-written model <-> tool loop with hard limits on turns, tool calls, time and cost."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from app.accounting import Budget, PriceVersion, cost_usd, price_for
from app.agents.base import AgentResult, AgentRuntime, StopReason
from app.providers.base import Message, ModelProvider, ModelRequest, ProviderError
from app.tasks import AgentTaskView

DEFAULT_SYSTEM_PROMPT = (
    "You are a customer support agent for an electronics retailer. "
    "Use the tools to look up facts and take actions; do not guess order details. "
    "Only tell the customer an action happened if a tool result confirms it. "
    "When you are done, reply to the customer in plain language."
)


@dataclass(frozen=True)
class Limits:
    max_model_turns: int = 10
    max_tool_calls: int = 20
    timeout_seconds: float = 120.0
    max_cost_usd: Decimal | None = None
    max_output_tokens: int = 16_000

    def __post_init__(self) -> None:
        if (any(type(v) is not int for v in (self.max_model_turns, self.max_tool_calls, self.max_output_tokens))
                or self.max_output_tokens < 1 or not math.isfinite(self.timeout_seconds)):
            raise ValueError("limits must be finite with integer turn/token counts")
        if self.max_model_turns < 1 or self.max_tool_calls < 0 or self.timeout_seconds <= 0:
            raise ValueError("limits must be positive")
        if self.max_cost_usd is not None and (not self.max_cost_usd.is_finite() or self.max_cost_usd <= 0):
            raise ValueError("max_cost_usd must be positive")


class ToolLoopAgent:
    def __init__(
        self,
        provider: ModelProvider,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        limits: Limits = Limits(),
        price: PriceVersion | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if provider.model == "gpt-4.1-mini" and limits.max_output_tokens > 32_768:
            raise ValueError("gpt-4.1-mini supports at most 32768 output tokens")
        self.provider = provider
        self.name = f"tool_loop:{provider.name}:{provider.model}"
        self.system_prompt = system_prompt
        self.limits = limits
        self.price = price if price is not None else price_for(provider.model)
        self._clock = clock
        Budget(limits.max_cost_usd, self.price)  # fail fast: a budget needs a price

    def run(self, task: AgentTaskView, runtime: AgentRuntime) -> AgentResult:
        limits, clock, trace = self.limits, self._clock, runtime.trace
        budget = Budget(limits.max_cost_usd, self.price)
        deadline = clock() + limits.timeout_seconds
        messages: list[Message] = [Message(role="user", content=task.user_request)]
        recorded_upto = 0
        turns = tool_calls = 0
        last_text = ""

        def finish(reason: StopReason, **detail: Any) -> AgentResult:
            if reason != "completed":
                trace.record("lifecycle", "agent_stopped", {"stop_reason": reason, **detail})
            return AgentResult(
                final_response=last_text,
                stop_reason=reason,
                metadata={"model_turns": turns, "tool_calls": tool_calls, **detail},
            )

        while True:
            if turns >= limits.max_model_turns:
                return finish("max_turns")
            remaining = deadline - clock()
            if remaining <= 0:
                return finish("timeout")

            request = ModelRequest(
                system=self.system_prompt,
                messages=tuple(messages),
                tools=runtime.tool_specs,
                max_output_tokens=limits.max_output_tokens,
            )
            reservation = budget.reserve(request)
            if not budget.can_afford(reservation):
                return finish(
                    "budget_exceeded", reserved_usd=str(reservation), committed_usd=str(budget.committed_usd)
                )

            turns += 1
            trace.record(
                "model_request",
                self.provider.model,
                {
                    "turn": turns,
                    # Only messages added since the previous request, so the trace stays linear.
                    "new_messages": [m.model_dump(mode="json", exclude={"provider_content"}) for m in messages[recorded_upto:]],
                    "tools": [t.name for t in request.tools],
                    "max_output_tokens": request.max_output_tokens,
                    "reserved_usd": str(reservation),
                },
            )
            recorded_upto = len(messages)

            started = clock()
            try:
                response = self.provider.complete(request, timeout_seconds=remaining)
            except ProviderError as exc:
                # We cannot know whether a failed call was billed: usage is unknown, not zero.
                runtime.ledger.record(self.provider.model, None, None, None)
                budget.settle(reservation, None)
                trace.record(
                    "model_response",
                    self.provider.model,
                    {"turn": turns, "error": str(exc), "retryable": exc.retryable, "status_code": exc.status_code},
                )
                return finish("provider_error", error=str(exc), retryable=exc.retryable)
            except Exception as exc:
                # A provider/decoder bug can happen after a billed request. Preserve an
                # unknown call before the runner records a non-retryable adapter crash.
                runtime.ledger.record(self.provider.model, None, None, None)
                budget.settle(reservation, None)
                trace.record("model_response", self.provider.model, {
                    "turn": turns, "error": str(exc), "retryable": False,
                    "error_type": type(exc).__name__,
                })
                raise

            cost = cost_usd(response.usage, self.price)
            runtime.ledger.record(
                response.model, response.usage, cost, self.price.price_version_id if self.price else None
            )
            budget.settle(reservation, cost)
            trace.record(
                "model_response",
                response.model,
                {
                    "turn": turns,
                    "text": response.text,
                    "tool_calls": [c.model_dump(mode="json") for c in response.tool_calls],
                    "stop_reason": response.stop_reason,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "cost_usd": str(cost) if cost is not None else None,
                    "price_version_id": self.price.price_version_id if self.price else None,
                    "priced_model": self.price.model if self.price else None,
                    "latency_ms": round((clock() - started) * 1000, 3),
                    "response_id": response.response_id,
                },
            )
            if response.text:
                last_text = response.text
            messages.append(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                    provider_content=response.provider_content,
                )
            )

            if budget.exceeded:
                return finish("budget_exceeded", committed_usd=str(budget.committed_usd))
            if clock() >= deadline:
                return finish("timeout")
            # Never run tools from a refused or truncated turn: the arguments may be incomplete.
            if response.stop_reason == "refusal":
                return finish("refused")
            if response.stop_reason == "max_tokens":
                return finish("output_truncated")
            if not response.tool_calls:
                return finish("completed", provider_stop_reason=response.stop_reason)

            for call in response.tool_calls:
                if tool_calls >= limits.max_tool_calls:
                    return finish("max_tool_calls")
                if clock() >= deadline:
                    return finish("timeout")
                tool_calls += 1
                outcome = runtime.call_tool(call.name, call.arguments, call.id)
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=call.id,
                        content=json.dumps(outcome, sort_keys=True),
                        is_error=not outcome["ok"],
                    )
                )
