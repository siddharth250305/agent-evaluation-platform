"""Usage, cost and budget accounting. Unknown usage stays unknown; it is never counted as free."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from app.providers.base import ModelRequest, Usage

_PER_MTOK = Decimal(1_000_000)


@dataclass(frozen=True)
class PriceVersion:
    price_version_id: str
    effective_date: date
    model: str
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_read_per_mtok: Decimal | None  # None = rate unknown
    cache_write_per_mtok: Decimal | None


# Anthropic first-party list prices (USD per million tokens) as of 2026-10-06.
# Cache writes use the 5-minute TTL rate, 1.25x the input rate.
PRICE_VERSIONS: tuple[PriceVersion, ...] = (
    PriceVersion(
        "anthropic-2026-10-06", date(2026, 10, 6), "claude-opus-5-5",
        Decimal("4.00"), Decimal("20.00"), Decimal("0.20"), Decimal("5.00"),
    ),
    PriceVersion(
        "anthropic-2026-10-06", date(2026, 10, 6), "claude-sonnet-5-5",
        Decimal("2.00"), Decimal("10.00"), Decimal("0.20"), Decimal("2.50"),
    ),
)


def price_for(model: str, on: date | None = None) -> PriceVersion | None:
    on = on or date.today()
    candidates = [p for p in PRICE_VERSIONS if p.model == model and p.effective_date <= on]
    return max(candidates, key=lambda p: p.effective_date, default=None)


def cost_usd(usage: Usage | None, price: PriceVersion | None) -> Decimal | None:
    """Exact cost, or None if any token count or any needed rate is unknown."""
    if usage is None or price is None:
        return None
    parts = (
        (usage.input_tokens, price.input_per_mtok),
        (usage.output_tokens, price.output_per_mtok),
        (usage.cache_read_input_tokens, price.cache_read_per_mtok),
        (usage.cache_creation_input_tokens, price.cache_write_per_mtok),
    )
    total = Decimal("0")
    for tokens, rate in parts:
        if tokens is None:
            return None
        if tokens == 0:
            continue
        if rate is None:
            return None
        total += Decimal(tokens) * rate / _PER_MTOK
    return total


class UsageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_calls: int
    unknown_usage_calls: int
    input_tokens: int | None
    output_tokens: int | None
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None
    cost_usd: Decimal | None  # None if any call's cost is unknown
    price_version_ids: tuple[str, ...]


@dataclass(frozen=True)
class _CallRecord:
    model: str
    usage: Usage | None
    cost_usd: Decimal | None
    price_version_id: str | None


class UsageLedger:
    """Every model call in an attempt, including failed ones (whose usage is unknown)."""

    def __init__(self) -> None:
        self._calls: list[_CallRecord] = []

    def record(self, model: str, usage: Usage | None, cost: Decimal | None, price_version_id: str | None) -> None:
        self._calls.append(_CallRecord(model, usage, cost, price_version_id))

    def summary(self) -> UsageSummary:
        def total(field: str) -> int | None:
            values = [getattr(c.usage, field) if c.usage else None for c in self._calls]
            return None if any(v is None for v in values) else sum(values)

        costs = [c.cost_usd for c in self._calls]
        return UsageSummary(
            model_calls=len(self._calls),
            unknown_usage_calls=sum(1 for c in costs if c is None),
            input_tokens=total("input_tokens"),
            output_tokens=total("output_tokens"),
            cache_read_input_tokens=total("cache_read_input_tokens"),
            cache_creation_input_tokens=total("cache_creation_input_tokens"),
            cost_usd=None if any(c is None for c in costs) else sum(costs, start=Decimal("0")),
            price_version_ids=tuple(sorted({c.price_version_id for c in self._calls if c.price_version_id})),
        )


class Budget:
    """Reserve an upper-bound estimate before each call, then settle with the actual cost.

    When a call's actual cost is unknown, the reservation stays committed, so
    unknown usage can only make the budget stricter, never looser.
    """

    def __init__(self, max_cost_usd: Decimal | None, price: PriceVersion | None) -> None:
        if max_cost_usd is not None and price is None:
            raise ValueError("cannot enforce a cost budget without a price version for the model")
        self.max_cost_usd = max_cost_usd
        self._price = price
        self.committed_usd = Decimal("0")

    def reserve(self, request: ModelRequest) -> Decimal:
        if self.max_cost_usd is None or self._price is None:
            return Decimal("0")
        # Rough upper bound: ~3 chars per token, priced at the dearest input rate.
        estimated_input = math.ceil(len(request.model_dump_json()) / 3)
        input_rate = max(self._price.input_per_mtok, self._price.cache_write_per_mtok or Decimal("0"))
        return (
            Decimal(estimated_input) * input_rate
            + Decimal(request.max_output_tokens) * self._price.output_per_mtok
        ) / _PER_MTOK

    def can_afford(self, reservation: Decimal) -> bool:
        return self.max_cost_usd is None or self.committed_usd + reservation <= self.max_cost_usd

    def settle(self, reservation: Decimal, actual: Decimal | None) -> None:
        self.committed_usd += actual if actual is not None else reservation

    @property
    def exceeded(self) -> bool:
        return self.max_cost_usd is not None and self.committed_usd > self.max_cost_usd
