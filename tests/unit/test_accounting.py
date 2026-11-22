from datetime import date
from decimal import Decimal

import pytest

from app.accounting import Budget, PriceVersion, UsageLedger, cost_usd, price_for
from app.providers.base import Message, ModelRequest, Usage

PRICE = PriceVersion(
    "test-v1", date(2026, 1, 1), "m", Decimal("4"), Decimal("20"), Decimal("0.20"), Decimal("5")
)
NO_CACHE_RATES = PriceVersion("test-v2", date(2026, 1, 1), "m", Decimal("4"), Decimal("20"), None, None)
FULL = Usage(input_tokens=1_000_000, output_tokens=100_000, cache_read_input_tokens=0, cache_creation_input_tokens=0)
REQUEST = ModelRequest(system="s", messages=(Message(role="user", content="hi"),), tools=(), max_output_tokens=1000)


def test_cost_with_complete_usage():
    assert cost_usd(FULL, PRICE) == Decimal("6")


def test_cost_includes_cache_reads_and_writes():
    usage = Usage(input_tokens=0, output_tokens=0, cache_read_input_tokens=1_000_000, cache_creation_input_tokens=1_000_000)
    assert cost_usd(usage, PRICE) == Decimal("5.20")


@pytest.mark.parametrize(
    "usage",
    [
        None,
        Usage(),
        Usage(input_tokens=10, output_tokens=None, cache_read_input_tokens=0, cache_creation_input_tokens=0),
        Usage(input_tokens=10, output_tokens=10, cache_read_input_tokens=None, cache_creation_input_tokens=0),
    ],
)
def test_missing_usage_is_unknown_never_zero(usage):
    assert cost_usd(usage, PRICE) is None


def test_unknown_price_means_unknown_cost():
    assert cost_usd(FULL, None) is None


def test_missing_rate_only_matters_when_those_tokens_were_used():
    used_no_cache = Usage(input_tokens=10, output_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    used_cache = Usage(input_tokens=10, output_tokens=10, cache_read_input_tokens=5, cache_creation_input_tokens=0)
    assert cost_usd(used_no_cache, NO_CACHE_RATES) is not None
    assert cost_usd(used_cache, NO_CACHE_RATES) is None


def test_price_lookup_is_dated():
    assert price_for("claude-opus-5-5").input_per_mtok == Decimal("4.00")
    assert price_for("claude-sonnet-5-5").output_per_mtok == Decimal("10.00")
    assert price_for("claude-opus-5-5", on=date(2026, 1, 1)) is None  # before the price version took effect
    assert price_for("scripted-model") is None


def test_ledger_totals_when_every_call_is_known():
    ledger = UsageLedger()
    ledger.record("m", FULL, Decimal("6"), "test-v1")
    ledger.record("m", FULL, Decimal("6"), "test-v1")
    summary = ledger.summary()
    assert (summary.model_calls, summary.unknown_usage_calls) == (2, 0)
    assert summary.input_tokens == 2_000_000
    assert summary.cost_usd == Decimal("12")
    assert summary.price_version_ids == ("test-v1",)


def test_one_unknown_call_makes_totals_unknown_not_zero():
    ledger = UsageLedger()
    ledger.record("m", FULL, Decimal("6"), "test-v1")
    ledger.record("m", None, None, None)
    summary = ledger.summary()
    assert (summary.model_calls, summary.unknown_usage_calls) == (2, 1)
    assert summary.input_tokens is None
    assert summary.cost_usd is None


def test_no_model_calls_costs_exactly_zero():
    summary = UsageLedger().summary()
    assert (summary.model_calls, summary.cost_usd, summary.input_tokens) == (0, Decimal("0"), 0)


def test_budget_requires_a_price():
    with pytest.raises(ValueError):
        Budget(Decimal("1"), None)


def test_budget_reserves_then_settles():
    budget = Budget(Decimal("1"), PRICE)
    reservation = budget.reserve(REQUEST)
    assert reservation >= Decimal("1000") * PRICE.output_per_mtok / Decimal(1_000_000)
    assert budget.can_afford(reservation)
    budget.settle(reservation, Decimal("0.10"))
    assert budget.committed_usd == Decimal("0.10")


def test_unknown_actual_cost_keeps_the_reservation_committed():
    budget = Budget(Decimal("1"), PRICE)
    reservation = budget.reserve(REQUEST)
    budget.settle(reservation, None)
    assert budget.committed_usd == reservation


def test_no_budget_reserves_nothing():
    budget = Budget(None, None)
    assert budget.reserve(REQUEST) == 0
    assert budget.can_afford(Decimal("1000000"))
    assert not budget.exceeded
