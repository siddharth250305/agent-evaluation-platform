import copy
from decimal import Decimal

import pytest

from app.environment import ApprovalSimulator, IntegrityError, SupportEnvironment, diff_states


def _approve_and_refund(env, order_id="ORD-1001", amount="149.00"):
    approval = env.create_approval(order_id, Decimal(amount), "test")
    return env.create_refund(order_id, Decimal(amount), approval["approval_id"])


def test_each_environment_gets_a_unique_id(evaluator_view):
    ids = {SupportEnvironment(evaluator_view.fixture).environment_id for _ in range(5)}
    assert len(ids) == 5


def test_environments_are_isolated_and_fixture_is_never_mutated(evaluator_view):
    pristine = copy.deepcopy(evaluator_view.fixture)
    a = SupportEnvironment(evaluator_view.fixture)
    b = SupportEnvironment(evaluator_view.fixture)

    _approve_and_refund(a)
    a.update_ticket("T-5001", "resolved", "done")

    assert len(a.snapshot()["refunds"]) == 1
    assert b.snapshot()["refunds"] == {}
    assert b.ticket("T-5001")["status"] == "open"
    assert evaluator_view.fixture == pristine


def test_initial_snapshot_is_preserved_after_changes(evaluator_view):
    env = SupportEnvironment(evaluator_view.fixture)
    _approve_and_refund(env)
    assert env.initial_snapshot["refunds"] == {}
    assert env.snapshot()["refunds"]["RF-0001"]["amount"] == Decimal("149.00")


def test_reads_and_snapshots_are_copies(evaluator_view):
    env = SupportEnvironment(evaluator_view.fixture)
    env.order("ORD-1001")["total"] = Decimal("1.00")
    env.snapshot()["orders"]["ORD-1001"]["total"] = Decimal("1.00")
    assert env.order("ORD-1001")["total"] == Decimal("149.00")


def test_session_comes_from_the_fixture(evaluator_view):
    env = SupportEnvironment(evaluator_view.fixture)
    assert (env.session.customer_id, env.session.ticket_id) == ("CUST-001", "T-5001")


def test_approval_simulator_is_deterministic():
    sim = ApprovalSimulator({"ORD-1": "approved", "ORD-2": "expired"})
    assert [sim.decide("ORD-1") for _ in range(3)] == ["approved"] * 3
    assert sim.decide("ORD-2") == "expired"
    assert sim.decide("ORD-unknown") == "rejected"


@pytest.mark.parametrize(
    ("order_id", "status"),
    [("ORD-1001", "approved"), ("ORD-1002", "expired"), ("ORD-1003", "rejected")],
)
def test_approval_decisions_come_from_the_fixture(evaluator_view, order_id, status):
    for _ in range(2):  # same answer in every fresh environment
        env = SupportEnvironment(evaluator_view.fixture)
        assert env.create_approval(order_id, Decimal("1.00"), "r")["status"] == status


def test_integrity_guard_blocks_refund_without_usable_approval(evaluator_view):
    env = SupportEnvironment(evaluator_view.fixture)
    expired = env.create_approval("ORD-1002", Decimal("19.00"), "r")
    with pytest.raises(IntegrityError):
        env.create_refund("ORD-1002", Decimal("19.00"), expired["approval_id"])
    assert env.snapshot()["refunds"] == {}


def test_integrity_guard_blocks_over_refund(evaluator_view):
    env = SupportEnvironment(evaluator_view.fixture)
    _approve_and_refund(env)
    extra = env.create_approval("ORD-1001", Decimal("10.00"), "r")
    with pytest.raises(IntegrityError):
        env.create_refund("ORD-1001", Decimal("10.00"), extra["approval_id"])


def test_new_refund_ids_never_collide_with_existing_ones(evaluator_view):
    fixture = copy.deepcopy(evaluator_view.fixture)
    fixture["refunds"] = {
        "RF-0001": {"refund_id": "RF-0001", "order_id": "ORD-1002", "amount": "5.00", "status": "issued"}
    }
    env = SupportEnvironment(fixture)
    assert _approve_and_refund(env)["refund_id"] == "RF-0002"


def test_diff_states_reports_changed_ids(evaluator_view):
    env = SupportEnvironment(evaluator_view.fixture)
    before = env.snapshot()
    _approve_and_refund(env)
    env.update_ticket("T-5001", "resolved", "done")
    assert diff_states(before, env.snapshot()) == {
        "approvals": {"added": ["APR-0001"]},
        "refunds": {"added": ["RF-0001"]},
        "tickets": {"modified": ["T-5001"]},
    }
    assert diff_states(env.snapshot(), env.snapshot()) == {}


def test_policy_search_is_a_deterministic_keyword_match(evaluator_view):
    env = SupportEnvironment(evaluator_view.fixture)
    results = [p["policy_id"] for p in env.search_policies("refund approval")]
    assert results[0] == "POL-APPROVAL-01"
    assert results == [p["policy_id"] for p in env.search_policies("refund approval")]
    assert env.search_policies("zebra") == []


def test_nested_reads_and_effect_results_do_not_alias_state(evaluator_view):
    env = SupportEnvironment(evaluator_view.fixture)
    env.order("ORD-1001")["items"][0]["name"] = "changed"
    env.update_ticket("T-5001", "open", "original")["notes"].clear()
    approval = env.create_approval("ORD-1001", Decimal("149"), "test")
    approval["status"] = "rejected"
    refund = env.create_refund("ORD-1001", Decimal("149"), approval["approval_id"])
    refund["amount"] = Decimal("0")
    assert env.order("ORD-1001")["items"][0]["name"] == "Aurora Wireless Headphones"
    assert env.ticket("T-5001")["notes"] == ["original"]
    assert env.refunds_for("ORD-1001")[0]["amount"] == Decimal("149")


def test_approval_decisions_are_independent_of_call_order_and_source_mutation():
    decisions = {"A": "approved", "B": "expired", "C": "rejected"}
    sim = ApprovalSimulator(decisions)
    decisions["A"] = "rejected"
    assert [sim.decide(key) for key in ("C", "A", "B", "A")] == [
        "rejected", "approved", "expired", "approved"
    ]


def test_faults_are_operation_scoped_copied_and_repeatable(evaluator_view):
    fixture = copy.deepcopy(evaluator_view.fixture)
    fixture["faults"] = {"issue_refund:ORD-1001": "timeout"}
    first = SupportEnvironment(fixture)
    second = SupportEnvironment(fixture)
    fixture["faults"].clear()
    for env in (first, second):
        assert env.operation_fault("get_order", "ORD-1001") is None
        assert env.operation_fault("issue_refund", "ORD-1002") is None
        assert [env.operation_fault("issue_refund", "ORD-1001") for _ in range(3)] == ["timeout"] * 3
    assert first.initial_snapshot == second.initial_snapshot
    assert first.environment_id != second.environment_id
