import json

import pytest

from app.environment import SupportEnvironment
from app.tools import TOOLS, CustomerContext, ToolGateway
from app.trace import TraceRecorder

ALL_TOOLS = tuple(TOOLS)


@pytest.fixture
def make(evaluator_view):
    def build(tools=ALL_TOOLS, ticket_id="T-5001"):
        env = SupportEnvironment(evaluator_view.fixture)
        trace = TraceRecorder()
        return env, trace, ToolGateway(env, CustomerContext("CUST-001", ticket_id), trace, tools)

    return build


def _last_decision(trace):
    return next(e for e in reversed(trace.events) if e.kind == "tool_decision").payload


def _approve(gw, order_id="ORD-1001", amount="149.00"):
    return gw.call("request_refund_approval", {"order_id": order_id, "amount": amount, "reason": "returned"})


def _refund(gw, approval_id, order_id="ORD-1001", amount="149.00"):
    return gw.call("issue_refund", {"order_id": order_id, "amount": amount, "approval_id": approval_id})


def _denial(gw, trace, name, args):
    outcome = gw.call(name, args)
    assert outcome["ok"] is False
    return outcome["error"]["code"], _last_decision(trace)["reason_code"]


# --- three separate facts per call -------------------------------------------


def test_allowed_mutation_records_attempt_decision_and_effect(make):
    _, trace, gw = make()
    approval = _approve(gw)
    raw_args = {"order_id": "ORD-1001", "amount": "149.00", "approval_id": approval["result"]["approval_id"]}

    refund = gw.call("issue_refund", raw_args, call_id="call_9")

    assert refund["ok"] and refund["result"]["amount"] == "149.00"
    attempt, decision, effect = trace.events[-3:]
    assert (attempt.kind, decision.kind, effect.kind) == ("tool_attempt", "tool_decision", "tool_effect")
    assert attempt.payload == {"call_id": "call_9", "args": raw_args}
    assert decision.payload == {"attempt_sequence": attempt.sequence, "decision": "allowed", "reason_code": "ALLOWED"}
    assert effect.payload["attempt_sequence"] == attempt.sequence
    assert effect.payload["status"] == "applied"
    assert effect.payload["state_changes"] == {
        "approvals": {"modified": ["APR-0001"]},
        "refunds": {"added": ["RF-0001"]},
    }


def test_denied_call_has_no_effect_and_changes_nothing(make):
    env, trace, gw = make()
    before = env.snapshot()
    outcome = _refund(gw, "APR-0001")
    assert outcome == {"ok": False, "error": {"code": "approval_not_found", "message": "no approval for this order"}}
    assert [e.kind for e in trace.events] == ["tool_attempt", "tool_decision"]
    assert _last_decision(trace)["decision"] == "denied"
    assert env.snapshot() == before


def test_read_tool_has_no_state_change(make):
    _, trace, gw = make()
    outcome = gw.call("get_order", {"order_id": "ORD-1001"})
    assert outcome["result"]["total"] == "149.00"
    assert "customer_id" not in outcome["result"]
    effect = trace.events[-1]
    assert (effect.payload["status"], effect.payload["state_changes"]) == ("no_change", {})


# --- identity and authorization ----------------------------------------------


def test_cross_customer_access_is_denied_and_looks_like_a_missing_order(make):
    _, trace, gw = make()
    assert _denial(gw, trace, "get_order", {"order_id": "ORD-2002"}) == ("order_not_found", "CROSS_CUSTOMER_ACCESS")
    assert _denial(gw, trace, "get_order", {"order_id": "ORD-9999"}) == ("order_not_found", "RESOURCE_NOT_FOUND")


def test_identity_cannot_be_supplied_as_an_argument(make):
    _, trace, gw = make()
    args = {"order_id": "ORD-2002", "customer_id": "CUST-002"}
    assert _denial(gw, trace, "get_order", args) == ("invalid_arguments", "INVALID_ARGUMENTS")


def test_ticket_updates_only_touch_the_session_ticket(make):
    env, trace, gw = make()
    outcome = gw.call("update_ticket", {"status": "resolved", "note": "Refund issued."})
    assert outcome["result"]["status"] == "resolved"
    assert env.ticket("T-6001")["status"] == "open"
    assert trace.events[-1].payload["state_changes"] == {"tickets": {"modified": ["T-5001"]}}


@pytest.mark.parametrize("ticket_id", [None, "T-6001"])  # no ticket / another customer's ticket
def test_ticket_update_without_an_owned_session_ticket_is_denied(make, ticket_id):
    _, trace, gw = make(ticket_id=ticket_id)
    args = {"status": "resolved", "note": "x"}
    assert _denial(gw, trace, "update_ticket", args) == ("ticket_not_found", "NO_SESSION_TICKET")


# --- strict schemas and allowlist ----------------------------------------------


@pytest.mark.parametrize("raw_args", [None, "ORD-1001", ["ORD-1001"], {}, {"order_id": "1001"}, {"order_id": 1001}])
def test_malformed_arguments_are_rejected_and_recorded_verbatim(make, raw_args):
    _, trace, gw = make()
    assert gw.call("get_order", raw_args)["error"]["code"] == "invalid_arguments"
    assert trace.events[0].payload["args"] == raw_args


def test_unknown_and_disallowed_tools_are_denied(make):
    _, trace, gw = make(tools=("get_order",))
    assert _denial(gw, trace, "delete_order", {}) == ("unknown_tool", "UNKNOWN_TOOL")
    assert _denial(gw, trace, "issue_refund", {}) == ("tool_not_allowed", "TOOL_NOT_ALLOWED")
    assert [spec.name for spec in gw.tool_specs] == ["get_order"]


def test_tool_specs_are_strict_json_schemas(make):
    _, _, gw = make()
    assert [spec.name for spec in gw.tool_specs] == list(ALL_TOOLS)
    for spec in gw.tool_specs:
        assert spec.input_schema["additionalProperties"] is False
        assert spec.input_schema["type"] == "object"


# --- preconditions -------------------------------------------------------------


def test_refund_approval_requires_a_received_return(make):
    _, trace, gw = make()
    args = {"order_id": "ORD-1003", "amount": "59.00", "reason": "r"}
    assert _denial(gw, trace, "request_refund_approval", args) == (
        "return_not_received",
        "PRECONDITION_RETURN_NOT_RECEIVED",
    )


def test_approval_cannot_exceed_refundable_balance(make):
    _, trace, gw = make()
    args = {"order_id": "ORD-1001", "amount": "150.00", "reason": "r"}
    assert _denial(gw, trace, "request_refund_approval", args) == (
        "amount_exceeds_refundable",
        "PRECONDITION_AMOUNT_EXCEEDS_REFUNDABLE",
    )


def test_expired_approval_blocks_refund(make):
    env, trace, gw = make()
    approval = _approve(gw, "ORD-1002", "19.00")
    assert approval["result"]["status"] == "expired"
    args = {"order_id": "ORD-1002", "amount": "19.00", "approval_id": approval["result"]["approval_id"]}
    assert _denial(gw, trace, "issue_refund", args) == ("approval_not_granted", "PRECONDITION_APPROVAL_NOT_GRANTED")
    assert env.snapshot()["refunds"] == {}


def test_approval_is_single_use(make):
    _, trace, gw = make()
    approval_id = _approve(gw)["result"]["approval_id"]
    assert _refund(gw, approval_id)["ok"]
    args = {"order_id": "ORD-1001", "amount": "149.00", "approval_id": approval_id}
    assert _denial(gw, trace, "issue_refund", args) == ("approval_already_used", "PRECONDITION_APPROVAL_ALREADY_USED")


def test_refund_cannot_exceed_the_approved_amount(make):
    _, trace, gw = make()
    approval_id = _approve(gw, amount="100.00")["result"]["approval_id"]
    args = {"order_id": "ORD-1001", "amount": "149.00", "approval_id": approval_id}
    assert _denial(gw, trace, "issue_refund", args) == (
        "amount_exceeds_approval",
        "PRECONDITION_AMOUNT_EXCEEDS_APPROVAL",
    )


def test_approval_for_one_order_cannot_refund_another(make):
    _, trace, gw = make()
    approval_id = _approve(gw)["result"]["approval_id"]
    args = {"order_id": "ORD-1002", "amount": "19.00", "approval_id": approval_id}
    assert _denial(gw, trace, "issue_refund", args) == ("approval_not_found", "PRECONDITION_APPROVAL_MISSING")


def test_partial_refunds_reduce_the_refundable_balance(make):
    _, trace, gw = make()
    assert _refund(gw, _approve(gw, amount="100.00")["result"]["approval_id"], amount="100.00")["ok"]
    status = gw.call("get_refund_status", {"order_id": "ORD-1001"})["result"]
    assert status["refundable_balance"] == "49.00"
    assert _approve(gw, amount="49.00")["ok"]
    args = {"order_id": "ORD-1001", "amount": "50.00", "reason": "r"}
    assert _denial(gw, trace, "request_refund_approval", args)[0] == "amount_exceeds_refundable"


def test_every_tool_returns_json_serializable_results(make):
    _, _, gw = make()
    approval_id = _approve(gw)["result"]["approval_id"]
    outcomes = [
        gw.call("get_order", {"order_id": "ORD-1001"}),
        gw.call("get_return_status", {"order_id": "ORD-1001"}),
        gw.call("search_policy", {"query": "refund approval"}),
        _refund(gw, approval_id),
        gw.call("get_refund_status", {"order_id": "ORD-1001"}),
        gw.call("update_ticket", {"status": "resolved", "note": "done"}),
    ]
    assert all(o["ok"] for o in outcomes)
    json.dumps(outcomes)
