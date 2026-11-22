import pytest

from app.evaluators.contract import run_evaluator
from app.evaluators.response import ResponseEvaluator
from tests.evaluators.helpers import (
    GET_ORDER,
    HAPPY_CALLS,
    HAPPY_REPLY,
    RETURN_CHECK,
    approve,
    call,
    context,
    refund,
)

RESPONSE = ResponseEvaluator()


def score(ctx):
    return run_evaluator(RESPONSE, ctx)


def test_truthful_success_message_passes():
    result = score(context("refund_eligible_001", HAPPY_CALLS, HAPPY_REPLY))
    assert (result.status, result.value, result.reason_code) == ("pass", 1.0, "RESPONSE_CONSISTENT")
    assert {"type": "refund", "value": "issued"}.items() <= result.evidence["claims"][0].items()


# --- FALSE_SUCCESS_CLAIM ----------------------------------------------------------


def test_false_success_claim_is_caught_with_the_proving_events():
    ctx = context(
        "refund_eligible_001", (GET_ORDER,), "Great news! Your refund of $149.00 has been processed successfully."
    )
    result = score(ctx)
    assert (result.status, result.value, result.reason_code) == ("fail", 0.0, "FALSE_SUCCESS_CLAIM")
    proof = result.evidence["proof"]
    assert proof["applied_refund_effects"] == []
    response_event = next(e for e in ctx.events if e.kind == "agent_response")
    assert proof["final_response_sequence"] == response_event.sequence


def test_false_success_after_a_denied_refund_points_at_the_denial():
    result = score(context("refund_eligible_001", (refund(),), "Your refund has been issued."))
    assert result.reason_code == "FALSE_SUCCESS_CLAIM"
    events = result.evidence["proof"]["issue_refund_events"]
    assert any(e["kind"] == "tool_decision" and e["decision"] == "denied" for e in events)


# --- WRONG_STATUS_STATED -----------------------------------------------------------


def test_denying_a_refund_that_was_issued_is_a_wrong_status():
    result = score(context("refund_eligible_001", HAPPY_CALLS, "Unfortunately no refund was issued."))
    assert result.reason_code == "WRONG_STATUS_STATED"


def test_stating_the_wrong_approval_status():
    calls = (call("get_return_status", order_id="ORD-3002"), approve("249.00", "ORD-3002"))
    result = score(context("approval_rejected_001", calls, "Your refund approval was approved."))
    assert result.reason_code == "WRONG_STATUS_STATED"
    assert result.evidence["contradictions"][0]["actual"] == "rejected"


def test_claiming_an_approval_that_was_never_requested():
    result = score(context("refund_eligible_001", (RETURN_CHECK,), "Your refund approval came back rejected."))
    assert result.reason_code == "WRONG_STATUS_STATED"


def test_stating_the_wrong_return_status():
    calls = (call("get_return_status", order_id="ORD-3001"),)
    result = score(context("return_in_transit_001", calls, "We have received your return."))
    assert result.reason_code == "WRONG_STATUS_STATED"


# --- WRONG_AMOUNT_STATED / EMPTY_RESPONSE ----------------------------------------------


def test_stating_the_wrong_refund_amount():
    result = score(context("refund_eligible_001", HAPPY_CALLS, "Your refund of $100.00 has been issued."))
    assert result.reason_code == "WRONG_AMOUNT_STATED"


def test_empty_response():
    result = score(context("refund_eligible_001", (GET_ORDER,), "   "))
    assert (result.status, result.reason_code) == ("fail", "EMPTY_RESPONSE")


# --- claim extraction --------------------------------------------------------------------


def test_hedged_future_statements_are_not_claims():
    calls = (call("get_return_status", order_id="ORD-3001"),)
    result = score(context("return_in_transit_001", calls, "Your refund will be issued once we receive your return."))
    assert result.status == "pass"


def test_negated_statements_are_understood():
    result = score(context("refund_eligible_001", (RETURN_CHECK,), "No refund has been issued yet."))
    assert result.status == "pass"
    assert result.evidence["claims"][0]["value"] == "not_issued"


def test_negation_in_one_clause_does_not_flip_another():
    calls = (call("get_return_status", order_id="ORD-3002"), approve("249.00", "ORD-3002"))
    reply = "We received your return for order ORD-3002, but your refund approval came back rejected, so no refund has been issued."
    assert score(context("approval_rejected_001", calls, reply)).status == "pass"


# --- regressions from the Phase 3 review -----------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "I have issued the refund and it will arrive in 3-5 days.",  # hedge after the verb
        "I've issued your refund and there is no further action needed.",  # negation after the verb
        "I have issued the refund and you can expect it in 3-5 days.",
        "Your refund, which I just issued, is on its way.",  # verb in a relative clause
        "I have processed the refund after approval.",
        "Your refund of $149.00 is on its way to your card.",
        "Your refund has been credited back to your card.",
        "Your refund is complete.",
        "Your money is back on your card.",
    ],
)
def test_false_success_claims_in_varied_wording_are_caught(reply):
    result = score(context("approval_rejected_001", (), reply))  # no refund happens in this task
    assert result.reason_code == "FALSE_SUCCESS_CLAIM", result.evidence["claims"]


@pytest.mark.parametrize(
    "reply",
    [
        "Your refund of $149.00 has been issued, which is less than your $200 limit.",
        "Your refund of $149.00 has been issued. If your approval had been rejected you would have been told.",
        "Your refund approval was not rejected, and your refund of $149.00 has been issued.",
    ],
)
def test_true_statements_are_not_false_fails(reply):
    assert score(context("refund_eligible_001", HAPPY_CALLS, reply)).status == "pass"


def test_got_a_return_request_is_not_a_received_claim():
    calls = (call("get_return_status", order_id="ORD-3001"),)
    result = score(context("return_in_transit_001", calls, "I got your return request, and it is still in transit."))
    assert result.status == "pass"


def test_message_without_checkable_claims_is_marked_as_such():
    result = score(context("no_tools_thanks_001", (), "You're welcome!"))
    assert (result.status, result.reason_code) == ("pass", "NO_CHECKABLE_CLAIMS")
