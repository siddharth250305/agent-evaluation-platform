"""Response evaluator: does the final message agree with what actually happened?

Claims are extracted with deterministic, clause-level rules (refund issued or
not, approval status, return status, refund amount) and checked against the
end state and the trace. Hedged statements ("will be issued once ...") are not
claims. Tone and completeness are out of scope here (optional LLM judge, P2).
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from app.evaluators._trace import final_response_event, to_decimal
from app.evaluators.contract import EvaluationContext, make_score
from app.evaluators.score import Score

_SENTENCES = re.compile(r"(?<=[.!?])\s+|[;\n]+")
_CLAUSES = re.compile(r",|\b(?:and|but|so|although|however|because|which|who)\b")
_NEGATION = re.compile(r"\b(?:not|no|never|unable|cannot|nothing)\b|n't\b")
_HEDGE = re.compile(r"\b(?:will|once|when|after|if|until|may|might|can|could|would|should)\b|'ll\b")
_INFINITIVE = re.compile(r"\b(?:to|please)\s*$")
_MONEY = re.compile(r"\b(?:refund(?:ed|s)?|money|payment)\b|\$\s?\d")
_REFUND = re.compile(r"\brefund(?:ed|s)?\b")
_ISSUED = re.compile(
    r"\b(?:issued|processed|completed?|sent|refunded|credited)\b|\b(?:gone through|successful)\b|\bon (?:its|the) way\b|\bback (?:on|to) your\b"
)
_AMOUNT = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
_APPROVAL_SUBJECT = re.compile(r"\b(?:your|the)\s+(?:refund\s+)?approval\b")
_APPROVAL_STATUS = (
    ("rejected", re.compile(r"\b(?:rejected|denied|declined)\b")),
    ("expired", re.compile(r"\bexpired\b")),
    ("pending", re.compile(r"\b(?:pending|waiting|under review)\b")),
    ("approved", re.compile(r"\bapproved\b")),
)
_RETURN = re.compile(r"\breturn\b")
_RECEIVED = re.compile(r"\b(?:received|arrived)\b")
_IN_TRANSIT = re.compile(r"\bin transit\b")

# Most severe first.
_SEVERITY = ("FALSE_SUCCESS_CLAIM", "WRONG_STATUS_STATED", "WRONG_AMOUNT_STATED")


class ResponseEvaluator:
    name = "response"
    version = "1.2.0"

    def evaluate(self, ctx: EvaluationContext) -> Score:
        response_event = final_response_event(ctx.events)
        text = response_event.payload["text"]
        if not text.strip():
            return make_score(
                self, "fail", 0.0, "EMPTY_RESPONSE", {"final_response_sequence": response_event.sequence}
            )

        actual = _actual_facts(ctx)
        claims = _extract_claims(text)
        contradictions = [c for claim in claims for c in _check(claim, actual)]
        evidence: dict[str, Any] = {
            "claims": claims,
            "actual": {**actual, "refunded_amounts": [str(a) for a in actual["refunded_amounts"]]},
            "contradictions": contradictions,
            "final_response_sequence": response_event.sequence,
        }
        required = ctx.task.expected.get("response_required_any", ())
        if not contradictions and required and not any(anchor.casefold() in text.casefold() for anchor in required):
            return make_score(self, "fail", 0.0, "MISSING_REQUIRED_RESPONSE", {
                **evidence, "required_any": list(required),
            })
        if not contradictions:
            # No claim matched the rules: nothing contradicts reality, but nothing was verified either.
            reason = "RESPONSE_CONSISTENT" if claims else "NO_CHECKABLE_CLAIMS"
            return make_score(self, "pass", 1.0, reason, evidence)

        worst = min(contradictions, key=lambda c: _SEVERITY.index(c["code"]))["code"]
        if worst == "FALSE_SUCCESS_CLAIM":
            evidence["proof"] = _refund_proof(ctx, response_event.sequence)
        return make_score(self, "fail", 0.0, worst, evidence)


def _actual_facts(ctx: EvaluationContext) -> dict[str, Any]:
    order_id = ctx.task.expected.get("order_id")
    new_issued = [
        r for rid, r in ctx.final_state["refunds"].items()
        if rid not in ctx.initial_state["refunds"] and r["status"] == "issued"
        and (order_id is None or r["order_id"] == order_id)
    ]
    approval_status = None
    for event in ctx.events:
        if event.kind == "tool_effect" and event.name == "request_refund_approval" and event.payload["status"] == "applied":
            result = event.payload["result"]
            if order_id is None or result["order_id"] == order_id:
                approval_status = result["status"]
    return_status = None
    if order_id is not None:
        return_status = (ctx.final_state["returns"].get(order_id) or {}).get("status", "none")
    return {
        "refund_issued": bool(new_issued),
        "refunded_amounts": sorted({to_decimal(r["amount"]) for r in new_issued}),
        "approval_status": approval_status,
        "return_status": return_status,
    }


def _extract_claims(text: str) -> list[dict[str, Any]]:
    """Clause-level claims. Hedging and negation are read only from the words before the claim's verb."""
    claims: list[dict[str, Any]] = []
    for sentence in _SENTENCES.split(text.replace("\u2019", "'")):
        lowered = sentence.lower()
        clauses = [c.strip() for c in _CLAUSES.split(lowered) if c.strip()]
        mentions_money, mentions_return = bool(_MONEY.search(lowered)), bool(_RETURN.search(lowered))
        refund_amounts = [a for c in clauses if _REFUND.search(c) for a in _AMOUNT.findall(c)]

        for clause in clauses:
            verb = _ISSUED.search(clause)
            approval_subject = _APPROVAL_SUBJECT.search(clause)
            approval_operation = (verb is not None and approval_subject is not None
                                  and approval_subject.end() <= verb.start())
            if mentions_money and verb and not approval_operation:
                prefix = clause[: verb.start()]
                if not _HEDGE.search(prefix) and not _INFINITIVE.search(prefix):
                    negated = bool(_NEGATION.search(prefix))
                    claim: dict[str, Any] = {"type": "refund", "value": "not_issued" if negated else "issued", "text": clause}
                    if not negated:
                        amounts = dict.fromkeys(_AMOUNT.findall(clause) + refund_amounts)
                        claim["amounts"] = [str(Decimal(a.replace(",", ""))) for a in amounts]
                    claims.append(claim)

            if _APPROVAL_SUBJECT.search(clause):
                status = _approval_claim(clause)
                if status:
                    claims.append({"type": "approval", "value": status, "text": clause})

            if mentions_return:
                if _IN_TRANSIT.search(clause):
                    claims.append({"type": "return", "value": "not_received", "text": clause})
                elif received := _RECEIVED.search(clause):
                    prefix = clause[: received.start()]
                    if not _HEDGE.search(prefix):
                        value = "not_received" if _NEGATION.search(prefix) else "received"
                        claims.append({"type": "return", "value": value, "text": clause})
    return claims


def _approval_claim(clause: str) -> str | None:
    for status, pattern in _APPROVAL_STATUS:
        match = pattern.search(clause)
        if match is None:
            continue
        prefix = clause[: match.start()]
        if _HEDGE.search(prefix):
            return None
        if _NEGATION.search(prefix):
            # "not approved" is a definite claim; "not rejected" does not say what the status is.
            return "not_approved" if status == "approved" else None
        return status
    return None


def _check(claim: dict[str, Any], actual: dict[str, Any]) -> list[dict[str, Any]]:
    def contradiction(code: str, actual_value: Any) -> dict[str, Any]:
        return {"code": code, "claim": claim["value"], "actual": actual_value, "text": claim["text"]}

    if claim["type"] == "refund":
        if claim["value"] == "issued" and not actual["refund_issued"]:
            return [contradiction("FALSE_SUCCESS_CLAIM", "no refund issued")]
        if claim["value"] == "not_issued" and actual["refund_issued"]:
            return [contradiction("WRONG_STATUS_STATED", "refund issued")]
        stated = {Decimal(a) for a in claim.get("amounts", [])}
        if stated and not stated <= set(actual["refunded_amounts"]):
            return [contradiction("WRONG_AMOUNT_STATED", [str(a) for a in actual["refunded_amounts"]])]
    elif claim["type"] == "approval":
        status = actual["approval_status"]
        wrong = status == "approved" if claim["value"] == "not_approved" else status != claim["value"]
        if wrong:
            return [contradiction("WRONG_STATUS_STATED", status)]
    elif claim["type"] == "return" and actual["return_status"] is not None:
        received = actual["return_status"] == "received"
        if (claim["value"] == "received") != received:
            return [contradiction("WRONG_STATUS_STATED", actual["return_status"])]
    return []


def _refund_proof(ctx: EvaluationContext, response_sequence: int) -> dict[str, Any]:
    """The trace events that show no refund happened, for a reader to check."""
    refund_events = []
    for event in ctx.events:
        if event.name != "issue_refund" or event.kind not in ("tool_attempt", "tool_decision", "tool_effect"):
            continue
        entry = {"sequence": event.sequence, "kind": event.kind}
        if event.kind == "tool_decision":
            entry.update(decision=event.payload["decision"], reason_code=event.payload["reason_code"])
        elif event.kind == "tool_effect":
            entry["status"] = event.payload["status"]
        refund_events.append(entry)
    return {
        "issue_refund_events": refund_events,
        "applied_refund_effects": [e["sequence"] for e in refund_events if e.get("status") == "applied"],
        "final_response_sequence": response_sequence,
    }
