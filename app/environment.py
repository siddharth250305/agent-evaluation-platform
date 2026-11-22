"""Isolated mock retailer backend, one instance per attempt.

The environment owns state and effects. Who may do what (identity,
authorization, preconditions) is decided by ``app.tools.gateway``; the
integrity guards here are a last line of defense, not the policy.
"""

from __future__ import annotations

import copy
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

ApprovalDecision = Literal["approved", "rejected", "expired"]

# Bump when environment behaviour changes; experiment manifests record it.
ENVIRONMENT_VERSION = "1.0.0"

_FIXTURE_KEYS = {
    "session",
    "customers",
    "orders",
    "returns",
    "refunds",
    "tickets",
    "policies",
    "approval_decisions",
    "faults",
}
_KEYED_COLLECTIONS = ("customers", "orders", "returns", "refunds", "approvals", "tickets")
_WORD = re.compile(r"[a-z0-9]+")


class IntegrityError(Exception):
    """An effect would corrupt state. Reaching this means the gateway let something through."""


@dataclass(frozen=True)
class Session:
    """Server-side session: who is signed in and which ticket this conversation belongs to."""

    customer_id: str
    ticket_id: str | None


class ApprovalSimulator:
    """Deterministic reviewer: the decision is a pure function of the order id."""

    def __init__(self, decisions: Mapping[str, ApprovalDecision], default: ApprovalDecision = "rejected") -> None:
        self._decisions = dict(decisions)
        self._default = default

    def decide(self, order_id: str) -> ApprovalDecision:
        return self._decisions.get(order_id, self._default)


def _money(value: Any) -> Decimal:
    return Decimal(str(value))


def _next_id(prefix: str, existing: Mapping[str, Any]) -> str:
    n = len(existing) + 1
    while f"{prefix}-{n:04d}" in existing:
        n += 1
    return f"{prefix}-{n:04d}"


class SupportEnvironment:
    def __init__(self, fixture: Mapping[str, Any], *, environment_id: str | None = None) -> None:
        unknown = set(fixture) - _FIXTURE_KEYS
        if unknown:
            raise ValueError(f"unknown fixture keys: {sorted(unknown)}")

        # Deep copy: the shared fixture must never be mutated by an attempt.
        state = copy.deepcopy(dict(fixture))
        self.environment_id = environment_id or uuid.uuid4().hex
        self._faults = state.get("faults", {})
        if not isinstance(self._faults, dict) or any(
            not isinstance(key, str) or value not in ("timeout", "http_503")
            for key, value in self._faults.items()
        ):
            raise ValueError("faults must map tool:resource keys to timeout or http_503")
        self.session = Session(
            customer_id=state["session"]["customer_id"],
            ticket_id=state["session"].get("ticket_id"),
        )
        self._approvals = ApprovalSimulator(state.get("approval_decisions", {}))
        self._state: dict[str, Any] = {
            "customers": state.get("customers", {}),
            "orders": {oid: {**o, "total": _money(o["total"])} for oid, o in state.get("orders", {}).items()},
            "returns": state.get("returns", {}),  # keyed by order_id
            "refunds": {rid: {**r, "amount": _money(r["amount"])} for rid, r in state.get("refunds", {}).items()},
            "approvals": {},
            "tickets": {tid: {**t, "notes": list(t.get("notes", []))} for tid, t in state.get("tickets", {}).items()},
            "policies": state.get("policies", []),
        }
        self.initial_snapshot = self.snapshot()

    def operation_fault(self, tool: str, resource: str) -> str | None:
        """Persistent pre-effect faults keyed by logical operation, never global call count."""
        return self._faults.get(f"{tool}:{resource}")

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    # --- reads (always return copies) ----------------------------------------

    def order(self, order_id: str) -> dict[str, Any] | None:
        return copy.deepcopy(self._state["orders"].get(order_id))

    def return_for(self, order_id: str) -> dict[str, Any] | None:
        return copy.deepcopy(self._state["returns"].get(order_id))

    def refunds_for(self, order_id: str) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(r)
            for _, r in sorted(self._state["refunds"].items())
            if r["order_id"] == order_id
        ]

    def approval(self, approval_id: str) -> dict[str, Any] | None:
        return copy.deepcopy(self._state["approvals"].get(approval_id))

    def ticket(self, ticket_id: str) -> dict[str, Any] | None:
        return copy.deepcopy(self._state["tickets"].get(ticket_id))

    def refundable(self, order_id: str) -> Decimal:
        refunded = sum(
            (r["amount"] for r in self._state["refunds"].values() if r["order_id"] == order_id),
            start=Decimal("0"),
        )
        return self._state["orders"][order_id]["total"] - refunded

    def search_policies(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        terms = set(_WORD.findall(query.lower()))
        scored = []
        for policy in self._state["policies"]:
            words = set(_WORD.findall(f"{policy['title']} {policy['text']}".lower()))
            overlap = len(terms & words)
            if overlap:
                scored.append((-overlap, policy["policy_id"], policy))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [copy.deepcopy(policy) for _, _, policy in scored[:limit]]

    # --- effects --------------------------------------------------------------

    def create_approval(self, order_id: str, amount: Decimal, reason: str) -> dict[str, Any]:
        if order_id not in self._state["orders"]:
            raise IntegrityError(f"approval for unknown order {order_id}")
        approvals = self._state["approvals"]
        approval = {
            "approval_id": _next_id("APR", approvals),
            "order_id": order_id,
            "amount": amount,
            "reason": reason,
            "status": self._approvals.decide(order_id),
            "consumed": False,
        }
        approvals[approval["approval_id"]] = approval
        return copy.deepcopy(approval)

    def create_refund(self, order_id: str, amount: Decimal, approval_id: str) -> dict[str, Any]:
        approval = self._state["approvals"].get(approval_id)
        usable = (
            approval is not None
            and approval["order_id"] == order_id
            and approval["status"] == "approved"
            and not approval["consumed"]
            and amount <= approval["amount"]
        )
        if not usable:
            raise IntegrityError(f"refund without a usable approval ({approval_id})")
        if amount > self.refundable(order_id):
            raise IntegrityError("refund exceeds refundable balance")
        refunds = self._state["refunds"]
        refund = {
            "refund_id": _next_id("RF", refunds),
            "order_id": order_id,
            "amount": amount,
            "approval_id": approval_id,
            "status": "issued",
        }
        approval["consumed"] = True
        refunds[refund["refund_id"]] = refund
        return copy.deepcopy(refund)

    def update_ticket(self, ticket_id: str, status: str, note: str) -> dict[str, Any]:
        ticket = self._state["tickets"].get(ticket_id)
        if ticket is None:
            raise IntegrityError(f"unknown ticket {ticket_id}")
        ticket["status"] = status
        ticket["notes"].append(note)
        return {"ticket_id": ticket_id, **copy.deepcopy(ticket)}


def diff_states(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Ids added, removed or modified per keyed collection; empty dict means no effect."""
    changes: dict[str, dict[str, list[str]]] = {}
    for name in _KEYED_COLLECTIONS:
        old, new = before.get(name, {}), after.get(name, {})
        delta = {
            "added": sorted(set(new) - set(old)),
            "removed": sorted(set(old) - set(new)),
            "modified": sorted(k for k in set(old) & set(new) if old[k] != new[k]),
        }
        delta = {kind: ids for kind, ids in delta.items() if ids}
        if delta:
            changes[name] = delta
    return changes
