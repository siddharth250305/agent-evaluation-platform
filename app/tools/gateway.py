"""The single enforcement point between an agent and the environment.

Every call is recorded as separate trace facts:

* ``tool_attempt``  - exactly what the agent asked for (raw name and arguments)
* ``tool_decision`` - allowed or denied, with an internal reason code
* ``tool_effect``   - what actually changed in the environment (allowed calls only)

Internal reason codes (e.g. ``CROSS_CUSTOMER_ACCESS``) are for evaluators; the
agent only sees a neutral error code (e.g. ``order_not_found``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from pydantic_core import to_jsonable_python

from app.environment import IntegrityError, SupportEnvironment, diff_states
from app.providers.base import ToolSpec
from app.tools.schemas import TOOLS, ToolArgs, ToolDefinition
from app.trace import TraceRecorder

# Denials at admission: the call does not match any offered tool schema. Every other
# denial (authorization, preconditions) was a well-formed call the policy refused.
SCHEMA_REJECTIONS = frozenset({"UNKNOWN_TOOL", "TOOL_NOT_ALLOWED", "INVALID_ARGUMENTS"})


@dataclass(frozen=True)
class CustomerContext:
    """Derived server-side from the session. Never taken from tool arguments."""

    customer_id: str
    ticket_id: str | None


class _Denied(Exception):
    def __init__(self, reason_code: str, agent_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.agent_code = agent_code
        self.message = message


class ToolGateway:
    def __init__(
        self,
        env: SupportEnvironment,
        context: CustomerContext,
        trace: TraceRecorder,
        allowed_tools: Iterable[str],
    ) -> None:
        self._env = env
        self._context = context
        self._trace = trace
        self._allowed = tuple(allowed_tools)

    @property
    def tool_specs(self) -> tuple[ToolSpec, ...]:
        return tuple(TOOLS[name].spec() for name in self._allowed if name in TOOLS)

    def call(self, name: str, raw_args: Any, call_id: str | None = None) -> dict[str, Any]:
        attempt = self._trace.record(
            "tool_attempt", name, {"call_id": call_id, "args": to_jsonable_python(raw_args, fallback=repr)}
        )
        link = {"attempt_sequence": attempt.sequence}

        try:
            definition, args = self._admit(name, raw_args)
            self._authorize(name, args)
        except _Denied as denied:
            self._trace.record(
                "tool_decision",
                name,
                {**link, "decision": "denied", "reason_code": denied.reason_code, "agent_error": denied.agent_code},
            )
            return {"ok": False, "error": {"code": denied.agent_code, "message": denied.message}}

        self._trace.record("tool_decision", name, {**link, "decision": "allowed", "reason_code": "ALLOWED"})
        resource = getattr(args, "order_id", None) or self._context.ticket_id or "public"
        fault = self._env.operation_fault(name, resource)
        if fault:
            self._trace.record("tool_effect", name, {
                **link, "status": "failed", "error": fault, "state_changes": {},
            })
            return {"ok": False, "error": {"code": fault, "message": "the operation could not be completed"}}
        before = self._env.snapshot() if definition.mutates else None
        try:
            result = to_jsonable_python(self._execute(name, args))
        except IntegrityError as exc:
            self._trace.record("tool_effect", name, {**link, "status": "failed", "error": str(exc), "state_changes": {}})
            return {"ok": False, "error": {"code": "operation_failed", "message": "the operation could not be completed"}}

        changes = diff_states(before, self._env.snapshot()) if before is not None else {}
        self._trace.record(
            "tool_effect",
            name,
            {**link, "status": "applied" if changes else "no_change", "state_changes": changes, "result": result},
        )
        return {"ok": True, "result": result}

    # --- admission: allowlist + strict schema ---------------------------------

    def _admit(self, name: str, raw_args: Any) -> tuple[ToolDefinition, ToolArgs]:
        definition = TOOLS.get(name)
        if definition is None:
            raise _Denied("UNKNOWN_TOOL", "unknown_tool", f"no tool named {name!r}")
        if name not in self._allowed:
            raise _Denied("TOOL_NOT_ALLOWED", "tool_not_allowed", f"{name!r} is not available for this task")
        if not isinstance(raw_args, dict):
            raise _Denied("INVALID_ARGUMENTS", "invalid_arguments", "arguments must be a JSON object")
        try:
            return definition, definition.args_model.model_validate(raw_args)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors())
            raise _Denied("INVALID_ARGUMENTS", "invalid_arguments", detail) from exc

    # --- authorization + preconditions ----------------------------------------

    def _authorize(self, name: str, args: Any) -> None:
        if name in ("get_order", "get_return_status", "get_refund_status"):
            self._owned_order(args.order_id)
        elif name == "request_refund_approval":
            self._owned_order(args.order_id)
            returned = self._env.return_for(args.order_id)
            if returned is None or returned["status"] != "received":
                raise _Denied("PRECONDITION_RETURN_NOT_RECEIVED", "return_not_received", "the return has not been received")
            self._within_refundable(args.order_id, args.amount)
        elif name == "issue_refund":
            self._owned_order(args.order_id)
            approval = self._env.approval(args.approval_id)
            if approval is None or approval["order_id"] != args.order_id:
                raise _Denied("PRECONDITION_APPROVAL_MISSING", "approval_not_found", "no approval for this order")
            if approval["status"] != "approved":
                raise _Denied(
                    "PRECONDITION_APPROVAL_NOT_GRANTED", "approval_not_granted", f"approval is {approval['status']}"
                )
            if approval["consumed"]:
                raise _Denied("PRECONDITION_APPROVAL_ALREADY_USED", "approval_already_used", "approval was already used")
            if args.amount > approval["amount"]:
                raise _Denied(
                    "PRECONDITION_AMOUNT_EXCEEDS_APPROVAL", "amount_exceeds_approval", "amount exceeds the approved amount"
                )
            self._within_refundable(args.order_id, args.amount)
        elif name == "update_ticket":
            ticket_id = self._context.ticket_id
            ticket = self._env.ticket(ticket_id) if ticket_id else None
            if ticket is None or ticket["customer_id"] != self._context.customer_id:
                raise _Denied("NO_SESSION_TICKET", "ticket_not_found", "there is no ticket for this conversation")
        # search_policy: public information, no authorization needed.

    def _owned_order(self, order_id: str) -> dict[str, Any]:
        order = self._env.order(order_id)
        # Same agent-facing error for "missing" and "someone else's" so other customers stay invisible.
        if order is None:
            raise _Denied("RESOURCE_NOT_FOUND", "order_not_found", f"order {order_id} not found on this account")
        if order["customer_id"] != self._context.customer_id:
            raise _Denied("CROSS_CUSTOMER_ACCESS", "order_not_found", f"order {order_id} not found on this account")
        return order

    def _within_refundable(self, order_id: str, amount: Any) -> None:
        if amount > self._env.refundable(order_id):
            raise _Denied(
                "PRECONDITION_AMOUNT_EXCEEDS_REFUNDABLE", "amount_exceeds_refundable", "amount exceeds refundable balance"
            )

    # --- execution --------------------------------------------------------------

    def _execute(self, name: str, args: Any) -> dict[str, Any]:
        env = self._env
        if name == "get_order":
            order = env.order(args.order_id)
            return {"order_id": args.order_id, **{k: v for k, v in order.items() if k != "customer_id"}}
        if name == "get_return_status":
            return {"order_id": args.order_id, **(env.return_for(args.order_id) or {"status": "none"})}
        if name == "get_refund_status":
            return {
                "order_id": args.order_id,
                "refunds": env.refunds_for(args.order_id),
                "refundable_balance": env.refundable(args.order_id),
            }
        if name == "search_policy":
            return {"results": env.search_policies(args.query)}
        if name == "request_refund_approval":
            return env.create_approval(args.order_id, args.amount, args.reason)
        if name == "issue_refund":
            return env.create_refund(args.order_id, args.amount, args.approval_id)
        if name == "update_ticket":
            ticket = env.update_ticket(self._context.ticket_id, args.status, args.note)
            return {key: ticket[key] for key in ("order_id", "status", "notes")}
        raise AssertionError(f"unhandled tool {name}")  # unreachable: _admit checked TOOLS
