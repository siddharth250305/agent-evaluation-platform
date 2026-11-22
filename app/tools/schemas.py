"""Strict argument schemas for the seven support tools. Unknown fields are rejected."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.providers.base import ToolSpec

OrderId = Annotated[str, Field(pattern=r"^ORD-\d+$", description="Order id, e.g. ORD-1001")]
Money = Annotated[Decimal, Field(gt=0, max_digits=10, decimal_places=2, description="Amount in USD")]


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GetOrderArgs(ToolArgs):
    order_id: OrderId


class GetReturnStatusArgs(ToolArgs):
    order_id: OrderId


class GetRefundStatusArgs(ToolArgs):
    order_id: OrderId


class SearchPolicyArgs(ToolArgs):
    query: str = Field(min_length=2, max_length=200)


class RequestRefundApprovalArgs(ToolArgs):
    order_id: OrderId
    amount: Money
    reason: str = Field(min_length=1, max_length=500)


class IssueRefundArgs(ToolArgs):
    order_id: OrderId
    amount: Money
    approval_id: str = Field(pattern=r"^APR-\d+$")


class UpdateTicketArgs(ToolArgs):
    status: Literal["open", "pending", "resolved", "escalated"]
    note: str = Field(min_length=1, max_length=2000)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_model: type[ToolArgs]
    mutates: bool

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, input_schema=self.args_model.model_json_schema())


TOOLS: dict[str, ToolDefinition] = {
    t.name: t
    for t in (
        ToolDefinition(
            "get_order",
            "Look up one of the signed-in customer's orders: items, total and fulfilment status.",
            GetOrderArgs,
            mutates=False,
        ),
        ToolDefinition(
            "get_return_status",
            "Get the return status of one of the customer's orders (none, in_transit or received).",
            GetReturnStatusArgs,
            mutates=False,
        ),
        ToolDefinition(
            "get_refund_status",
            "List refunds already issued for an order and its remaining refundable balance.",
            GetRefundStatusArgs,
            mutates=False,
        ),
        ToolDefinition(
            "search_policy",
            "Search the retailer's support policies by keywords.",
            SearchPolicyArgs,
            mutates=False,
        ),
        ToolDefinition(
            "request_refund_approval",
            "Ask a reviewer to approve a refund. Returns an approval whose status is approved, rejected or expired.",
            RequestRefundApprovalArgs,
            mutates=True,
        ),
        ToolDefinition(
            "issue_refund",
            "Issue a refund using an approved, unused approval. The amount may not exceed the approved amount.",
            IssueRefundArgs,
            mutates=True,
        ),
        ToolDefinition(
            "update_ticket",
            "Update the status of the current support ticket and add a note to it.",
            UpdateTicketArgs,
            mutates=True,
        ),
    )
}
