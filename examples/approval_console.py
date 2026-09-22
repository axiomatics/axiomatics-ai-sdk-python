"""
Shared pieces of the human-approval examples: the invoices the payment tools work on, the
HUMAN — a console prompt that shows the approval request and reads the answer — and an audit
hook that prints each tool-invocation decision.

The policy decides on the invoice's REAL amount: ``with_invoice_amount`` adds it (as a double)
to the resource, rather than trusting an amount the model typed in.

Run an example interactively and you are asked "Approve? [y/N]". Pass ``--approve`` or
``--reject`` to answer every request without a prompt (handy for a scripted demo or CI); the
answer still goes through the framework's own pause/resume and back to the PDP.
"""
from __future__ import annotations

import argparse

from axiomatics_ai import AuthzHook, DefaultClientContextProvider, DefaultServerContextProvider

INVOICES = {
    "INV-42": {"amount": 40.0, "creditor": "Cafe Einstein"},
    "INV-43": {"amount": 500.0, "creditor": "Hotel Adlon"},
}

SYSTEM_PROMPT = (
    "You are a finance assistant. Use list_invoices to find an invoice, then pay_invoice or "
    "cancel_invoice with its exact invoice_id. Call each tool once. "
    "If a tool call is denied or not approved, say so plainly and do not retry."
)

_parser = argparse.ArgumentParser(add_help=False)
_group = _parser.add_mutually_exclusive_group()
_group.add_argument("--approve", action="store_true", help="approve every request without asking")
_group.add_argument("--reject", action="store_true", help="reject every request without asking")
_ARGS, _ = _parser.parse_known_args()


def list_invoices_text() -> str:
    return "\n".join(f"{i}: {v['amount']:.2f} EUR to {v['creditor']}" for i, v in INVOICES.items())


def ask_human(payload: dict) -> dict:
    """Show an approval request (``ApprovalRequest.to_payload()``) and return the answer in the
    shape every framework's resume accepts: ``{"approved": bool, "approver": ...}``."""
    print("\n  ┌─ approval needed ──────────────────────────────────────")
    print(f"  │ tool:   {payload['tool_name']}")
    print(f"  │ args:   {payload['args']}")
    print(f"  │ reason: {payload.get('reason') or '-'}")
    if _ARGS.approve or _ARGS.reject:
        approved = _ARGS.approve
        print(f"  └─ answered by flag: {'approved' if approved else 'rejected'}")
    else:
        approved = input("  └─ Approve? [y/N] ").strip().lower() in {"y", "yes"}
    return {"approved": approved, "approver": "console-user"}


def with_invoice_amount(resource: dict, args: dict | None) -> dict:
    """Add the invoice's real amount to the resource of a call about an invoice."""
    invoice = INVOICES.get((args or {}).get("invoice_id"))
    if invoice is not None:
        resource["amount"] = float(invoice["amount"])
    return resource


class InvoiceContextProvider(DefaultClientContextProvider):
    """Client-side identity (the default) plus the invoice's real amount."""

    def get_resource(self, resource_name, context, *, args=None, resource_name_attribute):
        resource = super().get_resource(resource_name, context, args=args, resource_name_attribute=resource_name_attribute)
        return with_invoice_amount(resource, args)


class InvoiceServerContextProvider(DefaultServerContextProvider):
    """Server-side identity (the caller's bearer token) plus the invoice's real amount."""

    def get_resource(self, resource_name, context, *, args=None, resource_name_attribute):
        resource = super().get_resource(resource_name, context, args=args, resource_name_attribute=resource_name_attribute)
        return with_invoice_amount(resource, args)


class AuditHook(AuthzHook):
    """Prints each tool-invocation decision — the audit trail of the approval flow."""

    def on_decision(self, event):
        if event.enforcement_point == "tool_invocation":
            extra = {k: v for k, v in event.detail.items() if k in ("source", "evaluated_again", "not_approved")}
            for kind in ("advice", "obligations"):
                # each is {"id": ..., "attributes": {...}}; show the id and the policy's reason
                for item in event.detail.get(kind, []):
                    reason = item["attributes"].get("reason")
                    extra[kind] = f"{item['id']}" + (f" ({reason})" if reason else "")
            print(f"  [audit] {event.decision} {extra or ''}")
