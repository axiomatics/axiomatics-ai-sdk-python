"""
FastMCP server demonstrating ADS-driven post-processing with @protected_tool.

The tool always executes. The post_processor (strategy) is called after every
execution and is responsible for the ADS decision and any transformation:

  ColumnMaskStrategy   → Permit: full result returned; Deny: named fields masked
  Custom strategy      → full control; implement apply(result, context)

Two tools illustrate different post-processing strategies:

1. ``get_patient_records`` — uses ``ColumnMaskStrategy`` to mask known PII fields
   (``dob``, ``email``, ``phone``) on ADS Deny.

2. ``get_clinical_notes`` — uses a custom ``EmailRedactionStrategy`` that calls ADS
   and, on Deny, scans all string values with a regex and redacts embedded emails.

With domain_tool_decorator_masking.yaml loaded:
  alice → Permit on both tools → receives full, unmasked data
  bob   → Deny  on both tools → receives masked / email-redacted data

Prerequisites:
  - ADS running at http://localhost:8080 with domain_tool_decorator_masking.yaml loaded
  - fastmcp and PyJWT installed: uv sync --extra fastmcp

Run:
    uv run python examples/tool_decorator/masking_server.py
"""
from __future__ import annotations

import os

import re
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from axiomatics_ai import ColumnMaskStrategy
from axiomatics_ai.core.config import AuthZConfig
from axiomatics_ai.core.post_invocation import PostInvocationContext
from axiomatics_ai import PDPConnection, standard_tool_enforcement
from axiomatics_ai.tool_decorator import (
    configure_authz_decorator,
    protected_tool,
)

configure_authz_decorator(config=AuthZConfig(connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")), enforcement=standard_tool_enforcement()))

mcp = FastMCP("patient-records-server")

# ── Sample data ────────────────────────────────────────────────────────────────

_PATIENT_RECORDS = [
    {
        "id": "p1",
        "name": "Alice Smith",
        "dob": "1985-03-12",
        "email": "alice@hospital.example",
        "phone": "555-1234",
        "diagnosis": "hypertension",
    },
    {
        "id": "p2",
        "name": "Bob Jones",
        "dob": "1972-07-08",
        "email": "bob@hospital.example",
        "phone": "555-5678",
        "diagnosis": "diabetes",
    },
]

_CLINICAL_NOTES = [
    {
        "id": "n1",
        "author": "Dr. Carol",
        "text": "Patient contacted via alice@hospital.example. Follow-up scheduled.",
    },
    {
        "id": "n2",
        "author": "Dr. Dave",
        "text": "Referred to specialist. Contact: specialist@clinic.example for details.",
    },
    {
        "id": "n3",
        "author": "Dr. Carol",
        "text": "Routine checkup. No concerns.",
    },
]


# ── Tool 1: ColumnMaskStrategy ─────────────────────────────────────────────────
# alice (Permit) → full records including dob, email, phone
# bob   (Deny)   → records with dob, email, phone replaced by ***


@mcp.tool()
@protected_tool(post_processor=ColumnMaskStrategy("dob", "email", "phone"))
def get_patient_records(patient_id: str) -> list[dict]:
    """Return patient records. PII fields are masked for users without full access."""
    return [r for r in _PATIENT_RECORDS if r["id"] == patient_id] or _PATIENT_RECORDS


# ── Tool 2: custom strategy — regex email redaction ───────────────────────────
# alice (Permit) → full notes including email addresses in free text
# bob   (Deny)   → notes with embedded emails replaced by [EMAIL REDACTED]

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


class EmailRedactionStrategy:
    """
    Custom post-invocation strategy that redacts email addresses in free text.

    Calls ADS once for the whole result; on Deny, scans every string value
    and replaces any email pattern with ``[EMAIL REDACTED]``.

    Use this pattern when PII can appear anywhere in free-text values rather
    than in a known field. Use ``ColumnMaskStrategy`` when the fields are known.
    """

    def apply(self, result: Any, context: PostInvocationContext) -> Any:
        permitted = context.pdp_client.check_permission(
            subject=context.subject,
            action={"name": context.action_name},
            resource=context.resource,
            environment=context.environment,
        )
        if permitted:
            return result
        if not isinstance(result, list):
            return result
        return [
            {
                k: _EMAIL_RE.sub("[EMAIL REDACTED]", v) if isinstance(v, str) else v
                for k, v in record.items()
            }
            for record in result
        ]


@mcp.tool()
@protected_tool(post_processor=EmailRedactionStrategy())
def get_clinical_notes(note_id: str) -> list[dict]:
    """Return clinical notes. Email addresses in free text are redacted for users without full access."""
    return [r for r in _CLINICAL_NOTES if r["id"] == note_id] or _CLINICAL_NOTES


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
