"""Authorization helpers shared by every integration path.

JWT claim decoding for the default server provider, context resolution, and the thin
decision helpers the enforcement layer calls.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from axiomatics_ai.core.pdp import PdpClient, PdpDecision, strip_none

logger = logging.getLogger(__name__)


# ── JWT / HTTP header utility ──────────────────────────────────────────────────


def decode_bearer_jwt_unverified(headers: dict[str, str]) -> dict[str, Any]:
    """Read the claims of the ``Authorization: Bearer`` JWT **without verifying its signature**.

    **Trust boundary.** This SDK authorizes; it does not authenticate. This helper assumes the
    header was already verified by your authentication layer (ingress, API gateway, auth
    middleware). The claims are read as they are and become subject attributes, so an
    unverified token yields whatever subject it asserts. To verify inside the authorization
    layer instead (defence in depth, or because this process is the edge), use
    ``AgentIdentityContextProvider`` with a ``JwtVerification``. Requires PyJWT.

    Args:
        headers (dict[str, str]): The request's HTTP headers.

    Returns:
        dict[str, Any]: The token's claims; empty without a bearer token or when it doesn't
        decode.
    """
    from axiomatics_ai._extras import requires_extra

    with requires_extra("jwt", "Decoding bearer tokens"):
        import jwt

    auth = next((v for k, v in headers.items() if k.lower() == "authorization"), "")
    if not auth.lower().startswith("bearer "):
        return {}
    token = auth.split(" ", 1)[1]
    try:
        # Deliberate: authentication is the caller's responsibility and happens upstream of
        # this SDK (see the trust-boundary note above). Verifying here would duplicate the
        # auth layer inside the authorization layer, and this process has no IdP key to
        # verify against. Integrators who want verification here override get_subject.
        # nosemgrep: unverified-jwt-decode
        return jwt.decode(token, options={"verify_signature": False})
    except Exception:  # noqa: BLE001
        # Any malformed token yields no claims, i.e. an empty subject, which policy evaluates
        # as an unauthenticated caller (deny). PyJWT raises several unrelated exception types
        # here, so narrowing risks a 500 instead of a deny.
        logger.warning("Failed to decode JWT from Authorization header; returning empty dict")
        return {}


# ── Shared context resolution ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolvedAuthzContext:
    """The subject and environment a context provider produced for one request.

    Returned by ``resolve_authz_context``, so an enforcement step calls the provider once.

    Attributes:
        subject (dict): The subject attributes, ``None`` values removed.
        environment (dict): The environment attributes, ``None`` values removed.
    """
    subject: dict     # full subject attributes
    environment: dict # XACML environment attributes


def resolve_authz_context(
    provider: Any,
    context: Any,
    *,
    agent_name: str | None = None,
    agent_attribute: str | None = None,
) -> ResolvedAuthzContext:
    """Get the subject and environment from a context provider in one call.

    ``None`` values are stripped here, the single point where every path first consumes the
    provider, so the emptiness check and the decision-cache key see the same dicts the request
    builders do. A provider that returns ``None`` for an absent attribute therefore reads as no
    identity.

    Args:
        provider (ContextProvider): The context provider.
        context (Any): The runtime context or request headers.
        agent_name (str, optional): The name of the agent in the app making the call, added to
            the subject as ``agent_attribute`` (``AuthZConfig.acting_agent_attribute``). Not
            added to an empty subject (no identity stays no identity), nor over a value the
            provider set.
        agent_attribute (str, optional): The subject attribute ``agent_name`` is sent as.

    Returns:
        ResolvedAuthzContext: The subject and environment.
    """
    subject = strip_none(provider.get_subject(context))
    if subject and agent_name and agent_attribute:
        subject = {agent_attribute: agent_name, **subject}
    return ResolvedAuthzContext(
        subject=subject,
        environment=strip_none(provider.get_environment(context)),
    )


# ── Tool visibility ────────────────────────────────────────────────────────────


def get_permitted_tools(
    pdp_client: PdpClient,
    *,
    subject: dict,
    action: dict,
    tool_names: list[str],
    environment: dict,
    tool_name_attribute: str = "tool_name",
    fail_open: bool = False,
) -> set[str]:
    """Ask the PDP which tools are permitted, in one batched request.

    Args:
        pdp_client (PdpClient): The PDP client.
        subject (dict): Subject attributes.
        action (dict): Action attributes, e.g. ``{"name": "tool_visibility"}``.
        tool_names (list[str]): The tools to decide.
        environment (dict): Environment attributes.
        tool_name_attribute (str): The resource attribute each tool name is sent as.
        fail_open (bool): What an undecided name resolves to.

    Returns:
        set[str]: The permitted tool names (a ``PermittedNames``).
    """
    return pdp_client.check_permission_batch(
        subject=subject,
        action=action,
        resource_names=tool_names,
        environment=environment,
        resource_attr_id=tool_name_attribute,
        fail_open=fail_open,
    )


# ── Tool invocation ────────────────────────────────────────────────────────────


def evaluate_tool_call(
    pdp_client: PdpClient,
    *,
    subject: dict,
    action: dict,
    resource: dict,
    environment: dict,
) -> PdpDecision:
    """Ask the PDP for one call's decision, with its obligations and advice.

    The caller resolves it (``fail_open``, obligations, approval); see ``PdpClient.evaluate``.

    Args:
        pdp_client (PdpClient): The PDP client.
        subject (dict): Subject attributes.
        action (dict): Action attributes.
        resource (dict): Resource attributes.
        environment (dict): Environment attributes.

    Returns:
        PdpDecision: The decision, as returned.
    """
    return pdp_client.evaluate(
        subject=subject,
        action=action,
        resource=resource,
        environment=environment,
    )


# ── Post-invocation ────────────────────────────────────────────────────────────


def _record_hash(record: dict, index: int) -> str:
    """Stable correlation key for a record: short SHA-256 of its JSON, prefixed by index."""
    try:
        digest = hashlib.sha256(
            json.dumps(record, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
    except Exception:  # noqa: BLE001
        # A record that won't JSON-serialise still needs a correlation key; the positional
        # index is a correct, if less stable, fallback.
        digest = str(index)
    return f"rec_{index}_{digest}"


def get_permitted_records(
    pdp_client: PdpClient,
    *,
    records: list[dict],
    subject: dict,
    action: dict,
    environment: dict,
    key_attribute: str = "record_hash",
    fail_open: bool = False,
) -> list[dict]:
    """Ask the PDP which records are permitted, in one batched request (one sub-request per record).

    Each record gets a stable hash, sent as ``key_attribute``, to match decisions back to it;
    the record's fields go along as resource attributes so a policy can decide on content
    (e.g. ``doc_type``). Fields that can't be sent as attributes (nested dicts or lists) are
    left out.

    Args:
        pdp_client (PdpClient): The PDP client.
        records (list[dict]): The records to decide.
        subject (dict): Subject attributes.
        action (dict): Action attributes.
        environment (dict): Environment attributes.
        key_attribute (str): The attribute the per-record hash is sent as.
        fail_open (bool): What an undecided record resolves to.

    Returns:
        list[dict]: The permitted records, in their original order.
    """
    if not records:
        return records

    indexed = [(i, rec, _record_hash(rec, i)) for i, rec in enumerate(records)]

    permitted_record_hashes = pdp_client.check_permission_batch(
        subject=subject,
        action=action,
        resource_names=[rh for _, _, rh in indexed],
        environment=environment,
        resource_attr_id=key_attribute,
        resource_extra_attrs=[rec for _, rec, _ in indexed],
        fail_open=fail_open,
    )

    return [rec for _, rec, rh in indexed if rh in permitted_record_hashes]
