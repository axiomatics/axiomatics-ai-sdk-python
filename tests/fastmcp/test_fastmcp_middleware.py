"""
Unit tests for FastMCPAuthZMiddleware.

Each test builds a real FastMCP server with real tools, stubs out ADS at the
authz_helpers level and get_http_headers at the dependencies level, then uses
fastmcp.Client to trigger tools/list and tools/call and asserts on observable
outcomes:
  - which tools are visible after list filtering
  - whether a denied call raises ToolError
  - whether a permitted call returns actual output
  - that ADS is never called when a feature flag is disabled
  - that the correct subject dict reaches get_permitted_tools (subject extraction check)
  - that tool call arguments are forwarded as resource attributes to ADS
  - that JSON array responses are filtered via get_permitted_records
  - that non-JSON / non-list responses are passed through unchanged with a warning

No real ADS instance is required.
"""
from __future__ import annotations

import asyncio
import json

import jwt
import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

import axiomatics_ai.core.authz_helpers as authz_helpers_module
import axiomatics_ai.core.enforcement as enforcement_module
import axiomatics_ai.core.llm_enrichment as llm_module
import axiomatics_ai.fastmcp.fastmcp_middleware as fmw_module
from axiomatics_ai.core.authz_helpers import decode_bearer_jwt_unverified
from axiomatics_ai.core.config import AuthZConfig, DefaultServerContextProvider, AttributeNames
from axiomatics_ai.fastmcp.fastmcp_middleware import FastMCPAuthZMiddleware
from axiomatics_ai import Enforcer, EnforcementPoint, PDPConnection, RecordFilterStrategy, intent_enricher, standard_tool_enforcement
from tests.pdp_fakes import decision_seam
from axiomatics_ai.core.pdp import PdpDecision


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


_DOCUMENTS = [
    {"id": "d1", "title": "Getting Started", "doc_type": "public"},
    {"id": "d2", "title": "Internal Spec",   "doc_type": "classified"},
    {"id": "d3", "title": "API Reference",   "doc_type": "public"},
]


def _make_server() -> FastMCP:
    mcp = FastMCP("test-server")

    @mcp.tool()
    def search_docs(query: str) -> str:
        """Search documentation."""
        return f"results for {query}"

    @mcp.tool()
    def get_account_balance(account_id: str) -> str:
        """Get account balance."""
        return f"balance for {account_id}: $1,000"

    @mcp.tool()
    def send_email(to: str, body: str) -> str:
        """Send an email."""
        return f"email sent to {to}"

    @mcp.tool()
    def list_documents() -> str:
        """Return a JSON array of documents for record-filtering tests."""
        return json.dumps(_DOCUMENTS)

    return mcp


def _make_jwt(claims: dict) -> str:
    return jwt.encode(claims, "secret", algorithm="HS256")


def _auth_header(claims: dict) -> dict[str, str]:
    return {"authorization": f"Bearer {_make_jwt(claims)}"}


ALL_TOOLS = {"search_docs", "get_account_balance", "send_email", "list_documents"}
ALICE_CLAIMS = {"username": "alice", "tenant_id": "acme"}


# ── Tool visibility ─────────────────────────────────────────────────────────────

def test_list_tools_filters_denied_tools(monkeypatch):
    """Denied tools are absent from the list returned to the client."""
    monkeypatch.setattr(
        enforcement_module, "get_permitted_tools",
        lambda _ads, *, tool_names, **kw: {"search_docs", "get_account_balance"},
    )
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(
        AuthZConfig(enforcement=standard_tool_enforcement(invocation=False, post_invocation=False))
    ))

    async def run():
        async with Client(mcp) as client:
            return {t.name for t in await client.list_tools()}

    assert _run(run()) == {"search_docs", "get_account_balance"}


def test_list_tools_disabled_returns_all_tools_without_calling_ads(monkeypatch):
    """With no TOOL_VISIBILITY enforcer all tools are returned and ADS is never called."""
    ads_calls: list = []
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda *a, **kw: ads_calls.append(1) or set())
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(
        enforcement=None,
    )))

    async def run():
        async with Client(mcp) as client:
            return {t.name for t in await client.list_tools()}

    assert _run(run()) == ALL_TOOLS
    assert not ads_calls


def test_list_tools_fail_open_returns_all_tools_when_ads_unreachable(monkeypatch):
    """With fail_open=True all tools are returned when ADS returns None."""
    monkeypatch.setattr(
        enforcement_module, "get_permitted_tools",
        lambda _ads, *, tool_names, fail_open=False, **kw:
            set(tool_names) if fail_open else set(),
    )
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(
        connection=PDPConnection(fail_open=True), enforcement=standard_tool_enforcement(invocation=False, post_invocation=False),
    )))

    async def run():
        async with Client(mcp) as client:
            return {t.name for t in await client.list_tools()}

    assert _run(run()) == ALL_TOOLS


def test_list_tools_fail_closed_returns_no_tools_when_ads_unreachable(monkeypatch):
    """With fail_open=False (default) no tools are returned when ADS returns None."""
    monkeypatch.setattr(
        enforcement_module, "get_permitted_tools",
        lambda _ads, *, tool_names, fail_open=False, **kw:
            set(tool_names) if fail_open else set(),
    )
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(
        connection=PDPConnection(fail_open=False), enforcement=standard_tool_enforcement(invocation=False, post_invocation=False),
    )))

    async def run():
        async with Client(mcp) as client:
            return {t.name for t in await client.list_tools()}

    assert _run(run()) == set()


# ── Tool invocation ────────────────────────────────────────────────────

def test_call_tool_permit_executes_and_returns_output(monkeypatch):
    """When ADS permits the call, the tool executes and returns its actual output."""
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: True))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement(post_invocation=False))))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("search_docs", {"query": "API"})

    result = _run(run())
    assert "results for API" in result.data


def test_call_tool_deny_raises_tool_error(monkeypatch):
    """When ADS denies the call, ToolError is raised with an access-denied message."""
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: False))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement(post_invocation=False))))

    async def run():
        async with Client(mcp) as client:
            await client.call_tool("send_email", {"to": "bob@example.com", "body": "hi"})

    with pytest.raises(ToolError, match="Access denied"):
        _run(run())


def test_call_tool_authz_disabled_executes_without_calling_ads(monkeypatch):
    """With no BEFORE_TOOL enforcer the tool always executes and ADS is never asked."""
    ads_calls: list = []
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call",
                        decision_seam(lambda *a, **kw: ads_calls.append(1) or False))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(
        enforcement=standard_tool_enforcement(invocation=False, post_invocation=False),
    )))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("search_docs", {"query": "test"})

    result = _run(run())
    assert not ads_calls
    assert "results for test" in result.data


def test_call_tool_fail_open_permits_when_ads_unreachable(monkeypatch):
    """With fail_open=True the call proceeds when ADS returns None (unreachable)."""
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(
        enforcement_module, "evaluate_tool_call",
        lambda *a, **kw: PdpDecision(None),
    )
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(
        connection=PDPConnection(fail_open=True), enforcement=standard_tool_enforcement(post_invocation=False),
    )))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("search_docs", {"query": "test"})

    result = _run(run())
    assert "results for test" in result.data


def test_call_tool_fail_closed_denies_when_ads_unreachable(monkeypatch):
    """With fail_open=False (default) the call is denied when ADS returns None."""
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(
        enforcement_module, "evaluate_tool_call",
        lambda *a, **kw: PdpDecision(None),
    )
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(
        connection=PDPConnection(fail_open=False), enforcement=standard_tool_enforcement(post_invocation=False),
    )))

    async def run():
        async with Client(mcp) as client:
            await client.call_tool("search_docs", {"query": "test"})

    with pytest.raises(ToolError, match="Access denied"):
        _run(run())


# ── Subject extraction ─────────────────────────────────────────────────────────
#
# The subject-aware stub returns different tool sets depending on the subject
# extracted from the JWT. This verifies that the JWT claims reach get_permitted_tools
# as the subject — if extraction is broken the wrong set comes back.

def _subject_aware_filter(subject: dict, tool_names: list[str]) -> set[str]:
    if subject.get("username") == "alice":
        return set(tool_names)
    if subject.get("username") == "bob":
        return {"search_docs"} & set(tool_names)
    return set()


@pytest.mark.parametrize("claims,expected", [
    ({"username": "alice"}, ALL_TOOLS),
    ({"username": "bob"},   {"search_docs"}),
    ({"username": "charlie"}, set()),
])
def test_correct_subject_forwarded_to_get_permitted_tools(monkeypatch, claims, expected):
    """
    subject extracted from the JWT is forwarded to get_permitted_tools.
    The subject-aware stub returns different sets per user so if JWT extraction
    or subject forwarding is broken the assertion will fail.
    """
    monkeypatch.setattr(
        enforcement_module, "get_permitted_tools",
        lambda _ads, *, subject, tool_names, **kw:
            _subject_aware_filter(subject, tool_names),
    )
    monkeypatch.setattr(fmw_module, "get_http_headers",
                        lambda **kw: _auth_header(claims))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(
        enforcement=standard_tool_enforcement(invocation=False, post_invocation=False),
    )))

    async def run():
        async with Client(mcp) as client:
            return {t.name for t in await client.list_tools()}

    assert _run(run()) == expected, f"claims={claims!r}: expected {expected}"


# ── decode_bearer_jwt_unverified ─────────────────────────────────────────────────────────

def test_decode_bearer_jwt_returns_all_claims():
    """decode_bearer_jwt_unverified returns the full JWT claims dict."""
    claims = {"username": "alice", "tenant_id": "acme", "sub": "user-123"}
    headers = _auth_header(claims)

    result = decode_bearer_jwt_unverified(headers)

    assert result["username"] == "alice"
    assert result["tenant_id"] == "acme"
    assert result["sub"] == "user-123"


def test_decode_bearer_jwt_returns_empty_dict_when_no_auth_header():
    assert decode_bearer_jwt_unverified({}) == {}


def test_decode_bearer_jwt_returns_empty_dict_for_invalid_jwt():
    assert decode_bearer_jwt_unverified({"authorization": "Bearer not-a-jwt"}) == {}


def test_decode_bearer_jwt_returns_empty_dict_for_non_bearer_scheme():
    assert decode_bearer_jwt_unverified({"authorization": "Basic dXNlcjpwYXNz"}) == {}


def test_decode_bearer_jwt_is_case_insensitive_on_header_name():
    """FastMCP passes headers with their original HTTP capitalization (Authorization,
    not authorization). decode_bearer_jwt_unverified must handle both."""
    claims = {"username": "alice"}
    token = _make_jwt(claims)
    assert decode_bearer_jwt_unverified({"Authorization": f"Bearer {token}"})["username"] == "alice"


# ── Custom context provider ────────────────────────────────────────────────────
#
# The motivation: DefaultServerContextProvider forwards all JWT claims as subject
# attributes. A custom provider is needed when you want to:
#   - verify the JWT signature before trusting its claims
#   - read extra HTTP headers (e.g. x-tenant-id) into the environment category
#   - inject resource attributes derived from headers (e.g. tenant scope)
#   - override the action name based on a header value
#
# Each test adds one extra header alongside the standard Authorization header and
# verifies that the custom provider's output reaches the relevant ADS helper.


def test_custom_provider_get_subject_reads_custom_header(monkeypatch):
    """
    Custom provider reads identity from a non-Authorization header (``x-user-id``)
    instead of a JWT. The mapped subject dict must reach get_permitted_tools.

    Use case: internal service mesh where identity comes from a sidecar-injected
    header rather than a client-supplied JWT.
    """
    captured_subjects: list[dict] = []

    class MyProvider(DefaultServerContextProvider):
        def get_subject(self, headers):
            return {"service_id": headers.get("x-user-id", "unknown")}

    monkeypatch.setattr(
        enforcement_module, "get_permitted_tools",
        lambda _ads, *, subject, tool_names, **kw:
            captured_subjects.append(subject) or set(tool_names),
    )
    monkeypatch.setattr(
        fmw_module, "get_http_headers",
        lambda **kw: {"x-user-id": "svc-42"},
    )

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(
        AuthZConfig(enforcement=standard_tool_enforcement(invocation=False, post_invocation=False),
                    context_provider=MyProvider()),
    ))

    async def run():
        async with Client(mcp) as client:
            return {t.name for t in await client.list_tools()}

    assert _run(run()) == ALL_TOOLS
    assert captured_subjects, "get_permitted_tools was never called"
    assert captured_subjects[0] == {"service_id": "svc-42"}


def test_custom_provider_get_environment_reads_extra_headers(monkeypatch):
    """
    Custom provider reads ``x-forwarded-for`` and ``x-session-id`` into the
    XACML environment. Those attributes must reach evaluate_tool_call.

    Use case: IP-based or session-based access policies.
    """
    captured_envs: list[dict] = []

    class MyProvider(DefaultServerContextProvider):
        def get_environment(self, headers):
            return {
                "ip_address": headers.get("x-forwarded-for", ""),
                "session_id": headers.get("x-session-id", ""),
            }

    def capturing_permit(*a, environment, **kw):
        captured_envs.append(environment)
        return True

    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(capturing_permit))
    monkeypatch.setattr(
        fmw_module, "get_http_headers",
        lambda **kw: {
            **_auth_header(ALICE_CLAIMS),
            "x-forwarded-for": "203.0.113.5",
            "x-session-id": "sess-abc",
        },
    )

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(
        AuthZConfig(enforcement=standard_tool_enforcement(post_invocation=False),
                    context_provider=MyProvider()),
    ))

    async def run():
        async with Client(mcp) as client:
            await client.call_tool("search_docs", {"query": "hello"})

    _run(run())

    assert captured_envs, "evaluate_tool_call was never called"
    assert captured_envs[0]["ip_address"] == "203.0.113.5"
    assert captured_envs[0]["session_id"] == "sess-abc"


def test_custom_provider_get_resource_injects_tenant_scope(monkeypatch):
    """
    Custom provider adds a ``tenant_id`` resource attribute from the JWT so
    ADS policies can scope access by tenant.

    Use case: multi-tenant deployment where tool access is restricted per tenant.
    """
    captured_resources: list[dict] = []

    class MyProvider(DefaultServerContextProvider):
        def get_resource(self, resource_name, headers, *, args=None, resource_name_attribute):
            resource = super().get_resource(
                resource_name, headers, args=args, resource_name_attribute=resource_name_attribute,
            )
            claims = decode_bearer_jwt_unverified(headers)
            if claims.get("tenant_id"):
                resource["tenant_id"] = claims["tenant_id"]
            return resource

    def capturing_permit(*a, resource, **kw):
        captured_resources.append(resource)
        return True

    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(capturing_permit))
    monkeypatch.setattr(fmw_module, "get_http_headers",
                        lambda **kw: _auth_header(ALICE_CLAIMS))  # ALICE_CLAIMS has tenant_id

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(
        AuthZConfig(enforcement=standard_tool_enforcement(post_invocation=False),
                    context_provider=MyProvider()),
    ))

    async def run():
        async with Client(mcp) as client:
            await client.call_tool("get_account_balance", {"account_id": "ACC-001"})

    _run(run())

    assert captured_resources, "evaluate_tool_call was never called"
    resource = captured_resources[0]
    assert resource["tool_name"] == "get_account_balance"
    assert resource["account_id"] == "ACC-001"
    assert resource["tenant_id"] == "acme"


# ── context_provider resolution ─────────────────────────────────────────────────

def test_provider_set_only_on_config_is_used(monkeypatch):
    """Regression: a context_provider set on AuthZConfig (not passed as the constructor
    argument) must be the one that builds the ADS request. Before the fix the config field
    was silently dropped and DefaultServerContextProvider ran instead, so a user's custom
    subject attributes never reached ADS and every request was denied for a subject whose
    attribute names didn't match the policy dictionary — with no error and no log line."""
    captured_subjects: list[dict] = []

    class ConfigProvider(DefaultServerContextProvider):
        def get_subject(self, headers):
            return {"provider_marker": "from-config"}

    monkeypatch.setattr(
        enforcement_module, "get_permitted_tools",
        lambda _ads, *, subject, tool_names, **kw:
            captured_subjects.append(subject) or set(tool_names),
    )
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    # Provider set ONLY on the config — no context_provider= argument.
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(
        context_provider=ConfigProvider(),
        enforcement=standard_tool_enforcement(invocation=False, post_invocation=False),
    )))

    async def run():
        async with Client(mcp) as client:
            return {t.name for t in await client.list_tools()}

    assert _run(run()) == ALL_TOOLS
    assert captured_subjects, "get_permitted_tools was never called"
    assert captured_subjects[0] == {"provider_marker": "from-config"}


# ── Tool call arguments forwarded as resource attributes ───────────────────────

def test_call_tool_arguments_forwarded_to_evaluate_tool_call(monkeypatch):
    """
    Tool call arguments must arrive in the ``resource`` dict passed to
    evaluate_tool_call so ADS policies can match on argument values.
    """
    captured: list[dict] = []

    def capturing_permit(*a, resource, **kw):
        captured.append(resource)
        return True

    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(capturing_permit))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement(post_invocation=False))))

    async def run():
        async with Client(mcp) as client:
            await client.call_tool("get_account_balance", {"account_id": "ACC-001"})

    _run(run())

    assert captured, "evaluate_tool_call was never called"
    resource = captured[0]
    assert resource == {"tool_name": "get_account_balance", "account_id": "ACC-001"}


def test_custom_tool_name_attribute_reaches_every_tool_point(monkeypatch):
    """AuthZConfig.attribute_names.tool names the tool at visibility (batch attribute id),
    invocation, and ColumnMask post-invocation (single request) alike."""
    from axiomatics_ai import ColumnMaskStrategy
    from axiomatics_ai.core.pdp import PdpClient

    batch_attr_ids: list[str] = []
    invocation_resources: list[dict] = []
    post_resources: list[dict] = []

    def capturing_visibility(_ads, *, tool_names, tool_name_attribute, **kw):
        batch_attr_ids.append(tool_name_attribute)
        return set(tool_names)

    def capturing_permit(*a, resource, **kw):
        invocation_resources.append(resource)
        return True

    def capturing_post(self, *, resource, **kw):
        post_resources.append(resource)
        return True

    monkeypatch.setattr(enforcement_module, "get_permitted_tools", capturing_visibility)
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(capturing_permit))
    monkeypatch.setattr(PdpClient, "check_permission", capturing_post)
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(
        attribute_names=AttributeNames(tool="tid"),
        enforcement=standard_tool_enforcement(strategy=ColumnMaskStrategy("title")),
    )))

    async def run():
        async with Client(mcp) as client:
            await client.list_tools()
            await client.call_tool("list_documents", {})

    _run(run())

    assert batch_attr_ids and set(batch_attr_ids) == {"tid"}
    assert invocation_resources == [{"tid": "list_documents"}]
    assert post_resources == [{"tid": "list_documents"}]


@pytest.mark.parametrize("amount", ["5000", 5000.0, True])
def test_argument_of_another_type_than_the_schema_is_refused_before_the_pdp(monkeypatch, amount):
    """The PDP would see "5000" (string) while the tool runs on 5000 (integer) after coercion."""
    asked = []
    monkeypatch.setattr(enforcement_module, "get_permitted_tools", lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: asked.append(1) or True))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))
    mcp = FastMCP("bank")

    @mcp.tool
    def transfer(amount: int, note: str | None = None) -> str:
        return f"sent {amount}"

    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement())))

    async def run(args):
        async with Client(mcp) as client:
            return await client.call_tool("transfer", args)

    with pytest.raises(ToolError, match="does not match its schema"):
        _run(run({"amount": amount}))
    assert asked == []
    assert _run(run({"amount": 5000, "note": None})).data == "sent 5000"


# ── Post-invocation ───────────────────────────────────────────────────────────

def test_post_invocation_keeps_only_permitted_records(monkeypatch):
    """
    When the tool returns a JSON array of objects, get_permitted_records is called once
    and only the permitted records are returned to the client.
    """
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: True))
    # Allow only public documents
    monkeypatch.setattr(
        authz_helpers_module, "get_permitted_records",
        lambda _ads, *, records, **kw: [r for r in records if r.get("doc_type") == "public"],
    )
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement())))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("list_documents", {})

    result = _run(run())
    returned = json.loads(result.data)
    assert len(returned) == 2
    assert all(r["doc_type"] == "public" for r in returned)
    assert {r["id"] for r in returned} == {"d1", "d3"}


def test_post_invocation_disabled_returns_all_records_without_calling_ads(monkeypatch):
    """With no AFTER_TOOL enforcer, get_permitted_records is never called and full output is returned."""
    ads_calls: list = []
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: True))
    monkeypatch.setattr(authz_helpers_module, "get_permitted_records",
                        lambda *a, **kw: ads_calls.append(1) or [])
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement(post_invocation=False))))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("list_documents", {})

    result = _run(run())
    assert not ads_calls
    assert json.loads(result.data) == _DOCUMENTS


def test_post_invocation_plain_string_passed_through_with_warning(monkeypatch, caplog):
    """
    A tool that returns a plain string (not JSON) is incompatible with record
    filtering. The response is returned unchanged and a warning is logged.
    """
    import logging

    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: True))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement())))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("search_docs", {"query": "hello"})

    with caplog.at_level(logging.WARNING, logger="axiomatics_ai.core.enforcement"):
        result = _run(run())

    assert "results for hello" in result.data
    assert any("no list or object the strategy can inspect" in r.message for r in caplog.records)


def test_post_invocation_json_object_is_filtered_as_a_single_record(monkeypatch):
    """
    A tool that returns a JSON object (not an array) is a single record: it goes to the PDP
    like any other record, and a denied one is replaced by ``{}`` rather than returned as-is.
    """
    from fastmcp import FastMCP as _FastMCP

    mcp = _FastMCP("test-server-obj")

    @mcp.tool()
    def get_config() -> str:
        """Returns a JSON object, not an array."""
        return json.dumps({"version": "1.0", "env": "prod"})

    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: True))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))
    seen = []
    monkeypatch.setattr(
        authz_helpers_module, "get_permitted_records",
        lambda _ads, *, records, **kw: seen.extend(records) or [],
    )

    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement(visibility=False))))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("get_config", {})

    result = _run(run())

    assert seen == [{"version": "1.0", "env": "prod"}]
    assert json.loads(result.data) == {}


def test_post_invocation_skip_is_reported_to_hooks(monkeypatch):
    """
    A tool whose output holds no records is returned unfiltered — but that must be
    *visible*. Without a decision event, an audit consumer cannot tell "the strategy ran
    and permitted everything" from "the strategy never ran".
    """
    from axiomatics_ai.core.hooks import AuthzHook

    seen = []

    class Recorder(AuthzHook):
        def on_decision(self, event):
            seen.append(event)

    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: True))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(
        enforcement=standard_tool_enforcement(), hooks=[Recorder()],
    )))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("search_docs", {"query": "hello"})

    result = _run(run())

    assert "results for hello" in result.data          # unchanged, as before
    skips = [e for e in seen
             if e.enforcement_point == "post_invocation" and e.decision == "skipped"]
    assert len(skips) == 1
    assert "no list or object" in skips[0].detail["reason"]
    assert skips[0].detail["withheld"] is False


def test_post_invocation_fail_closed_withholds_unrecognized_output(monkeypatch):
    """With ``on_unrecognized_output="fail_closed"``, output the strategy cannot inspect
    (plain text) is withheld from the client instead of returned unfiltered."""

    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: True))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    enforcement = standard_tool_enforcement()
    enforcement[EnforcementPoint.AFTER_TOOL] = Enforcer(
        action_name="post_invocation", strategy=RecordFilterStrategy(),
        on_unrecognized_output="fail_closed",
    )
    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=enforcement)))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("search_docs", {"query": "hello"})

    with pytest.raises(ToolError, match="withheld"):
        _run(run())


def test_post_invocation_filters_structured_content(monkeypatch):
    """
    A tool with a typed return has ``structured_content`` beside its text block. Both must
    be filtered: a client reading ``structuredContent`` must not see the denied records.
    """
    from fastmcp import FastMCP as _FastMCP

    mcp = _FastMCP("test-server-structured")

    @mcp.tool()
    def list_typed() -> list[dict]:
        """Returns records with an output schema, so FastMCP adds structured_content."""
        return _DOCUMENTS

    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: True))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))
    monkeypatch.setattr(
        authz_helpers_module, "get_permitted_records",
        lambda _ads, *, records, **kw: [r for r in records if r.get("doc_type") == "public"],
    )
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement(visibility=False))))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("list_typed", {})

    result = _run(run())
    public = [d for d in _DOCUMENTS if d["doc_type"] == "public"]
    assert result.structured_content == {"result": public}
    assert json.loads(result.content[0].text) == public


def test_post_invocation_of_a_tool_returning_json_text_keeps_it_json(monkeypatch):
    """A ``-> str`` tool that returns JSON: the filtered text is JSON, not a JSON string literal."""
    from fastmcp import FastMCP as _FastMCP

    mcp = _FastMCP("test-server-json-text")

    @mcp.tool()
    def search_text() -> str:
        """Returns the records as JSON text."""
        return json.dumps(_DOCUMENTS)

    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: True))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))
    monkeypatch.setattr(
        authz_helpers_module, "get_permitted_records",
        lambda _ads, *, records, **kw: [r for r in records if r.get("doc_type") == "public"],
    )
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement(visibility=False))))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("search_text", {})

    result = _run(run())
    assert json.loads(result.content[0].text) == [d for d in _DOCUMENTS if d["doc_type"] == "public"]


def test_post_invocation_withholds_an_embedded_resource_beside_the_data_under_fail_closed(monkeypatch):
    """A resource block can restate the records the text block holds: under fail_closed it is
    withheld, not passed through unfiltered."""
    from fastmcp import FastMCP as _FastMCP
    from fastmcp.tools.tool import ToolResult
    from mcp.types import EmbeddedResource, TextContent, TextResourceContents

    mcp = _FastMCP("test-server-resource-block")

    @mcp.tool()
    def docs_with_resource():
        """Returns the records twice: as text and as an embedded JSON resource."""
        data = json.dumps(_DOCUMENTS)
        return ToolResult(content=[
            TextContent(type="text", text=data),
            EmbeddedResource(type="resource", resource=TextResourceContents(
                uri="mem://docs", mimeType="application/json", text=data,
            )),
        ])

    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: True))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))
    monkeypatch.setattr(
        authz_helpers_module, "get_permitted_records",
        lambda _ads, *, records, **kw: [r for r in records if r.get("doc_type") == "public"],
    )
    enforcement = standard_tool_enforcement(visibility=False)
    enforcement[EnforcementPoint.AFTER_TOOL] = Enforcer(
        action_name="post_invocation", strategy=RecordFilterStrategy(), on_unrecognized_output="fail_closed",
    )
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=enforcement)))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("docs_with_resource", {})

    result = _run(run())
    assert [b.type for b in result.content] == ["text", "text"]
    assert json.loads(result.content[0].text) == [d for d in _DOCUMENTS if d["doc_type"] == "public"]
    assert "withheld" in result.content[1].text
    assert "classified" not in json.dumps([b.model_dump() for b in result.content])


def test_post_invocation_filters_records_nested_under_a_dict_key(monkeypatch):
    """
    Records wrapped in a dict (``{"results": [...]}``) are filtered like a bare list.

    The previous hand-rolled shape check only accepted a top-level JSON array, so this
    shape passed through unfiltered on the fastmcp surface while the client-side adapters
    — which share ``extract_records`` — handled it correctly.
    """
    from fastmcp import FastMCP as _FastMCP

    mcp = _FastMCP("test-server-nested")

    @mcp.tool()
    def list_wrapped() -> str:
        """Returns records nested under a key."""
        return json.dumps({"results": _DOCUMENTS})

    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: True))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))
    monkeypatch.setattr(
        authz_helpers_module, "get_permitted_records",
        lambda _ads, *, records, **kw: [r for r in records if r["doc_type"] == "public"],
    )

    mcp.add_middleware(FastMCPAuthZMiddleware(
        AuthZConfig(enforcement=standard_tool_enforcement(visibility=False))
    ))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("list_wrapped", {})

    result = _run(run())

    returned = json.loads(result.data)
    titles = [r["title"] for r in returned["results"]]
    assert titles == ["Getting Started", "API Reference"]   # classified record removed
    assert "Internal Spec" not in result.data


def test_post_invocation_calls_filter_records_exactly_once(monkeypatch):
    """get_permitted_records must be called exactly once per tool call (one ADS multi-request)."""
    call_count = [0]

    def counting_filter(_ads, *, records, **kw):
        call_count[0] += 1
        return records  # permit all

    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: True))
    monkeypatch.setattr(authz_helpers_module, "get_permitted_records", counting_filter)
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement())))

    async def run():
        async with Client(mcp) as client:
            await client.call_tool("list_documents", {})

    _run(run())
    assert call_count[0] == 1


# ── Intent enrichers (IBAC) on a server path ────────────────────────────────────
#
# A server never sees the user's messages, so BEFORE_TOOL enrichers are always unavailable
# here: the middleware warns at construction, and fail_closed denies every call.

def _intent_config(**enforcer_kw) -> AuthZConfig:
    return AuthZConfig(enforcement={EnforcementPoint.BEFORE_TOOL: Enforcer(
        action_name="tool_invocation", enrichers=[intent_enricher(model="stub/model")], **enforcer_kw,
    )})


def test_before_tool_enrichers_log_a_warning_at_init(caplog):
    with caplog.at_level("WARNING", logger="axiomatics_ai.fastmcp.fastmcp_middleware"):
        FastMCPAuthZMiddleware(_intent_config())
    assert any("never sees the user's messages" in r.getMessage() for r in caplog.records)


def test_no_warning_without_before_tool_enrichers(caplog):
    with caplog.at_level("WARNING", logger="axiomatics_ai.fastmcp.fastmcp_middleware"):
        FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement()))
    assert not [r for r in caplog.records if r.name == "axiomatics_ai.fastmcp.fastmcp_middleware"]


def test_call_tool_with_intent_enricher_fail_closed_denies_without_ads(monkeypatch):
    llm_calls: list = []
    ads_calls: list = []
    monkeypatch.setattr(llm_module, "_litellm_complete_json", lambda *a, **kw: llm_calls.append(1) or "{}")
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(lambda *a, **kw: ads_calls.append(1) or True))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(_intent_config()))

    async def run():
        async with Client(mcp) as client:
            await client.call_tool("search_docs", {"query": "test"})

    with pytest.raises(ToolError, match="Access denied"):
        _run(run())
    assert not ads_calls and not llm_calls


def test_call_tool_with_intent_enricher_fail_open_sends_unavailable_sentinel(monkeypatch):
    seen_env: list = []
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call",
                        decision_seam(lambda *a, environment, **kw: seen_env.append(environment) or True))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))

    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(_intent_config(on_unavailable="fail_open")))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("search_docs", {"query": "test"})

    assert "results for test" in _run(run()).data
    assert seen_env[-1]["intent_available"] is False


def test_pdp_calls_run_off_the_event_loop(monkeypatch):
    # A slow PDP must not stall every MCP session: listing and calling ask it on a worker thread.
    import threading

    threads: dict[str, list] = {"list": [], "call": []}
    monkeypatch.setattr(enforcement_module, "get_permitted_tools",
                        lambda _ads, *, tool_names, **kw: threads["list"].append(threading.current_thread()) or set(tool_names))
    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", decision_seam(
        lambda *a, **kw: threads["call"].append(threading.current_thread()) or True))
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))
    mcp = _make_server()
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement(post_invocation=False))))

    async def run():
        async with Client(mcp) as client:
            await client.list_tools()
            await client.call_tool("search_docs", {"query": "x"})
        return threading.current_thread()

    loop_thread = _run(run())
    assert threads["list"] and threads["call"]
    assert loop_thread not in threads["list"] + threads["call"]


def test_a_rejected_pdp_request_reaches_the_client_as_a_generic_error(monkeypatch):
    """PdpRequestError names the PDP's URL and echoes its answer: that stays in the server log."""
    from axiomatics_ai.core.pdp import PdpRequestError

    def pdp_rejects(*args, **kwargs):
        raise PdpRequestError("PDP rejected the request at https://pdp.corp.internal/authorize with HTTP 400: attr echo")

    mcp = FastMCP("test-server-pdp-error")

    @mcp.tool()
    def hello() -> str:
        """Says hello."""
        return "hi"

    monkeypatch.setattr(enforcement_module, "evaluate_tool_call", pdp_rejects)
    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig(enforcement=standard_tool_enforcement(visibility=False))))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("hello", {})

    with pytest.raises(ToolError) as excinfo:
        _run(run())
    assert "Authorization is unavailable" in str(excinfo.value)
    assert "pdp.corp.internal" not in str(excinfo.value) and "attr echo" not in str(excinfo.value)


def test_a_pdp_error_raised_inside_a_tool_reaches_the_client_as_a_generic_error(monkeypatch):
    """FastMCP wraps an exception from a tool body (a wrapped RAG store's PDP call) in its own
    ToolError: the PDP's details must not travel in that one either."""
    from axiomatics_ai.core.pdp import PdpRequestError

    mcp = FastMCP("test-server-pdp-error-in-tool")

    @mcp.tool()
    def search() -> str:
        """Searches a store that asks the PDP itself."""
        raise PdpRequestError("PDP rejected the request at https://pdp.corp.internal/constraints with HTTP 400: attr echo")

    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig()))

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("search", {})

    with pytest.raises(ToolError) as excinfo:
        _run(run())
    assert "Authorization is unavailable" in str(excinfo.value)
    assert "pdp.corp.internal" not in str(excinfo.value)


def test_a_request_with_two_authorization_headers_is_anonymous(monkeypatch):
    """MCP's bearer auth checks the first header, a header dict keeps the last: trust neither."""
    import starlette.datastructures

    class _Request:
        headers = starlette.datastructures.Headers(raw=[(b"authorization", b"Bearer mine"), (b"authorization", b"Bearer forged")])

    monkeypatch.setattr(fmw_module, "get_http_request", lambda: _Request())
    assert fmw_module.without_duplicate_authorization({"authorization": "Bearer forged", "x": "1"}) == {"x": "1"}

    class _One:
        headers = starlette.datastructures.Headers(raw=[(b"authorization", b"Bearer mine")])

    monkeypatch.setattr(fmw_module, "get_http_request", lambda: _One())
    assert fmw_module.without_duplicate_authorization({"authorization": "Bearer mine"}) == {"authorization": "Bearer mine"}


def test_a_pdp_error_raised_by_a_resource_reaches_the_client_as_a_generic_error(monkeypatch):
    from axiomatics_ai.core.pdp import PdpRequestError

    mcp = FastMCP("test-server-pdp-error-in-resource")

    @mcp.resource("data://docs")
    def docs() -> str:
        """A resource backed by a store that asks the PDP."""
        raise PdpRequestError("PDP rejected the request at https://pdp.corp.internal with HTTP 400")

    monkeypatch.setattr(fmw_module, "get_http_headers", lambda **kw: _auth_header(ALICE_CLAIMS))
    mcp.add_middleware(FastMCPAuthZMiddleware(AuthZConfig()))

    async def run():
        async with Client(mcp) as client:
            return await client.read_resource("data://docs")

    with pytest.raises(Exception) as excinfo:
        _run(run())
    assert "Authorization is unavailable" in str(excinfo.value)
    assert "pdp.corp.internal" not in str(excinfo.value)


def test_an_invalid_elicitation_answer_is_a_rejection():
    from axiomatics_ai.core.pdp import ApprovalRequest

    class _Ctx:
        async def elicit(self, message, response_type):
            raise ValueError("'APPROVE' is not one of ['approve', 'reject']")

    request = ApprovalRequest(request_id="r", tool_name="pay", args={}, args_hash="h", reasons=(),
                              ttl_seconds=None, subject={}, asked_by=())
    answer = asyncio.run(fmw_module._ask_by_elicitation(_Ctx(), request))
    assert answer.approved is False and answer.note == "invalid elicitation answer"
