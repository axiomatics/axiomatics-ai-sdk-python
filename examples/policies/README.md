# Example policies

The authorization policies every example and integration test runs against, one file per
scenario. Each `domain_*.yaml` is a self-contained **Axiomatics ADS domain**: the attribute
declarations the policy needs plus the policy itself, ready for ADS to load. The example READMEs
say which domain each script expects.

## Which policy language, which PDP

We wrote these policies in **XACML 3.0** (the XML inside each domain file) because it is the
open OASIS standard, it is what the SDK's XACML JSON requests are evaluated against, and it
compiles from **ALFA**, the more readable authoring language the docs use for illustration. That
is our choice for the examples, not a requirement of the SDK.

The SDK only sends standard requests (XACML 3.0 JSON or OpenID AuthZEN) and reads standard
decisions. Anything behind that boundary is yours to choose:

- **Any policy language** your PDP evaluates: XACML written by hand or compiled from ALFA,
  Rego, Cedar, YAML rule sets, or a vendor DSL. The SDK never sees the policy, only the decision.
- **Any PDP** that speaks XACML 3.0 JSON or AuthZEN. Point `PDPConnection` at it and, if it
  mounts the endpoints at other paths, set `PdpEndpoints`. One exception: retrieval filtering for
  RAG needs partial evaluation, which today only Axiomatics ADS provides.

What must match between your policy and the SDK is the **attribute contract**, not the language:
the action names (`tool_visibility`, `tool_invocation`, `post_invocation`, ...), the resource
attribute ids (`tool_name`, `agent_name`, tool arguments, record fields), and the subject attributes
your context provider sends. [docs/enforcement-points.md](../../docs/enforcement-points.md)
lists exactly what arrives at each point so you can express the same rules in whatever language
your PDP takes.

## Files

| File | Scenario |
|---|---|
| `domain_all_authz_enabled.yaml` | tool visibility, tool invocation, and record filtering together; used by most agent examples and the direct example |
| `domain_tool_visibility_only.yaml` | tool visibility only |
| `domain_tool_invocation.yaml` | tool invocation with argument-level rules |
| `domain_post_invocation.yaml` | record filtering |
| `domain_tool_decorator_masking.yaml` | column masking via `@protected_tool` |
| `domain_custom_context_provider.yaml` | custom subject and environment attributes |
| `domain_mcp_tool_visibility.yaml` | langchain_mcp examples |
| `domain_mcp_custom_attrs.yaml` | custom action and resource attribute ids |
| `domain_llm_content_guard.yaml` | input topic guard and output PII guard, subject-aware |
| `domain_intent.yaml` | intent-based access control: tool calls must match the user's classified intent |
| `domain_human_approval.yaml` | human approval: large payments (Deny + advice) and cancellations (Permit + obligation) need a human's approval |
| `domain_agent_identity.yaml` | verified agent identity (RFC 9068) and an approved RAR grant (RFC 9396): user, agent, scopes, amount and creditor |
| `domain_rag_retrieval.yaml` | retrieval filtering (needs the CAQ licence) |
| `domain_a2a.yaml` | Agent2Agent invocation |
| `domain_gateway.yaml` | the gateway example (MCP and A2A routes) |
| `domain_list_attribute.yaml` | multi-valued attributes |
| `domain_simple_ui.yaml`, `domain_simple_ui_basic.yaml` | the Streamlit demo |
| `deployment.yaml` | ADS server settings; `domain.path` selects which domain is loaded |

The ADS jar and licence files also live here so ADS can be started from this directory; they are
git-ignored. How to run: [docs/running-ads.md](../../docs/running-ads.md).

## Adding a policy for another PDP

Put it in a subfolder named after the PDP with a short README saying how to load it and which
`PdpEndpoints` to set. Keep the attribute contract above so the same example scripts run unchanged.
The [direct example](../direct/authorizer_basic.py) is a convenient smoke test: it exercises a
batch decision, argument-level single decisions, and partial evaluation against whatever
`PDP_BASE_URL` points at.
