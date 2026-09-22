# Running Axiomatics ADS

Axiomatics Access Decision Service (ADS) is the PDP the SDK is developed and tested against. It
serves XACML 3.0 JSON and, from 26.1.6 with `authzen.enabled: true`, OpenID AuthZEN, and it is
the only PDP offering the partial evaluation the RAG path needs. The examples and integration
tests ship with ADS domain files under [`examples/policies/`](../examples/policies/).

If you run another PDP, skip this page and set `PDPConnection(base_url=..., endpoints=...)` as
described in [Configuration](configuration.md#endpoints).

## Files you need

In one directory (the examples use `examples/policies/`):

| File | What |
|---|---|
| `access-decision-service-<version>.jar` | the PDP; get it from Axiomatics |
| `axiomatics_ADS.license` or `axiomatics_ADS_CAQ.license` | your licence; the CAQ licence is required for retrieval filtering |
| `deployment.yaml` | server settings and which domain to load |
| `domain_<name>.yaml` | the policy domain: attribute declarations plus policies |

The jar and licence files are git-ignored; the domain files are committed.

## `deployment.yaml`

```yaml
license: file:./axiomatics_ADS_CAQ.license   # or axiomatics_ADS.license

engine:
  mode: ads_caq                                # ads_caq enables partial evaluation

domain:
  path: ./domain_all_authz_enabled.yaml        # one domain at a time — change per example
  refreshInterval: 30 seconds

server:
  port: 8080

logging:
  level.com.axiomatics.audit: info

authentication:
  enabled: false                               # local development only

authzen:
  enabled: true                                # exposes /access/v1/evaluation(s); optional

identity:
  constant: ads-test-local
```

## Start and check

```bash
cd examples/policies
java -jar access-decision-service-<version>.jar
# in another shell
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/health   # 200 when ready
```

ADS resolves `deployment.yaml` relative to the working directory. It serves one domain at a
time: stop it and change `domain.path` to switch examples. `examples/ads_runner.py` does the
patch-start-wait-stop cycle for you from a script, and the integration tests do the same per
test class.

## Domain files for the examples

| Domain | Used by |
|---|---|
| `domain_all_authz_enabled.yaml` | all three tool points: langchain_agents, fastmcp, direct, ADK, Strands, OpenAI, Claude examples |
| `domain_tool_visibility_only.yaml` | tool visibility only |
| `domain_tool_invocation.yaml` | tool invocation with argument-level rules |
| `domain_post_invocation.yaml` | record filtering |
| `domain_tool_decorator_masking.yaml` | column masking via `@protected_tool` |
| `domain_custom_context_provider.yaml` | custom subject and environment attributes |
| `domain_mcp_tool_visibility.yaml` | langchain_mcp examples |
| `domain_mcp_custom_attrs.yaml` | custom action and resource attribute ids |
| `domain_llm_content_guard.yaml` | input topic guard and output PII guard, subject-aware |
| `domain_human_approval.yaml` | human approval of tool calls (advice and obligation forms) |
| `domain_rag_retrieval.yaml` | retrieval filtering (needs the CAQ licence) |
| `domain_a2a.yaml` | Agent2Agent invocation |
| `domain_gateway.yaml` | the gateway example (MCP and A2A routes) |
| `domain_list_attribute.yaml` | multi-valued attributes |
| `domain_intent.yaml` | intent-based access control |
| `domain_agent_identity.yaml` | agent identity (verified tokens, delegation) |
| `domain_simple_ui.yaml`, `domain_simple_ui_basic.yaml` | the Streamlit demo |

## Declaring attributes

Every attribute a policy references must be declared in the domain's `attributes` block:

```yaml
attributes:
  username:
    xacmlId: username
    category: AccessSubject
    datatype: string
  name:
    xacmlId: name
    category: Action
    datatype: string
  tool_name:
    xacmlId: tool_name
    category: Resource
    datatype: string
```

Over XACML, attributes the SDK sends but the domain does not declare are ignored. Over AuthZEN,
ADS maps attributes strictly and rejects undeclared ones with HTTP 400, which the SDK raises as
`PdpRequestError`.

Full ADS documentation: [docs.axiomatics.com/access-decision-service](https://docs.axiomatics.com/access-decision-service/).

[Docs index](README.md)
