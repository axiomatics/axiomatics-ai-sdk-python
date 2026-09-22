# Enforcement points

This page is for whoever writes the policy. For each point where the SDK can intercept, it
shows when the point fires and exactly what the request to the PDP contains. The request shapes
are the same on every integration, so one policy is portable across frameworks.

Complete, runnable policies for every point are in [`examples/policies/`](../examples/policies/); the
[Content guards](content-guards.md) and [Post-invocation](post-invocation.md) pages each show
one worked policy in ALFA and XACML.

## Overview

| Point | Fires | Request | Default action name | Resource identifies |
|---|---|---|---|---|
| Input guard (`BEFORE_MODEL`) | before the model sees the prompt | single, LLM-enriched | `before_model` | nothing (the action names the point); classifier attributes only |
| Tool visibility (`TOOL_VISIBILITY`) | before the model sees the tool list | batch, one sub-request per tool | `tool_visibility` | tool name |
| Tool invocation (`BEFORE_TOOL`) | before a tool call runs | single, optionally intent-enriched | `tool_invocation` | tool name plus call arguments |
| Post-invocation (`AFTER_TOOL`) | after a tool returns | batch per record, or single | `post_invocation` | record fields, or tool name plus call arguments |
| Output guard (`AFTER_MODEL`) | before the answer reaches the user | single, LLM-enriched | `after_model` | nothing (the action names the point); classifier attributes only |
| A2A invocation (`A2A_INVOCATION`) | on an agent-to-agent message | single | `a2a_invocation` | target agent name, RPC method, skill (and optionally the message text) |
| Retrieval (`RetrievalConfig`) | on a vector-store search | partial evaluation (reverse query) | `retrieval` | left unbound; the PDP returns the condition |
| Direct (`Authorizer`) | wherever your code calls it | single or batch | your choice | your choice |

Action names are the `Enforcer.action_name` values in your `AuthZConfig`; change them there and
in the policy together. The examples below use the defaults and the default context provider
(subject attribute `username`).

## Wire formats

Every example request below is shown as **XACML JSON** (`PDPConnection(protocol="xacml")`, the
default). A *single* request is one `Request` with four categories; a *batch* request uses the
XACML Multiple Decision Profile: shared subject and action, one `Resource` entry per item, and
only the identifier (or correlation key) attribute marked `IncludeInResult` so the SDK can match
each decision back. Single requests never set `IncludeInResult`.

With `protocol="authzen"` the same attributes are sent as an **AuthZEN** request instead:
`subject.properties`, `action.name`, `resource.properties`, and `context` for the environment. A
batch becomes one `evaluations` request with one entry per item, and decisions come back in
order. The policy sees the same attribute ids either way.

## Attribute conventions

| Category | Comes from | Default contents |
|---|---|---|
| subject | context provider `get_subject` | `username`, `user_id`, `tenant_id`, `agent_id` (client); every JWT claim, or with `AgentIdentityContextProvider` `user_id`, `agent_id`, `actor_id`, `scopes`, ... (server); plus `agent_name`, the agent in the app making the call ([Sub-agents](#sub-agents)) |
| action | context provider `get_action` | `name` = the action name |
| resource | context provider `get_resource` | `tool_name` or `agent_name` (the identifier), plus call arguments or record fields |
| environment | context provider `get_environment` | `session_id` (client) or nothing (server); at tool invocation also any [intent attributes](intent-based-access.md) |

The resource identifier depends on the point:

| Point | Resource identifier | Configure with |
|---|---|---|
| tool visibility, tool invocation, post-invocation (`ColumnMaskStrategy`) | `tool_name` | `AuthZConfig.attribute_names.tool` |
| post-invocation (`RecordFilterStrategy`) | `record_hash`, an opaque correlation key | `RecordFilterStrategy(key_attribute=...)` |
| A2A invocation | `agent_name` | `AuthZConfig.attribute_names.agent` |
| input / output guards | none | |
| direct (`Authorizer`) | `resource_name` | `resource_name_attribute=` on each call |

The naming rule behind these defaults: ids use underscores (no dots), each identifier is named
after what it is (`tool_name`, `agent_name`, `resource_name`), and the resource never uses a bare
`name`, because the action already does (and ADS's AuthZEN mapping maps `action.name` onto it).
Every default is configurable. `attribute_names.tool` is one setting for all three tool points,
so the same policy attribute identifies the tool everywhere.

### Sub-agents

A sub-agent gets the same config as the agent you authorize, with no extra wiring:

| Framework | Covered |
|---|---|
| deepagents (`create_authorized_deep_agent`) | each sub-agent in `subagents`, and the general-purpose one |
| Google ADK (`authorized_adk_agent`) | `sub_agents` (also through `SequentialAgent` and other workflow agents) and the agent an `AgentTool` wraps |
| OpenAI Agents (`authorized_openai_agent`) | handoff targets, and agents used as tools (`agent.as_tool()`) |
| Strands (`authorized_strands_agent`) | agents used as tools (`agent.as_tool()`) |

The sub-agent's calls carry the caller's identity, and the subject names the agent making the
call in `agent_name` (`AuthZConfig.attribute_names.acting_agent`): the agent's `name`, or a
deepagents sub-agent's. One policy can therefore treat sub-agents differently:

```alfa
rule researcherOnlySearches {
    target clause agentName == "researcher"
    condition not(toolName == "search_docs")
    deny
}
```

`agent_name` is a name from your code, not an authenticated identity: it says which part of your
app is acting, under the caller's identity. An agent's own identity is `agent_id`, and the current
actor in a delegation chain is `actor_id` ([Agent identity](agent-identity.md)). It is added only
when there is a caller identity, and not over a value your context provider set. Turn it off with
`AttributeNames(acting_agent=None)`.

To give a sub-agent a different config, authorize it first with that config; the parent call
leaves an agent already authorized as it is. On deepagents, put your own
`LangChainAgentAuthZMiddleware(other_config)` in the sub-agent spec's `middleware`.

The tool that runs a sub-agent (deepagents' `task`, an ADK `AgentTool`, an `as_tool()` agent) is
authorized like any tool call, but its output, the sub-agent's answer, is not post-processed:
the sub-agent's own tool results already were.

Not covered: a deepagents compiled sub-agent (`{"runnable": ...}`) must be built with
`create_authorized_agent` or `create_authorized_deep_agent` (otherwise `ValueError`); a remote
sub-agent (`{"graph_id": ...}`, an A2A agent) runs elsewhere and enforces there; an OpenAI
`Handoff` built by hand names no agent and gets a warning; a cloned OpenAI agent
(`agent.clone()`) needs its own call.

Tool arguments are merged into the resource next to the identifier. An argument whose name
equals the identifier attribute is dropped with a warning: the identifier always wins, so a
model-chosen argument cannot impersonate another tool.

Values keep their Python type: `str`, `int`, `float`, `bool`, and lists of those become the
matching XACML datatypes (or JSON types over AuthZEN). Objects, nested lists and mixed-type
lists are dropped with one warning per attribute, whether they come from arguments, claims or
records ([Agent identity](agent-identity.md#values-the-pdp-cannot-carry)). See
[Identity](identity.md) to change any of this.

## 1. Tool visibility

One batch request per model turn, one sub-request per tool. Tools the PDP denies (or, with
`fail_open=False`, leaves undecided) are removed before the model sees the list.

XACML JSON, Multiple Decision Profile:

```json
{
  "Request": {
    "MultiRequests": { "RequestReference": [
      { "ReferenceId": ["subject", "action", "search_docs"] },
      { "ReferenceId": ["subject", "action", "get_account_balance"] }
    ]},
    "AccessSubject": [{ "Id": "subject", "Attribute": [{ "AttributeId": "username", "Value": ["alice"] }] }],
    "Action":        [{ "Id": "action",  "Attribute": [{ "AttributeId": "name", "Value": ["tool_visibility"] }] }],
    "Resource": [
      { "Id": "search_docs",         "Attribute": [{ "AttributeId": "tool_name", "Value": ["search_docs"], "IncludeInResult": true }] },
      { "Id": "get_account_balance", "Attribute": [{ "AttributeId": "tool_name", "Value": ["get_account_balance"], "IncludeInResult": true }] }
    ]
  }
}
```

| Category | Attribute | Example |
|---|---|---|
| subject | `username` (or yours) | `alice` |
| action | `name` | `tool_visibility` |
| resource | `tool_name` (`AuthZConfig.attribute_names.tool`) | `search_docs` |

**Tool names** are the bare tool name (`search_docs`) on every path, so policies are portable.
The `claude_agent` path receives namespaced MCP names (`mcp__<server>__<tool>`); it sends the
bare name as `tool_name` and the server as `mcp_server` (`AuthZConfig.attribute_names.mcp_server`).

## 2. Tool invocation

One request per tool call, before it runs. The tool's arguments are forwarded as resource
attributes, so a rule can decide on *which* account, *which* file, and so on.

XACML JSON, single request:

```json
{
  "Request": {
    "Category": [
      { "CategoryId": "urn:oasis:names:tc:xacml:1.0:subject-category:access-subject",
        "Attribute": [{ "AttributeId": "username", "Value": ["bob"] }] },
      { "CategoryId": "urn:oasis:names:tc:xacml:3.0:attribute-category:action",
        "Attribute": [{ "AttributeId": "name", "Value": ["tool_invocation"] }] },
      { "CategoryId": "urn:oasis:names:tc:xacml:3.0:attribute-category:resource",
        "Attribute": [
          { "AttributeId": "tool_name",  "Value": ["get_account_balance"] },
          { "AttributeId": "account_id", "Value": ["ACC-001"] } ] }
    ]
  }
}
```

The same request over AuthZEN:

```json
{
  "subject":  { "type": "subject",  "id": "subject",  "properties": { "username": "bob" } },
  "action":   { "name": "tool_invocation", "properties": {} },
  "resource": { "type": "resource", "id": "resource", "properties": { "tool_name": "get_account_balance", "account_id": "ACC-001" } }
}
```

| Category | Attribute | Example |
|---|---|---|
| subject | `username` (or yours) | `bob` |
| action | `name` | `tool_invocation` |
| resource | `tool_name` | `get_account_balance` |
| resource | one attribute per tool argument | `account_id` = `ACC-001` |

A rule that lets bob call `get_account_balance` only for his own account matches on the subject
and both resource values. In ALFA:

```alfa
rule bobOwnAccount {
    target clause username == "bob"
             and toolName == "get_account_balance"
             and accountId == "ACC-001"
    permit
}
```

In XACML 3.0 XML that is three `Match` elements inside one `AllOf`; see
`examples/policies/domain_tool_invocation.yaml`.

**Intent attributes.** When the `BEFORE_TOOL` Enforcer has an `intent_enricher`, the
environment also carries what the user asked for, classified from their own messages:

| Category | Attribute | Example |
|---|---|---|
| environment | `intent_actions` | `["search", "send"]` |
| environment | `intent_<action>_recipients`, one per configured action | `intent_send_recipients` = `["bob@corp.com"]` |
| environment | `intent_available` | `true` |

They override a same-named attribute from the context provider. Policy and limits on the
[Intent-based access control](intent-based-access.md) page.

## 3. Post-invocation

What the PDP receives depends on the strategy. With the default `RecordFilterStrategy` the tool
result is a list of records and the SDK sends one batch request with one sub-request per
record; each record's fields are resource attributes and denied records are dropped. With
`ColumnMaskStrategy` it is one single request for the whole result, with the same resource as
tool invocation (`tool_name` plus arguments), and on Deny the named fields are masked.

With `RecordFilterStrategy`, per sub-request:

| Category | Attribute | Example |
|---|---|---|
| subject | `username` (or yours) | `bob` |
| action | `name` | `post_invocation` |
| resource | one attribute per record field | `doc_type` = `classified` |
| resource | `record_hash` (`RecordFilterStrategy(key_attribute=...)`) | correlation only; never match on it |

Record sub-requests do not carry `tool_name`; use `action_overrides` on the `AFTER_TOOL`
Enforcer to send a different action name per tool.

Request and a worked policy on the [Post-invocation](post-invocation.md) page.

## 4. Input and output guards

One single request per prompt (`before_model`) or final answer (`after_model`). The resource
carries only the attributes an LLM enricher produced plus a `<enricher>_available` sentinel;
it has no identifier, because the action already names the point.

| Category | Attribute | Example |
|---|---|---|
| subject | `username` (or yours) | `alice` |
| action | `name` | `before_model` / `after_model` |
| resource | one attribute per enricher schema field | `injection_detected` = `true`, `attack_type` = `instruction_override` |
| resource | `<enricher>_available` | `prompt_injection_available` = `true` |

Request, presets, and a worked policy on the [Content guards](content-guards.md) page.

## 5. A2A invocation

One single request per inbound JSON-RPC call (server middleware: every method, `tasks/get`
included) or outbound `message/send` (client guard).

| Category | Attribute | Example |
|---|---|---|
| subject | JWT claims (server) or context (client) | `username` = `alice` |
| action | `name` | `a2a_invocation` |
| resource | `agent_name` (`AuthZConfig.attribute_names.agent`) | the `agent_name` you pass when attaching, e.g. `billing-agent` |
| resource | `method` | `message/send`, `tasks/get` |
| resource | `skill` | from the message metadata, when present, e.g. `refund` |
| resource | `message` | the message text; sent only with `include_message_text=True` |

`agent_name` is set explicitly when you attach the middleware or client guard, so the
policy-facing name never depends on a URL. Domain: `examples/policies/domain_a2a.yaml`.

## 6. Retrieval

Retrieval does not ask for a decision. The SDK sends the subject, action (`retrieval`), and
environment as *provided* attributes and leaves the document attributes unbound; the PDP
returns the condition on the unbound attributes under which the decision is Permit. The SDK
translates that condition into the vector store's native metadata filter.

This is a partial-evaluation request (ADS's CAQ `providedAttributes` body), not XACML JSON or
AuthZEN, and it goes to `PdpEndpoints.partial_evaluation` regardless of `protocol`:

```json
{
  "providedAttributes": [
    { "attribute": "department", "values": ["engineering"] },
    { "attribute": "level",      "values": [2] },
    { "attribute": "name",       "values": ["retrieval"] }
  ],
  "mockedAttributes": [],
  "pdpDecisionSet": ["PERMIT"],
  "excludeIndeterminate": false,
  "contextType": "NONE"
}
```

| Provided | Attribute | Example |
|---|---|---|
| subject | whatever your provider sends | `department` = `engineering`, `level` = `2` |
| action | `name` | `retrieval` |
| environment | whatever your provider sends | |

| Unbound (the policy references them; the PDP returns a condition on them) | Example |
|---|---|
| document metadata attributes | `resource.department`, `resource.level` |

Write the rule as an ordinary condition over subject and document attributes, for example
"readable when the document is public or in the reader's department, and its level is at most
the reader's". For alice above the PDP returns, in effect,
`(resource.department == "engineering" or resource.department == "public") and resource.level <= 2`,
which becomes the store filter. Metadata keys are the resource attribute ids with a
`resource.` or `doc.` prefix stripped, or whatever `RetrievalConfig.attribute_map` says.

Retrieval needs a PDP with partial evaluation; see [PDP compatibility](configuration.md#pdp-compatibility)
and the [rag](integrations/rag.md) page. Domain: `examples/policies/domain_rag_retrieval.yaml`.

## 7. Direct

`Authorizer.is_permitted` sends a single request and `is_permitted_batch` a batch request with
exactly the shapes above; you choose the action name, resource name, and arguments. The
resource name is sent as `resource_name` unless you pass `resource_name_attribute=`. Write the
policy the same way as for tool invocation or tool visibility. See [direct](integrations/direct.md).

## NotApplicable, Indeterminate, and `fail_open`

When no rule matches, the PDP returns NotApplicable; when evaluation errors, Indeterminate. The
SDK treats both as "no Permit": denied under the default `fail_open=False`, permitted under
`fail_open=True`. The same switch governs a PDP that is unreachable or times out. Over AuthZEN
the standard `decision` boolean carries only Permit or Deny, so `fail_open` matters there only
when the PDP is unreachable or, like ADS, echoes the underlying XACML decision. A request the
PDP rejects outright (HTTP 4xx) raises `PdpRequestError` regardless; see
[Troubleshooting](troubleshooting.md).

---

← Previous: [Agent identity](agent-identity.md) · [Docs index](README.md) · Next: [Content guards](content-guards.md) →
