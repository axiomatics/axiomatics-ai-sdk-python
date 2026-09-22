# Post-invocation

Post-invocation (`AFTER_TOOL`) transforms a tool's result after it runs and before the model
sees it. A **strategy** receives the parsed result and a `PostInvocationContext`, asks the PDP,
and returns the possibly transformed result. The strategy owns both the PDP call and the
transformation.

## Built-in strategies

| Strategy | PDP call | On Deny |
|---|---|---|
| `RecordFilterStrategy()` (default) | one batch request, one sub-request per record | denied records removed |
| `ColumnMaskStrategy("f1", "f2")` | one request for the whole result | named fields replaced with `***`, at any depth |

Set the strategy on the `AFTER_TOOL` Enforcer for the middleware paths, or pass it as
`post_processor=` on `@protected_tool`:

```python
from axiomatics_ai import AuthZConfig, ColumnMaskStrategy, EnforcementPoint, Enforcer, protected_tool

# middleware paths (langchain_agents, fastmcp, google_adk, strands, openai_agents, claude_agent)
config = AuthZConfig(enforcement={
    EnforcementPoint.AFTER_TOOL: Enforcer(
        action_name="post_invocation",
        strategy=ColumnMaskStrategy("ssn", "dob"),
    ),
})

# tool_decorator
@protected_tool(post_processor=ColumnMaskStrategy("ssn", "dob"))
def get_employee() -> dict: ...
```

`action_overrides={"tool_name": "other_action"}` on the Enforcer sends a different action name
for a specific tool.

## What the strategy sees

Each framework wraps a tool's output differently: a JSON string, an MCP result
(`structuredContent` plus text blocks), content blocks (a Strands `ToolResult`, a LangChain
`ToolMessage` from an MCP tool), or `{"result": ...}`. The SDK strips that wrapper, hands the
strategy the tool's data (a list or a dict), and rebuilds the result around what the strategy
returns. For an MCP result, `structuredContent` and the text blocks are rewritten together, so a
client reading either never sees a record the policy removed.

`RecordFilterStrategy` reads that data as records:

| Data | Treated as |
|---|---|
| a list | each dict in it is a record; each list in it is read the same way; other items (`null`, strings) are kept |
| a dict holding lists, at any depth of its dict values, e.g. `{"data": {"results": [...]}, "count": 3}` | every such list is filtered; other keys are kept |
| any other dict, e.g. `{"name": "Bob", "ssn": "..."}` | a single record, replaced by `{}` when denied |

All the records of one result go to the PDP in one batched request. A record's own fields are
never searched for more records. A tool that returns a dict is therefore filtered too. Its record goes to the PDP like any other,
so a record policy must permit it.

### Output the strategy cannot inspect

Output that holds no list or object — plain text, a scalar, an image — is logged as a warning
and reported to [hooks](hooks.md) as a `skipped` post-invocation decision. What happens next is
the Enforcer's `on_unrecognized_output`:

| Value | Effect |
|---|---|
| `"fail_open"` (default) | the output is returned unchanged |
| `"fail_closed"` | the output is withheld; the model gets a short message saying so, and the decision event is `deny` |

```python
Enforcer(action_name="post_invocation", strategy=RecordFilterStrategy(),
         on_unrecognized_output="fail_closed")
```

The same applies to a content block next to the data: when a tool returns its records in one
block and, say, a plain-text summary in another, the summary can restate what the strategy just
removed. It is logged and reported as `skipped` too; under `fail_closed` it is replaced by a
text block saying it was withheld, while the filtered records are kept.

`fail_open` is the default because `standard_tool_enforcement()` puts one `AFTER_TOOL` Enforcer
on every tool, including tools that legitimately return text ("Email sent"). Use `fail_closed`
when a tool that renders records as prose must not bypass the policy.

## Record filtering, in detail

`RecordFilterStrategy` sends one sub-request per record. Each record's fields become resource
attributes (sent flat), plus a stable `record_hash` (`RecordFilterStrategy(key_attribute=...)`)
the SDK uses to match decisions back to records. **Policy authors never match on the hash**;
match on record content such as `doc_type`. Record sub-requests do not carry `tool_name`.

`ColumnMaskStrategy` instead sends one single request with the same resource as tool
invocation: `tool_name` (`AuthZConfig.attribute_names.tool`) plus the call arguments.

The **XACML JSON** batch request (`protocol="xacml"`, Multiple Decision Profile) for two records
looks like this. Over AuthZEN (`protocol="authzen"`) it is one `evaluations` request with one
entry per record, and results are matched back by position instead of by the echoed hash.

```json
{
  "Request": {
    "MultiRequests": { "RequestReference": [
      { "ReferenceId": ["subject", "action", "rec_0_a1b2c3d4e5f6"] },
      { "ReferenceId": ["subject", "action", "rec_1_9f8e7d6c5b4a"] }
    ]},
    "AccessSubject": [{ "Id": "subject", "Attribute": [{ "AttributeId": "username", "Value": "bob" }] }],
    "Action":        [{ "Id": "action",  "Attribute": [{ "AttributeId": "name", "Value": "post_invocation" }] }],
    "Resource": [
      { "Id": "rec_0_a1b2c3d4e5f6", "Attribute": [
        { "AttributeId": "record_hash", "Value": "rec_0_a1b2c3d4e5f6", "IncludeInResult": true },
        { "AttributeId": "doc_type", "Value": "classified" } ]},
      { "Id": "rec_1_9f8e7d6c5b4a", "Attribute": [
        { "AttributeId": "record_hash", "Value": "rec_1_9f8e7d6c5b4a", "IncludeInResult": true },
        { "AttributeId": "doc_type", "Value": "public" } ]}
    ]
  }
}
```

A policy that keeps classified records away from bob, with first-applicable combining so the
deny fires before the catch-all permit. ALFA:

```alfa
namespace records {
    attribute username   { id = "username" type = string category = subjectCat }
    attribute actionName { id = "name"     type = string category = actionCat }
    attribute docType    { id = "doc_type" type = string category = resourceCat }

    policy recordRead {
        target clause actionName == "post_invocation"
        apply firstApplicable

        rule aliceReadsAll {
            target clause username == "alice"
            permit
        }

        rule bobDenyClassified {
            target clause username == "bob" and docType == "classified"
            deny
        }

        rule bobPermitRest {
            target clause username == "bob"
            permit
        }
    }
}
```

The deny rule in XACML 3.0 XML:

```xml
<xacml3:Rule RuleId="bob-deny-classified" Effect="Deny">
  <xacml3:Target><xacml3:AnyOf><xacml3:AllOf>
    <xacml3:Match MatchId="urn:oasis:names:tc:xacml:1.0:function:string-equal">
      <xacml3:AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">bob</xacml3:AttributeValue>
      <xacml3:AttributeDesignator MustBePresent="false"
          Category="urn:oasis:names:tc:xacml:1.0:subject-category:access-subject"
          AttributeId="username" DataType="http://www.w3.org/2001/XMLSchema#string"/>
    </xacml3:Match>
    <xacml3:Match MatchId="urn:oasis:names:tc:xacml:1.0:function:string-equal">
      <xacml3:AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">classified</xacml3:AttributeValue>
      <xacml3:AttributeDesignator MustBePresent="false"
          Category="urn:oasis:names:tc:xacml:3.0:attribute-category:resource"
          AttributeId="doc_type" DataType="http://www.w3.org/2001/XMLSchema#string"/>
    </xacml3:Match>
  </xacml3:AllOf></xacml3:AnyOf></xacml3:Target>
</xacml3:Rule>
```

`examples/policies/domain_post_invocation.yaml` is a complete working domain.

## Custom strategy

Implement `apply(result, context)`. Works on every path that has post-invocation. `result` is
the tool's data with the framework wrapper removed (see [What the strategy sees](#what-the-strategy-sees)):
a list or a dict on the middleware paths, the tool's own return value on `@protected_tool`.

```python
from axiomatics_ai import AuthZConfig, EnforcementPoint, Enforcer, PostInvocationContext

class RedactEmails:
    def apply(self, result, context: PostInvocationContext):
        permitted = context.pdp_client.check_permission(
            subject=context.subject,
            action=context.action,
            resource=context.resource,      # {"tool_name": tool_name, **args} from the context provider
            environment=context.environment,
        )
        if permitted:
            return result
        return redact(result)              # your transformation on Deny

config = AuthZConfig(enforcement={
    EnforcementPoint.AFTER_TOOL: Enforcer(action_name="post_invocation", strategy=RedactEmails()),
})
```

`PostInvocationContext` fields:

| Field | Type | Meaning |
|---|---|---|
| `pdp_client` | `PdpClient` | configured client; `check_permission` for one request, `check_permission_batch` for many |
| `subject` | `dict` | subject attributes from the context provider |
| `environment` | `dict` | environment attributes from the context provider |
| `tool_name` | `str` | the tool that ran |
| `args` | `dict` | its call arguments |
| `action` | `dict` | action attributes, e.g. `{"name": "post_invocation"}` |
| `action_name` | `str` | shortcut for `action["name"]` |
| `resource` | `dict` | built by the context provider exactly as for tool invocation; by default `{"tool_name": tool_name, **args}` |
| `fail_open` | `bool` | what to do when the PDP is unreachable |

---

← Previous: [Obligations and human approval](human-approval.md) · [Docs index](README.md) · Next: [Hooks](hooks.md) →
