# Content guards

The input (`BEFORE_MODEL`) and output (`AFTER_MODEL`) points guard the model's I/O. They differ
from the deterministic points in where the attributes come from: an **LLM enricher** classifies
the user prompt or the final answer into attributes (for example `injection_detected`,
`contains_ssn`), and the PDP decides on those attributes like on any others. The enricher is a
Policy Information Point; it never decides. Only real user prompts and the model's text are
inspected — including text the model sends alongside a tool call, since it reaches the user too
— while reasoning traces are skipped. On a turn that carries a tool call, a block replaces the
text and keeps the call, which `BEFORE_TOOL` still authorizes.

**Streaming.** The output guard checks a complete response. When an app streams tokens to the
user as they are generated (LangChain `stream`, ADK streaming partials, Strands and OpenAI
streamed runs), they have been shown before the guard decides, and a block only replaces the
response kept in the conversation. Don't stream to the user where the output guard must be able
to withhold an answer.

Requires the `llm-enrichment` extra.

## Configure

An enricher is an `EnricherSpec`: a pydantic verdict schema whose **field names become the
attribute ids**, plus classification instructions and a LiteLLM model string. Wrap it in
`LLMAttributeEnricher` and attach one or several to the point's `Enforcer`:

```python
from axiomatics_ai import AuthZConfig, Enforcer, EnforcementPoint, PDPConnection
from axiomatics_ai import prompt_injection_enricher, pii_enricher   # ready-made presets

AuthZConfig(
    connection=PDPConnection(base_url="http://localhost:8080"),
    enforcement={
        EnforcementPoint.BEFORE_MODEL: Enforcer(
            action_name="before_model",
            enrichers=[prompt_injection_enricher(model="gemini/gemini-3.8-flash")],
            block_message="Your request was blocked by policy.",   # optional; per-point default otherwise
            on_unavailable="fail_closed",                          # enricher can't classify: block (default) | "fail_open"
        ),
        EnforcementPoint.AFTER_MODEL: Enforcer(
            action_name="after_model",
            enrichers=[pii_enricher()],   # model from AXIOMATICS_ENRICHER_MODEL
        ),
    },
)
```

**Presets.** `prompt_injection_enricher` (→ `injection_detected`, `attack_type`),
`pii_enricher` (→ `contains_ssn`, `contains_credit_card`), `toxicity_enricher` (→ `toxic`,
`severity`). For domain-specific checks write an `EnricherSpec` with any pydantic schema:
booleans, strings, and ints are supported, and `Field(description=...)` on each field guides the
classifier.

**Model.** Pass `model=` per enricher as a fully qualified LiteLLM string
(`openai/gpt-5.6-luna`, `gemini/gemini-3.8-flash`, `ollama/...` for a local model), or set
`AXIOMATICS_ENRICHER_MODEL` once to use one LLM for all of them. The SDK never picks a default
model.

**Availability.** If the enricher cannot classify (LLM error), it still sends a
`<name>_available=false` sentinel so a policy can tell "checked clean" from "could not check".
`on_unavailable="fail_closed"` (the default) is a shortcut that blocks without calling the PDP;
`"fail_open"` falls through to the PDP with the sentinel set.

**Caching.** `EnricherSpec(cache_ttl_seconds=...)` reuses a successful classification for
identical content for that long (`0`, the default, turns it off; `cache_maxsize` bounds the
entries, default 1024). Failures are never cached.

**Placement.** `EnricherSpec.allowed_points` names the points an enricher is meant for; the
presets set it (`prompt_injection_enricher` → `BEFORE_MODEL`, `pii_enricher` → `AFTER_MODEL`,
`toxicity_enricher` → both, `intent_enricher` → `BEFORE_TOOL`). `config.validate()` warns when
an enricher sits at a point outside its `allowed_points`, or at a point that never runs
enrichers (`TOOL_VISIBILITY`, `AFTER_TOOL`, `A2A_INVOCATION`). Enrichers at `BEFORE_TOOL`
classify the user's messages instead of a prompt or answer; see
[Intent-based access control](intent-based-access.md).

## What the PDP receives

One request per model turn (input) or final answer (output):

- **Subject**: from the context provider
- **Action**: `{"name": "before_model"}` or `{"name": "after_model"}` (the Enforcer's `action_name`)
- **Resource**: every enricher attribute and each `<name>_available` sentinel, with no identifier (the action already names the point)
- **Environment**: from the context provider

Example **XACML JSON** request (`protocol="xacml"`), `before_model` for `alice` after the
prompt-injection preset ran. Over AuthZEN (`protocol="authzen"`) the same attributes travel as
`subject.properties`, `action.name`, and `resource.properties` of one evaluation request.

```json
{
  "Request": {
    "AccessSubject": { "Attribute": [
      { "AttributeId": "username", "DataType": "http://www.w3.org/2001/XMLSchema#string", "Value": ["alice"] }
    ]},
    "Action": { "Attribute": [
      { "AttributeId": "name", "DataType": "http://www.w3.org/2001/XMLSchema#string", "Value": ["before_model"] }
    ]},
    "Resource": { "Attribute": [
      { "AttributeId": "injection_detected", "DataType": "http://www.w3.org/2001/XMLSchema#boolean", "Value": [true] },
      { "AttributeId": "attack_type", "DataType": "http://www.w3.org/2001/XMLSchema#string", "Value": ["instruction_override"] },
      { "AttributeId": "prompt_injection_available", "DataType": "http://www.w3.org/2001/XMLSchema#boolean", "Value": [true] }
    ]}
  }
}
```

Attribute datatypes follow the pydantic schema.

## Writing the policy

Deny a flagged prompt, with first-applicable combining so the deny fires before a catch-all
permit. Because the subject is in the same request, the policy can combine *what was said* with
*who said it*: exempt an executive from a restricted-topic block, or let HR receive an answer
that contains an SSN.

ALFA:

```alfa
namespace contentguard {
    attribute username          { id = "username"           type = string  category = subjectCat }
    attribute actionName        { id = "name"               type = string  category = actionCat }
    attribute injectionDetected { id = "injection_detected" type = boolean category = resourceCat }
    attribute containsSsn       { id = "contains_ssn"       type = boolean category = resourceCat }

    policy inputGuard {
        target clause actionName == "before_model"
        apply firstApplicable

        rule permitExecutive {
            target clause username == "alice"
            permit
        }

        rule denyInjection {
            target clause injectionDetected == true
            deny
        }

        rule permitRest {
            permit
        }
    }

    policy outputGuard {
        target clause actionName == "after_model"
        apply firstApplicable

        rule permitHr {
            target clause username == "alice"
            permit
        }

        rule denySsn {
            target clause containsSsn == true
            deny
        }

        rule permitRest {
            permit
        }
    }
}
```

XACML 3.0 XML, the deny rule only:

```xml
<xacml3:Rule RuleId="deny-prompt-injection" Effect="Deny">
  <xacml3:Target><xacml3:AnyOf><xacml3:AllOf>
    <xacml3:Match MatchId="urn:oasis:names:tc:xacml:1.0:function:boolean-equal">
      <xacml3:AttributeValue DataType="http://www.w3.org/2001/XMLSchema#boolean">true</xacml3:AttributeValue>
      <xacml3:AttributeDesignator MustBePresent="false"
          Category="urn:oasis:names:tc:xacml:3.0:attribute-category:resource"
          AttributeId="injection_detected" DataType="http://www.w3.org/2001/XMLSchema#boolean"/>
    </xacml3:Match>
  </xacml3:AllOf></xacml3:AnyOf></xacml3:Target>
</xacml3:Rule>
```

`examples/policies/domain_llm_content_guard.yaml` is a complete domain: a topic guard on input and
an SSN / credit-card guard on output, both with a per-user exemption.

## Per-framework behaviour on a block

| Path | Input guard | Output guard |
|---|---|---|
| `langchain_agents`, `google_adk`, `strands` | refusal substituted in-band; model never called | answer replaced by the refusal |
| `openai_agents` | SDK guardrail: raises `InputGuardrailTripwireTriggered` (read `output_info["reason"]`) | raises `OutputGuardrailTripwireTriggered` |
| `claude_agent` | hard block, no model-authored refusal | not available (the SDK exposes no hook that sees the final answer) |
| server paths, `langchain_mcp`, `rag`, `a2a` | not applicable | not applicable |

## Observability

Every enricher run emits an `EnrichmentEvent` (point, enricher name, content inspected,
attributes produced, availability, latency) to the configured [hooks](hooks.md).

---

← Previous: [Enforcement points](enforcement-points.md) · [Docs index](README.md) · Next: [Intent-based access control](intent-based-access.md) →
