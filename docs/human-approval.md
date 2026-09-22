# Obligations, advice and human approval

A PDP can return more than Permit or Deny. XACML lets a policy attach **obligations** (the PEP
must carry them out) and **advice** (the PEP may) to a decision. The SDK reads both, on the
XACML path and on ADS's AuthZEN path, and acts on them at every enforcement point.

The one obligation the SDK understands itself is **human approval**: the policy says "this tool
call needs a human's approval", the agent pauses (or asks an approver), and the answer goes back
to the policy.

> **Preview.** Human approval works on every framework, but its guarantees are in-process: see
> [Known limitations](security.md#known-limitations) for resuming in another process and for
> FastMCP's elicitation.

## Obligations and advice

```python
from axiomatics_ai import AuthZConfig, ObligationsAndAdviceConfig

config = AuthZConfig(
    connection=...,
    enforcement=...,
    obligations_and_advice=ObligationsAndAdviceConfig(
        handlers={
            "log-approval": lambda obligation, ctx: audit.write(obligation.attributes),
            "notify-owner": None,        # handled elsewhere: accept it, run nothing
        },
        on_unknown="deny",               # the default
    ),
)
```

| Setting | Default | Meaning |
|---|---|---|
| `handlers` | `{}` | obligation id → a function `(Obligation, ObligationContext) -> bool \| None`, or `None` to accept the id without running anything |
| `on_unknown` | `"deny"` | what an obligation with no handler does to a Permit: `"deny"` (XACML: a PEP that can't fulfil an obligation must not permit) or `"ignore"` (permit, log each id once) |
| `approval` | `ApprovalConfig()` | human approval, below |

- **On Permit**, every obligation's handler runs before the action goes ahead. A handler that
  returns `False` or raises turns the Permit into a Deny. `ObligationContext` carries the
  request's subject, action, resource and environment.
- **On Deny**, the handlers of the Deny's obligations run too; the decision stays Deny.
- **Advice** is advisory: unknown advice is ignored.
- **Cache hits** still run the handlers: the decision cache stores the PDP's decision with its
  obligations, not the outcome.
- **Every bool path** applies these rules: tool visibility, invocation, post-invocation record
  filtering, the content guards, A2A and the [direct path](integrations/direct.md)'s
  `Authorizer`. `PdpClient.evaluate()` returns the raw decision if you need it.

Attribute values arrive as the PDP sent them. ADS sends every value as a string (`"300"`), and
an attribute assigned more than once becomes a list.

## Human approval

### Two ways to write the policy

A policy asks for approval with the id `require-approval` (configurable, see below), in one of two forms.
They mean different things, and the SDK treats them accordingly:

| Form | What the policy says | After the human approves | PDP calls |
|---|---|---|---|
| **Deny + advice** | "Denied. If a human approves, ask me again." | The SDK evaluates the call again with the environment attribute `approval_granted = true`, and that second decision is final. | 2 |
| **Permit + obligation** | "Permitted, provided a human approves." | The approval fulfils the obligation; the Permit stands. Any other obligations on it are handled as usual. | 1 |

A rejection is a Deny in both forms. Use the advice form when the rule after approval needs
conditions of its own ("approved *and* amount under 10 000"); use the obligation form when
approval is all it takes.

```alfa
rule approvedPayment {
    target clause toolName == "pay_invoice"
    condition approvalGranted == true
    permit
}

rule largePaymentNeedsApproval {
    target clause toolName == "pay_invoice"
    deny
    on deny {
        advice requireApproval {
            reason = "Payments over 100 need a human's approval"
        }
    }
}

rule cancellationNeedsApproval {
    target clause toolName == "cancel_invoice"
    permit
    on permit {
        obligation requireApproval {
            reason = "Cancelling an invoice needs a human's approval"
        }
    }
}
```

Declarations are omitted. The runnable domain is
[`examples/policies/domain_human_approval.yaml`](../examples/policies/domain_human_approval.yaml)
(XACML, loaded by the examples). The policy must stop asking once `approval_granted` is set:
if the second evaluation asks for approval again, the SDK denies.

The obligation or advice may carry two attributes the SDK reads by these fixed names:

| Attribute | Meaning |
|---|---|
| `reason` | shown to the human |
| `ttl` | seconds an approval may be reused for the same caller, tool and arguments; without it, every call is approved on its own. It is read from the decision that asks now: when a rule drops its `ttl`, earlier approvals are no longer reused |

Any other attribute (say `approver_role = "finance"`) reaches the approver untouched, as
`request.asked_by[i].attributes`. `config.invalidate_decisions()` drops remembered approvals and
paused calls too.

The approval attribute (`approval_granted` by default) is reserved: only the SDK sets it, after a
human's approval. If your context provider puts it in the environment, it is dropped (with a
warning, once), so a caller cannot approve its own call.

**Several approval ids, several approvals.** `ApprovalConfig(obligation_or_advice_ids=...)` is a
set: any of its ids asks for approval (e.g. `{"require-approval", "four-eyes"}` when domains use
different names). XACML collects obligations from every matching rule, so one decision can carry
several approval obligations or advice. They make **one** request: `reasons` lists every
reason, `asked_by` every obligation or advice, and `ttl_seconds` is the smallest `ttl` (single
use if any of them has none). **One human answer covers them all**; separate approvals per rule
(for example a manager *and* finance) are not supported yet. An approval id outside the set is an
ordinary obligation: with no handler, it denies.

Approval happens only **before a tool call**. Anywhere else (tool visibility, the content
guards, post-invocation, A2A) there is nothing to pause, so a `require-approval` obligation
can't be fulfilled and is a Deny, and `require-approval` advice is ignored. Put approval rules
on the tool-invocation action, not on tool visibility, or the tool is hidden and never called.

### Who answers

In order: the answer to the framework's own pause (on resume), a remembered approval, then the
configured **approver**:

```python
from axiomatics_ai import ApprovalConfig, ApprovalDecision, ObligationsAndAdviceConfig

async def approver(request) -> ApprovalDecision:
    # request.tool_name, request.args, request.reason, request.subject, request.ttl_seconds
    answer = await ask_on_slack(request)          # or a CIBA backchannel request, a console prompt, ...
    return ApprovalDecision(approved=answer.ok, approver=answer.user, note=answer.comment)

obligations_and_advice = ObligationsAndAdviceConfig(
    approval=ApprovalConfig(
        approver=approver,                # sync or async; optional
        obligation_or_advice_ids={"require-approval"},  # the ids your policies use
        attribute="approval_granted",     # the environment attribute sent after approval
    ),
)
```

The approver runs **inside the tool call** and blocks it until it answers, which is fine for a
chat prompt or a Slack button, and for an out-of-band channel such as CIBA polling. An approver
that raises, or returns anything but an `ApprovalDecision`, is a rejection.

Without an approver, the SDK uses the framework's own pause and resume:

| Framework | How it pauses | How the app resumes |
|---|---|---|
| [LangChain](integrations/langchain_agents.md) | `interrupt()` in the tool call; the run returns `__interrupt__` with the request | `agent.invoke(Command(resume={"approved": True}), config)`. Needs a **checkpointer**. |
| [Google ADK](integrations/google_adk.md) | tool confirmation: an `adk_request_confirmation` function call | a `FunctionResponse` named `adk_request_confirmation`, `{"confirmed": True, "payload": {...}}` |
| [Strands](integrations/strands.md) | an interrupt named `axiomatics-approval`; `stop_reason == "interrupt"` | `agent([{"interruptResponse": {"interruptId": ..., "response": {"approved": True}}}])` |
| [OpenAI Agents](integrations/openai_agents.md) | the SDK's own approval flow: `needs_approval` also asks the PDP, and `result.interruptions` holds the call | `state = result.to_state(); state.approve(item)` / `state.reject(item)`; `Runner.run(agent, state)` |
| [Claude Agent SDK](integrations/claude_agent.md) | none: only the approver can approve (the `PreToolUse` hook waits for it) | — |
| [FastMCP](integrations/fastmcp.md) (server) | asks the MCP client's user with an **elicitation** (approve / reject) | the client's elicitation handler answers. The client must support elicitation. |

The pause carries the request as JSON (`ApprovalRequest.to_payload()`): `type`
(`"axiomatics.approval_request"`), `request_id`, `tool_name`, `args`, `reason`, `reasons`, `ttl_seconds`. It
leaves out the subject, which may hold token claims.

`args` are the tool call's **own arguments**, as the model sent them — not the attributes your
context provider built for the PDP, which may forward only some of them (or add others). An
approval is bound to both: a remembered approval is reused only when the arguments and the PDP
resource are both unchanged. An answer may be `True`/`False`, a dict
`{"approved": bool, "approver": ..., "note": ...}`, or an `ApprovalDecision`; only an explicit
`True` approves.

A call that needs approval, with neither an approver nor a way to pause, is denied. The model is
told "'pay_invoice' was not approved, so it did not run." rather than "not authorized".

### What gets logged

Every step is a [decision event](hooks.md). All the events of one tool call share one
correlation id — also across a pause and its resume, since it is derived from the framework's
tool-call id — and one `request_id`:

| Event | When | `detail` |
|---|---|---|
| `approval_required` | the PDP asked for approval | `advice` or `obligations`: each `{"id", "attributes"}` (so `reason`, `ttl`); `request_id` |
| `approved` / `rejected` | the call was answered | `request_id`, `approver`, `note`, `source` |
| `permit` / `deny` | the final decision | after approval, `evaluated_again`: `true` for the advice form (the second PDP call), `false` for the obligation form; after a rejection, `not_approved: true` and `request_id` |

`source` says who answered: `interrupt` (a human, through the framework's pause), `remembered`
(a still-valid earlier approval), `approver` (your approver callback), or `none` — nobody could
be asked (no approver and no way to pause, or an MCP client without elicitation), so the call
was refused without a human's decision.

`approval_required` is logged once, when the call pauses. A paused call is remembered in
process with the PDP decision that asked for approval, so the PDP is not asked that again: not
when LangChain or Strands re-run the paused step on resume (before the answer reaches it), and
not when the answer is applied. The PDP is asked once for the obligation form, twice for the
advice form (the second time with the approval attribute) — on every framework, with or without
the decision cache; the `on_pdp_request` events show it. A call resumed in **another process**
isn't remembered there and is checked once more. A paused call is kept for the caller it was
paused for (another caller's call with the same call id is not taken for it), for an hour at
most, and is dropped by `config.invalidate_decisions()` for that caller.

**OpenAI:** the OpenAI SDK applies `state.reject()` itself, without calling this SDK's code, so
the rejection (and the final deny) is logged on the next turn from what this process remembers. A run resumed in
**another process** (a saved `RunState`) isn't logged as rejected; the earlier
`approval_required` is, and the tool never runs.

## Examples

`human_approval.py` in each of `examples/langchain_agents/`, `google_adk/`, `strands/`,
`openai_agents/` and `claude_agent/`, and `approval_server.py` + `approval_client.py` in
`examples/fastmcp/`: a finance assistant pays INV-42 (40 EUR, no approval), pays INV-43
(500 EUR, the advice form) and cancels INV-43 (the obligation form). The model is real; the
human is you, at the console, or `--approve` / `--reject`. They share
[`examples/approval_console.py`](../examples/approval_console.py), which also shows the policy
deciding on the invoice's real amount rather than one the model typed in.

## Standards

The shape follows the OpenID **AuthZEN Access Request and Approval Profile** (a working-group
draft): the PDP says "denied, but approvable", and after approval the PDP is asked again and
decides. [draft-ietf-wimse-aims](https://datatracker.ietf.org/doc/draft-ietf-wimse-aims/) §10.7
says a local confirmation is not an authorization by itself; here the PDP makes the final
decision, and the approval is an input to it. A signed approval token (an AuthZEN approval, or a
CIBA-issued token carrying RAR `authorization_details` for the exact call) is planned, as is
approval on the A2A and gateway paths.

---

← Previous: [Intent-based access control](intent-based-access.md) · [Docs index](README.md) · Next: [Post-invocation](post-invocation.md) →
