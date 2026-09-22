"""
Example: a custom AuthzHook — build your own audit / metrics from SDK events.

Subclass ``AuthzHook`` and override only the events you care about. Each method receives
one event object and returns nothing; do whatever you like with it. Hooks are observe-only
(they never change a decision), path-independent (attach once, fire on every path), and
isolated (an exception in a hook is logged, never breaks enforcement).

This ``AuditHook`` keeps its own state — a list of denials and a running latency total —
built purely from the two event levels:
  - on_decision:    the semantic outcome at each enforcement point (surface, decision, ...)
  - on_pdp_response: one per ADS wire call (latency, status, errors)

The ``correlation_id`` on every event ties the decision to the ADS call(s) behind it.

Prerequisites:
  - A running ADS loaded with examples/policies/domain_all_authz_enabled.yaml
    (set PDP_BASE_URL, or it defaults to http://localhost:8080).
  - LLM_MODEL and your API key in a .env file or environment variables.
"""
import os

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM
from langchain.tools import tool

from axiomatics_ai import LangChainAgentAuthZMiddleware, AuthZConfig, AuthzHook
from axiomatics_ai import PDPConnection, standard_tool_enforcement


class AuditHook(AuthzHook):
    """Collects denials and ADS latency into plain Python state you own."""

    def __init__(self):
        self.denials = []          # list of (surface, enforcement_point, resource)
        self.ads_calls = 0
        self.total_latency_ms = 0.0

    def on_decision(self, event):
        # fields: surface, enforcement_point, subject, action, resource, decision, cached, correlation_id
        if event.decision == "deny":
            self.denials.append((event.surface, event.enforcement_point, event.resource))

    def on_pdp_response(self, event):
        # fields: endpoint, url, protocol, status_code, latency_ms, error, correlation_id
        self.ads_calls += 1
        self.total_latency_ms += event.latency_ms

    def report(self):
        avg = self.total_latency_ms / self.ads_calls if self.ads_calls else 0.0
        print("\n── audit summary ───────────────────────────────────────────────────")
        print(f"ADS calls: {self.ads_calls}   avg latency: {avg:.1f}ms")
        print(f"denials ({len(self.denials)}):")
        for surface, point, resource in self.denials:
            print(f"  [{surface}] {point}: {resource}")


@tool
def get_account_balance(account_id: str) -> str:
    """Get the balance for an account."""
    return f"Balance for {account_id}: $1,000"


def main():
    ads_url = os.getenv("PDP_BASE_URL", "http://localhost:8080")

    audit = AuditHook()
    config = AuthZConfig(connection=PDPConnection(base_url=ads_url), enforcement=standard_tool_enforcement(), hooks=[audit])

    model = ChatLiteLLM(model=os.getenv("LLM_MODEL"))
    agent = create_agent(
        model=model,
        tools=[get_account_balance],
        middleware=[LangChainAgentAuthZMiddleware(config)],
    )

    # bob is only permitted to read ACC-001; the call for ACC-002 is denied by ADS. The
    # AuditHook records that denial and the latency of every ADS call behind the scenes.
    agent.invoke(
        {"messages": [{"role": "user", "content": "Get the balance for account ACC-002."}]},
        context={"username": "bob", "session_id": "s1"},
    )

    audit.report()


if __name__ == "__main__":
    main()
