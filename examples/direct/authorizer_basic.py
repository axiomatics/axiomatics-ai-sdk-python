"""
Direct path — ask the PDP from your own code, no framework adapter.

Requires the PDP running with examples/policies/domain_all_authz_enabled.yaml (the same policy the
langchain_agents examples use), reachable at PDP_BASE_URL (default http://localhost:8080).

    uv run python examples/direct/authorizer_basic.py

The policy says: alice sees every tool and may call any of them; bob does not see send_email
and may call get_account_balance only for account ACC-001. The application below decides *where*
to ask; the SDK builds the request and returns the answer. The policy identifies tools by the
`tool_name` attribute, so both calls send the resource under that name (the direct path's own
default is `resource_name`).
"""
import os

from axiomatics_ai import Authorizer, AuthZConfig, LoggingHook, PDPConnection

TOOLS = ["search_docs", "get_account_balance", "send_email"]


def main() -> None:
    config = AuthZConfig(
        connection=PDPConnection(base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080")),
        hooks=[LoggingHook()],   # optional: log every PDP call and decision
    )
    authz = Authorizer(config)

    health = authz.check_connection()
    if not health:
        raise SystemExit(f"PDP not reachable: {health.error}")
    print(f"PDP OK ({health.latency_ms:.0f} ms)\n")

    for user in ("alice", "bob"):
        ctx = {"username": user}

        # One batched request: which tools may this user see?
        visible = authz.is_permitted_batch(
            action="tool_visibility", resource_name_attribute="tool_name", resources=TOOLS, context=ctx,
        )
        print(f"{user}: visible tools -> {sorted(visible)}")

        # One request per call: may this user invoke the tool with these arguments?
        for account in ("ACC-001", "ACC-999"):
            ok = authz.is_permitted(
                action="tool_invocation", resource="get_account_balance", resource_name_attribute="tool_name",
                args={"account_id": account}, context=ctx,
            )
            print(f"{user}: get_account_balance(account_id={account}) -> {'permit' if ok else 'deny'}")

        # Partial evaluation: under what condition may this user invoke a tool? (ADS CAQ)
        condition = authz.partial_evaluation_text(action="tool_invocation", context=ctx)
        print(f"{user}: tool_invocation is permitted when: {condition or '(no rendering)'}\n")


if __name__ == "__main__":
    main()
