"""
MCP backend for the Strands example — the tools the agent consumes.

A standalone streamable-HTTP MCP server. Start it in its own process (like ADS), then run the
agent, which connects to it via MCP_URL:

    python examples/strands/mcp_backend.py                # serves http://127.0.0.1:9320/mcp
    python examples/strands/mcp_agent.py                  # connects via MCP_URL (default above)
"""
import os

from fastmcp import FastMCP

DEFAULT_PORT = 9320


def build_server() -> FastMCP:
    mcp = FastMCP("acme-tools")

    @mcp.tool
    def search_docs(query: str) -> list[dict]:
        """Search Acme's internal documentation and return matching articles."""
        return [
            {"id": "d1", "title": "Getting Started", "doc_type": "public"},
            {"id": "d2", "title": "Internal Spec", "doc_type": "classified"},
        ]

    @mcp.tool
    def get_account_balance(account_id: str) -> str:
        """Get the balance for an account."""
        return f"Balance for {account_id}: $1,000"

    @mcp.tool
    def send_email(to: str, body: str) -> str:
        """Send an email on behalf of the company."""
        return f"email sent to {to}"

    return mcp


if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", DEFAULT_PORT))
    print(f"MCP backend serving http://127.0.0.1:{port}/mcp  (Ctrl-C to stop)")
    build_server().run(transport="http", host="127.0.0.1", port=port)
