"""
Standalone MCP server for the tool_filtering.py example.

Run this in a separate terminal (or let tool_filtering.py spawn it via stdio):
    uv run python examples/langchain_mcp/mcp_server.py
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("axio-mcp-example")


@mcp.tool()
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return f"Results for '{query}': Getting Started Guide, API Reference, FAQ."


@mcp.tool()
def get_account_balance(account_id: str) -> str:
    """Retrieve the balance for an account."""
    return f"Account {account_id} has a balance of $1,250.00."


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject '{subject}'."


if __name__ == "__main__":
    mcp.run()
