"""
Account MCP server for the multi_server_tool_filtering_langgraph.py example.

Run directly for manual testing:
    uv run python examples/langchain_mcp/mcp_account_server.py
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("axio-mcp-account-example")


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
