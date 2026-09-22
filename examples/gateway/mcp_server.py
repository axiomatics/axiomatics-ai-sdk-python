"""
The MCP backend server — a plain FastMCP server, NO authorization here.

Authorization (tool visibility) is enforced by the gateway (gateway.py) in front of it.
This is just a normal MCP server exposing three tools. Run it, then gateway.py, then
mcp_langchain.py. See a2a_server.py for the full run order.

Config: none (no LLM needed for the server).
"""
from fastmcp import FastMCP

PORT = 9212

mcp = FastMCP("acme-tools")


@mcp.tool
def search_docs(query: str) -> str:
    """Search Acme's internal documentation."""
    return f"(docs matching {query!r})"


@mcp.tool
def get_account_balance(account_id: str) -> str:
    """Get the balance for an account."""
    return f"Balance for {account_id}: $1,000"


@mcp.tool
def send_email(to: str, body: str) -> str:
    """Send an email on behalf of the company."""
    return f"email sent to {to}"


if __name__ == "__main__":
    print(f"MCP backend (search_docs, get_account_balance, send_email) on :{PORT}/mcp")
    mcp.run(transport="http", host="127.0.0.1", port=PORT)
