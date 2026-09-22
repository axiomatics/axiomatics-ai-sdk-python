"""
Docs MCP server for the multi_server_tool_filtering_langgraph.py example.

Run directly for manual testing:
    uv run python examples/langchain_mcp/mcp_docs_server.py
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("axio-mcp-docs-example")


@mcp.tool()
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return f"Results for '{query}': Getting Started Guide, API Reference, FAQ."


if __name__ == "__main__":
    mcp.run()
