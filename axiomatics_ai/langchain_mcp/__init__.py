"""Tool visibility for MCP tools loaded with the LangChain MCP adapters.

``authorized_mcp_tools`` filters the loaded tools to those the caller may see.
"""
from axiomatics_ai._extras import requires_extra

with requires_extra("langchain-mcp", "axiomatics_ai.langchain_mcp"):
    from axiomatics_ai.langchain_mcp.wrap import authorized_mcp_tools

__all__ = [
    "authorized_mcp_tools",
]
