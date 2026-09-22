# Assets

Figures and pictures used by the README and the documentation.

Each Mermaid diagram that appears in the README or in `docs/` is also kept here as its own
file, so it can be rendered or exported (PNG / SVG) on its own and reused in slides or blog
posts. The copy in the page that uses it is the source of truth; when you change one, update
the other.

## Diagrams

- [how-it-fits.md](how-it-fits.md) — How it fits: application, gateway, and resources reach the PDP through the SDK (PEP)
- [interception-points.md](interception-points.md) — Where it intercepts: the seven enforcement points along one agent turn
- [sequence-client-side.md](sequence-client-side.md) — Request flow, client side (agents, langchain_mcp, rag, direct)
- [sequence-server-side.md](sequence-server-side.md) — Request flow, server side (fastmcp, tool_decorator, a2a server)
- [sequence-gateway.md](sequence-gateway.md) — Request flow through the gateway (reverse proxy in front of MCP / A2A)

## Pictures

Put exported PNG / SVG renders and screenshots in this folder next to the diagram they come
from, using the same base name (for example `how-it-fits.png`).
