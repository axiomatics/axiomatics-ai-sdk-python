# How it fits: application, gateway, and resources reach the PDP through the SDK (PEP)

The README embeds the rendered `how-it-fits.png`; this file is the editable Mermaid source. Re-export the PNG after changing it.

```mermaid
flowchart LR
    subgraph APP["AI application"]
        direction TB
        A1["Agents<br/>LangChain · Google ADK · Strands<br/>OpenAI Agents · Claude Agent SDK"]
        A2["Client-side MCP tools<br/>(e.g., LangChain MCP adapters)"]
        A3["RAG retrieval<br/>(any LangChain vector store)"]
        A4["Custom code<br/>(Authorizer)"]
    end
    subgraph GW["Gateway"]
        direction TB
        G1["Reverse proxy for MCP and A2A"]
    end
    subgraph RES["Resources"]
        direction TB
        R1["MCP servers<br/>(FastMCP middleware · tool decorators)"]
        R2["A2A agents<br/>(Client guard)"]
    end
    SDK["<b>axiomatics_ai</b><br/>Policy Enforcement Point (PEP)"]
    PDP["<b>Policy Decision Point (PDP)</b><br/>XACML 3.0 JSON · OpenID AuthZEN"]
    APP <--> GW
    GW <--> RES
    APP --> SDK
    GW --> SDK
    RES --> SDK
    SDK -- "Attributes" --> PDP
    PDP -- "Permit/Deny/Condition" --> SDK
```
