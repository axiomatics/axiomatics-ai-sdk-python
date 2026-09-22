"""
The A2A agent server — a plain ADK billing agent, NO authorization here.

Authorization is enforced by the gateway (gateway.py) that sits in front of this agent, so
this file is just a normal ADK A2A server. Run it, then run gateway.py, then a2a_client.py.

Run order (separate terminals):
  1. Start ADS with domain_gateway.yaml on PDP_BASE_URL.
  2. python examples/gateway/a2a_server.py        # this agent, on :9211
  3. python examples/gateway/mcp_server.py        # the MCP backend, on :9212 (for the MCP demo)
  4. python examples/gateway/gateway.py           # the gateway, on :9210 (fronts both)
  5. python examples/gateway/a2a_client.py        # or mcp_langchain.py

Config in .env: LLM_MODEL, OPENROUTER_API_KEY
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.a2a.utils.agent_to_a2a import to_a2a

load_dotenv()

PORT = 9211

agent = Agent(
    name="billing",
    model=LiteLlm(os.getenv("LLM_MODEL")),
    instruction=(
        "You are the billing assistant for Acme Corp. When asked about billing, state that "
        "the company billing address is '1200 Market Street, Suite 400, San Francisco, CA 94103'. "
        "Answer briefly."
    ),
)
app = to_a2a(agent, port=PORT)

if __name__ == "__main__":
    print(f"ADK billing agent (A2A) on :{PORT} — put the gateway in front of it")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
