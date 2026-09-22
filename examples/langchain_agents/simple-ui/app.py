import os
import uuid

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from agent import build_agent
from async_utils import run_async
from data_models import AppConfig, AppContext, authenticate_user



def load_config() -> AppConfig:
    return AppConfig(
        app_name=os.environ.get("APP_NAME", "axio-langchain-demo"),
        app_url=os.environ.get("APP_URL", "http://localhost:8501"),
        ads_base_url=os.getenv("PDP_BASE_URL", "http://localhost:8080"),
        invocation_mode=os.environ.get("INVOCATION_MODE", "async").strip().lower(),
        agent_runtime=os.environ.get("AGENT_RUNTIME", "vanilla").strip().lower(),
    )


st.set_page_config(page_title="Axiomatics Authorized Agent Example", layout="wide")

if "config" not in st.session_state:
    st.session_state.config = load_config()
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


_POLICY_MD = """
## Active Policy: `domain_simple_ui.yaml`

Controls what each user can do with the agent. MCP tools are only available in `async` invocation mode.

### Demo Users

| User | Password |
|------|----------|
| `alice` | `alice123` |
| `bob` | `bob123` |

### Tool Listing (`tool_visibility`)

Which tools the LLM is allowed to see and use.

| Tool | Source | alice | bob |
|------|--------|-------|-----|
| `search_docs` | direct | ✅ | ✅ |
| `create_ticket` | direct | ✅ | ❌ |
| `send_email` | direct | ✅ | ❌ |
| `get_account_balance` | MCP (async only) | ✅ | ✅ |
| `lookup_customer_profile` | MCP (async only) | ✅ | ❌ |

### Tool Call Authorization (`tool_invocation`)

Whether a specific tool call with its arguments is permitted.

| Tool call | alice | bob |
|-----------|-------|-----|
| `search_docs(query=*)` | ✅ | ✅ |
| `create_ticket(...)` | ✅ | ❌ |
| `send_email(...)` | ✅ | ❌ |
| `get_account_balance(account_id=ACC-001)` | ✅ | ✅ |
| `get_account_balance(account_id=<other>)` | ✅ | ❌ |
| `lookup_customer_profile(customer_id=*)` | ✅ | ❌ |

### Record Filtering (`post_invocation`)

Which records from a tool response are visible.

| Record type | alice | bob |
|-------------|-------|-----|
| `doc_type=public` | ✅ | ✅ |
| `doc_type=classified` | ✅ | ❌ |
"""


@st.dialog("Active Policy")
def show_policy():
    st.markdown(_POLICY_MD)


def logout():
    st.session_state.user = None
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


st.title("Axiomatics Authorized Agent Example")

with st.sidebar:
    if st.button("📋 View Active Policy"):
        show_policy()
    st.divider()
    st.subheader("Login")
    st.caption(f"Agent: {st.session_state.config.agent_runtime}")
    st.caption(f"Invocation: {st.session_state.config.invocation_mode}")

    if st.session_state.user is None:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                user = authenticate_user(username, password)
                if user is None:
                    st.error("Invalid username or password")
                else:
                    st.session_state.user = user
                    st.rerun()
    else:
        st.success(f"Logged in as {st.session_state.user.username}")
        st.write(f"User ID: {st.session_state.user.user_id}")
        st.write(f"Tenant ID: {st.session_state.user.tenant_id}")
        if st.button("Logout"):
            logout()
            st.rerun()


if st.session_state.user is None:
    st.info("Please log in to continue.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        app_context = AppContext(
            user_id=st.session_state.user.user_id,
            username=st.session_state.user.username,
            tenant_id=st.session_state.user.tenant_id,
            agent_name=st.session_state.config.default_agent_name,
            session_id=st.session_state.session_id,
        )
        agent = build_agent(st.session_state.config, app_context)
        agent_input = {
            "messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        }

        if st.session_state.config.invocation_mode == "sync":
            result = agent.invoke(agent_input, context=app_context)
        else:
            result = run_async(agent.ainvoke(agent_input, context=app_context))

        ai_messages = [msg for msg in result["messages"] if getattr(msg, "type", None) == "ai"]
        assistant_text = ai_messages[-1].content if ai_messages else ""

        st.markdown(assistant_text)
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
