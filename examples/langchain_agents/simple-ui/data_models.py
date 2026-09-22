from dataclasses import dataclass, field


@dataclass
class AppUser:
    user_id: str
    username: str
    tenant_id: str


@dataclass
class AppContext:
    """
    Runtime context passed to the agent on every invocation.
    ``LangChainAgentAuthZMiddleware`` reads this via the ``ContextProvider``.
    """
    user_id: str
    username: str
    tenant_id: str
    agent_name: str
    session_id: str
    available_tool_names: list = field(default_factory=list)
    allowed_tool_names: list = field(default_factory=list)


@dataclass
class AppConfig:
    app_name: str
    app_url: str
    ads_base_url: str
    invocation_mode: str = "async"
    agent_runtime: str = "vanilla"
    default_agent_name: str = "support-agent"
    request_timeout_seconds: float = 10.0


FAKE_USERS = {
    "alice": {"password": "alice123", "user": AppUser("user-alice", "alice", "tenant-acme")},
    "bob":   {"password": "bob123",   "user": AppUser("user-bob",   "bob",   "tenant-acme")},
}


def authenticate_user(username: str, password: str) -> AppUser | None:
    record = FAKE_USERS.get(username)
    if record and record["password"] == password:
        return record["user"]
    return None
