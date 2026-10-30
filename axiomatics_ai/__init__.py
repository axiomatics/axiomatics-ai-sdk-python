"""Top-level exports.

`core` is always importable (only core deps: httpx, pydantic, cachetools). The path-specific
re-exports below need optional dependencies (agents/rag need langchain), so they are guarded
— a lean install (e.g. the `gateway` or `a2a` extra, without langchain) can still
`import axiomatics_ai` and use `AuthZConfig`, the gateway, etc. Names whose extra is
not installed simply aren't exported.
"""
from axiomatics_ai.core import (
    AgentIdentityAttributes,
    AgentIdentityContextProvider,
    AuditLogHook,
    Claim,
    ClientIdMetadata,
    InvalidationEvent,
    JwtVerification,
    RarMapping,
    TokenVerificationError,
    AuthZConfig,
    Authorizer,
    ContextProvider,
    HealthCheckResult,
    PdpClient,
    PdpEndpoints,
    PdpRequestError,
    DefaultClientContextProvider,
    DefaultServerContextProvider,
    decode_bearer_jwt_unverified,
    PostInvocationStrategy,
    PostInvocationContext,
    RecordFilterStrategy,
    ColumnMaskStrategy,
    AuthzHook,
    LoggingHook,
    HookDispatcher,
    PdpRequestEvent,
    PdpResponseEvent,
    DecisionEvent,
    EnrichmentEvent,
    EnforcementPoint,
    Enforcer,
    PDPConnection,
    AttributeNames,
    RetrievalConfig,
    standard_tool_enforcement,
    ObligationsAndAdviceConfig,
    ApprovalConfig,
    ApprovalDecision,
    ApprovalRequest,
    Obligation,
    ObligationContext,
    EnforcementResult,
    EnricherSpec,
    LLMAttributeEnricher,
    prompt_injection_enricher,
    pii_enricher,
    intent_enricher,
    toxicity_enricher,
    verify_audit_trail,
    SsfReceiver,
)

__all__ = [
    "AgentIdentityAttributes",
    "AgentIdentityContextProvider",
    "ApprovalConfig",
    "ApprovalDecision",
    "ApprovalRequest",
    "AttributeNames",
    "AuditLogHook",
    "AuthZConfig",
    "Authorizer",
    "AuthzHook",
    "Claim",
    "ClientIdMetadata",
    "ColumnMaskStrategy",
    "ContextProvider",
    "DecisionEvent",
    "DefaultClientContextProvider",
    "DefaultServerContextProvider",
    "EnforcementPoint",
    "EnforcementResult",
    "Enforcer",
    "EnricherSpec",
    "EnrichmentEvent",
    "HealthCheckResult",
    "HookDispatcher",
    "InvalidationEvent",
    "JwtVerification",
    "LLMAttributeEnricher",
    "LoggingHook",
    "Obligation",
    "ObligationContext",
    "ObligationsAndAdviceConfig",
    "PDPConnection",
    "PdpClient",
    "PdpEndpoints",
    "PdpRequestError",
    "PdpRequestEvent",
    "PdpResponseEvent",
    "PostInvocationContext",
    "PostInvocationStrategy",
    "RarMapping",
    "RecordFilterStrategy",
    "RetrievalConfig",
    "SsfReceiver",
    "TokenVerificationError",
    "decode_bearer_jwt_unverified",
    "intent_enricher",
    "pii_enricher",
    "prompt_injection_enricher",
    "standard_tool_enforcement",
    "toxicity_enricher",
    "verify_audit_trail",
]

# agents path — needs langchain
try:
    from axiomatics_ai.langchain_agents import (
        LangChainAgentAuthZMiddleware,
        create_authorized_agent,
        create_authorized_deep_agent,
    )
    __all__ += ["LangChainAgentAuthZMiddleware", "create_authorized_agent", "create_authorized_deep_agent"]
except ImportError:
    pass

# rag path — needs langchain / langchain-community
try:
    from axiomatics_ai.rag import authz_context, to_authorized_vectorstore
    __all__ += ["authz_context", "to_authorized_vectorstore"]
except ImportError:
    pass

# fastmcp path — needs fastmcp
try:
    from axiomatics_ai.fastmcp import FastMCPAuthZMiddleware, to_authorized_fastmcp_server
    __all__ += ["FastMCPAuthZMiddleware", "to_authorized_fastmcp_server"]
except ImportError:
    pass

# langchain_mcp path — needs langchain-mcp
try:
    from axiomatics_ai.langchain_mcp import authorized_mcp_tools
    __all__ += ["authorized_mcp_tools"]
except ImportError:
    pass

# google_adk path — needs the google_adk deps (duck-typed at runtime)
try:
    from axiomatics_ai.google_adk import authorized_adk_agent
    __all__ += ["authorized_adk_agent"]
except ImportError:
    pass

# strands path — needs the strands extra
try:
    from axiomatics_ai.strands import authorized_strands_agent
    __all__ += ["authorized_strands_agent"]
except ImportError:
    pass

# openai_agents path — duck-typed, no dedicated extra
try:
    from axiomatics_ai.openai_agents import authorized_openai_agent
    __all__ += ["authorized_openai_agent"]
except ImportError:
    pass

# claude_agent path — needs the claude-agent-sdk extra
try:
    from axiomatics_ai.claude_agent import authorized_claude_options
    __all__ += ["authorized_claude_options"]
except ImportError:
    pass

# gateway path — experimental, not re-exported here (it will move to its own package):
# import it from axiomatics_ai.gateway.

# tool_decorator path — no dedicated extra
try:
    from axiomatics_ai.tool_decorator import protected_tool, configure_authz_decorator
    __all__ += ["configure_authz_decorator", "protected_tool"]
except ImportError:
    pass

# a2a path — needs the a2a extra
try:
    from axiomatics_ai.a2a import A2AAuthZMiddleware, A2AClientAuthZ, A2AAuthorizationError
    __all__ += ["A2AAuthZMiddleware", "A2AAuthorizationError", "A2AClientAuthZ"]
except ImportError:
    pass
