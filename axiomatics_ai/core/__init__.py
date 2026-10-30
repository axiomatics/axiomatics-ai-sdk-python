"""The core: configuration, the PDP client, enforcement, identity, hooks and presets.

Everything here is framework-independent; the adapters build on it. The public names are
also exported from the top-level ``axiomatics_ai`` package.
"""
from axiomatics_ai.core.config import (
    AuthZConfig,
    DefaultServerContextProvider,
    DefaultClientContextProvider,
    ContextProvider,
    EnforcementPoint,
    Enforcer,
    PDPConnection,
    AttributeNames,
    RetrievalConfig,
    standard_tool_enforcement,
)
from axiomatics_ai.core.authorizer import Authorizer
from axiomatics_ai.core.authz_helpers import decode_bearer_jwt_unverified
from axiomatics_ai.core.pdp import (
    Advice,
    ApprovalConfig,
    ApprovalDecision,
    ApprovalRequest,
    Decision,
    HealthCheckResult,
    Obligation,
    ObligationContext,
    ObligationsAndAdviceConfig,
    PdpClient,
    PdpDecision,
    PdpEndpoints,
    PdpRequestError,
)
from axiomatics_ai.core.cache import AuthzDecisionCacheWithTTL
from axiomatics_ai.core.post_invocation import (
    PostInvocationContext,
    PostInvocationStrategy,
)
from axiomatics_ai.core.enforcement import EnforcementAction, EnforcementResult
from axiomatics_ai.core.llm_enrichment import (
    EnricherSpec,
    EnrichmentOutcome,
    LLMAttributeEnricher,
)
from axiomatics_ai.core.presets import (
    ColumnMaskStrategy,
    RecordFilterStrategy,
    intent_enricher,
    pii_enricher,
    prompt_injection_enricher,
    toxicity_enricher,
)
from axiomatics_ai.core.hooks import (
    PdpRequestEvent,
    PdpResponseEvent,
    AuditLogHook,
    AuthzHook,
    DecisionEvent,
    EnrichmentEvent,
    HookDispatcher,
    InvalidationEvent,
    LoggingHook,
    verify_audit_trail,
)
from axiomatics_ai.core.shared_signals import SignalRejected, SsfReceiver
from axiomatics_ai.core.client_metadata import ClientIdMetadata
from axiomatics_ai.core.identity import (
    AgentIdentityAttributes,
    AgentIdentityContextProvider,
    Claim,
    JwtVerification,
    RarMapping,
    TokenVerificationError,
    verify_access_token,
)

__all__ = [
    "Advice",
    "AgentIdentityAttributes",
    "AgentIdentityContextProvider",
    "ApprovalConfig",
    "ApprovalDecision",
    "ApprovalRequest",
    "AttributeNames",
    "AuditLogHook",
    "AuthZConfig",
    "Authorizer",
    "AuthzDecisionCacheWithTTL",
    "AuthzHook",
    "Claim",
    "ClientIdMetadata",
    "ColumnMaskStrategy",
    "ContextProvider",
    "Decision",
    "DecisionEvent",
    "DefaultClientContextProvider",
    "DefaultServerContextProvider",
    "EnforcementAction",
    "EnforcementPoint",
    "EnforcementResult",
    "Enforcer",
    "EnricherSpec",
    "EnrichmentEvent",
    "EnrichmentOutcome",
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
    "PdpDecision",
    "PdpEndpoints",
    "PdpRequestError",
    "PdpRequestEvent",
    "PdpResponseEvent",
    "PostInvocationContext",
    "PostInvocationStrategy",
    "RarMapping",
    "RecordFilterStrategy",
    "RetrievalConfig",
    "SignalRejected",
    "SsfReceiver",
    "TokenVerificationError",
    "decode_bearer_jwt_unverified",
    "intent_enricher",
    "pii_enricher",
    "prompt_injection_enricher",
    "standard_tool_enforcement",
    "toxicity_enricher",
    "verify_access_token",
    "verify_audit_trail",
]
