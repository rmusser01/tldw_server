"""Profile schema and resolver primitives for MCP Unified."""

from .decisions import (
    PolicyDecision,
    PolicyDecisionCallState,
    PolicyDecisionOutcome,
    PolicyDecisionRule,
    PolicyDecisionSubject,
    PolicyDecisionVisibility,
    PolicyExplanation,
    PolicyMatchedRule,
    compile_profile_policy_rules,
    evaluate_profile_tool_decision,
    explain_profile_tool_decision,
    merge_policy_decisions,
)
from .models import MCPProfile, ProfilePolicy
from .presets import (
    ProfilePreset,
    duplicate_builtin_preset,
    get_builtin_preset,
    list_builtin_presets,
    validate_preset_safety,
)
from .resolution import (
    EffectivePolicy,
    EffectivePolicyResult,
    EffectivePolicyStatus,
    ProfileResolutionResult,
    ProfileResolutionStatus,
    build_effective_policy_result,
)
from .resolver import ProfileResolver, StoreBackedProfileResolver
from .store import (
    InMemoryProfileStore,
    ProfileAlreadyExistsError,
    ProfileStoreUnavailableError,
)

__all__ = [
    "EffectivePolicy",
    "EffectivePolicyResult",
    "EffectivePolicyStatus",
    "InMemoryProfileStore",
    "MCPProfile",
    "PolicyDecision",
    "PolicyDecisionCallState",
    "PolicyDecisionOutcome",
    "PolicyDecisionRule",
    "PolicyDecisionSubject",
    "PolicyDecisionVisibility",
    "PolicyExplanation",
    "PolicyMatchedRule",
    "ProfileAlreadyExistsError",
    "ProfilePolicy",
    "ProfilePreset",
    "ProfileResolver",
    "ProfileResolutionResult",
    "ProfileResolutionStatus",
    "ProfileStoreUnavailableError",
    "StoreBackedProfileResolver",
    "build_effective_policy_result",
    "compile_profile_policy_rules",
    "duplicate_builtin_preset",
    "evaluate_profile_tool_decision",
    "explain_profile_tool_decision",
    "get_builtin_preset",
    "list_builtin_presets",
    "merge_policy_decisions",
    "validate_preset_safety",
]
