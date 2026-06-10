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
from .path_grants import (
    PATH_GRANT_ACTIONS,
    PATH_GRANT_AUTHORING_KEYS,
    PATH_GRANT_EFFECTS,
    PathGrantCompilationResult,
    PathGrantDiagnostic,
    compile_hierarchical_path_grants,
    compile_policy_path_grants,
    has_path_grant_policy,
)
from .permission_rules import (
    PermissionRuleSubject,
    compile_permission_rules,
    evaluate_permission_rule_decision,
    parse_permission_rule,
)
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
    "PATH_GRANT_ACTIONS",
    "PATH_GRANT_AUTHORING_KEYS",
    "PATH_GRANT_EFFECTS",
    "PolicyDecision",
    "PolicyDecisionCallState",
    "PolicyDecisionOutcome",
    "PolicyDecisionRule",
    "PolicyDecisionSubject",
    "PolicyDecisionVisibility",
    "PolicyExplanation",
    "PolicyMatchedRule",
    "PathGrantCompilationResult",
    "PathGrantDiagnostic",
    "PermissionRuleSubject",
    "ProfileAlreadyExistsError",
    "ProfilePolicy",
    "ProfilePreset",
    "ProfileResolver",
    "ProfileResolutionResult",
    "ProfileResolutionStatus",
    "ProfileStoreUnavailableError",
    "StoreBackedProfileResolver",
    "build_effective_policy_result",
    "compile_hierarchical_path_grants",
    "compile_permission_rules",
    "compile_policy_path_grants",
    "compile_profile_policy_rules",
    "duplicate_builtin_preset",
    "evaluate_profile_tool_decision",
    "evaluate_permission_rule_decision",
    "explain_profile_tool_decision",
    "get_builtin_preset",
    "has_path_grant_policy",
    "list_builtin_presets",
    "merge_policy_decisions",
    "parse_permission_rule",
    "validate_preset_safety",
]
