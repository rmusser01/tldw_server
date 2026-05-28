"""Profile schema and resolver primitives for MCP Unified."""

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
from .store import InMemoryProfileStore, ProfileStoreUnavailableError

__all__ = [
    "EffectivePolicy",
    "EffectivePolicyResult",
    "EffectivePolicyStatus",
    "InMemoryProfileStore",
    "MCPProfile",
    "ProfilePolicy",
    "ProfilePreset",
    "ProfileResolver",
    "ProfileResolutionResult",
    "ProfileResolutionStatus",
    "ProfileStoreUnavailableError",
    "StoreBackedProfileResolver",
    "build_effective_policy_result",
    "duplicate_builtin_preset",
    "get_builtin_preset",
    "list_builtin_presets",
    "validate_preset_safety",
]
