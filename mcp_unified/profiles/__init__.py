"""Profile schema and resolver primitives for MCP Unified."""

from .models import MCPProfile, ProfilePolicy
from .presets import (
    ProfilePreset,
    duplicate_builtin_preset,
    get_builtin_preset,
    list_builtin_presets,
    validate_preset_safety,
)
from .resolver import ProfileResolver, StoreBackedProfileResolver
from .store import InMemoryProfileStore, ProfileStoreUnavailableError

__all__ = [
    "InMemoryProfileStore",
    "MCPProfile",
    "ProfilePolicy",
    "ProfilePreset",
    "ProfileResolver",
    "ProfileStoreUnavailableError",
    "StoreBackedProfileResolver",
    "duplicate_builtin_preset",
    "get_builtin_preset",
    "list_builtin_presets",
    "validate_preset_safety",
]
