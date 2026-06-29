"""
Typed internal contracts for UserProfiles read/update orchestration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any


def _freeze_contract_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_contract_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_contract_value(item) for item in value)
    return value


class ProfileContractMode(str, Enum):
    LEGACY_V1 = "legacy_v1"
    CLEAN_V2 = "clean_v2"


class EffectTiming(str, Enum):
    PRE_COMMIT = "pre_commit"
    POST_COMMIT = "post_commit"


class EffectPolicy(str, Enum):
    REQUIRED = "required"
    BEST_EFFORT = "best_effort"


@dataclass(frozen=True)
class ProfileReadRequest:
    actor_user_id: int | None
    target_user_id: int
    sections: frozenset[str] | None = None
    include_sources: bool = False
    include_raw: bool = False
    mask_secrets: bool = True
    contract_mode: ProfileContractMode = ProfileContractMode.LEGACY_V1


@dataclass(frozen=True)
class ProfileUpdateCommand:
    actor_user_id: int | None
    target_user_id: int
    updates: tuple[tuple[str, Any], ...]
    roles: frozenset[str]
    dry_run: bool
    expected_profile_version: datetime | None = None
    active_org_id: int | None = None
    active_team_id: int | None = None
    contract_mode: ProfileContractMode = ProfileContractMode.LEGACY_V1


@dataclass(frozen=True)
class UpdateMutation:
    key: str
    operation: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_contract_value(self.payload))


@dataclass(frozen=True)
class EffectDescriptor:
    name: str
    timing: EffectTiming
    policy: EffectPolicy
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_contract_value(self.payload))


@dataclass(frozen=True)
class UpdatePlan:
    command: ProfileUpdateCommand
    mutations: tuple[UpdateMutation, ...] = ()
    effects: tuple[EffectDescriptor, ...] = ()

    @property
    def pre_commit_effects(self) -> tuple[EffectDescriptor, ...]:
        return tuple(effect for effect in self.effects if effect.timing == EffectTiming.PRE_COMMIT)

    @property
    def post_commit_effects(self) -> tuple[EffectDescriptor, ...]:
        return tuple(effect for effect in self.effects if effect.timing == EffectTiming.POST_COMMIT)


@dataclass(frozen=True)
class PlannedUpdateResult:
    profile_version: datetime
    applied: tuple[str, ...] = ()
    rejected: tuple[Mapping[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rejected", _freeze_contract_value(self.rejected))
