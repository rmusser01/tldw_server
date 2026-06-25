from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

RPGSourceType = Literal["user", "system", "mcp", "model", "import"]


@dataclass(frozen=True, slots=True)
class RuleLicenseSummary:
    license_name: str
    source_title: str
    source_url: str
    attribution_required: bool
    commercial_use_allowed: bool
    notes: str = ""


@dataclass(frozen=True, slots=True)
class RuleAdapterInfo:
    adapter_key: str
    adapter_version: str
    display_name: str
    status: str
    source_version: str
    bundled_snippet_policy: str
    license_summary: RuleLicenseSummary
    mechanics_tags: dict[str, str]


@dataclass(frozen=True, slots=True)
class RPGCampaign:
    id: int
    owner_user_id: int
    title: str
    description: str | None
    default_adapter_key: str
    default_adapter_version: str
    settings: dict[str, Any]
    linked_rules_pack_refs: list[dict[str, Any]]
    version: int
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RPGSession:
    id: int
    campaign_id: int
    owner_user_id: int
    title: str
    status: str
    adapter_key: str
    adapter_version: str
    authority_settings: dict[str, Any]
    linked_chat_id: int | None
    active_rules_pack_refs: list[dict[str, Any]]
    current_snapshot_version: int
    last_event_sequence: int
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RPGSessionEvent:
    id: int
    session_id: int
    owner_user_id: int
    sequence_number: int
    event_type: str
    event_payload: dict[str, Any]
    source_type: RPGSourceType
    source_actor_id: str | None
    source_label: str | None
    operation_id: int | None
    event_schema_version: str
    adapter_key: str
    adapter_version: str
    proposal_id: int | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RPGSnapshotState:
    scene: dict[str, Any] = field(default_factory=dict)
    actors: dict[str, dict[str, Any]] = field(default_factory=dict)
    resources: dict[str, dict[str, Any]] = field(default_factory=dict)
    clocks: dict[str, dict[str, Any]] = field(default_factory=dict)
    rolls: list[dict[str, Any]] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)
    recap: str = ""
    quests: dict[str, dict[str, Any]] = field(default_factory=dict)
    npcs: dict[str, dict[str, Any]] = field(default_factory=dict)
    inventory: dict[str, dict[str, Any]] = field(default_factory=dict)
    locations: dict[str, dict[str, Any]] = field(default_factory=dict)
    factions: dict[str, dict[str, Any]] = field(default_factory=dict)
    rules_references: list[dict[str, Any]] = field(default_factory=list)
    unresolved_rulings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RPGSnapshotRecord:
    id: int
    session_id: int
    owner_user_id: int
    snapshot_version: int
    last_event_sequence: int
    reducer_version: str
    snapshot_schema_version: str
    snapshot_json: dict[str, Any]
    diagnostics_json: dict[str, Any]
    created_at: datetime
