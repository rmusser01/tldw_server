# RPG Campaign Session Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a generic RPG/TTRPG backend harness for campaigns, sessions, rules adapters, dice/check resolution, event proposals, rules lookup, REST APIs, and MCP tools without becoming a virtual tabletop.

**Architecture:** Add a focused `tldw_Server_API.app.core.RPG` package backed by per-user ChaChaNotes storage through a dedicated `RPGRepository`. Session events are the source of truth; snapshots are deterministic cached projections; model-sourced changes become proposals unless session authority settings allow direct commit.

**Tech Stack:** FastAPI, Pydantic v2, SQLite via `CharactersRAGDB`, Loguru, pytest, MCP Unified module APIs, existing AuthNZ token-scope/privilege catalog tooling.

---

## Reference Inputs

- Approved spec: `Docs/superpowers/specs/2026-06-25-rpg-campaign-session-runtime-design.md`
- Adjacent runtime pattern: `tldw_Server_API/app/core/VN_Play/`
- Adjacent storage pattern: `tldw_Server_API/app/core/DB_Management/VNPlay_DB.py`
- Adjacent endpoint pattern: `tldw_Server_API/app/api/v1/endpoints/vn_play.py`
- Router registration: `tldw_Server_API/app/api/v1/router_groups/content.py`
- MCP module pattern: `tldw_Server_API/app/core/MCP_unified/modules/implementations/quizzes_module.py`
- Privilege catalog and snapshot tooling:
  - `tldw_Server_API/Config_Files/privilege_catalog.yaml`
  - `Helper_Scripts/update_privilege_registry_snapshot.py`
  - `tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py`

## File Structure

Create the RPG runtime as focused files:

- `tldw_Server_API/app/core/RPG/__init__.py`: public package exports.
- `tldw_Server_API/app/core/RPG/constants.py`: adapter keys, event type strings, schema versions, reducer version, limits.
- `tldw_Server_API/app/core/RPG/errors.py`: domain exceptions mapped by REST and MCP layers.
- `tldw_Server_API/app/core/RPG/models.py`: core dataclasses for campaigns, sessions, events, snapshots, proposals, adapters, checks, citations.
- `tldw_Server_API/app/core/RPG/rules/__init__.py`: rule adapter package exports.
- `tldw_Server_API/app/core/RPG/rules/adapters.py`: `RuleAdapter` protocol, bundled D&D 5e SRD, PF2e, and Fate adapters, registry builder.
- `tldw_Server_API/app/core/RPG/rules/content_packs.py`: small bundled citation metadata and user-rules-pack reference models.
- `tldw_Server_API/app/core/RPG/rules/lookup.py`: rules lookup service over bundled snippets and linked RAG/media references.
- `tldw_Server_API/app/core/RPG/events.py`: event envelope validation, canonical request hashes, idempotency helpers.
- `tldw_Server_API/app/core/RPG/reducer.py`: pure event-to-snapshot reducer.
- `tldw_Server_API/app/core/RPG/dice.py`: dice expression parser and roller with injectable randomness for tests.
- `tldw_Server_API/app/core/RPG/checks.py`: adapter-backed check resolution.
- `tldw_Server_API/app/core/RPG/authority.py`: direct-commit versus proposal decisions.
- `tldw_Server_API/app/core/RPG/proposals.py`: proposal validation and preview helpers.
- `tldw_Server_API/app/core/RPG/context.py`: bounded session context builder with citation diagnostics.
- `tldw_Server_API/app/core/RPG/service.py`: application service used by REST and MCP.
- `tldw_Server_API/app/core/RPG/README.md`: concise runtime design and legal-content notes.
- `tldw_Server_API/app/core/DB_Management/RPG_DB.py`: repository/schema initialization around a per-user `CharactersRAGDB`.
- `tldw_Server_API/app/api/v1/schemas/rpg_schemas.py`: API request/response models.
- `tldw_Server_API/app/api/v1/endpoints/rpg.py`: REST endpoints under `/api/v1/rpg`.
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/rpg_module.py`: optional MCP module.

Modify integration points:

- `tldw_Server_API/app/api/v1/router_groups/content.py`: add route-key gated RPG router.
- `tldw_Server_API/app/core/MCP_unified/server.py`: optional `MCP_ENABLE_RPG_MODULE` registration.
- `tldw_Server_API/Config_Files/privilege_catalog.yaml`: add RPG privilege scopes and endpoint IDs.
- `tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json`: regenerate after endpoint metadata is added.

Create tests:

- `tldw_Server_API/tests/RPG/test_rules_adapters.py`
- `tldw_Server_API/tests/RPG/test_rpg_db.py`
- `tldw_Server_API/tests/RPG/test_rpg_events_reducer.py`
- `tldw_Server_API/tests/RPG/test_rpg_dice_checks.py`
- `tldw_Server_API/tests/RPG/test_rpg_service.py`
- `tldw_Server_API/tests/RPG/test_rpg_api.py`
- `tldw_Server_API/tests/RPG/test_rpg_rules_context.py`
- `tldw_Server_API/tests/RPG/test_rpg_mcp_module.py`

## Execution Notes

- Keep the implementation under a Backlog.md task and update touched files, verification, and final summary as work proceeds.
- Use `source .venv/bin/activate` before Python, pytest, Bandit, or helper commands.
- Keep bundled PF2e prose out of v1 unless an explicit source/license inventory is added in the same task; mechanics metadata and citations are enough for the adapter contract.
- Do not add maps, token positions, battlemap assets, shared tabletop synchronization, or a FoundryVTT-style canvas.
- Every mutating service path must use an idempotency key or reject repeated unsafe writes with a domain error.

### Task 1: Core RPG Models And Adapter Registry

**Files:**
- Create: `tldw_Server_API/app/core/RPG/__init__.py`
- Create: `tldw_Server_API/app/core/RPG/constants.py`
- Create: `tldw_Server_API/app/core/RPG/errors.py`
- Create: `tldw_Server_API/app/core/RPG/models.py`
- Create: `tldw_Server_API/app/core/RPG/rules/__init__.py`
- Create: `tldw_Server_API/app/core/RPG/rules/adapters.py`
- Test: `tldw_Server_API/tests/RPG/test_rules_adapters.py`

- [ ] **Step 1: Write failing adapter contract tests**

```python
from tldw_Server_API.app.core.RPG.rules.adapters import build_default_adapter_registry


def test_default_registry_exposes_version_pinned_adapters():
    registry = build_default_adapter_registry()

    assert sorted(registry.adapter_keys()) == ["dnd5e_srd", "fate", "pf2e"]
    dnd = registry.get("dnd5e_srd")
    assert dnd.source_version == "SRD 5.1"
    assert dnd.adapter_version == "1.0.0"
    assert dnd.license_summary.source_title == "Dungeons & Dragons SRD 5.1"


def test_pf2e_adapter_starts_metadata_only_for_rules_prose():
    pf2e = build_default_adapter_registry().get("pf2e")

    assert pf2e.status == "metadata_only"
    assert pf2e.bundled_snippet_policy == "citations_only"
    assert pf2e.mechanics_tags["resolution_family"] == "d20"


def test_fate_adapter_does_not_require_d20_fields():
    fate = build_default_adapter_registry().get("fate")

    actor_schema = fate.actor_schema()
    assert "aspects" in actor_schema["properties"]
    assert "stress" in actor_schema["properties"]
    assert "ability_scores" not in actor_schema["required"]
    assert fate.mechanics_tags["resolution_family"] == "fate"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rules_adapters.py -v`

Expected: FAIL because `tldw_Server_API.app.core.RPG` does not exist.

- [ ] **Step 3: Add core constants, errors, and dataclasses**

```python
# tldw_Server_API/app/core/RPG/constants.py
RPG_ADAPTER_DND5E_SRD = "dnd5e_srd"
RPG_ADAPTER_PF2E = "pf2e"
RPG_ADAPTER_FATE = "fate"

RPG_ADAPTER_VERSION_V1 = "1.0.0"
RPG_EVENT_SCHEMA_VERSION = "1.0.0"
RPG_SNAPSHOT_SCHEMA_VERSION = "1.0.0"
RPG_REDUCER_VERSION = "1.0.0"

RPG_SOURCE_TYPES = ("user", "system", "mcp", "model", "import")
RPG_PROPOSAL_STATUSES = ("pending", "applied", "rejected", "expired", "conflicted")

MAX_RPG_EVENT_PAYLOAD_BYTES = 64_000
MAX_RPG_CONTEXT_CHARS = 24_000
```

```python
# tldw_Server_API/app/core/RPG/errors.py
class RPGError(Exception):
    """Base RPG domain error."""


class RPGNotFoundError(RPGError):
    """Requested RPG object was not found for the current user."""


class RPGConflictError(RPGError):
    """Write could not be applied because state or idempotency conflicted."""


class RPGValidationError(RPGError):
    """Input failed RPG domain validation."""


class RPGPermissionError(RPGError):
    """Caller cannot perform the requested RPG action."""
```

```python
# tldw_Server_API/app/core/RPG/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

RPGSourceType = Literal["user", "system", "mcp", "model", "import"]
RPGProposalStatus = Literal["pending", "applied", "rejected", "expired", "conflicted"]


@dataclass(frozen=True, slots=True)
class RuleLicenseSummary:
    source_title: str
    source_url: str
    license: str
    license_url: str
    attribution: str


@dataclass(frozen=True, slots=True)
class RuleCitation:
    adapter_key: str
    source_title: str
    source_url: str
    license: str
    license_url: str
    attribution: str
    trust_level: str
    content_hash: str
    snippet_id: str
    source_version: str
    content_pack_version: str


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
    idempotency_key: str | None
    request_payload_hash: str | None
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
    notes: list[dict[str, Any]] = field(default_factory=list)
    recap: str = ""
    quests: dict[str, dict[str, Any]] = field(default_factory=dict)
    npcs: dict[str, dict[str, Any]] = field(default_factory=dict)
    inventory: dict[str, dict[str, Any]] = field(default_factory=dict)
    locations: dict[str, dict[str, Any]] = field(default_factory=dict)
    factions: dict[str, dict[str, Any]] = field(default_factory=dict)
    rules_references: list[dict[str, Any]] = field(default_factory=list)
    unresolved_rulings: dict[str, dict[str, Any]] = field(default_factory=dict)
```

- [ ] **Step 4: Add the adapter protocol and bundled registry**

```python
# tldw_Server_API/app/core/RPG/rules/adapters.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from tldw_Server_API.app.core.RPG.constants import (
    RPG_ADAPTER_DND5E_SRD,
    RPG_ADAPTER_FATE,
    RPG_ADAPTER_PF2E,
    RPG_ADAPTER_VERSION_V1,
)
from tldw_Server_API.app.core.RPG.errors import RPGNotFoundError
from tldw_Server_API.app.core.RPG.models import RuleAdapterInfo, RuleLicenseSummary


class RuleAdapter(Protocol):
    adapter_key: str
    adapter_version: str
    display_name: str
    status: str
    source_version: str
    bundled_snippet_policy: str
    license_summary: RuleLicenseSummary
    mechanics_tags: dict[str, str]

    def actor_schema(self) -> dict[str, Any]:
        raise NotImplementedError

    def check_schema(self) -> dict[str, Any]:
        raise NotImplementedError

    def info(self) -> RuleAdapterInfo:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class StaticRuleAdapter:
    adapter_key: str
    adapter_version: str
    display_name: str
    status: str
    source_version: str
    bundled_snippet_policy: str
    license_summary: RuleLicenseSummary
    mechanics_tags: dict[str, str]
    _actor_schema: dict[str, Any]
    _check_schema: dict[str, Any]

    def actor_schema(self) -> dict[str, Any]:
        return dict(self._actor_schema)

    def check_schema(self) -> dict[str, Any]:
        return dict(self._check_schema)

    def info(self) -> RuleAdapterInfo:
        return RuleAdapterInfo(
            adapter_key=self.adapter_key,
            adapter_version=self.adapter_version,
            display_name=self.display_name,
            status=self.status,
            source_version=self.source_version,
            bundled_snippet_policy=self.bundled_snippet_policy,
            license_summary=self.license_summary,
            mechanics_tags=dict(self.mechanics_tags),
        )


class RuleAdapterRegistry:
    def __init__(self, adapters: list[RuleAdapter]) -> None:
        self._adapters = {adapter.adapter_key: adapter for adapter in adapters}

    def adapter_keys(self) -> list[str]:
        return sorted(self._adapters)

    def get(self, adapter_key: str) -> RuleAdapter:
        try:
            return self._adapters[adapter_key]
        except KeyError as exc:
            raise RPGNotFoundError(f"Unknown RPG rules adapter: {adapter_key}") from exc

    def list_infos(self) -> list[RuleAdapterInfo]:
        return [self._adapters[key].info() for key in self.adapter_keys()]


def build_default_adapter_registry() -> RuleAdapterRegistry:
    return RuleAdapterRegistry(
        adapters=[
            StaticRuleAdapter(
                adapter_key=RPG_ADAPTER_DND5E_SRD,
                adapter_version=RPG_ADAPTER_VERSION_V1,
                display_name="D&D 5e SRD",
                status="bundled",
                source_version="SRD 5.1",
                bundled_snippet_policy="short_srd_citations",
                license_summary=RuleLicenseSummary(
                    source_title="Dungeons & Dragons SRD 5.1",
                    source_url="https://www.dndbeyond.com/srd",
                    license="Creative Commons Attribution 4.0 International",
                    license_url="https://creativecommons.org/licenses/by/4.0/",
                    attribution="Dungeons & Dragons SRD 5.1",
                ),
                mechanics_tags={"resolution_family": "d20"},
                _actor_schema={"properties": {"ability_scores": {"type": "object"}}, "required": []},
                _check_schema={"properties": {"dc": {"type": "integer"}}, "required": []},
            ),
            StaticRuleAdapter(
                adapter_key=RPG_ADAPTER_PF2E,
                adapter_version=RPG_ADAPTER_VERSION_V1,
                display_name="Pathfinder 2e",
                status="metadata_only",
                source_version="PF2e",
                bundled_snippet_policy="citations_only",
                license_summary=RuleLicenseSummary(
                    source_title="Pathfinder Second Edition",
                    source_url="https://2e.aonprd.com/",
                    license="ORC and Paizo Community Use references",
                    license_url="https://downloads.paizo.com/ORC_License_FINAL.pdf",
                    attribution="Pathfinder Second Edition rules references",
                ),
                mechanics_tags={"resolution_family": "d20"},
                _actor_schema={"properties": {"level": {"type": "integer"}}, "required": []},
                _check_schema={"properties": {"dc": {"type": "integer"}}, "required": []},
            ),
            StaticRuleAdapter(
                adapter_key=RPG_ADAPTER_FATE,
                adapter_version=RPG_ADAPTER_VERSION_V1,
                display_name="Fate",
                status="bundled",
                source_version="Fate SRD",
                bundled_snippet_policy="short_srd_citations",
                license_summary=RuleLicenseSummary(
                    source_title="Fate SRD",
                    source_url="https://fate-srd.com/",
                    license="Creative Commons Attribution 3.0 Unported",
                    license_url="https://creativecommons.org/licenses/by/3.0/",
                    attribution="Fate SRD",
                ),
                mechanics_tags={"resolution_family": "fate"},
                _actor_schema={
                    "properties": {"aspects": {"type": "array"}, "stress": {"type": "object"}},
                    "required": [],
                },
                _check_schema={"properties": {"ladder_target": {"type": "integer"}}, "required": []},
            ),
        ]
    )
```

- [ ] **Step 5: Export the public package symbols**

```python
# tldw_Server_API/app/core/RPG/__init__.py
from .models import RPGCampaign, RPGSession, RPGSessionEvent, RPGSnapshotState

__all__ = ["RPGCampaign", "RPGSession", "RPGSessionEvent", "RPGSnapshotState"]
```

Task 5 adds `RPGService` to these exports after `service.py` exists.

- [ ] **Step 6: Run focused tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rules_adapters.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/RPG tldw_Server_API/tests/RPG/test_rules_adapters.py
git commit -m "feat: add RPG rules adapter registry"
```

### Task 2: ChaChaNotes-Backed RPG Repository

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/RPG_DB.py`
- Test: `tldw_Server_API/tests/RPG/test_rpg_db.py`

- [ ] **Step 1: Write failing repository tests**

```python
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository


def test_repository_creates_campaign_session_and_initial_snapshot():
    db = CharactersRAGDB(":memory:", "test-client")
    repo = RPGRepository.initialized(db)

    campaign = repo.create_campaign(
        owner_user_id=42,
        title="Saltmarsh",
        description="Coastal trouble",
        default_adapter_key="dnd5e_srd",
        default_adapter_version="1.0.0",
        settings={},
        linked_rules_pack_refs=[],
    )
    session = repo.create_session(
        owner_user_id=42,
        campaign_id=campaign.id,
        title="Session 1",
        adapter_key="dnd5e_srd",
        adapter_version="1.0.0",
        authority_settings={"model_auto_commit": False},
        linked_chat_id=None,
        active_rules_pack_refs=[],
    )

    assert session.campaign_id == campaign.id
    assert session.last_event_sequence == 0
    assert repo.get_latest_snapshot(owner_user_id=42, session_id=session.id).snapshot_version == 0


def test_append_events_assigns_contiguous_sequence_numbers():
    db = CharactersRAGDB(":memory:", "test-client")
    repo = RPGRepository.initialized(db)
    campaign = repo.create_campaign(42, "Campaign", None, "fate", "1.0.0", {}, [])
    session = repo.create_session(42, campaign.id, "Opening", "fate", "1.0.0", {}, None, [])

    result = repo.append_session_events(
        owner_user_id=42,
        session_id=session.id,
        events=[
            {
                "event_type": "scene.updated",
                "event_payload": {"scene_id": "scene-start", "summary": "At the docks"},
                "source_type": "user",
            },
            {
                "event_type": "note.added",
                "event_payload": {"note_id": "note-1", "text": "Storm clouds gather"},
                "source_type": "user",
            },
        ],
        idempotency_key="req-1",
        request_payload_hash="hash-a",
        adapter_key="fate",
        adapter_version="1.0.0",
        proposal_id=None,
    )

    assert [event.sequence_number for event in result.events] == [1, 2]
    assert repo.get_session(owner_user_id=42, session_id=session.id).last_event_sequence == 2


def test_append_events_replays_same_idempotency_key_with_same_hash():
    db = CharactersRAGDB(":memory:", "test-client")
    repo = RPGRepository.initialized(db)
    campaign = repo.create_campaign(42, "Campaign", None, "fate", "1.0.0", {}, [])
    session = repo.create_session(42, campaign.id, "Opening", "fate", "1.0.0", {}, None, [])
    payload = [{"event_type": "note.added", "event_payload": {"note_id": "n1", "text": "A"}, "source_type": "user"}]

    first = repo.append_session_events(42, session.id, payload, "same-key", "hash-a", "fate", "1.0.0", None)
    second = repo.append_session_events(42, session.id, payload, "same-key", "hash-a", "fate", "1.0.0", None)

    assert [event.id for event in second.events] == [event.id for event in first.events]


def test_append_events_rejects_same_idempotency_key_with_different_hash():
    db = CharactersRAGDB(":memory:", "test-client")
    repo = RPGRepository.initialized(db)
    campaign = repo.create_campaign(42, "Campaign", None, "fate", "1.0.0", {}, [])
    session = repo.create_session(42, campaign.id, "Opening", "fate", "1.0.0", {}, None, [])
    payload = [{"event_type": "note.added", "event_payload": {"note_id": "n1", "text": "A"}, "source_type": "user"}]

    repo.append_session_events(42, session.id, payload, "same-key", "hash-a", "fate", "1.0.0", None)

    with pytest.raises(Exception, match="idempotency"):
        repo.append_session_events(42, session.id, payload, "same-key", "hash-b", "fate", "1.0.0", None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_db.py -v`

Expected: FAIL because `RPGRepository` does not exist.

- [ ] **Step 3: Implement schema initialization and row mapping**

```python
# tldw_Server_API/app/core/DB_Management/RPG_DB.py
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.RPG.constants import RPG_REDUCER_VERSION, RPG_SNAPSHOT_SCHEMA_VERSION
from tldw_Server_API.app.core.RPG.errors import RPGConflictError, RPGNotFoundError
from tldw_Server_API.app.core.RPG.models import RPGCampaign, RPGSession, RPGSessionEvent


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


@dataclass(frozen=True, slots=True)
class AppendEventsResult:
    events: list[RPGSessionEvent]
    replayed: bool


class RPGRepository:
    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    @classmethod
    def initialized(cls, db: CharactersRAGDB) -> "RPGRepository":
        repo = cls(db)
        repo.ensure_schema()
        return repo

    def ensure_schema(self) -> None:
        conn = self.db.get_connection()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS rpg_campaigns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                default_adapter_key TEXT NOT NULL,
                default_adapter_version TEXT NOT NULL,
                settings_json TEXT NOT NULL DEFAULT '{}',
                linked_rules_pack_refs_json TEXT NOT NULL DEFAULT '[]',
                version INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rpg_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id INTEGER NOT NULL,
                owner_user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                adapter_key TEXT NOT NULL,
                adapter_version TEXT NOT NULL,
                authority_settings_json TEXT NOT NULL DEFAULT '{}',
                linked_chat_id INTEGER,
                active_rules_pack_refs_json TEXT NOT NULL DEFAULT '[]',
                current_snapshot_version INTEGER NOT NULL DEFAULT 0,
                last_event_sequence INTEGER NOT NULL DEFAULT 0,
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rpg_session_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                owner_user_id INTEGER NOT NULL,
                sequence_number INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                event_payload_json TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_actor_id TEXT,
                source_label TEXT,
                idempotency_key TEXT,
                request_payload_hash TEXT,
                event_schema_version TEXT NOT NULL,
                adapter_key TEXT NOT NULL,
                adapter_version TEXT NOT NULL,
                proposal_id INTEGER,
                created_at TEXT NOT NULL,
                UNIQUE(owner_user_id, session_id, sequence_number),
                UNIQUE(owner_user_id, session_id, source_type, idempotency_key)
            );
            CREATE TABLE IF NOT EXISTS rpg_session_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                owner_user_id INTEGER NOT NULL,
                snapshot_version INTEGER NOT NULL,
                last_event_sequence INTEGER NOT NULL,
                reducer_version TEXT NOT NULL,
                snapshot_schema_version TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                diagnostics_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(owner_user_id, session_id, snapshot_version)
            );
            CREATE TABLE IF NOT EXISTS rpg_session_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                owner_user_id INTEGER NOT NULL,
                base_event_sequence INTEGER NOT NULL,
                base_snapshot_version INTEGER NOT NULL,
                proposed_events_json TEXT NOT NULL,
                patch_json TEXT,
                rationale TEXT,
                confidence REAL,
                source_type TEXT NOT NULL,
                source_actor_id TEXT,
                model_metadata_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                review_notes TEXT,
                created_at TEXT NOT NULL,
                applied_at TEXT,
                rejected_at TEXT
            );
            """
        )
        conn.commit()
        logger.debug("RPG repository schema ensured")
```

Add these repository method signatures and implement each one with parameterized SQL, JSON helpers using `json.dumps(value, sort_keys=True, separators=(",", ":"))`, and row mappers returning dataclasses from `models.py`.

```python
def create_campaign(
    self,
    owner_user_id: int,
    title: str,
    description: str | None,
    default_adapter_key: str,
    default_adapter_version: str,
    settings: dict[str, Any],
    linked_rules_pack_refs: list[dict[str, Any]],
) -> RPGCampaign:
    raise NotImplementedError

def create_session(
    self,
    owner_user_id: int,
    campaign_id: int,
    title: str,
    adapter_key: str,
    adapter_version: str,
    authority_settings: dict[str, Any],
    linked_chat_id: int | None,
    active_rules_pack_refs: list[dict[str, Any]],
) -> RPGSession:
    raise NotImplementedError

def get_session(self, owner_user_id: int, session_id: int) -> RPGSession:
    raise NotImplementedError

def get_event(self, owner_user_id: int, event_id: int) -> RPGSessionEvent:
    raise NotImplementedError

def get_latest_snapshot(self, owner_user_id: int, session_id: int) -> RPGSnapshotRecord:
    raise NotImplementedError
```

- [ ] **Step 4: Add idempotent append behavior**

```python
def append_session_events(
    self,
    owner_user_id: int,
    session_id: int,
    events: list[dict[str, Any]],
    idempotency_key: str | None,
    request_payload_hash: str | None,
    adapter_key: str,
    adapter_version: str,
    proposal_id: int | None,
) -> AppendEventsResult:
    if idempotency_key:
        replay = self._find_events_by_idempotency_key(owner_user_id, session_id, events[0]["source_type"], idempotency_key)
        if replay:
            if replay[0].request_payload_hash != request_payload_hash:
                raise RPGConflictError("idempotency_key_conflict")
            return AppendEventsResult(events=replay, replayed=True)

    conn = self.db.get_connection()
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute("BEGIN")
        session = self.get_session(owner_user_id=owner_user_id, session_id=session_id)
        next_sequence = session.last_event_sequence + 1
        inserted: list[RPGSessionEvent] = []
        for offset, event in enumerate(events):
            sequence = next_sequence + offset
            cursor = conn.execute(
                """
                INSERT INTO rpg_session_events (
                    session_id, owner_user_id, sequence_number, event_type, event_payload_json,
                    source_type, source_actor_id, source_label, idempotency_key, request_payload_hash,
                    event_schema_version, adapter_key, adapter_version, proposal_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    owner_user_id,
                    sequence,
                    event["event_type"],
                    self._to_json(event["event_payload"]),
                    event["source_type"],
                    event.get("source_actor_id"),
                    event.get("source_label"),
                    idempotency_key,
                    request_payload_hash,
                    event.get("event_schema_version", "1.0.0"),
                    adapter_key,
                    adapter_version,
                    proposal_id,
                    now,
                ),
            )
            inserted.append(self.get_event(owner_user_id=owner_user_id, event_id=int(cursor.lastrowid)))
        conn.execute(
            "UPDATE rpg_sessions SET last_event_sequence = ?, version = version + 1, updated_at = ? WHERE id = ? AND owner_user_id = ?",
            (inserted[-1].sequence_number, now, session_id, owner_user_id),
        )
        conn.commit()
        return AppendEventsResult(events=inserted, replayed=False)
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise RPGConflictError("rpg_event_append_conflict") from exc
    except Exception:
        conn.rollback()
        raise
```

- [ ] **Step 5: Run focused repository tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_db.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/RPG_DB.py tldw_Server_API/tests/RPG/test_rpg_db.py
git commit -m "feat: add RPG session repository"
```

### Task 3: Event Validation And Deterministic Reducer

**Files:**
- Create: `tldw_Server_API/app/core/RPG/events.py`
- Create: `tldw_Server_API/app/core/RPG/reducer.py`
- Modify: `tldw_Server_API/app/core/RPG/models.py`
- Test: `tldw_Server_API/tests/RPG/test_rpg_events_reducer.py`

- [ ] **Step 1: Write failing event and reducer tests**

```python
from tldw_Server_API.app.core.RPG.events import canonical_request_hash, validate_event_envelope
from tldw_Server_API.app.core.RPG.reducer import initial_snapshot, reduce_events


def test_canonical_request_hash_is_stable_for_key_order():
    left = {"events": [{"event_type": "note.added", "event_payload": {"text": "A", "note_id": "n1"}}]}
    right = {"events": [{"event_payload": {"note_id": "n1", "text": "A"}, "event_type": "note.added"}]}

    assert canonical_request_hash(left) == canonical_request_hash(right)


def test_validate_event_envelope_rejects_missing_stable_ids():
    event = {"event_type": "npc.upserted", "event_payload": {"name": "Ada"}, "source_type": "user"}

    with pytest.raises(ValueError, match="npc_id"):
        validate_event_envelope(event)


def test_reducer_rebuilds_same_snapshot_from_same_events():
    events = [
        {"event_type": "scene.updated", "event_payload": {"scene_id": "s1", "summary": "Rainy docks"}, "source_type": "user"},
        {"event_type": "npc.upserted", "event_payload": {"npc_id": "npc-ada", "name": "Ada"}, "source_type": "user"},
        {"event_type": "quest.upserted", "event_payload": {"quest_id": "q1", "title": "Find the map"}, "source_type": "user"},
    ]

    first = reduce_events(initial_snapshot(), events)
    second = reduce_events(initial_snapshot(), events)

    assert first == second
    assert first.scene["summary"] == "Rainy docks"
    assert first.npcs["npc-ada"]["name"] == "Ada"
    assert first.quests["q1"]["title"] == "Find the map"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_events_reducer.py -v`

Expected: FAIL because event validation and reducer modules do not exist.

- [ ] **Step 3: Implement canonical hashes and payload validators**

```python
# tldw_Server_API/app/core/RPG/events.py
from __future__ import annotations

import hashlib
import json
from typing import Any

from tldw_Server_API.app.core.RPG.constants import MAX_RPG_EVENT_PAYLOAD_BYTES, RPG_EVENT_SCHEMA_VERSION, RPG_SOURCE_TYPES

_REQUIRED_EVENT_IDS = {
    "actor.upserted": "actor_id",
    "npc.upserted": "npc_id",
    "quest.upserted": "quest_id",
    "inventory.item.upserted": "item_id",
    "location.upserted": "location_id",
    "faction.upserted": "faction_id",
    "clock.updated": "clock_id",
    "ruling.added": "ruling_id",
    "note.added": "note_id",
    "roll.recorded": "roll_id",
    "rule.reference.added": "reference_id",
    "scene.updated": "scene_id",
}


def canonical_request_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_event_envelope(event: dict[str, Any]) -> dict[str, Any]:
    event_type = str(event.get("event_type") or "").strip()
    source_type = str(event.get("source_type") or "").strip()
    payload = event.get("event_payload")
    if not event_type:
        raise ValueError("event_type is required")
    if source_type not in RPG_SOURCE_TYPES:
        raise ValueError("source_type is invalid")
    if not isinstance(payload, dict):
        raise ValueError("event_payload must be an object")
    encoded_size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    if encoded_size > MAX_RPG_EVENT_PAYLOAD_BYTES:
        raise ValueError("event_payload is too large")
    required_id = _REQUIRED_EVENT_IDS.get(event_type)
    if required_id and not str(payload.get(required_id) or "").strip():
        raise ValueError(f"{required_id} is required for {event_type}")
    normalized = dict(event)
    normalized["event_type"] = event_type
    normalized["source_type"] = source_type
    normalized["event_payload"] = dict(payload)
    normalized.setdefault("event_schema_version", RPG_EVENT_SCHEMA_VERSION)
    return normalized
```

- [ ] **Step 4: Implement pure reducer functions**

```python
# tldw_Server_API/app/core/RPG/reducer.py
from __future__ import annotations

from dataclasses import replace
from typing import Any

from tldw_Server_API.app.core.RPG.models import RPGSnapshotState


def initial_snapshot() -> RPGSnapshotState:
    return RPGSnapshotState()


def reduce_event(snapshot: RPGSnapshotState, event: dict[str, Any]) -> RPGSnapshotState:
    event_type = event["event_type"]
    payload = event["event_payload"]
    if event_type == "scene.updated":
        return replace(snapshot, scene={**snapshot.scene, **payload})
    if event_type == "actor.upserted":
        actors = dict(snapshot.actors)
        actors[payload["actor_id"]] = {**actors.get(payload["actor_id"], {}), **payload}
        return replace(snapshot, actors=actors)
    if event_type == "npc.upserted":
        npcs = dict(snapshot.npcs)
        npcs[payload["npc_id"]] = {**npcs.get(payload["npc_id"], {}), **payload}
        return replace(snapshot, npcs=npcs)
    if event_type == "quest.upserted":
        quests = dict(snapshot.quests)
        quests[payload["quest_id"]] = {**quests.get(payload["quest_id"], {}), **payload}
        return replace(snapshot, quests=quests)
    if event_type == "note.added":
        return replace(snapshot, notes=[*snapshot.notes, dict(payload)])
    if event_type == "rule.reference.added":
        return replace(snapshot, rules_references=[*snapshot.rules_references, dict(payload)])
    if event_type == "ruling.added":
        rulings = dict(snapshot.unresolved_rulings)
        rulings[payload["ruling_id"]] = dict(payload)
        return replace(snapshot, unresolved_rulings=rulings)
    raise ValueError(f"Unsupported RPG event type: {event_type}")


def reduce_events(snapshot: RPGSnapshotState, events: list[dict[str, Any]]) -> RPGSnapshotState:
    current = snapshot
    for event in events:
        current = reduce_event(current, event)
    return current
```

- [ ] **Step 5: Run focused reducer tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_events_reducer.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/RPG/events.py tldw_Server_API/app/core/RPG/reducer.py tldw_Server_API/app/core/RPG/models.py tldw_Server_API/tests/RPG/test_rpg_events_reducer.py
git commit -m "feat: add RPG event reducer"
```

### Task 4: Dice And Check Resolution

**Files:**
- Create: `tldw_Server_API/app/core/RPG/dice.py`
- Create: `tldw_Server_API/app/core/RPG/checks.py`
- Modify: `tldw_Server_API/app/core/RPG/models.py`
- Test: `tldw_Server_API/tests/RPG/test_rpg_dice_checks.py`

- [ ] **Step 1: Write failing dice and check tests**

```python
from tldw_Server_API.app.core.RPG.checks import resolve_check
from tldw_Server_API.app.core.RPG.dice import DiceRoller
from tldw_Server_API.app.core.RPG.rules.adapters import build_default_adapter_registry


def test_dice_roller_supports_basic_dice_and_modifier():
    roller = DiceRoller(randbelow=lambda sides: 3)

    result = roller.roll("2d6+1")

    assert result.total == 9
    assert result.rolls == [4, 4]
    assert result.modifier == 1


def test_dnd_check_uses_d20_total_against_dc():
    registry = build_default_adapter_registry()
    roller = DiceRoller(randbelow=lambda sides: 14)

    result = resolve_check(
        adapter=registry.get("dnd5e_srd"),
        roller=roller,
        payload={"roll_expression": "1d20+5", "dc": 18, "check_label": "Strength"},
    )

    assert result.total == 20
    assert result.outcome == "success"


def test_fate_check_uses_fate_dice_total_against_ladder_target():
    registry = build_default_adapter_registry()
    roller = DiceRoller(fate_values=[-1, 0, 1, 1])

    result = resolve_check(
        adapter=registry.get("fate"),
        roller=roller,
        payload={"skill_bonus": 2, "ladder_target": 3, "check_label": "Careful"},
    )

    assert result.total == 3
    assert result.outcome == "tie"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_dice_checks.py -v`

Expected: FAIL because dice and check modules do not exist.

- [ ] **Step 3: Add dice result dataclass and roller**

```python
# tldw_Server_API/app/core/RPG/models.py
@dataclass(frozen=True, slots=True)
class DiceRollResult:
    expression: str
    rolls: list[int]
    modifier: int
    total: int
    kept: list[int]


@dataclass(frozen=True, slots=True)
class CheckResult:
    adapter_key: str
    check_label: str
    total: int
    target: int | None
    outcome: str
    roll: DiceRollResult | None
    details: dict[str, Any]
```

```python
# tldw_Server_API/app/core/RPG/dice.py
from __future__ import annotations

import re
import secrets
from collections.abc import Callable

from tldw_Server_API.app.core.RPG.models import DiceRollResult

_DICE_RE = re.compile(r"^(?P<count>[1-9][0-9]?)d(?P<sides>[1-9][0-9]*)(?P<mod>[+-][0-9]+)?$")


class DiceRoller:
    def __init__(
        self,
        randbelow: Callable[[int], int] | None = None,
        fate_values: list[int] | None = None,
    ) -> None:
        self._randbelow = randbelow or secrets.randbelow
        self._fate_values = list(fate_values or [])

    def roll(self, expression: str) -> DiceRollResult:
        match = _DICE_RE.match(expression.strip().lower())
        if not match:
            raise ValueError("Unsupported dice expression")
        count = int(match.group("count"))
        sides = int(match.group("sides"))
        modifier = int(match.group("mod") or 0)
        rolls = [self._randbelow(sides) + 1 for _ in range(count)]
        total = sum(rolls) + modifier
        return DiceRollResult(expression=expression, rolls=rolls, modifier=modifier, total=total, kept=rolls)

    def roll_fate(self) -> list[int]:
        if self._fate_values:
            values = self._fate_values[:4]
            if len(values) != 4:
                raise ValueError("Fate checks require four fate dice")
            return values
        return [self._randbelow(3) - 1 for _ in range(4)]
```

- [ ] **Step 4: Add adapter-aware check resolver**

```python
# tldw_Server_API/app/core/RPG/checks.py
from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.RPG.dice import DiceRoller
from tldw_Server_API.app.core.RPG.models import CheckResult, DiceRollResult
from tldw_Server_API.app.core.RPG.rules.adapters import RuleAdapter


def _d20_outcome(total: int, dc: int | None) -> str:
    if dc is None:
        return "rolled"
    return "success" if total >= dc else "failure"


def _fate_outcome(total: int, target: int) -> str:
    if total < target:
        return "failure"
    if total == target:
        return "tie"
    if total >= target + 3:
        return "success_with_style"
    return "success"


def resolve_check(adapter: RuleAdapter, roller: DiceRoller, payload: dict[str, Any]) -> CheckResult:
    label = str(payload.get("check_label") or "Check")
    if adapter.mechanics_tags.get("resolution_family") == "fate":
        target = int(payload["ladder_target"])
        fate_total = sum(roller.roll_fate()) + int(payload.get("skill_bonus") or 0)
        return CheckResult(
            adapter_key=adapter.adapter_key,
            check_label=label,
            total=fate_total,
            target=target,
            outcome=_fate_outcome(fate_total, target),
            roll=None,
            details={"resolution_family": "fate"},
        )
    roll: DiceRollResult = roller.roll(str(payload.get("roll_expression") or "1d20"))
    dc = payload.get("dc")
    target = int(dc) if dc is not None else None
    return CheckResult(
        adapter_key=adapter.adapter_key,
        check_label=label,
        total=roll.total,
        target=target,
        outcome=_d20_outcome(roll.total, target),
        roll=roll,
        details={"resolution_family": "d20"},
    )
```

- [ ] **Step 5: Run focused dice/check tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_dice_checks.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/RPG/dice.py tldw_Server_API/app/core/RPG/checks.py tldw_Server_API/app/core/RPG/models.py tldw_Server_API/tests/RPG/test_rpg_dice_checks.py
git commit -m "feat: add RPG dice and check resolution"
```

### Task 5: RPG Service, Authority Policy, And Proposals

**Files:**
- Create: `tldw_Server_API/app/core/RPG/authority.py`
- Create: `tldw_Server_API/app/core/RPG/proposals.py`
- Create: `tldw_Server_API/app/core/RPG/service.py`
- Modify: `tldw_Server_API/app/core/RPG/__init__.py`
- Modify: `tldw_Server_API/app/core/DB_Management/RPG_DB.py`
- Test: `tldw_Server_API/tests/RPG/test_rpg_service.py`

- [ ] **Step 1: Write failing service authority tests**

```python
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RPG.service import RPGService


def _service():
    repo = RPGRepository.initialized(CharactersRAGDB(":memory:", "test-client"))
    return RPGService(repo=repo, owner_user_id=42)


def test_model_events_create_pending_proposal_by_default():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate")
    session = service.create_session(campaign.id, "Opening", adapter_key="fate")

    result = service.record_events(
        session_id=session.id,
        events=[{"event_type": "note.added", "event_payload": {"note_id": "n1", "text": "Suggested"}, "source_type": "model"}],
        idempotency_key="model-1",
    )

    assert result.committed_events == []
    assert result.proposal is not None
    assert result.proposal.status == "pending"


def test_user_events_commit_and_update_snapshot():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate")
    session = service.create_session(campaign.id, "Opening", adapter_key="fate")

    result = service.record_events(
        session_id=session.id,
        events=[{"event_type": "note.added", "event_payload": {"note_id": "n1", "text": "Observed"}, "source_type": "user"}],
        idempotency_key="user-1",
    )

    assert [event.sequence_number for event in result.committed_events] == [1]
    assert service.get_snapshot(session.id).snapshot.notes[0]["text"] == "Observed"


def test_apply_proposal_is_atomic_and_advances_snapshot_once():
    service = _service()
    campaign = service.create_campaign("Campaign", None, "fate")
    session = service.create_session(campaign.id, "Opening", adapter_key="fate")
    proposed = service.record_events(
        session_id=session.id,
        events=[
            {"event_type": "npc.upserted", "event_payload": {"npc_id": "npc-1", "name": "Ada"}, "source_type": "model"},
            {"event_type": "quest.upserted", "event_payload": {"quest_id": "q1", "title": "Find Ada"}, "source_type": "model"},
        ],
        idempotency_key="model-2",
    ).proposal

    applied = service.apply_proposal(session_id=session.id, proposal_id=proposed.id, review_notes="accepted")

    assert [event.sequence_number for event in applied.committed_events] == [1, 2]
    snapshot = service.get_snapshot(session.id).snapshot
    assert snapshot.npcs["npc-1"]["name"] == "Ada"
    assert snapshot.quests["q1"]["title"] == "Find Ada"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_service.py -v`

Expected: FAIL because `RPGService` does not exist.

- [ ] **Step 3: Implement authority decisions**

```python
# tldw_Server_API/app/core/RPG/authority.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuthorityDecision:
    action: str
    reason: str


def decide_authority(source_type: str, authority_settings: dict[str, object]) -> AuthorityDecision:
    if source_type in {"user", "system", "import"}:
        return AuthorityDecision(action="commit", reason="trusted_source")
    if source_type == "mcp" and authority_settings.get("mcp_auto_commit") is True:
        return AuthorityDecision(action="commit", reason="mcp_auto_commit_enabled")
    if source_type == "model" and authority_settings.get("model_auto_commit") is True:
        return AuthorityDecision(action="commit", reason="model_auto_commit_enabled")
    return AuthorityDecision(action="proposal", reason=f"{source_type}_requires_review")
```

- [ ] **Step 4: Implement service result types and orchestration**

```python
# tldw_Server_API/app/core/RPG/service.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RPG.authority import decide_authority
from tldw_Server_API.app.core.RPG.events import canonical_request_hash, validate_event_envelope
from tldw_Server_API.app.core.RPG.models import RPGCampaign, RPGSession, RPGSessionEvent, RPGSnapshotState
from tldw_Server_API.app.core.RPG.reducer import reduce_events
from tldw_Server_API.app.core.RPG.rules.adapters import RuleAdapterRegistry, build_default_adapter_registry


@dataclass(frozen=True, slots=True)
class RPGServiceProposal:
    id: int
    session_id: int
    status: str
    proposed_events: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RecordEventsResult:
    committed_events: list[RPGSessionEvent]
    proposal: RPGServiceProposal | None


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    snapshot_version: int
    last_event_sequence: int
    snapshot: RPGSnapshotState
    diagnostics: dict[str, Any]


class RPGService:
    def __init__(
        self,
        repo: RPGRepository,
        owner_user_id: int,
        adapter_registry: RuleAdapterRegistry | None = None,
    ) -> None:
        self.repo = repo
        self.owner_user_id = owner_user_id
        self.adapter_registry = adapter_registry or build_default_adapter_registry()

    def create_campaign(self, title: str, description: str | None, default_adapter_key: str) -> RPGCampaign:
        adapter = self.adapter_registry.get(default_adapter_key)
        return self.repo.create_campaign(
            owner_user_id=self.owner_user_id,
            title=title,
            description=description,
            default_adapter_key=adapter.adapter_key,
            default_adapter_version=adapter.adapter_version,
            settings={},
            linked_rules_pack_refs=[],
        )

    def create_session(self, campaign_id: int, title: str, adapter_key: str) -> RPGSession:
        adapter = self.adapter_registry.get(adapter_key)
        return self.repo.create_session(
            owner_user_id=self.owner_user_id,
            campaign_id=campaign_id,
            title=title,
            adapter_key=adapter.adapter_key,
            adapter_version=adapter.adapter_version,
            authority_settings={"model_auto_commit": False, "mcp_auto_commit": False},
            linked_chat_id=None,
            active_rules_pack_refs=[],
        )

    def record_events(self, session_id: int, events: list[dict[str, Any]], idempotency_key: str | None) -> RecordEventsResult:
        session = self.repo.get_session(owner_user_id=self.owner_user_id, session_id=session_id)
        normalized = [validate_event_envelope(event) for event in events]
        source_type = normalized[0]["source_type"]
        decision = decide_authority(source_type, session.authority_settings)
        request_hash = canonical_request_hash({"events": normalized})
        if decision.action == "proposal":
            proposal = self.repo.create_proposal(
                owner_user_id=self.owner_user_id,
                session_id=session_id,
                base_event_sequence=session.last_event_sequence,
                base_snapshot_version=session.current_snapshot_version,
                proposed_events=normalized,
                source_type=source_type,
                source_actor_id=normalized[0].get("source_actor_id"),
                model_metadata={},
            )
            return RecordEventsResult(committed_events=[], proposal=RPGServiceProposal(proposal.id, session_id, proposal.status, normalized))
        appended = self.repo.append_session_events(
            owner_user_id=self.owner_user_id,
            session_id=session_id,
            events=normalized,
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            adapter_key=session.adapter_key,
            adapter_version=session.adapter_version,
            proposal_id=None,
        )
        self._rebuild_and_store_snapshot(session_id=session_id, events=appended.events)
        return RecordEventsResult(committed_events=appended.events, proposal=None)
```

Add these service methods in the same file:

```python
def apply_proposal(self, session_id: int, proposal_id: int, review_notes: str | None = None) -> RecordEventsResult:
    session = self.repo.get_session(owner_user_id=self.owner_user_id, session_id=session_id)
    proposal = self.repo.get_proposal(owner_user_id=self.owner_user_id, proposal_id=proposal_id)
    if proposal.status != "pending":
        raise RPGConflictError("proposal_not_pending")
    if proposal.base_event_sequence != session.last_event_sequence:
        self.repo.mark_proposal_conflicted(self.owner_user_id, proposal_id)
        raise RPGConflictError("proposal_base_sequence_conflict")
    normalized = [validate_event_envelope(event) for event in proposal.proposed_events]
    request_hash = canonical_request_hash({"proposal_id": proposal_id, "events": normalized})
    appended = self.repo.append_session_events(
        owner_user_id=self.owner_user_id,
        session_id=session_id,
        events=normalized,
        idempotency_key=f"proposal:{proposal_id}:apply",
        request_payload_hash=request_hash,
        adapter_key=session.adapter_key,
        adapter_version=session.adapter_version,
        proposal_id=proposal_id,
    )
    self._rebuild_and_store_snapshot(session_id=session_id, events=appended.events)
    self.repo.mark_proposal_applied(self.owner_user_id, proposal_id, review_notes)
    return RecordEventsResult(committed_events=appended.events, proposal=None)


def reject_proposal(self, session_id: int, proposal_id: int, review_notes: str | None = None) -> RPGServiceProposal:
    self.repo.get_session(owner_user_id=self.owner_user_id, session_id=session_id)
    proposal = self.repo.mark_proposal_rejected(self.owner_user_id, proposal_id, review_notes)
    return RPGServiceProposal(proposal.id, session_id, proposal.status, proposal.proposed_events)


def get_snapshot(self, session_id: int) -> SnapshotResult:
    record = self.repo.get_latest_snapshot(owner_user_id=self.owner_user_id, session_id=session_id)
    return SnapshotResult(
        snapshot_version=record.snapshot_version,
        last_event_sequence=record.last_event_sequence,
        snapshot=RPGSnapshotState(**record.snapshot_json),
        diagnostics=record.diagnostics_json,
    )


def _rebuild_and_store_snapshot(self, session_id: int, events: list[RPGSessionEvent]) -> None:
    current = self.repo.get_latest_snapshot(owner_user_id=self.owner_user_id, session_id=session_id)
    event_payloads = [
        {"event_type": event.event_type, "event_payload": event.event_payload, "source_type": event.source_type}
        for event in events
    ]
    next_snapshot = reduce_events(RPGSnapshotState(**current.snapshot_json), event_payloads)
    self.repo.save_snapshot(
        owner_user_id=self.owner_user_id,
        session_id=session_id,
        snapshot_version=current.snapshot_version + 1,
        last_event_sequence=events[-1].sequence_number,
        snapshot=asdict(next_snapshot),
        diagnostics={"applied_event_count": len(events)},
    )
```

- [ ] **Step 5: Add repository proposal and snapshot methods**

Add these methods to `RPGRepository`. Each method filters by `owner_user_id` and raises `RPGNotFoundError` when a row is absent.

```python
def create_proposal(
    self,
    owner_user_id: int,
    session_id: int,
    base_event_sequence: int,
    base_snapshot_version: int,
    proposed_events: list[dict[str, Any]],
    source_type: str,
    source_actor_id: str | None,
    model_metadata: dict[str, Any],
) -> RPGProposalRecord:
    raise NotImplementedError

def get_proposal(self, owner_user_id: int, proposal_id: int) -> RPGProposalRecord:
    raise NotImplementedError

def mark_proposal_applied(self, owner_user_id: int, proposal_id: int, review_notes: str | None) -> RPGProposalRecord:
    raise NotImplementedError

def mark_proposal_rejected(self, owner_user_id: int, proposal_id: int, review_notes: str | None) -> RPGProposalRecord:
    raise NotImplementedError

def mark_proposal_conflicted(self, owner_user_id: int, proposal_id: int) -> RPGProposalRecord:
    raise NotImplementedError

def save_snapshot(self, owner_user_id: int, session_id: int, snapshot_version: int, last_event_sequence: int, snapshot: dict[str, Any], diagnostics: dict[str, Any]) -> RPGSnapshotRecord:
    raise NotImplementedError
```

- [ ] **Step 6: Run focused service tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_service.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/RPG/authority.py tldw_Server_API/app/core/RPG/proposals.py tldw_Server_API/app/core/RPG/service.py tldw_Server_API/app/core/RPG/__init__.py tldw_Server_API/app/core/DB_Management/RPG_DB.py tldw_Server_API/tests/RPG/test_rpg_service.py
git commit -m "feat: add RPG service authority flow"
```

### Task 6: REST Schemas, Endpoints, And Router Registration

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/rpg_schemas.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/rpg.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Test: `tldw_Server_API/tests/RPG/test_rpg_api.py`

- [ ] **Step 1: Write failing API tests**

```python
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def test_rpg_adapters_endpoint_lists_default_adapters():
    client = TestClient(app)

    response = client.get("/api/v1/rpg/rules/adapters", headers={"X-API-KEY": "test-api-key-12345"})

    assert response.status_code == 200
    keys = [item["adapter_key"] for item in response.json()["adapters"]]
    assert keys == ["dnd5e_srd", "fate", "pf2e"]


def test_create_campaign_session_and_record_user_event():
    client = TestClient(app)
    headers = {"X-API-KEY": "test-api-key-12345", "Idempotency-Key": "api-event-1"}

    campaign = client.post(
        "/api/v1/rpg/campaigns",
        headers=headers,
        json={"title": "Campaign", "default_adapter_key": "fate"},
    )
    assert campaign.status_code == 201
    campaign_id = campaign.json()["id"]

    session = client.post(
        f"/api/v1/rpg/campaigns/{campaign_id}/sessions",
        headers=headers,
        json={"title": "Opening", "adapter_key": "fate"},
    )
    assert session.status_code == 201
    session_id = session.json()["id"]

    event_response = client.post(
        f"/api/v1/rpg/sessions/{session_id}/events",
        headers=headers,
        json={
            "events": [
                {"event_type": "note.added", "event_payload": {"note_id": "n1", "text": "At the docks"}, "source_type": "user"}
            ]
        },
    )

    assert event_response.status_code == 200
    assert event_response.json()["committed_events"][0]["sequence_number"] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_api.py -v`

Expected: FAIL because the RPG router is not registered.

- [ ] **Step 3: Add Pydantic schemas**

```python
# tldw_Server_API/app/api/v1/schemas/rpg_schemas.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RPGCampaignCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4000)
    default_adapter_key: str = Field(min_length=1, max_length=80)


class RPGCampaignResponse(BaseModel):
    id: int
    title: str
    description: str | None
    default_adapter_key: str
    default_adapter_version: str
    status: str
    version: int


class RPGSessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200)
    adapter_key: str = Field(min_length=1, max_length=80)
    linked_chat_id: int | None = None


class RPGSessionResponse(BaseModel):
    id: int
    campaign_id: int
    title: str
    status: str
    adapter_key: str
    adapter_version: str
    current_snapshot_version: int
    last_event_sequence: int
    version: int


class RPGEventInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_type: str = Field(min_length=1, max_length=120)
    event_payload: dict[str, Any]
    source_type: Literal["user", "system", "mcp", "model", "import"]
    source_actor_id: str | None = Field(default=None, max_length=120)
    source_label: str | None = Field(default=None, max_length=200)


class RPGRecordEventsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[RPGEventInput] = Field(min_length=1, max_length=20)


class RPGRecordEventsResponse(BaseModel):
    committed_events: list[dict[str, Any]]
    proposal: dict[str, Any] | None
```

- [ ] **Step 4: Add endpoints with token scope metadata**

```python
# tldw_Server_API/app/api/v1/endpoints/rpg.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import TokenScopeGuard, User, get_request_user, rbac_rate_limit
from tldw_Server_API.app.api.v1.schemas.rpg_schemas import (
    RPGCampaignCreateRequest,
    RPGCampaignResponse,
    RPGRecordEventsRequest,
    RPGRecordEventsResponse,
    RPGSessionCreateRequest,
    RPGSessionResponse,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RPG.errors import RPGConflictError, RPGNotFoundError, RPGValidationError
from tldw_Server_API.app.core.RPG.service import RPGService

router = APIRouter(prefix="/rpg", tags=["rpg"])


def _owner_user_id(current_user: User) -> int:
    if current_user.id_int is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_user_id")
    return current_user.id_int


def _service(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> RPGService:
    return RPGService(repo=RPGRepository.initialized(db), owner_user_id=_owner_user_id(current_user))


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, RPGNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, RPGConflictError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, (RPGValidationError, ValueError)):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="rpg_internal_error")


@router.get(
    "/rules/adapters",
    dependencies=[
        Depends(rbac_rate_limit("rpg.rules.read")),
        Depends(TokenScopeGuard("rpg", require_if_present=True, endpoint_id="rpg.rules.read")),
    ],
)
def list_adapters(service: RPGService = Depends(_service)) -> dict[str, object]:
    return {"adapters": [adapter.__dict__ for adapter in service.adapter_registry.list_infos()]}


@router.post(
    "/campaigns",
    response_model=RPGCampaignResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[
        Depends(rbac_rate_limit("rpg.campaigns.manage")),
        Depends(TokenScopeGuard("rpg", require_if_present=True, endpoint_id="rpg.campaigns.manage")),
    ],
)
def create_campaign(request: RPGCampaignCreateRequest, service: RPGService = Depends(_service)) -> RPGCampaignResponse:
    try:
        campaign = service.create_campaign(request.title, request.description, request.default_adapter_key)
        return RPGCampaignResponse.model_validate(campaign.__dict__)
    except Exception as exc:
        raise _http_error(exc) from exc
```

Add these route handlers in the same file with the same `_service` and `_http_error` helpers:

```python
@router.get("/campaigns", dependencies=[Depends(rbac_rate_limit("rpg.campaigns.read")), Depends(TokenScopeGuard("rpg", require_if_present=True, endpoint_id="rpg.campaigns.read"))])
def list_campaigns(service: RPGService = Depends(_service)) -> dict[str, object]:
    return {"campaigns": [campaign.__dict__ for campaign in service.list_campaigns()]}


@router.post("/campaigns/{campaign_id}/sessions", response_model=RPGSessionResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(rbac_rate_limit("rpg.sessions.manage")), Depends(TokenScopeGuard("rpg", require_if_present=True, endpoint_id="rpg.sessions.manage"))])
def create_session(campaign_id: int, request: RPGSessionCreateRequest, service: RPGService = Depends(_service)) -> RPGSessionResponse:
    session = service.create_session(campaign_id=campaign_id, title=request.title, adapter_key=request.adapter_key)
    return RPGSessionResponse.model_validate(session.__dict__)


@router.get("/sessions/{session_id}", dependencies=[Depends(rbac_rate_limit("rpg.sessions.read")), Depends(TokenScopeGuard("rpg", require_if_present=True, endpoint_id="rpg.sessions.read"))])
def get_session(session_id: int, service: RPGService = Depends(_service)) -> dict[str, object]:
    return service.get_session_payload(session_id)


@router.get("/sessions/{session_id}/events", dependencies=[Depends(rbac_rate_limit("rpg.sessions.read")), Depends(TokenScopeGuard("rpg", require_if_present=True, endpoint_id="rpg.sessions.read"))])
def list_events(session_id: int, service: RPGService = Depends(_service)) -> dict[str, object]:
    return {"events": [event.__dict__ for event in service.list_events(session_id)]}


@router.post("/sessions/{session_id}/events", response_model=RPGRecordEventsResponse, dependencies=[Depends(rbac_rate_limit("rpg.sessions.manage")), Depends(TokenScopeGuard("rpg", require_if_present=True, endpoint_id="rpg.sessions.manage"))])
def record_events(session_id: int, request: RPGRecordEventsRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"), service: RPGService = Depends(_service)) -> RPGRecordEventsResponse:
    result = service.record_events(session_id=session_id, events=[event.model_dump() for event in request.events], idempotency_key=idempotency_key)
    return RPGRecordEventsResponse(committed_events=[event.__dict__ for event in result.committed_events], proposal=result.proposal.__dict__ if result.proposal else None)


@router.post("/sessions/{session_id}/rules/lookup", dependencies=[Depends(rbac_rate_limit("rpg.rules.read")), Depends(TokenScopeGuard("rpg", require_if_present=True, endpoint_id="rpg.rules.read"))])
def lookup_rules(session_id: int, request: RPGRulesLookupRequest, service: RPGService = Depends(_service)) -> dict[str, object]:
    return service.lookup_rules(session_id=session_id, query=request.query).__dict__


@router.post("/sessions/{session_id}/context", dependencies=[Depends(rbac_rate_limit("rpg.sessions.read")), Depends(TokenScopeGuard("rpg", require_if_present=True, endpoint_id="rpg.sessions.read"))])
def build_context(session_id: int, request: RPGContextBuildRequest, service: RPGService = Depends(_service)) -> dict[str, object]:
    return service.build_context(session_id=session_id, query=request.query, max_chars=request.max_chars).__dict__
```

Add `POST /sessions/{session_id}/rolls`, `POST /sessions/{session_id}/proposals`, `POST /sessions/{session_id}/proposals/{proposal_id}/apply`, and `POST /sessions/{session_id}/proposals/{proposal_id}/reject` with the same dependency style and endpoint IDs `rpg.sessions.manage`, `rpg.sessions.manage`, `rpg.proposals.review`, and `rpg.proposals.review`.

- [ ] **Step 5: Register the router**

```python
# tldw_Server_API/app/api/v1/router_groups/content.py
ImportedRouterSpec(
    import_path="tldw_Server_API.app.api.v1.endpoints.rpg",
    log_name="rpg",
    prefix=f"{API_V1_PREFIX}",
    tags=("rpg",),
    route_key="rpg",
)
```

Add this spec near adjacent content runtimes, before the VN route group tail.

- [ ] **Step 6: Run focused API tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_api.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/rpg_schemas.py tldw_Server_API/app/api/v1/endpoints/rpg.py tldw_Server_API/app/api/v1/router_groups/content.py tldw_Server_API/tests/RPG/test_rpg_api.py
git commit -m "feat: expose RPG REST runtime"
```

### Task 7: RPG Privilege Catalog And Route Snapshot

**Files:**
- Modify: `tldw_Server_API/Config_Files/privilege_catalog.yaml`
- Modify: `tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json`
- Test: `tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py`

- [ ] **Step 1: Run privilege sync tests before catalog edits**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py -v`

Expected: FAIL with missing `rpg.*` endpoint IDs after Task 6 is complete.

- [ ] **Step 2: Add RPG scopes and endpoint IDs to the catalog**

Add these entries under `scopes:` in `tldw_Server_API/Config_Files/privilege_catalog.yaml`:

```yaml
  - id: rpg
    description: Generic token scope bucket for RPG campaign and session APIs.
    resource_tags:
      - rpg
      - ttrpg
    sensitivity_tier: moderate
    rate_limit_class: standard
    default_roles:
      - admin
      - analyst
      - developer
    feature_flag_id: null
    ownership_predicates:
      - same_org
    doc_url: https://docs.example.com/privileges/rpg
  - id: rpg.rules.read
    description: Read RPG rules adapter metadata and cited rules lookup results.
    resource_tags:
      - rpg
      - rules
      - read
    sensitivity_tier: low
    rate_limit_class: standard
    default_roles:
      - admin
      - analyst
      - developer
      - viewer
    feature_flag_id: null
    ownership_predicates:
      - same_org
    doc_url: https://docs.example.com/privileges/rpg-rules-read
  - id: rpg.campaigns.read
    description: Read RPG campaign metadata owned by the current user or scope.
    resource_tags:
      - rpg
      - campaigns
      - read
    sensitivity_tier: moderate
    rate_limit_class: standard
    default_roles:
      - admin
      - analyst
      - developer
      - viewer
    feature_flag_id: null
    ownership_predicates:
      - same_org
    doc_url: https://docs.example.com/privileges/rpg-campaigns-read
  - id: rpg.campaigns.manage
    description: Create and update RPG campaigns.
    resource_tags:
      - rpg
      - campaigns
      - write
    sensitivity_tier: high
    rate_limit_class: elevated
    default_roles:
      - admin
      - developer
    feature_flag_id: null
    ownership_predicates:
      - same_org
    doc_url: https://docs.example.com/privileges/rpg-campaigns-manage
  - id: rpg.sessions.read
    description: Read RPG sessions, events, snapshots, and prompt context diagnostics.
    resource_tags:
      - rpg
      - sessions
      - read
    sensitivity_tier: moderate
    rate_limit_class: standard
    default_roles:
      - admin
      - analyst
      - developer
      - viewer
    feature_flag_id: null
    ownership_predicates:
      - same_org
    doc_url: https://docs.example.com/privileges/rpg-sessions-read
  - id: rpg.sessions.manage
    description: Create RPG sessions, append trusted events, and record dice/check results.
    resource_tags:
      - rpg
      - sessions
      - write
    sensitivity_tier: high
    rate_limit_class: elevated
    default_roles:
      - admin
      - developer
    feature_flag_id: null
    ownership_predicates:
      - same_org
    doc_url: https://docs.example.com/privileges/rpg-sessions-manage
  - id: rpg.proposals.review
    description: Apply or reject RPG state-change proposals.
    resource_tags:
      - rpg
      - proposals
      - review
    sensitivity_tier: restricted
    rate_limit_class: admin
    default_roles:
      - admin
      - developer
    feature_flag_id: null
    ownership_predicates:
      - same_org
    doc_url: https://docs.example.com/privileges/rpg-proposals-review
```

- [ ] **Step 3: Regenerate route registry snapshot**

Run: `source .venv/bin/activate && python Helper_Scripts/update_privilege_registry_snapshot.py`

Expected: `tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json` is updated and includes RPG routes with their endpoint IDs.

- [ ] **Step 4: Run privilege catalog tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/Config_Files/privilege_catalog.yaml tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json
git commit -m "feat: register RPG privileges"
```

### Task 8: Rules Lookup And Session Context Builder

**Files:**
- Create: `tldw_Server_API/app/core/RPG/rules/content_packs.py`
- Create: `tldw_Server_API/app/core/RPG/rules/lookup.py`
- Create: `tldw_Server_API/app/core/RPG/context.py`
- Modify: `tldw_Server_API/app/core/RPG/service.py`
- Test: `tldw_Server_API/tests/RPG/test_rpg_rules_context.py`

- [ ] **Step 1: Write failing rules lookup and context tests**

```python
from tldw_Server_API.app.core.RPG.context import SessionContextBuilder
from tldw_Server_API.app.core.RPG.models import RPGSnapshotState
from tldw_Server_API.app.core.RPG.rules.lookup import RulesLookupService


def test_rules_lookup_returns_citations_without_pf2e_prose():
    lookup = RulesLookupService()

    result = lookup.lookup(adapter_key="pf2e", query="dying condition", linked_rules_pack_refs=[])

    assert result.query == "dying condition"
    assert result.results
    assert all(item.text == "" for item in result.results)
    assert all(item.citation.adapter_key == "pf2e" for item in result.results)


def test_context_builder_includes_snapshot_and_rule_citations_with_budget():
    builder = SessionContextBuilder(max_chars=500)
    snapshot = RPGSnapshotState(
        scene={"summary": "Rain at the old docks"},
        npcs={"npc-1": {"npc_id": "npc-1", "name": "Ada"}},
        unresolved_rulings={"r1": {"ruling_id": "r1", "question": "How does stress clear"}},
    )

    context = builder.build(
        adapter_key="fate",
        session_title="Opening",
        snapshot=snapshot,
        rules_results=[],
    )

    assert "Opening" in context.text
    assert "Rain at the old docks" in context.text
    assert context.diagnostics["truncated"] is False
    assert context.diagnostics["rules_result_count"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_rules_context.py -v`

Expected: FAIL because lookup and context modules do not exist.

- [ ] **Step 3: Implement content pack citations and lookup result types**

```python
# tldw_Server_API/app/core/RPG/rules/content_packs.py
from __future__ import annotations

from dataclasses import dataclass

from tldw_Server_API.app.core.RPG.models import RuleCitation


@dataclass(frozen=True, slots=True)
class RuleLookupItem:
    text: str
    citation: RuleCitation
    score: float


@dataclass(frozen=True, slots=True)
class RuleLookupResult:
    query: str
    results: list[RuleLookupItem]
    diagnostics: dict[str, object]


PF2E_CITATIONS = [
    RuleCitation(
        adapter_key="pf2e",
        source_title="Archives of Nethys Pathfinder 2e",
        source_url="https://2e.aonprd.com/",
        license="ORC and Paizo Community Use references",
        license_url="https://downloads.paizo.com/ORC_License_FINAL.pdf",
        attribution="Pathfinder Second Edition rules references",
        trust_level="reference",
        content_hash="citation-only",
        snippet_id="pf2e-citation-index",
        source_version="PF2e",
        content_pack_version="1.0.0",
    )
]
```

```python
# tldw_Server_API/app/core/RPG/rules/lookup.py
from __future__ import annotations

from tldw_Server_API.app.core.RPG.rules.content_packs import PF2E_CITATIONS, RuleLookupItem, RuleLookupResult


class RulesLookupService:
    def lookup(self, adapter_key: str, query: str, linked_rules_pack_refs: list[dict[str, object]]) -> RuleLookupResult:
        if adapter_key == "pf2e":
            return RuleLookupResult(
                query=query,
                results=[RuleLookupItem(text="", citation=citation, score=0.5) for citation in PF2E_CITATIONS],
                diagnostics={"bundled_policy": "citations_only", "linked_rules_pack_count": len(linked_rules_pack_refs)},
            )
        return RuleLookupResult(
            query=query,
            results=[],
            diagnostics={"bundled_policy": "no_match", "linked_rules_pack_count": len(linked_rules_pack_refs)},
        )
```

- [ ] **Step 4: Implement bounded context builder**

```python
# tldw_Server_API/app/core/RPG/context.py
from __future__ import annotations

from dataclasses import dataclass

from tldw_Server_API.app.core.RPG.models import RPGSnapshotState
from tldw_Server_API.app.core.RPG.rules.content_packs import RuleLookupItem


@dataclass(frozen=True, slots=True)
class SessionContext:
    text: str
    diagnostics: dict[str, object]


class SessionContextBuilder:
    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars

    def build(
        self,
        adapter_key: str,
        session_title: str,
        snapshot: RPGSnapshotState,
        rules_results: list[RuleLookupItem],
    ) -> SessionContext:
        lines = [
            f"RPG session: {session_title}",
            f"Rules adapter: {adapter_key}",
            f"Scene: {snapshot.scene.get('summary', '')}",
            f"NPCs: {', '.join(sorted(npc.get('name', npc_id) for npc_id, npc in snapshot.npcs.items()))}",
            f"Open rulings: {len(snapshot.unresolved_rulings)}",
        ]
        if rules_results:
            lines.append("Rules citations:")
            for item in rules_results:
                citation = item.citation
                lines.append(f"- {citation.source_title}: {citation.source_url}")
        text = "\n".join(line for line in lines if line.strip())
        truncated = len(text) > self.max_chars
        if truncated:
            text = text[: self.max_chars]
        return SessionContext(
            text=text,
            diagnostics={"truncated": truncated, "rules_result_count": len(rules_results)},
        )
```

- [ ] **Step 5: Wire lookup and context methods into service**

Add these methods to `RPGService`:

```python
def lookup_rules(self, session_id: int, query: str) -> RuleLookupResult:
    session = self.repo.get_session(owner_user_id=self.owner_user_id, session_id=session_id)
    lookup = RulesLookupService()
    return lookup.lookup(
        adapter_key=session.adapter_key,
        query=query,
        linked_rules_pack_refs=session.active_rules_pack_refs,
    )


def build_context(self, session_id: int, query: str | None = None, max_chars: int = MAX_RPG_CONTEXT_CHARS) -> SessionContext:
    session = self.repo.get_session(owner_user_id=self.owner_user_id, session_id=session_id)
    snapshot = self.get_snapshot(session_id).snapshot
    rules = self.lookup_rules(session_id, query).results if query else []
    return SessionContextBuilder(max_chars=max_chars).build(
        adapter_key=session.adapter_key,
        session_title=session.title,
        snapshot=snapshot,
        rules_results=rules,
    )
```

- [ ] **Step 6: Run focused rules/context tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_rules_context.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/RPG/rules/content_packs.py tldw_Server_API/app/core/RPG/rules/lookup.py tldw_Server_API/app/core/RPG/context.py tldw_Server_API/app/core/RPG/service.py tldw_Server_API/tests/RPG/test_rpg_rules_context.py
git commit -m "feat: add RPG rules context builder"
```

### Task 9: MCP RPG Module And Optional Registration

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/rpg_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`
- Test: `tldw_Server_API/tests/RPG/test_rpg_mcp_module.py`

- [ ] **Step 1: Write failing MCP module tests**

```python
import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.rpg_module import RPGModule


@pytest.mark.asyncio
async def test_rpg_module_lists_read_and_write_tools():
    module = RPGModule(ModuleConfig(name="rpg"))

    tools = await module.get_tools()
    tool_by_name = {tool["name"]: tool for tool in tools}

    assert tool_by_name["rpg.adapters.list"]["annotations"]["readOnlyHint"] is True
    assert tool_by_name["rpg.events.record"]["metadata"]["category"] == "management"
    assert "rpg.proposals.apply" in tool_by_name


@pytest.mark.asyncio
async def test_rpg_module_lists_adapters_without_database_context():
    module = RPGModule(ModuleConfig(name="rpg"))

    result = await module.execute_tool("rpg.adapters.list", {}, context=None)

    assert [item["adapter_key"] for item in result["adapters"]] == ["dnd5e_srd", "fate", "pf2e"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_mcp_module.py -v`

Expected: FAIL because `RPGModule` does not exist.

- [ ] **Step 3: Implement MCP tool definitions and read-only adapter listing**

```python
# tldw_Server_API/app/core/MCP_unified/modules/implementations/rpg_module.py
from __future__ import annotations

from typing import Any

from loguru import logger

from tldw_Server_API.app.core.RPG.rules.adapters import build_default_adapter_registry

from ..base import BaseModule, create_tool_definition


class RPGModule(BaseModule):
    async def on_initialize(self) -> None:
        logger.info("Initializing RPG MCP module: {}", self.name)

    async def on_shutdown(self) -> None:
        logger.info("Shutting down RPG MCP module: {}", self.name)

    async def check_health(self) -> dict[str, bool]:
        return {"initialized": True, "adapter_registry": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name="rpg.adapters.list",
                description="List bundled RPG rules adapters.",
                parameters={"properties": {}},
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="rpg.sessions.get",
                description="Get RPG session state and current snapshot.",
                parameters={"properties": {"session_id": {"type": "integer"}}, "required": ["session_id"]},
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="rpg.events.list",
                description="List RPG session events.",
                parameters={"properties": {"session_id": {"type": "integer"}, "limit": {"type": "integer", "minimum": 1, "maximum": 200}}},
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="rpg.events.record",
                description="Record trusted RPG session events or create a proposal based on authority settings.",
                parameters={"properties": {"session_id": {"type": "integer"}, "events": {"type": "array"}, "idempotency_key": {"type": "string"}}, "required": ["session_id", "events"]},
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="rpg.roll",
                description="Resolve an RPG dice/check roll and record it when requested.",
                parameters={"properties": {"session_id": {"type": "integer"}, "check": {"type": "object"}, "record": {"type": "boolean", "default": True}}, "required": ["session_id", "check"]},
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="rpg.rules.lookup",
                description="Look up cited RPG rules references for a session.",
                parameters={"properties": {"session_id": {"type": "integer"}, "query": {"type": "string", "maxLength": 500}}, "required": ["session_id", "query"]},
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="rpg.context.build",
                description="Build bounded RPG session context with citation diagnostics.",
                parameters={"properties": {"session_id": {"type": "integer"}, "query": {"type": "string"}, "max_chars": {"type": "integer", "minimum": 1000, "maximum": 24000}}, "required": ["session_id"]},
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="rpg.proposals.apply",
                description="Apply a pending RPG proposal atomically.",
                parameters={"properties": {"session_id": {"type": "integer"}, "proposal_id": {"type": "integer"}, "review_notes": {"type": "string"}}, "required": ["session_id", "proposal_id"]},
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="rpg.proposals.reject",
                description="Reject a pending RPG proposal.",
                parameters={"properties": {"session_id": {"type": "integer"}, "proposal_id": {"type": "integer"}, "review_notes": {"type": "string"}}, "required": ["session_id", "proposal_id"]},
                metadata={"category": "management", "auth_required": True},
            ),
        ]

    async def get_resources(self) -> list[dict[str, Any]]:
        return []

    async def get_prompts(self) -> list[dict[str, Any]]:
        return []

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any = None) -> Any:
        if tool_name == "rpg.adapters.list":
            registry = build_default_adapter_registry()
            return {"adapters": [info.__dict__ for info in registry.list_infos()]}
        raise ValueError(f"Unknown RPG tool: {tool_name}")
```

Add database-backed tool execution in `execute_tool` after `rpg.adapters.list`:

```python
def _service_for_context(self, context: Any) -> RPGService:
    owner_user_id = int(getattr(context, "user_id", None) or getattr(context, "owner_user_id", None) or 0)
    if owner_user_id <= 0:
        raise ValueError("RPG MCP tools require an authenticated user context")
    db = self._open_chacha_db_for_user(owner_user_id)
    return RPGService(repo=RPGRepository.initialized(db), owner_user_id=owner_user_id)


async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any = None) -> Any:
    if tool_name == "rpg.adapters.list":
        registry = build_default_adapter_registry()
        return {"adapters": [info.__dict__ for info in registry.list_infos()]}
    service = self._service_for_context(context)
    if tool_name == "rpg.sessions.get":
        return service.get_session_payload(int(arguments["session_id"]))
    if tool_name == "rpg.events.list":
        events = service.list_events(int(arguments["session_id"]), limit=int(arguments.get("limit") or 100))
        return {"events": [event.__dict__ for event in events]}
    if tool_name == "rpg.events.record":
        result = service.record_events(int(arguments["session_id"]), list(arguments["events"]), arguments.get("idempotency_key"))
        return {"committed_events": [event.__dict__ for event in result.committed_events], "proposal": result.proposal.__dict__ if result.proposal else None}
    if tool_name == "rpg.rules.lookup":
        return service.lookup_rules(int(arguments["session_id"]), str(arguments["query"])).__dict__
    if tool_name == "rpg.context.build":
        return service.build_context(int(arguments["session_id"]), arguments.get("query"), int(arguments.get("max_chars") or 24000)).__dict__
    if tool_name == "rpg.proposals.apply":
        result = service.apply_proposal(int(arguments["session_id"]), int(arguments["proposal_id"]), arguments.get("review_notes"))
        return {"committed_events": [event.__dict__ for event in result.committed_events]}
    if tool_name == "rpg.proposals.reject":
        return service.reject_proposal(int(arguments["session_id"]), int(arguments["proposal_id"]), arguments.get("review_notes")).__dict__
    raise ValueError(f"Unknown RPG tool: {tool_name}")
```

- [ ] **Step 4: Add optional server registration**

```python
# tldw_Server_API/app/core/MCP_unified/server.py
if self._env_flag_enabled("MCP_ENABLE_RPG_MODULE"):
    if not any(m.get("id") == "rpg" for m in modules_to_load if isinstance(m, dict)):
        modules_to_load.append({
            "id": "rpg",
            "class": "tldw_Server_API.app.core.MCP_unified.modules.implementations.rpg_module:RPGModule",
            "enabled": True,
            "name": "RPG",
            "version": "1.0.0",
            "department": "management",
            "settings": {},
        })
        logger.info("MCP_ENABLE_RPG_MODULE=true; queuing RPGModule for registration")
```

Place this block beside other optional modules, before the final registration loop.

- [ ] **Step 5: Add tests for registered tool metadata**

Add this registry test:

```python
@pytest.mark.asyncio
async def test_rpg_module_write_tool_classification():
    module = RPGModule(ModuleConfig(name="rpg"))
    tools = {tool["name"]: tool for tool in await module.get_tools()}

    assert module.is_write_tool_def(tools["rpg.events.record"]) is True
    assert module.is_write_tool_def(tools["rpg.context.build"]) is False
```

- [ ] **Step 6: Run focused MCP tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_mcp_module.py tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/rpg_module.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/tests/RPG/test_rpg_mcp_module.py
git commit -m "feat: add RPG MCP module"
```

### Task 10: Documentation, Regression Checks, And Security Scan

**Files:**
- Create: `tldw_Server_API/app/core/RPG/README.md`

- [ ] **Step 1: Write the RPG runtime README**

```markdown
# RPG Runtime

The RPG runtime is a backend harness for tabletop roleplaying sessions. It stores campaigns, sessions, append-only events, cached snapshots, and reviewable state-change proposals in each user's ChaChaNotes database.

The runtime is not a virtual tabletop. It does not manage maps, token positions, lighting, walls, or live shared tabletop rendering.

Rules adapters provide mechanics metadata and check resolution for D&D 5e SRD, Pathfinder 2e, and Fate. Rules prose is citation-first and license-aware; user-provided rules packs are referenced through existing ingestion and retrieval systems rather than copied into RPG tables.

State changes from users, imports, and trusted system operations can commit directly. Model-sourced changes become proposals unless the session explicitly enables auto-commit. Proposal application validates base sequence and appends all accepted events atomically.
```

- [ ] **Step 2: Run the focused RPG suite**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG -v`

Expected: PASS.

- [ ] **Step 3: Run adjacent regression tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py -v`

Expected: PASS.

- [ ] **Step 4: Run Bandit on touched code**

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/RPG tldw_Server_API/app/core/DB_Management/RPG_DB.py tldw_Server_API/app/api/v1/endpoints/rpg.py tldw_Server_API/app/api/v1/schemas/rpg_schemas.py tldw_Server_API/app/core/MCP_unified/modules/implementations/rpg_module.py -f json -o /tmp/bandit_rpg_runtime.json`

Expected: exit code 0 or only non-actionable findings documented in the Backlog task with rationale.

- [ ] **Step 5: Run formatting and import sanity checks**

Run: `source .venv/bin/activate && python -m compileall -q tldw_Server_API/app/core/RPG tldw_Server_API/app/core/DB_Management/RPG_DB.py tldw_Server_API/app/api/v1/endpoints/rpg.py tldw_Server_API/app/api/v1/schemas/rpg_schemas.py tldw_Server_API/app/core/MCP_unified/modules/implementations/rpg_module.py`

Expected: PASS with no output.

- [ ] **Step 6: Update Backlog task final summary**

Record:

- Files created and modified.
- Test commands and outcomes.
- Bandit output path and outcome.
- Any deliberate skips with exact reason.
- Confirmation that no virtual tabletop canvas/map/token features were added.

- [ ] **Step 7: Final commit**

```bash
git add tldw_Server_API/app/core/RPG/README.md
git commit -m "docs: document RPG runtime"
```

## Self-Review Checklist

- Spec coverage:
  - Generic, non-Foundry RPG harness: Tasks 1, 5, 6, 9, 10.
  - D&D 5e SRD, Pathfinder 2e, Fate adapters: Tasks 1, 4, 8.
  - Hybrid persistence in per-user ChaChaNotes plus rules references: Tasks 2, 8.
  - RPG sessions as source of truth, chats optional: Tasks 2, 5, 6.
  - Append-only events, cached snapshots, idempotency, optimistic sequence checks: Tasks 2, 3, 5.
  - Mixed authority and proposals: Task 5.
  - REST and MCP surfaces: Tasks 6, 9.
  - Privilege catalog coverage: Task 7.
  - Legal conservatism for rules prose: Tasks 1, 8, 10.

- Type consistency:
  - `RPGRepository.initialized(db)` is introduced in Task 2 and reused in Tasks 5, 6, and 9.
  - `RPGService` is introduced in Task 5 and reused by REST, context, and MCP tasks.
  - `RPGSnapshotState`, `RPGSessionEvent`, `RuleAdapterRegistry`, `DiceRoller`, and `CheckResult` keep the same names across all tasks.
  - Endpoint IDs in Task 6 match catalog IDs in Task 7.

- Verification scope:
  - Focused RPG tests cover adapters, storage, reducer, dice/checks, service authority, REST, context, and MCP.
  - Adjacent regressions cover VN Play storage, MCP idempotency/category behavior, and privilege endpoint synchronization.
  - Bandit scans every touched Python path in the new feature.
