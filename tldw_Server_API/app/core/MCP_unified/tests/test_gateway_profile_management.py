"""Tests for standalone MCP gateway profile management helpers."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from mcp_unified.gateway.profiles import (
    GatewayProfileManagementError,
    GatewayProfileManager,
    GatewayProfileStoreMetadata,
)
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.profiles.store import (
    InMemoryProfileAssignmentStore,
    InMemoryProfileStore,
)
from mcp_unified.storage.models import AuditEvent, ProfileAssignment
from mcp_unified.storage.sqlite import SQLiteMCPStore

UTC = timezone.utc


class InMemoryAuditStore:
    """Small copy-isolated audit test double for manager audit assertions."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append_event(self, event: AuditEvent) -> AuditEvent:
        stored = event.model_copy(deep=True)
        self.events.append(stored)
        return stored.model_copy(deep=True)

    async def query_events(
        self,
        *,
        actor_id: str | None = None,
        profile_id: str | None = None,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEvent]:
        events = [
            event
            for event in self.events
            if (actor_id is None or event.actor_id == actor_id)
            and (profile_id is None or event.profile_id == profile_id)
            and (event_type is None or event.event_type == event_type)
        ]
        if limit is not None:
            events = events[:limit]
        return [event.model_copy(deep=True) for event in events]


class FailingAuditStore(InMemoryAuditStore):
    """Audit store double that simulates a transient append failure."""

    async def append_event(self, event: AuditEvent) -> AuditEvent:
        raise RuntimeError(f"audit unavailable for {event.event_type}")


def _manager(
    profile_store: InMemoryProfileStore,
    assignment_store: InMemoryProfileAssignmentStore | None = None,
    *,
    audit_store: InMemoryAuditStore | None = None,
    fallback_default_profile_id: str | None = None,
) -> GatewayProfileManager:
    return GatewayProfileManager(
        profile_store=profile_store,
        assignment_store=assignment_store or InMemoryProfileAssignmentStore(),
        audit_store=audit_store,
        store_metadata=GatewayProfileStoreMetadata(kind="memory", persistent=False),
        fallback_default_profile_id=fallback_default_profile_id,
    )


@pytest.mark.asyncio
async def test_gateway_profile_manager_lists_profiles_with_store_metadata() -> None:
    store = InMemoryProfileStore(
        [
            MCPProfile(id="reviewer", name="Reviewer"),
            MCPProfile(id="architect", name="Architect"),
        ]
    )
    assignment_store = InMemoryProfileAssignmentStore()
    manager = GatewayProfileManager(
        profile_store=store,
        assignment_store=assignment_store,
        store_metadata=GatewayProfileStoreMetadata(kind="memory", persistent=False),
    )

    payload = await manager.list_profiles()

    assert payload["ok"] is True
    assert [profile["id"] for profile in payload["profiles"]] == [
        "architect",
        "reviewer",
    ]
    assert payload["store"] == {"kind": "memory", "persistent": False}


@pytest.mark.asyncio
async def test_gateway_profile_manager_shows_json_safe_copy_isolated_profile() -> None:
    created_at = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)
    profile = MCPProfile(
        id="reviewer",
        name="Reviewer",
        metadata={"nested": {"value": "original"}},
        created_at=created_at,
        updated_at=created_at,
    )
    manager = _manager(InMemoryProfileStore([profile]))

    payload = await manager.show_profile("reviewer")

    assert payload == {
        "ok": True,
        "profile": profile.model_dump(mode="json"),
        "store": {"kind": "memory", "persistent": False},
    }
    assert isinstance(payload["profile"]["created_at"], str)

    payload["profile"]["metadata"]["nested"]["value"] = "mutated"
    second_payload = await manager.show_profile("reviewer")
    assert second_payload["profile"]["metadata"]["nested"]["value"] == "original"


@pytest.mark.asyncio
async def test_gateway_profile_manager_audits_missing_show_profile() -> None:
    audit_store = InMemoryAuditStore()
    manager = _manager(InMemoryProfileStore(), audit_store=audit_store)

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.show_profile("missing-profile")

    assert exc_info.value.reason_code == "profile_not_found"
    assert exc_info.value.to_payload()["profile_id"] == "missing-profile"
    assert [event.event_type for event in audit_store.events] == ["profile.show_failed"]
    assert audit_store.events[0].target_type == "profile"
    assert audit_store.events[0].target_id == "missing-profile"
    assert audit_store.events[0].payload == {
        "profile_id": "missing-profile",
        "reason_code": "profile_not_found",
    }
    assert "policy_document" not in json.dumps(audit_store.events[0].payload)


@pytest.mark.asyncio
async def test_duplicate_preset_uses_preset_id_as_default_stored_id() -> None:
    store = InMemoryProfileStore()
    manager = _manager(store)

    payload = await manager.duplicate_preset("project-researcher")

    assert payload["ok"] is True
    assert payload["profile"]["id"] == "project-researcher"
    assert payload["profile"]["name"] == "Project Researcher"
    assert payload["profile"]["preset_id"] == "project-researcher"
    assert payload["profile"]["preset_version"] == "2026.05.27"
    assert payload["profile"]["provenance"]["duplicated"] is True
    assert payload["store"] == {"kind": "memory", "persistent": False}

    stored = await store.get_profile("project-researcher")
    assert stored is not None
    assert stored.id == "project-researcher"


@pytest.mark.asyncio
async def test_duplicate_preset_accepts_custom_profile_id_and_name() -> None:
    manager = _manager(InMemoryProfileStore())

    payload = await manager.duplicate_preset(
        "project-researcher",
        profile_id="workspace-researcher",
        name="Workspace Researcher",
    )

    assert payload["profile"]["id"] == "workspace-researcher"
    assert payload["profile"]["name"] == "Workspace Researcher"
    assert payload["profile"]["preset_id"] == "project-researcher"


@pytest.mark.asyncio
async def test_duplicate_preset_rejects_unknown_preset() -> None:
    manager = _manager(InMemoryProfileStore())

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.duplicate_preset("missing-preset")

    assert exc_info.value.reason_code == "preset_not_found"
    assert exc_info.value.to_payload()["preset_id"] == "missing-preset"


@pytest.mark.asyncio
async def test_duplicate_preset_rejects_profile_id_collision() -> None:
    manager = _manager(InMemoryProfileStore([MCPProfile(id="project-researcher", name="Existing Researcher")]))

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.duplicate_preset("project-researcher")

    assert exc_info.value.reason_code == "profile_already_exists"
    assert exc_info.value.to_payload()["profile_id"] == "project-researcher"


@pytest.mark.asyncio
async def test_create_profile_persists_valid_profile_and_audits_success() -> None:
    created_at = datetime(2020, 1, 1, 12, 0, tzinfo=UTC)
    audit_store = InMemoryAuditStore()
    store = InMemoryProfileStore()
    manager = _manager(store, audit_store=audit_store)

    payload = await manager.create_profile(
        {
            "id": "custom-reviewer",
            "name": "Custom Reviewer",
            "metadata": {"owner": "qa"},
            "created_at": created_at.isoformat(),
            "updated_at": created_at.isoformat(),
        }
    )

    assert payload["ok"] is True
    assert payload["store"] == {"kind": "memory", "persistent": False}
    assert payload["profile"]["id"] == "custom-reviewer"
    assert payload["profile"]["metadata"] == {"owner": "qa"}
    assert payload["profile"]["created_at"] == created_at.isoformat()
    assert payload["profile"]["updated_at"] != created_at.isoformat()
    stored = await store.get_profile("custom-reviewer")
    assert stored is not None
    assert stored.name == "Custom Reviewer"
    assert stored.metadata == {"owner": "qa"}
    assert stored.created_at == created_at
    assert stored.updated_at > created_at
    assert stored.updated_at.isoformat() == payload["profile"]["updated_at"]
    assert stored.enabled is True
    assert [event.event_type for event in audit_store.events] == ["profile.created"]


@pytest.mark.asyncio
async def test_create_profile_rejects_duplicate_id() -> None:
    store = InMemoryProfileStore(
        [MCPProfile(id="existing", name="Existing", metadata={"owner": "original"})]
    )
    manager = _manager(store)

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.create_profile({"id": "existing", "name": "Duplicate"})

    assert exc_info.value.reason_code == "profile_already_exists"
    assert exc_info.value.to_payload()["profile_id"] == "existing"
    stored = await store.get_profile("existing")
    assert stored is not None
    assert stored.name == "Existing"
    assert stored.metadata == {"owner": "original"}


@pytest.mark.asyncio
async def test_create_profile_rejects_disabled_effective_default_assignment_id() -> None:
    assignment_store = InMemoryProfileAssignmentStore(
        [ProfileAssignment(id="gateway-default", profile_id="default", is_default=True)]
    )
    store = InMemoryProfileStore()
    manager = _manager(store, assignment_store)

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.create_profile({"id": "default", "name": "Default", "enabled": False})

    assert exc_info.value.reason_code == "profile_is_default"
    assert await store.get_profile("default") is None


@pytest.mark.asyncio
async def test_create_profile_allows_disabled_fallback_id_when_assignment_overrides_it() -> None:
    assignment_store = InMemoryProfileAssignmentStore(
        [ProfileAssignment(id="gateway-default", profile_id="assigned", is_default=True)]
    )
    manager = _manager(
        InMemoryProfileStore([MCPProfile(id="assigned", name="Assigned")]),
        assignment_store,
        fallback_default_profile_id="fallback",
    )

    payload = await manager.create_profile(
        {"id": "fallback", "name": "Fallback", "enabled": False}
    )

    assert payload["profile"]["id"] == "fallback"
    assert payload["profile"]["enabled"] is False


@pytest.mark.asyncio
async def test_create_profile_rejects_disabled_fallback_default_without_assignment() -> None:
    store = InMemoryProfileStore()
    manager = _manager(
        store,
        fallback_default_profile_id="fallback",
    )

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.create_profile(
            {"id": "fallback", "name": "Fallback", "enabled": False}
        )

    assert exc_info.value.reason_code == "profile_is_default"
    assert await store.get_profile("fallback") is None


@pytest.mark.asyncio
async def test_duplicate_preset_payload_mutation_does_not_mutate_store() -> None:
    store = InMemoryProfileStore()
    manager = _manager(store)

    payload = await manager.duplicate_preset("project-researcher")
    payload["profile"]["name"] = "Mutated"
    payload["profile"]["metadata"]["agent_metadata"]["ui_label"] = "Mutated"

    stored = await store.get_profile("project-researcher")
    assert stored is not None
    assert stored.name == "Project Researcher"
    assert stored.metadata["agent_metadata"]["ui_label"] == "Project Researcher"


@pytest.mark.asyncio
async def test_duplicate_preset_succeeds_when_audit_append_fails() -> None:
    store = InMemoryProfileStore()
    manager = _manager(store, audit_store=FailingAuditStore())

    payload = await manager.duplicate_preset("project-researcher")

    assert payload["ok"] is True
    assert payload["profile"]["id"] == "project-researcher"
    assert await store.get_profile("project-researcher") is not None


@pytest.mark.asyncio
async def test_set_default_profile_stores_gateway_default_assignment() -> None:
    store = InMemoryProfileStore([MCPProfile(id="reviewer", name="Reviewer")])
    assignment_store = InMemoryProfileAssignmentStore()
    manager = _manager(store, assignment_store)

    payload = await manager.set_default_profile("reviewer")

    assert payload["ok"] is True
    assert payload["profile"]["id"] == "reviewer"
    assert payload["assignment"]["id"] == "gateway-default"
    assert payload["assignment"]["profile_id"] == "reviewer"
    assert payload["assignment"]["is_default"] is True
    assert payload["assignment"]["enabled"] is True

    assignments = await assignment_store.list_assignments()
    assert len(assignments) == 1
    assert assignments[0].id == "gateway-default"
    assert assignments[0].profile_id == "reviewer"
    assert assignments[0].is_default is True
    assert assignments[0].enabled is True


@pytest.mark.asyncio
async def test_set_default_profile_overwrites_gateway_default_assignment() -> None:
    store = InMemoryProfileStore(
        [
            MCPProfile(id="reviewer", name="Reviewer"),
            MCPProfile(id="architect", name="Architect"),
        ]
    )
    assignment_store = InMemoryProfileAssignmentStore()
    manager = _manager(store, assignment_store)

    await manager.set_default_profile("reviewer")
    first_assignment = await assignment_store.get_assignment("gateway-default")
    assert first_assignment is not None

    await manager.set_default_profile("architect")

    assignments = await assignment_store.list_assignments()
    assert [assignment.id for assignment in assignments] == ["gateway-default"]
    assert assignments[0].profile_id == "architect"
    assert assignments[0].created_at == first_assignment.created_at
    assert assignments[0].updated_at >= first_assignment.updated_at


@pytest.mark.asyncio
async def test_set_default_profile_becomes_effective_with_future_legacy_default() -> None:
    future_updated_at = datetime(2099, 1, 1, tzinfo=UTC)
    store = InMemoryProfileStore(
        [
            MCPProfile(id="reviewer", name="Reviewer"),
            MCPProfile(id="legacy", name="Legacy"),
        ]
    )
    assignment_store = InMemoryProfileAssignmentStore(
        [
            ProfileAssignment(
                id="legacy-future",
                profile_id="legacy",
                is_default=True,
                updated_at=future_updated_at,
            )
        ]
    )
    manager = _manager(store, assignment_store)

    set_payload = await manager.set_default_profile("reviewer")
    default_payload = await manager.get_default_profile()

    assert set_payload["assignment"]["id"] == "gateway-default"
    assert default_payload["profile"]["id"] == "reviewer"
    assert default_payload["assignment"]["id"] == "gateway-default"

    gateway_default = await assignment_store.get_assignment("gateway-default")
    assert gateway_default is not None
    assert gateway_default.updated_at > future_updated_at


@pytest.mark.asyncio
async def test_get_default_profile_reads_assignment_store_before_fallback() -> None:
    store = InMemoryProfileStore(
        [
            MCPProfile(id="stored-default", name="Stored Default"),
            MCPProfile(id="fallback-default", name="Fallback Default"),
        ]
    )
    assignment_store = InMemoryProfileAssignmentStore(
        [
            ProfileAssignment(
                id="gateway-default",
                profile_id="stored-default",
                is_default=True,
            )
        ]
    )
    manager = _manager(
        store,
        assignment_store,
        fallback_default_profile_id="fallback-default",
    )

    payload = await manager.get_default_profile()

    assert payload["ok"] is True
    assert payload["profile"]["id"] == "stored-default"
    assert payload["assignment"]["id"] == "gateway-default"
    assert payload["default"]["source"] == "assignment"


@pytest.mark.asyncio
async def test_get_default_profile_rejects_missing_default() -> None:
    manager = _manager(InMemoryProfileStore())

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.get_default_profile()

    assert exc_info.value.reason_code == "default_profile_not_configured"


@pytest.mark.asyncio
async def test_get_default_profile_audits_missing_assigned_target_profile() -> None:
    audit_store = InMemoryAuditStore()
    assignment_store = InMemoryProfileAssignmentStore(
        [
            ProfileAssignment(
                id="gateway-default",
                profile_id="missing-default",
                is_default=True,
            )
        ]
    )
    manager = _manager(
        InMemoryProfileStore(),
        assignment_store,
        audit_store=audit_store,
    )

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.get_default_profile()

    assert exc_info.value.reason_code == "profile_not_found"
    assert exc_info.value.to_payload()["profile_id"] == "missing-default"
    assert [event.event_type for event in audit_store.events] == ["profile.default_read_failed"]
    assert audit_store.events[0].target_type == "profile_assignment"
    assert audit_store.events[0].target_id == "gateway-default"
    assert audit_store.events[0].payload == {
        "assignment_id": "gateway-default",
        "profile_id": "missing-default",
        "reason_code": "profile_not_found",
    }


@pytest.mark.asyncio
async def test_get_default_profile_audits_disabled_assigned_target_profile() -> None:
    audit_store = InMemoryAuditStore()
    assignment_store = InMemoryProfileAssignmentStore(
        [
            ProfileAssignment(
                id="gateway-default",
                profile_id="disabled-default",
                is_default=True,
            )
        ]
    )
    manager = _manager(
        InMemoryProfileStore(
            [
                MCPProfile(
                    id="disabled-default",
                    name="Disabled Default",
                    enabled=False,
                )
            ]
        ),
        assignment_store,
        audit_store=audit_store,
    )

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.get_default_profile()

    assert exc_info.value.reason_code == "profile_disabled"
    assert exc_info.value.to_payload()["profile_id"] == "disabled-default"
    assert [event.event_type for event in audit_store.events] == ["profile.default_read_failed"]
    assert audit_store.events[0].target_type == "profile_assignment"
    assert audit_store.events[0].target_id == "gateway-default"
    assert audit_store.events[0].payload == {
        "assignment_id": "gateway-default",
        "profile_id": "disabled-default",
        "reason_code": "profile_disabled",
    }


@pytest.mark.asyncio
async def test_set_default_profile_rejects_missing_target_profile() -> None:
    manager = _manager(InMemoryProfileStore())

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.set_default_profile("missing")

    assert exc_info.value.reason_code == "profile_not_found"
    assert exc_info.value.to_payload()["profile_id"] == "missing"


@pytest.mark.asyncio
async def test_set_default_profile_rejects_disabled_target_profile() -> None:
    manager = _manager(InMemoryProfileStore([MCPProfile(id="disabled", name="Disabled", enabled=False)]))

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.set_default_profile("disabled")

    assert exc_info.value.reason_code == "profile_disabled"
    assert exc_info.value.to_payload()["profile_id"] == "disabled"


@pytest.mark.asyncio
async def test_get_default_profile_chooses_newest_legacy_default_with_id_tie_breaker() -> None:
    older = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)
    newer = older + timedelta(minutes=5)
    store = InMemoryProfileStore(
        [
            MCPProfile(id="old", name="Old"),
            MCPProfile(id="new-a", name="New A"),
            MCPProfile(id="new-z", name="New Z"),
        ]
    )
    assignment_store = InMemoryProfileAssignmentStore(
        [
            ProfileAssignment(
                id="z-old",
                profile_id="old",
                is_default=True,
                updated_at=older,
            ),
            ProfileAssignment(
                id="z-new",
                profile_id="new-z",
                is_default=True,
                updated_at=newer,
            ),
            ProfileAssignment(
                id="a-new",
                profile_id="new-a",
                is_default=True,
                updated_at=newer,
            ),
        ]
    )
    manager = _manager(store, assignment_store)

    payload = await manager.get_default_profile()

    assert payload["profile"]["id"] == "new-a"
    assert payload["assignment"]["id"] == "a-new"


@pytest.mark.asyncio
async def test_patch_profile_replaces_all_allowed_policy_fields() -> None:
    original_updated_at = datetime(2020, 1, 1, 12, 0, tzinfo=UTC)
    original = MCPProfile(
        id="reviewer",
        name="Reviewer",
        description="old",
        enabled=True,
        metadata={"old": True},
        policy_document={
            "allowed_tools": ["old.tool"],
            "denied_tools": ["old.deny"],
            "capabilities": ["old-capability"],
            "denied_capabilities": ["old-denied-capability"],
            "tool_patterns": ["old.*"],
            "module_patterns": ["old.module.*"],
            "risk_classes": ["old-risk"],
            "resource_constraints": {"max_runtime_seconds": 30},
        },
        updated_at=original_updated_at,
    )
    store = InMemoryProfileStore([original])
    manager = _manager(store)

    stored_before = await store.get_profile("reviewer")
    assert stored_before is not None
    before_updated_at = stored_before.updated_at

    payload = await manager.patch_profile(
        "reviewer",
        {
            "name": "Senior Reviewer",
            "description": "new",
            "enabled": False,
            "metadata": {"new": True},
            "policy_document": {
                "allowed_tools": ["new.tool"],
                "denied_tools": ["new.deny"],
                "capabilities": ["new-capability"],
                "denied_capabilities": ["new-denied-capability"],
                "tool_patterns": ["new.*"],
                "module_patterns": ["new.module.*"],
                "risk_classes": ["new-risk"],
                "resource_constraints": {"max_runtime_seconds": 120},
            },
        },
    )

    profile = payload["profile"]
    assert payload["ok"] is True
    assert payload["store"] == {"kind": "memory", "persistent": False}
    assert profile["name"] == "Senior Reviewer"
    assert profile["description"] == "new"
    assert profile["enabled"] is False
    assert profile["metadata"] == {"new": True}
    assert profile["policy_document"]["allowed_tools"] == ["new.tool"]
    assert profile["policy_document"]["denied_tools"] == ["new.deny"]
    assert profile["policy_document"]["capabilities"] == ["new-capability"]
    assert profile["policy_document"]["denied_capabilities"] == ["new-denied-capability"]
    assert profile["policy_document"]["tool_patterns"] == ["new.*"]
    assert profile["policy_document"]["module_patterns"] == ["new.module.*"]
    assert profile["policy_document"]["risk_classes"] == ["new-risk"]
    assert profile["policy_document"]["resource_constraints"] == {"max_runtime_seconds": 120}

    stored = await store.get_profile("reviewer")
    assert stored is not None
    assert stored.name == "Senior Reviewer"
    assert stored.description == "new"
    assert stored.enabled is False
    assert stored.metadata == {"new": True}
    assert stored.policy_document.allowed_tools == ["new.tool"]
    assert stored.policy_document.denied_tools == ["new.deny"]
    assert stored.policy_document.capabilities == ["new-capability"]
    assert stored.policy_document.denied_capabilities == ["new-denied-capability"]
    assert stored.policy_document.tool_patterns == ["new.*"]
    assert stored.policy_document.module_patterns == ["new.module.*"]
    assert stored.policy_document.risk_classes == ["new-risk"]
    assert stored.policy_document.resource_constraints == {"max_runtime_seconds": 120}
    assert stored.updated_at > before_updated_at
    assert stored.updated_at.isoformat() == payload["profile"]["updated_at"]


@pytest.mark.asyncio
async def test_patch_profile_preserves_omitted_policy_document_field() -> None:
    original_updated_at = datetime(2020, 1, 1, 12, 0, tzinfo=UTC)
    original = MCPProfile(
        id="reviewer",
        name="Reviewer",
        policy_document={
            "allowed_tools": ["old.tool"],
            "denied_tools": ["old.deny"],
        },
        updated_at=original_updated_at,
    )
    store = InMemoryProfileStore([original])
    manager = _manager(store)

    stored_before = await store.get_profile("reviewer")
    assert stored_before is not None
    before_updated_at = stored_before.updated_at

    payload = await manager.patch_profile(
        "reviewer",
        {"policy_document": {"allowed_tools": ["new.tool"]}},
    )

    profile = payload["profile"]
    assert payload["ok"] is True
    assert payload["store"] == {"kind": "memory", "persistent": False}
    assert profile["policy_document"]["allowed_tools"] == ["new.tool"]
    assert profile["policy_document"]["denied_tools"] == ["old.deny"]

    stored = await store.get_profile("reviewer")
    assert stored is not None
    assert stored.policy_document.allowed_tools == ["new.tool"]
    assert stored.policy_document.denied_tools == ["old.deny"]
    assert stored.updated_at > before_updated_at
    assert stored.updated_at.isoformat() == payload["profile"]["updated_at"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "patch_document",
    [
        {"id": "renamed"},
        {"schema_version": "2026-05-31"},
        {"preset_id": "code-reviewer"},
        {"preset_version": "2026.05.27"},
        {"approval_policy": {"mode": "auto"}},
        {"path_scopes": [{"path": "/tmp"}]},
        {"external_server_grants": [{"server_id": "external"}]},
        {"credential_grants": [{"credential_id": "credential"}]},
        {"provenance": {"source": "manual"}},
        {"created_at": "2026-05-31T12:00:00+00:00"},
        {"updated_at": "2026-05-31T12:00:00+00:00"},
        {"policy_document": {"unknown_policy_field": True}},
        {},
        {"policy_document": {}},
    ],
)
async def test_patch_profile_rejects_invalid_patch_shapes(
    patch_document: dict[str, object],
) -> None:
    created_at = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)
    updated_at = datetime(2026, 5, 31, 12, 30, tzinfo=UTC)
    store = InMemoryProfileStore(
        [
            MCPProfile(
                id="reviewer",
                name="Reviewer",
                schema_version=7,
                preset_id="preset-original",
                preset_version="v-original",
                approval_policy={"mode": "manual"},
                path_scopes=[{"path": "/allowed"}],
                external_server_grants=[{"server_id": "existing"}],
                credential_grants=[{"slot": "existing"}],
                provenance={"source": "original"},
                created_at=created_at,
                updated_at=updated_at,
            )
        ]
    )
    manager = _manager(store)

    stored_before = await store.get_profile("reviewer")
    assert stored_before is not None
    before_dump = stored_before.model_dump(mode="json")

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.patch_profile("reviewer", patch_document)

    stored_after = await store.get_profile("reviewer")
    assert stored_after is not None
    assert exc_info.value.reason_code == "invalid_profile_patch"
    assert stored_after.model_dump(mode="json") == before_dump


@pytest.mark.asyncio
async def test_patch_profile_rejects_semantic_noop_without_touching_updated_at() -> None:
    created_at = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)
    updated_at = datetime(2026, 5, 31, 12, 30, tzinfo=UTC)
    store = InMemoryProfileStore(
        [
            MCPProfile(
                id="reviewer",
                name="Reviewer",
                schema_version=7,
                preset_id="preset-original",
                preset_version="v-original",
                approval_policy={"mode": "manual"},
                path_scopes=[{"path": "/allowed"}],
                external_server_grants=[{"server_id": "existing"}],
                credential_grants=[{"slot": "existing"}],
                provenance={"source": "original"},
                created_at=created_at,
                updated_at=updated_at,
            )
        ]
    )
    manager = _manager(store)

    stored_before = await store.get_profile("reviewer")
    assert stored_before is not None
    before_dump = stored_before.model_dump(mode="json")

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.patch_profile("reviewer", {"name": "Reviewer"})

    stored_after = await store.get_profile("reviewer")
    assert stored_after is not None
    assert exc_info.value.reason_code == "invalid_profile_patch"
    assert stored_after.model_dump(mode="json") == before_dump


@pytest.mark.asyncio
async def test_patch_profile_rejects_default_profile_disable() -> None:
    assignment_store = InMemoryProfileAssignmentStore(
        [ProfileAssignment(id="gateway-default", profile_id="reviewer", is_default=True)]
    )
    created_at = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)
    updated_at = datetime(2026, 5, 31, 12, 30, tzinfo=UTC)
    store = InMemoryProfileStore(
        [
            MCPProfile(
                id="reviewer",
                name="Reviewer",
                schema_version=7,
                preset_id="preset-original",
                preset_version="v-original",
                enabled=True,
                approval_policy={"mode": "manual"},
                path_scopes=[{"path": "/allowed"}],
                external_server_grants=[{"server_id": "existing"}],
                credential_grants=[{"slot": "existing"}],
                provenance={"source": "original"},
                created_at=created_at,
                updated_at=updated_at,
            )
        ]
    )
    manager = _manager(
        store,
        assignment_store,
    )

    stored_before = await store.get_profile("reviewer")
    assert stored_before is not None
    before_dump = stored_before.model_dump(mode="json")

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.patch_profile("reviewer", {"enabled": False})

    assert exc_info.value.reason_code == "profile_is_default"
    stored = await store.get_profile("reviewer")
    assert stored is not None
    assert stored.enabled is True
    assert stored.model_dump(mode="json") == before_dump


@pytest.mark.asyncio
async def test_delete_profile_removes_unassigned_non_default_profile() -> None:
    store = InMemoryProfileStore([MCPProfile(id="temporary", name="Temporary")])
    manager = _manager(store)

    payload = await manager.delete_profile("temporary")

    assert payload == {
        "ok": True,
        "profile_id": "temporary",
        "store": {"kind": "memory", "persistent": False},
    }
    assert await store.get_profile("temporary") is None


@pytest.mark.asyncio
async def test_delete_profile_rejects_missing_profile() -> None:
    manager = _manager(InMemoryProfileStore())

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.delete_profile("missing")

    assert exc_info.value.reason_code == "profile_not_found"
    assert exc_info.value.to_payload()["profile_id"] == "missing"


@pytest.mark.asyncio
async def test_delete_profile_rejects_effective_default_profile() -> None:
    assignment_store = InMemoryProfileAssignmentStore(
        [ProfileAssignment(id="gateway-default", profile_id="default", is_default=True)]
    )
    store = InMemoryProfileStore([MCPProfile(id="default", name="Default")])
    manager = _manager(
        store,
        assignment_store,
    )

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.delete_profile("default")

    assert exc_info.value.reason_code == "profile_is_default"
    assert await store.get_profile("default") is not None


@pytest.mark.asyncio
async def test_delete_profile_rejects_assigned_profile() -> None:
    assignment_store = InMemoryProfileAssignmentStore(
        [ProfileAssignment(id="workspace-assignment", profile_id="assigned", workspace_id="ws")]
    )
    store = InMemoryProfileStore([MCPProfile(id="assigned", name="Assigned")])
    manager = _manager(
        store,
        assignment_store,
    )

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.delete_profile("assigned")

    assert exc_info.value.reason_code == "profile_has_assignments"
    assert await store.get_profile("assigned") is not None
    assert await assignment_store.get_assignment("workspace-assignment") is not None


@pytest.mark.asyncio
async def test_profile_crud_failure_audits_are_compact_and_expected() -> None:
    audit_store = InMemoryAuditStore()
    assignment_store = InMemoryProfileAssignmentStore(
        [
            ProfileAssignment(id="gateway-default", profile_id="default", is_default=True),
            ProfileAssignment(
                id="workspace-assignment",
                profile_id="assigned",
                workspace_id="ws",
            ),
        ]
    )
    manager = _manager(
        InMemoryProfileStore(
            [
                MCPProfile(id="reviewer", name="Reviewer"),
                MCPProfile(id="default", name="Default"),
                MCPProfile(id="assigned", name="Assigned"),
            ]
        ),
        assignment_store,
        audit_store=audit_store,
    )

    future_owned_invalid_patches = [
        {"approval_policy": {"mode": "auto"}},
        {"path_scopes": [{"path": "/tmp"}]},
        {"external_server_grants": [{"server_id": "external"}]},
        {"credential_grants": [{"slot": "token"}]},
        {"provenance": {"source": "bad"}},
        {"created_at": "2026-05-31T12:00:00+00:00"},
        {"updated_at": "2026-05-31T12:00:00+00:00"},
    ]
    for patch_document in future_owned_invalid_patches:
        with pytest.raises(GatewayProfileManagementError):
            await manager.patch_profile("reviewer", patch_document)

    with pytest.raises(GatewayProfileManagementError):
        await manager.patch_profile("reviewer", {"name": "Reviewer"})
    with pytest.raises(GatewayProfileManagementError):
        await manager.patch_profile("default", {"enabled": False})
    with pytest.raises(GatewayProfileManagementError):
        await manager.delete_profile("default")
    with pytest.raises(GatewayProfileManagementError):
        await manager.delete_profile("assigned")

    assert [event.event_type for event in audit_store.events] == [
        "profile.patch_failed",
        "profile.patch_failed",
        "profile.patch_failed",
        "profile.patch_failed",
        "profile.patch_failed",
        "profile.patch_failed",
        "profile.patch_failed",
        "profile.patch_failed",
        "profile.patch_failed",
        "profile.delete_failed",
        "profile.delete_failed",
    ]
    assert [event.payload["reason_code"] for event in audit_store.events] == [
        "invalid_profile_patch",
        "invalid_profile_patch",
        "invalid_profile_patch",
        "invalid_profile_patch",
        "invalid_profile_patch",
        "invalid_profile_patch",
        "invalid_profile_patch",
        "invalid_profile_patch",
        "profile_is_default",
        "profile_is_default",
        "profile_has_assignments",
    ]

    serialized_payloads = json.dumps(
        [event.payload for event in audit_store.events],
        sort_keys=True,
    )
    assert "policy_document" not in serialized_payloads
    assert "approval_policy" not in serialized_payloads
    assert "path_scopes" not in serialized_payloads
    assert "external_server_grants" not in serialized_payloads
    assert "credential_grants" not in serialized_payloads
    assert "provenance" not in serialized_payloads
    assert "created_at" not in serialized_payloads
    assert "updated_at" not in serialized_payloads


@pytest.mark.asyncio
async def test_delete_profile_sqlite_guard_preserves_assigned_profile(tmp_path: Path) -> None:
    store = SQLiteMCPStore(tmp_path / "mcp.db")
    try:
        await store.upsert_profile(MCPProfile(id="assigned", name="Assigned"))
        await store.upsert_assignment(
            ProfileAssignment(id="workspace-assignment", profile_id="assigned", workspace_id="ws")
        )
        manager = GatewayProfileManager(
            profile_store=store,
            assignment_store=store,
            store_metadata=GatewayProfileStoreMetadata(kind="sqlite", persistent=True),
        )

        with pytest.raises(GatewayProfileManagementError) as exc_info:
            await manager.delete_profile("assigned")

        assert exc_info.value.reason_code == "profile_has_assignments"
        assert await store.get_profile("assigned") is not None
        assert await store.get_assignment("workspace-assignment") is not None
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_gateway_profile_manager_audits_successes_and_expected_failures() -> None:
    audit_store = InMemoryAuditStore()
    manager = _manager(
        InMemoryProfileStore(
            [
                MCPProfile(id="project-researcher", name="Existing Researcher"),
                MCPProfile(id="default", name="Default"),
                MCPProfile(id="disabled", name="Disabled", enabled=False),
            ]
        ),
        audit_store=audit_store,
    )

    await manager.duplicate_preset("code-reviewer")
    await manager.set_default_profile("default")

    with pytest.raises(GatewayProfileManagementError):
        await manager.duplicate_preset("project-researcher")
    with pytest.raises(GatewayProfileManagementError):
        await manager.duplicate_preset("missing-preset")
    with pytest.raises(GatewayProfileManagementError):
        await manager.set_default_profile("missing-profile")
    with pytest.raises(GatewayProfileManagementError):
        await manager.set_default_profile("disabled")

    assert [event.event_type for event in audit_store.events] == [
        "profile.duplicated_from_preset",
        "profile.default_changed",
        "profile.duplication_failed",
        "profile.duplication_failed",
        "profile.default_change_failed",
        "profile.default_change_failed",
    ]

    payloads_by_reason = {
        event.payload.get("reason_code"): event.payload
        for event in audit_store.events
        if event.payload.get("reason_code")
    }
    assert payloads_by_reason["profile_already_exists"]["profile_id"] == ("project-researcher")
    assert payloads_by_reason["preset_not_found"]["preset_id"] == "missing-preset"
    assert payloads_by_reason["profile_not_found"]["profile_id"] == "missing-profile"
    assert payloads_by_reason["profile_disabled"]["profile_id"] == "disabled"

    serialized_payloads = json.dumps(
        [event.payload for event in audit_store.events],
        sort_keys=True,
    )
    assert "policy_document" not in serialized_payloads
    assert "external_server_grants" not in serialized_payloads
    assert "credential_grants" not in serialized_payloads
