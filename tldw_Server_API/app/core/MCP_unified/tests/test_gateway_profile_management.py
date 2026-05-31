"""Tests for standalone MCP gateway profile management helpers."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

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
