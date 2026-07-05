from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.Setup.first_run_mcp_tools import (
    CATALOG_VERSION,
    CONFIRMATION_VERSION,
    PROFILE_DISPLAY_NAME,
    SETUP_ORIGIN,
    compute_first_run_policy_hash,
    generate_first_run_policy,
)
from tldw_Server_API.app.core.Setup.first_run_state import FirstRunState, FirstRunStatus
from tldw_Server_API.app.services.setup_mcp_tools_service import (
    McpToolsApplyRequest,
    SetupMcpToolsService,
)


@pytest.fixture
def first_run_state() -> FirstRunState:
    created_at = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)
    return FirstRunState(
        status=FirstRunStatus.IN_PROGRESS,
        current_step="mcp_tools",
        created_at=created_at,
        updated_at=created_at,
    )


@pytest.fixture
def tool_entries() -> list[dict[str, Any]]:
    return [
        {
            "tool_name": "knowledge.search",
            "module": "knowledge",
            "risk_class": "low",
            "mutates_state": False,
        },
        {
            "tool_name": "knowledge.get",
            "module": "knowledge",
            "risk_class": "low",
            "mutates_state": False,
        },
        {
            "tool_name": "notes.search",
            "module": "notes",
            "risk_class": "low",
            "mutates_state": False,
        },
        {
            "tool_name": "mcp.tools.list",
            "module": "mcp_discovery",
            "risk_class": "unclassified",
            "mutates_state": False,
        },
        {
            "tool_name": "web.search",
            "module": "web",
            "risk_class": "low",
            "mutates_state": False,
            "uses_network": True,
        },
        {
            "tool_name": "fs.read_text",
            "module": "filesystem",
            "risk_class": "low",
            "mutates_state": False,
            "uses_filesystem": True,
            "path_boundable": True,
        },
        {
            "tool_name": "notes.create",
            "module": "notes",
            "risk_class": "high",
            "mutates_state": True,
        },
        {
            "tool_name": "notes.delete",
            "module": "notes",
            "risk_class": "high",
            "mutates_state": True,
            "destructive": True,
            "uses_filesystem": True,
        },
    ]


@pytest.fixture
def fake_registry(tool_entries: list[dict[str, Any]]) -> "FakeToolRegistry":
    return FakeToolRegistry(tool_entries)


@pytest.fixture
def fake_hub() -> "FakeMcpHub":
    return FakeMcpHub()


class FakeToolRegistry:
    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.entries = entries

    async def list_entries(self) -> list[dict[str, Any]]:
        return deepcopy(self.entries)


class FakeMcpHub:
    def __init__(self) -> None:
        self.permission_profiles: list[dict[str, Any]] = []
        self.policy_assignments: list[dict[str, Any]] = []
        self.created_profiles: list[dict[str, Any]] = []
        self.updated_profiles: list[dict[str, Any]] = []
        self.created_assignments: list[dict[str, Any]] = []
        self.updated_assignments: list[dict[str, Any]] = []
        self._next_profile_id = 1
        self._next_assignment_id = 10

    async def list_permission_profiles(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return [
            deepcopy(profile)
            for profile in self.permission_profiles
            if (owner_scope_type is None or profile.get("owner_scope_type") == owner_scope_type)
            and (owner_scope_id is None or profile.get("owner_scope_id") == owner_scope_id)
        ]

    async def create_permission_profile(self, **payload: Any) -> dict[str, Any]:
        row = {"id": self._next_profile_id, **deepcopy(payload)}
        self._next_profile_id += 1
        self.permission_profiles.append(row)
        self.created_profiles.append(deepcopy(payload))
        return deepcopy(row)

    async def update_permission_profile(
        self,
        profile_id: int,
        *,
        actor_id: int | None = None,
        **update_fields: Any,
    ) -> dict[str, Any] | None:
        for profile in self.permission_profiles:
            if profile["id"] == profile_id:
                profile.update(deepcopy(update_fields))
                row = deepcopy(profile)
                self.updated_profiles.append(
                    {"profile_id": profile_id, "actor_id": actor_id, **deepcopy(update_fields)}
                )
                return row
        return None

    async def list_policy_assignments(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            deepcopy(assignment)
            for assignment in self.policy_assignments
            if (owner_scope_type is None or assignment.get("owner_scope_type") == owner_scope_type)
            and (owner_scope_id is None or assignment.get("owner_scope_id") == owner_scope_id)
            and (target_type is None or assignment.get("target_type") == target_type)
            and (target_id is None or assignment.get("target_id") == target_id)
        ]

    async def create_policy_assignment(self, **payload: Any) -> dict[str, Any]:
        row = {"id": self._next_assignment_id, **deepcopy(payload)}
        self._next_assignment_id += 1
        self.policy_assignments.append(row)
        self.created_assignments.append(deepcopy(payload))
        return deepcopy(row)

    async def update_policy_assignment(
        self,
        assignment_id: int,
        *,
        actor_id: int | None = None,
        **update_fields: Any,
    ) -> dict[str, Any] | None:
        for assignment in self.policy_assignments:
            if assignment["id"] == assignment_id:
                assignment.update(deepcopy(update_fields))
                row = deepcopy(assignment)
                self.updated_assignments.append(
                    {"assignment_id": assignment_id, "actor_id": actor_id, **deepcopy(update_fields)}
                )
                return row
        return None


def _setup_instance_id(state: FirstRunState) -> str:
    return f"first_run:{state.created_at.isoformat()}"


def _policy(
    *,
    state: FirstRunState,
    tool_entries: list[dict[str, Any]],
    selected_pack_ids: list[str] | None = None,
    selected_addon_ids: list[str] | None = None,
    confirmed_addon_ids: list[str] | None = None,
    confirmation_version: str | None = None,
) -> dict[str, Any]:
    return generate_first_run_policy(
        selected_pack_ids=selected_pack_ids or ["research"],
        selected_addon_ids=selected_addon_ids or [],
        confirmed_addon_ids=confirmed_addon_ids or [],
        confirmation_version=confirmation_version,
        setup_instance_id=_setup_instance_id(state),
        tool_entries=tool_entries,
    )


def _profile(
    profile_id: int,
    policy_document: dict[str, Any],
    *,
    name: str = PROFILE_DISPLAY_NAME,
    owner_scope_type: str = "global",
    owner_scope_id: int | None = None,
) -> dict[str, Any]:
    return {
        "id": profile_id,
        "name": name,
        "owner_scope_type": owner_scope_type,
        "owner_scope_id": owner_scope_id,
        "mode": "custom",
        "path_scope_object_id": None,
        "policy_document": deepcopy(policy_document),
        "is_active": True,
    }


def _default_assignment(
    assignment_id: int,
    profile_id: int,
    *,
    target_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": assignment_id,
        "target_type": "default",
        "target_id": target_id,
        "owner_scope_type": "global",
        "owner_scope_id": None,
        "profile_id": profile_id,
        "path_scope_object_id": None,
        "workspace_source_mode": None,
        "workspace_set_object_id": None,
        "inline_policy_document": {},
        "approval_policy_id": None,
        "is_active": True,
    }


@pytest.mark.asyncio
async def test_apply_creates_profile_and_default_assignment(
    first_run_state: FirstRunState,
    fake_hub: FakeMcpHub,
    fake_registry: FakeToolRegistry,
) -> None:
    service = SetupMcpToolsService(hub=fake_hub, tool_registry=fake_registry)

    result = await service.apply_selection(
        state=first_run_state,
        request=McpToolsApplyRequest(selected_pack_ids=["research"], selected_addon_ids=[]),
    )

    assert result.profile_id == 1
    assert result.assignment_id == 10
    assert fake_hub.created_profiles[0]["name"] == "First-run default"
    assert fake_hub.created_assignments[0]["target_type"] == "default"
    assert fake_hub.created_assignments[0]["profile_id"] == 1


@pytest.mark.asyncio
async def test_existing_profile_is_found_by_provenance_not_display_name(
    first_run_state: FirstRunState,
    fake_hub: FakeMcpHub,
    fake_registry: FakeToolRegistry,
    tool_entries: list[dict[str, Any]],
) -> None:
    unrelated = _policy(state=first_run_state, tool_entries=tool_entries)
    unrelated["first_run_mcp_tools"]["setup_instance_id"] = "first_run:other"
    matching = _policy(state=first_run_state, tool_entries=tool_entries, selected_pack_ids=["writing"])
    fake_hub.permission_profiles = [
        _profile(1, unrelated, name=PROFILE_DISPLAY_NAME),
        _profile(2, matching, name="User renamed this"),
    ]
    fake_hub.policy_assignments = [_default_assignment(10, 1)]
    service = SetupMcpToolsService(hub=fake_hub, tool_registry=fake_registry)

    result = await service.apply_selection(
        state=first_run_state,
        request=McpToolsApplyRequest(selected_pack_ids=["research"], selected_addon_ids=[]),
    )

    assert result.profile_id == 2
    assert fake_hub.created_profiles == []
    assert fake_hub.updated_profiles[0]["profile_id"] == 2
    assert fake_hub.updated_assignments[0]["assignment_id"] == 10
    assert fake_hub.updated_assignments[0]["profile_id"] == 2


@pytest.mark.asyncio
async def test_apply_filters_broad_hub_lists_to_global_null_scope_and_default_null_target(
    first_run_state: FirstRunState,
    fake_hub: FakeMcpHub,
    fake_registry: FakeToolRegistry,
    tool_entries: list[dict[str, Any]],
) -> None:
    matching_policy = _policy(state=first_run_state, tool_entries=tool_entries, selected_pack_ids=["writing"])
    fake_hub.permission_profiles = [
        _profile(99, matching_policy, owner_scope_id=7),
        _profile(1, matching_policy),
    ]
    fake_hub.policy_assignments = [
        _default_assignment(9, 99, target_id="persona:stale"),
        _default_assignment(10, 99),
    ]
    service = SetupMcpToolsService(hub=fake_hub, tool_registry=fake_registry)

    result = await service.apply_selection(
        state=first_run_state,
        request=McpToolsApplyRequest(selected_pack_ids=["research"], selected_addon_ids=[]),
    )

    assert result.profile_id == 1
    assert result.assignment_id == 10
    assert fake_hub.updated_profiles[0]["profile_id"] == 1
    assert fake_hub.updated_assignments[0]["assignment_id"] == 10
    assert fake_hub.policy_assignments[0]["profile_id"] == 99


@pytest.mark.asyncio
async def test_repeated_apply_updates_generated_policy_when_last_hash_matches(
    first_run_state: FirstRunState,
    fake_hub: FakeMcpHub,
    fake_registry: FakeToolRegistry,
    tool_entries: list[dict[str, Any]],
) -> None:
    existing_policy = _policy(state=first_run_state, tool_entries=tool_entries, selected_pack_ids=["writing"])
    fake_hub.permission_profiles = [_profile(1, existing_policy)]
    fake_hub.policy_assignments = [_default_assignment(10, 1)]
    service = SetupMcpToolsService(hub=fake_hub, tool_registry=fake_registry)

    result = await service.apply_selection(
        state=first_run_state,
        request=McpToolsApplyRequest(
            selected_pack_ids=["research"],
            selected_addon_ids=["external_network_read"],
        ),
    )

    assert result.status == "applied"
    assert fake_hub.updated_profiles[0]["profile_id"] == 1
    policy_document = fake_hub.updated_profiles[0]["policy_document"]
    assert "web.search" in policy_document["allowed_tools"]
    assert (
        policy_document["first_run_mcp_tools"]["last_generated_hash"]
        == policy_document["first_run_mcp_tools"]["generated_policy_hash"]
    )


@pytest.mark.asyncio
async def test_manual_edit_conflict_returns_structured_conflict_without_overwrite(
    first_run_state: FirstRunState,
    fake_hub: FakeMcpHub,
    fake_registry: FakeToolRegistry,
    tool_entries: list[dict[str, Any]],
) -> None:
    edited_policy = _policy(state=first_run_state, tool_entries=tool_entries)
    edited_policy["allowed_tools"].append("notes.delete")
    fake_hub.permission_profiles = [_profile(1, edited_policy)]
    service = SetupMcpToolsService(hub=fake_hub, tool_registry=fake_registry)

    result = await service.apply_selection(
        state=first_run_state,
        request=McpToolsApplyRequest(selected_pack_ids=["writing"], selected_addon_ids=[]),
    )

    assert result.status == "conflict"
    assert result.conflict == {
        "reason": "profile_manually_changed",
        "profile_id": 1,
        "current_hash": compute_first_run_policy_hash(edited_policy),
        "expected_hash": edited_policy["first_run_mcp_tools"]["last_generated_hash"],
    }
    assert fake_hub.updated_profiles == []
    assert fake_hub.created_assignments == []


@pytest.mark.asyncio
async def test_keep_existing_records_ids_and_current_effective_count_without_profile_update(
    first_run_state: FirstRunState,
    fake_hub: FakeMcpHub,
    fake_registry: FakeToolRegistry,
    tool_entries: list[dict[str, Any]],
) -> None:
    edited_policy = _policy(state=first_run_state, tool_entries=tool_entries)
    edited_policy["allowed_tools"].append("notes.delete")
    fake_hub.permission_profiles = [_profile(1, edited_policy)]
    fake_hub.policy_assignments = [_default_assignment(10, 1)]
    service = SetupMcpToolsService(hub=fake_hub, tool_registry=fake_registry)

    result = await service.apply_selection(
        state=first_run_state,
        request=McpToolsApplyRequest(
            selected_pack_ids=["writing"],
            selected_addon_ids=[],
            conflict_resolution="keep_existing",
            profile_id=1,
        ),
    )

    assert result.status == "applied"
    assert result.profile_id == 1
    assert result.assignment_id == 10
    assert result.effective_tool_count == len(edited_policy["allowed_tools"])
    assert fake_hub.updated_profiles == []


@pytest.mark.asyncio
async def test_replace_existing_overwrites_only_after_explicit_request(
    first_run_state: FirstRunState,
    fake_hub: FakeMcpHub,
    fake_registry: FakeToolRegistry,
    tool_entries: list[dict[str, Any]],
) -> None:
    edited_policy = _policy(state=first_run_state, tool_entries=tool_entries)
    edited_policy["allowed_tools"].append("notes.delete")
    fake_hub.permission_profiles = [_profile(1, edited_policy)]
    fake_hub.policy_assignments = [_default_assignment(10, 1)]
    service = SetupMcpToolsService(hub=fake_hub, tool_registry=fake_registry)

    conflict = await service.apply_selection(
        state=first_run_state,
        request=McpToolsApplyRequest(selected_pack_ids=["writing"], selected_addon_ids=[]),
    )
    replaced = await service.apply_selection(
        state=first_run_state,
        request=McpToolsApplyRequest(
            selected_pack_ids=["writing"],
            selected_addon_ids=[],
            conflict_resolution="replace_existing",
            profile_id=1,
        ),
    )

    assert conflict.status == "conflict"
    assert replaced.status == "applied"
    assert fake_hub.updated_profiles[0]["policy_document"]["allowed_tools"] == ["notes.search"]
    replaced_policy = fake_hub.updated_profiles[0]["policy_document"]
    assert (
        replaced_policy["first_run_mcp_tools"]["last_generated_hash"]
        == replaced_policy["first_run_mcp_tools"]["generated_policy_hash"]
    )


@pytest.mark.asyncio
async def test_step_payload_contains_only_allowlisted_mcp_tools_fields(
    first_run_state: FirstRunState,
    fake_hub: FakeMcpHub,
    fake_registry: FakeToolRegistry,
) -> None:
    service = SetupMcpToolsService(hub=fake_hub, tool_registry=fake_registry)

    result = await service.apply_selection(
        state=first_run_state,
        request=McpToolsApplyRequest(
            selected_pack_ids=["research"],
            selected_addon_ids=["local_file_read"],
            confirmed_addon_ids=["destructive_actions"],
            confirmation_version=CONFIRMATION_VERSION,
        ),
    )

    assert set(result.step_payload) == {
        "acknowledged",
        "selected_pack_ids",
        "selected_addon_ids",
        "confirmed_addon_ids",
        "confirmation_version",
        "validation_state",
        "profile_id",
        "assignment_id",
        "catalog_version",
        "effective_tool_count",
        "validated_at",
        "validation_message",
        "last_validation_run_id",
    }
    assert result.step_payload["profile_id"] == result.profile_id
    assert result.step_payload["assignment_id"] == result.assignment_id
    assert result.step_payload["validation_state"] == "not_run"
    assert result.step_payload["catalog_version"] == CATALOG_VERSION
    assert "effective_tools" not in result.step_payload
    assert "policy_document" not in result.step_payload
    assert "endpoint_config" not in result.step_payload
