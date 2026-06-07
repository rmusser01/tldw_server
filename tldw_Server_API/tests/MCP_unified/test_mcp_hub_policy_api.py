from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import mcp_hub_management
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.services.mcp_hub_service import McpHubConflictError


def _make_principal(
    *,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=7,
        api_key_id=None,
        subject="7",
        token_type="access",
        jti=None,
        roles=roles or [],
        permissions=permissions or [],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )


class _FakePolicyService:
    def __init__(self) -> None:
        self.create_policy_assignment_error: Exception | None = None
        self.update_policy_assignment_error: Exception | None = None
        self.add_policy_assignment_workspace_error: Exception | None = None
        self.permission_profiles = [
            {
                "id": 5,
                "name": "Process Exec",
                "description": None,
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "mode": "custom",
                "policy_document": {"capabilities": ["process.execute"]},
                "path_scope_object_id": None,
                "is_active": True,
                "created_by": 7,
                "updated_by": 7,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
        self.policy_assignments = [
            {
                "id": 11,
                "target_type": "persona",
                "target_id": "researcher",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "profile_id": None,
                "path_scope_object_id": None,
                "inline_policy_document": {"capabilities": ["process.execute"]},
                "approval_policy_id": None,
                "is_active": True,
                "has_override": True,
                "override_id": 31,
                "override_active": True,
                "override_updated_at": datetime.now(timezone.utc).isoformat(),
                "created_by": 7,
                "updated_by": 7,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
        self.path_scope_objects = [
            {
                "id": 41,
                "name": "Docs Only",
                "description": None,
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "path_scope_document": {
                    "path_scope_mode": "workspace_root",
                    "path_scope_enforcement": "approval_required_when_unenforceable",
                    "path_allowlist_prefixes": ["docs"],
                },
                "is_active": True,
                "created_by": 7,
                "updated_by": 7,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
        self.workspace_set_objects = [
            {
                "id": 51,
                "name": "Primary Workspace Set",
                "description": None,
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "is_active": True,
                "created_by": 7,
                "updated_by": 7,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
        self.shared_workspace_entries = [
            {
                "id": 71,
                "workspace_id": "shared-docs",
                "display_name": "Shared Docs",
                "absolute_root": "/srv/shared/docs",
                "owner_scope_type": "team",
                "owner_scope_id": 21,
                "is_active": True,
                "created_by": 7,
                "updated_by": 7,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
        self.workspace_set_members: dict[int, list[dict]] = {
            51: [
                {
                    "workspace_set_object_id": 51,
                    "workspace_id": "workspace-alpha",
                    "created_by": 7,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
        }
        self.assignment_workspaces: dict[int, list[dict]] = {
            11: [
                {
                    "assignment_id": 11,
                    "workspace_id": "workspace-alpha",
                    "created_by": 7,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
        }
        self.policy_overrides = {
            11: {
                "id": 31,
                "assignment_id": 11,
                "override_policy_document": {"allowed_tools": ["remote.fetch"]},
                "is_active": True,
                "broadens_access": True,
                "grant_authority_snapshot": {"permissions": ["grant.tool.invoke"]},
                "created_by": 7,
                "updated_by": 7,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        self.approval_policies = [
            {
                "id": 17,
                "name": "Outside Profile",
                "description": None,
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "mode": "ask_outside_profile",
                "rules": {"duration_options": ["once", "session"]},
                "is_active": True,
                "created_by": 7,
                "updated_by": 7,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
        self.approval_decisions = []

    async def get_workspace_set_readiness_summary(self, *, workspace_set_object_id: int):
        if int(workspace_set_object_id) == 51:
            return {
                "is_multi_root_ready": False,
                "warning_codes": ["multi_root_overlap_warning"],
                "warning_message": "May overlap with another trusted root in multi-root assignments.",
                "conflicting_workspace_ids": ["workspace-alpha", "workspace-beta"],
                "conflicting_workspace_roots": ["/repo", "/repo/docs"],
                "unresolved_workspace_ids": [],
            }
        return {
            "is_multi_root_ready": True,
            "warning_codes": [],
            "warning_message": None,
            "conflicting_workspace_ids": [],
            "conflicting_workspace_roots": [],
            "unresolved_workspace_ids": [],
        }

    async def get_shared_workspace_readiness_summary(self, *, shared_workspace_id: int):
        if int(shared_workspace_id) == 71:
            return {
                "is_multi_root_ready": False,
                "warning_codes": ["multi_root_overlap_warning"],
                "warning_message": "May conflict with other visible shared roots in multi-root assignments.",
                "conflicting_workspace_ids": ["shared-docs", "shared-docs-nested"],
                "conflicting_workspace_roots": ["/srv/shared/docs", "/srv/shared/docs/archive"],
                "unresolved_workspace_ids": [],
            }
        return {
            "is_multi_root_ready": True,
            "warning_codes": [],
            "warning_message": None,
            "conflicting_workspace_ids": [],
            "conflicting_workspace_roots": [],
            "unresolved_workspace_ids": [],
        }

    async def list_permission_profiles(self, **_kwargs):
        return self.permission_profiles

    async def get_permission_profile(self, profile_id: int):
        for profile in self.permission_profiles:
            if int(profile.get("id") or 0) == int(profile_id):
                return dict(profile)
        return None

    async def create_permission_profile(self, **kwargs):
        profile = {
            "id": 5,
            "name": kwargs["name"],
            "description": kwargs.get("description"),
            "owner_scope_type": kwargs["owner_scope_type"],
            "owner_scope_id": kwargs.get("owner_scope_id"),
            "mode": kwargs["mode"],
            "policy_document": kwargs["policy_document"],
            "path_scope_object_id": kwargs.get("path_scope_object_id"),
            "is_active": kwargs.get("is_active", True),
            "created_by": kwargs.get("actor_id"),
            "updated_by": kwargs.get("actor_id"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.permission_profiles = [profile]
        return profile

    async def update_permission_profile(self, profile_id: int, **kwargs):
        if profile_id != 5:
            return None
        profile = dict(self.permission_profiles[0])
        if kwargs.get("name") is not None:
            profile["name"] = kwargs["name"]
        if kwargs.get("description") is not None:
            profile["description"] = kwargs["description"]
        if kwargs.get("mode") is not None:
            profile["mode"] = kwargs["mode"]
        if kwargs.get("policy_document") is not None:
            profile["policy_document"] = kwargs["policy_document"]
        if "path_scope_object_id" in kwargs:
            profile["path_scope_object_id"] = kwargs.get("path_scope_object_id")
        if kwargs.get("is_active") is not None:
            profile["is_active"] = kwargs["is_active"]
        profile["updated_by"] = kwargs.get("actor_id")
        self.permission_profiles = [profile]
        return profile

    async def delete_permission_profile(self, profile_id: int, *, actor_id: int | None):
        return profile_id == 5 and actor_id == 7

    async def list_policy_assignments(self, **_kwargs):
        return list(self.policy_assignments)

    async def get_policy_assignment(self, assignment_id: int):
        for assignment in self.policy_assignments:
            if int(assignment.get("id") or 0) == int(assignment_id):
                return dict(assignment)
        return None

    async def create_policy_assignment(self, **kwargs):
        if self.create_policy_assignment_error is not None:
            raise self.create_policy_assignment_error
        assignment = {
            "id": 11,
            "target_type": kwargs["target_type"],
            "target_id": kwargs.get("target_id"),
            "owner_scope_type": kwargs["owner_scope_type"],
            "owner_scope_id": kwargs.get("owner_scope_id"),
            "profile_id": kwargs.get("profile_id"),
            "path_scope_object_id": kwargs.get("path_scope_object_id"),
            "workspace_source_mode": kwargs.get("workspace_source_mode"),
            "workspace_set_object_id": kwargs.get("workspace_set_object_id"),
            "inline_policy_document": kwargs["inline_policy_document"],
            "approval_policy_id": kwargs.get("approval_policy_id"),
            "is_active": kwargs.get("is_active", True),
            "has_override": False,
            "override_id": None,
            "override_active": False,
            "override_updated_at": None,
            "created_by": kwargs.get("actor_id"),
            "updated_by": kwargs.get("actor_id"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.policy_assignments = [assignment]
        return assignment

    async def update_policy_assignment(self, assignment_id: int, **kwargs):
        if self.update_policy_assignment_error is not None:
            raise self.update_policy_assignment_error
        if assignment_id != 11:
            return None
        assignment = dict(self.policy_assignments[0])
        if kwargs.get("target_type") is not None:
            assignment["target_type"] = kwargs["target_type"]
        if kwargs.get("target_id") is not None:
            assignment["target_id"] = kwargs["target_id"]
        if kwargs.get("inline_policy_document") is not None:
            assignment["inline_policy_document"] = kwargs["inline_policy_document"]
        if kwargs.get("profile_id") is not None:
            assignment["profile_id"] = kwargs["profile_id"]
        if "path_scope_object_id" in kwargs:
            assignment["path_scope_object_id"] = kwargs.get("path_scope_object_id")
        if "workspace_source_mode" in kwargs:
            assignment["workspace_source_mode"] = kwargs.get("workspace_source_mode")
        if "workspace_set_object_id" in kwargs:
            assignment["workspace_set_object_id"] = kwargs.get("workspace_set_object_id")
        if kwargs.get("approval_policy_id") is not None:
            assignment["approval_policy_id"] = kwargs["approval_policy_id"]
        if kwargs.get("is_active") is not None:
            assignment["is_active"] = kwargs["is_active"]
        assignment["updated_by"] = kwargs.get("actor_id")
        override = self.policy_overrides.get(assignment_id)
        assignment["has_override"] = override is not None
        assignment["override_id"] = override["id"] if override else None
        assignment["override_active"] = bool(override and override.get("is_active"))
        assignment["override_updated_at"] = override.get("updated_at") if override else None
        self.policy_assignments = [assignment]
        return assignment

    async def delete_policy_assignment(self, assignment_id: int, *, actor_id: int | None):
        self.policy_overrides.pop(assignment_id, None)
        return assignment_id == 11 and actor_id == 7

    async def get_policy_override(self, assignment_id: int):
        return dict(self.policy_overrides[assignment_id]) if assignment_id in self.policy_overrides else None

    async def upsert_policy_override(self, assignment_id: int, **kwargs):
        if assignment_id != 11:
            return None
        existing = self.policy_overrides.get(assignment_id)
        override_id = existing["id"] if existing else 31
        row = {
            "id": override_id,
            "assignment_id": assignment_id,
            "override_policy_document": kwargs["override_policy_document"],
            "is_active": kwargs.get("is_active", True),
            "broadens_access": kwargs.get("broadens_access", False),
            "grant_authority_snapshot": kwargs.get("grant_authority_snapshot", {}),
            "created_by": (existing or {}).get("created_by", kwargs.get("actor_id")),
            "updated_by": kwargs.get("actor_id"),
            "created_at": (existing or {}).get("created_at", datetime.now(timezone.utc).isoformat()),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.policy_overrides[assignment_id] = row
        self.policy_assignments = [
            {
                **assignment,
                "has_override": int(assignment.get("id") or 0) == assignment_id,
                "override_id": override_id if int(assignment.get("id") or 0) == assignment_id else None,
                "override_active": bool(row.get("is_active")) if int(assignment.get("id") or 0) == assignment_id else False,
                "override_updated_at": row["updated_at"] if int(assignment.get("id") or 0) == assignment_id else None,
            }
            for assignment in self.policy_assignments
        ]
        return dict(row)

    async def delete_policy_override(self, assignment_id: int, *, actor_id: int | None):
        deleted = assignment_id in self.policy_overrides and actor_id == 7
        self.policy_overrides.pop(assignment_id, None)
        self.policy_assignments = [
            {
                **assignment,
                "has_override": False if int(assignment.get("id") or 0) == assignment_id else assignment.get("has_override", False),
                "override_id": None if int(assignment.get("id") or 0) == assignment_id else assignment.get("override_id"),
                "override_active": False if int(assignment.get("id") or 0) == assignment_id else assignment.get("override_active", False),
                "override_updated_at": None if int(assignment.get("id") or 0) == assignment_id else assignment.get("override_updated_at"),
            }
            for assignment in self.policy_assignments
        ]
        return deleted

    async def list_approval_policies(self, **_kwargs):
        return list(self.approval_policies)

    async def create_approval_policy(self, **kwargs):
        policy = {
            "id": 17,
            "name": kwargs["name"],
            "description": kwargs.get("description"),
            "owner_scope_type": kwargs["owner_scope_type"],
            "owner_scope_id": kwargs.get("owner_scope_id"),
            "mode": kwargs["mode"],
            "rules": kwargs["rules"],
            "is_active": kwargs.get("is_active", True),
            "created_by": kwargs.get("actor_id"),
            "updated_by": kwargs.get("actor_id"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.approval_policies = [policy]
        return policy

    async def get_approval_policy(self, approval_policy_id: int):
        for policy in self.approval_policies:
            if int(policy.get("id") or 0) == int(approval_policy_id):
                return dict(policy)
        return None

    async def update_approval_policy(self, approval_policy_id: int, **kwargs):
        if approval_policy_id != 17:
            return None
        policy = dict(self.approval_policies[0])
        if kwargs.get("name") is not None:
            policy["name"] = kwargs["name"]
        if kwargs.get("description") is not None:
            policy["description"] = kwargs["description"]
        if kwargs.get("mode") is not None:
            policy["mode"] = kwargs["mode"]
        if kwargs.get("rules") is not None:
            policy["rules"] = kwargs["rules"]
        if kwargs.get("is_active") is not None:
            policy["is_active"] = kwargs["is_active"]
        policy["updated_by"] = kwargs.get("actor_id")
        self.approval_policies = [policy]
        return policy

    async def delete_approval_policy(self, approval_policy_id: int, *, actor_id: int | None):
        return approval_policy_id == 17 and actor_id == 7

    async def record_approval_decision(self, **kwargs):
        decision = {
            "id": 23,
            "approval_policy_id": kwargs.get("approval_policy_id"),
            "context_key": kwargs["context_key"],
            "conversation_id": kwargs.get("conversation_id"),
            "tool_name": kwargs["tool_name"],
            "scope_key": kwargs["scope_key"],
            "decision": kwargs["decision"],
            "expires_at": kwargs.get("expires_at"),
            "consume_on_match": bool(kwargs.get("consume_on_match")),
            "consumed_at": kwargs.get("consumed_at"),
            "created_by": kwargs.get("actor_id"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.approval_decisions = [decision]
        return decision

    async def list_path_scope_objects(self, **_kwargs):
        return list(self.path_scope_objects)

    async def get_path_scope_object(self, path_scope_object_id: int):
        for row in self.path_scope_objects:
            if int(row.get("id") or 0) == int(path_scope_object_id):
                return dict(row)
        return None

    async def validate_path_scope_object_reference(
        self,
        *,
        path_scope_object_id: int | None,
        target_scope_type: str,
        target_scope_id: int | None,
    ):
        if path_scope_object_id is None:
            return None
        row = await self.get_path_scope_object(path_scope_object_id)
        if row is None:
            raise HTTPException(status_code=404, detail="mcp_path_scope_object not found")
        return row

    async def create_path_scope_object(self, **kwargs):
        row = {
            "id": 41,
            "name": kwargs["name"],
            "description": kwargs.get("description"),
            "owner_scope_type": kwargs["owner_scope_type"],
            "owner_scope_id": kwargs.get("owner_scope_id"),
            "path_scope_document": kwargs["path_scope_document"],
            "is_active": kwargs.get("is_active", True),
            "created_by": kwargs.get("actor_id"),
            "updated_by": kwargs.get("actor_id"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.path_scope_objects = [row]
        return row

    async def update_path_scope_object(self, path_scope_object_id: int, **kwargs):
        if path_scope_object_id != 41:
            return None
        row = dict(self.path_scope_objects[0])
        if kwargs.get("name") is not None:
            row["name"] = kwargs["name"]
        if kwargs.get("description") is not None:
            row["description"] = kwargs["description"]
        if kwargs.get("path_scope_document") is not None:
            row["path_scope_document"] = kwargs["path_scope_document"]
        if kwargs.get("is_active") is not None:
            row["is_active"] = kwargs["is_active"]
        row["updated_by"] = kwargs.get("actor_id")
        self.path_scope_objects = [row]
        return row

    async def delete_path_scope_object(self, path_scope_object_id: int, *, actor_id: int | None):
        return path_scope_object_id == 41 and actor_id == 7

    async def list_workspace_set_objects(self, **_kwargs):
        return list(self.workspace_set_objects)

    async def get_workspace_set_object(self, workspace_set_object_id: int):
        for row in self.workspace_set_objects:
            if int(row.get("id") or 0) == int(workspace_set_object_id):
                return dict(row)
        return None

    async def validate_workspace_set_object_reference(
        self,
        *,
        workspace_set_object_id: int | None,
        target_scope_type: str,
        target_scope_id: int | None,
    ):
        if workspace_set_object_id is None:
            return None
        row = await self.get_workspace_set_object(workspace_set_object_id)
        if row is None:
            raise HTTPException(status_code=404, detail="mcp_workspace_set_object not found")
        if str(target_scope_type or "") != "user" or row.get("owner_scope_id") != target_scope_id:
            raise HTTPException(status_code=400, detail="invalid workspace set scope")
        return row

    async def create_workspace_set_object(self, **kwargs):
        row = {
            "id": 51,
            "name": kwargs["name"],
            "description": kwargs.get("description"),
            "owner_scope_type": kwargs["owner_scope_type"],
            "owner_scope_id": kwargs.get("owner_scope_id"),
            "is_active": kwargs.get("is_active", True),
            "created_by": kwargs.get("actor_id"),
            "updated_by": kwargs.get("actor_id"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.workspace_set_objects = [row]
        return row

    async def update_workspace_set_object(self, workspace_set_object_id: int, **kwargs):
        if workspace_set_object_id != 51:
            return None
        row = dict(self.workspace_set_objects[0])
        if kwargs.get("name") is not None:
            row["name"] = kwargs["name"]
        if kwargs.get("description") is not None:
            row["description"] = kwargs["description"]
        if kwargs.get("is_active") is not None:
            row["is_active"] = kwargs["is_active"]
        row["updated_by"] = kwargs.get("actor_id")
        self.workspace_set_objects = [row]
        return row

    async def delete_workspace_set_object(self, workspace_set_object_id: int, *, actor_id: int | None):
        return workspace_set_object_id == 51 and actor_id == 7

    async def list_workspace_set_members(self, workspace_set_object_id: int):
        return [dict(row) for row in self.workspace_set_members.get(workspace_set_object_id, [])]

    async def add_workspace_set_member(
        self,
        workspace_set_object_id: int,
        *,
        workspace_id: str,
        actor_id: int | None,
    ):
        rows = self.workspace_set_members.setdefault(workspace_set_object_id, [])
        if any(str(row.get("workspace_id")) == workspace_id for row in rows):
            raise McpHubConflictError("Workspace already attached to workspace set")
        row = {
            "workspace_set_object_id": workspace_set_object_id,
            "workspace_id": workspace_id,
            "created_by": actor_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        return dict(row)

    async def delete_workspace_set_member(
        self,
        workspace_set_object_id: int,
        workspace_id: str,
        *,
        actor_id: int | None,
    ):
        if actor_id != 7:
            return False
        rows = self.workspace_set_members.get(workspace_set_object_id, [])
        next_rows = [row for row in rows if str(row.get("workspace_id")) != workspace_id]
        deleted = len(next_rows) != len(rows)
        self.workspace_set_members[workspace_set_object_id] = next_rows
        return deleted

    async def list_shared_workspace_entries(self, **_kwargs):
        return list(self.shared_workspace_entries)

    async def get_shared_workspace_entry(self, shared_workspace_id: int):
        for row in self.shared_workspace_entries:
            if int(row.get("id") or 0) == int(shared_workspace_id):
                return dict(row)
        return None

    async def create_shared_workspace_entry(self, **kwargs):
        row = {
            "id": 71,
            "workspace_id": kwargs["workspace_id"],
            "display_name": kwargs.get("display_name"),
            "absolute_root": kwargs["absolute_root"],
            "owner_scope_type": kwargs["owner_scope_type"],
            "owner_scope_id": kwargs.get("owner_scope_id"),
            "is_active": kwargs.get("is_active", True),
            "created_by": kwargs.get("actor_id"),
            "updated_by": kwargs.get("actor_id"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.shared_workspace_entries = [row]
        return row

    async def update_shared_workspace_entry(self, shared_workspace_id: int, **kwargs):
        if shared_workspace_id != 71:
            return None
        row = dict(self.shared_workspace_entries[0])
        for key in ("workspace_id", "display_name", "absolute_root", "owner_scope_type", "owner_scope_id", "is_active"):
            if key in kwargs and kwargs.get(key) is not None:
                row[key] = kwargs[key]
        row["updated_by"] = kwargs.get("actor_id")
        self.shared_workspace_entries = [row]
        return row

    async def delete_shared_workspace_entry(self, shared_workspace_id: int, *, actor_id: int | None):
        return shared_workspace_id == 71 and actor_id == 7

    async def list_policy_assignment_workspaces(self, assignment_id: int):
        return [dict(row) for row in self.assignment_workspaces.get(assignment_id, [])]

    async def add_policy_assignment_workspace(self, assignment_id: int, *, workspace_id: str, actor_id: int | None):
        if self.add_policy_assignment_workspace_error is not None:
            raise self.add_policy_assignment_workspace_error
        rows = self.assignment_workspaces.setdefault(assignment_id, [])
        if any(str(row.get("workspace_id")) == workspace_id for row in rows):
            raise McpHubConflictError("Workspace already attached to assignment")
        row = {
            "assignment_id": assignment_id,
            "workspace_id": workspace_id,
            "created_by": actor_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        return dict(row)

    async def delete_policy_assignment_workspace(self, assignment_id: int, workspace_id: str, *, actor_id: int | None):
        if actor_id != 7:
            return False
        rows = self.assignment_workspaces.get(assignment_id, [])
        next_rows = [row for row in rows if str(row.get("workspace_id")) != workspace_id]
        deleted = len(next_rows) != len(rows)
        self.assignment_workspaces[assignment_id] = next_rows
        return deleted


class _FakePolicyResolver:
    async def resolve_for_context(self, *, user_id, metadata):
        return {
            "enabled": True,
            "allowed_tools": ["Bash(git *)"],
            "denied_tools": ["Bash(rm *)"],
            "capabilities": ["process.execute"],
            "approval_policy_id": 17,
            "approval_mode": "ask_outside_profile",
            "policy_document": {
                "allowed_tools": ["Bash(git *)"],
                "denied_tools": ["Bash(rm *)"],
                "capabilities": ["process.execute"],
                "path_scope_mode": "workspace_root",
                "path_scope_enforcement": "approval_required_when_unenforceable",
            },
            "sources": [
                {
                    "assignment_id": 11,
                    "target_type": "persona",
                    "target_id": metadata.get("persona_id"),
                    "owner_scope_type": "user",
                    "owner_scope_id": user_id,
                    "profile_id": 5
                }
            ],
            "provenance": [
                {
                    "field": "allowed_tools",
                    "value": ["Bash(git *)"],
                    "source_kind": "assignment_inline",
                    "assignment_id": 11,
                    "profile_id": 5,
                    "override_id": None,
                    "effect": "merged"
                },
                {
                    "field": "allowed_tools",
                    "value": ["remote.fetch"],
                    "source_kind": "assignment_override",
                    "assignment_id": 11,
                    "profile_id": 5,
                    "override_id": 31,
                    "effect": "merged"
                }
            ]
        }


class _FakePathPermissionPreviewService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def preview_effective_path_permission(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "tool_name": kwargs["tool_name"],
            "requested_action": kwargs["action"],
            "requested_path": kwargs["path"],
            "normalized_path": kwargs["path"],
            "outcome": "allow",
            "within_scope": True,
            "reason_code": None,
            "selected_assignment_id": 11,
            "profile_id": 5,
            "grant_source": "path_grants",
            "grant_outcome": "allowed",
            "matched_grant_prefix": "documents",
            "matched_grant_effect": "allow",
            "path_scope_mode": "workspace_root",
            "workspace_id": "workspace-alpha",
            "redacted": True,
            "path_decisions": [
                {
                    "requested_action": kwargs["action"],
                    "normalized_path": kwargs["path"],
                    "grant_outcome": "allowed",
                    "grant_source": "path_grants",
                    "matched_grant_prefix": "documents",
                    "matched_grant_effect": "allow",
                    "reason_code": None,
                    "redacted": True,
                }
            ],
        }


class _StructuredBadRequestError(BadRequestError):
    def __init__(self, detail: dict[str, object]) -> None:
        super().__init__(str(detail.get("message") or detail.get("code") or "invalid request"))
        self.detail = detail


class _FakeToolRegistryService:
    def __init__(self) -> None:
        self.list_entries_calls = 0
        self.entries = [
            {
                "tool_name": "notes.search",
                "display_name": "notes.search",
                "module": "notes",
                "category": "search",
                "risk_class": "low",
                "capabilities": ["filesystem.read"],
                "mutates_state": False,
                "uses_filesystem": False,
                "uses_processes": False,
                "uses_network": False,
                "uses_credentials": False,
                "supports_arguments_preview": True,
                "path_boundable": False,
                "path_argument_hints": ["path"],
                "metadata_source": "explicit",
                "metadata_warnings": [],
            },
            {
                "tool_name": "sandbox.run",
                "display_name": "sandbox.run",
                "module": "sandbox",
                "category": "execution",
                "risk_class": "high",
                "capabilities": ["process.execute"],
                "mutates_state": True,
                "uses_filesystem": True,
                "uses_processes": True,
                "uses_network": False,
                "uses_credentials": False,
                "supports_arguments_preview": True,
                "path_boundable": False,
                "path_argument_hints": [],
                "metadata_source": "heuristic",
                "metadata_warnings": ["Derived from tool category"],
            },
        ]

    async def list_entries(self):
        self.list_entries_calls += 1
        return list(self.entries)

    async def list_modules(self):
        return (await self.get_summary())["modules"]

    async def get_summary(self):
        entries = await self.list_entries()
        return {
            "entries": entries,
            "modules": [
                {
                    "module": "notes",
                    "display_name": "notes",
                    "tool_count": 1,
                    "risk_summary": {"low": 1, "medium": 0, "high": 0, "unclassified": 0},
                    "metadata_warnings": [],
                },
                {
                    "module": "sandbox",
                    "display_name": "sandbox",
                    "tool_count": 1,
                    "risk_summary": {"low": 0, "medium": 0, "high": 1, "unclassified": 0},
                    "metadata_warnings": ["Derived metadata present"],
                },
            ],
        }


def _build_app(
    principal: AuthPrincipal,
    service: _FakePolicyService | None = None,
    resolver: _FakePolicyResolver | None = None,
    *,
    tool_registry: _FakeToolRegistryService | None = None,
    path_permission_preview: _FakePathPermissionPreviewService | None = None,
    rate_limit_calls: list[str] | None = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(mcp_hub_management.router, prefix="/api/v1")

    async def _fake_get_auth_principal(_request: Request) -> AuthPrincipal:  # type: ignore[override]
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[mcp_hub_management.get_mcp_hub_service] = lambda: service or _FakePolicyService()
    app.dependency_overrides[mcp_hub_management.get_mcp_hub_policy_resolver_dep] = (
        lambda: resolver or _FakePolicyResolver()
    )
    if hasattr(mcp_hub_management, "get_mcp_hub_tool_registry_dep"):
        app.dependency_overrides[mcp_hub_management.get_mcp_hub_tool_registry_dep] = (
            lambda: tool_registry or _FakeToolRegistryService()
        )
    if path_permission_preview is not None and hasattr(
        mcp_hub_management,
        "get_mcp_hub_path_enforcement_service_dep",
    ):
        app.dependency_overrides[mcp_hub_management.get_mcp_hub_path_enforcement_service_dep] = (
            lambda: path_permission_preview
        )
    if rate_limit_calls is not None:
        async def _fake_check_rate_limit(_request: Request) -> None:
            rate_limit_calls.append("called")

        app.dependency_overrides[mcp_hub_management.check_rate_limit] = _fake_check_rate_limit
    return app


def test_create_permission_profile_requires_grant_authority_for_capabilities() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/permission-profiles",
            json={
                "name": "Process Exec",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "mode": "custom",
                "policy_document": {"capabilities": ["process.execute"]},
                "is_active": True,
            },
        )

    assert resp.status_code == 403
    assert "grant.process.execute" in resp.json()["detail"]


def test_create_permission_profile_rejects_string_capability_without_grant_authority() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/permission-profiles",
            json={
                "name": "Process Exec",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "mode": "custom",
                "policy_document": {"capabilities": "process.execute"},
                "is_active": True,
            },
        )

    assert resp.status_code == 403
    assert "grant.process.execute" in resp.json()["detail"]


def test_create_permission_profile_requires_grant_authority_for_allowed_tools() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/permission-profiles",
            json={
                "name": "Git Only",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "mode": "custom",
                "policy_document": {"allowed_tools": ["Bash(git *)"]},
                "is_active": True,
            },
        )

    assert resp.status_code == 403
    assert "grant.tool.invoke" in resp.json()["detail"]


def test_create_permission_profile_returns_created_payload_when_grant_is_present() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute"],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/permission-profiles",
            json={
                "name": "Process Exec",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "mode": "custom",
                "policy_document": {"capabilities": ["process.execute"]},
                "is_active": True,
            },
        )

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["name"] == "Process Exec"
    assert payload["policy_document"]["capabilities"] == ["process.execute"]
    assert payload["owner_scope_type"] == "user"


def test_create_permission_profile_normalizes_path_allowlist_prefixes() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE, "grant.filesystem.read"],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/permission-profiles",
            json={
                "name": "Workspace Narrow",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "mode": "custom",
                "policy_document": {
                    "capabilities": ["filesystem.read"],
                    "path_scope_mode": "workspace_root",
                    "path_scope_enforcement": "approval_required_when_unenforceable",
                    "path_allowlist_prefixes": ["./src/", "docs\\\\api", "src"],
                },
                "is_active": True,
            },
        )

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["policy_document"]["path_allowlist_prefixes"] == ["docs/api", "src"]


def test_create_path_scope_object_returns_created_payload() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/path-scope-objects",
            json={
                "name": "Docs Only",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "path_scope_document": {
                    "path_scope_mode": "workspace_root",
                    "path_scope_enforcement": "approval_required_when_unenforceable",
                    "path_allowlist_prefixes": ["./docs/", "docs/api"],
                },
                "is_active": True,
            },
        )

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["name"] == "Docs Only"
    assert payload["path_scope_document"]["path_allowlist_prefixes"] == ["docs", "docs/api"]


def test_add_assignment_workspace_returns_created_payload() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/policy-assignments/11/workspaces",
            json={"workspace_id": "workspace-beta"},
        )

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["assignment_id"] == 11
    assert payload["workspace_id"] == "workspace-beta"


def test_add_assignment_workspace_rejects_duplicate_workspace_id() -> None:
    service = _FakePolicyService()
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE],
        ),
        service=service,
    )

    with TestClient(app) as client:
        first = client.post(
            "/api/v1/mcp/hub/policy-assignments/11/workspaces",
            json={"workspace_id": "workspace-gamma"},
        )
        second = client.post(
            "/api/v1/mcp/hub/policy-assignments/11/workspaces",
            json={"workspace_id": "workspace-gamma"},
        )

    assert first.status_code == 201
    assert second.status_code == 409


def test_create_workspace_set_object_returns_created_payload() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/workspace-set-objects",
            json={
                "name": "Primary Workspace Set",
                "owner_scope_type": "user",
                "is_active": True,
            },
        )

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["name"] == "Primary Workspace Set"
    assert payload["owner_scope_type"] == "user"
    assert payload["owner_scope_id"] == 7
    assert "readiness_summary" in payload


def test_create_shared_workspace_entry_returns_created_payload() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/shared-workspaces",
            json={
                "workspace_id": "shared-docs",
                "display_name": "Shared Docs",
                "absolute_root": "/srv/shared/docs",
                "owner_scope_type": "team",
                "owner_scope_id": 21,
                "is_active": True,
            },
        )

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["workspace_id"] == "shared-docs"
    assert payload["owner_scope_type"] == "team"
    assert payload["owner_scope_id"] == 21
    assert "readiness_summary" in payload


def test_create_shared_workspace_entry_rejects_user_scope() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/shared-workspaces",
            json={
                "workspace_id": "local-docs",
                "display_name": "Local Docs",
                "absolute_root": "/srv/shared/docs",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "is_active": True,
            },
        )

    assert resp.status_code == 400


def test_create_policy_assignment_accepts_named_workspace_source_reference() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/policy-assignments",
            json={
                "target_type": "persona",
                "target_id": "researcher",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "profile_id": None,
                "workspace_source_mode": "named",
                "workspace_set_object_id": 51,
                "inline_policy_document": {},
                "approval_policy_id": None,
                "is_active": True,
            },
        )

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["workspace_source_mode"] == "named"
    assert payload["workspace_set_object_id"] == 51


def test_create_policy_assignment_returns_structured_overlap_detail() -> None:
    service = _FakePolicyService()
    service.create_policy_assignment_error = _StructuredBadRequestError(
        {
            "code": "assignment_multi_root_overlap",
            "message": "Named workspace source contains overlapping roots for multi-root execution.",
            "conflicting_workspace_ids": ["workspace-alpha", "workspace-beta"],
            "conflicting_workspace_roots": ["/repo", "/repo/docs"],
            "workspace_source_mode": "named",
        }
    )
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE],
        ),
        service=service,
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/policy-assignments",
            json={
                "target_type": "persona",
                "target_id": "researcher",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "profile_id": None,
                "workspace_source_mode": "named",
                "workspace_set_object_id": 51,
                "inline_policy_document": {},
                "approval_policy_id": None,
                "is_active": True,
            },
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == {
        "code": "assignment_multi_root_overlap",
        "message": "Named workspace source contains overlapping roots for multi-root execution.",
        "conflicting_workspace_ids": ["workspace-alpha", "workspace-beta"],
        "conflicting_workspace_roots": ["/repo", "/repo/docs"],
        "workspace_source_mode": "named",
    }


def test_create_policy_assignment_accepts_parent_scope_path_scope_object_reference() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/policy-assignments",
            json={
                "target_type": "persona",
                "target_id": "researcher",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "profile_id": None,
                "path_scope_object_id": 41,
                "inline_policy_document": {},
                "approval_policy_id": None,
                "is_active": True,
            },
        )

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["path_scope_object_id"] == 41


def test_create_policy_assignment_returns_created_payload() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute"],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/policy-assignments",
            json={
                "target_type": "persona",
                "target_id": "researcher",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "profile_id": None,
                "inline_policy_document": {"capabilities": ["process.execute"]},
                "approval_policy_id": None,
                "is_active": True,
            },
        )

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["target_type"] == "persona"
    assert payload["target_id"] == "researcher"
    assert payload["inline_policy_document"]["capabilities"] == ["process.execute"]


def test_create_approval_policy_returns_created_payload() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/approval-policies",
            json={
                "name": "Outside Profile",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "mode": "ask_outside_profile",
                "rules": {"duration_options": ["once", "session"]},
                "is_active": True,
            },
        )

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["name"] == "Outside Profile"
    assert payload["mode"] == "ask_outside_profile"
    assert payload["rules"]["duration_options"] == ["once", "session"]


def test_create_approval_policy_rejects_unsupported_duration_option() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[SYSTEM_CONFIGURE],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/approval-policies",
            json={
                "name": "Invalid Durations",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "mode": "ask_outside_profile",
                "rules": {"duration_options": ["forever"]},
                "is_active": True,
            },
        )

    assert resp.status_code == 422
    assert "duration_options" in resp.json()["detail"]


def test_record_approval_decision_returns_created_payload() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/approval-decisions",
            json={
                "approval_policy_id": 17,
                "context_key": "user:7|group:|persona:researcher",
                "conversation_id": "sess-1",
                "tool_name": "Bash",
                "scope_key": "tool:Bash|command:abc123",
                "decision": "approved",
                "duration": "once",
            },
        )

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["approval_policy_id"] == 17
    assert payload["context_key"] == "user:7|group:|persona:researcher"
    assert payload["decision"] == "approved"
    assert payload["consume_on_match"] is True


def test_record_approval_decision_for_session_duration_sets_server_side_expiry() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/approval-decisions",
            json={
                "approval_policy_id": 17,
                "context_key": "user:7|group:|persona:researcher",
                "conversation_id": "sess-1",
                "tool_name": "Bash",
                "scope_key": "tool:Bash|command:abc123",
                "decision": "approved",
                "duration": "session",
            },
        )

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["consume_on_match"] is False
    assert payload["expires_at"] is not None


def test_record_approval_decision_rejects_duration_not_allowed_by_policy() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/approval-decisions",
            json={
                "approval_policy_id": 17,
                "context_key": "user:7|group:|persona:researcher",
                "conversation_id": "sess-1",
                "tool_name": "Bash",
                "scope_key": "tool:Bash|command:abc123",
                "decision": "approved",
                "duration": "conversation",
            },
        )

    assert resp.status_code == 422
    assert "duration" in resp.json()["detail"]


def test_record_approval_decision_rejects_scoped_duration_without_conversation_id() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/approval-decisions",
            json={
                "approval_policy_id": 17,
                "context_key": "user:7|group:|persona:researcher",
                "tool_name": "Bash",
                "scope_key": "tool:Bash|command:abc123",
                "decision": "approved",
                "duration": "session",
            },
        )

    assert resp.status_code == 422
    assert "conversation_id" in resp.json()["detail"]


def test_record_approval_decision_rejects_foreign_context_key() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[],
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/approval-decisions",
            json={
                "approval_policy_id": 17,
                "context_key": "user:9|group:|persona:researcher",
                "conversation_id": "sess-1",
                "tool_name": "Bash",
                "scope_key": "tool:Bash|command:abc123",
                "decision": "approved",
                "duration": "once",
            },
        )

    assert resp.status_code == 403


def test_get_effective_policy_returns_resolved_payload() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[],
        )
    )

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/hub/effective-policy?persona_id=researcher&group_id=team-red")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["enabled"] is True
    assert payload["allowed_tools"] == ["Bash(git *)"]
    assert payload["approval_mode"] == "ask_outside_profile"
    assert payload["policy_document"]["path_scope_mode"] == "workspace_root"
    assert payload["policy_document"]["path_scope_enforcement"] == "approval_required_when_unenforceable"
    assert payload["sources"][0]["target_id"] == "researcher"
    assert payload["provenance"][1]["source_kind"] == "assignment_override"


def test_preview_effective_permission_returns_redacted_path_decision() -> None:
    preview_service = _FakePathPermissionPreviewService()
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[],
        ),
        path_permission_preview=preview_service,
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/effective-permission-preview",
            json={
                "persona_id": "researcher",
                "group_id": "team-red",
                "workspace_id": "workspace-alpha",
                "tool_name": "fs.write",
                "action": "write",
                "path": "documents/story.md",
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["tool_name"] == "fs.write"
    assert payload["requested_action"] == "write"
    assert payload["normalized_path"] == "documents/story.md"
    assert payload["outcome"] == "allow"
    assert payload["grant_source"] == "path_grants"
    assert payload["matched_grant_prefix"] == "documents"
    assert payload["redacted"] is True
    assert "/tmp/" not in repr(payload)
    assert preview_service.calls[0]["effective_policy"]["policy_document"]["path_scope_mode"] == "workspace_root"
    assert preview_service.calls[0]["context"].metadata["workspace_id"] == "workspace-alpha"


def test_preview_effective_permission_redacts_rejected_absolute_path() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[],
        ),
        path_permission_preview=McpHubPathEnforcementService(path_scope_service=object()),
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/effective-permission-preview",
            json={
                "persona_id": "researcher",
                "group_id": "team-red",
                "workspace_id": "workspace-alpha",
                "tool_name": "fs.read",
                "action": "read",
                "path": "/tmp/secrets.txt",
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["outcome"] == "deny"
    assert payload["reason_code"] == "path_must_be_workspace_relative"
    assert payload["requested_path"] == "<redacted>"
    assert "/tmp/secrets.txt" not in repr(payload)


def test_list_policy_assignments_includes_override_summary_fields() -> None:
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute", "grant.tool.invoke"],
        )
    )

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/hub/policy-assignments")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload[0]["has_override"] is True
    assert payload[0]["override_active"] is True
    assert payload[0]["override_id"] == 31


def test_get_policy_assignment_override_returns_payload() -> None:
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute", "grant.tool.invoke"],
        )
    )

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/hub/policy-assignments/11/override")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["assignment_id"] == 11
    assert payload["override_policy_document"]["allowed_tools"] == ["remote.fetch"]
    assert payload["is_active"] is True


def test_put_policy_assignment_override_returns_payload() -> None:
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute", "grant.tool.invoke"],
        )
    )

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/mcp/hub/policy-assignments/11/override",
            json={
                "override_policy_document": {"allowed_tools": ["remote.fetch"]},
                "is_active": True,
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["assignment_id"] == 11
    assert payload["override_policy_document"]["allowed_tools"] == ["remote.fetch"]
    assert payload["is_active"] is True


def test_put_policy_assignment_override_requires_grant_authority_for_broadened_delta() -> None:
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE],
        )
    )

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/mcp/hub/policy-assignments/11/override",
            json={
                "override_policy_document": {"allowed_tools": ["remote.fetch"]},
                "is_active": True,
            },
        )

    assert resp.status_code == 403
    assert "grant.tool.invoke" in resp.json()["detail"]


def test_put_policy_assignment_override_allows_narrower_path_allowlist_without_grant_authority() -> None:
    service = _FakePolicyService()
    service.policy_assignments = [
        {
            "id": 11,
            "target_type": "persona",
            "target_id": "researcher",
            "owner_scope_type": "user",
            "owner_scope_id": 7,
            "profile_id": None,
            "inline_policy_document": {
                "capabilities": ["filesystem.read"],
                "path_scope_mode": "workspace_root",
                "path_scope_enforcement": "approval_required_when_unenforceable",
            },
            "approval_policy_id": None,
            "is_active": True,
            "has_override": False,
            "override_id": None,
            "override_active": False,
            "override_updated_at": None,
            "created_by": 7,
            "updated_by": 7,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    service.policy_overrides = {}
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE],
        ),
        service=service,
    )

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/mcp/hub/policy-assignments/11/override",
            json={
                "override_policy_document": {"path_allowlist_prefixes": ["src", "docs"]},
                "is_active": True,
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["override_policy_document"]["path_allowlist_prefixes"] == ["docs", "src"]
    assert service.policy_overrides[11]["broadens_access"] is False


def test_put_policy_assignment_override_requires_grant_authority_for_wider_path_allowlist() -> None:
    service = _FakePolicyService()
    service.policy_assignments = [
        {
            "id": 11,
            "target_type": "persona",
            "target_id": "researcher",
            "owner_scope_type": "user",
            "owner_scope_id": 7,
            "profile_id": None,
            "inline_policy_document": {
                "capabilities": ["filesystem.read"],
                "path_scope_mode": "workspace_root",
                "path_scope_enforcement": "approval_required_when_unenforceable",
                "path_allowlist_prefixes": ["src"],
            },
            "approval_policy_id": None,
            "is_active": True,
            "has_override": False,
            "override_id": None,
            "override_active": False,
            "override_updated_at": None,
            "created_by": 7,
            "updated_by": 7,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    service.policy_overrides = {}
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE],
        ),
        service=service,
    )

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/mcp/hub/policy-assignments/11/override",
            json={
                "override_policy_document": {"path_allowlist_prefixes": ["src", "docs"]},
                "is_active": True,
            },
        )

    assert resp.status_code == 403
    assert "grant.filesystem.read" in resp.json()["detail"]


def test_delete_policy_assignment_override_returns_ok() -> None:
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute", "grant.tool.invoke"],
        )
    )

    with TestClient(app) as client:
        resp = client.delete("/api/v1/mcp/hub/policy-assignments/11/override")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_list_tool_registry_returns_normalized_entries() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[],
        ),
        tool_registry=_FakeToolRegistryService(),
    )

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/hub/tool-registry")

    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload) == 2
    assert payload[0]["tool_name"] == "notes.search"
    assert payload[0]["risk_class"] == "low"
    assert payload[0]["path_argument_hints"] == ["path"]
    assert payload[1]["tool_name"] == "sandbox.run"
    assert payload[1]["path_argument_hints"] == []
    assert payload[1]["metadata_warnings"] == ["Derived from tool category"]


def test_list_tool_registry_modules_returns_grouped_summary() -> None:
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[],
        ),
        tool_registry=_FakeToolRegistryService(),
    )

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/hub/tool-registry/modules")

    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload) == 2
    assert payload[0]["module"] == "notes"
    assert payload[0]["tool_count"] == 1
    assert payload[1]["module"] == "sandbox"
    assert payload[1]["risk_summary"]["high"] == 1


def test_get_tool_registry_summary_returns_entries_and_modules_from_one_scan() -> None:
    registry = _FakeToolRegistryService()
    app = _build_app(
        _make_principal(
            roles=[],
            permissions=[],
        ),
        tool_registry=registry,
    )

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/hub/tool-registry/summary")

    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["entries"]) == 2
    assert len(payload["modules"]) == 2
    assert registry.list_entries_calls == 1


def test_list_permission_profiles_returns_visible_rows() -> None:
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute"],
        )
    )

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/hub/permission-profiles")

    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload) == 1
    assert payload[0]["name"] == "Process Exec"


def test_update_permission_profile_requires_grant_authority_for_new_capability() -> None:
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute"],
        )
    )

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/mcp/hub/permission-profiles/5",
            json={"policy_document": {"capabilities": ["network.external"]}},
        )

    assert resp.status_code == 403
    assert "grant.network.external" in resp.json()["detail"]


def test_update_permission_profile_returns_404_when_missing() -> None:
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute"],
        )
    )

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/mcp/hub/permission-profiles/999",
            json={"name": "Missing"},
        )

    assert resp.status_code == 404


def test_delete_permission_profile_returns_ok() -> None:
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute"],
        )
    )

    with TestClient(app) as client:
        resp = client.delete("/api/v1/mcp/hub/permission-profiles/5")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_list_policy_assignments_returns_visible_rows() -> None:
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute"],
        )
    )

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/hub/policy-assignments")

    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload) == 1
    assert payload[0]["target_id"] == "researcher"


def test_update_policy_assignment_returns_updated_payload() -> None:
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute", "grant.network.external"],
        )
    )

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/mcp/hub/policy-assignments/11",
            json={"inline_policy_document": {"capabilities": ["network.external"]}},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["inline_policy_document"]["capabilities"] == ["network.external"]


def test_add_policy_assignment_workspace_returns_structured_unresolvable_detail() -> None:
    service = _FakePolicyService()
    service.add_policy_assignment_workspace_error = _StructuredBadRequestError(
        {
            "code": "assignment_workspace_unresolvable",
            "message": "Workspace source cannot resolve every workspace for multi-root execution.",
            "unresolved_workspace_ids": ["workspace-missing"],
            "workspace_source_mode": "inline",
            "workspace_trust_source": "user_local",
        }
    )
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute"],
        ),
        service=service,
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/policy-assignments/11/workspaces",
            json={"workspace_id": "workspace-missing"},
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == {
        "code": "assignment_workspace_unresolvable",
        "message": "Workspace source cannot resolve every workspace for multi-root execution.",
        "unresolved_workspace_ids": ["workspace-missing"],
        "workspace_source_mode": "inline",
        "workspace_trust_source": "user_local",
    }


def test_delete_policy_assignment_returns_ok() -> None:
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute"],
        )
    )

    with TestClient(app) as client:
        resp = client.delete("/api/v1/mcp/hub/policy-assignments/11")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_mcp_hub_routes_apply_rate_limit_dependency() -> None:
    rate_limit_calls: list[str] = []
    app = _build_app(
        _make_principal(
            permissions=[SYSTEM_CONFIGURE, "grant.process.execute"],
        ),
        rate_limit_calls=rate_limit_calls,
    )

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/hub/permission-profiles")

    assert resp.status_code == 200
    assert rate_limit_calls == ["called"]
