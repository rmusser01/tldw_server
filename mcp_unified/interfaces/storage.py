"""Storage protocols for standalone MCP profile and registry stores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from mcp_unified.profiles.models import MCPProfile
    from mcp_unified.storage.models import (
        ApprovalPolicyDocument,
        AuditEvent,
        CredentialGrant,
        ExternalServerDefinition,
        ProfileAssignment,
    )


class ProfileStore(Protocol):
    """Store for named MCP tool and permission profiles.

    Implementations return caller-owned ``MCPProfile`` instances so callers may
    inspect or mutate returned models without changing persisted state.
    """

    async def get_profile(self, profile_id: str) -> MCPProfile | None: ...

    async def list_profiles(self) -> list[MCPProfile]: ...

    async def upsert_profile(
        self,
        profile: MCPProfile,
    ) -> MCPProfile: ...

    async def delete_profile(self, profile_id: str) -> bool: ...


class ProfileAssignmentStore(Protocol):
    """Store for principal, workspace, and default-profile assignments."""

    async def get_assignment(self, assignment_id: str) -> ProfileAssignment | None: ...

    async def list_assignments(
        self,
        *,
        profile_id: str | None = None,
        principal_id: str | None = None,
        workspace_id: str | None = None,
    ) -> list[ProfileAssignment]: ...

    async def upsert_assignment(
        self,
        assignment: ProfileAssignment,
    ) -> ProfileAssignment: ...

    async def delete_assignment(self, assignment_id: str) -> bool: ...


class ApprovalPolicyStore(Protocol):
    """Store for reusable approval policy documents and profile bindings."""

    async def get_policy(self, policy_id: str) -> ApprovalPolicyDocument | None: ...

    async def list_policies(
        self,
        *,
        profile_id: str | None = None,
    ) -> list[ApprovalPolicyDocument]: ...

    async def upsert_policy(
        self,
        policy: ApprovalPolicyDocument,
    ) -> ApprovalPolicyDocument: ...

    async def delete_policy(self, policy_id: str) -> bool: ...


class CredentialGrantStore(Protocol):
    """Store for credential broker grant metadata, never secret values."""

    async def get_grant(self, grant_id: str) -> CredentialGrant | None: ...

    async def list_grants(
        self,
        *,
        profile_id: str | None = None,
        external_server_id: str | None = None,
    ) -> list[CredentialGrant]: ...

    async def upsert_grant(self, grant: CredentialGrant) -> CredentialGrant: ...

    async def delete_grant(self, grant_id: str) -> bool: ...


class ExternalRegistryStore(Protocol):
    """Store for external MCP server registry entries.

    ``list_servers`` preserves the existing host runtime manager shape: no
    filters and dict-compatible status rows. New typed stores can additionally
    expose ``list_server_definitions`` for model-based persistence queries.
    """

    async def get_server(self, server_id: str) -> ExternalServerDefinition | None: ...

    async def list_servers(self) -> list[ExternalServerDefinition] | list[dict[str, Any]]: ...

    async def list_server_definitions(
        self,
        *,
        enabled: bool | None = None,
    ) -> list[ExternalServerDefinition]: ...

    async def upsert_server(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition: ...

    async def delete_server(self, server_id: str) -> bool: ...


class AuditStore(Protocol):
    """Append-only audit sink for MCP policy and tool events."""

    async def append_event(self, event: AuditEvent) -> AuditEvent: ...

    async def query_events(
        self,
        *,
        actor_id: str | None = None,
        profile_id: str | None = None,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEvent]: ...
