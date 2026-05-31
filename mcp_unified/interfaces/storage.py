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


class ExternalRegistryStoreUnavailableError(RuntimeError):
    """Raised when an external registry store cannot serve requests."""


class ExternalServerAlreadyExistsError(RuntimeError):
    """Raised when an atomic external server create conflicts with an existing id."""

    def __init__(self, server_id: str) -> None:
        super().__init__(f"External server already exists: {server_id}")
        self.server_id = server_id


class ProfileStore(Protocol):
    """Store for named MCP tool and permission profiles.

    Implementations return caller-owned ``MCPProfile`` instances so callers may
    inspect or mutate returned models without changing persisted state.
    """

    async def get_profile(self, profile_id: str) -> MCPProfile | None:
        """Return a profile by id when present."""
        ...

    async def list_profiles(self) -> list[MCPProfile]:
        """Return all stored profiles."""
        ...

    async def upsert_profile(
        self,
        profile: MCPProfile,
    ) -> MCPProfile:
        """Create or replace a profile document."""
        ...

    async def create_profile(
        self,
        profile: MCPProfile,
    ) -> MCPProfile:
        """Create a profile and reject existing ids."""
        ...

    async def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile by id and report whether it existed."""
        ...


class GuardedProfileDeleteStore(Protocol):
    """Store capability for atomically deleting unassigned non-default profiles."""

    async def delete_profile_if_unassigned(
        self,
        profile_id: str,
        *,
        effective_default_profile_id: str | None,
    ) -> str:
        """Return deleted, not_found, is_default, or has_assignments."""
        raise NotImplementedError


class ProfileAssignmentStore(Protocol):
    """Store for principal, workspace, and default-profile assignments."""

    async def get_assignment(
        self,
        assignment_id: str,
    ) -> ProfileAssignment | None:
        """Return a profile assignment by id when present."""
        ...

    async def list_assignments(
        self,
        *,
        profile_id: str | None = None,
        principal_id: str | None = None,
        workspace_id: str | None = None,
    ) -> list[ProfileAssignment]:
        """Return profile assignments matching optional filters."""
        ...

    async def upsert_assignment(
        self,
        assignment: ProfileAssignment,
    ) -> ProfileAssignment:
        """Create or replace a profile assignment."""
        ...

    async def delete_assignment(self, assignment_id: str) -> bool:
        """Delete a profile assignment and report whether it existed."""
        ...


class ApprovalPolicyStore(Protocol):
    """Store for reusable approval policy documents and profile bindings."""

    async def get_policy(
        self,
        policy_id: str,
    ) -> ApprovalPolicyDocument | None:
        """Return an approval policy by id when present."""
        ...

    async def list_policies(
        self,
        *,
        profile_id: str | None = None,
    ) -> list[ApprovalPolicyDocument]:
        """Return approval policies matching optional filters."""
        ...

    async def upsert_policy(
        self,
        policy: ApprovalPolicyDocument,
    ) -> ApprovalPolicyDocument:
        """Create or replace an approval policy document."""
        ...

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete an approval policy and report whether it existed."""
        ...


class CredentialGrantStore(Protocol):
    """Store for credential broker grant metadata, never secret values."""

    async def get_grant(self, grant_id: str) -> CredentialGrant | None:
        """Return a credential grant by id when present."""
        ...

    async def list_grants(
        self,
        *,
        profile_id: str | None = None,
        external_server_id: str | None = None,
    ) -> list[CredentialGrant]:
        """Return credential grants matching optional filters."""
        ...

    async def upsert_grant(self, grant: CredentialGrant) -> CredentialGrant:
        """Create or replace credential grant metadata."""
        ...

    async def delete_grant(self, grant_id: str) -> bool:
        """Delete credential grant metadata and report whether it existed."""
        ...


class ExternalRegistryStore(Protocol):
    """Store for external MCP server registry entries.

    ``list_servers`` preserves the existing host runtime manager shape: no
    filters and dict-compatible status rows. New typed stores can additionally
    expose ``list_server_definitions`` for model-based persistence queries.
    """

    async def get_server(
        self,
        server_id: str,
    ) -> ExternalServerDefinition | None:
        """Return an external server definition by id when present."""
        ...

    async def list_servers(
        self,
    ) -> list[ExternalServerDefinition] | list[dict[str, Any]]:
        """Return external servers in the runtime manager-compatible shape."""
        ...

    async def list_server_definitions(
        self,
        *,
        enabled: bool | None = None,
    ) -> list[ExternalServerDefinition]:
        """Return typed external server definitions matching filters."""
        ...

    async def create_server(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition:
        """Create an external server definition and reject existing ids."""
        ...

    async def upsert_server(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition:
        """Create or replace an external server definition."""
        ...

    async def delete_server(self, server_id: str) -> bool:
        """Delete an external server and report whether it existed."""
        ...


class AuditStore(Protocol):
    """Append-only audit sink for MCP policy and tool events."""

    async def append_event(self, event: AuditEvent) -> AuditEvent:
        """Append an audit event and return the stored event."""
        ...

    async def query_events(
        self,
        *,
        actor_id: str | None = None,
        profile_id: str | None = None,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEvent]:
        """Return audit events matching optional filters."""
        ...
