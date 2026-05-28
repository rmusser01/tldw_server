"""SQLite-backed standalone MCP storage primitives."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

from mcp_unified.profiles.models import MCPProfile
from mcp_unified.storage.models import (
    ApprovalPolicyDocument,
    AuditEvent,
    CredentialGrant,
    ExternalServerDefinition,
    ProfileAssignment,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class SQLiteMCPStore:
    """Package-local SQLite implementation for MCP standalone stores."""

    SCHEMA_VERSION = 1
    _DELETE_BY_ID_SQL: ClassVar[dict[str, str]] = {
        "mcp_profiles": "DELETE FROM mcp_profiles WHERE id = ?",
        "mcp_profile_assignments": (
            "DELETE FROM mcp_profile_assignments WHERE id = ?"
        ),
        "mcp_approval_policies": "DELETE FROM mcp_approval_policies WHERE id = ?",
        "mcp_credential_grants": "DELETE FROM mcp_credential_grants WHERE id = ?",
        "mcp_external_servers": "DELETE FROM mcp_external_servers WHERE id = ?",
    }
    _SELECT_BY_ID_SQL: ClassVar[dict[str, str]] = {
        "mcp_profiles": "SELECT payload FROM mcp_profiles WHERE id = ?",
        "mcp_profile_assignments": (
            "SELECT payload FROM mcp_profile_assignments WHERE id = ?"
        ),
        "mcp_approval_policies": (
            "SELECT payload FROM mcp_approval_policies WHERE id = ?"
        ),
        "mcp_credential_grants": (
            "SELECT payload FROM mcp_credential_grants WHERE id = ?"
        ),
        "mcp_external_servers": (
            "SELECT payload FROM mcp_external_servers WHERE id = ?"
        ),
    }
    _SELECT_PAYLOAD_SQL: ClassVar[dict[str, str]] = {
        "mcp_profile_assignments": "SELECT payload FROM mcp_profile_assignments",
        "mcp_approval_policies": "SELECT payload FROM mcp_approval_policies",
        "mcp_credential_grants": "SELECT payload FROM mcp_credential_grants",
        "mcp_external_servers": "SELECT payload FROM mcp_external_servers",
        "mcp_audit_events": "SELECT payload FROM mcp_audit_events",
    }
    _FILTERABLE_COLUMNS: ClassVar[dict[str, frozenset[str]]] = {
        "mcp_profile_assignments": frozenset(
            {"profile_id", "principal_id", "workspace_id"}
        ),
        "mcp_approval_policies": frozenset({"profile_id"}),
        "mcp_credential_grants": frozenset(
            {"profile_id", "external_server_id"}
        ),
        "mcp_external_servers": frozenset({"enabled"}),
        "mcp_audit_events": frozenset({"actor_id", "profile_id", "event_type"}),
    }
    _ORDER_BY_SQL: ClassVar[dict[str, str]] = {
        "id ASC": "id ASC",
        "created_at DESC, id DESC": "created_at DESC, id DESC",
    }

    def __init__(self, path: str | Path) -> None:
        if str(path) == ":memory:":
            self.path = ":memory:"
        else:
            db_path = Path(path).expanduser()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(db_path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._initialize_schema()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    async def aclose(self) -> None:
        """Async-friendly close helper for callers managing stores generically."""
        self.close()

    async def get_profile(self, profile_id: str) -> MCPProfile | None:
        """Return a copy-isolated profile by id."""
        return self._get_model("mcp_profiles", profile_id, MCPProfile)

    async def list_profiles(self) -> list[MCPProfile]:
        """Return all profiles sorted by id."""
        rows = self._conn.execute(
            "SELECT payload FROM mcp_profiles ORDER BY id",
        ).fetchall()
        return [self._load_model(row["payload"], MCPProfile) for row in rows]

    async def upsert_profile(
        self,
        profile: MCPProfile | Mapping[str, Any],
    ) -> MCPProfile:
        """Store a profile document and return the persisted model."""
        validated = self._validate_profile(profile)
        payload = self._dump_model(validated)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO mcp_profiles(id, enabled, updated_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (
                    validated.id,
                    int(validated.enabled),
                    validated.updated_at.isoformat(),
                    payload,
                ),
            )
        return self._load_model(payload, MCPProfile)

    async def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile by id and return whether it existed."""
        return self._delete_by_id("mcp_profiles", profile_id)

    async def get_assignment(self, assignment_id: str) -> ProfileAssignment | None:
        """Return a profile assignment by id."""
        return self._get_model(
            "mcp_profile_assignments",
            assignment_id,
            ProfileAssignment,
        )

    async def list_assignments(
        self,
        *,
        profile_id: str | None = None,
        principal_id: str | None = None,
        workspace_id: str | None = None,
    ) -> list[ProfileAssignment]:
        """Return profile assignments matching optional filters."""
        rows = self._select_filtered_payloads(
            "mcp_profile_assignments",
            {
                "profile_id": profile_id,
                "principal_id": principal_id,
                "workspace_id": workspace_id,
            },
        )
        return [self._load_model(row["payload"], ProfileAssignment) for row in rows]

    async def upsert_assignment(
        self,
        assignment: ProfileAssignment,
    ) -> ProfileAssignment:
        """Store a profile assignment and return the persisted model."""
        payload = self._dump_model(assignment)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO mcp_profile_assignments(
                    id, profile_id, principal_id, workspace_id, is_default,
                    enabled, updated_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    profile_id = excluded.profile_id,
                    principal_id = excluded.principal_id,
                    workspace_id = excluded.workspace_id,
                    is_default = excluded.is_default,
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (
                    assignment.id,
                    assignment.profile_id,
                    assignment.principal_id,
                    assignment.workspace_id,
                    int(assignment.is_default),
                    int(assignment.enabled),
                    assignment.updated_at.isoformat(),
                    payload,
                ),
            )
        return self._load_model(payload, ProfileAssignment)

    async def delete_assignment(self, assignment_id: str) -> bool:
        """Delete a profile assignment by id and return whether it existed."""
        return self._delete_by_id("mcp_profile_assignments", assignment_id)

    async def get_policy(self, policy_id: str) -> ApprovalPolicyDocument | None:
        """Return an approval policy by id."""
        return self._get_model("mcp_approval_policies", policy_id, ApprovalPolicyDocument)

    async def list_policies(
        self,
        *,
        profile_id: str | None = None,
    ) -> list[ApprovalPolicyDocument]:
        """Return approval policies matching optional filters."""
        rows = self._select_filtered_payloads(
            "mcp_approval_policies",
            {"profile_id": profile_id},
        )
        return [self._load_model(row["payload"], ApprovalPolicyDocument) for row in rows]

    async def upsert_policy(
        self,
        policy: ApprovalPolicyDocument,
    ) -> ApprovalPolicyDocument:
        """Store an approval policy and return the persisted model."""
        payload = self._dump_model(policy)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO mcp_approval_policies(
                    id, profile_id, enabled, updated_at, payload
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    profile_id = excluded.profile_id,
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (
                    policy.id,
                    policy.profile_id,
                    int(policy.enabled),
                    policy.updated_at.isoformat(),
                    payload,
                ),
            )
        return self._load_model(payload, ApprovalPolicyDocument)

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete an approval policy by id and return whether it existed."""
        return self._delete_by_id("mcp_approval_policies", policy_id)

    async def get_grant(self, grant_id: str) -> CredentialGrant | None:
        """Return a credential grant by id."""
        return self._get_model("mcp_credential_grants", grant_id, CredentialGrant)

    async def list_grants(
        self,
        *,
        profile_id: str | None = None,
        external_server_id: str | None = None,
    ) -> list[CredentialGrant]:
        """Return credential grants matching optional filters."""
        rows = self._select_filtered_payloads(
            "mcp_credential_grants",
            {
                "profile_id": profile_id,
                "external_server_id": external_server_id,
            },
        )
        return [self._load_model(row["payload"], CredentialGrant) for row in rows]

    async def upsert_grant(self, grant: CredentialGrant) -> CredentialGrant:
        """Store a credential grant and return the persisted model."""
        payload = self._dump_model(grant)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO mcp_credential_grants(
                    id, profile_id, external_server_id, enabled, updated_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    profile_id = excluded.profile_id,
                    external_server_id = excluded.external_server_id,
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (
                    grant.id,
                    grant.profile_id,
                    grant.external_server_id,
                    int(grant.enabled),
                    grant.updated_at.isoformat(),
                    payload,
                ),
            )
        return self._load_model(payload, CredentialGrant)

    async def delete_grant(self, grant_id: str) -> bool:
        """Delete a credential grant by id and return whether it existed."""
        return self._delete_by_id("mcp_credential_grants", grant_id)

    async def get_server(self, server_id: str) -> ExternalServerDefinition | None:
        """Return an external server definition by id."""
        return self._get_model("mcp_external_servers", server_id, ExternalServerDefinition)

    async def list_servers(self) -> list[ExternalServerDefinition]:
        """Return all external server definitions sorted by id."""
        return await self.list_server_definitions()

    async def list_server_definitions(
        self,
        *,
        enabled: bool | None = None,
    ) -> list[ExternalServerDefinition]:
        """Return external server definitions matching optional enabled state."""
        rows = self._select_filtered_payloads(
            "mcp_external_servers",
            {"enabled": None if enabled is None else int(enabled)},
        )
        return [self._load_model(row["payload"], ExternalServerDefinition) for row in rows]

    async def upsert_server(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition:
        """Store an external server definition and return the persisted model."""
        payload = self._dump_model(server)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO mcp_external_servers(
                    id, enabled, transport, updated_at, payload
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    enabled = excluded.enabled,
                    transport = excluded.transport,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (
                    server.id,
                    int(server.enabled),
                    server.transport,
                    server.updated_at.isoformat(),
                    payload,
                ),
            )
        return self._load_model(payload, ExternalServerDefinition)

    async def delete_server(self, server_id: str) -> bool:
        """Delete an external server definition by id and return whether it existed."""
        return self._delete_by_id("mcp_external_servers", server_id)

    async def append_event(self, event: AuditEvent) -> AuditEvent:
        """Append an audit event and return the persisted event."""
        payload = self._dump_model(event)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO mcp_audit_events(
                    id, actor_id, profile_id, event_type, created_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.actor_id,
                    event.profile_id,
                    event.event_type,
                    event.created_at.isoformat(),
                    payload,
                ),
            )
        return self._load_model(payload, AuditEvent)

    async def query_events(
        self,
        *,
        actor_id: str | None = None,
        profile_id: str | None = None,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEvent]:
        """Return audit events matching optional filters, newest first."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        rows = self._select_filtered_payloads(
            "mcp_audit_events",
            {
                "actor_id": actor_id,
                "profile_id": profile_id,
                "event_type": event_type,
            },
            limit=limit,
            order_by="created_at DESC, id DESC",
        )
        return [self._load_model(row["payload"], AuditEvent) for row in rows]

    def _initialize_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mcp_storage_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._ensure_compatible_schema_version()
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mcp_profiles (
                    id TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mcp_profile_assignments (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    principal_id TEXT,
                    workspace_id TEXT,
                    is_default INTEGER NOT NULL,
                    enabled INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mcp_approval_policies (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT,
                    enabled INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mcp_credential_grants (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    external_server_id TEXT,
                    enabled INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mcp_external_servers (
                    id TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL,
                    transport TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mcp_audit_events (
                    id TEXT PRIMARY KEY,
                    actor_id TEXT,
                    profile_id TEXT,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mcp_assignments_profile ON mcp_profile_assignments(profile_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mcp_assignments_principal ON mcp_profile_assignments(principal_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mcp_assignments_workspace ON mcp_profile_assignments(workspace_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mcp_policies_profile ON mcp_approval_policies(profile_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mcp_grants_profile ON mcp_credential_grants(profile_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mcp_grants_external_server ON mcp_credential_grants(external_server_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mcp_external_enabled ON mcp_external_servers(enabled)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mcp_audit_actor ON mcp_audit_events(actor_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mcp_audit_profile ON mcp_audit_events(profile_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mcp_audit_event_type ON mcp_audit_events(event_type)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mcp_audit_created_at ON mcp_audit_events(created_at)"
            )

    def _ensure_compatible_schema_version(self) -> None:
        current = self._conn.execute(
            "SELECT value FROM mcp_storage_meta WHERE key = ?",
            ("schema_version",),
        ).fetchone()
        if current is not None and int(current["value"]) > self.SCHEMA_VERSION:
            raise RuntimeError(
                f"SQLite MCP store schema {current['value']} is newer than supported "
                f"schema {self.SCHEMA_VERSION}"
            )
        self._conn.execute(
            """
            INSERT INTO mcp_storage_meta(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            ("schema_version", str(self.SCHEMA_VERSION)),
        )

    def _delete_by_id(self, table: str, item_id: str) -> bool:
        with self._conn:
            cursor = self._conn.execute(
                self._DELETE_BY_ID_SQL[table],
                (item_id,),
            )
        return cursor.rowcount > 0

    def _get_model(
        self,
        table: str,
        item_id: str,
        model_cls: type[ModelT],
    ) -> ModelT | None:
        row = self._conn.execute(
            self._SELECT_BY_ID_SQL[table],
            (item_id,),
        ).fetchone()
        if row is None:
            return None
        return self._load_model(row["payload"], model_cls)

    def _select_filtered_payloads(
        self,
        table: str,
        filters: dict[str, Any],
        *,
        limit: int | None = None,
        order_by: str = "id ASC",
    ) -> list[sqlite3.Row]:
        self._validate_filter_request(table, filters, order_by)
        clauses: list[str] = []
        values: list[Any] = []
        for column, value in filters.items():
            if value is None:
                continue
            clauses.append(f"{column} = ?")
            values.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query_parts = [self._SELECT_PAYLOAD_SQL[table]]
        if where:
            query_parts.append(where)
        query_parts.extend(["ORDER BY", self._ORDER_BY_SQL[order_by]])
        if limit is not None:
            query_parts.append("LIMIT ?")
        if limit is not None:
            values.append(limit)
        return self._conn.execute(" ".join(query_parts), values).fetchall()

    def _validate_filter_request(
        self,
        table: str,
        filters: dict[str, Any],
        order_by: str,
    ) -> None:
        allowed_columns = self._FILTERABLE_COLUMNS.get(table)
        if allowed_columns is None:
            raise ValueError(f"Unsupported filter table: {table}")
        unknown_columns = set(filters) - allowed_columns
        if unknown_columns:
            raise ValueError(
                f"Unsupported filter columns for {table}: {sorted(unknown_columns)}"
            )
        if order_by not in self._ORDER_BY_SQL:
            raise ValueError(f"Unsupported order by clause: {order_by}")

    @staticmethod
    def _dump_model(model: BaseModel) -> str:
        return json.dumps(
            model.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _load_model(payload: str, model_cls: type[ModelT]) -> ModelT:
        return model_cls.model_validate_json(payload)

    @staticmethod
    def _validate_profile(profile: MCPProfile | Mapping[str, Any]) -> MCPProfile:
        if isinstance(profile, MCPProfile):
            return profile.model_copy(deep=True)
        return MCPProfile.model_validate(profile)


__all__ = ["SQLiteMCPStore"]
