"""SQLite-backed standalone MCP storage primitives.

The store uses SQLAlchemy Core as its database boundary so this standalone
package does not issue raw sqlite3 calls or import host DB_Management helpers.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from datetime import timezone
from pathlib import Path
from typing import Any, ClassVar, ParamSpec, TypeVar, cast

from pydantic import BaseModel
from sqlalchemy import (
    URL,
    Column,
    Engine,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    event,
    select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.pool import StaticPool

from mcp_unified.interfaces.storage import (
    ExternalRegistryStoreUnavailableError,
    ExternalServerAlreadyExistsError,
)
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.profiles.store import ProfileAlreadyExistsError
from mcp_unified.storage.models import (
    ApprovalPolicyDocument,
    AuditEvent,
    CredentialGrant,
    ExternalServerDefinition,
    ProfileAssignment,
)

ModelT = TypeVar("ModelT", bound=BaseModel)
ReturnT = TypeVar("ReturnT")
ParamsT = ParamSpec("ParamsT")


class SQLiteMCPStore:
    """SQLAlchemy-backed SQLite implementation for MCP standalone stores."""

    SCHEMA_VERSION = 1
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
    _ORDER_BY_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"id ASC", "created_at DESC, id DESC"}
    )

    def __init__(self, path: str | Path) -> None:
        if str(path) == ":memory:":
            self.path = ":memory:"
        else:
            db_path = Path(path).expanduser()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(db_path)

        self._metadata = MetaData()
        self._tables = self._build_tables(self._metadata)
        self._engine = self._create_engine()
        event.listen(self._engine, "connect", self._enable_foreign_keys)
        self._initialize_schema()

    def close(self) -> None:
        """Dispose the underlying database engine."""
        self._close_sync()

    async def aclose(self) -> None:
        """Async-friendly close helper for callers managing stores generically."""
        await self._run_db(self._close_sync)

    async def get_profile(self, profile_id: str) -> MCPProfile | None:
        """Return a copy-isolated profile by id."""
        return await self._run_db(self._get_profile_sync, profile_id)

    async def list_profiles(self) -> list[MCPProfile]:
        """Return all profiles sorted by id."""
        return await self._run_db(self._list_profiles_sync)

    async def upsert_profile(
        self,
        profile: MCPProfile | Mapping[str, Any],
    ) -> MCPProfile:
        """Store a profile document and return the persisted model."""
        return await self._run_db(self._upsert_profile_sync, profile)

    async def create_profile(
        self,
        profile: MCPProfile | Mapping[str, Any],
    ) -> MCPProfile:
        """Create a profile only when its id is absent."""
        return await self._run_db(self._create_profile_sync, profile)

    async def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile by id and return whether it existed."""
        return await self._run_db(self._delete_profile_sync, profile_id)

    async def delete_profile_if_unassigned(
        self,
        profile_id: str,
        *,
        effective_default_profile_id: str | None,
    ) -> str:
        """Atomically delete a profile only when it is not default or assigned."""
        return await self._run_db(
            self._delete_profile_if_unassigned_sync,
            profile_id,
            effective_default_profile_id=effective_default_profile_id,
        )

    async def get_assignment(self, assignment_id: str) -> ProfileAssignment | None:
        """Return a profile assignment by id."""
        return await self._run_db(self._get_assignment_sync, assignment_id)

    async def list_assignments(
        self,
        *,
        profile_id: str | None = None,
        principal_id: str | None = None,
        workspace_id: str | None = None,
    ) -> list[ProfileAssignment]:
        """Return profile assignments matching optional filters."""
        return await self._run_db(
            self._list_assignments_sync,
            profile_id=profile_id,
            principal_id=principal_id,
            workspace_id=workspace_id,
        )

    async def upsert_assignment(
        self,
        assignment: ProfileAssignment,
    ) -> ProfileAssignment:
        """Store a profile assignment and return the persisted model."""
        return await self._run_db(self._upsert_assignment_sync, assignment)

    async def delete_assignment(self, assignment_id: str) -> bool:
        """Delete a profile assignment by id and return whether it existed."""
        return await self._run_db(self._delete_assignment_sync, assignment_id)

    async def get_policy(self, policy_id: str) -> ApprovalPolicyDocument | None:
        """Return an approval policy by id."""
        return await self._run_db(self._get_policy_sync, policy_id)

    async def list_policies(
        self,
        *,
        profile_id: str | None = None,
    ) -> list[ApprovalPolicyDocument]:
        """Return approval policies matching optional filters."""
        return await self._run_db(self._list_policies_sync, profile_id=profile_id)

    async def upsert_policy(
        self,
        policy: ApprovalPolicyDocument,
    ) -> ApprovalPolicyDocument:
        """Store an approval policy and return the persisted model."""
        return await self._run_db(self._upsert_policy_sync, policy)

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete an approval policy by id and return whether it existed."""
        return await self._run_db(self._delete_policy_sync, policy_id)

    async def get_grant(self, grant_id: str) -> CredentialGrant | None:
        """Return a credential grant by id."""
        return await self._run_db(self._get_grant_sync, grant_id)

    async def list_grants(
        self,
        *,
        profile_id: str | None = None,
        external_server_id: str | None = None,
    ) -> list[CredentialGrant]:
        """Return credential grants matching optional filters."""
        return await self._run_db(
            self._list_grants_sync,
            profile_id=profile_id,
            external_server_id=external_server_id,
        )

    async def upsert_grant(self, grant: CredentialGrant) -> CredentialGrant:
        """Store a credential grant and return the persisted model."""
        return await self._run_db(self._upsert_grant_sync, grant)

    async def delete_grant(self, grant_id: str) -> bool:
        """Delete a credential grant by id and return whether it existed."""
        return await self._run_db(self._delete_grant_sync, grant_id)

    async def get_server(self, server_id: str) -> ExternalServerDefinition | None:
        """Return an external server definition by id."""
        return await self._run_external_registry_db(self._get_server_sync, server_id)

    async def list_servers(self) -> list[ExternalServerDefinition]:
        """Return all external server definitions sorted by id."""
        return await self.list_server_definitions()

    async def list_server_definitions(
        self,
        *,
        enabled: bool | None = None,
    ) -> list[ExternalServerDefinition]:
        """Return external server definitions matching optional enabled state."""
        return await self._run_external_registry_db(
            self._list_server_definitions_sync,
            enabled=enabled,
        )

    async def upsert_server(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition:
        """Store an external server definition and return the persisted model."""
        return await self._run_external_registry_db(self._upsert_server_sync, server)

    async def update_server(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition | None:
        """Update an existing external server definition without creating it."""
        return await self._run_external_registry_db(self._update_server_sync, server)

    async def create_server(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition:
        """Create an external server definition only when its id is absent."""
        return await self._run_external_registry_db(self._create_server_sync, server)

    async def delete_server(self, server_id: str) -> bool:
        """Delete an external server definition by id and return whether it existed."""
        return await self._run_external_registry_db(self._delete_server_sync, server_id)

    async def append_event(self, event: AuditEvent) -> AuditEvent:
        """Append an audit event and return the persisted event."""
        return await self._run_db(self._append_event_sync, event)

    async def query_events(
        self,
        *,
        actor_id: str | None = None,
        profile_id: str | None = None,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEvent]:
        """Return audit events matching optional filters, newest first."""
        return await self._run_db(
            self._query_events_sync,
            actor_id=actor_id,
            profile_id=profile_id,
            event_type=event_type,
            limit=limit,
        )

    async def _run_db(
        self,
        operation: Callable[ParamsT, ReturnT],
        *args: ParamsT.args,
        **kwargs: ParamsT.kwargs,
    ) -> ReturnT:
        return await asyncio.to_thread(operation, *args, **kwargs)

    async def _run_external_registry_db(
        self,
        operation: Callable[ParamsT, ReturnT],
        *args: ParamsT.args,
        **kwargs: ParamsT.kwargs,
    ) -> ReturnT:
        """Run external-registry DB work and normalize store outage errors."""

        try:
            return await self._run_db(operation, *args, **kwargs)
        except ExternalServerAlreadyExistsError:
            raise
        except (OSError, SQLAlchemyError) as exc:
            raise ExternalRegistryStoreUnavailableError(
                "External registry store unavailable"
            ) from exc

    def _close_sync(self) -> None:
        self._engine.dispose()

    def _get_profile_sync(self, profile_id: str) -> MCPProfile | None:
        return self._get_model("mcp_profiles", profile_id, MCPProfile)

    def _list_profiles_sync(self) -> list[MCPProfile]:
        rows = self._select_payloads("mcp_profiles", order_by="id ASC")
        return [self._load_model(row["payload"], MCPProfile) for row in rows]

    def _upsert_profile_sync(
        self,
        profile: MCPProfile | Mapping[str, Any],
    ) -> MCPProfile:
        validated = self._validate_profile(profile)
        payload = self._dump_model(validated)
        self._upsert_row(
            "mcp_profiles",
            {
                "id": validated.id,
                "enabled": int(validated.enabled),
                "updated_at": validated.updated_at.isoformat(),
                "payload": payload,
            },
            update_columns=("enabled", "updated_at", "payload"),
        )
        return self._load_model(payload, MCPProfile)

    def _create_profile_sync(
        self,
        profile: MCPProfile | Mapping[str, Any],
    ) -> MCPProfile:
        validated = self._validate_profile(profile)
        payload = self._dump_model(validated)
        profiles_table = self._table("mcp_profiles")
        statement = sqlite_insert(profiles_table).values(
            id=validated.id,
            enabled=int(validated.enabled),
            updated_at=validated.updated_at.isoformat(),
            payload=payload,
        )
        try:
            with self._engine.begin() as connection:
                result = connection.execute(
                    statement.on_conflict_do_nothing(
                        index_elements=[profiles_table.c.id],
                    )
                )
        except IntegrityError as exc:
            raise ProfileAlreadyExistsError(validated.id) from exc
        if not result.rowcount:
            raise ProfileAlreadyExistsError(validated.id)
        return self._load_model(payload, MCPProfile)

    def _delete_profile_sync(self, profile_id: str) -> bool:
        return self._delete_by_id("mcp_profiles", profile_id)

    def _delete_profile_if_unassigned_sync(
        self,
        profile_id: str,
        *,
        effective_default_profile_id: str | None,
    ) -> str:
        if profile_id == effective_default_profile_id:
            return "is_default"

        profiles_table = self._table("mcp_profiles")
        assignments_table = self._table("mcp_profile_assignments")
        with self._engine.begin() as connection:
            result = connection.execute(
                delete(profiles_table).where(
                    profiles_table.c.id == profile_id,
                    ~select(assignments_table.c.id)
                    .where(assignments_table.c.profile_id == profile_id)
                    .exists(),
                )
            )
            if result.rowcount and result.rowcount > 0:
                return "deleted"

            profile_row = connection.execute(
                select(profiles_table.c.id).where(profiles_table.c.id == profile_id)
            ).mappings().first()
            if profile_row is None:
                return "not_found"

            assignment_row = connection.execute(
                select(assignments_table.c.id)
                .where(assignments_table.c.profile_id == profile_id)
                .limit(1)
            ).mappings().first()
            if assignment_row is not None:
                return "has_assignments"
        return "not_found"

    def _get_assignment_sync(self, assignment_id: str) -> ProfileAssignment | None:
        return self._get_model(
            "mcp_profile_assignments",
            assignment_id,
            ProfileAssignment,
        )

    def _list_assignments_sync(
        self,
        *,
        profile_id: str | None = None,
        principal_id: str | None = None,
        workspace_id: str | None = None,
    ) -> list[ProfileAssignment]:
        rows = self._select_payloads(
            "mcp_profile_assignments",
            filters={
                "profile_id": profile_id,
                "principal_id": principal_id,
                "workspace_id": workspace_id,
            },
        )
        return [self._load_model(row["payload"], ProfileAssignment) for row in rows]

    def _upsert_assignment_sync(
        self,
        assignment: ProfileAssignment,
    ) -> ProfileAssignment:
        payload = self._dump_model(assignment)
        self._upsert_row(
            "mcp_profile_assignments",
            {
                "id": assignment.id,
                "profile_id": assignment.profile_id,
                "principal_id": assignment.principal_id,
                "workspace_id": assignment.workspace_id,
                "is_default": int(assignment.is_default),
                "enabled": int(assignment.enabled),
                "updated_at": assignment.updated_at.isoformat(),
                "payload": payload,
            },
            update_columns=(
                "profile_id",
                "principal_id",
                "workspace_id",
                "is_default",
                "enabled",
                "updated_at",
                "payload",
            ),
        )
        return self._load_model(payload, ProfileAssignment)

    def _delete_assignment_sync(self, assignment_id: str) -> bool:
        return self._delete_by_id("mcp_profile_assignments", assignment_id)

    def _get_policy_sync(self, policy_id: str) -> ApprovalPolicyDocument | None:
        return self._get_model("mcp_approval_policies", policy_id, ApprovalPolicyDocument)

    def _list_policies_sync(
        self,
        *,
        profile_id: str | None = None,
    ) -> list[ApprovalPolicyDocument]:
        rows = self._select_payloads(
            "mcp_approval_policies",
            filters={"profile_id": profile_id},
        )
        return [self._load_model(row["payload"], ApprovalPolicyDocument) for row in rows]

    def _upsert_policy_sync(
        self,
        policy: ApprovalPolicyDocument,
    ) -> ApprovalPolicyDocument:
        payload = self._dump_model(policy)
        self._upsert_row(
            "mcp_approval_policies",
            {
                "id": policy.id,
                "profile_id": policy.profile_id,
                "enabled": int(policy.enabled),
                "updated_at": policy.updated_at.isoformat(),
                "payload": payload,
            },
            update_columns=("profile_id", "enabled", "updated_at", "payload"),
        )
        return self._load_model(payload, ApprovalPolicyDocument)

    def _delete_policy_sync(self, policy_id: str) -> bool:
        return self._delete_by_id("mcp_approval_policies", policy_id)

    def _get_grant_sync(self, grant_id: str) -> CredentialGrant | None:
        return self._get_model("mcp_credential_grants", grant_id, CredentialGrant)

    def _list_grants_sync(
        self,
        *,
        profile_id: str | None = None,
        external_server_id: str | None = None,
    ) -> list[CredentialGrant]:
        rows = self._select_payloads(
            "mcp_credential_grants",
            filters={
                "profile_id": profile_id,
                "external_server_id": external_server_id,
            },
        )
        return [self._load_model(row["payload"], CredentialGrant) for row in rows]

    def _upsert_grant_sync(self, grant: CredentialGrant) -> CredentialGrant:
        payload = self._dump_model(grant)
        self._upsert_row(
            "mcp_credential_grants",
            {
                "id": grant.id,
                "profile_id": grant.profile_id,
                "external_server_id": grant.external_server_id,
                "enabled": int(grant.enabled),
                "updated_at": grant.updated_at.isoformat(),
                "payload": payload,
            },
            update_columns=(
                "profile_id",
                "external_server_id",
                "enabled",
                "updated_at",
                "payload",
            ),
        )
        return self._load_model(payload, CredentialGrant)

    def _delete_grant_sync(self, grant_id: str) -> bool:
        return self._delete_by_id("mcp_credential_grants", grant_id)

    def _get_server_sync(self, server_id: str) -> ExternalServerDefinition | None:
        return self._get_model("mcp_external_servers", server_id, ExternalServerDefinition)

    def _list_server_definitions_sync(
        self,
        *,
        enabled: bool | None = None,
    ) -> list[ExternalServerDefinition]:
        rows = self._select_payloads(
            "mcp_external_servers",
            filters={"enabled": None if enabled is None else int(enabled)},
        )
        return [self._load_model(row["payload"], ExternalServerDefinition) for row in rows]

    def _upsert_server_sync(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition:
        payload = self._dump_model(server)
        self._upsert_row(
            "mcp_external_servers",
            {
                "id": server.id,
                "enabled": int(server.enabled),
                "transport": server.transport,
                "updated_at": server.updated_at.isoformat(),
                "payload": payload,
            },
            update_columns=("enabled", "transport", "updated_at", "payload"),
        )
        return self._load_model(payload, ExternalServerDefinition)

    def _update_server_sync(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition | None:
        payload = self._dump_model(server)
        table = self._table("mcp_external_servers")
        statement = (
            table.update()
            .where(table.c.id == server.id)
            .values(
                enabled=int(server.enabled),
                transport=server.transport,
                updated_at=server.updated_at.isoformat(),
                payload=payload,
            )
        )
        with self._engine.begin() as connection:
            result = connection.execute(statement)
        if not result.rowcount:
            return None
        return self._load_model(payload, ExternalServerDefinition)

    def _create_server_sync(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition:
        payload = self._dump_model(server)
        table = self._table("mcp_external_servers")
        statement = sqlite_insert(table).values(
            id=server.id,
            enabled=int(server.enabled),
            transport=server.transport,
            updated_at=server.updated_at.isoformat(),
            payload=payload,
        )
        with self._engine.begin() as connection:
            result = connection.execute(
                statement.on_conflict_do_nothing(index_elements=[table.c.id])
            )
        if not result.rowcount:
            raise ExternalServerAlreadyExistsError(server.id)
        return self._load_model(payload, ExternalServerDefinition)

    def _delete_server_sync(self, server_id: str) -> bool:
        return self._delete_by_id("mcp_external_servers", server_id)

    def _append_event_sync(self, event: AuditEvent) -> AuditEvent:
        normalized = self._normalize_audit_event(event)
        payload = self._dump_model(normalized)
        self._insert_row(
            "mcp_audit_events",
            {
                "id": normalized.id,
                "actor_id": normalized.actor_id,
                "profile_id": normalized.profile_id,
                "event_type": normalized.event_type,
                "created_at": normalized.created_at.isoformat(),
                "payload": payload,
            },
        )
        return self._load_model(payload, AuditEvent)

    def _query_events_sync(
        self,
        *,
        actor_id: str | None = None,
        profile_id: str | None = None,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEvent]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        rows = self._select_payloads(
            "mcp_audit_events",
            filters={
                "actor_id": actor_id,
                "profile_id": profile_id,
                "event_type": event_type,
            },
            limit=limit,
            order_by="created_at DESC, id DESC",
        )
        return [self._load_model(row["payload"], AuditEvent) for row in rows]

    def _create_engine(self) -> Engine:
        engine_kwargs: dict[str, Any] = {
            "connect_args": {
                "timeout": 30.0,
                "check_same_thread": False,
            },
            "future": True,
        }
        if self.path == ":memory:":
            engine_kwargs["poolclass"] = StaticPool
        return create_engine(self._database_url(), **engine_kwargs)

    def _database_url(self) -> URL:
        return URL.create("sqlite", database=self.path)

    def _initialize_schema(self) -> None:
        meta_table = self._tables["mcp_storage_meta"]
        with self._engine.begin() as connection:
            meta_table.create(connection, checkfirst=True)
            self._ensure_compatible_schema_version(connection)
        self._metadata.create_all(self._engine)

    def _ensure_compatible_schema_version(self, connection: Any) -> None:
        meta_table = self._tables["mcp_storage_meta"]
        current = connection.execute(
            select(meta_table.c.value).where(meta_table.c.key == "schema_version")
        ).mappings().first()
        if current is not None and int(current["value"]) > self.SCHEMA_VERSION:
            raise RuntimeError(
                f"SQLite MCP store schema {current['value']} is newer than supported "
                f"schema {self.SCHEMA_VERSION}"
            )
        statement = sqlite_insert(meta_table).values(
            key="schema_version",
            value=str(self.SCHEMA_VERSION),
        )
        connection.execute(
            statement.on_conflict_do_update(
                index_elements=[meta_table.c.key],
                set_={"value": statement.excluded.value},
            )
        )

    def _upsert_row(
        self,
        table_name: str,
        values: Mapping[str, Any],
        *,
        update_columns: tuple[str, ...],
    ) -> None:
        table = self._table(table_name)
        statement = sqlite_insert(table).values(**values)
        connection_values = {
            column: getattr(statement.excluded, column)
            for column in update_columns
        }
        with self._engine.begin() as connection:
            connection.execute(
                statement.on_conflict_do_update(
                    index_elements=[table.c.id],
                    set_=connection_values,
                )
            )

    def _insert_row(self, table_name: str, values: Mapping[str, Any]) -> None:
        table = self._table(table_name)
        with self._engine.begin() as connection:
            connection.execute(table.insert().values(**values))

    def _delete_by_id(self, table_name: str, item_id: str) -> bool:
        table = self._table(table_name)
        with self._engine.begin() as connection:
            result = connection.execute(delete(table).where(table.c.id == item_id))
        return bool(result.rowcount and result.rowcount > 0)

    def _get_model(
        self,
        table_name: str,
        item_id: str,
        model_cls: type[ModelT],
    ) -> ModelT | None:
        table = self._table(table_name)
        with self._engine.connect() as connection:
            row = connection.execute(
                select(table.c.payload).where(table.c.id == item_id)
            ).mappings().first()
        if row is None:
            return None
        return self._load_model(row["payload"], model_cls)

    def _select_payloads(
        self,
        table_name: str,
        filters: Mapping[str, Any] | None = None,
        *,
        limit: int | None = None,
        order_by: str = "id ASC",
    ) -> list[Mapping[str, Any]]:
        filters = filters or {}
        self._validate_filter_request(table_name, filters, order_by)
        table = self._table(table_name)
        statement = select(table.c.payload)
        for column, value in filters.items():
            if value is None:
                continue
            statement = statement.where(table.c[column] == value)
        for column in self._order_by_columns(table, order_by):
            statement = statement.order_by(column)
        if limit is not None:
            statement = statement.limit(limit)
        with self._engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(statement).mappings().all()
            ]

    def _validate_filter_request(
        self,
        table_name: str,
        filters: Mapping[str, Any],
        order_by: str,
    ) -> None:
        allowed_columns = self._FILTERABLE_COLUMNS.get(table_name)
        if allowed_columns is None and filters:
            raise ValueError(f"Unsupported filter table: {table_name}")
        unknown_columns = set(filters) - (allowed_columns or frozenset())
        if unknown_columns:
            raise ValueError(
                f"Unsupported filter columns for {table_name}: {sorted(unknown_columns)}"
            )
        if order_by not in self._ORDER_BY_KEYS:
            raise ValueError(f"Unsupported order by clause: {order_by}")

    def _table(self, table_name: str) -> Table:
        try:
            return self._tables[table_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported table: {table_name}") from exc

    @staticmethod
    def _order_by_columns(table: Table, order_by: str) -> list[Any]:
        if order_by == "id ASC":
            return [table.c.id.asc()]
        if order_by == "created_at DESC, id DESC":
            return [table.c.created_at.desc(), table.c.id.desc()]
        raise ValueError(f"Unsupported order by clause: {order_by}")

    @staticmethod
    def _enable_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
        finally:
            cursor.close()

    @classmethod
    def _build_tables(cls, metadata: MetaData) -> dict[str, Table]:
        tables: dict[str, Table] = {}
        tables["mcp_storage_meta"] = Table(
            "mcp_storage_meta",
            metadata,
            Column("key", String, primary_key=True),
            Column("value", String, nullable=False),
        )
        tables["mcp_profiles"] = Table(
            "mcp_profiles",
            metadata,
            Column("id", String, primary_key=True),
            Column("enabled", Integer, nullable=False),
            Column("updated_at", String, nullable=False),
            Column("payload", Text, nullable=False),
        )
        tables["mcp_profile_assignments"] = Table(
            "mcp_profile_assignments",
            metadata,
            Column("id", String, primary_key=True),
            Column(
                "profile_id",
                String,
                ForeignKey("mcp_profiles.id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("principal_id", String),
            Column("workspace_id", String),
            Column("is_default", Integer, nullable=False),
            Column("enabled", Integer, nullable=False),
            Column("updated_at", String, nullable=False),
            Column("payload", Text, nullable=False),
        )
        tables["mcp_approval_policies"] = Table(
            "mcp_approval_policies",
            metadata,
            Column("id", String, primary_key=True),
            Column(
                "profile_id",
                String,
                ForeignKey("mcp_profiles.id", ondelete="CASCADE"),
            ),
            Column("enabled", Integer, nullable=False),
            Column("updated_at", String, nullable=False),
            Column("payload", Text, nullable=False),
        )
        tables["mcp_credential_grants"] = Table(
            "mcp_credential_grants",
            metadata,
            Column("id", String, primary_key=True),
            Column(
                "profile_id",
                String,
                ForeignKey("mcp_profiles.id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("external_server_id", String),
            Column("enabled", Integer, nullable=False),
            Column("updated_at", String, nullable=False),
            Column("payload", Text, nullable=False),
        )
        tables["mcp_external_servers"] = Table(
            "mcp_external_servers",
            metadata,
            Column("id", String, primary_key=True),
            Column("enabled", Integer, nullable=False),
            Column("transport", String, nullable=False),
            Column("updated_at", String, nullable=False),
            Column("payload", Text, nullable=False),
        )
        tables["mcp_audit_events"] = Table(
            "mcp_audit_events",
            metadata,
            Column("id", String, primary_key=True),
            Column("actor_id", String),
            Column("profile_id", String),
            Column("event_type", String, nullable=False),
            Column("created_at", String, nullable=False),
            Column("payload", Text, nullable=False),
        )
        cls._add_indexes(tables)
        return tables

    @staticmethod
    def _add_indexes(tables: Mapping[str, Table]) -> None:
        Index(
            "idx_mcp_assignments_profile",
            tables["mcp_profile_assignments"].c.profile_id,
        )
        Index(
            "idx_mcp_assignments_principal",
            tables["mcp_profile_assignments"].c.principal_id,
        )
        Index(
            "idx_mcp_assignments_workspace",
            tables["mcp_profile_assignments"].c.workspace_id,
        )
        Index(
            "idx_mcp_policies_profile",
            tables["mcp_approval_policies"].c.profile_id,
        )
        Index(
            "idx_mcp_grants_profile",
            tables["mcp_credential_grants"].c.profile_id,
        )
        Index(
            "idx_mcp_grants_external_server",
            tables["mcp_credential_grants"].c.external_server_id,
        )
        Index(
            "idx_mcp_external_enabled",
            tables["mcp_external_servers"].c.enabled,
        )
        Index("idx_mcp_audit_actor", tables["mcp_audit_events"].c.actor_id)
        Index("idx_mcp_audit_profile", tables["mcp_audit_events"].c.profile_id)
        Index(
            "idx_mcp_audit_event_type",
            tables["mcp_audit_events"].c.event_type,
        )
        Index(
            "idx_mcp_audit_created_at",
            tables["mcp_audit_events"].c.created_at,
        )

    @staticmethod
    def _dump_model(model: BaseModel) -> str:
        return json.dumps(
            model.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _load_model(payload: Any, model_cls: type[ModelT]) -> ModelT:
        return model_cls.model_validate_json(cast(str, payload))

    @staticmethod
    def _normalize_audit_event(event: AuditEvent) -> AuditEvent:
        return event.model_copy(
            update={"created_at": event.created_at.astimezone(timezone.utc)},
            deep=True,
        )

    @staticmethod
    def _validate_profile(profile: MCPProfile | Mapping[str, Any]) -> MCPProfile:
        if isinstance(profile, MCPProfile):
            return profile.model_copy(deep=True)
        return MCPProfile.model_validate(profile)


__all__ = ["SQLiteMCPStore"]
