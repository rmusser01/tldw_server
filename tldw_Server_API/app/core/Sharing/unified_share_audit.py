"""Unified Sharing audit writer and compatibility projection helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite
from loguru import logger

from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditContext,
    AuditEvent,
    UnifiedAuditService,
    _normalize_result,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection_async,
)
from tldw_Server_API.app.core.exceptions import ValidationError

_SHARE_EVENT_FILTER_SQL = (
    "(event_type LIKE 'share.%'"
    " OR event_type IN ('token.created','token.used','token.revoked','token.password_verified','token.password_failed'))"
)
_SHARE_AUDIT_STATE_TABLE = "share_audit_state"
_SHARE_AUDIT_SEQUENCE_KEY = "compatibility_id_seq"
_RESERVED_METADATA_KEYS = {
    "compatibility_id",
    "owner_user_id",
    "actor_user_id",
    "share_id",
    "token_id",
    "_legacy_created_at",
}



def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            raise TypeError("bool is not a valid integer")
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _load_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(loaded) if isinstance(loaded, dict) else {}
    return {}


def _visible_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in _RESERVED_METADATA_KEYS and value is not None
    }


def _normalize_timestamp_value(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text:
            if text.endswith('Z'):
                text = text[:-1] + '+00:00'
            try:
                parsed = datetime.fromisoformat(text)
                return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    return datetime.now(timezone.utc)


def _legacy_event_id(legacy_id: int) -> str:
    return f"share-audit-legacy-{legacy_id}"


@dataclass(frozen=True)
class SharingIdentityState:
    compatibility_ids: frozenset[int]
    legacy_ids: frozenset[int]
    max_compatibility_id: int


class UnifiedShareAuditWriter:
    """Write Sharing events into unified audit and project them back compatibly."""

    def __init__(
        self,
        db_path: str | None = None,
        service: UnifiedAuditService | None = None,
    ) -> None:
        resolved = Path(db_path) if db_path else DatabasePaths.get_shared_audit_db_path()
        self.db_path = str(resolved)
        if service is not None:
            self._service = service
            self._owns_service = False
        else:
            self._service = UnifiedAuditService(db_path=self.db_path, storage_mode="shared")
            self._owns_service = True
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self._service.initialize(start_background_tasks=False)
        # Mark initialized immediately so stop() can clean up the embedded
        # service if the compatibility steps below fail.
        self._initialized = True
        await self._ensure_compatibility_state()
        await self._sync_compatibility_floor()

    async def stop(self) -> None:
        if not self._initialized:
            return
        if self._owns_service:
            await self._service.stop()
        self._initialized = False

    async def _open_db(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        await configure_sqlite_connection_async(db)
        return db

    async def _ensure_compatibility_state(self) -> None:
        db = await self._open_db()
        try:
            await self._ensure_compatibility_state_db(db)
            await db.commit()
        finally:
            await db.close()

    async def _ensure_compatibility_state_db(self, db: aiosqlite.Connection) -> None:
        await db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_SHARE_AUDIT_STATE_TABLE} (
                key TEXT PRIMARY KEY,
                int_value INTEGER NOT NULL
            )
            """
        )
        await db.execute(
            f"""
            INSERT OR IGNORE INTO {_SHARE_AUDIT_STATE_TABLE} (key, int_value)
            VALUES (?, 0)
            """,
            (_SHARE_AUDIT_SEQUENCE_KEY,),
        )

    async def _current_sequence_value(self) -> int:
        db = await self._open_db()
        try:
            return await self._current_sequence_value_db(db)
        finally:
            await db.close()

    async def _current_sequence_value_db(self, db: aiosqlite.Connection) -> int:
        await self._ensure_compatibility_state_db(db)
        async with db.execute(
            f"SELECT int_value FROM {_SHARE_AUDIT_STATE_TABLE} WHERE key = ?",  # nosec B608
            (_SHARE_AUDIT_SEQUENCE_KEY,),
        ) as cur:
            row = await cur.fetchone()
        return int(row["int_value"]) if row is not None else 0

    async def get_identity_state(self) -> SharingIdentityState:
        await self.initialize()
        db = await self._open_db()
        try:
            return await self._load_identity_state_db(db)
        finally:
            await db.close()

    async def _load_identity_state_db(self, db: aiosqlite.Connection) -> SharingIdentityState:
        async with db.execute(
            f"SELECT metadata FROM audit_events WHERE {_SHARE_EVENT_FILTER_SQL}"  # nosec B608
        ) as cur:
            rows = await cur.fetchall()

        compatibility_ids: set[int] = set()
        legacy_ids: set[int] = set()
        max_id = 0
        for row in rows:
            metadata = _load_metadata(row["metadata"])
            compatibility_id = _coerce_int(
                metadata.get("compatibility_id")
                or metadata.get("legacy_share_audit_id")
            )
            if compatibility_id is not None:
                compatibility_ids.add(compatibility_id)
                max_id = max(max_id, compatibility_id)
            legacy_id = _coerce_int(metadata.get("legacy_share_audit_id"))
            if legacy_id is not None:
                legacy_ids.add(legacy_id)
        return SharingIdentityState(
            compatibility_ids=frozenset(compatibility_ids),
            legacy_ids=frozenset(legacy_ids),
            max_compatibility_id=max_id,
        )

    async def _max_existing_compatibility_id(self) -> int:
        db = await self._open_db()
        try:
            state = await self._load_identity_state_db(db)
            return state.max_compatibility_id
        finally:
            await db.close()

    async def _sync_compatibility_floor(self) -> None:
        floor = max(
            await self._current_sequence_value(),
            await self._max_existing_compatibility_id(),
        )
        await self.bump_compatibility_floor(floor)

    async def bump_compatibility_floor(self, floor: int) -> None:
        db = await self._open_db()
        try:
            await db.execute("BEGIN IMMEDIATE")
            await self._ensure_compatibility_state_db(db)
            current = await self._current_sequence_value_db(db)
            if floor > current:
                await db.execute(
                    f"UPDATE {_SHARE_AUDIT_STATE_TABLE} SET int_value = ? WHERE key = ?",  # nosec B608
                    (floor, _SHARE_AUDIT_SEQUENCE_KEY),
                )
            await db.commit()
        finally:
            await db.close()

    async def _allocate_compatibility_id_db(self, db: aiosqlite.Connection) -> int:
        await self._ensure_compatibility_state_db(db)
        await db.execute(
            f"UPDATE {_SHARE_AUDIT_STATE_TABLE} SET int_value = int_value + 1 WHERE key = ?",  # nosec B608
            (_SHARE_AUDIT_SEQUENCE_KEY,),
        )
        async with db.execute(
            f"SELECT int_value FROM {_SHARE_AUDIT_STATE_TABLE} WHERE key = ?",  # nosec B608
            (_SHARE_AUDIT_SEQUENCE_KEY,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise RuntimeError("Failed to allocate Sharing audit compatibility id")
        return int(row["int_value"])

    async def _bump_compatibility_floor_db(self, db: aiosqlite.Connection, floor: int) -> None:
        current = await self._current_sequence_value_db(db)
        if floor > current:
            await db.execute(
                f"UPDATE {_SHARE_AUDIT_STATE_TABLE} SET int_value = ? WHERE key = ?",  # nosec B608
                (floor, _SHARE_AUDIT_SEQUENCE_KEY),
            )

    async def _event_exists_db(self, db: aiosqlite.Connection, event_id: str) -> bool:
        async with db.execute(
            "SELECT 1 FROM audit_events WHERE event_id = ? LIMIT 1",
            (event_id,),
        ) as cur:
            return await cur.fetchone() is not None

    async def _compatibility_id_occupied_db(
        self,
        db: aiosqlite.Connection,
        compatibility_id: int,
    ) -> bool:
        state = await self._load_identity_state_db(db)
        return compatibility_id in state.compatibility_ids

    async def _load_previous_chain_hash_db(self, db: aiosqlite.Connection) -> str:
        async with db.execute(
            "SELECT chain_hash FROM audit_events "
            "WHERE chain_hash IS NOT NULL AND chain_hash != '' "
            "ORDER BY rowid DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        return str(row["chain_hash"]) if row and row["chain_hash"] else ""

    def _build_event(
        self,
        *,
        event_type: str,
        resource_type: str,
        resource_id: str,
        owner_user_id: int,
        actor_user_id: int | None = None,
        share_id: int | None = None,
        token_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        result: str = "success",
        compatibility_id: int,
        legacy_share_audit_id: int | None = None,
        timestamp: Any = None,
        event_id: str | None = None,
    ) -> AuditEvent:
        payload = dict(metadata or {})
        payload.update(
            {
                "compatibility_id": compatibility_id,
                "owner_user_id": owner_user_id,
                "actor_user_id": actor_user_id,
                "share_id": share_id,
                "token_id": token_id,
            }
        )
        if legacy_share_audit_id is not None:
            payload["legacy_share_audit_id"] = legacy_share_audit_id

        result_norm = _normalize_result(result)
        category = self._service._determine_category(event_type)
        severity = self._service._determine_severity(event_type, result_norm)
        context = AuditContext(
            user_id=str(actor_user_id) if actor_user_id is not None else None,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        tenant_user_id = self._service._resolve_tenant_id_for_write(
            raw_tenant=str(owner_user_id),
            context_user_id=context.user_id,
            event_type=event_type,
            category=category,
        )
        return AuditEvent(
            event_id=event_id or str(uuid4()),
            timestamp=_normalize_timestamp_value(timestamp),
            category=category,
            event_type=event_type,
            severity=severity,
            tenant_user_id=tenant_user_id,
            context=context,
            resource_type=resource_type,
            resource_id=resource_id,
            action=event_type,
            result=result_norm,
            metadata=payload,
        )

    async def import_legacy_event(
        self,
        *,
        legacy_share_audit_id: int | None = None,
        legacy_audit_id: int | None = None,
        event_type: str,
        resource_type: str,
        resource_id: str,
        owner_user_id: int,
        actor_user_id: int | None = None,
        share_id: int | None = None,
        token_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        result: str = "success",
        created_at: Any = None,
    ) -> int | None:
        """Import a legacy share-audit row while preserving its compatibility id."""
        legacy_id = legacy_share_audit_id
        if legacy_id is None:
            legacy_id = legacy_audit_id
        elif legacy_audit_id is not None and legacy_audit_id != legacy_id:
            raise ValidationError(
                "legacy_share_audit_id and legacy_audit_id must match when both are provided"
            )
        if legacy_id is None:
            raise ValidationError("legacy_share_audit_id is required")

        compatibility_id = await self._write_event_transaction(
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            owner_user_id=owner_user_id,
            actor_user_id=actor_user_id,
            share_id=share_id,
            token_id=token_id,
            metadata=metadata,
            ip_address=ip_address,
            user_agent=user_agent,
            result=result,
            compatibility_id=legacy_id,
            legacy_share_audit_id=legacy_id,
            timestamp=created_at,
            event_id=_legacy_event_id(legacy_id),
        )
        return compatibility_id

    async def _write_event_transaction(
        self,
        *,
        event_type: str,
        resource_type: str,
        resource_id: str,
        owner_user_id: int,
        actor_user_id: int | None = None,
        share_id: int | None = None,
        token_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        result: str = "success",
        compatibility_id: int | None = None,
        legacy_share_audit_id: int | None = None,
        timestamp: Any = None,
        event_id: str | None = None,
    ) -> int | None:
        await self.initialize()
        db = await self._open_db()
        try:
            await db.execute("BEGIN IMMEDIATE")
            await self._ensure_compatibility_state_db(db)

            if event_id is not None and await self._event_exists_db(db, event_id):
                await db.rollback()
                return None

            if compatibility_id is None:
                compatibility_id = await self._allocate_compatibility_id_db(db)
            else:
                if await self._compatibility_id_occupied_db(db, compatibility_id):
                    raise RuntimeError(
                        f"Sharing audit compatibility id {compatibility_id} is already in use"
                    )
                await self._bump_compatibility_floor_db(db, compatibility_id)

            event = self._build_event(
                event_type=event_type,
                resource_type=resource_type,
                resource_id=resource_id,
                owner_user_id=owner_user_id,
                actor_user_id=actor_user_id,
                share_id=share_id,
                token_id=token_id,
                metadata=metadata,
                ip_address=ip_address,
                user_agent=user_agent,
                result=result,
                compatibility_id=compatibility_id,
                legacy_share_audit_id=legacy_share_audit_id,
                timestamp=timestamp,
                event_id=event_id,
            )
            record = event.to_dict()
            self._service._ensure_record_tenant_ids([record])
            previous_hash = await self._load_previous_chain_hash_db(db)
            committed_chain_hash = self._service._apply_chain_hashes(
                [record],
                previous_hash=previous_hash,
            )
            await db.execute(self._service._event_insert_sql, record)
            await self._service._update_daily_stats(db, [event])
            await db.commit()
        finally:
            await db.close()

        self._service._last_chain_hash = committed_chain_hash
        self._service.stats["events_logged"] += 1
        return compatibility_id

    async def log_event(
        self,
        *,
        event_type: str,
        resource_type: str,
        resource_id: str,
        owner_user_id: int,
        actor_user_id: int | None = None,
        share_id: int | None = None,
        token_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        result: str = "success",
    ) -> int:
        compatibility_id = await self._write_event_transaction(
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            owner_user_id=owner_user_id,
            actor_user_id=actor_user_id,
            share_id=share_id,
            token_id=token_id,
            metadata=metadata,
            ip_address=ip_address,
            user_agent=user_agent,
            result=result,
        )
        if compatibility_id is None:
            raise RuntimeError("Sharing audit write unexpectedly returned no compatibility id")
        return compatibility_id

    async def query_events(
        self,
        *,
        owner_user_id: int | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        await self.initialize()
        conditions = [_SHARE_EVENT_FILTER_SQL]
        params: list[Any] = []
        if owner_user_id is not None:
            conditions.append(
                "(tenant_user_id = ? OR json_extract(metadata, '$.owner_user_id') = ?)"
            )
            owner_str = str(owner_user_id)
            params.extend([owner_str, owner_user_id])
        if resource_type is not None:
            conditions.append("resource_type = ?")
            params.append(resource_type)
        if resource_id is not None:
            conditions.append("resource_id = ?")
            params.append(resource_id)
        params.extend([limit, offset])

        query = f"""
            SELECT
                event_id,
                timestamp,
                event_type,
                tenant_user_id,
                context_user_id,
                context_ip_address,
                context_user_agent,
                resource_type,
                resource_id,
                metadata
            FROM audit_events
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp DESC, rowid DESC
            LIMIT ? OFFSET ?
        """

        db = await self._open_db()
        try:
            async with db.execute(query, params) as cur:
                rows = await cur.fetchall()
        finally:
            await db.close()

        projected: list[dict[str, Any]] = []
        for row in rows:
            result = self._project_row(dict(row))
            if result is not None:
                projected.append(result)
        return projected

    def _project_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        metadata = _load_metadata(row.get("metadata"))
        compatibility_id = _coerce_int(
            metadata.get("compatibility_id")
            or metadata.get("legacy_share_audit_id")
        )
        if compatibility_id is None:
            logger.warning(
                "Skipping sharing audit row without compatibility_id: event_id={}, event_type={}",
                row.get("event_id", "?"),
                row.get("event_type", "?"),
            )
            return None

        owner_user_id = _coerce_int(row.get("tenant_user_id"))
        if owner_user_id is None:
            owner_user_id = _coerce_int(metadata.get("owner_user_id"))
        if owner_user_id is None:
            logger.warning(
                "Skipping sharing audit row without owner_user_id: event_id={}, event_type={}",
                row.get("event_id", "?"),
                row.get("event_type", "?"),
            )
            return None

        actor_user_id = _coerce_int(metadata.get("actor_user_id"))
        if actor_user_id is None:
            actor_user_id = _coerce_int(row.get("context_user_id"))

        created_at = metadata.get("_legacy_created_at") or row.get("timestamp")

        return {
            "id": compatibility_id,
            "event_id": row.get("event_id"),
            "event_type": str(row.get("event_type") or ""),
            "actor_user_id": actor_user_id,
            "resource_type": row.get("resource_type"),
            "resource_id": row.get("resource_id"),
            "owner_user_id": owner_user_id,
            "share_id": _coerce_int(metadata.get("share_id")),
            "token_id": _coerce_int(metadata.get("token_id")),
            "metadata": _visible_metadata(metadata),
            "ip_address": row.get("context_ip_address"),
            "user_agent": row.get("context_user_agent"),
            "created_at": created_at,
        }
