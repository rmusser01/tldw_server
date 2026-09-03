"""Crash-resumable encryption-key rotation for canonical admin webhooks."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias

from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    DATABASE_PROTECTED_TABLE_ORDER,
    AdminWebhookRepository,
    MigrationState,
    WebhookRepositoryError,
    WebhookRepositoryErrorCode,
)
from tldw_Server_API.app.services import admin_system_ops_service as system_ops

from .audit import (
    OperationalAction,
    OperationalAudit,
    OperationalAuditSink,
    OperationalOutcome,
    WebhookOperationalReasonCode,
)
from .catalog import EVENT_API_VERSION
from .crypto import WebhookKeyError, WebhookKeyRing
from .domain import (
    PendingIncidentWebhookMarker,
    WebhookError,
    WebhookErrorCode,
)
from .events import decrypt_pending_incident_marker_body

PENDING_INCIDENT_MARKER_TABLE = "pending_incident_markers"
PROTECTED_TABLE_ORDER = DATABASE_PROTECTED_TABLE_ORDER + (
    PENDING_INCIDENT_MARKER_TABLE,
)

_ACTIVE_ROTATION_PHASES = frozenset(
    {"rewriting", "verifying", "awaiting_primary_cutover"}
)
_KEY_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_PENDING_INCIDENT_EVENTS = frozenset(
    {
        "incident.created",
        "incident.updated",
        "incident.resolved",
        "incident.notify",
    }
)


@dataclass(frozen=True)
class KeyRotationProgress:
    """Sanitized durable progress for one rotation operation."""

    operation_id: str
    source_key_id: str
    target_key_id: str
    phase: str
    table_cursor: str | None
    key_cursor: str | None
    processed_count: int
    verified_count: int
    started_at: datetime
    completed_at: datetime | None


_RotationOperation: TypeAlias = Callable[[datetime], Awaitable[KeyRotationProgress]]


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("rotation clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _validate_key_id(value: str) -> str:
    if not isinstance(value, str) or _KEY_ID_PATTERN.fullmatch(value) is None:
        raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
    return value


def _progress(state: MigrationState) -> KeyRotationProgress:
    if (
        state.rotation_operation_id is None
        or state.rotation_source_key_id is None
        or state.rotation_target_key_id is None
        or state.rotation_phase is None
        or state.rotation_started_at is None
    ):
        raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
    return KeyRotationProgress(
        operation_id=state.rotation_operation_id,
        source_key_id=state.rotation_source_key_id,
        target_key_id=state.rotation_target_key_id,
        phase=state.rotation_phase,
        table_cursor=state.rotation_table_cursor,
        key_cursor=state.rotation_key_cursor,
        processed_count=state.rotation_processed_count,
        verified_count=state.rotation_verified_count,
        started_at=state.rotation_started_at,
        completed_at=state.rotation_completed_at,
    )


def _mapped_error(exc: BaseException) -> WebhookError:
    if isinstance(exc, WebhookError):
        return exc
    if isinstance(exc, WebhookRepositoryError):
        if exc.code is WebhookRepositoryErrorCode.DATABASE_BUSY:
            return WebhookError(WebhookErrorCode.DATABASE_BUSY)
        if exc.code is WebhookRepositoryErrorCode.STALE_MIGRATION_STATE:
            return WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
        return WebhookError(WebhookErrorCode.OPERATION_FAILED)
    if isinstance(exc, WebhookKeyError):
        return WebhookError(WebhookErrorCode.KEY_UNAVAILABLE)
    if isinstance(exc, ValueError):
        return WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
    return WebhookError(WebhookErrorCode.OPERATION_FAILED)


def _operational_reason(error: WebhookError) -> WebhookOperationalReasonCode:
    return {
        WebhookErrorCode.VALIDATION_FAILED: (
            WebhookOperationalReasonCode.VALIDATION_FAILED
        ),
        WebhookErrorCode.PRECONDITION_FAILED: (
            WebhookOperationalReasonCode.PRECONDITION_FAILED
        ),
        WebhookErrorCode.MIGRATION_PENDING: (
            WebhookOperationalReasonCode.PRECONDITION_FAILED
        ),
        WebhookErrorCode.KEY_ROTATION_IN_PROGRESS: (
            WebhookOperationalReasonCode.PRECONDITION_FAILED
        ),
        WebhookErrorCode.DATABASE_BUSY: WebhookOperationalReasonCode.DATABASE_BUSY,
        WebhookErrorCode.AUDIT_UNAVAILABLE: (
            WebhookOperationalReasonCode.AUDIT_UNAVAILABLE
        ),
        WebhookErrorCode.KEY_UNAVAILABLE: WebhookOperationalReasonCode.KEY_UNAVAILABLE,
        WebhookErrorCode.KEY_CONFIGURATION_MISMATCH: (
            WebhookOperationalReasonCode.KEY_CONFIGURATION_MISMATCH
        ),
    }.get(error.code, WebhookOperationalReasonCode.OPERATION_FAILED)


class WebhookKeyRotationService:
    """Rotate every canonical protected value with durable bounded progress."""

    def __init__(
        self,
        *,
        repository: AdminWebhookRepository,
        key_ring: WebhookKeyRing,
        system_ops_path: Path | None = None,
        batch_size: int = 100,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(repository, AdminWebhookRepository):
            raise TypeError("repository is required")
        if not isinstance(key_ring, WebhookKeyRing):
            raise TypeError("key_ring is required")
        if isinstance(batch_size, bool) or not 1 <= batch_size <= 500:
            raise ValueError("batch_size must be between 1 and 500")
        self._repository = repository
        self._key_ring = key_ring
        self._system_ops_path = system_ops_path
        self._batch_size = batch_size
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def _store_path(self) -> Path:
        return self._system_ops_path or system_ops._STORE_PATH

    async def _emit(
        self,
        sink: OperationalAuditSink,
        *,
        operator_id: int,
        action: OperationalAction,
        operation_id: str,
        outcome: OperationalOutcome,
        request_id: str,
        reason_code: WebhookOperationalReasonCode | None = None,
    ) -> None:
        await sink(
            OperationalAudit(
                operator_id=operator_id,
                action=action,
                operation_id=operation_id,
                outcome=outcome,
                request_id=request_id,
                reason_code=reason_code,
            )
        )

    async def _run(
        self,
        *,
        action: OperationalAction,
        operation_id: str,
        operator_id: int,
        request_id: str,
        audit_sink: OperationalAuditSink,
        operation: _RotationOperation,
    ) -> KeyRotationProgress:
        at = _utc(self._clock())
        try:
            await self._emit(
                audit_sink,
                operator_id=operator_id,
                action=action,
                operation_id=operation_id,
                outcome="accepted",
                request_id=request_id,
            )
        except Exception:  # noqa: BLE001 - mandatory pre-operation audit fails closed
            raise WebhookError(WebhookErrorCode.AUDIT_UNAVAILABLE) from None

        try:
            result = await operation(at)
        except Exception as exc:  # noqa: BLE001 - sanitize the operational boundary
            mapped = _mapped_error(exc)
            with suppress(Exception):
                await self._emit(
                    audit_sink,
                    operator_id=operator_id,
                    action=action,
                    operation_id=operation_id,
                    outcome="failed",
                    request_id=request_id,
                    reason_code=_operational_reason(mapped),
                )
            raise mapped from None

        with suppress(Exception):
            await self._emit(
                audit_sink,
                operator_id=operator_id,
                action=action,
                operation_id=operation_id,
                outcome="completed",
                request_id=request_id,
            )
        return result

    async def start(
        self,
        operation_id: str,
        source_key_id: str,
        target_key_id: str,
        *,
        operator_id: int,
        request_id: str,
        audit_sink: OperationalAuditSink,
    ) -> KeyRotationProgress:
        """Start one exclusive forward-only rotation after mandatory audit."""

        async def operation(at: datetime) -> KeyRotationProgress:
            source = _validate_key_id(source_key_id)
            target = _validate_key_id(target_key_id)
            if source == target:
                raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
            if not self._key_ring.has_key(source) or not self._key_ring.has_key(target):
                raise WebhookError(WebhookErrorCode.KEY_UNAVAILABLE)

            async with self._repository.transaction() as tx:
                state = await tx.lock_migration_state()
                if state.phase != "complete" or state.completed_at is None:
                    raise WebhookError(WebhookErrorCode.MIGRATION_PENDING)
                if state.rotation_phase in _ACTIVE_ROTATION_PHASES:
                    if (
                        state.rotation_operation_id == operation_id
                        and state.rotation_source_key_id == source
                        and state.rotation_target_key_id == target
                    ):
                        return _progress(state)
                    raise WebhookError(WebhookErrorCode.KEY_ROTATION_IN_PROGRESS)
                if (
                    state.rotation_phase == "complete"
                    and state.rotation_operation_id == operation_id
                ):
                    if (
                        state.rotation_source_key_id != source
                        or state.rotation_target_key_id != target
                    ):
                        raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
                    return _progress(state)
                if (
                    state.active_primary_key_id != source
                    or self._key_ring.primary_id != source
                ):
                    raise WebhookError(WebhookErrorCode.KEY_CONFIGURATION_MISMATCH)
                updated = await tx.compare_and_set_migration_state(
                    expected_revision=state.state_revision,
                    updates={
                        "rotation_operation_id": operation_id,
                        "rotation_source_key_id": source,
                        "rotation_target_key_id": target,
                        "rotation_phase": "rewriting",
                        "rotation_table_cursor": PROTECTED_TABLE_ORDER[0],
                        "rotation_key_cursor": None,
                        "rotation_processed_count": 0,
                        "rotation_verified_count": 0,
                        "rotation_started_at": at,
                        "rotation_completed_at": None,
                    },
                    at=at,
                )
            return _progress(updated)

        return await self._run(
            action="admin_webhook.key_rotation.start",
            operation_id=operation_id,
            operator_id=operator_id,
            request_id=request_id,
            audit_sink=audit_sink,
            operation=operation,
        )

    def _require_rotation(
        self,
        state: MigrationState,
        *,
        operation_id: str,
        phases: frozenset[str],
        required_primary: str,
    ) -> tuple[str, str]:
        if state.rotation_operation_id != operation_id or state.rotation_phase not in phases:
            raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
        source = state.rotation_source_key_id
        target = state.rotation_target_key_id
        if source is None or target is None:
            raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
        if not self._key_ring.has_key(source) or not self._key_ring.has_key(target):
            raise WebhookError(WebhookErrorCode.KEY_UNAVAILABLE)
        if (
            state.active_primary_key_id != source
            or self._key_ring.primary_id != required_primary
        ):
            raise WebhookError(WebhookErrorCode.KEY_CONFIGURATION_MISMATCH)
        return source, target

    async def resume(
        self,
        operation_id: str,
        *,
        operator_id: int,
        request_id: str,
        audit_sink: OperationalAuditSink,
    ) -> KeyRotationProgress:
        """Resume bounded committed batches until rewriting is complete."""

        async def operation(at: datetime) -> KeyRotationProgress:
            while True:
                state = await self._repository.get_migration_state()
                source = state.rotation_source_key_id
                if source is None:
                    raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
                if state.rotation_operation_id != operation_id:
                    raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
                if state.rotation_phase in {
                    "verifying",
                    "awaiting_primary_cutover",
                    "complete",
                }:
                    return _progress(state)
                self._require_rotation(
                    state,
                    operation_id=operation_id,
                    phases=frozenset({"rewriting"}),
                    required_primary=source,
                )
                if state.rotation_table_cursor == PENDING_INCIDENT_MARKER_TABLE:
                    await self._rewrite_file_batch(operation_id=operation_id, at=at)
                elif state.rotation_table_cursor in DATABASE_PROTECTED_TABLE_ORDER:
                    await self._rewrite_database_batch(operation_id=operation_id, at=at)
                else:
                    raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)

        return await self._run(
            action="admin_webhook.key_rotation.resume",
            operation_id=operation_id,
            operator_id=operator_id,
            request_id=request_id,
            audit_sink=audit_sink,
            operation=operation,
        )

    async def _rewrite_database_batch(
        self,
        *,
        operation_id: str,
        at: datetime,
    ) -> MigrationState:
        async with self._repository.transaction() as tx:
            state = await tx.lock_migration_state()
            source = state.rotation_source_key_id
            if source is None:
                raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
            _, target = self._require_rotation(
                state,
                operation_id=operation_id,
                phases=frozenset({"rewriting"}),
                required_primary=source,
            )
            table = state.rotation_table_cursor
            if table not in DATABASE_PROTECTED_TABLE_ORDER:
                raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
            rows = await tx.page_protected_rows(
                table=table,
                after=state.rotation_key_cursor,
                limit=self._batch_size,
                inventory_at=state.rotation_started_at,
            )
            if not rows:
                next_index = PROTECTED_TABLE_ORDER.index(table) + 1
                return await tx.compare_and_set_migration_state(
                    expected_revision=state.state_revision,
                    updates={
                        "rotation_table_cursor": PROTECTED_TABLE_ORDER[next_index],
                        "rotation_key_cursor": None,
                    },
                    at=at,
                )

            for row in rows:
                if row.protected.key_id == target:
                    self._key_ring.decrypt_bytes(
                        purpose=row.purpose,
                        identity=row.envelope_identity,
                        protected=row.protected,
                    )
                    continue
                replacement = self._key_ring.reencrypt_to_key(
                    row.protected,
                    purpose=row.purpose,
                    identity=row.envelope_identity,
                    target_key_id=target,
                )
                replaced = await tx.replace_protected_value(
                    row,
                    expected_ciphertext=row.protected.ciphertext_json,
                    replacement=replacement,
                )
                if not replaced:
                    raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)

            return await tx.compare_and_set_migration_state(
                expected_revision=state.state_revision,
                updates={
                    "rotation_key_cursor": rows[-1].row_identity,
                    "rotation_processed_count": (
                        state.rotation_processed_count + len(rows)
                    ),
                },
                at=at,
            )

    def _read_pending_markers(
        self,
    ) -> tuple[dict[str, object], list[PendingIncidentWebhookMarker]]:
        store = system_ops._load_store_strict(self._store_path)
        raw_markers = store.get("webhook_pending_events", [])
        if not isinstance(raw_markers, list):
            raise ValueError("pending incident marker collection is invalid")
        markers = [
            PendingIncidentWebhookMarker.from_store_record(value)
            for value in raw_markers
        ]
        if any(
            marker.api_version != EVENT_API_VERSION
            or marker.event_type not in _PENDING_INCIDENT_EVENTS
            for marker in markers
        ):
            raise ValueError("pending incident marker catalog value is invalid")
        markers.sort(key=lambda marker: marker.event_id)
        if len({marker.event_id for marker in markers}) != len(markers):
            raise ValueError("pending incident marker IDs are not unique")
        return store, markers

    async def _rewrite_file_batch(
        self,
        *,
        operation_id: str,
        at: datetime,
    ) -> MigrationState:
        state = await self._repository.get_migration_state()
        source = state.rotation_source_key_id
        if source is None:
            raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
        _, target = self._require_rotation(
            state,
            operation_id=operation_id,
            phases=frozenset({"rewriting"}),
            required_primary=source,
        )
        if state.rotation_table_cursor != PENDING_INCIDENT_MARKER_TABLE:
            raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)

        with system_ops._STORE_LOCK, system_ops._store_file_lock(
            store_path=self._store_path
        ):
            store, markers = self._read_pending_markers()
            page = [
                marker
                for marker in markers
                if state.rotation_key_cursor is None
                or marker.event_id > state.rotation_key_cursor
            ][: self._batch_size]
            if not page:
                last_key = None
                observed_count = 0
                completed = True
            else:
                raw_markers = store.get("webhook_pending_events", [])
                if not isinstance(raw_markers, list):
                    raise ValueError("pending incident marker collection is invalid")
                marker_positions = {
                    str(value.get("event_id")): index
                    for index, value in enumerate(raw_markers)
                    if isinstance(value, dict)
                }
                changed = False
                for marker in page:
                    _, source_identity = decrypt_pending_incident_marker_body(
                        self._key_ring,
                        marker,
                    )
                    target_identity = dict(marker.envelope_identity)
                    if (
                        marker.body.key_id == target
                        and source_identity == target_identity
                    ):
                        continue
                    replacement = self._key_ring.reencrypt_to_key(
                        marker.body,
                        purpose=marker.envelope_purpose,
                        identity=source_identity,
                        target_key_id=target,
                        target_identity=target_identity,
                    )
                    position = marker_positions[marker.event_id]
                    raw_markers[position] = replace(
                        marker,
                        body=replacement,
                    ).to_store_record()
                    changed = True
                if changed:
                    system_ops._atomic_write_store(self._store_path, store)
                last_key = page[-1].event_id
                observed_count = len(page)
                completed = False

        return await self._persist_file_batch_progress(
            operation_id=operation_id,
            expected_revision=state.state_revision,
            last_key=last_key,
            observed_count=observed_count,
            completed=completed,
            at=at,
        )

    async def _persist_file_batch_progress(
        self,
        *,
        operation_id: str,
        expected_revision: int,
        last_key: str | None,
        observed_count: int,
        completed: bool,
        at: datetime,
    ) -> MigrationState:
        async with self._repository.transaction() as tx:
            state = await tx.lock_migration_state()
            source = state.rotation_source_key_id
            if source is None or state.state_revision != expected_revision:
                raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
            self._require_rotation(
                state,
                operation_id=operation_id,
                phases=frozenset({"rewriting"}),
                required_primary=source,
            )
            if state.rotation_table_cursor != PENDING_INCIDENT_MARKER_TABLE:
                raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
            updates: dict[str, object] = {
                "rotation_key_cursor": last_key,
                "rotation_processed_count": (
                    state.rotation_processed_count + observed_count
                ),
            }
            if completed:
                updates.update(
                    {
                        "rotation_phase": "verifying",
                        "rotation_table_cursor": None,
                        "rotation_key_cursor": None,
                        "rotation_verified_count": 0,
                    }
                )
            return await tx.compare_and_set_migration_state(
                expected_revision=state.state_revision,
                updates=updates,
                at=at,
            )

    async def _verify_database_inventory(
        self,
        *,
        target: str,
        inventory_at: datetime,
    ) -> int:
        count = 0
        for table in DATABASE_PROTECTED_TABLE_ORDER:
            after: str | None = None
            while True:
                rows = await self._repository.page_protected_rows(
                    table=table,
                    after=after,
                    limit=self._batch_size,
                    inventory_at=inventory_at,
                )
                if not rows:
                    break
                for row in rows:
                    if row.protected.key_id != target:
                        raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
                    self._key_ring.decrypt_bytes(
                        purpose=row.purpose,
                        identity=row.envelope_identity,
                        protected=row.protected,
                    )
                count += len(rows)
                after = rows[-1].row_identity
        return count

    def _verify_file_inventory(self, *, target: str) -> int:
        with system_ops._STORE_LOCK, system_ops._store_file_lock(
            store_path=self._store_path
        ):
            _, markers = self._read_pending_markers()
            for marker in markers:
                if marker.body.key_id != target:
                    raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
                decrypt_pending_incident_marker_body(self._key_ring, marker)
        return len(markers)

    async def _verified_inventory_count(
        self,
        *,
        target: str,
        inventory_at: datetime,
    ) -> int:
        database_count = await self._verify_database_inventory(
            target=target,
            inventory_at=inventory_at,
        )
        file_count = self._verify_file_inventory(target=target)
        return database_count + file_count

    async def verify(
        self,
        operation_id: str,
        *,
        operator_id: int,
        request_id: str,
        audit_sink: OperationalAuditSink,
    ) -> KeyRotationProgress:
        """Read every rotated value before allowing primary cutover."""

        async def operation(at: datetime) -> KeyRotationProgress:
            state = await self._repository.get_migration_state()
            if state.rotation_operation_id != operation_id:
                raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
            if state.rotation_phase in {"awaiting_primary_cutover", "complete"}:
                return _progress(state)
            source = state.rotation_source_key_id
            if source is None:
                raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
            _, target = self._require_rotation(
                state,
                operation_id=operation_id,
                phases=frozenset({"verifying"}),
                required_primary=source,
            )
            if state.rotation_started_at is None:
                raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
            verified_count = await self._verified_inventory_count(
                target=target,
                inventory_at=state.rotation_started_at,
            )
            if verified_count != state.rotation_processed_count:
                raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
            async with self._repository.transaction() as tx:
                current = await tx.lock_migration_state()
                if current.state_revision != state.state_revision:
                    raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
                updated = await tx.compare_and_set_migration_state(
                    expected_revision=current.state_revision,
                    updates={
                        "rotation_phase": "awaiting_primary_cutover",
                        "rotation_verified_count": verified_count,
                    },
                    at=at,
                )
            return _progress(updated)

        return await self._run(
            action="admin_webhook.key_rotation.verify",
            operation_id=operation_id,
            operator_id=operator_id,
            request_id=request_id,
            audit_sink=audit_sink,
            operation=operation,
        )

    async def finalize(
        self,
        operation_id: str,
        *,
        operator_id: int,
        request_id: str,
        audit_sink: OperationalAuditSink,
    ) -> KeyRotationProgress:
        """Verify again and atomically switch the durable ordinary-write primary."""

        async def operation(at: datetime) -> KeyRotationProgress:
            state = await self._repository.get_migration_state()
            if state.rotation_operation_id != operation_id:
                raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
            if state.rotation_phase == "complete":
                return _progress(state)
            target = state.rotation_target_key_id
            if target is None:
                raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
            _, target = self._require_rotation(
                state,
                operation_id=operation_id,
                phases=frozenset({"awaiting_primary_cutover"}),
                required_primary=target,
            )
            if state.rotation_started_at is None:
                raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
            verified_count = await self._verified_inventory_count(
                target=target,
                inventory_at=state.rotation_started_at,
            )
            if verified_count != state.rotation_processed_count:
                raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
            async with self._repository.transaction() as tx:
                current = await tx.lock_migration_state()
                if current.state_revision != state.state_revision:
                    raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
                updated = await tx.compare_and_set_migration_state(
                    expected_revision=current.state_revision,
                    updates={
                        "active_primary_key_id": target,
                        "rotation_phase": "complete",
                        "rotation_verified_count": verified_count,
                        "rotation_completed_at": at,
                    },
                    at=at,
                )
            return _progress(updated)

        return await self._run(
            action="admin_webhook.key_rotation.finalize",
            operation_id=operation_id,
            operator_id=operator_id,
            request_id=request_id,
            audit_sink=audit_sink,
            operation=operation,
        )
