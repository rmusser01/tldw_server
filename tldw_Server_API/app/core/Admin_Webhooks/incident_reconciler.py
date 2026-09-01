"""Crash-convergent persistence for pending incident webhook markers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from .config import AdminWebhookSettings
from .crypto import WebhookKeyError, WebhookKeyRingLoadResult
from .domain import (
    EventSourceKind,
    PendingIncidentWebhookMarker,
    WebhookError,
    WebhookErrorCode,
)
from .events import parse_canonical_event_body
from .producer import AdminWebhookEventProducer, ProductionEventPreparation

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Admin_Webhooks.observability import (
        AdminWebhookMetrics,
    )
    from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
        AdminWebhookRepository,
    )


class IncidentReconcileCrashPoint(str, Enum):
    """Deterministic boundaries used to prove restart convergence."""

    BEFORE_DB_TRANSACTION = "before_db_transaction"
    AFTER_EVENT_INSERT = "after_event_insert"
    AFTER_DB_COMMIT_BEFORE_REMOVE = "after_db_commit_before_remove"
    AFTER_IN_MEMORY_REMOVE_BEFORE_SAVE = "after_in_memory_remove_before_save"
    AFTER_REMOVE = "after_remove"


class PendingIncidentEventReconciler:
    """Move protected incident markers into canonical event storage exactly once."""

    def __init__(
        self,
        *,
        repository: AdminWebhookRepository,
        key_ring_result: WebhookKeyRingLoadResult,
        settings: AdminWebhookSettings,
        delivery_id_factory: Callable[[], str],
        store_path: Path | None = None,
        crash_injector: Callable[[IncidentReconcileCrashPoint], None] | None = None,
        metrics: AdminWebhookMetrics | None = None,
    ) -> None:
        if not callable(delivery_id_factory):
            raise TypeError("delivery ID factory is required")
        if crash_injector is not None and not callable(crash_injector):
            raise TypeError("crash injector must be callable")
        self._repository = repository
        self._key_ring_result = key_ring_result
        self._settings = settings
        self._store_path = store_path
        self._crash_injector = crash_injector
        self._metrics = metrics
        self._producer = AdminWebhookEventProducer(
            repository=repository,
            settings=settings,
            key_ring_result=key_ring_result,
            event_id_factory=lambda: str(uuid4()),
            delivery_id_factory=delivery_id_factory,
            clock=lambda: _unreachable_clock(),
        )

    def _crash(self, point: IncidentReconcileCrashPoint) -> None:
        injector = self._crash_injector
        if injector is not None:
            injector(point)

    def _active_store_path(self) -> Path:
        from tldw_Server_API.app.services import admin_system_ops_service as system_ops

        return self._store_path or system_ops._STORE_PATH

    def _read_page_sync(self, limit: int) -> tuple[PendingIncidentWebhookMarker, ...]:
        from tldw_Server_API.app.services import admin_system_ops_service as system_ops

        store_path = self._active_store_path()
        with system_ops._STORE_LOCK, system_ops._store_file_lock(store_path=store_path):
            if not store_path.exists():
                return ()
            store = system_ops._load_store_strict(store_path)
            markers = system_ops._pending_incident_markers(store)
        return tuple(sorted(markers, key=lambda marker: (marker.created_at, marker.event_id))[:limit])

    def _remove_exact_sync(self, expected: PendingIncidentWebhookMarker) -> None:
        from tldw_Server_API.app.services import admin_system_ops_service as system_ops

        store_path = self._active_store_path()
        expected_record = expected.to_store_record()
        with system_ops._STORE_LOCK, system_ops._store_file_lock(store_path=store_path):
            if not store_path.exists():
                return
            store = system_ops._load_store_strict(store_path)
            markers = system_ops._pending_incident_markers(store)
            records = store.get("webhook_pending_events")
            if not isinstance(records, list):
                raise ValueError("pending incident marker collection is invalid")
            for index, marker in enumerate(markers):
                if marker.event_id != expected.event_id:
                    continue
                if records[index] != expected_record:
                    raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
                del records[index]
                self._crash(IncidentReconcileCrashPoint.AFTER_IN_MEMORY_REMOVE_BEFORE_SAVE)
                system_ops._atomic_write_store(store_path, store)
                break
            else:
                return
        self._crash(IncidentReconcileCrashPoint.AFTER_REMOVE)

    def _decode_marker(self, marker: PendingIncidentWebhookMarker) -> dict[str, object]:
        ring = self._key_ring_result.ring
        if ring is None:
            raise WebhookError(WebhookErrorCode.KEY_UNAVAILABLE)
        try:
            plaintext = ring.decrypt_bytes(
                purpose=marker.envelope_purpose,
                identity=marker.envelope_identity,
                protected=marker.body,
            )
            if len(plaintext) != marker.body_size_bytes:
                raise ValueError("pending marker body size is invalid")
            return parse_canonical_event_body(
                plaintext,
                event_id=marker.event_id,
                event_type=marker.event_type,
                api_version=marker.api_version,
                created_at=marker.created_at,
            )
        except WebhookKeyError:
            raise WebhookError(WebhookErrorCode.KEY_UNAVAILABLE) from None
        except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError):
            raise WebhookError(WebhookErrorCode.VALIDATION_FAILED) from None

    async def _capture_marker(self, marker: PendingIncidentWebhookMarker) -> None:
        data = self._decode_marker(marker)
        try:
            source_kind = EventSourceKind(marker.source_kind)
            preparation = ProductionEventPreparation(
                event_id=marker.event_id,
                created_at=marker.created_at,
                source_component=marker.source_component,
                source_request_id=marker.source_request_id,
            )
        except (TypeError, ValueError):
            raise WebhookError(WebhookErrorCode.VALIDATION_FAILED) from None
        self._crash(IncidentReconcileCrashPoint.BEFORE_DB_TRANSACTION)
        async with self._repository.transaction() as tx:
            result = await self._producer.capture_in_transaction(
                preparation,
                tx=tx,
                event_type=marker.event_type,
                source_kind=source_kind,
                aggregate_type=marker.aggregate_type,
                aggregate_id=marker.aggregate_id,
                aggregate_version=marker.aggregate_version,
                source_command_id=marker.source_command_id,
                data=data,
            )
            self._crash(IncidentReconcileCrashPoint.AFTER_EVENT_INSERT)
        if result.inserted and self._metrics is not None:
            self._metrics.events_committed(
                event_type=marker.event_type,
                fanout_count=len(result.deliveries),
            )
        self._crash(IncidentReconcileCrashPoint.AFTER_DB_COMMIT_BEFORE_REMOVE)

    async def reconcile_once(self, *, limit: int = 100) -> int:
        """Reconcile one deterministic bounded marker page."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        markers = await asyncio.to_thread(self._read_page_sync, limit)
        reconciled = 0
        for marker in markers:
            await self._capture_marker(marker)
            await asyncio.to_thread(self._remove_exact_sync, marker)
            reconciled += 1
        return reconciled


def _unreachable_clock() -> datetime:
    raise RuntimeError("incident reconciler does not allocate event timestamps")


__all__ = [
    "IncidentReconcileCrashPoint",
    "PendingIncidentEventReconciler",
]
