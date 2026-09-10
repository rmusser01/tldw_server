"""Package-owned email backfill coordinator helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def run_email_legacy_backfill_batch(
    self,
    *,
    batch_size: int = 500,
    tenant_id: str | None = None,
    backfill_key: str = "legacy_media_email",
) -> dict[str, Any]:
    """
    Backfill one batch of legacy email Media rows into normalized email tables.

    Progress is checkpointed in `email_backfill_state` by `(tenant_id, backfill_key)`
    so repeated calls resume from the prior `last_media_id`.
    """

    try:
        batch_size_int = int(batch_size)
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        raise InputError("batch_size must be an integer.") from exc  # noqa: TRY003
    if batch_size_int <= 0:
        raise InputError("batch_size must be greater than zero.")  # noqa: TRY003

    resolved_tenant = self._resolve_email_tenant_id(tenant_id)
    resolved_key = self._normalize_email_backfill_key(backfill_key)
    now_iso = datetime.now(timezone.utc).isoformat()

    with self.transaction() as conn:
        self._ensure_email_backfill_state_row(
            conn,
            tenant_id=resolved_tenant,
            backfill_key=resolved_key,
        )
        self._execute_with_connection(
            conn,
            (
                "UPDATE email_backfill_state SET "
                "status = 'running', "
                "started_at = COALESCE(started_at, ?), "
                "completed_at = NULL, "
                "updated_at = CURRENT_TIMESTAMP "
                "WHERE tenant_id = ? AND backfill_key = ?"
            ),
            (now_iso, resolved_tenant, resolved_key),
        )
        state_before = self._fetch_email_backfill_state_row(
            conn,
            tenant_id=resolved_tenant,
            backfill_key=resolved_key,
        )
        if state_before is None:
            raise DatabaseError("Failed to initialize email backfill state.")  # noqa: TRY003
        cursor_start = int(state_before.get("last_media_id") or 0)
        rows = self._fetchall_with_connection(
            conn,
            (
                "SELECT m.id, m.url, m.title, m.content, m.author, m.ingestion_date, "
                "(SELECT dv.safe_metadata "
                " FROM DocumentVersions dv "
                " WHERE dv.media_id = m.id AND dv.deleted = 0 "
                " ORDER BY dv.version_number DESC, dv.id DESC "
                " LIMIT 1) AS safe_metadata_json, "
                "EXISTS(SELECT 1 FROM email_messages em WHERE em.media_id = m.id) AS already_backfilled "
                "FROM Media m "
                "WHERE m.deleted = 0 "
                "AND m.system_operation_id IS NULL "
                "AND lower(COALESCE(m.type, '')) = 'email' "
                "AND m.id > ? "
                "ORDER BY m.id ASC "
                "LIMIT ?"
            ),
            (cursor_start, batch_size_int),
        )

    if not rows:
        with self.transaction() as conn:
            state = self._fetch_email_backfill_state_row(
                conn,
                tenant_id=resolved_tenant,
                backfill_key=resolved_key,
            )
            if state is None:
                raise DatabaseError("Failed to load email backfill state.")  # noqa: TRY003
            final_status = (
                "completed_with_errors"
                if int(state.get("failed_count") or 0) > 0
                else "completed"
            )
            self._execute_with_connection(
                conn,
                (
                    "UPDATE email_backfill_state SET "
                    "status = ?, "
                    "completed_at = COALESCE(completed_at, ?), "
                    "updated_at = CURRENT_TIMESTAMP "
                    "WHERE tenant_id = ? AND backfill_key = ?"
                ),
                (final_status, now_iso, resolved_tenant, resolved_key),
            )
            state_after = self._fetch_email_backfill_state_row(
                conn,
                tenant_id=resolved_tenant,
                backfill_key=resolved_key,
            )
        return {
            "tenant_id": resolved_tenant,
            "backfill_key": resolved_key,
            "batch_size": batch_size_int,
            "cursor_start": cursor_start,
            "cursor_end": cursor_start,
            "scanned": 0,
            "ingested": 0,
            "skipped": 0,
            "failed": 0,
            "completed": True,
            "status": final_status,
            "state": state_after,
        }

    scanned = 0
    ingested = 0
    skipped = 0
    failed = 0
    cursor_end = cursor_start

    for row in rows:
        media_id = int(row.get("id") or 0)
        if media_id <= 0:
            continue
        cursor_end = media_id
        scanned += 1

        delta_success = 0
        delta_skipped = 0
        delta_failed = 0
        row_error: str | None = None

        already_backfilled = bool(row.get("already_backfilled"))
        if already_backfilled:
            skipped += 1
            delta_skipped = 1
        else:
            try:
                metadata_map = self._parse_email_backfill_safe_metadata(
                    row.get("safe_metadata_json")
                )
                email_meta = metadata_map.get("email")
                if not isinstance(email_meta, dict):
                    email_meta = {}
                    metadata_map["email"] = email_meta

                subject_fallback = str(row.get("title") or "").strip()
                from_fallback = str(row.get("author") or "").strip()
                date_fallback = str(row.get("ingestion_date") or "").strip()
                if subject_fallback and not str(email_meta.get("subject") or "").strip():
                    email_meta["subject"] = subject_fallback
                if from_fallback and not str(email_meta.get("from") or "").strip():
                    email_meta["from"] = from_fallback
                if date_fallback and not str(email_meta.get("date") or "").strip():
                    email_meta["date"] = date_fallback

                provider, source_key, source_message_id = self._derive_email_backfill_source_fields(
                    metadata_map=metadata_map,
                    media_url=row.get("url"),
                    tenant_id=resolved_tenant,
                )
                if source_message_id and not str(email_meta.get("source_message_id") or "").strip():
                    email_meta["source_message_id"] = source_message_id

                body_text = str(row.get("content") or "")
                labels = self._collect_email_labels(metadata_map)

                self.upsert_email_message_graph(
                    media_id=media_id,
                    metadata=metadata_map,
                    body_text=body_text,
                    tenant_id=resolved_tenant,
                    provider=provider,
                    source_key=source_key,
                    source_message_id=source_message_id,
                    labels=labels,
                )
                ingested += 1
                delta_success = 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                delta_failed = 1
                row_error = f"{type(exc).__name__}: {exc}"

        self._update_email_backfill_progress(
            tenant_id=resolved_tenant,
            backfill_key=resolved_key,
            last_media_id=media_id,
            delta_processed=1,
            delta_success=delta_success,
            delta_skipped=delta_skipped,
            delta_failed=delta_failed,
            status="running",
            last_error=row_error,
        )

    with self.transaction() as conn:
        remaining = self._fetchone_with_connection(
            conn,
            (
                "SELECT id FROM Media "
                "WHERE deleted = 0 "
                "AND system_operation_id IS NULL "
                "AND lower(COALESCE(type, '')) = 'email' "
                "AND id > ? "
                "LIMIT 1"
            ),
            (cursor_end,),
        )
        has_more = bool(remaining)
        state_after = self._fetch_email_backfill_state_row(
            conn,
            tenant_id=resolved_tenant,
            backfill_key=resolved_key,
        )
        if state_after is None:
            raise DatabaseError("Failed to load email backfill state after batch.")  # noqa: TRY003
        final_status = str(state_after.get("status") or "running")
        if not has_more:
            final_status = (
                "completed_with_errors"
                if int(state_after.get("failed_count") or 0) > 0
                else "completed"
            )
            self._execute_with_connection(
                conn,
                (
                    "UPDATE email_backfill_state SET "
                    "status = ?, "
                    "completed_at = COALESCE(completed_at, ?), "
                    "updated_at = CURRENT_TIMESTAMP "
                    "WHERE tenant_id = ? AND backfill_key = ?"
                ),
                (final_status, now_iso, resolved_tenant, resolved_key),
            )
            state_after = self._fetch_email_backfill_state_row(
                conn,
                tenant_id=resolved_tenant,
                backfill_key=resolved_key,
            )
            if state_after is None:
                raise DatabaseError("Failed to persist final email backfill state.")  # noqa: TRY003

    return {
        "tenant_id": resolved_tenant,
        "backfill_key": resolved_key,
        "batch_size": batch_size_int,
        "cursor_start": cursor_start,
        "cursor_end": cursor_end,
        "scanned": scanned,
        "ingested": ingested,
        "skipped": skipped,
        "failed": failed,
        "completed": not has_more,
        "status": final_status,
        "state": state_after,
    }


def run_email_legacy_backfill_worker(
    self,
    *,
    batch_size: int = 500,
    tenant_id: str | None = None,
    backfill_key: str = "legacy_media_email",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """
    Worker-style loop for the legacy email backfill.

    Runs `run_email_legacy_backfill_batch` repeatedly until completion or
    until `max_batches` is reached.
    """

    max_batches_int: int | None = None
    if max_batches is not None:
        try:
            max_batches_int = int(max_batches)
        except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
            raise InputError("max_batches must be an integer or None.") from exc  # noqa: TRY003
        if max_batches_int <= 0:
            raise InputError("max_batches must be greater than zero when provided.")  # noqa: TRY003

    resolved_tenant = self._resolve_email_tenant_id(tenant_id)
    resolved_key = self._normalize_email_backfill_key(backfill_key)

    batches_run = 0
    scanned_total = 0
    ingested_total = 0
    skipped_total = 0
    failed_total = 0
    completed = False
    stop_reason = "max_batches"
    last_batch: dict[str, Any] | None = None

    while True:
        if max_batches_int is not None and batches_run >= max_batches_int:
            break

        batch_result = self.run_email_legacy_backfill_batch(
            batch_size=batch_size,
            tenant_id=resolved_tenant,
            backfill_key=resolved_key,
        )
        last_batch = batch_result
        batches_run += 1
        scanned_total += int(batch_result.get("scanned") or 0)
        ingested_total += int(batch_result.get("ingested") or 0)
        skipped_total += int(batch_result.get("skipped") or 0)
        failed_total += int(batch_result.get("failed") or 0)

        if bool(batch_result.get("completed")):
            completed = True
            stop_reason = "completed"
            break

        # Safety valve: avoid infinite loops if a batch made no forward progress.
        if int(batch_result.get("scanned") or 0) <= 0:
            completed = True
            stop_reason = "no_progress"
            break

    final_state = self.get_email_legacy_backfill_state(
        tenant_id=resolved_tenant,
        backfill_key=resolved_key,
    )
    return {
        "tenant_id": resolved_tenant,
        "backfill_key": resolved_key,
        "batch_size": int(batch_size),
        "max_batches": max_batches_int,
        "batches_run": batches_run,
        "scanned": scanned_total,
        "ingested": ingested_total,
        "skipped": skipped_total,
        "failed": failed_total,
        "completed": completed,
        "stop_reason": stop_reason,
        "last_batch": last_batch,
        "state": final_state,
    }


__all__ = [
    "run_email_legacy_backfill_batch",
    "run_email_legacy_backfill_worker",
]
