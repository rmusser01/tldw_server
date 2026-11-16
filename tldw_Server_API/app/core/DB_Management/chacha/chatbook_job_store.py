"""Persist Chatbooks import snapshots without reviving cancelled jobs."""

import json
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType


def _merge_receipt_metadata(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Retain current metadata, admitting only public result fields from a stale worker."""
    merged = dict(current)
    for key in ("imported_items", "inventory_summary", "skipped_non_restorable"):
        value = incoming.get(key)
        if isinstance(value, dict):
            previous = merged.get(key)
            merged[key] = {**(previous if isinstance(previous, dict) else {}), **value}
    if isinstance(incoming.get("id_maps"), dict):
        maps = dict(merged.get("id_maps") or {})
        for kind, additions in incoming["id_maps"].items():
            if not isinstance(additions, dict):
                continue
            aliases = dict(maps.get(kind) or {})
            for alias, destination in additions.items():
                aliases[alias] = None if alias in aliases and aliases[alias] != destination else destination
            maps[kind] = aliases
        merged["id_maps"] = maps
    return merged


def _retain_cancelled_receipt(db: Any, values: tuple[Any, ...]) -> None:
    """Merge a late terminal result under the adapter's SQLite/PG transaction lock."""
    with db.transaction(preserve_existing=True):
        query = "SELECT warnings, metadata FROM import_jobs WHERE job_id = ? AND user_id = ? AND status = ?"
        if db.backend_type == BackendType.POSTGRESQL:
            query += " FOR UPDATE"
        row = db.execute_query(query, (values[0], values[1], "cancelled")).fetchone()
        if row is None:
            return
        current = dict(row)
        warnings = json.loads(current["warnings"] or "[]")
        warnings += [item for item in json.loads(values[15]) if item not in warnings]
        metadata = _merge_receipt_metadata(json.loads(current["metadata"] or "{}"), json.loads(values[16]))
        db.execute_query(
            "UPDATE import_jobs SET total_items = ?, processed_items = ?, successful_items = ?, "
            "failed_items = ?, skipped_items = ?, warnings = ?, metadata = ? "
            "WHERE job_id = ? AND user_id = ? AND status = ?",
            (*values[9:14], json.dumps(warnings), json.dumps(metadata, ensure_ascii=False), values[0], values[1], "cancelled"),
            commit=False,
            log_params=False,
        )


def save_import_job(db: Any, values: tuple[Any, ...], *, commit: bool = True) -> None:
    """Write the serialized import row through the existing database adapter.

    Cancellation is checked in the same statement as the update. A late terminal
    result can retain public receipts without replacing the cancelled lifecycle.
    """
    cursor = db.execute_query("""
        INSERT INTO import_jobs (
            job_id, user_id, status, chatbook_path,
            created_at, started_at, completed_at, error_message,
            progress_percentage, total_items, processed_items,
            successful_items, failed_items, skipped_items,
            conflicts, warnings, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(job_id) DO UPDATE SET
            user_id = excluded.user_id,
            status = excluded.status,
            chatbook_path = excluded.chatbook_path,
            created_at = excluded.created_at,
            started_at = excluded.started_at,
            completed_at = excluded.completed_at,
            error_message = excluded.error_message,
            progress_percentage = excluded.progress_percentage,
            total_items = excluded.total_items,
            processed_items = excluded.processed_items,
            successful_items = excluded.successful_items,
            failed_items = excluded.failed_items,
            skipped_items = excluded.skipped_items,
            conflicts = excluded.conflicts,
            warnings = excluded.warnings,
            metadata = excluded.metadata
        WHERE import_jobs.status != 'cancelled' OR excluded.status = 'cancelled'
    """, values, commit=commit, log_params=False)
    if values[2] in {"failed", "completed"} and getattr(cursor, "rowcount", None) == 0:
        metadata = json.loads(values[16])
        if any(isinstance(metadata.get(key), dict) for key in ("imported_items", "inventory_summary", "skipped_non_restorable", "id_maps")):
            _retain_cancelled_receipt(db, values)


def fail_stale_import_job(
    db: Any, *, job_id: str, user_id: str, selected_status: str, cutoff: str,
) -> None:
    """Fail a cleanup candidate only while its owned stale selection still holds."""
    db.execute_query(
        "UPDATE import_jobs SET status = ?, error_message = ? "
        "WHERE job_id = ? AND user_id = ? AND status = ? AND created_at < ?",
        ("failed", "Marked failed by orphan cleanup (stale job)", job_id, user_id, selected_status, cutoff),
        commit=True,
    )
