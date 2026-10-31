"""Package-owned email message mutation helpers."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def _normalize_email_label_values(values: list[str] | str | None) -> dict[str, str]:
    if values is None:
        return {}
    raw_values: list[str] = []
    if isinstance(values, str):
        raw_values.extend(part.strip() for part in values.split(","))
    elif isinstance(values, list):
        raw_values.extend(str(part or "").strip() for part in values)

    out: dict[str, str] = {}
    for value in raw_values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key not in out:
            out[key] = text
    return out


def _resolve_email_message_row_for_source_message(
    self,
    conn,
    *,
    tenant_id: str,
    source_id: int,
    source_message_id: str,
) -> dict[str, Any] | None:
    return self._fetchone_with_connection(
        conn,
        (
            "SELECT id, media_id, label_text, raw_metadata_json "
            "FROM email_messages "
            "WHERE tenant_id = ? AND source_id = ? AND source_message_id = ? "
            "LIMIT 1"
        ),
        (tenant_id, int(source_id), source_message_id),
    )


def apply_email_label_delta(
    self,
    *,
    provider: str,
    source_key: str,
    source_message_id: str,
    labels_added: list[str] | str | None = None,
    labels_removed: list[str] | str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    resolved_tenant = self._resolve_email_tenant_id(tenant_id)
    resolved_provider = str(provider or "").strip().lower() or "upload"
    resolved_source_key = str(source_key or "").strip()
    resolved_message_key = str(source_message_id or "").strip()
    if not resolved_source_key:
        raise InputError("source_key is required for email label delta.")  # noqa: TRY003
    if not resolved_message_key:
        raise InputError("source_message_id is required for email label delta.")  # noqa: TRY003

    added_map = _normalize_email_label_values(labels_added)
    removed_map = _normalize_email_label_values(labels_removed)
    if not added_map and not removed_map:
        return {
            "applied": False,
            "reason": "empty_delta",
            "tenant_id": resolved_tenant,
            "provider": resolved_provider,
            "source_key": resolved_source_key,
            "source_message_id": resolved_message_key,
            "labels": [],
        }

    overlap_keys = set(added_map.keys()) & set(removed_map.keys())
    for key in overlap_keys:
        added_map.pop(key, None)
        removed_map.pop(key, None)

    with self.transaction() as conn:
        source_row_id = self._resolve_email_sync_source_row_id(
            conn,
            tenant_id=resolved_tenant,
            provider=resolved_provider,
            source_key=resolved_source_key,
            create_if_missing=False,
        )
        if source_row_id is None:
            return {
                "applied": False,
                "reason": "source_not_found",
                "tenant_id": resolved_tenant,
                "provider": resolved_provider,
                "source_key": resolved_source_key,
                "source_message_id": resolved_message_key,
                "labels": [],
            }

        message_row = _resolve_email_message_row_for_source_message(
            self,
            conn,
            tenant_id=resolved_tenant,
            source_id=source_row_id,
            source_message_id=resolved_message_key,
        )
        if message_row is None:
            return {
                "applied": False,
                "reason": "message_not_found",
                "tenant_id": resolved_tenant,
                "provider": resolved_provider,
                "source_key": resolved_source_key,
                "source_message_id": resolved_message_key,
                "labels": [],
            }

        email_message_id = int(message_row["id"])
        media_id = int(message_row["media_id"])

        removed_count = 0
        for label_key in removed_map:
            label_row = self._fetchone_with_connection(
                conn,
                (
                    "SELECT id FROM email_labels "
                    "WHERE tenant_id = ? AND label_key = ? "
                    "LIMIT 1"
                ),
                (resolved_tenant, label_key),
            )
            if not label_row:
                continue
            delete_cursor = self._execute_with_connection(
                conn,
                (
                    "DELETE FROM email_message_labels "
                    "WHERE email_message_id = ? AND label_id = ?"
                ),
                (email_message_id, int(label_row["id"])),
            )
            removed_count += int(getattr(delete_cursor, "rowcount", 0) or 0)

        added_count = 0
        for label_key, label_name in added_map.items():
            self._execute_with_connection(
                conn,
                (
                    "INSERT INTO email_labels (tenant_id, label_key, label_name) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(tenant_id, label_key) "
                    "DO UPDATE SET label_name = EXCLUDED.label_name, updated_at = CURRENT_TIMESTAMP"
                ),
                (resolved_tenant, label_key, label_name),
            )
            label_row = self._fetchone_with_connection(
                conn,
                (
                    "SELECT id FROM email_labels "
                    "WHERE tenant_id = ? AND label_key = ? "
                    "LIMIT 1"
                ),
                (resolved_tenant, label_key),
            )
            if not label_row:
                continue
            insert_cursor = self._execute_with_connection(
                conn,
                (
                    "INSERT INTO email_message_labels (email_message_id, label_id) "
                    "VALUES (?, ?) ON CONFLICT DO NOTHING"
                ),
                (email_message_id, int(label_row["id"])),
            )
            added_count += int(getattr(insert_cursor, "rowcount", 0) or 0)

        label_rows = self._fetchall_with_connection(
            conn,
            (
                "SELECT el.label_name AS label_name "
                "FROM email_message_labels eml "
                "JOIN email_labels el ON el.id = eml.label_id "
                "WHERE eml.email_message_id = ? AND el.tenant_id = ? "
                "ORDER BY el.label_name ASC"
            ),
            (email_message_id, resolved_tenant),
        )
        final_labels = [
            str(row.get("label_name") or "").strip()
            for row in label_rows
            if str(row.get("label_name") or "").strip()
        ]
        label_text = ", ".join(final_labels) if final_labels else None

        raw_metadata_json = message_row.get("raw_metadata_json")
        metadata_json_out = raw_metadata_json
        if isinstance(raw_metadata_json, str) and raw_metadata_json.strip():
            with suppress(_MEDIA_NONCRITICAL_EXCEPTIONS):
                metadata_obj = json.loads(raw_metadata_json)
                if isinstance(metadata_obj, dict):
                    metadata_obj["labels"] = final_labels
                    email_obj = metadata_obj.get("email")
                    if isinstance(email_obj, dict):
                        email_obj["labels"] = final_labels
                    metadata_json_out = json.dumps(metadata_obj, ensure_ascii=False)

        self._execute_with_connection(
            conn,
            (
                "UPDATE email_messages "
                "SET label_text = ?, raw_metadata_json = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ?"
            ),
            (label_text, metadata_json_out, email_message_id),
        )

        if self.backend_type == BackendType.SQLITE:
            with suppress(_MEDIA_NONCRITICAL_EXCEPTIONS):
                self._execute_with_connection(
                    conn,
                    (
                        "INSERT OR REPLACE INTO email_fts "
                        "(rowid, subject, body_text, from_text, to_text, cc_text, bcc_text, label_text) "
                        "SELECT id, COALESCE(subject, ''), COALESCE(body_text, ''), "
                        "COALESCE(from_text, ''), COALESCE(to_text, ''), "
                        "COALESCE(cc_text, ''), COALESCE(bcc_text, ''), COALESCE(label_text, '') "
                        "FROM email_messages WHERE id = ?"
                    ),
                    (email_message_id,),
                )

        return {
            "applied": bool(added_count or removed_count),
            "reason": "ok",
            "tenant_id": resolved_tenant,
            "provider": resolved_provider,
            "source_key": resolved_source_key,
            "source_message_id": resolved_message_key,
            "source_id": int(source_row_id),
            "email_message_id": email_message_id,
            "media_id": media_id,
            "added_count": int(added_count),
            "removed_count": int(removed_count),
            "labels": final_labels,
        }


def reconcile_email_message_state(
    self,
    *,
    provider: str,
    source_key: str,
    source_message_id: str,
    tenant_id: str | None = None,
    deleted: bool | None = None,
) -> dict[str, Any]:
    resolved_tenant = self._resolve_email_tenant_id(tenant_id)
    resolved_provider = str(provider or "").strip().lower() or "upload"
    resolved_source_key = str(source_key or "").strip()
    resolved_message_key = str(source_message_id or "").strip()
    if not resolved_source_key:
        raise InputError("source_key is required for email state reconciliation.")  # noqa: TRY003
    if not resolved_message_key:
        raise InputError("source_message_id is required for email state reconciliation.")  # noqa: TRY003

    if deleted is None:
        return {
            "applied": False,
            "reason": "no_state_change",
            "tenant_id": resolved_tenant,
            "provider": resolved_provider,
            "source_key": resolved_source_key,
            "source_message_id": resolved_message_key,
        }

    with self.transaction() as conn:
        source_row_id = self._resolve_email_sync_source_row_id(
            conn,
            tenant_id=resolved_tenant,
            provider=resolved_provider,
            source_key=resolved_source_key,
            create_if_missing=False,
        )
        if source_row_id is None:
            return {
                "applied": False,
                "reason": "source_not_found",
                "tenant_id": resolved_tenant,
                "provider": resolved_provider,
                "source_key": resolved_source_key,
                "source_message_id": resolved_message_key,
            }

        message_row = _resolve_email_message_row_for_source_message(
            self,
            conn,
            tenant_id=resolved_tenant,
            source_id=source_row_id,
            source_message_id=resolved_message_key,
        )
        if message_row is None:
            return {
                "applied": False,
                "reason": "message_not_found",
                "tenant_id": resolved_tenant,
                "provider": resolved_provider,
                "source_key": resolved_source_key,
                "source_message_id": resolved_message_key,
                "source_id": int(source_row_id),
            }

        media_id = int(message_row["media_id"])
        media_row = self._fetchone_with_connection(
            conn,
            "SELECT deleted FROM Media WHERE id = ? AND system_operation_id IS NULL LIMIT 1",
            (media_id,),
        )
        media_deleted = bool((media_row or {}).get("deleted"))

    if bool(deleted):
        if media_deleted:
            return {
                "applied": False,
                "reason": "already_deleted",
                "tenant_id": resolved_tenant,
                "provider": resolved_provider,
                "source_key": resolved_source_key,
                "source_message_id": resolved_message_key,
                "media_id": media_id,
            }
        removed = bool(self.soft_delete_media(media_id, cascade=True))
        return {
            "applied": removed,
            "reason": "deleted" if removed else "delete_failed",
            "tenant_id": resolved_tenant,
            "provider": resolved_provider,
            "source_key": resolved_source_key,
            "source_message_id": resolved_message_key,
            "media_id": media_id,
        }

    return {
        "applied": False,
        "reason": "unsupported_state",
        "tenant_id": resolved_tenant,
        "provider": resolved_provider,
        "source_key": resolved_source_key,
        "source_message_id": resolved_message_key,
        "media_id": media_id,
    }
