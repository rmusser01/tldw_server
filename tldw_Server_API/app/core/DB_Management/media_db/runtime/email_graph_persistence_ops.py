"""Package-owned email graph persistence helpers."""

from __future__ import annotations

import json
from contextlib import suppress
from email.utils import getaddresses
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def _resolve_email_tenant_id(self, tenant_id: str | None = None) -> str:
    """Resolve tenant scope for email-native tables."""

    explicit = str(tenant_id or "").strip()
    if explicit:
        return explicit

    scope = get_scope()
    if scope is not None:
        if scope.effective_org_id is not None:
            return f"org:{int(scope.effective_org_id)}"
        if scope.user_id is not None:
            return f"user:{int(scope.user_id)}"
    return str(self.client_id)


def upsert_email_message_graph(
    self,
    *,
    media_id: int,
    metadata: dict[str, Any] | None = None,
    body_text: str | None = None,
    tenant_id: str | None = None,
    provider: str = "upload",
    source_key: str | None = None,
    source_message_id: str | None = None,
    labels: list[str] | str | None = None,
) -> dict[str, Any]:
    """
    Upsert a normalized email message graph (message, participants, labels, attachments).

    This is the Stage-1 persistence bridge used by archive/upload ingestion.
    """

    media_id_int = int(media_id)
    if media_id_int <= 0:
        raise InputError("media_id must be a positive integer")  # noqa: TRY003

    metadata_map = metadata if isinstance(metadata, dict) else {}
    email_meta = metadata_map.get("email") if isinstance(metadata_map.get("email"), dict) else {}

    resolved_tenant = self._resolve_email_tenant_id(tenant_id)
    resolved_provider = str(provider or "upload").strip() or "upload"
    resolved_source_key = str(
        source_key
        or metadata_map.get("source_key")
        or metadata_map.get("source")
        or metadata_map.get("filename")
        or "upload"
    ).strip() or "upload"
    resolved_source_message_id = str(
        source_message_id
        or email_meta.get("source_message_id")
        or email_meta.get("id")
        or metadata_map.get("source_message_id")
        or ""
    ).strip() or None
    resolved_message_id = str(
        email_meta.get("message_id") or metadata_map.get("message_id") or ""
    ).strip() or None
    resolved_subject = str(
        email_meta.get("subject") or metadata_map.get("title") or ""
    ).strip() or None
    resolved_from = str(email_meta.get("from") or "").strip() or None
    resolved_to = str(email_meta.get("to") or "").strip() or None
    resolved_cc = str(email_meta.get("cc") or "").strip() or None
    resolved_bcc = str(email_meta.get("bcc") or "").strip() or None
    resolved_internal_date = self._parse_email_internal_date(
        email_meta.get("internal_date") or metadata_map.get("internal_date")
        or email_meta.get("date") or metadata_map.get("date")
    )

    attachments_raw = email_meta.get("attachments")
    attachments: list[dict[str, Any]] = (
        [attachment for attachment in attachments_raw if isinstance(attachment, dict)]
        if isinstance(attachments_raw, list)
        else []
    )
    normalized_labels = self._collect_email_labels(metadata_map, labels)
    label_text = ", ".join(normalized_labels) if normalized_labels else None
    has_attachments = bool(attachments)
    raw_metadata_json = json.dumps(metadata_map, ensure_ascii=False) if metadata_map else None

    with self.transaction() as conn:
        self._execute_with_connection(
            conn,
            (
                "INSERT INTO email_sources "
                "(tenant_id, provider, source_key, display_name, status) "
                "VALUES (?, ?, ?, ?, 'active') "
                "ON CONFLICT(tenant_id, provider, source_key) "
                "DO UPDATE SET "
                "display_name = COALESCE(EXCLUDED.display_name, email_sources.display_name), "
                "updated_at = CURRENT_TIMESTAMP"
            ),
            (
                resolved_tenant,
                resolved_provider,
                resolved_source_key,
                resolved_source_key,
            ),
        )
        source_row = self._fetchone_with_connection(
            conn,
            (
                "SELECT id FROM email_sources "
                "WHERE tenant_id = ? AND provider = ? AND source_key = ? "
                "LIMIT 1"
            ),
            (resolved_tenant, resolved_provider, resolved_source_key),
        )
        if not source_row:
            raise DatabaseError("Failed to resolve email source after upsert.")  # noqa: TRY003
        source_id = int(source_row["id"])

        existing_message: dict[str, Any] | None = None
        match_strategy = "new"

        if resolved_source_message_id:
            existing_message = self._fetchone_with_connection(
                conn,
                (
                    "SELECT id FROM email_messages "
                    "WHERE tenant_id = ? AND source_id = ? AND source_message_id = ? "
                    "LIMIT 1"
                ),
                (resolved_tenant, source_id, resolved_source_message_id),
            )
            if existing_message:
                match_strategy = "source_message_id"

        if existing_message is None and resolved_message_id:
            existing_message = self._fetchone_with_connection(
                conn,
                (
                    "SELECT id FROM email_messages "
                    "WHERE tenant_id = ? AND source_id = ? AND message_id = ? "
                    "LIMIT 1"
                ),
                (resolved_tenant, source_id, resolved_message_id),
            )
            if existing_message:
                match_strategy = "message_id"

        if existing_message is None:
            existing_message = self._fetchone_with_connection(
                conn,
                "SELECT id, tenant_id, source_id, source_message_id, message_id "
                "FROM email_messages WHERE media_id = ? LIMIT 1",
                (media_id_int,),
            )
            if existing_message:
                old_provider_id = existing_message["source_message_id"]
                old_message_id = existing_message["message_id"]
                if (
                    existing_message["tenant_id"] != resolved_tenant
                    or int(existing_message["source_id"]) != source_id
                    or (old_provider_id and resolved_source_message_id and old_provider_id != resolved_source_message_id)
                    or (old_message_id and resolved_message_id and old_message_id != resolved_message_id)
                ):
                    raise InputError("Media row already belongs to a different email identity")  # noqa: TRY003
                match_strategy = "media_id"

        if existing_message is not None:
            email_message_id = int(existing_message["id"])
            self._execute_with_connection(
                conn,
                (
                    "UPDATE email_messages SET "
                    "media_id = ?, "
                    "source_id = ?, "
                    "source_message_id = ?, "
                    "message_id = ?, "
                    "subject = ?, "
                    "body_text = ?, "
                    "internal_date = ?, "
                    "from_text = ?, "
                    "to_text = ?, "
                    "cc_text = ?, "
                    "bcc_text = ?, "
                    "label_text = ?, "
                    "has_attachments = ?, "
                    "raw_metadata_json = ?, "
                    "updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = ?"
                ),
                (
                    media_id_int,
                    source_id,
                    resolved_source_message_id,
                    resolved_message_id,
                    resolved_subject,
                    str(body_text or ""),
                    resolved_internal_date,
                    resolved_from,
                    resolved_to,
                    resolved_cc,
                    resolved_bcc,
                    label_text,
                    bool(has_attachments),
                    raw_metadata_json,
                    email_message_id,
                ),
            )
        else:
            self._execute_with_connection(
                conn,
                (
                    "INSERT INTO email_messages ("
                    "tenant_id, media_id, source_id, source_message_id, message_id, "
                    "subject, body_text, internal_date, from_text, to_text, cc_text, bcc_text, "
                    "label_text, has_attachments, raw_metadata_json"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    resolved_tenant,
                    media_id_int,
                    source_id,
                    resolved_source_message_id,
                    resolved_message_id,
                    resolved_subject,
                    str(body_text or ""),
                    resolved_internal_date,
                    resolved_from,
                    resolved_to,
                    resolved_cc,
                    resolved_bcc,
                    label_text,
                    bool(has_attachments),
                    raw_metadata_json,
                ),
            )
            inserted_message = self._fetchone_with_connection(
                conn,
                (
                    "SELECT id FROM email_messages "
                    "WHERE tenant_id = ? AND media_id = ? LIMIT 1"
                ),
                (resolved_tenant, media_id_int),
            )
            if not inserted_message and resolved_source_message_id:
                inserted_message = self._fetchone_with_connection(
                    conn,
                    (
                        "SELECT id FROM email_messages "
                        "WHERE tenant_id = ? AND source_id = ? AND source_message_id = ? LIMIT 1"
                    ),
                    (resolved_tenant, source_id, resolved_source_message_id),
                )
            if not inserted_message and resolved_message_id:
                inserted_message = self._fetchone_with_connection(
                    conn,
                    (
                        "SELECT id FROM email_messages "
                        "WHERE tenant_id = ? AND source_id = ? AND message_id = ? LIMIT 1"
                    ),
                    (resolved_tenant, source_id, resolved_message_id),
                )
            if not inserted_message:
                raise DatabaseError("Failed to resolve email message row after insert.")  # noqa: TRY003
            email_message_id = int(inserted_message["id"])

        self._execute_with_connection(
            conn,
            "DELETE FROM email_message_participants WHERE email_message_id = ?",
            (email_message_id,),
        )
        self._execute_with_connection(
            conn,
            "DELETE FROM email_message_labels WHERE email_message_id = ?",
            (email_message_id,),
        )
        self._execute_with_connection(
            conn,
            "DELETE FROM email_attachments WHERE email_message_id = ?",
            (email_message_id,),
        )

        for role_name, value in (
            ("from", resolved_from),
            ("to", resolved_to),
            ("cc", resolved_cc),
            ("bcc", resolved_bcc),
        ):
            role_text = str(value or "").strip()
            if not role_text:
                continue
            for display_name, email_addr in getaddresses([role_text]):
                normalized_addr = self._normalize_email_address(email_addr)
                if not normalized_addr:
                    continue
                display = str(display_name or "").strip() or None
                self._execute_with_connection(
                    conn,
                    (
                        "INSERT INTO email_participants (tenant_id, email_normalized, display_name) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT(tenant_id, email_normalized) "
                        "DO UPDATE SET display_name = COALESCE(EXCLUDED.display_name, email_participants.display_name)"
                    ),
                    (resolved_tenant, normalized_addr, display),
                )
                participant_row = self._fetchone_with_connection(
                    conn,
                    (
                        "SELECT id FROM email_participants "
                        "WHERE tenant_id = ? AND email_normalized = ? LIMIT 1"
                    ),
                    (resolved_tenant, normalized_addr),
                )
                if participant_row:
                    self._execute_with_connection(
                        conn,
                        (
                            "INSERT INTO email_message_participants "
                            "(email_message_id, participant_id, role) "
                            "VALUES (?, ?, ?) ON CONFLICT DO NOTHING"
                        ),
                        (
                            email_message_id,
                            int(participant_row["id"]),
                            role_name,
                        ),
                    )

        for label_name in normalized_labels:
            label_key = str(label_name).strip().lower()
            if not label_key:
                continue
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
                    "WHERE tenant_id = ? AND label_key = ? LIMIT 1"
                ),
                (resolved_tenant, label_key),
            )
            if label_row:
                self._execute_with_connection(
                    conn,
                    (
                        "INSERT INTO email_message_labels (email_message_id, label_id) "
                        "VALUES (?, ?) ON CONFLICT DO NOTHING"
                    ),
                    (email_message_id, int(label_row["id"])),
                )

        for attachment in attachments:
            filename = str(
                attachment.get("filename") or attachment.get("name") or ""
            ).strip() or None
            content_type = str(attachment.get("content_type") or "").strip() or None
            size_bytes_raw = attachment.get("size_bytes")
            if size_bytes_raw is None:
                size_bytes_raw = attachment.get("size")
            with suppress(_MEDIA_NONCRITICAL_EXCEPTIONS):
                size_bytes_raw = int(size_bytes_raw) if size_bytes_raw is not None else None
            if isinstance(size_bytes_raw, bool):
                size_bytes_raw = None
            size_bytes = size_bytes_raw if isinstance(size_bytes_raw, int) else None
            content_id = str(attachment.get("content_id") or "").strip() or None
            disposition = str(attachment.get("disposition") or "").strip() or None
            extracted_text_available = bool(
                attachment.get("extracted_text_available")
                or attachment.get("text_extracted")
            )
            self._execute_with_connection(
                conn,
                (
                    "INSERT INTO email_attachments ("
                    "email_message_id, filename, content_type, size_bytes, "
                    "content_id, disposition, extracted_text_available"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    email_message_id,
                    filename,
                    content_type,
                    size_bytes,
                    content_id,
                    disposition,
                    bool(extracted_text_available),
                ),
            )

        if self.backend_type == BackendType.SQLITE:
            with suppress(_MEDIA_NONCRITICAL_EXCEPTIONS):
                self._execute_with_connection(
                    conn,
                    (
                        "INSERT OR REPLACE INTO email_fts "
                        "(rowid, subject, body_text, from_text, to_text, cc_text, bcc_text, label_text) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    (
                        email_message_id,
                        resolved_subject or "",
                        str(body_text or ""),
                        resolved_from or "",
                        resolved_to or "",
                        resolved_cc or "",
                        resolved_bcc or "",
                        label_text or "",
                    ),
                )

        return {
            "email_message_id": email_message_id,
            "source_id": source_id,
            "tenant_id": resolved_tenant,
            "match_strategy": match_strategy,
        }
