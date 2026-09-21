"""Read the accepted Media payload before populating a normalized email graph."""

from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db.api import get_document_version, get_media_by_id
from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError


def read_persisted_email_content(
    db: Any,
    media_id: int,
    *,
    tenant_id: str,
) -> tuple[dict[str, Any], str]:
    """Return saved metadata/body even when an incoming overwrite was declined.

    Reading the latest saved version also supports backfilling a missing native
    graph without replacing an existing Media record with the reimport payload.
    """
    media = get_media_by_id(db, media_id)
    if media is None:
        raise InputError("Persisted email Media record is unavailable.")
    version = get_document_version(db, media_id, include_content=False)
    raw_metadata = (version or {}).get("safe_metadata")
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) and raw_metadata else raw_metadata
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise InputError("Persisted email metadata must be an object.")
    metadata = dict(metadata)
    if not isinstance(metadata.get("email"), dict) or not metadata["email"]:
        # Older versions stripped parsed email fields but retained them in the native graph.
        row = db.execute_query(
            "SELECT raw_metadata_json FROM email_messages WHERE media_id = ? AND tenant_id = ? LIMIT 1",
            (media_id, tenant_id),
        ).fetchone()
        if row and row["raw_metadata_json"]:
            legacy_metadata = json.loads(row["raw_metadata_json"])
            if not isinstance(legacy_metadata, dict):
                raise InputError("Persisted email graph metadata must be an object.")
            metadata = {**legacy_metadata, **metadata, "email": legacy_metadata.get("email", {})}
    metadata.setdefault("title", media.get("title"))
    metadata.setdefault("author", media.get("author"))
    metadata.setdefault("source_url", media.get("url"))
    return metadata, str(media.get("content") or "")
