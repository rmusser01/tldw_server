"""Stable, source-scoped identity for email rows in the legacy media store."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.parse import urlsplit


def email_identity_url(
    *, metadata: dict[str, Any], source_url: str, content_hash: str, tenant_id: str
) -> tuple[str, dict[str, Any]]:
    """Return an opaque identity URL and metadata retaining the original source.

    A provider-native ID wins over the RFC header; content is a fallback only
    for messages without either ID. Source and tenant are part of every key.
    Keeping the source in metadata makes reusing a stored URL idempotent too.
    """
    email = metadata.get("email") if isinstance(metadata.get("email"), dict) else {}
    is_gmail = source_url.startswith("gmail://")
    provider = str(metadata.get("email_source_provider") or ("gmail" if is_gmail else "upload"))
    source = str(metadata.get("source_key") or (urlsplit(source_url).netloc if is_gmail else source_url) or "upload")
    provider_id = str(
        email.get("source_message_id") or email.get("id") or metadata.get("source_message_id") or ""
    ).strip()
    message_id = str(email.get("message_id") or metadata.get("message_id") or "").strip()
    kind, identity = (
        ("provider", provider_id) if provider_id else ("message", message_id) if message_id else ("hash", content_hash)
    )
    key = json.dumps([tenant_id, provider, source, kind, identity], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"email://identity/{digest}", {**metadata, "source_key": source, "email_source_provider": provider}
