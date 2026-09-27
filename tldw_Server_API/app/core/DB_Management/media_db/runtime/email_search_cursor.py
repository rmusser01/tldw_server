"""Scoped email keyset positions; cursors never grant access to message rows."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime
from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError


def email_cursor_scope(tenant: str, query: str | None, include_deleted: bool) -> str:
    """Fingerprint the search scope without embedding query text in the token."""
    raw = json.dumps([tenant, str(query or "").strip(), include_deleted]).encode()
    return hashlib.sha256(raw).hexdigest()


def encode_email_cursor(scope: str, as_of: datetime, row: dict[str, Any]) -> str:
    """Encode a versioned, URL-safe position from a returned database row."""
    date = row["internal_date"]
    if isinstance(date, datetime):
        date = date.isoformat()
    payload = [1, scope, as_of.isoformat(), date, int(row["email_message_id"])]
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")


def decode_email_cursor(cursor: str, scope: str) -> tuple[datetime, str | None, int]:
    """Reject malformed/incompatible tokens and return their keyset position."""
    try:
        if not isinstance(cursor, str) or len(cursor) > 4096 or not re.fullmatch(r"[A-Za-z0-9_-]+", cursor):
            raise ValueError("Invalid token encoding")
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw).decode().rstrip("=") != cursor:
            raise ValueError("Noncanonical token encoding")
        payload = json.loads(raw)
        if not isinstance(payload, list) or len(payload) != 5:
            raise ValueError("Invalid token shape")
        version, token_scope, as_of, date, message_id = payload
        if type(version) is not int or version != 1 or token_scope != scope:
            raise ValueError("Incompatible search scope or version")
        if type(message_id) is not int or not 0 < message_id <= 9223372036854775807:
            raise ValueError("Invalid message ID")
        if not isinstance(as_of, str):
            raise ValueError("Invalid reference date")
        reference = datetime.fromisoformat(as_of)
        if reference.utcoffset() is None:
            raise ValueError("Missing reference timezone")
        if date is not None:
            if not isinstance(date, str) or datetime.fromisoformat(date).utcoffset() is None:
                raise ValueError("Invalid message date")
        return reference, date, message_id
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise InputError("Invalid email cursor or incompatible search scope.") from exc
