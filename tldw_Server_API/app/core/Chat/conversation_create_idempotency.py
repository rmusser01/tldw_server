"""Client-supplied conversation ids for ``POST /chats/`` (D7 P3).

A client may name a new conversation with its own UUID, so a retried create
cannot leave a second (or empty) chat behind. The id is the conversation's
primary key, so the database keeps at most one row per id even when two
requests race. The create request's fingerprint is stored on that row by the
same INSERT, and decides what a later create with the same id means:

* same owner, same request, live chat: replay the existing chat;
* same owner, same request, chat in trash: gone (the id stays taken);
* anything else (another owner, a different request, or a chat that was not
  created with a client id): conflict, without revealing the existing chat.

"Same request" is the fingerprint below: the fields the client sent, after
schema validation and normalization, plus the query options that change what
gets created. Key order, whitespace and ignored extra fields do not matter; an
explicit ``null`` does, because some fields treat it as a choice.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from typing import Any

FINGERPRINT_SCHEMA_VERSION = 1

CHAT_ID_CONFLICT = "chat_id_conflict"
CHAT_DELETED = "chat_deleted"

_CANONICAL_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_RESERVED_UUIDS = frozenset({"00000000-0000-0000-0000-000000000000", "ffffffff-ffff-ffff-ffff-ffffffffffff"})


class ClientChatIdError(Exception):
    """A client chat id that cannot be created or replayed; carries no chat data."""

    def __init__(self, code: str, status_code: int) -> None:
        """Keep only a bounded code and status, never the existing row."""
        super().__init__(code)
        self.code = code
        self.status_code = status_code


def normalize_client_conversation_id(value: str) -> str:
    """Return the lowercase canonical form of a hyphenated UUID, or raise ValueError.

    Only the 36-character 8-4-4-4-12 form is accepted (any case), so the id the
    client keeps and the id the server stores differ at most in letter case.
    The nil and max UUIDs are refused because unrelated clients may share them.
    """
    candidate = value.lower()
    if _CANONICAL_UUID.fullmatch(candidate) is None:
        raise ValueError("id must be a UUID in 8-4-4-4-12 hexadecimal form")
    if candidate in _RESERVED_UUIDS:
        raise ValueError("id must not be the nil or max UUID")
    return candidate


def conversation_create_fingerprint(body: Mapping[str, Any], options: Mapping[str, Any]) -> str:
    """Hash a create request canonically.

    Args:
        body: The validated request fields the client set, excluding ``id``.
        options: Query options that change what the create produces.

    Returns:
        A 64-character lowercase SHA-256 hex digest.
    """
    payload = {"schema_version": FINGERPRINT_SCHEMA_VERSION, "body": dict(body), "options": dict(options)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def resolve_client_chat_replay(
    existing: Mapping[str, Any] | None,
    *,
    owner_id: str,
    fingerprint: str,
) -> Mapping[str, Any] | None:
    """Decide what an existing row with the requested id means for this create.

    Args:
        existing: The row with the requested id, including a trashed one, or
            None when the caller can see no such row.
        owner_id: The authenticated caller.
        fingerprint: The fingerprint of the current request.

    Returns:
        The existing chat to replay, or None when the id is free.

    Raises:
        ClientChatIdError: 409 ``chat_id_conflict`` for another owner's chat, a
            different request or a chat created without a client id; 410
            ``chat_deleted`` for the caller's own matching chat in trash.
    """
    if existing is None:
        return None
    if str(existing.get("client_id") or "").strip() != str(owner_id).strip():
        raise ClientChatIdError(CHAT_ID_CONFLICT, 409)
    stored = existing.get("create_request_fingerprint")
    if not isinstance(stored, str) or not hmac.compare_digest(stored, fingerprint):
        raise ClientChatIdError(CHAT_ID_CONFLICT, 409)
    if existing.get("deleted"):
        raise ClientChatIdError(CHAT_DELETED, 410)
    return existing


__all__ = [
    "CHAT_DELETED",
    "CHAT_ID_CONFLICT",
    "ClientChatIdError",
    "conversation_create_fingerprint",
    "normalize_client_conversation_id",
    "resolve_client_chat_replay",
]
