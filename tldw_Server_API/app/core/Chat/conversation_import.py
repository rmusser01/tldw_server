"""Lossless import of an "On this device" chat into the signed-in account (D7 P8).

``POST /api/v1/chats/import`` takes one conversation and its whole message
graph. This module is the part that needs no database: it checks the request,
puts the messages in an order that can be written (parents first) and
fingerprints the request so a retry can be told from a different import.

What is kept exactly: the conversation id, message ids, parent links, roles,
text, timestamps (as UTC, to the millisecond) and image bytes. Nothing is
stamped with the time of the import.

What is refused instead of being dropped or repaired:

* a parent that is not part of the import, a cycle, or a repeated message id;
* generation metadata outside the allow-list in ``generation_metadata``, or on
  a message that is not an assistant reply;
* a message with neither text nor an image, because the message store cannot
  hold one;
* a timestamp without an offset, before 1970, or further in the future than
  the clock-skew allowance. A future timestamp would sort the imported message
  after every message sent later.

Idempotency
-----------
The fingerprint is a SHA-256 over the normalized request, including the
conversation id. It is stored in the owner-only provenance of every imported
message (``history_admission_json``), next to the ``parent_graph_v1``
interpretation that lets a branching import be read as an ordinary parent
graph. An import repeated by the same owner with the same fingerprint is a
replay; any other use of the conversation id is a conflict. The comparison is
with the original request, not with the chat as it is now, so a retry still
replays after the chat was renamed, edited or continued.

This mirrors client-id chat creation (D7 P3, ``POST /chats/``) without
depending on it, and needs no schema change. Because the fingerprint lives
with the messages, an import must have at least one message. Once P3's
``conversations.create_request_fingerprint`` column exists, the import
fingerprint can move there and ``normalize_client_conversation_id`` and the
replay rules can be shared with ``conversation_create_idempotency``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from tldw_Server_API.app.core.Chat.generation_metadata import (
    GenerationMetadataError,
    validate_generation_metadata,
)

IMPORT_FINGERPRINT_SCHEMA_VERSION = 1
IMPORT_AUTHORITY_VERSION = 1
DEFAULT_MAX_FUTURE_SKEW_SECONDS = 300

CHAT_ID_CONFLICT = "chat_id_conflict"
CHAT_DELETED = "chat_deleted"
MESSAGE_ID_CONFLICT = "message_id_conflict"

_CANONICAL_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_RESERVED_UUIDS = frozenset({"00000000-0000-0000-0000-000000000000", "ffffffff-ffff-ffff-ffff-ffffffffffff"})
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

_CONVERSATION_FIELDS = (
    "title",
    "state",
    "created_at",
    "last_modified",
    "character_id",
    "assistant_kind",
    "assistant_id",
    "persona_memory_mode",
    "parent_conversation_id",
    "forked_from_message_id",
)

_REPLAY_MESSAGES = {
    CHAT_ID_CONFLICT: "This chat id is already in use. Retry with the original request, or use a new id.",
    CHAT_DELETED: "The chat imported with this id is in trash. Restore it, or import it with a new id.",
}


class ChatImportError(Exception):
    """An import that is refused, with a stable code and nothing of any existing chat."""

    def __init__(self, code: str, status_code: int, message: str, **context: Any) -> None:
        """Keep a bounded code, HTTP status, a message for people and optional context."""
        super().__init__(code)
        self.code = code
        self.status_code = status_code
        self.message = message
        self.context = context

    def detail(self) -> dict[str, Any]:
        """Return the response detail: ``error_code``, ``message`` and any context."""
        return {"error_code": self.code, "message": self.message, **self.context}


@dataclass(frozen=True)
class ChatImportLimits:
    """Size limits for one import; the caller supplies the configured values."""

    max_content_chars: int
    max_image_bytes: int
    max_total_image_bytes: int
    max_future_skew_seconds: int = DEFAULT_MAX_FUTURE_SKEW_SECONDS


@dataclass(frozen=True)
class PreparedChatImport:
    """A validated import, ready to be written in one transaction.

    Attributes:
        conversation_id: The canonical conversation id.
        conversation: Conversation fields with normalized timestamps.
        messages: Messages in an order where every parent precedes its
            children; otherwise the order they were sent in.
        fingerprint: SHA-256 hex digest of the normalized request.
    """

    conversation_id: str
    conversation: dict[str, Any]
    messages: tuple[dict[str, Any], ...]
    fingerprint: str


def normalize_client_conversation_id(value: str) -> str:
    """Return the lowercase canonical form of a hyphenated UUID, or raise ValueError.

    Only the 36-character 8-4-4-4-12 form is accepted (any case), so the id the
    client keeps and the id the server stores differ at most in letter case.
    The nil and max UUIDs are refused because unrelated clients may share them.
    """
    candidate = value.lower() if isinstance(value, str) else ""
    if _CANONICAL_UUID.fullmatch(candidate) is None:
        raise ValueError("id must be a UUID in 8-4-4-4-12 hexadecimal form")
    if candidate in _RESERVED_UUIDS:
        raise ValueError("id must not be the nil or max UUID")
    return candidate


def _is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_HEX.fullmatch(value) is not None


def import_message_authority(fingerprint: str) -> dict[str, Any]:
    """Return the provenance stored with every imported message.

    ``parent_graph_v1`` marks the message as part of an explicit parent graph,
    so a branching import needs no legacy review. ``import`` records which
    request wrote it.
    """
    if not _is_sha256_hex(fingerprint):
        raise ValueError("fingerprint must be a lowercase SHA-256 hex digest")
    return {
        "version": 1,
        "interpretation": {"kind": "parent_graph_v1"},
        "settled": True,
        "import": {"version": IMPORT_AUTHORITY_VERSION, "request_fingerprint": fingerprint},
    }


def import_fingerprint_from_authority(raw: Any) -> str | None:
    """Return the import fingerprint in a message's provenance, or None if it has none."""
    authority = raw
    if isinstance(raw, str):
        try:
            authority = json.loads(raw)
        except ValueError:
            return None
    if not isinstance(authority, Mapping):
        return None
    marker = authority.get("import")
    if not isinstance(marker, Mapping) or marker.get("version") != IMPORT_AUTHORITY_VERSION:
        return None
    fingerprint = marker.get("request_fingerprint")
    return fingerprint if _is_sha256_hex(fingerprint) else None


def chat_id_conflict() -> ChatImportError:
    """Return the refusal for a conversation id the caller cannot use or replay."""
    return ChatImportError(CHAT_ID_CONFLICT, 409, _REPLAY_MESSAGES[CHAT_ID_CONFLICT])


def resolve_import_replay(
    existing: Mapping[str, Any] | None,
    *,
    owner_id: Any,
    stored_fingerprint: str | None,
    fingerprint: str,
) -> Mapping[str, Any] | None:
    """Decide what an existing conversation with the requested id means for this import.

    Args:
        existing: The row with the requested id, including a trashed one, or
            None when the caller can see no such row.
        owner_id: The authenticated caller.
        stored_fingerprint: The import fingerprint stored with that chat's
            messages, or None if it was not created by an import.
        fingerprint: The fingerprint of the current request.

    Returns:
        The existing chat to replay, or None when the id is free.

    Raises:
        ChatImportError: 409 ``chat_id_conflict`` for another owner's chat, a
            different request or a chat that was not imported; 410
            ``chat_deleted`` for the caller's own matching import in trash.
    """
    if existing is None:
        return None
    if str(existing.get("client_id") or "").strip() != str(owner_id).strip():
        raise chat_id_conflict()
    if not isinstance(stored_fingerprint, str) or not hmac.compare_digest(stored_fingerprint, fingerprint):
        raise chat_id_conflict()
    if existing.get("deleted"):
        raise ChatImportError(CHAT_DELETED, 410, _REPLAY_MESSAGES[CHAT_DELETED])
    return existing


def _timestamp(value: Any, *, field: str, latest: datetime, message_id: str | None = None) -> datetime:
    """Return ``value`` in UTC at millisecond precision, or refuse it."""
    context = {"field": field, **({"message_id": message_id} if message_id is not None else {})}
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ChatImportError("invalid_timestamp", 422, "Timestamps need a UTC offset, for example a trailing Z.", **context)
    moment = value.astimezone(timezone.utc)
    if moment < _EPOCH:
        raise ChatImportError("invalid_timestamp", 422, "Timestamps before 1970 are not supported.", **context)
    if moment > latest:
        raise ChatImportError(
            "timestamp_in_future",
            422,
            "A timestamp is ahead of the server clock. Check this device's clock and try again.",
            **context,
        )
    return moment.replace(microsecond=moment.microsecond // 1000 * 1000)


def _format_timestamp(value: datetime) -> str:
    """Format like the message store does: ``2026-09-01T10:00:00.250Z``."""
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _text(value: Any, *, field: str, message_id: str | None = None) -> str:
    """Return text unchanged, refusing text the database cannot store.

    PostgreSQL text cannot hold NUL. Neither backend can hold half of a
    surrogate pair, which JSON can carry as an escape when a client cut a
    string in the middle of an emoji.
    """
    text = value if isinstance(value, str) else ""
    context = {"field": field, **({"message_id": message_id} if message_id is not None else {})}
    if "\x00" in text:
        raise ChatImportError("invalid_text", 422, "Text must not contain NUL characters.", **context)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise ChatImportError(
            "invalid_text", 422, "Text contains an unpaired surrogate and cannot be stored as UTF-8.", **context
        ) from None
    return text


def _decode_image(value: Any, *, max_bytes: int, message_id: str) -> bytes:
    """Decode one image sent as a base64 data URL or as bare base64."""
    invalid = ChatImportError(
        "invalid_image", 422, "An image could not be read. Send it as a base64 data URL.", message_id=message_id
    )
    if not isinstance(value, str):
        raise invalid
    encoded = value
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or not header.endswith(";base64"):
            raise invalid
    encoded = "".join(encoded.split())
    if not encoded:
        raise invalid
    if len(encoded) > ((max_bytes + 2) // 3) * 4:
        raise ChatImportError(
            "image_too_large", 413, f"An image is larger than {max_bytes} bytes.", message_id=message_id, limit=max_bytes
        )
    try:
        data = base64.b64decode(encoded + "=" * (-len(encoded) % 4), validate=True)
    except (binascii.Error, ValueError):
        raise invalid from None
    if not data:
        raise invalid
    if len(data) > max_bytes:
        raise ChatImportError(
            "image_too_large", 413, f"An image is larger than {max_bytes} bytes.", message_id=message_id, limit=max_bytes
        )
    return data


def _generation_metadata(value: Any, *, role: str, message_id: str) -> dict[str, Any]:
    """Return allow-listed generation metadata; anything else is refused, never dropped."""
    if value is None or (isinstance(value, Mapping) and not value):
        return {}
    if role != "assistant":
        raise ChatImportError(
            "unsupported_metadata",
            422,
            "Generation metadata is only stored for assistant messages.",
            message_id=message_id,
        )
    try:
        return validate_generation_metadata(value)
    except GenerationMetadataError as error:
        raise ChatImportError("unsupported_metadata", 422, str(error), message_id=message_id) from None


def _parents_first(messages: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return message positions so that every parent precedes its children.

    Messages otherwise keep the order they were sent in. That order breaks ties
    between equal timestamps, because the message store orders by timestamp and
    then by write order.
    """
    position: dict[str, int] = {}
    for index, message in enumerate(messages):
        message_id = message["id"]
        if message_id in position:
            raise ChatImportError(
                "duplicate_message_id", 422, "Two messages in the import have the same id.", message_id=message_id
            )
        position[message_id] = index
    for message in messages:
        parent = message.get("parent_message_id")
        if parent is not None and parent not in position:
            raise ChatImportError(
                "missing_parent",
                422,
                "A message names a parent that is not part of the import.",
                message_id=message["id"],
            )

    ordered: list[int] = []
    placed: set[str] = set()
    for message in messages:
        pending: list[int] = []
        walking: set[str] = set()
        current: str | None = message["id"]
        while current is not None and current not in placed:
            if current in walking:
                raise ChatImportError(
                    "cyclic_ancestry", 422, "Message parents form a cycle.", message_id=message["id"]
                )
            walking.add(current)
            pending.append(position[current])
            current = messages[position[current]].get("parent_message_id")
        for index in reversed(pending):
            placed.add(messages[index]["id"])
            ordered.append(index)
    return ordered


def chat_import_fingerprint(
    conversation_id: str, conversation: Mapping[str, Any], messages: Sequence[Mapping[str, Any]]
) -> str:
    """Hash a normalized import canonically.

    The conversation id is part of the hash because the fingerprint is stored
    with the messages: provenance copied to another conversation never reads as
    an import of that conversation.

    Args:
        conversation_id: The canonical conversation id.
        conversation: Normalized conversation fields (see ``_CONVERSATION_FIELDS``).
        messages: Normalized messages in the order they were sent, each with
            ``id``, ``parent_message_id``, ``role``, ``content``, ``timestamp``,
            ``images`` (digests, not bytes) and ``metadata``.

    Returns:
        A 64-character lowercase SHA-256 hex digest.
    """
    payload = {
        "schema_version": IMPORT_FINGERPRINT_SCHEMA_VERSION,
        "kind": "chat_import",
        "conversation_id": conversation_id,
        "conversation": {field: conversation.get(field) for field in _CONVERSATION_FIELDS},
        "messages": list(messages),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def prepare_chat_import(
    request: Mapping[str, Any],
    *,
    limits: ChatImportLimits,
    now: datetime,
    inspect_image: Callable[[bytes], str],
    content_length: Callable[[str], int] = len,
) -> PreparedChatImport:
    """Validate an import request and return what to write.

    Args:
        request: The parsed request body (``ChatImportRequest.model_dump()``).
        limits: Size limits to apply.
        now: The server clock, used only to refuse timestamps in the future.
        inspect_image: Returns the MIME type of decoded image bytes, or raises
            ``ChatImportError`` when they are not an image that can be stored.
        content_length: Measures message text the way the message path does.

    Returns:
        The normalized conversation, the messages in write order and the
        request fingerprint.

    Raises:
        ChatImportError: The request cannot be imported as sent. Nothing has
            been written when this is raised.
        ValueError: ``request["id"]`` is not a usable UUID.
    """
    conversation_id = normalize_client_conversation_id(request["id"])
    latest = now.astimezone(timezone.utc) + timedelta(seconds=limits.max_future_skew_seconds)

    title = _text(request.get("title"), field="title")
    if not title.strip():
        raise ChatImportError("invalid_title", 422, "A chat needs a title.", field="title")

    source_messages = list(request.get("messages") or ())
    order = _parents_first(source_messages)

    image_bytes = 0
    normalized: list[dict[str, Any]] = []
    canonical: list[dict[str, Any]] = []
    newest: datetime | None = None
    for message in source_messages:
        message_id = message["id"]
        role = message["role"]
        content = _text(message.get("content"), field="content", message_id=message_id)
        if content_length(content) > limits.max_content_chars:
            raise ChatImportError(
                "message_content_too_large",
                413,
                f"A message is longer than {limits.max_content_chars} characters.",
                message_id=message_id,
                limit=limits.max_content_chars,
            )
        moment = _timestamp(message.get("timestamp"), field="timestamp", latest=latest, message_id=message_id)
        newest = moment if newest is None or moment > newest else newest
        metadata = _generation_metadata(message.get("metadata"), role=role, message_id=message_id)

        images: list[dict[str, Any]] = []
        digests: list[list[Any]] = []
        for encoded in message.get("images") or ():
            data = _decode_image(encoded, max_bytes=limits.max_image_bytes, message_id=message_id)
            image_bytes += len(data)
            if image_bytes > limits.max_total_image_bytes:
                raise ChatImportError(
                    "images_too_large",
                    413,
                    f"The images in this chat total more than {limits.max_total_image_bytes} bytes.",
                    limit=limits.max_total_image_bytes,
                )
            try:
                mime = inspect_image(data)
            except ChatImportError as error:
                error.context.setdefault("message_id", message_id)
                raise
            images.append({"data": data, "mime": mime})
            digests.append([mime, hashlib.sha256(data).hexdigest(), len(data)])

        if not content.strip() and not images:
            raise ChatImportError(
                "empty_message", 422, "A message needs text or an image.", message_id=message_id
            )

        timestamp = _format_timestamp(moment)
        normalized.append(
            {
                "id": message_id,
                "parent_message_id": message.get("parent_message_id"),
                "sender": role,
                "content": content,
                "timestamp": timestamp,
                "images": images,
                "extra_metadata": {"sender_role": role, **metadata},
            }
        )
        canonical.append(
            {
                "id": message_id,
                "parent_message_id": message.get("parent_message_id"),
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "images": digests,
                "metadata": metadata or None,
            }
        )

    created = _timestamp(request.get("created_at"), field="created_at", latest=latest)
    if request.get("last_modified") is not None:
        modified = _timestamp(request["last_modified"], field="last_modified", latest=latest)
    else:
        # Not sent: the chat was last touched when its newest message was written.
        modified = max(created, newest) if newest is not None else created

    conversation = {
        "title": title,
        "state": request.get("state"),
        "created_at": _format_timestamp(created),
        "last_modified": _format_timestamp(modified),
        "character_id": request.get("character_id"),
        "assistant_kind": request.get("assistant_kind"),
        "assistant_id": request.get("assistant_id"),
        "persona_memory_mode": request.get("persona_memory_mode"),
        "parent_conversation_id": request.get("parent_conversation_id"),
        "forked_from_message_id": request.get("forked_from_message_id"),
    }
    return PreparedChatImport(
        conversation_id=conversation_id,
        conversation=conversation,
        messages=tuple(normalized[index] for index in order),
        fingerprint=chat_import_fingerprint(conversation_id, conversation, canonical),
    )


__all__ = [
    "CHAT_DELETED",
    "CHAT_ID_CONFLICT",
    "DEFAULT_MAX_FUTURE_SKEW_SECONDS",
    "MESSAGE_ID_CONFLICT",
    "ChatImportError",
    "ChatImportLimits",
    "PreparedChatImport",
    "chat_id_conflict",
    "chat_import_fingerprint",
    "import_fingerprint_from_authority",
    "import_message_authority",
    "normalize_client_conversation_id",
    "prepare_chat_import",
    "resolve_import_replay",
]
