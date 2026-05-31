from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDBError,
    ConflictError,
    FTSQueryTranslator,
    InputError,
    _CHACHA_NONCRITICAL_EXCEPTIONS,
    logger,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class MessageStore:
    """Focused persistence seam for message CRUD operations."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Message creation
    # ------------------------------------------------------------------

    def add_message(self, msg_data: dict[str, Any]) -> str | None:
        """
        Adds a new message to a conversation, optionally with image data.

        `id` (UUID string) is auto-generated if not provided in `msg_data`.
        Requires 'conversation_id', 'sender'. Message must have 'content' (text) or image attachments.
        `client_id` defaults to DB instance's `client_id`. `version` is set to 1.
        `timestamp` defaults to current UTC time if not provided; `last_modified` is set to current UTC time.

        Verifies that the parent conversation (given by `conversation_id`) exists and is not deleted.
        FTS updates (`messages_fts` for content) and `sync_log` entries are handled by SQL triggers.

        Args:
            msg_data: Dictionary with message data.
                      Required: 'conversation_id', 'sender'. At least one of 'content' or images.
                      Optional: 'id', 'parent_message_id', 'content' (str),
                                'image_data' (bytes), 'image_mime_type' (str, required if image_data present),
                                'images' (iterable of {'data','mime'}), 'timestamp', 'ranking', 'client_id'.

        Returns:
            The string UUID of the newly added message.

        Raises:
            InputError: If required fields are missing, if both 'content' and attachments are absent,
                        or if the parent conversation is not found or is deleted.
            ConflictError: If a message with the provided 'id' (if any) already exists.
            CharactersRAGDBError: For other database errors (e.g., FK violation for conversation_id).
        """
        images_payload_raw = msg_data.pop('images', None)
        normalized_images: list[tuple[bytes, str]] = []
        if images_payload_raw:
            for entry in images_payload_raw:
                img_bytes: bytes | None = None
                img_mime: str | None = None
                if isinstance(entry, dict):
                    img_bytes = entry.get("data") or entry.get("image_data")
                    img_mime = entry.get("mime") or entry.get("image_mime_type")
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    img_bytes, img_mime = entry[0], entry[1]
                if img_bytes is None or img_mime is None:
                    continue
                if isinstance(img_bytes, memoryview):
                    img_bytes = img_bytes.tobytes()
                normalized_images.append((img_bytes, str(img_mime)))

        # Enforce maximum image sizes (single and multi-image) using settings override
        try:
            from tldw_Server_API.app.core.config import settings  # noqa: E402
            _max_img_bytes = int(settings.get("MAX_MESSAGE_IMAGE_BYTES", 5 * 1024 * 1024))
        except _CHACHA_NONCRITICAL_EXCEPTIONS:
            _max_img_bytes = 5 * 1024 * 1024  # 5MB default

        # Validate primary image size if present
        primary_img = msg_data.get('image_data')
        if isinstance(primary_img, memoryview):
            primary_img = primary_img.tobytes()
        if isinstance(primary_img, (bytes, bytearray)) and len(primary_img) > _max_img_bytes:
            raise InputError(  # noqa: TRY003
                f"Primary image attachment exceeds maximum size of {_max_img_bytes} bytes"
            )

        # Validate any additional images provided via 'images'
        if normalized_images:
            for b, _m in normalized_images:
                if b is None:
                    continue
                if isinstance(b, memoryview):
                    b = b.tobytes()
                if isinstance(b, (bytes, bytearray)) and len(b) > _max_img_bytes:
                    raise InputError(  # noqa: TRY003
                        f"Message image attachment exceeds maximum size of {_max_img_bytes} bytes"
                    )

        msg_id = msg_data.get('id') or self._db._generate_uuid()

        required_fields = ['conversation_id', 'sender']
        for field in required_fields:
            if field not in msg_data:
                raise InputError(f"Required field '{field}' is missing for message.")  # noqa: TRY003
        if not msg_data.get('content') and not msg_data.get('image_data') and not normalized_images:
            raise InputError("Message must have text content or image data.")  # noqa: TRY003
        if msg_data.get('image_data') and not msg_data.get('image_mime_type'):
            raise InputError("image_mime_type is required if image_data is provided.")  # noqa: TRY003

        if normalized_images and not msg_data.get('image_data'):
            first_bytes, first_mime = normalized_images[0]
            msg_data['image_data'] = first_bytes
            msg_data['image_mime_type'] = first_mime

        client_id = msg_data.get('client_id') or self._db.client_id
        if not client_id:
            raise InputError("Client ID is required for message.")  # noqa: TRY003

        now = self._db._get_current_utc_timestamp_iso()
        timestamp = msg_data.get('timestamp') or now

        query = """
                INSERT INTO messages (id, conversation_id, parent_message_id, sender, content,
                                      image_data, image_mime_type,
                                      timestamp, ranking, last_modified, client_id, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
        if self._db.backend_type == BackendType.POSTGRESQL:
            params = (
                msg_id, msg_data['conversation_id'], msg_data.get('parent_message_id'),
                msg_data['sender'], msg_data.get('content', ''),
                msg_data.get('image_data'), msg_data.get('image_mime_type'),
                timestamp, msg_data.get('ranking'), now, client_id, 1, False
            )
        else:
            params = (
                msg_id, msg_data['conversation_id'], msg_data.get('parent_message_id'),
                msg_data['sender'], msg_data.get('content', ''),
                msg_data.get('image_data'), msg_data.get('image_mime_type'),
                timestamp, msg_data.get('ranking'), now, client_id, 1, 0
            )
        try:
            with self._db.transaction():
                conv_cursor = self._db.execute_query(
                    "SELECT 1 FROM conversations WHERE id = ? AND deleted = FALSE",
                    (msg_data['conversation_id'],),
                )
                if not conv_cursor.fetchone():
                    raise InputError(  # noqa: TRY003, TRY301
                        f"Cannot add message: Conversation ID '{msg_data['conversation_id']}' not found or deleted."
                    )
                self._db.execute_query(query, params)
                if normalized_images:
                    self._insert_message_images(msg_id, normalized_images)
            logger.info(
                'Added message ID: {} to conversation {} (Images stored: {}).',
                msg_id,
                msg_data['conversation_id'],
                len(normalized_images) if normalized_images else ("Yes" if msg_data.get('image_data') else "No"),
            )
            return msg_id  # noqa: TRY300
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: messages.id" in str(e):
                raise ConflictError(  # noqa: TRY003
                    f"Message with ID '{msg_id}' already exists.",
                    entity="messages",
                    entity_id=msg_id,
                ) from e
            raise CharactersRAGDBError(f"Database integrity error adding message: {e}") from e  # noqa: TRY003
        except InputError:
            raise
        except CharactersRAGDBError as e:
            logger.error(f"Database error adding message: {e}")
            raise

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _insert_message_images(self, message_id: str, images: list[tuple[bytes, str]]) -> None:
        """Insert or replace message images for the given message."""
        if not images:
            return
        params: list[tuple[str, int, bytes, str]] = []
        for idx, (img_bytes, img_mime) in enumerate(images):
            if img_bytes is None or img_mime is None:
                continue
            if isinstance(img_bytes, memoryview):
                img_bytes = img_bytes.tobytes()
            params.append((message_id, idx, img_bytes, img_mime))
        if not params:
            return
        query = (
            "INSERT INTO message_images (message_id, position, image_data, image_mime_type) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(message_id, position) DO UPDATE SET "
            "image_data=excluded.image_data, image_mime_type=excluded.image_mime_type, "
            "created_at=CURRENT_TIMESTAMP"
        )
        self._db.execute_many(query, params, commit=False)

    def append_message_image(
        self,
        message_id: str,
        image_bytes: bytes,
        mime_type: str,
        *,
        commit: bool = True,
    ) -> int:
        """Append one image to a message after the current maximum image position."""
        if isinstance(image_bytes, memoryview):
            image_bytes = image_bytes.tobytes()
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise InputError("image_bytes must be bytes-like.")  # noqa: TRY003
        if not mime_type:
            raise InputError("mime_type is required for message images.")  # noqa: TRY003

        try:
            from tldw_Server_API.app.core.config import settings  # noqa: E402

            max_image_bytes = int(settings.get("MAX_MESSAGE_IMAGE_BYTES", 5 * 1024 * 1024))
        except _CHACHA_NONCRITICAL_EXCEPTIONS:
            max_image_bytes = 5 * 1024 * 1024
        if len(image_bytes) > max_image_bytes:
            raise InputError(  # noqa: TRY003
                f"Message image attachment exceeds maximum size of {max_image_bytes} bytes"
            )

        def _append_once() -> int:
            cursor = self._db.execute_query(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM message_images WHERE message_id = ?",
                (message_id,),
            )
            row = cursor.fetchone()
            position = int(row[0] if row is not None else 0)
            self._db.execute_query(
                """
                INSERT INTO message_images (message_id, position, image_data, image_mime_type)
                VALUES (?, ?, ?, ?)
                """,
                (message_id, position, bytes(image_bytes), str(mime_type)),
                commit=False,
            )
            return position

        def _append_with_retries(*, transactional: bool) -> int:
            last_error: Exception | None = None
            for _ in range(5):
                try:
                    if transactional:
                        with self._db.transaction():
                            return _append_once()
                    return _append_once()
                except sqlite3.IntegrityError as exc:
                    last_error = exc
                    continue
            raise ConflictError(  # noqa: TRY003
                f"Concurrent append conflict for message image positions on message_id={message_id}",
            ) from last_error

        if not commit:
            return _append_with_retries(transactional=False)
        return _append_with_retries(transactional=True)

    def get_message_images(self, message_id: str) -> list[dict[str, Any]]:
        """Fetch all images associated with a message, ordered by position."""
        try:
            cursor = self._db.execute_query(
                "SELECT message_id, position, image_data, image_mime_type FROM message_images "
                "WHERE message_id = ? ORDER BY position ASC",
                (message_id,),
            )
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            images: list[dict[str, Any]] = []
            for row in rows:
                record = dict(row) if isinstance(row, dict) else {columns[idx]: row[idx] for idx in range(len(columns))}
                img_bytes = record.get("image_data")
                if isinstance(img_bytes, memoryview):
                    record["image_data"] = img_bytes.tobytes()
                images.append(record)
            return images  # noqa: TRY300
        except CharactersRAGDBError as e:
            logger.error(f"Failed to fetch images for message {message_id}: {e}")
            return []

    # ------------------------------------------------------------------
    # Message retrieval
    # ------------------------------------------------------------------

    def get_message_conversation_id(self, message_id: str) -> str | None:
        """Return the conversation_id for a message if it exists and is not deleted."""
        query = "SELECT conversation_id FROM messages WHERE id = ? AND deleted = FALSE"
        try:
            cursor = self._db.execute_query(query, (message_id,))
            row = cursor.fetchone()
            if not row:
                return None
            if isinstance(row, dict):
                return row.get("conversation_id")
            return row[0] if row else None
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching conversation_id for message {message_id}: {e}")
            raise

    def get_message_by_id(self, message_id: str, include_deleted: bool = False) -> dict[str, Any] | None:
        """
        Retrieves a specific message by its UUID.

        Only non-deleted messages are returned. Includes all fields, such as
        `image_data` (BLOB) and `image_mime_type` if present.

        Args:
            message_id: The string UUID of the message.

        Returns:
            A dictionary with message data if found and not deleted, else None.

        Raises:
            CharactersRAGDBError: For database errors.
        """
        deleted_clause = "" if include_deleted else "AND m.deleted = FALSE AND c.deleted = FALSE"
        query = (
            "SELECT m.id, m.conversation_id, m.parent_message_id, m.sender, m.content, "
            "m.image_data, m.image_mime_type, m.timestamp, m.ranking, m.last_modified, "
            "m.version, m.client_id, m.deleted "
            "FROM messages m "
            "JOIN conversations c ON c.id = m.conversation_id "
            f"WHERE m.id = ? {deleted_clause}"  # nosec B608
        )
        try:
            cursor = self._db.execute_query(query, (message_id,))
            row = cursor.fetchone()
            if not row:
                return None
            if isinstance(row, dict):
                record = dict(row)
            else:
                columns = [col[0] for col in cursor.description] if cursor.description else []
                record = {columns[idx]: row[idx] for idx in range(len(columns))}
            img_blob = record.get("image_data")
            if isinstance(img_blob, memoryview):
                record["image_data"] = img_blob.tobytes()
            record["images"] = self.get_message_images(message_id)
            return record  # noqa: TRY300
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching message ID {message_id}: {e}")
            raise

    def append_message_from_sync(
        self,
        *,
        stable_message_id: str,
        conversation_id: str,
        sender: str,
        content: str | None,
        timestamp: str | None,
        sync_client_id: str,
        object_revision: int,
        payload_hash: str,
        parent_message_id: str | None = None,
        ranking: int | None = None,
        projection_message_id: str | None = None,
    ) -> dict[str, Any]:
        """Append a chat message from Sync v2 with stable-ID dedupe and divergence preservation."""

        normalized_stable_id = str(stable_message_id).strip()
        if not normalized_stable_id:
            raise InputError("stable_message_id cannot be empty.")  # noqa: TRY003
        if object_revision < 1:
            raise InputError("object_revision must be greater than zero.")  # noqa: TRY003

        projection_id = projection_message_id or normalized_stable_id
        forced_conflict = projection_message_id is not None
        existing_versions = self.get_messages_by_sync_stable_id(normalized_stable_id, include_deleted=True)
        projection_id_blocked = False
        for version in existing_versions:
            sync_meta = ((version.get("metadata") or {}).get("extra") or {}).get("sync_v2") or {}
            if sync_meta.get("payload_hash") == payload_hash:
                return {
                    "message_id": version["id"],
                    "stable_message_id": normalized_stable_id,
                    "created": False,
                    "idempotent": True,
                    "conflict": bool(sync_meta.get("projection_conflict")),
                }
        for version in existing_versions:
            if version["id"] == projection_id:
                sync_meta = ((version.get("metadata") or {}).get("extra") or {}).get("sync_v2", {})
                sync_payload_hash = sync_meta.get("payload_hash")
                if sync_payload_hash and sync_payload_hash != payload_hash:
                    projection_id_blocked = True
                    continue
                if (
                    not sync_payload_hash
                    and not self._sync_projection_matches(
                        version,
                        conversation_id=conversation_id,
                        parent_message_id=parent_message_id,
                        sender=sender,
                        content=content,
                        timestamp=timestamp,
                        ranking=ranking,
                        sync_client_id=sync_client_id,
                    )
                ):
                    projection_id_blocked = True
                    continue
                projection_conflict = forced_conflict or bool(
                    sync_meta.get("projection_conflict")
                )
                self._set_sync_v2_message_metadata_or_raise(
                    message_id=projection_id,
                    stable_message_id=normalized_stable_id,
                    payload_hash=payload_hash,
                    object_revision=object_revision,
                    projection_conflict=projection_conflict,
                )
                return {
                    "message_id": projection_id,
                    "stable_message_id": normalized_stable_id,
                    "created": False,
                    "idempotent": True,
                    "conflict": projection_conflict,
                }

        is_conflict = forced_conflict or bool(existing_versions)
        if is_conflict and (projection_message_id is None or projection_id_blocked):
            projection_id = self._available_sync_conflict_projection_id(
                normalized_stable_id,
                object_revision,
                existing_versions,
            )

        message_id = self.add_message(
            {
                "id": projection_id,
                "conversation_id": conversation_id,
                "parent_message_id": parent_message_id,
                "sender": sender,
                "content": content or "",
                "timestamp": timestamp,
                "ranking": ranking,
                "client_id": sync_client_id,
            }
        )
        if message_id is None:
            raise CharactersRAGDBError("Failed to append Sync v2 message projection.")  # noqa: TRY003
        if object_revision != 1:
            self._db.execute_query(
                "UPDATE messages SET version = ?, client_id = ? WHERE id = ?",
                (object_revision, sync_client_id, message_id),
                commit=True,
            )
        self._set_sync_v2_message_metadata_or_raise(
            message_id=message_id,
            stable_message_id=normalized_stable_id,
            payload_hash=payload_hash,
            object_revision=object_revision,
            projection_conflict=is_conflict,
        )
        return {
            "message_id": message_id,
            "stable_message_id": normalized_stable_id,
            "created": True,
            "idempotent": False,
            "conflict": is_conflict,
        }

    def tombstone_message_from_sync(
        self,
        *,
        stable_message_id: str,
        sync_client_id: str,
        object_revision: int,
        object_hash: str,
    ) -> bool:
        """Soft-delete all projections for a stable message from an accepted Sync v2 tombstone."""

        normalized_stable_id = str(stable_message_id).strip()
        if not normalized_stable_id:
            raise InputError("stable_message_id cannot be empty.")  # noqa: TRY003
        if object_revision < 1:
            raise InputError("object_revision must be greater than zero.")  # noqa: TRY003

        existing_versions = self.get_messages_by_sync_stable_id(normalized_stable_id, include_deleted=True)
        matched_versions = [
            version
            for version in existing_versions
            if (((version.get("metadata") or {}).get("extra") or {}).get("sync_v2") or {}).get("payload_hash")
            == object_hash
        ]
        if not matched_versions:
            matched_versions = [
                version
                for version in existing_versions
                if version["id"] == normalized_stable_id
                and not (((version.get("metadata") or {}).get("extra") or {}).get("sync_v2") or {}).get(
                    "payload_hash"
                )
            ]
        if not matched_versions:
            raise ConflictError(  # noqa: TRY003
                "Message projection matching Sync v2 tombstone base hash was not found.",
                entity="messages",
                entity_id=normalized_stable_id,
            )

        matched_ids = {str(version["id"]) for version in matched_versions}
        target_ids = [str(version["id"]) for version in existing_versions]
        now = self._db._get_current_utc_timestamp_iso()
        updated = 0
        with self._db.transaction() as conn:
            for target_id in target_ids:
                cursor = conn.execute(
                    """
                    UPDATE messages
                       SET deleted = ?,
                           last_modified = ?,
                           version = ?,
                           client_id = ?
                     WHERE id = ?
                    """,
                    (True, now, object_revision, sync_client_id, target_id),
                )
                updated += cursor.rowcount
        if updated == 0:
            raise ConflictError(  # noqa: TRY003
                "Message not found for Sync v2 tombstone.",
                entity="messages",
                entity_id=normalized_stable_id,
            )
        for version in existing_versions:
            sync_meta = dict(((version.get("metadata") or {}).get("extra") or {}).get("sync_v2") or {})
            sync_meta.setdefault("stable_message_id", normalized_stable_id)
            sync_meta.setdefault("payload_hash", object_hash if str(version["id"]) in matched_ids else "")
            sync_meta.update(
                {
                    "object_revision": object_revision,
                    "tombstoned": True,
                }
            )
            persisted = self.set_message_metadata_extra(version["id"], {"sync_v2": sync_meta}, merge=True)
            if not persisted:
                raise CharactersRAGDBError(  # noqa: TRY003
                    f"Failed to persist Sync v2 tombstone metadata for message {version['id']}."
                )
        return True

    def get_messages_by_sync_stable_id(
        self,
        stable_message_id: str,
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch message projections associated with a Sync v2 stable message ID."""

        self._db._ensure_message_metadata_table()
        deleted_clause = "" if include_deleted else "AND m.deleted = FALSE AND c.deleted = FALSE"
        query = (
            "SELECT m.id, m.conversation_id, m.parent_message_id, m.sender, m.content, "
            "m.image_data, m.image_mime_type, m.timestamp, m.ranking, m.last_modified, "
            "m.version, m.client_id, m.deleted, mm.tool_calls_json, mm.extra_json "
            "FROM messages m "
            "LEFT JOIN message_metadata mm ON mm.message_id = m.id "
            "JOIN conversations c ON c.id = m.conversation_id "
            f"WHERE 1 = 1 {deleted_clause} "  # nosec B608
            "ORDER BY m.timestamp ASC, m.id ASC"
        )
        cursor = self._db.execute_query(query)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description] if cursor.description else []
        results: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row) if isinstance(row, dict) else {columns[idx]: row[idx] for idx in range(len(columns))}
            extra = json.loads(record.pop("extra_json") or "{}")
            if not isinstance(extra, dict):
                extra = {}
            tool_calls = json.loads(record.pop("tool_calls_json") or "null")
            metadata = {"tool_calls": tool_calls, "extra": extra}
            sync_meta = extra.get("sync_v2") if isinstance(extra, dict) else None
            fallback_match = record["id"] == stable_message_id or str(record["id"]).startswith(
                f"{stable_message_id}__sync_conflict__"
            )
            metadata_match = isinstance(sync_meta, dict) and sync_meta.get("stable_message_id") == stable_message_id
            if not metadata_match and not fallback_match:
                continue
            if not isinstance(sync_meta, dict):
                extra["sync_v2"] = {"stable_message_id": stable_message_id}
            img_blob = record.get("image_data")
            if isinstance(img_blob, memoryview):
                record["image_data"] = img_blob.tobytes()
            record["images"] = self.get_message_images(record["id"])
            record["metadata"] = metadata
            results.append(record)
        return results

    def _set_sync_v2_message_metadata_or_raise(
        self,
        *,
        message_id: str,
        stable_message_id: str,
        payload_hash: str,
        object_revision: int,
        projection_conflict: bool,
    ) -> None:
        persisted = self.set_message_metadata_extra(
            message_id,
            {
                "sync_v2": {
                    "stable_message_id": stable_message_id,
                    "payload_hash": payload_hash,
                    "object_revision": object_revision,
                    "projection_conflict": projection_conflict,
                }
            },
            merge=True,
        )
        if not persisted:
            raise CharactersRAGDBError(  # noqa: TRY003
                f"Failed to persist Sync v2 metadata for message {message_id}."
            )

    @staticmethod
    def _sync_projection_matches(
        version: dict[str, Any],
        *,
        conversation_id: str,
        parent_message_id: str | None,
        sender: str,
        content: str | None,
        timestamp: str | None,
        ranking: int | None,
        sync_client_id: str,
    ) -> bool:
        """Return whether a metadata-less row matches the incoming Sync v2 projection."""

        if version.get("conversation_id") != conversation_id:
            return False
        if version.get("parent_message_id") != parent_message_id:
            return False
        if version.get("sender") != sender:
            return False
        if version.get("content") != (content or ""):
            return False
        if timestamp is not None and version.get("timestamp") != timestamp:
            return False
        if version.get("ranking") != ranking:
            return False
        return version.get("client_id") == sync_client_id

    @staticmethod
    def _available_sync_conflict_projection_id(
        stable_message_id: str,
        object_revision: int,
        existing_versions: list[dict[str, Any]],
    ) -> str:
        existing_ids = {str(version["id"]) for version in existing_versions}
        base_projection_id = f"{stable_message_id}__sync_conflict__{object_revision}"
        projection_id = base_projection_id
        suffix = 2
        while projection_id in existing_ids:
            projection_id = f"{base_projection_id}_{suffix}"
            suffix += 1
        return projection_id

    # ------------------------------------------------------------------
    # Message listing / querying
    # ------------------------------------------------------------------

    def get_messages_for_conversation(self, conversation_id: str, limit: int = 100, offset: int = 0,
                                      order_by_timestamp: str = "ASC", include_deleted: bool = False) -> list[dict[str, Any]]:
        """
        Lists messages for a specific conversation.
        Returns non-deleted messages, ordered by `timestamp` according to `order_by_timestamp`.
        Crucially, it also ensures the parent conversation is not soft-deleted.
        """
        if order_by_timestamp.upper() not in ["ASC", "DESC"]:
            raise InputError("order_by_timestamp must be 'ASC' or 'DESC'.")  # noqa: TRY003

        # The new query joins with conversations to check its 'deleted' status.
        delete_clause = "" if include_deleted else "AND m.deleted = FALSE"

        query = """
            SELECT m.id, m.conversation_id, m.parent_message_id, m.sender, m.content,
                   m.image_data, m.image_mime_type, m.timestamp, m.ranking,
                   m.last_modified, m.version, m.client_id, m.deleted
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.conversation_id = ?
              {delete_clause}
              AND c.deleted = FALSE
            ORDER BY m.timestamp {order_by_timestamp}
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608
        try:
            cursor = self._db.execute_query(query, (conversation_id, limit, offset))
            raw_rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            results: list[dict[str, Any]] = []
            for row in raw_rows:
                record = dict(row) if isinstance(row, dict) else {columns[idx]: row[idx] for idx in range(len(columns))}
                image_blob = record.get("image_data")
                if isinstance(image_blob, memoryview):
                    record["image_data"] = image_blob.tobytes()
                record["images"] = self.get_message_images(record["id"])
                results.append(record)
            return results  # noqa: TRY300
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching messages for conversation ID {conversation_id}: {e}")
            raise

    def count_root_messages_for_conversation(self, conversation_id: str) -> int:
        """Count root (parentless) messages for a conversation."""
        query = (
            "SELECT COUNT(1) FROM messages m "
            "JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.conversation_id = ? AND m.parent_message_id IS NULL "
            "AND m.deleted = FALSE AND c.deleted = FALSE"
        )
        try:
            cursor = self._db.execute_query(query, (conversation_id,))
            row = cursor.fetchone()
            if row is None:
                return 0
            try:
                return int(row[0])
            except _CHACHA_NONCRITICAL_EXCEPTIONS:
                return int(row.get("COUNT(1)") or row.get("count") or 0)
        except CharactersRAGDBError as e:
            logger.error("Database error counting root messages for conversation {}: {}", conversation_id, e)
            raise

    def get_root_messages_for_conversation(
        self,
        conversation_id: str,
        *,
        limit: int,
        offset: int,
        order_by_timestamp: str = "ASC",
    ) -> list[dict[str, Any]]:
        """Fetch root (parentless) messages with minimal columns for tree building."""
        if order_by_timestamp.upper() not in ["ASC", "DESC"]:
            raise InputError("order_by_timestamp must be 'ASC' or 'DESC'.")  # noqa: TRY003
        query = """
            SELECT m.id, m.parent_message_id, m.sender, m.content, m.timestamp
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.conversation_id = ?
              AND m.parent_message_id IS NULL
              AND m.deleted = FALSE
              AND c.deleted = FALSE
            ORDER BY m.timestamp {order_by_timestamp}
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608
        try:
            cursor = self._db.execute_query(query, (conversation_id, limit, offset))
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            results: list[dict[str, Any]] = []
            for row in rows:
                record = dict(row) if isinstance(row, dict) else {columns[idx]: row[idx] for idx in range(len(columns))}
                results.append(record)
            return results  # noqa: TRY300
        except CharactersRAGDBError as e:
            logger.error("Database error fetching root messages for conversation {}: {}", conversation_id, e)
            raise

    def get_messages_for_conversation_by_parent_ids(
        self,
        conversation_id: str,
        parent_ids: list[str],
        *,
        order_by_timestamp: str = "ASC",
    ) -> list[dict[str, Any]]:
        """Fetch child messages for the given parent IDs with minimal columns."""
        if not parent_ids:
            return []
        if order_by_timestamp.upper() not in ["ASC", "DESC"]:
            raise InputError("order_by_timestamp must be 'ASC' or 'DESC'.")  # noqa: TRY003
        placeholders = ",".join(["?"] * len(parent_ids))
        query = """
            SELECT m.id, m.parent_message_id, m.sender, m.content, m.timestamp
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.conversation_id = ?
              AND m.parent_message_id IN ({placeholders})
              AND m.deleted = FALSE
              AND c.deleted = FALSE
            ORDER BY m.timestamp {order_by_timestamp}
        """.format_map(locals())  # nosec B608
        params = [conversation_id, *parent_ids]
        try:
            cursor = self._db.execute_query(query, tuple(params))
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            results: list[dict[str, Any]] = []
            for row in rows:
                record = dict(row) if isinstance(row, dict) else {columns[idx]: row[idx] for idx in range(len(columns))}
                results.append(record)
            return results  # noqa: TRY300
        except CharactersRAGDBError as e:
            logger.error(
                'Database error fetching child messages for conversation {}: {}',
                conversation_id,
                e,
            )
            raise

    def has_system_message_for_conversation(
        self,
        conversation_id: str,
        include_deleted: bool = False,
    ) -> bool:
        """Check whether a conversation has at least one system message."""
        if include_deleted:
            query = """
                SELECT 1
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.conversation_id = ?
                  AND lower(m.sender) = 'system'
                  AND c.deleted = ?
                LIMIT 1
            """
            params = (conversation_id, False)
        else:
            query = """
                SELECT 1
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.conversation_id = ?
                  AND lower(m.sender) = 'system'
                  AND m.deleted = ?
                  AND c.deleted = ?
                LIMIT 1
            """
            params = (conversation_id, False, False)
        try:
            cursor = self._db.execute_query(query, params)
            return cursor.fetchone() is not None
        except CharactersRAGDBError as e:
            logger.error(
                'Database error checking system messages for conversation ID {}: {}',
                conversation_id,
                e,
            )
            raise

    # ------------------------------------------------------------------
    # Message update
    # ------------------------------------------------------------------

    def update_message(self, message_id: str, update_data: dict[str, Any], expected_version: int) -> bool | None:
        """
        Updates an existing message using optimistic locking.

        Succeeds if `expected_version` matches the current database version.
        `version` is incremented, `last_modified` updated, and `client_id` set.
        Updatable fields from `update_data`: 'content', 'ranking', 'parent_message_id'.
        Image data can also be updated: 'image_data' and 'image_mime_type'.
        If 'image_data' is set to `None` in `update_data`, both 'image_data' and
        'image_mime_type' columns will be set to NULL in the database.
        Other fields in `update_data` are ignored. `update_data` must not be empty.

        FTS updates (`messages_fts` for content changes) and `sync_log` entries
        are handled by SQL triggers.

        Args:
            message_id: The UUID of the message to update.
            update_data: Dictionary with fields to update. Must not be empty.
                         If 'image_data' is updated, 'image_mime_type' should also be
                         provided, unless 'image_data' is set to None.
            expected_version: The client's expected version of the record.

        Returns:
            True if the update was successful.

        Raises:
            InputError: If `update_data` is empty.
            ConflictError: If the message is not found, is soft-deleted, or if `expected_version`
                           does not match the current database version.
            CharactersRAGDBError: For database integrity errors (e.g., invalid `parent_message_id`)
                                  or other database issues.
        """
        if not update_data:
            raise InputError("No data provided for message update.")  # noqa: TRY003

        now = self._db._get_current_utc_timestamp_iso()
        fields_to_update_sql = []
        params_for_set_clause = []

        allowed_to_update = ['content', 'ranking', 'parent_message_id', 'image_data', 'image_mime_type']

        # Special handling for clearing image
        if 'image_data' in update_data and update_data['image_data'] is None:
            fields_to_update_sql.append("image_data = NULL")
            fields_to_update_sql.append("image_mime_type = NULL")
            # Remove these keys from update_data to avoid processing them again
            # in the loop if they were explicitly set to None
            # This isn't strictly necessary with current loop logic but good for clarity
            update_data.pop('image_data', None)
            update_data.pop('image_mime_type', None)

        for key, value in update_data.items():
            if key in allowed_to_update:
                fields_to_update_sql.append(f"{key} = ?")
                params_for_set_clause.append(value)
            elif key not in ['id', 'conversation_id', 'sender', 'timestamp', 'last_modified', 'version', 'client_id', 'deleted']:
                logger.warning(
                    f"Attempted to update immutable or unknown field '{key}' in message ID {message_id}, skipping.")

        if not fields_to_update_sql:  # If only image was cleared, this list might be empty now if no other fields
            logger.info(f"No updatable content fields provided for message ID {message_id}, but metadata will be updated if version matches.")
            # Proceed to metadata update; SQL query will be constructed accordingly

        next_version_val = expected_version + 1

        current_fields_to_update_sql = list(fields_to_update_sql)
        current_params_for_set_clause = list(params_for_set_clause)

        current_fields_to_update_sql.extend(["last_modified = ?", "version = ?", "client_id = ?"])
        current_params_for_set_clause.extend([now, next_version_val, self._db.client_id])

        where_values = [message_id, expected_version]
        final_params_for_execute = tuple(current_params_for_set_clause + where_values)

        query = f"UPDATE messages SET {', '.join(current_fields_to_update_sql)} WHERE id = ? AND version = ? AND deleted = FALSE"  # nosec B608

        try:
            with self._db.transaction() as conn:
                current_db_version = self._db._get_current_db_version(conn, "messages", "id", message_id)

                if current_db_version != expected_version:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Message ID {message_id} update failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="messages", entity_id=message_id
                    )

                cursor = conn.execute(query, final_params_for_execute)

                if cursor.rowcount == 0:
                    check_again_cursor = conn.execute("SELECT version, deleted FROM messages WHERE id = ?",
                                                      (message_id,))
                    final_state = check_again_cursor.fetchone()
                    msg = f"Update for message ID {message_id} (expected v{expected_version}) affected 0 rows."
                    if not final_state:
                        msg = f"Message ID {message_id} disappeared."
                    elif final_state['deleted']:
                        msg = f"Message ID {message_id} was soft-deleted concurrently."
                    elif final_state['version'] != expected_version:
                        msg = f"Message ID {message_id} version changed to {final_state['version']} concurrently."
                    raise ConflictError(msg, entity="messages", entity_id=message_id)  # noqa: TRY301

                logger.info(
                    f"Updated message ID {message_id} from version {expected_version} to version {next_version_val}. Fields updated: {fields_to_update_sql if fields_to_update_sql else 'None'}")
                return True
        except sqlite3.IntegrityError as e:
            logger.error(f"SQLite integrity error updating message ID {message_id} (expected v{expected_version}): {e}",
                         exc_info=True)
            raise CharactersRAGDBError(f"Database integrity error updating message: {e}") from e  # noqa: TRY003
        except ConflictError:
            raise
        except InputError:  # Should not be raised from here directly, but for completeness
            raise
        except CharactersRAGDBError as e:
            logger.error(f"Database error updating message ID {message_id} (expected v{expected_version}): {e}",
                         exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Soft delete
    # ------------------------------------------------------------------

    def soft_delete_message(self, message_id: str, expected_version: int) -> bool | None:
        """
        Soft-deletes a message using optimistic locking.

        Sets `deleted` to 1, updates `last_modified`, increments `version`, and sets `client_id`.
        Succeeds if `expected_version` matches the current DB version and the record is active.
        If already soft-deleted, returns True (idempotent).

        FTS updates (removal from `messages_fts`) and `sync_log` entries are handled by SQL triggers.

        Args:
            message_id: The UUID of the message to soft-delete.
            expected_version: The client's expected version of the record.

        Returns:
            True if the soft-delete was successful or if the message was already soft-deleted.

        Raises:
            ConflictError: If not found (and not already deleted), or if active with a version mismatch.
            CharactersRAGDBError: For other database errors.
        """
        now = self._db._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1

        query = "UPDATE messages SET deleted = TRUE, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ? AND deleted = FALSE"
        params = (now, next_version_val, self._db.client_id, message_id, expected_version)

        try:
            with self._db.transaction() as conn:
                try:
                    current_db_version = self._db._get_current_db_version(conn, "messages", "id", message_id)
                except ConflictError:
                    check_status_cursor = conn.execute("SELECT deleted, version FROM messages WHERE id = ?",
                                                       (message_id,))
                    record_status = check_status_cursor.fetchone()
                    if record_status and record_status['deleted']:
                        logger.info(f"Message ID {message_id} already soft-deleted. Success (idempotent).")
                        return True
                    raise  # Re-raise if not found or other conflict

                if current_db_version != expected_version:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Soft delete for Message ID {message_id} failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="messages", entity_id=message_id
                    )

                cursor = conn.execute(query, params)

                if cursor.rowcount == 0:
                    check_again_cursor = conn.execute("SELECT version, deleted FROM messages WHERE id = ?",
                                                      (message_id,))
                    final_state = check_again_cursor.fetchone()
                    msg = f"Soft delete for message ID {message_id} (expected v{expected_version}) affected 0 rows."
                    if not final_state:
                        msg = f"Message ID {message_id} disappeared."
                    elif final_state['deleted']:
                        logger.info(f"Message ID {message_id} was soft-deleted concurrently. Success.")
                        return True
                    elif final_state['version'] != expected_version:
                        msg = f"Message ID {message_id} version changed to {final_state['version']} concurrently."
                    else:
                        msg = f"Soft delete for message ID {message_id} (expected v{expected_version}) affected 0 rows."
                    raise ConflictError(msg, entity="messages", entity_id=message_id)  # noqa: TRY301

                logger.info(
                    f"Soft-deleted message ID {message_id} (was v{expected_version}), new version {next_version_val}.")
                return True
        except ConflictError:
            raise
        except CharactersRAGDBError as e:
            logger.error(f"Database error soft-deleting message ID {message_id} (expected v{expected_version}): {e}",
                         exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Full-text search
    # ------------------------------------------------------------------

    def search_messages_by_content(
        self,
        content_query: str,
        conversation_id: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Searches messages by content using FTS.

        Matches against the 'content' field in `messages_fts`.
        Optionally filters by `conversation_id`. Returns non-deleted messages,
        ordered by relevance (rank).

        Args:
            content_query: The search term for content. Supports FTS query syntax.
            conversation_id: Optional conversation UUID to filter results.
            limit: Maximum number of results. Defaults to 10.
            offset: Number of matching rows to skip. Defaults to 0.

        Returns:
            A list of matching message dictionaries. Can be empty.

        Raises:
            CharactersRAGDBError: For database search errors.
        """
        if self._db.backend_type == BackendType.POSTGRESQL:
            tsquery = FTSQueryTranslator.normalize_query(content_query, 'postgresql')
            if not tsquery:
                logger.debug("Message content query normalized to empty tsquery for input '{}'", content_query)
                return []

            base_query = [
                "SELECT m.*, ts_rank(m.messages_fts_tsv, to_tsquery('english', ?)) AS rank",
                "FROM messages m",
                "WHERE m.deleted = FALSE",
                "AND m.messages_fts_tsv @@ to_tsquery('english', ?)",
            ]
            params_list: list[Any] = [tsquery, tsquery]

            if conversation_id:
                base_query.append("AND m.conversation_id = ?")
                params_list.append(conversation_id)

            base_query.append("ORDER BY rank DESC, m.last_modified DESC")
            base_query.append("LIMIT ? OFFSET ?")
            params_list.extend([limit, offset])

            try:
                cursor = self._db.execute_query("\n".join(base_query), tuple(params_list))
                return [dict(row) for row in cursor.fetchall()]
            except CharactersRAGDBError as exc:
                logger.error("PostgreSQL FTS search failed for messages term '{}': {}", content_query, exc)
                raise

        safe_literal = content_query.replace('"', '""')
        safe_search_term = f'"{safe_literal}"' if '"' in content_query else safe_literal
        base_query = """
                     SELECT m.*
                     FROM messages_fts, messages m
                     WHERE messages_fts.rowid = m.rowid \
                       AND messages_fts MATCH ? \
                       AND m.deleted = FALSE \
                     """
        params_list = [safe_search_term]
        if conversation_id:
            base_query += " AND m.conversation_id = ?"
            params_list.append(conversation_id)

        base_query += " ORDER BY bm25(messages_fts) ASC, m.last_modified DESC LIMIT ? OFFSET ?"
        params_list.extend([limit, offset])

        try:
            cursor = self._db.execute_query(base_query, tuple(params_list))
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError as e:
            logger.error("Error searching messages for content '{}': {}", safe_search_term, e)
            raise

    # ------------------------------------------------------------------
    # Message metadata
    # ------------------------------------------------------------------

    def add_message_metadata(self, message_id: str, tool_calls: Any | None = None, extra: Any | None = None) -> bool:
        """Upsert per-message metadata such as tool calls.

        Stores JSON-serialized metadata in an auxiliary table `message_metadata`.
        The table is created on-demand if missing.
        """
        try:
            self._db._ensure_message_metadata_table()
            if self._db.backend_type == BackendType.SQLITE:
                query = (
                    "INSERT INTO message_metadata(message_id, tool_calls_json, extra_json, last_modified) "
                    "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(message_id) DO UPDATE SET tool_calls_json=excluded.tool_calls_json, "
                    "extra_json=excluded.extra_json, last_modified=CURRENT_TIMESTAMP"
                )
                self._db.execute_query(
                    query,
                    (
                        message_id,
                        json.dumps(tool_calls) if tool_calls is not None else None,
                        json.dumps(extra) if extra is not None else None,
                    ),
                    commit=True,
                )
                return True

            upsert = (
                "INSERT INTO message_metadata(message_id, tool_calls_json, extra_json, last_modified) "
                "VALUES (%s, %s, %s, NOW()) "
                "ON CONFLICT (message_id) DO UPDATE SET tool_calls_json = EXCLUDED.tool_calls_json, "
                "extra_json = EXCLUDED.extra_json, last_modified = NOW()"
            )
            self._db.backend.execute(
                upsert,
                (
                    message_id,
                    json.dumps(tool_calls) if tool_calls is not None else None,
                    json.dumps(extra) if extra is not None else None,
                ),
            )
            return True  # noqa: TRY300
        except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"add_message_metadata failed for message {message_id}: {e}")
            return False

    def get_message_metadata(self, message_id: str) -> dict[str, Any] | None:
        """Fetch metadata for a message if present."""
        try:
            self._db._ensure_message_metadata_table()
            if self._db.backend_type == BackendType.SQLITE:
                cursor = self._db.execute_query(
                    "SELECT tool_calls_json, extra_json, last_modified FROM message_metadata WHERE message_id = ?",
                    (message_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                tc, ex, lm = row
            else:
                result = self._db.backend.execute(
                    "SELECT tool_calls_json, extra_json, last_modified FROM message_metadata WHERE message_id = %s",
                    (message_id,)
                )
                r = result.fetchone()
                if not r:
                    return None
                tc, ex, lm = r
            return {
                "tool_calls": json.loads(tc) if tc else None,
                "extra": json.loads(ex) if ex else None,
                "last_modified": lm,
            }
        except _CHACHA_NONCRITICAL_EXCEPTIONS:
            return None

    def get_message_metadata_map(self, message_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch metadata for multiple messages in a single query."""
        if not message_ids:
            return {}

        try:
            self._db._ensure_message_metadata_table()
            if self._db.backend_type == BackendType.SQLITE:
                placeholders = ",".join(["?"] * len(message_ids))
                query = (
                    "SELECT message_id, tool_calls_json, extra_json, last_modified "
                    f"FROM message_metadata WHERE message_id IN ({placeholders})"  # nosec B608
                )
                cursor = self._db.execute_query(query, tuple(message_ids))
                rows = cursor.fetchall()
            else:
                placeholders = ",".join(["%s"] * len(message_ids))
                query = (
                    "SELECT message_id, tool_calls_json, extra_json, last_modified "
                    f"FROM message_metadata WHERE message_id IN ({placeholders})"  # nosec B608
                )
                result = self._db.backend.execute(query, tuple(message_ids))
                rows = result.fetchall()

            metadata_by_message_id: dict[str, dict[str, Any]] = {}
            for row in rows:
                try:
                    message_id = str(row["message_id"])
                    tc = row["tool_calls_json"]
                    ex = row["extra_json"]
                    lm = row["last_modified"]
                except _CHACHA_NONCRITICAL_EXCEPTIONS:
                    message_id = str(row[0])
                    tc = row[1]
                    ex = row[2]
                    lm = row[3]
                metadata_by_message_id[message_id] = {
                    "tool_calls": json.loads(tc) if tc else None,
                    "extra": json.loads(ex) if ex else None,
                    "last_modified": lm,
                }
            return metadata_by_message_id
        except _CHACHA_NONCRITICAL_EXCEPTIONS:
            return {}

    def set_message_metadata_extra(self, message_id: str, extra: dict[str, Any], merge: bool = True) -> bool:
        """Set or merge structured extra metadata for a message.

        Expected shape for `extra`:
          {
            "tool_results": { "<tool_call_id>": <any-json-serializable> },
            ... other namespaced keys ...,
            "version": 1
          }

        If merge=True and existing extra exists, perform a shallow merge; nested maps like
        tool_results are merged key-wise.
        """
        try:
            current = self.get_message_metadata(message_id) or {}
            current_extra = current.get('extra') or {}
            if merge and isinstance(current_extra, dict) and isinstance(extra, dict):
                merged = dict(current_extra)
                # Merge tool_results specially
                tr_existing = merged.get('tool_results') if isinstance(merged.get('tool_results'), dict) else {}
                tr_incoming = extra.get('tool_results') if isinstance(extra.get('tool_results'), dict) else {}
                if tr_existing or tr_incoming:
                    merged['tool_results'] = {**tr_existing, **tr_incoming}
                # Merge top-level keys (favor incoming)
                for k, v in extra.items():
                    if k == 'tool_results':
                        continue
                    merged[k] = v
                new_extra = merged
            else:
                new_extra = extra
            return self.add_message_metadata(message_id, tool_calls=current.get('tool_calls'), extra=new_extra)
        except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"set_message_metadata_extra failed for {message_id}: {e}")
            return False

    def set_message_rag_context(
        self,
        message_id: str,
        rag_context: dict[str, Any],
        merge: bool = True
    ) -> bool:
        """
        Store RAG context (citations, retrieved documents, search settings) with a message.

        This persists RAG search results and citations in message_metadata.extra_json
        under the 'rag_context' key for later retrieval and export.

        Args:
            message_id: The message ID to attach RAG context to
            rag_context: Dict containing:
                - search_query: The original search query
                - search_mode: Search mode used (fts/vector/hybrid)
                - settings_snapshot: Key RAG settings used
                - retrieved_documents: List of retrieved docs with scores/excerpts
                - generated_answer: AI-generated answer (if any)
                - citations: Citation metadata
                - claims_verified: Verification results (if any)
                - timestamp: ISO timestamp
                - feedback_id: Analytics ID
            merge: If True, merge with existing extra data; if False, replace entire extra

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get current metadata to preserve other extra fields
            current = self.get_message_metadata(message_id) or {}
            current_extra = current.get('extra') or {}

            if merge and isinstance(current_extra, dict):
                # Merge: preserve existing extra fields, update rag_context
                new_extra = dict(current_extra)
                new_extra['rag_context'] = rag_context
            else:
                # Replace: only keep rag_context
                new_extra = {'rag_context': rag_context}

            return self.add_message_metadata(
                message_id,
                tool_calls=current.get('tool_calls'),
                extra=new_extra
            )
        except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"set_message_rag_context failed for {message_id}: {e}")
            return False

    def get_message_rag_context(self, message_id: str) -> dict[str, Any] | None:
        """
        Retrieve RAG context stored with a message.

        Returns the rag_context dict from message_metadata.extra_json,
        or None if no RAG context is stored.
        """
        try:
            metadata = self.get_message_metadata(message_id)
            if not metadata:
                return None
            extra = metadata.get('extra')
            if not isinstance(extra, dict):
                return None
            return extra.get('rag_context')
        except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"get_message_rag_context failed for {message_id}: {e}")
            return None

    def get_messages_with_rag_context(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
        include_rag_context: bool = True
    ) -> list[dict[str, Any]]:
        """
        Retrieve messages for a conversation with optional RAG context attached.

        This is optimized for the Knowledge QA page to load conversation history
        with full citation data.

        Args:
            conversation_id: The conversation to fetch messages from
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            include_rag_context: If True, attach rag_context to each message

        Returns:
            List of message dicts, each optionally including 'rag_context' key
        """
        try:
            messages = self.get_messages_for_conversation(
                conversation_id,
                limit=limit,
                offset=offset
            )

            if not include_rag_context:
                return messages

            # Attach RAG context to each message
            for msg in messages:
                msg_id = msg.get('id')
                if msg_id:
                    rag_context = self.get_message_rag_context(msg_id)
                    if rag_context:
                        msg['rag_context'] = rag_context

            return messages  # noqa: TRY300
        except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"get_messages_with_rag_context failed for conversation {conversation_id}: {e}")
            return []

    # ------------------------------------------------------------------
    # Count / query helpers
    # ------------------------------------------------------------------

    def count_messages_for_conversation(self, conversation_id: str, include_deleted: bool = False) -> int:
        """
        Count messages for a conversation, ensuring the parent conversation is active.

        Args:
            conversation_id: Conversation UUID
            include_deleted: If True, include soft-deleted messages

        Returns:
            Integer count of messages.

        Raises:
            CharactersRAGDBError on database failure.
        """
        base_query = (
            "SELECT COUNT(1) FROM messages m "
            "JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.conversation_id = ? AND c.deleted = FALSE"
        )
        params = [conversation_id]
        if not include_deleted:
            base_query += " AND m.deleted = FALSE"
        try:
            cursor = self._db.execute_query(base_query, tuple(params))
            row = cursor.fetchone()
            # row may be tuple or dict depending on connection row factory
            if row is None:
                return 0
            try:
                return int(row[0])
            except _CHACHA_NONCRITICAL_EXCEPTIONS:
                return int(row.get("COUNT(1)") or row.get("count") or 0)
        except CharactersRAGDBError as e:
            logger.error(f"Database error counting messages for conversation {conversation_id}: {e}")
            raise

    def count_messages_for_conversations(
        self,
        conversation_ids: list[str],
        include_deleted: bool = False,
    ) -> dict[str, int]:
        """
        Count messages for multiple conversations in a single query.

        Args:
            conversation_ids: List of conversation UUIDs.
            include_deleted: If True, include soft-deleted messages.

        Returns:
            Mapping of conversation_id -> message count.
        """
        if not conversation_ids:
            return {}
        placeholders = ",".join(["?"] * len(conversation_ids))
        base_query = (
            f"SELECT m.conversation_id, COUNT(1) as cnt "  # nosec B608
            f"FROM messages m "
            f"JOIN conversations c ON m.conversation_id = c.id "
            f"WHERE m.conversation_id IN ({placeholders}) AND c.deleted = FALSE"
        )
        if not include_deleted:
            base_query += " AND m.deleted = FALSE"
        base_query += " GROUP BY m.conversation_id"
        try:
            cursor = self._db.execute_query(base_query, tuple(conversation_ids))
            rows = cursor.fetchall()
            result: dict[str, int] = dict.fromkeys(conversation_ids, 0)
            for row in rows:
                if isinstance(row, dict):
                    conv_id = row.get("conversation_id")
                    cnt = row.get("cnt") or row.get("COUNT(1)") or 0
                else:
                    conv_id = row[0]
                    cnt = row[1]
                if conv_id is not None:
                    result[str(conv_id)] = int(cnt or 0)
            return result  # noqa: TRY300
        except CharactersRAGDBError as e:
            logger.error("Database error counting messages for conversations: {}", e)
            raise

    def get_latest_message_for_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Fetch the most recent non-deleted message for a conversation."""
        query = (
            "SELECT m.id, m.timestamp, m.content, m.sender "
            "FROM messages m JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.conversation_id = ? AND m.deleted = FALSE AND c.deleted = FALSE "
            "ORDER BY m.timestamp DESC LIMIT 1"
        )
        try:
            cursor = self._db.execute_query(query, (conversation_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row) if isinstance(row, dict) else {
                "id": row[0],
                "timestamp": row[1],
                "content": row[2],
                "sender": row[3],
            }
        except CharactersRAGDBError as exc:
            logger.error("Database error fetching latest message for conversation {}: {}", conversation_id, exc)
            raise

    def count_messages_since(
        self,
        conversation_id: str,
        since_message_id: str | None,
    ) -> int:
        """Count messages after the given message_id within a conversation."""
        if not since_message_id:
            return self.count_messages_for_conversation(conversation_id)

        try:
            since_message = self.get_message_by_id(since_message_id)
        except CharactersRAGDBError:
            return self.count_messages_for_conversation(conversation_id)

        if not since_message:
            return self.count_messages_for_conversation(conversation_id)

        since_timestamp = since_message.get("timestamp")
        if not since_timestamp:
            return self.count_messages_for_conversation(conversation_id)

        query = (
            "SELECT COUNT(1) FROM messages m "
            "JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.conversation_id = ? AND m.deleted = FALSE AND c.deleted = FALSE "
            "AND m.timestamp > ?"
        )
        try:
            cursor = self._db.execute_query(query, (conversation_id, since_timestamp))
            row = cursor.fetchone()
            if row is None:
                return 0
            try:
                return int(row[0])
            except _CHACHA_NONCRITICAL_EXCEPTIONS:
                return int(row.get("COUNT(1)") or row.get("count") or 0)
        except CharactersRAGDBError as exc:
            logger.error("Database error counting messages after {}: {}", since_message_id, exc)
            raise
