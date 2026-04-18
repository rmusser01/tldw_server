from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDBError,
    ConflictError,
    InputError,
    _CHACHA_NONCRITICAL_EXCEPTIONS,
    logger,
)
from tldw_Server_API.app.core.config import settings

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class MessageStore:
    """Focused persistence seam for ChaCha message lifecycle behavior."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def _ensure_message_metadata_table(self) -> None:
        """Ensure the message_metadata table exists for the active backend."""
        if self._db.backend_type == BackendType.SQLITE:
            self._db.execute_query(
                """
                CREATE TABLE IF NOT EXISTS message_metadata(
                  message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
                  tool_calls_json TEXT,
                  extra_json TEXT,
                  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """,
                script=False,
                commit=True,
            )
            return

        if self._db.backend_type == BackendType.POSTGRESQL:
            self._db.backend.execute(
                """
                CREATE TABLE IF NOT EXISTS message_metadata(
                  message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
                  tool_calls_json TEXT,
                  extra_json TEXT,
                  last_modified TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """
            )
            return

        raise NotImplementedError(
            f"message_metadata table creation not supported for backend {self._db.backend_type.value}"
        )

    def add_message_metadata(
        self,
        message_id: str,
        tool_calls: Any | None = None,
        extra: Any | None = None,
    ) -> bool:
        """Upsert per-message metadata such as tool calls."""
        try:
            self._ensure_message_metadata_table()
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
            return True
        except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"add_message_metadata failed for message {message_id}: {exc}")
            return False

    def get_message_metadata(self, message_id: str) -> dict[str, Any] | None:
        """Fetch metadata for a message if present."""
        try:
            self._ensure_message_metadata_table()
            if self._db.backend_type == BackendType.SQLITE:
                cursor = self._db.execute_query(
                    "SELECT tool_calls_json, extra_json, last_modified FROM message_metadata WHERE message_id = ?",
                    (message_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                tool_calls_json, extra_json, last_modified = row
            else:
                result = self._db.backend.execute(
                    "SELECT tool_calls_json, extra_json, last_modified FROM message_metadata WHERE message_id = %s",
                    (message_id,),
                )
                row = result.fetchone()
                if not row:
                    return None
                tool_calls_json, extra_json, last_modified = row
            return {
                "tool_calls": json.loads(tool_calls_json) if tool_calls_json else None,
                "extra": json.loads(extra_json) if extra_json else None,
                "last_modified": last_modified,
            }
        except _CHACHA_NONCRITICAL_EXCEPTIONS:
            return None

    def get_message_metadata_map(self, message_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch metadata for multiple messages in a single query."""
        if not message_ids:
            return {}

        try:
            self._ensure_message_metadata_table()
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
                    tool_calls_json = row["tool_calls_json"]
                    extra_json = row["extra_json"]
                    last_modified = row["last_modified"]
                except _CHACHA_NONCRITICAL_EXCEPTIONS:
                    message_id = str(row[0])
                    tool_calls_json = row[1]
                    extra_json = row[2]
                    last_modified = row[3]
                metadata_by_message_id[message_id] = {
                    "tool_calls": json.loads(tool_calls_json) if tool_calls_json else None,
                    "extra": json.loads(extra_json) if extra_json else None,
                    "last_modified": last_modified,
                }
            return metadata_by_message_id
        except _CHACHA_NONCRITICAL_EXCEPTIONS:
            return {}

    def get_conversation_citations(self, conversation_id: str) -> list[dict[str, Any]]:
        """Retrieve all citations from a conversation's messages."""
        try:
            messages = self.get_messages_for_conversation(conversation_id, limit=1000)
            citations_by_id: dict[str, dict[str, Any]] = {}

            for message in messages:
                message_id = message.get("id")
                if not message_id:
                    continue

                rag_context = self._db.get_message_rag_context(message_id)
                if not rag_context:
                    continue

                retrieved_docs = rag_context.get("retrieved_documents", [])
                for document in retrieved_docs:
                    doc_id = document.get("id") or document.get("chunk_id") or f"anon_{len(citations_by_id)}"
                    if doc_id not in citations_by_id:
                        citations_by_id[doc_id] = {
                            **document,
                            "message_ids": [message_id],
                            "first_cited_at": message.get("timestamp"),
                        }
                    elif message_id not in citations_by_id[doc_id]["message_ids"]:
                        citations_by_id[doc_id]["message_ids"].append(message_id)

            return list(citations_by_id.values())
        except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"get_conversation_citations failed for conversation {conversation_id}: {exc}")
            return []

    def count_messages_for_conversation(self, conversation_id: str, include_deleted: bool = False) -> int:
        """Count messages for a conversation while ensuring the parent conversation is active."""
        base_query = (
            "SELECT COUNT(1) FROM messages m "
            "JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.conversation_id = ? AND c.deleted = 0"
        )
        params = [conversation_id]
        if not include_deleted:
            base_query += " AND m.deleted = 0"
        try:
            cursor = self._db.execute_query(base_query, tuple(params))
            row = cursor.fetchone()
            if row is None:
                return 0
            try:
                return int(row[0])
            except _CHACHA_NONCRITICAL_EXCEPTIONS:
                return int(row.get("COUNT(1)") or row.get("count") or 0)
        except CharactersRAGDBError as exc:
            logger.error(f"Database error counting messages for conversation {conversation_id}: {exc}")
            raise

    def count_messages_for_conversations(
        self,
        conversation_ids: list[str],
        include_deleted: bool = False,
    ) -> dict[str, int]:
        """Count messages for multiple conversations in a single query."""
        if not conversation_ids:
            return {}

        placeholders = ",".join(["?"] * len(conversation_ids))
        base_query = (
            f"SELECT m.conversation_id, COUNT(1) as cnt "  # nosec B608
            "FROM messages m "
            "JOIN conversations c ON m.conversation_id = c.id "
            f"WHERE m.conversation_id IN ({placeholders}) AND c.deleted = 0"  # nosec B608
        )
        if not include_deleted:
            base_query += " AND m.deleted = 0"
        base_query += " GROUP BY m.conversation_id"
        try:
            cursor = self._db.execute_query(base_query, tuple(conversation_ids))
            rows = cursor.fetchall()
            result: dict[str, int] = dict.fromkeys(conversation_ids, 0)
            for row in rows:
                if isinstance(row, dict):
                    conversation_id = row.get("conversation_id")
                    count = row.get("cnt") or row.get("COUNT(1)") or 0
                else:
                    conversation_id = row[0]
                    count = row[1]
                if conversation_id is not None:
                    result[str(conversation_id)] = int(count or 0)
            return result
        except CharactersRAGDBError as exc:
            logger.error("Database error counting messages for conversations: {}", exc)
            raise

    def get_latest_message_for_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Fetch the most recent non-deleted message for a conversation."""
        query = (
            "SELECT m.id, m.timestamp, m.content, m.sender "
            "FROM messages m JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.conversation_id = ? AND m.deleted = 0 AND c.deleted = 0 "
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
            "WHERE m.conversation_id = ? AND m.deleted = 0 AND c.deleted = 0 "
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

    def add_message(self, msg_data: dict[str, Any]) -> str | None:
        """Add a new message to a conversation, optionally with image attachments."""
        images_payload_raw = msg_data.pop("images", None)
        normalized_images: list[tuple[bytes, str]] = []
        if images_payload_raw:
            for entry in images_payload_raw:
                image_bytes: bytes | None = None
                image_mime: str | None = None
                if isinstance(entry, dict):
                    image_bytes = entry.get("data") or entry.get("image_data")
                    image_mime = entry.get("mime") or entry.get("image_mime_type")
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    image_bytes, image_mime = entry[0], entry[1]
                if image_bytes is None or image_mime is None:
                    continue
                if isinstance(image_bytes, memoryview):
                    image_bytes = image_bytes.tobytes()
                normalized_images.append((image_bytes, str(image_mime)))

        try:
            max_image_bytes = int(settings.get("MAX_MESSAGE_IMAGE_BYTES", 5 * 1024 * 1024))
        except _CHACHA_NONCRITICAL_EXCEPTIONS:
            max_image_bytes = 5 * 1024 * 1024

        primary_image = msg_data.get("image_data")
        if isinstance(primary_image, memoryview):
            primary_image = primary_image.tobytes()
        if isinstance(primary_image, (bytes, bytearray)) and len(primary_image) > max_image_bytes:
            raise InputError(
                f"Primary image attachment exceeds maximum size of {max_image_bytes} bytes"
            )  # noqa: TRY003

        for image_bytes, _image_mime in normalized_images:
            if image_bytes is None:
                continue
            if isinstance(image_bytes, memoryview):
                image_bytes = image_bytes.tobytes()
            if isinstance(image_bytes, (bytes, bytearray)) and len(image_bytes) > max_image_bytes:
                raise InputError(
                    f"Message image attachment exceeds maximum size of {max_image_bytes} bytes"
                )  # noqa: TRY003

        message_id = msg_data.get("id") or self._db._generate_uuid()

        required_fields = ["conversation_id", "sender", "content"]
        for field in required_fields:
            if field not in msg_data:
                raise InputError(f"Required field '{field}' is missing for message.")  # noqa: TRY003
        if not msg_data.get("content") and not msg_data.get("image_data") and not normalized_images:
            raise InputError("Message must have text content or image data.")  # noqa: TRY003
        if msg_data.get("image_data") and not msg_data.get("image_mime_type"):
            raise InputError("image_mime_type is required if image_data is provided.")  # noqa: TRY003

        if normalized_images and not msg_data.get("image_data"):
            first_bytes, first_mime = normalized_images[0]
            msg_data["image_data"] = first_bytes
            msg_data["image_mime_type"] = first_mime

        client_id = msg_data.get("client_id") or self._db.client_id
        if not client_id:
            raise InputError("Client ID is required for message.")  # noqa: TRY003

        now = self._db._get_current_utc_timestamp_iso()
        timestamp = msg_data.get("timestamp") or now

        query = """
                INSERT INTO messages (id, conversation_id, parent_message_id, sender, content,
                                      image_data, image_mime_type,
                                      timestamp, ranking, last_modified, client_id, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
        if self._db.backend_type == BackendType.POSTGRESQL:
            params = (
                message_id,
                msg_data["conversation_id"],
                msg_data.get("parent_message_id"),
                msg_data["sender"],
                msg_data.get("content", ""),
                msg_data.get("image_data"),
                msg_data.get("image_mime_type"),
                timestamp,
                msg_data.get("ranking"),
                now,
                client_id,
                1,
                False,
            )
        else:
            params = (
                message_id,
                msg_data["conversation_id"],
                msg_data.get("parent_message_id"),
                msg_data["sender"],
                msg_data.get("content", ""),
                msg_data.get("image_data"),
                msg_data.get("image_mime_type"),
                timestamp,
                msg_data.get("ranking"),
                now,
                client_id,
                1,
                0,
            )
        try:
            with self._db.transaction():
                conversation_cursor = self._db.execute_query(
                    "SELECT 1 FROM conversations WHERE id = ? AND deleted = 0",
                    (msg_data["conversation_id"],),
                )
                if not conversation_cursor.fetchone():
                    raise InputError(
                        f"Cannot add message: Conversation ID '{msg_data['conversation_id']}' not found or deleted."
                    )  # noqa: TRY003, TRY301
                self._db.execute_query(query, params)
                if normalized_images:
                    self._insert_message_images(message_id, normalized_images)
            logger.info(
                "Added message ID: {} to conversation {} (Images stored: {}).",
                message_id,
                msg_data["conversation_id"],
                len(normalized_images) if normalized_images else ("Yes" if msg_data.get("image_data") else "No"),
            )
            return message_id
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: messages.id" in str(exc):
                raise ConflictError(
                    f"Message with ID '{message_id}' already exists.",
                    entity="messages",
                    entity_id=message_id,
                ) from exc  # noqa: TRY003
            raise CharactersRAGDBError(f"Database integrity error adding message: {exc}") from exc  # noqa: TRY003
        except InputError:
            raise
        except CharactersRAGDBError as exc:
            logger.error(f"Database error adding message: {exc}")
            raise

    def _insert_message_images(self, message_id: str, images: list[tuple[bytes, str]]) -> None:
        """Insert or replace message images for the given message."""
        if not images:
            return
        params: list[tuple[str, int, bytes, str]] = []
        for idx, (image_bytes, image_mime) in enumerate(images):
            if image_bytes is None or image_mime is None:
                continue
            if isinstance(image_bytes, memoryview):
                image_bytes = image_bytes.tobytes()
            params.append((message_id, idx, image_bytes, image_mime))
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
                record = dict(row) if isinstance(row, dict) else {
                    columns[idx]: row[idx] for idx in range(len(columns))
                }
                image_bytes = record.get("image_data")
                if isinstance(image_bytes, memoryview):
                    record["image_data"] = image_bytes.tobytes()
                images.append(record)
            return images
        except CharactersRAGDBError as exc:
            logger.error(f"Failed to fetch images for message {message_id}: {exc}")
            return []

    def get_message_conversation_id(self, message_id: str) -> str | None:
        """Return the conversation_id for a message if it exists and is not deleted."""
        query = "SELECT conversation_id FROM messages WHERE id = ? AND deleted = 0"
        try:
            cursor = self._db.execute_query(query, (message_id,))
            row = cursor.fetchone()
            if not row:
                return None
            if isinstance(row, dict):
                return row.get("conversation_id")
            return row[0] if row else None
        except CharactersRAGDBError as exc:
            logger.error(f"Database error fetching conversation_id for message {message_id}: {exc}")
            raise

    def get_message_by_id(self, message_id: str) -> dict[str, Any] | None:
        """Retrieve a specific non-deleted message by UUID."""
        query = (
            "SELECT m.id, m.conversation_id, m.parent_message_id, m.sender, m.content, "
            "m.image_data, m.image_mime_type, m.timestamp, m.ranking, m.last_modified, "
            "m.version, m.client_id, m.deleted "
            "FROM messages m "
            "JOIN conversations c ON c.id = m.conversation_id "
            "WHERE m.id = ? AND m.deleted = 0 AND c.deleted = 0"
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
            image_blob = record.get("image_data")
            if isinstance(image_blob, memoryview):
                record["image_data"] = image_blob.tobytes()
            record["images"] = self.get_message_images(message_id)
            return record
        except CharactersRAGDBError as exc:
            logger.error(f"Database error fetching message ID {message_id}: {exc}")
            raise

    def get_messages_for_conversation(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
        order_by_timestamp: str = "ASC",
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """List messages for a specific conversation."""
        if order_by_timestamp.upper() not in ["ASC", "DESC"]:
            raise InputError("order_by_timestamp must be 'ASC' or 'DESC'.")  # noqa: TRY003

        delete_clause = "" if include_deleted else "AND m.deleted = 0"
        query = """
            SELECT m.id, m.conversation_id, m.parent_message_id, m.sender, m.content,
                   m.image_data, m.image_mime_type, m.timestamp, m.ranking,
                   m.last_modified, m.version, m.client_id, m.deleted
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.conversation_id = ?
              {delete_clause}
              AND c.deleted = 0
            ORDER BY m.timestamp {order_by_timestamp}
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608
        try:
            cursor = self._db.execute_query(query, (conversation_id, limit, offset))
            raw_rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            results: list[dict[str, Any]] = []
            for row in raw_rows:
                record = dict(row) if isinstance(row, dict) else {
                    columns[idx]: row[idx] for idx in range(len(columns))
                }
                image_blob = record.get("image_data")
                if isinstance(image_blob, memoryview):
                    record["image_data"] = image_blob.tobytes()
                record["images"] = self.get_message_images(record["id"])
                results.append(record)
            return results
        except CharactersRAGDBError as exc:
            logger.error(f"Database error fetching messages for conversation ID {conversation_id}: {exc}")
            raise

    def count_root_messages_for_conversation(self, conversation_id: str) -> int:
        """Count root (parentless) messages for a conversation."""
        query = (
            "SELECT COUNT(1) FROM messages m "
            "JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.conversation_id = ? AND m.parent_message_id IS NULL "
            "AND m.deleted = 0 AND c.deleted = 0"
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
        except CharactersRAGDBError as exc:
            logger.error("Database error counting root messages for conversation {}: {}", conversation_id, exc)
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
              AND m.deleted = 0
              AND c.deleted = 0
            ORDER BY m.timestamp {order_by_timestamp}
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608
        try:
            cursor = self._db.execute_query(query, (conversation_id, limit, offset))
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            results: list[dict[str, Any]] = []
            for row in rows:
                record = dict(row) if isinstance(row, dict) else {
                    columns[idx]: row[idx] for idx in range(len(columns))
                }
                results.append(record)
            return results
        except CharactersRAGDBError as exc:
            logger.error("Database error fetching root messages for conversation {}: {}", conversation_id, exc)
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
              AND m.deleted = 0
              AND c.deleted = 0
            ORDER BY m.timestamp {order_by_timestamp}
        """.format_map(locals())  # nosec B608
        params = [conversation_id, *parent_ids]
        try:
            cursor = self._db.execute_query(query, tuple(params))
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            results: list[dict[str, Any]] = []
            for row in rows:
                record = dict(row) if isinstance(row, dict) else {
                    columns[idx]: row[idx] for idx in range(len(columns))
                }
                results.append(record)
            return results
        except CharactersRAGDBError as exc:
            logger.error(
                "Database error fetching child messages for conversation {}: {}",
                conversation_id,
                exc,
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
        except CharactersRAGDBError as exc:
            logger.error(
                "Database error checking system messages for conversation ID {}: {}",
                conversation_id,
                exc,
            )
            raise

    def update_message(self, message_id: str, update_data: dict[str, Any], expected_version: int) -> bool | None:
        """Update an existing message using optimistic locking."""
        if not update_data:
            raise InputError("No data provided for message update.")  # noqa: TRY003

        now = self._db._get_current_utc_timestamp_iso()
        fields_to_update_sql = []
        params_for_set_clause = []
        allowed_to_update = ["content", "ranking", "parent_message_id", "image_data", "image_mime_type"]

        if "image_data" in update_data and update_data["image_data"] is None:
            fields_to_update_sql.append("image_data = NULL")
            fields_to_update_sql.append("image_mime_type = NULL")
            update_data.pop("image_data", None)
            update_data.pop("image_mime_type", None)

        for key, value in update_data.items():
            if key in allowed_to_update:
                fields_to_update_sql.append(f"{key} = ?")
                params_for_set_clause.append(value)
            elif key not in [
                "id",
                "conversation_id",
                "sender",
                "timestamp",
                "last_modified",
                "version",
                "client_id",
                "deleted",
            ]:
                logger.warning(
                    f"Attempted to update immutable or unknown field '{key}' in message ID {message_id}, skipping."
                )

        if not fields_to_update_sql:
            logger.info(
                f"No updatable content fields provided for message ID {message_id}, but metadata will be updated if version matches."
            )

        next_version_value = expected_version + 1
        current_fields_to_update_sql = list(fields_to_update_sql)
        current_params_for_set_clause = list(params_for_set_clause)
        current_fields_to_update_sql.extend(["last_modified = ?", "version = ?", "client_id = ?"])
        current_params_for_set_clause.extend([now, next_version_value, self._db.client_id])

        where_values = [message_id, expected_version]
        final_params_for_execute = tuple(current_params_for_set_clause + where_values)
        query = (
            f"UPDATE messages SET {', '.join(current_fields_to_update_sql)} "  # nosec B608
            "WHERE id = ? AND version = ? AND deleted = 0"
        )

        try:
            with self._db.transaction() as conn:
                current_db_version = self._db._get_current_db_version(conn, "messages", "id", message_id)

                if current_db_version != expected_version:
                    raise ConflictError(
                        f"Message ID {message_id} update failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="messages",
                        entity_id=message_id,
                    )  # noqa: TRY003, TRY301

                cursor = conn.execute(query, final_params_for_execute)

                if cursor.rowcount == 0:
                    check_again_cursor = conn.execute(
                        "SELECT version, deleted FROM messages WHERE id = ?",
                        (message_id,),
                    )
                    final_state = check_again_cursor.fetchone()
                    message = f"Update for message ID {message_id} (expected v{expected_version}) affected 0 rows."
                    if not final_state:
                        message = f"Message ID {message_id} disappeared."
                    elif final_state["deleted"]:
                        message = f"Message ID {message_id} was soft-deleted concurrently."
                    elif final_state["version"] != expected_version:
                        message = (
                            f"Message ID {message_id} version changed to {final_state['version']} concurrently."
                        )
                    raise ConflictError(message, entity="messages", entity_id=message_id)  # noqa: TRY301

                logger.info(
                    "Updated message ID {} from version {} to version {}. Fields updated: {}",
                    message_id,
                    expected_version,
                    next_version_value,
                    fields_to_update_sql if fields_to_update_sql else "None",
                )
                return True
        except sqlite3.IntegrityError as exc:
            logger.error(
                f"SQLite integrity error updating message ID {message_id} (expected v{expected_version}): {exc}",
                exc_info=True,
            )
            raise CharactersRAGDBError(f"Database integrity error updating message: {exc}") from exc  # noqa: TRY003
        except ConflictError:
            raise
        except InputError:
            raise
        except CharactersRAGDBError as exc:
            logger.error(
                f"Database error updating message ID {message_id} (expected v{expected_version}): {exc}",
                exc_info=True,
            )
            raise

    def soft_delete_message(self, message_id: str, expected_version: int) -> bool | None:
        """Soft-delete a message using optimistic locking."""
        now = self._db._get_current_utc_timestamp_iso()
        next_version_value = expected_version + 1

        query = (
            "UPDATE messages SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
            "WHERE id = ? AND version = ? AND deleted = 0"
        )
        params = (now, next_version_value, self._db.client_id, message_id, expected_version)

        try:
            with self._db.transaction() as conn:
                try:
                    current_db_version = self._db._get_current_db_version(conn, "messages", "id", message_id)
                except ConflictError:
                    check_status_cursor = conn.execute(
                        "SELECT deleted, version FROM messages WHERE id = ?",
                        (message_id,),
                    )
                    record_status = check_status_cursor.fetchone()
                    if record_status and record_status["deleted"]:
                        logger.info(f"Message ID {message_id} already soft-deleted. Success (idempotent).")
                        return True
                    raise

                if current_db_version != expected_version:
                    raise ConflictError(
                        f"Soft delete for Message ID {message_id} failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="messages",
                        entity_id=message_id,
                    )  # noqa: TRY003, TRY301

                cursor = conn.execute(query, params)

                if cursor.rowcount == 0:
                    check_again_cursor = conn.execute(
                        "SELECT version, deleted FROM messages WHERE id = ?",
                        (message_id,),
                    )
                    final_state = check_again_cursor.fetchone()
                    message = (
                        f"Soft delete for message ID {message_id} (expected v{expected_version}) affected 0 rows."
                    )
                    if not final_state:
                        message = f"Message ID {message_id} disappeared."
                    elif final_state["deleted"]:
                        logger.info(f"Message ID {message_id} was soft-deleted concurrently. Success.")
                        return True
                    elif final_state["version"] != expected_version:
                        message = (
                            f"Message ID {message_id} version changed to {final_state['version']} concurrently."
                        )
                    raise ConflictError(message, entity="messages", entity_id=message_id)  # noqa: TRY301

                logger.info(
                    "Soft-deleted message ID {} (was v{}), new version {}.",
                    message_id,
                    expected_version,
                    next_version_value,
                )
                return True
        except ConflictError:
            raise
        except CharactersRAGDBError as exc:
            logger.error(
                f"Database error soft-deleting message ID {message_id} (expected v{expected_version}): {exc}",
                exc_info=True,
            )
            raise
