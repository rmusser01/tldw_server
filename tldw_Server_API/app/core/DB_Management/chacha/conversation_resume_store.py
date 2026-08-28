"""Persistence seam for immutable character-conversation resume state."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    BehaviorSnapshotV1,
    build_behavior_snapshot,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError
from tldw_Server_API.app.core.DB_Management.db_errors import NotFoundError

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


_SNAPSHOT_COLUMNS = (
    "status",
    "schema_version",
    "canonical_json",
    "digest",
    "size_bytes",
    "created_at",
)
_RESUME_SETTINGS_KEY = "roleplayResumeV1"


class ConversationResumeStore:
    """Store immutable snapshots and coherent conversation resume fences."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    @staticmethod
    def _row_to_dict(row: Any, columns: tuple[str, ...]) -> dict[str, Any]:
        if isinstance(row, dict):
            return dict(row)
        mapping = getattr(row, "_mapping", None)
        if mapping is not None:
            return dict(mapping)
        try:
            return dict(row)
        except (TypeError, ValueError):
            return dict(zip(columns, row, strict=True))

    @staticmethod
    def _empty_snapshot(status: str) -> dict[str, Any]:
        return {
            "status": status,
            "schema_version": None,
            "canonical_json": None,
            "digest": None,
            "size_bytes": None,
            "created_at": None,
            "payload": None,
        }

    def put_behavior_snapshot(
        self,
        conversation_id: str,
        snapshot: BehaviorSnapshotV1,
        *,
        conn: Any,
    ) -> None:
        """Persist one valid snapshot from its immutable canonical-byte authority."""
        if not isinstance(snapshot, BehaviorSnapshotV1):
            raise InputError("snapshot must be a BehaviorSnapshotV1 instance")
        canonical_bytes = snapshot.canonical_bytes
        if not isinstance(canonical_bytes, bytes):
            raise InputError("snapshot canonical_bytes must be immutable bytes")
        expected_digest = f"sha256:{hashlib.sha256(canonical_bytes).hexdigest()}"
        if snapshot.digest != expected_digest:
            raise InputError("snapshot digest does not match canonical bytes")
        if snapshot.size_bytes != len(canonical_bytes):
            raise InputError("snapshot size does not match canonical bytes")
        if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise InputError("snapshot schema version is not supported")
        try:
            canonical_json = canonical_bytes.decode("utf-8")
            rebuilt = build_behavior_snapshot(json.loads(canonical_json))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise InputError("snapshot canonical bytes are not a valid behavior snapshot") from exc
        if rebuilt.canonical_bytes != canonical_bytes:
            raise InputError("snapshot bytes are not canonical")

        conn.execute(
            """
            INSERT INTO conversation_behavior_snapshots(
                conversation_id, status, schema_version, canonical_json, digest, size_bytes
            )
            VALUES (?, 'valid', ?, ?, ?, ?)
            """,
            (
                conversation_id,
                snapshot.schema_version,
                canonical_json,
                snapshot.digest,
                snapshot.size_bytes,
            ),
        )

    def put_creation_settings(
        self,
        conversation_id: str,
        settings: dict[str, Any],
        *,
        conn: Any,
    ) -> None:
        """Insert immutable version-1 creation settings in a caller transaction."""
        try:
            payload = json.dumps(
                settings,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise InputError("creation settings must be finite JSON") from exc
        conn.execute(
            """
            INSERT INTO conversation_settings(
                conversation_id, settings_json, settings_version, last_modified
            )
            VALUES (?, ?, 1, CURRENT_TIMESTAMP)
            """,
            (conversation_id, payload),
        )

    def _get_snapshot(self, conversation_id: str, conn: Any) -> dict[str, Any]:
        result = conn.execute(
            """
            SELECT status, schema_version, canonical_json, digest, size_bytes, created_at
              FROM conversation_behavior_snapshots
             WHERE conversation_id = ?
            """,
            (conversation_id,),
        )
        row = result.fetchone()
        if row is None:
            return self._empty_snapshot("missing")
        record = self._row_to_dict(row, _SNAPSHOT_COLUMNS)
        status = str(record.get("status") or "invalid")
        if status != "valid":
            return self._empty_snapshot(status if status in {"missing", "invalid"} else "invalid")

        canonical_json = record.get("canonical_json")
        if not isinstance(canonical_json, str):
            return self._empty_snapshot("invalid")
        try:
            rebuilt = build_behavior_snapshot(json.loads(canonical_json))
        except (json.JSONDecodeError, TypeError, ValueError):
            return self._empty_snapshot("invalid")
        if (
            rebuilt.schema_version != record.get("schema_version")
            or rebuilt.canonical_bytes.decode("utf-8") != canonical_json
            or rebuilt.digest != record.get("digest")
            or rebuilt.size_bytes != record.get("size_bytes")
        ):
            return self._empty_snapshot("invalid")
        return {
            **record,
            "payload": rebuilt.payload,
        }

    def get_conversation_behavior_snapshot(self, conversation_id: str) -> dict[str, Any]:
        """Read and validate a stored snapshot; absence is explicitly legacy-missing."""
        with self._db.transaction() as conn:
            return self._get_snapshot(conversation_id, conn)

    def get_roleplay_resume_state(
        self,
        conversation_id: str,
        *,
        conn: Any | None = None,
    ) -> dict[str, Any]:
        """Read snapshot and version fences from one caller-owned transaction."""
        transaction = nullcontext(conn) if conn is not None else self._db.transaction()
        with transaction as transaction_conn:
            result = transaction_conn.execute(
                """
                SELECT c.history_version, cs.settings_json, cs.settings_version,
                       (SELECT COUNT(*) FROM messages m
                         WHERE m.conversation_id = c.id AND m.deleted = FALSE) AS message_count
                  FROM conversations c
                  LEFT JOIN conversation_settings cs ON cs.conversation_id = c.id
                 WHERE c.id = ? AND c.deleted = FALSE
                """,
                (conversation_id,),
            )
            row = result.fetchone()
            if row is None:
                raise NotFoundError("Conversation not found.")
            record = self._row_to_dict(
                row,
                ("history_version", "settings_json", "settings_version", "message_count"),
            )
            settings_json = record.get("settings_json")
            try:
                settings = json.loads(settings_json) if settings_json is not None else None
            except (TypeError, json.JSONDecodeError):
                settings = None
            snapshot = self._get_snapshot(conversation_id, transaction_conn)
            readiness = settings.get(_RESUME_SETTINGS_KEY) if isinstance(settings, dict) else None
            effective_completion = (
                readiness.get("effectiveCompletion") if isinstance(readiness, dict) else None
            )
            stored_eligible = (
                readiness.get("resumeEligible") if isinstance(readiness, dict) else False
            )
            stored_reason = (
                readiness.get("resumeIneligibleReason") if isinstance(readiness, dict) else None
            )
            snapshot_status = snapshot["status"]
            if snapshot_status != "valid":
                resume_eligible = False
                resume_ineligible_reason = f"behavior_snapshot_{snapshot_status}"
            elif stored_eligible is True and isinstance(effective_completion, dict):
                resume_eligible = True
                resume_ineligible_reason = None
            else:
                resume_eligible = False
                resume_ineligible_reason = stored_reason or "incomplete_creation_settings"

            tail_result = transaction_conn.execute(
                """
                SELECT id, version FROM messages
                 WHERE conversation_id = ? AND deleted = FALSE
                 ORDER BY timestamp DESC, id DESC
                 LIMIT 1
                """,
                (conversation_id,),
            )
            tail_row = tail_result.fetchone()
            tail = (
                self._row_to_dict(tail_row, ("id", "version"))
                if tail_row is not None
                else None
            )
            return {
                "conversation_id": conversation_id,
                "behavior_snapshot": snapshot,
                "settings": settings,
                "settings_version": record.get("settings_version"),
                "history_version": int(record["history_version"]),
                "message_count": int(record.get("message_count") or 0),
                "tail": {
                    "message_id": tail.get("id") if tail else None,
                    "message_version": int(tail["version"]) if tail else None,
                },
                "resume_eligible": resume_eligible,
                "resume_ineligible_reason": resume_ineligible_reason,
                "effective_completion": effective_completion,
            }


__all__ = ["ConversationResumeStore"]
