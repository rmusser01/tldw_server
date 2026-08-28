"""Persistence seam for immutable character-conversation resume state."""

from __future__ import annotations

import hashlib
import json
import math
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
_READINESS_KEYS = frozenset(
    {"resumeEligible", "resumeIneligibleReason", "effectiveCompletion"}
)
_EFFECTIVE_KEYS = frozenset({"provider", "model", "sampling"})
_SAMPLING_KEYS = frozenset({"temperature", "top_p", "repetition_penalty", "stop"})
_INELIGIBLE_REASONS = frozenset(
    {
        "incomplete_creation_settings",
        "incomplete_effective_settings",
        "invalid_effective_settings",
    }
)


def _validate_closed_sampling(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != _SAMPLING_KEYS:
        return None
    normalized: dict[str, Any] = {}
    for key, minimum, maximum in (
        ("temperature", 0.0, 2.0),
        ("top_p", 0.0, 1.0),
        ("repetition_penalty", 0.0, 3.0),
    ):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        item = float(item)
        if not math.isfinite(item) or not minimum <= item <= maximum:
            return None
        normalized[key] = item
    stop = value.get("stop")
    if (
        not isinstance(stop, list)
        or len(stop) > 64
        or any(not isinstance(item, str) for item in stop)
    ):
        return None
    normalized["stop"] = list(stop)
    return normalized


def _validate_effective_completion(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != _EFFECTIVE_KEYS:
        return None
    provider_value = value.get("provider")
    model_value = value.get("model")
    if not isinstance(provider_value, str) or not isinstance(model_value, str):
        return None
    provider = provider_value.strip()
    model = model_value.strip()
    sampling = _validate_closed_sampling(value.get("sampling"))
    if (
        not provider
        or not model
        or provider_value != provider
        or model_value != model
        or provider.casefold() != provider
        or sampling is None
    ):
        return None
    return {"provider": provider, "model": model, "sampling": sampling}


def _validate_readiness(
    settings: Any,
    *,
    settings_present: bool,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    if not settings_present:
        return False, "incomplete_creation_settings", None
    if not isinstance(settings, dict):
        return False, "invalid_effective_settings", None
    readiness = settings.get(_RESUME_SETTINGS_KEY)
    if readiness is None:
        return False, "incomplete_creation_settings", None
    if not isinstance(readiness, dict) or set(readiness) != _READINESS_KEYS:
        return False, "invalid_effective_settings", None

    eligible = readiness.get("resumeEligible")
    reason = readiness.get("resumeIneligibleReason")
    effective = readiness.get("effectiveCompletion")
    if not isinstance(eligible, bool):
        return False, "invalid_effective_settings", None
    if eligible:
        validated = _validate_effective_completion(effective)
        if reason is not None or validated is None:
            return False, "invalid_effective_settings", None
        return True, None, validated
    if effective is not None or reason not in _INELIGIBLE_REASONS:
        return False, "invalid_effective_settings", None
    return False, reason, None


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

    def get_roleplay_resume_summaries(
        self,
        conversation_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Read bounded list metadata without loading or canonicalizing snapshot bodies."""
        ordered_ids = list(dict.fromkeys(str(item) for item in conversation_ids if item))
        if not ordered_ids:
            return {}
        if len(ordered_ids) > 200:
            raise InputError("resume summary reads are limited to 200 conversations")

        placeholders = ",".join("?" for _ in ordered_ids)
        with self._db.transaction() as conn:
            result = conn.execute(
                f"""
                SELECT c.id, c.history_version, cs.settings_json, cs.settings_version,
                       bs.status AS snapshot_status, bs.schema_version, bs.digest
                  FROM conversations c
                  LEFT JOIN conversation_settings cs ON cs.conversation_id = c.id
                  LEFT JOIN conversation_behavior_snapshots bs ON bs.conversation_id = c.id
                 WHERE c.id IN ({placeholders}) AND c.deleted = FALSE
                """,  # nosec B608 -- placeholders are generated, values stay parameterized
                tuple(ordered_ids),
            )
            columns = (
                "id",
                "history_version",
                "settings_json",
                "settings_version",
                "snapshot_status",
                "schema_version",
                "digest",
            )
            summaries: dict[str, dict[str, Any]] = {}
            for row in result.fetchall():
                record = self._row_to_dict(row, columns)
                conversation_id = str(record["id"])
                settings_json = record.get("settings_json")
                try:
                    settings = json.loads(settings_json) if settings_json is not None else None
                except (TypeError, json.JSONDecodeError):
                    settings = None
                eligible, reason, _effective = _validate_readiness(
                    settings,
                    settings_present=settings_json is not None,
                )

                raw_status = str(record.get("snapshot_status") or "missing")
                digest = record.get("digest")
                valid_digest = (
                    isinstance(digest, str)
                    and digest.startswith("sha256:")
                    and len(digest) == 71
                    and all(char in "0123456789abcdef" for char in digest[7:])
                )
                snapshot_status = (
                    "valid"
                    if raw_status == "valid"
                    and record.get("schema_version") == SNAPSHOT_SCHEMA_VERSION
                    and valid_digest
                    else "missing"
                    if raw_status == "missing"
                    else "invalid"
                )
                if snapshot_status != "valid":
                    eligible = False
                    reason = f"behavior_snapshot_{snapshot_status}"

                summaries[conversation_id] = {
                    "behavior_snapshot": {
                        "status": snapshot_status,
                        "schema_version": (
                            record.get("schema_version")
                            if snapshot_status == "valid"
                            else None
                        ),
                        "digest": digest if snapshot_status == "valid" else None,
                    },
                    "resume_eligible": eligible,
                    "resume_ineligible_reason": None if eligible else reason,
                    "settings_version": record.get("settings_version"),
                    "history_version": int(record["history_version"]),
                }
            return summaries

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
            stored_eligible, stored_reason, effective_completion = _validate_readiness(
                settings,
                settings_present=settings_json is not None,
            )
            snapshot_status = snapshot["status"]
            if snapshot_status != "valid":
                resume_eligible = False
                resume_ineligible_reason = f"behavior_snapshot_{snapshot_status}"
                effective_completion = None
            elif stored_eligible:
                resume_eligible = True
                resume_ineligible_reason = None
            else:
                resume_eligible = False
                resume_ineligible_reason = stored_reason

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
