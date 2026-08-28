"""Persistence seam for immutable character-conversation resume state."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import (
    DEFAULT_MAX_SNAPSHOT_BYTES,
    SNAPSHOT_SCHEMA_VERSION,
    BehaviorSnapshotV1,
    build_behavior_snapshot,
    is_credential_key,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import BackendType, InputError
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
_MATERIALIZED_SETTINGS_KEY = "roleplayBehaviorV1"
_MATERIALIZED_VALUE_KEYS = frozenset(
    {
        "assistant_overlay",
        "base_snapshot",
        "behavior_controls",
        "effective_completion",
        "greeting",
        "memory",
        "participants",
        "prompt_preset",
        "world_books",
    }
)
_REQUIRED_MATERIALIZED_VALUE_KEYS = frozenset(
    {"base_snapshot", "behavior_controls", "effective_completion"}
)
_BASE_SNAPSHOT_KEYS = frozenset({"schema_version", "digest"})
_BEHAVIOR_CONTROL_KEYS = frozenset(
    {
        "applied_overrides",
        "author_note",
        "auto_summary",
        "greeting",
        "memory_scope",
        "pinned_message_ids",
        "preset_scope",
        "prompt_context",
        "turn_taking_mode",
    }
)
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


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise InputError("Materialized behavior settings must be finite JSON.") from exc


def _reject_credentials(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if is_credential_key(str(key)):
                raise InputError(f"{path} contains credential-bearing key {key!r}.")
            _reject_credentials(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_credentials(item, path=f"{path}[{index}]")


def _validate_materialized_authority(values: dict[str, Any]) -> None:
    if not _REQUIRED_MATERIALIZED_VALUE_KEYS.issubset(values):
        raise InputError("Materialized behavior settings are incomplete.")
    base_snapshot = values.get("base_snapshot")
    if not isinstance(base_snapshot, dict) or set(base_snapshot) != _BASE_SNAPSHOT_KEYS:
        raise InputError("Materialized behavior settings require a base snapshot binding.")
    digest = base_snapshot.get("digest")
    if (
        base_snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        or not isinstance(digest, str)
        or len(digest) != 71
        or not digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise InputError("Materialized behavior settings have an invalid base snapshot binding.")
    controls = values.get("behavior_controls")
    if not isinstance(controls, dict) or set(controls) != _BEHAVIOR_CONTROL_KEYS:
        raise InputError("Materialized behavior settings require complete behavior controls.")


def build_materialized_behavior_settings(
    values: dict[str, Any],
    *,
    max_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
) -> dict[str, Any]:
    """Build the closed, canonical behavior-settings record stored with a chat."""
    if not isinstance(values, dict) or not set(values).issubset(_MATERIALIZED_VALUE_KEYS):
        raise InputError("Materialized behavior settings contain unsupported fields.")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise InputError("Materialized behavior settings max_bytes must be positive.")
    _validate_materialized_authority(values)
    if _validate_effective_completion(values.get("effective_completion")) is None:
        raise InputError("Materialized behavior settings require valid effective completion.")
    _reject_credentials(values, path="materialized_behavior")
    canonical_values = json.loads(_canonical_json(values))
    digest_payload = {
        "schemaVersion": 1,
        "values": canonical_values,
    }
    canonical_payload = _canonical_json(digest_payload).encode("utf-8")
    if len(canonical_payload) > max_bytes:
        raise InputError(
            "Materialized behavior settings size "
            f"{len(canonical_payload)} exceeds maximum {max_bytes} bytes."
        )
    digest = f"sha256:{hashlib.sha256(canonical_payload).hexdigest()}"
    return {
        "schemaVersion": 1,
        "digest": digest,
        "values": canonical_values,
    }


def _validate_materialized_behavior_settings(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {"schemaVersion", "digest", "values"}:
        return None
    if value.get("schemaVersion") != 1 or not isinstance(value.get("digest"), str):
        return None
    values = value.get("values")
    if not isinstance(values, dict) or not set(values).issubset(_MATERIALIZED_VALUE_KEYS):
        return None
    if _validate_effective_completion(values.get("effective_completion")) is None:
        return None
    try:
        _validate_materialized_authority(values)
        rebuilt = build_materialized_behavior_settings(values)
    except InputError:
        return None
    if rebuilt != value:
        return None
    return {
        "schema_version": 1,
        "digest": rebuilt["digest"],
        "values": rebuilt["values"],
    }


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

    def get_roleplay_resume_state(
        self,
        conversation_id: str,
        *,
        conn: Any | None = None,
        lock_for_update: bool = False,
    ) -> dict[str, Any]:
        """Read snapshot and version fences from one caller-owned transaction."""
        transaction = nullcontext(conn) if conn is not None else self._db.transaction()
        with transaction as transaction_conn:
            if lock_for_update and self._db.backend_type == BackendType.POSTGRESQL:
                locked = transaction_conn.execute(
                    """
                    SELECT id FROM conversations
                     WHERE id = ? AND deleted = FALSE
                     FOR UPDATE
                    """,
                    (conversation_id,),
                ).fetchone()
                if locked is None:
                    raise NotFoundError("Conversation not found.")
            result = transaction_conn.execute(
                """
                SELECT c.history_version, cs.settings_json, cs.settings_version,
                       (SELECT COUNT(*) FROM messages m
                         WHERE m.conversation_id = c.id AND m.deleted = FALSE) AS message_count,
                       (SELECT m.id FROM messages m
                         WHERE m.conversation_id = c.id AND m.deleted = FALSE
                         ORDER BY m.timestamp DESC, m.id DESC
                         LIMIT 1) AS tail_message_id,
                       (SELECT m.version FROM messages m
                         WHERE m.conversation_id = c.id AND m.deleted = FALSE
                         ORDER BY m.timestamp DESC, m.id DESC
                         LIMIT 1) AS tail_message_version
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
                (
                    "history_version",
                    "settings_json",
                    "settings_version",
                    "message_count",
                    "tail_message_id",
                    "tail_message_version",
                ),
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
            materialized_raw = (
                settings.get(_MATERIALIZED_SETTINGS_KEY)
                if isinstance(settings, dict)
                else None
            )
            materialized_settings = _validate_materialized_behavior_settings(
                materialized_raw
            )
            materialized_invalid = (
                isinstance(settings, dict)
                and _MATERIALIZED_SETTINGS_KEY in settings
                and materialized_settings is None
            )
            materialized_binding_valid = False
            if materialized_settings is not None and snapshot["status"] == "valid":
                base_snapshot = materialized_settings["values"]["base_snapshot"]
                materialized_binding_valid = (
                    base_snapshot["schema_version"] == snapshot["schema_version"]
                    and base_snapshot["digest"] == snapshot["digest"]
                )
            if materialized_settings is not None and materialized_binding_valid:
                effective_completion = materialized_settings["values"][
                    "effective_completion"
                ]
            snapshot_status = snapshot["status"]
            if snapshot_status != "valid":
                resume_eligible = False
                resume_ineligible_reason = f"behavior_snapshot_{snapshot_status}"
                effective_completion = None
            elif stored_eligible and (
                materialized_invalid
                or materialized_settings is None
                or not materialized_binding_valid
            ):
                resume_eligible = False
                resume_ineligible_reason = "invalid_effective_settings"
                effective_completion = None
            elif stored_eligible:
                resume_eligible = True
                resume_ineligible_reason = None
            else:
                resume_eligible = False
                resume_ineligible_reason = stored_reason

            tail_message_id = record.get("tail_message_id")
            tail_message_version = record.get("tail_message_version")
            return {
                "conversation_id": conversation_id,
                "behavior_snapshot": snapshot,
                "settings": settings,
                "materialized_settings": materialized_settings,
                "settings_version": record.get("settings_version"),
                "history_version": int(record["history_version"]),
                "message_count": int(record.get("message_count") or 0),
                "tail": {
                    "message_id": tail_message_id,
                    "message_version": (
                        int(tail_message_version)
                        if tail_message_version is not None
                        else None
                    ),
                },
                "resume_eligible": resume_eligible,
                "resume_ineligible_reason": resume_ineligible_reason,
                "effective_completion": effective_completion,
            }


__all__ = ["ConversationResumeStore", "build_materialized_behavior_settings"]
