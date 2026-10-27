"""Persistence seam for immutable character-conversation resume state."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
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
        "dictionaries",
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


def validate_materialized_behavior_settings(
    value: Any,
    *,
    snapshot_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one stored materialized authority and optional snapshot binding."""
    validated = _validate_materialized_behavior_settings(value)
    if validated is None:
        raise InputError("Stored roleplay materialized behavior authority is invalid.")
    if snapshot_binding is not None:
        base_snapshot = validated["values"]["base_snapshot"]
        if base_snapshot != snapshot_binding:
            raise InputError(
                "Stored roleplay materialized behavior authority has an invalid snapshot binding."
            )
    return dict(value)


def validate_roleplay_readiness_settings(
    value: Any,
    *,
    materialized_behavior: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate the closed readiness marker against materialized authority."""
    if not isinstance(value, dict) or set(value) != _READINESS_KEYS:
        raise InputError("Stored roleplay resume readiness authority is invalid.")
    eligible = value.get("resumeEligible")
    reason = value.get("resumeIneligibleReason")
    effective = value.get("effectiveCompletion")
    if eligible is True:
        validated_effective = _validate_effective_completion(effective)
        materialized_effective = (
            materialized_behavior.get("values", {}).get("effective_completion")
            if isinstance(materialized_behavior, dict)
            else None
        )
        if (
            reason is not None
            or validated_effective is None
            or validated_effective != materialized_effective
        ):
            raise InputError("Stored roleplay resume readiness authority is invalid.")
    elif eligible is False:
        if effective is not None or reason not in _INELIGIBLE_REASONS:
            raise InputError("Stored roleplay resume readiness authority is invalid.")
        if materialized_behavior is not None:
            raise InputError("Ineligible roleplay settings cannot contain behavior authority.")
    else:
        raise InputError("Stored roleplay resume readiness authority is invalid.")
    return dict(value)


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

    def append_idempotent_user_message(
        self,
        conversation_id: str,
        request: dict[str, Any],
        *,
        owner_client_id: str,
        validate_message_count: Callable[[int], object] | None = None,
        conn: Any | None = None,
    ) -> dict[str, Any]:
        """Append one protected Resume input under the existing conversation lock.

        Owner-only message authority retains the immutable original request digest
        and receipt; later public message edits cannot rewrite retry intent.
        """
        from pydantic import ValidationError

        from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import (
            RoleplayResumeConflictError,
            require_roleplay_resume_fence,
            roleplay_resume_fence,
        )
        from tldw_Server_API.app.core.Character_Chat.roleplay_resume_schemas import (
            RoleplayResumeUserAppendRequest,
            RoleplayResumeUserAppendResponse,
        )

        body = RoleplayResumeUserAppendRequest.model_validate(request).model_dump(mode="json")
        message_store = self._db.message_store
        intent = message_store._history_intent_digest(body)
        with nullcontext(conn) if conn is not None else self._db.transaction() as active:
            message_store._lock_history_owner(active, conversation_id, owner_client_id)
            state = self.get_roleplay_resume_state(conversation_id, conn=active, owner_client_id=owner_client_id)
            conversation = state["conversation"]
            if conversation.get("scope_type") not in (None, "global") or conversation.get("workspace_id") is not None:
                raise NotFoundError("Conversation not found.")
            scope = {"scope_type": "global", "workspace_id": None}
            row = active.execute(
                "SELECT conversation_id, client_id, history_admission_json FROM messages WHERE id = ?",
                (body["message_id"],),
            ).fetchone()
            if row is not None:
                existing = self._row_to_dict(row, ("conversation_id", "client_id", "history_admission_json"))
                if str(existing["client_id"]) != owner_client_id:
                    raise NotFoundError("Message not found.")
                try:
                    authority = json.loads(existing["history_admission_json"] or "null")
                    original = authority.get("roleplay_resume_v1") if isinstance(authority, dict) else None
                    if (
                        existing["conversation_id"] != conversation_id
                        or not isinstance(original, dict)
                        or type(authority.get("version")) is not int
                        or authority["version"] != 1
                        or authority.get("settled") is not True
                        or authority.get("interpretation") != {"kind": "parent_graph_v1"}
                        or original.get("owner_client_id") != owner_client_id
                        or original.get("scope") != scope
                        or original.get("intent_digest") != intent
                    ):
                        raise RoleplayResumeConflictError("idempotency_conflict")
                    receipt = RoleplayResumeUserAppendResponse.model_validate(original["receipt"]).model_dump(
                        mode="json"
                    )
                    expected_receipt = {
                        "version": 1,
                        "conversation_id": conversation_id,
                        "message_id": body["message_id"],
                        "message_version": 1,
                        "parent_message_id": body["expected"]["tail"]["message_id"],
                        "fence": {
                            **body["expected"],
                            "history_version": body["expected"]["history_version"] + 1,
                            "tail": {"message_id": body["message_id"], "message_version": 1},
                        },
                    }
                    if receipt != expected_receipt:
                        raise RoleplayResumeConflictError("idempotency_conflict")
                    return receipt
                except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
                    raise RoleplayResumeConflictError("idempotency_conflict") from exc
            require_roleplay_resume_fence(state, body["expected"])
            self._roleplay_resume_messages(
                state,
                owner_client_id=owner_client_id,
                conn=active,
                proposed_input=body["content"],
            )
            if validate_message_count is not None:
                validate_message_count(state["message_count"] + 1)
            mid = self._db.add_message(
                {
                    "id": body["message_id"],
                    "conversation_id": conversation_id,
                    "client_id": owner_client_id,
                    "sender": "user",
                    "content": body["content"],
                    "parent_message_id": state["tail"]["message_id"],
                },
                conn=active,
            )
            if mid != body["message_id"]:
                raise InputError("Resume input was not inserted.")
            admitted = self.get_roleplay_resume_state(conversation_id, conn=active, owner_client_id=owner_client_id)
            if admitted["tail"] != {"message_id": mid, "message_version": 1}:
                raise RoleplayResumeConflictError("history_fence_changed")
            receipt = RoleplayResumeUserAppendResponse.model_validate(
                {
                    "version": 1,
                    "conversation_id": conversation_id,
                    "message_id": mid,
                    "message_version": 1,
                    "parent_message_id": state["tail"]["message_id"],
                    "fence": roleplay_resume_fence(admitted),
                }
            ).model_dump(mode="json")
            message_store._write_history_authority(
                active,
                mid,
                {
                    "version": 1,
                    "settled": True,
                    "interpretation": {"kind": "parent_graph_v1"},
                    "roleplay_resume_v1": {
                        "owner_client_id": owner_client_id,
                        "scope": scope,
                        "intent_digest": intent,
                        "receipt": receipt,
                    },
                },
            )
            return receipt

    def _roleplay_resume_messages(
        self,
        state: dict[str, Any],
        *,
        owner_client_id: str,
        conn: Any,
        proposed_input: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
        """Capture complete ancestry and saved resources under the owner transaction."""
        from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import (
            RoleplayResumeConflictError,
        )
        from tldw_Server_API.app.core.Character_Chat.roleplay_resume_prompt import (
            MAX_ROLEPLAY_PROMPT_BYTES,
            build_roleplay_resume_messages,
        )
        from tldw_Server_API.app.core.Chat.chat_service import (
            _format_selected_history_messages,
        )
        from tldw_Server_API.app.core.Chat.history_selection import (
            HistorySelectionError,
            resolve_parent_path,
        )

        store = self._db.message_store
        cid = state["conversation_id"]
        owner_key = store._history_owner_key(owner_client_id, None)
        try:
            snapshot = store.get_conversation_history_snapshot(cid, owner_client_id=owner_client_id, conn=conn)
            if snapshot.interpretation_status.get("kind") != "parent_graph_v1":
                raise HistorySelectionError("legacy_review_required")
            tail = state["tail"]["message_id"]
            path = resolve_parent_path(
                snapshot.nodes,
                ({"kind": "after_message", "message_id": tail} if tail else {"kind": "empty"}),
            )
            if any(node["settled"] is not True for node in path):
                raise HistorySelectionError("unsettled_history")
            ids = tuple(node["id"] for node in path)
            # Gate text/tool/metadata bytes before fetching complete history payloads.
            if self._db.backend_type == BackendType.POSTGRESQL:
                budget_query = "SELECT COALESCE(SUM(octet_length(COALESCE(m.content, '')) + octet_length(COALESCE(mm.tool_calls_json, '')) + octet_length(COALESCE(mm.extra_json, ''))), 0) AS prompt_bytes FROM messages m LEFT JOIN message_metadata mm ON mm.message_id = m.id WHERE m.conversation_id = ? AND m.id IN (SELECT jsonb_array_elements_text(CAST(? AS JSONB)))"
            else:
                budget_query = "SELECT COALESCE(SUM(length(CAST(COALESCE(m.content, '') AS BLOB)) + length(CAST(COALESCE(mm.tool_calls_json, '') AS BLOB)) + length(CAST(COALESCE(mm.extra_json, '') AS BLOB))), 0) AS prompt_bytes FROM messages m LEFT JOIN message_metadata mm ON mm.message_id = m.id WHERE m.conversation_id = ? AND m.id IN (SELECT value FROM json_each(?))"
            prompt_bytes = conn.execute(budget_query, (cid, json.dumps(ids))).fetchone()["prompt_bytes"]
            if prompt_bytes > MAX_ROLEPLAY_PROMPT_BYTES:
                raise RoleplayResumeConflictError("resume_context_too_large")
            fresh, content = store._read_history_snapshot(
                cid,
                owner_client_id=owner_client_id,
                owner_key=owner_key,
                projection_id=None,
                conn=conn,
                selected_ids=ids,
                include_message_versions=True,
            )
            if (
                fresh.source_digest != snapshot.source_digest
                or fresh.fences != snapshot.fences
                or fresh.storage_context_digest != snapshot.storage_context_digest
                or int(fresh.fences.history) != state["history_version"]
            ):
                raise RoleplayResumeConflictError("history_fence_changed")
            # The projector's empty-history pass validates saved behavior before admission.
            context, _ = build_roleplay_resume_messages(state, [])
            history = _format_selected_history_messages(
                snapshot,
                content,
                context["card"],
                user_name=context["user_name"],
                replace_user_placeholders=False,
            )
            if proposed_input is not None:
                history.append({"role": "user", "content": proposed_input})
            context, messages = build_roleplay_resume_messages(state, history)
            return (
                context,
                messages,
                {
                    "source_digest": fresh.source_digest,
                    "storage_context_digest": fresh.storage_context_digest,
                },
            )
        except HistorySelectionError as exc:
            raise RoleplayResumeConflictError("resume_context_unsupported") from exc

    def _roleplay_completion_receipt(
        self,
        conn: Any,
        conversation_id: str,
        body: dict[str, Any],
        owner_client_id: str,
        *,
        content: str | None = None,
    ) -> dict[str, Any] | None:
        """Read only original owner authority, never infer completion from edited text."""
        from pydantic import ValidationError

        from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import (
            RoleplayResumeConflictError,
        )
        from tldw_Server_API.app.core.Character_Chat.roleplay_resume_schemas import (
            CharacterChatCompletionV2Response,
            RoleplayResumeCompletionV1,
        )

        row = conn.execute(
            "SELECT conversation_id, client_id, history_admission_json FROM messages WHERE id = ?",
            (body["assistant_message_id"],),
        ).fetchone()
        if row is None:
            return None
        if str(row["client_id"]) != owner_client_id:
            raise NotFoundError("Message not found.")
        try:
            authority = json.loads(row["history_admission_json"] or "null")
            original = authority["roleplay_completion_v1"]
            stored_request = RoleplayResumeCompletionV1.model_validate(original["request"]).model_dump(mode="json")
            receipt = CharacterChatCompletionV2Response.model_validate(original["receipt"]).model_dump(mode="json")
            result = receipt["tldw_roleplay_resume_v1"]
            expected_post = {
                **body["expected"],
                "history_version": body["expected"]["history_version"] + 1,
                "tail": {
                    "message_id": body["assistant_message_id"],
                    "message_version": 1,
                },
            }
            if (
                type(authority.get("version")) is not int
                or authority["version"] != 1
                or authority.get("settled") is not True
                or authority.get("interpretation") != {"kind": "parent_graph_v1"}
                or row["conversation_id"] != conversation_id
                or original["owner_client_id"] != owner_client_id
                or original["scope"] != {"scope_type": "global", "workspace_id": None}
                or stored_request != body
                or original["receipt"].get("saved") is not True
                or receipt["chat_id"] != conversation_id
                or receipt["user_message_id"] != body["input_message_id"]
                or receipt["assistant_message_id"] != body["assistant_message_id"]
                or result is None
                or result["input_message_id"] != body["input_message_id"]
                or result["input_message_version"] != body["input_message_version"]
                or result["assistant_message_id"] != body["assistant_message_id"]
                or result["assistant_message_version"] != 1
                or result["expected"] != body["expected"]
                or result["fence"] != expected_post
                or result["unsaved_reason"] is not None
                or original["content_digest"]
                != self._db.message_store._history_intent_digest({"content": receipt["assistant_content"]})
                or (content is not None and content != receipt["assistant_content"])
            ):
                raise RoleplayResumeConflictError("idempotency_conflict")
            return receipt
        except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            raise RoleplayResumeConflictError("idempotency_conflict") from exc

    def commit_roleplay_resume_generation(
        self,
        conversation_id: str,
        prepared: dict[str, Any],
        content: str,
        *,
        owner_client_id: str,
        validate_message_count: Callable[[int], object] | None = None,
        conn: Any | None = None,
    ) -> dict[str, Any]:
        """CAS one generated reply against all original behavior and history authority."""
        from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import (
            RoleplayResumeConflictError,
            roleplay_resume_fence,
        )
        from tldw_Server_API.app.core.Character_Chat.roleplay_resume_schemas import (
            CharacterChatCompletionV2Response,
            RoleplayResumeCompletionV1,
        )

        if not isinstance(content, str) or not content.strip() or len(content) > 1_000_000:
            raise InputError("Roleplay reply must contain text within the message limit.")
        body = RoleplayResumeCompletionV1.model_validate(
            {
                "version": 1,
                "input_message_id": prepared["input_message_id"],
                "input_message_version": prepared["input_message_version"],
                "assistant_message_id": prepared["assistant_message_id"],
                "expected": prepared["fence"],
            }
        ).model_dump(mode="json")
        with nullcontext(conn) if conn is not None else self._db.transaction() as active:
            self._db.message_store._lock_history_owner(active, conversation_id, owner_client_id)
            state = self.get_roleplay_resume_state(conversation_id, conn=active, owner_client_id=owner_client_id)
            if (
                state["conversation"].get("scope_type") not in (None, "global")
                or state["conversation"].get("workspace_id") is not None
            ):
                raise NotFoundError("Conversation not found.")
            replay = self._roleplay_completion_receipt(active, conversation_id, body, owner_client_id, content=content)
            if replay is not None:
                return replay
            current = self.prepare_roleplay_resume_generation(
                conversation_id, body, owner_client_id=owner_client_id, conn=active
            )
            if current["history_authority"] != prepared["history_authority"]:
                raise RoleplayResumeConflictError("history_fence_changed")
            if validate_message_count is not None:
                validate_message_count(state["message_count"] + 1)
            message_id = self._db.add_message(
                {
                    "id": body["assistant_message_id"],
                    "conversation_id": conversation_id,
                    "client_id": owner_client_id,
                    "sender": "assistant",
                    "content": content,
                    "parent_message_id": body["input_message_id"],
                },
                conn=active,
            )
            resulting = self.get_roleplay_resume_state(conversation_id, conn=active, owner_client_id=owner_client_id)
            post_fence = roleplay_resume_fence(resulting)
            expected_post = {
                **body["expected"],
                "history_version": body["expected"]["history_version"] + 1,
                "tail": {
                    "message_id": body["assistant_message_id"],
                    "message_version": 1,
                },
            }
            if message_id != body["assistant_message_id"] or post_fence != expected_post:
                raise RoleplayResumeConflictError("history_fence_changed")
            context = prepared["context"]
            effective = prepared["effective_completion"]
            receipt = CharacterChatCompletionV2Response.model_validate(
                {
                    "chat_id": conversation_id,
                    "character_id": context["character_id"],
                    "provider": effective["provider"],
                    "model": effective["model"],
                    "saved": True,
                    "user_message_id": body["input_message_id"],
                    "assistant_message_id": message_id,
                    "assistant_content": content,
                    "speaker_character_id": context["character_id"],
                    "speaker_character_name": context["card"]["name"],
                    "lorebook_diagnostics": context.get("lorebook_diagnostics"),
                    "tldw_roleplay_resume_v1": {
                        "version": 1,
                        "input_message_id": body["input_message_id"],
                        "input_message_version": body["input_message_version"],
                        "assistant_message_id": message_id,
                        "assistant_message_version": 1,
                        "expected": body["expected"],
                        "fence": post_fence,
                    },
                }
            ).model_dump(mode="json")
            self._db.message_store._write_history_authority(
                active,
                message_id,
                {
                    "version": 1,
                    "settled": True,
                    "interpretation": {"kind": "parent_graph_v1"},
                    "roleplay_completion_v1": {
                        "owner_client_id": owner_client_id,
                        "scope": {"scope_type": "global", "workspace_id": None},
                        "request": body,
                        "content_digest": self._db.message_store._history_intent_digest({"content": content}),
                        "receipt": receipt,
                    },
                },
            )
            return receipt

    def prepare_roleplay_resume_generation(
        self,
        conversation_id: str,
        request: dict[str, Any],
        *,
        owner_client_id: str,
        conn: Any | None = None,
    ) -> dict[str, Any]:
        """Freeze exact acknowledged input, complete history and saved behavior before dispatch."""
        from pydantic import ValidationError

        from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import (
            RoleplayResumeConflictError,
            require_roleplay_resume_fence,
            roleplay_resume_fence,
        )
        from tldw_Server_API.app.core.Character_Chat.roleplay_resume_schemas import (
            RoleplayResumeCompletionV1,
            RoleplayResumeUserAppendResponse,
        )

        body = RoleplayResumeCompletionV1.model_validate(request).model_dump(mode="json")
        with nullcontext(conn) if conn is not None else self._db.transaction() as active:
            self._db.message_store._lock_history_owner(active, conversation_id, owner_client_id)
            state = self.get_roleplay_resume_state(conversation_id, conn=active, owner_client_id=owner_client_id)
            if (
                state["conversation"].get("scope_type") not in (None, "global")
                or state["conversation"].get("workspace_id") is not None
            ):
                raise NotFoundError("Conversation not found.")
            if body["assistant_message_id"] == body["input_message_id"]:
                raise RoleplayResumeConflictError("idempotency_conflict")
            replay = self._roleplay_completion_receipt(active, conversation_id, body, owner_client_id)
            if replay is not None:
                return {"completed": replay}
            require_roleplay_resume_fence(state, body["expected"])
            if state["tail"] != {
                "message_id": body["input_message_id"],
                "message_version": body["input_message_version"],
            }:
                raise RoleplayResumeConflictError("resume_input_changed")
            row = active.execute(
                "SELECT sender, version, content, parent_message_id, history_admission_json FROM messages WHERE id = ? AND conversation_id = ? AND client_id = ? AND deleted = FALSE",
                (body["input_message_id"], conversation_id, owner_client_id),
            ).fetchone()
            if row is None or row["sender"] != "user" or row["version"] != body["input_message_version"]:
                raise RoleplayResumeConflictError("resume_input_changed")
            try:
                authority = json.loads(row["history_admission_json"] or "null")
                original = authority["roleplay_resume_v1"]
                receipt = RoleplayResumeUserAppendResponse.model_validate(original["receipt"]).model_dump(mode="json")
                parent = row["parent_message_id"]
                parent_row = (
                    active.execute(
                        "SELECT version FROM messages WHERE id = ? AND conversation_id = ? AND deleted = FALSE",
                        (parent, conversation_id),
                    ).fetchone()
                    if parent is not None
                    else None
                )
                if parent is not None and parent_row is None:
                    raise RoleplayResumeConflictError("resume_input_changed")
                original_fence = {
                    **receipt["fence"],
                    "history_version": receipt["fence"]["history_version"] - 1,
                    "tail": {
                        "message_id": parent,
                        "message_version": (parent_row["version"] if parent_row is not None else None),
                    },
                }
                original_intent = self._db.message_store._history_intent_digest(
                    {
                        "message_id": body["input_message_id"],
                        "content": row["content"],
                        "expected": original_fence,
                    }
                )
                if (
                    type(authority.get("version")) is not int
                    or authority["version"] != 1
                    or authority.get("settled") is not True
                    or authority.get("interpretation") != {"kind": "parent_graph_v1"}
                    or original["intent_digest"] != original_intent
                    or receipt["parent_message_id"] != parent
                    or original["owner_client_id"] != owner_client_id
                    or original["scope"] != {"scope_type": "global", "workspace_id": None}
                    or receipt["conversation_id"] != conversation_id
                    or receipt["message_id"] != body["input_message_id"]
                    or receipt["message_version"] != body["input_message_version"]
                    or receipt["fence"] != body["expected"]
                ):
                    raise RoleplayResumeConflictError("resume_input_changed")
            except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
                raise RoleplayResumeConflictError("resume_input_changed") from exc
            context, messages, history_authority = self._roleplay_resume_messages(
                state, owner_client_id=owner_client_id, conn=active
            )
            return {
                "conversation_id": conversation_id,
                "input_message_id": body["input_message_id"],
                "input_message_version": body["input_message_version"],
                "assistant_message_id": body["assistant_message_id"],
                "fence": roleplay_resume_fence(state),
                "effective_completion": state["effective_completion"],
                "messages": messages,
                "history_authority": history_authority,
                "context": context,
            }

    def get_roleplay_resume_state(
        self,
        conversation_id: str,
        *,
        conn: Any | None = None,
        lock_for_update: bool = False,
        owner_client_id: str | None = None,
    ) -> dict[str, Any]:
        """Read owned conversation identity, snapshot, and fences coherently."""
        normalized_owner = (
            str(owner_client_id).strip() if owner_client_id is not None else None
        )
        transaction = nullcontext(conn) if conn is not None else self._db.transaction()
        with transaction as transaction_conn:
            if lock_for_update and self._db.backend_type == BackendType.POSTGRESQL:
                locked = transaction_conn.execute(
                    """
                    SELECT id FROM conversations
                     WHERE id = ? AND deleted = FALSE
                       AND (CAST(? AS TEXT) IS NULL OR client_id = ?)
                     FOR UPDATE
                    """,
                    (conversation_id, normalized_owner, normalized_owner),
                ).fetchone()
                if locked is None:
                    raise NotFoundError("Conversation not found.")
            result = transaction_conn.execute(
                """
                SELECT c.history_version, cs.settings_json, cs.settings_version,
                       c.id AS conversation_id, c.version AS conversation_version,
                       c.client_id, c.character_id, c.assistant_kind, c.assistant_id,
                       c.persona_memory_mode, c.scope_type, c.workspace_id,
                       c.required_projection_version, c.native_creation_operation_kind,
                       c.native_creation_operation_id,
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
                   AND (CAST(? AS TEXT) IS NULL OR c.client_id = ?)
                """,
                (conversation_id, normalized_owner, normalized_owner),
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
                    "conversation_id",
                    "conversation_version",
                    "client_id",
                    "character_id",
                    "assistant_kind",
                    "assistant_id",
                    "persona_memory_mode",
                    "scope_type",
                    "workspace_id",
                    "required_projection_version",
                    "native_creation_operation_kind",
                    "native_creation_operation_id",
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
                "conversation": {
                    "id": record["conversation_id"],
                    "version": int(record["conversation_version"]),
                    "client_id": record.get("client_id"),
                    "character_id": record.get("character_id"),
                    "assistant_kind": record.get("assistant_kind"),
                    "assistant_id": record.get("assistant_id"),
                    "assistant_binding_mode": (
                        "snapshot_v1"
                        if record.get("assistant_kind") == "character"
                        and record.get("character_id") is None
                        and record.get("assistant_id") == "snapshot:" + str(record["conversation_id"])
                        and record.get("required_projection_version") == "native-fork-v1"
                        and record.get("native_creation_operation_kind") == "native_fork_v1"
                        and record.get("native_creation_operation_id")
                        else None
                    ),
                    "persona_memory_mode": record.get("persona_memory_mode"),
                    "scope_type": record.get("scope_type"),
                    "workspace_id": record.get("workspace_id"),
                },
                "behavior_snapshot": snapshot,
                "settings": settings,
                "materialized_settings": materialized_settings,
                "dictionary_catalog_owner": (
                    self._db.owner_user_id if self._db.backend_type == BackendType.POSTGRESQL else None
                ),
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


__all__ = [
    "ConversationResumeStore",
    "build_materialized_behavior_settings",
    "validate_materialized_behavior_settings",
    "validate_roleplay_readiness_settings",
]
