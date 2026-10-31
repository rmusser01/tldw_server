"""Pure native-fork retention, separate from H1's stricter send admission.

Accepted rich values are inventoried, never executed or resolved from live sources.
A projection is not a capability grant: composers must qualify every required effect.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import (
    NativeAssetManifestEntryV1,
    NativeForkBindingTemplateV1,
    NativeForkCaptureRequestV1,
    NativeForkCaptureV1,
    NativeForkRequestV1,
    NativeScopeV1,
)
from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import (
    build_behavior_snapshot,
    is_credential_key,
)
from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import (
    PROMPT_COMPLETION_SETTING_CLASSIFICATION,
    validate_resumable_behavior_boole,
)
from tldw_Server_API.app.core.Character_Chat.chat_settings_validation import validate_chat_settings_storage
from tldw_Server_API.app.core.Character_Chat.modules.character_utils import (
    CHAR_SENDER_ALIASES,
    SYSTEM_ALIASES,
    TOOL_ALIASES,
    USER_SENDER_ALIASES,
    map_sender_to_role,
    sanitize_sender_name,
)
from tldw_Server_API.app.core.Chat.history_selection import (
    HistorySelectionError,
    _freeze_json,
    _wire_json,
    selection_digest,
)
from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import (
    build_materialized_behavior_settings,
    validate_materialized_behavior_settings,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError
from tldw_Server_API.app.core.exceptions import BehaviorSnapshotValidationError


@dataclass(frozen=True)
class AuthorizedNativeOwner:
    """Authenticated namespace and authorized scope, never request authority."""

    client_id: str
    owner_key: str
    scope: NativeScopeV1


@dataclass(frozen=True)
class ProjectedNativeForkContext:
    """Detached canonical accepted bytes; binding allocation happens at commit."""

    settings_json: str
    snapshot_schema: str | None
    snapshot_json: str | None
    snapshot_digest: str | None
    binding: NativeForkBindingTemplateV1 | None
    retained_context_digest: str
    required_effects: tuple[str, ...]


def _canonical(value: Any) -> str:
    """Encode canonical retained JSON without raw storage formatting."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    """Compute semantic SHA-256 over canonical retained values."""
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def native_fork_request_tuple(request: NativeForkRequestV1) -> list[Any]:
    """The fixed cross-language semantic tuple in H2 §3 (no fence counters)."""
    selection = request.input.selection
    return [
        "tldw-native-fork",
        1,
        request.operation_id,
        request.owner_key,
        [request.scope.kind, request.scope.workspace_id],
        request.source_conversation_id,
        request.projection_version,
        [selection.interpretation.kind, getattr(selection.interpretation, "projection_id", None)],
        [selection.cursor.kind, getattr(selection.cursor, "message_id", None)],
        [[member.id, member.revision] for member in selection.messages],
        request.retained_context_digest,
        [
            [
                asset.reference_id,
                asset.reference_revision,
                asset.asset_id,
                asset.revision,
                asset.representation,
                asset.role,
                asset.context_enabled,
                asset.hash,
                asset.fidelity_disposition,
            ]
            for asset in request.asset_manifest
        ],
        request.fidelity,
        request.child_title,
    ]


def native_fork_request_digest(request: NativeForkRequestV1) -> str:
    """Hash exact validated Unicode with compact UTF-8 JSON, without normalization."""
    encoded = json.dumps(
        native_fork_request_tuple(request), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unsupported(reason: str) -> None:
    raise HistorySelectionError("unsupported_native_fork_" + reason)


_SUMMARY_POLICY = {"enabled", "thresholdMessages", "messageThreshold", "windowMessages", "recentWindowMessages"}
_SUMMARY_CACHE = {"content", "sourceRange", "updatedAt", "compressedCount"}
_EXCLUDED_SETTINGS = {"updatedAt", "characterMemoryExtraction", "roleplayPendingGreetingV1"}
_UNSUPPORTED_SETTINGS = {
    "assistantOverlay",
    "deepResearchAttachment",
    "deepResearchAttachmentHistory",
    "deepResearchPinnedAttachment",
}
_METADATA_FIELDS = {
    "sender_role",
    "sender_name",
    "tool_name",
    "tool_call_id",
    "content_placeholder_reason",
    "client_message_id",
    "image_details",
    "status",
    "rag_context",
}


def _project_settings(settings: dict[str, Any], selected_ids: set[str]) -> dict[str, Any]:
    """Retain accepted policy and selected pins; source cache/run fields are absent."""
    try:
        validate_resumable_behavior_boole(settings)
    except InputError as exc:
        raise HistorySelectionError("unsupported_native_fork_settings") from exc
    for key in ("provider", "model", "authorNote"):
        if key in settings and not isinstance(settings[key], str):
            _unsupported("settings")
    if settings.get("turnTakingMode", "single") != "single":
        _unsupported("participants")
    unknown = (
        set(settings)
        - set(PROMPT_COMPLETION_SETTING_CLASSIFICATION)
        - {"roleplayResumeV1", "roleplayBehaviorV1", "roleplayPendingGreetingV1"}
    )
    if unknown or any(settings.get(key) for key in _UNSUPPORTED_SETTINGS):
        _unsupported("settings")
    result = {key: value for key, value in settings.items() if key not in _EXCLUDED_SETTINGS}
    if "summary" in result:
        summary = result["summary"]
        if not isinstance(summary, dict) or set(summary) - _SUMMARY_POLICY - _SUMMARY_CACHE:
            _unsupported("summary")
        result["summary"] = {key: value for key, value in summary.items() if key in _SUMMARY_POLICY}
        if not result["summary"]:
            result.pop("summary")
    if "pinnedMessageIds" in result:
        pins = result["pinnedMessageIds"]
        if not isinstance(pins, list) or any(not isinstance(pin, str) for pin in pins):
            _unsupported("pins")
        result["pinnedMessageIds"] = list(dict.fromkeys(pin for pin in pins if pin in selected_ids))
    return result


def _project_memory(memory: Any) -> Any:
    """Classify known accepted memory provenance before keeping authored content."""
    if memory is None:
        return None
    if not isinstance(memory, dict) or set(memory) - {"content", "source", "version", "persona_memory_entries"}:
        _unsupported("memory")
    if memory.get("source") not in {"creation_request", "persona_memory_entries"}:
        _unsupported("memory_provenance")
    entries = memory.get("persona_memory_entries", [])
    if not isinstance(entries, list):
        _unsupported("memory")
    retained = []
    for entry in entries:
        if not isinstance(entry, dict):
            _unsupported("memory")
        kind = entry.get("memory_type")
        if kind in {"summary", "compaction"}:
            continue
        if kind not in {"manual", "fact", "relationship", "event", "preference"}:
            _unsupported("memory_provenance")
        if kind != "manual" and entry.get("source_conversation_id") is not None:
            _unsupported("unclassified_derived_memory")
        retained.append(entry)
    if memory.get("source") == "persona_memory_entries" and not retained and not memory.get("content"):
        return None
    return {**memory, "persona_memory_entries": retained}


def _project_citations(value: Any) -> dict[str, Any]:
    """Bounded historical excerpts/labels, with source dereference authority removed."""
    known = {
        "search_query",
        "search_mode",
        "settings_snapshot",
        "retrieved_documents",
        "generated_answer",
        "citations",
        "claims_verified",
        "timestamp",
        "feedback_id",
        "trust_state",
        "trust_reason_codes",
        "trust_evidence_origin",
        "knowledge_trust",
    }
    if not isinstance(value, dict) or set(value) - known:
        _unsupported("citations")
    kept = {}
    for field in ("retrieved_documents", "citations"):
        documents = value.get(field) or []
        if not isinstance(documents, list):
            _unsupported("citations")
        projected = []
        for document in documents:
            allowed = {
                "id",
                "source_id",
                "source_type",
                "title",
                "score",
                "chunk_id",
                "excerpt",
                "url",
                "page_number",
                "line_range",
                "metadata",
                "evidence_origin",
                "source_status",
                "unavailable_reason",
            }
            if not isinstance(document, dict) or set(document) - allowed:
                _unsupported("citations")
            entry = {
                key: document[key]
                for key in ("title", "excerpt", "source_type", "page_number", "line_range")
                if document.get(key) is not None
            }
            if any(not isinstance(entry[key], str) for key in ("title", "excerpt", "source_type") if key in entry):
                _unsupported("citations")
            if "page_number" in entry and (type(entry["page_number"]) is not int or entry["page_number"] < 1):
                _unsupported("citations")
            if "line_range" in entry and (
                not isinstance(entry["line_range"], list)
                or len(entry["line_range"]) != 2
                or any(type(line) is not int or line < 1 for line in entry["line_range"])
            ):
                _unsupported("citations")
            projected.append(entry)
        if projected:
            kept[field] = projected
    if len(_canonical(kept).encode("utf-8")) > 1024 * 1024:
        _unsupported("citations_size")
    return kept


def project_native_fork_messages(
    rows: Sequence[Mapping[str, Any]],
    *,
    context: ProjectedNativeForkContext | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Immutable retained rows for the later ID-remapping commit adapter.

    Revisions represent retained content only. They are never H1 provenance or
    child admission/settlement authority; the writer must allocate fresh authority.
    """
    names = ()
    if context is not None and context.snapshot_json is not None:
        names = tuple(
            participant["identity"]["name"] for participant in json.loads(context.snapshot_json)["participants"]
        )
        if context.binding is not None:
            names += (context.binding.display_name,)
    return tuple(_freeze_json(row) for row in _project_messages(rows, set(), names))


def _contains_credentials(value: Any) -> bool:
    """Structured historical replay must not retain source secret parameters."""
    if isinstance(value, dict):
        return any(is_credential_key(key) or _contains_credentials(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_credentials(item) for item in value)
    return False


def _embedded_asset_manifest(rows: Sequence[Mapping[str, Any]]) -> tuple[NativeAssetManifestEntryV1, ...]:
    """Identify already owned DB image bytes without adopting external sources."""
    assets = []
    for row in rows:
        for index, image in enumerate(row.get("images", ())):
            try:
                header, encoded = image.split(";base64,", 1)
                if header not in {"data:image/png", "data:image/jpeg", "data:image/gif", "data:image/webp"}:
                    _unsupported("images")
                digest = hashlib.sha256(base64.b64decode(encoded, validate=True)).hexdigest()
            except (ValueError, TypeError, binascii.Error) as exc:
                raise HistorySelectionError("unsupported_native_fork_images") from exc
            references = row.get("assets", ())
            if len(references) != len(row.get("images", ())):
                _unsupported("image_reference_binding")
            reference = references[index]
            if reference.get("kind") != "image":
                _unsupported("image_reference_binding")
            reference_id = reference["id"]
            revision = reference["revision"]
            assets.append(
                NativeAssetManifestEntryV1(
                    reference_id=reference_id,
                    reference_revision=revision,
                    asset_id=reference_id,
                    revision=revision,
                    representation="embedded_image_v1",
                    role="message_image",
                    context_enabled=True,
                    hash=digest,
                    fidelity_disposition="retained",
                )
            )
    return tuple(assets)


def _project_messages(
    rows: Sequence[Mapping[str, Any]],
    effects: set[str],
    accepted_names: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Validate settled text, ordered embedded images and closed inert tool replay."""
    retained = []
    pending: set[str] = set()
    seen_calls: set[str] = set()
    for row in rows:
        if row.get("settled") is not True:
            raise HistorySelectionError("unsettled_message")
        text = row.get("message")
        if not isinstance(text, str):
            _unsupported("message")
        # Native legacy image events are text mirrors, not immutable asset claims.
        if text.lstrip().startswith("[[tldw:image-generation-event:"):
            _unsupported("generated_event_requires_native_retention")
        extra = _wire_json(row.get("extra_metadata"))
        extra = {} if extra is None else extra
        if not isinstance(extra, dict) or set(extra) - _METADATA_FIELDS:
            _unsupported("metadata")
        if "client_message_id" in extra:
            source_retry_id = extra.pop("client_message_id")
            if not isinstance(source_retry_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", source_retry_id):
                _unsupported("metadata")
        if "rag_context" in extra:
            extra["rag_context"] = _project_citations(extra["rag_context"])
            effects.add("historical_citations")
        if any(not isinstance(value, str) for key, value in extra.items() if key not in {"rag_context", "image_details"}):
            _unsupported("metadata")
        role = extra.get("sender_role")
        if role is None:
            sender = row.get("role")
            if not isinstance(sender, str):
                _unsupported("role")
            normalized = sender.strip().lower()
            known_names = {name.strip().lower() for name in accepted_names}
            known_names.update(sanitize_sender_name(name).strip().lower() for name in accepted_names)
            aliases = set(USER_SENDER_ALIASES) | set(CHAR_SENDER_ALIASES) | set(SYSTEM_ALIASES) | set(TOOL_ALIASES)
            if normalized not in aliases | known_names and not any(
                normalized.startswith(prefix + ":") for prefix in (*SYSTEM_ALIASES, *TOOL_ALIASES)
            ):
                _unsupported("role")
            role = map_sender_to_role(sender, accepted_names[0] if accepted_names else None)
        if role not in {"user", "assistant", "system", "tool"}:
            _unsupported("role")
        if extra.get("status") not in {None, "complete", "completed", "stopped", "failed"}:
            raise HistorySelectionError("unsettled_message")
        images = _wire_json(row.get("images", []))
        if not isinstance(images, list) or any(
            not isinstance(image, str) or not image.startswith("data:image/") for image in images
        ):
            _unsupported("images")
        details = extra.get("image_details")
        if details is not None and (
            not isinstance(details, list)
            or len(details) != len(images)
            or any(not isinstance(detail, str) or detail not in {"auto", "high", "low"} for detail in details)
        ):
            _unsupported("image_details")
        if images:
            extra["image_details"] = details if details is not None else ["auto"] * len(images)
            effects.add("embedded_image_v1")
        else:
            extra.pop("image_details", None)
        calls = _wire_json(row.get("tool_calls"))
        calls = [] if calls is None else calls
        if not isinstance(calls, list):
            _unsupported("tool_replay")
        placeholder = extra.get("content_placeholder_reason")
        if placeholder == "image_attachment":
            version = row.get("_message_version")
            if role != "user" or not images or type(version) is not int or version < 1:
                _unsupported("image_placeholder")
            if version == 1 and text == f"<Image attachment x{len(images)}>":
                text = ""
            extra.pop("content_placeholder_reason")
        elif placeholder is not None and (placeholder != "tool_calls" or not calls):
            _unsupported("tool_replay_placeholder")
        if pending and role != "tool":
            _unsupported("incomplete_tool_replay")
        if calls:
            if role != "assistant" or not isinstance(calls, list):
                _unsupported("tool_replay")
            for call in calls:
                if (
                    not isinstance(call, dict)
                    or set(call) != {"id", "type", "function"}
                    or call["type"] != "function"
                    or not isinstance(call["id"], str)
                    or not call["id"]
                    or not isinstance(call["function"], dict)
                    or set(call["function"]) != {"name", "arguments"}
                    or not all(isinstance(value, str) for value in call["function"].values())
                ):
                    _unsupported("tool_replay")
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    _canonical(arguments)  # Reject NaN/Infinity accepted by json.loads.
                except (ValueError, TypeError) as exc:
                    raise HistorySelectionError("unsupported_native_fork_tool_replay_arguments") from exc
                if _contains_credentials(arguments):
                    _unsupported("tool_replay_credentials")
                if call["id"] in seen_calls:
                    _unsupported("tool_replay")
                pending.add(call["id"])
                seen_calls.add(call["id"])
            effects.add("tool_replay")
        if role == "tool":
            call_id = extra.get("tool_call_id")
            if call_id not in pending:
                _unsupported("incomplete_tool_replay")
            pending.remove(call_id)
        elif "tool_call_id" in extra:
            _unsupported("tool_replay")
        message = {
            "id": row["id"],
            "role": role,
            "message": text,
            "images": images,
            "tool_calls": calls,
            "extra_metadata": extra,
        }
        message["revision"] = _digest(message)
        retained.append(message)
    if pending:
        _unsupported("incomplete_tool_replay")
    return retained


def project_native_fork_context(
    state: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
    asset_manifest: Sequence[NativeAssetManifestEntryV1],
) -> ProjectedNativeForkContext:
    """Canonical retained bytes from detached coherent state, with no live lookups."""
    state = _wire_json(state)
    conversation = state["conversation"]
    raw_settings = state.get("settings")
    if state.get("settings_present", state.get("settings_version") is not None) and not isinstance(raw_settings, dict):
        raise HistorySelectionError("invalid_settings")
    settings = raw_settings if isinstance(raw_settings, dict) else {}
    effects: set[str] = set()
    snapshot_state = state.get("behavior_snapshot") or {}
    if not isinstance(snapshot_state, dict):
        _unsupported("invalid_snapshot")
    snapshot_payload = snapshot_state.get("payload") or {}
    if not isinstance(snapshot_payload, dict):
        _unsupported("invalid_snapshot")
    participants = snapshot_payload.get("participants", [])
    if not isinstance(participants, list) or any(
        not isinstance(participant, dict)
        or not isinstance(participant.get("identity"), dict)
        or not isinstance(participant["identity"].get("name"), str)
        for participant in participants
    ):
        _unsupported("invalid_snapshot")
    materialized = settings.get("roleplayBehaviorV1") or {}
    if not isinstance(materialized, dict) or not isinstance(materialized.get("values", {}), dict):
        _unsupported("invalid_materialized_behavior")
    overridden = materialized.get("values", {}).get("participants", [])
    if not isinstance(overridden, list) or any(
        not isinstance(participant, dict)
        or not isinstance(participant.get("identity"), dict)
        or not isinstance(participant["identity"].get("name"), str)
        for participant in overridden
    ):
        _unsupported("invalid_materialized_behavior")
    accepted_names = tuple(
        participant.get("identity", {}).get("name", "") for participant in [*participants, *overridden]
    )
    messages = _project_messages(selected_rows, effects, accepted_names)
    selected_ids = {row["id"] for row in messages}
    if len(selected_ids) != len(messages):
        raise HistorySelectionError("duplicate_message_id")
    settings = _project_settings(settings, selected_ids)
    snapshot = state.get("behavior_snapshot") or {}
    kind = conversation.get("assistant_kind")
    source_id = conversation.get("character_id") or conversation.get("assistant_id")
    if kind == "persona" or conversation.get("persona_memory_mode"):
        _unsupported("persona")
    binding = None
    projected_snapshot = None
    if kind is None and source_id is None:
        if snapshot.get("status") != "missing" or "roleplayBehaviorV1" in settings or "roleplayResumeV1" in settings:
            _unsupported("unbound_snapshot")
    else:
        if kind not in {None, "character"} or snapshot.get("status") != "valid":
            _unsupported("snapshot")
        try:
            accepted = build_behavior_snapshot(snapshot["payload"])
            if accepted.digest != snapshot.get("digest") or accepted.schema_version != snapshot.get("schema_version"):
                _unsupported("snapshot_binding")
            payload = accepted.payload
            participants = payload["participants"]
            snapshot_bound_source = (
                conversation.get("assistant_binding_mode") == "snapshot_v1"
                and conversation.get("character_id") is None
                and conversation.get("id") is not None
                and conversation.get("assistant_id") == "snapshot:" + str(conversation["id"])
            )
            if len(participants) != 1 or (
                not snapshot_bound_source and participants[0]["source"]["id"] != str(source_id)
            ):
                _unsupported("participants")
            envelope = settings.get("roleplayBehaviorV1")
            if envelope is None:
                _unsupported("invalid_materialized_behavior")
            validate_materialized_behavior_settings(
                envelope, snapshot_binding={"schema_version": accepted.schema_version, "digest": accepted.digest}
            )
            values = envelope["values"]
            if values.get("assistant_overlay") or values["behavior_controls"].get("turn_taking_mode") != "single":
                _unsupported("behavior_controls")
            accepted_participants = values.get("participants", participants)
            if len(accepted_participants) != 1 or accepted_participants[0]["source"] != participants[0]["source"]:
                _unsupported("participants")
            # Validate overridden accepted participants through the same canonical builder.
            build_behavior_snapshot({**payload, "participants": accepted_participants})
            for participant in participants + (values.get("participants") or []):
                extensions = participant["prompt"]["prompt_relevant_extensions"]
                card_extensions = extensions.get("character_extensions") or {}
                if (
                    set(extensions) - {"prompt_preset", "character_extensions"}
                    or not isinstance(card_extensions, dict)
                    or set(card_extensions) - {"prompt_preset", "promptPreset", "tldw"}
                    or not isinstance(card_extensions.get("tldw", {}), dict)
                    or set(card_extensions.get("tldw", {})) - {"prompt_preset", "promptPreset"}
                ):
                    _unsupported("prompt_extensions")
                participant["default_memory"] = _project_memory(participant["default_memory"])
                for field in ("world_books", "exemplars", "default_memory"):
                    if participant.get(field):
                        effects.add(field)
            projected_snapshot = build_behavior_snapshot(payload)
            controls = values["behavior_controls"]
            if (
                not isinstance(controls.get("applied_overrides"), dict)
                or not isinstance(controls.get("auto_summary"), dict)
                or not isinstance(controls.get("pinned_message_ids"), list)
            ):
                _unsupported("invalid_materialized_behavior")
            controls["applied_overrides"] = _project_settings(controls["applied_overrides"], selected_ids)
            controls["auto_summary"].pop("summary", None)
            controls["pinned_message_ids"] = [pin for pin in controls["pinned_message_ids"] if pin in selected_ids]
            values.pop("greeting", None)
            values["base_snapshot"] = {
                "schema_version": projected_snapshot.schema_version,
                "digest": projected_snapshot.digest,
            }
            settings["roleplayBehaviorV1"] = build_materialized_behavior_settings(values)
            for field in ("world_books", "memory", "prompt_preset"):
                if values.get(field):
                    effects.add(field)
            if controls["auto_summary"].get("enabled"):
                effects.add("summary_policy")
            effects.add("accepted_character_v1")
            binding = NativeForkBindingTemplateV1(
                mode="snapshot_v1",
                primary_participant_id=participants[0]["source"]["id"],
                display_name=accepted_participants[0]["identity"]["name"],
                snapshot_schema=str(projected_snapshot.schema_version),
                snapshot_digest=projected_snapshot.digest,
            )
        except (InputError, BehaviorSnapshotValidationError, KeyError, TypeError) as exc:
            raise HistorySelectionError("unsupported_native_fork_invalid_materialized_behavior") from exc
    try:
        settings = validate_chat_settings_storage(
            settings, reject_credentials=True, allow_internal=True, behavior_snapshot=projected_snapshot
        )
    except InputError as exc:
        raise HistorySelectionError("unsupported_native_fork_invalid_settings") from exc
    for key, effect in (
        ("pinnedMessageIds", "history_pins"),
        ("authorNote", "author_note"),
        ("characterMemoryById", "memory"),
        ("conversationContext", "prompt_context"),
        ("chatGenerationOverride", "sampling"),
        ("generationOverrides", "sampling"),
        ("autoSummaryEnabled", "summary_policy"),
        ("summary", "summary_policy"),
        ("model", "effective_completion"),
        ("provider", "effective_completion"),
    ):
        if settings.get(key):
            effects.add(effect)
    assets = [asset.model_dump(mode="json") for asset in asset_manifest]
    for asset in asset_manifest:
        effects.add(asset.representation)
    settings_json = _canonical(settings)
    snapshot_json = projected_snapshot.canonical_bytes.decode("utf-8") if projected_snapshot else None
    digest = _digest(
        [
            "native-fork-v1",
            [conversation.get("scope_type") or "global", conversation.get("workspace_id")],
            settings,
            json.loads(snapshot_json) if snapshot_json else None,
            binding.model_dump(mode="json") if binding else None,
            messages,
            assets,
        ]
    )
    return ProjectedNativeForkContext(
        settings_json,
        str(projected_snapshot.schema_version) if projected_snapshot else None,
        snapshot_json,
        projected_snapshot.digest if projected_snapshot else None,
        binding,
        digest,
        tuple(sorted(effects)),
    )


def capture_native_fork(
    db: Any, owner: AuthorizedNativeOwner, capture_request: NativeForkCaptureRequestV1
) -> NativeForkCaptureV1:
    """Read-only fork review, using the same coherent adapter future commit consumes."""
    view = capture_request.view
    if view.owner_key not in {None, owner.owner_key}:
        raise HistorySelectionError("owner_conversation_mismatch")
    state, selection, rows = db.message_store.read_native_fork_source(
        view.model_dump(mode="json"),
        owner_client_id=owner.client_id,
        owner_key=owner.owner_key,
        scope_type=owner.scope.kind,
        workspace_id=owner.scope.workspace_id,
    )
    # Native external claims are added by the later asset-store unit. Never infer
    # authority from a legacy URL/MediaFiles ID or a generated-event text preview.
    assets = _embedded_asset_manifest(rows)
    projected = project_native_fork_context(state, rows, assets)
    selection["messages"] = [
        {"id": row["id"], "revision": row["revision"]} for row in project_native_fork_messages(rows, context=projected)
    ]
    selection["selection_digest"] = selection_digest(selection)
    selected_ids = {row["id"] for row in rows}
    settings = state.get("settings") or {}
    values = (state.get("materialized_settings") or {}).get("values", {})
    pins = (*settings.get("pinnedMessageIds", ()), *values.get("behavior_controls", {}).get("pinned_message_ids", ()))
    reasons = ("omitted_history_pins",) if any(pin not in selected_ids for pin in pins) else ()
    return NativeForkCaptureV1(
        protocol="native_atomic_v1",
        projection_version="native-fork-v1",
        owner_key=owner.owner_key,
        scope=owner.scope,
        source_conversation_id=view.conversation_id,
        input={"kind": "normal", "selection": selection},
        retained_context_digest=projected.retained_context_digest,
        asset_manifest=assets,
        fidelity=capture_request.fidelity,
        required_effects=projected.required_effects,
        reasons=reasons,
    )
