"""Strict dormant Sync v1 contracts for Notes tasks and task activity."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any, Literal, NoReturn, TypeVar, cast
from uuid import RFC_4122, UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from tldw_Server_API.app.core.exceptions import NotesTaskContractError

from .models import normalize_sync_timestamp

_UUID4_MESSAGE = "IDs must be canonical lowercase UUIDv4 strings"
_SAFE_KEY_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_SAFE_ACTOR_ID_RE = re.compile(r"[A-Za-z0-9._:@/+-]{1,128}")
_ESTIMATE_RE = re.compile(r"[0-9]{1,6}[mhd]")
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_JS_SAFE_INTEGER = 9_007_199_254_740_991
_WEEKDAYS = ("mo", "tu", "we", "th", "fr", "sa", "su")
_TASK_RESERVED_KEYS = frozenset(
    {
        "task_id",
        "note_id",
        "title",
        "description",
        "status",
        "completed_at",
        "priority",
        "due_date",
        "estimate",
        "recurrence",
        "assignee_id",
        "tags",
        "custom",
    }
)
_TASK_METADATA_KEYS = frozenset(
    {
        "description",
        "priority",
        "due_date",
        "estimate",
        "recurrence",
        "assignee_id",
        "tags",
        "custom",
    }
)
_ACTIVITY_METADATA_FORBIDDEN_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "credential",
        "credentials",
        "markdown",
        "password",
        "raw_markdown",
        "secret",
        "token",
    }
)
_DRIFT_REASONS = frozenset(
    {
        "missing_marker_base",
        "malformed_marker",
        "duplicate_marker",
        "marker_scope_mismatch",
        "base_unavailable",
        "both_changed",
        "ambiguous_legacy_match",
        "unsupported_markdown",
    }
)
_PROJECTION_STATUSES = frozenset({"live", "unlinked", "ambiguous", "deleted"})
_ACTOR_TYPES = frozenset({"user", "agent", "tool", "system", "legacy"})
_LEGACY_FIELDS = frozenset(
    {
        "id",
        "task_id",
        "note_id",
        "event_type",
        "actor_type",
        "actor_id",
        "tool_name",
        "policy_mode",
        "approval_id",
        "old_value",
        "new_value",
        "created_at",
        "client_id",
    }
)

TaskStatus = Literal["open", "done"]
TaskPriority = Literal["low", "medium", "high"]
TaskFrequency = Literal["daily", "weekly", "monthly", "yearly"]
TaskRecurrenceState = Literal["active", "paused", "completed"]
TaskEventType = Literal[
    "created",
    "updated",
    "completed",
    "reopened",
    "deleted",
    "restored",
    "projection_linked",
    "projection_unlinked",
    "projection_drift",
    "corrected",
]
TaskActorType = Literal["user", "agent", "tool", "system", "legacy"]
TaskActivitySource = Literal[
    "client",
    "rest",
    "mcp",
    "markdown_reconciliation",
    "repair",
    "trusted_bootstrap_v1",
]
DeleteReason = Literal["user_request", "correction", "policy"]


class _FrozenJsonDict(dict[str, Any]):
    """JSON object that retains normal dict serialization without mutation."""

    def _reject_mutation(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError("canonical JSON values are immutable")

    def __copy__(self) -> _FrozenJsonDict:
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> _FrozenJsonDict:
        return self

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __ior__ = _reject_mutation
    clear = _reject_mutation
    pop = _reject_mutation
    popitem = _reject_mutation
    setdefault = _reject_mutation
    update = _reject_mutation


class _FrozenJsonList(list[Any]):
    """JSON array that retains normal list serialization without mutation."""

    def _reject_mutation(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError("canonical JSON values are immutable")

    def __copy__(self) -> _FrozenJsonList:
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> _FrozenJsonList:
        return self

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __iadd__ = _reject_mutation
    __imul__ = _reject_mutation
    append = _reject_mutation
    clear = _reject_mutation
    extend = _reject_mutation
    insert = _reject_mutation
    pop = _reject_mutation
    remove = _reject_mutation
    reverse = _reject_mutation
    sort = _reject_mutation


class NotesTaskRecurrenceV1(BaseModel):
    """Validated recurrence rule and state; this contract does not schedule work."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    frequency: TaskFrequency
    interval: StrictInt = Field(ge=1, le=365)
    by_weekday: tuple[StrictStr, ...]
    until: StrictStr | None
    state: TaskRecurrenceState
    occurrence_index: StrictInt = Field(ge=0, le=2_147_483_647)

    @field_validator("by_weekday", mode="before")
    @classmethod
    def _tuple_weekdays(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise ValueError("recurrence by_weekday must be an array")
        return tuple(value)

    @model_validator(mode="after")
    def _validate_rule(self) -> NotesTaskRecurrenceV1:
        if len(set(self.by_weekday)) != len(self.by_weekday):
            raise ValueError("recurrence by_weekday values must be unique")
        if any(day not in _WEEKDAYS for day in self.by_weekday):
            raise ValueError("recurrence by_weekday contains an unknown weekday")
        if tuple(sorted(self.by_weekday, key=_WEEKDAYS.index)) != self.by_weekday:
            raise ValueError("recurrence by_weekday must be in Monday-to-Sunday order")
        if self.frequency != "weekly" and self.by_weekday:
            raise ValueError("recurrence by_weekday is allowed only for weekly rules")
        if self.until is not None:
            _validate_date(self.until, "recurrence until")
        return self


class NotesTaskV1Payload(BaseModel):
    """Complete canonical whole-object payload for ``notes.task`` version 1."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    task_id: StrictStr
    note_id: StrictStr
    title: StrictStr = Field(min_length=1, max_length=2_000)
    description: StrictStr | None = Field(max_length=16_000)
    status: TaskStatus
    completed_at: StrictStr | None
    priority: TaskPriority | None
    due_date: StrictStr | None
    estimate: StrictStr | None
    recurrence: NotesTaskRecurrenceV1 | None
    assignee_id: StrictStr | None
    tags: tuple[StrictStr, ...]
    custom: dict[str, Any]

    @field_validator("task_id", "note_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        return _canonical_uuid4(value)

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        if value.strip() != value:
            raise ValueError("task title must already be stripped")
        if _has_control(value):
            raise ValueError("task title cannot contain CR, LF, or control characters")
        return value

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str | None) -> str | None:
        if value is not None and _has_control(value, allowed="\n\t"):
            raise ValueError("task description contains a disallowed control character")
        return value

    @field_validator("completed_at")
    @classmethod
    def _validate_completed_at(cls, value: str | None) -> str | None:
        return None if value is None else _canonical_timestamp(value, "completed_at")

    @field_validator("due_date")
    @classmethod
    def _validate_due_date(cls, value: str | None) -> str | None:
        return None if value is None else _validate_date(value, "due_date")

    @field_validator("estimate")
    @classmethod
    def _validate_estimate(cls, value: str | None) -> str | None:
        if value is not None and _ESTIMATE_RE.fullmatch(value) is None:
            raise ValueError("task estimate must match [0-9]{1,6}[mhd]")
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def _validate_tags(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise ValueError("task tags must be an array")
        tags = tuple(value)
        if len(tags) > 32:
            raise ValueError("task tags may contain at most 32 values")
        normalized: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            if not isinstance(tag, str):
                raise ValueError("task tags must be strings")
            if not 1 <= len(tag) <= 64:
                raise ValueError("task tags must contain 1 to 64 code points")
            if tag.strip() != tag:
                raise ValueError("task tags must already be trimmed")
            if unicodedata.normalize("NFKC", tag) != tag:
                raise ValueError("task tags must already be NFKC normalized")
            if _has_control(tag):
                raise ValueError("task tags cannot contain control characters")
            folded = tag.casefold()
            if folded in seen:
                raise ValueError("task tags must be unique under casefold")
            seen.add(folded)
            normalized.append(tag)
        return tuple(sorted(normalized, key=lambda item: (item.casefold(), item)))

    @field_validator("custom")
    @classmethod
    def _validate_custom(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 32:
            raise ValueError("task custom may contain at most 32 keys")
        for key in value:
            if not isinstance(key, str) or _SAFE_KEY_RE.fullmatch(key) is None:
                raise ValueError("task custom keys must use 1 to 64 safe characters")
            if key in _TASK_RESERVED_KEYS:
                raise ValueError("task custom cannot contain reserved wire keys")
        _validate_json_object(value, label="task custom", max_depth=4, max_bytes=16 * 1_024)
        return cast(dict[str, Any], _freeze_json(value))

    @model_validator(mode="after")
    def _validate_completion(self) -> NotesTaskV1Payload:
        if (self.status == "open" and self.completed_at is not None) or (
            self.status == "done" and self.completed_at is None
        ):
            raise ValueError("task completion requires open/null or done/timestamp")
        return self


class NotesTaskActivityV1(BaseModel):
    """Complete immutable create payload for ``notes.task_activity`` version 1."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    activity_id: StrictStr
    note_id: StrictStr
    task_id: StrictStr | None
    event_type: TaskEventType
    actor_type: TaskActorType
    actor_id: StrictStr | None = Field(max_length=128)
    source_device_id: StrictStr | None
    client_occurred_at: StrictStr
    source_kind: TaskActivitySource
    corrects_activity_id: StrictStr | None
    old_value: dict[str, Any] | None
    new_value: dict[str, Any] | None
    metadata: dict[str, Any]

    @field_validator("activity_id", "note_id")
    @classmethod
    def _validate_required_ids(cls, value: str) -> str:
        return _canonical_uuid4(value)

    @field_validator("task_id", "source_device_id", "corrects_activity_id")
    @classmethod
    def _validate_optional_ids(cls, value: str | None) -> str | None:
        return None if value is None else _canonical_uuid4(value)

    @field_validator("actor_id")
    @classmethod
    def _validate_actor_id(cls, value: str | None) -> str | None:
        if value is not None and _SAFE_ACTOR_ID_RE.fullmatch(value) is None:
            raise ValueError("activity actor_id must use 1 to 128 safe characters")
        return value

    @field_validator("client_occurred_at")
    @classmethod
    def _validate_client_occurred_at(cls, value: str) -> str:
        return _canonical_timestamp(value, "client_occurred_at")

    @field_validator("old_value", "new_value")
    @classmethod
    def _validate_transition_json(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if value is not None:
            _validate_json_object(
                value, label="activity transition value", max_depth=4, max_bytes=16 * 1_024
            )
        return cast(dict[str, Any], _freeze_json(value)) if value is not None else None

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 16:
            raise ValueError("activity metadata may contain at most 16 keys")
        _validate_json_object(
            value, label="activity metadata", max_depth=3, max_bytes=8 * 1_024
        )
        if any(key.casefold() in _ACTIVITY_METADATA_FORBIDDEN_KEYS for key in _walk_keys(value)):
            raise ValueError("activity metadata cannot contain credentials or raw Markdown")
        return cast(dict[str, Any], _freeze_json(value))

    @model_validator(mode="after")
    def _validate_correction_target(self) -> NotesTaskActivityV1:
        if (self.event_type == "corrected") != (self.corrects_activity_id is not None):
            raise ValueError("corrected activity must have exactly one correction target")
        return self


class NotesTaskActivityTombstoneV1(BaseModel):
    """Exact one-way tombstone payload for one immutable activity."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    note_id: StrictStr
    task_id: StrictStr | None
    deleted_at: StrictStr
    delete_reason: DeleteReason

    @field_validator("note_id")
    @classmethod
    def _validate_note_id(cls, value: str) -> str:
        return _canonical_uuid4(value)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: str | None) -> str | None:
        return None if value is None else _canonical_uuid4(value)

    @field_validator("deleted_at")
    @classmethod
    def _validate_deleted_at(cls, value: str) -> str:
        return _canonical_timestamp(value, "deleted_at")


ContractModel = TypeVar("ContractModel", bound=BaseModel)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one validated value as canonical UTF-8 JSON."""

    try:
        _validate_json_value(value, "value")
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise NotesTaskContractError("value is not canonical UTF-8 JSON") from exc


def parse_notes_task_v1(
    payload: Mapping[str, object] | NotesTaskV1Payload,
    *,
    owner_user_id: str,
) -> NotesTaskV1Payload:
    """Parse and owner-bind one complete live task payload."""

    parsed = _parse_model(NotesTaskV1Payload, payload, "notes.task v1 payload")
    if parsed.assignee_id is not None and parsed.assignee_id != owner_user_id:
        raise NotesTaskContractError(
            "notes.task v1 assignee must be the authenticated personal-dataset owner"
        )
    return parsed


def parse_notes_task_tombstone_v1(
    payload: Mapping[str, object] | NotesTaskV1Payload,
    *,
    owner_user_id: str,
) -> NotesTaskV1Payload:
    """Parse a task tombstone using the same complete whole-object payload."""

    return parse_notes_task_v1(payload, owner_user_id=owner_user_id)


def notes_task_object_hash(
    payload: NotesTaskV1Payload,
    *,
    revision: int,
    deleted: bool,
) -> str:
    """Hash exact task payload, identity, revision, adapter, and lifecycle."""

    _positive_revision(revision, "notes.task")
    if not isinstance(payload, NotesTaskV1Payload):
        raise NotesTaskContractError("notes.task hash requires a parsed v1 payload")
    semantic = {
        "adapter_version": 1,
        "domain": "notes.task",
        "identity": {"note_id": payload.note_id, "task_id": payload.task_id},
        "lifecycle": "tombstone" if deleted else "live",
        "payload": payload.model_dump(mode="json"),
        "revision": revision,
    }
    return _sha256(semantic)


def parse_notes_task_activity_v1(
    payload: Mapping[str, object] | NotesTaskActivityV1,
    *,
    owner_user_id: str,
    bound_actor_type: str,
    bound_actor_id: object,
    authenticated_device_id: str | None,
    trusted_server_origin: bool,
) -> NotesTaskActivityV1:
    """Parse an immutable activity create and verify server-bound provenance."""

    parsed = _parse_model(
        NotesTaskActivityV1, payload, "notes.task_activity v1 payload"
    )
    if parsed.actor_type != bound_actor_type or parsed.actor_id != bound_actor_id:
        raise NotesTaskContractError(
            "notes.task_activity actor provenance does not match the server binding"
        )
    if parsed.actor_type == "user" and parsed.actor_id != owner_user_id:
        raise NotesTaskContractError(
            "notes.task_activity user actor must equal the authenticated owner"
        )
    if parsed.source_kind == "client":
        if trusted_server_origin:
            raise NotesTaskContractError("client activity cannot use trusted server provenance")
        if authenticated_device_id is None or parsed.source_device_id != authenticated_device_id:
            raise NotesTaskContractError(
                "notes.task_activity source device does not match the authenticated device"
            )
    elif not trusted_server_origin or parsed.source_device_id is not None:
        raise NotesTaskContractError(
            "trusted server activity provenance requires a null source device"
        )
    _validate_event_values(
        parsed.event_type,
        parsed.old_value,
        parsed.new_value,
        owner_user_id=owner_user_id,
    )
    return parsed


def parse_notes_task_activity_tombstone_v1(
    payload: Mapping[str, object] | NotesTaskActivityTombstoneV1,
    *,
    envelope_created_at_client: str,
    original_activity: NotesTaskActivityV1,
) -> NotesTaskActivityTombstoneV1:
    """Parse the exact revision-2 tombstone and bind its parent and timestamp."""

    parsed = _parse_model(
        NotesTaskActivityTombstoneV1,
        payload,
        "notes.task_activity v1 tombstone",
    )
    expected_deleted_at = _normalized_envelope_timestamp(envelope_created_at_client)
    if parsed.deleted_at != expected_deleted_at:
        raise NotesTaskContractError(
            "activity tombstone deleted_at must equal normalized created_at_client"
        )
    if (
        parsed.note_id != original_activity.note_id
        or parsed.task_id != original_activity.task_id
    ):
        raise NotesTaskContractError(
            "activity tombstone parent identities must equal the original create"
        )
    return parsed


def notes_task_activity_object_hash(
    payload: NotesTaskActivityV1 | NotesTaskActivityTombstoneV1,
    *,
    revision: int,
    deleted: bool,
    activity_id: str | None = None,
    original_create_hash: str | None = None,
) -> str:
    """Hash the exact immutable activity create or its one-way tombstone."""

    if deleted:
        if revision != 2:
            raise NotesTaskContractError("activity tombstone must use revision 2")
        if not isinstance(payload, NotesTaskActivityTombstoneV1):
            raise NotesTaskContractError("activity tombstone hash requires a parsed tombstone")
        if activity_id is None:
            raise NotesTaskContractError("activity tombstone hash requires activity_id")
        canonical_activity_id = _canonical_uuid4(activity_id)
        if original_create_hash is None or _SHA256_RE.fullmatch(original_create_hash) is None:
            raise NotesTaskContractError(
                "activity tombstone hash requires the original create fingerprint"
            )
        identity = {
            "activity_id": canonical_activity_id,
            "note_id": payload.note_id,
            "task_id": payload.task_id,
        }
    else:
        if revision != 1:
            raise NotesTaskContractError("activity create must use revision 1")
        if not isinstance(payload, NotesTaskActivityV1):
            raise NotesTaskContractError("activity create hash requires a parsed create")
        if activity_id is not None or original_create_hash is not None:
            raise NotesTaskContractError("activity create hash has no tombstone binding inputs")
        identity = {
            "activity_id": payload.activity_id,
            "note_id": payload.note_id,
            "task_id": payload.task_id,
        }
    semantic = {
        "adapter_version": 1,
        "domain": "notes.task_activity",
        "identity": identity,
        "lifecycle": "tombstone" if deleted else "live",
        "original_create_hash": original_create_hash if deleted else None,
        "payload": payload.model_dump(mode="json"),
        "revision": revision,
    }
    return _sha256(semantic)


def convert_legacy_task_event(
    legacy_event: Mapping[str, object],
    *,
    owner_user_id: str,
    resolved_task_note_id: str | None,
) -> NotesTaskActivityV1:
    """Convert one source-verified legacy task event or fail closed."""

    if not isinstance(legacy_event, Mapping):
        raise NotesTaskContractError("legacy task event must be an object")
    raw = dict(legacy_event)
    unknown = set(raw).difference(_LEGACY_FIELDS)
    required = _LEGACY_FIELDS.difference({"client_id"})
    if unknown or not required.issubset(raw):
        raise NotesTaskContractError("legacy task event has unknown or missing fields")

    activity_id = _canonical_uuid4(raw["id"])
    task_id = _optional_uuid4(raw["task_id"])
    note_id = _optional_uuid4(raw["note_id"])
    if task_id is not None:
        if resolved_task_note_id is None:
            raise NotesTaskContractError("legacy task event parent task is not verified")
        resolved_note = _canonical_uuid4(resolved_task_note_id)
        if note_id is None:
            note_id = resolved_note
        elif note_id != resolved_note:
            raise NotesTaskContractError("legacy task event parent identities do not match")
    elif note_id is None:
        raise NotesTaskContractError("legacy task event must have a verified parent")

    occurred_at = _normalized_envelope_timestamp(raw["created_at"])
    old_value = _copy_json_object(raw["old_value"], "legacy old_value")
    new_value = _copy_json_object(raw["new_value"], "legacy new_value")
    idempotency_key = None
    if new_value is not None and "idempotency_key" in new_value:
        idempotency_key = new_value.pop("idempotency_key")
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 256:
            raise NotesTaskContractError(
                "legacy idempotency_key must be a string of 1 to 256 characters"
            )

    event_type = raw["event_type"]
    canonical_type: str
    canonical_old: dict[str, Any] | None
    canonical_new: dict[str, Any] | None
    if event_type == "created" and old_value is None and set(new_value or {}) == {
        "text",
        "status",
        "metadata",
    }:
        if new_value is None:
            raise NotesTaskContractError("legacy created payload is missing new_value")
        title = _validate_standalone_title(new_value["text"])
        status = new_value["status"]
        if status not in {"open", "done"}:
            raise NotesTaskContractError("legacy created status is invalid")
        canonical_type = "created"
        canonical_old = None
        canonical_new = {
            "title": title,
            "status": status,
            "completed_at": occurred_at if status == "done" else None,
            "metadata": _expand_legacy_metadata(new_value["metadata"]),
        }
    elif event_type == "updated" and old_value is not None and new_value is not None:
        key_set = set(old_value)
        if key_set != set(new_value) or key_set not in (
            {"metadata"},
            {"text", "metadata"},
        ):
            raise NotesTaskContractError("legacy updated event has a noncanonical shape")
        canonical_type = "updated"
        canonical_old = {"metadata": _expand_legacy_metadata(old_value["metadata"])}
        canonical_new = {"metadata": _expand_legacy_metadata(new_value["metadata"])}
        if "text" in key_set:
            canonical_old = {
                "title": _validate_standalone_title(old_value["text"]),
                **canonical_old,
            }
            canonical_new = {
                "title": _validate_standalone_title(new_value["text"]),
                **canonical_new,
            }
    elif event_type == "status_changed" and old_value == {"status": "open"} and new_value == {
        "status": "done"
    }:
        canonical_type, canonical_old, canonical_new = "completed", old_value, new_value
    elif event_type == "status_changed" and old_value == {"status": "done"} and new_value == {
        "status": "open"
    }:
        canonical_type, canonical_old, canonical_new = "reopened", old_value, new_value
    elif event_type == "unlinked" and old_value == {
        "projection_status": "live"
    } and new_value == {"projection_status": "unlinked"}:
        canonical_type, canonical_old, canonical_new = (
            "projection_unlinked",
            old_value,
            new_value,
        )
    elif event_type == "deleted":
        canonical_type, canonical_old, canonical_new = "deleted", old_value, new_value
    else:
        raise NotesTaskContractError("legacy task event does not match an approved mapping")

    actor_type = raw["actor_type"]
    if not isinstance(actor_type, str) or actor_type not in _ACTOR_TYPES:
        raise NotesTaskContractError("legacy task event actor_type is invalid")
    actor_id = raw["actor_id"]
    if actor_id is not None and (
        not isinstance(actor_id, str) or _SAFE_ACTOR_ID_RE.fullmatch(actor_id) is None
    ):
        raise NotesTaskContractError("legacy task event actor_id is invalid")
    metadata: dict[str, Any] = {"legacy_source_verified": True}
    if idempotency_key is not None:
        metadata["origin_request_fingerprint"] = (
            "sha256:" + hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        )
    legacy_context: dict[str, str] = {}
    for field_name in ("tool_name", "policy_mode", "approval_id"):
        value = raw[field_name]
        if value is None:
            continue
        if not isinstance(value, str) or not 1 <= len(value) <= 128:
            raise NotesTaskContractError(
                f"legacy task event {field_name} must contain 1 to 128 characters"
            )
        legacy_context[field_name] = value
    if legacy_context:
        metadata["legacy_context"] = legacy_context
    client_id = raw.get("client_id")
    if client_id is not None and (
        not isinstance(client_id, str) or not 1 <= len(client_id) <= 128
    ):
        raise NotesTaskContractError("legacy task event client_id is invalid")

    return parse_notes_task_activity_v1(
        {
            "activity_id": activity_id,
            "note_id": note_id,
            "task_id": task_id,
            "event_type": canonical_type,
            "actor_type": actor_type,
            "actor_id": actor_id,
            "source_device_id": None,
            "client_occurred_at": occurred_at,
            "source_kind": "trusted_bootstrap_v1",
            "corrects_activity_id": None,
            "old_value": canonical_old,
            "new_value": canonical_new,
            "metadata": metadata,
        },
        owner_user_id=owner_user_id,
        bound_actor_type=actor_type,
        bound_actor_id=actor_id,
        authenticated_device_id=None,
        trusted_server_origin=True,
    )


def _parse_model(
    model: type[ContractModel],
    payload: Mapping[str, object] | ContractModel,
    label: str,
) -> ContractModel:
    if isinstance(payload, model):
        return payload
    if not isinstance(payload, Mapping):
        raise NotesTaskContractError(f"{label} must be an object")
    try:
        return model.model_validate(dict(payload))
    except ValidationError as exc:
        message = str(exc).replace(
            "Extra inputs are not permitted", "extra inputs are not permitted"
        )
        raise NotesTaskContractError(f"{label}: {message}") from exc


def _canonical_uuid4(value: object) -> str:
    if not isinstance(value, str):
        raise NotesTaskContractError(_UUID4_MESSAGE)
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise NotesTaskContractError(_UUID4_MESSAGE) from exc
    if parsed.version != 4 or parsed.variant != RFC_4122 or str(parsed) != value:
        raise NotesTaskContractError(_UUID4_MESSAGE)
    return value


def _optional_uuid4(value: object) -> str | None:
    return None if value is None else _canonical_uuid4(value)


def _canonical_timestamp(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a canonical RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a canonical RFC 3339 UTC timestamp"
        ) from exc
    normalized = normalize_sync_timestamp(parsed)
    if parsed.tzinfo is None or parsed.utcoffset() is None or normalized != value:
        raise ValueError(f"{field_name} must be a canonical RFC 3339 UTC timestamp")
    return value


def _normalized_envelope_timestamp(value: object) -> str:
    if not isinstance(value, (str, datetime)):
        raise NotesTaskContractError("created_at_client must be an RFC 3339 timestamp")
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NotesTaskContractError("created_at_client must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise NotesTaskContractError("created_at_client must include a timezone")
    normalized = normalize_sync_timestamp(parsed)
    if normalized is None:
        raise NotesTaskContractError("created_at_client is required")
    return normalized


def _validate_date(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a real canonical date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a real canonical date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field_name} must be a real canonical date")
    return value


def _has_control(value: str, *, allowed: str = "") -> bool:
    return any(char not in allowed and unicodedata.category(char) == "Cc" for char in value)


def _validate_json_object(
    value: object,
    *,
    label: str,
    max_depth: int,
    max_bytes: int,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    _validate_json_value(value, label, max_depth=max_depth)
    if len(canonical_json_bytes(value)) > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes // 1_024} KiB")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenJsonDict(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return _FrozenJsonList(_freeze_json(item) for item in value)
    return value


def _validate_json_value(
    value: object,
    label: str,
    *,
    max_depth: int | None = None,
) -> None:
    active_containers: set[int] = set()
    stack: list[tuple[object, int, bool]] = [(value, 0, False)]
    while stack:
        current, parent_depth, leaving = stack.pop()
        if leaving:
            active_containers.remove(id(current))
            continue
        if current is None or isinstance(current, (str, bool)):
            continue
        if isinstance(current, int):
            if not -_JS_SAFE_INTEGER <= current <= _JS_SAFE_INTEGER:
                raise ValueError(f"{label} integers must be within the JS safe range")
            continue
        if isinstance(current, float):
            raise ValueError(f"{label} cannot contain floating-point values")
        if not isinstance(current, (dict, list)):
            raise ValueError(f"{label} must contain only JSON values")
        depth = parent_depth + 1
        if max_depth is not None and depth > max_depth:
            raise ValueError(f"{label} exceeds maximum depth {max_depth}")
        container_id = id(current)
        if container_id in active_containers:
            raise ValueError(f"{label} cannot contain circular values")
        active_containers.add(container_id)
        stack.append((current, parent_depth, True))
        if isinstance(current, dict):
            for key, item in current.items():
                if not isinstance(key, str) or _SAFE_KEY_RE.fullmatch(key) is None:
                    raise ValueError(
                        f"{label} JSON object keys must use 1 to 64 safe ASCII characters"
                    )
                stack.append((item, depth, False))
        else:
            stack.extend((item, depth, False) for item in current)


def _walk_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _positive_revision(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise NotesTaskContractError(f"{label} revision must be a positive integer")
    return value


def _sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _task_metadata_from_payload(parsed: NotesTaskV1Payload) -> dict[str, object]:
    dumped = parsed.model_dump(mode="json")
    return {key: dumped[key] for key in _TASK_METADATA_KEYS}


def _validate_task_metadata(
    value: object, *, owner_user_id: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _TASK_METADATA_KEYS:
        raise NotesTaskContractError("activity task metadata must be the complete exact object")
    wire_value = cast(dict[str, object], json.loads(canonical_json_bytes(value)))
    parsed = parse_notes_task_v1(
        {
            "task_id": "00000000-0000-4000-8000-000000000001",
            "note_id": "00000000-0000-4000-8000-000000000002",
            "title": "metadata validation",
            "status": "open",
            "completed_at": None,
            **wire_value,
        },
        owner_user_id=owner_user_id,
    )
    canonical = _task_metadata_from_payload(parsed)
    if canonical != wire_value:
        raise NotesTaskContractError("activity task metadata is not canonical")
    return canonical


def _validate_standalone_title(value: object) -> str:
    if not isinstance(value, str):
        raise NotesTaskContractError("legacy task title must be a string")
    try:
        parsed = NotesTaskV1Payload.model_validate(
            {
                "task_id": "00000000-0000-4000-8000-000000000001",
                "note_id": "00000000-0000-4000-8000-000000000002",
                "title": value,
                "description": None,
                "status": "open",
                "completed_at": None,
                "priority": None,
                "due_date": None,
                "estimate": None,
                "recurrence": None,
                "assignee_id": None,
                "tags": [],
                "custom": {},
            }
        )
    except ValidationError as exc:
        raise NotesTaskContractError("legacy task title is invalid") from exc
    return parsed.title


def _validate_event_values(
    event_type: str,
    old_value: dict[str, Any] | None,
    new_value: dict[str, Any] | None,
    *,
    owner_user_id: str,
) -> None:
    if event_type == "created":
        if old_value is not None or set(new_value or {}) != {
            "title",
            "status",
            "completed_at",
            "metadata",
        }:
            raise NotesTaskContractError("created activity requires the exact snapshot shape")
        if new_value is None:
            raise NotesTaskContractError("created activity is missing new_value")
        _validate_standalone_title(new_value["title"])
        status = new_value["status"]
        completed_at = new_value["completed_at"]
        if status not in {"open", "done"} or (
            status == "open" and completed_at is not None
        ) or (status == "done" and completed_at is None):
            raise NotesTaskContractError("created activity has an invalid completion snapshot")
        if completed_at is not None:
            _canonical_timestamp(completed_at, "completed_at")
        _validate_task_metadata(new_value["metadata"], owner_user_id=owner_user_id)
        return
    if event_type == "updated":
        if old_value is None or new_value is None or set(old_value) != set(new_value):
            raise NotesTaskContractError("updated activity requires matching old/new keys")
        if set(old_value) not in ({"metadata"}, {"title", "metadata"}) or old_value == new_value:
            raise NotesTaskContractError("updated activity has a noncanonical change shape")
        _validate_task_metadata(old_value["metadata"], owner_user_id=owner_user_id)
        _validate_task_metadata(new_value["metadata"], owner_user_id=owner_user_id)
        if "title" in old_value:
            _validate_standalone_title(old_value["title"])
            _validate_standalone_title(new_value["title"])
        return
    exact = {
        "completed": ({"status": "open"}, {"status": "done"}),
        "reopened": ({"status": "done"}, {"status": "open"}),
        "projection_unlinked": (
            {"projection_status": "live"},
            {"projection_status": "unlinked"},
        ),
    }
    if event_type in exact:
        if (old_value, new_value) != exact[event_type]:
            raise NotesTaskContractError(f"{event_type} activity has a noncanonical shape")
        return
    if event_type == "deleted":
        if (
            old_value is None
            or new_value is None
            or set(old_value) != {"deleted", "projection_status"}
            or old_value["deleted"] is not False
            or old_value["projection_status"] not in {"live", "unlinked", "ambiguous"}
            or set(new_value) != {"deleted", "projection_status"}
            or new_value["deleted"] is not True
            or new_value["projection_status"] != "deleted"
        ):
            raise NotesTaskContractError("deleted activity has a noncanonical shape")
        return
    if event_type == "restored":
        if (
            old_value is None
            or new_value is None
            or set(old_value) != {"deleted", "projection_status"}
            or old_value["deleted"] is not True
            or old_value["projection_status"] != "deleted"
            or set(new_value) != {"deleted", "projection_status"}
            or new_value["deleted"] is not False
            or new_value["projection_status"] not in {"live", "unlinked"}
        ):
            raise NotesTaskContractError("restored activity has a noncanonical shape")
        return
    if event_type == "projection_linked":
        if (
            old_value is None
            or new_value is None
            or set(old_value) != {"projection_status"}
            or old_value["projection_status"] not in {"unlinked", "ambiguous"}
            or new_value != {"projection_status": "live"}
        ):
            raise NotesTaskContractError("projection_linked activity has a noncanonical shape")
        return
    if event_type == "projection_drift":
        if (
            old_value is not None
            or new_value is None
            or set(new_value) != {"reason_code"}
            or new_value["reason_code"] not in _DRIFT_REASONS
        ):
            raise NotesTaskContractError("projection_drift activity has a noncanonical shape")
        return
    if event_type == "corrected":
        _validate_corrected_values(old_value, new_value, owner_user_id=owner_user_id)
        return
    raise NotesTaskContractError("activity event_type is not supported")


def _validate_corrected_values(
    old_value: dict[str, Any] | None,
    new_value: dict[str, Any] | None,
    *,
    owner_user_id: str,
) -> None:
    if not old_value or not new_value or set(old_value) != set(new_value) or old_value == new_value:
        raise NotesTaskContractError("corrected activity requires a changed non-empty key subset")
    keys = set(old_value)
    allowed_subsets = (
        {"title", "status", "completed_at", "metadata"},
        {"title", "metadata"},
        {"metadata"},
        {"status"},
        {"deleted", "projection_status"},
        {"projection_status"},
        {"reason_code"},
    )
    if not any(keys.issubset(schema) for schema in allowed_subsets):
        raise NotesTaskContractError("corrected activity does not match a target event schema")
    for value in (old_value, new_value):
        if "title" in value:
            _validate_standalone_title(value["title"])
        if "metadata" in value:
            _validate_task_metadata(value["metadata"], owner_user_id=owner_user_id)
        if "status" in value and value["status"] not in {"open", "done"}:
            raise NotesTaskContractError("corrected activity status is invalid")
        if "completed_at" in value and value["completed_at"] is not None:
            _canonical_timestamp(value["completed_at"], "completed_at")
        if "deleted" in value and type(value["deleted"]) is not bool:
            raise NotesTaskContractError("corrected activity deleted flag is invalid")
        if "projection_status" in value and value["projection_status"] not in _PROJECTION_STATUSES:
            raise NotesTaskContractError("corrected activity projection status is invalid")
        if "reason_code" in value and value["reason_code"] not in _DRIFT_REASONS:
            raise NotesTaskContractError("corrected activity drift reason is invalid")
        if {"status", "completed_at"}.issubset(value) and (
            (value["status"] == "open" and value["completed_at"] is not None)
            or (value["status"] == "done" and value["completed_at"] is None)
        ):
            raise NotesTaskContractError(
                "corrected activity completion snapshot is invalid"
            )
        if {"deleted", "projection_status"}.issubset(value) and (
            (value["deleted"] is True)
            != (value["projection_status"] == "deleted")
        ):
            raise NotesTaskContractError(
                "corrected activity lifecycle snapshot is invalid"
            )


def _copy_json_object(value: object, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise NotesTaskContractError(f"{label} must be an object or null")
    copied = dict(value)
    try:
        _validate_json_object(
            copied,
            label=label,
            max_depth=4,
            max_bytes=16 * 1_024,
        )
    except ValueError as exc:
        raise NotesTaskContractError(str(exc)) from exc
    return copied


def _expand_legacy_metadata(value: object) -> dict[str, object]:
    if value is None:
        metadata: dict[str, object] = {}
    elif isinstance(value, Mapping):
        metadata = dict(value)
    else:
        raise NotesTaskContractError("legacy task metadata must be an object")
    if set(metadata).difference({"due_date", "priority", "estimate"}):
        raise NotesTaskContractError("legacy task metadata contains unknown keys")
    expanded: dict[str, object] = {
        "description": None,
        "priority": None,
        "due_date": None,
        "estimate": None,
        "recurrence": None,
        "assignee_id": None,
        "tags": [],
        "custom": {},
    }
    expanded.update(metadata)
    return _validate_task_metadata(expanded, owner_user_id="__legacy_owner__")


__all__ = [
    "NotesTaskActivityTombstoneV1",
    "NotesTaskActivityV1",
    "NotesTaskContractError",
    "NotesTaskRecurrenceV1",
    "NotesTaskV1Payload",
    "canonical_json_bytes",
    "convert_legacy_task_event",
    "notes_task_activity_object_hash",
    "notes_task_object_hash",
    "parse_notes_task_activity_tombstone_v1",
    "parse_notes_task_activity_v1",
    "parse_notes_task_tombstone_v1",
    "parse_notes_task_v1",
]
