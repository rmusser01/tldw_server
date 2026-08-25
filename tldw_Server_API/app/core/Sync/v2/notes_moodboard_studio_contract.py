"""Strict dormant Sync v1 contracts for Notes moodboards and Studio documents."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, NoReturn, TypeVar, cast
from uuid import RFC_4122, UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

JS_SAFE_INTEGER_MAX = 9_007_199_254_740_991
JS_SAFE_INTEGER_MIN = -JS_SAFE_INTEGER_MAX
SYNC_ENVELOPE_MAX_BYTES = 262_144

_SAFE_KEY_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_EXTENSION_WORD_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+"
)
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_RFC3339_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})"
)
_UUID4_MESSAGE = "IDs must be canonical lowercase UUIDv4 strings"
_DIAGNOSTIC_CODE_RE = re.compile(r"[a-z0-9_.-]{1,64}")
_DIAGRAM_TYPES = frozenset(
    {"flowchart", "sequence", "class", "state", "er", "gantt", "pie"}
)
_AUTHORITY_IDENTITY_EXTENSION_CONCEPTS = frozenset(
    {
        "base_revision",
        "canonical_revision",
        "client",
        "client_envelope_id",
        "owner",
        "owner_id",
        "owner_user_id",
        "device",
        "device_id",
        "user_id",
        "dataset",
        "dataset_id",
        "client_id",
        "identity",
        "idempotency_key",
        "mutation_group_id",
        "sync_id",
        "object_id",
        "object_revision",
        "parent_id",
        "moodboard_id",
        "note_id",
        "placement_id",
        "revision",
        "source_note_id",
    }
)
_LIFECYCLE_EXTENSION_CONCEPTS = frozenset(
    {
        "deleted",
        "deleted_at",
        "is_deleted",
        "lifecycle",
        "restore_intent",
        "restored_at",
        "tombstone",
    }
)
_CREDENTIAL_EXTENSION_CONCEPTS = frozenset(
    {
        "access_token",
        "api_key",
        "auth",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "token",
    }
)
_TRANSIENT_EXTENSION_CONCEPTS = frozenset(
    {
        "current_selection",
        "drag",
        "drag_state",
        "failure",
        "focus",
        "hover",
        "hovered",
        "preview",
        "prompt",
        "raw_output",
        "raw_response",
        "request_metadata",
        "selection",
        "temporary_drag",
        "viewport",
        "zoom",
    }
)
_DOMAIN_OPERATION_EXTENSION_CONCEPTS = frozenset(
    {
        "adapter_version",
        "domain",
        "operation",
        "payload",
        "protocol_version",
        "render_version",
        "schema_version",
    }
)
_PROVENANCE_EXTENSION_CONCEPTS = frozenset(
    {
        "accepted_at",
        "accepted_provenance",
        "attestation",
        "model",
        "provenance",
        "provider",
        "source_revision",
    }
)
_HASH_EXTENSION_CONCEPTS = frozenset(
    {
        "base_hash",
        "canonical_hash",
        "checksum",
        "companion_content_hash",
        "digest",
        "excerpt_hash",
        "hash",
        "note_hash",
        "payload_hash",
        "render_hash",
        "result_hash",
        "source_hash",
    }
)
_SCOPE_EXTENSION_CONCEPTS = frozenset(
    {"scope", "scope_type", "workspace", "workspace_id"}
)
_GENERATED_CACHE_EXTENSION_CONCEPTS = frozenset(
    {
        "cache",
        "cached",
        "cached_svg",
        "generated",
        "generated_at",
        "generation",
        "projection",
        "rendered_cache",
    }
)

Attestation = Literal["server", "client_declared", "trusted_bootstrap_v1"]
ProvenanceKind = Literal[
    "manual", "derive", "regenerate", "diagram", "legacy_bootstrap"
]
DiagramType = Literal["flowchart", "sequence", "class", "state", "er", "gantt", "pie"]


class NotesMoodboardStudioContractError(ValueError):
    """Raised when dormant moodboard or Studio canonical state is invalid."""


class _FrozenJsonDict(dict[str, Any]):
    """JSON object that preserves normal serialization while rejecting mutation."""

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
    """JSON array that preserves normal serialization while rejecting mutation."""

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


class MoodboardUpdatedBoundsV1(BaseModel):
    """Inclusive normalized modification-time bounds for one smart rule."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    after: StrictStr | None
    before: StrictStr | None

    @field_validator("after", "before")
    @classmethod
    def _normalize_bound(cls, value: str | None) -> str | None:
        return None if value is None else _normalize_timestamp(value)

    @model_validator(mode="after")
    def _validate_order(self) -> MoodboardUpdatedBoundsV1:
        if self.after is not None and self.before is not None:
            if _parse_normalized_timestamp(self.after) > _parse_normalized_timestamp(
                self.before
            ):
                raise ValueError("updated after must be less than or equal to before")
        return self


class MoodboardSmartRuleV1(BaseModel):
    """Portable canonical smart-match rule; matching results remain derived."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    query: StrictStr | None = Field(max_length=2_000)
    keyword_tokens: tuple[StrictStr, ...]
    collection_sync_ids: tuple[StrictStr, ...]
    sources: tuple[StrictStr, ...]
    updated: MoodboardUpdatedBoundsV1

    @field_validator("query", mode="before")
    @classmethod
    def _normalize_query(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        return _normalize_match_text(value)

    @field_validator("keyword_tokens", mode="before")
    @classmethod
    def _normalize_keyword_tokens(cls, value: object) -> tuple[str, ...]:
        values = _string_array(value, "keyword_tokens", maximum=100, item_maximum=255)
        return _normalized_string_set(
            values, label="keyword_tokens", item_maximum=255
        )

    @field_validator("collection_sync_ids", mode="before")
    @classmethod
    def _normalize_collections(cls, value: object) -> tuple[str, ...]:
        values = _string_array(
            value, "collection_sync_ids", maximum=100, item_maximum=36
        )
        return tuple(sorted({_canonical_uuid4(item) for item in values}))

    @field_validator("sources", mode="before")
    @classmethod
    def _normalize_sources(cls, value: object) -> tuple[str, ...]:
        values = _string_array(value, "sources", maximum=50, item_maximum=255)
        return _normalized_string_set(values, label="sources", item_maximum=255)


class MoodboardCanvasV1(BaseModel):
    """Canonical board canvas settings without replica-local view state."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    layout_mode: Literal["masonry", "freeform"]
    metadata: dict[str, Any]

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: object) -> dict[str, Any]:
        return _bounded_extension_map(
            value,
            label="canvas metadata",
            max_keys=64,
            max_depth=4,
            max_bytes=16 * 1_024,
        )


class NotesMoodboardV1(BaseModel):
    """Complete canonical whole-object payload for ``notes.moodboard`` v1."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    moodboard_id: StrictStr
    name: StrictStr = Field(min_length=1, max_length=255)
    description: StrictStr | None = Field(max_length=2_000)
    smart_rule: MoodboardSmartRuleV1 | None
    canvas: MoodboardCanvasV1

    @field_validator("moodboard_id")
    @classmethod
    def _validate_moodboard_id(cls, value: str) -> str:
        return _canonical_uuid4(value)


class NotesMoodboardNoteV1(BaseModel):
    """Complete canonical placement payload for ``notes.moodboard_note`` v1."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    moodboard_id: StrictStr
    note_id: StrictStr
    x: StrictInt = Field(ge=JS_SAFE_INTEGER_MIN, le=JS_SAFE_INTEGER_MAX)
    y: StrictInt = Field(ge=JS_SAFE_INTEGER_MIN, le=JS_SAFE_INTEGER_MAX)
    width: StrictInt = Field(ge=1, le=1_000_000)
    height: StrictInt = Field(ge=1, le=1_000_000)
    order_index: StrictInt = Field(ge=JS_SAFE_INTEGER_MIN, le=JS_SAFE_INTEGER_MAX)
    display: dict[str, Any]

    @field_validator("moodboard_id", "note_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        return _canonical_uuid4(value)

    @field_validator("display")
    @classmethod
    def _validate_display(cls, value: object) -> dict[str, Any]:
        return _bounded_extension_map(
            value,
            label="placement display",
            max_keys=32,
            max_depth=4,
            max_bytes=8 * 1_024,
        )


class StudioCueSectionV1(BaseModel):
    """One canonical cue section."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: StrictStr = Field(min_length=1, max_length=128)
    kind: Literal["cue"]
    title: StrictStr = Field(min_length=1, max_length=500)
    items: tuple[StrictStr, ...]

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[str, ...]:
        return _string_array(value, "cue items", maximum=200, item_maximum=2_000)


class StudioContentSectionV1(BaseModel):
    """One canonical notes or summary section."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: StrictStr = Field(min_length=1, max_length=128)
    kind: Literal["notes", "summary"]
    title: StrictStr = Field(min_length=1, max_length=500)
    content: StrictStr

    @field_validator("content")
    @classmethod
    def _validate_content_bytes(cls, value: str) -> str:
        if _utf8_length(value, "section content") > 65_536:
            raise ValueError("section content must be at most 65536 UTF-8 bytes")
        return value


StudioSectionV1 = Annotated[
    StudioCueSectionV1 | StudioContentSectionV1,
    Field(discriminator="kind"),
]


class StudioSectionsV1(BaseModel):
    """Closed sections-only canonical Studio render state."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sections: tuple[StudioSectionV1, ...]

    @field_validator("sections", mode="before")
    @classmethod
    def _tuple_sections(cls, value: object) -> tuple[object, ...]:
        values = _array(value, "Studio sections")
        if len(values) > 100:
            raise ValueError("Studio sections may contain at most 100 sections")
        return values

    @model_validator(mode="after")
    def _unique_ids(self) -> StudioSectionsV1:
        ids = [section.id for section in self.sections]
        if len(set(ids)) != len(ids):
            raise ValueError("Studio section IDs must be unique")
        return self


class StudioDiagramSourceV1(BaseModel):
    """Closed canonical diagram-source projection for one selected section."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: StrictStr = Field(min_length=1, max_length=128)
    title: StrictStr = Field(min_length=1, max_length=500)
    kind: Literal["cue", "notes", "summary"]
    content: StrictStr


class StudioDiagramManifestV1(BaseModel):
    """Closed accepted diagram state; rendered caches are deliberately absent."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    diagram_type: DiagramType
    source_section_ids: tuple[StrictStr, ...]
    source_graph: tuple[StudioDiagramSourceV1, ...]
    diagram: StrictStr
    format: Literal["mermaid"]
    status: Literal["ready"]
    render_hash: StrictStr

    @field_validator("source_section_ids", mode="before")
    @classmethod
    def _validate_source_ids(cls, value: object) -> tuple[str, ...]:
        values = _string_array(
            value, "diagram source_section_ids", maximum=50, item_maximum=128
        )
        if len(set(values)) != len(values):
            raise ValueError("diagram source_section_ids must be unique")
        return values

    @field_validator("source_graph", mode="before")
    @classmethod
    def _tuple_source_graph(cls, value: object) -> tuple[object, ...]:
        return _array(value, "diagram source_graph")

    @field_validator("diagram")
    @classmethod
    def _validate_diagram(cls, value: str) -> str:
        normalized = _normalize_line_endings(value)
        if _utf8_length(normalized, "diagram") > 131_072:
            raise ValueError("diagram must be at most 131072 UTF-8 bytes")
        return normalized

    @field_validator("render_hash")
    @classmethod
    def _validate_render_hash(cls, value: str) -> str:
        return _canonical_sha256(value)


class StudioAcceptedProvenanceV1(BaseModel):
    """Closed accepted-transition facts; prompts and raw provider output are absent."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: ProvenanceKind
    attestation: Attestation
    provider: StrictStr | None = Field(max_length=100)
    model: StrictStr | None = Field(max_length=200)
    accepted_at: StrictStr
    source_revision: StrictInt | None = Field(ge=1, le=JS_SAFE_INTEGER_MAX)
    source_hash: StrictStr | None
    result_hash: StrictStr

    @field_validator("accepted_at")
    @classmethod
    def _normalize_accepted_at(cls, value: str) -> str:
        return _normalize_timestamp(value)

    @field_validator("source_hash")
    @classmethod
    def _validate_source_hash(cls, value: str | None) -> str | None:
        return None if value is None else _canonical_sha256(value)

    @field_validator("result_hash")
    @classmethod
    def _validate_result_hash(cls, value: str) -> str:
        return _canonical_sha256(value)

    @model_validator(mode="after")
    def _validate_pairs_and_attestation(self) -> StudioAcceptedProvenanceV1:
        if (self.provider is None) != (self.model is None):
            raise ValueError("provenance provider and model must be null or present together")
        provider_required = self.kind in {"derive", "diagram"}
        if provider_required != (self.provider is not None):
            raise ValueError(
                "provenance provider and model are required only for derive and diagram"
            )
        if (self.source_revision is None) != (self.source_hash is None):
            raise ValueError("provenance source revision and hash must be null or present together")
        if self.kind == "legacy_bootstrap":
            if self.attestation != "trusted_bootstrap_v1":
                raise ValueError("legacy_bootstrap requires trusted_bootstrap_v1 attestation")
        elif self.attestation == "trusted_bootstrap_v1":
            raise ValueError("trusted_bootstrap_v1 attestation is only legacy_bootstrap")
        return self


class NotesStudioDocumentV1(BaseModel):
    """Complete accepted whole-object payload for ``notes.studio_document`` v1."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    note_id: StrictStr
    source_note_id: StrictStr | None
    payload_json: StudioSectionsV1
    template_type: Literal["lined", "grid", "cornell"]
    handwriting_mode: Literal["off", "accented"]
    excerpt_snapshot: StrictStr | None = Field(max_length=5_000_000)
    excerpt_hash: StrictStr | None
    diagram_manifest_json: StudioDiagramManifestV1 | None
    companion_content_hash: StrictStr
    render_version: Literal[1]
    note_revision: StrictInt = Field(ge=1, le=JS_SAFE_INTEGER_MAX)
    note_hash: StrictStr
    accepted_provenance: StudioAcceptedProvenanceV1

    @field_validator("note_id")
    @classmethod
    def _validate_note_id(cls, value: str) -> str:
        return _canonical_uuid4(value)

    @field_validator("source_note_id")
    @classmethod
    def _validate_source_note_id(cls, value: str | None) -> str | None:
        return None if value is None else _canonical_uuid4(value)

    @field_validator("excerpt_snapshot")
    @classmethod
    def _normalize_excerpt(cls, value: str | None) -> str | None:
        return None if value is None else _normalize_line_endings(value)

    @field_validator("diagram_manifest_json", mode="before")
    @classmethod
    def _normalize_diagram_source_order(
        cls, value: object, info: ValidationInfo
    ) -> object:
        if isinstance(value, StudioDiagramManifestV1):
            manifest = value.model_dump(mode="json")
        elif isinstance(value, Mapping):
            manifest = dict(value)
        else:
            return value
        payload_json = info.data.get("payload_json")
        source_ids = manifest.get("source_section_ids")
        if not isinstance(payload_json, StudioSectionsV1) or not isinstance(
            source_ids, Sequence
        ):
            return manifest
        if isinstance(source_ids, (str, bytes, bytearray)) or any(
            not isinstance(source_id, str) for source_id in source_ids
        ):
            return manifest
        selected = tuple(source_ids)
        if len(set(selected)) != len(selected):
            return manifest
        document_ids = tuple(section.id for section in payload_json.sections)
        if not set(selected).issubset(document_ids):
            return manifest
        manifest["source_section_ids"] = [
            section_id for section_id in document_ids if section_id in selected
        ]
        return manifest

    @field_validator("excerpt_hash", "companion_content_hash", "note_hash")
    @classmethod
    def _validate_hashes(cls, value: str | None) -> str | None:
        return None if value is None else _canonical_sha256(value)

    @model_validator(mode="after")
    def _validate_bindings_and_hashes(self) -> NotesStudioDocumentV1:
        if self.source_note_id == self.note_id:
            raise ValueError("Studio source_note_id cannot equal note_id")
        if (self.excerpt_snapshot is None) != (self.excerpt_hash is None):
            raise ValueError("Studio excerpt snapshot and hash must be null or present together")
        if self.excerpt_snapshot is not None:
            if self.source_note_id is None:
                raise ValueError("Studio excerpt requires source_note_id")
            if _text_sha256(self.excerpt_snapshot) != self.excerpt_hash:
                raise ValueError("Studio excerpt_hash does not match excerpt_snapshot")
        has_source_binding = self.accepted_provenance.source_revision is not None
        if (self.source_note_id is not None) != has_source_binding:
            raise ValueError(
                "Studio source_note_id requires the exact provenance source revision and hash"
            )
        if self.diagram_manifest_json is not None:
            _validate_diagram_against_sections(
                self.diagram_manifest_json, self.payload_json
            )
        expected_result_hash = _sha256(_studio_result_semantic(self))
        if self.accepted_provenance.result_hash != expected_result_hash:
            raise ValueError("Studio provenance result_hash does not match accepted state")
        return self


ContractModel = TypeVar("ContractModel", bound=BaseModel)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one canonical JSON value as compact sorted UTF-8 bytes."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    try:
        _validate_canonical_json(value, "value")
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except NotesMoodboardStudioContractError:
        raise
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise NotesMoodboardStudioContractError(
            "value is not canonical UTF-8 JSON"
        ) from exc


def parse_notes_moodboard_v1(
    payload: Mapping[str, object] | NotesMoodboardV1,
) -> NotesMoodboardV1:
    """Parse one complete normalized live moodboard payload."""

    return _parse_model(NotesMoodboardV1, payload, "notes.moodboard v1 payload")


def parse_notes_moodboard_tombstone_v1(
    payload: Mapping[str, object] | NotesMoodboardV1,
) -> NotesMoodboardV1:
    """Parse a complete moodboard tombstone payload without losing state."""

    return parse_notes_moodboard_v1(payload)


def parse_notes_moodboard_note_v1(
    payload: Mapping[str, object] | NotesMoodboardNoteV1,
) -> NotesMoodboardNoteV1:
    """Parse one complete normalized live manual-placement payload."""

    return _parse_model(
        NotesMoodboardNoteV1, payload, "notes.moodboard_note v1 payload"
    )


def parse_notes_moodboard_note_tombstone_v1(
    payload: Mapping[str, object] | NotesMoodboardNoteV1,
) -> NotesMoodboardNoteV1:
    """Parse a complete placement tombstone payload without losing layout."""

    return parse_notes_moodboard_note_v1(payload)


def parse_notes_studio_document_v1(
    payload: Mapping[str, object] | NotesStudioDocumentV1,
    *,
    bound_attestation: Attestation,
    bound_accepted_at: str,
) -> NotesStudioDocumentV1:
    """Parse accepted Studio state and verify its server-bound acceptance facts."""

    parsed = _parse_model(
        NotesStudioDocumentV1, payload, "notes.studio_document v1 payload"
    )
    if parsed.accepted_provenance.attestation != bound_attestation:
        raise NotesMoodboardStudioContractError(
            "Studio provenance does not match the server-bound attestation"
        )
    normalized_accepted_at = _normalize_timestamp(bound_accepted_at)
    if parsed.accepted_provenance.accepted_at != normalized_accepted_at:
        raise NotesMoodboardStudioContractError(
            "Studio provenance does not match the server-bound acceptance time"
        )
    return parsed


def parse_notes_studio_document_tombstone_v1(
    payload: Mapping[str, object] | NotesStudioDocumentV1,
) -> NotesStudioDocumentV1:
    """Parse a complete retained Studio tombstone without restamping provenance."""

    return _parse_model(
        NotesStudioDocumentV1, payload, "notes.studio_document v1 tombstone"
    )


def placement_object_id(
    payload: Mapping[str, object] | NotesMoodboardNoteV1,
) -> str:
    """Return the deterministic namespaced identity for one placement pair."""

    if isinstance(payload, NotesMoodboardNoteV1):
        moodboard_id = payload.moodboard_id
        note_id = payload.note_id
    elif isinstance(payload, Mapping):
        moodboard_id = _canonical_uuid4(payload.get("moodboard_id"))
        note_id = _canonical_uuid4(payload.get("note_id"))
    else:
        raise NotesMoodboardStudioContractError("placement identity requires an object")
    semantic = {
        "domain": "notes.moodboard_note",
        "members": [moodboard_id, note_id],
        "schema_version": 1,
    }
    digest = hashlib.sha256(canonical_json_bytes(semantic)).hexdigest()
    return f"notes.moodboard_note:sha256:{digest}"


def notes_moodboard_object_hash(
    payload: NotesMoodboardV1,
    *,
    revision: int,
    deleted: bool,
) -> str:
    """Hash exact moodboard identity, payload, revision, adapter, and lifecycle."""

    _validate_hash_inputs(payload, NotesMoodboardV1, revision, deleted, "notes.moodboard")
    return _object_hash(
        domain="notes.moodboard",
        identity={"moodboard_id": payload.moodboard_id},
        payload=payload,
        revision=revision,
        deleted=deleted,
    )


def notes_moodboard_note_object_hash(
    payload: NotesMoodboardNoteV1,
    *,
    revision: int,
    deleted: bool,
) -> str:
    """Hash exact placement identity, payload, revision, adapter, and lifecycle."""

    _validate_hash_inputs(
        payload, NotesMoodboardNoteV1, revision, deleted, "notes.moodboard_note"
    )
    return _object_hash(
        domain="notes.moodboard_note",
        identity={
            "moodboard_id": payload.moodboard_id,
            "note_id": payload.note_id,
            "placement_id": placement_object_id(payload),
        },
        payload=payload,
        revision=revision,
        deleted=deleted,
    )


def notes_studio_document_object_hash(
    payload: NotesStudioDocumentV1,
    *,
    revision: int,
    deleted: bool,
) -> str:
    """Hash exact Studio identity, payload, revision, adapter, and lifecycle."""

    _validate_hash_inputs(
        payload, NotesStudioDocumentV1, revision, deleted, "notes.studio_document"
    )
    return _object_hash(
        domain="notes.studio_document",
        identity={"note_id": payload.note_id},
        payload=payload,
        revision=revision,
        deleted=deleted,
    )


def studio_result_hash(
    payload: Mapping[str, object] | NotesStudioDocumentV1,
) -> str:
    """Hash accepted Studio content-bearing state without recursive provenance."""

    try:
        return _sha256(_studio_result_semantic(payload))
    except NotesMoodboardStudioContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise NotesMoodboardStudioContractError(
            "Studio result hash requires the complete content-bearing state"
        ) from exc


def diagram_render_hash(*, diagram_type: str, context: str, diagram: str) -> str:
    """Hash exact accepted diagram inputs after canonical line-ending normalization."""

    if diagram_type not in _DIAGRAM_TYPES:
        raise NotesMoodboardStudioContractError("diagram_type is not supported")
    if not isinstance(context, str) or not isinstance(diagram, str):
        raise NotesMoodboardStudioContractError("diagram context and diagram must be strings")
    return _sha256(
        {
            "diagram_type": diagram_type,
            "context": _normalize_line_endings(context),
            "diagram": _normalize_line_endings(diagram),
        }
    )


def legacy_diagnostic_hash(source: object) -> str:
    """Return a bounded hash of legacy source evidence without exposing its content."""

    if isinstance(source, (bytes, bytearray)):
        source_bytes = bytes(source)
    elif isinstance(source, str):
        try:
            source_bytes = source.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise NotesMoodboardStudioContractError(
                "legacy diagnostic source is not valid UTF-8"
            ) from exc
    else:
        source_bytes = canonical_json_bytes(source)
    if len(source_bytes) > SYNC_ENVELOPE_MAX_BYTES:
        raise NotesMoodboardStudioContractError(
            f"legacy diagnostic source must be at most {SYNC_ENVELOPE_MAX_BYTES} bytes"
        )
    return "sha256:" + hashlib.sha256(source_bytes).hexdigest()


def legacy_source_diagnostic(code: str, source: object) -> dict[str, str]:
    """Return one bounded privacy-safe legacy blocker descriptor."""

    if not isinstance(code, str) or _DIAGNOSTIC_CODE_RE.fullmatch(code) is None:
        raise NotesMoodboardStudioContractError(
            "legacy diagnostic code must use at most 64 safe lowercase characters"
        )
    return {"code": code, "source_hash": legacy_diagnostic_hash(source)}


def _parse_model(
    model: type[ContractModel],
    payload: Mapping[str, object] | ContractModel,
    label: str,
) -> ContractModel:
    if isinstance(payload, model):
        return payload
    if not isinstance(payload, Mapping):
        raise NotesMoodboardStudioContractError(f"{label} must be an object")
    try:
        return model.model_validate(dict(payload))
    except ValidationError as exc:
        message = str(exc).replace(
            "Extra inputs are not permitted", "extra inputs are not permitted"
        )
        message = message.replace("union_tag_invalid", "union tag invalid")
        raise NotesMoodboardStudioContractError(f"{label}: {message}") from exc
    except NotesMoodboardStudioContractError:
        raise


def _object_hash(
    *,
    domain: str,
    identity: dict[str, str],
    payload: BaseModel,
    revision: int,
    deleted: bool,
) -> str:
    return _sha256(
        {
            "adapter_version": 1,
            "domain": domain,
            "identity": identity,
            "lifecycle": "tombstone" if deleted else "live",
            "payload": payload.model_dump(mode="json"),
            "revision": revision,
        }
    )


def _validate_hash_inputs(
    payload: object,
    model: type[BaseModel],
    revision: object,
    deleted: object,
    domain: str,
) -> None:
    if not isinstance(payload, model):
        raise NotesMoodboardStudioContractError(
            f"{domain} hash requires a parsed v1 payload"
        )
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise NotesMoodboardStudioContractError(
            f"{domain} revision must be a positive integer"
        )
    if revision > JS_SAFE_INTEGER_MAX:
        raise NotesMoodboardStudioContractError(f"{domain} revision must be JS-safe")
    if type(deleted) is not bool:
        raise NotesMoodboardStudioContractError(
            f"{domain} deleted flag must be a strict boolean"
        )


def _studio_result_semantic(
    payload: Mapping[str, object] | NotesStudioDocumentV1,
) -> dict[str, object]:
    if isinstance(payload, NotesStudioDocumentV1):
        values = payload.model_dump(mode="json")
    elif isinstance(payload, Mapping):
        values = dict(payload)
    else:
        raise NotesMoodboardStudioContractError(
            "Studio result hash requires a payload object"
        )
    excerpt = values["excerpt_snapshot"]
    if excerpt is not None:
        if not isinstance(excerpt, str):
            raise NotesMoodboardStudioContractError("Studio excerpt must be a string or null")
        excerpt = _normalize_line_endings(excerpt)
    manifest = values["diagram_manifest_json"]
    if manifest is not None:
        if isinstance(manifest, BaseModel):
            manifest = manifest.model_dump(mode="json")
        if not isinstance(manifest, Mapping):
            raise NotesMoodboardStudioContractError(
                "Studio diagram manifest must be an object or null"
            )
        manifest = {
            key: manifest[key]
            for key in (
                "diagram_type",
                "source_section_ids",
                "source_graph",
                "diagram",
                "format",
                "status",
                "render_hash",
            )
        }
        diagram = manifest["diagram"]
        if isinstance(diagram, str):
            manifest["diagram"] = _normalize_line_endings(diagram)
    payload_json = values["payload_json"]
    if isinstance(payload_json, BaseModel):
        payload_json = payload_json.model_dump(mode="json")
    return {
        "note_id": values["note_id"],
        "source_note_id": values["source_note_id"],
        "payload_json": payload_json,
        "template_type": values["template_type"],
        "handwriting_mode": values["handwriting_mode"],
        "excerpt_snapshot": excerpt,
        "excerpt_hash": values["excerpt_hash"],
        "diagram_manifest_json": manifest,
        "companion_content_hash": values["companion_content_hash"],
        "render_version": values["render_version"],
        "note_revision": values["note_revision"],
        "note_hash": values["note_hash"],
    }


def _validate_diagram_against_sections(
    manifest: StudioDiagramManifestV1,
    payload_json: StudioSectionsV1,
) -> None:
    selected = set(manifest.source_section_ids)
    known_ids = {section.id for section in payload_json.sections}
    if not selected.issubset(known_ids):
        raise ValueError("diagram source_section_ids must name an existing section")
    expected_ids = tuple(
        section.id for section in payload_json.sections if section.id in selected
    )
    if manifest.source_section_ids != expected_ids:
        raise ValueError("diagram source_section_ids must use document order")
    expected_graph: list[dict[str, str]] = []
    for section in payload_json.sections:
        if section.id not in selected:
            continue
        content = (
            "\n".join(section.items)
            if isinstance(section, StudioCueSectionV1)
            else section.content
        )
        expected_graph.append(
            {
                "id": section.id,
                "title": section.title,
                "kind": section.kind,
                "content": content,
            }
        )
    actual_graph = [item.model_dump(mode="json") for item in manifest.source_graph]
    if actual_graph != expected_graph:
        raise ValueError("diagram source_graph must equal selected current sections")
    context = _diagram_context(expected_graph)
    expected_hash = diagram_render_hash(
        diagram_type=manifest.diagram_type,
        context=context,
        diagram=manifest.diagram,
    )
    if manifest.render_hash != expected_hash:
        raise ValueError("diagram render_hash does not match accepted diagram inputs")


def _diagram_context(rows: Sequence[Mapping[str, str]]) -> str:
    parts: list[str] = []
    for row in rows:
        title = row["title"].strip()
        content = row["content"].strip()
        if title:
            parts.append(title)
        if content:
            parts.append(content)
    return "\n".join(parts) if parts else "Notes Studio diagram"


def _bounded_extension_map(
    value: object,
    *,
    label: str,
    max_keys: int,
    max_depth: int,
    max_bytes: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    copied = dict(value)
    key_count = _validate_extension_value(
        copied, label=label, max_depth=max_depth, depth=1
    )
    if key_count > max_keys:
        raise ValueError(f"{label} may contain at most {max_keys} keys")
    if len(canonical_json_bytes(copied)) > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes // 1_024} KiB")
    return cast(dict[str, Any], _freeze_json(copied))


def _validate_extension_value(
    value: object,
    *,
    label: str,
    max_depth: int,
    depth: int,
) -> int:
    if depth > max_depth:
        raise ValueError(f"{label} exceeds maximum depth {max_depth}")
    if isinstance(value, Mapping):
        count = 0
        for key, item in value.items():
            if not isinstance(key, str) or _SAFE_KEY_RE.fullmatch(key) is None:
                raise ValueError(
                    f"{label} keys must use 1 to 64 safe ASCII characters"
                )
            _validate_extension_key(key, label)
            count += 1
            if isinstance(item, (Mapping, list, tuple)):
                count += _validate_extension_value(
                    item,
                    label=label,
                    max_depth=max_depth,
                    depth=depth + 1,
                )
            else:
                _validate_extension_scalar(item, label)
        return count
    if isinstance(value, (list, tuple)):
        count = 0
        for item in value:
            if isinstance(item, (Mapping, list, tuple)):
                count += _validate_extension_value(
                    item,
                    label=label,
                    max_depth=max_depth,
                    depth=depth + 1,
                )
            else:
                _validate_extension_scalar(item, label)
        return count
    _validate_extension_scalar(value, label)
    return 0


def _validate_extension_scalar(value: object, label: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str):
            _utf8_length(value, label)
        return
    if isinstance(value, int):
        if not JS_SAFE_INTEGER_MIN <= value <= JS_SAFE_INTEGER_MAX:
            raise ValueError(f"{label} integers must be JS-safe")
        return
    if isinstance(value, float):
        raise ValueError(f"{label} uses integer-only canonical numbers")
    raise ValueError(f"{label} contains a noncanonical JSON value")


def _validate_extension_key(key: str, label: str) -> None:
    tokens = _semantic_extension_tokens(key)
    if _contains_extension_concept(
        tokens, _AUTHORITY_IDENTITY_EXTENSION_CONCEPTS
    ):
        raise ValueError(f"{label} cannot contain reserved authority or identity keys")
    if _contains_extension_concept(tokens, _LIFECYCLE_EXTENSION_CONCEPTS):
        raise ValueError(f"{label} cannot contain reserved lifecycle keys")
    if _contains_extension_concept(tokens, _CREDENTIAL_EXTENSION_CONCEPTS):
        raise ValueError(f"{label} cannot contain credential keys")
    if _contains_extension_concept(tokens, _TRANSIENT_EXTENSION_CONCEPTS):
        raise ValueError(f"{label} cannot contain transient UI or operation keys")
    if _contains_extension_concept(
        tokens, _DOMAIN_OPERATION_EXTENSION_CONCEPTS
    ):
        raise ValueError(f"{label} cannot contain domain or operation keys")
    if _contains_extension_concept(tokens, _PROVENANCE_EXTENSION_CONCEPTS):
        raise ValueError(f"{label} cannot contain provenance keys")
    if _contains_extension_concept(tokens, _HASH_EXTENSION_CONCEPTS):
        raise ValueError(f"{label} cannot contain hash keys")
    if _contains_extension_concept(tokens, _SCOPE_EXTENSION_CONCEPTS):
        raise ValueError(f"{label} cannot contain scope keys")
    if _contains_extension_concept(tokens, _GENERATED_CACHE_EXTENSION_CONCEPTS):
        raise ValueError(f"{label} cannot contain generated or cached values")


def _semantic_extension_tokens(key: str) -> tuple[str, ...]:
    return tuple(
        word.casefold()
        for component in re.split(r"[_.-]+", key)
        for word in _EXTENSION_WORD_RE.findall(component)
    )


def _contains_extension_concept(
    tokens: tuple[str, ...], concepts: frozenset[str]
) -> bool:
    for concept in concepts:
        target = concept.replace("_", "")
        max_parts = concept.count("_") + 1
        for start in range(len(tokens)):
            candidate = ""
            for token in tokens[start : start + max_parts]:
                candidate += token
                if candidate == target:
                    return True
                if len(candidate) >= len(target):
                    break
    return False


def _validate_canonical_json(value: object, label: str) -> None:
    active: set[int] = set()
    stack: list[tuple[object, bool]] = [(value, False)]
    while stack:
        current, leaving = stack.pop()
        if leaving:
            active.remove(id(current))
            continue
        if current is None or isinstance(current, bool):
            continue
        if isinstance(current, str):
            _utf8_length(current, label)
            continue
        if isinstance(current, int):
            if not JS_SAFE_INTEGER_MIN <= current <= JS_SAFE_INTEGER_MAX:
                raise NotesMoodboardStudioContractError(
                    f"{label} integers must be JS-safe"
                )
            continue
        if isinstance(current, float):
            raise NotesMoodboardStudioContractError(
                f"{label} uses integer-only canonical numbers"
            )
        if not isinstance(current, (Mapping, list, tuple)):
            raise NotesMoodboardStudioContractError(
                f"{label} must contain only canonical JSON values"
            )
        current_id = id(current)
        if current_id in active:
            raise NotesMoodboardStudioContractError(
                f"{label} cannot contain circular values"
            )
        active.add(current_id)
        stack.append((current, True))
        if isinstance(current, Mapping):
            for key, item in current.items():
                if not isinstance(key, str) or _SAFE_KEY_RE.fullmatch(key) is None:
                    raise NotesMoodboardStudioContractError(
                        f"{label} object keys must use 1 to 64 safe ASCII characters"
                    )
                stack.append((item, False))
        else:
            stack.extend((item, False) for item in current)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenJsonDict(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return _FrozenJsonList(_freeze_json(item) for item in value)
    return value


def _array(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ValueError(f"{label} must be an array")
    return tuple(value)


def _string_array(
    value: object,
    label: str,
    *,
    maximum: int,
    item_maximum: int,
) -> tuple[str, ...]:
    values = _array(value, label)
    if len(values) > maximum:
        raise ValueError(f"{label} may contain at most {maximum} values")
    for item in values:
        if not isinstance(item, str):
            raise ValueError(f"{label} must contain only strings")
        if len(item) > item_maximum:
            raise ValueError(
                f"{label} values must contain at most {item_maximum} characters"
            )
        _utf8_length(item, label)
    return cast(tuple[str, ...], values)


def _normalized_string_set(
    values: Sequence[str], *, label: str, item_maximum: int
) -> tuple[str, ...]:
    normalized = {_normalize_match_text(value) for value in values}
    if any(len(value) > item_maximum for value in normalized):
        raise ValueError(
            f"{label} values must contain at most {item_maximum} characters"
        )
    return tuple(sorted(normalized, key=lambda item: (item.casefold(), item)))


def _normalize_match_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _canonical_uuid4(value: object) -> str:
    if not isinstance(value, str):
        raise NotesMoodboardStudioContractError(_UUID4_MESSAGE)
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise NotesMoodboardStudioContractError(_UUID4_MESSAGE) from exc
    if parsed.version != 4 or parsed.variant != RFC_4122 or str(parsed) != value:
        raise NotesMoodboardStudioContractError(_UUID4_MESSAGE)
    return value


def _canonical_sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("hash must use lowercase SHA-256 form sha256:<64 hex>")
    return value


def _normalize_timestamp(value: object) -> str:
    if not isinstance(value, str) or _RFC3339_TIMESTAMP_RE.fullmatch(value) is None:
        raise ValueError("timestamp must be an RFC 3339 value with a timezone")
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError as exc:
        raise ValueError("timestamp must be an RFC 3339 value with a timezone") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be an RFC 3339 value with a timezone")
    utc = parsed.astimezone(timezone.utc)
    base = utc.strftime("%Y-%m-%dT%H:%M:%S")
    if utc.microsecond:
        base += "." + f"{utc.microsecond:06d}".rstrip("0")
    return base + "Z"


def _parse_normalized_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _normalize_line_endings(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _utf8_length(value: str, label: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise NotesMoodboardStudioContractError(
            f"{label} is not valid UTF-8"
        ) from exc


def _text_sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(
        _normalize_line_endings(value).encode("utf-8")
    ).hexdigest()


def _sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "JS_SAFE_INTEGER_MAX",
    "JS_SAFE_INTEGER_MIN",
    "MoodboardCanvasV1",
    "MoodboardSmartRuleV1",
    "MoodboardUpdatedBoundsV1",
    "NotesMoodboardNoteV1",
    "NotesMoodboardStudioContractError",
    "NotesMoodboardV1",
    "NotesStudioDocumentV1",
    "StudioAcceptedProvenanceV1",
    "StudioContentSectionV1",
    "StudioCueSectionV1",
    "StudioDiagramManifestV1",
    "StudioDiagramSourceV1",
    "StudioSectionsV1",
    "canonical_json_bytes",
    "diagram_render_hash",
    "legacy_diagnostic_hash",
    "legacy_source_diagnostic",
    "notes_moodboard_note_object_hash",
    "notes_moodboard_object_hash",
    "notes_studio_document_object_hash",
    "parse_notes_moodboard_note_tombstone_v1",
    "parse_notes_moodboard_note_v1",
    "parse_notes_moodboard_tombstone_v1",
    "parse_notes_moodboard_v1",
    "parse_notes_studio_document_tombstone_v1",
    "parse_notes_studio_document_v1",
    "placement_object_id",
    "studio_result_hash",
]
