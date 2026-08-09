from __future__ import annotations

"""Public Sync v2 contracts and identities for Notes organization domains."""

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from .models import NOTES_ORGANIZATION_DOMAINS, SyncDomain

_RESOURCE_DOMAINS = frozenset(
    {
        "notes.keyword",
        "notes.keyword_collection",
        "notes.folder",
    }
)
_LINK_MEMBERS: dict[str, tuple[str, ...]] = {
    "notes.keyword_link": ("subject_type", "subject_id", "keyword_sync_id"),
    "notes.keyword_collection_link": ("collection_sync_id", "keyword_sync_id"),
    "notes.folder_link": ("note_id", "folder_sync_id"),
}


class NotesOrganizationValidationError(ValueError):
    """Validation failure with a stable Notes organization Sync error code."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class KeywordUpsertPayload(BaseModel):
    """Validated public payload for a keyword resource upsert."""

    model_config = ConfigDict(extra="forbid")

    keyword: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class KeywordLinkPayload(BaseModel):
    """Validated public payload for a keyword membership link."""

    model_config = ConfigDict(extra="forbid")

    subject_type: Literal["note", "conversation"]
    subject_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    keyword_sync_id: str


class KeywordCollectionUpsertPayload(BaseModel):
    """Validated public payload for a keyword collection upsert."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
    parent_sync_id: str | None = None


class KeywordCollectionLinkPayload(BaseModel):
    """Validated public payload for a keyword collection membership link."""

    model_config = ConfigDict(extra="forbid")

    collection_sync_id: str
    keyword_sync_id: str


class FolderUpsertPayload(BaseModel):
    """Validated public payload for a folder upsert."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    parent_sync_id: str | None = None


class FolderLinkPayload(BaseModel):
    """Validated public payload for a folder membership link."""

    model_config = ConfigDict(extra="forbid")

    note_id: str
    folder_sync_id: str


_UPSERT_PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    "notes.keyword": KeywordUpsertPayload,
    "notes.keyword_link": KeywordLinkPayload,
    "notes.keyword_collection": KeywordCollectionUpsertPayload,
    "notes.keyword_collection_link": KeywordCollectionLinkPayload,
    "notes.folder": FolderUpsertPayload,
    "notes.folder_link": FolderLinkPayload,
}


def parse_notes_organization_payload(
    domain: SyncDomain,
    operation: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate and normalize a Notes organization payload for one operation."""

    if domain not in NOTES_ORGANIZATION_DOMAINS:
        raise NotesOrganizationValidationError(
            "notes_organization_domain_invalid",
            f"unsupported Notes organization domain: {domain}",
        )
    if operation not in {"upsert", "tombstone"}:
        raise NotesOrganizationValidationError(
            "notes_organization_operation_invalid",
            f"unsupported Notes organization operation: {operation}",
        )
    if not isinstance(payload, Mapping):
        raise NotesOrganizationValidationError(
            "notes_organization_payload_invalid",
            "Notes organization payload must be an object",
        )

    if operation == "tombstone" and domain in _RESOURCE_DOMAINS:
        if payload:
            raise NotesOrganizationValidationError(
                "notes_organization_payload_invalid",
                f"{domain} tombstone payload must be empty",
            )
        return {}

    try:
        parsed = _UPSERT_PAYLOAD_MODELS[domain].model_validate(dict(payload))
    except ValidationError as exc:
        raise NotesOrganizationValidationError(
            "notes_organization_payload_invalid",
            str(exc),
        ) from exc
    normalized = cast(dict[str, object], parsed.model_dump())
    _validate_resource_references(domain, normalized)
    return normalized


def new_organization_sync_id() -> str:
    """Return a canonical lowercase UUIDv4 string."""

    return str(uuid.uuid4())


def validate_resource_sync_id(value: str) -> str:
    """Return a canonical UUIDv4 string or raise a Sync validation error."""

    if not isinstance(value, str):
        raise NotesOrganizationValidationError(
            "notes_organization_resource_sync_id_invalid",
            "resource sync_id must be a canonical UUIDv4 string",
        )
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise NotesOrganizationValidationError(
            "notes_organization_resource_sync_id_invalid",
            "resource sync_id must be a canonical UUIDv4 string",
        ) from exc
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122 or str(parsed) != value:
        raise NotesOrganizationValidationError(
            "notes_organization_resource_sync_id_invalid",
            "resource sync_id must be a canonical UUIDv4 string",
        )
    return value


def organization_link_id(domain: SyncDomain, members: Sequence[str]) -> str:
    """Hash canonical UTF-8 JSON with sorted keys and compact separators."""

    expected_members = _LINK_MEMBERS.get(domain)
    if expected_members is None:
        raise NotesOrganizationValidationError(
            "notes_organization_link_identity_invalid",
            f"unsupported Notes organization relationship domain: {domain}",
        )
    if len(members) != len(expected_members) or any(not isinstance(member, str) for member in members):
        raise NotesOrganizationValidationError(
            "notes_organization_link_identity_invalid",
            f"{domain} requires exactly {len(expected_members)} string identity members",
        )
    canonical = json.dumps(
        {"domain": domain, "members": list(members), "schema_version": 1},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"{domain}:sha256:{hashlib.sha256(canonical).hexdigest()}"


def validate_organization_object_id(
    domain: SyncDomain,
    object_id: str,
    payload: Mapping[str, object],
) -> None:
    """Validate a resource UUID or recompute and compare a relationship ID."""

    if domain in _RESOURCE_DOMAINS:
        validate_resource_sync_id(object_id)
        return

    member_fields = _LINK_MEMBERS.get(domain)
    if member_fields is None:
        raise NotesOrganizationValidationError(
            "notes_organization_domain_invalid",
            f"unsupported Notes organization domain: {domain}",
        )
    members = []
    for field_name in member_fields:
        value = payload.get(field_name)
        if not isinstance(value, str):
            raise NotesOrganizationValidationError(
                "notes_organization_payload_invalid",
                f"{domain} payload requires string identity field {field_name}",
            )
        members.append(value)
    expected_id = organization_link_id(domain, members)
    if object_id != expected_id:
        raise NotesOrganizationValidationError(
            "notes_organization_link_identity_invalid",
            f"{domain} object_id does not match its identity payload",
        )


def _validate_resource_references(domain: SyncDomain, payload: Mapping[str, object]) -> None:
    """Validate every resource reference carried by a Notes organization payload."""

    if domain == "notes.keyword_link":
        _validate_resource_reference(payload, "keyword_sync_id")
        if payload.get("subject_type") == "note":
            _validate_resource_reference(payload, "subject_id")
        return
    if domain in {"notes.keyword_collection", "notes.folder"}:
        _validate_resource_reference(payload, "parent_sync_id", optional=True)
        return
    if domain == "notes.keyword_collection_link":
        _validate_resource_reference(payload, "collection_sync_id")
        _validate_resource_reference(payload, "keyword_sync_id")
        return
    if domain == "notes.folder_link":
        _validate_resource_reference(payload, "note_id")
        _validate_resource_reference(payload, "folder_sync_id")


def _validate_resource_reference(
    payload: Mapping[str, object],
    field_name: str,
    *,
    optional: bool = False,
) -> None:
    """Require one payload resource reference to use the canonical UUIDv4 format."""

    value = payload.get(field_name)
    if optional and value is None:
        return
    validate_resource_sync_id(cast(str, value))


__all__ = [
    "FolderLinkPayload",
    "FolderUpsertPayload",
    "KeywordCollectionLinkPayload",
    "KeywordCollectionUpsertPayload",
    "KeywordLinkPayload",
    "KeywordUpsertPayload",
    "NotesOrganizationValidationError",
    "new_organization_sync_id",
    "organization_link_id",
    "parse_notes_organization_payload",
    "validate_organization_object_id",
    "validate_resource_sync_id",
]
