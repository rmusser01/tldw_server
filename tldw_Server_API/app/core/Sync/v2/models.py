from __future__ import annotations

"""Internal storage models for Sync v2 M1."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, cast
from uuid import UUID

from .notes_link_contract import (
    NOTES_LINK_LABEL_MAX_CHARS,
    NOTES_LINK_PROPERTIES_MAX_BYTES,
    NOTES_LINK_PROPERTIES_MAX_DEPTH,
    NOTES_LINK_PROPERTIES_MAX_KEYS,
    NOTES_LINK_REASON_MAX_CHARS,
    NOTES_LINK_WEIGHT_MAX,
)

SyncDomain = Literal[
    "notes.note",
    "chat.conversation",
    "chat.message",
    "attachment.ref",
    "workspaces.workspace",
    "workspaces.source_ref",
    "source_cache.entry",
    "media.item",
    "media.keyword",
    "media.keyword_link",
    "notes.keyword",
    "notes.keyword_link",
    "notes.keyword_collection",
    "notes.keyword_collection_link",
    "notes.folder",
    "notes.folder_link",
    "notes.link",
    "notes.task",
    "notes.task_activity",
    "notes.moodboard",
    "notes.moodboard_note",
    "notes.studio_document",
]
SyncOperation = Literal["upsert", "append", "tombstone"]
DatasetScopeType = Literal["personal", "workspace"]
EncryptionPolicy = Literal[
    "server_trusted_v1",
    "passphrase_wrapped_v1",
    "device_wrapped_v1",
    "client_private_v1",
]
SyncKeyWrappedFor = Literal["server", "passphrase", "device", "recovery"]
SyncKeyRewrapStatus = Literal["not_required", "pending", "complete", "failed", "blocked"]
ConflictStatus = Literal["unresolved", "resolved", "dismissed"]
SyncApplyStatus = Literal["pending", "applied", "failed", "conflict", "superseded"]
SyncBlobAvailabilityStatus = Literal[
    "metadata_only",
    "uploading",
    "available",
    "deleting",
    "verify_failed",
    "quarantined",
    "deleted",
]
SyncAttachmentBindingAvailability = Literal["available", "metadata_only"]


def normalize_sync_timestamp(value: object | None) -> str | None:
    """Normalize backend-native and ISO timestamps to the canonical UTC string."""

    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


SyncBlobUploadStatus = Literal[
    "created",
    "uploading",
    "complete",
    "cancelled",
    "expired",
    "verify_failed",
]
SyncDeviceStatus = Literal["pending_authorization", "active", "paused", "revoked"]
SyncDeviceAuthorizationStatus = Literal["pending", "approved", "rejected"]
SyncBackgroundLeaseStatus = Literal["acquired", "refreshed", "held_by_other"]
SyncRestoreCompletenessStatus = Literal[
    "metadata_ready",
    "blocked_by_conflicts",
    "blob_incomplete",
    "content_complete",
    "verified_complete",
]

M1_SYNC_DOMAINS: list[SyncDomain] = [
    "notes.note",
    "chat.conversation",
    "chat.message",
    "attachment.ref",
]
M1_SYNC_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    "notes.note": ["upsert", "tombstone"],
    "chat.conversation": ["upsert", "tombstone"],
    "chat.message": ["append", "tombstone"],
    "attachment.ref": ["upsert", "tombstone"],
}
NOTES_ORGANIZATION_DOMAINS: tuple[SyncDomain, ...] = (
    "notes.keyword",
    "notes.keyword_link",
    "notes.keyword_collection",
    "notes.keyword_collection_link",
    "notes.folder",
    "notes.folder_link",
)
NOTES_ORGANIZATION_SYNC_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    domain: ["upsert", "tombstone"] for domain in NOTES_ORGANIZATION_DOMAINS
}
NOTES_LINK_DOMAINS: tuple[SyncDomain, ...] = ("notes.link",)
NOTES_LINK_SYNC_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    "notes.link": ["upsert", "tombstone"]
}
NOTES_TASK_SYNC_DOMAINS: tuple[SyncDomain, ...] = (
    "notes.task",
    "notes.task_activity",
)
NOTES_TASK_SYNC_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    domain: ["upsert", "tombstone"] for domain in NOTES_TASK_SYNC_DOMAINS
}
NOTES_MOODBOARD_STUDIO_DOMAINS: tuple[SyncDomain, ...] = (
    "notes.moodboard",
    "notes.moodboard_note",
    "notes.studio_document",
)
NOTES_MOODBOARD_STUDIO_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    domain: ["upsert", "tombstone"] for domain in NOTES_MOODBOARD_STUDIO_DOMAINS
}
WORKSPACE_SYNC_DOMAINS: list[SyncDomain] = [
    "workspaces.workspace",
    "workspaces.source_ref",
]
WORKSPACE_SYNC_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    "workspaces.workspace": ["upsert", "tombstone"],
    "workspaces.source_ref": ["upsert", "tombstone"],
}
SOURCE_CACHE_SYNC_DOMAINS: list[SyncDomain] = ["source_cache.entry"]
SOURCE_CACHE_SYNC_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    "source_cache.entry": ["upsert", "tombstone"],
}
MEDIA_SYNC_DOMAINS: list[SyncDomain] = [
    "media.item",
    "media.keyword",
    "media.keyword_link",
]
MEDIA_SYNC_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    "media.item": ["upsert", "tombstone"],
    "media.keyword": ["upsert", "tombstone"],
    "media.keyword_link": ["upsert", "tombstone"],
}
SYNC_V2_SUPPORTED_DOMAINS: list[SyncDomain] = (
    list(M1_SYNC_DOMAINS)
    + list(WORKSPACE_SYNC_DOMAINS)
    + list(SOURCE_CACHE_SYNC_DOMAINS)
    + list(MEDIA_SYNC_DOMAINS)
    + list(NOTES_ORGANIZATION_DOMAINS)
    + list(NOTES_LINK_DOMAINS)
)
SYNC_V2_SUPPORTED_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    **M1_SYNC_OPERATIONS,
    **WORKSPACE_SYNC_OPERATIONS,
    **SOURCE_CACHE_SYNC_OPERATIONS,
    **MEDIA_SYNC_OPERATIONS,
    **NOTES_ORGANIZATION_SYNC_OPERATIONS,
    **NOTES_LINK_SYNC_OPERATIONS,
}
SYNC_V2_KNOWN_DOMAINS: tuple[SyncDomain, ...] = (
    *SYNC_V2_SUPPORTED_DOMAINS,
    *NOTES_TASK_SYNC_DOMAINS,
    *NOTES_MOODBOARD_STUDIO_DOMAINS,
)
SYNC_V2_INTERNAL_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    **SYNC_V2_SUPPORTED_OPERATIONS,
    **NOTES_TASK_SYNC_OPERATIONS,
    **NOTES_MOODBOARD_STUDIO_OPERATIONS,
}
SYNC_V2_MAX_ADAPTER_VERSION_DOMAINS = 100
SYNC_V2_MAX_ADAPTER_VERSIONS_PER_DOMAIN = 8
DEFAULT_M1_ENCRYPTION_POLICY: EncryptionPolicy = "server_trusted_v1"
SYNC_V2_ENCRYPTION_POLICIES: list[EncryptionPolicy] = [
    "server_trusted_v1",
    "passphrase_wrapped_v1",
    "device_wrapped_v1",
    "client_private_v1",
]
STRICT_ENCRYPTION_POLICIES: list[EncryptionPolicy] = [
    "passphrase_wrapped_v1",
    "device_wrapped_v1",
    "client_private_v1",
]
CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE = "sync_server_frontend_client_private_disabled"
SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION = (
    "sync_rebase_required_after_conflict_resolution"
)
CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_MESSAGE = (
    "Server-front-end mutation is disabled for client_private_v1 datasets "
    "because opaque fields cannot be inspected or re-encrypted by the server."
)
NOTES_NOTE_TITLE_MAX_CHARS = 255
NOTES_NOTE_CONTENT_MAX_CHARS = 5_000_000
NOTES_NOTE_CANONICAL_PAYLOAD_FIELDS: frozenset[str] = frozenset(
    {"title", "content", "conversation_id", "message_id"}
)
SYNC_KEY_WRAPPED_FOR_VALUES: list[SyncKeyWrappedFor] = [
    "server",
    "passphrase",
    "device",
    "recovery",
]
SYNC_KEY_REWRAP_STATUSES: list[SyncKeyRewrapStatus] = [
    "not_required",
    "pending",
    "complete",
    "failed",
    "blocked",
]


def _discoverable_pydantic_object_schema(model: type[Any]) -> dict[str, object]:
    """Translate a Pydantic object schema into the Sync discovery vocabulary."""

    generated = model.model_json_schema()
    properties: dict[str, object] = {}
    key_map = {"minLength": "min_length", "maxLength": "max_length"}
    for field_name, raw_field in generated["properties"].items():
        field_schema = {
            key_map.get(key, key): value
            for key, value in raw_field.items()
            if key not in {"title", "default", "const", "anyOf"}
        }
        if "const" in raw_field:
            field_schema["enum"] = [raw_field["const"]]
        if "anyOf" in raw_field:
            variants = raw_field["anyOf"]
            field_schema["type"] = [variant["type"] for variant in variants]
            for variant in variants:
                if "maxLength" in variant:
                    field_schema["max_length"] = variant["maxLength"]
        if field_name in {"attachment_id", "parent_object_id"}:
            field_schema["canonical_lowercase"] = True
        elif field_name == "blob_hash":
            field_schema.update(format="sha256", canonical_lowercase=True)
        elif field_name in {"created_at", "last_modified", "deleted_at"}:
            field_schema["format"] = "date-time"
        properties[field_name] = field_schema
    return {
        "required": generated["required"],
        "properties": properties,
        "additional_properties": generated["additionalProperties"],
    }


def sync_v2_domain_schemas() -> dict[SyncDomain, dict[str, object]]:
    """Return client-discoverable payload contracts for versioned Sync domains."""

    from .attachment_refs_v2 import (
        AttachmentRefV2Payload,
        AttachmentRefV2TombstonePayload,
    )

    keyword_link_schema = {
        "required": ["subject_type", "subject_id", "keyword_sync_id"],
        "properties": {
            "subject_type": {"enum": ["note", "conversation"]},
            "subject_id": {"type": "string"},
            "keyword_sync_id": {"type": "string"},
        },
        "additional_properties": False,
    }
    collection_link_schema = {
        "required": ["collection_sync_id", "keyword_sync_id"],
        "properties": {
            "collection_sync_id": {"type": "string"},
            "keyword_sync_id": {"type": "string"},
        },
        "additional_properties": False,
    }
    folder_link_schema = {
        "required": ["note_id", "folder_sync_id"],
        "properties": {
            "note_id": {"type": "string"},
            "folder_sync_id": {"type": "string"},
        },
        "additional_properties": False,
    }
    notes_link_required = [
        "source_note_id",
        "target_note_id",
        "type",
        "directed",
        "weight",
        "label",
        "properties",
        "created_at",
        "last_modified",
        "created_by",
    ]
    notes_link_properties = {
        "source_note_id": {
            "type": "string",
            "format": "uuid4",
            "canonical_lowercase": True,
        },
        "target_note_id": {
            "type": "string",
            "format": "uuid4",
            "canonical_lowercase": True,
        },
        "type": {"enum": ["manual"]},
        "directed": {"type": "boolean"},
        "weight": {"type": "number", "minimum": 0, "maximum": NOTES_LINK_WEIGHT_MAX},
        "label": {"type": ["string", "null"], "max_length": NOTES_LINK_LABEL_MAX_CHARS},
        "properties": {
            "type": "object",
            "max_properties": NOTES_LINK_PROPERTIES_MAX_KEYS,
            "max_depth": NOTES_LINK_PROPERTIES_MAX_DEPTH,
            "max_bytes": NOTES_LINK_PROPERTIES_MAX_BYTES,
        },
        "created_at": {"type": "string", "format": "date-time"},
        "last_modified": {"type": "string", "format": "date-time"},
        "created_by": {"type": "string"},
    }
    notes_link_constraints = {
        "distinct_endpoints": True,
        "undirected_endpoint_order": "source_note_id <= target_note_id",
    }
    attachment_ref_upsert = _discoverable_pydantic_object_schema(
        AttachmentRefV2Payload
    )
    attachment_ref_tombstone = _discoverable_pydantic_object_schema(
        AttachmentRefV2TombstonePayload
    )
    return {
        "attachment.ref": {
            "schema_version": 2,
            "encryption_policy": DEFAULT_M1_ENCRYPTION_POLICY,
            "upsert": attachment_ref_upsert,
            "tombstone": attachment_ref_tombstone,
            "restore": {
                "operation": "upsert",
                "routing_metadata": {"restore_intent": True},
                "requires_current_base": True,
            },
            "derived_fields": [
                "availability",
                "resolved_blob_id",
                "storage_status",
                "retention_released_at",
            ],
        },
        "notes.note": {
            "schema_version": 1,
            "encryption_policy": DEFAULT_M1_ENCRYPTION_POLICY,
            "upsert": {
                "required": ["title", "content"],
                "properties": {
                    "title": {"type": "string", "max_length": NOTES_NOTE_TITLE_MAX_CHARS},
                    "content": {"type": "string", "max_length": NOTES_NOTE_CONTENT_MAX_CHARS},
                    "conversation_id": {"type": ["string", "null"]},
                    "message_id": {"type": ["string", "null"]},
                },
                "additional_properties": False,
            },
            "tombstone": {"operation": "tombstone"},
            "restore": {
                "operation": "upsert",
                "routing_metadata": {"restore_intent": True},
                "requires_current_base": True,
            },
        },
        "notes.keyword": {
            "schema_version": 1,
            "encryption_policy": DEFAULT_M1_ENCRYPTION_POLICY,
            "upsert": {
                "required": ["keyword"],
                "properties": {"keyword": {"type": "string", "max_length": 100}},
                "additional_properties": False,
            },
            "tombstone": {"operation": "tombstone"},
        },
        "notes.keyword_link": {
            "schema_version": 1,
            "encryption_policy": DEFAULT_M1_ENCRYPTION_POLICY,
            "upsert": keyword_link_schema,
            "tombstone": keyword_link_schema,
        },
        "notes.keyword_collection": {
            "schema_version": 1,
            "encryption_policy": DEFAULT_M1_ENCRYPTION_POLICY,
            "upsert": {
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "max_length": 255},
                    "parent_sync_id": {"type": ["string", "null"]},
                },
                "additional_properties": False,
            },
            "tombstone": {"operation": "tombstone"},
        },
        "notes.keyword_collection_link": {
            "schema_version": 1,
            "encryption_policy": DEFAULT_M1_ENCRYPTION_POLICY,
            "upsert": collection_link_schema,
            "tombstone": collection_link_schema,
        },
        "notes.folder": {
            "schema_version": 1,
            "encryption_policy": DEFAULT_M1_ENCRYPTION_POLICY,
            "upsert": {
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "max_length": 500},
                    "parent_sync_id": {"type": ["string", "null"]},
                },
                "additional_properties": False,
            },
            "tombstone": {"operation": "tombstone"},
        },
        "notes.folder_link": {
            "schema_version": 1,
            "encryption_policy": DEFAULT_M1_ENCRYPTION_POLICY,
            "upsert": folder_link_schema,
            "tombstone": folder_link_schema,
        },
        "notes.link": {
            "schema_version": 1,
            "encryption_policy": DEFAULT_M1_ENCRYPTION_POLICY,
            "upsert": {
                "required": notes_link_required,
                "properties": notes_link_properties,
                "additional_properties": False,
                "constraints": notes_link_constraints,
            },
            "tombstone": {
                "required": [*notes_link_required, "deleted_at"],
                "properties": {
                    **notes_link_properties,
                    "deleted_at": {"type": "string", "format": "date-time"},
                    "reason": {
                        "type": ["string", "null"],
                        "max_length": NOTES_LINK_REASON_MAX_CHARS,
                    },
                },
                "additional_properties": False,
                "constraints": notes_link_constraints,
            },
        },
    }


def _sync_v2_internal_domain_schemas() -> dict[SyncDomain, dict[str, object]]:
    """Return private known-domain contracts without advertising dormant domains."""

    from .notes_moodboard_studio_contract import (
        NotesMoodboardNoteV1,
        NotesMoodboardV1,
        NotesStudioDocumentV1,
    )
    from .notes_task_contract import (
        NotesTaskActivityTombstoneV1,
        NotesTaskActivityV1,
        NotesTaskV1Payload,
    )

    schemas = sync_v2_domain_schemas()
    task_schema = NotesTaskV1Payload.model_json_schema()
    activity_schema = NotesTaskActivityV1.model_json_schema()
    activity_tombstone_schema = NotesTaskActivityTombstoneV1.model_json_schema()
    moodboard_schema = NotesMoodboardV1.model_json_schema()
    placement_schema = NotesMoodboardNoteV1.model_json_schema()
    studio_schema = NotesStudioDocumentV1.model_json_schema()
    schemas.update(
        {
            "notes.task": {
                "schema_version": 1,
                "operations": ["upsert", "tombstone"],
                "upsert": task_schema,
                "tombstone": task_schema,
            },
            "notes.task_activity": {
                "schema_version": 1,
                "operations": ["upsert", "tombstone"],
                "upsert": activity_schema,
                "tombstone": activity_tombstone_schema,
            },
            "notes.moodboard": {
                "schema_version": 1,
                "operations": ["upsert", "tombstone"],
                "upsert": moodboard_schema,
                "tombstone": moodboard_schema,
            },
            "notes.moodboard_note": {
                "schema_version": 1,
                "operations": ["upsert", "tombstone"],
                "upsert": placement_schema,
                "tombstone": placement_schema,
            },
            "notes.studio_document": {
                "schema_version": 1,
                "operations": ["upsert", "tombstone"],
                "upsert": studio_schema,
                "tombstone": studio_schema,
            },
        }
    )
    return schemas


def sync_v2_advertised_domain_schemas(
    domain_schemas: Mapping[SyncDomain, dict[str, object]],
    *,
    advertised_domains: Sequence[SyncDomain],
) -> dict[SyncDomain, dict[str, object]]:
    """Select schemas that are already approved for public advertisement."""

    return {
        domain: domain_schemas[domain]
        for domain in advertised_domains
        if domain in domain_schemas
    }


def sync_v2_server_supported_adapter_versions(
    *,
    notes_task_sync_ready: bool = False,
) -> dict[SyncDomain, list[int]]:
    """Return bounded server-supported versions independently of writability."""

    domains = [
        *SYNC_V2_SUPPORTED_DOMAINS,
        *(NOTES_TASK_SYNC_DOMAINS if notes_task_sync_ready else ()),
    ]
    return {
        domain: ([1, 2] if domain == "attachment.ref" else [1])
        for domain in domains
    }


def sync_v2_dataset_writable_adapter_versions(
    dataset: SyncDataset | None = None,
    *,
    notes_attachment_sync_enabled: bool = False,
    supports_attachments: bool = False,
    notes_task_sync_ready: bool = False,
) -> dict[SyncDomain, list[int]]:
    """Return versions writable under one authoritative dataset/settings gate."""

    versions: dict[SyncDomain, list[int]] = {
        domain: []
        for domain in SYNC_V2_SUPPORTED_DOMAINS
    }
    if dataset is None:
        return versions
    enrolled = set(dataset.domains)
    for domain in SYNC_V2_SUPPORTED_DOMAINS:
        if domain in enrolled and domain != "attachment.ref":
            versions[domain] = [1]
    if sync_v2_attachment_ref_v2_is_writable(
        dataset,
        notes_attachment_sync_enabled=notes_attachment_sync_enabled,
        supports_attachments=supports_attachments,
    ):
        versions["attachment.ref"] = [2]
    if notes_task_sync_ready:
        versions.update(dict.fromkeys(NOTES_TASK_SYNC_DOMAINS, [1]))
    return versions


def sync_v2_attachment_ref_v2_is_writable(
    dataset: SyncDataset | None,
    *,
    notes_attachment_sync_enabled: bool,
    supports_attachments: bool,
) -> bool:
    """Return whether attachment.ref v2 mutations are writable for a dataset."""

    if dataset is None:
        return False
    attachment_state = dataset.metadata.get("notes_attachment_v2")
    return bool(
        notes_attachment_sync_enabled
        and supports_attachments
        and dataset.encryption_policy == DEFAULT_M1_ENCRYPTION_POLICY
        and {"notes.note", "attachment.ref"}.issubset(dataset.domains)
        and isinstance(attachment_state, Mapping)
        and attachment_state.get("state") == "ready"
    )


def normalize_sync_v2_requested_domains(value: object) -> list[SyncDomain]:
    """Validate, bound, and deduplicate one requested Sync-domain sequence."""

    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise ValueError("requested_domains must be a list")
    domains = list(value)
    if len(domains) > SYNC_V2_MAX_ADAPTER_VERSION_DOMAINS:
        raise ValueError(
            "requested_domains may contain at most "
            f"{SYNC_V2_MAX_ADAPTER_VERSION_DOMAINS} domains"
        )
    known = set(SYNC_V2_KNOWN_DOMAINS)
    for domain in domains:
        if not isinstance(domain, str) or domain not in known:
            raise ValueError(
                f"requested_domains contains unknown Sync domain: {domain}"
            )
    normalized = [cast(SyncDomain, domain) for domain in dict.fromkeys(domains)]
    requested_task_domains = set(normalized).intersection(NOTES_TASK_SYNC_DOMAINS)
    if requested_task_domains and requested_task_domains != set(
        NOTES_TASK_SYNC_DOMAINS
    ):
        raise ValueError("requested_domains must include both Notes task domains")
    return normalized


def normalize_supported_adapter_versions(
    value: object | None,
    *,
    requested_domains: Sequence[str],
) -> dict[SyncDomain, list[int]]:
    """Validate a bounded device version map; omission preserves version 1."""

    requested = normalize_sync_v2_requested_domains(requested_domains)
    if value is None:
        return {domain: [1] for domain in requested}
    if not isinstance(value, Mapping):
        raise ValueError("supported_adapter_versions must be an object")
    if len(value) > SYNC_V2_MAX_ADAPTER_VERSION_DOMAINS:
        raise ValueError(
            "supported_adapter_versions may contain at most "
            f"{SYNC_V2_MAX_ADAPTER_VERSION_DOMAINS} domains"
        )

    known = set(SYNC_V2_KNOWN_DOMAINS)
    requested_set = set(requested)
    normalized: dict[SyncDomain, list[int]] = {
        domain: [1] for domain in requested
    }
    for raw_domain, raw_versions in value.items():
        if not isinstance(raw_domain, str) or raw_domain not in known:
            raise ValueError(
                f"supported_adapter_versions contains unknown Sync domain: {raw_domain}"
            )
        if raw_domain not in requested_set:
            raise ValueError(
                "supported_adapter_versions domains must also be requested"
            )
        if not isinstance(raw_versions, Sequence) or isinstance(
            raw_versions, (str, bytes, bytearray)
        ):
            raise ValueError(
                "supported_adapter_versions values must be non-empty version lists"
            )
        versions = list(raw_versions)
        if not versions:
            raise ValueError(
                "supported_adapter_versions values must be non-empty version lists"
            )
        if len(versions) > SYNC_V2_MAX_ADAPTER_VERSIONS_PER_DOMAIN:
            raise ValueError(
                "supported_adapter_versions may contain at most "
                f"{SYNC_V2_MAX_ADAPTER_VERSIONS_PER_DOMAIN} versions per domain"
            )
        if any(isinstance(version, bool) or not isinstance(version, int) or version < 1 for version in versions):
            raise ValueError(
                "supported_adapter_versions must contain positive integers"
            )
        if len(set(versions)) != len(versions):
            raise ValueError("supported_adapter_versions contains duplicate adapter versions")
        normalized[raw_domain] = sorted(versions)
    return normalized


def server_frontend_mutation_enabled_for_policy(policy: EncryptionPolicy | str) -> bool:
    """Return whether server-origin writes can safely materialize this policy."""

    return policy != "client_private_v1"


def server_frontend_mutation_blockers_for_policy(policy: EncryptionPolicy | str) -> list[str]:
    """Return stable blocker codes for server-front-end mutation under a policy."""

    if server_frontend_mutation_enabled_for_policy(policy):
        return []
    return [CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE]


def client_private_server_frontend_limitation_warning() -> dict[str, str]:
    """Return the public warning for client-private server-front-end limitations."""

    return {
        "code": CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE,
        "message": CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_MESSAGE,
    }


def validate_notes_note_upsert_payload(
    payload: Mapping[str, object],
) -> dict[str, str | None]:
    """Validate and return the lossless version-1 ``notes.note`` payload."""

    unexpected = set(payload).difference(NOTES_NOTE_CANONICAL_PAYLOAD_FIELDS)
    if unexpected:
        raise ValueError(
            "notes.note upsert payload contains unsupported fields: "
            + ", ".join(sorted(unexpected))
        )

    title = payload.get("title")
    content = payload.get("content")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("notes.note upsert payload requires a non-empty string title")
    if len(title) > NOTES_NOTE_TITLE_MAX_CHARS:
        raise ValueError(
            f"notes.note title must be at most {NOTES_NOTE_TITLE_MAX_CHARS} characters"
        )
    if not isinstance(content, str) or not content:
        raise ValueError("notes.note upsert payload requires non-empty string content")
    if len(content) > NOTES_NOTE_CONTENT_MAX_CHARS:
        raise ValueError(
            f"notes.note content must be at most {NOTES_NOTE_CONTENT_MAX_CHARS} characters"
        )

    backlinks: dict[str, str | None] = {}
    for field_name in ("conversation_id", "message_id"):
        value = payload.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"notes.note {field_name} must be a string or null")
        backlinks[field_name] = value

    return {
        "title": title,
        "content": content,
        **backlinks,
    }


@dataclass(frozen=True, slots=True)
class SyncEncryptionPolicyMetadata:
    """Validated public metadata for a Sync v2 dataset encryption policy."""

    policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    key_epoch: int = 1
    attestation: dict[str, Any] = field(default_factory=dict)
    kdf_metadata: dict[str, Any] = field(default_factory=dict)
    recovery_key_record_id: str | None = None
    device_key_record_ids: list[str] = field(default_factory=list)
    server_materialization: str | None = None

    def __post_init__(self) -> None:
        attestation = dict(self.attestation)
        kdf_metadata = dict(self.kdf_metadata)
        device_key_record_ids = [
            str(record_id).strip()
            for record_id in self.device_key_record_ids
            if str(record_id).strip()
        ]
        _validate_encryption_policy_metadata(
            policy=self.policy,
            key_epoch=self.key_epoch,
            attestation=attestation,
            kdf_metadata=kdf_metadata,
            recovery_key_record_id=self.recovery_key_record_id,
            device_key_record_ids=device_key_record_ids,
            server_materialization=self.server_materialization,
        )
        object.__setattr__(self, "attestation", attestation)
        object.__setattr__(self, "kdf_metadata", kdf_metadata)
        object.__setattr__(self, "device_key_record_ids", device_key_record_ids)


def _validate_encryption_policy_metadata(
    *,
    policy: EncryptionPolicy,
    key_epoch: int,
    attestation: dict[str, Any],
    kdf_metadata: dict[str, Any],
    recovery_key_record_id: str | None,
    device_key_record_ids: list[str],
    server_materialization: str | None,
) -> None:
    if policy not in SYNC_V2_ENCRYPTION_POLICIES:
        raise ValueError(f"unsupported Sync v2 encryption policy: {policy}")
    if isinstance(key_epoch, bool) or key_epoch < 1:
        raise ValueError("encryption policy key_epoch must be greater than or equal to 1")
    if policy == "server_trusted_v1":
        _validate_server_trusted_policy_metadata(attestation)
        return
    if policy == "passphrase_wrapped_v1":
        _validate_passphrase_wrapped_policy_metadata(
            kdf_metadata=kdf_metadata,
            recovery_key_record_id=recovery_key_record_id,
        )
        return
    if policy == "device_wrapped_v1":
        if not device_key_record_ids:
            raise ValueError("device_wrapped_v1 requires at least one device key record")
        return
    if policy == "client_private_v1" and server_materialization != "metadata_only":
        raise ValueError("client_private_v1 requires metadata_only server materialization")


def _validate_server_trusted_policy_metadata(attestation: dict[str, Any]) -> None:
    if attestation.get("configured") is not True:
        raise ValueError("server_trusted_v1 requires configured attestation metadata")
    if not str(attestation.get("scope") or "").strip():
        raise ValueError("server_trusted_v1 requires attestation scope metadata")
    covers = attestation.get("covers")
    if not isinstance(covers, list) or not any(str(item).strip() for item in covers):
        raise ValueError("server_trusted_v1 requires covered storage metadata")


def _validate_passphrase_wrapped_policy_metadata(
    *,
    kdf_metadata: dict[str, Any],
    recovery_key_record_id: str | None,
) -> None:
    if not str(kdf_metadata.get("algorithm") or "").strip():
        raise ValueError("passphrase_wrapped_v1 requires KDF algorithm metadata")
    params_hash = str(kdf_metadata.get("params_hash") or "").strip()
    if not params_hash.startswith("sha256:") or params_hash == "sha256:":
        raise ValueError("passphrase_wrapped_v1 requires a sha256 KDF params hash")
    if not str(recovery_key_record_id or "").strip():
        raise ValueError("passphrase_wrapped_v1 requires a recovery key record reference")


def _coalesce_identity(primary: str | None, legacy: str | None, *, field_name: str) -> str:
    value = primary or legacy
    if not value:
        raise ValueError(f"{field_name} is required")
    return value


def _coalesce_payload(
    payload: dict[str, Any],
    payload_clear: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if payload:
        return dict(payload), dict(payload)
    if payload_clear:
        return {}, dict(payload_clear)
    return {}, {}


@dataclass(frozen=True, slots=True)
class SyncDeviceUpsert:
    """Device registration data accepted by the Sync v2 store."""

    device_id: str
    user_id: str
    display_name: str
    client_type: str
    client_version: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    status: SyncDeviceStatus = "active"
    user_label: str | None = None
    authorized_at: str | None = None
    revoked_at: str | None = None
    revoked_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDevice:
    """Stored Sync v2 device metadata."""

    device_id: str
    user_id: str
    display_name: str
    client_type: str
    client_version: str | None
    capabilities: dict[str, Any]
    registered_at: str
    last_seen_at: str
    status: SyncDeviceStatus = "active"
    user_label: str | None = None
    authorized_at: str | None = None
    revoked_at: str | None = None
    revoked_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceAuthorizationCreate:
    """Device authorization request accepted by the Sync v2 store."""

    authorization_id: str
    dataset_id: str
    user_id: str
    device_id: str
    authorization_method: str
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceAuthorization:
    """Stored device authorization request."""

    authorization_id: str
    dataset_id: str
    user_id: str
    device_id: str
    authorization_method: str
    status: SyncDeviceAuthorizationStatus
    requested_at: str
    approved_at: str | None = None
    approving_device_id: str | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceDomainAckCreate:
    """Per-device domain acknowledgment accepted by the Sync v2 store."""

    dataset_id: str
    device_id: str
    domain: SyncDomain
    through_server_sequence: int
    applied_at: str
    adapter_version: int = 1
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        if self.adapter_version < 1:
            raise ValueError("Sync adapter version must be positive")


@dataclass(frozen=True, slots=True)
class SyncDeviceDomainAck:
    """Stored per-device domain acknowledgment."""

    dataset_id: str
    device_id: str
    domain: SyncDomain
    through_server_sequence: int
    applied_at: str
    updated_at: str
    adapter_version: int = 1
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceBlobAckCreate:
    """Per-device blob verification acknowledgment accepted by the Sync v2 store."""

    dataset_id: str
    device_id: str
    attachment_id: str
    payload_hash: str
    verified_at: str
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceBlobAck:
    """Stored per-device blob verification acknowledgment."""

    dataset_id: str
    device_id: str
    attachment_id: str
    payload_hash: str
    verified_at: str
    updated_at: str
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceBlobIdAckCreate:
    """Immutable blob-ID verification evidence accepted from a v2 device."""

    dataset_id: str
    device_id: str
    blob_id: str
    payload_hash: str
    verified_at: str
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceBlobIdAck:
    """Stored immutable blob-ID verification evidence."""

    dataset_id: str
    device_id: str
    blob_id: str
    payload_hash: str
    verified_at: str
    updated_at: str
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceAcknowledgmentSummary:
    """Aggregated device acknowledgments for one dataset/device."""

    dataset_id: str
    device_id: str
    domain_acks: dict[SyncDomain, SyncDeviceDomainAck] = field(default_factory=dict)
    blob_acks: list[SyncDeviceBlobAck] = field(default_factory=list)
    version_acks: list[SyncDeviceDomainAck] = field(default_factory=list)
    blob_id_acks: list[SyncDeviceBlobIdAck] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SyncBackgroundPolicyUpsert:
    """Background sync policy and user intent accepted by the Sync v2 store."""

    dataset_id: str
    device_id: str
    enabled: bool = True
    minimum_interval_seconds: int = 300
    backoff_floor_seconds: int = 60
    max_batch_size: int = 100
    max_blob_bytes_per_run: int | None = None
    respect_metered_networks: bool = True
    maintenance_window: dict[str, Any] | None = None
    paused_reason: str | None = None
    pending_local_changes: bool = False


@dataclass(frozen=True, slots=True)
class SyncBackgroundPolicy:
    """Stored background sync policy and user intent for one dataset/device."""

    dataset_id: str
    device_id: str
    enabled: bool
    minimum_interval_seconds: int
    backoff_floor_seconds: int
    max_batch_size: int
    max_blob_bytes_per_run: int | None
    respect_metered_networks: bool
    maintenance_window: dict[str, Any] | None
    paused_reason: str | None
    pending_local_changes: bool
    updated_at: str


@dataclass(frozen=True, slots=True)
class SyncBackgroundLeaseCreate:
    """Advisory background sync lease request accepted by the Sync v2 store."""

    dataset_id: str
    device_id: str
    lease_id: str
    ttl_seconds: int
    requested_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncBackgroundLease:
    """Stored advisory background sync lease."""

    dataset_id: str
    device_id: str
    lease_id: str
    status: SyncBackgroundLeaseStatus
    acquired: bool
    expires_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class SyncBackgroundDomainStatus:
    """Aggregated background sync status for one domain."""

    domain: SyncDomain
    last_server_sequence: int = 0
    last_pulled_sequence: int = 0
    cursor_lag_count: int = 0
    unresolved_conflicts: int = 0
    replayable_failures: int = 0
    last_successful_push_at: str | None = None
    last_successful_pull_at: str | None = None
    blob_completeness: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncDatasetCreate:
    """Dataset enrollment data accepted by the Sync v2 store."""

    dataset_id: str
    owner_user_id: str
    scope_type: DatasetScopeType = "personal"
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    domains: list[SyncDomain] = field(default_factory=lambda: list(M1_SYNC_DOMAINS))
    workspace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    archived_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDataset:
    """Stored Sync v2 dataset metadata."""

    dataset_id: str
    owner_user_id: str
    scope_type: DatasetScopeType
    encryption_policy: EncryptionPolicy
    domains: list[SyncDomain]
    workspace_id: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    archived_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncEnvelopeCreate:
    """Envelope data accepted by the Sync v2 store."""

    dataset_id: str
    client_envelope_id: str
    domain: SyncDomain
    operation: SyncOperation
    object_id: str | None = None
    entity_id: str | None = None
    device_id: str | None = None
    client_profile_id: str | None = None
    client_sequence: int | None = None
    server_cursor: int | None = None
    server_sequence: int | None = None
    base_server_cursor: int | None = None
    base_object_revision: int | None = None
    base_object_hash: str | None = None
    object_revision: int | None = None
    parent_id: str | None = None
    schema_version: int = 1
    payload: dict[str, Any] = field(default_factory=dict)
    payload_hash: str | None = None
    payload_size_bytes: int | None = None
    created_at_client: str | None = None
    received_at_server: str | None = None
    deleted: bool = False
    encryption_metadata: dict[str, Any] = field(
        default_factory=lambda: {"policy": DEFAULT_M1_ENCRYPTION_POLICY}
    )
    status: str = "accepted"
    apply_status: SyncApplyStatus = "pending"
    apply_error_code: str | None = None
    apply_error_message: str | None = None
    applied_at: str | None = None
    payload_ciphertext: str | None = None
    payload_clear: dict[str, Any] = field(default_factory=dict)
    stable_key: str | None = None
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    routing_metadata: dict[str, Any] = field(default_factory=dict)
    adapter_version: int = 1
    base_version: str | int | None = None
    entity_version: str | int | None = None
    client_timestamp: str | None = None
    server_timestamp: str | None = None
    mutation_group_id: str | None = None
    mutation_step: int | None = None
    mutation_step_count: int | None = None
    mutation_plan_hash: str | None = None

    def __post_init__(self) -> None:
        _validate_mutation_group_metadata(
            mutation_group_id=self.mutation_group_id,
            mutation_step=self.mutation_step,
            mutation_step_count=self.mutation_step_count,
            mutation_plan_hash=self.mutation_plan_hash,
        )
        object_id = _coalesce_identity(self.object_id, self.entity_id, field_name="object_id")
        payload, payload_clear = _coalesce_payload(self.payload, self.payload_clear)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "entity_id", object_id)
        server_cursor = self.server_cursor if self.server_cursor is not None else self.server_sequence
        object.__setattr__(self, "server_cursor", server_cursor)
        object.__setattr__(self, "server_sequence", server_cursor)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "payload_clear", payload_clear)
        object.__setattr__(self, "schema_version", self.schema_version or self.adapter_version)
        object.__setattr__(self, "adapter_version", self.adapter_version or self.schema_version)
        created_at_client = normalize_sync_timestamp(
            self.created_at_client or self.client_timestamp
        )
        object.__setattr__(self, "created_at_client", created_at_client)
        object.__setattr__(self, "client_timestamp", created_at_client)
        if self.received_at_server is None and self.server_timestamp is not None:
            object.__setattr__(self, "received_at_server", self.server_timestamp)
        if self.server_timestamp is None and self.received_at_server is not None:
            object.__setattr__(self, "server_timestamp", self.received_at_server)
        if self.base_object_revision is None and isinstance(self.base_version, int):
            object.__setattr__(self, "base_object_revision", self.base_version)
        if self.object_revision is None and isinstance(self.entity_version, int):
            object.__setattr__(self, "object_revision", self.entity_version)


@dataclass(frozen=True, slots=True)
class SyncEnvelope:
    """Stored Sync v2 envelope."""

    dataset_id: str
    client_envelope_id: str
    domain: SyncDomain
    operation: SyncOperation
    server_cursor: int | None = None
    object_id: str | None = None
    entity_id: str | None = None
    server_sequence: int | None = None
    envelope_id: str | None = None
    device_id: str | None = None
    client_profile_id: str | None = None
    client_sequence: int | None = None
    base_server_cursor: int | None = None
    base_object_revision: int | None = None
    base_object_hash: str | None = None
    object_revision: int | None = None
    parent_id: str | None = None
    schema_version: int = 1
    payload: dict[str, Any] = field(default_factory=dict)
    payload_hash: str | None = None
    payload_size_bytes: int | None = None
    created_at_client: str | None = None
    received_at_server: str | None = None
    deleted: bool = False
    encryption_metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "accepted"
    apply_status: SyncApplyStatus = "pending"
    apply_error_code: str | None = None
    apply_error_message: str | None = None
    applied_at: str | None = None
    payload_ciphertext: str | None = None
    payload_clear: dict[str, Any] = field(default_factory=dict)
    stable_key: str | None = None
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    routing_metadata: dict[str, Any] = field(default_factory=dict)
    adapter_version: int = 1
    base_version: str | int | None = None
    entity_version: str | int | None = None
    client_timestamp: str | None = None
    server_timestamp: str | None = None
    mutation_group_id: str | None = None
    mutation_step: int | None = None
    mutation_step_count: int | None = None
    mutation_plan_hash: str | None = None

    def __post_init__(self) -> None:
        _validate_mutation_group_metadata(
            mutation_group_id=self.mutation_group_id,
            mutation_step=self.mutation_step,
            mutation_step_count=self.mutation_step_count,
            mutation_plan_hash=self.mutation_plan_hash,
        )
        object_id = _coalesce_identity(self.object_id, self.entity_id, field_name="object_id")
        server_cursor = self.server_cursor if self.server_cursor is not None else self.server_sequence
        if server_cursor is None:
            raise ValueError("server_cursor is required")
        payload, payload_clear = _coalesce_payload(self.payload, self.payload_clear)
        object.__setattr__(self, "server_cursor", server_cursor)
        object.__setattr__(self, "server_sequence", server_cursor)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "entity_id", object_id)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "payload_clear", payload_clear)
        object.__setattr__(self, "schema_version", self.schema_version or self.adapter_version)
        object.__setattr__(self, "adapter_version", self.adapter_version or self.schema_version)
        created_at_client = normalize_sync_timestamp(
            self.created_at_client or self.client_timestamp
        )
        object.__setattr__(self, "created_at_client", created_at_client)
        object.__setattr__(self, "client_timestamp", created_at_client)
        if self.received_at_server is None and self.server_timestamp is not None:
            object.__setattr__(self, "received_at_server", self.server_timestamp)
        if self.server_timestamp is None and self.received_at_server is not None:
            object.__setattr__(self, "server_timestamp", self.received_at_server)
        if self.base_object_revision is None and isinstance(self.base_version, int):
            object.__setattr__(self, "base_object_revision", self.base_version)
        if self.object_revision is None and isinstance(self.entity_version, int):
            object.__setattr__(self, "object_revision", self.entity_version)


def _validate_mutation_group_metadata(
    *,
    mutation_group_id: str | None,
    mutation_step: int | None,
    mutation_step_count: int | None,
    mutation_plan_hash: str | None,
) -> None:
    values = (
        mutation_group_id,
        mutation_step,
        mutation_step_count,
        mutation_plan_hash,
    )
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError("Sync mutation group metadata must be supplied as a complete set")
    if not isinstance(mutation_group_id, str) or not mutation_group_id.strip():
        raise ValueError("Sync mutation group id must be a non-empty string")
    if (
        isinstance(mutation_step, bool)
        or not isinstance(mutation_step, int)
        or mutation_step < 0
    ):
        raise ValueError("Sync mutation group step must be a zero-based integer")
    if (
        isinstance(mutation_step_count, bool)
        or not isinstance(mutation_step_count, int)
        or mutation_step_count <= 0
    ):
        raise ValueError("Sync mutation group step count must be a positive integer")
    if mutation_step >= mutation_step_count:
        raise ValueError("Sync mutation group step must be less than its step count")
    if (
        not isinstance(mutation_plan_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", mutation_plan_hash) is None
    ):
        raise ValueError("Sync mutation group plan hash must be lowercase SHA-256 hex")


@dataclass(frozen=True, slots=True)
class SyncDomainEnvelopeSummary:
    """Aggregate envelope health for one dataset domain."""

    domain: SyncDomain
    envelope_count: int = 0
    pending_apply_count: int = 0
    failed_apply_count: int = 0
    last_envelope: SyncEnvelope | None = None
    last_failed_envelope: SyncEnvelope | None = None


@dataclass(frozen=True, slots=True)
class SyncObjectState:
    """Materialized latest object state tracked by Sync v2."""

    dataset_id: str
    domain: SyncDomain
    object_id: str
    object_revision: int
    object_hash: str
    latest_server_cursor: int
    deleted: bool = False
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceCursor:
    """Per-device pull cursor for one domain."""

    dataset_id: str
    device_id: str
    domain: SyncDomain
    last_pulled_sequence: int
    adapter_version: int = 1
    max_delivered_sequence: int = 0
    updated_at: str | None = None

    def __post_init__(self) -> None:
        if self.adapter_version < 1:
            raise ValueError("Sync adapter version must be positive")
        if self.last_pulled_sequence < 0 or self.max_delivered_sequence < 0:
            raise ValueError("Sync cursor sequences must be non-negative")
        if self.max_delivered_sequence > self.last_pulled_sequence:
            raise ValueError("Sync delivered watermark cannot exceed scan cursor")


@dataclass(frozen=True, slots=True)
class SyncConflictCreate:
    """Conflict metadata accepted by the Sync v2 store."""

    conflict_id: str
    dataset_id: str
    domain: SyncDomain
    conflict_type: str
    object_id: str | None = None
    entity_id: str | None = None
    base_envelope_id: str | None = None
    local_envelope_id: str | None = None
    remote_envelope_id: str | None = None
    server_cursor: int | None = None
    server_sequence: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object_id = _coalesce_identity(self.object_id, self.entity_id, field_name="object_id")
        server_cursor = self.server_cursor if self.server_cursor is not None else self.server_sequence
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "entity_id", object_id)
        object.__setattr__(self, "server_cursor", server_cursor)
        object.__setattr__(self, "server_sequence", server_cursor)


@dataclass(frozen=True, slots=True)
class SyncConflict:
    """Stored Sync v2 conflict metadata."""

    conflict_id: str
    dataset_id: str
    domain: SyncDomain
    object_id: str | None
    conflict_type: str
    status: ConflictStatus
    base_envelope_id: str | None
    local_envelope_id: str | None
    remote_envelope_id: str | None
    server_cursor: int | None
    metadata: dict[str, Any]
    created_at: str
    entity_id: str | None = None
    server_sequence: int | None = None
    resolved_at: str | None = None
    resolved_by_envelope_id: str | None = None
    resolved_by_device_id: str | None = None
    resolution_action: str | None = None
    resolution_notes: str | None = None

    def __post_init__(self) -> None:
        object_id = _coalesce_identity(self.object_id, self.entity_id, field_name="object_id")
        server_cursor = self.server_cursor if self.server_cursor is not None else self.server_sequence
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "entity_id", object_id)
        object.__setattr__(self, "server_cursor", server_cursor)
        object.__setattr__(self, "server_sequence", server_cursor)


@dataclass(frozen=True, slots=True)
class SyncKeyRecordCreate:
    """Encrypted key material accepted by the Sync v2 store."""

    key_record_id: str
    dataset_id: str
    user_id: str
    key_purpose: str
    wrapped_key_blob: str
    device_id: str | None = None
    kdf_metadata: dict[str, Any] = field(default_factory=dict)
    recovery_hint: str | None = None
    rotation_of_key_record_id: str | None = None
    rotation_source_key_record_ids: tuple[str, ...] = ()
    revoked_at: str | None = None
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    key_epoch: int = 1
    active_from_server_sequence: int | None = None
    superseded_at: str | None = None
    wrapped_for: SyncKeyWrappedFor = "recovery"
    rewrap_status: SyncKeyRewrapStatus = "not_required"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rotation_source_key_record_ids",
            _normalize_key_record_source_ids(self.rotation_source_key_record_ids),
        )
        _validate_key_record_rotation_metadata(
            encryption_policy=self.encryption_policy,
            key_epoch=self.key_epoch,
            active_from_server_sequence=self.active_from_server_sequence,
            wrapped_for=self.wrapped_for,
            rewrap_status=self.rewrap_status,
        )


@dataclass(frozen=True, slots=True)
class SyncKeyRecord:
    """Stored encrypted key material metadata."""

    key_record_id: str
    dataset_id: str
    user_id: str
    key_purpose: str
    wrapped_key_blob: str
    device_id: str | None
    kdf_metadata: dict[str, Any]
    recovery_hint: str | None
    rotation_of_key_record_id: str | None
    created_at: str
    rotation_source_key_record_ids: tuple[str, ...] = ()
    revoked_at: str | None = None
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    key_epoch: int = 1
    active_from_server_sequence: int | None = None
    superseded_at: str | None = None
    wrapped_for: SyncKeyWrappedFor = "recovery"
    rewrap_status: SyncKeyRewrapStatus = "not_required"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rotation_source_key_record_ids",
            _normalize_key_record_source_ids(self.rotation_source_key_record_ids),
        )
        _validate_key_record_rotation_metadata(
            encryption_policy=self.encryption_policy,
            key_epoch=self.key_epoch,
            active_from_server_sequence=self.active_from_server_sequence,
            wrapped_for=self.wrapped_for,
            rewrap_status=self.rewrap_status,
        )


def _normalize_key_record_source_ids(source_ids: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(source_id).strip()
                for source_id in source_ids or ()
                if str(source_id).strip()
            }
        )
    )


def _validate_key_record_rotation_metadata(
    *,
    encryption_policy: EncryptionPolicy,
    key_epoch: int,
    active_from_server_sequence: int | None,
    wrapped_for: SyncKeyWrappedFor,
    rewrap_status: SyncKeyRewrapStatus,
) -> None:
    if encryption_policy not in SYNC_V2_ENCRYPTION_POLICIES:
        raise ValueError(f"unsupported Sync v2 encryption policy: {encryption_policy}")
    if isinstance(key_epoch, bool) or key_epoch < 1:
        raise ValueError("Sync key record key_epoch must be greater than or equal to 1")
    if (
        active_from_server_sequence is not None
        and (
            isinstance(active_from_server_sequence, bool)
            or active_from_server_sequence < 0
        )
    ):
        raise ValueError("Sync key record active_from_server_sequence must be non-negative")
    if wrapped_for not in SYNC_KEY_WRAPPED_FOR_VALUES:
        raise ValueError(f"unsupported Sync key wrapped_for value: {wrapped_for}")
    if rewrap_status not in SYNC_KEY_REWRAP_STATUSES:
        raise ValueError(f"unsupported Sync key rewrap_status value: {rewrap_status}")


@dataclass(frozen=True, slots=True)
class SyncKeyRotationKeyRecord:
    """Redacted key-record metadata returned by key rotation flows."""

    key_record_id: str
    key_epoch: int
    encryption_policy: EncryptionPolicy
    wrapped_for: SyncKeyWrappedFor
    rewrap_status: SyncKeyRewrapStatus
    device_id: str | None = None
    key_purpose: str = "dataset_recovery"
    active_from_server_sequence: int | None = None
    superseded_at: str | None = None
    revoked_at: str | None = None
    rotation_of_key_record_id: str | None = None


@dataclass(frozen=True, slots=True)
class SyncKeyRotationEnvelopeRange:
    """Accepted envelope range retained under old key material."""

    from_server_sequence: int | None = None
    through_server_sequence: int | None = None
    envelope_count: int = 0


@dataclass(frozen=True, slots=True)
class SyncKeyRotationResult:
    """Redacted key rotation preview or commit result."""

    dataset_id: str
    target_encryption_policy: EncryptionPolicy
    next_key_epoch: int
    active_from_server_sequence: int
    can_commit: bool
    committed: bool
    retained_envelope_range: SyncKeyRotationEnvelopeRange
    affected_key_records: list[SyncKeyRotationKeyRecord] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    device_ids: list[str] = field(default_factory=list)
    recovery_target_count: int = 0
    rotation_id: str | None = None
    new_key_record: SyncKeyRotationKeyRecord | None = None


@dataclass(frozen=True, slots=True)
class SyncAttachmentCreate:
    """Attachment payload accepted by the Sync v2 store."""

    attachment_id: str
    dataset_id: str
    domain: SyncDomain
    content_type: str
    size_bytes: int
    payload_ciphertext: str
    payload_hash: str
    object_id: str | None = None
    entity_id: str | None = None
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object_id = _coalesce_identity(self.object_id, self.entity_id, field_name="object_id")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "entity_id", object_id)


@dataclass(frozen=True, slots=True)
class SyncAttachment:
    """Stored attachment payload metadata."""

    attachment_id: str
    dataset_id: str
    domain: SyncDomain
    object_id: str | None
    content_type: str
    size_bytes: int
    payload_ciphertext: str
    payload_hash: str
    encryption_policy: EncryptionPolicy
    metadata: dict[str, Any]
    created_at: str
    entity_id: str | None = None
    stored: bool = True

    def __post_init__(self) -> None:
        object_id = _coalesce_identity(self.object_id, self.entity_id, field_name="object_id")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "entity_id", object_id)


def _validate_attachment_binding_identity(
    *,
    attachment_id: str,
    attachment_revision: int,
    blob_hash: str,
    size_bytes: int,
    establishing_server_cursor: int,
    availability_at_acceptance: SyncAttachmentBindingAvailability,
) -> None:
    """Validate immutable attachment-revision binding fields at the store boundary."""

    try:
        parsed_attachment_id = UUID(attachment_id)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("attachment binding attachment_id must be canonical UUIDv4") from exc
    if parsed_attachment_id.version != 4 or str(parsed_attachment_id) != attachment_id:
        raise ValueError("attachment binding attachment_id must be canonical UUIDv4")
    if isinstance(attachment_revision, bool) or attachment_revision < 1:
        raise ValueError("attachment binding revision must be positive")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", blob_hash) is None:
        raise ValueError("attachment binding blob_hash must be lowercase SHA-256")
    if isinstance(size_bytes, bool) or size_bytes < 1:
        raise ValueError("attachment binding size_bytes must be positive")
    if isinstance(establishing_server_cursor, bool) or establishing_server_cursor < 1:
        raise ValueError("attachment binding establishing cursor must be positive")
    if availability_at_acceptance not in {"available", "metadata_only"}:
        raise ValueError("attachment binding acceptance availability is invalid")


@dataclass(frozen=True, slots=True)
class SyncAttachmentRevisionBindingCreate:
    """Immutable attachment revision binding accepted by the Sync v2 store."""

    dataset_id: str
    attachment_id: str
    attachment_revision: int
    blob_hash: str
    size_bytes: int
    establishing_server_cursor: int
    availability_at_acceptance: SyncAttachmentBindingAvailability
    resolved_blob_id: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("attachment binding dataset_id must be non-empty")
        _validate_attachment_binding_identity(
            attachment_id=self.attachment_id,
            attachment_revision=self.attachment_revision,
            blob_hash=self.blob_hash,
            size_bytes=self.size_bytes,
            establishing_server_cursor=self.establishing_server_cursor,
            availability_at_acceptance=self.availability_at_acceptance,
        )
        if self.resolved_blob_id is not None and not self.resolved_blob_id.strip():
            raise ValueError("attachment binding resolved_blob_id must be non-empty")
        if (
            self.availability_at_acceptance == "available"
            and self.resolved_blob_id is None
        ):
            raise ValueError(
                "attachment binding available acceptance requires resolved_blob_id"
            )
        if (
            self.availability_at_acceptance == "metadata_only"
            and self.resolved_blob_id is not None
        ):
            raise ValueError(
                "attachment binding metadata_only acceptance forbids resolved_blob_id"
            )


@dataclass(frozen=True, slots=True)
class SyncAttachmentRevisionBinding:
    """Stored immutable revision identity plus monotonic blob lifecycle pointers."""

    dataset_id: str
    attachment_id: str
    attachment_revision: int
    blob_hash: str
    size_bytes: int
    establishing_server_cursor: int
    availability_at_acceptance: SyncAttachmentBindingAvailability
    resolved_blob_id: str | None
    retention_released_at: str | None
    created_at: str

    def __post_init__(self) -> None:
        _validate_attachment_binding_identity(
            attachment_id=self.attachment_id,
            attachment_revision=self.attachment_revision,
            blob_hash=self.blob_hash,
            size_bytes=self.size_bytes,
            establishing_server_cursor=self.establishing_server_cursor,
            availability_at_acceptance=self.availability_at_acceptance,
        )


@dataclass(frozen=True, slots=True)
class SyncNotesAttachmentSourceMap:
    """Stable bootstrap attachment identity for one hashed legacy source key."""

    dataset_id: str
    bootstrap_id: str
    source_key_hash: str
    note_id: str
    attachment_id: str
    created_at: str

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.bootstrap_id.strip():
            raise ValueError("attachment source map identity must be non-empty")
        if not self.note_id.strip():
            raise ValueError("attachment source map note_id must be non-empty")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.source_key_hash) is None:
            raise ValueError("attachment source map hash must be lowercase SHA-256")
        try:
            parsed_attachment_id = UUID(self.attachment_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("attachment source map ID must be canonical UUIDv4") from exc
        if parsed_attachment_id.version != 4 or str(parsed_attachment_id) != self.attachment_id:
            raise ValueError("attachment source map ID must be canonical UUIDv4")


@dataclass(frozen=True, slots=True)
class SyncNotesAttachmentCleanupCandidate:
    """Non-authoritative legacy source evidence retained after canonical import."""

    dataset_id: str
    bootstrap_id: str
    source_key_hash: str
    attachment_id: str
    source_relative_path: str = field(repr=False)
    source_path_hash: str
    source_blob_hash: str
    source_size_bytes: int
    source_modified_ns: int
    created_at: str

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.bootstrap_id.strip():
            raise ValueError("attachment cleanup identity must be non-empty")
        if not self.source_relative_path.strip():
            raise ValueError("attachment cleanup source path must be non-empty")
        for value in (self.source_key_hash, self.source_path_hash, self.source_blob_hash):
            if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ValueError("attachment cleanup hash must be lowercase SHA-256")
        try:
            parsed_attachment_id = UUID(self.attachment_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("attachment cleanup ID must be canonical UUIDv4") from exc
        if parsed_attachment_id.version != 4 or str(parsed_attachment_id) != self.attachment_id:
            raise ValueError("attachment cleanup ID must be canonical UUIDv4")
        if isinstance(self.source_size_bytes, bool) or self.source_size_bytes < 1:
            raise ValueError("attachment cleanup size must be positive")
        if isinstance(self.source_modified_ns, bool) or self.source_modified_ns < 0:
            raise ValueError("attachment cleanup modified time is invalid")


@dataclass(frozen=True, slots=True)
class SyncDatasetStorageNamespace:
    """Server-issued opaque physical storage namespace for one dataset."""

    dataset_id: str
    owner_user_id: str
    storage_namespace_id: str
    created_at: str

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.owner_user_id.strip():
            raise ValueError("storage namespace owner and dataset must be non-empty")
        if re.fullmatch(r"[0-9a-f]{32}", self.storage_namespace_id) is None:
            raise ValueError("storage namespace ID must be 32 lowercase hexadecimal characters")


@dataclass(frozen=True, slots=True)
class SyncBlobUploadSession:
    """Core metadata for a resumable Sync v2 M2 blob upload session."""

    upload_id: str
    dataset_id: str
    owner_user_id: str
    attachment_id: str
    domain: SyncDomain
    object_id: str
    status: SyncBlobUploadStatus
    chunk_size: int
    chunk_count: int
    size_bytes: int
    payload_hash: str
    content_type: str
    device_id: str | None = None
    uploaded_chunks: list[int] = field(default_factory=list)
    missing_chunks: list[int] = field(default_factory=list)
    quota: dict[str, Any] = field(default_factory=dict)
    expires_at: str | None = None
    blob_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncBlobUploadSessionCreate:
    """Upload-session metadata accepted by the Sync v2 M2 store."""

    upload_id: str
    dataset_id: str
    owner_user_id: str
    device_id: str | None
    attachment_id: str
    domain: SyncDomain
    object_id: str
    content_type: str
    size_bytes: int
    payload_hash: str
    chunk_size: int
    chunk_count: int
    reserved_quota_bytes: int
    status: SyncBlobUploadStatus = "created"
    idempotency_key: str | None = None
    expires_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncBlobChunkCreate:
    """Chunk metadata accepted by the Sync v2 M2 store."""

    upload_id: str
    dataset_id: str
    chunk_index: int
    offset_bytes: int
    size_bytes: int
    chunk_hash: str
    storage_key: str


@dataclass(frozen=True, slots=True)
class SyncBlobChunk:
    """Stored chunk metadata for one upload session."""

    upload_id: str
    dataset_id: str
    chunk_index: int
    offset_bytes: int
    size_bytes: int
    chunk_hash: str
    storage_key: str
    received_at: str


@dataclass(frozen=True, slots=True)
class SyncBlobObjectCreate:
    """Committed blob metadata accepted by the Sync v2 M2 store."""

    blob_id: str
    dataset_id: str
    owner_user_id: str
    attachment_id: str
    payload_hash: str
    content_type: str
    size_bytes: int
    storage_backend: str
    storage_key: str
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    status: SyncBlobAvailabilityStatus = "available"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncBlobObject:
    """Committed blob metadata stored by Sync v2 M2."""

    blob_id: str
    dataset_id: str
    owner_user_id: str
    attachment_id: str
    payload_hash: str
    content_type: str
    size_bytes: int
    encryption_policy: EncryptionPolicy
    storage_backend: str
    storage_key: str
    status: SyncBlobAvailabilityStatus
    ref_count: int
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    deleted_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncBlobQuotaUsage:
    """Quota counters for committed and pending Sync v2 M2 blobs."""

    owner_user_id: str
    dataset_id: str | None = None
    reserved_blob_bytes: int = 0
    used_blob_bytes: int = 0
    active_upload_count: int = 0


@dataclass(frozen=True, slots=True)
class SyncBlobDownloadChunk:
    """Core chunk entry used by a resumable Sync v2 M2 blob download manifest."""

    chunk_index: int
    offset_bytes: int
    size_bytes: int
    chunk_hash: str
    download_url: str | None = None


@dataclass(frozen=True, slots=True)
class SyncBlobDownloadManifest:
    """Core manifest describing resumable Sync v2 M2 blob download availability."""

    dataset_id: str
    attachment_id: str
    availability: SyncBlobAvailabilityStatus
    content_type: str
    size_bytes: int
    payload_hash: str
    chunks: list[SyncBlobDownloadChunk] = field(default_factory=list)
    blob_id: str | None = None
    expires_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncRestoreDomainCompleteness:
    """Per-domain restore completeness counters for Sync v2 M2."""

    domain: SyncDomain
    status: SyncRestoreCompletenessStatus
    selected_count: int = 0
    safe_apply_count: int = 0
    conflict_count: int = 0
    tombstone_count: int = 0
    required_blob_count: int = 0
    available_blob_count: int = 0
    missing_blob_count: int = 0
    verified_blob_count: int = 0


@dataclass(frozen=True, slots=True)
class SyncRestoreBlobCompleteness:
    """Per-blob restore completeness detail for Sync v2 M2."""

    attachment_id: str
    payload_hash: str
    size_bytes: int
    content_type: str
    parent_domain: SyncDomain
    parent_object_id: str
    server_availability: SyncBlobAvailabilityStatus
    download_status: str | None = None
    required_for_restore: bool = True


@dataclass(frozen=True, slots=True)
class SyncRestoreManifestStats:
    """Database-side aggregate statistics for one restore-manifest dataset."""

    approximate_counts: dict[str, int] = field(default_factory=dict)
    byte_estimates: dict[str, int] = field(default_factory=dict)
    last_updated_at: str | None = None
    unresolved_conflicts: int = 0
    attachment_availability: dict[str, int] = field(default_factory=dict)
    attachment_size_classes: dict[str, int] = field(default_factory=dict)
    key_recovery_available: bool = False


__all__ = [
    "CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE",
    "CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_MESSAGE",
    "ConflictStatus",
    "DEFAULT_M1_ENCRYPTION_POLICY",
    "DatasetScopeType",
    "EncryptionPolicy",
    "M1_SYNC_DOMAINS",
    "M1_SYNC_OPERATIONS",
    "NOTES_ORGANIZATION_DOMAINS",
    "NOTES_ORGANIZATION_SYNC_OPERATIONS",
    "NOTES_LINK_DOMAINS",
    "NOTES_LINK_SYNC_OPERATIONS",
    "NOTES_TASK_SYNC_DOMAINS",
    "NOTES_TASK_SYNC_OPERATIONS",
    "NOTES_MOODBOARD_STUDIO_DOMAINS",
    "NOTES_MOODBOARD_STUDIO_OPERATIONS",
    "NOTES_NOTE_CANONICAL_PAYLOAD_FIELDS",
    "NOTES_NOTE_CONTENT_MAX_CHARS",
    "NOTES_NOTE_TITLE_MAX_CHARS",
    "MEDIA_SYNC_DOMAINS",
    "MEDIA_SYNC_OPERATIONS",
    "STRICT_ENCRYPTION_POLICIES",
    "SOURCE_CACHE_SYNC_DOMAINS",
    "SOURCE_CACHE_SYNC_OPERATIONS",
    "SYNC_KEY_REWRAP_STATUSES",
    "SYNC_KEY_WRAPPED_FOR_VALUES",
    "SYNC_V2_ENCRYPTION_POLICIES",
    "SYNC_V2_KNOWN_DOMAINS",
    "SYNC_V2_SUPPORTED_DOMAINS",
    "SYNC_V2_SUPPORTED_OPERATIONS",
    "SyncApplyStatus",
    "SyncAttachmentBindingAvailability",
    "SyncAttachment",
    "SyncAttachmentCreate",
    "SyncAttachmentRevisionBinding",
    "SyncAttachmentRevisionBindingCreate",
    "SyncNotesAttachmentCleanupCandidate",
    "SyncNotesAttachmentSourceMap",
    "SyncBackgroundDomainStatus",
    "SyncBackgroundLease",
    "SyncBackgroundLeaseCreate",
    "SyncBackgroundLeaseStatus",
    "SyncBackgroundPolicy",
    "SyncBackgroundPolicyUpsert",
    "SyncBlobAvailabilityStatus",
    "SyncBlobChunk",
    "SyncBlobChunkCreate",
    "SyncBlobDownloadChunk",
    "SyncBlobDownloadManifest",
    "SyncBlobObject",
    "SyncBlobObjectCreate",
    "SyncBlobQuotaUsage",
    "SyncBlobUploadSession",
    "SyncBlobUploadSessionCreate",
    "SyncBlobUploadStatus",
    "SyncConflict",
    "SyncConflictCreate",
    "SyncDataset",
    "SyncDatasetCreate",
    "SyncDatasetStorageNamespace",
    "SyncDevice",
    "SyncDeviceCursor",
    "SyncDeviceUpsert",
    "SyncDomain",
    "SyncDomainEnvelopeSummary",
    "SyncEnvelope",
    "SyncEnvelopeCreate",
    "SyncEncryptionPolicyMetadata",
    "SyncKeyRewrapStatus",
    "SyncKeyRecord",
    "SyncKeyRecordCreate",
    "SyncKeyRotationEnvelopeRange",
    "SyncKeyRotationKeyRecord",
    "SyncKeyRotationResult",
    "SyncKeyWrappedFor",
    "SyncObjectState",
    "SyncOperation",
    "SyncRestoreBlobCompleteness",
    "SyncRestoreCompletenessStatus",
    "SyncRestoreDomainCompleteness",
    "SyncRestoreManifestStats",
    "WORKSPACE_SYNC_DOMAINS",
    "WORKSPACE_SYNC_OPERATIONS",
    "client_private_server_frontend_limitation_warning",
    "server_frontend_mutation_blockers_for_policy",
    "server_frontend_mutation_enabled_for_policy",
    "sync_v2_advertised_domain_schemas",
    "sync_v2_domain_schemas",
    "validate_notes_note_upsert_payload",
]
