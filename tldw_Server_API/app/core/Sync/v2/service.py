from __future__ import annotations

"""Business service for Sync v2 protocol operations."""

import base64
import binascii
import hashlib
import hmac
import inspect
import json
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Literal
from uuid import RFC_4122, UUID, uuid4

from loguru import logger
from tldw_profile_core import SERIALIZED_SCHEMA_VERSION

from tldw_Server_API.app.core.Notes.attachment_policy import (
    NoteAttachmentPolicyError,
    canonicalize_note_attachment_file_name,
    validate_note_attachment_content_type,
    validate_note_attachment_original_file_name,
)

from .adapters import (
    ATTACHMENT_REF_SERVER_AVAILABILITY,
    AdapterAccepted,
    AdapterConflict,
    AdapterDeferred,
    AdapterRejected,
    AttachmentRefValidationError,
    SyncAdapterContext,
    SyncAdapterRegistry,
    SyncDomainAdapter,
    extract_attachment_ref_metadata,
)
from .attachment_refs_v2 import parse_attachment_ref_v2_payload
from .blob_store import LocalSyncBlobStore, SyncBlobStoreError
from .errors import (
    PersonalContextStorageEncryptionUnavailableError,
    SyncHeadConflictError,
    SyncIdempotencyConflictError,
    SyncInvalidDomainError,
    SyncMaterializationBusyError,
    SyncMaterializationPredecessorError,
    SyncStoreError,
)
from .materializers import MaterializationResult, SyncMaterializer
from .materializers.guarded_product_mutation import (
    GuardedProductMutation,
    has_guard_required_routing_key,
)
from .models import (
    DEFAULT_M1_ENCRYPTION_POLICY,
    M1_SYNC_DOMAINS,
    NOTES_LINK_DOMAINS,
    NOTES_MOODBOARD_STUDIO_DOMAINS,
    NOTES_ORGANIZATION_DOMAINS,
    NOTES_TASK_SYNC_DOMAINS,
    NOTES_TASK_SYNC_OPERATIONS,
    PERSONAL_CONTEXT_SYNC_DOMAINS,
    SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION,
    SYNC_V2_ENCRYPTION_POLICIES,
    SYNC_V2_KNOWN_DOMAINS,
    SYNC_V2_SUPPORTED_DOMAINS,
    SYNC_V2_SUPPORTED_OPERATIONS,
    WORKSPACE_SYNC_DOMAINS,
    ConflictStatus,
    EncryptionPolicy,
    SyncAttachment,
    SyncAttachmentCreate,
    SyncAttachmentRevisionBinding,
    SyncBackgroundDomainStatus,
    SyncBackgroundLease,
    SyncBackgroundLeaseCreate,
    SyncBackgroundPolicy,
    SyncBackgroundPolicyUpsert,
    SyncBlobChunk,
    SyncBlobChunkCreate,
    SyncBlobDownloadChunk,
    SyncBlobDownloadManifest,
    SyncBlobObject,
    SyncBlobObjectCreate,
    SyncBlobUploadSession,
    SyncBlobUploadSessionCreate,
    SyncConflict,
    SyncConflictCreate,
    SyncDataset,
    SyncDatasetCreate,
    SyncDevice,
    SyncDeviceAcknowledgmentSummary,
    SyncDeviceAuthorization,
    SyncDeviceAuthorizationCreate,
    SyncDeviceBlobAckCreate,
    SyncDeviceBlobIdAckCreate,
    SyncDeviceCursor,
    SyncDeviceDomainAckCreate,
    SyncDeviceUpsert,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncKeyRecord,
    SyncKeyRecordCreate,
    SyncKeyRewrapStatus,
    SyncKeyRotationKeyRecord,
    SyncKeyRotationResult,
    SyncKeyWrappedFor,
    SyncOperation,
    SyncRestoreBlobCompleteness,
    SyncRestoreCompletenessStatus,
    SyncRestoreDomainCompleteness,
    _sync_v2_internal_domain_schemas,
    client_private_server_frontend_limitation_warning,
    normalize_supported_adapter_versions,
    normalize_sync_timestamp,
    normalize_sync_v2_requested_domains,
    sync_v2_advertised_domain_schemas,
    sync_v2_attachment_ref_v2_is_writable,
    sync_v2_dataset_writable_adapter_versions,
    sync_v2_domain_schemas,
    sync_v2_server_supported_adapter_versions,
)
from .mutation_group_validation import (
    StoredMutationGroupValidationError,
    mutation_group_plan_hash,
    validate_stored_mutation_group,
)
from .notes_moodboard_studio_readiness import (
    NOTES_MOODBOARD_STUDIO_SERVER_METADATA_KEYS,
    redact_notes_moodboard_studio_server_metadata,
)
from .notes_task_contract import (
    NotesTaskV1Payload,
    notes_task_activity_object_hash,
    parse_notes_task_activity_v1,
    parse_notes_task_v1,
)
from .notes_task_readiness import (
    NOTES_TASK_SERVER_METADATA_KEYS,
    notes_task_sync_is_ready,
    redact_notes_task_server_metadata,
)
from .profile import (
    PersonalContextBootstrap,
    SyncNotesAttachmentBootstrapDiagnostics,
    SyncProfileStatus,
    SyncRecoveryActionDescriptor,
    SyncV2ProfileManager,
)
from .replay import SyncReplayRepairer, SyncReplayRepairResult
from .restore import (
    OBJECT_RESTORE_DOMAINS,
    WHOLE_OBJECT_RESTORE_DOMAINS,
    LocalRestoreInventoryItem,
    RestorePlanningError,
    attachment_available_locally,
    attachment_restore_status,
    attachment_verified_locally,
    build_local_inventory_index,
    find_local_inventory_item,
    local_inventory_matches,
    order_restore_envelopes,
    restore_action_for_domain,
)
from .security import (
    PrivatePayloadValidationError,
    SyncV2ServerTrustedEncryptionStatus,
    server_trusted_encryption_status_from_env,
    validate_private_payload,
)
from .store import SyncV2Store

SYNC_PULL_TOKEN_MAX_ENCODED_BYTES = 32_768
SYNC_PULL_TOKEN_MAX_DECODED_BYTES = 24_576
SYNC_PULL_TOKEN_MAX_STREAMS = 800
SYNC_PULL_TOKEN_VERSION = 1
SYNC_PULL_TOKEN_CLOCK_SKEW_SECONDS = 300
SYNC_RETENTION_BINDING_PAGE_SIZE = 1000

SYNC_DATASET_RECOVERY_KEY_PURPOSE = "dataset_recovery"
SYNC_KEY_RECOVERY_MAX_WRAPPED_KEY_BYTES = 64 * 1024
_SERVER_ORIGIN_DEVICE_ID = "server-origin"


def _require_client_device_id(device_id: str) -> None:
    if device_id == _SERVER_ORIGIN_DEVICE_ID:
        raise SyncStoreError("Sync server-origin is a reserved device identifier")


def _safe_projection_error_message(exc: Exception) -> str:
    return f"Projection failed: {type(exc).__name__}"


def _notes_task_activity_metadata(payload: NotesTaskV1Payload) -> dict[str, object]:
    """Return the canonical task metadata snapshot used by activity events."""

    wire = payload.model_dump(mode="json")
    return {
        key: wire[key]
        for key in (
            "description",
            "priority",
            "due_date",
            "estimate",
            "recurrence",
            "assignee_id",
            "tags",
            "custom",
        )
    }


def _notes_task_projection_status(head: SyncEnvelope | None) -> str:
    """Infer the last durable projection state without consulting product cache."""

    if head is None:
        return "unlinked"
    raw_anchor = head.routing_metadata.get("task_projection")
    if isinstance(raw_anchor, Mapping) and raw_anchor.get("linked") is True:
        return "live"
    return "unlinked"


def _client_task_activity_values(
    *,
    before: NotesTaskV1Payload | None,
    after: NotesTaskV1Payload,
    operation: SyncOperation,
    prior_head: SyncEnvelope | None,
    restore_intent: bool,
) -> tuple[str, dict[str, object] | None, dict[str, object]]:
    """Derive the sole canonical lifecycle event for an authenticated task push."""

    after_metadata = _notes_task_activity_metadata(after)
    if before is None:
        return (
            "created",
            None,
            {
                "title": after.title,
                "status": after.status,
                "completed_at": after.completed_at,
                "metadata": after_metadata,
            },
        )
    if operation == "tombstone":
        return (
            "deleted",
            {
                "deleted": False,
                "projection_status": _notes_task_projection_status(prior_head),
            },
            {"deleted": True, "projection_status": "deleted"},
        )
    if restore_intent:
        return (
            "restored",
            {"deleted": True, "projection_status": "deleted"},
            {
                "deleted": False,
                "projection_status": _notes_task_projection_status(prior_head),
            },
        )
    if (before.status, after.status) == ("open", "done"):
        return "completed", {"status": "open"}, {"status": "done"}
    if (before.status, after.status) == ("done", "open"):
        return "reopened", {"status": "done"}, {"status": "open"}
    before_metadata = _notes_task_activity_metadata(before)
    if before.title != after.title:
        return (
            "updated",
            {"title": before.title, "metadata": before_metadata},
            {"title": after.title, "metadata": after_metadata},
        )
    if before_metadata != after_metadata:
        return "updated", {"metadata": before_metadata}, {"metadata": after_metadata}
    raise SyncStoreError("notes_task_transition_has_no_activity")


def _notes_task_client_group_id(
    dataset_id: str,
    device_id: str,
    client_envelope_id: str,
) -> str:
    """Return the stable group identity controlled by authenticated input."""

    digest = hashlib.sha256(
        f"{dataset_id}:{device_id}:{client_envelope_id}".encode()
    ).hexdigest()
    return f"notes-task-client-group-{digest[:32]}"


def _same_client_task_submission(
    stored: SyncEnvelope,
    incoming: SyncEnvelopeCreate,
) -> bool:
    """Compare only client-controlled task envelope fields for exact replay."""

    fields = (
        "dataset_id",
        "client_envelope_id",
        "domain",
        "operation",
        "object_id",
        "device_id",
        "client_profile_id",
        "client_sequence",
        "base_server_cursor",
        "base_object_revision",
        "base_object_hash",
        "object_revision",
        "parent_id",
        "schema_version",
        "payload",
        "payload_clear",
        "payload_ciphertext",
        "payload_hash",
        "payload_size_bytes",
        "created_at_client",
        "deleted",
        "encryption_metadata",
        "stable_key",
        "dependencies",
        "adapter_version",
        "base_version",
        "entity_version",
    )
    stored_routing = dict(stored.routing_metadata)
    stored_routing.pop("task_projection", None)
    return stored_routing == dict(incoming.routing_metadata) and all(
        getattr(stored, field) == getattr(incoming, field) for field in fields
    )


def _key_recovery_metadata_string(
    metadata: Mapping[str, object],
    *keys: str,
    nested_parent: str | None = None,
) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = metadata.get(nested_parent) if nested_parent else None
    if isinstance(nested, Mapping):
        for key in keys:
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _redact_private_sync_server_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Remove private server readiness metadata from public dataset views."""
    return redact_notes_moodboard_studio_server_metadata(
        redact_notes_task_server_metadata(metadata)
    )


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _decode_pull_token_segment(segment: str) -> bytes:
    """Decode one unpadded URL-safe base64 pull-token segment."""

    padding = "=" * (-len(segment) % 4)
    return base64.b64decode(
        (segment + padding).encode("ascii"),
        altchars=b"-_",
        validate=True,
    )


def _device_supports_adapter_version(
    device: SyncDevice,
    domain: SyncDomain,
    adapter_version: int,
) -> bool:
    """Return whether the stored device negotiation permits one envelope."""

    version_map = device.capabilities.get("supported_adapter_versions")
    if version_map is None:
        return adapter_version == 1
    if not isinstance(version_map, Mapping):
        return False
    versions = version_map.get(domain)
    if not isinstance(versions, Sequence) or isinstance(
        versions,
        (str, bytes, bytearray),
    ):
        return False
    return any(
        not isinstance(version, bool)
        and isinstance(version, int)
        and version == adapter_version
        for version in versions
    )


def _normalize_device_capability_patch(
    existing: Mapping[str, object],
    merged: dict[str, object],
    *,
    patch: Mapping[str, object],
    existing_registration: bool = False,
) -> dict[str, object]:
    """Protect negotiated adapter versions while merging ordinary capabilities."""

    existing_versions = existing.get("supported_adapter_versions")
    patch_versions = patch.get("supported_adapter_versions")
    patch_requested = patch.get("requested_domains")
    has_version_patch = "supported_adapter_versions" in patch
    has_requested_patch = "requested_domains" in patch
    if not has_version_patch and not has_requested_patch:
        return merged
    if has_version_patch and patch_versions is None:
        raise SyncStoreError("Sync device adapter version capabilities are invalid")

    existing_requested_raw = existing.get("requested_domains")
    existing_requested: list[SyncDomain] = []
    if existing_requested_raw is not None:
        try:
            existing_requested = normalize_sync_v2_requested_domains(
                existing_requested_raw
            )
        except ValueError as exc:
            raise SyncStoreError(
                "Sync device adapter version capabilities are invalid"
            ) from exc
    if isinstance(existing_versions, Mapping):
        existing_requested.extend(
            key for key in existing_versions if isinstance(key, str)
        )
    elif existing_requested_raw is None and existing_registration:
        existing_requested = list(M1_SYNC_DOMAINS)
        existing_supported = existing.get("supported_domains")
        if isinstance(existing_supported, list):
            supported = {
                item for item in existing_supported if isinstance(item, str)
            }
            existing_requested = [
                domain for domain in existing_requested if domain in supported
            ]
    requested: list[SyncDomain] = list(existing_requested)
    if patch_requested is not None:
        try:
            requested.extend(normalize_sync_v2_requested_domains(patch_requested))
        except ValueError as exc:
            raise SyncStoreError(
                "Sync device adapter version capabilities are invalid"
            ) from exc
    if isinstance(patch_versions, Mapping):
        requested.extend(key for key in patch_versions if isinstance(key, str))

    try:
        requested = normalize_sync_v2_requested_domains(
            list(dict.fromkeys(requested))
        )
        prior = normalize_supported_adapter_versions(
            existing_versions,
            requested_domains=existing_requested,
        )
        normalized_patch = normalize_supported_adapter_versions(
            patch_versions,
            requested_domains=requested,
        )
    except ValueError as exc:
        raise SyncStoreError(
            "Sync device adapter version capabilities are invalid"
        ) from exc

    candidate = dict(prior)
    for domain in requested:
        candidate.setdefault(domain, [1])
    if isinstance(patch_versions, Mapping):
        for domain in patch_versions:
            candidate[domain] = normalized_patch[domain]
    for domain, versions in prior.items():
        if not set(versions).issubset(candidate.get(domain, [])):
            raise SyncStoreError(
                "Sync device adapter version capabilities cannot remove active versions"
            )

    merged["requested_domains"] = requested
    merged["supported_adapter_versions"] = candidate
    return merged


def _validated_device_capabilities(
    existing: SyncDevice | None,
    incoming: Mapping[str, object],
    *,
    merge_capabilities: bool,
    negotiation_patch: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Resolve and validate one capability write against the current device."""

    existing_capabilities = existing.capabilities if existing is not None else {}
    updated = (
        dict(existing_capabilities)
        if merge_capabilities and existing is not None
        else {}
    )
    updated.update(incoming)
    patch = dict(incoming if negotiation_patch is None else negotiation_patch)
    if existing is not None and existing.status == "active":
        for field_name in ("requested_domains", "supported_adapter_versions"):
            if field_name not in patch and field_name in existing_capabilities:
                patch[field_name] = existing_capabilities[field_name]
    if not patch:
        return updated
    return _normalize_device_capability_patch(
        existing_capabilities,
        updated,
        patch=patch,
        existing_registration=existing is not None,
    )


def _atomic_negotiation_patch(
    current: SyncDevice | None,
    submitted: Mapping[str, object],
    incoming: Mapping[str, object],
) -> dict[str, object]:
    """Union validated negotiation additions with the transaction-current row."""

    patch = dict(incoming)
    negotiation_fields = {"requested_domains", "supported_adapter_versions"}
    if not negotiation_fields.intersection(submitted):
        return patch

    requested: list[SyncDomain] = []
    version_maps: list[dict[SyncDomain, list[int]]] = []
    for capabilities in (
        current.capabilities if current is not None else {},
        submitted,
    ):
        raw_requested = capabilities.get("requested_domains")
        try:
            capability_requested = (
                normalize_sync_v2_requested_domains(raw_requested)
                if raw_requested is not None
                else []
            )
        except ValueError as exc:
            raise SyncStoreError(
                "Sync device adapter version capabilities are invalid"
            ) from exc
        raw_versions = capabilities.get("supported_adapter_versions")
        if isinstance(raw_versions, Mapping):
            capability_requested.extend(
                domain for domain in raw_versions if isinstance(domain, str)
            )
        elif raw_versions is not None:
            raise SyncStoreError(
                "Sync device adapter version capabilities are invalid"
            )
        try:
            capability_requested = normalize_sync_v2_requested_domains(
                list(dict.fromkeys(capability_requested))
            )
            version_maps.append(
                normalize_supported_adapter_versions(
                    raw_versions,
                    requested_domains=capability_requested,
                )
            )
        except ValueError as exc:
            raise SyncStoreError(
                "Sync device adapter version capabilities are invalid"
            ) from exc
        requested.extend(capability_requested)

    try:
        requested = normalize_sync_v2_requested_domains(
            list(dict.fromkeys(requested))
        )
    except ValueError as exc:
        raise SyncStoreError(
            "Sync device adapter version capabilities are invalid"
        ) from exc
    unioned_versions: dict[SyncDomain, list[int]] = {}
    for version_map in version_maps:
        for domain, versions in version_map.items():
            unioned_versions[domain] = sorted(
                set(unioned_versions.get(domain, ())).union(versions)
            )
    patch["requested_domains"] = requested
    patch["supported_adapter_versions"] = unioned_versions
    return patch


@dataclass(frozen=True, slots=True)
class PersonalContextSyncCapabilities:
    """Bounded Personal Context contract advertised through Sync v2."""

    available: bool = False
    blockers: tuple[str, ...] = ("personal_context_profile_key_unavailable",)
    ongoing_sync_version: Literal[0, 1] = 0
    ongoing_sync_blockers: tuple[str, ...] = ()
    activation_epoch: str | None = None
    continuity_token: str | None = None
    authorization_policy: Literal["server_trusted_v1"] = "server_trusted_v1"
    min_schema_version: int = 1
    max_schema_version: int = 1
    integrity_algorithm: Literal["hmac-sha256-v1"] = "hmac-sha256-v1"
    integrity_key_distribution: Literal["wrapped-bootstrap-v1"] = "wrapped-bootstrap-v1"
    privacy_cleanup_ack: Literal["personal-context-cleanup-v1"] = "personal-context-cleanup-v1"
    purge_generation: Literal["personal-context-purge-v1"] = "personal-context-purge-v1"
    max_record_bytes: int = 16_384
    max_search_results: int = 20
    max_proposals_per_turn: int = 5
    max_proposals_per_session: int = 25
    max_unresolved_proposals: int = 200


def personal_context_sync_capabilities_from_env() -> PersonalContextSyncCapabilities:
    """Return fail-closed Personal Context readiness from shared schema and key custody."""

    blockers: list[str] = []
    if SERIALIZED_SCHEMA_VERSION != 1:
        blockers.append("personal_context_schema_unsupported")

    encoded_key = os.getenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", "").strip()
    try:
        master_key = base64.b64decode(encoded_key, validate=True)
    except (binascii.Error, ValueError):
        master_key = b""
    if len(master_key) != 32:
        blockers.append("personal_context_profile_key_unavailable")

    return PersonalContextSyncCapabilities(
        available=not blockers,
        blockers=tuple(blockers),
    )


@dataclass(frozen=True, slots=True)
class SyncV2Settings:
    """Server settings surfaced through Sync v2 capabilities."""

    protocol_version: str = "sync-v2-m1"
    min_supported_protocol_version: str = "sync-v2-m1"
    max_batch_size: int = 100
    max_pull_page_size: int = 100
    max_envelope_payload_bytes: int = 262_144
    max_attachment_bytes: int = 1_048_576
    supports_attachments: bool = False
    max_blob_bytes: int | None = None
    max_chunk_bytes: int = 4_194_304
    max_active_blob_uploads: int = 8
    user_blob_quota_bytes: int | None = None
    reserved_blob_bytes: int = 0
    used_blob_bytes: int = 0
    blob_storage_backend: str = "local_fs"
    blob_checksum_algorithm: str = "sha256"
    supports_resumable_upload: bool = True
    supports_resumable_download: bool = True
    supports_chunk_checksums: bool = True
    supported_domains: list[SyncDomain] = field(default_factory=lambda: list(SYNC_V2_SUPPORTED_DOMAINS))
    operations: dict[SyncDomain, list[SyncOperation]] = field(
        default_factory=lambda: {domain: list(operations) for domain, operations in SYNC_V2_SUPPORTED_OPERATIONS.items()}
    )
    encryption_policies: list[EncryptionPolicy] = field(default_factory=lambda: [DEFAULT_M1_ENCRYPTION_POLICY])
    server_trusted_encryption: SyncV2ServerTrustedEncryptionStatus = field(
        default_factory=server_trusted_encryption_status_from_env
    )
    personal_context: PersonalContextSyncCapabilities = field(
        default_factory=personal_context_sync_capabilities_from_env
    )
    restore_manifest_scan_limit: int = 10_000
    restore_preview_candidate_limit: int = 50_000
    restore_preview_action_limit: int = 10_000
    pull_token_signing_secret: str | None = None
    pull_token_ttl_seconds: int = 3_600


@dataclass(frozen=True, slots=True)
class SyncV2Capabilities:
    protocol_version: str
    min_supported_protocol_version: str
    supported_domains: list[SyncDomain]
    operations: dict[SyncDomain, list[SyncOperation]]
    encryption: dict[str, object]
    blob_transfer: dict[str, object]
    encryption_policies: list[EncryptionPolicy]
    personal_context: PersonalContextSyncCapabilities
    max_batch_size: int
    max_envelope_payload_bytes: int
    max_attachment_bytes: int
    domain_schemas: dict[SyncDomain, dict[str, object]] = field(
        default_factory=sync_v2_domain_schemas
    )
    supported_adapter_versions: dict[SyncDomain, list[int]] = field(
        default_factory=sync_v2_server_supported_adapter_versions
    )
    writable_adapter_versions: dict[SyncDomain, list[int]] = field(
        default_factory=sync_v2_dataset_writable_adapter_versions
    )
    quota: dict[str, object] = field(default_factory=dict)
    supports_restore_manifest: bool = True
    supports_conflicts: bool = True
    supports_attachments: bool = True
    compatibility_flags: dict[str, bool] = field(default_factory=dict)
    server_time: str | None = None
    warnings: list[dict[str, str]] = field(default_factory=list)


WorkspaceAccessChecker = Callable[[str, str, str], bool]


@dataclass(frozen=True, slots=True)
class SyncDeviceRegistration:
    device: SyncDevice
    server_capabilities: SyncV2Capabilities
    required_actions: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SyncDatasetEnrollment:
    dataset: SyncDataset
    cursors: dict[str, str] = field(default_factory=dict)
    key_setup_required: bool = False


@dataclass(frozen=True, slots=True)
class SyncBackgroundStatus:
    dataset_id: str
    device_id: str
    policy: SyncBackgroundPolicy
    lease: SyncBackgroundLease | None = None
    domains: list[SyncBackgroundDomainStatus] = field(default_factory=list)
    conflict_count: int = 0
    replayable_failure_count: int = 0
    quota_pressure: dict[str, object] = field(default_factory=dict)
    restore_completeness: SyncRestoreCompletenessStatus = "metadata_ready"
    server_time: str | None = None


@dataclass(frozen=True, slots=True)
class SyncRetentionCandidate:
    """One read-only retention, compaction, or blob-GC candidate."""

    candidate_type: str
    dataset_id: str
    domain: SyncDomain | None = None
    object_id: str | None = None
    server_sequence: int | None = None
    blob_id: str | None = None
    attachment_id: str | None = None
    attachment_revision: int | None = None
    payload_hash: str | None = None
    size_bytes: int | None = None
    blockers: list[str] = field(default_factory=list)
    required_device_ids: list[str] = field(default_factory=list)
    unacknowledged_device_ids: list[str] = field(default_factory=list)
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SyncRetentionDryRunResult:
    """Read-only retention scan result."""

    dataset_id: str
    dry_run: bool = True
    mutation_performed: bool = False
    evaluated_at: str | None = None
    audit_mode: bool = True
    minimum_envelope_age_seconds: int = 0
    minimum_tombstone_age_seconds: int = 0
    offline_restore_window_seconds: int = 0
    candidate_count: int = 0
    blocked_count: int = 0
    blocker_counts: dict[str, int] = field(default_factory=dict)
    candidates: list[SyncRetentionCandidate] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SyncRetentionApplyResult:
    """Guarded retention compaction/GC apply result."""

    dataset_id: str
    dry_run: bool = True
    mutation_performed: bool = False
    confirmation_required: bool = False
    evaluated_at: str | None = None
    candidate_count: int = 0
    applied_count: int = 0
    blocked_count: int = 0
    skipped_count: int = 0
    blockers: list[str] = field(default_factory=list)
    blocker_counts: dict[str, int] = field(default_factory=dict)
    domain_compactions: list[dict[str, object]] = field(default_factory=list)
    binding_releases: list[dict[str, object]] = field(default_factory=list)
    blob_gc: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SyncDiagnosticsDomain:
    """Redacted diagnostics for one dataset domain."""

    domain: SyncDomain
    envelope_count: int = 0
    object_count: int = 0
    latest_server_sequence: int = 0
    failed_apply_count: int = 0
    unresolved_conflict_count: int = 0


@dataclass(frozen=True, slots=True)
class SyncDiagnosticsDeviceDomainLag:
    """Redacted cursor lag for one device/domain pair."""

    domain: SyncDomain
    last_pulled_sequence: int = 0
    latest_server_sequence: int = 0
    lag_count: int = 0


@dataclass(frozen=True, slots=True)
class SyncDiagnosticsDevice:
    """Redacted diagnostics for one sync device."""

    device_id: str
    status: str
    last_seen_at: str | None = None
    domain_lag: list[SyncDiagnosticsDeviceDomainLag] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SyncDiagnosticsBlobHealth:
    """Redacted blob/upload diagnostics."""

    blob_object_count: int = 0
    available_blob_bytes: int = 0
    active_upload_count: int = 0
    reserved_blob_bytes: int = 0
    quota_limit_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class SyncDiagnosticsKeySummary:
    """Redacted key-record diagnostics without wrapped key material."""

    key_record_count: int = 0
    active_key_record_count: int = 0
    revoked_key_record_count: int = 0
    superseded_key_record_count: int = 0
    rewrap_pending_count: int = 0
    recovery_available: bool = False


@dataclass(frozen=True, slots=True)
class SyncDiagnosticsRetentionSummary:
    """Redacted retention dry-run summary."""

    dry_run: bool = True
    mutation_performed: bool = False
    candidate_count: int = 0
    blocked_count: int = 0
    blocker_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncAttachmentDiagnosticSample:
    """One bounded owner-authorized attachment lifecycle sample."""

    category: str
    code: str
    attachment_id: str | None = None
    blob_id: str | None = None
    server_cursor: int | None = None
    recovery_actions: list[SyncRecoveryActionDescriptor] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SyncAttachmentDiagnostics:
    """Bounded read-only Notes attachment lifecycle diagnostics."""

    counts: dict[str, int] = field(default_factory=dict)
    samples: list[SyncAttachmentDiagnosticSample] = field(default_factory=list)
    recovery_actions: list[SyncRecoveryActionDescriptor] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SyncDiagnosticsReport:
    """Redacted dataset diagnostics report."""

    dataset_id: str
    generated_at: str | None = None
    domains: list[SyncDiagnosticsDomain] = field(default_factory=list)
    devices: list[SyncDiagnosticsDevice] = field(default_factory=list)
    blob_health: SyncDiagnosticsBlobHealth = field(default_factory=SyncDiagnosticsBlobHealth)
    key_summary: SyncDiagnosticsKeySummary = field(default_factory=SyncDiagnosticsKeySummary)
    retention: SyncDiagnosticsRetentionSummary = field(default_factory=SyncDiagnosticsRetentionSummary)
    attachment_lifecycle: SyncAttachmentDiagnostics = field(
        default_factory=SyncAttachmentDiagnostics
    )


@dataclass(frozen=True, slots=True)
class SyncPushAccepted:
    client_envelope_id: str
    server_sequence: int
    domain: SyncDomain
    entity_id: str
    object_revision: int | None = None
    apply_status: str | None = None
    apply_error_code: str | None = None
    apply_error_message: str | None = None


@dataclass(frozen=True, slots=True)
class SyncPushRejected:
    client_envelope_id: str
    error_code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class SyncPushConflict:
    conflict_id: str
    client_envelope_id: str
    domain: SyncDomain
    entity_id: str
    server_sequence: int | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class SyncPushResult:
    dataset_id: str
    accepted: list[SyncPushAccepted] = field(default_factory=list)
    rejected: list[SyncPushRejected] = field(default_factory=list)
    conflicts: list[SyncPushConflict] = field(default_factory=list)
    next_cursor: str | None = None


@dataclass(frozen=True, slots=True)
class SyncPullResult:
    dataset_id: str
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    envelopes: list[SyncEnvelope] = field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


@dataclass(frozen=True, slots=True)
class SyncRestoreManifestDevice:
    device_id: str
    display_name: str | None
    client_type: str | None
    client_version: str | None
    last_seen_at: str | None
    revoked_at: str | None


@dataclass(frozen=True, slots=True)
class SyncRestoreManifestDataset:
    dataset_id: str
    scope_type: str
    encryption_policy: EncryptionPolicy
    domains: list[SyncDomain]
    workspace_id: str | None
    approximate_counts: dict[str, int] = field(default_factory=dict)
    byte_estimates: dict[str, int] = field(default_factory=dict)
    last_updated_at: str | None = None
    unresolved_conflicts: int = 0
    attachment_availability: dict[str, int] = field(default_factory=dict)
    attachment_size_classes: dict[str, int] = field(default_factory=dict)
    key_recovery_available: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncRestoreManifest:
    datasets: list[SyncRestoreManifestDataset] = field(default_factory=list)
    devices: list[SyncRestoreManifestDevice] = field(default_factory=list)
    generated_at: str | None = None
    filters_applied: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncRestorePreviewDataset:
    dataset_id: str
    domains: list[SyncDomain]
    approximate_counts: dict[str, int] = field(default_factory=dict)
    byte_estimates: dict[str, int] = field(default_factory=dict)
    latest_cursor: int | None = None
    latest_cursors: dict[str, int] = field(default_factory=dict)
    envelope_ranges: list[SyncRestorePreviewEnvelopeRange] = field(default_factory=list)
    total_count: int = 0
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    key_recovery_available: bool = False


@dataclass(frozen=True, slots=True)
class SyncRestorePreviewEnvelopeRange:
    dataset_id: str
    domain: SyncDomain
    from_cursor: int
    to_cursor: int
    envelope_count: int


@dataclass(frozen=True, slots=True)
class SyncRestorePreviewObject:
    dataset_id: str
    domain: SyncDomain
    object_id: str
    action: str
    server_revision: int | None = None
    server_hash: str | None = None
    server_cursor: int | None = None
    local_revision: int | None = None
    local_hash: str | None = None
    local_deleted: bool | None = None
    deleted: bool = False
    parent_id: str | None = None


@dataclass(frozen=True, slots=True)
class SyncRestorePreviewObjectConflict:
    dataset_id: str
    domain: SyncDomain
    object_id: str
    conflict_type: str
    server_revision: int | None = None
    server_hash: str | None = None
    server_cursor: int | None = None
    server_deleted: bool = False
    local_revision: int | None = None
    local_hash: str | None = None
    local_deleted: bool = False
    message: str | None = None


@dataclass(frozen=True, slots=True)
class SyncRestoreOrderedAction:
    """One safe, executable step in the canonical restore-preview plan."""

    plan_index: int
    action: str
    dataset_id: str
    domain: SyncDomain
    object_id: str
    operation: SyncOperation
    server_cursor: int
    adapter_version: int
    mutation_group_id: str | None = None
    mutation_step: int | None = None
    mutation_step_count: int | None = None
    code: str | None = None


@dataclass(frozen=True, slots=True)
class SyncRestorePreviewAttachmentRef:
    dataset_id: str
    attachment_id: str
    object_id: str
    parent_domain: SyncDomain
    parent_object_id: str
    content_type: str
    size_bytes: int
    payload_hash: str
    availability: str
    server_cursor: int
    adapter_version: int


@dataclass(frozen=True, slots=True)
class SyncRestorePreviewWarning:
    code: str
    message: str
    dataset_id: str | None = None
    attachment_id: str | None = None
    object_id: str | None = None
    payload_hash: str | None = None


@dataclass(frozen=True, slots=True)
class SyncRestorePreview:
    datasets: list[SyncRestorePreviewDataset] = field(default_factory=list)
    ordered_actions: list[SyncRestoreOrderedAction] = field(default_factory=list)
    safe_applies: list[SyncRestorePreviewObject] = field(default_factory=list)
    object_conflicts: list[SyncRestorePreviewObjectConflict] = field(default_factory=list)
    tombstones: list[SyncRestorePreviewObject] = field(default_factory=list)
    attachment_refs: list[SyncRestorePreviewAttachmentRef] = field(default_factory=list)
    missing_blobs: list[SyncRestorePreviewAttachmentRef] = field(default_factory=list)
    envelope_ranges: list[SyncRestorePreviewEnvelopeRange] = field(default_factory=list)
    total_counts: dict[str, int] = field(default_factory=dict)
    encryption: dict[str, object] = field(default_factory=dict)
    key_status: dict[str, dict[str, bool]] = field(default_factory=dict)
    warnings: list[SyncRestorePreviewWarning] = field(default_factory=list)
    generated_at: str | None = None
    filters_applied: dict[str, object] = field(default_factory=dict)
    restore_status: SyncRestoreCompletenessStatus | None = None
    domain_details: list[SyncRestoreDomainCompleteness] = field(default_factory=list)
    blob_details: list[SyncRestoreBlobCompleteness] = field(default_factory=list)
    metadata_only_allowed: bool = True


class SyncV2Service:
    """Core Sync v2 service with injected persistence and adapter dependencies."""

    def __init__(
        self,
        *,
        store: SyncV2Store,
        adapters: SyncAdapterRegistry,
        materializers: Mapping[SyncDomain, SyncMaterializer] | None = None,
        clock: Callable[[], str] | None = None,
        id_factory: Callable[[str], str] | None = None,
        blob_store: LocalSyncBlobStore | None = None,
        settings: SyncV2Settings | None = None,
        workspace_access_checker: WorkspaceAccessChecker | None = None,
        dataset_bootstrapper: object | None = None,
        notes_link_bootstrapper: object | None = None,
        notes_attachment_bootstrapper: object | None = None,
        notes_task_bootstrapper: object | None = None,
        notes_task_activity_bootstrapper: object | None = None,
        personal_context_service_resolver: Callable[[str], object] | None = None,
        personal_context_key_wrapper: Callable[..., str] | None = None,
        personal_context_key_fingerprint: Callable[..., str] | None = None,
        personal_context_authority_id: str = "tldw-server",
    ) -> None:
        self.store = store
        self.adapters = adapters
        self.materializers = dict(materializers or {})
        self.clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self.id_factory = id_factory or (lambda prefix: f"{prefix}-{uuid4().hex}")
        self.blob_store = blob_store
        self.settings = settings or SyncV2Settings()
        self.workspace_access_checker = workspace_access_checker
        self.dataset_bootstrapper = dataset_bootstrapper
        self.notes_link_bootstrapper = notes_link_bootstrapper
        self.notes_attachment_bootstrapper = notes_attachment_bootstrapper
        self.notes_task_bootstrapper = notes_task_bootstrapper
        self.notes_task_activity_bootstrapper = notes_task_activity_bootstrapper
        self.personal_context_service_resolver = personal_context_service_resolver
        self.personal_context_key_wrapper = personal_context_key_wrapper
        self.personal_context_key_fingerprint = personal_context_key_fingerprint
        self.personal_context_authority_id = (
            str(personal_context_authority_id).strip() or "tldw-server"
        )

    def _notes_task_domains_ready(self, dataset: SyncDataset | None) -> bool:
        """Return the single service-level task/activity activation predicate."""

        if dataset is None or not notes_task_sync_is_ready(
            domains=dataset.domains,
            metadata=dataset.metadata,
        ):
            return False
        try:
            adapters_ready = all(
                self.adapters.supports_version(domain, 1)
                for domain in NOTES_TASK_SYNC_DOMAINS
            )
        except KeyError:
            return False
        if not adapters_ready:
            return False
        task_db = getattr(self.materializers.get("notes.task"), "note_db", None)
        activity_db = getattr(
            self.materializers.get("notes.task_activity"),
            "note_db",
            None,
        )
        if task_db is None or task_db is not activity_db:
            return False
        try:
            return (
                task_db.task_store.resolve_task_compatibility_dataset_id(
                    owner_user_id=dataset.owner_user_id
                )
                == dataset.dataset_id
            )
        except Exception:  # noqa: BLE001 - malformed product authority fails closed.
            return False

    def _personal_context_domains_ready(
        self,
        dataset: SyncDataset | None = None,
    ) -> bool:
        """Return whether every Personal Context domain has a usable v1 path."""

        for domain in PERSONAL_CONTEXT_SYNC_DOMAINS:
            if not self.adapters.has_domain(domain):
                return False
            adapter = self.adapters.get(domain)
            if (
                not self.adapters.supports_version(domain, 1)
                or not getattr(adapter, "storage_encryption_ready", False)
                or getattr(self.materializers.get(domain), "domain", None) != domain
            ):
                return False
            if dataset is not None:
                key_custody_ready = getattr(adapter, "key_custody_ready", None)
                if not callable(key_custody_ready) or not key_custody_ready(dataset):
                    return False
        return True

    def capabilities(
        self,
        *,
        user_id: str | None = None,
        dataset_id: str | None = None,
    ) -> SyncV2Capabilities:
        """Return capabilities, optionally bound to one authorized dataset."""

        dataset: SyncDataset | None = None
        if dataset_id is not None:
            if user_id is None:
                raise SyncStoreError(
                    "Sync dataset was not found or is not accessible"
                )
            dataset = self._require_dataset_access(
                user_id=user_id,
                dataset_id=dataset_id,
            )
        try:
            attachment_adapter = self.adapters.get("attachment.ref")
        except KeyError:
            attachment_v2_writes_enabled = False
        else:
            attachment_v2_writes_enabled = bool(
                getattr(attachment_adapter, "v2_writes_enabled", False)
            )
        notes_task_ready = self._notes_task_domains_ready(dataset)
        personal_context_transport_ready = self._personal_context_domains_ready(
            dataset if self.settings.personal_context.available else None
        )
        private_dormant_domains = {
            *NOTES_MOODBOARD_STUDIO_DOMAINS,
            *NOTES_TASK_SYNC_DOMAINS,
        }
        supported_domains = [
            domain
            for domain in self.settings.supported_domains
            if domain not in private_dormant_domains
        ]
        operations = {
            domain: list(domain_operations)
            for domain, domain_operations in self.settings.operations.items()
            if domain not in private_dormant_domains
        }
        if notes_task_ready:
            supported_domains.extend(NOTES_TASK_SYNC_DOMAINS)
            operations.update(
                {
                    domain: list(domain_operations)
                    for domain, domain_operations in NOTES_TASK_SYNC_OPERATIONS.items()
                }
            )
        blob_transfer: dict[str, object] = {"supported": False}
        quota: dict[str, object] = {}
        if self.settings.supports_attachments:
            max_blob_bytes = self.settings.max_blob_bytes or self.settings.max_attachment_bytes
            blob_transfer = {
                "supported": True,
                "resumable_upload": self.settings.supports_resumable_upload,
                "resumable_download": self.settings.supports_resumable_download,
                "chunk_checksums": self.settings.supports_chunk_checksums,
                "full_checksum": self.settings.blob_checksum_algorithm,
                "storage_backend": self.settings.blob_storage_backend,
            }
            quota = {
                "max_blob_bytes": max_blob_bytes,
                "max_chunk_bytes": self.settings.max_chunk_bytes,
                "max_active_uploads": self.settings.max_active_blob_uploads,
                "user_blob_quota_bytes": self.settings.user_blob_quota_bytes,
                "reserved_blob_bytes": self.settings.reserved_blob_bytes,
                "used_blob_bytes": self.settings.used_blob_bytes,
            }
        warnings = list(self.settings.server_trusted_encryption.warnings)
        personal_context = self.settings.personal_context
        if not personal_context_transport_ready:
            personal_context = replace(
                personal_context,
                available=False,
                blockers=tuple(
                    dict.fromkeys(
                        (
                            *personal_context.blockers,
                            "personal_context_transport_unavailable",
                        )
                    )
                ),
            )
        if (
            "server_trusted_v1" not in self.settings.encryption_policies
            or not self.settings.server_trusted_encryption.ready
        ):
            personal_context = replace(
                personal_context,
                available=False,
                blockers=tuple(
                    dict.fromkeys(
                        (
                            *personal_context.blockers,
                            "personal_context_server_trusted_unavailable",
                        )
                    )
                ),
            )
        compatibility_flags: dict[str, bool] = {}
        if "client_private_v1" in self.settings.encryption_policies:
            compatibility_flags["server_frontend_client_private_mutation"] = False
            warnings = _append_warning_once(
                warnings,
                client_private_server_frontend_limitation_warning(),
            )
        return SyncV2Capabilities(
            protocol_version=self.settings.protocol_version,
            min_supported_protocol_version=self.settings.min_supported_protocol_version,
            supported_domains=supported_domains,
            operations=operations,
            encryption=self.settings.server_trusted_encryption.encryption,
            blob_transfer=blob_transfer,
            encryption_policies=list(self.settings.encryption_policies),
            personal_context=personal_context,
            max_batch_size=self.settings.max_batch_size,
            max_envelope_payload_bytes=self.settings.max_envelope_payload_bytes,
            max_attachment_bytes=self.settings.max_attachment_bytes,
            domain_schemas=sync_v2_advertised_domain_schemas(
                (
                    _sync_v2_internal_domain_schemas()
                    if notes_task_ready
                    else sync_v2_domain_schemas()
                ),
                advertised_domains=supported_domains,
            ),
            supported_adapter_versions=sync_v2_server_supported_adapter_versions(
                notes_task_sync_ready=notes_task_ready,
                personal_context_sync_ready=personal_context_transport_ready,
            ),
            writable_adapter_versions=sync_v2_dataset_writable_adapter_versions(
                dataset,
                notes_attachment_sync_enabled=attachment_v2_writes_enabled,
                supports_attachments=self.settings.supports_attachments,
                notes_task_sync_ready=notes_task_ready,
                personal_context_sync_ready=personal_context.available,
            ),
            quota=quota,
            supports_attachments=self.settings.supports_attachments,
            compatibility_flags=compatibility_flags,
            server_time=self.clock() or None,
            warnings=warnings,
        )

    def register_device(
        self,
        *,
        user_id: str,
        display_name: str,
        client_type: str,
        device_id: str | None = None,
        client_version: str | None = None,
        capabilities: dict[str, object] | None = None,
    ) -> SyncDeviceRegistration:
        resolved_device_id = device_id or self.id_factory("device")
        _require_client_device_id(resolved_device_id)
        device = self._upsert_device(
            SyncDeviceUpsert(
                device_id=resolved_device_id,
                user_id=user_id,
                display_name=display_name,
                client_type=client_type,
                client_version=client_version,
                capabilities=dict(capabilities or {}),
            )
        )
        return SyncDeviceRegistration(device=device, server_capabilities=self.capabilities())

    def _upsert_device(
        self,
        device: SyncDeviceUpsert,
        *,
        merge_capabilities: bool = False,
    ) -> SyncDevice:
        """Validate negotiation state before every service-owned device upsert."""

        observed = self.store.get_device(device.user_id, device.device_id)
        submitted_capabilities = _validated_device_capabilities(
            observed,
            device.capabilities,
            merge_capabilities=merge_capabilities,
        )

        def resolve_capabilities(current: SyncDevice | None) -> dict[str, object]:
            negotiation_patch = _atomic_negotiation_patch(
                current,
                submitted_capabilities,
                device.capabilities,
            )
            return _validated_device_capabilities(
                current,
                device.capabilities,
                merge_capabilities=merge_capabilities,
                negotiation_patch=negotiation_patch,
            )

        return self.store.upsert_device(
            replace(device, capabilities=submitted_capabilities),
            capabilities_resolver=resolve_capabilities,
        )

    def list_devices(
        self,
        *,
        user_id: str,
        include_revoked: bool = False,
    ) -> list[SyncDevice]:
        """List registered devices for a user."""

        return self.store.list_devices_for_user(
            user_id,
            include_revoked=include_revoked,
        )

    def update_device(
        self,
        *,
        user_id: str,
        device_id: str,
        display_name: str | None = None,
        user_label: str | None = None,
        client_version: str | None = None,
        capabilities: dict[str, object] | None = None,
    ) -> SyncDevice:
        """Update mutable device metadata without changing lifecycle state."""

        existing = self.store.get_device(user_id, device_id)
        if existing is None:
            raise SyncStoreError("Sync device was not found or is not accessible")
        return self._upsert_device(
            SyncDeviceUpsert(
                device_id=existing.device_id,
                user_id=existing.user_id,
                display_name=display_name or existing.display_name,
                client_type=existing.client_type,
                client_version=(
                    client_version
                    if client_version is not None
                    else existing.client_version
                ),
                capabilities=dict(capabilities or {}),
                status=existing.status,
                user_label=user_label if user_label is not None else existing.user_label,
                authorized_at=existing.authorized_at,
                revoked_at=existing.revoked_at,
                revoked_reason=existing.revoked_reason,
            ),
            merge_capabilities=True,
        )

    def create_device_authorization(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        authorization_method: str,
        idempotency_key: str | None = None,
    ) -> SyncDeviceAuthorization:
        """Create a pending device authorization request."""

        self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        return self.store.create_device_authorization(
            SyncDeviceAuthorizationCreate(
                authorization_id=self.id_factory("device-authorization"),
                dataset_id=dataset_id,
                user_id=user_id,
                device_id=device_id,
                authorization_method=authorization_method,
                idempotency_key=idempotency_key,
            )
        )

    def approve_device_authorization(
        self,
        authorization_id: str,
        *,
        user_id: str,
        dataset_id: str,
        approving_device_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> SyncDeviceAuthorization:
        """Approve a pending device authorization and activate the device."""

        self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        if approving_device_id is not None:
            self._require_registered_device(user_id, approving_device_id)
        return self.store.approve_device_authorization(
            authorization_id,
            user_id=user_id,
            dataset_id=dataset_id,
            approving_device_id=approving_device_id,
            idempotency_key=idempotency_key,
        )

    def revoke_device(
        self,
        *,
        user_id: str,
        device_id: str,
        reason: str | None = None,
        revoke_key_records: bool = False,
    ) -> SyncDevice:
        """Revoke a device from future sync operations."""

        return self.store.revoke_device(
            user_id=user_id,
            device_id=device_id,
            reason=reason,
            revoke_key_records=revoke_key_records,
        )

    def pause_device(self, *, user_id: str, device_id: str) -> SyncDevice:
        """Pause a device so it cannot perform device-scoped sync calls."""

        existing = self.store.get_device(user_id, device_id)
        if existing is None or existing.status == "revoked":
            raise SyncStoreError("Sync device was not found or is not accessible")
        return self._upsert_device(
            SyncDeviceUpsert(
                device_id=existing.device_id,
                user_id=existing.user_id,
                display_name=existing.display_name,
                client_type=existing.client_type,
                client_version=existing.client_version,
                capabilities={},
                status="paused",
                user_label=existing.user_label,
                authorized_at=existing.authorized_at,
            ),
            merge_capabilities=True,
        )

    def resume_device(self, *, user_id: str, device_id: str) -> SyncDevice:
        """Resume a paused device after user approval."""

        existing = self.store.get_device(user_id, device_id)
        if existing is None or existing.status in {"revoked", "pending_authorization"}:
            raise SyncStoreError("Sync device was not found or is not accessible")
        return self._upsert_device(
            SyncDeviceUpsert(
                device_id=existing.device_id,
                user_id=existing.user_id,
                display_name=existing.display_name,
                client_type=existing.client_type,
                client_version=existing.client_version,
                capabilities={},
                status="active",
                user_label=existing.user_label,
                authorized_at=existing.authorized_at or self.clock(),
            ),
            merge_capabilities=True,
        )

    def acknowledge_device_state(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        domain_acks: Sequence[SyncDeviceDomainAckCreate] = (),
        blob_acks: Sequence[SyncDeviceBlobAckCreate] = (),
        blob_id_acks: Sequence[SyncDeviceBlobIdAckCreate] = (),
    ) -> SyncDeviceAcknowledgmentSummary:
        """Record a device's durable application/verification acknowledgments."""

        self._require_registered_device(user_id, device_id)
        self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        return self.store.acknowledge_device_state_atomic(
            dataset_id,
            device_id,
            domain_acks=domain_acks,
            blob_acks=blob_acks,
            blob_id_acks=blob_id_acks,
        )

    def get_background_policy(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
    ) -> SyncBackgroundPolicy:
        """Return stored or default background sync policy for a dataset/device."""

        self._require_registered_device(user_id, device_id)
        self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        stored = self.store.get_background_policy(dataset_id, device_id)
        if stored is not None:
            return stored
        return self._default_background_policy(dataset_id, device_id)

    def update_background_policy(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        enabled: bool | None = None,
        minimum_interval_seconds: int | None = None,
        backoff_floor_seconds: int | None = None,
        max_batch_size: int | None = None,
        max_blob_bytes_per_run: int | None = None,
        respect_metered_networks: bool | None = None,
        maintenance_window: dict[str, object] | None = None,
        paused_reason: str | None = None,
        pending_local_changes: bool | None = None,
    ) -> SyncBackgroundPolicy:
        """Persist background sync policy hints and user pause/resume intent."""

        current = self.get_background_policy(
            user_id=user_id,
            dataset_id=dataset_id,
            device_id=device_id,
        )
        resolved_enabled = current.enabled if enabled is None else enabled
        return self.store.upsert_background_policy(
            SyncBackgroundPolicyUpsert(
                dataset_id=dataset_id,
                device_id=device_id,
                enabled=resolved_enabled,
                minimum_interval_seconds=(
                    current.minimum_interval_seconds
                    if minimum_interval_seconds is None
                    else minimum_interval_seconds
                ),
                backoff_floor_seconds=(
                    current.backoff_floor_seconds
                    if backoff_floor_seconds is None
                    else backoff_floor_seconds
                ),
                max_batch_size=current.max_batch_size if max_batch_size is None else max_batch_size,
                max_blob_bytes_per_run=(
                    current.max_blob_bytes_per_run
                    if max_blob_bytes_per_run is None
                    else max_blob_bytes_per_run
                ),
                respect_metered_networks=(
                    current.respect_metered_networks
                    if respect_metered_networks is None
                    else respect_metered_networks
                ),
                maintenance_window=(
                    current.maintenance_window
                    if maintenance_window is None
                    else dict(maintenance_window)
                ),
                paused_reason=(
                    paused_reason
                    if paused_reason is not None
                    else (None if resolved_enabled else current.paused_reason)
                ),
                pending_local_changes=(
                    current.pending_local_changes
                    if pending_local_changes is None
                    else pending_local_changes
                ),
            )
        )

    def acquire_background_lease(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        lease_id: str | None = None,
        ttl_seconds: int = 120,
    ) -> SyncBackgroundLease:
        """Acquire or refresh a short-lived advisory background sync lease."""

        self._require_registered_device(user_id, device_id)
        self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        return self.store.acquire_background_lease(
            SyncBackgroundLeaseCreate(
                dataset_id=dataset_id,
                device_id=device_id,
                lease_id=lease_id or self.id_factory("background-lease"),
                ttl_seconds=ttl_seconds,
                requested_at=self.clock(),
            )
        )

    def background_status(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
    ) -> SyncBackgroundStatus:
        """Return profile-level and per-domain background sync status."""

        self._require_registered_device(user_id, device_id)
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        policy = self.get_background_policy(
            user_id=user_id,
            dataset_id=dataset_id,
            device_id=device_id,
        )
        domains = self.store.summarize_background_domains(
            dataset_id,
            device_id,
            domains=dataset.domains,
        )
        lease = self.store.get_background_lease(dataset_id, device_id)
        quota = self.store.summarize_blob_quota(user_id, dataset_id=dataset_id)
        quota_limit = self.settings.user_blob_quota_bytes
        quota_used = quota.used_blob_bytes + quota.reserved_blob_bytes
        quota_pressure = {
            "reserved_blob_bytes": quota.reserved_blob_bytes,
            "used_blob_bytes": quota.used_blob_bytes,
            "active_upload_count": quota.active_upload_count,
            "limit_bytes": quota_limit,
            "pressure_ratio": (
                round(quota_used / quota_limit, 4)
                if quota_limit and quota_limit > 0
                else 0.0
            ),
        }
        conflict_count = sum(domain.unresolved_conflicts for domain in domains)
        replayable_failure_count = sum(domain.replayable_failures for domain in domains)
        missing_blob_count = sum(
            domain.blob_completeness.get("missing_blob_count", 0)
            for domain in domains
        )
        if conflict_count:
            restore_completeness: SyncRestoreCompletenessStatus = "blocked_by_conflicts"
        elif missing_blob_count:
            restore_completeness = "blob_incomplete"
        elif any(domain.last_server_sequence for domain in domains):
            restore_completeness = "content_complete"
        else:
            restore_completeness = "metadata_ready"
        return SyncBackgroundStatus(
            dataset_id=dataset_id,
            device_id=device_id,
            policy=policy,
            lease=lease,
            domains=domains,
            conflict_count=conflict_count,
            replayable_failure_count=replayable_failure_count,
            quota_pressure=quota_pressure,
            restore_completeness=restore_completeness,
            server_time=self.clock(),
        )

    def diagnostics(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None = None,
        retention_limit: int | None = None,
        attachment_sample_limit: int = 0,
        attachment_total_sample_limit: int = 500,
    ) -> SyncDiagnosticsReport:
        """Return redacted Sync v2 dataset diagnostics."""

        if device_id is not None:
            self._require_registered_device(user_id, device_id)
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        if (
            isinstance(attachment_sample_limit, bool)
            or attachment_sample_limit < 0
            or attachment_sample_limit > 100
        ):
            raise SyncStoreError(
                "sync_attachment_diagnostic_category_sample_limit_exceeded"
            )
        if (
            isinstance(attachment_total_sample_limit, bool)
            or attachment_total_sample_limit < 0
            or attachment_total_sample_limit > 500
        ):
            raise SyncStoreError("sync_attachment_diagnostic_total_sample_limit_exceeded")
        scan_limit = self.settings.restore_manifest_scan_limit
        envelopes = self.store.list_accepted_envelopes_for_replay(
            dataset_id,
            since_cursor=0,
            limit=scan_limit,
        )
        conflicts = self.store.list_conflicts(dataset_id, status="unresolved")
        active_devices = self._retention_active_devices(dataset)
        blob_quota = self.store.summarize_blob_quota(user_id, dataset_id=dataset_id)
        key_records = self.store.list_key_records(dataset_id, user_id=user_id)
        retention = self.retention_dry_run(
            user_id=user_id,
            dataset_id=dataset_id,
            device_id=device_id,
            audit_mode=True,
            limit=retention_limit or min(scan_limit, 100),
        )
        domain_diagnostics = self._diagnostics_domains(
            domains=dataset.domains,
            envelopes=envelopes,
            conflicts=conflicts,
        )
        attachment_lifecycle = self._attachment_lifecycle_diagnostics(
            dataset=dataset,
            user_id=user_id,
            envelopes=envelopes,
            conflicts=conflicts,
            retention=retention,
            sample_limit=attachment_sample_limit,
            total_sample_limit=attachment_total_sample_limit,
        )
        return SyncDiagnosticsReport(
            dataset_id=dataset_id,
            generated_at=self.clock(),
            domains=domain_diagnostics,
            devices=self._diagnostics_devices(
                dataset_id=dataset_id,
                domains=dataset.domains,
                devices=active_devices,
            ),
            blob_health=SyncDiagnosticsBlobHealth(
                blob_object_count=attachment_lifecycle.counts.get("blob_total", 0),
                available_blob_bytes=blob_quota.used_blob_bytes,
                active_upload_count=blob_quota.active_upload_count,
                reserved_blob_bytes=blob_quota.reserved_blob_bytes,
                quota_limit_bytes=self.settings.user_blob_quota_bytes,
            ),
            key_summary=self._diagnostics_key_summary(key_records),
            retention=SyncDiagnosticsRetentionSummary(
                dry_run=retention.dry_run,
                mutation_performed=retention.mutation_performed,
                candidate_count=retention.candidate_count,
                blocked_count=retention.blocked_count,
                blocker_counts=dict(retention.blocker_counts),
            ),
            attachment_lifecycle=attachment_lifecycle,
        )

    def retention_compact(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None = None,
        domains: Sequence[SyncDomain] | None = None,
        confirm: bool = False,
        apply_envelope_compaction: bool = True,
        apply_tombstone_prune: bool = True,
        apply_binding_release: bool = True,
        apply_blob_gc: bool = True,
        minimum_envelope_age_seconds: int = 0,
        minimum_tombstone_age_seconds: int = 0,
        offline_restore_window_seconds: int = 0,
        limit: int | None = None,
    ) -> SyncRetentionApplyResult:
        """Apply unblocked retention candidates with conservative guards."""

        dry_run = self.retention_dry_run(
            user_id=user_id,
            dataset_id=dataset_id,
            device_id=device_id,
            domains=domains,
            audit_mode=False,
            minimum_envelope_age_seconds=minimum_envelope_age_seconds,
            minimum_tombstone_age_seconds=minimum_tombstone_age_seconds,
            offline_restore_window_seconds=offline_restore_window_seconds,
            limit=limit,
        )
        selected = [
            candidate
            for candidate in dry_run.candidates
            if _retention_apply_candidate_enabled(
                candidate,
                apply_envelope_compaction=apply_envelope_compaction,
                apply_tombstone_prune=apply_tombstone_prune,
                apply_binding_release=apply_binding_release,
                apply_blob_gc=apply_blob_gc,
            )
        ]
        if not confirm:
            return SyncRetentionApplyResult(
                dataset_id=dataset_id,
                dry_run=True,
                mutation_performed=False,
                confirmation_required=True,
                evaluated_at=self.clock(),
                candidate_count=dry_run.candidate_count,
                blocked_count=dry_run.blocked_count,
                skipped_count=dry_run.candidate_count - len(selected),
                blockers=["retention_confirmation_required"],
                blocker_counts=dict(dry_run.blocker_counts),
            )

        blocked = [
            candidate
            for candidate in selected
            if candidate.blockers
            and candidate.candidate_type not in {"binding_release", "blob_gc"}
        ]
        if blocked:
            return SyncRetentionApplyResult(
                dataset_id=dataset_id,
                dry_run=False,
                mutation_performed=False,
                evaluated_at=self.clock(),
                candidate_count=dry_run.candidate_count,
                blocked_count=len(blocked),
                skipped_count=dry_run.candidate_count - len(selected),
                blockers=["retention_blocked_candidates_present"],
                blocker_counts=_retention_blocker_counts(blocked),
            )

        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        domain_compactions, revalidated_domain_blocked = (
            self._apply_retention_domain_compactions(
            dataset=dataset,
            candidates=[
                candidate
                for candidate in selected
                if candidate.candidate_type in {"envelope_compaction", "tombstone_prune"}
            ],
            minimum_envelope_age_seconds=minimum_envelope_age_seconds,
            minimum_tombstone_age_seconds=minimum_tombstone_age_seconds,
            offline_restore_window_seconds=offline_restore_window_seconds,
            )
        )
        initially_blocked_bindings = [
            candidate
            for candidate in selected
            if candidate.candidate_type == "binding_release" and candidate.blockers
        ]
        binding_releases, revalidated_binding_blocked = (
            self._apply_retention_binding_releases(
                dataset=dataset,
                candidates=[
                    candidate
                    for candidate in selected
                    if candidate.candidate_type == "binding_release"
                    and not candidate.blockers
                ],
                minimum_envelope_age_seconds=minimum_envelope_age_seconds,
                minimum_tombstone_age_seconds=minimum_tombstone_age_seconds,
                offline_restore_window_seconds=offline_restore_window_seconds,
            )
        )
        blob_gc, revalidated_blob_blocked, blob_fence_mutated = (
            self._apply_retention_blob_gc(
                dataset=dataset,
                candidates=[
                    candidate
                    for candidate in selected
                    if candidate.candidate_type == "blob_gc"
                ],
                minimum_tombstone_age_seconds=minimum_tombstone_age_seconds,
                offline_restore_window_seconds=offline_restore_window_seconds,
            )
        )
        applied_count = sum(
            int(item["candidate_count"]) for item in domain_compactions
        ) + len(binding_releases) + len(blob_gc)
        revalidated_blocked = (
            revalidated_domain_blocked
            + initially_blocked_bindings
            + revalidated_binding_blocked
            + revalidated_blob_blocked
        )
        return SyncRetentionApplyResult(
            dataset_id=dataset_id,
            dry_run=False,
            mutation_performed=applied_count > 0 or blob_fence_mutated,
            evaluated_at=self.clock(),
            candidate_count=dry_run.candidate_count,
            applied_count=applied_count,
            blocked_count=len(revalidated_blocked),
            skipped_count=(
                dry_run.candidate_count - len(selected) + len(revalidated_blocked)
            ),
            blockers=(
                ["retention_revalidation_blocked"]
                if revalidated_blocked
                else []
            ),
            blocker_counts=_retention_blocker_counts(revalidated_blocked),
            domain_compactions=domain_compactions,
            binding_releases=binding_releases,
            blob_gc=blob_gc,
        )

    def retention_dry_run(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None = None,
        domains: Sequence[SyncDomain] | None = None,
        audit_mode: bool = True,
        minimum_envelope_age_seconds: int = 0,
        minimum_tombstone_age_seconds: int = 0,
        offline_restore_window_seconds: int = 0,
        limit: int | None = None,
    ) -> SyncRetentionDryRunResult:
        """Return read-only retention/GC candidates and safety blockers."""

        if device_id is not None:
            self._require_registered_device(user_id, device_id)
        if minimum_envelope_age_seconds < 0 or minimum_tombstone_age_seconds < 0:
            raise SyncStoreError("Sync retention age windows must be non-negative")
        if offline_restore_window_seconds < 0:
            raise SyncStoreError("Sync retention restore window must be non-negative")
        scan_limit = limit or self.settings.restore_manifest_scan_limit
        if scan_limit <= 0:
            raise SyncStoreError("Sync retention scan limit must be greater than zero")

        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        selected_domains = [
            domain for domain in dataset.domains if domains is None or domain in domains
        ]
        active_devices = self._retention_active_devices(dataset)
        envelopes = self.store.list_accepted_envelopes_for_replay(
            dataset_id,
            since_cursor=0,
            limit=scan_limit,
        )
        latest_by_object = self._retention_latest_envelopes_by_object(envelopes)
        restore_window_blocked = self._retention_restore_window_active(
            active_devices,
            offline_restore_window_seconds,
        )
        workspace_ack_scope_blocked = self._retention_workspace_ack_scope_blocked(dataset)
        candidates: list[SyncRetentionCandidate] = []

        for envelope in envelopes:
            if envelope.domain not in selected_domains:
                continue
            candidate_type = self._retention_envelope_candidate_type(
                envelope,
                latest_by_object=latest_by_object,
            )
            if candidate_type is None:
                continue
            blockers: list[str] = []
            if audit_mode:
                blockers.append("retention_audit_mode")
            window_seconds = (
                minimum_tombstone_age_seconds
                if candidate_type == "tombstone_prune"
                else minimum_envelope_age_seconds
            )
            window_blocker = (
                "retention_tombstone_window_active"
                if candidate_type == "tombstone_prune"
                else "retention_envelope_window_active"
            )
            if self._retention_window_active(envelope.server_timestamp, window_seconds):
                blockers.append(window_blocker)
            if restore_window_blocked:
                blockers.append("retention_restore_window_active")
            if workspace_ack_scope_blocked:
                blockers.append("retention_workspace_ack_scope_unknown")
            unacknowledged = self._retention_unacknowledged_devices(
                dataset_id=dataset_id,
                domain=envelope.domain,
                adapter_version=envelope.adapter_version,
                server_sequence=envelope.server_sequence,
                active_devices=active_devices,
            )
            if unacknowledged:
                blockers.append("retention_unacknowledged_device")
            blockers.extend(
                blocker
                for blocker in self._notes_task_retention_blockers(
                    dataset=dataset,
                    envelope=envelope,
                )
                if blocker not in blockers
            )
            candidates.append(
                SyncRetentionCandidate(
                    candidate_type=candidate_type,
                    dataset_id=dataset_id,
                    domain=envelope.domain,
                    object_id=envelope.object_id,
                    server_sequence=envelope.server_sequence,
                    blockers=blockers,
                    required_device_ids=[device.device_id for device in active_devices],
                    unacknowledged_device_ids=unacknowledged,
                    reason=(
                        "superseded envelope"
                        if candidate_type == "envelope_compaction"
                        else "tombstone retained for deletion window"
                    ),
                )
            )

        remaining_budget = scan_limit - len(envelopes)
        if "attachment.ref" in selected_domains and remaining_budget > 0:
            binding_candidates = self._retention_binding_release_candidates(
                dataset=dataset,
                active_devices=active_devices,
                audit_mode=audit_mode,
                restore_window_blocked=restore_window_blocked,
                minimum_envelope_age_seconds=minimum_envelope_age_seconds,
                minimum_tombstone_age_seconds=minimum_tombstone_age_seconds,
                limit=remaining_budget,
            )
            candidates.extend(binding_candidates)
            remaining_budget -= len(binding_candidates)
        if "attachment.ref" in selected_domains and remaining_budget > 0:
            candidates.extend(
                self._retention_blob_candidates(
                    dataset=dataset,
                    active_devices=active_devices,
                    audit_mode=audit_mode,
                    restore_window_blocked=restore_window_blocked,
                    minimum_tombstone_age_seconds=minimum_tombstone_age_seconds,
                    limit=remaining_budget,
                )
            )

        blocker_counts = _retention_blocker_counts(candidates)
        return SyncRetentionDryRunResult(
            dataset_id=dataset_id,
            evaluated_at=self.clock(),
            audit_mode=audit_mode,
            minimum_envelope_age_seconds=minimum_envelope_age_seconds,
            minimum_tombstone_age_seconds=minimum_tombstone_age_seconds,
            offline_restore_window_seconds=offline_restore_window_seconds,
            candidate_count=len(candidates),
            blocked_count=sum(1 for candidate in candidates if candidate.blockers),
            blocker_counts=blocker_counts,
            candidates=candidates,
        )

    def enroll_dataset(
        self,
        *,
        user_id: str,
        dataset_id: str | None = None,
        scope_type: str = "personal",
        domains: Sequence[SyncDomain] | None = None,
        encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY,
        workspace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> SyncDatasetEnrollment:
        requested_domains = set(domains or ())
        requested_metadata = dict(metadata or {})
        reserved_metadata = {
            "notes_organization_v1",
            "notes_link_v1",
            "notes_attachment_v2",
            "default_personal",
            "client_family",
            "personal_context",
            *NOTES_MOODBOARD_STUDIO_SERVER_METADATA_KEYS,
            *NOTES_TASK_SERVER_METADATA_KEYS,
        }
        if (
            requested_domains.intersection(
                {
                    *NOTES_ORGANIZATION_DOMAINS,
                    *NOTES_LINK_DOMAINS,
                    *PERSONAL_CONTEXT_SYNC_DOMAINS,
                }
            )
            or reserved_metadata.intersection(requested_metadata)
        ):
            raise SyncStoreError("sync_reserved_dataset_enrollment")
        requested_task_domains = requested_domains.intersection(
            NOTES_TASK_SYNC_DOMAINS
        )
        if requested_task_domains and requested_task_domains != set(
            NOTES_TASK_SYNC_DOMAINS
        ):
            raise SyncStoreError("notes_task_sync_domains_incomplete")
        existing = (
            self.store.get_dataset(dataset_id, owner_user_id=user_id)
            if dataset_id is not None
            else None
        )
        if (
            existing is not None
            and set(NOTES_TASK_SYNC_DOMAINS).issubset(existing.domains)
            and not requested_task_domains
        ):
            raise SyncStoreError("notes_task_sync_disable_forbidden")
        self._require_server_trusted_encryption_ready()
        if scope_type == "workspace":
            self._require_workspace_sync_access(user_id=user_id, workspace_id=workspace_id)
            enrolled_domains = list(domains or WORKSPACE_SYNC_DOMAINS)
        else:
            enrolled_domains = list(domains or M1_SYNC_DOMAINS)
        if requested_task_domains:
            if (
                scope_type != "personal"
                or encryption_policy != DEFAULT_M1_ENCRYPTION_POLICY
                or "notes.note" not in enrolled_domains
            ):
                raise SyncStoreError("notes_task_sync_enrollment_invalid")
            if (
                existing is None
                or existing.metadata.get("default_personal") is not True
                or existing.metadata.get("client_family") != "chatbook"
            ):
                raise SyncStoreError("notes_task_sync_enrollment_invalid")
            requested_metadata = {
                "default_personal": True,
                "client_family": "chatbook",
                **requested_metadata,
            }
            if notes_task_sync_is_ready(
                domains=existing.domains,
                metadata=existing.metadata,
            ):
                requested_metadata.update(
                    {
                        key: existing.metadata[key]
                        for key in NOTES_TASK_SERVER_METADATA_KEYS
                        if key in existing.metadata
                    }
                )
            if existing is None or not notes_task_sync_is_ready(
                domains=existing.domains,
                metadata=existing.metadata,
            ):
                enrolled_domains = [
                    domain
                    for domain in enrolled_domains
                    if domain not in NOTES_TASK_SYNC_DOMAINS
                ]
        dataset = self.store.enroll_dataset(
            SyncDatasetCreate(
                dataset_id=dataset_id or self.id_factory("dataset"),
                owner_user_id=user_id,
                scope_type=scope_type,
                encryption_policy=encryption_policy,
                domains=enrolled_domains,
                workspace_id=workspace_id,
                metadata=requested_metadata,
            )
        )
        if requested_task_domains:
            dataset = self._activate_notes_task_sync(dataset)
        return SyncDatasetEnrollment(
            dataset=replace(
                dataset,
                metadata=_redact_private_sync_server_metadata(dataset.metadata),
            ),
            cursors=dict.fromkeys(dataset.domains, "0"),
            key_setup_required=False,
        )

    def _activate_notes_task_sync(self, dataset: SyncDataset) -> SyncDataset:
        """Rekey product state and advance one resumable dual-bootstrap page."""

        task_bootstrapper = self.notes_task_bootstrapper
        activity_bootstrapper = self.notes_task_activity_bootstrapper
        task_db = getattr(task_bootstrapper, "note_db", None)
        activity_db = getattr(activity_bootstrapper, "note_db", None)
        if (
            task_bootstrapper is None
            or activity_bootstrapper is None
            or task_db is None
            or task_db is not activity_db
            or any(
                not self.adapters.has_domain(domain)
                or not self.adapters.supports_version(domain, 1)
                or domain not in self.materializers
                for domain in NOTES_TASK_SYNC_DOMAINS
            )
        ):
            raise SyncStoreError("notes_task_activation_unavailable")

        with self.store.retention_domain_guard(
            dataset.dataset_id,
            "notes.note",
            ("notes-task-activation",),
        ):
            task_db.task_store.bind_local_task_graph_to_dataset(
                owner_user_id=dataset.owner_user_id,
                target_dataset_id=dataset.dataset_id,
            )
        current = self.store.begin_notes_task_activation(
            dataset.dataset_id,
            owner_user_id=dataset.owner_user_id,
        )
        current = task_bootstrapper.bootstrap(service=self, dataset=current)
        task_state = current.metadata.get("notes_task_v1")
        if not isinstance(task_state, Mapping) or task_state.get("state") != "ready":
            return current
        current = activity_bootstrapper.bootstrap(service=self, dataset=current)
        activity_state = current.metadata.get("notes_task_activity_v1")
        if (
            not isinstance(activity_state, Mapping)
            or activity_state.get("state") != "ready"
        ):
            return current
        return self.store.activate_notes_task_domains(
            current.dataset_id,
            owner_user_id=current.owner_user_id,
        )

    def profile(
        self,
        *,
        user_id: str,
        device_id: str | None = None,
    ) -> SyncProfileStatus:
        """Return current profile state without creating sync records."""

        return self._profile_manager().profile(user_id=user_id, device_id=device_id)

    def bootstrap_profile(
        self,
        *,
        user_id: str,
        mode: str,
        device_id: str | None = None,
        device_name: str | None = None,
        client_profile_id: str | None = None,
        client_family: str = "chatbook",
        client_version: str | None = None,
        client_instance: dict[str, object] | None = None,
        requested_domains: Sequence[SyncDomain] | None = None,
    ) -> SyncProfileStatus:
        """Idempotently bootstrap the user's default Sync v2 M1 profile."""

        if device_id is not None:
            _require_client_device_id(device_id)
        return self._profile_manager().bootstrap_profile(
            user_id=user_id,
            mode=mode,
            device_id=device_id,
            device_name=device_name,
            client_profile_id=client_profile_id,
            client_family=client_family,
            client_version=client_version,
            client_instance=dict(client_instance or {}),
            requested_domains=requested_domains,
        )

    def bootstrap_personal_context(
        self,
        *,
        user_id: str,
        device_id: str,
        required_schema_version: int | None = None,
        required_quotas: Mapping[str, int] | None = None,
        expected_purge_generation: int | None = None,
    ) -> PersonalContextBootstrap:
        """Return an authenticated device's canonical first-link snapshot."""

        return self._profile_manager().bootstrap_personal_context(
            user_id=user_id,
            device_id=device_id,
            authority_id=self.personal_context_authority_id,
            required_schema_version=required_schema_version,
            required_quotas=required_quotas,
            expected_purge_generation=expected_purge_generation,
        )

    def complete_personal_context_link(
        self,
        *,
        user_id: str,
        device_id: str,
        dataset_id: str,
        bootstrap_cursor: str,
    ) -> None:
        """Open the narrow post-review Personal Context push transition."""

        self._profile_manager().complete_personal_context_link(
            user_id=user_id,
            device_id=device_id,
            dataset_id=dataset_id,
            bootstrap_cursor=bootstrap_cursor,
        )

    def profile_status(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None = None,
    ) -> SyncProfileStatus:
        """Return profile-level and per-domain status for an existing dataset."""

        return self._profile_manager().profile_status(
            user_id=user_id,
            dataset_id=dataset_id,
            device_id=device_id,
        )

    def notes_attachment_bootstrap_diagnostics(
        self,
        *,
        user_id: str,
        dataset_id: str | None = None,
        sample_limit: int = 0,
        dry_run: bool = False,
    ) -> SyncNotesAttachmentBootstrapDiagnostics:
        """Return bounded read-only diagnostics for legacy attachment bootstrap."""

        return self._profile_manager().notes_attachment_bootstrap_diagnostics(
            user_id=user_id,
            dataset_id=dataset_id,
            sample_limit=sample_limit,
            dry_run=dry_run,
        )

    def _expand_task_client_push(
        self,
        *,
        dataset: SyncDataset,
        device: SyncDevice,
        envelope: SyncEnvelopeCreate,
    ) -> tuple[SyncEnvelopeCreate, ...]:
        """Expand one authenticated task envelope into its closed activity group."""

        if envelope.domain != "notes.task":
            raise SyncStoreError("notes_task_client_group_invalid")
        if any(
            value is not None
            for value in (
                envelope.mutation_group_id,
                envelope.mutation_step,
                envelope.mutation_step_count,
                envelope.mutation_plan_hash,
            )
        ):
            raise SyncStoreError("notes_task_client_group_invalid")
        task_state = dataset.metadata.get("notes_task_v1")
        activity_state = dataset.metadata.get("notes_task_activity_v1")
        if (
            "notes.task_activity" not in dataset.domains
            or not isinstance(task_state, Mapping)
            or task_state.get("state") != "ready"
            or not isinstance(activity_state, Mapping)
            or activity_state.get("state") != "ready"
        ):
            raise SyncStoreError("notes_task_sync_not_ready")

        prior_head: SyncEnvelope | None = None
        if envelope.base_server_cursor is not None:
            candidate = self.store.get_envelope_by_server_cursor(
                envelope.base_server_cursor
            )
            if (
                candidate is None
                or candidate.dataset_id != dataset.dataset_id
                or candidate.domain != "notes.task"
                or candidate.object_id != envelope.object_id
                or candidate.object_revision != envelope.base_object_revision
                or candidate.payload_hash != envelope.base_object_hash
            ):
                raise SyncStoreError("notes_task_client_base_invalid")
            prior_head = candidate

        after = parse_notes_task_v1(
            envelope.payload,
            owner_user_id=dataset.owner_user_id,
        )
        before = (
            parse_notes_task_v1(
                prior_head.payload,
                owner_user_id=dataset.owner_user_id,
            )
            if prior_head is not None
            else None
        )
        restore_intent = envelope.routing_metadata.get("restore_intent") is True
        event_type, old_value, new_value = _client_task_activity_values(
            before=before,
            after=after,
            operation=envelope.operation,
            prior_head=prior_head,
            restore_intent=restore_intent,
        )
        occurred_at = normalize_sync_timestamp(envelope.created_at_client)
        if occurred_at is None:
            raise SyncStoreError("notes_task_client_timestamp_invalid")
        from .notes_task_coordinator import _task_activity_id

        activity_id = _task_activity_id(
            (
                dataset.dataset_id,
                device.device_id,
                envelope.client_envelope_id,
                event_type,
                envelope.payload_hash,
            )
        )
        activity_payload = parse_notes_task_activity_v1(
            {
                "activity_id": activity_id,
                "note_id": after.note_id,
                "task_id": after.task_id,
                "event_type": event_type,
                "actor_type": "user",
                "actor_id": dataset.owner_user_id,
                "source_device_id": device.device_id,
                "client_occurred_at": occurred_at,
                "source_kind": "client",
                "corrects_activity_id": None,
                "old_value": old_value,
                "new_value": new_value,
                "metadata": {},
            },
            owner_user_id=dataset.owner_user_id,
            bound_actor_type="user",
            bound_actor_id=dataset.owner_user_id,
            authenticated_device_id=device.device_id,
            trusted_server_origin=False,
        )
        activity_wire = activity_payload.model_dump(mode="json")
        mutation_group_id = _notes_task_client_group_id(
            dataset.dataset_id,
            device.device_id,
            envelope.client_envelope_id,
        )
        placeholder_plan_hash = "0" * 64
        note_step: SyncEnvelopeCreate | None = None
        projection_anchor: dict[str, object] | None = None
        from .notes_task_coordinator import _projection_anchor_from_envelope

        projects_new_task = prior_head is None and envelope.operation == "upsert"
        if projects_new_task or prior_head is not None:
            from tldw_Server_API.app.core.Notes_Tasks.markdown_parser import (
                parse_note_checklists,
            )
            from tldw_Server_API.app.core.Notes_Tasks.projection_markers import (
                TaskMarker,
                task_marker_hash,
            )

            from .notes_task_coordinator import (
                TASK_PROJECTION_ROUTING_KEY,
                TaskProjectionGroupMetadata,
                append_task_projection_to_note,
                project_task_payload_into_note,
                remove_task_projection_from_note,
            )
            from .server_origin import canonical_payload_hash

            base_anchor = (
                _projection_anchor_from_envelope(prior_head)
                if prior_head is not None
                else None
            )
            if projects_new_task or (base_anchor is not None and base_anchor.linked):
                note_head = self.store.get_current_head(
                    dataset.dataset_id,
                    "notes.note",
                    after.note_id,
                )
                if note_head is None or note_head.object_revision is None:
                    raise SyncStoreError("notes_task_projection_base_invalid")
                note_wire = dict(note_head.payload)
                current_items = parse_note_checklists(
                    note_id=after.note_id,
                    note_version=int(note_head.object_revision),
                    content=str(note_wire.get("content") or ""),
                ).items
                if projects_new_task or restore_intent:
                    marker_base_is_valid = not any(
                        item.marker is not None
                        and item.marker.task_id == after.task_id
                        for item in current_items
                    )
                else:
                    if prior_head is None:
                        raise SyncStoreError("notes_task_projection_base_invalid")
                    expected_marker = TaskMarker(
                        task_id=after.task_id,
                        revision=int(prior_head.object_revision or 0),
                        object_hash=str(prior_head.payload_hash or ""),
                    )
                    task_markers = [
                        item.marker
                        for item in current_items
                        if item.marker is not None
                        and item.marker.task_id == after.task_id
                    ]
                    marker_base_is_valid = task_markers == [expected_marker]
                if not marker_base_is_valid:
                    raise SyncStoreError("notes_task_projection_base_invalid")
                projection_kwargs = {
                    "content": str(note_wire.get("content") or ""),
                    "note_id": after.note_id,
                    "note_revision": int(note_head.object_revision),
                    "task_id": after.task_id,
                    "task_revision": int(envelope.object_revision or 0),
                    "task_hash": str(envelope.payload_hash or ""),
                    "payload": envelope.payload,
                }
                if envelope.operation == "tombstone":
                    if prior_head is None:
                        raise SyncStoreError("notes_task_projection_base_invalid")
                    note_wire["content"] = remove_task_projection_from_note(
                        **projection_kwargs,
                        base_revision=int(prior_head.object_revision or 0),
                        base_hash=str(prior_head.payload_hash or ""),
                    )
                elif projects_new_task or restore_intent:
                    note_wire["content"] = append_task_projection_to_note(
                        **projection_kwargs,
                    )
                else:
                    if prior_head is None:
                        raise SyncStoreError("notes_task_projection_base_invalid")
                    note_wire["content"] = project_task_payload_into_note(
                        **projection_kwargs,
                        base_revision=int(prior_head.object_revision or 0),
                        base_hash=str(prior_head.payload_hash or ""),
                    )
                note_hash, note_size = canonical_payload_hash(note_wire)
                note_envelope_id = (
                    f"notes-task-note-client-{activity_id.replace('-', '')}"
                )
                marker = TaskMarker(
                    task_id=after.task_id,
                    revision=int(envelope.object_revision or 0),
                    object_hash=str(envelope.payload_hash or ""),
                )
                projection_anchor = TaskProjectionGroupMetadata(
                    projection_version=1,
                    task_id=after.task_id,
                    task_envelope_id=envelope.client_envelope_id,
                    task_revision=marker.revision,
                    task_hash=marker.object_hash,
                    note_envelope_id=note_envelope_id,
                    note_hash=note_hash,
                    linked=True,
                    marker_hash=task_marker_hash(marker),
                ).as_routing_value()
                note_step = SyncEnvelopeCreate(
                    dataset_id=dataset.dataset_id,
                    client_envelope_id=note_envelope_id,
                    domain="notes.note",
                    operation="upsert",
                    object_id=after.note_id,
                    device_id=device.device_id,
                    base_server_cursor=note_head.server_cursor,
                    base_object_revision=note_head.object_revision,
                    base_object_hash=note_head.payload_hash,
                    object_revision=int(note_head.object_revision) + 1,
                    payload=note_wire,
                    payload_hash=note_hash,
                    payload_size_bytes=note_size,
                    created_at_client=occurred_at,
                    deleted=False,
                    encryption_metadata=dict(envelope.encryption_metadata),
                    status="accepted",
                    mutation_group_id=mutation_group_id,
                    mutation_step=2,
                    mutation_step_count=3,
                    mutation_plan_hash=placeholder_plan_hash,
                )
        step_count = 3 if note_step is not None else 2
        task_routing = dict(envelope.routing_metadata)
        activity_routing: dict[str, object] = {}
        if projection_anchor is not None:
            task_routing[TASK_PROJECTION_ROUTING_KEY] = projection_anchor
            activity_routing[TASK_PROJECTION_ROUTING_KEY] = projection_anchor
        task_step = replace(
            envelope,
            device_id=device.device_id,
            status="accepted",
            routing_metadata=task_routing,
            mutation_group_id=mutation_group_id,
            mutation_step=0,
            mutation_step_count=step_count,
            mutation_plan_hash=placeholder_plan_hash,
        )
        activity_step = SyncEnvelopeCreate(
            dataset_id=dataset.dataset_id,
            client_envelope_id=f"notes-task-activity-client-{activity_id.replace('-', '')}",
            domain="notes.task_activity",
            operation="upsert",
            object_id=activity_id,
            device_id=device.device_id,
            object_revision=1,
            entity_version=1,
            parent_id=after.note_id,
            schema_version=1,
            adapter_version=1,
            payload=activity_wire,
            payload_hash=notes_task_activity_object_hash(
                activity_payload,
                revision=1,
                deleted=False,
            ),
            payload_size_bytes=len(
                json.dumps(
                    activity_wire,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
            created_at_client=occurred_at,
            deleted=False,
            encryption_metadata=dict(envelope.encryption_metadata),
            routing_metadata=activity_routing,
            status="accepted",
            mutation_group_id=mutation_group_id,
            mutation_step=1,
            mutation_step_count=step_count,
            mutation_plan_hash=placeholder_plan_hash,
        )
        plan = (
            (task_step, activity_step, note_step)
            if note_step is not None
            else (task_step, activity_step)
        )
        plan_hash = mutation_group_plan_hash(plan)
        return tuple(
            replace(step, mutation_plan_hash=plan_hash) for step in plan
        )

    def _task_client_adapter_context(
        self,
        *,
        dataset: SyncDataset,
        device: SyncDevice,
        planned_task: SyncEnvelopeCreate | None = None,
        derived_activity: bool = False,
        derived_projection: bool = False,
    ) -> SyncAdapterContext:
        """Build the authenticated overlay used to preflight a task group."""

        def get_head(domain: SyncDomain, object_id: str):
            if (
                planned_task is not None
                and domain == "notes.task"
                and object_id == planned_task.object_id
            ):
                return planned_task
            return self.store.get_current_head(dataset.dataset_id, domain, object_id)

        return SyncAdapterContext(
            get_head=get_head,
            get_authorized_note=lambda note_id: get_head("notes.note", note_id),
            get_authorized_task=lambda task_id: get_head("notes.task", task_id),
            list_heads=lambda domain: self._list_current_heads_for_adapter(
                dataset.dataset_id,
                domain,
            ),
            authenticated_actor_type="user",
            authenticated_actor_id=dataset.owner_user_id,
            authenticated_device_id=device.device_id,
            coordinator_derived_task_activity=derived_activity,
            coordinator_derived_task_projection=derived_projection,
        )

    def _repair_task_client_projection_cache(
        self,
        *,
        dataset: SyncDataset,
        envelopes: Sequence[SyncEnvelope],
    ) -> None:
        """Rebuild the disposable locator after a linked client group applies."""

        if len(envelopes) != 3 or envelopes[0].operation == "tombstone":
            return
        task_materializer = self.materializers.get("notes.task")
        note_db = getattr(task_materializer, "note_db", None)
        if note_db is None:
            raise SyncStoreError("notes_task_projection_cache_unavailable")
        note = note_db.get_note_by_id(envelopes[2].object_id)
        if note is None:
            raise SyncStoreError("notes_task_projection_cache_unavailable")
        from tldw_Server_API.app.core.Notes_Tasks.markdown_parser import (
            parse_note_checklists,
        )

        from .notes_task_coordinator import rebuild_task_projection_cache

        matches = [
            item
            for item in parse_note_checklists(
                note_id=str(note["id"]),
                note_version=int(note["version"]),
                content=str(note.get("content") or ""),
            ).items
            if item.marker is not None
            and item.marker.task_id == envelopes[0].object_id
        ]
        if len(matches) != 1:
            raise SyncStoreError("notes_task_projection_cache_unavailable")
        rebuilt = rebuild_task_projection_cache(
            task_store=note_db.task_store,
            sync_store=self.store,
            owner_user_id=dataset.owner_user_id,
            dataset_id=dataset.dataset_id,
            note_id=str(note["id"]),
            item=matches[0],
        )
        if rebuilt.projection is None:
            raise SyncStoreError(
                rebuilt.reason_code or "notes_task_projection_cache_unavailable"
            )

    def _push_task_client_group(
        self,
        *,
        dataset: SyncDataset,
        device: SyncDevice,
        envelope: SyncEnvelopeCreate,
    ) -> SyncPushResult:
        """Preflight, atomically append, and fully materialize one task group."""

        group_id = _notes_task_client_group_id(
            dataset.dataset_id,
            device.device_id,
            envelope.client_envelope_id,
        )
        inserted = self.store.list_mutation_group(dataset.dataset_id, group_id)
        if inserted:
            try:
                validate_stored_mutation_group(
                    inserted,
                    dataset_id=dataset.dataset_id,
                    mutation_group_id=group_id,
                )
            except StoredMutationGroupValidationError:
                return self._task_client_idempotency_rejection(dataset, envelope)
            if (
                len(inserted) not in {2, 3}
                or [item.domain for item in inserted]
                not in (
                    ["notes.task", "notes.task_activity"],
                    ["notes.task", "notes.task_activity", "notes.note"],
                )
                or not _same_client_task_submission(inserted[0], envelope)
            ):
                return self._task_client_idempotency_rejection(dataset, envelope)
        else:
            try:
                plan = self._expand_task_client_push(
                    dataset=dataset,
                    device=device,
                    envelope=envelope,
                )
            except Exception as exc:  # noqa: BLE001 - contract errors are sanitized.
                return SyncPushResult(
                    dataset_id=dataset.dataset_id,
                    rejected=[
                        SyncPushRejected(
                            client_envelope_id=envelope.client_envelope_id,
                            error_code="notes_task_payload_invalid",
                            message="notes.task compound mutation validation failed",
                            retryable=isinstance(exc, SyncMaterializationBusyError),
                        )
                    ],
                )

            task_outcome = self._evaluate_envelope(
                dataset,
                plan[0],
                context=self._task_client_adapter_context(
                    dataset=dataset,
                    device=device,
                    derived_projection=len(plan) == 3,
                ),
            )
            if not isinstance(task_outcome, AdapterAccepted):
                return self._task_client_outcome_result(
                    dataset=dataset,
                    envelope=plan[0],
                    outcome=task_outcome,
                )
            activity_outcome = self._evaluate_envelope(
                dataset,
                plan[1],
                context=self._task_client_adapter_context(
                    dataset=dataset,
                    device=device,
                    planned_task=plan[0],
                    derived_activity=True,
                    derived_projection=len(plan) == 3,
                ),
            )
            if not isinstance(activity_outcome, AdapterAccepted):
                return self._task_client_outcome_result(
                    dataset=dataset,
                    envelope=plan[0],
                    outcome=activity_outcome,
                )
            if len(plan) == 3:
                note_outcome = self._evaluate_envelope(
                    dataset,
                    plan[2],
                    context=self._task_client_adapter_context(
                        dataset=dataset,
                        device=device,
                        planned_task=plan[0],
                    ),
                )
                if not isinstance(note_outcome, AdapterAccepted):
                    return self._task_client_outcome_result(
                        dataset=dataset,
                        envelope=plan[0],
                        outcome=note_outcome,
                    )
            try:
                inserted = self.store.insert_envelopes_atomic(plan)
            except SyncIdempotencyConflictError:
                return self._task_client_idempotency_rejection(dataset, envelope)
            except SyncHeadConflictError:
                outcome = AdapterConflict(
                    client_envelope_id=envelope.client_envelope_id,
                    domain="notes.task",
                    entity_id=envelope.object_id,
                    conflict_type="stale_base_state",
                    message="Sync object changed after request preflight",
                )
                return self._task_client_outcome_result(
                    dataset=dataset,
                    envelope=plan[0],
                    outcome=outcome,
                )

        try:
            from .server_origin_batch import materialize_accepted_mutation_group

            materialized = materialize_accepted_mutation_group(
                service=self,
                dataset=dataset,
                envelopes=inserted,
            )
            self._repair_task_client_projection_cache(
                dataset=dataset,
                envelopes=materialized.envelopes,
            )
        except Exception as exc:  # noqa: BLE001 - accepted groups remain replayable.
            logger.warning(
                "Task compound projection remains incomplete for {}: {}",
                envelope.client_envelope_id,
                str(exc) if isinstance(exc, SyncStoreError) else type(exc).__name__,
            )
            return SyncPushResult(
                dataset_id=dataset.dataset_id,
                rejected=[
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="sync_projection_failed",
                        message="Task projection is incomplete; retry the same envelope",
                        retryable=True,
                    )
                ],
            )

        task = materialized.envelopes[0]
        accepted = self._push_accepted_from_envelope(task)
        return SyncPushResult(
            dataset_id=dataset.dataset_id,
            accepted=[accepted],
            next_cursor=str(accepted.server_sequence),
        )

    @staticmethod
    def _task_client_idempotency_rejection(
        dataset: SyncDataset,
        envelope: SyncEnvelopeCreate,
    ) -> SyncPushResult:
        """Return the stable changed-envelope-ID rejection."""

        return SyncPushResult(
            dataset_id=dataset.dataset_id,
            rejected=[
                SyncPushRejected(
                    client_envelope_id=envelope.client_envelope_id,
                    error_code="idempotency_conflict",
                    message="Sync envelope ID was reused with different content",
                )
            ],
        )

    def _task_client_outcome_result(
        self,
        *,
        dataset: SyncDataset,
        envelope: SyncEnvelopeCreate,
        outcome: AdapterRejected | AdapterConflict | AdapterDeferred,
    ) -> SyncPushResult:
        """Translate one closed preflight outcome to the public push shape."""

        if isinstance(outcome, AdapterConflict):
            try:
                conflict = self._store_preflight_conflict(dataset, envelope, outcome)
            except SyncStoreError:
                return SyncPushResult(
                    dataset_id=dataset.dataset_id,
                    rejected=[
                        SyncPushRejected(
                            client_envelope_id=envelope.client_envelope_id,
                            error_code="sync_projection_busy",
                            message="Projection is busy; retry later",
                            retryable=True,
                        )
                    ],
                )
            return SyncPushResult(dataset_id=dataset.dataset_id, conflicts=[conflict])
        if isinstance(outcome, AdapterDeferred):
            return SyncPushResult(
                dataset_id=dataset.dataset_id,
                rejected=[
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="adapter_deferred",
                        message=outcome.message,
                        retryable=True,
                    )
                ],
            )
        return SyncPushResult(
            dataset_id=dataset.dataset_id,
            rejected=[
                SyncPushRejected(
                    client_envelope_id=envelope.client_envelope_id,
                    error_code=outcome.error_code,
                    message=outcome.message,
                    retryable=outcome.retryable,
                )
            ],
        )

    def push(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        envelopes: Sequence[SyncEnvelopeCreate],
        base_server_cursor: int | None = None,
        stop_on_conflict: bool = False,
    ) -> SyncPushResult:
        # The top-level cursor is a client dataset checkpoint; object conflict checks use envelope bases.
        _ = base_server_cursor
        if device_id == _SERVER_ORIGIN_DEVICE_ID:
            return SyncPushResult(
                dataset_id=dataset_id,
                rejected=[
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="reserved_device_id",
                        message="Sync device identifier is reserved for server use",
                    )
                    for envelope in envelopes
                ],
            )
        device = self._require_registered_device(user_id, device_id)
        try:
            dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        except SyncStoreError:
            return SyncPushResult(
                dataset_id=dataset_id,
                rejected=[
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="dataset_not_found_or_forbidden",
                        message="Sync dataset was not found or is not accessible",
                    )
                    for envelope in envelopes
                ],
            )

        accepted: list[SyncPushAccepted] = []
        rejected: list[SyncPushRejected] = []
        conflicts: list[SyncPushConflict] = []
        stopped_after_conflict = False

        for index, envelope in enumerate(envelopes):
            if stopped_after_conflict:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="stopped_after_conflict",
                        message="Sync push stopped after a previous envelope conflicted.",
                        retryable=True,
                    )
                )
                continue
            if index >= self.settings.max_batch_size:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="batch_limit_exceeded",
                        message="Sync push batch exceeded the server envelope limit",
                    )
                )
                continue
            if envelope.dataset_id != dataset_id:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="dataset_mismatch",
                        message="Envelope dataset_id must match the push dataset_id",
                    )
                )
                continue
            if envelope.device_id is not None and envelope.device_id != device_id:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="device_mismatch",
                        message="Envelope device_id must match the authenticated push device_id",
                    )
                )
                continue
            if envelope.domain not in dataset.domains:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="domain_not_enrolled",
                        message=f"Sync domain is not enrolled for this dataset: {envelope.domain}",
                    )
                )
                continue
            if envelope.domain in PERSONAL_CONTEXT_SYNC_DOMAINS and not _personal_context_link_is_complete(
                self.store, dataset, user_id=user_id, device_id=device_id
            ):
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="personal_context_link_incomplete",
                        message="Personal Context reconciliation is not complete",
                    )
                )
                continue
            if has_guard_required_routing_key(envelope.routing_metadata):
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="reserved_routing_metadata",
                        message="Sync envelope contains reserved routing metadata",
                    )
                )
                continue
            if (
                envelope.domain in NOTES_TASK_SYNC_DOMAINS
                and not self._notes_task_domains_ready(dataset)
            ):
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="notes_task_sync_not_ready",
                        message="Notes task Sync is not ready for this dataset",
                        retryable=True,
                    )
                )
                continue
            if self._payload_exceeds_size_limit(envelope):
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="payload_too_large",
                        message="Sync envelope payload exceeds the server size limit",
                    )
                )
                continue
            if (
                self.adapters.has_domain(envelope.domain)
                and self.adapters.supports_version(
                    envelope.domain,
                    envelope.adapter_version,
                )
                and not _device_supports_adapter_version(
                    device,
                    envelope.domain,
                    envelope.adapter_version,
                )
            ):
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="device_adapter_version_not_advertised",
                        message=(
                            "Sync device did not advertise this adapter version "
                            f"for {envelope.domain}"
                        ),
                    )
                )
                continue
            envelope = replace(envelope, device_id=envelope.device_id or device_id)
            if envelope.domain == "notes.task":
                task_result = self._push_task_client_group(
                    dataset=dataset,
                    device=device,
                    envelope=envelope,
                )
                accepted.extend(task_result.accepted)
                rejected.extend(task_result.rejected)
                conflicts.extend(task_result.conflicts)
                if stop_on_conflict and task_result.conflicts:
                    stopped_after_conflict = True
                continue
            if envelope.domain == "notes.task_activity" and (
                envelope.operation != "upsert"
                or envelope.payload.get("event_type") != "corrected"
                or not isinstance(
                    envelope.payload.get("corrects_activity_id"),
                    str,
                )
            ):
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="notes_task_activity_origin_invalid",
                        message=(
                            "Clients may append only exact authorized task "
                            "activity corrections"
                        ),
                    )
                )
                continue
            try:
                existing = self.store.get_existing_envelope_for_idempotency(
                    replace(envelope, status="accepted")
                )
            except SyncIdempotencyConflictError:
                try:
                    existing = self.store.get_existing_envelope_for_idempotency(
                        replace(envelope, status="conflict")
                    )
                except SyncIdempotencyConflictError:
                    rejected.append(
                        SyncPushRejected(
                            client_envelope_id=envelope.client_envelope_id,
                            error_code="idempotency_conflict",
                            message="Sync envelope ID was reused with different content",
                        )
                    )
                    continue
            if (
                existing is not None
                and existing.status == "accepted"
                and existing.apply_status in {"applied", "superseded"}
            ):
                accepted.append(self._push_accepted_from_envelope(existing))
                continue
            try:
                outcome = self._evaluate_envelope(
                    dataset,
                    envelope,
                    context=(
                        self._task_client_adapter_context(
                            dataset=dataset,
                            device=device,
                        )
                        if envelope.domain == "notes.task_activity"
                        else None
                    ),
                )
            except KeyError:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="unknown_domain",
                        message=f"Sync domain is not registered: {envelope.domain}",
                    )
                )
                continue
            except PrivatePayloadValidationError as exc:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="private_payload_validation_failed",
                        message=str(exc),
                    )
                )
                continue

            if isinstance(outcome, AdapterRejected):
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=outcome.client_envelope_id,
                        error_code=outcome.error_code,
                        message=outcome.message,
                        retryable=outcome.retryable,
                    )
                )
                continue
            if isinstance(outcome, AdapterDeferred):
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=outcome.client_envelope_id,
                        error_code="adapter_deferred",
                        message=outcome.message,
                        retryable=True,
                    )
                )
                continue
            if isinstance(outcome, AdapterConflict):
                try:
                    conflicts.append(
                        self._store_preflight_conflict(dataset, envelope, outcome)
                    )
                except SyncIdempotencyConflictError:
                    rejected.append(
                        SyncPushRejected(
                            client_envelope_id=envelope.client_envelope_id,
                            error_code="idempotency_conflict",
                            message="Sync envelope ID was reused with different content",
                        )
                    )
                except SyncMaterializationBusyError:
                    rejected.append(
                        SyncPushRejected(
                            client_envelope_id=envelope.client_envelope_id,
                            error_code="sync_projection_busy",
                            message="Projection is busy; retry later",
                            retryable=True,
                        )
                    )
                except PersonalContextStorageEncryptionUnavailableError:
                    rejected.append(
                        SyncPushRejected(
                            client_envelope_id=envelope.client_envelope_id,
                            error_code="personal_context_storage_unavailable",
                            message=(
                                "Personal Context storage encryption is unavailable"
                            ),
                            retryable=True,
                        )
                    )
                if stop_on_conflict:
                    stopped_after_conflict = True
                continue

            try:
                storage_envelope = self._protect_personal_context_for_storage(
                    dataset,
                    replace(envelope, status="accepted"),
                )
            except PersonalContextStorageEncryptionUnavailableError:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="personal_context_storage_unavailable",
                        message="Personal Context storage encryption is unavailable",
                        retryable=True,
                    )
                )
                continue

            try:
                inserted = self.store.insert_envelope(storage_envelope)
            except SyncInvalidDomainError:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="domain_not_enrolled",
                        message=f"Sync domain is not enrolled for this dataset: {envelope.domain}",
                    )
                )
                continue
            except SyncIdempotencyConflictError:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="idempotency_conflict",
                        message="Sync envelope ID was reused with different content",
                    )
                )
                continue
            except SyncHeadConflictError:
                outcome = AdapterConflict(
                    client_envelope_id=envelope.client_envelope_id,
                    domain=envelope.domain,
                    entity_id=envelope.object_id,
                    conflict_type="stale_base_state",
                    message="Sync object changed after request preflight",
                )
                try:
                    conflicts.append(
                        self._store_preflight_conflict(dataset, envelope, outcome)
                    )
                except SyncIdempotencyConflictError:
                    rejected.append(
                        SyncPushRejected(
                            client_envelope_id=envelope.client_envelope_id,
                            error_code="idempotency_conflict",
                            message="Sync envelope ID was reused with different content",
                        )
                    )
                except SyncMaterializationBusyError:
                    rejected.append(
                        SyncPushRejected(
                            client_envelope_id=envelope.client_envelope_id,
                            error_code="sync_projection_busy",
                            message="Projection is busy; retry later",
                            retryable=True,
                        )
                    )
                except PersonalContextStorageEncryptionUnavailableError:
                    rejected.append(
                        SyncPushRejected(
                            client_envelope_id=envelope.client_envelope_id,
                            error_code="personal_context_storage_unavailable",
                            message=(
                                "Personal Context storage encryption is unavailable"
                            ),
                            retryable=True,
                        )
                    )
                if stop_on_conflict:
                    stopped_after_conflict = True
                continue
            except SyncMaterializationPredecessorError as exc:
                if exc.conflict_id and exc.domain and exc.entity_id:
                    conflicts.append(
                        SyncPushConflict(
                            conflict_id=exc.conflict_id,
                            client_envelope_id=envelope.client_envelope_id,
                            domain=exc.domain,  # type: ignore[arg-type]
                            entity_id=exc.entity_id,
                            server_sequence=exc.server_sequence,
                            message="An unresolved materialization conflict must be resolved before appending more changes",
                        )
                    )
                else:
                    rejected.append(
                        SyncPushRejected(
                            client_envelope_id=envelope.client_envelope_id,
                            error_code="sync_projection_predecessor_unresolved",
                            message="An unresolved materialization conflict must be resolved before appending more changes",
                            retryable=True,
                        )
                    )
                if stop_on_conflict:
                    stopped_after_conflict = True
                continue
            if inserted.apply_status not in {"applied", "superseded"}:
                materialization = self._materialize_envelope(inserted)
                inserted = self._envelope_snapshot(inserted)
                if materialization.status == "conflict":
                    conflicts.append(
                        self._store_materialization_conflict(
                            dataset,
                            inserted,
                            materialization,
                        )
                    )
                    if stop_on_conflict:
                        stopped_after_conflict = True
                    continue
            accepted.append(self._push_accepted_from_envelope(inserted))

        sequences = [item.server_sequence for item in accepted]
        sequences.extend(item.server_sequence for item in conflicts if item.server_sequence is not None)
        next_sequence = max(sequences, default=None)
        return SyncPushResult(
            dataset_id=dataset_id,
            accepted=accepted,
            rejected=rejected,
            conflicts=conflicts,
            next_cursor=str(next_sequence) if next_sequence is not None else None,
        )

    def pull(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        cursor: str | int | None = None,
        domains: Sequence[SyncDomain] | None = None,
        page_size: int | None = None,
        include_own_changes: bool = False,
    ) -> SyncPullResult:
        device = self._require_registered_device(user_id, device_id)
        if page_size is not None and page_size < 1:
            raise SyncStoreError("Sync pull page_size must be greater than zero")
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)

        selected_domains = self._selected_pull_domains(dataset, device, domains)
        streams = self._pull_adapter_streams(device, selected_domains)
        if any(adapter_version != 1 for _domain, adapter_version in streams) or (
            isinstance(cursor, str) and "." in cursor
        ):
            return self._pull_versioned(
                dataset=dataset,
                device=device,
                cursor=cursor,
                streams=streams,
                page_size=page_size,
                include_own_changes=include_own_changes,
            )
        since_sequence = self._resolve_cursor(dataset_id, device_id, cursor, selected_domains)
        page_limit = min(page_size or self.settings.max_pull_page_size, self.settings.max_pull_page_size)
        raw_envelopes, visible = self._scan_pull_page(
            dataset_id=dataset_id,
            device_id=device_id,
            since_sequence=since_sequence,
            domains=selected_domains,
            page_limit=page_limit,
            include_own_changes=include_own_changes,
            adapter_versions=[1],
        )

        page = [
            self._restore_personal_context_from_storage(dataset, envelope)
            for envelope in visible[:page_limit]
        ]
        has_visible_lookahead = len(visible) > page_limit
        has_more = has_visible_lookahead or len(raw_envelopes) > page_limit
        if has_visible_lookahead and page:
            next_sequence = page[-1].server_sequence
        else:
            next_sequence = max(
                (envelope.server_sequence for envelope in raw_envelopes),
                default=since_sequence,
            )
        if raw_envelopes:
            self._update_cursors(
                dataset_id,
                device_id,
                selected_domains,
                next_sequence if cursor is None else None,
                delivered=page,
            )
        return SyncPullResult(
            dataset_id=dataset_id,
            encryption_policy=dataset.encryption_policy,
            envelopes=page,
            next_cursor=str(next_sequence),
            has_more=has_more,
        )

    def restore_manifest(
        self,
        *,
        user_id: str,
        device_id: str | None = None,
        dataset_ids: Sequence[str] | None = None,
        domains: Sequence[SyncDomain] | None = None,
    ) -> SyncRestoreManifest:
        if device_id is not None:
            self._require_registered_device(user_id, device_id)
        selected_domains = set(domains or [])
        datasets = self._accessible_datasets(user_id=user_id, dataset_ids=dataset_ids)
        devices = [
            SyncRestoreManifestDevice(
                device_id=device.device_id,
                display_name=device.display_name,
                client_type=device.client_type,
                client_version=device.client_version,
                last_seen_at=device.last_seen_at,
                revoked_at=device.revoked_at,
            )
            for device in self.store.list_devices_for_user(user_id)
        ]

        manifest_datasets = [
            self._manifest_dataset(dataset, user_id=user_id, domains=selected_domains) for dataset in datasets
        ]
        return SyncRestoreManifest(
            datasets=manifest_datasets,
            devices=devices,
            generated_at=self.clock() or None,
            filters_applied={
                "dataset_ids": list(dataset_ids or []),
                "domains": list(domains or []),
            },
        )

    def restore_preview(
        self,
        *,
        user_id: str,
        device_id: str | None = None,
        dataset_ids: Sequence[str] | None = None,
        domains: Sequence[SyncDomain] | None = None,
        selected_object_ids: Sequence[str] | None = None,
        selected_attachment_ids: Sequence[str] | None = None,
        metadata_only: bool = False,
        local_inventory: Sequence[Mapping[str, object]] | None = None,
        attachment_availability: Mapping[str, str] | None = None,
    ) -> SyncRestorePreview:
        """Return metadata needed to preview a restore plan for Sync v2 M1."""

        restore_device = (
            self._require_registered_device(user_id, device_id)
            if device_id is not None
            else None
        )
        selected_domains = set(domains or [])
        selected_object_id_set = _normalize_selection_set(selected_object_ids)
        selected_attachment_id_set = _normalize_selection_set(selected_attachment_ids)
        local_index = build_local_inventory_index(local_inventory)
        datasets = self._accessible_datasets(user_id=user_id, dataset_ids=dataset_ids)
        if self.settings.restore_preview_candidate_limit < 1:
            raise SyncStoreError("sync_restore_candidate_limit_invalid")
        if self.settings.restore_preview_action_limit < 1:
            raise SyncStoreError("sync_restore_action_limit_invalid")

        preview_datasets: list[SyncRestorePreviewDataset] = []
        ordered_actions: list[SyncRestoreOrderedAction] = []
        safe_applies: list[SyncRestorePreviewObject] = []
        object_conflicts: list[SyncRestorePreviewObjectConflict] = []
        tombstones: list[SyncRestorePreviewObject] = []
        attachment_refs: list[SyncRestorePreviewAttachmentRef] = []
        missing_blobs: list[SyncRestorePreviewAttachmentRef] = []
        blob_details: list[SyncRestoreBlobCompleteness] = []
        envelope_ranges: list[SyncRestorePreviewEnvelopeRange] = []
        total_counts: dict[str, int] = {}
        key_status: dict[str, dict[str, bool]] = {}
        warnings: list[SyncRestorePreviewWarning] = []
        planned_inventory_keys: set[tuple[str, SyncDomain, str, int]] = set()
        candidate_count = 0
        planned_action_count = 0

        for dataset in datasets:
            dataset_domains = [
                domain for domain in dataset.domains if not selected_domains or domain in selected_domains
            ]
            adapter_versions_by_domain = {
                domain: self._restore_adapter_versions(restore_device, domain)
                for domain in dataset_domains
            }
            stats = self.store.summarize_restore_manifest_dataset(
                dataset.dataset_id,
                user_id=user_id,
                domains=dataset_domains,
            )
            key_status[dataset.dataset_id] = {
                "key_recovery_available": stats.key_recovery_available,
            }
            if not stats.key_recovery_available:
                warnings.append(
                    SyncRestorePreviewWarning(
                        code="sync_key_recovery_missing",
                        message=(
                            "No active Sync v2 key recovery bundle is available "
                            "for this dataset."
                        ),
                        dataset_id=dataset.dataset_id,
                    )
                )
            domain_envelopes: dict[SyncDomain, list[SyncEnvelope]] = {}
            for domain in dataset_domains:
                envelopes = self._list_restore_preview_domain_envelopes(
                    dataset_id=dataset.dataset_id,
                    domain=domain,
                    adapter_versions=adapter_versions_by_domain[domain],
                    max_candidates=(
                        self.settings.restore_preview_candidate_limit
                        - candidate_count
                    ),
                )
                candidate_count += len(envelopes)
                domain_envelopes[domain] = envelopes
            approximate_counts = dict(stats.approximate_counts)
            byte_estimates = dict(stats.byte_estimates)
            if "attachment.ref" in domain_envelopes:
                attachment_envelopes = domain_envelopes["attachment.ref"]
                approximate_counts["attachment.ref"] = len(attachment_envelopes)
                byte_estimates["attachment.ref"] = sum(
                    envelope.payload_size_bytes or 0
                    for envelope in attachment_envelopes
                )
            approximate_counts = {
                domain: count
                for domain, count in approximate_counts.items()
                if domain in dataset_domains and count
            }
            byte_estimates = {
                domain: count
                for domain, count in byte_estimates.items()
                if domain in dataset_domains and count
            }
            for domain, count in approximate_counts.items():
                total_counts[domain] = total_counts.get(domain, 0) + count
            latest_cursors: dict[str, int] = {}
            dataset_ranges: list[SyncRestorePreviewEnvelopeRange] = []
            for domain, envelopes in domain_envelopes.items():
                cursors = [envelope.server_cursor or 0 for envelope in envelopes if envelope.server_cursor]
                if not cursors:
                    continue
                latest_cursors[domain] = max(cursors)
                dataset_range = SyncRestorePreviewEnvelopeRange(
                    dataset_id=dataset.dataset_id,
                    domain=domain,
                    from_cursor=min(cursors),
                    to_cursor=max(cursors),
                    envelope_count=len(cursors),
                )
                dataset_ranges.append(dataset_range)
                envelope_ranges.append(dataset_range)
            latest_cursor = max(latest_cursors.values(), default=None)
            preview_datasets.append(
                SyncRestorePreviewDataset(
                    dataset_id=dataset.dataset_id,
                    domains=dataset_domains,
                    approximate_counts=approximate_counts,
                    byte_estimates=byte_estimates,
                    latest_cursor=latest_cursor,
                    latest_cursors=latest_cursors,
                    envelope_ranges=dataset_ranges,
                    total_count=sum(approximate_counts.values()),
                    encryption_policy=dataset.encryption_policy,
                    key_recovery_available=stats.key_recovery_available,
                )
            )

            latest_object_envelopes: dict[tuple[SyncDomain, str], SyncEnvelope] = {}
            for domain in dataset_domains:
                if domain not in OBJECT_RESTORE_DOMAINS:
                    continue
                for envelope in domain_envelopes.get(domain, []):
                    if domain == "attachment.ref" and envelope.adapter_version != 2:
                        continue
                    if envelope.apply_status == "superseded":
                        continue
                    latest_object_envelopes[(domain, envelope.object_id)] = envelope
            selected_envelopes = [
                envelope
                for envelope in latest_object_envelopes.values()
                if not selected_object_id_set or envelope.object_id in selected_object_id_set
            ]
            try:
                restore_envelopes = self._expand_restore_mutation_groups(
                    dataset_id=dataset.dataset_id,
                    envelopes=selected_envelopes,
                    latest_object_envelopes=latest_object_envelopes,
                    selected_domains=selected_domains,
                    selected_object_ids=selected_object_id_set,
                    adapter_versions_by_domain=adapter_versions_by_domain,
                )
                remaining_actions = (
                    self.settings.restore_preview_action_limit - planned_action_count
                )
                if len(restore_envelopes) > remaining_actions:
                    raise SyncStoreError("sync_restore_action_limit_exceeded")
                ordered_restore_envelopes = order_restore_envelopes(
                    restore_envelopes,
                    max_actions=remaining_actions,
                )
                planned_action_count += len(ordered_restore_envelopes)
            except (RestorePlanningError, StoredMutationGroupValidationError) as exc:
                raise SyncStoreError("sync_restore_plan_invalid") from exc
            restore_fingerprints: list[tuple[int | None, str | None, bool]] = []
            final_plan_index_by_identity: dict[tuple[SyncDomain, str], int] = {}
            for plan_index, envelope in enumerate(ordered_restore_envelopes):
                domain = envelope.domain
                object_id = envelope.object_id
                object_state = self.store.get_object_state(dataset.dataset_id, domain, object_id)
                server_revision = (
                    object_state.object_revision
                    if object_state is not None and object_state.latest_server_cursor == envelope.server_cursor
                    else envelope.object_revision
                )
                if server_revision is None and domain in {"notes.note", "chat.conversation"}:
                    server_revision = 1
                server_hash = (
                    object_state.object_hash
                    if object_state is not None and object_state.latest_server_cursor == envelope.server_cursor
                    else envelope.payload_hash
                )
                deleted = (
                    object_state.deleted
                    if object_state is not None and object_state.latest_server_cursor == envelope.server_cursor
                    else envelope.operation == "tombstone" or envelope.deleted
                )
                restore_fingerprints.append((server_revision, server_hash, deleted))
                final_plan_index_by_identity[(domain, object_id)] = plan_index

            initially_matching_final_keys: set[
                tuple[str, SyncDomain, str, int]
            ] = set()
            for (domain, object_id), plan_index in final_plan_index_by_identity.items():
                envelope = ordered_restore_envelopes[plan_index]
                local_item = find_local_inventory_item(
                    local_index,
                    dataset_id=dataset.dataset_id,
                    domain=domain,
                    object_id=object_id,
                    adapter_version=envelope.adapter_version,
                )
                server_revision, server_hash, deleted = restore_fingerprints[plan_index]
                if local_item is not None and local_inventory_matches(
                    local_item,
                    object_revision=server_revision,
                    object_hash=server_hash,
                    deleted=deleted,
                ):
                    initially_matching_final_keys.add(
                        (
                            dataset.dataset_id,
                            domain,
                            object_id,
                            envelope.adapter_version,
                        )
                    )

            for plan_index, envelope in enumerate(ordered_restore_envelopes):
                domain = envelope.domain
                object_id = envelope.object_id
                server_revision, server_hash, deleted = restore_fingerprints[plan_index]
                local_item = find_local_inventory_item(
                    local_index,
                    dataset_id=dataset.dataset_id,
                    domain=domain,
                    object_id=object_id,
                    adapter_version=envelope.adapter_version,
                )
                local_matches = (
                    local_item is not None
                    and local_inventory_matches(
                        local_item,
                        object_revision=server_revision,
                        object_hash=server_hash,
                        deleted=deleted,
                    )
                )
                inventory_key = (
                    dataset.dataset_id,
                    domain,
                    object_id,
                    envelope.adapter_version,
                )
                if envelope.apply_status == "conflict":
                    conflict_type = "stored_apply_conflict"
                    object_conflicts.append(
                        SyncRestorePreviewObjectConflict(
                            dataset_id=dataset.dataset_id,
                            domain=domain,
                            object_id=object_id,
                            conflict_type=conflict_type,
                            server_revision=server_revision,
                            server_hash=server_hash,
                            server_cursor=envelope.server_cursor,
                            server_deleted=deleted,
                            local_revision=(
                                local_item.object_revision
                                if local_item is not None
                                else None
                            ),
                            local_hash=(
                                local_item.object_hash
                                if local_item is not None
                                else None
                            ),
                            local_deleted=(
                                local_item.deleted if local_item is not None else False
                            ),
                            message="Stored restore candidate is blocked by an apply conflict.",
                        )
                    )
                    ordered_actions.append(
                        SyncRestoreOrderedAction(
                            plan_index=len(ordered_actions),
                            action="conflict",
                            dataset_id=envelope.dataset_id,
                            domain=domain,
                            object_id=object_id,
                            operation=envelope.operation,
                            server_cursor=envelope.server_cursor or 0,
                            adapter_version=envelope.adapter_version,
                            mutation_group_id=envelope.mutation_group_id,
                            mutation_step=envelope.mutation_step,
                            mutation_step_count=envelope.mutation_step_count,
                            code="sync_restore_stored_apply_conflict",
                        )
                    )
                    continue
                local_matches_tombstone_base = (
                    deleted
                    and local_item is not None
                    and (
                        envelope.base_object_revision is not None
                        or envelope.base_object_hash is not None
                    )
                    and local_inventory_matches(
                        local_item,
                        object_revision=envelope.base_object_revision,
                        object_hash=envelope.base_object_hash,
                        deleted=False,
                    )
                )
                can_apply = (
                    local_item is None
                    or local_matches
                    or local_matches_tombstone_base
                    or inventory_key in planned_inventory_keys
                    or (
                        inventory_key in initially_matching_final_keys
                        and plan_index
                        < final_plan_index_by_identity[(domain, object_id)]
                    )
                )
                if not can_apply:
                    conflict_type = (
                        "whole_object_conflict"
                        if domain in WHOLE_OBJECT_RESTORE_DOMAINS
                        else "stable_id_conflict"
                    )
                    object_conflicts.append(
                        SyncRestorePreviewObjectConflict(
                            dataset_id=dataset.dataset_id,
                            domain=domain,
                            object_id=object_id,
                            conflict_type=conflict_type,
                            server_revision=server_revision,
                            server_hash=server_hash,
                            server_cursor=envelope.server_cursor,
                            server_deleted=deleted,
                            local_revision=local_item.object_revision,
                            local_hash=local_item.object_hash,
                            local_deleted=local_item.deleted,
                            message="Local object differs from the server restore candidate.",
                        )
                    )
                    ordered_actions.append(
                        SyncRestoreOrderedAction(
                            plan_index=len(ordered_actions),
                            action="conflict",
                            dataset_id=envelope.dataset_id,
                            domain=domain,
                            object_id=object_id,
                            operation=envelope.operation,
                            server_cursor=envelope.server_cursor or 0,
                            adapter_version=envelope.adapter_version,
                            mutation_group_id=envelope.mutation_group_id,
                            mutation_step=envelope.mutation_step,
                            mutation_step_count=envelope.mutation_step_count,
                            code=conflict_type,
                        )
                    )
                    continue
                if deleted:
                    ordered_actions.append(
                        SyncRestoreOrderedAction(
                            plan_index=len(ordered_actions),
                            action="tombstone",
                            dataset_id=envelope.dataset_id,
                            domain=domain,
                            object_id=object_id,
                            operation=envelope.operation,
                            server_cursor=envelope.server_cursor or 0,
                            adapter_version=envelope.adapter_version,
                            mutation_group_id=envelope.mutation_group_id,
                            mutation_step=envelope.mutation_step,
                            mutation_step_count=envelope.mutation_step_count,
                        )
                    )
                    tombstones.append(
                        SyncRestorePreviewObject(
                            dataset_id=dataset.dataset_id,
                            domain=domain,
                            object_id=object_id,
                            action=restore_action_for_domain(
                                domain,
                                deleted=True,
                                local_present=local_item is not None,
                            ),
                            server_revision=server_revision,
                            server_hash=server_hash,
                            server_cursor=envelope.server_cursor,
                            local_revision=local_item.object_revision if local_item is not None else None,
                            local_hash=local_item.object_hash if local_item is not None else None,
                            local_deleted=local_item.deleted if local_item is not None else None,
                            deleted=True,
                            parent_id=envelope.parent_id,
                        )
                    )
                    local_index[inventory_key] = LocalRestoreInventoryItem(
                        dataset_id=dataset.dataset_id,
                        domain=domain,
                        object_id=object_id,
                        adapter_version=envelope.adapter_version,
                        object_revision=server_revision,
                        object_hash=server_hash,
                        deleted=True,
                    )
                    planned_inventory_keys.add(inventory_key)
                    continue
                if can_apply:
                    action = (
                        "noop"
                        if local_matches
                        else restore_action_for_domain(
                            domain,
                            deleted=False,
                            local_present=False,
                        )
                    )
                    ordered_actions.append(
                        SyncRestoreOrderedAction(
                            plan_index=len(ordered_actions),
                            action="noop" if action == "noop" else "apply",
                            dataset_id=envelope.dataset_id,
                            domain=domain,
                            object_id=object_id,
                            operation=envelope.operation,
                            server_cursor=envelope.server_cursor or 0,
                            adapter_version=envelope.adapter_version,
                            mutation_group_id=envelope.mutation_group_id,
                            mutation_step=envelope.mutation_step,
                            mutation_step_count=envelope.mutation_step_count,
                        )
                    )
                    safe_applies.append(
                        SyncRestorePreviewObject(
                            dataset_id=dataset.dataset_id,
                            domain=domain,
                            object_id=object_id,
                            action=action,
                            server_revision=server_revision,
                            server_hash=server_hash,
                            server_cursor=envelope.server_cursor,
                            local_revision=local_item.object_revision if local_item is not None else None,
                            local_hash=local_item.object_hash if local_item is not None else None,
                            local_deleted=local_item.deleted if local_item is not None else None,
                            deleted=False,
                            parent_id=envelope.parent_id,
                        )
                    )
                    local_index[inventory_key] = LocalRestoreInventoryItem(
                        dataset_id=dataset.dataset_id,
                        domain=domain,
                        object_id=object_id,
                        adapter_version=envelope.adapter_version,
                        object_revision=server_revision,
                        object_hash=server_hash,
                        deleted=False,
                    )
                    planned_inventory_keys.add(inventory_key)
                    continue

            seen_refs: set[tuple[str, str]] = set()
            latest_attachment_envelopes: dict[str, SyncEnvelope] = {}
            for envelope in domain_envelopes.get("attachment.ref", []):
                latest_attachment_envelopes[envelope.object_id] = envelope
            for envelope in sorted(latest_attachment_envelopes.values(), key=lambda item: item.server_cursor or 0):
                if envelope.operation == "tombstone" or envelope.deleted:
                    continue
                try:
                    metadata = extract_attachment_ref_metadata(envelope)
                except AttachmentRefValidationError:
                    continue
                if selected_attachment_id_set and (
                    metadata.attachment_id not in selected_attachment_id_set
                    and metadata.payload_hash not in selected_attachment_id_set
                ):
                    continue
                if (
                    not selected_attachment_id_set
                    and selected_object_id_set
                    and metadata.parent_object_id not in selected_object_id_set
                ):
                    continue
                ref_key = (metadata.attachment_id, metadata.payload_hash)
                if ref_key in seen_refs:
                    continue
                seen_refs.add(ref_key)
                server_blob = None
                server_availability = "metadata_only"
                if self.settings.supports_attachments:
                    blob_owner_user_id = self._blob_owner_user_id(
                        dataset=dataset,
                        user_id=user_id,
                    )
                    if envelope.adapter_version == 2:
                        binding = (
                            self.store.get_attachment_revision_binding(
                                dataset.dataset_id,
                                metadata.attachment_id,
                                envelope.object_revision,
                                owner_user_id=dataset.owner_user_id,
                            )
                            if envelope.object_revision is not None
                            else None
                        )
                        if binding is not None and binding.resolved_blob_id is not None:
                            candidate = self.store.get_blob_object(
                                dataset.dataset_id,
                                blob_id=binding.resolved_blob_id,
                                owner_user_id=blob_owner_user_id,
                                include_unavailable=True,
                            )
                            if (
                                candidate is not None
                                and candidate.payload_hash == binding.blob_hash
                                and candidate.size_bytes == binding.size_bytes
                            ):
                                server_blob = candidate
                    else:
                        server_blob = self.store.get_blob_object(
                            dataset.dataset_id,
                            attachment_id=metadata.attachment_id,
                            payload_hash=metadata.payload_hash,
                            owner_user_id=blob_owner_user_id,
                        )
                    if server_blob is not None:
                        server_availability = server_blob.status
                metadata_claims_server_blob = (
                    envelope.adapter_version == 1
                    and
                    not self.settings.supports_attachments
                    and _attachment_ref_has_server_blob(metadata.availability)
                )
                if metadata_claims_server_blob:
                    server_availability = "available"
                download_status = attachment_restore_status(
                    attachment_availability,
                    attachment_id=metadata.attachment_id,
                    payload_hash=metadata.payload_hash,
                )
                verified_locally = attachment_verified_locally(
                    attachment_availability,
                    attachment_id=metadata.attachment_id,
                    payload_hash=metadata.payload_hash,
                )
                required_for_restore = not metadata_only
                blob_details.append(
                    SyncRestoreBlobCompleteness(
                        attachment_id=metadata.attachment_id,
                        payload_hash=metadata.payload_hash,
                        size_bytes=metadata.size_bytes,
                        content_type=metadata.content_type,
                        parent_domain=metadata.parent_domain,
                        parent_object_id=metadata.parent_object_id,
                        server_availability=server_availability,
                        download_status=download_status,
                        required_for_restore=required_for_restore,
                    )
                )
                summary = SyncRestorePreviewAttachmentRef(
                    dataset_id=dataset.dataset_id,
                    attachment_id=metadata.attachment_id,
                    object_id=envelope.object_id,
                    parent_domain=metadata.parent_domain,
                    parent_object_id=metadata.parent_object_id,
                    content_type=metadata.content_type,
                    size_bytes=metadata.size_bytes,
                    payload_hash=metadata.payload_hash,
                    availability=server_availability,
                    server_cursor=envelope.server_cursor or 0,
                    adapter_version=envelope.adapter_version,
                )
                attachment_refs.append(summary)
                if required_for_restore and server_availability != "available" and not attachment_available_locally(
                    attachment_availability,
                    attachment_id=metadata.attachment_id,
                    payload_hash=metadata.payload_hash,
                ):
                    missing_blobs.append(summary)
                    warnings.append(
                        SyncRestorePreviewWarning(
                            code="sync_attachment_blob_missing",
                            message=("Attachment blob is not available from the Sync v2 " "M1 server."),
                            dataset_id=dataset.dataset_id,
                            attachment_id=metadata.attachment_id,
                            object_id=envelope.object_id,
                            payload_hash=metadata.payload_hash,
                        )
                    )
                if verified_locally and server_availability != "available":
                    warnings.append(
                        SyncRestorePreviewWarning(
                            code="sync_attachment_blob_verified_without_server_copy",
                            message=(
                                "Attachment blob is verified locally but is not available "
                                "from the Sync v2 server."
                            ),
                            dataset_id=dataset.dataset_id,
                            attachment_id=metadata.attachment_id,
                            object_id=envelope.object_id,
                            payload_hash=metadata.payload_hash,
                        )
                    )

        domain_details = _build_restore_domain_details(
            safe_applies=safe_applies,
            tombstones=tombstones,
            object_conflicts=object_conflicts,
            blob_details=blob_details,
            metadata_only=metadata_only,
        )
        restore_status = _restore_status_from_domain_details(
            domain_details,
            has_blob_selection=bool(blob_details),
            metadata_only=metadata_only,
        )

        return SyncRestorePreview(
            datasets=preview_datasets,
            ordered_actions=ordered_actions,
            safe_applies=safe_applies,
            object_conflicts=object_conflicts,
            tombstones=tombstones,
            attachment_refs=attachment_refs,
            missing_blobs=missing_blobs,
            envelope_ranges=envelope_ranges,
            total_counts=dict(sorted(total_counts.items())),
            encryption=self.settings.server_trusted_encryption.encryption,
            key_status=key_status,
            warnings=warnings,
            generated_at=self.clock() or None,
            filters_applied={
                "dataset_ids": list(dataset_ids or []),
                "domains": list(domains or []),
                "selected_object_ids": sorted(selected_object_id_set),
                "selected_attachment_ids": sorted(selected_attachment_id_set),
                "metadata_only": metadata_only,
            },
            restore_status=restore_status,
            domain_details=domain_details,
            blob_details=blob_details,
            metadata_only_allowed=True,
        )

    def repair(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None = None,
        domains: Sequence[SyncDomain] | None = None,
        since_cursor: int = 0,
        failed_only: bool = False,
        limit: int | None = None,
    ) -> SyncReplayRepairResult:
        """Replay accepted envelopes to repair server-side materialized projections."""

        if since_cursor < 0:
            raise SyncStoreError("Invalid sync cursor: repair since_cursor must be non-negative")
        if limit is not None and limit < 1:
            raise SyncStoreError("Sync repair limit must be greater than zero")
        if device_id is not None:
            self._require_registered_device(user_id, device_id)
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        selected_domains = self._selected_domains(dataset, domains)

        def _repair_materialize(
            envelope: SyncEnvelope,
            store: SyncV2Store | None,
        ) -> MaterializationResult:
            materialization = self._materialize_envelope(envelope, store=store)
            if materialization.status == "conflict":
                snapshot = self._envelope_snapshot(envelope, store=store)
                self._store_materialization_conflict(
                    dataset,
                    snapshot,
                    materialization,
                    store=store,
                )
            return materialization

        def _repair_snapshot(
            envelope: SyncEnvelope,
            store: SyncV2Store | None,
        ) -> SyncEnvelope:
            return self._envelope_snapshot(envelope, store=store)

        return SyncReplayRepairer(
            store=self.store,
            materializers=self.materializers,
            materialize=_repair_materialize,
            snapshot=_repair_snapshot,
            scan_limit=self.settings.restore_manifest_scan_limit,
        ).run(
            dataset_id=dataset.dataset_id,
            domains=selected_domains,
            since_cursor=since_cursor,
            failed_only=failed_only,
            limit=limit,
        )

    def store_attachment(
        self,
        *,
        user_id: str,
        dataset_id: str,
        domain: SyncDomain,
        entity_id: str,
        attachment_id: str,
        content_type: str,
        size_bytes: int,
        payload_ciphertext: str,
        payload_hash: str,
        encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY,
        metadata: dict[str, object] | None = None,
    ) -> SyncAttachment:
        """Persist a small encrypted attachment payload for later restore."""

        blob_store = self._require_blob_transfer()
        if encryption_policy != DEFAULT_M1_ENCRYPTION_POLICY:
            raise SyncStoreError("Sync attachment persistence requires server_trusted_v1 encryption")
        payload = payload_ciphertext.encode("utf-8")
        if (
            size_bytes > self.settings.max_attachment_bytes
            or _ciphertext_exceeds_attachment_limit(
                payload_ciphertext,
                self.settings.max_attachment_bytes,
            )
            or len(payload) != size_bytes
        ):
            raise SyncStoreError("Sync attachment payload exceeds the server size limit")
        if not payload_ciphertext:
            raise SyncStoreError("Sync attachment payload_ciphertext is required")
        self._validate_sha256_hash(payload_hash, field_name="payload_hash")
        self._require_blob_dataset(user_id=user_id, dataset_id=dataset_id, domain=domain)
        blob_id = self.id_factory("blob")
        upload_id = self.id_factory("blob-upload")
        storage_key = self._write_single_chunk_blob(
            blob_store=blob_store,
            upload_id=upload_id,
            payload=payload,
            payload_hash=payload_hash,
        )
        blob = self.store.complete_blob_upload(
            SyncBlobObjectCreate(
                blob_id=blob_id,
                dataset_id=dataset_id,
                owner_user_id=user_id,
                attachment_id=attachment_id,
                payload_hash=payload_hash,
                content_type=content_type,
                size_bytes=size_bytes,
                storage_backend=self.settings.blob_storage_backend,
                storage_key=storage_key,
                encryption_policy=encryption_policy,
                metadata=dict(metadata or {}),
            )
        )
        attachment_metadata = dict(metadata or {})
        attachment_metadata.update(
            {
                "blob_id": blob.blob_id,
                "storage_backend": blob.storage_backend,
                "storage_key": blob.storage_key,
            }
        )
        return self.store.store_attachment(
            SyncAttachmentCreate(
                attachment_id=attachment_id,
                dataset_id=dataset_id,
                domain=domain,
                entity_id=entity_id,
                content_type=content_type,
                size_bytes=size_bytes,
                payload_ciphertext=payload_ciphertext,
                payload_hash=payload_hash,
                encryption_policy=encryption_policy,
                metadata=attachment_metadata,
            )
        )

    def create_blob_upload_session(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None,
        domain: SyncDomain,
        entity_id: str,
        attachment_id: str,
        content_type: str,
        size_bytes: int,
        payload_hash: str,
        chunk_size: int,
        chunk_count: int,
        idempotency_key: str | None = None,
        encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY,
        metadata: dict[str, object] | None = None,
        trusted_notes_attachment_bootstrap_id: str | None = None,
    ) -> SyncBlobUploadSession:
        """Create or resume a quota-checked M2 blob upload session."""

        self._require_blob_transfer()
        if encryption_policy != DEFAULT_M1_ENCRYPTION_POLICY:
            raise SyncStoreError("Sync blob upload requires server_trusted_v1 encryption")
        if device_id is not None:
            self._require_registered_device(user_id, device_id)
        dataset = self._require_blob_dataset(
            user_id=user_id,
            dataset_id=dataset_id,
            domain=domain,
        )
        self._validate_blob_limits(
            user_id=user_id,
            dataset_id=dataset_id,
            size_bytes=size_bytes,
            chunk_size=chunk_size,
            chunk_count=chunk_count,
        )
        self._validate_sha256_hash(payload_hash, field_name="payload_hash")
        normalized_metadata = dict(metadata or {})
        if domain == "attachment.ref":
            normalized_metadata = self._normalize_notes_attachment_upload_metadata(
                dataset=dataset,
                entity_id=entity_id,
                attachment_id=attachment_id,
                content_type=content_type,
                metadata=normalized_metadata,
                trusted_bootstrap_id=trusted_notes_attachment_bootstrap_id,
            )
        elif "notes_attachment_intent" in normalized_metadata:
            raise SyncStoreError(
                "notes_attachment_intent is reserved for attachment.ref uploads"
            )
        return self.store.create_blob_upload_session(
            SyncBlobUploadSessionCreate(
                upload_id=self.id_factory("blob-upload"),
                dataset_id=dataset_id,
                owner_user_id=user_id,
                device_id=device_id,
                attachment_id=attachment_id,
                domain=domain,
                object_id=entity_id,
                content_type=content_type,
                size_bytes=size_bytes,
                payload_hash=payload_hash,
                chunk_size=chunk_size,
                chunk_count=chunk_count,
                reserved_quota_bytes=size_bytes,
                idempotency_key=idempotency_key,
                metadata=normalized_metadata,
            )
        )

    def _normalize_notes_attachment_upload_metadata(
        self,
        *,
        dataset: SyncDataset,
        entity_id: str,
        attachment_id: str,
        content_type: str,
        metadata: dict[str, object],
        trusted_bootstrap_id: str | None = None,
    ) -> dict[str, object]:
        """Validate and bind one strict immutable Notes attachment upload intent."""

        try:
            adapter = self.adapters.get("attachment.ref")
        except KeyError as exc:
            raise SyncStoreError("Notes attachment upload intent is unavailable") from exc
        state = dataset.metadata.get("notes_attachment_v2")
        trusted_bootstrap = bool(
            trusted_bootstrap_id is not None
            and isinstance(state, Mapping)
            and state.get("state") == "initializing"
            and state.get("bootstrap_id") == trusted_bootstrap_id
        )
        if not trusted_bootstrap and not sync_v2_attachment_ref_v2_is_writable(
            dataset,
            notes_attachment_sync_enabled=bool(
                getattr(adapter, "v2_writes_enabled", False)
            ),
            supports_attachments=self.settings.supports_attachments,
        ):
            raise SyncStoreError("Notes attachment upload intent is unavailable")
        raw_intent = metadata.get("notes_attachment_intent")
        if not isinstance(raw_intent, Mapping):
            raise SyncStoreError(
                "attachment.ref uploads require notes_attachment_intent"
            )
        if "_notes_attachment_binding" in metadata:
            raise SyncStoreError("Notes attachment upload metadata is reserved")
        intent_type = raw_intent.get("intent")
        allowed = (
            {"intent", "note_id", "attachment_id", "file_name"}
            if intent_type == "create"
            else {
                "intent",
                "note_id",
                "attachment_id",
                "base_server_cursor",
                "base_object_revision",
                "base_object_hash",
            }
            if intent_type == "replace"
            else set()
        )
        if not allowed or set(raw_intent) != allowed:
            raise SyncStoreError("Notes attachment upload intent is invalid")
        note_id = _canonical_attachment_uuid(raw_intent.get("note_id"), "note_id")
        intent_attachment_id = _canonical_attachment_uuid(
            raw_intent.get("attachment_id"),
            "attachment_id",
        )
        if intent_attachment_id != attachment_id or intent_attachment_id != entity_id:
            raise SyncStoreError(
                "Notes attachment upload intent does not match its attachment identity"
            )
        try:
            canonical_content_type = validate_note_attachment_content_type(content_type)
        except NoteAttachmentPolicyError as exc:
            raise SyncStoreError(str(exc)) from exc
        if canonical_content_type != content_type:
            raise SyncStoreError("Notes attachment content type must be canonical")

        canonical_intent = dict(raw_intent)
        if intent_type == "create":
            try:
                file_name, _ = canonicalize_note_attachment_file_name(
                    raw_intent.get("file_name")
                )
                original_file_name = validate_note_attachment_original_file_name(
                    file_name
                )
            except NoteAttachmentPolicyError as exc:
                raise SyncStoreError(str(exc)) from exc
            canonical_intent["file_name"] = file_name
            bound_names = {
                "file_name": file_name,
                "original_file_name": original_file_name,
            }
        else:
            base_cursor = raw_intent.get("base_server_cursor")
            base_revision = raw_intent.get("base_object_revision")
            base_hash = raw_intent.get("base_object_hash")
            if (
                isinstance(base_cursor, bool)
                or not isinstance(base_cursor, int)
                or base_cursor < 1
                or isinstance(base_revision, bool)
                or not isinstance(base_revision, int)
                or base_revision < 1
                or not isinstance(base_hash, str)
            ):
                raise SyncStoreError("Notes attachment replacement base is invalid")
            self._validate_sha256_hash(base_hash, field_name="base_object_hash")
            head = self.store.get_current_head(
                dataset.dataset_id,
                "attachment.ref",
                attachment_id,
            )
            if (
                head is None
                or head.adapter_version != 2
                or head.operation != "upsert"
                or head.deleted
                or head.parent_id != note_id
                or (
                    head.server_cursor,
                    head.object_revision,
                    head.payload_hash,
                )
                != (base_cursor, base_revision, base_hash)
            ):
                raise SyncStoreError(
                    "Notes attachment replacement base does not match the current head"
                )
            current = parse_attachment_ref_v2_payload("upsert", head.payload)
            bound_names = {
                "file_name": current.file_name,
                "original_file_name": current.original_file_name,
            }
        return {
            **metadata,
            "notes_attachment_intent": canonical_intent,
            "_notes_attachment_binding": {
                "intent": intent_type,
                "note_id": note_id,
                "attachment_id": intent_attachment_id,
                **bound_names,
            },
        }

    def get_blob_upload_session(
        self,
        *,
        user_id: str,
        dataset_id: str,
        upload_id: str,
    ) -> SyncBlobUploadSession:
        """Return an upload session after checking dataset ownership."""

        self._require_blob_dataset(user_id=user_id, dataset_id=dataset_id)
        session = self.store.get_blob_upload_session(upload_id, dataset_id=dataset_id)
        if session is None:
            raise SyncStoreError(f"Sync blob upload session not found: {upload_id}")
        if session.device_id is not None:
            self._require_registered_device(user_id, session.device_id)
        return session

    def upload_blob_chunk(
        self,
        *,
        user_id: str,
        dataset_id: str,
        upload_id: str,
        chunk_index: int,
        offset_bytes: int,
        chunk_payload: bytes,
        chunk_hash: str,
    ) -> SyncBlobChunk:
        """Verify, store, and record one resumable upload chunk."""

        blob_store = self._require_blob_transfer()
        session = self.get_blob_upload_session(
            user_id=user_id,
            dataset_id=dataset_id,
            upload_id=upload_id,
        )
        self._validate_sha256_hash(chunk_hash, field_name="chunk_hash")
        if len(chunk_payload) > self.settings.max_chunk_bytes:
            raise SyncStoreError("Sync blob chunk exceeds the server size limit")
        if chunk_index < 0 or chunk_index >= session.chunk_count:
            raise SyncStoreError("Sync blob chunk index is outside the upload session")
        if offset_bytes < 0:
            raise SyncStoreError("Sync blob chunk offset is invalid")
        expected_offset = session.chunk_size * chunk_index
        if offset_bytes != expected_offset:
            raise SyncStoreError("Sync blob chunk offset does not match the upload session")
        expected_size = min(session.chunk_size, max(session.size_bytes - offset_bytes, 0))
        if len(chunk_payload) != expected_size:
            raise SyncStoreError("Sync blob chunk size does not match the upload session")
        existing_chunk = self.store.get_blob_chunk(
            upload_id,
            chunk_index,
            dataset_id=dataset_id,
        )
        if existing_chunk is not None and (
            existing_chunk.offset_bytes != offset_bytes
            or existing_chunk.size_bytes != len(chunk_payload)
            or existing_chunk.chunk_hash != chunk_hash
        ):
            raise SyncIdempotencyConflictError(
                "Sync blob chunk was reused with different content"
            )
        try:
            storage_key = blob_store.write_upload_chunk(
                upload_id=upload_id,
                chunk_index=chunk_index,
                payload=chunk_payload,
                expected_hash=chunk_hash,
            )
        except SyncBlobStoreError as exc:
            raise SyncStoreError(str(exc)) from exc
        return self.store.record_blob_chunk(
            SyncBlobChunkCreate(
                upload_id=upload_id,
                dataset_id=dataset_id,
                chunk_index=chunk_index,
                offset_bytes=offset_bytes,
                size_bytes=len(chunk_payload),
                chunk_hash=chunk_hash,
                storage_key=storage_key,
            )
        )

    def complete_blob_upload(
        self,
        *,
        user_id: str,
        dataset_id: str,
        upload_id: str,
    ) -> SyncBlobObject:
        """Verify all chunks and mark a blob object available."""

        blob_store = self._require_blob_transfer()
        session = self.get_blob_upload_session(
            user_id=user_id,
            dataset_id=dataset_id,
            upload_id=upload_id,
        )
        if session.status == "complete" and session.blob_id is not None:
            existing = self.store.get_blob_object(
                dataset_id,
                blob_id=session.blob_id,
                owner_user_id=user_id,
            )
            if existing is None:
                raise SyncStoreError("Completed Sync blob upload is unavailable")
            return existing
        if session.missing_chunks:
            raise SyncStoreError("Sync blob upload session is missing chunks")
        storage_namespace_id: str | None = None
        if session.domain == "attachment.ref":
            namespace = self.store.get_or_create_storage_namespace(
                dataset_id,
                owner_user_id=user_id,
            )
            storage_namespace_id = namespace.storage_namespace_id
        expected_storage_key = (
            blob_store.legacy_storage_key(session.payload_hash)
            if storage_namespace_id is None
            else blob_store.namespace_storage_key(
                storage_namespace_id,
                session.payload_hash,
            )
        )
        blob_create = SyncBlobObjectCreate(
            blob_id=self.id_factory("blob"),
            dataset_id=dataset_id,
            owner_user_id=user_id,
            attachment_id=session.attachment_id,
            payload_hash=session.payload_hash,
            content_type=session.content_type,
            size_bytes=session.size_bytes,
            storage_backend=self.settings.blob_storage_backend,
            storage_key=expected_storage_key,
            encryption_policy=DEFAULT_M1_ENCRYPTION_POLICY,
            metadata={},
        )
        with self.store.blob_write_guard(
            dataset_id,
            session.domain,
            session.object_id,
        ) as guarded:
            guarded.require_blob_upload_completion_allowed(blob_create)
            try:
                storage_key = blob_store.commit_upload(
                    upload_id=upload_id,
                    payload_hash=session.payload_hash,
                    chunk_indexes=list(range(session.chunk_count)),
                    storage_namespace_id=storage_namespace_id,
                )
            except SyncBlobStoreError as exc:
                raise SyncStoreError(str(exc)) from exc
            if storage_key != expected_storage_key:
                raise SyncStoreError("Sync blob storage key changed during commit")
            blob = guarded.complete_blob_upload(blob_create)
        try:
            blob_store.discard_upload(upload_id)
        except OSError as exc:
            logger.warning(
                "Sync blob upload completed but cleanup failed for upload_id={}: {}",
                upload_id,
                exc,
            )
        return blob

    def require_completed_notes_attachment_upload(
        self,
        *,
        user_id: str,
        dataset_id: str,
        upload_id: str,
        note_id: str,
        attachment_id: str,
    ) -> tuple[SyncBlobUploadSession, SyncBlobObject]:
        """Return one completed upload only when its immutable Notes intent matches."""

        session = self.get_blob_upload_session(
            user_id=user_id,
            dataset_id=dataset_id,
            upload_id=upload_id,
        )
        intent = session.metadata.get("notes_attachment_intent")
        if (
            session.owner_user_id != user_id
            or session.domain != "attachment.ref"
            or session.object_id != attachment_id
            or session.attachment_id != attachment_id
            or not isinstance(intent, Mapping)
            or intent.get("note_id") != note_id
            or intent.get("attachment_id") != attachment_id
        ):
            raise SyncStoreError("Notes attachment upload intent does not match")
        if session.status != "complete" or session.blob_id is None:
            raise SyncStoreError("Notes attachment upload is not complete")
        blob = self.store.get_blob_object(
            dataset_id,
            blob_id=session.blob_id,
            owner_user_id=user_id,
        )
        if (
            blob is None
            or blob.status != "available"
            or blob.payload_hash != session.payload_hash
            or blob.size_bytes != session.size_bytes
            or blob.content_type != session.content_type
        ):
            raise SyncStoreError("Notes attachment upload blob is unavailable")
        return session, blob

    def cancel_blob_upload(
        self,
        *,
        user_id: str,
        dataset_id: str,
        upload_id: str,
    ) -> SyncBlobUploadSession:
        """Cancel an upload session and remove staged chunks."""

        blob_store = self._require_blob_transfer()
        session = self.get_blob_upload_session(
            user_id=user_id,
            dataset_id=dataset_id,
            upload_id=upload_id,
        )
        session = self.store.cancel_blob_upload_session(session.upload_id, dataset_id=dataset_id)
        blob_store.discard_upload(upload_id)
        return session

    def blob_download_manifest(
        self,
        *,
        user_id: str,
        dataset_id: str,
        attachment_id: str,
        chunk_size: int | None = None,
    ) -> SyncBlobDownloadManifest:
        """Return resumable download metadata for an available blob or metadata-only ref."""

        blob_store = self._require_blob_transfer()
        dataset = self._require_blob_dataset(user_id=user_id, dataset_id=dataset_id)
        blob = self.store.get_blob_object(
            dataset_id,
            attachment_id=attachment_id,
            owner_user_id=self._blob_owner_user_id(dataset=dataset, user_id=user_id),
        )
        if blob is None:
            return self._metadata_only_blob_manifest(
                dataset_id=dataset_id,
                attachment_id=attachment_id,
            )
        self._validate_blob_storage_metadata(blob_store=blob_store, blob=blob)
        normalized_chunk_size = self._normalize_download_chunk_size(chunk_size)
        hasher = hashlib.sha256()
        chunks: list[SyncBlobDownloadChunk] = []
        offset = 0
        try:
            for index, payload in enumerate(
                blob_store.iter_blob(
                    blob.storage_key,
                    chunk_size=normalized_chunk_size,
                )
            ):
                hasher.update(payload)
                chunks.append(
                    SyncBlobDownloadChunk(
                        chunk_index=index,
                        offset_bytes=offset,
                        size_bytes=len(payload),
                        chunk_hash=_sha256_bytes(payload),
                        download_url=(
                            f"/api/v1/sync/attachments/{attachment_id}"
                            f"?dataset_id={dataset_id}&offset={offset}"
                            f"&size={len(payload)}"
                        ),
                    )
                )
                offset += len(payload)
        except (OSError, SyncBlobStoreError) as exc:
            raise SyncStoreError("Sync blob was not found or is not accessible") from exc
        if offset != blob.size_bytes or "sha256:" + hasher.hexdigest() != blob.payload_hash:
            raise SyncStoreError("Sync blob storage integrity check failed")
        return SyncBlobDownloadManifest(
            dataset_id=dataset_id,
            attachment_id=attachment_id,
            blob_id=blob.blob_id,
            availability="available",
            content_type=blob.content_type,
            size_bytes=blob.size_bytes,
            payload_hash=blob.payload_hash,
            chunks=chunks,
        )

    def iter_blob_bytes(
        self,
        *,
        user_id: str,
        dataset_id: str,
        attachment_id: str,
        offset: int = 0,
        size: int | None = None,
    ) -> Iterator[bytes]:
        """Yield a byte range from an available Sync v2 M2 blob."""

        if offset < 0:
            raise SyncStoreError("Sync blob download offset is invalid")
        if size is not None and size < 0:
            raise SyncStoreError("Sync blob download size is invalid")
        if size is not None:
            self._normalize_download_chunk_size(size)
        blob_store, blob = self._require_download_blob(
            user_id=user_id,
            dataset_id=dataset_id,
            attachment_id=attachment_id,
        )
        self._validate_blob_storage_metadata(blob_store=blob_store, blob=blob)
        return self._iter_blob_storage_range(
            blob_store=blob_store,
            storage_key=blob.storage_key,
            offset=offset,
            size=size,
        )

    def blob_download_metadata(
        self,
        *,
        user_id: str,
        dataset_id: str,
        attachment_id: str,
    ) -> SyncBlobObject:
        """Return metadata for an available blob without reading its payload."""

        blob_store, blob = self._require_download_blob(
            user_id=user_id,
            dataset_id=dataset_id,
            attachment_id=attachment_id,
        )
        self._validate_blob_storage_metadata(blob_store=blob_store, blob=blob)
        return blob

    def _iter_blob_storage_range(
        self,
        *,
        blob_store: LocalSyncBlobStore,
        storage_key: str,
        offset: int,
        size: int | None,
    ) -> Iterator[bytes]:
        try:
            yield from blob_store.iter_blob(
                storage_key,
                offset=offset,
                size=size,
                chunk_size=self.settings.max_chunk_bytes,
            )
        except (OSError, SyncBlobStoreError) as exc:
            raise SyncStoreError("Sync blob was not found or is not accessible") from exc

    def read_blob_bytes(
        self,
        *,
        user_id: str,
        dataset_id: str,
        attachment_id: str,
        offset: int = 0,
        size: int | None = None,
    ) -> bytes:
        """Read a byte range from an available Sync v2 M2 blob."""

        if offset < 0:
            raise SyncStoreError("Sync blob download offset is invalid")
        if size is not None and size < 0:
            raise SyncStoreError("Sync blob download size is invalid")
        return b"".join(
            self.iter_blob_bytes(
                user_id=user_id,
                dataset_id=dataset_id,
                attachment_id=attachment_id,
                offset=offset,
                size=size,
            )
        )

    def list_conflicts(
        self,
        *,
        user_id: str,
        dataset_id: str,
        status: ConflictStatus | None = None,
    ) -> list[SyncConflict]:
        self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        return self.store.list_conflicts(dataset_id, status=status)

    def resolve_conflict(
        self,
        *,
        user_id: str,
        conflict_id: str,
        dataset_id: str | None = None,
        action: str,
        resolution_envelope: SyncEnvelopeCreate | None = None,
        resolved_by_envelope_id: str | None = None,
        resolved_by_device_id: str | None = None,
        notes: str | None = None,
    ) -> SyncConflict:
        conflict = self.store.get_conflict(conflict_id)
        if conflict is None:
            raise SyncStoreError("Sync conflict was not found or is not accessible")
        if dataset_id is not None and conflict.dataset_id != dataset_id:
            raise SyncStoreError("Sync conflict was not found or is not accessible")
        try:
            dataset = self._require_dataset_access(user_id=user_id, dataset_id=conflict.dataset_id)
        except SyncStoreError as exc:
            raise SyncStoreError("Sync conflict was not found or is not accessible") from exc
        if action not in {"overwrite", "duplicate_rename", "skip"}:
            raise SyncStoreError(f"Sync conflict resolution action is not supported: {action}")
        if conflict.status != "unresolved":
            if self._is_conflict_resolution_replay(
                conflict,
                action=action,
                resolution_envelope=resolution_envelope,
                resolved_by_envelope_id=resolved_by_envelope_id,
                resolved_by_device_id=resolved_by_device_id,
                notes=notes,
            ):
                return conflict
            raise SyncStoreError("Sync conflict is already resolved")
        if resolved_by_device_id is not None:
            self._require_registered_device(user_id, resolved_by_device_id)
        if action in {"overwrite", "duplicate_rename"} and resolution_envelope is None:
            raise SyncStoreError(f"Sync {action} requires a resolution envelope")
        if action == "skip" and resolution_envelope is not None:
            raise SyncStoreError(f"Sync {action} must not include a resolution envelope")
        resolution_device_id = resolved_by_device_id
        if resolution_envelope is not None:
            if has_guard_required_routing_key(resolution_envelope.routing_metadata):
                raise SyncStoreError(
                    "Sync resolution envelope contains reserved routing metadata"
                )
            resolution_device_id = resolved_by_device_id or resolution_envelope.device_id
            self._require_registered_device(user_id, resolution_device_id or "")
            if resolution_envelope.dataset_id != dataset.dataset_id:
                raise SyncStoreError("Sync resolution envelope dataset_id must match the conflict dataset")
            if resolution_envelope.domain != conflict.domain:
                raise SyncStoreError("Sync resolution envelope must target the conflict domain")
            if action == "duplicate_rename":
                if resolution_envelope.entity_id == conflict.entity_id:
                    raise SyncStoreError("Sync duplicate_rename resolution envelope must use a distinct object_id")
            elif resolution_envelope.entity_id != conflict.entity_id:
                raise SyncStoreError("Sync resolution envelope must target the conflict entity")
            if (
                resolved_by_device_id is not None
                and resolution_envelope.device_id is not None
                and resolution_envelope.device_id != resolved_by_device_id
            ):
                raise SyncStoreError("Sync resolution envelope device_id must match resolved_by_device_id")
            if self._payload_exceeds_size_limit(resolution_envelope):
                raise SyncStoreError("Sync resolution envelope payload exceeds the server size limit")

        if conflict.server_sequence is None:
            raise SyncStoreError("Sync conflict has no canonical source envelope")
        source = self.store.get_envelope_by_server_cursor(conflict.server_sequence)
        if (
            source is None
            or source.dataset_id != dataset.dataset_id
            or source.client_envelope_id != conflict.local_envelope_id
            or source.domain != conflict.domain
            or source.object_id != conflict.entity_id
        ):
            raise SyncStoreError("Sync conflict source envelope was not found")

        with self.store.materialization_guard(
            [source],
            require_predecessors=False,
        ) as guarded_store:
            claimed_conflict = guarded_store.claim_conflict_resolution(
                conflict_id,
                dataset_id=dataset.dataset_id,
                resolved_by_device_id=resolution_device_id,
                resolution_action=action,
                resolution_notes=notes,
            )
            guarded_source = guarded_store.require_conflict_resolution_predecessors_applied(
                conflict_id,
                dataset_id=dataset.dataset_id,
                resolved_by_device_id=resolution_device_id,
                resolution_action=action,
                resolution_notes=notes,
            )
            rebase_plan = guarded_store.stage_later_claimed_conflict_rebase_plan(
                conflict_id,
                dataset_id=dataset.dataset_id,
                resolved_by_device_id=resolution_device_id,
                resolution_action=action,
                resolution_notes=notes,
            )
            inserted: SyncEnvelope | None = None
            if resolution_envelope is not None:
                resolution_context = self._conflict_resolution_adapter_context(
                    dataset,
                    conflict=claimed_conflict,
                    source=guarded_source,
                    resolution_envelope=resolution_envelope,
                    action=action,
                    store=guarded_store,
                )
                try:
                    outcome = self._evaluate_envelope(
                        dataset,
                        resolution_envelope,
                        context=resolution_context,
                    )
                except PrivatePayloadValidationError as exc:
                    raise SyncStoreError(
                        "Sync resolution envelope private payload validation failed"
                    ) from exc
                if not isinstance(outcome, AdapterAccepted):
                    raise SyncStoreError("Sync resolution envelope was not accepted")
                inserted = guarded_store.insert_claimed_conflict_resolution_envelope(
                    replace(
                        resolution_envelope,
                        device_id=resolution_device_id,
                        status="accepted",
                    ),
                    conflict_id=conflict_id,
                    dataset_id=dataset.dataset_id,
                    resolved_by_device_id=resolution_device_id,
                    resolution_action=action,
                    resolution_notes=notes,
                )
            if inserted is not None:
                if inserted.apply_status != "applied":
                    materialization = self._materialize_envelope(
                        inserted,
                        store=guarded_store,
                    )
                    inserted = self._envelope_snapshot(inserted, store=guarded_store)
                    if materialization.status != "applied" or inserted.apply_status != "applied":
                        raise SyncStoreError("Sync resolution envelope was not applied")
                resolved_by_envelope_id = inserted.envelope_id
            guarded_store.rebase_later_claimed_conflict_envelopes(
                conflict_id,
                dataset_id=dataset.dataset_id,
                resolved_by_device_id=resolution_device_id,
                resolution_action=action,
                resolution_notes=notes,
                expected_server_cursors=rebase_plan,
            )
            guarded_store.terminalize_claimed_conflict_envelope(
                conflict_id,
                dataset_id=dataset.dataset_id,
                resolved_by_device_id=resolution_device_id,
                resolution_action=action,
                resolution_notes=notes,
                apply_error_code=(
                    "sync_conflict_skipped"
                    if action == "skip"
                    else "sync_conflict_superseded"
                ),
            )
            resolved_status: ConflictStatus = (
                "dismissed" if action == "skip" else "resolved"
            )
            return guarded_store.resolve_conflict(
                conflict_id,
                dataset_id=dataset.dataset_id,
                server_cursor=conflict.server_sequence,
                status=resolved_status,
                resolved_by_envelope_id=resolved_by_envelope_id,
                resolved_by_device_id=resolution_device_id,
                resolution_action=action,
                resolution_notes=notes,
            )

    def _is_conflict_resolution_replay(
        self,
        conflict: SyncConflict,
        *,
        action: str,
        resolution_envelope: SyncEnvelopeCreate | None,
        resolved_by_envelope_id: str | None,
        resolved_by_device_id: str | None,
        notes: str | None,
    ) -> bool:
        if action != conflict.resolution_action or notes != conflict.resolution_notes:
            return False
        effective_device_id = resolved_by_device_id
        if effective_device_id is None and resolution_envelope is not None:
            effective_device_id = resolution_envelope.device_id
        if effective_device_id != conflict.resolved_by_device_id:
            return False
        effective_envelope_id = resolved_by_envelope_id
        if effective_envelope_id is None and resolution_envelope is not None:
            existing = self._find_existing_resolution_envelope(
                conflict.dataset_id,
                resolution_envelope,
                effective_device_id=effective_device_id,
            )
            if existing is None:
                return False
            effective_envelope_id = existing.envelope_id
        return effective_envelope_id == conflict.resolved_by_envelope_id

    def _find_existing_resolution_envelope(
        self,
        dataset_id: str,
        resolution_envelope: SyncEnvelopeCreate,
        *,
        effective_device_id: str | None,
    ) -> SyncEnvelope | None:
        if resolution_envelope.dataset_id != dataset_id:
            return None
        try:
            return self.store.get_existing_envelope_for_idempotency(
                replace(
                    resolution_envelope,
                    device_id=effective_device_id,
                    status="accepted",
                )
            )
        except SyncIdempotencyConflictError:
            return None

    def store_key_recovery_bundle(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None,
        key_purpose: str,
        wrapped_key_blob: str,
        kdf_metadata: dict[str, object] | None = None,
        recovery_hint: str | None = None,
        rotation_of_key_record_id: str | None = None,
        encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY,
        key_epoch: int = 1,
        active_from_server_sequence: int | None = None,
        superseded_at: str | None = None,
        wrapped_for: SyncKeyWrappedFor = "recovery",
        rewrap_status: SyncKeyRewrapStatus = "not_required",
    ) -> SyncKeyRecord:
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        if device_id is not None:
            self._require_registered_device(user_id, device_id)
        self._validate_key_recovery_bundle(
            user_id=user_id,
            dataset_id=dataset.dataset_id,
            key_purpose=key_purpose,
            wrapped_key_blob=wrapped_key_blob,
            kdf_metadata=kdf_metadata,
            rotation_of_key_record_id=rotation_of_key_record_id,
        )
        return self.store.store_key_record(
            SyncKeyRecordCreate(
                key_record_id=self.id_factory("key"),
                dataset_id=dataset.dataset_id,
                user_id=user_id,
                device_id=device_id,
                key_purpose=key_purpose,
                wrapped_key_blob=wrapped_key_blob,
                kdf_metadata=dict(kdf_metadata or {}),
                recovery_hint=recovery_hint,
                rotation_of_key_record_id=rotation_of_key_record_id,
                encryption_policy=encryption_policy,
                key_epoch=key_epoch,
                active_from_server_sequence=active_from_server_sequence,
                superseded_at=superseded_at,
                wrapped_for=wrapped_for,
                rewrap_status=rewrap_status,
            )
        )

    def list_key_recovery_bundles(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None = None,
        key_purpose: str | None = "dataset_recovery",
    ) -> list[SyncKeyRecord]:
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        return [
            record
            for record in self.store.list_key_records(
                dataset.dataset_id,
                user_id=user_id,
                device_id=device_id,
                key_purpose=key_purpose,
            )
            if record.revoked_at is None
        ]

    def preview_key_rotation(
        self,
        *,
        user_id: str,
        dataset_id: str,
        target_encryption_policy: EncryptionPolicy,
        source_key_record_ids: Sequence[str] | None = None,
    ) -> SyncKeyRotationResult:
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        if target_encryption_policy not in SYNC_V2_ENCRYPTION_POLICIES:
            self._raise_invalid_key_rotation()

        all_records = self.store.list_key_records(
            dataset.dataset_id,
            user_id=user_id,
            key_purpose=SYNC_DATASET_RECOVERY_KEY_PURPOSE,
        )
        active_records = [
            record
            for record in all_records
            if record.revoked_at is None and record.superseded_at is None
        ]
        selected_records, blockers = self._select_key_rotation_sources(
            active_records=active_records,
            all_records=all_records,
            source_key_record_ids=source_key_record_ids,
        )
        if not selected_records:
            blockers.append("sync_key_rotation_no_active_source_keys")

        highest_epoch = max((record.key_epoch for record in all_records), default=0)
        retained_range = self.store.get_dataset_envelope_range(dataset.dataset_id)
        active_from = (retained_range.through_server_sequence or 0) + 1
        summaries = [self._key_rotation_record_summary(record) for record in selected_records]
        return SyncKeyRotationResult(
            dataset_id=dataset.dataset_id,
            target_encryption_policy=target_encryption_policy,
            next_key_epoch=max(highest_epoch + 1, 1),
            active_from_server_sequence=active_from,
            can_commit=not blockers,
            committed=False,
            retained_envelope_range=retained_range,
            affected_key_records=summaries,
            blockers=list(dict.fromkeys(blockers)),
            device_ids=sorted(
                {
                    record.device_id
                    for record in selected_records
                    if record.device_id is not None
                }
            ),
            recovery_target_count=len(selected_records),
        )

    def commit_key_rotation(
        self,
        *,
        user_id: str,
        dataset_id: str,
        rotation_id: str,
        target_encryption_policy: EncryptionPolicy,
        wrapped_key_blob: str,
        kdf_metadata: Mapping[str, object] | None = None,
        source_key_record_ids: Sequence[str] | None = None,
        wrapped_for: SyncKeyWrappedFor = "recovery",
        rewrap_status: SyncKeyRewrapStatus = "complete",
        recovery_hint: str | None = None,
    ) -> SyncKeyRotationResult:
        clean_rotation_id = str(rotation_id or "").strip()
        if not clean_rotation_id:
            self._raise_invalid_key_rotation()

        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        self._validate_key_recovery_bundle(
            user_id=user_id,
            dataset_id=dataset.dataset_id,
            key_purpose=SYNC_DATASET_RECOVERY_KEY_PURPOSE,
            wrapped_key_blob=wrapped_key_blob,
            kdf_metadata=kdf_metadata,
            rotation_of_key_record_id=None,
        )
        new_key_record_id = self._key_rotation_record_id(
            user_id=user_id,
            dataset_id=dataset.dataset_id,
            rotation_id=clean_rotation_id,
        )
        new_record, superseded_records, retained_range = self.store.commit_key_rotation(
            SyncKeyRecordCreate(
                key_record_id=new_key_record_id,
                dataset_id=dataset.dataset_id,
                user_id=user_id,
                key_purpose=SYNC_DATASET_RECOVERY_KEY_PURPOSE,
                wrapped_key_blob=wrapped_key_blob,
                kdf_metadata=dict(kdf_metadata or {}),
                recovery_hint=recovery_hint,
                encryption_policy=target_encryption_policy,
                wrapped_for=wrapped_for,
                rewrap_status=rewrap_status,
            ),
            source_key_record_ids=source_key_record_ids or [],
            superseded_at=self.clock(),
        )
        return self._key_rotation_result_from_records(
            dataset_id=dataset.dataset_id,
            target_encryption_policy=target_encryption_policy,
            next_key_epoch=new_record.key_epoch,
            active_from_server_sequence=new_record.active_from_server_sequence or 1,
            retained_range=retained_range,
            source_records=superseded_records,
            new_record=new_record,
            rotation_id=clean_rotation_id,
            committed=True,
        )

    def _validate_key_recovery_bundle(
        self,
        *,
        user_id: str,
        dataset_id: str,
        key_purpose: str,
        wrapped_key_blob: str,
        kdf_metadata: Mapping[str, object] | None,
        rotation_of_key_record_id: str | None,
    ) -> None:
        if key_purpose != SYNC_DATASET_RECOVERY_KEY_PURPOSE:
            self._raise_invalid_key_recovery_bundle()
        if not isinstance(wrapped_key_blob, str) or not wrapped_key_blob.strip():
            self._raise_invalid_key_recovery_bundle()
        if len(wrapped_key_blob.encode("utf-8")) > SYNC_KEY_RECOVERY_MAX_WRAPPED_KEY_BYTES:
            self._raise_invalid_key_recovery_bundle()

        metadata = dict(kdf_metadata or {})
        algorithm = _key_recovery_metadata_string(
            metadata,
            "algorithm",
            "wrapping_algorithm",
            nested_parent="wrapping",
        )
        salt = _key_recovery_metadata_string(metadata, "salt", nested_parent="kdf")
        if algorithm is None or salt is None:
            self._raise_invalid_key_recovery_bundle()

        if rotation_of_key_record_id is None:
            return
        active_rotation_target = any(
            record.key_record_id == rotation_of_key_record_id and record.revoked_at is None
            for record in self.store.list_key_records(
                dataset_id,
                user_id=user_id,
                key_purpose=SYNC_DATASET_RECOVERY_KEY_PURPOSE,
            )
        )
        if not active_rotation_target:
            self._raise_invalid_key_recovery_bundle()

    @staticmethod
    def _raise_invalid_key_recovery_bundle() -> None:
        raise SyncStoreError("Sync key recovery bundle is invalid")

    @staticmethod
    def _raise_invalid_key_rotation() -> None:
        raise SyncStoreError("Sync key rotation is invalid")

    @staticmethod
    def _key_rotation_record_id(*, user_id: str, dataset_id: str, rotation_id: str) -> str:
        namespace = f"{user_id}\0{dataset_id}\0{rotation_id}".encode()
        return f"key-rotation-{hashlib.sha256(namespace).hexdigest()[:32]}"

    @staticmethod
    def _key_rotation_record_summary(record: SyncKeyRecord) -> SyncKeyRotationKeyRecord:
        return SyncKeyRotationKeyRecord(
            key_record_id=record.key_record_id,
            key_epoch=record.key_epoch,
            encryption_policy=record.encryption_policy,
            wrapped_for=record.wrapped_for,
            rewrap_status=record.rewrap_status,
            device_id=record.device_id,
            key_purpose=record.key_purpose,
            active_from_server_sequence=record.active_from_server_sequence,
            superseded_at=record.superseded_at,
            revoked_at=record.revoked_at,
            rotation_of_key_record_id=record.rotation_of_key_record_id,
        )

    def _select_key_rotation_sources(
        self,
        *,
        active_records: Sequence[SyncKeyRecord],
        all_records: Sequence[SyncKeyRecord],
        source_key_record_ids: Sequence[str] | None,
    ) -> tuple[list[SyncKeyRecord], list[str]]:
        requested_ids = [
            str(record_id).strip()
            for record_id in source_key_record_ids or []
            if str(record_id).strip()
        ]
        requested_ids = list(dict.fromkeys(requested_ids))
        if not requested_ids:
            return list(active_records), []

        active_by_id = {record.key_record_id: record for record in active_records}
        all_by_id = {record.key_record_id: record for record in all_records}
        selected: list[SyncKeyRecord] = []
        blockers: list[str] = []
        for record_id in requested_ids:
            active_record = active_by_id.get(record_id)
            if active_record is not None:
                selected.append(active_record)
                continue
            if record_id in all_by_id:
                blockers.append("sync_key_rotation_source_inactive")
            else:
                blockers.append("sync_key_rotation_source_missing")
        return selected, blockers

    def _existing_key_rotation_sources(
        self,
        *,
        existing_records: Sequence[SyncKeyRecord],
        existing_new_record: SyncKeyRecord,
        source_key_record_ids: Sequence[str] | None,
    ) -> list[SyncKeyRecord]:
        requested_ids = [
            str(record_id).strip()
            for record_id in source_key_record_ids or []
            if str(record_id).strip()
        ]
        if not requested_ids and existing_new_record.rotation_of_key_record_id:
            requested_ids = [existing_new_record.rotation_of_key_record_id]
        record_by_id = {record.key_record_id: record for record in existing_records}
        return [
            record_by_id[record_id]
            for record_id in dict.fromkeys(requested_ids)
            if record_id in record_by_id
        ]

    @staticmethod
    def _validate_existing_key_rotation_record(
        record: SyncKeyRecord,
        *,
        target_encryption_policy: EncryptionPolicy,
        wrapped_key_blob: str,
        kdf_metadata: Mapping[str, object] | None,
        wrapped_for: SyncKeyWrappedFor,
        rewrap_status: SyncKeyRewrapStatus,
    ) -> None:
        if (
            record.encryption_policy != target_encryption_policy
            or record.wrapped_key_blob != wrapped_key_blob
            or record.kdf_metadata != dict(kdf_metadata or {})
            or record.wrapped_for != wrapped_for
            or record.rewrap_status != rewrap_status
        ):
            raise SyncIdempotencyConflictError(
                "Sync key rotation ID was reused with different key material"
            )

    def _key_rotation_result_from_records(
        self,
        *,
        dataset_id: str,
        target_encryption_policy: EncryptionPolicy,
        next_key_epoch: int,
        active_from_server_sequence: int,
        retained_range,
        source_records: Sequence[SyncKeyRecord],
        new_record: SyncKeyRecord,
        rotation_id: str,
        committed: bool,
    ) -> SyncKeyRotationResult:
        affected = [self._key_rotation_record_summary(record) for record in source_records]
        return SyncKeyRotationResult(
            dataset_id=dataset_id,
            target_encryption_policy=target_encryption_policy,
            next_key_epoch=next_key_epoch,
            active_from_server_sequence=active_from_server_sequence,
            can_commit=True,
            committed=committed,
            retained_envelope_range=retained_range,
            affected_key_records=affected,
            blockers=[],
            device_ids=sorted(
                {
                    record.device_id
                    for record in source_records
                    if record.device_id is not None
                }
            ),
            recovery_target_count=len(source_records),
            rotation_id=rotation_id,
            new_key_record=self._key_rotation_record_summary(new_record),
        )

    def _conflict_resolution_adapter_context(
        self,
        dataset: SyncDataset,
        *,
        conflict: SyncConflict,
        source: SyncEnvelope,
        resolution_envelope: SyncEnvelopeCreate,
        action: str,
        store: SyncV2Store,
    ) -> SyncAdapterContext:
        """Build a bound projected-head view for one claimed resolution."""

        del action
        conflict_marked = (
            conflict.conflict_type
            == SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION
        )
        source_marked = (
            source.apply_error_code
            == SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION
        )
        if conflict_marked != source_marked:
            raise SyncStoreError("Sync rebase conflict marker does not match its source")
        through_server_cursor = (
            None
            if source.status != "accepted" or conflict_marked
            else source.server_cursor
        )
        snapshot = {
            (head.domain, head.object_id): head
            for head in store.list_latest_applied_heads(
                dataset.dataset_id,
                through_server_cursor=through_server_cursor,
            )
        }

        def get_head(domain: SyncDomain, object_id: str) -> SyncEnvelope | None:
            return snapshot.get((domain, object_id))

        def list_heads(domain: SyncDomain) -> tuple[SyncEnvelope, ...]:
            return tuple(
                head
                for (head_domain, _object_id), head in snapshot.items()
                if head_domain == domain
            )

        current = get_head(
            resolution_envelope.domain,
            resolution_envelope.object_id,
        )
        return SyncAdapterContext(
            prior_envelopes=(current,) if current is not None else (),
            get_head=get_head,
            list_heads=list_heads,
        )

    def _evaluate_envelope(
        self,
        dataset: SyncDataset,
        envelope: SyncEnvelopeCreate,
        *,
        context: SyncAdapterContext | None = None,
    ) -> AdapterAccepted | AdapterRejected | AdapterConflict | AdapterDeferred:
        if envelope.adapter_version not in self.adapters.get(envelope.domain).supported_adapter_versions:
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="unsupported_adapter_version",
                message=f"Adapter version is not supported for {envelope.domain}",
            )
        if dataset.encryption_policy == "client_private_v1":
            validate_private_payload(
                payload_ciphertext=envelope.payload_ciphertext,
                payload_clear=envelope.payload_clear,
            )
        if context is None:
            current = self.store.get_current_head(
                dataset.dataset_id,
                envelope.domain,
                envelope.object_id,
            )
            if current is not None:
                current = self._restore_personal_context_from_storage(
                    dataset,
                    current,
                )
            context = SyncAdapterContext(
                prior_envelopes=(current,) if current is not None else (),
                get_head=lambda domain, object_id: self._restore_personal_context_optional_head(
                    dataset,
                    self.store.get_current_head(
                        dataset.dataset_id, domain, object_id
                    ),
                ),
                list_heads=lambda domain: tuple(
                    self._restore_personal_context_from_storage(dataset, item)
                    for item in self._list_current_heads_for_adapter(
                        dataset.dataset_id, domain
                    )
                ),
                supports_attachments=self.settings.supports_attachments,
            )
        else:
            context = replace(
                context,
                supports_attachments=self.settings.supports_attachments,
            )
        adapter = self.adapters.get(envelope.domain)
        return _call_adapter_evaluate(adapter, envelope, dataset=dataset, context=context)

    def _protect_personal_context_for_storage(
        self,
        dataset: SyncDataset,
        envelope: SyncEnvelopeCreate,
    ) -> SyncEnvelopeCreate:
        """Encrypt Personal Context bodies before generic Sync persistence."""

        if envelope.domain not in PERSONAL_CONTEXT_SYNC_DOMAINS:
            return envelope
        adapter = self.adapters.get(envelope.domain)
        protector = getattr(adapter, "protect_for_storage", None)
        if protector is None:
            raise PersonalContextStorageEncryptionUnavailableError(
                "Personal Context storage encryption is unavailable"
            )
        try:
            return protector(envelope, dataset=dataset)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise PersonalContextStorageEncryptionUnavailableError(
                "Personal Context storage encryption is unavailable"
            ) from exc

    def _restore_personal_context_from_storage(
        self,
        dataset: SyncDataset,
        envelope: SyncEnvelope,
    ) -> SyncEnvelope:
        """Restore one authenticated Personal Context body for use or delivery."""

        if envelope.domain not in PERSONAL_CONTEXT_SYNC_DOMAINS:
            return envelope
        adapter = self.adapters.get(envelope.domain)
        restorer = getattr(adapter, "restore_from_storage", None)
        if restorer is None:
            raise SyncStoreError("Personal Context stored envelope is unavailable")
        try:
            restored = restorer(envelope, dataset=dataset)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise SyncStoreError(
                "Personal Context stored envelope is unavailable"
            ) from exc
        if not isinstance(restored, SyncEnvelope):
            raise SyncStoreError("Personal Context stored envelope is unavailable")
        return restored

    def _restore_personal_context_optional_head(
        self,
        dataset: SyncDataset,
        envelope: SyncEnvelope | None,
    ) -> SyncEnvelope | None:
        if envelope is None:
            return None
        return self._restore_personal_context_from_storage(dataset, envelope)

    def _list_current_heads_for_adapter(
        self,
        dataset_id: str,
        domain: SyncDomain,
        *,
        store: SyncV2Store | None = None,
    ) -> tuple[SyncEnvelope, ...]:
        """Load current heads through bounded pages for adapter-wide checks."""

        page_size = 1000
        offset = 0
        heads: list[SyncEnvelope] = []
        active_store = store or self.store
        while True:
            page = active_store.list_current_heads(
                dataset_id,
                domain,
                limit=page_size,
                offset=offset,
            )
            heads.extend(page)
            if len(page) < page_size:
                return tuple(heads)
            offset += page_size

    def _require_registered_device(self, user_id: str, device_id: str) -> SyncDevice:
        if not device_id:
            raise SyncStoreError("Sync device was not found or is not accessible")
        device = self.store.get_device(user_id, device_id)
        if (
            device is not None
            and device.revoked_at is None
            and device.status == "active"
        ):
            return device
        raise SyncStoreError("Sync device was not found or is not accessible")

    def _require_dataset_access(self, *, user_id: str, dataset_id: str) -> SyncDataset:
        dataset = self.store.get_dataset(dataset_id)
        if dataset is None or dataset.archived_at is not None:
            raise SyncStoreError("Sync dataset was not found or is not accessible")
        if dataset.scope_type == "personal":
            if dataset.owner_user_id != user_id:
                raise SyncStoreError("Sync dataset was not found or is not accessible")
            return dataset
        if dataset.scope_type == "workspace":
            self._require_workspace_sync_access(
                user_id=user_id,
                workspace_id=dataset.workspace_id,
            )
            return dataset
        raise SyncStoreError("Sync dataset was not found or is not accessible")

    def _accessible_datasets(
        self,
        *,
        user_id: str,
        dataset_ids: Sequence[str] | None = None,
    ) -> list[SyncDataset]:
        if dataset_ids:
            return [
                self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
                for dataset_id in dict.fromkeys(dataset_ids)
            ]
        datasets: list[SyncDataset] = []
        for dataset in self.store.list_datasets_for_user(user_id):
            try:
                self._require_dataset_access(
                    user_id=user_id,
                    dataset_id=dataset.dataset_id,
                )
            except SyncStoreError:
                continue
            datasets.append(dataset)
        return datasets

    def _require_workspace_sync_access(
        self,
        *,
        user_id: str,
        workspace_id: str | None,
    ) -> None:
        if not workspace_id or not workspace_id.strip():
            raise SyncStoreError("Sync workspace was not found or is not accessible")
        if self.workspace_access_checker is None:
            raise SyncStoreError("Sync workspace was not found or is not accessible")
        try:
            granted = bool(self.workspace_access_checker(user_id, workspace_id, "sync"))
        except Exception as exc:
            raise SyncStoreError("Sync workspace was not found or is not accessible") from exc
        if not granted:
            raise SyncStoreError("Sync workspace was not found or is not accessible")

    def _default_background_policy(
        self,
        dataset_id: str,
        device_id: str,
    ) -> SyncBackgroundPolicy:
        return SyncBackgroundPolicy(
            dataset_id=dataset_id,
            device_id=device_id,
            enabled=True,
            minimum_interval_seconds=300,
            backoff_floor_seconds=60,
            max_batch_size=self.settings.max_batch_size,
            max_blob_bytes_per_run=(
                self.settings.max_blob_bytes
                if self.settings.max_blob_bytes is not None
                else self.settings.max_attachment_bytes
            ),
            respect_metered_networks=True,
            maintenance_window=None,
            paused_reason=None,
            pending_local_changes=False,
            updated_at=self.clock(),
        )

    def _diagnostics_domains(
        self,
        *,
        domains: Sequence[SyncDomain],
        envelopes: Sequence[SyncEnvelope],
        conflicts: Sequence[SyncConflict],
    ) -> list[SyncDiagnosticsDomain]:
        stats = {
            domain: {
                "envelope_count": 0,
                "object_ids": set(),
                "latest_server_sequence": 0,
                "failed_apply_count": 0,
                "unresolved_conflict_count": 0,
            }
            for domain in domains
        }
        for envelope in envelopes:
            if envelope.domain not in stats:
                continue
            domain_stats = stats[envelope.domain]
            domain_stats["envelope_count"] = int(domain_stats["envelope_count"]) + 1
            domain_stats["latest_server_sequence"] = max(
                int(domain_stats["latest_server_sequence"]),
                int(envelope.server_sequence or 0),
            )
            if envelope.apply_status == "failed":
                domain_stats["failed_apply_count"] = int(domain_stats["failed_apply_count"]) + 1
            object_ids = domain_stats["object_ids"]
            if isinstance(object_ids, set):
                object_ids.add(envelope.object_id)
        for conflict in conflicts:
            if conflict.domain in stats:
                stats[conflict.domain]["unresolved_conflict_count"] = (
                    int(stats[conflict.domain]["unresolved_conflict_count"]) + 1
                )
        return [
            SyncDiagnosticsDomain(
                domain=domain,
                envelope_count=int(domain_stats["envelope_count"]),
                object_count=len(domain_stats["object_ids"]),
                latest_server_sequence=int(domain_stats["latest_server_sequence"]),
                failed_apply_count=int(domain_stats["failed_apply_count"]),
                unresolved_conflict_count=int(domain_stats["unresolved_conflict_count"]),
            )
            for domain, domain_stats in stats.items()
        ]

    def _attachment_lifecycle_diagnostics(
        self,
        *,
        dataset: SyncDataset,
        user_id: str,
        envelopes: Sequence[SyncEnvelope],
        conflicts: Sequence[SyncConflict],
        retention: SyncRetentionDryRunResult,
        sample_limit: int,
        total_sample_limit: int,
    ) -> SyncAttachmentDiagnostics:
        """Build bounded read-only attachment lifecycle counts and safe hints."""

        counts = {
            "registry_total": 0,
            "registry_live": 0,
            "registry_hidden": 0,
            "registry_tombstoned": 0,
            "binding_total": 0,
            "metadata_only": 0,
            "missing": 0,
            "blob_total": 0,
            "available": 0,
            "verify_failed": 0,
            "quarantined": 0,
            "deleting": 0,
            "deleted": 0,
            "orphan": 0,
            "active_uploads": 0,
            "cleanup_candidates": 0,
            "retention_blockers": sum(retention.blocker_counts.values()),
            "projection_pending": 0,
            "projection_failed": 0,
            "unresolved_conflicts": sum(
                conflict.domain == "attachment.ref" for conflict in conflicts
            ),
        }
        if "attachment.ref" in dataset.domains:
            projection = self.store.summarize_domain_envelopes(
                dataset.dataset_id,
                "attachment.ref",
            )
            counts["projection_pending"] = projection.pending_apply_count
            counts["projection_failed"] = projection.failed_apply_count
        sample_buckets: dict[str, list[SyncAttachmentDiagnosticSample]] = {}
        actions: list[SyncRecoveryActionDescriptor] = []

        def descriptor(
            action: str,
            reason_code: str,
            *,
            target_type: str = "dataset",
            target_id: str | None = None,
            requires_confirmation: bool = False,
        ) -> SyncRecoveryActionDescriptor:
            return SyncRecoveryActionDescriptor(
                action=action,
                reason_code=reason_code,
                target_type=target_type,
                target_id=target_id,
                requires_confirmation=requires_confirmation,
            )

        def add_action(action: SyncRecoveryActionDescriptor) -> None:
            action = replace(action, target_type="dataset", target_id=None)
            identity = (
                action.action,
                action.reason_code,
                action.target_type,
                action.target_id,
            )
            if all(
                (
                    item.action,
                    item.reason_code,
                    item.target_type,
                    item.target_id,
                )
                != identity
                for item in actions
            ):
                actions.append(action)

        def add_sample(sample: SyncAttachmentDiagnosticSample) -> None:
            if sample_limit == 0:
                return
            bucket = sample_buckets.setdefault(sample.category, [])
            if len(bucket) < sample_limit:
                bucket.append(sample)

        registry_row = self.store.db.execute(
            """
            SELECT COUNT(*) AS total_count,
                   COALESCE(SUM(CASE WHEN envelope.operation = 'tombstone'
                                     THEN 1 ELSE 0 END), 0) AS tombstoned_count
              FROM sync_current_heads AS head
              JOIN sync_envelopes AS envelope
                ON envelope.server_sequence = head.latest_server_cursor
              JOIN sync_datasets AS dataset ON dataset.dataset_id = head.dataset_id
             WHERE head.dataset_id = ? AND dataset.owner_user_id = ?
               AND head.domain = 'attachment.ref' AND envelope.adapter_version = 2
            """,
            (dataset.dataset_id, dataset.owner_user_id),
        ).rows[0]
        counts["registry_total"] = int(registry_row.get("total_count") or 0)
        counts["registry_tombstoned"] = int(
            registry_row.get("tombstoned_count") or 0
        )
        counts["registry_live"] = (
            counts["registry_total"] - counts["registry_tombstoned"]
        )
        attachment_heads = (
            [
                head
                for head in self.store.list_current_heads(
                    dataset.dataset_id,
                    "attachment.ref",
                    limit=min(self.settings.restore_manifest_scan_limit, 1_000),
                    offset=0,
                )
                if head.adapter_version == 2
            ]
            if "attachment.ref" in dataset.domains
            else []
        )
        counts["registry_scan_truncated"] = int(
            counts["registry_total"] > len(attachment_heads)
        )
        for head in attachment_heads:
            if head.operation == "tombstone":
                category = "registry_tombstoned"
                action = descriptor(
                    "restore_attachment",
                    "sync_attachment_tombstoned",
                    target_type="attachment",
                    target_id=head.object_id,
                )
            else:
                parent_id = None
                payload = head.payload or head.payload_clear
                if isinstance(payload, Mapping):
                    raw_parent = payload.get("parent_object_id")
                    if isinstance(raw_parent, str):
                        parent_id = raw_parent
                parent = (
                    self.store.get_current_head(
                        dataset.dataset_id,
                        "notes.note",
                        parent_id,
                    )
                    if parent_id
                    else None
                )
                if parent is not None and parent.operation == "tombstone":
                    category = "registry_hidden"
                    action = descriptor(
                        "restore_note",
                        "sync_attachment_parent_note_tombstoned",
                        target_type="attachment",
                        target_id=head.object_id,
                    )
                else:
                    category = "registry_live"
                    action = None
            if category == "registry_hidden":
                counts["registry_hidden"] += 1
                counts["registry_live"] -= 1
            sample_actions = [] if action is None else [action]
            if action is not None:
                add_action(action)
            add_sample(
                SyncAttachmentDiagnosticSample(
                    category=category,
                    code=f"sync_attachment_{category}",
                    attachment_id=head.object_id,
                    server_cursor=head.server_cursor,
                    recovery_actions=sample_actions,
                )
            )

        binding_row = self.store.db.execute(
            """
            SELECT COUNT(*) AS total_count,
                   COALESCE(SUM(CASE WHEN binding.resolved_blob_id IS NULL
                                     THEN 1 ELSE 0 END), 0) AS metadata_only_count,
                   COALESCE(SUM(CASE WHEN binding.resolved_blob_id IS NOT NULL
                                          AND blob.blob_id IS NULL
                                     THEN 1 ELSE 0 END), 0) AS missing_count
              FROM sync_attachment_revision_bindings AS binding
              JOIN sync_datasets AS dataset
                ON dataset.dataset_id = binding.dataset_id
              LEFT JOIN sync_blob_objects AS blob
                ON blob.dataset_id = binding.dataset_id
               AND blob.blob_id = binding.resolved_blob_id
             WHERE binding.dataset_id = ? AND dataset.owner_user_id = ?
               AND binding.retention_released_at IS NULL
            """,
            (dataset.dataset_id, dataset.owner_user_id),
        ).rows[0]
        counts["binding_total"] = int(binding_row.get("total_count") or 0)
        counts["metadata_only"] = int(
            binding_row.get("metadata_only_count") or 0
        )
        counts["missing"] = int(binding_row.get("missing_count") or 0)
        for category, action_name, reason_code, predicate in (
            (
                "metadata_only",
                "retry_upload",
                "sync_attachment_blob_metadata_only",
                "binding.resolved_blob_id IS NULL",
            ),
            (
                "missing",
                "retry_upload",
                "sync_attachment_blob_missing",
                "binding.resolved_blob_id IS NOT NULL AND blob.blob_id IS NULL",
            ),
        ):
            if counts[category] == 0:
                continue
            add_action(descriptor(action_name, reason_code))
            if sample_limit == 0:
                continue
            rows = self.store.db.execute(
                f"""
                SELECT binding.attachment_id, binding.establishing_server_cursor,
                       binding.resolved_blob_id
                  FROM sync_attachment_revision_bindings AS binding
                  JOIN sync_datasets AS dataset
                    ON dataset.dataset_id = binding.dataset_id
                  LEFT JOIN sync_blob_objects AS blob
                    ON blob.dataset_id = binding.dataset_id
                   AND blob.blob_id = binding.resolved_blob_id
                 WHERE binding.dataset_id = ? AND dataset.owner_user_id = ?
                   AND binding.retention_released_at IS NULL AND {predicate}
                 ORDER BY binding.establishing_server_cursor, binding.attachment_id
                 LIMIT ?
                """,  # nosec B608 - predicate is selected from fixed literals above.
                (dataset.dataset_id, dataset.owner_user_id, sample_limit),
            ).rows
            for row in rows:
                action = descriptor(
                    action_name,
                    reason_code,
                    target_type="attachment",
                    target_id=str(row["attachment_id"]),
                )
                add_sample(
                    SyncAttachmentDiagnosticSample(
                        category=category,
                        code=reason_code,
                        attachment_id=str(row["attachment_id"]),
                        blob_id=(
                            None
                            if row.get("resolved_blob_id") is None
                            else str(row["resolved_blob_id"])
                        ),
                        server_cursor=int(row["establishing_server_cursor"]),
                        recovery_actions=[action],
                    )
                )

        for row in self.store.db.execute(
            """
            SELECT blob.status, COUNT(*) AS status_count
              FROM sync_blob_objects AS blob
              JOIN sync_datasets AS dataset ON dataset.dataset_id = blob.dataset_id
             WHERE blob.dataset_id = ? AND dataset.owner_user_id = ?
             GROUP BY blob.status
            """,
            (dataset.dataset_id, dataset.owner_user_id),
        ).rows:
            status_name = str(row["status"])
            status_count = int(row.get("status_count") or 0)
            counts["blob_total"] += status_count
            counts[status_name] = counts.get(status_name, 0) + status_count

        blob_actions = {
            "verify_failed": ("retry_verify", "sync_attachment_blob_verify_failed"),
            "quarantined": (
                "release_quarantine",
                "sync_attachment_blob_quarantined",
            ),
            "deleting": ("gc_retry", "sync_attachment_blob_deleting"),
            "deleted": ("retry_upload", "sync_attachment_blob_deleted"),
        }
        for status_name, (action_name, reason_code) in blob_actions.items():
            if counts.get(status_name, 0) == 0:
                continue
            add_action(descriptor(action_name, reason_code))
            if sample_limit == 0:
                continue
            for blob in self.store.list_blob_objects_for_dataset_page(
                dataset.dataset_id,
                status=status_name,
                limit=sample_limit,
            ):
                action = descriptor(
                    action_name,
                    reason_code,
                    target_type="blob",
                    target_id=blob.blob_id,
                )
                add_sample(
                    SyncAttachmentDiagnosticSample(
                        category=status_name,
                        code=reason_code,
                        attachment_id=blob.attachment_id,
                        blob_id=blob.blob_id,
                        recovery_actions=[action],
                    )
                )

        orphan_row = self.store.db.execute(
            """
            SELECT COUNT(*) AS orphan_count
              FROM sync_blob_objects AS blob
              JOIN sync_datasets AS dataset ON dataset.dataset_id = blob.dataset_id
             WHERE blob.dataset_id = ? AND dataset.owner_user_id = ?
               AND blob.status = 'available'
               AND NOT EXISTS (
                    SELECT 1 FROM sync_attachment_revision_bindings AS binding
                     WHERE binding.dataset_id = blob.dataset_id
                       AND binding.resolved_blob_id = blob.blob_id
                       AND binding.retention_released_at IS NULL
               )
            """,
            (dataset.dataset_id, dataset.owner_user_id),
        ).rows[0]
        counts["orphan"] = int(orphan_row.get("orphan_count") or 0)
        if counts["orphan"]:
            add_action(
                descriptor(
                    "wait_for_retention",
                    "sync_attachment_blob_orphaned",
                    requires_confirmation=True,
                )
            )
            if sample_limit:
                rows = self.store.db.execute(
                    """
                    SELECT blob.attachment_id, blob.blob_id
                      FROM sync_blob_objects AS blob
                      JOIN sync_datasets AS dataset
                        ON dataset.dataset_id = blob.dataset_id
                     WHERE blob.dataset_id = ? AND dataset.owner_user_id = ?
                       AND blob.status = 'available'
                       AND NOT EXISTS (
                            SELECT 1
                              FROM sync_attachment_revision_bindings AS binding
                             WHERE binding.dataset_id = blob.dataset_id
                               AND binding.resolved_blob_id = blob.blob_id
                               AND binding.retention_released_at IS NULL
                       )
                     ORDER BY blob.updated_at, blob.blob_id LIMIT ?
                    """,
                    (dataset.dataset_id, dataset.owner_user_id, sample_limit),
                ).rows
                for row in rows:
                    action = descriptor(
                        "wait_for_retention",
                        "sync_attachment_blob_orphaned",
                        target_type="blob",
                        target_id=str(row["blob_id"]),
                        requires_confirmation=True,
                    )
                    add_sample(
                        SyncAttachmentDiagnosticSample(
                            category="orphan",
                            code=action.reason_code,
                            attachment_id=str(row["attachment_id"]),
                            blob_id=str(row["blob_id"]),
                            recovery_actions=[action],
                        )
                    )

        quota = self.store.summarize_blob_quota(user_id, dataset_id=dataset.dataset_id)
        counts["active_uploads"] = quota.active_upload_count
        if counts["active_uploads"]:
            add_action(descriptor("resume_upload", "sync_attachment_upload_incomplete"))
            if sample_limit:
                rows = self.store.db.execute(
                    """
                    SELECT upload.upload_id, upload.attachment_id
                      FROM sync_blob_upload_sessions AS upload
                      JOIN sync_datasets AS dataset
                        ON dataset.dataset_id = upload.dataset_id
                     WHERE upload.dataset_id = ? AND dataset.owner_user_id = ?
                       AND upload.status IN ('created', 'uploading')
                     ORDER BY upload.created_at, upload.upload_id LIMIT ?
                    """,
                    (dataset.dataset_id, dataset.owner_user_id, sample_limit),
                ).rows
                for row in rows:
                    action = descriptor(
                        "resume_upload",
                        "sync_attachment_upload_incomplete",
                        target_type="upload",
                        target_id=str(row["upload_id"]),
                    )
                    add_sample(
                        SyncAttachmentDiagnosticSample(
                            category="active_upload",
                            code=action.reason_code,
                            attachment_id=str(row["attachment_id"]),
                            recovery_actions=[action],
                        )
                    )

        attachment_metadata = dataset.metadata.get("notes_attachment_v2")
        bootstrap_id = (
            attachment_metadata.get("bootstrap_id")
            if isinstance(attachment_metadata, Mapping)
            else None
        )
        if isinstance(bootstrap_id, str) and bootstrap_id:
            cleanup_row = self.store.db.execute(
                """
                SELECT COUNT(*) AS cleanup_count
                  FROM sync_notes_attachment_cleanup_candidates AS cleanup
                  JOIN sync_datasets AS dataset
                    ON dataset.dataset_id = cleanup.dataset_id
                 WHERE cleanup.dataset_id = ? AND dataset.owner_user_id = ?
                   AND cleanup.bootstrap_id = ?
                """,
                (dataset.dataset_id, dataset.owner_user_id, bootstrap_id),
            ).rows[0]
            counts["cleanup_candidates"] = int(
                cleanup_row.get("cleanup_count") or 0
            )
            if counts["cleanup_candidates"]:
                add_action(
                    descriptor(
                        "wait_for_retention",
                        "sync_attachment_cleanup_candidate_retained",
                        requires_confirmation=True,
                    )
                )
            cleanup = (
                self.store.list_notes_attachment_cleanup_candidates(
                    dataset.dataset_id,
                    owner_user_id=dataset.owner_user_id,
                    bootstrap_id=bootstrap_id,
                    limit=sample_limit,
                )
                if sample_limit
                else ()
            )
            for item in cleanup:
                cleanup_action = descriptor(
                    "wait_for_retention",
                    "sync_attachment_cleanup_candidate_retained",
                    target_type="attachment",
                    target_id=item.attachment_id,
                    requires_confirmation=True,
                )
                add_sample(
                    SyncAttachmentDiagnosticSample(
                        category="cleanup_candidate",
                        code=cleanup_action.reason_code,
                        attachment_id=item.attachment_id,
                        recovery_actions=[cleanup_action],
                    )
                )

        if counts["projection_pending"] or counts["projection_failed"]:
            add_action(
                descriptor(
                    "repair_projection",
                    "sync_attachment_projection_incomplete",
                )
            )
            for envelope in envelopes:
                if envelope.domain != "attachment.ref" or envelope.apply_status not in {
                    "pending",
                    "failed",
                }:
                    continue
                action = descriptor(
                    "repair_projection",
                    "sync_attachment_projection_incomplete",
                    target_type="envelope",
                    target_id=envelope.client_envelope_id,
                )
                add_sample(
                    SyncAttachmentDiagnosticSample(
                        category=f"projection_{envelope.apply_status}",
                        code=(
                            envelope.apply_error_code
                            or "sync_attachment_projection_incomplete"
                        ),
                        attachment_id=envelope.object_id,
                        server_cursor=envelope.server_cursor,
                        recovery_actions=[action],
                    )
                )
        if counts["unresolved_conflicts"]:
            add_action(
                descriptor("resolve_conflict", "sync_attachment_conflict_unresolved")
            )
            for conflict in conflicts:
                if conflict.domain != "attachment.ref":
                    continue
                action = descriptor(
                    "resolve_conflict",
                    "sync_attachment_conflict_unresolved",
                    target_type="conflict",
                    target_id=conflict.conflict_id,
                )
                add_sample(
                    SyncAttachmentDiagnosticSample(
                        category="conflict",
                        code="sync_attachment_conflict_unresolved",
                        attachment_id=conflict.entity_id,
                        server_cursor=conflict.server_sequence,
                        recovery_actions=[action],
                    )
                )
        bootstrap_state = (
            str(attachment_metadata.get("state"))
            if isinstance(attachment_metadata, Mapping)
            and attachment_metadata.get("state") is not None
            else "not_started"
        )
        counts[f"bootstrap_{bootstrap_state}"] = 1
        if bootstrap_state in {"initializing", "failed"}:
            add_action(
                descriptor(
                    "bootstrap_resume",
                    "sync_attachment_bootstrap_incomplete",
                )
            )
        if counts["retention_blockers"]:
            add_action(
                descriptor(
                    "wait_for_retention",
                    "sync_attachment_retention_blocked",
                    requires_confirmation=True,
                )
            )

        samples = [sample for bucket in sample_buckets.values() for sample in bucket]
        if len(samples) > total_sample_limit:
            raise SyncStoreError("sync_attachment_diagnostic_total_sample_limit_exceeded")
        return SyncAttachmentDiagnostics(
            counts=counts,
            samples=samples,
            recovery_actions=actions,
        )

    def _diagnostics_devices(
        self,
        *,
        dataset_id: str,
        domains: Sequence[SyncDomain],
        devices: Sequence[SyncDevice],
    ) -> list[SyncDiagnosticsDevice]:
        diagnostics: list[SyncDiagnosticsDevice] = []
        for device in devices:
            domain_statuses = self.store.summarize_background_domains(
                dataset_id,
                device.device_id,
                domains=domains,
            )
            diagnostics.append(
                SyncDiagnosticsDevice(
                    device_id=device.device_id,
                    status=device.status,
                    last_seen_at=device.last_seen_at,
                    domain_lag=[
                        SyncDiagnosticsDeviceDomainLag(
                            domain=status.domain,
                            last_pulled_sequence=status.last_pulled_sequence,
                            latest_server_sequence=status.last_server_sequence,
                            lag_count=status.cursor_lag_count,
                        )
                        for status in domain_statuses
                    ],
                )
            )
        return diagnostics

    def _diagnostics_key_summary(
        self,
        key_records: Sequence[SyncKeyRecord],
    ) -> SyncDiagnosticsKeySummary:
        revoked_count = sum(1 for record in key_records if record.revoked_at is not None)
        superseded_count = sum(
            1 for record in key_records if record.superseded_at is not None
        )
        active_count = sum(
            1
            for record in key_records
            if record.revoked_at is None and record.superseded_at is None
        )
        rewrap_pending_count = sum(
            1 for record in key_records if record.rewrap_status == "pending"
        )
        recovery_available = any(
            record.key_purpose == SYNC_DATASET_RECOVERY_KEY_PURPOSE
            and record.revoked_at is None
            for record in key_records
        )
        return SyncDiagnosticsKeySummary(
            key_record_count=len(key_records),
            active_key_record_count=active_count,
            revoked_key_record_count=revoked_count,
            superseded_key_record_count=superseded_count,
            rewrap_pending_count=rewrap_pending_count,
            recovery_available=recovery_available,
        )

    def _apply_retention_domain_compactions(
        self,
        *,
        dataset: SyncDataset,
        candidates: Sequence[SyncRetentionCandidate],
        minimum_envelope_age_seconds: int,
        minimum_tombstone_age_seconds: int,
        offline_restore_window_seconds: int,
    ) -> tuple[list[dict[str, object]], list[SyncRetentionCandidate]]:
        """Revalidate each domain page under its dataset materialization fence."""

        grouped: dict[SyncDomain, list[SyncRetentionCandidate]] = {}
        for candidate in candidates:
            if candidate.domain is None or candidate.server_sequence is None:
                continue
            grouped.setdefault(candidate.domain, []).append(candidate)
        applied: list[dict[str, object]] = []
        blocked: list[SyncRetentionCandidate] = []
        for domain, domain_candidates in sorted(grouped.items()):
            object_ids = sorted(
                {
                    candidate.object_id
                    for candidate in domain_candidates
                    if candidate.object_id is not None
                }
            )
            if not object_ids:
                continue
            with self.store.retention_domain_guard(
                dataset.dataset_id,
                domain,
                object_ids,
            ) as guarded:
                active_devices = self._retention_active_devices(
                    dataset,
                    store=guarded,
                )
                restore_window_blocked = self._retention_restore_window_active(
                    active_devices,
                    offline_restore_window_seconds,
                )
                revalidated: list[SyncRetentionCandidate] = []
                for candidate in domain_candidates:
                    envelope = (
                        guarded.get_envelope_by_server_cursor(candidate.server_sequence)
                        if candidate.server_sequence is not None
                        else None
                    )
                    current_head = (
                        guarded.get_current_head(
                            dataset.dataset_id,
                            domain,
                            candidate.object_id,
                        )
                        if candidate.object_id is not None
                        else None
                    )
                    candidate_type = (
                        self._retention_envelope_candidate_type(
                            envelope,
                            latest_by_object={
                                (domain, candidate.object_id): current_head
                            },
                        )
                        if envelope is not None
                        and current_head is not None
                        and candidate.object_id is not None
                        else None
                    )
                    blockers: list[str] = []
                    if (
                        envelope is None
                        or envelope.dataset_id != dataset.dataset_id
                        or envelope.domain != domain
                        or envelope.object_id != candidate.object_id
                        or candidate_type != candidate.candidate_type
                    ):
                        blockers.append("retention_candidate_changed")
                    else:
                        window_seconds = (
                            minimum_tombstone_age_seconds
                            if candidate.candidate_type == "tombstone_prune"
                            else minimum_envelope_age_seconds
                        )
                        if self._retention_window_active(
                            envelope.server_timestamp,
                            window_seconds,
                        ):
                            blockers.append(
                                "retention_tombstone_window_active"
                                if candidate.candidate_type == "tombstone_prune"
                                else "retention_envelope_window_active"
                            )
                        if restore_window_blocked:
                            blockers.append("retention_restore_window_active")
                        if self._retention_workspace_ack_scope_blocked(dataset):
                            blockers.append("retention_workspace_ack_scope_unknown")
                        unacknowledged = self._retention_unacknowledged_devices(
                            dataset_id=dataset.dataset_id,
                            domain=domain,
                            adapter_version=envelope.adapter_version,
                            server_sequence=envelope.server_sequence,
                            active_devices=active_devices,
                            store=guarded,
                        )
                        if unacknowledged:
                            blockers.append("retention_unacknowledged_device")
                        blockers.extend(
                            blocker
                            for blocker in self._notes_task_retention_blockers(
                                dataset=dataset,
                                envelope=envelope,
                                store=guarded,
                            )
                            if blocker not in blockers
                        )
                    if blockers:
                        revalidated.append(
                            replace(candidate, blockers=blockers)
                        )
                if revalidated:
                    blocked.extend(revalidated)
                    continue
                through_sequence = max(
                    candidate.server_sequence or 0 for candidate in domain_candidates
                )
                stored_sequence = guarded.record_domain_compaction(
                    dataset.dataset_id,
                    domain,
                    through_server_sequence=through_sequence,
                    state={
                        "compacted_at": self.clock(),
                        "candidate_count": len(domain_candidates),
                        "through_server_sequence": through_sequence,
                    },
                )
            applied.append(
                {
                    "domain": domain,
                    "through_server_sequence": stored_sequence,
                    "candidate_count": len(domain_candidates),
                }
            )
        return applied, blocked

    def _notes_task_retention_blockers(
        self,
        *,
        dataset: SyncDataset,
        envelope: SyncEnvelope,
        store: SyncV2Store | None = None,
    ) -> list[str]:
        """Return exact immutable-anchor and open-drift retention blockers."""

        active_store = store or self.store
        group = [envelope]
        if envelope.mutation_group_id is not None:
            try:
                group = active_store.list_mutation_group(
                    dataset.dataset_id,
                    envelope.mutation_group_id,
                )
                validate_stored_mutation_group(
                    group,
                    dataset_id=dataset.dataset_id,
                    mutation_group_id=envelope.mutation_group_id,
                )
            except (StoredMutationGroupValidationError, SyncStoreError):
                return ["retention_task_projection_repair"]
        task_members = [member for member in group if member.domain == "notes.task"]
        task_materializer = self.materializers.get("notes.task")
        note_db = getattr(task_materializer, "note_db", None)
        task_store = getattr(note_db, "task_store", None)
        blockers: list[str] = []
        if task_store is not None:
            for member in group:
                if member.server_cursor is None or member.payload_hash is None:
                    continue
                if member.domain == "notes.task" and member.object_revision is not None:
                    if task_store.has_open_task_projection_drift_for_task_envelope(
                        owner_user_id=dataset.owner_user_id,
                        dataset_id=dataset.dataset_id,
                        task_id=member.object_id,
                        object_revision=member.object_revision,
                        object_hash=member.payload_hash,
                        server_cursor=member.server_cursor,
                    ):
                        blockers.append("retention_task_projection_drift")
                        break
                if member.domain == "notes.note" and task_store.has_open_task_projection_drift_for_note_envelope(
                    owner_user_id=dataset.owner_user_id,
                    dataset_id=dataset.dataset_id,
                    note_id=member.object_id,
                    object_hash=member.payload_hash,
                    server_cursor=member.server_cursor,
                ):
                    blockers.append("retention_task_projection_drift")
                    break
        if not task_members:
            return blockers
        if any(member.apply_status != "applied" for member in group):
            return ["retention_task_projection_repair"]

        from .notes_task_coordinator import _projection_anchor_from_envelope

        member_ids = {member.client_envelope_id for member in group}
        for task_member in task_members:
            current = active_store.get_current_head(
                dataset.dataset_id,
                "notes.task",
                task_member.object_id,
            )
            current_anchor = (
                _projection_anchor_from_envelope(current)
                if current is not None
                else None
            )
            if (
                current_anchor is not None
                and current_anchor.linked
                and {
                    current_anchor.task_envelope_id,
                    current_anchor.note_envelope_id,
                }
                & member_ids
            ):
                blockers.append("retention_task_projection_anchor")

        if task_store is None:
            return [*blockers, "retention_task_projection_authority_unavailable"]
        return list(dict.fromkeys(blockers))

    def _apply_retention_binding_releases(
        self,
        *,
        dataset: SyncDataset,
        candidates: Sequence[SyncRetentionCandidate],
        minimum_envelope_age_seconds: int,
        minimum_tombstone_age_seconds: int,
        offline_restore_window_seconds: int,
    ) -> tuple[list[dict[str, object]], list[SyncRetentionCandidate]]:
        """Revalidate and monotonically release historical bindings under the fence."""

        applied: list[dict[str, object]] = []
        blocked: list[SyncRetentionCandidate] = []
        for candidate in candidates:
            if candidate.attachment_id is None or candidate.attachment_revision is None:
                continue
            guard_key = candidate.blob_id or candidate.attachment_id
            with self.store.retention_guard(dataset.dataset_id, guard_key) as guarded:
                current_dataset = guarded.get_dataset(
                    dataset.dataset_id,
                    owner_user_id=dataset.owner_user_id,
                )
                binding = guarded.get_attachment_revision_binding(
                    dataset.dataset_id,
                    candidate.attachment_id,
                    candidate.attachment_revision,
                    owner_user_id=dataset.owner_user_id,
                )
                if (
                    current_dataset is None
                    or binding is None
                    or binding.retention_released_at is not None
                ):
                    continue
                current_devices = self._retention_active_devices(
                    current_dataset,
                    store=guarded,
                )
                revalidated = self._retention_binding_release_candidate(
                    dataset=current_dataset,
                    binding=binding,
                    active_devices=current_devices,
                    audit_mode=False,
                    restore_window_blocked=self._retention_restore_window_active(
                        current_devices,
                        offline_restore_window_seconds,
                    ),
                    minimum_envelope_age_seconds=minimum_envelope_age_seconds,
                    minimum_tombstone_age_seconds=minimum_tombstone_age_seconds,
                    store=guarded,
                )
                if revalidated.blockers:
                    blocked.append(revalidated)
                    continue
                released = guarded.release_attachment_revision_binding(
                    dataset.dataset_id,
                    binding.attachment_id,
                    binding.attachment_revision,
                    released_at=self.clock(),
                    owner_user_id=dataset.owner_user_id,
                )
            applied.append(
                {
                    "attachment_id": released.attachment_id,
                    "attachment_revision": released.attachment_revision,
                    "blob_id": released.resolved_blob_id,
                    "payload_hash": released.blob_hash,
                    "size_bytes": released.size_bytes,
                    "establishing_server_cursor": released.establishing_server_cursor,
                }
            )
        return applied, blocked

    def _apply_retention_blob_gc(
        self,
        *,
        dataset: SyncDataset,
        candidates: Sequence[SyncRetentionCandidate],
        minimum_tombstone_age_seconds: int,
        offline_restore_window_seconds: int,
    ) -> tuple[
        list[dict[str, object]],
        list[SyncRetentionCandidate],
        bool,
    ]:
        applied: list[dict[str, object]] = []
        blocked: list[SyncRetentionCandidate] = []
        fence_mutated = False
        for candidate in candidates:
            if candidate.blob_id is None:
                continue
            namespace_id: str | None = None
            with self.store.retention_guard(
                dataset.dataset_id,
                candidate.blob_id,
            ) as guarded:
                current_dataset = guarded.get_dataset(
                    dataset.dataset_id,
                    owner_user_id=dataset.owner_user_id,
                )
                blob = guarded.lock_blob_object_for_retention(
                    dataset.dataset_id,
                    candidate.blob_id,
                    owner_user_id=dataset.owner_user_id,
                )
                if current_dataset is None or blob is None:
                    continue
                if blob.status == "deleted":
                    continue
                active_devices = (
                    []
                    if blob.status == "deleting"
                    else self._retention_active_devices(
                        current_dataset,
                        store=guarded,
                    )
                )
                revalidated = self._retention_blob_candidate(
                    dataset=current_dataset,
                    blob=blob,
                    active_devices=active_devices,
                    audit_mode=False,
                    restore_window_blocked=(
                        False
                        if blob.status == "deleting"
                        else self._retention_restore_window_active(
                            active_devices,
                            offline_restore_window_seconds,
                        )
                    ),
                    minimum_tombstone_age_seconds=minimum_tombstone_age_seconds,
                    store=guarded,
                )
                if revalidated.blockers:
                    blocked.append(revalidated)
                    continue
                namespace = guarded.get_storage_namespace(
                    dataset.dataset_id,
                    owner_user_id=dataset.owner_user_id,
                )
                if namespace is None:
                    blocked.append(
                        replace(
                            revalidated,
                            blockers=["retention_blob_storage_key_not_namespaced"],
                        )
                    )
                    continue
                namespace_id = namespace.storage_namespace_id
                if blob.status == "available":
                    blob = guarded.fence_blob_object_deleting(
                        dataset.dataset_id,
                        candidate.blob_id,
                    )
                    if blob is None or blob.status != "deleting":
                        continue
                    fence_mutated = True
                elif blob.status != "deleting":
                    continue
            if self.blob_store is None or namespace_id is None:
                blocked.append(
                    replace(
                        candidate,
                        blockers=["retention_blob_storage_unavailable"],
                    )
                )
                continue
            try:
                self.blob_store.delete_namespace_blob(
                    storage_key=blob.storage_key,
                    storage_namespace_id=namespace_id,
                    payload_hash=blob.payload_hash,
                    expected_size=blob.size_bytes,
                )
            except SyncBlobStoreError:
                blocked.append(
                    replace(
                        candidate,
                        blockers=["retention_blob_delete_retry"],
                        reason="physical blob deletion retry",
                    )
                )
                continue
            with self.store.retention_guard(
                dataset.dataset_id,
                candidate.blob_id,
            ) as guarded:
                current = guarded.lock_blob_object_for_retention(
                    dataset.dataset_id,
                    candidate.blob_id,
                    owner_user_id=dataset.owner_user_id,
                )
                if current is None or current.status == "deleted":
                    continue
                if current.status != "deleting":
                    continue
                blob = guarded.finalize_blob_object_deleted(
                    dataset.dataset_id,
                    candidate.blob_id,
                )
                if blob is None or blob.status != "deleted":
                    continue
            applied.append(
                {
                    "attachment_id": blob.attachment_id,
                    "blob_id": blob.blob_id,
                    "payload_hash": blob.payload_hash,
                    "size_bytes": blob.size_bytes,
                }
            )
        return applied, blocked, fence_mutated

    def _retention_active_devices(
        self,
        dataset: SyncDataset,
        *,
        store: SyncV2Store | None = None,
    ) -> list[SyncDevice]:
        devices = (store or self.store).list_devices_for_user(dataset.owner_user_id)
        return sorted(
            [
                device
                for device in devices
                if device.status == "active" and device.revoked_at is None
            ],
            key=lambda device: device.device_id,
        )

    def _retention_workspace_ack_scope_blocked(self, dataset: SyncDataset) -> bool:
        return dataset.scope_type == "workspace"

    def _retention_latest_envelopes_by_object(
        self,
        envelopes: Sequence[SyncEnvelope],
    ) -> dict[tuple[SyncDomain, str], SyncEnvelope]:
        latest: dict[tuple[SyncDomain, str], SyncEnvelope] = {}
        for envelope in envelopes:
            key = (envelope.domain, envelope.object_id)
            existing = latest.get(key)
            if existing is None or envelope.server_sequence > existing.server_sequence:
                latest[key] = envelope
        return latest

    def _retention_envelope_candidate_type(
        self,
        envelope: SyncEnvelope,
        *,
        latest_by_object: Mapping[tuple[SyncDomain, str], SyncEnvelope],
    ) -> str | None:
        latest = latest_by_object.get((envelope.domain, envelope.object_id))
        if latest is None:
            return None
        if envelope.operation == "tombstone" or envelope.deleted:
            return "tombstone_prune"
        if latest.operation == "tombstone" or latest.deleted:
            return None
        if latest.server_sequence != envelope.server_sequence:
            return "envelope_compaction"
        return None

    def _retention_restore_window_active(
        self,
        active_devices: Sequence[SyncDevice],
        offline_restore_window_seconds: int,
    ) -> bool:
        if offline_restore_window_seconds <= 0:
            return False
        return any(
            self._retention_window_active(
                device.last_seen_at,
                offline_restore_window_seconds,
            )
            for device in active_devices
        )

    def _retention_unacknowledged_devices(
        self,
        *,
        dataset_id: str,
        domain: SyncDomain,
        adapter_version: int,
        server_sequence: int,
        active_devices: Sequence[SyncDevice],
        store: SyncV2Store | None = None,
    ) -> list[str]:
        active_store = store or self.store
        unacknowledged: list[str] = []
        for device in active_devices:
            if not _device_supports_adapter_version(
                device,
                domain,
                adapter_version,
            ):
                continue
            ack = active_store.get_device_domain_ack(
                dataset_id,
                device.device_id,
                domain,
                adapter_version=adapter_version,
            )
            if ack is None or ack.through_server_sequence < server_sequence:
                unacknowledged.append(device.device_id)
        return unacknowledged

    def _retention_blob_candidates(
        self,
        *,
        dataset: SyncDataset,
        active_devices: Sequence[SyncDevice],
        audit_mode: bool,
        restore_window_blocked: bool,
        minimum_tombstone_age_seconds: int,
        limit: int,
        store: SyncV2Store | None = None,
    ) -> list[SyncRetentionCandidate]:
        active_store = store or self.store
        candidates: list[SyncRetentionCandidate] = []
        for status in ("deleting", "available"):
            remaining = limit - len(candidates)
            if remaining <= 0:
                break
            for blob in active_store.list_blob_objects_for_dataset_page(
                dataset.dataset_id,
                status=status,
                limit=remaining,
            ):
                candidates.append(
                    self._retention_blob_candidate(
                        dataset=dataset,
                        blob=blob,
                        active_devices=active_devices,
                        audit_mode=audit_mode,
                        restore_window_blocked=restore_window_blocked,
                        minimum_tombstone_age_seconds=minimum_tombstone_age_seconds,
                        store=active_store,
                    )
                )
        return candidates

    def _retention_binding_release_candidates(
        self,
        *,
        dataset: SyncDataset,
        active_devices: Sequence[SyncDevice],
        audit_mode: bool,
        restore_window_blocked: bool,
        minimum_envelope_age_seconds: int,
        minimum_tombstone_age_seconds: int,
        limit: int,
        store: SyncV2Store | None = None,
    ) -> list[SyncRetentionCandidate]:
        """Return a bounded keyset scan of unreleased binding candidates."""

        active_store = store or self.store
        candidates: list[SyncRetentionCandidate] = []
        after_cursor = 0
        after_attachment_id = ""
        after_attachment_revision = 0
        while len(candidates) < limit:
            page_limit = min(SYNC_RETENTION_BINDING_PAGE_SIZE, limit - len(candidates))
            bindings = active_store.list_unreleased_attachment_revision_bindings(
                dataset.dataset_id,
                owner_user_id=dataset.owner_user_id,
                after_establishing_server_cursor=after_cursor,
                after_attachment_id=after_attachment_id,
                after_attachment_revision=after_attachment_revision,
                limit=page_limit,
            )
            if not bindings:
                break
            candidates.extend(
                self._retention_binding_release_candidate(
                    dataset=dataset,
                    binding=binding,
                    active_devices=active_devices,
                    audit_mode=audit_mode,
                    restore_window_blocked=restore_window_blocked,
                    minimum_envelope_age_seconds=minimum_envelope_age_seconds,
                    minimum_tombstone_age_seconds=minimum_tombstone_age_seconds,
                    store=active_store,
                )
                for binding in bindings
            )
            last = bindings[-1]
            after_cursor = last.establishing_server_cursor
            after_attachment_id = last.attachment_id
            after_attachment_revision = last.attachment_revision
            if len(bindings) < page_limit:
                break
        return candidates

    def _retention_binding_release_candidate(
        self,
        *,
        dataset: SyncDataset,
        binding: SyncAttachmentRevisionBinding,
        active_devices: Sequence[SyncDevice],
        audit_mode: bool,
        restore_window_blocked: bool,
        minimum_envelope_age_seconds: int,
        minimum_tombstone_age_seconds: int,
        store: SyncV2Store | None = None,
    ) -> SyncRetentionCandidate:
        """Evaluate immutable binding evidence without mutable reference counters."""

        active_store = store or self.store
        blockers: list[str] = []
        if audit_mode:
            blockers.append("retention_audit_mode")
        if restore_window_blocked:
            blockers.append("retention_restore_window_active")
        if self._retention_workspace_ack_scope_blocked(dataset):
            blockers.append("retention_workspace_ack_scope_unknown")

        envelope = active_store.get_envelope_by_server_cursor(
            binding.establishing_server_cursor
        )
        envelope_valid = (
            envelope is not None
            and envelope.dataset_id == dataset.dataset_id
            and envelope.domain == "attachment.ref"
            and envelope.adapter_version == 2
            and envelope.object_id == binding.attachment_id
            and envelope.object_revision == binding.attachment_revision
        )
        if not envelope_valid:
            blockers.append("retention_blob_binding_invalid")
        elif self._retention_window_active(
            envelope.server_timestamp,
            minimum_envelope_age_seconds,
        ):
            blockers.append("retention_envelope_window_active")

        head = active_store.get_current_head(
            dataset.dataset_id,
            "attachment.ref",
            binding.attachment_id,
        )
        state = active_store.get_object_state(
            dataset.dataset_id,
            "attachment.ref",
            binding.attachment_id,
        )
        current_revision = None if head is None else head.object_revision
        if (
            head is None
            or current_revision is None
            or current_revision < binding.attachment_revision
        ):
            blockers.append("retention_blob_binding_invalid")
        current_binding = current_revision == binding.attachment_revision
        current_live = current_binding and (
            (state is not None and not state.deleted)
            or (head is not None and head.operation != "tombstone" and not head.deleted)
        )
        if current_live:
            blockers.append("retention_active_blob_reference")
        elif head is not None and (
            head.operation == "tombstone" or head.deleted
        ) and self._retention_window_active(
            head.server_timestamp,
            minimum_tombstone_age_seconds,
        ):
            blockers.append("retention_tombstone_window_active")

        required_devices = [
            device
            for device in active_devices
            if _device_supports_adapter_version(device, "attachment.ref", 2)
        ]
        ref_unacknowledged: list[str] = []
        blob_unacknowledged: list[str] = []
        blob: SyncBlobObject | None = None
        if not current_live:
            ref_unacknowledged = self._retention_unacknowledged_devices(
                dataset_id=dataset.dataset_id,
                domain="attachment.ref",
                adapter_version=2,
                server_sequence=binding.establishing_server_cursor,
                active_devices=required_devices,
                store=active_store,
            )
            if ref_unacknowledged:
                blockers.append("retention_blob_ref_unacknowledged")
            if binding.resolved_blob_id is not None:
                blob = active_store.get_blob_object(
                    dataset.dataset_id,
                    blob_id=binding.resolved_blob_id,
                    owner_user_id=dataset.owner_user_id,
                    include_unavailable=True,
                )
                if (
                    blob is None
                    or blob.payload_hash != binding.blob_hash
                    or blob.size_bytes != binding.size_bytes
                ):
                    blockers.append("retention_blob_binding_invalid")
                elif blob.status == "quarantined":
                    blockers.append("retention_blob_quarantined")
                elif blob.status not in {"available", "deleted"}:
                    blockers.append("retention_blob_repair_pending")
                blob_unacknowledged = self._retention_blob_unacknowledged_devices(
                    dataset_id=dataset.dataset_id,
                    attachment_id=binding.attachment_id,
                    blob_id=binding.resolved_blob_id,
                    payload_hash=binding.blob_hash,
                    adapter_version=2,
                    active_devices=required_devices,
                    store=active_store,
                )
                if blob_unacknowledged:
                    blockers.append("retention_blob_unverified_by_device")

        return SyncRetentionCandidate(
            candidate_type="binding_release",
            dataset_id=dataset.dataset_id,
            domain="attachment.ref",
            object_id=binding.attachment_id,
            server_sequence=binding.establishing_server_cursor,
            blob_id=binding.resolved_blob_id,
            attachment_id=binding.attachment_id,
            attachment_revision=binding.attachment_revision,
            payload_hash=binding.blob_hash,
            size_bytes=binding.size_bytes,
            blockers=list(dict.fromkeys(blockers)),
            required_device_ids=[device.device_id for device in required_devices],
            unacknowledged_device_ids=sorted(
                set(ref_unacknowledged) | set(blob_unacknowledged)
            ),
            reason="historical attachment revision binding",
        )

    def _retention_blob_candidate(
        self,
        *,
        dataset: SyncDataset,
        blob: SyncBlobObject,
        active_devices: Sequence[SyncDevice],
        audit_mode: bool,
        restore_window_blocked: bool,
        minimum_tombstone_age_seconds: int,
        store: SyncV2Store | None = None,
    ) -> SyncRetentionCandidate:
        active_store = store or self.store
        blockers: list[str] = []
        if audit_mode:
            blockers.append("retention_audit_mode")
        blockers.extend(
            self._retention_blob_storage_blockers(
                dataset=dataset,
                blob=blob,
                store=active_store,
            )
        )
        if blob.status == "deleting":
            return SyncRetentionCandidate(
                candidate_type="blob_gc",
                dataset_id=dataset.dataset_id,
                domain="attachment.ref",
                object_id=blob.attachment_id,
                blob_id=blob.blob_id,
                attachment_id=blob.attachment_id,
                payload_hash=blob.payload_hash,
                size_bytes=blob.size_bytes,
                blockers=list(dict.fromkeys(blockers)),
                reason="physical blob deletion retry",
            )
        if restore_window_blocked:
            blockers.append("retention_restore_window_active")
        if self._retention_workspace_ack_scope_blocked(dataset):
            blockers.append("retention_workspace_ack_scope_unknown")
        adapter_version, binding_cursor, binding_valid, binding_protected = (
            self._retention_blob_adapter_version(
                dataset=dataset,
                blob=blob,
                store=active_store,
            )
        )
        if self._retention_blob_tombstone_window_active(
            dataset=dataset,
            blob=blob,
            adapter_version=adapter_version,
            window_seconds=minimum_tombstone_age_seconds,
            store=active_store,
        ):
            blockers.append("retention_tombstone_window_active")
        if adapter_version == 2:
            if binding_protected:
                blockers.append("retention_active_blob_reference")
        else:
            state = active_store.get_object_state(
                dataset.dataset_id,
                "attachment.ref",
                blob.attachment_id,
            )
            if (
                (state is not None and not state.deleted)
                or self._retention_attachment_ref_active(
                    dataset_id=dataset.dataset_id,
                    attachment_id=blob.attachment_id,
                    store=active_store,
                )
            ):
                blockers.append("retention_active_blob_reference")
        required_devices = (
            [
                device
                for device in active_devices
                if _device_supports_adapter_version(
                    device,
                    "attachment.ref",
                    adapter_version,
                )
            ]
            if adapter_version == 2
            else list(active_devices)
        )
        if not binding_valid:
            blockers.append("retention_blob_binding_invalid")
        ref_unacknowledged = (
            self._retention_unacknowledged_devices(
                dataset_id=dataset.dataset_id,
                domain="attachment.ref",
                adapter_version=2,
                server_sequence=binding_cursor,
                active_devices=required_devices,
                store=active_store,
            )
            if adapter_version == 2
            and binding_valid
            and binding_cursor is not None
            else []
        )
        if ref_unacknowledged:
            blockers.append("retention_blob_ref_unacknowledged")
        blob_unacknowledged = self._retention_blob_unacknowledged_devices(
            dataset_id=dataset.dataset_id,
            attachment_id=blob.attachment_id,
            blob_id=blob.blob_id,
            payload_hash=blob.payload_hash,
            adapter_version=adapter_version,
            active_devices=required_devices,
            store=active_store,
        )
        if blob_unacknowledged:
            blockers.append("retention_blob_unverified_by_device")
        return SyncRetentionCandidate(
            candidate_type="blob_gc",
            dataset_id=dataset.dataset_id,
            domain="attachment.ref",
            object_id=blob.attachment_id,
            blob_id=blob.blob_id,
            attachment_id=blob.attachment_id,
            payload_hash=blob.payload_hash,
            size_bytes=blob.size_bytes,
            blockers=blockers,
            required_device_ids=[device.device_id for device in required_devices],
            unacknowledged_device_ids=sorted(
                set(ref_unacknowledged) | set(blob_unacknowledged)
            ),
            reason="server blob retained for attachment restore",
        )

    def _retention_blob_storage_blockers(
        self,
        *,
        dataset: SyncDataset,
        blob: SyncBlobObject,
        store: SyncV2Store,
    ) -> list[str]:
        if self.blob_store is None:
            return ["retention_blob_storage_unavailable"]
        namespace = store.get_storage_namespace(
            dataset.dataset_id,
            owner_user_id=dataset.owner_user_id,
        )
        if namespace is None:
            return ["retention_blob_storage_key_not_namespaced"]
        try:
            expected_key = self.blob_store.namespace_storage_key(
                namespace.storage_namespace_id,
                blob.payload_hash,
            )
        except SyncBlobStoreError:
            return ["retention_blob_storage_key_not_namespaced"]
        if (
            blob.storage_backend != self.settings.blob_storage_backend
            or blob.storage_key != expected_key
        ):
            return ["retention_blob_storage_key_not_namespaced"]
        return []

    def _retention_blob_adapter_version(
        self,
        *,
        dataset: SyncDataset,
        blob: SyncBlobObject,
        store: SyncV2Store | None = None,
    ) -> tuple[int, int | None, bool, bool]:
        active_store = store or self.store
        after_cursor = 0
        after_attachment_id = ""
        after_attachment_revision = 0
        binding_cursor: int | None = None
        binding_valid = True
        binding_protected = False
        found_unreleased = False
        while True:
            bindings = active_store.list_attachment_revision_bindings_for_blob(
                dataset.dataset_id,
                blob.blob_id,
                owner_user_id=dataset.owner_user_id,
                after_establishing_server_cursor=after_cursor,
                after_attachment_id=after_attachment_id,
                after_attachment_revision=after_attachment_revision,
                limit=SYNC_RETENTION_BINDING_PAGE_SIZE,
            )
            if not bindings:
                break
            found_unreleased = True
            for binding in bindings:
                binding_cursor = max(
                    binding_cursor or 0,
                    binding.establishing_server_cursor,
                )
                binding_valid = binding_valid and self._retention_blob_binding_valid(
                    dataset=dataset,
                    blob=blob,
                    binding=binding,
                    store=active_store,
                )
                binding_protected = (
                    binding_protected
                    or self._retention_v2_binding_protected(
                        dataset_id=dataset.dataset_id,
                        binding=binding,
                        store=active_store,
                    )
                )
            last_binding = bindings[-1]
            after_cursor = last_binding.establishing_server_cursor
            after_attachment_id = last_binding.attachment_id
            after_attachment_revision = last_binding.attachment_revision
            if len(bindings) < SYNC_RETENTION_BINDING_PAGE_SIZE:
                break
        if found_unreleased:
            return 2, binding_cursor, binding_valid, binding_protected

        binding = active_store.get_attachment_revision_binding_for_blob(
            dataset.dataset_id,
            blob.blob_id,
            owner_user_id=dataset.owner_user_id,
        )
        if binding is not None:
            return (
                2,
                None,
                self._retention_blob_binding_valid(
                    dataset=dataset,
                    blob=blob,
                    binding=binding,
                    store=active_store,
                ),
                False,
            )

        has_v2_history = active_store.has_attachment_ref_v2_history(
            dataset.dataset_id,
            blob.attachment_id,
            owner_user_id=dataset.owner_user_id,
        )
        return (2, None, False, False) if has_v2_history else (1, None, True, False)

    def _retention_blob_binding_valid(
        self,
        *,
        dataset: SyncDataset,
        blob: SyncBlobObject,
        binding: SyncAttachmentRevisionBinding,
        store: SyncV2Store | None = None,
    ) -> bool:
        envelope = (store or self.store).get_envelope_by_server_cursor(
            binding.establishing_server_cursor
        )
        return (
            envelope is not None
            and envelope.dataset_id == dataset.dataset_id
            and envelope.domain == "attachment.ref"
            and envelope.adapter_version == 2
            and envelope.object_id == binding.attachment_id
            and envelope.object_revision == binding.attachment_revision
            and binding.resolved_blob_id == blob.blob_id
            and binding.blob_hash == blob.payload_hash
            and binding.size_bytes == blob.size_bytes
        )

    def _retention_blob_tombstone_window_active(
        self,
        *,
        dataset: SyncDataset,
        blob: SyncBlobObject,
        adapter_version: int,
        window_seconds: int,
        store: SyncV2Store | None = None,
    ) -> bool:
        if window_seconds <= 0:
            return False
        active_store = store or self.store
        attachment_ids: set[str] = set()
        if adapter_version == 2:
            after_cursor = 0
            after_attachment_id = ""
            after_attachment_revision = 0
            while True:
                bindings = active_store.list_attachment_revision_bindings_for_blob(
                    dataset.dataset_id,
                    blob.blob_id,
                    owner_user_id=dataset.owner_user_id,
                    after_establishing_server_cursor=after_cursor,
                    after_attachment_id=after_attachment_id,
                    after_attachment_revision=after_attachment_revision,
                    limit=SYNC_RETENTION_BINDING_PAGE_SIZE,
                )
                if not bindings:
                    break
                attachment_ids.update(binding.attachment_id for binding in bindings)
                last_binding = bindings[-1]
                after_cursor = last_binding.establishing_server_cursor
                after_attachment_id = last_binding.attachment_id
                after_attachment_revision = last_binding.attachment_revision
                if len(bindings) < SYNC_RETENTION_BINDING_PAGE_SIZE:
                    break
        else:
            attachment_ids.add(blob.attachment_id)
        for attachment_id in attachment_ids:
            head = active_store.get_current_head(
                dataset.dataset_id,
                "attachment.ref",
                attachment_id,
            )
            if head is not None and self._retention_window_active(
                head.server_timestamp,
                window_seconds,
            ):
                return True
        return False

    def _retention_v2_binding_protected(
        self,
        *,
        dataset_id: str,
        binding: SyncAttachmentRevisionBinding,
        store: SyncV2Store | None = None,
    ) -> bool:
        active_store = store or self.store
        state = active_store.get_object_state(
            dataset_id,
            "attachment.ref",
            binding.attachment_id,
        )
        head = active_store.get_current_head(
            dataset_id,
            "attachment.ref",
            binding.attachment_id,
        )
        if head is None:
            return True
        if (
            head.dataset_id != dataset_id
            or head.domain != "attachment.ref"
            or head.object_id != binding.attachment_id
        ):
            return True
        return (state is not None and not state.deleted) or (
            head.operation != "tombstone" and not head.deleted
        )

    def _retention_attachment_ref_active(
        self,
        *,
        dataset_id: str,
        attachment_id: str,
        store: SyncV2Store | None = None,
    ) -> bool:
        envelopes = (store or self.store).list_envelopes_for_entity(
            dataset_id,
            "attachment.ref",
            entity_id=attachment_id,
            limit=1,
        )
        if not envelopes:
            return False
        latest = envelopes[0]
        return latest.operation != "tombstone" and not latest.deleted

    def _retention_blob_unacknowledged_devices(
        self,
        *,
        dataset_id: str,
        attachment_id: str,
        blob_id: str,
        payload_hash: str,
        adapter_version: int,
        active_devices: Sequence[SyncDevice],
        store: SyncV2Store | None = None,
    ) -> list[str]:
        active_store = store or self.store
        unacknowledged: list[str] = []
        for device in active_devices:
            summary = active_store.list_device_acknowledgments(
                dataset_id,
                device.device_id,
            )
            acknowledged = (
                any(
                    ack.blob_id == blob_id and ack.payload_hash == payload_hash
                    for ack in summary.blob_id_acks
                )
                if adapter_version == 2
                else any(
                    ack.attachment_id == attachment_id and ack.payload_hash == payload_hash
                    for ack in summary.blob_acks
                )
            )
            if not acknowledged:
                unacknowledged.append(device.device_id)
        return unacknowledged

    def _retention_window_active(
        self,
        server_timestamp: str | None,
        minimum_age_seconds: int,
    ) -> bool:
        if minimum_age_seconds <= 0:
            return False
        timestamp = _parse_sync_timestamp(server_timestamp)
        now = _parse_sync_timestamp(self.clock())
        if timestamp is None or now is None:
            return True
        return (now - timestamp).total_seconds() < minimum_age_seconds

    def _require_server_trusted_encryption_ready(self) -> None:
        if not self.settings.server_trusted_encryption.encryption.get("ready", False):
            raise SyncStoreError(
                "sync_encryption_attestation_required: Sync v2 M1 requires "
                "server_trusted_v1 at-rest encryption readiness before dataset enrollment"
            )

    def _payload_exceeds_size_limit(self, envelope: SyncEnvelopeCreate) -> bool:
        max_bytes = self.settings.max_envelope_payload_bytes
        if envelope.payload_size_bytes is not None and envelope.payload_size_bytes > max_bytes:
            return True
        actual_size = 0
        if envelope.payload_ciphertext is not None:
            actual_size += len(envelope.payload_ciphertext.encode("utf-8"))
        payload_size = _compact_json_size(envelope.payload)
        payload_clear_size = _compact_json_size(envelope.payload_clear)
        if envelope.payload and envelope.payload_clear and envelope.payload != envelope.payload_clear:
            actual_size += payload_size + payload_clear_size
        else:
            actual_size += max(payload_size, payload_clear_size)
        actual_size += _compact_json_size(envelope.routing_metadata)
        actual_size += _compact_json_size(envelope.dependencies)
        return actual_size > max_bytes

    def _store_conflict(
        self,
        dataset: SyncDataset,
        envelope: SyncEnvelopeCreate,
        outcome: AdapterConflict,
        *,
        store: SyncV2Store | None = None,
    ) -> SyncPushConflict:
        active_store = store or self.store
        storage_envelope = self._protect_personal_context_for_storage(
            dataset,
            replace(envelope, status="conflict"),
        )
        inserted = active_store.insert_envelope(storage_envelope)
        existing = active_store.get_unresolved_conflict_for_envelope(
            dataset.dataset_id,
            local_envelope_id=envelope.client_envelope_id,
            server_sequence=inserted.server_sequence,
        )
        if existing is not None:
            return SyncPushConflict(
                conflict_id=existing.conflict_id,
                client_envelope_id=outcome.client_envelope_id,
                domain=existing.domain,
                entity_id=existing.entity_id,
                server_sequence=existing.server_sequence,
                message=outcome.message,
            )
        conflict = active_store.insert_conflict(
            SyncConflictCreate(
                conflict_id=self.id_factory("conflict"),
                dataset_id=dataset.dataset_id,
                domain=outcome.domain,
                entity_id=outcome.entity_id,
                conflict_type=outcome.conflict_type,
                local_envelope_id=envelope.client_envelope_id,
                server_sequence=inserted.server_sequence,
                metadata=dict(outcome.metadata),
            )
        )
        return SyncPushConflict(
            conflict_id=conflict.conflict_id,
            client_envelope_id=outcome.client_envelope_id,
            domain=outcome.domain,
            entity_id=outcome.entity_id,
            server_sequence=inserted.server_sequence,
            message=outcome.message,
        )

    def _store_preflight_conflict(
        self,
        dataset: SyncDataset,
        envelope: SyncEnvelopeCreate,
        outcome: AdapterConflict,
    ) -> SyncPushConflict:
        """Atomically prefer an accepted projection blocker over preflight history."""

        with self.store.materialization_guard(
            [envelope],
            require_predecessors=False,
        ) as guarded_store:
            blocker = guarded_store.get_unresolved_materialization_conflict(
                dataset.dataset_id
            )
            if blocker is not None:
                return SyncPushConflict(
                    conflict_id=blocker.conflict_id,
                    client_envelope_id=envelope.client_envelope_id,
                    domain=blocker.domain,
                    entity_id=blocker.entity_id,
                    server_sequence=blocker.server_sequence,
                    message=(
                        "An unresolved materialization conflict must be resolved "
                        "before appending more changes"
                    ),
                )
            return self._store_conflict(
                dataset,
                envelope,
                outcome,
                store=guarded_store,
            )

    def _materialize_envelope(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store | None = None,
        guarded_mutation: GuardedProductMutation | None = None,
    ) -> MaterializationResult:
        materializer = self.materializers.get(envelope.domain)
        if materializer is None:
            return MaterializationResult(status="skipped")
        if envelope.apply_status == "superseded":
            return MaterializationResult(status="skipped")
        if store is None:
            try:
                with self.store.materialization_guard([envelope]) as guarded_store:
                    return self._materialize_envelope(
                        envelope,
                        store=guarded_store,
                        guarded_mutation=guarded_mutation,
                    )
            except SyncMaterializationBusyError:
                return MaterializationResult(
                    status="failed",
                    error_code="sync_projection_busy",
                    message="Projection is busy; retry later",
                )
            except SyncMaterializationPredecessorError:
                return MaterializationResult(
                    status="failed",
                    error_code="sync_projection_predecessor_unresolved",
                    message="An earlier projection must finish first",
                )
            except Exception as exc:  # noqa: BLE001 - commit/lock failures are retryable.
                return MaterializationResult(
                    status="failed",
                    error_code="sync_projection_failed",
                    message=_safe_projection_error_message(exc),
                )
        try:
            clear_envelope = envelope
            if envelope.domain in PERSONAL_CONTEXT_SYNC_DOMAINS:
                dataset = store.get_dataset(envelope.dataset_id)
                if dataset is None:
                    raise SyncStoreError("Sync dataset was not found")
                clear_envelope = self._restore_personal_context_from_storage(
                    dataset,
                    envelope,
                )
            if guarded_mutation is None:
                result = materializer.apply(clear_envelope, store=store)
            else:
                guarded_mutation.require_identity(envelope.domain, envelope.object_id)
                result = materializer.apply(
                    clear_envelope,
                    store=store,
                    guarded_mutation=guarded_mutation,
                )
            if result.status == "conflict":
                self._store_materialization_conflict(
                    envelope.dataset_id,
                    self._envelope_snapshot(envelope, store=store),
                    result,
                    store=store,
                )
            return result
        except Exception as exc:  # noqa: BLE001 - materializer failures are captured as replayable sync state.
            error_code = "sync_projection_failed"
            error_message = _safe_projection_error_message(exc)
            if envelope.server_cursor is not None:
                store.mark_envelope_apply_status(
                    envelope.server_cursor,
                    apply_status="failed",
                    apply_error_code=error_code,
                    apply_error_message=error_message,
                )
            return MaterializationResult(
                status="failed",
                error_code=error_code,
                message=error_message,
            )

    def _envelope_snapshot(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store | None = None,
    ) -> SyncEnvelope:
        """Reload an envelope after projection updates apply status fields."""

        if envelope.server_cursor is None:
            return envelope
        candidates = (store or self.store).list_envelopes_after(
            envelope.dataset_id,
            max(envelope.server_cursor - 1, 0),
            limit=1,
            domains=[envelope.domain],
            status=None,
        )
        for candidate in candidates:
            if candidate.server_cursor == envelope.server_cursor:
                return candidate
        return envelope

    def _push_accepted_from_envelope(self, envelope: SyncEnvelope) -> SyncPushAccepted:
        object_revision = envelope.object_revision
        state = self.store.get_object_state(
            envelope.dataset_id,
            envelope.domain,
            envelope.object_id,
        )
        if state is not None and state.latest_server_cursor == envelope.server_cursor:
            object_revision = state.object_revision
        return SyncPushAccepted(
            client_envelope_id=envelope.client_envelope_id,
            server_sequence=envelope.server_sequence,
            domain=envelope.domain,
            entity_id=envelope.entity_id,
            object_revision=object_revision,
            apply_status=envelope.apply_status,
            apply_error_code=envelope.apply_error_code,
            apply_error_message=envelope.apply_error_message,
        )

    def _store_materialization_conflict(
        self,
        dataset: SyncDataset | str,
        envelope: SyncEnvelope,
        result: MaterializationResult,
        *,
        store: SyncV2Store | None = None,
    ) -> SyncPushConflict:
        sync_store = store or self.store
        dataset_id = dataset.dataset_id if isinstance(dataset, SyncDataset) else dataset
        existing = sync_store.get_unresolved_conflict_for_envelope(
            dataset_id,
            local_envelope_id=envelope.client_envelope_id,
            server_sequence=envelope.server_sequence,
        )
        if existing is not None:
            return SyncPushConflict(
                conflict_id=existing.conflict_id,
                client_envelope_id=envelope.client_envelope_id,
                domain=existing.domain,
                entity_id=existing.entity_id,
                server_sequence=existing.server_sequence,
                message=result.message,
            )
        conflict = sync_store.insert_conflict(
            SyncConflictCreate(
                conflict_id=self.id_factory("conflict"),
                dataset_id=dataset_id,
                domain=envelope.domain,
                entity_id=envelope.entity_id,
                conflict_type=result.conflict_type or "materialization_conflict",
                local_envelope_id=envelope.client_envelope_id,
                server_sequence=envelope.server_sequence,
                metadata=dict(result.metadata),
            )
        )
        return SyncPushConflict(
            conflict_id=conflict.conflict_id,
            client_envelope_id=envelope.client_envelope_id,
            domain=envelope.domain,
            entity_id=envelope.entity_id,
            server_sequence=envelope.server_sequence,
            message=result.message,
        )

    def _resolve_cursor(
        self,
        dataset_id: str,
        device_id: str,
        cursor: str | int | None,
        domains: Sequence[SyncDomain] | None,
    ) -> int:
        """Resolve a legacy cursor from the request or stored domain watermarks."""

        if cursor is not None:
            return self._parse_cursor(cursor)
        cursor_domains = list(domains or self.settings.supported_domains)
        cursors: list[int] = []
        for domain in cursor_domains:
            stored = self.store.get_device_cursor(dataset_id, device_id, domain)
            cursors.append(stored.last_pulled_sequence if stored is not None else 0)
        return min(cursors, default=0)

    def _pull_adapter_streams(
        self,
        device: SyncDevice,
        domains: Sequence[SyncDomain],
    ) -> list[tuple[SyncDomain, int]]:
        """Return mutually supported domain-version streams for a device pull."""

        version_set = self._pull_version_set(device)
        streams = [
            (domain, version)
            for domain in domains
            for version in version_set.get(domain, ())
        ]
        if domains and not streams:
            raise SyncStoreError("sync_device_adapter_version_not_supported")
        if len(streams) > SYNC_PULL_TOKEN_MAX_STREAMS:
            raise SyncStoreError("sync_pull_token_too_large")
        return streams

    def _pull_version_set(self, device: SyncDevice) -> dict[SyncDomain, list[int]]:
        """Intersect a device's advertised adapter versions with server support."""

        requested = _device_requested_domains(device)
        try:
            advertised = normalize_supported_adapter_versions(
                device.capabilities.get("supported_adapter_versions"),
                requested_domains=requested,
            )
        except ValueError as exc:
            raise SyncStoreError("Sync device adapter version capabilities are invalid") from exc
        return {
            domain: [
                version
                for version in versions
                if self.adapters.has_domain(domain)
                and self.adapters.supports_version(domain, version)
            ]
            for domain, versions in advertised.items()
        }

    def _pull_versioned(
        self,
        *,
        dataset: SyncDataset,
        device: SyncDevice,
        cursor: str | int | None,
        streams: Sequence[tuple[SyncDomain, int]],
        page_size: int | None,
        include_own_changes: bool,
    ) -> SyncPullResult:
        """Pull a token-paginated page without advancing past hidden conflicts."""

        version_set = self._pull_version_set(device)
        if cursor is None:
            watermarks: dict[tuple[SyncDomain, int], int] = {}
            for domain, adapter_version in streams:
                stored = self.store.get_device_cursor(
                    dataset.dataset_id,
                    device.device_id,
                    domain,
                    adapter_version=adapter_version,
                )
                watermarks[(domain, adapter_version)] = (
                    stored.last_pulled_sequence if stored is not None else 0
                )
        else:
            if not isinstance(cursor, str):
                raise SyncStoreError("sync_pull_token_invalid")
            watermarks = self._decode_pull_token(
                cursor,
                dataset_id=dataset.dataset_id,
                device_id=device.device_id,
                version_set=version_set,
                streams=streams,
            )

        page_limit = min(
            page_size or self.settings.max_pull_page_size,
            self.settings.max_pull_page_size,
        )
        raw_envelopes, visible, blocker_cursor = self._scan_versioned_pull_page(
            dataset_id=dataset.dataset_id,
            device_id=device.device_id,
            watermarks=watermarks,
            page_limit=page_limit,
            include_own_changes=include_own_changes,
        )
        page = visible[:page_limit]
        has_visible_lookahead = len(visible) > page_limit
        has_more = has_visible_lookahead or len(raw_envelopes) > page_limit
        safe_raw_envelopes = [
            envelope
            for envelope in raw_envelopes
            if blocker_cursor is None or envelope.server_sequence < blocker_cursor
        ]
        boundary = (
            page[-1].server_sequence
            if has_visible_lookahead and page
            else max(
                (envelope.server_sequence for envelope in safe_raw_envelopes),
                default=0,
            )
        )
        next_watermarks = dict(watermarks)
        for envelope in raw_envelopes:
            if envelope.server_sequence > boundary:
                continue
            key = (envelope.domain, envelope.adapter_version)
            next_watermarks[key] = max(
                next_watermarks.get(key, 0),
                envelope.server_sequence,
            )
        delivered = {
            stream: max(
                (
                    envelope.server_sequence
                    for envelope in page
                    if (envelope.domain, envelope.adapter_version) == stream
                ),
                default=0,
            )
            for stream in streams
        }
        for domain, adapter_version in streams:
            self.store.update_device_cursor(
                SyncDeviceCursor(
                    dataset_id=dataset.dataset_id,
                    device_id=device.device_id,
                    domain=domain,
                    adapter_version=adapter_version,
                    last_pulled_sequence=next_watermarks[(domain, adapter_version)],
                    max_delivered_sequence=delivered[(domain, adapter_version)],
                )
            )
        return SyncPullResult(
            dataset_id=dataset.dataset_id,
            encryption_policy=dataset.encryption_policy,
            envelopes=page,
            next_cursor=self._encode_pull_token(
                dataset_id=dataset.dataset_id,
                device_id=device.device_id,
                version_set=version_set,
                watermarks=next_watermarks,
            ),
            has_more=has_more,
        )

    def _scan_versioned_pull_page(
        self,
        *,
        dataset_id: str,
        device_id: str,
        watermarks: Mapping[tuple[SyncDomain, int], int],
        page_limit: int,
        include_own_changes: bool,
    ) -> tuple[list[SyncEnvelope], list[SyncEnvelope], int | None]:
        """Return raw and visible versioned candidates plus any blocking cursor."""

        candidates: dict[int, SyncEnvelope] = {}
        for (domain, adapter_version), watermark in watermarks.items():
            for envelope in self.store.list_envelopes_after(
                dataset_id,
                watermark,
                limit=page_limit + 1,
                domains=[domain],
                adapter_versions=[adapter_version],
                status="accepted",
                exclude_device_id=None if include_own_changes else device_id,
            ):
                candidates[envelope.server_sequence] = envelope
        raw = sorted(candidates.values(), key=lambda item: item.server_sequence)[
            : page_limit + 1
        ]
        blocker = self.store.get_unresolved_materialization_conflict(dataset_id)
        blocker_cursor = (
            blocker.server_sequence
            if blocker is not None
            and blocker.conflict_type
            != SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION
            else None
        )
        visible = [
            envelope
            for envelope in raw
            if envelope.apply_status not in {"conflict", "superseded"}
            and (blocker_cursor is None or envelope.server_sequence < blocker_cursor)
        ]
        return raw, visible, blocker_cursor

    def _pull_token_secret(self) -> bytes:
        secret = (
            self.settings.pull_token_signing_secret
            or os.getenv("SYNC_V2_PULL_TOKEN_SIGNING_SECRET")
            or ""
        ).strip()
        if not secret:
            raise SyncStoreError("sync_pull_token_signing_unavailable")
        return secret.encode("utf-8")

    def _encode_pull_token(
        self,
        *,
        dataset_id: str,
        device_id: str,
        version_set: Mapping[SyncDomain, Sequence[int]],
        watermarks: Mapping[tuple[SyncDomain, int], int],
        ttl_seconds: int | None = None,
    ) -> str:
        now = _parse_sync_timestamp(self.clock()) or datetime.now(timezone.utc)
        token_ttl = (
            self.settings.pull_token_ttl_seconds
            if ttl_seconds is None
            else ttl_seconds
        )
        payload = {
            "v": SYNC_PULL_TOKEN_VERSION,
            "dataset_id": dataset_id,
            "device_id": device_id,
            "iat": int(now.timestamp()),
            "exp": int(now.timestamp()) + token_ttl,
            "vs": [
                [domain, list(versions)]
                for domain, versions in sorted(version_set.items())
            ],
            "wm": [
                [domain, adapter_version, sequence]
                for (domain, adapter_version), sequence in sorted(watermarks.items())
            ],
        }
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        if len(raw) > SYNC_PULL_TOKEN_MAX_DECODED_BYTES:
            raise SyncStoreError("sync_pull_token_too_large")
        payload_segment = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        signature = hmac.digest(self._pull_token_secret(), raw, "sha256")
        signature_segment = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
        token = f"{payload_segment}.{signature_segment}"
        if len(token) > SYNC_PULL_TOKEN_MAX_ENCODED_BYTES:
            raise SyncStoreError("sync_pull_token_too_large")
        return token

    def _decode_pull_token(
        self,
        token: str,
        *,
        dataset_id: str,
        device_id: str,
        version_set: Mapping[SyncDomain, Sequence[int]],
        streams: Sequence[tuple[SyncDomain, int]],
    ) -> dict[tuple[SyncDomain, int], int]:
        try:
            encoded = token.encode("ascii")
        except UnicodeEncodeError as exc:
            raise SyncStoreError("sync_pull_token_invalid") from exc
        if len(encoded) > SYNC_PULL_TOKEN_MAX_ENCODED_BYTES:
            raise SyncStoreError("sync_pull_token_too_large")
        try:
            payload_segment, signature_segment = token.split(".")
            raw = _decode_pull_token_segment(payload_segment)
            signature = _decode_pull_token_segment(signature_segment)
        except (ValueError, TypeError) as exc:
            raise SyncStoreError("sync_pull_token_invalid") from exc
        if len(raw) > SYNC_PULL_TOKEN_MAX_DECODED_BYTES:
            raise SyncStoreError("sync_pull_token_too_large")
        expected_signature = hmac.digest(self._pull_token_secret(), raw, "sha256")
        if not hmac.compare_digest(signature, expected_signature):
            raise SyncStoreError("sync_pull_token_invalid")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SyncStoreError("sync_pull_token_invalid") from exc
        expected_version_set = [
            [domain, list(versions)]
            for domain, versions in sorted(version_set.items())
        ]
        if not isinstance(payload, dict) or payload.get("v") != SYNC_PULL_TOKEN_VERSION:
            raise SyncStoreError("sync_pull_token_invalid")
        if payload.get("dataset_id") != dataset_id or payload.get("device_id") != device_id:
            raise SyncStoreError("sync_pull_token_invalid")
        if payload.get("vs") != expected_version_set:
            raise SyncStoreError("sync_pull_restart_required")
        now = _parse_sync_timestamp(self.clock()) or datetime.now(timezone.utc)
        iat = payload.get("iat")
        exp = payload.get("exp")
        if (
            isinstance(iat, bool)
            or not isinstance(iat, int)
            or iat > int(now.timestamp()) + SYNC_PULL_TOKEN_CLOCK_SKEW_SECONDS
            or isinstance(exp, bool)
            or not isinstance(exp, int)
            or exp < int(now.timestamp()) - SYNC_PULL_TOKEN_CLOCK_SKEW_SECONDS
        ):
            raise SyncStoreError("sync_pull_token_invalid")
        raw_watermarks = payload.get("wm")
        if not isinstance(raw_watermarks, list):
            raise SyncStoreError("sync_pull_token_invalid")
        if len(raw_watermarks) > SYNC_PULL_TOKEN_MAX_STREAMS:
            raise SyncStoreError("sync_pull_token_too_large")
        expected_streams = set(streams)
        watermarks: dict[tuple[SyncDomain, int], int] = {}
        for item in raw_watermarks:
            if not isinstance(item, list) or len(item) != 3:
                raise SyncStoreError("sync_pull_token_invalid")
            domain, adapter_version, sequence = item
            if (
                not isinstance(domain, str)
                or isinstance(adapter_version, bool)
                or not isinstance(adapter_version, int)
                or isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
                or (domain, adapter_version) not in expected_streams
                or (domain, adapter_version) in watermarks
            ):
                raise SyncStoreError("sync_pull_token_invalid")
            watermarks[(domain, adapter_version)] = sequence
        if set(watermarks) != expected_streams:
            raise SyncStoreError("sync_pull_restart_required")
        return watermarks

    def _parse_cursor(self, cursor: str | int | None) -> int:
        if cursor is None:
            return 0
        try:
            parsed = int(cursor)
        except (TypeError, ValueError) as exc:
            raise SyncStoreError(f"Invalid sync cursor: {cursor!r}") from exc
        if parsed < 0:
            raise SyncStoreError(f"Invalid sync cursor: {cursor!r}")
        return parsed

    def _selected_domains(
        self,
        dataset: SyncDataset,
        domains: Sequence[SyncDomain] | None,
    ) -> list[SyncDomain]:
        allowed = set(dataset.domains)
        requested = list(domains or dataset.domains)
        return [domain for domain in requested if domain in allowed and self.adapters.has_domain(domain)]

    def _selected_pull_domains(
        self,
        dataset: SyncDataset,
        device: SyncDevice,
        domains: Sequence[SyncDomain] | None,
    ) -> list[SyncDomain]:
        requested_by_device = _device_requested_domains(device)
        device_set = set(requested_by_device)
        if domains is not None:
            unsupported = sorted(set(domains).difference(device_set))
            if unsupported:
                raise SyncStoreError("sync_device_domain_not_supported")
            requested = list(domains)
        else:
            requested = requested_by_device
        selected = self._selected_domains(dataset, requested)
        if set(selected).intersection(NOTES_ORGANIZATION_DOMAINS):
            metadata = dataset.metadata.get("notes_organization_v1")
            state = metadata.get("state") if isinstance(metadata, Mapping) else None
            if state != "ready":
                raise SyncStoreError("notes_organization_sync_not_ready")
        if "notes.link" in selected:
            metadata = dataset.metadata.get("notes_link_v1")
            state = metadata.get("state") if isinstance(metadata, Mapping) else None
            if state != "ready":
                raise SyncStoreError("notes_link_sync_not_ready")
        return selected

    def _profile_manager(self) -> SyncV2ProfileManager:
        return SyncV2ProfileManager(
            store=self.store,
            capabilities_factory=self.capabilities,
            id_factory=self.id_factory,
            scan_limit=self.settings.restore_manifest_scan_limit,
            service=self,
            dataset_bootstrapper=self.dataset_bootstrapper,
            notes_link_bootstrapper=self.notes_link_bootstrapper,
            notes_attachment_bootstrapper=self.notes_attachment_bootstrapper,
        )

    def _personal_context_service_for_user(self, user_id: str) -> object:
        """Resolve the canonical Personal Context service after Sync auth checks."""

        resolver = self.personal_context_service_resolver
        if resolver is not None:
            return resolver(user_id)
        raise SyncStoreError("personal_context_key_custody_unavailable")

    def _update_cursors(
        self,
        dataset_id: str,
        device_id: str,
        domains: Sequence[SyncDomain],
        sequence: int | None,
        *,
        delivered: Sequence[SyncEnvelope],
    ) -> None:
        for domain in domains:
            delivered_sequence = max(
                (
                    envelope.server_sequence
                    for envelope in delivered
                    if envelope.domain == domain and envelope.adapter_version == 1
                ),
                default=0,
            )
            if sequence is None and delivered_sequence == 0:
                continue
            self.store.update_device_cursor(
                SyncDeviceCursor(
                    dataset_id=dataset_id,
                    device_id=device_id,
                    domain=domain,
                    adapter_version=1,
                    last_pulled_sequence=(
                        delivered_sequence if sequence is None else sequence
                    ),
                    max_delivered_sequence=delivered_sequence,
                )
            )

    def _scan_pull_page(
        self,
        *,
        dataset_id: str,
        device_id: str,
        since_sequence: int,
        domains: Sequence[SyncDomain],
        page_limit: int,
        include_own_changes: bool,
        adapter_versions: Sequence[int] | None = None,
    ) -> tuple[list[SyncEnvelope], list[SyncEnvelope]]:
        raw = self.store.list_envelopes_after(
            dataset_id,
            since_sequence,
            limit=page_limit + 1,
            domains=domains,
            adapter_versions=adapter_versions,
            status="accepted",
            exclude_device_id=None if include_own_changes else device_id,
        )
        blocker = self.store.get_unresolved_materialization_conflict(dataset_id)
        blocker_cursor = (
            blocker.server_sequence
            if blocker is not None
            and blocker.conflict_type
            != SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION
            else None
        )
        visible = [
            envelope
            for envelope in raw
            if envelope.apply_status not in {"conflict", "superseded"}
            and (
                blocker_cursor is None
                or envelope.server_sequence < blocker_cursor
            )
        ]
        return raw, visible

    def _expand_restore_mutation_groups(
        self,
        *,
        dataset_id: str,
        envelopes: Sequence[SyncEnvelope],
        latest_object_envelopes: Mapping[tuple[SyncDomain, str], SyncEnvelope],
        selected_domains: set[SyncDomain],
        selected_object_ids: set[str],
        adapter_versions_by_domain: Mapping[SyncDomain, Sequence[int]] | None = None,
    ) -> list[SyncEnvelope]:
        """Expand selected restore heads to complete persisted mutation units."""

        candidates = {
            (envelope.server_cursor, envelope.client_envelope_id): envelope
            for envelope in envelopes
        }
        queued = list(envelopes)
        processed_groups: set[str] = set()
        while queued:
            envelope = queued.pop()
            group_id = envelope.mutation_group_id
            if not group_id or group_id in processed_groups:
                continue
            group = self.store.list_mutation_group(dataset_id, group_id)
            validate_stored_mutation_group(
                group,
                dataset_id=dataset_id,
                mutation_group_id=group_id,
            )
            active_group = [
                member for member in group if member.apply_status != "superseded"
            ]
            group_was_terminally_split = len(active_group) != len(group)
            if selected_domains and any(
                member.domain not in selected_domains for member in active_group
            ):
                raise RestorePlanningError("Restore domain filter splits a mutation group")
            if selected_object_ids and any(
                member.object_id not in selected_object_ids for member in active_group
            ):
                raise RestorePlanningError("Restore object filter splits a mutation group")
            if adapter_versions_by_domain is not None and any(
                member.adapter_version
                not in adapter_versions_by_domain.get(member.domain, ())
                for member in active_group
            ):
                raise RestorePlanningError(
                    "Restore adapter-version filter splits a mutation group"
                )
            processed_groups.add(group_id)
            for member in active_group:
                if member.domain not in OBJECT_RESTORE_DOMAINS:
                    raise RestorePlanningError("Restore mutation group contains unsupported domain")
                restore_member = member
                if group_was_terminally_split:
                    restore_member = replace(
                        member,
                        mutation_group_id=None,
                        mutation_step=None,
                        mutation_step_count=None,
                        mutation_plan_hash=None,
                    )
                candidates[(member.server_cursor, member.client_envelope_id)] = restore_member
                latest = latest_object_envelopes.get((member.domain, member.object_id))
                if latest is None or latest.server_cursor == member.server_cursor:
                    continue
                key = (latest.server_cursor, latest.client_envelope_id)
                if key not in candidates:
                    candidates[key] = latest
                    queued.append(latest)
        return list(candidates.values())

    def _list_restore_preview_domain_envelopes(
        self,
        *,
        dataset_id: str,
        domain: SyncDomain,
        adapter_versions: Sequence[int],
        max_candidates: int,
    ) -> list[SyncEnvelope]:
        page_limit = self.settings.restore_manifest_scan_limit
        if page_limit < 1:
            raise SyncStoreError("Sync restore manifest scan limit must be greater than zero")
        envelopes: list[SyncEnvelope] = []
        since_sequence = 0
        while True:
            request_limit = min(
                page_limit,
                max(1, max_candidates - len(envelopes) + 1),
            )
            page = self.store.list_envelopes_after(
                dataset_id,
                since_sequence,
                limit=request_limit,
                domains=[domain],
                adapter_versions=adapter_versions,
                status="accepted",
            )
            if not page:
                break
            envelopes.extend(page)
            if len(envelopes) > max_candidates:
                raise SyncStoreError("sync_restore_candidate_limit_exceeded")
            next_sequence = max(envelope.server_sequence for envelope in page)
            if next_sequence <= since_sequence:
                raise SyncStoreError("Sync restore manifest cursor did not advance")
            since_sequence = next_sequence
            if len(page) < request_limit:
                break
        return envelopes

    def _restore_adapter_versions(
        self,
        device: SyncDevice | None,
        domain: SyncDomain,
    ) -> list[int]:
        """Return server-supported versions safe to expose to one restore client."""

        if device is None:
            return [1]
        try:
            supported = self.adapters.get(domain).supported_adapter_versions
        except KeyError:
            return []
        return sorted(
            version
            for version in supported
            if _device_supports_adapter_version(device, domain, version)
        )

    def _manifest_dataset(
        self,
        dataset: SyncDataset,
        *,
        user_id: str,
        domains: set[SyncDomain],
    ) -> SyncRestoreManifestDataset:
        selected_domains = [domain for domain in dataset.domains if not domains or domain in domains]
        stats = self.store.summarize_restore_manifest_dataset(
            dataset.dataset_id,
            user_id=user_id,
            domains=selected_domains,
        )
        last_updated_at = stats.last_updated_at or dataset.updated_at
        if last_updated_at < dataset.updated_at:
            last_updated_at = dataset.updated_at
        metadata = (
            {}
            if dataset.encryption_policy == "client_private_v1"
            else _redact_private_sync_server_metadata(dataset.metadata)
        )
        return SyncRestoreManifestDataset(
            dataset_id=dataset.dataset_id,
            scope_type=dataset.scope_type,
            encryption_policy=dataset.encryption_policy,
            domains=selected_domains,
            workspace_id=dataset.workspace_id,
            approximate_counts=stats.approximate_counts,
            byte_estimates=stats.byte_estimates,
            last_updated_at=last_updated_at,
            unresolved_conflicts=stats.unresolved_conflicts,
            attachment_availability=stats.attachment_availability,
            attachment_size_classes=stats.attachment_size_classes,
            key_recovery_available=stats.key_recovery_available,
            metadata=metadata,
        )

    def _require_blob_transfer(self) -> LocalSyncBlobStore:
        if not self.settings.supports_attachments or self.blob_store is None:
            raise SyncStoreError(
                "sync_blob_transfer_not_supported: Sync v2 M1 does not support binary blob transfer"
            )
        return self.blob_store

    def _metadata_only_blob_manifest(
        self,
        *,
        dataset_id: str,
        attachment_id: str,
    ) -> SyncBlobDownloadManifest:
        envelopes = self.store.list_envelopes_for_entity(
            dataset_id,
            "attachment.ref",
            entity_id=attachment_id,
            limit=1,
        )
        if not envelopes or envelopes[0].operation == "tombstone":
            raise SyncStoreError("Sync blob was not found or is not accessible")
        try:
            metadata = extract_attachment_ref_metadata(envelopes[0])
        except AttachmentRefValidationError as exc:
            raise SyncStoreError("Sync blob was not found or is not accessible") from exc
        if metadata.attachment_id != attachment_id:
            raise SyncStoreError("Sync blob was not found or is not accessible")
        availability = "deleted" if metadata.availability == "deleted" else "metadata_only"
        return SyncBlobDownloadManifest(
            dataset_id=dataset_id,
            attachment_id=metadata.attachment_id,
            availability=availability,
            content_type=metadata.content_type,
            size_bytes=metadata.size_bytes,
            payload_hash=metadata.payload_hash,
            chunks=[],
        )

    def _normalize_download_chunk_size(self, chunk_size: int | None) -> int:
        normalized = self.settings.max_chunk_bytes if chunk_size is None else chunk_size
        if normalized <= 0:
            raise SyncStoreError("Sync blob download chunk size is invalid")
        if normalized > self.settings.max_chunk_bytes:
            raise SyncStoreError("Sync blob chunk exceeds the server size limit")
        return normalized

    def _require_download_blob(
        self,
        *,
        user_id: str,
        dataset_id: str,
        attachment_id: str,
    ) -> tuple[LocalSyncBlobStore, SyncBlobObject]:
        """Return a verified blob-store handle and metadata for byte serving."""

        blob_store = self._require_blob_transfer()
        dataset = self._require_blob_dataset(user_id=user_id, dataset_id=dataset_id)
        blob = self.store.get_blob_object(
            dataset_id,
            attachment_id=attachment_id,
            owner_user_id=self._blob_owner_user_id(dataset=dataset, user_id=user_id),
        )
        if blob is None:
            raise SyncStoreError("Sync blob was not found or is not accessible")
        return blob_store, blob

    def _blob_owner_user_id(self, *, dataset: SyncDataset, user_id: str) -> str | None:
        if dataset.scope_type == "workspace":
            return None
        return user_id

    def _validate_blob_storage_metadata(
        self,
        *,
        blob_store: LocalSyncBlobStore,
        blob: SyncBlobObject,
    ) -> None:
        """Validate cheap committed-blob metadata before streaming content."""

        try:
            size_bytes = blob_store.blob_size(blob.storage_key)
        except (OSError, SyncBlobStoreError) as exc:
            raise SyncStoreError("Sync blob was not found or is not accessible") from exc
        if size_bytes != blob.size_bytes:
            raise SyncStoreError("Sync blob storage integrity check failed")

    def _require_blob_dataset(
        self,
        *,
        user_id: str,
        dataset_id: str,
        domain: SyncDomain | None = None,
    ) -> SyncDataset:
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        if domain is not None and domain not in dataset.domains:
            raise SyncInvalidDomainError(f"Sync domain is not enrolled for this dataset: {domain}")
        return dataset

    def _validate_blob_limits(
        self,
        *,
        user_id: str,
        dataset_id: str,
        size_bytes: int,
        chunk_size: int,
        chunk_count: int,
    ) -> None:
        max_blob_bytes = self.settings.max_blob_bytes or self.settings.max_attachment_bytes
        if size_bytes <= 0 or size_bytes > max_blob_bytes:
            raise SyncStoreError("Sync attachment payload exceeds the server size limit")
        if chunk_size <= 0 or chunk_size > self.settings.max_chunk_bytes:
            raise SyncStoreError("Sync blob chunk exceeds the server size limit")
        if chunk_count <= 0 or chunk_size * chunk_count < size_bytes:
            raise SyncStoreError("Sync blob chunk shape is invalid")
        quota = self.store.summarize_blob_quota(user_id, dataset_id=dataset_id)
        if (
            self.settings.user_blob_quota_bytes is not None
            and quota.used_blob_bytes + quota.reserved_blob_bytes + size_bytes
            > self.settings.user_blob_quota_bytes
        ):
            raise SyncStoreError("Sync blob quota exceeded")
        if quota.active_upload_count >= self.settings.max_active_blob_uploads:
            raise SyncStoreError("Sync blob active upload limit exceeded")

    def _validate_sha256_hash(self, value: str, *, field_name: str) -> None:
        prefix = "sha256:"
        if not value.startswith(prefix) or len(value) != len(prefix) + 64:
            raise SyncStoreError(f"Sync blob {field_name} must be sha256:<64 hex chars>")
        try:
            bytes.fromhex(value[len(prefix) :])
        except ValueError as exc:
            raise SyncStoreError(f"Sync blob {field_name} digest must be hex") from exc
        if value != value.lower():
            raise SyncStoreError(
                f"Sync blob {field_name} digest must be lowercase hex"
            )

    def _write_single_chunk_blob(
        self,
        *,
        blob_store: LocalSyncBlobStore,
        upload_id: str,
        payload: bytes,
        payload_hash: str,
    ) -> str:
        try:
            blob_store.write_upload_chunk(
                upload_id=upload_id,
                chunk_index=0,
                payload=payload,
                expected_hash=payload_hash,
            )
            return blob_store.commit_upload(
                upload_id=upload_id,
                payload_hash=payload_hash,
                chunk_indexes=[0],
            )
        except SyncBlobStoreError as exc:
            raise SyncStoreError(str(exc)) from exc

    def _latest_cursor_for_domains(
        self,
        dataset_id: str,
        domains: Sequence[SyncDomain],
    ) -> int | None:
        if not domains:
            return None
        envelopes = self.store.list_envelopes_after(
            dataset_id,
            0,
            limit=self.settings.restore_manifest_scan_limit,
            domains=domains,
        )
        return max((envelope.server_cursor or 0 for envelope in envelopes), default=None)


def _compact_json_size(value: object) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _canonical_attachment_uuid(value: object, field_name: str) -> str:
    """Validate and return a canonical lowercase attachment UUIDv4."""

    if not isinstance(value, str):
        raise SyncStoreError(f"Notes attachment {field_name} is invalid")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise SyncStoreError(f"Notes attachment {field_name} is invalid") from exc
    if parsed.version != 4 or parsed.variant != RFC_4122 or str(parsed) != value:
        raise SyncStoreError(f"Notes attachment {field_name} is invalid")
    return value


def _parse_sync_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _retention_blocker_counts(
    candidates: Sequence[SyncRetentionCandidate],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        for blocker in candidate.blockers:
            counts[blocker] = counts.get(blocker, 0) + 1
    return counts


def _retention_apply_candidate_enabled(
    candidate: SyncRetentionCandidate,
    *,
    apply_envelope_compaction: bool,
    apply_tombstone_prune: bool,
    apply_binding_release: bool,
    apply_blob_gc: bool,
) -> bool:
    if candidate.candidate_type == "envelope_compaction":
        return apply_envelope_compaction
    if candidate.candidate_type == "tombstone_prune":
        return apply_tombstone_prune
    if candidate.candidate_type == "binding_release":
        return apply_binding_release
    if candidate.candidate_type == "blob_gc":
        return apply_blob_gc
    return False


def _ciphertext_exceeds_attachment_limit(
    payload_ciphertext: str,
    max_attachment_bytes: int,
) -> bool:
    """Return whether persisted ciphertext text exceeds the attachment cap."""

    return len(payload_ciphertext.encode("utf-8")) > max_attachment_bytes


def _attachment_ref_has_server_blob(availability: str) -> bool:
    return availability.strip().lower() in ATTACHMENT_REF_SERVER_AVAILABILITY


def _normalize_selection_set(values: Sequence[str] | None) -> set[str]:
    selected: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if text:
            selected.add(text)
    return selected


def _build_restore_domain_details(
    *,
    safe_applies: Sequence[SyncRestorePreviewObject],
    tombstones: Sequence[SyncRestorePreviewObject],
    object_conflicts: Sequence[SyncRestorePreviewObjectConflict],
    blob_details: Sequence[SyncRestoreBlobCompleteness],
    metadata_only: bool,
) -> list[SyncRestoreDomainCompleteness]:
    counts: dict[SyncDomain, dict[str, int]] = {}

    def domain_counts(domain: SyncDomain) -> dict[str, int]:
        return counts.setdefault(
            domain,
            {
                "selected_count": 0,
                "safe_apply_count": 0,
                "conflict_count": 0,
                "tombstone_count": 0,
                "required_blob_count": 0,
                "available_blob_count": 0,
                "missing_blob_count": 0,
                "verified_blob_count": 0,
            },
        )

    for item in safe_applies:
        bucket = domain_counts(item.domain)
        bucket["selected_count"] += 1
        bucket["safe_apply_count"] += 1
    for item in tombstones:
        bucket = domain_counts(item.domain)
        bucket["selected_count"] += 1
        bucket["tombstone_count"] += 1
    for item in object_conflicts:
        bucket = domain_counts(item.domain)
        bucket["selected_count"] += 1
        bucket["conflict_count"] += 1
    for item in blob_details:
        bucket = domain_counts("attachment.ref")
        bucket["selected_count"] += 1
        if item.required_for_restore:
            bucket["required_blob_count"] += 1
            if item.server_availability == "available":
                bucket["available_blob_count"] += 1
            elif item.download_status not in {
                "available",
                "present",
                "stored",
                "server",
                "verified",
                "verified_complete",
            }:
                bucket["missing_blob_count"] += 1
            if item.download_status in {"verified", "verified_complete"}:
                bucket["verified_blob_count"] += 1

    domain_order = {domain: index for index, domain in enumerate(SYNC_V2_SUPPORTED_DOMAINS)}
    details: list[SyncRestoreDomainCompleteness] = []
    for domain, values in sorted(
        counts.items(),
        key=lambda item: domain_order.get(item[0], len(domain_order)),
    ):
        details.append(
            SyncRestoreDomainCompleteness(
                domain=domain,
                status=_restore_status_for_counts(values, metadata_only=metadata_only),
                selected_count=values["selected_count"],
                safe_apply_count=values["safe_apply_count"],
                conflict_count=values["conflict_count"],
                tombstone_count=values["tombstone_count"],
                required_blob_count=values["required_blob_count"],
                available_blob_count=values["available_blob_count"],
                missing_blob_count=values["missing_blob_count"],
                verified_blob_count=values["verified_blob_count"],
            )
        )
    return details


def _restore_status_for_counts(
    values: Mapping[str, int],
    *,
    metadata_only: bool,
) -> SyncRestoreCompletenessStatus:
    if values["conflict_count"] > 0:
        return "blocked_by_conflicts"
    required_blob_count = values["required_blob_count"]
    if required_blob_count == 0:
        if metadata_only and values["selected_count"] > 0:
            return "metadata_ready"
        return "content_complete"
    if values["missing_blob_count"] > 0:
        return "blob_incomplete"
    if values["verified_blob_count"] >= required_blob_count:
        return "verified_complete"
    return "content_complete"


def _restore_status_from_domain_details(
    domain_details: Sequence[SyncRestoreDomainCompleteness],
    *,
    has_blob_selection: bool,
    metadata_only: bool,
) -> SyncRestoreCompletenessStatus:
    if any(item.status == "blocked_by_conflicts" for item in domain_details):
        return "blocked_by_conflicts"
    if any(item.status == "blob_incomplete" for item in domain_details):
        return "blob_incomplete"
    if metadata_only and has_blob_selection:
        return "metadata_ready"
    required_blob_count = sum(item.required_blob_count for item in domain_details)
    verified_blob_count = sum(item.verified_blob_count for item in domain_details)
    if required_blob_count > 0 and verified_blob_count >= required_blob_count:
        return "verified_complete"
    return "content_complete"


def _append_warning_once(
    warnings: list[dict[str, str]],
    warning: dict[str, str],
) -> list[dict[str, str]]:
    code = warning.get("code")
    if code and any(item.get("code") == code for item in warnings):
        return warnings
    return [*warnings, warning]


def _device_requested_domains(device: SyncDevice) -> list[SyncDomain]:
    raw_requested = device.capabilities.get("requested_domains")
    requested = (
        [item for item in raw_requested if isinstance(item, str)]
        if isinstance(raw_requested, list)
        else list(M1_SYNC_DOMAINS)
    )
    raw_supported = device.capabilities.get("supported_domains")
    if isinstance(raw_supported, list):
        supported = {item for item in raw_supported if isinstance(item, str)}
        requested = [item for item in requested if item in supported]
    known = set(SYNC_V2_KNOWN_DOMAINS)
    return [item for item in requested if item in known]


def _personal_context_link_is_complete(
    store: SyncV2Store, dataset: SyncDataset, *, user_id: str, device_id: str
) -> bool:
    """Return whether reviewed first-link reconciliation admitted profile writes."""

    state = dataset.metadata.get("personal_context")
    if not isinstance(state, Mapping):
        return False
    return (
        isinstance(state.get("profile_id"), str)
        and isinstance(state.get("integrity_key_id"), str)
        and isinstance(state.get("purge_generation"), int)
        and store.has_personal_context_link_receipt(
            user_id=user_id,
            dataset_id=dataset.dataset_id,
            device_id=device_id,
            profile_id=state["profile_id"],
            integrity_key_id=state["integrity_key_id"],
            purge_generation=state["purge_generation"],
        )
    )


def _call_adapter_evaluate(
    adapter: SyncDomainAdapter,
    envelope: SyncEnvelopeCreate,
    *,
    dataset: SyncDataset,
    context: SyncAdapterContext,
) -> AdapterAccepted | AdapterRejected | AdapterConflict | AdapterDeferred:
    evaluate = adapter.evaluate_envelope
    if _evaluate_accepts_context(evaluate):
        return evaluate(envelope, dataset=dataset, context=context)
    return evaluate(envelope, dataset=dataset)


def _evaluate_accepts_context(evaluate: Callable[..., object]) -> bool:
    parameters = inspect.signature(evaluate).parameters.values()
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or (parameter.name == "context" and parameter.kind != inspect.Parameter.POSITIONAL_ONLY)
        for parameter in parameters
    )


__all__ = [
    "PersonalContextSyncCapabilities",
    "SyncDatasetEnrollment",
    "SyncDiagnosticsBlobHealth",
    "SyncDiagnosticsDevice",
    "SyncDiagnosticsDeviceDomainLag",
    "SyncDiagnosticsDomain",
    "SyncDiagnosticsKeySummary",
    "SyncDiagnosticsReport",
    "SyncDiagnosticsRetentionSummary",
    "SyncDeviceRegistration",
    "SyncPullResult",
    "SyncPushAccepted",
    "SyncPushConflict",
    "SyncPushRejected",
    "SyncPushResult",
    "SyncRetentionApplyResult",
    "SyncRetentionCandidate",
    "SyncRetentionDryRunResult",
    "SyncRestoreManifest",
    "SyncRestoreManifestDataset",
    "SyncRestoreManifestDevice",
    "SyncRestorePreview",
    "SyncRestorePreviewAttachmentRef",
    "SyncRestorePreviewDataset",
    "SyncRestorePreviewWarning",
    "SyncV2Capabilities",
    "SyncV2Service",
    "SyncV2Settings",
    "personal_context_sync_capabilities_from_env",
]
