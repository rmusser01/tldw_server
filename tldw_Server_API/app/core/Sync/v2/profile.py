from __future__ import annotations

"""Profile bootstrap and status helpers for Sync v2 M1."""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import SyncStoreError
from .models import (
    DEFAULT_M1_ENCRYPTION_POLICY,
    M1_SYNC_DOMAINS,
    NOTES_LINK_DOMAINS,
    NOTES_ORGANIZATION_DOMAINS,
    NOTES_TASK_SYNC_DOMAINS,
    SyncDataset,
    SyncDevice,
    SyncDeviceUpsert,
    SyncDomain,
    SyncEnvelope,
    client_private_server_frontend_limitation_warning,
    normalize_supported_adapter_versions,
    server_frontend_mutation_blockers_for_policy,
    server_frontend_mutation_enabled_for_policy,
)
from .notes_moodboard_studio_readiness import (
    parse_notes_moodboard_studio_readiness_record,
)
from .notes_task_readiness import parse_notes_task_readiness_record
from .store import SyncV2Store

SYNC_V2_M1_PROTOCOL_VERSION = "sync-v2-m1"
BOOTSTRAP_MODES = frozenset({"server_frontend", "offline_sync"})
DEFAULT_CLIENT_FAMILY = "chatbook"
_DORMANT_TASK_READINESS_MISSING = object()


@dataclass(frozen=True, slots=True)
class SyncProfileDeviceStatus:
    """Public device registration status included in Sync profile responses."""

    device_id: str | None
    registered: bool
    client_profile_id: str | None = None
    last_seen_at: str | None = None
    mode: str | None = None
    client_type: str | None = None
    client_version: str | None = None


@dataclass(frozen=True, slots=True)
class SyncProfileDatasetStatus:
    """Public default dataset metadata included in Sync profile responses."""

    dataset_id: str
    scope: str
    default_personal: bool
    client_family: str | None
    domains: list[SyncDomain]
    created_at: str | None = None
    updated_at: str | None = None
    encryption_policy: str = DEFAULT_M1_ENCRYPTION_POLICY
    server_frontend_mutation_enabled: bool = True
    server_frontend_mutation_blockers: list[str] = field(default_factory=list)
    notes_organization: dict[str, object] | None = None
    notes_link: dict[str, object] | None = None
    notes_attachment: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class SyncNotesAttachmentCleanupSample:
    """Public-safe cleanup evidence with no legacy name or path."""

    source_key_hash: str
    attachment_id: str
    state: str = "captured"
    blocker_code: str | None = None


@dataclass(frozen=True, slots=True)
class SyncRecoveryActionDescriptor:
    """Machine-readable, non-mutating recovery guidance."""

    action: str
    reason_code: str
    target_type: str = "dataset"
    target_id: str | None = None
    retryable: bool = True
    requires_confirmation: bool = False


@dataclass(frozen=True, slots=True)
class SyncNotesAttachmentBootstrapDiagnostics:
    """Bounded read-only attachment bootstrap diagnostics."""

    state: str
    captured_count: int = 0
    expected_count: int = 0
    cursor: str | None = None
    error_code: str | None = None
    dry_run: bool = False
    source_candidate_count: int | None = None
    source_candidate_count_is_lower_bound: bool = False
    cleanup_candidates: list[SyncNotesAttachmentCleanupSample] = field(
        default_factory=list
    )
    recovery_actions: list[SyncRecoveryActionDescriptor] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SyncDormantTaskDomainReadiness:
    """Privacy-safe internal readiness for one dormant task domain."""

    state: str = "not_enrolled"
    source_count: int = 0
    cursor: str | None = None
    source_fingerprint: str | None = None
    reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDormantTaskReadinessDiagnostics:
    """Owner-scoped dormant task readiness without source payload values."""

    task: SyncDormantTaskDomainReadiness
    task_activity: SyncDormantTaskDomainReadiness
    task_activity_capture_enabled: bool = False


@dataclass(frozen=True, slots=True)
class SyncDormantMoodboardStudioDomainReadiness:
    """Privacy-safe internal readiness for one dormant moodboard/Studio domain."""

    state: str = "not_enrolled"
    source_count: int = 0
    cursor: str | None = None
    source_fingerprint: str | None = None
    reason_code: str | None = None
    resume_phase: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDormantMoodboardStudioReadinessDiagnostics:
    """Owner-scoped dormant moodboard/Studio readiness without source values."""

    moodboard: SyncDormantMoodboardStudioDomainReadiness
    moodboard_note: SyncDormantMoodboardStudioDomainReadiness
    studio_document: SyncDormantMoodboardStudioDomainReadiness
    moodboard_capture_enabled: bool = False
    studio_document_capture_enabled: bool = False


@dataclass(frozen=True, slots=True)
class SyncProfileDomainStatus:
    """Per-domain Sync health summary for a profile dataset."""

    domain: SyncDomain
    last_server_cursor: int = 0
    envelope_count: int = 0
    pending_apply_count: int = 0
    pending_apply: int = 0
    failed_apply_count: int = 0
    unresolved_conflicts: int = 0
    last_apply_status: str | None = None
    last_apply_result: dict[str, Any] = field(default_factory=dict)
    repair_status: dict[str, Any] = field(default_factory=dict)
    server_frontend_mutation_enabled: bool = True
    server_frontend_mutation_blockers: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SyncProfileStatus:
    """Read-only Sync v2 M1 profile status returned by service methods."""

    protocol_version: str
    min_supported_protocol_version: str
    profile_bootstrapped: bool
    user_id: str
    active_dataset_id: str | None
    device: SyncProfileDeviceStatus | None
    dataset: SyncProfileDatasetStatus | None
    server_cursor: int
    capabilities: Any
    domain_status: list[SyncProfileDomainStatus] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    created: bool | None = None


class SyncV2ProfileManager:
    """Build and bootstrap Sync v2 M1 profile state through the store facade."""

    def __init__(
        self,
        *,
        store: SyncV2Store,
        capabilities_factory: Callable[..., Any],
        id_factory: Callable[[str], str],
        scan_limit: int,
        service: Any | None = None,
        dataset_bootstrapper: Any | None = None,
        notes_link_bootstrapper: Any | None = None,
        notes_attachment_bootstrapper: Any | None = None,
    ) -> None:
        self.store = store
        self.capabilities_factory = capabilities_factory
        self.id_factory = id_factory
        self.scan_limit = scan_limit
        self.service = service
        self.dataset_bootstrapper = dataset_bootstrapper
        self.notes_link_bootstrapper = notes_link_bootstrapper
        self.notes_attachment_bootstrapper = notes_attachment_bootstrapper

    def profile(self, *, user_id: str, device_id: str | None = None) -> SyncProfileStatus:
        """Return current profile state without creating devices or datasets."""

        dataset = self._default_personal_dataset(user_id)
        return self._build_profile(
            user_id=user_id,
            dataset=dataset,
            device_id=device_id,
            created=None,
        )

    def bootstrap_profile(
        self,
        *,
        user_id: str,
        mode: str,
        device_id: str | None = None,
        device_name: str | None = None,
        client_profile_id: str | None = None,
        client_family: str = DEFAULT_CLIENT_FAMILY,
        client_version: str | None = None,
        client_instance: dict[str, Any] | None = None,
        requested_domains: Sequence[SyncDomain] | None = None,
    ) -> SyncProfileStatus:
        """Idempotently create the default personal dataset and device state."""

        normalized_mode = mode.strip()
        if normalized_mode not in BOOTSTRAP_MODES:
            raise SyncStoreError("Sync profile bootstrap mode is invalid")
        if client_family != DEFAULT_CLIENT_FAMILY:
            raise SyncStoreError("Sync v2 M1 profile bootstrap requires chatbook client_family")
        requested = list(requested_domains or M1_SYNC_DOMAINS)
        capabilities = self.capabilities_factory()
        requested_task_domains = set(requested).intersection(NOTES_TASK_SYNC_DOMAINS)
        if requested_task_domains and requested_task_domains != set(
            NOTES_TASK_SYNC_DOMAINS
        ):
            raise SyncStoreError("notes_task_sync_domains_incomplete")
        invalid_domains = sorted(
            set(requested).difference(
                {*capabilities.supported_domains, *NOTES_TASK_SYNC_DOMAINS}
            )
        )
        if invalid_domains:
            raise SyncStoreError(
                "Sync v2 M1 profile bootstrap requested unsupported domains: "
                + ", ".join(invalid_domains)
            )
        organization_requested = set(requested).intersection(NOTES_ORGANIZATION_DOMAINS)
        if organization_requested and organization_requested != set(NOTES_ORGANIZATION_DOMAINS):
            raise SyncStoreError("notes_organization_sync_domains_incomplete")
        notes_link_requested = bool(set(requested).intersection(NOTES_LINK_DOMAINS))
        notes_attachment_requested = "attachment.ref" in requested
        encryption = getattr(capabilities, "encryption", {})
        if not encryption.get("ready", False):
            raise SyncStoreError(
                "sync_encryption_attestation_required: Sync v2 M1 requires "
                "server_trusted_v1 at-rest encryption readiness before bootstrap"
            )

        existing = self._default_personal_dataset(user_id)
        resolved_device_id = self._resolve_bootstrap_device_id(
            user_id=user_id,
            device_id=device_id,
            client_profile_id=client_profile_id,
        )
        canonical_client_instance = dict(client_instance or {})
        supported_adapter_versions = normalize_supported_adapter_versions(
            canonical_client_instance.pop("supported_adapter_versions", None),
            requested_domains=requested,
        )
        if self.service is None:
            raise SyncStoreError("Sync profile device registration service is unavailable")
        self.service._upsert_device(
            SyncDeviceUpsert(
                device_id=resolved_device_id,
                user_id=user_id,
                display_name=device_name or "Chatbook device",
                client_type=client_family,
                client_version=client_version or _client_version(client_instance),
                capabilities={
                    "client_profile_id": client_profile_id,
                    "sync_mode": normalized_mode,
                    "client_family": client_family,
                    "client_instance": canonical_client_instance,
                    "requested_domains": requested,
                    "supported_adapter_versions": supported_adapter_versions,
                },
            )
        )
        dataset = self.store.get_or_create_default_personal_dataset(user_id)
        if organization_requested:
            dataset = self.store.begin_notes_organization_bootstrap(
                dataset.dataset_id,
                owner_user_id=user_id,
                bootstrap_id=self.id_factory("notes-organization-bootstrap"),
            )
            if self.dataset_bootstrapper is not None:
                if self.service is None:
                    raise SyncStoreError("Notes organization bootstrap service is unavailable")
                dataset = self.dataset_bootstrapper.bootstrap(
                    service=self.service,
                    user_id=user_id,
                    dataset=dataset,
                )
        if notes_link_requested:
            dataset = self.store.begin_notes_link_bootstrap(
                dataset.dataset_id,
                owner_user_id=user_id,
                bootstrap_id=self.id_factory("notes-link-bootstrap"),
            )
            if self.notes_link_bootstrapper is not None:
                if self.service is None:
                    raise SyncStoreError("Notes link bootstrap service is unavailable")
                dataset = self.notes_link_bootstrapper.bootstrap(
                    service=self.service,
                    user_id=user_id,
                    dataset=dataset,
                )
        if notes_attachment_requested:
            dataset = self.store.begin_notes_attachment_bootstrap(
                dataset.dataset_id,
                owner_user_id=user_id,
                bootstrap_id=self.id_factory("notes-attachment-bootstrap"),
            )
            if self.notes_attachment_bootstrapper is not None:
                if self.service is None:
                    raise SyncStoreError(
                        "Notes attachment bootstrap service is unavailable"
                    )
                dataset = self.notes_attachment_bootstrapper.bootstrap(
                    service=self.service,
                    user_id=user_id,
                    dataset=dataset,
                )
        if requested_task_domains:
            if self.service is None:
                raise SyncStoreError("Notes task bootstrap service is unavailable")
            dataset = self.service._activate_notes_task_sync(dataset)
        return self._build_profile(
            user_id=user_id,
            dataset=dataset,
            device_id=resolved_device_id,
            created=existing is None,
        )

    def profile_status(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None = None,
    ) -> SyncProfileStatus:
        """Return status for an existing profile dataset."""

        dataset = self.store.get_dataset(dataset_id, owner_user_id=user_id)
        if dataset is None:
            raise SyncStoreError("Sync dataset was not found or is not accessible")
        return self._build_profile(
            user_id=user_id,
            dataset=dataset,
            device_id=device_id,
            created=None,
        )

    def notes_task_readiness_diagnostics(
        self,
        *,
        user_id: str,
        dataset_id: str,
    ) -> SyncDormantTaskReadinessDiagnostics:
        """Return owner-only, payload-free readiness for dormant task domains."""

        dataset = self.store.get_dataset(dataset_id, owner_user_id=user_id)
        if dataset is None:
            raise SyncStoreError("Sync dataset was not found or is not accessible")
        return SyncDormantTaskReadinessDiagnostics(
            task=_safe_dormant_task_readiness(
                dataset.metadata.get(
                    "notes_task_v1",
                    _DORMANT_TASK_READINESS_MISSING,
                ),
                domain="notes_task",
            ),
            task_activity=_safe_dormant_task_readiness(
                dataset.metadata.get(
                    "notes_task_activity_v1",
                    _DORMANT_TASK_READINESS_MISSING,
                ),
                domain="notes_task_activity",
            ),
            task_activity_capture_enabled=(
                dataset.metadata.get("task_activity_capture_enabled") is True
            ),
        )

    def notes_moodboard_studio_readiness_diagnostics(
        self,
        *,
        user_id: str,
        dataset_id: str,
    ) -> SyncDormantMoodboardStudioReadinessDiagnostics:
        """Return owner-only, payload-free readiness for dormant domains."""

        dataset = self.store.get_dataset(dataset_id, owner_user_id=user_id)
        if dataset is None:
            raise SyncStoreError("Sync dataset was not found or is not accessible")
        return SyncDormantMoodboardStudioReadinessDiagnostics(
            moodboard=_safe_dormant_moodboard_studio_readiness(
                dataset.metadata.get(
                    "notes_moodboard_v1",
                    _DORMANT_TASK_READINESS_MISSING,
                ),
                readiness_key="notes_moodboard_v1",
                domain="notes_moodboard",
            ),
            moodboard_note=_safe_dormant_moodboard_studio_readiness(
                dataset.metadata.get(
                    "notes_moodboard_note_v1",
                    _DORMANT_TASK_READINESS_MISSING,
                ),
                readiness_key="notes_moodboard_note_v1",
                domain="notes_moodboard_note",
            ),
            studio_document=_safe_dormant_moodboard_studio_readiness(
                dataset.metadata.get(
                    "notes_studio_document_v1",
                    _DORMANT_TASK_READINESS_MISSING,
                ),
                readiness_key="notes_studio_document_v1",
                domain="notes_studio_document",
            ),
            moodboard_capture_enabled=(
                dataset.metadata.get("moodboard_capture_enabled") is True
            ),
            studio_document_capture_enabled=(
                dataset.metadata.get("studio_document_capture_enabled") is True
            ),
        )

    def notes_attachment_bootstrap_diagnostics(
        self,
        *,
        user_id: str,
        dataset_id: str | None = None,
        sample_limit: int = 0,
        dry_run: bool = False,
    ) -> SyncNotesAttachmentBootstrapDiagnostics:
        """Return bounded source/bootstrap evidence without mutating state."""

        if isinstance(sample_limit, bool) or sample_limit < 0:
            raise SyncStoreError("sync_attachment_bootstrap_sample_limit_invalid")
        if sample_limit > 100:
            raise SyncStoreError("sync_attachment_bootstrap_sample_limit_exceeded")
        if dataset_id is None:
            dataset = self._default_personal_dataset(user_id)
        else:
            dataset = self.store.get_dataset(dataset_id, owner_user_id=user_id)
            if dataset is None:
                raise SyncStoreError(
                    "Sync dataset was not found or is not accessible"
                )

        status = _safe_notes_attachment_status(dataset) if dataset is not None else None
        state = str(status["state"]) if status is not None else "not_started"
        captured_count = (
            _safe_non_negative_int(status.get("captured_count")) if status else 0
        )
        expected_count = (
            _safe_non_negative_int(status.get("expected_count")) if status else 0
        )
        error_code = status.get("error_code") if status else None
        cursor = None
        if dataset is not None:
            attachment_metadata = dataset.metadata.get("notes_attachment_v2")
            source_cursor = (
                attachment_metadata.get("source_cursor")
                if isinstance(attachment_metadata, Mapping)
                else None
            )
            if isinstance(source_cursor, str) and source_cursor:
                cursor = "sha256:" + hashlib.sha256(
                    source_cursor.encode("utf-8")
                ).hexdigest()
        samples: list[SyncNotesAttachmentCleanupSample] = []
        if dataset is not None and sample_limit:
            metadata = dataset.metadata.get("notes_attachment_v2")
            bootstrap_id = (
                metadata.get("bootstrap_id") if isinstance(metadata, Mapping) else None
            )
            if isinstance(bootstrap_id, str) and bootstrap_id:
                samples = [
                    SyncNotesAttachmentCleanupSample(
                        source_key_hash=item.source_key_hash,
                        attachment_id=item.attachment_id,
                    )
                    for item in self.store.list_notes_attachment_cleanup_candidates(
                        dataset.dataset_id,
                        owner_user_id=user_id,
                        bootstrap_id=bootstrap_id,
                        limit=sample_limit,
                    )
                ]

        source_candidate_count: int | None = None
        source_candidate_count_is_lower_bound = False
        if dry_run:
            if self.service is None or self.notes_attachment_bootstrapper is None:
                raise SyncStoreError(
                    "Notes attachment bootstrap diagnostics are unavailable"
                )
            raw = self.notes_attachment_bootstrapper.dry_run(
                service=self.service,
                user_id=user_id,
            )
            if not isinstance(raw, Mapping):
                raise SyncStoreError("notes_attachment_bootstrap_dry_run_invalid")
            source_candidate_count = _safe_non_negative_int(
                raw.get("candidate_count")
            )
            source_candidate_count_is_lower_bound = (
                raw.get("candidate_count_is_lower_bound") is True
            )
            dry_run_error = raw.get("error_code")
            if isinstance(dry_run_error, str):
                error_code = dry_run_error

        return SyncNotesAttachmentBootstrapDiagnostics(
            state=state,
            captured_count=captured_count,
            expected_count=expected_count,
            cursor=cursor if isinstance(cursor, str) else None,
            error_code=error_code if isinstance(error_code, str) else None,
            dry_run=dry_run,
            source_candidate_count=source_candidate_count,
            source_candidate_count_is_lower_bound=(
                source_candidate_count_is_lower_bound
            ),
            cleanup_candidates=samples,
            recovery_actions=(
                [
                    SyncRecoveryActionDescriptor(
                        action="bootstrap_resume",
                        reason_code="sync_attachment_bootstrap_incomplete",
                    )
                ]
                if state in {"initializing", "failed"}
                else []
            ),
        )

    def _build_profile(
        self,
        *,
        user_id: str,
        dataset: SyncDataset | None,
        device_id: str | None,
        created: bool | None,
    ) -> SyncProfileStatus:
        capabilities = self.capabilities_factory(
            user_id=user_id,
            dataset_id=dataset.dataset_id if dataset is not None else None,
        )
        device = self._device_status(user_id, device_id)
        dataset_status = _dataset_status(dataset) if dataset is not None else None
        domain_status = (
            self._domain_status(dataset=dataset)
            if dataset is not None
            else []
        )
        server_cursor = max(
            (item.last_server_cursor for item in domain_status),
            default=0,
        )
        warnings = list(getattr(capabilities, "warnings", []))
        if dataset is not None and not server_frontend_mutation_enabled_for_policy(
            dataset.encryption_policy
        ):
            _append_warning_once(
                warnings,
                client_private_server_frontend_limitation_warning(),
            )
        return SyncProfileStatus(
            protocol_version=SYNC_V2_M1_PROTOCOL_VERSION,
            min_supported_protocol_version=SYNC_V2_M1_PROTOCOL_VERSION,
            profile_bootstrapped=dataset is not None,
            user_id=user_id,
            active_dataset_id=dataset.dataset_id if dataset is not None else None,
            device=device,
            dataset=dataset_status,
            server_cursor=server_cursor,
            capabilities=capabilities,
            domain_status=domain_status,
            warnings=warnings,
            created=created,
        )

    def _default_personal_dataset(self, user_id: str) -> SyncDataset | None:
        for dataset in self.store.list_datasets_for_user(user_id):
            if (
                dataset.scope_type == "personal"
                and dataset.metadata.get("default_personal") is True
                and dataset.metadata.get("client_family") == DEFAULT_CLIENT_FAMILY
            ):
                return dataset
        return None

    def _resolve_bootstrap_device_id(
        self,
        *,
        user_id: str,
        device_id: str | None,
        client_profile_id: str | None,
    ) -> str:
        if device_id is not None:
            return device_id
        if client_profile_id:
            for device in self.store.list_devices_for_user(user_id):
                if (
                    device.revoked_at is None
                    and device.capabilities.get("client_profile_id") == client_profile_id
                ):
                    return device.device_id
        return self.id_factory("device")

    def _device_status(
        self,
        user_id: str,
        device_id: str | None,
    ) -> SyncProfileDeviceStatus | None:
        devices = self.store.list_devices_for_user(user_id)
        if device_id is None:
            return _device_status(devices[0]) if devices else None
        for device in devices:
            if device.device_id == device_id and device.revoked_at is None:
                return _device_status(device)
        return SyncProfileDeviceStatus(device_id=device_id, registered=False)

    def _domain_status(
        self,
        *,
        dataset: SyncDataset,
    ) -> list[SyncProfileDomainStatus]:
        conflicts = self.store.list_conflicts(dataset.dataset_id, status="unresolved")
        return [
            self._single_domain_status(
                dataset=dataset,
                domain=domain,
                unresolved_conflicts=sum(
                    1 for conflict in conflicts if conflict.domain == domain
                ),
            )
            for domain in dataset.domains
        ]

    def _single_domain_status(
        self,
        *,
        dataset: SyncDataset,
        domain: SyncDomain,
        unresolved_conflicts: int,
    ) -> SyncProfileDomainStatus:
        summary = self.store.summarize_domain_envelopes(dataset.dataset_id, domain)
        last = summary.last_envelope
        last_apply_result = _last_apply_result(last)
        return SyncProfileDomainStatus(
            domain=domain,
            last_server_cursor=last.server_cursor if last is not None else 0,
            envelope_count=summary.envelope_count,
            pending_apply_count=summary.pending_apply_count,
            pending_apply=summary.pending_apply_count,
            failed_apply_count=summary.failed_apply_count,
            unresolved_conflicts=unresolved_conflicts,
            last_apply_status=last.apply_status if last is not None else None,
            last_apply_result=last_apply_result,
            repair_status=_repair_status(
                summary.pending_apply_count,
                summary.failed_apply_count,
                summary.last_failed_envelope,
            ),
            server_frontend_mutation_enabled=server_frontend_mutation_enabled_for_policy(
                dataset.encryption_policy
            ),
            server_frontend_mutation_blockers=server_frontend_mutation_blockers_for_policy(
                dataset.encryption_policy
            ),
        )


def _client_version(client_instance: dict[str, Any] | None) -> str | None:
    if not client_instance:
        return None
    value = client_instance.get("app_version")
    return str(value) if value is not None else None


def _device_status(device: SyncDevice) -> SyncProfileDeviceStatus:
    mode = _optional_str(device.capabilities.get("sync_mode"))
    if mode not in BOOTSTRAP_MODES:
        mode = None
    return SyncProfileDeviceStatus(
        device_id=device.device_id,
        registered=True,
        client_profile_id=_optional_str(device.capabilities.get("client_profile_id")),
        last_seen_at=device.last_seen_at,
        mode=mode,
        client_type=device.client_type,
        client_version=device.client_version,
    )


def _dataset_status(dataset: SyncDataset) -> SyncProfileDatasetStatus:
    return SyncProfileDatasetStatus(
        dataset_id=dataset.dataset_id,
        scope=dataset.scope_type,
        default_personal=dataset.metadata.get("default_personal") is True,
        client_family=_optional_str(dataset.metadata.get("client_family")),
        domains=list(dataset.domains),
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        encryption_policy=dataset.encryption_policy,
        server_frontend_mutation_enabled=server_frontend_mutation_enabled_for_policy(
            dataset.encryption_policy
        ),
        server_frontend_mutation_blockers=server_frontend_mutation_blockers_for_policy(
            dataset.encryption_policy
        ),
        notes_organization=_safe_notes_organization_status(dataset),
        notes_link=_safe_notes_link_status(dataset),
        notes_attachment=_safe_notes_attachment_status(dataset),
    )


def _safe_notes_organization_status(dataset: SyncDataset) -> dict[str, object] | None:
    metadata = dataset.metadata.get("notes_organization_v1")
    if not isinstance(metadata, dict):
        return None
    state = metadata.get("state")
    error_code = metadata.get("error_code")
    if state not in {"initializing", "ready", "failed"}:
        state = "failed"
        error_code = "notes_organization_bootstrap_state_invalid"
    return {
        "state": state,
        "captured_count": _safe_non_negative_int(metadata.get("captured_count")),
        "expected_count": _safe_non_negative_int(metadata.get("expected_count")),
        "error_code": error_code if isinstance(error_code, str) else None,
    }


def _safe_notes_link_status(dataset: SyncDataset) -> dict[str, object] | None:
    metadata = dataset.metadata.get("notes_link_v1")
    if not isinstance(metadata, dict):
        return None
    state = metadata.get("state")
    error_code = metadata.get("error_code")
    if state not in {"initializing", "ready", "failed"}:
        state = "failed"
        error_code = "notes_link_bootstrap_state_invalid"
    return {
        "state": state,
        "captured_count": _safe_non_negative_int(metadata.get("captured_count")),
        "expected_count": _safe_non_negative_int(metadata.get("expected_count")),
        "error_code": error_code if isinstance(error_code, str) else None,
    }


def _safe_notes_attachment_status(
    dataset: SyncDataset,
) -> dict[str, object] | None:
    metadata = dataset.metadata.get("notes_attachment_v2")
    if not isinstance(metadata, Mapping):
        return None
    state = metadata.get("state")
    error_code = metadata.get("error_code")
    if state not in {"initializing", "ready", "failed"}:
        state = "failed"
        error_code = "notes_attachment_bootstrap_state_invalid"
    return {
        "state": state,
        "captured_count": _safe_non_negative_int(metadata.get("captured_count")),
        "expected_count": _safe_non_negative_int(metadata.get("expected_count")),
        "error_code": error_code if isinstance(error_code, str) else None,
    }


def _safe_dormant_task_readiness(
    metadata: object,
    *,
    domain: str,
) -> SyncDormantTaskDomainReadiness:
    if metadata is _DORMANT_TASK_READINESS_MISSING:
        return SyncDormantTaskDomainReadiness()

    invalid = SyncDormantTaskDomainReadiness(
        state="blocked",
        reason_code=f"{domain}_readiness_state_invalid",
    )
    readiness_key = (
        "notes_task_v1"
        if domain == "notes_task"
        else "notes_task_activity_v1"
    )
    result = parse_notes_task_readiness_record(
        metadata,
        readiness_key=readiness_key,
    )
    if result.record is None:
        return invalid
    record = result.record
    cursor_hash = (
        "sha256:"
        + hashlib.sha256(record.source_cursor.encode("utf-8")).hexdigest()
        if record.source_cursor is not None
        else None
    )
    return SyncDormantTaskDomainReadiness(
        state=record.state,
        source_count=record.source_count,
        cursor=cursor_hash,
        source_fingerprint=record.source_fingerprint,
        reason_code=record.reason_code,
    )


def _safe_dormant_moodboard_studio_readiness(
    metadata: object,
    *,
    readiness_key: str,
    domain: str,
) -> SyncDormantMoodboardStudioDomainReadiness:
    if metadata is _DORMANT_TASK_READINESS_MISSING:
        return SyncDormantMoodboardStudioDomainReadiness()

    invalid = SyncDormantMoodboardStudioDomainReadiness(
        state="blocked",
        reason_code=f"{domain}_readiness_state_invalid",
    )
    result = parse_notes_moodboard_studio_readiness_record(
        metadata,
        readiness_key=readiness_key,
    )
    if result.record is None:
        return invalid
    record = result.record
    cursor_hash = (
        "sha256:"
        + hashlib.sha256(record.source_cursor.encode("utf-8")).hexdigest()
        if record.source_cursor is not None
        else None
    )
    return SyncDormantMoodboardStudioDomainReadiness(
        state=record.state,
        source_count=record.source_count,
        cursor=cursor_hash,
        source_fingerprint=record.source_fingerprint,
        reason_code=record.reason_code,
        resume_phase=record.resume_phase,
    )


def _safe_non_negative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _append_warning_once(
    warnings: list[dict[str, str]],
    warning: dict[str, str],
) -> None:
    code = warning.get("code")
    if code and any(item.get("code") == code for item in warnings):
        return
    warnings.append(warning)


def _last_apply_result(envelope: SyncEnvelope | None) -> dict[str, Any]:
    if envelope is None:
        return {}
    result: dict[str, Any] = {
        "status": envelope.apply_status,
        "server_cursor": envelope.server_cursor,
        "client_envelope_id": envelope.client_envelope_id,
    }
    if envelope.envelope_id is not None:
        result["envelope_id"] = envelope.envelope_id
    if envelope.apply_error_code is not None:
        result["error_code"] = envelope.apply_error_code
    if envelope.apply_error_message is not None:
        result["error_message"] = envelope.apply_error_message
    if envelope.applied_at is not None:
        result["applied_at"] = envelope.applied_at
    return result


def _repair_status(
    pending_apply_count: int,
    failed_apply_count: int,
    last_failed: SyncEnvelope | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": (
            "repair_needed" if pending_apply_count or failed_apply_count else "healthy"
        ),
        "pending_apply_count": pending_apply_count,
        "failed_apply_count": failed_apply_count,
    }
    if last_failed is not None:
        result["last_failed_cursor"] = last_failed.server_cursor
        result["last_failed_client_envelope_id"] = last_failed.client_envelope_id
        if last_failed.apply_error_code is not None:
            result["last_error_code"] = last_failed.apply_error_code
        if last_failed.apply_error_message is not None:
            result["last_error_message"] = last_failed.apply_error_message
    return result


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


__all__ = [
    "BOOTSTRAP_MODES",
    "DEFAULT_CLIENT_FAMILY",
    "SYNC_V2_M1_PROTOCOL_VERSION",
    "SyncDormantMoodboardStudioDomainReadiness",
    "SyncDormantMoodboardStudioReadinessDiagnostics",
    "SyncDormantTaskDomainReadiness",
    "SyncDormantTaskReadinessDiagnostics",
    "SyncNotesAttachmentBootstrapDiagnostics",
    "SyncNotesAttachmentCleanupSample",
    "SyncProfileDatasetStatus",
    "SyncProfileDeviceStatus",
    "SyncProfileDomainStatus",
    "SyncProfileStatus",
    "SyncV2ProfileManager",
]
