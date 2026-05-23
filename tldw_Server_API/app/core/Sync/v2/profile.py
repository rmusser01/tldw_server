from __future__ import annotations

"""Profile bootstrap and status helpers for Sync v2 M1."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import SyncStoreError
from .models import (
    DEFAULT_M1_ENCRYPTION_POLICY,
    M1_SYNC_DOMAINS,
    SyncDataset,
    SyncDevice,
    SyncDeviceUpsert,
    SyncDomain,
    SyncEnvelope,
)
from .store import SyncV2Store

SYNC_V2_M1_PROTOCOL_VERSION = "sync-v2-m1"
BOOTSTRAP_MODES = frozenset({"server_frontend", "offline_sync"})
DEFAULT_CLIENT_FAMILY = "chatbook"


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
        capabilities_factory: Callable[[], Any],
        id_factory: Callable[[str], str],
        scan_limit: int,
    ) -> None:
        self.store = store
        self.capabilities_factory = capabilities_factory
        self.id_factory = id_factory
        self.scan_limit = scan_limit

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
        invalid_domains = sorted(set(requested_domains or []).difference(M1_SYNC_DOMAINS))
        if invalid_domains:
            raise SyncStoreError(
                "Sync v2 M1 profile bootstrap requested unsupported domains: "
                + ", ".join(invalid_domains)
            )
        capabilities = self.capabilities_factory()
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
        self.store.upsert_device(
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
                    "client_instance": dict(client_instance or {}),
                    "requested_domains": list(requested_domains or M1_SYNC_DOMAINS),
                },
            )
        )
        dataset = self.store.get_or_create_default_personal_dataset(user_id)
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

    def _build_profile(
        self,
        *,
        user_id: str,
        dataset: SyncDataset | None,
        device_id: str | None,
        created: bool | None,
    ) -> SyncProfileStatus:
        capabilities = self.capabilities_factory()
        device = self._device_status(user_id, device_id)
        dataset_status = _dataset_status(dataset) if dataset is not None else None
        domain_status = (
            self._domain_status(user_id=user_id, dataset=dataset)
            if dataset is not None
            else []
        )
        server_cursor = max(
            (item.last_server_cursor for item in domain_status),
            default=0,
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
            warnings=list(getattr(capabilities, "warnings", [])),
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
        user_id: str,
        dataset: SyncDataset,
    ) -> list[SyncProfileDomainStatus]:
        stats = self.store.summarize_restore_manifest_dataset(
            dataset.dataset_id,
            user_id=user_id,
            domains=dataset.domains,
        )
        conflicts = self.store.list_conflicts(dataset.dataset_id, status="unresolved")
        return [
            self._single_domain_status(
                dataset=dataset,
                domain=domain,
                envelope_count=stats.approximate_counts.get(domain, 0),
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
        envelope_count: int,
        unresolved_conflicts: int,
    ) -> SyncProfileDomainStatus:
        envelopes = self._all_domain_envelopes(dataset.dataset_id, domain)
        last = _last_envelope(envelopes)
        last_apply_result = _last_apply_result(last)
        pending_apply_count = sum(
            1 for envelope in envelopes if envelope.apply_status == "pending"
        )
        failed_apply_count = sum(
            1 for envelope in envelopes if envelope.apply_status == "failed"
        )
        return SyncProfileDomainStatus(
            domain=domain,
            last_server_cursor=last.server_cursor if last is not None else 0,
            envelope_count=envelope_count,
            pending_apply_count=pending_apply_count,
            pending_apply=pending_apply_count,
            failed_apply_count=failed_apply_count,
            unresolved_conflicts=unresolved_conflicts,
            last_apply_status=last.apply_status if last is not None else None,
            last_apply_result=last_apply_result,
            repair_status=_repair_status(envelopes, failed_apply_count),
        )

    def _all_domain_envelopes(
        self,
        dataset_id: str,
        domain: SyncDomain,
    ) -> list[SyncEnvelope]:
        page_size = max(1, self.scan_limit)
        cursor = 0
        envelopes: list[SyncEnvelope] = []
        while True:
            page = self.store.list_envelopes_after(
                dataset_id,
                cursor,
                domains=[domain],
                limit=page_size,
            )
            if not page:
                return envelopes
            envelopes.extend(page)
            next_cursor = max(envelope.server_cursor or cursor for envelope in page)
            if next_cursor <= cursor:
                return envelopes
            cursor = next_cursor


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
    )


def _last_envelope(envelopes: Sequence[SyncEnvelope]) -> SyncEnvelope | None:
    if not envelopes:
        return None
    return max(envelopes, key=lambda envelope: envelope.server_cursor or 0)


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
    envelopes: Sequence[SyncEnvelope],
    failed_apply_count: int,
) -> dict[str, Any]:
    failed_envelopes = [envelope for envelope in envelopes if envelope.apply_status == "failed"]
    last_failed = _last_envelope(failed_envelopes)
    result: dict[str, Any] = {
        "status": "repair_needed" if failed_apply_count else "healthy",
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
    "SyncProfileDatasetStatus",
    "SyncProfileDeviceStatus",
    "SyncProfileDomainStatus",
    "SyncProfileStatus",
    "SyncV2ProfileManager",
]
