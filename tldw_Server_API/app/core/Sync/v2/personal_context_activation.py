"""Install protected activation baselines through the existing Sync bootstrap path.

Baseline envelopes have a dedicated journal, not ordinary publication ordinals:
they may contain more than one publication batch or have a zero source watermark.
The canonical journal owns continuity and coverage; Sync only owns delivery receipts.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5

from tldw_profile_core import ProfileManifest, ProfileProposal, ProfileRecord, ProfileScope
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.core.Personalization.personal_context_activation import PersonalContextActivationService
from tldw_Server_API.app.core.Personalization.personal_context_crypto import EncryptedEnvelope, EnvelopeCipher
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    PreparedPersonalContextActivation,
)

from .errors import SyncStoreError
from .models import PERSONAL_CONTEXT_SYNC_DOMAINS, SyncDataset
from .personal_context_ongoing_contract import PersonalContextActivationReceipt, PersonalContextExchangeProof
from .profile import (
    _PERSONAL_CONTEXT_BOOTSTRAP_TOKEN_TTL_SECONDS,
    PersonalContextBootstrap,
    PersonalContextBootstrapIntegrityKey,
    _personal_context_bootstrap_quotas,
    _personal_context_quotas_compatible,
)

if TYPE_CHECKING:
    from .service import SyncV2Service


def _activation_now() -> datetime:
    """Use one UTC clock for protected delivery expiration."""

    return datetime.now(timezone.utc)


def _expired(row: Mapping[str, Any]) -> bool:
    """Reject malformed expiry metadata instead of treating it as unlimited."""

    expiry = datetime.fromisoformat(row["expires_at"])
    if expiry.tzinfo is None:
        raise SyncStoreError("personal_context_activation_required")
    return expiry <= _activation_now()


def _identity(prepared: PreparedPersonalContextActivation, dataset: SyncDataset, user_id: str) -> dict[str, Any]:
    """Bind protected baseline bytes to canonical and authenticated transport identity."""

    return {
        "activation_id": prepared.activation_id,
        "dataset_id": dataset.dataset_id,
        "user_id": user_id,
        "profile_id": prepared.profile_id,
        "device_id": prepared.device_id,
        "baseline_digest": prepared.baseline_digest,
        "purge_generation": prepared.purge_generation,
        "publication_watermark": prepared.publication_watermark,
    }


def _cipher(service: SyncV2Service, dataset: SyncDataset) -> EnvelopeCipher:
    """Reuse the same per-profile Sync storage key as ordinary authority envelopes."""

    adapter = service.adapters.get("personal_context.manifest")
    resolver = getattr(adapter, "encryption_key_resolver", None)
    if resolver is None:
        raise SyncStoreError("personal_context_activation_required")
    key, version = resolver(dataset)
    return EnvelopeCipher(key, key_version=version)


def _protected_baseline(
    prepared: PreparedPersonalContextActivation, identity: Mapping[str, Any], cipher: EnvelopeCipher
) -> str:
    """Encrypt deterministic baseline envelopes; no canonical body enters generic SQL."""

    baseline = json.loads(prepared.baseline)
    objects = [(kind, value) for kind in ("scopes", "records", "proposals") for value in baseline[kind]]
    objects.append(("manifest", baseline["manifest"]))
    result = []
    for ordinal, (kind, value) in enumerate(objects):
        envelope_id = str(uuid5(NAMESPACE_URL, f"personal-context-activation:{prepared.activation_id}:{ordinal}"))
        aad = canonical_json_bytes({**identity, "envelope_id": envelope_id})
        encrypted = cipher.encrypt(canonical_json_bytes({"kind": kind, "value": value}), aad)
        fields = {
            name: base64.b64encode(value).decode("ascii") if isinstance(value, bytes) else value
            for name, value in asdict(encrypted).items()
        }
        result.append({"envelope_id": envelope_id, **fields})
    return canonical_json_bytes(result).decode("utf-8")


def _verify_install(
    prepared: PreparedPersonalContextActivation,
    row: Mapping[str, Any],
    identity: Mapping[str, Any],
    cipher: EnvelopeCipher,
) -> bool:
    """Authenticate every pinned envelope, checkpoint and receipt before coverage."""

    if any(row.get(key) != value for key, value in identity.items()):
        return False
    if type(row.get("home_server_cursor")) is not int or row["home_server_cursor"] < 0:
        return False
    baseline: dict[str, Any] = {"scopes": [], "records": [], "proposals": []}
    encrypted = json.loads(row["envelopes_json"])
    for ordinal, entry in enumerate(encrypted):
        envelope_id = str(uuid5(NAMESPACE_URL, f"personal-context-activation:{prepared.activation_id}:{ordinal}"))
        if entry["envelope_id"] != envelope_id:
            return False
        envelope = EncryptedEnvelope(
            **{
                name: base64.b64decode(entry[name], validate=True)
                if name
                in {
                    "nonce",
                    "wrapped_dek",
                    "wrapped_dek_nonce",
                    "ciphertext",
                }
                else entry[name]
                for name in EncryptedEnvelope.__dataclass_fields__
            }
        )
        clear = json.loads(cipher.decrypt(envelope, canonical_json_bytes({**identity, "envelope_id": envelope_id})))
        if clear["kind"] == "manifest":
            if "manifest" in baseline or ordinal != len(encrypted) - 1:
                return False
            baseline["manifest"] = clear["value"]
        elif clear["kind"] in baseline and clear["kind"] != "manifest":
            baseline[clear["kind"]].append(clear["value"])
        else:
            return False
    receipt = hashlib.sha256(
        canonical_json_bytes(
            {
                **identity,
                "home_server_cursor": row["home_server_cursor"],
                "envelopes_json": row["envelopes_json"],
                "expires_at": row["expires_at"],
            }
        )
    ).hexdigest()
    return hmac.compare_digest(canonical_json_bytes(baseline), prepared.baseline) and hmac.compare_digest(
        receipt, row["receipt_id"]
    )


def _receipt(prepared: PreparedPersonalContextActivation) -> PersonalContextActivationReceipt:
    """Expose only bounded canonical installation facts."""

    manifest = json.loads(prepared.baseline)["manifest"]
    return PersonalContextActivationReceipt(
        activation_id=prepared.activation_id,
        baseline_digest=prepared.baseline_digest,
        purge_generation=prepared.purge_generation,
        publication_watermark=prepared.publication_watermark,
        home_server_cursor=prepared.home_server_cursor or 0,
        home_manifest_revision=manifest["revision"],
        home_manifest_version_id=manifest["current_version_id"],
        state=prepared.state,
    )


def prepare_activation(
    service: SyncV2Service,
    *,
    user_id: str,
    device_id: str,
    required_schema_version: int | None = None,
    required_quotas: Mapping[str, int] | None = None,
    expected_purge_generation: int | None = None,
) -> PersonalContextBootstrap:
    """Prepare, install and verify a linked device baseline without enabling rollout."""

    device = service._require_registered_device(user_id, device_id)
    capability = service.capabilities(user_id=user_id).personal_context
    if not capability.available or not service._personal_context_domains_ready():
        raise SyncStoreError("personal_context_capability_unavailable")
    if required_schema_version is not None and not (
        capability.min_schema_version <= required_schema_version <= capability.max_schema_version
    ):
        raise SyncStoreError("personal_context_schema_incompatible")
    quotas = _personal_context_bootstrap_quotas(capability)
    if not _personal_context_quotas_compatible(required_quotas, quotas):
        raise SyncStoreError("personal_context_quota_incompatible")
    canonical = service._personal_context_service_for_user(user_id)
    repository = canonical._repository
    manifest = canonical.get_manifest()
    if expected_purge_generation is not None and expected_purge_generation != manifest.purge_generation:
        raise SyncStoreError("personal_context_purge_generation_stale")
    dataset = service.store.personal_context_dataset_for_profile(user_id=user_id, profile_id=manifest.profile_id)
    if dataset is None:
        raise SyncStoreError("personal_context_activation_required")
    dataset = service._require_dataset_access(user_id=user_id, dataset_id=dataset.dataset_id)
    integrity_key_id, integrity_key = canonical.sync_integrity_key(manifest.profile_id)
    activations = PersonalContextActivationService(repository)
    if not service.store.has_personal_context_link_receipt(
        user_id=user_id,
        dataset_id=dataset.dataset_id,
        device_id=device_id,
        profile_id=manifest.profile_id,
        integrity_key_id=integrity_key_id,
        purge_generation=manifest.purge_generation,
    ):
        raise SyncStoreError("personal_context_activation_required")
    if service.personal_context_relay is not None:
        service.personal_context_relay.relay_profile(
            user_id=user_id,
            profile_id=manifest.profile_id,
            dataset_id=dataset.dataset_id,
            after_server_cursor=None,
        )
    prepared = activations.prepare(manifest.profile_id, device_id=device_id)
    identity = _identity(prepared, dataset, user_id)
    cipher = _cipher(service, dataset)
    with service.store.personal_context_authority_guard(dataset.dataset_id, prepared.profile_id) as guarded:
        stored = guarded.get_personal_context_activation(prepared.activation_id)
        if stored is not None and _expired(stored):
            if not _verify_install(prepared, stored, identity, cipher):
                raise SyncStoreError("personal_context_activation_receipt_mismatch")
            with activations.publications.profile_lease(prepared.profile_id) as lease:
                repository.expire_activation(
                    prepared.activation_id, prepared.baseline_digest, sync_receipt_id=stored["receipt_id"], lease=lease
                )
            prepared = None
    if prepared is None:
        if service.personal_context_relay is not None:
            service.personal_context_relay.relay_profile(
                user_id=user_id, profile_id=manifest.profile_id, dataset_id=dataset.dataset_id, after_server_cursor=None
            )
        prepared = activations.prepare(manifest.profile_id, device_id=device_id)
        identity = _identity(prepared, dataset, user_id)
    with service.store.personal_context_authority_guard(dataset.dataset_id, prepared.profile_id) as guarded:

        def install(current: PreparedPersonalContextActivation) -> Mapping[str, Any]:
            """Commit Sync while the verified canonical generation guard remains held."""

            stored = guarded.get_personal_context_activation(current.activation_id)
            if stored is None:
                history = guarded.get_dataset_envelope_range(dataset.dataset_id)
                values = {
                    **identity,
                    "home_server_cursor": history.through_server_sequence or 0,
                    "envelopes_json": _protected_baseline(current, identity, cipher),
                    "expires_at": (
                        _activation_now() + timedelta(seconds=_PERSONAL_CONTEXT_BOOTSTRAP_TOKEN_TTL_SECONDS)
                    ).isoformat(),
                }
                values["receipt_id"] = hashlib.sha256(canonical_json_bytes(values)).hexdigest()
                stored = guarded.install_personal_context_activation(**values)
            if not _verify_install(current, stored, identity, cipher):
                raise SyncStoreError("personal_context_activation_receipt_mismatch")
            if _expired(stored):
                raise SyncStoreError("personal_context_activation_required")
            guarded.commit_personal_context_authority()
            return stored

        installed = activations.install(
            prepared.activation_id,
            prepared.baseline_digest,
            install=install,
            verify=lambda current, row: _verify_install(current, row, identity, cipher),
        )
    with service.store.personal_context_authority_guard(dataset.dataset_id, prepared.profile_id) as guarded:
        with activations.publications.profile_lease(prepared.profile_id) as lease:
            with repository.activation_install_guard(installed.activation_id, installed.baseline_digest, lease=lease):
                guarded.mirror_personal_context_activation(
                    dataset_id=dataset.dataset_id,
                    user_id=user_id,
                    profile_id=installed.profile_id,
                    purge_generation=installed.purge_generation,
                    activation_epoch=installed.activation_epoch,
                    continuity_token=installed.continuity_token,
                )
                guarded.commit_personal_context_authority()
    baseline = json.loads(installed.baseline)
    streams = service._pull_adapter_streams(device, list(PERSONAL_CONTEXT_SYNC_DOMAINS))
    watermarks = dict.fromkeys(streams, installed.home_server_cursor or 0)
    cursor = service._encode_pull_token(
        dataset_id=dataset.dataset_id,
        device_id=device_id,
        version_set=service._pull_version_set(device),
        watermarks=watermarks,
        ttl_seconds=max(service.settings.pull_token_ttl_seconds, _PERSONAL_CONTEXT_BOOTSTRAP_TOKEN_TTL_SECONDS),
    )
    manifest = ProfileManifest.model_validate(baseline["manifest"])
    scopes = tuple(ProfileScope.model_validate(x) for x in baseline["scopes"])
    records = tuple(ProfileRecord.model_validate(x) for x in baseline["records"])
    proposals = tuple(ProfileProposal.model_validate(x) for x in baseline["proposals"])
    bootstrap_cursor = canonical._sync_bootstrap_cursor(
        manifest=manifest, scopes=scopes, records=records, proposals=proposals, integrity_key_id=integrity_key_id
    )
    key_record = service._profile_manager()._device_integrity_key_record(
        user_id=user_id,
        dataset=dataset,
        device=device,
        integrity_key_id=integrity_key_id,
        integrity_key=integrity_key,
        bootstrap_cursor=bootstrap_cursor,
        transport_watermarks=watermarks,
    )
    return PersonalContextBootstrap(
        dataset_id=dataset.dataset_id,
        authority_id=service.personal_context_authority_id,
        manifest=manifest,
        scopes=scopes,
        records=records,
        proposals=proposals,
        purge_generation=manifest.purge_generation,
        schema_version=capability.max_schema_version,
        quotas=quotas,
        cursor=bootstrap_cursor,
        link_state="complete",
        integrity_key=PersonalContextBootstrapIntegrityKey(
            integrity_key_id=integrity_key_id,
            key_record_id=key_record.key_record_id,
            wrapped_key_blob=key_record.wrapped_key_blob,
        ),
        sync_transport_cursor=cursor,
        activation=_receipt(installed),
        personal_context_exchange=PersonalContextExchangeProof(
            ongoing_sync_version=1,
            activation_epoch=installed.activation_epoch,
            continuity_token=installed.continuity_token,
        ),
    )


def acknowledge_activation(
    service: SyncV2Service,
    *,
    user_id: str,
    dataset_id: str,
    device_id: str,
    activation_id: str,
    baseline_digest: str,
    local_receipt_id: str,
    exchange: PersonalContextExchangeProof,
) -> tuple[PersonalContextActivationReceipt, PersonalContextExchangeProof]:
    """Commit the exact Sync acknowledgment before marking the canonical device active."""

    dataset = service._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
    service._require_registered_device(user_id, device_id)
    repository = service._personal_context_service_for_user(user_id)._repository
    activations = PersonalContextActivationService(repository)
    prepared = repository.load_activation(activation_id)
    identity = _identity(prepared, dataset, user_id)
    cipher = _cipher(service, dataset)
    current = PersonalContextExchangeProof(
        ongoing_sync_version=1, activation_epoch=prepared.activation_epoch, continuity_token=prepared.continuity_token
    )
    if (
        prepared.device_id != device_id
        or prepared.baseline_digest != baseline_digest
        or not hmac.compare_digest(current.activation_epoch, exchange.activation_epoch)
        or not hmac.compare_digest(current.continuity_token, exchange.continuity_token)
    ):
        raise SyncStoreError("personal_context_activation_required")
    with service.store.personal_context_authority_guard(dataset_id, prepared.profile_id) as guarded:
        with activations.publications.profile_lease(prepared.profile_id) as lease:
            with repository.activation_install_guard(activation_id, baseline_digest, lease=lease) as live:
                stored = guarded.get_personal_context_activation(activation_id)
                if stored is None or not _verify_install(live, stored, identity, cipher):
                    raise SyncStoreError("personal_context_activation_receipt_mismatch")
                if _expired(stored):
                    raise SyncStoreError("personal_context_activation_required")
                ack = guarded.acknowledge_personal_context_activation(
                    activation_id=activation_id,
                    dataset_id=dataset_id,
                    user_id=user_id,
                    device_id=device_id,
                    baseline_digest=baseline_digest,
                    local_receipt_id=local_receipt_id,
                )
                guarded.commit_personal_context_authority()
            active = repository.confirm_activation_device(
                activation_id,
                baseline_digest,
                device_id,
                ack["receipt_id"],
                local_receipt_id=local_receipt_id,
                dataset_id=dataset_id,
            )
    return _receipt(active), current
