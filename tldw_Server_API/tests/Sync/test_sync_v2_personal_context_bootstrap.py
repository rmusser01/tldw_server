from __future__ import annotations

import base64
import hashlib
import hmac
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.core.DB_Management import Sync_DB as sync_db_module
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.personal_context import (
    PersonalContextDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.errors import (
    SyncIdempotencyConflictError,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.factory import (
    _personal_context_wrapping_key_fingerprint,
    _wrap_personal_context_integrity_key,
)
from tldw_Server_API.app.core.Sync.v2.materializers.personal_context import (
    PersonalContextMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    PERSONAL_CONTEXT_SYNC_DOMAINS,
    SyncDatasetCreate,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import (
    PersonalContextSyncCapabilities,
    SyncV2Service,
    SyncV2Settings,
)
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    global_scope,
    manifest,
    preference_record,
    proposal,
)

pytestmark = pytest.mark.unit

_INTEGRITY_KEY = b"k" * 32
_PLAINTEXT_CANARY = "bootstrap-private-canary"


class _CanonicalService:
    def __init__(self) -> None:
        self.manifest = manifest()
        self.scope = global_scope(profile_id=self.manifest.profile_id)
        self.record = preference_record(
            profile_id=self.manifest.profile_id,
            value=_PLAINTEXT_CANARY,
        )
        self.proposal = proposal(
            profile_id=self.manifest.profile_id,
        )
        self.records = (self.record,)
        self.proposals = (self.proposal,)
        self.profile_exists = True
        self.created = 0
        self.applied: list[object] = []
        self.integrity_key_id = "personal-context-integrity-v1"
        self.integrity_key = _INTEGRITY_KEY

    def create_profile(self, *, runtime_enabled: bool = False):
        del runtime_enabled
        self.created += 1
        self.profile_exists = True
        return self.manifest

    def get_manifest(self):
        if not self.profile_exists:
            raise KeyError("profile")
        return self.manifest

    def ensure_sync_profile(self):
        if not self.profile_exists:
            return self.create_profile()
        return self.manifest

    def list_scopes(self):
        return (self.scope,)

    def list_records(self, *, include_archived: bool = False):
        del include_archived
        return self.records

    def list_proposals(self, *, pending_only: bool = True, limit: int = 50, offset: int = 0):
        del pending_only, limit, offset
        return self.proposals

    def sync_integrity_key(self, profile_id: str) -> tuple[str, bytes]:
        assert profile_id == self.manifest.profile_id
        return self.integrity_key_id, self.integrity_key

    def sync_bootstrap_snapshot(self):
        entries = [
            f"manifest:{self.manifest.profile_id}:{self.manifest.current_version_id}",
            f"purge:{self.manifest.purge_generation}",
            f"integrity:{self.integrity_key_id}",
            f"scope:{self.scope.scope_id}:{self.scope.version_id}",
            *(
                f"record:{item.record_id}:{item.version_id}" for item in self.records
            ),
            *(
                "proposal:" + item.proposal_id + ":" + hashlib.sha256(
                    item.model_dump_json().encode("utf-8")
                ).hexdigest()
                for item in self.proposals
            ),
        ]
        return type("Snapshot", (), {
            "manifest": self.manifest, "scopes": self.list_scopes(),
            "records": self.records, "proposals": self.proposals,
            "integrity_key_id": self.integrity_key_id,
            "integrity_key": self.integrity_key,
            "cursor": "personal-context-bootstrap-v1:" + hashlib.sha256(
                "\x1e".join(sorted(entries)).encode("utf-8")
            ).hexdigest(),
        })()

    def apply_sync_object(self, **values: object) -> object:
        self.applied.append(values)
        return values["value"]


def _service(tmp_path: Path) -> tuple[SyncV2Service, _CanonicalService]:
    canonical = _CanonicalService()
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
    adapters = SyncAdapterRegistry(
        [
            PersonalContextDomainAdapter(
                domain=domain,
                integrity_key_resolver=lambda _dataset, _key_id: _INTEGRITY_KEY,
                encryption_key_resolver=lambda _dataset: (b"e" * 32, 1),
            )
            for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
        ]
    )
    service = SyncV2Service(
        store=store,
        adapters=adapters,
        materializers={
            domain: PersonalContextMaterializer(
                domain=domain,
                service_resolver=lambda _user_id: canonical,
            )
            for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
        },
        personal_context_service_resolver=lambda _user_id: canonical,
        personal_context_key_wrapper=lambda *, device, integrity_key, integrity_key_id: (
            "wrapped:"
            f"{device.device_id}:{integrity_key_id}:"
            f"{hashlib.sha256(integrity_key).hexdigest()}"
        ),
        personal_context_key_fingerprint=lambda *, device: f"fingerprint:{device.device_id}",
        settings=SyncV2Settings(
            personal_context=PersonalContextSyncCapabilities(available=True, blockers=()),
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="managed_storage",
                server_trusted_enabled=True,
                auth_mode="multi_user",
            ),
        ),
    )
    service.register_device(
        user_id="user-a",
        display_name="Chatbook A",
        client_type="chatbook",
        device_id="device-a",
        capabilities={
            "supported_adapter_versions": {
                domain: [1] for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
            }
        },
    )
    return service, canonical


def _bootstrap(service: SyncV2Service, **overrides: object):
    values: dict[str, object] = {
        "user_id": "user-a",
        "device_id": "device-a",
        "required_schema_version": 1,
        "required_quotas": {"max_record_bytes": 16_384},
    }
    values.update(overrides)
    return service.bootstrap_personal_context(**values)




def _record_envelope(bootstrap) -> SyncEnvelopeCreate:
    payload = preference_record(
        profile_id=bootstrap.manifest.profile_id,
    ).model_dump(mode="json")
    tag = hmac.new(_INTEGRITY_KEY, canonical_json_bytes(payload), hashlib.sha256)
    return SyncEnvelopeCreate(
        dataset_id=bootstrap.dataset_id,
        client_envelope_id="device-a:record:1",
        device_id="device-a",
        domain="personal_context.record",
        operation="upsert",
        object_id=str(payload["record_id"]),
        parent_id=str(payload["scope_id"]),
        adapter_version=1,
        schema_version=1,
        payload=payload,
        payload_hash=f"hmac-sha256-v1:{tag.hexdigest()}",
        payload_size_bytes=len(canonical_json_bytes(payload)),
        routing_metadata={
            "integrity_key_id": bootstrap.integrity_key.integrity_key_id,
            "profile_id": bootstrap.manifest.profile_id,
            "purge_generation": bootstrap.purge_generation,
        },
        encryption_metadata={"policy": "server_trusted_v1"},
    )


def test_bootstrap_returns_canonical_heads_and_one_cursor_for_empty_or_populated_profile(
    tmp_path: Path,
) -> None:
    service, canonical = _service(tmp_path)
    populated = _bootstrap(service)

    assert populated.manifest == canonical.manifest
    assert populated.scopes == (canonical.scope,)
    assert populated.records == (canonical.record,)
    assert populated.proposals == (canonical.proposal,)
    assert populated.cursor
    assert populated.cursor == _bootstrap(service).cursor
    assert populated.link_state == "bootstrap_pending"

    canonical.records = ()
    canonical.proposals = ()
    empty = _bootstrap(service)

    assert empty.manifest == canonical.manifest
    assert empty.scopes == (canonical.scope,)
    assert empty.records == ()
    assert empty.proposals == ()
    assert empty.cursor != populated.cursor


def test_bootstrap_is_idempotent_and_serializes_first_profile_link(tmp_path: Path) -> None:
    service, canonical = _service(tmp_path)
    canonical.profile_exists = False

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, retry = tuple(executor.map(lambda _index: _bootstrap(service), range(2)))

    assert retry.dataset_id == first.dataset_id
    assert retry.manifest.profile_id == first.manifest.profile_id
    assert canonical.created == 1


def test_concurrent_absent_personal_context_bindings_accept_identical_winner(
    tmp_path: Path,
) -> None:
    """Two callers that pre-read absence converge on the first durable binding."""

    service, canonical = _service(tmp_path)
    dataset = service.store.get_or_create_default_personal_dataset("user-a")
    barrier = threading.Barrier(2)

    def bind_from_absent_read():
        barrier.wait(timeout=5)
        return service.store.bind_personal_context_dataset(
            dataset_id=dataset.dataset_id,
            user_id="user-a",
            expected_binding=None,
            profile_id=canonical.manifest.profile_id,
            authority_id="authority-a",
            integrity_key_id=canonical.integrity_key_id,
            purge_generation=canonical.manifest.purge_generation,
            link_state="bootstrap_pending",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = tuple(executor.map(lambda _index: bind_from_absent_read(), range(2)))

    assert first.dataset_id == second.dataset_id == dataset.dataset_id
    durable = service.store.get_dataset(dataset.dataset_id, owner_user_id="user-a")
    assert durable is not None
    assert durable.metadata["personal_context"]["integrity_key_id"] == canonical.integrity_key_id


def test_stale_personal_context_binding_cannot_overwrite_new_key_or_receipt(
    tmp_path: Path,
) -> None:
    """A caller with v1 state cannot replace a committed v2 binding under lock."""

    service, canonical = _service(tmp_path)
    bootstrap = _bootstrap(service)
    dataset = service.store.get_dataset(bootstrap.dataset_id, owner_user_id="user-a")
    assert dataset is not None
    before = dataset.metadata["personal_context"]
    service.store.bind_personal_context_dataset(
        dataset_id=dataset.dataset_id,
        user_id="user-a",
        expected_binding=dict(before),
        profile_id=str(before["profile_id"]),
        authority_id=str(before["authority_id"]),
        integrity_key_id="personal-context-integrity-v2",
        purge_generation=2,
        link_state=str(before["link_state"]),
    )
    service.store.complete_personal_context_link_receipt(
        user_id="user-a",
        dataset_id=dataset.dataset_id,
        device_id="device-a",
        profile_id=str(before["profile_id"]),
        integrity_key_id="personal-context-integrity-v2",
        purge_generation=2,
        bootstrap_cursor="v2-cursor",
    )

    with pytest.raises(SyncStoreError, match="personal_context_link_binding_stale"):
        service.store.bind_personal_context_dataset(
            dataset_id=dataset.dataset_id,
            user_id="user-a",
            expected_binding=dict(before),
            profile_id=str(before["profile_id"]),
            authority_id=str(before["authority_id"]),
            integrity_key_id=str(before["integrity_key_id"]),
            purge_generation=int(before["purge_generation"]),
            link_state=str(before["link_state"]),
        )

    durable = service.store.get_dataset(dataset.dataset_id, owner_user_id="user-a")
    assert durable is not None
    assert durable.metadata["personal_context"]["integrity_key_id"] == "personal-context-integrity-v2"
    assert durable.metadata["personal_context"]["purge_generation"] == 2
    assert service.store.has_personal_context_link_receipt(
        user_id="user-a",
        dataset_id=dataset.dataset_id,
        device_id="device-a",
        profile_id=str(before["profile_id"]),
        integrity_key_id="personal-context-integrity-v2",
        purge_generation=2,
    )


def test_personal_context_bootstrap_service_accepts_no_client_authority_argument() -> None:
    """Authority is server configured rather than a client-callable service input."""

    assert "authority_id" not in inspect.signature(
        SyncV2Service.bootstrap_personal_context
    ).parameters


def test_generic_enrollment_has_no_personal_context_escape_hatch() -> None:
    """Generic enrollment always retains server-owned Personal Context state."""

    assert "preserve_personal_context" not in inspect.signature(
        SyncDatabase.enroll_dataset
    ).parameters


def test_bootstrap_returns_registered_device_wrapped_integrity_key_only(tmp_path: Path) -> None:
    service, _canonical = _service(tmp_path)

    bootstrap = _bootstrap(service)

    assert bootstrap.integrity_key.integrity_key_id == "personal-context-integrity-v1"
    assert bootstrap.integrity_key.wrapped_key_blob.startswith("wrapped:device-a:")
    records = service.store.list_key_records(
        bootstrap.dataset_id,
        user_id="user-a",
        device_id="device-a",
        key_purpose="personal_context_integrity",
    )
    assert [record.wrapped_key_blob for record in records] == [
        bootstrap.integrity_key.wrapped_key_blob
    ]
    assert _INTEGRITY_KEY.hex() not in str(bootstrap)


def test_concurrent_real_rsa_bootstraps_share_one_durable_wrapper(tmp_path: Path) -> None:
    """Randomized RSA wrappers converge on the durable winner after an insert race."""

    service, _canonical = _service(tmp_path)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    service.register_device(
        user_id="user-a",
        display_name="Chatbook A",
        client_type="chatbook",
        device_id="device-a",
        capabilities={
            "supported_adapter_versions": {
                domain: [1] for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
            },
            "personal_context_wrapping_public_key": public_key,
        },
    )
    wrapper_barrier = threading.Barrier(2)

    def synchronized_real_wrapper(**kwargs: object) -> str:
        wrapper_barrier.wait(timeout=5)
        return _wrap_personal_context_integrity_key(**kwargs)

    service.personal_context_key_wrapper = synchronized_real_wrapper
    service.personal_context_key_fingerprint = _personal_context_wrapping_key_fingerprint

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = tuple(executor.map(lambda _index: _bootstrap(service), range(2)))

    records = service.store.list_key_records(
        first.dataset_id,
        user_id="user-a",
        device_id="device-a",
        key_purpose="personal_context_integrity",
    )
    assert len(records) == 1
    assert first.integrity_key.wrapped_key_blob == records[0].wrapped_key_blob
    assert second.integrity_key.wrapped_key_blob == records[0].wrapped_key_blob
    ciphertext = base64.urlsafe_b64decode(records[0].wrapped_key_blob.split(":", 1)[1])
    assert private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=b"personal-context:personal-context-integrity-v1",
        ),
    ) == _INTEGRITY_KEY
    assert _INTEGRITY_KEY.hex() not in str((first, second, records))


@pytest.mark.parametrize(
    "mismatch",
    [
        "owner",
        "dataset",
        "device",
        "purpose",
        "wrapped_for",
        "rewrap_status",
        "revoked",
        "policy",
        "integrity_key_id",
        "wrapping_key_fingerprint",
    ],
)
def test_conflicting_rsa_key_record_winner_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    """A conflicting wrapper row is accepted only when every durable fence matches."""

    service, canonical = _service(tmp_path)
    bootstrap = _bootstrap(service)
    dataset = service.store.get_dataset(bootstrap.dataset_id, owner_user_id="user-a")
    device = service.store.get_device("user-a", "device-a")
    records = service.store.list_key_records(
        bootstrap.dataset_id,
        user_id="user-a",
        device_id="device-a",
        key_purpose="personal_context_integrity",
    )
    assert dataset is not None and device is not None and len(records) == 1
    winner = records[0]
    values: dict[str, object] = {}
    if mismatch == "owner":
        values["user_id"] = "foreign-user"
    elif mismatch == "dataset":
        values["dataset_id"] = "foreign-dataset"
    elif mismatch == "device":
        values["device_id"] = "foreign-device"
    elif mismatch == "purpose":
        values["key_purpose"] = "foreign-purpose"
    elif mismatch == "wrapped_for":
        values["wrapped_for"] = "recovery"
    elif mismatch == "rewrap_status":
        values["rewrap_status"] = "not_required"
    elif mismatch == "revoked":
        values["revoked_at"] = "2026-08-30T00:00:00+00:00"
    elif mismatch == "policy":
        values["encryption_policy"] = "server_trusted_v1"
    elif mismatch == "integrity_key_id":
        values["kdf_metadata"] = {
            **winner.kdf_metadata,
            "integrity_key_id": "foreign-integrity-key",
        }
    else:
        values["kdf_metadata"] = {
            **winner.kdf_metadata,
            "wrapping_key_fingerprint": "foreign-fingerprint",
        }
    mismatched_winner = replace(winner, **values)
    list_calls = 0

    def hide_then_return_mismatch(*_args: object, **_kwargs: object):
        nonlocal list_calls
        list_calls += 1
        return [] if list_calls == 1 else [mismatched_winner]

    def conflict_insert(*_args: object, **_kwargs: object):
        raise SyncIdempotencyConflictError("different randomized wrapper")

    monkeypatch.setattr(service.store, "list_key_records", hide_then_return_mismatch)
    monkeypatch.setattr(service.store, "store_key_record", conflict_insert)

    with pytest.raises(Exception) as exc_info:
        service._profile_manager()._device_integrity_key_record(
            user_id="user-a",
            dataset=dataset,
            device=device,
            integrity_key_id=canonical.integrity_key_id,
            integrity_key=canonical.integrity_key,
        )

    assert getattr(exc_info.value, "reason_code", None) == "personal_context_key_custody_unavailable"
    assert winner.wrapped_key_blob not in str(exc_info.value)


def test_bootstrap_requires_transactional_canonical_snapshot(tmp_path: Path) -> None:
    """A legacy canonical service cannot bypass the atomic snapshot boundary."""

    service, _canonical = _service(tmp_path)
    service.personal_context_service_resolver = lambda _user_id: object()

    with pytest.raises(Exception) as exc_info:
        _bootstrap(service)

    assert getattr(exc_info.value, "reason_code", None) == "personal_context_snapshot_unavailable"


def test_integrity_key_transition_invalidates_old_completion_and_rebinds_fresh_bootstrap(
    tmp_path: Path,
) -> None:
    """Link admission is bound to the canonical integrity-key generation."""

    service, canonical = _service(tmp_path)
    first = _bootstrap(service)
    canonical.integrity_key_id = "personal-context-integrity-v2"
    canonical.integrity_key = b"b" * 32
    second = _bootstrap(service)

    assert second.cursor != first.cursor
    assert second.integrity_key.integrity_key_id == "personal-context-integrity-v2"
    assert "personal-context-integrity-v2" in second.integrity_key.wrapped_key_blob

    with pytest.raises(Exception) as exc_info:
        service.complete_personal_context_link(
            user_id="user-a",
            device_id="device-a",
            dataset_id=first.dataset_id,
            bootstrap_cursor=first.cursor,
        )
    assert getattr(exc_info.value, "reason_code", None) == "personal_context_bootstrap_cursor_stale"
    assert not service.store.has_personal_context_link_receipt(
        user_id="user-a",
        dataset_id=first.dataset_id,
        device_id="device-a",
        profile_id=canonical.manifest.profile_id,
        integrity_key_id="personal-context-integrity-v2",
        purge_generation=canonical.manifest.purge_generation,
    )

    service.complete_personal_context_link(
        user_id="user-a",
        device_id="device-a",
        dataset_id=second.dataset_id,
        bootstrap_cursor=second.cursor,
    )
    assert service.store.has_personal_context_link_receipt(
        user_id="user-a",
        dataset_id=second.dataset_id,
        device_id="device-a",
        profile_id=canonical.manifest.profile_id,
        integrity_key_id="personal-context-integrity-v2",
        purge_generation=canonical.manifest.purge_generation,
    )


@pytest.mark.parametrize(
    ("user_id", "device_id", "reason"),
    [
        ("user-b", "device-a", "personal_context_device_unavailable"),
        ("user-a", "missing-device", "personal_context_device_unavailable"),
    ],
)
def test_bootstrap_fails_closed_before_profile_disclosure_for_wrong_identity(
    tmp_path: Path,
    user_id: str,
    device_id: str,
    reason: str,
) -> None:
    service, _canonical = _service(tmp_path)

    with pytest.raises(Exception) as exc_info:
        service.bootstrap_personal_context(
            user_id=user_id,
            device_id=device_id,
        )

    assert getattr(exc_info.value, "reason_code", None) == reason
    assert _PLAINTEXT_CANARY not in str(exc_info.value)
    assert _INTEGRITY_KEY.hex() not in str(exc_info.value)


def test_bootstrap_blocks_capability_schema_quota_purge_and_key_custody_without_body(
    tmp_path: Path,
) -> None:
    service, canonical = _service(tmp_path)
    cases = [
        ({"required_schema_version": 2}, "personal_context_schema_incompatible"),
        ({"required_quotas": {"max_record_bytes": 16_385}}, "personal_context_quota_incompatible"),
        ({"expected_purge_generation": 1}, "personal_context_purge_generation_stale"),
    ]
    for kwargs, reason in cases:
        with pytest.raises(Exception) as exc_info:
            _bootstrap(service, **kwargs)
        assert getattr(exc_info.value, "reason_code", None) == reason
        assert _PLAINTEXT_CANARY not in str(exc_info.value)

    service.personal_context_key_wrapper = None
    with pytest.raises(Exception) as exc_info:
        _bootstrap(service)
    assert (
        getattr(exc_info.value, "reason_code", None)
        == "personal_context_key_custody_unavailable"
    )
    assert canonical.created == 0


def test_bootstrap_rejects_revoked_or_partially_ready_service_without_disclosure(
    tmp_path: Path,
) -> None:
    service, _canonical = _service(tmp_path)
    service.revoke_device(user_id="user-a", device_id="device-a")

    with pytest.raises(Exception) as exc_info:
        _bootstrap(service)
    assert getattr(exc_info.value, "reason_code", None) == "personal_context_device_unavailable"
    assert _PLAINTEXT_CANARY not in str(exc_info.value)

    service, _canonical = _service(tmp_path / "partial")
    service.settings = replace(
        service.settings,
        personal_context=PersonalContextSyncCapabilities(
            available=False,
            blockers=("personal_context_transport_unavailable",),
        ),
    )
    with pytest.raises(Exception) as exc_info:
        _bootstrap(service)
    assert (
        getattr(exc_info.value, "reason_code", None)
        == "personal_context_capability_unavailable"
    )


def test_personal_context_push_stays_blocked_until_narrow_completion_transition(
    tmp_path: Path,
) -> None:
    service, _canonical = _service(tmp_path)
    bootstrap = _bootstrap(service)
    envelope = _record_envelope(bootstrap)

    before = service.push(
        user_id="user-a",
        dataset_id=bootstrap.dataset_id,
        device_id="device-a",
        envelopes=[envelope],
    )
    assert before.accepted == []
    assert before.rejected[0].error_code == "personal_context_link_incomplete"

    service.complete_personal_context_link(
        user_id="user-a",
        device_id="device-a",
        dataset_id=bootstrap.dataset_id,
        bootstrap_cursor=bootstrap.cursor,
    )
    after = service.push(
        user_id="user-a",
        dataset_id=bootstrap.dataset_id,
        device_id="device-a",
        envelopes=[envelope],
    )
    assert len(after.accepted) == 1

    service.register_device(
        user_id="user-a",
        display_name="Chatbook B",
        client_type="chatbook",
        device_id="device-b",
        capabilities={
            "supported_adapter_versions": {
                domain: [1] for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
            }
        },
    )
    other_device = replace(
        envelope,
        client_envelope_id="device-b:record:1",
        device_id="device-b",
    )
    blocked_other_device = service.push(
        user_id="user-a",
        dataset_id=bootstrap.dataset_id,
        device_id="device-b",
        envelopes=[other_device],
    )
    assert blocked_other_device.rejected[0].error_code == "personal_context_link_incomplete"


def test_generic_enrollment_cannot_forge_personal_context_binding(tmp_path: Path) -> None:
    service, _canonical = _service(tmp_path)

    with pytest.raises(Exception) as exc_info:
        service.enroll_dataset(
            user_id="user-a",
            domains=["personal_context.record"],
            metadata={"personal_context": {"link_state": "complete"}},
        )

    assert "sync_reserved_dataset_enrollment" in str(exc_info.value)


def test_generic_reenrollment_preserves_server_owned_personal_context_state(
    tmp_path: Path,
) -> None:
    """An old generic client cannot erase canonical Personal Context admission state."""

    service, _canonical = _service(tmp_path)
    bootstrap = _bootstrap(service)
    before = service.store.get_dataset(bootstrap.dataset_id)
    assert before is not None
    assert "personal_context" in before.metadata

    service.enroll_dataset(
        user_id="user-a",
        dataset_id=bootstrap.dataset_id,
        domains=["notes.note"],
        metadata={"label": "old-client-update"},
    )

    durable = service.store.get_dataset(bootstrap.dataset_id)
    assert durable is not None
    binding = durable.metadata["personal_context"]
    assert set(PERSONAL_CONTEXT_SYNC_DOMAINS).issubset(durable.domains)
    assert binding["profile_id"] == bootstrap.manifest.profile_id
    assert binding["integrity_key_id"] == bootstrap.integrity_key.integrity_key_id
    assert binding["purge_generation"] == bootstrap.purge_generation


def test_personal_context_binding_preserves_update_committed_before_its_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A binding refresh merges the locked row instead of a stale profile snapshot."""

    service, _canonical = _service(tmp_path)
    first = _bootstrap(service)
    original_bind = service.store.bind_personal_context_dataset
    interleaved = False

    def bind_after_ordinary_update(**kwargs: object):
        nonlocal interleaved
        if not interleaved:
            interleaved = True
            current = service.store.get_dataset(
                str(kwargs["dataset_id"]),
                owner_user_id="user-a",
            )
            assert current is not None
            service.store.enroll_dataset(
                SyncDatasetCreate(
                    dataset_id=current.dataset_id,
                    owner_user_id=current.owner_user_id,
                    scope_type=current.scope_type,
                    encryption_policy=current.encryption_policy,
                    domains=[*current.domains, "source_cache.entry"],
                    workspace_id=current.workspace_id,
                    metadata={**current.metadata, "ordinary_update": "keep"},
                    archived_at=current.archived_at,
                )
            )
        return original_bind(**kwargs)

    monkeypatch.setattr(
        service.store,
        "bind_personal_context_dataset",
        bind_after_ordinary_update,
    )
    refreshed = _bootstrap(service)

    durable = service.store.get_dataset(first.dataset_id, owner_user_id="user-a")
    assert refreshed.dataset_id == first.dataset_id
    assert durable is not None
    assert durable.metadata["ordinary_update"] == "keep"
    assert "source_cache.entry" in durable.domains


def test_personal_context_binding_rejects_profile_or_authority_mismatch_under_lock(
    tmp_path: Path,
) -> None:
    """The narrow binding operation refuses a state changed before its row lock."""

    service, _canonical = _service(tmp_path)
    bootstrap = _bootstrap(service)
    dataset = service.store.get_dataset(bootstrap.dataset_id, owner_user_id="user-a")
    assert dataset is not None
    binding = dataset.metadata["personal_context"]

    with pytest.raises(SyncStoreError, match="personal_context_link_binding_stale"):
        service.store.bind_personal_context_dataset(
            dataset_id=dataset.dataset_id,
            user_id="user-a",
            expected_binding={**binding, "profile_id": "other-profile"},
            profile_id="new-profile",
            authority_id=str(binding["authority_id"]),
            integrity_key_id=str(binding["integrity_key_id"]),
            purge_generation=int(binding["purge_generation"]),
            link_state=str(binding["link_state"]),
        )


def test_factory_wraps_integrity_key_to_registered_device_public_key() -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    device = type(
        "Device",
        (),
        {"capabilities": {"personal_context_wrapping_public_key": public_key}},
    )()

    wrapped = _wrap_personal_context_integrity_key(
        device=device,
        integrity_key=_INTEGRITY_KEY,
        integrity_key_id="personal-context-integrity-v1",
    )

    ciphertext = base64.urlsafe_b64decode(wrapped.split(":", 1)[1])
    assert private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=b"personal-context:personal-context-integrity-v1",
        ),
    ) == _INTEGRITY_KEY


def test_bootstrap_rewraps_after_registered_public_key_rotation(tmp_path: Path) -> None:
    service, _canonical = _service(tmp_path)
    old_private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    new_private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    def register_key(private_key) -> None:
        service.register_device(
            user_id="user-a", display_name="Chatbook A", client_type="chatbook",
            device_id="device-a", capabilities={
                "supported_adapter_versions": {domain: [1] for domain in PERSONAL_CONTEXT_SYNC_DOMAINS},
                "personal_context_wrapping_public_key": private_key.public_key().public_bytes(
                    serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode("utf-8"),
            },
        )
    service.personal_context_key_wrapper = _wrap_personal_context_integrity_key
    service.personal_context_key_fingerprint = _personal_context_wrapping_key_fingerprint
    register_key(old_private)
    first = _bootstrap(service)
    register_key(new_private)
    second = _bootstrap(service)
    assert first.integrity_key.key_record_id != second.integrity_key.key_record_id
    ciphertext = base64.urlsafe_b64decode(second.integrity_key.wrapped_key_blob.split(":", 1)[1])
    assert new_private.decrypt(ciphertext, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=b"personal-context:personal-context-integrity-v1")) == _INTEGRITY_KEY
    with pytest.raises(ValueError):
        old_private.decrypt(ciphertext, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=b"personal-context:personal-context-integrity-v1"))


def test_link_receipt_schema_and_cas_are_portable_and_not_request_time_ddl() -> None:
    for schema in (sync_db_module.SYNC_SQLITE_SCHEMA, sync_db_module.SYNC_POSTGRES_SCHEMA):
        assert "CREATE TABLE IF NOT EXISTS sync_personal_context_link_receipts" in schema
        assert "PRIMARY KEY (user_id, dataset_id, device_id)" in schema
    source = inspect.getsource(SyncDatabase.complete_personal_context_link_receipt)
    assert "CREATE TABLE" not in source
    assert "DELETE FROM sync_personal_context_link_receipts" not in source
    assert "ON CONFLICT(user_id, dataset_id, device_id) DO UPDATE" in source
    assert "personal_context_link_binding_stale" in source




def test_bootstrap_never_persists_plaintext_in_sync_metadata_or_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    service, _canonical = _service(tmp_path)

    bootstrap = _bootstrap(service)
    durable = b"".join(path.read_bytes() for path in tmp_path.iterdir())

    assert _PLAINTEXT_CANARY.encode() not in durable
    assert _INTEGRITY_KEY not in durable
    assert _PLAINTEXT_CANARY not in caplog.text
    assert _INTEGRITY_KEY.hex() not in caplog.text
    assert _PLAINTEXT_CANARY not in str(service.store.get_dataset(bootstrap.dataset_id).metadata)
