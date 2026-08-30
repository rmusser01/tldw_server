from __future__ import annotations

import hashlib
import hmac
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.personal_context import (
    PersonalContextDomainAdapter,
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
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.service import (
    PersonalContextSyncCapabilities,
    SyncV2Service,
    SyncV2Settings,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
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
        return "personal-context-integrity-v1", _INTEGRITY_KEY

    def sync_bootstrap_snapshot(self):
        entries = [
            f"manifest:{self.manifest.profile_id}:{self.manifest.current_version_id}",
            f"purge:{self.manifest.purge_generation}",
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
            "integrity_key_id": "personal-context-integrity-v1",
            "integrity_key": _INTEGRITY_KEY,
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
        "authority_id": "authority-a",
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
            authority_id="authority-a",
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
