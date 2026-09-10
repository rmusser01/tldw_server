"""Bootstrap compatibility, cursor, and first-link authorization regressions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import inspect
import json
import threading
import uuid
from concurrent.futures import (
    ThreadPoolExecutor,
)
from concurrent.futures import (
    TimeoutError as FutureTimeoutError,
)
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from tldw_profile_core import ProfileRecord
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.core.DB_Management import Sync_DB as sync_db_module
from tldw_Server_API.app.core.DB_Management.Personal_Context_Key_Store import ServerProfileKeyProvider
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    CanonicalApplyReceipt,
    IngressIdentity,
    PersonalContextPublicationJournal,
    PersonalContextPublicationRelayStore,
    PublicationObject,
)
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    PersonalContextService,
)
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
    SyncEnvelope,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.personal_context_ongoing_contract import PersonalContextExchangeProof
from tldw_Server_API.app.core.Sync.v2.personal_context_relay import PersonalContextRelay
from tldw_Server_API.app.core.Sync.v2.profile import PersonalContextBootstrap, PersonalContextBootstrapError
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
    encoded_master_key,
    global_scope,
    manifest,
    preference_record,
    proposal,
)

pytestmark = pytest.mark.unit

_INTEGRITY_KEY = b"k" * 32
_PLAINTEXT_CANARY = "bootstrap-private-canary"
_EXCHANGE = PersonalContextExchangeProof(
    ongoing_sync_version=1,
    activation_epoch="epoch_0123456789abcdef",
    continuity_token="continuity_0123456789abcdef",
)


class _CanonicalService:
    """Canonical snapshot fake with explicit per-device activation authorization."""

    database: PersonalizationDB

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
        self._repository = self
        self.activated_devices: set[tuple[str, str, str]] = set()

    def validate_activation_exchange(
        self,
        *,
        profile_id: str,
        device_id: str,
        dataset_id: str,
        activation_epoch: str,
        continuity_token: str,
    ) -> PersonalContextExchangeProof:
        """Accept only explicitly acknowledged fixture devices and their exact pair."""
        proof = PersonalContextExchangeProof(
            ongoing_sync_version=1,
            activation_epoch=activation_epoch,
            continuity_token=continuity_token,
        )
        if (profile_id, dataset_id, device_id) not in self.activated_devices or proof != _EXCHANGE:
            raise ValueError("personal_context_activation_required")
        return proof

    def create_profile(self, *, runtime_enabled: bool = False):
        del runtime_enabled
        self.created += 1
        self.profile_exists = True
        return self.manifest

    def get_manifest(self):
        if not self.profile_exists:
            raise KeyError("profile")
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
            *(f"record:{item.record_id}:{item.version_id}" for item in self.records),
            *(
                "proposal:"
                + item.proposal_id
                + ":"
                + hashlib.sha256(item.model_dump_json().encode("utf-8")).hexdigest()
                for item in self.proposals
            ),
        ]
        return type(
            "Snapshot",
            (),
            {
                "manifest": self.manifest,
                "scopes": self.list_scopes(),
                "records": self.records,
                "proposals": self.proposals,
                "integrity_key_id": self.integrity_key_id,
                "integrity_key": self.integrity_key,
                "cursor": "personal-context-bootstrap-v1:"
                + hashlib.sha256("\x1e".join(sorted(entries)).encode("utf-8")).hexdigest(),
            },
        )()

    def plan_sync_bootstrap(self):
        snapshot = self.sync_bootstrap_snapshot()
        snapshot.materialized = self.profile_exists
        return snapshot

    def materialize_sync_bootstrap(self, *, profile_id: str, bootstrap_cursor: str):
        snapshot = self.plan_sync_bootstrap()
        if profile_id != snapshot.manifest.profile_id or bootstrap_cursor != snapshot.cursor:
            raise ValueError("stale bootstrap plan")
        self.profile_exists = True
        snapshot.materialized = True
        return snapshot

    def apply_sync_object(self, **values: object) -> object:
        self.applied.append(values)
        return values["value"]

    def apply_sync_ingress(
        self,
        *,
        identity: IngressIdentity,
        domain: str,
        value: ProfileRecord,
        base_object_hash: str | None,
    ) -> CanonicalApplyReceipt:
        """Exercise real ingress receipt validation around the existing apply fake."""
        self.apply_sync_object(domain=domain, value=value, base_object_hash=base_object_hash)
        return CanonicalApplyReceipt(
            resulting_object_id=value.record_id,
            resulting_version_id=value.version_id,
            manifest_revision=self.manifest.revision,
            manifest_version_id=self.manifest.current_version_id,
            purge_generation=identity.purge_generation,
            publication_batch_id="bootstrap-fixture-batch",
            profile_publication_sequence=1,
            receipt_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    "tldw:personal-context:ingress:"
                    f"{identity.dataset_id}:{identity.device_id}:{identity.client_envelope_id}",
                )
            ),
            dataset_id=identity.dataset_id,
            device_id=identity.device_id,
            client_envelope_id=identity.client_envelope_id,
            canonical_payload_digest=identity.canonical_payload_digest,
            wire_entity_version=identity.wire_entity_version,
        )


def _service(
    tmp_path: Path, *, publication_journal: bool = False, monkeypatch: pytest.MonkeyPatch | None = None
) -> tuple[SyncV2Service, _CanonicalService]:
    """Build real Sync storage, optionally with real encrypted authority sources."""
    canonical = _CanonicalService()
    if publication_journal:
        assert monkeypatch is not None
        monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
        canonical.database = PersonalizationDB.for_path(tmp_path / "publication.db")
        provider = ServerProfileKeyProvider(canonical.database)
        with canonical.database.transaction(immediate=True) as connection:
            keys = provider.create(canonical.manifest.profile_id, connection=connection)
            provider.replace_encryption_key(
                canonical.manifest.profile_id,
                encryption_key=b"e" * 32,
                integrity_key=_INTEGRITY_KEY,
                expected_key_version=keys.key_version,
                integrity_key_version=keys.integrity_key_version,
                connection=connection,
            )
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
            f"wrapped:{device.device_id}:{integrity_key_id}:{hashlib.sha256(integrity_key).hexdigest()}"
        ),
        personal_context_key_fingerprint=lambda *, device: f"fingerprint:{device.device_id}",
        settings=SyncV2Settings(
            personal_context=PersonalContextSyncCapabilities(available=True, blockers=()),
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="managed_storage",
                server_trusted_enabled=True,
                auth_mode="multi_user",
            ),
            pull_token_signing_secret="personal-context-bootstrap-test-secret",
        ),
    )
    service.register_device(
        user_id="user-a",
        display_name="Chatbook A",
        client_type="chatbook",
        device_id="device-a",
        capabilities={"supported_adapter_versions": {domain: [1] for domain in PERSONAL_CONTEXT_SYNC_DOMAINS}},
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


def _complete_and_activate(
    service: SyncV2Service,
    canonical: _CanonicalService,
    bootstrap: PersonalContextBootstrap,
    *,
    device_id: str = "device-a",
) -> None:
    """Complete the real link before acknowledging this fake's v1 baseline.

    Snapshot/cursor tests retain their original transport boundaries. Real
    encrypted activation installation and receipts have separate integration tests.
    """
    service.complete_personal_context_link(
        user_id="user-a",
        device_id=device_id,
        dataset_id=bootstrap.dataset_id,
        bootstrap_cursor=bootstrap.cursor,
    )
    canonical.activated_devices.add((bootstrap.manifest.profile_id, bootstrap.dataset_id, device_id))
    dataset = service.store.get_dataset(bootstrap.dataset_id)
    metadata = dataset.metadata
    metadata["personal_context"].update(_EXCHANGE.model_dump(mode="json"))
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            "UPDATE sync_datasets SET metadata_json = ? WHERE dataset_id = ?",
            (json.dumps(metadata), bootstrap.dataset_id),
            connection=connection,
        )


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
        entity_version=str(payload["version_id"]),
        routing_metadata={
            "integrity_key_id": bootstrap.integrity_key.integrity_key_id,
            "profile_id": bootstrap.manifest.profile_id,
            "purge_generation": bootstrap.purge_generation,
        },
        encryption_metadata={"policy": "server_trusted_v1"},
    )


def _transport_record_envelope(
    service: SyncV2Service,
    bootstrap,
    *,
    revision: int,
    previous=None,
) -> SyncEnvelopeCreate:
    """Build one accepted encrypted revision through the production protector."""

    envelope = _client_transport_record_envelope(
        bootstrap,
        revision=revision,
        previous=previous,
    )
    dataset = service.store.get_dataset(bootstrap.dataset_id, owner_user_id="user-a")
    assert dataset is not None
    return service._protect_personal_context_for_storage(
        dataset,
        replace(envelope, apply_status="applied"),
    )


def _publish_transport_record(
    service: SyncV2Service, bootstrap: PersonalContextBootstrap, *, revision: int, previous: SyncEnvelope | None = None
) -> SyncEnvelope:
    """Publish a real acknowledged encrypted source without fabricating pull authority."""
    canonical = service._personal_context_service_for_user("user-a")
    database = canonical.database
    envelope = _client_transport_record_envelope(bootstrap, revision=revision, previous=previous)
    keys = ServerProfileKeyProvider(database).load(bootstrap.manifest.profile_id)
    with database.transaction(immediate=True) as connection:
        receipt = PersonalContextPublicationJournal(keys).append_batch(
            connection,
            profile_id=bootstrap.manifest.profile_id,
            purge_generation=0,
            objects=(
                PublicationObject(
                    domain=envelope.domain,
                    object_id=envelope.object_id,
                    version_id=str(envelope.entity_version),
                    operation="upsert",
                    role="semantic",
                    canonical=canonical_json_bytes(envelope.payload),
                ),
            ),
            now="2026-09-04T12:00:00Z",
        )
    result = PersonalContextRelay(
        publications=PersonalContextPublicationRelayStore(database),
        stage_authority=service.stage_personal_context_authority,
        finalize_authority=service.finalize_personal_context_authority,
        cancel_authority=service.cancel_personal_context_authority,
    ).relay_profile(
        user_id="user-a",
        profile_id=bootstrap.manifest.profile_id,
        dataset_id=bootstrap.dataset_id,
        after_server_cursor=None,
        wall_time_ms=5_000,
    )
    assert result.continuation == "complete"
    with database.transaction() as connection:
        row = connection.execute(
            "SELECT sync_server_cursor FROM personal_context_publication_rows WHERE profile_publication_sequence = ?",
            (receipt.profile_publication_sequence,),
        ).fetchone()
    stored = service.store.get_envelope_by_server_cursor(row[0])
    assert stored is not None
    return stored


def _client_transport_record_envelope(
    bootstrap,
    *,
    revision: int,
    previous=None,
) -> SyncEnvelopeCreate:
    """Build one clear client revision for the production push path."""

    payload = preference_record(
        profile_id=bootstrap.manifest.profile_id,
        record_id="transport-record",
        version_id=f"transport-record-v{revision}",
        parent_version_id=(None if revision == 1 else f"transport-record-v{revision - 1}"),
        value=f"value-{revision}",
    ).model_dump(mode="json")
    payload_bytes = canonical_json_bytes(payload)
    tag = hmac.new(_INTEGRITY_KEY, payload_bytes, hashlib.sha256)
    envelope = SyncEnvelopeCreate(
        dataset_id=bootstrap.dataset_id,
        client_envelope_id=f"device-a:transport-record:{revision}",
        device_id="device-a",
        domain="personal_context.record",
        operation="upsert",
        object_id=str(payload["record_id"]),
        parent_id=str(payload["scope_id"]),
        adapter_version=1,
        schema_version=1,
        payload=payload,
        payload_hash=f"hmac-sha256-v1:{tag.hexdigest()}",
        payload_size_bytes=len(payload_bytes),
        object_revision=revision,
        entity_version=str(payload["version_id"]),
        base_server_cursor=(None if previous is None else previous.server_sequence),
        base_object_revision=(None if previous is None else previous.object_revision),
        base_object_hash=(None if previous is None else previous.payload_hash),
        routing_metadata={
            "integrity_key_id": bootstrap.integrity_key.integrity_key_id,
            "profile_id": bootstrap.manifest.profile_id,
            "purge_generation": bootstrap.purge_generation,
        },
        encryption_metadata={"policy": "server_trusted_v1"},
    )
    return envelope


def _apply_record_to_fake_canonical(canonical: _CanonicalService, **values: object) -> object:
    """Apply one record like the canonical materializer target used in production."""

    value = values["value"]
    canonical.applied.append(values)
    if values["domain"] == "personal_context.record":
        canonical.records = (
            *(
                item
                for item in canonical.records
                if item.record_id != value.record_id
            ),
            value,
        )
    return value


def _register_transport_device(service: SyncV2Service, device_id: str) -> None:
    service.register_device(
        user_id="user-a",
        display_name=device_id,
        client_type="chatbook",
        device_id=device_id,
        capabilities={
            "supported_adapter_versions": {
                domain: [1] for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
            }
        },
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
    assert populated.sync_transport_cursor
    assert populated.sync_transport_cursor != populated.cursor
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
    assert retry.manifest == first.manifest
    assert retry.scopes == first.scopes
    assert canonical.created == 0
    assert canonical.profile_exists is False

    service.complete_personal_context_link(
        user_id="user-a",
        device_id="device-a",
        dataset_id=first.dataset_id,
        bootstrap_cursor=first.cursor,
    )

    assert canonical.profile_exists is True
    assert canonical.manifest == first.manifest
    assert canonical.scope == first.scopes[0]


def test_unknown_zero_minimum_quota_is_satisfied(tmp_path: Path) -> None:
    service, canonical = _service(tmp_path)

    bootstrap = _bootstrap(
        service,
        required_quotas={"future_sync_quota": 0},
    )

    assert bootstrap.manifest == canonical.manifest
    assert bootstrap.quotas["future_sync_quota"] == 0


def test_maximum_unknown_zero_minimum_quotas_return_a_bounded_complete_map(
    tmp_path: Path,
) -> None:
    service, canonical = _service(tmp_path)
    required_quotas = {f"future_quota_{index:02d}": 0 for index in range(32)}

    bootstrap = _bootstrap(service, required_quotas=required_quotas)

    assert bootstrap.manifest == canonical.manifest
    assert bootstrap.quotas == required_quotas


def test_maximum_quota_incompatibility_reports_only_every_requested_quota(
    tmp_path: Path,
) -> None:
    service, _canonical = _service(tmp_path)
    required_quotas = {f"future_quota_{index:02d}": 0 for index in range(32)}
    required_quotas["future_quota_31"] = 1

    with pytest.raises(PersonalContextBootstrapError) as exc_info:
        _bootstrap(service, required_quotas=required_quotas)

    assert exc_info.value.reason_code == "personal_context_quota_incompatible"
    assert exc_info.value.attention == {
        "kind": "quota_incompatible",
        "required_quotas": required_quotas,
        "available_quotas": dict.fromkeys(required_quotas, 0),
        "insufficient_quotas": ["future_quota_31"],
    }


def test_bootstrap_sync_transport_cursor_is_accepted_by_private_pull_parser(
    tmp_path: Path,
) -> None:
    service, _canonical = _service(tmp_path)

    bootstrap = _bootstrap(service)
    _complete_and_activate(service, _canonical, bootstrap)
    pulled = service.pull(
        user_id="user-a",
        dataset_id=bootstrap.dataset_id,
        device_id="device-a",
        cursor=bootstrap.sync_transport_cursor,
        personal_context_exchange=_EXCHANGE,
        domains=PERSONAL_CONTEXT_SYNC_DOMAINS,
    )

    assert pulled.envelopes == []
    assert pulled.next_cursor
    with pytest.raises(SyncStoreError):
        service.pull(
            user_id="user-a",
            dataset_id=bootstrap.dataset_id,
            device_id="device-a",
            cursor=bootstrap.cursor,
            personal_context_exchange=_EXCHANGE,
            domains=PERSONAL_CONTEXT_SYNC_DOMAINS,
        )


def test_bootstrap_transport_watermark_skips_retained_history_and_delivers_later_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _canonical = _service(tmp_path, publication_journal=True, monkeypatch=monkeypatch)
    first_device = _bootstrap(service)
    prior = _publish_transport_record(service, first_device, revision=1)
    prior = _publish_transport_record(service, first_device, revision=2, previous=prior)
    _register_transport_device(service, "device-b")

    reviewed = _bootstrap(service, device_id="device-b")
    _complete_and_activate(service, _canonical, reviewed, device_id="device-b")
    later = _publish_transport_record(service, first_device, revision=3, previous=prior)
    pulled = service.pull(
        user_id="user-a",
        dataset_id=reviewed.dataset_id,
        device_id="device-b",
        cursor=reviewed.sync_transport_cursor,
        personal_context_exchange=_EXCHANGE,
        domains=PERSONAL_CONTEXT_SYNC_DOMAINS,
    )

    assert [item.server_sequence for item in pulled.envelopes] == [
        later.server_sequence
    ]
    assert pulled.envelopes[0].entity_version == "transport-record-v3"
    with pytest.raises(SyncStoreError, match="sync_pull_token_invalid"):
        service.pull(
            user_id="user-a",
            dataset_id=reviewed.dataset_id,
            device_id="device-b",
            cursor=reviewed.sync_transport_cursor,
            personal_context_exchange=_EXCHANGE,
            domains=["personal_context.record"],
        )


def test_bootstrap_transport_cursor_retry_preserves_boundary_and_supports_slow_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _canonical = _service(tmp_path, publication_journal=True, monkeypatch=monkeypatch)
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    service.clock = lambda: now.isoformat()
    first = _bootstrap(service)
    _complete_and_activate(service, _canonical, first)
    device = service.store.get_device("user-a", "device-a")
    assert device is not None
    streams = service._pull_adapter_streams(device, PERSONAL_CONTEXT_SYNC_DOMAINS)
    version_set = service._pull_version_set(device)
    first_watermarks = service._decode_pull_token(
        first.sync_transport_cursor,
        dataset_id=first.dataset_id,
        device_id="device-a",
        version_set=version_set,
        streams=streams,
    )
    later = _publish_transport_record(service, first, revision=1)
    service.personal_context_key_fingerprint = lambda *, device: (
        f"rotated-fingerprint:{device.device_id}"
    )

    now += timedelta(days=29)
    assert service.pull(
        user_id="user-a",
        dataset_id=first.dataset_id,
        device_id="device-a",
        cursor=first.sync_transport_cursor,
        personal_context_exchange=_EXCHANGE,
        domains=PERSONAL_CONTEXT_SYNC_DOMAINS,
        include_own_changes=True,
    ).envelopes[0].server_sequence == later.server_sequence
    retry = _bootstrap(service)
    _complete_and_activate(service, _canonical, retry)
    retry_watermarks = service._decode_pull_token(
        retry.sync_transport_cursor,
        dataset_id=retry.dataset_id,
        device_id="device-a",
        version_set=version_set,
        streams=streams,
    )
    assert retry.cursor == first.cursor
    assert retry_watermarks == first_watermarks

    now += timedelta(days=1, seconds=301)
    with pytest.raises(SyncStoreError, match="sync_pull_token_invalid"):
        service.pull(
            user_id="user-a",
            dataset_id=first.dataset_id,
            device_id="device-a",
            cursor=first.sync_transport_cursor,
            personal_context_exchange=_EXCHANGE,
            domains=PERSONAL_CONTEXT_SYNC_DOMAINS,
            include_own_changes=True,
        )
    assert service.pull(
        user_id="user-a",
        dataset_id=retry.dataset_id,
        device_id="device-a",
        cursor=retry.sync_transport_cursor,
        personal_context_exchange=_EXCHANGE,
        domains=PERSONAL_CONTEXT_SYNC_DOMAINS,
        include_own_changes=True,
    ).envelopes[0].server_sequence == later.server_sequence


def test_bootstrap_transport_snapshot_serializes_concurrent_sqlite_insert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, canonical = _service(tmp_path, publication_journal=True, monkeypatch=monkeypatch)
    source = _bootstrap(service)
    _register_transport_device(service, "device-b")
    entered_snapshot = threading.Event()
    release_snapshot = threading.Event()
    original_plan = canonical.plan_sync_bootstrap

    def blocked_plan():
        entered_snapshot.set()
        assert release_snapshot.wait(timeout=5)
        return original_plan()

    canonical.plan_sync_bootstrap = blocked_plan  # type: ignore[method-assign]
    with ThreadPoolExecutor(max_workers=2) as executor:
        bootstrap_future = executor.submit(_bootstrap, service, device_id="device-b")
        assert entered_snapshot.wait(timeout=5)
        insert_future = executor.submit(
            _publish_transport_record, service, source, revision=1,
        )
        with pytest.raises(FutureTimeoutError):
            insert_future.result(timeout=0.2)
        release_snapshot.set()
        reviewed = bootstrap_future.result(timeout=5)
        inserted = insert_future.result(timeout=5)

    _complete_and_activate(service, canonical, reviewed, device_id="device-b")
    pulled = service.pull(
        user_id="user-a",
        dataset_id=reviewed.dataset_id,
        device_id="device-b",
        cursor=reviewed.sync_transport_cursor,
        personal_context_exchange=_EXCHANGE,
        domains=PERSONAL_CONTEXT_SYNC_DOMAINS,
    )
    assert [item.server_sequence for item in pulled.envelopes] == [
        inserted.server_sequence
    ]


def test_bootstrap_rejects_real_push_paused_before_canonical_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signed boundary cannot cover an accepted-but-unmaterialized push."""

    service, canonical = _service(tmp_path)
    source = _bootstrap(service)
    service.complete_personal_context_link(
        user_id="user-a",
        device_id="device-a",
        dataset_id=source.dataset_id,
        bootstrap_cursor=source.cursor,
    )
    _complete_and_activate(service, canonical, source)
    _register_transport_device(service, "device-b")
    canonical.apply_sync_object = lambda **values: _apply_record_to_fake_canonical(
        canonical, **values
    )  # type: ignore[method-assign]
    entered_materialization = threading.Event()
    release_materialization = threading.Event()
    original_materialize = service._materialize_envelope

    def blocked_materialize(envelope, **kwargs):
        entered_materialization.set()
        assert release_materialization.wait(timeout=5)
        return original_materialize(envelope, **kwargs)

    monkeypatch.setattr(service, "_materialize_envelope", blocked_materialize)
    with ThreadPoolExecutor(max_workers=2) as executor:
        push_future = executor.submit(
            service.push,
            user_id="user-a",
            dataset_id=source.dataset_id,
            device_id="device-a",
            envelopes=[_client_transport_record_envelope(source, revision=1)],
            personal_context_exchange=_EXCHANGE,
        )
        assert entered_materialization.wait(timeout=5)
        try:
            with pytest.raises(Exception) as exc_info:
                _bootstrap(service, device_id="device-b")
        finally:
            release_materialization.set()
        pushed = push_future.result(timeout=5)

    assert getattr(exc_info.value, "reason_code", None) == (
        "personal_context_projection_incomplete"
    )
    assert "transport-record" not in str(exc_info.value)
    assert _PLAINTEXT_CANARY not in str(exc_info.value)
    assert len(pushed.accepted) == 1

    reviewed = _bootstrap(service, device_id="device-b")
    _complete_and_activate(service, canonical, reviewed, device_id="device-b")
    pulled = service.pull(
        user_id="user-a",
        dataset_id=reviewed.dataset_id,
        device_id="device-b",
        cursor=reviewed.sync_transport_cursor,
        personal_context_exchange=_EXCHANGE,
        domains=PERSONAL_CONTEXT_SYNC_DOMAINS,
    )
    assert pulled.envelopes == []
    assert any(item.record_id == "transport-record" for item in reviewed.records)


def test_failed_materialization_blocks_bootstrap_until_guarded_replay_succeeds(
    tmp_path: Path,
) -> None:
    """Repair must finish canonically before bootstrap can watermark its sequence."""

    service, canonical = _service(tmp_path)
    source = _bootstrap(service)
    service.complete_personal_context_link(
        user_id="user-a",
        device_id="device-a",
        dataset_id=source.dataset_id,
        bootstrap_cursor=source.cursor,
    )
    _complete_and_activate(service, canonical, source)
    _register_transport_device(service, "device-b")

    def fail_apply(**_values: object) -> object:
        raise RuntimeError("projection-private-canary")

    canonical.apply_sync_object = fail_apply  # type: ignore[method-assign]
    pushed = service.push(
        user_id="user-a",
        dataset_id=source.dataset_id,
        device_id="device-a",
        envelopes=[_client_transport_record_envelope(source, revision=1)],
        personal_context_exchange=_EXCHANGE,
    )
    assert len(pushed.accepted) == 1
    sequence = pushed.accepted[0].server_sequence
    stored = service.store.get_envelope_by_server_cursor(sequence)
    assert stored is not None and stored.apply_status == "failed"

    with pytest.raises(Exception) as exc_info:
        _bootstrap(service, device_id="device-b")
    assert getattr(exc_info.value, "reason_code", None) == (
        "personal_context_projection_incomplete"
    )
    assert "projection-private-canary" not in str(exc_info.value)

    canonical.apply_sync_object = lambda **values: _apply_record_to_fake_canonical(
        canonical, **values
    )  # type: ignore[method-assign]
    replayed = service.repair(
        user_id="user-a",
        dataset_id=source.dataset_id,
        domains=["personal_context.record"],
        failed_only=True,
    )
    assert replayed.applied_count == 1, replayed.domain_results
    repaired = service.store.get_envelope_by_server_cursor(sequence)
    assert repaired is not None and repaired.apply_status == "applied"

    reviewed = _bootstrap(service, device_id="device-b")
    service.complete_personal_context_link(
        user_id="user-a",
        device_id="device-b",
        dataset_id=reviewed.dataset_id,
        bootstrap_cursor=reviewed.cursor,
    )
    _complete_and_activate(service, canonical, reviewed, device_id="device-b")
    assert service.pull(
        user_id="user-a",
        dataset_id=reviewed.dataset_id,
        device_id="device-b",
        cursor=reviewed.sync_transport_cursor,
        personal_context_exchange=_EXCHANGE,
        domains=PERSONAL_CONTEXT_SYNC_DOMAINS,
    ).envelopes == []


@pytest.mark.parametrize("apply_status", ["pending", "conflict", "unknown"])
def test_bootstrap_rejects_nonterminal_or_unknown_personal_context_apply_status(
    tmp_path: Path,
    apply_status: str,
) -> None:
    service, _canonical = _service(tmp_path)
    source = _bootstrap(service)
    _register_transport_device(service, "device-b")
    inserted = service.store.insert_envelope(
        _transport_record_envelope(service, source, revision=1)
    )
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            "UPDATE sync_envelopes SET apply_status = ? WHERE server_sequence = ?",
            (apply_status, inserted.server_sequence),
            connection=connection,
        )

    with pytest.raises(Exception) as exc_info:
        _bootstrap(service, device_id="device-b")

    assert getattr(exc_info.value, "reason_code", None) == (
        "personal_context_projection_incomplete"
    )
    assert "transport-record" not in str(exc_info.value)


def test_personal_context_service_has_no_materializing_sync_profile_helper() -> None:
    assert "ensure_sync_profile" not in PersonalContextService.__dict__


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
        "bootstrap_cursor",
        "transport_watermarks",
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
    elif mismatch == "wrapping_key_fingerprint":
        values["kdf_metadata"] = {
            **winner.kdf_metadata,
            "wrapping_key_fingerprint": "foreign-fingerprint",
        }
    elif mismatch == "bootstrap_cursor":
        values["kdf_metadata"] = {
            **winner.kdf_metadata,
            "bootstrap_cursor": "foreign-bootstrap-cursor",
        }
    else:
        values["kdf_metadata"] = {
            **winner.kdf_metadata,
            "transport_watermarks": [["personal_context.record", 1, -1]],
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
            bootstrap_cursor=bootstrap.cursor,
            transport_watermarks={
                (domain, 1): 0 for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
            },
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
        (
            {"required_schema_version": 2},
            "personal_context_schema_incompatible",
            {
                "kind": "schema_incompatible",
                "required_schema_version": 2,
                "server_min_schema_version": 1,
                "server_max_schema_version": 1,
            },
        ),
        (
            {"required_quotas": {"max_record_bytes": 16_385}},
            "personal_context_quota_incompatible",
            {
                "kind": "quota_incompatible",
                "required_quotas": {"max_record_bytes": 16_385},
                "available_quotas": {"max_record_bytes": 16_384},
                "insufficient_quotas": ["max_record_bytes"],
            },
        ),
        (
            {"expected_purge_generation": 1},
            "personal_context_purge_generation_stale",
            {
                "kind": "purge_generation_mismatch",
                "expected_purge_generation": 1,
                "current_purge_generation": 0,
            },
        ),
    ]
    for kwargs, reason, attention in cases:
        with pytest.raises(Exception) as exc_info:
            _bootstrap(service, **kwargs)
        assert getattr(exc_info.value, "reason_code", None) == reason
        assert getattr(exc_info.value, "attention", None) == attention
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

    with pytest.raises(SyncStoreError, match="personal_context_activation_required"):
        service.push(
            user_id="user-a",
            dataset_id=bootstrap.dataset_id,
            device_id="device-a",
            envelopes=[envelope],
        )
    assert _canonical.applied == []

    service.complete_personal_context_link(
        user_id="user-a",
        device_id="device-a",
        dataset_id=bootstrap.dataset_id,
        bootstrap_cursor=bootstrap.cursor,
    )
    _complete_and_activate(service, _canonical, bootstrap)
    after = service.push(
        user_id="user-a",
        dataset_id=bootstrap.dataset_id,
        device_id="device-a",
        envelopes=[envelope],
        personal_context_exchange=_EXCHANGE,
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
    applied_before = list(_canonical.applied)
    with pytest.raises(SyncStoreError, match="personal_context_activation_required"):
        service.push(
            user_id="user-a",
            dataset_id=bootstrap.dataset_id,
            device_id="device-b",
            envelopes=[other_device],
            personal_context_exchange=_EXCHANGE,
        )
    assert _canonical.applied == applied_before


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
