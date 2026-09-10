"""Exercise durable activation installation and exact device acknowledgments."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Personalization.personal_context_service import PersonalContextService
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.models import PERSONAL_CONTEXT_SYNC_DOMAINS, SyncDatasetCreate
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store
from tldw_Server_API.tests.Sync.test_sync_v2_personal_context_certification import (
    _DEVICE_ID,
    _USER_ID,
    _register_device,
)
from tldw_Server_API.tests.Sync.test_sync_v2_personal_context_certification import (
    production_factories as production_factories,
)
from tldw_Server_API.tests.Sync.test_sync_v2_personal_context_exchange_gate import _client

pytestmark = pytest.mark.unit


@pytest.fixture(params=["sqlite", "postgres"])
def activation_store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[SyncV2Store]:
    """Use the shared isolated PostgreSQL fixture or a real SQLite database."""

    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        database = SyncDatabase(backend=backend)
    else:
        database = SyncDatabase(sqlite_path=tmp_path / "activation-sync.db")
    store = SyncV2Store(database)
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="activation-dataset",
            owner_user_id="activation-user",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=sorted(PERSONAL_CONTEXT_SYNC_DOMAINS),
        )
    )
    try:
        yield store
    finally:
        if backend is not None:
            backend.get_pool().close_all()


def _installation() -> dict[str, object]:
    """Return content-free install identity plus already-encrypted fixture bytes."""

    return {
        "activation_id": "activation-0123456789",
        "dataset_id": "activation-dataset",
        "user_id": "activation-user",
        "profile_id": "profile-0123456789",
        "device_id": "activation-device",
        "baseline_digest": "a" * 64,
        "purge_generation": 0,
        "publication_watermark": 3,
        "home_server_cursor": 7,
        "receipt_id": "install-receipt-0123456789",
        "envelopes_json": '[{"ciphertext":"opaque"}]',
        "expires_at": "2099-01-01T00:00:00+00:00",
    }


def test_installation_replays_exactly_after_reopening(activation_store: SyncV2Store) -> None:
    """Restart must retain the first immutable installation, not replace its bytes."""

    with activation_store.personal_context_authority_guard("activation-dataset", "profile-0123456789") as guarded:
        first = guarded.install_personal_context_activation(**_installation())
    reopened = SyncV2Store(activation_store.db)
    with reopened.personal_context_authority_guard("activation-dataset", "profile-0123456789") as guarded:
        replay = guarded.install_personal_context_activation(**_installation())
    assert replay == first
    assert replay["home_server_cursor"] == 7


@pytest.mark.parametrize(
    "field,value",
    [
        ("baseline_digest", "b" * 64),
        ("home_server_cursor", 8),
        ("device_id", "different-device"),
        ("envelopes_json", "[]"),
    ],
)
def test_changed_installation_replay_is_rejected(activation_store: SyncV2Store, field: str, value: object) -> None:
    """An activation ID never names a different baseline or checkpoint."""

    with activation_store.personal_context_authority_guard("activation-dataset", "profile-0123456789") as guarded:
        guarded.install_personal_context_activation(**_installation())
    changed = {**_installation(), field: value}
    with pytest.raises(SyncStoreError, match="activation_receipt_mismatch"):
        with activation_store.personal_context_authority_guard("activation-dataset", "profile-0123456789") as guarded:
            guarded.install_personal_context_activation(**changed)


def test_device_acknowledgment_is_exact_and_durable(activation_store: SyncV2Store) -> None:
    """The same acknowledgment replays; changing its local receipt fails closed."""

    with activation_store.personal_context_authority_guard("activation-dataset", "profile-0123456789") as guarded:
        guarded.install_personal_context_activation(**_installation())
        first = guarded.acknowledge_personal_context_activation(
            activation_id="activation-0123456789",
            dataset_id="activation-dataset",
            user_id="activation-user",
            device_id="activation-device",
            baseline_digest="a" * 64,
            local_receipt_id="local-receipt-0123456789",
        )
    with activation_store.personal_context_authority_guard("activation-dataset", "profile-0123456789") as guarded:
        replay = guarded.acknowledge_personal_context_activation(
            activation_id="activation-0123456789",
            dataset_id="activation-dataset",
            user_id="activation-user",
            device_id="activation-device",
            baseline_digest="a" * 64,
            local_receipt_id="local-receipt-0123456789",
        )
        assert replay == first
        with pytest.raises(SyncStoreError, match="activation_receipt_mismatch"):
            guarded.acknowledge_personal_context_activation(
                activation_id="activation-0123456789",
                dataset_id="activation-dataset",
                user_id="activation-user",
                device_id="activation-device",
                baseline_digest="a" * 64,
                local_receipt_id="changed-receipt-0123456789",
            )


def _set_exchange_version(service: SyncV2Service, dataset_id: str, version: int) -> None:
    """Explicitly simulate rollout without creating a forged continuity proof."""

    dataset = service.store.get_dataset(dataset_id)
    metadata = dataset.metadata
    metadata["personal_context"]["ongoing_sync_version"] = version
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            "UPDATE sync_datasets SET metadata_json = ? WHERE dataset_id = ?",
            (json.dumps(metadata), dataset_id),
            connection=connection,
        )


def test_production_activation_requires_ack_and_replays(production_factories) -> None:
    """Real factories install one baseline, then require an exact local acknowledgment."""

    canonical, service = production_factories
    canonical.create_profile(runtime_enabled=False)
    _register_device(service)
    initial = service.bootstrap_personal_context(user_id=_USER_ID, device_id=_DEVICE_ID)
    service.complete_personal_context_link(
        user_id=_USER_ID,
        device_id=_DEVICE_ID,
        dataset_id=initial.dataset_id,
        bootstrap_cursor=initial.cursor,
    )
    prepared = service.prepare_personal_context_activation(user_id=_USER_ID, device_id=_DEVICE_ID)
    assert prepared.activation.state == "installed"
    _set_exchange_version(service, initial.dataset_id, 1)
    with pytest.raises(SyncStoreError, match="activation_required"):
        service.verified_active_exchange(
            user_id=_USER_ID,
            dataset_id=initial.dataset_id,
            device_id=_DEVICE_ID,
            exchange=prepared.personal_context_exchange,
        )
    receipt, proof = service.acknowledge_personal_context_activation(
        user_id=_USER_ID,
        dataset_id=initial.dataset_id,
        device_id=_DEVICE_ID,
        activation_id=prepared.activation.activation_id,
        baseline_digest=prepared.activation.baseline_digest,
        local_receipt_id="local-installation-0123456789",
        exchange=prepared.personal_context_exchange,
    )
    assert receipt.state == "active"
    assert (
        service.verified_active_exchange(
            user_id=_USER_ID,
            dataset_id=initial.dataset_id,
            device_id=_DEVICE_ID,
            exchange=proof,
        )
        == proof
    )
    _set_exchange_version(service, initial.dataset_id, 0)
    with pytest.raises(SyncStoreError, match="activation_required"):
        service.verified_active_exchange(
            user_id=_USER_ID,
            dataset_id=initial.dataset_id,
            device_id=_DEVICE_ID,
            exchange=proof,
        )
    _set_exchange_version(service, initial.dataset_id, 1)
    assert (
        service.verified_active_exchange(
            user_id=_USER_ID,
            dataset_id=initial.dataset_id,
            device_id=_DEVICE_ID,
            exchange=proof,
        )
        == proof
    )
    replay = service.prepare_personal_context_activation(user_id=_USER_ID, device_id=_DEVICE_ID)
    assert replay.activation.activation_id == prepared.activation.activation_id
    assert replay.manifest == prepared.manifest
    assert service.capabilities().personal_context.ongoing_sync_version == 0


def test_ongoing_bootstrap_stays_closed_before_rollout(production_factories) -> None:
    """An explicit version-one request cannot bypass version-zero readiness."""

    _canonical, service = production_factories
    with _client(service) as client:
        response = client.post(
            "/api/v1/sync/personal-context/bootstrap",
            json={
                "device_id": _DEVICE_ID,
                "ongoing_sync_version": 1,
            },
        )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "personal_context_ongoing_sync_unavailable"


@pytest.fixture
def linked_activation(
    production_factories: tuple[PersonalContextService, SyncV2Service],
) -> tuple[PersonalContextService, SyncV2Service]:
    """Create a reviewed first link through actual canonical and Sync services."""

    canonical, service = production_factories
    canonical.create_profile(runtime_enabled=False)
    _register_device(service)
    initial = service.bootstrap_personal_context(user_id=_USER_ID, device_id=_DEVICE_ID)
    service.complete_personal_context_link(
        user_id=_USER_ID, device_id=_DEVICE_ID, dataset_id=initial.dataset_id, bootstrap_cursor=initial.cursor
    )
    return canonical, service


def test_sync_install_survives_failure_before_canonical_coverage(
    linked_activation: tuple[PersonalContextService, SyncV2Service], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed coverage CAS reuses the already committed exact Sync baseline."""

    canonical, service = linked_activation
    repository_type = type(canonical._repository)
    original = repository_type.complete_activation_install

    def interrupted(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected coverage interruption")

    monkeypatch.setattr(repository_type, "complete_activation_install", interrupted)
    with pytest.raises(SyncStoreError, match="activation_required"):
        service.prepare_personal_context_activation(user_id=_USER_ID, device_id=_DEVICE_ID)
    rows = service.store.db.execute("SELECT * FROM sync_personal_context_activations").rows
    assert len(rows) == 1
    first = dict(rows[0])
    monkeypatch.setattr(repository_type, "complete_activation_install", original)
    result = service.prepare_personal_context_activation(user_id=_USER_ID, device_id=_DEVICE_ID)
    assert result.activation.activation_id == first["activation_id"]
    assert service.store.get_personal_context_activation(first["activation_id"]) == first


def test_tampered_sync_baseline_cannot_gain_canonical_coverage(
    linked_activation: tuple[PersonalContextService, SyncV2Service], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ciphertext identity is reverified even when Sync claims installation succeeded."""

    canonical, service = linked_activation
    repository_type = type(canonical._repository)
    original = repository_type.complete_activation_install

    def interrupted(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected coverage interruption")

    monkeypatch.setattr(repository_type, "complete_activation_install", interrupted)
    with pytest.raises(SyncStoreError):
        service.prepare_personal_context_activation(user_id=_USER_ID, device_id=_DEVICE_ID)
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            "UPDATE sync_personal_context_activations SET envelopes_json = ?", ("[]",), connection=connection
        )
    monkeypatch.setattr(repository_type, "complete_activation_install", original)
    with pytest.raises(SyncStoreError, match="activation_receipt_mismatch"):
        service.prepare_personal_context_activation(user_id=_USER_ID, device_id=_DEVICE_ID)


def test_expired_sync_installation_requires_fresh_baseline(
    linked_activation: tuple[PersonalContextService, SyncV2Service], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An expired unacknowledged delivery receipt cannot silently reactivate a device."""

    _canonical, service = linked_activation
    result = service.prepare_personal_context_activation(user_id=_USER_ID, device_id=_DEVICE_ID)
    from tldw_Server_API.app.core.Sync.v2 import personal_context_activation as activation_module

    future = datetime.now(timezone.utc) + timedelta(days=31)
    monkeypatch.setattr(activation_module, "_activation_now", lambda: future)
    with pytest.raises(SyncStoreError):
        service.acknowledge_personal_context_activation(
            user_id=_USER_ID,
            device_id=_DEVICE_ID,
            dataset_id=result.dataset_id,
            activation_id=result.activation.activation_id,
            baseline_digest=result.activation.baseline_digest,
            local_receipt_id="local-receipt-expired-12345",
            exchange=result.personal_context_exchange,
        )
    renewed = service.prepare_personal_context_activation(user_id=_USER_ID, device_id=_DEVICE_ID)
    assert renewed.activation.activation_id != result.activation.activation_id
    assert renewed.personal_context_exchange == result.personal_context_exchange


def test_sync_ack_survives_failure_before_canonical_ack(linked_activation, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry after an independently committed Sync ack must use the same local receipt."""

    canonical, service = linked_activation
    result = service.prepare_personal_context_activation(user_id=_USER_ID, device_id=_DEVICE_ID)
    repository_type = type(canonical._repository)
    original = repository_type.confirm_activation_device

    def interrupted(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected acknowledgment interruption")

    monkeypatch.setattr(repository_type, "confirm_activation_device", interrupted)
    arguments = {
        "user_id": _USER_ID,
        "device_id": _DEVICE_ID,
        "dataset_id": result.dataset_id,
        "activation_id": result.activation.activation_id,
        "baseline_digest": result.activation.baseline_digest,
        "local_receipt_id": "durable-local-ack-0123456789",
        "exchange": result.personal_context_exchange,
    }
    with pytest.raises(SyncStoreError, match="activation_required"):
        service.acknowledge_personal_context_activation(**arguments)
    stored = service.store.get_personal_context_activation_ack(result.activation.activation_id, _DEVICE_ID)
    assert stored is not None
    monkeypatch.setattr(repository_type, "confirm_activation_device", original)
    receipt, _proof = service.acknowledge_personal_context_activation(**arguments)
    assert receipt.state == "active"
    assert service.store.get_personal_context_activation_ack(result.activation.activation_id, _DEVICE_ID) == stored
    with pytest.raises(SyncStoreError, match="activation_receipt_mismatch"):
        service.acknowledge_personal_context_activation(
            **{**arguments, "local_receipt_id": "different-local-ack-0123456789"}
        )


def test_direct_purge_removes_sync_activation_history(linked_activation) -> None:
    """Existing authorized purge also removes new encrypted baseline and ack storage."""

    canonical, service = linked_activation
    result = service.prepare_personal_context_activation(user_id=_USER_ID, device_id=_DEVICE_ID)
    service.acknowledge_personal_context_activation(
        user_id=_USER_ID,
        device_id=_DEVICE_ID,
        dataset_id=result.dataset_id,
        activation_id=result.activation.activation_id,
        baseline_digest=result.activation.baseline_digest,
        local_receipt_id="purge-local-ack-0123456789",
        exchange=result.personal_context_exchange,
    )
    canonical.set_after_commit_purge_cleanup(
        lambda intent: canonical.cleanup_sync_history(intent, user_id=_USER_ID, sync=service)
    )
    canonical.purge_profile(mode="everywhere", confirmation="DELETE EVERYWHERE", expected_purge_generation=0)
    assert service.store.get_personal_context_activation(result.activation.activation_id) is None
    assert service.store.get_personal_context_activation_ack(result.activation.activation_id, _DEVICE_ID) is None


def test_authenticated_activation_endpoints_use_real_journals(
    linked_activation, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ready-route branch echoes persisted receipts and rejects another owner."""

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user

    _canonical, service = linked_activation
    capabilities = service.capabilities

    def ready(**kwargs):
        current = capabilities(**kwargs)
        return replace(current, personal_context=replace(current.personal_context, ongoing_sync_version=1))

    monkeypatch.setattr(service, "capabilities", ready)
    with _client(service) as client:
        client.app.dependency_overrides[get_request_user] = lambda: User(id=_USER_ID, username="activation-owner")
        response = client.post(
            "/api/v1/sync/personal-context/bootstrap", json={"device_id": _DEVICE_ID, "ongoing_sync_version": 1}
        )
        assert response.status_code == 200
        prepared = response.json()
        arguments = {
            "dataset_id": prepared["dataset_id"],
            "device_id": _DEVICE_ID,
            "activation_id": prepared["activation"]["activation_id"],
            "baseline_digest": prepared["activation"]["baseline_digest"],
            "local_receipt_id": "http-local-receipt-0123456789",
            "personal_context_exchange": prepared["personal_context_exchange"],
        }
        response = client.post("/api/v1/sync/personal-context/activation/acknowledge", json=arguments)
        assert response.status_code == 200
        assert response.json()["receipt"]["state"] == "active"
        assert response.json()["personal_context_exchange"] == prepared["personal_context_exchange"]
        client.app.dependency_overrides[get_request_user] = lambda: User(id="other-owner", username="other-owner")
        denied = client.post("/api/v1/sync/personal-context/activation/acknowledge", json=arguments)
        assert denied.status_code in {403, 404}


def test_protected_sync_baseline_has_no_plaintext_and_survives_rotation(linked_activation) -> None:
    """Real Sync files retain ciphertext only and ordinary key rotation preserves replay."""

    from tldw_Server_API.tests.Personalization.personal_context_test_support import preference_record

    canonical, service = linked_activation
    canary = "activation-private-canary-c5c38e9b"
    record = preference_record(value=canary)
    canonical.create_manual_record(
        scope_id=canonical.list_scopes()[0].scope_id,
        payload=record.payload,
        semantic_key=None,
        controls=record.controls,
    )
    result = service.prepare_personal_context_activation(user_id=_USER_ID, device_id=_DEVICE_ID)
    assert any(item.payload.value == canary for item in result.records)
    stored = service.store.get_personal_context_activation(result.activation.activation_id)
    assert canary not in json.dumps(stored)
    path = Path(service.store.db.backend.config.sqlite_path)
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists():
            assert canary.encode() not in candidate.read_bytes()
    canonical._repository.rotate_encryption_key(canonical.get_manifest().profile_id)
    replay = service.prepare_personal_context_activation(user_id=_USER_ID, device_id=_DEVICE_ID)
    assert replay.activation.activation_id == result.activation.activation_id
    assert replay.records == result.records


@pytest.mark.parametrize("expected_generation", [None, 0])
def test_purge_between_link_authorization_and_preparation_rejects_install(
    linked_activation, monkeypatch: pytest.MonkeyPatch, expected_generation: int | None
) -> None:
    """A receipt for generation zero cannot install a generation-one baseline."""
    from tldw_Server_API.app.core.Personalization.personal_context_activation import PersonalContextActivationService

    canonical, service = linked_activation
    prepare = PersonalContextActivationService.prepare

    def purge_then_prepare(self, *args, **kwargs):
        canonical.purge_profile(mode="everywhere", confirmation="DELETE EVERYWHERE", expected_purge_generation=0)
        return prepare(self, *args, **kwargs)

    monkeypatch.setattr(PersonalContextActivationService, "prepare", purge_then_prepare)
    code = "purge_generation_stale" if expected_generation is not None else "activation_required"
    with pytest.raises(SyncStoreError, match=code):
        service.prepare_personal_context_activation(
            user_id=_USER_ID, device_id=_DEVICE_ID, expected_purge_generation=expected_generation
        )
    assert not service.store.db.execute("SELECT * FROM sync_personal_context_activations").rows
    dataset = service.store.personal_context_dataset_for_profile(
        user_id=_USER_ID, profile_id=canonical.get_manifest().profile_id
    )
    assert "activation_epoch" not in dataset.metadata["personal_context"]


def test_activation_waiting_for_relay_lease_does_not_hold_sync_transaction(
    linked_activation, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relay owning the profile lease can finish its Sync stage during install."""
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from threading import Event, current_thread

    from tldw_Server_API.app.core.Personalization.personal_context_activation import PersonalContextActivationService
    from tldw_Server_API.app.core.Personalization.personal_context_publication import (
        PersonalContextPublicationRelayStore,
    )

    canonical, service = linked_activation
    profile_id = canonical.get_manifest().profile_id
    dataset = service.store.personal_context_dataset_for_profile(user_id=_USER_ID, profile_id=profile_id)
    prepared, relay_owned, installer_waiting, stage_done = Event(), Event(), Event(), Event()
    prepare = PersonalContextActivationService.prepare
    lease_method = PersonalContextPublicationRelayStore.profile_lease

    def prepare_then_pause(self, *args, **kwargs):
        result = prepare(self, *args, **kwargs)
        prepared.set()
        assert relay_owned.wait(5)
        return result

    @contextmanager
    def observe_lease(self, requested_profile, **kwargs):
        if prepared.is_set() and current_thread().name != "relay-stage":
            installer_waiting.set()
        with lease_method(self, requested_profile, **kwargs) as lease:
            yield lease

    def relay_stage():
        current_thread().name = "relay-stage"
        assert prepared.wait(5)
        with lease_method(PersonalContextPublicationRelayStore(canonical._repository.database), profile_id):
            relay_owned.set()
            assert installer_waiting.wait(5)
            with service.store.personal_context_authority_guard(dataset.dataset_id, profile_id):
                stage_done.set()

    monkeypatch.setattr(PersonalContextActivationService, "prepare", prepare_then_pause)
    monkeypatch.setattr(PersonalContextPublicationRelayStore, "profile_lease", observe_lease)
    with ThreadPoolExecutor(max_workers=2) as executor:
        relay = executor.submit(relay_stage)
        activation = executor.submit(
            service.prepare_personal_context_activation, user_id=_USER_ID, device_id=_DEVICE_ID
        )
        relay.result(timeout=15)
        result = activation.result(timeout=15)
    assert stage_done.is_set()
    assert result.activation.state == "installed"


def test_push_after_commit_relay_does_not_wait_for_installers_profile_lease(
    linked_activation, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real push commits ingress while a competing installer waits for its Sync guard."""
    import hashlib
    import hmac
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from threading import Event, current_thread

    from tldw_profile_core.canonical import canonical_json_bytes

    from tldw_Server_API.app.core.Personalization.personal_context_publication import (
        PersonalContextPublicationRelayStore,
    )
    from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelopeCreate
    from tldw_Server_API.app.core.Sync.v2.personal_context_ongoing_contract import PersonalContextExchangeProof
    from tldw_Server_API.tests.Personalization.personal_context_test_support import preference_record
    from tldw_Server_API.tests.Sync.test_sync_v2_personal_context_certification import _seed_exchange

    canonical, service = linked_activation
    manifest = canonical.get_manifest()
    dataset = service.store.personal_context_dataset_for_profile(user_id=_USER_ID, profile_id=manifest.profile_id)
    exchange = PersonalContextExchangeProof.model_validate(_seed_exchange(service, dataset.dataset_id))
    key_id, key = canonical.sync_integrity_key(manifest.profile_id)
    payload = {
        **preference_record(manifest.profile_id, record_id="racing-ingress-record").model_dump(mode="json"),
        "scope_id": canonical.list_scopes()[0].scope_id,
    }
    clear = canonical_json_bytes(payload)
    envelope = SyncEnvelopeCreate(
        dataset_id=dataset.dataset_id,
        client_envelope_id="racing-push-after-commit",
        device_id=_DEVICE_ID,
        domain="personal_context.record",
        operation="upsert",
        object_id=payload["record_id"],
        parent_id=payload["scope_id"],
        adapter_version=1,
        schema_version=1,
        payload=payload,
        payload_hash="hmac-sha256-v1:" + hmac.new(key, clear, hashlib.sha256).hexdigest(),
        payload_size_bytes=len(clear),
        entity_version=payload["version_id"],
        routing_metadata={"integrity_key_id": key_id, "profile_id": manifest.profile_id, "purge_generation": 0},
        encryption_metadata={"policy": "server_trusted_v1"},
    )
    committed, installer_owned, callback_entered, installer_done = Event(), Event(), Event(), Event()
    repository_type = type(canonical._repository)
    apply_ingress = repository_type.apply_ingress_and_publish
    lease_method = PersonalContextPublicationRelayStore.profile_lease
    skipped_callback_claims = []

    def pause_after_ingress(self, *args, **kwargs):
        result = apply_ingress(self, *args, **kwargs)
        committed.set()
        assert installer_owned.wait(5)
        return result

    @contextmanager
    def observe_callback_lease(self, profile_id, **kwargs):
        callback = committed.is_set() and current_thread().name == "push-worker"
        if callback:
            callback_entered.set()
        with lease_method(self, profile_id, **kwargs) as lease:
            if callback and lease is None:
                skipped_callback_claims.append(profile_id)
            yield lease

    def installer():
        assert committed.wait(5)
        publications = PersonalContextPublicationRelayStore(canonical._repository.database)
        with lease_method(publications, manifest.profile_id):
            installer_owned.set()
            assert callback_entered.wait(5)
            with service.store.personal_context_authority_guard(dataset.dataset_id, manifest.profile_id):
                installer_done.set()

    def push():
        current_thread().name = "push-worker"
        return service.push(
            user_id=_USER_ID,
            dataset_id=dataset.dataset_id,
            device_id=_DEVICE_ID,
            envelopes=[envelope],
            personal_context_exchange=exchange,
        )

    monkeypatch.setattr(repository_type, "apply_ingress_and_publish", pause_after_ingress)
    monkeypatch.setattr(PersonalContextPublicationRelayStore, "profile_lease", observe_callback_lease)
    with ThreadPoolExecutor(max_workers=2) as executor:
        installation = executor.submit(installer)
        pushed = executor.submit(push)
        result = pushed.result(timeout=15)
        assert not result.rejected
        installation.result(timeout=15)
    assert installer_done.is_set()
    assert skipped_callback_claims
    assert not result.rejected
    assert len(result.accepted) == 1
    stored = service.store.get_envelope_by_client_id(dataset.dataset_id, envelope.client_envelope_id)
    assert stored.apply_status == "applied"
    assert canonical.get_record(payload["record_id"]).payload == preference_record().payload
    recovered = service.personal_context_relay.relay_profile(
        user_id=_USER_ID,
        profile_id=manifest.profile_id,
        dataset_id=dataset.dataset_id,
        after_server_cursor=None,
        wall_time_ms=1000,
    )
    assert recovered.source_exhausted
    with canonical._repository.database.transaction() as connection:
        rows = connection.execute(
            "SELECT row_state FROM personal_context_publication_rows WHERE opaque_object_id = ?",
            (payload["record_id"],),
        ).fetchall()
    assert [row[0] for row in rows] == ["acknowledged"]
