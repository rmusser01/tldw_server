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
