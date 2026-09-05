"""Activation preparation and coverage survive real Personalization restarts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.exceptions import PublicationActivationPending
from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    PersonalContextPublicationJournal,
    PersonalContextPublicationRelayStore,
    PublicationObject,
    PublicationRelayLease,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
)
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    PersonalContextService,
)
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    encoded_master_key,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("legacy", [False, True])
def test_abandoned_preparation_expires_without_covering_source(service: PersonalContextService, legacy: bool) -> None:
    """Another device's relay retires a stale baseline without losing queued writes."""
    from tldw_Server_API.app.core.Sync.v2.personal_context_relay import PersonalContextRelay

    manifest = service.create_profile()
    repository = service._repository
    publications = PersonalContextPublicationRelayStore(repository.database)
    if legacy:
        with repository.database.transaction(immediate=True) as connection:
            connection.execute("DELETE FROM personal_context_publication_rows")
            connection.execute("DELETE FROM personal_context_publication_batches")
            connection.execute("DELETE FROM personal_context_publication_profiles")
    with publications.profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="abandoned-device", lease=lease)
    with repository.database.transaction(immediate=True) as connection:
        connection.execute("UPDATE personal_context_activations SET created_at = '2000-01-01T00:00:00.000Z'")
        before = [tuple(row) for row in connection.execute("SELECT * FROM personal_context_publication_rows")]
    relay = PersonalContextRelay(publications=publications, stage_authority=lambda *_args: None)
    result = relay.relay_profile(
        user_id="user-a",
        profile_id=manifest.profile_id,
        dataset_id="dataset-a",
        after_server_cursor=None,
        row_budget=1,
    )
    assert result.continuation == "personal_context_relay_pending"
    assert result.inspected_rows == 1
    with repository.database.transaction() as connection:
        row = connection.execute("SELECT * FROM personal_context_activations").fetchone()
        assert row["state"] == "expired"
        assert row["activation_id"] == prepared.activation_id
        assert row["baseline_digest"] == prepared.baseline_digest
        assert row["ciphertext"] == row["wrapped_dek"] == row["nonce"] == row["wrapped_dek_nonce"] == b""
        assert [tuple(row) for row in connection.execute("SELECT * FROM personal_context_publication_rows")] == before
        profile = connection.execute(
            "SELECT activation_covered_through_sequence FROM personal_context_publication_profiles"
        ).fetchone()
        assert profile is None if legacy else profile[0] == 0
    with publications.profile_lease(manifest.profile_id) as lease:
        with pytest.raises(ValueError, match="activation_required"):
            repository.complete_activation_install(
                prepared.activation_id,
                prepared.baseline_digest,
                "orphan-sync-receipt",
                home_server_cursor=0,
                lease=lease,
            )
        batch = publications.earliest_nonterminal_batch(manifest.profile_id, row_limit=100, lease=lease)
        assert (batch is None) is legacy
        fresh = repository.prepare_activation(manifest.profile_id, device_id="another-device", lease=lease)
        assert fresh.activation_id != prepared.activation_id
    with pytest.raises(ValueError, match="activation_required"):
        repository.confirm_activation_device(
            prepared.activation_id,
            prepared.baseline_digest,
            "abandoned-device",
            "orphan-ack",
            local_receipt_id="local-receipt",
            dataset_id="dataset-a",
        )


def test_live_preparation_reports_pending_instead_of_drained(service: PersonalContextService) -> None:
    """A live baseline fence is retryable pending, even before ordinary source lookup."""
    from tldw_Server_API.app.core.Sync.v2.personal_context_relay import PersonalContextRelay

    manifest = service.create_profile()
    repository = service._repository
    publications = PersonalContextPublicationRelayStore(repository.database)
    with publications.profile_lease(manifest.profile_id) as lease:
        repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
    result = PersonalContextRelay(publications=publications, stage_authority=lambda *_args: None).relay_profile(
        user_id="user-a",
        profile_id=manifest.profile_id,
        dataset_id="dataset-a",
        after_server_cursor=None,
    )
    assert result.continuation == "personal_context_relay_pending"
    assert not result.source_exhausted


@pytest.mark.parametrize("unowned", [None, PublicationRelayLease("wrong-profile", "wrong-owner")])
def test_abandoned_preparation_requires_current_lease(
    service: PersonalContextService,
    unowned: PublicationRelayLease | None,
) -> None:
    """Neither a lease-free caller nor a stale owner may shred a prepared baseline."""
    manifest = service.create_profile()
    repository = service._repository
    publications = PersonalContextPublicationRelayStore(repository.database)
    with publications.profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
    with repository.database.transaction(immediate=True) as connection:
        connection.execute("UPDATE personal_context_activations SET created_at = '2000-01-01T00:00:00.000Z'")
    with pytest.raises(RuntimeError):
        publications.earliest_nonterminal_batch(manifest.profile_id, row_limit=1, lease=unowned)
    assert repository.load_activation(prepared.activation_id) == prepared


@pytest.fixture()
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PersonalContextService:
    """Create a real encrypted canonical profile under the trusted pytest root."""
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    return PersonalContextService(PersonalContextRepository(PersonalizationDB.for_path(tmp_path / "activation.db")))


def test_prepare_restarts_at_whole_batch_watermark(service: PersonalContextService) -> None:
    """A restart returns identical encrypted baseline bytes and an unsplit batch."""
    manifest = service.create_profile()
    repository = service._repository
    publications = PersonalContextPublicationRelayStore(repository.database)
    with publications.profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
    with repository.database.transaction() as connection:
        last = connection.execute(
            "SELECT MAX(profile_publication_sequence) FROM personal_context_publication_batches"
        ).fetchone()[0]
        ciphertext = connection.execute("SELECT ciphertext FROM personal_context_activations").fetchone()[0]
    restarted = PersonalContextRepository(PersonalizationDB.for_path(repository.database.db_path))
    with publications.profile_lease(manifest.profile_id) as lease:
        replay = restarted.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
    assert prepared == replay
    assert prepared.publication_watermark == last
    assert prepared.activation_epoch is None
    with repository.database.transaction() as connection:
        assert connection.execute("SELECT ciphertext FROM personal_context_activations").fetchone()[0] == ciphertext


def test_coverage_cas_precedes_compaction_and_preserves_terminal_proof(
    service: PersonalContextService,
) -> None:
    """Source bodies cannot disappear before exact install coverage is durable."""
    manifest = service.create_profile()
    repository = service._repository
    publications = PersonalContextPublicationRelayStore(repository.database)
    with publications.profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
        with pytest.raises(ValueError, match="activation"):
            repository.compact_activation(prepared.activation_id)
        installed = repository.complete_activation_install(
            prepared.activation_id,
            prepared.baseline_digest,
            "sync-receipt-0123456789",
            home_server_cursor=1,
            lease=lease,
        )
    repository.compact_activation(prepared.activation_id)
    with repository.database.transaction() as connection:
        proof = connection.execute(
            "SELECT activation_covered_through_sequence FROM personal_context_publication_profiles"
        ).fetchone()[0]
        statuses = connection.execute("SELECT DISTINCT status FROM personal_context_publication_batches").fetchall()
        bodies = connection.execute("SELECT ciphertext FROM personal_context_publication_rows").fetchall()
    assert proof == prepared.publication_watermark
    assert [row[0] for row in statuses] == ["covered_by_activation"]
    assert all(row[0] == b"" for row in bodies)
    assert installed.activation_epoch and installed.continuity_token
    with publications.profile_lease(manifest.profile_id) as lease:
        assert (
            repository.complete_activation_install(
                prepared.activation_id,
                prepared.baseline_digest,
                "sync-receipt-0123456789",
                home_server_cursor=1,
                lease=lease,
            )
            == installed
        )


def test_prepared_activation_blocks_relay_until_coverage_commits(service: PersonalContextService) -> None:
    """A racing canonical edit waits behind the baseline checkpoint after restart."""
    manifest = service.create_profile()
    repository = service._repository
    publications = PersonalContextPublicationRelayStore(repository.database)
    with publications.profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
    service.create_manual_record(
        scope_id=service.list_scopes()[0].scope_id,
        payload={"kind": "preference", "subject": "race", "polarity": "like", "value": "racing value"},
        semantic_key={"namespace": "preference", "subject": "race"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    assert json.loads(prepared.baseline)["records"] == []
    with publications.profile_lease(manifest.profile_id) as lease:
        with pytest.raises(PublicationActivationPending):
            publications.earliest_nonterminal_batch(manifest.profile_id, row_limit=100, lease=lease)
        repository.complete_activation_install(
            prepared.activation_id,
            prepared.baseline_digest,
            "sync-receipt-0123456789",
            home_server_cursor=0,
            lease=lease,
        )
    repository.compact_activation(prepared.activation_id)
    with publications.profile_lease(manifest.profile_id) as lease:
        batch = publications.earliest_nonterminal_batch(manifest.profile_id, row_limit=100, lease=lease)
    assert batch is not None
    assert batch.profile_publication_sequence == prepared.publication_watermark + 1
    assert b"racing value" in batch.rows[0].canonical


def test_baseline_survives_profile_key_rotation(service: PersonalContextService) -> None:
    """Rotating the profile key must rewrap the pending baseline as well as source rows."""
    manifest = service.create_profile()
    repository = service._repository
    with PersonalContextPublicationRelayStore(repository.database).profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
    repository.rotate_encryption_key(manifest.profile_id)
    assert repository.load_activation(prepared.activation_id) == prepared


def test_acknowledgment_is_device_bound_and_continuity_is_canonical(service: PersonalContextService) -> None:
    """Only the exact installed device receipt enables the persisted current pair."""
    manifest = service.create_profile()
    repository = service._repository
    with PersonalContextPublicationRelayStore(repository.database).profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
        installed = repository.complete_activation_install(
            prepared.activation_id,
            prepared.baseline_digest,
            "sync-install-0123456789",
            home_server_cursor=0,
            lease=lease,
        )
    arguments = {
        "profile_id": manifest.profile_id,
        "device_id": "device-a",
        "dataset_id": "dataset-a",
        "activation_epoch": installed.activation_epoch,
        "continuity_token": installed.continuity_token,
    }
    with pytest.raises(ValueError, match="activation_required"):
        repository.validate_activation_exchange(**arguments)
    for _ in range(2):
        active = repository.confirm_activation_device(
            installed.activation_id,
            installed.baseline_digest,
            "device-a",
            "sync-ack-0123456789",
            local_receipt_id="local-ack-0123456789",
            dataset_id="dataset-a",
        )
        assert active.state == "active"
    assert repository.validate_activation_exchange(**arguments).continuity_token == installed.continuity_token
    with pytest.raises(ValueError, match="activation_required"):
        repository.validate_activation_exchange(**{**arguments, "device_id": "device-b"})
    with pytest.raises(ValueError, match="activation_required"):
        repository.validate_activation_exchange(**{**arguments, "dataset_id": "dataset-b"})
    with pytest.raises(ValueError, match="activation_required"):
        repository.confirm_activation_device(
            installed.activation_id,
            installed.baseline_digest,
            "device-a",
            "different-sync-ack",
            local_receipt_id="local-ack-0123456789",
            dataset_id="dataset-a",
        )
    with repository.database.transaction(immediate=True) as connection:
        connection.execute("UPDATE personal_context_publication_profiles SET continuity_token = NULL")
    with pytest.raises(ValueError, match="activation_required"):
        repository.validate_activation_exchange(**arguments)


def test_existing_purge_removes_readable_baseline_and_invalidates_pair(service: PersonalContextService) -> None:
    """The new encrypted store participates in the existing canonical purge boundary."""
    manifest = service.create_profile()
    repository = service._repository
    with PersonalContextPublicationRelayStore(repository.database).profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
        repository.complete_activation_install(
            prepared.activation_id,
            prepared.baseline_digest,
            "sync-install-0123456789",
            home_server_cursor=0,
            lease=lease,
        )
    service.purge_profile(mode="everywhere", confirmation="DELETE EVERYWHERE", expected_purge_generation=0)
    with pytest.raises(ValueError, match="activation_required"):
        repository.load_activation(prepared.activation_id)
    with repository.database.transaction() as connection:
        row = connection.execute(
            "SELECT activation_epoch, continuity_token FROM personal_context_publication_profiles"
        ).fetchone()
    assert tuple(row) == (None, None)


def test_new_baseline_required_after_continuity_invalidation(service: PersonalContextService) -> None:
    """A repeated bootstrap cannot return a permanently invalid installed pair."""
    manifest = service.create_profile()
    repository = service._repository
    publications = PersonalContextPublicationRelayStore(repository.database)
    with publications.profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
        installed = repository.complete_activation_install(
            prepared.activation_id,
            prepared.baseline_digest,
            "sync-install-0123456789",
            home_server_cursor=0,
            lease=lease,
        )
    with repository.database.transaction(immediate=True) as connection:
        connection.execute("UPDATE personal_context_publication_profiles SET continuity_token = NULL")
    with publications.profile_lease(manifest.profile_id) as lease:
        replacement = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
    assert replacement.activation_id != installed.activation_id
    assert replacement.state == "prepared"


@pytest.mark.parametrize("installed", [False, True])
def test_another_device_cannot_cover_unpublished_changes(
    service: PersonalContextService,
    installed: bool,
) -> None:
    """New baselines cannot hide unpublished edits from devices with an older baseline."""
    manifest = service.create_profile()
    repository = service._repository
    publications = PersonalContextPublicationRelayStore(repository.database)
    with publications.profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
        if installed:
            repository.complete_activation_install(
                prepared.activation_id,
                prepared.baseline_digest,
                "sync-install-0123456789",
                home_server_cursor=0,
                lease=lease,
            )
    service.create_manual_record(
        scope_id=service.list_scopes()[0].scope_id,
        payload={"kind": "preference", "subject": "second", "polarity": "like", "value": "second device"},
        semantic_key={"namespace": "preference", "subject": "second"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    with publications.profile_lease(manifest.profile_id) as lease:
        with pytest.raises(ValueError, match="activation_pending"):
            repository.prepare_activation(manifest.profile_id, device_id="device-b", lease=lease)


def test_service_requires_exact_sync_verification_before_coverage(service: PersonalContextService) -> None:
    """An unverified independent receipt leaves the encrypted preparation untouched."""
    from tldw_Server_API.app.core.Personalization.personal_context_activation import PersonalContextActivationService

    manifest = service.create_profile()
    activation_service = PersonalContextActivationService(service._repository)
    prepared = activation_service.prepare(manifest.profile_id, device_id="device-a")
    with pytest.raises(ValueError, match="activation_required"):
        activation_service.install(
            prepared.activation_id,
            prepared.baseline_digest,
            install=lambda _prepared: {"receipt_id": "receipt-0123456789", "home_server_cursor": 0},
            verify=lambda _prepared, _receipt: False,
        )
    assert service._repository.load_activation(prepared.activation_id).state == "prepared"


def test_three_row_batch_watermark_is_sequence_not_row_count(service: PersonalContextService) -> None:
    """A whole committed batch with two semantic objects has one activation sequence."""
    from tldw_profile_core import canonical_bytes

    manifest = service.create_profile()
    record = service.create_manual_record(
        scope_id=service.list_scopes()[0].scope_id,
        payload={"kind": "preference", "subject": "batch", "polarity": "like", "value": "three rows"},
        semantic_key={"namespace": "preference", "subject": "batch"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    repository = service._repository
    scope = service.list_scopes()[0]
    head = service.get_manifest()
    with repository.database.transaction(immediate=True) as connection:
        journal = PersonalContextPublicationJournal(repository._keys.load(manifest.profile_id, connection=connection))
        batch = journal.append_batch(
            connection,
            profile_id=manifest.profile_id,
            purge_generation=0,
            now="2026-09-05T00:00:00Z",
            objects=(
                PublicationObject(
                    "personal_context.scope",
                    scope.scope_id,
                    scope.version_id,
                    "upsert",
                    "semantic",
                    canonical_bytes(scope),
                ),
                PublicationObject(
                    "personal_context.record",
                    record.record_id,
                    record.version_id,
                    "upsert",
                    "semantic",
                    canonical_bytes(record),
                ),
                PublicationObject(
                    "personal_context.manifest",
                    head.profile_id,
                    head.current_version_id,
                    "upsert",
                    "manifest",
                    canonical_bytes(head),
                ),
            ),
        )
    with PersonalContextPublicationRelayStore(repository.database).profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
    assert batch.batch_size == 3
    assert prepared.publication_watermark == batch.profile_publication_sequence
    assert len(json.loads(prepared.baseline)["records"]) == 1


def test_legacy_exact_heads_prepare_at_zero_without_source_history(service: PersonalContextService) -> None:
    """A legacy linked profile does not need a fabricated source batch to activate."""
    manifest = service.create_profile()
    repository = service._repository
    with repository.database.transaction(immediate=True) as connection:
        connection.execute("DELETE FROM personal_context_publication_rows")
        connection.execute("DELETE FROM personal_context_publication_batches")
        connection.execute("DELETE FROM personal_context_publication_profiles")
    with PersonalContextPublicationRelayStore(repository.database).profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
        installed = repository.complete_activation_install(
            prepared.activation_id,
            prepared.baseline_digest,
            "sync-install-0123456789",
            home_server_cursor=0,
            lease=lease,
        )
    assert prepared.publication_watermark == 0
    assert installed.state == "installed"


def test_baseline_metadata_tampering_is_authenticated(service: PersonalContextService) -> None:
    """A changed watermark cannot authorize covering unrepresented source batches."""
    from tldw_Server_API.app.core.Personalization.personal_context_crypto import EnvelopeAuthenticationError

    manifest = service.create_profile()
    repository = service._repository
    publications = PersonalContextPublicationRelayStore(repository.database)
    with publications.profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
    with repository.database.transaction(immediate=True) as connection:
        connection.execute("UPDATE personal_context_activations SET publication_watermark = publication_watermark + 1")
    with pytest.raises(EnvelopeAuthenticationError):
        repository.load_activation(prepared.activation_id)


def test_stale_lease_cannot_commit_coverage(service: PersonalContextService) -> None:
    """An expired or displaced installer leaves the baseline prepared."""
    manifest = service.create_profile()
    repository = service._repository
    with PersonalContextPublicationRelayStore(repository.database).profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
    with pytest.raises(ValueError, match="activation_required"):
        repository.complete_activation_install(
            prepared.activation_id,
            prepared.baseline_digest,
            "sync-install-0123456789",
            home_server_cursor=0,
            lease=PublicationRelayLease(manifest.profile_id, "stale-owner"),
        )
    assert repository.load_activation(prepared.activation_id).state == "prepared"


def test_activation_storage_never_persists_plaintext_canary(service: PersonalContextService) -> None:
    """The database and WAL only retain protected baseline and publication bytes."""
    manifest = service.create_profile()
    canary = "TASK13162-PROTECTED-CANARY-8ad3c965"
    service.create_manual_record(
        scope_id=service.list_scopes()[0].scope_id,
        payload={"kind": "preference", "subject": "canary", "polarity": "like", "value": canary},
        semantic_key={"namespace": "preference", "subject": "canary"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    repository = service._repository
    with PersonalContextPublicationRelayStore(repository.database).profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
    assert canary.encode() in prepared.baseline
    for path in Path(repository.database.db_path).parent.glob("activation.db*"):
        assert canary.encode() not in path.read_bytes()
    assert canary not in repr(prepared)


def test_missing_publication_journal_rolls_back_canonical_write(
    service: PersonalContextService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capability gap may not accept a canonical edit without its source journal."""

    manifest = service.create_profile()

    def unavailable(*args: Any, **kwargs: Any) -> None:
        """Simulate the durable source store refusing publication."""
        raise RuntimeError("publication storage unavailable")

    monkeypatch.setattr(PersonalContextPublicationJournal, "append_batch", unavailable)
    with pytest.raises(RuntimeError, match="publication storage unavailable"):
        service.create_manual_record(
            scope_id=service.list_scopes()[0].scope_id,
            payload={"kind": "preference", "subject": "gap", "polarity": "like", "value": "must roll back"},
            semantic_key={"namespace": "preference", "subject": "gap"},
            controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
        )
    assert service.get_manifest() == manifest
    assert service.list_records() == ()


def test_baseline_install_has_bounded_lease_for_large_snapshot(
    service: PersonalContextService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An activation may encrypt a full snapshot longer than the ordinary relay budget."""
    import time

    from tldw_Server_API.app.core.Personalization.personal_context_activation import PersonalContextActivationService
    from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
        PreparedPersonalContextActivation,
    )

    manifest = service.create_profile()
    activation_service = PersonalContextActivationService(service._repository)
    prepared = activation_service.prepare(manifest.profile_id, device_id="device-a")
    wall = [time.time_ns()]
    monkeypatch.setattr(time, "time_ns", lambda: wall[0])

    def install(_prepared: PreparedPersonalContextActivation) -> dict[str, Any]:
        """Advance a deterministic clock across a two-second independent install."""
        wall[0] += 2_000_000_000
        return {"receipt_id": "receipt-0123456789", "home_server_cursor": 0}

    installed = activation_service.install(
        prepared.activation_id,
        prepared.baseline_digest,
        install=install,
        verify=lambda _prepared, _receipt: True,
    )
    assert installed.state == "installed"


def test_exchange_validation_reads_content_free_activation_proof(
    service: PersonalContextService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ordinary exchange validates proof without decrypting the whole baseline."""
    from tldw_Server_API.app.core.Personalization.personal_context_crypto import (
        EncryptedEnvelope,
        EnvelopeCipher,
    )

    manifest = service.create_profile()
    repository = service._repository
    with PersonalContextPublicationRelayStore(repository.database).profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
        installed = repository.complete_activation_install(
            prepared.activation_id,
            prepared.baseline_digest,
            "sync-install-0123456789",
            home_server_cursor=0,
            lease=lease,
        )
    repository.confirm_activation_device(
        installed.activation_id,
        installed.baseline_digest,
        "device-a",
        "sync-ack-0123456789",
        local_receipt_id="local-ack-0123456789",
        dataset_id="dataset-a",
    )
    decrypt = EnvelopeCipher.decrypt

    def guarded_decrypt(self: EnvelopeCipher, envelope: EncryptedEnvelope, aad: bytes) -> bytes:
        """Forbid bulk baseline decryption while retaining ordinary manifest verification."""
        assert b"personal-context-activation-v1" not in aad
        return decrypt(self, envelope, aad)

    monkeypatch.setattr(EnvelopeCipher, "decrypt", guarded_decrypt)
    repository.validate_activation_exchange(
        profile_id=manifest.profile_id,
        device_id="device-a",
        dataset_id="dataset-a",
        activation_epoch=installed.activation_epoch,
        continuity_token=installed.continuity_token,
    )


def test_canonical_writer_racing_snapshot_commits_after_its_whole_batch_watermark(
    service: PersonalContextService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A separate writer cannot commit between the exact-head read and watermark save."""
    manifest = service.create_profile()
    repository = service._repository
    writer = PersonalContextService(
        PersonalContextRepository(
            PersonalizationDB.for_path(repository.database.db_path),
        )
    )
    record = writer.build_manual_record(
        scope_id=writer.list_scopes()[0].scope_id,
        payload={"kind": "preference", "subject": "racing", "polarity": "like", "value": "after snapshot"},
        semantic_key={"namespace": "preference", "subject": "racing"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    snapshot_read, writer_started = Event(), Event()
    snapshot = repository.sync_bootstrap_snapshot

    def held_snapshot(*args: Any, **kwargs: Any) -> Any:
        """Expose the interleaving while the canonical read transaction remains live."""
        result = snapshot(*args, **kwargs)
        snapshot_read.set()
        assert writer_started.wait(5)
        return result

    def racing_write() -> None:
        """Commit from a separate database instance after activation has read its heads."""
        assert snapshot_read.wait(5)
        writer_started.set()
        writer.create_record(record)

    monkeypatch.setattr(repository, "sync_bootstrap_snapshot", held_snapshot)
    with ThreadPoolExecutor(max_workers=1) as executor:
        write = executor.submit(racing_write)
        with PersonalContextPublicationRelayStore(repository.database).profile_lease(manifest.profile_id) as lease:
            prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
        write.result(timeout=10)
    assert json.loads(prepared.baseline)["records"] == []
    assert writer.list_records() == (record,)
    with repository.database.transaction() as connection:
        sequence = connection.execute(
            "SELECT MAX(profile_publication_sequence) FROM personal_context_publication_batches"
        ).fetchone()[0]
    assert sequence == prepared.publication_watermark + 1


def test_same_millisecond_replacement_replays_latest_insert(
    service: PersonalContextService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Random activation IDs must not decide which durable preparation is current."""
    from types import SimpleNamespace
    from uuid import UUID

    from tldw_Server_API.app.core.DB_Management import Personal_Context_Repository as repository_module

    manifest = service.create_profile()
    repository = service._repository
    identifiers = iter((UUID(int=2), UUID(int=1)))
    monkeypatch.setattr(repository_module, "uuid", SimpleNamespace(uuid4=lambda: next(identifiers)))
    monkeypatch.setattr(repository_module, "_now_text", lambda: "2026-09-05T00:00:00.000Z")
    publications = PersonalContextPublicationRelayStore(repository.database)
    with publications.profile_lease(manifest.profile_id) as lease:
        old = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
        repository.complete_activation_install(
            old.activation_id,
            old.baseline_digest,
            "sync-install-0123456789",
            home_server_cursor=0,
            lease=lease,
        )
    with repository.database.transaction(immediate=True) as connection:
        connection.execute("UPDATE personal_context_publication_profiles SET continuity_token = NULL")
    with publications.profile_lease(manifest.profile_id) as lease:
        current = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
        replay = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
    assert replay == current


def test_installed_replay_never_returns_invalidated_continuity(service: PersonalContextService) -> None:
    """A broken current proof cannot be revived by replaying an older install receipt."""
    manifest = service.create_profile()
    repository = service._repository
    publications = PersonalContextPublicationRelayStore(repository.database)
    with publications.profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
        repository.complete_activation_install(
            prepared.activation_id,
            prepared.baseline_digest,
            "sync-install-0123456789",
            home_server_cursor=0,
            lease=lease,
        )
    with repository.database.transaction(immediate=True) as connection:
        connection.execute("UPDATE personal_context_publication_profiles SET continuity_token = NULL")
    with publications.profile_lease(manifest.profile_id) as lease:
        with pytest.raises(ValueError, match="activation_required"):
            repository.complete_activation_install(
                prepared.activation_id,
                prepared.baseline_digest,
                "sync-install-0123456789",
                home_server_cursor=0,
                lease=lease,
            )


@pytest.mark.parametrize("installed", [False, True])
def test_expired_sync_baseline_permits_new_preparation_without_losing_coverage(
    service: PersonalContextService,
    installed: bool,
) -> None:
    """Verified expiry releases one interrupted activation without erasing terminal proof."""
    manifest = service.create_profile()
    repository = service._repository
    publications = PersonalContextPublicationRelayStore(repository.database)
    with publications.profile_lease(manifest.profile_id) as lease:
        prepared = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
        if installed:
            repository.complete_activation_install(
                prepared.activation_id,
                prepared.baseline_digest,
                "sync-install-0123456789",
                home_server_cursor=0,
                lease=lease,
            )
        for _ in range(2):
            repository.expire_activation(
                prepared.activation_id,
                prepared.baseline_digest,
                sync_receipt_id="sync-install-0123456789",
                lease=lease,
            )
        fresh = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
    assert fresh.activation_id != prepared.activation_id
    assert fresh.baseline == prepared.baseline
    with pytest.raises(ValueError, match="activation_required"):
        repository.load_activation(prepared.activation_id)
    with repository.database.transaction() as connection:
        row = connection.execute(
            "SELECT state, ciphertext FROM personal_context_activations WHERE activation_id = ?",
            (prepared.activation_id,),
        ).fetchone()
        covered = connection.execute(
            "SELECT activation_covered_through_sequence FROM personal_context_publication_profiles"
        ).fetchone()[0]
    assert tuple(row) == ("expired", b"")
    assert covered == (prepared.publication_watermark if installed else 0)


def test_expiring_one_device_keeps_peers_continuity_and_acknowledgment(service: PersonalContextService) -> None:
    """Expiration cannot revoke the shared profile pair or another device's receipt."""
    manifest = service.create_profile()
    repository = service._repository
    publications = PersonalContextPublicationRelayStore(repository.database)
    active = {}
    for device_id in ("device-a", "device-b"):
        with publications.profile_lease(manifest.profile_id) as lease:
            prepared = repository.prepare_activation(manifest.profile_id, device_id=device_id, lease=lease)
            repository.complete_activation_install(
                prepared.activation_id,
                prepared.baseline_digest,
                f"sync-install-{device_id}",
                home_server_cursor=0,
                lease=lease,
            )
        active[device_id] = repository.confirm_activation_device(
            prepared.activation_id,
            prepared.baseline_digest,
            device_id,
            f"sync-ack-{device_id}",
            local_receipt_id=f"local-ack-{device_id}",
            dataset_id="dataset-a",
        )
    expired = active["device-a"]
    with publications.profile_lease(manifest.profile_id) as lease:
        repository.expire_activation(
            expired.activation_id,
            expired.baseline_digest,
            sync_receipt_id="sync-install-device-a",
            lease=lease,
        )
    with pytest.raises(ValueError, match="activation_required"):
        repository.validate_activation_exchange(
            profile_id=manifest.profile_id,
            device_id="device-a",
            dataset_id="dataset-a",
            activation_epoch=expired.activation_epoch,
            continuity_token=expired.continuity_token,
        )
    peer = active["device-b"]
    assert (
        repository.validate_activation_exchange(
            profile_id=manifest.profile_id,
            device_id="device-b",
            dataset_id="dataset-a",
            activation_epoch=peer.activation_epoch,
            continuity_token=peer.continuity_token,
        ).continuity_token
        == peer.continuity_token
    )
    repository.rotate_encryption_key(manifest.profile_id)
    assert repository.load_activation(peer.activation_id).baseline == peer.baseline


def test_expired_latest_activation_cannot_fall_back_to_older_device_ack(service: PersonalContextService) -> None:
    """A device's superseded receipt is not a route around its expired current baseline."""
    manifest = service.create_profile()
    repository = service._repository
    publications = PersonalContextPublicationRelayStore(repository.database)
    identifiers = []
    for index in range(2):
        with publications.profile_lease(manifest.profile_id) as lease:
            prepared = repository.prepare_activation(
                manifest.profile_id,
                device_id="device-a",
                fresh=True,
                lease=lease,
            )
            installed = repository.complete_activation_install(
                prepared.activation_id,
                prepared.baseline_digest,
                f"sync-install-0123456789-{index}",
                home_server_cursor=0,
                lease=lease,
            )
        repository.confirm_activation_device(
            installed.activation_id,
            installed.baseline_digest,
            "device-a",
            f"sync-ack-0123456789-{index}",
            local_receipt_id=f"local-ack-0123456789-{index}",
            dataset_id="dataset-a",
        )
        identifiers.append(installed.activation_id)
    with publications.profile_lease(manifest.profile_id) as lease:
        repository.expire_activation(
            installed.activation_id,
            installed.baseline_digest,
            sync_receipt_id="sync-install-0123456789-1",
            lease=lease,
        )
    with pytest.raises(ValueError, match="activation_required"):
        repository.validate_activation_exchange(
            profile_id=manifest.profile_id,
            device_id="device-a",
            dataset_id="dataset-a",
            activation_epoch=installed.activation_epoch,
            continuity_token=installed.continuity_token,
        )
    with publications.profile_lease(manifest.profile_id) as lease:
        fresh = repository.prepare_activation(manifest.profile_id, device_id="device-a", lease=lease)
    assert fresh.activation_id not in identifiers
