from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from tldw_profile_core import (
    ProfileControls,
    ProfileManifest,
    ProfileProposal,
    ProfileRecord,
    ProfileScope,
    ProposalState,
    SyncMode,
    canonical_bytes,
)
from tldw_profile_core.models import AgentVisibility

from tldw_Server_API.app.core.DB_Management import (
    Personal_Context_Repository as personal_context_repository,
)
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Personalization.personal_context_crypto import (
    EnvelopeAuthenticationError,
)
from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    IngressIdentity,
    PersonalContextPublicationJournal,
    PublicationObject,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ProfileStorageLockedError,
)
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    PersonalContextService,
    ProfileConflictError,
    RecordMutation,
)
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    encoded_master_key,
)

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


@pytest.fixture()
def database(tmp_path, monkeypatch) -> PersonalizationDB:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    return PersonalizationDB.for_path(tmp_path / "personalization.db")


@pytest.fixture()
def service(database: PersonalizationDB) -> PersonalContextService:
    counter: dict[str, int] = {}

    def identifiers(label: str) -> str:
        counter[label] = counter.get(label, 0) + 1
        return f"{label}-{counter[label]}"

    return PersonalContextService(
        PersonalContextRepository(database),
        clock=lambda: NOW,
        id_factory=identifiers,
    )


def _record(
    service: PersonalContextService,
    *,
    record_id: str = "record-a",
    value: str = "concise",
) -> ProfileRecord:
    scope = service.list_scopes()[0]
    created = service.build_manual_record(
        scope_id=scope.scope_id,
        payload={"kind": "preference", "subject": record_id, "polarity": "like", "value": value},
        semantic_key={"namespace": "preference", "subject": record_id},
        controls=ProfileControls(
            sync_mode=SyncMode.SYNCABLE,
            agent_visibility=AgentVisibility.AGENT_VISIBLE,
        ),
    )
    return created.model_copy(update={"record_id": record_id})


def _ingress(service: PersonalContextService, envelope_id: str) -> dict[str, object]:
    record = _record(service, record_id="record-ingress")
    return {
        "identity": IngressIdentity(
            dataset_id="dataset-a",
            device_id="device-a",
            client_envelope_id=envelope_id,
            canonical_payload_digest=_digest(record),
            purge_generation=0,
            wire_entity_version=record.version_id,
        ),
        "domain": "personal_context.record",
        "value": record,
        "base_object_hash": None,
    }


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def _identity(value: object, envelope_id: str) -> IngressIdentity:
    wire_entity_version = getattr(value, "version_id", None) or getattr(
        value, "current_version_id", None
    )
    if isinstance(value, ProfileProposal):
        wire_entity_version = "sync-proposal-sha256:" + hashlib.sha256(
            canonical_bytes(value)
        ).hexdigest()
    return IngressIdentity(
        dataset_id="dataset-a",
        device_id="device-a",
        client_envelope_id=envelope_id,
        canonical_payload_digest=_digest(value),
        purge_generation=0,
        wire_entity_version=str(wire_entity_version),
    )


def test_record_mutation_commits_manifest_and_publication_batch_atomically(
    service: PersonalContextService,
    database: PersonalizationDB,
) -> None:
    service.create_profile()
    record = service.create_record(_record(service))

    with database.transaction() as connection:
        rows = connection.execute(
            """
            SELECT role, batch_ordinal, batch_size
            FROM personal_context_publication_rows
            WHERE profile_publication_sequence = 2
            ORDER BY batch_ordinal
            """
        ).fetchall()
        version = connection.execute(
            """
            SELECT 1 FROM personal_context_object_versions
            WHERE object_type = 'record' AND object_id = ? AND version_id = ?
            """,
            (record.record_id, record.version_id),
        ).fetchone()

    assert version is not None
    assert [row["role"] for row in rows] == ["semantic", "manifest"]
    assert {(row["batch_ordinal"], row["batch_size"]) for row in rows} == {(0, 2), (1, 2)}


def test_ingress_replay_returns_original_result_without_second_manifest_advance(
    service: PersonalContextService,
    database: PersonalizationDB,
) -> None:
    service.create_profile()

    ingress = _ingress(service, "client-envelope-1")
    first = service.apply_sync_ingress(**ingress)
    replay = service.apply_sync_ingress(**ingress)

    assert replay == first
    assert service.get_manifest().revision == first.manifest_revision

    with database.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE personal_context_ingress_receipts "
            "SET wire_entity_version = '' WHERE receipt_id = ?",
            (first.receipt_id,),
        )

    legacy_replay = service.apply_sync_ingress(**ingress)

    assert legacy_replay == first
    with database.transaction() as connection:
        stored_wire_version = connection.execute(
            "SELECT wire_entity_version FROM personal_context_ingress_receipts "
            "WHERE receipt_id = ?",
            (first.receipt_id,),
        ).fetchone()[0]
    assert stored_wire_version == first.wire_entity_version


def test_legacy_ingress_receipt_mismatch_does_not_backfill_wire_identity(
    service: PersonalContextService,
    database: PersonalizationDB,
) -> None:
    service.create_profile()
    ingress = _ingress(service, "client-envelope-legacy-mismatch")
    first = service.apply_sync_ingress(**ingress)
    with database.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE personal_context_ingress_receipts "
            "SET wire_entity_version = '', canonical_payload_digest = ? "
            "WHERE receipt_id = ?",
            ("sha256:" + "0" * 64, first.receipt_id),
        )

    with pytest.raises(ValueError, match="ingress identity reused"):
        service.apply_sync_ingress(**ingress)

    with database.transaction() as connection:
        stored_wire_version = connection.execute(
            "SELECT wire_entity_version FROM personal_context_ingress_receipts "
            "WHERE receipt_id = ?",
            (first.receipt_id,),
        ).fetchone()[0]
    assert stored_wire_version == ""


def test_legacy_receipt_does_not_backfill_when_source_ciphertext_is_corrupt(
    service: PersonalContextService,
    database: PersonalizationDB,
) -> None:
    service.create_profile()
    ingress = _ingress(service, "client-envelope-corrupt-source")
    first = service.apply_sync_ingress(**ingress)
    with database.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE personal_context_ingress_receipts "
            "SET wire_entity_version = '' WHERE receipt_id = ?",
            (first.receipt_id,),
        )
        connection.execute(
            "UPDATE personal_context_publication_rows "
            "SET ciphertext = ? WHERE publication_batch_id = ? "
            "AND opaque_object_id = ?",
            (b"corrupt", first.publication_batch_id, first.resulting_object_id),
        )

    with pytest.raises(ValueError, match="ingress identity reused"):
        service.apply_sync_ingress(**ingress)

    with database.transaction() as connection:
        stored_wire_version = connection.execute(
            "SELECT wire_entity_version FROM personal_context_ingress_receipts "
            "WHERE receipt_id = ?",
            (first.receipt_id,),
        ).fetchone()[0]
    assert stored_wire_version == ""


def test_ingress_id_reuse_with_a_different_digest_is_rejected_before_mutation(
    service: PersonalContextService,
) -> None:
    service.create_profile()
    first = _ingress(service, "client-envelope-1")
    service.apply_sync_ingress(**first)
    changed_record = ProfileRecord.model_validate(first["value"]).model_copy(
        update={"version_id": "record-ingress-changed"}
    )
    changed = dict(first)
    changed["value"] = changed_record
    changed["identity"] = _identity(changed_record, "client-envelope-1")

    with pytest.raises(ValueError, match="ingress identity reused"):
        service.apply_sync_ingress(**changed)

    assert service.get_manifest().revision == 1


def test_ingress_rejects_stale_scope_base_hash_inside_the_write_transaction(
    service: PersonalContextService,
) -> None:
    manifest = service.create_profile()
    original = service.list_scopes()[0]
    current_scope = original.model_copy(
        update={"version_id": "scope-direct-v2", "updated_at": NOW + timedelta(seconds=2)}
    )
    current_manifest = ProfileManifest.model_validate(
        {
            **manifest.model_dump(mode="python"),
            "revision": 1,
            "updated_at": NOW + timedelta(seconds=2),
            "current_version_id": "manifest-direct-v2",
        }
    )
    service._repository.commit_scope_and_manifest(
        current_scope,
        current_manifest,
        expected_scope_version=original.version_id,
        expected_manifest_version=manifest.current_version_id,
    )
    stale = original.model_copy(
        update={"version_id": "scope-stale-v2", "updated_at": NOW + timedelta(seconds=1)}
    )

    with pytest.raises(ProfileConflictError, match="changed concurrently"):
        service.apply_sync_ingress(
            identity=_identity(stale, "scope-stale"),
            domain="personal_context.scope",
            value=stale,
            base_object_hash=_digest(original),
        )

    assert service._repository.get_scope(manifest.profile_id, original.scope_id) == current_scope


def test_ingress_rejects_immutable_record_and_missing_scope_updates(
    service: PersonalContextService,
) -> None:
    service.create_profile()
    current = service.create_record(_record(service))
    immutable = current.model_copy(
        update={
            "version_id": "record-immutable-v2",
            "parent_version_id": current.version_id,
            "created_at": current.created_at + timedelta(seconds=1),
            "updated_at": current.updated_at + timedelta(seconds=1),
        }
    )
    missing_scope = current.model_copy(
        update={
            "record_id": "record-missing-scope",
            "version_id": "record-missing-scope-v1",
            "scope_id": "scope-missing",
            "parent_version_id": None,
        }
    )

    with pytest.raises(ProfileConflictError, match="changed concurrently"):
        service.apply_sync_ingress(
            identity=_identity(immutable, "record-immutable"),
            domain="personal_context.record",
            value=immutable,
            base_object_hash=_digest(current),
        )
    with pytest.raises(KeyError, match="scope"):
        service.apply_sync_ingress(
            identity=_identity(missing_scope, "record-missing-scope"),
            domain="personal_context.record",
            value=missing_scope,
            base_object_hash=None,
        )


def test_ingress_rejects_a_second_global_scope_and_accepts_pending_to_terminal_proposal(
    service: PersonalContextService,
) -> None:
    manifest = service.create_profile()
    global_scope = service.list_scopes()[0]
    duplicate_global = ProfileScope.model_validate(
        {
            **global_scope.model_dump(mode="python"),
            "scope_id": "scope-other-global",
            "version_id": "scope-other-global-v1",
        }
    )

    with pytest.raises(ProfileConflictError, match="changed concurrently"):
        service.apply_sync_ingress(
            identity=_identity(duplicate_global, "scope-other-global"),
            domain="personal_context.scope",
            value=duplicate_global,
            base_object_hash=None,
        )

    from tldw_Server_API.tests.Personalization.personal_context_test_support import proposal

    pending = ProfileProposal.model_validate(
        {
            **proposal(profile_id=manifest.profile_id).model_dump(mode="python"),
            "scope_id": global_scope.scope_id,
            "proposed_record": _record(service, record_id="proposal-record"),
        }
    )
    service.create_proposal(pending)
    terminal = ProfileProposal.model_validate(
        {
            **pending.model_dump(mode="python"),
            "state": ProposalState.REJECTED,
            "proposed_record": None,
            "confidence": None,
        }
    )

    receipt = service.apply_sync_ingress(
        identity=_identity(terminal, "proposal-terminal"),
        domain="personal_context.proposal",
        value=terminal,
        base_object_hash=_digest(pending),
    )

    assert receipt.resulting_object_id == pending.proposal_id
    expected_wire_version = "sync-proposal-sha256:" + hashlib.sha256(
        canonical_bytes(terminal)
    ).hexdigest()
    assert receipt.wire_entity_version == expected_wire_version
    assert service.apply_sync_ingress(
        identity=_identity(terminal, "proposal-terminal"),
        domain="personal_context.proposal",
        value=terminal,
        base_object_hash=_digest(pending),
    ) == receipt
    assert service._repository.get_proposal(manifest.profile_id, pending.proposal_id) == terminal


def test_device_only_ingress_is_excluded_from_the_authority_journal(
    service: PersonalContextService,
    database: PersonalizationDB,
) -> None:
    service.create_profile()
    ingress = _ingress(service, "client-envelope-device-only")
    record = ProfileRecord.model_validate(ingress["value"])
    ingress["value"] = record.model_copy(
        update={
            "controls": ProfileControls(
                sync_mode=SyncMode.DEVICE_ONLY,
                agent_visibility=AgentVisibility.AGENT_VISIBLE,
            )
        }
    )

    with pytest.raises(ValueError, match="Device-only"):
        service.apply_sync_ingress(**ingress)

    with database.transaction() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM personal_context_publication_batches"
        ).fetchone()[0] == 1


def test_concurrent_mutations_allocate_distinct_contiguous_publication_sequences(
    service: PersonalContextService,
    database: PersonalizationDB,
) -> None:
    service.create_profile()

    def create(index: int) -> ProfileRecord:
        for _attempt in range(2):
            try:
                return service.create_record(_record(service, record_id=f"record-{index}"))
            except ProfileConflictError:
                continue
        raise AssertionError("concurrent writer did not retry against the new manifest head")

    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(executor.map(create, range(2)))

    with database.transaction() as connection:
        rows = connection.execute(
            """
            SELECT profile_publication_sequence
            FROM personal_context_publication_batches
            ORDER BY profile_publication_sequence
            """
        ).fetchall()

    assert [row[0] for row in rows] == [1, 2, 3]


def test_publication_rows_do_not_store_canonical_payload_or_domain_in_cleartext(
    service: PersonalContextService,
    database: PersonalizationDB,
) -> None:
    service.create_profile()
    canary = "PUBLICATION-CANARY-DO-NOT-PERSIST-PLAINTEXT"
    service.create_record(_record(service, value=canary))

    with database.transaction() as connection:
        rows = connection.execute("SELECT * FROM personal_context_publication_rows").fetchall()
        durable = "".join(str(value) for row in rows for value in row)

    assert canary not in durable
    assert "personal_context.record" not in durable


def test_compaction_keeps_latest_head_and_does_not_touch_newer_sequences(
    service: PersonalContextService,
    database: PersonalizationDB,
) -> None:
    service.create_profile()
    first = service.create_record(_record(service, record_id="record-a"))
    service.update_record(
        first.record_id,
        RecordMutation(
            payload={
                "kind": "preference",
                "subject": "record-a",
                "polarity": "like",
                "value": "terse",
            }
        ),
        expected_version_id=first.version_id,
    )
    service.create_record(_record(service, record_id="record-b"))

    service._repository.compact_pre_activation("profile-1", through_sequence=3)

    with database.transaction() as connection:
        older = connection.execute(
            """
            SELECT row_state FROM personal_context_publication_rows
            WHERE profile_publication_sequence = 2 AND batch_ordinal = 0
            """
        ).fetchone()
        newer = connection.execute(
            """
            SELECT row_state FROM personal_context_publication_rows
            WHERE profile_publication_sequence = 4
            """
        ).fetchall()

    assert older[0] == "staged"
    assert {row[0] for row in newer} == {"pending"}


def test_publication_aead_rejects_tampered_operation_metadata(
    service: PersonalContextService,
    database: PersonalizationDB,
) -> None:
    service.create_profile()
    service.create_record(_record(service))
    keys = service._repository.key_material_for_test("profile-1")
    with database.transaction(immediate=True) as connection:
        connection.execute(
            """
            UPDATE personal_context_publication_rows
            SET operation = 'tombstone'
            WHERE profile_publication_sequence = 2 AND batch_ordinal = 0
            """
        )
        tampered = connection.execute(
            """
            SELECT * FROM personal_context_publication_rows
            WHERE profile_publication_sequence = 2 AND batch_ordinal = 0
            """
        ).fetchone()

    with pytest.raises(EnvelopeAuthenticationError):
        PersonalContextPublicationJournal(keys).decrypt_row(tampered)


def test_expiring_proposal_at_repository_lock_still_commits_its_publication(
    service: PersonalContextService,
    database: PersonalizationDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    from tldw_Server_API.tests.Personalization.personal_context_test_support import proposal

    pending = ProfileProposal.model_validate(
        {
            **proposal(profile_id=manifest.profile_id).model_dump(mode="python"),
            "scope_id": scope.scope_id,
            "proposed_record": _record(service, record_id="expiring-proposal-record"),
        }
    )
    service.create_proposal(pending)
    with database.transaction() as connection:
        before = connection.execute(
            "SELECT COUNT(*) FROM personal_context_publication_batches"
        ).fetchone()[0]

    class RepositoryExpiredClock(datetime):
        @classmethod
        def now(cls, _tz=None) -> datetime:
            return NOW + timedelta(days=91)

    monkeypatch.setattr(personal_context_repository, "datetime", RepositoryExpiredClock)
    with pytest.raises(ValueError, match="proposal has expired"):
        service.review_proposal(pending.proposal_id, action="accept")

    with database.transaction() as connection:
        after = connection.execute(
            "SELECT COUNT(*) FROM personal_context_publication_batches"
        ).fetchone()[0]
    assert after == before + 1
    assert service._repository.get_proposal(manifest.profile_id, pending.proposal_id).state is ProposalState.EXPIRED


def test_purge_terminalizes_old_publications_and_makes_them_undecryptable(
    service: PersonalContextService,
    database: PersonalizationDB,
) -> None:
    service.create_profile()
    service.create_record(_record(service))
    keys = service._repository.key_material_for_test("profile-1")
    with database.transaction() as connection:
        old_row = connection.execute(
            """
            SELECT * FROM personal_context_publication_rows
            WHERE profile_publication_sequence = 2 AND batch_ordinal = 0
            """
        ).fetchone()

    service.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )

    with database.transaction() as connection:
        purged_row = connection.execute(
            """
            SELECT * FROM personal_context_publication_rows
            WHERE profile_publication_sequence = 2 AND batch_ordinal = 0
            """
        ).fetchone()
        status = connection.execute(
            """
            SELECT status FROM personal_context_publication_batches
            WHERE profile_publication_sequence = 2
            """
        ).fetchone()

    assert PersonalContextPublicationJournal(keys).decrypt_row(old_row)[0] == "personal_context.record"
    assert status[0] == "purge_terminal"
    with pytest.raises(EnvelopeAuthenticationError):
        PersonalContextPublicationJournal(keys).decrypt_row(purged_row)


def test_relay_applied_purge_cannot_invoke_direct_journal_shredding(
    service: PersonalContextService,
    database: PersonalizationDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = service.create_profile()
    service.create_record(_record(service))
    keys = service._repository.key_material_for_test(manifest.profile_id)

    def unexpected_shredding(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("relay must not authorize journal destruction")

    monkeypatch.setattr(
        PersonalContextPublicationJournal,
        "cryptographically_shred_row",
        unexpected_shredding,
    )

    barrier = service.apply_sync_object(
        domain="personal_context.purge",
        value={
            "schema_version": 1,
            "profile_id": manifest.profile_id,
            "purge_generation": 1,
        },
        actor_type="sync",
        actor_id="device-a",
    )

    assert barrier["purge_generation"] == 1
    with database.transaction() as connection:
        old_row = connection.execute(
            """
            SELECT * FROM personal_context_publication_rows
            WHERE profile_publication_sequence = 2 AND batch_ordinal = 0
            """
        ).fetchone()
    assert PersonalContextPublicationJournal(keys).decrypt_row(old_row)[0] == "personal_context.record"


def test_direct_purge_shreds_previously_compacted_old_generation_rows(
    service: PersonalContextService,
    database: PersonalizationDB,
) -> None:
    service.create_profile()
    first = service.create_record(_record(service, record_id="compacted-record"))
    service.update_record(
        first.record_id,
        RecordMutation(
            payload={
                "kind": "preference",
                "subject": first.record_id,
                "polarity": "like",
                "value": "changed",
            }
        ),
        expected_version_id=first.version_id,
    )
    keys = service._repository.key_material_for_test("profile-1")
    service._repository.compact_pre_activation("profile-1", through_sequence=3)
    with database.transaction() as connection:
        compacted = connection.execute(
            """
            SELECT * FROM personal_context_publication_rows
            WHERE profile_publication_sequence = 2 AND batch_ordinal = 0
            """
        ).fetchone()
    assert PersonalContextPublicationJournal(keys).decrypt_row(compacted)[0] == "personal_context.record"

    service.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )

    with database.transaction() as connection:
        shredded = connection.execute(
            """
            SELECT * FROM personal_context_publication_rows
            WHERE profile_publication_sequence = 2 AND batch_ordinal = 0
            """
        ).fetchone()
        current_generation = connection.execute(
            """
            SELECT * FROM personal_context_publication_rows
            WHERE purge_generation = 1
            """
        ).fetchone()
    with pytest.raises(EnvelopeAuthenticationError):
        PersonalContextPublicationJournal(keys).decrypt_row(shredded)
    assert PersonalContextPublicationJournal(keys).decrypt_row(current_generation)[0] == "personal_context.manifest"
    with database.transaction(immediate=True) as connection:
        PersonalContextPublicationJournal.cryptographically_shred_row(
            connection,
            shredded,
        )
    with pytest.raises(EnvelopeAuthenticationError):
        PersonalContextPublicationJournal(keys).decrypt_row(shredded)


def test_mutation_and_compaction_never_invoke_direct_journal_shredding(
    service: PersonalContextService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service.create_profile()
    first = service.create_record(_record(service, record_id="non-destructive-record"))

    def unexpected_shredding(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("only direct confirmed purge may destroy journal data")

    monkeypatch.setattr(
        PersonalContextPublicationJournal,
        "cryptographically_shred_row",
        unexpected_shredding,
    )
    service.update_record(
        first.record_id,
        RecordMutation(
            payload={
                "kind": "preference",
                "subject": first.record_id,
                "polarity": "like",
                "value": "updated",
            }
        ),
        expected_version_id=first.version_id,
    )

    assert service._repository.compact_pre_activation("profile-1", through_sequence=3) > 0


def test_key_rotation_skips_purge_shredded_publication_rows(
    service: PersonalContextService,
) -> None:
    service.create_profile()
    service.create_record(_record(service))
    service.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )

    rotated = service._repository.rotate_encryption_key("profile-1")

    assert rotated.key_version == 2


def test_compaction_rejects_future_watermarks_and_missing_profile_keys(
    service: PersonalContextService,
    database: PersonalizationDB,
) -> None:
    service.create_profile()
    service.create_record(_record(service))

    with pytest.raises(ValueError, match="watermark"):
        service._repository.compact_pre_activation("profile-1", through_sequence=99)

    with database.transaction(immediate=True) as connection:
        connection.execute(
            "DELETE FROM personal_context_profile_keys WHERE profile_id = ?",
            ("profile-1",),
        )
    with pytest.raises(ProfileStorageLockedError):
        service._repository.compact_pre_activation("profile-1", through_sequence=2)


def test_compaction_distinguishes_same_object_id_in_different_domains(
    service: PersonalContextService,
    database: PersonalizationDB,
) -> None:
    service.create_profile()
    keys = service._repository.key_material_for_test("profile-1")
    journal = PersonalContextPublicationJournal(keys)
    with database.transaction(immediate=True) as connection:
        journal.append_batch(
            connection,
            profile_id="profile-1",
            purge_generation=0,
            objects=(
                PublicationObject(
                    domain="personal_context.record",
                    object_id="shared-opaque-id",
                    version_id="record-v1",
                    operation="upsert",
                    role="semantic",
                    canonical=b"{}",
                ),
            ),
            now="2026-09-03T12:00:00.000Z",
        )
        journal.append_batch(
            connection,
            profile_id="profile-1",
            purge_generation=0,
            objects=(
                PublicationObject(
                    domain="personal_context.scope",
                    object_id="shared-opaque-id",
                    version_id="scope-v1",
                    operation="upsert",
                    role="semantic",
                    canonical=b"{}",
                ),
            ),
            now="2026-09-03T12:00:01.000Z",
        )

    service._repository.compact_pre_activation("profile-1", through_sequence=3)

    with database.transaction() as connection:
        states = connection.execute(
            """
            SELECT row_state FROM personal_context_publication_rows
            WHERE profile_publication_sequence IN (2, 3)
            ORDER BY profile_publication_sequence
            """
        ).fetchall()
    assert [row[0] for row in states] == ["pending", "pending"]
