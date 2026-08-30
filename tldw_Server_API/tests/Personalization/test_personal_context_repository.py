from __future__ import annotations

import sqlite3
from datetime import timedelta

import pytest
from tldw_profile_core import (
    ProfileManifest,
    ProfileRecord,
    ProfileScope,
    ProposalState,
    RecordState,
)

from tldw_Server_API.app.core.DB_Management import (
    Personal_Context_Repository as personal_context_repository,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ConcurrentProfileUpdateError,
    ProfileIntegrityError,
    ProfileStorageLockedError,
)
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    encoded_master_key,
    global_scope,
    manifest,
    preference_record,
    proposal,
)


@pytest.fixture
def repository(tmp_path, monkeypatch) -> PersonalContextRepository:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key(b"a"))
    return PersonalContextRepository(PersonalizationDB.for_path(tmp_path / "Personalization.db"))


def test_existing_personalization_database_gains_canonical_schema_and_transaction(
    tmp_path,
) -> None:
    db = PersonalizationDB.for_path(tmp_path / "Personalization.db")

    with db.transaction(immediate=True) as connection:
        connection.execute(
            "INSERT INTO personal_context_receipts(profile_id, receipt_id, version_id, created_at) VALUES (?, ?, ?, ?)",
            ("p", "r", "v", "2026-08-30T00:00:00.000Z"),
        )

    with sqlite3.connect(db.db_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {
        "personal_context_profile_keys",
        "personal_context_object_versions",
        "personal_context_object_heads",
        "personal_context_runtime_heads",
        "personal_context_receipts",
    }.issubset(tables)


def test_personalization_transaction_rolls_back_on_failure(tmp_path) -> None:
    db = PersonalizationDB.for_path(tmp_path / "Personalization.db")

    with pytest.raises(RuntimeError, match="rollback"):
        with db.transaction(immediate=True) as connection:
            connection.execute(
                "INSERT INTO personal_context_receipts("
                "profile_id, receipt_id, version_id, created_at"
                ") VALUES (?, ?, ?, ?)",
                ("p", "r", "v", "2026-08-30T00:00:00.000Z"),
            )
            raise RuntimeError("rollback")

    with sqlite3.connect(db.db_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM personal_context_receipts").fetchone()[0]
    assert count == 0


def test_profile_manifest_scope_and_record_roundtrip(repository) -> None:
    repository.create_profile(manifest(), global_scope())
    record = preference_record()

    repository.commit_record_version(record, expected_version_id=None)

    assert repository.get_manifest("profile-a") == manifest()
    assert repository.get_scope("profile-a", "profile-a-global") == global_scope()
    assert repository.get_record("profile-a", "record-a") == record


def test_profile_reopens_through_personalization_database_for_user(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key(b"a"))
    monkeypatch.setattr(
        DatabasePaths,
        "get_personalization_db_path",
        lambda user_id: tmp_path / f"{user_id}.db",
    )
    PersonalContextRepository(PersonalizationDB.for_user("user-a")).create_profile(manifest(), global_scope())

    reopened = PersonalContextRepository(PersonalizationDB.for_user("user-a"))

    assert reopened.get_manifest("profile-a") == manifest()


def test_missing_key_row_with_surviving_profile_state_fails_closed(
    repository,
) -> None:
    repository.create_profile(manifest(), global_scope())
    repository.commit_record_version(preference_record(), expected_version_id=None)
    with repository.database.transaction(immediate=True) as connection:
        connection.execute(
            "DELETE FROM personal_context_profile_keys WHERE profile_id = ?",
            ("profile-a",),
        )

    with pytest.raises(ProfileStorageLockedError, match="unavailable"):
        repository.create_profile(manifest("profile-b"), global_scope("profile-b"))

    assert repository.get_manifest("profile-b") is None


def test_manifest_and_scope_heads_support_optimistic_revisions(repository) -> None:
    first_manifest = manifest()
    first_scope = global_scope()
    repository.create_profile(first_manifest, first_scope)
    next_manifest = ProfileManifest.model_validate(
        {
            **first_manifest.model_dump(mode="python"),
            "revision": 1,
            "updated_at": first_manifest.updated_at + timedelta(seconds=1),
            "current_version_id": "manifest-v2",
        }
    )
    next_scope = ProfileScope.model_validate(
        {
            **first_scope.model_dump(mode="python"),
            "version_id": "scope-v2",
            "updated_at": first_scope.updated_at + timedelta(seconds=1),
        }
    )

    repository.commit_manifest_version(
        next_manifest,
        expected_version_id="manifest-v1",
    )
    repository.commit_scope(next_scope, expected_version_id="scope-v1")

    assert repository.get_manifest("profile-a") == next_manifest
    assert repository.get_scope("profile-a", "profile-a-global") == next_scope
    with pytest.raises(ConcurrentProfileUpdateError):
        repository.commit_scope(first_scope, expected_version_id="scope-v1")


def test_record_head_compare_and_set_is_transactional(repository) -> None:
    repository.create_profile(manifest(), global_scope())
    original = preference_record()
    repository.commit_record_version(original, expected_version_id=None)
    attempted = preference_record(
        version_id="record-v2",
        parent_version_id="record-v1",
        value="detailed",
    )

    with pytest.raises(ConcurrentProfileUpdateError):
        repository.commit_record_version(
            attempted,
            expected_version_id="stale-version",
        )

    assert repository.get_record("profile-a", "record-a") == original
    assert not repository.version_exists("profile-a", "record", "record-a", "record-v2")


def test_cross_profile_object_id_is_indistinguishable_from_unknown(repository) -> None:
    repository.create_profile(manifest(), global_scope())
    repository.commit_record_version(preference_record(), expected_version_id=None)

    assert repository.get_record("profile-b", "record-a") is None
    assert repository.get_record("profile-b", "does-not-exist") is None


def test_deleted_record_is_a_content_free_tombstone(repository) -> None:
    repository.create_profile(manifest(), global_scope())
    original = preference_record(value="PRIVATE-DELETE-CONTENT")
    repository.commit_record_version(original, expected_version_id=None)
    tombstone = preference_record(
        version_id="record-v2",
        parent_version_id="record-v1",
        state=RecordState.DELETED,
    )

    repository.commit_record_version(
        tombstone,
        expected_version_id="record-v1",
    )

    assert repository.get_record("profile-a", "record-a") == tombstone
    assert tombstone.payload is None and tombstone.semantic_key is None


def test_terminal_proposal_replaces_content_with_receipt(repository) -> None:
    repository.create_profile(manifest(), global_scope())
    pending = proposal(value="PRIVATE-PROPOSAL-CONTENT")
    repository.commit_proposal(
        pending,
        expected_manifest_version="manifest-v1",
    )

    resolved = repository.resolve_proposal("profile-a", "proposal-a", ProposalState.REJECTED)

    assert resolved.state is ProposalState.REJECTED
    assert resolved.proposed_record is None
    assert resolved.confidence is None
    assert repository.get_proposal("profile-a", "proposal-a") == resolved
    assert repository.proposal_version_count("profile-a", "proposal-a") == 1


def test_runtime_policy_is_encrypted_and_optimistic(repository) -> None:
    repository.create_profile(manifest(), global_scope())
    policy = {"enabled": True, "authority": "propose"}

    repository.set_runtime_policy(
        "profile-a",
        "profile-a-global",
        version_id="runtime-v1",
        expected_version_id=None,
        expected_manifest_version="manifest-v1",
        policy=policy,
    )

    assert repository.get_runtime_policy("profile-a", "profile-a-global") == ("runtime-v1", policy)
    with pytest.raises(ConcurrentProfileUpdateError):
        repository.set_runtime_policy(
            "profile-a",
            "profile-a-global",
            version_id="runtime-v2",
            expected_version_id="stale",
            expected_manifest_version="manifest-v1",
            policy={"enabled": False},
        )


def test_standalone_writes_cannot_cross_a_committed_purge_barrier(repository) -> None:
    original = manifest()
    repository.create_profile(original, global_scope())
    barrier = ProfileManifest.model_validate(
        {
            **original.model_dump(mode="python"),
            "revision": 1,
            "purge_generation": 1,
            "updated_at": original.updated_at + timedelta(seconds=1),
            "current_version_id": "manifest-purge-v2",
        }
    )
    repository.purge_profile(
        barrier,
        expected_manifest_version=original.current_version_id,
    )

    with pytest.raises(ConcurrentProfileUpdateError, match="manifest head"):
        repository.commit_proposal(
            proposal(),
            expected_manifest_version=original.current_version_id,
        )
    with pytest.raises(ConcurrentProfileUpdateError, match="manifest head"):
        repository.set_runtime_policy(
            original.profile_id,
            "profile-runtime",
            version_id="runtime-after-purge",
            expected_version_id=None,
            expected_manifest_version=original.current_version_id,
            policy={"enabled": True},
        )
    with pytest.raises(ConcurrentProfileUpdateError, match="purge barrier"):
        repository.commit_record_version(
            preference_record(),
            expected_version_id=None,
        )
    with pytest.raises(ConcurrentProfileUpdateError, match="purge barrier"):
        repository.commit_scope(
            global_scope().model_copy(update={"version_id": "scope-after-purge"}),
            expected_version_id=None,
        )

    assert repository.get_proposal(original.profile_id, "proposal-a") is None
    assert repository.get_runtime_policy(original.profile_id, "profile-runtime") is None


def test_key_provider_persists_replacement_integrity_material(repository) -> None:
    repository.create_profile(manifest(), global_scope())
    original = repository.key_material_for_test("profile-a")
    replacement_integrity_key = b"z" * 32

    with repository.database.transaction(immediate=True) as connection:
        repository._keys.replace_encryption_key(
            "profile-a",
            encryption_key=original.encryption_key,
            integrity_key=replacement_integrity_key,
            expected_key_version=original.key_version,
            integrity_key_version=original.integrity_key_version + 1,
            connection=connection,
        )

    reloaded = repository.key_material_for_test("profile-a")
    assert reloaded.integrity_key == replacement_integrity_key
    assert reloaded.integrity_key_version == original.integrity_key_version + 1


def test_terminal_proposal_retention_prunes_oldest_receipts(
    repository,
    monkeypatch,
) -> None:
    repository.create_profile(manifest(), global_scope())
    monkeypatch.setattr(personal_context_repository, "_MAX_PROPOSAL_HEADS", 1)
    first = proposal(proposal_id="proposal-retained-first")
    repository.commit_proposal(
        first,
        expected_manifest_version="manifest-v1",
    )
    repository.reject_proposal("profile-a", first.proposal_id)
    second = proposal(proposal_id="proposal-retained-second")

    repository.commit_proposal(
        second,
        expected_manifest_version="manifest-v1",
    )

    assert repository.get_proposal("profile-a", first.proposal_id) is None
    assert repository.list_proposals("profile-a", limit=1) == (second,)
    with repository.database.transaction() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM personal_context_receipts WHERE profile_id = ?",
                ("profile-a",),
            ).fetchone()[0]
            == 0
        )


def test_ciphertext_tamper_fails_before_model_parse(repository, monkeypatch) -> None:
    repository.create_profile(manifest(), global_scope())
    repository.commit_record_version(preference_record(), expected_version_id=None)
    with repository.database.transaction(immediate=True) as connection:
        row = connection.execute(
            """
            SELECT ciphertext FROM personal_context_object_versions
            WHERE profile_id = ? AND object_type = 'record' AND object_id = ?
            """,
            ("profile-a", "record-a"),
        ).fetchone()
        damaged = bytearray(row["ciphertext"])
        damaged[-1] ^= 1
        connection.execute(
            """
            UPDATE personal_context_object_versions SET ciphertext = ?
            WHERE profile_id = ? AND object_type = 'record' AND object_id = ?
            """,
            (bytes(damaged), "profile-a", "record-a"),
        )

    def parse_must_not_run(*_args, **_kwargs):
        raise AssertionError("model parse ran before authentication")

    monkeypatch.setattr(ProfileRecord, "model_validate_json", parse_must_not_run)
    with pytest.raises(ProfileIntegrityError, match="authentication"):
        repository.get_record("profile-a", "record-a")


def test_stored_envelope_schema_version_tamper_fails_closed(repository) -> None:
    repository.create_profile(manifest(), global_scope())
    repository.commit_record_version(preference_record(), expected_version_id=None)
    with repository.database.transaction(immediate=True) as connection:
        connection.execute(
            """
            UPDATE personal_context_object_versions SET schema_version = 2
            WHERE profile_id = ? AND object_type = 'record' AND object_id = ?
            """,
            ("profile-a", "record-a"),
        )

    with pytest.raises(ProfileIntegrityError, match="schema version"):
        repository.get_record("profile-a", "record-a")


def test_encryption_key_rotation_rewraps_deks_without_changing_ciphertext(
    repository,
) -> None:
    repository.create_profile(manifest(), global_scope())
    record = preference_record()
    repository.commit_record_version(record, expected_version_id=None)
    before = repository.encrypted_version_details("profile-a", "record", "record-a")
    before_keys = repository.key_material_for_test("profile-a")
    sync_storage_before = repository.sync_encryption_key("profile-a")

    rotated = repository.rotate_encryption_key("profile-a")
    after = repository.encrypted_version_details("profile-a", "record", "record-a")
    sync_storage_after = repository.sync_encryption_key("profile-a")

    assert rotated.key_version == before_keys.key_version + 1
    assert rotated.integrity_key == before_keys.integrity_key
    assert rotated.encryption_key != before_keys.encryption_key
    assert after["ciphertext"] == before["ciphertext"]
    assert after["wrapped_dek"] != before["wrapped_dek"]
    assert after["key_version"] == rotated.key_version
    assert sync_storage_after == sync_storage_before
    assert repository.get_record("profile-a", "record-a") == record


def test_key_rotation_rolls_back_every_rewrap_when_one_envelope_is_invalid(
    repository,
) -> None:
    repository.create_profile(manifest(), global_scope())
    repository.commit_record_version(preference_record(), expected_version_id=None)
    manifest_before = repository.encrypted_version_details("profile-a", "manifest", "profile-a")
    keys_before = repository.key_material_for_test("profile-a")
    with repository.database.transaction(immediate=True) as connection:
        row = connection.execute(
            """
            SELECT wrapped_dek FROM personal_context_object_versions
            WHERE profile_id = ? AND object_type = 'record' AND object_id = ?
            """,
            ("profile-a", "record-a"),
        ).fetchone()
        damaged = bytearray(row["wrapped_dek"])
        damaged[-1] ^= 1
        connection.execute(
            """
            UPDATE personal_context_object_versions SET wrapped_dek = ?
            WHERE profile_id = ? AND object_type = 'record' AND object_id = ?
            """,
            (bytes(damaged), "profile-a", "record-a"),
        )

    with pytest.raises(ProfileIntegrityError, match="authentication"):
        repository.rotate_encryption_key("profile-a")

    assert repository.encrypted_version_details("profile-a", "manifest", "profile-a") == manifest_before
    assert repository.key_material_for_test("profile-a") == keys_before
