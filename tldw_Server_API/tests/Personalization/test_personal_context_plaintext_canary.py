from __future__ import annotations

from pathlib import Path

import pytest
from tldw_profile_core import (
    ConstraintPayload,
    ConventionPayload,
    CorrectionPayload,
    GoalPayload,
    IdentityPayload,
    LegacyUnclassifiedPayload,
    PreferencePayload,
    ProfileRecord,
    RecordKind,
    RelationshipPayload,
    SemanticKey,
    WorkingContextPayload,
)

from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ProfileIntegrityError,
)
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    encoded_master_key,
    global_scope,
    manifest,
    preference_record,
    proposal,
)


def _durable_bytes(root: Path) -> bytes:
    return b"".join(path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file())


def test_canonical_bodies_never_appear_in_database_sidecars_or_logs(tmp_path, monkeypatch, capsys) -> None:
    record_canary = "PROFILE-RECORD-CANARY-DO-NOT-PERSIST-PLAINTEXT"
    proposal_canary = "PROFILE-PROPOSAL-CANARY-DO-NOT-PERSIST-PLAINTEXT"
    runtime_canary = "PROFILE-RUNTIME-CANARY-DO-NOT-PERSIST-PLAINTEXT"
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key(b"a"))
    db = PersonalizationDB.for_path(tmp_path / "Personalization.db")
    repository = PersonalContextRepository(db)
    repository.create_profile(manifest(), global_scope())
    repository.commit_record_version(
        preference_record(value=record_canary),
        expected_version_id=None,
    )
    repository.commit_proposal(proposal(value=proposal_canary))
    repository.set_runtime_policy(
        "profile-a",
        "profile-a-global",
        version_id="runtime-v1",
        expected_version_id=None,
        policy={"note": runtime_canary},
    )
    repository.close()

    durable = _durable_bytes(tmp_path)
    captured = capsys.readouterr()
    assert record_canary.encode("utf-8") not in durable
    assert proposal_canary.encode("utf-8") not in durable
    assert runtime_canary.encode("utf-8") not in durable
    assert record_canary not in captured.out + captured.err
    assert proposal_canary not in captured.out + captured.err
    assert runtime_canary not in captured.out + captured.err


def test_every_canonical_record_kind_stays_out_of_durable_storage(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key(b"a"))
    repository = PersonalContextRepository(PersonalizationDB.for_path(tmp_path / "Personalization.db"))
    repository.create_profile(manifest(), global_scope())
    payloads = (
        IdentityPayload(subject="identity", value="CANARY-IDENTITY"),
        PreferencePayload(subject="preference", polarity="like", value="CANARY-PREFERENCE"),
        RelationshipPayload(subject="relationship", value="CANARY-RELATIONSHIP"),
        CorrectionPayload(subject="correction", value="CANARY-CORRECTION"),
        ConstraintPayload(subject="constraint", value="CANARY-CONSTRAINT"),
        GoalPayload(subject="goal", outcome="CANARY-GOAL"),
        ConventionPayload(subject="convention", value="CANARY-CONVENTION"),
        WorkingContextPayload(subject="working-context", value="CANARY-WORKING-CONTEXT"),
        LegacyUnclassifiedPayload(text="CANARY-LEGACY-UNCLASSIFIED"),
    )
    canaries: list[str] = []
    base = preference_record()
    for index, payload in enumerate(payloads):
        kind = RecordKind(payload.kind)
        record = ProfileRecord.model_validate(
            {
                **base.model_dump(mode="python"),
                "record_id": f"record-kind-{index}",
                "kind": kind,
                "payload": payload,
                "semantic_key": SemanticKey(
                    namespace=kind.value,
                    subject=f"record-kind-{index}",
                ),
                "version_id": f"record-kind-{index}-v1",
                "no_expiry": kind is RecordKind.WORKING_CONTEXT,
            }
        )
        repository.commit_record_version(record, expected_version_id=None)
        canaries.extend(
            value
            for value in payload.model_dump(mode="python").values()
            if isinstance(value, str) and value.startswith("CANARY-")
        )
    repository.close()

    durable = _durable_bytes(tmp_path)
    assert len(canaries) == len(payloads)
    assert all(canary.encode("utf-8") not in durable for canary in canaries)


def test_rejected_proposal_removes_old_ciphertext_and_wrapped_dek(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key(b"a"))
    db = PersonalizationDB.for_path(tmp_path / "Personalization.db")
    repository = PersonalContextRepository(db)
    repository.create_profile(manifest(), global_scope())
    repository.commit_proposal(proposal(value="SHRED-ME"))
    old_ciphertext, old_wrapped_dek = repository.encrypted_version_material("profile-a", "proposal", "proposal-a")

    repository.reject_proposal("profile-a", "proposal-a")
    repository.close()

    durable = _durable_bytes(tmp_path)
    assert old_ciphertext not in durable
    assert old_wrapped_dek not in durable


def test_repository_authentication_errors_never_include_plaintext(tmp_path, monkeypatch) -> None:
    canary = "PROFILE-EXCEPTION-CANARY-DO-NOT-LEAK"
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key(b"a"))
    repository = PersonalContextRepository(PersonalizationDB.for_path(tmp_path / "Personalization.db"))
    repository.create_profile(manifest(), global_scope())
    repository.commit_record_version(preference_record(value=canary), expected_version_id=None)
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

    with pytest.raises(ProfileIntegrityError) as caught:
        repository.get_record("profile-a", "record-a")

    assert canary not in str(caught.value)
