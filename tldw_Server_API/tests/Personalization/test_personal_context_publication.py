from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from tldw_profile_core import ProfileControls, ProfileRecord, SyncMode
from tldw_profile_core.models import AgentVisibility

from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    IngressIdentity,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
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
            canonical_payload_digest="sha256:ingress-payload-a",
            purge_generation=0,
        ),
        "domain": "personal_context.record",
        "value": record,
        "base_object_hash": None,
    }


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
) -> None:
    service.create_profile()

    first = service.apply_sync_ingress(**_ingress(service, "client-envelope-1"))
    replay = service.apply_sync_ingress(**_ingress(service, "client-envelope-1"))

    assert replay == first
    assert service.get_manifest().revision == first.manifest_revision


def test_ingress_id_reuse_with_a_different_digest_is_rejected_before_mutation(
    service: PersonalContextService,
) -> None:
    service.create_profile()
    first = _ingress(service, "client-envelope-1")
    service.apply_sync_ingress(**first)
    changed = dict(first)
    changed["identity"] = IngressIdentity(
        dataset_id="dataset-a",
        device_id="device-a",
        client_envelope_id="client-envelope-1",
        canonical_payload_digest="sha256:different-payload",
        purge_generation=0,
    )

    with pytest.raises(ValueError, match="ingress identity reused"):
        service.apply_sync_ingress(**changed)

    assert service.get_manifest().revision == 1


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

    assert older[0] == "shredded"
    assert {row[0] for row in newer} == {"pending"}
