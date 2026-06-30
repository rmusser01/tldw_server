import json
from typing import Generator

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import (
    VNAssetPacksRepository,
    ensure_vn_asset_tables,
)


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="vn-assets-test-client")
    yield database
    database.close_connection()


def test_vn_asset_tables_are_created(chacha_db: CharactersRAGDB) -> None:
    ensure_vn_asset_tables(chacha_db)

    cursor = chacha_db.execute_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vn_asset_%'"
    )
    table_names = {row[0] for row in cursor.fetchall()}
    assert {
        "vn_asset_packs",
        "vn_asset_slots",
        "vn_asset_items",
        "vn_asset_batches",
    }.issubset(table_names)


def test_ensure_vn_asset_tables_rejects_non_sqlite_before_transaction() -> None:
    class NonSqliteDB:
        backend_type = BackendType.POSTGRESQL

        def transaction(self):
            raise AssertionError("transaction should not be opened for unsupported backends")

    with pytest.raises(NotImplementedError, match="SQLite ChaChaNotes"):
        ensure_vn_asset_tables(NonSqliteDB())  # type: ignore[arg-type]


def test_repository_rejects_non_sqlite_backend() -> None:
    class NonSqliteDB:
        backend_type = BackendType.POSTGRESQL

    with pytest.raises(NotImplementedError, match="SQLite ChaChaNotes"):
        VNAssetPacksRepository(NonSqliteDB())  # type: ignore[arg-type]


def test_repository_constructor_does_not_create_schema(chacha_db: CharactersRAGDB) -> None:
    VNAssetPacksRepository(chacha_db)

    cursor = chacha_db.execute_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vn_asset_%'"
    )
    assert cursor.fetchall() == []


def test_idempotency_claim_replays_completed_response_and_rejects_payload_conflict(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNAssetPacksRepository.initialized(chacha_db)

    first_record, claimed = repo.claim_idempotency_record(
        owner_user_id=42,
        scope="vn_asset_export",
        resource_id="pack:1",
        idempotency_key="export-1",
        payload_hash="hash-a",
    )
    assert claimed is True
    assert first_record["status"] == "in_progress"

    second_record, second_claimed = repo.claim_idempotency_record(
        owner_user_id=42,
        scope="vn_asset_export",
        resource_id="pack:1",
        idempotency_key="export-1",
        payload_hash="hash-a",
    )
    assert second_claimed is False
    assert second_record["status"] == "in_progress"

    repo.complete_idempotency_record(
        owner_user_id=42,
        scope="vn_asset_export",
        resource_id="pack:1",
        idempotency_key="export-1",
        payload_hash="hash-a",
        response={"job_id": "job-1"},
    )
    completed_record, replay_claimed = repo.claim_idempotency_record(
        owner_user_id=42,
        scope="vn_asset_export",
        resource_id="pack:1",
        idempotency_key="export-1",
        payload_hash="hash-a",
    )
    assert replay_claimed is False
    assert completed_record["status"] == "completed"
    assert json.loads(completed_record["response_json"]) == {"job_id": "job-1"}

    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.claim_idempotency_record(
            owner_user_id=42,
            scope="vn_asset_export",
            resource_id="pack:1",
            idempotency_key="export-1",
            payload_hash="hash-b",
        )


def test_create_idempotency_record_is_conflict_tolerant_for_same_payload(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNAssetPacksRepository.initialized(chacha_db)

    first = repo.create_idempotency_record(
        owner_user_id=42,
        scope="vn_asset_export",
        resource_id="pack:1",
        idempotency_key="legacy-export-1",
        payload_hash="hash-a",
        response={"job_id": "job-1"},
    )
    second = repo.create_idempotency_record(
        owner_user_id=42,
        scope="vn_asset_export",
        resource_id="pack:1",
        idempotency_key="legacy-export-1",
        payload_hash="hash-a",
        response={"job_id": "job-1"},
    )

    assert second["id"] == first["id"]
    assert second["status"] == "completed"
    assert json.loads(second["response_json"]) == {"job_id": "job-1"}
    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.create_idempotency_record(
            owner_user_id=42,
            scope="vn_asset_export",
            resource_id="pack:1",
            idempotency_key="legacy-export-1",
            payload_hash="hash-b",
            response={"job_id": "job-2"},
        )


def test_ensure_vn_asset_tables_preserves_outer_transaction_rollback(chacha_db: CharactersRAGDB) -> None:
    character_name = "Rolled Back Before VN Schema"

    with pytest.raises(RuntimeError, match="force rollback"):
        with chacha_db.transaction() as conn:
            conn.execute(
                "INSERT INTO character_cards (name, client_id, version) VALUES (?, ?, 1)",
                (character_name, chacha_db.client_id),
            )
            ensure_vn_asset_tables(chacha_db)
            raise RuntimeError("force rollback")

    cursor = chacha_db.execute_query(
        "SELECT id FROM character_cards WHERE name = ?",
        (character_name,),
    )
    assert cursor.fetchone() is None


def test_create_pack_requires_existing_primary_character(chacha_db: CharactersRAGDB) -> None:
    repo = VNAssetPacksRepository(chacha_db)

    with pytest.raises(ValueError, match="primary_character_not_found"):
        repo.create_pack(owner_user_id=1, primary_character_id=9999, title="Pack")


def test_create_pack_writes_minimum_row_and_json_defaults(chacha_db: CharactersRAGDB) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository(chacha_db)

    pack = repo.create_pack(owner_user_id=42, primary_character_id=character_id, title="Starter Pack")

    cursor = chacha_db.execute_query("SELECT * FROM vn_asset_packs WHERE id = ?", (pack["id"],))
    row = cursor.fetchone()
    assert row is not None
    assert row["owner_user_id"] == 42
    assert row["title"] == "Starter Pack"
    assert row["primary_character_id"] == character_id
    assert row["status"] == "draft"
    assert row["content_rating"] == "general"
    assert json.loads(row["source_world_book_ids_json"]) == []
    assert row["deleted"] == 0
    assert row["version"] == 1


def test_matrix_slot_creation_supports_multi_hop_dependencies(chacha_db: CharactersRAGDB) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=42, primary_character_id=character_id, title="Starter Pack")

    slots = repo.create_slots_for_matrix(
        pack_id=pack["id"],
        slot_specs=[
            {
                "asset_type": "background",
                "slot_key": "background.interior",
                "variant_count": 1,
            },
            {
                "asset_type": "depth_companion",
                "slot_key": "depth.interior",
                "variant_count": 0,
                "depends_on_slot_key": "background.interior",
            },
            {
                "asset_type": "trim_mask",
                "slot_key": "trim.depth.interior",
                "variant_count": 0,
                "depends_on_slot_key": "depth.interior",
            },
        ],
    )

    slots_by_key = {slot["slot_key"]: slot for slot in slots}
    assert slots_by_key["depth.interior"]["depends_on_slot_id"] == slots_by_key["background.interior"]["id"]
    assert slots_by_key["trim.depth.interior"]["depends_on_slot_id"] == slots_by_key["depth.interior"]["id"]


def test_list_packs_for_setup_applies_owner_query_and_bounded_pagination(
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    repo.create_pack(
        owner_user_id=42,
        primary_character_id=character_id,
        title="Archive Alpha",
        description="Station archive pack.",
    )
    repo.create_pack(
        owner_user_id=42,
        primary_character_id=character_id,
        title="Archive Beta",
        description="Secondary archive pack.",
    )
    repo.create_pack(
        owner_user_id=7,
        primary_character_id=character_id,
        title="Archive Other Owner",
    )

    rows, has_more = repo.list_packs_for_setup(
        owner_user_id=42,
        query="archive",
        limit=1,
        offset=0,
    )

    assert [row["title"] for row in rows] == ["Archive Alpha"]
    assert has_more is True

    next_rows, next_has_more = repo.list_packs_for_setup(
        owner_user_id=42,
        query="archive",
        limit=1,
        offset=1,
    )

    assert [row["title"] for row in next_rows] == ["Archive Beta"]
    assert next_has_more is False


def test_latest_completed_import_provenance_by_pack_ids_uses_latest_completed_row(
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(
        owner_user_id=42,
        primary_character_id=character_id,
        title="Imported Pack",
    )
    other_pack = repo.create_pack(
        owner_user_id=7,
        primary_character_id=character_id,
        title="Other Owner Pack",
    )
    preview = repo.create_import_preview(
        owner_user_id=42,
        job_id="preview-job",
        status="completed",
        archive_path="test-artifacts/preview.vnpack",
    )
    repo.create_import_journal(
        owner_user_id=42,
        preview_id=int(preview["id"]),
        job_id="older-completed",
        status="completed",
        stage="completed",
        trust_mode="trusted_restore",
        target_mode="create_new",
        target_pack_id=int(pack["id"]),
        completed_at="2026-05-08T00:00:00Z",
    )
    repo.create_import_journal(
        owner_user_id=42,
        preview_id=int(preview["id"]),
        job_id="failed-newer",
        status="failed",
        stage="failed",
        trust_mode="trusted_restore",
        target_mode="create_new",
        target_pack_id=int(pack["id"]),
        completed_at="2026-05-10T00:00:00Z",
    )
    repo.create_import_journal(
        owner_user_id=42,
        preview_id=int(preview["id"]),
        job_id="newer-completed",
        status="completed",
        stage="completed",
        trust_mode="untrusted_import",
        target_mode="create_new",
        target_pack_id=int(pack["id"]),
        completed_at="2026-05-09T00:00:00Z",
    )
    other_preview = repo.create_import_preview(
        owner_user_id=7,
        job_id="other-preview",
        status="completed",
        archive_path="test-artifacts/other.vnpack",
    )
    repo.create_import_journal(
        owner_user_id=7,
        preview_id=int(other_preview["id"]),
        job_id="other-owner",
        status="completed",
        stage="completed",
        trust_mode="trusted_restore",
        target_mode="create_new",
        target_pack_id=int(other_pack["id"]),
        completed_at="2026-05-10T00:00:00Z",
    )

    provenance = repo.latest_completed_import_provenance_by_pack_ids(
        owner_user_id=42,
        pack_ids=[int(pack["id"]), int(other_pack["id"])],
    )

    assert set(provenance) == {int(pack["id"])}
    assert provenance[int(pack["id"])]["job_id"] == "newer-completed"
    assert provenance[int(pack["id"])]["trust_mode"] == "untrusted_import"
