import json
from collections.abc import Generator

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="vn-portability-test-client")
    yield database
    database.close_connection()


def test_portability_tables_are_created(chacha_db: CharactersRAGDB) -> None:
    VNAssetPacksRepository.initialized(chacha_db)

    table_names = {
        row["name"]
        for row in chacha_db.execute_query(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }

    assert "vn_pack_portability_jobs" in table_names
    assert "vn_pack_import_previews" in table_names
    assert "vn_pack_import_journal" in table_names


def test_portability_job_round_trips(chacha_db: CharactersRAGDB) -> None:
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "VN Portability Primary"})
    pack = repo.create_pack(
        owner_user_id=7,
        primary_character_id=character_id,
        title="Portable Pack",
    )

    created = repo.create_portability_job(
        owner_user_id=7,
        job_id="job-export-1",
        operation="export",
        status="queued",
        stage="queued",
        pack_id=pack["id"],
        progress={"current": 0, "total": 3},
        warnings=["pending-image"],
    )

    loaded = repo.get_portability_job(created["id"], owner_user_id=7)
    assert loaded is not None
    assert loaded["job_id"] == "job-export-1"
    assert loaded["operation"] == "export"
    assert json.loads(loaded["progress_json"]) == {"current": 0, "total": 3}
    assert json.loads(loaded["warnings_json"]) == ["pending-image"]
    assert repo.get_portability_job(created["id"], owner_user_id=99) is None

    updated = repo.update_portability_job(
        "job-export-1",
        {
            "status": "processing",
            "stage": "assembling",
            "progress": {"current": 1, "total": 3},
        },
        owner_user_id=7,
    )

    assert updated is not None
    assert updated["status"] == "processing"
    assert updated["stage"] == "assembling"
    assert json.loads(updated["progress_json"]) == {"current": 1, "total": 3}
    assert json.loads(updated["warnings_json"]) == ["pending-image"]
    assert repo.get_portability_job_by_job_id("job-export-1", owner_user_id=7)["id"] == created["id"]
    assert repo.get_portability_job_by_job_id("job-export-1", owner_user_id=99) is None


def test_import_preview_round_trips(chacha_db: CharactersRAGDB) -> None:
    repo = VNAssetPacksRepository.initialized(chacha_db)

    created = repo.create_import_preview(
        owner_user_id=7,
        job_id="job-preview-1",
        status="ready",
        archive_path="/tmp/incoming.tldw-vnpack",
        archive_sha256="archive-hash",
        canonical_payload_fingerprint="payload-fingerprint",
        schema_version="tldw.vnpack.v1",
        bundle_summary={"title": "Starter Pack"},
        validation_warnings=["redacted-provenance"],
        conflicts=[{"kind": "pack_title"}],
        proposed_plan={"mode": "create_new"},
        quota_estimate={"bytes": 123},
        required_choices=["target_mode"],
    )

    loaded = repo.get_import_preview(created["id"], owner_user_id=7)
    assert loaded is not None
    assert loaded["job_id"] == "job-preview-1"
    assert loaded["archive_sha256"] == "archive-hash"
    assert json.loads(loaded["bundle_summary_json"]) == {"title": "Starter Pack"}
    assert json.loads(loaded["validation_warnings_json"]) == ["redacted-provenance"]
    assert json.loads(loaded["conflicts_json"]) == [{"kind": "pack_title"}]
    assert json.loads(loaded["proposed_plan_json"]) == {"mode": "create_new"}
    assert json.loads(loaded["quota_estimate_json"]) == {"bytes": 123}
    assert json.loads(loaded["required_choices_json"]) == ["target_mode"]
    assert repo.get_import_preview(created["id"], owner_user_id=99) is None

    updated = repo.update_import_preview(
        created["id"],
        {"status": "expired", "validation_warnings": ["expired"]},
        owner_user_id=7,
    )

    assert updated is not None
    assert updated["status"] == "expired"
    assert json.loads(updated["bundle_summary_json"]) == {"title": "Starter Pack"}
    assert json.loads(updated["validation_warnings_json"]) == ["expired"]


def test_import_journal_round_trips(chacha_db: CharactersRAGDB) -> None:
    repo = VNAssetPacksRepository.initialized(chacha_db)
    preview = repo.create_import_preview(
        owner_user_id=7,
        job_id="job-preview-2",
        status="ready",
        archive_path="/tmp/import.tldw-vnpack",
    )

    created = repo.create_import_journal(
        owner_user_id=7,
        preview_id=preview["id"],
        job_id="job-import-1",
        status="processing",
        stage="copying_assets",
        trust_mode="untrusted",
        target_mode="create_new",
        archive_path="/tmp/import.tldw-vnpack",
        archive_sha256="archive-hash",
        canonical_payload_fingerprint="payload-fingerprint",
        id_maps={"slots": {"1": 10}},
        created_records={"packs": [42]},
        cleanup_status={"generated_files": "pending"},
        warnings=["missing-optional-readme"],
    )

    loaded = repo.get_import_journal(created["id"], owner_user_id=7)
    assert loaded is not None
    assert loaded["job_id"] == "job-import-1"
    assert loaded["stage"] == "copying_assets"
    assert loaded["archive_path"] == "/tmp/import.tldw-vnpack"
    assert json.loads(loaded["id_maps_json"]) == {"slots": {"1": 10}}
    assert json.loads(loaded["created_records_json"]) == {"packs": [42]}
    assert json.loads(loaded["cleanup_status_json"]) == {"generated_files": "pending"}
    assert json.loads(loaded["warnings_json"]) == ["missing-optional-readme"]
    assert repo.get_import_journal(created["id"], owner_user_id=99) is None

    updated = repo.update_import_journal(
        created["id"],
        {
            "status": "completed",
            "stage": "done",
            "archive_path": "/tmp/import-restored.tldw-vnpack",
            "warnings": [],
        },
        owner_user_id=7,
    )

    assert updated is not None
    assert updated["status"] == "completed"
    assert updated["stage"] == "done"
    assert updated["archive_path"] == "/tmp/import-restored.tldw-vnpack"
    assert updated["archive_sha256"] == "archive-hash"
    assert json.loads(updated["created_records_json"]) == {"packs": [42]}
    assert json.loads(updated["warnings_json"]) == []
