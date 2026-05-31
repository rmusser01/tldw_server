from __future__ import annotations

import json
import zipfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.VN_Assets.jobs import (
    VN_PACK_EXPORT_JOB_TYPE,
    build_pack_export_payload,
    create_pack_export_job,
    pack_export_idempotency_key,
)
from tldw_Server_API.app.core.VN_Assets.portability.constants import (
    ASSET_BYTES_STATUS_MISSING,
    ASSET_BYTES_STATUS_PRESENT,
    CHECKSUMS_PATH,
    MANIFEST_PATH,
    REQUIRED_MEMBERS,
    VNPACK_SCHEMA_VERSION,
)
from tldw_Server_API.app.core.VN_Assets.portability import exporter as portability_exporter
from tldw_Server_API.app.core.VN_Assets.portability.exporter import VNPackExporter
from tldw_Server_API.app.core.VN_Assets.portability.fingerprints import sha256_file
from tldw_Server_API.app.core.VN_Assets.portability.models import VNPackExportOptions

USER_ID = 42
PNG_BYTES = b"\x89PNG\r\n\x1a\nexport-test-png"
PRIVATE_PROMPT = "private prompt text that must not be exported by default"


class FakeGeneratedFilesRepo:
    def __init__(self, records: dict[int, dict[str, Any]]) -> None:
        self.records = records
        self.requested_ids: list[int] = []

    async def get_file_by_id(self, file_id: int) -> dict[str, Any] | None:
        self.requested_ids.append(file_id)
        return self.records.get(file_id)


class FakeJobs:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        job = {"id": len(self.created) + 1, "status": "queued", **kwargs}
        self.created.append(job)
        return job


@pytest.fixture
def chacha_db(tmp_path: Path) -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="vn-export-test")
    yield database
    database.close_connection()


@pytest.fixture
def repo(chacha_db: CharactersRAGDB) -> VNAssetPacksRepository:
    return VNAssetPacksRepository.initialized(chacha_db)


@pytest.fixture
def character_id(chacha_db: CharactersRAGDB) -> int:
    return chacha_db.add_character_card(
        {
            "name": "Mira",
            "description": "A careful archivist.",
            "personality": "Patient and exacting.",
            "scenario": "Cataloging an orbital library.",
        }
    )


@pytest.fixture
def pack_with_export_items(
    repo: VNAssetPacksRepository,
    character_id: int,
) -> dict[str, Any]:
    pack = repo.create_pack(
        owner_user_id=USER_ID,
        primary_character_id=character_id,
        title="Portable Pack",
        content_rating="general",
        scenario_notes="A quiet archive.",
        style_prompt="clean anime style",
    )
    slot = repo.create_slot(
        pack_id=int(pack["id"]),
        asset_type="sprite",
        slot_key="sprite.primary.neutral",
        labels={"expression": "neutral", "pose": "front"},
        variant_count=2,
    )
    present_item = repo.create_item(
        pack_id=int(pack["id"]),
        slot_id=int(slot["id"]),
        variant_index=0,
        generated_file_id=700,
        storage_ref="vn_assets/present.png",
        mime_type="image/png",
        width=512,
        height=768,
        bytes=len(PNG_BYTES),
        review_status="approved",
        preferred=True,
        source_prompt_snapshot={
            "prompt": PRIVATE_PROMPT,
            "negative_prompt": "private negative prompt",
            "token_estimates": {"prompt": 7},
        },
        source_context_snapshot={
            "secret_context": "private world-book excerpt",
        },
        backend_metadata={
            "backend": "comfyui",
            "model": "test-model",
            "seed": 123,
            "steps": 20,
        },
    )
    missing_item = repo.create_item(
        pack_id=int(pack["id"]),
        slot_id=int(slot["id"]),
        variant_index=1,
        generated_file_id=701,
        storage_ref="vn_assets/missing.png",
        mime_type="image/png",
        width=512,
        height=768,
        bytes=1024,
        review_status="draft",
        source_prompt_snapshot={"prompt": PRIVATE_PROMPT},
    )
    return {"pack": pack, "slot": slot, "present_item": present_item, "missing_item": missing_item}


@pytest.mark.asyncio
async def test_export_pack_writes_backup_archive_with_redacted_missing_assets(
    repo: VNAssetPacksRepository,
    pack_with_export_items: dict[str, Any],
    tmp_path: Path,
) -> None:
    present_item = pack_with_export_items["present_item"]
    missing_item = pack_with_export_items["missing_item"]
    files_repo = FakeGeneratedFilesRepo(
        {
            700: {
                "id": 700,
                "user_id": USER_ID,
                "filename": "present.png",
                "original_filename": "present.png",
                "storage_path": "vn_assets/present.png",
                "mime_type": "image/png",
                "source_feature": "vn_assets",
                "source_ref": f"vn_asset_item:{present_item['id']}",
            }
        }
    )

    def read_generated_file_bytes(record: dict[str, Any]) -> bytes:
        assert record["id"] == 700
        return PNG_BYTES

    exporter = VNPackExporter(
        repo=repo,
        owner_user_id=USER_ID,
        generated_files_repo=files_repo,
        read_generated_file_bytes=read_generated_file_bytes,
        staging_root=tmp_path / "exports",
    )

    result = await exporter.export_pack(
        pack_id=int(pack_with_export_items["pack"]["id"]),
        options=VNPackExportOptions(),
    )

    assert result.archive_sha256 == sha256_file(result.archive_path)
    assert result.file_size_bytes == result.archive_path.stat().st_size
    assert any("missing" in warning for warning in result.warnings)

    with zipfile.ZipFile(result.archive_path) as archive:
        names = set(archive.namelist())
        assert REQUIRED_MEMBERS <= names
        assert "metadata/provenance.json" in names

        manifest = json.loads(archive.read(MANIFEST_PATH))
        assert manifest["schema_version"] == VNPACK_SCHEMA_VERSION
        assert manifest["archive_profile"] == "backup"
        assert manifest["counts"]["items"] == 2
        assert manifest["counts"]["assets_with_bytes"] == 1
        assert "archive_sha256" not in manifest

        items = json.loads(archive.read("metadata/items.json"))["items"]
        assert len(items) == 2
        present_export = next(item for item in items if item["source_item_id"] == present_item["id"])
        missing_export = next(item for item in items if item["source_item_id"] == missing_item["id"])
        assert present_export["asset_bytes_status"] == ASSET_BYTES_STATUS_PRESENT
        assert present_export["asset_path"].startswith("assets/items/")
        assert present_export["asset_sha256"]
        assert missing_export["asset_bytes_status"] == ASSET_BYTES_STATUS_MISSING
        assert "asset_path" not in missing_export

        checksums = json.loads(archive.read(CHECKSUMS_PATH))
        assert "metadata/pack.json" in checksums
        assert "metadata/slots.json" in checksums
        assert "metadata/items.json" in checksums
        assert "metadata/provenance.json" in checksums
        assert present_export["asset_path"] in checksums
        assert all(path in checksums or path == CHECKSUMS_PATH for path in REQUIRED_MEMBERS)

        provenance = json.loads(archive.read("metadata/provenance.json"))
        serialized_provenance = json.dumps(provenance)
        assert PRIVATE_PROMPT not in serialized_provenance
        present_provenance = next(
            item for item in provenance["items"] if item["source_item_id"] == present_item["id"]
        )
        assert present_provenance["prompt"]["prompt_present"] is True
        assert present_provenance["prompt"]["prompt_sha256"]


@pytest.mark.asyncio
async def test_export_pack_writes_asset_bytes_without_final_payload_accumulation(
    repo: VNAssetPacksRepository,
    pack_with_export_items: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    present_item = pack_with_export_items["present_item"]
    files_repo = FakeGeneratedFilesRepo(
        {
            700: {
                "id": 700,
                "user_id": USER_ID,
                "filename": "present.png",
                "original_filename": "present.png",
                "storage_path": "vn_assets/present.png",
                "mime_type": "image/png",
                "source_feature": "vn_assets",
                "source_ref": f"vn_asset_item:{present_item['id']}",
            }
        }
    )
    exporter = VNPackExporter(
        repo=repo,
        owner_user_id=USER_ID,
        generated_files_repo=files_repo,
        read_generated_file_bytes=lambda _record: PNG_BYTES,
        staging_root=tmp_path / "exports",
    )
    payload_paths_written_at_end: list[str] = []
    original_write_payloads = portability_exporter._write_payloads_to_archive

    def spy_write_payloads(archive: zipfile.ZipFile, payloads: dict[str, bytes]) -> None:
        payload_paths_written_at_end.extend(payloads)
        original_write_payloads(archive, payloads)

    monkeypatch.setattr(portability_exporter, "_write_payloads_to_archive", spy_write_payloads)

    result = await exporter.export_pack(
        pack_id=int(pack_with_export_items["pack"]["id"]),
        options=VNPackExportOptions(),
    )

    assert all(not path.startswith("assets/") for path in payload_paths_written_at_end)
    with zipfile.ZipFile(result.archive_path) as archive:
        items = json.loads(archive.read("metadata/items.json"))["items"]
        present_export = next(item for item in items if item["source_item_id"] == present_item["id"])
        assert archive.read(present_export["asset_path"]) == PNG_BYTES


@pytest.mark.asyncio
async def test_export_pack_includes_character_payload_and_semantic_item_keys(
    repo: VNAssetPacksRepository,
    pack_with_export_items: dict[str, Any],
    tmp_path: Path,
) -> None:
    present_item = pack_with_export_items["present_item"]
    files_repo = FakeGeneratedFilesRepo(
        {
            700: {
                "id": 700,
                "user_id": USER_ID,
                "filename": "present.png",
                "original_filename": "present.png",
                "storage_path": "vn_assets/present.png",
                "mime_type": "image/png",
                "source_feature": "vn_assets",
                "source_ref": f"vn_asset_item:{present_item['id']}",
            }
        }
    )
    exporter = VNPackExporter(
        repo=repo,
        owner_user_id=USER_ID,
        generated_files_repo=files_repo,
        read_generated_file_bytes=lambda _record: PNG_BYTES,
        staging_root=tmp_path / "exports",
    )

    result = await exporter.export_pack(
        pack_id=int(pack_with_export_items["pack"]["id"]),
        options=VNPackExportOptions(include_character_payload=True),
    )

    with zipfile.ZipFile(result.archive_path) as archive:
        manifest = json.loads(archive.read(MANIFEST_PATH))
        checksums = json.loads(archive.read(CHECKSUMS_PATH))
        items = json.loads(archive.read("metadata/items.json"))["items"]
        character = json.loads(archive.read("metadata/character.json"))["character"]

    assert manifest["include_character"] is True
    assert "metadata/character.json" in checksums
    assert character["name"] == "Mira"
    assert all(item["asset_type"] == "sprite" for item in items)
    assert all(item["slot_key"] == "sprite.primary.neutral" for item in items)


@pytest.mark.asyncio
async def test_world_book_payload_export_fails_when_referenced_payloads_are_not_available(
    repo: VNAssetPacksRepository,
    character_id: int,
    tmp_path: Path,
) -> None:
    pack = repo.create_pack(
        owner_user_id=USER_ID,
        primary_character_id=character_id,
        title="World Book Pack",
        source_world_book_ids=[101],
    )
    exporter = VNPackExporter(
        repo=repo,
        owner_user_id=USER_ID,
        generated_files_repo=FakeGeneratedFilesRepo({}),
        read_generated_file_bytes=lambda _record: PNG_BYTES,
        staging_root=tmp_path / "exports",
    )

    with pytest.raises(ValueError, match="world_book_payloads_unavailable"):
        await exporter.export_pack(
            pack_id=int(pack["id"]),
            options=VNPackExportOptions(include_world_book_payloads=True),
        )


@pytest.mark.asyncio
async def test_canonical_payload_fingerprint_ignores_local_ids(
    repo: VNAssetPacksRepository,
    character_id: int,
    tmp_path: Path,
) -> None:
    async def export_clone(index: int) -> str:
        pack = repo.create_pack(
            owner_user_id=USER_ID,
            primary_character_id=character_id,
            title="Clone Pack",
            content_rating="general",
            source_world_book_ids=[100 + index],
            scenario_notes="Same scene.",
            style_prompt="same style",
        )
        slot = repo.create_slot(
            pack_id=int(pack["id"]),
            asset_type="sprite",
            slot_key="sprite.primary.neutral",
            labels={"expression": "neutral"},
            variant_count=1,
        )
        item = repo.create_item(
            pack_id=int(pack["id"]),
            slot_id=int(slot["id"]),
            variant_index=0,
            generated_file_id=800 + index,
            storage_ref=f"vn_assets/clone-{index}.png",
            mime_type="image/png",
            width=512,
            height=768,
            bytes=len(PNG_BYTES),
            review_status="approved",
            preferred=True,
            source_prompt_snapshot={"prompt": "same prompt"},
            source_context_snapshot={
                "stable_context": "same lore",
                "source_pack_id": int(pack["id"]),
                "source_slot_id": int(slot["id"]),
            },
            backend_metadata={"backend": "comfyui", "model": "same-model", "seed": 123},
        )
        files_repo = FakeGeneratedFilesRepo(
            {
                800 + index: {
                    "id": 800 + index,
                    "user_id": USER_ID,
                    "filename": f"clone-{index}.png",
                    "storage_path": f"vn_assets/clone-{index}.png",
                    "mime_type": "image/png",
                    "source_feature": "vn_assets",
                    "source_ref": f"vn_asset_item:{item['id']}",
                }
            }
        )
        exporter = VNPackExporter(
            repo=repo,
            owner_user_id=USER_ID,
            generated_files_repo=files_repo,
            read_generated_file_bytes=lambda _record: PNG_BYTES,
            staging_root=tmp_path / f"exports-{index}",
        )
        result = await exporter.export_pack(pack_id=int(pack["id"]), options=VNPackExportOptions())
        return result.canonical_payload_fingerprint

    assert await export_clone(1) == await export_clone(2)


@pytest.mark.asyncio
async def test_strict_export_fails_when_item_bytes_are_missing(
    repo: VNAssetPacksRepository,
    pack_with_export_items: dict[str, Any],
    tmp_path: Path,
) -> None:
    exporter = VNPackExporter(
        repo=repo,
        owner_user_id=USER_ID,
        generated_files_repo=FakeGeneratedFilesRepo({}),
        read_generated_file_bytes=lambda _record: PNG_BYTES,
        staging_root=tmp_path / "exports",
    )

    with pytest.raises(ValueError, match="missing_asset_bytes"):
        await exporter.export_pack(
            pack_id=int(pack_with_export_items["pack"]["id"]),
            options=VNPackExportOptions(strict=True),
        )


def test_pack_export_job_helpers_use_portability_group_and_options() -> None:
    jobs = FakeJobs()
    options = {"strict": False, "include_full_provenance": False}

    payload = build_pack_export_payload(
        pack_id=5,
        portability_job_id=9,
        request_id="req-1",
        user_id=USER_ID,
        options=options,
    )
    idempotency_key = pack_export_idempotency_key(
        user_id=USER_ID,
        pack_id=5,
        request_id="req-1",
        options=options,
    )
    job = create_pack_export_job(
        jobs,
        pack_id=5,
        portability_job_id=9,
        request_id="req-1",
        user_id=USER_ID,
        options=options,
    )

    assert payload == {
        "pack_id": 5,
        "portability_job_id": 9,
        "request_id": "req-1",
        "user_id": USER_ID,
        "options": options,
    }
    assert idempotency_key.startswith("vn_assets:user:42:pack:5:portability:export:req-1:")
    assert job["job_type"] == VN_PACK_EXPORT_JOB_TYPE
    assert job["batch_group"] == "vn_assets:user:42:pack:5:portability:export:req-1"
    assert job["idempotency_key"] == idempotency_key


@pytest.mark.asyncio
async def test_export_worker_updates_portability_job_row(
    repo: VNAssetPacksRepository,
    pack_with_export_items: dict[str, Any],
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    pack_id = int(pack_with_export_items["pack"]["id"])
    portability_job = repo.create_portability_job(
        owner_user_id=USER_ID,
        job_id="job-export-1",
        operation="export",
        status="queued",
        stage="queued",
        pack_id=pack_id,
    )
    files_repo = FakeGeneratedFilesRepo(
        {
            700: {
                "id": 700,
                "user_id": USER_ID,
                "filename": "present.png",
                "storage_path": "vn_assets/present.png",
                "mime_type": "image/png",
                "source_feature": "vn_assets",
                "source_ref": f"vn_asset_item:{pack_with_export_items['present_item']['id']}",
            }
        }
    )
    worker = VNAssetGenerationWorker(
        repo=repo,
        jobs_manager=FakeJobs(),
        generated_files_repo=files_repo,
        read_generated_file_bytes=lambda _record: PNG_BYTES,
        export_staging_root=tmp_path / "exports",
    )

    result = await worker.handle_job_async(
        {
            "id": "job-export-1",
            "job_type": VN_PACK_EXPORT_JOB_TYPE,
            "owner_user_id": str(USER_ID),
            "payload": {
                "pack_id": pack_id,
                "portability_job_id": portability_job["id"],
                "request_id": "req-worker",
                "user_id": USER_ID,
                "options": {"strict": False},
            },
        }
    )

    updated = repo.get_portability_job(portability_job["id"], owner_user_id=USER_ID)
    assert result["status"] == "exported"
    assert Path(result["archive_path"]).is_file()
    assert updated is not None
    assert updated["status"] == "completed"
    assert updated["stage"] == "completed"
    assert updated["archive_sha256"] == result["archive_sha256"]
    assert updated["canonical_payload_fingerprint"] == result["canonical_payload_fingerprint"]
    assert json.loads(updated["progress_json"])["file_size_bytes"] == result["file_size_bytes"]


@pytest.mark.asyncio
async def test_export_worker_resolves_api_created_portability_row_by_job_id(
    repo: VNAssetPacksRepository,
    pack_with_export_items: dict[str, Any],
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    pack_id = int(pack_with_export_items["pack"]["id"])
    portability_job = repo.create_portability_job(
        owner_user_id=USER_ID,
        job_id="123",
        operation="export",
        status="queued",
        stage="queued",
        pack_id=pack_id,
    )
    files_repo = FakeGeneratedFilesRepo(
        {
            700: {
                "id": 700,
                "user_id": USER_ID,
                "filename": "present.png",
                "storage_path": "vn_assets/present.png",
                "mime_type": "image/png",
                "source_feature": "vn_assets",
                "source_ref": f"vn_asset_item:{pack_with_export_items['present_item']['id']}",
            }
        }
    )
    worker = VNAssetGenerationWorker(
        repo=repo,
        jobs_manager=FakeJobs(),
        generated_files_repo=files_repo,
        read_generated_file_bytes=lambda _record: PNG_BYTES,
        export_staging_root=tmp_path / "exports",
    )

    result = await worker.handle_job_async(
        {
            "id": 123,
            "job_type": VN_PACK_EXPORT_JOB_TYPE,
            "owner_user_id": str(USER_ID),
            "payload": {
                "pack_id": pack_id,
                "portability_job_id": 0,
                "request_id": "req-api",
                "user_id": USER_ID,
                "options": {"strict": False},
            },
        }
    )

    updated = repo.get_portability_job(portability_job["id"], owner_user_id=USER_ID)
    assert result["portability_job_id"] == portability_job["id"]
    assert updated is not None
    assert updated["status"] == "completed"
