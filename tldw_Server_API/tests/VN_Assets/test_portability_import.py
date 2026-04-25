from __future__ import annotations

import json
import zipfile
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.VN_Assets.jobs import (
    VN_PACK_IMPORT_COMMIT_JOB_TYPE,
    build_pack_import_commit_payload,
    create_pack_import_commit_job,
    pack_import_commit_idempotency_key,
)
from tldw_Server_API.app.core.VN_Assets.portability.constants import (
    ASSET_BYTES_STATUS_MISSING,
    ASSET_BYTES_STATUS_PRESENT,
    CHECKSUMS_PATH,
    MANIFEST_PATH,
    TRUST_MODE_TRUSTED_RESTORE,
    TRUST_MODE_UNTRUSTED_IMPORT,
    VNPACK_SCHEMA_VERSION,
)
from tldw_Server_API.app.core.VN_Assets.portability.fingerprints import (
    canonical_json_bytes,
    canonical_payload_fingerprint,
    sha256_bytes,
    sha256_file,
)
from tldw_Server_API.app.core.VN_Assets.portability.importer import VNPackImporter
from tldw_Server_API.app.core.VN_Assets.portability.preview import VNPackImportPreviewer

USER_ID = 42
PNG_BYTES = b"\x89PNG\r\n\x1a\nimport-test-png"


class FakeGeneratedFileSaver:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []
        self.fail_on_call = fail_on_call

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        call_number = len(self.calls)
        if self.fail_on_call == call_number:
            raise RuntimeError("save_failed")
        record = {
            "id": 9000 + call_number,
            "user_id": kwargs["user_id"],
            "storage_path": f"vn_assets/imported-{kwargs['item_id']}.png",
            "mime_type": "image/png",
            "source_feature": "vn_assets",
            "source_ref": f"vn_asset_item:{kwargs['item_id']}",
            "file_size_bytes": len(kwargs["image_bytes"]),
        }
        self.records.append(record)
        return record


class FakeJobs:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        job = {"id": len(self.created) + 1, "status": "queued", **kwargs}
        self.created.append(job)
        return job


@pytest.fixture
def chacha_db(tmp_path: Path) -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="vn-import-test")
    yield database
    database.close_connection()


@pytest.fixture
def repo(chacha_db: CharactersRAGDB) -> VNAssetPacksRepository:
    return VNAssetPacksRepository.initialized(chacha_db)


@pytest.fixture
def target_character_id(chacha_db: CharactersRAGDB) -> int:
    character_id = chacha_db.add_character_card(
        {
            "name": "Local Mira",
            "description": "Local character selected for import.",
            "personality": "Careful.",
            "scenario": "Restoring an archive.",
        }
    )
    assert character_id is not None
    return int(character_id)


def _pack_payload() -> dict[str, Any]:
    return {
        "pack": {
            "source_pack_id": 100,
            "title": "Imported Backup",
            "description": "A portable backup pack.",
            "status": "draft",
            "content_rating": "general",
            "primary_character_id": 200,
            "source_world_book_ids": [],
            "scenario_notes": "A restored scene.",
            "style_prompt": "clean style",
            "negative_prompt": "low quality",
            "default_backend": "comfyui",
            "default_model": "test-model",
            "default_dimensions": {"width": 128, "height": 256, "format": "png"},
            "style_lock": {"palette": "soft"},
            "generation_budget": {"max_items": 2},
        }
    }


def _slot_payload() -> dict[str, Any]:
    return {
        "slots": [
            {
                "source_slot_id": 300,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.neutral",
                "labels": {"expression": "neutral"},
                "prompt_template": "neutral pose",
                "negative_prompt_template": "blur",
                "variant_count": 2,
                "width": 128,
                "height": 256,
                "requires_review": True,
                "required_for_runtime": True,
                "status": "approved",
            }
        ]
    }


def _item_payload(*, two_present: bool = False) -> dict[str, Any]:
    items = [
        {
            "source_item_id": 400,
            "source_slot_id": 300,
            "asset_type": "sprite",
            "slot_key": "sprite.primary.neutral",
            "variant_index": 0,
            "mime_type": "image/png",
            "width": 128,
            "height": 256,
            "bytes": len(PNG_BYTES),
            "review_status": "approved",
            "preferred": True,
            "source": "generated",
            "asset_bytes_status": ASSET_BYTES_STATUS_PRESENT,
            "asset_path": "assets/items/neutral-0.png",
            "asset_sha256": sha256_bytes(PNG_BYTES),
            "asset_size_bytes": len(PNG_BYTES),
        },
        {
            "source_item_id": 401,
            "source_slot_id": 300,
            "asset_type": "sprite",
            "slot_key": "sprite.primary.neutral",
            "variant_index": 1,
            "mime_type": "image/png",
            "width": 128,
            "height": 256,
            "bytes": 2048,
            "review_status": "approved",
            "preferred": False,
            "source": "generated",
            "asset_bytes_status": ASSET_BYTES_STATUS_MISSING,
        },
    ]
    if two_present:
        second = dict(items[1])
        second["asset_bytes_status"] = ASSET_BYTES_STATUS_PRESENT
        second["asset_path"] = "assets/items/neutral-1.png"
        second["asset_sha256"] = sha256_bytes(PNG_BYTES + b"-two")
        second["asset_size_bytes"] = len(PNG_BYTES + b"-two")
        items[1] = second
    return {"items": items}


def _write_import_archive(archive_path: Path, *, two_present: bool = False) -> Path:
    pack = _pack_payload()
    slots = _slot_payload()
    items = _item_payload(two_present=two_present)
    return _write_import_archive_from_payloads(
        archive_path,
        pack=pack,
        slots=slots,
        items=items,
        asset_payloads={
            str(item["asset_path"]): PNG_BYTES if item["variant_index"] == 0 else PNG_BYTES + b"-two"
            for item in items["items"]
            if item["asset_bytes_status"] == ASSET_BYTES_STATUS_PRESENT
        },
    )


def _write_import_archive_from_payloads(
    archive_path: Path,
    *,
    pack: dict[str, Any],
    slots: dict[str, Any],
    items: dict[str, Any],
    asset_payloads: dict[str, bytes] | None = None,
) -> Path:
    payloads = {
        "metadata/pack.json": canonical_json_bytes(pack),
        "metadata/slots.json": canonical_json_bytes(slots),
        "metadata/items.json": canonical_json_bytes(items),
    }
    payloads.update(asset_payloads or {})

    checksums = {path: sha256_bytes(content) for path, content in sorted(payloads.items())}
    manifest = {
        "schema_version": VNPACK_SCHEMA_VERSION,
        "archive_profile": "backup",
        "pack_title": pack["pack"]["title"],
        "content_rating": "general",
        "source_pack_fingerprint": "source-pack-fingerprint",
        "canonical_payload_fingerprint": canonical_payload_fingerprint(
            {"pack": pack["pack"], "slots": slots["slots"], "items": items["items"]}
        ),
        "counts": {
            "slots": len(slots["slots"]),
            "items": len(items["items"]),
            "assets_with_bytes": sum(
                1
                for item in items["items"]
                if item.get("asset_bytes_status") == ASSET_BYTES_STATUS_PRESENT
            ),
        },
        "include_images": True,
        "include_character": False,
        "include_world_books": False,
        "provenance_mode": "redacted",
        "encryption": {"encrypted": False, "scheme": None},
        "sections": [{"path": path, "sha256": digest} for path, digest in sorted(checksums.items())],
        "warnings": [],
    }
    payloads[MANIFEST_PATH] = canonical_json_bytes(manifest)
    checksums[MANIFEST_PATH] = sha256_bytes(payloads[MANIFEST_PATH])
    payloads[CHECKSUMS_PATH] = canonical_json_bytes(checksums)

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in sorted(payloads.items()):
            archive.writestr(path, content)
    return archive_path


async def _create_completed_preview(
    repo: VNAssetPacksRepository,
    archive_path: Path,
    *,
    expires_at: str | None = None,
    archive_sha256: str | None = None,
) -> dict[str, Any]:
    preview_result = await VNPackImportPreviewer(repo=repo).create_preview(
        archive_path=archive_path,
        owner_user_id=USER_ID,
    )
    return repo.create_import_preview(
        owner_user_id=USER_ID,
        job_id="preview-job-1",
        status="completed",
        archive_path=str(archive_path),
        archive_sha256=archive_sha256 or preview_result["archive_sha256"],
        canonical_payload_fingerprint=preview_result["canonical_payload_fingerprint"],
        schema_version=preview_result["schema_version"],
        bundle_summary=preview_result["bundle_summary"],
        validation_warnings=preview_result["validation_warnings"],
        conflicts=preview_result["conflicts"],
        proposed_plan=preview_result["proposed_plan"],
        quota_estimate=preview_result["quota_estimate"],
        required_choices=preview_result["required_choices"],
        expires_at=expires_at or (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    )


@pytest.mark.asyncio
async def test_import_commit_rejects_expired_preview(
    repo: VNAssetPacksRepository,
    target_character_id: int,
    tmp_path: Path,
) -> None:
    archive_path = _write_import_archive(tmp_path / "expired.tldw-vnpack")
    preview = await _create_completed_preview(
        repo,
        archive_path,
        expires_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
    )
    importer = VNPackImporter(
        repo=repo,
        owner_user_id=USER_ID,
        save_vn_asset_image=FakeGeneratedFileSaver(),
    )

    with pytest.raises(ValueError, match="import_preview_expired"):
        await importer.import_pack(
            preview_id=int(preview["id"]),
            job_id="job-import-expired",
            trust_mode=TRUST_MODE_TRUSTED_RESTORE,
            target_mode="create_new",
            character_action="link_existing_character",
            target_character_id=target_character_id,
        )


@pytest.mark.asyncio
async def test_import_commit_rejects_mutated_archive_checksum(
    repo: VNAssetPacksRepository,
    target_character_id: int,
    tmp_path: Path,
) -> None:
    archive_path = _write_import_archive(tmp_path / "mutated.tldw-vnpack")
    preview = await _create_completed_preview(repo, archive_path, archive_sha256="0" * 64)
    importer = VNPackImporter(
        repo=repo,
        owner_user_id=USER_ID,
        save_vn_asset_image=FakeGeneratedFileSaver(),
    )

    with pytest.raises(ValueError, match="import_archive_checksum_changed"):
        await importer.import_pack(
            preview_id=int(preview["id"]),
            job_id="job-import-mutated",
            trust_mode=TRUST_MODE_TRUSTED_RESTORE,
            target_mode="create_new",
            character_action="link_existing_character",
            target_character_id=target_character_id,
        )


@pytest.mark.asyncio
async def test_import_commit_preflights_total_asset_quota_before_creating_pack(
    repo: VNAssetPacksRepository,
    target_character_id: int,
    tmp_path: Path,
) -> None:
    archive_path = _write_import_archive(tmp_path / "quota.tldw-vnpack")
    preview = await _create_completed_preview(repo, archive_path)
    saver = FakeGeneratedFileSaver()
    quota_checks: list[int] = []

    async def preflight_quota(user_id: int, new_bytes: int, *, raise_on_exceed: bool = False) -> tuple[bool, dict[str, Any]]:
        quota_checks.append(new_bytes)
        raise RuntimeError("quota_failed")

    importer = VNPackImporter(
        repo=repo,
        owner_user_id=USER_ID,
        save_vn_asset_image=saver,
        preflight_storage_quota=preflight_quota,
    )

    with pytest.raises(RuntimeError, match="quota_failed"):
        await importer.import_pack(
            preview_id=int(preview["id"]),
            job_id="job-import-quota",
            trust_mode=TRUST_MODE_TRUSTED_RESTORE,
            target_mode="create_new",
            character_action="link_existing_character",
            target_character_id=target_character_id,
        )

    assert quota_checks == [len(PNG_BYTES)]
    assert saver.calls == []
    assert repo.list_packs(owner_user_id=USER_ID) == []


@pytest.mark.asyncio
async def test_create_new_trusted_restore_imports_pack_items_and_journal(
    repo: VNAssetPacksRepository,
    target_character_id: int,
    tmp_path: Path,
) -> None:
    archive_path = _write_import_archive(tmp_path / "trusted.tldw-vnpack")
    preview = await _create_completed_preview(repo, archive_path)
    saver = FakeGeneratedFileSaver()
    importer = VNPackImporter(repo=repo, owner_user_id=USER_ID, save_vn_asset_image=saver)

    result = await importer.import_pack(
        preview_id=int(preview["id"]),
        job_id="job-import-trusted",
        trust_mode=TRUST_MODE_TRUSTED_RESTORE,
        target_mode="create_new",
        character_action="link_existing_character",
        target_character_id=target_character_id,
    )

    new_pack = repo.get_pack(result["pack_id"])
    assert new_pack is not None
    assert new_pack["title"] == "Imported Backup"
    assert int(new_pack["primary_character_id"]) == target_character_id

    slots = repo.list_slots(result["pack_id"])
    assert [slot["slot_key"] for slot in slots] == ["sprite.primary.neutral"]

    items = repo.list_items(result["pack_id"])
    assert len(items) == 2
    present = next(item for item in items if item["variant_index"] == 0)
    missing = next(item for item in items if item["variant_index"] == 1)
    assert present["review_status"] == "approved"
    assert bool(present["preferred"]) is True
    assert present["generated_file_id"] == saver.records[0]["id"]
    assert saver.records[0]["user_id"] == USER_ID
    assert saver.records[0]["source_ref"] == f"vn_asset_item:{present['id']}"
    assert saver.calls[0]["pack_id"] == result["pack_id"]
    assert saver.calls[0]["item_id"] == present["id"]
    assert saver.calls[0]["check_quota"] is True
    assert missing["review_status"] == "hidden"
    assert bool(missing["preferred"]) is False
    assert missing["generated_file_id"] is None
    assert missing["storage_ref"] is None

    journal = repo.get_import_journal(result["import_id"], owner_user_id=USER_ID)
    assert journal is not None
    assert journal["status"] == "completed"
    id_maps = json.loads(journal["id_maps_json"])
    created_records = json.loads(journal["created_records_json"])
    assert id_maps["packs"]["100"] == result["pack_id"]
    assert id_maps["slots"]["300"] == slots[0]["id"]
    assert id_maps["items"]["400"] == present["id"]
    assert created_records["generated_file_ids"] == [saver.records[0]["id"]]


@pytest.mark.asyncio
async def test_untrusted_import_resets_byte_backed_items_to_draft(
    repo: VNAssetPacksRepository,
    target_character_id: int,
    tmp_path: Path,
) -> None:
    archive_path = _write_import_archive(tmp_path / "untrusted.tldw-vnpack")
    preview = await _create_completed_preview(repo, archive_path)
    importer = VNPackImporter(
        repo=repo,
        owner_user_id=USER_ID,
        save_vn_asset_image=FakeGeneratedFileSaver(),
    )

    result = await importer.import_pack(
        preview_id=int(preview["id"]),
        job_id="job-import-untrusted",
        trust_mode=TRUST_MODE_UNTRUSTED_IMPORT,
        target_mode="create_new",
        character_action="link_existing_character",
        target_character_id=target_character_id,
    )

    items = repo.list_items(result["pack_id"])
    present = next(item for item in items if item["variant_index"] == 0)
    missing = next(item for item in items if item["variant_index"] == 1)
    assert present["review_status"] == "draft"
    assert bool(present["preferred"]) is False
    assert missing["review_status"] == "hidden"


@pytest.mark.asyncio
async def test_import_failure_unregisters_created_generated_files(
    repo: VNAssetPacksRepository,
    target_character_id: int,
    tmp_path: Path,
) -> None:
    archive_path = _write_import_archive(tmp_path / "failure.tldw-vnpack", two_present=True)
    preview = await _create_completed_preview(repo, archive_path)
    saver = FakeGeneratedFileSaver(fail_on_call=2)
    unregistered: list[tuple[int, bool]] = []

    async def unregister(file_id: int, *, hard_delete: bool = False) -> bool:
        unregistered.append((file_id, hard_delete))
        return True

    importer = VNPackImporter(
        repo=repo,
        owner_user_id=USER_ID,
        save_vn_asset_image=saver,
        unregister_generated_file=unregister,
    )

    with pytest.raises(RuntimeError, match="save_failed"):
        await importer.import_pack(
            preview_id=int(preview["id"]),
            job_id="job-import-failure",
            trust_mode=TRUST_MODE_TRUSTED_RESTORE,
            target_mode="create_new",
            character_action="link_existing_character",
            target_character_id=target_character_id,
        )

    journal = repo.get_import_journal(1, owner_user_id=USER_ID)
    assert journal is not None
    assert journal["status"] == "failed"
    cleanup_status = json.loads(journal["cleanup_status_json"])
    assert unregistered == [(saver.records[0]["id"], True)]
    assert cleanup_status["unregistered_generated_file_ids"] == [saver.records[0]["id"]]


def test_pack_import_commit_job_helpers_use_import_group() -> None:
    jobs = FakeJobs()

    payload = build_pack_import_commit_payload(
        import_id=9,
        preview_id=5,
        request_id="req-1",
        user_id=USER_ID,
        trust_mode=TRUST_MODE_TRUSTED_RESTORE,
        target_mode="create_new",
        character_action="link_existing_character",
        target_character_id=12,
        target_pack_id=33,
    )
    idempotency_key = pack_import_commit_idempotency_key(
        user_id=USER_ID,
        preview_id=5,
        import_id=9,
        request_id="req-1",
    )
    job = create_pack_import_commit_job(
        jobs,
        import_id=9,
        preview_id=5,
        request_id="req-1",
        user_id=USER_ID,
        trust_mode=TRUST_MODE_TRUSTED_RESTORE,
        target_mode="create_new",
        character_action="link_existing_character",
        target_character_id=12,
        target_pack_id=33,
    )

    assert payload["import_id"] == 9
    assert payload["preview_id"] == 5
    assert payload["target_character_id"] == 12
    assert payload["target_pack_id"] == 33
    assert idempotency_key == "vn_assets:user:42:portability:import-commit:5:9:req-1"
    assert job["job_type"] == VN_PACK_IMPORT_COMMIT_JOB_TYPE
    assert job["batch_group"] == idempotency_key
    assert job["payload"]["target_pack_id"] == 33


@pytest.mark.asyncio
async def test_update_existing_non_destructive(
    repo: VNAssetPacksRepository,
    target_character_id: int,
    tmp_path: Path,
) -> None:
    existing_pack = repo.create_pack(
        owner_user_id=USER_ID,
        primary_character_id=target_character_id,
        title="Imported Backup",
    )
    existing_slot = repo.create_slot(
        pack_id=int(existing_pack["id"]),
        asset_type="sprite",
        slot_key="sprite.primary.neutral",
        labels={"expression": "neutral"},
        variant_count=1,
    )
    existing_item = repo.create_item(
        pack_id=int(existing_pack["id"]),
        slot_id=int(existing_slot["id"]),
        variant_index=0,
        generated_file_id=777,
        storage_ref="vn_assets/local-existing.png",
        review_status="approved",
        preferred=True,
        source_context_snapshot={"source_item_fingerprint": "local-item-fp"},
    )
    pack = _pack_payload()
    slots = {
        "slots": [
            {
                "source_slot_id": 300,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.neutral",
                "labels": {"expression": "neutral"},
                "variant_count": 1,
            },
            {
                "source_slot_id": 301,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.smile",
                "labels": {"expression": "smile"},
                "variant_count": 1,
            },
        ]
    }
    items = {
        "items": [
            {
                "source_item_id": 400,
                "source_slot_id": 300,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.neutral",
                "variant_index": 0,
                "review_status": "approved",
                "preferred": True,
                "source_item_fingerprint": "local-item-fp",
                "asset_bytes_status": ASSET_BYTES_STATUS_MISSING,
            },
            {
                "source_item_id": 401,
                "source_slot_id": 301,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.smile",
                "variant_index": 0,
                "mime_type": "image/png",
                "width": 128,
                "height": 256,
                "bytes": len(PNG_BYTES),
                "review_status": "approved",
                "preferred": True,
                "asset_bytes_status": ASSET_BYTES_STATUS_PRESENT,
                "asset_path": "assets/items/smile.png",
                "asset_sha256": sha256_bytes(PNG_BYTES),
                "asset_size_bytes": len(PNG_BYTES),
            },
        ]
    }
    archive_path = _write_import_archive_from_payloads(
        tmp_path / "update-existing.tldw-vnpack",
        pack=pack,
        slots=slots,
        items=items,
        asset_payloads={"assets/items/smile.png": PNG_BYTES},
    )
    preview = await _create_completed_preview(repo, archive_path)
    saver = FakeGeneratedFileSaver()
    unregistered: list[int] = []

    async def unregister(file_id: int, *, hard_delete: bool = False) -> bool:
        unregistered.append(file_id)
        return True

    importer = VNPackImporter(
        repo=repo,
        owner_user_id=USER_ID,
        save_vn_asset_image=saver,
        unregister_generated_file=unregister,
    )

    result = await importer.import_pack(
        preview_id=int(preview["id"]),
        job_id="job-update-existing",
        trust_mode=TRUST_MODE_TRUSTED_RESTORE,
        target_mode="update_existing",
        character_action="link_existing_character",
        target_character_id=target_character_id,
        target_pack_id=int(existing_pack["id"]),
        conflict_decisions={"confirm_diff_ids": []},
    )

    assert result["pack_id"] == existing_pack["id"]
    refreshed_existing_item = repo.get_item(int(existing_item["id"]))
    assert refreshed_existing_item is not None
    assert refreshed_existing_item["generated_file_id"] == 777
    assert refreshed_existing_item["storage_ref"] == "vn_assets/local-existing.png"
    assert bool(refreshed_existing_item["preferred"]) is True
    all_slots = repo.list_slots(int(existing_pack["id"]))
    assert {slot["slot_key"] for slot in all_slots} == {
        "sprite.primary.neutral",
        "sprite.primary.smile",
    }
    all_items = repo.list_items(int(existing_pack["id"]))
    assert len(all_items) == 2
    added_item = next(item for item in all_items if item["id"] != existing_item["id"])
    assert added_item["generated_file_id"] == saver.records[0]["id"]
    assert saver.calls[0]["item_id"] == added_item["id"]
    assert unregistered == []

    journal = repo.get_import_journal(result["import_id"], owner_user_id=USER_ID)
    assert journal is not None
    id_maps = json.loads(journal["id_maps_json"])
    created_records = json.loads(journal["created_records_json"])
    assert id_maps["items"]["400"] == existing_item["id"]
    assert id_maps["items"]["401"] == added_item["id"]
    assert created_records["slot_ids"] == [next(slot["id"] for slot in all_slots if slot["slot_key"].endswith("smile"))]
    assert created_records["item_ids"] == [added_item["id"]]


@pytest.mark.asyncio
async def test_update_existing_refuses_risky_diffs_without_confirmation(
    repo: VNAssetPacksRepository,
    target_character_id: int,
    tmp_path: Path,
) -> None:
    existing_pack = repo.create_pack(
        owner_user_id=USER_ID,
        primary_character_id=target_character_id,
        title="Imported Backup",
    )
    repo.create_slot(
        pack_id=int(existing_pack["id"]),
        asset_type="sprite",
        slot_key="sprite.primary.neutral",
        labels={"expression": "angry"},
        variant_count=1,
    )
    archive_path = _write_import_archive(tmp_path / "risky-update.tldw-vnpack")
    preview = await _create_completed_preview(repo, archive_path)
    importer = VNPackImporter(
        repo=repo,
        owner_user_id=USER_ID,
        save_vn_asset_image=FakeGeneratedFileSaver(),
    )

    with pytest.raises(ValueError, match="update_confirmation_required"):
        await importer.import_pack(
            preview_id=int(preview["id"]),
            job_id="job-update-risks",
            trust_mode=TRUST_MODE_TRUSTED_RESTORE,
            target_mode="update_existing",
            character_action="link_existing_character",
            target_character_id=target_character_id,
            target_pack_id=int(existing_pack["id"]),
            conflict_decisions={"confirm_diff_ids": []},
        )

    assert repo.list_items(int(existing_pack["id"])) == []


@pytest.mark.asyncio
async def test_update_existing_requires_manual_resolution_for_variant_index_match(
    repo: VNAssetPacksRepository,
    target_character_id: int,
    tmp_path: Path,
) -> None:
    existing_pack = repo.create_pack(
        owner_user_id=USER_ID,
        primary_character_id=target_character_id,
        title="Imported Backup",
    )
    slot = repo.create_slot(
        pack_id=int(existing_pack["id"]),
        asset_type="sprite",
        slot_key="sprite.primary.neutral",
        labels={"expression": "neutral"},
        prompt_template="neutral pose",
        negative_prompt_template="blur",
        variant_count=2,
        width=128,
        height=256,
    )
    existing_item = repo.create_item(
        pack_id=int(existing_pack["id"]),
        slot_id=int(slot["id"]),
        variant_index=0,
        review_status="draft",
    )
    archive_path = _write_import_archive(tmp_path / "manual-resolution.tldw-vnpack")
    preview = await _create_completed_preview(repo, archive_path)
    saver = FakeGeneratedFileSaver()
    importer = VNPackImporter(
        repo=repo,
        owner_user_id=USER_ID,
        save_vn_asset_image=saver,
    )

    with pytest.raises(ValueError, match="update_manual_resolution_required"):
        await importer.import_pack(
            preview_id=int(preview["id"]),
            job_id="job-update-manual-resolution",
            trust_mode=TRUST_MODE_TRUSTED_RESTORE,
            target_mode="update_existing",
            character_action="link_existing_character",
            target_character_id=target_character_id,
            target_pack_id=int(existing_pack["id"]),
            conflict_decisions={"confirm_all_risky_diffs": True},
        )

    assert [item["id"] for item in repo.list_items(int(existing_pack["id"]))] == [existing_item["id"]]
    assert saver.calls == []


@pytest.mark.asyncio
async def test_update_existing_refuses_changed_plan_after_preview(
    repo: VNAssetPacksRepository,
    target_character_id: int,
    tmp_path: Path,
) -> None:
    existing_pack = repo.create_pack(
        owner_user_id=USER_ID,
        primary_character_id=target_character_id,
        title="Imported Backup",
    )
    repo.create_slot(
        pack_id=int(existing_pack["id"]),
        asset_type="sprite",
        slot_key="sprite.primary.neutral",
        labels={"expression": "neutral"},
        variant_count=1,
    )
    pack = _pack_payload()
    slots = {
        "slots": [
            {
                "source_slot_id": 300,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.neutral",
                "labels": {"expression": "neutral"},
                "variant_count": 1,
            },
            {
                "source_slot_id": 301,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.smile",
                "labels": {"expression": "smile"},
                "variant_count": 1,
            },
        ]
    }
    items = {
        "items": [
            {
                "source_item_id": 401,
                "source_slot_id": 301,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.smile",
                "variant_index": 0,
                "mime_type": "image/png",
                "width": 128,
                "height": 256,
                "bytes": len(PNG_BYTES),
                "review_status": "approved",
                "preferred": True,
                "asset_bytes_status": ASSET_BYTES_STATUS_PRESENT,
                "asset_path": "assets/items/smile.png",
                "asset_sha256": sha256_bytes(PNG_BYTES),
                "asset_size_bytes": len(PNG_BYTES),
            }
        ]
    }
    archive_path = _write_import_archive_from_payloads(
        tmp_path / "changed-plan.tldw-vnpack",
        pack=pack,
        slots=slots,
        items=items,
        asset_payloads={"assets/items/smile.png": PNG_BYTES},
    )
    preview = await _create_completed_preview(repo, archive_path)
    repo.create_slot(
        pack_id=int(existing_pack["id"]),
        asset_type="sprite",
        slot_key="sprite.primary.smile",
        labels={"expression": "smile"},
        variant_count=1,
    )
    saver = FakeGeneratedFileSaver()
    importer = VNPackImporter(
        repo=repo,
        owner_user_id=USER_ID,
        save_vn_asset_image=saver,
    )

    with pytest.raises(ValueError, match="update_plan_changed"):
        await importer.import_pack(
            preview_id=int(preview["id"]),
            job_id="job-update-plan-changed",
            trust_mode=TRUST_MODE_TRUSTED_RESTORE,
            target_mode="update_existing",
            character_action="link_existing_character",
            target_character_id=target_character_id,
            target_pack_id=int(existing_pack["id"]),
            conflict_decisions={"confirm_diff_ids": []},
        )

    assert saver.calls == []
    assert repo.list_items(int(existing_pack["id"])) == []
