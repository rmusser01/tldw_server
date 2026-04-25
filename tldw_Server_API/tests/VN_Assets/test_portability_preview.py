from __future__ import annotations

import json
import zipfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.VN_Assets.jobs import VN_PACK_IMPORT_PREVIEW_JOB_TYPE
from tldw_Server_API.app.core.VN_Assets.portability.constants import (
    ASSET_BYTES_STATUS_MISSING,
    ASSET_BYTES_STATUS_PRESENT,
    CHECKSUMS_PATH,
    MANIFEST_PATH,
    VNPACK_SCHEMA_VERSION,
)
from tldw_Server_API.app.core.VN_Assets.portability.fingerprints import (
    canonical_json_bytes,
    canonical_payload_fingerprint,
    sha256_bytes,
)
from tldw_Server_API.app.core.VN_Assets.portability.preview import VNPackImportPreviewer

PNG_BYTES = b"\x89PNG\r\n\x1a\npreview-test-png"


@pytest.fixture
def chacha_db(tmp_path: Path) -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="vn-preview-test")
    yield database
    database.close_connection()


@pytest.fixture
def repo(chacha_db: CharactersRAGDB) -> VNAssetPacksRepository:
    return VNAssetPacksRepository.initialized(chacha_db)


@pytest.fixture
def character_id(chacha_db: CharactersRAGDB) -> int:
    character_id = chacha_db.add_character_card(
        {
            "name": "Preview Mira",
            "description": "Local character for update preview tests.",
        }
    )
    assert character_id is not None
    return int(character_id)


def _base_pack() -> dict[str, Any]:
    return {
        "pack": {
            "source_pack_id": 10,
            "title": "Preview Pack",
            "description": None,
            "status": "draft",
            "content_rating": "general",
            "primary_character_id": 99,
            "source_world_book_ids": [],
        }
    }


def _base_slots() -> dict[str, Any]:
    return {
        "slots": [
            {
                "source_slot_id": 20,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.neutral",
                "labels": {"expression": "neutral"},
                "variant_count": 2,
                "required_for_runtime": True,
            }
        ]
    }


def _base_items(*, missing_only: bool = False) -> dict[str, Any]:
    items = [
        {
            "source_item_id": 30,
            "source_slot_id": 20,
            "asset_type": "sprite",
            "slot_key": "sprite.primary.neutral",
            "variant_index": 0,
            "mime_type": "image/png",
            "width": 128,
            "height": 256,
            "bytes": len(PNG_BYTES),
            "review_status": "approved",
            "preferred": True,
            "asset_bytes_status": ASSET_BYTES_STATUS_PRESENT,
            "asset_path": "assets/items/neutral.png",
            "asset_sha256": sha256_bytes(PNG_BYTES),
            "asset_size_bytes": len(PNG_BYTES),
        },
        {
            "source_item_id": 31,
            "source_slot_id": 20,
            "asset_type": "sprite",
            "slot_key": "sprite.primary.neutral",
            "variant_index": 1,
            "mime_type": "image/png",
            "width": 128,
            "height": 256,
            "bytes": 1234,
            "review_status": "draft",
            "preferred": False,
            "asset_bytes_status": ASSET_BYTES_STATUS_MISSING,
        },
    ]
    if missing_only:
        items = [items[1]]
    return {"items": items}


def _write_vnpack(
    archive_path: Path,
    *,
    manifest_overrides: dict[str, Any] | None = None,
    pack: dict[str, Any] | None = None,
    slots: dict[str, Any] | None = None,
    items: dict[str, Any] | None = None,
    character: dict[str, Any] | None = None,
    tamper_checksum_path: str | None = None,
    omit_members: set[str] | None = None,
    extra_members: dict[str, bytes] | None = None,
) -> Path:
    pack_payload = pack or _base_pack()
    slots_payload = slots or _base_slots()
    items_payload = items or _base_items()
    omit_members = omit_members or set()
    payloads: dict[str, bytes] = {
        "metadata/pack.json": canonical_json_bytes(pack_payload),
        "metadata/slots.json": canonical_json_bytes(slots_payload),
        "metadata/items.json": canonical_json_bytes(items_payload),
    }
    if character is not None:
        payloads["metadata/character.json"] = canonical_json_bytes(character)
    item_records = items_payload.get("items", [])
    if not isinstance(item_records, list):
        item_records = []
    for item in item_records:
        if item.get("asset_bytes_status") == ASSET_BYTES_STATUS_PRESENT and item.get("asset_path"):
            payloads[str(item["asset_path"])] = PNG_BYTES

    checksums = {path: sha256_bytes(content) for path, content in sorted(payloads.items())}
    fingerprint = canonical_payload_fingerprint(
        {
            "pack": pack_payload["pack"],
            "slots": slots_payload["slots"],
            "items": item_records,
            "character": None if character is None else character.get("character"),
        }
    )
    manifest = {
        "schema_version": VNPACK_SCHEMA_VERSION,
        "archive_profile": "backup",
        "pack_title": pack_payload["pack"]["title"],
        "content_rating": pack_payload["pack"]["content_rating"],
        "canonical_payload_fingerprint": fingerprint,
        "source_pack_fingerprint": "source-pack-fingerprint",
        "counts": {
            "slots": len(slots_payload["slots"]),
            "items": len(item_records),
            "assets_with_bytes": sum(
                1
                for item in item_records
                if item.get("asset_bytes_status") == ASSET_BYTES_STATUS_PRESENT
            ),
        },
        "include_images": True,
        "include_character": character is not None,
        "include_world_books": False,
        "provenance_mode": "redacted",
        "encryption": {"encrypted": False, "scheme": None},
        "sections": [{"path": path, "sha256": digest} for path, digest in sorted(checksums.items())],
        "warnings": [],
    }
    manifest.update(manifest_overrides or {})
    payloads[MANIFEST_PATH] = canonical_json_bytes(manifest)
    checksums[MANIFEST_PATH] = sha256_bytes(payloads[MANIFEST_PATH])
    if tamper_checksum_path:
        checksums[tamper_checksum_path] = "0" * 64
    payloads[CHECKSUMS_PATH] = canonical_json_bytes(checksums)
    payloads.update(extra_members or {})

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in sorted(payloads.items()):
            if path not in omit_members:
                archive.writestr(path, content)
    return archive_path


@pytest.mark.asyncio
async def test_preview_rejects_traversal_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.tldw-vnpack"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(MANIFEST_PATH, b"{}")
        archive.writestr("metadata/pack.json", b"{}")
        archive.writestr("metadata/slots.json", b"{}")
        archive.writestr("metadata/items.json", b"{}")
        archive.writestr(CHECKSUMS_PATH, b"{}")
        archive.writestr("../escape.png", b"bad")

    previewer = VNPackImportPreviewer()
    with pytest.raises(ValueError, match="unsafe_archive_member"):
        await previewer.create_preview(archive_path=archive_path, owner_user_id=42)


@pytest.mark.asyncio
async def test_preview_rejects_missing_required_file(tmp_path: Path) -> None:
    archive_path = _write_vnpack(
        tmp_path / "missing.tldw-vnpack",
        omit_members={"metadata/items.json"},
    )

    previewer = VNPackImportPreviewer()
    with pytest.raises(ValueError, match="missing_required_archive_member"):
        await previewer.create_preview(archive_path=archive_path, owner_user_id=42)


@pytest.mark.asyncio
async def test_preview_rejects_checksum_mismatch(tmp_path: Path) -> None:
    archive_path = _write_vnpack(
        tmp_path / "checksum.tldw-vnpack",
        tamper_checksum_path="metadata/items.json",
    )

    previewer = VNPackImportPreviewer()
    with pytest.raises(ValueError, match="checksum_mismatch"):
        await previewer.create_preview(archive_path=archive_path, owner_user_id=42)


@pytest.mark.asyncio
async def test_preview_streams_asset_members_for_checksum_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive_path = _write_vnpack(tmp_path / "streamed-assets.tldw-vnpack")
    original_read = zipfile.ZipFile.read

    def reject_asset_read(
        archive: zipfile.ZipFile,
        name: str | zipfile.ZipInfo,
        pwd: bytes | None = None,
    ) -> bytes:
        filename = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
        if filename.startswith("assets/"):
            pytest.fail(f"asset member should be streamed instead of read fully: {filename}")
        return original_read(archive, name, pwd)

    monkeypatch.setattr(zipfile.ZipFile, "read", reject_asset_read)

    preview = await VNPackImportPreviewer().create_preview(
        archive_path=archive_path,
        owner_user_id=42,
    )

    assert preview["bundle_summary"]["assets_with_bytes"] == 1
    assert preview["quota_estimate"]["asset_bytes"] == len(PNG_BYTES)


@pytest.mark.asyncio
async def test_preview_rejects_malformed_metadata(tmp_path: Path) -> None:
    archive_path = _write_vnpack(
        tmp_path / "malformed.tldw-vnpack",
        items={"items": "not-a-list"},
    )

    previewer = VNPackImportPreviewer()
    with pytest.raises(ValueError, match="malformed_metadata: metadata/items.json"):
        await previewer.create_preview(archive_path=archive_path, owner_user_id=42)


@pytest.mark.asyncio
async def test_preview_rejects_unsupported_schema(tmp_path: Path) -> None:
    archive_path = _write_vnpack(
        tmp_path / "unsupported.tldw-vnpack",
        manifest_overrides={"schema_version": "tldw.vnpack.v99"},
    )

    previewer = VNPackImportPreviewer()
    with pytest.raises(ValueError, match="unsupported_vnpack_schema"):
        await previewer.create_preview(archive_path=archive_path, owner_user_id=42)


@pytest.mark.asyncio
async def test_missing_character_payload_requires_link_or_fail_action(tmp_path: Path) -> None:
    archive_path = _write_vnpack(tmp_path / "no-character.tldw-vnpack")

    preview = await VNPackImportPreviewer().create_preview(
        archive_path=archive_path,
        owner_user_id=42,
    )

    choice = next(item for item in preview["required_choices"] if item["choice_id"] == "primary_character")
    assert choice["allowed_actions"] == ["link_existing_character", "fail_import"]
    assert choice["default_action"] == "link_existing_character"


@pytest.mark.asyncio
async def test_missing_asset_bytes_report_counts_and_required_slot_impact(tmp_path: Path) -> None:
    archive_path = _write_vnpack(
        tmp_path / "missing-assets.tldw-vnpack",
        items=_base_items(missing_only=True),
    )

    preview = await VNPackImportPreviewer().create_preview(
        archive_path=archive_path,
        owner_user_id=42,
    )

    assert preview["bundle_summary"]["missing_asset_items"] == 1
    assert preview["bundle_summary"]["required_slots_impacted_by_missing_bytes"] == 1
    assert preview["quota_estimate"]["asset_bytes"] == 0
    assert preview["validation_warnings"] == [
        "missing_asset_bytes:sprite:sprite.primary.neutral:variant:1"
    ]


@pytest.mark.asyncio
async def test_preview_canonical_fingerprint_is_deterministic(tmp_path: Path) -> None:
    archive_path = _write_vnpack(tmp_path / "stable.tldw-vnpack")

    first = await VNPackImportPreviewer().create_preview(
        archive_path=archive_path,
        owner_user_id=42,
    )
    second = await VNPackImportPreviewer().create_preview(
        archive_path=archive_path,
        owner_user_id=42,
    )

    assert first["archive_sha256"] == second["archive_sha256"]
    assert first["canonical_payload_fingerprint"] == second["canonical_payload_fingerprint"]
    assert len(first["canonical_payload_fingerprint"]) == 64


@pytest.mark.asyncio
async def test_import_preview_worker_updates_preview_and_portability_rows(
    repo: VNAssetPacksRepository,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    archive_path = _write_vnpack(tmp_path / "worker-preview.tldw-vnpack")
    preview = repo.create_import_preview(
        owner_user_id=42,
        job_id="123",
        status="queued",
        archive_path=str(archive_path),
    )
    portability_job = repo.create_portability_job(
        owner_user_id=42,
        job_id="123",
        operation="import_preview",
        status="queued",
        stage="queued",
        preview_id=int(preview["id"]),
        archive_path=str(archive_path),
    )
    worker = VNAssetGenerationWorker(repo=repo, jobs_manager=object())

    result = await worker.handle_job_async(
        {
            "id": "123",
            "job_type": VN_PACK_IMPORT_PREVIEW_JOB_TYPE,
            "owner_user_id": "42",
            "payload": {
                "preview_id": int(preview["id"]),
                "archive_path": str(archive_path),
                "request_id": "req-worker",
                "user_id": 42,
            },
        }
    )

    updated_preview = repo.get_import_preview(int(preview["id"]), owner_user_id=42)
    updated_job = repo.get_portability_job(int(portability_job["id"]), owner_user_id=42)
    assert result["status"] == "previewed"
    assert updated_preview is not None
    assert updated_preview["status"] == "completed"
    assert updated_preview["archive_sha256"] == result["archive_sha256"]
    assert updated_preview["canonical_payload_fingerprint"] == result["canonical_payload_fingerprint"]
    assert updated_job is not None
    assert updated_job["status"] == "completed"
    assert updated_job["stage"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["deleted", "cancelled"])
async def test_import_preview_worker_does_not_resurrect_terminal_preview(
    repo: VNAssetPacksRepository,
    tmp_path: Path,
    terminal_status: str,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

    archive_path = tmp_path / f"{terminal_status}.tldw-vnpack"
    preview = repo.create_import_preview(
        owner_user_id=42,
        job_id="123",
        status=terminal_status,
        archive_path=str(archive_path),
    )
    repo.create_portability_job(
        owner_user_id=42,
        job_id="123",
        operation="import_preview",
        status="cancelled",
        stage=terminal_status,
        preview_id=int(preview["id"]),
        archive_path=str(archive_path),
    )
    worker = VNAssetGenerationWorker(repo=repo, jobs_manager=object())

    result = await worker.handle_job_async(
        {
            "id": "123",
            "job_type": VN_PACK_IMPORT_PREVIEW_JOB_TYPE,
            "owner_user_id": "42",
            "payload": {
                "preview_id": int(preview["id"]),
                "archive_path": str(archive_path),
                "request_id": "req-worker",
                "user_id": 42,
            },
        }
    )

    updated_preview = repo.get_import_preview(int(preview["id"]), owner_user_id=42)
    updated_job = repo.get_portability_job_by_job_id("123", owner_user_id=42)
    assert result["status"] == "cancelled"
    assert updated_preview is not None
    assert updated_preview["status"] == terminal_status
    assert updated_job is not None
    assert updated_job["status"] == "cancelled"
    assert updated_job["stage"] == terminal_status


@pytest.mark.asyncio
async def test_update_existing_identity_rules(
    repo: VNAssetPacksRepository,
    character_id: int,
    tmp_path: Path,
) -> None:
    pack = repo.create_pack(
        owner_user_id=42,
        primary_character_id=character_id,
        title="Preview Pack",
    )
    slot = repo.create_slot(
        pack_id=int(pack["id"]),
        asset_type="sprite",
        slot_key="sprite.primary.neutral",
        labels={"expression": "serious"},
        variant_count=5,
    )
    repo.create_item(
        pack_id=int(pack["id"]),
        slot_id=int(slot["id"]),
        variant_index=0,
        review_status="approved",
        generated_file_id=1001,
        storage_ref="vn_assets/local-fingerprint.png",
        source_context_snapshot={"source_item_fingerprint": "fp-first"},
    )
    repo.create_item(
        pack_id=int(pack["id"]),
        slot_id=int(slot["id"]),
        variant_index=1,
        review_status="approved",
        generated_file_id=1002,
        storage_ref="vn_assets/local-checksum.png",
        backend_metadata={"vnpack_import": {"source_asset_sha256": sha256_bytes(PNG_BYTES)}},
    )
    repo.create_item(
        pack_id=int(pack["id"]),
        slot_id=int(slot["id"]),
        variant_index=2,
        review_status="draft",
    )
    repo.create_item(
        pack_id=int(pack["id"]),
        slot_id=int(slot["id"]),
        variant_index=3,
        source_context_snapshot={"source_item_fingerprint": "fp-dupe"},
    )
    repo.create_item(
        pack_id=int(pack["id"]),
        slot_id=int(slot["id"]),
        variant_index=4,
        source_context_snapshot={"source_item_fingerprint": "fp-dupe"},
    )
    items = {
        "items": [
            {
                "source_item_id": 30,
                "source_slot_id": 20,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.neutral",
                "variant_index": 0,
                "review_status": "approved",
                "preferred": False,
                "source_item_fingerprint": "fp-first",
                "asset_bytes_status": ASSET_BYTES_STATUS_MISSING,
            },
            {
                "source_item_id": 31,
                "source_slot_id": 20,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.neutral",
                "variant_index": 1,
                "mime_type": "image/png",
                "review_status": "approved",
                "preferred": False,
                "asset_bytes_status": ASSET_BYTES_STATUS_PRESENT,
                "asset_path": "assets/items/checksum.png",
                "asset_sha256": sha256_bytes(PNG_BYTES),
                "asset_size_bytes": len(PNG_BYTES),
            },
            {
                "source_item_id": 32,
                "source_slot_id": 20,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.neutral",
                "variant_index": 2,
                "review_status": "draft",
                "preferred": False,
                "asset_bytes_status": ASSET_BYTES_STATUS_MISSING,
            },
            {
                "source_item_id": 33,
                "source_slot_id": 20,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.neutral",
                "variant_index": 3,
                "review_status": "draft",
                "preferred": False,
                "source_item_fingerprint": "fp-dupe",
                "asset_bytes_status": ASSET_BYTES_STATUS_MISSING,
            },
            {
                "source_item_id": 34,
                "source_slot_id": 20,
                "asset_type": "sprite",
                "slot_key": "sprite.primary.neutral",
                "variant_index": 5,
                "review_status": "draft",
                "preferred": False,
                "asset_bytes_status": ASSET_BYTES_STATUS_MISSING,
            },
        ]
    }
    archive_path = _write_vnpack(tmp_path / "update-preview.tldw-vnpack", items=items)

    preview = await VNPackImportPreviewer(repo=repo).create_preview(
        archive_path=archive_path,
        owner_user_id=42,
    )

    update_plan = preview["proposed_plan"]["update_existing"]
    candidate = update_plan["candidate_packs"][0]
    assert candidate["target_pack_id"] == pack["id"]
    assert candidate["matched_slots"] == [
        {
            "source_slot_id": 20,
            "local_slot_id": slot["id"],
            "identity": "sprite:sprite.primary.neutral",
        }
    ]
    matched_by_source = {item["source_item_id"]: item for item in candidate["matched_items"]}
    assert matched_by_source[30]["match_kind"] == "source_item_fingerprint"
    assert matched_by_source[31]["match_kind"] == "slot_checksum"
    assert candidate["added_items"] == [{"source_item_id": 34, "source_slot_id": 20}]

    diffs = candidate["diffs"]
    assert any(
        diff["kind"] == "slot_metadata_diff"
        and diff["source_slot_id"] == 20
        and diff["requires_confirmation"] is True
        for diff in diffs
    )
    assert any(
        diff["kind"] == "item_variant_index_ambiguous"
        and diff["source_item_id"] == 32
        and diff["requires_confirmation"] is True
        for diff in diffs
    )
    assert any(
        diff["kind"] == "item_duplicate_match"
        and diff["source_item_id"] == 33
        and diff["severity"] == "blocking"
        for diff in diffs
    )
