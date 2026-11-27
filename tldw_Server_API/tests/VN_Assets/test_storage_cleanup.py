from __future__ import annotations

import asyncio as real_asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from os import utime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import vn_assets as vn_assets_endpoint
from tldw_Server_API.app.api.v1.endpoints.vn_assets import router as vn_assets_router
from tldw_Server_API.app.api.v1.schemas.storage_schemas import GeneratedFile
from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VNAssetCleanupRequest,
    VNAssetPackCreate,
    VNAssetReviewRequest,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.DB_Management.VNPlay_DB import VNPlayRepository
from tldw_Server_API.app.core.DB_Management.VNScripts_DB import VNScriptsRepository
from tldw_Server_API.app.core.VN_Assets.cleanup_blockers import VNAssetCleanupBlockerProvider
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Platform.idempotency import canonical_multipart_payload_hash, canonical_payload_hash

USER_ID = 42
PNG_BYTES = b"\x89PNG\r\n\x1a\nvn-asset-test"
VALID_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeStorageService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.records: dict[str, dict[str, Any]] = {}

    async def get_vn_generated_file(self, *, user_id: int, source_ref: str) -> dict[str, Any] | None:
        """Return an owned registration only when its stored bytes are valid."""
        from tldw_Server_API.app.core.VN_Assets.storage import resolve_vn_asset_storage_path

        record = self.records.get(source_ref)
        if record is None or record["user_id"] != user_id:
            return None
        path = resolve_vn_asset_storage_path(user_id=user_id, storage_path=record["storage_path"])
        if not path.is_file() or path.stat().st_size != record["file_size_bytes"]:
            raise RuntimeError("Registered VN file has missing or invalid bytes")
        return record

    async def register_generated_file(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        record = {"id": 901, "is_deleted": False, **kwargs}
        self.records[kwargs["source_ref"]] = record
        return record


class FakeGeneratedFilesRepo:
    def __init__(self) -> None:
        self.records: dict[int, dict[str, Any]] = {}
        self.accessed_ids: list[int] = []
        self.hard_deleted_ids: list[int] = []

    async def get_file_by_id(self, file_id: int) -> dict[str, Any] | None:
        return self.records.get(file_id)

    async def get_files_by_ids(self, file_ids: list[int]) -> list[dict[str, Any]]:
        return [self.records[file_id] for file_id in file_ids if file_id in self.records]

    async def list_files(
        self, *, user_id: int, source_feature: str, offset: int, limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        records = [
            record for record in self.records.values()
            if record["user_id"] == user_id and record["source_feature"] == source_feature and not record["is_deleted"]
        ]
        return records[offset:offset + limit], len(records)

    async def registered_storage_paths(self, *, user_id: int, storage_paths: list[str]) -> set[str]:
        return {
            record["storage_path"] for record in self.records.values()
            if record["user_id"] == user_id and record["storage_path"] in storage_paths
        }

    async def update_accessed_at(self, file_id: int) -> None:
        self.accessed_ids.append(file_id)

    async def hard_delete_file(self, file_id: int) -> bool:
        self.hard_deleted_ids.append(file_id)
        self.records.pop(file_id, None)
        return True


@pytest.fixture
def chacha_db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="vn-assets-storage-test")
    yield database
    database.close_connection()


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
def fake_generated_files_repo() -> FakeGeneratedFilesRepo:
    return FakeGeneratedFilesRepo()


@pytest.fixture
def outputs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "outputs"
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", staticmethod(lambda user_id: path))
    return path


@pytest.fixture
def client(
    chacha_db: CharactersRAGDB,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> TestClient:
    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")

    async def override_user() -> User:
        return User(id=USER_ID, username="vn-storage-user")

    async def override_chacha_db() -> CharactersRAGDB:
        return chacha_db

    async def override_generated_files_repo() -> FakeGeneratedFilesRepo:
        return fake_generated_files_repo

    async def override_storage_service() -> SimpleNamespace:
        async def unregister_generated_file(file_id: int, hard_delete: bool = False) -> bool:
            if not hard_delete:
                return False
            return await fake_generated_files_repo.hard_delete_file(file_id)

        return SimpleNamespace(unregister_generated_file=unregister_generated_file)

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db
    generated_files_repo_dep = getattr(vn_assets_endpoint, "_generated_files_repo", None)
    if generated_files_repo_dep is not None:
        app.dependency_overrides[generated_files_repo_dep] = override_generated_files_repo
    storage_service_dep = getattr(vn_assets_endpoint, "_storage_service", None)
    if storage_service_dep is not None:
        app.dependency_overrides[storage_service_dep] = override_storage_service
    return TestClient(app)


@pytest.fixture
def asset_with_generated_file(
    chacha_db: CharactersRAGDB,
    character_id: int,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
    outputs_dir: Path,
) -> SimpleNamespace:
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    pack = service.create_pack(VNAssetPackCreate(title="Storage Pack", primary_character_id=character_id))
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]
    repo = VNAssetPacksRepository.initialized(chacha_db)
    item = repo.create_item(
        pack_id=pack.id,
        slot_id=slot.id,
        variant_index=0,
        generated_file_id=700,
        mime_type="image/png",
    )
    storage_path = "vn_assets/fixture.png"
    file_path = outputs_dir / storage_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(PNG_BYTES)
    fake_generated_files_repo.records[700] = {
        "id": 700,
        "user_id": USER_ID,
        "filename": "fixture.png",
        "original_filename": "fixture.png",
        "storage_path": storage_path,
        "mime_type": "image/png",
        "file_size_bytes": len(PNG_BYTES),
        "source_feature": "vn_assets",
        "source_ref": f"vn_asset_item:{item['id']}",
        "is_deleted": False,
    }
    return SimpleNamespace(pack_id=pack.id, slot_id=slot.id, item_id=item["id"], file_id=700)


@pytest.mark.asyncio
async def test_save_vn_asset_image_registers_vn_source_feature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.repos import generated_files_repo
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.Storage import generated_file_helpers

    outputs_dir = tmp_path / "outputs"
    fake_service = FakeStorageService()

    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", staticmethod(lambda user_id: outputs_dir))

    async def fake_get_storage_service() -> FakeStorageService:
        return fake_service

    monkeypatch.setattr(generated_file_helpers, "get_storage_service", fake_get_storage_service)

    record = await generated_file_helpers.save_and_register_vn_asset_image(
        user_id=1,
        image_bytes=PNG_BYTES,
        image_format="png",
        pack_id=10,
        item_id=20,
        asset_type="sprite",
        labels={"expression": "happy"},
        check_quota=False,
    )

    assert generated_files_repo.SOURCE_FEATURE_VN_ASSETS in generated_files_repo.VALID_SOURCE_FEATURES
    assert record["source_feature"] == "vn_assets"
    assert record["source_ref"] == "vn_asset_item:20"
    assert record["folder_tag"] == "vn-assets/10"
    assert record["file_category"] == "image"
    assert record["mime_type"] == "image/png"
    assert record["tags"] == [
        "vn_pack:10",
        "vn_item:20",
        "asset_type:sprite",
        "label:expression:happy",
    ]
    assert record["check_quota"] is False
    assert (outputs_dir / record["storage_path"]).read_bytes() == PNG_BYTES


@pytest.mark.asyncio
async def test_save_vn_asset_image_cleans_file_when_storage_service_lookup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.Storage import generated_file_helpers

    outputs_dir = tmp_path / "outputs"
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", staticmethod(lambda user_id: outputs_dir))

    async def fail_get_storage_service() -> Any:
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(generated_file_helpers, "get_storage_service", fail_get_storage_service)

    with pytest.raises(RuntimeError, match="storage unavailable"):
        await generated_file_helpers.save_and_register_vn_asset_image(
            user_id=1,
            image_bytes=PNG_BYTES,
            image_format="png",
            pack_id=10,
            item_id=20,
            asset_type="sprite",
        )

    assert [path for path in outputs_dir.glob("**/*") if path.is_file()] == []


def test_storage_schema_accepts_vn_assets_source_feature() -> None:
    now = datetime.now(timezone.utc)

    file_record = GeneratedFile(
        id=700,
        uuid="vn-file-uuid",
        user_id=USER_ID,
        filename="fixture.png",
        storage_path="vn_assets/fixture.png",
        created_at=now,
        updated_at=now,
        file_category="image",
        source_feature="vn_assets",
    )

    assert file_record.source_feature == "vn_assets"


def test_unlink_vn_asset_storage_file_treats_delete_race_as_missing(
    outputs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.VN_Assets.storage import unlink_vn_asset_storage_file

    file_path = outputs_dir / "vn_assets" / "race.png"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(PNG_BYTES)

    def race_unlink(self: Path) -> None:
        raise FileNotFoundError(str(self))

    monkeypatch.setattr(Path, "unlink", race_unlink)

    assert unlink_vn_asset_storage_file(user_id=USER_ID, storage_path="vn_assets/race.png") is False


def test_content_endpoint_streams_owned_generated_file(
    client: TestClient,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> None:
    response = client.get(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/items/"
        f"{asset_with_generated_file.item_id}/content"
    )

    assert response.status_code == 200
    assert response.content == PNG_BYTES
    assert response.headers["content-type"].startswith("image/png")
    assert fake_generated_files_repo.accessed_ids == [asset_with_generated_file.file_id]


def test_content_endpoint_streams_documented_octet_stream(
    client: TestClient,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> None:
    record = fake_generated_files_repo.records[asset_with_generated_file.file_id]
    record["mime_type"] = "application/octet-stream"
    record["filename"] = "fixture.bin"
    record["original_filename"] = "fixture.bin"

    response = client.get(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/items/"
        f"{asset_with_generated_file.item_id}/content"
    )

    assert response.status_code == 200
    assert response.content == PNG_BYTES
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert fake_generated_files_repo.accessed_ids == [asset_with_generated_file.file_id]


def test_content_endpoint_denies_cross_user_generated_file(
    client: TestClient,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> None:
    fake_generated_files_repo.records[asset_with_generated_file.file_id]["user_id"] = 7

    response = client.get(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/items/"
        f"{asset_with_generated_file.item_id}/content"
    )

    assert response.status_code == 404
    assert fake_generated_files_repo.accessed_ids == []


def test_content_endpoint_rejects_storage_path_escape(
    client: TestClient,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> None:
    fake_generated_files_repo.records[asset_with_generated_file.file_id]["storage_path"] = "../fixture.png"

    response = client.get(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/items/"
        f"{asset_with_generated_file.item_id}/content"
    )

    assert response.status_code == 404
    assert fake_generated_files_repo.accessed_ids == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_feature", "chat"),
        ("source_ref", "vn_asset_item:999999"),
        ("is_deleted", True),
    ],
)
def test_content_endpoint_rejects_invalid_generated_file_provenance(
    client: TestClient,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
    field: str,
    value: object,
) -> None:
    fake_generated_files_repo.records[asset_with_generated_file.file_id][field] = value

    response = client.get(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/items/"
        f"{asset_with_generated_file.item_id}/content"
    )

    assert response.status_code == 404
    assert fake_generated_files_repo.accessed_ids == []


def test_content_endpoint_rejects_disallowed_media_type_before_serving(
    client: TestClient,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> None:
    fake_generated_files_repo.records[asset_with_generated_file.file_id]["mime_type"] = "text/plain"

    response = client.get(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/items/"
        f"{asset_with_generated_file.item_id}/content"
    )

    assert response.status_code == 404
    assert fake_generated_files_repo.accessed_ids == []


def test_preview_endpoint_uses_same_owned_image_validation_as_content(
    client: TestClient,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> None:
    response = client.get(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/items/"
        f"{asset_with_generated_file.item_id}/preview"
    )

    assert response.status_code == 200
    assert response.content == PNG_BYTES
    assert response.headers["content-type"].startswith("image/png")
    assert fake_generated_files_repo.accessed_ids == [asset_with_generated_file.file_id]


def test_preview_endpoint_rejects_octet_stream_content(
    client: TestClient,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> None:
    fake_generated_files_repo.records[asset_with_generated_file.file_id]["mime_type"] = (
        "application/octet-stream"
    )

    response = client.get(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/items/"
        f"{asset_with_generated_file.item_id}/preview"
    )

    assert response.status_code == 404
    assert fake_generated_files_repo.accessed_ids == []


def test_content_endpoint_blocks_policy_denied_item_before_touching_file(
    client: TestClient,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> None:
    fake_generated_files_repo.records[asset_with_generated_file.file_id]["policy_blocked"] = True

    response = client.get(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/items/"
        f"{asset_with_generated_file.item_id}/content"
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "policy_blocked"
    assert fake_generated_files_repo.accessed_ids == []


@pytest.mark.asyncio
async def test_cleanup_dry_run_does_not_delete_files(
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
    outputs_dir: Path,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    service.review_item(
        asset_with_generated_file.item_id,
        VNAssetReviewRequest(review_status="rejected"),
    )

    response = await service.cleanup_pack(
        asset_with_generated_file.pack_id,
        VNAssetCleanupRequest(dry_run=True, statuses=["rejected"]),
        files_repo=fake_generated_files_repo,
    )

    assert response.files_would_delete == 1
    assert response.files_deleted == 0
    assert response.reclaimed_bytes == len(PNG_BYTES)
    assert fake_generated_files_repo.hard_deleted_ids == []
    assert fake_generated_files_repo.records[asset_with_generated_file.file_id]["is_deleted"] is False
    assert (outputs_dir / "vn_assets/fixture.png").read_bytes() == PNG_BYTES
    item = service.get_item_for_pack(asset_with_generated_file.pack_id, asset_with_generated_file.item_id)
    assert item.generated_file_id == asset_with_generated_file.file_id


@pytest.mark.asyncio
async def test_cleanup_executes_rejected_file_delete_and_removes_item(
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
    outputs_dir: Path,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    service.review_item(
        asset_with_generated_file.item_id,
        VNAssetReviewRequest(review_status="rejected"),
    )

    response = await service.cleanup_pack(
        asset_with_generated_file.pack_id,
        VNAssetCleanupRequest(dry_run=False, statuses=["rejected"]),
        files_repo=fake_generated_files_repo,
    )

    assert response.files_would_delete == 1
    assert response.files_deleted == 1
    assert response.removed_item_ids == [asset_with_generated_file.item_id]
    assert response.reclaimed_bytes == len(PNG_BYTES)
    assert fake_generated_files_repo.hard_deleted_ids == [asset_with_generated_file.file_id]
    assert not (outputs_dir / "vn_assets/fixture.png").exists()
    assert service.repo.get_item(asset_with_generated_file.item_id) is None


@pytest.mark.asyncio
async def test_cleanup_requires_confirmation_for_approved_assets(
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    service.review_item(
        asset_with_generated_file.item_id,
        VNAssetReviewRequest(review_status="approved"),
    )

    with pytest.raises(ValueError, match="cleanup_confirmation_required"):
        await service.cleanup_pack(
            asset_with_generated_file.pack_id,
            VNAssetCleanupRequest(
                dry_run=False,
                statuses=["approved"],
                include_approved=True,
            ),
            files_repo=fake_generated_files_repo,
        )


@pytest.mark.asyncio
async def test_cleanup_skips_generated_file_referenced_by_another_item(
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
    outputs_dir: Path,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    service.review_item(
        asset_with_generated_file.item_id,
        VNAssetReviewRequest(review_status="rejected"),
    )
    repo = VNAssetPacksRepository.initialized(chacha_db)
    repo.create_item(
        pack_id=asset_with_generated_file.pack_id,
        slot_id=asset_with_generated_file.slot_id,
        variant_index=1,
        generated_file_id=asset_with_generated_file.file_id,
        mime_type="image/png",
        review_status="draft",
    )

    response = await service.cleanup_pack(
        asset_with_generated_file.pack_id,
        VNAssetCleanupRequest(dry_run=False, statuses=["rejected"]),
        files_repo=fake_generated_files_repo,
    )

    assert response.files_would_delete == 0
    assert response.files_deleted == 0
    assert response.skipped_file_ids == [asset_with_generated_file.file_id]
    assert fake_generated_files_repo.hard_deleted_ids == []
    assert (outputs_dir / "vn_assets/fixture.png").exists()


class BlockingCleanupProvider:
    async def find_blockers(
        self,
        *,
        pack_id: int,
        owner_user_id: int,
        candidates: list[dict[str, Any]],
    ) -> dict[int, list[dict[str, str]]]:
        assert pack_id > 0
        assert owner_user_id == USER_ID
        return {
            int(candidate["generated_file_id"]): [
                {
                    "code": "active_vn_session",
                    "message": "File is still referenced by an active VN session.",
                }
            ]
            for candidate in candidates
        }


@pytest.mark.asyncio
async def test_cleanup_skips_blocked_generated_files_from_pluggable_provider(
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
    outputs_dir: Path,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    service.review_item(
        asset_with_generated_file.item_id,
        VNAssetReviewRequest(review_status="rejected"),
    )

    response = await service.cleanup_pack(
        asset_with_generated_file.pack_id,
        VNAssetCleanupRequest(dry_run=False, statuses=["rejected"]),
        files_repo=fake_generated_files_repo,
        blocker_provider=BlockingCleanupProvider(),
    )

    assert response.files_would_delete == 0
    assert response.files_deleted == 0
    assert response.blocked_count == 1
    assert response.skipped_file_ids == [asset_with_generated_file.file_id]
    assert response.cleanup_blocked == [
        {
            "item_id": asset_with_generated_file.item_id,
            "file_id": asset_with_generated_file.file_id,
            "blockers": [
                {
                    "code": "active_vn_session",
                    "message": "File is still referenced by an active VN session.",
                }
            ],
        }
    ]
    assert fake_generated_files_repo.hard_deleted_ids == []
    assert (outputs_dir / "vn_assets/fixture.png").exists()


@pytest.mark.asyncio
async def test_cleanup_skips_generated_files_referenced_by_published_script_manifest(
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
    outputs_dir: Path,
) -> None:
    scripts_repo = VNScriptsRepository.initialized(chacha_db)
    script = scripts_repo.create_script(
        owner_user_id=USER_ID,
        title="Published route",
        primary_asset_pack_id=asset_with_generated_file.pack_id,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
        status="published",
    )
    scripts_repo.create_manifest_snapshot(
        owner_user_id=USER_ID,
        script_id=int(script["id"]),
        asset_pack_id=asset_with_generated_file.pack_id,
        version_id=1,
        manifest={
            "assets": {
                "sprites": [
                    {
                        "item_id": asset_with_generated_file.item_id,
                        "generated_file_id": asset_with_generated_file.file_id,
                    }
                ]
            }
        },
        manifest_hash="manifest-hash",
    )
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    service.review_item(
        asset_with_generated_file.item_id,
        VNAssetReviewRequest(review_status="rejected"),
    )

    response = await service.cleanup_pack(
        asset_with_generated_file.pack_id,
        VNAssetCleanupRequest(dry_run=False, statuses=["rejected"]),
        files_repo=fake_generated_files_repo,
        blocker_provider=VNAssetCleanupBlockerProvider(chacha_db),
    )

    assert response.files_deleted == 0
    assert response.blocked_count == 1
    assert response.skipped_file_ids == [asset_with_generated_file.file_id]
    assert response.cleanup_blocked[0]["blockers"][0]["code"] == "published_script_manifest"
    assert fake_generated_files_repo.hard_deleted_ids == []
    assert (outputs_dir / "vn_assets/fixture.png").exists()


def test_cleanup_endpoint_dry_run(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    service.review_item(
        asset_with_generated_file.item_id,
        VNAssetReviewRequest(review_status="hidden"),
    )

    response = client.post(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/cleanup",
        json={"dry_run": True, "statuses": ["hidden"]},
    )

    assert response.status_code == 200
    assert response.json()["files_would_delete"] == 1
    assert fake_generated_files_repo.hard_deleted_ids == []


@pytest.mark.parametrize(
    ("review_status", "age_days", "eligible"),
    [
        ("rejected", 29, False),
        ("rejected", 30, True),
        ("hidden", 6, False),
        ("hidden", 7, True),
        ("approved", 45, False),
        ("draft", 45, False),
        ("hidden", None, False),
    ],
)
def test_retention_preview_applies_age_and_review_policy(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
    review_status: str,
    age_days: int | None,
    eligible: bool,
) -> None:
    item_id = asset_with_generated_file.item_id
    repo = VNAssetPacksRepository.initialized(chacha_db)
    repo.update_item_review(item_id, review_status=review_status, preferred=False)
    old = datetime.now(timezone.utc) - timedelta(days=60)
    updated = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat() if age_days is not None else "invalid"
    with chacha_db.transaction() as conn:
        conn.execute("UPDATE vn_asset_items SET created_at = ?, updated_at = ? WHERE id = ?", (old.isoformat(), updated, item_id))

    response = client.post(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/cleanup",
        json={"retention_only": True, "statuses": [review_status]},
    )

    assert response.status_code == 200
    assert response.json()["candidate_item_ids"] == ([item_id] if eligible else [])
    assert response.json()["files_would_delete"] == int(eligible)
    assert fake_generated_files_repo.hard_deleted_ids == []


@pytest.mark.parametrize("operation", ["generation", "export", "import"])
def test_retention_preview_defers_active_pack_work(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    operation: str,
) -> None:
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack_id = asset_with_generated_file.pack_id
    if operation == "generation":
        repo.create_batch(pack_id=pack_id, requested_by_user_id=USER_ID, status="processing")
    elif operation == "export":
        repo.create_portability_job(
            owner_user_id=USER_ID, job_id="export", operation="export", status="queued", stage="queued", pack_id=pack_id,
        )
    else:
        preview = repo.create_import_preview(
            owner_user_id=USER_ID, job_id="preview", status="completed", archive_path="staged.tldw-vnpack",
        )
        repo.create_import_journal(
            owner_user_id=USER_ID, preview_id=preview["id"], job_id="import", status="queued", stage="queued",
            trust_mode="trusted_restore", target_mode="update_existing",
        )

    response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/cleanup", json={"retention_only": True},
    )

    assert response.status_code == 200
    assert response.json()["retention_deferred"] is True
    assert response.json()["candidate_item_ids"] == []


def test_retention_mode_cannot_execute_cleanup(
    client: TestClient, asset_with_generated_file: SimpleNamespace, fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> None:
    response = client.post(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/cleanup",
        json={"dry_run": False, "retention_only": True, "idempotency_key": "retention-conflict"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "retention_preview_requires_dry_run"
    assert fake_generated_files_repo.hard_deleted_ids == []


def test_cleanup_replays_receipt_created_before_retention_preview(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> None:
    pack_id = asset_with_generated_file.pack_id
    payload = {"dry_run": False, "statuses": ["rejected"], "idempotency_key": "old-cleanup"}
    old_request = VNAssetCleanupRequest.model_validate(payload).model_dump(
        mode="json", exclude={"idempotency_key", "retention_only"},
    )
    VNAssetPacksRepository.initialized(chacha_db).create_idempotency_record(
        owner_user_id=USER_ID, scope="vn_asset_cleanup", resource_id=f"pack:{pack_id}",
        idempotency_key="old-cleanup", payload_hash=canonical_payload_hash({"pack_id": pack_id, "request": old_request}),
        response={"dry_run": False, "files_deleted": 2},
    )

    response = client.post(f"/api/v1/vn/vn-assets/packs/{pack_id}/cleanup", json=payload)

    assert response.status_code == 200
    assert response.json()["files_deleted"] == 2
    assert fake_generated_files_repo.hard_deleted_ids == []


def _write_orphan_scan_fixture(outputs_dir: Path, item_id: int, *, age_hours: int = 48) -> str:
    ref = f"vn_assets/2025/10/01/vn_asset_{item_id}_20251001_120000_abcdef012345.png"
    path = outputs_dir / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_BYTES)
    timestamp = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).timestamp()
    utime(path, (timestamp, timestamp))
    return ref


def test_orphan_report_pages_owned_files_without_flagging_live_or_trashed_paths(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
    outputs_dir: Path,
    tmp_path: Path,
) -> None:
    orphan = _write_orphan_scan_fixture(outputs_dir, 8001)
    _write_orphan_scan_fixture(outputs_dir, 8002, age_hours=1)
    trashed = _write_orphan_scan_fixture(outputs_dir, 8003)
    other_feature = _write_orphan_scan_fixture(outputs_dir, 8004)
    owner_item_path = _write_orphan_scan_fixture(outputs_dir, 8005)
    repo = VNAssetPacksRepository.initialized(chacha_db)
    repo.update_item_storage(
        asset_with_generated_file.item_id, storage_ref=owner_item_path,
        generated_file_id=asset_with_generated_file.file_id, mime_type="image/png",
        bytes=len(PNG_BYTES), width=None, height=None,
    )
    for file_id, user_id, storage_path, deleted in (
        (901, 43, orphan, False),
        (902, USER_ID, trashed.replace("/", "\\"), True),
        (903, USER_ID, other_feature, False),
    ):
        fake_generated_files_repo.records[file_id] = {
            "id": file_id, "user_id": user_id, "source_feature": "image_generation",
            "storage_path": storage_path, "is_deleted": deleted,
        }
    symlink = (outputs_dir / orphan).with_name("vn_asset_8006_20251001_120000_abcdef012345.png")
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(PNG_BYTES)
    symlink.symlink_to(outside)
    endpoint = "/api/v1/vn/vn-assets/storage/orphans"

    findings: list[dict[str, Any]] = []
    for offset, next_offset in ((0, 2), (2, 4), (4, None)):
        response = client.get(endpoint, params={"source": "files", "offset": offset, "limit": 2})
        assert response.status_code == 200
        assert response.json()["next_offset"] == next_offset
        findings.extend(response.json()["findings"])

    assert [(row["kind"], row["storage_ref"], row["bytes"]) for row in findings] == [
        ("unregistered_bytes", orphan, len(PNG_BYTES)),
    ]
    assert all((outputs_dir / path).is_file() for path in [orphan, trashed, other_feature, owner_item_path])
    assert fake_generated_files_repo.hard_deleted_ids == []


def test_orphan_report_separates_missing_and_orphaned_registrations(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
    outputs_dir: Path,
    character_id: int,
) -> None:
    ago = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    files = fake_generated_files_repo.records
    files[700]["created_at"] = ago
    (outputs_dir / files[700]["storage_path"]).unlink()
    orphan = _write_orphan_scan_fixture(outputs_dir, 8001)
    runtime = _write_orphan_scan_fixture(outputs_dir, 8002)
    for file_id, item_id, path in ((701, 8001, orphan), (702, 8002, runtime)):
        files[file_id] = {**files[700], "id": file_id, "source_ref": f"vn_asset_item:{item_id}", "storage_path": path}
    scripts = VNScriptsRepository.initialized(chacha_db)
    script = scripts.create_script(
        owner_user_id=USER_ID, title="Published route", primary_asset_pack_id=asset_with_generated_file.pack_id,
        policy_profile_id="local_default", generation_profile_id="story_default", status="published",
    )
    scripts.create_manifest_snapshot(
        owner_user_id=USER_ID, script_id=script["id"], asset_pack_id=asset_with_generated_file.pack_id, version_id=1,
        manifest={"assets": {"sprites": [{"generated_file_id": 702}]}}, manifest_hash="runtime-file",
    )
    repo = VNAssetPacksRepository.initialized(chacha_db)
    missing = repo.create_item(
        pack_id=asset_with_generated_file.pack_id, slot_id=asset_with_generated_file.slot_id,
        variant_index=1, generated_file_id=799,
    )
    trashed = repo.create_item(
        pack_id=asset_with_generated_file.pack_id, slot_id=asset_with_generated_file.slot_id,
        variant_index=2, generated_file_id=703,
    )
    files[703] = {**files[700], "id": 703, "source_ref": f"vn_asset_item:{trashed['id']}", "is_deleted": True}
    foreign = VNAssetPackService(chacha_db, owner_user_id=43).create_pack(
        VNAssetPackCreate(title="Other owner's pack", primary_character_id=character_id),
    )
    foreign_slot = repo.create_slot(pack_id=foreign.id, asset_type="sprite", slot_key="foreign")
    foreign_item = repo.create_item(pack_id=foreign.id, slot_id=foreign_slot["id"], generated_file_id=798)
    with chacha_db.transaction() as conn:
        conn.execute(
            "UPDATE vn_asset_items SET updated_at = ? WHERE id IN (?, ?, ?)",
            (ago, missing["id"], trashed["id"], foreign_item["id"]),
        )
    endpoint = "/api/v1/vn/vn-assets/storage/orphans"
    registrations = client.get(endpoint, params={"source": "registrations"})
    items = client.get(endpoint, params={"source": "items"})

    assert registrations.status_code == items.status_code == 200
    assert [(row["kind"], row["generated_file_id"]) for row in registrations.json()["findings"]] == [
        ("missing_bytes", 700), ("unreferenced_registration", 701),
    ]
    assert [(row["kind"], row["item_id"]) for row in items.json()["findings"]] == [
        ("missing_registration", missing["id"]),
    ]
    assert (outputs_dir / orphan).exists() and (outputs_dir / runtime).exists()
    assert set(files) == {700, 701, 702, 703}
    assert client.get(endpoint, params={"limit": 101}).status_code == 422
    assert client.get(endpoint, params={"offset": 2**63}).status_code == 422


def test_orphan_report_distinguishes_replaced_files_from_unfinished_item_bindings(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    outputs_dir: Path,
) -> None:
    repo = VNAssetPacksRepository.initialized(chacha_db)
    bound_item = asset_with_generated_file.item_id
    repo.update_item_storage(
        bound_item, generated_file_id=asset_with_generated_file.file_id, storage_ref="vn_assets/fixture.png",
        mime_type="image/png", width=None, height=None, bytes=len(PNG_BYTES),
    )
    replaced_path = _write_orphan_scan_fixture(outputs_dir, bound_item)
    pending = repo.create_item(
        pack_id=asset_with_generated_file.pack_id, slot_id=asset_with_generated_file.slot_id, variant_index=1,
    )
    pending_path = _write_orphan_scan_fixture(outputs_dir, pending["id"])
    published_path = _write_orphan_scan_fixture(outputs_dir, 8001)
    scripts = VNScriptsRepository.initialized(chacha_db)
    script = scripts.create_script(
        owner_user_id=USER_ID, title="Published route", primary_asset_pack_id=asset_with_generated_file.pack_id,
        policy_profile_id="local_default", generation_profile_id="story_default", status="published",
    )
    scripts.create_manifest_snapshot(
        owner_user_id=USER_ID, script_id=script["id"], asset_pack_id=asset_with_generated_file.pack_id,
        version_id=1, manifest={"assets": {"sprites": [{"storage_ref": published_path}]}}, manifest_hash="runtime-path",
    )
    play_path = _write_orphan_scan_fixture(outputs_dir, 8002)
    play = VNPlayRepository.initialized(chacha_db)
    session = play.create_session(
        owner_user_id=USER_ID, mode="freeform", title="Published scene",
        primary_character_id=repo.get_pack(asset_with_generated_file.pack_id)["primary_character_id"],
        vn_asset_pack_id=asset_with_generated_file.pack_id,
    )
    play.append_event(
        session_id=session["id"], owner_user_id=USER_ID, event_type="scene_updated",
        event_payload={"assets": [{"storage_ref": play_path}]},
    )

    response = client.get("/api/v1/vn/vn-assets/storage/orphans", params={"source": "files"})

    assert response.status_code == 200
    assert [row["storage_ref"] for row in response.json()["findings"]] == [replaced_path]
    assert (outputs_dir / pending_path).exists()
    assert (outputs_dir / published_path).exists()
    assert (outputs_dir / play_path).exists()


def test_orphan_report_defers_during_active_owner_storage_work(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    character_id: int,
) -> None:
    repo = VNAssetPacksRepository.initialized(chacha_db)
    other_pack = VNAssetPackService(chacha_db, owner_user_id=43).create_pack(
        VNAssetPackCreate(title="Other owner's pack", primary_character_id=character_id),
    )
    repo.create_batch(pack_id=other_pack.id, requested_by_user_id=43, status="processing")
    endpoint = "/api/v1/vn/vn-assets/storage/orphans"
    assert client.get(endpoint, params={"source": "files"}).json()["scan_deferred"] is False

    repo.create_batch(
        pack_id=asset_with_generated_file.pack_id, requested_by_user_id=USER_ID, status="enqueued",
    )

    response = client.get(endpoint, params={"source": "files"})

    assert response.status_code == 200
    assert response.json()["scan_deferred"] is True
    assert response.json()["findings"] == []


def test_storage_reports_use_scoped_jobs_for_bound_portability_activity(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    character_id: int,
) -> None:
    repo = VNAssetPacksRepository.initialized(chacha_db)
    other_pack = VNAssetPackService(chacha_db, owner_user_id=USER_ID).create_pack(
        VNAssetPackCreate(title="Second pack", primary_character_id=character_id),
    )
    jobs: list[dict[str, Any]] = []

    def list_jobs(*, owner_user_id: str, domain: str, job_type: str, status: str, limit: int) -> list[dict[str, Any]]:
        assert owner_user_id == str(USER_ID) and domain == "vn_assets"
        return [job for job in jobs if job["job_type"] == job_type and job["status"] == status][:limit]

    client.app.dependency_overrides[vn_assets_endpoint._job_manager] = lambda: SimpleNamespace(list_jobs=list_jobs)
    first_pack = asset_with_generated_file.pack_id
    repo.create_portability_job(
        owner_user_id=USER_ID, job_id="17", operation="export", status="processing", stage="archive", pack_id=first_pack,
    )
    root = "/api/v1/vn/vn-assets"
    assert client.get(f"{root}/storage/orphans").json()["scan_deferred"] is False

    jobs.append({
        "job_type": "vn_pack_import_commit", "status": "queued",
        "payload": {"target_mode": "update_existing", "target_pack_id": other_pack.id},
    })
    first = client.post(f"{root}/packs/{first_pack}/cleanup", json={"retention_only": True})
    second = client.post(f"{root}/packs/{other_pack.id}/cleanup", json={"retention_only": True})

    assert first.json()["retention_deferred"] is False
    assert second.json()["retention_deferred"] is True
    assert client.get(f"{root}/storage/orphans").json()["scan_deferred"] is True


def test_cleanup_endpoint_replays_same_idempotency_key_and_conflicts_on_different_payload(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    service.review_item(
        asset_with_generated_file.item_id,
        VNAssetReviewRequest(review_status="hidden"),
    )
    payload = {
        "dry_run": False,
        "statuses": ["hidden"],
        "idempotency_key": "cleanup-pack-1",
    }

    first = client.post(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/cleanup",
        json=payload,
    )
    replay = client.post(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/cleanup",
        json=payload,
    )
    conflict = client.post(
        f"/api/v1/vn/vn-assets/packs/{asset_with_generated_file.pack_id}/cleanup",
        json={**payload, "statuses": ["rejected"]},
    )

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert fake_generated_files_repo.hard_deleted_ids == [asset_with_generated_file.file_id]
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"


def test_upload_endpoint_creates_draft_uploaded_item(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    outputs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Storage import generated_file_helpers

    fake_service = FakeStorageService()
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    pack = service.create_pack(VNAssetPackCreate(title="Upload Pack", primary_character_id=character_id))
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]

    async def fake_get_storage_service() -> FakeStorageService:
        return fake_service

    monkeypatch.setattr(generated_file_helpers, "get_storage_service", fake_get_storage_service)

    response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/items/upload",
        data={
            "slot_id": str(slot.id),
            "variant_index": "0",
            "idempotency_key": "upload-create-1",
        },
        files={"file": ("sprite.png", VALID_PNG_BYTES, "image/png")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["pack_id"] == pack.id
    assert body["slot_id"] == slot.id
    assert body["variant_index"] == 0
    assert body["review_status"] == "draft"
    assert body["source"] == "uploaded"
    assert body["generated_file_id"] == 901
    assert body["width"] == 1
    assert body["height"] == 1
    assert fake_service.calls[0]["source_feature"] == "vn_assets"
    assert fake_service.calls[0]["source_ref"] == f"vn_asset_item:{body['id']}"
    assert (outputs_dir / fake_service.calls[0]["storage_path"]).read_bytes() == VALID_PNG_BYTES


def test_upload_endpoint_replays_same_idempotency_key_and_conflicts_on_different_file(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    outputs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Storage import generated_file_helpers

    fake_service = FakeStorageService()
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    pack = service.create_pack(VNAssetPackCreate(title="Upload Pack", primary_character_id=character_id))
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]

    async def fake_get_storage_service() -> FakeStorageService:
        return fake_service

    monkeypatch.setattr(generated_file_helpers, "get_storage_service", fake_get_storage_service)

    first = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/items/upload",
        data={"slot_id": str(slot.id), "variant_index": "0", "idempotency_key": "upload-item-1"},
        files={"file": ("sprite.png", VALID_PNG_BYTES, "image/png")},
    )
    replay = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/items/upload",
        data={"slot_id": str(slot.id), "variant_index": "0", "idempotency_key": "upload-item-1"},
        files={"file": ("sprite.png", VALID_PNG_BYTES, "image/png")},
    )
    conflict = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/items/upload",
        data={"slot_id": str(slot.id), "variant_index": "0", "idempotency_key": "upload-item-1"},
        files={"file": ("sprite.png", VALID_PNG_BYTES + b"changed", "image/png")},
    )

    assert first.status_code == 201
    assert replay.status_code == 201
    assert replay.json() == first.json()
    assert len(fake_service.calls) == 1
    assert len(service.repo.list_items(pack.id)) == 1
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_conflict"


def test_upload_endpoint_in_progress_key_does_not_release_existing_claim(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    pack = service.create_pack(VNAssetPackCreate(title="Upload Pack", primary_character_id=character_id))
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]
    idempotency_key = "upload-in-progress"
    payload_hash = canonical_multipart_payload_hash(
        {
            "pack_id": pack.id,
            "slot_id": slot.id,
            "variant_index": 0,
        },
        file_sha256=hashlib.sha256(VALID_PNG_BYTES).hexdigest(),
        file_size=len(VALID_PNG_BYTES),
        filename="sprite.png",
        content_type="image/png",
    )
    service.repo.claim_idempotency_record(
        owner_user_id=USER_ID,
        scope="vn_asset_item_upload",
        resource_id=f"pack:{pack.id}:slot:{slot.id}",
        idempotency_key=idempotency_key,
        payload_hash=payload_hash,
    )

    response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/items/upload",
        data={"slot_id": str(slot.id), "variant_index": "0", "idempotency_key": idempotency_key},
        files={"file": ("sprite.png", VALID_PNG_BYTES, "image/png")},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "idempotency_key_in_progress"
    record = service.repo.get_idempotency_record(
        owner_user_id=USER_ID,
        scope="vn_asset_item_upload",
        resource_id=f"pack:{pack.id}:slot:{slot.id}",
        idempotency_key=idempotency_key,
    )
    assert record is not None
    assert record["status"] == "in_progress"
    assert service.repo.list_items(pack.id) == []


def test_upload_endpoint_requires_idempotency_key(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    pack = service.create_pack(VNAssetPackCreate(title="Upload Pack", primary_character_id=character_id))
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]

    response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/items/upload",
        data={"slot_id": str(slot.id), "variant_index": "0"},
        files={"file": ("sprite.png", VALID_PNG_BYTES, "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "idempotency_key_required"


def test_upload_endpoint_rejects_oversized_asset_before_storage_registration(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    pack = service.create_pack(VNAssetPackCreate(title="Upload Pack", primary_character_id=character_id))
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]

    monkeypatch.setattr(
        vn_assets_endpoint,
        "_get_vn_asset_upload_max_bytes",
        lambda: len(VALID_PNG_BYTES) - 1,
        raising=False,
    )

    response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/items/upload",
        data={
            "slot_id": str(slot.id),
            "variant_index": "0",
            "idempotency_key": "upload-too-large-1",
        },
        files={"file": ("sprite.png", VALID_PNG_BYTES, "image/png")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "vn_asset_upload_too_large"
    assert service.repo.list_items(pack.id) == []


@pytest.mark.parametrize("variant_index", [-1, 1])
def test_upload_endpoint_rejects_invalid_variant_index(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    outputs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant_index: int,
) -> None:
    from tldw_Server_API.app.core.Storage import generated_file_helpers

    fake_service = FakeStorageService()
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    pack = service.create_pack(VNAssetPackCreate(title="Upload Pack", primary_character_id=character_id))
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]

    async def fake_get_storage_service() -> FakeStorageService:
        return fake_service

    monkeypatch.setattr(generated_file_helpers, "get_storage_service", fake_get_storage_service)

    response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack.id}/items/upload",
        data={
            "slot_id": str(slot.id),
            "variant_index": str(variant_index),
            "idempotency_key": f"upload-invalid-variant-{variant_index}",
        },
        files={"file": ("sprite.png", VALID_PNG_BYTES, "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_variant_index"
    assert service.repo.list_items(pack.id) == []
    assert fake_service.calls == []
    assert list(outputs_dir.glob("**/*")) == []


@pytest.mark.asyncio
async def test_upload_item_rolls_back_item_when_storage_registration_fails(
    chacha_db: CharactersRAGDB,
    character_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Storage import generated_file_helpers

    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    pack = service.create_pack(VNAssetPackCreate(title="Upload Pack", primary_character_id=character_id))
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]

    async def fail_register(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("storage registration failed")

    monkeypatch.setattr(
        generated_file_helpers,
        "save_and_register_vn_asset_image",
        fail_register,
    )

    with pytest.raises(RuntimeError, match="storage registration failed"):
        await service.upload_item(
            pack.id,
            slot_id=slot.id,
            image_bytes=VALID_PNG_BYTES,
            mime_type="image/png",
            variant_index=0,
        )

    assert service.repo.list_items(pack.id) == []


@pytest.mark.asyncio
async def test_cleanup_deletes_item_before_unlinking_file(
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.VN_Assets import service as service_module

    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    service.review_item(
        asset_with_generated_file.item_id,
        VNAssetReviewRequest(review_status="rejected"),
    )
    unlink_calls: list[str] = []

    def fake_unlink_vn_asset_storage_file(*, user_id: int, storage_path: str) -> bool:
        item = service.repo.get_item(asset_with_generated_file.item_id)
        assert item is None
        unlink_calls.append(storage_path)
        return True

    monkeypatch.setattr(
        service_module,
        "unlink_vn_asset_storage_file",
        fake_unlink_vn_asset_storage_file,
    )

    response = await service.cleanup_pack(
        asset_with_generated_file.pack_id,
        VNAssetCleanupRequest(dry_run=False, statuses=["rejected"]),
        files_repo=fake_generated_files_repo,
    )

    assert response.files_deleted == 1
    assert unlink_calls == ["vn_assets/fixture.png"]
    assert service.repo.get_item(asset_with_generated_file.item_id) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cleanup_completed_v1_preserves_ledger_and_removes_file(
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
    outputs_dir: Path,
) -> None:
    """Remove completed V1 storage while preserving a deletion receipt and counters.

    Args:
        chacha_db: Isolated native VN database.
        asset_with_generated_file: Owned asset and registered file identity.
        fake_generated_files_repo: Public storage boundary used by this control.
        outputs_dir: Private physical file root.

    Returns:
        None; asserts deletion, exact retained identity and unchanged counters.
    """
    asset = asset_with_generated_file
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    batch = service.repo.create_batch(
        pack_id=asset.pack_id, requested_by_user_id=USER_ID, total_variants=1,
        recipes=[{"slot_id": asset.slot_id, "variant_index": 0, "recipe": {"prompt": "frozen"}}],
    )
    service.repo.delete_item(asset.item_id)
    reserved = service.repo.reserve_variant_item(
        batch_id=batch["id"], slot_id=asset.slot_id, variant_index=0,
        item_fields={"pack_id": asset.pack_id, "generated_file_id": asset.file_id, "mime_type": "image/png"},
    )
    asset = SimpleNamespace(**{**vars(asset), "item_id": reserved["id"]})
    fake_generated_files_repo.records[asset.file_id]["source_ref"] = f"vn_asset_item:{asset.item_id}"
    service.repo.complete_variant(
        batch_id=batch["id"], slot_id=asset.slot_id, variant_index=0, item_id=asset.item_id,
    )
    service.review_item(asset.item_id, VNAssetReviewRequest(review_status="rejected"))
    before_recipe = service.repo.get_variant_outcome(batch["id"], asset.slot_id, 0)
    before_batch = service.repo.get_batch(batch["id"])
    unregistered: list[int] = []

    async def unregister(file_id: int, *, hard_delete: bool) -> bool:
        assert hard_delete is True
        unregistered.append(file_id)
        return await fake_generated_files_repo.hard_delete_file(file_id)

    response = await service.cleanup_pack(
        asset.pack_id, VNAssetCleanupRequest(dry_run=False, statuses=["rejected"]),
        files_repo=fake_generated_files_repo, unregister_generated_file=unregister,
    )

    assert response.removed_item_ids == [asset.item_id]
    assert response.files_deleted == 1
    assert response.reclaimed_bytes == len(PNG_BYTES)
    assert service.repo.get_item(asset.item_id) is None
    assert fake_generated_files_repo.records == {}
    assert unregistered == [asset.file_id]
    assert not (outputs_dir / "vn_assets/fixture.png").exists()
    after_recipe = service.repo.get_variant_outcome(batch["id"], asset.slot_id, 0)
    assert after_recipe == {**before_recipe, "item_id": None, "deleted_item_json": after_recipe["deleted_item_json"]}
    assert json.loads(after_recipe["deleted_item_json"]) == {
        "batch_id": batch["id"], "generated_file_id": asset.file_id, "id": asset.item_id,
        "owner_user_id": USER_ID, "pack_id": asset.pack_id, "slot_id": asset.slot_id, "variant_index": 0,
    }
    assert service.repo.get_batch(batch["id"]) == before_batch


@pytest.mark.asyncio
async def test_cleanup_offloads_storage_unlink_to_thread(
    chacha_db: CharactersRAGDB,
    asset_with_generated_file: SimpleNamespace,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.VN_Assets import service as service_module

    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    service.review_item(
        asset_with_generated_file.item_id,
        VNAssetReviewRequest(review_status="rejected"),
    )
    to_thread_calls: list[Any] = []

    class FakeAsyncio:
        async def to_thread(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
            to_thread_calls.append(func)
            return func(*args, **kwargs)

        async def gather(self, *awaitables: Any) -> list[Any]:
            return list(await real_asyncio.gather(*awaitables))

    monkeypatch.setattr(service_module, "asyncio", FakeAsyncio(), raising=False)

    response = await service.cleanup_pack(
        asset_with_generated_file.pack_id,
        VNAssetCleanupRequest(dry_run=False, statuses=["rejected"]),
        files_repo=fake_generated_files_repo,
    )

    assert response.files_deleted == 1
    assert service_module.unlink_vn_asset_storage_file in to_thread_calls
