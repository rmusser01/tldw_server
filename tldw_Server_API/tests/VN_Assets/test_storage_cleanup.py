from __future__ import annotations

import asyncio as real_asyncio
import hashlib
from datetime import datetime, timezone
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
from tldw_Server_API.app.core.DB_Management.VNScripts_DB import VNScriptsRepository
from tldw_Server_API.app.core.VN_Assets.cleanup_blockers import VNAssetCleanupBlockerProvider
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Platform.idempotency import canonical_multipart_payload_hash

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

    async def register_generated_file(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"id": 901, **kwargs}


class FakeGeneratedFilesRepo:
    def __init__(self) -> None:
        self.records: dict[int, dict[str, Any]] = {}
        self.accessed_ids: list[int] = []
        self.hard_deleted_ids: list[int] = []

    async def get_file_by_id(self, file_id: int) -> dict[str, Any] | None:
        return self.records.get(file_id)

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
