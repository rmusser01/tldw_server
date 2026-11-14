"""Public VN lists enforce recipe publication independently of attached bytes."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import vn_assets
from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VNAssetBulkReviewRequest,
    VNAssetPackCreate,
    VNAssetReviewRequest,
    VNAssetSlotCreate,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import (
    chacha_db as chacha_db,
)
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import (
    character_id as character_id,
)
from tldw_Server_API.tests.VN_Assets.test_storage_cleanup import (
    PNG_BYTES,
    USER_ID,
    FakeGeneratedFilesRepo,
)
from tldw_Server_API.tests.VN_Assets.test_storage_cleanup import (
    fake_generated_files_repo as fake_generated_files_repo,
)
from tldw_Server_API.tests.VN_Assets.test_storage_cleanup import (
    outputs_dir as outputs_dir,
)

pytestmark = pytest.mark.integration
UNPUBLISHED = ("planned", "failed", "cancelled", "planned_unattached")


@dataclass
class VisibilityCase:
    """Native metadata and real bytes behind the service and authenticated router."""

    service: VNAssetPackService
    pack_id: int
    slot_id: int
    ids: dict[str, int]
    paths: list[Path]
    files: FakeGeneratedFilesRepo

    def snapshot(self) -> dict[str, Any]:
        """Read complete native rows and bytes without relying on list visibility."""
        repo = self.service.repo
        batches = repo.list_batches(self.pack_id)
        return {
            "pack": repo.get_pack(self.pack_id),
            "slots": repo.list_slots(self.pack_id),
            "items": [repo.get_item(item_id) for item_id in self.ids.values()],
            "batches": batches,
            "recipes": [repo.list_batch_recipes(batch["id"]) for batch in batches],
            "outcomes": [
                repo.get_variant_outcome(batch["id"], recipe["slot_id"], recipe["variant_index"])
                for batch in batches for recipe in repo.list_batch_recipes(batch["id"])
            ],
            "files": deepcopy(self.files.records),
            "bytes": [path.read_bytes() for path in self.paths],
        }


@pytest.fixture
def visibility_case(
    chacha_db: CharactersRAGDB,
    character_id: int,
    outputs_dir: Path,
    fake_generated_files_repo: FakeGeneratedFilesRepo,
) -> VisibilityCase:
    """Persist attached/nonattached outcomes through existing repository APIs."""
    service = VNAssetPackService(chacha_db, owner_user_id=USER_ID)
    repo = service.repo
    pack = service.create_pack(VNAssetPackCreate(title="Visibility", primary_character_id=character_id))
    slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="primary"))
    batch = repo.create_batch(
        pack_id=pack.id, requested_by_user_id=USER_ID, total_variants=5,
        recipes=[{"slot_id": slot.id, "variant_index": index, "recipe": {}} for index in range(5)],
    )
    ids: dict[str, int] = {}
    paths: list[Path] = []

    def storage_fields(name: str) -> dict[str, Any]:
        """Create actual bytes and owned generated-file metadata for one item."""
        file_id = 700 + len(paths)
        storage_path = f"vn_assets/{name}.png"
        path = outputs_dir / storage_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG_BYTES)
        paths.append(path)
        fake_generated_files_repo.records[file_id] = {
            "id": file_id, "user_id": USER_ID, "filename": path.name,
            "original_filename": path.name, "storage_path": storage_path,
            "mime_type": "image/png", "file_size_bytes": len(PNG_BYTES),
            "source_feature": "vn_assets", "is_deleted": False,
        }
        return {
            "generated_file_id": file_id, "storage_ref": storage_path,
            "mime_type": "image/png", "bytes": len(PNG_BYTES),
        }

    names = ("planned", "failed", "completed_approved", "planned_unattached", "completed_draft")
    for index, name in enumerate(names):
        fields = {} if name == "planned_unattached" else storage_fields(name)
        item = repo.reserve_variant_item(
            batch_id=batch["id"], slot_id=slot.id, variant_index=index,
            item_fields={"pack_id": pack.id, **fields},
        )
        ids[name] = item["id"]
    repo.fail_variant(batch_id=batch["id"], slot_id=slot.id, variant_index=1, error="model failed")
    for index, name in ((2, "completed_approved"), (4, "completed_draft")):
        repo.complete_variant(batch_id=batch["id"], slot_id=slot.id, variant_index=index, item_id=ids[name])
    service.review_item(ids["completed_approved"], VNAssetReviewRequest(review_status="approved", preferred=True))
    cancelled = repo.create_batch(
        pack_id=pack.id, requested_by_user_id=USER_ID, total_variants=1,
        recipes=[{"slot_id": slot.id, "variant_index": 5, "recipe": {}}],
    )
    ids["cancelled"] = repo.reserve_variant_item(
        batch_id=cancelled["id"], slot_id=slot.id, variant_index=5,
        item_fields={"pack_id": pack.id, **storage_fields("cancelled")},
    )["id"]
    repo.cancel_batch(cancelled["id"])
    for name, status, attached in (
        ("legacy_hidden_attached", "hidden", True),
        ("legacy_hidden_unattached", "hidden", False),
        ("legacy_draft_unattached", "draft", False),
    ):
        ids[name] = repo.create_item(
            pack_id=pack.id, slot_id=slot.id, review_status=status,
            **(storage_fields(name) if attached else {}),
        )["id"]
    for item_id in ids.values():
        item = repo.get_item(item_id)
        assert item is not None
        if item["generated_file_id"] is not None:
            fake_generated_files_repo.records[item["generated_file_id"]]["source_ref"] = f"vn_asset_item:{item_id}"
    return VisibilityCase(service, pack.id, slot.id, ids, paths, fake_generated_files_repo)


@pytest.fixture
def visibility_client(visibility_case: VisibilityCase) -> Iterator[TestClient]:
    """Use real HTTP handlers with the existing owner/service/storage dependencies."""
    app = FastAPI()
    app.include_router(vn_assets.router, prefix="/api/v1/vn")

    async def current_user() -> User:
        """Provide the actual pack owner to route authorization."""
        return User(id=USER_ID, username="visibility-owner")

    async def current_service() -> VNAssetPackService:
        """Share native rows and the public repository boundary with service tests."""
        return visibility_case.service

    async def current_files() -> FakeGeneratedFilesRepo:
        """Resolve the existing generated-file double backed by actual PNG bytes."""
        return visibility_case.files

    app.dependency_overrides[get_request_user] = current_user
    app.dependency_overrides[vn_assets._service] = current_service
    app.dependency_overrides[vn_assets._generated_files_repo] = current_files
    with TestClient(app) as client:
        yield client


def include_native_reservations(case: VisibilityCase, monkeypatch: pytest.MonkeyPatch) -> None:
    """Broaden only the public candidate seam using complete persisted item rows.

    Native list_items already excludes unfinished recipes. This independent
    service-contract probe keeps all native transitions/reads real and tests
    defense at the materialized candidate boundary, not a native query leak.
    """
    repo = case.service.repo
    native_list = repo.list_items

    def candidates(pack_id: int) -> list[dict[str, Any]]:
        """Return native visible rows plus actual linked unpublished reservations."""
        assert pack_id == case.pack_id
        rows = native_list(pack_id)
        for name in UNPUBLISHED:
            item = repo.get_item(case.ids[name])
            assert item is not None
            rows.append(item)
        return sorted(rows, key=lambda row: row["id"])

    monkeypatch.setattr(repo, "list_items", candidates)


@pytest.mark.parametrize("candidate_boundary", [False, True], ids=["native", "broader-candidates"])
@pytest.mark.parametrize("outcome", UNPUBLISHED)
def test_service_list_excludes_unpublished_items(
    visibility_case: VisibilityCase, monkeypatch: pytest.MonkeyPatch,
    candidate_boundary: bool, outcome: str,
) -> None:
    """File attachment cannot publish any unfinished linked recipe in the service."""
    if candidate_boundary:
        include_native_reservations(visibility_case, monkeypatch)
    listed = visibility_case.service.list_items(visibility_case.pack_id)
    assert visibility_case.ids[outcome] not in [item.id for item in listed]


@pytest.mark.parametrize("candidate_boundary", [False, True], ids=["native", "broader-candidates"])
@pytest.mark.parametrize("outcome", UNPUBLISHED)
def test_http_list_excludes_unpublished_items(
    visibility_case: VisibilityCase, visibility_client: TestClient,
    monkeypatch: pytest.MonkeyPatch, candidate_boundary: bool, outcome: str,
) -> None:
    """HTTP list responses cannot expose attached planned/failed/cancelled rows."""
    if candidate_boundary:
        include_native_reservations(visibility_case, monkeypatch)
    response = visibility_client.get(f"/api/v1/vn/vn-assets/packs/{visibility_case.pack_id}/items")
    assert response.status_code == 200
    assert visibility_case.ids[outcome] not in [item["id"] for item in response.json()]


@pytest.mark.parametrize("outcome", UNPUBLISHED)
def test_unpublished_list_review_and_file_access_agree(
    visibility_case: VisibilityCase, visibility_client: TestClient, outcome: str,
) -> None:
    """Unfinished items remain absent, unreviewable and inaccessible as files."""
    case = visibility_case
    item_id = case.ids[outcome]
    before = case.snapshot()
    assert item_id not in [item.id for item in case.service.list_items(case.pack_id)]
    with pytest.raises(ValueError, match="^item_not_found$"):
        case.service.get_item_for_pack(case.pack_id, item_id)
    with pytest.raises(ValueError, match="^item_not_found$"):
        case.service.review_item_for_pack(case.pack_id, item_id, VNAssetReviewRequest(review_status="approved"))
    with pytest.raises(ValueError, match="^item_not_found$"):
        case.service.bulk_review_items_for_pack(
            case.pack_id, VNAssetBulkReviewRequest(item_ids=[case.ids["completed_approved"], item_id], review_status="rejected"),
        )
    base = f"/api/v1/vn/vn-assets/packs/{case.pack_id}/items"
    assert visibility_client.patch(f"{base}/{item_id}/review", json={"review_status": "approved"}).status_code == 404
    assert visibility_client.post(f"{base}/bulk-review", json={
        "item_ids": [case.ids["completed_approved"], item_id], "review_status": "rejected",
    }).status_code == 404
    for kind in ("content", "preview"):
        response = visibility_client.get(f"{base}/{item_id}/{kind}")
        assert response.status_code == 404
        assert response.json()["detail"] == "item_not_found"
    assert case.snapshot() == before


@pytest.mark.parametrize("candidate_boundary", [False, True], ids=["native", "broader-candidates"])
@pytest.mark.parametrize("review_status", ["draft", "hidden", "approved"])
def test_completed_and_legacy_visibility_is_preserved(
    visibility_case: VisibilityCase, visibility_client: TestClient,
    monkeypatch: pytest.MonkeyPatch, candidate_boundary: bool, review_status: str,
) -> None:
    """Completed and unlinked visibility retains review decisions and real access."""
    case = visibility_case
    case.service.review_item(case.ids["completed_draft"], VNAssetReviewRequest(review_status=review_status))
    if candidate_boundary:
        include_native_reservations(case, monkeypatch)
    expected = [case.ids[name] for name in (
        "completed_approved", "completed_draft", "legacy_hidden_attached", "legacy_draft_unattached",
    )]
    before = case.snapshot()
    assert [item.id for item in case.service.list_items(case.pack_id)] == expected
    base = f"/api/v1/vn/vn-assets/packs/{case.pack_id}/items"
    response = visibility_client.get(base)
    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == expected
    for name in ("completed_approved", "completed_draft", "legacy_hidden_attached"):
        item_id = case.ids[name]
        assert case.service.get_item_for_pack(case.pack_id, item_id).id == item_id
        for kind in ("content", "preview"):
            content = visibility_client.get(f"{base}/{item_id}/{kind}")
            assert content.status_code == 200
            assert content.content == PNG_BYTES
    assert case.snapshot() == before


@pytest.mark.parametrize("candidate_boundary", [False, True], ids=["native", "broader-candidates"])
def test_list_filtering_preserves_native_state_and_bytes(
    visibility_case: VisibilityCase, visibility_client: TestClient,
    monkeypatch: pytest.MonkeyPatch, candidate_boundary: bool,
) -> None:
    """Repeated list reads never alter items, approvals, recipes, counters or bytes."""
    case = visibility_case
    if candidate_boundary:
        include_native_reservations(case, monkeypatch)
    before = case.snapshot()
    for _read in range(2):
        case.service.list_items(case.pack_id)
        assert visibility_client.get(f"/api/v1/vn/vn-assets/packs/{case.pack_id}/items").status_code == 200
    assert case.snapshot() == before
