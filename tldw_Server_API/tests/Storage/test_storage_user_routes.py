"""Route-level tests for user storage folders/files/trash (audit F2)."""
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import storage as storage_endpoints


def _app(mock_user) -> FastAPI:
    app = FastAPI()
    app.include_router(storage_endpoints.router, prefix="/api/v1")
    app.dependency_overrides[storage_endpoints.get_request_user] = lambda: mock_user
    return app


@pytest.fixture
def client(mock_user, mock_storage_service, monkeypatch):
    monkeypatch.setattr(
        storage_endpoints, "_get_service", AsyncMock(return_value=mock_storage_service)
    )
    with TestClient(_app(mock_user)) as c:
        yield c


class TestFolderRoutes:
    @pytest.mark.unit
    def test_create_folder_returns_normalized_tag(self, client):
        resp = client.post("/api/v1/storage/folders", json={"name": "reports"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["folder_tag"] == "reports"

    @pytest.mark.unit
    def test_create_folder_empty_name_is_422(self, client):
        # FolderCreateRequest enforces min_length=1 at schema level
        resp = client.post("/api/v1/storage/folders", json={"name": ""})
        assert resp.status_code == 422

    @pytest.mark.unit
    def test_list_folders_empty(self, client):
        resp = client.get("/api/v1/storage/folders")
        assert resp.status_code == 200
        assert resp.json()["folders"] == []


class TestFileMutationRoutes:
    @pytest.mark.unit
    def test_patch_unknown_file_is_404(self, client):
        resp = client.patch("/api/v1/storage/files/999999", json={"folder_tag": "archive"})
        assert resp.status_code == 404

    @pytest.mark.unit
    def test_delete_unknown_file_is_404(self, client):
        resp = client.delete("/api/v1/storage/files/999999")
        assert resp.status_code == 404


class TestTrashMutationRoutes:
    @pytest.mark.unit
    def test_restore_unknown_item_is_404(self, client):
        resp = client.post("/api/v1/storage/trash/restore/999999")
        assert resp.status_code == 404

    @pytest.mark.unit
    def test_restore_other_users_file_is_403(self, client, mock_files_repo):
        mock_files_repo.get_file_by_id = AsyncMock(
            return_value={"id": 5, "user_id": 999, "is_deleted": True}
        )
        resp = client.post("/api/v1/storage/trash/restore/5")
        assert resp.status_code == 403

    @pytest.mark.unit
    def test_restore_not_deleted_file_is_400(self, client, mock_files_repo):
        mock_files_repo.get_file_by_id = AsyncMock(
            return_value={"id": 5, "user_id": 1, "is_deleted": False}
        )
        resp = client.post("/api/v1/storage/trash/restore/5")
        assert resp.status_code == 400

    @pytest.mark.unit
    def test_permanent_delete_unknown_item_is_404(self, client):
        resp = client.delete("/api/v1/storage/trash/999999")
        assert resp.status_code == 404

    @pytest.mark.unit
    def test_permanent_delete_other_users_file_is_403(self, client, mock_files_repo):
        mock_files_repo.get_file_by_id = AsyncMock(
            return_value={"id": 5, "user_id": 999, "is_deleted": True}
        )
        resp = client.delete("/api/v1/storage/trash/5")
        assert resp.status_code == 403
