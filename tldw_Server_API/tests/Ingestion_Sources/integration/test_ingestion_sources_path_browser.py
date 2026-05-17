from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def path_browser_client(monkeypatch, tmp_path):
    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep

    app = FastAPI()
    app.include_router(ep.router, prefix="/api/v1")
    app.dependency_overrides[ep.get_request_user] = lambda: SimpleNamespace(id=7)

    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    (allowed_root / "notes").mkdir()
    (allowed_root / "media").mkdir()
    (allowed_root / "readme.md").write_text("not a directory", encoding="utf-8")
    outside_root = tmp_path / "outside"
    outside_root.mkdir()

    monkeypatch.setattr(ep, "get_ingestion_source_allowed_roots", lambda: (allowed_root,))
    monkeypatch.setattr(
        ep,
        "_can_create_local_directory_ingestion_source_for_request",
        lambda *, current_user, request: True,
    )

    return {
        "client": TestClient(app),
        "allowed_root": allowed_root,
        "outside_root": outside_root,
        "endpoint_module": ep,
    }


@pytest.mark.integration
def test_browse_directories_lists_allowed_roots(path_browser_client):
    response = path_browser_client["client"].get("/api/v1/ingestion-sources/browse-directories")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["current_path"] is None
    assert payload["parent_path"] is None
    assert payload["roots"] == [
        {
            "name": path_browser_client["allowed_root"].name,
            "path": str(path_browser_client["allowed_root"]),
            "is_root": True,
        }
    ]
    assert payload["entries"] == payload["roots"]


@pytest.mark.integration
def test_browse_directories_lists_only_immediate_child_directories(path_browser_client):
    allowed_root = path_browser_client["allowed_root"]

    response = path_browser_client["client"].get(
        "/api/v1/ingestion-sources/browse-directories",
        params={"path": str(allowed_root)},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["current_path"] == str(allowed_root)
    assert payload["parent_path"] is None
    assert payload["entries"] == [
        {"name": "media", "path": str(allowed_root / "media"), "is_root": False},
        {"name": "notes", "path": str(allowed_root / "notes"), "is_root": False},
    ]


@pytest.mark.integration
def test_browse_directories_rejects_paths_outside_allowed_roots(path_browser_client):
    response = path_browser_client["client"].get(
        "/api/v1/ingestion-sources/browse-directories",
        params={"path": str(path_browser_client["outside_root"])},
    )

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Directory path is outside configured ingestion source roots"


@pytest.mark.integration
def test_browse_directories_rejects_file_paths(path_browser_client):
    response = path_browser_client["client"].get(
        "/api/v1/ingestion-sources/browse-directories",
        params={"path": str(path_browser_client["allowed_root"] / "readme.md")},
    )

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Directory path is not a readable directory"


@pytest.mark.integration
def test_browse_directories_requires_local_directory_entitlement(path_browser_client, monkeypatch):
    ep = path_browser_client["endpoint_module"]
    monkeypatch.setattr(
        ep,
        "_can_create_local_directory_ingestion_source_for_request",
        lambda *, current_user, request: False,
    )

    response = path_browser_client["client"].get("/api/v1/ingestion-sources/browse-directories")

    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "Local directory ingestion sources are not enabled for this user"
