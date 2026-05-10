from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.vn_assets import router as vn_assets_router
from tldw_Server_API.app.api.v1.endpoints.vn_capabilities import router as vn_capabilities_router
from tldw_Server_API.app.api.v1.endpoints.vn_play import router as vn_play_router

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(vn_capabilities_router, prefix="/api/v1/vn")
    app.include_router(vn_assets_router, prefix="/api/v1/vn")
    app.include_router(vn_play_router, prefix="/api/v1/vn")
    return TestClient(app)


def test_vn_capabilities_returns_canonical_paths(client: TestClient) -> None:
    response = client.get("/api/v1/vn/vn-capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "vn_capabilities.v1"
    assert body["base_path"] == "/api/v1/vn"
    assert body["resources"]["assets"] == "/api/v1/vn/vn-assets"
    assert body["resources"]["scripts"] == "/api/v1/vn/vn-scripts"
    assert body["resources"]["play"] == "/api/v1/vn/vn-play"
    assert body["resources"]["policy"] == "/api/v1/vn/vn-policy"
    assert body["resources"]["audio"] == "/api/v1/vn/vn-audio"
    assert body["enabled_modules"]["assets"] is True
    assert body["enabled_modules"]["play"] is True
    assert body["enabled_modules"]["scripts"] is False
    assert body["enabled_modules"]["policy"] is False
    assert body["enabled_modules"]["audio"] is False
    assert body["features"]["asset_generation"] is True
    assert body["features"]["asset_portability"] is True
    assert body["features"]["realtime_image_generation"] is False
    assert body["route_migration"]["canonical"] == "/api/v1/vn/vn-*"
    assert body["route_migration"]["supersedes"] == ["/api/v1/vn-assets", "/api/v1/vn-play"]
    assert "visible_policy_profiles" in body
    assert "visible_generation_profiles" in body
    assert "image/png" in body["supported_media_types"]["image"]
    assert "audio/mpeg" in body["supported_media_types"]["audio"]
