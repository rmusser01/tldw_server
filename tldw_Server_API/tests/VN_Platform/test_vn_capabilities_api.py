from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.vn_assets import router as vn_assets_router
from tldw_Server_API.app.api.v1.endpoints.vn_capabilities import router as vn_capabilities_router
from tldw_Server_API.app.api.v1.endpoints.vn_play import router as vn_play_router
from tldw_Server_API.app.api.v1.endpoints.vn_policy import router as vn_policy_router
from tldw_Server_API.app.api.v1.endpoints.vn_scripts import router as vn_scripts_router

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(vn_capabilities_router, prefix="/api/v1/vn")
    app.include_router(vn_assets_router, prefix="/api/v1/vn")
    app.include_router(vn_scripts_router, prefix="/api/v1/vn")
    app.include_router(vn_play_router, prefix="/api/v1/vn")
    app.include_router(vn_policy_router, prefix="/api/v1/vn")
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
    assert body["enabled_modules"]["scripts"] is True
    assert body["enabled_modules"]["policy"] is True
    assert body["enabled_modules"]["audio"] is False
    assert body["features"]["asset_generation"] is True
    assert body["features"]["asset_portability"] is True
    assert body["features"]["scripted_story"] is True
    assert body["features"]["scripted_generation"] is True
    assert body["features"]["scripted_generation_confirmation"] is True
    assert body["features"]["scripted_generation_revision_activation"] is True
    assert body["features"]["scripted_generation_history"] is True
    assert body["features"]["scripted_generation_debug_detail"] is True
    assert body["features"]["tts_jobs"] is False
    assert body["features"]["realtime_image_generation"] is False
    assert body["limits"]["max_automatic_generation_batch_count"] == 1
    assert body["scripted_generation"]["enabled"] is True
    assert body["scripted_generation"]["output_schemas"] == [
        "narrative_dialogue",
        "choice_set",
        "scene_update",
    ]
    assert body["scripted_generation"]["confirmation_supported"] is True
    assert body["scripted_generation"]["revision_activation_supported"] is True
    assert body["scripted_generation"]["history_supported"] is True
    assert body["scripted_generation"]["debug_detail_supported"] is True
    assert body["scripted_generation"]["dynamic_choice_supported"] is True
    assert body["scripted_generation"]["scene_update_supported"] is True
    assert "suggestive" in body["supported_content_ratings"]
    assert body["route_migration"]["canonical"] == "/api/v1/vn/vn-*"
    assert body["route_migration"]["supersedes"] == ["/api/v1/vn-assets", "/api/v1/vn-play"]
    assert "visible_policy_profiles" in body
    assert "visible_generation_profiles" in body
    assert "image/png" in body["supported_media_types"]["image"]
    assert "audio/mpeg" in body["supported_media_types"]["audio"]


def test_vn_capabilities_disable_scripted_generation_details_without_scripts() -> None:
    app = FastAPI()
    app.include_router(vn_capabilities_router, prefix="/api/v1/vn")
    app.include_router(vn_play_router, prefix="/api/v1/vn")
    response = TestClient(app).get("/api/v1/vn/vn-capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["enabled_modules"]["play"] is True
    assert body["enabled_modules"]["scripts"] is False
    assert body["features"]["scripted_generation"] is False
    assert body["scripted_generation"]["enabled"] is False
    assert body["scripted_generation"]["confirmation_supported"] is False
    assert body["scripted_generation"]["dynamic_choice_supported"] is False
    assert body["scripted_generation"]["scene_update_supported"] is False
    assert body["scripted_generation"]["moderation_blocked_raw_reveal_supported"] is False
