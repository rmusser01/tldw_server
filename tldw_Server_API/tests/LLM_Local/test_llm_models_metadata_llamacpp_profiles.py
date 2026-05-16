from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import llm_providers
from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import (
    LlamaCppAsset,
    LlamaCppAssetsResponse,
)
from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import (
    LlamaCppProfile,
    LlamaCppProfileMode,
)


class _SupervisorStub:
    def __init__(self, profiles: list[LlamaCppProfile]) -> None:
        self._profiles = profiles

    def list_profiles(self) -> list[LlamaCppProfile]:
        return list(self._profiles)


def _asset(asset_id: str, kind: str, path: Path) -> LlamaCppAsset:
    path.write_text(kind)
    return LlamaCppAsset(
        asset_id=asset_id,
        kind=kind,
        identity_basis="resolved_path",
        path=str(path),
        resolved_path=str(path.resolve()),
        display_name=path.name,
        source="models_dir",
        size_bytes=path.stat().st_size,
    )


def _client_for_profiles(
    monkeypatch,
    *,
    profiles: list[LlamaCppProfile],
    assets: list[LlamaCppAsset],
) -> TestClient:
    allowed_roots = [
        Path(asset.resolved_path).parent
        for asset in assets
        if asset.resolved_path
    ]

    async def _configured_providers(*_args, **_kwargs):
        return {"providers": [], "default_provider": None, "total_configured": 0}

    monkeypatch.setattr(llm_providers, "get_configured_providers_async", _configured_providers)
    monkeypatch.setattr(llm_providers, "apply_llm_provider_overrides_to_listing", lambda result: result)
    monkeypatch.setattr(llm_providers, "list_image_models_for_catalog", lambda: [])
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "scan_assets",
        lambda: LlamaCppAssetsResponse(assets=assets, warnings=[]),
    )
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "resolve_asset_path",
        lambda raw_path, **_kwargs: Path(raw_path).resolve(),
    )
    monkeypatch.setattr(
        llamacpp_inventory_service,
        "_allowed_bases_for_config",
        lambda _config: allowed_roots,
    )

    app = FastAPI()
    app.include_router(llm_providers.router, prefix="/api/v1")
    app.state.llm_manager = SimpleNamespace(llamacpp_supervisor=_SupervisorStub(profiles))
    return TestClient(app)


def test_models_metadata_includes_managed_llamacpp_profiles(monkeypatch, tmp_path):
    base = _asset("gguf:base", "gguf", tmp_path / "qwen-vl.gguf")
    mmproj = _asset("mmproj:projector", "mmproj", tmp_path / "mmproj-qwen-vl.gguf")
    profile = LlamaCppProfile(
        profile_id="vision",
        name="Vision profile",
        mode=LlamaCppProfileMode.VISION,
        model_id=base.asset_id,
        mmproj_model_id=mmproj.asset_id,
        provider_alias="llamacpp-vision",
    )

    with _client_for_profiles(monkeypatch, profiles=[profile], assets=[base, mmproj]) as client:
        response = client.get("/api/v1/llm/models/metadata")

    assert response.status_code == 200, response.text
    models = response.json()["models"]
    entry = next(item for item in models if item["llamacpp_profile_id"] == "vision")
    assert entry["provider"] == "llama.cpp"
    assert entry["model"] == "llamacpp-vision"
    assert entry["name"] == "Vision profile"
    assert entry["type"] == "chat"
    assert entry["source"] == "managed_llamacpp_profile"
    assert entry["capabilities"]["vision"] is True
    assert entry["modalities"]["input"] == ["text", "image"]
    assert entry["modalities"]["output"] == ["text"]
    assert entry["is_configured"] is True
    assert entry["catalog_only"] is False


def test_models_metadata_includes_disabled_profile_as_unconfigured(monkeypatch, tmp_path):
    base = _asset("gguf:base", "gguf", tmp_path / "disabled.gguf")
    profile = LlamaCppProfile(
        profile_id="disabled",
        name="Disabled profile",
        enabled=False,
        mode=LlamaCppProfileMode.CHAT,
        model_id=base.asset_id,
        provider_alias="llamacpp-disabled",
    )

    with _client_for_profiles(monkeypatch, profiles=[profile], assets=[base]) as client:
        response = client.get("/api/v1/llm/models/metadata")

    assert response.status_code == 200, response.text
    models = response.json()["models"]
    entry = next(item for item in models if item["llamacpp_profile_id"] == "disabled")
    assert entry["model"] == "llamacpp-disabled"
    assert entry["is_configured"] is False
    assert entry["provider_is_configured"] is False
    assert entry["catalog_only"] is False


def test_models_metadata_filters_managed_llamacpp_profiles(monkeypatch, tmp_path):
    base = _asset("gguf:base", "gguf", tmp_path / "chat.gguf")
    mmproj = _asset("mmproj:projector", "mmproj", tmp_path / "mmproj-chat.gguf")
    vision = LlamaCppProfile(
        profile_id="vision",
        name="Vision",
        mode=LlamaCppProfileMode.VISION,
        model_id=base.asset_id,
        mmproj_model_id=mmproj.asset_id,
        provider_alias="llamacpp-vision",
    )
    embedding = LlamaCppProfile(
        profile_id="embeddings",
        name="Embeddings",
        mode=LlamaCppProfileMode.EMBEDDING,
        model_id=base.asset_id,
        provider_alias="llamacpp-embeddings",
    )

    with _client_for_profiles(monkeypatch, profiles=[vision, embedding], assets=[base, mmproj]) as client:
        image_response = client.get("/api/v1/llm/models/metadata?type=chat&input_modality=image")
        embedding_response = client.get("/api/v1/llm/models/metadata?type=embedding")

    assert image_response.status_code == 200, image_response.text
    image_profile_ids = {
        item.get("llamacpp_profile_id")
        for item in image_response.json()["models"]
        if item.get("source") == "managed_llamacpp_profile"
    }
    assert image_profile_ids == {"vision"}

    assert embedding_response.status_code == 200, embedding_response.text
    embedding_profile_ids = {
        item.get("llamacpp_profile_id")
        for item in embedding_response.json()["models"]
        if item.get("source") == "managed_llamacpp_profile"
    }
    assert embedding_profile_ids == {"embeddings"}


def test_models_metadata_keeps_stale_llamacpp_profile_as_warning_entry(monkeypatch):
    profile = LlamaCppProfile(
        profile_id="stale",
        name="Stale local profile",
        mode=LlamaCppProfileMode.VISION,
        model_id="gguf:missing",
        mmproj_model_id="mmproj:missing",
        provider_alias="llamacpp-stale",
    )

    with _client_for_profiles(monkeypatch, profiles=[profile], assets=[]) as client:
        response = client.get("/api/v1/llm/models/metadata")

    assert response.status_code == 200, response.text
    models = response.json()["models"]
    entry = next(item for item in models if item["llamacpp_profile_id"] == "stale")
    assert entry["model"] == "llamacpp-stale"
    assert entry["capabilities"]["vision"] is False
    assert entry["capability_warnings"]
    assert "gguf:missing" in entry["capability_warnings"][0]
