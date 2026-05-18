from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
import tldw_Server_API.app.api.v1.endpoints.setup as setup_endpoint


def _make_client() -> TestClient:
    return TestClient(app)


class _EmptyBundleCatalog:
    bundles: list[SimpleNamespace] = []


@pytest.fixture()
def _readiness_api_setup(monkeypatch):
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {
            "enabled": True,
            "setup_completed": False,
            "needs_setup": True,
            "config_path": "config.txt",
            "allow_remote_setup_access": False,
            "remote_access_env_override": False,
            "remote_access_active": False,
            "placeholder_fields": [],
        },
    )
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_config_snapshot",
        lambda: {
            "config_path": "config.txt",
            "sections": [
                {
                    "name": "API",
                    "fields": [
                        {"key": "default_api", "value": "openai", "placeholder": False},
                        {"key": "openai_api_key", "value": "", "placeholder": True},
                    ],
                },
                {
                    "name": "Embeddings",
                    "fields": [
                        {"key": "embedding_provider", "value": "huggingface", "placeholder": False},
                        {
                            "key": "embedding_model",
                            "value": "Qwen/Qwen3-Embedding-0.6B",
                            "placeholder": False,
                        },
                    ],
                },
            ],
        },
    )
    monkeypatch.setattr(
        setup_endpoint.audio_profile_service,
        "detect_machine_profile",
        lambda: {
            "platform": "linux",
            "arch": "x86_64",
            "apple_silicon": False,
            "cuda_available": False,
            "ffmpeg_available": True,
            "espeak_available": True,
            "free_disk_gb": 64.0,
            "network_available_for_downloads": True,
        },
    )
    monkeypatch.setattr(
        setup_endpoint.audio_profile_service,
        "recommend_audio_bundles",
        lambda *args, **kwargs: {"recommendations": [], "excluded": []},
    )
    monkeypatch.setattr(setup_endpoint, "get_audio_bundle_catalog", lambda: _EmptyBundleCatalog())


def test_readiness_profiles_available_during_local_first_run(_readiness_api_setup):
    client = _make_client()
    response = client.get("/api/v1/setup/readiness/profiles", headers={"host": "localhost"})

    assert response.status_code == 200
    assert [lane["lane_id"] for lane in response.json()["lanes"]] == ["chat", "embeddings_rag", "speech"]
    assert response.json()["setup_access"]["mode"] == "first_run"


def test_readiness_status_reports_overlays_separately(_readiness_api_setup):
    client = _make_client()
    response = client.get("/api/v1/setup/readiness/status", headers={"host": "localhost"})

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["overlays"], list)
    assert all(lane["status"] != "restart_required" for lane in payload["lanes"])


def test_readiness_preview_route_does_not_write_or_echo_secret(monkeypatch, _readiness_api_setup):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("readiness preview route must not write config")

    monkeypatch.setattr(setup_endpoint.setup_manager, "update_config", fail_if_called)

    client = _make_client()
    response = client.post(
        "/api/v1/setup/readiness/preview",
        headers={"host": "localhost"},
        json={
            "profile_id": "advanced_custom",
            "lanes": {
                "chat": {
                    "mode": "hosted",
                    "provider": "openai",
                    "api_key": "sk-sensitive",
                    "model": "gpt-4.1-mini",
                }
            },
        },
    )

    assert response.status_code == 200
    assert called is False
    payload = response.json()
    assert "sk-sensitive" not in str(payload)
    assert payload["secret_fields"][0]["state"] == "submitted"
