from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.main import app
import tldw_Server_API.app.api.v1.endpoints.setup as setup_endpoint
from tldw_Server_API.app.core.Setup.readiness_store import SetupReadinessStore


def _make_client() -> TestClient:
    return TestClient(app)


class _EmptyBundleCatalog:
    bundles: list[SimpleNamespace] = []


@pytest.fixture()
def _readiness_api_setup(monkeypatch, tmp_path):
    store = SetupReadinessStore(tmp_path / "setup_readiness.json")
    monkeypatch.setattr(setup_endpoint.readiness_store, "get_setup_readiness_store", lambda: store)
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
    return store


@pytest.fixture()
def _admin_readiness_api_setup(monkeypatch, tmp_path):
    store = SetupReadinessStore(tmp_path / "setup_readiness.json")
    monkeypatch.setattr(setup_endpoint.readiness_store, "get_setup_readiness_store", lambda: store)
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {
            "enabled": False,
            "setup_completed": True,
            "completed": True,
            "needs_setup": False,
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
                    "fields": [{"key": "default_api", "value": "openai", "placeholder": False}],
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
    return store


async def _admin_principal(_request):
    return AuthPrincipal(
        kind="user",
        user_id=42,
        roles=["admin"],
        permissions=["system.configure"],
        is_admin=False,
    )


async def _regular_user_principal(_request):
    return AuthPrincipal(kind="user", user_id=42, roles=["user"], permissions=[], is_admin=False)


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


def test_readiness_provision_returns_pollable_status_without_waiting_for_download(
    monkeypatch,
    _readiness_api_setup,
):
    config_updates: list[dict[str, dict[str, str]]] = []
    install_plans: list[dict[str, object]] = []

    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "update_config",
        lambda updates: config_updates.append(updates),
    )
    monkeypatch.setattr(setup_endpoint, "execute_install_plan", lambda plan: install_plans.append(plan))

    client = _make_client()
    preview_response = client.post(
        "/api/v1/setup/readiness/preview",
        headers={"host": "localhost"},
        json={
            "profile_id": "advanced_custom",
            "lanes": {
                "chat": {"mode": "skip"},
                "embeddings_rag": {
                    "mode": "local",
                    "provider": "huggingface",
                    "model": "Qwen/Qwen3-Embedding-0.6B",
                },
            },
        },
    )
    assert preview_response.status_code == 200

    response = client.post(
        "/api/v1/setup/readiness/provision",
        headers={"host": "localhost"},
        json={"preview_id": preview_response.json()["preview_id"], "confirmed": True},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status_url"] == "/api/v1/setup/readiness/status"
    assert payload["install_plan_submitted"] is True
    assert config_updates == [
        {
            "Embeddings": {
                "embedding_provider": "huggingface",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            }
        }
    ]
    assert install_plans[0]["embeddings"]["huggingface"] == ["Qwen/Qwen3-Embedding-0.6B"]

    status_response = client.get("/api/v1/setup/readiness/status", headers={"host": "localhost"})
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["operation_id"] == payload["operation_id"]
    assert status_payload["operation_status"] == "queued"


def test_readiness_provision_requires_explicit_confirmation(_readiness_api_setup):
    client = _make_client()
    response = client.post(
        "/api/v1/setup/readiness/provision",
        headers={"host": "localhost"},
        json={"preview_id": "preview-1", "confirmed": False},
    )

    assert response.status_code == 400


def test_readiness_verify_route_persists_speech_warnings(monkeypatch, _readiness_api_setup):
    async def fake_verify(bundle_id, resource_profile, tts_choice=None):
        return {
            "status": "ready",
            "stt_health": {"status": "ready"},
            "tts_health": {"status": "failed"},
        }

    monkeypatch.setattr(setup_endpoint.install_manager, "verify_audio_bundle_async", fake_verify)

    client = _make_client()
    response = client.post(
        "/api/v1/setup/readiness/verify",
        headers={"host": "localhost"},
        json={
            "selection": {
                "profile_id": "advanced_custom",
                "lanes": {
                    "speech": {
                        "bundle_id": "cpu_local",
                        "resource_profile": "balanced",
                    }
                },
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready_with_warnings"
    assert payload["lanes"]["speech"]["status"] == "ready_with_warnings"

    status_response = client.get("/api/v1/setup/readiness/status", headers={"host": "localhost"})
    assert status_response.status_code == 200
    assert status_response.json()["readiness_status"] == "ready_with_warnings"


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("get", "/api/v1/setup/admin/readiness/profiles", None),
        ("get", "/api/v1/setup/admin/readiness/status", None),
        ("post", "/api/v1/setup/admin/readiness/preview", {"lanes": {"chat": {"mode": "skip"}}}),
        ("post", "/api/v1/setup/admin/readiness/provision", {"confirmed": False}),
        ("post", "/api/v1/setup/admin/readiness/verify", {"selection": {"lanes": {"chat": {"mode": "skip"}}}}),
    ],
)
def test_admin_readiness_routes_require_admin(
    monkeypatch,
    _admin_readiness_api_setup,
    method,
    path,
    json_body,
):
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_auth_principal",
        _regular_user_principal,
    )

    client = _make_client()
    request_kwargs = {"json": json_body} if json_body is not None else {}
    response = getattr(client, method)(path, **request_kwargs)

    assert response.status_code == 403


def test_admin_readiness_profiles_available_after_setup_completed(monkeypatch, _admin_readiness_api_setup):
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_auth_principal",
        _admin_principal,
    )

    client = _make_client()
    response = client.get("/api/v1/setup/admin/readiness/profiles")

    assert response.status_code == 200
    payload = response.json()
    assert payload["setup_access"]["mode"] == "admin"
    assert [lane["lane_id"] for lane in payload["lanes"]] == ["chat", "embeddings_rag", "speech"]
