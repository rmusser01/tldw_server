from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.main import app
import tldw_Server_API.app.api.v1.endpoints.setup as setup_endpoint


@contextmanager
def _make_client():
    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()


class _BundleCatalogStub:
    def __init__(self) -> None:
        self.bundles = [SimpleNamespace(bundle_id="cpu_local", model_dump=lambda: {"bundle_id": "cpu_local"})]


@pytest.fixture()
def _shared_audio_setup(monkeypatch):
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {"enabled": True, "needs_setup": False, "completed": True},
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
    monkeypatch.setattr(setup_endpoint, "get_audio_bundle_catalog", lambda: _BundleCatalogStub())
    monkeypatch.setattr(
        setup_endpoint.install_manager,
        "get_install_status_snapshot",
        lambda: {"status": "idle"},
    )
    monkeypatch.setattr(
        setup_endpoint.install_manager,
        "execute_audio_bundle",
        lambda bundle_id, resource_profile, tts_choice=None, safe_rerun=False: {
            "status": "completed",
            "bundle_id": bundle_id,
            "resource_profile": resource_profile,
            "tts_choice": tts_choice,
            "safe_rerun": safe_rerun,
        },
    )

    async def _fake_verify_audio_bundle_async(bundle_id, resource_profile, tts_choice=None):
        return {
            "status": "ready",
            "bundle_id": bundle_id,
            "resource_profile": resource_profile,
            "tts_choice": tts_choice,
        }

    monkeypatch.setattr(
        setup_endpoint.install_manager,
        "verify_audio_bundle_async",
        _fake_verify_audio_bundle_async,
    )

@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("get", "/api/v1/setup/admin/install-status", None),
        ("get", "/api/v1/setup/admin/audio/recommendations", None),
        (
            "post",
            "/api/v1/setup/admin/audio/provision",
            {"bundle_id": "cpu_local", "resource_profile": "balanced"},
        ),
        (
            "post",
            "/api/v1/setup/admin/audio/verify",
            {"bundle_id": "cpu_local", "resource_profile": "balanced"},
        ),
    ],
)
def test_shared_audio_installer_routes_require_admin(monkeypatch, _shared_audio_setup, method, path, json_body):
    async def fake_get_auth_principal(_request):
        return AuthPrincipal(kind="user", user_id=42, roles=["user"], permissions=[], is_admin=False)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_auth_principal",
        fake_get_auth_principal,
    )

    with _make_client() as client:
        request_kwargs = {"json": json_body} if json_body is not None else {}
        response = getattr(client, method)(path, **request_kwargs)

    assert response.status_code == 403


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("get", "/api/v1/setup/admin/install-status", None),
        ("get", "/api/v1/setup/admin/audio/recommendations", None),
        (
            "post",
            "/api/v1/setup/admin/audio/provision",
            {"bundle_id": "cpu_local", "resource_profile": "balanced"},
        ),
        (
            "post",
            "/api/v1/setup/admin/audio/verify",
            {"bundle_id": "cpu_local", "resource_profile": "balanced"},
        ),
    ],
)
def test_shared_audio_installer_routes_allow_admin(monkeypatch, _shared_audio_setup, method, path, json_body):
    async def fake_get_auth_principal(_request):
        return AuthPrincipal(
            kind="user",
            user_id=42,
            roles=["admin"],
            permissions=["system.configure"],
            is_admin=False,
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_auth_principal",
        fake_get_auth_principal,
    )

    with _make_client() as client:
        request_kwargs = {"json": json_body} if json_body is not None else {}
        response = getattr(client, method)(path, **request_kwargs)

    assert response.status_code == 200
