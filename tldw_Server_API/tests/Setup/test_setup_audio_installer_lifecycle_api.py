from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Setup.audio_bundle_catalog import get_audio_bundle_catalog
from tldw_Server_API.app.core.Setup.audio_profile_service import MachineProfile
import tldw_Server_API.app.api.v1.endpoints.setup as setup_endpoint


def _make_client():
    # Scope the context-managed client to setup routes; full-app lifespan starts unrelated workers.
    test_app = FastAPI()
    test_app.include_router(setup_endpoint.router, prefix="/api/v1")
    return TestClient(test_app)


def _request(method: str, path: str, **kwargs):
    with _make_client() as client:
        return getattr(client, method)(path, **kwargs)


class _BundleCatalogStub:
    def __init__(self) -> None:
        catalog = get_audio_bundle_catalog()
        self.bundles = [
            SimpleNamespace(
                bundle_id="cpu_local",
                model_dump=lambda: catalog.bundle_by_id("cpu_local").model_dump(),
            )
        ]


@pytest.fixture()
def _admin_audio_installer_setup(monkeypatch):
    captured = {}

    async def fake_get_auth_principal(_request):
        return AuthPrincipal(
            kind="user",
            user_id=7,
            roles=["admin"],
            permissions=["system.configure"],
            is_admin=False,
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_auth_principal",
        fake_get_auth_principal,
    )
    monkeypatch.setattr(
        setup_endpoint.audio_profile_service,
        "detect_machine_profile",
        lambda: MachineProfile(
            platform="darwin",
            arch="arm64",
            apple_silicon=True,
            cuda_available=False,
            ffmpeg_available=True,
            espeak_available=True,
            free_disk_gb=64.0,
            network_available_for_downloads=True,
        ),
    )
    monkeypatch.setattr(
        setup_endpoint.audio_profile_service,
        "recommend_audio_bundles",
        lambda *args, **kwargs: {
            "recommendations": [
                {
                    "bundle_id": "cpu_local",
                    "resource_profile": "balanced",
                    "selection_key": "v2:cpu_local:balanced",
                }
            ],
            "excluded": [],
        },
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
        lambda bundle_id, resource_profile, safe_rerun=False, tts_choice=None: {
            "status": "completed",
            "bundle_id": bundle_id,
            "resource_profile": resource_profile,
            "safe_rerun": safe_rerun,
            "tts_choice": tts_choice,
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

    return captured


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
def test_admin_audio_installer_routes_remain_available_after_setup_completed(
    monkeypatch,
    _admin_audio_installer_setup,
    method,
    path,
    json_body,
):
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {"enabled": False, "setup_completed": True, "needs_setup": False},
    )

    request_kwargs = {"json": json_body} if json_body is not None else {}
    response = _request(method, path, **request_kwargs)

    assert response.status_code == 200


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
def test_admin_audio_installer_routes_stay_unavailable_without_setup_or_completion(
    monkeypatch,
    _admin_audio_installer_setup,
    method,
    path,
    json_body,
):
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {"enabled": False, "setup_completed": False, "needs_setup": False},
    )

    request_kwargs = {"json": json_body} if json_body is not None else {}
    response = _request(method, path, **request_kwargs)

    assert response.status_code == 404


def test_admin_audio_recommendations_include_curated_tts_choices(
    monkeypatch,
    _admin_audio_installer_setup,
):
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {"enabled": True, "setup_completed": False, "needs_setup": True},
    )

    response = _request("get", "/api/v1/setup/admin/audio/recommendations")

    assert response.status_code == 200
    payload = response.json()
    profile = payload["recommendations"][0]["profile"]
    assert profile["default_tts_choice"] == "kokoro"
    assert {choice["choice_id"] for choice in profile["tts_choices"]} == {"kokoro", "kitten_tts"}


def test_admin_audio_provision_and_verify_accept_tts_choice(
    monkeypatch,
    _admin_audio_installer_setup,
):
    captured = {}

    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {"enabled": True, "setup_completed": False, "needs_setup": True},
    )
    async def _fake_execute_audio_bundle_provision(payload, allow_completed_when_disabled=False):
        return {
            "status": "completed",
            "bundle_id": payload.bundle_id,
            "resource_profile": payload.resource_profile,
            "tts_choice": payload.tts_choice,
        }

    monkeypatch.setattr(
        setup_endpoint,
        "_execute_audio_bundle_provision",
        _fake_execute_audio_bundle_provision,
    )

    async def _fake_execute_audio_bundle_verification(payload, allow_completed_when_disabled=False):
        captured["verify_tts_choice"] = payload.tts_choice
        return {
            "status": "ready",
            "bundle_id": payload.bundle_id,
            "resource_profile": payload.resource_profile,
            "tts_choice": payload.tts_choice,
        }

    monkeypatch.setattr(
        setup_endpoint,
        "_execute_audio_bundle_verification",
        _fake_execute_audio_bundle_verification,
    )

    with _make_client() as client:
        provision_response = client.post(
            "/api/v1/setup/admin/audio/provision",
            json={
                "bundle_id": "cpu_local",
                "resource_profile": "balanced",
                "tts_choice": "kitten_tts",
            },
        )
        verify_response = client.post(
            "/api/v1/setup/admin/audio/verify",
            json={
                "bundle_id": "cpu_local",
                "resource_profile": "balanced",
                "tts_choice": "kitten_tts",
            },
        )

    assert provision_response.status_code == 200
    assert verify_response.status_code == 200
    assert provision_response.json()["tts_choice"] == "kitten_tts"
    assert verify_response.json()["tts_choice"] == "kitten_tts"
    assert captured["verify_tts_choice"] == "kitten_tts"


def test_setup_complete_accepts_direct_omnivoice_install_plan(monkeypatch):
    install_calls = []

    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {"enabled": True, "setup_completed": False, "needs_setup": True},
    )
    monkeypatch.setattr(setup_endpoint.setup_manager, "mark_setup_completed", lambda _value: None)
    monkeypatch.setattr(setup_endpoint.setup_manager, "update_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        setup_endpoint,
        "execute_install_plan",
        lambda payload: install_calls.append(payload),
    )

    response = _request(
        "post",
        "/api/v1/setup/complete",
        json={
            "disable_first_time_setup": False,
            "install_plan": {
                "stt": [],
                "tts": [{"engine": "omnivoice"}],
                "embeddings": {"huggingface": [], "custom": [], "onnx": []},
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["install_plan_submitted"] is True
    assert install_calls == [
        {
            "stt": [],
            "tts": [{"engine": "omnivoice", "variants": []}],
            "embeddings": {"huggingface": [], "custom": [], "onnx": []},
        }
    ]


def test_admin_audio_provision_rejects_invalid_tts_choice_with_400(
    monkeypatch,
    _admin_audio_installer_setup,
):
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {"enabled": True, "setup_completed": False, "needs_setup": True},
    )
    monkeypatch.setattr(
        setup_endpoint.install_manager,
        "execute_audio_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Unknown curated TTS choice 'bogus_choice'")),
    )

    response = _request(
        "post",
        "/api/v1/setup/admin/audio/provision",
        json={
            "bundle_id": "cpu_local",
            "resource_profile": "balanced",
            "tts_choice": "bogus_choice",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == setup_endpoint.INVALID_AUDIO_BUNDLE_REQUEST_DETAIL


def test_admin_audio_verify_rejects_invalid_tts_choice_with_400(
    monkeypatch,
    _admin_audio_installer_setup,
):
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {"enabled": True, "setup_completed": False, "needs_setup": True},
    )

    async def _raise_invalid_choice(*args, **kwargs):
        raise ValueError("Unknown curated TTS choice 'bogus_choice'")

    monkeypatch.setattr(
        setup_endpoint.install_manager,
        "verify_audio_bundle_async",
        _raise_invalid_choice,
    )

    response = _request(
        "post",
        "/api/v1/setup/admin/audio/verify",
        json={
            "bundle_id": "cpu_local",
            "resource_profile": "balanced",
            "tts_choice": "bogus_choice",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == setup_endpoint.INVALID_AUDIO_BUNDLE_REQUEST_DETAIL


def test_audio_pack_export_rejects_invalid_tts_choice_with_400(
    monkeypatch,
    _admin_audio_installer_setup,
):
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {"enabled": True, "setup_completed": False, "needs_setup": True},
    )
    monkeypatch.setattr(
        setup_endpoint.audio_readiness_store,
        "get_audio_readiness_store",
        lambda: SimpleNamespace(load=lambda: {"installed_asset_manifests": []}),
    )
    monkeypatch.setattr(
        setup_endpoint.audio_pack_service,
        "build_audio_pack_manifest",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("Unknown curated TTS choice 'bogus_choice'")),
    )

    response = _request(
        "post",
        "/api/v1/setup/audio/packs/export",
        json={
            "bundle_id": "cpu_local",
            "resource_profile": "balanced",
            "tts_choice": "bogus_choice",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == setup_endpoint.INVALID_AUDIO_PACK_EXPORT_REQUEST_DETAIL


def test_audio_pack_import_rejects_invalid_tts_choice_with_400(
    monkeypatch,
    _admin_audio_installer_setup,
):
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {"enabled": True, "setup_completed": False, "needs_setup": True},
    )
    monkeypatch.setattr(
        setup_endpoint.audio_pack_service,
        "register_imported_audio_pack",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Unknown curated TTS choice 'bogus_choice'")),
    )

    response = _request(
        "post",
        "/api/v1/setup/audio/packs/import",
        json={"pack_path": "/tmp/invalid-audio-pack.json"},
    )

    assert response.status_code == 400
    assert "Unknown curated TTS choice" in response.json()["detail"]
