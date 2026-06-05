from fastapi.testclient import TestClient
from types import SimpleNamespace

from tldw_Server_API.app.main import app
import tldw_Server_API.app.api.v1.endpoints.setup as setup_endpoint


def _make_client():
    return TestClient(app)


class _CompletionReadyStore:
    def validate_completion_ready(self):
        return None

    def mark_completed_with_legacy_flag(self, mark_legacy_complete):
        mark_legacy_complete()
        return None


def test_setup_complete_does_not_imply_audio_ready(mocker, tmp_path):
    store = setup_endpoint.audio_readiness_store.AudioReadinessStore(
        tmp_path / "audio_readiness.json"
    )

    mocker.patch.object(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        return_value={"enabled": True, "needs_setup": True, "setup_completed": False, "completed": False},
    )
    mocker.patch.object(setup_endpoint.setup_manager, "mark_setup_completed")
    mocker.patch.object(
        setup_endpoint.audio_readiness_store,
        "get_audio_readiness_store",
        return_value=store,
    )
    mocker.patch.object(
        setup_endpoint,
        "_run_first_run_store_call",
        side_effect=lambda callback: callback(_CompletionReadyStore()),
    )

    with _make_client() as client:
        response = client.post("/api/v1/setup/complete", json={"disable_first_time_setup": False})
        readiness = client.get("/api/v1/setup/audio/readiness")

    assert response.status_code == 200
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "not_started"


def test_audio_readiness_reset_endpoint_restores_default_state(mocker, tmp_path):
    store = setup_endpoint.audio_readiness_store.AudioReadinessStore(
        tmp_path / "audio_readiness.json"
    )
    store.update(
        status="failed",
        selected_bundle_id="cpu_local",
        remediation_items=["FFmpeg missing"],
    )

    mocker.patch.object(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        return_value={"enabled": True, "needs_setup": False},
    )
    mocker.patch.object(
        setup_endpoint.audio_readiness_store,
        "get_audio_readiness_store",
        return_value=store,
    )

    with _make_client() as client:
        response = client.post("/api/v1/setup/audio/readiness/reset")
        readiness = client.get("/api/v1/setup/audio/readiness")

    assert response.status_code == 200
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "not_started"
    assert readiness.json()["selected_bundle_id"] is None


def test_audio_provision_endpoint_accepts_resource_profile(mocker):
    mocker.patch.object(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        return_value={"enabled": True, "needs_setup": True},
    )
    execute_mock = mocker.patch.object(
        setup_endpoint.install_manager,
        "execute_audio_bundle",
        return_value={"status": "completed", "bundle_id": "cpu_local", "resource_profile": "balanced"},
    )

    with _make_client() as client:
        response = client.post(
            "/api/v1/setup/audio/provision",
            json={"bundle_id": "cpu_local", "resource_profile": "balanced"},
        )

    assert response.status_code == 200
    execute_mock.assert_called_once_with(
        "cpu_local",
        resource_profile="balanced",
        tts_choice=None,
        safe_rerun=False,
    )


def test_audio_provision_endpoint_offloads_bundle_execution(monkeypatch, mocker):
    to_thread_calls = []

    async def fake_to_thread(func, /, *args, **kwargs):
        to_thread_calls.append((func, args, kwargs))
        return {"status": "completed", "bundle_id": "cpu_local", "resource_profile": "balanced"}

    monkeypatch.setattr(
        setup_endpoint,
        "asyncio",
        SimpleNamespace(to_thread=fake_to_thread),
        raising=False,
    )
    mocker.patch.object(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        return_value={"enabled": True, "needs_setup": True},
    )

    with _make_client() as client:
        response = client.post(
            "/api/v1/setup/audio/provision",
            json={"bundle_id": "cpu_local", "resource_profile": "balanced", "safe_rerun": True},
        )

    assert response.status_code == 200
    assert to_thread_calls == [
        (
            setup_endpoint.install_manager.execute_audio_bundle,
            ("cpu_local",),
            {"resource_profile": "balanced", "tts_choice": None, "safe_rerun": True},
        )
    ]


def test_audio_provision_endpoint_masks_bundle_lookup_details(mocker):
    mocker.patch.object(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        return_value={"enabled": True, "needs_setup": True},
    )
    mocker.patch.object(
        setup_endpoint.install_manager,
        "execute_audio_bundle",
        side_effect=KeyError("catalog internals"),
    )

    with _make_client() as client:
        response = client.post(
            "/api/v1/setup/audio/provision",
            json={"bundle_id": "cpu_local", "resource_profile": "balanced"},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Audio bundle not found"
    assert "catalog internals" not in response.text
