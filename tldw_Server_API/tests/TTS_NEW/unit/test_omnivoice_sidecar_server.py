from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import OmniVoiceSynthesizeRequest


@pytest.fixture
def test_client():
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import create_app

    runtime = FakeOmniVoiceRuntime(scratch_dir=Path("/tmp/omnivoice-test-scratch"))  # nosec B108
    app = create_app(sidecar_token="test-sidecar-token", runtime=runtime)  # nosec B106
    with TestClient(app) as client:
        yield client


@pytest.fixture
def fake_runtime_client(tmp_path: Path):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import create_app

    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    runtime = FakeOmniVoiceRuntime(scratch_dir=scratch_dir)
    app = create_app(sidecar_token="test-sidecar-token", runtime=runtime)  # nosec B106
    with TestClient(app) as client:
        yield client, runtime


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"X-TLDW-Sidecar-Token": "test-sidecar-token"}


@pytest.mark.unit
def test_sidecar_requires_auth_header_for_health(test_client: TestClient):
    response = test_client.get("/health")

    assert response.status_code == 401  # nosec B101


@pytest.mark.unit
def test_sidecar_requires_auth_header_for_synthesize(test_client: TestClient):
    response = test_client.post("/v1/synthesize", json={"text": "hi", "mode": "auto"})

    assert response.status_code == 401  # nosec B101


@pytest.mark.unit
def test_sidecar_accepts_authorized_health_probe(test_client: TestClient):
    response = test_client.get("/health", headers={"X-TLDW-Sidecar-Token": "test-sidecar-token"})

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert payload["status"] == "ok"  # nosec B101
    assert payload["ready"] is True  # nosec B101


@pytest.mark.unit
def test_sidecar_status_returns_runtime_status(fake_runtime_client, auth_headers: dict[str, str]):
    client, runtime = fake_runtime_client

    response = client.get("/status", headers=auth_headers)

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert payload["status"] == "ready"  # nosec B101
    assert payload["ready"] is True  # nosec B101
    assert payload["model"] == "local-model"  # nosec B101
    assert payload["model_path"] is None  # nosec B101
    assert runtime.status_calls == 1  # nosec B101


@pytest.mark.unit
def test_sidecar_status_redacts_status_hook_model_path(tmp_path: Path, auth_headers: dict[str, str]):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import create_app

    runtime = FakeOmniVoiceRuntime(scratch_dir=tmp_path / "secret-model-parent")
    app = create_app(sidecar_token="test-sidecar-token", runtime=runtime)  # nosec B106
    with TestClient(app) as client:
        response = client.get("/status", headers=auth_headers)

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert payload["model"] == "local-model"  # nosec B101
    assert payload["model_path"] is None  # nosec B101


@pytest.mark.unit
def test_sidecar_status_sanitizes_status_hook_path_like_model(tmp_path: Path, auth_headers: dict[str, str]):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import create_app

    runtime = FakeOmniVoiceRuntime(
        scratch_dir=tmp_path / "secret-model-parent",
        status_model=str(tmp_path / "models" / "local-model"),
    )
    app = create_app(sidecar_token="test-sidecar-token", runtime=runtime)  # nosec B106
    with TestClient(app) as client:
        response = client.get("/status", headers=auth_headers)

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert payload["model"] == "local-model"  # nosec B101
    assert payload["model_path"] is None  # nosec B101


@pytest.mark.unit
def test_sidecar_status_redacts_attribute_runtime_model_path(tmp_path: Path, auth_headers: dict[str, str]):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import create_app

    runtime = FakeAttributeStatusRuntime(model_path=tmp_path / "models" / "local-model")
    app = create_app(sidecar_token="test-sidecar-token", runtime=runtime)  # nosec B106
    with TestClient(app) as client:
        response = client.get("/status", headers=auth_headers)

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert payload["status"] == "ready"  # nosec B101
    assert payload["ready"] is True  # nosec B101
    assert payload["model"] == "local-model"  # nosec B101
    assert payload["model_path"] is None  # nosec B101


@pytest.mark.unit
def test_sidecar_synthesize_sanitizes_path_like_model_header(tmp_path: Path, auth_headers: dict[str, str]):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import create_app

    runtime = FakeOmniVoiceRuntime(
        scratch_dir=tmp_path / "scratch",
        synthesize_model=str(tmp_path / "models" / "local-model"),
    )
    app = create_app(sidecar_token="test-sidecar-token", runtime=runtime)  # nosec B106
    with TestClient(app) as client:
        response = client.post(
            "/v1/synthesize",
            headers=auth_headers,
            json={"text": "hi", "mode": "auto"},
        )

    assert response.status_code == 200  # nosec B101
    assert response.headers["X-OmniVoice-Model"] == "local-model"  # nosec B101


@pytest.mark.unit
def test_sidecar_lifecycle_success_routes_declare_runtime_status_response_model(fake_runtime_client):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import OmniVoiceRuntimeStatus

    client, _runtime = fake_runtime_client
    response_models = {
        route.path: route.response_model
        for route in client.app.routes
        if getattr(route, "path", None) in {"/control/warmup", "/control/reload", "/control/shutdown"}
    }

    assert response_models == {  # nosec B101
        "/control/warmup": OmniVoiceRuntimeStatus,
        "/control/reload": OmniVoiceRuntimeStatus,
        "/control/shutdown": OmniVoiceRuntimeStatus,
    }


@pytest.mark.unit
def test_sidecar_health_is_authorized_and_does_not_touch_runtime(
    fake_runtime_client,
    auth_headers: dict[str, str],
):
    client, runtime = fake_runtime_client

    response = client.get("/health", headers=auth_headers)

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert payload["status"] == "ok"  # nosec B101
    assert payload["ready"] is True  # nosec B101
    assert runtime.status_calls == 0  # nosec B101
    assert runtime.load_calls == 0  # nosec B101
    assert runtime.synthesize_calls == []  # nosec B101


@pytest.mark.unit
def test_sidecar_warmup_loads_runtime_and_reports_status(fake_runtime_client, auth_headers: dict[str, str]):
    client, runtime = fake_runtime_client

    response = client.post("/control/warmup", headers=auth_headers)

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert payload["status"] == "ready"  # nosec B101
    assert payload["ready"] is True  # nosec B101
    assert runtime.load_calls == 1  # nosec B101
    assert runtime.status_calls == 1  # nosec B101


@pytest.mark.unit
def test_sidecar_reload_calls_runtime_hook(tmp_path: Path, auth_headers: dict[str, str]):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import create_app

    runtime = FakeReloadableOmniVoiceRuntime(scratch_dir=tmp_path)
    app = create_app(sidecar_token="test-sidecar-token", runtime=runtime)  # nosec B106
    with TestClient(app) as client:
        response = client.post("/control/reload", headers=auth_headers)

    assert response.status_code == 200  # nosec B101
    assert response.json()["status"] == "ready"  # nosec B101
    assert runtime.reload_calls == 1  # nosec B101
    assert runtime.load_calls == 0  # nosec B101


@pytest.mark.unit
def test_sidecar_reload_without_runtime_hook_returns_structured_unsupported(
    fake_runtime_client,
    auth_headers: dict[str, str],
):
    client, runtime = fake_runtime_client

    response = client.post("/control/reload", headers=auth_headers)

    assert response.status_code == 501  # nosec B101
    assert response.json() == {  # nosec B101
        "error": {
            "code": "RUNTIME_RELOAD_UNSUPPORTED",
            "message": "OmniVoice runtime reload is not supported",
            "retryable": False,
        }
    }
    assert runtime.load_calls == 0  # nosec B101


@pytest.mark.unit
def test_sidecar_shutdown_calls_runtime_hook(tmp_path: Path, auth_headers: dict[str, str]):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import create_app

    runtime = FakeShutdownOmniVoiceRuntime(scratch_dir=tmp_path)
    app = create_app(sidecar_token="test-sidecar-token", runtime=runtime)  # nosec B106
    with TestClient(app) as client:
        response = client.post("/control/shutdown", headers=auth_headers)

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert payload["status"] == "shutting-down"  # nosec B101
    assert payload["ready"] is False  # nosec B101
    assert runtime.shutdown_calls == 1  # nosec B101


@pytest.mark.unit
def test_sidecar_shutdown_without_runtime_hook_returns_structured_unsupported(
    fake_runtime_client,
    auth_headers: dict[str, str],
):
    client, runtime = fake_runtime_client

    response = client.post("/control/shutdown", headers=auth_headers)

    assert response.status_code == 501  # nosec B101
    assert response.json() == {  # nosec B101
        "error": {
            "code": "RUNTIME_SHUTDOWN_UNSUPPORTED",
            "message": "OmniVoice runtime shutdown is not supported",
            "retryable": False,
        }
    }
    assert runtime.load_calls == 0  # nosec B101
    assert runtime.synthesize_calls == []  # nosec B101


@pytest.mark.unit
def test_sidecar_synthesize_returns_runtime_wav_bytes_and_headers(
    fake_runtime_client,
    auth_headers: dict[str, str],
):
    client, runtime = fake_runtime_client
    expected_audio = _build_test_wav()

    response = client.post(
        "/v1/synthesize",
        headers=auth_headers,
        json={"text": "hi", "mode": "auto"},
    )

    assert response.status_code == 200  # nosec B101
    assert response.content == expected_audio  # nosec B101
    assert response.headers["X-OmniVoice-Audio-Format"] == "wav"  # nosec B101
    assert response.headers["X-OmniVoice-Sample-Rate"] == "24000"  # nosec B101
    assert response.headers["X-OmniVoice-Channels"] == "1"  # nosec B101
    assert response.headers["X-OmniVoice-Provider"] == "omnivoice"  # nosec B101
    assert response.headers["X-OmniVoice-Mode"] == "auto"  # nosec B101
    assert response.headers["X-OmniVoice-Model"] == "local-model"  # nosec B101
    assert len(runtime.synthesize_calls) == 1  # nosec B101


@pytest.mark.unit
def test_sidecar_runtime_availability_error_returns_structured_503(
    tmp_path: Path,
    auth_headers: dict[str, str],
):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import create_app

    runtime = FakeOmniVoiceRuntime(
        scratch_dir=tmp_path,
        synthesize_error=OmniVoiceRuntimeError(
            "MODEL_NOT_AVAILABLE",
            "OmniVoice requires a configured local model directory",
            retryable=False,
        ),
    )
    app = create_app(sidecar_token="test-sidecar-token", runtime=runtime)  # nosec B106
    with TestClient(app) as client:
        response = client.post(
            "/v1/synthesize",
            headers=auth_headers,
            json={"text": "hi", "mode": "auto"},
        )

    assert response.status_code == 503  # nosec B101
    assert response.json() == {  # nosec B101
        "error": {
            "code": "MODEL_NOT_AVAILABLE",
            "message": "OmniVoice requires a configured local model directory",
            "retryable": False,
        }
    }


@pytest.mark.unit
def test_sidecar_clone_reference_outside_scratch_returns_structured_error(
    fake_runtime_client,
    tmp_path: Path,
    auth_headers: dict[str, str],
):
    client, _runtime = fake_runtime_client
    outside_reference = tmp_path / "outside.wav"
    outside_reference.write_bytes(_build_test_wav())

    response = client.post(
        "/v1/synthesize",
        headers=auth_headers,
        json={
            "text": "hi",
            "mode": "clone",
            "requested_sample_rate": 24000,
            "reference_audio_path": str(outside_reference),
            "reference_text": "reference transcript",
        },
    )

    assert response.status_code == 422  # nosec B101
    assert response.json() == {  # nosec B101
        "error": {
            "code": "REFERENCE_PATH_NOT_ALLOWED",
            "message": "OmniVoice clone reference audio path is outside managed directories",
            "retryable": False,
        }
    }


@pytest.mark.unit
def test_sidecar_accepts_authorized_auto_synthesize(fake_runtime_client, auth_headers: dict[str, str]):
    client, _runtime = fake_runtime_client

    response = client.post(
        "/v1/synthesize",
        headers=auth_headers,
        json={"text": "hi", "mode": "auto"},
    )

    assert response.status_code == 200  # nosec B101
    assert response.headers["X-OmniVoice-Sample-Rate"] == "24000"  # nosec B101
    assert response.headers["X-OmniVoice-Mode"] == "auto"  # nosec B101
    assert response.content.startswith(b"RIFF")  # nosec B101


@pytest.mark.unit
def test_sidecar_runtime_rejects_non_loopback_bind_host():
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import validate_loopback_host

    with pytest.raises(ValueError, match="loopback"):
        validate_loopback_host("0.0.0.0")  # nosec B104


@pytest.mark.unit
def test_sidecar_runtime_normalizes_localhost_to_loopback():
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import validate_loopback_host

    assert validate_loopback_host("localhost") == "127.0.0.1"  # nosec B101


@pytest.mark.unit
def test_synthesize_request_requires_generation_object_shape():
    req = OmniVoiceSynthesizeRequest(text="hi", mode="auto", generation={"num_step": 8})

    assert req.generation.compact() == {"num_step": 8}  # nosec B101


@pytest.mark.unit
def test_synthesize_request_rejects_unknown_top_level_keys():
    with pytest.raises(ValidationError):
        OmniVoiceSynthesizeRequest(text="hi", mode="auto", temperature=0.7)


@pytest.mark.unit
def test_synthesize_request_rejects_unknown_generation_keys():
    with pytest.raises(ValidationError, match="generation"):
        OmniVoiceSynthesizeRequest(text="hi", mode="auto", generation={"unknown": 1})


@pytest.mark.unit
def test_synthesize_request_rejects_mode_field_conflicts():
    with pytest.raises(ValidationError, match="mode=auto"):
        OmniVoiceSynthesizeRequest(text="hi", mode="auto", instruct="warm", generation={})


@pytest.mark.unit
def test_synthesize_request_rejects_mixed_design_and_clone_inputs():
    with pytest.raises(ValidationError, match="instruct"):
        OmniVoiceSynthesizeRequest(
            text="hi",
            mode="clone",
            instruct="warm",
            reference_audio_path="/managed/ref.wav",
            reference_text="reference transcript",
            generation={},
        )


@pytest.mark.unit
def test_synthesize_request_accepts_design_mode_with_instruct():
    req = OmniVoiceSynthesizeRequest(text="hi", mode="design", instruct="warm narrator", generation={})

    assert req.instruct == "warm narrator"  # nosec B101


@pytest.mark.unit
def test_synthesize_request_clone_requires_reference_text_and_path():
    with pytest.raises(ValidationError, match="reference_text"):
        OmniVoiceSynthesizeRequest(
            text="hi",
            mode="clone",
            reference_audio_path="/managed/ref.wav",
            generation={},
        )


@pytest.mark.unit
def test_sidecar_clone_mode_requires_reference_audio_path(test_client: TestClient):
    response = test_client.post(
        "/v1/synthesize",
        headers={"X-TLDW-Sidecar-Token": "test-sidecar-token"},
        json={"text": "hi", "mode": "clone", "requested_sample_rate": 24000},
    )

    assert response.status_code == 422  # nosec B101


@pytest.mark.unit
def test_sidecar_clone_mode_rejects_directory_reference(test_client: TestClient, tmp_path: Path):
    response = test_client.post(
        "/v1/synthesize",
        headers={"X-TLDW-Sidecar-Token": "test-sidecar-token"},
        json={
            "text": "hi",
            "mode": "clone",
            "requested_sample_rate": 24000,
            "reference_audio_path": str(tmp_path),
            "reference_text": "reference transcript",
        },
    )

    assert response.status_code == 422  # nosec B101
