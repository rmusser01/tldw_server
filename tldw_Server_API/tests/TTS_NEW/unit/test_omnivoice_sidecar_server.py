from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter import OmniVoiceAdapter
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
def test_sidecar_accepts_authorized_auto_synthesize(test_client: TestClient):
    response = test_client.post(
        "/v1/synthesize",
        headers={"X-TLDW-Sidecar-Token": "test-sidecar-token"},
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
def test_synthesize_request_rejects_auto_mode_reference_audio_path():
    with pytest.raises(ValidationError, match="mode=auto"):
        OmniVoiceSynthesizeRequest(
            text="hi",
            mode="auto",
            reference_audio_path="/managed/ref.wav",
            generation={},
        )


@pytest.mark.unit
def test_synthesize_request_rejects_auto_mode_reference_text():
    with pytest.raises(ValidationError, match="reference_text"):
        OmniVoiceSynthesizeRequest(
            text="hi",
            mode="auto",
            reference_text="reference transcript",
            generation={},
        )


@pytest.mark.unit
def test_synthesize_request_rejects_design_mode_reference_text():
    with pytest.raises(ValidationError, match="reference_text"):
        OmniVoiceSynthesizeRequest(
            text="hi",
            mode="design",
            instruct="warm narrator",
            reference_text="reference transcript",
            generation={},
        )


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
def test_omnivoice_adapter_sidecar_payload_matches_protocol_schema():
    adapter = OmniVoiceAdapter({"sample_rate": 24000})
    request = TTSRequest(
        text="hi",
        voice="narrator",
        format=AudioFormat.WAV,
        stream=False,
        target_sample_rate=22050,
    )

    payload = adapter._build_sidecar_payload(
        request,
        mode="auto",
        sample_rate=22050,
        reference_audio_path=None,
    )
    parsed = OmniVoiceSynthesizeRequest(**payload)

    assert payload["requested_sample_rate"] == 22050  # nosec B101
    assert "sample_rate" not in payload  # nosec B101
    assert parsed.requested_sample_rate == 22050  # nosec B101


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
