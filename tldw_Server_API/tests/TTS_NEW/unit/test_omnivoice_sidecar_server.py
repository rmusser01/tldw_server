from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter import OmniVoiceAdapter
from tldw_Server_API.app.core.TTS.adapters.omnivoice_runtime import (
    OmniVoiceRuntimeError,
    OmniVoiceSynthesizeResult,
)
from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import (
    OmniVoiceRuntimeStatus,
    OmniVoiceSynthesizeRequest,
)


def _build_test_wav(*, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x01" * sample_width * channels * 64)
    return buffer.getvalue()


class FakeOmniVoiceRuntime:
    def __init__(self, *, scratch_dir: Path, synthesize_error: OmniVoiceRuntimeError | None = None) -> None:
        self.status_calls = 0
        self.load_calls = 0
        self.synthesize_calls: list[OmniVoiceSynthesizeRequest] = []
        self.synthesize_error = synthesize_error
        self.status = self.get_status
        self.last_error_code = None
        self.model = "loaded-model"
        self._model_id = "local-model"
        self._model_path = scratch_dir / "local-model"

    async def get_status(self) -> OmniVoiceRuntimeStatus:
        self.status_calls += 1
        return OmniVoiceRuntimeStatus(
            status="ready",
            ready=True,
            model="local-model",
            model_path=str(self._model_path),
        )

    async def load(self) -> object:
        self.load_calls += 1
        return self.model

    async def synthesize(self, request: OmniVoiceSynthesizeRequest) -> OmniVoiceSynthesizeResult:
        self.synthesize_calls.append(request)
        if request.reference_audio_path:
            self._validate_reference_audio_path(request.reference_audio_path)
        if self.synthesize_error is not None:
            raise self.synthesize_error
        return OmniVoiceSynthesizeResult(
            audio_bytes=_build_test_wav(),
            audio_format="wav",
            sample_rate=24000,
            channels=1,
            cold_start=False,
            model="local-model",
        )

    def _validate_reference_audio_path(self, reference_audio_path: str) -> None:
        reference_path = Path(reference_audio_path).expanduser().resolve(strict=False)
        scratch_path = self._model_path.parent.resolve(strict=False)
        try:
            reference_path.relative_to(scratch_path)
        except ValueError as exc:
            raise OmniVoiceRuntimeError(
                "REFERENCE_PATH_NOT_ALLOWED",
                "OmniVoice clone reference audio path is outside managed directories",
                retryable=False,
            ) from exc


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
    assert payload["status"] == "idle_stopped"  # nosec B101
    assert payload["ready"] is False  # nosec B101


@pytest.mark.unit
def test_sidecar_status_returns_runtime_status(fake_runtime_client, auth_headers: dict[str, str]):
    client, runtime = fake_runtime_client

    response = client.get("/status", headers=auth_headers)

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert payload["status"] == "ready"  # nosec B101
    assert payload["ready"] is True  # nosec B101
    assert payload["model"] == "local-model"  # nosec B101
    assert runtime.status_calls == 1  # nosec B101


@pytest.mark.unit
def test_sidecar_health_is_authorized_and_status_backed_without_loading(
    fake_runtime_client,
    auth_headers: dict[str, str],
):
    client, runtime = fake_runtime_client

    response = client.get("/health", headers=auth_headers)

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert payload["status"] == "ready"  # nosec B101
    assert payload["ready"] is True  # nosec B101
    assert runtime.status_calls == 1  # nosec B101
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
@pytest.mark.parametrize("field_name", ["instruct", "reference_audio_path", "reference_text"])
def test_synthesize_request_rejects_auto_mode_empty_forbidden_fields(field_name: str):
    with pytest.raises(ValidationError, match=field_name):
        OmniVoiceSynthesizeRequest(text="hi", mode="auto", generation={}, **{field_name: ""})


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
@pytest.mark.parametrize("field_name", ["reference_audio_path", "reference_text"])
def test_synthesize_request_rejects_design_mode_empty_forbidden_fields(field_name: str):
    with pytest.raises(ValidationError, match=field_name):
        OmniVoiceSynthesizeRequest(
            text="hi",
            mode="design",
            instruct="warm narrator",
            generation={},
            **{field_name: ""},
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
def test_synthesize_request_rejects_clone_mode_empty_instruct():
    with pytest.raises(ValidationError, match="instruct"):
        OmniVoiceSynthesizeRequest(
            text="hi",
            mode="clone",
            instruct="",
            reference_audio_path="/managed/ref.wav",
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
@pytest.mark.parametrize("field_name", ["reference_audio_path", "reference_text"])
def test_synthesize_request_clone_requires_non_empty_reference_fields(field_name: str):
    kwargs = {
        "reference_audio_path": "/managed/ref.wav",
        "reference_text": "reference transcript",
        field_name: "",
    }
    with pytest.raises(ValidationError, match=field_name):
        OmniVoiceSynthesizeRequest(text="hi", mode="clone", generation={}, **kwargs)


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
