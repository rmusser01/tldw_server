from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


class _FakeRuntime:
    runtime_mode = "real"

    def __init__(self):
        self.requests = []

    def health(self):
        from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import OmniVoiceHealthResponse

        return OmniVoiceHealthResponse(
            runtime_mode=self.runtime_mode,
            model_loaded=True,
            model_ready=True,
        )

    def warmup(self):
        return self.health()

    def reload(self):
        return self.health()

    def shutdown(self):
        return self.health().model_copy(update={"status": "shutting-down", "ready": False})

    def synthesize(self, request):
        from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import OmniVoiceSynthesizeResponse

        self.requests.append(request)
        return (
            b"fake-wav",
            OmniVoiceSynthesizeResponse(sample_rate=44100, mode=request.mode),
        )


@pytest.fixture
def test_client():
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import create_app

    app = create_app(sidecar_token="test-sidecar-token")  # nosec B106
    with TestClient(app) as client:
        yield client


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
    assert payload["runtime_mode"] == "stub"  # nosec B101
    assert payload["model_loaded"] is False  # nosec B101
    assert payload["model_ready"] is True  # nosec B101
    assert payload["last_error"] is None  # nosec B101


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
def test_sidecar_clone_mode_requires_reference_audio_path(test_client: TestClient):
    response = test_client.post(
        "/v1/synthesize",
        headers={"X-TLDW-Sidecar-Token": "test-sidecar-token"},
        json={"text": "hi", "mode": "clone", "sample_rate": 24000},
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
            "sample_rate": 24000,
            "reference_audio_path": str(tmp_path),
        },
    )

    assert response.status_code == 422  # nosec B101


@pytest.mark.unit
def test_sidecar_synthesize_delegates_to_configured_runtime():
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import create_app

    runtime = _FakeRuntime()
    app = create_app(sidecar_token="test-sidecar-token", runtime=runtime)  # nosec B106

    with TestClient(app) as client:
        response = client.post(
            "/v1/synthesize",
            headers={"X-TLDW-Sidecar-Token": "test-sidecar-token"},
            json={"text": "hello", "mode": "auto"},
        )

    assert response.status_code == 200  # nosec B101
    assert response.content == b"fake-wav"  # nosec B101
    assert response.headers["X-OmniVoice-Audio-Format"] == "wav"  # nosec B101
    assert response.headers["X-OmniVoice-Sample-Rate"] == "44100"  # nosec B101
    assert runtime.requests[0].text == "hello"  # nosec B101


@pytest.mark.unit
def test_sidecar_runtime_failure_returns_503_without_stub_fallback():
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import OmniVoiceRuntimeError, create_app

    class _FailingRuntime(_FakeRuntime):
        runtime_mode = "real"

        def synthesize(self, request):  # noqa: ARG002
            raise OmniVoiceRuntimeError("OmniVoice runtime dependency missing: soundfile")

    app = create_app(sidecar_token="test-sidecar-token", runtime=_FailingRuntime())  # nosec B106

    with TestClient(app) as client:
        response = client.post(
            "/v1/synthesize",
            headers={"X-TLDW-Sidecar-Token": "test-sidecar-token"},
            json={"text": "hello", "mode": "auto"},
        )

    assert response.status_code == 503  # nosec B101
    assert "soundfile" in response.json()["detail"]  # nosec B101
    assert response.content != b"fake-wav"  # nosec B101


@pytest.mark.unit
def test_real_runtime_generates_wav_for_plain_text_with_fake_model():
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import OmniVoiceSynthesizeRequest
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import RealOmniVoiceRuntime

    calls = {"loader": [], "generate": [], "writer": []}

    class _FakeModel:
        sampling_rate = 24000

        def generate(self, **kwargs):
            calls["generate"].append(kwargs)
            return [[0.0, 0.1, -0.1]]

    def _fake_loader(*, model_id, device, dtype):
        calls["loader"].append({"model_id": model_id, "device": device, "dtype": dtype})
        return _FakeModel()

    def _fake_wav_writer(buffer, audio, sample_rate):
        calls["writer"].append({"audio": audio, "sample_rate": sample_rate})
        buffer.write(b"real-wav")

    runtime = RealOmniVoiceRuntime(
        model_id="k2-fsa/OmniVoice",
        device="cpu",
        dtype="float32",
        model_loader=_fake_loader,
        wav_writer=_fake_wav_writer,
    )

    audio_bytes, metadata = runtime.synthesize(OmniVoiceSynthesizeRequest(text="hello", mode="auto"))

    assert audio_bytes == b"real-wav"  # nosec B101
    assert metadata.sample_rate == 24000  # nosec B101
    assert metadata.mode == "auto"  # nosec B101
    assert calls["loader"] == [{"model_id": "k2-fsa/OmniVoice", "device": "cpu", "dtype": "float32"}]  # nosec B101
    assert calls["generate"] == [{"text": "hello"}]  # nosec B101
    assert calls["writer"][0]["sample_rate"] == 24000  # nosec B101
    assert runtime.health().model_loaded is True  # nosec B101
    assert runtime.health().model_ready is True  # nosec B101


@pytest.mark.unit
def test_real_runtime_converts_tensor_audio_to_cpu_numpy_mono_before_writing():
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import OmniVoiceSynthesizeRequest
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import RealOmniVoiceRuntime

    calls = {"writer": []}

    class _FakeArray:
        def __init__(self, tensor_calls):
            self._tensor_calls = tensor_calls

        def squeeze(self):
            self._tensor_calls.append("squeeze")
            return "mono-audio"

    class _FakeTensor:
        def __init__(self):
            self.calls = []

        def detach(self):
            self.calls.append("detach")
            return self

        def cpu(self):
            self.calls.append("cpu")
            return self

        def numpy(self):
            self.calls.append("numpy")
            return _FakeArray(self.calls)

    generated_audio = _FakeTensor()

    class _FakeModel:
        sampling_rate = 24000

        def generate(self, **kwargs):  # noqa: ARG002
            return generated_audio

    def _fake_wav_writer(buffer, audio, sample_rate):
        calls["writer"].append({"audio": audio, "sample_rate": sample_rate})
        buffer.write(b"tensor-wav")

    runtime = RealOmniVoiceRuntime(
        model_id="k2-fsa/OmniVoice",
        device="cpu",
        dtype="float32",
        model_loader=lambda **kwargs: _FakeModel(),  # noqa: ARG005
        wav_writer=_fake_wav_writer,
    )

    audio_bytes, metadata = runtime.synthesize(OmniVoiceSynthesizeRequest(text="hello", mode="auto"))

    assert audio_bytes == b"tensor-wav"  # nosec B101
    assert metadata.sample_rate == 24000  # nosec B101
    assert generated_audio.calls == ["detach", "cpu", "numpy", "squeeze"]  # nosec B101
    assert calls["writer"] == [{"audio": "mono-audio", "sample_rate": 24000}]  # nosec B101


@pytest.mark.unit
def test_real_runtime_passes_clone_reference_inputs_to_fake_model(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import OmniVoiceSynthesizeRequest
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import RealOmniVoiceRuntime

    calls = {"generate": []}
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"wav")

    class _FakeModel:
        sampling_rate = 24000

        def generate(self, **kwargs):
            calls["generate"].append(kwargs)
            return [[0.0]]

    runtime = RealOmniVoiceRuntime(
        model_id="k2-fsa/OmniVoice",
        device="cpu",
        dtype="float32",
        model_loader=lambda **kwargs: _FakeModel(),  # noqa: ARG005
        wav_writer=lambda buffer, audio, sample_rate: buffer.write(b"clone-wav"),  # noqa: ARG005
    )

    request = OmniVoiceSynthesizeRequest(
        text="say this",
        mode="clone",
        reference_audio_path=str(reference_audio),
        reference_text="reference transcript",
    )

    audio_bytes, metadata = runtime.synthesize(request)

    assert audio_bytes == b"clone-wav"  # nosec B101
    assert metadata.mode == "clone"  # nosec B101
    assert calls["generate"] == [  # nosec B101
        {
            "text": "say this",
            "ref_audio": str(reference_audio),
            "ref_text": "reference transcript",
        }
    ]


@pytest.mark.unit
def test_real_runtime_allows_clone_without_reference_text_for_auto_transcription(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import OmniVoiceSynthesizeRequest
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import RealOmniVoiceRuntime

    calls = {"generate": []}
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"wav")

    class _FakeModel:
        sampling_rate = 24000

        def generate(self, **kwargs):
            calls["generate"].append(kwargs)
            return [[0.0]]

    runtime = RealOmniVoiceRuntime(
        model_id="k2-fsa/OmniVoice",
        device="cpu",
        dtype="float32",
        model_loader=lambda **kwargs: _FakeModel(),  # noqa: ARG005
        wav_writer=lambda buffer, audio, sample_rate: buffer.write(b"clone-wav"),  # noqa: ARG005
    )

    request = OmniVoiceSynthesizeRequest(
        text="say this",
        mode="clone",
        reference_audio_path=str(reference_audio),
    )

    audio_bytes, metadata = runtime.synthesize(request)

    assert audio_bytes == b"clone-wav"  # nosec B101
    assert metadata.mode == "clone"  # nosec B101
    assert calls["generate"] == [  # nosec B101
        {
            "text": "say this",
            "ref_audio": str(reference_audio),
            "ref_text": None,
        }
    ]


@pytest.mark.unit
def test_real_runtime_passes_generation_controls_to_fake_model():
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import OmniVoiceSynthesizeRequest
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import RealOmniVoiceRuntime

    calls = {"generate": []}

    class _FakeModel:
        sampling_rate = 24000

        def generate(self, **kwargs):
            calls["generate"].append(kwargs)
            return [[0.0]]

    runtime = RealOmniVoiceRuntime(
        model_id="k2-fsa/OmniVoice",
        device="cpu",
        dtype="float32",
        model_loader=lambda **kwargs: _FakeModel(),  # noqa: ARG005
        wav_writer=lambda buffer, audio, sample_rate: buffer.write(b"controlled-wav"),  # noqa: ARG005
    )

    request = OmniVoiceSynthesizeRequest(
        text="hello",
        mode="auto",
        language="English",
        instruct="calm female narrator",
        duration=2.5,
        speed=1.1,
        generation_params={
            "num_step": 16,
            "guidance_scale": 1.5,
            "denoise": True,
            "postprocess_output": False,
            "preprocess_prompt": False,
        },
    )

    runtime.synthesize(request)

    assert calls["generate"] == [  # nosec B101
        {
            "text": "hello",
            "language": "English",
            "instruct": "calm female narrator",
            "duration": 2.5,
            "speed": 1.1,
            "num_step": 16,
            "guidance_scale": 1.5,
            "denoise": True,
            "postprocess_output": False,
            "preprocess_prompt": False,
        }
    ]


@pytest.mark.unit
def test_real_runtime_health_does_not_wait_for_generation_lock():
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import RealOmniVoiceRuntime

    runtime = RealOmniVoiceRuntime(model_loader=lambda **kwargs: None)  # noqa: ARG005
    runtime._lock.acquire()
    try:
        result = {}

        def _probe_health():
            result["health"] = runtime.health()

        thread = threading.Thread(target=_probe_health)
        thread.start()
        thread.join(timeout=0.1)

        assert thread.is_alive() is False  # nosec B101
        assert result["health"].ready is True  # nosec B101
    finally:
        runtime._lock.release()
        thread.join(timeout=1.0)


@pytest.mark.unit
def test_sidecar_env_loader_selects_real_runtime(monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import RealOmniVoiceRuntime, _load_app_from_env

    monkeypatch.setenv("OMNIVOICE_SIDECAR_TOKEN", "test-sidecar-token")
    monkeypatch.setenv("OMNIVOICE_RUNTIME_MODE", "real")
    monkeypatch.setenv("OMNIVOICE_MODEL", "local-omnivoice")
    monkeypatch.setenv("OMNIVOICE_DEVICE", "cpu")
    monkeypatch.setenv("OMNIVOICE_DTYPE", "float32")

    app = _load_app_from_env()
    runtime = app.state.omnivoice_runtime

    assert isinstance(runtime, RealOmniVoiceRuntime)  # nosec B101
    assert runtime.model_id == "local-omnivoice"  # nosec B101
    assert runtime.device == "cpu"  # nosec B101
    assert runtime.dtype == "float32"  # nosec B101


@pytest.mark.unit
def test_sidecar_shutdown_requests_uvicorn_exit():
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import create_app

    runtime = _FakeRuntime()
    app = create_app(sidecar_token="test-sidecar-token", runtime=runtime)  # nosec B106
    app.state.uvicorn_server = SimpleNamespace(should_exit=False)

    with TestClient(app) as client:
        response = client.post(
            "/control/shutdown",
            headers={"X-TLDW-Sidecar-Token": "test-sidecar-token"},
        )

    assert response.status_code == 200  # nosec B101
    assert app.state.uvicorn_server.should_exit is True  # nosec B101
