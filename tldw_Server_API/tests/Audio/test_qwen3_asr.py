"""Unit tests for Audio_Transcription_Qwen3ASR module."""

import sys
import types

import pytest


def _import_module():
    """Import the module lazily to avoid heavy dependencies in test collection."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Qwen3ASR as qwen3

    return qwen3


@pytest.mark.unit
def test_planned_local_model_log_does_not_expose_absolute_path(
    tmp_path,
    monkeypatch,
):
    qwen3 = _import_module()
    model_path = tmp_path / "private-user" / "model"
    model_path.mkdir(parents=True)
    captured = []

    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return None

    fake_loader = types.SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: FakeModel(),
    )
    fake_torch = types.SimpleNamespace(
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForCausalLM=fake_loader,
            AutoProcessor=fake_loader,
        ),
    )
    monkeypatch.setattr(
        qwen3.logger,
        "info",
        lambda message, *args: captured.append(message.format(*args)),
    )
    qwen3._MODEL_CACHE.clear()

    qwen3._load_qwen3_asr_model(
        {
            "model_path": str(model_path),
            "device": "cpu",
            "dtype": "float32",
            "allow_download": False,
            "model_revision": None,
            "planned_device": "cpu",
        }
    )

    assert captured
    assert str(model_path) not in "\n".join(captured)


@pytest.mark.unit
def test_resolve_settings_defaults(monkeypatch):
    """Test that settings resolve with sensible defaults."""
    qwen3 = _import_module()

    def fake_get_stt_config():
        return {}

    monkeypatch.setattr(qwen3, "get_stt_config", fake_get_stt_config)

    settings = qwen3._resolve_settings()

    assert settings["enabled"] is False
    assert settings["model_path"] == "./models/qwen3_asr/1.7B"
    assert settings["device"] == "cuda"
    assert settings["dtype"] == "bfloat16"
    assert settings["max_batch_size"] == 32
    assert settings["max_new_tokens"] == 4096
    assert settings["allow_download"] is False
    assert settings["sample_rate"] == 16000
    assert settings["aligner_enabled"] is False
    assert settings["aligner_path"] == "./models/qwen3_asr/aligner"
    assert settings["backend"] == "transformers"
    assert settings["vllm_base_url"] == ""


@pytest.mark.unit
def test_resolve_settings_from_config(monkeypatch):
    """Test that settings are properly read from config."""
    qwen3 = _import_module()

    def fake_get_stt_config():
        return {
            "qwen3_asr_enabled": True,
            "qwen3_asr_model_path": "/custom/path/0.6B",
            "qwen3_asr_device": "cpu",
            "qwen3_asr_dtype": "float16",
            "qwen3_asr_max_batch_size": 16,
            "qwen3_asr_max_new_tokens": 2048,
            "qwen3_asr_allow_download": True,
            "qwen3_asr_sample_rate": 22050,
            "qwen3_asr_aligner_enabled": True,
            "qwen3_asr_aligner_path": "/custom/aligner",
            "qwen3_asr_backend": "vllm",
            "qwen3_asr_vllm_gpu_memory_utilization": 0.8,
            "qwen3_asr_vllm_base_url": "http://localhost:8000",
        }

    monkeypatch.setattr(qwen3, "get_stt_config", fake_get_stt_config)

    settings = qwen3._resolve_settings()

    assert settings["enabled"] is True
    assert settings["model_path"] == "/custom/path/0.6B"
    assert settings["device"] == "cpu"
    assert settings["dtype"] == "float16"
    assert settings["max_batch_size"] == 16
    assert settings["max_new_tokens"] == 2048
    assert settings["allow_download"] is True
    assert settings["sample_rate"] == 22050
    assert settings["aligner_enabled"] is True
    assert settings["aligner_path"] == "/custom/aligner"
    assert settings["backend"] == "vllm"
    assert settings["vllm_gpu_memory_utilization"] == 0.8
    assert settings["vllm_base_url"] == "http://localhost:8000"


@pytest.mark.unit
def test_as_bool_helper():
    """Test boolean conversion helper."""
    qwen3 = _import_module()

    # True values
    assert qwen3._as_bool("true", False) is True
    assert qwen3._as_bool("True", False) is True
    assert qwen3._as_bool("1", False) is True
    assert qwen3._as_bool("yes", False) is True
    assert qwen3._as_bool("on", False) is True
    assert qwen3._as_bool(True, False) is True

    # False values
    assert qwen3._as_bool("false", True) is False
    assert qwen3._as_bool("False", True) is False
    assert qwen3._as_bool("0", True) is False
    assert qwen3._as_bool("no", True) is False
    assert qwen3._as_bool("off", True) is False
    assert qwen3._as_bool(False, True) is False

    # Default values
    assert qwen3._as_bool(None, True) is True
    assert qwen3._as_bool(None, False) is False
    assert qwen3._as_bool("invalid", True) is True


@pytest.mark.unit
def test_as_int_helper():
    """Test integer conversion helper."""
    qwen3 = _import_module()

    assert qwen3._as_int("42", 0) == 42
    assert qwen3._as_int(42, 0) == 42
    assert qwen3._as_int("  100  ", 0) == 100
    assert qwen3._as_int(None, 99) == 99
    assert qwen3._as_int("invalid", 50) == 50


@pytest.mark.unit
def test_as_str_helper():
    """Test string conversion helper."""
    qwen3 = _import_module()

    assert qwen3._as_str("hello", "default") == "hello"
    assert qwen3._as_str("  trimmed  ", "default") == "trimmed"
    assert qwen3._as_str(None, "default") == "default"
    assert qwen3._as_str("", "default") == "default"
    assert qwen3._as_str(123, "default") == "123"


@pytest.mark.unit
def test_as_float_helper():
    """Test float conversion helper."""
    qwen3 = _import_module()

    assert qwen3._as_float("3.14", 0.0) == 3.14
    assert qwen3._as_float(2.5, 0.0) == 2.5
    assert qwen3._as_float("  0.7  ", 0.0) == 0.7
    assert qwen3._as_float(None, 1.0) == 1.0
    assert qwen3._as_float("invalid", 0.5) == 0.5


@pytest.mark.unit
def test_is_qwen3_asr_available_disabled(monkeypatch):
    """Test availability check when disabled."""
    qwen3 = _import_module()

    def fake_get_stt_config():
        return {"qwen3_asr_enabled": False}

    monkeypatch.setattr(qwen3, "get_stt_config", fake_get_stt_config)

    assert qwen3.is_qwen3_asr_available() is False


@pytest.mark.unit
def test_is_qwen3_asr_available_missing_model(monkeypatch, tmp_path):
    """Test availability check when model path doesn't exist."""
    qwen3 = _import_module()

    def fake_get_stt_config():
        return {
            "qwen3_asr_enabled": True,
            "qwen3_asr_model_path": str(tmp_path / "nonexistent"),
        }

    monkeypatch.setattr(qwen3, "get_stt_config", fake_get_stt_config)

    assert qwen3.is_qwen3_asr_available() is False


@pytest.mark.unit
def test_is_qwen3_asr_aligner_available_disabled(monkeypatch):
    """Test aligner availability when disabled."""
    qwen3 = _import_module()

    def fake_get_stt_config():
        return {"qwen3_asr_aligner_enabled": False}

    monkeypatch.setattr(qwen3, "get_stt_config", fake_get_stt_config)

    assert qwen3.is_qwen3_asr_aligner_available() is False


@pytest.mark.unit
def test_is_qwen3_asr_aligner_available_missing_path(monkeypatch, tmp_path):
    """Test aligner availability when path doesn't exist."""
    qwen3 = _import_module()

    def fake_get_stt_config():
        return {
            "qwen3_asr_aligner_enabled": True,
            "qwen3_asr_aligner_path": str(tmp_path / "nonexistent"),
        }

    monkeypatch.setattr(qwen3, "get_stt_config", fake_get_stt_config)

    assert qwen3.is_qwen3_asr_aligner_available() is False


@pytest.mark.unit
def test_get_qwen3_asr_capabilities(monkeypatch, tmp_path):
    """Test capabilities reporting."""
    qwen3 = _import_module()

    model_path = tmp_path / "model"
    model_path.mkdir()

    def fake_get_stt_config():
        return {
            "qwen3_asr_enabled": True,
            "qwen3_asr_model_path": str(model_path),
            "qwen3_asr_device": "cuda",
            "qwen3_asr_aligner_enabled": False,
            "qwen3_asr_backend": "transformers",
        }

    monkeypatch.setattr(qwen3, "get_stt_config", fake_get_stt_config)

    caps = qwen3.get_qwen3_asr_capabilities()

    assert caps["enabled"] is True
    assert caps["model_path"] == str(model_path)
    assert caps["device"] == "cuda"
    assert caps["word_timestamps"] is False
    assert caps["streaming"] is False
    assert caps["backend"] == "transformers"


@pytest.mark.unit
def test_normalize_artifact():
    """Test artifact normalization."""
    qwen3 = _import_module()

    artifact = qwen3._normalize_artifact(
        text="Hello world",
        duration_seconds=2.5,
        language_hint="en",
        detected_language="en",
        model_path="./models/qwen3_asr/1.7B",
        words=[
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
        ],
    )

    assert artifact["text"] == "Hello world"
    assert artifact["language"] == "en"
    assert len(artifact["segments"]) == 1
    assert artifact["segments"][0]["Text"] == "Hello world"
    assert artifact["segments"][0]["start_seconds"] == 0.0
    assert artifact["segments"][0]["end_seconds"] == 2.5
    assert artifact["diarization"]["enabled"] is False
    assert artifact["usage"]["duration_ms"] == 2500
    assert artifact["metadata"]["provider"] == "qwen3-asr"
    assert artifact["metadata"]["model"] == "./models/qwen3_asr/1.7B"
    assert artifact["metadata"]["source"] == "local"
    assert "words" in artifact
    assert len(artifact["words"]) == 2


@pytest.mark.unit
def test_normalize_artifact_without_words():
    """Test artifact normalization without word timestamps."""
    qwen3 = _import_module()

    artifact = qwen3._normalize_artifact(
        text="Test transcription",
        duration_seconds=1.0,
        language_hint=None,
        detected_language="zh",
        model_path="./models/qwen3_asr/0.6B",
        words=None,
    )

    assert artifact["text"] == "Test transcription"
    assert artifact["language"] == "zh"
    assert "words" not in artifact


@pytest.mark.unit
def test_transcribe_disabled_raises_error(monkeypatch):
    """Test that transcription raises error when disabled."""
    qwen3 = _import_module()

    def fake_get_stt_config():
        return {"qwen3_asr_enabled": False}

    monkeypatch.setattr(qwen3, "get_stt_config", fake_get_stt_config)

    from tldw_Server_API.app.core.exceptions import BadRequestError

    with pytest.raises(BadRequestError) as exc_info:
        qwen3.transcribe_with_qwen3_asr("/path/to/audio.wav")

    assert "disabled" in str(exc_info.value).lower()


@pytest.mark.unit
def test_check_cancel_raises_on_cancellation(monkeypatch):
    """Test that cancellation check raises TranscriptionCancelled."""
    qwen3 = _import_module()

    from tldw_Server_API.app.core.exceptions import TranscriptionCancelled

    def cancel_check():
        return True

    with pytest.raises(TranscriptionCancelled):
        qwen3._check_cancel(cancel_check, label="test")


@pytest.mark.unit
def test_check_cancel_passes_when_not_cancelled():
    """Test that cancellation check passes when not cancelled."""
    qwen3 = _import_module()

    def cancel_check():
        return False

    # Should not raise
    qwen3._check_cancel(cancel_check, label="test")


@pytest.mark.unit
def test_check_cancel_passes_when_none():
    """Test that cancellation check passes when callback is None."""
    qwen3 = _import_module()

    # Should not raise
    qwen3._check_cancel(None, label="test")


@pytest.mark.unit
def test_validate_model_path_exists(tmp_path):
    """Test model path validation when path exists."""
    qwen3 = _import_module()

    model_path = tmp_path / "model"
    model_path.mkdir()

    result = qwen3._validate_model_path(str(model_path), allow_download=False)
    assert result == model_path


@pytest.mark.unit
def test_validate_model_path_missing_no_download(tmp_path):
    """Test model path validation when path missing and download disabled."""
    qwen3 = _import_module()

    from tldw_Server_API.app.core.exceptions import BadRequestError

    missing_path = str(tmp_path / "nonexistent")

    with pytest.raises(BadRequestError) as exc_info:
        qwen3._validate_model_path(missing_path, allow_download=False)

    assert "does not exist" in str(exc_info.value)
    assert "hf download" in str(exc_info.value)


@pytest.mark.unit
def test_validate_model_path_missing_with_download(tmp_path):
    """Test model path validation when path missing but download allowed."""
    qwen3 = _import_module()

    from pathlib import Path

    missing_path = str(tmp_path / "nonexistent")

    result = qwen3._validate_model_path(missing_path, allow_download=True)
    assert result == Path(missing_path)


@pytest.mark.unit
def test_get_torch_dtype(monkeypatch):
    """Test torch dtype mapping."""
    qwen3 = _import_module()

    torch = types.ModuleType("torch")
    torch.float32 = object()
    torch.float16 = object()
    torch.bfloat16 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert qwen3._get_torch_dtype("float32") == torch.float32
    assert qwen3._get_torch_dtype("fp32") == torch.float32
    assert qwen3._get_torch_dtype("float16") == torch.float16
    assert qwen3._get_torch_dtype("fp16") == torch.float16
    assert qwen3._get_torch_dtype("bfloat16") == torch.bfloat16
    assert qwen3._get_torch_dtype("bf16") == torch.bfloat16
    # Default to bfloat16 for unknown
    assert qwen3._get_torch_dtype("unknown") == torch.bfloat16


@pytest.mark.unit
def test_resolve_device_cpu():
    """Test device resolution for CPU."""
    qwen3 = _import_module()

    assert qwen3._resolve_device("cpu") == "cpu"
    assert qwen3._resolve_device("CPU") == "cpu"
    assert qwen3._resolve_device("  cpu  ") == "cpu"


@pytest.mark.unit
def test_resolve_audio_path_valid(tmp_path):
    """Test audio path resolution for valid path."""
    qwen3 = _import_module()

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00" * 100)

    result = qwen3._resolve_audio_path(str(audio_file), tmp_path)
    assert result == audio_file


@pytest.mark.unit
def test_resolve_audio_path_outside_base_dir(tmp_path):
    """Test audio path resolution rejects paths outside base_dir."""
    qwen3 = _import_module()

    from tldw_Server_API.app.core.exceptions import BadRequestError

    base_dir = tmp_path / "allowed"
    base_dir.mkdir()

    outside_file = tmp_path / "outside" / "test.wav"

    with pytest.raises(BadRequestError) as exc_info:
        qwen3._resolve_audio_path(str(outside_file), base_dir)

    assert "rejected" in str(exc_info.value).lower()


@pytest.mark.unit
def test_transcribe_vllm_http_missing_url(monkeypatch, tmp_path):
    """Test vLLM HTTP transcription fails when base_url is empty."""
    qwen3 = _import_module()

    from tldw_Server_API.app.core.exceptions import BadRequestError

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00" * 100)

    settings = {
        "vllm_base_url": "",
        "sample_rate": 16000,
    }

    with pytest.raises(BadRequestError) as exc_info:
        qwen3._transcribe_vllm_http(audio_file, settings, None, None)

    assert "vllm base url not configured" in str(exc_info.value).lower()


@pytest.mark.unit
def test_transcribe_vllm_http_success(monkeypatch, tmp_path):
    """Test vLLM HTTP transcription with mocked httpx response."""
    qwen3 = _import_module()

    # Create a mock audio file
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00" * 100)

    settings = {
        "vllm_base_url": ("http://localhost:8000/proxy?legacy=value"),
        "request_model": "Qwen/Qwen3-ASR-1.7B",
        "sample_rate": 16000,
    }

    # Mock httpx.Client
    class MockResponse:
        def __init__(self):
            self.status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "text": "Hello world",
                "language": "en",
                "duration": 2.5,
            }

    client_kwargs = {}
    request_data = {}
    request_urls = []

    class MockClient:
        def __init__(self, **kwargs):
            client_kwargs.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, files=None, data=None):
            request_urls.append(url)
            request_data.update(data or {})
            return MockResponse()

    # Mock httpx
    mock_httpx = types.ModuleType("httpx")
    mock_httpx.Client = MockClient
    mock_httpx.HTTPStatusError = Exception
    mock_httpx.RequestError = Exception

    http_module_attr = "http" + "x"
    monkeypatch.setattr(qwen3, http_module_attr, mock_httpx, raising=False)

    # Mock _load_audio to avoid actual file reading
    def fake_load_audio(path, *, target_sample_rate):
        import numpy as np

        return np.zeros(16000, dtype="float32"), 16000, 1.0

    monkeypatch.setattr(qwen3, "_load_audio", fake_load_audio)

    # Import httpx into the module namespace for the test
    import sys

    sys.modules["httpx"] = mock_httpx

    try:
        result = qwen3._transcribe_vllm_http(audio_file, settings, "en", None)

        assert result["text"] == "Hello world"
        assert result["language"] == "en"
        assert result["metadata"]["model"] == ("vllm:http://localhost:8000/proxy?legacy=value")
        assert result["usage"]["duration_ms"] == 2500
        assert client_kwargs["follow_redirects"] is False
        assert request_data["model"] == "Qwen/Qwen3-ASR-1.7B"
        assert request_urls == ["http://localhost:8000/proxy?legacy=value/v1/audio/transcriptions"]
    finally:
        if "httpx" in sys.modules and sys.modules["httpx"] is mock_httpx:
            del sys.modules["httpx"]


@pytest.mark.unit
def test_transcribe_routes_to_vllm(monkeypatch, tmp_path):
    """Test that transcribe_with_qwen3_asr routes to vLLM when configured."""
    qwen3 = _import_module()

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00" * 100)

    # Track if _transcribe_vllm_http was called
    vllm_called = {"value": False}

    def fake_transcribe_vllm_http(path, settings, language, cancel_check):
        vllm_called["value"] = True
        return {
            "text": "vllm result",
            "language": "en",
            "segments": [],
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": 1000, "tokens": None},
            "metadata": {"provider": "qwen3-asr", "model": "vllm:http://localhost:8000", "source": "vllm"},
        }

    monkeypatch.setattr(qwen3, "_transcribe_vllm_http", fake_transcribe_vllm_http)

    def fake_get_stt_config():
        return {
            "qwen3_asr_enabled": True,
            "qwen3_asr_backend": "vllm",
            "qwen3_asr_vllm_base_url": "http://localhost:8000",
        }

    monkeypatch.setattr(qwen3, "get_stt_config", fake_get_stt_config)

    result = qwen3.transcribe_with_qwen3_asr(str(audio_file), base_dir=tmp_path)

    assert vllm_called["value"] is True
    assert result["text"] == "vllm result"


@pytest.mark.unit
def test_get_capabilities_streaming_disabled_by_default(monkeypatch, tmp_path):
    """Test capabilities show streaming=False when vLLM not configured."""
    qwen3 = _import_module()

    model_path = tmp_path / "model"
    model_path.mkdir()

    def fake_get_stt_config():
        return {
            "qwen3_asr_enabled": True,
            "qwen3_asr_model_path": str(model_path),
            "qwen3_asr_backend": "transformers",
        }

    monkeypatch.setattr(qwen3, "get_stt_config", fake_get_stt_config)

    caps = qwen3.get_qwen3_asr_capabilities()

    assert caps["streaming"] is False
    assert caps["vllm_base_url"] is None


@pytest.mark.unit
def test_get_capabilities_streaming_enabled_with_vllm(monkeypatch, tmp_path):
    """Test capabilities show streaming=True when vLLM is configured."""
    qwen3 = _import_module()

    model_path = tmp_path / "model"
    model_path.mkdir()

    def fake_get_stt_config():
        return {
            "qwen3_asr_enabled": True,
            "qwen3_asr_model_path": str(model_path),
            "qwen3_asr_backend": "vllm",
            "qwen3_asr_vllm_base_url": "http://localhost:8000",
        }

    monkeypatch.setattr(qwen3, "get_stt_config", fake_get_stt_config)

    caps = qwen3.get_qwen3_asr_capabilities()

    assert caps["streaming"] is True
    assert caps["vllm_base_url"] == "http://localhost:8000"
    assert caps["backend"] == "vllm"
