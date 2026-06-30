import types
import importlib
import importlib.machinery
import sys
import builtins

import pytest


# Stub heavyweight audio deps before adapter imports to avoid local
# ctranslate2/torch dynamic-load aborts in constrained test environments.
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    _fake_torch.Tensor = object
    _fake_torch.nn = types.SimpleNamespace(Module=object)
    _fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = _fake_torch

if "faster_whisper" not in sys.modules:
    _fake_fw = types.ModuleType("faster_whisper")
    _fake_fw.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", loader=None)

    class _StubWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    _fake_fw.WhisperModel = _StubWhisperModel
    _fake_fw.BatchedInferencePipeline = _StubWhisperModel
    sys.modules["faster_whisper"] = _fake_fw

if "transformers" not in sys.modules:
    _fake_tf = types.ModuleType("transformers")
    _fake_tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class _StubProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    _fake_tf.AutoProcessor = _StubProcessor
    _fake_tf.Qwen2AudioForConditionalGeneration = _StubModel
    sys.modules["transformers"] = _fake_tf


_EXCEPTIONS_MODULE = "tldw_Server_API.app.core.exceptions"
_AUDIO_LIB_MODULE = "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib"


def _install_py39_compat_stubs() -> None:
    exceptions_stub = types.ModuleType(_EXCEPTIONS_MODULE)
    exceptions_stub.BadRequestError = type("BadRequestError", (Exception,), {})
    exceptions_stub.CancelCheckError = type("CancelCheckError", (Exception,), {})
    exceptions_stub.TranscriptionCancelled = type("TranscriptionCancelled", (Exception,), {})
    exceptions_stub.InvalidStoragePathError = type("InvalidStoragePathError", (Exception,), {})
    exceptions_stub.StorageUnavailableError = type("StorageUnavailableError", (Exception,), {})
    exceptions_stub.NetworkError = type("NetworkError", (Exception,), {})
    exceptions_stub.RetryExhaustedError = type("RetryExhaustedError", (Exception,), {})
    exceptions_stub.__file__ = __file__

    def _exception_getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(str(name), (Exception,), {})

    exceptions_stub.__getattr__ = _exception_getattr  # type: ignore[assignment]
    sys.modules[_EXCEPTIONS_MODULE] = exceptions_stub

    audio_lib_stub = types.ModuleType(_AUDIO_LIB_MODULE)
    audio_lib_stub.__file__ = __file__

    def _parse_transcription_model(model_name: str):
        normalized = (model_name or "").strip()
        lowered = normalized.lower()
        if lowered.startswith("parakeet"):
            return "parakeet", normalized, None
        if lowered.startswith("qwen2audio"):
            return "qwen2audio", normalized, None
        if lowered.startswith("vibevoice"):
            return "vibevoice", normalized, None
        if lowered.startswith("external:"):
            return "external", normalized, None
        return "whisper", normalized, None

    def _speech_to_text(*args, **kwargs):
        return [], kwargs.get("selected_source_lang")

    audio_lib_stub.parse_transcription_model = _parse_transcription_model
    audio_lib_stub.speech_to_text = _speech_to_text
    audio_lib_stub.strip_whisper_metadata_header = lambda segments: segments
    sys.modules[_AUDIO_LIB_MODULE] = audio_lib_stub


def _import_module():
    module_name = "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter"
    try:
        # Local import so tests don't break when heavy STT deps are absent.
        return importlib.import_module(module_name)
    except TypeError as exc:
        # Python 3.9 cannot import some project modules that use PEP-604
        # runtime unions. Inject a minimal exceptions stub for STT tests.
        if "unsupported operand type(s) for |" not in str(exc):
            raise
        _install_py39_compat_stubs()
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)


@pytest.mark.unit
def test_default_provider_name_uses_stt_settings(monkeypatch):
    spa = _import_module()

    # Simulate STT-Settings with both keys present; default_transcriber should win.
    def fake_get_stt_config():
        return {
            "default_stt_provider": "parakeet",
            "default_transcriber": "faster_whisper",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    registry = spa.SttProviderRegistry()
    assert registry.get_default_provider_name() == "faster-whisper"


@pytest.mark.unit
def test_default_provider_name_falls_back_to_stt_provider(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():

        return {
            "default_stt_provider": "parakeet",
            # No default_transcriber key
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    registry = spa.SttProviderRegistry()
    assert registry.get_default_provider_name() == "parakeet"


@pytest.mark.unit
def test_get_adapter_unknown_provider_falls_back_to_faster_whisper(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():

        return {
            "default_stt_provider": "parakeet",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    registry = spa.SttProviderRegistry()
    adapter = registry.get_adapter("unknown-provider")
    assert adapter.name.value == "faster-whisper"


@pytest.mark.unit
def test_resolve_provider_for_model_uses_parser(monkeypatch):
    spa = _import_module()

    # Provide a simple, deterministic parser implementation so we don't depend
    # on the exact behavior of Audio_Transcription_Lib here.
    def fake_parse_transcription_model(model_name: str):
        if model_name.startswith("parakeet"):
            return ("parakeet", "parakeet", "onnx")
        if model_name.startswith("qwen2audio"):
            return ("qwen2audio", model_name, None)
        if model_name.startswith("vibevoice"):
            return ("vibevoice", model_name, None)
        return ("whisper", model_name, None)

    monkeypatch.setattr(spa, "parse_transcription_model", fake_parse_transcription_model)

    registry = spa.SttProviderRegistry()

    provider, model, variant = registry.resolve_provider_for_model("parakeet-onnx")
    assert provider == "parakeet"
    assert model == "parakeet"
    assert variant == "onnx"

    provider, model, variant = registry.resolve_provider_for_model("qwen2audio-test")
    assert provider == "qwen2audio"
    assert model == "qwen2audio-test"
    assert variant is None

    provider, model, variant = registry.resolve_provider_for_model("vibevoice-asr")
    assert provider == "vibevoice"
    assert model == "vibevoice-asr"
    assert variant is None

    # Whisper-family models should normalize to faster-whisper.
    provider, model, variant = registry.resolve_provider_for_model("whisper-1")
    assert provider == "faster-whisper"
    assert model == "whisper-1"
    assert variant is None


@pytest.mark.unit
def test_resolve_provider_for_model_keeps_parakeet_when_parser_import_fails(monkeypatch):
    spa = _import_module()
    registry = spa.SttProviderRegistry()
    real_import = builtins.__import__

    monkeypatch.delitem(sys.modules, _AUDIO_LIB_MODULE, raising=False)

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name == "Audio_Transcription_Lib":
            raise ImportError("simulated parser import failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    provider, model, variant = registry.resolve_provider_for_model("parakeet-onnx")

    assert provider == "parakeet"
    assert model == "parakeet"
    assert variant == "onnx"


@pytest.mark.unit
def test_resolve_provider_for_model_allows_external_prefix():
    spa = _import_module()

    registry = spa.SttProviderRegistry()
    provider, model, variant = registry.resolve_provider_for_model("external:custom")
    assert provider == "external"
    assert model == "external:custom"
    assert variant is None


@pytest.mark.unit
def test_resolve_provider_for_model_uses_config_default(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():
        return {
            "default_transcriber": "parakeet",
            "nemo_model_variant": "mlx",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    registry = spa.SttProviderRegistry()
    provider, model, variant = registry.resolve_provider_for_model(None)
    assert provider == "parakeet"
    assert model == "parakeet-mlx"
    assert variant == "mlx"


@pytest.mark.unit
def test_resolve_provider_for_model_uses_vibevoice_defaults(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():
        return {
            "default_transcriber": "vibevoice-asr",
            "vibevoice_model_id": "microsoft/VibeVoice-ASR",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    registry = spa.SttProviderRegistry()
    provider, model, variant = registry.resolve_provider_for_model(None)
    assert provider == "vibevoice"
    assert model == "microsoft/VibeVoice-ASR"
    assert variant is None


@pytest.mark.unit
def test_resolve_default_transcription_model_uses_whisper_fallback(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():
        return {"default_transcriber": "faster-whisper"}

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    default_model = spa.resolve_default_transcription_model("whisper-1")
    assert default_model == "whisper-1"


@pytest.mark.unit
def test_resolve_default_transcription_model_prefers_batch_default(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():
        return {
            "default_batch_transcription_model": "parakeet-onnx",
            "default_transcriber": "faster-whisper",
            "default_stt_provider": "faster-whisper",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    default_model = spa.resolve_default_transcription_model("whisper-1")
    assert default_model == "parakeet-onnx"


@pytest.mark.unit
def test_resolve_default_transcription_model_uses_canonical_parakeet_onnx_alias(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():
        return {
            "default_transcriber": "parakeet",
            "default_stt_provider": "parakeet",
            "nemo_model_variant": "onnx",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    default_model = spa.resolve_default_transcription_model("whisper-1")
    assert default_model == "parakeet-tdt-0.6b-v3-onnx"


@pytest.mark.unit
def test_capabilities_exposed_for_known_providers():
    spa = _import_module()

    registry = spa.SttProviderRegistry()

    fw_caps = registry.get_capabilities("faster-whisper")
    assert fw_caps.supports_batch is True
    assert fw_caps.supports_streaming is True

    parakeet_caps = registry.get_capabilities("parakeet")
    assert parakeet_caps.supports_batch is True
    assert parakeet_caps.supports_streaming is True

    canary_caps = registry.get_capabilities("canary")
    assert canary_caps.supports_batch is True
    assert canary_caps.supports_streaming is False

    qwen_caps = registry.get_capabilities("qwen2audio")
    assert qwen_caps.supports_batch is True
    assert qwen_caps.supports_streaming is False

    vibe_caps = registry.get_capabilities("vibevoice")
    assert vibe_caps.supports_batch is True
    assert vibe_caps.supports_streaming is False
    assert vibe_caps.supports_diarization is True

    qwen3_caps = registry.get_capabilities("qwen3-asr")
    assert qwen3_caps.supports_batch is True
    assert qwen3_caps.supports_streaming is False
    assert qwen3_caps.supports_diarization is False
    assert "word timestamps" in (qwen3_caps.notes or "").lower()


@pytest.mark.unit
def test_resolve_provider_for_model_qwen3_asr_variants(monkeypatch):
    spa = _import_module()

    # Test qwen3-asr-1.7b
    registry = spa.SttProviderRegistry()
    provider, model, variant = registry.resolve_provider_for_model("qwen3-asr-1.7b")
    assert provider == "qwen3-asr"
    assert model == "Qwen/Qwen3-ASR-1.7B"
    assert variant is None

    # Test qwen3-asr-0.6b
    provider, model, variant = registry.resolve_provider_for_model("qwen3-asr-0.6b")
    assert provider == "qwen3-asr"
    assert model == "Qwen/Qwen3-ASR-0.6B"
    assert variant is None

    # Test bare qwen3-asr defaults to 1.7B
    provider, model, variant = registry.resolve_provider_for_model("qwen3-asr")
    assert provider == "qwen3-asr"
    assert model == "Qwen/Qwen3-ASR-1.7B"
    assert variant is None


@pytest.mark.unit
def test_resolve_provider_for_model_qwen3_asr_aliases(monkeypatch):
    spa = _import_module()

    registry = spa.SttProviderRegistry()

    # Test underscore variant
    provider, model, variant = registry.resolve_provider_for_model("qwen3_asr_1.7b")
    assert provider == "qwen3-asr"
    assert model == "Qwen/Qwen3-ASR-1.7B"

    # Test mixed case
    provider, model, variant = registry.resolve_provider_for_model("Qwen3-ASR-0.6B")
    assert provider == "qwen3-asr"
    assert model == "Qwen/Qwen3-ASR-0.6B"


@pytest.mark.unit
def test_normalize_provider_name_qwen3_asr():
    spa = _import_module()
    registry = spa.SttProviderRegistry()

    # Test various aliases
    assert registry.normalize_provider_name("qwen3-asr") == "qwen3-asr"
    assert registry.normalize_provider_name("qwen3_asr") == "qwen3-asr"
    assert registry.normalize_provider_name("qwen3asr") == "qwen3-asr"
    assert registry.normalize_provider_name("Qwen3-ASR") == "qwen3-asr"


@pytest.mark.unit
def test_normalize_provider_name_additional_aliases():
    spa = _import_module()
    registry = spa.SttProviderRegistry()

    assert registry.normalize_provider_name("whisper") == "faster-whisper"
    assert registry.normalize_provider_name("vibevoice_asr") == "vibevoice"
    assert registry.normalize_provider_name("nemo-parakeet") == "parakeet"


@pytest.mark.unit
def test_default_provider_name_whisper_alias_maps_to_faster_whisper(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():
        return {"default_transcriber": "whisper"}

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)
    registry = spa.SttProviderRegistry()

    assert registry.get_default_provider_name() == "faster-whisper"


@pytest.mark.unit
def test_transcribe_batch_whisper_normalizes_artifact(monkeypatch, tmp_path):
    spa = _import_module()

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    def fake_speech_to_text(
        path,
        whisper_model,
        selected_source_lang,
        vad_filter,
        diarize,
        word_timestamps,
        return_language,
        initial_prompt=None,
        task="transcribe",
        base_dir=None,
        cancel_check=None,
    ):

        assert str(path) == str(audio_file)
        assert whisper_model == "tiny"
        assert selected_source_lang is None
        assert task == "transcribe"

        segments = [
            {"Text": "hello", "start_seconds": 0.0, "end_seconds": 0.5},
            {"Text": "world", "start_seconds": 0.5, "end_seconds": 1.0},
        ]
        return segments, "en"

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    monkeypatch.setattr(atlib, "speech_to_text", fake_speech_to_text)
    monkeypatch.setattr(atlib, "strip_whisper_metadata_header", lambda segs: segs)

    adapter = spa.FasterWhisperAdapter()
    artifact = adapter.transcribe_batch(
        str(audio_file),
        model="tiny",
        language=None,
        task="transcribe",
        word_timestamps=False,
    )

    assert artifact["text"] == "hello world"
    assert artifact["language"] == "en"
    assert isinstance(artifact["segments"], list)
    # Default diarization and usage contract
    assert artifact["diarization"]["enabled"] is False
    assert artifact["diarization"]["speakers"] is None
    assert artifact["usage"]["duration_ms"] is None


@pytest.mark.unit
def test_transcribe_batch_parakeet_normalizes_artifact(monkeypatch, tmp_path):
    spa = _import_module()

    audio_file = tmp_path / "sample_parakeet.wav"
    audio_file.write_bytes(b"\x00" * 1024)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    def fake_speech_to_text(
        path,
        whisper_model,
        selected_source_lang,
        vad_filter,
        diarize,
        return_language,
        base_dir=None,
        cancel_check=None,
    ):

        assert str(path) == str(audio_file)
        # Parakeet adapter encodes model name into whisper_model
        assert whisper_model == "parakeet-standard"
        segments = [
            {"Text": "parakeet", "start_seconds": 0.0, "end_seconds": 0.5},
            {"Text": "ok", "start_seconds": 0.5, "end_seconds": 1.0},
        ]
        return segments, "en"

    monkeypatch.setattr(atlib, "speech_to_text", fake_speech_to_text, raising=True)

    adapter = spa.ParakeetAdapter()
    artifact = adapter.transcribe_batch(
        str(audio_file),
        model="parakeet-standard",
        language="en",
    )

    assert artifact["text"] == "parakeet ok"
    assert artifact["language"] == "en"
    assert isinstance(artifact["segments"], list)
    assert artifact["metadata"]["provider"] == "parakeet"
    assert artifact["metadata"]["model"] == "parakeet-standard"
    assert artifact["diarization"]["enabled"] is False


@pytest.mark.unit
def test_transcribe_batch_canary_normalizes_artifact(monkeypatch, tmp_path):
    spa = _import_module()

    # Create a minimal valid WAV file for soundfile to read
    import numpy as np
    import soundfile as sf

    audio_file = tmp_path / "sample_canary.wav"
    data = np.zeros(1600, dtype="float32")
    sf.write(str(audio_file), data, 16000)

    # Provide a lightweight fake Nemo module so we don't depend on real Nemo.
    import sys
    fake_nemo_mod = types.ModuleType(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo"
    )

    def fake_transcribe_with_canary(audio_np, sample_rate, language, task="transcribe", target_language=None):

        assert sample_rate == 16000
        return "canary transcript"

    fake_nemo_mod.transcribe_with_canary = fake_transcribe_with_canary
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo",
        fake_nemo_mod,
    )

    adapter = spa.CanaryAdapter()
    artifact = adapter.transcribe_batch(
        str(audio_file),
        model="nemo-canary-1b",
        language="en",
    )

    assert artifact["text"] == "canary transcript"
    assert isinstance(artifact["segments"], list)
    assert artifact["segments"][0]["Text"] == "canary transcript"
    assert artifact["metadata"]["provider"] == "canary"
    assert artifact["diarization"]["enabled"] is False


@pytest.mark.unit
def test_transcribe_batch_external_normalizes_artifact(monkeypatch, tmp_path):
    spa = _import_module()

    audio_file = tmp_path / "sample_external.wav"
    audio_file.write_bytes(b"\x00" * 1024)

    # Stub external provider module to avoid real HTTP calls
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider as ext_mod

    def fake_transcribe_with_external_provider(
        path,
        provider_name="default",
        language=None,
        sample_rate=None,
        base_dir=None,
    ):

        assert str(path) == str(audio_file)
        assert base_dir is None
        return "external transcript"

    monkeypatch.setattr(
        ext_mod,
        "transcribe_with_external_provider",
        fake_transcribe_with_external_provider,
        raising=True,
    )

    adapter = spa.ExternalAdapter()
    artifact = adapter.transcribe_batch(
        str(audio_file),
        model="external:myprovider",
        language="en",
    )

    assert artifact["text"] == "external transcript"
    assert isinstance(artifact["segments"], list)
    assert artifact["segments"][0]["Text"] == "external transcript"
    assert artifact["metadata"]["provider"] == "external"
    assert artifact["metadata"]["external_provider_name"] == "myprovider"
    assert artifact["diarization"]["enabled"] is False


@pytest.mark.unit
def test_transcribe_batch_external_passes_base_dir(monkeypatch, tmp_path):
    spa = _import_module()

    audio_file = tmp_path / "external_base_dir.wav"
    audio_file.write_bytes(b"\x00" * 2048)
    base_dir = tmp_path / "base"
    base_dir.mkdir()

    captured = {}

    def fake_transcribe_with_external_provider(
        path,
        provider_name="default",
        language=None,
        sample_rate=None,
        base_dir=None,
    ):

        captured["path"] = str(path)
        captured["base_dir"] = base_dir
        return "external ok"

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider as ext_mod

    monkeypatch.setattr(
        ext_mod,
        "transcribe_with_external_provider",
        fake_transcribe_with_external_provider,
        raising=True,
    )

    adapter = spa.ExternalAdapter()
    artifact = adapter.transcribe_batch(
        str(audio_file),
        model="external:stub",
        language=None,
        base_dir=base_dir,
    )

    assert artifact["text"] == "external ok"
    assert captured["path"] == str(audio_file)
    assert captured["base_dir"] == base_dir


@pytest.mark.unit
def test_transcribe_batch_qwen3_asr_normalizes_artifact(monkeypatch, tmp_path):
    spa = _import_module()

    audio_file = tmp_path / "sample_qwen3.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import sys

    fake_qwen3_mod = types.ModuleType(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Qwen3ASR"
    )

    def fake_transcribe_with_qwen3_asr(
        audio_path,
        *,
        model_path=None,
        language=None,
        word_timestamps=False,
        base_dir=None,
        cancel_check=None,
    ):
        return {
            "text": "qwen3 transcript",
            "language": language or "en",
            "segments": [
                {"start_seconds": 0.0, "end_seconds": 1.0, "Text": "qwen3 transcript"}
            ],
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": 1000, "tokens": None},
            "metadata": {
                "provider": "qwen3-asr",
                "model": model_path or "./models/qwen3_asr/1.7B",
                "source": "local",
            },
        }

    fake_qwen3_mod.transcribe_with_qwen3_asr = fake_transcribe_with_qwen3_asr
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Qwen3ASR",
        fake_qwen3_mod,
    )

    def fake_get_stt_config():
        return {
            "qwen3_asr_enabled": True,
            "qwen3_asr_model_path": "./models/qwen3_asr/1.7B",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    adapter = spa.Qwen3ASRAdapter()
    artifact = adapter.transcribe_batch(
        str(audio_file),
        model="./models/qwen3_asr/1.7B",
        language="en",
    )

    assert artifact["text"] == "qwen3 transcript"
    assert artifact["language"] == "en"
    assert isinstance(artifact["segments"], list)
    assert artifact["segments"][0]["Text"] == "qwen3 transcript"
    assert artifact["metadata"]["provider"] == "qwen3-asr"
    assert artifact["diarization"]["enabled"] is False


@pytest.mark.unit
def test_transcribe_batch_qwen3_asr_with_word_timestamps(monkeypatch, tmp_path):
    spa = _import_module()

    audio_file = tmp_path / "sample_qwen3_timestamps.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import sys

    fake_qwen3_mod = types.ModuleType(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Qwen3ASR"
    )

    captured = {}

    def fake_transcribe_with_qwen3_asr(
        audio_path,
        *,
        model_path=None,
        language=None,
        word_timestamps=False,
        base_dir=None,
        cancel_check=None,
    ):
        captured["word_timestamps"] = word_timestamps
        artifact = {
            "text": "hello world",
            "language": "en",
            "segments": [
                {"start_seconds": 0.0, "end_seconds": 1.0, "Text": "hello world"}
            ],
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": 1000, "tokens": None},
            "metadata": {
                "provider": "qwen3-asr",
                "model": model_path or "./models/qwen3_asr/1.7B",
                "source": "local",
            },
        }
        if word_timestamps:
            artifact["words"] = [
                {"word": "hello", "start": 0.0, "end": 0.4},
                {"word": "world", "start": 0.5, "end": 1.0},
            ]
        return artifact

    fake_qwen3_mod.transcribe_with_qwen3_asr = fake_transcribe_with_qwen3_asr
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Qwen3ASR",
        fake_qwen3_mod,
    )

    def fake_get_stt_config():
        return {
            "qwen3_asr_enabled": True,
            "qwen3_asr_model_path": "./models/qwen3_asr/1.7B",
            "qwen3_asr_aligner_enabled": True,
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    adapter = spa.Qwen3ASRAdapter()
    artifact = adapter.transcribe_batch(
        str(audio_file),
        model="./models/qwen3_asr/1.7B",
        language="en",
        word_timestamps=True,
    )

    assert captured["word_timestamps"] is True
    assert "words" in artifact
    assert len(artifact["words"]) == 2
    assert artifact["words"][0]["word"] == "hello"


@pytest.mark.unit
def test_qwen3_asr_adapter_uses_config_default(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():
        return {
            "default_transcriber": "qwen3-asr",
            "qwen3_asr_model_path": "./custom/model/path",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    registry = spa.SttProviderRegistry()
    provider, model, variant = registry.resolve_provider_for_model(None)

    assert provider == "qwen3-asr"
    assert model == "./custom/model/path"
    assert variant is None
