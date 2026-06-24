import io
import sys
import types

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException

from tldw_Server_API.app.core.Audio import tokenizer_service


def _install_fake_tokenizer_module(monkeypatch):
    module = types.ModuleType("qwen_tts")

    class FakeTokenizer:
        total_upsample = 2

        def __init__(self, model=None, **_kwargs):
            self.model = model

        def __call__(self, codes):
            return codes

        def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
            context_size = left_context_size
            return codes[..., context_size * self.total_upsample:]

    module.Qwen3TTSTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "qwen_tts", module)
    return FakeTokenizer


def test_serialize_audio_output_wav_wraps_numpy_audio():
    pcm = np.array([0, 1000, -1000, 0], dtype=np.int16)
    wav_bytes = tokenizer_service._serialize_audio_output(pcm, 24000, "wav")
    assert wav_bytes[:4] == b"RIFF"
    decoded, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="int16")
    assert sample_rate == 24000
    assert decoded.size > 0


def test_serialize_audio_output_wav_passthrough():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(240, dtype=np.float32), 24000, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    output = tokenizer_service._serialize_audio_output(wav_bytes, 24000, "wav")
    assert output == wav_bytes


def test_load_qwen3_tokenizer_instantiates_fake_backend(monkeypatch):
    tokenizer_cls = _install_fake_tokenizer_module(monkeypatch)

    tokenizer = tokenizer_service._load_qwen3_tokenizer(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz/",
        allow_download=False,
    )
    assert isinstance(tokenizer, tokenizer_cls)
    assert tokenizer.model == "Qwen/Qwen3-TTS-Tokenizer-12Hz/"


def test_load_qwen3_tokenizer_uses_from_pretrained_when_available(monkeypatch):
    module = types.ModuleType("qwen_tts")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, local_files_only=False):
            instance = cls()
            instance.model = model_id
            instance.local_files_only = local_files_only
            return instance

    module.Qwen3TTSTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "qwen_tts", module)

    tokenizer = tokenizer_service._load_qwen3_tokenizer("Qwen/Qwen3-TTS-Tokenizer-12Hz", allow_download=False)
    assert tokenizer.model == "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    assert tokenizer.local_files_only is True

    tokenizer = tokenizer_service._load_qwen3_tokenizer("Qwen/Qwen3-TTS-Tokenizer-12Hz", allow_download=True)
    assert tokenizer.model == "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    assert tokenizer.local_files_only is False


def test_load_qwen3_tokenizer_fails_closed_when_local_only_not_supported(monkeypatch):
    module = types.ModuleType("qwen_tts")

    class DownloadOnlyTokenizer:
        @classmethod
        def from_pretrained(cls, model_id):
            instance = cls()
            instance.model = model_id
            return instance

    module.Qwen3TTSTokenizer = DownloadOnlyTokenizer
    monkeypatch.setitem(sys.modules, "qwen_tts", module)

    with pytest.raises(HTTPException) as exc_info:
        tokenizer_service._load_qwen3_tokenizer("Qwen/Qwen3-TTS-Tokenizer-12Hz", allow_download=False)

    assert exc_info.value.status_code == 501
    assert "local-only" in str(exc_info.value.detail)

    tokenizer = tokenizer_service._load_qwen3_tokenizer("Qwen/Qwen3-TTS-Tokenizer-12Hz", allow_download=True)
    assert tokenizer.model == "Qwen/Qwen3-TTS-Tokenizer-12Hz"


def test_load_qwen3_tokenizer_loader_function_fails_closed_without_download_permission(monkeypatch):
    module = types.ModuleType("qwen_tts")

    def load_tokenizer(model_id):
        return {"model": model_id}

    module.load_tokenizer = load_tokenizer
    monkeypatch.setitem(sys.modules, "qwen_tts", module)

    with pytest.raises(HTTPException) as exc_info:
        tokenizer_service._load_qwen3_tokenizer("Qwen/Qwen3-TTS-Tokenizer-12Hz", allow_download=False)

    assert exc_info.value.status_code == 501
    assert "local-only" in str(exc_info.value.detail)

    tokenizer = tokenizer_service._load_qwen3_tokenizer("Qwen/Qwen3-TTS-Tokenizer-12Hz", allow_download=True)
    assert tokenizer == {"model": "Qwen/Qwen3-TTS-Tokenizer-12Hz"}


def test_load_qwen3_tokenizer_loader_function_honors_local_files_only(monkeypatch):
    module = types.ModuleType("qwen_tts")

    def load_tokenizer(model_id, local_files_only=False):
        return {"model": model_id, "local_files_only": local_files_only}

    module.load_tokenizer = load_tokenizer
    monkeypatch.setitem(sys.modules, "qwen_tts", module)

    tokenizer = tokenizer_service._load_qwen3_tokenizer("Qwen/Qwen3-TTS-Tokenizer-12Hz", allow_download=False)
    assert tokenizer == {"model": "Qwen/Qwen3-TTS-Tokenizer-12Hz", "local_files_only": True}

    tokenizer = tokenizer_service._load_qwen3_tokenizer("Qwen/Qwen3-TTS-Tokenizer-12Hz", allow_download=True)
    assert tokenizer == {"model": "Qwen/Qwen3-TTS-Tokenizer-12Hz", "local_files_only": False}


def test_decode_base64_payload_rejects_invalid_input():
    with pytest.raises(HTTPException) as exc_info:
        tokenizer_service._decode_base64_payload("not valid base64", request_id="req-tokenizer")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["request_id"] == "req-tokenizer"


def test_read_audio_from_raw_pcm_rejects_invalid_sample_rate():
    with pytest.raises(HTTPException) as exc_info:
        tokenizer_service._read_audio_from_bytes(b"\x00\x00", sample_rate_hint=0, request_id="req-tokenizer")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["request_id"] == "req-tokenizer"


def test_read_audio_from_raw_pcm_rejects_odd_byte_length():
    with pytest.raises(HTTPException) as exc_info:
        tokenizer_service._read_audio_from_bytes(b"\x00", sample_rate_hint=24000, request_id="req-tokenizer")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["request_id"] == "req-tokenizer"


def test_load_qwen3_tokenizer_sanitizes_missing_package_detail(monkeypatch):
    def _raise_missing_package(name):
        assert name == "qwen_tts"
        raise ModuleNotFoundError("qwen_tts unavailable at /private/qwen/token")

    monkeypatch.setattr(tokenizer_service.importlib, "import_module", _raise_missing_package)

    with pytest.raises(HTTPException) as exc_info:
        tokenizer_service._load_qwen3_tokenizer("Qwen/Qwen3-TTS-Tokenizer-12Hz", allow_download=False)

    assert exc_info.value.status_code == 501
    assert exc_info.value.detail == "qwen-tts package not available"
    assert "/private/qwen/token" not in str(exc_info.value.detail)


def test_load_qwen3_tokenizer_maps_model_path_error(monkeypatch):
    module = types.ModuleType("qwen_tts")

    class BadTokenizer:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs
            raise RuntimeError("HFValidationError: Repo id must be in the form 'repo_name'")

    module.Qwen3TTSTokenizer = BadTokenizer
    monkeypatch.setitem(sys.modules, "qwen_tts", module)

    with pytest.raises(RuntimeError, match="Repo id must be in the form 'repo_name'"):
        tokenizer_service._load_qwen3_tokenizer("Qwen/Bad-Tokenizer/", allow_download=False)


def test_load_qwen3_tokenizer_maps_rope_compat_error(monkeypatch):
    module = types.ModuleType("qwen_tts")

    class BadTokenizer:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs
            raise RuntimeError("KeyError: 'default' while setting rope")

    module.Qwen3TTSTokenizer = BadTokenizer
    monkeypatch.setitem(sys.modules, "qwen_tts", module)

    with pytest.raises(RuntimeError, match="KeyError: 'default' while setting rope"):
        tokenizer_service._load_qwen3_tokenizer("Qwen/Tokenizer", allow_download=False)
