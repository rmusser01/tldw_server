from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf  # type: ignore

import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_VibeVoice as vv
from tldw_Server_API.app.core.exceptions import STTExecutionPlanError


def _minimal_settings(tmp_path: Path) -> dict[str, Any]:
    return {
        "enabled": True,
        "model_id": "microsoft/VibeVoice-ASR",
        "device": "cpu",
        "dtype": "float32",
        "cache_dir": str(tmp_path / "models"),
        "allow_download": False,
        "sample_rate": 16000,
        "max_new_tokens": 128,
        "vllm_enabled": False,
        "vllm_base_url": "",
        "vllm_model_id": "microsoft/VibeVoice-ASR",
        "vllm_api_key": None,
        "vllm_timeout_seconds": 60,
    }


@pytest.mark.unit
def test_normalize_artifact_from_segments() -> None:
    raw_resp = {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker_id": 1},
            {"start": 1.0, "end": 2.0, "text": "world", "speaker_label": "SPEAKER_1"},
        ],
    }

    artifact = vv._normalize_artifact(
        raw_resp,
        duration_seconds=2.0,
        language_hint=None,
        model_id="microsoft/VibeVoice-ASR",
        source="local",
        hotwords=[],
    )

    assert artifact["text"] == "hello world"
    assert artifact["language"] == "en"
    assert artifact["diarization"]["enabled"] is True
    assert artifact["diarization"]["speakers"] == 1
    assert len(artifact["segments"]) == 2
    assert artifact["segments"][0]["speaker"] == "SPEAKER_1"


@pytest.mark.unit
def test_transcribe_prefers_vllm_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_file = tmp_path / "sample.wav"
    sf.write(str(audio_file), np.zeros(1600, dtype="float32"), 16000)

    settings = _minimal_settings(tmp_path)
    settings["vllm_enabled"] = True
    settings["vllm_base_url"] = "http://127.0.0.1:8000"

    monkeypatch.setattr(vv, "_resolve_settings", lambda: dict(settings), raising=True)

    captured: dict[str, Any] = {}

    def _fake_vllm(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "text": "vllm path",
            "language": "en",
            "segments": [{"start_seconds": 0.0, "end_seconds": 0.1, "Text": "vllm path"}],
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": 100, "tokens": None},
            "metadata": {"provider": "vibevoice", "model": "override", "source": "vllm_http"},
        }

    monkeypatch.setattr(vv, "_transcribe_via_vllm_http", _fake_vllm, raising=True)
    monkeypatch.setattr(vv, "_transcribe_local", lambda **_: (_ for _ in ()).throw(AssertionError("local called")), raising=True)

    artifact = vv.transcribe_with_vibevoice(str(audio_file), model_id="override")
    assert artifact["metadata"]["source"] == "vllm_http"
    assert captured["settings"]["model_id"] == "override"
    assert captured["settings"]["vllm_model_id"] == "override"


@pytest.mark.unit
def test_vllm_failure_falls_back_to_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _minimal_settings(tmp_path)
    settings["vllm_enabled"] = True
    settings["vllm_base_url"] = "http://127.0.0.1:8000"

    monkeypatch.setattr(vv, "_resolve_settings", lambda: dict(settings), raising=True)
    monkeypatch.setattr(vv, "_transcribe_via_vllm_http", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")), raising=True)
    monkeypatch.setattr(vv, "_load_audio", lambda *_args, **_kwargs: (np.zeros(1600, dtype="float32"), 16000, 0.1), raising=True)

    called = {"local": 0}

    def _fake_local(**kwargs: Any) -> dict[str, Any]:
        called["local"] += 1
        return {
            "text": "local path",
            "language": kwargs.get("language"),
            "segments": [{"start_seconds": 0.0, "end_seconds": 0.1, "Text": "local path"}],
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": 100, "tokens": None},
            "metadata": {"provider": "vibevoice", "model": kwargs["settings"]["model_id"], "source": "local"},
        }

    monkeypatch.setattr(vv, "_transcribe_local", _fake_local, raising=True)

    artifact = vv.transcribe_with_vibevoice(str(tmp_path / "audio.wav"))
    assert artifact["metadata"]["source"] == "local"
    assert called["local"] == 1


@pytest.mark.unit
def test_vllm_http_disables_redirects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    settings = _minimal_settings(tmp_path)
    settings["vllm_base_url"] = "http://127.0.0.1:8000"
    captured: dict[str, Any] = {}

    def fake_fetch_json(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"text": "redirect-safe"}

    from tldw_Server_API.app.core import http_client

    monkeypatch.setattr(
        http_client,
        "fetch_json",
        fake_fetch_json,
    )
    monkeypatch.setattr(
        vv,
        "_audio_duration_seconds",
        lambda _path: 1.0,
    )

    artifact = vv._transcribe_via_vllm_http(
        audio_path=audio,
        base_dir=tmp_path,
        settings=settings,
        language="en",
        hotwords=[],
        cancel_check=None,
    )

    assert captured["allow_redirects"] is False
    assert artifact["text"] == "redirect-safe"


@pytest.mark.unit
def test_planned_local_processor_never_drops_language_or_hotwords() -> None:
    calls: list[dict[str, Any]] = []

    def processor(**kwargs: Any) -> object:
        calls.append(kwargs)
        if "language" in kwargs or "hotwords" in kwargs:
            raise TypeError("unsupported optional input")
        return object()

    with pytest.raises(STTExecutionPlanError, match="semantics"):
        vv._build_processor_inputs(
            processor,
            audio_np=np.zeros(10, dtype="float32"),
            sample_rate=16000,
            language="en",
            hotwords=["private-hotword"],
            strict_semantics=True,
        )

    assert len(calls) == 1


@pytest.mark.unit
def test_planned_local_transcribe_never_retries_without_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class Model:
        def transcribe(self, _audio: object, **kwargs: Any) -> object:
            calls.append(kwargs)
            if "language" in kwargs:
                raise TypeError("language unsupported")
            return {"text": "silently dropped language"}

    monkeypatch.setattr(
        vv,
        "_load_local_components",
        lambda _settings: (object(), Model(), "cpu"),
    )

    with pytest.raises(STTExecutionPlanError, match="semantics"):
        vv._transcribe_local(
            audio_np=np.zeros(10, dtype="float32"),
            sample_rate=16000,
            duration_seconds=0.1,
            settings={
                "model_id": "local-model",
                "max_new_tokens": 10,
                "strict_semantics": True,
            },
            language="en",
            hotwords=[],
            cancel_check=None,
        )

    assert len(calls) == 1
