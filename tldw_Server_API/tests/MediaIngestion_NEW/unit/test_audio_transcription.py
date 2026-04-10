import json
import os
from pathlib import Path

import pytest
import importlib

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
    ConversionError,
    perform_transcription,
    speech_to_text,
    convert_to_wav,
    is_transcription_error_message,
    strip_whisper_metadata_header,
)
import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib


def _patch_transcript_cache_root(monkeypatch, tmp_path) -> Path:


    """
    Configure an isolated transcript cache root for tests.
    Disables allowed media base dir checks to simplify test setup.
    """
    temp_root = tmp_path / "temp_root"
    temp_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(atlib.tempfile, "gettempdir", lambda: str(temp_root))
    monkeypatch.setattr(atlib, "_ALLOWED_MEDIA_BASE_DIRS", None)
    return temp_root


@pytest.mark.unit
def test_coerce_float_supports_default_fallback():
    assert atlib._coerce_float("3.5", 0.0) == pytest.approx(3.5)
    assert atlib._coerce_float("not-a-number", 1.25) == pytest.approx(1.25)
    assert atlib._coerce_float(None, 2.5) == pytest.approx(2.5)
    assert atlib._coerce_float("not-a-number") is None


@pytest.mark.unit
def test_convert_to_wav_includes_duration(monkeypatch, tmp_path):
    input_file = tmp_path / "input.mp3"
    input_file.write_bytes(b"\x00" * 2048)

    commands = []

    atlib._FFMPEG_VERSION_CHECKED = False
    atlib._FFMPEG_CMD_FOR_VERSION = None

    def fake_run(cmd, *args, **kwargs):

        commands.append(cmd)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        # Simulate ffmpeg writing the output file during the conversion command
        if "-i" in cmd and cmd:
            output_target = Path(cmd[-1])
            output_target.write_bytes(b"RIFF")

        return Result()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.validate_audio_file",
        lambda path: (True, ""),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.subprocess.run",
        fake_run,
    )

    output_path = convert_to_wav(str(input_file), offset=5, end_time=9, overwrite=True)

    assert Path(output_path).suffix == ".wav"
    assert len(commands) >= 2  # first is version check
    conversion_cmd = commands[1]
    assert "-t" in conversion_cmd
    # duration should be end_time - offset => 4 seconds
    assert "4" in conversion_cmd


@pytest.mark.unit
def test_convert_to_wav_rejects_invalid_range(tmp_path):
    invalid_clip = tmp_path / "clip.mp4"
    invalid_clip.write_bytes(b"\x00" * 2048)
    with pytest.raises(ConversionError):
        convert_to_wav(str(invalid_clip), offset=10, end_time=9)


@pytest.mark.unit
def test_convert_to_wav_rejects_symlink_input(tmp_path):
    target = tmp_path / "real.mp3"
    target.write_bytes(b"\x00" * 2048)
    link = tmp_path / "link.mp3"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks not supported on this platform")

    with pytest.raises(ConversionError):
        convert_to_wav(str(link), overwrite=True)


@pytest.mark.unit
def test_convert_to_wav_respects_ffmpeg_path(monkeypatch, tmp_path):
    ffmpeg_path = tmp_path / "ffmpeg-bin"
    ffmpeg_path.write_text("#!/bin/sh\necho ffmpeg\n")

    input_file = tmp_path / "input.mp3"
    input_file.write_bytes(b"\x00" * 2048)

    commands = []

    def fake_run(cmd, *args, **kwargs):

        commands.append(cmd)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setenv("FFMPEG_PATH", str(ffmpeg_path))
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.validate_audio_file",
        lambda path: (True, ""),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.subprocess.run",
        fake_run,
    )

    output_path = convert_to_wav(str(input_file), overwrite=True)

    assert Path(output_path).name == "input.wav"
    # First call is version check, second is conversion
    assert commands[0][0] == str(ffmpeg_path)
    assert commands[1][0] == str(ffmpeg_path)


@pytest.mark.unit
def test_convert_to_wav_avoids_redundant_version_checks(monkeypatch, tmp_path):
    """
    convert_to_wav should avoid spawning an extra `ffmpeg -version`
    process on every call once a given ffmpeg binary has been verified.
    """
    input_file = tmp_path / "input_twice.mp3"
    input_file.write_bytes(b"\x00" * 2048)

    commands = []

    def fake_run(cmd, *args, **kwargs):

        commands.append(cmd)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        # Simulate ffmpeg writing the output file during the conversion command
        if "-i" in cmd and cmd:
            output_target = Path(cmd[-1])
            output_target.write_bytes(b"RIFF")

        return Result()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.validate_audio_file",
        lambda path: (True, ""),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.subprocess.run",
        fake_run,
    )

    # Reset version-check cache for deterministic assertions
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    atlib._FFMPEG_VERSION_CHECKED = False
    atlib._FFMPEG_CMD_FOR_VERSION = None

    # Call convert_to_wav twice; the ffmpeg -version probe should run once.
    convert_to_wav(str(input_file), overwrite=True)
    convert_to_wav(str(input_file), overwrite=True)

    version_checks = [cmd for cmd in commands if "-version" in cmd]
    assert len(version_checks) == 1


@pytest.mark.unit
def test_convert_to_wav_can_skip_prevalidation(monkeypatch, tmp_path):
    """
    When STT_SKIP_AUDIO_PREVALIDATION / skip_audio_prevalidation is enabled,
    convert_to_wav should not invoke validate_audio_file.
    """
    input_file = tmp_path / "input_skip.mp3"
    input_file.write_bytes(b"\x00" * 2048)

    commands = []

    def fake_run(cmd, *args, **kwargs):

        commands.append(cmd)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        # Simulate ffmpeg writing the output file during the conversion command
        if "-i" in cmd and cmd:
            output_target = Path(cmd[-1])
            output_target.write_bytes(b"RIFF")

        return Result()

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    # Ensure any calls to validate_audio_file would fail the test if prevalidation is not skipped.
    def _fail_validate(_):
        raise AssertionError("validate_audio_file should not be called when prevalidation is skipped")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.validate_audio_file",
        _fail_validate,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.subprocess.run",
        fake_run,
    )

    # Flip the module-level flag to skip prevalidation for this test.
    original_skip = atlib.SKIP_AUDIO_PREVALIDATION
    atlib.SKIP_AUDIO_PREVALIDATION = True
    try:
        output_path = convert_to_wav(str(input_file), overwrite=True)
        assert Path(output_path).suffix == ".wav"
    finally:
        atlib.SKIP_AUDIO_PREVALIDATION = original_skip


@pytest.mark.unit
def test_audio_transcription_lib_processing_choice_safe_when_config_missing(monkeypatch):
    """
    Audio_Transcription_Lib should not raise at import time
    when load_and_log_configs returns None or a dict without
    'processing_choice'. The module-level processing_choice
    should safely default to 'cpu'.
    """
    import tldw_Server_API.app.core.config as core_config
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    # Force load_and_log_configs used by the STT module to return None
    monkeypatch.setattr(core_config, "load_and_log_configs", lambda: None, raising=True)

    reloaded = importlib.reload(atlib)

    assert hasattr(reloaded, "processing_choice")
    assert reloaded.processing_choice == "cpu"


@pytest.mark.unit
def test_speech_to_text_persists_cache_files(monkeypatch, tmp_path):
    temp_root = _patch_transcript_cache_root(monkeypatch, tmp_path)
    audio_dir = temp_root / "inputs"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_file = audio_dir / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)
    cache_dir = temp_root / atlib.TRANSCRIPT_CACHE_DIR_NAME

    class _FakeSeg:
        def __init__(self, text="hello"):
            self.start = 0.0
            self.end = 1.0
            self.text = text

    class _FakeInfo:
        language = "en"
        language_probability = 0.9

    class _FakeModel:
        def transcribe(self, *_args, **_kwargs):
            return [_FakeSeg()], _FakeInfo()

    # Ensure the stub model is returned regardless of check_download_status flag
    monkeypatch.setattr(atlib, "get_whisper_model", lambda *_args, **_kwargs: _FakeModel())
    monkeypatch.setattr(atlib, "processing_choice", "cpu")

    segments = speech_to_text(str(audio_file), whisper_model="tiny", selected_source_lang="en")
    assert segments

    out_file = cache_dir / f"{audio_file.stem}-whisper_model-tiny.segments.json"
    pretty_file = cache_dir / f"{audio_file.stem}-whisper_model-tiny.segments_pretty.json"

    assert out_file.exists()
    payload = json.loads(out_file.read_text())
    assert payload.get("segments")
    assert pretty_file.exists()


@pytest.mark.unit
def test_perform_transcription_regenerates_on_invalid_cache(monkeypatch, tmp_path):
    temp_root = _patch_transcript_cache_root(monkeypatch, tmp_path)
    audio_dir = temp_root / "inputs"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_file = audio_dir / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    model_name = "fake-model"
    cache_dir = temp_root / atlib.TRANSCRIPT_CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_name_sanitized = atlib._sanitize_transcription_model_name(model_name)
    cache_path = cache_dir / f"{audio_file.stem}-transcription_model-{model_name_sanitized}.segments.json"
    cache_path.write_text("not valid json")

    regenerated = [{"Text": "regen"}]
    regen_called = {"called": False}

    def fake_convert_to_wav(_path, *_args, **_kwargs):

        return str(audio_file)

    monkeypatch.setattr(atlib, "convert_to_wav", fake_convert_to_wav)

    def fake_run_stt(
        _path,
        model,
        _vad_filter=False,
        selected_source_lang="en",
        **kwargs,
    ):

        assert kwargs.get("base_dir") is None
        regen_called["called"] = True
        return {
            "text": "regen",
            "language": selected_source_lang,
            "segments": regenerated,
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": None, "tokens": None},
            "metadata": {"provider": "faster-whisper", "model": model},
        }

    monkeypatch.setattr(atlib, "run_stt_batch_via_registry", fake_run_stt)

    audio_path, segments = perform_transcription(
        str(audio_file),
        offset=0,
        transcription_model=model_name,
        vad_use=False,
        overwrite=False,
        transcription_language="en",
    )

    assert regen_called["called"] is True
    assert audio_path == str(audio_file)
    assert segments == regenerated


@pytest.mark.unit
def test_prune_transcript_cache_limits_files(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import prune_transcript_cache

    cache_dir = tmp_path
    files = []
    for i in range(3):
        f = cache_dir / f"audio-whisper_model-tiny.segments{i}.json"
        f.write_text("data")
        # set older mtimes for earlier files
        os.utime(f, (f.stat().st_atime, f.stat().st_mtime - (i + 1) * 100))
        files.append(f)

    prune_transcript_cache(cache_dir, max_files_per_source=1)
    remaining = list(cache_dir.glob("audio-whisper_model-tiny.segments*.json"))
    assert len(remaining) == 1
    # newest file (i=0) should remain
    assert remaining[0].name == "audio-whisper_model-tiny.segments0.json"


@pytest.mark.unit
def test_prune_transcript_cache_age(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import prune_transcript_cache

    old_file = tmp_path / "audio-whisper_model-tiny.segments_old.json"
    new_file = tmp_path / "audio-whisper_model-tiny.segments_new.json"
    old_file.write_text("old")
    new_file.write_text("new")

    # Make old file older than 2 days
    two_days_seconds = 2 * 86400 + 10
    os.utime(old_file, (old_file.stat().st_atime - two_days_seconds, old_file.stat().st_mtime - two_days_seconds))

    prune_transcript_cache(tmp_path, max_age_days=1)
    assert not old_file.exists()
    assert new_file.exists()


@pytest.mark.unit
def test_speech_to_text_respects_persist_toggle(monkeypatch, tmp_path):
    temp_root = _patch_transcript_cache_root(monkeypatch, tmp_path)
    audio_dir = temp_root / "inputs"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_file = audio_dir / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)
    cache_dir = temp_root / atlib.TRANSCRIPT_CACHE_DIR_NAME

    class _Seg:
        start = 0.0
        end = 1.0
        text = "hello"

    class _Info:
        language = "en"
        language_probability = 0.9

    class _Model:
        def transcribe(self, *args, **kwargs):
            return [_Seg()], _Info()

    monkeypatch.setattr(atlib, "get_whisper_model", lambda *args, **kwargs: _Model())
    monkeypatch.setattr(atlib, "processing_choice", "cpu")

    segments = speech_to_text(str(audio_file), whisper_model="tiny", selected_source_lang="en", persist_segments=False)
    assert segments

    out_file = cache_dir / f"{audio_file.stem}-whisper_model-tiny.segments.json"
    assert not out_file.exists()


@pytest.mark.unit
def test_speech_to_text_rejects_symlink_input(tmp_path):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)
    link = tmp_path / "sample_link.wav"
    try:
        link.symlink_to(audio_file)
    except OSError:
        pytest.skip("symlinks not supported on this platform")

    with pytest.raises(ValueError):
        speech_to_text(str(link), whisper_model="tiny", selected_source_lang="en", persist_segments=False)


@pytest.mark.unit
def test_speech_to_text_rejects_path_outside_base_dir(tmp_path):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 16)
    base_dir = tmp_path / "allowed"
    base_dir.mkdir()

    with pytest.raises(ValueError):
        speech_to_text(
            str(audio_file),
            whisper_model="tiny",
            selected_source_lang="en",
            base_dir=base_dir,
        )


@pytest.mark.unit
def test_speech_to_text_rejects_path_outside_allowed_roots(monkeypatch, tmp_path):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()

    monkeypatch.setattr(atlib, "_get_allowed_media_base_dirs", lambda: [allowed_root])

    with pytest.raises(ValueError):
        speech_to_text(
            str(audio_file),
            whisper_model="tiny",
            selected_source_lang="en",
        )


@pytest.mark.unit
def test_validate_whisper_model_identifier_rejects_path(tmp_path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
        validate_whisper_model_identifier,
    )

    with pytest.raises(ValueError):
        validate_whisper_model_identifier(str(tmp_path / "model.bin"))


@pytest.mark.unit
def test_validate_whisper_model_identifier_allows_hf_id():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
        validate_whisper_model_identifier,
    )

    assert (
        validate_whisper_model_identifier("openai/whisper-large-v3")
        == "openai/whisper-large-v3"
    )


@pytest.mark.unit
def test_validate_qwen2audio_model_identifier_allows_hf_id():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
        validate_qwen2audio_model_identifier,
    )

    assert (
        validate_qwen2audio_model_identifier("Qwen/Qwen2-Audio-7B-Instruct")
        == "Qwen/Qwen2-Audio-7B-Instruct"
    )


@pytest.mark.unit
def test_validate_qwen2audio_model_identifier_rejects_path_outside_base(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
        validate_qwen2audio_model_identifier,
    )

    base_root = tmp_path / "models" / "Whisper"
    base_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(atlib, "WHISPER_MODEL_BASE_DIR", base_root)

    outside_root = tmp_path / "outside" / "qwen"
    outside_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        validate_qwen2audio_model_identifier(str(outside_root))


@pytest.mark.unit
def test_validate_qwen2audio_model_identifier_allows_local_path_under_base(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
        validate_qwen2audio_model_identifier,
    )

    base_root = tmp_path / "models" / "Whisper"
    model_path = base_root / "qwen-local"
    model_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(atlib, "WHISPER_MODEL_BASE_DIR", base_root)

    assert validate_qwen2audio_model_identifier(str(model_path)) == str(model_path.resolve(strict=False))


@pytest.mark.unit
def test_resolve_whisper_download_root_rejects_outside_base(monkeypatch, tmp_path):
    base_root = tmp_path / "models" / "Whisper"
    base_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(atlib, "WHISPER_MODEL_BASE_DIR", base_root)

    outside_root = tmp_path / "outside"
    with pytest.raises(ValueError):
        atlib._resolve_whisper_download_root(str(outside_root))


@pytest.mark.unit
def test_resolve_project_root_prefers_repo_root_over_nested_models_dir(monkeypatch, tmp_path):
    """
    Regression test: when both `<repo>/models` and
    `<repo>/tldw_Server_API/models` exist, project root resolution must pick
    the actual repo root so Whisper uses the shared top-level `./models`.
    """
    repo_root = tmp_path / "repo"
    (repo_root / ".git").mkdir(parents=True, exist_ok=True)
    (repo_root / "models").mkdir(parents=True, exist_ok=True)
    (repo_root / "tldw_Server_API" / "models").mkdir(parents=True, exist_ok=True)

    fake_module = (
        repo_root
        / "tldw_Server_API"
        / "app"
        / "core"
        / "Ingestion_Media_Processing"
        / "Audio"
        / "Audio_Transcription_Lib.py"
    )
    fake_module.parent.mkdir(parents=True, exist_ok=True)
    fake_module.touch()

    monkeypatch.setattr(atlib, "__file__", str(fake_module))

    assert atlib._resolve_project_root() == repo_root.resolve()


@pytest.mark.unit
def test_is_transcription_error_message_covers_external_provider_module():
    msg = "External provider module not available. Please check installation."
    assert is_transcription_error_message(msg) is True


@pytest.mark.unit
def test_is_transcription_error_message_covers_error_in_transcription():
    msg = "Error in transcription: underlying failure"
    assert is_transcription_error_message(msg) is True


@pytest.mark.unit
def test_strip_whisper_metadata_header_removes_prefix():
    header = (
        "This text was transcribed using whisper model: distil-large-v3\n"
        "Detected language: en\n\n"
        "Hello world"
    )
    segments = [{"Text": header, "start_seconds": 0.0, "end_seconds": 1.0}]
    out = strip_whisper_metadata_header(segments)
    assert out[0]["Text"] == "Hello world"


@pytest.mark.unit
def test_transcribe_audio_uses_safe_default_provider(monkeypatch):
    # Ensure that when no transcription_provider is given and config is missing
    # STT-Settings/default_transcriber, transcribe_audio falls back to
    # faster-whisper instead of raising KeyError.
    import numpy as np
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    # Return an empty config to simulate missing STT-Settings section
    monkeypatch.setattr(
        atlib,
        "get_stt_config",
        lambda: {},
    )

    # Stub speech_to_text so we don't run a real model
    called = {}

    def fake_speech_to_text(path, whisper_model="distil-large-v3", selected_source_lang=None, **kwargs):

        called["model"] = whisper_model
        return [{"Text": "hello"}]

    monkeypatch.setattr(atlib, "speech_to_text", fake_speech_to_text)

    audio_data = np.zeros(1600, dtype=np.float32)
    result = atlib.transcribe_audio(audio_data, transcription_provider=None, sample_rate=16000)

    assert result == "hello"
    # Default model comes from transcribe_audio's whisper_model argument
    assert called["model"] == "distil-large-v3"


@pytest.mark.unit
def test_speech_to_text_qwen2audio_disabled_falls_back_to_whisper(monkeypatch, tmp_path):
    """When Qwen2Audio is disabled via config, speech_to_text with a qwen2audio model
    should fall back to Whisper without raising."""
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    # Disable Qwen2Audio via config gating
    monkeypatch.setattr(
        atlib,
        "load_and_log_configs",
        lambda: {"STT-Settings": {"qwen2audio_enabled": "false"}},
    )

    class _Seg:
        start = 0.0
        end = 1.0
        text = "fallback whisper"

    class _Info:
        language = "en"
        language_probability = 0.9

    class _Model:
        def transcribe(self, *args, **kwargs):
            return [_Seg()], _Info()

    # Ensure Whisper fallback is cheap and deterministic
    monkeypatch.setattr(atlib, "get_whisper_model", lambda *args, **kwargs: _Model())
    monkeypatch.setattr(atlib, "processing_choice", "cpu")

    # Use a model name that routes to provider 'qwen2audio'
    segments = atlib.speech_to_text(str(audio_file), whisper_model="qwen2audio-test", selected_source_lang="en")
    assert isinstance(segments, list)
    assert segments


@pytest.mark.unit
def test_speech_to_text_qwen2audio_disabled_skips_audio_decode(monkeypatch, tmp_path):
    """When Qwen2Audio is disabled, route directly to Whisper fallback without decoding via librosa."""
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import sys
    import types
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    monkeypatch.setattr(
        atlib,
        "load_and_log_configs",
        lambda: {"STT-Settings": {"qwen2audio_enabled": "false"}},
    )

    class _Seg:
        start = 0.0
        end = 1.0
        text = "fallback whisper"

    class _Info:
        language = "en"
        language_probability = 0.9

    class _Model:
        def transcribe(self, *args, **kwargs):
            return [_Seg()], _Info()

    monkeypatch.setattr(atlib, "get_whisper_model", lambda *args, **kwargs: _Model())
    monkeypatch.setattr(atlib, "processing_choice", "cpu")

    decode_calls = {"count": 0}

    def _unexpected_decode(*_args, **_kwargs):
        decode_calls["count"] += 1
        raise AssertionError("librosa.load should not be called when Qwen2Audio is disabled")

    fake_librosa = types.SimpleNamespace(
        load=_unexpected_decode,
        get_duration=lambda **_kwargs: 0.0,
    )
    monkeypatch.setattr(atlib, "librosa", fake_librosa, raising=False)
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    segments = atlib.speech_to_text(
        str(audio_file),
        whisper_model="qwen2audio-test",
        selected_source_lang="en",
    )
    assert isinstance(segments, list)
    assert segments
    assert decode_calls["count"] == 0


@pytest.mark.unit
def test_speech_to_text_parakeet_failure_falls_back_to_valid_whisper_model(monkeypatch, tmp_path):
    """Parakeet fallback should use a valid faster-whisper model identifier."""
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    def fake_parakeet(*_args, **_kwargs):
        raise ImportError("No module named 'librosa'")

    class _Seg:
        start = 0.0
        end = 1.0
        text = "fallback whisper"

    class _Info:
        language = "en"
        language_probability = 0.9

    class _Model:
        def transcribe(self, *_args, **_kwargs):
            return [_Seg()], _Info()

    captured: dict[str, str] = {}

    def fake_get_whisper_model(model_name, device, check_download_status=False):
        captured["model_name"] = model_name
        return _Model()

    monkeypatch.setattr(atlib, "speech_to_text_parakeet", fake_parakeet)
    monkeypatch.setattr(atlib, "get_whisper_model", fake_get_whisper_model)
    monkeypatch.setattr(atlib, "processing_choice", "cpu")

    segments = atlib.speech_to_text(
        str(audio_file),
        whisper_model="parakeet-mlx",
        selected_source_lang="en",
    )

    assert captured.get("model_name") == "distil-large-v3"
    assert isinstance(segments, list)
    assert "fallback whisper" in (segments[0].get("Text") or "")


@pytest.mark.unit
def test_speech_to_text_does_not_return_model_downloading_sentinel(monkeypatch, tmp_path):
    """
    speech_to_text should always return real transcript segments,
    not a special 'model_downloading' sentinel payload.
    """
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    class _Seg:
        start = 0.0
        end = 1.0
        text = "hello world"

    class _Info:
        language = "en"
        language_probability = 0.9

    class _Model:
        def transcribe(self, *args, **kwargs):
            return [_Seg()], _Info()

    calls = []

    def fake_get_whisper_model(model_name, device, check_download_status=False):

        calls.append(check_download_status)
        return _Model()

    monkeypatch.setattr(atlib, "get_whisper_model", fake_get_whisper_model)
    monkeypatch.setattr(atlib, "processing_choice", "cpu")

    segments = atlib.speech_to_text(str(audio_file), whisper_model="tiny", selected_source_lang="en")

    # Ensure we only called get_whisper_model once without any download-status check
    assert calls == [False]
    assert isinstance(segments, list)
    assert len(segments) == 1
    assert segments[0].get("Text") == "This text was transcribed using whisper model: tiny\nDetected language: en\n\nhello world"


@pytest.mark.unit
def test_speech_to_text_return_language_consistent_for_whisper(monkeypatch, tmp_path):
    """When return_language=True, Whisper branch should return (segments, lang)."""
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    class _Seg:
        start = 0.0
        end = 1.0
        text = "hello"

    class _Info:
        language = "en"
        language_probability = 0.9

    class _Model:
        def transcribe(self, *args, **kwargs):
            return [_Seg()], _Info()

    monkeypatch.setattr(atlib, "get_whisper_model", lambda *args, **kwargs: _Model())
    monkeypatch.setattr(atlib, "processing_choice", "cpu")

    segments, lang = atlib.speech_to_text(
        str(audio_file),
        whisper_model="tiny",
        selected_source_lang=None,
        return_language=True,
    )

    assert isinstance(segments, list)
    assert lang == "en"


@pytest.mark.unit
def test_get_whisper_model_respects_compute_type_override(monkeypatch):
    """
    get_whisper_model should honor the STT-Settings.whisper_compute_type override
    when present, instead of always deriving compute_type from the device.
    """
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    captured = {}

    class _StubModel:
        def __init__(self, *args, **kwargs):
            captured["compute_type"] = kwargs.get("compute_type")

    # Ensure a clean cache so the stub is exercised
    atlib.whisper_model_cache.clear()
    monkeypatch.setattr(atlib, "WhisperModel", _StubModel)
    # Force an override that differs from the default CUDA/CPU heuristic
    monkeypatch.setattr(atlib, "WHISPER_COMPUTE_TYPE_OVERRIDE", "int8_float16", raising=False)

    _ = atlib.get_whisper_model("tiny", "cuda", check_download_status=False)
    assert captured.get("compute_type") == "int8_float16"


@pytest.mark.unit
def test_resolve_processing_choice_prefers_env_override(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    monkeypatch.setenv("PROCESSING_CHOICE", "cpu")

    assert atlib._resolve_processing_choice({"processing_choice": "cuda"}) == "cpu"


@pytest.mark.unit
def test_speech_to_text_return_language_consistent_for_parakeet(monkeypatch, tmp_path):
    """When return_language=True, Parakeet branch should return (segments, lang_or_none)."""
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    def fake_parakeet(audio_file_path, variant, selected_source_lang, vad_filter, base_dir=None):

        assert str(audio_file_path) == str(audio_file)
        assert variant == "standard"
        return [{"start_seconds": 0.0, "end_seconds": 1.0, "Text": "parakeet"}]

    monkeypatch.setattr(atlib, "speech_to_text_parakeet", fake_parakeet)

    segments, lang = atlib.speech_to_text(
        str(audio_file),
        whisper_model="parakeet-standard",
        selected_source_lang="de",
        return_language=True,
    )

    assert isinstance(segments, list)
    assert segments[0]["Text"] == "parakeet"
    # For non-Whisper providers we surface the selected_source_lang
    assert lang == "de"


@pytest.mark.unit
def test_speech_to_text_parakeet_mlx_uses_buffered_chunking_when_long(monkeypatch, tmp_path):
    pytest.importorskip("librosa")

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import librosa
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription as buffered_mod
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX as mlx_mod

    monkeypatch.setattr(atlib, "get_stt_config", lambda: {
        "mlx_chunk_duration": "30.0",
        "mlx_overlap_duration": "5.0",
        "buffered_merge_algo": "lcs",
    })
    monkeypatch.setattr(librosa, "get_duration", lambda path: 120.0)
    monkeypatch.setattr(mlx_mod, "check_mlx_available", lambda: True)

    captured = {}

    def fake_transcribe_long_audio(
        audio_path,
        model_name="parakeet",
        variant="mlx",
        chunk_duration=None,
        total_buffer=None,
        merge_algo="middle",
        **_kwargs,
    ):
        captured.update({
            "audio_path": audio_path,
            "model_name": model_name,
            "variant": variant,
            "chunk_duration": chunk_duration,
            "total_buffer": total_buffer,
            "merge_algo": merge_algo,
        })
        return "chunked text"

    monkeypatch.setattr(buffered_mod, "transcribe_long_audio", fake_transcribe_long_audio)

    segments = atlib.speech_to_text_parakeet(
        str(audio_file),
        variant="mlx",
        selected_source_lang="en",
        vad_filter=False,
    )

    assert captured["audio_path"] == str(audio_file)
    assert captured["chunk_duration"] == 30.0
    assert captured["total_buffer"] == 40.0
    assert captured["merge_algo"] == "lcs"
    assert segments[0]["Text"] == "chunked text"


@pytest.mark.unit
def test_speech_to_text_parakeet_mlx_passes_chunk_duration_when_short(monkeypatch, tmp_path):
    pytest.importorskip("librosa")

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import librosa
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription as buffered_mod
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX as mlx_mod

    monkeypatch.setattr(atlib, "get_stt_config", lambda: {
        "mlx_chunk_duration": "30.0",
        "mlx_overlap_duration": "5.0",
    })
    monkeypatch.setattr(librosa, "get_duration", lambda path: 10.0)
    monkeypatch.setattr(mlx_mod, "check_mlx_available", lambda: True)

    def fail_transcribe_long_audio(*_args, **_kwargs):
        raise AssertionError("Buffered chunking should not be used for short audio")

    monkeypatch.setattr(buffered_mod, "transcribe_long_audio", fail_transcribe_long_audio)

    captured = {}

    def fake_transcribe_with_parakeet_mlx(audio_path, **kwargs):
        captured.update(kwargs)
        return "short text"

    monkeypatch.setattr(mlx_mod, "transcribe_with_parakeet_mlx", fake_transcribe_with_parakeet_mlx)

    segments = atlib.speech_to_text_parakeet(
        str(audio_file),
        variant="mlx",
        selected_source_lang="en",
        vad_filter=False,
    )

    assert captured["chunk_duration"] == 30.0
    assert captured["overlap_duration"] == 5.0
    assert segments[0]["Text"] == "short text"


@pytest.mark.unit
def test_speech_to_text_parakeet_mlx_uses_structured_timing_when_available(monkeypatch, tmp_path):
    pytest.importorskip("librosa")

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import librosa
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX as mlx_mod

    monkeypatch.setattr(atlib, "get_stt_config", lambda: {
        "mlx_chunk_duration": "30.0",
        "mlx_overlap_duration": "5.0",
    })
    monkeypatch.setattr(librosa, "get_duration", lambda path: 10.0)
    monkeypatch.setattr(mlx_mod, "check_mlx_available", lambda: True)

    captured = {}

    def fake_transcribe_with_parakeet_mlx(audio_path, **kwargs):
        captured.update(kwargs)
        return {
            "text": "hello world",
            "sentences": [
                {
                    "text": "hello world",
                    "start": 1.1,
                    "end": 2.4,
                    "duration": 1.3,
                    "confidence": 0.88,
                    "tokens": [
                        {
                            "id": 1,
                            "text": "hello",
                            "start": 1.1,
                            "end": 1.6,
                            "duration": 0.5,
                            "confidence": 0.91,
                        },
                        {
                            "id": 2,
                            "text": " world",
                            "start": 1.6,
                            "end": 2.4,
                            "duration": 0.8,
                            "confidence": 0.85,
                        },
                    ],
                }
            ],
        }

    monkeypatch.setattr(mlx_mod, "transcribe_with_parakeet_mlx", fake_transcribe_with_parakeet_mlx)

    segments = atlib.speech_to_text_parakeet(
        str(audio_file),
        variant="mlx",
        selected_source_lang="en",
        vad_filter=False,
    )

    assert captured.get("return_structured") is True
    assert len(segments) == 1
    assert segments[0]["start_seconds"] == pytest.approx(1.1)
    assert segments[0]["end_seconds"] == pytest.approx(2.4)
    assert segments[0]["Text"] == "hello world"
    assert isinstance(segments[0].get("words"), list)
    assert segments[0]["words"][0]["word"] == "hello"
    assert segments[0]["words"][0]["start"] == pytest.approx(1.1)
    assert segments[0]["words"][0]["end"] == pytest.approx(1.6)


@pytest.mark.unit
def test_speech_to_text_parakeet_mlx_buffered_prefers_structured_merge_output(monkeypatch, tmp_path):
    pytest.importorskip("librosa")

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import librosa
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription as buffered_mod
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX as mlx_mod

    monkeypatch.setattr(atlib, "get_stt_config", lambda: {
        "mlx_chunk_duration": "30.0",
        "mlx_overlap_duration": "5.0",
    })
    monkeypatch.setattr(librosa, "get_duration", lambda path: 120.0)
    monkeypatch.setattr(mlx_mod, "check_mlx_available", lambda: True)

    captured = {}

    def fake_transcribe_long_audio(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "text": "chunked hello world",
            "sentences": [
                {
                    "text": "chunked hello world",
                    "start": 2.0,
                    "end": 4.0,
                    "duration": 2.0,
                    "confidence": 0.9,
                    "tokens": [
                        {"id": 1, "text": "chunked", "start": 2.0, "end": 2.6, "duration": 0.6, "confidence": 0.92},
                        {"id": 2, "text": " hello", "start": 2.6, "end": 3.2, "duration": 0.6, "confidence": 0.89},
                        {"id": 3, "text": " world", "start": 3.2, "end": 4.0, "duration": 0.8, "confidence": 0.88},
                    ],
                }
            ],
        }

    monkeypatch.setattr(buffered_mod, "transcribe_long_audio", fake_transcribe_long_audio)

    segments = atlib.speech_to_text_parakeet(
        str(audio_file),
        variant="mlx",
        selected_source_lang="en",
        vad_filter=False,
    )

    assert captured.get("return_structured") is True
    assert len(segments) == 1
    assert segments[0]["start_seconds"] == pytest.approx(2.0)
    assert segments[0]["end_seconds"] == pytest.approx(4.0)
    assert segments[0]["Text"] == "chunked hello world"
    assert segments[0]["words"][1]["word"] == "hello"


@pytest.mark.unit
def test_speech_to_text_parakeet_mlx_uses_buffered_chunking_when_duration_unknown(monkeypatch, tmp_path):
    pytest.importorskip("librosa")

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import librosa
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription as buffered_mod
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX as mlx_mod

    monkeypatch.setattr(atlib, "get_stt_config", lambda: {
        "mlx_chunk_duration": "30.0",
        "mlx_overlap_duration": "5.0",
    })

    def raise_duration(*_args, **_kwargs):
        raise RuntimeError("duration lookup failed")

    monkeypatch.setattr(librosa, "get_duration", raise_duration)
    monkeypatch.setattr(mlx_mod, "check_mlx_available", lambda: True)

    def fail_transcribe_with_parakeet_mlx(*_args, **_kwargs):
        raise AssertionError("Direct MLX path should not be used when duration is unknown")

    monkeypatch.setattr(mlx_mod, "transcribe_with_parakeet_mlx", fail_transcribe_with_parakeet_mlx)

    captured = {}

    def fake_transcribe_long_audio(*_args, **kwargs):
        captured.update(kwargs)
        return "chunked text"

    monkeypatch.setattr(buffered_mod, "transcribe_long_audio", fake_transcribe_long_audio)

    segments = atlib.speech_to_text_parakeet(
        str(audio_file),
        variant="mlx",
        selected_source_lang="en",
        vad_filter=False,
    )

    assert captured.get("chunk_duration") == 30.0
    assert segments[0]["Text"] == "chunked text"


@pytest.mark.unit
def test_speech_to_text_parakeet_mlx_falls_back_on_invalid_buffered_settings(monkeypatch, tmp_path):
    pytest.importorskip("librosa")

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import librosa
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription as buffered_mod
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX as mlx_mod

    monkeypatch.setattr(atlib, "get_stt_config", lambda: {
        "mlx_chunk_duration": "30.0",
        "mlx_overlap_duration": "5.0",
        "buffered_total_buffer": "10.0",
        "buffered_merge_algo": "bogus",
    })
    monkeypatch.setattr(librosa, "get_duration", lambda path: 120.0)
    monkeypatch.setattr(mlx_mod, "check_mlx_available", lambda: True)

    captured = {}

    def fake_transcribe_long_audio(*_args, **kwargs):
        captured.update(kwargs)
        return "chunked text"

    monkeypatch.setattr(buffered_mod, "transcribe_long_audio", fake_transcribe_long_audio)

    segments = atlib.speech_to_text_parakeet(
        str(audio_file),
        variant="mlx",
        selected_source_lang="en",
        vad_filter=False,
    )

    assert captured.get("total_buffer") is None
    assert captured.get("merge_algo") == "middle"
    assert segments[0]["Text"] == "chunked text"


@pytest.mark.unit
def test_speech_to_text_parakeet_mlx_defaults_when_chunk_duration_missing(monkeypatch, tmp_path):
    pytest.importorskip("librosa")

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import librosa
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription as buffered_mod
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX as mlx_mod

    monkeypatch.setattr(atlib, "get_stt_config", lambda: {})
    monkeypatch.setattr(librosa, "get_duration", lambda path: 120.0)
    monkeypatch.setattr(mlx_mod, "check_mlx_available", lambda: True)

    captured = {}

    def fake_transcribe_long_audio(*_args, **kwargs):
        captured.update(kwargs)
        return "chunked text"

    monkeypatch.setattr(buffered_mod, "transcribe_long_audio", fake_transcribe_long_audio)

    segments = atlib.speech_to_text_parakeet(
        str(audio_file),
        variant="mlx",
        selected_source_lang="en",
        vad_filter=False,
    )

    assert captured.get("chunk_duration") == 30.0
    assert captured.get("total_buffer") == 40.0
    assert segments[0]["Text"] == "chunked text"


@pytest.mark.unit
def test_speech_to_text_parakeet_mlx_disables_chunking_when_zero(monkeypatch, tmp_path):
    pytest.importorskip("librosa")

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import librosa
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription as buffered_mod
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX as mlx_mod

    monkeypatch.setattr(atlib, "get_stt_config", lambda: {
        "mlx_chunk_duration": "0",
        "mlx_overlap_duration": "0",
    })
    monkeypatch.setattr(librosa, "get_duration", lambda path: 120.0)
    monkeypatch.setattr(mlx_mod, "check_mlx_available", lambda: True)

    def fail_transcribe_long_audio(*_args, **_kwargs):
        raise AssertionError("Buffered chunking should be disabled when chunk duration is zero")

    monkeypatch.setattr(buffered_mod, "transcribe_long_audio", fail_transcribe_long_audio)

    captured = {}

    def fake_transcribe_with_parakeet_mlx(_audio_path, **kwargs):
        captured.update(kwargs)
        return "direct text"

    monkeypatch.setattr(mlx_mod, "transcribe_with_parakeet_mlx", fake_transcribe_with_parakeet_mlx)

    segments = atlib.speech_to_text_parakeet(
        str(audio_file),
        variant="mlx",
        selected_source_lang="en",
        vad_filter=False,
    )

    assert captured.get("chunk_duration") is None
    assert captured.get("overlap_duration") == 15.0
    assert segments[0]["Text"] == "direct text"


@pytest.mark.unit
def test_speech_to_text_return_language_consistent_for_qwen2audio(monkeypatch, tmp_path):
    """When return_language=True, Qwen2Audio branch should return (segments, lang_or_none)."""
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    def fake_qwen(audio_file_path, selected_source_lang, vad_filter, base_dir=None):

        assert str(audio_file_path) == str(audio_file)
        return [{"start_seconds": 0.0, "end_seconds": 1.0, "Text": "qwen"}]

    monkeypatch.setattr(atlib, "speech_to_text_qwen2audio", fake_qwen)

    segments, lang = atlib.speech_to_text(
        str(audio_file),
        whisper_model="qwen2audio-test",
        selected_source_lang=None,
        return_language=True,
    )

    assert isinstance(segments, list)
    assert segments[0]["Text"] == "qwen"
    # selected_source_lang was None, so language is None
    assert lang is None


@pytest.mark.unit
def test_to_normalized_stt_artifact_basic():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (  # type: ignore
        to_normalized_stt_artifact,
    )

    segments = [
        {"Text": "hello", "start_seconds": 0.0, "end_seconds": 0.5},
        {"Text": "world", "start_seconds": 0.5, "end_seconds": 1.0},
    ]
    artifact = to_normalized_stt_artifact(
        text="hello world",
        segments=segments,
        language="en",
        provider="faster-whisper",
        model="tiny",
        duration_seconds=1.0,
    )

    assert artifact["text"] == "hello world"
    assert artifact["language"] == "en"
    assert isinstance(artifact["segments"], list)
    assert artifact["usage"]["duration_ms"] == 1000
    assert artifact["metadata"]["provider"] == "faster-whisper"
    assert artifact["metadata"]["model"] == "tiny"
    assert artifact["diarization"]["enabled"] is False
    assert artifact["diarization"]["speakers"] is None
