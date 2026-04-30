import pytest

from tldw_Server_API.app.core.TTS import audio_converter as ac_mod

pytestmark = pytest.mark.unit


async def _assert_exception_log_is_sanitized(monkeypatch, raw_marker: str, call, expected_message: str):
    messages = []

    async def _fake_cpe(*_cmd, **_kwargs):
        raise RuntimeError(raw_marker)

    monkeypatch.setattr(ac_mod.asyncio, "create_subprocess_exec", _fake_cpe)
    sink_id = ac_mod.logger.add(messages.append, format="{message}")
    try:
        result = await call()
    finally:
        ac_mod.logger.remove(sink_id)

    log_output = "\n".join(messages)
    assert expected_message in log_output
    assert raw_marker not in log_output
    return result


@pytest.mark.asyncio
async def test_convert_to_wav_subprocess_exception_logs_generic_error_without_raw_exception(
    monkeypatch,
    tmp_path,
):
    raw_marker = "RAW_CONVERT_TO_WAV_SECRET_MARKER"

    result = await _assert_exception_log_is_sanitized(
        monkeypatch,
        raw_marker,
        lambda: ac_mod.AudioConverter.convert_to_wav(
            tmp_path / "in.mp3",
            tmp_path / "out.wav",
        ),
        "Audio conversion error",
    )

    assert result is False


@pytest.mark.asyncio
async def test_get_duration_subprocess_exception_logs_generic_error_without_raw_exception(
    monkeypatch,
    tmp_path,
):
    raw_marker = "RAW_GET_DURATION_SECRET_MARKER"

    result = await _assert_exception_log_is_sanitized(
        monkeypatch,
        raw_marker,
        lambda: ac_mod.AudioConverter.get_duration(tmp_path / "in.wav"),
        "Error getting duration",
    )

    assert result == 0.0


@pytest.mark.asyncio
async def test_trim_silence_subprocess_exception_logs_generic_error_without_raw_exception(
    monkeypatch,
    tmp_path,
):
    raw_marker = "RAW_TRIM_SILENCE_SECRET_MARKER"

    result = await _assert_exception_log_is_sanitized(
        monkeypatch,
        raw_marker,
        lambda: ac_mod.AudioConverter.trim_silence(
            tmp_path / "in.wav",
            tmp_path / "out.wav",
        ),
        "Silence trimming error",
    )

    assert result is False


@pytest.mark.asyncio
async def test_convert_format_subprocess_exception_logs_generic_error_without_raw_exception(
    monkeypatch,
    tmp_path,
):
    raw_marker = "RAW_CONVERT_FORMAT_SECRET_MARKER"

    result = await _assert_exception_log_is_sanitized(
        monkeypatch,
        raw_marker,
        lambda: ac_mod.AudioConverter.convert_format(
            tmp_path / "in.wav",
            tmp_path / "out.mp3",
            "mp3",
        ),
        "Format conversion error",
    )

    assert result is False


@pytest.mark.asyncio
async def test_validate_duration_exception_logs_generic_error_without_raw_exception(
    monkeypatch,
    tmp_path,
):
    raw_marker = "RAW_VALIDATE_DURATION_SECRET_MARKER"
    messages = []

    async def _failing_duration(_path):
        raise RuntimeError(raw_marker)

    monkeypatch.setattr(ac_mod.AudioConverter, "get_duration", staticmethod(_failing_duration))
    sink_id = ac_mod.logger.add(messages.append, format="{message}")
    try:
        is_valid, duration = await ac_mod.AudioConverter.validate_duration(tmp_path / "in.wav")
    finally:
        ac_mod.logger.remove(sink_id)

    log_output = "\n".join(messages)
    assert "Duration validation error" in log_output
    assert raw_marker not in log_output
    assert is_valid is False
    assert duration == 0.0


@pytest.mark.asyncio
async def test_get_audio_info_exception_logs_generic_error_without_raw_exception(
    monkeypatch,
    tmp_path,
):
    raw_marker = "RAW_GET_AUDIO_INFO_SECRET_MARKER"

    async def _duration(_path):
        return 0.0

    monkeypatch.setattr(ac_mod.AudioConverter, "get_duration", staticmethod(_duration))
    result = await _assert_exception_log_is_sanitized(
        monkeypatch,
        raw_marker,
        lambda: ac_mod.AudioConverter.get_audio_info(tmp_path / "in.wav"),
        "Error getting audio info",
    )

    assert result["duration"] == 0.0
    assert result["format"] == "wav"


@pytest.mark.asyncio
async def test_normalize_audio_subprocess_exception_logs_generic_error_without_raw_exception(
    monkeypatch,
    tmp_path,
):
    raw_marker = "RAW_NORMALIZE_AUDIO_SECRET_MARKER"

    result = await _assert_exception_log_is_sanitized(
        monkeypatch,
        raw_marker,
        lambda: ac_mod.AudioConverter.normalize_audio(
            tmp_path / "in.wav",
            tmp_path / "out.wav",
        ),
        "Audio normalization error",
    )

    assert result is False


@pytest.mark.asyncio
async def test_extract_segment_subprocess_exception_logs_generic_error_without_raw_exception(
    monkeypatch,
    tmp_path,
):
    raw_marker = "RAW_EXTRACT_SEGMENT_SECRET_MARKER"

    result = await _assert_exception_log_is_sanitized(
        monkeypatch,
        raw_marker,
        lambda: ac_mod.AudioConverter.extract_segment(
            tmp_path / "in.wav",
            tmp_path / "out.wav",
            start_time=0.0,
            duration=1.0,
        ),
        "Segment extraction error",
    )

    assert result is False


@pytest.mark.asyncio
async def test_resample_audio_subprocess_exception_logs_generic_error_without_raw_exception(
    monkeypatch,
    tmp_path,
):
    raw_marker = "RAW_RESAMPLE_AUDIO_SECRET_MARKER"

    result = await _assert_exception_log_is_sanitized(
        monkeypatch,
        raw_marker,
        lambda: ac_mod.AudioConverter.resample_audio(
            tmp_path / "in.wav",
            tmp_path / "out.wav",
            16000,
        ),
        "Resampling error",
    )

    assert result is False
