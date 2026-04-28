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
