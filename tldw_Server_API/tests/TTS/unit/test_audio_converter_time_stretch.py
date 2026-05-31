import pytest

from tldw_Server_API.app.core.TTS import audio_converter as ac_mod

pytestmark = pytest.mark.unit


def test_build_atempo_filter_splits_large_ratio():
    filter_spec = ac_mod.AudioConverter._build_atempo_filter(3.0)
    assert "atempo=2" in filter_spec
    assert "atempo=1.5" in filter_spec
    assert "," in filter_spec


@pytest.mark.asyncio
async def test_time_stretch_builds_ffmpeg_command(monkeypatch, tmp_path):
    captured = {}

    class _FakeProc:
        def __init__(self):
            self.returncode = 0

        async def communicate(self):
            return b"", b""

    async def _fake_cpe(*cmd, **_kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(ac_mod.asyncio, "create_subprocess_exec", _fake_cpe)

    in_path = tmp_path / "in.wav"
    in_path.write_bytes(b"audio")
    out_path = tmp_path / "out.wav"

    ok = await ac_mod.AudioConverter.time_stretch(in_path, out_path, 1.25)
    assert ok
    cmd = " ".join(captured["cmd"])
    assert "ffmpeg" in cmd
    assert "-filter:a" in captured["cmd"]
    assert "atempo=1.25" in cmd


@pytest.mark.asyncio
async def test_time_stretch_rejects_non_positive_ratio(tmp_path):
    in_path = tmp_path / "in.wav"
    in_path.write_bytes(b"audio")
    out_path = tmp_path / "out.wav"
    ok = await ac_mod.AudioConverter.time_stretch(in_path, out_path, 0)
    assert ok is False


@pytest.mark.asyncio
async def test_time_stretch_noop_copy_logs_generic_error_without_raw_exception(tmp_path):
    raw_marker = "RAW_COPY_SECRET_MARKER"
    messages = []
    sink_id = ac_mod.logger.add(messages.append, format="{message}")
    try:
        ok = await ac_mod.AudioConverter.time_stretch(
            tmp_path / f"{raw_marker}.wav",
            tmp_path / "out.wav",
            1.0,
        )
    finally:
        ac_mod.logger.remove(sink_id)

    assert ok is False
    log_output = "\n".join(messages)
    assert "Time-stretch noop copy failed" in log_output
    assert raw_marker not in log_output


@pytest.mark.asyncio
async def test_time_stretch_subprocess_exception_logs_generic_error_without_raw_exception(
    monkeypatch,
    tmp_path,
):
    raw_marker = "RAW_SUBPROCESS_SECRET_MARKER"
    messages = []

    async def _fake_cpe(*_cmd, **_kwargs):
        raise RuntimeError(raw_marker)

    monkeypatch.setattr(ac_mod.asyncio, "create_subprocess_exec", _fake_cpe)
    sink_id = ac_mod.logger.add(messages.append, format="{message}")
    try:
        in_path = tmp_path / "in.wav"
        in_path.write_bytes(b"audio")
        ok = await ac_mod.AudioConverter.time_stretch(in_path, tmp_path / "out.wav", 1.25)
    finally:
        ac_mod.logger.remove(sink_id)

    assert ok is False
    log_output = "\n".join(messages)
    assert "Time-stretch error" in log_output
    assert raw_marker not in log_output
