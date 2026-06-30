import os

import numpy as np
import pytest

from tldw_Server_API.app.core.TTS import streaming_audio_writer
from tldw_Server_API.app.core.TTS.streaming_audio_writer import StreamingAudioWriter


class _FailingClose:
    def __init__(self, secret_detail: str):
        self.secret_detail = secret_detail

    def close(self):
        raise RuntimeError(f"cleanup failed for {self.secret_detail}")


def _capture_logs(level: str):
    logged_messages: list[str] = []
    sink_id = streaming_audio_writer.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level=level,
    )
    return logged_messages, sink_id


def test_close_container_failure_log_sanitizes_exception_text():
    writer = StreamingAudioWriter(format="wav", sample_rate=24000)
    secret_detail = "/Users/example/private/container-token-sk-test"
    writer.container = _FailingClose(secret_detail)
    logged_messages, sink_id = _capture_logs("ERROR")

    try:
        writer.close()
    finally:
        streaming_audio_writer.logger.remove(sink_id)

    assert getattr(writer, "container", None) is not None
    assert any("Error closing container" in message for message in logged_messages)
    assert all(secret_detail not in message for message in logged_messages)
    assert all("cleanup failed for" not in message for message in logged_messages)
    assert all("RuntimeError" in message for message in logged_messages)


def test_close_output_buffer_failure_log_sanitizes_exception_text():
    writer = StreamingAudioWriter(format="wav", sample_rate=24000)
    secret_detail = "/Users/example/private/output-buffer-token-sk-test"
    writer.output_buffer = _FailingClose(secret_detail)
    logged_messages, sink_id = _capture_logs("ERROR")

    try:
        writer.close()
    finally:
        streaming_audio_writer.logger.remove(sink_id)

    assert getattr(writer, "output_buffer", None) is not None
    assert any("Error closing output buffer" in message for message in logged_messages)
    assert all(secret_detail not in message for message in logged_messages)
    assert all("cleanup failed for" not in message for message in logged_messages)
    assert all("RuntimeError" in message for message in logged_messages)


def test_close_wav_temp_file_removal_log_sanitizes_path_and_exception_text(monkeypatch):
    writer = StreamingAudioWriter(format="wav", sample_rate=24000)
    secret_path = "/Users/example/private/wav-token-sk-test.pcm"
    writer._wav_file_path = secret_path
    logged_messages, sink_id = _capture_logs("DEBUG")

    def fail_remove(path):
        raise OSError(f"cannot remove {path}")

    monkeypatch.setattr(streaming_audio_writer.os, "remove", fail_remove)

    try:
        writer.close()
    finally:
        streaming_audio_writer.logger.remove(sink_id)

    assert writer._wav_file_path is None
    assert any("Error removing WAV temp file" in message for message in logged_messages)
    assert all(secret_path not in message for message in logged_messages)
    assert all("cannot remove" not in message for message in logged_messages)
    assert all("OSError" in message for message in logged_messages)


def test_finalize_wav_file_cleanup_log_sanitizes_path_and_exception_text(monkeypatch, tmp_path):
    writer = StreamingAudioWriter(format="wav", sample_rate=24000)
    secret_path = tmp_path / "wav-finalizer-token-sk-test.pcm"
    secret_path.write_bytes(b"\x00\x00")
    writer._wav_file_path = str(secret_path)
    logged_messages, sink_id = _capture_logs("DEBUG")

    def fail_remove(path):
        raise OSError(f"cannot remove finalizer path {path}")

    monkeypatch.setattr(streaming_audio_writer.os, "remove", fail_remove)

    try:
        data = writer._finalize_wav_from_file()
    finally:
        streaming_audio_writer.logger.remove(sink_id)
        writer._wav_file_path = None
        secret_path.unlink(missing_ok=True)

    assert data.startswith(b"RIFF")
    removal_messages = [
        message for message in logged_messages if "Error removing temp WAV file" in message
    ]
    assert removal_messages
    assert all(str(secret_path) not in message for message in logged_messages)
    assert all("cannot remove finalizer path" not in message for message in logged_messages)
    assert all("OSError" in message for message in removal_messages)


def test_wav_spill_success_log_sanitizes_temp_path():
    writer = StreamingAudioWriter(
        format="wav",
        sample_rate=24000,
        max_in_memory_bytes=1024,
    )
    big_chunk = np.zeros(2000, dtype=np.int16)
    logged_messages, sink_id = _capture_logs("WARNING")

    try:
        writer.write_chunk(big_chunk)
        spill_path = writer._wav_file_path
    finally:
        streaming_audio_writer.logger.remove(sink_id)
        writer.close()

    assert spill_path
    assert not os.path.exists(spill_path)
    assert any(
        "StreamingAudioWriter WAV buffer spilled to disk" in message
        for message in logged_messages
    )
    assert all(spill_path not in message for message in logged_messages)


def test_wav_spill_failure_log_sanitizes_exception_text(monkeypatch):
    writer = StreamingAudioWriter(format="wav", sample_rate=24000)
    secret_detail = "/Users/example/private/spill-token-sk-test.pcm"
    logged_messages, sink_id = _capture_logs("ERROR")

    def fail_mkstemp(*args, **kwargs):
        raise RuntimeError(f"cannot create temp file at {secret_detail}")

    monkeypatch.setattr(streaming_audio_writer.tempfile, "mkstemp", fail_mkstemp)

    try:
        with pytest.raises(RuntimeError, match="cannot create temp file"):
            writer._spill_wav_buffer_to_file()
    finally:
        streaming_audio_writer.logger.remove(sink_id)
        writer.close()

    assert any("Failed to spill WAV buffer to temp file" in message for message in logged_messages)
    assert all(secret_detail not in message for message in logged_messages)
    assert all("cannot create temp file" not in message for message in logged_messages)
    assert all("RuntimeError" in message for message in logged_messages)
