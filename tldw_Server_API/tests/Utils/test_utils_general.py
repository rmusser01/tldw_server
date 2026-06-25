import builtins
import tempfile
import zipfile
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Utils import System_Checks_Lib
from tldw_Server_API.app.core.Utils import Utils


class _TqdmStub:
    def __init__(self, *_, **__):
        pass

    def update(self, _amount):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_file_resumes_without_truncation(monkeypatch, tmp_path):
    dest = tmp_path / "file.bin"
    # http_client.download uses a ".part" suffix for resume writes
    part_path = dest.with_suffix(dest.suffix + ".part")
    part_path.write_bytes(b"12345")

    # Stub the centralized downloader used by Utils.download_file
    def fake_download(*, url, dest, resume=False, retry=None, **_kwargs):  # signature-compatible
        assert resume is True
        dpath = Path(dest)
        tpath = dpath.with_suffix(dpath.suffix + ".part")
        # Simulate server returning the remaining bytes and atomic rename
        with open(tpath, "ab") as f:
            f.write(b"67890")
        Path(tpath).replace(dpath)
        return dpath

    monkeypatch.setattr(Utils, "download", fake_download)
    monkeypatch.setattr(Utils, "tqdm", _TqdmStub, raising=False)

    Utils.download_file("https://example.com/file.bin", str(dest))

    assert dest.read_bytes() == b"1234567890"
    # Ensure the temporary part file is cleaned up
    assert not part_path.exists()


def test_extract_text_from_segments_collects_all_segments():
    segments = [
        {"Time_Start": 0, "Time_End": 1, "Text": "First"},
        {"Time_Start": 1, "Time_End": 2, "Text": "Second"},
    ]
    result = Utils.extract_text_from_segments(segments, include_timestamps=True)
    assert result.splitlines() == [
        "0s - 1s | First",
        "1s - 2s | Second",
    ]


def test_extract_text_from_segments_handles_nested_dict():
    segments = {
        "outer": {
            "inner": {
                "Time_Start": 2,
                "Time_End": 3,
                "Text": "Nested text",
            }
        }
    }
    result = Utils.extract_text_from_segments(segments, include_timestamps=False)
    assert result == "Nested text"


def test_extract_text_from_segments_rounds_timestamps_to_hundredths():
    segments = [
        {
            "Time_Start": 0.835524770187703,
            "Time_End": 1.0966262608713602,
            "Text": "What?",
        }
    ]
    result = Utils.extract_text_from_segments(segments, include_timestamps=True)
    assert result == "0.84s - 1.1s | What?"


def test_extract_text_from_segments_trace_logs_shape_not_raw_text(monkeypatch):
    trace_messages = []

    class Recorder:
        def trace(self, message):
            trace_messages.append(str(message))

        def error(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(Utils, "logger", Recorder())

    result = Utils.extract_text_from_segments(
        [{"Time_Start": 0, "Time_End": 1, "Text": "private transcript"}],
        include_timestamps=False,
    )

    assert result == "private transcript"
    assert trace_messages
    assert all("private transcript" not in message for message in trace_messages)


def test_extract_text_from_segments_error_logs_shape_not_raw_text(monkeypatch):
    error_messages = []

    class Recorder:
        def trace(self, *_args, **_kwargs):
            pass

        def error(self, message):
            error_messages.append(str(message))

    recorder = Recorder()
    monkeypatch.setattr(Utils, "logger", recorder)
    monkeypatch.setattr(Utils, "logging", recorder)

    result = Utils.extract_text_from_segments(
        [{"Caption": "private transcript"}],
        include_timestamps=False,
    )

    assert result == "Error: Unable to extract transcription"
    assert error_messages
    assert all("private transcript" not in message for message in error_messages)


def test_save_temp_file_normalizes_and_preserves_content(monkeypatch):
    class DummyUpload:
        def __init__(self, name, data):
            self.name = name
            self._data = data

        def read(self):
            return self._data

    upload = DummyUpload("../evil.txt", b"payload")
    saved_path = Utils.save_temp_file(upload)

    temp_dir = Path(tempfile.gettempdir()).resolve()
    resolved_saved = Path(saved_path).resolve()

    assert resolved_saved.parent == temp_dir
    assert resolved_saved.exists()
    assert b"payload" == resolved_saved.read_bytes()

    resolved_saved.unlink()


def test_safe_read_file_handles_empty_decodes(monkeypatch):
    class FakeBytes(bytes):
        def decode(self, encoding="utf-8", errors="strict"):
            return ""

    class DummyFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return FakeBytes(b"data")

    monkeypatch.setattr(builtins, "open", lambda *_args, **_kwargs: DummyFile())
    monkeypatch.setattr(Utils.chardet, "detect", lambda _raw: {"encoding": "ascii"})

    result = Utils.safe_read_file("dummy-path")

    assert isinstance(result, str)
    assert "Unable to decode" in result


def test_download_file_checksum_mismatch_cleans_up(monkeypatch, tmp_path):
    dest = tmp_path / "file.bin"

    def fake_download(*, url, dest, resume=False, retry=None, **_kwargs):
        Path(dest).write_bytes(b"bad-content")
        return Path(dest)

    monkeypatch.setattr(Utils, "download", fake_download)

    with pytest.raises(ValueError):
        Utils.download_file(
            "https://example.com/file.bin",
            str(dest),
            expected_checksum="0" * 64,
        )

    assert not dest.exists()


def test_zip_validator_rejects_windows_traversal(tmp_path):
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("..\\evil.txt", "bad")

    is_valid, error, _ = Utils.ZipValidator.validate_zip_file(str(zip_path))

    assert is_valid is False
    assert "Invalid file paths detected" in error


def test_sanitize_filename_fallback_for_empty_input():
    sanitized = Utils.sanitize_filename('<>:"/\\|?*')

    assert sanitized == "untitled"


def test_format_metadata_as_text_accepts_duration_string():
    metadata = {"duration": "01:02:03"}

    text = Utils.format_metadata_as_text(metadata)

    assert "Duration: 01:02:03" in text


def test_extract_media_id_strips_trailing_punctuation():
    msg = "Success. Media ID: abc123."

    assert Utils.extract_media_id_from_result_string(msg) == "abc123"


def test_decide_cpugpu_defaults_on_eof(monkeypatch):
    System_Checks_Lib.processing_choice = "cpu"
    monkeypatch.setattr(System_Checks_Lib, "input", _raise_eof, raising=False)

    selection = System_Checks_Lib.decide_cpugpu()

    assert selection == "cpu"


def test_check_ffmpeg_handles_unknown_os(monkeypatch):
    System_Checks_Lib.userOS = "Unknown"
    monkeypatch.setattr(System_Checks_Lib.shutil, "which", lambda *_: None)
    monkeypatch.setattr(System_Checks_Lib.os.path, "exists", lambda *_: False)
    monkeypatch.setattr(System_Checks_Lib.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(System_Checks_Lib, "input", _raise_eof, raising=False)

    result = System_Checks_Lib.check_ffmpeg()

    assert result is False


def test_cuda_check_uses_resolved_direct_subprocess_invocation(monkeypatch):
    System_Checks_Lib.processing_choice = "cpu"
    recorded = {}

    def _fake_check_output(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return "GPU info line\nCUDA Version: 12.4\n"

    monkeypatch.setattr(
        System_Checks_Lib.shutil,
        "which",
        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    )
    monkeypatch.setattr(System_Checks_Lib.subprocess, "check_output", _fake_check_output)

    result = System_Checks_Lib.cuda_check()

    assert result is True
    assert System_Checks_Lib.processing_choice == "cuda"
    assert recorded["cmd"] == ["/usr/bin/nvidia-smi"]
    assert "shell" not in recorded["kwargs"]


def test_download_ffmpeg_does_not_download_unverified_binary(monkeypatch):
    calls = []

    def fake_download(**_kwargs):
        calls.append("download")
        raise AssertionError("download should not be called")

    monkeypatch.setattr(System_Checks_Lib, "input", lambda *_args, **_kwargs: "y", raising=False)
    monkeypatch.setattr(System_Checks_Lib, "download", fake_download, raising=False)

    assert System_Checks_Lib.download_ffmpeg() is False
    assert calls == []


def test_utils_no_user_database_path_placeholder():
    assert not hasattr(Utils, "get_user_database_path")


def _raise_eof(*_args, **_kwargs):
    raise EOFError()
