from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio_files


class _FakeResponse:
    def __init__(self, headers: dict, chunks: list[bytes]):
        self.headers = headers
        self._chunks = chunks
        self.iter_content_calls = 0

    def iter_content(self, chunk_size: int):
        self.iter_content_calls += 1
        yield from self._chunks

    def raise_for_status(self) -> None:

        return


@pytest.mark.unit
def test_download_audio_aborts_when_stream_exceeds_limit(monkeypatch, tmp_path):
    """Ensure streaming downloads stop once they exceed configured limit."""
    monkeypatch.setattr(audio_files, "MAX_FILE_SIZE", 1024)
    dummy_uuid = SimpleNamespace(hex="1234567890abcdef")
    monkeypatch.setattr(audio_files.uuid, "uuid4", lambda: dummy_uuid)

    faux_response = _FakeResponse(
        headers={"content-type": "audio/mpeg"},
        chunks=[b"a" * 600, b"b" * 600],
    )
    with pytest.raises(audio_files.AudioFileSizeError):
        audio_files.download_audio_file(
            "https://example.com/file.mp3",
            str(tmp_path),
            downloader=lambda *_, **__: faux_response,
        )

    expected_path = tmp_path / "file_12345678.mp3"
    assert not expected_path.exists()


@pytest.mark.unit
def test_download_audio_rejects_when_content_length_exceeds_limit(monkeypatch, tmp_path):
    """Ensure header-declared oversize files fail fast before streaming."""
    monkeypatch.setattr(audio_files, "MAX_FILE_SIZE", 1024)
    dummy_uuid = SimpleNamespace(hex="abcdef1234567890")
    monkeypatch.setattr(audio_files.uuid, "uuid4", lambda: dummy_uuid)
    calls = []

    faux_response = _FakeResponse(
        headers={"content-length": "2048", "content-type": "audio/mpeg"},
        chunks=[b"x"],
    )

    def fake_downloader(*args, **kwargs):
        calls.append((args, kwargs))
        return faux_response

    with pytest.raises(audio_files.AudioFileSizeError):
        audio_files.download_audio_file(
            "https://example.com/file.mp3",
            str(tmp_path),
            downloader=fake_downloader,
        )

    expected_path = tmp_path / "file_abcdef12.mp3"
    assert not expected_path.exists()
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("https://example.com/file.mp3",)
    assert kwargs["stream"] is True
    assert faux_response.iter_content_calls == 0


@pytest.mark.unit
def test_download_audio_rejects_unexpected_content_type(monkeypatch, tmp_path):
    dummy_uuid = SimpleNamespace(hex="abcdef1234567890")
    monkeypatch.setattr(audio_files.uuid, "uuid4", lambda: dummy_uuid)

    faux_response = _FakeResponse(headers={"content-type": "text/plain"}, chunks=[b"x"])
    with pytest.raises(audio_files.AudioDownloadError):
        audio_files.download_audio_file(
            "https://example.com/music.mp3",
            str(tmp_path),
            downloader=lambda *_, **__: faux_response,
        )
    expected_path = tmp_path / "music_abcdef12.mp3"
    assert not expected_path.exists()
