"""Media processing with analysis refuses an over-quota user before any work (LLM usage accounting spec §4a)."""

import io
import wave
from types import SimpleNamespace

import pymupdf
import pytest
from ebooklib import epub

import tldw_Server_API.app.api.v1.endpoints.media as media_mod
from tldw_Server_API.app.api.v1.API_Deps.media_processing_deps import media_analysis_requested
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import Video_DL_Ingestion_Lib
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib
from tldw_Server_API.tests.helpers.llm_quota import (
    assert_llm_quota_402,
    refuse_llm_work,
    users_own_key,
)
from tldw_Server_API.tests.helpers.offline_network import block_live_network


class _StubQuotaService:
    async def check_quota(self, user_id, size_bytes, raise_on_exceed=False):
        """Always allow in tests."""
        return True, {
            "current_usage_mb": 0,
            "new_size_mb": float(size_bytes) / (1024 * 1024),
            "quota_mb": 999999,
            "available_mb": 999999,
        }


@pytest.fixture()
def quota_service_stub(monkeypatch):
    """Patch the storage quota service so no DB is touched (as tests/Media/test_media_usage_events.py does)."""
    import tldw_Server_API.app.services.storage_quota_service as quota_mod

    monkeypatch.setattr(quota_mod, "get_storage_quota_service", lambda: _StubQuotaService())
    yield


@pytest.fixture()
def transcribed(monkeypatch):
    """Speech recognition returns a fixed transcript (as the media ingest job tests stub it, and the video library's
    as ``test_video_summary_service_prompt.py`` does), so no model and no ffmpeg run."""
    segments = [{"start_seconds": 0, "end_seconds": 1, "Text": "Send the report on Friday."}]
    monkeypatch.setattr(Audio_Files, "speech_to_text", lambda **_kwargs: segments)
    monkeypatch.setattr(Audio_Files, "check_transcription_model_status", lambda _: {"available": True, "usable": True})
    monkeypatch.setattr(Video_DL_Ingestion_Lib, "perform_transcription", lambda **_kwargs: (None, segments))


def _wav() -> bytes:
    """A tenth of a second of silence."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 800)
    return out.getvalue()


def _pdf() -> bytes:
    """A one-page PDF with text."""
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "Project Cedar opens on 24 January 2027.")
        return document.tobytes()


def _epub(tmp_path) -> bytes:
    """A one-chapter EPUB."""
    book = epub.EpubBook()
    book.set_identifier("own-key-test")
    book.set_title("Cedar")
    book.set_language("en")
    chapter = epub.EpubHtml(title="Cedar", file_name="cedar.xhtml", lang="en")
    chapter.content = "<h1>Cedar</h1><p>Project Cedar opens on 24 January 2027.</p>"
    book.add_item(chapter)
    book.spine = [chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(tmp_path / "cedar.epub", book)
    return (tmp_path / "cedar.epub").read_bytes()


_ANALYSIS = {"perform_analysis": "true", "api_name": "openai"}

# The last value is how many analysis calls an admitted request makes.
_OTHER_HANDLERS = [
    pytest.param("/api/v1/media/process-audios", _ANALYSIS, [("files", ("a.wav", _wav(), "audio/wav"))], 1, id="audios"),
    pytest.param("/api/v1/media/process-videos", _ANALYSIS, [("files", ("v.mp4", b"fake", "video/mp4"))], 1, id="videos"),
    pytest.param("/api/v1/media/process-documents", _ANALYSIS, [("files", ("d.txt", b"fake", "text/plain"))], 1, id="documents"),
    pytest.param(
        "/api/v1/media/process-pdfs",
        {**_ANALYSIS, "pdf_parsing_engine": "pymupdf"},
        [("files", ("d.pdf", _pdf(), "application/pdf"))],
        1,
        id="pdfs",
    ),
    pytest.param("/api/v1/media/process-emails", _ANALYSIS, [("files", ("m.eml", b"fake", "message/rfc822"))], 1, id="emails"),
    pytest.param("/api/v1/media/add", {**_ANALYSIS, "media_type": "document"}, [("files", ("d.txt", b"fake", "text/plain"))], 1, id="add"),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("perform_analysis", "api_name", "expected"),
    [(True, "openai", True), (True, None, False), (True, "", False), (True, "None", False), (False, "openai", False)],
)
def test_analysis_is_requested_only_with_a_provider(perform_analysis: bool, api_name: str | None, expected: bool) -> None:
    """The same condition the processing libraries use to decide whether to call the LLM."""
    form = SimpleNamespace(perform_analysis=perform_analysis, api_name=api_name)
    assert media_analysis_requested(form) is expected


@pytest.mark.integration
@pytest.mark.parametrize("own_key", [True, False], ids=["own-key", "server-key"])
def test_ebook_analysis_names_its_provider_before_processing(
    client_with_single_user, quota_service_stub, monkeypatch: pytest.MonkeyPatch, tmp_path, own_key: bool
) -> None:
    """Past a limit, an ebook whose analysis runs on the user's own key is processed and analysed on that key; on the
    server's key the user gets the 402 and the ebook is never processed (TASK-15426)."""
    client, _usage_logger = client_with_single_user
    processed: list[object] = []
    summarized: list[dict] = []
    process_epub = media_mod.books.process_epub

    def _process_epub(*args: object, **kwargs: object) -> dict:
        processed.append(kwargs)
        return process_epub(*args, **kwargs)

    monkeypatch.setattr(media_mod.books, "process_epub", _process_epub)
    monkeypatch.setattr(
        Summarization_General_Lib, "_summarize_via_adapter", lambda **kwargs: summarized.append(kwargs) or "Summary."
    )
    refuse_llm_work(monkeypatch)
    asked = users_own_key(monkeypatch, "openai" if own_key else "anthropic")
    with block_live_network() as attempts:
        response = client.post(
            "/api/v1/media/process-ebooks",
            data={"perform_analysis": "true", "api_name": "openai"},
            files=[("files", ("cedar.epub", _epub(tmp_path), "application/epub+zip"))],
        )
    assert attempts == []
    assert asked == ["openai"]
    if own_key:
        assert response.status_code != 402, response.text
        assert len(processed) == 1
        assert [getattr(k["provider_credentials"], "api_key", None) for k in summarized] == ["sk-users-own-key"]
    else:
        assert_llm_quota_402(response)
        assert (processed, summarized) == ([], [])


@pytest.mark.integration
def test_ebook_processing_without_analysis_is_not_refused(
    client_with_single_user, quota_service_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a provider there is no LLM work, so an over-quota user is not refused."""
    client, _usage_logger = client_with_single_user
    monkeypatch.setattr(
        media_mod.books,
        "process_epub",
        lambda *args, **kwargs: {"status": "Success", "content": "", "metadata": {}},
    )
    refuse_llm_work(monkeypatch)
    response = client.post(
        "/api/v1/media/process-ebooks",
        files=[("files", ("sample.epub", b"fake", "application/epub+zip"))],
    )
    assert response.status_code != 402, response.text


@pytest.mark.integration
@pytest.mark.parametrize("own_key", [True, False], ids=["own-key", "server-key"])
@pytest.mark.parametrize(("path", "data", "files", "analyses"), _OTHER_HANDLERS)
def test_media_handlers_name_the_analysis_provider_to_the_llm_quota(
    client_with_single_user,
    quota_service_stub,
    transcribed,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    data: dict,
    files: list,
    analyses: int,
    own_key: bool,
) -> None:
    """Past a limit, analysis whose provider runs on the user's own key is admitted and runs on that key; on the
    server's key the handler answers the 402 before touching its input (TASK-15426). Nothing leaves the machine."""
    client, _usage_logger = client_with_single_user
    summarized: list[dict] = []
    monkeypatch.setattr(
        Summarization_General_Lib, "_summarize_via_adapter", lambda **kwargs: summarized.append(kwargs) or "Summary."
    )
    refuse_llm_work(monkeypatch)
    asked = users_own_key(monkeypatch, "openai" if own_key else "anthropic")

    with block_live_network() as attempts:
        response = client.post(path, data=data, files=files)

    assert attempts == []
    assert asked == ["openai"]
    if own_key:
        assert response.status_code != 402, response.text
        assert [getattr(k["provider_credentials"], "api_key", None) for k in summarized] == ["sk-users-own-key"] * analyses
    else:
        assert_llm_quota_402(response)
        assert summarized == []


@pytest.mark.integration
def test_another_llm_stage_keeps_the_refusal_for_an_own_key_user(
    client_with_single_user, quota_service_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request that also asks for LLM chunk boundaries makes a call the boundary cannot place on the user's key,
    so an over-limit user gets the 402, though they stored a key for the analysis (TASK-15426)."""
    client, _usage_logger = client_with_single_user
    summarized: list[dict] = []
    monkeypatch.setattr(
        Summarization_General_Lib, "_summarize_via_adapter", lambda **kwargs: summarized.append(kwargs) or "Summary."
    )
    refuse_llm_work(monkeypatch)
    asked = users_own_key(monkeypatch, "openai")
    with block_live_network() as attempts:
        response = client.post(
            "/api/v1/media/process-documents",
            data={**_ANALYSIS, "auto_chunking_use_llm": "true"},
            files=[("files", ("d.txt", b"fake", "text/plain"))],
        )
    assert attempts == []
    assert_llm_quota_402(response)
    assert (asked, summarized) == (["openai"], [])


@pytest.mark.integration
def test_another_llm_stage_still_analyses_on_an_under_limit_users_own_key(
    client_with_single_user, quota_service_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under the limit, a request that also turns on another LLM stage still runs its analysis on the user's own key
    for the analysis provider; only the other stage keeps the server's key (TASK-15426). Chunking is off here, so
    the LLM chunk boundaries the request asks for make no call."""
    client, _usage_logger = client_with_single_user
    summarized: list[dict] = []
    monkeypatch.setattr(
        Summarization_General_Lib, "_summarize_via_adapter", lambda **kwargs: summarized.append(kwargs) or "Summary."
    )
    asked = users_own_key(monkeypatch, "openai")
    with block_live_network() as attempts:
        response = client.post(
            "/api/v1/media/process-documents",
            data={**_ANALYSIS, "auto_chunking_use_llm": "true", "perform_chunking": "false"},
            files=[("files", ("d.txt", b"Project Cedar opens on 24 January 2027.", "text/plain"))],
        )
    assert attempts == []
    assert response.status_code == 200, response.text
    assert [getattr(k["provider_credentials"], "api_key", None) for k in summarized] == ["sk-users-own-key"]
    assert asked == ["openai"]
