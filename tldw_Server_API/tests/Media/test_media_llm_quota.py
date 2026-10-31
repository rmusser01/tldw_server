"""Media processing with analysis refuses an over-quota user before any work (LLM usage accounting spec §4a)."""

from types import SimpleNamespace

import pytest

import tldw_Server_API.app.api.v1.endpoints.media as media_mod
from tldw_Server_API.app.api.v1.API_Deps.media_processing_deps import media_analysis_requested
from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work


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


_ANALYSIS = {"perform_analysis": "true", "api_name": "openai"}

_OTHER_HANDLERS = [
    pytest.param("/api/v1/media/process-audios", _ANALYSIS, [("files", ("a.mp3", b"fake", "audio/mpeg"))], id="audios"),
    pytest.param("/api/v1/media/process-videos", _ANALYSIS, [("files", ("v.mp4", b"fake", "video/mp4"))], id="videos"),
    pytest.param("/api/v1/media/process-documents", _ANALYSIS, [("files", ("d.txt", b"fake", "text/plain"))], id="documents"),
    pytest.param("/api/v1/media/process-pdfs", _ANALYSIS, [("files", ("d.pdf", b"%PDF-1.4 fake", "application/pdf"))], id="pdfs"),
    pytest.param("/api/v1/media/process-emails", _ANALYSIS, [("files", ("m.eml", b"fake", "message/rfc822"))], id="emails"),
    pytest.param("/api/v1/media/add", {**_ANALYSIS, "media_type": "document"}, [("files", ("d.txt", b"fake", "text/plain"))], id="add"),
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
def test_ebook_analysis_is_refused_before_processing(
    client_with_single_user, quota_service_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An over-quota user asking for analysis gets the 402, and the ebook is never processed."""
    client, _usage_logger = client_with_single_user
    processed: list[object] = []

    def _process_epub(*args: object, **kwargs: object) -> dict:
        processed.append(kwargs)
        return {"status": "Success", "content": "", "metadata": {}}

    monkeypatch.setattr(media_mod.books, "process_epub", _process_epub)
    refuse_llm_work(monkeypatch)
    response = client.post(
        "/api/v1/media/process-ebooks",
        data={"perform_analysis": "true", "api_name": "openai"},
        files=[("files", ("sample.epub", b"fake", "application/epub+zip"))],
    )
    assert_llm_quota_402(response)
    assert processed == []


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
@pytest.mark.parametrize(("path", "data", "files"), _OTHER_HANDLERS)
def test_other_media_handlers_refuse_analysis_for_an_over_quota_user(
    client_with_single_user, quota_service_stub, monkeypatch: pytest.MonkeyPatch, path: str, data: dict, files: list
) -> None:
    """Each remaining handler answers the 402 before touching its input."""
    client, _usage_logger = client_with_single_user
    refuse_llm_work(monkeypatch)
    response = client.post(path, data=data, files=files)
    assert_llm_quota_402(response)
