from __future__ import annotations

import contextlib

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import research
from tldw_Server_API.app.api.v1.schemas.research_schemas import (
    ArxivSearchRequestForm,
    SemanticScholarSearchRequestForm,
)
from tldw_Server_API.app.api.v1.schemas.websearch_schemas import WebSearchRequest


class _LoggerStub:
    def __init__(self) -> None:
        self.debug_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.error_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.info_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.warning_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def debug(self, *args, **kwargs) -> None:
        self.debug_calls.append((args, kwargs))

    def error(self, *args, **kwargs) -> None:
        self.error_calls.append((args, kwargs))

    def info(self, *args, **kwargs) -> None:
        self.info_calls.append((args, kwargs))

    def warning(self, *args, **kwargs) -> None:
        self.warning_calls.append((args, kwargs))


_SEARCH_SENSITIVE_MARKERS = (
    "arxiv fetch exploded",
    "search backend exploded",
    "/private/research/cache.xml",
    "/private/research/search-cache.db",
)


def _assert_sanitized_error_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.error_calls
    assert any(args and args[0] == expected_message for args, _kwargs in logger_stub.error_calls)
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)

    rendered = repr(logger_stub.error_calls)
    for marker in _SEARCH_SENSITIVE_MARKERS:
        assert marker not in rendered


@pytest.mark.unit
def test_process_and_ingest_arxiv_paper_uses_media_repository_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDb:
        def __init__(self) -> None:
            self.closed = False

        def close_connection(self) -> None:
            self.closed = True

    class _FakeRepo:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_media_with_keywords(self, **kwargs):
            self.calls.append(kwargs)
            return 51, "arxiv-uuid", "stored"

    fake_db = _FakeDb()
    fake_repo = _FakeRepo()

    monkeypatch.setattr(research, "fetch_arxiv_xml", lambda paper_id: f"<xml>{paper_id}</xml>")
    monkeypatch.setattr(
        research,
        "convert_xml_to_markdown",
        lambda xml: ("# Paper\n\nBody", "Paper Title", ["Ada Lovelace", "Alan Turing"], ["cs.AI", "cs.IR"]),
    )
    managed_calls = []

    @contextlib.contextmanager
    def _fake_managed_media_database(client_id, *, initialize=True, **kwargs):
        managed_calls.append(
            {
                "client_id": client_id,
                "initialize": initialize,
                "kwargs": kwargs,
            }
        )
        try:
            yield fake_db
        finally:
            fake_db.close_connection()

    monkeypatch.setattr(
        research,
        "managed_media_database",
        _fake_managed_media_database,
        raising=False,
    )
    monkeypatch.setattr(research, "get_media_repository", lambda db: fake_repo, raising=False)

    message = research.process_and_ingest_arxiv_paper("1234.5678", "ml,rag")

    assert message == "arXiv paper 'Paper Title' ingested successfully."
    assert fake_db.closed is True
    assert managed_calls == [
        {
            "client_id": "research_ingest",
            "initialize": False,
            "kwargs": {},
        }
    ]
    assert len(fake_repo.calls) == 1
    payload = dict(fake_repo.calls[0])
    ingestion_date = payload.pop("ingestion_date")
    assert payload == {
        "url": "https://arxiv.org/abs/1234.5678",
        "title": "Paper Title",
        "media_type": "document",
        "content": "# Paper\n\nBody",
        "keywords": ["arxiv", "cs.AI", "cs.IR", "ml", "rag"],
        "prompt": "No prompt for arXiv papers",
        "analysis_content": "arXiv paper ingested from XML",
        "transcription_model": "None",
        "author": "Ada Lovelace, Alan Turing",
    }
    assert isinstance(ingestion_date, str)


@pytest.mark.unit
def test_process_and_ingest_arxiv_paper_sanitizes_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    def _fail_fetch(_paper_id):
        raise RuntimeError("arxiv fetch exploded at /private/research/cache.xml")

    monkeypatch.setattr(research, "logger", logger_stub, raising=True)
    monkeypatch.setattr(research, "fetch_arxiv_xml", _fail_fetch)

    message = research.process_and_ingest_arxiv_paper("1234.5678", "ml,rag")

    assert message == "Error processing arXiv paper"
    assert "arxiv fetch exploded" not in message
    assert "/private/research/cache.xml" not in message
    _assert_sanitized_error_log(logger_stub, "Error processing arXiv paper")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_arxiv_search_endpoint_sanitizes_unexpected_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(research, "logger", logger_stub, raising=True)

    def _fail_search(*_args, **_kwargs):
        raise RuntimeError("search backend exploded /private/research/search-cache.db")

    monkeypatch.setattr(research, "search_arxiv_custom_api", _fail_search)

    with pytest.raises(HTTPException) as exc_info:
        await research.arxiv_search_endpoint(
            search_params=ArxivSearchRequestForm(query="rag", page=1, results_per_page=10),
            Token=None,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "An unexpected error occurred while searching arXiv"
    _assert_sanitized_error_log(
        logger_stub,
        "Unexpected error during arXiv search execution",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_semantic_scholar_search_endpoint_sanitizes_unexpected_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(research, "logger", logger_stub, raising=True)

    def _fail_search(*_args, **_kwargs):
        raise RuntimeError("search backend exploded /private/research/search-cache.db")

    monkeypatch.setattr(research, "search_papers_semantic_scholar", _fail_search)

    with pytest.raises(HTTPException) as exc_info:
        await research.semantic_scholar_search_endpoint(
            search_params=SemanticScholarSearchRequestForm(
                query="rag",
                fields_of_study=None,
                publication_types=None,
                year_range=None,
                venue=None,
                min_citations=None,
                page=1,
                results_per_page=10,
            ),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "An unexpected error occurred while searching Semantic Scholar"
    _assert_sanitized_error_log(
        logger_stub,
        "Unexpected error during Semantic Scholar search execution",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websearch_endpoint_sanitizes_unexpected_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(research, "logger", logger_stub, raising=True)

    def _fail_generate_and_search(*_args, **_kwargs):
        raise RuntimeError("search backend exploded /private/research/search-cache.db")

    monkeypatch.setattr(research, "generate_and_search", _fail_generate_and_search)

    try:
        with pytest.raises(HTTPException) as exc_info:
            await research.websearch_endpoint(
                payload=WebSearchRequest(query="rag", engine="google"),
                request=None,
                current_user=object(),
                db=object(),
            )
    finally:
        research.shutdown_websearch_executor(wait=True, cancel_futures=True)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Websearch failed"
    _assert_sanitized_error_log(logger_stub, "websearch endpoint failed")
