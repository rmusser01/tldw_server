import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import paper_search


pytestmark = pytest.mark.unit


class LoggerStub:
    def __init__(self):
        self.errors: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def error(self, message, *args, **kwargs):
        self.errors.append((str(message), args, kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint_call", "provider", "helper_name", "expected_log"),
    [
        (
            lambda: paper_search.paper_search_arxiv_by_id("1706.03762"),
            paper_search.Arxiv,
            "get_arxiv_by_id",
            "Unexpected arXiv by-id error",
        ),
        (
            lambda: paper_search.paper_search_semantic_scholar_by_id("paper-123"),
            paper_search.Semantic_Scholar,
            "get_paper_details_semantic_scholar",
            "Unexpected Semantic Scholar by-id error",
        ),
        (
            lambda: paper_search.paper_search_pubmed_by_id("12345678"),
            paper_search.PubMed,
            "get_pubmed_by_id",
            "Unexpected PubMed by-id error",
        ),
        (
            lambda: paper_search.chemrxiv_item_by_id("chemrxiv-123"),
            paper_search.ChemRxiv,
            "get_item_by_id",
            "Unexpected ChemRxiv by-id error",
        ),
        (
            lambda: paper_search.earthrxiv_by_id("earthrxiv-123"),
            paper_search.EarthRxiv,
            "get_item_by_id",
            "Unexpected EarthArXiv by-id error",
        ),
        (
            lambda: paper_search.osf_by_id("osf-123"),
            paper_search.OSF,
            "get_preprint_by_id",
            "Unexpected OSF by-id error",
        ),
    ],
)
async def test_by_id_unexpected_errors_log_sanitized_fallback(
    monkeypatch,
    endpoint_call,
    provider,
    helper_name,
    expected_log,
):
    def fail_provider(*_args, **_kwargs):
        raise RuntimeError("paper backend exploded /private/paper.db")

    logger_stub = LoggerStub()
    monkeypatch.setattr(provider, helper_name, fail_provider)
    monkeypatch.setattr(paper_search, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await endpoint_call()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Unexpected provider error"

    log_text = repr(logger_stub.errors)
    assert expected_log in log_text
    assert "paper backend exploded" not in log_text
    assert "/private/paper.db" not in log_text
    assert "exc_info" not in log_text
