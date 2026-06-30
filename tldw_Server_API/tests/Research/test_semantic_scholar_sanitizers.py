import pytest

from tldw_Server_API.app.core.Third_Party import Semantic_Scholar as semantic_scholar


pytestmark = pytest.mark.unit


def test_search_papers_semantic_scholar_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("semantic scholar token at /private/s2.key")

    monkeypatch.setattr(semantic_scholar, "fetch_json", fail_fetch_json)

    data, error = semantic_scholar.search_papers_semantic_scholar("retrieval")

    assert data is None
    assert error == "Semantic Scholar search failed."
    assert "semantic scholar token" not in error
    assert "/private/s2.key" not in error


def test_get_paper_details_semantic_scholar_sanitizes_fetch_failures(monkeypatch):
    def fail_fetch_json(**_kwargs):
        raise RuntimeError("semantic scholar detail token at /private/s2-detail.key")

    monkeypatch.setattr(semantic_scholar, "fetch_json", fail_fetch_json)

    data, error = semantic_scholar.get_paper_details_semantic_scholar(
        "paper-id-from-/private/request"
    )

    assert data is None
    assert error == "Semantic Scholar paper details lookup failed."
    assert "semantic scholar detail token" not in error
    assert "/private/s2-detail.key" not in error
    assert "paper-id-from-/private/request" not in error
