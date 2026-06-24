import pytest

from tldw_Server_API.app.core.Third_Party import BioRxiv as biorxiv


pytestmark = pytest.mark.unit


_BIORXIV_ERROR_CASES = (
    pytest.param(
        "fetch",
        lambda: biorxiv._raw_get("/details/biorxiv/10.1101/private/na"),
        2,
        "BioRxiv raw request failed.",
        "BioRxiv raw request timed out.",
        ("bio raw token", "/private/biorxiv-raw.key", "10.1101/private"),
        id="raw-get",
    ),
    pytest.param(
        "_get_json",
        lambda: biorxiv.search_biorxiv("private query", limit=1),
        2,
        "BioRxiv request failed.",
        "BioRxiv request timed out.",
        ("bio search token", "/private/biorxiv-search.key", "private query"),
        id="search",
    ),
    pytest.param(
        "_get_json",
        lambda: biorxiv.get_biorxiv_by_doi("10.1101/private-doi"),
        1,
        "BioRxiv DOI request failed.",
        "BioRxiv DOI request timed out.",
        ("bio doi token", "/private/biorxiv-doi.key", "10.1101/private-doi"),
        id="by-doi",
    ),
    pytest.param(
        "_get_json",
        lambda: biorxiv.search_biorxiv_pubs(q="private published", limit=1),
        2,
        "BioRxiv published metadata request failed.",
        "BioRxiv published metadata request timed out.",
        ("bio pubs token", "/private/biorxiv-pubs.key", "private published"),
        id="published-metadata",
    ),
    pytest.param(
        "_get_json",
        lambda: biorxiv.get_biorxiv_published_by_doi("10.1101/private-published"),
        1,
        "BioRxiv published DOI request failed.",
        "BioRxiv published DOI request timed out.",
        ("bio published doi token", "/private/biorxiv-published-doi.key", "10.1101/private-published"),
        id="published-by-doi",
    ),
    pytest.param(
        "_get_json",
        lambda: biorxiv.search_biorxiv_publisher("10.1101", limit=1),
        2,
        "BioRxiv publisher request failed.",
        "BioRxiv publisher request timed out.",
        ("bio publisher token", "/private/biorxiv-publisher.key", "10.1101"),
        id="publisher",
    ),
    pytest.param(
        "_get_json",
        lambda: biorxiv.search_biorxiv_pub(limit=1),
        2,
        "BioRxiv published article request failed.",
        "BioRxiv published article request timed out.",
        ("bio pub token", "/private/biorxiv-pub.key"),
        id="published-article",
    ),
    pytest.param(
        "_get_json",
        lambda: biorxiv.search_biorxiv_funder("biorxiv", "https://ror.org/private", limit=1),
        2,
        "BioRxiv funder request failed.",
        "BioRxiv funder request timed out.",
        ("bio funder token", "/private/biorxiv-funder.key", "https://ror.org/private"),
        id="funder",
    ),
    pytest.param(
        "_get_json",
        lambda: biorxiv.get_biorxiv_summary("m"),
        1,
        "BioRxiv summary request failed.",
        "BioRxiv summary request timed out.",
        ("bio summary token", "/private/biorxiv-summary.key"),
        id="summary",
    ),
    pytest.param(
        "_get_json",
        lambda: biorxiv.get_biorxiv_usage("m"),
        1,
        "BioRxiv usage request failed.",
        "BioRxiv usage request timed out.",
        ("bio usage token", "/private/biorxiv-usage.key"),
        id="usage",
    ),
)


@pytest.mark.parametrize(
    "patch_attr, call_provider, error_index, expected_error, _expected_timeout, sensitive_terms",
    _BIORXIV_ERROR_CASES,
)
def test_biorxiv_provider_paths_sanitize_fetch_failures(
    monkeypatch,
    patch_attr,
    call_provider,
    error_index,
    expected_error,
    _expected_timeout,
    sensitive_terms,
):
    def fail_fetch(*_args, **_kwargs):
        raise RuntimeError(f"{sensitive_terms[0]} at {sensitive_terms[1]}")

    monkeypatch.setattr(biorxiv, patch_attr, fail_fetch)

    result = call_provider()
    error = result[error_index]

    assert error == expected_error
    for term in sensitive_terms:
        assert term not in error


@pytest.mark.parametrize(
    "patch_attr, call_provider, error_index, _expected_error, expected_timeout, sensitive_terms",
    _BIORXIV_ERROR_CASES,
)
def test_biorxiv_provider_paths_preserve_timeout_classification(
    monkeypatch,
    patch_attr,
    call_provider,
    error_index,
    _expected_error,
    expected_timeout,
    sensitive_terms,
):
    def fail_fetch(*_args, **_kwargs):
        raise TimeoutError(f"timed out for {sensitive_terms[0]} at {sensitive_terms[1]}")

    monkeypatch.setattr(biorxiv, patch_attr, fail_fetch)

    result = call_provider()
    error = result[error_index]

    assert error == expected_timeout
    for term in sensitive_terms:
        assert term not in error


def test_search_biorxiv_filtered_query_continues_after_empty_filtered_batch(monkeypatch):
    cursors: list[int] = []

    def fake_get_json(url, params=None, timeout=15):
        del params, timeout
        cursor = int(url.rstrip("/").split("/")[-1])
        cursors.append(cursor)
        if cursor == 0:
            collection = [
                {
                    "doi": f"10.1101/nonmatch.{i}",
                    "title": f"Unrelated preprint {i}",
                    "authors": "No Match",
                    "abstract": "irrelevant",
                    "server": "biorxiv",
                    "version": 1,
                }
                for i in range(100)
            ]
            return {"messages": [{"count": "100", "total": "101"}], "collection": collection}
        if cursor == 100:
            return {
                "messages": [{"count": "1", "total": "101"}],
                "collection": [
                    {
                        "doi": "10.1101/target",
                        "title": "Target kinase discovery",
                        "authors": "Match Author",
                        "abstract": "relevant",
                        "server": "biorxiv",
                        "version": 1,
                    }
                ],
            }
        return {"messages": [{"count": "0", "total": "101"}], "collection": []}

    monkeypatch.setattr(biorxiv, "_get_json", fake_get_json)
    monkeypatch.setattr(biorxiv.time, "sleep", lambda _seconds: None)

    items, total, error = biorxiv.search_biorxiv("target kinase", limit=1)

    assert error is None
    assert [item["doi"] for item in items or []] == ["10.1101/target"]
    assert total == 1
    assert cursors == [0, 100]


def test_search_biorxiv_pubs_filtered_query_continues_after_empty_filtered_batch(monkeypatch):
    cursors: list[int] = []

    def fake_get_json(url, timeout=15, params=None):
        del timeout, params
        cursor = int(url.rstrip("/").split("/")[-1])
        cursors.append(cursor)
        if cursor == 0:
            collection = [
                {
                    "biorxiv_doi": f"10.1101/pub-nonmatch.{i}",
                    "preprint_title": f"Unrelated publication {i}",
                    "preprint_authors": "No Match",
                    "preprint_abstract": "irrelevant",
                }
                for i in range(100)
            ]
            return {"messages": [{"count": "100"}], "collection": collection}
        if cursor == 100:
            return {
                "messages": [{"count": "1"}],
                "collection": [
                    {
                        "biorxiv_doi": "10.1101/pub-target",
                        "preprint_title": "Target clinical publication",
                        "preprint_authors": "Match Author",
                        "preprint_abstract": "relevant",
                    }
                ],
            }
        return {"messages": [{"count": "0"}], "collection": []}

    monkeypatch.setattr(biorxiv, "_get_json", fake_get_json)
    monkeypatch.setattr(biorxiv.time, "sleep", lambda _seconds: None)

    items, total, error = biorxiv.search_biorxiv_pubs(q="target clinical", limit=1)

    assert error is None
    assert [item["biorxiv_doi"] for item in items or []] == ["10.1101/pub-target"]
    assert total == 1
    assert cursors == [0, 100]
