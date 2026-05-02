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
