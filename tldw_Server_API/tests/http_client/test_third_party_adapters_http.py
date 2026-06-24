import os
from typing import Any

import httpx

import tldw_Server_API.app.core.http_client as http_client
from tldw_Server_API.app.core.Third_Party import HAL as hal
from tldw_Server_API.app.core.Third_Party import Crossref as crossref
from tldw_Server_API.app.core.Third_Party import OpenAlex as openalex
from tldw_Server_API.app.core.Third_Party import Elsevier_Scopus as scopus
from tldw_Server_API.app.core.Third_Party import EarthRxiv as earth
from tldw_Server_API.app.core.Third_Party import IEEE_Xplore as ieee
from tldw_Server_API.app.core.Third_Party import OSF as osf
from tldw_Server_API.app.core.Third_Party import BioRxiv as biorxiv
from tldw_Server_API.app.core.Third_Party import IACR as iacr
from tldw_Server_API.app.core.Third_Party import Figshare as figshare
from tldw_Server_API.app.core.Third_Party import Semantic_Scholar as semantic_scholar
from tldw_Server_API.app.core.Third_Party import Springer_Nature as springer


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_hal_raw_media_types(monkeypatch):
    # Allow HAL host in egress policy
    monkeypatch.setenv("EGRESS_ALLOWLIST", "api.archives-ouvertes.fr")

    def handler(request: httpx.Request) -> httpx.Response:
        # Validate target host
        assert request.url.host == "api.archives-ouvertes.fr"
        wt = request.url.params.get("wt") or "json"
        if wt == "xml" or wt == "xml-tei":
            return httpx.Response(200, headers={"content-type": "application/xml"}, content=b"<root/>", request=request)
        if wt in ("csv", "bibtex", "endnote"):
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"id,title\n1,a\n", request=request)
        return httpx.Response(200, headers={"content-type": "application/json"}, content=b"{}", request=request)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    monkeypatch.setattr(http_client, "create_client", fake_create_client)

    # XML
    content, media_type, err = hal.raw({"wt": "xml"})
    assert err is None
    assert media_type == "application/xml"
    assert content == b"<root/>"
    # CSV/text
    content, media_type, err = hal.raw({"wt": "csv"})
    assert err is None
    assert media_type == "text/plain"
    assert content.startswith(b"id,title")
    # JSON default
    content, media_type, err = hal.raw({})
    assert err is None
    assert media_type == "application/json"


def test_crossref_get_by_doi_404_and_success(monkeypatch):
    monkeypatch.setenv("EGRESS_ALLOWLIST", "api.crossref.org")
    def handler(request: httpx.Request) -> httpx.Response:
        # Works lookup path
        if request.url.path.startswith("/works/"):
            doi = request.url.path.split("/works/")[-1]
            if doi == "10.404/none":
                return httpx.Response(404, request=request)
            # Success
            body = {
                "message": {
                    "DOI": doi,
                    "title": ["T"],
                    "author": [{"given": "A", "family": "B"}],
                    "container-title": ["J"],
                    "issued": {"date-parts": [[2020, 1, 1]]},
                    "link": [{"content-type": "application/pdf", "URL": "http://x/pdf"}],
                    "URL": f"https://doi.org/{doi}",
                }
            }
            return httpx.Response(200, json=body, request=request)
        # Search endpoint (not used here)
        return httpx.Response(200, json={"message": {"items": [], "total-results": 0}}, request=request)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    monkeypatch.setattr(http_client, "create_client", fake_create_client)

    item, err = crossref.get_crossref_by_doi("10.404/none")
    assert item is None and err is None
    item, err = crossref.get_crossref_by_doi("10.123/ok")
    assert err is None and item is not None
    assert item["doi"] == "10.123/ok"


def test_openalex_get_by_doi_404_and_success(monkeypatch):
    monkeypatch.setenv("EGRESS_ALLOWLIST", "api.openalex.org")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/doi:"):
            doi = request.url.path.split("/works/doi:")[-1]
            if doi == "10.404/none":
                return httpx.Response(404, request=request)
            body = {
                "id": "W1",
                "title": "OpenAlex Title",
                "authorships": [{"author": {"display_name": "Auth"}}],
                "publication_date": "2021-01-01",
                "doi": doi,
                "open_access": {"oa_url": "http://x/pdf"},
                "primary_location": {"landing_page_url": "http://x"},
            }
            return httpx.Response(200, json=body, request=request)
        return httpx.Response(404, request=request)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    monkeypatch.setattr(http_client, "create_client", fake_create_client)

    item, err = openalex.get_openalex_by_doi("10.404/none")
    assert item is None and err is None
    item, err = openalex.get_openalex_by_doi("10.321/ok")
    assert err is None and item is not None
    assert item["doi"] == "10.321/ok"


def test_scopus_get_by_doi_404_and_success(monkeypatch):
    monkeypatch.setenv("EGRESS_ALLOWLIST", "api.elsevier.com")
    os.environ["ELSEVIER_API_KEY"] = "test_key"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.elsevier.com"
        if request.url.path == "/content/search/scopus":
            q = request.url.params.get("query", "")
            if q == "DOI(10.404/none)":
                return httpx.Response(404, request=request)
            if q == "DOI(10.123/ok)":
                body = {
                    "search-results": {
                        "opensearch:totalResults": "1",
                        "entry": [
                            {
                                "eid": "2-s2.0-123",
                                "dc:title": "Title",
                                "prism:doi": "10.123/ok",
                                "prism:publicationName": "J",
                                "prism:coverDate": "2020-01-01",
                            }
                        ],
                    }
                }
                return httpx.Response(200, json=body, request=request)
        return httpx.Response(500, request=request)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    monkeypatch.setattr(http_client, "create_client", fake_create_client)

    item, err = scopus.get_scopus_by_doi("10.404/none")
    assert item is None and err is None
    item, err = scopus.get_scopus_by_doi("10.123/ok")
    assert err is None and item is not None
    assert item["doi"] == "10.123/ok"


def test_eartharxiv_get_by_doi_404_and_success(monkeypatch):
    monkeypatch.setenv("EGRESS_ALLOWLIST", "api.osf.io")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.osf.io"
        if request.url.path == "/v2/preprints/":
            params = dict(request.url.params)
            if params.get("filter[provider]") == "eartharxiv" and params.get("filter[doi]") == "10.404/none":
                # First pass: return 404 to trigger fallback path
                return httpx.Response(404, request=request)
            if params.get("filter[provider]") == "eartharxiv" and params.get("q") == "10.404/none":
                # Fallback returns 200 but empty data
                return httpx.Response(200, json={"data": []}, request=request)
            if params.get("filter[provider]") == "eartharxiv" and params.get("filter[doi]") == "10.321/ok":
                body = {
                    "data": [
                        {
                            "id": "X1",
                            "attributes": {
                                "title": "T",
                                "description": "A",
                                "date_published": "2021-01-01",
                                "doi": "10.321/ok",
                            },
                        }
                    ]
                }
                return httpx.Response(200, json=body, request=request)
        return httpx.Response(500, request=request)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    monkeypatch.setattr(http_client, "create_client", fake_create_client)

    item, err = earth.get_item_by_doi("10.404/none")
    assert item is None and err is None
    item, err = earth.get_item_by_doi("10.321/ok")
    assert err is None and item is not None
    assert item["doi"] == "10.321/ok"


def test_ieee_get_by_doi_404_and_success(monkeypatch):
    monkeypatch.setenv("EGRESS_ALLOWLIST", "ieeexploreapi.ieee.org")
    os.environ["IEEE_API_KEY"] = "abc"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "ieeexploreapi.ieee.org"
        if request.url.path == "/api/v1/search/articles":
            q = request.url.params.get("querytext", "")
            if q == "doi:10.404/none":
                return httpx.Response(200, json={"articles": []}, request=request)
            if q == "doi:10.1111/ok":
                body = {"total_records": 1, "articles": [{"doi": "10.1111/ok", "title": "T"}]}
                return httpx.Response(200, json=body, request=request)
        return httpx.Response(500, request=request)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    monkeypatch.setattr(http_client, "create_client", fake_create_client)

    item, err = ieee.get_ieee_by_doi("10.404/none")
    assert item is None and err is None
    item, err = ieee.get_ieee_by_doi("10.1111/ok")
    assert err is None and item is not None
    assert item["doi"] == "10.1111/ok"


def test_osf_get_by_doi_404_and_success(monkeypatch):
    monkeypatch.setenv("EGRESS_ALLOWLIST", "api.osf.io")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.osf.io"
        if request.url.path == "/v2/preprints/":
            params = dict(request.url.params)
            # Exact filters
            if params.get("filter[doi]") == "10.404/none" or params.get("filter[article_doi]") == "10.404/none":
                return httpx.Response(200, json={"data": []}, request=request)
            if params.get("filter[doi]") == "10.222/ok" or params.get("filter[article_doi]") == "10.222/ok":
                body = {"data": [{"id": "P1", "attributes": {"title": "T", "doi": "10.222/ok"}, "links": {"html": "http://x"}}]}
                return httpx.Response(200, json=body, request=request)
            # Fallback free-text path
            if params.get("q") == "10.333/also-ok":
                body = {"data": [{"id": "P2", "attributes": {"title": "Q", "doi": "10.333/also-ok"}, "links": {"html": "http://y"}}]}
                return httpx.Response(200, json=body, request=request)
        return httpx.Response(404, request=request)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    monkeypatch.setattr(http_client, "create_client", fake_create_client)

    # Not found, then found via exact filter
    item, err = osf.get_preprint_by_doi("10.404/none")
    assert item is None and err is None
    item, err = osf.get_preprint_by_doi("10.222/ok")
    assert err is None and item is not None and item["doi"] == "10.222/ok"
    # Fallback via q
    item, err = osf.get_preprint_by_doi("10.333/also-ok")
    assert err is None and item is not None and item["doi"] == "10.333/also-ok"


def test_biorxiv_get_by_doi_404_and_success(monkeypatch):
    monkeypatch.setenv("EGRESS_ALLOWLIST", "api.biorxiv.org")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.biorxiv.org"
        if request.url.path.startswith("/details/biorxiv/") and request.url.path.endswith("/na"):
            doi = request.url.path.split("/details/biorxiv/")[-1].split("/na")[0]
            if doi == "10.404/none":
                return httpx.Response(200, json={"collection": []}, request=request)
            if doi == "10.987/ok":
                body = {"collection": [{"doi": doi, "title": "T", "server": "biorxiv", "version": 1}]}
                return httpx.Response(200, json=body, request=request)
        return httpx.Response(404, request=request)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    monkeypatch.setattr(http_client, "create_client", fake_create_client)

    item, err = biorxiv.get_biorxiv_by_doi("10.404/none")
    assert item is None and err is None
    item, err = biorxiv.get_biorxiv_by_doi("10.987/ok")
    assert err is None and item is not None and item["doi"] == "10.987/ok"


def test_iacr_fetch_conference_and_raw(monkeypatch):
    monkeypatch.setenv("EGRESS_ALLOWLIST", "www.iacr.org")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "www.iacr.org"
        assert request.url.path == "/cryptodb/data/api/conf.php"
        # Return simple JSON with expected structure
        body = {"conference": request.url.params.get("venue"), "year": request.url.params.get("year")}
        # Echo back params for verification
        if request.headers.get("accept", "").startswith("application/json") or True:
            return httpx.Response(200, json=body, request=request)
        return httpx.Response(200, content=b"{}", headers={"content-type": "application/json"}, request=request)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    monkeypatch.setattr(http_client, "create_client", fake_create_client)

    data, err = iacr.fetch_conference("crypto", 2017)
    assert err is None and data is not None
    content, media_type, err2 = iacr.fetch_conference_raw("crypto", 2017)
    assert err2 is None and media_type.startswith("application/json") and content


def test_figshare_search_and_oai_raw(monkeypatch):
    monkeypatch.setenv("EGRESS_ALLOWLIST", "api.figshare.com")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.figshare.com"
        if request.method == "POST" and request.url.path == "/v2/articles/search":
            resp = {
                "count": 1,
                "items": [
                    {
                        "id": 1,
                        "title": "T",
                        "authors": [{"full_name": "A"}],
                        "description": "D",
                        "published_date": "2021-01-01",
                        "files": [{"name": "x.pdf", "download_url": "http://x"}],
                    }
                ],
            }
            return httpx.Response(200, json=resp, request=request)
        if request.method == "GET" and request.url.path == "/v2/oai":
            return httpx.Response(200, headers={"content-type": "application/xml"}, content=b"<oai/>", request=request)
        return httpx.Response(404, request=request)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    monkeypatch.setattr(http_client, "create_client", fake_create_client)

    items, total, err = figshare.search_articles("q", page=1, page_size=1)
    assert err is None and items and total >= 1
    content, media_type, err2 = figshare.oai_raw({"verb": "Identify"})
    assert err2 is None and content == b"<oai/>" and media_type == "application/xml"


def test_semantic_scholar_search_surfaces_http_error_status(monkeypatch):
    monkeypatch.setenv("EGRESS_ALLOWLIST", "api.semanticscholar.org")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.semanticscholar.org"
        return httpx.Response(429, json={"message": "Too Many Requests"}, request=request)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    monkeypatch.setattr(http_client, "create_client", fake_create_client)

    data, err = semantic_scholar.search_papers_semantic_scholar("retrieval")

    assert data is None
    assert err is not None
    assert "HTTP Error: 429" in err


def test_ieee_search_surfaces_http_error_status(monkeypatch):
    monkeypatch.setenv("EGRESS_ALLOWLIST", "ieeexploreapi.ieee.org")
    monkeypatch.setenv("IEEE_API_KEY", "bad-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "ieeexploreapi.ieee.org"
        return httpx.Response(401, json={"message": "invalid key"}, request=request)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    monkeypatch.setattr(http_client, "create_client", fake_create_client)

    items, total, err = ieee.search_ieee("retrieval", 0, 10)

    assert items is None
    assert total == 0
    assert err is not None
    assert "HTTP Error: 401" in err


def test_springer_search_surfaces_http_error_status(monkeypatch):
    monkeypatch.setenv("EGRESS_ALLOWLIST", "api.springernature.com")
    monkeypatch.setenv("SPRINGER_NATURE_API_KEY", "bad-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.springernature.com"
        return httpx.Response(500, json={"error": "server error"}, request=request)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    monkeypatch.setattr(http_client, "create_client", fake_create_client)

    items, total, err = springer.search_springer("retrieval", 0, 10)

    assert items is None
    assert total == 0
    assert err is not None
    assert "HTTP Error: 500" in err


def test_scopus_lookup_surfaces_non_404_http_error_status(monkeypatch):
    monkeypatch.setenv("EGRESS_ALLOWLIST", "api.elsevier.com")
    monkeypatch.setenv("ELSEVIER_API_KEY", "bad-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.elsevier.com"
        return httpx.Response(500, json={"service-error": "unavailable"}, request=request)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    monkeypatch.setattr(http_client, "create_client", fake_create_client)

    item, err = scopus.get_scopus_by_doi("10.500/error")

    assert item is None
    assert err is not None
    assert "HTTP Error: 500" in err
