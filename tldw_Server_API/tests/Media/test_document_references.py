# Tests for Document References Endpoint
#
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints.media import document_references as refs_mod
from tldw_Server_API.app.api.v1.schemas.document_references import ReferenceEntry
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


async def _allow_non_authz_dep() -> None:
    return None


app = FastAPI()
app.include_router(refs_mod.router, prefix="/api/v1/media")


def _install_route_dependency_overrides() -> None:
    for route in app.routes:
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        for dep in getattr(dependant, "dependencies", []):
            call = getattr(dep, "call", None)
            if getattr(call, "_tldw_rate_limit_resource", None) is not None:
                app.dependency_overrides[call] = _allow_non_authz_dep


@pytest.fixture(autouse=True)
def reset_app_dependency_overrides():
    app.dependency_overrides.clear()
    _install_route_dependency_overrides()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_user():
    user = MagicMock()
    user.id = 1
    user.username = "testuser"
    return user


@pytest.fixture
def mock_db(tmp_path):
    db = MagicMock()
    db.db_path_str = str(tmp_path / "test_media.db")
    return db


class _LoggerStub:
    def __init__(self) -> None:
        self.error_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.debug_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.warning_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def error(self, *args: Any, **kwargs: Any) -> None:
        self.error_calls.append((args, kwargs))

    def debug(self, *_args: Any, **_kwargs: Any) -> None:
        self.debug_calls.append((_args, _kwargs))

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        self.warning_calls.append((_args, _kwargs))


class _FailingReferenceCacheDb:
    def transaction(self):
        raise RuntimeError("references cache failed at /private/references-cache.db")

    def execute_query(self, *_args: Any, **_kwargs: Any):
        raise RuntimeError("references cache failed at /private/references-cache.db")


class _ReferenceCacheRepositoryStub:
    def __init__(
        self,
        *,
        cached_result: tuple[list[str], int] | None = None,
        load_exc: Exception | None = None,
        save_exc: Exception | None = None,
    ) -> None:
        self.cached_result = cached_result
        self.load_exc = load_exc
        self.save_exc = save_exc
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def get_parsed_references_cache(self, *args: Any, **kwargs: Any) -> tuple[list[str], int] | None:
        self.calls.append(("get_parsed_references_cache", args, kwargs))
        if self.load_exc is not None:
            raise self.load_exc
        return self.cached_result

    def upsert_parsed_references_cache(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("upsert_parsed_references_cache", args, kwargs))
        if self.save_exc is not None:
            raise self.save_exc


def _patch_reference_repository(monkeypatch: pytest.MonkeyPatch, repo: _ReferenceCacheRepositoryStub) -> None:
    class RepositoryFactory:
        @staticmethod
        def from_media_db(db: Any) -> _ReferenceCacheRepositoryStub:
            repo.calls.append(("from_media_db", (db,), {}))
            return repo

    monkeypatch.setattr(refs_mod, "DocumentWorkspaceRepository", RepositoryFactory, raising=False)


def _assert_log_sanitized(
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]],
    expected_message: str,
) -> None:
    assert calls
    assert [args[0] for args, _kwargs in calls if args] == [expected_message]
    assert all(not kwargs.get("exc_info") for _args, kwargs in calls)
    rendered_calls = repr(calls)
    assert "references cache failed" not in rendered_calls
    assert "/private/references-cache.db" not in rendered_calls


def _assert_log_contains_sanitized(
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]],
    expected_message: str,
) -> None:
    assert calls
    messages = [args[0] for args, _kwargs in calls if args]
    assert expected_message in messages
    assert all(not kwargs.get("exc_info") for _args, kwargs in calls)
    rendered_calls = repr(calls)
    assert "references cache failed" not in rendered_calls
    assert "/private/references-cache.db" not in rendered_calls


def _assert_sanitized_debug_messages(
    logger_mock: MagicMock,
    expected_messages: list[str],
    forbidden_markers: list[str],
) -> None:
    messages = [args[0] for args, _kwargs in logger_mock.debug.call_args_list if args]
    assert messages == expected_messages
    rendered_calls = repr(logger_mock.debug.call_args_list)
    for marker in forbidden_markers:
        assert marker not in rendered_calls


@pytest.mark.asyncio
async def test_references_endpoint_extracts_basic(mock_user, mock_db):
    content = (
        "Intro text\\n"
        "References\\n"
        "[1] Smith, J. (2020). Example Paper. https://doi.org/10.1234/abcd\\n"
        "[2] Doe, A. (2019). Another Paper. arXiv:2101.12345\\n"
    )
    mock_db.get_media_by_id = MagicMock(return_value={"id": 1, "content": content})

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with patch.object(refs_mod, "get_cached_response", return_value=None):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/references?enrich=false")

    assert response.status_code == 200
    data = response.json()
    assert data["has_references"] is True
    assert len(data["references"]) == 2
    assert data["references"][0]["doi"] == "10.1234/abcd"
    assert data["references"][1]["arxiv_id"] == "2101.12345"
    assert data["total_detected"] == 2
    assert data["truncated"] is False

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_references_endpoint_applies_pagination_window(mock_user, mock_db):
    content = (
        "Intro text\\n"
        "References\\n"
        "[1] Smith, J. (2020). Example 1.\\n"
        "[2] Smith, J. (2020). Example 2.\\n"
        "[3] Smith, J. (2020). Example 3.\\n"
        "[4] Smith, J. (2020). Example 4.\\n"
        "[5] Smith, J. (2020). Example 5.\\n"
        "[6] Smith, J. (2020). Example 6.\\n"
    )
    mock_db.get_media_by_id = MagicMock(return_value={"id": 1, "content": content})

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with patch.object(refs_mod, "get_cached_response", return_value=None):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/references?enrich=false&offset=2&limit=2")

    assert response.status_code == 200
    data = response.json()
    assert data["returned_count"] == 2
    assert data["total_available"] == 6
    assert data["has_more"] is True
    assert data["next_offset"] == 4
    assert data["pagination"] == {
        "mode": "offset",
        "limit": 2,
        "offset": 2,
        "total": 6,
        "has_more": True,
        "next_offset": 4,
    }
    assert "Example 3" in data["references"][0]["raw_text"]
    assert "Example 4" in data["references"][1]["raw_text"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_references_endpoint_applies_search_before_pagination(mock_user, mock_db):
    content = (
        "Intro text\\n"
        "References\\n"
        "[1] Smith, J. (2020). Example 1.\\n"
        "[2] Smith, J. (2020). Example 2.\\n"
        "[3] Smith, J. (2020). Example 3.\\n"
        "[4] Smith, J. (2020). Example 4.\\n"
        "[5] Smith, J. (2020). Example 5.\\n"
        "[6] Smith, J. (2020). Example 6.\\n"
    )
    mock_db.get_media_by_id = MagicMock(return_value={"id": 1, "content": content})

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with patch.object(refs_mod, "get_cached_response", return_value=None):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/references?enrich=false&offset=0&limit=2&search=Example%206")

    assert response.status_code == 200
    data = response.json()
    assert data["returned_count"] == 1
    assert data["total_available"] == 1
    assert data["has_more"] is False
    assert data["next_offset"] is None
    assert data["pagination"] == {
        "mode": "offset",
        "limit": 2,
        "offset": 0,
        "total": 1,
        "has_more": False,
        "next_offset": None,
    }
    assert "Example 6" in data["references"][0]["raw_text"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_references_endpoint_parse_cap_limits_available_refs(mock_user, mock_db):
    content = (
        "Intro text\\n"
        "References\\n"
        "[1] Smith, J. (2020). Example 1.\\n"
        "[2] Smith, J. (2020). Example 2.\\n"
        "[3] Smith, J. (2020). Example 3.\\n"
        "[4] Smith, J. (2020). Example 4.\\n"
        "[5] Smith, J. (2020). Example 5.\\n"
        "[6] Smith, J. (2020). Example 6.\\n"
    )
    mock_db.get_media_by_id = MagicMock(return_value={"id": 1, "content": content})

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with patch.object(refs_mod, "get_cached_response", return_value=None):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/references?enrich=false&parse_cap=3&limit=50")

    assert response.status_code == 200
    data = response.json()
    assert data["total_detected"] == 6
    assert data["total_available"] == 3
    assert data["truncated"] is True
    assert len(data["references"]) == 3
    assert data["has_more"] is False

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_references_endpoint_uses_parsed_reference_db_cache(mock_user, mock_db, monkeypatch):
    content = "Intro text\\n" "References\\n" "[1] This content should not be parsed when cache is present.\\n"
    mock_db.get_media_by_id = MagicMock(return_value={"id": 1, "content": content})
    repo = _ReferenceCacheRepositoryStub(
        cached_result=(
            [
                "[1] Cached Author. 2020. Cached Example One.",
                "[2] Cached Author. 2021. Cached Example Two.",
            ],
            2,
        )
    )
    _patch_reference_repository(monkeypatch, repo)

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with (
        patch.object(refs_mod, "get_cached_response", return_value=None),
        patch.object(
            refs_mod,
            "_split_references_with_meta",
            side_effect=AssertionError("Parser should not run on cache hit"),
        ),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/references?enrich=false")

    assert response.status_code == 200
    data = response.json()
    assert data["total_detected"] == 2
    assert data["total_available"] == 2
    assert len(data["references"]) == 2
    assert "Cached Example One" in data["references"][0]["raw_text"]
    assert [call[0] for call in repo.calls] == ["from_media_db", "get_parsed_references_cache"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_references_endpoint_cache_hit(mock_user, mock_db):
    cached_payload = {
        "media_id": 1,
        "has_references": True,
        "references": [{"raw_text": "Cached ref", "title": "Cached Title"}],
        "enrichment_source": None,
        "total_detected": 1,
        "truncated": False,
    }
    mock_db.get_media_by_id = MagicMock(side_effect=AssertionError("DB should not be called"))

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with patch.object(refs_mod, "get_cached_response", return_value=("etag", cached_payload)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/references")

    assert response.status_code == 200
    data = response.json()
    assert data["references"][0]["title"] == "Cached Title"
    assert data["pagination"] == {
        "mode": "offset",
        "limit": 1,
        "offset": 0,
        "total": 1,
        "has_more": False,
        "next_offset": None,
    }

    app.dependency_overrides.clear()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_references_endpoint_sanitizes_db_fetch_error_log(
    mock_user,
    mock_db,
    monkeypatch,
):
    mock_db.get_media_by_id = MagicMock(side_effect=RuntimeError("reference db failed at /private/references.db"))
    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db
    logger_stub = _LoggerStub()
    monkeypatch.setattr(refs_mod, "logger", logger_stub, raising=True)

    try:
        with patch.object(refs_mod, "get_cached_response", return_value=None):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/v1/media/1/references")

        assert response.status_code == 500
        assert response.json()["detail"] == "Database error while fetching media item"
        assert [args[0] for args, _kwargs in logger_stub.error_calls if args] == ["Database error fetching media item"]
        assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)
        rendered_calls = repr(logger_stub.error_calls)
        assert "reference db failed" not in rendered_calls
        assert "/private/references.db" not in rendered_calls
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_references_endpoint_sanitizes_parsed_cache_load_error(
    mock_user,
    mock_db,
    monkeypatch,
):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(refs_mod, "logger", logger_stub, raising=True)
    repo = _ReferenceCacheRepositoryStub(
        load_exc=RuntimeError("references cache failed at /private/references-cache.db")
    )
    _patch_reference_repository(monkeypatch, repo)
    mock_db.get_media_by_id = MagicMock(
        return_value={
            "id": 1,
            "content": "Intro text\nReferences\n[1] Parsed Author. 2020. Parsed Example.\n",
        }
    )

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db
    try:
        with (
            patch.object(refs_mod, "get_cached_response", return_value=None),
            patch.object(
                refs_mod,
                "_split_references_with_meta",
                return_value=(["[1] Parsed Author. 2020. Parsed Example."], 1, False),
            ),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/v1/media/1/references?enrich=false")

        assert response.status_code == 200
        _assert_log_contains_sanitized(
            logger_stub.debug_calls,
            "Failed loading parsed references cache",
        )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_references_endpoint_sanitizes_parsed_cache_save_error(
    mock_user,
    mock_db,
    monkeypatch,
):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(refs_mod, "logger", logger_stub, raising=True)
    repo = _ReferenceCacheRepositoryStub(
        save_exc=RuntimeError("references cache failed at /private/references-cache.db")
    )
    _patch_reference_repository(monkeypatch, repo)
    mock_db.get_media_by_id = MagicMock(
        return_value={
            "id": 1,
            "content": "Intro text\nReferences\n[1] Parsed Author. 2020. Parsed Example.\n",
        }
    )

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db
    try:
        with (
            patch.object(refs_mod, "get_cached_response", return_value=None),
            patch.object(
                refs_mod,
                "_split_references_with_meta",
                return_value=(["[1] Parsed Author. 2020. Parsed Example."], 1, False),
            ),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/api/v1/media/1/references?enrich=false")

        assert response.status_code == 200
        _assert_log_contains_sanitized(
            logger_stub.debug_calls,
            "Failed saving parsed references cache",
        )
    finally:
        app.dependency_overrides.clear()


def test_apply_crossref_data_sets_fields():
    ref = ReferenceEntry(raw_text="raw")
    item = {
        "title": "Crossref Title",
        "authors": "A. Author",
        "journal": "Journal of Tests",
        "pub_date": "2022-01-01",
        "doi": "10.5555/xyz",
        "url": "https://doi.org/10.5555/xyz",
        "pdf_url": "https://example.com/paper.pdf",
    }
    updated = refs_mod._apply_crossref_data(ref, item)
    assert updated.title == "Crossref Title"
    assert updated.authors == "A. Author"
    assert updated.venue == "Journal of Tests"
    assert updated.year == 2022
    assert updated.doi == "10.5555/xyz"
    assert updated.open_access_pdf == "https://example.com/paper.pdf"


def test_apply_arxiv_data_sets_fields():
    ref = ReferenceEntry(raw_text="raw")
    item = {
        "id": "2101.12345",
        "title": "arXiv Title",
        "authors": "A. Author",
        "published_date": "2021-02-03",
        "pdf_url": "https://arxiv.org/pdf/2101.12345.pdf",
    }
    updated = refs_mod._apply_arxiv_data(ref, item)
    assert updated.title == "arXiv Title"
    assert updated.authors == "A. Author"
    assert updated.year == 2021
    assert updated.arxiv_id == "2101.12345"
    assert updated.url == "https://arxiv.org/abs/2101.12345"
    assert updated.open_access_pdf == "https://arxiv.org/pdf/2101.12345.pdf"


def test_build_references_cache_key_includes_scope(mock_db):
    key = refs_mod._build_references_cache_key(
        7,
        enrich=True,
        user_id="42",
        db_scope=mock_db.db_path_str,
        offset=0,
        limit=50,
        parse_cap=None,
        search_query=None,
    )
    assert "user:42" in key
    assert f"db:{mock_db.db_path_str}" in key
    assert "enrich" in key


def test_build_references_cache_key_includes_reference_index(mock_db):
    key = refs_mod._build_references_cache_key(
        7,
        enrich=True,
        user_id="42",
        db_scope=mock_db.db_path_str,
        offset=10,
        limit=25,
        parse_cap=200,
        search_query=None,
        reference_index=3,
    )
    assert ":idx:3" in key
    assert ":offset:10:limit:25:cap:200" in key


def test_find_reference_section_rejects_inline_references_mentions():
    content = (
        "This section references prior work in many fields.\n"
        "The references therein are useful context.\n"
        "But there is no references heading in this document.\n"
    )
    assert refs_mod._find_reference_section(content) is None


def test_find_reference_section_accepts_markdown_bold_heading():
    content = "## Method\n" "Some body text.\n\n" "### **References**\n" "Smith, J. 2020. Example reference.\n"
    section = refs_mod._find_reference_section(content)
    assert section is not None
    assert "Smith, J. 2020. Example reference." in section


def test_find_reference_section_accepts_bold_heading_without_markdown_hash():
    content = "Discussion\n" "Some body text.\n\n" "**References**\n" "Smith, J. 2020. Example reference.\n"
    section = refs_mod._find_reference_section(content)
    assert section is not None
    assert "Smith, J. 2020. Example reference." in section


def test_find_reference_section_stops_before_appendix_heading():
    content = (
        "References\n"
        "Akari Asai et al. 2020. Learning to retrieve reasoning paths.\n\n"
        "A Appendix\n"
        "A.1 Prompts for Query Decomposition\n"
    )
    section = refs_mod._find_reference_section(content)
    assert section is not None
    assert "Akari Asai" in section
    assert "A Appendix" not in section
    assert "A.1 Prompts" not in section


def test_find_reference_section_keeps_wrapped_reference_title_lines_without_year():
    content = (
        "References\n"
        "[PMZ19] Georgia Papadogeorgou, Fabrizia Mealli, and Corwin M Zigler.\n"
        "Causal inference with interfering units for cluster and population level\n"
        "treatment allocation programs. Biometrics, 75(3):778-787, 2019.\n\n"
        "A Appendix\n"
        "Supplementary material.\n"
    )
    section = refs_mod._find_reference_section(content)
    assert section is not None
    assert "Causal inference with interfering units" in section
    assert "treatment allocation programs" in section
    assert "A Appendix" not in section


def test_find_reference_section_keeps_title_case_continuation_lines():
    content = (
        "References\n"
        "[1] He Y., Migkas K., et al., 2025. Characterising Galaxy Cluster Scaling Relations as\n"
        "Cosmic Isotropy Tracers Using the FLAMINGO Simulations\n"
        "(arXiv:2504.01745).\n\n"
        "Appendix\n"
        "Extra derivations.\n"
    )
    section = refs_mod._find_reference_section(content)
    assert section is not None
    assert "Cosmic Isotropy Tracers Using the FLAMINGO Simulations" in section
    assert "Appendix" not in section


def test_find_reference_section_fallback_detects_dense_numbered_tail_without_heading():
    content = (
        "Introduction\n"
        "Body text without explicit references heading.\n\n"
        "Some trailing section text.\n\n"
        "[41] First Author. 2021. Paper One.\n"
        "[42] Second Author. 2022. Paper Two.\n"
        "[43] Third Author. 2023. Paper Three.\n"
        "[44] Fourth Author. 2024. Paper Four.\n"
        "[45] Fifth Author. 2025. Paper Five.\n"
        "[46] Sixth Author. 2026. Paper Six.\n"
    )
    section = refs_mod._find_reference_section(content)
    assert section is not None
    assert "[41] First Author." in section
    assert "[46] Sixth Author." in section


def test_split_references_filters_appendix_like_noise():
    refs_text = (
        "Akari Asai et al. 2020. Learning to retrieve reasoning paths.\n\n"
        "Appendix A. Additional notes.\n\n"
        "Jinze Bai et al. 2023. Qwen technical report. arXiv:2309.16609.\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert any("Akari Asai" in r for r in refs)
    assert any("Qwen technical report" in r for r in refs)
    assert all("Appendix A." not in r for r in refs)


def test_split_references_filters_figure_noise_blocks():
    refs_text = (
        "Figure 3. Retrieval pipeline overview.\n\n"
        "Akari Asai et al. 2020. Learning to retrieve reasoning paths.\n\n"
        "Table 2. Ablation metrics.\n\n"
        "Jinze Bai et al. 2023. Qwen technical report. arXiv:2309.16609.\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert any("Akari Asai" in r for r in refs)
    assert any("Qwen technical report" in r for r in refs)
    assert all("Figure 3" not in r for r in refs)
    assert all("Table 2" not in r for r in refs)


def test_split_references_supports_bracket_code_labels():
    refs_text = (
        "[BBI [+] 21] Patrick Bajari, Brian Burdick, Guido W. Imbens, Lorenzo Masoero, "
        "James McQueen, Thomas Richardson, and Ido M. Rosen. Multiple randomization designs, 2021.\n\n"
        "[BP98] Sergey Brin and Lawrence Page. The anatomy of a large-scale hypertextual "
        "Web search engine. Computer Networks and ISDN Systems, 30(1):107-117, 1998.\n\n"
        "13\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert len(refs) >= 2
    assert any("Multiple randomization designs" in r for r in refs)
    assert any("hypertextual Web search engine" in r for r in refs)
    assert all(r.strip() != "13" for r in refs)


def test_split_references_with_meta_reports_truncation():
    refs_text = "\n\n".join(
        f"[{idx}] Author, A. {2000 + (idx % 20)}. Example reference {idx}." for idx in range(1, 113)
    )
    refs, total_detected, truncated = refs_mod._split_references_with_meta(refs_text)
    assert total_detected == 112
    assert truncated is True
    assert len(refs) == refs_mod.MAX_PARSED_REFERENCES


def test_split_references_supports_double_bracket_number_labels():
    refs_text = (
        "[[68] T. Jia, Y. Sang, and X. Zhang, Phys. Rev. D 111, 083531 (2025).]\n\n"
        "[69] D. G. Figueroa, A. Florio, F. Torrenti, and W. Valkenburg. 2021. JCAP.\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert len(refs) >= 2
    assert any("T. Jia, Y. Sang, and X. Zhang" in r for r in refs)
    assert any("D. G. Figueroa" in r for r in refs)


def test_split_references_supports_non_comma_author_names():
    refs_text = (
        "[1] Samuel A Alexander. Infinite graphs in systematic biology, with an application "
        "to the species problem. Acta biotheoretica, 61:181-201, 2013.\n\n"
        "[2] Samuel A Alexander. Self-referential theories. The Journal of Symbolic Logic, "
        "85(4):1687-1716, 2020.\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert len(refs) >= 2
    assert any("Infinite graphs in systematic biology" in r for r in refs)
    assert any("Self-referential theories" in r for r in refs)


def test_split_references_ignores_markdown_link_fragments_without_authors():
    refs_text = (
        "Abazajian K. N., Adelman-McCarthy J. K., Ageros M. A., et al.\n"
        "[2009, ApJS, 182, 543](http://dx.doi.org/10.1088/0067-0049/182/2/543)\n"
        "[Abazajian K., Addison G., Adshead P., et al., 2022, arXiv e-prints, p. arXiv:2203.08024]"
        "(http://dx.doi.org/10.48550/arXiv.2203.08024)\n\n"
        "Brunner H., Liu T., Lamer G., Georgakakis A., Merloni A., et al.\n"
        "[2022, A&A, 661, A1](http://dx.doi.org/10.1051/0004-6361/202141266)\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert len(refs) >= 2
    assert any("Abazajian K." in r for r in refs)
    assert any("Brunner H." in r for r in refs)
    assert all("[2009, ApJS, 182, 543]" not in r.strip() for r in refs)


def test_split_references_skips_broken_markdown_fragment_lines():
    refs_text = (
        "Learn-](https://openreview.net/forum?id=abc)\n"
        "Arian Askari, Roxana Petcu, Chuan Meng, and Suzan Verberne. 2025. "
        "Self-seeding and multi-intent self-instructing llms. arXiv:2402.11633.\n\n"
        "Jinze Bai, Shuai Bai, Yunfei Chu, and others. 2023. "
        "Qwen technical report. arXiv:2309.16609.\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert len(refs) >= 2
    assert all("Learn-](https://openreview.net/forum?id=abc)" not in r for r in refs)
    assert any("Self-seeding and multi-intent self-instructing llms" in r for r in refs)
    assert any("Qwen technical report" in r for r in refs)


def test_split_references_rejects_prose_lines_with_year_and_url_noise():
    refs_text = (
        "Today, President Joe Biden announced a new policy in 2026. "
        "https://example.com/policy-update\n\n"
        "Akari Asai, Kazuma Hashimoto, Hannaneh Hajishirzi, Richard Socher, and "
        "Caiming Xiong. 2020. Learning to retrieve reasoning paths. "
        "In ICLR 2020.\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert any("Learning to retrieve reasoning paths" in r for r in refs)
    assert all("President Joe Biden" not in r for r in refs)


def test_split_references_merges_continuation_lines_in_numbered_mode():
    refs_text = (
        "[5] L. Kofman, A. D. Linde, and A. A. Starobinsky, Phys.\n"
        "Rev. Lett. 73, 3195 (1994), arXiv:hep-th/9405187.\n\n"
        "[6] L. Kofman, A. D. Linde, and A. A. Starobinsky, Phys.\n"
        "Rev. D 56, 3258 (1997), arXiv:hep-ph/9704452.\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert len(refs) == 2
    assert any("Rev. Lett. 73, 3195 (1994)" in r for r in refs)
    assert any("Rev. D 56, 3258 (1997)" in r for r in refs)


def test_split_references_keeps_structured_split_when_line_model_over_merges():
    refs_text = (
        "[R1] 2020. First survey result in compact format. https://doi.org/10.1000/r1\n"
        "[R2] 2021. Second survey result in compact format. https://doi.org/10.1000/r2\n"
        "[R3] 2022. Third survey result in compact format. https://doi.org/10.1000/r3\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert len(refs) == 3
    assert any("First survey result" in r for r in refs)
    assert any("Second survey result" in r for r in refs)
    assert any("Third survey result" in r for r in refs)


def test_split_references_keeps_standalone_year_markdown_entries():
    refs_text = (
        "[2019, ApJS, 182, 543](http://dx.doi.org/10.1088/0067-0049/182/2/543)\n\n"
        "[2021, Phys. Rev. D, 103, 123456](http://dx.doi.org/10.1103/PhysRevD.103.123456)\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert len(refs) >= 2
    assert any("ApJS, 182, 543" in r for r in refs)
    assert any("Phys. Rev. D, 103, 123456" in r for r in refs)


def test_split_references_supports_unicode_author_surnames_for_new_entries():
    refs_text = (
        "Ebrahimian E., Krishnan C., Mondol R., Sheikh-Jabbari M. M.\n"
        "[2024, Journal of Cosmology and Astroparticle Physics, 2024, 036]"
        "(http://dx.doi.org/10.1088/1475-7516/2024/01/036)\n"
        "[Gaztañaga E., Sravan Kumar K., 2024, J. Cosmol. Astropart.]"
        "(http://dx.doi.org/10.1088/1475-7516/2024/06/001)\n"
        "[Phys., 2024, 001](http://dx.doi.org/10.1088/1475-7516/2024/06/001)\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert len(refs) >= 2
    assert any("Ebrahimian E." in r for r in refs)
    assert any("Gaztañaga E." in r for r in refs)


def test_split_references_merges_split_journal_fragment_markdown_continuation():
    refs_text = (
        "Constantin A., Harvey T. R., von Hausegger S., Lukas A., 2023,\n"
        "[Classical and Quantum Gravity, 40, 245015](http://dx.doi.org/10.1088/1361-6382/ad0b36)\n\n"
        "[Gelman A., Rubin D. B., 1992, Stat. Sci., 7, 457](http://dx.doi.org/10.1214/ss/1177011136)\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert len(refs) >= 2
    assert any("Constantin A." in r and "Classical and Quantum Gravity" in r for r in refs)
    assert any("Gelman A." in r for r in refs)


def test_split_references_merges_split_arxiv_e_prints_fragment():
    refs_text = (
        "[Desmond H., Stiskalek R., Najera J. A., Banik I., 2025, arXiv e-]"
        "(http://dx.doi.org/10.48550/arXiv.2511.03394)\n"
        "[prints, p. arXiv:2511.03394](http://dx.doi.org/10.48550/arXiv.2511.03394)\n\n"
        "[Hoekstra H., 2013, Space Science Reviews, 177, 247]"
        "(http://dx.doi.org/10.1007/s11214-013-9994-5)\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert len(refs) >= 2
    assert any("Desmond H." in r and "prints, p. arXiv:2511.03394" in r for r in refs)
    assert any("Hoekstra H." in r for r in refs)


def test_split_references_merges_author_prefix_with_markdown_year_line():
    refs_text = (
        "Górski K. M., Hivon E., Banday A. J., Wandelt B. D., Hansen\n"
        "[F. K., Reinecke M., Bartelmann M., 2005, ApJ, 622, 759]"
        "(http://dx.doi.org/10.1086/427976)\n\n"
        "Courtois H. M., Dupuy A., Guinet D., Baulieu G., Ruppin F.,\n"
        "[Brenas P., 2023, A&A, 670, L15](http://dx.doi.org/10.1051/0004-6361/202245331)\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert len(refs) == 2
    assert any("Górski K. M." in r and "2005, ApJ, 622, 759" in r for r in refs)
    assert any("Courtois H. M." in r and "2023, A&A, 670, L15" in r for r in refs)


def test_split_references_merges_arxiv_category_fragment_tail():
    refs_text = (
        "[17] M. Piani and J. Rubio, JCAP **12**, 002, arXiv:2304.13056\n"
        "[[hep-ph].]\n"
        "[18] S.-Y. Zhou et al., JHEP **10**, 026, arXiv:1304.6094\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert len(refs) == 2
    assert any("2304.13056" in r for r in refs)
    assert not any(r.strip() == "[[hep-ph].]" for r in refs)


def test_parse_reference_basic_normalizes_markdown_link_text_but_keeps_identifiers():
    raw = (
        "[Abbott T. M. C., et al., 2022, Physical Review D, 105, 023520]"
        "(http://dx.doi.org/10.1103/PhysRevD.105.023520)"
    )
    parsed = refs_mod._parse_reference_basic(raw)

    assert "](" not in parsed.raw_text
    assert parsed.raw_text.startswith("Abbott T. M. C., et al., 2022")
    assert parsed.doi == "10.1103/PhysRevD.105.023520"
    assert parsed.url == "https://doi.org/10.1103/PhysRevD.105.023520"
    assert parsed.year == 2022


def test_parse_reference_basic_normalizes_broken_markdown_fragment():
    raw = (
        "Learn-](https://openreview.net/forum?id=abc) Samarth Bhargav, Georgios "
        "Sidiropoulos, and Evangelos Kanoulas. 2022. It's on the tip of my tongue."
    )
    parsed = refs_mod._parse_reference_basic(raw)

    assert "](" not in parsed.raw_text
    assert "Learn-](https://" not in parsed.raw_text
    assert not parsed.raw_text.startswith("Learn- ")
    assert "Samarth Bhargav" in parsed.raw_text
    assert parsed.year == 2022


def test_parse_reference_basic_infers_year_from_arxiv_id_when_missing():
    raw = "[17] M. Piani and J. Rubio, JCAP **12**, 002, arXiv:2304.13056"
    parsed = refs_mod._parse_reference_basic(raw)

    assert parsed.arxiv_id == "2304.13056"
    assert parsed.year == 2023


def test_split_references_keeps_19th_century_entries_in_numbered_lists():
    refs_text = (
        "[7] Joseph T Chang. Reply to discussants: Recent common ancestors of all present-day individuals. "
        "_Advances in Applied Probability_, 31(4):1036-1038, 1999.\n"
        "[8] Charles Darwin. _The Voyage of the Beagle_ . 1839.\n"
        "[9] Charles Darwin. _On The Origin of Species_ . John Murray, 6th edition, 1872.\n"
        "[10] Francis Darwin. The Knight-Darwin Law. _Nature_, 58(1513):630-632, 1898.\n"
        "[11] Peter Donnelly et al. Discussion: Recent common ancestors of all present-day individuals. "
        "_Advances in Applied Probability_, 31(4):1027-1035, 1999.\n"
    )
    refs = refs_mod._split_references(refs_text)
    assert any(r.startswith("[8] Charles Darwin") for r in refs)
    assert any(r.startswith("[9] Charles Darwin") for r in refs)
    assert any(r.startswith("[10] Francis Darwin") for r in refs)


def test_extract_year_supports_19th_century_dates():
    assert refs_mod._extract_year("Charles Darwin. _The Voyage of the Beagle_ . 1839.") == 1839
    assert refs_mod._extract_year("Francis Darwin (1898). The Knight-Darwin Law.") == 1898


@pytest.mark.asyncio
async def test_references_endpoint_enriches_only_requested_reference_index(mock_user, mock_db):
    content = (
        "Intro text\\n"
        "References\\n"
        "[1] Smith, J. (2020). Example Paper. https://doi.org/10.1234/abcd\\n"
        "[2] Doe, A. (2019). Another Paper. arXiv:2101.12345\\n"
    )
    mock_db.get_media_by_id = MagicMock(return_value={"id": 1, "content": content})

    async def enrich_semantic(refs: list[ReferenceEntry]):
        assert len(refs) == 1
        out = [r.model_copy() for r in refs]
        out[0].citation_count = 77
        return out, True

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with (
        patch.object(refs_mod, "get_cached_response", return_value=None),
        patch.object(refs_mod, "_is_provider_cooldown", return_value=False),
        patch.object(
            refs_mod,
            "_enrich_with_semantic_scholar",
            new=AsyncMock(side_effect=enrich_semantic),
        ),
        patch.object(
            refs_mod,
            "_enrich_with_crossref",
            new=AsyncMock(side_effect=lambda refs: (refs, False)),
        ),
        patch.object(
            refs_mod,
            "_enrich_with_arxiv",
            new=AsyncMock(side_effect=lambda refs: (refs, False)),
        ),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/references?enrich=true&reference_index=1")

    assert response.status_code == 200
    data = response.json()
    assert data["references"][0].get("citation_count") is None
    assert data["references"][1]["citation_count"] == 77
    assert "semantic_scholar" in (data.get("enrichment_source") or "")

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_references_endpoint_reference_index_out_of_range_returns_400(mock_user, mock_db):
    content = "References\\n" "[1] Smith, J. (2020). Example Paper. https://doi.org/10.1234/abcd\\n"
    mock_db.get_media_by_id = MagicMock(return_value={"id": 1, "content": content})
    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with patch.object(refs_mod, "get_cached_response", return_value=None):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/references?enrich=true&reference_index=5")

    assert response.status_code == 400
    assert response.json()["detail"] == "reference_index out of range"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_references_endpoint_skips_enrichment_when_all_providers_in_cooldown(mock_user, mock_db):
    content = "References\\n" "[1] Smith, J. (2020). Example Paper. https://doi.org/10.1234/abcd\\n"
    mock_db.get_media_by_id = MagicMock(return_value={"id": 1, "content": content})
    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    semantic = AsyncMock(side_effect=lambda refs: (refs, True))
    crossref = AsyncMock(side_effect=lambda refs: (refs, True))
    arxiv = AsyncMock(side_effect=lambda refs: (refs, True))

    with (
        patch.object(refs_mod, "get_cached_response", return_value=None),
        patch.object(refs_mod, "_is_provider_cooldown", return_value=True),
        patch.object(refs_mod, "_enrich_with_semantic_scholar", new=semantic),
        patch.object(refs_mod, "_enrich_with_crossref", new=crossref),
        patch.object(refs_mod, "_enrich_with_arxiv", new=arxiv),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/references?enrich=true")

    assert response.status_code == 200
    semantic.assert_not_awaited()
    crossref.assert_not_awaited()
    arxiv.assert_not_awaited()

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_references_endpoint_sanitizes_enrichment_failure_log(mock_user, mock_db, monkeypatch):
    content = "References\\n" "[1] Smith, J. (2020). Example Paper. https://doi.org/10.1234/abcd\\n"
    mock_db.get_media_by_id = MagicMock(return_value={"id": 1, "content": content})
    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db
    logger_stub = _LoggerStub()
    monkeypatch.setattr(refs_mod, "logger", logger_stub, raising=True)

    with (
        patch.object(refs_mod, "get_cached_response", return_value=None),
        patch.object(refs_mod, "_is_provider_cooldown", return_value=False),
        patch.object(
            refs_mod,
            "_enrich_with_semantic_scholar",
            new=AsyncMock(side_effect=RuntimeError("enrichment failed at /private/provider.key")),
        ),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/references?enrich=true")

    assert response.status_code == 200
    assert response.json()["enrichment_source"] is None
    assert [args[0] for args, _kwargs in logger_stub.warning_calls if args] == [
        "Failed to enrich references"
    ]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.warning_calls)
    rendered_calls = repr(logger_stub.warning_calls)
    assert "media_id" not in rendered_calls
    assert "enrichment failed" not in rendered_calls
    assert "/private/provider.key" not in rendered_calls

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_enrich_with_semantic_scholar_caps_external_calls_at_five():
    refs = [ReferenceEntry(raw_text=f"Ref {i}", doi=f"10.1234/{i}") for i in range(7)]
    call_count = 0

    async def fake_to_thread(_func, *_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return {"paperId": "p1", "citationCount": 1, "title": "Paper"}, None

    with (
        patch.object(refs_mod, "_get_cached_external", return_value=None),
        patch.object(refs_mod, "_set_cached_external", return_value=None),
        patch.object(refs_mod.asyncio, "to_thread", side_effect=fake_to_thread),
    ):
        enriched, performed = await refs_mod._enrich_with_semantic_scholar(refs)

    assert performed is True
    assert len(enriched) == 7
    assert call_count == 5


@pytest.mark.asyncio
async def test_enrich_with_semantic_scholar_sanitizes_lookup_failure_logs(monkeypatch):
    logger_mock = MagicMock()
    monkeypatch.setattr(refs_mod, "logger", logger_mock)
    refs = [
        ReferenceEntry(
            raw_text="Ref 1",
            doi="10.9999/private-doi",
            arxiv_id="2401.99999",
            title="Sensitive Title /private/reference.txt",
        )
    ]

    with (
        patch.object(refs_mod, "_get_cached_external", return_value=None),
        patch.object(refs_mod, "_set_cached_external", return_value=None),
        patch.object(refs_mod.asyncio, "sleep", new=AsyncMock()),
        patch.object(
            refs_mod.asyncio,
            "to_thread",
            new=AsyncMock(side_effect=RuntimeError("provider leaked /private/token")),
        ),
    ):
        enriched, performed = await refs_mod._enrich_with_semantic_scholar(refs)

    assert performed is False
    assert enriched[0] == refs[0]
    _assert_sanitized_debug_messages(
        logger_mock,
        [
            "Semantic Scholar DOI lookup failed",
            "Semantic Scholar arXiv lookup failed",
            "Semantic Scholar title search failed",
        ],
        [
            "10.9999/private-doi",
            "2401.99999",
            "Sensitive Title",
            "provider leaked",
            "/private/token",
            "/private/reference.txt",
        ],
    )


@pytest.mark.asyncio
async def test_enrich_with_crossref_sets_cooldown_on_rate_limit():
    refs = [ReferenceEntry(raw_text=f"Ref {i}", doi=f"10.1234/{i}") for i in range(3)]
    call_count = 0

    async def fake_to_thread(_func, *_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return None, "429 Too Many Requests"

    with (
        patch.object(refs_mod, "_get_cached_external", return_value=None),
        patch.object(refs_mod, "_set_cached_external", return_value=None),
        patch.object(refs_mod, "_set_provider_cooldown") as set_cooldown,
        patch.object(refs_mod.asyncio, "to_thread", side_effect=fake_to_thread),
    ):
        enriched, performed = await refs_mod._enrich_with_crossref(refs)

    assert performed is False
    assert len(enriched) == 3
    assert call_count == 1
    set_cooldown.assert_called_once_with("crossref")


@pytest.mark.asyncio
async def test_enrich_with_crossref_and_arxiv_sanitizes_lookup_failure_logs(monkeypatch):
    logger_mock = MagicMock()
    monkeypatch.setattr(refs_mod, "logger", logger_mock)

    with (
        patch.object(refs_mod, "_get_cached_external", return_value=None),
        patch.object(refs_mod, "_set_cached_external", return_value=None),
        patch.object(
            refs_mod.asyncio,
            "to_thread",
            new=AsyncMock(side_effect=RuntimeError("provider leaked /private/token")),
        ),
    ):
        crossref_enriched, crossref_performed = await refs_mod._enrich_with_crossref(
            [ReferenceEntry(raw_text="Ref 1", doi="10.9999/private-doi")]
        )

    with (
        patch.object(refs_mod, "_get_cached_external", return_value=None),
        patch.object(refs_mod, "_set_cached_external", return_value=None),
        patch.object(
            refs_mod.asyncio,
            "to_thread",
            new=AsyncMock(side_effect=RuntimeError("provider leaked /private/token")),
        ),
    ):
        arxiv_enriched, arxiv_performed = await refs_mod._enrich_with_arxiv(
            [ReferenceEntry(raw_text="Ref 2", arxiv_id="2401.99999")]
        )

    assert crossref_performed is False
    assert arxiv_performed is False
    assert crossref_enriched[0].doi == "10.9999/private-doi"
    assert arxiv_enriched[0].arxiv_id == "2401.99999"
    _assert_sanitized_debug_messages(
        logger_mock,
        [
            "Crossref lookup failed",
            "arXiv lookup failed",
        ],
        [
            "10.9999/private-doi",
            "2401.99999",
            "provider leaked",
            "/private/token",
        ],
    )


@pytest.mark.asyncio
async def test_enrich_with_crossref_uses_cached_external_result_without_network_call():
    refs = [ReferenceEntry(raw_text="Ref 1", doi="10.1234/abc")]
    cached_item = {
        "title": "Cached Crossref Title",
        "authors": "A. Author",
        "journal": "Journal",
        "pub_date": "2021-01-01",
        "doi": "10.1234/abc",
        "url": "https://doi.org/10.1234/abc",
        "pdf_url": "https://example.com/cached.pdf",
    }

    with (
        patch.object(
            refs_mod,
            "_get_cached_external",
            return_value=(cached_item, None),
        ),
        patch.object(refs_mod, "_set_cached_external", return_value=None),
        patch.object(refs_mod.asyncio, "to_thread", new=AsyncMock()) as to_thread_mock,
    ):
        enriched, performed = await refs_mod._enrich_with_crossref(refs)

    assert performed is True
    assert enriched[0].title == "Cached Crossref Title"
    to_thread_mock.assert_not_awaited()
