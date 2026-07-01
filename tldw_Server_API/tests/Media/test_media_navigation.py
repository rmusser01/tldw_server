from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints.media import navigation as navigation_mod
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.main import app

FAKE_PDF_BYTES = b"%PDF-1.4\nfake pdf bytes"


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_user():
    user = MagicMock()
    user.id = 1
    user.username = "testuser"
    return user


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.backend_type = "sqlite"
    db.get_media_by_id = MagicMock()
    db.get_media_file = MagicMock()
    db.execute_query = MagicMock()
    return db


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.exists = AsyncMock(return_value=True)
    storage.get_size = AsyncMock(return_value=1024)
    storage.retrieve = AsyncMock(return_value=io.BytesIO(FAKE_PDF_BYTES))
    return storage


@pytest.mark.asyncio
async def test_navigation_prefers_pdf_outline(mock_user, mock_db, mock_storage):
    mock_db.get_media_by_id.return_value = {
        "id": 1,
        "type": "pdf",
        "title": "Test PDF",
        "version": 7,
        "last_modified": "2026-02-09T10:00:00Z",
    }
    mock_db.get_media_file.return_value = {
        "storage_path": "user_1/media/1/original.pdf",
        "mime_type": "application/pdf",
    }

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    outline_entries = [
        SimpleNamespace(level=1, title="Chapter 1", page=1),
        SimpleNamespace(level=2, title="Section 1.1", page=3),
    ]

    with patch.object(
        navigation_mod,
        "get_storage_backend",
        return_value=mock_storage,
    ), patch.object(
        navigation_mod,
        "_check_pymupdf_available",
        return_value=True,
    ), patch.object(
        navigation_mod,
        "_extract_pdf_outline",
        return_value=(outline_entries, 12),
    ), patch.object(
        navigation_mod,
        "get_cached_response",
        return_value=None,
    ), patch.object(
        navigation_mod,
        "cache_response",
        return_value="etag",
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/navigation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["media_id"] == 1
    assert payload["available"] is True
    assert payload["source_order_used"] == ["pdf_outline"]
    assert payload["stats"]["node_count"] == 2
    assert payload["stats"]["returned_node_count"] == 2
    assert payload["stats"]["truncated"] is False
    assert payload["nodes"][0]["target_type"] == "page"
    assert payload["nodes"][0]["path_label"] == "1"
    assert payload["nodes"][1]["path_label"] == "1.1"
    mock_db.execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_navigation_falls_back_to_document_structure_index(mock_user, mock_db):
    mock_db.get_media_by_id.return_value = {
        "id": 1,
        "type": "document",
        "title": "Structured Doc",
        "version": 3,
        "last_modified": "2026-02-09T11:00:00Z",
    }
    mock_db.get_media_file.return_value = None

    dsi_rows = [
        {
            "id": 10,
            "parent_id": None,
            "level": 1,
            "title": "Chapter 1",
            "start_char": 0,
            "end_char": 100,
            "order_index": 0,
            "path": "Chapter 1",
        },
        {
            "id": 11,
            "parent_id": 10,
            "level": 2,
            "title": "Section 1.1",
            "start_char": 10,
            "end_char": 40,
            "order_index": 0,
            "path": "Chapter 1 / Section 1.1",
        },
    ]

    def _execute_query(query, _params):
        if "DocumentStructureIndex" in query:
            return _Cursor(dsi_rows)
        if "UnvectorizedMediaChunks" in query:
            return _Cursor([])
        raise AssertionError(f"Unexpected query: {query}")

    mock_db.execute_query.side_effect = _execute_query

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with patch.object(navigation_mod, "get_cached_response", return_value=None), patch.object(
        navigation_mod,
        "cache_response",
        return_value="etag",
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/navigation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_order_used"] == [
        "pdf_outline",
        "generated_toc",
        "document_structure_index",
    ]
    assert payload["nodes"][0]["id"] == "dsi:10"
    assert payload["nodes"][0]["target_type"] == "char_range"
    assert payload["nodes"][0]["path_label"] == "1"
    assert payload["nodes"][1]["parent_id"] == "dsi:10"
    assert payload["nodes"][1]["path_label"] == "1.1"


@pytest.mark.asyncio
async def test_navigation_falls_back_to_chunk_metadata(mock_user, mock_db):
    mock_db.get_media_by_id.return_value = {
        "id": 1,
        "type": "text",
        "title": "Chunked Doc",
        "version": 4,
        "last_modified": "2026-02-09T12:00:00Z",
    }
    mock_db.get_media_file.return_value = None

    chunk_rows = [
        {
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 80,
            "metadata": json.dumps({"section_path": "Chapter 12 > Section 5"}),
        },
        {
            "chunk_index": 1,
            "start_char": 80,
            "end_char": 120,
            "metadata": json.dumps({"section_path": "Chapter 12 > Section 6"}),
        },
    ]

    def _execute_query(query, _params):
        if "DocumentStructureIndex" in query:
            return _Cursor([])
        if "UnvectorizedMediaChunks" in query:
            return _Cursor(chunk_rows)
        raise AssertionError(f"Unexpected query: {query}")

    mock_db.execute_query.side_effect = _execute_query

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with patch.object(navigation_mod, "get_cached_response", return_value=None), patch.object(
        navigation_mod,
        "cache_response",
        return_value="etag",
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/navigation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_order_used"] == [
        "pdf_outline",
        "generated_toc",
        "document_structure_index",
        "transcript_segment",
        "chunk_metadata",
    ]
    assert payload["available"] is True
    assert len(payload["nodes"]) == 3
    assert payload["nodes"][0]["title"] == "Chapter 12"
    assert payload["nodes"][1]["title"] == "Section 5"
    assert payload["nodes"][2]["title"] == "Section 6"
    assert payload["nodes"][1]["target_type"] == "char_range"


@pytest.mark.asyncio
async def test_navigation_falls_back_to_transcript_segments(mock_user, mock_db):
    mock_db.get_media_by_id.return_value = {
        "id": 1,
        "type": "video",
        "title": "Transcript Video",
        "version": 6,
        "last_modified": "2026-02-09T12:30:00Z",
    }
    mock_db.get_media_file.return_value = None

    def _execute_query(query, _params):
        if "DocumentStructureIndex" in query:
            return _Cursor([])
        if "UnvectorizedMediaChunks" in query:
            return _Cursor([])
        raise AssertionError(f"Unexpected query: {query}")

    mock_db.execute_query.side_effect = _execute_query

    transcript_payload = {
        "text": "Intro details Follow-up details",
        "segments": [
            {
                "text": "Intro details",
                "start_seconds": 0.0,
                "end_seconds": 12.5,
            },
            {
                "Text": "Follow-up details",
                "start_seconds": 12.5,
                "end_seconds": 27.0,
                "speaker": "Host",
            },
        ],
    }

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with patch.object(navigation_mod, "get_media_transcripts", return_value=[{"transcription": json.dumps(transcript_payload)}]), patch.object(
        navigation_mod,
        "get_cached_response",
        return_value=None,
    ), patch.object(
        navigation_mod,
        "cache_response",
        return_value="etag",
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/navigation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_order_used"] == [
        "pdf_outline",
        "generated_toc",
        "document_structure_index",
        "transcript_segment",
    ]
    assert payload["available"] is True
    assert len(payload["nodes"]) == 2
    assert payload["nodes"][0]["source"] == "transcript_segment"
    assert payload["nodes"][0]["target_type"] == "time_range"
    assert payload["nodes"][0]["target_start"] == 0.0
    assert payload["nodes"][0]["target_end"] == 12.5
    assert payload["nodes"][1]["title"].startswith("Host:")


@pytest.mark.asyncio
async def test_navigation_generated_fallback_is_opt_in(mock_user, mock_db):
    mock_db.get_media_by_id.return_value = {
        "id": 1,
        "type": "text",
        "title": "Generated fallback source",
        "version": 7,
        "last_modified": "2026-02-09T14:00:00Z",
    }
    mock_db.get_media_file.return_value = None

    def _execute_query(query, _params):
        if "DocumentStructureIndex" in query:
            return _Cursor([])
        if "UnvectorizedMediaChunks" in query:
            return _Cursor([])
        raise AssertionError(f"Unexpected query: {query}")

    mock_db.execute_query.side_effect = _execute_query

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    generated_source_text = (
        "# Chapter 1\n"
        "First chapter body.\n\n"
        "## Section 1.1\n"
        "Section details here.\n\n"
        "# Chapter 2\n"
        "Second chapter body."
    )

    with patch.object(navigation_mod, "get_media_transcripts", return_value=[]), patch.object(
        navigation_mod,
        "get_document_version",
        return_value={"content": generated_source_text},
    ), patch.object(
        navigation_mod,
        "get_cached_response",
        return_value=None,
    ), patch.object(
        navigation_mod,
        "cache_response",
        return_value="etag",
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            no_fallback_resp = await client.get("/api/v1/media/1/navigation")
            fallback_resp = await client.get(
                "/api/v1/media/1/navigation?include_generated_fallback=true"
            )

    assert no_fallback_resp.status_code == 200
    no_fallback_payload = no_fallback_resp.json()
    assert no_fallback_payload["available"] is False
    assert no_fallback_payload["nodes"] == []
    assert no_fallback_payload["source_order_used"] == [
        "pdf_outline",
        "generated_toc",
        "document_structure_index",
        "transcript_segment",
        "chunk_metadata",
    ]

    assert fallback_resp.status_code == 200
    fallback_payload = fallback_resp.json()
    assert fallback_payload["available"] is True
    assert fallback_payload["nodes"][0]["source"] == "generated"
    assert fallback_payload["nodes"][0]["target_type"] == "char_range"
    assert fallback_payload["source_order_used"] == [
        "pdf_outline",
        "generated_toc",
        "document_structure_index",
        "transcript_segment",
        "chunk_metadata",
        "generated",
    ]


@pytest.mark.asyncio
async def test_navigation_prefers_generated_when_pdf_outline_is_sparse(mock_user, mock_db, mock_storage):
    mock_db.get_media_by_id.return_value = {
        "id": 1,
        "type": "pdf",
        "title": "Sparse outline PDF",
        "version": 8,
        "last_modified": "2026-02-09T19:00:00Z",
    }
    mock_db.get_media_file.return_value = {
        "storage_path": "user_1/media/1/original.pdf",
        "mime_type": "application/pdf",
    }

    def _execute_query(query, _params):
        if "DocumentStructureIndex" in query:
            return _Cursor([])
        if "UnvectorizedMediaChunks" in query:
            return _Cursor([])
        raise AssertionError(f"Unexpected query: {query}")

    mock_db.execute_query.side_effect = _execute_query

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    sparse_outline_entries = [
        SimpleNamespace(level=1, title="Cover", page=1),
        SimpleNamespace(level=1, title="Contents", page=2),
    ]

    generated_source_text = (
        "# Chapter 1\n"
        "Intro.\n\n"
        "## Section 1.1\n"
        "Details.\n\n"
        "## Section 1.2\n"
        "More details."
    )

    with patch.object(navigation_mod, "get_storage_backend", return_value=mock_storage), patch.object(
        navigation_mod,
        "_check_pymupdf_available",
        return_value=True,
    ), patch.object(
        navigation_mod,
        "_extract_pdf_outline",
        return_value=(sparse_outline_entries, 20),
    ), patch.object(
        navigation_mod,
        "get_document_version",
        return_value={"content": generated_source_text},
    ), patch.object(
        navigation_mod,
        "get_cached_response",
        return_value=None,
    ), patch.object(
        navigation_mod,
        "cache_response",
        return_value="etag",
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/navigation?include_generated_fallback=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["nodes"]
    assert payload["nodes"][0]["source"] == "generated"
    assert payload["source_order_used"] == [
        "pdf_outline",
        "generated_toc",
        "document_structure_index",
        "transcript_segment",
        "chunk_metadata",
        "generated",
    ]


@pytest.mark.asyncio
async def test_navigation_keeps_sparse_pdf_outline_when_no_fallback_enabled(mock_user, mock_db, mock_storage):
    mock_db.get_media_by_id.return_value = {
        "id": 1,
        "type": "pdf",
        "title": "Sparse outline PDF",
        "version": 8,
        "last_modified": "2026-02-09T19:15:00Z",
    }
    mock_db.get_media_file.return_value = {
        "storage_path": "user_1/media/1/original.pdf",
        "mime_type": "application/pdf",
    }

    def _execute_query(query, _params):
        if "DocumentStructureIndex" in query:
            return _Cursor([])
        if "UnvectorizedMediaChunks" in query:
            return _Cursor([])
        raise AssertionError(f"Unexpected query: {query}")

    mock_db.execute_query.side_effect = _execute_query

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    sparse_outline_entries = [
        SimpleNamespace(level=1, title="Cover", page=1),
        SimpleNamespace(level=1, title="Contents", page=2),
    ]

    with patch.object(navigation_mod, "get_storage_backend", return_value=mock_storage), patch.object(
        navigation_mod,
        "_check_pymupdf_available",
        return_value=True,
    ), patch.object(
        navigation_mod,
        "_extract_pdf_outline",
        return_value=(sparse_outline_entries, 20),
    ), patch.object(
        navigation_mod,
        "get_cached_response",
        return_value=None,
    ), patch.object(
        navigation_mod,
        "cache_response",
        return_value="etag",
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/navigation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["nodes"]
    assert payload["nodes"][0]["source"] == "pdf_outline"
    assert payload["source_order_used"] == [
        "pdf_outline",
        "generated_toc",
        "document_structure_index",
        "transcript_segment",
        "chunk_metadata",
    ]


def test_generated_heading_filter_skips_noisy_titles():
    content = (
        "# Overview\n"
        "Body\n\n"
        "## !!!!====!!!!\n"
        "Noise\n\n"
        "## 12345\n"
        "Numeric only\n\n"
        "## Results and Discussion\n"
        "Findings\n"
    )
    nodes = navigation_mod._extract_generated_heading_nodes(content)
    titles = [node["title"] for node in nodes]

    assert "Overview" in titles
    assert "Results and Discussion" in titles
    assert "!!!!====!!!!" not in titles
    assert "12345" not in titles


@pytest.mark.asyncio
async def test_navigation_prefers_generated_toc_when_contents_block_exists(mock_user, mock_db):
    mock_db.get_media_by_id.return_value = {
        "id": 1,
        "type": "pdf",
        "title": "TOC-heavy doc",
        "version": 9,
        "last_modified": "2026-02-09T20:15:00Z",
    }
    mock_db.get_media_file.return_value = None

    def _execute_query(query, _params):
        if "DocumentStructureIndex" in query:
            return _Cursor([])
        if "UnvectorizedMediaChunks" in query:
            return _Cursor([])
        raise AssertionError(f"Unexpected query: {query}")

    mock_db.execute_query.side_effect = _execute_query

    toc_content = (
        "**CONTENTS**\n\n"
        "I. Introduction 2\n"
        "II. Methods 4\n"
        "A. Data 5\n"
        "B. Training Setup 6\n\n"
        "**I.** **INTRODUCTION**\n"
        "Body starts here."
    )

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with patch.object(navigation_mod, "get_document_version", return_value={"content": toc_content}), patch.object(
        navigation_mod,
        "get_cached_response",
        return_value=None,
    ), patch.object(
        navigation_mod,
        "cache_response",
        return_value="etag",
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/navigation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["nodes"]
    assert payload["nodes"][0]["source"] == "generated_toc"
    assert payload["nodes"][0]["title"] == "I. Introduction"
    assert payload["nodes"][0]["target_type"] == "page"
    assert payload["nodes"][0]["target_start"] == 2
    assert payload["source_order_used"] == ["pdf_outline", "generated_toc"]


def test_generated_toc_parser_extracts_hierarchical_page_nodes():
    content = (
        "Preface\n\n"
        "**TABLE OF CONTENTS**\n\n"
        "Chapter 1 3\n"
        "A. First subtopic 4\n"
        "1. Fine detail 5\n"
        "Chapter 2 8\n\n"
        "**Chapter 1**\n"
        "Main body."
    )

    db = MagicMock()
    media = {"type": "pdf", "content": content}

    with patch.object(navigation_mod, "_get_media_text", return_value=content):
        nodes = navigation_mod._extract_generated_toc_nodes(media_id=42, db=db, media=media)

    assert [node["title"] for node in nodes[:4]] == [
        "Chapter 1",
        "A. First subtopic",
        "1. Fine detail",
        "Chapter 2",
    ]
    assert [node["level"] for node in nodes[:4]] == [1, 2, 3, 1]
    assert [node["target_type"] for node in nodes[:4]] == ["page", "page", "page", "page"]
    assert [node["target_start"] for node in nodes[:4]] == [3, 4, 5, 8]


def test_generated_toc_parser_promotes_appendix_after_references_and_skips_short_junk():
    content = (
        "**CONTENTS**\n\n"
        "VII. Model comparison and robustness requirements 19\n"
        "A. Likelihood-ratio statistic and Gaussian-equivalent preference 19\n"
        "References 22\n"
        "A. r_d-independent DESI DR2 F_AP(z) data vector and covariance 24\n"
        "D L 5\n\n"
        "**I. INTRODUCTION**\n"
        "Body."
    )

    db = MagicMock()
    media = {"type": "pdf", "content": content}

    with patch.object(navigation_mod, "_get_media_text", return_value=content):
        nodes = navigation_mod._extract_generated_toc_nodes(media_id=77, db=db, media=media)

    titles = [node["title"] for node in nodes]
    assert "D L" not in titles
    assert "A. r_d-independent DESI DR2 F_AP(z) data vector and covariance" in titles

    appendix_node = next(
        node
        for node in nodes
        if node["title"] == "A. r_d-independent DESI DR2 F_AP(z) data vector and covariance"
    )
    references_node = next(node for node in nodes if node["title"] == "References")

    assert appendix_node["level"] == 1
    assert appendix_node["parent_id"] is None
    assert references_node["level"] == 1


def test_clean_navigation_title_preserves_variable_subscripts_from_italic_markup():
    raw = "A. _rd_ -independent DESI DR2 _F_ AP( _z_ ) data vector and covariance 24"
    cleaned = navigation_mod._clean_navigation_title(raw)
    assert cleaned == "A. r_d-independent DESI DR2 F_AP(z) data vector and covariance 24"


def test_clean_navigation_title_preserves_unclosed_html_like_text():
    assert navigation_mod._clean_navigation_title("Section <div") == "Section <div"


def test_clean_navigation_title_preserves_literal_unpaired_asterisks():
    assert navigation_mod._clean_navigation_title("A* search and **bold** note") == "A* search and bold note"


@pytest.mark.asyncio
async def test_navigation_guardrails_and_parent_filter(mock_user, mock_db):
    mock_db.get_media_by_id.return_value = {
        "id": 1,
        "type": "document",
        "title": "Guardrails Doc",
        "version": 5,
        "last_modified": "2026-02-09T13:00:00Z",
    }
    mock_db.get_media_file.return_value = None

    dsi_rows = [
        {
            "id": 10,
            "parent_id": None,
            "level": 1,
            "title": "Root",
            "start_char": 0,
            "end_char": 300,
            "order_index": 0,
            "path": "Root",
        },
        {
            "id": 11,
            "parent_id": 10,
            "level": 2,
            "title": "Child A",
            "start_char": 10,
            "end_char": 100,
            "order_index": 0,
            "path": "Root / Child A",
        },
        {
            "id": 12,
            "parent_id": 10,
            "level": 2,
            "title": "Child B",
            "start_char": 100,
            "end_char": 200,
            "order_index": 1,
            "path": "Root / Child B",
        },
        {
            "id": 13,
            "parent_id": 11,
            "level": 3,
            "title": "Grandchild",
            "start_char": 20,
            "end_char": 60,
            "order_index": 0,
            "path": "Root / Child A / Grandchild",
        },
    ]

    def _execute_query(query, _params):
        if "DocumentStructureIndex" in query:
            return _Cursor(dsi_rows)
        if "UnvectorizedMediaChunks" in query:
            return _Cursor([])
        raise AssertionError(f"Unexpected query: {query}")

    mock_db.execute_query.side_effect = _execute_query

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with patch.object(navigation_mod, "get_cached_response", return_value=None), patch.object(
        navigation_mod,
        "cache_response",
        return_value="etag",
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/navigation?max_depth=2&max_nodes=1")
            child_response = await client.get("/api/v1/media/1/navigation?parent_id=dsi:10")

    assert response.status_code == 200
    payload = response.json()
    assert payload["stats"]["node_count"] == 3
    assert payload["stats"]["returned_node_count"] == 1
    assert payload["stats"]["truncated"] is True
    assert len(payload["nodes"]) == 1

    assert child_response.status_code == 200
    child_payload = child_response.json()
    assert child_payload["stats"]["truncated"] is False
    assert child_payload["stats"]["node_count"] == 2
    assert len(child_payload["nodes"]) == 2
    assert all(node["parent_id"] == "dsi:10" for node in child_payload["nodes"])


@pytest.mark.asyncio
async def test_navigation_returns_404_when_media_missing(mock_user, mock_db):
    mock_db.get_media_by_id.return_value = None

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with patch.object(navigation_mod, "get_cached_response", return_value=None):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/999/navigation")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
