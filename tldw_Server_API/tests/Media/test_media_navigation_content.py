from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints.media import navigation as navigation_mod
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.Media import navigation as navigation_core
from tldw_Server_API.app.main import app


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
    db.lookup_section_by_heading = MagicMock(return_value=None)
    return db


def _sample_char_node(node_id: str = "dsi:10") -> dict:
    return {
        "id": node_id,
        "parent_id": None,
        "level": 1,
        "title": "Section 1",
        "order": 0,
        "path_label": "1",
        "target_type": "char_range",
        "target_start": 0,
        "target_end": 26,
        "target_href": None,
        "source": "document_structure_index",
        "confidence": 0.95,
    }


def test_generated_heading_parent_contains_child_sections():
    content = "# Chapter\nIntroduction\n## Child\nEvidence\n# Next\nOther"
    nodes = navigation_core._extract_generated_heading_nodes(content)
    assert nodes[0]["target_end"] == content.index("# Next")


@pytest.mark.asyncio
@pytest.mark.parametrize("mapped", [True, False])
async def test_pdf_section_content_reports_exact_or_unavailable(mock_user, mock_db, mapped):
    content = "Short\n\nA much longer chapter body\n\nLast"
    start, end = content.index("A much"), content.index("Last")
    page_map = [
        {"page_number": 1, "start_char": 0, "end_char": 5},
        {"page_number": 2, "start_char": start, "end_char": end - 2},
        {"page_number": 3, "start_char": end, "end_char": len(content)},
    ]
    mock_db.get_media_by_id.return_value = {
        "id": 1,
        "type": "pdf",
        "title": "Book",
        "version": 1,
        "content": content,
        "safe_metadata": {
            "page_map": page_map, "page_map_content_hash": hashlib.sha256(content.encode()).hexdigest(),
        } if mapped else {},
    }
    nodes = [
        {
            **_sample_char_node(f"pdf_outline:{page}"),
            "target_type": "page",
            "target_start": page,
            "target_end": None,
            "source": "pdf_outline",
        }
        for page in (2, 3)
    ]
    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db
    with (
        patch.object(navigation_mod, "_select_source_nodes", return_value=(nodes, ["pdf_outline"])),
        patch.object(navigation_mod, "_get_media_text", return_value=content),
        patch.object(navigation_mod, "get_cached_response", return_value=None),
        patch.object(navigation_mod, "cache_response", return_value="etag"),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/navigation/pdf_outline:2/content")
    assert response.status_code == 200
    result = response.json()
    assert result["content"] == (content[start:end] if mapped else "")
    assert result["coverage_status"] == ("exact" if mapped else "unavailable")
    assert result["resolved_start_char"] == (start if mapped else None)
    assert result["navigation_version"].startswith("media_1:v1:")


@pytest.mark.asyncio
@pytest.mark.parametrize("content,intrinsic_format", [
    ("## Section 1\nBody line here", "markdown"),
    ("## Trade\n|Thailand<br>Taiwan|\n|Exports &amp; imports|", "html"),
    ("Exact plain source.", "plain"),
])
async def test_navigation_content_auto_preserves_intrinsic_source_and_offsets(mock_user, mock_db, content, intrinsic_format):
    mock_db.get_media_by_id.return_value = {
        "id": 1,
        "type": "document",
        "title": "Doc",
        "version": 3,
        "last_modified": "2026-02-09T15:00:00Z",
    }
    sample_nodes = [_sample_char_node()]
    sample_nodes[0]["target_end"] = len(content)

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with (
        patch.object(
            navigation_mod,
            "_select_source_nodes",
            return_value=(sample_nodes, ["pdf_outline", "document_structure_index"]),
        ),
        patch.object(
            navigation_core,
            "get_document_version",
            return_value={"content": content},
        ),
        patch.object(
            navigation_mod,
            "get_cached_response",
            return_value=None,
        ),
        patch.object(
            navigation_mod,
            "cache_response",
            return_value="etag",
        ),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/navigation/dsi:10/content?format=auto")

    assert response.status_code == 200
    payload = response.json()
    assert payload["content_format"] == intrinsic_format
    assert payload["content"] == content
    assert payload["resolved_end_char"] - payload["resolved_start_char"] == len(content)
    assert intrinsic_format in payload["available_formats"]
    assert payload["alternate_content"] is None


@pytest.mark.asyncio
async def test_navigation_content_html_request_with_alternates(mock_user, mock_db):
    mock_db.get_media_by_id.return_value = {
        "id": 1,
        "type": "document",
        "title": "Doc",
        "version": 4,
        "last_modified": "2026-02-09T16:00:00Z",
    }
    sample_nodes = [_sample_char_node()]

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with (
        patch.object(
            navigation_mod,
            "_select_source_nodes",
            return_value=(sample_nodes, ["pdf_outline", "document_structure_index"]),
        ),
        patch.object(
            navigation_core,
            "get_document_version",
            return_value={"content": "Plain section content only"},
        ),
        patch.object(
            navigation_mod,
            "get_cached_response",
            return_value=None,
        ),
        patch.object(
            navigation_mod,
            "cache_response",
            return_value="etag",
        ),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/api/v1/media/1/navigation/dsi:10/content?format=html&include_alternates=true",
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["content_format"] == "html"
    assert payload["content"].startswith("<p>")
    assert payload["available_formats"] == ["plain"]
    assert payload["alternate_content"] == {"plain": "Plain section content only"}


@pytest.mark.asyncio
async def test_navigation_content_returns_contract_404_for_unknown_node(mock_user, mock_db):
    mock_db.get_media_by_id.return_value = {
        "id": 1,
        "type": "document",
        "title": "Doc",
        "version": 5,
        "last_modified": "2026-02-09T17:00:00Z",
    }
    sample_nodes = [_sample_char_node("dsi:known")]

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with (
        patch.object(
            navigation_mod,
            "_select_source_nodes",
            return_value=(sample_nodes, ["pdf_outline", "document_structure_index"]),
        ),
        patch.object(
            navigation_mod,
            "get_cached_response",
            return_value=None,
        ),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/navigation/dsi:missing/content")

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["error_code"] == "NAVIGATION_NODE_NOT_FOUND"
    assert detail["media_id"] == 1
    assert detail["node_id"] == "dsi:missing"
    assert detail["navigation_version"].startswith("media_1:v5:")


@pytest.mark.asyncio
async def test_navigation_content_does_not_expand_unmapped_segment_to_full_transcript(mock_user, mock_db):
    mock_db.get_media_by_id.return_value = {
        "id": 1,
        "type": "video",
        "title": "Video",
        "version": 2,
        "last_modified": "2026-02-09T18:00:00Z",
    }
    sample_nodes = [
        {
            "id": "t:1",
            "parent_id": None,
            "level": 1,
            "title": "Intro Segment",
            "order": 0,
            "path_label": "1",
            "target_type": "time_range",
            "target_start": 5.0,
            "target_end": 10.0,
            "target_href": None,
            "source": "transcript_segment",
            "confidence": 0.8,
        }
    ]

    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    with (
        patch.object(
            navigation_mod,
            "_select_source_nodes",
            return_value=(sample_nodes, ["pdf_outline", "document_structure_index", "chunk_metadata"]),
        ),
        patch.object(
            navigation_core,
            "get_document_version",
            return_value=None,
        ),
        patch.object(
            navigation_core,
            "get_latest_transcription",
            return_value="Transcript fallback content",
        ),
        patch.object(
            navigation_mod,
            "get_cached_response",
            return_value=None,
        ),
        patch.object(
            navigation_mod,
            "cache_response",
            return_value="etag",
        ),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/media/1/navigation/t:1/content?format=plain")

    assert response.status_code == 200
    payload = response.json()
    assert payload["content_format"] == "plain"
    assert payload["content"] == ""
    assert payload["coverage_status"] == "unavailable"
    assert payload["target"]["target_type"] == "time_range"
