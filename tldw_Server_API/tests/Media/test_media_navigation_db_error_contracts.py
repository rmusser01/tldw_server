from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.media import navigation as navigation_mod
from tldw_Server_API.app.api.v1.schemas.media_navigation_schemas import (
    MediaNavigationContentQueryParams,
    MediaNavigationNode,
    MediaNavigationQueryParams,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError


class _DatabaseErrorDb:
    def lookup_section_by_heading(self, media_id: int, heading: str):
        raise DatabaseError("lookup failed")


class _ProgrammerErrorDb:
    def lookup_section_by_heading(self, media_id: int, heading: str):
        raise ValueError("bad lookup state")


class _GetMediaByIdErrorDb:
    def get_media_by_id(self, media_id: int, *, include_deleted: bool, include_trash: bool):
        raise DatabaseError("navigation database exploded")


class _GetMediaByIdDb:
    def get_media_by_id(self, media_id: int, *, include_deleted: bool, include_trash: bool):
        return {
            "id": media_id,
            "type": "document",
            "title": "Doc",
            "content": "Navigation content from media fallback.",
            "version": 1,
            "last_modified": "2026-02-09T10:00:00Z",
        }


class _ExecuteQueryErrorDb:
    backend_type = "sqlite"

    def execute_query(self, query: str, params: tuple[object, ...]):
        raise DatabaseError("navigation source database exploded")


class _GetMediaFileErrorDb:
    def get_media_file(self, media_id: int, file_kind: str):
        raise DatabaseError("navigation file lookup exploded")


class _PdfFileDb:
    def get_media_file(self, media_id: int, file_kind: str):
        return {
            "storage_path": "user_1/media/7/original.pdf",
            "mime_type": "application/pdf",
        }


class _PdfStorageErrorBackend:
    async def exists(self, storage_path: str) -> bool:
        raise navigation_mod.StorageError("navigation storage exploded")


class _PdfStorageBackend:
    async def exists(self, storage_path: str) -> bool:
        return True

    async def get_size(self, storage_path: str) -> int:
        return 128

    async def retrieve(self, storage_path: str):
        return object()


class _LargePdfStorageBackend:
    async def exists(self, storage_path: str) -> bool:
        return True

    async def get_size(self, storage_path: str) -> int:
        return navigation_mod.MAX_OUTLINE_FILE_SIZE + 1


def test_derive_content_span_swallows_typed_database_errors(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)

    node = MediaNavigationNode(
        id="node-1",
        parent_id=None,
        level=1,
        title="Results",
        order=0,
        path_label="1",
        target_type="href",
        target_href="#results",
        source="generated_toc",
        confidence=0.5,
    )

    result = navigation_mod._derive_content_span(
        node=node,
        all_nodes=[node],
        media={"type": "document"},
        db=_DatabaseErrorDb(),
        media_id=7,
        content_length=500,
    )

    if result is not None:
        pytest.fail(f"expected None fallback for DatabaseError, got {result!r}")
    fake_logger.debug.assert_called_once_with("Section heading lookup failed")


def test_derive_content_span_does_not_hide_non_database_failures() -> None:
    node = MediaNavigationNode(
        id="node-1",
        parent_id=None,
        level=1,
        title="Results",
        order=0,
        path_label="1",
        target_type="href",
        target_href="#results",
        source="generated_toc",
        confidence=0.5,
    )

    with pytest.raises(ValueError):
        navigation_mod._derive_content_span(
            node=node,
            all_nodes=[node],
            media={"type": "document"},
            db=_ProgrammerErrorDb(),
            media_id=7,
            content_length=500,
        )


@pytest.mark.asyncio
async def test_get_media_navigation_sanitizes_db_fetch_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)

    with pytest.raises(HTTPException) as exc_info:
        await navigation_mod.get_media_navigation(
            media_id=7,
            params=MediaNavigationQueryParams(),
            db=_GetMediaByIdErrorDb(),
            current_user=SimpleNamespace(id=1),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error while fetching media item"
    fake_logger.error.assert_called_once_with("Database error fetching media for navigation")


@pytest.mark.asyncio
async def test_get_media_navigation_content_sanitizes_db_fetch_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)

    with pytest.raises(HTTPException) as exc_info:
        await navigation_mod.get_media_navigation_content(
            media_id=7,
            node_id="dsi:10",
            params=MediaNavigationContentQueryParams(),
            db=_GetMediaByIdErrorDb(),
            current_user=SimpleNamespace(id=1),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error while fetching media item"
    fake_logger.error.assert_called_once_with("Database error fetching media for navigation content")


@pytest.mark.asyncio
async def test_get_media_navigation_sanitizes_invalid_cached_payload_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)
    monkeypatch.setattr(navigation_mod, "get_cached_response", lambda _cache_key: ("etag", {}))
    monkeypatch.setattr(navigation_mod, "cache_response", lambda *_args, **_kwargs: "etag")

    async def _empty_source_nodes(**_kwargs):
        return [], []

    monkeypatch.setattr(navigation_mod, "_select_source_nodes", _empty_source_nodes)

    response = await navigation_mod.get_media_navigation(
        media_id=7,
        params=MediaNavigationQueryParams(),
        db=_GetMediaByIdDb(),
        current_user=SimpleNamespace(id=1),
    )

    assert response.available is False
    fake_logger.debug.assert_any_call("Ignoring invalid cached navigation payload")


@pytest.mark.asyncio
async def test_get_media_navigation_content_sanitizes_invalid_cached_payload_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)
    monkeypatch.setattr(navigation_mod, "get_cached_response", lambda _cache_key: ("etag", {}))
    monkeypatch.setattr(navigation_mod, "cache_response", lambda *_args, **_kwargs: "etag")

    async def _source_nodes(**_kwargs):
        return [
            {
                "id": "dsi:10",
                "parent_id": None,
                "level": 1,
                "title": "Section",
                "order": 0,
                "path_label": "1",
                "target_type": "char_range",
                "target_start": 0,
                "target_end": 18,
                "target_href": None,
                "source": "document_structure_index",
                "confidence": 0.95,
            }
        ], ["document_structure_index"]

    monkeypatch.setattr(navigation_mod, "_select_source_nodes", _source_nodes)

    response = await navigation_mod.get_media_navigation_content(
        media_id=7,
        node_id="dsi:10",
        params=MediaNavigationContentQueryParams(),
        db=_GetMediaByIdDb(),
        current_user=SimpleNamespace(id=1),
    )

    assert response.content.startswith("Navigation content")
    fake_logger.debug.assert_any_call("Ignoring invalid cached navigation content payload")


def test_extract_document_structure_nodes_sanitizes_query_failure_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)

    result = navigation_mod._extract_document_structure_nodes(7, _ExecuteQueryErrorDb())

    assert result == []
    fake_logger.warning.assert_called_once_with("Navigation source document_structure_index query failed")


def test_extract_transcript_segment_nodes_sanitizes_query_failure_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)

    def _raise_transcript_error(db, media_id: int):
        raise DatabaseError("transcript navigation source exploded")

    monkeypatch.setattr(navigation_mod, "get_media_transcripts", _raise_transcript_error)

    result = navigation_mod._extract_transcript_segment_nodes(7, object(), {"type": "video"})

    assert result == []
    fake_logger.warning.assert_called_once_with("Navigation source transcript_segment query failed")


def test_extract_chunk_metadata_nodes_sanitizes_query_failure_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)

    result = navigation_mod._extract_chunk_metadata_nodes(7, _ExecuteQueryErrorDb())

    assert result == []
    fake_logger.warning.assert_called_once_with("Navigation source chunk_metadata query failed")


def test_materialize_navigation_nodes_sanitizes_invalid_payload_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)

    result = navigation_mod._materialize_navigation_nodes([{}])

    assert result == []
    fake_logger.debug.assert_called_once_with("Skipping invalid navigation node payload")


@pytest.mark.asyncio
async def test_select_source_nodes_sanitizes_empty_generated_fallback_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)

    async def _empty_pdf_outline(*_args, **_kwargs):
        return []

    monkeypatch.setattr(navigation_mod, "_extract_pdf_outline_nodes", _empty_pdf_outline)
    monkeypatch.setattr(navigation_mod, "_extract_generated_toc_nodes", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(navigation_mod, "_extract_document_structure_nodes", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(navigation_mod, "_extract_transcript_segment_nodes", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(navigation_mod, "_extract_chunk_metadata_nodes", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(navigation_mod, "_extract_generated_fallback_nodes", lambda *_args, **_kwargs: [])

    nodes, sources = await navigation_mod._select_source_nodes(
        media_id=7,
        db=object(),
        media={"type": "document"},
        include_generated_fallback=True,
    )

    assert nodes == []
    assert sources[-1] == "generated"
    fake_logger.debug.assert_any_call("Generated fallback requested but produced no nodes")


@pytest.mark.asyncio
async def test_select_source_nodes_sanitizes_sparse_pdf_fallback_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)

    sparse_outline = [
        {
            "id": "pdf_outline:0",
            "parent_id": None,
            "level": 1,
            "title": "Title Page",
            "order": 0,
            "path_label": "1",
            "target_type": "page",
            "target_start": 1,
            "target_end": None,
            "target_href": None,
            "source": "pdf_outline",
            "confidence": 1.0,
        }
    ]

    async def _sparse_pdf_outline(*_args, **_kwargs):
        return sparse_outline

    monkeypatch.setattr(navigation_mod, "_extract_pdf_outline_nodes", _sparse_pdf_outline)
    monkeypatch.setattr(navigation_mod, "_extract_generated_toc_nodes", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(navigation_mod, "_extract_document_structure_nodes", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(navigation_mod, "_extract_transcript_segment_nodes", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(navigation_mod, "_extract_chunk_metadata_nodes", lambda *_args, **_kwargs: [])

    nodes, sources = await navigation_mod._select_source_nodes(
        media_id=7,
        db=object(),
        media={"type": "pdf"},
        include_generated_fallback=False,
    )

    assert nodes == sparse_outline
    assert sources == list(navigation_mod.NAVIGATION_SOURCE_PRIORITY)
    fake_logger.debug.assert_any_call("Navigation source pdf_outline produced sparse structure; trying fallback sources")


def test_get_media_text_sanitizes_document_version_failure_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)

    def _raise_document_version_error(*_args, **_kwargs):
        raise DatabaseError("document text backend exploded")

    monkeypatch.setattr(navigation_mod, "get_document_version", _raise_document_version_error)

    result = navigation_mod._get_media_text(7, {"type": "document", "content": "fallback text"}, object())

    assert result == "fallback text"
    fake_logger.debug.assert_called_once_with("Failed to fetch latest document version")


def test_get_media_text_sanitizes_latest_transcript_failure_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)
    monkeypatch.setattr(navigation_mod, "get_document_version", lambda *_args, **_kwargs: None)

    def _raise_transcript_error(*_args, **_kwargs):
        raise DatabaseError("latest transcript backend exploded")

    monkeypatch.setattr(navigation_mod, "get_latest_transcription", _raise_transcript_error)

    result = navigation_mod._get_media_text(7, {"type": "video", "content": "video fallback"}, object())

    assert result == "video fallback"
    fake_logger.debug.assert_called_once_with("Failed to fetch latest transcript")


@pytest.mark.asyncio
async def test_extract_pdf_outline_nodes_sanitizes_file_record_failure_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)
    monkeypatch.setattr(navigation_mod, "_check_pymupdf_available", lambda: True)

    result = await navigation_mod._extract_pdf_outline_nodes(
        media_id=7,
        db=_GetMediaFileErrorDb(),
        media={"type": "pdf"},
    )

    assert result == []
    fake_logger.warning.assert_called_once_with("Navigation source pdf_outline failed to fetch file record")


@pytest.mark.asyncio
async def test_extract_pdf_outline_nodes_sanitizes_large_file_skip_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)
    monkeypatch.setattr(navigation_mod, "_check_pymupdf_available", lambda: True)
    monkeypatch.setattr(navigation_mod, "get_storage_backend", lambda: _LargePdfStorageBackend())

    result = await navigation_mod._extract_pdf_outline_nodes(
        media_id=7,
        db=_PdfFileDb(),
        media={"type": "pdf"},
    )

    assert result == []
    fake_logger.debug.assert_called_once_with("Navigation source pdf_outline skipped due to file size")


@pytest.mark.asyncio
async def test_extract_pdf_outline_nodes_sanitizes_storage_failure_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)
    monkeypatch.setattr(navigation_mod, "_check_pymupdf_available", lambda: True)
    monkeypatch.setattr(navigation_mod, "get_storage_backend", lambda: _PdfStorageErrorBackend())

    result = await navigation_mod._extract_pdf_outline_nodes(
        media_id=7,
        db=_PdfFileDb(),
        media={"type": "pdf"},
    )

    assert result == []
    fake_logger.warning.assert_called_once_with("Navigation source pdf_outline storage access failed")


@pytest.mark.asyncio
async def test_extract_pdf_outline_nodes_sanitizes_extraction_failure_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(navigation_mod, "logger", fake_logger)
    monkeypatch.setattr(navigation_mod, "_check_pymupdf_available", lambda: True)
    monkeypatch.setattr(navigation_mod, "get_storage_backend", lambda: _PdfStorageBackend())

    def _raise_pdf_outline_error(_pdf_file):
        raise RuntimeError("navigation pdf parser exploded")

    monkeypatch.setattr(navigation_mod, "_extract_pdf_outline", _raise_pdf_outline_error)

    result = await navigation_mod._extract_pdf_outline_nodes(
        media_id=7,
        db=_PdfFileDb(),
        media={"type": "pdf"},
    )

    assert result == []
    fake_logger.warning.assert_called_once_with("Navigation source pdf_outline extraction failed")
