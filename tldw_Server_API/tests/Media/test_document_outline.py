# Tests for Document Outline/TOC Extraction Endpoint
#
from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from tldw_Server_API.app.api.v1.endpoints.media import document_outline as outline_endpoint
from tldw_Server_API.app.main import app
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


# Sample PDF bytes (minimal valid PDF header for testing)
# This is NOT a real PDF but enough for mocking purposes
FAKE_PDF_BYTES = b"%PDF-1.4\nfake pdf content"


class _LoggerStub:
    def __init__(self):
        self.debug_calls = []
        self.error_calls = []
        self.warning_calls = []
        self.exception_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))


_OUTLINE_HELPER_SENSITIVE_MARKERS = (
    "outline helper leaked",
    "/private/tmp/pdf-outline",
)


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    user = MagicMock()
    user.id = 1
    user.username = "testuser"
    return user


@pytest.fixture
def mock_db():
    """Create a mock MediaDatabase."""
    db = MagicMock()
    db.get_media_by_id = MagicMock()
    db.get_media_file = MagicMock()
    return db


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = AsyncMock()
    storage.exists = AsyncMock(return_value=True)
    storage.get_size = AsyncMock(return_value=1024)  # 1KB
    storage.retrieve = AsyncMock(return_value=io.BytesIO(FAKE_PDF_BYTES))
    return storage


class TestDocumentOutlineEndpoint:
    """Tests for the document outline extraction endpoint."""

    @pytest.mark.asyncio
    async def test_valid_pdf_with_outline(self, mock_user, mock_db, mock_storage):
        """Test extracting outline from a valid PDF with TOC."""
        # Setup mock media and file records
        mock_db.get_media_by_id.return_value = {"id": 1, "type": "pdf", "title": "Test PDF"}
        mock_db.get_media_file.return_value = {
            "storage_path": "user_1/media/1/original.pdf",
            "mime_type": "application/pdf",
        }

        # Mock PyMuPDF extraction
        mock_outline_entries = [
            MagicMock(level=1, title="Chapter 1", page=1),
            MagicMock(level=2, title="Section 1.1", page=3),
            MagicMock(level=1, title="Chapter 2", page=10),
        ]

        app.dependency_overrides[get_request_user] = lambda: mock_user
        app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

        with patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline.get_storage_backend",
            return_value=mock_storage,
        ), patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline._check_pymupdf_available",
            return_value=True,
        ), patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline._extract_pdf_outline",
            return_value=(
                [
                    MagicMock(level=1, title="Chapter 1", page=1),
                    MagicMock(level=2, title="Section 1.1", page=3),
                ],
                50,
            ),
        ) as mock_extract:
            # Make the mock return proper OutlineEntry objects
            from tldw_Server_API.app.api.v1.schemas.document_outline import OutlineEntry
            mock_extract.return_value = (
                [
                    OutlineEntry(level=1, title="Chapter 1", page=1),
                    OutlineEntry(level=2, title="Section 1.1", page=3),
                ],
                50,
            )

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/api/v1/media/1/outline")

            assert response.status_code == 200
            data = response.json()
            assert data["media_id"] == 1
            assert data["has_outline"] is True
            assert data["total_pages"] == 50
            assert len(data["entries"]) == 2
            assert data["entries"][0]["title"] == "Chapter 1"
            assert data["entries"][0]["level"] == 1
            assert data["entries"][0]["page"] == 1

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_valid_pdf_without_outline(self, mock_user, mock_db, mock_storage):
        """Test extracting outline from a PDF without TOC."""
        mock_db.get_media_by_id.return_value = {"id": 1, "type": "pdf", "title": "Test PDF"}
        mock_db.get_media_file.return_value = {
            "storage_path": "user_1/media/1/original.pdf",
            "mime_type": "application/pdf",
        }

        app.dependency_overrides[get_request_user] = lambda: mock_user
        app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

        with patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline.get_storage_backend",
            return_value=mock_storage,
        ), patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline._check_pymupdf_available",
            return_value=True,
        ), patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline._extract_pdf_outline",
            return_value=([], 25),  # Empty outline, 25 pages
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/api/v1/media/1/outline")

            assert response.status_code == 200
            data = response.json()
            assert data["media_id"] == 1
            assert data["has_outline"] is False
            assert data["total_pages"] == 25
            assert len(data["entries"]) == 0

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_media_not_found(self, mock_user, mock_db):
        """Test 404 response when media_id does not exist."""
        mock_db.get_media_by_id.return_value = None

        app.dependency_overrides[get_request_user] = lambda: mock_user
        app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

        with patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline._check_pymupdf_available",
            return_value=True,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/api/v1/media/9999/outline")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_non_pdf_media_type(self, mock_user, mock_db):
        """Test empty outline response for non-PDF media types."""
        mock_db.get_media_by_id.return_value = {"id": 1, "type": "video", "title": "Test Video"}

        app.dependency_overrides[get_request_user] = lambda: mock_user
        app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

        with patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline._check_pymupdf_available",
            return_value=True,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/api/v1/media/1/outline")

            assert response.status_code == 200
            data = response.json()
            assert data["has_outline"] is False
            assert data["entries"] == []
            assert data["total_pages"] == 0

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_missing_file_in_storage(self, mock_user, mock_db, mock_storage):
        """Test empty outline response when file is missing from storage."""
        mock_db.get_media_by_id.return_value = {"id": 1, "type": "pdf", "title": "Test PDF"}
        mock_db.get_media_file.return_value = {
            "storage_path": "user_1/media/1/original.pdf",
            "mime_type": "application/pdf",
        }
        mock_storage.exists.return_value = False

        app.dependency_overrides[get_request_user] = lambda: mock_user
        app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

        with patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline.get_storage_backend",
            return_value=mock_storage,
        ), patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline._check_pymupdf_available",
            return_value=True,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/api/v1/media/1/outline")

            assert response.status_code == 200
            data = response.json()
            assert data["has_outline"] is False
            assert data["entries"] == []

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_file_too_large(self, mock_user, mock_db, mock_storage):
        """Test 413 response when file exceeds size limit."""
        mock_db.get_media_by_id.return_value = {"id": 1, "type": "pdf", "title": "Large PDF"}
        mock_db.get_media_file.return_value = {
            "storage_path": "user_1/media/1/original.pdf",
            "mime_type": "application/pdf",
        }
        # Set file size to exceed the 500MB limit
        mock_storage.get_size.return_value = 600 * 1024 * 1024  # 600MB

        app.dependency_overrides[get_request_user] = lambda: mock_user
        app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

        with patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline.get_storage_backend",
            return_value=mock_storage,
        ), patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline._check_pymupdf_available",
            return_value=True,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/api/v1/media/1/outline")

            assert response.status_code == 413
            assert "too large" in response.json()["detail"].lower()

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_pymupdf_not_installed(self, mock_user, mock_db):
        """Test 501 response when PyMuPDF is not available."""
        mock_db.get_media_by_id.return_value = {"id": 1, "type": "pdf", "title": "Test PDF"}

        app.dependency_overrides[get_request_user] = lambda: mock_user
        app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

        with patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline._check_pymupdf_available",
            return_value=False,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/api/v1/media/1/outline")

            assert response.status_code == 501
            assert "not installed" in response.json()["detail"].lower()

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_invalid_media_id_format(self):
        """Test 422 response for invalid media_id format (non-integer)."""
        # No dependency overrides needed - validation happens before endpoint
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/media/not-a-number/outline")

        # FastAPI returns 404 for path parameters that don't match the expected type
        # when using {media_id:int} path converter
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_database_error_handling(self, mock_user, mock_db):
        """Test 500 response when database error occurs."""
        mock_db.get_media_by_id.side_effect = Exception("Database connection failed")

        app.dependency_overrides[get_request_user] = lambda: mock_user
        app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

        with patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline._check_pymupdf_available",
            return_value=True,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/api/v1/media/1/outline")

            assert response.status_code == 500
            assert "database" in response.json()["detail"].lower()

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_non_pdf_mime_type_returns_empty(self, mock_user, mock_db, mock_storage):
        """Test empty outline when file has non-PDF MIME type and extension."""
        mock_db.get_media_by_id.return_value = {"id": 1, "type": "document", "title": "Word Doc"}
        mock_db.get_media_file.return_value = {
            "storage_path": "user_1/media/1/original.docx",
            "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }

        app.dependency_overrides[get_request_user] = lambda: mock_user
        app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

        with patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline.get_storage_backend",
            return_value=mock_storage,
        ), patch(
            "tldw_Server_API.app.api.v1.endpoints.media.document_outline._check_pymupdf_available",
            return_value=True,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/api/v1/media/1/outline")

            assert response.status_code == 200
            data = response.json()
            assert data["has_outline"] is False
            assert data["entries"] == []

        app.dependency_overrides.clear()


class TestExtractPdfOutline:
    """Unit tests for the _extract_pdf_outline function."""

    def test_extract_filters_empty_titles(self):
        """Test that entries with empty titles are filtered out."""
        from tldw_Server_API.app.api.v1.endpoints.media.document_outline import (
            _extract_pdf_outline,
        )

        # Create a mock PDF document with PyMuPDF
        with patch("pymupdf.open") as mock_open:
            mock_doc = MagicMock()
            mock_doc.page_count = 10
            mock_doc.get_toc.return_value = [
                [1, "Valid Title", 1],
                [1, "", 2],  # Empty title - should be filtered
                [2, "   ", 3],  # Whitespace only - should be filtered
                [1, "Another Valid", 5],
            ]
            mock_doc.close = MagicMock()
            mock_open.return_value = mock_doc

            entries, total_pages = _extract_pdf_outline(FAKE_PDF_BYTES)

            assert total_pages == 10
            assert len(entries) == 2
            assert entries[0].title == "Valid Title"
            assert entries[1].title == "Another Valid"

    def test_extract_handles_binaryio(self):
        """Test that the function handles BinaryIO input."""
        from tldw_Server_API.app.api.v1.endpoints.media.document_outline import (
            _extract_pdf_outline,
        )

        pdf_stream = io.BytesIO(FAKE_PDF_BYTES)

        with patch("pymupdf.open") as mock_open:
            mock_doc = MagicMock()
            mock_doc.page_count = 5
            mock_doc.get_toc.return_value = [[1, "Test", 1]]
            mock_doc.close = MagicMock()
            mock_open.return_value = mock_doc

            entries, total_pages = _extract_pdf_outline(pdf_stream)

            assert total_pages == 5
            # Verify pymupdf.open was called with the bytes read from the stream
            mock_open.assert_called_once()

    def test_extract_clamps_level_to_valid_range(self):
        """Test that outline levels are clamped to 1-6 range."""
        from tldw_Server_API.app.api.v1.endpoints.media.document_outline import (
            _extract_pdf_outline,
        )

        with patch("pymupdf.open") as mock_open:
            mock_doc = MagicMock()
            mock_doc.page_count = 10
            mock_doc.get_toc.return_value = [
                [0, "Level 0", 1],  # Should become 1
                [7, "Level 7", 2],  # Should become 6
                [10, "Level 10", 3],  # Should become 6
                [-1, "Negative", 4],  # Should become 1
            ]
            mock_doc.close = MagicMock()
            mock_open.return_value = mock_doc

            entries, _ = _extract_pdf_outline(FAKE_PDF_BYTES)

            assert len(entries) == 4
            assert entries[0].level == 1
            assert entries[1].level == 6
            assert entries[2].level == 6
            assert entries[3].level == 1

    def test_extract_clamps_page_to_valid_range(self):
        """Test that page numbers are clamped to valid range."""
        from tldw_Server_API.app.api.v1.endpoints.media.document_outline import (
            _extract_pdf_outline,
        )

        with patch("pymupdf.open") as mock_open:
            mock_doc = MagicMock()
            mock_doc.page_count = 10
            mock_doc.get_toc.return_value = [
                [1, "Page 0", 0],  # Should become 1
                [1, "Page 100", 100],  # Should become 10 (max pages)
                [1, "Negative", -5],  # Should become 1
            ]
            mock_doc.close = MagicMock()
            mock_open.return_value = mock_doc

            entries, _ = _extract_pdf_outline(FAKE_PDF_BYTES)

            assert len(entries) == 3
            assert entries[0].page == 1
            assert entries[1].page == 10
            assert entries[2].page == 1

    def test_extract_sanitizes_open_failure_log(self, monkeypatch):
        logger_stub = _LoggerStub()
        monkeypatch.setattr(outline_endpoint, "logger", logger_stub, raising=True)

        with patch(
            "pymupdf.open",
            side_effect=RuntimeError("outline helper leaked /private/tmp/pdf-outline"),
        ):
            entries, total_pages = outline_endpoint._extract_pdf_outline(FAKE_PDF_BYTES)

        assert entries == []
        assert total_pages == 0
        assert [args[0] for args, _kwargs in logger_stub.error_calls if args] == [
            "Error extracting PDF outline"
        ]

        rendered_calls = repr(logger_stub.error_calls)
        for marker in _OUTLINE_HELPER_SENSITIVE_MARKERS:
            assert marker not in rendered_calls
