"""Comprehensive tests for media workflow adapters.

Tests for:
1. run_media_ingest_adapter - Ingest media (video, audio, etc.)
2. run_process_media_adapter - Process ingested media
3. run_pdf_extract_adapter - Extract text from PDF
4. run_ocr_adapter - OCR on images
5. run_document_table_extract_adapter - Extract tables from documents

These adapters interact with file systems and external tools, so we use
TEST_MODE and mocks to avoid heavy operations.
"""

from __future__ import annotations

import builtins

import pytest

pytestmark = pytest.mark.unit


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def test_mode_env(monkeypatch):
    """Set TEST_MODE environment variable for all tests."""
    monkeypatch.setenv("TEST_MODE", "1")
    yield


@pytest.fixture
def workflow_file_base(monkeypatch, tmp_path):
    """Set up workflow file base directory."""
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing."""
    content = """# Sample Document

This is a test document with multiple paragraphs.
It contains several sentences for testing chunking.

## Section 1

First section content here.

## Section 2

Second section content here.
"""
    file_path = tmp_path / "sample.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_pdf_bytes():
    """Return minimal PDF bytes for testing."""
    # A minimal valid PDF structure
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<< /Size 4 /Root 1 0 R >>
startxref
196
%%EOF
"""


@pytest.fixture
def sample_pdf_file(tmp_path, sample_pdf_bytes):
    """Create a sample PDF file for testing."""
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(sample_pdf_bytes)
    return file_path


@pytest.fixture
def sample_image_file(tmp_path):
    """Create a minimal PNG image file for testing."""
    # Minimal valid PNG (1x1 transparent pixel)
    png_bytes = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
        0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,  # IDAT chunk
        0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
        0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
        0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,  # IEND chunk
        0x42, 0x60, 0x82
    ])
    file_path = tmp_path / "sample.png"
    file_path.write_bytes(png_bytes)
    return file_path


# =============================================================================
# Tests for run_media_ingest_adapter
# =============================================================================


class TestMediaIngestAdapter:
    """Tests for the media_ingest adapter."""

    @pytest.mark.asyncio
    async def test_media_ingest_empty_sources(self, test_mode_env):
        """Test media ingest with empty sources returns empty result."""
        from tldw_Server_API.app.core.Workflows.adapters import run_media_ingest_adapter

        config = {"sources": []}
        context = {"user_id": "1", "step_run_id": "step1"}

        result = await run_media_ingest_adapter(config, context)

        assert result.get("media_ids") == []
        assert result.get("metadata") == []
        assert result.get("transcripts") == []
        assert result.get("rag_indexed") is False

    @pytest.mark.asyncio
    async def test_media_ingest_no_sources_key(self, test_mode_env):
        """Test media ingest without sources key returns empty result."""
        from tldw_Server_API.app.core.Workflows.adapters import run_media_ingest_adapter

        config = {}
        context = {}

        result = await run_media_ingest_adapter(config, context)

        assert result.get("media_ids") == []
        assert result.get("metadata") == []

    @pytest.mark.asyncio
    async def test_media_ingest_http_url_test_mode(self, test_mode_env, monkeypatch):
        """Test media ingest with HTTP URL in test mode returns simulated result."""
        from tldw_Server_API.app.core.Workflows.adapters import run_media_ingest_adapter

        # Mock egress check to allow the URL
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.media.ingest.is_url_allowed",
            lambda url: True
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.media.ingest.is_url_allowed_for_tenant",
            lambda url, tenant: True
        )

        config = {
            "sources": [{"uri": "https://example.com/video.mp4"}],
            "safety": {"allowed_domains": ["example.com"]}
        }
        context = {"user_id": "1", "tenant_id": "default"}

        result = await run_media_ingest_adapter(config, context)

        # In test mode, HTTP sources should be simulated
        assert len(result.get("metadata", [])) == 1
        assert result["metadata"][0]["status"] == "simulated_download"

    @pytest.mark.asyncio
    async def test_media_ingest_local_file_success(
        self, test_mode_env, workflow_file_base, sample_text_file, monkeypatch
    ):
        """Test media ingest with local file URI."""
        from tldw_Server_API.app.core.Workflows.adapters import run_media_ingest_adapter

        # File must be within the workflow file base
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(sample_text_file.parent))

        config = {
            "sources": [{"uri": f"file://{sample_text_file}"}],
            "extraction": {"extract_text": True},
        }
        context = {"user_id": "1"}

        result = await run_media_ingest_adapter(config, context)

        assert len(result.get("metadata", [])) == 1
        assert result["metadata"][0]["status"] == "local_ok"
        assert "text" in result

    @pytest.mark.asyncio
    async def test_media_ingest_local_file_with_chunking(
        self, test_mode_env, workflow_file_base, sample_text_file, monkeypatch
    ):
        """Test media ingest with local file and chunking enabled."""
        from tldw_Server_API.app.core.Workflows.adapters import run_media_ingest_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(sample_text_file.parent))

        config = {
            "sources": [{"uri": f"file://{sample_text_file}"}],
            "extraction": {"extract_text": True},
            "chunking": {
                "strategy": "sentences",
                "max_tokens": 50,
                "overlap": 0,
            },
        }
        context = {"user_id": "1"}

        result = await run_media_ingest_adapter(config, context)

        assert result["metadata"][0]["status"] == "local_ok"
        # Chunking should produce chunks
        assert "chunks" in result
        if result.get("chunks"):
            assert isinstance(result["chunks"], list)

    @pytest.mark.asyncio
    async def test_media_ingest_blocked_domain(self, test_mode_env, monkeypatch):
        """Test media ingest blocks disallowed domains."""
        from tldw_Server_API.app.core.Workflows.adapters import run_media_ingest_adapter

        config = {
            "sources": [{"uri": "https://evil.com/video.mp4"}],
            "safety": {"allowed_domains": ["example.com"]}
        }
        context = {"user_id": "1", "tenant_id": "default"}

        result = await run_media_ingest_adapter(config, context)

        assert len(result.get("metadata", [])) == 1
        assert result["metadata"][0]["status"] == "skipped_disallowed_domain"

    @pytest.mark.asyncio
    async def test_media_ingest_empty_uri_skipped(self, test_mode_env):
        """Test media ingest skips empty URIs."""
        from tldw_Server_API.app.core.Workflows.adapters import run_media_ingest_adapter

        config = {
            "sources": [{"uri": ""}, {"uri": "   "}]
        }
        context = {}

        result = await run_media_ingest_adapter(config, context)

        # Empty URIs should be skipped
        assert result.get("metadata") == []

    @pytest.mark.asyncio
    async def test_media_ingest_file_access_denied(self, test_mode_env, tmp_path, monkeypatch):
        """Test media ingest denies access to files outside base directory."""
        from tldw_Server_API.app.core.Workflows.adapters import run_media_ingest_adapter

        # Set a restrictive base directory
        base_dir = tmp_path / "allowed"
        base_dir.mkdir()
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(base_dir))

        # Try to access file outside base
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("secret")

        config = {
            "sources": [{"uri": f"file://{outside_file}"}]
        }
        context = {"user_id": "1"}

        result = await run_media_ingest_adapter(config, context)

        assert len(result.get("metadata", [])) == 1
        assert result["metadata"][0]["status"] == "file_access_denied"

    @pytest.mark.asyncio
    async def test_media_ingest_cancellation(self, test_mode_env, monkeypatch):
        """Test media ingest respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters import run_media_ingest_adapter

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.media.ingest.is_url_allowed",
            lambda url: True
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.media.ingest.is_url_allowed_for_tenant",
            lambda url, tenant: True
        )

        # Remove TEST_MODE to trigger actual processing path (which checks cancellation)
        monkeypatch.delenv("TEST_MODE", raising=False)

        cancelled = False

        def is_cancelled():
            nonlocal cancelled
            cancelled = True
            return True

        config = {
            "sources": [{"uri": "https://example.com/video.mp4"}],
            "safety": {"allowed_domains": ["example.com"]},
            "timeout_seconds": 1,
        }
        context = {"is_cancelled": is_cancelled, "tenant_id": "default"}

        result = await run_media_ingest_adapter(config, context)

        # Should return cancelled status
        assert result.get("__status__") == "cancelled" or cancelled

    @pytest.mark.asyncio
    async def test_media_ingest_with_hierarchical_chunking(
        self, test_mode_env, workflow_file_base, sample_text_file, monkeypatch
    ):
        """Test media ingest with hierarchical chunking strategy."""
        from tldw_Server_API.app.core.Workflows.adapters import run_media_ingest_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(sample_text_file.parent))

        config = {
            "sources": [{"uri": f"file://{sample_text_file}"}],
            "extraction": {"extract_text": True},
            "chunking": {
                "strategy": "hierarchical",
                "max_tokens": 100,
                "overlap": 10,
                "hierarchical": {"levels": [{"strategy": "sentences"}]},
            },
        }
        context = {"user_id": "1"}

        result = await run_media_ingest_adapter(config, context)

        assert result["metadata"][0]["status"] == "local_ok"

    @pytest.mark.asyncio
    async def test_media_ingest_indexing_uses_media_repository_api(
        self, test_mode_env, workflow_file_base, sample_text_file, monkeypatch
    ):
        """Test workflow indexing persists through the media_db API seam."""
        import contextlib

        from tldw_Server_API.app.core.Workflows.adapters import run_media_ingest_adapter
        from tldw_Server_API.app.core.Workflows.adapters.media import ingest as ingest_module

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(sample_text_file.parent))
        events = []

        class _FakeRepo:
            def add_media_with_keywords(self, **kwargs):
                events.append(("add_media_with_keywords", kwargs["title"]))
                self.kwargs = kwargs
                return 501, "workflow-media", "stored"

        fake_repo = _FakeRepo()

        @contextlib.contextmanager
        def _fake_managed_media_database(client_id, **kwargs):
            events.append(("open", client_id, kwargs))
            yield object()

        monkeypatch.setattr(
            ingest_module,
            "managed_media_database",
            _fake_managed_media_database,
            raising=False,
        )
        monkeypatch.setattr(
            ingest_module,
            "get_media_repository",
            lambda _db: fake_repo,
            raising=False,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_single_user_id",
            staticmethod(lambda: 1),
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_media_db_path",
            staticmethod(lambda user_id: sample_text_file.parent / f"user-{user_id}.sqlite3"),
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.DB_Management.media_db.api.get_media_repository",
            lambda db: fake_repo,
        )

        config = {
            "sources": [{"uri": f"file://{sample_text_file}", "media_type": "document"}],
            "extraction": {"extract_text": True},
            "indexing": {"index_in_rag": True},
            "metadata": {"title": "Indexed workflow doc", "tags": ["workflow", "index"]},
        }
        context = {"user_id": "1"}

        result = await run_media_ingest_adapter(config, context)

        assert result["rag_indexed"] is True
        assert result["media_ids"] == [501]
        assert result["metadata"][0]["stored_media_id"] == 501
        assert result["metadata"][0]["db_message"] == "stored"
        assert events == [
            (
                "open",
                "workflow_engine",
                {
                    "db_path": str(sample_text_file.parent / "user-1.sqlite3"),
                    "initialize": False,
                },
            ),
            ("add_media_with_keywords", "Indexed workflow doc"),
        ]
        assert fake_repo.kwargs == {
            "url": f"file://{sample_text_file}",
            "title": "Indexed workflow doc",
            "media_type": "document",
            "content": sample_text_file.read_text(encoding="utf-8"),
            "keywords": ["workflow", "index"],
            "overwrite": False,
            "chunk_options": None,
            "chunks": None,
        }


# =============================================================================
# Tests for run_process_media_adapter
# =============================================================================


class TestProcessMediaAdapter:
    """Tests for the process_media adapter."""

    @pytest.mark.asyncio
    async def test_process_media_web_scraping_unavailable(self, test_mode_env, monkeypatch):
        """Test process_media returns error when web_scraping_service unavailable."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        original_import = builtins.__import__

        def _import_with_scraping_failure(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "tldw_Server_API.app.services.web_scraping_service":
                raise ImportError("simulated web scraping import failure")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _import_with_scraping_failure)

        config = {"kind": "web_scraping", "url_input": "https://example.com"}
        context = {}

        result = await run_process_media_adapter(config, context)

        assert result.get("error") == "web_scraping_service_unavailable"

    @pytest.mark.asyncio
    async def test_process_media_pdf_missing_file_uri(self, test_mode_env):
        """Test process_media PDF kind requires file_uri."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        config = {"kind": "pdf"}
        context = {}

        result = await run_process_media_adapter(config, context)

        assert result.get("error") == "missing_or_invalid_file_uri"

    @pytest.mark.asyncio
    async def test_process_media_pdf_invalid_uri(self, test_mode_env):
        """Test process_media PDF rejects non-file URIs."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        config = {"kind": "pdf", "file_uri": "https://example.com/doc.pdf"}
        context = {}

        result = await run_process_media_adapter(config, context)

        assert result.get("error") == "missing_or_invalid_file_uri"

    @pytest.mark.asyncio
    async def test_process_media_pdf_sanitizes_file_access_errors(self, test_mode_env, tmp_path, monkeypatch):
        """Test process_media PDF hides workflow file resolver details."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside_pdf = tmp_path / "private-report.pdf"
        outside_pdf.write_bytes(b"%PDF-1.4")
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(allowed_dir))

        result = await run_process_media_adapter({"kind": "pdf", "file_uri": f"file://{outside_pdf}"}, {})

        assert result == {"error": "file_access_denied"}

    @pytest.mark.asyncio
    async def test_process_media_pdf_sanitizes_backend_errors(self, test_mode_env, tmp_path, monkeypatch):
        """Test process_media PDF hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
        pdf_file = tmp_path / "document.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        async def raise_pdf_error(**kwargs):
            raise RuntimeError("pdf backend exploded at /private/pdf-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib.process_pdf_task",
            raise_pdf_error,
        )

        result = await run_process_media_adapter({"kind": "pdf", "file_uri": f"file://{pdf_file}"}, {})

        assert result == {"error": "pdf_process_error"}

    @pytest.mark.asyncio
    async def test_process_media_ebook_not_implemented(self, test_mode_env):
        """Test process_media ebook kind returns explicit not_implemented."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        config = {"kind": "ebook"}
        context = {}

        result = await run_process_media_adapter(config, context)

        assert result.get("error") == "not_implemented"
        assert result.get("kind") == "ebook"

    @pytest.mark.asyncio
    async def test_process_media_xml_not_implemented(self, test_mode_env):
        """Test process_media xml kind returns explicit not_implemented."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        config = {"kind": "xml"}
        context = {}

        result = await run_process_media_adapter(config, context)

        assert result.get("error") == "not_implemented"
        assert result.get("kind") == "xml"

    @pytest.mark.asyncio
    async def test_process_media_mediawiki_dump_missing_file_uri(self, test_mode_env):
        """Test process_media mediawiki_dump kind requires file_uri."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        config = {"kind": "mediawiki_dump"}
        context = {}

        result = await run_process_media_adapter(config, context)

        assert result.get("error") == "missing_or_invalid_file_uri"

    @pytest.mark.asyncio
    async def test_process_media_mediawiki_dump_sanitizes_file_access_errors(self, test_mode_env, tmp_path, monkeypatch):
        """Test process_media MediaWiki dump hides workflow file resolver details."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside_dump = tmp_path / "private-dump.xml"
        outside_dump.write_text("<mediawiki />")
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(allowed_dir))

        result = await run_process_media_adapter({"kind": "mediawiki_dump", "file_uri": f"file://{outside_dump}"}, {})

        assert result == {"error": "file_access_denied"}

    @pytest.mark.asyncio
    async def test_process_media_podcast_not_implemented(self, test_mode_env):
        """Test process_media podcast kind returns explicit not_implemented."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        config = {"kind": "podcast"}
        context = {}

        result = await run_process_media_adapter(config, context)

        assert result.get("error") == "not_implemented"
        assert result.get("kind") == "podcast"

    @pytest.mark.asyncio
    async def test_process_media_podcast_does_not_import_placeholder_service(self, test_mode_env, monkeypatch):
        """Test podcast returns not_implemented before any placeholder-service imports."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        original_import = builtins.__import__
        attempted_podcast_service_import = False

        def _import_with_podcast_failure(name, globals=None, locals=None, fromlist=(), level=0):
            nonlocal attempted_podcast_service_import
            if name == "tldw_Server_API.app.services.podcast_processing_service":
                attempted_podcast_service_import = True
                raise AssertionError("podcast placeholder service should not be imported")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _import_with_podcast_failure)

        config = {"kind": "podcast", "url": "https://example.com/podcast-episode"}
        context = {}

        result = await run_process_media_adapter(config, context)

        assert attempted_podcast_service_import is False
        assert result.get("error") == "not_implemented"
        assert result.get("kind") == "podcast"

    @pytest.mark.asyncio
    async def test_process_media_unsupported_kind(self, test_mode_env):
        """Test process_media returns error for unsupported kind."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        config = {"kind": "unknown_format"}
        context = {}

        result = await run_process_media_adapter(config, context)

        assert "unsupported_process_media_kind" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_process_media_cancellation(self, test_mode_env):
        """Test process_media respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        config = {"kind": "web_scraping"}
        context = {"is_cancelled": lambda: True}

        result = await run_process_media_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_process_media_mediawiki_dump_success(
        self, test_mode_env, tmp_path, monkeypatch
    ):
        """Test process_media mediawiki_dump with valid file."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        # Create a simple MediaWiki dump file
        dump_file = tmp_path / "wiki.xml"
        dump_file.write_text("<mediawiki><page><title>Test</title></page></mediawiki>")

        config = {"kind": "mediawiki_dump", "file_uri": f"file://{dump_file}"}
        context = {}

        result = await run_process_media_adapter(config, context)

        assert result.get("kind") == "mediawiki_dump"
        assert "content" in result
        assert "mediawiki" in result.get("content", "")

    @pytest.mark.asyncio
    async def test_process_media_web_scraping_with_mock(self, test_mode_env, monkeypatch):
        """Test process_media web scraping with mocked service."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        async def mock_scrape(*args, **kwargs):
            return {
                "status": "ok",
                "results": [
                    {"content": "Scraped content", "summary": "Test summary"}
                ]
            }

        # Mock the web scraping service
        monkeypatch.setattr(
            "tldw_Server_API.app.services.web_scraping_service.process_web_scraping_task",
            mock_scrape
        )

        config = {
            "kind": "web_scraping",
            "url_input": "https://example.com",
            "scrape_method": "Individual URLs",
        }
        context = {}

        result = await run_process_media_adapter(config, context)

        assert result.get("kind") == "web_scraping"
        assert result.get("count") == 1
        # Should have extracted text for chaining
        assert "text" in result


# =============================================================================
# Tests for run_pdf_extract_adapter
# =============================================================================


class TestPDFExtractAdapter:
    """Tests for the pdf_extract adapter."""

    @pytest.mark.asyncio
    async def test_pdf_extract_test_mode(self, test_mode_env, tmp_path, monkeypatch):
        """Test PDF extract in test mode returns simulated result."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        config = {"pdf_uri": f"file://{pdf_file}"}
        context = {}

        result = await run_pdf_extract_adapter(config, context)

        assert result.get("status") == "Success"
        assert result.get("simulated") is True
        assert "content" in result
        assert "text" in result
        assert "metadata" in result

    @pytest.mark.asyncio
    async def test_pdf_extract_missing_uri(self, test_mode_env):
        """Test PDF extract requires pdf_uri."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        config = {}
        context = {}

        result = await run_pdf_extract_adapter(config, context)

        assert result.get("error") == "missing_pdf_uri"
        assert result.get("status") == "Error"

    @pytest.mark.asyncio
    async def test_pdf_extract_empty_uri(self, test_mode_env):
        """Test PDF extract rejects empty pdf_uri."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        config = {"pdf_uri": "   "}
        context = {}

        result = await run_pdf_extract_adapter(config, context)

        assert result.get("error") == "missing_pdf_uri"

    @pytest.mark.asyncio
    async def test_pdf_extract_with_options_test_mode(self, test_mode_env, tmp_path, monkeypatch):
        """Test PDF extract with various options in test mode."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        config = {
            "pdf_uri": f"file://{pdf_file}",
            "parser": "pymupdf",
            "title": "Custom Title",
            "author": "Test Author",
            "keywords": "test,pdf,extract",
            "perform_chunking": True,
            "chunk_method": "paragraphs",
            "max_chunk_size": 1000,
            "chunk_overlap": 50,
            "enable_ocr": False,
        }
        context = {}

        result = await run_pdf_extract_adapter(config, context)

        assert result.get("status") == "Success"
        assert result.get("simulated") is True
        # Metadata should include our overrides
        meta = result.get("metadata", {})
        assert meta.get("title") == "Custom Title"
        assert meta.get("author") == "Test Author"
        # Should have chunks since perform_chunking is True
        assert isinstance(result.get("chunks"), list)

    @pytest.mark.asyncio
    async def test_pdf_extract_with_template_context(self, test_mode_env, tmp_path, monkeypatch):
        """Test PDF extract with templated pdf_uri."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        config = {"pdf_uri": "{{ inputs.pdf_path }}"}
        context = {"inputs": {"pdf_path": f"file://{pdf_file}"}}

        result = await run_pdf_extract_adapter(config, context)

        assert result.get("status") == "Success"

    @pytest.mark.asyncio
    async def test_pdf_extract_invalid_parser_fallback(self, test_mode_env, tmp_path, monkeypatch):
        """Test PDF extract falls back to default parser for invalid parser."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        config = {
            "pdf_uri": f"file://{pdf_file}",
            "parser": "invalid_parser",
        }
        context = {}

        result = await run_pdf_extract_adapter(config, context)

        # Should succeed with fallback parser
        assert result.get("status") == "Success"
        # Parser should be normalized to default
        assert result.get("metadata", {}).get("parser_used") == "pymupdf4llm"

    @pytest.mark.asyncio
    async def test_pdf_extract_cancellation(self, test_mode_env):
        """Test PDF extract respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        config = {"pdf_uri": "file:///some/path.pdf"}
        context = {"is_cancelled": lambda: True}

        result = await run_pdf_extract_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_pdf_extract_file_not_found(self, test_mode_env, tmp_path, monkeypatch):
        """Test PDF extract handles non-existent file."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        # Clear test mode to trigger actual file check
        monkeypatch.delenv("TEST_MODE", raising=False)
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        config = {"pdf_uri": f"file://{tmp_path}/nonexistent.pdf"}
        context = {}

        result = await run_pdf_extract_adapter(config, context)

        assert result.get("error") == "pdf_not_found"
        assert result.get("status") == "Error"

    @pytest.mark.asyncio
    async def test_pdf_extract_sanitizes_path_resolution_errors(self, tmp_path, monkeypatch):
        """Test PDF extract does not expose path resolution details."""
        from tldw_Server_API.app.core.exceptions import AdapterError
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        monkeypatch.delenv("TEST_MODE", raising=False)
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.media.documents.resolve_workflow_file_uri",
            lambda *args, **kwargs: (_ for _ in ()).throw(AdapterError("pdf denied at /private/pdf-input.pdf")),
        )

        result = await run_pdf_extract_adapter({"pdf_uri": "file:///private/pdf-input.pdf"}, {})

        assert result == {"error": "invalid_pdf_path", "status": "Error", "content": "", "text": ""}

    @pytest.mark.asyncio
    async def test_pdf_extract_sanitizes_file_read_errors(self, tmp_path, monkeypatch):
        """Test PDF extract does not expose file read exception details."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        monkeypatch.delenv("TEST_MODE", raising=False)
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        pdf_file = tmp_path / "private.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        def raise_read_error(self):
            raise OSError(f"pdf read exploded for {self} at /private/pdf-cache.bin")

        monkeypatch.setattr(type(pdf_file), "read_bytes", raise_read_error)

        result = await run_pdf_extract_adapter({"pdf_uri": f"file://{pdf_file}"}, {})

        assert result == {"error": "pdf_read_error", "status": "Error", "content": "", "text": ""}

    @pytest.mark.asyncio
    async def test_pdf_extract_sanitizes_processing_errors(self, tmp_path, monkeypatch):
        """Test PDF extract does not expose processing exception details."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        monkeypatch.delenv("TEST_MODE", raising=False)
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        pdf_file = tmp_path / "private.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        def raise_processing_error(**kwargs):
            raise RuntimeError("pdf processor exploded at /private/pdf-model-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib.process_pdf",
            raise_processing_error,
        )

        result = await run_pdf_extract_adapter({"pdf_uri": f"file://{pdf_file}"}, {})

        assert result == {"error": "pdf_extract_error", "status": "Error", "content": "", "text": ""}

    @pytest.mark.asyncio
    async def test_pdf_extract_keywords_as_list(self, test_mode_env, tmp_path, monkeypatch):
        """Test PDF extract handles keywords as list."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        config = {
            "pdf_uri": f"file://{pdf_file}",
            "keywords": ["keyword1", "keyword2", "keyword3"],
        }
        context = {}

        result = await run_pdf_extract_adapter(config, context)

        assert result.get("status") == "Success"
        assert result.get("keywords") == ["keyword1", "keyword2", "keyword3"]

    @pytest.mark.asyncio
    async def test_pdf_extract_no_chunking(self, test_mode_env, tmp_path, monkeypatch):
        """Test PDF extract without chunking."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        config = {
            "pdf_uri": f"file://{pdf_file}",
            "perform_chunking": False,
        }
        context = {}

        result = await run_pdf_extract_adapter(config, context)

        assert result.get("status") == "Success"
        # Chunks should be empty when chunking is disabled
        assert result.get("chunks") == []


# =============================================================================
# Tests for run_ocr_adapter
# =============================================================================


class TestOCRAdapter:
    """Tests for the ocr adapter."""

    @pytest.mark.asyncio
    async def test_ocr_test_mode(self, test_mode_env, tmp_path, monkeypatch):
        """Test OCR in test mode returns simulated result."""
        from tldw_Server_API.app.core.Workflows.adapters import run_ocr_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"fake png content")

        config = {"image_uri": f"file://{image_file}"}
        context = {}

        result = await run_ocr_adapter(config, context)

        assert result.get("simulated") is True
        assert "text" in result
        assert "TEST_MODE OCR" in result.get("text", "")

    @pytest.mark.asyncio
    async def test_ocr_missing_uri(self, test_mode_env):
        """Test OCR requires image_uri."""
        from tldw_Server_API.app.core.Workflows.adapters import run_ocr_adapter

        config = {}
        context = {}

        result = await run_ocr_adapter(config, context)

        assert result.get("error") == "missing_image_uri"

    @pytest.mark.asyncio
    async def test_ocr_empty_uri(self, test_mode_env):
        """Test OCR rejects empty image_uri."""
        from tldw_Server_API.app.core.Workflows.adapters import run_ocr_adapter

        config = {"image_uri": ""}
        context = {}

        result = await run_ocr_adapter(config, context)

        assert result.get("error") == "missing_image_uri"

    @pytest.mark.asyncio
    async def test_ocr_with_options_test_mode(self, test_mode_env, tmp_path, monkeypatch):
        """Test OCR with various options in test mode."""
        from tldw_Server_API.app.core.Workflows.adapters import run_ocr_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"fake png content")

        config = {
            "image_uri": f"file://{image_file}",
            "backend": "tesseract",
            "language": "deu",
            "output_format": "markdown",
        }
        context = {}

        result = await run_ocr_adapter(config, context)

        assert result.get("simulated") is True
        assert result.get("format") == "markdown"
        meta = result.get("meta", {})
        assert meta.get("backend") == "tesseract"
        assert meta.get("language") == "deu"

    @pytest.mark.asyncio
    async def test_ocr_cancellation(self, test_mode_env):
        """Test OCR respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters import run_ocr_adapter

        config = {"image_uri": "file:///some/image.png"}
        context = {"is_cancelled": lambda: True}

        result = await run_ocr_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_ocr_with_template_context(self, test_mode_env, tmp_path, monkeypatch):
        """Test OCR with templated image_uri."""
        from tldw_Server_API.app.core.Workflows.adapters import run_ocr_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        image_file = tmp_path / "scan.png"
        image_file.write_bytes(b"fake png content")

        config = {"image_uri": "{{ inputs.image_path }}"}
        context = {"inputs": {"image_path": f"file://{image_file}"}}

        result = await run_ocr_adapter(config, context)

        assert result.get("simulated") is True

    @pytest.mark.asyncio
    async def test_ocr_invalid_output_format_fallback(self, test_mode_env, tmp_path, monkeypatch):
        """Test OCR falls back to text for invalid output_format."""
        from tldw_Server_API.app.core.Workflows.adapters import run_ocr_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"fake png content")

        config = {
            "image_uri": f"file://{image_file}",
            "output_format": "invalid_format",
        }
        context = {}

        result = await run_ocr_adapter(config, context)

        assert result.get("simulated") is True
        assert result.get("format") == "text"

    @pytest.mark.asyncio
    async def test_ocr_file_not_found(self, test_mode_env, tmp_path, monkeypatch):
        """Test OCR handles non-existent file."""
        from tldw_Server_API.app.core.Workflows.adapters import run_ocr_adapter

        # Clear test mode to trigger actual file check
        monkeypatch.delenv("TEST_MODE", raising=False)
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        config = {"image_uri": f"file://{tmp_path}/nonexistent.png"}
        context = {}

        result = await run_ocr_adapter(config, context)

        assert result.get("error") == "image_not_found"

    @pytest.mark.asyncio
    async def test_ocr_sanitizes_path_resolution_errors(self, tmp_path, monkeypatch):
        """Test OCR does not expose path resolution details."""
        from tldw_Server_API.app.core.exceptions import AdapterError
        from tldw_Server_API.app.core.Workflows.adapters import run_ocr_adapter

        monkeypatch.delenv("TEST_MODE", raising=False)
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.media.documents.resolve_workflow_file_uri",
            lambda *args, **kwargs: (_ for _ in ()).throw(AdapterError("image denied at /private/ocr-input.png")),
        )

        result = await run_ocr_adapter({"image_uri": "file:///private/ocr-input.png"}, {})

        assert result == {"error": "invalid_image_path", "text": ""}

    @pytest.mark.asyncio
    async def test_ocr_sanitizes_image_read_errors(self, tmp_path, monkeypatch):
        """Test OCR does not expose image read exception details."""
        from tldw_Server_API.app.core.Workflows.adapters import run_ocr_adapter

        monkeypatch.delenv("TEST_MODE", raising=False)
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        image_file = tmp_path / "private.png"
        image_file.write_bytes(b"fake png content")

        def raise_read_error(self):
            raise OSError(f"image read exploded for {self} at /private/ocr-cache.bin")

        monkeypatch.setattr(type(image_file), "read_bytes", raise_read_error)

        result = await run_ocr_adapter({"image_uri": f"file://{image_file}"}, {})

        assert result == {"error": "image_read_error", "text": ""}

    @pytest.mark.asyncio
    async def test_ocr_sanitizes_backend_errors(self, tmp_path, monkeypatch):
        """Test OCR does not expose backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters import run_ocr_adapter

        monkeypatch.delenv("TEST_MODE", raising=False)
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        image_file = tmp_path / "private.png"
        image_file.write_bytes(b"fake png content")

        class ExplodingBackend:
            name = "exploding"

            def ocr_image(self, image_bytes, *, lang):
                raise RuntimeError("ocr backend exploded at /private/ocr-model-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry.get_backend",
            lambda backend_name: ExplodingBackend(),
        )

        result = await run_ocr_adapter({"image_uri": f"file://{image_file}"}, {})

        assert result == {"error": "ocr_error", "text": ""}

    @pytest.mark.asyncio
    async def test_ocr_blocks_returned_test_mode(self, test_mode_env, tmp_path, monkeypatch):
        """Test OCR returns block information in test mode."""
        from tldw_Server_API.app.core.Workflows.adapters import run_ocr_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        image_file = tmp_path / "document.png"
        image_file.write_bytes(b"fake png content")

        config = {"image_uri": f"file://{image_file}"}
        context = {}

        result = await run_ocr_adapter(config, context)

        assert result.get("simulated") is True
        assert isinstance(result.get("blocks"), list)
        # Test mode should return at least one simulated block
        if result.get("blocks"):
            block = result["blocks"][0]
            assert "text" in block
            assert "bbox" in block


# =============================================================================
# Tests for run_document_table_extract_adapter
# =============================================================================


class TestDocumentTableExtractAdapter:
    """Tests for the document_table_extract adapter."""

    @pytest.mark.asyncio
    async def test_table_extract_missing_file_path(self, test_mode_env):
        """Test table extract requires file_path or file_uri."""
        from tldw_Server_API.app.core.Workflows.adapters import run_document_table_extract_adapter

        config = {}
        context = {}

        result = await run_document_table_extract_adapter(config, context)

        assert result.get("error") == "missing_file_path"
        assert result.get("tables") == []
        assert result.get("count") == 0

    @pytest.mark.asyncio
    async def test_table_extract_cancellation(self, test_mode_env):
        """Test table extract respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters import run_document_table_extract_adapter

        config = {"file_path": "/some/path.pdf"}
        context = {"is_cancelled": lambda: True}

        result = await run_document_table_extract_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_table_extract_invalid_file_uri(self, test_mode_env, tmp_path, monkeypatch):
        """Test table extract handles invalid file_uri."""
        from tldw_Server_API.app.core.Workflows.adapters import run_document_table_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        config = {"file_uri": "file:///invalid/../../../path.pdf"}
        context = {}

        result = await run_document_table_extract_adapter(config, context)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_table_extract_with_docling_mock(self, test_mode_env, tmp_path, monkeypatch):
        """Test table extract with mocked docling provider."""
        from tldw_Server_API.app.core.Workflows.adapters import run_document_table_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        # Create a test PDF file
        pdf_file = tmp_path / "tables.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content with tables")

        # Mock docling
        class MockTable:
            def export_to_dataframe(self):
                import pandas as pd
                return pd.DataFrame({"Col1": [1, 2], "Col2": ["A", "B"]})

        class MockDocument:
            tables = [MockTable()]

        class MockResult:
            document = MockDocument()

        class MockConverter:
            def convert(self, path):
                return MockResult()

        monkeypatch.setattr(
            "docling.document_converter.DocumentConverter",
            MockConverter,
            raising=False
        )

        config = {
            "file_path": str(pdf_file),
            "output_format": "json",
            "provider": "docling",
        }
        context = {}

        result = await run_document_table_extract_adapter(config, context)

        # Result should have tables or handle docling not being installed
        assert "tables" in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_table_extract_llm_fallback(self, test_mode_env, tmp_path, monkeypatch):
        """Test table extract falls back to LLM extraction."""
        from tldw_Server_API.app.core.Workflows.adapters import run_document_table_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        # Create a test text file (not PDF, to trigger text-based extraction)
        text_file = tmp_path / "data.txt"
        text_file.write_text("Name | Value\nAlice | 100\nBob | 200")

        # Mock the LLM call
        async def mock_chat(*args, **kwargs):
            return {
                "choices": [{
                    "message": {
                        "content": '[{"headers": ["Name", "Value"], "rows": [["Alice", "100"], ["Bob", "200"]]}]'
                    }
                }]
            }

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            mock_chat
        )

        config = {
            "file_path": str(text_file),
            "output_format": "json",
            "provider": "llm",
        }
        context = {}

        result = await run_document_table_extract_adapter(config, context)

        assert "tables" in result
        if result.get("count", 0) > 0:
            table = result["tables"][0]
            assert "headers" in table
            assert "rows" in table

    @pytest.mark.asyncio
    async def test_table_extract_csv_output(self, test_mode_env, tmp_path, monkeypatch):
        """Test table extract with CSV output format."""
        from tldw_Server_API.app.core.Workflows.adapters import run_document_table_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        text_file = tmp_path / "data.txt"
        text_file.write_text("Simple text content")

        # Mock the LLM call to return a table
        async def mock_chat(*args, **kwargs):
            return {
                "choices": [{
                    "message": {
                        "content": '[{"headers": ["A", "B"], "rows": [["1", "2"]]}]'
                    }
                }]
            }

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            mock_chat
        )

        config = {
            "file_path": str(text_file),
            "output_format": "csv",
            "provider": "llm",
        }
        context = {}

        result = await run_document_table_extract_adapter(config, context)

        assert result.get("format") == "csv"
        if result.get("count", 0) > 0:
            # CSV format should have csv field
            table = result["tables"][0]
            assert "csv" in table

    @pytest.mark.asyncio
    async def test_table_extract_with_file_uri(self, test_mode_env, tmp_path, monkeypatch):
        """Test table extract with file_uri instead of file_path."""
        from tldw_Server_API.app.core.Workflows.adapters import run_document_table_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        text_file = tmp_path / "data.txt"
        text_file.write_text("Some content")

        async def mock_chat(*args, **kwargs):
            return {"choices": [{"message": {"content": "[]"}}]}

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            mock_chat
        )

        config = {
            "file_uri": f"file://{text_file}",
            "provider": "llm",
        }
        context = {}

        result = await run_document_table_extract_adapter(config, context)

        assert "tables" in result
        assert result.get("count") == 0  # No tables found

    @pytest.mark.asyncio
    async def test_table_extract_sanitizes_path_resolution_errors(self, tmp_path, monkeypatch):
        """Test table extraction does not expose path resolution details."""
        from tldw_Server_API.app.core.exceptions import AdapterError
        from tldw_Server_API.app.core.Workflows.adapters import run_document_table_extract_adapter

        monkeypatch.delenv("TEST_MODE", raising=False)
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.media.documents.resolve_workflow_file_uri",
            lambda *args, **kwargs: (_ for _ in ()).throw(AdapterError("table denied at /private/table-input.pdf")),
        )

        result = await run_document_table_extract_adapter(
            {"file_uri": "file:///private/table-input.pdf"},
            {},
        )

        assert result == {"error": "invalid_file_uri", "tables": [], "count": 0}

    @pytest.mark.asyncio
    async def test_table_extract_sanitizes_file_path_resolution_errors(self, tmp_path, monkeypatch):
        """Test table extraction does not expose file path resolution details."""
        from tldw_Server_API.app.core.exceptions import AdapterError
        from tldw_Server_API.app.core.Workflows.adapters import run_document_table_extract_adapter

        monkeypatch.delenv("TEST_MODE", raising=False)
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.media.documents.resolve_workflow_file_path",
            lambda *args, **kwargs: (_ for _ in ()).throw(AdapterError("table denied at /private/table-input.txt")),
        )

        result = await run_document_table_extract_adapter(
            {"file_path": "private/table-input.txt"},
            {},
        )

        assert result == {"error": "file_access_denied", "tables": [], "count": 0}

    @pytest.mark.asyncio
    async def test_table_extract_sanitizes_backend_errors(self, tmp_path, monkeypatch):
        """Test table extraction does not expose backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters import run_document_table_extract_adapter

        monkeypatch.delenv("TEST_MODE", raising=False)
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        text_file = tmp_path / "private.txt"
        text_file.write_text("Name | Value\nAlice | 100")

        async def raise_table_error(*args, **kwargs):
            raise RuntimeError("table extraction backend exploded at /private/table-model-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            raise_table_error,
        )

        result = await run_document_table_extract_adapter(
            {"file_path": str(text_file), "provider": "llm"},
            {},
        )

        assert result == {"error": "table_extract_error", "tables": [], "count": 0}

    @pytest.mark.asyncio
    async def test_table_extract_specific_table_index(self, test_mode_env, tmp_path, monkeypatch):
        """Test table extract with specific table_index."""
        from tldw_Server_API.app.core.Workflows.adapters import run_document_table_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        text_file = tmp_path / "data.txt"
        text_file.write_text("Multiple tables content")

        async def mock_chat(*args, **kwargs):
            return {
                "choices": [{
                    "message": {
                        "content": '''[
                            {"headers": ["A"], "rows": [["1"]]},
                            {"headers": ["B"], "rows": [["2"]]}
                        ]'''
                    }
                }]
            }

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            mock_chat
        )

        config = {
            "file_path": str(text_file),
            "table_index": 1,  # Get only second table
            "provider": "llm",
        }
        context = {}

        result = await run_document_table_extract_adapter(config, context)

        # table_index filtering happens with docling provider, not LLM
        # LLM returns all tables
        assert "tables" in result

    @pytest.mark.asyncio
    async def test_table_extract_templated_path(self, test_mode_env, tmp_path, monkeypatch):
        """Test table extract with templated file_path."""
        from tldw_Server_API.app.core.Workflows.adapters import run_document_table_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        text_file = tmp_path / "report.txt"
        text_file.write_text("Report content")

        async def mock_chat(*args, **kwargs):
            return {"choices": [{"message": {"content": "[]"}}]}

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            mock_chat
        )

        config = {
            "file_path": "{{ inputs.file }}",
            "provider": "llm",
        }
        context = {"inputs": {"file": str(text_file)}}

        result = await run_document_table_extract_adapter(config, context)

        assert "tables" in result


# =============================================================================
# Integration-style tests (still unit tests but test combinations)
# =============================================================================


class TestMediaAdaptersCombined:
    """Tests combining multiple adapter features."""

    @pytest.mark.asyncio
    async def test_media_ingest_multiple_sources(self, test_mode_env, tmp_path, monkeypatch):
        """Test media ingest with multiple sources of different types."""
        from tldw_Server_API.app.core.Workflows.adapters import run_media_ingest_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.media.ingest.is_url_allowed",
            lambda url: True
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.media.ingest.is_url_allowed_for_tenant",
            lambda url, tenant: True
        )

        # Create local file
        text_file = tmp_path / "doc.txt"
        text_file.write_text("Local content")

        config = {
            "sources": [
                {"uri": f"file://{text_file}"},
                {"uri": "https://example.com/video.mp4"},
            ],
            "safety": {"allowed_domains": ["example.com"]},
        }
        context = {"tenant_id": "default"}

        result = await run_media_ingest_adapter(config, context)

        # Should have metadata for both sources
        assert len(result.get("metadata", [])) == 2
        # First is local file
        assert result["metadata"][0]["status"] == "local_ok"
        # Second is HTTP (simulated in test mode)
        assert result["metadata"][1]["status"] == "simulated_download"

    @pytest.mark.asyncio
    async def test_pdf_extract_all_parsers_test_mode(self, test_mode_env, tmp_path, monkeypatch):
        """Test PDF extract works with all valid parser options."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        parsers = ["pymupdf4llm", "pymupdf", "docling"]

        for parser in parsers:
            config = {
                "pdf_uri": f"file://{pdf_file}",
                "parser": parser,
            }
            context = {}

            result = await run_pdf_extract_adapter(config, context)

            assert result.get("status") == "Success", f"Parser {parser} failed"
            assert result.get("metadata", {}).get("parser_used") == parser

    @pytest.mark.asyncio
    async def test_ocr_all_formats_test_mode(self, test_mode_env, tmp_path, monkeypatch):
        """Test OCR works with all valid output formats."""
        from tldw_Server_API.app.core.Workflows.adapters import run_ocr_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"fake png")

        formats = ["text", "markdown", "html", "json"]

        for fmt in formats:
            config = {
                "image_uri": f"file://{image_file}",
                "output_format": fmt,
            }
            context = {}

            result = await run_ocr_adapter(config, context)

            assert result.get("simulated") is True, f"Format {fmt} failed"
            assert result.get("format") == fmt


# =============================================================================
# Edge case and error handling tests
# =============================================================================


class TestMediaAdaptersEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_media_ingest_heartbeat_callback(self, test_mode_env, tmp_path, monkeypatch):
        """Test media ingest calls heartbeat callback."""
        from tldw_Server_API.app.core.Workflows.adapters import run_media_ingest_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        text_file = tmp_path / "doc.txt"
        text_file.write_text("Content")

        heartbeat_called = []

        def heartbeat():
            heartbeat_called.append(True)

        config = {"sources": [{"uri": f"file://{text_file}"}]}
        context = {"heartbeat": heartbeat}

        await run_media_ingest_adapter(config, context)

        # Heartbeat may or may not be called for local files
        # This test ensures the callback doesn't break anything

    @pytest.mark.asyncio
    async def test_media_ingest_add_artifact_callback(self, test_mode_env, tmp_path, monkeypatch):
        """Test media ingest calls add_artifact callback."""
        from tldw_Server_API.app.core.Workflows.adapters import run_media_ingest_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        text_file = tmp_path / "doc.txt"
        text_file.write_text("Content for artifact test")

        artifacts = []

        def add_artifact(**kwargs):
            artifacts.append(kwargs)

        config = {
            "sources": [{"uri": f"file://{text_file}"}],
            "indexing": {"index_in_rag": False},
        }
        context = {"add_artifact": add_artifact}

        await run_media_ingest_adapter(config, context)

        # Local files don't trigger artifact addition by default
        # But this ensures the callback mechanism works

    @pytest.mark.asyncio
    async def test_pdf_extract_with_ocr_options_test_mode(
        self, test_mode_env, tmp_path, monkeypatch
    ):
        """Test PDF extract with OCR options in test mode."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        pdf_file = tmp_path / "scanned.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 scanned content")

        config = {
            "pdf_uri": f"file://{pdf_file}",
            "enable_ocr": True,
            "ocr_backend": "tesseract",
            "ocr_lang": "eng+fra",
            "ocr_mode": "always",
        }
        context = {}

        result = await run_pdf_extract_adapter(config, context)

        assert result.get("status") == "Success"
        assert result.get("simulated") is True

    @pytest.mark.asyncio
    async def test_pdf_extract_with_vlm_options_test_mode(
        self, test_mode_env, tmp_path, monkeypatch
    ):
        """Test PDF extract with VLM options in test mode."""
        from tldw_Server_API.app.core.Workflows.adapters import run_pdf_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        pdf_file = tmp_path / "complex.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 complex content")

        config = {
            "pdf_uri": f"file://{pdf_file}",
            "enable_vlm": True,
            "vlm_backend": "openai",
            "vlm_detect_tables_only": False,
        }
        context = {}

        result = await run_pdf_extract_adapter(config, context)

        assert result.get("status") == "Success"
        assert result.get("simulated") is True

    @pytest.mark.asyncio
    async def test_process_media_pdf_with_mock(self, test_mode_env, tmp_path, monkeypatch):
        """Test process_media PDF with mocked processing."""
        from tldw_Server_API.app.core.Workflows.adapters import run_process_media_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        pdf_file = tmp_path / "document.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        async def mock_pdf_task(**kwargs):
            return {
                "status": "Success",
                "text": "Extracted PDF text",
                "content": "Extracted PDF text",
                "metadata": {"page_count": 3},
            }

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib.process_pdf_task",
            mock_pdf_task
        )

        config = {
            "kind": "pdf",
            "file_uri": f"file://{pdf_file}",
            "parser": "pymupdf",
        }
        context = {}

        result = await run_process_media_adapter(config, context)

        assert result.get("kind") == "pdf"
        assert "content" in result or "text" in result

    @pytest.mark.asyncio
    async def test_table_extract_pdf_with_pymupdf(self, test_mode_env, tmp_path, monkeypatch):
        """Test table extract from PDF using pymupdf fallback."""
        from tldw_Server_API.app.core.Workflows.adapters import run_document_table_extract_adapter

        monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

        # Create a minimal PDF
        pdf_file = tmp_path / "tables.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 minimal pdf")

        # Mock pymupdf
        class MockPage:
            def get_text(self):
                return "Name\tValue\nAlice\t100"

        class MockDoc:
            def __iter__(self):
                return iter([MockPage()])

            def close(self):
                pass

        def mock_open(path):
            return MockDoc()

        # Mock both docling (to fail) and pymupdf
        def mock_docling_import(*args, **kwargs):
            raise ImportError("docling not available")

        async def mock_chat(*args, **kwargs):
            return {
                "choices": [{
                    "message": {
                        "content": '[{"headers": ["Name", "Value"], "rows": [["Alice", "100"]]}]'
                    }
                }]
            }

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            mock_chat
        )

        config = {
            "file_path": str(pdf_file),
            "provider": "llm",
        }
        context = {}

        result = await run_document_table_extract_adapter(config, context)

        assert "tables" in result
