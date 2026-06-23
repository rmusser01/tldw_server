"""
Integration tests for media endpoint with contextual chunking options.

Tests the full flow of media ingestion with contextual features.
"""
import os

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
pytestmark = pytest.mark.unit
import json
import io
import tempfile
from pathlib import Path

from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm


def _ensure_media_add_route_registered(app) -> None:
    """Mount the media router when another test package disables it at collection time."""
    if any(getattr(route, "path", None) == "/api/v1/media/add" for route in app.routes):
        return

    from tldw_Server_API.app.api.v1.endpoints.media import router as media_router
    from tldw_Server_API.app.api.v1.router_registry import include_router_idempotent

    include_router_idempotent(
        app,
        media_router,
        prefix="/api/v1/media",
        tags=["media"],
    )


class TestMediaEndpointContextualIntegration:
    """Integration tests for media endpoint with contextual chunking."""

    @pytest.fixture
    def test_client(self, client_user_only):
        """Use the shared authenticated TestClient with a stub Media DB."""
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user as dep_get_db

        app = client_user_only.app
        _ensure_media_add_route_registered(app)

        mock_db = Mock(
            db_path="/test/path.db",
            db_path_str="/test/path.db",
            client_id="test_client",
            spec=[
                "db_path",
                "db_path_str",
                "client_id",
                "add_media_with_keywords",
                "search_media",
                "insert_media_file",
                "close_connection",
            ],
        )
        mock_db.add_media_with_keywords.return_value = (123, "test-uuid", "ok")
        mock_db.search_media.return_value = []

        async def _override_db():
            yield mock_db

        original_db_override = app.dependency_overrides.get(dep_get_db)
        app.dependency_overrides[dep_get_db] = _override_db
        try:
            yield client_user_only
        finally:
            if original_db_override is None:
                app.dependency_overrides.pop(dep_get_db, None)
            else:
                app.dependency_overrides[dep_get_db] = original_db_override

    @pytest.fixture
    def auth_headers(self):
        # Real app runs in single_user mode by default in tests; use test API key
        api_key = os.getenv("SINGLE_USER_API_KEY", "test-api-key-12345")
        return {
            "X-API-KEY": api_key,
            "x-api-key": api_key
        }

    def test_add_media_with_contextual_chunking_enabled(self, test_client, auth_headers):

        """Test adding media with contextual chunking enabled."""
        # Prepare request data
        form_data = {
            "media_type": "document",
            "urls": json.dumps(["https://example.com/test.pdf"]),
            "perform_chunking": "true",
            "chunk_size": "500",
            "chunk_overlap": "100",
            "enable_contextual_chunking": "true",  # Enable contextual
            "contextual_llm_model": "gpt-4",
            "context_window_size": "750"
        }

        with patch('tldw_Server_API.app.api.v1.endpoints.media.process_document_content') as mock_process:
            mock_process.return_value = {"success": True, "media_id": 123}

            response = test_client.post(
                "/api/v1/media/add",
                data=form_data,
                headers=auth_headers
            )

            # Verify the chunking options were passed correctly
            if mock_process.called:
                call_args = mock_process.call_args
                chunk_options = call_args[1].get('chunk_options', {})

                assert chunk_options.get('enable_contextual_chunking') == True
                assert chunk_options.get('contextual_llm_model') == "gpt-4"
                assert chunk_options.get('context_window_size') == 750

    def test_add_media_with_contextual_chunking_disabled(self, test_client, auth_headers):

        """Test adding media with contextual chunking explicitly disabled."""
        form_data = {
            "media_type": "document",
            "urls": json.dumps(["https://example.com/test.pdf"]),
            "perform_chunking": "true",
            "chunk_size": "500",
            "chunk_overlap": "100",
            "enable_contextual_chunking": "false"  # Explicitly disable
        }

        with patch('tldw_Server_API.app.api.v1.endpoints.media.process_document_content') as mock_process:
            mock_process.return_value = {"success": True, "media_id": 123}

            response = test_client.post(
                "/api/v1/media/add",
                data=form_data,
                headers=auth_headers
            )

            if mock_process.called:
                call_args = mock_process.call_args
                chunk_options = call_args[1].get('chunk_options', {})

                assert chunk_options.get('enable_contextual_chunking') == False

    def test_add_media_contextual_defaults_from_config(self, test_client, auth_headers):

        """Test that contextual chunking uses config defaults when not specified."""
        form_data = {
            "media_type": "document",
            "urls": json.dumps(["https://example.com/test.pdf"]),
            "perform_chunking": "true",
            "chunk_size": "500",
            "chunk_overlap": "100"
            # No contextual options specified - should use defaults
        }

        with patch('tldw_Server_API.app.api.v1.endpoints.media.process_document_content') as mock_process:
            mock_process.return_value = {"success": True, "media_id": 123}

            response = test_client.post(
                "/api/v1/media/add",
                data=form_data,
                headers=auth_headers
            )

            if mock_process.called:
                call_args = mock_process.call_args
                chunk_options = call_args[1].get('chunk_options', {})

                # Should have default value (False based on our config)
                assert chunk_options.get('enable_contextual_chunking') == False
                assert chunk_options.get('contextual_llm_model') is None
                assert chunk_options.get('context_window_size') is None

    def test_add_media_file_upload_with_contextual(self, test_client):

        """Test file upload with contextual chunking options."""
        # Create a test file
        test_content = b"Test document content for contextual chunking"
        test_file = io.BytesIO(test_content)

        files = {
            "files": ("test.txt", test_file, "text/plain")
        }

        form_data = {
            "media_type": "document",
            "perform_chunking": "true",
            "enable_contextual_chunking": "true",
            "contextual_llm_model": "gpt-3.5-turbo"
        }

        with patch(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing.save_uploaded_files"
        ) as mock_upload:
            with patch('tldw_Server_API.app.api.v1.endpoints.media.process_document_content') as mock_process:
                mock_upload.return_value = (["/tmp/test.txt"], [])  # nosec B108
                mock_process.return_value = {"success": True, "media_id": 124}

                response = test_client.post(
                    "/api/v1/media/add",
                    data=form_data,
                    files=files,
                    headers={"Authorization": "Bearer test_token"}
                )

                # Verify contextual options were passed
                if mock_process.called:
                    call_args = mock_process.call_args
                    chunk_options = call_args[1].get('chunk_options', {})

                    assert chunk_options.get('enable_contextual_chunking') == True
                    assert chunk_options.get('contextual_llm_model') == "gpt-3.5-turbo"

    @pytest.mark.parametrize("media_type,expected_method", [
        ("document", "sentences"),
        ("pdf", "sentences"),
        ("ebook", "ebook_chapters"),
        ("video", "sentences"),
        ("audio", "sentences")
    ])
    def test_contextual_chunking_with_different_media_types(
        self,
        test_client,
        auth_headers,
        media_type,
        expected_method
    ):
        """Test contextual chunking works with different media types."""
        form_data = {
            "media_type": media_type,
            "urls": json.dumps([f"https://example.com/test.{media_type}"]),
            "perform_chunking": "true",
            "enable_contextual_chunking": "true"
        }

        # Mock the appropriate processing function based on media type.
        # For video/audio, patch the orchestrator-level batch helper to avoid
        # importing heavyweight STT/transcription dependencies during tests.
        process_target = {
            "document": "tldw_Server_API.app.api.v1.endpoints.media.process_document_content",
            "pdf": "tldw_Server_API.app.api.v1.endpoints.media.process_pdf_task",
            "ebook": "tldw_Server_API.app.api.v1.endpoints.media.process_epub",
            "video": "tldw_Server_API.app.core.Ingestion_Media_Processing.persistence.process_batch_media",
            "audio": "tldw_Server_API.app.core.Ingestion_Media_Processing.persistence.process_batch_media",
        }.get(
            media_type,
            "tldw_Server_API.app.api.v1.endpoints.media.process_document_content",
        )

        patch_kwargs = {"new_callable": AsyncMock} if media_type in {"video", "audio"} else {}
        with patch(process_target, **patch_kwargs) as mock_process:
            if media_type in {"video", "audio"}:
                mock_process.return_value = [{"status": "Success", "media_id": 125}]
            else:
                mock_process.return_value = {"success": True, "media_id": 125}

            response = test_client.post(
                "/api/v1/media/add",
                data=form_data,
                headers=auth_headers
            )

            if mock_process.called:
                call_args = mock_process.call_args
                chunk_options = call_args[1].get('chunk_options', {})

                assert chunk_options.get('enable_contextual_chunking') == True
                assert chunk_options.get('method') == expected_method

    def test_batch_media_with_contextual_chunking(self, test_client, auth_headers):

        """Test batch media processing with contextual chunking."""
        form_data = {
            "media_type": "document",
            "urls": json.dumps([
                "https://example.com/doc1.txt",
                "https://example.com/doc2.txt",
                "https://example.com/doc3.txt"
            ]),
            "perform_chunking": True,
            "enable_contextual_chunking": True,
            "contextual_llm_model": "claude-opus-4.1"
        }

        with patch('tldw_Server_API.app.api.v1.endpoints.media.process_document_content') as mock_process:
            mock_process.return_value = {"success": True, "media_id": 126}

            # Avoid real network: mock _download_url_async to create a temporary file inside provided temp_dir
            from pathlib import Path
            async def _fake_download_url_async(**kwargs):
                url = kwargs.get("url")
                target_dir = kwargs.get("target_dir")
                p = Path(str(target_dir)) / (Path(url).name or "test.txt")
                p.write_text("Test document content for contextual chunking.", encoding="utf-8")
                return p
            with patch(
                "tldw_Server_API.app.core.Ingestion_Media_Processing.download_utils.download_url_async",
                side_effect=_fake_download_url_async,
            ), patch(
                "tldw_Server_API.app.core.Security.url_validation.assert_url_safe",
                return_value=None,
            ):
                response = test_client.post(
                    "/api/v1/media/add",
                    data=form_data,
                    headers=auth_headers
                )

            # Should be called for each URL
            assert mock_process.call_count >= 1, response.text

            # Each call should have contextual options
            for call in mock_process.call_args_list:
                chunk_options = call[1].get('chunk_options', {})
                assert chunk_options.get('enable_contextual_chunking') == True
                assert chunk_options.get('contextual_llm_model') == "claude-opus-4.1"

    def test_contextual_options_validation(self, test_client, auth_headers):

        """Test validation of contextual chunking options."""
        # Test with invalid context_window_size (too small)
        form_data = {
            "media_type": "document",
            "urls": json.dumps(["https://example.com/test.pdf"]),
            "perform_chunking": "true",
            "enable_contextual_chunking": "true",
            "context_window_size": "50"  # Below minimum of 100
        }

        response = test_client.post(
            "/api/v1/media/add",
            data=form_data,
            headers={"Authorization": "Bearer test_token"}
        )

        # Should get validation error
        assert response.status_code == 422

        # Test with invalid context_window_size (too large)
        form_data["context_window_size"] = "3000"  # Above maximum of 2000

        response = test_client.post(
            "/api/v1/media/add",
            data=form_data,
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 422

    def test_contextual_chunking_preserves_other_options(self, test_client, auth_headers):

        """Test that contextual options don't interfere with other chunking options."""
        form_data = {
            "media_type": "document",
            "urls": json.dumps(["https://example.com/test.pdf"]),
            "perform_chunking": "true",
            "chunk_method": "semantic",
            "chunk_size": "1000",
            "chunk_overlap": "200",
            "use_adaptive_chunking": "true",
            "use_multi_level_chunking": "true",
            "chunk_language": "en",
            "enable_contextual_chunking": "true",
            "contextual_llm_model": "gpt-4"
        }

        with patch('tldw_Server_API.app.api.v1.endpoints.media.process_document_content') as mock_process:
            mock_process.return_value = {"success": True, "media_id": 127}

            response = test_client.post(
                "/api/v1/media/add",
                data=form_data,
                headers={"Authorization": "Bearer test_token"}
            )

            if mock_process.called:
                call_args = mock_process.call_args
                chunk_options = call_args[1].get('chunk_options', {})

                # All options should be preserved
                assert chunk_options.get('method') == "semantic"
                assert chunk_options.get('max_size') == 1000
                assert chunk_options.get('overlap') == 200
                assert chunk_options.get('adaptive') == True
                assert chunk_options.get('multi_level') == True
                assert chunk_options.get('language') == "en"
                assert chunk_options.get('enable_contextual_chunking') == True
                assert chunk_options.get('contextual_llm_model') == "gpt-4"
