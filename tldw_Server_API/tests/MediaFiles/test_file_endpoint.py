# tests/MediaFiles/test_file_endpoint.py
"""
Integration tests for the media file serving endpoint.

Tests cover:
- GET /api/v1/media/{id}/file - retrieving original files
- HEAD /api/v1/media/{id}/file - checking file existence
- Response headers (Content-Type, Content-Disposition, Content-Length)
- 404 responses for missing media/files
- RFC 5987 Content-Disposition header encoding
"""
import tempfile
from io import BytesIO
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.tests.test_utils import create_test_media


def _principal_override():


    """Create a test principal override for authentication."""
    async def _override(request=None) -> AuthPrincipal:
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="test-user",
            token_type="single_user",
            jti=None,
            roles=["admin"],
            permissions=["media.read"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(
                principal=principal,
                ip=None,
                user_agent=None,
                request_id=None,
            )
        return principal
    return _override


async def _mock_retrieve_stream(path: str, chunk_size: int = 65536):
    """Mock async generator for retrieve_stream."""
    yield b"%PDF-1.4 mock PDF content"


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = MagicMock()
    storage.exists = AsyncMock(return_value=True)
    storage.retrieve = AsyncMock(return_value=BytesIO(b"%PDF-1.4 mock PDF content"))
    storage.retrieve_stream = MagicMock(side_effect=_mock_retrieve_stream)
    storage.get_size = AsyncMock(return_value=25)
    return storage


class TestGetMediaFile:
    """Tests for GET /api/v1/media/{id}/file endpoint."""

    @pytest.mark.integration
    def test_get_file_returns_pdf_content(self, tmp_path, mock_storage):
        """Test that GET /media/{id}/file returns the file content."""
        from tldw_Server_API.app.main import app as fastapi_app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

        # Setup: create database with media and file record
        db_path = tmp_path / "media.db"
        seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        media_id = create_test_media(seed_db, title="Test PDF", content="PDF content")
        seed_db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="user1/media/1/test.pdf",
            original_filename="test.pdf",
            file_size=25,
            mime_type="application/pdf",
        )
        seed_db.close_connection()

        async def _override_user() -> User:
            return User(id=1, username="tester", email=None, is_active=True)

        async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
            override_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
            try:
                yield override_db
            finally:
                override_db.close_connection()

        fastapi_app.dependency_overrides[get_request_user] = _override_user
        fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
        fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

        try:
            with patch(
                "tldw_Server_API.app.core.Storage.get_storage_backend",
                return_value=mock_storage
            ):
                with TestClient(fastapi_app) as client:
                    response = client.get(f"/api/v1/media/{media_id}/file")

                    assert response.status_code == 200
                    assert response.headers["Content-Type"] == "application/pdf"
                    assert "Content-Disposition" in response.headers
                    assert "inline" in response.headers["Content-Disposition"]
                    assert b"PDF" in response.content
        finally:
            fastapi_app.dependency_overrides.pop(get_request_user, None)
            fastapi_app.dependency_overrides.pop(get_auth_principal, None)
            fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)

    @pytest.mark.integration
    def test_get_file_includes_content_length(self, tmp_path, mock_storage):
        """Test that GET /media/{id}/file includes Content-Length header."""
        from tldw_Server_API.app.main import app as fastapi_app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

        db_path = tmp_path / "media.db"
        seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        media_id = create_test_media(seed_db, title="Test PDF", content="PDF content")
        seed_db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="user1/media/1/test.pdf",
            original_filename="test.pdf",
            file_size=12345,  # Known size
            mime_type="application/pdf",
        )
        seed_db.close_connection()

        async def _override_user() -> User:
            return User(id=1, username="tester", email=None, is_active=True)

        async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
            override_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
            try:
                yield override_db
            finally:
                override_db.close_connection()

        fastapi_app.dependency_overrides[get_request_user] = _override_user
        fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
        fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

        try:
            with patch(
                "tldw_Server_API.app.core.Storage.get_storage_backend",
                return_value=mock_storage
            ):
                with TestClient(fastapi_app) as client:
                    response = client.get(f"/api/v1/media/{media_id}/file")

                    assert response.status_code == 200
                    assert "Content-Length" in response.headers
                    assert response.headers["Content-Length"] == "12345"
        finally:
            fastapi_app.dependency_overrides.pop(get_request_user, None)
            fastapi_app.dependency_overrides.pop(get_auth_principal, None)
            fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)

    @pytest.mark.integration
    def test_get_file_404_when_media_not_found(self, tmp_path):
        """Test that GET returns 404 when media doesn't exist."""
        from tldw_Server_API.app.main import app as fastapi_app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

        db_path = tmp_path / "media.db"
        seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        seed_db.close_connection()

        async def _override_user() -> User:
            return User(id=1, username="tester", email=None, is_active=True)

        async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
            override_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
            try:
                yield override_db
            finally:
                override_db.close_connection()

        fastapi_app.dependency_overrides[get_request_user] = _override_user
        fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
        fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

        try:
            with TestClient(fastapi_app) as client:
                response = client.get("/api/v1/media/99999/file")
                assert response.status_code == 404
        finally:
            fastapi_app.dependency_overrides.pop(get_request_user, None)
            fastapi_app.dependency_overrides.pop(get_auth_principal, None)
            fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)

    @pytest.mark.integration
    def test_get_file_404_when_no_file_record(self, tmp_path, mock_storage):
        """Test that GET returns 404 when media exists but has no file."""
        from tldw_Server_API.app.main import app as fastapi_app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

        db_path = tmp_path / "media.db"
        seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        media_id = create_test_media(seed_db, title="No File Media", content="Content")
        # Deliberately NOT adding a file record
        seed_db.close_connection()

        async def _override_user() -> User:
            return User(id=1, username="tester", email=None, is_active=True)

        async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
            override_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
            try:
                yield override_db
            finally:
                override_db.close_connection()

        fastapi_app.dependency_overrides[get_request_user] = _override_user
        fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
        fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

        try:
            with TestClient(fastapi_app) as client:
                response = client.get(f"/api/v1/media/{media_id}/file")
                assert response.status_code == 404
                assert "No original file" in response.json()["detail"]
        finally:
            fastapi_app.dependency_overrides.pop(get_request_user, None)
            fastapi_app.dependency_overrides.pop(get_auth_principal, None)
            fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)


class TestGetMediaFileHeaders:
    """Tests for response headers on GET /api/v1/media/{id}/file."""

    @pytest.mark.integration
    def test_content_disposition_uses_rfc5987_encoding(self, tmp_path, mock_storage):
        """Test that Content-Disposition uses RFC 5987 encoding for filenames."""
        from tldw_Server_API.app.main import app as fastapi_app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

        db_path = tmp_path / "media.db"
        seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        media_id = create_test_media(seed_db, title="Unicode Test", content="Content")
        # Use a filename with Unicode characters
        seed_db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="user1/media/1/test.pdf",
            original_filename="文档.pdf",  # Chinese characters
            file_size=100,
            mime_type="application/pdf",
        )
        seed_db.close_connection()

        async def _override_user() -> User:
            return User(id=1, username="tester", email=None, is_active=True)

        async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
            override_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
            try:
                yield override_db
            finally:
                override_db.close_connection()

        fastapi_app.dependency_overrides[get_request_user] = _override_user
        fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
        fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

        try:
            with patch(
                "tldw_Server_API.app.core.Storage.get_storage_backend",
                return_value=mock_storage
            ):
                with TestClient(fastapi_app) as client:
                    response = client.get(f"/api/v1/media/{media_id}/file")

                    assert response.status_code == 200
                    content_disp = response.headers["Content-Disposition"]
                    # Should have RFC 5987 encoding
                    assert "filename*=UTF-8''" in content_disp
                    # Should have percent-encoded Unicode
                    assert "%E6%96%87%E6%A1%A3" in content_disp  # 文档 encoded
        finally:
            fastapi_app.dependency_overrides.pop(get_request_user, None)
            fastapi_app.dependency_overrides.pop(get_auth_principal, None)
            fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)

    @pytest.mark.integration
    def test_accept_ranges_header_present(self, tmp_path, mock_storage):
        """Test that Accept-Ranges header is present in response."""
        from tldw_Server_API.app.main import app as fastapi_app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

        db_path = tmp_path / "media.db"
        seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        media_id = create_test_media(seed_db, title="Test PDF", content="Content")
        seed_db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="user1/media/1/test.pdf",
            original_filename="test.pdf",
            file_size=100,
            mime_type="application/pdf",
        )
        seed_db.close_connection()

        async def _override_user() -> User:
            return User(id=1, username="tester", email=None, is_active=True)

        async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
            override_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
            try:
                yield override_db
            finally:
                override_db.close_connection()

        fastapi_app.dependency_overrides[get_request_user] = _override_user
        fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
        fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

        try:
            with patch(
                "tldw_Server_API.app.core.Storage.get_storage_backend",
                return_value=mock_storage
            ):
                with TestClient(fastapi_app) as client:
                    response = client.get(f"/api/v1/media/{media_id}/file")

                    assert response.status_code == 200
                    assert response.headers.get("Accept-Ranges") == "bytes"
        finally:
            fastapi_app.dependency_overrides.pop(get_request_user, None)
            fastapi_app.dependency_overrides.pop(get_auth_principal, None)
            fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)


class TestRangeRequests:
    """Tests for HTTP Range request support."""

    @pytest.mark.integration
    def test_range_request_returns_206(self, tmp_path, mock_storage):
        """Test that Range requests return 206 Partial Content."""
        from tldw_Server_API.app.main import app as fastapi_app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

        db_path = tmp_path / "media.db"
        seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        media_id = create_test_media(seed_db, title="Test PDF", content="Content")
        seed_db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="user1/media/1/test.pdf",
            original_filename="test.pdf",
            file_size=1000,
            mime_type="application/pdf",
        )
        seed_db.close_connection()

        async def _override_user() -> User:
            return User(id=1, username="tester", email=None, is_active=True)

        async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
            override_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
            try:
                yield override_db
            finally:
                override_db.close_connection()

        fastapi_app.dependency_overrides[get_request_user] = _override_user
        fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
        fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

        try:
            with patch(
                "tldw_Server_API.app.core.Storage.get_storage_backend",
                return_value=mock_storage
            ):
                with TestClient(fastapi_app) as client:
                    response = client.get(
                        f"/api/v1/media/{media_id}/file",
                        headers={"Range": "bytes=0-10"}
                    )

                    assert response.status_code == 206
                    assert "Content-Range" in response.headers
        finally:
            fastapi_app.dependency_overrides.pop(get_request_user, None)
            fastapi_app.dependency_overrides.pop(get_auth_principal, None)
            fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)


class TestETagCaching:
    """Tests for ETag-based caching."""

    @pytest.mark.integration
    def test_etag_header_present_when_checksum_available(self, tmp_path, mock_storage):
        """Test that ETag header is present when file has checksum."""
        from tldw_Server_API.app.main import app as fastapi_app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

        db_path = tmp_path / "media.db"
        seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        media_id = create_test_media(seed_db, title="Test PDF", content="Content")
        seed_db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="user1/media/1/test.pdf",
            original_filename="test.pdf",
            file_size=100,
            mime_type="application/pdf",
            checksum="abc123def456",
        )
        seed_db.close_connection()

        async def _override_user() -> User:
            return User(id=1, username="tester", email=None, is_active=True)

        async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
            override_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
            try:
                yield override_db
            finally:
                override_db.close_connection()

        fastapi_app.dependency_overrides[get_request_user] = _override_user
        fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
        fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

        try:
            with patch(
                "tldw_Server_API.app.core.Storage.get_storage_backend",
                return_value=mock_storage
            ):
                with TestClient(fastapi_app) as client:
                    response = client.get(f"/api/v1/media/{media_id}/file")

                    assert response.status_code == 200
                    assert "ETag" in response.headers
                    assert response.headers["ETag"] == '"abc123def456"'
        finally:
            fastapi_app.dependency_overrides.pop(get_request_user, None)
            fastapi_app.dependency_overrides.pop(get_auth_principal, None)
            fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)

    @pytest.mark.integration
    def test_if_none_match_returns_304(self, tmp_path, mock_storage):
        """Test that matching If-None-Match returns 304 Not Modified."""
        from tldw_Server_API.app.main import app as fastapi_app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

        db_path = tmp_path / "media.db"
        seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        media_id = create_test_media(seed_db, title="Test PDF", content="Content")
        seed_db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="user1/media/1/test.pdf",
            original_filename="test.pdf",
            file_size=100,
            mime_type="application/pdf",
            checksum="abc123def456",
        )
        seed_db.close_connection()

        async def _override_user() -> User:
            return User(id=1, username="tester", email=None, is_active=True)

        async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
            override_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
            try:
                yield override_db
            finally:
                override_db.close_connection()

        fastapi_app.dependency_overrides[get_request_user] = _override_user
        fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
        fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

        try:
            with patch(
                "tldw_Server_API.app.core.Storage.get_storage_backend",
                return_value=mock_storage
            ):
                with TestClient(fastapi_app) as client:
                    response = client.get(
                        f"/api/v1/media/{media_id}/file",
                        headers={"If-None-Match": '"abc123def456"'}
                    )

                    assert response.status_code == 304
                    assert response.content == b""
        finally:
            fastapi_app.dependency_overrides.pop(get_request_user, None)
            fastapi_app.dependency_overrides.pop(get_auth_principal, None)
            fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)


class TestHeadMediaFile:
    """Tests for HEAD /api/v1/media/{id}/file endpoint."""

    @pytest.mark.integration
    def test_head_returns_headers_without_body(self, tmp_path, mock_storage):
        """Test that HEAD returns headers but no body."""
        from tldw_Server_API.app.main import app as fastapi_app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

        db_path = tmp_path / "media.db"
        seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        media_id = create_test_media(seed_db, title="Test PDF", content="Content")
        seed_db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path="user1/media/1/test.pdf",
            original_filename="test.pdf",
            file_size=5000,
            mime_type="application/pdf",
        )
        seed_db.close_connection()

        async def _override_user() -> User:
            return User(id=1, username="tester", email=None, is_active=True)

        async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
            override_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
            try:
                yield override_db
            finally:
                override_db.close_connection()

        fastapi_app.dependency_overrides[get_request_user] = _override_user
        fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
        fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

        try:
            with patch(
                "tldw_Server_API.app.core.Storage.get_storage_backend",
                return_value=mock_storage
            ):
                with TestClient(fastapi_app) as client:
                    response = client.head(f"/api/v1/media/{media_id}/file")

                    assert response.status_code == 200
                    assert response.headers["Content-Type"] == "application/pdf"
                    assert response.headers["Content-Length"] == "25"  # From mock
                    assert response.headers["Accept-Ranges"] == "bytes"
                    # HEAD should have empty body
                    assert response.content == b""
        finally:
            fastapi_app.dependency_overrides.pop(get_request_user, None)
            fastapi_app.dependency_overrides.pop(get_auth_principal, None)
            fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)

    @pytest.mark.integration
    def test_head_404_for_missing_media(self, tmp_path):
        """Test that HEAD returns 404 for non-existent media."""
        from tldw_Server_API.app.main import app as fastapi_app
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

        db_path = tmp_path / "media.db"
        seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        seed_db.close_connection()

        async def _override_user() -> User:
            return User(id=1, username="tester", email=None, is_active=True)

        async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
            override_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
            try:
                yield override_db
            finally:
                override_db.close_connection()

        fastapi_app.dependency_overrides[get_request_user] = _override_user
        fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
        fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

        try:
            with TestClient(fastapi_app) as client:
                response = client.head("/api/v1/media/99999/file")
                assert response.status_code == 404
        finally:
            fastapi_app.dependency_overrides.pop(get_request_user, None)
            fastapi_app.dependency_overrides.pop(get_auth_principal, None)
            fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)
