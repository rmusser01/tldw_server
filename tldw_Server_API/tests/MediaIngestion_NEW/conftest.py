"""
Media Ingestion Module Test Configuration and Fixtures

Provides fixtures for testing media ingestion functionality including
file validation, transcription providers, and processing pipelines.
"""

import os
import tempfile
import shutil
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional, Tuple
from unittest.mock import MagicMock, AsyncMock, Mock
from datetime import datetime
import uuid
import wave
import struct

import pytest
from fastapi.testclient import TestClient
# pydub is optional for these tests; avoid importing to prevent hard dependency
import numpy as np

# Import actual components for integration tests
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import (
    ValidationResult,
    FileValidationError
)

# =====================================================================
# Test Markers
# =====================================================================

def pytest_configure(config):

    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "unit: Unit tests with minimal mocking")
    config.addinivalue_line("markers", "integration: Integration tests with real components")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Tests that take > 1 second")
    config.addinivalue_line("markers", "requires_ffmpeg: Tests requiring FFmpeg")
    config.addinivalue_line("markers", "requires_whisper: Tests requiring Whisper model")
    config.addinivalue_line("markers", "parakeet_mlx: Parakeet MLX variant tests")
    config.addinivalue_line("markers", "parakeet_onnx: Parakeet ONNX variant tests")
    config.addinivalue_line("markers", "nemo: Nemo transcription tests")
    config.addinivalue_line("markers", "streaming: Streaming transcription tests")

# =====================================================================
# Environment Configuration
# =====================================================================

@pytest.fixture
def test_env_vars():
    """Set up test environment variables."""
    original_env = os.environ.copy()

    # Set test mode
    os.environ["TEST_MODE"] = "true"
    # Force single-user mode and deterministic API key so auth matches headers
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = os.getenv("SINGLE_USER_TEST_API_KEY", "test-api-key-12345")
    os.environ["TRANSCRIPTION_PROVIDER"] = "whisper"
    os.environ["MAX_FILE_SIZE"] = "100000000"  # 100MB for tests

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

# =====================================================================
# Database Fixtures
# =====================================================================

@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        db_path = Path(temp_dir) / "test_media.db"
        yield db_path

@pytest.fixture
def media_database(temp_db_path) -> Generator[MediaDatabase, None, None]:
    """Create a real MediaDatabase instance for testing."""
    db = MediaDatabase(
        db_path=str(temp_db_path),
        client_id="test_client"
    )
    db.initialize_db()
    try:
        yield db
    finally:
        db.close_connection()

@pytest.fixture
def populated_media_db(media_database) -> MediaDatabase:
    """Create a MediaDatabase with test data."""
    # Add test media items
    from uuid import uuid4

    test_items = [
        {
            "media_id": str(uuid4()),
            "title": "Test Video",
            "content": "This is a test video transcription.",
            "media_type": "video",
            "url": "http://example.com/video.mp4",
            "ingestion_date": datetime.now().isoformat()
        },
        {
            "media_id": str(uuid4()),
            "title": "Test Audio",
            "content": "This is a test audio transcription.",
            "media_type": "audio",
            "url": "http://example.com/audio.mp3",
            "ingestion_date": datetime.now().isoformat()
        },
        {
            "media_id": str(uuid4()),
            "title": "Test Document",
            "content": "This is a test document content.",
            "media_type": "document",
            "url": "http://example.com/document.pdf",
            "ingestion_date": datetime.now().isoformat()
        }
    ]

    for item in test_items:
        # Map to new DB API (add_media_with_keywords)
        media_database.add_media_with_keywords(
            url=item.get("url"),
            title=item.get("title"),
            media_type=item.get("media_type"),
            content=item.get("content"),
            author="Test",
            ingestion_date=item.get("ingestion_date"),
            overwrite=False,
            keywords=[],
        )

    return media_database

# =====================================================================
# Test Media File Fixtures
# =====================================================================

@pytest.fixture
def test_media_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test media files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        media_dir = Path(temp_dir)
        yield media_dir

@pytest.fixture
def test_audio_file(test_media_dir) -> Path:
    """Create a test audio file (WAV format)."""
    audio_path = test_media_dir / "test_audio.wav"

    # Generate a simple sine wave audio
    sample_rate = 44100
    duration = 2  # seconds
    frequency = 440  # A4 note

    # Generate samples
    t = np.linspace(0, duration, int(sample_rate * duration))
    samples = np.sin(2 * np.pi * frequency * t)

    # Convert to 16-bit PCM
    samples = (samples * 32767).astype(np.int16)

    # Write WAV file
    with wave.open(str(audio_path), 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())

    return audio_path

@pytest.fixture
def test_video_file(test_media_dir) -> Path:
    """Create a test video file (mock)."""
    # For testing, we'll create a dummy file
    # Real video generation would require ffmpeg
    video_path = test_media_dir / "test_video.mp4"

    # Create a dummy MP4 file with minimal header
    # This is just for file validation tests
    with open(video_path, 'wb') as f:
        # Write a minimal MP4 header (ftyp box)
        f.write(b'\x00\x00\x00\x20ftypisom\x00\x00\x02\x00')
        f.write(b'isomiso2mp41\x00\x00\x00\x08free')
        # Add some dummy data
        f.write(b'\x00' * 1024)

    return video_path

@pytest.fixture
def test_pdf_file(test_media_dir) -> Path:
    """Create a test PDF file."""
    pdf_path = test_media_dir / "test_document.pdf"

    # Create a minimal valid PDF
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF Document) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
365
%%EOF"""

    with open(pdf_path, 'wb') as f:
        f.write(pdf_content)

    return pdf_path

@pytest.fixture
def test_text_file(test_media_dir) -> Path:
    """Create a test text file."""
    text_path = test_media_dir / "test_document.txt"

    with open(text_path, 'w', encoding='utf-8') as f:
        f.write("""This is a test document.
        It contains multiple lines.
        And some special characters: áéíóú ñ €.

        It also has multiple paragraphs.

        This is the final paragraph.""")

    return text_path

@pytest.fixture
def malicious_file(test_media_dir) -> Path:
    """Create a file with potentially malicious content for security testing."""
    mal_path = test_media_dir / "malicious.txt"

    with open(mal_path, 'w') as f:
        f.write("<?php system($_GET['cmd']); ?>")
        f.write("\n<script>alert('XSS')</script>")
        f.write("\n../../etc/passwd")

    return mal_path

# =====================================================================
# Mock Fixtures for Unit Tests
# =====================================================================

@pytest.fixture
def mock_media_db():
    """Mock MediaDatabase for unit tests."""
    mock_db = MagicMock(spec=MediaDatabase)

    mock_db.add_media.return_value = 1
    mock_db.get_media.return_value = {
        "id": 1,
        "media_id": "test-media-123",
        "title": "Test Media",
        "content": "Test content"
    }
    mock_db.search_media_items.return_value = []

    return mock_db

@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for unit tests."""
    mock_model = MagicMock()

    mock_model.transcribe.return_value = {
        "text": "This is a test transcription.",
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "This is a test transcription."
            }
        ],
        "language": "en"
    }

    return mock_model

@pytest.fixture
def mock_ffmpeg():
    """Mock FFmpeg operations for unit tests."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = b"Output from ffmpeg"
        yield mock_run

# =====================================================================
# Request/Response Fixtures
# =====================================================================

@pytest.fixture
def basic_media_request() -> Dict[str, Any]:
    """Basic media processing request."""
    return {
        "url": "http://example.com/media.mp4",
        "title": "Test Media",
        "media_type": "video",
        "chunk_method": "tokens",
        "max_chunk_size": 500
    }

@pytest.fixture
def audio_process_request() -> Dict[str, Any]:
    """Audio processing request."""
    return {
        "title": "Test Audio",
        "author": "Test Author",
        "audio_urls": ["http://example.com/audio.mp3"],
        "diarize": False,
        "transcribe": True,
        "chunk_method": "sentences",
        "max_chunk_size": 1000
    }

@pytest.fixture
def document_process_request() -> Dict[str, Any]:
    """Document processing request."""
    return {
        "title": "Test Document",
        "author": "Test Author",
        "content": "This is test document content that needs to be processed.",
        "chunk_method": "words",
        "max_chunk_size": 200,
        "language": "en"
    }

# =====================================================================
# Transcription Provider Fixtures
# =====================================================================

@pytest.fixture
def parakeet_mlx_config():
    """Configuration for Parakeet MLX provider."""
    return {
        "model_name": "parakeet-mlx-small",
        "device": "cpu",
        "batch_size": 8,
        "language": "en"
    }

@pytest.fixture
def parakeet_onnx_config():
    """Configuration for Parakeet ONNX provider."""
    return {
        "model_path": "/models/parakeet-onnx",
        "providers": ["CPUExecutionProvider"],
        "batch_size": 4,
        "language": "en"
    }

@pytest.fixture
def nemo_config():
    """Configuration for Nemo provider."""
    return {
        "model_name": "QuartzNet15x5Base-En",
        "device": "cpu",
        "batch_size": 1,
        "language": "en"
    }

# =====================================================================
# Validation Result Fixtures
# =====================================================================

@pytest.fixture
def valid_validation_result():
    """A valid file validation result."""
    return ValidationResult(
        is_valid=True,
        issues=[],
        file_path=Path("/tmp/valid_file.txt"),  # nosec B108
        detected_mime_type="text/plain",
        detected_extension=".txt"
    )

@pytest.fixture
def invalid_validation_result():
    """An invalid file validation result."""
    return ValidationResult(
        is_valid=False,
        issues=["File type not allowed", "File size exceeds limit"],
        file_path=Path("/tmp/invalid_file.exe"),  # nosec B108
        detected_mime_type="application/x-executable",
        detected_extension=".exe"
    )

# =====================================================================
# Chunk Processing Fixtures
# =====================================================================

@pytest.fixture
def sample_text_for_chunking():
    """Sample text for testing chunking strategies."""
    return """This is the first sentence. This is the second sentence.

    This is a new paragraph with multiple sentences. It contains important information
    that should be preserved during chunking. The chunking algorithm needs to handle
    this properly.

    Here's another paragraph. It's shorter. But still important.

    Final paragraph with concluding remarks. The chunking should maintain context."""

@pytest.fixture
def expected_chunks():
    """Expected chunks for different strategies."""
    return {
        "sentences": [
            "This is the first sentence.",
            "This is the second sentence.",
            "This is a new paragraph with multiple sentences.",
            "It contains important information that should be preserved during chunking.",
            "The chunking algorithm needs to handle this properly.",
            "Here's another paragraph.",
            "It's shorter.",
            "But still important.",
            "Final paragraph with concluding remarks.",
            "The chunking should maintain context."
        ],
        "words": [
            # Chunks by word count would depend on max_chunk_size
        ],
        "tokens": [
            # Token-based chunks would depend on tokenizer
        ]
    }

# =====================================================================
# API Client Fixtures
# =====================================================================

@pytest.fixture
def test_client(test_env_vars):
    """Create a test client for the FastAPI app."""
    from tldw_Server_API.app.main import app
    with TestClient(app) as client:
        yield client

@pytest.fixture
def auth_headers(test_env_vars):
    """Authentication headers for API requests.

    - If AUTH_MODE=single_user: use X-API-KEY from settings directly.
    - If AUTH_MODE=multi_user: provision a test user and create a real API key
      via AuthNZ services, then return it as X-API-KEY. This ensures routes that
      depend on get_request_user (multi-user path) authenticate correctly.
    """
    import asyncio
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    import os
    from pathlib import Path
    import tempfile
    s = get_settings()
    # Fast path: single-user mode uses configured API key
    if s.AUTH_MODE == "single_user":
        return {"X-API-KEY": s.SINGLE_USER_API_KEY}

    async def _provision_multi_user_headers() -> dict:
        # Ensure Users DB is initialized and create a test admin user if needed
        from tldw_Server_API.app.core.DB_Management.Users_DB import get_users_db, reset_users_db
        from tldw_Server_API.app.core.AuthNZ.password_service import get_password_service
        from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager, reset_api_key_manager
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        # Proactively use a temporary SQLite AuthNZ DB in multi_user tests to avoid
        # external Postgres requirements. This still exercises the multi_user auth path.
        try:
            from urllib.parse import urlparse
            scheme = (urlparse(s.DATABASE_URL).scheme or "").lower()
        except Exception:
            scheme = ""
        if scheme.startswith("postgres"):
            tmp_dir = Path(tempfile.mkdtemp(prefix="tldw_authnz_"))
            sqlite_path = tmp_dir / "users.db"
            os.environ["DATABASE_URL"] = f"sqlite:///{sqlite_path}"
            # Reset settings and pools to pick up the override
            reset_settings()
            await reset_db_pool()
            await reset_api_key_manager()
            await reset_users_db()

        users_db = await get_users_db()
        username = "media_ingest_test_admin"
        email = "media_ingest_test_admin@example.com"
        user = await users_db.get_user_by_username(username)
        if not user:
            # Create a strong password that meets policy; we never log or use it for login
            pw_service = get_password_service()
            pw_hash = pw_service.hash_password("Str0ng!MediaIngestTest1")
            user = await users_db.create_user(
                username=username,
                email=email,
                password_hash=pw_hash,
                role="admin",
                is_active=True,
                is_superuser=True,
            )

        # Issue a real API key for this user (admin scope)
        api_mgr = await get_api_key_manager()
        key_info = await api_mgr.create_api_key(
            user_id=int(user["id"] if isinstance(user, dict) else user.id),
            name="media_ingest_test",
            description="Temporary key for MediaIngestion_NEW tests",
            scope="admin",
            expires_in_days=7,
        )
        return {"X-API-KEY": key_info["key"]}

    # Create key synchronously for sync tests
    return asyncio.run(_provision_multi_user_headers())

# =====================================================================
# Cleanup Fixtures
# =====================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Any cleanup code here
    import gc
    gc.collect()
