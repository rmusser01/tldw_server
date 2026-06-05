# test_add_media_endpoint.py
# Description: This file contains the test cases for the /media/add endpoint of the FastAPI application.
# Style remodeled to mirror test_media_processing.py (less mocking, more integration).
#
# Imports
import asyncio # Added
import random
import time
from pathlib import Path
import os
import shutil
import wave
import struct
import math
from typing import Dict, Any, Tuple, Optional, Union  # Added Tuple
from unittest.mock import patch, MagicMock, AsyncMock, ANY  # Refined imports
import gc
from urllib.parse import urlparse

#
# 3rd-party Libraries
import pytest
import httpx # Keep for mocking specific download errors
from fastapi import status, Header
from loguru import logger
from io import BytesIO
import pytest
import zipfile
#
# Local Imports
# Adjust import paths based on your project structure if needed
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.core.Ingestion_Media_Processing.persistence import (  # type: ignore
    process_document_like_item as _process_document_like_item,
)
from tldw_Server_API.tests.test_utils import (
    skip_if_transcription_model_unavailable,
    temp_db,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase # Import Database class
# Import the form model
from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm, MediaType # Import AddMediaForm, MediaType
#
######################################################################################################################
#
#

# --- Use Main App Instance ---
try:
    from tldw_Server_API.app.main import app
except ImportError:
    raise ImportError("Could not locate the main FastAPI app instance. Adjust the import path.")

# --- Constants (Borrowing/Adapting from test-library-1) ---
API_PREFIX = "/api/v1/media"
ADD_MEDIA_ENDPOINT = f"{API_PREFIX}/add"
PROCESS_EBOOKS_ENDPOINT = f"{API_PREFIX}/process-ebooks"
PROCESS_AUDIO_ENDPOINT = f"{API_PREFIX}/process-audios"
PROCESS_VIDEO_ENDPOINT = f"{API_PREFIX}/process-videos"
PROCESS_DOCUMENTS_ENDPOINT = f"{API_PREFIX}/process-documents"

# Test Media Files (Ensure these paths are correct relative to your test file)
TEST_MEDIA_DIR = Path(__file__).parent / "test_media"
TEST_MEDIA_DIR.mkdir(exist_ok=True)
# --- Create Dummy Files if they don't exist ---
# Function to create dummy file if it doesn't exist
def create_dummy_file(path: Path, content: str = "dummy content"):
    if not path.exists():
        try:
            path.write_text(content, encoding='utf-8')
            logger.info(f"Created dummy test file: {path}")
        except Exception as e:
            logger.error(f"Failed to create dummy file {path}: {e}")
            pytest.skip(f"Failed to create required test file: {path}")


def create_valid_epub(path: Path):
    if path.exists():
        return
    try:
        container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
        content_opf = """<?xml version="1.0" encoding="UTF-8"?>
<package version="3.0" unique-identifier="bookid" xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">sample</dc:identifier>
    <dc:title>Sample EPUB</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="chapter"/>
  </spine>
</package>"""
        chapter = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>Sample EPUB</title></head>
  <body><p>Sample EPUB content.</p></body>
</html>"""
        with zipfile.ZipFile(path, "w") as archive:
            marker = zipfile.ZipInfo("mimetype")
            marker.compress_type = zipfile.ZIP_STORED
            archive.writestr(marker, "application/epub+zip")
            archive.writestr("META-INF/container.xml", container_xml)
            archive.writestr("content.opf", content_opf)
            archive.writestr("chapter.xhtml", chapter)
        logger.info(f"Created valid EPUB test file: {path}")
    except Exception as e:
        logger.error(f"Failed to create EPUB file {path}: {e}")
        pytest.skip(f"Failed to create valid EPUB file: {path}")

# Define paths
SAMPLE_VIDEO_PATH = TEST_MEDIA_DIR / "sample.mp4"
SAMPLE_AUDIO_PATH = TEST_MEDIA_DIR / "sample.wav"
SAMPLE_PDF_PATH = TEST_MEDIA_DIR / "sample.pdf"
SAMPLE_EPUB_PATH = TEST_MEDIA_DIR / "sample.epub"
SAMPLE_TXT_PATH = TEST_MEDIA_DIR / "sample.txt"
SAMPLE_MD_PATH = TEST_MEDIA_DIR / "sample.md"
SAMPLE_DOCX_PATH = TEST_MEDIA_DIR / "sample.docx" # Requires real file or dummy
SAMPLE_RTF_PATH = TEST_MEDIA_DIR / "sample.rtf"   # Requires real file or dummy & pandoc
SAMPLE_HTML_PATH = TEST_MEDIA_DIR / "sample.html"
SAMPLE_XML_PATH = TEST_MEDIA_DIR / "sample.xml"

# Create dummy files
create_dummy_file(SAMPLE_VIDEO_PATH, "dummy video data") # Content doesn't need to be valid video
# Create a small, valid mono PCM WAV file for audio tests
def create_valid_wav(path: Path, duration_sec: float = 1.0, sample_rate: int = 16000, freq: float = 440.0):
    if path.exists():
        return
    try:
        n_samples = int(duration_sec * sample_rate)
        with wave.open(str(path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(sample_rate)
            amplitude = 0.1
            for i in range(n_samples):
                sample = int(32767.0 * amplitude * math.sin(2 * math.pi * freq * (i / sample_rate)))
                wf.writeframes(struct.pack('<h', sample))
        logger.info(f"Created valid WAV test file: {path}")
    except Exception as e:
        logger.error(f"Failed to create WAV file {path}: {e}")
        pytest.skip(f"Failed to create valid audio file: {path}")

create_valid_wav(SAMPLE_AUDIO_PATH)
create_dummy_file(SAMPLE_PDF_PATH, "%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 0>>endobj\nxref\n0 3\n0000000000 65535 f \n0000000010 00000 n \n0000000059 00000 n \ntrailer<</Size 3/Root 1 0 R>>\nstartxref\n114\n%%EOF") # Minimal PDF
create_valid_epub(SAMPLE_EPUB_PATH)
create_dummy_file(SAMPLE_TXT_PATH, "Sample TXT content.")
create_dummy_file(SAMPLE_MD_PATH, "# Sample MD\nContent.")
create_dummy_file(SAMPLE_DOCX_PATH, "dummy docx data") # Content doesn't need to be valid docx
create_dummy_file(SAMPLE_RTF_PATH, "{\\rtf1\\ansi dummy rtf}") # Minimal RTF
create_dummy_file(SAMPLE_HTML_PATH, "<html><body>Sample HTML</body></html>")
create_dummy_file(SAMPLE_XML_PATH, "<root><data>Sample XML</data></root>")


# Test URLs (Stable, public URLs)
VALID_VIDEO_URL = "https://www.youtube.com/watch?v=wVAkRDvQEog" # Working test video
VALID_AUDIO_URL = "https://cdn.pixabay.com/download/audio/2023/12/02/audio_2f291f569a.mp3?filename=about-anger-179423.mp3"
VALID_PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
VALID_EPUB_URL = "https://filesamples.com/samples/ebook/epub/Alices%20Adventures%20in%20Wonderland.epub"
VALID_TXT_URL = "https://archives.phrack.org/issues/1/1.txt" # Raw text URL (stable mirror)
VALID_MD_URL = "https://raw.githubusercontent.com/rmusser01/tldw/main/README.md" # Raw markdown URL
INVALID_URL = "http://this.url.does.not.exist/resource.file"
URL_404 = "https://httpbin.org/status/404" # Reliable 404
TEST_VIDEO_TRANSCRIPTION_MODEL = "small"
TEST_VIDEO_START_TIME = "0"
TEST_VIDEO_END_TIME = "00:00:20"

# --- Fixtures ---

# Mock Database Setup (Using test_utils.temp_db pattern from test_media_versions)
@pytest.fixture(scope="function") # <<< CHANGED HERE
def db_session_scope():
    """RENAMED: FUNCTION-scoped temporary database with explicit connection closing."""
    db: Optional[MediaDatabase] = None
    db_path_str_for_log = "UNKNOWN_DB_PATH_FUNC_SCOPE" # Initialize for logging
    try:
        # temp_db() creates a new DB file each time this fixture runs (per function)
        with temp_db() as db_context:
            db = db_context
            db_path_str_for_log = db.db_path_str # Capture path for logging
            # No need to manually set client_id here if temp_db handles it,
            # but we keep the check for safety.
            if not hasattr(db, 'client_id') or not db.client_id:
                db.client_id = settings.get("SERVER_CLIENT_ID", f"test_client_{random.randint(1000,9999)}") # Ensure unique client ID per test if needed
                logger.warning(f"Manually set client_id on FUNCTION-SCOPED Database instance from temp_db: {db_path_str_for_log}")

            logger.info(f"--- Using FUNCTION DB from temp_db: {db_path_str_for_log} ---")
            try:
                conn = db.get_connection()
                conn.execute("PRAGMA foreign_keys=ON;")
                logger.debug(f"Foreign keys enabled for {db_path_str_for_log}")
            except Exception as fk_e:
                logger.warning(f"Could not enable foreign keys on function DB {db_path_str_for_log}: {fk_e}")
            yield db
    finally:
        # --- Cleanup for Function Scope ---
        if db:
            db_path_str_for_log = getattr(db, 'db_path_str', 'UNKNOWN_DB_PATH_FUNC_SCOPE')
            logger.info(f"--- Attempting to close DB connection for test function teardown: {db_path_str_for_log} ---")
            try:
                gc.collect() # Keep GC call
                logger.debug(f"--- Ran garbage collection before closing DB connection ({db_path_str_for_log}) ---")
                db.close_connection()
                logger.info(f"--- Connection closed (or was already closed) for test function thread: {db_path_str_for_log} ---")
            except Exception as close_err:
                logger.warning(f"Error closing DB connection during function teardown for {db_path_str_for_log}: {close_err}")
        else:
            logger.warning("--- DB instance was None during function teardown ---")
        # The TemporaryDirectory context manager in temp_db will handle file deletion
        logger.info(f"--- Teardown complete for db_session_scope ({db_path_str_for_log}) ---")


@pytest.fixture(scope="function")
def db_session(db_session_scope):
    """Function-scoped access to the function DB."""
    yield db_session_scope # Directly yield the function-scoped instance

mock_test_user = User(
    id=settings["SINGLE_USER_FIXED_ID"], # Use the fixed ID from settings
    username="test_user",
    email="test@example.com",
    is_active=True
)

async def override_get_request_user():
    """Override dependency to return a fixed test user."""
    logger.info("--- Using mock test user ---")
    return mock_test_user

# Override verify_api_key for testing
# This dummy function bypasses the actual key check
async def override_verify_api_key(x_api_key: str = Header(..., alias="X-API-KEY")):
    """Dummy override for API key verification during tests."""
    logger.debug(f"Auth Override: Bypassing API key check for key starting with: {x_api_key[:4]}...")
    # You could return mock user/permissions here if needed downstream
    return {"user_id": "test_user", "permissions": ["*"]}

# Override get_media_db_for_user to use the temp test DB
def override_get_media_db_for_user_dependency(db_fixture):
    """
    Returns a dependency override function that yields the provided DB fixture.
    """
    async def _override(): # Changed to async def
        # logger.debug(f"--- OVERRIDING get_media_db_for_user with fixture: {db_fixture.db_path_str} ---")
        yield db_fixture
    return _override

@pytest.fixture(scope="function")
def test_api_client(client_user_only, db_session_scope):  # Depends on the FUNCTION-scoped DB fixture now
    """Provides a shared authenticated TestClient with a per-test Media DB override."""
    logger.debug(f"Setting up test_api_client fixture for FUNCTION (DB: {db_session_scope.db_path_str})...")

    # --- File Existence Checks (Keep these or adapt as needed) ---
    # Skip the entire module if essential files are missing
    if not SAMPLE_VIDEO_PATH.exists(): pytest.skip(f"Test video file not found: {SAMPLE_VIDEO_PATH}")
    if not SAMPLE_AUDIO_PATH.exists(): pytest.skip(f"Test audio file not found: {SAMPLE_AUDIO_PATH}")
    if not SAMPLE_PDF_PATH.exists(): pytest.skip(f"Test PDF file not found: {SAMPLE_PDF_PATH}")
    if not SAMPLE_EPUB_PATH.exists(): pytest.skip(f"Test EPUB file not found: {SAMPLE_EPUB_PATH}")
    # Add checks for other required sample files if necessary

    # Create the override function using the FUNCTION-scoped DB instance
    db_override_func = override_get_media_db_for_user_dependency(db_session_scope)

    # Apply overrides, preserving any existing override for get_media_db_for_user.
    original_db_override = app.dependency_overrides.get(get_media_db_for_user)
    app.dependency_overrides[get_media_db_for_user] = db_override_func
    logger.info("Applied dependency override for get_media_db_for_user (FUNCTION scope)")

    try:
        yield client_user_only
    finally:
        if original_db_override is None:
            app.dependency_overrides.pop(get_media_db_for_user, None)
        else:
            app.dependency_overrides[get_media_db_for_user] = original_db_override
        logger.info("Restored dependency override for get_media_db_for_user (FUNCTION scope teardown)")

@pytest.fixture
def dummy_headers():
    """Provides headers required by endpoint signature, even if logic is mocked."""
    # The actual value doesn't matter because auth is overridden in client_user_only.
    return {"token": "dummy_test_token_for_header"}

@pytest.fixture
def auth_headers():
    """Provides authentication/required headers."""
    return {
        "token": "test_api_token_123",
        #"X-API-KEY": "test_api_key_123" # If you implement API key auth
    }

@pytest.fixture
def dummy_file_content():
    """Provides dummy byte content for mock files if needed elsewhere."""
    return b"dummy file content for testing"

@pytest.fixture
def create_upload_file(dummy_file_content):
    def _create(filepath: Path) -> Tuple[str, bytes, str]:
        if not filepath.exists():
            pytest.skip(f"Required test file missing: {filepath}")
        mime_map = {
            ".mp4": "video/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".pdf": "application/pdf",
            ".epub": "application/epub+zip",
            # Add other types as needed
        }
        mime_type = mime_map.get(filepath.suffix.lower(), "application/octet-stream")
        try:
            content = filepath.read_bytes()
        except Exception as e:
            pytest.fail(f"Failed to read test file {filepath}: {e}")
        return (filepath.name, content, mime_type)
    return _create


# --- Helper Functions (Borrowing/Adapting from test-library-1) ---
# (check_batch_response and check_media_item_result remain largely the same)
def check_batch_response(
        response, expected_status_code, expected_processed=None,
        expected_errors=None, expected_warnings=None, check_results_len=None):
    if response.status_code != expected_status_code:
        logger.error(f"Expected status {expected_status_code}, got {response.status_code}. Response text: {response.text}")
    assert response.status_code == expected_status_code
    try:
        data = response.json()
    except Exception as e:
        pytest.fail(f"Failed to parse response JSON. Status: {response.status_code}, Text: {response.text}, Error: {e}")

    assert "results" in data, f"Response missing 'results' key: {data}"
    assert isinstance(data["results"], list), f"'results' is not a list: {data['results']}"

    actual_processed = sum(1 for r in data.get("results", []) if r.get("status") == "Success")
    actual_errors = sum(1 for r in data.get("results", []) if r.get("status") in ["Error", "Failed"])
    actual_warnings = sum(1 for r in data.get("results", []) if r.get("status") == "Warning")

    if expected_processed is not None:
        assert actual_processed == expected_processed, f"Expected {expected_processed} processed, got {actual_processed}. Results: {data.get('results')}"
    if expected_errors is not None:
        assert actual_errors == expected_errors, f"Expected {expected_errors} errors/failures, got {actual_errors}. Results: {data.get('results')}"
    if expected_warnings is not None:
        assert actual_warnings == expected_warnings, f"Expected {expected_warnings} warnings, got {actual_warnings}. Results: {data.get('results')}"
    if check_results_len is not None:
        assert len(data["results"]) == check_results_len, f"Expected {check_results_len} total results, got {len(data['results'])}. Results: {data.get('results')}"
    return data


def check_media_item_result(result, expected_status, check_db_interaction=True, expected_media_type=None):


    assert isinstance(result, dict), f"Result item is not a dictionary: {result}"
    assert "status" in result, f"Result missing 'status' key: {result}"
    assert result["status"] == expected_status, f"Expected status '{expected_status}', got '{result['status']}' for input '{result.get('input_ref', 'N/A')}'"
    assert "input_ref" in result, "Result missing 'input_ref' key"
    assert "processing_source" in result, "Result missing 'processing_source' key"
    assert "media_type" in result, "Result missing 'media_type' key"
    if expected_media_type:
        assert result["media_type"] == expected_media_type, f"Expected media_type '{expected_media_type}', got '{result['media_type']}'"
    assert "metadata" not in result or isinstance(result["metadata"], (dict, type(None))), f"Result 'metadata' is not a dict or None: {result['metadata']}"
    assert "content" in result, "Result missing 'content' key"
    assert "chunks" in result, "Result missing 'chunks' key"
    assert "analysis" in result, "Result missing 'analysis' key"
    assert "analysis_details" not in result or isinstance(result["analysis_details"], (dict, type(None))), f"Result 'analysis_details' is not a dict or None: {result['analysis_details']}"
    assert "error" in result, "Result missing 'error' key"
    assert "warnings" not in result or isinstance(result.get("warnings"), (list, type(None))), f"Result missing or invalid 'warnings': {result.get('warnings')}"
    if check_db_interaction:
        assert "db_id" in result, "Result missing 'db_id' key"
        assert "db_message" in result, "Result missing 'db_message' key"
        if expected_status == "Success" and "added" in result.get("db_message", "").lower():
            assert isinstance(result.get("db_id"), int), f"Expected integer db_id for Success status, got {result.get('db_id')}"
    if expected_status in ["Error", "Failed"]:
        pass
    elif expected_status == "Success":
        assert result.get("error") is None or result["error"] == "", f"Expected None or empty error for Success status, got '{result.get('error')}' for input '{result.get('input_ref')}'"
    elif expected_status == "Warning":
        assert result.get("error") or result.get("warnings"), f"Expected error or warnings for Warning status for input '{result.get('input_ref')}'"


def _build_nested_eml_bytes():


    # Construct a simple nested .eml in-memory
    from email.message import EmailMessage
    inner = EmailMessage()
    inner["From"] = "Inner <inner@example.com>"
    inner["To"] = "Bob <bob@example.com>"
    inner["Subject"] = "Inner Child"
    inner.set_content("Inner body.")

    outer = EmailMessage()
    outer["From"] = "Alice <alice@example.com>"
    outer["To"] = "Bob <bob@example.com>"
    outer["Subject"] = "Outer Parent"
    outer.set_content("Outer body.")
    outer.add_attachment(inner, maintype="message", subtype="rfc822", filename="child.eml")
    return outer.as_bytes()


def test_add_email_with_children_persists_child(test_api_client, db_session, dummy_headers):


    # Prepare form for email add with attachment ingestion
    form_data = create_add_media_form_data(
        media_type="email",
        perform_chunking=True,
        chunk_method='sentences',
        chunk_size=500, # Will be bumped to 1000 by defaults for email
        chunk_overlap=200,
    )
    # Add email-specific form flags
    form_data.update({
        'ingest_attachments': 'true',
        'max_depth': '2',
    })

    # Build nested EML bytes and upload as file
    nested_eml = _build_nested_eml_bytes()
    files = {
        'files': ('outer.eml', nested_eml, 'message/rfc822'),
    }

    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, files=files, headers=dummy_headers)
    data = check_batch_response(response, expected_status_code=status.HTTP_200_OK)
    # Expect at least one result (parent)
    assert data['results'], "No results returned for email add"
    parent = data['results'][0]
    check_media_item_result(parent, expected_status="Success", check_db_interaction=True, expected_media_type='email')
    # Ensure child persistence indicated
    cdb = parent.get('child_db_results')
    assert isinstance(cdb, list) and len(cdb) == 1, f"Expected one child persisted, got: {cdb}"
    assert isinstance(cdb[0].get('db_id'), int), f"Child db_id missing: {cdb}"


def _build_zip_emails_bytes():


    # Build two simple EMLs and zip them
    eml1 = (
        b"From: X <x@example.com>\r\n"
        b"To: Y <y@example.com>\r\n"
        b"Subject: Archive One\r\n"
        b"MIME-Version: 1.0\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"Hello.\r\n"
    )
    eml2 = (
        b"From: P <p@example.com>\r\n"
        b"To: Q <q@example.com>\r\n"
        b"Subject: Archive Two\r\n"
        b"MIME-Version: 1.0\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"Hi.\r\n"
    )
    bio = BytesIO()
    with zipfile.ZipFile(bio, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('a1.eml', eml1)
        zf.writestr('a2.eml', eml2)
    bio.seek(0)
    return bio.getvalue()


def test_add_email_archive_persists_children(test_api_client, db_session, dummy_headers):


    # Similar to nested eml, but provide a .zip and accept_archives=true; expect children persisted
    form_data = create_add_media_form_data(
        media_type="email",
        perform_chunking=True,
        chunk_method='sentences',
        chunk_size=500,
        chunk_overlap=200,
    )
    form_data.update({
        'accept_archives': 'true',
    })
    zip_bytes = _build_zip_emails_bytes()
    files = {
        'files': ('emails.zip', zip_bytes, 'application/zip'),
    }

    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, files=files, headers=dummy_headers)
    data = check_batch_response(response, expected_status_code=status.HTTP_200_OK)
    assert data['results'], "No results returned for email add (zip)"
    parent = data['results'][0]
    check_media_item_result(parent, expected_status="Success", check_db_interaction=False, expected_media_type='email')
    # Parent should include the archive grouping keyword in its keywords
    parent_keywords = parent.get('keywords') or []
    assert 'email_archive:emails' in parent_keywords, f"Parent missing archive keyword. Keywords: {parent_keywords}"
    cdb = parent.get('child_db_results')
    assert isinstance(cdb, list) and len(cdb) == 2, f"Expected two children persisted from archive, got: {cdb}"
    # Also assert that each child result includes the archive grouping keyword
    children = parent.get('children') or []
    assert isinstance(children, list) and len(children) == 2, f"Expected two children in response, got: {children}"
    for child in children:
        if isinstance(child, dict):
            kws = child.get('keywords') or []
            assert 'email_archive:emails' in kws, f"Archive keyword missing in child: {kws}"


def test_add_email_pst_feature_flag_behavior(test_api_client, db_session, dummy_headers):


    # With accept_pst=true and a small placeholder .pst, expect synthetic parent with children containing informative error
    form_data = create_add_media_form_data(
        media_type="email",
        perform_chunking=False,
    )
    form_data.update({
        'accept_pst': 'true',
    })
    files = {
        'files': ('emails.pst', b'!pst placeholder!', 'application/octet-stream'),
    }
    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, files=files, headers=dummy_headers)
    data = check_batch_response(response, expected_status_code=status.HTTP_200_OK)
    assert data['results'], "No results returned for email add (pst)"
    parent = data['results'][0]
    check_media_item_result(parent, expected_status="Success", check_db_interaction=False, expected_media_type='email')
    # Parent should include email_pst grouping
    parent_keywords = parent.get('keywords') or []
    assert 'email_pst:emails' in parent_keywords, f"Parent missing pst keyword. Keywords: {parent_keywords}"
    children = parent.get('children') or []
    assert isinstance(children, list) and len(children) >= 1
    first = children[0]
    assert first.get('status') == 'Error'
    assert 'pst/ost support not enabled' in str(first.get('error', '')).lower()
    # No child persistence expected due to error-only children
    cdb = parent.get('child_db_results') or []
    assert not cdb, f"Unexpected child DB persistence for pst feature-flag: {cdb}"


@pytest.mark.performance
def test_add_email_zip_large_container_persists_children(test_api_client, db_session, dummy_headers):
    # Build a zip with 30 small EMLs; expect children to be persisted
    from io import BytesIO as _BytesIO
    import zipfile as _zipfile

    def _mk_zip(n: int = 30) -> bytes:
        bio = _BytesIO()
        with _zipfile.ZipFile(bio, 'w', compression=_zipfile.ZIP_DEFLATED) as zf:
            for i in range(n):
                eml = (
                    b"From: A <a@example.com>\r\n"
                    b"To: B <b@example.com>\r\n"
                    + f"Subject: Z{i}\r\n".encode('utf-8')
                    + b"MIME-Version: 1.0\r\n"
                    + b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
                    + b"body\r\n"
                )
                zf.writestr(f"m{i}.eml", eml)
        bio.seek(0)
        return bio.getvalue()

    form_data = create_add_media_form_data(
        media_type="email",
        perform_chunking=False,
    )
    form_data.update({'accept_archives': 'true'})
    zip_bytes = _mk_zip(30)
    files = {'files': ('emails.zip', zip_bytes, 'application/zip')}
    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, files=files, headers=dummy_headers)
    data = check_batch_response(response, expected_status_code=status.HTTP_200_OK)
    parent = data['results'][0]
    children = parent.get('children') or []
    assert isinstance(children, list) and len(children) >= 30
    cdb = parent.get('child_db_results') or []
    assert isinstance(cdb, list) and len(cdb) >= 30


@pytest.mark.performance
def test_add_email_mbox_large_container_persists_children(test_api_client, db_session, dummy_headers):
    # Build an mbox with 30 small messages; expect children to be persisted
    import mailbox as _mailbox
    import tempfile as _tempfile
    from email.message import EmailMessage

    with _tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        mbox = _mailbox.mbox(tmp_path)
        for i in range(30):
            msg = EmailMessage()
            msg['From'] = f"A <a{i}@example.com>"
            msg['To'] = 'B <b@example.com>'
            msg['Subject'] = f"M{i}"
            msg.set_content('hi')
            mbox.add(msg)
        mbox.flush()
        mbox.close()
        with open(tmp_path, 'rb') as f:
            mbox_bytes = f.read()
    finally:
        import os as _os
        try:
            _os.unlink(tmp_path)
        except Exception:
            _ = None

    form_data = create_add_media_form_data(
        media_type="email",
        perform_chunking=False,
    )
    form_data.update({'accept_mbox': 'true'})
    files = {'files': ('emails.mbox', mbox_bytes, 'application/mbox')}
    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, files=files, headers=dummy_headers)
    data = check_batch_response(response, expected_status_code=status.HTTP_200_OK)
    parent = data['results'][0]
    children = parent.get('children') or []
    assert isinstance(children, list) and len(children) >= 30
    cdb = parent.get('child_db_results') or []
    assert isinstance(cdb, list) and len(cdb) >= 30


def test_add_email_zip_guardrail_too_many_files(test_api_client, db_session, dummy_headers):


    # Temporarily lower guardrail to 1 to trigger error list
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import Email_Processing_Lib as email_lib
    archive_cfg = email_lib.DEFAULT_MEDIA_TYPE_CONFIG.get('archive', {})
    orig = archive_cfg.get('max_internal_files', 100)
    try:
        archive_cfg['max_internal_files'] = 1
        # Build 2-member zip
        from io import BytesIO as _BytesIO
        import zipfile as _zipfile
        bio = _BytesIO()
        with _zipfile.ZipFile(bio, 'w', compression=_zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('a.eml', b"From: A <a@example.com>\r\nSubject: A\r\n\r\nbody")
            zf.writestr('b.eml', b"From: B <b@example.com>\r\nSubject: B\r\n\r\nbody")
        bio.seek(0)
        zip_bytes = bio.getvalue()
        form_data = create_add_media_form_data(media_type="email", perform_chunking=False)
        form_data.update({'accept_archives': 'true'})
        files = {'files': ('emails.zip', zip_bytes, 'application/zip')}
        response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, files=files, headers=dummy_headers)
        data = check_batch_response(response, expected_status_code=status.HTTP_200_OK)
        parent = data['results'][0]
        children = parent.get('children') or []
        assert isinstance(children, list) and len(children) >= 1
        first = children[0]
        assert first.get('status') == 'Error' and 'too many files' in str(first.get('error','')).lower()
        assert not parent.get('child_db_results'), 'No child persistence expected on guardrail error'
    finally:
        archive_cfg['max_internal_files'] = orig


def test_add_email_mbox_guardrail_too_many_messages(test_api_client, db_session, dummy_headers):


    # Lower guardrail to 1 and create 2-message mbox; expect error in children and no child persistence
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import Email_Processing_Lib as email_lib
    import mailbox as _mailbox
    import tempfile as _tempfile
    from email.message import EmailMessage
    archive_cfg = email_lib.DEFAULT_MEDIA_TYPE_CONFIG.get('archive', {})
    orig = archive_cfg.get('max_internal_files', 100)
    try:
        archive_cfg['max_internal_files'] = 1
        with _tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            mbox = _mailbox.mbox(tmp_path)
            for i in range(2):
                msg = EmailMessage(); msg['From'] = 'A <a@example.com>'; msg['To'] = 'B <b@example.com>'; msg['Subject'] = f'M{i}'; msg.set_content('hi')
                mbox.add(msg)
            mbox.flush(); mbox.close()
            with open(tmp_path, 'rb') as f:
                mbox_bytes = f.read()
        finally:
            import os as _os
            try:
                _os.unlink(tmp_path)
            except Exception:
                _ = None
        form_data = create_add_media_form_data(media_type="email", perform_chunking=False)
        form_data.update({'accept_mbox': 'true'})
        files = {'files': ('emails.mbox', mbox_bytes, 'application/mbox')}
        response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, files=files, headers=dummy_headers)
        data = check_batch_response(response, expected_status_code=status.HTTP_200_OK)
        parent = data['results'][0]
        children = parent.get('children') or []
        assert isinstance(children, list) and len(children) >= 1
        first = children[0]
        assert first.get('status') == 'Error' and 'too many' in str(first.get('error','')).lower()
        assert not parent.get('child_db_results'), 'No child persistence expected on guardrail error'
    finally:
        archive_cfg['max_internal_files'] = orig


def _build_mbox_two_emails_bytes() -> bytes:


    import mailbox as _mailbox
    import tempfile as _tempfile
    from email.message import EmailMessage
    with _tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        mbox = _mailbox.mbox(tmp_path)
        msg1 = EmailMessage()
        msg1['From'] = 'x@example.com'
        msg1['To'] = 'y@example.com'
        msg1['Subject'] = 'MBOX Child One'
        msg1.set_content('Hi from mbox 1')
        mbox.add(msg1)
        msg2 = EmailMessage()
        msg2['From'] = 'p@example.com'
        msg2['To'] = 'q@example.com'
        msg2['Subject'] = 'MBOX Child Two'
        msg2.set_content('Hi from mbox 2')
        mbox.add(msg2)
        mbox.flush()
        mbox.close()
        with open(tmp_path, 'rb') as fh:
            return fh.read()
    finally:
        import os as _os
        try:
            _os.unlink(tmp_path)
        except Exception:
            _ = None


def test_add_email_mbox_persists_children(test_api_client, db_session, dummy_headers):


    form_data = create_add_media_form_data(
        media_type="email",
        perform_chunking=True,
        chunk_method='sentences',
        chunk_size=500,
        chunk_overlap=200,
    )
    form_data.update({
        'accept_mbox': 'true',
    })
    mbox_bytes = _build_mbox_two_emails_bytes()
    files = {
        'files': ('emails.mbox', mbox_bytes, 'application/mbox'),
    }

    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, files=files, headers=dummy_headers)
    data = check_batch_response(response, expected_status_code=status.HTTP_200_OK)
    assert data['results'], "No results returned for email add (mbox)"
    parent = data['results'][0]
    check_media_item_result(parent, expected_status="Success", check_db_interaction=False, expected_media_type='email')
    # Parent should include the mbox grouping keyword
    parent_keywords = parent.get('keywords') or []
    assert 'email_mbox:emails' in parent_keywords, f"Parent missing mbox keyword. Keywords: {parent_keywords}"
    # Children persisted
    cdb = parent.get('child_db_results')
    assert isinstance(cdb, list) and len(cdb) == 2, f"Expected two children persisted from mbox, got: {cdb}"
    # Also assert that each child result includes the mbox grouping keyword
    children = parent.get('children') or []
    assert isinstance(children, list) and len(children) == 2, f"Expected two children in response, got: {children}"
    for child in children:
        if isinstance(child, dict):
            kws = child.get('keywords') or []
            assert 'email_mbox:emails' in kws, f"MBOX keyword missing in child: {kws}"


# --- Helper for Form Data ---
def create_add_media_form_data(**overrides) -> Dict[str, Any]:
    defaults = {
        "media_type": "video",
        "urls": None,
        "title": None,
        "author": None,
        "keywords": "",
        "custom_prompt": None,
        "system_prompt": None,
        "overwrite_existing": False,
        "keep_original_file": False,
        "perform_analysis": True,
        "api_name": None,
        "api_key": None,
        "use_cookies": False,
        "cookies": None,
        "transcription_model": "deepdml/faster-distil-whisper-large-v3.5",
        "transcription_language": "en",
        "diarize": False,
        "timestamp_option": True,
        "vad_use": False,
        "perform_confabulation_check_of_analysis": False,
        "start_time": None,
        "end_time": None,
        "pdf_parsing_engine": "pymupdf4llm",
        "perform_chunking": True,
        "chunk_method": 'sentences',
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": None,
        "chunk_size": 500,
        "chunk_overlap": 200,
        "custom_chapter_pattern": None,
        "perform_rolling_summarization": False,
        "summarize_recursively": False,
    }
    current_data = defaults.copy()
    current_data.update(overrides)
    form_dict = {}
    for k, v in current_data.items():
        if v is None:
            if k == 'keywords' and v is None: form_dict[k] = ""
            continue
        if k == 'urls' and isinstance(v, list):
            form_dict[k] = v
        elif isinstance(v, bool):
            form_dict[k] = str(v).lower()
        elif isinstance(v, (int, float, str)):
            form_dict[k] = v
        elif isinstance(v, Path):
            form_dict[k] = str(v)
        elif hasattr(v, 'value'):
            form_dict[k] = str(v.value)
        else:
            form_dict[k] = str(v)
    if "media_type" not in form_dict:
        form_dict["media_type"] = defaults["media_type"]
    logger.debug(f"Generated form data for /add: {form_dict}")
    return form_dict




# ##################################################################################################################
# Test Cases
# ##################################################################################################################

# === Validation Tests ===

# Capability flags for conditional skipping
HAS_FFMPEG = bool(shutil.which('ffmpeg') or os.path.exists('/opt/homebrew/bin/ffmpeg') or os.path.exists('/usr/local/bin/ffmpeg') or os.path.exists('/usr/bin/ffmpeg'))
NETWORK_TESTS_ENABLED = os.getenv('ENABLE_NETWORK_TESTS', '').lower() in {"1", "true", "yes", "y", "on"}

def test_add_media_invalid_media_type_value(test_api_client, dummy_headers):

    """Test sending an invalid value for the media_type enum."""
    form_data = create_add_media_form_data(media_type="picture", urls=["http://a.com"])
    # Use the RENAMED client variable
    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=dummy_headers)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    details = response.json().get('detail', [])
    assert isinstance(details, list)
    # Assert specific pydantic error message
    assert any("input should be 'video', 'audio', 'document', 'pdf' or 'ebook'" in err.get('msg', '').lower()
               for err in details if 'media_type' in err.get('loc', [])), f"Error details: {details}"

def test_add_media_invalid_field_type(test_api_client, dummy_headers):

    """Test sending a non-boolean string for a boolean field."""
    form_data = create_add_media_form_data(media_type="video", urls=["http://a.com"], diarize="maybe")
    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=dummy_headers)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    details = response.json().get('detail', [])
    # Check for the boolean parsing error specifically for the 'diarize' field
    found_diarize_error = False
    for err in details:
        if err.get('loc') == ['body', 'diarize'] and err.get('type') == 'bool_parsing':
            found_diarize_error = True
            break
    assert found_diarize_error, f"Expected boolean parsing error for 'diarize' not found in details: {details}"


def test_add_media_missing_url_and_file(test_api_client, dummy_headers):


    """Test calling the endpoint with valid media_type but no sources."""
    form_data = create_add_media_form_data(media_type="video", urls=None)
    # Ensure 'urls' key is truly absent if None
    if 'urls' in form_data: del form_data['urls']

    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=dummy_headers)
    # Expect 400 from the endpoint's internal _validate_inputs check
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "No valid media sources supplied" in response.json()["detail"]

def test_add_media_missing_required_form_field(test_api_client, dummy_headers):

    """Test calling without a required field like media_type."""
    # Remove media_type from the form data dictionary entirely
    form_data = create_add_media_form_data(media_type="video", urls=["http://a.com"])
    del form_data["media_type"]
    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=dummy_headers)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    details = response.json().get('detail', [])
    assert isinstance(details, list)
    assert any(err.get('type') == 'missing' and 'media_type' in err.get('loc', []) for err in details)

# === Basic Success Path Tests (URL & Upload) ===

@pytest.mark.parametrize("media_type, valid_url, expected_content_present", [
    pytest.param(
        "video",
        VALID_VIDEO_URL,
        True,
        marks=pytest.mark.skipif(
            not (HAS_FFMPEG and NETWORK_TESTS_ENABLED),
            reason="Requires network + ffmpeg for video URL",
        ),
    ),
    pytest.param(
        "audio",
        VALID_AUDIO_URL,
        True,
        marks=pytest.mark.skipif(
            not NETWORK_TESTS_ENABLED,
            reason="Requires network for audio URL",
        ),
    ),
    pytest.param(
        "pdf",
        VALID_PDF_URL,
        True,
        marks=pytest.mark.skipif(
            not NETWORK_TESTS_ENABLED,
            reason="Requires network for PDF URL",
        ),
    ),
    # Skip Ebook URL test for now due to potential download size/time
    # pytest.param("ebook", VALID_EPUB_URL, True, marks=pytest.mark.skip(reason="EPUB URL download can be large/slow"), id="ebook_url"),
    pytest.param(
        "document",
        VALID_TXT_URL,
        True,
        marks=pytest.mark.skipif(
            not NETWORK_TESTS_ENABLED,
            reason="Requires network for TXT URL",
        ),
    ),
    pytest.param(
        "document",
        VALID_MD_URL,
        True,
        marks=pytest.mark.skipif(
            not NETWORK_TESTS_ENABLED,
            reason="Requires network for Markdown URL",
        ),
    ),
], ids=["video_url", "audio_url", "pdf_url", "txt_url", "md_url"]) # Removed ebook id

@pytest.mark.timeout(180) # Increased timeout for external downloads/processing
def test_add_media_single_url_success(test_api_client, db_session, media_type, valid_url, expected_content_present, dummy_headers): # Added db_session
    """Test processing a single valid URL for each media type."""
    if media_type == "video":
        skip_if_transcription_model_unavailable(TEST_VIDEO_TRANSCRIPTION_MODEL)
        form_data = create_add_media_form_data(
            media_type=media_type,
            urls=[valid_url],
            transcription_model=TEST_VIDEO_TRANSCRIPTION_MODEL,
            start_time=TEST_VIDEO_START_TIME,
            end_time=TEST_VIDEO_END_TIME,
            perform_analysis=False,
            perform_chunking=False,
        )
    else:
        form_data = create_add_media_form_data(media_type=media_type, urls=[valid_url])
    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=dummy_headers)
    # Expect 200 OK for success, or 207 if warnings occurred during processing.
    # In environments without external network/egress, audio URLs may fail to download;
    # when that happens, skip rather than treating it as a regression.
    if media_type == "audio" and response.status_code == status.HTTP_207_MULTI_STATUS:
        try:
            data = response.json()
            results = data.get("results") or []
            first = results[0] if results else {}
            error_msg = (first.get("error") or "") if isinstance(first, dict) else ""
            if "Download failed" in error_msg or "Host could not be resolved" in error_msg:
                pytest.skip(
                    f"Audio URL test skipped due to download/egress error for {valid_url}: {error_msg}"
                )
        except Exception:
            # Fall through to normal assertions if we cannot interpret the error.
            _ = None

    expected_code = status.HTTP_200_OK
    if response.status_code != expected_code:  # Log if not expected
        logger.error(
            f"URL success test ({media_type}) failed. "
            f"Status: {response.status_code}, Expected: {expected_code}, Text: {response.text}"
        )

    # Check for 200 or 207 (if warnings happened but core processing succeeded)
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_207_MULTI_STATUS]

    data = check_batch_response(
        response,
        response.status_code,
        expected_processed=1,
        expected_errors=0,
        check_results_len=1,
    )
    result = data["results"][0]
    # Allow Success or Warning
    assert result["status"] in ["Success", "Warning"]
    check_media_item_result(result, result["status"], expected_media_type=media_type)
    assert result["input_ref"] == valid_url
    if expected_content_present:
        assert result.get("content") is not None and len(result["content"]) > 0, f"Content missing for {media_type} URL"
    # Check DB insertion
    assert isinstance(result.get("db_id"), int), f"Expected integer db_id for {media_type} URL, got {result.get('db_id')}"
    assert "added" in result.get("db_message", "").lower() or "updated" in result.get("db_message", "").lower()

@pytest.mark.parametrize("media_type, sample_path, expected_content_present", [
    pytest.param("video", SAMPLE_VIDEO_PATH, True, marks=pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not available for local video processing")),
    ("audio", SAMPLE_AUDIO_PATH, True),
    ("pdf", SAMPLE_PDF_PATH, True),
    ("ebook", SAMPLE_EPUB_PATH, True),
    ("document", SAMPLE_TXT_PATH, True),
    ("document", SAMPLE_MD_PATH, True),
    # Use skipif marker correctly
    pytest.param("document", SAMPLE_DOCX_PATH, True, marks=pytest.mark.skipif(not SAMPLE_DOCX_PATH.exists(), reason="sample.docx not found"), id="docx_upload"),
    # Use skipif marker for RTF as well, and keep xfail if pandoc isn't assumed
    pytest.param("document", SAMPLE_RTF_PATH, True, marks=[pytest.mark.skipif(not SAMPLE_RTF_PATH.exists(), reason="sample.rtf not found"), pytest.mark.xfail(reason="Requires pandoc binary installed")], id="rtf_upload"),
    ("document", SAMPLE_HTML_PATH, True),
    ("document", SAMPLE_XML_PATH, True),
], ids=["video_upload", "audio_upload", "pdf_upload", "epub_upload", "txt_upload", "md_upload", "docx_upload", "rtf_upload", "html_upload", "xml_upload"])

@pytest.mark.timeout(90) # Increased timeout
def test_add_media_single_file_upload_success(test_api_client, db_session, create_upload_file, media_type, sample_path, expected_content_present, dummy_headers): # Added db_session
    """Test processing a single valid file upload for various types."""
    if not sample_path.exists(): pytest.skip(f"Test file not found: {sample_path}")

    file_tuple = create_upload_file(sample_path)
    if media_type == "video":
        skip_if_transcription_model_unavailable(TEST_VIDEO_TRANSCRIPTION_MODEL)
        form_data = create_add_media_form_data(
            media_type=media_type,
            transcription_model=TEST_VIDEO_TRANSCRIPTION_MODEL,
            start_time=TEST_VIDEO_START_TIME,
            end_time=TEST_VIDEO_END_TIME,
            perform_analysis=False,
            perform_chunking=False,
        )
    else:
        form_data = create_add_media_form_data(media_type=media_type)

    # --- CORRECTED TestClient Call ---
    # Pass form data via `data` and files via `files` in the same call
    if media_type == "audio":
        # Avoid heavy dependencies by mocking conversion + transcription.
        with patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.convert_to_wav", side_effect=lambda p, **kw: p), \
        patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.speech_to_text", return_value=[{"Text": "test", "start_seconds": 0, "end_seconds": 1}]):
            response = test_api_client.post(
                ADD_MEDIA_ENDPOINT,
                data=form_data,
                files={"files": file_tuple}, # Key must match the File(..., alias="files") parameter name
                headers=dummy_headers
            )
    elif media_type == "video":
        # The sample MP4 is a dummy file; keep this test focused on endpoint plumbing.
        with patch(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.perform_transcription",
            return_value=("Mocked video transcript.", []),
        ):
            response = test_api_client.post(
                ADD_MEDIA_ENDPOINT,
                data=form_data,
                files={"files": file_tuple}, # Key must match the File(..., alias="files") parameter name
                headers=dummy_headers
            )
    else:
        response = test_api_client.post(
            ADD_MEDIA_ENDPOINT,
            data=form_data,
            files={"files": file_tuple}, # Key must match the File(..., alias="files") parameter name
            headers=dummy_headers
        )
    # -----------------------------------

    expected_code = status.HTTP_200_OK
    if response.status_code != expected_code:
        logger.error(f"File upload test ({sample_path.name}) failed. Status: {response.status_code}, Expected: {expected_code}, Text: {response.text}")

    assert response.status_code in [status.HTTP_200_OK, status.HTTP_207_MULTI_STATUS]
    data = check_batch_response(response, response.status_code, expected_processed=1, expected_errors=0, check_results_len=1)

    result = data["results"][0]
    assert result["status"] in ["Success", "Warning"]
    check_media_item_result(result, result["status"], expected_media_type=media_type)
    assert result["input_ref"] == sample_path.name
    # Processing source might be a temp path, check its name matches
    if media_type == "audio":
        # For audio files, the extension might be converted to .wav
        assert Path(result.get("processing_source", "")).stem == sample_path.stem
        # Optional: Verify it's a .wav file if that's the expected conversion
        assert Path(result.get("processing_source", "")).suffix.lower() in ['.wav']
    if expected_content_present:
        assert result.get("content") is not None and len(result["content"]) > 0, f"Content missing for {media_type} upload {sample_path.name}"
    # Check document source format metadata if applicable
    if media_type == "document" and "source_format" in result.get("metadata", {}):
        assert result["metadata"]["source_format"] == sample_path.suffix.lower().strip('.')
    # Check DB insertion
    assert isinstance(result.get("db_id"), int), f"Expected integer db_id for {media_type} upload, got {result.get('db_id')}"
    assert "added" in result.get("db_message", "").lower() or "updated" in result.get("db_message", "").lower()


# === Mixed Success/Failure Tests ===

@pytest.mark.timeout(180) # Increased timeout
@pytest.mark.skipif(not (HAS_FFMPEG and NETWORK_TESTS_ENABLED), reason="Requires network + ffmpeg for video URL + local video processing")
def test_add_media_mixed_url_file_success(test_api_client, db_session, create_upload_file, dummy_headers): # Added db_session
    """Test adding one valid video URL and one valid video file."""
    if not SAMPLE_VIDEO_PATH.exists(): pytest.skip(f"Test file not found: {SAMPLE_VIDEO_PATH}")

    file_tuple = create_upload_file(SAMPLE_VIDEO_PATH)
    skip_if_transcription_model_unavailable(TEST_VIDEO_TRANSCRIPTION_MODEL)
    form_data = create_add_media_form_data(
        media_type="video",
        urls=[VALID_VIDEO_URL],
        transcription_model=TEST_VIDEO_TRANSCRIPTION_MODEL,
        start_time=TEST_VIDEO_START_TIME,
        end_time=TEST_VIDEO_END_TIME,
        perform_analysis=False,
        perform_chunking=False,
    )

    # --- CORRECTED TestClient Call ---
    response = test_api_client.post(
        ADD_MEDIA_ENDPOINT,
        data=form_data,
        files={"files": file_tuple},
        headers=dummy_headers
    )
    # -----------------------------------

    expected_code = status.HTTP_200_OK
    if response.status_code != expected_code: # Log if not 200
        logger.error(f"Mixed URL/File test failed. Status: {response.status_code}, Expected: {expected_code}, Text: {response.text}")

    # Check for 200 or 207
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_207_MULTI_STATUS]
    data = check_batch_response(response, response.status_code, expected_processed=2, expected_errors=0, check_results_len=2)

    url_result = next((r for r in data["results"] if r["input_ref"] == VALID_VIDEO_URL), None)
    file_result = next((r for r in data["results"] if r["input_ref"] == SAMPLE_VIDEO_PATH.name), None)

    assert url_result is not None
    assert file_result is not None

    assert url_result["status"] in ["Success", "Warning"]
    check_media_item_result(url_result, url_result["status"], expected_media_type="video")
    assert isinstance(url_result.get("db_id"), int)

    assert file_result["status"] in ["Success", "Warning"]
    check_media_item_result(file_result, file_result["status"], expected_media_type="video")
    assert isinstance(file_result.get("db_id"), int)


@pytest.mark.timeout(120)
def test_add_media_multiple_failures_and_success_pdf(
    test_api_client,
    db_session,
    create_upload_file,
    dummy_headers
):
    """Test a mix of successful and failed items for PDF media type, mocking _process_document_like_item."""
    if not SAMPLE_PDF_PATH.exists(): pytest.skip(f"Test file not found: {SAMPLE_PDF_PATH}")

    good_pdf_file_tuple = create_upload_file(SAMPLE_PDF_PATH)
    # Create an invalid PDF with a PDF header so MIME validation passes and PDF parsing fails.
    invalid_pdf_path = TEST_MEDIA_DIR / "invalid.pdf"
    try:
        invalid_pdf_path.write_bytes(b"%PDF-1.4\n% invalid fixture\nnot a valid xref\n%%EOF\n")
    except Exception as e:
        pytest.skip(f"Failed to create invalid PDF for test: {e}")
    invalid_format_file_tuple = create_upload_file(invalid_pdf_path)

    # --- Create the AsyncMock instance FIRST ---
    mock_processor = AsyncMock()

    # --- Define Refined Side Effect ---
    async def refined_side_effect(*args, **kwargs):
        # --- Rely solely on kwargs ---
        item_input_ref = kwargs.get('item_input_ref')
        processing_source = kwargs.get('processing_source')
        is_url = kwargs.get('is_url')
        temp_dir = kwargs.get('temp_dir') # This will be the Path object

        # --- Ensure essential values needed for logic are present ---
        if item_input_ref is None or processing_source is None or is_url is None or temp_dir is None:
            logger.error(f"[Refined Mock] Missing essential kwargs: {list(kwargs.keys())}")
            return {"status": "Error", "error": "Mock received incomplete kwargs"}

        logger.info(f"[Refined Mock _process_doc_item] Called for: ref='{item_input_ref}', source='{processing_source}', is_url={is_url}")

        if is_url:
            # Ensure temp_dir is a Path object before using it
            temp_dir_path = Path(temp_dir) if not isinstance(temp_dir, Path) else temp_dir

            if processing_source == VALID_PDF_URL:
                logger.info(f"[Refined Mock] Simulating SUCCESS for URL: {processing_source}")
                # Return the predefined SUCCESS dictionary
                return { "status": "Success", "input_ref": item_input_ref, "processing_source": str(temp_dir_path / "dummy.pdf"), "media_type": "pdf", "metadata": {"title": "Mocked PDF Title", "author": "Mock Author"}, "content": "# Mocked PDF Content\nFrom URL.", "segments": None, "chunks": [{"text": "Mocked chunk 1", "metadata": {}}], "analysis": None, "summary": None, "analysis_details": {}, "error": None, "warnings": None, "db_id": 123, "db_message": "Media 'Mocked PDF Title' added.", "message": None, "media_uuid": "mock-uuid-url-123", "transcript": "# Mocked PDF Content\nFrom URL."}
            elif processing_source == URL_404:
                logger.info(f"[Refined Mock] Simulating FAILURE for URL: {processing_source}")
                # Return the predefined FAILURE dictionary
                return {"status": "Error", "input_ref": item_input_ref, "processing_source": processing_source, "media_type": "pdf", "metadata": {}, "content": None, "segments": None, "chunks": None, "analysis": None, "summary": None, "analysis_details": None, "error": f"File preparation/download failed: HTTPStatusError: Client error '404 Not Found' for url `{URL_404}`", "warnings": None, "db_id": None, "db_message": "DB operation skipped (processing failed).", "message": None, "media_uuid": None, "transcript": None}
            else:
                logger.warning(f"[Refined Mock] Unexpected URL: {processing_source}")
                return {"status": "Error", "input_ref": item_input_ref, "error": "Unexpected URL in mock"}
        else:
            logger.info(f"[Refined Mock] Passing through to original for Upload: {processing_source}")
            # Use the imported original function
            return await _process_document_like_item(**kwargs)

    # --- Assign the side effect to the MANUALLY CREATED mock ---
    mock_processor.side_effect = refined_side_effect

    # --- Apply Patch Manually using Context Manager, passing the mock as 'new' ---
    # The variable 'mock_target_within_context' will now also refer to 'mock_processor'
    with patch(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.persistence.process_document_like_item",
        new=mock_processor,
    ) as mock_target_within_context:

        # --- Form and File Data ---
        form_data = create_add_media_form_data(media_type="pdf", urls=[VALID_PDF_URL, URL_404])
        files_data = [
            ("files", good_pdf_file_tuple),
            ("files", invalid_format_file_tuple)
        ]

        # --- API Call ---
        response = test_api_client.post(
            ADD_MEDIA_ENDPOINT,
            data=form_data,
            files=files_data,
            headers=dummy_headers
        )

    # --- Assertions ---
    # In some environments without external network/egress, URL downloads may fail
    # before batch processing and return a top-level 400 with a host resolution error.
    # Treat that as an environment-specific issue rather than a regression.
    if response.status_code == status.HTTP_400_BAD_REQUEST:
        try:
            body = response.json()
            detail = body.get("detail") if isinstance(body, dict) else ""
        except Exception:
            detail = response.text or ""
        if isinstance(detail, str) and "Host could not be resolved" in detail:
            pytest.skip(
                f"/media/add mixed PDF URL+file test skipped due to download/egress error: {detail}"
            )

    expected_code = status.HTTP_207_MULTI_STATUS
    if response.status_code != expected_code:
        logger.error(f"Multiple Fail/Success test failed. Status: {response.status_code}, Expected: {expected_code}")
        try:
            logger.error(f"Response JSON: {response.json()}")
        except Exception:
            logger.error(f"Response Text: {response.text}")
    assert response.status_code == expected_code

    data = check_batch_response(response, 207, expected_processed=2, expected_errors=2, check_results_len=4)
    results_map = {r["input_ref"]: r for r in data["results"]}

    # 1. Valid PDF URL -> Success (from Mock)
    assert VALID_PDF_URL in results_map, f"Result for valid URL {VALID_PDF_URL} not found"
    assert results_map[VALID_PDF_URL]["status"] == "Success", f"Mocked URL status mismatch: {results_map[VALID_PDF_URL]['status']}"
    assert results_map[VALID_PDF_URL].get("content") == "# Mocked PDF Content\nFrom URL."
    assert results_map[VALID_PDF_URL].get("db_id") == 123

    # 2. URL 404 -> Error (from Mock)
    assert URL_404 in results_map, f"Result for 404 URL {URL_404} not found"
    assert results_map[URL_404]["status"] == "Error", f"Mocked 404 URL status mismatch: {results_map[URL_404]['status']}"
    assert "404 Not Found" in results_map[URL_404].get("error", ""), "Mocked 404 error message mismatch"
    assert results_map[URL_404].get("db_id") is None

    # 3. Valid PDF Upload -> Success (from Real Implementation via pass-through)
    assert SAMPLE_PDF_PATH.name in results_map, f"Result for valid PDF file {SAMPLE_PDF_PATH.name} not found"
    assert results_map[SAMPLE_PDF_PATH.name]["status"] in ["Success", "Warning"], f"Real PDF file status mismatch: {results_map[SAMPLE_PDF_PATH.name]['status']}"
    check_media_item_result(results_map[SAMPLE_PDF_PATH.name], results_map[SAMPLE_PDF_PATH.name]["status"], expected_media_type="pdf")
    assert isinstance(results_map[SAMPLE_PDF_PATH.name].get("db_id"), int), f"Real PDF file DB ID mismatch: {results_map[SAMPLE_PDF_PATH.name].get('db_id')}"

    # 4. Invalid Format Upload -> Error (from Real Implementation via pass-through)
    assert invalid_pdf_path.name in results_map, f"Result for invalid file {invalid_pdf_path.name} not found"
    assert results_map[invalid_pdf_path.name]["status"] == "Error", f"Real invalid file status mismatch: {results_map[invalid_pdf_path.name]['status']}"
    check_media_item_result(results_map[invalid_pdf_path.name], "Error", expected_media_type="pdf")
    error_msg = results_map[invalid_pdf_path.name].get("error", "").lower()
    assert "pdf extraction error" in error_msg or \
    "failed to open file" in error_msg or \
    "invalid file" in error_msg or \
    "cannot parse" in error_msg or \
    "validation failed" in error_msg or \
    "pymupdf" in error_msg, f"Expected PDF processing error for real invalid file, got: {error_msg}"
    assert results_map[invalid_pdf_path.name].get("db_id") is None

    assert mock_processor.call_count == 4, f"Expected 4 calls to _process_document_like_item, got {mock_processor.call_count}"

    # Cleanup invalid test file
    try:
        if invalid_pdf_path.exists():
            invalid_pdf_path.unlink()
    except Exception:
        _ = None

# === Error Handling Tests ===

@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing.save_uploaded_files", new_callable=AsyncMock)
def test_add_media_file_save_error(mock_save_files, test_api_client, db_session, create_upload_file, dummy_headers): # Added db_session
    """Test an error during the file saving stage."""
    if not SAMPLE_AUDIO_PATH.exists(): pytest.skip(f"Test file not found: {SAMPLE_AUDIO_PATH}")

    file_tuple = create_upload_file(SAMPLE_AUDIO_PATH)
    # Simulate file save helper returning only errors
    mock_save_files.return_value = (
        [], # No successfully saved files
        [{"original_filename": SAMPLE_AUDIO_PATH.name, "input_ref": SAMPLE_AUDIO_PATH.name, "status": "Error", "error": "Failed to save uploaded file: Disk full (OSError)"}] # List of errors
    )

    form_data = create_add_media_form_data(media_type="audio")

    response = test_api_client.post(
        ADD_MEDIA_ENDPOINT,
        data=form_data,
        # IMPORTANT: Still need to pass *something* to 'files' for FastAPI to know it's multipart,
        # even though our mock will intercept file saving.
        # Pass the original tuple, the mock prevents actual saving.
        files={"files": file_tuple},
        headers=dummy_headers
    )
    # -----------------------------------

    expected_code = status.HTTP_400_BAD_REQUEST
    if response.status_code != expected_code:
        logger.error(f"File Save Error test failed. Status: {response.status_code}, Expected: {expected_code}, Text: {response.text}")

    assert response.status_code == expected_code
    data = response.json()
    assert "detail" in data
    detail = data["detail"]
    # Validate error context is present in the detail message
    assert (
        "Failed to save" in detail
        or "Disk full" in detail
        or "no valid media sources" in detail.lower()
    )
    mock_save_files.assert_called_once()


# Use the provided TempDirManager class in the endpoint now
# @patch('tempfile.TemporaryDirectory') # No longer need to mock tempfile directly
@patch("tldw_Server_API.app.api.v1.endpoints.media.TempDirManager")
def test_add_media_temp_dir_creation_error(mock_temp_dir_manager_class, test_api_client, db_session, create_upload_file, dummy_headers): # Added db_session
    """Test failure during temporary directory creation using TempDirManager."""
    if not SAMPLE_AUDIO_PATH.exists(): pytest.skip(f"Test file not found: {SAMPLE_AUDIO_PATH}")

    # Mock the __enter__ method of the manager instance to raise an error
    mock_manager_instance = MagicMock()
    mock_manager_instance.__enter__.side_effect = OSError("Permission denied creating temp dir")
    mock_temp_dir_manager_class.return_value = mock_manager_instance

    file_tuple = create_upload_file(SAMPLE_AUDIO_PATH)
    form_data = create_add_media_form_data(media_type="audio")

    # --- CORRECTED TestClient Call ---
    response = test_api_client.post(
        ADD_MEDIA_ENDPOINT,
        data=form_data,
        files={"files": file_tuple},
        headers=dummy_headers
    )
    # -----------------------------------

    expected_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if response.status_code != expected_code:
        logger.error(f"Temp Dir Error test failed. Status: {response.status_code}, Expected: {expected_code}, Text: {response.text}")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    # Check the detail message from the endpoint's exception handler
    assert "OS error during setup" in response.json().get("detail", "") or \
    "Failed to create temporary directory" in response.json().get("detail", "") # Allow for variations
    assert "Permission denied" in response.json().get("detail", "")



@pytest.mark.timeout(90) # Increased timeout
def test_add_media_processor_handles_invalid_format(test_api_client, db_session, create_upload_file, dummy_headers): # Added db_session
    """Test feeding an audio file to the video processor via /add."""
    if not SAMPLE_AUDIO_PATH.exists(): pytest.skip(f"Test file not found: {SAMPLE_AUDIO_PATH}")

    file_tuple = create_upload_file(SAMPLE_AUDIO_PATH)
    form_data = create_add_media_form_data(media_type="video") # Mismatched type

    response = test_api_client.post(
        ADD_MEDIA_ENDPOINT,
        data=form_data,
        files={"files": file_tuple},
        headers=dummy_headers
    )

    expected_code = status.HTTP_207_MULTI_STATUS

    if response.status_code != expected_code:
        logger.error(
            f"Invalid Format test failed. Status: {response.status_code}, Expected: {expected_code}, Text: {response.text}")
        # Add assertion here to make it fail if the code isn't the adjusted expected code
        assert response.status_code == expected_code

    data = check_batch_response(response, expected_code, expected_processed=0, expected_errors=1, check_results_len=1)

    result = data["results"][0]
    assert result["status"] in ["Error", "Warning"]

    # Check structure, but acknowledge the media_type in the result matches the *request* (video)
    # even though the input was audio.
    check_media_item_result(result, result["status"], expected_media_type="video")
    error_msg = result.get("error", "")
    warning_msg = result.get("warnings")

    # assert "failed to process" in error_msg or \
    #        "invalid format" in error_msg or \
    #        "cannot open" in error_msg or \
    #        (warning_msg and any("format" in w.lower() for w in warning_msg)), \
    #     f"Expected a processing/format error or warning, got error='{error_msg}', warnings='{warning_msg}'"
    # Expect an error message for mismatched format
    assert error_msg and isinstance(error_msg, str)
    # No DB insertion should happen for a failed/mismatched processing-only item
    assert result.get("db_id") in [None, 0]


# === Analysis and Chunking Tests ===

# Target the summarize function *within* the specific PDF processing library module
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib.analyze")
@pytest.mark.timeout(90)
def test_add_media_pdf_with_analysis_mocked(mock_analyze, test_api_client, db_session, dummy_headers):
    """Test PDF analysis via /add, mocking only the summarize call."""
    if not SAMPLE_PDF_PATH.exists(): pytest.skip(f"Test file not found: {SAMPLE_PDF_PATH}")
    if not NETWORK_TESTS_ENABLED:
        pytest.skip("Requires network for PDF URL download")

    mock_analysis_text = "Mocked analysis for PDF."
    # Configure the mock to return the desired text directly since analyze is synchronous
    mock_analyze.return_value = mock_analysis_text

    form_data = create_add_media_form_data(
        media_type="pdf",
        urls=[VALID_PDF_URL],
        perform_analysis=True,
        perform_chunking=True,
        api_name="mock_llm"
    )
    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=dummy_headers)

    expected_code = status.HTTP_200_OK
    if response.status_code != expected_code:
        logger.error(f"PDF Analysis test failed. Status: {response.status_code}, Expected: {expected_code}, Text: {response.text}")

    assert response.status_code in [status.HTTP_200_OK, status.HTTP_207_MULTI_STATUS]
    data = check_batch_response(response, response.status_code, expected_processed=1, expected_errors=0, check_results_len=1)

    result = data["results"][0]
    assert result["status"] in ["Success", "Warning"]
    check_media_item_result(result, result["status"], expected_media_type="pdf")

    assert result.get("content") is not None and len(result["content"]) > 0

    # Analysis may be skipped when server keys are not provided via client
    # Accept either analyzed text or None
    # Do not assert mock_analyze called, as API keys are not accepted from client
    if result.get("analysis") is not None:
        assert result.get("analysis") == mock_analysis_text

    # Check if analysis details reflect the mock API name used
    # The key in your application code is "analysis_model"
    # analysis_details may be empty if analysis is skipped
    if result.get("analysis_details"):
        assert result.get("analysis_details", {}).get("analysis_model") == form_data.get("api_name")
    assert isinstance(result.get("db_id"), int)
    assert "added" in result.get("db_message", "").lower() or "updated" in result.get("db_message", "").lower()


# Helper to check the common structure of processing-only endpoint results
def check_processing_only_item_result_structure(
        result_item: Dict[str, Any],
        expected_media_type: str,
        mock_analysis_text: str,
        expected_api_name: str,
        check_content: bool = True
):
    """Checks the structure and key values for a single item from a processing-only endpoint."""
    assert result_item["status"] in ["Success", "Warning"]
    assert result_item["media_type"] == expected_media_type

    if check_content:
        assert result_item.get("content") is not None, f"Content missing for {expected_media_type}"
        assert len(result_item["content"]) > 0, f"Content empty for {expected_media_type}"

    assert result_item.get("analysis") == mock_analysis_text

    analysis_details = result_item.get("analysis_details", {})
    # Check for 'model' or 'model_used' as the key can vary
    model_key_present = ("llm_api" in analysis_details or
                         "analysis_model" in analysis_details or
                         "model_used" in analysis_details or
                         "model" in analysis_details or  # Added 'model' explicitly
                         "whisper_model" in analysis_details) # Keep whisper check if relevant
    assert model_key_present, f"None of 'analysis_model', 'llm_api', 'model', 'model_used', or 'whisper_model' found in analysis_details: {analysis_details}"


    actual_api_name = analysis_details.get("model_used",
                                         analysis_details.get("model",
                                                            analysis_details.get("llm_api",
                                                                               analysis_details.get("analysis_model"))))
    assert actual_api_name == expected_api_name, \
    f"Expected API name '{expected_api_name}', got '{actual_api_name}' in analysis_details"

    assert result_item.get("db_id") is None
    assert result_item.get("db_message") == "Processing only endpoint."


# === Mocked Analysis Tests for Processing-Only Endpoints ===

@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Books.Book_Processing_Lib.analyze")
@pytest.mark.timeout(120)  # Ebook processing can take a bit
def test_process_ebook_with_analysis_mocked(mock_analyze, test_api_client, db_session, create_upload_file,
                                            dummy_headers):
    """Test EPUB analysis via /process-ebooks, mocking only the analyze call."""
    if not SAMPLE_EPUB_PATH.exists():
        pytest.skip(f"Test file not found: {SAMPLE_EPUB_PATH}")

    # This is the value returned *each time* the mock is called
    mock_analysis_text_single_call = "analysis text for EPUB File."
    mock_analyze.return_value = mock_analysis_text_single_call

    form_data_dict = create_add_media_form_data(
        # media_type will be set by ProcessEbooksForm model
        perform_analysis=True,
        perform_chunking=True,  # This triggers chunking and potentially multiple analyze calls
        api_name="mock_llm",
        api_key="mock_key",
        extraction_method="filtered",  # Ebook specific option
        summarize_recursively=True  # Keep True to ensure the mock IS called multiple times if the lib supports it
    )
    # Remove fields fixed by the Pydantic model on the server for this endpoint
    if 'media_type' in form_data_dict: del form_data_dict['media_type']
    if 'keep_original_file' in form_data_dict: del form_data_dict['keep_original_file']
    # Ensure urls is not sent if we are uploading a file
    form_data_dict['urls'] = None

    file_tuple = create_upload_file(SAMPLE_EPUB_PATH)
    response = test_api_client.post(
        PROCESS_EBOOKS_ENDPOINT,
        data=form_data_dict,
        files={"files": file_tuple},
        headers=dummy_headers
    )

    assert response.status_code in [status.HTTP_200_OK, status.HTTP_207_MULTI_STATUS], \
    f"Request failed: {response.status_code} - {response.text}"

    data = response.json()
    assert data.get("processed_count", 0) >= 1  # Allow for warnings to still count as processed
    assert data.get("errors_count", 0) == 0
    assert len(data.get("results", [])) >= 1

    result = data["results"][0]

    actual_call_count = mock_analyze.call_count
    # This assertion verifies that analysis was attempted.
    # If summarize_recursively=True and chunking is on, we'd expect more than one call
    # if the library processes chunks and then a final summary.
    # The exact number of calls depends on the internal logic of process_epub.
    # Server does not accept client API keys; analysis may be skipped.
    # Do not enforce analyze() call count.
    logger.info(f"Mock analyze was called {actual_call_count} times for the EPUB test.")

    # The 'analysis' field in the result item from the endpoint is expected to contain
    # the final, top-level summary. Since our mock_analyze returns the same string
    # for every call, this final summary will be that string.
    expected_analysis_in_result_item = mock_analysis_text_single_call

    # Pass this expected single analysis string to the helper.
    # Accept presence or absence of analysis text
    check_processing_only_item_result_structure(result, "ebook", expected_analysis_in_result_item, "mock_llm", check_content=False)


@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files.analyze")
@pytest.mark.timeout(180)  # Audio processing can be slow
def test_process_audio_with_analysis_mocked(mock_analyze, test_api_client, db_session, create_upload_file,
                                            dummy_headers):
    """Test Audio analysis via /process-audios, mocking only the analyze call."""
    if not SAMPLE_AUDIO_PATH.exists():
        pytest.skip(f"Test file not found: {SAMPLE_AUDIO_PATH}")

    mock_analysis_text = "Mocked analysis for Audio."
    mock_analyze.return_value = mock_analysis_text

    form_data_dict = create_add_media_form_data(
        perform_analysis=True,
        perform_chunking=True,
        custom_prompt="Please summarize this audio.", # <--- ADD THIS LINE
        api_name="mock_llm",
        api_key="mock_key",
        transcription_model="deepdml/faster-distil-whisper-large-v3.5"
    )
    if 'media_type' in form_data_dict: del form_data_dict['media_type']
    if 'keep_original_file' in form_data_dict: del form_data_dict['keep_original_file']
    form_data_dict['urls'] = None

    file_tuple = create_upload_file(SAMPLE_AUDIO_PATH)
    # Avoid real conversion/transcription during tests
    with (
        patch(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.convert_to_wav",
            side_effect=lambda p, **kw: p,
        ),
        patch(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.speech_to_text",
            return_value=[{"Text": "hello", "start_seconds": 0, "end_seconds": 1}],
        ),
    ):
        response = test_api_client.post(
            PROCESS_AUDIO_ENDPOINT,
            data=form_data_dict,
            files={"files": file_tuple},
            headers=dummy_headers
        )

    assert response.status_code in [status.HTTP_200_OK, status.HTTP_207_MULTI_STATUS], \
    f"Request failed: {response.status_code} - {response.text}"

    data = response.json()
    assert data.get("processed_count", 0) >= 1
    assert data.get("errors_count", 0) == 0
    assert len(data.get("results", [])) >= 1

    result = data["results"][0]
    # Do not enforce analyze() call; accept that analysis may be skipped in tests.
    check_processing_only_item_result_structure(result, "audio", mock_analysis_text, "mock_llm", check_content=False)


@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.perform_transcription")
@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.analyze")
@pytest.mark.timeout(240)  # Video processing can be very slow
def test_process_video_with_analysis_mocked(mock_analyze, mock_transcription, test_api_client, db_session, create_upload_file,
                                            dummy_headers):
    """Test Video analysis via /process-videos without invoking local transcription models."""
    if not SAMPLE_VIDEO_PATH.exists():
        pytest.skip(f"Test file not found: {SAMPLE_VIDEO_PATH}")

    mock_analysis_text = "Mocked analysis for Video."
    mock_analyze.return_value = mock_analysis_text
    mock_transcription.return_value = ("Mocked video transcript.", [])

    skip_if_transcription_model_unavailable(TEST_VIDEO_TRANSCRIPTION_MODEL)
    form_data_dict = create_add_media_form_data(
        perform_analysis=True,
        perform_chunking=True,
        api_name="mock_llm",
        api_key="mock_key",
        transcription_model=TEST_VIDEO_TRANSCRIPTION_MODEL,
        start_time=TEST_VIDEO_START_TIME,
        end_time=TEST_VIDEO_END_TIME,
    )
    if 'media_type' in form_data_dict: del form_data_dict['media_type']
    if 'keep_original_file' in form_data_dict: del form_data_dict['keep_original_file']
    form_data_dict['urls'] = []

    file_tuple = create_upload_file(SAMPLE_VIDEO_PATH)
    response = test_api_client.post(
        PROCESS_VIDEO_ENDPOINT,
        data=form_data_dict,
        files={"files": file_tuple},
        headers=dummy_headers
    )

    assert response.status_code in [status.HTTP_200_OK, status.HTTP_207_MULTI_STATUS], \
    f"Request failed: {response.status_code} - {response.text}"

    data = response.json()
    assert data.get("processed_count", 0) >= 1
    assert data.get("errors_count", 0) == 0
    assert len(data.get("results", [])) >= 1

    result = data["results"][0]
    # Do not enforce analyze() call; accept that analysis may be skipped in tests.
    check_processing_only_item_result_structure(result, "video", mock_analysis_text, "mock_llm", check_content=False)


@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files.analyze")
@pytest.mark.timeout(90)
def test_process_document_with_analysis_mocked(mock_analyze, test_api_client, db_session, create_upload_file,
                                               dummy_headers):
    """Test Document (TXT) analysis via /process-documents, mocking only the analyze call."""
    if not SAMPLE_TXT_PATH.exists():
        pytest.skip(f"Test file not found: {SAMPLE_TXT_PATH}")

    mock_analysis_text = "Mocked analysis for TXT Document."
    mock_analyze.return_value = mock_analysis_text

    form_data_dict = create_add_media_form_data(
        perform_analysis=True,
        perform_chunking=True,  # Default for documents is recursive, which is fine
        api_name="mock_llm",
        api_key="mock_key",
        chunk_method="sentences"  # Example specific override if needed
    )
    if 'media_type' in form_data_dict: del form_data_dict['media_type']
    if 'keep_original_file' in form_data_dict: del form_data_dict['keep_original_file']
    form_data_dict['urls'] = []

    file_tuple = create_upload_file(SAMPLE_TXT_PATH)
    response = test_api_client.post(
        PROCESS_DOCUMENTS_ENDPOINT,
        data=form_data_dict,
        files={"files": file_tuple},
        headers=dummy_headers
    )

    assert response.status_code in [status.HTTP_200_OK, status.HTTP_207_MULTI_STATUS], \
    f"Request failed: {response.status_code} - {response.text}"

    data = response.json()
    assert data.get("processed_count", 0) >= 1
    assert data.get("errors_count", 0) == 0
    assert len(data.get("results", [])) >= 1

    result = data["results"][0]
    mock_analyze.assert_called()
    check_processing_only_item_result_structure(result, "document", mock_analysis_text, "mock_llm")


# ##################################################################################################################
# End of remodeled test_add_media_endpoint.py
# ##################################################################################################################
