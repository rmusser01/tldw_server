# test_media_processing.py
# Description: This file contains the test cases for the media processing endpoints
# (endpoints that DO NOT persist to DB) of the tldw application.
#
# Imports
import hashlib
import os
import sqlite3
import sys
import uuid
from pathlib import Path
import time
import json # Added for potential debugging
from typing import Dict
from unittest.mock import patch, AsyncMock, MagicMock # Added AsyncMock, MagicMock

# 3rd-party Libraries
import pytest
from fastapi import status # Added
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.core.DB_Management.DB_Manager import get_all_document_versions
from tldw_Server_API.app.core.Utils.Utils import logging
from tldw_Server_API.tests.MediaDB2.test_sqlite_db import get_entity_version, get_latest_log
from tldw_Server_API.tests.Media_Ingestion_Modification.test_add_media_endpoint import override_get_request_user
# Local Imports
# --- Test Utilities ---
from tldw_Server_API.tests.test_utils import temp_db

# --- App and Dependencies for Overriding ---
try:
    # Import app instance and specific dependencies to override
    from tldw_Server_API.app.main import app as fastapi_app_instance, app
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User, _single_user_instance
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_db_for_user
    from tldw_Server_API.app.core.config import settings
    from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import Database, get_document_version, \
    search_media_db  # If type hints needed
except ImportError as e:
    raise ImportError(f"Could not locate the FastAPI app instance or dependencies: {e}")


######################################################################################################################
# Constants
# Assume single-user mode for tests, get the key from settings
try:
    TEST_API_KEY = settings.get("SINGLE_USER_API_KEY", "default_test_key_if_not_set")
    if not settings.get("SINGLE_USER_MODE", False):
        logger.warning("SINGLE_USER_MODE is False in settings, X-API-KEY auth might not work as expected.")
except Exception as e:
    logger.error(f"Could not load settings to get API key: {e}. Using default.")
    TEST_API_KEY = "default_test_key_if_not_set"

# --- Constants for Test Files and URLs ---
TEST_MEDIA_DIR = Path(__file__).parent / "test_media"
TEST_MEDIA_DIR.mkdir(exist_ok=True) # Ensure directory exists
SAMPLE_VIDEO_PATH = TEST_MEDIA_DIR / "sample.mp4"
SAMPLE_AUDIO_PATH = TEST_MEDIA_DIR / "sample.mp3"
SAMPLE_PDF_PATH = TEST_MEDIA_DIR / "sample.pdf"
SAMPLE_EPUB_PATH = TEST_MEDIA_DIR / "sample.epub"
SAMPLE_TXT_PATH = TEST_MEDIA_DIR / "sample.txt"
SAMPLE_MD_PATH = TEST_MEDIA_DIR / "sample.md"
SAMPLE_DOCX_PATH = TEST_MEDIA_DIR / "sample.docx"
SAMPLE_RTF_PATH = TEST_MEDIA_DIR / "sample.rtf"
SAMPLE_HTML_PATH = TEST_MEDIA_DIR / "sample.html"
SAMPLE_XML_PATH = TEST_MEDIA_DIR / "sample.xml"

# Create basic text files if they don't exist
if not SAMPLE_TXT_PATH.exists(): SAMPLE_TXT_PATH.write_text("Sample TXT for processing.", encoding='utf-8')
if not SAMPLE_MD_PATH.exists(): SAMPLE_MD_PATH.write_text("# Sample MD\nFor processing.", encoding='utf-8')
if not SAMPLE_HTML_PATH.exists(): SAMPLE_HTML_PATH.write_text("<html><body>Sample HTML for processing.</body></html>", encoding='utf-8')
if not SAMPLE_XML_PATH.exists(): SAMPLE_XML_PATH.write_text("<root><item>Sample XML for processing</item></root>", encoding='utf-8')

# Test URLs
VALID_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VALID_AUDIO_URL = "https://cdn.pixabay.com/download/audio/2023/12/02/audio_2f291f569a.mp3?filename=about-anger-179423.mp3"
VALID_PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
VALID_EPUB_URL = "https://filesamples.com/samples/ebook/epub/Alices%20Adventures%20in%20Wonderland.epub"
VALID_TXT_URL = "https://raw.githubusercontent.com/rmusser01/tldw/main/LICENSE.txt"
VALID_MD_URL = "https://raw.githubusercontent.com/rmusser01/tldw/main/README.md"
VALID_HTML_URL = "https://example.com/" # Use example.com for basic HTML
INVALID_URL = "http://this.url.definitely.does.not.exist.invalid/resource.mp4"
URL_404 = "https://httpbin.org/status/404"


# --- Fixtures ---


# --- Database Fixture ---
@pytest.fixture(scope="module")
def client_module_processing(db_instance_session): # Assuming similar DB fixture pattern
    """ TestClient for media processing tests with overrides """
    def override_get_db_for_user_processing():
        yield db_instance_session

    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_request_user] = override_get_request_user
    app.dependency_overrides[get_db_for_user] = override_get_db_for_user_processing
    logger.info("Applied dependency overrides for get_request_user and get_db_for_user (processing)")

    with TestClient(fastapi_app_instance) as c:
        yield c

    app.dependency_overrides = original_overrides
    logger.info("Restored original dependency overrides (processing)")

@pytest.fixture(scope="session")
def db_instance_session_proc():
    """Session-scoped temporary database for processing tests."""
    db = None
    try:
        with temp_db() as db:
            yield db
    finally:
        if db and hasattr(db, 'close_all_connections'):
            print(f"--- Closing ALL session DB connections for test_media_processing: {db.db_path_str} ---")
            db.close_all_connections()

@pytest.fixture(scope="function")
def db_session_proc(db_instance_session_proc):
     """Function-scoped access to the session DB. No cleanup needed for processing tests."""
     yield db_instance_session_proc
     # No DB modifications expected in these endpoints, so cleanup might be omitted
     # logger.debug("DB session yield finished for processing test.")


@pytest.fixture
def dummy_headers():
    """Provides headers required by endpoint signature, even if logic is mocked."""
    # The actual value doesn't matter because get_request_user is mocked
    return {"token": "dummy_test_token_for_header"}

# --- Authentication Override Fixture ---
@pytest.fixture(scope="module")
def override_auth_proc():
    """Overrides the main get_request_user dependency for the processing tests."""
    async def _override_get_request_user_proc_test():
        logger.debug("--- AUTH OVERRIDE (Processing): Returning single_user_instance ---")
        # Ensure the instance uses the correct ID from settings
        _single_user_instance.id = settings.get("SINGLE_USER_FIXED_ID", 1)
        return _single_user_instance
    yield _override_get_request_user_proc_test

# --- DB Override Fixture Function ---
def override_get_db_for_user_proc(db_session):
    """Dependency override factory for processing tests."""
    def _override():
        # logger.debug(f"--- DB OVERRIDE (Processing): Providing DB session: {db_session.db_path_str} ---")
        yield db_session
    return _override

# --- Combined Client Fixture for Processing Tests ---
@pytest.fixture(scope="module")
def client(db_instance_session_proc, override_auth_proc):
    """Provides a TestClient instance for the processing tests with overrides."""
    # Ensure test media files exist (run once per module)
    required_files = [
        SAMPLE_VIDEO_PATH, SAMPLE_AUDIO_PATH, SAMPLE_PDF_PATH, SAMPLE_EPUB_PATH,
        SAMPLE_TXT_PATH, SAMPLE_MD_PATH, SAMPLE_HTML_PATH, SAMPLE_XML_PATH
    ]
    # Skip module if essential files are missing
    for f_path in required_files:
        if not f_path.exists():
            pytest.skip(f"Essential test file missing, skipping module: {f_path}")
    # Optional files checked within tests (DOCX, RTF)

    # Apply DB and Auth overrides specific to this module
    app.dependency_overrides[get_db_for_user] = override_get_db_for_user_proc(db_instance_session_proc)
    app.dependency_overrides[get_request_user] = override_auth_proc
    logger.info("--- TestClient (Processing) created with DB and Auth overrides ---")

    with TestClient(fastapi_app_instance) as c:
        yield c

    # Cleanup overrides after all tests in the module run
    app.dependency_overrides.clear()
    logger.info("--- TestClient (Processing) DB and Auth overrides cleared ---")


# --- Define the factory directly in the test file for isolation ---
@pytest.fixture(scope="function")  # Function scope for the factory itself, so each test gets a fresh one if needed
def memory_db_factory_local():  # Using a different name to avoid conflicts if one exists elsewhere
    """
    Factory fixture that provides a function to create
    new, isolated in-memory Database instances for TestDocumentVersioningV2.
    """
    created_db_instances = []

    def _create_db(client_id_prefix: str = "mem_db_client_v2"):
        generated_uuid_str = str(uuid.uuid4())  # Direct UUID generation
        unique_client_id = f"{client_id_prefix}_{generated_uuid_str}"

        db_instance = Database(db_path=":memory:", client_id=unique_client_id)

        try:
            with db_instance.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT 1 FROM media_fts LIMIT 1")
                except sqlite3.OperationalError as e:
                    if "no such table" in str(e).lower():
                        logging.warning(
                            f"FTS tables not found for {db_instance.db_path_str} (client: {unique_client_id}), attempting to create.")
                        if hasattr(Database, '_FTS_TABLES_SQL') and Database._FTS_TABLES_SQL:
                            conn.executescript(Database._FTS_TABLES_SQL)
                            logging.info(
                                f"FTS tables created for {db_instance.db_path_str} (client: {unique_client_id}).")
                        else:
                            logging.error(
                                "Database._FTS_TABLES_SQL is not defined or is empty. Cannot create FTS tables.")
                            pytest.fail("Cannot create FTS tables: _FTS_TABLES_SQL is missing or empty.")
                    else:
                        raise
        except Exception as schema_exc:
            logging.error(
                f"Failed to ensure schema for {db_instance.db_path_str} (client: {unique_client_id}): {schema_exc}",
                exc_info=True)
            pytest.fail(f"Schema creation failed for in-memory DB: {schema_exc}")

        created_db_instances.append(db_instance)
        return db_instance

    yield _create_db  # The fixture yields the creator function

    for db in created_db_instances:
        if hasattr(db, 'close_all_connections'):
            db.close_all_connections()
        elif hasattr(db, 'close_connection'):
            db.close_connection()


# --- Helper Functions ---

def check_batch_response(
        response,
        expected_status_code,
        expected_processed=None,
        expected_errors=None,
        check_results_len=None,
):
    """Helper to check common aspects of the batch response."""
    if response.status_code != expected_status_code:
        logger.error(f"Expected status {expected_status_code}, got {response.status_code}. Response text: {response.text}")
    assert response.status_code == expected_status_code
    try:
        data = response.json()
    except json.JSONDecodeError:
        pytest.fail(f"Failed to decode JSON. Status: {response.status_code}. Text: {response.text}")

    # Check top-level structure based on endpoint's response model
    # Processing endpoints might directly return a dict with counts/results
    assert "results" in data, f"Response missing 'results' key: {data}"
    assert "processed_count" in data, f"Response missing 'processed_count' key: {data}"
    assert "errors_count" in data, f"Response missing 'errors_count' key: {data}"
    assert "errors" in data, f"Response missing 'errors' key: {data}"
    assert isinstance(data["results"], list), f"'results' is not a list: {data}"

    # Check counts match expected values
    if expected_processed is not None:
        assert data["processed_count"] == expected_processed, f"Expected processed_count {expected_processed}, got {data['processed_count']}"
    if expected_errors is not None:
        assert data["errors_count"] == expected_errors, f"Expected errors_count {expected_errors}, got {data['errors_count']}"
    # Optionally check the length of the detailed errors list matches errors_count
    # assert len(data["errors"]) == data["errors_count"]

    if check_results_len is not None:
        assert len(data["results"]) == check_results_len, f"Expected {check_results_len} total results, got {len(data['results'])}"

    # Check consistency: processed + errors should equal total results length
    # assert data["processed_count"] + data["errors_count"] == len(data["results"])

    return data


def check_media_item_result(result, expected_status, check_db_fields=False): # Default check_db_fields=False
    """
    Helper to check structure of a single item in the results list.
    Updated for processing endpoints (no DB fields expected).
    """
    assert isinstance(result, dict), f"Result item is not a dictionary: {result}"
    assert "status" in result, "Result missing 'status' key"
    assert result["status"] == expected_status, f"Expected status '{expected_status}', got '{result['status']}'"
    assert "input_ref" in result, "Result missing 'input_ref' key"
    assert "processing_source" in result, "Result missing 'processing_source' key"
    assert "media_type" in result, "Result missing 'media_type' key"
    # Metadata might be None on error
    assert "metadata" in result, "Result missing 'metadata'"
    if result["status"] != "Error":
        assert isinstance(result.get("metadata"), dict), "'metadata' should be dict on success/warning"

    assert "content" in result, "Result missing 'content' key" # Allowed to be None or empty string
    assert "chunks" in result, "Result missing 'chunks' key" # Allowed to be None
    assert "analysis" in result, "Result missing 'analysis' key" # Allowed to be None
    # Analysis details should always be a dict, even if empty
    assert "analysis_details" in result and isinstance(result["analysis_details"], dict), "Result missing or invalid 'analysis_details'"
    assert "error" in result, "Result missing 'error' key" # Allowed to be None
    # Warnings should be None or a list
    assert "warnings" in result, "Result missing 'warnings'"
    assert result["warnings"] is None or isinstance(result["warnings"], list), "'warnings' must be None or list"

    if check_db_fields:
        # These fields SHOULD NOT be present or should be explicitly None/default message
        # for processing-only endpoints.
        assert "db_id" in result, "Result missing 'db_id' key"
        assert result["db_id"] is None, f"Expected db_id to be None for processing endpoint, got {result['db_id']}"
        assert "db_message" in result, "Result missing 'db_message' key"
        assert result["db_message"] in ["Processing only endpoint."], \
            f"Unexpected db_message for processing endpoint: {result['db_message']}"

    if expected_status == "Error":
        assert result["error"] is not None and result["error"] != "", "Expected non-empty 'error' for Error status"
    elif expected_status == "Success":
        assert result["error"] is None or result["error"] == "", \
             f"Expected None or empty error for Success status, got '{result['error']}'"


# --- Test Classes ---

class TestProcessVideos:
    ENDPOINT = "/api/v1/media/process-videos"

    def test_process_video_url_success(self, client, dummy_headers):
        """Test processing a single valid video URL."""
        form_data = {"urls": [VALID_VIDEO_URL], "perform_analysis": "false"}
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success", check_db_fields=True) # Check DB fields are explicitly None/default
        assert result["media_type"] == "video"
        assert result["input_ref"] == VALID_VIDEO_URL
        assert result["content"] is not None and len(result["content"]) > 0

    def test_process_video_upload_success(self, client, dummy_headers):
        """Test processing a single valid video file upload."""
        form_data = {"perform_analysis": "false"}
        with open(SAMPLE_VIDEO_PATH, "rb") as f:
            files = {"files": (SAMPLE_VIDEO_PATH.name, f, "video/mp4")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=dummy_headers)

        if response.status_code == 400 and "error parsing the body" in response.text.lower():
            pytest.fail("Still getting 400 'error parsing body' after auth fix (video upload).")

        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success", check_db_fields=True)
        assert result["media_type"] == "video"
        # In processing endpoints, input_ref *should* be the original filename
        assert result["input_ref"] == SAMPLE_VIDEO_PATH.name
        assert result["content"] is not None and len(result["content"]) > 0

    def test_process_video_multiple_success(self, client, dummy_headers):
        """Test processing multiple valid inputs (URL and Upload)."""
        form_data = {"urls": [VALID_VIDEO_URL], "perform_analysis": "false"}
        with open(SAMPLE_VIDEO_PATH, "rb") as f:
            files = {"files": (SAMPLE_VIDEO_PATH.name, f, "video/mp4")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=dummy_headers)

        if response.status_code == 400 and "error parsing the body" in response.text.lower():
            pytest.fail("Still getting 400 'error parsing body' after auth fix (video multi).")

        data = check_batch_response(response, 200, expected_processed=2, expected_errors=0, check_results_len=2)
        check_media_item_result(data["results"][0], "Success", check_db_fields=True)
        check_media_item_result(data["results"][1], "Success", check_db_fields=True)
        assert {r["media_type"] for r in data["results"]} == {"video"}
        assert {r["input_ref"] for r in data["results"]} == {VALID_VIDEO_URL, SAMPLE_VIDEO_PATH.name}

    def test_process_video_multi_status_mixed(self, client, dummy_headers):
        """Test processing one valid URL and one invalid URL -> 207."""
        form_data = {"urls": [VALID_VIDEO_URL, INVALID_URL], "perform_analysis": "false"}
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)

        # No sleep needed for sync processing endpoint failures
        data = check_batch_response(response, 207, expected_processed=1, expected_errors=1, check_results_len=2)

        success_result = next((r for r in data["results"] if r["status"] == "Success"), None)
        error_result = next((r for r in data["results"] if r["status"] == "Error"), None)

        assert success_result is not None
        assert error_result is not None
        check_media_item_result(success_result, "Success", check_db_fields=True)
        check_media_item_result(error_result, "Error", check_db_fields=True)
        assert success_result["input_ref"] == VALID_VIDEO_URL
        assert error_result["input_ref"] == INVALID_URL
        assert error_result["error"] is not None
        # Check for the specific metadata error OR common network errors
        assert "Failed to extract metadata" in error_result["error"] \
               or "failed to resolve" in error_result["error"] \
               or "Name or service not known" in error_result["error"] \
               or "timed out" in error_result["error"]

    def test_process_video_no_input(self, client, dummy_headers):
        """Test sending request with no URLs or files."""
        # Need to send *some* valid form data key for TestClient to not error early
        # Send a default value that doesn't affect processing much
        response = client.post(self.ENDPOINT, data={"perform_analysis": "false"}, headers=dummy_headers)
        # Expect 400 based on _validate_inputs logic inside the endpoint
        assert response.status_code == 400
        assert "No valid media sources supplied" in response.json()["detail"]

    def test_process_video_validation_error(self, client, dummy_headers):
        """Test sending invalid form data (e.g., bad chunk overlap)."""
        form_data = {
            "urls": [VALID_VIDEO_URL],
            "chunk_size": "100",
            "chunk_overlap": "200"  # Overlap > size
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)
        assert response.status_code == 422
        print("DEBUG: Actual response detail:", response.json().get('detail'))
        details = response.json().get('detail', [])  # Get details safely
        # print("DEBUG: Actual response detail:", details) # You can remove this print now if desired

        expected_error_fragment = "chunk_overlap must be less than chunk_size"  # <-- Key change here
        found_error = False
        if isinstance(details, list):
            for err_item in details:
                # Handle detail items being dictionaries
                msg = str(err_item.get("msg", "")) if isinstance(err_item, dict) else str(err_item)
                if expected_error_fragment in msg:
                    found_error = True
                    break
        elif isinstance(details, str):  # Handle if detail is just a string (less common)
            if expected_error_fragment in details:
                found_error = True

        assert found_error, f"Expected error message containing '{expected_error_fragment}' not found in detail: {details}"

    @pytest.mark.skip(reason="Analysis requires LLM setup or mocking")
    def test_process_video_with_analysis_and_chunking(self, client, dummy_headers):
        """Test enabling analysis and chunking (requires setup)."""
        form_data = {
            "urls": [VALID_VIDEO_URL],
            "perform_analysis": "true",
            "perform_chunking": "true",
            "chunk_size": "500",
            "chunk_overlap": "100",
            "api_name": "openai", # Replace or ensure configured/mocked
            "api_key": os.environ.get("OPENAI_API_KEY", "skip") # Replace or ensure configured/mocked
        }
        if form_data["api_key"] == "skip":
            pytest.skip("OPENAI_API_KEY not set, skipping analysis test")

        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success", check_db_fields=True)
        assert result["analysis"] is not None and len(result["analysis"]) > 0
        assert result["chunks"] is not None and len(result["chunks"]) > 0


class TestProcessAudios:
    ENDPOINT = "/api/v1/media/process-audios"

    def test_process_audio_url_success_no_analysis_no_chunking(self, client, dummy_headers):
        form_data = {
            "urls": [VALID_AUDIO_URL],
            "perform_analysis": "false",
            "perform_chunking": "false"
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success", check_db_fields=True)
        assert result["media_type"] == "audio"
        assert result["input_ref"] == VALID_AUDIO_URL
        # Expect content because transcription should still happen
        assert result["content"] is not None # Might be empty if transcription fails silently, but shouldn't be None
        assert isinstance(result["content"], str)
        assert result["segments"] is not None and isinstance(result["segments"], list)
        # Analysis and Chunks should be None as they were disabled
        assert result["analysis"] == "[Analysis Not Requested]" or result["analysis"] is None, f"Analysis should be None/Not Requested, got: {result['analysis']}"
        assert result["chunks"] is None, f"Chunks should be None, got: {result['chunks']}"
        # Check content length only if dummy file is guaranteed to produce output
        # On real audio, check > 0: assert len(result["content"]) > 0

    def test_process_audio_upload_success_defaults(self, client, dummy_headers):
        """Test processing audio file upload with default settings (chunking=True, analysis=True)."""
        # Requires analysis setup (API keys) or a mock LLM
        pytest.skip("Skipping test requiring analysis until LLM/API config is confirmed/mocked.")

        # Minimal form data, rely on defaults
        form_data = {
             # Add API key/name if needed for analysis, or ensure defaults work
             "api_name": "mock_llm", # Example: Use a mock if available
             # "api_key": "dummy_key"
        }
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            files = {"files": (SAMPLE_AUDIO_PATH.name, f, "audio/mpeg")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=dummy_headers)

        if response.status_code == 400 and "error parsing the body" in response.text.lower():
            pytest.fail("Still getting 400 'error parsing body' after auth fix (audio defaults).")

        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["media_type"] == "audio"
        assert result["input_ref"] == SAMPLE_AUDIO_PATH.name
        assert result["content"] is not None and len(result["content"]) > 0
        assert result["segments"] is not None and len(result["segments"]) > 0
        assert result["chunks"] is not None and len(result["chunks"]) > 0 # Expect chunks due to default
        assert result["analysis"] is not None and len(result["analysis"]) > 0 # Expect analysis due to default

    def test_process_audio_multi_status_mixed(self, client, dummy_headers):
        """Test one valid upload and one invalid URL -> 207."""
        form_data = {
            "urls": [URL_404], # Use a reliable 404 URL
            "perform_analysis": "false", # Disable analysis for faster test
            "perform_chunking": "true"   # Keep chunking enabled (default)
        }
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            # Ensure the dummy audio file has some content for transcription to work
            files = {"files": (SAMPLE_AUDIO_PATH.name, f, "audio/mpeg")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=dummy_headers)

        if response.status_code == 400 and "error parsing the body" in response.text.lower():
            pytest.fail("Still getting 400 'error parsing body' after auth fix (audio mixed).")

        data = check_batch_response(response, 207, expected_processed=1, expected_errors=1, check_results_len=2)

        success_result = next((r for r in data["results"] if r["status"] == "Success"), None)
        error_result = next((r for r in data["results"] if r["status"] == "Error"), None)

        assert success_result is not None, "Could not find success result"
        assert error_result is not None, "Could not find error result"

        check_media_item_result(success_result, "Success", check_db_fields=True)
        check_media_item_result(error_result, "Error", check_db_fields=True)

        # Check input refs carefully
        assert success_result["input_ref"] == SAMPLE_AUDIO_PATH.name
        assert error_result["input_ref"] == URL_404

        # Check error message for the failed URL
        assert error_result["error"] is not None
        assert "Download failed" in error_result["error"] or "404" in error_result["error"]

        # Check successful item results (assuming defaults enabled chunking)
        assert success_result["content"] is not None
        assert success_result["chunks"] is not None # Chunking was true

    def test_process_audio_upload_success(self, client, dummy_headers):
        """Test processing a single valid audio file upload."""
        form_data = {"perform_analysis": "false"}
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            files = {"files": (SAMPLE_AUDIO_PATH.name, f, "audio/mpeg")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=dummy_headers)

        if response.status_code == 400 and "error parsing the body" in response.text.lower():
            pytest.fail("Still getting 400 'error parsing body' after auth fix (audio upload success).")

        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        check_media_item_result(data["results"][0], "Success")
        assert data["results"][0]["media_type"] == "audio"
        assert data["results"][0]["input_ref"] == SAMPLE_AUDIO_PATH.name
        assert data["results"][0]["content"] is not None and len(data["results"][0]["content"]) > 0

    def test_process_audio_no_input(self, client, dummy_headers):
        """Test sending request with no URLs or files."""
        response = client.post(self.ENDPOINT, data={"perform_analysis": "false"}, headers=dummy_headers)
        assert response.status_code == 400
        assert "No valid media sources supplied" in response.json()["detail"]

    def test_process_audio_upload_invalid_format_pdf(self, client, dummy_headers):
        """Test uploading a non-audio file (PDF) which should fail early."""  # <-- Updated docstring slightly
        form_data = {"perform_analysis": "false"}  # Disable analysis
        with open(SAMPLE_PDF_PATH, "rb") as f:
            files = {"files": (SAMPLE_PDF_PATH.name, f, "application/pdf")}  # Correct MIME for PDFs
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=dummy_headers)

        if response.status_code == 400 and "error parsing the body" in response.text.lower():
            pytest.fail("Still getting 400 'error parsing body' after auth fix (audio invalid format).")

        logger.debug(f"Test received response status: {response.status_code}")
        try:
            response_data_in_test = response.json()
            logger.debug(f"Test received response JSON: {response_data_in_test}")
        except Exception as e:
            logger.error(f"Test failed to parse response JSON: {e}")
            response_data_in_test = None
        # Expect 207 because the request itself was handled, but yielded no processable items.
        # Expect 0 processed, 0 errors (processing errors), and 0 results because the file was rejected before processing.
        data = check_batch_response(response, 207, expected_processed=0, expected_errors=0, check_results_len=0)

        assert not data["results"], "Expected the 'results' list to be empty"

class TestProcessPdfs:
    ENDPOINT = "/api/v1/media/process-pdfs"

    def test_process_pdf_url_success(self, client, dummy_headers):
        """Test processing a single valid PDF URL."""
        # Use pymupdf4llm parser by default
        form_data = {"urls": [VALID_PDF_URL], "perform_analysis": "false"}
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success", check_db_fields=True)
        assert result["media_type"] == "pdf"
        assert result["input_ref"] == VALID_PDF_URL
        assert result["metadata"] is not None and isinstance(result["metadata"], dict)
        assert result["content"] is not None and len(result["content"]) > 0

    def test_process_pdf_upload_success(self, client, dummy_headers):
        """Test processing a single valid PDF file upload."""
        form_data = {"perform_analysis": "false"}
        with open(SAMPLE_PDF_PATH, "rb") as f:
            files = {"files": (SAMPLE_PDF_PATH.name, f, "application/pdf")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=dummy_headers)

        if response.status_code == 400 and "error parsing the body" in response.text.lower():
            pytest.fail("Still getting 400 'error parsing body' after auth fix (pdf upload).")

        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success", check_db_fields=True)
        assert result["media_type"] == "pdf"
        assert result["input_ref"] == SAMPLE_PDF_PATH.name
        assert result["content"] is not None and len(result["content"]) > 0

    def test_process_pdf_multi_status_mixed(self, client, dummy_headers):
        """Test one valid PDF upload, one invalid URL -> 207."""
        form_data = {"urls": [INVALID_URL], "perform_analysis": "false"}
        with open(SAMPLE_PDF_PATH, "rb") as f:
            files = {"files": (SAMPLE_PDF_PATH.name, f, "application/pdf")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=dummy_headers)

        if response.status_code == 400 and "error parsing the body" in response.text.lower():
            pytest.fail("Still getting 400 'error parsing body' after auth fix (pdf mixed).")

        data = check_batch_response(response, 207, expected_processed=1, expected_errors=1, check_results_len=2)
        success_result = next((r for r in data["results"] if r["status"] == "Success"), None)
        error_result = next((r for r in data["results"] if r["status"] == "Error"), None)

        assert success_result is not None
        assert error_result is not None
        check_media_item_result(success_result, "Success", check_db_fields=True)
        check_media_item_result(error_result, "Error", check_db_fields=True)
        assert success_result["input_ref"] == SAMPLE_PDF_PATH.name
        assert error_result["input_ref"] == INVALID_URL
        assert "Download failed" in error_result["error"] or "Download/preparation failed" in error_result["error"]

    def test_process_pdf_no_input(self, client, dummy_headers):
        response = client.post(self.ENDPOINT, data={"perform_analysis": "false"}, headers=dummy_headers)
        assert response.status_code == 400
        assert "No valid media sources supplied" in response.json()["detail"]

    def test_process_pdf_upload_not_a_pdf(self, client):
        form_data = {} # Minimal data
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            files = {"files": (SAMPLE_AUDIO_PATH.name, f, "audio/mpeg")}
            response = client.post(self.ENDPOINT, data=form_data, files=files)

        if response.status_code == 400 and "error parsing the body" in response.text.lower():
             pytest.fail("Still getting 400 'error parsing body' after auth fix (pdf invalid format).")

        # Endpoint might reject early based on filename/mime or library might fail
        data = check_batch_response(response, 207, expected_processed=0, expected_errors=1, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Error", check_db_fields=True)
        assert result["input_ref"] == SAMPLE_AUDIO_PATH.name
        # Check for errors related to PDF parsing
        assert "PDF processing failed" in result["error"] or "Invalid file type" in result["error"] or "failed to extract text" in result["error"]

    @patch("tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib.analyze")
    @pytest.mark.skip(reason="Analysis requires LLM setup or mocking")
    def test_process_pdf_with_analysis_and_chunking(self, mock_analyze, dummy_headers):
        """Test PDF analysis and chunking."""
        mock_analysis_text = "This is the mocked analysis result."
        # If analyze is async:
        async def async_mock_analyze(*args, **kwargs): return mock_analysis_text
        mock_analyze.side_effect = async_mock_analyze
        # Else: mock_analyze.return_value = mock_analysis_text

        form_data = {
            "urls": [VALID_PDF_URL],
            "perform_analysis": "true",
            "perform_chunking": "true",
            "chunk_size": "300",
            "chunk_overlap": "50",
            "api_name": "mock_api",
            "api_key": "mock_key"
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success", check_db_fields=True)

        mock_analyze.assert_called()
        assert result["analysis"] is not None
        assert mock_analysis_text in result["analysis"]
        assert result["chunks"] is not None
        assert isinstance(result["chunks"], list)
        assert len(result["chunks"]) > 0 # Should have at least one chunk
        # You could add more specific checks on chunk content/metadata if needed

        # Check that the mock was called (optional but good practice)
        mock_analyze.assert_called()



# =============================================================================
# Ebook Processing Tests
# =============================================================================
class TestProcessEbooks:
    ENDPOINT = "/api/v1/media/process-ebooks" # Make sure this path matches your router prefix + endpoint path

    # --- Happy Path Tests ---

    def test_process_ebook_url_success_defaults(self, client, dummy_headers):
        """Test processing a single valid EPUB URL with default settings."""
        # Defaults: analysis=True, chunking=True, extraction='filtered'
        # Skip analysis for speed if not mocking/configured
        pytest.skip("Skipping analysis test until LLM mock/config confirmed.")
        form_data = {
            "urls": [VALID_EPUB_URL],
            "api_name": "mock_api", # Needed if analysis=True default
            "api_key": "mock_key"   # Needed if analysis=True default
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["media_type"] == "ebook"
        assert result["input_ref"] == VALID_EPUB_URL
        assert result["content"] is not None and len(result["content"]) > 0
        assert result["chunks"] is not None and len(result["chunks"]) > 0 # Expect chunks due to default
        assert result["analysis"] is not None and len(result["analysis"]) > 0 # Expect analysis due to default

    def test_process_ebook_url_success_no_analysis_no_chunking(self, client, dummy_headers):
        """Test processing EPUB URL, disabling analysis and chunking."""
        form_data = {
            "urls": [VALID_EPUB_URL],
            "perform_analysis": "false",
            "perform_chunking": "false",
            "extraction_method": "basic" # Test another extraction method
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["media_type"] == "ebook"
        assert result["input_ref"] == VALID_EPUB_URL
        assert result["content"] is not None and len(result["content"]) > 0
        # Library creates one chunk if chunking is off
        assert result["chunks"] is not None and len(result["chunks"]) == 1
        assert result["analysis"] is None # Analysis was disabled

    def test_process_ebook_upload_success_defaults(self, client, dummy_headers):
        """Test processing a single valid EPUB file upload with defaults."""
        # Skip analysis for speed if not mocking/configured
        form_data = {"perform_analysis": "false"}
        with open(SAMPLE_EPUB_PATH, "rb") as f:
            # Common EPUB MIME type, though server might not strictly check it
            files = {"files": (SAMPLE_EPUB_PATH.name, f, "application/epub+zip")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=dummy_headers)

        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["media_type"] == "ebook"
        assert result["input_ref"] == SAMPLE_EPUB_PATH.name
        assert result["content"] is not None and len(result["content"]) > 0
        assert result["chunks"] is not None and len(result["chunks"]) > 0 # Default chunking=True

    def test_process_ebook_multiple_success(self, client, dummy_headers):
        """Test processing multiple valid inputs (URL and Upload)."""
        form_data = {"urls": [VALID_EPUB_URL], "perform_analysis": "false"}
        with open(SAMPLE_EPUB_PATH, "rb") as f:
            files = {"files": (SAMPLE_EPUB_PATH.name, f, "application/epub+zip")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=dummy_headers)

        data = check_batch_response(response, 200, expected_processed=2, expected_errors=0, check_results_len=2)
        results = data["results"]
        assert len(results) == 2
        # Check both results individually
        found_url = False
        found_file = False
        for res in results:
            check_media_item_result(res, "Success")
            assert res["media_type"] == "ebook"
            if res["input_ref"] == VALID_EPUB_URL:
                found_url = True
            elif res["input_ref"] == SAMPLE_EPUB_PATH.name:
                found_file = True
        assert found_url and found_file, "Did not find results for both URL and file"

    def test_process_ebook_overrides(self, client, dummy_headers):
        """Test applying title, author, and keyword overrides."""
        test_title = "My Custom Ebook Title"
        test_author = "Testy McTestface"
        test_keywords_str = "test,ebook,override"
        test_keywords_list = ["test", "ebook", "override"]

        form_data = {
            "urls": [VALID_EPUB_URL],
            "title": test_title,
            "author": test_author,
            "keywords_str": test_keywords_str, # Use keywords_str for form data
            "perform_analysis": "false"
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["metadata"]["title"] == test_title
        assert result["metadata"]["author"] == test_author
        assert result["keywords"] == test_keywords_list # Check the parsed list

    # --- Error Handling Tests ---

    def test_process_ebook_multi_status_mixed(self, client, dummy_headers):
        """Test processing one valid URL and one invalid URL -> 207."""
        form_data = {"urls": [VALID_EPUB_URL, INVALID_URL], "perform_analysis": "false"}
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)

        # Give potentially slow download/timeout a moment
        time.sleep(5)

        data = check_batch_response(response, 207, expected_processed=1, expected_errors=1, check_results_len=2)

        success_result = next((r for r in data["results"] if r["status"] == "Success"), None)
        error_result = next((r for r in data["results"] if r["status"] == "Error"), None)

        assert success_result is not None, "Success result not found"
        assert error_result is not None, "Error result not found"
        check_media_item_result(success_result, "Success")
        check_media_item_result(error_result, "Error")
        assert success_result["input_ref"] == VALID_EPUB_URL
        assert error_result["input_ref"] == INVALID_URL
        assert error_result["error"] is not None
        assert "Download/preparation failed" in error_result["error"] # Check download helper error

    def test_process_ebook_no_input(self, client, dummy_headers):
        """Test sending request with no URLs or files."""
        response = client.post(self.ENDPOINT, data={}, headers=dummy_headers)
        # Expect 400 based on _validate_inputs logic
        assert response.status_code == 400, f"Expected 400, got {response.status_code}. Body: {response.text}"
        assert "At least one 'url' in the 'urls' list or one 'file' in the 'files' list must be provided." in response.json()["detail"]

    def test_process_ebook_upload_invalid_format(self, client, dummy_headers):
        """Test uploading a non-EPUB file (e.g., PDF), expecting upload helper rejection."""
        form_data = {}
        with open(SAMPLE_PDF_PATH, "rb") as f:
            files = {"files": (SAMPLE_PDF_PATH.name, f, "application/pdf")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=dummy_headers)

        # _save_uploaded_files should reject based on extension ".epub"
        data = check_batch_response(response, 207, expected_processed=0, expected_errors=1, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Error")
        assert result["input_ref"] == SAMPLE_PDF_PATH.name
        # Check for the key part of the actual error message
        assert "Invalid file type" in result["error"]
        assert ".pdf" in result["error"]  # Also check that the specific wrong extension is mentioned
        assert ".epub" in result["error"]  # And that the allowed extension is mentioned

    def test_process_ebook_validation_error_bad_method(self, client, dummy_headers):
        """Test sending invalid form data (invalid extraction_method)."""
        form_data = {
            "urls": [VALID_EPUB_URL],
            "extraction_method": "invalid_method" # Not in Literal[...]
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)
        assert response.status_code == 422  # Unprocessable Entity
        detail = response.json()["detail"]
        assert isinstance(detail, list) and len(detail) > 0
        # Check for the specific validation error message
        assert any("Input should be 'filtered', 'markdown' or 'basic'" in err.get("msg", "") for err in detail)
        assert any(err.get("loc") == ["body", "extraction_method"] for err in detail) # Check location


    # --- Option Variation Tests ---

    @pytest.mark.parametrize("method", ['filtered', 'markdown', 'basic'])
    def test_process_ebook_options_extraction_method(self, method, client, dummy_headers):
        """Test different valid extraction methods."""
        form_data = {
            "urls": [VALID_EPUB_URL],
            "extraction_method": method,
            "perform_analysis": "false"
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["content"] is not None # Content should exist, might differ based on method

    # --- Mocked Analysis Test ---

    # IMPORTANT: Replace "path.to.your_ebook_processing_module.analyze" with the correct path
    @patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Books.Book_Processing_Lib.analyze")
    def test_process_ebook_with_analysis_mocked(self, mock_analyze, client):
        """Test enabling analysis with mocking."""
        mock_analysis_text = "This is the mocked ebook analysis."
        mock_analyze.return_value = mock_analysis_text

        form_data = {
            "urls": [VALID_EPUB_URL],
            "perform_analysis": "true",
            "perform_chunking": "true", # Analysis often depends on chunks
            "api_name": "mock_api",     # Need to provide these even if mocking analyze directly
            "api_key": "mock_key"       # Depending on process_epub implementation checks
        }
        response = client.post(self.ENDPOINT, data=form_data)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")

        # Check mock was called (at least once, maybe more if chunked)
        mock_analyze.assert_called()
        # Check the *final* analysis result
        assert result["analysis"] is not None
        # The final result might be the mocked text joined, or a recursive summary
        # For simplicity here, we check if the mocked text is *in* the final result
        assert mock_analysis_text in result["analysis"]
        assert result["chunks"] is not None and len(result["chunks"]) > 0
        # Check if analysis was added to chunk metadata
        assert all('analysis' in chunk.get('metadata', {}) for chunk in result["chunks"])



# =============================================================================
# Document Processing Tests
# =============================================================================
class TestProcessDocuments:
    ENDPOINT = "/api/v1/media/process-documents" # Adjust if needed

    # --- Helper to create dummy files ---
    @pytest.fixture(autouse=True, scope="class")
    def create_sample_doc_files(self):
        """Create dummy files for tests if they don't exist."""
        TEST_MEDIA_DIR.mkdir(exist_ok=True)
        if not SAMPLE_TXT_PATH.exists(): SAMPLE_TXT_PATH.write_text("This is sample text content.\nSecond line.", encoding='utf-8')
        if not SAMPLE_MD_PATH.exists(): SAMPLE_MD_PATH.write_text("# Sample Markdown\n\nThis is *markdown*.\n\n- Item 1\n- Item 2", encoding='utf-8')
        # Cannot easily create binary docx/rtf here, assume they exist or skip tests needing them
        if not SAMPLE_HTML_PATH.exists(): SAMPLE_HTML_PATH.write_text("<html><head><title>Sample HTML</title><meta name='author' content='HTML Author'></head><body><p>Hello World</p></body></html>", encoding='utf-8')
        if not SAMPLE_XML_PATH.exists(): SAMPLE_XML_PATH.write_text("<root><title>Sample XML</title><data>Some XML data</data></root>", encoding='utf-8')
        # NOTE: For DOCX/RTF tests to pass conversion, ensure valid files exist at SAMPLE_DOCX_PATH/SAMPLE_RTF_PATH
        # Or mock the conversion functions (docx2txt.process, pypandoc.convert_file)

    # --- Happy Path Tests ---

    @pytest.mark.parametrize("file_path, mime_type, expected_format", [
        (SAMPLE_TXT_PATH, "text/plain", "txt"),
        (SAMPLE_MD_PATH, "text/markdown", "md"),
        (SAMPLE_HTML_PATH, "text/html", "html"),
        (SAMPLE_XML_PATH, "application/xml", "xml"),
        pytest.param(SAMPLE_DOCX_PATH, "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx", marks=pytest.mark.skipif(not SAMPLE_DOCX_PATH.exists(), reason="sample.docx not found")),
        pytest.param(SAMPLE_RTF_PATH, "application/rtf", "rtf", marks=[pytest.mark.skipif(not SAMPLE_RTF_PATH.exists(), reason="sample.rtf not found"), pytest.mark.xfail(reason="Requires pandoc binary installed")])
    ])
    def test_process_doc_upload_various_formats(self, file_path, mime_type, expected_format, client, dummy_headers):
        """Test uploading various supported document formats."""
        form_data = {"perform_analysis": "false"}
        with open(file_path, "rb") as f:
            files = {"files": (file_path.name, f, mime_type)}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=dummy_headers)

        if response.status_code == 400 and "error parsing the body" in response.text.lower():
            pytest.fail(f"Still getting 400 'error parsing body' after auth fix ({file_path.name}).")

        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["media_type"] == "document"
        assert result["input_ref"] == file_path.name
        assert result["source_format"] == expected_format
        assert result["content"] is not None and len(result["content"]) > 0
        assert result["chunks"] is not None and len(result["chunks"]) > 0 # Default chunking=True

    @pytest.mark.parametrize("url, check_content_part, expected_status, expected_error_part", [
        (VALID_TXT_URL, "license", 200, None),
        (VALID_MD_URL, "FastAPI", 200, None),
        pytest.param(VALID_HTML_URL, None, 207, "does not have an allowed extension",
                     marks=pytest.mark.skipif(not VALID_HTML_URL, reason="VALID_HTML_URL not defined"))
    ])
    def test_process_doc_url_various_formats(self, url, check_content_part, expected_status, expected_error_part, client, dummy_headers):
        """Test processing various document URLs."""
        form_data = {"urls": [url], "perform_analysis": "false"}
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)

        # Adjust expected counts based on status
        expected_processed = 1 if expected_status == 200 else 0
        expected_errors = 1 if expected_status == 207 else 0

        # Use the check_batch_response helper, passing the expected status
        data = check_batch_response(response, expected_status,
                                    expected_processed=expected_processed,
                                    expected_errors=expected_errors,
                                    check_results_len=1)
        result = data["results"][0]

        if expected_status == 200:
            check_media_item_result(result, "Success")
            assert result["media_type"] == "document"
            assert result["input_ref"] == url
            assert result["content"] is not None and len(result["content"]) > 0
            assert check_content_part in result["content"] # Check if expected text is present
            assert result["chunks"] is not None and len(result["chunks"]) > 0
        else: # Expected 207 (failure)
            check_media_item_result(result, "Error")
            assert result["input_ref"] == url
            assert result["error"] is not None
            assert expected_error_part in result["error"]

    def test_process_doc_multiple_success(self, client, dummy_headers):
        """Test processing multiple valid inputs (URL and Upload)."""
        form_data = {"urls": [VALID_TXT_URL], "perform_analysis": "false"}
        with open(SAMPLE_MD_PATH, "rb") as f:
            files = {"files": (SAMPLE_MD_PATH.name, f, "text/markdown")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=dummy_headers)

        data = check_batch_response(response, 200, expected_processed=2, expected_errors=0, check_results_len=2)
        results = data["results"]
        assert len(results) == 2
        found_url = False
        found_file = False
        for res in results:
            check_media_item_result(res, "Success")
            assert res["media_type"] == "document"
            if res["input_ref"] == VALID_TXT_URL: found_url = True
            elif res["input_ref"] == SAMPLE_MD_PATH.name: found_file = True
        assert found_url and found_file

    def test_process_doc_overrides_and_options(self, client, dummy_headers):
        """Test title/author/keywords overrides and disabling chunking."""
        test_title = "My Doc Title"
        test_author = "Doc Author"
        test_keywords_str = "doc,test,override"
        test_keywords_list = ["doc", "test", "override"]

        form_data = {
            "urls": [VALID_TXT_URL],
            "title": test_title,
            "author": test_author,
            "keywords_str": test_keywords_str,
            "perform_analysis": "false",
            "perform_chunking": "false" # Disable chunking
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["metadata"]["title"] == test_title
        assert result["metadata"]["author"] == test_author
        assert result["keywords"] == test_keywords_list
        assert result["chunks"] is not None and len(result["chunks"]) == 1 # Expect 1 chunk when disabled

    # --- Error Handling Tests ---

    def test_process_doc_multi_status_mixed(self, client, dummy_headers):
        """Test processing one valid URL and one invalid URL -> 207."""
        form_data = {"urls": [VALID_TXT_URL, INVALID_URL], "perform_analysis": "false"}
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)

        time.sleep(5) # Give download time to fail

        data = check_batch_response(response, 207, expected_processed=1, expected_errors=1, check_results_len=2)
        success_result = next((r for r in data["results"] if r["status"] == "Success"), None)
        error_result = next((r for r in data["results"] if r["status"] == "Error"), None)

        assert success_result is not None
        assert error_result is not None
        check_media_item_result(success_result, "Success")
        check_media_item_result(error_result, "Error")
        assert success_result["input_ref"] == VALID_TXT_URL
        assert error_result["input_ref"] == INVALID_URL
        assert "Download/preparation failed" in error_result["error"]

    def test_process_doc_no_input(self, client, dummy_headers):
        """Test sending request with no URLs or files."""
        response = client.post(self.ENDPOINT, data={}, headers=dummy_headers)
        assert response.status_code == 400
        assert "At least one 'url' in the 'urls' list or one 'file' in the 'files' list must be provided." in response.json()["detail"]

    def test_process_doc_upload_invalid_extension(self, client, dummy_headers):
        """Test uploading a file with an unsupported extension (e.g., epub)."""
        # Use epub as an example of unsupported by this endpoint's ALLOWED_DOC_EXTENSIONS
        if not SAMPLE_EPUB_PATH.exists(): pytest.skip("sample.epub needed for invalid format test")
        form_data = {}
        with open(SAMPLE_EPUB_PATH, "rb") as f:
            files = {"files": (SAMPLE_EPUB_PATH.name, f, "application/epub+zip")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers= dummy_headers)

        # _save_uploaded_files should reject based on ALLOWED_DOC_EXTENSIONS
        data = check_batch_response(response, 207, expected_processed=0, expected_errors=1, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Error")
        assert result["input_ref"] == SAMPLE_EPUB_PATH.name
        assert "Upload error: Invalid file type" in result["error"] # Check based on _save_uploaded_files msg
        assert ".epub" in result["error"]

    @pytest.mark.skipif(not SAMPLE_RTF_PATH.exists(), reason="sample.rtf not found")
    @patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files.convert_file", side_effect=ValueError("Mocked pandoc failure"))
    def test_process_doc_rtf_conversion_failure(self, mock_convert, client):
        """Test RTF processing when pandoc conversion fails."""
        form_data = {"perform_analysis": "false"}
        with open(SAMPLE_RTF_PATH, "rb") as f:
            files = {"files": (SAMPLE_RTF_PATH.name, f, "application/rtf")}
            response = client.post(self.ENDPOINT, data=form_data, files=files)

        data = check_batch_response(response, 207, expected_processed=0, expected_errors=1, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Error")
        assert result["input_ref"] == SAMPLE_RTF_PATH.name
        assert "RTF conversion failed" in result["error"] # Check the prefix
        assert "Mocked pandoc failure" in result["error"] # Check the original mocked message is included

    def test_process_doc_validation_error_chunking(self, client, dummy_headers):
        """Test invalid chunking parameters -> 422."""
        form_data = {
            "urls": [VALID_TXT_URL],
            "chunk_size": "50",
            "chunk_overlap": "100" # Overlap > size
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)
        assert response.status_code == 422
        assert "chunk_overlap must be less than chunk_size" in str(response.json())

    # --- Mocked Analysis Test ---
    # IMPORTANT: Update patch path if needed
    @patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files.analyze")
    def test_process_doc_with_analysis_mocked(self, mock_analyze, client, dummy_headers):
        """Test enabling analysis with mocking."""
        mock_analysis_text = "This is the mocked document analysis."
        mock_analyze.return_value = mock_analysis_text

        form_data = {
            "urls": [VALID_TXT_URL],
            "perform_analysis": "true",
            "perform_chunking": "true",
            "api_name": "mock_api",
            "api_key": "mock_key"
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=dummy_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")

        mock_analyze.assert_called()
        assert result["analysis"] is not None
        assert mock_analysis_text in result["analysis"] # Check if mocked text is present
        assert result["chunks"] is not None and len(result["chunks"]) > 0
        # Check analysis_details
        assert result["analysis_details"]["analysis_model"] == "mock_api"


# ============================================================
# Test Class for Document Versioning and Rollback (V2)
# ============================================================
class TestDocumentVersioningV2:

    @pytest.fixture
    def db_instance(self, memory_db_factory_local):  # <--- Use the locally defined factory
        """Provides a fresh in-memory DB for each test in this class."""
        # The FTS creation is now handled inside _create_db by the factory
        db = memory_db_factory_local("versioning_client_v2")
        return db

    # === Test Document Version Creation (implicitly tested by add_media_with_keywords) ===
    # We rely on add_media_with_keywords to create the initial version correctly.
    # We can add specific tests if create_document_version is used independently.

    def test_create_document_version_standalone(self, db_instance):
        """Test creating subsequent document versions explicitly."""
        title = "Doc Version Test"
        content1 = "Content V1"
        # Setup initial media and first version
        media_id, media_uuid, _ = db_instance.add_media_with_keywords(
            title=title, content=content1, media_type="document"
        )
        assert get_entity_version(db_instance, "Media", media_uuid) == 1
        initial_doc_version = get_document_version(db_instance, media_id, version_number=1)
        assert initial_doc_version is not None
        assert initial_doc_version['content'] == content1
        assert initial_doc_version['version'] == 1 # Sync version of the doc version record

        # Create a second version using the instance method
        content2 = "Content V2 - Updated"
        prompt2 = "Prompt for V2"
        analysis2 = "Analysis for V2"
        # Use the instance method directly
        result_dict = db_instance.create_document_version(
            media_id=media_id,
            content=content2,
            prompt=prompt2,
            analysis_content=analysis2
        )

        assert isinstance(result_dict, Dict)
        assert result_dict['media_id'] == media_id
        assert result_dict['version_number'] == 2 # Second version
        new_doc_version_uuid = result_dict['uuid']

        # Verify DB state for version 2
        v2_data = get_document_version(db_instance, media_id, version_number=2)
        assert v2_data is not None
        assert v2_data['uuid'] == new_doc_version_uuid
        assert v2_data['content'] == content2
        assert v2_data['prompt'] == prompt2
        assert v2_data['analysis_content'] == analysis2
        assert v2_data['version'] == 1 # Initial sync version for *this* row
        assert v2_data['client_id'] == db_instance.client_id
        assert not v2_data['deleted']

        # Verify Sync Log for the new DocumentVersion
        log_entry = get_latest_log(db_instance, new_doc_version_uuid)
        assert log_entry is not None
        assert log_entry['entity'] == 'DocumentVersions'
        assert log_entry['operation'] == 'create'
        assert log_entry['version'] == 1
        assert log_entry['client_id'] == db_instance.client_id
        payload = json.loads(log_entry['payload'])
        assert payload['uuid'] == new_doc_version_uuid
        assert payload['version_number'] == 2
        assert payload['content'] == content2

    # === Test Getting Document Versions ===

    @pytest.mark.skipif(get_document_version is None, reason="get_document_version not imported")
    def test_get_document_version_latest(self, db_instance):
        media_id, _, _ = db_instance.add_media_with_keywords(title="GetLatest", content="V1", media_type="doc")
        db_instance.create_document_version(media_id=media_id, content="V2 Content", prompt="P2", analysis_content="A2")

        result = get_document_version(db_instance=db_instance, media_id=media_id) # version_number=None gets latest active

        assert result is not None
        assert result['version_number'] == 2
        assert result['content'] == "V2 Content"
        assert result['prompt'] == "P2"
        assert result['analysis_content'] == "A2"
        assert not result['deleted'] # Ensure it's active

    @pytest.mark.skipif(get_document_version is None, reason="get_document_version not imported")
    def test_get_document_version_specific(self, db_instance):
        media_id, _, _ = db_instance.add_media_with_keywords(title="GetSpecific", content="V1 Content", prompt="P1", analysis_content="A1", media_type="doc")
        db_instance.create_document_version(media_id=media_id, content="V2 Content", prompt="P2", analysis_content="A2")

        result = get_document_version(db_instance=db_instance, media_id=media_id, version_number=1)

        assert result is not None
        assert result['version_number'] == 1
        assert result['content'] == "V1 Content"
        assert result['prompt'] == "P1"
        assert result['analysis_content'] == "A1"
        assert not result['deleted']

    @pytest.mark.skipif(get_document_version is None, reason="get_document_version not imported")
    def test_get_document_version_without_content(self, db_instance):
        media_id, _, _ = db_instance.add_media_with_keywords(title="GetNoContent", content="V1 Content", prompt="P1", media_type="doc")

        result = get_document_version(db_instance=db_instance, media_id=media_id, version_number=1, include_content=False)

        assert result is not None
        assert result['version_number'] == 1
        assert 'content' not in result # Content should be excluded
        assert result['prompt'] == "P1"

    @pytest.mark.skipif(get_document_version is None, reason="get_document_version not imported")
    def test_get_document_version_not_found(self, db_instance):
        media_id, _, _ = db_instance.add_media_with_keywords(title="GetNotFound", content="V1", media_type="doc")

        # Specific version doesn't exist
        result_specific = get_document_version(db_instance=db_instance, media_id=media_id, version_number=5)
        assert result_specific is None

        # Media ID doesn't exist
        result_bad_media = get_document_version(db_instance=db_instance, media_id=999)
        assert result_bad_media is None

    # === Test Getting All Document Versions ===

    @pytest.mark.skipif(get_all_document_versions is None, reason="get_all_document_versions not imported")
    def test_get_all_document_versions(self, db_instance):
        media_id, _, _ = db_instance.add_media_with_keywords(title="GetAll", content="V1", prompt="P1", analysis_content="A1", media_type="doc")
        db_instance.create_document_version(media_id=media_id, content="V2", prompt="P2", analysis_content="A2")
        # Create a deleted version - should NOT be returned by default
        deleted_v3_info = db_instance.create_document_version(media_id=media_id, content="V3 - Deleted")
        db_instance.soft_delete_document_version(version_uuid=deleted_v3_info['uuid'])


        # Default: include_content=True, include_deleted=False
        results = get_all_document_versions(db_instance=db_instance, media_id=media_id)

        assert len(results) == 2 # Only active versions
        # Results are ordered DESC by version_number
        assert results[0]['version_number'] == 2
        assert results[0]['content'] == "V2"
        assert results[0]['prompt'] == "P2"
        assert results[0]['analysis_content'] == "A2"
        assert not results[0]['deleted']
        assert results[1]['version_number'] == 1
        assert results[1]['content'] == "V1"
        assert results[1]['prompt'] == "P1"
        assert results[1]['analysis_content'] == "A1"
        assert not results[1]['deleted']

    @pytest.mark.skipif(get_all_document_versions is None, reason="get_all_document_versions not imported")
    def test_get_all_document_versions_pagination(self, db_instance):
        media_id, _, _ = db_instance.add_media_with_keywords(title="GetAllPaginated", content="V1", media_type="doc")
        db_instance.create_document_version(media_id=media_id, content="V2")
        db_instance.create_document_version(media_id=media_id, content="V3")

        # Get page 1 (limit 2) - Should be V3, V2
        results_p1 = get_all_document_versions(db_instance=db_instance, media_id=media_id, limit=2, offset=0)
        # Get page 2 (limit 2, skip 2) - Should be V1
        results_p2 = get_all_document_versions(db_instance=db_instance, media_id=media_id, limit=2, offset=2)

        assert len(results_p1) == 2
        assert results_p1[0]['version_number'] == 3 # Latest
        assert results_p1[1]['version_number'] == 2

        assert len(results_p2) == 1
        assert results_p2[0]['version_number'] == 1

    # === Test Deleting Document Versions ===

    def test_delete_document_version_success(self, db_instance):
        media_id, _, _ = db_instance.add_media_with_keywords(title="DeleteVersion", content="V1", media_type="doc")
        v2_info = db_instance.create_document_version(media_id=media_id, content="V2")
        v2_uuid = v2_info['uuid']
        initial_sync_version = get_entity_version(db_instance, "DocumentVersions", v2_uuid) # Should be 1

        result = db_instance.soft_delete_document_version(version_uuid=v2_uuid)

        assert result is True # Method returns bool

        # Verify DB state for version 2
        cursor = db_instance.execute_query("SELECT deleted, version FROM DocumentVersions WHERE uuid = ?", (v2_uuid,))
        row = cursor.fetchone()
        assert row is not None
        assert row['deleted'] == 1
        assert row['version'] == initial_sync_version + 1

        # Verify V1 still exists and is active
        v1 = get_document_version(db_instance, media_id=media_id, version_number=1)
        assert v1 is not None
        assert not v1['deleted']

        # Verify V2 is considered deleted by get_document_version
        v2_get = get_document_version(db_instance, media_id=media_id, version_number=2)
        assert v2_get is None # Default get function should filter deleted

        # Verify Sync Log
        log_entry = get_latest_log(db_instance, v2_uuid)
        assert log_entry['operation'] == 'delete'
        assert log_entry['entity'] == 'DocumentVersions'
        assert log_entry['version'] == initial_sync_version + 1

    def test_delete_last_document_version_fails(self, db_instance):
        media_id, media_uuid, _ = db_instance.add_media_with_keywords(title="DeleteLastVersion", content="V1", media_type="doc")
        v1_info = get_document_version(db_instance, media_id=media_id, version_number=1)
        assert v1_info is not None
        v1_uuid = v1_info['uuid']

        result = db_instance.soft_delete_document_version(version_uuid=v1_uuid)

        assert result is False # Should fail to delete the last active one

        # Verify version 1 still exists and is active
        v1_check = get_document_version(db_instance, media_id=media_id, version_number=1)
        assert v1_check is not None
        assert not v1_check['deleted']
        # Verify DB state didn't change
        assert get_entity_version(db_instance, "DocumentVersions", v1_uuid) == 1 # Still version 1

    def test_delete_nonexistent_version_fails(self, db_instance):
        media_id, _, _ = db_instance.add_media_with_keywords(title="DeleteNonExistent", content="V1", media_type="doc")
        non_existent_uuid = db_instance._generate_uuid() # Generate a valid but unused UUID

        result = db_instance.soft_delete_document_version(version_uuid=non_existent_uuid)

        assert result is False # Fails because UUID doesn't exist

    # === Test Rollback to Version ===

    def test_rollback_to_version_success(self, db_instance):
        # 1. Setup: Media with two versions
        title = "Rollback Test"
        content1 = "Version 1 content - Original"
        prompt1 = "Prompt V1"
        analysis1 = "Analysis V1"
        content2 = "Version 2 content - Changed"
        prompt2 = "Prompt V2" # Add distinct prompt for V2

        media_id, media_uuid, _ = db_instance.add_media_with_keywords(
            title=title, content=content1, prompt=prompt1, analysis_content=analysis1, media_type="doc"
        ) # Creates V1 doc version implicitly
        media_version_after_v1 = get_entity_version(db_instance, "Media", media_uuid) # Should be 1

        v2_info = db_instance.create_document_version(media_id=media_id, content=content2, prompt=prompt2)
        v2_num = v2_info['version_number'] # Should be 2

        # 2. Perform Rollback to version 1
        result = db_instance.rollback_to_version(media_id=media_id, target_version_number=1)

        # 3. Verify Result Dictionary
        assert 'success' in result
        assert 'error' not in result
        assert result['new_document_version_number'] == 3 # V3 created from V1 state
        assert result['new_media_version'] == media_version_after_v1 + 1 # Media version incremented
        new_doc_ver_uuid = result['new_document_version_uuid']
        assert isinstance(new_doc_ver_uuid, str)
        assert f"Rolled back to version 1. State saved as new version 3." in result['success']

        # 4. Verify New Document Version (V3)
        latest_version_data = get_document_version(db_instance=db_instance, media_id=media_id) # Get latest (should be V3)
        assert latest_version_data is not None
        assert latest_version_data['version_number'] == 3
        assert latest_version_data['uuid'] == new_doc_ver_uuid
        assert latest_version_data['content'] == content1 # Content matches V1
        assert latest_version_data['prompt'] == prompt1 # Prompt matches V1
        assert latest_version_data['analysis_content'] == analysis1 # Analysis matches V1
        assert latest_version_data['version'] == 1 # Sync version of V3 record
        assert latest_version_data['client_id'] == db_instance.client_id
        assert not latest_version_data['deleted']

        # 5. Verify Media Record Update
        media_row = db_instance.get_media_by_id(media_id)
        assert media_row is not None
        assert media_row['content'] == content1 # Media content updated
        expected_hash = hashlib.sha256(content1.encode()).hexdigest()
        assert media_row['content_hash'] == expected_hash
        assert media_row['version'] == media_version_after_v1 + 1 # Media sync version incremented
        assert media_row['client_id'] == db_instance.client_id # Client ID updated
        # last_modified should also be updated, harder to assert exact value

        # 6. Verify Sync Logs
        # Check DocumentVersion create log for V3
        v3_log = get_latest_log(db_instance, new_doc_ver_uuid)
        assert v3_log is not None
        assert v3_log['entity'] == 'DocumentVersions'
        assert v3_log['operation'] == 'create'
        assert v3_log['version'] == 1
        assert v3_log['client_id'] == db_instance.client_id
        payload_v3 = json.loads(v3_log['payload'])
        assert payload_v3['content'] == content1

        # Check Media update log
        media_log = get_latest_log(db_instance, media_uuid)
        assert media_log is not None
        assert media_log['entity'] == 'Media'
        assert media_log['operation'] == 'update'
        assert media_log['version'] == media_version_after_v1 + 1
        assert media_log['client_id'] == db_instance.client_id
        payload_media = json.loads(media_log['payload'])
        assert payload_media['content'] == content1
        assert payload_media['uuid'] == media_uuid
        assert payload_media['rolled_back_to_doc_ver_uuid'] == new_doc_ver_uuid # Check extra context

        # 7. Verify FTS Update
        # Search for the rolled-back content (V1 content)
        results_fts, total_fts = search_media_db(db_instance, search_query='"Original"', search_fields=["content"])
        assert total_fts == 1
        assert results_fts[0]['id'] == media_id

        # Search for the previous content (V2 content) - should not be found in main search
        results_fts_old, total_fts_old = search_media_db(db_instance, search_query='"Changed"', search_fields=["content"])
        assert total_fts_old == 0

    def test_rollback_to_nonexistent_version(self, db_instance):
        media_id, _, _ = db_instance.add_media_with_keywords(title="RollbackNonExistent", content="V1", media_type="doc")
        initial_media_version = get_entity_version(db_instance, "Media", _)

        result = db_instance.rollback_to_version(media_id=media_id, target_version_number=5) # Version 5 doesn't exist

        assert 'error' in result
        assert 'success' not in result
        assert result['error'] == 'Rollback target version 5 not found or inactive.'

        # Verify media record was NOT updated
        assert get_entity_version(db_instance, "Media", _) == initial_media_version
        media_content = db_instance.get_media_by_id(media_id)['content']
        assert media_content == "V1"

        # Verify no new document version was created
        latest_version = get_document_version(db_instance, media_id=media_id)
        assert latest_version['version_number'] == 1

    def test_rollback_to_latest_version_fails(self, db_instance):
        media_id, media_uuid, _ = db_instance.add_media_with_keywords(title="RollbackLatest", content="V1", media_type="doc")
        db_instance.create_document_version(media_id=media_id, content="V2") # V2 is latest

        result = db_instance.rollback_to_version(media_id=media_id, target_version_number=2) # Attempt rollback to latest

        assert 'error' in result
        assert 'success' not in result
        assert result['error'] == 'Cannot rollback to the current latest version number.'

        # Verify no changes occurred
        assert get_entity_version(db_instance, "Media", media_uuid) == 1 # Still initial media version
        latest_version = get_document_version(db_instance, media_id=media_id)
        assert latest_version['version_number'] == 2 # Still V2
        assert latest_version['content'] == "V2"

    def test_rollback_target_version_no_content(self, db_instance):
        media_id, media_uuid, _ = db_instance.add_media_with_keywords(title="RollbackNoContent", content="V1", media_type="doc")
        # Manually insert a version with NULL content to simulate the edge case
        v2_uuid = db_instance._generate_uuid()
        with db_instance.transaction() as conn:
             conn.execute(
                 """INSERT INTO DocumentVersions (media_id, version_number, content, uuid, last_modified, version, client_id, deleted)
                VALUES (?, ?, ?, ?, ?, 1, ?, 0)""",
             (media_id, 2, "", v2_uuid, db_instance._get_current_utc_timestamp_str(), db_instance.client_id)
             )

        result = db_instance.rollback_to_version(media_id=media_id, target_version_number=2)

        assert 'error' in result
        assert 'success' not in result
        # Check the exact error message based on your implementation
        assert 'Cannot rollback to the current latest version number.' in result['error'] or 'target_content is None' in result['error'] # Adjust based on actual message

        # Verify no changes occurred
        assert get_entity_version(db_instance, "Media", media_uuid) == 1
        media_content = db_instance.get_media_by_id(media_id)['content']
        assert media_content == "V1"
        latest_version = get_document_version(db_instance, media_id=media_id, version_number=1)
        assert latest_version is not None

#
# End of test_media_processing.py
#######################################################################################################################
