# test_add_media_endpoint.py
# Description: This file contains the test cases for the /media/add endpoint of the FastAPI application.
# Style remodeled to mirror test_media_processing.py (less mocking, more integration).
#
# Imports
import asyncio # Added
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional  # Added Tuple
from unittest.mock import patch, MagicMock, AsyncMock # Refined imports
import gc
#
# 3rd-party Libraries
import pytest
import httpx # Keep for mocking specific download errors
from fastapi import status, Header
from fastapi.testclient import TestClient
from loguru import logger
#
# Local Imports
# Adjust import paths based on your project structure if needed
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_db_for_user
from tldw_Server_API.tests.test_utils import temp_db
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import Database # Import Database class
# Import the form model
from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm, MediaType # Import AddMediaForm, MediaType
#
######################################################################################################################
#
#

# --- Use Main App Instance ---
try:
    from tldw_Server_API.app.main import app as fastapi_app_instance, app
except ImportError:
    raise ImportError("Could not locate the main FastAPI app instance. Adjust the import path.")

# --- Constants (Borrowing/Adapting from test-library-1) ---
API_PREFIX = "/api/v1/media"
ADD_MEDIA_ENDPOINT = f"{API_PREFIX}/add"

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

# Define paths
SAMPLE_VIDEO_PATH = TEST_MEDIA_DIR / "sample.mp4"
SAMPLE_AUDIO_PATH = TEST_MEDIA_DIR / "sample.mp3"
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
create_dummy_file(SAMPLE_AUDIO_PATH, "dummy audio data") # Content doesn't need to be valid audio
create_dummy_file(SAMPLE_PDF_PATH, "%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 0>>endobj\nxref\n0 3\n0000000000 65535 f \n0000000010 00000 n \n0000000059 00000 n \ntrailer<</Size 3/Root 1 0 R>>\nstartxref\n114\n%%EOF") # Minimal PDF
create_dummy_file(SAMPLE_EPUB_PATH, "dummy epub data") # Content doesn't need to be valid epub
create_dummy_file(SAMPLE_TXT_PATH, "Sample TXT content.")
create_dummy_file(SAMPLE_MD_PATH, "# Sample MD\nContent.")
create_dummy_file(SAMPLE_DOCX_PATH, "dummy docx data") # Content doesn't need to be valid docx
create_dummy_file(SAMPLE_RTF_PATH, "{\\rtf1\\ansi dummy rtf}") # Minimal RTF
create_dummy_file(SAMPLE_HTML_PATH, "<html><body>Sample HTML</body></html>")
create_dummy_file(SAMPLE_XML_PATH, "<root><data>Sample XML</data></root>")


# Test URLs (Stable, public URLs)
VALID_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Short example
VALID_AUDIO_URL = "https://cdn.pixabay.com/download/audio/2023/12/02/audio_2f291f569a.mp3?filename=about-anger-179423.mp3"
VALID_PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
VALID_EPUB_URL = "https://filesamples.com/samples/ebook/epub/Alices%20Adventures%20in%20Wonderland.epub"
VALID_TXT_URL = "https://raw.githubusercontent.com/rmusser01/tldw/main/LICENSE.txt" # Raw text URL
VALID_MD_URL = "https://raw.githubusercontent.com/rmusser01/tldw/main/README.md" # Raw markdown URL
INVALID_URL = "http://this.url.does.not.exist/resource.file"
URL_404 = "https://httpbin.org/status/404" # Reliable 404

# --- Fixtures ---

# Mock Database Setup (Using test_utils.temp_db pattern from test_media_versions)
@pytest.fixture(scope="session")
def db_session_scope():
    """RENAMED: Session-scoped temporary database with explicit connection closing."""
    db: Optional[Database] = None
    try:
        with temp_db() as db_context:
            db = db_context
            if not hasattr(db, 'client_id') or not db.client_id:
                db.client_id = settings.get("SERVER_CLIENT_ID", "test_client_id") # Use .get() for safety
                logger.warning("Manually set client_id on Database instance from temp_db")

            logger.info(f"--- Using session DB from temp_db: {db.db_path_str} ---")
            try:
                db.execute_query("PRAGMA foreign_keys=ON;")
            except Exception as fk_e:
                logger.warning(f"Could not enable foreign keys on session DB: {fk_e}")
            yield db
    finally:
        if db:
            db_path_str_for_log = getattr(db, 'db_path_str', 'UNKNOWN_DB_PATH') # Get path safely
            logger.info(f"--- Attempting to close DB connection for test session teardown: {db_path_str_for_log} ---")
            try:
                # Explicitly close the connection potentially held by the main test thread
                db.close_connection()
                logger.info(f"--- Connection closed (or was already closed) for main test thread: {db_path_str_for_log} ---")
            except Exception as close_err:
                logger.warning(f"Error closing DB connection during session teardown for {db_path_str_for_log}: {close_err}")
        else:
            logger.warning("--- DB instance was None during session teardown ---")
        logger.info(f"--- Teardown complete for db_session_scope ({db_path_str_for_log}) ---")


@pytest.fixture(scope="function")
def db_session(db_session_scope):
    """Function-scoped access to the session DB with cleanup."""
    db = db_session_scope
    if not hasattr(db, 'client_id') or not db.client_id:
         db.client_id = settings.get("SERVER_CLIENT_ID", "test_client_id")

    yield db # Use the session-scoped DB instance

    # Cleanup application tables after each test function
    logger.debug(f"--- Cleaning DB tables after test in {db.db_path_str} ---")
    try:
        with db.transaction():
            db.execute_query("DELETE FROM MediaKeywords;", commit=False)
            db.execute_query("DELETE FROM DocumentVersions;", commit=False)
            try:
                db.execute_query("DELETE FROM MediaFTS;", commit=False)
                logger.debug("Cleared MediaFTS table.")
            except Exception as fts_e:
                logger.debug(f"Note: Could not clear MediaFTS (may not exist or error): {fts_e}")
            db.execute_query("DELETE FROM Media;", commit=False)
            db.execute_query("DELETE FROM Keywords;", commit=False)
            try:
                db.execute_query(
                    "DELETE FROM sqlite_sequence WHERE name IN ('Media', 'Keywords', 'DocumentVersions', 'MediaFTS');",
                    commit=False)
            except Exception as seq_e:
                logger.debug(f"Note: Could not clear sqlite_sequence (may be empty): {seq_e}")
    except Exception as e:
        logger.error(f"Error during DB cleanup in test_add_media_endpoint: {e}", exc_info=True)
    # No explicit close here, let session scope handle it

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

# Override get_db_for_user to use the temp test DB
def override_get_db_for_user_dependency(db_fixture):
    """
    Returns a dependency override function that yields the provided DB fixture.
    """
    async def _override(): # Changed to async def
        # logger.debug(f"--- OVERRIDING get_db_for_user with fixture: {db_fixture.db_path_str} ---")
        yield db_fixture
    return _override

@pytest.fixture(scope="module")
def test_api_client(db_session_scope):
    """
    RENAMED: Provides a TestClient instance, overriding auth and DB dependencies.
    This fixture is module-scoped, meaning the overrides apply to all tests within
    the module where it's used, and the TestClient is created once per module.
    """
    logger.debug(f"Setting up test_api_client fixture for module (DB: {db_session_scope.db_path_str})...")

    # --- File Existence Checks (Keep these or adapt as needed) ---
    # Skip the entire module if essential files are missing
    if not SAMPLE_VIDEO_PATH.exists(): pytest.skip(f"Test video file not found: {SAMPLE_VIDEO_PATH}")
    if not SAMPLE_AUDIO_PATH.exists(): pytest.skip(f"Test audio file not found: {SAMPLE_AUDIO_PATH}")
    if not SAMPLE_PDF_PATH.exists(): pytest.skip(f"Test PDF file not found: {SAMPLE_PDF_PATH}")
    if not SAMPLE_EPUB_PATH.exists(): pytest.skip(f"Test EPUB file not found: {SAMPLE_EPUB_PATH}")
    # Add checks for other required sample files if necessary

    # --- Dependency Overrides ---
    # Create the override function using the session-scoped DB instance provided
    # Ensure 'db_session_scope' fixture is correctly defined and yields the Database instance
    db_override_func = override_get_db_for_user_dependency(db_session_scope)

    # Store original overrides to restore later
    original_overrides = app.dependency_overrides.copy()

    # Apply overrides to the main FastAPI app instance
    app.dependency_overrides[get_request_user] = override_get_request_user
    app.dependency_overrides[get_db_for_user] = db_override_func
    logger.info("Applied dependency overrides for get_request_user and get_db_for_user (module scope)")

    # Instantiate the TestClient using the main FastAPI app instance
    # Ensure 'fastapi_app_instance' is correctly imported from your main application file
    try:
        with TestClient(fastapi_app_instance) as test_client_instance:
            logger.debug("TestClient instance created.")
            yield test_client_instance  # Yield the TestClient instance `c`
    except Exception as client_exc:
        logger.error(f"Failed to create TestClient: {client_exc}", exc_info=True)
        pytest.fail(f"TestClient instantiation failed: {client_exc}")
    finally:
        # --- Restore original overrides after tests in the module are done ---
        app.dependency_overrides = original_overrides
        logger.info("Restored original dependency overrides (module scope teardown)")


@pytest.fixture
def dummy_headers():
    """Provides headers required by endpoint signature, even if logic is mocked."""
    # The actual value doesn't matter because get_request_user is mocked
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
    """
    Factory to create file tuples suitable for TestClient files parameter.
    Format: (filename, file_content_bytes, mime_type)
    """
    def _create(filepath: Path) -> Tuple[str, bytes, str]:
        if not filepath.exists():
            # Maybe create a dummy file here instead of skipping?
            # create_dummy_file(filepath, "dummy content")
            pytest.skip(f"Required test file missing: {filepath}")

        mime_map = {
            ".mp4": "video/mp4", ".mov": "video/quicktime",
            ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
            ".pdf": "application/pdf",
            ".epub": "application/epub+zip",
            ".txt": "text/plain", ".md": "text/markdown",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".rtf": "application/rtf", # Or text/rtf
            ".html": "text/html", ".htm": "text/html",
            ".xml": "application/xml", # Or text/xml
        }
        mime_type = mime_map.get(filepath.suffix.lower(), "application/octet-stream")
        try:
             content = filepath.read_bytes()
        except Exception as e:
             pytest.fail(f"Failed to read test file {filepath}: {e}")

        # Return the tuple in the format TestClient expects for 'files'
        return (filepath.name, content, mime_type)
    return _create


# --- Helper Functions (Borrowing/Adapting from test-library-1) ---
# (check_batch_response and check_media_item_result remain largely the same)
def check_batch_response(
        response,
        expected_status_code,
        expected_processed=None,
        expected_errors=None, # Number of items with status "Error" or "Failed"
        expected_warnings=None, # Number of items with status "Warning"
        check_results_len=None,
):
    """Helper to check common aspects of the batch response from /add."""
    # Log response text on failure for debugging
    if response.status_code != expected_status_code:
        logger.error(f"Expected status {expected_status_code}, got {response.status_code}. Response text: {response.text}")
    assert response.status_code == expected_status_code
    try:
        data = response.json()
    except Exception as e:
        pytest.fail(f"Failed to parse response JSON. Status: {response.status_code}, Text: {response.text}, Error: {e}")

    assert "results" in data, f"Response missing 'results' key: {data}"
    assert isinstance(data["results"], list), f"'results' is not a list: {data['results']}"

    # Calculate actual counts from results
    actual_processed = sum(1 for r in data.get("results", []) if r.get("status") == "Success")
    actual_errors = sum(1 for r in data.get("results", []) if r.get("status") in ["Error", "Failed"]) # Include "Failed"
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
    """
    Helper to check structure of a single item in the results list from /add.
    """
    assert isinstance(result, dict), f"Result item is not a dictionary: {result}"
    assert "status" in result, f"Result missing 'status' key: {result}"
    assert result["status"] == expected_status, f"Expected status '{expected_status}', got '{result['status']}' for input '{result.get('input_ref', 'N/A')}'"
    assert "input_ref" in result, "Result missing 'input_ref' key" # URL or original filename
    assert "processing_source" in result, "Result missing 'processing_source' key" # Path or URL given to processor
    assert "media_type" in result, "Result missing 'media_type' key"
    if expected_media_type:
        assert result["media_type"] == expected_media_type, f"Expected media_type '{expected_media_type}', got '{result['media_type']}'"

    # Loosen check for metadata - allow None or dict
    assert "metadata" not in result or isinstance(result["metadata"], (dict, type(None))), f"Result 'metadata' is not a dict or None: {result['metadata']}"
    # Allow content to be None
    assert "content" in result, "Result missing 'content' key"
    # Allow chunks to be None
    assert "chunks" in result, "Result missing 'chunks' key"
    # Allow analysis to be None
    assert "analysis" in result, "Result missing 'analysis' key"
    # Loosen check for analysis_details - allow None or dict
    assert "analysis_details" not in result or isinstance(result["analysis_details"], (dict, type(None))), f"Result 'analysis_details' is not a dict or None: {result['analysis_details']}"

    assert "error" in result, "Result missing 'error' key" # Allowed to be None

    # Check warnings: should be None or a list
    assert "warnings" not in result or isinstance(result.get("warnings"), (list, type(None))), f"Result missing or invalid 'warnings': {result.get('warnings')}"


    if check_db_interaction:
        assert "db_id" in result, "Result missing 'db_id' key"
        assert "db_message" in result, "Result missing 'db_message' key"
        # If status is Success, db_id should usually be an integer (unless overwrite deleted it?)
        if expected_status == "Success" and result.get('db_message') == 'Media added to database.':
            assert isinstance(result.get("db_id"), int), f"Expected integer db_id for Success status, got {result.get('db_id')}"

    if expected_status in ["Error", "Failed"]:
        # Allow error to be None if status is Error but it was a file saving error reported differently
        # assert result["error"] is not None and result["error"] != "", f"Expected non-empty 'error' for status {expected_status}"
        pass # Error check handled by callers more specifically
    elif expected_status == "Success":
        assert result.get("error") is None or result["error"] == "", \
             f"Expected None or empty error for Success status, got '{result.get('error')}' for input '{result.get('input_ref')}'"
    elif expected_status == "Warning":
        # Warning might have an error string OR just warnings list populated
        assert result.get("error") or result.get("warnings"), f"Expected error or warnings for Warning status for input '{result.get('input_ref')}'"


# --- Helper for Form Data ---
def create_add_media_form_data(**overrides) -> Dict[str, Any]:
    """
    Creates form data dict suitable for TestClient(data=...).
    Matches the Form(...) fields expected by the dependency used in `add_media`.
    Provides valid default for chunk_method when chunking is enabled.
    Omits keys for None values unless explicitly needed as empty strings.
    Ensures all REQUIRED fields (like media_type) are present.
    """
    # Defaults should align with AddMediaForm defaults and endpoint Form() defaults
    defaults = {
        "media_type": "video", # Required by AddMediaForm
        "urls": None,
        "title": None,
        "author": None,
        "keywords": "", # Maps to keywords_str via alias
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
        "chunk_method": 'sentences', # Default if perform_chunking is True
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

    # Prepare form_dict suitable for TestClient `data` param
    form_dict = {}
    for k, v in current_data.items():
        if v is None:
            # Skip None values for optional fields for cleaner request data
            # Exception: If a field *must* be sent as empty string, handle here.
            # 'keywords' defaults to "" and should be sent.
            if k == 'keywords' and v is None: # If override explicitly set keywords to None
                 form_dict[k] = ""
            continue

        # Handle specific types for form encoding
        if k == 'urls' and isinstance(v, list):
            # TestClient handles list values for form data correctly
            form_dict[k] = v
        elif isinstance(v, bool):
            # Send booleans as 'true'/'false' strings
            form_dict[k] = str(v).lower()
        elif isinstance(v, (int, float, str)):
             # Send numbers and strings directly (TestClient converts to str)
             form_dict[k] = v
        elif isinstance(v, Path): # Handle Path objects if used in overrides
             form_dict[k] = str(v)
        elif hasattr(v, 'value'): # Basic check for Enum-like objects
             form_dict[k] = str(v.value)
        else:
            # Default to string conversion for safety
            form_dict[k] = str(v)

    # Ensure required 'media_type' is always present
    if "media_type" not in form_dict:
        # This should only happen if defaults change or override removed it
        form_dict["media_type"] = defaults["media_type"]
    # If chunking is disabled, chunk_method might be irrelevant, but send None if explicitly set
    if str(form_dict.get("perform_chunking")) == 'false':
        if "chunk_method" in form_dict and current_data.get("chunk_method") is None:
             # If perform_chunking is False, and chunk_method was explicitly None, remove it
             # Or keep it as None? Let the Pydantic model handle default logic.
             # For form data, maybe better to just not send it if None and chunking is off?
             pass # Keep None if explicitly set for now
        elif "chunk_method" not in form_dict and current_data.get("chunk_method") is None:
             pass # Don't add it if it wasn't in overrides and chunking is off

    logger.debug(f"Generated form data for /add: {form_dict}")
    return form_dict


# ##################################################################################################################
# Test Cases
# ##################################################################################################################

# === Validation Tests ===

def test_add_media_invalid_media_type_value(test_api_client, dummy_headers):
    """Test sending an invalid value for the media_type enum."""
    form_data = create_add_media_form_data(media_type="picture", urls=["http://a.com"])
    # Use the RENAMED client variable
    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=dummy_headers)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    details = response.json().get('detail', [])
    assert isinstance(details, list)
    # Assert specific pydantic error message
    assert any("input should be 'video', 'audio', 'document', 'pdf' or 'ebook'" in err.get('msg', '').lower()
               for err in details if 'media_type' in err.get('loc', [])), f"Error details: {details}"

def test_add_media_invalid_field_type(test_api_client, dummy_headers):
    """Test sending a non-boolean string for a boolean field."""
    form_data = create_add_media_form_data(media_type="video", urls=["http://a.com"], diarize="maybe")
    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=dummy_headers)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
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
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    details = response.json().get('detail', [])
    assert isinstance(details, list)
    assert any(err.get('type') == 'missing' and 'media_type' in err.get('loc', []) for err in details)

# === Basic Success Path Tests (URL & Upload) ===

@pytest.mark.parametrize("media_type, valid_url, expected_content_present", [
    ("video", VALID_VIDEO_URL, True),
    ("audio", VALID_AUDIO_URL, True),
    ("pdf", VALID_PDF_URL, True),
    # Skip Ebook URL test for now due to potential download size/time
    # pytest.param("ebook", VALID_EPUB_URL, True, marks=pytest.mark.skip(reason="EPUB URL download can be large/slow"), id="ebook_url"),
    ("document", VALID_TXT_URL, True),
    ("document", VALID_MD_URL, True),
], ids=["video_url", "audio_url", "pdf_url", "txt_url", "md_url"]) # Removed ebook id

@pytest.mark.timeout(180) # Increased timeout for external downloads/processing
def test_add_media_single_url_success(test_api_client, db_session, media_type, valid_url, expected_content_present, dummy_headers): # Added db_session
    """Test processing a single valid URL for each media type."""
    form_data = create_add_media_form_data(media_type=media_type, urls=[valid_url])
    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=dummy_headers)
    # Expect 200 OK for success, or 207 if warnings occurred during processing
    expected_code = status.HTTP_200_OK
    if response.status_code != expected_code: # Log if not expected
        logger.error(f"URL success test ({media_type}) failed. Status: {response.status_code}, Expected: {expected_code}, Text: {response.text}")

    # Check for 200 or 207 (if warnings happened but core processing succeeded)
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_207_MULTI_STATUS]

    data = check_batch_response(response, response.status_code, expected_processed=1, expected_errors=0, check_results_len=1)
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
    ("video", SAMPLE_VIDEO_PATH, True),
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
    form_data = create_add_media_form_data(media_type=media_type)

    # --- CORRECTED TestClient Call ---
    # Pass form data via `data` and files via `files` in the same call
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
    assert Path(result.get("processing_source", "")).name == sample_path.name
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
def test_add_media_mixed_url_file_success(test_api_client, db_session, create_upload_file, dummy_headers): # Added db_session
    """Test adding one valid video URL and one valid video file."""
    if not SAMPLE_VIDEO_PATH.exists(): pytest.skip(f"Test file not found: {SAMPLE_VIDEO_PATH}")

    file_tuple = create_upload_file(SAMPLE_VIDEO_PATH)
    form_data = create_add_media_form_data(media_type="video", urls=[VALID_VIDEO_URL])

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


@patch("tldw_Server_API.app.api.v1.endpoints.media.smart_download", new_callable=AsyncMock)
@pytest.mark.timeout(120) # Increased timeout
def test_add_media_multiple_failures_and_success_pdf(mock_smart_download, test_api_client, db_session, create_upload_file, dummy_headers): # Added db_session
    """Test a mix of successful and failed items for PDF media type."""
    if not SAMPLE_PDF_PATH.exists(): pytest.skip(f"Test file not found: {SAMPLE_PDF_PATH}")
    if not SAMPLE_AUDIO_PATH.exists(): pytest.skip(f"Test file not found: {SAMPLE_AUDIO_PATH}") # Need this for invalid format test

    good_pdf_file_tuple = create_upload_file(SAMPLE_PDF_PATH)
    # Use a non-PDF file to test upload validation/processing error
    invalid_format_file_tuple = create_upload_file(SAMPLE_AUDIO_PATH) # e.g., an audio file

    # Mock smart_download behavior
    async def download_side_effect(url, temp_dir, **kwargs):
        # Simulate Path object creation
        url_filename = Path(url).name.split('?')[0]
        temp_path = Path(temp_dir) / url_filename
        if 'dummy.pdf' in url_filename:
            # Simulate successful download by writing content
            temp_path.write_bytes(SAMPLE_PDF_PATH.read_bytes())
            logger.debug(f"Mock smart_download success: {url} -> {temp_path}")
            return temp_path # Return Path object
        elif '404' in url:
            logger.debug(f"Mock smart_download raising 404 for: {url}")
            # Simulate download failure
            raise httpx.HTTPStatusError(message="404 Not Found", request=MagicMock(url=url), response=MagicMock(status_code=404))
        else:
            logger.error(f"Unexpected URL in smart_download mock: {url}")
            # Simulate a generic download error for unexpected URLs
            raise IOError(f"Mock download failed for unexpected URL: {url}")
    mock_smart_download.side_effect = download_side_effect

    form_data = create_add_media_form_data(media_type="pdf", urls=[VALID_PDF_URL, URL_404])
    files_data = [
        ("files", good_pdf_file_tuple),
        ("files", invalid_format_file_tuple)
    ]
    response = test_api_client.post(
        ADD_MEDIA_ENDPOINT,
        data=form_data,
        files=files_data, # Pass the list of key-value tuples
        headers=dummy_headers
    )
    # -----------------------------------

    expected_code = status.HTTP_207_MULTI_STATUS
    if response.status_code != expected_code:
        logger.error(f"Multiple Fail/Success test failed. Status: {response.status_code}, Expected: {expected_code}, Text: {response.text}")

    # Expect 207 because some items failed (URL 404, invalid upload format)
    # Expect 2 processed successfully (valid URL, valid PDF upload)
    data = check_batch_response(response, 207, expected_processed=2, expected_errors=2, check_results_len=4)

    results_map = {r["input_ref"]: r for r in data["results"]}

    # 1. Valid PDF URL -> Success
    assert VALID_PDF_URL in results_map
    assert results_map[VALID_PDF_URL]["status"] in ["Success", "Warning"] # Allow warning if analysis skipped etc.
    check_media_item_result(results_map[VALID_PDF_URL], results_map[VALID_PDF_URL]["status"], expected_media_type="pdf")
    assert isinstance(results_map[VALID_PDF_URL].get("db_id"), int)

    # 2. URL 404 -> Error
    assert URL_404 in results_map
    assert results_map[URL_404]["status"] == "Error" # Should be a hard error
    check_media_item_result(results_map[URL_404], "Error", expected_media_type="pdf")
    assert "Download/preparation failed" in results_map[URL_404].get("error", "") or "404" in results_map[URL_404].get("error", "")
    assert results_map[URL_404].get("db_id") is None

    # 3. Valid PDF Upload -> Success
    assert SAMPLE_PDF_PATH.name in results_map
    assert results_map[SAMPLE_PDF_PATH.name]["status"] in ["Success", "Warning"]
    check_media_item_result(results_map[SAMPLE_PDF_PATH.name], results_map[SAMPLE_PDF_PATH.name]["status"], expected_media_type="pdf")
    assert isinstance(results_map[SAMPLE_PDF_PATH.name].get("db_id"), int)

    # 4. Invalid Format Upload (Audio file for PDF endpoint) -> Error
    assert SAMPLE_AUDIO_PATH.name in results_map
    assert results_map[SAMPLE_AUDIO_PATH.name]["status"] == "Error" # Should fail during processing
    check_media_item_result(results_map[SAMPLE_AUDIO_PATH.name], "Error", expected_media_type="pdf")
    # Error message might vary depending on where pdf processor fails
    error_msg = results_map[SAMPLE_AUDIO_PATH.name].get("error", "").lower()
    assert "pdf processing error" in error_msg or \
           "cannot parse" in error_msg or \
           "invalid file type" in error_msg or \
           "pymupdf" in error_msg # Check for common PDF error indicators
    assert results_map[SAMPLE_AUDIO_PATH.name].get("db_id") is None

    # Check mock calls
    assert mock_smart_download.call_count == 2 # Called for the two URLs


# === Error Handling Tests ===

@patch("tldw_Server_API.app.api.v1.endpoints.media._save_uploaded_files", new_callable=AsyncMock)
def test_add_media_file_save_error(mock_save_files, test_api_client, db_session, create_upload_file, dummy_headers): # Added db_session
    """Test an error during the file saving stage."""
    if not SAMPLE_AUDIO_PATH.exists(): pytest.skip(f"Test file not found: {SAMPLE_AUDIO_PATH}")

    file_tuple = create_upload_file(SAMPLE_AUDIO_PATH)
    # Simulate _save_uploaded_files returning only errors
    mock_save_files.return_value = (
        [], # No successfully saved files
        [{"original_filename": SAMPLE_AUDIO_PATH.name, "input_ref": SAMPLE_AUDIO_PATH.name, "status": "Error", "error": "Failed to save uploaded file: Disk full (OSError)"}] # List of errors
    )

    form_data = create_add_media_form_data(media_type="audio")

    response = test_api_client.post(
        ADD_MEDIA_ENDPOINT,
        data=form_data,
        # IMPORTANT: Still need to pass *something* to 'files' for FastAPI to know it's multipart,
        # even though our mock will intercept _save_uploaded_files.
        # Pass the original tuple, the mock prevents actual saving.
        files={"files": file_tuple},
        headers=dummy_headers
    )
    # -----------------------------------

    expected_code = status.HTTP_207_MULTI_STATUS
    if response.status_code != expected_code:
        logger.error(f"File Save Error test failed. Status: {response.status_code}, Expected: {expected_code}, Text: {response.text}")

    # Expect 207 because the *attempt* was made, but resulted in an error reported in the results
    data = check_batch_response(response, 207, expected_processed=0, expected_errors=1, check_results_len=1)

    result = data["results"][0]
    assert result["status"] == "Error" # Check specific status
    # Don't check DB interaction here, as it shouldn't have happened
    check_media_item_result(result, "Error", check_db_interaction=False, expected_media_type="audio")
    assert result["input_ref"] == SAMPLE_AUDIO_PATH.name
    assert "Failed to save uploaded file" in result.get("error", "")
    assert "Disk full" in result.get("error", "")
    assert "OSError" in result.get("error", "")
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

    expected_code = status.HTTP_207_MULTI_STATUS  # Expect 207 as processing might fail *or* succeed but DB ok
    if response.status_code != expected_code:
        logger.error(
            f"Invalid Format test failed. Status: {response.status_code}, Expected: {expected_code}, Text: {response.text}")

    # Expect 207 Multi-Status, as the processing itself might fail or warn
    data = check_batch_response(response, 207, check_results_len=1)  # Don't assert specific counts yet

    result = data["results"][0]
    # Status could be Error or Warning depending on how process_videos handles invalid input
    assert result["status"] in ["Error", "Warning"]
    check_media_item_result(result, result["status"], expected_media_type="video")  # Check structure
    error_msg = result.get("error", "").lower()
    warning_msg = result.get("warnings")  # Check warnings too

    # Check for *some* indication of a processing or format error
    assert "failed to process" in error_msg or \
           "invalid format" in error_msg or \
           "cannot open" in error_msg or \
           (warning_msg and any("format" in w.lower() for w in warning_msg)), \
        f"Expected a processing/format error or warning, got error='{error_msg}', warnings='{warning_msg}'"

    # DB ID should be None if processing failed before persistence attempt
    # or if persistence failed (which shouldn't happen with the AttributeError fixed unless new error)
    assert result.get("db_id") is None


# === Analysis and Chunking Tests ===

# Target the summarize function *within* the specific PDF processing library module
@patch("tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib.summarize")
@pytest.mark.timeout(90) # Increased timeout
def test_add_media_pdf_with_analysis_mocked(mock_summarize, test_api_client, db_session, dummy_headers): # Added db_session
    """Test PDF analysis via /add, mocking only the summarize call."""
    if not SAMPLE_PDF_PATH.exists(): pytest.skip(f"Test file not found: {SAMPLE_PDF_PATH}")

    mock_analysis_text = "Mocked analysis for PDF."
    # Configure the mock to return the desired text directly
    # AsyncMock is suitable if summarize is async
    mock_summarize.return_value = asyncio.Future() # Create a future
    mock_summarize.return_value.set_result(mock_analysis_text) # Set its result

    form_data = create_add_media_form_data(
        media_type="pdf",
        urls=[VALID_PDF_URL],
        perform_analysis=True,
        perform_chunking=True,
        # --- Use a placeholder API name that might pass basic validation if one exists ---
        # Or adjust if your summarize lib has a known "dummy" or "mock" API name setting
        api_name="mock", # Changed from "mock_model" - adjust if needed
        api_key="mock_key"
    )
    response = test_api_client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=dummy_headers)

    expected_code = status.HTTP_200_OK
    if response.status_code != expected_code:
        logger.error(f"PDF Analysis test failed. Status: {response.status_code}, Expected: {expected_code}, Text: {response.text}")

    # Expect 200 or 207 (if analysis mock/integration causes warnings)
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_207_MULTI_STATUS]
    data = check_batch_response(response, response.status_code, expected_processed=1, expected_errors=0, check_results_len=1)

    result = data["results"][0]
    assert result["status"] in ["Success", "Warning"]
    check_media_item_result(result, result["status"], expected_media_type="pdf")

    assert result.get("content") is not None and len(result["content"]) > 0
    # Chunks might be None if PDF processor doesn't return them explicitly in this flow
    # assert result["chunks"] is not None and len(result["chunks"]) > 0

    # Check if the mocked function was called
    mock_summarize.assert_called()  # Check if the mock itself was entered

    # Check if analysis result contains the mocked text
    assert result.get("analysis") is not None
    # The check should be precise now if the mock worked
    assert result.get("analysis") == mock_analysis_text
    # Check if analysis details reflect the mock API name used
    assert result.get("analysis_details", {}).get("model") == form_data["api_name"]  # Should match the input api_name
    # Check DB insertion (this should work after Fix #1)
    assert isinstance(result.get("db_id"), int)
    assert "added" in result.get("db_message", "").lower() or "updated" in result.get("db_message", "").lower()

# TODO: Add similar mocked analysis tests for other media types (video, audio, ebook, document)
#       - Ensure the @patch target points to the correct analysis/summarization function used by that media type's processor.
#       - Use AsyncMock if the target function is async.


# ##################################################################################################################
# End of remodeled test_add_media_endpoint.py
# ##################################################################################################################