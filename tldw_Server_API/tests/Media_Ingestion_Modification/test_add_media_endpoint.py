# test_add_media_endpoint.py
# Description: This file contains the test cases for the /media/add endpoint of the FastAPI application.
# Style remodeled to mirror test_media_processing.py (less mocking, more integration).
#
# Imports
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock # Refined imports
#
# 3rd-party Libraries
import pytest
import httpx # Keep for mocking specific download errors
from fastapi import status, Header
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.tests.test_utils import temp_db

#
######################################################################################################################
#
#

# --- Use Main App Instance ---
try:
    from tldw_Server_API.app.main import app as fastapi_app_instance
except ImportError:
    raise ImportError("Could not locate the main FastAPI app instance. Adjust the import path.")

# --- Constants (Borrowing/Adapting from test-library-1) ---
API_PREFIX = "/api/v1/media"
ADD_MEDIA_ENDPOINT = f"{API_PREFIX}/add"

# Test Media Files (Ensure these paths are correct relative to your test file)
TEST_MEDIA_DIR = Path(__file__).parent / "test_media"
TEST_MEDIA_DIR.mkdir(exist_ok=True)
SAMPLE_VIDEO_PATH = TEST_MEDIA_DIR / "sample.mp4"
SAMPLE_AUDIO_PATH = TEST_MEDIA_DIR / "sample.mp3"
SAMPLE_PDF_PATH = TEST_MEDIA_DIR / "sample.pdf"
SAMPLE_EPUB_PATH = TEST_MEDIA_DIR / "sample.epub"
SAMPLE_TXT_PATH = TEST_MEDIA_DIR / "sample.txt"
SAMPLE_MD_PATH = TEST_MEDIA_DIR / "sample.md"
SAMPLE_DOCX_PATH = TEST_MEDIA_DIR / "sample.docx" # Requires real file
SAMPLE_RTF_PATH = TEST_MEDIA_DIR / "sample.rtf"   # Requires real file & pandoc
SAMPLE_HTML_PATH = TEST_MEDIA_DIR / "sample.html"
SAMPLE_XML_PATH = TEST_MEDIA_DIR / "sample.xml"

# Create dummy text files if missing
if not SAMPLE_TXT_PATH.exists(): SAMPLE_TXT_PATH.write_text("Sample TXT content.", encoding='utf-8')
if not SAMPLE_MD_PATH.exists(): SAMPLE_MD_PATH.write_text("# Sample MD\nContent.", encoding='utf-8')
if not SAMPLE_HTML_PATH.exists(): SAMPLE_HTML_PATH.write_text("<html><body>Sample HTML</body></html>", encoding='utf-8')
if not SAMPLE_XML_PATH.exists(): SAMPLE_XML_PATH.write_text("<root><data>Sample XML</data></root>", encoding='utf-8')

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
def db_instance_session():
    """Session-scoped temporary database."""
    try:
        with temp_db() as db:
            yield db
    finally:
        if db and hasattr(db, 'close_all_connections'):
            print(f"--- Closing ALL session DB connections for test_add_media_endpoint: {db.db_path_str} ---")
            db.close_all_connections()

@pytest.fixture(scope="function")
def db_session(db_instance_session):
     """Function-scoped access to the session DB with cleanup."""
     yield db_instance_session
     # Cleanup after test
     try:
        with db_instance_session.transaction():
            db_instance_session.execute_query("DELETE FROM MediaKeywords;", commit=False)
            db_instance_session.execute_query("DELETE FROM DocumentVersions;", commit=False)
            db_instance_session.execute_query("DELETE FROM Media;", commit=False)
            db_instance_session.execute_query("DELETE FROM Keywords;", commit=False)
        try:
            db_instance_session.execute_query("DELETE FROM sqlite_sequence WHERE name IN ('Media', 'Keywords', 'DocumentVersions');", commit=True)
        except Exception: pass
     except Exception as e:
         print(f"Error during DB cleanup in test_add_media_endpoint: {e}")

# Override verify_api_key for testing
# This dummy function bypasses the actual key check
async def override_verify_api_key(x_api_key: str = Header(..., alias="X-API-KEY")):
    """Dummy override for API key verification during tests."""
    logger.debug(f"Auth Override: Bypassing API key check for key starting with: {x_api_key[:4]}...")
    # You could return mock user/permissions here if needed downstream
    return {"user_id": "test_user", "permissions": ["*"]}

# Override get_db_for_user to use the temp test DB
def override_get_db_for_user_add_media(db_session):
    """Dependency override to provide the test DB session."""
    def _override():
        # print(f"--- OVERRIDING get_db_for_user for test_add_media with: {db_session.db_path_str} ---")
        yield db_session
    return _override

@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance for the main app."""
    # Check for essential binary files needed for core tests
    if not SAMPLE_VIDEO_PATH.exists(): pytest.skip(f"Test video file not found: {SAMPLE_VIDEO_PATH}")
    if not SAMPLE_AUDIO_PATH.exists(): pytest.skip(f"Test audio file not found: {SAMPLE_AUDIO_PATH}")
    if not SAMPLE_PDF_PATH.exists(): pytest.skip(f"Test PDF file not found: {SAMPLE_PDF_PATH}")
    if not SAMPLE_EPUB_PATH.exists(): pytest.skip(f"Test EPUB file not found: {SAMPLE_EPUB_PATH}")

    with TestClient(fastapi_app_instance) as c:
        yield c

# Removed mock_temp_dir fixture as it was only used by the removed mocking fixture

@pytest.fixture
def auth_headers():
    """Provides authentication/required headers."""
    return {
        "token": "test_api_token_123",
        #"X-API-KEY": "test_api_key_123"
    }

@pytest.fixture
def dummy_file_content():
    """Provides dummy byte content for mock files if needed elsewhere."""
    return b"dummy file content for testing"

@pytest.fixture
def create_upload_file(dummy_file_content):
    """Factory to create file tuples for TestClient files parameter."""
    def _create(filepath: Path):
        if not filepath.exists():
             pytest.skip(f"Required test file missing: {filepath}")
        mime_map = {
            ".mp4": "video/mp4", ".mov": "video/quicktime",
            ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
            ".pdf": "application/pdf",
            ".epub": "application/epub+zip",
            ".txt": "text/plain", ".md": "text/markdown",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".rtf": "application/rtf",
            ".html": "text/html", ".htm": "text/html",
            ".xml": "application/xml",
        }
        mime_type = mime_map.get(filepath.suffix.lower(), "application/octet-stream")
        return (filepath.name, filepath.read_bytes(), mime_type)
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
    """
    Helper to check structure of a single item in the results list from /add.
    """
    # ... (implementation remains the same as previous corrected version) ...
    assert isinstance(result, dict), f"Result item is not a dictionary: {result}"
    assert "status" in result, "Result missing 'status' key"
    assert result["status"] == expected_status, f"Expected status '{expected_status}', got '{result['status']}'"
    assert "input_ref" in result, "Result missing 'input_ref' key" # URL or original filename
    assert "processing_source" in result, "Result missing 'processing_source' key" # Path or URL given to processor
    assert "media_type" in result, "Result missing 'media_type' key"
    if expected_media_type:
        assert result["media_type"] == expected_media_type, f"Expected media_type '{expected_media_type}', got '{result['media_type']}'"
    assert "metadata" in result and isinstance(result["metadata"], dict), "Result missing or invalid 'metadata'"
    assert "content" in result, "Result missing 'content' key" # Allowed to be None on error
    assert "chunks" in result, "Result missing 'chunks' key" # Allowed to be None
    assert "analysis" in result, "Result missing 'analysis' key" # Allowed to be None
    assert "analysis_details" in result and isinstance(result["analysis_details"], dict), "Result missing or invalid 'analysis_details'"
    assert "error" in result, "Result missing 'error' key" # Allowed to be None
    # Ensure warnings is always a list, even if empty
    assert "warnings" in result and isinstance(result.get("warnings"), list), f"Result missing or invalid 'warnings': {result.get('warnings')}"

    if check_db_interaction:
        assert "db_id" in result, "Result missing 'db_id' key"
        assert "db_message" in result, "Result missing 'db_message' key"

    if expected_status in ["Error", "Failed"]:
        assert result["error"] is not None and result["error"] != "", f"Expected non-empty 'error' for status {expected_status}"
    elif expected_status == "Success":
        assert result["error"] is None or result["error"] == "", \
             f"Expected None or empty error for Success status, got '{result['error']}'"
    elif expected_status == "Warning":
        assert result["error"] or result.get("warnings"), "Expected error or warnings for Warning status"


# --- Helper for Form Data (REVISED) ---
def create_add_media_form_data(**overrides) -> Dict[str, Any]:
    """
    Creates form data dict suitable for TestClient(data=...).
    Ensure ALL required fields from the endpoint's Pydantic model
    are included here with valid defaults.
    """
    # *** Review and update these defaults meticulously ***
    defaults = {
        "media_type": "video", # Default assumption
        "urls": None,
        "keywords": "",
        "perform_analysis": False,
        "perform_chunking": False,
        "keep_original_file": True,
        # *** Added based on previous output ***
        "transcription_model": "tiny",
        # *** Assuming these are required/have defaults in the endpoint ***
        "transcription_language": "en",
        "diarize": False,
        "timestamp_option": True,
        "vad_use": False,
        "start_time": None,
        "end_time": None,
        "perform_confabulation_check_of_analysis": False,
        "pdf_parsing_engine": "pymupdf4llm",
        "chunk_method": None,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": None,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "custom_chapter_pattern": None,
        "title": None,
        "author": None,
        "custom_prompt": None,
        "system_prompt": None,
        "overwrite_existing": False,
        "api_name": None,
        "api_key": None,
        "use_cookies": False,
        "cookies": None,
        "perform_rolling_summarization": False,
        "summarize_recursively": False,
        # *** Add ANY other required Form fields here ***
        # Example: "some_other_required_field": "default_value",
    }

    # Update defaults with provided overrides
    current_data = defaults.copy()
    current_data.update(overrides)

    # Prepare for TestClient: filter None, convert types to string where needed
    form_dict = {}
    for k, v in current_data.items():
         if v is not None: # Skip None values entirely
            if isinstance(v, bool):
                form_dict[k] = str(v).lower() # 'true' / 'false'
            elif isinstance(v, (int, float)):
                 form_dict[k] = str(v)
            elif isinstance(v, list) and k == 'urls':
                 # Only include urls if the list is not empty
                 if v:
                     form_dict[k] = v
            elif isinstance(v, str):
                # Include empty strings only for specific fields if necessary
                # Generally, optional string fields can be omitted if empty
                if v or k in ["keywords", "cookies", "title", "author", "custom_prompt", "system_prompt", "chunk_language", "chunk_method", "api_name", "api_key", "custom_chapter_pattern", "pdf_parsing_engine", "transcription_model", "transcription_language"]:
                    form_dict[k] = v
            else:
                # Handle other types if necessary, otherwise convert to string
                form_dict[k] = str(v)


    # Special case: remove media_type if explicitly passed as None for testing
    if overrides.get("media_type") is None and "media_type" in form_dict:
         del form_dict["media_type"]

    # If 'urls' ended up empty after filtering, remove it
    if k == 'urls' and not form_dict.get('urls'):
        if 'urls' in form_dict:
            del form_dict['urls']

    logger.debug(f"Generated form data: {form_dict}")
    return form_dict


# ##################################################################################################################
# Test Cases
# ##################################################################################################################

# === Validation Tests ===

def test_add_media_invalid_media_type_value(client, auth_headers):
    """Test sending an invalid value for the media_type enum."""
    form_data = create_add_media_form_data(media_type="picture", urls=["http://a.com"])
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=auth_headers)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "Input should be 'video', 'audio', 'document', 'pdf' or 'ebook'" in str(response.json()['detail'])

def test_add_media_invalid_field_type(client, auth_headers):
    """Test sending a non-boolean string for a boolean field."""
    # Provide a valid media_type otherwise this might fail first
    form_data = create_add_media_form_data(media_type="video", urls=["http://a.com"], diarize="maybe")
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=auth_headers)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "'diarize'" in str(response.json()['detail'])
    assert "Input should be a valid boolean" in str(response.json()['detail'])

def test_add_media_missing_url_and_file(client, auth_headers):
    """Test calling the endpoint with neither URLs nor files provided."""
    # Need to provide media_type as it's required before source check often
    form_data = create_add_media_form_data(media_type="video", urls=None)
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=auth_headers)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    # Check that *specific* validation errors related to sources occurred
    details = response.json()['detail']
    # This assertion depends heavily on how your endpoint implements the check
    # Option 1: Pydantic validator on the model
    assert any("least one url or file must be provided" in err.get('msg','').lower() for err in details), f"Expected source validation error, got: {details}"
    # Option 2: Manual check raising HTTPException (less likely for 422)
    # assert "At least one 'url' or 'file' must be provided" in details # If endpoint raises 400 directly

def test_add_media_missing_required_form_field(client, auth_headers):
    """Test calling without a required field like media_type."""
    form_data = create_add_media_form_data(media_type=None, urls=["http://a.com"])
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=auth_headers)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert any(err.get('loc') and 'media_type' in err['loc'] and err.get('type') == 'missing' for err in response.json()['detail'])

# === Basic Success Path Tests (URL & Upload) ===

@pytest.mark.parametrize("media_type, valid_url, expected_content_present", [
    ("video", VALID_VIDEO_URL, True),
    ("audio", VALID_AUDIO_URL, True),
    ("pdf", VALID_PDF_URL, True),
    ("ebook", VALID_EPUB_URL, True),
    ("document", VALID_TXT_URL, True),
    ("document", VALID_MD_URL, True),
], ids=["video_url", "audio_url", "pdf_url", "ebook_url", "txt_url", "md_url"])
@pytest.mark.timeout(120)
def test_add_media_single_url_success(client, auth_headers, media_type, valid_url, expected_content_present):
    """Test processing a single valid URL for each media type."""
    form_data = create_add_media_form_data(media_type=media_type, urls=[valid_url])
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=auth_headers)
    data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
    result = data["results"][0]
    check_media_item_result(result, "Success", expected_media_type=media_type)
    assert result["input_ref"] == valid_url
    if expected_content_present:
        assert result["content"] is not None and len(result["content"]) > 0, f"Content missing for {media_type} URL"


@pytest.mark.parametrize("media_type, sample_path, expected_content_present", [
    ("video", SAMPLE_VIDEO_PATH, True),
    ("audio", SAMPLE_AUDIO_PATH, True),
    ("pdf", SAMPLE_PDF_PATH, True),
    ("ebook", SAMPLE_EPUB_PATH, True),
    ("document", SAMPLE_TXT_PATH, True),
    ("document", SAMPLE_MD_PATH, True),
    pytest.param("document", SAMPLE_DOCX_PATH, True, marks=pytest.mark.skipif(not SAMPLE_DOCX_PATH.exists(), reason="sample.docx not found"), id="docx_upload"),
    pytest.param("document", SAMPLE_RTF_PATH, True, marks=[pytest.mark.skipif(not SAMPLE_RTF_PATH.exists(), reason="sample.rtf not found"), pytest.mark.xfail(reason="Requires pandoc binary installed")], id="rtf_upload"),
    ("document", SAMPLE_HTML_PATH, True),
    ("document", SAMPLE_XML_PATH, True),
], ids=["video_upload", "audio_upload", "pdf_upload", "epub_upload", "txt_upload", "md_upload", "docx_upload", "rtf_upload", "html_upload", "xml_upload"])
@pytest.mark.timeout(60)
def test_add_media_single_file_upload_success(client, auth_headers, create_upload_file, media_type, sample_path, expected_content_present):
    """Test processing a single valid file upload for various types."""
    if not sample_path.exists(): pytest.skip(f"Test file not found: {sample_path}")
    file_tuple = create_upload_file(sample_path)
    form_data = create_add_media_form_data(media_type=media_type)
    files_param = {"files": [file_tuple]}
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, files=files_param, headers=auth_headers)
    # Getting 400 Bad Request here suggests an issue potentially *before* form validation,
    # perhaps in how the files list itself is handled, or a required field missing *only* for file uploads.
    # Print response text to see if there's a specific error message.
    if response.status_code != 200:
        logger.error(f"File upload test failed for {sample_path.name}. Status: {response.status_code}, Text: {response.text}")
    data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
    result = data["results"][0]
    check_media_item_result(result, "Success", expected_media_type=media_type)
    assert result["input_ref"] == sample_path.name
    assert Path(result["processing_source"]).name == sample_path.name
    if expected_content_present:
        assert result["content"] is not None and len(result["content"]) > 0, f"Content missing for {media_type} upload {sample_path.name}"
    if media_type == "document" and "source_format" in result.get("metadata", {}):
        assert result["metadata"]["source_format"] == sample_path.suffix.lower().strip('.')


# === Mixed Success/Failure Tests ===

@pytest.mark.timeout(120)
def test_add_media_mixed_url_file_success(client, auth_headers, create_upload_file):
    """Test adding one valid video URL and one valid video file."""
    file_tuple = create_upload_file(SAMPLE_VIDEO_PATH)
    form_data = create_add_media_form_data(media_type="video", urls=[VALID_VIDEO_URL])
    files_param = {"files": [file_tuple]}
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, files=files_param, headers=auth_headers)
    if response.status_code != 200: # Add logging
        logger.error(f"Mixed URL/File test failed. Status: {response.status_code}, Text: {response.text}")
    data = check_batch_response(response, 200, expected_processed=2, expected_errors=0, check_results_len=2)
    url_result = next((r for r in data["results"] if r["input_ref"] == VALID_VIDEO_URL), None)
    file_result = next((r for r in data["results"] if r["input_ref"] == SAMPLE_VIDEO_PATH.name), None)
    assert url_result is not None
    assert file_result is not None
    check_media_item_result(url_result, "Success", expected_media_type="video")
    check_media_item_result(file_result, "Success", expected_media_type="video")


@patch("tldw_Server_API.app.api.v1.endpoints.media.smart_download", new_callable=AsyncMock)
@pytest.mark.timeout(60)
def test_add_media_multiple_failures_and_success_pdf(mock_smart_download, client, auth_headers, create_upload_file):
    """Test a mix of successful and failed items for PDF media type."""
    # ... (mock setup remains the same) ...
    good_pdf_file_tuple = create_upload_file(SAMPLE_PDF_PATH)
    invalid_format_file_tuple = create_upload_file(SAMPLE_AUDIO_PATH)

    async def download_side_effect(url, temp_dir, **kwargs):
        filename = Path(url).name.split('?')[0] # Handle query params in URL filename
        if 'dummy.pdf' in filename:
            temp_path = Path(temp_dir) / filename
            temp_path.write_bytes(SAMPLE_PDF_PATH.read_bytes())
            logger.debug(f"Mock download success: {url} -> {temp_path}")
            return temp_path
        elif '404' in url:
            logger.debug(f"Mock download raising 404 for: {url}")
            raise httpx.HTTPStatusError(message="404 Not Found", request=MagicMock(url=url), response=MagicMock(status_code=404))
        else:
            pytest.fail(f"Unexpected URL in smart_download mock: {url}")
    mock_smart_download.side_effect = download_side_effect

    form_data = create_add_media_form_data(media_type="pdf", urls=[VALID_PDF_URL, URL_404])
    files_param = {"files": [good_pdf_file_tuple, invalid_format_file_tuple]}

    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, files=files_param, headers=auth_headers)
    # Expect 207 because some items failed
    # If still getting 400, log it.
    if response.status_code != 207:
        logger.error(f"Multiple Fail/Success test failed. Status: {response.status_code}, Text: {response.text}")
    data = check_batch_response(response, 207, expected_processed=2, expected_errors=2, check_results_len=4)

    # ... (assertions remain the same) ...
    results_map = {}
    for r in data["results"]:
        results_map[r["input_ref"]] = r

    assert results_map[VALID_PDF_URL]["status"] == "Success"
    check_media_item_result(results_map[VALID_PDF_URL], "Success", expected_media_type="pdf")
    assert results_map[URL_404]["status"] in ["Error", "Failed"]
    check_media_item_result(results_map[URL_404], results_map[URL_404]["status"], expected_media_type="pdf")
    assert "Download failed" in results_map[URL_404]["error"] or "404" in results_map[URL_404]["error"]
    assert results_map[SAMPLE_PDF_PATH.name]["status"] == "Success"
    check_media_item_result(results_map[SAMPLE_PDF_PATH.name], "Success", expected_media_type="pdf")
    assert results_map[SAMPLE_AUDIO_PATH.name]["status"] in ["Error", "Failed"]
    check_media_item_result(results_map[SAMPLE_AUDIO_PATH.name], results_map[SAMPLE_AUDIO_PATH.name]["status"], expected_media_type="pdf")
    assert "Invalid file format" in results_map[SAMPLE_AUDIO_PATH.name]["error"] or \
           "PDF processing error" in results_map[SAMPLE_AUDIO_PATH.name]["error"] or \
           "cannot parse" in results_map[SAMPLE_AUDIO_PATH.name]["error"].lower()
    assert mock_smart_download.call_count == 2


# === Error Handling Tests ===

@patch("tldw_Server_API.app.api.v1.endpoints.media._save_uploaded_files", new_callable=AsyncMock)
def test_add_media_file_save_error(mock_save_files, client, auth_headers, create_upload_file):
    """Test an error during the file saving stage."""
    # ... (mock setup remains the same) ...
    file_tuple = create_upload_file(SAMPLE_AUDIO_PATH)
    mock_save_files.return_value = ([], [{"input_ref": SAMPLE_AUDIO_PATH.name, "error": "Failed to save uploaded file: Disk full (OSError)"}])

    form_data = create_add_media_form_data(media_type="audio")
    files_param = {"files": [file_tuple]}
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, files=files_param, headers=auth_headers)
    # If still getting 400, log it
    if response.status_code != 207:
        logger.error(f"File Save Error test failed. Status: {response.status_code}, Text: {response.text}")
    data = check_batch_response(response, 207, expected_processed=0, expected_errors=1, check_results_len=1)
    # ... (assertions remain the same) ...
    result = data["results"][0]
    assert result["status"] in ["Error", "Failed"]
    check_media_item_result(result, result["status"])
    assert result["input_ref"] == SAMPLE_AUDIO_PATH.name
    assert "Failed to save uploaded file" in result["error"]
    assert "Disk full" in result["error"]
    assert "OSError" in result["error"]

@patch('tempfile.TemporaryDirectory')
def test_add_media_temp_dir_creation_error(mock_temp_dir_class, client, auth_headers, create_upload_file):
    """Test failure during temporary directory creation."""
    # ... (mock setup remains the same) ...
    mock_temp_dir_instance = MagicMock()
    mock_temp_dir_instance.__enter__.side_effect = OSError("Permission denied creating temp dir")
    mock_temp_dir_class.return_value = mock_temp_dir_instance

    file_tuple = create_upload_file(SAMPLE_AUDIO_PATH)
    form_data = create_add_media_form_data(media_type="audio")
    files_param = {"files": [file_tuple]}
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, files=files_param, headers=auth_headers)
    # If still getting 400, log it
    if response.status_code != 500:
        logger.error(f"Temp Dir Error test failed. Status: {response.status_code}, Text: {response.text}")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to create temporary directory" in response.json()["detail"]
    assert "Permission denied" in response.json()["detail"]

@pytest.mark.timeout(60)
def test_add_media_processor_handles_invalid_format(client, auth_headers, create_upload_file):
    """Test feeding an audio file to the video processor via /add."""
    file_tuple = create_upload_file(SAMPLE_AUDIO_PATH)
    form_data = create_add_media_form_data(media_type="video") # Mismatched type
    files_param = {"files": [file_tuple]}
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, files=files_param, headers=auth_headers)
    # If still getting 400, log it
    if response.status_code != 207:
        logger.error(f"Invalid Format test failed. Status: {response.status_code}, Text: {response.text}")
    data = check_batch_response(response, 207, expected_processed=0, expected_errors=1, check_results_len=1)
    # ... (assertions remain the same) ...
    result = data["results"][0]
    assert result["status"] in ["Error", "Failed"]
    check_media_item_result(result, result["status"], expected_media_type="video")
    assert "Video processing failed" in result["error"] or \
           "FFmpeg" in result["error"] or \
           "Invalid data found when processing input" in result["error"]

# === Analysis and Chunking Tests ===

@patch("tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib.summarize")
@pytest.mark.timeout(60)
def test_add_media_pdf_with_analysis_mocked(mock_summarize, client, auth_headers):
    """Test PDF analysis via /add, mocking only the summarize call."""
    # ... (mock setup remains the same) ...
    mock_analysis_text = "Mocked analysis for PDF."
    mock_summarize.return_value = mock_analysis_text

    form_data = create_add_media_form_data(
        media_type="pdf",
        urls=[VALID_PDF_URL],
        perform_analysis=True,
        perform_chunking=True,
        api_name="mock_model",
        api_key="mock_key"
    )
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers=auth_headers)
    # If still getting 422, log it
    if response.status_code != 200:
        logger.error(f"PDF Analysis test failed. Status: {response.status_code}, Text: {response.text}")
    data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
    # ... (assertions remain the same) ...
    result = data["results"][0]
    check_media_item_result(result, "Success", expected_media_type="pdf")
    assert result["content"] is not None and len(result["content"]) > 0
    assert result["chunks"] is not None and len(result["chunks"]) > 0
    mock_summarize.assert_called()
    assert result["analysis"] is not None
    assert mock_analysis_text in result["analysis"]
    assert result["analysis_details"].get("model") == "mock_model"


# Add more tests for other media types with analysis mocked similarly,
# ensuring the @patch target points to the correct analysis function for each type.

# Example for video (assuming analysis calls a 'summarize_transcript' function)
# @patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.summarize_transcript")
# @pytest.mark.timeout(120)
# def test_add_media_video_with_analysis_mocked(mock_summarize_transcript, client, auth_headers):
#     # ... test implementation similar to PDF one ...
#     pass


# ##################################################################################################################
# End of remodeled test_add_media_endpoint.py
# ##################################################################################################################