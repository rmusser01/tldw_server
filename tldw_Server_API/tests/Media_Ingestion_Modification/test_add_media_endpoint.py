# test_add_media_endpoint.py
# # Description: This file contains the test cases for the /media/add endpoint of the FastAPI application.
#
# Imports
import shutil
import tempfile
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any, Optional
#
# 3rd-party Libraries
import pytest
import requests.exceptions
from unittest.mock import patch, MagicMock, mock_open, ANY, AsyncMock
import requests.exceptions
from fastapi import FastAPI, BackgroundTasks, UploadFile, HTTPException, Header, File, status
from fastapi.testclient import TestClient

# --- Use Main App Instance ---
try:
    # Assume main app is in tldw_Server_API.app.main
    from tldw_Server_API.app.main import app as fastapi_app_instance
    # Assume the router is added in main.py or similar
    # If not, you might need to include it here if testing in complete isolation
    # from tldw_Server_API.app.api.v1.endpoints.media import router as media_router
    # fastapi_app_instance.include_router(media_router, prefix="/api/v1/media")
except ImportError:
    raise ImportError("Could not locate the main FastAPI app instance. Adjust the import path.")

# Local Imports (ensure these paths are correct relative to your test execution)
from tldw_Server_API.app.api.v1.endpoints.media import add_media # Endpoint function itself (optional if only testing via client)
# Import dependencies if needed for direct assertion/mocking, adjust paths as necessary
# from tldw_Server_API.app.core.config import settings
# from tldw_Server_API.app.core.media_processing import process_videos, process_audio_files, etc.
# from tldw_Server_API.app.core.db_management import add_media_to_database

# --- Constants ---
API_PREFIX = "/api/v1/media"
ADD_MEDIA_ENDPOINT = f"{API_PREFIX}/add"

# --- Mock Objects / Fixtures ---

@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance for the main app."""
    # Add any necessary setup for the main app if required for these tests
    # e.g., overriding specific dependencies for the whole module
    with TestClient(fastapi_app_instance) as c:
        yield c

@pytest.fixture
def dummy_file_content():
    """Provides dummy byte content for mock files."""
    return b"dummy file content for testing"

@pytest.fixture
def create_upload_file(dummy_file_content):
    """Factory to create mock UploadFile objects."""
    def _create(filename="test_upload.mp4", content_type="video/mp4", content=None):
        file_content = content if content is not None else dummy_file_content
        file_bytes = BytesIO(file_content)
        # Note: TestClient handles creating the UploadFile structure from the files= parameter.
        # This fixture is more useful if you need an UploadFile object directly.
        # For TestClient, you typically pass a tuple: (filename, file_like_object, content_type)
        # Let's return the tuple format for direct use with TestClient files parameter
        return (filename, file_bytes, content_type)
    return _create

@pytest.fixture
def mock_temp_dir(mocker):
    """Creates a real temporary directory for file operations in tests."""
    temp_dir = tempfile.mkdtemp()
    # print(f"Created temp dir for test: {temp_dir}") # Debugging
    yield Path(temp_dir)
    # print(f"Removing temp dir: {temp_dir}") # Debugging
    shutil.rmtree(temp_dir, ignore_errors=True)


# --- Unified Mocking Fixture ---
@pytest.fixture(autouse=True) # Applied to all tests in this file
def mock_endpoint_dependencies(mocker, mock_temp_dir):
    """
    Mocks external dependencies called *by* the add_media endpoint logic.
    Focuses on mocking boundaries: network, file system (partially), core processors, DB.
    """
    # --- File System Mocks (Using Real Temp Dir where possible) ---
    # Mock mkdtemp to control the *parent* of where TempDirManager might try to create
    # Or, more simply, mock TempDirManager's behavior if it's complex.
    # Let's mock the specific functions used for saving/cleanup.
    fake_media_temp_path = mock_temp_dir / "media_processing_temp"
    fake_media_temp_path.mkdir(exist_ok=True)

    # Mock mkdtemp used *within* the endpoint/helpers (if any, besides file saving)
    # This might be needed if the endpoint itself creates other temp dirs.
    # If file saving uses mock_temp_dir, this might not be strictly needed.
    mocker.patch('tempfile.mkdtemp', return_value=str(fake_media_temp_path))

    # Mock shutil.rmtree used for cleanup
    mock_rmtree = mocker.patch('shutil.rmtree')

    # Mock open for saving uploads/downloads within the *specific* fake temp path
    # We use the real 'open' but control the directory via mock_temp_dir
    # This makes file saving more realistic. Only mock open if absolutely necessary
    # to simulate errors.
    mock_open_instance = mocker.patch('builtins.open', mock_open()) # Keep if needed for specific error tests

    # Mock Path methods only if essential and causing issues with real temp dir
    # mocker.patch.object(Path, 'unlink')
    # mocker.patch.object(Path, 'exists')

    # --- Network Mocks ---
    mock_requests_get = mocker.patch('requests.get')
    # Default mock response (can be customized per test)
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.content = b'downloaded content'
    mock_response.raise_for_status = MagicMock()
    mock_response.url = "http://mockedurl.com/default"
    mock_requests_get.return_value = mock_response

    # --- Mock CORE Processing Functions ---
    # Target the functions *as imported/used* in the media endpoint module
    # Adjust 'tldw_Server_API.app.api.v1.endpoints.media.function_name' if the import location differs
    base_path = 'tldw_Server_API.app.api.v1.endpoints.media.'

    # Example realistic return structure (adjust based on your actual functions)
    # Ensure keys like 'status', 'db_id', 'input_ref', 'error', 'content', etc. match
    # what the endpoint logic expects from these core functions.
    def create_success_result(input_ref, db_id="mock_db_id", content="Mock content", media_type="unknown"):
        return {
            "status": "Success",
            "input_ref": input_ref,
            "processing_source": "url" if "http" in str(input_ref) else "upload",
            "media_type": media_type,
            "metadata": {"title": "Mock Title"},
            "content": content,
            "chunks": [{"text": content, "metadata": {}}] if content else None,
            "analysis": "Mock analysis" if content else None,
            "analysis_details": {"model": "mock_model"},
            "error": None,
            "warnings": [],
            "db_id": db_id,
            "db_message": "Added to DB" # Simulate DB add success within the processor
        }

    def create_error_result(input_ref, error_msg="Processing failed", media_type="unknown"):
         return {
            "status": "Error",
            "input_ref": input_ref,
            "processing_source": "url" if "http" in str(input_ref) else "upload",
            "media_type": media_type,
            "metadata": {},
            "content": None,
            "chunks": None,
            "analysis": None,
            "analysis_details": {},
            "error": error_msg,
            "warnings": [],
            "db_id": None,
            "db_message": None
        }

    # Mock Batch Processors (Video, Audio)
    mock_process_videos = mocker.patch(
        base_path + 'process_videos', # Sync batch processor
        return_value={
            "results": [create_success_result("mock_video_input_ref", "vid_batch_mock", media_type="video")],
            "processed_count": 1,
            "errors_count": 0,
            "errors": []
        }
    )
    mock_process_audio_files = mocker.patch(
        base_path + 'process_audio_files', # Sync batch processor
        return_value={
            "results": [create_success_result("mock_audio_input_ref", "aud_batch_mock", media_type="audio")],
            "processed_count": 1,
            "errors_count": 0,
            "errors": []
        }
    )
    # Mock Individual Processors (PDF, Document, Ebook)
    # Assume these are called individually per item within the endpoint loop
    mock_process_pdf_task = mocker.patch(
        base_path + 'process_pdf_task', # Async individual processor
        new_callable=AsyncMock,
        return_value=create_success_result("Mock_pdf_input_ref", "pdf_mock_abc", media_type="pdf")
    )
    # Note: import_plain_text_file / import_epub returning strings is unusual.
    # It's better if they return structured data like the others.
    # Assuming they now return a dict similar to create_success_result for consistency.
    mock_import_plain_text_file = mocker.patch(
        base_path + 'import_plain_text_file', # Sync individual processor
        return_value=create_success_result("mock_doc.txt", "doc_mock_789", media_type="document")
    )
    mock_import_epub = mocker.patch(
        base_path + 'import_epub', # Sync individual processor
        return_value=create_success_result("mock_ebook.epub", "ebook_mock_def", media_type="ebook")
    )

    # --- Mock DB Functions (Only if called *directly* by endpoint/helpers, not by core processors) ---
    # If core processors handle DB interaction, these mocks might not be needed here.
    # Let's assume core processors return db_id/db_message.
    mock_add_db = mocker.patch(
        base_path + 'add_media_to_database', # Sync DB function (maybe redundant)
        return_value={"id": "db_mock_id_direct", "message": "Added directly to DB"}
    )
    # This helper might not be needed if core functions return structured data
    mock_extract_id_string = mocker.patch(
        base_path + 'extract_media_id_from_result_string',
        return_value="extracted_db_id_fallback"
    )

    # --- Mock BackgroundTasks ---
    mock_bg_tasks_instance = MagicMock(spec=BackgroundTasks)
    # How to ensure the endpoint receives this mock?
    # FastAPI injects BackgroundTasks automatically. Patching it globally might be tricky.
    # For testing background tasks, often integration tests are better.
    # We can check if shutil.rmtree is called (or not) instead.

    # --- Mock Utilities ---
    # Make sanitize_filename slightly more realistic if needed
    mocker.patch(
        base_path + "sanitize_filename",
        side_effect=lambda f: f"sanitized_{Path(f).stem}{Path(f).suffix}"
    )

    # Return dict of mocks for tests to use/assert
    return {
        "requests_get": mock_requests_get,
        "process_videos": mock_process_videos,
        "process_audio_files": mock_process_audio_files,
        "process_pdf_task": mock_process_pdf_task,
        "import_plain_text_file": mock_import_plain_text_file,
        "import_epub": mock_import_epub,
        "add_db": mock_add_db,
        "extract_id_string": mock_extract_id_string,
        "rmtree": mock_rmtree,
        "open": mock_open_instance,
        "bg_tasks": mock_bg_tasks_instance, # Note: Injecting this mock requires more setup
        "temp_dir": mock_temp_dir,
        "fake_media_temp_path": fake_media_temp_path, # The path used inside mkdtemp mock
    }

def create_success_result(input_ref, db_id="mock_db_id", content="Mock content", media_type="unknown"):
    return {
        "status": "Success",
        "input_ref": input_ref,
        "processing_source": "url" if "http" in str(input_ref) else "upload",
        "media_type": media_type,
        "metadata": {"title": "Mock Title"},
        "content": content,
        "chunks": [{"text": content, "metadata": {}}] if content else None,
        "analysis": "Mock analysis" if content else None,
        "analysis_details": {"model": "mock_model"},
        "error": None,
        "warnings": [],
        "db_id": db_id,
        "db_message": "Added to DB" # Simulate DB add success within the processor
    }

def create_error_result(input_ref, error_msg="Processing failed", media_type="unknown"):
     return {
        "status": "Error",
        "input_ref": input_ref,
        "processing_source": "url" if "http" in str(input_ref) else "upload",
        "media_type": media_type,
        "metadata": {},
        "content": None,
        "chunks": None,
        "analysis": None,
        "analysis_details": {},
        "error": error_msg,
        "warnings": [],
        "db_id": None,
        "db_message": None
    }

# --- Helper for Form Data (Returns Dict) ---
def create_form_data(media_type: Optional[str] = "video", **overrides) -> Dict[str, Any]:
    """Creates form data dict suitable for TestClient(data=...)."""
    defaults = {
        "media_type": media_type,
        "urls": None, # Will be a list if provided
        "keywords": "test, default",
        # ... include all other relevant form fields from your MediaItemCreate model ...
        "whisper_model": "tiny",
        "transcription_language": "en",
        "diarize": False,
        "timestamp_option": True,
        "vad_use": False,
        "start_time": None,
        "end_time": None,
        "perform_confabulation_check_of_analysis": False,
        "pdf_parsing_engine": "pymupdf4llm",
        "perform_chunking": False,
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
        "keep_original_file": False, # Default to cleanup
        "perform_analysis": True,
        "api_name": None,
        "api_key": None,
        "use_cookies": False,
        "cookies": None,
        "perform_rolling_summarization": False,
        "summarize_recursively": False,
    }
    data = defaults.copy() # Use copy to avoid modifying defaults
    data.update(overrides)

    # Prepare for TestClient: filter None, convert bools, handle lists
    form_dict = {}
    for k, v in data.items():
        if v is not None:
            if isinstance(v, bool):
                form_dict[k] = str(v).lower() # 'true' / 'false'
            elif isinstance(v, (int, float)):
                 form_dict[k] = str(v)
            elif isinstance(v, list) and k == 'urls':
                # TestClient handles list values for 'urls' directly if the key is repeated
                # Pydantic Form(...) usually handles this. Pass the list.
                 form_dict[k] = v
            elif isinstance(v, str):
                form_dict[k] = v
            # Add other type conversions if needed

    # Special case: remove media_type if explicitly testing missing field
    if media_type is None and "media_type" in form_dict:
        del form_dict["media_type"]

    return form_dict

# ##################################################################################################################
# Test Cases
# ##################################################################################################################

# === Validation Tests ===

def test_add_media_invalid_media_type_value(client):
    """Test sending an invalid value for the media_type enum."""
    form_data = create_form_data(media_type="picture", urls=["http://a.com"])
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "Input should be 'video', 'audio', 'document', 'pdf' or 'ebook'" in str(response.json())

def test_add_media_invalid_field_type(client):
    """Test sending a non-boolean string for a boolean field."""
    form_data = create_form_data(media_type="video", urls=["http://a.com"])
    form_data["diarize"] = "maybe" # Invalid boolean representation
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "diarize" in str(response.json())
    assert "Input should be a valid boolean" in str(response.json())

def test_add_media_missing_url_and_file(client):
    """Test calling the endpoint with neither URLs nor files provided."""
    form_data = create_form_data(media_type="video", urls=None) # Explicitly no URLs
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "At least one 'url' in the 'urls' list or one 'file' in the 'files' list must be provided" in response.json()["detail"]

def test_add_media_missing_required_form_field(client):
    """Test calling without a required field like media_type."""
    form_data = create_form_data(media_type=None, urls=["http://a.com"]) # Explicitly remove media_type
    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "media_type" in str(response.json())
    assert "Field required" in str(response.json())

def test_add_media_cookies_validation(client, mock_endpoint_dependencies):
    """Test validation for use_cookies and cookies fields."""
    # Case 1: use_cookies=True but cookies is missing (should fail model validation)
    form_data_no_cookie = create_form_data(media_type="video", urls=["http://a.com"], use_cookies=True, cookies=None)
    response_no_cookie = client.post(ADD_MEDIA_ENDPOINT, data=form_data_no_cookie, headers={"token": "fake-token"})
    assert response_no_cookie.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "Cookie string must be provided" in str(response_no_cookie.json())

    # Case 2: Both provided (should pass model validation, mock processing handles rest)
    form_data_with_cookie = create_form_data(media_type="video", urls=["http://a.com"], use_cookies=True, cookies="session=123")
    # Ensure the mock processor returns success for this case
    mock_endpoint_dependencies["process_videos"].return_value = {
        "results": [create_success_result("http://a.com", "vid_cookie_mock", media_type="video")],
        "processed_count": 1, "errors_count": 0, "errors": []
    }
    response_with_cookie = client.post(ADD_MEDIA_ENDPOINT, data=form_data_with_cookie, headers={"token": "fake-token"})
    assert response_with_cookie.status_code == status.HTTP_200_OK
    # Check the result from the mocked processor
    assert response_with_cookie.json()["results"][0]["status"] == "Success"
    assert response_with_cookie.json()["results"][0]["db_id"] == "vid_cookie_mock"
    assert response_with_cookie.json()["results"][0]["input_ref"] == "http://a.com"


# === File Handling Tests ===

def test_add_media_single_file_upload_success_cleanup(client, create_upload_file, mock_endpoint_dependencies):
    """Test successful upload of one file with cleanup enabled (default)."""
    test_file_tuple = create_upload_file(filename="audio.mp3", content_type="audio/mpeg")
    form_data = create_form_data(media_type="audio", keep_original_file=False) # Explicit cleanup
    fake_temp_path = mock_endpoint_dependencies["fake_media_temp_path"]
    expected_saved_path = fake_temp_path / "sanitized_audio.mp3"

    # Configure mock processor for this specific file input
    mock_endpoint_dependencies["process_audio_files"].return_value = {
        "results": [create_success_result(str(expected_saved_path), "aud_clean_mock", media_type="audio")],
        "processed_count": 1, "errors_count": 0, "errors": []
    }

    response = client.post(
        ADD_MEDIA_ENDPOINT,
        files={"files": test_file_tuple}, # Use dict format for TestClient files
        data=form_data,
        headers={"token": "fake-token"}
    )

    assert response.status_code == status.HTTP_200_OK
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Success"
    assert results[0]["db_id"] == "aud_clean_mock"
    assert results[0]["input_ref"] == str(expected_saved_path) # Input ref is the path given to processor

    # Check that the file was "saved" (mock open called with correct path)
    # Note: If using real 'open', check file exists in mock_temp_dir instead
    mock_endpoint_dependencies["open"].assert_called_with(expected_saved_path, "wb")

    # Check that cleanup (rmtree) was called via background task mechanism
    # Direct assertion on bg_tasks.add_task is hard without injection.
    # Check if rmtree was called (assuming it's added to bg task).
    mock_endpoint_dependencies["rmtree"].assert_called_once()
    # We expect it to be called on the parent directory created by mkdtemp mock
    mock_endpoint_dependencies["rmtree"].assert_called_with(str(fake_temp_path))


def test_add_media_single_file_upload_success_no_cleanup(client, create_upload_file, mock_endpoint_dependencies):
    """Test successful upload of one file with cleanup disabled."""
    test_file_tuple = create_upload_file("audio.mp3", content_type="audio/mpeg")
    form_data = create_form_data(media_type="audio", keep_original_file=True) # Cleanup disabled
    fake_temp_path = mock_endpoint_dependencies["fake_media_temp_path"]
    expected_saved_path = fake_temp_path / "sanitized_audio.mp3"

    # Configure mock processor
    mock_endpoint_dependencies["process_audio_files"].return_value = {
        "results": [create_success_result(str(expected_saved_path), "aud_noclean_mock", media_type="audio")],
        "processed_count": 1, "errors_count": 0, "errors": []
    }

    response = client.post(
        ADD_MEDIA_ENDPOINT,
        files={"files": test_file_tuple},
        data=form_data,
        headers={"token": "fake-token"}
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["results"][0]["status"] == "Success"

    # Check that cleanup (rmtree) was NOT called
    mock_endpoint_dependencies["rmtree"].assert_not_called()

@patch('builtins.open', side_effect=OSError("Disk full")) # Patch open directly for this test
def test_add_media_file_save_io_error(mock_open_error, client, create_upload_file, mock_endpoint_dependencies):
    """Test an OSError during the file saving stage."""
    test_file_tuple = create_upload_file("audio.mp3", content_type="audio/mpeg")
    form_data = create_form_data(media_type="audio")

    response = client.post(
        ADD_MEDIA_ENDPOINT,
        files={"files": test_file_tuple},
        data=form_data,
        headers={"token": "fake-token"}
    )

    # The endpoint should catch this error per file and return 207
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Failed" # Should be Failed, not Error maybe? Check endpoint logic
    assert results[0]["input_ref"] == "audio.mp3" # Input ref is original filename before saving fails
    assert "Failed to save uploaded file" in results[0]["error"]
    assert "Disk full" in results[0]["error"] # Check the specific OSError message
    assert "OSError" in results[0]["error"] # Check the exception type name

    # Ensure the audio processor was NOT called because saving failed
    mock_endpoint_dependencies["process_audio_files"].assert_not_called()

# Test temp dir creation error - relies on mocking 'tempfile.mkdtemp'
@patch('tempfile.mkdtemp', side_effect=OSError("Permission denied creating temp dir"))
def test_add_media_temp_dir_creation_error(mock_mkdtemp_error, client, create_upload_file):
    """Test failure during temporary directory creation (at endpoint start)."""
    # This patch needs to happen *before* the endpoint tries to create the temp dir.
    # The autouse fixture might interfere slightly, but patching here should take precedence.
    test_file_tuple = create_upload_file("audio.mp3", content_type="audio/mpeg")
    form_data = create_form_data(media_type="audio")

    response = client.post(
        ADD_MEDIA_ENDPOINT,
        files={"files": test_file_tuple},
        data=form_data,
        headers={"token": "fake-token"}
    )
    # If mkdtemp fails right at the start within the endpoint's context manager/setup,
    # it should likely raise a 500 Internal Server Error.
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    # Check detail message if the endpoint wraps the error
    assert "Failed to create temporary directory" in response.json()["detail"]
    assert "Permission denied" in response.json()["detail"]


# === Download Error Tests ===

def test_add_media_pdf_download_404(client, mock_endpoint_dependencies):
    """Test a 404 error when downloading a URL."""
    pdf_url = "http://example.com/not_found.pdf"
    form_data = create_form_data(media_type="pdf", urls=[pdf_url])

    # Configure requests.get mock for 404
    mock_response_404 = MagicMock(spec=requests.Response)
    mock_response_404.status_code = 404
    mock_response_404.url = pdf_url
    mock_response_404.reason = "Not Found"
    mock_response_404.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "404 Client Error: Not Found", response=mock_response_404
    )
    mock_endpoint_dependencies["requests_get"].return_value = mock_response_404

    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers={"token": "fake-token"})

    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["input_ref"] == pdf_url
    assert results[0]["status"] == "Failed"
    assert "Download failed" in results[0]["error"]
    assert "404 Client Error" in results[0]["error"]

    # Check requests.get call
    mock_endpoint_dependencies["requests_get"].assert_called_once_with(pdf_url, timeout=ANY, allow_redirects=True, cookies=None, headers=ANY) # Added headers=ANY
    # Ensure PDF processor was NOT called
    mock_endpoint_dependencies["process_pdf_task"].assert_not_called()

def test_add_media_doc_download_network_error(client, mock_endpoint_dependencies):
    """Test a network connection error during URL download."""
    doc_url = "http://example.com/unreachable.doc"
    form_data = create_form_data(media_type="document", urls=[doc_url])

    # Configure requests.get to raise ConnectionError
    mock_endpoint_dependencies["requests_get"].side_effect = requests.exceptions.ConnectionError("DNS lookup failed")

    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["input_ref"] == doc_url
    assert results[0]["status"] == "Failed"
    assert "Download failed" in results[0]["error"]
    assert "DNS lookup failed" in results[0]["error"]
    # Ensure Document processor was NOT called
    mock_endpoint_dependencies["import_plain_text_file"].assert_not_called()

# === Processing Function Error Tests ===

def test_add_media_video_processing_exception(client, mock_endpoint_dependencies):
    """Test an exception raised by the batch video processor."""
    video_url = "http://example.com/bad_video.mp4"
    form_data = create_form_data(media_type="video", urls=[video_url])

    # Configure the batch processor mock to raise an error
    mock_endpoint_dependencies["process_videos"].side_effect = ValueError("Bad video muxer")

    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS # Endpoint handles batch errors gracefully
    results = response.json()["results"]
    # The endpoint might return a single "Failed" item summarizing the batch error,
    # or it might try to return individual failures if the processor indicates which input failed.
    # Assuming a general failure reported back.
    assert len(results) >= 1 # Could be 1 general failure or 1 per input
    # Let's assume the endpoint creates a failure record for the input
    fail_result = next((r for r in results if r["input_ref"] == video_url), None)
    assert fail_result is not None
    assert fail_result["status"] == "Failed"
    # Check the error message structure from _process_batch_media or similar handler
    assert "Batch video processing error" in fail_result["error"]
    assert "ValueError: Bad video muxer" in fail_result["error"]

def test_add_media_audio_processing_returns_error_status(client, mock_endpoint_dependencies):
    """Test the audio processor returning a failure status in its result dict."""
    audio_url = "http://example.com/noisy.mp3"
    form_data = create_form_data(media_type="audio", urls=[audio_url])

    # Configure the batch processor mock to return a failure dict for the item
    mock_endpoint_dependencies["process_audio_files"].return_value = {
        "results": [create_error_result(audio_url, "Audio too noisy", media_type="audio")],
        "processed_count": 0,
        "errors_count": 1,
        "errors": [{"input": audio_url, "error": "Audio too noisy"}]
    }

    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Error" # Matches the status from create_error_result
    assert results[0]["input_ref"] == audio_url
    assert results[0]["error"] == "Audio too noisy"


def test_add_media_pdf_processing_exception(client, mock_endpoint_dependencies):
    """Test an exception raised by the async PDF processor."""
    pdf_url = "http://example.com/bad.pdf"
    form_data = create_form_data(media_type="pdf", urls=[pdf_url])
    fake_temp_path = mock_endpoint_dependencies["fake_media_temp_path"]
    # Assume download succeeds, saving to a temp path before processing is called
    expected_saved_path = fake_temp_path / "sanitized_bad.pdf"

    # Mock requests.get to succeed for download part
    mock_response_ok = MagicMock(spec=requests.Response, status_code=200, content=b'bad pdf content', url=pdf_url)
    mock_response_ok.raise_for_status = MagicMock()
    mock_endpoint_dependencies["requests_get"].return_value = mock_response_ok

    # Configure the async mock PDF processor to raise the error
    async def mock_pdf_raiser(*args, **kwargs):
        raise ValueError("Cannot parse PDF")
    mock_endpoint_dependencies["process_pdf_task"].side_effect = mock_pdf_raiser

    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers={"token": "fake-token"})

    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["input_ref"] == pdf_url # Original input reference
    assert results[0]["status"] == "Failed"
    # Check the error format from the endpoint's exception handling
    assert "File processing error" in results[0]["error"] # Or similar wrapper message
    assert "ValueError: Cannot parse PDF" in results[0]["error"]

    # Check that the processor was called (after download and save)
    mock_endpoint_dependencies["process_pdf_task"].assert_awaited_once()
    # Check args passed to processor if needed (e.g., the temp file path)
    # _, call_kwargs = mock_endpoint_dependencies["process_pdf_task"].call_args
    # assert call_kwargs['input_ref'] == str(expected_saved_path) # Might be path or URL

# === Database / Result Handling Error Tests ===
# These tests depend on whether the core processors handle DB ops or the endpoint does.
# Assuming core processors return DB status/ID.

def test_add_media_pdf_processor_returns_db_error(client, mock_endpoint_dependencies):
    """Test when the PDF processor succeeds processing but reports a DB error."""
    pdf_url = "http://example.com/good_but_db_fail.pdf"
    form_data = create_form_data(media_type="pdf", urls=[pdf_url])

    # Mock download success
    mock_response_ok = MagicMock(spec=requests.Response, status_code=200, content=b'good pdf', url=pdf_url)
    mock_response_ok.raise_for_status = MagicMock()
    mock_endpoint_dependencies["requests_get"].return_value = mock_response_ok

    # Mock PDF processing to succeed but return DB error info
    async def mock_pdf_db_fail(*args, **kwargs):
        result = create_success_result(pdf_url, media_type="pdf", content="PDF text")
        result["status"] = "Warning" # Indicate partial success
        result["db_id"] = None
        result["db_message"] = "DB connection failed during save"
        result["error"] = "Processed successfully, but failed to add to database: DB connection failed during save"
        return result
    mock_endpoint_dependencies["process_pdf_task"].side_effect = mock_pdf_db_fail

    response = client.post(ADD_MEDIA_ENDPOINT, data=form_data, headers={"token": "fake-token"})

    assert response.status_code == status.HTTP_207_MULTI_STATUS # Overall status reflects the warning/error
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Warning" # Check the status reported back
    assert results[0]["input_ref"] == pdf_url
    assert "Processed successfully, but failed to add to database" in results[0]["error"]
    assert "DB connection failed" in results[0]["error"]
    assert results[0].get("db_id") is None

# === Success Path Examples (Multi-Input) ===

def test_add_media_mixed_url_file_success(client, create_upload_file, mock_endpoint_dependencies):
    """Test adding one video URL and one video file successfully."""
    video_url = "http://example.com/good_video.mp4"
    test_file_tuple = create_upload_file("another.mp4", content_type="video/mp4")
    form_data = create_form_data(media_type="video", urls=[video_url])
    fake_temp_path = mock_endpoint_dependencies["fake_media_temp_path"]
    expected_file_input_path = str(fake_temp_path / "sanitized_another.mp4")

    # Mock the batch processor to return success results for both inputs
    mock_endpoint_dependencies["process_videos"].return_value = {
        "results": [
            create_success_result(video_url, "vid_mock_url", media_type="video"),
            create_success_result(expected_file_input_path, "vid_mock_file", media_type="video")
        ],
        "processed_count": 2, "errors_count": 0, "errors": []
    }

    response = client.post(
        ADD_MEDIA_ENDPOINT,
        files={"files": test_file_tuple},
        data=form_data,
        headers={"token": "fake-token"}
    )

    # Both succeeded, expect 200
    assert response.status_code == status.HTTP_200_OK
    results = response.json()["results"]
    assert len(results) == 2
    results_by_input = {r["input_ref"]: r for r in results} # Use actual input_ref from result

    # Check result for URL input
    url_result = results_by_input.get(video_url)
    assert url_result is not None
    assert url_result["status"] == "Success"
    assert url_result["db_id"] == "vid_mock_url"

    # Check result for File input
    file_result = results_by_input.get(expected_file_input_path)
    assert file_result is not None
    assert file_result["status"] == "Success"
    assert file_result["db_id"] == "vid_mock_file"

    # Verify the batch processor was called once with expected inputs
    mock_endpoint_dependencies["process_videos"].assert_called_once()
    call_args, call_kwargs = mock_endpoint_dependencies["process_videos"].call_args
    # Check the 'inputs' list passed to the batch processor
    assert call_kwargs['inputs'] == [video_url, expected_file_input_path]
    # Check other relevant args passed
    assert call_kwargs['whisper_model'] == 'tiny'

def test_add_media_multiple_failures_and_success(client, create_upload_file, mock_endpoint_dependencies):
    """Test a mix of successful and failed items across URLs and files for PDF."""
    good_pdf_url = "http://example.com/good.pdf"
    bad_dl_url = "http://example.com/notfound.pdf" # Download fails
    good_pdf_file = create_upload_file("report.pdf", content_type="application/pdf")
    bad_process_file = create_upload_file("corrupt.pdf", content_type="application/pdf") # Processing fails

    form_data = create_form_data(media_type="pdf", urls=[good_pdf_url, bad_dl_url])
    files_dict = {
        "files": [ # Pass list of tuples for multiple files
             good_pdf_file,
             bad_process_file,
        ]
    }
    fake_temp_path = mock_endpoint_dependencies["fake_media_temp_path"]
    good_file_path = fake_temp_path / "sanitized_report.pdf"
    bad_file_path = fake_temp_path / "sanitized_corrupt.pdf"

    # --- Mock Behaviors ---
    # 1. Mock requests.get: Success for good_pdf_url, 404 for bad_dl_url
    mock_response_good = MagicMock(spec=requests.Response, status_code=200, content=b'good pdf', url=good_pdf_url)
    mock_response_good.raise_for_status = MagicMock()
    mock_response_bad = MagicMock(spec=requests.Response, status_code=404, url=bad_dl_url, reason="Not Found")
    mock_response_bad.raise_for_status.side_effect = requests.exceptions.HTTPError("404", response=mock_response_bad)
    def req_side_effect(url, *args, **kwargs):
        if url == good_pdf_url: return mock_response_good
        if url == bad_dl_url: return mock_response_bad
        pytest.fail(f"Unexpected URL in requests.get mock: {url}")
    mock_endpoint_dependencies["requests_get"].side_effect = req_side_effect

    # 2. Mock process_pdf_task: Success for good items, Exception for corrupt file
    async def pdf_task_side_effect(*args, **kwargs):
        input_ref = kwargs.get("input_ref") # This might be URL or temp path
        # filename = kwargs.get("filename") # Processor might receive original filename too

        # Determine input based on path or URL
        if input_ref == good_pdf_url:
             return create_success_result(good_pdf_url, "pdf_db_good_url", media_type="pdf")
        elif input_ref == str(good_file_path):
            return create_success_result(str(good_file_path), "pdf_db_good_file", media_type="pdf")
        elif input_ref == str(bad_file_path):
             raise ValueError("Cannot parse corrupt PDF")
        else:
             pytest.fail(f"Unexpected input in pdf_task_side_effect: {input_ref}")
    mock_endpoint_dependencies["process_pdf_task"].side_effect = pdf_task_side_effect

    # --- Run Test ---
    response = client.post(ADD_MEDIA_ENDPOINT, files=files_dict, data=form_data, headers={"token": "fake-token"})

    # --- Assertions ---
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 4 # good_url, bad_url, good_file, bad_process_file

    # Find results by original input reference (URL or original filename)
    results_map = {}
    for r in results:
        # Need to map the temp path back to original filename for uploaded files
        if str(good_file_path) in r["input_ref"]: results_map["report.pdf"] = r
        elif str(bad_file_path) in r["input_ref"]: results_map["corrupt.pdf"] = r
        elif bad_dl_url == r["input_ref"]: results_map[bad_dl_url] = r
        elif good_pdf_url == r["input_ref"]: results_map[good_pdf_url] = r
        else: results_map[r["input_ref"]] = r # Should cover URLs

    # Check good URL
    assert results_map[good_pdf_url]["status"] == "Success"
    assert results_map[good_pdf_url]["db_id"] == "pdf_db_good_url"

    # Check bad download URL
    assert results_map[bad_dl_url]["status"] == "Failed"
    assert "Download failed" in results_map[bad_dl_url]["error"]
    assert "404" in results_map[bad_dl_url]["error"]

    # Check good file
    assert results_map["report.pdf"]["status"] == "Success"
    assert results_map["report.pdf"]["db_id"] == "pdf_db_good_file"
    assert results_map["report.pdf"]["input_ref"] == str(good_file_path) # Actual input to processor

    # Check bad processing file
    assert results_by_input["corrupt.pdf"]["status"] == "Failed"
    assert "File/Input error: Cannot parse corrupt PDF" in results_by_input["corrupt.pdf"]["error"]

    # Check call counts
    assert mock_dependencies["requests_get"].call_count == 2
    assert mock_dependencies["process_pdf_task"].call_count == 3 # Called for good_pdf_url and good_pdf_file, bad_process_file
    # DB add should be called twice (for the two successful PDF processes)
    assert mock_dependencies["add_db"].call_count == 2


# ##################################################################################################################
# End of test_add_media_endpoint.py
# ##################################################################################################################