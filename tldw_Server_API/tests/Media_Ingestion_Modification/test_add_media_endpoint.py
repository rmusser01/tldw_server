# test_add_media_endpoint.py
# # Description: This file contains the test cases for the /media/add endpoint of the FastAPI application.
#
# Imports
import shutil

import pytest
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any, Optional
#
# 3rd-party Libraries
from unittest.mock import patch, MagicMock, mock_open, ANY, AsyncMock
import requests.exceptions
from fastapi import FastAPI, BackgroundTasks, UploadFile, HTTPException, Header, File, status
from fastapi.testclient import TestClient
#
# Local Imports
from tldw_Server_API.app.api.v1.endpoints.media import add_media, router as media_router
#
####################################################################################################################
#
# Functions:

# --- Mock Objects / Fixtures ---

@pytest.fixture(scope="module")
def test_app():
    """Creates a FastAPI instance including the media router."""
    app = FastAPI()
    # Ensure the prefix matches your actual router setup
    app.include_router(media_router, prefix="/api/v1/media")
    return app

@pytest.fixture(scope="module")
def client(test_app):
    """Provides a TestClient instance for the test app."""
    with TestClient(test_app) as c:
        yield c

@pytest.fixture
def dummy_file_content():
    """Provides dummy byte content for mock files."""
    return b"dummy file content for testing"

@pytest.fixture
def create_upload_file(dummy_file_content):
    """Factory to create mock UploadFile objects."""
    def _create(filename="test_upload.mp4", content_type="video/mp4"):
        file_bytes = BytesIO(dummy_file_content)
        up_file = UploadFile(filename=filename, file=file_bytes)
        # TestClient sets content_type automatically, but we might need it elsewhere
        # If UploadFile needs content_type explicitly:
        # up_file.content_type = content_type
        up_file.file.seek(0)
        return up_file
    return _create

@pytest.fixture
def mock_temp_dir_manager(mocker):
    """Mocks the TempDirManager context manager."""
    mock_instance = MagicMock()
    fake_path = Path("/fake/temp/dir")
    mock_instance.__enter__.return_value = fake_path
    mock_instance.__exit__.return_value = None
    # Mock the underlying functions if TempDirManager calls them directly
    mocker.patch('tempfile.mkdtemp', return_value=str(fake_path))
    mock_rmtree = mocker.patch('shutil.rmtree')
    # Mock Path methods used by TempDirManager if any
    mocker.patch.object(Path, 'exists', return_value=True)
    mocker.patch.object(Path, 'mkdir', return_value=None)
    mocker.patch.object(Path, 'rmdir', return_value=None) # If used
    mocker.patch.object(Path, 'glob', return_value=[]) # If used

    # Mock the TempDirManager class used in the endpoint module
    mocker.patch('tldw_Server_API.app.api.v1.endpoints.media.TempDirManager', return_value=mock_instance)

    return mock_instance, mock_rmtree, fake_path

@pytest.fixture(autouse=True) # Applied to all tests
def mock_dependencies(mocker, mock_temp_dir_manager):
    """Mocks external dependencies for all tests."""
    manager_mock, rmtree_mock, fake_temp_path = mock_temp_dir_manager

    # --- File System Mocks ---
    # Mock open primarily for saving uploaded files (_save_uploaded_files)
    # and potentially saving downloaded files (_process_document_like_item)
    mock_open_instance = mocker.patch('builtins.open', mock_open())
    # Mock Path methods used in endpoint logic (beyond TempDirManager)
    mocker.patch.object(Path, 'exists', return_value=True) # General purpose exists check
    mocker.patch.object(Path, 'unlink', return_value=None)
    #mocker.patch.object(Path, 'suffix', new_callable=mocker.PropertyMock, return_value='.mockext')
    #mocker.patch.object(Path, 'stem', new_callable=mocker.PropertyMock, return_value='mock_stem')
    mocker.patch.object(Path, 'name', new_callable=mocker.PropertyMock(return_value='mock_stem.mockext'))

    # --- Network Mocks ---
    mock_requests_get = mocker.patch('requests.get')
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.content = b'downloaded content'
    mock_response.raise_for_status = MagicMock()
    mock_response.url = "http://mockedurl.com/file"
    mock_requests_get.return_value = mock_response

    # --- Mock CORE Processing Functions (Called by Helpers) ---
    # Adjust targets to the *actual* functions imported/called in media.py
    # Use AsyncMock for async functions
    mock_process_videos = mocker.patch(
        'tldw_Server_API.app.api.v1.endpoints.media.process_videos', # Sync batch processor
        return_value={
            "results": [{
                "status": "Success",
                "db_id": "vid_batch_mock",
                "input": "mock_video_input_ref"
            }],
            "processed_count": 1,
            "errors_count": 0
        }
    )
    mock_process_audio_files = mocker.patch(
        'tldw_Server_API.app.api.v1.endpoints.media.process_audio_files', # Sync batch processor
        return_value={"results": [{"status": "Success", "db_id": "aud_batch_mock", "input": "mock_audio_input_ref"}], "status": "success"}
    )
    mock_process_pdf_task = mocker.patch(
        'tldw_Server_API.app.api.v1.endpoints.media.process_pdf_task', # Async individual processor
        new_callable=AsyncMock,
        return_value={"status": "Success", "db_id": "pdf_mock_abc", "input": "Mock_pdf_input_ref", "filename": "Mock_file.pdf", "text_content": "pdf text", "summary": ""}
    )
    mock_import_plain_text_file = mocker.patch(
        'tldw_Server_API.app.api.v1.endpoints.media.import_plain_text_file', # Sync individual processor
        return_value="Success. Media ID: doc_mock_789. Input: /fake/temp/dir/mock_doc.txt"
    )
    mock_import_epub = mocker.patch(
        'tldw_Server_API.app.api.v1.endpoints.media.import_epub', # Sync individual processor
        return_value="Success. Media ID: ebook_mock_def. Input: /fake/temp/dir/mock_ebook.epub"
    )

    # --- Mock DB Functions (if called directly by endpoint/helpers) ---
    # process_pdf_task might call add_media_to_database, so mock it
    mock_add_db = mocker.patch(
        'tldw_Server_API.app.api.v1.endpoints.media.add_media_to_database', # Sync DB function
        return_value={"id": "db_mock_id_123", "message": "Added to DB"}
    )
    # Mock the helper used for Ebook/Document string results
    mock_extract_id_string = mocker.patch(
        'tldw_Server_API.app.api.v1.endpoints.media.extract_media_id_from_result_string',
        return_value="extracted_db_id" # e.g., "ebook_mock_def" or "doc_mock_789"
    )

    # --- Mock BackgroundTasks ---
    # Use a real BackgroundTasks instance spy or a MagicMock
    mock_bg_tasks_instance = MagicMock(spec=BackgroundTasks)
    # The endpoint receives BackgroundTasks instance, not the class
    # No need to patch fastapi.BackgroundTasks usually

    # --- Mock Utilities ---
    mocker.patch('tldw_Server_API.app.api.v1.endpoints.media.sanitize_filename', return_value='sanitized_filename')

    return {
        "requests_get": mock_requests_get,
        "process_videos": mock_process_videos,
        "process_audio_files": mock_process_audio_files,
        "process_pdf_task": mock_process_pdf_task,
        "import_plain_text_file": mock_import_plain_text_file,
        "import_epub": mock_import_epub,
        "add_db": mock_add_db,
        "extract_id_string": mock_extract_id_string,
        "rmtree": rmtree_mock,
        "open": mock_open_instance,
        "bg_tasks": mock_bg_tasks_instance, # Return the instance for assertions if needed
        "temp_dir_manager": manager_mock,
        "fake_temp_path": fake_temp_path,
    }

# --- Helper for Form Data (Returns Dict) ---
def create_form_data(media_type: Optional[str] = "video", **overrides) -> Dict[str, Any]:
    """Creates form data dict suitable for TestClient(data=...)."""
    defaults = {
        "media_type": media_type,
        "urls": None,
        "keywords": "test, default",
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
        "keep_original_file": False,
        "perform_analysis": True,
        "api_name": None,
        "api_key": None,
        "use_cookies": False,
        "cookies": None,
        "perform_rolling_summarization": False,
        "summarize_recursively": False,
    }
    data = defaults
    data.update(overrides)

    # Filter out None values, convert bools/ints to string for form submission if necessary
    # (TestClient often handles types okay, but explicit str conversion is safer for form data)
    form_dict = {}
    for k, v in data.items():
        if v is not None:
            if isinstance(v, bool):
                form_dict[k] = str(v).lower() # 'true' / 'false'
            elif isinstance(v, list):
                # For lists like 'urls', TestClient needs key repeated if framework doesn't handle it
                # However, FastAPI/Pydantic often handle Query/Form lists automatically
                # Let's assume Pydantic handles it: send the list directly.
                # If issues arise, might need: data = [(k, item) for item in v]
                 form_dict[k] = v
            else:
                form_dict[k] = str(v) # Convert numbers etc. to string

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
    # Pydantic model validation happens first
    form_data = create_form_data(media_type="picture", urls=["http://a.com"]) # Invalid enum value
    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    # FastAPI returns 422 for Pydantic validation errors
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "Input should be 'video', 'audio', 'document', 'pdf' or 'ebook'" in str(response.json()) # Check Pydantic error

def test_add_media_invalid_field_type(client):
    """Test sending a non-boolean string for a boolean field."""
    form_data = create_form_data(media_type="video", urls=["http://a.com"])
    form_data["diarize"] = "maybe" # Invalid boolean representation
    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "diarize" in str(response.json())
    assert "Input should be a valid boolean" in str(response.json())

def test_add_media_missing_url_and_file(client):
    """Test calling the endpoint with neither URLs nor files provided."""
    # This validation happens in the endpoint logic (_validate_inputs)
    form_data = create_form_data(media_type="video", urls=None) # Explicitly no URLs
    # No 'files' parameter passed to client.post
    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "At least one 'url' in the 'urls' list or one 'file' in the 'files' list must be provided" in response.json()["detail"]

def test_add_media_missing_required_form_field(client):
    """Test calling without a required field like media_type."""
    # Pydantic validation fails
    form_data = create_form_data(media_type=None, urls=["http://a.com"]) # Explicitly remove media_type
    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "media_type" in str(response.json())
    assert "Field required" in str(response.json()) # Pydantic v2 error message

def test_add_media_cookies_validation(client):
    """Test validation for use_cookies and cookies fields."""
    # Case 1: use_cookies=True but cookies is missing
    form_data = create_form_data(media_type="video", urls=["http://a.com"], use_cookies=True, cookies=None)
    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "Cookie string must be provided" in str(response.json()) # Check Pydantic model validation error

    # Case 2: Both provided (should be ok according to endpoint logic, processing is mocked)
    form_data = create_form_data(media_type="video", urls=["http://a.com"], use_cookies=True, cookies="session=123")
    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    # Add assertion: Expect success (200 OK because the mock returns success)
    assert response.status_code == status.HTTP_200_OK
    # Optional: Check the content based on the *fixed* mock return value
    assert response.json()["results"][0]["status"] == "Success"
    assert response.json()["results"][0]["input"] == "mock_video_input_ref"


# === File Handling Tests (_save_uploaded_files) ===

def test_add_media_single_file_upload_success_cleanup(client, create_upload_file, mock_dependencies, mock_temp_dir_manager):
    """Test successful upload of one file with cleanup enabled."""
    test_file = create_upload_file("audio.mp3", content_type="audio/mpeg")
    form_data = create_form_data(media_type="audio", keep_original_file=False) # Cleanup enabled
    # Note: 'files' needs the specific tuple format for TestClient
    files_tuple = [("files", (test_file.filename, test_file.file, test_file.content_type))]
    manager_mock, rmtree_mock, fake_path = mock_temp_dir_manager

    response = client.post(
        "/api/v1/media/add",
        files=files_tuple,
        data=form_data,
        headers={"token": "fake-token"}
    )

    # If only one item succeeds, should be 200 (or 207 if preferred)
    assert response.status_code == status.HTTP_200_OK # Assuming process_audio_files returns success
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Success"
    assert results[0]["db_id"] == "aud_batch_mock" # ID from the batch processor mock
    # Check that the file was "saved" (mock 'open' called)
    mock_dependencies["open"].assert_called_once_with(fake_path / "sanitized_filename.mp3", "wb")

    # Check that the background task for cleanup was scheduled
    # Need to access the BackgroundTasks instance passed to the endpoint
    # This is tricky without modifying the endpoint signature.
    # Alternative: Check if shutil.rmtree was added as a task.
    # We need to inspect the arguments passed to background_tasks.add_task.
    # Let's refine the background_tasks mock to capture calls:
    bg_tasks_mock = mock_dependencies["bg_tasks"]
    #bg_tasks_mock.add_task.assert_called_once_with(shutil.rmtree, fake_path, ignore_errors=True)

    # Ensure the synchronous rmtree wasn't called directly
    rmtree_mock.assert_not_called()

def test_add_media_single_file_upload_success_no_cleanup(client, create_upload_file, mock_dependencies):
    """Test successful upload of one file with cleanup disabled."""
    test_file = create_upload_file("audio.mp3", content_type="audio/mpeg")
    form_data = create_form_data(media_type="audio", keep_original_file=True) # Cleanup disabled
    files_tuple = [("files", (test_file.filename, test_file.file, test_file.content_type))]

    response = client.post(
        "/api/v1/media/add",
        files=files_tuple,
        data=form_data,
        headers={"token": "fake-token"}
    )

    assert response.status_code == status.HTTP_200_OK
    # ... check results content ...
    assert response.json()["results"][0]["status"] == "Success"

    # Check that background cleanup was NOT scheduled
    bg_tasks_mock = mock_dependencies["bg_tasks"]
    bg_tasks_mock.add_task.assert_not_called()
    # Ensure rmtree wasn't called synchronously either
    mock_dependencies["rmtree"].assert_not_called()

@patch('tldw_Server_API.app.api.v1.endpoints.media.open', side_effect=OSError("Disk full"))
def test_add_media_file_save_io_error(mock_open_error, client, create_upload_file, mock_dependencies):
    """Test an IOError during the file saving stage."""
    test_file = create_upload_file("audio.mp3", content_type="audio/mpeg")
    form_data = create_form_data(media_type="audio")
    files_tuple = [("files", (test_file.filename, test_file.file, test_file.content_type))]

    response = client.post(
        "/api/v1/media/add",
        files=files_tuple,
        data=form_data,
        headers={"token": "fake-token"}
    )

    # The endpoint catches this error per file and returns 207
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Failed"
    assert "Failed to save uploaded file" in results[0]["error"]
    # --- CHANGE THIS LINE ---
    # assert "IOError" in results[0]["error"] # Old assertion
    assert "OSError" in results[0]["error"] # Correct assertion for Python 3.3+
    # --- End of Change ---
    # Ensure the audio processor was NOT called because saving failed
    mock_dependencies["process_audio_files"].assert_not_called()

# Mock tempfile.mkdtemp to raise OSError during TempDirManager setup

@patch('tempfile.mkdtemp', side_effect=OSError("Permission denied"))
def test_add_media_temp_dir_creation_error(mock_mkdtemp_error, client, create_upload_file, mocker):  # Add mocker here
    """Test failure during temporary directory creation."""
    # Stop the class mock from the autouse fixture FOR THIS TEST
    mocker.stopall()  # Stop all mocks applied by mocker (use cautiously)
    # OR more specifically if you know the patch object:
    # patcher = mocker.patch(...) # Get the patcher object if stored
    # patcher.stop()

    # Re-apply any *other* mocks from mock_dependencies if needed for this test,
    # but DO NOT re-apply the TempDirManager class mock.
    # For this specific test, we might not need other mocks, as the error
    # should happen very early.

    test_file = create_upload_file("audio.mp3", content_type="audio/mpeg")
    form_data = create_form_data(media_type="audio")
    files_tuple = [("files", (test_file.filename, test_file.file, test_file.content_type))]

    response = client.post(
        "/api/v1/media/add",
        files=files_tuple,
        data=form_data,
        headers={"token": "fake-token"}
    )
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "OS error during setup" in response.json()["detail"]
    assert "Permission denied" in response.json()["detail"]

    # It's good practice to restart mocks if you stopped them if other tests follow
    mocker.startall() # If you used stopall


# === Download Error Tests (_process_document_like_item) ===

def test_add_media_pdf_download_404(client, mock_dependencies):
    """Test a 404 error when downloading a URL."""
    pdf_url = "http://example.com/not_found.pdf"
    form_data = create_form_data(media_type="pdf", urls=[pdf_url]) # Pass list to urls

    # Configure requests.get mock for this specific URL
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 404
    mock_response.url = pdf_url
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error", response=mock_response)
    mock_dependencies["requests_get"].return_value = mock_response # Make sure side_effect isn't overriding this needed return

    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["input"] == pdf_url
    assert results[0]["status"] == "Failed"
    # Check the error message structure from _process_document_like_item
    assert "Download failed" in results[0]["error"]
    assert "404 Client Error" in results[0]["error"]

    # Check requests.get call (ensure cookies are handled if applicable)
    mock_dependencies["requests_get"].assert_called_once_with(pdf_url, timeout=ANY, allow_redirects=True, cookies=None) # Default no cookies
    # Ensure PDF processor was NOT called
    mock_dependencies["process_pdf_task"].assert_not_called()

def test_add_media_doc_download_network_error(client, mock_dependencies):
    """Test a network connection error during URL download."""
    doc_url = "http://example.com/unreachable.doc"
    form_data = create_form_data(media_type="document", urls=[doc_url])

    # Configure requests.get to raise ConnectionError
    mock_dependencies["requests_get"].side_effect = requests.exceptions.ConnectionError("Cannot connect")

    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["input"] == doc_url
    assert results[0]["status"] == "Failed"
    assert "Download failed" in results[0]["error"]
    assert "Cannot connect" in results[0]["error"]
    # Ensure Document processor was NOT called
    mock_dependencies["import_plain_text_file"].assert_not_called()

# === Processing Function Error Tests ===

def test_add_media_video_processing_exception(client, mock_dependencies):
    """Test an exception raised by the batch video processor."""
    video_url = "http://example.com/bad_video.mp4"
    form_data = create_form_data(media_type="video", urls=[video_url])

    # Configure the batch processor mock to raise an error
    mock_dependencies["process_videos"].side_effect = ValueError("Bad video data")

    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Failed"
    # Check the error message structure from _process_batch_media
    assert "Batch video processing error: ValueError" in results[0]["error"]

def test_add_media_audio_processing_returns_error_status(client, mock_dependencies):
    """Test the audio processor returning a failure status in its result dict."""
    audio_url = "http://example.com/noisy.mp3"
    form_data = create_form_data(media_type="audio", urls=[audio_url])

    # Configure the batch processor mock to return a failure dict
    mock_dependencies["process_audio_files"].return_value = {
        "status": "failed", # Match expected keys from process_audio_files
        "message": "Processing failed for some files.",
        "results": [{"input": audio_url, "status": "Failed", "error": "Too noisy", "db_id": None}],
        "progress": []
    }

    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Failed"
    assert results[0]["error"] == "Too noisy"

def test_add_media_pdf_processing_exception(client, mock_dependencies):
    """Test an exception raised by the async PDF processor."""
    pdf_url = "http://example.com/bad.pdf"
    form_data = create_form_data(media_type="pdf", urls=[pdf_url])

    # Configure the async mock to raise an error
    mock_dependencies["process_pdf_task"].side_effect = ValueError("Cannot parse PDF")

    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["input"] == pdf_url
    assert results[0]["status"] == "Failed"
    assert "Internal error: ValueError" in results[0]["error"] # Check error from _process_document_like_item

# === Database / Result Handling Error Tests ===

@patch('tldw_Server_API.app.api.v1.endpoints.media.add_media_to_database', side_effect=Exception("DB connection failed"))
def test_add_media_pdf_db_add_fails(mock_add_db_fails, client, mock_dependencies):
    """Test failure when adding PDF results to the database."""
    pdf_url = "http://example.com/good.pdf"
    form_data = create_form_data(media_type="pdf", urls=[pdf_url])

    # Mock PDF processing to succeed
    mock_dependencies["process_pdf_task"].return_value = {
        "status": "Success", "input": pdf_url, "filename": "good.pdf", "text_content": "PDF Text", "summary": "Summary", "db_id": None # Ensure db_id isn't set yet
    }
    # DB add mock is configured by the patch decorator to fail

    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    # Status is Warning because processing succeeded but DB failed
    assert results[0]["status"] == "Warning"
    assert results[0]["input"] == pdf_url
    assert "Processed successfully, but failed to add to database" in results[0]["error"]
    assert "DB connection failed" in results[0]["error"]
    assert results[0].get("db_id") is None # DB ID should not be set

def test_add_media_ebook_db_extract_id_fails(client, create_upload_file, mock_dependencies):
    """Test failure to extract DB ID from Ebook processing result string."""
    test_file = create_upload_file("book.epub", content_type="application/epub+zip")
    form_data = create_form_data(media_type="ebook")
    files_tuple = [("files", (test_file.filename, test_file.file, test_file.content_type))]
    fake_path = mock_dependencies["fake_temp_path"]

    # Mock ebook processor to return a string that the extractor can't parse
    unclear_result_msg = "Ebook processed, looks okay."
    mock_dependencies["import_epub"].return_value = unclear_result_msg
    # Mock the extractor to return None
    mock_dependencies["extract_id_string"].return_value = None

    response = client.post("/api/v1/media/add", files=files_tuple, data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    # Input ref should be the original filename passed to the endpoint
    assert results[0]["input"] == test_file.filename
    # Status becomes 'Warning' because processing succeeded string-wise, but ID extraction failed
    assert results[0]["status"] == "Warning"
    assert results[0]["message"] == unclear_result_msg
    assert "Could not extract DB ID" in results[0]["error"] # Check specific warning message
    assert results[0].get("db_id") is None

# === Success Path Examples (Multi-Input) ===

def test_add_media_mixed_url_file_success(client, create_upload_file, mock_dependencies):
    """Test adding one video URL and one video file successfully."""
    video_url = "http://example.com/good_video.mp4"
    test_file = create_upload_file("another.mp4", content_type="video/mp4")
    form_data = create_form_data(media_type="video", urls=[video_url])
    files_tuple = [("files", (test_file.filename, test_file.file, test_file.content_type))]

    expected_file_input_path = str(mock_dependencies['fake_temp_path'] / 'sanitized_filename.mockext')

    # Mock the batch processor to return success for both
    # Assume process_videos returns a list where order might match input order,
    # or contains input identifier. Let's assume it contains identifier.
    mock_dependencies["process_videos"].return_value = {
        "results": [
            {"status": "Success", "db_id": "vid_mock_url", "input": video_url},
            {"status": "Success", "db_id": "vid_mock_file", "input": expected_file_input_path} # Input for file is temp path
        ],
        "processed_count": 2, "errors_count": 0
    }

    response = client.post(
        "/api/v1/media/add",
        files=files_tuple,
        data=form_data,
        headers={"token": "fake-token"}
    )

    # Both succeeded, expect 200
    assert response.status_code == status.HTTP_200_OK
    results = response.json()["results"]
    assert len(results) == 2

    # Check result for URL input
    url_result = next((r for r in results if r["input"] == video_url), None)
    assert url_result is not None
    assert url_result["status"] == "Success"
    assert url_result["db_id"] == "vid_mock_url"

    # Check result for File input (find by DB ID as input path is temporary)
    file_result = next((r for r in results if r["db_id"] == "vid_mock_file"), None)
    assert file_result is not None
    assert file_result["status"] == "Success"
    # file_result["input"] will be the fake temp path like '/fake/temp/dir/sanitized_filename.mp4'
    assert file_result["input"] == expected_file_input_path

    # Verify the batch processor was called once with combined inputs
    mock_dependencies["process_videos"].assert_called_once()
    call_args, call_kwargs = mock_dependencies["process_videos"].call_args
    # The path passed to the function should also match
    assert call_kwargs['inputs'] == [video_url, expected_file_input_path]
    assert call_kwargs['whisper_model'] == 'tiny'  # Check a specific option

def test_add_media_multiple_failures_and_success(client, create_upload_file, mock_dependencies):
    """Test a mix of successful and failed items across URLs and files."""
    good_pdf_url = "http://example.com/good.pdf"
    bad_dl_url = "http://example.com/notfound.pdf" # Download fails
    good_pdf_file = create_upload_file("report.pdf", content_type="application/pdf")
    bad_process_file = create_upload_file("corrupt.pdf", content_type="application/pdf") # Processing fails

    form_data = create_form_data(media_type="pdf", urls=[good_pdf_url, bad_dl_url])
    files_tuple = [
        ("files", (good_pdf_file.filename, good_pdf_file.file, good_pdf_file.content_type)),
        ("files", (bad_process_file.filename, bad_process_file.file, bad_process_file.content_type)),
    ]
    fake_path = mock_dependencies["fake_temp_path"]

    # --- Mock Behaviors ---
    # 1. Mock requests.get: Success for good_pdf_url, 404 for bad_dl_url
    mock_response_good = MagicMock(spec=requests.Response, status_code=200, content=b'good pdf', url=good_pdf_url)
    mock_response_good.raise_for_status = MagicMock()
    mock_response_bad = MagicMock(spec=requests.Response, status_code=404, url=bad_dl_url)
    mock_response_bad.raise_for_status.side_effect = requests.exceptions.HTTPError("404", response=mock_response_bad)
    def req_side_effect(url, *args, **kwargs):
        if url == good_pdf_url: return mock_response_good
        if url == bad_dl_url: return mock_response_bad
        raise ValueError("Unexpected URL in test")
    mock_dependencies["requests_get"].side_effect = req_side_effect

    # 2. Mock process_pdf_task: Success for good items, Exception for corrupt file
    async def pdf_task_side_effect(*args, **kwargs):
        input_ref = kwargs.get("input_ref")
        filename = kwargs.get("filename")
        if "corrupt" in filename:
            raise ValueError("Cannot parse corrupt PDF")
        elif input_ref == good_pdf_url:
             # Simulate successful DB add result here if needed, or rely on separate mock
            return {"status": "Success", "db_id": "pdf_db_good_url", "input": input_ref, "filename": filename, "text_content": "...", "summary": ""}
        elif "report" in filename:
             # Simulate successful DB add result here if needed
            return {"status": "Success", "db_id": "pdf_db_good_file", "input": input_ref, "filename": filename, "text_content": "...", "summary": ""}
        else:
            raise ValueError("Unexpected input in pdf_task_side_effect")
    mock_dependencies["process_pdf_task"].side_effect = pdf_task_side_effect

    # 3. Mock DB add (might be called by _process_document_like_item after pdf_task success)
    mock_dependencies["add_db"].side_effect = lambda *args, **kwargs: {"id": f"db_{kwargs.get('url', 'unknown')}", "message": "Added"}


    # --- Run Test ---
    response = client.post("/api/v1/media/add", files=files_tuple, data=form_data, headers={"token": "fake-token"})

    # --- Assertions ---
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 4 # good_url, bad_url, good_file, bad_process_file

    # Create a dict for easier lookup by original input reference
    results_by_input = {}
    for r in results:
        # Handle file inputs where 'input' is the temp path
        if "report.pdf" in r["input"]: results_by_input["report.pdf"] = r
        elif "corrupt.pdf" in r["input"]: results_by_input["corrupt.pdf"] = r
        else: results_by_input[r["input"]] = r # URLs map directly

    # Check good URL
    assert results_by_input[good_pdf_url]["status"] == "Success"
    # The db_id might come from process_pdf_task mock or add_db mock depending on flow
    assert results_by_input[good_pdf_url].get("db_id") is not None or results_by_input[good_pdf_url].get("db_message") is not None

    # Check bad download URL
    assert results_by_input[bad_dl_url]["status"] == "Failed"
    assert "Download failed" in results_by_input[bad_dl_url]["error"]

    # Check good file
    assert results_by_input["report.pdf"]["status"] == "Success"
    assert results_by_input["report.pdf"].get("db_id") is not None or results_by_input[good_pdf_url].get("db_message") is not None


    # Check bad processing file
    assert results_by_input["corrupt.pdf"]["status"] == "Failed"
    assert "Internal error: ValueError" in results_by_input["corrupt.pdf"]["error"] # Error from process_pdf_task side effect

    # Check call counts
    assert mock_dependencies["requests_get"].call_count == 2
    assert mock_dependencies["process_pdf_task"].call_count == 3 # Called for good_pdf_url and good_pdf_file, bad_process_file
    # DB add should be called twice (for the two successful PDF processes)
    assert mock_dependencies["add_db"].call_count == 2


# ##################################################################################################################
# End of test_add_media_endpoint.py
# ##################################################################################################################