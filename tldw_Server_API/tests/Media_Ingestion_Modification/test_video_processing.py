# test_video_processing.py
# # # Description: This file contains unit tests for the video processing endpoint of the tldw application.
#
# Imports
import pytest
import pytest_asyncio
import asyncio
from httpx import AsyncClient
from fastapi import FastAPI, status, UploadFile, BackgroundTasks, HTTPException
from unittest.mock import AsyncMock, MagicMock, patch # Use patch from unittest.mock
from pathlib import Path
import io
#
# Local Imports
from tldw_Server_API.app import main
#
#
# Define placeholder structures for mocks
MOCK_USER_INFO = {"user_id": "test_user"}
MOCK_SETTINGS = {"DEFAULT_TRANSCRIPTION_MODEL": "mock-model"}

# --- Fixtures ---

@pytest.fixture(scope="module")
def event_loop():
    """Overrides pytest default function scoped event loop"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="module")
async def client() -> AsyncClient:
    """Provides an AsyncClient for testing the app."""
    async with AsyncClient(app=main, base_url="http://test") as c:
        yield c

# --- Test Cases ---

@pytest.mark.asyncio
async def test_process_videos_success_url(client: AsyncClient, mocker):
    """Test successful processing with a single URL input."""

    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    mock_saved_files = [] # No files uploaded
    mock_file_errors = [] # No file errors

    # Mock return value for the library's process_videos function
    mock_process_videos_result = {
        "processed_count": 1,
        "errors_count": 0,
        "errors": [],
        "results": [
            {
                "status": "Success",
                "input_ref": test_url,
                "processing_source": test_url, # Assuming library returns URL for URL input
                "media_type": "video",
                "metadata": {"title": "Test Video"},
                "content": "This is the transcript.",
                "segments": [{"text": "This is the transcript."}],
                "chunks": None,
                "analysis": "This is the analysis.",
                "analysis_details": {"whisper_model": "mock-model"},
                "error": None,
                "warnings": None,
                "db_id": None,
                "db_message": None,
                "message": None,
            }
        ],
        "confabulation_results": None
    }

    # Patch dependencies
    # Per-user token mock
    #mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.verify_token", return_value=MOCK_USER_INFO)
    #mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.get_settings", return_value=MagicMock(**MOCK_SETTINGS))
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media._save_uploaded_files", return_value=(mock_saved_files, mock_file_errors))
    # Patch the library function that gets called by the endpoint
    mock_lib_process = mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.process_videos", return_value=mock_process_videos_result)

    # Prepare form data
    form_data = {
        "urls": [test_url],
        # Add other required form fields with default/test values
        "keywords": "test",
        "transcription_model": "provided-model", # Test overriding default
        # ... include other non-optional fields from the endpoint signature ...
        "perform_analysis": True,
        "diarize": False,
        "timestamp_option": True,
        "vad_use": False,
        "perform_confabulation_check_of_analysis": False,
        "perform_chunking": False, # Keep it simple for this test
        # ... Add other required bools/ints/etc ...
        "chunk_size": 500,
        "chunk_overlap": 200,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "summarize_recursively": False,
        "overwrite_existing": False,
        "use_cookies": False,
    }

    response = await client.post("/api/v1/media/process-videos", data=form_data)

    assert response.status_code == status.HTTP_200_OK
    json_response = response.json()
    assert json_response["processed_count"] == 1
    assert json_response["errors_count"] == 0
    assert len(json_response["results"]) == 1
    assert json_response["results"][0]["status"] == "Success"
    assert json_response["results"][0]["input_ref"] == test_url
    assert json_response["results"][0]["analysis"] == "This is the analysis."

    # Assert that the library function was called correctly
    mock_lib_process.assert_called_once()
    call_args = mock_lib_process.call_args[1] # Get keyword arguments
    assert call_args["inputs"] == [test_url]
    assert call_args["transcription_model"] == "provided-model" # Check form value was used
    assert call_args["perform_chunking"] == False


@pytest.mark.asyncio
async def test_process_videos_success_file(client: AsyncClient, mocker):
    """Test successful processing with a single file upload."""

    test_filename = "test_video.mp4"
    mock_file_content = b"dummy video content"
    mock_temp_dir = Path("/fake/temp/dir")
    mock_saved_path = mock_temp_dir / test_filename

    # Mock _save_uploaded_files behavior
    mock_saved_files = [{"path": mock_saved_path, "original_filename": test_filename, "input_ref": test_filename}]
    mock_file_errors = []

    mock_process_videos_result = {
        "processed_count": 1, "errors_count": 0, "errors": [],
        "results": [{
            "status": "Success", "input_ref": test_filename,
            "processing_source": str(mock_saved_path), # Library should use the path
            "media_type": "video", "metadata": {"title": "test_video.mp4"},
            "content": "File transcript.", "analysis": "File analysis.",
            "error": None, # Other fields omitted for brevity...
        }],
        "confabulation_results": None
    }

    #mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.verify_token", return_value=MOCK_USER_INFO)
    #mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.get_settings", return_value=MagicMock(**MOCK_SETTINGS))
    # Use AsyncMock if the patched function is async
    mock_save = mocker.patch("tldw_Server_API.app.api.v1.endpoints.media._save_uploaded_files", new_callable=AsyncMock, return_value=(mock_saved_files, mock_file_errors))
    mock_lib_process = mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.process_videos", return_value=mock_process_videos_result)
    # Mock TempDirManager to avoid actual disk I/O
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.TempDirManager.__enter__", return_value=mock_temp_dir)
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.TempDirManager.__exit__")
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.shutil.rmtree") # Mock cleanup

    # Prepare form data (minimal) and files
    form_data = {"keywords": "test", "transcription_model": "mock-model"} # Add other required fields
    files = {"files": (test_filename, mock_file_content, "video/mp4")}

    response = await client.post("/api/v1/media/process-videos", data=form_data, files=files)

    assert response.status_code == status.HTTP_200_OK
    json_response = response.json()
    assert json_response["errors_count"] == 0
    assert len(json_response["results"]) == 1
    assert json_response["results"][0]["status"] == "Success"
    assert json_response["results"][0]["input_ref"] == test_filename

    # Assert save was called (optional, depends on TempDir mock)
    # mock_save.assert_awaited_once() # Check if save was called

    # Assert library process was called with the temp path
    mock_lib_process.assert_called_once()
    call_args = mock_lib_process.call_args[1]
    assert call_args["inputs"] == [str(mock_saved_path)]

@pytest.mark.asyncio
async def test_process_videos_partial_error(client: AsyncClient, mocker):
    """Test processing with one success and one processing error."""
    url1 = "https://example.com/success.mp4"
    url2 = "https://example.com/fail.mp4"

    mock_process_videos_result = {
        "processed_count": 2, # Library might report processing attempt count
        "errors_count": 1,
        "errors": ["Processing failed for fail.mp4"],
        "results": [
            { "status": "Success", "input_ref": url1, "analysis": "Success analysis", "error": None, "media_type": "video"}, # Simplified
            { "status": "Error", "input_ref": url2, "analysis": None, "error": "Something bad happened", "media_type": "video"} # Simplified
        ],
        "confabulation_results": None
    }

    #mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.verify_token", return_value=MOCK_USER_INFO)
    #mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.get_settings", return_value=MagicMock(**MOCK_SETTINGS))
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media._save_uploaded_files", return_value=([], [])) # No files
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.process_videos", return_value=mock_process_videos_result)

    form_data = {"urls": [url1, url2], "keywords": "test", "transcription_model": "mock-model"} # Add required fields

    response = await client.post("/api/v1/media/process-videos", data=form_data)

    assert response.status_code == status.HTTP_207_MULTI_STATUS # Partial success -> 207
    json_response = response.json()
    assert json_response["errors_count"] == 1
    assert len(json_response["results"]) == 2
    assert json_response["results"][0]["status"] == "Success"
    assert json_response["results"][1]["status"] == "Error"
    assert "Processing failed for fail.mp4" in json_response["errors"]

@pytest.mark.asyncio
async def test_process_videos_file_save_error(client: AsyncClient, mocker):
    """Test when saving an uploaded file fails."""
    test_filename = "bad_upload.mp4"
    mock_file_content = b"more dummy content"

    # Mock _save_uploaded_files to return an error
    mock_saved_files = []
    mock_file_errors = [{"input": test_filename, "status": "Failed", "error": "Disk full simulation"}]

    # Mock process_videos not to be called as no valid inputs remain
    mock_lib_process = mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.process_videos")

    #mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.verify_token", return_value=MOCK_USER_INFO)
    #mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.get_settings", return_value=MagicMock(**MOCK_SETTINGS))
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media._save_uploaded_files", new_callable=AsyncMock, return_value=(mock_saved_files, mock_file_errors))
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.TempDirManager.__enter__", return_value=Path("/fake/temp"))
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.TempDirManager.__exit__")
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.shutil.rmtree")

    form_data = {"keywords": "test", "transcription_model": "mock-model"} # Add required fields
    files = {"files": (test_filename, mock_file_content, "video/mp4")}

    response = await client.post("/api/v1/media/process-videos", data=form_data, files=files)

    assert response.status_code == status.HTTP_207_MULTI_STATUS # File save errors lead to 207
    json_response = response.json()
    assert json_response["errors_count"] == 1
    assert len(json_response["results"]) == 1 # Contains the formatted file error
    assert json_response["results"][0]["status"] == "Error"
    assert json_response["results"][0]["input_ref"] == test_filename
    assert json_response["results"][0]["error"] == "Disk full simulation"
    assert "Disk full simulation" in json_response["errors"]

    # Ensure the library processing function was NOT called
    mock_lib_process.assert_not_called()

@pytest.mark.asyncio
async def test_process_videos_no_input(client: AsyncClient, mocker):
    """Test providing no URLs and no files."""
    #mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.verify_token", return_value=MOCK_USER_INFO)
    #mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.get_settings", return_value=MagicMock(**MOCK_SETTINGS))
    # _validate_inputs should raise the exception before other mocks are needed

    form_data = {"keywords": "test", "transcription_model": "mock-model"} # Add required fields

    response = await client.post("/api/v1/media/process-videos", data=form_data)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "one 'url' in the 'urls' list or one 'file'" in response.text

# @pytest.mark.asyncio
# async def test_process_videos_auth_failure(client: AsyncClient, mocker):
#     """Test when authentication fails."""
#     # Mock verify_token to raise the auth exception
#     mock_verify = mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.verify_token")
#     mock_verify.side_effect = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
#
#     form_data = {"urls": ["http://example.com"], "keywords": "test"} # Minimal data
#
#     response = await client.post("/api/v1/media/process-videos", data=form_data)
#
#     assert response.status_code == status.HTTP_401_UNAUTHORIZED
#     assert "Invalid token" in response.text

# Add more tests for:
# - Mixed URL and file inputs
# - Different chunking options being passed correctly to the mock
# - Cases where process_videos returns unexpected data format
# - Validation errors (e.g., bad time format - handled by Pydantic usually)
# - Cookie usage (asserting they are passed to process_videos if used)