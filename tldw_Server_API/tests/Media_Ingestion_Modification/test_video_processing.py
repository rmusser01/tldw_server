# test_video_processing.py
# # # Description: This file contains unit tests for the video processing endpoint of the tldw application.
#
# Imports
from typing import Any, AsyncGenerator, Dict

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
try:
    # Adjust the import path based on your project structure
    from tldw_Server_API.app.main import app as fastapi_app_instance
except ImportError as e:
    pytest.fail(f"Failed to import FastAPI app instance from tldw_Server_API.app.main: {e}")
except AttributeError:
     pytest.fail(f"Could not find an object named 'app' in tldw_Server_API.app.main. Make sure your FastAPI instance is named 'app'.")

#
#
# Define placeholder structures for mocks
MOCK_USER_INFO = {"user_id": "test_user"}
MOCK_SETTINGS = {"DEFAULT_TRANSCRIPTION_MODEL": "mock-model"}
# Define path for authentication mock - ADJUST IF verify_token is imported differently in media.py
#AUTH_MOCK_PATH = "tldw_Server_API.app.api.v1.endpoints.media.verify_token"
# --- Fixtures ---

@pytest.fixture(scope="module")
def event_loop():
    """Overrides pytest default function scoped event loop"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="module")
async def client() -> AsyncGenerator[AsyncClient, Any]:
    """Provides an AsyncClient for testing the app."""
    # print("\n--- Registered Routes ---") # Keep this for debugging if needed
    # for route in fastapi_app_instance.routes:
    #      if hasattr(route, "path"):
    #          print(f"Path: {route.path}, Methods: {getattr(route, 'methods', 'N/A')}, Name: {getattr(route, 'name', 'N/A')}")
    # print("-----------------------\n")
    async with AsyncClient(app=fastapi_app_instance, base_url="http://test") as c:
        yield c

# --- Helper Function ---
def clean_form_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Removes keys with None values from a dictionary."""
    return {k: v for k, v in data.items() if v is not None}


# --- Test Cases ---

@pytest.mark.asyncio
async def test_process_videos_success_url(client: AsyncClient, mocker):
    """Test successful processing with a single URL input."""

    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    mock_saved_files = [] # No files uploaded
    mock_file_errors = [] # No file errors

    mock_process_videos_result = { # Mock result from the library function
        "processed_count": 1, "errors_count": 0, "errors": [],
        "results": [{
            "status": "Success", "input_ref": test_url, "processing_source": test_url,
            "media_type": "video", "metadata": {"title": "Test Video"},
            "content": "This is the transcript.", "segments": [{"text": "This is the transcript."}],
            "chunks": None, "analysis": "This is the analysis.",
            "analysis_details": {"whisper_model": "mock-model"}, "error": None, "warnings": None,
            "db_id": None, "db_message": None, "message": None,
        }], "confabulation_results": None
    }

    # --- Mocks ---
    # Uncomment the next line if your endpoint uses authentication
    # mocker.patch(AUTH_MOCK_PATH, return_value=MOCK_USER_INFO)
    mocker.patch(
        "tldw_Server_API.app.api.v1.endpoints.media._save_uploaded_files", # lowercase media
        new_callable=AsyncMock, # Use AsyncMock since the function is async
        return_value=(mock_saved_files, mock_file_errors)
    )
    mock_lib_process = mocker.patch(
        "tldw_Server_API.app.api.v1.endpoints.media.process_videos", # This assumes process_videos is imported in media.py
        return_value=mock_process_videos_result
    )

    # --- Prepare Raw Form Data (with Nones and string representations) ---
    raw_form_data = {
        "urls": [test_url], # List is fine for httpx with `data=`
        "keywords": "test",
        "transcription_model": "provided-model",
        "perform_analysis": "true", # Send as string
        "diarize": "false",
        "timestamp_option": "true",
        "vad_use": "false",
        "perform_confabulation_check_of_analysis": "false",
        "perform_chunking": "false",
        "use_adaptive_chunking": "false",
        "use_multi_level_chunking": "false",
        "summarize_recursively": "false",
        "overwrite_existing": "false",
        "use_cookies": "false",
        "chunk_size": "500", # Send as string
        "chunk_overlap": "200",
        "title": None, # Will be removed by clean_form_data
        "author": None,
        "custom_prompt": None,
        "system_prompt": None,
        "start_time": None,
        "end_time": None,
        "api_name": None,
        "api_key": None,
        "cookies": None,
        "pdf_parsing_engine": "pymupdf4llm", # Default from endpoint
        "chunk_method": None, # Will be removed by clean_form_data
        "chunk_language": None,
        "custom_chapter_pattern": None,
        "perform_rolling_summarization": "false",
    }

    # --- Clean the data for sending (remove None values) ---
    form_data_to_send = clean_form_data(raw_form_data)

    # --- Make the request ---
    response = await client.post("/api/v1/media/process-videos", data=form_data_to_send)

    # --- Optional: Debugging ---
    if response.status_code != status.HTTP_200_OK:
        print(f"\n--- DEBUG: test_process_videos_success_url Failed ({response.status_code}) ---")
        print("Data Sent:", form_data_to_send)
        try:
            print("Response JSON:", response.json())
        except:
            print("Response Text:", response.text)
        print("--- END DEBUG ---\n")

    # --- Assertions ---
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
    assert call_args["perform_chunking"] is False # Should be coerced back to bool


@pytest.mark.asyncio
async def test_process_videos_success_file(client: AsyncClient, mocker):
    """Test successful processing with a single file upload."""
    # --- Mocks & Setup ---
    test_filename = "test_video.mp4"
    mock_file_content = b"dummy video content"
    mock_temp_dir = Path("/fake/temp/dir")
    mock_saved_path = mock_temp_dir / test_filename
    mock_saved_files = [{"path": mock_saved_path, "original_filename": test_filename, "input_ref": test_filename}]
    mock_file_errors = []
    mock_process_videos_result = { # Result from library function
        "processed_count": 1, "errors_count": 0, "errors": [],
        "results": [{
            "status": "Success", "input_ref": test_filename, "processing_source": str(mock_saved_path),
            "media_type": "video", "metadata": {"title": "test_video.mp4"},
            "content": "File transcript.", "analysis": "File analysis.", "error": None,
             "segments": [], "chunks": None, "analysis_details": {}, "warnings": None,
             "db_id": None, "db_message": None, "message": None,
        }],
        "confabulation_results": None
    }
    # mocker.patch(AUTH_MOCK_PATH, return_value=MOCK_USER_INFO) # Uncomment if auth is active
    mock_save = mocker.patch(
        "tldw_Server_API.app.api.v1.endpoints.media._save_uploaded_files", # lowercase media
        new_callable=AsyncMock,
        return_value=(mock_saved_files, mock_file_errors)
    )
    mock_lib_process = mocker.patch(
        "tldw_Server_API.app.api.v1.endpoints.media.process_videos",
        return_value=mock_process_videos_result
    )
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.TempDirManager.__enter__", return_value=mock_temp_dir)
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.TempDirManager.__exit__")
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.shutil.rmtree") # Patch shutil import used in endpoint

    # --- Prepare Raw Form Data ---
    raw_form_data = {
        "keywords": "test",
        "transcription_model": "mock-model",
        "perform_analysis": "true", # String boolean
        "diarize": "false",
        "timestamp_option": "true",
        "vad_use": "false",
        "perform_confabulation_check_of_analysis": "false",
        "perform_chunking": "false",
        "chunk_size": "500", # String integer
        "chunk_overlap": "200",
        "use_adaptive_chunking": "false",
        "use_multi_level_chunking": "false",
        "summarize_recursively": "false",
        "overwrite_existing": "false",
        "use_cookies": "false",
        "title": None, # Remove
        "author": None,
        "custom_prompt": None,
        "system_prompt": None,
        "start_time": None,
        "end_time": None,
        "api_name": None,
        "api_key": None,
        "cookies": None,
        "pdf_parsing_engine": "pymupdf4llm",
        "chunk_method": None, # Remove
        "chunk_language": None,
        "custom_chapter_pattern": None,
        "perform_rolling_summarization": "false",
        # 'urls' is implicitly None/missing here
    }
    # --- Clean data and prepare files ---
    form_data_to_send = clean_form_data(raw_form_data)
    files_to_send = {"files": (test_filename, mock_file_content, "video/mp4")}

    # --- Make the request ---
    response = await client.post("/api/v1/media/process-videos", data=form_data_to_send, files=files_to_send)

    # --- Debugging ---
    if response.status_code != status.HTTP_200_OK:
        print(f"\n--- DEBUG: test_process_videos_success_file Failed ({response.status_code}) ---")
        print("Data Sent:", form_data_to_send)
        print("Files Sent:", files_to_send)
        try:
            print("Response JSON:", response.json())
        except:
            print("Response Text:", response.text)
        print("--- END DEBUG ---\n")

    # --- Assertions ---
    assert response.status_code == status.HTTP_200_OK
    json_response = response.json()
    assert json_response["processed_count"] == 1
    assert json_response["errors_count"] == 0
    # ... rest of assertions ...
    mock_lib_process.assert_called_once()
    call_args = mock_lib_process.call_args[1]
    assert call_args["inputs"] == [str(mock_saved_path)]


@pytest.mark.asyncio
async def test_process_videos_success_file(client: AsyncClient, mocker):
    """Test successful processing with a single file upload."""
    # --- Mocks & Setup ---
    test_filename = "test_video.mp4"
    mock_file_content = b"dummy video content"
    mock_temp_dir = Path("/fake/temp/dir")
    mock_saved_path = mock_temp_dir / test_filename
    mock_saved_files = [{"path": mock_saved_path, "original_filename": test_filename, "input_ref": test_filename}]
    mock_file_errors = []
    mock_process_videos_result = { # Result from library function
        "processed_count": 1, "errors_count": 0, "errors": [],
        "results": [{
            "status": "Success", "input_ref": test_filename, "processing_source": str(mock_saved_path),
            "media_type": "video", "metadata": {"title": "test_video.mp4"},
            "content": "File transcript.", "analysis": "File analysis.", "error": None,
             "segments": [], "chunks": None, "analysis_details": {}, "warnings": None,
             "db_id": None, "db_message": None, "message": None,
        }],
        "confabulation_results": None
    }
    # mocker.patch(AUTH_MOCK_PATH, return_value=MOCK_USER_INFO) # Uncomment if auth is active
    mock_save = mocker.patch(
        "tldw_Server_API.app.api.v1.endpoints.media._save_uploaded_files", # lowercase media
        new_callable=AsyncMock,
        return_value=(mock_saved_files, mock_file_errors)
    )
    mock_lib_process = mocker.patch(
        "tldw_Server_API.app.api.v1.endpoints.media.process_videos",
        return_value=mock_process_videos_result
    )
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.TempDirManager.__enter__", return_value=mock_temp_dir)
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.TempDirManager.__exit__")
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.shutil.rmtree") # Patch shutil import used in endpoint

    # --- Prepare Raw Form Data ---
    raw_form_data = {
        "keywords": "test",
        "transcription_model": "mock-model",
        "perform_analysis": "true", # String boolean
        "diarize": "false",
        "timestamp_option": "true",
        "vad_use": "false",
        "perform_confabulation_check_of_analysis": "false",
        "perform_chunking": "false",
        "chunk_size": "500", # String integer
        "chunk_overlap": "200",
        "use_adaptive_chunking": "false",
        "use_multi_level_chunking": "false",
        "summarize_recursively": "false",
        "overwrite_existing": "false",
        "use_cookies": "false",
        "title": None, # Remove
        "author": None,
        "custom_prompt": None,
        "system_prompt": None,
        "start_time": None,
        "end_time": None,
        "api_name": None,
        "api_key": None,
        "cookies": None,
        "pdf_parsing_engine": "pymupdf4llm",
        "chunk_method": None, # Remove
        "chunk_language": None,
        "custom_chapter_pattern": None,
        "perform_rolling_summarization": "false",
        # 'urls' is implicitly None/missing here
    }
    # --- Clean data and prepare files ---
    form_data_to_send = clean_form_data(raw_form_data)
    files_to_send = {"files": (test_filename, mock_file_content, "video/mp4")}

    # --- Make the request ---
    response = await client.post("/api/v1/media/process-videos", data=form_data_to_send, files=files_to_send)

    # --- Debugging ---
    if response.status_code != status.HTTP_200_OK:
        print(f"\n--- DEBUG: test_process_videos_success_file Failed ({response.status_code}) ---")
        print("Data Sent:", form_data_to_send)
        print("Files Sent:", files_to_send)
        try:
            print("Response JSON:", response.json())
        except:
            print("Response Text:", response.text)
        print("--- END DEBUG ---\n")

    # --- Assertions ---
    assert response.status_code == status.HTTP_200_OK
    json_response = response.json()
    assert json_response["processed_count"] == 1
    assert json_response["errors_count"] == 0
    # ... rest of assertions ...
    mock_lib_process.assert_called_once()
    call_args = mock_lib_process.call_args[1]
    assert call_args["inputs"] == [str(mock_saved_path)]


@pytest.mark.asyncio
async def test_process_videos_file_save_error(client: AsyncClient, mocker):
    """Test when saving an uploaded file fails."""
    # --- Mocks & Setup ---
    test_filename = "bad_upload.mp4"
    mock_file_content = b"more dummy content"
    mock_saved_files = [] # No files successfully saved
    mock_file_errors = [{"input": test_filename, "status": "Failed", "error": "Disk full simulation"}]
    mock_lib_process = mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.process_videos") # Mock library process (shouldn't be called)
    # mocker.patch(AUTH_MOCK_PATH, return_value=MOCK_USER_INFO) # Uncomment if auth is active
    mocker.patch(
        "tldw_Server_API.app.api.v1.endpoints.media._save_uploaded_files", # lowercase media
        new_callable=AsyncMock,
        return_value=(mock_saved_files, mock_file_errors) # Return the error
    )
    # Mock TempDirManager and shutil
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.TempDirManager.__enter__", return_value=Path("/fake/temp"))
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.TempDirManager.__exit__")
    mocker.patch("tldw_Server_API.app.api.v1.endpoints.media.shutil.rmtree")

    # --- Prepare Raw Form Data ---
    raw_form_data = {
        "keywords": "test",
        "transcription_model": "mock-model",
        "perform_analysis": "true", # String boolean
        "diarize": "false",
        "timestamp_option": "true",
        "vad_use": "false",
        "perform_confabulation_check_of_analysis": "false",
        "perform_chunking": "false",
        "chunk_size": "500", # String integer
        "chunk_overlap": "200",
        "use_adaptive_chunking": "false",
        "use_multi_level_chunking": "false",
        "summarize_recursively": "false",
        "overwrite_existing": "false",
        "use_cookies": "false",
        "title": None, # Remove
        "author": None,
        "custom_prompt": None,
        "system_prompt": None,
        "start_time": None,
        "end_time": None,
        "api_name": None,
        "api_key": None,
        "cookies": None,
        "pdf_parsing_engine": "pymupdf4llm",
        "chunk_method": None, # Remove
        "chunk_language": None,
        "custom_chapter_pattern": None,
        "perform_rolling_summarization": "false",
    }
    form_data_to_send = clean_form_data(raw_form_data)
    files_to_send = {"files": (test_filename, mock_file_content, "video/mp4")}

    # --- Make the request ---
    response = await client.post("/api/v1/media/process-videos", data=form_data_to_send, files=files_to_send)

    # --- Debugging ---
    if response.status_code != status.HTTP_207_MULTI_STATUS:
         print(f"\n--- DEBUG: test_process_videos_file_save_error Failed ({response.status_code}) ---")
         print("Data Sent:", form_data_to_send)
         print("Files Sent:", files_to_send)
         try:
             print("Response JSON:", response.json())
         except:
             print("Response Text:", response.text)
         print("--- END DEBUG ---\n")

    # --- Assertions ---
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    json_response = response.json()
    assert json_response["errors_count"] == 1
    assert len(json_response["results"]) == 1 # Contains the formatted file error
    assert json_response["results"][0]["status"] == "Failed" # Expect the status from the mock error
    # --- End update ---
    assert json_response["results"][0]["input_ref"] == test_filename
    assert json_response["results"][0]["error"] == "Disk full simulation"
    assert "Disk full simulation" in json_response["errors"]

    # Ensure the library processing function was NOT called because saving failed
    mock_lib_process.assert_not_called()

@pytest.mark.asyncio
async def test_process_videos_no_input(client: AsyncClient, mocker):
    """Test providing no URLs and no files."""
    # --- Mocks ---
    # mocker.patch(AUTH_MOCK_PATH, return_value=MOCK_USER_INFO) # Uncomment if auth is active
    # No need to mock _save_uploaded_files or process_videos as internal validation should fail

    # --- Prepare Raw Form Data (no urls/files) ---
    raw_form_data = {
        "keywords": "test",
        "transcription_model": "mock-model",
        "perform_analysis": "true", # String boolean
        "diarize": "false",
        "timestamp_option": "true",
        "vad_use": "false",
        "perform_confabulation_check_of_analysis": "false",
        "perform_chunking": "false",
        "chunk_size": "500", # String integer
        "chunk_overlap": "200",
        "use_adaptive_chunking": "false",
        "use_multi_level_chunking": "false",
        "summarize_recursively": "false",
        "overwrite_existing": "false",
        "use_cookies": "false",
        "title": None, # Remove
        "author": None,
        "custom_prompt": None,
        "system_prompt": None,
        "start_time": None,
        "end_time": None,
        "api_name": None,
        "api_key": None,
        "cookies": None,
        "pdf_parsing_engine": "pymupdf4llm",
        "chunk_method": None, # Remove
        "chunk_language": None,
        "custom_chapter_pattern": None,
        "perform_rolling_summarization": "false",
        # --- 'urls' field is missing ---
    }
    # --- Clean data ---
    form_data_to_send = clean_form_data(raw_form_data) # 'urls' is already absent

    # --- Make the request (no files attached) ---
    response = await client.post("/api/v1/media/process-videos", data=form_data_to_send)

    # --- Debugging ---
    # The internal `_validate_inputs` check (after dependency validation) should raise 400
    if response.status_code != status.HTTP_400_BAD_REQUEST:
         print(f"\n--- DEBUG: test_process_videos_no_input Failed ({response.status_code}) ---")
         print("Data Sent:", form_data_to_send)
         try:
             print("Response JSON:", response.json())
         except:
             print("Response Text:", response.text)
         print("--- END DEBUG ---\n")

    # --- Assertions ---
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    # Check the specific error message from your _validate_inputs function
    # --- Update this line ---
    expected_error_detail = "At least one 'url' in the 'urls' list or one 'file' in the 'files' list must be provided."
    # --- End update ---
    # Check if the exact detail message is present in the response JSON or text
    response_json = response.json()
    assert "detail" in response_json
    assert response_json["detail"] == expected_error_detail



# @pytest.mark.asyncio
# async def test_process_videos_auth_failure(client: AsyncClient, mocker):
#     """Test when authentication fails."""
#     # --- Mocks ---
#     # IMPORTANT: Ensure AUTH_MOCK_PATH is correct AND auth dependency is active in endpoint
#     # Mock verify_token to raise the auth exception
#     mock_verify = mocker.patch(AUTH_MOCK_PATH)
#     mock_verify.side_effect = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
#
#     # --- Prepare Minimal Form Data ---
#     # Data doesn't need to be fully valid if auth fails first, but must be present.
#     # Sending an empty dict might bypass form parsing, so send something.
#     raw_form_data = {
#         "urls": ["http://example.com"], # Provide one required input source
#         "keywords": "auth_fail_test",   # Provide a required string field
#         # Other fields can be omitted as validation might not be reached
#     }
#     form_data_to_send = clean_form_data(raw_form_data) # Only urls and keywords remain
#
#     # --- Make the request ---
#     response = await client.post("/api/v1/media/process-videos", data=form_data_to_send)
#
#     # --- Debugging ---
#     if response.status_code != status.HTTP_401_UNAUTHORIZED:
#         print(f"\n--- DEBUG: test_process_videos_auth_failure Failed ({response.status_code}) ---")
#         print("Data Sent:", form_data_to_send)
#         try:
#             print("Response JSON:", response.json())
#         except:
#             print("Response Text:", response.text)
#         print("--- END DEBUG ---\n")
#
#
#     # --- Assertions ---
#     assert response.status_code == status.HTTP_401_UNAUTHORIZED
#     # Ensure the detail message matches what the mock raises
#     assert "Invalid token" in response.text