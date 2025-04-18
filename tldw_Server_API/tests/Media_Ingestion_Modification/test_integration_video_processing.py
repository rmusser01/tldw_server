# tldw_Server_API/tests/Media_Ingestion_Modification/test_integration_video_processing.py

import pytest
import pytest_asyncio
import asyncio
from httpx import AsyncClient
from fastapi import FastAPI, status, UploadFile, BackgroundTasks, HTTPException
from unittest.mock import AsyncMock, MagicMock, patch, ANY  # Added ANY
from pathlib import Path
import io
import shutil  # Needed for potential rmtree patching/verification
import tempfile  # For checking temp dir existence
from typing import Any, AsyncGenerator, Dict, List

# --- Test Fixtures and Helpers ---
# Reuse from unit tests if they are in a common conftest.py or shared module
# Otherwise, redefine them here.

# Assuming unit test file structure for imports:
try:
    from tldw_Server_API.app.main import app as fastapi_app_instance
    # We might need ProcessVideosForm if we want to validate mock call arguments deeply
    from tldw_Server_API.app.api.v1.schemas.media_models import ProcessVideosForm
except ImportError as e:
    pytest.fail(f"Failed to import FastAPI app instance or schemas: {e}")
except AttributeError:
    pytest.fail(f"Could not find 'app' in tldw_Server_API.app.main.")

# Mocks placeholders
MOCK_USER_INFO = {"user_id": "integration_test_user"}
AUTH_MOCK_PATH = "tldw_Server_API.app.api.v1.endpoints.media.verify_token"  # Adjust if needed


# Fixtures
@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="module")
async def client() -> AsyncGenerator[AsyncClient, Any]:
    # Use the real app instance for integration tests
    async with AsyncClient(app=fastapi_app_instance, base_url="http://test") as c:
        yield c


# Helper to clean form data (remove None values)
def clean_form_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Removes keys with None values from a dictionary."""
    return {k: v for k, v in data.items() if v is not None}


# --- Integration Test Cases ---

@pytest.mark.integration  # Mark as integration test
@pytest.mark.asyncio
async def test_process_videos_integration_url_success(client: AsyncClient, mocker):
    """
    Integration Test: Process a video URL successfully.
    - Mocks `process_videos` to avoid actual heavy processing.
    - Allows `TempDirManager` to run (but it shouldn't be used for URLs).
    - Verifies `process_videos` is called with correct URL input.
    """
    test_url = "https://example.com/test_video.mp4"

    # Mock result expected FROM process_videos library function
    mock_lib_result_data = {
        "processed_count": 1, "errors_count": 0, "errors": [],
        "results": [{
            "status": "Success", "input_ref": test_url, "processing_source": test_url,
            "media_type": "video", "metadata": {"title": "Mocked URL Video"},
            "content": "Mock transcript for URL.", "segments": [],
            "chunks": None, "analysis": "Mock analysis for URL.",
            "analysis_details": {"model": "mock"}, "error": None, "warnings": None,
            # db_id/db_message should be added by endpoint if missing
        }], "confabulation_results": None
    }

    # --- Mocks ---
    # Mock authentication if needed
    # mocker.patch(AUTH_MOCK_PATH, return_value=MOCK_USER_INFO)

    # Mock the core processing function
    mock_lib_process = mocker.patch(
        "tldw_Server_API.app.api.v1.endpoints.media.process_videos",
        return_value=mock_lib_result_data
    )
    # Spy on _save_uploaded_files to ensure it's NOT called for URLs
    mock_save_files = mocker.patch(
        "tldw_Server_API.app.api.v1.endpoints.media._save_uploaded_files",
        new_callable=AsyncMock,  # Must be AsyncMock if original is async
        return_value=([], [])  # Return empty success
    )

    # --- Prepare Form Data ---
    raw_form_data = {
        "urls": [test_url],
        "keywords": "integration,url",
        "transcription_model": "integration-model",
        "perform_analysis": "true",
        "diarize": "false", "timestamp_option": "true", "vad_use": "false",
        "perform_confabulation_check_of_analysis": "false", "perform_chunking": "false",
        "chunk_size": "500", "chunk_overlap": "200", "use_adaptive_chunking": "false",
        "use_multi_level_chunking": "false", "summarize_recursively": "false",
        "overwrite_existing": "false", "use_cookies": "false",
        "pdf_parsing_engine": "pymupdf4llm",
        "perform_rolling_summarization": "false",
        # Optional fields are None -> will be removed
        "title": None, "author": None, "custom_prompt": None, "system_prompt": None,
        "start_time": None, "end_time": None, "api_name": None, "api_key": None,
        "cookies": None, "chunk_method": None, "chunk_language": None,
        "custom_chapter_pattern": None,
    }
    form_data_to_send = clean_form_data(raw_form_data)

    # --- Make Request ---
    response = await client.post("/api/v1/media/process-videos", data=form_data_to_send)

    # --- Assertions ---
    assert response.status_code == status.HTTP_200_OK
    json_response = response.json()

    # Check overall structure
    assert json_response["processed_count"] == 1
    assert json_response["errors_count"] == 0
    assert len(json_response["results"]) == 1

    # Check specific result (should match mock, with potentially added null db fields)
    expected_result = mock_lib_result_data["results"][0].copy()
    expected_result.setdefault("db_id", None)
    expected_result.setdefault("db_message", None)
    assert json_response["results"][0] == expected_result

    # Verify mocks
    mock_lib_process.assert_called_once()
    call_args = mock_lib_process.call_args[1]  # Keyword args
    assert call_args["inputs"] == [test_url]
    assert call_args["transcription_model"] == "integration-model"

    # Verify _save_uploaded_files WAS called, but with an empty list for files
    mock_save_files.assert_called_once()
    call_args_save = mock_save_files.call_args[0]  # Positional args
    assert call_args_save[0] == []  # Check the first argument (files) was empty list
    assert isinstance(call_args_save[1], Path)  # Check the second argument was a Path (temp_dir)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_process_videos_integration_file_success(client: AsyncClient, mocker):
    """
    Integration Test: Process a file upload successfully.
    - Allows `TempDirManager` to create a real temp directory.
    - Allows `_save_uploaded_files` to run (partially mocked if needed, or fully).
    - Mocks `process_videos` to avoid actual heavy processing.
    - Verifies `process_videos` is called with a temp file path.
    - Verifies temp directory cleanup is triggered.
    """
    test_filename = "integration_test.mp4"
    test_file_content = b"this is minimal fake video content"
    file_content_stream = io.BytesIO(test_file_content)

    # --- Mocks ---
    # Mock authentication if needed
    # mocker.patch(AUTH_MOCK_PATH, return_value=MOCK_USER_INFO)

    # We need _save_uploaded_files to return info about the saved file
    # Let's allow the real TempDirManager to run, but mock the actual file write
    # inside _save_uploaded_files to avoid disk I/O, while still returning
    # the expected path structure. This requires knowing _save_uploaded_files internals.
    # Alternative: Let _save_uploaded_files run fully (simpler but touches disk).

    # --- Let's try the simpler approach: Mock process_videos, let others run ---
    # Mock process_videos - it will receive the *real* temporary path
    mock_lib_result_data = {
        "processed_count": 1, "errors_count": 0, "errors": [],
        "results": [{
            "status": "Success",
            "input_ref": test_filename,  # Original filename
            "processing_source": f"placeholder/temp/{test_filename}",
            "media_type": "video", "metadata": {"title": test_filename},
            "content": "Mock transcript for file.", "segments": [],
            "chunks": None, "analysis": "Mock analysis for file.",
            "analysis_details": {"model": "mock"}, "error": None, "warnings": None,
        }], "confabulation_results": None
    }
    mock_lib_process = mocker.patch(
        "tldw_Server_API.app.api.v1.endpoints.media.process_videos",
        return_value=mock_lib_result_data
    )

    # Spy on shutil.rmtree to verify cleanup is scheduled
    mock_rmtree = mocker.patch("shutil.rmtree")

    # --- Prepare Form Data & Files ---
    raw_form_data = {
        "keywords": "integration,file",
        "transcription_model": "integration-model",
        "perform_analysis": "true",
        "diarize": "false", "timestamp_option": "true", "vad_use": "false",
        "perform_confabulation_check_of_analysis": "false", "perform_chunking": "true",  # Test chunking enabled
        "chunk_method": "sentences",  # Provide a valid chunk method
        "chunk_size": "300", "chunk_overlap": "50", "use_adaptive_chunking": "false",
        "use_multi_level_chunking": "false", "summarize_recursively": "false",
        "overwrite_existing": "false", "use_cookies": "false",
        "pdf_parsing_engine": "pymupdf4llm",
        "perform_rolling_summarization": "false",
        # Optional fields are None -> will be removed
        "urls": None, "title": None, "author": None, "custom_prompt": None, "system_prompt": None,
        "start_time": None, "end_time": None, "api_name": None, "api_key": None,
        "cookies": None, "chunk_language": None, "custom_chapter_pattern": None,
    }
    form_data_to_send = clean_form_data(raw_form_data)
    files_to_send = {"files": (test_filename, file_content_stream, "video/mp4")}

    # --- Make Request ---
    response = await client.post("/api/v1/media/process-videos", data=form_data_to_send, files=files_to_send)

    # --- Assertions ---
    assert response.status_code == status.HTTP_200_OK
    json_response = response.json()
    assert json_response["processed_count"] == 1
    assert json_response["errors_count"] == 0
    assert len(json_response["results"]) == 1

    # Check specific result fields that DON'T depend on the temp path directly
    result = json_response["results"][0]
    assert result["status"] == "Success"
    assert result["input_ref"] == test_filename
    # We know result["processing_source"] is the placeholder, so we don't check it here.
    # assert isinstance(result["processing_source"], str) # This would still pass
    assert result["analysis"] == "Mock analysis for file."
    assert result.get("db_id") is None
    assert result.get("db_message") is None

    # --- Verify process_videos call arguments (THIS IS WHERE WE CHECK THE PATH) ---
    mock_lib_process.assert_called_once()
    call_args = mock_lib_process.call_args[1]  # Keyword arguments passed to process_videos
    assert isinstance(call_args["inputs"], list)
    assert len(call_args["inputs"]) == 1
    # Check the path passed to the mocked process_videos function
    actual_processing_path = call_args["inputs"][0]
    assert isinstance(actual_processing_path, str)
    assert tempfile.gettempdir() in actual_processing_path  # Check it's in the OS temp dir
    assert test_filename in actual_processing_path  # Check it contains the filename
    assert Path(actual_processing_path).is_absolute()  # Check it's an absolute path
    # --- End Path Verification ---
    assert call_args["perform_chunking"] is True
    assert call_args["chunk_method"] == "sentences"
    assert call_args["max_chunk_size"] == 300

    # Verify cleanup call is scheduled
    await asyncio.sleep(0.01)
    mock_rmtree.assert_called_once()
    # Check the path passed to rmtree is the parent dir of the processed file
    # Use the path we extracted from the process_videos call args
    temp_dir_path = Path(actual_processing_path).parent
    mock_rmtree.assert_called_with(temp_dir_path, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_process_videos_integration_partial_error(client: AsyncClient, mocker):
    """
    Integration Test: Process multiple inputs (URL+File) with a processing error on one.
    - Mocks `process_videos` to return mixed results.
    - Allows file saving and temp dir management to run.
    """
    test_url = "https://example.com/success.mp4"
    test_filename = "fail_video.mp4"
    test_file_content = b"minimal failing content"
    file_content_stream = io.BytesIO(test_file_content)

    # Mock process_videos to return mixed results
    # It will receive a list containing the URL and the real temporary path
    mock_lib_result_data = {
        "processed_count": 2,  # Total attempts
        "errors_count": 1,
        "errors": ["Processing failed for fail_video.mp4"],
        "results": [
            {  # Success for URL (no change needed here)
                "status": "Success", "input_ref": test_url, "processing_source": test_url,
                "media_type": "video", "metadata": {}, "content": "URL ok",
                "segments": [], "chunks": None, "analysis": "URL analysis",
                "analysis_details": {}, "error": None, "warnings": None,
            },
            {  # Error for File
                "status": "Error", "input_ref": test_filename,  # Original filename
                # --- Fix: Use a placeholder string instead of ANY ---
                "processing_source": f"placeholder/temp/{test_filename}_failed",
                # --- End Fix ---
                "media_type": "video", "metadata": {}, "content": "",
                "segments": None, "chunks": None, "analysis": None,
                "analysis_details": None, "error": "Something bad happened during processing", "warnings": None,
            }
        ], "confabulation_results": None
    }
    mock_lib_process = mocker.patch(
        "tldw_Server_API.app.api.v1.endpoints.media.process_videos",
        return_value=mock_lib_result_data
    )
    # Spy on rmtree for cleanup verification
    mock_rmtree = mocker.patch("shutil.rmtree")
    # Mock auth if needed
    # mocker.patch(AUTH_MOCK_PATH, return_value=MOCK_USER_INFO)

    # --- Prepare Form Data & Files ---
    raw_form_data = {
        "urls": [test_url],
        "keywords": "integration,partial",
        "transcription_model": "integration-model",
        "perform_analysis": "true",
        # ... other params ...
        "perform_chunking": "false",
        "chunk_size": "500", "chunk_overlap": "200",
    }
    form_data_to_send = clean_form_data(raw_form_data)
    files_to_send = {"files": (test_filename, file_content_stream, "video/mp4")}

    # --- Make Request ---
    response = await client.post("/api/v1/media/process-videos", data=form_data_to_send, files=files_to_send)

    # --- Assertions ---
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    json_response = response.json()
    assert json_response["processed_count"] == 2
    assert json_response["errors_count"] == 1
    assert len(json_response["results"]) == 2
    assert "Processing failed for fail_video.mp4" in json_response["errors"]

    # Check individual results
    assert json_response["results"][0]["status"] == "Success"
    assert json_response["results"][0]["input_ref"] == test_url
    assert json_response["results"][1]["status"] == "Error"
    assert json_response["results"][1]["input_ref"] == test_filename
    assert isinstance(json_response["results"][1]["processing_source"], str)
    # We can't easily check if it contains the tempdir because the mock didn't use it,
    # but we know it was called with the correct temp path input.
    # assert tempfile.gettempdir() in json_response["results"][1]["processing_source"] # This might fail
    assert test_filename in json_response["results"][1]["processing_source"] # Check filename presence
    assert json_response["results"][1]["error"] == "Something bad happened during processing"

    # Verify process_videos call
    mock_lib_process.assert_called_once()
    call_args = mock_lib_process.call_args[1]
    assert isinstance(call_args["inputs"], list)
    assert len(call_args["inputs"]) == 2
    assert call_args["inputs"][0] == test_url  # URL first
    assert test_filename in call_args["inputs"][1]  # Temp path for file second

    # Verify cleanup (should still happen)
    await asyncio.sleep(0.01)
    mock_rmtree.assert_called_once()
    temp_dir_path = Path(call_args["inputs"][1]).parent
    mock_rmtree.assert_called_with(temp_dir_path, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_process_videos_integration_file_save_error(client: AsyncClient, mocker):
    """
    Integration Test: Simulate a failure during file saving.
    - Mocks `_save_uploaded_files` to return an error.
    - Allows `TempDirManager` to run.
    - Verifies `process_videos` is NOT called.
    """
    test_filename = "save_fail.mp4"
    test_file_content = b"content that won't be saved"
    file_content_stream = io.BytesIO(test_file_content)

    # --- Mocks ---
    # Mock authentication if needed
    # mocker.patch(AUTH_MOCK_PATH, return_value=MOCK_USER_INFO)

    # Mock _save_uploaded_files specifically to return an error
    mock_save_files_error = [{"input": test_filename, "status": "Failed", "error": "Simulated disk full"}]
    mock_save_files = mocker.patch(
        "tldw_Server_API.app.api.v1.endpoints.media._save_uploaded_files",
        new_callable=AsyncMock,
        return_value=([], mock_save_files_error)  # No saved files, one error
    )

    # Mock process_videos (it shouldn't be called)
    mock_lib_process = mocker.patch(
        "tldw_Server_API.app.api.v1.endpoints.media.process_videos"
    )
    # Spy on rmtree
    mock_rmtree = mocker.patch("shutil.rmtree")

    # --- Prepare Form Data & Files ---
    raw_form_data = {"keywords": "integration,save-error"}
    form_data_to_send = clean_form_data(raw_form_data)
    files_to_send = {"files": (test_filename, file_content_stream, "video/mp4")}

    # --- Make Request ---
    response = await client.post("/api/v1/media/process-videos", data=form_data_to_send, files=files_to_send)

    # --- Assertions ---
    # Should be 207 because *some* operation (saving) was attempted and failed
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    json_response = response.json()
    assert json_response["processed_count"] == 0  # No items made it to processing
    assert json_response["errors_count"] == 1
    assert len(json_response["results"]) == 1  # The adapted file error result
    assert "Simulated disk full" in json_response["errors"]

    # Check the formatted error result
    result = json_response["results"][0]
    assert result["status"] == "Failed"  # Status from the mock error
    assert result["input_ref"] == test_filename
    assert result["error"] == "Simulated disk full"

    # Verify process_videos was NOT called
    mock_lib_process.assert_not_called()

    # Verify cleanup (TempDirManager should still run and trigger cleanup)
    await asyncio.sleep(0.01)
    mock_rmtree.assert_called_once()
    # Cannot easily assert the path here as no processing path was generated


@pytest.mark.integration
@pytest.mark.asyncio
async def test_process_videos_integration_no_input(client: AsyncClient, mocker):
    """
    Integration Test: Request with no URL or file.
    - Relies on internal endpoint validation (`_validate_inputs`).
    - No processing mocks needed.
    """
    # --- Mocks ---
    # Mock authentication if needed
    # mocker.patch(AUTH_MOCK_PATH, return_value=MOCK_USER_INFO)

    # --- Prepare Form Data ---
    raw_form_data = {"keywords": "integration,no-input"}  # Minimal required data
    form_data_to_send = clean_form_data(raw_form_data)

    # --- Make Request ---
    # No `urls` in data, no `files` parameter
    response = await client.post("/api/v1/media/process-videos", data=form_data_to_send)

    # --- Assertions ---
    # Internal _validate_inputs should raise 400
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    response_json = response.json()
    assert "detail" in response_json
    assert "one 'url' in the 'urls' list or one 'file'" in response_json["detail"]

# Only include this test if authentication is actually enabled/required by the endpoint
# @pytest.mark.integration
# @pytest.mark.asyncio
# async def test_process_videos_integration_auth_failure(client: AsyncClient, mocker):
#     """
#     Integration Test: Authentication failure.
#     - Mocks the authentication dependency to raise 401.
#     """
#     # --- Mocks ---
#     mock_verify = mocker.patch(AUTH_MOCK_PATH)
#     mock_verify.side_effect = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid integration token")

#     # --- Prepare Minimal Form Data ---
#     raw_form_data = {
#         "urls": ["http://example.com/any"], # Need some input to pass initial checks if auth runs late
#         "keywords": "integration,auth-fail",
#     }
#     form_data_to_send = clean_form_data(raw_form_data)

#     # --- Make Request ---
#     response = await client.post("/api/v1/media/process-videos", data=form_data_to_send)

#     # --- Assertions ---
#     assert response.status_code == status.HTTP_401_UNAUTHORIZED
#     response_json = response.json()
#     assert "detail" in response_json
#     assert response_json["detail"] == "Invalid integration token"

