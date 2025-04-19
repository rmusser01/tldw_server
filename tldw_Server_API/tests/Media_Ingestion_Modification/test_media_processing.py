# test_media_processing.py
# Description: This file contains the test cases for the media processing endpoints of the tldw application.
#
# Imports
import os
import sys
from pathlib import Path
import time
#
import pytest
from fastapi.testclient import TestClient
from loguru import logger
#
######################################################################################################################
#
# Functions:
# --- Test Setup ---

# Assuming your FastAPI app instance is created in 'app.main'
# Adjust the import path according to your project structure
try:
    from tldw_Server_API.app.main import app as fastapi_app_instance
except ImportError:
    # Handle cases where the structure might be slightly different
    # This might happen if 'app' is not directly in the root
    # You might need to adjust PYTHONPATH or the import statement
    print("Failed to import 'app' from 'app.main'. Adjust import path.")
    # As a fallback, try importing from the root if main.py is there
    try:
        from main import app
    except ImportError:
        raise ImportError("Could not locate the FastAPI app instance.")

# --- Constants for Test Files and URLs ---
TEST_MEDIA_DIR = Path(__file__).parent / "test_media"
SAMPLE_VIDEO_PATH = TEST_MEDIA_DIR / "sample.mp4"
SAMPLE_AUDIO_PATH = TEST_MEDIA_DIR / "sample.mp3"
SAMPLE_PDF_PATH = TEST_MEDIA_DIR / "sample.pdf"
INVALID_FILE_PATH = TEST_MEDIA_DIR / "not_a_real_file.xyz"

# Use stable, short, publicly accessible URLs for testing
# Replace with actual URLs known to work
VALID_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example: Rick Astley (short duration helps)
VALID_AUDIO_URL = "https://cdn.pixabay.com/download/audio/2022/11/22/audio_2f238b9a8e.mp3"  # Example public domain audio
VALID_PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"  # Example public PDF

INVALID_URL = "http://this.url.does.not.exist/resource.mp4"
URL_404 = "https://httpbin.org/status/404"  # URL that returns 404


# --- Fixtures ---

@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance for the tests."""
    # Ensure test media files exist
    if not SAMPLE_VIDEO_PATH.exists():
        pytest.skip(f"Test video file not found: {SAMPLE_VIDEO_PATH}")
    if not SAMPLE_AUDIO_PATH.exists():
        pytest.skip(f"Test audio file not found: {SAMPLE_AUDIO_PATH}")
    if not SAMPLE_PDF_PATH.exists():
        pytest.skip(f"Test PDF file not found: {SAMPLE_PDF_PATH}")

    with TestClient(fastapi_app_instance) as c:
        yield c


@pytest.fixture(scope="module")
def auth_headers():
    """Provides dummy authentication headers if needed."""
    # Replace 'YOUR_TEST_TOKEN' with a valid token if auth is enforced
    # If no auth, return empty dict
    return {"token": "YOUR_TEST_TOKEN"}  # Or {} if no auth


# --- Helper Functions ---

def check_batch_response(
        response,
        expected_status_code,
        expected_processed=None,
        expected_errors=None,
        check_results_len=None,
):
    """Helper to check common aspects of the batch response."""
    assert response.status_code == expected_status_code
    data = response.json()
    assert "results" in data
    assert "processed_count" in data
    assert "errors_count" in data
    assert "errors" in data

    if expected_processed is not None:
        assert data["processed_count"] == expected_processed
    if expected_errors is not None:
        assert data["errors_count"] == expected_errors
    if check_results_len is not None:
        assert len(data["results"]) == check_results_len
    return data  # Return parsed data for further checks


def check_media_item_result(result, expected_status, check_db_id=True):
    """Helper to check structure of a single item in the results list."""
    assert "status" in result
    assert result["status"] == expected_status
    assert "input_ref" in result
    assert "processing_source" in result
    assert "media_type" in result
    assert "metadata" in result
    assert "content" in result  # Should exist, might be empty on error
    # analysis, segments, chunks can be None
    assert "analysis" in result
    assert "segments" in result
    assert "chunks" in result
    assert "analysis_details" in result
    assert "error" in result
    assert "warnings" in result
    if check_db_id:
        assert "db_id" in result
        assert result["db_id"] is None  # Crucial for /process-* endpoints
        assert "db_message" in result
        assert result["db_message"] == "Processing only endpoint."

    if expected_status == "Error":
        assert result["error"] is not None
    # Add more specific checks based on expected_status if needed


# --- Test Classes ---

class TestProcessVideos:
    ENDPOINT = "/api/v1/media/process-videos"

    def test_process_video_url_success(self, client, auth_headers):
        """Test processing a single valid video URL."""
        form_data = {"urls": [VALID_VIDEO_URL], "perform_analysis": "false"}  # Faster without analysis
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        check_media_item_result(data["results"][0], "Success")
        assert data["results"][0]["media_type"] == "video"
        assert data["results"][0]["input_ref"] == VALID_VIDEO_URL
        assert len(data["results"][0]["content"]) > 0  # Check transcript exists

    def test_process_video_upload_success(self, client, auth_headers):
        """Test processing a single valid video file upload."""
        form_data = {"perform_analysis": "false"}
        with open(SAMPLE_VIDEO_PATH, "rb") as f:
            files = {"files": (SAMPLE_VIDEO_PATH.name, f, "video/mp4")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)

        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        check_media_item_result(data["results"][0], "Success")
        assert data["results"][0]["media_type"] == "video"
        assert data["results"][0]["input_ref"] == SAMPLE_VIDEO_PATH.name
        assert len(data["results"][0]["content"]) > 0

    def test_process_video_multiple_success(self, client, auth_headers):
        """Test processing multiple valid inputs (URL and Upload)."""
        form_data = {"urls": [VALID_VIDEO_URL], "perform_analysis": "false"}
        with open(SAMPLE_VIDEO_PATH, "rb") as f:
            files = {"files": (SAMPLE_VIDEO_PATH.name, f, "video/mp4")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)

        data = check_batch_response(response, 200, expected_processed=2, expected_errors=0, check_results_len=2)
        check_media_item_result(data["results"][0], "Success")  # URL result might be first or second
        check_media_item_result(data["results"][1], "Success")
        assert {r["media_type"] for r in data["results"]} == {"video"}

    def test_process_video_multi_status_mixed(self, client, auth_headers):
        """Test processing one valid URL and one invalid URL -> 207."""
        form_data = {"urls": [VALID_VIDEO_URL, INVALID_URL], "perform_analysis": "false"}
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)

        # It might take time for the invalid URL to fail
        time.sleep(5)  # Small delay might be needed depending on how timeouts are handled

        data = check_batch_response(response, 207, expected_processed=1, expected_errors=1, check_results_len=2)

        # Find the success and error results (order might vary)
        success_result = next((r for r in data["results"] if r["status"] == "Success"), None)
        error_result = next((r for r in data["results"] if r["status"] == "Error"), None)

        assert success_result is not None
        assert error_result is not None
        check_media_item_result(success_result, "Success")
        check_media_item_result(error_result, "Error")
        assert success_result["input_ref"] == VALID_VIDEO_URL
        assert error_result["input_ref"] == INVALID_URL
        assert error_result["error"] is not None

    def test_process_video_no_input(self, client, auth_headers):
        """Test sending request with no URLs or files."""
        response = client.post(self.ENDPOINT, data={}, headers=auth_headers)
        # Expect 400 based on _validate_inputs logic
        assert response.status_code == 400
        assert "At least one 'url' in the 'urls' list or one 'file' in the 'files' list must be provided." in response.json()["detail"]

    def test_process_video_validation_error(self, client, auth_headers):
        """Test sending invalid form data (e.g., bad chunk overlap)."""
        form_data = {
            "urls": [VALID_VIDEO_URL],
            "chunk_size": "100",
            "chunk_overlap": "200"  # Overlap > size
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)
        assert response.status_code == 422  # Unprocessable Entity
        assert "chunk_overlap must be less than chunk_size" in str(response.json())

    def test_process_video_with_analysis_and_chunking(self, client, auth_headers):
        """Test enabling analysis and chunking."""
        form_data = {
            "urls": [VALID_VIDEO_URL],
            "perform_analysis": "true",
            "perform_chunking": "true",
            "chunk_size": "500",  # Adjust if needed for your test video
            "chunk_overlap": "100",
            # Add api_name/api_key if required by your analysis library and not configured globally
            "api_name": "openai",
            "api_key": "lol-yea-right"
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["analysis"] is not None and len(result["analysis"]) > 0
        assert result["chunks"] is not None and len(result["chunks"]) > 0


class TestProcessAudios:
    ENDPOINT = "/api/v1/media/process-audios"

    def test_process_audio_url_success(self, client, auth_headers):
        """Test processing a single valid audio URL."""
        form_data = {"urls": [VALID_AUDIO_URL], "perform_analysis": "false"}
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        check_media_item_result(data["results"][0], "Success")
        assert data["results"][0]["media_type"] == "audio"
        assert len(data["results"][0]["content"]) > 0

    def test_process_audio_upload_success(self, client, auth_headers):
        """Test processing a single valid audio file upload."""
        form_data = {"perform_analysis": "false"}
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            files = {"files": (SAMPLE_AUDIO_PATH.name, f, "audio/mpeg")}  # Use correct MIME type
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)

        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        check_media_item_result(data["results"][0], "Success")
        assert data["results"][0]["media_type"] == "audio"
        assert len(data["results"][0]["content"]) > 0

    def test_process_audio_multi_status_mixed(self, client, auth_headers):
        """Test processing one valid upload and one 404 URL -> 207."""
        form_data = {"urls": [URL_404], "perform_analysis": "false"}
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            files = {"files": (SAMPLE_AUDIO_PATH.name, f, "audio/mpeg")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)

        # It might take time for the invalid URL to fail
        time.sleep(5)  # Small delay might be needed

        data = check_batch_response(response, 207, expected_processed=1, expected_errors=1, check_results_len=2)
        # Find the success and error results (order might vary)
        success_result = next((r for r in data["results"] if r["status"] == "Success"), None)
        error_result = next((r for r in data["results"] if r["status"] == "Error"), None)

        assert success_result is not None
        assert error_result is not None
        check_media_item_result(success_result, "Success")
        check_media_item_result(error_result, "Error")
        assert success_result["input_ref"] == SAMPLE_AUDIO_PATH.name
        assert error_result["input_ref"] == URL_404
        assert "Download failed" in error_result["error"] or "404" in error_result["error"]  # Check error message

    def test_process_audio_no_input(self, client, auth_headers):
        """Test sending request with no URLs or files."""
        response = client.post(self.ENDPOINT, data={}, headers=auth_headers)
        assert response.status_code == 400
        assert "No valid media suources supplied. At least one 'url' in the 'urls' list or one 'file' in the 'files' list must be provided." in response.json()["detail"]

    def test_process_audio_upload_invalid_format(self, client, auth_headers):
        """Test uploading a non-audio file (e.g., PDF)."""
        form_data = {}
        # Upload PDF as audio
        with open(SAMPLE_PDF_PATH, "rb") as f:
            files = {"files": (SAMPLE_PDF_PATH.name, f, "application/pdf")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)

        # Expect 207 because the processing library should fail for this item
        data = check_batch_response(response, 207, expected_processed=0, expected_errors=1, check_results_len=1)
        check_media_item_result(data["results"][0], "Error")
        assert "Error" in data["results"][0]["status"]  # Library should return an error status
        # Check specific error message if known (e.g., "ffmpeg error", "invalid format")
        assert data["results"][0]["error"] is not None


class TestProcessPdfs:
    ENDPOINT = "/api/v1/media/process-pdfs"

    def test_process_pdf_url_success(self, client, auth_headers):
        """Test processing a single valid PDF URL."""
        # Use pymupdf4llm parser by default
        form_data = {"urls": [VALID_PDF_URL], "perform_analysis": "false"}
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        check_media_item_result(data["results"][0], "Success")
        assert data["results"][0]["media_type"] == "pdf"
        assert data["results"][0]["metadata"] is not None
        assert data["results"][0]["metadata"].get("title") is not None
        assert len(data["results"][0]["content"]) > 0  # Check extracted text exists

    def test_process_pdf_upload_success(self, client, auth_headers):
        """Test processing a single valid PDF file upload."""
        form_data = {"perform_analysis": "false"}
        with open(SAMPLE_PDF_PATH, "rb") as f:
            files = {"files": (SAMPLE_PDF_PATH.name, f, "application/pdf")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)

        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        check_media_item_result(data["results"][0], "Success")
        assert data["results"][0]["media_type"] == "pdf"
        assert len(data["results"][0]["content"]) > 0

    # Add test for specific parser if needed
    # def test_process_pdf_upload_specific_parser(self, client, auth_headers):
    #     """Test processing PDF upload with a specific parser."""
    #     # NOTE: Requires the specified parser to be functional in your env
    #     form_data = {"pdf_parsing_engine": "pymupdf", "perform_analysis": "false"}
    #     with open(SAMPLE_PDF_PATH, "rb") as f:
    #         files = {"files": (SAMPLE_PDF_PATH.name, f, "application/pdf")}
    #         response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)

    #     data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
    #     check_media_item_result(data["results"][0], "Success")
    #     assert data["results"][0]["analysis_details"]["parser_used"] == "pymupdf" # Check if parser info is logged

    def test_process_pdf_multi_status_mixed(self, client, auth_headers):
        """Test one valid PDF upload, one invalid URL -> 207."""
        form_data = {"urls": [INVALID_URL], "perform_analysis": "false"}
        with open(SAMPLE_PDF_PATH, "rb") as f:
            files = {"files": (SAMPLE_PDF_PATH.name, f, "application/pdf")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)

        time.sleep(5)  # Give time for URL download to fail

        data = check_batch_response(response, 207, expected_processed=1, expected_errors=1, check_results_len=2)
        success_result = next((r for r in data["results"] if r["status"] == "Success"), None)
        error_result = next((r for r in data["results"] if r["status"] == "Error"), None)

        assert success_result is not None
        assert error_result is not None
        check_media_item_result(success_result, "Success")
        check_media_item_result(error_result, "Error")
        assert success_result["input_ref"] == SAMPLE_PDF_PATH.name
        assert error_result["input_ref"] == INVALID_URL
        assert "Download failed" in error_result["error"]

    def test_process_pdf_no_input(self, client, auth_headers):
        """Test sending request with no URLs or files."""
        response = client.post(self.ENDPOINT, data={}, headers=auth_headers)
        assert response.status_code == 400
        assert "No valid media suources supplied. At least one 'url' in the 'urls' list or one 'file' in the 'files' list must be provided." in response.json()["detail"]

    def test_process_pdf_upload_not_a_pdf(self, client, auth_headers):
        """Test uploading a non-PDF file (e.g., audio)."""
        form_data = {}
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            files = {"files": (SAMPLE_AUDIO_PATH.name, f, "audio/mpeg")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)

        # Endpoint checks magic bytes or relies on library error
        # If magic bytes check is done in endpoint -> 207 with input error
        # If check is done in library -> 207 with processing error
        data = check_batch_response(response, 207, expected_processed=0, expected_errors=1, check_results_len=1)
        check_media_item_result(data["results"][0], "Error")
        assert "Invalid file format" in data["results"][0]["error"] or "PDF Error" in data["results"][0][
            "error"]  # Check error type

    def test_process_pdf_with_analysis_and_chunking(self, client, auth_headers):
        """Test PDF analysis and chunking."""
        form_data = {
            "urls": [VALID_PDF_URL],
            "perform_analysis": "true",
            "perform_chunking": "true",
            "chunk_size": "300",  # Smaller chunk size for PDF text
            "chunk_overlap": "50"
            # Add api_name/api_key if needed
            # "api_name": "your_api",
            # "api_key": "your_key"
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["analysis"] is not None and len(result["analysis"]) > 0
        assert result["chunks"] is not None and len(result["chunks"]) > 0

