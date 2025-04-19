# test_media_processing.py
# Description: This file contains the test cases for the media processing endpoints of the tldw application.
#
# Imports
import os
import sys
from pathlib import Path
import time
from unittest.mock import patch

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
SAMPLE_EPUB_PATH = TEST_MEDIA_DIR / "sample.epub"
INVALID_FILE_PATH = TEST_MEDIA_DIR / "not_a_real_file.xyz"

# Use stable, short, publicly accessible URLs for testing
# Replace with actual URLs known to work
VALID_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example: Rick Astley (short duration helps)
VALID_AUDIO_URL = "https://cdn.pixabay.com/download/audio/2023/12/02/audio_2f291f569a.mp3?filename=about-anger-179423.mp3"  # Example public domain audio
VALID_PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"  # Example public PDF
VALID_EPUB_URL = "https://filesamples.com/samples/ebook/epub/Alices%20Adventures%20in%20Wonderland.epub"  # Example public EPUB
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
    if not SAMPLE_EPUB_PATH.exists():
        pytest.skip(f"Test EPUB file not found: {SAMPLE_EPUB_PATH}")

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


def check_media_item_result(result, expected_status, check_db_fields=True):
    """
    Helper to check structure of a single item in the results list.
    Updated to check for 'analysis' and 'chunks'.
    """
    assert isinstance(result, dict), f"Result item is not a dictionary: {result}"
    assert "status" in result, "Result missing 'status' key"
    assert result["status"] == expected_status, f"Expected status '{expected_status}', got '{result['status']}'"
    assert "input_ref" in result, "Result missing 'input_ref' key"
    assert "processing_source" in result, "Result missing 'processing_source' key"
    assert "media_type" in result, "Result missing 'media_type' key"
    assert "metadata" in result and isinstance(result["metadata"], dict), "Result missing or invalid 'metadata'"
    assert "content" in result, "Result missing 'content' key" # Allowed to be None or empty string
    assert "chunks" in result, "Result missing 'chunks' key" # Added check, allowed to be None
    assert "analysis" in result, "Result missing 'analysis' key" # Added check, allowed to be None
    assert "analysis_details" in result and isinstance(result["analysis_details"], dict), "Result missing or invalid 'analysis_details'"
    assert "error" in result, "Result missing 'error' key" # Allowed to be None
    assert "warnings" in result, "Result missing 'warnings' key" # Allowed to be None or list

    if check_db_fields:
        assert "db_id" in result, "Result missing 'db_id' key"
        assert result["db_id"] is None, f"Expected db_id to be None, got {result['db_id']}"
        assert "db_message" in result, "Result missing 'db_message' key"
        # Allow None or specific message for flexibility from library vs endpoint
        assert result["db_message"] in [None, "Processing only endpoint."], \
            f"Unexpected db_message: {result['db_message']}"

    if expected_status == "Error":
        assert result["error"] is not None, "Expected non-None 'error' for Error status"
    elif expected_status == "Success":
        # For success, error should ideally be None, but allow empty string too
        assert result["error"] is None or result["error"] == "", \
             f"Expected None or empty error for Success status, got {result['error']}"


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
    ENDPOINT = "/api/v1/media/process-audios" # Make sure this path is correct

    def test_process_audio_url_success_no_analysis_no_chunking(self, client, auth_headers):
        """Test processing audio URL, explicitly disabling analysis and chunking."""
        form_data = {
            "urls": [VALID_AUDIO_URL],
            "perform_analysis": "false", # Send as string 'false' for form data
            "perform_chunking": "false"
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
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


    def test_process_audio_upload_success_defaults(self, client, auth_headers):
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
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)

        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["media_type"] == "audio"
        assert result["input_ref"] == SAMPLE_AUDIO_PATH.name
        assert result["content"] is not None and len(result["content"]) > 0
        assert result["segments"] is not None and len(result["segments"]) > 0
        assert result["chunks"] is not None and len(result["chunks"]) > 0 # Expect chunks due to default
        assert result["analysis"] is not None and len(result["analysis"]) > 0 # Expect analysis due to default

    def test_process_audio_multi_status_mixed(self, client, auth_headers):
        """Test one valid upload and one invalid URL -> 207."""
        form_data = {
            "urls": [URL_404], # Use a reliable 404 URL
            "perform_analysis": "false", # Disable analysis for faster test
            "perform_chunking": "true"   # Keep chunking enabled (default)
        }
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            # Ensure the dummy audio file has some content for transcription to work
            files = {"files": (SAMPLE_AUDIO_PATH.name, f, "audio/mpeg")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)

        # No need to sleep, the request is sync, failures should be reported
        # If downloads were async background tasks, sleep might be needed

        data = check_batch_response(response, 207, expected_processed=1, expected_errors=1, check_results_len=2)

        success_result = next((r for r in data["results"] if r["status"] == "Success"), None)
        error_result = next((r for r in data["results"] if r["status"] == "Error"), None)

        assert success_result is not None, "Could not find success result"
        assert error_result is not None, "Could not find error result"

        check_media_item_result(success_result, "Success")
        check_media_item_result(error_result, "Error")

        # Check input refs carefully
        assert success_result["input_ref"] == SAMPLE_AUDIO_PATH.name
        assert error_result["input_ref"] == URL_404

        # Check error message for the failed URL
        assert error_result["error"] is not None
        assert "Download failed" in error_result["error"] or "404" in error_result["error"]

        # Check successful item results (assuming defaults enabled chunking)
        assert success_result["content"] is not None
        assert success_result["chunks"] is not None # Chunking was true

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

    def test_process_audio_no_input(self, client, auth_headers):
        """Test sending request with no URLs or files."""
        response = client.post(self.ENDPOINT, data={}, headers=auth_headers)
        # Expect 400 based on _validate_inputs logic in the endpoint
        assert response.status_code == 400, f"Expected 400, got {response.status_code}. Body: {response.text}"
        assert "No valid media sources supplied" in response.json()["detail"]

    def test_process_audio_upload_invalid_format_pdf(self, client, auth_headers):
        """Test uploading a non-audio file (PDF) which should fail conversion."""
        form_data = {"perform_analysis": "false"} # Disable analysis
        with open(SAMPLE_PDF_PATH, "rb") as f:
            files = {"files": (SAMPLE_PDF_PATH.name, f, "application/pdf")} # Correct MIME for PDFs
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)
        logger.debug(f"Test received response status: {response.status_code}")
        try:
            response_data_in_test = response.json()
            logger.debug(f"Test received response JSON: {response_data_in_test}")
        except Exception as e:
             logger.error(f"Test failed to parse response JSON: {e}")
             response_data_in_test = None

        # Expect 207 because the batch processes items individually
        # The PDF item should fail during the ffmpeg conversion step
        data = check_batch_response(response, 207, expected_processed=0, expected_errors=1, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Error")
        assert result["input_ref"] == SAMPLE_PDF_PATH.name
        assert result["error"] is not None
        # Check for error indicating conversion failure
        assert "Audio conversion failed" in result["error"] or "FFmpeg conversion failed" in result["error"]


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

    # Add test for specific parser if needed FIXME
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
        assert "No valid media sources supplied. At least one 'url' in the 'urls' list or one 'file' in the 'files' list must be provided." in response.json()["detail"]

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
        assert "Invalid file format" in data["results"][0]["error"] or "PDF Extraction Error." in data["results"][0][
            "error"]  # Check error type

    @patch(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.PDF."
        "PDF_Processing_Lib.summarize"  # ← patch the local copy
    )
    def test_process_pdf_with_analysis_and_chunking(self, mock_summarize, client, auth_headers):
        """Test PDF analysis and chunking."""
        mock_analysis_text = "This is the mocked analysis result."
        mock_summarize.return_value = mock_analysis_text

        form_data = {
            "urls": [VALID_PDF_URL],
            "perform_analysis": "true", # Analysis is enabled
            "perform_chunking": "true",
            "chunk_size": "300",
            "chunk_overlap": "50",
            "api_name": "mock_api",
            "api_key": "mock_key"
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")

        # Check that analysis was performed (mocked)
        mock_summarize.assert_called()
        assert result["analysis"] is not None
        assert result["analysis"] == mock_analysis_text # Check content
        assert len(result["analysis"]) > 0

        # Check that chunking happened (might need more specific checks depending on PDF content)
        assert result["chunks"] is not None
        assert isinstance(result["chunks"], list)
        assert len(result["chunks"]) > 0 # Should have at least one chunk
        # You could add more specific checks on chunk content/metadata if needed

        # Check that the mock was called (optional but good practice)
        mock_summarize.assert_called()



# =============================================================================
# Ebook Processing Tests
# =============================================================================
class TestProcessEbooks:
    ENDPOINT = "/api/v1/media/process-ebooks" # Make sure this path matches your router prefix + endpoint path

    # --- Happy Path Tests ---

    def test_process_ebook_url_success_defaults(self, client, auth_headers):
        """Test processing a single valid EPUB URL with default settings."""
        # Defaults: analysis=True, chunking=True, extraction='filtered'
        # Skip analysis for speed if not mocking/configured
        pytest.skip("Skipping analysis test until LLM mock/config confirmed.")
        form_data = {
            "urls": [VALID_EPUB_URL],
            "api_name": "mock_api", # Needed if analysis=True default
            "api_key": "mock_key"   # Needed if analysis=True default
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["media_type"] == "ebook"
        assert result["input_ref"] == VALID_EPUB_URL
        assert result["content"] is not None and len(result["content"]) > 0
        assert result["chunks"] is not None and len(result["chunks"]) > 0 # Expect chunks due to default
        assert result["analysis"] is not None and len(result["analysis"]) > 0 # Expect analysis due to default

    def test_process_ebook_url_success_no_analysis_no_chunking(self, client, auth_headers):
        """Test processing EPUB URL, disabling analysis and chunking."""
        form_data = {
            "urls": [VALID_EPUB_URL],
            "perform_analysis": "false",
            "perform_chunking": "false",
            "extraction_method": "basic" # Test another extraction method
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["media_type"] == "ebook"
        assert result["input_ref"] == VALID_EPUB_URL
        assert result["content"] is not None and len(result["content"]) > 0
        # Library creates one chunk if chunking is off
        assert result["chunks"] is not None and len(result["chunks"]) == 1
        assert result["analysis"] is None # Analysis was disabled

    def test_process_ebook_upload_success_defaults(self, client, auth_headers):
        """Test processing a single valid EPUB file upload with defaults."""
        # Skip analysis for speed if not mocking/configured
        form_data = {"perform_analysis": "false"}
        with open(SAMPLE_EPUB_PATH, "rb") as f:
            # Common EPUB MIME type, though server might not strictly check it
            files = {"files": (SAMPLE_EPUB_PATH.name, f, "application/epub+zip")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)

        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["media_type"] == "ebook"
        assert result["input_ref"] == SAMPLE_EPUB_PATH.name
        assert result["content"] is not None and len(result["content"]) > 0
        assert result["chunks"] is not None and len(result["chunks"]) > 0 # Default chunking=True

    def test_process_ebook_multiple_success(self, client, auth_headers):
        """Test processing multiple valid inputs (URL and Upload)."""
        form_data = {"urls": [VALID_EPUB_URL], "perform_analysis": "false"}
        with open(SAMPLE_EPUB_PATH, "rb") as f:
            files = {"files": (SAMPLE_EPUB_PATH.name, f, "application/epub+zip")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)

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

    def test_process_ebook_overrides(self, client, auth_headers):
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
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["metadata"]["title"] == test_title
        assert result["metadata"]["author"] == test_author
        assert result["keywords"] == test_keywords_list # Check the parsed list

    # --- Error Handling Tests ---

    def test_process_ebook_multi_status_mixed(self, client, auth_headers):
        """Test processing one valid URL and one invalid URL -> 207."""
        form_data = {"urls": [VALID_EPUB_URL, INVALID_URL], "perform_analysis": "false"}
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)

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

    def test_process_ebook_no_input(self, client, auth_headers):
        """Test sending request with no URLs or files."""
        response = client.post(self.ENDPOINT, data={}, headers=auth_headers)
        # Expect 400 based on _validate_inputs logic
        assert response.status_code == 400, f"Expected 400, got {response.status_code}. Body: {response.text}"
        assert "At least one 'url' in the 'urls' list or one 'file' in the 'files' list must be provided." in response.json()["detail"]

    def test_process_ebook_upload_invalid_format(self, client, auth_headers):
        """Test uploading a non-EPUB file (e.g., PDF), expecting upload helper rejection."""
        form_data = {}
        with open(SAMPLE_PDF_PATH, "rb") as f:
            files = {"files": (SAMPLE_PDF_PATH.name, f, "application/pdf")}
            response = client.post(self.ENDPOINT, data=form_data, files=files, headers=auth_headers)

        # _save_uploaded_files should reject based on extension ".epub"
        data = check_batch_response(response, 207, expected_processed=0, expected_errors=1, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Error")
        assert result["input_ref"] == SAMPLE_PDF_PATH.name
        assert "Upload error: Invalid file extension" in result["error"]

    def test_process_ebook_validation_error_bad_method(self, client, auth_headers):
        """Test sending invalid form data (invalid extraction_method)."""
        form_data = {
            "urls": [VALID_EPUB_URL],
            "extraction_method": "invalid_method" # Not in Literal[...]
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)
        assert response.status_code == 422  # Unprocessable Entity
        detail = response.json()["detail"]
        assert isinstance(detail, list) and len(detail) > 0
        # Check for the specific validation error message
        assert any("Input should be 'filtered', 'markdown' or 'basic'" in err.get("msg", "") for err in detail)
        assert any(err.get("loc") == ["body", "extraction_method"] for err in detail) # Check location


    # --- Option Variation Tests ---

    @pytest.mark.parametrize("method", ['filtered', 'markdown', 'basic'])
    def test_process_ebook_options_extraction_method(self, method, client, auth_headers):
        """Test different valid extraction methods."""
        form_data = {
            "urls": [VALID_EPUB_URL],
            "extraction_method": method,
            "perform_analysis": "false"
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")
        assert result["content"] is not None # Content should exist, might differ based on method

    # --- Mocked Analysis Test ---

    # IMPORTANT: Replace "path.to.your_ebook_processing_module.summarize" with the correct path
    @patch("tldw_Server_API.app.core.Ingestion_Media_Processing.Books.Book_Processing_Lib.summarize")
    def test_process_ebook_with_analysis_mocked(self, mock_summarize, client, auth_headers):
        """Test enabling analysis with mocking."""
        mock_analysis_text = "This is the mocked ebook analysis."
        mock_summarize.return_value = mock_analysis_text

        form_data = {
            "urls": [VALID_EPUB_URL],
            "perform_analysis": "true",
            "perform_chunking": "true", # Analysis often depends on chunks
            "api_name": "mock_api",     # Need to provide these even if mocking summarize directly
            "api_key": "mock_key"       # Depending on process_epub implementation checks
        }
        response = client.post(self.ENDPOINT, data=form_data, headers=auth_headers)
        data = check_batch_response(response, 200, expected_processed=1, expected_errors=0, check_results_len=1)
        result = data["results"][0]
        check_media_item_result(result, "Success")

        # Check mock was called (at least once, maybe more if chunked)
        mock_summarize.assert_called()
        # Check the *final* analysis result
        assert result["analysis"] is not None
        # The final result might be the mocked text joined, or a recursive summary
        # For simplicity here, we check if the mocked text is *in* the final result
        assert mock_analysis_text in result["analysis"]
        assert result["chunks"] is not None and len(result["chunks"]) > 0
        # Check if analysis was added to chunk metadata
        assert all('analysis' in chunk.get('metadata', {}) for chunk in result["chunks"])



