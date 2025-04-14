# test_add_media_endpoint.py
# # Description: This file contains the test cases for the /media/add endpoint of the FastAPI application.
#
# Imports
import pytest
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any
#
# 3rd-party Libraries
from unittest.mock import patch, MagicMock, mock_open, ANY
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
    app = FastAPI()
    app.include_router(media_router, prefix="/api/v1/media")
    return app

@pytest.fixture(scope="module")
def client(test_app):
    with TestClient(test_app) as c:
        yield c

@pytest.fixture
def dummy_file_content():
    return b"dummy file content for testing"

@pytest.fixture
def create_upload_file(dummy_file_content):
    """Factory to create mock UploadFile objects."""
    def _create(filename="test_upload.mp4"):
        file_bytes = BytesIO(dummy_file_content)
        # Use UploadFile constructor directly
        up_file = UploadFile(filename=filename, file=file_bytes)
        # Reset pointer for reading
        up_file.file.seek(0)
        return up_file
    return _create

@pytest.fixture
def mock_temp_dir_manager(mocker):
    """Mocks the TempDirManager context manager."""
    mock_instance = MagicMock()
    mock_instance.__enter__.return_value = Path("/fake/temp/dir") # Simulate entering context
    mock_instance.__exit__.return_value = None # Simulate exiting context
    mock_instance.get_path.return_value = Path("/fake/temp/dir")
    mocker.patch('tldw_Server_API.app.api.v1.endpoints.media.TempDirManager', return_value=mock_instance)
    # Also mock underlying mkdtemp/rmtree if TempDirManager uses them directly and isn't mocked itself
    mocker.patch('tempfile.mkdtemp', return_value="/fake/temp/dir")
    mock_rmtree = mocker.patch('shutil.rmtree')
    return mock_instance, mock_rmtree

@pytest.fixture(autouse=True)
def mock_dependencies(mocker, mock_temp_dir_manager):
    """Mocks external dependencies for all tests."""
    # File system mocks (some handled by mock_temp_dir_manager)
    mock_open_instance = mocker.patch('builtins.open', mock_open())
    mock_os_path_exists = mocker.patch('os.path.exists', return_value=True) # Crucial for cleanup checks
    mocker.patch('pathlib.Path.unlink', return_value=None) # Mock unlink to avoid errors on fake paths

    # Network mocks
    mock_requests_get = mocker.patch('requests.get')
    mock_response = MagicMock(spec=requests.Response) # Use spec for better mocking
    mock_response.status_code = 200
    mock_response.content = b'downloaded content'
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {'content-type': 'application/octet-stream'} # Add headers
    mock_response.url = "http://mockedurl.com/file" # Add url attribute
    # Add json method if needed by code under test
    mock_response.json = MagicMock(return_value={})
    mock_requests_get.return_value = mock_response

    # --- Mock SINGLE-ITEM Processing Functions ---
    # Adjust return values to match the expected dict structure
    mock_process_video = mocker.patch('tldw_Server_API.app.api.v1.endpoints.media.process_video_item',
                                      return_value={"status": "Success", "db_id": "vid_mock_123", "input": ANY})
    mock_process_audio = mocker.patch('tldw_Server_API.app.api.v1.endpoints.media.process_audio_item',
                                      return_value={"status": "Success", "db_id": "aud_mock_456", "input": ANY})
    mock_process_docs = mocker.patch('tldw_Server_API.app.api.v1.endpoints.media.process_document_item',
                                     return_value={"status": "Success", "db_id": "doc_mock_789", "input": ANY})
    mock_process_pdf = mocker.patch('tldw_Server_API.app.api.v1.endpoints.media.process_pdf_item',
                                    return_value={"status": "Success", "db_id": "pdf_mock_abc", "input": ANY})
    # Make ebook return the expected *string* first, the endpoint handles the rest
    mock_import_epub = mocker.patch('tldw_Server_API.app.api.v1.endpoints.media.import_ebook_item',
                                     return_value="Success. Media ID: ebook_mock_def. Input: /fake/temp/dir/mock_ebook.epub")

    # Mock DB functions (if endpoint calls them directly, otherwise mock the processors above)
    # If processors handle DB internally, these might not be needed directly by endpoint tests
    mock_add_db = mocker.patch('tldw_Server_API.app.api.v1.endpoints.media.add_media_to_database', return_value={"media_id": "db_mock_id"})
    mock_extract_id = mocker.patch('tldw_Server_API.app.api.v1.endpoints.media.extract_id_from_result', return_value="db_mock_id") # Might not be called directly now
    mock_extract_epub_id = mocker.patch('tldw_Server_API.app.api.v1.endpoints.media.extract_media_id_from_result_string', return_value="ebook_mock_def")

    # Mock BackgroundTasks
    mock_bg_tasks_instance = MagicMock(spec=BackgroundTasks)
    mocker.patch('fastapi.BackgroundTasks', return_value=mock_bg_tasks_instance)

    # Mock Path name attribute needed for downloads/filenames
    mocker.patch.object(Path, 'name', new_callable=mocker.PropertyMock, return_value='mock_filename.ext')


    return {
        "requests_get": mock_requests_get,
        "process_video": mock_process_video,
        "process_audio": mock_process_audio,
        "process_docs": mock_process_docs,
        "process_pdf": mock_process_pdf,
        "import_epub": mock_import_epub,
        "add_db": mock_add_db,
        "extract_id": mock_extract_id, # May be unused
        "extract_epub_id": mock_extract_epub_id, # Endpoint uses this
        "mkdtemp": mocker.patch('tempfile.mkdtemp'), # Mock underlying if needed
        "rmtree": mock_temp_dir_manager[1], # Get rmtree mock from manager fixture
        "open": mock_open_instance,
        "bg_tasks": mock_bg_tasks_instance,
        "os_path_exists": mock_os_path_exists,
        "temp_dir_manager": mock_temp_dir_manager[0], # Get manager mock itself
    }

# --- Helper for Form Data (Handles Lists) ---
def create_form_data(media_type: str, urls: List[str] = None, **overrides) -> Dict[str, Any]:
    """Creates form data dict, handling lists for URLs."""
    defaults = {
        "media_type": media_type,
        "keywords": "test, default",
        "whisper_model": "tiny",
        "diarize": False,
        "timestamp_option": True,
        "keep_original_file": False,
        "overwrite_existing": False,
        "perform_analysis": True,
        "perform_chunking": False, # Keep False for simpler default tests
        "chunk_size": 500,
        "chunk_overlap": 100,
        "use_cookies": False,
        # Add other defaults from your endpoint signature here...
        "pdf_parsing_engine": "pymupdf4llm",
    }
    data = defaults
    data.update(overrides)

    # Handle list format for TestClient data - needs repeated keys
    form_data_list = []
    if urls:
        for url in urls:
            form_data_list.append(("urls", url))
        if "urls" in data: del data["urls"] # Remove from dict if handled as list

    # Add other key-value pairs
    for k, v in data.items():
        if v is not None: # Don't send None values
            form_data_list.append((k, str(v))) # Convert bools/ints to string for form data

    # Convert list of tuples to dict suitable for 'data' param if no lists needed,
    # but keep as list of tuples if 'urls' or other lists are present.
    # For simplicity with 'urls', we return the list of tuples.
    # If only single values, TestClient's data param works fine with a dict.
    # Since we need list support for 'urls', using list of tuples is safer.
    return form_data_list # Return list of tuples

# --- Test Cases ---

# === Validation Tests ===
def test_add_media_invalid_media_type(client):
    # Need to send data as list of tuples because create_form_data returns that
    form_data = create_form_data(media_type="invalid_type", urls=["http://a.com"]) # Provide a URL to pass other validation
    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid media_type" in response.json()["detail"]

def test_add_media_missing_url_and_file(client):
    # Explicitly provide no URLs and no files
    form_data = create_form_data(media_type="video") # No urls=[] here
    # Find and remove 'urls' if accidentally added by default create_form_data logic
    form_data = [item for item in form_data if item[0] != 'urls']

    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "must be provided" in response.json()["detail"]

def test_add_media_missing_required_form_field(client):
    # TestClient needs 'data' or 'files', but FastAPI validates Form fields from the body
    # Send intentionally incomplete data
    response = client.post("/api/v1/media/add", data=[("urls", "http://a.com")], headers={"token": "fake-token"}) # Missing media_type
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "media_type" in str(response.json())
    assert "field required" in str(response.json()).lower()

def test_add_media_not_implemented_type(client, mock_dependencies):
    # Test a valid type that might hit a NotImplementedError inside processing loop
    form_data = create_form_data(media_type="video", urls=["http://example.com/video.mp4"])
    # Make the *processor* raise NotImplementedError
    mock_dependencies["process_video"].side_effect = NotImplementedError("Video processing not ready")

    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    # Endpoint now catches this per-item and returns 207
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Failed"
    assert "NotImplementedError" in results[0]["error"] # Check specific error if needed
    assert "Video processing not ready" in results[0]["error"]

# === File Handling Tests ===
def test_add_media_single_file_upload_success_cleanup(client, create_upload_file, mock_dependencies, mock_temp_dir_manager):
    test_file = create_upload_file("audio.mp3")
    form_data = create_form_data(media_type="audio", keep_original_file=False)
    files_tuple = [("files", (test_file.filename, test_file.file, test_file.content_type))]

    response = client.post(
        "/api/v1/media/add",
        files=files_tuple,
        data=form_data,
        headers={"token": "fake-token"}
    )

    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Success"
    assert results[0]["db_id"] == "aud_mock_456"
    assert test_file.filename in results[0]["input"] # Input should contain the temp path

    # Check cleanup via context manager __exit__
    manager_mock, rmtree_mock = mock_temp_dir_manager
    manager_mock.__exit__.assert_called_once()
    # Check rmtree directly if __exit__ calls it (depends on TempDirManager impl)
    # Assuming __exit__ calls rmtree when keep_original_file=False
    rmtree_mock.assert_called_once_with(Path("/fake/temp/dir"))
    mock_dependencies["bg_tasks"].add_task.assert_not_called() # Cleanup is synchronous via __exit__

def test_add_media_single_file_upload_success_no_cleanup(client, create_upload_file, mock_dependencies, mock_temp_dir_manager):
    test_file = create_upload_file("audio.mp3")
    form_data = create_form_data(media_type="audio", keep_original_file=True) # Keep file
    files_tuple = [("files", (test_file.filename, test_file.file, test_file.content_type))]

    response = client.post(
        "/api/v1/media/add",
        files=files_tuple,
        data=form_data,
        headers={"token": "fake-token"}
    )

    assert response.status_code == status.HTTP_207_MULTI_STATUS # Still 207
    # ... check results ...
    assert response.json()["results"][0]["status"] == "Success"

    # Context manager __exit__ still called, but shouldn't call rmtree
    manager_mock, rmtree_mock = mock_temp_dir_manager
    manager_mock.__exit__.assert_called_once()
    rmtree_mock.assert_not_called() # rmtree should NOT be called
    mock_dependencies["bg_tasks"].add_task.assert_not_called() # No BG task

# Mock 'open' within the endpoint's file saving loop to simulate write error
@patch('tldw_Server_API.app.api.v1.endpoints.media.open', side_effect=IOError("Disk full"))
def test_add_media_file_save_io_error(mock_open_error, client, create_upload_file, mock_dependencies):
    test_file = create_upload_file("audio.mp3")
    form_data = create_form_data(media_type="audio")
    files_tuple = [("files", (test_file.filename, test_file.file, test_file.content_type))]

    response = client.post(
        "/api/v1/media/add",
        files=files_tuple,
        data=form_data,
        headers={"token": "fake-token"}
    )

    assert response.status_code == status.HTTP_207_MULTI_STATUS # Endpoint continues, reports failure
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Failed"
    assert "Failed to save uploaded file" in results[0]["error"]
    assert "IOError" in results[0]["error"]
    # Ensure processor was NOT called for this failed file
    mock_dependencies["process_audio"].assert_not_called()

# Mock 'tempfile.mkdtemp' used by TempDirManager to simulate creation error
@patch('tempfile.mkdtemp', side_effect=OSError("Permission denied"))
def test_add_media_temp_dir_creation_error(mock_mkdtemp_error, client, create_upload_file):
    # This error happens *before* the loop, so it raises HTTPException directly
    test_file = create_upload_file("audio.mp3")
    form_data = create_form_data(media_type="audio")
    files_tuple = [("files", (test_file.filename, test_file.file, test_file.content_type))]

    response = client.post(
        "/api/v1/media/add",
        files=files_tuple,
        data=form_data,
        headers={"token": "fake-token"}
    )
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to create temporary directory" in response.json()["detail"]
    assert "Permission denied" in response.json()["detail"]

# === Download Error Tests ===
def test_add_media_pdf_download_404(client, mock_dependencies):
    pdf_url = "http://example.com/not_found.pdf"
    form_data = create_form_data(media_type="pdf", urls=[pdf_url])

    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 404
    mock_response.url = pdf_url
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error", response=mock_response)
    mock_dependencies["requests_get"].return_value = mock_response

    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["input"] == pdf_url
    assert results[0]["status"] == "Failed"
    assert "Failed to download file" in results[0]["error"]
    assert "404" in results[0]["error"]
    mock_dependencies["requests_get"].assert_called_once_with(pdf_url, timeout=ANY, headers=ANY, cookies=ANY)
    mock_dependencies["process_pdf"].assert_not_called()

def test_add_media_doc_download_network_error(client, mock_dependencies):
    doc_url = "http://example.com/unreachable.doc"
    form_data = create_form_data(media_type="document", urls=[doc_url])

    mock_dependencies["requests_get"].side_effect = requests.exceptions.ConnectionError("Cannot connect")

    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Failed"
    assert "Failed to download file" in results[0]["error"]
    assert "Cannot connect" in results[0]["error"]
    mock_dependencies["process_docs"].assert_not_called()

# === Processing Function Error Tests ===
def test_add_media_video_processing_exception(client, mock_dependencies):
    video_url = "http://example.com/bad_video.mp4"
    form_data = create_form_data(media_type="video", urls=[video_url])
    mock_dependencies["process_video"].side_effect = ValueError("Bad video data")

    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Failed"
    assert "Input error: Bad video data" in results[0]["error"] # Specific error message check

def test_add_media_audio_processing_returns_error_status(client, mock_dependencies):
    audio_url = "http://example.com/noisy.mp3"
    form_data = create_form_data(media_type="audio", urls=[audio_url])
    # Configure mock to return a non-success status dict
    mock_dependencies["process_audio"].return_value = {
        "status": "Failed", "message": "Transcription failed noise", "input": audio_url, "error": "Too noisy"
    }

    response = client.post("/api/v1/media/add", data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Failed"
    assert results[0]["error"] == "Too noisy" # Check specific error
    assert results[0].get("message") == "Transcription failed noise" # Check message if needed

def test_add_media_pdf_processing_returns_none(client, create_upload_file, mock_dependencies):
    # Test unexpected return type from processor
    test_file = create_upload_file("document.pdf")
    form_data = create_form_data(media_type="pdf")
    files_tuple = [("files", (test_file.filename, test_file.file, test_file.content_type))]
    mock_dependencies["process_pdf"].return_value = None # Invalid return type

    response = client.post("/api/v1/media/add", files=files_tuple, data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "Failed"
    assert "unexpected result type: NoneType" in results[0]["error"]

# === Database / Result Handling Error Tests ===
def test_add_media_ebook_db_extract_id_fails(client, create_upload_file, mock_dependencies):
    test_file = create_upload_file("book.epub")
    form_data = create_form_data(media_type="ebook")
    files_tuple = [("files", (test_file.filename, test_file.file, test_file.content_type))]

    # Processor succeeds but returns unclear message
    epub_result_msg = "Processing finished okay maybe."
    mock_dependencies["import_epub"].return_value = epub_result_msg
    mock_dependencies["extract_epub_id"].return_value = None # Simulate extraction failure

    response = client.post("/api/v1/media/add", files=files_tuple, data=form_data, headers={"token": "fake-token"})
    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 1
    # Status is Warning because processing succeeded but ID extraction failed
    assert results[0]["status"] == "Warning"
    assert results[0]["message"] == epub_result_msg
    assert "failed to extract DB ID" in results[0]["error"]
    assert results[0].get("db_id") is None


# === Success Path Examples (Multi-Input) ===
def test_add_media_mixed_url_file_success(client, create_upload_file, mock_dependencies):
    video_url = "http://example.com/good_video.mp4"
    test_file = create_upload_file("another.mp4")
    form_data = create_form_data(media_type="video", urls=[video_url], perform_chunking=True)
    files_tuple = [("files", (test_file.filename, test_file.file, test_file.content_type))]

    # Mock processor to return distinct IDs based on input
    def side_effect_video(*args, **kwargs):
        input_ref = kwargs.get("input_ref")
        id_suffix = "url" if isinstance(input_ref, str) else "file"
        return {"status": "Success", "db_id": f"vid_mock_{id_suffix}", "input": input_ref}
    mock_dependencies["process_video"].side_effect = side_effect_video

    response = client.post(
        "/api/v1/media/add",
        files=files_tuple,
        data=form_data,
        headers={"token": "fake-token"}
    )

    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 2

    # Check result for URL input
    url_result = next((r for r in results if r["input"] == video_url), None)
    assert url_result is not None
    assert url_result["status"] == "Success"
    assert url_result["db_id"] == "vid_mock_url"

    # Check result for File input (input ref will be the temp path)
    file_result = next((r for r in results if test_file.filename in r["input"]), None)
    assert file_result is not None
    assert file_result["status"] == "Success"
    assert file_result["db_id"] == "vid_mock_file"

    # Verify processor was called twice
    assert mock_dependencies["process_video"].call_count == 2
    # Check args for one of the calls (e.g., the URL call)
    mock_dependencies["process_video"].assert_any_call(
        input_ref=video_url,
        # Add other specific args passed by endpoint
        keywords=['test', 'default'],
        chunk_options=ANY, # Check chunk options passed correctly if needed
        perform_analysis=True,
        store_in_db=True,
        diarize=False,
        whisper_model='tiny',
        language='en',
        timestamp_option=True,
        confab_check=False,
        # Use ANY for less critical args
        custom_prompt=ANY, system_prompt=ANY, overwrite_existing=ANY,
        api_name=ANY, api_key=ANY
    )

def test_add_media_multiple_failures_and_success(client, create_upload_file, mock_dependencies):
    good_url = "http://example.com/good.pdf"
    bad_url_404 = "http://example.com/notfound.pdf"
    good_file = create_upload_file("report.pdf")
    # bad_file_save (will fail during save)
    bad_file_process = create_upload_file("corrupt.pdf")

    form_data = create_form_data(media_type="pdf", urls=[good_url, bad_url_404])
    files_tuple = [
        ("files", (good_file.filename, good_file.file, good_file.content_type)),
        ("files", (bad_file_process.filename, bad_file_process.file, bad_file_process.content_type)),
    ]

    # Mock requests for 404
    mock_response_404 = MagicMock(spec=requests.Response)
    mock_response_404.status_code = 404
    mock_response_404.url = bad_url_404
    mock_response_404.raise_for_status.side_effect = requests.exceptions.HTTPError("404", response=mock_response_404)

    # Mock requests for good URL
    mock_response_200 = MagicMock(spec=requests.Response)
    mock_response_200.status_code = 200
    mock_response_200.content = b'good pdf content'
    mock_response_200.url = good_url
    mock_response_200.raise_for_status = MagicMock()

    mock_dependencies["requests_get"].side_effect = lambda url, *args, **kwargs: mock_response_200 if url == good_url else mock_response_404

    # Mock processing failure for one file
    def side_effect_pdf(*args, **kwargs):
        filename = kwargs.get("filename")
        input_ref = kwargs.get("input_ref")
        if "corrupt" in filename:
            raise ValueError("Cannot parse corrupt PDF")
        else:
            # Determine if URL or File based on input_ref structure or filename
            id_suffix = "url" if ".com" in input_ref else "file"
            return {"status": "Success", "db_id": f"pdf_mock_{id_suffix}", "input": input_ref, "filename": filename}
    mock_dependencies["process_pdf"].side_effect = side_effect_pdf

    response = client.post("/api/v1/media/add", files=files_tuple, data=form_data, headers={"token": "fake-token"})

    assert response.status_code == status.HTTP_207_MULTI_STATUS
    results = response.json()["results"]
    assert len(results) == 4 # good_url, bad_url, good_file, bad_file_process

    # Check statuses
    statuses = {r["input"].split('/')[-1].split('\\')[-1]: r["status"] for r in results} # Use basename/url as key
    assert statuses[good_url] == "Success"
    assert statuses[bad_url_404] == "Failed"
    assert statuses[good_file.filename] == "Success" # Input ref will be temp path, check filename? Or check DB ID
    assert statuses[bad_file_process.filename] == "Failed" # Input ref will be temp path

    # Find specific results
    good_url_res = next(r for r in results if r["input"] == good_url)
    bad_url_res = next(r for r in results if r["input"] == bad_url_404)
    good_file_res = next(r for r in results if r.get("db_id") == "pdf_mock_file")
    bad_file_res = next(r for r in results if "corrupt.pdf" in r["input"]) # Find by filename in path

    assert "pdf_mock_url" in good_url_res.get("db_id", "")
    assert "Failed to download" in bad_url_res["error"]
    assert "pdf_mock_file" in good_file_res.get("db_id", "")
    assert "Cannot parse corrupt PDF" in bad_file_res["error"]

    # Check calls
    assert mock_dependencies["requests_get"].call_count == 2
    assert mock_dependencies["process_pdf"].call_count == 3 # Called for good_url, good_file, bad_file_process

#
# End of test_add_media_endpoint.py
########################################################################################################################
