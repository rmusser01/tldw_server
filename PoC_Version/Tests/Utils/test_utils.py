# test_utils.py
# Description: This file contains the test cases for the Utils.py file in the App_Function_Libraries directory.
#
# Imports
import tempfile
import os
import hashlib
import sys
from unittest.mock import patch
#
# Third-party library imports
import pytest
import requests



#
# Add the tldw directory (one level up from Tests) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tldw')))
#
# Local Imports
from App_Function_Libraries.Utils import Utils
from App_Function_Libraries.Utils.Utils import (
    extract_text_from_segments, verify_checksum, create_download_directory,
    normalize_title, convert_to_seconds, is_valid_url, generate_unique_identifier,
    safe_read_file, cleanup_downloads, format_metadata_as_text, download_file,
    sanitize_filename, generate_unique_filename, clean_youtube_url
)
################################################################################################################################################################
# Test: extract_text_from_segments
##############################

def test_extract_text_from_segments_without_timestamps():
    segments = [{'Text': 'Hello world!'}]
    result = extract_text_from_segments(segments, include_timestamps=False)
    assert result == 'Hello world!'


def test_extract_text_from_segments_with_timestamps():
    segments = [{'Text': 'Hello world!', 'Time_Start': 0, 'Time_End': 5}]
    result = extract_text_from_segments(segments, include_timestamps=True)
    assert result == '0s - 5s | Hello world!'


def test_extract_text_from_segments_with_empty_segment():
    segments = [{}]
    result = extract_text_from_segments(segments)
    assert result == 'Error: Unable to extract transcription'


##############################
# Test: verify_checksum
##############################

def test_verify_checksum():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b'Test content')

    sha256_hash = hashlib.sha256(b'Test content').hexdigest()
    assert verify_checksum(temp_file.name, sha256_hash) == True
    assert verify_checksum(temp_file.name, "invalidchecksum") == False


##############################
# Test: create_download_directory
##############################

def test_create_download_directory(monkeypatch):
    def mock_makedirs(path, exist_ok):
        assert path == os.path.join("Results", "Test_Title")

    monkeypatch.setattr(os, 'makedirs', mock_makedirs)
    result = create_download_directory("Test Title")
    assert result == os.path.join("Results", "Test_Title")


##############################
# Test: normalize_title
##############################

# FIXME: This test is failing
def test_normalize_title():
    assert normalize_title('This/Is:A*Test?', preserve_spaces=False) == 'This_Is_A_Test'
    assert normalize_title('NoSpecialCharacters', preserve_spaces=False) == 'NoSpecialCharacters'
    assert normalize_title('Test with spaces', preserve_spaces=True) == 'Test with spaces'
    assert normalize_title('Test with spaces', preserve_spaces=False) == 'Test_with_spaces'


##############################
# Test: convert_to_seconds
##############################

def test_convert_to_seconds():
    assert convert_to_seconds('1:02:03') == 3723
    assert convert_to_seconds('02:03') == 123
    assert convert_to_seconds('45') == 45
    assert convert_to_seconds('') == 0


def test_convert_to_seconds_invalid():
    with pytest.raises(ValueError):
        convert_to_seconds('invalid')


##############################
# Test: is_valid_url
##############################

def test_is_valid_url():
    assert is_valid_url("http://example.com") == True
    assert is_valid_url("https://example.com") == True
    assert is_valid_url("ftp://example.com") == True
    assert is_valid_url("invalid-url") == False


##############################
# Test: generate_unique_identifier
##############################

def test_generate_unique_identifier():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b'Test content')

    result = generate_unique_identifier(temp_file.name)
    assert result.startswith('local:')


##############################
# Test: safe_read_file
##############################

def test_safe_read_file_valid():
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
        temp_file.write('Test content')
    assert safe_read_file(temp_file.name) == 'Test content'


def test_safe_read_file_file_not_found():
    result = safe_read_file('non_existent_file.txt')
    assert result.startswith('File not found')


def test_safe_read_file_invalid_encoding():
    with tempfile.NamedTemporaryFile(delete=False, mode='wb') as temp_file:
        temp_file.write(b'\x80\x81\x82')
    result = safe_read_file(temp_file.name)
    assert 'Unable to decode the file' in result


##############################
# Test: cleanup_downloads
##############################

def test_cleanup_downloads(monkeypatch):
    # Set the downloaded_files in the utils module
    Utils.downloaded_files = ['/path/to/file1', '/path/to/file2']
    removed_files = []

    def mock_exists(path):
        return True

    def mock_remove(path):
        removed_files.append(path)

    # Patch the os.path.exists and os.remove in the utils module
    monkeypatch.setattr(os.path, 'exists', mock_exists)
    monkeypatch.setattr(os, 'remove', mock_remove)

    # Call the function under test
    Utils.cleanup_downloads()

    # Assert that the removed_files contains the expected paths
    assert set(removed_files) == {'/path/to/file1', '/path/to/file2'}
    # Optionally, assert that downloaded_files remains unchanged if that's expected
    #assert set(Utils.downloaded_files) == {'/path/to/file1', '/path/to/file2'}


##############################
# Test: format_metadata_as_text
##############################
# FIXME: This test is failing
def test_format_metadata_as_text():
    metadata = {
        'title': 'Sample Video',
        'upload_date': '20230615',
        'view_count': 1000000,
        'like_count': 50000,
        'duration': 3725,
        'tags': ['tag1', 'tag2']
    }
    result = format_metadata_as_text(metadata)
    assert "Title: Sample Video" in result
    assert "Upload date: 2023-06-15" in result
    assert "View count: 1,000,000" in result
    assert "Duration: 01:02:05" in result


def test_format_metadata_as_text_no_metadata():
    result = format_metadata_as_text(None)
    assert result == "No metadata available"


##############################
# Test: download_file
##############################

@pytest.fixture
def mock_request_get(mocker):
    def mock_response(*args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.headers = {'content-length': '1024'}
                self.status_code = 200

            def iter_content(self, chunk_size=8192):
                yield b'Test content'

            def raise_for_status(self):
                pass

        return MockResponse()

    mocker.patch('requests.get', mock_response)


def test_download_file(mock_request_get, tmpdir):
    dest_path = os.path.join(tmpdir, 'test_file.txt')
    result = download_file('http://example.com/test', dest_path)
    assert os.path.exists(dest_path)
    assert result == dest_path


def test_download_file_retry_failure(mocker):
    def mock_response(*args, **kwargs):
        raise requests.exceptions.RequestException("Download failed")

    mocker.patch('requests.get', mock_response)

    with pytest.raises(Exception, match="Download failed"):
        download_file('http://example.com/test', '/path/to/file', max_retries=2)


##############################
# Test: sanitize_filename
##############################

def test_sanitize_filename():
    assert sanitize_filename('valid_filename.txt') == 'valid_filename.txt'
    assert sanitize_filename('invalid:/\\filename.txt') == 'invalidfilename.txt'
    assert sanitize_filename('file with spaces.txt') == 'file with spaces.txt'


##############################
# Test: generate_unique_filename
##############################

def test_generate_unique_filename(monkeypatch):
    def mock_exists(path):
        if 'file.txt' in path:
            return True
        return False

    monkeypatch.setattr(os.path, 'exists', mock_exists)
    base_path = '/path/to/files'
    base_filename = 'file.txt'
    result = generate_unique_filename(base_path, base_filename)
    assert result == 'file_1.txt'

    monkeypatch.setattr(os.path, 'exists', lambda x: False)
    result_no_collision = generate_unique_filename(base_path, base_filename)
    assert result_no_collision == 'file.txt'


##############################
# Test: clean_youtube_url
##############################

def test_clean_youtube_url():
    url = 'https://www.youtube.com/watch?v=abc123&list=xyz456'
    expected_clean_url = 'https://www.youtube.com/watch?v=abc123'
    assert clean_youtube_url(url) == expected_clean_url

    url_without_list = 'https://www.youtube.com/watch?v=abc123'
    assert clean_youtube_url(url_without_list) == url_without_list
