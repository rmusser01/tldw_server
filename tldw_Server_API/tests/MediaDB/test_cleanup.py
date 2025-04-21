# tests/test_cleanup.py
import pytest
import os
from unittest.mock import patch
from tldw_Server_API.app.core.Utils.Utils import cleanup_downloads

@pytest.fixture
def temp_download_files():
    files = ['test1.mp4', 'test2.mp3', 'test3.txt']
    for file in files:
        with open(file, 'w') as f:
            f.write('test content')
    yield files
    for file in files:
        if os.path.exists(file):
            os.remove(file)

@patch('tldw_Server_API.app.core.Utils.Utils.downloaded_files', new_callable=lambda: ['test1.mp4', 'test2.mp3', 'test3.txt'])
def test_cleanup_downloads(mock_downloaded_files, temp_download_files):
    cleanup_downloads()
    for file in temp_download_files:
        assert not os.path.exists(file)