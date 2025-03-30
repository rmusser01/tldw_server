# Media_Wiki_Tests.py
# Description: Unit tests for the Media_Wiki module.
#
# Usage:
# pip install pytest pytest-asyncio
# pytest Media_Wiki_Tests.py
#
# Imports
import pytest
import asyncio
from unittest.mock import patch, MagicMock
# Local Imports
from Media_Wiki import parse_mediawiki_dump, optimized_chunking, process_single_item, import_mediawiki_dump, load_mediawiki_import_config
#
# #######################################################################################################################
#
# Functions:



@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_mwxml_dump():
    mock_dump = MagicMock()
    mock_page = MagicMock()
    mock_page.title = "Test Page"
    mock_page.namespace = 0
    mock_page.id = 1
    mock_revision = MagicMock()
    mock_revision.id = 1
    mock_revision.timestamp = "2021-01-01T00:00:00Z"
    mock_revision.text = "Test content"
    mock_page.revisions = [mock_revision]
    mock_dump.pages = [mock_page]
    return mock_dump

def test_parse_mediawiki_dump(mock_mwxml_dump):
    with patch('mwxml.Dump.from_file', return_value=mock_mwxml_dump), \
         patch('mwparserfromhell.parse') as mock_parse:
        mock_parse.return_value.strip_code.return_value = "Stripped content"
        result = list(parse_mediawiki_dump("dummy_path"))
        assert len(result) == 1
        assert result[0]['title'] == "Test Page"
        assert result[0]['content'] == "Stripped content"
        assert result[0]['namespace'] == 0
        assert result[0]['page_id'] == 1
        assert result[0]['revision_id'] == 1

def test_optimized_chunking():
    test_text = "== Section 1 ==\nContent 1\n== Section 2 ==\nContent 2"
    chunk_options = {'max_size': 50}
    result = optimized_chunking(test_text, chunk_options)
    assert len(result) == 2
    assert result[0]['text'].startswith("== Section 1 ==")
    assert result[1]['text'].startswith("== Section 2 ==")
    assert 'metadata' in result[0] and 'section' in result[0]['metadata']

@pytest.mark.asyncio
async def test_process_single_item():
    with patch('Media_Wiki.check_media_exists', return_value=False), \
         patch('Media_Wiki.add_media_with_keywords', return_value=1), \
         patch('Media_Wiki.process_and_store_content') as mock_process_store:
        await process_single_item("Test content", "Test Title", "TestWiki", {'max_size': 100})
        mock_process_store.assert_called()
        # Add more detailed assertions here

@pytest.mark.asyncio
async def test_import_mediawiki_dump():
    with patch('Media_Wiki.parse_mediawiki_dump') as mock_parse, \
         patch('Media_Wiki.process_single_item') as mock_process, \
         patch('Media_Wiki.load_checkpoint', return_value=0), \
         patch('Media_Wiki.save_checkpoint'), \
         patch('os.remove'):
        mock_parse.return_value = [{'page_id': 1, 'title': 'Test', 'content': 'Content'}]
        result = await import_mediawiki_dump("dummy_path", "TestWiki")
        assert "Successfully imported" in result
        mock_process.assert_called_once()

def test_import_mediawiki_dump_file_not_found():
    with patch('Media_Wiki.parse_mediawiki_dump', side_effect=FileNotFoundError):
        result = asyncio.run(import_mediawiki_dump("non_existent_path", "TestWiki"))
        assert "Error: File not found" in result

def test_load_mediawiki_import_config():
    with patch('builtins.open', MagicMock()):
        with patch('yaml.safe_load', return_value={'test_key': 'test_value'}):
            config = load_mediawiki_import_config()
            assert 'test_key' in config
            assert config['test_key'] == 'test_value'