# test_chunk_lib.py
#
#
# Imports
import json
import os
import sys
#
# External library imports
import pytest
#
# Add the project root (parent directory of App_Function_Libraries) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
#
print(f"Project root added to sys.path: {project_root}")
#
# Local imports
from PoC_Version.App_Function_Libraries.Chunk_Lib import improved_chunking_process
#
################################################################################################################################################################
#
# Test: improved_chunking_process


def test_chunk_json_list():
    json_text = '''
    [
        {"id": 1, "content": "Item 1"},
        {"id": 2, "content": "Item 2"},
        {"id": 3, "content": "Item 3"},
        {"id": 4, "content": "Item 4"},
        {"id": 5, "content": "Item 5"}
    ]
    '''
    chunk_options = {
        'method': 'json',
        'max_size': 2,
        'overlap': 1
    }
    chunks = improved_chunking_process(json_text, chunk_options)
    assert len(chunks) == 5  # Updated expectation
    assert json.loads(chunks[0]['text']) == [
        {"id": 1, "content": "Item 1"},
        {"id": 2, "content": "Item 2"}
    ]
    assert json.loads(chunks[1]['text']) == [
        {"id": 2, "content": "Item 2"},
        {"id": 3, "content": "Item 3"}
    ]
    assert json.loads(chunks[2]['text']) == [
        {"id": 3, "content": "Item 3"},
        {"id": 4, "content": "Item 4"}
    ]
    assert json.loads(chunks[3]['text']) == [
        {"id": 4, "content": "Item 4"},
        {"id": 5, "content": "Item 5"}
    ]
    assert json.loads(chunks[4]['text']) == [
        {"id": 5, "content": "Item 5"}
    ]


def test_chunk_json_dict():
    json_text = '''
    {
        "metadata": {
            "title": "Test Document",
            "author": "Author Name"
        },
        "data": {
            "section1": "Content 1",
            "section2": "Content 2",
            "section3": "Content 3",
            "section4": "Content 4",
            "section5": "Content 5"
        }
    }
    '''
    chunk_options = {
        'method': 'json',
        'max_size': 2,
        'overlap': 1
    }
    chunks = improved_chunking_process(json_text, chunk_options)

    # Expected Chunks: 4
    expected_chunks = [
        {
            "metadata": {
                "title": "Test Document",
                "author": "Author Name"
            },
            "data": {
                "section1": "Content 1",
                "section2": "Content 2"
            }
        },
        {
            "metadata": {
                "title": "Test Document",
                "author": "Author Name"
            },
            "data": {
                "section2": "Content 2",
                "section3": "Content 3"
            }
        },
        {
            "metadata": {
                "title": "Test Document",
                "author": "Author Name"
            },
            "data": {
                "section3": "Content 3",
                "section4": "Content 4"
            }
        },
        {
            "metadata": {
                "title": "Test Document",
                "author": "Author Name"
            },
            "data": {
                "section4": "Content 4",
                "section5": "Content 5"
            }
        }
    ]

    assert len(chunks) == len(expected_chunks), f"Expected {len(expected_chunks)} chunks, got {len(chunks)}"

    for i, expected in enumerate(expected_chunks):
        actual = json.loads(chunks[i]['text'])
        assert actual == expected, f"Chunk {i + 1} does not match expected."

def test_invalid_json():
    invalid_json_text = '''
    {
        "metadata": {
            "title": "Invalid JSON",
            "author": "Author Name"
        },
        "data": [
            {"id": 1, "content": "Item 1"},
            {"id": 2, "content": "Item 2"}
    '''
    chunk_options = {
        'method': 'json',
        'max_size': 2,
        'overlap': 1
    }
    with pytest.raises(ValueError):
        improved_chunking_process(invalid_json_text, chunk_options)

def test_unsupported_json_structure():
    json_text = '"Just a plain string, not an object or array."'
    chunk_options = {
        'method': 'json',
        'max_size': 2,
        'overlap': 1
    }
    with pytest.raises(ValueError):
        improved_chunking_process(json_text, chunk_options)