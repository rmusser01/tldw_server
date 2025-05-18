# test_chromadb.py
# Description: This file contains the test cases for the ChromaDB_Library.py file in the App_Function_Libraries directory.
#
# Imports
import os
import sys
from unittest.mock import patch, MagicMock
# Third-party library imports
import pytest



#
####################################################################################################
#
# Status: FIXME

# Add the project root (parent directory of App_Function_Libraries) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

print(f"Project root added to sys.path: {project_root}")

# Local Imports
#from App_Function_Libraries.Utils.Utils import load_and_log_configs
from PoC_Version.App_Function_Libraries.RAG.ChromaDB_Library import (
    process_and_store_content, check_embedding_status,
    reset_chroma_collection, vector_search, store_in_chroma, batched, embedding_api_url
)

#
############################################
# Fixtures for Reusable Mocking and Setup
############################################

default_api_endpoint = "openai"


# Fixture to mock a ChromaDB collection
@pytest.fixture
def mock_collection():
    mock_col = MagicMock()
    # Mock the upsert method
    mock_col.upsert = MagicMock()
    # Mock the get method to return embeddings and complete metadatas
    mock_col.get.return_value = {
        'embeddings': [[0.1, 0.2], [0.3, 0.4]],
        'metadatas': [
            {'embedding_model': 'text-embedding-3-small', 'embedding_provider': 'openai'},
            {'embedding_model': 'text-embedding-3-small', 'embedding_provider': 'openai'}
        ]
    }
    # Mock the query method for vector_search
    mock_col.query.return_value = {
        'documents': [["Document 1", "Document 2"]],
        'metadatas': [
            {"embedding_model": "text-embedding-3-small", "embedding_provider": "openai"},
            {"embedding_model": "text-embedding-3-small", "embedding_provider": "openai"}
        ]
    }
    return mock_col

@pytest.fixture
def mock_chroma_client():
    with patch('App_Function_Libraries.RAG.ChromaDB_Library.chromadb.PersistentClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_database(mocker):
    """Fixture to mock the database with an execute_query method."""
    mock_db = MagicMock()
    mock_db.execute_query = MagicMock()
    return mock_db

##############################
# Test: preprocess_all_content
##############################

@pytest.fixture
def mock_unprocessed_media(mocker):
    """Fixture to mock unprocessed media data."""
    return [(1, "Test Content", "video", "test_file.mp4")]

@pytest.fixture
def mock_process_and_store(mocker):
    """Fixture to mock process_and_store_content."""
    return mocker.patch("App_Function_Libraries.RAG.ChromaDB_Library.process_and_store_content")

@pytest.fixture
def mock_mark_media_processed(mocker):
    """Fixture to mock mark_media_as_processed."""
    return mocker.patch("App_Function_Libraries.RAG.ChromaDB_Library.mark_media_as_processed")


##############################
# Test: process_and_store_content
##############################

@pytest.fixture
def mock_chunk_for_embedding(mocker):
    """Fixture to mock chunk_for_embedding."""
    return mocker.patch("App_Function_Libraries.RAG.ChromaDB_Library.chunk_for_embedding", return_value=[
        {"text": "Chunk 1 text", "metadata": {"start_index": 0, "end_index": 100, "file_name": "test.mp4", "relative_position": 0.1}},
        {"text": "Chunk 2 text", "metadata": {"start_index": 101, "end_index": 200, "file_name": "test.mp4", "relative_position": 0.2}}
    ])

@pytest.fixture
def mock_process_chunks(mocker):
    """Fixture to mock process_chunks."""
    return mocker.patch("App_Function_Libraries.RAG.ChromaDB_Library.process_chunks")

@pytest.fixture
def mock_create_embeddings_batch(mocker):
    """Fixture to mock create_embeddings_batch."""
    return mocker.patch("App_Function_Libraries.RAG.ChromaDB_Library.create_embeddings_batch", return_value=[[0.1, 0.2], [0.3, 0.4]])

@pytest.fixture
def mock_situate_context(mocker):
    """Fixture to mock situate_context."""
    return mocker.patch("App_Function_Libraries.RAG.ChromaDB_Library.situate_context", return_value="Context for chunk")


@patch('App_Function_Libraries.RAG.ChromaDB_Library.chroma_client')
@patch('App_Function_Libraries.RAG.ChromaDB_Library.create_embeddings_batch')
@patch('App_Function_Libraries.RAG.ChromaDB_Library.situate_context')
@patch('App_Function_Libraries.RAG.ChromaDB_Library.chunk_for_embedding')
@patch('App_Function_Libraries.RAG.ChromaDB_Library.process_chunks')
def test_process_and_store_content(mock_process_chunks, mock_chunk_for_embedding, mock_situate_context,
                                   mock_create_embeddings_batch, mock_chroma_client):
    mock_database = MagicMock()
    mock_chunk_for_embedding.return_value = [{
        'text': 'Chunk 1',
        'metadata': {
            'start_index': 0,
            'end_index': 10,
            'file_name': 'test.mp4',
            'relative_position': 0.5
        }
    }]
    mock_situate_context.return_value = "Contextualized chunk"
    mock_create_embeddings_batch.return_value = [[0.1, 0.2, 0.3]]
    mock_collection = MagicMock()
    mock_chroma_client.get_collection.side_effect = Exception("Collection not found")
    mock_chroma_client.create_collection.return_value = mock_collection

    process_and_store_content(
        database=mock_database,
        content="Test Content",
        collection_name="test_collection",
        media_id=1,
        file_name="test.mp4",
        create_embeddings=True,
        create_contextualized=True
    )

    mock_chunk_for_embedding.assert_called_once()
    mock_process_chunks.assert_called_once()
    mock_situate_context.assert_called_once()
    mock_create_embeddings_batch.assert_called_once()

    # Check if get_collection was called
    mock_chroma_client.get_collection.assert_called_once_with(name="test_collection")

    # Check if create_collection was called after get_collection raised an exception
    mock_chroma_client.create_collection.assert_called_once_with(name="test_collection")

    mock_collection.upsert.assert_called_once()

    # Check for both execute_query calls
    assert mock_database.execute_query.call_count == 2
    mock_database.execute_query.assert_any_call('UPDATE Media SET vector_processing = 1 WHERE id = ?', (1,))
    mock_database.execute_query.assert_any_call('INSERT OR REPLACE INTO media_fts (rowid, title, content) SELECT id, title, content FROM Media WHERE id = ?', (1,))

##############################
# Test: check_embedding_status
##############################

@patch('App_Function_Libraries.RAG.ChromaDB_Library.chroma_client')
def test_check_embedding_status(mock_chroma_client):
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    mock_collection.get.return_value = {'ids': ['id1', 'id2'],
                                        'embeddings': [[0.1, 0.2], [0.3, 0.4]],
                                        'metadatas': [{"key1": "value1"}, {"key2": "value2"}]}

    status, details = check_embedding_status("Test Item", {"Test Item": 1})

    assert "Embedding exists" in status, f"Expected embedding to exist, got status: {status}"
    mock_chroma_client.get_or_create_collection.assert_called_once_with(name="all_content_embeddings")

##############################
# Test: reset_chroma_collection
##############################

@patch('App_Function_Libraries.RAG.ChromaDB_Library.chroma_client')
def test_reset_chroma_collection(mock_chroma_client):
    reset_chroma_collection("test_collection")

    mock_chroma_client.delete_collection.assert_called_once_with("test_collection")
    mock_chroma_client.create_collection.assert_called_once_with("test_collection")

##############################
# Test: store_in_chroma
##############################

@patch('App_Function_Libraries.RAG.ChromaDB_Library.chroma_client')
def test_store_in_chroma(mock_chroma_client):
    mock_collection = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection
    mock_collection.get.return_value = {
        'ids': ['id1', 'id2'],
        'embeddings': [[0.1, 0.2], [0.3, 0.4]],
        'metadatas': [{"key1": "value1"}, {"key2": "value2"}],
        'documents': ["Text 1", "Text 2"]
    }

    store_in_chroma(
        collection_name="test_collection",
        texts=["Text 1", "Text 2"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        ids=["id1", "id2"],
        metadatas=[{"key1": "value1"}, {"key2": "value2"}]
    )

    mock_chroma_client.get_collection.assert_called_once_with(name="test_collection")
    mock_collection.upsert.assert_called_once_with(
        documents=["Text 1", "Text 2"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        ids=["id1", "id2"],
        metadatas=[{"key1": "value1"}, {"key2": "value2"}]
    )

##############################
# Test: vector_search
##############################

@pytest.fixture
def mock_create_embedding(mocker):
    """Fixture to mock create_embedding."""
    return mocker.patch("App_Function_Libraries.RAG.ChromaDB_Library.create_embedding", return_value=[0.1, 0.2])

@patch('App_Function_Libraries.RAG.ChromaDB_Library.chroma_client')
@patch('App_Function_Libraries.RAG.ChromaDB_Library.create_embedding')
def test_vector_search(mock_create_embedding, mock_chroma_client):
    mock_collection = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection
    mock_collection.get.return_value = {
        'metadatas': [{'embedding_model': 'test_model', 'embedding_provider': 'test_provider'}]
    }
    mock_collection.query.return_value = {
        'documents': [["Document 1"]],
        'metadatas': [{"metadata1": "value1"}]
    }
    mock_create_embedding.return_value = [0.1, 0.2, 0.3]

    results = vector_search("test_collection", "query text")

    mock_chroma_client.get_collection.assert_called_once_with(name="test_collection")
    mock_collection.get.assert_called_once_with(limit=10, include=["metadatas"])
    mock_create_embedding.assert_called_once_with("query text", 'test_provider', 'test_model', embedding_api_url)
    mock_collection.query.assert_called_once()

    assert len(results) == 1
    assert results[0]['content'] == "Document 1"
    assert results[0]['metadata'] == "metadata1"

##############################
# Parametrized Test: batched
##############################

@pytest.mark.parametrize("iterable, batch_size, expected_batches", [
    ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]),
    ([1, 2], 3, [[1, 2]]),
    ([], 2, [])
])
def test_batched(iterable, batch_size, expected_batches):
    batches = list(batched(iterable, batch_size))
    assert batches == expected_batches

#
# End of File
####################################################################################################
