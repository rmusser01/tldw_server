# Tests/RAG/test_rag_functions.py
import configparser
import os
import sys
import pytest
#from unittest.mock import MagicMock
#from typing import List, Dict, Any

# Adjust the path to the parent directory of App_Function_Libraries
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)

# Import the functions to test
from PoC_Version.App_Function_Libraries.RAG.RAG_Library_2 import (
    fetch_relevant_media_ids,
    #perform_vector_search,
    perform_full_text_search,
    enhanced_rag_pipeline,
    #enhanced_rag_pipeline_chat,
    generate_answer,
    #fetch_relevant_chat_ids,
    #fetch_all_chat_ids,
    #filter_results_by_keywords,
    #extract_media_id_from_result
)


def test_fetch_relevant_media_ids_success(mocker):
    """Test fetch_relevant_media_ids with valid keywords."""
    mock_fetch_keywords_for_media = mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.fetch_keywords_for_media',
        side_effect=lambda keyword: {
            'geography': [1, 2],
            'cities': [2, 3, 4]
        }.get(keyword, [])
    )

    keywords = ['geography', 'cities']
    result = fetch_relevant_media_ids(keywords)
    assert sorted(result) == [1, 2, 3, 4]

    mock_fetch_keywords_for_media.assert_any_call('geography')
    mock_fetch_keywords_for_media.assert_any_call('cities')
    assert mock_fetch_keywords_for_media.call_count == 2


def test_perform_full_text_search_with_relevant_ids(mocker):
    """Test perform_full_text_search with relevant_ids provided."""
    # Create a transformed response matching the expected format
    transformed_response = [
        {'content': 'Full text document 1', 'metadata': {'media_id': 1}},
        {'content': 'Full text document 3', 'metadata': {'media_id': 3}},
    ]

    # Mock the search functions mapping
    search_function_mock = lambda query, fts_top_k, relevant_ids: transformed_response
    search_functions_mock = {
        "Media DB": search_function_mock
    }
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.search_functions', search_functions_mock)

    query = 'full text query'
    database_type = "Media DB"
    relevant_ids = "1,3"

    result = perform_full_text_search(query, database_type, relevant_ids)

    expected = [
        {'content': 'Full text document 1', 'metadata': {'media_id': 1}},
        {'content': 'Full text document 3', 'metadata': {'media_id': 3}},
    ]
    assert result == expected


def test_perform_full_text_search_without_relevant_ids(mocker):
    """Test perform_full_text_search without relevant_ids."""
    # Create a transformed response matching the expected format
    transformed_response = [
        {'content': 'Full text document 1', 'metadata': {'media_id': 1}},
        {'content': 'Full text document 2', 'metadata': {'media_id': 2}},
    ]

    # Mock the search functions mapping
    search_function_mock = lambda query, fts_top_k, relevant_ids: transformed_response
    search_functions_mock = {
        "Media DB": search_function_mock
    }
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.search_functions', search_functions_mock)

    query = 'full text query'
    database_type = "Media DB"
    relevant_ids = ""

    result = perform_full_text_search(query, database_type, relevant_ids)

    expected = [
        {'content': 'Full text document 1', 'metadata': {'media_id': 1}},
        {'content': 'Full text document 2', 'metadata': {'media_id': 2}},
    ]
    assert result == expected


@pytest.mark.parametrize("database_type,search_module_path,mock_response", [
    (
        "Media DB",
        'App_Function_Libraries.DB.SQLite_DB.search_media_db',
        [{'content': 'Media DB document 1', 'metadata': {'media_id': '1'}}]
    ),
    (
        "RAG Chat",
        'App_Function_Libraries.DB.RAG_QA_Chat_DB.search_rag_chat',
        [{'content': 'RAG Chat document 1', 'metadata': {'media_id': '1'}}]
    ),
    (
        "RAG Notes",
        'App_Function_Libraries.DB.RAG_QA_Chat_DB.search_rag_notes',
        [{'content': 'RAG Notes document 1', 'metadata': {'media_id': '1'}}]
    ),
    (
        "Character Chat",
        'App_Function_Libraries.DB.Character_Chat_DB.search_character_chat',
        [{'content': 'Character Chat document 1', 'metadata': {'media_id': '1'}}]
    ),
    (
        "Character Cards",
        'App_Function_Libraries.DB.Character_Chat_DB.search_character_cards',
        [{'content': 'Character Cards document 1', 'metadata': {'media_id': '1'}}]
    )
])
def test_perform_full_text_search_different_db_types(mocker, database_type, search_module_path, mock_response):
    """Test perform_full_text_search with different database types."""
    # Mock the search functions mapping with already transformed response
    search_functions_mock = {
        database_type: lambda query, fts_top_k, relevant_ids: mock_response
    }
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.search_functions', search_functions_mock)

    query = 'test query'
    relevant_ids = "1"

    result = perform_full_text_search(query, database_type, relevant_ids)
    assert result == mock_response


def test_enhanced_rag_pipeline_success(mocker):
    """Test enhanced_rag_pipeline with a successful flow."""
    # Mock config
    mock_config = configparser.ConfigParser()
    mock_config['Embeddings'] = {'provider': 'openai'}
    mock_config['chat_dictionaries'] = {'default_rag_prompt': 'You are answering a question'}
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.config', mock_config)

    # Mock metric functions
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.log_counter')
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.log_histogram')

    # Mock search results
    mock_search_results = [
        {'content': 'Test content 1', 'id': 1},
        {'content': 'Test content 2', 'id': 2}
    ]

    # Mock core functions
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.fetch_relevant_media_ids',
        return_value=[1, 2, 3]
    )
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.perform_vector_search',
        return_value=mock_search_results
    )
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.perform_full_text_search',
        return_value=mock_search_results
    )
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.generate_answer',
        return_value="Generated answer"
    )

    # Mock Ranker
    mock_ranker = mocker.Mock()
    mock_ranker.rerank.return_value = [
        {'id': 0, 'score': 0.9},
        {'id': 1, 'score': 0.8}
    ]
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.Ranker', return_value=mock_ranker)

    result = enhanced_rag_pipeline(
        query='test query',
        api_choice='OpenAI',
        keywords='keyword1,keyword2',
        database_types=["Media DB"],  # Already correct as a list
        apply_re_ranking=True
    )

    assert isinstance(result, dict)
    assert result['answer'] == "Generated answer"
    assert "Test content" in result['context']

    assert isinstance(result, dict)
    assert result['answer'] == "Generated answer"
    assert "Test content" in result['context']


def test_enhanced_rag_pipeline_error_handling(mocker):
    """Test enhanced_rag_pipeline error handling when a critical error occurs."""
    # Mock config
    mock_config = configparser.ConfigParser()
    mock_config['Embeddings'] = {'provider': 'openai'}
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.config', mock_config)

    # Mock metric functions
    mock_log_counter = mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.log_counter')

    # Mock generate_answer to raise an exception (this will trigger the main try-catch)
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.generate_answer',
        side_effect=Exception("Critical error in answer generation")
    )

    # Other mocks return empty results
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.perform_vector_search',
        return_value=[]
    )
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.perform_full_text_search',
        return_value=[]
    )
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.fetch_relevant_media_ids',
        return_value=[]
    )

    result = enhanced_rag_pipeline(
        query='test query',
        api_choice='OpenAI',
        keywords='keyword1',
        database_types=["Media DB"]
    )

    # Verify the attempt was logged
    mock_log_counter.assert_any_call(
        "enhanced_rag_pipeline_attempt",
        labels={"api_choice": "OpenAI"}
    )

    # Verify error counter was logged
    mock_log_counter.assert_any_call(
        "enhanced_rag_pipeline_error",
        labels={"api_choice": "OpenAI", "error": "Critical error in answer generation"}
    )

    # Check error response
    assert isinstance(result, dict)
    assert "An error occurred" in result['answer']
    assert result['context'] == ''


def test_enhanced_rag_pipeline_critical_error(mocker):
    """Test enhanced_rag_pipeline with a critical error that should stop execution."""
    # Mock config
    mock_config = configparser.ConfigParser()
    mock_config['Embeddings'] = {'provider': 'openai'}
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.config', mock_config)

    # Mock ALL functions to raise exceptions
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.fetch_relevant_media_ids',
        side_effect=Exception("Database connection failed")
    )
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.perform_vector_search',
        side_effect=Exception("Vector search failed")
    )
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.perform_full_text_search',
        side_effect=Exception("Full-text search failed")
    )
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.generate_answer',
        side_effect=Exception("Answer generation failed")
    )

    result = enhanced_rag_pipeline(
        query='test query',
        api_choice='OpenAI',
        keywords='keyword1',
        database_types=["Media DB"]
    )

    # Check error response
    assert isinstance(result, dict)
    assert "error" in result or "An error occurred" in result['answer']
    assert result['context'] == ""


def test_generate_answer_success(mocker):
    """Test generate_answer with successful API call."""
    # Build a test config dictionary that mimics the structure returned by load_and_log_configs()
    test_config = {
        "openai_api": {"api_key": "test_key"},  # This is what the code will look up
        "chat_dictionaries": {
            "chat_dict_RAG_prompts": "RAG_Prompts.md",
            "default_rag_prompt": "Some default prompt text or file reference"
        }
    }

    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.load_and_log_configs',
        return_value=test_config
    )

    # Mock the summarization function
    mock_summarize = mocker.patch(
        'App_Function_Libraries.Summarization.Summarization_General_Lib.summarize_with_openai',
        return_value='API response'
    )

    # Patch the function that uses the dictionary file
    mock_parse = mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.parse_user_dict_markdown_file',
        return_value={'some_key': 'some_value'}
    )

    # Now call generate_answer
    result = generate_answer('OpenAI', 'Test context', 'Test query')
    assert result == 'API response'

    # Optionally, assert that parse was called
    mock_parse.assert_called_once()


def test_enhanced_rag_pipeline_no_results(mocker):
    """Test enhanced_rag_pipeline when no results are found."""
    mock_config = configparser.ConfigParser()
    mock_config['Embeddings'] = {'provider': 'openai'}

    # Mock empty search results
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.fetch_relevant_media_ids',
        return_value=[]
    )
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.perform_vector_search',
        return_value=[]
    )
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.perform_full_text_search',
        return_value=[]
    )
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.generate_answer',
        return_value="Fallback answer"
    )


    result = enhanced_rag_pipeline(
        query='test query',
        api_choice='OpenAI',
        keywords=None,
        database_types=["Media DB"]
    )

    assert isinstance(result, dict)
    assert "No relevant information" in result['answer']
    assert "Fallback answer" in result['answer']
    assert "test query" in result['context']


if __name__ == '__main__':
    pytest.main(['-v'])