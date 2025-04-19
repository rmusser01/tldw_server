# tests/test_search_functions.py
#
# Updated import statement
from PoC_Version.App_Function_Libraries.DB.DB_Manager import search_media_db, search_media_database
#
#
####################################################################################################
# Test Status:
# Working as of 2024-10-01

import pytest
from unittest.mock import patch, MagicMock
import sqlite3


# Modify the functions to accept a connection parameter for testing
@pytest.fixture
def mock_db():
    """Create a mock database for search tests"""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.connection = conn
    return conn, cursor


def test_search_media_db(mock_db):
    conn, cursor = mock_db
    test_results = [
        (1, 'http://example.com', 'Test Title', 'video', 'content', 'author', '2023-01-01', 'prompt', 'summary')
    ]

    # Set up cursor mock
    cursor.fetchall.return_value = test_results

    with patch('App_Function_Libraries.DB.DB_Manager.Database.get_connection') as mock_get_conn:
        with patch('App_Function_Libraries.DB.DB_Manager.Database.execute_query') as mock_execute:
            # Configure mocks
            mock_get_conn.return_value.__enter__.return_value = conn
            mock_execute.return_value = test_results

            # Execute test
            results = search_media_db('Test', ['title'], '')

            # Verify results
            assert len(results) == 1
            assert results[0][2] == 'Test Title'

            # Verify SQL query
            actual_query = cursor.execute.call_args[0][0]
            actual_params = cursor.execute.call_args[0][1]
            assert 'SELECT DISTINCT Media.id, Media.url, Media.title' in actual_query
            assert 'WHERE Media.title LIKE ?' in actual_query
            assert '%Test%' in actual_params


def test_search_media_db_with_keywords(mock_db):
    conn, cursor = mock_db
    test_results = [
        (1, 'http://example.com', 'Test Title', 'video', 'content', 'author', '2023-01-01', 'prompt', 'summary')
    ]

    cursor.fetchall.return_value = test_results

    with patch('App_Function_Libraries.DB.DB_Manager.Database.get_connection') as mock_get_conn:
        with patch('App_Function_Libraries.DB.DB_Manager.Database.execute_query') as mock_execute:
            mock_get_conn.return_value.__enter__.return_value = conn
            mock_execute.return_value = test_results

            results = search_media_db('Test', ['title'], 'keyword1,keyword2')

            assert len(results) == 1
            actual_query = cursor.execute.call_args[0][0]
            actual_params = cursor.execute.call_args[0][1]
            assert 'EXISTS (SELECT 1 FROM MediaKeywords mk JOIN Keywords k ON mk.keyword_id = k.id' in actual_query
            assert '%keyword1%' in actual_params
            assert '%keyword2%' in actual_params


def test_search_media_db_pagination(mock_db):
    conn, cursor = mock_db

    page1_results = [
        (1, 'http://example1.com', 'First Title', 'video', 'content1', 'author1', '2023-01-01', 'prompt1', 'summary1')]
    page2_results = [(2, 'http://example2.com', 'Second Title', 'article', 'content2', 'author2', '2023-01-02',
                      'prompt2', 'summary2')]

    with patch('App_Function_Libraries.DB.DB_Manager.Database.get_connection') as mock_get_conn:
        with patch('App_Function_Libraries.DB.DB_Manager.Database.execute_query') as mock_execute:
            mock_get_conn.return_value.__enter__.return_value = conn
            mock_execute.side_effect = [page1_results, page2_results]
            cursor.fetchall.side_effect = [page1_results, page2_results]

            results_page_1 = search_media_db('', ['title'], '', page=1, results_per_page=1)
            results_page_2 = search_media_db('', ['title'], '', page=2, results_per_page=1)

            assert len(results_page_1) == 1
            assert len(results_page_2) == 1
            assert results_page_1[0][2] == 'First Title'
            assert results_page_2[0][2] == 'Second Title'
            assert results_page_1 != results_page_2


def test_search_media_database(mock_db):
    conn, cursor = mock_db
    test_results = [
        (1, 'Test Title', 'http://example.com')
    ]

    with patch('App_Function_Libraries.DB.DB_Manager.Database.get_connection') as mock_get_conn:
        with patch('App_Function_Libraries.DB.DB_Manager.Database.execute_query') as mock_execute:
            mock_get_conn.return_value.__enter__.return_value = conn
            mock_execute.return_value = test_results
            cursor.fetchall.return_value = test_results

            results = search_media_database('Test')

            assert len(results) == 1
            assert results[0] == (1, 'Test Title', 'http://example.com')
            actual_query = cursor.execute.call_args[0][0]
            actual_params = cursor.execute.call_args[0][1]
            assert 'SELECT id, title, url FROM Media WHERE title LIKE ?' in actual_query
            assert '%Test%' in actual_params


def test_search_media_database_error(mock_db):
    conn, cursor = mock_db
    test_error = sqlite3.Error("Test database error")

    with patch('App_Function_Libraries.DB.DB_Manager.Database.get_connection') as mock_get_conn:
        with patch('App_Function_Libraries.DB.DB_Manager.Database.execute_query') as mock_execute:
            mock_get_conn.return_value.__enter__.return_value = conn
            mock_execute.side_effect = test_error
            cursor.execute.side_effect = test_error

            with pytest.raises(Exception) as exc_info:
                search_media_database('Test')

            assert str(exc_info.value) == "Error searching media database: Test database error"


def test_search_media_db_invalid_page():
    with pytest.raises(ValueError, match="Page number must be 1 or greater."):
        search_media_db('Test', ['title'], '', page=0, results_per_page=10)

#
# End of File
####################################################################################################
