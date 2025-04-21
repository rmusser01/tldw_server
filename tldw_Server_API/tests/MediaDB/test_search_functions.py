# tests/test_search_functions.py
import pytest
import sqlite3
from typing import List, Dict, Any
import hashlib

# Adjust import paths
try:
    from app.db.database_setup import Database, DatabaseError
    from app.db.search_functions import search_media_db
    from app.db.media_functions import add_media_with_keywords
except ImportError as e:
    print(f"Error importing functions for search tests: {e}")
    raise

# Use the fixture from conftest.py
# No need for mock_db fixture here if using real DB instance

# Helper to add test data
def add_search_test_data(db: Database):
    add_media_with_keywords(url="search1", title="First Test Document", media_type="document", content="Content mentioning apple and banana.", keywords=["fruit", "test", "alpha"], prompt="p1", analysis_content="a1", author="Alice", ingestion_date="2024-01-01", db_instance=db)
    add_media_with_keywords(url="search2", title="Second Test Video", media_type="video", content="Video transcript about orange.", keywords=["fruit", "video", "beta"], prompt="p2", analysis_content="a2", author="Bob", ingestion_date="2024-01-02", db_instance=db)
    add_media_with_keywords(url="search3", title="Third Document", media_type="document", content="Another one, apple related.", keywords=["test", "gamma"], prompt="p3", analysis_content="a3", author="Alice", ingestion_date="2024-01-03", db_instance=db)
    # Add a trashed item
    media_id_trash, _ = add_media_with_keywords(url="search4_trash", title="Trashed Item", media_type="document", content="This should not appear.", keywords=["trash"], prompt="p4", analysis_content="a4", author="Charlie", ingestion_date="2024-01-04", db_instance=db)
    if media_id_trash:
         with db.transaction() as conn:
             conn.execute("UPDATE Media SET is_trash = 1 WHERE id = ?", (media_id_trash,))


def test_search_media_db_by_title_fts(memory_db_instance: Database):
    db = memory_db_instance
    add_search_test_data(db)

    results, total = search_media_db(
        search_query="Second", search_fields=['title', 'content'], keywords=[], db_instance=db
    )

    assert total == 1
    assert len(results) == 1
    assert isinstance(results[0], Dict)
    assert results[0]['title'] == "Second Test Video"
    assert results[0]['url'] == "search2"

def test_search_media_db_by_content_fts(memory_db_instance: Database):
    db = memory_db_instance
    add_search_test_data(db)

    # Search for "apple" which is in content of item 1 and 3
    results, total = search_media_db(
        search_query="apple", search_fields=['content'], keywords=[], db_instance=db
    )

    assert total == 2
    assert len(results) == 2
    # Results ordered by date desc
    assert results[0]['title'] == "Third Document"
    assert results[1]['title'] == "First Test Document"

def test_search_media_db_by_keyword_single(memory_db_instance: Database):
    db = memory_db_instance
    add_search_test_data(db)

    results, total = search_media_db(
        search_query=None, search_fields=[], keywords=["video"], db_instance=db
    )

    assert total == 1
    assert len(results) == 1
    assert results[0]['title'] == "Second Test Video"

def test_search_media_db_by_keywords_multiple_and(memory_db_instance: Database):
    db = memory_db_instance
    add_search_test_data(db)

    # Search for items with BOTH 'fruit' AND 'test'
    results, total = search_media_db(
        search_query=None, search_fields=[], keywords=["fruit", "test"], db_instance=db
    )

    assert total == 1
    assert len(results) == 1
    assert results[0]['title'] == "First Test Document" # Only item 1 has both

def test_search_media_db_by_keywords_no_match(memory_db_instance: Database):
    db = memory_db_instance
    add_search_test_data(db)

    results, total = search_media_db(
        search_query=None, search_fields=[], keywords=["fruit", "gamma"], db_instance=db
    )

    assert total == 0
    assert len(results) == 0

def test_search_media_db_by_author_like(memory_db_instance: Database):
    db = memory_db_instance
    add_search_test_data(db)

    # Search for author 'Alice' (case-insensitive via COLLATE NOCASE)
    results, total = search_media_db(
        search_query="aliCe", search_fields=['author'], keywords=[], db_instance=db
    )

    assert total == 2
    assert len(results) == 2
    titles = {r['title'] for r in results}
    assert "First Test Document" in titles
    assert "Third Document" in titles

def test_search_media_db_combined_fts_keyword_author(memory_db_instance: Database):
    db = memory_db_instance
    add_search_test_data(db)

    # Search content for 'apple', keyword 'test', author 'Alice'
    results, total = search_media_db(
        search_query="apple", search_fields=['content', 'author'], keywords=["test"], db_instance=db
    )
    # Item 1: content=apple, kw=test, author=Alice -> MATCH
    # Item 3: content=apple, kw=test, author=Alice -> MATCH
    # We search author via LIKE, content via FTS, keyword via JOIN/HAVING

    assert total == 2 # Should match item 1 and 3
    assert len(results) == 2
    titles = {r['title'] for r in results}
    assert "First Test Document" in titles
    assert "Third Document" in titles


def test_search_media_db_pagination(memory_db_instance: Database):
    db = memory_db_instance
    add_search_test_data(db) # Adds 3 non-trashed items

    # Get page 1, 2 items per page
    results_p1, total_p1 = search_media_db(
        search_query=None, search_fields=[], keywords=[],
        page=1, results_per_page=2, db_instance=db
    )
    # Get page 2, 2 items per page
    results_p2, total_p2 = search_media_db(
        search_query=None, search_fields=[], keywords=[],
        page=2, results_per_page=2, db_instance=db
    )

    assert total_p1 == 3
    assert total_p2 == 3
    assert len(results_p1) == 2
    assert len(results_p2) == 1

    # Check content based on default date ordering (desc)
    assert results_p1[0]['title'] == "Third Document" # Newest
    assert results_p1[1]['title'] == "Second Test Video"
    assert results_p2[0]['title'] == "First Test Document" # Oldest

def test_search_media_db_excludes_trashed(memory_db_instance: Database):
    db = memory_db_instance
    add_search_test_data(db) # Includes one trashed item

    # Search for anything, should not include the trashed item
    results, total = search_media_db(
        search_query="document", search_fields=['title', 'content', 'type'], keywords=[], db_instance=db
    )

    # Total matches should be 3 (items 1, 3)
    assert total == 2 # Actually, only item 1 and 3 are type 'document'
    assert len(results) == 2
    for r in results:
        assert r['title'] != "Trashed Item"


# Remove tests for old search_media_database function
# def test_search_media_database(mock_db): ...
# def test_search_media_database_error(mock_db): ...