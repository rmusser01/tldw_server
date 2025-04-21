# tests/test_keyword_functions.py
import hashlib
import logging
import os
import pytest
import sqlite3
from typing import List, Tuple, Any, Dict

from tldw_Server_API.app.core.DB_Management.DB_Manager import get_all_document_versions, get_document_version

# Adjust import paths
try:
    from app.db.database_setup import Database, DatabaseError, InputError
    # Assuming keyword/media functions are separated
    from app.db.keyword_functions import (
        add_keyword, delete_keyword, fetch_all_keywords,
        fetch_keywords_for_media, update_keywords_for_media
        # remove keywords_browser_interface, display_keywords, export_keywords_to_csv if they are UI/CLI helpers
    )
    from app.db.media_functions import add_media_with_keywords
    # from app.db.article_ingestion import ingest_article_to_db # If testing this too
except ImportError as e:
    print(f"Error importing functions for keyword tests: {e}")
    raise

# Use the fixture from conftest.py
# No need for mock_db or mock_get_connection fixtures here anymore

# --- Test add_keyword ---
def test_add_keyword_success(memory_db_instance: Database):
    result = add_keyword("test_keyword", db_instance=memory_db_instance)
    assert isinstance(result, int) # Assuming it returns the keyword ID
    assert result > 0

    # Verify in DB
    with memory_db_instance.transaction() as conn:
        cursor=conn.cursor()
        cursor.execute("SELECT keyword FROM Keywords WHERE id = ?", (result,))
        kw = cursor.fetchone()
        assert kw is not None
        assert kw['keyword'] == "test_keyword"

def test_add_keyword_duplicate(memory_db_instance: Database):
    kw = "duplicate_test"
    id1 = add_keyword(kw, db_instance=memory_db_instance)
    id2 = add_keyword(kw, db_instance=memory_db_instance) # Should ignore or return existing ID
    assert id1 == id2 # Assuming INSERT OR IGNORE, IDs should match

def test_add_keyword_empty_or_none(memory_db_instance: Database):
    with pytest.raises((DatabaseError, ValueError, InputError)): # Depending on function validation
        add_keyword("", db_instance=memory_db_instance)
    with pytest.raises((DatabaseError, ValueError, InputError, TypeError)):
        add_keyword(None, db_instance=memory_db_instance)

# --- Test delete_keyword ---
def test_delete_keyword_success(memory_db_instance: Database):
    kw = "to_delete"
    add_keyword(kw, db_instance=memory_db_instance)
    result = delete_keyword(kw, db_instance=memory_db_instance)

    # Adjust assertion based on actual return value (e.g., dict or bool)
    if isinstance(result, Dict):
        assert "deleted successfully" in result.get('success', '')
    else:
        assert result is True # Or whatever success looks like

    # Verify deleted
    with memory_db_instance.transaction() as conn:
        cursor=conn.cursor()
        cursor.execute("SELECT id FROM Keywords WHERE keyword = ?", (kw,))
        assert cursor.fetchone() is None

def test_delete_keyword_not_found(memory_db_instance: Database):
    result = delete_keyword("nonexistent_keyword", db_instance=memory_db_instance)
    if isinstance(result, Dict):
        assert "not found" in result.get('error', result.get('message', ''))
    else:
        assert result is False # Or handle potential exception

# --- Test fetch_all_keywords ---
def test_fetch_all_keywords(memory_db_instance: Database):
    add_keyword("test1", db_instance=memory_db_instance)
    add_keyword("test2", db_instance=memory_db_instance)
    keywords = fetch_all_keywords(db_instance=memory_db_instance)
    assert isinstance(keywords, list)
    # The function likely returns a list of strings
    assert "test1" in keywords
    assert "test2" in keywords
    assert len(keywords) >= 2

# --- Test fetch_keywords_for_media ---
def test_fetch_keywords_for_media(memory_db_instance: Database):
    media_id, _ = add_media_with_keywords(
        url="http://fetch.com", title="Fetch Test", media_type="article",
        content="Fetch content", keywords=["fetch_kw1", "test_kw"], # Pass as list
        prompt="p", analysis_content="a", transcription_model=None, author="au",
        ingestion_date="2023-06-01", db_instance=memory_db_instance
    )
    keywords = fetch_keywords_for_media(media_id, db_instance=memory_db_instance)
    assert isinstance(keywords, list)
    assert "fetch_kw1" in keywords
    assert "test_kw" in keywords
    assert len(keywords) == 2

def test_fetch_keywords_for_media_no_keywords(memory_db_instance: Database):
     media_id, _ = add_media_with_keywords(
         url="http://fetch_no_kw.com", title="Fetch No KW Test", media_type="article",
         content="Fetch content no kw", keywords=[], # Empty list
         prompt="p", analysis_content="a", transcription_model=None, author="au",
         ingestion_date="2023-06-01", db_instance=memory_db_instance
     )
     keywords = fetch_keywords_for_media(media_id, db_instance=memory_db_instance)
     assert isinstance(keywords, list)
     # Should it return empty list or ['default']? Check function implementation.
     # Assuming empty list if none explicitly added beyond potential default logic.
     assert len(keywords) == 0 or keywords == ['default'] # Check which applies


# --- Test update_keywords_for_media ---
def test_update_keywords_for_media(memory_db_instance: Database):
    media_id, _ = add_media_with_keywords(
        url="http://update.com", title="Update Test", media_type="article",
        content="Update content", keywords=["initial1", "initial2"], # Start with some KWs
        prompt="p", analysis_content="a", transcription_model=None, author="au",
        ingestion_date="2023-06-01", db_instance=memory_db_instance
    )

    # Update with new keywords
    new_keywords = ["updated1", "updated2", "updated3"]
    result = update_keywords_for_media(media_id, new_keywords, db_instance=memory_db_instance)

    # Assuming update_keywords_for_media returns success message/bool
    assert result is True or "updated successfully" in str(result) # Check success indication

    # Verify the keywords in the database
    fetched_keywords = fetch_keywords_for_media(media_id, db_instance=memory_db_instance)
    assert isinstance(fetched_keywords, list)
    assert sorted(fetched_keywords) == sorted(new_keywords) # Compare contents ignoring order
    assert "initial1" not in fetched_keywords

def test_update_keywords_for_media_empty_list(memory_db_instance: Database):
     media_id, _ = add_media_with_keywords(
         url="http://update_empty.com", title="Update Empty Test", media_type="article",
         content="Update content empty", keywords=["kw1"], # Start with one
         prompt="p", analysis_content="a", transcription_model=None, author="au",
         ingestion_date="2023-06-01", db_instance=memory_db_instance
     )
     result = update_keywords_for_media(media_id, [], db_instance=memory_db_instance) # Update with empty list
     assert result is True or "updated successfully" in str(result)

     fetched_keywords = fetch_keywords_for_media(media_id, db_instance=memory_db_instance)
     assert len(fetched_keywords) == 0 # Should remove all keywords


# === Tests for add_media_with_keywords specific logic ===

def test_add_media_with_keywords_basic(memory_db_instance: Database):
    url = "http://addbasic.com"
    title = "Add Basic"
    content = "Basic content"
    keywords = ["basic_kw", "test"]
    prompt = "Basic prompt"
    analysis = "Basic analysis"
    hash_expected = hashlib.sha256(content.encode()).hexdigest()

    media_id, message = add_media_with_keywords(
        url=url, title=title, media_type="text", content=content, keywords=keywords,
        prompt=prompt, analysis_content=analysis, transcription_model="N/A",
        author="Tester", ingestion_date="2024-01-10", db_instance=memory_db_instance
    )

    assert isinstance(media_id, int)
    assert "added successfully" in message

    # Verify Media table
    with memory_db_instance.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Media WHERE id = ?", (media_id,))
        media_row = cursor.fetchone()
        assert media_row is not None
        assert media_row['url'] == url
        assert media_row['title'] == title
        assert media_row['content'] == content
        assert media_row['content_hash'] == hash_expected
        assert media_row['chunking_status'] == 'pending' # Check default status

    # Verify Keywords
    fetched_kws = fetch_keywords_for_media(media_id, memory_db_instance)
    assert sorted(fetched_kws) == sorted(keywords)

    # Verify Document Version 1
    version1 = get_document_version(media_id, 1, db_instance=memory_db_instance)
    assert version1 is not None
    assert version1['content'] == content
    assert version1['prompt'] == prompt
    assert version1['analysis_content'] == analysis


def test_add_media_with_keywords_overwrite(memory_db_instance: Database):
    url = "http://overwrite.com"
    # Add initial version
    media_id1, _ = add_media_with_keywords(
        url=url, title="Overwrite V1", media_type="text", content="Content V1",
        keywords=["v1"], prompt="P1", analysis_content="A1", transcription_model="N/A",
        author="Tester", ingestion_date="2024-01-10", db_instance=memory_db_instance
    )
    # Add second time with overwrite=True
    content_v2 = "Content V2 - Updated"
    keywords_v2 = ["v2_kw"]
    prompt_v2 = "Prompt V2"
    hash_v2 = hashlib.sha256(content_v2.encode()).hexdigest()

    media_id2, message = add_media_with_keywords(
        url=url, title="Overwrite V2", media_type="text", content=content_v2,
        keywords=keywords_v2, prompt=prompt_v2, analysis_content="A2", transcription_model="N/A",
        author="Tester", ingestion_date="2024-01-11",
        overwrite=True, db_instance=memory_db_instance
    )

    assert media_id1 == media_id2 # Should update the same record
    assert "updated successfully" in message

    # Verify Media table updated
    with memory_db_instance.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT title, content, content_hash, ingestion_date FROM Media WHERE id = ?", (media_id2,))
        media_row = cursor.fetchone()
        assert media_row is not None
        assert media_row['title'] == "Overwrite V2"
        assert media_row['content'] == content_v2
        assert media_row['content_hash'] == hash_v2
        assert media_row['ingestion_date'] == "2024-01-11"

    # Verify Keywords updated (old ones removed)
    fetched_kws = fetch_keywords_for_media(media_id2, memory_db_instance)
    assert fetched_kws == keywords_v2

    # Verify *New* Version created (version 2)
    version2 = get_document_version(media_id2, 2, db_instance=memory_db_instance)
    assert version2 is not None
    assert version2['content'] == content_v2
    assert version2['prompt'] == prompt_v2
    assert version2['analysis_content'] == "A2"

    # Verify Version 1 still exists (history kept)
    version1 = get_document_version(media_id2, 1, db_instance=memory_db_instance)
    assert version1 is not None
    assert version1['content'] == "Content V1"


def test_add_media_with_keywords_no_overwrite(memory_db_instance: Database):
     url = "http://no_overwrite.com"
     # Add initial version
     media_id1, _ = add_media_with_keywords(
         url=url, title="NoOverwrite V1", media_type="text", content="Content V1",
         keywords=["v1"], prompt="P1", analysis_content="A1", transcription_model="N/A",
         author="Tester", ingestion_date="2024-01-10", db_instance=memory_db_instance
     )
     # Add second time with overwrite=False (default)
     media_id2, message = add_media_with_keywords(
         url=url, title="NoOverwrite V2", media_type="text", content="Content V2",
         keywords=["v2"], prompt="P2", analysis_content="A2", transcription_model="N/A",
         author="Tester", ingestion_date="2024-01-11",
         overwrite=False, db_instance=memory_db_instance # Explicitly false
     )
     assert media_id1 == media_id2 # Should return the existing ID
     assert "already exists and was not overwritten" in message

     # Verify Media table NOT updated
     with memory_db_instance.transaction() as conn:
         cursor = conn.cursor()
         cursor.execute("SELECT title, content FROM Media WHERE id = ?", (media_id1,))
         media_row = cursor.fetchone()
         assert media_row is not None
         assert media_row['title'] == "NoOverwrite V1" # Still V1 title
         assert media_row['content'] == "Content V1"

     # Verify only one version exists
     versions = get_all_document_versions(media_id1, db_instance=memory_db_instance)
     assert len(versions) == 1
     assert versions[0]['version_number'] == 1

# Remove tests for ingest_article_to_db unless that function is also updated and used
# Remove tests for UI helpers unless needed: keywords_browser_interface, display_keywords, export_keywords_to_csv