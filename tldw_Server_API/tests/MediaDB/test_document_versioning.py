# tests/test_document_versioning.py
import logging

import pytest
import sqlite3
from typing import Dict, Any

# Adjust import paths based on your project structure
try:
    from tldw_Server_API.app.core.DB_Management.Media_DB import (
        Database,
        DatabaseError,
        create_document_version,
        get_document_version,
        get_all_document_versions,
        delete_document_version,
        rollback_to_version,
    )
except ImportError as e:
    print(f"Error importing versioning functions: {e}")
    raise

# Use the fixture from conftest.py
# No need for mock_db or mock_get_connection fixtures here anymore


def insert_test_media(db: Database, media_id: int = 1, title: str = 'Test Doc') -> int:
    """Helper to insert a basic media record."""
    url = f"http://test.com/{media_id}"
    content = f"Initial content for {media_id}"
    # Ensure hashlib is imported if not already
    import hashlib
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    try:
        # Use the transaction context manager from the db instance
        with db.transaction() as conn: # Make sure to use the passed 'db' instance
            conn.execute("""
                INSERT OR IGNORE INTO Media (id, url, title, type, content, content_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (media_id, url, title, 'document', content, content_hash))
            # Logging here should use db.db_path_str if needed
            # logging.debug(f"Inserted test media {media_id} into {db.db_path_str}") # Example
        return media_id
    except Exception as e:
        # Use db.db_path_str in logging
        logging.error(f"Error inserting test media {media_id} into {db.db_path_str}: {e}", exc_info=True)
        # Re-raise or handle as needed for test clarity
        raise # Re-raise the exception so the test knows something went wrong here

# Import hashlib if needed for helper
import hashlib


# === Test create_document_version ===

def test_create_first_version(memory_db_instance: Database):
    db = memory_db_instance
    media_id = insert_test_media(db, 1)
    content = "Version 1 content"
    prompt = "Prompt for v1"
    analysis = "Analysis for v1"

    result_dict = create_document_version(
        media_id=media_id,
        content=content,
        prompt=prompt,
        analysis_content=analysis,
        db_instance=db
    )

    assert isinstance(result_dict, Dict)
    assert result_dict['media_id'] == media_id
    assert result_dict['version_number'] == 1 # First version

    # Verify DB state
    with db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT content, prompt, analysis_content
            FROM DocumentVersions
            WHERE media_id = ? AND version_number = ?""",
            (media_id, 1)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row['content'] == content
        assert row['prompt'] == prompt
        assert row['analysis_content'] == analysis

def test_create_subsequent_version(memory_db_instance: Database):
    db = memory_db_instance
    media_id = insert_test_media(db, 2)
    # Create first version
    create_document_version(media_id=media_id, content="V1", db_instance=db)
    # Create second version
    content2 = "Version 2 content"
    result_dict = create_document_version(media_id=media_id, content=content2, prompt="P2", db_instance=db)

    assert result_dict['version_number'] == 2

    # Verify DB state for version 2
    with db.transaction() as conn:
        cursor = conn.cursor()
        # Add analysis_content to the SELECT
        cursor.execute(
            "SELECT content, prompt, analysis_content FROM DocumentVersions WHERE media_id = ? AND version_number = ?",
            (media_id, 2))
        row = cursor.fetchone()
        assert row is not None
        assert row['content'] == content2
        assert row['prompt'] == "P2"
        assert row['analysis_content'] is None  # Now this key exists

# Add test for creating version with non-existent media ID (should fail due to FOREIGN KEY)
# def test_create_version_invalid_media_id(memory_db_instance: Database):
#     with pytest.raises(DatabaseError): # Expect DatabaseError wrapping sqlite3.IntegrityError
#         create_document_version(media_id=999, content="Test", db_instance=memory_db_instance)

# === Test get_document_version ===

def test_get_document_version_latest(memory_db_instance: Database):
    db = memory_db_instance
    media_id = insert_test_media(db, 3)
    create_document_version(media_id=media_id, content="V1 Content", prompt="P1", analysis_content="A1", db_instance=db)
    create_document_version(media_id=media_id, content="V2 Content", prompt="P2", analysis_content="A2", db_instance=db)

    result = get_document_version(media_id=media_id, db_instance=db) # version_number=None gets latest

    assert result is not None
    assert result['version_number'] == 2
    assert result['content'] == "V2 Content"
    assert result['prompt'] == "P2"
    assert result['analysis_content'] == "A2"

def test_get_document_version_specific(memory_db_instance: Database):
    db = memory_db_instance
    media_id = insert_test_media(db, 4)
    create_document_version(media_id=media_id, content="V1 Content", prompt="P1", analysis_content="A1", db_instance=db)
    create_document_version(media_id=media_id, content="V2 Content", prompt="P2", analysis_content="A2", db_instance=db)

    result = get_document_version(media_id=media_id, version_number=1, db_instance=db)

    assert result is not None
    assert result['version_number'] == 1
    assert result['content'] == "V1 Content"
    assert result['prompt'] == "P1"
    assert result['analysis_content'] == "A1"

def test_get_document_version_without_content(memory_db_instance: Database):
    db = memory_db_instance
    media_id = insert_test_media(db, 5)
    create_document_version(media_id=media_id, content="V1 Content", prompt="P1", analysis_content="A1", db_instance=db)

    result = get_document_version(media_id=media_id, version_number=1, include_content=False, db_instance=db)

    assert result is not None
    assert result['version_number'] == 1
    assert 'content' not in result # Content should be excluded
    assert result['prompt'] == "P1"

def test_get_document_version_not_found(memory_db_instance: Database):
    db = memory_db_instance
    media_id = insert_test_media(db, 6)
    # No versions created yet
    result_latest = get_document_version(media_id=media_id, db_instance=db)
    result_specific = get_document_version(media_id=media_id, version_number=5, db_instance=db)

    assert result_latest is None
    assert result_specific is None

# === Test get_all_document_versions ===

def test_get_all_document_versions(memory_db_instance: Database):
    db = memory_db_instance
    media_id = insert_test_media(db, 7)
    create_document_version(media_id=media_id, content="V1", prompt="P1", analysis_content="A1", db_instance=db)
    create_document_version(media_id=media_id, content="V2", prompt="P2", analysis_content="A2", db_instance=db)

    results = get_all_document_versions(media_id=media_id, include_content=True, db_instance=db)

    assert len(results) == 2
    # Results are ordered DESC by version_number
    assert results[0]['version_number'] == 2
    assert results[0]['content'] == "V2"  # This assertion should now pass
    assert results[0]['prompt'] == "P2"
    assert results[0]['analysis_content'] == "A2"
    assert results[1]['version_number'] == 1
    assert results[1]['content'] == "V1"
    assert results[1]['prompt'] == "P1"
    assert results[1]['analysis_content'] == "A1"

def test_get_all_document_versions_pagination(memory_db_instance: Database):
    db = memory_db_instance
    media_id = insert_test_media(db, 8)
    create_document_version(media_id=media_id, content="V1", db_instance=db)
    create_document_version(media_id=media_id, content="V2", db_instance=db)
    create_document_version(media_id=media_id, content="V3", db_instance=db)

    # Get page 1 (limit 2)
    results_p1 = get_all_document_versions(media_id=media_id, limit=2, offset=0, db_instance=db)
    # Get page 2 (limit 2, skip 2)
    results_p2 = get_all_document_versions(media_id=media_id, limit=2, offset=2, db_instance=db)

    assert len(results_p1) == 2
    assert results_p1[0]['version_number'] == 3 # Latest
    assert results_p1[1]['version_number'] == 2

    assert len(results_p2) == 1
    assert results_p2[0]['version_number'] == 1

# === Test delete_document_version ===

def test_delete_document_version_success(memory_db_instance: Database):
    db = memory_db_instance
    media_id = insert_test_media(db, 9)
    create_document_version(media_id=media_id, content="V1", db_instance=db)
    create_document_version(media_id=media_id, content="V2", db_instance=db) # Need > 1 version

    result = delete_document_version(media_id=media_id, version_number=1, db_instance=db)

    assert 'success' in result
    assert result['success'] == 'Version 1 deleted successfully' # Match new message

    # Verify version 1 is gone, version 2 remains
    v1 = get_document_version(media_id=media_id, version_number=1, db_instance=db)
    v2 = get_document_version(media_id=media_id, version_number=2, db_instance=db)
    assert v1 is None
    assert v2 is not None

def test_delete_last_document_version(memory_db_instance: Database):
    db = memory_db_instance
    media_id = insert_test_media(db, 10)
    create_document_version(media_id=media_id, content="V1", db_instance=db) # Only one version

    result = delete_document_version(media_id=media_id, version_number=1, db_instance=db)

    assert 'error' in result
    assert result['error'] == 'Cannot delete the last version'

    # Verify version 1 still exists
    v1 = get_document_version(media_id=media_id, version_number=1, db_instance=db)
    assert v1 is not None

def test_delete_nonexistent_version(memory_db_instance: Database):
    db = memory_db_instance
    media_id = insert_test_media(db, 11)
    create_document_version(media_id=media_id, content="V1", db_instance=db)
    create_document_version(media_id=media_id, content="V2", db_instance=db)

    result = delete_document_version(media_id=media_id, version_number=3, db_instance=db) # Version 3 doesn't exist

    assert 'error' in result
    assert result['error'] == 'Version not found'

# === Test rollback_to_version ===

def test_rollback_to_version_success(memory_db_instance: Database):
    db = memory_db_instance
    media_id = insert_test_media(db, 12)
    content1 = "Version 1 content - Original"
    content2 = "Version 2 content - Changed"
    prompt1 = "Prompt V1"
    analysis1 = "Analysis V1"

    create_document_version(media_id=media_id, content=content1, prompt=prompt1, analysis_content=analysis1, db_instance=db) # v1
    create_document_version(media_id=media_id, content=content2, db_instance=db) # v2

    # Rollback to version 1
    result = rollback_to_version(media_id=media_id, version_number=1, db_instance=db)

    assert 'success' in result
    assert result['new_version_number'] == 3 # New version created
    assert f"Successfully rolled back to version 1. State saved as new version {result['new_version_number']}." in result['success']

    # Verify latest version (v3) has content/prompt/analysis from v1
    latest_version = get_document_version(media_id=media_id, db_instance=db)
    assert latest_version is not None
    assert latest_version['version_number'] == 3
    assert latest_version['content'] == content1
    assert latest_version['prompt'] == prompt1
    assert latest_version['analysis_content'] == analysis1

    # Verify the main Media record was updated
    with db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT content, content_hash FROM Media WHERE id = ?", (media_id,))
        media_row = cursor.fetchone()
        assert media_row is not None
        assert media_row['content'] == content1
        expected_hash = hashlib.sha256(content1.encode()).hexdigest()
        assert media_row['content_hash'] == expected_hash


def test_rollback_to_nonexistent_version(memory_db_instance: Database):
    db = memory_db_instance
    media_id = insert_test_media(db, 13)
    create_document_version(media_id=media_id, content="V1", db_instance=db)

    result = rollback_to_version(media_id=media_id, version_number=5, db_instance=db) # Version 5 doesn't exist

    assert 'error' in result
    assert result['error'] == 'Version 5 not found'