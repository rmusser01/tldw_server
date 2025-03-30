# test_document_versioning.py
# Refactored tests for document versioning in SQLite database.
#
# Imports
import pytest
import sqlite3
from unittest.mock import patch
from PoC_Version.App_Function_Libraries.DB.SQLite_DB import (
    create_document_version,
    get_document_version,
    get_all_document_versions,
    delete_document_version,
    rollback_to_version,
)
#
################################################################################################################################################################
#
# Test: create_document_version

@pytest.fixture
def mock_db():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create necessary tables
    cursor.execute('''
        CREATE TABLE Media (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE DocumentVersions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            version_number INTEGER NOT NULL,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (media_id) REFERENCES Media(id)
        )
    ''')

    # Insert a test media entry
    cursor.execute("INSERT INTO Media (id, title) VALUES (1, 'Test Document')")
    conn.commit()

    yield conn

    conn.close()


@pytest.fixture
def mock_get_connection(mock_db):
    with patch('App_Function_Libraries.DB.SQLite_DB.db.get_connection') as mock:
        mock.return_value = mock_db
        yield mock


def test_create_document_version(mock_get_connection):
    media_id = 1
    content = "Test content"

    version = create_document_version(media_id, content)

    assert version == 1

    cursor = mock_get_connection.return_value.cursor()
    cursor.execute("SELECT * FROM DocumentVersions WHERE media_id = ?", (media_id,))
    result = cursor.fetchone()

    assert result is not None
    assert result[1] == media_id
    assert result[2] == 1
    assert result[3] == content


def test_create_document_version_invalid_media_id(mock_get_connection):
    media_id = 999  # Non-existent media_id
    content = "Test content"

    with pytest.raises(ValueError):
        create_document_version(media_id, content)


def test_get_document_version_latest(mock_get_connection):
    media_id = 1
    content1 = "Version 1 content"
    content2 = "Version 2 content"

    create_document_version(media_id, content1)
    create_document_version(media_id, content2)

    result = get_document_version(media_id)

    assert result['version_number'] == 2
    assert result['content'] == content2


def test_get_document_version_specific(mock_get_connection):
    media_id = 1
    content1 = "Version 1 content"
    content2 = "Version 2 content"

    create_document_version(media_id, content1)
    create_document_version(media_id, content2)

    result = get_document_version(media_id, 1)

    assert result['version_number'] == 1
    assert result['content'] == content1


def test_get_all_document_versions(mock_get_connection):
    media_id = 1
    content1 = "Version 1 content"
    content2 = "Version 2 content"

    create_document_version(media_id, content1)
    create_document_version(media_id, content2)

    results = get_all_document_versions(media_id)

    assert len(results) == 2
    assert results[0]['version_number'] == 2
    assert results[0]['content'] == content2
    assert results[1]['version_number'] == 1
    assert results[1]['content'] == content1


def test_delete_document_version(mock_get_connection):
    media_id = 1
    content = "Test content"

    create_document_version(media_id, content)

    result = delete_document_version(media_id, 1)

    assert result['success'] == "Document version 1 for media_id 1 deleted successfully"

    cursor = mock_get_connection.return_value.cursor()
    cursor.execute("SELECT * FROM DocumentVersions WHERE media_id = ?", (media_id,))
    result = cursor.fetchone()

    assert result is None


def test_rollback_to_version(mock_get_connection):
    media_id = 1
    content1 = "Version 1 content"
    content2 = "Version 2 content"

    create_document_version(media_id, content1)
    create_document_version(media_id, content2)

    result = rollback_to_version(media_id, 1)

    assert result['success'] == "Rolled back to version 1 for media_id 1"
    assert result['new_version_number'] == 3

    latest_version = get_document_version(media_id)
    assert latest_version['version_number'] == 3
    assert latest_version['content'] == content1
