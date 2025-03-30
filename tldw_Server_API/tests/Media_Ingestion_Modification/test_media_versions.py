# test_media_versions.py
# Description: This file contains tests for the media ingestion and modification endpoints.
#
# Imports
import json
import os
import tempfile
import time
import unittest
from sqlite3 import IntegrityError
from uuid import uuid4

import pytest
#
# 3rd-party Libraries
from fastapi.testclient import TestClient

#
# Local Imports
from tldw_Server_API.app.core.DB_Management.DB_Dependency import get_db_manager
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.DB_Manager import get_all_document_versions, get_document_version, \
    create_document_version
from tldw_Server_API.app.core.DB_Management.DB_Manager import get_full_media_details
from tldw_Server_API.app.core.DB_Management.SQLite_DB import Database, create_tables
from tldw_Server_API.tests.test_utils import create_test_media, temp_db as temp_db_context
#
########################################################################################################################
#
# Functions:

client = TestClient(app)
def override_get_db_manager():
    # Use the same temporary DB created in your setUpClass of one of your test classes.
    from tldw_Server_API.tests.test_utils import temp_db
    # Create a temporary DB and yield it; adjust as needed for your test setup.
    with temp_db() as test_db:
        yield test_db

@pytest.fixture(scope="module")
def temp_db_fixture():
    # Create a temporary directory and database file
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test.db")
        temp_db_instance = Database(db_path)
        create_tables(temp_db_instance)  # Ensure the schema is set up
        yield temp_db_instance
        temp_db_instance.close_connection()

@pytest.fixture(scope="module")
def test_client(temp_db_fixture):
    # Override the get_db_manager dependency to use the temporary DB
    def override_get_db_manager():
        yield temp_db_fixture

    app.dependency_overrides[get_db_manager] = override_get_db_manager
    client_instance = TestClient(app)
    yield client_instance
    app.dependency_overrides.clear()

def test_example(client):
    # Now all routes that depend on get_db_manager will use the temporary DB
    response = client.get("/some-endpoint")
    assert response.status_code == 200

class TestMediaVersionEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tldw_Server_API.tests.test_utils import temp_db as temp_db_context
        cls.temp_db_context = temp_db_context()  # Use the context manager from test_utils
        cls.db = cls.temp_db_context.__enter__()

        with cls.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(MediaModifications)")
            columns = cursor.fetchall()
            print("FUCKME MediaModifications columns:", columns)

            cursor.execute("PRAGMA index_list(MediaModifications)")
            indexes = cursor.fetchall()
            print("FUCKMETWICE MediaModifications indexes:", indexes)

        # Enable foreign keys
        cls.db.execute_query("PRAGMA foreign_keys=ON")

        # Insert initial media data.
        cls.db.execute_query('''
            INSERT INTO Media (title, type, content, author)
            VALUES (?, ?, ?, ?)
        ''', ("Test Media", "document", "Initial content", "Tester"))
        media_info = cls.db.execute_query("SELECT last_insert_rowid()")
        cls.media_id = media_info[0][0]

        # Create the initial version.
        create_document_version(
            media_id=cls.media_id,
            content="Initial content",
            prompt="Initial prompt",
            summary="Initial summary"
        )

        # Verify MediaModifications table exists and has UNIQUE constraint
        cls.db.execute_query("SELECT 1 FROM MediaModifications LIMIT 1")  # Check table exists
        indexes = cls.db.execute_query("PRAGMA index_list(MediaModifications)")
        unique_indexes = [idx[1] for idx in indexes if idx[2] == 1]  # Check for unique indexes
        assert 'media_id' in unique_indexes, "Unique index on media_id is missing!"

    @classmethod
    def tearDownClass(cls):
        # Exit the temporary DB context. This will close the DB and remove the temporary directory.
        cls.temp_db_context.__exit__(None, None, None)

    def setUp(self):
        self.transaction = self.db.transaction().__enter__()

    def tearDown(self):
        self.transaction.__exit__(None, None, None)

    # Helper methods
    def create_version(self, content="Test content"):
        return client.post(
            f"/api/v1/media/{self.media_id}/versions",
            json={"content": content, "prompt": "Test", "summary": "Test"}
        )

    # Version creation tests (4 cases)
    def test_create_valid_version(self):
        response = self.create_version()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["version_number"], 2)

    def test_create_version_invalid_media_id(self):
        response = client.post("/api/v1/media/9999/versions", json={"content": "Test"})
        self.assertEqual(response.status_code, 400)

    def test_create_version_missing_fields(self):
        response = client.post(
            f"/api/v1/media/{self.media_id}/versions",
            json={"content": "Test"}  # Missing prompt/summary
        )
        self.assertEqual(response.status_code, 422)

    def test_create_version_large_content(self):
        large_content = "A" * 10_000_000  # 10MB content
        response = self.create_version(large_content)
        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.json()["content_length"], 10_000_000)

    # Version listing tests (3 cases)
    def test_list_versions_empty(self):
        self.db.execute_query("DELETE FROM DocumentVersions")
        response = client.get(f"/api/v1/media/{self.media_id}/versions")
        self.assertEqual(response.status_code, 404)

    def test_list_versions_pagination(self):
        # Create 15 versions
        for i in range(15):
            self.create_version(f"Content {i}")

        response = client.get(
            f"/api/v1/media/{self.media_id}/versions",
            params={"limit": 5, "offset": 10}
        )
        data = response.json()
        self.assertEqual(len(data), 5)
        self.assertEqual(data[0]["version_number"], 15 - 10)  # Verify ordering

    def test_list_versions_include_content(self):
        response = client.get(
            f"/api/v1/media/{self.media_id}/versions",
            params={"include_content": True}
        )
        self.assertIn("Initial content", response.json()[0]["content"])

    # Version retrieval tests (3 cases)
    def test_get_specific_version(self):
        response = client.get(f"/api/v1/media/{self.media_id}/versions/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["version_number"], 1)

    def test_get_nonexistent_version(self):
        response = client.get(f"/api/v1/media/{self.media_id}/versions/999")
        self.assertEqual(response.status_code, 404)

    def test_get_version_content_toggle(self):
        response = client.get(
            f"/api/v1/media/{self.media_id}/versions/1",
            params={"include_content": False}
        )
        self.assertIsNone(response.json().get("content"))

    # Version deletion tests (3 cases)
    def test_delete_version_success(self):
        response = client.delete(f"/api/v1/media/{self.media_id}/versions/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["success"], "Version 1 deleted")

    def test_delete_nonexistent_version(self):
        response = client.delete(f"/api/v1/media/{self.media_id}/versions/999")
        self.assertEqual(response.status_code, 404)

    def test_delete_last_version(self):
        self.db.execute_query("DELETE FROM DocumentVersions")
        response = client.delete(f"/api/v1/media/{self.media_id}/versions/1")
        self.assertEqual(response.status_code, 404)

    # Rollback tests (3 cases)
    def test_rollback_valid_version(self):
        self.create_version()
        response = client.post(
            f"/api/v1/media/{self.media_id}/versions/rollback",
            json={"version_number": 1}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["new_version_number"], 3)

    def test_rollback_invalid_version(self):
        response = client.post(
            f"/api/v1/media/{self.media_id}/versions/rollback",
            json={"version_number": 999}
        )
        self.assertEqual(response.status_code, 400)

    def test_rollback_without_content(self):
        response = client.post(
            f"/api/v1/media/{self.media_id}/versions/rollback",
            json={"version_number": 1, "preserve_content": False}
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("content", response.json())


class TestMediaEndpoints(unittest.TestCase):
    """Contains tests for media endpoints including listing, filtering, and detail retrieval."""

    @classmethod
    def setUpClass(cls):
        from tldw_Server_API.tests.test_utils import temp_db as temp_db_context
        cls.temp_db_cm = temp_db_context()
        cls.db = cls.temp_db_cm.__enter__()
        cls.db.execute_query("PRAGMA foreign_keys=ON")

        # Insert a default media record
        cls.db.execute_query('''
            INSERT INTO Media (title, type, content, author)
            VALUES (?, ?, ?, ?)
        ''', ("Test Media", "document", "Initial content", "Tester"))
        media_info = cls.db.execute_query("SELECT last_insert_rowid()")
        cls.media_id = media_info[0][0]
        default_media_id = media_info[0][0]

        create_document_version(
            media_id=cls.media_id,
            content="Initial content",
            prompt="Initial prompt",
            summary="Initial summary"
        )

        indexes = cls.db.execute_query("PRAGMA index_list(MediaModifications)")
        unique_indexes = [idx[1] for idx in indexes if idx[2] == 1]
        assert 'media_id' in unique_indexes, "Unique index on media_id is missing!"

        # Insert additional media for filter testing
        cls.db.execute_query('''
            INSERT INTO Media (title, type, content, author)
            VALUES (?, ?, ?, ?)
        ''', ("Test Video", "video", "Video content", "Tester"))
        video_info = cls.db.execute_query("SELECT last_insert_rowid()")
        video_id = video_info[0][0]

        # Insert an audio record
        cls.db.execute_query('''
            INSERT INTO Media (title, type, content, author)
            VALUES (?, ?, ?, ?)
        ''', ("Test Audio", "audio", "Audio content", "Tester"))
        audio_info = cls.db.execute_query("SELECT last_insert_rowid()")
        audio_id = audio_info[0][0]

        # Store the IDs for later reference in tests.
        cls.media_ids = {"default": default_media_id, "video": video_id, "audio": audio_id}

        # Optionally, create an initial document version for the default media.
        create_document_version(
            media_id=default_media_id,
            content="Initial content",
            prompt="Initial prompt",
            summary="Initial summary"
        )
    # @classmethod
    # def setUpClass(cls):
    #     from tldw_Server_API.tests.test_utils import temp_db
    #     cls.temp_db_cm = temp_db()
    #     cls.db = cls.temp_db_cm.__enter__()
    #     cls.db.execute_query("PRAGMA foreign_keys=ON")
    #
    #     cls.db.execute_query('''
    #         INSERT INTO Media (title, type, content, author)
    #         VALUES (?, ?, ?, ?)
    #     ''', ("Test Media", "document", "Initial content", "Tester"))
    #     media_info = cls.db.execute_query("SELECT last_insert_rowid()")
    #     cls.media_id = media_info[0][0]
    #
    #     create_document_version(
    #         media_id=cls.media_id,
    #         content="Initial content",
    #         prompt="Initial prompt",
    #         summary="Initial summary"
    #     )
        # Verify MediaModifications table exists and has UNIQUE constraint
        cls.db.execute_query("SELECT 1 FROM MediaModifications LIMIT 1")  # Check table exists
        indexes = cls.db.execute_query("PRAGMA index_list(MediaModifications)")
        unique_indexes = [idx[1] for idx in indexes if idx[2] == 1]  # Check for unique indexes
        assert 'media_id' in unique_indexes, "Unique index on media_id is missing!"

    @classmethod
    def tearDownClass(cls):
        cls.temp_db_cm.__exit__(None, None, None)

    def setUp(self):
        # Begin a new transaction for each test
        self.transaction = self.db.transaction().__enter__()

    def tearDown(self):
        # Roll back any changes after each test
        self.transaction.__exit__(None, None, None)

    # Media listing tests (4 cases)
    def test_get_all_media_default_pagination(self):
        response = client.get("/api/v1/media")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict, "Expected response to be a dictionary.")
        self.assertIn("items", data, "Response JSON missing 'items' key.")
        self.assertIn("pagination", data, "Response JSON missing 'pagination' key.")
        self.assertIsInstance(data["items"], list, "'items' should be a list.")
        self.assertIsInstance(data["pagination"], dict, "'pagination' should be a dictionary.")

        # Validate structure of an item, if present
        if data["items"]:
            first_item = data["items"][0]
            self.assertIn("id", first_item)
            self.assertIn("title", first_item)
            self.assertIn("url", first_item)

        # Validate pagination structure
        self.assertIn("page", data["pagination"])
        self.assertIn("results_per_page", data["pagination"])
        self.assertIn("total_pages", data["pagination"])
        self.assertIsInstance(data["pagination"]["page"], int)
        self.assertIsInstance(data["pagination"]["results_per_page"], int)
        self.assertIsInstance(data["pagination"]["total_pages"], int)

    def test_get_all_media_custom_pagination(self):
        response = client.get("/api/v1/media?page=1&results_per_page=2")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict, "Expected response to be a dictionary.")
        self.assertIn("items", data, "Response JSON missing 'items' key.")
        self.assertIsInstance(data["items"], list, "'items' should be a list.")
        self.assertEqual(len(data["items"]), 2)
        self.assertIn("pagination", data, "Response JSON missing 'pagination' key.")
        self.assertIsInstance(data["pagination"], dict, "'pagination' should be a dictionary.")
        self.assertEqual(data["pagination"]["page"], 1)
        self.assertEqual(data["pagination"]["results_per_page"], 2)
        self.assertIn("total_pages", data["pagination"])

    def test_get_all_media_filter_by_type(self):
        response = client.get("/api/v1/media?media_type=video")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict, "Expected response to be a dictionary.")
        self.assertIn("items", data, "Response JSON missing 'items' key.")
        self.assertIsInstance(data["items"], list, "'items' should be a list.")
        if data["items"]:
            first_item = data["items"][0]
            self.assertIn("title", first_item)
            self.assertEqual(first_item["title"], "Test Video")
        self.assertIn("pagination", data, "Response JSON missing 'pagination' key.")

    def test_get_all_media_invalid_params(self):
        response = client.get("/api/v1/media?page=-1&results_per_page=1000")
        self.assertEqual(response.status_code, 422)

    # Media detail tests (6 cases)
    def test_get_media_item_video(self):
        response = client.get(f"/api/v1/media/{self.media_ids['video']}")
        data = response.json()
        self.assertEqual(data["source"]["type"], "video")
        self.assertGreater(data["content"]["word_count"], 0)

    def test_get_media_item_audio(self):
        response = client.get(f"/api/v1/media/{self.media_ids['audio']}")
        data = response.json()
        self.assertEqual(data["source"]["type"], "audio")

    def test_get_nonexistent_media_item(self):
        response = client.get("/api/v1/media/999999")
        self.assertEqual(response.status_code, 404)

    def test_get_media_item_keywords(self):
        # Use Database instance instead of raw connection
        self.db.execute_query('''
            INSERT INTO MediaModifications (media_id, prompt, summary, keywords)
            VALUES (?, ?, ?, ?)
        ''', (self.media_ids['video'], "Test prompt", "Test summary", "test,demo"))

        response = client.get(f"/api/v1/media/{self.media_ids['video']}")
        self.assertCountEqual(response.json()["keywords"], ["test", "demo"])

    def test_media_item_content_parsing(self):
        response = client.get(f"/api/v1/media/{self.media_ids['video']}")
        data = response.json()
        self.assertIn("Transcript line", data["content"]["text"])
        self.assertEqual(data["processing"]["model"], "unknown")

    # Update tests (2 cases)
    def test_update_media_item(self):
        response = client.put(
            f"/api/v1/media/{self.media_ids['video']}",
            json={"content": "Updated", "keywords": ["test"]}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["keywords"], ["test"])

    def test_update_invalid_media_item(self):
        response = client.put("/api/v1/media/999999", json={"content": "Test"})
        self.assertEqual(response.status_code, 404)


class TestSecurityAndPerformance(unittest.TestCase):
    """Contains 6 test cases for security and performance"""

    def test_response_time(self):
        start = time.time()
        response = client.get("/api/v1/media")
        self.assertLess(time.time() - start, 0.5)  # 500ms threshold

    def test_sql_injection_attempt(self):
        response = client.get("/api/v1/media?page=1%3B DROP TABLE Media")
        self.assertEqual(response.status_code, 422)

    def test_content_type_enforcement(self):
        response = client.post(
            "/api/v1/media/1/versions",
            content='{"content": "test"}',  # Valid JSON content
            headers={"Content-Type": "text/plain"}
        )
        self.assertEqual(response.status_code, 422)

    def test_cors_headers(self):
        response = client.options("/api/v1/media")
        self.assertIn("access-control-allow-origin", [h.lower() for h in response.headers.keys()])

    def test_sensitive_data_exposure(self):
        response = client.get(f"/api/v1/media/1/versions")
        self.assertNotIn("database_password", response.text)


if __name__ == "__main__":
    unittest.main(failfast=True, verbosity=2)