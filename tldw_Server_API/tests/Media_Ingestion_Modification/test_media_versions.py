# test_media_versions.py
# Description: This file contains tests for the media ingestion and modification endpoints.
#
# Imports
import unittest
from sqlite3 import IntegrityError
#
# 3rd-party Libraries
from fastapi.testclient import TestClient
#
# Local Imports
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.DB_Manager import get_all_document_versions, get_document_version, \
    create_document_version
from tldw_Server_API.app.core.DB_Management.DB_Manager import get_full_media_details
from tldw_Server_API.app.core.DB_Management.SQLite_DB import Database, create_tables

#
########################################################################################################################
#
# Functions:

client = TestClient(app)


class TestMediaVersions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up test database
        cls.db = Database(":memory:")  # In-memory DB for tests
        cls.db.execute_query("PRAGMA foreign_keys = ON")
        create_tables(cls.db)

        # Add test media
        cls.db.execute_query('''
            INSERT INTO Media (title, type, content) 
            VALUES (?, ?, ?)
        ''', ("Test Media", "document", "Initial content"))
        cls.media_id = cls.db.execute_query("SELECT last_insert_rowid()")[0][0]

    @classmethod
    def tearDownClass(cls):
        cls.db.close_connection()

    def setUp(self):
        # Start each test with a clean versions table
        self.db.execute_query("DELETE FROM DocumentVersions")
        self.db.execute_query("DELETE FROM MediaModifications")

        # Create initial version
        create_document_version(
            media_id=self.media_id,
            content="Initial content",
            prompt="Initial prompt",
            summary="Initial summary",
            conn=self.db.get_connection()
        )

    # Helper methods
    def create_test_version(self, content="Test content"):
        response = client.post(
            f"/media/{self.media_id}/versions",
            json={
                "content": content,
                "prompt": "Test prompt",
                "summary": "Test summary"
            }
        )
        return response

    # Tests begin here
    def test_create_version(self):
        response = self.create_test_version()
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["version_number"], 2)  # First version created in setUp
        self.assertGreater(data["content_length"], 0)

    def test_create_version_invalid_media(self):
        response = client.post(
            "/media/9999/versions",
            json={"content": "Test"}
        )
        self.assertEqual(response.status_code, 400)

    def test_list_versions(self):
        # Create second version
        self.create_test_version("Second content")

        response = client.get(f"/media/{self.media_id}/versions")
        self.assertEqual(response.status_code, 200)
        versions = response.json()
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[0]["version_number"], 2)
        self.assertEqual(versions[1]["version_number"], 1)

    def test_get_version(self):
        # First version exists from setUp
        response = client.get(
            f"/media/{self.media_id}/versions/1",
            params={"include_content": True}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["version_number"], 1)
        self.assertIn("Initial content", data["content"])

    def test_get_nonexistent_version(self):
        response = client.get(f"/media/{self.media_id}/versions/999")
        self.assertEqual(response.status_code, 404)

    def test_delete_version(self):
        response = client.delete(f"/media/{self.media_id}/versions/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["success"], "Version 1 deleted")

        # Verify deletion
        response = client.get(f"/media/{self.media_id}/versions/1")
        self.assertEqual(response.status_code, 404)

    def test_delete_nonexistent_version(self):
        response = client.delete(f"/media/{self.media_id}/versions/999")
        self.assertEqual(response.status_code, 404)

    def test_rollback_version(self):
        # Create second version
        self.create_test_version("Second content")

        response = client.post(
            f"/media/{self.media_id}/versions/rollback",
            json={"version_number": 1}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["new_version_number"], 3)  # 2 existing + new rollback

        # Verify rollback content
        version = get_document_version(self.media_id, 3, include_content=True)
        self.assertIn("Initial content", version["content"])

    def test_rollback_nonexistent_version(self):
        response = client.post(
            f"/media/{self.media_id}/versions/rollback",
            json={"version_number": 999}
        )
        self.assertEqual(response.status_code, 400)

    def test_update_media_with_versioning(self):
        response = client.put(
            f"/media/{self.media_id}",
            json={
                "content": "Updated content",
                "keywords": ["test"]
            }
        )
        self.assertEqual(response.status_code, 200)

        # Verify new version was created
        versions = get_all_document_versions(self.media_id)
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[0]["version_number"], 2)

        # Verify keywords updated
        media = get_full_media_details(self.media_id)
        self.assertIn("test", media["keywords"])


if __name__ == "__main__":
    unittest.main()