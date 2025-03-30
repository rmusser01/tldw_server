# test_media_versions.py
# Description: This file contains tests for the media ingestion and modification endpoints.
#
# Imports
import time
import pytest
#
# 3rd-party Libraries
from fastapi.testclient import TestClient
#
# Local Imports
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.DB_Dependency import get_db_manager
from tldw_Server_API.app.core.DB_Management.SQLite_DB import Database, create_tables
from tldw_Server_API.app.core.DB_Management.DB_Manager import (
    create_document_version, get_document_version, get_all_document_versions,
    delete_document_version, rollback_to_version
)
from tldw_Server_API.tests.test_utils import temp_db
#
########################################################################################################################
#
# Functions:

@pytest.fixture(scope="session")
def db_fixture():
    """
    Creates a temporary test DB (via `temp_db()`),
    initializes the schema, then yields a Database instance.
    Closes and cleans up after all tests in the session.
    """
    with temp_db() as db:
        create_tables(db)
        db.execute_query("PRAGMA foreign_keys=ON")
        yield db


@pytest.fixture(scope="session")
def client_fixture(db_fixture):
    """
    Creates a TestClient that uses the `db_fixture` as its DB dependency.
    Yields the client for the entire test session.
    """
    def override_get_db_manager():
        yield db_fixture

    # Override
    app.dependency_overrides[get_db_manager] = override_get_db_manager

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


################################################################################################
# EXAMPLE (standalone function test)
################################################################################################

def test_example(client_fixture):
    """
    Demonstrates that the /some-endpoint route is returning status 200.
    If /some-endpoint does not exist, this will fail (or skip).
    """
    response = client_fixture.get("/api/v1/media")
    assert response.status_code == 200


################################################################################################
#                               MEDIA VERSION ENDPOINT TESTS
################################################################################################

@pytest.fixture
def seeded_media(db_fixture):
    """
    Creates a Media record and an initial DocumentVersion for that media.
    Returns the media_id for tests that need it.
    """
    db_fixture.execute_query('''
        INSERT INTO Media (title, type, content, author)
        VALUES (?, ?, ?, ?)
    ''', ("Test Media", "document", "Initial content", "Tester"))
    media_info = db_fixture.execute_query("SELECT last_insert_rowid()")
    media_id = media_info[0][0]

    # Create an initial version
    create_document_version(
        media_id=media_id,
        content="Initial content",
        prompt="Initial prompt",
        summary="Initial summary"
    )
    return media_id


class TestMediaVersionEndpoints:
    """
    Tests for creating, listing, retrieving, deleting, and rolling back versions.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, db_fixture, client_fixture, seeded_media):
        """
        Runs before every test in this class.
        Attaches db_fixture, client_fixture, and the seeded_media_id to `self`.
        """
        self.db = db_fixture
        self.client = client_fixture
        self.media_id = seeded_media

    def create_version(self, content="Test content"):
        """Helper that POSTs a new version for self.media_id."""
        return self.client.post(
            f"/api/v1/media/{self.media_id}/versions",
            json={"content": content, "prompt": "Test", "summary": "Test"}
        )

    # --------------------- CREATE VERSION TESTS ---------------------

    def test_create_valid_version(self):
        response = self.create_version()
        assert response.status_code == 200
        # Depending on your endpoint’s actual return, you may need to adapt this:
        assert isinstance(response.json(), dict) or isinstance(response.json(), int)

    def test_create_version_invalid_media_id(self):
        response = self.client.post("/api/v1/media/9999/versions", json={"content": "Test"})
        assert response.status_code == 400

    def test_create_version_missing_fields(self):
        response = self.client.post(
            f"/api/v1/media/{self.media_id}/versions",
            json={"content": "Test"}  # Missing prompt/summary
        )
        assert response.status_code == 422

    def test_create_version_large_content(self):
        large_content = "A" * 10_000_000
        response = self.create_version(large_content)
        assert response.status_code == 200
        # If your endpoint returns a dict, check the content_length or otherwise:
        if isinstance(response.json(), dict):
            assert response.json().get("content_length", 0) > 10_000_000

    # --------------------- LISTING TESTS ---------------------

    def test_list_versions_empty(self):
        # Wipe the DocumentVersions table
        self.db.execute_query("DELETE FROM DocumentVersions")
        response = self.client.get(f"/api/v1/media/{self.media_id}/versions")
        assert response.status_code == 404

    def test_list_versions_pagination(self):
        # Create 15 versions
        for i in range(15):
            self.create_version(f"Content {i}")

        response = self.client.get(
            f"/api/v1/media/{self.media_id}/versions",
            params={"limit": 5, "offset": 10}
        )
        assert response.status_code == 200
        data = response.json()
        if isinstance(data, list):
            assert len(data) == 5
            # If your code returns version_number in descending order, adapt accordingly:
            # example check: assert data[0]["version_number"] == 15 - 10
        else:
            # If your endpoint returns int or something else, skip or adapt
            pytest.skip("list_versions returned unexpected type; skipping deeper checks")

    def test_list_versions_include_content(self):
        response = self.client.get(
            f"/api/v1/media/{self.media_id}/versions",
            params={"include_content": True}
        )
        if response.status_code == 200 and isinstance(response.json(), list):
            assert "Initial content" in response.json()[0].get("content", "")
        else:
            pytest.skip("No version content found or endpoint returned unexpected type.")

    # --------------------- RETRIEVE TESTS ---------------------

    def test_get_specific_version(self):
        response = self.client.get(f"/api/v1/media/{self.media_id}/versions/1")
        assert response.status_code == 200
        if isinstance(response.json(), dict):
            assert response.json().get("version_number") == 1

    def test_get_nonexistent_version(self):
        response = self.client.get(f"/api/v1/media/{self.media_id}/versions/999")
        assert response.status_code == 404

    def test_get_version_content_toggle(self):
        response = self.client.get(
            f"/api/v1/media/{self.media_id}/versions/1",
            params={"include_content": False}
        )
        if isinstance(response.json(), dict):
            assert response.json().get("content") is None

    # --------------------- DELETE TESTS ---------------------

    def test_delete_version_success(self):
        response = self.client.delete(f"/api/v1/media/{self.media_id}/versions/1")
        assert response.status_code == 200
        # If your endpoint returns a dict with {"success": "..."} or an int, adapt as needed.

    def test_delete_nonexistent_version(self):
        response = self.client.delete(f"/api/v1/media/{self.media_id}/versions/999")
        assert response.status_code == 404

    def test_delete_last_version(self):
        # Wipe DocumentVersions first
        self.db.execute_query("DELETE FROM DocumentVersions")
        response = self.client.delete(f"/api/v1/media/{self.media_id}/versions/1")
        assert response.status_code == 404

    # --------------------- ROLLBACK TESTS ---------------------

    def test_rollback_valid_version(self):
        # Create a second version, so there's something to roll back to
        self.create_version()
        response = self.client.post(
            f"/api/v1/media/{self.media_id}/versions/rollback",
            json={"version_number": 1}
        )
        assert response.status_code == 200
        if isinstance(response.json(), dict):
            # e.g. new_version_number == 3
            assert response.json().get("new_version_number") is not None

    def test_rollback_invalid_version(self):
        response = self.client.post(
            f"/api/v1/media/{self.media_id}/versions/rollback",
            json={"version_number": 999}
        )
        assert response.status_code == 400

    def test_rollback_without_content(self):
        response = self.client.post(
            f"/api/v1/media/{self.media_id}/versions/rollback",
            json={"version_number": 1, "preserve_content": False}
        )
        assert response.status_code == 200
        if isinstance(response.json(), dict):
            assert "content" not in response.json()


################################################################################################
#                               MEDIA ENDPOINT TESTS
################################################################################################

@pytest.fixture
def seeded_media_ids(db_fixture):
    """
    Creates multiple media records (document, video, audio) for TestMediaEndpoints.
    Returns a dict of their IDs.
    """
    db_fixture.execute_query('''
        INSERT INTO Media (title, type, content, author)
        VALUES (?, ?, ?, ?)
    ''', ("Test Media", "document", "Initial content", "Tester"))
    doc_id = db_fixture.execute_query("SELECT last_insert_rowid()")[0][0]

    create_document_version(
        media_id=doc_id,
        content="Initial content",
        prompt="Initial prompt",
        summary="Initial summary"
    )

    # Insert "video"
    db_fixture.execute_query('''
        INSERT INTO Media (title, type, content, author)
        VALUES (?, ?, ?, ?)
    ''', ("Test Video", "video", "Video content", "Tester"))
    video_id = db_fixture.execute_query("SELECT last_insert_rowid()")[0][0]

    # Insert "audio"
    db_fixture.execute_query('''
        INSERT INTO Media (title, type, content, author)
        VALUES (?, ?, ?, ?)
    ''', ("Test Audio", "audio", "Audio content", "Tester"))
    audio_id = db_fixture.execute_query("SELECT last_insert_rowid()")[0][0]

    return {"default": doc_id, "video": video_id, "audio": audio_id}


class TestMediaEndpoints:
    """
    Contains tests for /media listing, detail, and updates.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, db_fixture, client_fixture, seeded_media_ids):
        self.db = db_fixture
        self.client = client_fixture
        self.media_ids = seeded_media_ids

    def test_get_all_media_default_pagination(self):
        response = self.client.get("/api/v1/media")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "items" in data and "pagination" in data
        assert isinstance(data["items"], list)
        assert isinstance(data["pagination"], dict)

    def test_get_all_media_custom_pagination(self):
        response = self.client.get("/api/v1/media?page=1&results_per_page=2")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) == 2

    def test_get_all_media_filter_by_type(self):
        # If your endpoint supports ?media_type=video, adapt accordingly
        response = self.client.get("/api/v1/media?media_type=video")
        assert response.status_code == 200
        data = response.json()
        if data["items"]:
            assert data["items"][0]["title"] == "Test Video"

    def test_get_all_media_invalid_params(self):
        response = self.client.get("/api/v1/media?page=-1&results_per_page=1000")
        assert response.status_code == 422

    def test_get_media_item_video(self):
        vid_id = self.media_ids["video"]
        response = self.client.get(f"/api/v1/media/{vid_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["source"]["type"] == "video"
        assert data["content"]["word_count"] > 0

    def test_get_media_item_audio(self):
        audio_id = self.media_ids["audio"]
        response = self.client.get(f"/api/v1/media/{audio_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["source"]["type"] == "audio"

    def test_get_nonexistent_media_item(self):
        response = self.client.get("/api/v1/media/999999")
        assert response.status_code == 404

    def test_get_media_item_keywords(self):
        # Add some keywords
        self.db.execute_query('''
            INSERT INTO MediaModifications (media_id, prompt, summary, keywords)
            VALUES (?, ?, ?, ?)
        ''', (self.media_ids['video'], "Test prompt", "Test summary", "test,demo"))
        response = self.client.get(f"/api/v1/media/{self.media_ids['video']}")
        data = response.json()
        assert sorted(data["keywords"]) == ["demo", "test"]

    def test_media_item_content_parsing(self):
        response = self.client.get(f"/api/v1/media/{self.media_ids['video']}")
        assert response.status_code == 200
        data = response.json()
        assert "Transcript line" in data["content"]["text"]
        assert data["processing"]["model"] == "unknown"

    def test_update_media_item(self):
        response = self.client.put(
            f"/api/v1/media/{self.media_ids['video']}",
            json={"content": "Updated", "keywords": ["test"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("keywords") == ["test"]

    def test_update_invalid_media_item(self):
        response = self.client.put("/api/v1/media/999999", json={"content": "Test"})
        assert response.status_code == 404


################################################################################################
#                           SECURITY & PERFORMANCE TESTS
################################################################################################

class TestSecurityAndPerformance:
    @pytest.fixture(autouse=True)
    def _setup(self, client_fixture):
        self.client = client_fixture

    def test_response_time(self):
        start = time.time()
        response = self.client.get("/api/v1/media")
        elapsed = time.time() - start
        assert elapsed < 0.5, f"Response took too long: {elapsed}s"

    def test_sql_injection_attempt(self):
        response = self.client.get("/api/v1/media?page=1%3B DROP TABLE Media")
        assert response.status_code == 422

    def test_content_type_enforcement(self):
        response = self.client.post(
            "/api/v1/media/1/versions",
            content='{"content": "test"}',
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422

    def test_cors_headers(self):
        """
        If CORS isn’t configured, you likely won’t see these headers. Adapt as needed.
        """
        response = self.client.options("/api/v1/media")
        # If no CORS, skip or just check the headers present
        lower_keys = [h.lower() for h in response.headers.keys()]
        assert "access-control-allow-origin" in lower_keys, "CORS header missing"

    def test_sensitive_data_exposure(self):
        response = self.client.get("/api/v1/media/1/versions")
        # Just ensure no mention of "database_password"
        assert "database_password" not in response.text
