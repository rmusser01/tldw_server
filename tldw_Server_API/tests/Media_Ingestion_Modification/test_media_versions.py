# test_media_versions.py
# Description: This file contains tests for the media versioning endpoints.
#
# Imports
import sys
import time
import uuid
import pytest
#
# Third-party Libraries
from fastapi.testclient import TestClient
from fastapi import status # Use status codes from fastapi
#
# Local Imports
# --- Use Main App Instance ---
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.main import app as fastapi_app_instance, app
    # Import specific DB functions used directly in tests/fixtures
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import (
        Database
    )
    # Import the utility for temporary DB if it's defined elsewhere
from tldw_Server_API.tests.test_utils import temp_db
#
#######################################################################################################################
#
# --- Fixtures ---

@pytest.fixture(scope="session")
def db_instance_session():
    """
    Uses the temp_db context manager from test_utils to get an initialized Database instance.
    """
    # temp_db now handles creation and setup (via Database.__init__)
    db = None # Initialize db to None
    try:
        with temp_db() as db:
            # db object provided by temp_db is already initialized with schema
            # Optionally enable foreign keys again here if needed, although Database init might do it
            try:
                 db.execute_query("PRAGMA foreign_keys=ON;")
            except Exception as fk_e:
                 print(f"Warning: Could not enable foreign keys on session DB: {fk_e}")
            yield db
        # Cleanup (closing connection) is handled by temp_db's finally block
    finally:
         if db:
             if hasattr(db, 'close_all_connections'):
                 db.close_all_connections()
             elif hasattr(db, 'close_connection'):
                 # Fallback if close_all_connections doesn't exist
                 db.close_connection()
             else:
                 print(f"--- Warning: DB instance {db.db_path_str} has no close_all_connections or close_connection method ---")
         else:
             print("--- DB instance was not created, skipping close ---")


@pytest.fixture(scope="function")
def db_session(db_instance_session):
     """
     Provides access to the session-scoped DB instance for each test function.
     Includes cleanup logic after each test.
     """
     yield db_instance_session
     # Explicit cleanup after each test
     # print("Cleaning up DB after test...") # Debugging
     try:
        # Delete data from tables in reverse order of dependency using the provided Database instance methods
        with db_instance_session.transaction(): # Use transaction for cleanup
            db_instance_session.execute_query("DELETE FROM MediaKeywords;")
            db_instance_session.execute_query("DELETE FROM DocumentVersions;")
            # Add delete statements for other relevant tables if needed
            # e.g., db_instance_session.execute_query("DELETE FROM UnvectorizedMediaChunks;")
            # e.g., db_instance_session.execute_query("DELETE FROM MediaChunks;")
            # e.g., db_instance_session.execute_query("DELETE FROM Transcripts;")
            db_instance_session.execute_query("DELETE FROM Media;")
            db_instance_session.execute_query("DELETE FROM Keywords;")
        # Reset autoincrement (optional, but good for consistency) - needs commit outside transaction usually
        try:
            # These need separate commits potentially, or run outside transaction
            db_instance_session.execute_query("DELETE FROM sqlite_sequence WHERE name IN ('Media', 'Keywords', 'DocumentVersions', 'MediaKeywords', 'Transcripts', 'MediaChunks', 'UnvectorizedMediaChunks');", commit=True)
        except Exception as seq_e:
             print(f"Warning: Could not reset sequences - {seq_e}") # Non-fatal usually

     except Exception as e:
         print(f"Error during DB cleanup: {e}") # Avoid masking test failures

# Global reference for shutdown handler (consider if needed)
test_db_instance_ref = None

@pytest.fixture(scope="module")
def client_module(db_instance_session):
    """
    Creates a TestClient for the module, overriding the DB dependency to use the session-scoped test DB.
    """
    def override_get_media_db_for_user():
        # print(f"--- OVERRIDING get_media_db_for_user with: {db_instance_session.db_path_str} ---")
        yield db_instance_session

    global test_db_instance_ref
    test_db_instance_ref = db_instance_session # Store the reference for shutdown

    # Store original overrides
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_media_db_for_user] = override_get_media_db_for_user

    with TestClient(fastapi_app_instance) as client:
        yield client

    # Restore original overrides AFTER client is closed
    app.dependency_overrides = original_overrides
    # test_db_instance_ref = None # Consider if shutdown handler is still needed
    # print("--- CLEARED get_media_db_for_user override ---")

# --- Seeding Fixtures ---
@pytest.fixture(scope="function") # Run for each test function
def seeded_document_media(db_session):
    """Creates a Media record (type=document) and an initial DocumentVersion."""
    try:
        media_id = None
        media_uuid = str(uuid.uuid4()) # Generate UUID
        current_time = db_session._get_current_utc_timestamp_str() # Get timestamp
        client_id = db_session.client_id # Get client ID from the db instance

        with db_session.transaction():
            # Execute insert and immediately get the last rowid
            cursor = db_session.execute_query(
                """INSERT INTO Media
                   (title, type, content, author, content_hash, uuid, last_modified, client_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "Test Document", "document", "Initial content v1", "Seed Tester",
                    f"hash_doc_{time.time()}", # content_hash
                    media_uuid,                 # uuid
                    current_time,               # last_modified
                    client_id                   # client_id
                ),
                 commit=False # Commit handled by transaction context
            )
            media_id = cursor.lastrowid # Get ID from the cursor after insert
            if media_id is None:
                # Fallback fetch lastrow ID
                id_cursor = db_session.execute_query("SELECT last_insert_rowid();")
                id_row = id_cursor.fetchone()
                media_id = id_row[0] if id_row else None

            if media_id is None:
                raise RuntimeError("Failed to retrieve media_id after insertion.")

            # Create an initial version using the imported function
            version_res = Database.create_document_version(
                self=db_session,  # Pass the db_instance
                media_id=media_id,
                content="Initial content v1",
                prompt="Initial prompt v1",
                analysis_content="Initial summary v1",
            )
        # Transaction commits automatically here if no exceptions occurred
        # print(f"Seeded media ID: {media_id}, Initial version result: {version_res}") # Debugging
        return media_id
    except Exception as e:
         pytest.fail(f"Failed to seed document media: {e}")


@pytest.fixture(scope="function")
def seeded_multi_media(db_session):
    """Creates multiple media records (doc, video, audio) with keywords for list/detail tests."""
    media_ids = {}
    try:
        with db_session.transaction():
            current_time = db_session._get_current_utc_timestamp_str() # Get time once for batch
            client_id = db_session.client_id

            # --- Document ---
            doc_uuid = str(uuid.uuid4())
            cursor_doc = db_session.execute_query(
                """INSERT INTO Media
                   (title, type, content, author, content_hash, uuid, last_modified, client_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "Multi Test Doc", "document", "Doc content v1", "Multi Tester",
                    f"hash_mdoc_{time.time()}", # content_hash
                    doc_uuid,                   # uuid
                    current_time,               # last_modified
                    client_id                   # client_id
                ),
                commit=False
            )
            media_ids["document"] = cursor_doc.lastrowid
            if media_ids["document"] is None:  # Fallback fetch lastrow ID
                id_cursor = db_session.execute_query("SELECT last_insert_rowid();")
                id_row = id_cursor.fetchone()
                media_ids["document"] = id_row[0] if id_row else None
            if media_ids["document"]:  # Check if ID was obtained
                Database.create_document_version(
                    self=db_session,  # Pass the db_instance
                    media_id=media_ids["document"],  # Pass media_id first
                    content="Doc content v1",
                    prompt="Doc prompt",
                    analysis_content="Doc summary",
                )
            else:
                pytest.fail("Failed to retrieve media_id for document during seeding.")


            # --- Video ---
            vid_uuid = str(uuid.uuid4())
            cursor_vid = db_session.execute_query(
                """INSERT INTO Media
                   (title, type, content, author, content_hash, uuid, last_modified, client_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "Multi Test Video", "video", '{"webpage_url": "http://vid.com"}\n\nTranscript v1', "Multi Tester",
                    f"hash_mvid_{time.time()}", # content_hash
                    vid_uuid,                   # uuid
                    current_time,               # last_modified
                    client_id                   # client_id
                ),
                commit=False
            )
            media_ids["video"] = cursor_vid.lastrowid
            if media_ids["video"] is None:
                id_cursor = db_session.execute_query("SELECT last_insert_rowid();")
                id_row = id_cursor.fetchone()
                media_ids["video"] = id_row[0] if id_row else None


            # --- Audio ---
            aud_uuid = str(uuid.uuid4())
            cursor_aud = db_session.execute_query(
                """INSERT INTO Media
                   (title, type, content, author, content_hash, uuid, last_modified, client_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "Multi Test Audio", "audio", '{"webpage_url": "http://aud.com"}\n\nAudio Transcript v1',
                    "Multi Tester",
                    f"hash_maud_{time.time()}", # content_hash
                    aud_uuid,                   # uuid
                    current_time,               # last_modified
                    client_id                   # client_id
                ),
                commit=False
            )
            media_ids["audio"] = cursor_aud.lastrowid
            if media_ids["audio"] is None:
                id_cursor = db_session.execute_query("SELECT last_insert_rowid();")
                id_row = id_cursor.fetchone()
                media_ids["audio"] = id_row[0] if id_row else None

            # Add keywords
            keywords = ["multi", "test", "seed"]
            for media_id in media_ids.values():
                if media_id is None: continue  # Skip if ID wasn't retrieved
                for keyword in keywords:
                    keyword_id = None  # Reset keyword_id for each iteration

                    kw_cursor = db_session.execute_query("SELECT id FROM Keywords WHERE keyword = ?", (keyword,))
                    kw_row = kw_cursor.fetchone()  # Fetch the first row
                    if kw_row:
                        keyword_id = kw_row[0]  # Get ID from the fetched row
                    else:
                        # Insert keyword and get its ID
                        kw_ins_cursor = db_session.execute_query(
                            "INSERT INTO Keywords (keyword, uuid, last_modified, version, client_id) VALUES (?, ?, ?, ?, ?)",
                            (keyword, str(uuid.uuid4()), current_time, 1, client_id), # Add missing NOT NULL columns
                            commit=False
                        )
                        # Fallback needed if cursor doesn't return ID reliably
                        id_cursor = db_session.execute_query("SELECT last_insert_rowid();")
                        id_row = id_cursor.fetchone()
                        keyword_id = id_row[0] if id_row else None

                    if keyword_id:
                        db_session.execute_query(
                            "INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)",
                            (media_id, keyword_id), commit=False
                        )
                    else:
                        print(
                            f"Warning: Could not determine keyword_id for keyword '{keyword}' for media_id {media_id}")  # Improved warning

        # Transaction commits here
        # Ensure all IDs were captured
        if None in media_ids.values():
             print(f"Warning: Some media IDs were not captured during seeding: {media_ids}")
             pytest.fail("Failed to retrieve all media IDs during seeding.")

        return media_ids

    except Exception as e:
         pytest.fail(f"Failed to seed multi media: {e}")


# --- Test Classes ---
class TestMediaVersionEndpoints:
    """Tests for creating, listing, retrieving, deleting, and rolling back versions."""

    MEDIA_ID_INVALID = 999999

    @pytest.fixture(autouse=True)
    def _setup_class(self, client_module, db_session, seeded_document_media):
        """Setup runs before every test method in this class."""
        self.client = client_module
        self.db = db_session
        self.media_id = seeded_document_media # Gets a fresh media item with 1 version for each test

    def _create_version_request(self, media_id, content="Test content", prompt="Test prompt", analysis_content="Test summary"):
        """Helper to make the POST request to create a version."""
        return self.client.post(
            f"/api/v1/media/{media_id}/versions",
            json={"content": content, "prompt": prompt, "analysis_content": analysis_content}
        )

    # --------------------- CREATE VERSION TESTS ---------------------

    def test_create_valid_version(self):
        """Test creating a second version successfully."""
        response = self._create_version_request(self.media_id, "Content v2")
        assert response.status_code == status.HTTP_201_CREATED # RESTful creation status
        data = response.json()
        assert isinstance(data, dict)
        assert "media_id" in data
        assert "version_number" in data
        assert data["media_id"] == self.media_id
        assert data["version_number"] == 2 # Should be the second version

    def test_create_version_nonexistent_media_id(self):
        """Test creating version for a media ID that doesn't exist."""
        response = self._create_version_request(self.MEDIA_ID_INVALID)
        # API should check if media_id exists before creating version. Expect 404.
        assert response.status_code == status.HTTP_404_NOT_FOUND
        # Check detail message assuming the endpoint provides one
        assert "not found or deleted" in response.json().get("detail", "")


    def test_create_version_invalid_payload_type(self):
        """Test creating version with incorrect payload type (e.g., int instead of str)."""
        response = self.client.post(
            f"/api/v1/media/{self.media_id}/versions",
            json={"content": 123, "prompt": "p", "analysis_content": "s"} # Invalid content type, corrected key
        )
        # Pydantic validation should fail
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_version_missing_fields(self):
        """Test creating version with missing required fields."""
        response = self.client.post(
            f"/api/v1/media/{self.media_id}/versions",
            json={"content": "Test Content Only"}  # Missing prompt/analysis_content
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        detail = response.json().get("detail", [])  # Ensure detail is a list
        assert isinstance(detail, list), f"Expected detail to be a list, got: {type(detail)}"

        # Check if 'prompt' is the field name in the location list for any error
        prompt_error_found = any(
            isinstance(item.get("loc"), list) and len(item["loc"]) > 1 and item["loc"][-1] == "prompt"
            for item in detail
        )
        # Check if 'analysis' (from Pydantic model VersionCreateRequest) is the field name
        analysis_content_error_found = any(
            isinstance(item.get("loc"), list) and len(item["loc"]) > 1 and item["loc"][-1] == "analysis_content"
            for item in detail
        )
        assert prompt_error_found, f"Validation error for 'prompt' not found in details: {detail}"
        assert analysis_content_error_found, f"Validation error for 'analysis_content' not found in details: {detail}"


    # --------------------- LISTING TESTS ---------------------

    def test_list_versions_single_exists(self):
        """Test listing versions when only the initial seeded version exists."""
        response = self.client.get(f"/api/v1/media/{self.media_id}/versions")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["version_number"] == 1
        assert "content" not in data[0] # Content excluded by default

    def test_list_versions_multiple_exist(self):
        """Test listing versions after creating more."""
        self._create_version_request(self.media_id, "Content v2")
        self._create_version_request(self.media_id, "Content v3")

        response = self.client.get(f"/api/v1/media/{self.media_id}/versions")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3
        # Assuming default order is by version number descending
        assert [v["version_number"] for v in data] == [3, 2, 1]

    def test_list_versions_empty(self, db_session):
        """Test listing versions for a media item that exists but has no versions."""
        empty_media_id = None
        try:  # Add try/except for seeding robustness
            media_uuid = str(uuid.uuid4())
            current_time = db_session._get_current_utc_timestamp_str()
            client_id = db_session.client_id
            with db_session.transaction():
                media_res = db_session.execute_query(
                    """INSERT INTO Media
                       (title, type, content, author, content_hash, uuid, last_modified, client_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("Empty Media", "document", "No versions yet", "Tester", f"hash_empty_{time.time()}",
                     media_uuid, current_time, client_id), # Add missing columns
                    commit=False  # Commit handled by transaction
                )
                cursor = db_session.execute_query("SELECT last_insert_rowid();")
                row = cursor.fetchone()
                empty_media_id = row[0] if row else None

            assert empty_media_id is not None, "Failed to retrieve media ID after insertion"

            response = self.client.get(f"/api/v1/media/{empty_media_id}/versions")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 0

        except Exception as e:
            pytest.fail(f"Error during test setup or execution in test_list_versions_empty: {e}")

    def test_list_versions_nonexistent_media_id(self):
        """Test listing versions for a non-existent media ID."""
        response = self.client.get(f"/api/v1/media/{self.MEDIA_ID_INVALID}/versions")
        # Should return 404 if the media item itself doesn't exist
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Media item not found or deleted" in response.json().get("detail", "")

    def test_list_versions_pagination(self):
        """Test pagination (limit and offset) for listing versions."""
        self._create_version_request(self.media_id, "Content v2")
        self._create_version_request(self.media_id, "Content v3")
        self._create_version_request(self.media_id, "Content v4") # Now 4 versions

        # Get page 2 (offset=2) with limit=2 (should get versions 2 and 1)
        response = self.client.get(f"/api/v1/media/{self.media_id}/versions?offset=2&limit=2")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        # Assuming default order is by version number ascending
        assert [v["version_number"] for v in data] == [4, 3]

    def test_list_versions_include_content(self):
        """Test the include_content=true query parameter."""
        self._create_version_request(self.media_id, "Content v2")
        response = self.client.get(f"/api/v1/media/{self.media_id}/versions?include_content=true")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        # Assuming descending order
        assert "content" in data[0]
        assert data[0]["content"] == "Content v2"
        assert data[0]["version_number"] == 2 # Verify correct item
        assert "content" in data[1]
        assert data[1]["content"] == "Initial content v1"
        assert data[1]["version_number"] == 1 # Verify correct item

    # --------------------- RETRIEVE TESTS ---------------------

    def test_get_specific_version_exists(self):
        """Test retrieving a specific, existing version."""
        response = self.client.get(f"/api/v1/media/{self.media_id}/versions/1") # Get seeded version
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, dict)
        assert data["version_number"] == 1
        assert data["content"] == "Initial content v1" # Content included by default

    def test_get_specific_version_nonexistent(self):
        """Test retrieving a version number that doesn't exist."""
        response = self.client.get(f"/api/v1/media/{self.media_id}/versions/99")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Version not found" in response.json().get("detail", "")

    def test_get_specific_version_nonexistent_media_id(self):
        """Test retrieving a version for a media ID that doesn't exist."""
        response = self.client.get(f"/api/v1/media/{self.MEDIA_ID_INVALID}/versions/1")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        # Should ideally indicate media not found first
        assert "Version not found or media" in response.json().get("detail", "")

    def test_get_specific_version_content_toggle_false(self):
        """Test retrieving a version with include_content=false."""
        response = self.client.get(f"/api/v1/media/{self.media_id}/versions/1?include_content=false")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, dict)
        assert data["version_number"] == 1
        assert "content" not in data # Content should be excluded

    # --------------------- DELETE TESTS ---------------------

    def test_delete_version_success(self):
        """Test deleting a version when multiple exist."""
        self._create_version_request(self.media_id, "Content v2") # Create v2
        self._create_version_request(self.media_id, "Content v3") # Create v3

        # Get UUID of version 2 before deleting
        response_get_v2 = self.client.get(f"/api/v1/media/{self.media_id}/versions/2")
        assert response_get_v2.status_code == status.HTTP_200_OK
        v2_uuid = response_get_v2.json().get("uuid")
        assert v2_uuid is not None

        # Delete version 2 (using UUID, assuming endpoint accepts it, or modify to use version_number)
        # Assuming endpoint uses /versions/{version_number_or_uuid} or similar. Let's stick to version_number based on test names.
        response_del = self.client.delete(f"/api/v1/media/{self.media_id}/versions/2")
        assert response_del.status_code == status.HTTP_204_NO_CONTENT # Successful deletion

        # Verify it's gone
        response_get = self.client.get(f"/api/v1/media/{self.media_id}/versions/2")
        assert response_get.status_code == status.HTTP_404_NOT_FOUND

        # Verify others remain
        response_list = self.client.get(f"/api/v1/media/{self.media_id}/versions")
        assert response_list.status_code == status.HTTP_200_OK
        remaining_versions = [v["version_number"] for v in response_list.json()]
        assert remaining_versions == [3, 1] # Check remaining versions

    def test_delete_nonexistent_version(self):
        """Test deleting a version number that doesn't exist."""
        response = self.client.delete(f"/api/v1/media/{self.media_id}/versions/99")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Active media or specific active version not found." in response.json().get("detail", "")

    def test_delete_nonexistent_media_id(self):
        """Test deleting a version for a media ID that doesn't exist."""
        response = self.client.delete(f"/api/v1/media/{self.MEDIA_ID_INVALID}/versions/1")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Active media or specific active version not found." in response.json().get("detail", "")

    def test_delete_last_version_fails(self):
        """Test that deleting the only remaining version is forbidden."""
        # self.media_id starts with only version 1
        response = self.client.delete(f"/api/v1/media/{self.media_id}/versions/1")
        # Should be forbidden (e.g., 400 Bad Request or 409 Conflict)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        # Check the specific detail message from the endpoint
        assert "Cannot delete the only active version of the document." in response.json().get("detail", "")

        # Verify the version still exists
        response_get = self.client.get(f"/api/v1/media/{self.media_id}/versions/1")
        assert response_get.status_code == status.HTTP_200_OK

    # --------------------- ROLLBACK TESTS ---------------------

    def test_rollback_valid_version(self):
        """Test rolling back to a previous version (v1)."""
        v1_response = self.client.get(f"/api/v1/media/{self.media_id}/versions/1?include_content=true")
        assert v1_response.status_code == status.HTTP_200_OK
        v1_data = v1_response.json()

        self._create_version_request(self.media_id, "Content v2", "Prompt v2", "Summary v2") # Create v2
        self._create_version_request(self.media_id, "Content v3", "Prompt v3", "Summary v3") # Create v3 (current)

        # Rollback to version 1
        response = self.client.post(
            f"/api/v1/media/{self.media_id}/versions/rollback",
            json={"version_number": 1}
        )
        assert response.status_code == status.HTTP_200_OK # Or 201 if creating a new version record
        data = response.json()
        assert "new_document_version_number" in data
        new_version_num = data["new_document_version_number"]
        assert new_version_num == 4  # Should create version 4
        assert "new_media_version" in data  # Check this key exists if your endpoint returns it
        new_media_sync_version = data.get("new_media_version")  # Get safely
        #assert new_media_sync_version == media_version_after_v1 + 1

        # Verify the new version (v4) content matches the rolled-back version (v1)
        response_get_new = self.client.get(f"/api/v1/media/{self.media_id}/versions/{new_version_num}?include_content=true")
        assert response_get_new.status_code == status.HTTP_200_OK
        new_version_data = response_get_new.json()
        assert new_version_data["content"] == v1_data["content"] # Content matches v1
        assert new_version_data["prompt"] == v1_data["prompt"]
        assert new_version_data["analysis_content"] == v1_data["analysis_content"]

        # Verify listing shows the new version
        response_list = self.client.get(f"/api/v1/media/{self.media_id}/versions")
        assert response_list.status_code == status.HTTP_200_OK
        versions = [v["version_number"] for v in response_list.json()]
        assert versions == [4, 3, 2, 1]

    def test_rollback_to_current_version_fails(self):
        """Test attempting to rollback to the latest version (should fail or be no-op)."""
        self._create_version_request(self.media_id, "Content v2") # Create v2 (current)

        response = self.client.post(
            f"/api/v1/media/{self.media_id}/versions/rollback",
            json={"version_number": 2} # Rollback to latest
        )
        # Expect error as it doesn't make sense
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Cannot rollback to the current latest version" in response.json().get("detail", "")


    def test_rollback_to_nonexistent_version(self):
        """Test rolling back to a version number that doesn't exist."""
        self._create_version_request(self.media_id, "Content v2") # Create v2

        response = self.client.post(
            f"/api/v1/media/{self.media_id}/versions/rollback",
            json={"version_number": 99} # Non-existent version
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND # Target version not found
        # Adjust expected message based on actual endpoint implementation
        assert "Rollback target version 99 not found or inactive" in response.json().get("detail", "")


    def test_rollback_nonexistent_media_id(self):
        """Test rollback on a media ID that doesn't exist."""
        response = self.client.post(
            f"/api/v1/media/{self.MEDIA_ID_INVALID}/versions/rollback",
            json={"version_number": 1}
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found or deleted." in response.json().get("detail", "")


class TestMediaListDetailEndpoints:
    """Tests for /media listing (GET /media) and detail (GET /media/{media_id})."""

    MEDIA_ID_INVALID = 999999

    @pytest.fixture(autouse=True)
    def _setup_class(self, client_module, db_session, seeded_multi_media):
        """Setup runs before every test method using the multi-media seed."""
        self.client = client_module
        self.db = db_session
        self.media_ids = seeded_multi_media # Has document, video, audio IDs

    # --------------------- LIST (/media) TESTS ---------------------

    def test_get_all_media_default(self):
        """Test default listing of media items."""
        response = self.client.get("/api/v1/media")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert "pagination" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) > 0 # Should have seeded items
        # Check default pagination structure
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["results_per_page"] > 0 # Default value
        assert data["pagination"]["total_items"] >= len(self.media_ids)
        assert data["pagination"]["total_pages"] >= 1

    def test_get_all_media_pagination(self):
        """Test custom pagination for listing media."""
        # Get page 1, 2 items per page
        response = self.client.get("/api/v1/media?page=1&results_per_page=2")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data["items"], list)
        assert len(data["items"]) <= 2
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["results_per_page"] == 2
        assert data["pagination"]["total_items"] >= len(self.media_ids)

        # Get page 2, 2 items per page
        response_p2 = self.client.get("/api/v1/media?page=2&results_per_page=2")
        assert response_p2.status_code == status.HTTP_200_OK
        data_p2 = response_p2.json()
        assert isinstance(data_p2["items"], list)
        # Ensure items are different from page 1 if total > results_per_page
        if data["pagination"]["total_items"] > 2:
            # Check if page 2 actually contains items
            if data["pagination"]["total_items"] > data["pagination"]["results_per_page"] * (data["pagination"]["page"] -1 ):
                 assert len(data_p2["items"]) > 0
                 item_ids_p1 = {item["id"] for item in data["items"]}
                 item_ids_p2 = {item["id"] for item in data_p2["items"]}
                 assert not item_ids_p1.intersection(item_ids_p2) # No overlap
            else:
                 assert len(data_p2["items"]) == 0 # Should be empty if past the last page
        else:
            assert len(data_p2["items"]) == 0 # Should be empty if only one page


    def test_get_all_media_invalid_pagination_params(self):
        """Test invalid pagination parameters."""
        response = self.client.get("/api/v1/media?page=0&results_per_page=10") # Page must be >= 1
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        response = self.client.get("/api/v1/media?page=1&results_per_page=0") # results must be >= 1
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_all_media_includes_basic_info(self):
        """Check if list items contain expected basic fields."""
        response = self.client.get("/api/v1/media?results_per_page=3") # Get all seeded items
        assert response.status_code == status.HTTP_200_OK
        items = response.json()["items"]
        assert len(items) >= 3

        for item in items:
            assert "id" in item
            assert "title" in item
            assert "type" in item
            assert "content" not in item # Content should not be in list view
            assert "versions" not in item # Version details usually excluded

    # --------------------- DETAIL (/media/{id}) TESTS ---------------------

    def test_get_media_item_document(self):
        """Test retrieving details of a document media item."""
        doc_id = self.media_ids["document"]
        response = self.client.get(f"/api/v1/media/{doc_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["media_id"] == doc_id
        assert data["source"]["type"] == "document"
        assert data["source"]["title"] == "Multi Test Doc"
        assert "content" in data  # Check the 'content' dict exists
        assert "text" in data["content"]  # Check 'text' inside 'content'
        assert data["content"]["text"] == "Doc content v1"
        assert "keywords" in data
        assert isinstance(data["keywords"], list)  # Keywords is top-level
        assert set(data["keywords"]) == {"multi", "test", "seed"}
        assert "versions" in data # Check if versions are included in detail view
        assert isinstance(data["versions"], list)
        assert len(data["versions"]) >= 1 # Should have at least the initial version

    def test_get_media_item_video(self):
        """Test retrieving details of a video media item."""
        vid_id = self.media_ids["video"]
        response = self.client.get(f"/api/v1/media/{vid_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["media_id"] == vid_id
        assert data["source"]["type"] == "video"
        assert data["source"]["title"] == "Multi Test Video"
        assert "content" in data
        assert "Transcript v1" in data["content"]["text"] # Check relevant part of content
        assert "webpage_url" in data["content"]["metadata"] # Check structured content
        assert "keywords" in data
        assert set(data["keywords"]) == {"multi", "test", "seed"}
        # Videos might not have 'versions' in the same way as documents
        # Verify 'versions' key exists but is likely empty or contains minimal info if not versioned like docs
        assert "versions" in data
        assert isinstance(data["versions"], list)
        assert len(data["versions"]) == 0 # Assuming videos don't have DocumentVersions

    def test_get_media_item_audio(self):
        """Test retrieving details of an audio media item."""
        audio_id = self.media_ids["audio"]
        response = self.client.get(f"/api/v1/media/{audio_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["media_id"] == audio_id
        assert data["source"]["type"] == "audio"
        assert data["source"]["title"] == "Multi Test Audio"
        assert "content" in data
        assert "Audio Transcript v1" in data["content"]["text"]
        assert "webpage_url" in data["content"]["metadata"] # Check metadata
        assert "keywords" in data
        assert set(data["keywords"]) == {"multi", "test", "seed"}
        assert "versions" in data
        assert isinstance(data["versions"], list)
        assert len(data["versions"]) == 0 # Assuming audio doesn't have DocumentVersions

    def test_get_nonexistent_media_item(self):
        """Test retrieving a media item with an ID that doesn't exist."""
        response = self.client.get(f"/api/v1/media/{self.MEDIA_ID_INVALID}")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Media not found" in response.json().get("detail", "")

    # --------------------- UPDATE (PUT /media/{id}) TESTS - Basic Placeholder ---------------------
    # These need to be adapted based on the actual PUT endpoint implementation and Pydantic model

    def test_update_media_item_title(self):
        """Test updating the title of a media item."""
        doc_id = self.media_ids["document"]
        new_title = "Updated Document Title"
        payload = {"title": new_title} # Minimal update payload

        response = self.client.put(f"/api/v1/media/{doc_id}", json=payload)

        # Check response status - 200 OK or 202 Accepted are common
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["media_id"] == doc_id

        # Verify by fetching again
        response_get = self.client.get(f"/api/v1/media/{doc_id}")
        assert response_get.status_code == status.HTTP_200_OK
        assert response_get.json()["source"]["title"] == new_title

    def test_update_media_item_nonexistent(self):
        """Test updating a media item that doesn't exist."""
        payload = {"title": "Won't Work"}
        response = self.client.put(f"/api/v1/media/{self.MEDIA_ID_INVALID}", json=payload)
        assert response.status_code == status.HTTP_404_NOT_FOUND
        # Adjust expected message based on actual PUT endpoint error handling
        assert "Media item not found or is inactive/trashed" in response.json().get("detail", "")


    def test_update_media_item_invalid_payload(self):
        """Test updating with invalid data type."""
        doc_id = self.media_ids["document"]
        payload = {"title": 12345} # Invalid type for title
        response = self.client.put(f"/api/v1/media/{doc_id}", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestSecurityAndPerformance:
    """Basic checks for security headers and response time."""

    @pytest.fixture(autouse=True)
    def _setup_class(self, client_module): # Use the shared client_module
        self.client = client_module

    def test_list_media_response_time(self):
        """Check response time for the media list endpoint."""
        start_time = time.perf_counter()
        response = self.client.get("/api/v1/media")
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        assert response.status_code == status.HTTP_200_OK # Should pass now if fixtures are fixed
        assert elapsed < 2.0, f"Response time ({elapsed:.2f}s) exceeded limit"

    def test_sql_injection_attempt_param(self):
        """Test query parameter for basic SQL injection pattern."""
        # FastAPI/Pydantic usually handles type validation preventing basic injection here
        response = self.client.get("/api/v1/media?page=1;DROP TABLE Media;")
        # Expect validation error due to non-integer page
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # FIXME - test doesn't get skipped?
    @pytest.mark.skipif(sys.platform.startswith("win32"), reason="Skipping on Windows due to PermissionError during teardown (DB file lock issue)")
    def test_content_type_enforcement_json(self, seeded_document_media):
        """Test that endpoints expecting JSON reject incorrect Content-Type."""
        # Use an endpoint that expects JSON (e.g., create version)
        # Need a valid media ID for the path, even if the payload causes the failure
        # Assume media ID 1 exists from fixtures
        media_id_for_test = seeded_document_media # Use the ID from the fixture
        response = self.client.post(
            f"/api/v1/media/{media_id_for_test}/versions",
            content='{"content": "test", "prompt": "p", "analysis_content": "s"}',
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code in [status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, status.HTTP_422_UNPROCESSABLE_ENTITY]


    # def test_cors_headers_presence(self):
    #     """Check if standard CORS headers are present on OPTIONS request."""
    #     # This assumes CORS middleware is configured in the main app
    #     response = self.client.options("/api/v1/media") # OPTIONS request to list endpoint
    #     # Status code for OPTIONS can vary (200 or 204 common)
    #     assert response.status_code in [status.HTTP_200_OK, status.HTTP_204_NO_CONTENT]
    #
    #     # Check for essential CORS headers (case-insensitive check)
    #     response_headers_lower = {k.lower() for k in response.headers}
    #     if "access-control-allow-origin" not in response_headers_lower:
    #          pytest.skip("CORS headers not configured or middleware not active for test app instance.")
    #
    #     assert "access-control-allow-origin" in response_headers_lower
    #     assert "access-control-allow-methods" in response_headers_lower
    #     assert "access-control-allow-headers" in response_headers_lower

    # Add more tests if needed (e.g., authentication checks, rate limiting)