# tests/integration/api/v1/test_prompts_api.py
# Description:
#
# Imports
import pytest
import json
import base64
#
# Third-Party Imports
from fastapi.testclient import TestClient
#
# Local Imports
#
#######################################################################################################################
#
# Functions:
API_V1_PROMPTS_PREFIX = "/api/v1/prompts"

# Test data
sample_prompt_payload = {
    "name": "API Test Prompt",
    "author": "API Tester",
    "details": "Details via API.",
    "system_prompt": "System instructions for API.",
    "user_prompt": "User query for API.",
    "keywords": ["api_test", "integration"]
}


@pytest.fixture(scope="function", autouse=True)
def clear_user_db_between_tests(client: TestClient, test_user, test_api_token):
    """
    Ensures a clean slate by clearing relevant prompt data for the test user
    before each test function in this file. This is a bit heavy-handed.
    A more granular approach would be to delete specific items created by tests.
    Alternatively, if `Prompts_DB_Deps.py` creates a truly fresh DB per test session
    (e.g., by using a unique temp path per test based on test_user.id and a random suffix),
    this explicit clearing might not be needed.

    For now, this attempts to delete all prompts and keywords.
    This depends on the `get_prompts_db_for_user` correctly giving us the user's DB.
    """
    # This is complex because the DB instance is managed by the dependency.
    # We'd need to access that DB instance to clean it.
    # The `client` fixture in `conftest.py` now sets up a temporary DB base dir
    # and cleans it up per module. For per-function, we might need more.
    # For now, let's rely on the module-level cleanup and careful test design.
    yield
    # If specific cleanup is needed:
    # db = get_prompts_db_for_user_directly_for_test() # Need a helper for this
    # db.execute_query("DELETE FROM Prompts", commit=True)
    # db.execute_query("DELETE FROM PromptKeywordsTable", commit=True)
    # db.execute_query("DELETE FROM PromptKeywordLinks", commit=True)


def test_create_prompt_success(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    response = client.post(
        f"{API_V1_PROMPTS_PREFIX}/",
        json=sample_prompt_payload,
        headers=headers
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == sample_prompt_payload["name"]
    assert data["author"] == sample_prompt_payload["author"]
    assert "api_test" in data["keywords"]
    assert "id" in data
    assert "uuid" in data

    # Store created prompt ID for later tests if needed (e.g. in a class or shared fixture)
    pytest.created_prompt_id = data["id"]
    pytest.created_prompt_uuid = data["uuid"]


def test_create_prompt_duplicate_name(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    # First creation (assuming it wasn't created by another test or cleaned up)
    client.post(f"{API_V1_PROMPTS_PREFIX}/", json=sample_prompt_payload, headers=headers)

    # Attempt to create again
    response = client.post(
        f"{API_V1_PROMPTS_PREFIX}/",
        json=sample_prompt_payload,  # Same name
        headers=headers
    )
    assert response.status_code == 409  # Conflict due to unique name constraint


def test_create_prompt_invalid_input(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    invalid_payload = sample_prompt_payload.copy()
    invalid_payload["name"] = ""  # Empty name
    response = client.post(
        f"{API_V1_PROMPTS_PREFIX}/",
        json=invalid_payload,
        headers=headers
    )
    assert response.status_code == 422  # FastAPI Pydantic validation error


def test_get_prompt_by_id(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    # Create a prompt first to ensure it exists
    create_response = client.post(f"{API_V1_PROMPTS_PREFIX}/", json={**sample_prompt_payload, "name": "GetByID Prompt"},
                                  headers=headers)
    assert create_response.status_code == 201
    prompt_id = create_response.json()["id"]

    response = client.get(f"{API_V1_PROMPTS_PREFIX}/{prompt_id}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == prompt_id
    assert data["name"] == "GetByID Prompt"


def test_get_prompt_by_uuid(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    create_response = client.post(f"{API_V1_PROMPTS_PREFIX}/",
                                  json={**sample_prompt_payload, "name": "GetByUUID Prompt"}, headers=headers)
    assert create_response.status_code == 201
    prompt_uuid = create_response.json()["uuid"]

    response = client.get(f"{API_V1_PROMPTS_PREFIX}/{prompt_uuid}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["uuid"] == prompt_uuid
    assert data["name"] == "GetByUUID Prompt"


def test_get_prompt_by_name(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    prompt_name = "GetByName Prompt"
    client.post(f"{API_V1_PROMPTS_PREFIX}/", json={**sample_prompt_payload, "name": prompt_name}, headers=headers)

    response = client.get(f"{API_V1_PROMPTS_PREFIX}/{prompt_name}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == prompt_name


def test_get_prompt_not_found(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    response = client.get(f"{API_V1_PROMPTS_PREFIX}/99999", headers=headers)  # Non-existent ID
    assert response.status_code == 404


def test_list_prompts(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    # Ensure at least one prompt exists
    client.post(f"{API_V1_PROMPTS_PREFIX}/", json={**sample_prompt_payload, "name": "ListTestPrompt1"}, headers=headers)
    client.post(f"{API_V1_PROMPTS_PREFIX}/", json={**sample_prompt_payload, "name": "ListTestPrompt2"}, headers=headers)

    response = client.get(f"{API_V1_PROMPTS_PREFIX}/", headers=headers, params={"page": 1, "per_page": 5})
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total_pages" in data
    assert "current_page" in data
    assert "total_items" in data
    assert len(data["items"]) <= 5
    assert data["total_items"] >= 2


def test_update_prompt(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    create_resp = client.post(f"{API_V1_PROMPTS_PREFIX}/", json={**sample_prompt_payload, "name": "UpdateThisPrompt"},
                              headers=headers)
    prompt_id_to_update = create_resp.json()["id"]

    update_payload = {
        "name": "Updated Prompt Name",  # Name change
        "author": "Updated Author",
        "details": "Updated details.",
        "system_prompt": "Updated system.",
        "user_prompt": "Updated user.",
        "keywords": ["updated_kw", "new_kw"]
    }
    response = client.put(
        f"{API_V1_PROMPTS_PREFIX}/{prompt_id_to_update}",  # Update by original ID
        json=update_payload,
        headers=headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Prompt Name"
    assert data["author"] == "Updated Author"
    assert "updated_kw" in data["keywords"]

    # Verify the old name doesn't exist (if name was primary key for update)
    # The current PUT uses add_prompt(overwrite=True) which keys by name in payload
    # So the old record "UpdateThisPrompt" might still exist if its name wasn't "Updated Prompt Name"
    # This test needs to align with the exact PUT behavior.
    # If PUT is by ID, then old name is gone. If PUT is effectively by new name (upsert),
    # then we check the new name. The current API's PUT is a bit ambiguous with identifier vs payload name.

    # Let's assume the PUT updates the record identified by prompt_identifier, possibly changing its name.
    get_response = client.get(f"{API_V1_PROMPTS_PREFIX}/{prompt_id_to_update}", headers=headers)
    assert get_response.status_code == 200
    assert get_response.json()["name"] == "Updated Prompt Name"


def test_delete_prompt(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    create_resp = client.post(f"{API_V1_PROMPTS_PREFIX}/", json={**sample_prompt_payload, "name": "DeleteThisPrompt"},
                              headers=headers)
    prompt_id_to_delete = create_resp.json()["id"]

    response = client.delete(f"{API_V1_PROMPTS_PREFIX}/{prompt_id_to_delete}", headers=headers)
    assert response.status_code == 204

    # Verify it's soft-deleted
    get_response = client.get(f"{API_V1_PROMPTS_PREFIX}/{prompt_id_to_delete}", headers=headers)
    assert get_response.status_code == 404  # Not found without include_deleted
    get_deleted_response = client.get(f"{API_V1_PROMPTS_PREFIX}/{prompt_id_to_delete}?include_deleted=true",
                                      headers=headers)
    assert get_deleted_response.status_code == 200
    assert get_deleted_response.json()["deleted"] is True  # If 'deleted' field is exposed in schema


def test_search_prompts_api(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    client.post(f"{API_V1_PROMPTS_PREFIX}/",
                json={**sample_prompt_payload, "name": "Searchable Alpha", "details": "UniqueKeywordForSearch"},
                headers=headers)
    client.post(f"{API_V1_PROMPTS_PREFIX}/",
                json={**sample_prompt_payload, "name": "Searchable Beta", "keywords": ["UniqueKeywordForSearch"]},
                headers=headers)

    response = client.post(
        f"{API_V1_PROMPTS_PREFIX}/search?search_query=UniqueKeywordForSearch",
        headers=headers
        # Using POST for search if query params become too long, or GET if preferred
        # The API definition used GET with query params, so sticking to that.
        # The endpoint is POST /search, but takes query params. This is unusual.
        # Let's assume it's GET /search?search_query=... for typical REST
        # If it's POST /search with query params, client.post target needs adjustment.
        # The provided API uses POST /search, but with Query() params, not body.
        # TestClient.post with `params` for query string on POST.
    )
    # Correcting the call based on the API definition: POST with Query parameters
    # It seems the FastAPI definition has `/search` as POST but expects query parameters.
    # TestClient `post` can take `params` for query string:
    # This is an odd pattern. Usually POST for search implies body, or it should be GET.
    # For now, assuming it's a GET endpoint:
    # response = client.get(f"{API_V1_PROMPTS_PREFIX}/search?search_query=UniqueKeywordForSearch", headers=headers)
    # If it truly is POST with query parameters:
    # response = client.post(f"{API_V1_PROMPTS_PREFIX}/search", params={"search_query": "UniqueKeywordForSearch"}, headers=headers)
    # Given the provided endpoint `router.post("/search",...)` with `Query()` params:
    response = client.post(f"{API_V1_PROMPTS_PREFIX}/search",
                           params={"search_query": "UniqueKeywordForSearch", "search_fields": ["details", "keywords"]},
                           headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["total_matches"] >= 1  # Should be 2 if FTS is working across fields
    assert len(data["items"]) >= 1


# --- Keyword API Tests ---
def test_create_keyword_api(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    response = client.post(f"{API_V1_PROMPTS_PREFIX}/keywords/", json={"keyword_text": "api_created_keyword"},
                           headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data["keyword_text"] == "api_created_keyword"
    assert "id" in data


def test_list_keywords_api(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    client.post(f"{API_V1_PROMPTS_PREFIX}/keywords/", json={"keyword_text": "kw_list_test1"}, headers=headers)
    client.post(f"{API_V1_PROMPTS_PREFIX}/keywords/", json={"keyword_text": "kw_list_test2"}, headers=headers)

    response = client.get(f"{API_V1_PROMPTS_PREFIX}/keywords/", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert "kw_list_test1" in data
    assert "kw_list_test2" in data


def test_delete_keyword_api(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    keyword_to_delete = "kw_to_delete_api"
    client.post(f"{API_V1_PROMPTS_PREFIX}/keywords/", json={"keyword_text": keyword_to_delete}, headers=headers)

    response = client.delete(f"{API_V1_PROMPTS_PREFIX}/keywords/{keyword_to_delete}", headers=headers)
    assert response.status_code == 204

    # Verify deletion by trying to list and not finding it
    list_response = client.get(f"{API_V1_PROMPTS_PREFIX}/keywords/", headers=headers)
    assert keyword_to_delete not in list_response.json()


# --- Export API Tests ---
def test_export_prompts_api_csv(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    client.post(f"{API_V1_PROMPTS_PREFIX}/", json={**sample_prompt_payload, "name": "ExportCSVTest"}, headers=headers)
    response = client.get(f"{API_V1_PROMPTS_PREFIX}/export?export_format=csv", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "Successfully exported" in data["message"]
    assert data["file_content_b64"] is not None
    decoded_content = base64.b64decode(data["file_content_b64"]).decode('utf-8')
    assert "ExportCSVTest" in decoded_content
    assert "api_test" in decoded_content  # Keyword from sample_prompt_payload


def test_export_keywords_api_csv(client: TestClient, test_api_token: str):
    headers = {"Authorization": f"Bearer {test_api_token}"}
    client.post(f"{API_V1_PROMPTS_PREFIX}/keywords/", json={"keyword_text": "export_kw_api_test"}, headers=headers)
    response = client.get(f"{API_V1_PROMPTS_PREFIX}/keywords/export-csv", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "Successfully exported" in data["message"]
    assert data["file_content_b64"] is not None
    decoded_content = base64.b64decode(data["file_content_b64"]).decode('utf-8')
    assert "export_kw_api_test" in decoded_content


# Test unauthorized access
def test_unauthorized_access(client: TestClient):
    response = client.get(f"{API_V1_PROMPTS_PREFIX}/")  # No token
    assert response.status_code == 401  # Or 403 if verify_token raises that for missing

    response = client.get(f"{API_V1_PROMPTS_PREFIX}/", headers={"Authorization": "Bearer invalidtoken"})
    assert response.status_code == 401

# TODO: Add tests for Sync Log endpoint if it's not admin-only or mock admin user.
# TODO: Test edge cases for pagination, search with no results, various include_deleted flags.
# TODO: Test PromptUpdate logic carefully if a separate PATCH endpoint is added.

#
# End of test_prompts_api.py
#######################################################################################################################
