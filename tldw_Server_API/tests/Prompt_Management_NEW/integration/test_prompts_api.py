"""
Integration tests for Prompt Management API endpoints.

Tests the complete API flow with real components, no mocking.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.prompts import router as prompts_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

pytestmark = pytest.mark.integration

# ========================================================================
# Prompt CRUD Endpoint Tests
# ========================================================================

class TestPromptCRUDEndpoints:
    """Test prompt CRUD operations through API."""

    @pytest.mark.integration
    def test_create_prompt_endpoint(self, test_client, auth_headers, sample_prompt):
        """Test creating a prompt via API."""
        payload = {
            'name': sample_prompt['name'],
            'author': sample_prompt.get('author'),
            'details': sample_prompt.get('content'),
            'keywords': sample_prompt.get('keywords', [])
        }
        response = test_client.post(
            "/api/v1/prompts",
            json=payload,
            headers=auth_headers
        )

        assert response.status_code == 201
        data = response.json()
        assert 'id' in data and data['id'] > 0
        assert data['name'] == payload['name']

    @pytest.mark.integration
    def test_legacy_create_prompt_endpoint_returns_created(self, test_client, auth_headers):
        """Legacy /create route should behave like a create operation."""
        response = test_client.post(
            "/api/v1/prompts/create",
            json={
                "name": "Legacy Create Prompt",
                "content": "Legacy content",
                "author": "legacy_test",
            },
            headers=auth_headers,
        )

        assert response.status_code == 201, response.text
        data = response.json()
        assert "prompt_id" in data
        assert isinstance(data["prompt_id"], int)
        assert data["prompt_id"] > 0

    @pytest.mark.integration
    def test_get_prompt_endpoint(self, test_client, auth_headers):
        """Test getting a prompt via API."""
        create_response = test_client.post(
            "/api/v1/prompts",
            json={
                'name': 'API Test Prompt',
                'details': 'Test content {{variable}}',
                'author': 'api_test',
                'keywords': ['test', 'api']
            },
            headers=auth_headers
        )
        prompt_id = create_response.json()['id']

        # Then get it
        response = test_client.get(
            f"/api/v1/prompts/{prompt_id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data['name'] == 'API Test Prompt'
        assert data.get('details') == 'Test content {{variable}}'
        assert 'test' in data.get('keywords', [])

    @pytest.mark.integration
    def test_list_prompts_endpoint(self, test_client, auth_headers):
        """Test listing prompts via API."""
        # Seed a couple prompts
        for i in range(2):
            test_client.post(
                "/api/v1/prompts",
                json={'name': f'List Seed {i}', 'author': 'seed', 'details': f'Details {i}'},
                headers=auth_headers
            )
        response = test_client.get(
            "/api/v1/prompts",
            params={'page': 1, 'per_page': 2},
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert 'items' in data
        assert len(data['items']) <= 2

    @pytest.mark.integration
    def test_update_prompt_endpoint(self, test_client, auth_headers):
        """Test updating a prompt via API."""
        # Create prompt
        create_response = test_client.post(
            "/api/v1/prompts",
            json={'name': 'Update Test', 'details': 'Original content', 'author': 'test'},
            headers=auth_headers
        )
        prompt = create_response.json()
        prompt_id = prompt['id']

        # Update it: PromptCreate schema requires name; keep name, change details
        update_response = test_client.put(
            f"/api/v1/prompts/{prompt_id}",
            json={
                'name': prompt['name'],
                'author': prompt.get('author'),
                'details': 'Updated content {{new_var}}',
                'system_prompt': prompt.get('system_prompt'),
                'user_prompt': prompt.get('user_prompt'),
                'keywords': prompt.get('keywords', [])
            },
            headers=auth_headers
        )

        assert update_response.status_code == 200
        data = update_response.json()
        assert data['id'] == prompt_id
        assert data.get('details') == 'Updated content {{new_var}}'

    @pytest.mark.integration
    def test_record_prompt_usage_endpoint(self, test_client, auth_headers):
        """Test recording prompt usage increments usage_count and updates last_used_at."""
        create_response = test_client.post(
            "/api/v1/prompts",
            json={'name': 'Usage Test', 'details': 'Track this prompt', 'author': 'test'},
            headers=auth_headers
        )
        prompt_id = create_response.json()['id']

        first_use = test_client.post(
            f"/api/v1/prompts/{prompt_id}/use",
            headers=auth_headers
        )
        assert first_use.status_code == 200
        first_payload = first_use.json()
        assert first_payload.get('usage_count') == 1
        assert first_payload.get('last_used_at') is not None

        second_use = test_client.post(
            f"/api/v1/prompts/{prompt_id}/use",
            headers=auth_headers
        )
        assert second_use.status_code == 200
        second_payload = second_use.json()
        assert second_payload.get('usage_count') == 2
        assert second_payload.get('last_used_at') is not None

    @pytest.mark.integration
    def test_delete_prompt_endpoint(self, test_client, auth_headers):
        """Test deleting a prompt via API."""
        create_response = test_client.post(
            "/api/v1/prompts",
            json={'name': 'Delete Test', 'details': 'To be deleted', 'author': 'test'},
            headers=auth_headers
        )
        prompt_id = create_response.json()['id']

        delete_response = test_client.delete(
            f"/api/v1/prompts/{prompt_id}",
            headers=auth_headers
        )

        assert delete_response.status_code == 204

        # Verify it's deleted (soft delete)
        get_response = test_client.get(
            f"/api/v1/prompts/{prompt_id}",
            headers=auth_headers
        )
        assert get_response.status_code == 404

# ========================================================================
# Version Management Endpoint Tests
# ========================================================================

class TestVersionEndpoints:
    """Versioning endpoints are not available in current API."""

    @pytest.mark.integration
    def test_versions_endpoints(self, test_client, auth_headers):
        """Test listing and restoring prompt versions via API."""
        create_response = test_client.post(
            "/api/v1/prompts",
            json={
                'name': 'Versioned Prompt',
                'details': 'Initial version content',
                'author': 'api_test'
            },
            headers=auth_headers
        )
        prompt = create_response.json()
        prompt_id = prompt['id']

        update_response = test_client.put(
            f"/api/v1/prompts/{prompt_id}",
            json={
                'name': prompt['name'],
                'author': prompt.get('author'),
                'details': 'Updated version content',
                'system_prompt': prompt.get('system_prompt'),
                'user_prompt': prompt.get('user_prompt'),
                'keywords': prompt.get('keywords', [])
            },
            headers=auth_headers
        )
        assert update_response.status_code == 200

        versions_response = test_client.get(
            f"/api/v1/prompts/{prompt_id}/versions",
            headers=auth_headers
        )
        assert versions_response.status_code == 200
        versions = versions_response.json()
        assert len(versions) >= 2
        version_numbers = [v['version'] for v in versions]
        assert version_numbers == sorted(version_numbers)

        restore_response = test_client.post(
            f"/api/v1/prompts/{prompt_id}/versions/1/restore",
            headers=auth_headers
        )
        assert restore_response.status_code == 200
        restored = restore_response.json()
        assert restored.get('details') == 'Initial version content'

# ========================================================================
# Search and Filter Endpoint Tests
# ========================================================================

class TestSearchEndpoints:
    """Test search and filter endpoints."""

    @pytest.mark.integration
    def test_search_prompts_endpoint(self, test_client, auth_headers):
        """Test searching prompts via API."""
        # Seed a prompt
        test_client.post(
            "/api/v1/prompts",
            json={'name': 'Assistant Prompt', 'author': 'test', 'details': 'You are a helpful assistant.'},
            headers=auth_headers
        )
        response = test_client.post(
            "/api/v1/prompts/search",
            params={'search_query': 'assistant', 'search_fields': ['details']},
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert 'items' in data and data['total_matches'] >= 1

    @pytest.mark.integration
    def test_search_by_keywords(self, test_client, auth_headers):
        """Search using keywords field via FTS."""
        # Seed a prompt with keywords by creating then updating keywords through create payload
        test_client.post(
            "/api/v1/prompts",
            json={'name': 'Keyword Prompt', 'author': 'test', 'details': 'Has keywords', 'keywords': ['test', 'assistant']},
            headers=auth_headers
        )
        response = test_client.post(
            "/api/v1/prompts/search",
            params={'search_query': 'assistant', 'search_fields': ['keywords']},
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data['total_matches'] >= 1

    @pytest.mark.integration
    def test_search_by_author(self, test_client, auth_headers):
        """Search by author via search endpoint."""
        test_client.post(
            "/api/v1/prompts",
            json={'name': 'Author Prompt', 'author': 'author_user', 'details': 'foobar'},
            headers=auth_headers
        )
        response = test_client.post(
            "/api/v1/prompts/search",
            params={'search_query': 'author_user', 'search_fields': ['author']},
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data['total_matches'] >= 1

# ========================================================================
# Import/Export Endpoint Tests
# ========================================================================

class TestImportExportEndpoints:
    """Test export endpoints; import endpoints not present in current API."""

    @pytest.mark.integration
    def test_export_prompts_endpoint(self, test_client, auth_headers):
        """Test exporting prompts via API (CSV or Markdown)."""
        # Seed a prompt
        test_client.post(
            "/api/v1/prompts",
            json={'name': 'Export Prompt', 'author': 'test', 'details': 'Export me', 'keywords': ['exp']},
            headers=auth_headers
        )
        response = test_client.get(
            "/api/v1/prompts/export",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        # file_content_b64 may be None if no prompts, but here we seeded one
        assert 'file_content_b64' in data

    @pytest.mark.integration
    def test_import_prompts_endpoint(self, test_client, auth_headers):
        """Test importing prompts via API."""
        payload = {
            "prompts": [
                {
                    "name": "ImportPromptOne",
                    "content": "Imported content one",
                    "author": "import_test",
                    "keywords": ["import"]
                },
                {
                    "name": "ImportPromptTwo",
                    "content": "Imported content two",
                    "author": "import_test",
                    "keywords": ["import", "two"]
                }
            ]
        }
        response = test_client.post(
            "/api/v1/prompts/import",
            json=payload,
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data['imported'] == 2
        assert data['failed'] == 0
        assert len(data['prompt_ids']) == 2

        get_response = test_client.get(
            "/api/v1/prompts/ImportPromptOne",
            headers=auth_headers
        )
        assert get_response.status_code == 200
        imported_prompt = get_response.json()
        assert imported_prompt.get('details') == 'Imported content one'

# ========================================================================
# Template Processing Endpoint Tests
# ========================================================================

class TestTemplateEndpoints:
    """Template-specific endpoints are not available in current API."""

    @pytest.mark.integration
    def test_template_endpoints(self, test_client, auth_headers):
        """Test template variable extraction and rendering endpoints."""
        variables_response = test_client.post(
            "/api/v1/prompts/templates/variables",
            json={"template": "Hello {{name}}, today is {{day}}."},
            headers=auth_headers
        )
        assert variables_response.status_code == 200
        variables = variables_response.json().get('variables', [])
        assert 'name' in variables
        assert 'day' in variables

        render_response = test_client.post(
            "/api/v1/prompts/templates/render",
            json={
                "template": "Hello {{name}}, today is {{day}}.",
                "variables": {"name": "Ada", "day": "Monday"}
            },
            headers=auth_headers
        )
        assert render_response.status_code == 200
        rendered = render_response.json().get('rendered')
        assert "Ada" in rendered
        assert "Monday" in rendered
        assert "{{" not in rendered

# ========================================================================
# Bulk Operations Endpoint Tests
# ========================================================================

class TestBulkOperationsEndpoints:
    """Bulk endpoints not present in current API."""

    @pytest.mark.integration
    def test_bulk_endpoints(self, test_client, auth_headers):
        """Test bulk keyword update and delete endpoints."""
        prompt_ids = []
        for i in range(3):
            response = test_client.post(
                "/api/v1/prompts",
                json={
                    'name': f'Bulk Prompt {i}',
                    'details': f'Bulk content {i}',
                    'author': 'bulk_test'
                },
                headers=auth_headers
            )
            prompt_ids.append(response.json()['id'])

        keywords_response = test_client.post(
            "/api/v1/prompts/bulk/keywords",
            json={
                "prompt_ids": prompt_ids,
                "add_keywords": ["bulk-tag"],
                "remove_keywords": []
            },
            headers=auth_headers
        )
        assert keywords_response.status_code == 200
        keywords_data = keywords_response.json()
        assert keywords_data['updated'] == len(prompt_ids)
        assert keywords_data['failed'] == 0

        get_response = test_client.get(
            f"/api/v1/prompts/{prompt_ids[0]}",
            headers=auth_headers
        )
        assert get_response.status_code == 200
        assert 'bulk-tag' in get_response.json().get('keywords', [])

        delete_response = test_client.post(
            "/api/v1/prompts/bulk/delete",
            json={"prompt_ids": prompt_ids},
            headers=auth_headers
        )
        assert delete_response.status_code == 200
        delete_data = delete_response.json()
        assert delete_data['deleted'] == len(prompt_ids)
        assert delete_data['failed'] == 0

        verify_response = test_client.get(
            f"/api/v1/prompts/{prompt_ids[0]}",
            headers=auth_headers
        )
        assert verify_response.status_code == 404

# ========================================================================
# Collection Management Endpoint Tests
# ========================================================================

class TestCollectionEndpoints:
    """Test collection management endpoints."""

    @pytest.mark.integration
    def test_list_collections_empty_endpoint(self, test_client, auth_headers):
        """Listing collections should return an empty list when none exist."""
        response = test_client.get(
            "/api/v1/prompts/collections",
            headers=auth_headers,
        )

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload == {
            "collections": [],
            "total": 0,
            "limit": 200,
            "offset": 0,
            "has_more": False,
            "next_offset": None,
            "pagination": {
                "mode": "offset",
                "total": 0,
                "limit": 200,
                "offset": 0,
                "has_more": False,
                "next_offset": None,
            },
        }

    @pytest.mark.integration
    def test_create_collection_endpoint(self, test_client, auth_headers):
        """Test creating a collection via API."""
        # Create prompts for collection
        prompt_ids = []
        for i in range(3):
            response = test_client.post(
                "/api/v1/prompts/create",
                json={
                    'name': f'Collection Item {i}',
                    'content': f'Content {i}',
                    'author': 'test'
                },
                headers=auth_headers
            )
            prompt_ids.append(response.json()['prompt_id'])

        # Create collection
        collection_response = test_client.post(
            "/api/v1/prompts/collections/create",
            json={
                'name': 'Test Collection',
                'description': 'A test collection',
                'prompt_ids': prompt_ids
            },
            headers=auth_headers
        )

        assert collection_response.status_code == 200
        data = collection_response.json()
        assert 'collection_id' in data
        assert data['collection_id'] > 0

    @pytest.mark.integration
    def test_get_collection_endpoint(self, test_client, auth_headers):
        """Test getting a collection via API."""
        # Create collection
        create_response = test_client.post(
            "/api/v1/prompts/collections/create",
            json={
                'name': 'Get Test',
                'description': 'Test',
                'prompt_ids': []
            },
            headers=auth_headers
        )
        collection_id = create_response.json()['collection_id']

        # Get collection
        get_response = test_client.get(
            f"/api/v1/prompts/collections/{collection_id}",
            headers=auth_headers
        )

        assert get_response.status_code == 200
        data = get_response.json()
        assert data['name'] == 'Get Test'
        assert data['description'] == 'Test'

    @pytest.mark.integration
    def test_list_collections_endpoint(self, test_client, auth_headers):
        """Test listing collections via API."""
        first = test_client.post(
            "/api/v1/prompts/collections/create",
            json={"name": "List Collection A", "description": "A", "prompt_ids": []},
            headers=auth_headers,
        )
        second = test_client.post(
            "/api/v1/prompts/collections/create",
            json={"name": "List Collection B", "description": "B", "prompt_ids": []},
            headers=auth_headers,
        )
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text

        list_response = test_client.get(
            "/api/v1/prompts/collections",
            headers=auth_headers,
        )

        assert list_response.status_code == 200, list_response.text
        payload = list_response.json()
        assert "collections" in payload
        names = {item["name"] for item in payload["collections"]}
        assert {"List Collection A", "List Collection B"}.issubset(names)
        assert payload["total"] >= 2
        assert payload["limit"] == 200
        assert payload["offset"] == 0
        assert payload["pagination"]["mode"] == "offset"
        assert payload["pagination"]["total"] >= 2
        assert payload["pagination"]["limit"] == 200
        assert payload["pagination"]["offset"] == 0
        assert payload["pagination"]["has_more"] is False
        assert payload["pagination"]["next_offset"] is None
        assert payload["has_more"] is False
        assert payload["next_offset"] is None

    @pytest.mark.integration
    def test_update_collection_endpoint(self, test_client, auth_headers):
        """Test updating collection metadata and prompt membership."""
        prompt_one = test_client.post(
            "/api/v1/prompts/create",
            json={"name": "Update Col Prompt 1", "content": "P1", "author": "test"},
            headers=auth_headers,
        )
        prompt_two = test_client.post(
            "/api/v1/prompts/create",
            json={"name": "Update Col Prompt 2", "content": "P2", "author": "test"},
            headers=auth_headers,
        )
        assert prompt_one.status_code == 201, prompt_one.text
        assert prompt_two.status_code == 201, prompt_two.text

        create_response = test_client.post(
            "/api/v1/prompts/collections/create",
            json={
                "name": "Original Collection",
                "description": "Original",
                "prompt_ids": [prompt_one.json()["prompt_id"]],
            },
            headers=auth_headers,
        )
        assert create_response.status_code == 200, create_response.text
        collection_id = create_response.json()["collection_id"]

        update_response = test_client.put(
            f"/api/v1/prompts/collections/{collection_id}",
            json={
                "name": "Updated Collection",
                "description": "Updated description",
                "prompt_ids": [
                    prompt_one.json()["prompt_id"],
                    prompt_two.json()["prompt_id"],
                ],
            },
            headers=auth_headers,
        )

        assert update_response.status_code == 200, update_response.text
        updated_payload = update_response.json()
        assert updated_payload["name"] == "Updated Collection"
        assert updated_payload["description"] == "Updated description"
        assert sorted(updated_payload["prompt_ids"]) == sorted(
            [prompt_one.json()["prompt_id"], prompt_two.json()["prompt_id"]]
        )

        get_response = test_client.get(
            f"/api/v1/prompts/collections/{collection_id}",
            headers=auth_headers,
        )
        assert get_response.status_code == 200, get_response.text
        persisted = get_response.json()
        assert persisted["name"] == "Updated Collection"
        assert sorted(persisted["prompt_ids"]) == sorted(
            [prompt_one.json()["prompt_id"], prompt_two.json()["prompt_id"]]
        )


class TestCollectionTenantIsolation:
    """Collections should remain isolated by authenticated user scope."""

    @pytest.mark.integration
    def test_collection_ids_are_not_shared_across_users(self, test_env_vars, auth_headers):
        app = FastAPI()
        app.include_router(prompts_router, prefix="/api/v1/prompts")

        active_user = {"id": 101}

        async def _override_user():
            uid = int(active_user["id"])
            return User(
                id=uid,
                username=f"user_{uid}",
                email=None,
                role="admin",
                is_active=True,
                is_verified=True,
                is_superuser=True,
                roles=["admin"],
                permissions=["*"],
                is_admin=True,
            )

        app.dependency_overrides[get_request_user] = _override_user

        with TestClient(app) as client:
            p1 = client.post(
                "/api/v1/prompts",
                json={"name": "Tenant A Prompt 1", "details": "A1"},
                headers=auth_headers,
            )
            p2 = client.post(
                "/api/v1/prompts",
                json={"name": "Tenant A Prompt 2", "details": "A2"},
                headers=auth_headers,
            )
            assert p1.status_code == 201, p1.text
            assert p2.status_code == 201, p2.text

            create_collection = client.post(
                "/api/v1/prompts/collections/create",
                json={
                    "name": "Tenant A Collection",
                    "description": "Owned by A",
                    "prompt_ids": [p1.json()["id"], p2.json()["id"]],
                },
                headers=auth_headers,
            )
            assert create_collection.status_code == 200, create_collection.text
            collection_id = create_collection.json()["collection_id"]

            active_user["id"] = 202
            cross_tenant_get = client.get(
                f"/api/v1/prompts/collections/{collection_id}",
                headers=auth_headers,
            )
            assert cross_tenant_get.status_code == 404, cross_tenant_get.text

            active_user["id"] = 101
            owner_get = client.get(
                f"/api/v1/prompts/collections/{collection_id}",
                headers=auth_headers,
            )
            assert owner_get.status_code == 200, owner_get.text
            assert owner_get.json()["name"] == "Tenant A Collection"

# ========================================================================
# Error Handling Endpoint Tests
# ========================================================================

class TestErrorHandling:
    """Test API error handling."""

    @pytest.mark.integration
    def test_not_found_error(self, test_client, auth_headers):
        """Test 404 for non-existent prompt."""
        response = test_client.get(
            "/api/v1/prompts/99999",
            headers=auth_headers
        )

        assert response.status_code == 404
        data = response.json()
        assert 'detail' in data

    @pytest.mark.integration
    def test_validation_error(self, test_client, auth_headers):
        """Test 422 for invalid data."""
        response = test_client.post(
            "/api/v1/prompts",
            json={
                'name': '',  # Empty name
                'details': 'Content'
            },
            headers=auth_headers
        )

        assert response.status_code == 422
        data = response.json()
        assert 'detail' in data

    @pytest.mark.integration
    def test_unauthorized_error(self, test_client):
        """Test 401 for missing auth."""
        response = test_client.get("/api/v1/prompts")

        assert response.status_code in [401, 403]

    @pytest.mark.integration
    def test_duplicate_prompt_error(self, test_client, auth_headers):
        """Test handling duplicate prompt names."""
        # Create first prompt
        first_response = test_client.post(
            "/api/v1/prompts",
            json={
                'name': 'Unique Name',
                'details': 'Content 1',
                'author': 'test'
            },
            headers=auth_headers
        )
        # Creation may return 201 (Created) or 200 depending on API semantics
        assert first_response.status_code in [200, 201]

        # Try to create duplicate
        duplicate_response = test_client.post(
            "/api/v1/prompts",
            json={
                'name': 'Unique Name',
                'details': 'Content 2',
                'author': 'test'
            },
            headers=auth_headers
        )

        assert duplicate_response.status_code in [400, 409]

# ========================================================================
# Pagination and Filtering Tests
# ========================================================================

class TestPaginationAndFiltering:
    """Test pagination and filtering in endpoints."""

    @pytest.mark.integration
    def test_pagination_endpoint(self, test_client, auth_headers):
        """Test pagination in list endpoint."""
        # Seed a few prompts
        for i in range(4):
            test_client.post(
                "/api/v1/prompts",
                json={'name': f'Paginate {i}', 'author': 'test', 'details': f'Details {i}'},
                headers=auth_headers
            )
        # Get first page
        page1_response = test_client.get(
            "/api/v1/prompts",
            params={'page': 1, 'per_page': 2},
            headers=auth_headers
        )

        assert page1_response.status_code == 200
        page1_data = page1_response.json()
        assert len(page1_data['items']) <= 2

        # Get second page
        page2_response = test_client.get(
            "/api/v1/prompts",
            params={'page': 2, 'per_page': 2},
            headers=auth_headers
        )

        assert page2_response.status_code == 200
        page2_data = page2_response.json()

        # Ensure different results
        if page1_data['items'] and page2_data['items']:
            assert page1_data['items'][0]['id'] != page2_data['items'][0]['id']

    @pytest.mark.integration
    def test_sorting_endpoint(self, test_client, auth_headers):
        """Test sorting in list endpoint."""
        name_a = "Sort Test A"
        name_z = "Sort Test Z"
        for name in (name_a, name_z):
            test_client.post(
                "/api/v1/prompts",
                json={'name': name, 'author': 'sort_test', 'details': f'Details for {name}'},
                headers=auth_headers
            )

        response = test_client.get(
            "/api/v1/prompts",
            params={'page': 1, 'per_page': 100, 'sort_by': 'name', 'sort_order': 'asc'},
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        names = [item['name'] for item in data.get('items', [])]
        assert name_a in names and name_z in names
        assert names.index(name_a) < names.index(name_z)
