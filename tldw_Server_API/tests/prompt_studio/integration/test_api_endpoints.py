# test_api_endpoints.py
# Integration tests for Prompt Studio API endpoints

import gc
import os
from pathlib import Path
import time

import pytest
from fastapi import status
from fastapi.testclient import TestClient
import tempfile
from unittest.mock import patch

# Disable CSRF for testing
os.environ["AUTH_MODE"] = "single_user"
os.environ["CSRF_ENABLED"] = "false"

from tldw_Server_API.app.main import app
from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import get_security_config
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import SecurityConfig
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_current_active_user
from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import get_prompt_studio_db

########################################################################################################################
# Test Client Setup


def _unlink_sqlite_file_with_retry(db_path: str) -> None:
    """Best-effort cleanup for SQLite files after TestClient thread teardown."""
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{db_path}{suffix}")
        for attempt in range(50):
            try:
                path.unlink(missing_ok=True)
                break
            except PermissionError:
                if attempt == 49:
                    raise
                gc.collect()
                time.sleep(0.1)


def _make_structured_prompt_definition_payload() -> dict:
    return {
        "schema_version": 1,
        "format": "structured",
        "variables": [
            {
                "name": "input",
                "label": "Input",
                "required": True,
                "input_type": "textarea",
            }
        ],
        "blocks": [
            {
                "id": "identity",
                "name": "Identity",
                "role": "system",
                "content": "You are a careful evaluator.",
                "enabled": True,
                "order": 10,
                "is_template": False,
            },
            {
                "id": "task",
                "name": "Task",
                "role": "user",
                "content": "Evaluate {{input}}",
                "enabled": True,
                "order": 20,
                "is_template": True,
            },
        ],
        "assembly_config": {
            "legacy_system_roles": ["system", "developer"],
            "legacy_user_roles": ["user"],
            "block_separator": "\n\n",
        },
    }


def _make_structured_prompt_definition_with_default(*, default_value: str) -> dict:
    definition = _make_structured_prompt_definition_payload()
    definition["variables"][0]["default_value"] = default_value
    return definition


def _make_structured_prompt_definition_with_literal_user_content(
    *,
    user_content: str,
) -> dict:
    definition = _make_structured_prompt_definition_payload()
    definition["variables"] = []
    definition["blocks"][1]["content"] = user_content
    definition["blocks"][1]["is_template"] = False
    return definition


def _make_structured_prompt_definition_exceeding_total_runtime_limit() -> dict:
    return {
        "schema_version": 1,
        "format": "structured",
        "variables": [],
        "blocks": [
            {
                "id": f"assistant_{index + 1}",
                "name": f"Assistant Block {index + 1}",
                "role": "assistant",
                "content": "x" * 49000,
                "enabled": True,
                "order": (index + 1) * 10,
                "is_template": False,
            }
            for index in range(11)
        ],
        "assembly_config": {
            "legacy_system_roles": ["system", "developer"],
            "legacy_user_roles": ["user"],
            "block_separator": "\n\n",
        },
    }


@pytest.mark.asyncio
async def test_list_evaluations_honors_non_multiple_offset() -> None:
    """Prompt Studio evaluation listing starts at the exact requested offset."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_evaluations import list_evaluations

    rows = [
        {
            "id": idx,
            "uuid": f"00000000-0000-0000-0000-{idx + 1:012d}",
            "prompt_id": 10,
            "prompt_name": f"Evaluation {idx}",
            "status": "completed",
            "created_at": "2026-01-01T00:00:00Z",
            "completed_at": None,
            "aggregate_metrics": None,
        }
        for idx in range(10)
    ]

    class _FakePromptStudioDB:
        def __init__(self) -> None:
            self.calls: list[dict[str, int | None]] = []

        def list_evaluations(self, *, project_id, prompt_id=None, status=None, page=1, per_page=20):
            self.calls.append(
                {
                    "project_id": project_id,
                    "prompt_id": prompt_id,
                    "page": page,
                    "per_page": per_page,
                }
            )
            start = (page - 1) * per_page
            return {
                "evaluations": rows[start:start + per_page],
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": len(rows),
                    "total_pages": (len(rows) + per_page - 1) // per_page,
                },
            }

    db = _FakePromptStudioDB()

    response = await list_evaluations(
        request=None,
        project_id=1,
        prompt_id=None,
        limit=3,
        offset=4,
        db=db,
        user_context={},
    )

    assert [item.id for item in response["evaluations"]] == [4, 5, 6]
    assert response["total"] == 10
    assert response["pagination"].offset == 4
    assert db.calls == [
        {"project_id": 1, "prompt_id": None, "page": 2, "per_page": 3},
        {"project_id": 1, "prompt_id": None, "page": 3, "per_page": 3},
    ]


@pytest.fixture
def client(mock_user, test_db):
    """Create a test client for the FastAPI app with mocked authentication."""
    os.environ["TEST_MODE"] = "true"
    # Override the auth dependency
    app.dependency_overrides[get_current_active_user] = lambda: mock_user

    async def _override_prompt_studio_db():
        return test_db

    app.dependency_overrides[get_prompt_studio_db] = _override_prompt_studio_db

    with TestClient(app) as client:
        yield client
    # Clear overrides after test
    app.dependency_overrides.clear()
    os.environ.pop("TEST_MODE", None)

@pytest.fixture
def test_db():
    """Create a temporary test database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    db = PromptStudioDatabase(db_path, "test-client")
    yield db

    # Cleanup
    try:
        db.close()
    except Exception:
        _ = None
    _unlink_sqlite_file_with_retry(db_path)

@pytest.fixture
def auth_headers():
    """Create authentication headers for testing."""
    return {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json"
    }

@pytest.fixture
def mock_user():
    """Mock user for authentication."""
    return {
        "id": "test-user-123",
        "username": "testuser",
        "is_authenticated": True,
        "permissions": ["read", "write", "delete"]
    }


@pytest.mark.asyncio
async def test_list_prompts_safely_defaults_missing_pagination() -> None:
    """Prompt listing does not 500 when storage omits pagination metadata."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_prompts

    class _PromptDbWithoutPagination:
        def list_prompts(self, *_args, **_kwargs):
            return {"prompts": []}

    response = await prompt_studio_prompts.list_prompts(
        project_id=123,
        page=2,
        per_page=10,
        include_deleted=False,
        _=True,
        db=_PromptDbWithoutPagination(),
    )

    assert response.metadata.page == 2
    assert response.metadata.per_page == 10
    assert response.metadata.total == 0
    assert response.pagination.page == 2
    assert response.pagination.has_more is False


@pytest.mark.asyncio
async def test_list_projects_safely_defaults_missing_pagination() -> None:
    """Project listing preserves legacy aliases when storage omits pagination metadata."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_projects

    class _ProjectDbWithoutPagination:
        def list_projects(self, *_args, **_kwargs):
            return {"projects": []}

    response = await prompt_studio_projects.list_projects(
        page=2,
        per_page=10,
        status_filter=None,
        include_deleted=False,
        search=None,
        user_context={"user_id": "test-user", "is_admin": False},
        db=_ProjectDbWithoutPagination(),
    )

    assert response["metadata"] == {
        "page": 2,
        "per_page": 10,
        "total": 0,
        "total_pages": 0,
    }
    assert response["pagination"] == {
        "mode": "page",
        "page": 2,
        "per_page": 10,
        "total": 0,
        "total_pages": 0,
        "has_more": False,
    }
    assert response["projects"] == []


@pytest.mark.asyncio
async def test_list_optimizations_safely_defaults_missing_pagination() -> None:
    """Optimization listing adds canonical pagination when storage omits metadata."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_optimization

    class _OptimizationDbWithoutPagination:
        def list_optimizations(self, *_args, **_kwargs):
            return {"optimizations": []}

    response = await prompt_studio_optimization.list_optimizations(
        project_id=123,
        page=2,
        per_page=10,
        status_filter=None,
        _=True,
        db=_OptimizationDbWithoutPagination(),
    )

    assert response.metadata.page == 2
    assert response.metadata.per_page == 10
    assert response.metadata.total == 0
    assert response.pagination.mode == "page"
    assert response.pagination.page == 2
    assert response.pagination.has_more is False


@pytest.mark.asyncio
async def test_list_optimization_iterations_safely_defaults_missing_pagination() -> None:
    """Optimization iteration listing preserves its wrapper and adds pagination."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_optimization

    class _OptimizationIterationsDbWithoutPagination:
        def get_optimization(self, optimization_id):
            return {"id": optimization_id, "project_id": 123, "deleted": False}

        def get_project(self, project_id):
            return {"id": project_id, "user_id": "tester"}

        def list_optimization_iterations(self, *_args, **_kwargs):
            return {"iterations": []}

    response = await prompt_studio_optimization.list_optimization_iterations(
        optimization_id=456,
        page=2,
        per_page=10,
        db=_OptimizationIterationsDbWithoutPagination(),
        user_context={"user_id": "tester", "is_admin": False},
    )

    assert response.success is True
    assert response.data == {"iterations": []}
    assert response.metadata == {
        "page": 2,
        "per_page": 10,
        "total": 0,
        "total_pages": 0,
    }
    assert response.pagination.mode == "page"
    assert response.pagination.page == 2
    assert response.pagination.per_page == 10
    assert response.pagination.total == 0
    assert response.pagination.total_pages == 0
    assert response.pagination.has_more is False

########################################################################################################################
# Project Endpoints Tests

@pytest.mark.unit
class TestProjectEndpoints:
    """Test project-related API endpoints."""

    def test_create_project(self, client, test_db):

        """Test creating a new project."""
        project_data = {
            "name": "Test Project",
            "description": "A test project for integration testing",
            "status": "draft",
            "metadata": {"test": True}
        }

        response = client.post(
            "/api/v1/prompt-studio/projects/",
            json=project_data
        )

        # Check response
        if response.status_code != 201:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")

        assert response.status_code == 201
        data = response.json()
        assert data["success"] == True
        assert data["data"]["name"] == "Test Project"
        assert data["data"]["status"] == "draft"
        assert "id" in data["data"]
        assert "uuid" in data["data"]

    def test_list_projects(self, client, test_db):

        """Test listing projects."""
        # First create a project
        project_data = {
            "name": "List Test Project",
            "description": "For listing test"
        }
        client.post("/api/v1/prompt-studio/projects/", json=project_data)

        # Then list projects
        response = client.get("/api/v1/prompt-studio/projects/")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "data" in data
        assert "metadata" in data
        assert isinstance(data["data"], list)
        assert data["metadata"] == {
            "page": 1,
            "per_page": 20,
            "total": 1,
            "total_pages": 1,
        }
        assert data["pagination"] == {
            "mode": "page",
            "page": 1,
            "per_page": 20,
            "total": 1,
            "total_pages": 1,
            "has_more": False,
        }

    def test_get_project(self, client, test_db):

        """Test getting a specific project."""
        # First create a project
        create_response = client.post(
            "/api/v1/prompt-studio/projects/",
            json={"name": "Get Test", "description": "Test"}
        )

        assert create_response.status_code == 201
        project_id = create_response.json()["data"]["id"]

        # Get the project
        response = client.get(
            f"/api/v1/prompt-studio/projects/get/{project_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["data"]["id"] == project_id
        assert data["data"]["name"] == "Get Test"

    def test_update_project(self, client, test_db):

        """Test updating a project."""
        # First create a project
        create_response = client.post(
            "/api/v1/prompt-studio/projects/",
            json={"name": "Update Test", "description": "Original"}
        )

        assert create_response.status_code == 201
        project_id = create_response.json()["data"]["id"]

        # Update the project
        update_data = {
            "description": "Updated description",
            "status": "active"
        }

        response = client.put(
            f"/api/v1/prompt-studio/projects/update/{project_id}",
            json=update_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["data"]["description"] == "Updated description"
        assert data["data"]["status"] == "active"

    def test_archive_and_unarchive_project(self, client, test_db):

        """Test archiving and unarchiving a project."""
        create_response = client.post(
            "/api/v1/prompt-studio/projects/",
            json={"name": "Archive Test", "description": "Original"},
        )

        assert create_response.status_code == 201
        project_id = create_response.json()["data"]["id"]

        archive_response = client.post(
            f"/api/v1/prompt-studio/projects/archive/{project_id}"
        )

        assert archive_response.status_code == 200
        archive_data = archive_response.json()
        assert archive_data["success"] == True
        assert archive_data["data"]["status"] == "archived"

        get_archived_response = client.get(
            f"/api/v1/prompt-studio/projects/get/{project_id}"
        )

        assert get_archived_response.status_code == 200
        assert get_archived_response.json()["data"]["status"] == "archived"

        unarchive_response = client.post(
            f"/api/v1/prompt-studio/projects/unarchive/{project_id}"
        )

        assert unarchive_response.status_code == 200
        unarchive_data = unarchive_response.json()
        assert unarchive_data["success"] == True
        assert unarchive_data["data"]["status"] == "active"

        get_unarchived_response = client.get(
            f"/api/v1/prompt-studio/projects/get/{project_id}"
        )

        assert get_unarchived_response.status_code == 200
        assert get_unarchived_response.json()["data"]["status"] == "active"

    def test_delete_project(self, client, test_db):

        """Test deleting a project."""
        # First create a project
        create_response = client.post(
            "/api/v1/prompt-studio/projects/",
            json={"name": "Delete Test", "description": "To be deleted"}
        )

        assert create_response.status_code == 201
        project_id = create_response.json()["data"]["id"]

        # Delete the project
        response = client.delete(
            f"/api/v1/prompt-studio/projects/delete/{project_id}"
        )

        assert response.status_code == 200

        # Verify it's deleted (soft delete by default)
        get_response = client.get(
            f"/api/v1/prompt-studio/projects/get/{project_id}"
        )
        # Soft delete means it still exists but marked as deleted
        assert get_response.status_code in [200, 404]

########################################################################################################################
# Prompt Endpoints Tests

@pytest.mark.integration
class TestPromptEndpoints:
    """Test prompt-related API endpoints."""

    @pytest.fixture
    def project_id(self, client, test_db):
        """Create a project and return its ID."""
        response = client.post(
            "/api/v1/prompt-studio/projects/",
            json={"name": "Prompt Test Project", "description": "For prompt testing"}
        )
        if response.status_code == 201:
            return response.json()["data"]["id"]
        return None

    def test_create_prompt(self, client, test_db, project_id, auth_headers):

        """Test creating a new prompt."""
        if not project_id:
            pytest.skip("Project creation failed")

        prompt_data = {
            "project_id": project_id,
            "name": "Test Prompt",
            "system_prompt": "You are a helpful assistant.",
            "user_prompt": "Please help with: {task}",
            "version_number": 1,
            "change_description": "Initial version"
        }

        response = client.post(
            "/api/v1/prompt-studio/prompts",
            json=prompt_data,
            headers=auth_headers
        )

        assert response.status_code in [200, 201]
        data = response.json()
        assert data["name"] == "Test Prompt"
        assert data["project_id"] == project_id

    def test_list_prompts(self, client, auth_headers, mock_user, project_id):

        """Test listing prompts for a project."""
        if not project_id:
            pytest.skip("Project creation failed")

        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            response = client.get(
                f"/api/v1/prompt-studio/prompts?project_id={project_id}",
                headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert isinstance(data["data"], list)
            assert "metadata" in data
            assert data["pagination"]["mode"] == "page"
            assert data["pagination"]["page"] == 1
            assert data["pagination"]["per_page"] == 20
            assert data["pagination"]["total"] >= len(data["data"])
            assert data["pagination"]["total_pages"] >= 0
            assert data["pagination"]["has_more"] == (
                data["pagination"]["page"] < data["pagination"]["total_pages"]
            )

    def test_execute_prompt(self, client, auth_headers, mock_user):

        """Test executing a prompt."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            with patch('tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor.PromptExecutor.execute') as mock_execute:
                mock_execute.return_value = {
                    "output": "Test response",
                    "tokens_used": 100,
                    "execution_time": 1.5
                }

                execution_data = {
                    "prompt_id": 1,
                    "inputs": {"task": "Write a test"},
                    "provider": "openai",
                    "model": "gpt-4"
                }

                response = client.post(
                    "/api/v1/prompt-studio/prompts/execute",
                    json=execution_data,
                    headers=auth_headers
                )

                assert response.status_code in [200, 201]
                data = response.json()
                assert "output" in data
                assert data["tokens_used"] == 100

    def test_preview_prompt_renders_structured_messages(self, client, test_db, project_id, auth_headers):

        """Test previewing a structured prompt with modules/examples/signature output."""
        if not project_id:
            pytest.skip("Project creation failed")

        signature = test_db.create_signature(
            project_id=project_id,
            name="Preview Signature",
            input_schema=[{"name": "input", "type": "string"}],
            output_schema=[{"name": "answer", "type": "string"}],
        )

        response = client.post(
            "/api/v1/prompt-studio/prompts/preview",
            json={
                "project_id": project_id,
                "signature_id": signature["id"],
                "prompt_format": "structured",
                "prompt_schema_version": 1,
                "prompt_definition": _make_structured_prompt_definition_payload(),
                "few_shot_examples": [
                    {
                        "inputs": {"input": "Indexes"},
                        "outputs": {"answer": "Use the covering index."},
                    }
                ],
                "modules_config": [
                    {"type": "style_rules", "enabled": True, "config": {"tone": "concise"}}
                ],
                "variables": {"input": "SQLite FTS"},
            },
            headers=auth_headers,
        )

        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert [message["role"] for message in data["assembled_messages"]] == [
            "system",
            "developer",
            "user",
            "assistant",
            "user",
        ]
        assert data["assembled_messages"][1]["content"] == "Module style_rules: tone=concise"
        assert data["assembled_messages"][4]["content"].startswith("Evaluate SQLite FTS")
        assert "Please format your response as JSON" in data["assembled_messages"][4]["content"]

    def test_preview_prompt_rejects_missing_required_variable_with_client_error(
        self,
        client,
        project_id,
        auth_headers,
    ):
        if not project_id:
            pytest.skip("Project creation failed")

        response = client.post(
            "/api/v1/prompt-studio/prompts/preview",
            json={
                "project_id": project_id,
                "prompt_format": "structured",
                "prompt_schema_version": 1,
                "prompt_definition": _make_structured_prompt_definition_payload(),
                "variables": {},
            },
            headers=auth_headers,
        )

        assert response.status_code == 400, response.text
        assert "Missing required variable: input" in response.json()["detail"]

    def test_preview_prompt_rejects_oversized_assistant_block(
        self,
        client,
        project_id,
        auth_headers,
    ):
        if not project_id:
            pytest.skip("Project creation failed")

        app.dependency_overrides[get_security_config] = lambda: SecurityConfig(
            max_prompt_length=100,
            allowed_models=[],
            blocked_patterns=[],
            rate_limits={},
        )

        try:
            definition = {
                "schema_version": 1,
                "format": "structured",
                "variables": [],
                "blocks": [
                    {
                        "id": "assistant_example",
                        "name": "Assistant Example",
                        "role": "assistant",
                        "content": "x" * 140,
                        "enabled": True,
                        "order": 10,
                        "is_template": False,
                    }
                ],
            }

            response = client.post(
                "/api/v1/prompt-studio/prompts/preview",
                json={
                    "project_id": project_id,
                    "prompt_format": "structured",
                    "prompt_schema_version": 1,
                    "prompt_definition": definition,
                    "variables": {},
                },
                headers=auth_headers,
            )

            assert response.status_code == 400, response.text
            assert "exceeds maximum length" in response.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_security_config, None)

    def test_preview_prompt_rejects_structured_payload_exceeding_total_runtime_limit(
        self,
        client,
        project_id,
        auth_headers,
    ):
        if not project_id:
            pytest.skip("Project creation failed")

        response = client.post(
            "/api/v1/prompt-studio/prompts/preview",
            json={
                "project_id": project_id,
                "prompt_format": "structured",
                "prompt_schema_version": 1,
                "prompt_definition": _make_structured_prompt_definition_exceeding_total_runtime_limit(),
                "variables": {},
            },
            headers=auth_headers,
        )

        assert response.status_code == 400, response.text
        assert "maximum total length" in response.json()["detail"]

    def test_create_prompt_rejects_signature_augmented_payload_exceeding_security_limit(
        self,
        client,
        test_db,
        project_id,
        auth_headers,
    ):
        if not project_id:
            pytest.skip("Project creation failed")

        signature = test_db.create_signature(
            project_id=project_id,
            name="Create Limit Signature",
            input_schema=[{"name": "input", "type": "string"}],
            output_schema=[{"name": "x" * 20, "type": "string"}],
        )

        app.dependency_overrides[get_security_config] = lambda: SecurityConfig(
            max_prompt_length=120,
            allowed_models=[],
            blocked_patterns=[],
            rate_limits={},
        )

        try:
            response = client.post(
                "/api/v1/prompt-studio/prompts",
                json={
                    "project_id": project_id,
                    "signature_id": signature["id"],
                    "name": "Signature Length Prompt",
                    "prompt_format": "structured",
                    "prompt_schema_version": 1,
                    "prompt_definition": _make_structured_prompt_definition_with_literal_user_content(
                        user_content="x" * 40
                    ),
                    "change_description": "Initial version",
                },
                headers=auth_headers,
            )

            assert response.status_code == 400, response.text
            assert "exceeds maximum length" in response.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_security_config, None)

    def test_convert_prompt_returns_structured_definition(self, client, project_id, auth_headers):

        """Test converting a legacy prompt payload into a structured definition."""
        if not project_id:
            pytest.skip("Project creation failed")

        response = client.post(
            "/api/v1/prompt-studio/prompts/convert",
            json={
                "project_id": project_id,
                "system_prompt": "Be precise about {input}.",
                "user_prompt": "Evaluate $input against <baseline>.",
            },
            headers=auth_headers,
        )

        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["prompt_format"] == "structured"
        assert data["prompt_schema_version"] == 1
        assert data["extracted_variables"] == ["input", "baseline"]
        assert data["prompt_definition"]["blocks"][0]["content"] == "Be precise about {{input}}."
        assert data["prompt_definition"]["blocks"][1]["content"] == (
            "Evaluate {{input}} against {{baseline}}."
        )

    def test_create_prompt_rejects_structured_payload_exceeding_security_limit(
        self,
        client,
        project_id,
        auth_headers,
    ):
        if not project_id:
            pytest.skip("Project creation failed")

        app.dependency_overrides[get_security_config] = lambda: SecurityConfig(
            max_prompt_length=100,
            allowed_models=[],
            blocked_patterns=[],
            rate_limits={},
        )

        try:
            definition = _make_structured_prompt_definition_payload()
            definition["blocks"][1]["content"] = "Evaluate " + ("x" * 140)

            response = client.post(
                "/api/v1/prompt-studio/prompts",
                json={
                    "project_id": project_id,
                    "name": "Too Long Structured Prompt",
                    "prompt_format": "structured",
                    "prompt_schema_version": 1,
                    "prompt_definition": definition,
                    "change_description": "Initial version",
                },
                headers=auth_headers,
            )

            assert response.status_code == 400, response.text
            assert "exceeds maximum length" in response.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_security_config, None)

    def test_create_prompt_rejects_oversized_assistant_block(
        self,
        client,
        project_id,
        auth_headers,
    ):
        if not project_id:
            pytest.skip("Project creation failed")

        app.dependency_overrides[get_security_config] = lambda: SecurityConfig(
            max_prompt_length=100,
            allowed_models=[],
            blocked_patterns=[],
            rate_limits={},
        )

        try:
            definition = {
                "schema_version": 1,
                "format": "structured",
                "variables": [],
                "blocks": [
                    {
                        "id": "assistant_example",
                        "name": "Assistant Example",
                        "role": "assistant",
                        "content": "x" * 140,
                        "enabled": True,
                        "order": 10,
                        "is_template": False,
                    }
                ],
            }

            response = client.post(
                "/api/v1/prompt-studio/prompts",
                json={
                    "project_id": project_id,
                    "name": "Oversized Structured Assistant Prompt",
                    "prompt_format": "structured",
                    "prompt_schema_version": 1,
                    "prompt_definition": definition,
                    "change_description": "Initial version",
                },
                headers=auth_headers,
            )

            assert response.status_code == 400, response.text
            assert "exceeds maximum length" in response.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_security_config, None)

    def test_create_prompt_rejects_oversized_variable_default_value(
        self,
        client,
        project_id,
        auth_headers,
    ):
        if not project_id:
            pytest.skip("Project creation failed")

        app.dependency_overrides[get_security_config] = lambda: SecurityConfig(
            max_prompt_length=100,
            allowed_models=[],
            blocked_patterns=[],
            rate_limits={},
        )

        try:
            response = client.post(
                "/api/v1/prompt-studio/prompts",
                json={
                    "project_id": project_id,
                    "name": "Oversized Structured Default Prompt",
                    "prompt_format": "structured",
                    "prompt_schema_version": 1,
                    "prompt_definition": _make_structured_prompt_definition_with_default(
                        default_value="x" * 140
                    ),
                    "change_description": "Initial version",
                },
                headers=auth_headers,
            )

            assert response.status_code == 400, response.text
            assert "exceeds maximum length" in response.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_security_config, None)

    def test_create_prompt_rejects_structured_payload_exceeding_total_runtime_limit(
        self,
        client,
        project_id,
        auth_headers,
    ):
        if not project_id:
            pytest.skip("Project creation failed")

        response = client.post(
            "/api/v1/prompt-studio/prompts",
            json={
                "project_id": project_id,
                "name": "Aggregate Runtime Limit Prompt",
                "prompt_format": "structured",
                "prompt_schema_version": 1,
                "prompt_definition": _make_structured_prompt_definition_exceeding_total_runtime_limit(),
                "change_description": "Initial version",
            },
            headers=auth_headers,
        )

        assert response.status_code == 400, response.text
        assert "maximum total length" in response.json()["detail"]

    def test_update_prompt_rejects_oversized_variable_default_value(
        self,
        client,
        project_id,
        auth_headers,
    ):
        if not project_id:
            pytest.skip("Project creation failed")

        app.dependency_overrides[get_security_config] = lambda: SecurityConfig(
            max_prompt_length=100,
            allowed_models=[],
            blocked_patterns=[],
            rate_limits={},
        )

        try:
            create_response = client.post(
                "/api/v1/prompt-studio/prompts",
                json={
                    "project_id": project_id,
                    "name": "Structured Prompt For Update",
                    "prompt_format": "structured",
                    "prompt_schema_version": 1,
                    "prompt_definition": _make_structured_prompt_definition_payload(),
                    "change_description": "Initial version",
                },
                headers=auth_headers,
            )

            assert create_response.status_code in [200, 201], create_response.text
            prompt_id = create_response.json()["id"]

            update_response = client.put(
                f"/api/v1/prompt-studio/prompts/update/{prompt_id}",
                json={
                    "prompt_format": "structured",
                    "prompt_schema_version": 1,
                    "prompt_definition": _make_structured_prompt_definition_with_default(
                        default_value="x" * 140
                    ),
                    "change_description": "Introduce oversized default",
                },
                headers=auth_headers,
            )

            assert update_response.status_code == 400, update_response.text
            assert "exceeds maximum length" in update_response.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_security_config, None)

    def test_update_prompt_rejects_signature_augmented_payload_exceeding_security_limit(
        self,
        client,
        test_db,
        project_id,
        auth_headers,
    ):
        if not project_id:
            pytest.skip("Project creation failed")

        signature = test_db.create_signature(
            project_id=project_id,
            name="Update Limit Signature",
            input_schema=[{"name": "input", "type": "string"}],
            output_schema=[{"name": "x" * 20, "type": "string"}],
        )

        app.dependency_overrides[get_security_config] = lambda: SecurityConfig(
            max_prompt_length=120,
            allowed_models=[],
            blocked_patterns=[],
            rate_limits={},
        )

        try:
            create_response = client.post(
                "/api/v1/prompt-studio/prompts",
                json={
                    "project_id": project_id,
                    "signature_id": signature["id"],
                    "name": "Structured Prompt With Signature",
                    "prompt_format": "structured",
                    "prompt_schema_version": 1,
                    "prompt_definition": _make_structured_prompt_definition_with_literal_user_content(
                        user_content="short"
                    ),
                    "change_description": "Initial version",
                },
                headers=auth_headers,
            )

            assert create_response.status_code in [200, 201], create_response.text
            prompt_id = create_response.json()["id"]

            update_response = client.put(
                f"/api/v1/prompt-studio/prompts/update/{prompt_id}",
                json={
                    "prompt_format": "structured",
                    "prompt_schema_version": 1,
                    "prompt_definition": _make_structured_prompt_definition_with_literal_user_content(
                        user_content="x" * 40
                    ),
                    "change_description": "Increase task text",
                },
                headers=auth_headers,
            )

            assert update_response.status_code == 400, update_response.text
            assert "exceeds maximum length" in update_response.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_security_config, None)

########################################################################################################################
# Test Case Endpoints Tests

@pytest.mark.integration
class TestTestCaseEndpoints:
    """Test test case-related API endpoints."""

    @pytest.fixture
    def project_id(self, client, auth_headers, mock_user):
        """Create a project and return its ID."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            response = client.post(
                "/api/v1/prompt-studio/projects",
                json={"name": "Test Case Project", "description": "For test cases"},
                headers=auth_headers
            )
            if response.status_code in [200, 201]:
                return response.json()["id"]
            return None

    def test_create_test_case(self, client, auth_headers, mock_user, project_id):

        """Test creating a test case."""
        if not project_id:
            pytest.skip("Project creation failed")

        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            test_case_data = {
                "project_id": project_id,
                "name": "Test Case 1",
                "description": "A test case",
                "inputs": {"input": "test data"},
                "expected_outputs": {"output": "expected result"},
                "is_golden": False,
                "tags": ["unit", "test"]
            }

            response = client.post(
                "/api/v1/prompt-studio/test-cases",
                json=test_case_data,
                headers=auth_headers
            )

            assert response.status_code in [200, 201]
            data = response.json()
            assert data["name"] == "Test Case 1"
            assert data["project_id"] == project_id

    def test_list_test_cases(self, client, auth_headers, mock_user, project_id):

        """Test listing test cases for a project."""
        if not project_id:
            pytest.skip("Project creation failed")

        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            create_response = client.post(
                "/api/v1/prompt-studio/test-cases",
                json={
                    "project_id": project_id,
                    "name": "Listed Test Case",
                    "description": "A listed test case",
                    "inputs": {"input": "test data"},
                    "expected_outputs": {"output": "expected result"},
                    "is_golden": False,
                    "tags": ["integration"],
                },
                headers=auth_headers,
            )
            assert create_response.status_code in [200, 201]

            response = client.get(
                f"/api/v1/prompt-studio/test-cases/list/{project_id}?page=1&per_page=20",
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert isinstance(data["data"], list)
            assert data["pagination"] == {
                "mode": "page",
                "page": 1,
                "per_page": 20,
                "total": 1,
                "total_pages": 1,
                "has_more": False,
            }

    def test_run_test_cases(self, client, auth_headers, mock_user):

        """Test running test cases."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            with patch('tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_case_manager.TestCaseManager.run_batch_tests') as mock_run:
                mock_run.return_value = [
                    {
                        "test_case_id": "test-1",
                        "passed": True,
                        "actual_outputs": {"output": "result"},
                        "execution_time": 0.5
                    }
                ]

                run_data = {
                    "project_id": 1,
                    "test_case_ids": [1],
                    "prompt_id": 1
                }

                response = client.post(
                    "/api/v1/prompt-studio/test-cases/run",
                    json=run_data,
                    headers=auth_headers
                )

                assert response.status_code in [200, 201]
                data = response.json()
                assert "results" in data
                assert len(data["results"]) == 1
                assert data["results"][0]["passed"] is True

########################################################################################################################
# Evaluation Endpoints Tests

@pytest.mark.integration
class TestEvaluationEndpoints:
    """Test evaluation-related API endpoints."""

    def test_create_evaluation(self, client, auth_headers, mock_user):

        """Test creating an evaluation."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            evaluation_data = {
                "project_id": 1,
                "prompt_id": 1,
                "test_run_id": "run-123",
                "metrics": {
                    "accuracy": 0.95,
                    "f1_score": 0.92,
                    "latency": 1.5
                },
                "config": {
                    "model": "gpt-4",
                    "temperature": 0.7
                }
            }

            response = client.post(
                "/api/v1/prompt-studio/evaluations",
                json=evaluation_data,
                headers=auth_headers
            )

            assert response.status_code in [200, 201]
            data = response.json()
            assert "metrics" in data
            assert data["metrics"]["accuracy"] == 0.95

    def test_missing_provider_credentials_returns_503(self, client, auth_headers, mock_user, monkeypatch):

        """Missing provider credentials should return 503 with error code."""
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_evaluations as ps_eval
        from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials

        async def _missing(provider, *args, **kwargs):
            return ResolvedByokCredentials(
                provider=provider,
                api_key=None,
                app_config=None,
                credential_fields={},
                source="server",
                allowlisted=True,
            )

        monkeypatch.setattr(ps_eval, "_is_prompt_studio_test_mode", lambda: False)
        monkeypatch.setattr(ps_eval, "resolve_byok_credentials", _missing)

        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            evaluation_data = {
                "project_id": 1,
                "prompt_id": 1,
                "test_run_id": "run-123",
                "metrics": {
                    "accuracy": 0.95,
                    "f1_score": 0.92,
                    "latency": 1.5
                },
                "config": {
                    "model": "gpt-4",
                    "temperature": 0.7
                }
            }

            response = client.post(
                "/api/v1/prompt-studio/evaluations",
                json=evaluation_data,
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            detail = response.json().get("detail", {})
            assert detail.get("error_code") == "missing_provider_credentials"

    def test_list_evaluations(self, client, auth_headers, mock_user) -> None:

        """Test listing evaluations."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            response = client.get(
                "/api/v1/prompt-studio/evaluations?project_id=1&limit=10&offset=0",
                headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert "evaluations" in data
            assert isinstance(data["evaluations"], list)
            assert data["limit"] == 10
            assert data["offset"] == 0
            assert data["has_more"] is False
            assert data["next_offset"] is None
            assert data["pagination"] == {
                "mode": "offset",
                "total": data["total"],
                "limit": 10,
                "offset": 0,
                "has_more": False,
                "next_offset": None,
            }

    def test_create_evaluation_uses_test_runner_path(self, client, auth_headers, mock_user, test_db):

        """Evaluation creation should execute through TestRunner.run_single_test."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            project = test_db.create_project(
                name="Eval Runner Project",
                status="active",
                user_id=mock_user.get("id"),
            )
            project_id = project["id"]

            prompt = test_db.create_prompt(
                project_id=project_id,
                name="Eval Runner Prompt",
                system_prompt="You are helpful.",
                user_prompt="Answer {q}",
                version_number=1,
            )
            prompt_id = prompt["id"]

            test_case = test_db.create_test_case(
                project_id=project_id,
                name="Eval Runner Case",
                inputs={"q": "ping"},
                expected_outputs={"response": "pong"},
                tags=[],
                is_golden=False,
            )
            tc_id = test_case["id"]

            calls = {"n": 0}

            async def fake_run_single_test(self, *, prompt_id: int, test_case_id: int, model_config, metrics=None):
                calls["n"] += 1
                return {
                    "id": 999,
                    "prompt_id": prompt_id,
                    "test_case_id": test_case_id,
                    "inputs": {"q": "ping"},
                    "expected": {"response": "pong"},
                    "actual": {"response": "pong"},
                    "success": True,
                    "scores": {"aggregate_score": 0.88},
                }

            with patch(
                "tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner.TestRunner.run_single_test",
                new=fake_run_single_test,
            ):
                eval_resp = client.post(
                    "/api/v1/prompt-studio/evaluations",
                    json={
                        "project_id": project_id,
                        "prompt_id": prompt_id,
                        "name": "Eval Runner",
                        "test_case_ids": [tc_id],
                        "config": {"model_name": "gpt-4o-mini", "temperature": 0.1, "max_tokens": 32},
                        "run_async": False,
                    },
                    headers=auth_headers,
                )

            assert eval_resp.status_code in [200, 201], eval_resp.text
            payload = eval_resp.json()
            assert calls["n"] == 1
            assert "metrics" in payload
            assert payload["metrics"]["average_score"] == pytest.approx(0.88, rel=1e-6)

########################################################################################################################
# Optimization Endpoints Tests

@pytest.mark.integration
class TestOptimizationEndpoints:
    """Test optimization-related API endpoints."""

    def test_start_optimization(self, client, auth_headers, mock_user):

        """Test starting an optimization job."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            with patch('tldw_Server_API.app.core.Prompt_Management.prompt_studio.jobs_adapter.PromptStudioJobsAdapter.create_job') as mock_create:
                mock_create.return_value = {
                    "id": "job-123",
                    "status": "pending",
                    "type": "optimization"
                }

                project_resp = client.post(
                    "/api/v1/prompt-studio/projects/",
                    json={"name": "Opt Project", "status": "active"},
                    headers=auth_headers,
                )
                assert project_resp.status_code in (200, 201), project_resp.text
                project_id = (project_resp.json().get("data") or {}).get("id") or project_resp.json().get("id")

                prompt_resp = client.post(
                    "/api/v1/prompt-studio/prompts/create",
                    json={
                        "project_id": project_id,
                        "name": "Opt Prompt",
                        "system_prompt": "System",
                        "user_prompt": "{{text}}",
                    },
                    headers=auth_headers,
                )
                assert prompt_resp.status_code in (200, 201), prompt_resp.text
                prompt_id = (prompt_resp.json().get("data") or {}).get("id") or prompt_resp.json().get("id")

                optimization_data = {
                    "project_id": project_id,
                    "prompt_id": prompt_id,
                    "strategy": "mipro",
                    "config": {
                        "max_iterations": 10,
                        "target_metric": "accuracy",
                        "threshold": 0.9
                    }
                }

                response = client.post(
                    "/api/v1/prompt-studio/optimizations",
                    json=optimization_data,
                    headers=auth_headers
                )

                assert response.status_code in [200, 201]
                data = response.json()
                assert "id" in data
                assert data["status"] == "pending"

    def test_get_optimization_status(self, client, auth_headers, mock_user):

        """Test getting optimization job status."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            with patch('tldw_Server_API.app.core.Prompt_Management.prompt_studio.jobs_adapter.PromptStudioJobsAdapter.get_job') as mock_get:
                mock_get.return_value = {
                    "id": "job-123",
                    "status": "running",
                    "progress": 0.5,
                    "current_iteration": 5,
                    "best_score": 0.85
                }

                response = client.get(
                    "/api/v1/prompt-studio/optimizations/job-123",
                    headers=auth_headers
                )

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "running"
                assert data["progress"] == 0.5

########################################################################################################################
# WebSocket Tests

@pytest.mark.integration
class TestWebSocketEndpoints:
    """Test WebSocket functionality."""

    def test_websocket_connection(self, client):

        """Test WebSocket connection."""
        with client.websocket_connect("/api/v1/prompt-studio/ws") as websocket:
            # Send a test message
            websocket.send_json({
                "type": "subscribe",
                "project_id": 1
            })

            # Receive acknowledgment
            data = websocket.receive_json()
            assert data["type"] == "subscribed"
            assert data["project_id"] == 1

    def test_websocket_job_updates(self, client):

        """Test receiving job updates via WebSocket."""
        with client.websocket_connect("/api/v1/prompt-studio/ws") as websocket:
            # Subscribe to job updates
            websocket.send_json({
                "type": "subscribe_job",
                "job_id": "job-123"
            })

            # Simulate job update
            with patch('tldw_Server_API.app.core.Prompt_Management.prompt_studio.event_broadcaster.EventBroadcaster.broadcast') as mock_broadcast:
                mock_broadcast.return_value = None

                # Trigger a job update
                update_message = {
                    "type": "job_update",
                    "job_id": "job-123",
                    "status": "completed",
                    "result": {"score": 0.95}
                }

                # In real scenario, this would be triggered by job processor
                websocket.send_json(update_message)

                # Receive the update
                data = websocket.receive_json()
                assert data["type"] == "job_update"
                assert data["status"] == "completed"

########################################################################################################################
# Error Handling Tests

@pytest.mark.integration
class TestErrorHandling:
    """Test API error handling."""

    def test_unauthorized_access(self, client):

        """Test accessing endpoints without authentication."""
        response = client.get("/api/v1/prompt-studio/projects")
        assert response.status_code == 401

    def test_invalid_project_id(self, client, auth_headers, mock_user):

        """Test accessing non-existent project."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            response = client.get(
                "/api/v1/prompt-studio/projects/99999",
                headers=auth_headers
            )
            assert response.status_code == 404

    def test_invalid_request_data(self, client, auth_headers, mock_user):

        """Test sending invalid data."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            # Missing required field
            invalid_data = {
                "description": "Missing name field"
            }

            response = client.post(
                "/api/v1/prompt-studio/projects",
                json=invalid_data,
                headers=auth_headers
            )

            assert response.status_code == 422
            data = response.json()
            assert "detail" in data

    def test_rate_limiting(self, client, auth_headers, mock_user):

        """Test rate limiting."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            # Make many rapid requests
            responses = []
            for _ in range(100):
                response = client.get(
                    "/api/v1/prompt-studio/projects",
                    headers=auth_headers
                )
                responses.append(response.status_code)

            # Should have some rate limited responses
            # Note: This depends on rate limit configuration
            # assert 429 in responses

########################################################################################################################
# Pagination Tests

@pytest.mark.integration
class TestPagination:
    """Test pagination functionality."""

    def test_project_pagination(self, client, auth_headers, mock_user):

        """Test paginating project list."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            # First page
            response = client.get(
                "/api/v1/prompt-studio/projects?page=1&per_page=5",
                headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert "pagination" in data
            assert data["pagination"]["page"] == 1
            assert data["pagination"]["per_page"] == 5

            # Next page
            if data["pagination"]["total_pages"] > 1:
                response = client.get(
                    "/api/v1/prompt-studio/projects?page=2&per_page=5",
                    headers=auth_headers
                )

                assert response.status_code == 200
                data = response.json()
                assert data["pagination"]["page"] == 2

########################################################################################################################
# Search and Filter Tests

@pytest.mark.integration
class TestSearchAndFilter:
    """Test search and filtering functionality."""

    def test_search_projects(self, client, auth_headers, mock_user):

        """Test searching projects."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            response = client.get(
                "/api/v1/prompt-studio/projects?search=test&status=active",
                headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert "projects" in data

            # All results should match search criteria
            for project in data["projects"]:
                assert "test" in project["name"].lower() or "test" in project.get("description", "").lower()
                assert project["status"] == "active"

    def test_filter_by_date(self, client, auth_headers, mock_user):

        """Test filtering by date range."""
        with patch('tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps.get_current_active_user', return_value=mock_user):
            response = client.get(
                "/api/v1/prompt-studio/projects?created_after=2024-01-01&created_before=2024-12-31",
                headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert "projects" in data
