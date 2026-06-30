# test_chunking_templates.py
"""
Comprehensive tests for chunking template functionality.
Tests database operations, API endpoints, and template application.
"""

import json
import os
import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Set test API key for single-user mode
os.environ["SINGLE_USER_API_KEY"] = "test-api-key-that-is-long-enough"

from fastapi.testclient import TestClient
from fastapi import FastAPI

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.Chunking.templates import TemplateProcessor, ChunkingTemplate, TemplateStage
from tldw_Server_API.app.core.Chunking import template_initialization as template_init
from tldw_Server_API.app.core.Chunking.template_initialization import (
    load_builtin_templates,
    initialize_chunking_templates,
    ensure_templates_initialized,
)
from tldw_Server_API.app.api.v1.endpoints.chunking_templates import router as templates_router
from tldw_Server_API.app.api.v1.schemas.chunking_templates_schemas import (
    ChunkingTemplateCreate,
    ChunkingTemplateResponse,
    TemplateConfig,
)


# Fixtures
@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing with proper cleanup."""
    db_path = str(tmp_path / "chunking_templates.db")

    # Initialize database with schema (including ChunkingTemplates table)
    db = MediaDatabase(db_path=db_path, client_id="test_client")

    yield db, db_path

    # Cleanup
    try:
        db.close_connection()
    except:
        _ = None

    # Delete the database file
    try:
        if os.path.exists(db_path):
            os.unlink(db_path)
    except:
        _ = None


@pytest.fixture
def test_client(temp_db):
    """Create a test client for API testing with proper database override."""
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user

    db, db_path = temp_db

    # Override the database dependency
    def override_get_db():
        return db

    app.dependency_overrides[get_media_db_for_user] = override_get_db

    with TestClient(app) as client:
        yield client

    # Clear the override after the test
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    """Authentication headers for API requests."""
    api_key = get_settings().SINGLE_USER_API_KEY or os.getenv("SINGLE_USER_API_KEY", "test-api-key-that-is-long-enough")
    return {"X-API-KEY": api_key, "Content-Type": "application/json"}


@pytest.fixture
def sample_template():
    """Create a sample template for testing."""
    return {
        "name": "test_template",
        "description": "Test template for unit tests",
        "tags": ["test", "sample"],
        "preprocessing": [{"operation": "normalize_whitespace", "config": {"max_line_breaks": 2}}],
        "chunking": {"method": "sentences", "config": {"max_size": 5, "overlap": 1}},
        "postprocessing": [{"operation": "filter_empty", "config": {"min_length": 10}}],
    }


# Database Tests
class TestDatabaseOperations:
    """Test database CRUD operations for chunking templates."""

    def test_create_template(self, temp_db):
        """Test creating a new template."""
        db, _ = temp_db

        template = db.create_chunking_template(
            name="test_template",
            template_json='{"chunking": {"method": "words"}}',
            description="Test template",
            tags=["test"],
        )

        assert template is not None
        assert template["name"] == "test_template"
        assert template["description"] == "Test template"
        assert template["is_builtin"] is False
        assert "test" in template["tags"]

    def test_create_duplicate_template(self, temp_db):
        """Test that duplicate template names are rejected."""
        db, _ = temp_db

        # Create first template
        db.create_chunking_template(
            name="unique_template", template_json='{"chunking": {"method": "words"}}', description="First template"
        )

        # Try to create duplicate
        with pytest.raises(Exception) as exc_info:
            db.create_chunking_template(
                name="unique_template",
                template_json='{"chunking": {"method": "sentences"}}',
                description="Duplicate template",
            )

        assert "already exists" in str(exc_info.value)

    def test_get_template(self, temp_db):
        """Test retrieving a template."""
        db, _ = temp_db

        # Create template
        created = db.create_chunking_template(
            name="get_test", template_json='{"chunking": {"method": "words"}}', description="Get test"
        )

        # Get by name
        template = db.get_chunking_template(name="get_test")
        assert template is not None
        assert template["name"] == "get_test"
        assert template["description"] == "Get test"

        # Get non-existent
        template = db.get_chunking_template(name="non_existent")
        assert template is None

    def test_list_templates(self, temp_db):
        """Test listing templates."""
        db, _ = temp_db

        # Create multiple templates
        for i in range(3):
            db.create_chunking_template(
                name=f"list_test_{i}",
                template_json='{"chunking": {"method": "words"}}',
                description=f"List test {i}",
                tags=[f"tag_{i}", "common"],
            )

        # List all
        templates = db.list_chunking_templates()
        assert len(templates) >= 3

        # List with filters
        templates = db.list_chunking_templates(tags=["common"])
        assert len(templates) >= 3
        assert all("common" in t["tags"] for t in templates)

        templates = db.list_chunking_templates(tags=["tag_1"])
        assert any(t["name"] == "list_test_1" for t in templates)

    def test_update_template(self, temp_db):
        """Test updating a template."""
        db, _ = temp_db

        # Create template
        created = db.create_chunking_template(
            name="update_test", template_json='{"chunking": {"method": "words"}}', description="Original description"
        )

        # Update it
        result = db.update_chunking_template(
            name="update_test", description="Updated description", tags=["updated", "test"]
        )

        assert result is True

        # Get the updated template to verify changes
        updated = db.get_chunking_template(name="update_test")
        assert updated["description"] == "Updated description"
        assert "updated" in updated["tags"]
        assert "test" in updated["tags"]
        assert updated["version"] == created["version"] + 1

    def test_cannot_update_builtin(self, temp_db):
        """Test that built-in templates cannot be updated."""
        db, _ = temp_db

        # Create a builtin template
        db.create_chunking_template(
            name="builtin_test",
            template_json='{"chunking": {"method": "words"}}',
            description="Built-in template",
            is_builtin=True,
        )

        # Try to update it
        with pytest.raises(Exception) as exc_info:
            db.update_chunking_template(name="builtin_test", description="Modified builtin")

        assert "built-in" in str(exc_info.value).lower()

    def test_delete_template(self, temp_db):
        """Test deleting a template (soft delete)."""
        db, _ = temp_db

        # Create template
        db.create_chunking_template(
            name="delete_test", template_json='{"chunking": {"method": "words"}}', description="To be deleted"
        )

        # Delete it
        result = db.delete_chunking_template(name="delete_test")
        assert result is True

        # Verify it's gone
        template = db.get_chunking_template(name="delete_test")
        assert template is None

        # Try to delete non-existent
        result = db.delete_chunking_template(name="non_existent")
        assert result is False

    def test_cannot_delete_builtin(self, temp_db):
        """Test that built-in templates cannot be deleted."""
        db, _ = temp_db

        # Create a builtin template
        db.create_chunking_template(
            name="builtin_delete_test",
            template_json='{"chunking": {"method": "words"}}',
            description="Built-in template",
            is_builtin=True,
        )

        # Try to delete it
        with pytest.raises(Exception) as exc_info:
            db.delete_chunking_template(name="builtin_delete_test")

        assert "built-in" in str(exc_info.value).lower()


# Template Initialization Tests
class TestTemplateInitialization:
    """Test template loading and initialization."""

    def test_load_builtin_templates(self):
        """Test loading built-in templates from files."""
        templates = load_builtin_templates()

        assert isinstance(templates, list)
        # Should have loaded the templates we created
        template_names = {t["name"] for t in templates}
        expected_names = {
            "academic_paper",
            "code_documentation",
            "chat_conversation",
            "book_chapters",
            "transcript_dialogue",
            "legal_document",
        }
        assert expected_names.issubset(template_names)

    def test_seed_builtin_templates(self, temp_db):
        """Test seeding built-in templates into database."""
        db, _ = temp_db

        # Initialize templates
        initialize_chunking_templates(db)

        # Check that built-in templates exist
        templates = db.list_chunking_templates(include_builtin=True, include_custom=False)
        assert len(templates) > 0

        # Verify academic_paper template
        academic = db.get_chunking_template(name="academic_paper")
        assert academic is not None
        assert academic["is_builtin"] is True
        assert "research" in academic["tags"]

    def test_initialize_chunking_templates_uses_managed_media_database_for_owned_db(self, monkeypatch, tmp_path):
        events = []
        db_path = str(tmp_path / "chunking-templates.db")

        class FakeDb:
            def seed_builtin_templates(self, templates):
                events.append(("seed", templates))
                return len(templates)

        class FakeManagedDbContext:
            def __init__(self, **kwargs):
                events.append(("managed", kwargs))

            def __enter__(self):
                events.append("enter")
                return FakeDb()

            def __exit__(self, exc_type, exc, tb):
                events.append("exit")
                return False

        monkeypatch.setattr(
            template_init,
            "load_builtin_templates",
            lambda: [{"name": "built-in", "template": {"name": "built-in"}}],
        )
        monkeypatch.setattr(
            template_init,
            "managed_media_database",
            lambda client_id, **kwargs: FakeManagedDbContext(client_id=client_id, **kwargs),
        )

        count = template_init.initialize_chunking_templates(
            db_path=db_path,
            client_id="template-test",
        )

        assert count == 1
        assert events == [
            ("managed", {"client_id": "template-test", "db_path": db_path, "initialize": False}),
            "enter",
            ("seed", [{"name": "built-in", "template": {"name": "built-in"}}]),
            "exit",
        ]

    def test_update_builtin_templates_uses_managed_media_database_for_owned_db(self, monkeypatch, tmp_path):
        events = []
        db_path = str(tmp_path / "chunking-templates.db")

        class FakeDb:
            def get_chunking_template(self, name):
                events.append(("get", name))
                return {
                    "name": name,
                    "is_builtin": True,
                    "template_json": json.dumps({"name": name, "version": 1}),
                }

            def update_chunking_template(self, **kwargs):
                events.append(("update", kwargs["name"]))
                return True

        class FakeManagedDbContext:
            def __init__(self, **kwargs):
                events.append(("managed", kwargs))

            def __enter__(self):
                events.append("enter")
                return FakeDb()

            def __exit__(self, exc_type, exc, tb):
                events.append("exit")
                return False

        monkeypatch.setattr(
            template_init,
            "load_builtin_templates",
            lambda: [
                {
                    "name": "built-in",
                    "description": "desc",
                    "tags": ["x"],
                    "template": {"name": "built-in", "version": 2},
                }
            ],
        )
        monkeypatch.setattr(
            template_init,
            "managed_media_database",
            lambda client_id, **kwargs: FakeManagedDbContext(client_id=client_id, **kwargs),
        )

        count = template_init.update_builtin_templates(
            db_path=db_path,
            client_id="template-test",
        )

        assert count == 1
        assert events == [
            ("managed", {"client_id": "template-test", "db_path": db_path, "initialize": False}),
            "enter",
            ("get", "built-in"),
            ("update", "built-in"),
            "exit",
        ]

    def test_ensure_templates_initialized_uses_managed_media_database_for_existing_check(self, monkeypatch, tmp_path):
        events = []
        db_path = str(tmp_path / "chunking-templates.db")

        class FakeDb:
            def list_chunking_templates(self, **kwargs):
                events.append(("list", kwargs))
                return [{"name": "built-in"}]

        class FakeManagedDbContext:
            def __init__(self, **kwargs):
                events.append(("managed", kwargs))

            def __enter__(self):
                events.append("enter")
                return FakeDb()

            def __exit__(self, exc_type, exc, tb):
                events.append("exit")
                return False

        monkeypatch.setattr(template_init, "initialize_chunking_templates", lambda **_kwargs: 0)
        monkeypatch.setattr(
            template_init,
            "managed_media_database",
            lambda client_id, **kwargs: FakeManagedDbContext(client_id=client_id, **kwargs),
        )

        result = template_init.ensure_templates_initialized(db_path=db_path)

        assert result is True
        assert events == [
            ("managed", {"client_id": "system", "db_path": db_path, "initialize": False}),
            "enter",
            ("list", {"include_builtin": True, "include_custom": False}),
            "exit",
        ]


# API Endpoint Tests
class TestAPIEndpoints:
    """Test REST API endpoints for template management."""

    def test_list_templates_endpoint(self, test_client, auth_headers, temp_db):
        """Test GET /api/v1/chunking/templates endpoint."""
        db, _ = temp_db

        # Create test templates
        db.create_chunking_template(
            name="api_test1",
            template_json='{"chunking": {"method": "words"}}',
            description="API test 1",
            tags=["api", "test"],
        )

        # Test listing
        response = test_client.get("/api/v1/chunking/templates", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "templates" in data
        assert "total" in data
        assert data["total"] >= 1

        # Test filtering
        response = test_client.get("/api/v1/chunking/templates?tags=api", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert all("api" in t["tags"] for t in data["templates"])

    def test_get_template_endpoint(self, test_client, auth_headers, temp_db):
        """Test GET /api/v1/chunking/templates/{name} endpoint."""
        db, _ = temp_db

        # Create template
        db.create_chunking_template(
            name="get_api_test", template_json='{"chunking": {"method": "words"}}', description="Get API test"
        )

        # Get existing template
        response = test_client.get("/api/v1/chunking/templates/get_api_test", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "get_api_test"
        assert data["description"] == "Get API test"

        # Get non-existent template
        response = test_client.get("/api/v1/chunking/templates/non_existent", headers=auth_headers)
        assert response.status_code == 404

    def test_create_template_endpoint(self, test_client, auth_headers, sample_template):
        """Test POST /api/v1/chunking/templates endpoint."""
        # Create template via API
        request_data = {
            "name": sample_template["name"],
            "description": sample_template["description"],
            "tags": sample_template["tags"],
            "template": {
                "preprocessing": sample_template["preprocessing"],
                "chunking": sample_template["chunking"],
                "postprocessing": sample_template["postprocessing"],
            },
        }

        response = test_client.post("/api/v1/chunking/templates", json=request_data, headers=auth_headers)
        assert response.status_code == 201

        data = response.json()
        assert data["name"] == sample_template["name"]
        assert data["description"] == sample_template["description"]
        assert data["is_builtin"] is False

        # Try to create duplicate
        response = test_client.post("/api/v1/chunking/templates", json=request_data, headers=auth_headers)
        assert response.status_code == 409

    def test_update_template_endpoint(self, test_client, auth_headers, temp_db):
        """Test PUT /api/v1/chunking/templates/{name} endpoint."""
        db, _ = temp_db

        # Create template first
        db.create_chunking_template(
            name="update_api_test", template_json='{"chunking": {"method": "words"}}', description="Original"
        )

        # Update via API
        update_data = {"description": "Updated via API", "tags": ["updated", "api"]}

        response = test_client.put("/api/v1/chunking/templates/update_api_test", json=update_data, headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["description"] == "Updated via API"
        assert data["tags"] == ["updated", "api"]

        # Update non-existent
        response = test_client.put("/api/v1/chunking/templates/non_existent", json=update_data, headers=auth_headers)
        assert response.status_code == 404

    def test_delete_template_endpoint(self, test_client, auth_headers, temp_db):
        """Test DELETE /api/v1/chunking/templates/{name} endpoint."""
        db, _ = temp_db

        # Create template first
        db.create_chunking_template(
            name="delete_api_test", template_json='{"chunking": {"method": "words"}}', description="To delete"
        )

        # Delete via API
        response = test_client.delete("/api/v1/chunking/templates/delete_api_test", headers=auth_headers)
        assert response.status_code == 204

        # Verify deleted
        response = test_client.get("/api/v1/chunking/templates/delete_api_test", headers=auth_headers)
        assert response.status_code == 404

        # Delete non-existent
        response = test_client.delete("/api/v1/chunking/templates/non_existent", headers=auth_headers)
        assert response.status_code == 404

    def test_validate_template_endpoint(self, test_client, auth_headers):
        """Test POST /api/v1/chunking/templates/validate endpoint."""
        # Valid template
        valid_template = {
            "preprocessing": [{"operation": "normalize_whitespace", "config": {}}],
            "chunking": {"method": "sentences", "config": {"max_size": 5}},
            "postprocessing": [],
        }

        response = test_client.post("/api/v1/chunking/templates/validate", json=valid_template, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["errors"] is None

        # Invalid template (missing chunking)
        invalid_template = {"preprocessing": [{"operation": "normalize_whitespace"}]}

        response = test_client.post("/api/v1/chunking/templates/validate", json=invalid_template, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert data["errors"] is not None

    def test_validate_template_with_hierarchical_boundaries_valid(self, test_client, auth_headers):
        """Validate accepts hierarchical_template with a small set of safe boundaries."""
        payload = {
            "chunking": {
                "method": "sentences",
                "config": {
                    "max_size": 5,
                    "hierarchical": True,
                    "hierarchical_template": {
                        "boundaries": [
                            {"kind": "header_atx", "pattern": r"^\s*#{1,6}\s+.+$", "flags": "m"},
                            {"kind": "abstract", "pattern": r"^\s*Abstract\b", "flags": "im"},
                        ]
                    },
                },
            }
        }
        resp = test_client.post("/api/v1/chunking/templates/validate", json=payload, headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is True
        assert body.get("errors") in (None, [])

    def test_validate_template_with_hierarchical_boundaries_limits(self, test_client, auth_headers):
        """Validation rejects too many boundaries and overlong patterns/flags."""
        # >20 boundaries should trigger an error
        many_boundaries = [{"kind": f"k{i}", "pattern": r"^X$", "flags": "m"} for i in range(25)]
        payload_too_many = {
            "chunking": {
                "method": "sentences",
                "config": {"hierarchical": True, "hierarchical_template": {"boundaries": many_boundaries}},
            }
        }
        resp = test_client.post("/api/v1/chunking/templates/validate", json=payload_too_many, headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is False
        assert any("boundaries" in (err.get("field") or "") for err in body.get("errors", []))

        # Overlong pattern
        long_pattern = "^" + ("a" * 1200) + "$"
        payload_long_pattern = {
            "chunking": {
                "method": "sentences",
                "config": {
                    "hierarchical": True,
                    "hierarchical_template": {"boundaries": [{"kind": "x", "pattern": long_pattern, "flags": "m"}]},
                },
            }
        }
        resp2 = test_client.post("/api/v1/chunking/templates/validate", json=payload_long_pattern, headers=auth_headers)
        assert resp2.status_code == 200
        body2 = resp2.json()
        assert body2["valid"] is False
        assert any("pattern" in (err.get("field") or "") for err in body2.get("errors", []))

        # Overlong flags
        payload_long_flags = {
            "chunking": {
                "method": "sentences",
                "config": {
                    "hierarchical": True,
                    "hierarchical_template": {"boundaries": [{"kind": "x", "pattern": r"^.$", "flags": "i" * 50}]},
                },
            }
        }
        resp3 = test_client.post("/api/v1/chunking/templates/validate", json=payload_long_flags, headers=auth_headers)
        assert resp3.status_code == 200
        body3 = resp3.json()
        assert body3["valid"] is False
        assert any("flags" in (err.get("field") or "") for err in body3.get("errors", []))


# Template Processing Tests
class TestTemplateProcessing:
    """Test template processing functionality."""

    def test_template_processor_operations(self):
        """Test individual operations in TemplateProcessor."""
        processor = TemplateProcessor()

        # Test normalize_whitespace
        text = "Test    text\n\n\n\nwith    spaces"
        result = processor._normalize_whitespace(text, {"max_line_breaks": 2})
        assert "\n\n\n\n" not in result
        assert "    " not in result

        # Test filter_empty
        chunks = ["", "valid chunk", "   ", "another valid"]
        result = processor._filter_empty(chunks, {"min_length": 5})
        assert "" not in result
        assert "valid chunk" in result

        # Test merge_small
        chunks = ["a", "b", "longer chunk", "c", "another long chunk"]
        result = processor._merge_small(chunks, {"min_size": 10, "separator": " "})
        assert all(len(chunk) >= 10 or chunk == result[-1] for chunk in result)

    def test_process_template(self):
        """Test processing text through a template."""
        processor = TemplateProcessor()

        # Create a simple template
        template = ChunkingTemplate(
            name="test_processing",
            description="Test template",
            base_method="sentences",
            stages=[
                TemplateStage("preprocess", [{"type": "normalize_whitespace", "params": {"max_line_breaks": 1}}]),
                TemplateStage("chunk", [{"method": "sentences", "max_size": 2}]),
                TemplateStage("postprocess", [{"type": "filter_empty", "params": {"min_length": 5}}]),
            ],
            default_options={"max_size": 2},
        )

        text = "This is a test.    Another sentence.\n\n\nShort.\nOne more test sentence."

        # Mock the chunker since it's not available in test
        with patch("tldw_Server_API.app.core.Chunking.templates.Chunker") as mock_chunker:
            mock_instance = MagicMock()
            mock_instance.chunk_text.return_value = ["This is a test. Another sentence.", "One more test sentence."]
            mock_chunker.return_value = mock_instance

            result = processor.process_template(text, template)

            assert isinstance(result, list)
            assert len(result) == 2  # After filtering out "Short."

    def test_process_template_db_style_schema(self):
        """TemplateProcessor should accept DB-style operation/config schema via stages."""
        processor = TemplateProcessor()

        # Build stages using DB-style operation/config entries
        stages = [
            TemplateStage(
                name="preprocess",
                operations=[{"operation": "normalize_whitespace", "config": {"max_line_breaks": 1}}],
                enabled=True,
            ),
            TemplateStage(
                name="chunk",
                operations=[{"method": "words", "config": {"max_size": 3, "overlap": 0}}],
                enabled=True,
            ),
            TemplateStage(
                name="postprocess",
                operations=[{"operation": "filter_empty", "config": {"min_length": 1}}],
                enabled=True,
            ),
        ]

        tmpl = ChunkingTemplate(
            name="db_style", description="DB style schema", base_method="words", stages=stages, default_options={}
        )

        text = "One   two three\n\n\nFour five six seven"
        result = processor.process_template(text, tmpl)
        assert isinstance(result, list)
        assert len(result) >= 2


# Integration Tests
class TestIntegration:
    """Test integration between components."""

    def test_apply_template_endpoint(self, test_client, auth_headers, temp_db):
        """Test applying a template to text via API."""
        db, _ = temp_db

        # Create a template
        db.create_chunking_template(
            name="apply_test",
            template_json=json.dumps(
                {
                    "preprocessing": [],
                    "chunking": {"method": "words", "config": {"max_size": 10, "overlap": 2}},
                    "postprocessing": [],
                }
            ),
            description="Template for apply test",
        )

        # Apply it to text
        request_data = {
            "text": "This is a long text that should be chunked according to the template configuration.",
            "template_name": "apply_test",
            "options": {},
        }

        # Mock the TemplateProcessor to avoid complex setup
        with patch("tldw_Server_API.app.api.v1.endpoints.chunking_templates.TemplateProcessor") as mock_processor:
            mock_instance = MagicMock()
            mock_instance.process_template.return_value = [
                "This is a long text",
                "that should be chunked",
                "according to the template",
                "configuration.",
            ]
            mock_processor.return_value = mock_instance

            response = test_client.post("/api/v1/chunking/templates/apply", json=request_data, headers=auth_headers)
            assert response.status_code == 200

            data = response.json()
            assert data["template_name"] == "apply_test"
            assert "chunks" in data
            assert len(data["chunks"]) == 4
            assert "metadata" in data

    def test_apply_template_endpoint_real_processor(self, test_client, auth_headers, temp_db):
        """Apply template without mocking, ensure end-to-end DB schema mapping works."""
        db, _ = temp_db

        # Create a simple words-based template that should work with real processor
        db.create_chunking_template(
            name="apply_real",
            template_json=json.dumps(
                {
                    "preprocessing": [{"operation": "normalize_whitespace", "config": {"max_line_breaks": 1}}],
                    "chunking": {"method": "words", "config": {"max_size": 4, "overlap": 0}},
                    "postprocessing": [{"operation": "filter_empty", "config": {"min_length": 1}}],
                }
            ),
            description="Real processor apply test",
        )

        req = {
            "text": "Alpha   beta gamma\n\nDelta epsilon zeta eta theta",
            "template_name": "apply_real",
        }

        resp = test_client.post("/api/v1/chunking/templates/apply", json=req, headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["template_name"] == "apply_real"
        assert isinstance(body["chunks"], list)
        assert len(body["chunks"]) >= 2

    def test_template_in_chunking_endpoint(self):
        """Test using templates in the main chunking endpoint."""
        from tldw_Server_API.app.api.v1.endpoints.chunking import process_text_for_chunking_json
        from tldw_Server_API.app.api.v1.schemas.chunking_schema import ChunkingTextRequest, ChunkingOptionsRequest

        # This would require more complex mocking of the chunking endpoint
        # For now, we just verify the schema accepts template_name
        options = ChunkingOptionsRequest(
            template_name="test_template",
            method="words",  # This should be overridden by template
            max_size=100,
            overlap=50,  # Explicitly set overlap < max_size to avoid validation error
        )

        assert options.template_name == "test_template"
        assert hasattr(options, "template_name")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
