"""
Integration tests for Chatbook service using real test database.
No mocking - uses actual components.
"""
import pytest
import json
import zipfile
import os
import shutil
from pathlib import Path
from datetime import datetime
import asyncio
from uuid import uuid4

import yaml

from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookVersion,
    ContentItem,
    ContentType,
    ExportStatus,
    ImportStatus
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def test_static_openapi_import_contract_uses_direct_form_fields():
    openapi_path = Path(__file__).resolve().parents[3] / "Docs" / "API-related" / "chatbook_openapi.yaml"
    spec = yaml.safe_load(openapi_path.read_text(encoding="utf-8"))

    import_schema = spec["paths"]["/chatbooks/import"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    import_properties = import_schema["properties"]
    create_response = spec["components"]["schemas"]["CreateChatbookResponse"]["properties"]

    assert "file" in import_properties
    assert "options" not in import_properties
    assert "conflict_resolution" in import_properties
    assert "/chatbooks/export/continue" in spec["paths"]
    assert "file_path" not in create_response


def manifest_to_dict(manifest):
    """Helper to convert manifest to dict handling enums."""
    data = manifest.to_dict()
    if hasattr(manifest.version, 'value'):
        data['version'] = manifest.version.value
    return data


def stage_export_for_import(service: ChatbookService, export_path: str) -> str:
    """Copy an export into the chatbooks temp directory for safe import."""
    src = Path(export_path)
    staged = service.temp_dir / f"import_{uuid4().hex}_{src.name}"
    shutil.copy2(src, staged)
    return str(staged)


@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary database path."""
    db_path = tmp_path / "test_chatbook.db"
    return str(db_path)


@pytest.fixture
def test_db(test_db_path):
    """Create a real test database."""
    db = CharactersRAGDB(db_path=test_db_path, client_id="test_client")

    yield db

    # Cleanup - don't call close() as it might not exist
    if os.path.exists(test_db_path):
        os.unlink(test_db_path)


@pytest.fixture
def service(test_db, tmp_path):
    """Create ChatbookService with real database."""
    # Set up test environment
    os.environ['PYTEST_CURRENT_TEST'] = 'test'
    os.environ['USER_DB_BASE_DIR'] = str(tmp_path)

    service = ChatbookService(user_id="test_user", db=test_db)

    yield service

    # Cleanup
    if hasattr(service, 'temp_dir') and service.temp_dir.exists():
        shutil.rmtree(service.temp_dir, ignore_errors=True)


@pytest.fixture
def sample_chatbook_file(service):
    """Create a sample chatbook file for import tests."""
    chatbook_path = service.temp_dir / f"sample_{uuid4().hex}.chatbook"

    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Sample Chatbook",
        description="Test chatbook for integration tests",
        author="Test Author",
        user_id="test_user"
    )

    with zipfile.ZipFile(chatbook_path, 'w') as zf:
        manifest_dict = manifest_to_dict(manifest)
        manifest_dict['content_items'] = [
            {
                "id": "note2",
                "type": "note",
                "title": "Imported Note",
                "file_path": "content/notes/note_note2.md"
            }
        ]
        zf.writestr('manifest.json', json.dumps(manifest_dict))

        zf.writestr('content/notes/note_note2.md', """---
id: note2
title: Imported Note
---

Imported content""")

    return str(chatbook_path)


class TestChatbookIntegration:
    """Integration tests for Chatbook service."""

    def test_service_initialization(self, service, test_db):
        """Test that service initializes with real database."""
        assert service.db == test_db
        assert service.user_id == "test_user"

        # Check that directories were created
        assert service.export_dir.exists()
        assert service.import_dir.exists()
        assert service.temp_dir.exists()

    def test_create_export_job(self, service):
        """Test creating an export job with real database."""
        result = service.create_export_job(
            name="Test Export",
            description="Test export job",
            content_types=["conversations", "notes"]
        )

        assert result is not None
        assert "job_id" in result
        job_id = result["job_id"]
        assert len(job_id) == 36  # UUID format

        # Verify job was saved to database
        job = service.get_export_job(job_id)
        assert job is not None
        assert job.chatbook_name == "Test Export"
        assert job.status == ExportStatus.PENDING

    @pytest.mark.asyncio
    async def test_export_chatbook_full_cycle(self, service):
        """Test full export cycle with real database and file system."""
        note_id = service.db.add_note(
            title="Integration Export Note",
            content="Integration export content",
        )
        result = await service.create_chatbook(
            name="Integration Test Export",
            description="Testing full export",
            content_selections={
                ContentType.NOTE: [str(note_id)],
            },
            async_mode=False,
        )

        success, message, file_path = result
        assert success is True
        assert message == "Chatbook created successfully"
        assert file_path is not None
        assert os.path.exists(file_path)

        with zipfile.ZipFile(file_path, 'r') as zf:
            assert 'manifest.json' in zf.namelist()
            manifest_data = json.loads(zf.read('manifest.json'))
            assert manifest_data['name'] == "Integration Test Export"
            assert manifest_data['description'] == "Testing full export"
            assert len(manifest_data["content_items"]) == 1
            assert manifest_data["content_items"][0]["type"] == "note"

    @pytest.mark.asyncio
    async def test_import_chatbook_full_cycle(self, service, sample_chatbook_file):
        """Test full import cycle with real database."""
        result = await service.import_chatbook(
            file_path=sample_chatbook_file,
            conflict_resolution="skip",
        )

        success, message, details = result
        assert success is True
        assert message == "Successfully imported 1/1 items"
        assert details == {"imported_items": {"note": 1}, "warnings": []}
        imported_notes = service.db.list_notes(limit=10, offset=0)
        assert any(note["title"] == "Imported Note" for note in imported_notes)

    def test_preview_export(self, service):
        """Test preview export with real database data."""
        service.db.add_note(
            title="Preview Note",
            content="Preview export content",
        )
        result = service.preview_export(
            content_types=["notes"]
        )

        assert "notes" in result
        assert result["notes"] == 1

    def test_validate_chatbook_file(self, service, sample_chatbook_file):
        """Test validating a real chatbook file."""
        result = service.validate_chatbook_file(sample_chatbook_file)

        assert result["is_valid"] is True
        assert result["manifest"] is not None
        assert result["error"] is None

    def test_validate_invalid_file(self, service):
        """Test validating an invalid file."""
        invalid_file = service.temp_dir / f"invalid_{uuid4().hex}.txt"
        invalid_file.write_text("Not a chatbook")

        result = service.validate_chatbook_file(str(invalid_file))

        assert result["is_valid"] is False
        assert result["error"] is not None

    def test_list_export_jobs(self, service):
        """Test listing export jobs from real database."""
        # Create a few jobs
        result1 = service.create_export_job(
            name="Export 1",
            description="First export",
            content_types=["conversations"]
        )
        job_id1 = result1["job_id"]

        result2 = service.create_export_job(
            name="Export 2",
            description="Second export",
            content_types=["notes"]
        )
        job_id2 = result2["job_id"]

        # List jobs
        jobs = service.list_export_jobs()

        assert len(jobs) >= 2
        job_ids = [job["job_id"] for job in jobs]
        assert job_id1 in job_ids
        assert job_id2 in job_ids

    def test_get_export_job_status(self, service):
        """Test getting export job status from real database."""
        # Create a job
        result = service.create_export_job(
            name="Status Test",
            description="Testing status",
            content_types=["conversations"]
        )
        job_id = result["job_id"]

        # Get status
        status = service.get_export_job_status(job_id)

        assert status["job_id"] == job_id
        assert status["status"] == "pending"
        assert status["chatbook_name"] == "Status Test"

    def test_clean_old_exports(self, service, tmp_path):
        """Test cleaning old exports with real file system."""
        # Create some old export files
        old_file1 = service.export_dir / "old1.chatbook"
        old_file2 = service.export_dir / "old2.chatbook"
        old_file1.touch()
        old_file2.touch()

        # Create export jobs in database
        service.db.execute_query("""
            INSERT INTO export_jobs (job_id, user_id, status, chatbook_name, output_path, created_at)
            VALUES (?, ?, ?, ?, ?, datetime('now', '-10 days'))
        """, ("old1", "test_user", "completed", "Old 1", str(old_file1)))

        service.db.execute_query("""
            INSERT INTO export_jobs (job_id, user_id, status, chatbook_name, output_path, created_at)
            VALUES (?, ?, ?, ?, ?, datetime('now', '-10 days'))
        """, ("old2", "test_user", "completed", "Old 2", str(old_file2)))

        # Clean old exports
        count = service.clean_old_exports(days_old=7)

        assert count == 2
        assert not old_file1.exists()
        assert not old_file2.exists()

    def test_get_statistics(self, service):
        """Test getting statistics from real database."""
        # Create some test data
        service.create_export_job(name="Export 1", description="Test 1", content_types=[])
        service.create_export_job(name="Export 2", description="Test 2", content_types=[])

        # Get statistics
        stats = service.get_statistics()

        assert "exports" in stats
        assert "imports" in stats

        # Check export stats
        if "pending" in stats["exports"]:
            assert stats["exports"]["pending"] >= 2

    @pytest.mark.asyncio
    async def test_export_import_roundtrip(self, service, tmp_path):
        """Test full export and import roundtrip with real components."""
        note_id = service.db.add_note(
            title="Roundtrip Note",
            content="Roundtrip content",
        )
        export_result = await service.create_chatbook(
            name="Roundtrip Test",
            description="Testing roundtrip",
            content_selections={
                ContentType.NOTE: [str(note_id)],
            },
            async_mode=False,
        )

        success, message, export_path = export_result
        assert success is True
        assert message == "Chatbook created successfully"
        assert export_path is not None

        import_path = stage_export_for_import(service, export_path)
        import_result = await service.import_chatbook(
            file_path=import_path,
            conflict_resolution="rename",
        )

        import_success, import_message, details = import_result
        assert import_success is True
        assert import_message == "Successfully imported 1/1 items"
        assert details == {"imported_items": {"note": 1}, "warnings": []}

    @pytest.mark.asyncio
    async def test_async_export_job(self, service):
        """Test async export job processing."""
        note_id = service.db.add_note(
            title="Async Export Note",
            content="Async export content",
        )
        result = await service.create_chatbook(
            name="Async Export",
            description="Testing async",
            content_selections={ContentType.NOTE: [str(note_id)]},
            async_mode=True,
        )

        success, message, job_id = result

        assert success is True
        assert message == f"Export job started: {job_id}"
        assert job_id is not None

        status = service.get_export_job_status(job_id)
        assert status["job_id"] == job_id
        assert status["status"] == "pending"

    @pytest.mark.asyncio
    async def test_import_with_conflicts(self, service, sample_chatbook_file):
        """Test import with conflict handling."""
        result1 = await service.import_chatbook(
            file_path=sample_chatbook_file,
            conflict_resolution="skip",
        )
        assert result1[0] is True
        assert result1[1] == "Successfully imported 1/1 items"
        assert result1[2] == {"imported_items": {"note": 1}, "warnings": []}

        result2 = await service.import_chatbook(
            file_path=sample_chatbook_file,
            conflict_resolution="skip",
        )

        assert result2[0] is True
        assert result2[1] == "Import completed: All 1 items were skipped"
        assert result2[2] == {"imported_items": {}, "warnings": []}

    def test_cancel_export_job(self, service):
        """Test cancelling an export job."""
        # Create a job
        result = service.create_export_job(
            name="Cancel Test",
            description="Testing cancellation",
            content_types=[]
        )
        job_id = result["job_id"]

        # Cancel it
        cancel_result = service.cancel_export_job(job_id)
        assert cancel_result is True

        # Check status
        job = service.get_export_job(job_id)
        assert job.status == ExportStatus.CANCELLED
