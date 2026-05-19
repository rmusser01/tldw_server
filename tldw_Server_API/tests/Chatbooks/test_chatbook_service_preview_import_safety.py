import json
import zipfile
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_models import ConflictResolution
from tldw_Server_API.app.core.Chatbooks.exceptions import SecurityError
from tldw_Server_API.tests.Chatbooks.test_chatbook_service import (  # noqa: F401
    manifest_to_dict,
    mock_db,
    sample_manifest,
    service,
)


def test_resolve_import_archive_path_rejects_outside_paths(service):
    """Path resolution should raise SecurityError with a stable violation type."""
    with pytest.raises(SecurityError) as exc_info:
        service._resolve_import_archive_path("../../outside.chatbook")

    assert exc_info.value.context.get("violation_type") == "import_path_outside_allowed_directories"


def test_import_chatbook_cleans_temp_dir_on_failure(service, tmp_path):
    """Temporary extraction directories should not linger after import errors."""
    temp_dir = tmp_path / "chatbooks_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    service.temp_dir = temp_dir

    bad_manifest = {
        "version": "invalid",
        "name": "Broken Chatbook",
        "description": "Invalid version should trigger failure",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "content_items": [],
        "configuration": {},
        "statistics": {},
        "metadata": {},
        "user_info": {},
    }

    archive_path = temp_dir / "broken.chatbook"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(bad_manifest))

    assert not any(temp_dir.glob("import_*"))

    success, message, _ = service._import_chatbook_sync(
        file_path=str(archive_path),
        content_selections=None,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        import_media=True,
        import_embeddings=False,
    )

    assert success is False
    assert "Error importing chatbook" in message
    assert not any(temp_dir.glob("import_*"))


def test_preview_chatbook_cleans_temp_dir_on_failure(service, tmp_path):
    """Preview extractions must be removed even when parsing fails."""
    temp_dir = tmp_path / "chatbooks_preview_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    service.temp_dir = temp_dir

    bad_manifest = {
        "version": "invalid",
        "name": "Preview Failure",
        "description": "Invalid version forces parse error",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "content_items": [],
        "configuration": {},
        "statistics": {},
        "metadata": {},
        "user_info": {},
    }

    archive_path = temp_dir / "broken_preview.chatbook"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(bad_manifest))

    assert not any(temp_dir.glob("preview_*"))

    manifest, error = service.preview_chatbook(str(archive_path))

    assert manifest is None
    assert error is not None
    assert not any(temp_dir.glob("preview_*"))


def test_validate_chatbook_file(service, sample_manifest):
    """Test validating a chatbook file structure."""
    chatbook_path = service.temp_dir / f"validate_{uuid4().hex}.chatbook"
    with zipfile.ZipFile(chatbook_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest_to_dict(sample_manifest)))
        zf.writestr("conversations/test.json", "{}")

    result = service.validate_chatbook_file(str(chatbook_path))

    assert result["is_valid"] is True
    assert "manifest" in result


def test_validate_invalid_chatbook(service):
    """Test validating an invalid chatbook file."""
    invalid_path = service.temp_dir / f"invalid_{uuid4().hex}.txt"
    invalid_path.write_bytes(b"Not a zip file")

    result = service.validate_chatbook_file(str(invalid_path))

    assert result["is_valid"] is False
    assert "error" in result
