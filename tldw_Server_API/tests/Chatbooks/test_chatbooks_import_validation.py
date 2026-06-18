import hashlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock
import zipfile

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.Chatbooks.chatbook_models import ConflictResolution
from tldw_Server_API.app.core.Chatbooks.chatbook_validators import ChatbookValidator
from tldw_Server_API.tests.Chatbooks.test_chatbook_security import (
    build_dangerous_file_archive_bytes,
    build_symlink_archive_bytes,
    build_traversal_archive_bytes,
)


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))

    mock_db = MagicMock()
    mock_db.execute_query.return_value = []
    connection = MagicMock()
    connection.execute = MagicMock()
    connection.close = MagicMock()
    mock_db.get_connection.return_value = connection

    return ChatbookService(user_id="test_user", db=mock_db)


def _write_v1_1_note_chatbook(
    archive_path: Path,
    *,
    inventory_hash: str | None = None,
    inventory_entries: list[dict] | None = None,
    extra_files: dict[str, bytes] | None = None,
    features_used: list[str] | None = None,
    unsupported_feature_behavior: str | None = None,
) -> Path:
    note_path = "content/notes/note_1.md"
    note_payload = b"---\ntitle: Imported Note\n---\n\nBody"
    actual_hash = f"sha256:{hashlib.sha256(note_payload).hexdigest()}"
    expected_hash = inventory_hash or actual_hash
    features = features_used or ["file_inventory", "integrity_metadata"]
    compatibility: dict[str, str] = {
        "min_reader_version": "1.0.0",
        "recommended_reader_version": "1.1.0",
    }
    if unsupported_feature_behavior is not None:
        compatibility["unsupported_feature_behavior"] = unsupported_feature_behavior
    if inventory_entries is None:
        inventory_entries = [
            {
                "path": note_path,
                "media_type": "text/markdown",
                "size_bytes": len(note_payload),
                "integrity": {
                    "status": "verified",
                    "algorithm": "sha256",
                    "value": expected_hash,
                },
                "role": "payload",
                "content_item_ids": ["1"],
            }
        ]

    manifest = {
        "version": "1.1.0",
        "name": "Import Validation",
        "description": "v1.1 import validation fixture",
        "author": None,
        "created_at": "2026-06-18T12:00:00+00:00",
        "updated_at": "2026-06-18T12:00:00+00:00",
        "export_id": "import-validation",
        "content_items": [
            {
                "id": "1",
                "type": "note",
                "title": "Imported Note",
                "description": None,
                "created_at": None,
                "updated_at": None,
                "tags": [],
                "metadata": {},
                "file_path": note_path,
                "checksum": expected_hash,
            }
        ],
        "relationships": [],
        "configuration": {
            "include_media": False,
            "include_embeddings": False,
            "include_generated_content": True,
            "media_quality": "compressed",
            "max_file_size_mb": 100,
        },
        "statistics": {
            "total_conversations": 0,
            "total_notes": 1,
            "total_characters": 0,
            "total_media_items": 0,
            "total_prompts": 0,
            "total_evaluations": 0,
            "total_embeddings": 0,
            "total_world_books": 0,
            "total_dictionaries": 0,
            "total_documents": 0,
            "total_explainer_sessions": 0,
            "total_size_bytes": len(note_payload),
        },
        "metadata": {"tags": [], "categories": [], "language": "en", "license": None},
        "user_info": {"user_id": "test_user"},
        "features_used": features,
        "producer": {"name": "tldw_server"},
        "source_instance": {},
        "compatibility": compatibility,
        "file_inventory": inventory_entries,
    }

    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(note_path, note_payload)
        for path, payload in (extra_files or {}).items():
            zf.writestr(path, payload)
        zf.writestr("manifest.json", json.dumps(manifest))
    return archive_path


@pytest.mark.parametrize(
    ("archive_bytes", "expected_error"),
    [
        (build_symlink_archive_bytes(), "symlink"),
        (build_traversal_archive_bytes(), "unsafe"),
        (build_dangerous_file_archive_bytes(), "dangerous"),
    ],
    ids=["symlink", "path-traversal", "dangerous-file-type"],
)
def test_validate_chatbook_file_rejects_malicious_archive_members(
    service,
    archive_bytes,
    expected_error,
):
    archive_path = service.import_dir / "malicious.chatbook"
    archive_path.write_bytes(archive_bytes)

    validation = service.validate_chatbook_file(str(archive_path))

    assert validation["is_valid"] is False
    assert validation["manifest"] is None
    assert expected_error in (validation["error"] or "").lower()


def test_validate_chatbook_file_resolves_tokens_before_archive_validation(
    service,
    monkeypatch,
):
    archive_path = service.import_dir / "tokenized.chatbook"
    archive_path.write_bytes(build_traversal_archive_bytes(member_name="content/notes/safe.md"))
    token = service._build_import_file_token(archive_path)
    seen_paths: list[str] = []

    def _record_path(path: str):
        seen_paths.append(path)
        return True, None

    monkeypatch.setattr(ChatbookValidator, "validate_zip_file", _record_path)

    validation = service.validate_chatbook_file(token)

    assert validation["is_valid"] is True
    assert seen_paths == [str(archive_path.resolve())]


def test_v1_1_import_rejects_checksum_mismatch_before_writes(service):
    archive_path = _write_v1_1_note_chatbook(
        service.import_dir / "checksum_mismatch.chatbook",
        inventory_hash=f"sha256:{'0' * 64}",
    )
    import_notes = MagicMock()
    service._import_notes = import_notes
    service.db.add_note = MagicMock(return_value=1)

    success, message, details = service._import_chatbook_sync(
        file_path=str(archive_path),
        content_selections=None,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        import_media=False,
        import_embeddings=False,
    )

    assert success is False
    assert "checksum" in message.lower() or "validation" in message.lower()
    assert details is not None
    assert any("checksum" in error.lower() or "validation" in error.lower() for error in details["errors"])
    assert details["imported_items"] == {}
    import_notes.assert_not_called()
    service.db.add_note.assert_not_called()


def test_v1_1_import_rejects_empty_inventory_before_writes(service):
    archive_path = _write_v1_1_note_chatbook(
        service.import_dir / "empty_inventory.chatbook",
        inventory_entries=[],
    )
    import_notes = MagicMock()
    service._import_notes = import_notes
    service.db.add_note = MagicMock(return_value=1)

    success, message, details = service._import_chatbook_sync(
        file_path=str(archive_path),
        content_selections=None,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        import_media=False,
        import_embeddings=False,
    )

    assert success is False
    assert "inventory" in message.lower() or "validation" in message.lower()
    assert details is not None
    assert any("missing inventory entry" in error.lower() for error in details["errors"])
    assert details["imported_items"] == {}
    import_notes.assert_not_called()
    service.db.add_note.assert_not_called()


def test_v1_1_import_rejects_missing_payload_inventory_entry_before_writes(service):
    readme_payload = b"# Import Validation\n"
    readme_hash = f"sha256:{hashlib.sha256(readme_payload).hexdigest()}"
    archive_path = _write_v1_1_note_chatbook(
        service.import_dir / "missing_payload_inventory.chatbook",
        inventory_entries=[
            {
                "path": "README.md",
                "media_type": "text/markdown",
                "size_bytes": len(readme_payload),
                "integrity": {
                    "status": "verified",
                    "algorithm": "sha256",
                    "value": readme_hash,
                },
                "role": "readme",
                "content_item_ids": [],
            }
        ],
        extra_files={"README.md": readme_payload},
    )
    import_notes = MagicMock()
    service._import_notes = import_notes
    service.db.add_note = MagicMock(return_value=1)

    success, message, details = service._import_chatbook_sync(
        file_path=str(archive_path),
        content_selections=None,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        import_media=False,
        import_embeddings=False,
    )

    assert success is False
    assert "inventory" in message.lower() or "validation" in message.lower()
    assert details is not None
    assert any("content/notes/note_1.md" in error for error in details["errors"])
    assert any("missing inventory entry" in error.lower() for error in details["errors"])
    assert details["imported_items"] == {}
    import_notes.assert_not_called()
    service.db.add_note.assert_not_called()


def test_v1_1_import_rejects_unknown_feature_policy_before_writes(service):
    archive_path = _write_v1_1_note_chatbook(
        service.import_dir / "unknown_feature_reject.chatbook",
        features_used=["file_inventory", "future_feature"],
        unsupported_feature_behavior="reject_import",
    )
    import_notes = MagicMock()
    service._import_notes = import_notes
    service.db.add_note = MagicMock(return_value=1)

    success, message, details = service._import_chatbook_sync(
        file_path=str(archive_path),
        content_selections=None,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        import_media=False,
        import_embeddings=False,
    )

    assert success is False
    assert "unsupported" in message.lower() or "validation" in message.lower()
    assert details is not None
    combined = " ".join(details["errors"] + details["warnings"]).lower()
    assert "future_feature" in combined
    assert "unsupported" in combined
    assert details["imported_items"] == {}
    import_notes.assert_not_called()
    service.db.add_note.assert_not_called()


def test_v1_1_import_warns_for_unknown_feature_without_reject_policy(service):
    archive_path = _write_v1_1_note_chatbook(
        service.import_dir / "unknown_feature_warn.chatbook",
        features_used=["file_inventory", "future_feature"],
    )

    def _mark_note_imported(*args, **kwargs):
        status = args[-1]
        status.processed_items += 1
        status.successful_items += 1

    import_notes = MagicMock(side_effect=_mark_note_imported)
    service._import_notes = import_notes

    success, message, details = service._import_chatbook_sync(
        file_path=str(archive_path),
        content_selections=None,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        import_media=False,
        import_embeddings=False,
    )

    assert success is True, message
    assert details is not None
    assert any("future_feature" in warning for warning in details["warnings"])
    import_notes.assert_called_once()
