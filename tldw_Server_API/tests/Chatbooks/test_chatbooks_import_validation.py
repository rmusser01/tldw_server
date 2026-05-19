import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
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
