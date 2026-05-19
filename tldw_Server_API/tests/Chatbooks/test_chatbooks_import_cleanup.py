from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))

    db = MagicMock()
    db.execute_query.return_value = []

    return ChatbookService(user_id="1", db=db)


def test_try_delete_import_file_resolves_temp_token(service):
    staged = Path(service.temp_dir) / "imports" / "sample.chatbook"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"zip")

    token = service._build_import_file_token(staged)

    assert service._try_delete_import_file(token) == 1
    assert not staged.exists()


def test_try_delete_import_file_resolves_import_token(service):
    staged = Path(service.import_dir) / "sample.chatbook"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"zip")

    token = service._build_import_file_token(staged)

    assert token.startswith("import/")
    assert service._try_delete_import_file(token) == 1
    assert not staged.exists()


def test_try_delete_import_file_rejects_outside_base_dirs(service, tmp_path):
    outside = tmp_path / "outside.chatbook"
    outside.write_bytes(b"zip")

    assert service._try_delete_import_file(str(outside)) == 0
    assert outside.exists()
