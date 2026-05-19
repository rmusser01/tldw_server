import json
import os
import shutil
import zipfile
from pathlib import Path

import jsonschema
import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def manifest_contract_service(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))

    db_path = tmp_path / "manifest_contract.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="manifest-contract")
    service = ChatbookService(user_id="test_user", db=db)

    yield service

    if hasattr(service, "temp_dir") and service.temp_dir.exists():
        shutil.rmtree(service.temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_real_export_manifest_matches_canonical_schema(manifest_contract_service):
    success, _message, archive_path = await manifest_contract_service.create_chatbook(
        name="Schema Contract",
        description="Contract validation export",
        content_selections={},
        async_mode=False,
    )

    assert success is True
    assert archive_path is not None

    with zipfile.ZipFile(archive_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))

    schema_path = Path(__file__).resolve().parents[3] / "Docs" / "Schemas" / "chatbooks_manifest_v1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    jsonschema.validate(manifest, schema)
    assert manifest["version"] == "1.0.0"
    assert "entries" not in manifest
