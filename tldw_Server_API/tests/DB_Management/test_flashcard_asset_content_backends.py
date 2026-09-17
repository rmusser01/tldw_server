"""Flashcard image content works with SQLite rows and PostgreSQL mappings."""

import hashlib
import io
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from tldw_Server_API.app.api.v1.endpoints import flashcards
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import BackendType, CharactersRAGDB


def _build_png_bytes():
    buffer = io.BytesIO()
    Image.new("RGBA", (1, 1), (255, 0, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


PNG_BYTES = _build_png_bytes()


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def asset_db(request, tmp_path):
    """Use the official per-test PostgreSQL fixture and a real SQLite control."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "asset-content.db", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_connection()
        if backend is not None:
            backend.get_pool().close_all()


def _client(db):
    app = FastAPI()
    app.include_router(flashcards.router, prefix="/api/v1")
    app.dependency_overrides[flashcards.get_chacha_db_for_user] = lambda: db
    return TestClient(app, raise_server_exceptions=False)


def _save_asset(db):
    return db.add_flashcard_asset(
        image_bytes=PNG_BYTES,
        mime_type="image/png",
        original_filename="pixel.png",
        width=1,
        height=1,
    )


@pytest.mark.integration
def test_populated_asset_content_round_trips_exact_bytes(asset_db):
    asset_uuid = _save_asset(asset_db)
    metadata = asset_db.get_flashcard_asset(asset_uuid)
    assert metadata["uuid"] == asset_uuid
    assert metadata["byte_size"] == len(PNG_BYTES)
    assert metadata["sha256"] == hashlib.sha256(PNG_BYTES).hexdigest()
    assert asset_db.get_flashcard_asset_content(asset_uuid) == PNG_BYTES


@pytest.mark.integration
def test_actual_upload_and_repeated_content_http_requests_return_image(asset_db):
    with _client(asset_db) as client:
        uploaded = client.post("/api/v1/flashcards/assets", files={"file": ("pixel.png", PNG_BYTES, "image/png")})
        assert uploaded.status_code == 200, uploaded.text
        metadata = uploaded.json()
        assert metadata["byte_size"] == len(PNG_BYTES)
        assert (metadata["width"], metadata["height"]) == (1, 1)
        assert metadata["reference"] == f"flashcard-asset://{metadata['asset_uuid']}"
        for _ in range(2):
            content = client.get(f"/api/v1/flashcards/assets/{metadata['asset_uuid']}/content")
            assert content.status_code == 200, content.text
            assert content.headers["content-type"] == "image/png"
            assert content.content == PNG_BYTES


@pytest.mark.integration
@pytest.mark.parametrize("state", ["missing", "deleted"])
def test_unavailable_asset_returns_none_and_actual_http_404(asset_db, state):
    asset_uuid = str(uuid.uuid4())
    if state == "deleted":
        asset_uuid = _save_asset(asset_db)
        with asset_db.transaction() as conn:
            conn.execute("UPDATE flashcard_assets SET deleted = ? WHERE uuid = ?", (True, asset_uuid))
    assert asset_db.get_flashcard_asset_content(asset_uuid) is None
    with _client(asset_db) as client:
        content = client.get(f"/api/v1/flashcards/assets/{asset_uuid}/content")
        assert content.status_code == 404, content.text
        assert content.json() == {"detail": "Flashcard asset not found"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "blob", [PNG_BYTES, memoryview(PNG_BYTES), b"", None], ids=["bytes", "memoryview", "empty", "null"]
)
def test_named_binary_row_preserves_conversion_and_defensive_null(monkeypatch, blob):
    """NULL is defensive only: both real database schemas require image_data."""
    db = object.__new__(CharactersRAGDB)
    db.client_id = "row-owner"
    monkeypatch.setattr(CharactersRAGDB, "backend_type", property(lambda _self: BackendType.POSTGRESQL))
    monkeypatch.setattr(
        db, "execute_query", lambda *_args, **_kwargs: SimpleNamespace(fetchone=lambda: {"image_data": blob})
    )
    result = db.get_flashcard_asset_content("row-boundary-control")
    assert result == (None if blob is None else bytes(blob))
    assert result is None or isinstance(result, bytes)
