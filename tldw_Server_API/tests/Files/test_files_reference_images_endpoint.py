import asyncio
import shutil
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.Image_Generation.reference_images import ReferenceImageOperationalError
from tldw_Server_API.app.core.Storage.filesystem_storage import FileSystemStorage
from tldw_Server_API.app.core.Storage.storage_interface import StorageError
from tldw_Server_API.tests.test_utils import create_test_media


pytestmark = pytest.mark.integration


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (1, 1), (255, 0, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _create_test_image_media(db: MediaDatabase, *, title: str, content_hash: str) -> int:
    cursor = db.execute_query(
        "INSERT INTO Media (title, type, content, author, content_hash, uuid, last_modified, client_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            title,
            "image",
            "image content",
            "Test Author",
            content_hash,
            f"uuid-{content_hash}",
            db._get_current_utc_timestamp_str(),
            db.client_id or "test_client",
        ),
        commit=True,
    )
    media_id = getattr(cursor, "lastrowid", None)
    if media_id:
        return media_id
    raise RuntimeError("Failed to create test image media")


class FailingStorage:
    async def exists(self, path: str) -> bool:
        raise StorageError("backend down", path=path)

    async def retrieve(self, path: str):
        raise AssertionError("retrieve should not be called when exists fails")


class ReadFailingHandle:
    def read(self) -> bytes:
        raise OSError("read failed")


class ReadFailingStorage:
    async def exists(self, path: str) -> bool:
        return True

    async def retrieve(self, path: str) -> ReadFailingHandle:
        return ReadFailingHandle()


@pytest.fixture()
def client_with_user(monkeypatch):
    async def override_user():
        return User(id=321, username="tester", email=None, is_active=True)

    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "321")

    base_dir = Path.cwd() / "Databases" / "test_user_dbs_files_reference_images"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    users_db_path = base_dir / "users.db"
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{users_db_path}")

    app = None
    try:
        from importlib import import_module, reload
        from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_single_user_rbac_seed_if_needed
        from tldw_Server_API.app.core.DB_Management.Users_DB import reset_users_db

        asyncio.run(reset_db_pool())
        asyncio.run(reset_users_db())
        asyncio.run(ensure_single_user_rbac_seed_if_needed())

        mod = import_module("tldw_Server_API.app.main")
        mod = reload(mod)
        app = mod.app
        app.dependency_overrides[get_request_user] = override_user
        with TestClient(app) as client:
            yield client
    finally:
        if app is not None:
            app.dependency_overrides.clear()
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass

def test_reference_images_endpoint_returns_only_eligible_images(client_with_user) -> None:
    png_bytes = _png_bytes()
    base_dir = Path.cwd() / "Databases" / "test_user_dbs_files_reference_images"
    media_db = MediaDatabase(db_path=str(base_dir / "321" / "Media_DB_v2.db"), client_id="321")
    storage = FileSystemStorage(base_path=base_dir / "user_files")
    try:
        media_id = _create_test_image_media(media_db, title="Poster source", content_hash="endpoint-image-1")
        storage_path = asyncio.run(
            storage.store(user_id="321", media_id=media_id, filename="poster-source.png", data=png_bytes, mime_type="image/png")
        )
        media_db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path=storage_path,
            original_filename="poster-source.png",
            file_size=len(png_bytes),
            mime_type="image/png",
        )
        missing_media_id = _create_test_image_media(media_db, title="Missing source", content_hash="endpoint-image-2")
        media_db.insert_media_file(
            media_id=missing_media_id,
            file_type="original",
            storage_path="321/media/404/missing.png",
            original_filename="missing.png",
            file_size=len(png_bytes),
            mime_type="image/png",
        )
        cross_user_media_id = _create_test_image_media(media_db, title="Cross-user source", content_hash="endpoint-image-4")
        cross_user_storage_path = asyncio.run(
            storage.store(user_id="999", media_id=cross_user_media_id, filename="cross-user.png", data=png_bytes, mime_type="image/png")
        )
        media_db.insert_media_file(
            media_id=cross_user_media_id,
            file_type="original",
            storage_path=cross_user_storage_path,
            original_filename="cross-user.png",
            file_size=len(png_bytes),
            mime_type="image/png",
        )
        document_media_id = create_test_media(media_db, title="Document with preview", content="doc content")
        preview_storage_path = asyncio.run(
            storage.store(user_id="321", media_id=document_media_id, filename="preview.png", data=png_bytes, mime_type="image/png")
        )
        media_db.insert_media_file(
            media_id=document_media_id,
            file_type="thumbnail",
            storage_path=preview_storage_path,
            original_filename="preview.png",
            file_size=len(png_bytes),
            mime_type="image/png",
        )
        trashed_media_id = _create_test_image_media(media_db, title="Trashed source", content_hash="endpoint-image-3")
        trashed_storage_path = asyncio.run(
            storage.store(user_id="321", media_id=trashed_media_id, filename="trashed.png", data=png_bytes, mime_type="image/png")
        )
        media_db.insert_media_file(
            media_id=trashed_media_id,
            file_type="original",
            storage_path=trashed_storage_path,
            original_filename="trashed.png",
            file_size=len(png_bytes),
            mime_type="image/png",
        )
        assert media_db.mark_as_trash(trashed_media_id) is True

        async def _override_media_db():
            try:
                yield media_db
            finally:
                pass

        client_with_user.app.dependency_overrides[get_media_db_for_user] = _override_media_db
        with patch("tldw_Server_API.app.core.Image_Generation.reference_images.get_storage_backend", return_value=storage):
            response = client_with_user.get("/api/v1/files/reference-images")
    finally:
        client_with_user.app.dependency_overrides.pop(get_media_db_for_user, None)
        media_db.close_connection()

    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["file_id"] == 1
    assert items[0]["title"] == "Poster source"
    assert items[0]["mime_type"] == "image/png"
    assert items[0]["width"] == 1
    assert items[0]["height"] == 1
    assert items[0]["created_at"]


def test_reference_images_endpoint_surfaces_storage_failures(client_with_user) -> None:
    png_bytes = _png_bytes()
    base_dir = Path.cwd() / "Databases" / "test_user_dbs_files_reference_images"
    media_db = MediaDatabase(db_path=str(base_dir / "321" / "Media_DB_v2.db"), client_id="321")
    storage = FileSystemStorage(base_path=base_dir / "user_files")
    try:
        media_id = _create_test_image_media(media_db, title="Poster source", content_hash="endpoint-storage-failure")
        storage_path = asyncio.run(
            storage.store(user_id="321", media_id=media_id, filename="poster-source.png", data=png_bytes, mime_type="image/png")
        )
        media_db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path=storage_path,
            original_filename="poster-source.png",
            file_size=len(png_bytes),
            mime_type="image/png",
        )

        async def _override_media_db():
            try:
                yield media_db
            finally:
                pass

        client_with_user.app.dependency_overrides[get_media_db_for_user] = _override_media_db
        with patch("tldw_Server_API.app.core.Image_Generation.reference_images.get_storage_backend", return_value=FailingStorage()):
            response = client_with_user.get("/api/v1/files/reference-images")
    finally:
        client_with_user.app.dependency_overrides.pop(get_media_db_for_user, None)
        media_db.close_connection()

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "reference_image_storage_unavailable"


def test_reference_images_endpoint_sanitizes_operational_failures(client_with_user, monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import files as files_endpoint

    async def _raise_operational_error(*_args, **_kwargs):
        raise ReferenceImageOperationalError("reference image backend exploded")

    monkeypatch.setattr(
        files_endpoint,
        "list_reference_image_candidates",
        _raise_operational_error,
    )

    response = client_with_user.get("/api/v1/files/reference-images")

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "reference_image_storage_unavailable"


def test_reference_images_endpoint_surfaces_read_failures(client_with_user) -> None:
    png_bytes = _png_bytes()
    base_dir = Path.cwd() / "Databases" / "test_user_dbs_files_reference_images"
    media_db = MediaDatabase(db_path=str(base_dir / "321" / "Media_DB_v2.db"), client_id="321")
    storage = FileSystemStorage(base_path=base_dir / "user_files")
    try:
        media_id = _create_test_image_media(media_db, title="Poster source", content_hash="endpoint-read-failure")
        storage_path = asyncio.run(
            storage.store(user_id="321", media_id=media_id, filename="poster-source.png", data=png_bytes, mime_type="image/png")
        )
        media_db.insert_media_file(
            media_id=media_id,
            file_type="original",
            storage_path=storage_path,
            original_filename="poster-source.png",
            file_size=len(png_bytes),
            mime_type="image/png",
        )

        async def _override_media_db():
            try:
                yield media_db
            finally:
                pass

        client_with_user.app.dependency_overrides[get_media_db_for_user] = _override_media_db
        with patch("tldw_Server_API.app.core.Image_Generation.reference_images.get_storage_backend", return_value=ReadFailingStorage()):
            response = client_with_user.get("/api/v1/files/reference-images")
    finally:
        client_with_user.app.dependency_overrides.pop(get_media_db_for_user, None)
        media_db.close_connection()

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "reference_image_storage_unavailable"
