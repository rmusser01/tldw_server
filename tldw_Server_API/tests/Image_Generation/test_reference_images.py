from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.exceptions import FileArtifactsValidationError
from tldw_Server_API.app.core.Image_Generation import reference_images as reference_images_module
from tldw_Server_API.app.core.Image_Generation.reference_images import (
    ReferenceImageOperationalError,
    list_reference_image_candidates,
    resolve_reference_image,
)
from tldw_Server_API.app.core.Storage.filesystem_storage import FileSystemStorage
from tldw_Server_API.app.core.Storage.storage_interface import StorageError
from tldw_Server_API.tests.test_utils import create_test_media


pytestmark = pytest.mark.unit


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (1, 1), (255, 0, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _create_test_image_media(db: MediaDatabase, *, title: str, content: str, content_hash: str) -> int:
    cursor = db.execute_query(
        "INSERT INTO Media (title, type, content, author, content_hash, uuid, last_modified, client_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            title,
            "image",
            content,
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


@pytest.fixture()
def media_db(tmp_path: Path) -> MediaDatabase:
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="777")
    yield db
    db.close_connection()


@pytest.fixture()
def storage(tmp_path: Path) -> FileSystemStorage:
    return FileSystemStorage(base_path=tmp_path / "user_files")


async def _store_image(
    storage: FileSystemStorage,
    *,
    user_id: str,
    media_id: int,
    filename: str,
    content: bytes,
) -> str:
    return await storage.store(user_id=user_id, media_id=media_id, filename=filename, data=content, mime_type="image/png")


class FailingStorage:
    async def exists(self, path: str) -> bool:
        raise StorageError("backend down", path=path)

    async def retrieve(self, path: str):
        raise AssertionError("retrieve should not be reached when exists fails")


class ReadFailingHandle:
    def read(self) -> bytes:
        raise OSError("read failed")


class ReadFailingStorage:
    async def exists(self, path: str) -> bool:
        return True

    async def retrieve(self, path: str) -> ReadFailingHandle:
        return ReadFailingHandle()


class MustNotReadStorage:
    async def exists(self, path: str) -> bool:
        raise AssertionError("oversized reference candidate should not hit storage")

    async def retrieve(self, path: str):
        raise AssertionError("oversized reference candidate should not hit storage")


@pytest.mark.asyncio
async def test_resolve_reference_image_returns_normalized_image(media_db: MediaDatabase, storage: FileSystemStorage) -> None:
    png_bytes = _png_bytes()
    media_id = _create_test_image_media(
        media_db,
        title="Poster source",
        content="image content",
        content_hash="image-hash-1",
    )
    storage_path = await _store_image(storage, user_id="777", media_id=media_id, filename="reference.png", content=png_bytes)
    media_db.insert_media_file(
        media_id=media_id,
        file_type="original",
        storage_path=storage_path,
        original_filename="reference.png",
        file_size=len(png_bytes),
        mime_type="image/png",
    )
    row = media_db.get_media_file(media_id, "original")

    resolved = await resolve_reference_image(media_db, user_id="777", file_id=row["id"], storage=storage)

    assert resolved.file_id == row["id"]
    assert resolved.filename == "reference.png"
    assert resolved.mime_type == "image/png"
    assert resolved.width == 1
    assert resolved.height == 1
    assert resolved.bytes_len == len(png_bytes)
    assert resolved.content == png_bytes
    assert resolved.temp_path is None


@pytest.mark.asyncio
async def test_resolve_reference_image_rejects_missing_durable_storage(media_db: MediaDatabase, storage: FileSystemStorage) -> None:
    png_bytes = _png_bytes()
    media_id = _create_test_image_media(
        media_db,
        title="Poster source",
        content="image content",
        content_hash="image-hash-2",
    )
    media_db.insert_media_file(
        media_id=media_id,
        file_type="original",
        storage_path="777/media/1/missing.png",
        original_filename="missing.png",
        file_size=len(png_bytes),
        mime_type="image/png",
    )
    row = media_db.get_media_file(media_id, "original")

    with pytest.raises(FileArtifactsValidationError, match="reference_image_invalid"):
        await resolve_reference_image(media_db, user_id="777", file_id=row["id"], storage=storage)


@pytest.mark.asyncio
async def test_resolve_reference_image_rejects_cross_user_storage_path(media_db: MediaDatabase, storage: FileSystemStorage) -> None:
    png_bytes = _png_bytes()
    media_id = _create_test_image_media(
        media_db,
        title="Cross-user image",
        content="image content",
        content_hash="image-hash-cross-user",
    )
    other_user_storage_path = await _store_image(
        storage,
        user_id="778",
        media_id=media_id,
        filename="cross-user.png",
        content=png_bytes,
    )
    media_db.insert_media_file(
        media_id=media_id,
        file_type="original",
        storage_path=other_user_storage_path,
        original_filename="cross-user.png",
        file_size=len(png_bytes),
        mime_type="image/png",
    )
    row = media_db.get_media_file(media_id, "original")

    with pytest.raises(FileArtifactsValidationError, match="reference_image_invalid"):
        await resolve_reference_image(media_db, user_id="777", file_id=row["id"], storage=storage)


@pytest.mark.asyncio
async def test_resolve_reference_image_rejects_traversal_storage_path(media_db: MediaDatabase, storage: FileSystemStorage) -> None:
    png_bytes = _png_bytes()
    media_id = _create_test_image_media(
        media_db,
        title="Traversal image",
        content="image content",
        content_hash="image-hash-traversal",
    )
    media_db.insert_media_file(
        media_id=media_id,
        file_type="original",
        storage_path="777/media/1/../../../778/media/2/secret.png",
        original_filename="secret.png",
        file_size=len(png_bytes),
        mime_type="image/png",
    )
    row = media_db.get_media_file(media_id, "original")

    with pytest.raises(FileArtifactsValidationError, match="reference_image_invalid"):
        await resolve_reference_image(media_db, user_id="777", file_id=row["id"], storage=storage)


@pytest.mark.asyncio
async def test_list_reference_image_candidates_filters_ineligible_records(media_db: MediaDatabase, storage: FileSystemStorage) -> None:
    png_bytes = _png_bytes()
    eligible_media_id = _create_test_image_media(
        media_db,
        title="Eligible image",
        content="image content",
        content_hash="image-hash-3",
    )
    eligible_storage_path = await _store_image(
        storage,
        user_id="777",
        media_id=eligible_media_id,
        filename="eligible.png",
        content=png_bytes,
    )
    media_db.insert_media_file(
        media_id=eligible_media_id,
        file_type="original",
        storage_path=eligible_storage_path,
        original_filename="eligible.png",
        file_size=len(png_bytes),
        mime_type="image/png",
    )
    missing_media_id = _create_test_image_media(
        media_db,
        title="Missing bytes",
        content="image content",
        content_hash="image-hash-4",
    )
    media_db.insert_media_file(
        media_id=missing_media_id,
        file_type="original",
        storage_path="777/media/999/missing.png",
        original_filename="missing.png",
        file_size=len(png_bytes),
        mime_type="image/png",
    )
    cross_user_media_id = _create_test_image_media(
        media_db,
        title="Cross-user image",
        content="image content",
        content_hash="image-hash-cross-list",
    )
    cross_user_storage_path = await _store_image(
        storage,
        user_id="778",
        media_id=cross_user_media_id,
        filename="cross-user.png",
        content=png_bytes,
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
    preview_storage_path = await _store_image(
        storage,
        user_id="777",
        media_id=document_media_id,
        filename="preview.png",
        content=png_bytes,
    )
    media_db.insert_media_file(
        media_id=document_media_id,
        file_type="preview",
        storage_path=preview_storage_path,
        original_filename="preview.png",
        file_size=len(png_bytes),
        mime_type="image/png",
    )
    non_image_media_id = create_test_media(media_db, title="Not an image", content="image content")
    text_storage_path = await storage.store(
        user_id="777",
        media_id=non_image_media_id,
        filename="notes.txt",
        data=b"not image",
        mime_type="text/plain",
    )
    media_db.insert_media_file(
        media_id=non_image_media_id,
        file_type="original",
        storage_path=text_storage_path,
        original_filename="notes.txt",
        file_size=9,
        mime_type="text/plain",
    )
    trashed_media_id = _create_test_image_media(
        media_db,
        title="Trashed image",
        content="image content",
        content_hash="image-hash-5",
    )
    trashed_storage_path = await _store_image(
        storage,
        user_id="777",
        media_id=trashed_media_id,
        filename="trashed.png",
        content=png_bytes,
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
    deleted_media_id = _create_test_image_media(
        media_db,
        title="Deleted image",
        content="image content",
        content_hash="image-hash-6",
    )
    deleted_storage_path = await _store_image(
        storage,
        user_id="777",
        media_id=deleted_media_id,
        filename="deleted.png",
        content=png_bytes,
    )
    media_db.insert_media_file(
        media_id=deleted_media_id,
        file_type="original",
        storage_path=deleted_storage_path,
        original_filename="deleted.png",
        file_size=len(png_bytes),
        mime_type="image/png",
    )
    deleted_row = media_db.get_media_file(deleted_media_id, "original")
    media_db.soft_delete_media_file(deleted_row["id"])

    candidates = await list_reference_image_candidates(media_db, user_id="777", storage=storage)

    assert [(candidate.file_id, candidate.title, candidate.mime_type) for candidate in candidates] == [
        (1, "Eligible image", "image/png"),
    ]


@pytest.mark.asyncio
async def test_list_reference_image_candidates_propagates_storage_failures(media_db: MediaDatabase, storage: FileSystemStorage) -> None:
    png_bytes = _png_bytes()
    media_id = _create_test_image_media(
        media_db,
        title="Eligible image",
        content="image content",
        content_hash="image-hash-storage-failure",
    )
    storage_path = await _store_image(
        storage,
        user_id="777",
        media_id=media_id,
        filename="eligible.png",
        content=png_bytes,
    )
    media_db.insert_media_file(
        media_id=media_id,
        file_type="original",
        storage_path=storage_path,
        original_filename="eligible.png",
        file_size=len(png_bytes),
        mime_type="image/png",
    )

    with pytest.raises(ReferenceImageOperationalError, match="reference_image_storage_unavailable"):
        await list_reference_image_candidates(media_db, user_id="777", storage=FailingStorage())


@pytest.mark.asyncio
async def test_list_reference_image_candidates_propagates_read_failures(media_db: MediaDatabase, storage: FileSystemStorage) -> None:
    png_bytes = _png_bytes()
    media_id = _create_test_image_media(
        media_db,
        title="Eligible image",
        content="image content",
        content_hash="image-hash-read-failure",
    )
    storage_path = await _store_image(
        storage,
        user_id="777",
        media_id=media_id,
        filename="eligible.png",
        content=png_bytes,
    )
    media_db.insert_media_file(
        media_id=media_id,
        file_type="original",
        storage_path=storage_path,
        original_filename="eligible.png",
        file_size=len(png_bytes),
        mime_type="image/png",
    )

    with pytest.raises(ReferenceImageOperationalError, match="reference_image_storage_unavailable"):
        await list_reference_image_candidates(media_db, user_id="777", storage=ReadFailingStorage())


@pytest.mark.asyncio
async def test_list_reference_image_candidates_pages_past_newer_invalid_rows(media_db: MediaDatabase, storage: FileSystemStorage) -> None:
    png_bytes = _png_bytes()
    valid_media_id = _create_test_image_media(
        media_db,
        title="Old valid image",
        content="image content",
        content_hash="image-hash-valid-oldest",
    )
    valid_storage_path = await _store_image(
        storage,
        user_id="777",
        media_id=valid_media_id,
        filename="valid.png",
        content=png_bytes,
    )
    media_db.insert_media_file(
        media_id=valid_media_id,
        file_type="original",
        storage_path=valid_storage_path,
        original_filename="valid.png",
        file_size=len(png_bytes),
        mime_type="image/png",
    )
    for index in range(60):
        invalid_media_id = _create_test_image_media(
            media_db,
            title=f"New invalid image {index}",
            content="image content",
            content_hash=f"image-hash-invalid-{index}",
        )
        media_db.insert_media_file(
            media_id=invalid_media_id,
            file_type="original",
            storage_path=f"777/media/{invalid_media_id}/missing-{index}.png",
            original_filename=f"missing-{index}.png",
            file_size=len(png_bytes),
            mime_type="image/png",
        )

    candidates = await list_reference_image_candidates(media_db, user_id="777", limit=1, storage=storage)

    assert [(candidate.file_id, candidate.title) for candidate in candidates] == [(1, "Old valid image")]


@pytest.mark.asyncio
async def test_list_reference_image_candidates_skips_oversized_rows_before_storage(
    monkeypatch,
    media_db: MediaDatabase,
) -> None:
    monkeypatch.setattr(
        reference_images_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(inline_max_bytes=3),
        raising=False,
    )

    media_id = _create_test_image_media(
        media_db,
        title="Oversized image",
        content="image content",
        content_hash="image-hash-oversized",
    )
    media_db.insert_media_file(
        media_id=media_id,
        file_type="original",
        storage_path=f"777/media/{media_id}/oversized.png",
        original_filename="oversized.png",
        file_size=10,
        mime_type="image/png",
    )

    candidates = await list_reference_image_candidates(media_db, user_id="777", storage=MustNotReadStorage())

    assert candidates == []
