"""MediaFiles must preserve its contract on SQLite and official PostgreSQL."""

from types import SimpleNamespace

import pytest
from fastapi import Response
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.media import item as item_endpoint
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_files_repository import (
    MediaFilesRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.services.media_details_service import (
    get_full_media_details_rich,
)

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "postgresql"])
def media_db(request):
    backend = None
    if request.param == "postgresql":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = MediaDatabase(":memory:", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_connection()
        if backend is not None:
            backend.get_pool().close_all()


def add_media(db, suffix="one"):
    media_id, _, _ = db.add_media_with_keywords(
        title=f"Synthetic MediaFiles {suffix}", content=f"Public synthetic source {suffix}.",
        media_type="document", url=f"file:///synthetic-{suffix}.txt", keywords=[],
    )
    return media_id


@pytest.mark.asyncio
async def test_rich_detail_without_original_preserves_content(media_db, monkeypatch):
    media_id = add_media(media_db)
    assert media_db.get_media_file(media_id, "original") is None
    detail = get_full_media_details_rich(media_db, media_id)
    assert detail["content"]["text"] == "Public synthetic source one."
    assert detail["has_original_file"] is False
    assert detail["original_file_url"] is None
    monkeypatch.setattr(item_endpoint, "_is_test_mode", lambda: False)
    response = Response()
    payload = await item_endpoint.get_media_item(
        request=Request({"type": "http", "method": "GET", "path": f"/api/v1/media/{media_id}", "headers": []}),
        response=response, media_id=media_id, db=media_db,
        current_user=SimpleNamespace(id="1"), include_content=True,
        include_versions=True, include_version_content=False, if_none_match=None,
    )
    assert response.status_code == 200
    assert payload["content"]["text"] == "Public synthetic source one."
    assert payload["has_original_file"] is False
    assert response.headers["etag"]


def test_registration_latest_type_deletion_and_rich_detail(media_db):
    media_id = add_media(media_db)
    repo = MediaFilesRepository(media_db)
    original_uuid = repo.insert(media_id, "original", "older.txt", original_filename="don't:bind?.txt")
    latest_uuid = repo.insert(media_id, "original", "latest.txt", file_size=42, mime_type="text/plain")
    repo.insert(media_id, "thumbnail", "thumb.png")
    records = repo.list_for_media(media_id)
    assert [row["file_type"] for row in records] == ["original", "original", "thumbnail"]
    assert records[0]["uuid"] == original_uuid
    assert records[0]["original_filename"] == "don't:bind?.txt"
    latest = repo.get_for_media(media_id)
    assert latest["uuid"] == latest_uuid
    assert latest["file_size"] == 42
    assert get_full_media_details_rich(media_db, media_id)["has_original_file"] is True
    repo.soft_delete(latest["id"])
    assert repo.get_for_media(media_id)["uuid"] == original_uuid
    assert repo.get_for_media(media_id, include_deleted=True)["uuid"] == latest_uuid
    assert len(repo.list_for_media(media_id)) == 2
    assert len(repo.list_for_media(media_id, include_deleted=True)) == 3
    repo.soft_delete(latest["id"], hard_delete=True)
    repo.soft_delete(latest["id"], hard_delete=True)
    assert len(repo.list_for_media(media_id, include_deleted=True)) == 2


def test_bulk_delete_is_scoped_and_retained_references_are_preserved(media_db):
    first, second = add_media(media_db), add_media(media_db, "two")
    repo = MediaFilesRepository(media_db)
    repo.insert(first, "original", "shared.txt")
    first_old = repo.get_for_media(first)
    assert repo.has_retained_references("shared.txt", set()) is True
    assert repo.has_retained_references("shared.txt", {first_old["id"]}) is False
    repo.insert(first, "original", "replacement.txt")
    assert repo.has_retained_references("shared.txt", set()) is False
    repo.insert(second, "original", "shared.txt")
    second_file = repo.get_for_media(second)
    assert repo.has_retained_references("shared.txt", {first_old["id"]}) is True
    repo.soft_delete_for_media(first)
    assert repo.list_for_media(first) == []
    assert all(row["deleted"] for row in repo.list_for_media(first, include_deleted=True))
    assert repo.get_for_media(second)["uuid"] == second_file["uuid"]
    repo.soft_delete_for_media(first, hard_delete=True)
    assert repo.list_for_media(first, include_deleted=True) == []
    assert repo.get_for_media(second)["uuid"] == second_file["uuid"]


def test_caller_rollback_preserves_original_registration(media_db):
    media_id = add_media(media_db)
    repo = MediaFilesRepository(media_db)
    original_uuid = repo.insert(media_id, "original", "original.txt")
    original_id = repo.get_for_media(media_id)["id"]
    with pytest.raises(RuntimeError, match="rollback requested"):
        with media_db.transaction():
            repo.insert(media_id, "original", "uncommitted.txt")
            repo.soft_delete(original_id)
            raise RuntimeError("rollback requested")
    rows = repo.list_for_media(media_id, include_deleted=True)
    assert [(row["uuid"], bool(row["deleted"])) for row in rows] == [(original_uuid, False)]


def test_database_query_failure_is_not_reported_as_missing_file(media_db, monkeypatch):
    repo = MediaFilesRepository(media_db)

    def fail_query(*_args, **_kwargs):
        raise DatabaseError("controlled query failure")

    monkeypatch.setattr(media_db, "_fetchall_with_connection", fail_query)
    with pytest.raises(DatabaseError, match="controlled query failure"):
        repo.get_for_media(1)
