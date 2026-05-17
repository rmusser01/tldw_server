import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase


pytestmark = pytest.mark.unit


@pytest.fixture()
def collections_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CollectionsDatabase:
    base_dir = tmp_path / "user_dbs"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    db = CollectionsDatabase.for_user(user_id=8042)
    try:
        yield db
    finally:
        db.close()
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def test_conference_collection_persists_planned_and_resolved_items(
    collections_db: CollectionsDatabase,
) -> None:
    collection = collections_db.create_media_collection(
        name="PyCon 2026",
        kind="conference",
        description="Talks to ingest after the conference.",
        source_url="https://www.youtube.com/playlist?list=PLpycon2026",
        metadata={"conference_name": "PyCon", "event_year": "2026"},
        default_tags=["pycon", "python"],
    )
    planned = collections_db.add_media_collection_item(
        collection_id=collection.id,
        source_url="https://www.youtube.com/watch?v=opening",
        normalized_source_id="youtube:video:opening",
        source_kind="youtube_video",
        ordinal=1,
        title="Opening Keynote",
        speaker="Ada Lovelace",
        published_at="2026-05-01",
        status="planned",
        tags=["keynote"],
        metadata={"track": "Main"},
    )

    collections_db.resolve_media_collection_item(
        planned.id,
        media_id=123,
        content_item_id=456,
        status="completed",
        latest_job_id="job-123",
    )

    loaded = collections_db.get_media_collection(collection.id)

    assert loaded.id == collection.id
    assert loaded.name == "PyCon 2026"
    assert loaded.source_url == "https://www.youtube.com/playlist?list=PLpycon2026"
    assert loaded.metadata["conference_name"] == "PyCon"
    assert loaded.default_tags == ["pycon", "python"]
    assert len(loaded.items) == 1
    assert loaded.items[0].ordinal == 1
    assert loaded.items[0].source_url == "https://www.youtube.com/watch?v=opening"
    assert loaded.items[0].title == "Opening Keynote"
    assert loaded.items[0].speaker == "Ada Lovelace"
    assert loaded.items[0].metadata["track"] == "Main"
    assert loaded.items[0].tags == ["keynote"]
    assert loaded.items[0].status == "completed"
    assert loaded.items[0].media_id == 123
    assert loaded.items[0].content_item_id == 456
    assert loaded.items[0].latest_job_id == "job-123"


def test_planned_collection_items_do_not_create_or_overwrite_content_items(
    collections_db: CollectionsDatabase,
) -> None:
    source_url = "https://www.youtube.com/watch?v=repeated"
    collection = collections_db.create_media_collection(
        name="Conference playlist",
        kind="conference",
    )

    first_planned = collections_db.add_media_collection_item(
        collection_id=collection.id,
        source_url=source_url,
        normalized_source_id="youtube:video:repeated",
        source_kind="youtube_video",
        title="Talk in the playlist",
        status="planned",
    )

    assert first_planned.id > 0
    assert collections_db.get_content_item_by_url(source_url) is None

    resolved_content = collections_db.upsert_content_item(
        origin="media",
        origin_type="video",
        origin_id=987,
        url=source_url,
        canonical_url=source_url,
        domain="youtube.com",
        title="Resolved media title",
        summary=None,
        content_hash=None,
        word_count=None,
        published_at=None,
        status="completed",
        media_id=987,
        tags=["resolved"],
    )

    second_planned = collections_db.add_media_collection_item(
        collection_id=collection.id,
        source_url=source_url,
        normalized_source_id="youtube:video:repeated",
        source_kind="youtube_video",
        title="Same talk appears twice in the plan",
        status="planned",
    )

    content_after_plans = collections_db.get_content_item_by_url(source_url)
    loaded = collections_db.get_media_collection(collection.id)

    assert content_after_plans is not None
    assert content_after_plans.id == resolved_content.id
    assert content_after_plans.title == "Resolved media title"
    assert [item.id for item in loaded.items] == [first_planned.id, second_planned.id]
    assert [item.content_item_id for item in loaded.items] == [None, None]


def test_media_collections_can_be_listed_updated_and_soft_deleted(
    collections_db: CollectionsDatabase,
) -> None:
    collections_db.create_media_collection(name="Other", kind="manual")
    conference = collections_db.create_media_collection(
        name="Original title",
        kind="conference",
        metadata={"conference_name": "Original"},
    )

    updated = collections_db.update_media_collection(
        conference.id,
        name="Updated title",
        metadata={"conference_name": "Updated"},
        default_tags=["conference"],
    )
    listed, total = collections_db.list_media_collections(kind="conference", page=1, size=20)

    assert updated.name == "Updated title"
    assert updated.metadata["conference_name"] == "Updated"
    assert updated.default_tags == ["conference"]
    assert total == 1
    assert [item.id for item in listed] == [conference.id]

    assert collections_db.delete_media_collection(conference.id) is True

    listed_after_delete, total_after_delete = collections_db.list_media_collections(
        kind="conference",
        page=1,
        size=20,
    )
    assert total_after_delete == 0
    assert listed_after_delete == []


def test_media_collections_router_exposes_collection_crud(
    collections_db: CollectionsDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-12345")

    from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import (
        get_collections_db_for_user,
    )
    from tldw_Server_API.app.api.v1.endpoints.media import collections as media_collections

    app = FastAPI()
    app.include_router(media_collections.router, prefix="/api/v1/media", tags=["media"])
    app.dependency_overrides[get_collections_db_for_user] = lambda: collections_db

    with TestClient(app) as client:
        create_response = client.post(
            "/api/v1/media/collections",
            json={
                "name": "Conference 2026",
                "kind": "conference",
                "source_url": "https://www.youtube.com/playlist?list=PLtest",
                "metadata": {"conference_name": "Conference", "event_year": "2026"},
                "default_tags": ["conference"],
            },
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert create_response.status_code == 201, create_response.text
        collection = create_response.json()
        collection_id = collection["id"]

        item_response = client.post(
            f"/api/v1/media/collections/{collection_id}/items",
            json={
                "source_url": "https://www.youtube.com/watch?v=abc123",
                "normalized_source_id": "youtube:video:abc123",
                "source_kind": "youtube_video",
                "ordinal": 1,
                "title": "Opening Keynote",
                "speaker": "Ada Lovelace",
                "status": "planned",
                "metadata": {"track": "Main"},
                "tags": ["keynote"],
            },
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert item_response.status_code == 201, item_response.text
        item_id = item_response.json()["id"]

        patch_response = client.patch(
            f"/api/v1/media/collections/{collection_id}/items/{item_id}",
            json={
                "status": "completed",
                "media_id": 321,
                "content_item_id": 654,
                "latest_job_id": "job-321",
            },
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert patch_response.status_code == 200, patch_response.text
        assert patch_response.json()["status"] == "completed"

        get_response = client.get(
            f"/api/v1/media/collections/{collection_id}",
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert get_response.status_code == 200, get_response.text
        loaded = get_response.json()
        assert loaded["metadata"]["conference_name"] == "Conference"
        assert loaded["items"][0]["media_id"] == 321
        assert loaded["items"][0]["content_item_id"] == 654
