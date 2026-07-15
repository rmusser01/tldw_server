import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase

pytestmark = pytest.mark.unit


@pytest.fixture()
def collections_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CollectionsDatabase:
    base_dir = tmp_path / "user_dbs"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "USER_DB_BASE_DIR", str(base_dir), raising=False)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    db = CollectionsDatabase.for_user(user_id=8042)
    try:
        yield db
    finally:
        db.close()


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


def test_collection_item_ordinals_are_unique_within_collection(
    collections_db: CollectionsDatabase,
) -> None:
    collection = collections_db.create_media_collection(
        name="Conference playlist",
        kind="conference",
    )
    first = collections_db.add_media_collection_item(
        collection_id=collection.id,
        source_url="https://example.com/talk-1",
        ordinal=1,
        status="planned",
    )
    second = collections_db.add_media_collection_item(
        collection_id=collection.id,
        source_url="https://example.com/talk-2",
        ordinal=2,
        status="planned",
    )

    with pytest.raises(ValueError, match="media_collection_item_ordinal_duplicate"):
        collections_db.add_media_collection_item(
            collection_id=collection.id,
            source_url="https://example.com/talk-duplicate",
            ordinal=1,
            status="planned",
        )

    with pytest.raises(ValueError, match="media_collection_item_ordinal_duplicate"):
        collections_db.update_media_collection_item(second.id, ordinal=first.ordinal)


def test_resolving_collection_item_clears_stale_failure_metadata(
    collections_db: CollectionsDatabase,
) -> None:
    collection = collections_db.create_media_collection(
        name="Conference playlist",
        kind="conference",
    )
    planned = collections_db.add_media_collection_item(
        collection_id=collection.id,
        source_url="https://example.com/talk-failed-once",
        ordinal=1,
        status="planned",
    )
    collections_db.update_media_collection_item_status(
        planned.id,
        status="failed",
        error_summary="private video",
        warnings=["download failed"],
        latest_job_id="job-failed",
    )

    resolved = collections_db.resolve_media_collection_item(
        planned.id,
        media_id=123,
        content_item_id=456,
        status="completed",
        latest_job_id="job-retry",
    )

    assert resolved.status == "completed"
    assert resolved.error_summary is None
    assert resolved.warnings == []
    assert resolved.latest_job_id == "job-retry"


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


def test_create_media_collection_with_items_is_one_atomic_bulk_operation(
    collections_db: CollectionsDatabase,
) -> None:
    created = collections_db.create_media_collection_with_items(
        name="Playlist plan",
        kind="playlist_ingest",
        source_url="https://www.youtube.com/playlist?list=PLatomic",
        items=[
            {
                "source_url": "https://www.youtube.com/watch?v=one",
                "normalized_source_id": "youtube:video:one",
                "source_kind": "youtube_video",
                "ordinal": 1,
                "title": "One",
            },
            {
                "source_url": "https://www.youtube.com/watch?v=two",
                "normalized_source_id": "youtube:video:two",
                "source_kind": "youtube_video",
                "ordinal": 2,
                "title": "Two",
            },
        ],
    )

    assert created.name == "Playlist plan"
    assert [item.ordinal for item in created.items] == [1, 2]
    assert [item.status for item in created.items] == ["planned", "planned"]
    assert [item.title for item in created.items] == ["One", "Two"]


def test_create_media_collection_with_items_reads_result_before_commit(
    collections_db: CollectionsDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collections_db,
        "get_media_collection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("post-commit readback")),
    )

    created = collections_db.create_media_collection_with_items(
        name="Transactional readback",
        kind="playlist_ingest",
        items=[{"source_url": "https://example.com/one", "ordinal": 1}],
    )

    assert created.name == "Transactional readback"
    assert [item.status for item in created.items] == ["planned"]


def test_playlist_collection_can_be_reconciled_by_internal_run_marker(
    collections_db: CollectionsDatabase,
) -> None:
    created = collections_db.create_media_collection_with_items(
        name="Reconciled plan",
        kind="playlist_ingest",
        metadata={"playlist_ingest_run_id": "run-123"},
        items=[{"source_url": "https://example.com/one", "ordinal": 1}],
    )

    reconciled = collections_db.get_playlist_ingest_collection_for_run("run-123")

    assert reconciled.id == created.id
    assert [item.id for item in reconciled.items] == [created.items[0].id]


def test_playlist_collection_reconciliation_reports_missing_marker_as_not_found(
    collections_db: CollectionsDatabase,
) -> None:
    with pytest.raises(KeyError, match="media_collection_not_found"):
        collections_db.get_playlist_ingest_collection_for_run("missing-run")


def test_playlist_collection_reconciliation_rejects_ambiguous_exact_markers(
    collections_db: CollectionsDatabase,
) -> None:
    for name in ("First plan", "Second plan"):
        collections_db.create_media_collection_with_items(
            name=name,
            kind="playlist_ingest",
            metadata={"playlist_ingest_run_id": "duplicate-run"},
            items=[{"source_url": "https://example.com/one", "ordinal": 1}],
        )

    with pytest.raises(DatabaseError) as exc_info:
        collections_db.get_playlist_ingest_collection_for_run("duplicate-run")

    assert str(exc_info.value) == "playlist_ingest_collection_marker_ambiguous"


def test_create_media_collection_with_items_rolls_back_collection_and_memberships(
    collections_db: CollectionsDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_execute = collections_db.backend.execute
    item_inserts = 0

    def fail_second_item(query, params=(), connection=None):
        nonlocal item_inserts
        if "INSERT INTO media_collection_items" in query:
            item_inserts += 1
            if item_inserts == 2:
                raise RuntimeError("synthetic item insert failure")
        return original_execute(query, params, connection=connection)

    monkeypatch.setattr(collections_db.backend, "execute", fail_second_item)

    with pytest.raises(RuntimeError, match="synthetic item insert failure"):
        collections_db.create_media_collection_with_items(
            name="Must roll back",
            kind="playlist_ingest",
            items=[
                {"source_url": "https://example.com/one", "ordinal": 1},
                {"source_url": "https://example.com/two", "ordinal": 2},
            ],
        )

    monkeypatch.setattr(collections_db.backend, "execute", original_execute)
    collections, total = collections_db.list_media_collections(kind="playlist_ingest")
    assert total == 0
    assert collections == []


def test_playlist_collection_claim_transfers_initialization_token_atomically(
    collections_db: CollectionsDatabase,
) -> None:
    created = collections_db.create_media_collection_with_items(
        name="Claimable plan",
        kind="playlist_ingest",
        metadata={
            "playlist_ingest_run_id": "run-claim",
            "playlist_ingest_initialization_token": "token-a",
        },
        items=[{"source_url": "https://example.com/one", "ordinal": 1}],
    )
    item_ids = [item.id for item in created.items]

    claimed = collections_db.claim_playlist_ingest_collection(
        created.id,
        run_id="run-claim",
        initialization_token="token-b",
        expected_item_ids=item_ids,
    )

    assert claimed.metadata["playlist_ingest_run_id"] == "run-claim"
    assert claimed.metadata["playlist_ingest_initialization_token"] == "token-b"
    assert [item.id for item in claimed.items] == item_ids

    with pytest.raises(ValueError, match="playlist_ingest_initialization_token_invalid"):
        collections_db.claim_playlist_ingest_collection(
            created.id,
            run_id="run-claim",
            initialization_token="x" * 256,
            expected_item_ids=item_ids,
        )


def test_playlist_collection_claim_rejects_owner_and_run_mismatch(
    collections_db: CollectionsDatabase,
) -> None:
    created = collections_db.create_media_collection_with_items(
        name="Owner-scoped plan",
        kind="playlist_ingest",
        metadata={
            "playlist_ingest_run_id": "run-owner",
            "playlist_ingest_initialization_token": "token-owner",
        },
        items=[{"source_url": "https://example.com/one", "ordinal": 1}],
    )
    item_ids = [item.id for item in created.items]

    with pytest.raises(ValueError, match="media_collection_claim_mismatch"):
        collections_db.claim_playlist_ingest_collection(
            created.id,
            run_id="other-run",
            initialization_token="token-b",
            expected_item_ids=item_ids,
        )

    other_owner = CollectionsDatabase.from_backend(user_id="other-owner", backend=collections_db.backend)
    with pytest.raises(KeyError, match="media_collection_not_found"):
        other_owner.claim_playlist_ingest_collection(
            created.id,
            run_id="run-owner",
            initialization_token="token-b",
            expected_item_ids=item_ids,
        )


def test_discard_media_collection_rejects_wrong_ownership_token(
    collections_db: CollectionsDatabase,
) -> None:
    created = collections_db.create_media_collection_with_items(
        name="Owned plan",
        kind="playlist_ingest",
        metadata={
            "playlist_ingest_run_id": "run-discard",
            "playlist_ingest_initialization_token": "token-current",
        },
        items=[{"source_url": "https://example.com/one", "ordinal": 1}],
    )

    with pytest.raises(ValueError, match="media_collection_discard_mismatch"):
        collections_db.discard_media_collection(
            created.id,
            expected_item_ids=[created.items[0].id],
            expected_run_id="run-discard",
            expected_initialization_token="token-stale",
        )

    assert collections_db.get_media_collection(created.id).id == created.id


def test_discard_media_collection_removes_just_created_plan_and_memberships(
    collections_db: CollectionsDatabase,
) -> None:
    created = collections_db.create_media_collection_with_items(
        name="Compensated plan",
        kind="playlist_ingest",
        metadata={
            "playlist_ingest_run_id": "run-discard",
            "playlist_ingest_initialization_token": "token-current",
        },
        items=[
            {"source_url": "https://example.com/one", "ordinal": 1},
            {"source_url": "https://example.com/two", "ordinal": 2},
        ],
    )
    expected_item_ids = [item.id for item in created.items]

    with pytest.raises(ValueError, match="media_collection_discard_mismatch"):
        collections_db.discard_media_collection(
            True,
            expected_item_ids=expected_item_ids,
            expected_run_id="run-discard",
            expected_initialization_token="token-current",
        )

    with pytest.raises(ValueError, match="media_collection_discard_mismatch"):
        collections_db.discard_media_collection(
            created.id,
            expected_item_ids=[created.items[0].id],
            expected_run_id="run-discard",
            expected_initialization_token="token-current",
        )
    assert collections_db.get_media_collection(created.id).id == created.id

    assert (
        collections_db.discard_media_collection(
            created.id,
            expected_item_ids=expected_item_ids,
            expected_run_id="run-discard",
            expected_initialization_token="token-current",
        )
        is True
    )

    collections, total = collections_db.list_media_collections(kind="playlist_ingest")
    assert total == 0
    assert collections == []
    assert (
        collections_db.backend.execute(
            "SELECT COUNT(*) AS total FROM media_collection_items WHERE collection_id = ?",
            (created.id,),
        ).first["total"]
        == 0
    )
    assert (
        collections_db.backend.execute(
            "SELECT COUNT(*) AS total FROM media_collections WHERE id = ?",
            (created.id,),
        ).first["total"]
        == 0
    )


def test_restore_media_collection_item_plan_requires_exact_resolved_write(
    collections_db: CollectionsDatabase,
) -> None:
    created = collections_db.create_media_collection_with_items(
        name="Compensated action",
        kind="playlist_ingest",
        items=[{"source_url": "https://example.com/one", "ordinal": 1}],
    )
    resolved = collections_db.resolve_media_collection_item(
        created.items[0].id,
        media_id=17,
        status="completed",
    )

    with pytest.raises(ValueError, match="media_collection_restore_mismatch"):
        collections_db.restore_media_collection_item_plan(
            resolved.id,
            expected_media_id=17,
            expected_status="completed",
            expected_updated_at="wrong-write-token",
        )
    unchanged = collections_db.get_media_collection_item(resolved.id)
    assert unchanged.status == "completed"
    assert unchanged.media_id == 17

    restored = collections_db.restore_media_collection_item_plan(
        resolved.id,
        expected_media_id=17,
        expected_status="completed",
        expected_updated_at=resolved.updated_at,
    )
    assert restored.status == "planned"
    assert restored.media_id is None
    repeated = collections_db.restore_media_collection_item_plan(
        resolved.id,
        expected_media_id=17,
        expected_status="completed",
        expected_updated_at=resolved.updated_at,
    )
    assert repeated.updated_at == restored.updated_at


@pytest.mark.parametrize(
    ("item_id", "media_id"),
    [
        (True, 17),
        ("1", 17),
        (1, True),
        (1, "17"),
    ],
)
def test_resolve_media_collection_item_rejects_coerced_ids_without_write(
    collections_db: CollectionsDatabase,
    item_id,
    media_id,
) -> None:
    created = collections_db.create_media_collection_with_items(
        name="Strict identifiers",
        kind="playlist_ingest",
        items=[{"source_url": "https://example.com/one", "ordinal": 1}],
    )
    actual_item_id = created.items[0].id if item_id == 1 and type(item_id) is int else item_id

    with pytest.raises(ValueError, match="media_collection_resolve_mismatch"):
        collections_db.resolve_media_collection_item(actual_item_id, media_id=media_id)

    unchanged = collections_db.get_media_collection_item(created.items[0].id)
    assert unchanged.status == "planned"
    assert unchanged.media_id is None


def test_resolve_media_collection_item_is_idempotent_only_for_exact_result(
    collections_db: CollectionsDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = collections_db.create_media_collection_with_items(
        name="Idempotent resolution",
        kind="playlist_ingest",
        items=[{"source_url": "https://example.com/one", "ordinal": 1}],
    )
    timestamps = iter(["2026-07-12T12:00:01+00:00", "2026-07-12T12:00:02+00:00"])
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.Collections_DB._utcnow_iso",
        lambda: next(timestamps),
    )

    resolved = collections_db.resolve_media_collection_item(
        created.items[0].id,
        media_id=17,
        status="completed",
    )
    repeated = collections_db.resolve_media_collection_item(
        created.items[0].id,
        media_id=17,
        status="completed",
    )

    assert repeated.updated_at == resolved.updated_at
    with pytest.raises(ValueError, match="media_collection_resolve_mismatch"):
        collections_db.resolve_media_collection_item(
            created.items[0].id,
            media_id=18,
            status="completed",
        )


def test_resolve_media_collection_item_normalizes_job_id_before_exact_retry(
    collections_db: CollectionsDatabase,
) -> None:
    created = collections_db.create_media_collection_with_items(
        name="Canonical job identity",
        kind="playlist_ingest",
        items=[{"source_url": "https://example.com/one", "ordinal": 1}],
    )

    resolved = collections_db.resolve_media_collection_item(
        created.items[0].id,
        media_id=17,
        status="completed",
        latest_job_id=" job-17 ",
    )
    repeated = collections_db.resolve_media_collection_item(
        created.items[0].id,
        media_id=17,
        status="completed",
        latest_job_id=" job-17 ",
    )

    assert repeated.latest_job_id == "job-17"
    assert repeated.updated_at == resolved.updated_at


def test_playlist_resolution_rejects_reserved_idempotency_identity(
    collections_db: CollectionsDatabase,
) -> None:
    created = collections_db.create_media_collection_with_items(
        name="Reserved identity",
        kind="playlist_ingest",
        items=[{"source_url": "https://example.com/one", "ordinal": 1}],
    )
    collections_db.backend.execute(
        "UPDATE media_collection_items SET idempotency_key = ? WHERE id = ? AND user_id = ?",
        ("reserved", created.items[0].id, collections_db.user_id),
    )

    with pytest.raises(ValueError, match="media_collection_resolve_mismatch"):
        collections_db.resolve_media_collection_item(created.items[0].id, media_id=17)

    unchanged = collections_db.get_media_collection_item(created.items[0].id)
    assert unchanged.status == "planned"
    assert unchanged.media_id is None


def test_concurrent_collection_resolution_has_one_exact_result(
    collections_db: CollectionsDatabase,
) -> None:
    created = collections_db.create_media_collection_with_items(
        name="Concurrent resolution",
        kind="playlist_ingest",
        items=[{"source_url": "https://example.com/one", "ordinal": 1}],
    )
    item_id = created.items[0].id

    def resolve(media_id: int):
        try:
            return collections_db.resolve_media_collection_item(item_id, media_id=media_id)
        except ValueError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(resolve, (17, 18)))

    persisted = collections_db.get_media_collection_item(item_id)
    successful = [result for result in results if not isinstance(result, Exception)]
    rejected = [result for result in results if isinstance(result, ValueError)]
    assert len(successful) == 1
    assert len(rejected) == 1
    assert persisted.media_id in {17, 18}
    assert successful[0].media_id == persisted.media_id


def test_resolve_media_collection_item_rolls_back_when_result_cannot_be_read(
    collections_db: CollectionsDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = collections_db.create_media_collection_with_items(
        name="Atomic action",
        kind="playlist_ingest",
        items=[{"source_url": "https://example.com/one", "ordinal": 1}],
    )
    original_execute = collections_db.backend.execute

    def fail_result_read(query, params=None, connection=None):
        if connection is not None and "SELECT id, user_id, collection_id, ordinal" in query:
            raise RuntimeError("result read failed")
        return original_execute(query, params, connection=connection)

    monkeypatch.setattr(collections_db.backend, "execute", fail_result_read)
    with pytest.raises(RuntimeError, match="result read failed"):
        collections_db.resolve_media_collection_item(
            created.items[0].id,
            media_id=17,
            status="completed",
        )
    monkeypatch.setattr(collections_db.backend, "execute", original_execute)

    unchanged = collections_db.get_media_collection_item(created.items[0].id)
    assert unchanged.status == "planned"
    assert unchanged.media_id is None


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
