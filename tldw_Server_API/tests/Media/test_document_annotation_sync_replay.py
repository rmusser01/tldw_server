"""Real SQLite persistence for annotation creation retries."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.media.document_annotations import sync_annotations
from tldw_Server_API.app.api.v1.schemas.document_annotations import AnnotationCreate, AnnotationSyncRequest
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.repositories.document_workspace_repository import (
    DocumentWorkspaceRepository,
)


@pytest.fixture
def document_db(tmp_path):
    db = MediaDatabase(db_path=tmp_path / "media.db", client_id="1")
    media_ids = []
    for index in (1, 2):
        media_id, _, _ = db.add_media_with_keywords(
            url=f"https://example.com/rowan-{index}.pdf",
            title=f"Rowan {index}",
            media_type="pdf",
            content=f"Selene {index}",
            owner_user_id=1,
        )
        media_ids.append(media_id)
    try:
        yield db, media_ids
    finally:
        db.close_connection()


def payload(text="Selene", client_ids=True):
    return AnnotationSyncRequest(
        annotations=[AnnotationCreate(location="1", text=text, note="Retain exact note")],
        client_ids=["fixed-client-intent"] if client_ids else None,
    )


@pytest.mark.asyncio
async def test_matching_client_intent_replays_one_saved_annotation(document_db):
    db, ids = document_db
    first = await sync_annotations(payload(), media_id=ids[0], db=db, current_user=SimpleNamespace(id=1))
    second = await sync_annotations(payload(), media_id=ids[0], db=db, current_user=SimpleNamespace(id=1))
    rows = DocumentWorkspaceRepository(db).list_annotations(media_id=ids[0], user_id="1")
    assert (second.id_mapping, len(rows), second.annotations) == (first.id_mapping, 1, first.annotations)  # nosec B101 # Lost-ACK replay must not duplicate or rewrite saved data.


@pytest.mark.asyncio
async def test_changed_client_intent_conflicts_without_overwrite(document_db):
    db, ids = document_db
    first = await sync_annotations(payload(), media_id=ids[0], db=db, current_user=SimpleNamespace(id=1))
    with pytest.raises(HTTPException) as error:
        await sync_annotations(payload("Changed intent"), media_id=ids[0], db=db, current_user=SimpleNamespace(id=1))
    rows = DocumentWorkspaceRepository(db).list_annotations(media_id=ids[0], user_id="1")
    assert (error.value.status_code, [row["text"] for row in rows], [row["id"] for row in rows]) == (
        409,
        ["Selene"],
        [first.annotations[0].id],
    )  # nosec B101 # Reused identity cannot silently overwrite prior content.


@pytest.mark.asyncio
async def test_client_intent_identity_is_scoped_to_document(document_db):
    db, ids = document_db
    first = await sync_annotations(payload(), media_id=ids[0], db=db, current_user=SimpleNamespace(id=1))
    second = await sync_annotations(payload(), media_id=ids[1], db=db, current_user=SimpleNamespace(id=1))
    assert first.annotations[0].id != second.annotations[0].id  # nosec B101 # Identical client labels in different documents remain distinct.


@pytest.mark.asyncio
async def test_legacy_batch_without_client_ids_keeps_new_creation_semantics(document_db):
    db, ids = document_db
    first = await sync_annotations(
        payload(client_ids=False), media_id=ids[0], db=db, current_user=SimpleNamespace(id=1)
    )
    second = await sync_annotations(
        payload(client_ids=False), media_id=ids[0], db=db, current_user=SimpleNamespace(id=1)
    )
    assert first.annotations[0].id != second.annotations[0].id  # nosec B101 # Existing callers without an intent identity remain compatible.


def test_replay_ack_returns_the_row_checked_inside_its_transaction(document_db, monkeypatch):
    from contextlib import contextmanager
    from datetime import datetime, timezone

    db, ids = document_db
    repo = DocumentWorkspaceRepository(db)
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "id": "ann_admitted",
        "location": "1",
        "text": "Selene",
        "color": "yellow",
        "note": "Checked note",
        "annotation_type": "highlight",
        "chapter_title": None,
        "percentage": None,
        "created_at": now,
        "updated_at": now,
    }
    transaction = db.transaction
    armed = True

    @contextmanager
    def changed_after_commit():
        nonlocal armed
        with transaction() as conn:
            yield conn
        if armed:
            armed = False
            repo.update_annotation(
                annotation_id=row["id"], media_id=ids[0], user_id="1", note="A later edit", updated_at=now
            )

    monkeypatch.setattr(db, "transaction", changed_after_commit)
    acknowledged = repo.sync_annotations(media_id=ids[0], user_id="1", annotation_rows=[row], allow_replay=True)
    assert acknowledged[0]["note"] == "Checked note"  # nosec B101 # ACK describes the checked transaction, not a subsequent writer.


def test_sync_client_id_rejects_a_lone_unicode_surrogate():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AnnotationSyncRequest(
            annotations=[AnnotationCreate(location="1", text="Selene")], client_ids=["invalid-\ud800"]
        )


@pytest.fixture(params=["sqlite", "postgres"])
def replay_store(request, tmp_path):
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    backend = request.getfixturevalue("pg_restricted_backend") if request.param == "postgres" else None
    path = str(tmp_path / "replay.db")
    from loguru import logger

    metadata = []

    def observe(message):
        extra = message.record["extra"]
        if extra.get("sqlstate"):
            metadata.append((str(extra.get("exception_type")), str(extra["sqlstate"])))

    sink = logger.add(observe, level="ERROR")
    try:
        with scoped_context(user_id=401, is_admin=False):
            db = MediaDatabase(db_path=path, client_id="401", backend=backend)
            media_id, _, _ = db.add_media_with_keywords(
                title="Rowan annotations", media_type="pdf", content="Selene", owner_user_id=401
            )
            try:
                yield db, media_id, backend, path
            finally:
                db.close_connection()
    finally:
        logger.remove(sink)
        if metadata:
            print("DOCUMENT_DB_ERROR_CLASSES=" + repr(metadata))


@pytest.mark.asyncio
async def test_replay_batch_conflict_rolls_back_new_rows_and_retains_input_order(replay_store):
    db, media_id, _, _ = replay_store
    user = SimpleNamespace(id=401)
    first = await sync_annotations(payload(), media_id=media_id, db=db, current_user=user)
    conflicting = AnnotationSyncRequest(
        annotations=[AnnotationCreate(location="2", text="New row"), AnnotationCreate(location="1", text="Changed")],
        client_ids=["new-intent", "fixed-client-intent"],
    )
    with pytest.raises(HTTPException) as failure:
        await sync_annotations(conflicting, media_id=media_id, db=db, current_user=user)
    rows = DocumentWorkspaceRepository(db).list_annotations(media_id=media_id, user_id="401")
    assert (failure.value.status_code, [row["id"] for row in rows]) == (409, [first.annotations[0].id])  # nosec B101 # A later conflict rolls back earlier batch inserts.
    matching = AnnotationSyncRequest(
        annotations=[AnnotationCreate(location="2", text="New row"), payload().annotations[0]],
        client_ids=["new-intent", "fixed-client-intent"],
    )
    saved = await sync_annotations(matching, media_id=media_id, db=db, current_user=user)
    assert [row.text for row in saved.annotations] == ["New row", "Selene"]  # nosec B101 # Matching replay and new inserts preserve the input order.


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["edit", "delete"])
async def test_replay_does_not_resurrect_or_overwrite_a_changed_saved_annotation(replay_store, action):
    from datetime import datetime, timezone

    db, media_id, _, _ = replay_store
    user = SimpleNamespace(id=401)
    first = await sync_annotations(payload(), media_id=media_id, db=db, current_user=user)
    repo = DocumentWorkspaceRepository(db)
    args = {
        "annotation_id": first.annotations[0].id,
        "media_id": media_id,
        "user_id": "401",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if action == "edit":
        repo.update_annotation(**args, note="Later saved note")
    else:
        repo.soft_delete_annotation(**args)
    before = repo.list_annotations(media_id=media_id, user_id="401")
    with pytest.raises(HTTPException) as failure:
        await sync_annotations(payload(), media_id=media_id, db=db, current_user=user)
    assert (failure.value.status_code, repo.list_annotations(media_id=media_id, user_id="401")) == (409, before)  # nosec B101 # Retry cannot undo a later edit or deletion.


def test_concurrent_matching_creation_has_one_persistent_identity(replay_store):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    db, media_id, backend, path = replay_store
    barrier = threading.Barrier(2)

    def save():
        with scoped_context(user_id=401, is_admin=False):
            own_db = MediaDatabase(db_path=path, client_id="401", backend=backend, existing_only=True)
            try:
                barrier.wait(timeout=10)
                return (
                    asyncio.run(
                        sync_annotations(payload(), media_id=media_id, db=own_db, current_user=SimpleNamespace(id=401))
                    )
                    .annotations[0]
                    .id
                )
            finally:
                own_db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as executor:
        saved_ids = list(executor.map(lambda _: save(), range(2)))
    rows = DocumentWorkspaceRepository(db).list_annotations(media_id=media_id, user_id="401")
    assert saved_ids == [rows[0]["id"], rows[0]["id"]] and len(rows) == 1  # nosec B101 # Concurrent lost-ACK retries create one canonical row.


@pytest.mark.parametrize("ids", [[], ["same", "same"], [" "], ["x" * 129]])
def test_http_invalid_creation_ids_do_not_write(document_db, ids):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.api.v1.endpoints.media import document_annotations as endpoints

    db, media_ids = document_db
    app = FastAPI()
    app.include_router(endpoints.router, prefix="/api/v1/media")
    app.dependency_overrides[endpoints.get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[endpoints.get_media_db_for_user] = lambda: db
    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/media/{media_ids[0]}/annotations/sync",
            json={"annotations": [{"location": "1", "text": "Selene"}], "client_ids": ids},
        )
    assert (
        response.status_code,
        DocumentWorkspaceRepository(db).list_annotations(media_id=media_ids[0], user_id="1"),
    ) == (422, [])  # nosec B101 # Wire validation precedes writes, including malformed intent labels.


@pytest.mark.asyncio
async def test_replay_preserves_an_existing_client_column_identity(document_db):
    db, ids = document_db
    annotation = payload().annotations[0].model_dump(mode="json")
    row = {
        **annotation, "id": "ann_existing_random_identity", "client_id": "fixed-client-intent",
        "created_at": "2026-10-25T00:00:00+00:00", "updated_at": "2026-10-25T00:00:00+00:00",
    }
    repo = DocumentWorkspaceRepository(db)
    repo.sync_annotations(media_id=ids[0], user_id="1", annotation_rows=[row])
    replay = await sync_annotations(payload(), media_id=ids[0], db=db, current_user=SimpleNamespace(id=1))
    assert replay.id_mapping == {"fixed-client-intent": row["id"]}  # nosec B101 -- Older random server IDs survive exact creation replay.
    assert len(repo.list_annotations(media_id=ids[0], user_id="1")) == 1  # nosec B101 -- Upgrade replay does not create a second annotation.
