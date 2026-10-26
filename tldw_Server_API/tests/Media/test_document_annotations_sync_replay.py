"""Replay safety for POST /media/{id}/annotations/sync (idempotent on client_id)."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints.media import document_annotations as annotations_endpoint
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.repositories.document_workspace_repository import (
    DocumentWorkspaceRepository,
)


@pytest.fixture(params=["sqlite", "postgres"])
def media_db(request):
    if request.param == "sqlite":
        db = MediaDatabase(db_path=":memory:", client_id="annotation-sync-replay")
    else:
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        db = MediaDatabase(db_path=":memory:", client_id="annotation-sync-replay-pg", backend=backend)
    try:
        yield db
    finally:
        db.close_connection()


def _user(user_id: int) -> object:
    return type("User", (), {"id": user_id})()


def _batch() -> annotations_endpoint.AnnotationSyncRequest:
    return annotations_endpoint.AnnotationSyncRequest(
        annotations=[
            annotations_endpoint.AnnotationCreate(location="page-1", text="First"),
            annotations_endpoint.AnnotationCreate(location="page-2", text="Second", note="n"),
        ],
        client_ids=["ann_client_1", "ann_client_2"],
    )


async def test_resent_sync_batch_creates_each_annotation_once(media_db) -> None:
    media_id, _, _ = media_db.add_media_with_keywords(title="Doc", media_type="text", content="abc", keywords=None)

    first = await annotations_endpoint.sync_annotations(
        media_id=media_id, body=_batch(), db=media_db, current_user=_user(1)
    )
    replay = await annotations_endpoint.sync_annotations(
        media_id=media_id, body=_batch(), db=media_db, current_user=_user(1)
    )

    assert set(first.id_mapping) == {"ann_client_1", "ann_client_2"}
    assert replay.id_mapping == first.id_mapping
    assert [a.id for a in replay.annotations] == [a.id for a in first.annotations]
    repo = DocumentWorkspaceRepository.from_media_db(media_db)
    stored = repo.list_annotations(media_id=media_id, user_id="1")
    assert sorted(row["id"] for row in stored) == sorted(first.id_mapping.values())

    # A replay after the annotation was deleted does not resurrect or duplicate it.
    assert repo.soft_delete_annotation(
        annotation_id=first.id_mapping["ann_client_1"], media_id=media_id, user_id="1", updated_at="2026-10-25T00:00:00+00:00"
    )
    await annotations_endpoint.sync_annotations(media_id=media_id, body=_batch(), db=media_db, current_user=_user(1))
    assert [row["id"] for row in repo.list_annotations(media_id=media_id, user_id="1")] == [
        first.id_mapping["ann_client_2"]
    ]

    # The same client ids from another user are a different scope.
    other = await annotations_endpoint.sync_annotations(
        media_id=media_id, body=_batch(), db=media_db, current_user=_user(2)
    )
    assert set(other.id_mapping.values()).isdisjoint(first.id_mapping.values())


async def test_sync_without_client_ids_still_inserts_every_time(media_db) -> None:
    media_id, _, _ = media_db.add_media_with_keywords(title="Doc", media_type="text", content="abc", keywords=None)
    body = annotations_endpoint.AnnotationSyncRequest(
        annotations=[annotations_endpoint.AnnotationCreate(location="page-1", text="No client id")],
    )

    await annotations_endpoint.sync_annotations(media_id=media_id, body=body, db=media_db, current_user=_user(1))
    await annotations_endpoint.sync_annotations(media_id=media_id, body=body, db=media_db, current_user=_user(1))

    stored = DocumentWorkspaceRepository.from_media_db(media_db).list_annotations(media_id=media_id, user_id="1")
    assert len(stored) == 2


def _row(annotation_id: str, client_id: str | None) -> dict:
    return {
        "id": annotation_id,
        "client_id": client_id,
        "location": "page-1",
        "text": "t",
        "color": "yellow",
        "annotation_type": "highlight",
        "created_at": "2026-10-25T00:00:00+00:00",
        "updated_at": "2026-10-25T00:00:00+00:00",
    }


def test_postgres_upgrade_keeps_legacy_rows_and_dedupes_new_syncs(pg_database_config) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="annotation-client-id-upgrade", backend=backend)
    try:
        # Recreate a pre-upgrade table holding rows that predate client_id.
        with backend.transaction() as conn:
            backend.execute("DROP INDEX idx_annotations_client_id", connection=conn)
            backend.execute("ALTER TABLE document_annotations DROP COLUMN client_id", connection=conn)
            for legacy_id in ("ann_legacy_1", "ann_legacy_2"):
                backend.execute(
                    "INSERT INTO document_annotations (id, media_id, user_id, location, text, created_at, updated_at)"
                    " VALUES (%s, 1, '1', 'p', 't', 't', 't')",
                    (legacy_id,),
                    connection=conn,
                )

        db._initialize_schema()

        repo = DocumentWorkspaceRepository.from_media_db(db)
        repo.sync_annotations(media_id=1, user_id="1", annotation_rows=[_row("ann_new_a", "client-x")])
        repo.sync_annotations(media_id=1, user_id="1", annotation_rows=[_row("ann_new_b", "client-x")])
        assert sorted(row["id"] for row in repo.list_annotations(media_id=1, user_id="1")) == [
            "ann_legacy_1",
            "ann_legacy_2",
            "ann_new_a",
        ]
    finally:
        db.close_connection()
