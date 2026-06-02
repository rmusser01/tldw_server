from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.repositories.document_workspace_repository import (
    DocumentWorkspaceRepository,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _NoRawExecuteConn:
    def execute(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Repository must use MediaDatabase execution helpers")


class _CursorStub:
    def __init__(self, *, rowcount: int = 1) -> None:
        self.rowcount = rowcount
        self.lastrowid = 1

    def fetchone(self) -> dict[str, Any] | None:
        return None


class _PostgresLikeDb:
    def __init__(
        self,
        *,
        fetchone_results: list[dict[str, Any] | None] | None = None,
        fetchall_results: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self.backend_type = BackendType.POSTGRESQL
        self.conn = _NoRawExecuteConn()
        self.calls: list[tuple[str, str, Any]] = []
        self.fetchone_results = list(fetchone_results or [])
        self.fetchall_results = list(fetchall_results or [])

    @contextmanager
    def transaction(self):
        yield self.conn

    def _execute_with_connection(self, conn: Any, query: str, params: Any = None) -> _CursorStub:
        assert conn is self.conn
        self.calls.append(("execute", query, params))
        return _CursorStub(rowcount=1)

    def _fetchone_with_connection(self, conn: Any, query: str, params: Any = None) -> dict[str, Any] | None:
        assert conn is self.conn
        self.calls.append(("fetchone", query, params))
        if self.fetchone_results:
            return self.fetchone_results.pop(0)
        return None

    def _fetchall_with_connection(self, conn: Any, query: str, params: Any = None) -> list[dict[str, Any]]:
        assert conn is self.conn
        self.calls.append(("fetchall", query, params))
        if self.fetchall_results:
            return self.fetchall_results.pop(0)
        return []


def test_document_workspace_repository_round_trips_reading_progress() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="document-workspace-progress")
    repo = DocumentWorkspaceRepository.from_media_db(db)
    try:
        assert repo.get_reading_progress(media_id=7, user_id="1") is None

        saved = repo.upsert_reading_progress(
            media_id=7,
            user_id="1",
            current_page=12,
            total_pages=40,
            zoom_level=125,
            view_mode="continuous",
            cfi="epubcfi(/6/2)",
            percentage=30.5,
            last_read_at=_now(),
        )

        assert saved["media_id"] == 7
        assert saved["current_page"] == 12
        assert saved["view_mode"] == "continuous"
        assert saved["cfi"] == "epubcfi(/6/2)"
        assert saved["percentage"] == 30.5
        assert repo.get_reading_progress(media_id=7, user_id="1") == saved

        assert repo.delete_reading_progress(media_id=7, user_id="1") is True
        assert repo.get_reading_progress(media_id=7, user_id="1") is None
        assert repo.delete_reading_progress(media_id=7, user_id="1") is False
    finally:
        db.close_connection()


def test_document_workspace_repository_uses_backend_helpers_for_postgres_reading_progress() -> None:
    row = {
        "media_id": 7,
        "user_id": "1",
        "current_page": 2,
        "total_pages": 10,
        "zoom_level": 100,
        "view_mode": "single",
        "cfi": None,
        "percentage": None,
        "last_read_at": _now(),
    }
    db = _PostgresLikeDb(fetchone_results=[row, row])
    repo = DocumentWorkspaceRepository.from_media_db(db)

    assert repo.get_reading_progress(media_id=7, user_id="1") == row
    saved = repo.upsert_reading_progress(
        media_id=7,
        user_id="1",
        current_page=2,
        total_pages=10,
        zoom_level=100,
        view_mode="single",
        cfi=None,
        percentage=None,
        last_read_at=row["last_read_at"],
    )

    assert saved == row
    assert db.calls[0][0] == "fetchone"
    assert db.calls[1][0] == "execute"
    assert "ON CONFLICT (media_id, user_id) DO UPDATE" in db.calls[1][1]
    assert "INSERT OR REPLACE" not in db.calls[1][1]
    assert db.calls[2][0] == "fetchone"


def test_document_workspace_repository_manages_annotation_lifecycle() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="document-workspace-annotations")
    repo = DocumentWorkspaceRepository.from_media_db(db)
    created_at = _now()
    updated_at = _now()
    try:
        created = repo.create_annotation(
            annotation_id="ann_repo_1",
            media_id=9,
            user_id="1",
            location="page-2",
            text="Important text",
            color="yellow",
            note=None,
            annotation_type="highlight",
            chapter_title="Chapter 1",
            percentage=20.0,
            created_at=created_at,
            updated_at=updated_at,
        )

        assert created["id"] == "ann_repo_1"
        assert created["chapter_title"] == "Chapter 1"
        assert repo.list_annotations(media_id=9, user_id="1") == [created]
        assert repo.get_annotation(annotation_id="ann_repo_1", media_id=9, user_id="1") == created

        changed = repo.update_annotation(
            annotation_id="ann_repo_1",
            media_id=9,
            user_id="1",
            text="Updated text",
            color="blue",
            note="review",
            updated_at=_now(),
        )

        assert changed is not None
        assert changed["text"] == "Updated text"
        assert changed["color"] == "blue"
        assert changed["note"] == "review"

        unchanged = repo.update_annotation(
            annotation_id="ann_repo_1",
            media_id=9,
            user_id="1",
            updated_at=_now(),
        )
        assert unchanged == changed

        assert repo.soft_delete_annotation(annotation_id="ann_repo_1", media_id=9, user_id="1", updated_at=_now())
        assert repo.list_annotations(media_id=9, user_id="1") == []
        assert not repo.soft_delete_annotation(annotation_id="ann_repo_1", media_id=9, user_id="1", updated_at=_now())
    finally:
        db.close_connection()


def test_document_workspace_repository_uses_backend_helpers_for_postgres_annotations() -> None:
    now = _now()
    row = {
        "id": "ann_pg",
        "location": "page-1",
        "text": "Important",
        "color": "yellow",
        "note": None,
        "annotation_type": "highlight",
        "chapter_title": None,
        "percentage": None,
        "created_at": now,
        "updated_at": now,
    }
    db = _PostgresLikeDb(
        fetchall_results=[[row]],
        fetchone_results=[row, row, row, row, row],
    )
    repo = DocumentWorkspaceRepository.from_media_db(db)

    assert repo.list_annotations(media_id=7, user_id="1") == [row]
    assert repo.create_annotation(
        annotation_id="ann_pg",
        media_id=7,
        user_id="1",
        location="page-1",
        text="Important",
        color="yellow",
        note=None,
        annotation_type="highlight",
        chapter_title=None,
        percentage=None,
        created_at=now,
        updated_at=now,
    ) == row
    assert repo.update_annotation(
        annotation_id="ann_pg",
        media_id=7,
        user_id="1",
        text="Changed",
        updated_at=now,
    ) == row
    assert repo.soft_delete_annotation(
        annotation_id="ann_pg",
        media_id=7,
        user_id="1",
        updated_at=now,
    ) is True
    assert repo.sync_annotations(
        media_id=7,
        user_id="1",
        annotation_rows=[
            {
                "id": "ann_pg",
                "location": "page-1",
                "text": "Important",
                "color": "yellow",
                "note": None,
                "annotation_type": "highlight",
                "chapter_title": None,
                "percentage": None,
                "created_at": now,
                "updated_at": now,
            }
        ],
    ) == [row]

    call_kinds = [call[0] for call in db.calls]
    assert call_kinds == [
        "fetchall",
        "execute",
        "fetchone",
        "fetchone",
        "execute",
        "fetchone",
        "execute",
        "execute",
        "fetchone",
    ]


def test_document_workspace_repository_syncs_annotation_rows() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="document-workspace-sync")
    repo = DocumentWorkspaceRepository.from_media_db(db)
    now = _now()
    try:
        synced = repo.sync_annotations(
            media_id=11,
            user_id="1",
            annotation_rows=[
                {
                    "id": "ann_sync_1",
                    "location": "page-1",
                    "text": "First",
                    "color": "green",
                    "note": None,
                    "annotation_type": "highlight",
                    "chapter_title": None,
                    "percentage": None,
                    "created_at": now,
                    "updated_at": now,
                },
                {
                    "id": "ann_sync_2",
                    "location": "page-2",
                    "text": "Second",
                    "color": "pink",
                    "note": "note",
                    "annotation_type": "page_note",
                    "chapter_title": "Second Chapter",
                    "percentage": 50.0,
                    "created_at": now,
                    "updated_at": now,
                },
            ],
        )

        assert [row["id"] for row in synced] == ["ann_sync_1", "ann_sync_2"]
        assert [row["id"] for row in repo.list_annotations(media_id=11, user_id="1")] == [
            "ann_sync_1",
            "ann_sync_2",
        ]
    finally:
        db.close_connection()


def test_document_workspace_repository_replaces_parsed_reference_cache_per_parser_version() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="document-workspace-references")
    repo = DocumentWorkspaceRepository.from_media_db(db)
    try:
        assert (
            repo.get_parsed_references_cache(
                media_id=5,
                user_id="1",
                parser_version="11",
                content_hash="old",
            )
            is None
        )

        repo.upsert_parsed_references_cache(
            media_id=5,
            user_id="1",
            parser_version="11",
            content_hash="old",
            references=["[1] Old"],
            total_detected=1,
            updated_at=_now(),
        )
        assert repo.get_parsed_references_cache(
            media_id=5,
            user_id="1",
            parser_version="11",
            content_hash="old",
        ) == (["[1] Old"], 1)

        repo.upsert_parsed_references_cache(
            media_id=5,
            user_id="1",
            parser_version="11",
            content_hash="new",
            references=["[1] New", "[2] Newer"],
            total_detected=4,
            updated_at=_now(),
        )

        assert (
            repo.get_parsed_references_cache(
                media_id=5,
                user_id="1",
                parser_version="11",
                content_hash="old",
            )
            is None
        )
        assert repo.get_parsed_references_cache(
            media_id=5,
            user_id="1",
            parser_version="11",
            content_hash="new",
        ) == (["[1] New", "[2] Newer"], 4)
    finally:
        db.close_connection()
