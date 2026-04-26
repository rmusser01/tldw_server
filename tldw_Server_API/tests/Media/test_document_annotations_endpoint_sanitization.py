import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.media import document_annotations as annotations_endpoint

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.debug_calls = []
        self.error_calls = []
        self.info_calls = []
        self.warning_calls = []
        self.exception_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def info(self, *args, **kwargs):
        self.info_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))


_SENSITIVE_MARKERS = (
    "annotations backend leaked",
    "/private/tmp/document-annotations.db",
)


def _backend_failure() -> RuntimeError:
    return RuntimeError("annotations backend leaked /private/tmp/document-annotations.db")


def _patch_logger(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(annotations_endpoint, "logger", logger_stub, raising=True)
    return logger_stub


def _assert_sanitized_error_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.exception_calls == []
    assert logger_stub.error_calls

    matching_messages = [args[0] for args, _kwargs in logger_stub.error_calls if args]
    assert expected_message in matching_messages
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)

    rendered_calls = repr(logger_stub.error_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


def _assert_sanitized_warning_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.warning_calls

    matching_messages = [args[0] for args, _kwargs in logger_stub.warning_calls if args]
    assert expected_message in matching_messages
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.warning_calls)

    rendered_calls = repr(logger_stub.warning_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


def _user() -> object:
    return type("User", (), {"id": 1})()


def _annotation_row(annotation_id: str = "ann_existing") -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": annotation_id,
        "location": "page-1",
        "text": "Important text",
        "color": "yellow",
        "note": None,
        "annotation_type": "highlight",
        "chapter_title": None,
        "percentage": None,
        "created_at": now,
        "updated_at": now,
    }


class _CursorStub:
    def __init__(self, *, row: dict[str, Any] | None = None):
        self._row = row

    def fetchone(self) -> dict[str, Any] | None:
        return self._row


class _ConnectionStub:
    def __init__(self, *, row: dict[str, Any] | None = None):
        self._row = row

    def execute(self, *_args, **_kwargs) -> _CursorStub:
        return _CursorStub(row=self._row)


class _FailingAnnotationsDb:
    def __init__(self, outcomes: list[Any]):
        self._outcomes = list(outcomes)
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def get_media_by_id(
        self,
        media_id: int,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any]:
        return {
            "id": media_id,
            "title": f"Media {media_id}",
            "type": "document",
            "deleted": int(include_deleted),
            "is_trash": int(include_trash),
        }

    @contextmanager
    def transaction(self):
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome

        if outcome == "real":
            try:
                yield self._conn
                self._conn.commit()
            finally:
                pass
            return

        if outcome == "existing_annotation":
            yield _ConnectionStub(row=_annotation_row())
            return

        raise AssertionError(f"Unknown transaction outcome: {outcome!r}")

    def close(self) -> None:
        self._conn.close()


def test_ensure_annotations_table_sanitizes_warning_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    db = _FailingAnnotationsDb([_backend_failure()])

    try:
        annotations_endpoint._ensure_annotations_table(db)
    finally:
        db.close()

    _assert_sanitized_warning_log(
        logger_stub,
        "Could not create annotations table",
    )


async def test_list_annotations_sanitizes_fetch_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    db = _FailingAnnotationsDb(["real", _backend_failure()])

    try:
        with pytest.raises(HTTPException) as exc_info:
            await annotations_endpoint.list_annotations(media_id=42, db=db, current_user=_user())
    finally:
        db.close()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to fetch annotations"
    _assert_sanitized_error_log(logger_stub, "Error fetching annotations")


async def test_create_annotation_sanitizes_create_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    db = _FailingAnnotationsDb(["real", _backend_failure()])

    try:
        with pytest.raises(HTTPException) as exc_info:
            await annotations_endpoint.create_annotation(
                media_id=42,
                body=annotations_endpoint.AnnotationCreate(
                    location="page-1",
                    text="Important text",
                ),
                db=db,
                current_user=_user(),
            )
    finally:
        db.close()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create annotation"
    _assert_sanitized_error_log(logger_stub, "Error creating annotation")


async def test_update_annotation_sanitizes_fetch_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    db = _FailingAnnotationsDb(["real", _backend_failure()])

    try:
        with pytest.raises(HTTPException) as exc_info:
            await annotations_endpoint.update_annotation(
                media_id=42,
                annotation_id="ann_existing",
                body=annotations_endpoint.AnnotationUpdate(text="Updated text"),
                db=db,
                current_user=_user(),
            )
    finally:
        db.close()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to fetch annotation"
    _assert_sanitized_error_log(logger_stub, "Error fetching annotation")


async def test_update_annotation_sanitizes_update_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    db = _FailingAnnotationsDb(["real", "existing_annotation", _backend_failure()])

    try:
        with pytest.raises(HTTPException) as exc_info:
            await annotations_endpoint.update_annotation(
                media_id=42,
                annotation_id="ann_existing",
                body=annotations_endpoint.AnnotationUpdate(text="Updated text"),
                db=db,
                current_user=_user(),
            )
    finally:
        db.close()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update annotation"
    _assert_sanitized_error_log(logger_stub, "Error updating annotation")


async def test_delete_annotation_sanitizes_delete_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    db = _FailingAnnotationsDb(["real", _backend_failure()])

    try:
        with pytest.raises(HTTPException) as exc_info:
            await annotations_endpoint.delete_annotation(
                media_id=42,
                annotation_id="ann_existing",
                db=db,
                current_user=_user(),
            )
    finally:
        db.close()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to delete annotation"
    _assert_sanitized_error_log(logger_stub, "Error deleting annotation")


async def test_sync_annotations_sanitizes_sync_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    db = _FailingAnnotationsDb(["real", _backend_failure()])

    try:
        with pytest.raises(HTTPException) as exc_info:
            await annotations_endpoint.sync_annotations(
                media_id=42,
                body=annotations_endpoint.AnnotationSyncRequest(
                    annotations=[
                        annotations_endpoint.AnnotationCreate(
                            location="page-1",
                            text="Important text",
                        )
                    ],
                    client_ids=["client-1"],
                ),
                db=db,
                current_user=_user(),
            )
    finally:
        db.close()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to sync annotations"
    _assert_sanitized_error_log(logger_stub, "Error syncing annotations")
