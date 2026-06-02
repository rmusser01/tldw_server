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


class _MediaOnlyAnnotationsDb:
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

    def close(self) -> None:
        return None


class _RepositoryStub:
    def __init__(
        self,
        *,
        exceptions: dict[str, Exception] | None = None,
        list_result: list[dict[str, Any]] | None = None,
        get_result: dict[str, Any] | None = None,
        update_result: dict[str, Any] | None = None,
        soft_delete_result: bool = True,
        sync_result: list[dict[str, Any]] | None = None,
    ):
        self.exceptions = exceptions or {}
        self.list_result = list_result or []
        self.get_result = get_result
        self.update_result = update_result
        self.soft_delete_result = soft_delete_result
        self.sync_result = sync_result or []
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))
        if name in self.exceptions:
            raise self.exceptions[name]

    def list_annotations(self, *, media_id: int, user_id: str) -> list[dict[str, Any]]:
        self._record("list_annotations", media_id=media_id, user_id=user_id)
        return list(self.list_result)

    def create_annotation(self, **kwargs: Any) -> dict[str, Any]:
        self._record("create_annotation", **kwargs)
        if self.sync_result:
            return self.sync_result[0]
        return _annotation_row(kwargs["annotation_id"])

    def get_annotation(self, *, annotation_id: str, media_id: int, user_id: str) -> dict[str, Any] | None:
        self._record("get_annotation", annotation_id=annotation_id, media_id=media_id, user_id=user_id)
        return self.get_result

    def update_annotation(self, **kwargs: Any) -> dict[str, Any] | None:
        self._record("update_annotation", **kwargs)
        return self.update_result

    def soft_delete_annotation(self, **kwargs: Any) -> bool:
        self._record("soft_delete_annotation", **kwargs)
        return self.soft_delete_result

    def sync_annotations(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._record("sync_annotations", **kwargs)
        return list(self.sync_result)


def _patch_repository(monkeypatch: pytest.MonkeyPatch, repo: _RepositoryStub) -> None:
    class RepositoryFactory:
        @staticmethod
        def from_media_db(db: Any) -> _RepositoryStub:
            repo.calls.append(("from_media_db", (db,), {}))
            return repo

    monkeypatch.setattr(
        annotations_endpoint,
        "DocumentWorkspaceRepository",
        RepositoryFactory,
        raising=False,
    )


async def test_list_annotations_delegates_to_repository(monkeypatch):
    repo = _RepositoryStub(list_result=[_annotation_row()])
    _patch_repository(monkeypatch, repo)

    result = await annotations_endpoint.list_annotations(
        media_id=42,
        db=_MediaOnlyAnnotationsDb(),
        current_user=_user(),
    )

    assert result.total_count == 1
    assert result.annotations[0].id == "ann_existing"
    assert [call[0] for call in repo.calls] == ["from_media_db", "list_annotations"]
    assert repo.calls[1][2] == {"media_id": 42, "user_id": "1"}


async def test_list_annotations_sanitizes_fetch_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    _patch_repository(monkeypatch, _RepositoryStub(exceptions={"list_annotations": _backend_failure()}))

    with pytest.raises(HTTPException) as exc_info:
        await annotations_endpoint.list_annotations(media_id=42, db=_MediaOnlyAnnotationsDb(), current_user=_user())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to fetch annotations"
    _assert_sanitized_error_log(logger_stub, "Error fetching annotations")


async def test_create_annotation_sanitizes_create_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    _patch_repository(monkeypatch, _RepositoryStub(exceptions={"create_annotation": _backend_failure()}))

    with pytest.raises(HTTPException) as exc_info:
        await annotations_endpoint.create_annotation(
            media_id=42,
            body=annotations_endpoint.AnnotationCreate(
                location="page-1",
                text="Important text",
            ),
            db=_MediaOnlyAnnotationsDb(),
            current_user=_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create annotation"
    _assert_sanitized_error_log(logger_stub, "Error creating annotation")


async def test_update_annotation_sanitizes_fetch_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    _patch_repository(monkeypatch, _RepositoryStub(exceptions={"get_annotation": _backend_failure()}))

    with pytest.raises(HTTPException) as exc_info:
        await annotations_endpoint.update_annotation(
            media_id=42,
            annotation_id="ann_existing",
            body=annotations_endpoint.AnnotationUpdate(text="Updated text"),
            db=_MediaOnlyAnnotationsDb(),
            current_user=_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to fetch annotation"
    _assert_sanitized_error_log(logger_stub, "Error fetching annotation")


async def test_update_annotation_sanitizes_update_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    _patch_repository(
        monkeypatch,
        _RepositoryStub(
            exceptions={"update_annotation": _backend_failure()},
            get_result=_annotation_row(),
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await annotations_endpoint.update_annotation(
            media_id=42,
            annotation_id="ann_existing",
            body=annotations_endpoint.AnnotationUpdate(text="Updated text"),
            db=_MediaOnlyAnnotationsDb(),
            current_user=_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update annotation"
    _assert_sanitized_error_log(logger_stub, "Error updating annotation")


async def test_delete_annotation_sanitizes_delete_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    _patch_repository(monkeypatch, _RepositoryStub(exceptions={"soft_delete_annotation": _backend_failure()}))

    with pytest.raises(HTTPException) as exc_info:
        await annotations_endpoint.delete_annotation(
            media_id=42,
            annotation_id="ann_existing",
            db=_MediaOnlyAnnotationsDb(),
            current_user=_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to delete annotation"
    _assert_sanitized_error_log(logger_stub, "Error deleting annotation")


async def test_sync_annotations_sanitizes_sync_error_log(monkeypatch):
    logger_stub = _patch_logger(monkeypatch)
    _patch_repository(monkeypatch, _RepositoryStub(exceptions={"sync_annotations": _backend_failure()}))

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
            db=_MediaOnlyAnnotationsDb(),
            current_user=_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to sync annotations"
    _assert_sanitized_error_log(logger_stub, "Error syncing annotations")
