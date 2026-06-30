from __future__ import annotations

from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from tldw_Server_API.app.api.v1.endpoints.rag_unified import (
    _apply_media_collection_scope,
)
from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGRequest


class _FakeCollectionsDb:
    def __init__(self, items: list[SimpleNamespace]) -> None:
        self.items = items
        self.requested_collection_id: int | None = None

    def get_media_collection(self, collection_id: int) -> SimpleNamespace:
        self.requested_collection_id = collection_id
        return SimpleNamespace(id=collection_id, items=self.items)


def _item(status: str, media_id: int | None) -> SimpleNamespace:
    return SimpleNamespace(status=status, media_id=media_id)


class _MissingCollectionDb:
    def get_media_collection(self, collection_id: int) -> SimpleNamespace:
        raise KeyError("media_collection_not_found")


def test_conference_collection_scope_limits_rag_to_ready_media() -> None:
    request = UnifiedRAGRequest(query="keynote", collection_id=7)
    db = _FakeCollectionsDb(
        [
            _item("completed", 101),
            _item("skipped_existing", 102),
            _item("processing", 103),
            _item("failed", 104),
            _item("completed", None),
        ]
    )

    scoped = _apply_media_collection_scope(request, db)

    assert db.requested_collection_id == 7
    assert scoped.include_media_ids == [101, 102]


def test_conference_collection_scope_intersects_explicit_media_ids() -> None:
    request = UnifiedRAGRequest(
        query="keynote",
        collection_id=7,
        include_media_ids=[101, 999],
    )
    db = _FakeCollectionsDb(
        [
            _item("completed", 101),
            _item("completed", 102),
        ]
    )

    scoped = _apply_media_collection_scope(request, db)

    assert scoped.include_media_ids == [101]


def test_conference_collection_scope_fails_closed_when_nothing_is_ready() -> None:
    request = UnifiedRAGRequest(query="keynote", collection_id=7)
    db = _FakeCollectionsDb(
        [
            _item("planned", None),
            _item("failed", 104),
            _item("cancelled", 105),
        ]
    )

    scoped = _apply_media_collection_scope(request, db)

    assert scoped.include_media_ids == [-1]


def test_conference_collection_scope_reports_missing_collection() -> None:
    request = UnifiedRAGRequest(query="keynote", collection_id=7)

    with pytest.raises(HTTPException) as exc_info:
        _apply_media_collection_scope(request, _MissingCollectionDb())

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "media_collection_not_found"
