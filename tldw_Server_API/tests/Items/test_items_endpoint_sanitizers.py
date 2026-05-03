from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import items
from tldw_Server_API.app.api.v1.schemas.items_schemas import ItemsBulkRequest
from tldw_Server_API.app.core.DB_Management.Collections_DB import ContentItemRow


class _LoggerStub:
    def __init__(self) -> None:
        self.error_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_records.append((message, args, kwargs))


class _FailingCollectionsDb:
    def update_content_item(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("items backend exploded at /private/items.db")


class _ListingCollectionsDb:
    def list_content_items(self, **kwargs: Any) -> tuple[list[ContentItemRow], int]:
        assert kwargs["page"] == 2
        assert kwargs["size"] == 1
        return (
            [
                ContentItemRow(
                    id=42,
                    user_id="1",
                    origin="reading",
                    origin_type=None,
                    origin_id=None,
                    url="https://example.test/item",
                    canonical_url=None,
                    domain="example.test",
                    title="Example Item",
                    summary=None,
                    notes=None,
                    content_hash=None,
                    word_count=None,
                    published_at=None,
                    status="saved",
                    favorite=False,
                    metadata_json=None,
                    media_id=None,
                    job_id=None,
                    run_id=None,
                    source_id=None,
                    read_at=None,
                    created_at="2026-05-02T00:00:00+00:00",
                    updated_at="2026-05-02T00:00:00+00:00",
                    tags=[],
                )
            ],
            3,
        )


@pytest.mark.asyncio
async def test_list_items_includes_canonical_page_pagination() -> None:
    response = await items.list_items(
        ids=None,
        q=None,
        tags=None,
        domain=None,
        date_from=None,
        date_to=None,
        status_filter=None,
        favorite=None,
        origin=None,
        job_id=None,
        run_id=None,
        page=2,
        size=1,
        current_user=object(),
        db=object(),
        collections_db=_ListingCollectionsDb(),
    )

    assert response.page == 2
    assert response.size == 1
    assert response.total == 3
    assert response.pagination.model_dump(mode="json") == {
        "mode": "page",
        "page": 2,
        "per_page": 1,
        "total": 3,
        "total_pages": 3,
        "has_more": True,
    }


@pytest.mark.asyncio
async def test_bulk_update_items_sanitizes_per_item_failure_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(items, "logger", logger_stub)

    response = await items.bulk_update_items(
        payload=ItemsBulkRequest(
            item_ids=[42],
            action="set_favorite",
            favorite=True,
        ),
        current_user=object(),
        collections_db=_FailingCollectionsDb(),
    )

    assert response.total == 1
    assert response.succeeded == 0
    assert response.failed == 1
    assert response.results[0].item_id == 42
    assert response.results[0].success is False
    assert response.results[0].error == "update_failed"
    assert logger_stub.error_records == [("bulk_update_items failed", (), {})]
