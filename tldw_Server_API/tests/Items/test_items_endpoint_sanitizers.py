from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import items
from tldw_Server_API.app.api.v1.schemas.items_schemas import ItemsBulkRequest


class _LoggerStub:
    def __init__(self) -> None:
        self.error_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_records.append((message, args, kwargs))


class _FailingCollectionsDb:
    def update_content_item(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("items backend exploded at /private/items.db")


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
