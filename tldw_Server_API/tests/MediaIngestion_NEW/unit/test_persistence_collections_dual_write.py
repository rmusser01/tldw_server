from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence
import tldw_Server_API.app.core.DB_Management.Collections_DB as collections_db_module


pytestmark = pytest.mark.unit


class _FakeCollectionsDatabase:
    from_backend_calls: list[tuple[int, Any]] = []
    for_user_calls: list[int] = []
    should_raise: bool = False
    last_instance: "_FakeCollectionsDatabase | None" = None

    def __init__(self) -> None:
        self.upsert_calls: list[dict[str, Any]] = []
        self.resolve_calls: list[dict[str, Any]] = []
        self.closed = False
        _FakeCollectionsDatabase.last_instance = self

    @classmethod
    def reset(cls) -> None:
        cls.from_backend_calls = []
        cls.for_user_calls = []
        cls.should_raise = False
        cls.last_instance = None

    @classmethod
    def from_backend(cls, *, user_id: int, backend: Any) -> "_FakeCollectionsDatabase":
        cls.from_backend_calls.append((user_id, backend))
        return cls()

    @classmethod
    def for_user(cls, *, user_id: int) -> "_FakeCollectionsDatabase":
        cls.for_user_calls.append(user_id)
        return cls()

    def upsert_content_item(self, **kwargs: Any) -> Any:
        self.upsert_calls.append(kwargs)
        if self.should_raise:
            raise RuntimeError("collections failure")
        return SimpleNamespace(id=901)

    def resolve_media_collection_item(self, item_id: int, **kwargs: Any) -> Any:
        self.resolve_calls.append({"item_id": item_id, **kwargs})
        return SimpleNamespace(id=item_id)

    def close(self) -> None:
        self.closed = True


def test_sync_media_add_results_to_collections_writes_expected_payload(monkeypatch):
    _FakeCollectionsDatabase.reset()
    monkeypatch.setattr(
        collections_db_module,
        "CollectionsDatabase",
        _FakeCollectionsDatabase,
    )

    backend = object()
    results = [
        {
            "status": "Success",
            "db_id": "123",
            "input_ref": "https://Example.COM/path/to/item",
            "processing_source": "https://Example.COM/path/to/item",
            "media_type": "document",
            "media_uuid": "uuid-123",
            "content": "alpha beta gamma",
            "summary": "A short summary",
            "metadata": {
                "keywords": ["Alpha", "topic"],
                "published_at": "2026-01-01T00:00:00Z",
            },
        },
        {"status": "Error", "db_id": None, "input_ref": "ignored"},
    ]

    persistence.sync_media_add_results_to_collections(
        results=results,
        form_data=SimpleNamespace(
            media_type="document",
            keywords=["topic", "FromForm"],
            title=None,
        ),
        current_user=SimpleNamespace(id=42),
        db=SimpleNamespace(backend=backend),
    )

    assert _FakeCollectionsDatabase.from_backend_calls == [(42, backend)]
    assert _FakeCollectionsDatabase.for_user_calls == []

    instance = _FakeCollectionsDatabase.last_instance
    assert instance is not None
    assert instance.closed is True
    assert len(instance.upsert_calls) == 1

    payload = instance.upsert_calls[0]
    assert payload["origin"] == "media_add"
    assert payload["origin_type"] == "document"
    assert payload["origin_id"] == 123
    assert payload["media_id"] == 123
    assert payload["source_id"] == 123
    assert payload["canonical_url"] == "media://123"
    assert payload["url"] == "https://Example.COM/path/to/item"
    assert payload["domain"] == "example.com"
    assert payload["tags"] == ["topic", "fromform", "alpha"]
    assert payload["status"] == "saved"
    assert payload["favorite"] is False

    metadata = payload["metadata"]
    assert metadata["origin"] == "media_add"
    assert metadata["provenance"]["entrypoint"] == "/api/v1/media/add"
    assert metadata["provenance"]["media_id"] == 123
    assert metadata["provenance"]["source_url"] == "https://Example.COM/path/to/item"

    assert results[0]["collections_item_id"] == 901
    assert results[0]["collections_origin"] == "media_add"


def test_sync_media_add_results_resolves_planned_collection_item(monkeypatch):
    _FakeCollectionsDatabase.reset()
    monkeypatch.setattr(
        collections_db_module,
        "CollectionsDatabase",
        _FakeCollectionsDatabase,
    )

    results = [
        {
            "status": "Success",
            "db_id": "456",
            "input_ref": "https://example.com/conf-talk",
            "processing_source": "https://example.com/conf-talk",
            "media_type": "video",
        }
    ]

    persistence.sync_media_add_results_to_collections(
        results=results,
        form_data=SimpleNamespace(
            media_type="video",
            keywords=[],
            media_collection_item_id="88",
            media_ingest_job_id="1234",
        ),
        current_user=SimpleNamespace(id=42),
        db=SimpleNamespace(backend=object()),
    )

    instance = _FakeCollectionsDatabase.last_instance
    assert instance is not None
    assert instance.resolve_calls == [
        {
            "item_id": 88,
            "media_id": 456,
            "status": "completed",
            "latest_job_id": "1234",
        }
    ]


def test_sync_media_add_results_to_collections_adds_warning_on_failure(monkeypatch):
    _FakeCollectionsDatabase.reset()
    _FakeCollectionsDatabase.should_raise = True
    monkeypatch.setattr(
        collections_db_module,
        "CollectionsDatabase",
        _FakeCollectionsDatabase,
    )

    results = [
        {
            "status": "Success",
            "db_id": 7,
            "input_ref": "https://example.com/failure-case",
            "processing_source": "https://example.com/failure-case",
            "media_type": "document",
        }
    ]

    persistence.sync_media_add_results_to_collections(
        results=results,
        form_data=SimpleNamespace(media_type="document", keywords=[]),
        current_user=SimpleNamespace(id=99),
        db=SimpleNamespace(backend=object()),
    )

    warnings = results[0].get("warnings") or []
    assert any("Collections dual-write failed: collections failure" in msg for msg in warnings)
    assert "collections_item_id" not in results[0]

    instance = _FakeCollectionsDatabase.last_instance
    assert instance is not None
    assert instance.closed is True
