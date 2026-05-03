from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint


class _Cursor:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[dict[str, object]]:
        return self._rows


class _RateLimiter:
    async def check_user_rate_limit(self, user_id: int, operation: str) -> tuple[bool, dict[str, object]]:
        return True, {}


class _User:
    id = 1


class _KeywordLinkDB:
    client_id = "test-client"

    def execute_query(self, sql: str, params: tuple[object, ...] = ()) -> _Cursor:
        if "collection_keywords" in sql:
            return _Cursor(
                [
                    {"collection_id": 10, "keyword_id": 1},
                    {"collection_id": 11, "keyword_id": 2},
                ]
            )
        if "conversation_keywords" in sql:
            return _Cursor(
                [
                    {"conversation_id": "conv-a", "keyword_id": 1},
                    {"conversation_id": "conv-b", "keyword_id": 2},
                ]
            )
        raise AssertionError(f"Unexpected SQL: {sql}")

    def count_collection_keyword_links(self) -> int:
        return 3

    def count_conversation_keyword_links(self) -> int:
        return 3


class _NotesForKeywordDB:
    client_id = "test-client"

    def get_keyword_by_id(self, keyword_id: int) -> dict[str, object]:
        assert keyword_id == 7
        return {"id": 7, "keyword": "research"}

    def get_notes_for_keyword(self, keyword_id: int, limit: int, offset: int) -> list[dict[str, object]]:
        assert keyword_id == 7
        assert limit == 2
        assert offset == 0
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return [
            {
                "id": "note-a",
                "title": "A",
                "content": "Alpha",
                "created_at": now,
                "last_modified": now,
                "version": 1,
                "client_id": "test-client",
                "deleted": False,
            },
            {
                "id": "note-b",
                "title": "B",
                "content": "Beta",
                "created_at": now,
                "last_modified": now,
                "version": 1,
                "client_id": "test-client",
                "deleted": False,
            },
        ]

    def get_note_counts_for_keywords(self, keyword_ids: list[int]) -> dict[int, int]:
        assert keyword_ids == [7]
        return {7: 3}


@pytest.mark.asyncio
async def test_collection_keyword_links_include_canonical_offset_pagination() -> None:
    """Collection keyword link lists should expose canonical offset metadata."""
    response = await notes_endpoint.list_collection_keyword_links_endpoint(
        db=_KeywordLinkDB(),
        limit=2,
        offset=0,
        rate_limiter=_RateLimiter(),
        current_user=_User(),
        _=None,
    )

    assert response["links"] == [
        {"collection_id": 10, "keyword_id": 1},
        {"collection_id": 11, "keyword_id": 2},
    ]
    assert response["pagination"].model_dump(mode="json") == {
        "mode": "offset",
        "limit": 2,
        "offset": 0,
        "total": 3,
        "has_more": True,
        "next_offset": 2,
    }
    assert response["has_more"] is True
    assert response["next_offset"] == 2


@pytest.mark.asyncio
async def test_conversation_keyword_links_include_canonical_offset_pagination() -> None:
    """Conversation keyword link lists should expose canonical offset metadata."""
    response = await notes_endpoint.list_conversation_keyword_links_endpoint(
        db=_KeywordLinkDB(),
        ids=None,
        limit=2,
        offset=0,
        rate_limiter=_RateLimiter(),
        current_user=_User(),
        _=None,
    )

    assert response["links"] == [
        {"conversation_id": "conv-a", "keyword_id": 1},
        {"conversation_id": "conv-b", "keyword_id": 2},
    ]
    assert response["pagination"].model_dump(mode="json") == {
        "mode": "offset",
        "limit": 2,
        "offset": 0,
        "total": 3,
        "has_more": True,
        "next_offset": 2,
    }
    assert response["has_more"] is True
    assert response["next_offset"] == 2


@pytest.mark.asyncio
async def test_notes_for_keyword_include_canonical_offset_pagination() -> None:
    """Notes-for-keyword lists should expose canonical offset metadata."""
    response = await notes_endpoint.get_notes_for_keyword_endpoint(
        keyword_id=7,
        db=_NotesForKeywordDB(),
        limit=2,
        offset=0,
        rate_limiter=_RateLimiter(),
        current_user=_User(),
        _=None,
    )

    assert response.keyword_id == 7
    assert [note.id for note in response.notes] == ["note-a", "note-b"]
    assert response.pagination.model_dump(mode="json") == {
        "mode": "offset",
        "limit": 2,
        "offset": 0,
        "total": 3,
        "has_more": True,
        "next_offset": 2,
    }
    assert response.has_more is True
    assert response.next_offset == 2
