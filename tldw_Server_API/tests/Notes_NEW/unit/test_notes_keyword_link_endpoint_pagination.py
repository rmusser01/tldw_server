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
        if "COUNT(*)" in sql and "collection_keywords" in sql:
            return _Cursor([{"total": 3}])
        if "COUNT(*)" in sql and "conversation_keywords" in sql:
            return _Cursor([{"total": 3}])
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
