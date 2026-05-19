from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import pytest

from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError


def _expect_true(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


class _BrokenLegacyDb:
    client_id = "test-client"
    db_path_str = ":memory:"

    def execute_query(self, query: str, params: Any = None) -> Any:
        raise RuntimeError("query boom")

    def transaction(self):
        return nullcontext(object())

    def _fetchall_with_connection(self, connection: Any, query: str, params: Any = None) -> Any:
        raise RuntimeError("fetchall boom")

    def _fetchone_with_connection(self, connection: Any, query: str, params: Any = None) -> Any:
        raise RuntimeError("fetchone boom")

    def _execute_with_connection(self, connection: Any, query: str, params: Any = None) -> Any:
        raise RuntimeError("exec boom")

    def _get_current_utc_timestamp_str(self) -> str:
        return "2026-04-07T00:00:00Z"

    def _log_sync_event(
        self,
        connection: Any,
        table_name: str,
        entity_uuid: str,
        action: str,
        version: int,
        payload: Any = None,
    ) -> None:
        return None

    def initialize_db(self) -> None:
        return None

    def close_connection(self) -> None:
        return None


class _Cursor:
    def __init__(self, rows: list[Any] | None = None, row: Any = None) -> None:
        self._rows = [] if rows is None else list(rows)
        self._row = row

    def fetchall(self) -> list[Any]:
        return list(self._rows)

    def fetchone(self) -> Any:
        return self._row


class _RowlessLegacyDb(_BrokenLegacyDb):
    def execute_query(self, query: str, params: Any = None) -> Any:
        return _Cursor(rows=[], row=None)

    def _execute_with_connection(self, connection: Any, query: str, params: Any = None) -> Any:
        return _Cursor(rows=[], row=None)


@pytest.mark.parametrize(
    ("label", "reader"),
    [
        ("has_unvectorized_chunks", lambda db: media_db_api.has_unvectorized_chunks(db, 1)),
        ("get_unvectorized_chunk_count", lambda db: media_db_api.get_unvectorized_chunk_count(db, 1)),
        (
            "get_unvectorized_anchor_index_for_offset",
            lambda db: media_db_api.get_unvectorized_anchor_index_for_offset(db, 1, 5),
        ),
        (
            "get_unvectorized_chunk_index_by_uuid",
            lambda db: media_db_api.get_unvectorized_chunk_index_by_uuid(db, 1, "chunk-1"),
        ),
        (
            "get_unvectorized_chunk_by_index",
            lambda db: media_db_api.get_unvectorized_chunk_by_index(db, 1, 0),
        ),
        (
            "get_unvectorized_chunks_in_range",
            lambda db: media_db_api.get_unvectorized_chunks_in_range(db, 1, 0, 1),
        ),
        ("lookup_section_for_offset", lambda db: media_db_api.lookup_section_for_offset(db, 1, 10)),
        ("lookup_section_by_heading", lambda db: media_db_api.lookup_section_by_heading(db, 1, "intro")),
    ],
)
def test_media_db_read_helpers_raise_database_error_on_backend_failures(
    label: str,
    reader: Callable[[Any], Any],
) -> None:
    with pytest.raises(DatabaseError) as exc_info:
        reader(_BrokenLegacyDb())

    _expect_true(label in str(exc_info.value), f"expected {label} in error message: {exc_info.value}")


def test_lookup_section_by_heading_keeps_not_found_semantics() -> None:
    result = media_db_api.lookup_section_by_heading(_RowlessLegacyDb(), 1, "missing")
    _expect_true(result is None, f"expected None for missing heading, got {result!r}")


def test_get_unvectorized_chunks_in_range_keeps_empty_result_semantics() -> None:
    result = media_db_api.get_unvectorized_chunks_in_range(_RowlessLegacyDb(), 1, 0, 1)
    _expect_true(result == [], f"expected empty list for missing chunks, got {result!r}")
