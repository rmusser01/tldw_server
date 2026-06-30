from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.services import admin_usage_service


class _CursorStub:
    def __init__(self, *, row: Any = None, rows: list[Any] | None = None) -> None:
        self._row = row
        self._rows = rows if rows is not None else ([] if row is None else [row])

    async def fetchone(self) -> Any:
        return self._row

    async def fetchall(self) -> list[Any]:
        return list(self._rows)


class _SQLiteTxConnWithPoolHelperTraps:
    def __init__(self) -> None:
        self._is_sqlite = True
        self.execute_calls: list[tuple[str, Any]] = []
        self.fetchval_called = False
        self.fetchall_called = False

    async def fetchval(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - should never run
        self.fetchval_called = True
        raise AssertionError("sqlite transaction path should not use db.fetchval")

    async def fetchall(self, *args: Any, **kwargs: Any) -> list[Any]:  # pragma: no cover - should never run
        self.fetchall_called = True
        raise AssertionError("sqlite transaction path should not use db.fetchall")

    async def execute(self, query: str, params: Any) -> _CursorStub:
        q = str(query).lower()
        self.execute_calls.append((str(query), params))
        if "count(*)" in q:
            return _CursorStub(row=(1,))
        if "ifnull(bytes_in_total,0)" in q:
            return _CursorStub(
                rows=[
                    {
                        "user_id": 1,
                        "day": "2026-02-09",
                        "requests": 3,
                        "errors": 0,
                        "bytes_total": 1200,
                        "bytes_in_total": 800,
                        "latency_avg_ms": 42.5,
                    }
                ]
            )
        raise AssertionError(f"Unexpected query: {query!r}")


class _SQLiteTxConnTopWithPoolHelperTraps:
    def __init__(self) -> None:
        self._is_sqlite = True
        self.execute_calls: list[tuple[str, Any]] = []
        self.fetchall_called = False

    async def fetchall(self, *args: Any, **kwargs: Any) -> list[Any]:  # pragma: no cover - should never run
        self.fetchall_called = True
        raise AssertionError("sqlite transaction path should not use db.fetchall")

    async def execute(self, query: str, params: Any) -> _CursorStub:
        q = str(query).lower()
        self.execute_calls.append((str(query), params))
        if "sum(requests)" in q and "ifnull(sum(bytes_in_total),0)" in q:
            return _CursorStub(
                rows=[
                    {
                        "user_id": 7,
                        "requests": 21,
                        "errors": 1,
                        "bytes_total": 4096,
                        "bytes_in_total": 2048,
                        "latency_avg_ms": 12.3,
                    }
                ]
            )
        raise AssertionError(f"Unexpected query: {query!r}")


class _PostgresTxConnWithSqliteTrap:
    def __init__(self) -> None:
        self._is_sqlite = False
        self.execute_calls: list[tuple[str, Any]] = []
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, params: Any) -> _CursorStub:  # pragma: no cover - should never run
        self.execute_calls.append((str(query), params))
        raise AssertionError("postgres path should not use sqlite execute()")

    async def fetchval(self, query: str, *args: Any) -> int:
        self.fetchval_calls.append((str(query), tuple(args)))
        if "?" in str(query):
            raise AssertionError("postgres path should not use sqlite placeholders")
        return 1

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((str(query), tuple(args)))
        if "?" in str(query):
            raise AssertionError("postgres path should not use sqlite placeholders")
        return [
            {
                "user_id": 11,
                "day": "2026-02-09",
                "requests": 5,
                "errors": 0,
                "bytes_total": 2048,
                "bytes_in_total": 1024,
                "latency_avg_ms": 10.1,
            }
        ]


class _PostgresSummaryFallbackConn:
    def __init__(self) -> None:
        self._is_sqlite = False
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((str(query), tuple(args)))
        if len(self.fetch_calls) == 1:
            raise RuntimeError("cache columns missing")
        return [
            {
                "group_value": "openai",
                "requests": 2,
                "errors": 0,
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "total_cost_usd": 0.02,
                "latency_avg_ms": 11.5,
            }
        ]


@pytest.mark.asyncio
async def test_fetch_usage_daily_sqlite_tx_path_ignores_pool_helpers(
) -> None:
    db = _SQLiteTxConnWithPoolHelperTraps()

    rows, total, has_in = await admin_usage_service.fetch_usage_daily(
        db,
        user_id=None,
        org_ids=None,
        start=None,
        end=None,
        page=1,
        limit=10,
    )

    assert total == 1
    assert has_in is True
    assert rows and rows[0]["user_id"] == 1
    assert rows[0]["bytes_in_total"] == 800
    assert db.fetchval_called is False
    assert db.fetchall_called is False
    assert db.execute_calls, "sqlite transaction path should use execute()"


@pytest.mark.asyncio
async def test_fetch_usage_top_sqlite_tx_path_ignores_pool_helpers(
) -> None:
    db = _SQLiteTxConnTopWithPoolHelperTraps()

    rows = await admin_usage_service.fetch_usage_top(
        db,
        start=None,
        end=None,
        limit=10,
        metric="requests",
        org_ids=None,
    )

    assert rows and rows[0]["user_id"] == 7
    assert rows[0]["requests"] == 21
    assert db.fetchall_called is False
    assert db.execute_calls, "sqlite transaction path should use execute()"


@pytest.mark.asyncio
async def test_fetch_usage_daily_postgres_tx_path_uses_fetch_helpers() -> None:
    db = _PostgresTxConnWithSqliteTrap()

    rows, total, has_in = await admin_usage_service.fetch_usage_daily(
        db,
        user_id=None,
        org_ids=None,
        start=None,
        end=None,
        page=1,
        limit=10,
    )

    assert total == 1
    assert has_in is True
    assert rows and rows[0]["user_id"] == 11
    assert db.fetchval_calls
    assert db.fetch_calls
    assert not db.execute_calls


@pytest.mark.asyncio
async def test_fetch_llm_usage_summary_postgres_fallback_preserves_cache_shape() -> None:
    db = _PostgresSummaryFallbackConn()

    rows = await admin_usage_service.fetch_llm_usage_summary(
        db,
        group_by="provider",
        provider=None,
        start=None,
        end=None,
        org_ids=None,
    )

    assert len(db.fetch_calls) == 2
    assert rows == [
        {
            "group_value": "openai",
            "requests": 2,
            "errors": 0,
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "total_cost_usd": 0.02,
            "latency_avg_ms": 11.5,
            "group_value_secondary": None,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "billable_input_tokens": 0,
            "provider_usage_count": 0,
            "stream_estimate_count": 0,
            "disconnect_estimate_count": 0,
            "missing_usage_count": 0,
            "local_diagnostic_count": 0,
            "estimated_usage_count": 0,
        }
    ]
