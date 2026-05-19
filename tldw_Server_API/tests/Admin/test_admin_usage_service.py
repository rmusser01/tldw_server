from __future__ import annotations

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_scope_service
from tldw_Server_API.app.services import admin_usage_service
from tldw_Server_API.app.services.admin_usage_service import get_cost_attribution


pytestmark = pytest.mark.unit


class _FailingUsageCursor:
    async def fetchall(self) -> list[dict]:
        raise AssertionError("fetchall should not be reached after execute failure")


class _FailingUsageDb:
    _is_sqlite = True

    async def execute(self, _query: str) -> _FailingUsageCursor:
        raise RuntimeError("llm_usage_v2 missing")


class _UsageCursorStub:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = list(rows)

    async def fetchall(self) -> list[dict[str, object]]:
        return list(self._rows)


class _ScopedSqliteUsageDb:
    _is_sqlite = True

    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, query: str, params: tuple[object, ...]) -> _UsageCursorStub:
        self.execute_calls.append((str(query), params))
        return _UsageCursorStub(
            [
                {
                    "entity_id": 42,
                    "request_count": 3,
                    "total_tokens": 1000,
                    "prompt_tokens": 400,
                    "completion_tokens": 600,
                }
            ]
        )


class _PgSummaryFallbackDb:
    _is_sqlite = False

    def __init__(self) -> None:
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query: str, *params: object) -> list[dict[str, object]]:
        self.fetch_calls.append((str(query), params))
        if len(self.fetch_calls) == 1:
            raise RuntimeError("cache columns missing")
        return [
            {
                "group_value": "openai",
                "requests": 1,
                "errors": 0,
                "input_tokens": 100,
                "output_tokens": 25,
                "total_tokens": 125,
                "total_cost_usd": 0.001,
                "latency_avg_ms": 12.5,
            }
        ]


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )


@pytest.mark.asyncio
async def test_get_cost_attribution_surfaces_backend_failures() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await get_cost_attribution(
            principal=_admin_principal(),
            db=_FailingUsageDb(),
            group_by="user",
            range_days=7,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Cost attribution is currently unavailable"


@pytest.mark.asyncio
async def test_get_cost_attribution_sqlite_restricts_results_to_admin_org_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _ScopedSqliteUsageDb()

    async def _fake_get_admin_org_ids(_principal: AuthPrincipal) -> list[int] | None:
        return [42]

    monkeypatch.setattr(admin_scope_service, "get_admin_org_ids", _fake_get_admin_org_ids)

    result = await get_cost_attribution(
        principal=_admin_principal(),
        db=db,
        group_by="org",
        range_days=7,
    )

    assert result["items"][0]["entity_id"] == 42
    assert len(db.execute_calls) == 1
    query, params = db.execute_calls[0]
    assert "org_id IN (?)" in query
    assert params == ("-7 days", 42)


@pytest.mark.asyncio
async def test_get_cost_attribution_returns_empty_items_when_admin_scope_has_no_org_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _ScopedSqliteUsageDb()

    async def _fake_get_admin_org_ids(_principal: AuthPrincipal) -> list[int] | None:
        return []

    monkeypatch.setattr(admin_scope_service, "get_admin_org_ids", _fake_get_admin_org_ids)

    result = await get_cost_attribution(
        principal=_admin_principal(),
        db=db,
        group_by="user",
        range_days=7,
    )

    assert result == {"group_by": "user", "range_days": 7, "items": []}
    assert db.execute_calls == []


@pytest.mark.asyncio
async def test_llm_usage_summary_pg_fallback_preserves_cache_summary_shape() -> None:
    db = _PgSummaryFallbackDb()

    rows = await admin_usage_service.fetch_llm_usage_summary(
        db,
        group_by="provider",
        provider=None,
        start=None,
        end=None,
        org_ids=None,
    )

    assert len(db.fetch_calls) == 2
    fallback_sql = db.fetch_calls[1][0]
    for fragment in (
        "0 AS cached_input_tokens",
        "0 AS cache_write_input_tokens",
        "0 AS cache_read_input_tokens",
        "SUM(COALESCE(prompt_tokens,0)) AS billable_input_tokens",
        "0 AS provider_usage_count",
        "0 AS stream_estimate_count",
        "0 AS disconnect_estimate_count",
        "0 AS missing_usage_count",
        "0 AS local_diagnostic_count",
        "SUM(CASE WHEN estimated THEN 1 ELSE 0 END) AS estimated_usage_count",
    ):
        assert fragment in fallback_sql
    assert rows[0]["group_value"] == "openai"
    assert rows[0]["cached_input_tokens"] == 0
    assert rows[0]["billable_input_tokens"] == 0
    assert rows[0]["estimated_usage_count"] == 0
    assert rows[0]["latency_avg_ms"] == 12.5


@pytest.mark.asyncio
async def test_get_usage_daily_includes_canonical_page_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_get_admin_org_ids(_principal: AuthPrincipal) -> list[int] | None:
        return None

    async def _fake_fetch_usage_daily(*_args, **kwargs):
        assert kwargs["page"] == 2
        assert kwargs["limit"] == 1
        return (
            [
                {
                    "user_id": 42,
                    "day": "2026-05-02",
                    "requests": 5,
                    "errors": 0,
                    "bytes_total": 2048,
                    "bytes_in_total": 1024,
                    "latency_avg_ms": 12.5,
                }
            ],
            3,
            True,
        )

    monkeypatch.setattr(admin_scope_service, "get_admin_org_ids", _fake_get_admin_org_ids)
    monkeypatch.setattr(admin_usage_service, "fetch_usage_daily", _fake_fetch_usage_daily)

    result = await admin_usage_service.get_usage_daily(
        principal=_admin_principal(),
        db=object(),
        user_id=None,
        start=None,
        end=None,
        page=2,
        limit=1,
        org_id=None,
    )

    assert result.total == 3
    assert result.page == 2
    assert result.limit == 1
    assert result.pagination.model_dump(mode="json") == {
        "mode": "page",
        "page": 2,
        "per_page": 1,
        "total": 3,
        "total_pages": 3,
        "has_more": True,
    }
