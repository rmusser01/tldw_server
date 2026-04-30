from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin import admin_rate_limits
from tldw_Server_API.app.api.v1.schemas.admin_rbac_schemas import RateLimitUpsertRequest
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_rate_limits_service


class _CursorStub:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = list(rows)

    async def fetchall(self) -> list[Any]:
        return list(self._rows)


class _SqliteRowLike:
    def __init__(self, keys: list[str], values: tuple[Any, ...]) -> None:
        self._keys = list(keys)
        self._values = tuple(values)

    def keys(self) -> list[str]:
        return list(self._keys)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return self._values[key]
        idx = self._keys.index(str(key))
        return self._values[idx]


class _SqliteDbStub:
    def __init__(self) -> None:
        self._is_sqlite = True
        self.execute_calls: list[tuple[str, Any]] = []

    async def execute(self, query: str, params: Any = ()) -> _CursorStub:
        self.execute_calls.append((str(query), params))
        normalized = str(query).lower()
        if "from rbac_role_rate_limits" in normalized:
            return _CursorStub(
                [
                    _SqliteRowLike(
                        ["scope", "id", "resource", "limit_per_min", "burst"],
                        ("role", 7, "/api/v1/chat/completions", 30, 5),
                    )
                ]
            )
        if "from rbac_user_rate_limits" in normalized:
            return _CursorStub(
                [
                    _SqliteRowLike(
                        ["scope", "id", "resource", "limit_per_min", "burst"],
                        ("user", 11, "/api/v1/rag/search", 12, 2),
                    )
                ]
            )
        raise AssertionError(f"Unexpected query: {query!r}")


class _PostgresDbStub:
    def __init__(self) -> None:
        self._is_sqlite = False
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((str(query), tuple(args)))
        normalized = str(query).lower()
        if "from rbac_role_rate_limits" in normalized:
            return [
                {
                    "scope": "role",
                    "id": 3,
                    "resource": "/api/v1/media/search",
                    "limit_per_min": 60,
                    "burst": 10,
                }
            ]
        if "from rbac_user_rate_limits" in normalized:
            return [
                {
                    "scope": "user",
                    "id": 9,
                    "resource": "/api/v1/media/ingest/jobs",
                    "limit_per_min": 6,
                    "burst": 1,
                }
            ]
        raise AssertionError(f"Unexpected query: {query!r}")


class _ExplodingRateLimitsDb:
    async def execute(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("rate limits backend exploded at /private/rate-limits.db")


def _platform_admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )


async def _assert_rate_limits_log_sanitized(
    call: Callable[[], Awaitable[Any]],
    *,
    expected_log: str,
    raw_marker: str,
) -> None:
    messages: list[str] = []
    sink_id = admin_rate_limits.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(HTTPException) as excinfo:
            await call()
    finally:
        admin_rate_limits.logger.remove(sink_id)

    assert excinfo.value.status_code == 500  # nosec B101
    joined = "\n".join(messages)
    assert expected_log in joined  # nosec B101
    assert raw_marker not in joined  # nosec B101
    assert "/private/" not in joined  # nosec B101


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    ("call_factory", "expected_log"),
    [
        (
            lambda: admin_rate_limits.list_admin_rate_limits(db=_ExplodingRateLimitsDb()),
            "Failed to list admin rate limits",
        ),
        (
            lambda: admin_rate_limits.upsert_role_rate_limit(
                role_id=7,
                payload=RateLimitUpsertRequest(
                    resource="/api/v1/chat/completions",
                    limit_per_min=30,
                    burst=5,
                ),
                db=_ExplodingRateLimitsDb(),
            ),
            "Failed to upsert role rate limit",
        ),
        (
            lambda: admin_rate_limits.clear_role_rate_limits(
                role_id=7,
                db=_ExplodingRateLimitsDb(),
            ),
            "Failed to clear role rate limits",
        ),
        (
            lambda: admin_rate_limits.upsert_user_rate_limit(
                user_id=11,
                payload=RateLimitUpsertRequest(
                    resource="/api/v1/rag/search",
                    limit_per_min=12,
                    burst=2,
                ),
                principal=_platform_admin_principal(),
                db=_ExplodingRateLimitsDb(),
            ),
            "Failed to upsert user rate limit",
        ),
    ],
)
async def test_admin_rate_limit_endpoints_sanitize_backend_failure_logs(
    monkeypatch: pytest.MonkeyPatch,
    call_factory: Callable[[], Awaitable[Any]],
    expected_log: str,
) -> None:
    async def _fake_is_postgres() -> bool:
        return False

    async def _allow_admin_scope(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        admin_rate_limits,
        "_get_is_postgres_backend_fn",
        lambda: _fake_is_postgres,
    )
    monkeypatch.setattr(admin_rate_limits, "_enforce_admin_user_scope", _allow_admin_scope)

    await _assert_rate_limits_log_sanitized(
        call_factory,
        expected_log=expected_log,
        raw_marker="rate limits backend exploded",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_admin_rate_limits_reads_sqlite_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_is_postgres() -> bool:
        return False

    monkeypatch.setattr(
        admin_rate_limits,
        "_get_is_postgres_backend_fn",
        lambda: _fake_is_postgres,
    )
    db = _SqliteDbStub()

    response = await admin_rate_limits.list_admin_rate_limits(db=db)

    assert [item.model_dump() for item in response] == [  # nosec B101
        {
            "scope": "role",
            "id": 7,
            "resource": "/api/v1/chat/completions",
            "limit_per_min": 30,
            "burst": 5,
        },
        {
            "scope": "user",
            "id": 11,
            "resource": "/api/v1/rag/search",
            "limit_per_min": 12,
            "burst": 2,
        },
    ]
    assert len(db.execute_calls) == 2  # nosec B101


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_admin_rate_limits_reads_postgres_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_is_postgres() -> bool:
        return True

    monkeypatch.setattr(
        admin_rate_limits,
        "_get_is_postgres_backend_fn",
        lambda: _fake_is_postgres,
    )
    db = _PostgresDbStub()

    response = await admin_rate_limits.list_admin_rate_limits(db=db)

    assert [item.model_dump() for item in response] == [  # nosec B101
        {
            "scope": "role",
            "id": 3,
            "resource": "/api/v1/media/search",
            "limit_per_min": 60,
            "burst": 10,
        },
        {
            "scope": "user",
            "id": 9,
            "resource": "/api/v1/media/ingest/jobs",
            "limit_per_min": 6,
            "burst": 1,
        },
    ]
    assert len(db.fetch_calls) == 2  # nosec B101


class _FetchDbStub:
    def __init__(self) -> None:
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((str(query), tuple(args)))
        normalized = str(query).lower()
        if "from rbac_user_rate_limits" in normalized:
            return [
                {
                    "resource": "/api/v1/rag/search",
                    "limit_per_min": 5,
                    "burst": 1,
                }
            ]
        if "from rbac_role_rate_limits" in normalized:
            return [
                {
                    "resource": "/api/v1",
                    "limit_per_min": 20,
                    "burst": 4,
                    "role_name": "moderator",
                }
            ]
        raise AssertionError(f"Unexpected query: {query!r}")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_simulate_rate_limit_uses_fetch_and_prefers_matching_user_limit() -> None:
    db = _FetchDbStub()

    response = await admin_rate_limits.simulate_rate_limit(
        payload=admin_rate_limits.RateLimitSimRequest(
            user_id=11,
            endpoint="/api/v1/rag/search/query",
        ),
        principal=_platform_admin_principal(),
        db=db,
    )

    assert response.limit_source == "user"  # nosec B101
    assert response.effective_limit_per_min == 5  # nosec B101
    assert response.effective_burst == 1  # nosec B101
    assert response.would_allow is True  # nosec B101
    assert len(db.fetch_calls) == 2  # nosec B101


def test_matches_endpoint_requires_exact_or_path_boundary_match() -> None:
    row = {"resource": "/api/v1/users"}

    assert admin_rate_limits_service._matches_endpoint(row, "/api/v1/users") is True  # nosec B101
    assert admin_rate_limits_service._matches_endpoint(row, "/api/v1/users/123") is True  # nosec B101
    assert admin_rate_limits_service._matches_endpoint(row, "/api/v1/users-export") is False  # nosec B101
    assert admin_rate_limits_service._matches_endpoint({"resource": ""}, "/api/v1/users") is False  # nosec B101
