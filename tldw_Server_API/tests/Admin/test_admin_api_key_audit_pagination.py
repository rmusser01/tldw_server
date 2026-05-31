from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_api_keys_service


class _FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    async def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeSQLiteAuditDB:
    def __init__(self) -> None:
        self.queries: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, params: tuple[Any, ...]) -> _FakeCursor:
        self.queries.append((query, params))
        query_lower = query.lower()
        if "from api_keys" in query_lower:
            return _FakeCursor([(42,)])
        if "count(*)" in query_lower:
            return _FakeCursor([(2,)])
        if "from api_key_audit_log" in query_lower:
            return _FakeCursor(
                [
                    (
                        11,
                        7,
                        "rotated",
                        42,
                        "127.0.0.1",
                        "pytest",
                        {"reason": "scheduled"},
                        None,
                    )
                ]
            )
        raise AssertionError(f"Unexpected query: {query}")


@pytest.mark.asyncio
async def test_api_key_audit_log_returns_offset_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admin API-key audit log exposes canonical offset pagination metadata."""

    async def _allow_scope(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _is_sqlite() -> bool:
        return False

    monkeypatch.setattr(
        admin_api_keys_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    db = _FakeSQLiteAuditDB()
    principal = AuthPrincipal(kind="user", user_id=1, is_admin=True)

    response = await admin_api_keys_service.get_api_key_audit_log(
        principal,
        7,
        limit=1,
        offset=1,
        db=db,
        is_pg_fn=_is_sqlite,
    )

    assert response.total == 2
    assert response.limit == 1
    assert response.offset == 1
    assert response.pagination.model_dump() == {
        "mode": "offset",
        "limit": 1,
        "offset": 1,
        "total": 2,
        "has_more": False,
        "next_offset": None,
    }
    assert response.has_more is False
    assert response.next_offset is None
