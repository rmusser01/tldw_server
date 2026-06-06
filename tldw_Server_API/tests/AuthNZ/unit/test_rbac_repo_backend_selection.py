from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.rbac_repo import AuthnzRbacRepo
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType


pytestmark = pytest.mark.unit


class _BackendResult:
    rows = [{"id": 1, "name": "admin", "description": "Admin", "is_system": False}]


class _RoleBackend:
    def __init__(self, backend_type: BackendType) -> None:
        self.backend_type = backend_type
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, query: str, params: tuple[Any, ...]) -> _BackendResult:
        self.execute_calls.append((str(query), params))
        return _BackendResult()


class _UserDb:
    def __init__(self, backend_type: BackendType) -> None:
        self.backend = _RoleBackend(backend_type)


def test_get_user_roles_postgres_uses_boolean_is_system_default() -> None:
    db = _UserDb(BackendType.POSTGRESQL)
    repo = AuthnzRbacRepo()
    repo.__dict__["_db"] = db

    roles = repo.get_user_roles(42)

    assert roles[0]["name"] == "admin"
    query, params = db.backend.execute_calls[0]
    assert params == (42,)
    assert "COALESCE(r.is_system, FALSE)" in query
    assert "COALESCE(r.is_system, 0)" not in query


def test_get_user_roles_sqlite_keeps_integer_is_system_default() -> None:
    db = _UserDb(BackendType.SQLITE)
    repo = AuthnzRbacRepo()
    repo.__dict__["_db"] = db

    repo.get_user_roles(42)

    query, _params = db.backend.execute_calls[0]
    assert "COALESCE(r.is_system, 0)" in query
