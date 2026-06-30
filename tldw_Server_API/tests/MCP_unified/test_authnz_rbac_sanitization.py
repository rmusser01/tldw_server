"""Regression tests for MCP AuthNZ RBAC log sanitization."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.MCP_unified.auth import authnz_rbac
from tldw_Server_API.app.core.MCP_unified.auth.authnz_rbac import AuthNZRBAC
from tldw_Server_API.app.core.MCP_unified.auth.rbac import Action, Resource


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, message, *args, **kwargs) -> None:
        self.messages.append(" ".join([str(message), *(str(arg) for arg in args)]))

    def warning(self, message, *args, **kwargs) -> None:
        self.messages.append(" ".join([str(message), *(str(arg) for arg in args)]))


class _ExplodingPool:
    async def fetchone(self, *_args, **_kwargs):
        raise RuntimeError("rbac backend exploded token=SECRET path=/private/rbac.db")

    async def execute(self, *_args, **_kwargs):
        raise RuntimeError("rbac backend exploded token=SECRET path=/private/rbac.db")


def _assert_logs_sanitized(logger_stub: _LoggerStub, expected_message: str) -> None:
    rendered = "\n".join(logger_stub.messages)
    assert expected_message in rendered
    assert "rbac backend exploded" not in rendered
    assert "SECRET" not in rendered
    assert "/private/rbac.db" not in rendered


@pytest.mark.asyncio
async def test_check_permission_failure_log_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(authnz_rbac, "logger", logger_stub)

    allowed = await AuthNZRBAC(db_pool=_ExplodingPool()).check_permission(
        user_id="1",
        resource=Resource.MEDIA,
        action=Action.READ,
    )

    assert allowed is False
    _assert_logs_sanitized(logger_stub, "AuthNZ RBAC check failed")


@pytest.mark.asyncio
async def test_ensure_permission_exists_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(authnz_rbac, "logger", logger_stub)

    await AuthNZRBAC(db_pool=_ExplodingPool())._ensure_permission_exists(
        "tools.execute:SECRET_TOOL",
    )

    _assert_logs_sanitized(logger_stub, "Ensure permission exists failed")
    assert "SECRET_TOOL" not in "\n".join(logger_stub.messages)
