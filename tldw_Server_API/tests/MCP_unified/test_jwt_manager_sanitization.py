from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from jose import JWTError

from tldw_Server_API.app.core.MCP_unified.auth import jwt_manager
from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import JWTManager

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.error_messages: list[str] = []
        self.warning_messages: list[str] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_messages.append(message.format(*args) if args else message)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warning_messages.append(message.format(*args) if args else message)


def test_verify_password_sanitizes_verification_exception_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _LoggerStub()
    manager = JWTManager.__new__(JWTManager)

    def _raise_leaky_verify(_plain_password: str, _hashed_password: str) -> bool:
        raise ValueError("bad hash for /private/mcp/users.db using secret-password")

    monkeypatch.setattr(jwt_manager, "logger", logger)
    monkeypatch.setattr(jwt_manager.pwd_context, "verify", _raise_leaky_verify)

    assert manager.verify_password("secret-password", "bad-hash") is False
    assert logger.error_messages == ["Password verification failed"]
    assert "/private/mcp/users.db" not in logger.error_messages[0]
    assert "secret-password" not in logger.error_messages[0]
    assert "ValueError" not in logger.error_messages[0]


def test_verify_token_sanitizes_invalid_token_exception_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _LoggerStub()
    manager = JWTManager.__new__(JWTManager)
    manager.config = SimpleNamespace(jwt_algorithm="HS256")
    manager._revoked_tokens = set()

    def _raise_leaky_decode(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise JWTError("invalid token raw.jwt.secret from /private/mcp/tokens.db")

    monkeypatch.setattr(jwt_manager, "logger", logger)
    monkeypatch.setattr(jwt_manager.jwt, "decode", _raise_leaky_decode)
    monkeypatch.setattr(manager, "_get_secret_key", lambda: "test-secret")

    with pytest.raises(HTTPException) as excinfo:
        manager.verify_token("raw.jwt.secret")

    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == "Invalid token"
    assert logger.warning_messages == ["Invalid token"]
    assert "raw.jwt.secret" not in logger.warning_messages[0]
    assert "/private/mcp/tokens.db" not in logger.warning_messages[0]
    assert "JWTError" not in logger.warning_messages[0]
