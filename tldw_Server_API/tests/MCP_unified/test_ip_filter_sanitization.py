"""Regression tests for MCP IP filter sanitizer behavior."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.MCP_unified.security import ip_filter
from tldw_Server_API.app.core.MCP_unified.security.ip_filter import IPAccessController


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, message, *args, **kwargs) -> None:
        self.messages.append(" ".join([str(message), *(str(arg) for arg in args)]))


class _BadForwardedFor:
    def split(self, *_args, **_kwargs):
        raise RuntimeError("xff parse exploded token=SECRET path=/private/proxy/header")


def test_resolve_client_ip_parse_failure_log_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()
    controller = IPAccessController(
        allowed=[],
        blocked=[],
        trust_x_forwarded_for=True,
        trusted_proxy_depth=0,
        trusted_proxies=[],
    )

    monkeypatch.setattr(ip_filter, "logger", logger_stub)

    assert controller.resolve_client_ip("127.0.0.1", _BadForwardedFor()) == "127.0.0.1"

    rendered = "\n".join(logger_stub.messages)
    assert "Failed to parse X-Forwarded-For header" in rendered
    assert "xff parse exploded" not in rendered
    assert "SECRET" not in rendered
    assert "/private/proxy/header" not in rendered
