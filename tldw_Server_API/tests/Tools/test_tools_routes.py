from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.errors.append(message)

    def warning(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.warnings.append(message)


@pytest.mark.unit
def test_tools_list(client_user_only):
    # Ensure tools route is enabled (config.txt sets enable = tools; env can override)
    os.environ.setdefault("ROUTES_ENABLE", "tools")

    r = client_user_only.get("/api/v1/tools")
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data, dict)
    assert "tools" in data and isinstance(data["tools"], list)


@pytest.mark.unit
def test_tools_execute_dry_run_when_available(client_user_only):
    os.environ.setdefault("ROUTES_ENABLE", "tools")
    # Promote deterministic module autoload in startup (Media module)
    os.environ.setdefault("TEST_MODE", "1")

    # List to pick a tool name if present
    r = client_user_only.get("/api/v1/tools")
    assert r.status_code == 200, r.text
    tools = r.json().get("tools", [])

    # If no tools registered, just assert the shape (environment-dependent)
    if not tools:
        assert isinstance(tools, list)
        return

    picked = None
    for t in tools:
        if isinstance(t, dict) and t.get("name"):
            picked = t["name"]
            break

    if picked:
        resp = client_user_only.post(
            "/api/v1/tools/execute",
            json={"tool_name": picked, "arguments": {}, "dry_run": True},
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload.get("ok") is True
        assert isinstance(payload.get("result"), dict)
        assert payload["result"].get("validated") is True


@pytest.mark.asyncio
async def test_list_tools_generic_failure_log_is_sanitized(monkeypatch):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import tools

    class _FailingToolExecutor:
        async def list_tools(self, *, user_id: str | None, client_id: str | None):  # noqa: ARG002
            raise RuntimeError("tools backend exploded at /private/tools.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(tools, "ToolExecutor", _FailingToolExecutor)
    monkeypatch.setattr(tools, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await tools.list_tools_endpoint(current_user=SimpleNamespace(id=1, username="alice"))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list tools"
    assert logger_stub.errors == ["tools.list failed"]
    assert "tools backend exploded" not in str(logger_stub.errors)
    assert "/private/tools.db" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_list_tools_parse_fallback_logs_are_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import tools

    class _MalformedToolExecutor:
        async def list_tools(self, *, user_id: str | None, client_id: str | None):  # noqa: ARG002
            return {
                "tools": [
                    {
                        "name": "secret.tool",
                        "description": "safe description",
                        "module": "secret.module",
                        "inputSchema": "/private/schema.json",
                        "canExecute": True,
                    },
                    {
                        "name": "dropped.tool",
                        "description": {"raw": "/private/description.json"},
                        "module": "secret.module",
                        "inputSchema": "/private/dropped-schema.json",
                        "canExecute": True,
                    },
                ]
            }

    logger_stub = _LoggerStub()
    monkeypatch.setattr(tools, "ToolExecutor", _MalformedToolExecutor)
    monkeypatch.setattr(tools, "logger", logger_stub)

    response = await tools.list_tools_endpoint(current_user=SimpleNamespace(id=1, username="alice"))

    assert [tool.name for tool in response.tools] == ["secret.tool"]
    assert logger_stub.warnings == [
        "Failed to parse tool info, falling back to best-effort mapping",
        "Failed to parse tool info, falling back to best-effort mapping",
    ]
    assert logger_stub.errors == ["Failed to best-effort map tool info"]
    rendered_logs = " ".join([*logger_stub.warnings, *logger_stub.errors])
    assert "secret.tool" not in rendered_logs
    assert "dropped.tool" not in rendered_logs
    assert "/private/" not in rendered_logs


@pytest.mark.asyncio
async def test_execute_tool_generic_failure_log_is_sanitized(monkeypatch):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import tools
    from tldw_Server_API.app.api.v1.schemas.tools import ExecuteToolRequest

    class _FailingToolExecutor:
        async def execute(self, **_kwargs):
            raise RuntimeError("tool backend exploded at /private/tool.sock")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(tools, "ToolExecutor", _FailingToolExecutor)
    monkeypatch.setattr(tools, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await tools.execute_tool_endpoint(
            req=ExecuteToolRequest(tool_name="demo.tool", arguments={}),
            current_user=SimpleNamespace(id=1, username="alice"),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Tool execution failed"
    assert logger_stub.errors == ["tools.execute failed"]
    assert "tool backend exploded" not in str(logger_stub.errors)
    assert "/private/tool.sock" not in str(logger_stub.errors)
