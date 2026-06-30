from __future__ import annotations

import builtins
import sys
import types
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin import admin_tools

pytestmark = pytest.mark.unit


class _FakeCollector:
    def get_internal_metrics(self, *, period_seconds: int) -> dict[str, object]:
        assert period_seconds == 3600
        return {
            "module_alpha_tools_call": {
                "labels": [
                    {"labels": {"module": "alpha", "tool": "search"}, "count": 2, "avg": 0.1},
                    {"labels": {"module": "alpha", "tool": "search"}, "count": 1, "avg": 0.2},
                    {"labels": {"module": "alpha", "tool": "summarize"}, "count": 1, "avg": 0.4},
                    {"labels": {"module": "beta", "tool": "lookup"}, "count": 3, "avg": 0.05},
                ]
            }
        }


class _LoggerStub:
    def __init__(self) -> None:
        self.error_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.warning_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_records.append((message, args, kwargs))

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warning_records.append((message, args, kwargs))


def _render_log_records(records: list[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> str:
    rendered: list[str] = []
    for message, args, kwargs in records:
        try:
            rendered.append(message.format(*args, **kwargs))
        except (IndexError, KeyError, ValueError):
            rendered.append(message)
        rendered.extend(repr(arg) for arg in args)
        rendered.extend(f"{key}={value!r}" for key, value in kwargs.items())
    return "\n".join(rendered)


def _assert_sanitized_error_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.error_records == [(expected_message, (), {})]
    joined = _render_log_records(logger_stub.error_records)
    assert "backend exploded" not in joined
    assert "/private/" not in joined
    assert "secret-catalog" not in joined
    assert "secret.tool" not in joined


def _assert_sanitized_warning_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.warning_records == [(expected_message, (), {})]
    joined = _render_log_records(logger_stub.warning_records)
    assert "backend exploded" not in joined
    assert "/private/" not in joined


@pytest.mark.asyncio
async def test_get_mcp_tool_usage_aggregates_metrics_from_label_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.SimpleNamespace(get_metrics_collector=lambda: _FakeCollector())
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.MCP_unified.monitoring.metrics",
        fake_module,
    )

    result = await admin_tools.get_mcp_tool_usage(period_seconds=3600)

    assert result.modules["alpha"].calls == 4
    assert result.modules["alpha"].avg_latency_ms == 200.0
    assert result.modules["beta"].calls == 3
    assert result.modules["beta"].avg_latency_ms == 50.0
    assert result.tools["alpha.search"].calls == 3
    assert result.tools["alpha.search"].avg_latency_ms == 133.3
    assert result.tools["alpha.summarize"].calls == 1
    assert result.tools["alpha.summarize"].avg_latency_ms == 400.0


@pytest.mark.asyncio
async def test_get_mcp_tool_usage_sanitizes_metrics_import_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    real_import = builtins.__import__

    def _raise_metrics_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "tldw_Server_API.app.core.MCP_unified.monitoring.metrics":
            raise ImportError("metrics backend exploded at /private/metrics.py")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(
        sys.modules,
        "tldw_Server_API.app.core.MCP_unified.monitoring.metrics",
        raising=False,
    )
    monkeypatch.setattr(builtins, "__import__", _raise_metrics_import)
    monkeypatch.setattr(admin_tools, "logger", logger_stub)

    result = await admin_tools.get_mcp_tool_usage(period_seconds=3600)

    assert result.period_seconds == 3600
    assert result.modules == {}
    assert result.tools == {}
    _assert_sanitized_warning_log(logger_stub, "MCP metrics module unavailable")


@pytest.mark.asyncio
async def test_list_tool_catalogs_sanitizes_generic_error_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()

    async def _raise_list(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("tool catalog list backend exploded at /private/admin-tools.db")

    monkeypatch.setattr(admin_tools, "logger", logger_stub)
    monkeypatch.setattr(admin_tools.admin_tool_catalog_service, "list_tool_catalogs", _raise_list)

    with pytest.raises(HTTPException) as exc_info:
        await admin_tools.list_tool_catalogs(db=object())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list tool catalogs"
    _assert_sanitized_error_log(logger_stub, "Failed to list tool catalogs")


@pytest.mark.asyncio
async def test_create_tool_catalog_sanitizes_generic_error_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()

    async def _raise_create(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("tool catalog create backend exploded at /private/admin-tools.db")

    monkeypatch.setattr(admin_tools, "logger", logger_stub)
    monkeypatch.setattr(admin_tools.admin_tool_catalog_service, "create_tool_catalog", _raise_create)

    with pytest.raises(HTTPException) as exc_info:
        await admin_tools.create_tool_catalog(
            payload=admin_tools.ToolCatalogCreateRequest(name="secret-catalog"),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create tool catalog"
    _assert_sanitized_error_log(logger_stub, "Failed to create tool catalog")


@pytest.mark.asyncio
async def test_delete_tool_catalog_sanitizes_generic_error_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()

    async def _get_catalog(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"id": 42, "name": "secret-catalog"}

    async def _raise_delete(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("tool catalog delete backend exploded at /private/admin-tools.db")

    monkeypatch.setattr(admin_tools, "logger", logger_stub)
    monkeypatch.setattr(admin_tools.admin_tool_catalog_service, "get_tool_catalog", _get_catalog)
    monkeypatch.setattr(admin_tools.admin_tool_catalog_service, "delete_tool_catalog", _raise_delete)

    with pytest.raises(HTTPException) as exc_info:
        await admin_tools.delete_tool_catalog(catalog_id=42, db=object())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to delete tool catalog"
    _assert_sanitized_error_log(logger_stub, "Failed to delete tool catalog")


@pytest.mark.asyncio
async def test_list_tool_catalog_entries_sanitizes_generic_error_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()

    async def _get_catalog(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"id": 42, "name": "secret-catalog"}

    async def _raise_list_entries(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("tool catalog entries backend exploded at /private/admin-tools.db")

    monkeypatch.setattr(admin_tools, "logger", logger_stub)
    monkeypatch.setattr(admin_tools.admin_tool_catalog_service, "get_tool_catalog", _get_catalog)
    monkeypatch.setattr(admin_tools.admin_tool_catalog_service, "list_tool_catalog_entries", _raise_list_entries)

    with pytest.raises(HTTPException) as exc_info:
        await admin_tools.list_tool_catalog_entries(catalog_id=42, db=object())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list tool catalog entries"
    _assert_sanitized_error_log(logger_stub, "Failed to list tool catalog entries")


@pytest.mark.asyncio
async def test_add_tool_catalog_entry_sanitizes_generic_error_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()

    async def _get_catalog(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"id": 42, "name": "secret-catalog"}

    async def _raise_add_entry(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("tool catalog entry add backend exploded at /private/admin-tools.db")

    monkeypatch.setattr(admin_tools, "logger", logger_stub)
    monkeypatch.setattr(admin_tools.admin_tool_catalog_service, "get_tool_catalog", _get_catalog)
    monkeypatch.setattr(admin_tools.admin_tool_catalog_service, "add_tool_catalog_entry", _raise_add_entry)

    with pytest.raises(HTTPException) as exc_info:
        await admin_tools.add_tool_catalog_entry(
            catalog_id=42,
            payload=admin_tools.ToolCatalogEntryCreateRequest(tool_name="secret.tool"),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to add tool catalog entry"
    _assert_sanitized_error_log(logger_stub, "Failed to add tool catalog entry")


@pytest.mark.asyncio
async def test_delete_tool_catalog_entry_sanitizes_generic_error_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()

    async def _get_catalog(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"id": 42, "name": "secret-catalog"}

    async def _raise_delete_entry(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("tool catalog entry delete backend exploded at /private/admin-tools.db")

    monkeypatch.setattr(admin_tools, "logger", logger_stub)
    monkeypatch.setattr(admin_tools.admin_tool_catalog_service, "get_tool_catalog", _get_catalog)
    monkeypatch.setattr(admin_tools.admin_tool_catalog_service, "delete_tool_catalog_entry", _raise_delete_entry)

    with pytest.raises(HTTPException) as exc_info:
        await admin_tools.delete_tool_catalog_entry(catalog_id=42, tool_name="secret.tool", db=object())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to delete tool catalog entry"
    _assert_sanitized_error_log(logger_stub, "Failed to delete tool catalog entry")
