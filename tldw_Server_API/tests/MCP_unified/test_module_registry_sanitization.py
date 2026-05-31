from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules import registry as registry_mod
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.registry import (
    ModuleRegistration,
    ModuleRegistry,
    ModuleStatus,
)


class _LoggerCapture:
    def __init__(self, level: str = "DEBUG") -> None:
        self.messages: list[str] = []
        self._sink_id = registry_mod.logger.add(
            lambda message: self.messages.append(str(message.record.get("message") or "")),
            level=level,
        )

    def close(self) -> None:
        registry_mod.logger.remove(self._sink_id)

    @property
    def rendered(self) -> str:
        return "\n".join(self.messages)


class _InitFailsModule(BaseModule):
    async def initialize(self) -> None:
        raise RuntimeError("init leaked /private/module-init.db with sk-init-secret")

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return []

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        return {"tool": tool_name, "arguments": arguments, "context": context}


class _RegistryUpdateFailsModule:
    async def get_tools(self) -> list[dict[str, Any]]:
        raise RuntimeError("registry update leaked /private/tools.json with sk-tools-secret")


class _ShutdownFailsModule:
    async def shutdown(self) -> None:
        raise RuntimeError("shutdown leaked /private/shutdown.db with sk-shutdown-secret")


class _OperationFailsModule:
    def __init__(self, detail: str) -> None:
        self.detail = detail

    async def some_operation(self) -> None:
        return None

    async def execute_with_circuit_breaker(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(self.detail)


def _registration(module_id: str, module: Any) -> ModuleRegistration:
    return ModuleRegistration(
        module_id=module_id,
        module_type=type(module),
        module_instance=module,
        config=ModuleConfig(name=module_id, description=f"{module_id} module"),
        status=ModuleStatus.ACTIVE,
        registered_at=datetime.now(timezone.utc),
    )


def _assert_no_sensitive_details(rendered_logs: str, sensitive_detail: str) -> None:
    assert sensitive_detail not in rendered_logs
    assert "/private/" not in rendered_logs
    assert "sk-" not in rendered_logs


@pytest.mark.asyncio
async def test_health_monitor_loop_error_log_sanitizes_exception_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_detail = "health loop leaked /private/health.db with sk-health-secret"
    registry = ModuleRegistry()

    async def _raise_health_error() -> None:
        raise RuntimeError(sensitive_detail)

    async def _cancel_after_error_sleep(_delay: float) -> None:
        raise asyncio.CancelledError

    capture = _LoggerCapture(level="ERROR")
    monkeypatch.setattr(registry, "check_all_health", _raise_health_error)
    monkeypatch.setattr(registry_mod.asyncio, "sleep", _cancel_after_error_sleep)

    try:
        with pytest.raises(asyncio.CancelledError):
            await registry._health_monitor_loop()
    finally:
        capture.close()

    assert "Error in health monitor loop" in capture.rendered
    _assert_no_sensitive_details(capture.rendered, sensitive_detail)


@pytest.mark.asyncio
async def test_register_module_failure_logs_sanitize_exception_detail() -> None:
    sensitive_detail = "init leaked /private/module-init.db with sk-init-secret"
    registry = ModuleRegistry()
    capture = _LoggerCapture(level="ERROR")

    try:
        await registry.register_module(
            "init_module",
            _InitFailsModule,
            ModuleConfig(name="init_module", description="Init module"),
        )
    finally:
        capture.close()

    status = await registry.get_module_status("init_module")
    assert status is not None
    assert status["status"] == "error"
    assert "Failed to initialize module init_module" in capture.rendered
    assert "Module initialization failed: init_module" in capture.rendered
    _assert_no_sensitive_details(capture.rendered, sensitive_detail)


@pytest.mark.asyncio
async def test_update_registries_failure_log_sanitizes_exception_detail() -> None:
    sensitive_detail = "registry update leaked /private/tools.json with sk-tools-secret"
    registry = ModuleRegistry()
    capture = _LoggerCapture(level="ERROR")

    try:
        await registry._update_registries("catalog_module", _RegistryUpdateFailsModule())
    finally:
        capture.close()

    assert "Failed to update registries for catalog_module" in capture.rendered
    _assert_no_sensitive_details(capture.rendered, sensitive_detail)


@pytest.mark.asyncio
async def test_refresh_module_registries_preserves_existing_mappings_when_rebuild_fails() -> None:
    registry = ModuleRegistry()
    registry._modules["catalog_module"] = _registration("catalog_module", _RegistryUpdateFailsModule())
    registry._tool_registry["old.tool"] = "catalog_module"

    refreshed = await registry.refresh_module_registries("catalog_module")

    assert refreshed is False
    assert registry.get_module_id_for_tool("old.tool") == "catalog_module"


@pytest.mark.asyncio
async def test_unregister_module_shutdown_log_sanitizes_exception_detail() -> None:
    sensitive_detail = "shutdown leaked /private/shutdown.db with sk-shutdown-secret"
    registry = ModuleRegistry()
    registry._modules["shutdown_module"] = _registration("shutdown_module", _ShutdownFailsModule())
    capture = _LoggerCapture(level="ERROR")

    try:
        await registry.unregister_module("shutdown_module")
    finally:
        capture.close()

    assert "Error shutting down module shutdown_module" in capture.rendered
    _assert_no_sensitive_details(capture.rendered, sensitive_detail)


@pytest.mark.asyncio
async def test_shutdown_all_failure_log_sanitizes_exception_detail() -> None:
    sensitive_detail = "shutdown leaked /private/shutdown.db with sk-shutdown-secret"
    registry = ModuleRegistry()
    registry._modules["shutdown_module"] = _registration("shutdown_module", _ShutdownFailsModule())
    capture = _LoggerCapture(level="ERROR")

    try:
        await registry.shutdown_all()
    finally:
        capture.close()

    assert "Error shutting down module shutdown_module" in capture.rendered
    _assert_no_sensitive_details(capture.rendered, sensitive_detail)


@pytest.mark.asyncio
async def test_failover_logs_sanitize_exception_detail_and_preserve_module_ids() -> None:
    primary_detail = "primary leaked /private/primary.db with sk-primary-secret"
    fallback_detail = "fallback leaked /private/fallback.db with sk-fallback-secret"
    registry = ModuleRegistry()
    registry._modules["primary_module"] = _registration(
        "primary_module",
        _OperationFailsModule(primary_detail),
    )
    registry._modules["fallback_module"] = _registration(
        "fallback_module",
        _OperationFailsModule(fallback_detail),
    )
    capture = _LoggerCapture(level="WARNING")

    try:
        with pytest.raises(Exception, match="All modules failed"):
            await registry.execute_with_failover(
                "primary_module",
                ["fallback_module"],
                "some_operation",
            )
    finally:
        capture.close()

    assert "Primary module primary_module failed" in capture.rendered
    assert "Fallback module fallback_module failed" in capture.rendered
    _assert_no_sensitive_details(capture.rendered, primary_detail)
    _assert_no_sensitive_details(capture.rendered, fallback_detail)
