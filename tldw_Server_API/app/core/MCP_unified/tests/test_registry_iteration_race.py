import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.registry import ModuleRegistry


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _GateLookupModule(BaseModule):
    gate_enter: asyncio.Event | None = None
    gate_release: asyncio.Event | None = None
    gate_kind: str | None = None

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return []

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        return None

    async def _maybe_gate(self, kind: str) -> None:
        enter = self.__class__.gate_enter
        release = self.__class__.gate_release
        gate_kind = self.__class__.gate_kind
        if gate_kind == kind and enter is not None and release is not None and not enter.is_set():
            enter.set()
            await release.wait()

    async def has_tool(self, tool_name: str) -> bool:
        await self._maybe_gate("tool")
        return False

    async def has_resource(self, uri: str) -> bool:
        await self._maybe_gate("resource")
        return False

    async def has_prompt(self, name: str) -> bool:
        await self._maybe_gate("prompt")
        return False


async def _run_lookup_race(method_name: str, lookup_arg: str) -> None:
    registry = ModuleRegistry()
    await registry.register_module("race_mod_a", _GateLookupModule, ModuleConfig(name="race_mod_a"))
    await registry.register_module("race_mod_b", _GateLookupModule, ModuleConfig(name="race_mod_b"))

    enter = asyncio.Event()
    release = asyncio.Event()
    _GateLookupModule.gate_enter = enter
    _GateLookupModule.gate_release = release
    _GateLookupModule.gate_kind = method_name

    lookup_fn = getattr(registry, f"find_module_for_{method_name}")
    lookup_task = asyncio.create_task(lookup_fn(lookup_arg))

    await asyncio.wait_for(enter.wait(), timeout=2.0)
    await registry.register_module("race_mod_new", _GateLookupModule, ModuleConfig(name="race_mod_new"))
    release.set()

    result = await asyncio.wait_for(lookup_task, timeout=2.0)
    _ensure(result is None, f"Lookup race should not resolve a missing module: {result!r}")

    _GateLookupModule.gate_enter = None
    _GateLookupModule.gate_release = None
    _GateLookupModule.gate_kind = None


@pytest.mark.asyncio
async def test_find_module_for_tool_handles_registry_mutation_during_iteration():
    await _run_lookup_race("tool", "missing_tool")


@pytest.mark.asyncio
async def test_find_module_for_resource_handles_registry_mutation_during_iteration():
    await _run_lookup_race("resource", "missing://resource")


@pytest.mark.asyncio
async def test_find_module_for_prompt_handles_registry_mutation_during_iteration():
    await _run_lookup_race("prompt", "missing_prompt")


@pytest.mark.asyncio
async def test_check_all_health_handles_registry_mutation_during_iteration():
    registry = ModuleRegistry()
    await registry.register_module("race_mod_a", _GateLookupModule, ModuleConfig(name="race_mod_a"))

    enter = asyncio.Event()
    release = asyncio.Event()

    async def _health() -> dict[str, bool]:
        enter.set()
        await release.wait()
        return {"ok": True}

    module = await registry.get_module("race_mod_a")
    _ensure(module is not None, "race_mod_a failed to register")
    module.check_health = _health  # type: ignore[assignment]
    module._health.last_check = None

    task = asyncio.create_task(registry.check_all_health())
    await asyncio.wait_for(enter.wait(), timeout=2.0)
    await registry.register_module("race_mod_b", _GateLookupModule, ModuleConfig(name="race_mod_b"))
    release.set()

    result = await asyncio.wait_for(task, timeout=2.0)
    _ensure("race_mod_a" in result, f"Health result should include the in-flight module: {result!r}")


@pytest.mark.asyncio
async def test_list_registrations_handles_registry_mutation_during_iteration():
    registry = ModuleRegistry()
    await registry.register_module("race_mod_a", _GateLookupModule, ModuleConfig(name="race_mod_a"))

    enter = asyncio.Event()
    release = asyncio.Event()

    original_snapshot = registry._snapshot_registrations

    async def _snapshot() -> dict[str, Any]:
        enter.set()
        await release.wait()
        return await original_snapshot()

    registry._snapshot_registrations = _snapshot  # type: ignore[assignment]

    task = asyncio.create_task(registry.list_registrations())
    await asyncio.wait_for(enter.wait(), timeout=2.0)
    await registry.register_module("race_mod_b", _GateLookupModule, ModuleConfig(name="race_mod_b"))
    release.set()

    result = await asyncio.wait_for(task, timeout=2.0)
    _ensure(
        any(item and item.get("module_id") == "race_mod_a" for item in result),
        f"List registrations should include the in-flight module snapshot: {result!r}",
    )
