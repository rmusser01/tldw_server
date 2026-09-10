from __future__ import annotations

import asyncio

import pytest

from tldw_Server_API.app.core.Infrastructure import provider_registry as provider_registry_module
from tldw_Server_API.app.core.Infrastructure.provider_registry import (
    ProviderRegistryBase,
    ProviderRegistryConfig,
    ProviderStatus,
)

pytestmark = pytest.mark.unit


class _AsyncAdapter:
    def __init__(self) -> None:
        self.ready = True


@pytest.mark.asyncio
async def test_get_adapter_async_uses_async_materializer_and_cache() -> None:
    calls = {"count": 0}

    async def _materialize(provider_name: str, spec: object) -> object:
        calls["count"] += 1
        if isinstance(spec, type):
            return spec()
        return spec

    registry: ProviderRegistryBase[object] = ProviderRegistryBase(
        adapter_materializer_async=_materialize,
        adapter_validator=lambda adapter: isinstance(adapter, _AsyncAdapter),
    )
    registry.register_adapter("async", _AsyncAdapter)

    adapter1 = await registry.get_adapter_async("async")
    adapter2 = await registry.get_adapter_async("async")

    assert isinstance(adapter1, _AsyncAdapter)
    assert adapter2 is adapter1
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_get_adapter_async_respects_retry_window_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}
    now = {"value": 1000.0}
    monkeypatch.setattr(provider_registry_module.time, "time", lambda: now["value"])

    async def _flaky_materialize(provider_name: str, spec: object) -> object:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("boom")
        return _AsyncAdapter()

    registry: ProviderRegistryBase[object] = ProviderRegistryBase(
        config=ProviderRegistryConfig(failure_retry_seconds=0.02),
        adapter_materializer_async=_flaky_materialize,
        adapter_validator=lambda adapter: isinstance(adapter, _AsyncAdapter),
    )
    registry.register_adapter("flaky", _AsyncAdapter)

    assert await registry.get_adapter_async("flaky") is None
    assert registry.get_status("flaky") == ProviderStatus.FAILED

    # Retry window active: should not re-attempt materialization yet.
    assert await registry.get_adapter_async("flaky") is None
    assert calls["count"] == 1

    now["value"] += 0.03

    adapter = await registry.get_adapter_async("flaky")
    assert isinstance(adapter, _AsyncAdapter)
    assert calls["count"] == 2
    assert registry.get_status("flaky") == ProviderStatus.ENABLED


@pytest.mark.asyncio
async def test_get_adapter_async_falls_back_to_sync_materialization() -> None:
    registry: ProviderRegistryBase[object] = ProviderRegistryBase(
        adapter_validator=lambda adapter: isinstance(adapter, _AsyncAdapter)
    )
    registry.register_adapter("sync-class", _AsyncAdapter)

    adapter = await registry.get_adapter_async("sync-class")

    assert isinstance(adapter, _AsyncAdapter)
    assert registry.get_status("sync-class") == ProviderStatus.ENABLED


@pytest.mark.asyncio
async def test_get_adapter_async_does_not_cache_superseded_registration() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class _FirstAdapter(_AsyncAdapter):
        pass

    class _SecondAdapter(_AsyncAdapter):
        pass

    async def _materialize(provider_name: str, spec: object) -> object:
        if spec is _FirstAdapter:
            started.set()
            await release.wait()
        assert isinstance(spec, type)
        return spec()

    registry: ProviderRegistryBase[object] = ProviderRegistryBase(
        adapter_materializer_async=_materialize,
        adapter_validator=lambda adapter: isinstance(adapter, _AsyncAdapter),
    )
    registry.register_adapter("async", _FirstAdapter)
    first_task = asyncio.create_task(registry.get_adapter_async("async"))
    await started.wait()

    registry.register_adapter("async", _SecondAdapter)
    release.set()

    assert await first_task is None
    assert registry.get_cached_adapters() == {}
    second = await registry.get_adapter_async("async")
    assert isinstance(second, _SecondAdapter)
    assert registry.get_status("async") == ProviderStatus.ENABLED


@pytest.mark.asyncio
async def test_async_stale_materialization_disposer_failure_is_logged_once() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    discarded: list[tuple[str, object]] = []
    logged_messages: list[str] = []

    class _FirstAdapter(_AsyncAdapter):
        pass

    class _SecondAdapter(_AsyncAdapter):
        pass

    async def _materialize(provider_name: str, spec: object) -> object:
        if spec is _FirstAdapter:
            started.set()
            await release.wait()
        assert isinstance(spec, type)
        return spec()

    async def _discard(provider_name: str, adapter: object) -> None:
        discarded.append((provider_name, adapter))
        raise RuntimeError("discard callback secret must not be logged")

    registry: ProviderRegistryBase[object] = ProviderRegistryBase(
        adapter_materializer_async=_materialize,
        adapter_disposer_async=_discard,
        adapter_validator=lambda adapter: isinstance(adapter, _AsyncAdapter),
    )
    registry.register_adapter("async", _FirstAdapter)
    first_task = asyncio.create_task(registry.get_adapter_async("async"))
    await started.wait()
    registry.register_adapter("async", _SecondAdapter)
    release.set()

    sink_id = provider_registry_module.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        assert await first_task is None
    finally:
        provider_registry_module.logger.remove(sink_id)

    assert len(discarded) == 1
    assert discarded[0][0] == "async"
    assert isinstance(discarded[0][1], _FirstAdapter)
    assert registry.get_cached_adapters() == {}
    assert any("RuntimeError" in message for message in logged_messages)
    assert all("discard callback secret" not in message for message in logged_messages)
