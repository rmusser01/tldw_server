from __future__ import annotations

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
