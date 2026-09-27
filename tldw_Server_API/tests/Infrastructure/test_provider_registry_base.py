from __future__ import annotations

import threading
import time

import pytest

from tldw_Server_API.app.core.Infrastructure.provider_registry import (
    ProviderRegistryBase,
    ProviderRegistryConfig,
    ProviderStatus,
)

pytestmark = pytest.mark.unit


class _DummyAdapter:
    def __init__(self) -> None:
        self.ready = True


class _InvalidAdapter:
    def __init__(self) -> None:
        self.ready = False


def test_register_and_cache_class_adapter() -> None:
    registry: ProviderRegistryBase[_DummyAdapter] = ProviderRegistryBase()
    registry.register_adapter("dummy", _DummyAdapter)

    adapter1 = registry.get_adapter("dummy")
    adapter2 = registry.get_adapter("dummy")

    assert adapter1 is not None
    assert adapter1 is adapter2
    assert registry.get_status("dummy") == ProviderStatus.ENABLED


def test_class_adapter_initializes_lazily_once_and_is_cached() -> None:
    state = {"init_count": 0}

    class _LazyAdapter:
        def __init__(self) -> None:
            state["init_count"] += 1
            self.ok = True

    registry: ProviderRegistryBase[object] = ProviderRegistryBase()
    registry.register_adapter("lazy", _LazyAdapter)

    assert state["init_count"] == 0
    first = registry.get_adapter("lazy")
    assert first is not None
    assert state["init_count"] == 1

    second = registry.get_adapter("lazy")
    assert second is first
    assert state["init_count"] == 1


def test_sync_get_adapter_materializes_once_under_concurrency() -> None:
    state = {"init_count": 0}
    state_lock = threading.Lock()
    start_event = threading.Event()
    results: list[object | None] = []
    errors: list[BaseException] = []

    class _SlowAdapter:
        def __init__(self) -> None:
            with state_lock:
                state["init_count"] += 1
            time.sleep(0.05)

    registry: ProviderRegistryBase[object] = ProviderRegistryBase()
    registry.register_adapter("slow", _SlowAdapter)

    def _worker() -> None:
        try:
            start_event.wait(timeout=1)
            results.append(registry.get_adapter("slow"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    start_event.set()
    for thread in threads:
        thread.join(timeout=2)

    assert errors == []
    assert len(results) == 6
    assert len({id(adapter) for adapter in results}) == 1
    assert state["init_count"] == 1


def test_sync_stale_materialization_calls_disposer_once() -> None:
    discarded: list[tuple[str, object]] = []

    class _FirstAdapter:
        pass

    class _SecondAdapter:
        pass

    def _materialize(provider_name: str, spec: object) -> object:
        assert isinstance(spec, type)
        adapter = spec()
        registry.register_adapter(provider_name, _SecondAdapter)
        return adapter

    registry: ProviderRegistryBase[object] = ProviderRegistryBase(
        adapter_materializer=_materialize,
        adapter_disposer=lambda provider_name, adapter: discarded.append(
            (provider_name, adapter)
        ),
    )
    registry.register_adapter("provider", _FirstAdapter)

    assert registry.get_adapter("provider") is None
    assert len(discarded) == 1
    assert discarded[0][0] == "provider"
    assert isinstance(discarded[0][1], _FirstAdapter)
    assert registry.get_cached_adapters() == {}


def test_register_dotted_path_adapter() -> None:
    registry: ProviderRegistryBase[object] = ProviderRegistryBase()
    registry.register_adapter("counter", "collections.Counter")

    adapter = registry.get_adapter("counter")

    assert adapter is not None
    assert adapter.__class__.__name__ == "Counter"


def test_register_instance_adapter_reuses_instance() -> None:
    instance = _DummyAdapter()
    registry: ProviderRegistryBase[_DummyAdapter] = ProviderRegistryBase()
    registry.register_adapter("dummy", instance)

    assert registry.get_adapter("dummy") is instance


def test_adapter_spec_validator_accepts_class_instance_and_dotted_path() -> None:
    def is_dummy_adapter(value: object) -> bool:
        if isinstance(value, type):
            return issubclass(value, _DummyAdapter) or value is list
        return isinstance(value, (_DummyAdapter, list))

    registry: ProviderRegistryBase[object] = ProviderRegistryBase(
        adapter_spec_validator=is_dummy_adapter
    )

    registry.register_adapter("dummy-class", _DummyAdapter)
    registry.register_adapter("dummy-instance", _DummyAdapter())
    registry.register_adapter(
        "dummy-dotted",
        "builtins.list",
    )

    assert isinstance(registry.get_adapter("dummy-class"), _DummyAdapter)
    assert isinstance(registry.get_adapter("dummy-instance"), _DummyAdapter)
    assert isinstance(registry.get_adapter("dummy-dotted"), list)


def test_adapter_spec_validator_rejects_invalid_spec_types() -> None:
    def is_dummy_adapter(value: object) -> bool:
        if isinstance(value, type):
            return issubclass(value, _DummyAdapter)
        return isinstance(value, _DummyAdapter)

    registry: ProviderRegistryBase[object] = ProviderRegistryBase(
        adapter_spec_validator=is_dummy_adapter
    )

    with pytest.raises(TypeError, match="Adapter spec is not valid"):
        registry.register_adapter("bad-class", _InvalidAdapter)

    with pytest.raises(TypeError, match="Adapter spec is not valid"):
        registry.register_adapter("bad-instance", _InvalidAdapter())

    with pytest.raises(TypeError, match="Adapter spec is not valid"):
        registry.register_adapter("bad-dotted", "collections.Counter")


def test_adapter_validator_rejects_invalid_instance_at_registration() -> None:
    registry: ProviderRegistryBase[object] = ProviderRegistryBase(
        adapter_validator=lambda adapter: isinstance(adapter, _DummyAdapter)
    )

    with pytest.raises(TypeError, match="Adapter instance is not valid"):
        registry.register_adapter("bad-instance", _InvalidAdapter())


def test_register_invalid_dotted_path_is_lazy_without_spec_validator() -> None:
    registry: ProviderRegistryBase[object] = ProviderRegistryBase()

    registry.register_adapter("broken-dotted", "not.a.real.module.Adapter")

    assert registry.get_status("broken-dotted") == ProviderStatus.ENABLED
    assert registry.get_adapter("broken-dotted") is None
    assert registry.get_status("broken-dotted") == ProviderStatus.FAILED


def test_reregister_adapter_invalidates_cached_instance() -> None:
    class _AdapterA:
        def __init__(self) -> None:
            self.marker = "a"

    class _AdapterB:
        def __init__(self) -> None:
            self.marker = "b"

    registry: ProviderRegistryBase[object] = ProviderRegistryBase()
    registry.register_adapter("provider", _AdapterA)
    first = registry.get_adapter("provider")

    assert first is not None
    assert getattr(first, "marker", None) == "a"

    registry.register_adapter("provider", _AdapterB)
    second = registry.get_adapter("provider")

    assert second is not None
    assert second is not first
    assert getattr(second, "marker", None) == "b"


def test_reregister_adapter_clears_previous_failure_state() -> None:
    class _FailingAdapter:
        def __init__(self) -> None:
            raise RuntimeError("boom")

    class _HealthyAdapter:
        def __init__(self) -> None:
            self.ok = True

    registry: ProviderRegistryBase[object] = ProviderRegistryBase()
    registry.register_adapter("provider", _FailingAdapter)

    assert registry.get_adapter("provider") is None
    assert registry.get_status("provider") == ProviderStatus.FAILED

    registry.register_adapter("provider", _HealthyAdapter)
    adapter = registry.get_adapter("provider")

    assert adapter is not None
    assert registry.get_status("provider") == ProviderStatus.ENABLED


def test_name_normalization_and_alias_resolution() -> None:
    registry: ProviderRegistryBase[_DummyAdapter] = ProviderRegistryBase()
    registry.register_adapter("faster-whisper", _DummyAdapter, aliases=["fw"])

    assert registry.resolve_provider_name("faster_whisper") == "faster-whisper"
    assert registry.resolve_provider_name("FW") == "faster-whisper"
    assert registry.get_adapter("FW") is not None


def test_disabled_provider_returns_none_and_disabled_status() -> None:
    registry: ProviderRegistryBase[_DummyAdapter] = ProviderRegistryBase()
    registry.register_adapter("dummy", _DummyAdapter, enabled=False)

    assert registry.get_adapter("dummy") is None
    assert registry.get_status("dummy") == ProviderStatus.DISABLED


def test_unknown_provider_status_is_unknown() -> None:
    registry: ProviderRegistryBase[_DummyAdapter] = ProviderRegistryBase()

    assert registry.get_adapter("missing") is None
    assert registry.get_status("missing") == ProviderStatus.UNKNOWN


def test_failed_provider_without_retry_stays_failed() -> None:
    class _AlwaysFail:
        def __init__(self) -> None:
            raise RuntimeError("boom")

    registry: ProviderRegistryBase[object] = ProviderRegistryBase()
    registry.register_adapter("broken", _AlwaysFail)

    assert registry.get_adapter("broken") is None
    assert registry.get_status("broken") == ProviderStatus.FAILED
    assert registry.get_adapter("broken") is None
    assert registry.get_status("broken") == ProviderStatus.FAILED


def test_failed_provider_can_retry_after_window() -> None:
    state = {"attempts": 0}

    class _Flaky:
        def __init__(self) -> None:
            state["attempts"] += 1
            if state["attempts"] == 1:
                raise RuntimeError("first attempt fails")
            self.ok = True

    registry: ProviderRegistryBase[object] = ProviderRegistryBase(
        config=ProviderRegistryConfig(failure_retry_seconds=0.02)
    )
    registry.register_adapter("flaky", _Flaky)

    assert registry.get_adapter("flaky") is None
    assert registry.get_status("flaky") == ProviderStatus.FAILED

    time.sleep(0.03)

    adapter = registry.get_adapter("flaky")
    assert adapter is not None
    assert state["attempts"] == 2
    assert registry.get_status("flaky") == ProviderStatus.ENABLED
