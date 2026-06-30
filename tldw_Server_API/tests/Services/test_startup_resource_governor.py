from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _install_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    **attributes: object,
) -> ModuleType:
    module = ModuleType(module_name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def _import_startup_resource_governor() -> ModuleType:
    sys.modules.pop("tldw_Server_API.app.services.startup_resource_governor", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_resource_governor")


class _FakeReloadConfig:
    def __init__(self, *, enabled: bool, interval_sec: int) -> None:
        self.enabled = enabled
        self.interval_sec = interval_sec


class _FakePolicyLoader:
    def __init__(self, path: str, reload_config: _FakeReloadConfig) -> None:
        self.path = path
        self.reload_config = reload_config
        self.calls: list[str] = []
        self._snapshot = SimpleNamespace(version=3, policies={"chat.default": {}}, route_map={})
        self.on_change_callbacks: list[object] = []

    async def load_once(self) -> None:
        self.calls.append("load_once")

    async def start_auto_reload(self) -> None:
        self.calls.append("start_auto_reload")

    def get_snapshot(self) -> SimpleNamespace:
        return self._snapshot

    def add_on_change(self, callback: object) -> None:
        self.on_change_callbacks.append(callback)


class _FakeGovernor:
    def __init__(self, *, policy_loader: object) -> None:
        self.policy_loader = policy_loader


@pytest.mark.asyncio
async def test_init_resource_governor_configures_file_loader_and_memory_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = SimpleNamespace(state=SimpleNamespace(), routes=[])
    created_loaders: list[_FakePolicyLoader] = []

    def _make_loader(path: str, reload_config: _FakeReloadConfig) -> _FakePolicyLoader:
        loader = _FakePolicyLoader(path, reload_config)
        created_loaders.append(loader)
        return loader

    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.config",
        rg_backend=lambda: "memory",
        rg_enabled=lambda default=False: True,
        rg_policy_path=lambda: "/tmp/rg-policies.yaml",
        rg_policy_reload_enabled=lambda: True,
        rg_policy_reload_interval_sec=lambda: 15,
        rg_policy_store=lambda: "file",
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.Resource_Governance",
        MemoryResourceGovernor=_FakeGovernor,
        RedisResourceGovernor=_FakeGovernor,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.Resource_Governance.policy_loader",
        PolicyLoader=_make_loader,
        PolicyReloadConfig=_FakeReloadConfig,
        db_policy_loader=lambda *_args, **_kwargs: None,
        default_policy_loader=lambda: None,
    )

    startup_rg = _import_startup_resource_governor()

    await startup_rg.init_resource_governor(app)

    assert len(created_loaders) == 1
    loader = created_loaders[0]
    assert loader.path == "/tmp/rg-policies.yaml"
    assert loader.reload_config.enabled is True
    assert loader.reload_config.interval_sec == 15
    assert loader.calls == ["load_once", "start_auto_reload"]
    assert app.state.rg_policy_loader is loader
    assert app.state.rg_policy_store == "file"
    assert app.state.rg_policy_version == 3
    assert app.state.rg_policy_count == 1
    assert isinstance(app.state.rg_governor, _FakeGovernor)
    assert app.state.rg_governor.policy_loader is loader
    assert len(loader.on_change_callbacks) == 1

    loader.on_change_callbacks[0](SimpleNamespace(version=9, policies={"a": {}, "b": {}}, route_map={}))

    assert app.state.rg_policy_version == 9
    assert app.state.rg_policy_count == 2


@pytest.mark.asyncio
async def test_init_resource_governor_falls_back_to_file_loader_when_db_store_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = SimpleNamespace(state=SimpleNamespace(), routes=[])
    fallback_loader = _FakePolicyLoader("/tmp/fallback.yaml", _FakeReloadConfig(enabled=False, interval_sec=60))
    warning_messages: list[str] = []

    class _BrokenStore:
        def __init__(self) -> None:
            raise RuntimeError("db store boom")

    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.config",
        rg_backend=lambda: "memory",
        rg_enabled=lambda default=False: False,
        rg_policy_path=lambda: "/tmp/rg-policies.yaml",
        rg_policy_reload_enabled=lambda: False,
        rg_policy_reload_interval_sec=lambda: 60,
        rg_policy_store=lambda: "db",
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.Resource_Governance",
        MemoryResourceGovernor=_FakeGovernor,
        RedisResourceGovernor=_FakeGovernor,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.Resource_Governance.policy_loader",
        PolicyLoader=_FakePolicyLoader,
        PolicyReloadConfig=_FakeReloadConfig,
        db_policy_loader=lambda *_args, **_kwargs: None,
        default_policy_loader=lambda: fallback_loader,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.Resource_Governance.authnz_policy_store",
        AuthNZPolicyStore=_BrokenStore,
    )

    startup_rg = _import_startup_resource_governor()
    monkeypatch.setattr(
        startup_rg.logger,
        "warning",
        lambda message, *args, **kwargs: warning_messages.append(str(message)),
    )

    await startup_rg.init_resource_governor(app)

    assert app.state.rg_policy_loader is fallback_loader
    assert app.state.rg_policy_store == "file"
    assert isinstance(app.state.rg_governor, _FakeGovernor)
    assert app.state.rg_governor.policy_loader is fallback_loader
    assert any("falling back to file" in message for message in warning_messages)


@pytest.mark.asyncio
async def test_init_resource_governor_warns_when_enabled_without_initialized_governor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = SimpleNamespace(state=SimpleNamespace(), routes=[])
    warning_messages: list[str] = []

    class _BrokenGovernor:
        def __init__(self, *, policy_loader: object) -> None:
            raise RuntimeError(f"boom: {policy_loader}")

    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.config",
        rg_backend=lambda: "memory",
        rg_enabled=lambda default=False: True,
        rg_policy_path=lambda: "/tmp/rg-policies.yaml",
        rg_policy_reload_enabled=lambda: False,
        rg_policy_reload_interval_sec=lambda: 30,
        rg_policy_store=lambda: "file",
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.Resource_Governance",
        MemoryResourceGovernor=_BrokenGovernor,
        RedisResourceGovernor=_BrokenGovernor,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.Resource_Governance.policy_loader",
        PolicyLoader=_FakePolicyLoader,
        PolicyReloadConfig=_FakeReloadConfig,
        db_policy_loader=lambda *_args, **_kwargs: None,
        default_policy_loader=lambda: None,
    )

    startup_rg = _import_startup_resource_governor()
    monkeypatch.setattr(
        startup_rg.logger,
        "warning",
        lambda message, *args, **kwargs: warning_messages.append(str(message)),
    )

    await startup_rg.init_resource_governor(app)

    assert getattr(app.state, "rg_policy_store", None) == "file"
    assert getattr(app.state, "rg_governor", None) is None
    assert any("initialization failed/skipped" in message for message in warning_messages)
    assert any("enabled but not initialized" in message for message in warning_messages)
