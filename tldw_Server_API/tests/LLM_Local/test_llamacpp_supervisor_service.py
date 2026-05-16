from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import (
    LlamaCppProfileCreateRequest,
    LlamaCppProfileUpdateRequest,
)
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamaCppConfig
from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_store import JsonLlamaCppProfileStore
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import (
    LlamaCppPortPolicy,
    LlamaCppProfile,
    LlamaCppProfileConflictError,
    LlamaCppRuntime,
    LlamaCppRuntimeState,
)
from tldw_Server_API.app.core.Local_LLM.llamacpp_supervisor_service import LlamaCppSupervisor


def make_config(tmp_path: Path) -> LlamaCppConfig:
    executable = tmp_path / "bin" / "llama-server"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return LlamaCppConfig(
        executable_path=executable,
        models_dir=models_dir,
        default_host="127.0.0.1",
        default_port=8080,
        default_n_gpu_layers=0,
        default_ctx_size=2048,
        default_threads=None,
        port_autoselect=True,
        port_probe_max=3,
        allowed_paths=[models_dir],
        readiness_timeout=0.1,
        stderr_read_timeout=0.1,
        log_output_file=None,
    )


def make_model(config: LlamaCppConfig, name: str = "model.gguf") -> Path:
    model_path = config.models_dir / name
    model_path.write_text("not really gguf", encoding="utf-8")
    return model_path


def profile(
    profile_id: str,
    *,
    model_path: str,
    port: int = 8181,
    enabled: bool = True,
    port_policy: LlamaCppPortPolicy = LlamaCppPortPolicy.EXPLICIT,
) -> LlamaCppProfile:
    return LlamaCppProfile(
        profile_id=profile_id,
        name=f"Profile {profile_id}",
        enabled=enabled,
        model_id=f"gguf:{profile_id}",
        model_path=model_path,
        host="127.0.0.1",
        port=port,
        port_policy=port_policy,
        server_args={"ctx_size": 4096},
    )


class FakeRunner:
    def __init__(self, profile_id: str, calls: dict[str, int]):
        self.profile_id = profile_id
        self.calls = calls
        self.runtime = LlamaCppRuntime(profile_id=profile_id, state=LlamaCppRuntimeState.DEFINED)
        self.cleaned = False
        self.stop_calls = 0

    async def start(self, model_path: Path, profile: LlamaCppProfile) -> LlamaCppRuntime:
        self.calls[self.profile_id] = self.calls.get(self.profile_id, 0) + 1
        await asyncio.sleep(0)
        self.runtime = LlamaCppRuntime(
            profile_id=self.profile_id,
            state=LlamaCppRuntimeState.RUNNING,
            pid=1000 + len(self.calls),
            host=profile.host,
            port=profile.port,
            endpoint=f"http://{profile.host}:{profile.port}",
            model_id=profile.model_id,
            model_path=str(model_path),
        )
        return self.runtime

    async def stop(self) -> LlamaCppRuntime:
        self.stop_calls += 1
        self.runtime = self.runtime.model_copy(
            update={"state": LlamaCppRuntimeState.STOPPED, "pid": None, "message": "Stopped"}
        )
        return self.runtime

    def status(self) -> LlamaCppRuntime:
        return self.runtime

    def cleanup_sync(self) -> None:
        self.cleaned = True
        self.runtime = self.runtime.model_copy(update={"state": LlamaCppRuntimeState.STOPPED, "pid": None})


class FakeRunnerFactory:
    def __init__(self):
        self.calls: dict[str, int] = {}
        self.runners: dict[str, FakeRunner] = {}

    def __call__(self, config: LlamaCppConfig, profile_id: str) -> FakeRunner:
        runner = FakeRunner(profile_id, self.calls)
        self.runners[profile_id] = runner
        return runner


class BlockingFakeRunner(FakeRunner):
    def __init__(
        self,
        profile_id: str,
        calls: dict[str, int],
        started: asyncio.Event,
        release: asyncio.Event,
        started_profiles: list[str],
    ):
        super().__init__(profile_id, calls)
        self.started = started
        self.release = release
        self.started_profiles = started_profiles

    async def start(self, model_path: Path, profile: LlamaCppProfile) -> LlamaCppRuntime:
        self.started_profiles.append(self.profile_id)
        self.started.set()
        await self.release.wait()
        return await super().start(model_path, profile)


class BlockingRunnerFactory(FakeRunnerFactory):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.started_profiles: list[str] = []

    def __call__(self, config: LlamaCppConfig, profile_id: str) -> BlockingFakeRunner:
        runner = BlockingFakeRunner(profile_id, self.calls, self.started, self.release, self.started_profiles)
        self.runners[profile_id] = runner
        return runner


def make_supervisor(tmp_path: Path) -> tuple[LlamaCppSupervisor, LlamaCppConfig, FakeRunnerFactory]:
    config = make_config(tmp_path)
    store = JsonLlamaCppProfileStore(tmp_path / "profiles.json")
    factory = FakeRunnerFactory()
    supervisor = LlamaCppSupervisor(config=config, store=store, runner_factory=factory)
    return supervisor, config, factory


@pytest.mark.asyncio
async def test_supervisor_starts_two_profiles_on_distinct_ports(tmp_path: Path):
    supervisor, config, factory = make_supervisor(tmp_path)
    first_model = make_model(config, "one.gguf")
    second_model = make_model(config, "two.gguf")
    await supervisor.create_profile(
        LlamaCppProfileCreateRequest(
            profile_id="one",
            name="One",
            model_path=str(first_model),
            port=8181,
        )
    )
    await supervisor.create_profile(
        LlamaCppProfileCreateRequest(
            profile_id="two",
            name="Two",
            model_path=str(second_model),
            port=8182,
        )
    )

    await supervisor.start_profile("one")
    await supervisor.start_profile("two")
    states = supervisor.list_runtimes()

    assert {state.profile_id for state in states} == {"one", "two"}
    assert {state.port for state in states} == {8181, 8182}
    assert factory.calls == {"one": 1, "two": 1}


@pytest.mark.asyncio
async def test_supervisor_rejects_duplicate_enabled_explicit_port(tmp_path: Path):
    supervisor, config, _factory = make_supervisor(tmp_path)
    model_path = make_model(config)
    await supervisor.create_profile(
        LlamaCppProfileCreateRequest(profile_id="one", name="One", model_path=str(model_path), port=8181)
    )

    with pytest.raises(LlamaCppProfileConflictError):
        await supervisor.create_profile(
            LlamaCppProfileCreateRequest(profile_id="two", name="Two", model_path=str(model_path), port=8181)
        )


@pytest.mark.asyncio
async def test_supervisor_serializes_same_profile_start(tmp_path: Path):
    supervisor, config, factory = make_supervisor(tmp_path)
    model_path = make_model(config)
    await supervisor.create_profile(
        LlamaCppProfileCreateRequest(profile_id="one", name="One", model_path=str(model_path), port=8181)
    )

    await asyncio.gather(supervisor.start_profile("one"), supervisor.start_profile("one"))

    assert factory.calls["one"] == 1
    assert supervisor.get_runtime("one").state == LlamaCppRuntimeState.RUNNING


@pytest.mark.asyncio
async def test_supervisor_serializes_autoselect_starts_before_port_probe(tmp_path: Path):
    config = make_config(tmp_path)
    store = JsonLlamaCppProfileStore(tmp_path / "profiles.json")
    factory = BlockingRunnerFactory()
    supervisor = LlamaCppSupervisor(config=config, store=store, runner_factory=factory)
    first_model = make_model(config, "one.gguf")
    second_model = make_model(config, "two.gguf")
    await supervisor.create_profile(
        LlamaCppProfileCreateRequest(
            profile_id="one",
            name="One",
            model_path=str(first_model),
            port_policy=LlamaCppPortPolicy.AUTOSELECT,
        )
    )
    await supervisor.create_profile(
        LlamaCppProfileCreateRequest(
            profile_id="two",
            name="Two",
            model_path=str(second_model),
            port_policy=LlamaCppPortPolicy.AUTOSELECT,
        )
    )

    first_task = asyncio.create_task(supervisor.start_profile("one"))
    await factory.started.wait()
    second_task = asyncio.create_task(supervisor.start_profile("two"))
    await asyncio.sleep(0)

    assert factory.started_profiles == ["one"]

    factory.release.set()
    await asyncio.gather(first_task, second_task)

    assert factory.calls == {"one": 1, "two": 1}


@pytest.mark.asyncio
async def test_supervisor_serializes_profile_mutations_with_start(tmp_path: Path):
    config = make_config(tmp_path)
    store = JsonLlamaCppProfileStore(tmp_path / "profiles.json")
    factory = BlockingRunnerFactory()
    supervisor = LlamaCppSupervisor(config=config, store=store, runner_factory=factory)
    model_path = make_model(config)
    await supervisor.create_profile(
        LlamaCppProfileCreateRequest(profile_id="one", name="One", model_path=str(model_path), port=8181)
    )

    start_task = asyncio.create_task(supervisor.start_profile("one"))
    await factory.started.wait()
    update_task = asyncio.create_task(supervisor.update_profile("one", LlamaCppProfileUpdateRequest(name="Updated")))
    delete_task = asyncio.create_task(supervisor.delete_profile("one"))
    await asyncio.sleep(0)

    assert update_task.done() is False
    assert delete_task.done() is False

    factory.release.set()
    runtime = await start_task
    updated = await update_task
    deleted = await delete_task

    assert runtime.state == LlamaCppRuntimeState.RUNNING
    assert updated.name == "Updated"
    assert deleted is True
    assert supervisor.list_profiles() == []
    assert factory.runners["one"].stop_calls == 1


@pytest.mark.asyncio
async def test_supervisor_rejects_null_update_for_non_nullable_profile_fields(tmp_path: Path):
    supervisor, config, _factory = make_supervisor(tmp_path)
    model_path = make_model(config)
    await supervisor.create_profile(
        LlamaCppProfileCreateRequest(profile_id="one", name="One", model_path=str(model_path), port=8181)
    )

    with pytest.raises(ValidationError):
        await supervisor.update_profile("one", LlamaCppProfileUpdateRequest(host=None))

    assert supervisor.store.get("one").host == "127.0.0.1"


@pytest.mark.asyncio
async def test_supervisor_delete_running_profile_awaits_stop_before_removing(tmp_path: Path):
    supervisor, config, factory = make_supervisor(tmp_path)
    model_path = make_model(config)
    await supervisor.create_profile(
        LlamaCppProfileCreateRequest(profile_id="one", name="One", model_path=str(model_path), port=8181)
    )
    await supervisor.start_profile("one")

    deleted = await supervisor.delete_profile("one")

    assert deleted is True
    assert supervisor.list_profiles() == []
    assert factory.runners["one"].stop_calls == 1
    assert factory.runners["one"].cleaned is False


@pytest.mark.asyncio
async def test_supervisor_stop_pause_resume_and_cleanup(tmp_path: Path):
    supervisor, config, factory = make_supervisor(tmp_path)
    model_path = make_model(config)
    await supervisor.create_profile(
        LlamaCppProfileCreateRequest(profile_id="one", name="One", model_path=str(model_path), port=8181)
    )

    await supervisor.start_profile("one")
    stopped = await supervisor.stop_profile("one", disable=True)
    paused = await supervisor.pause_profile("one")
    resumed = await supervisor.resume_profile("one")
    await supervisor.start_profile("one")
    supervisor.cleanup_sync()
    deleted = await supervisor.delete_profile("one")

    assert stopped.state == LlamaCppRuntimeState.STOPPED
    assert deleted is True
    assert supervisor.list_profiles() == []
    assert paused.state == LlamaCppRuntimeState.PAUSED
    assert resumed.state == LlamaCppRuntimeState.STOPPED
    assert factory.runners["one"].cleaned is True


@pytest.mark.asyncio
async def test_supervisor_default_profile_bridge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_supervisor_service as supervisor_module

    supervisor, config, _factory = make_supervisor(tmp_path)
    model_path = make_model(config)
    monkeypatch.setattr(supervisor_module.llamacpp_inventory_service, "resolve_model_id", lambda _model_id: model_path)

    profile_result = await supervisor.ensure_default_profile_from_model("gguf:default", {"ctx_size": 1024})
    runtime = await supervisor.start_default_by_model("gguf:default", {"ctx_size": 1024})
    compat = supervisor.default_status_compat()
    stopped = await supervisor.stop_default()

    assert profile_result.profile_id == "default"
    assert runtime.state == LlamaCppRuntimeState.RUNNING
    assert compat["status"] == "running"
    assert compat["backend"] == "llamacpp"
    assert compat["model"] == str(model_path)
    assert stopped.state == LlamaCppRuntimeState.STOPPED


@pytest.mark.asyncio
async def test_supervisor_serializes_default_start_profile_updates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_supervisor_service as supervisor_module

    config = make_config(tmp_path)
    first_model = make_model(config, "first.gguf")
    second_model = make_model(config, "second.gguf")
    store = JsonLlamaCppProfileStore(tmp_path / "profiles.json")
    factory = BlockingRunnerFactory()
    supervisor = LlamaCppSupervisor(config=config, store=store, runner_factory=factory)
    model_paths = {"gguf:first": first_model, "gguf:second": second_model}
    monkeypatch.setattr(
        supervisor_module.llamacpp_inventory_service,
        "resolve_model_id",
        lambda model_id: model_paths[model_id],
    )

    first_task = asyncio.create_task(supervisor.start_default_by_model("gguf:first", {"port": 8181}))
    await factory.started.wait()
    second_task = asyncio.create_task(supervisor.start_default_by_model("gguf:second", {"port": 8182}))
    await asyncio.sleep(0)

    assert second_task.done() is False
    assert store.get("default").model_id == "gguf:first"

    factory.release.set()
    first_runtime = await first_task
    second_runtime = await second_task

    assert first_runtime.model_id == "gguf:first"
    assert second_runtime.model_id == "gguf:second"
    assert store.get("default").model_id == "gguf:second"
    assert factory.calls == {"default": 2}


@pytest.mark.asyncio
async def test_supervisor_default_start_swaps_running_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_supervisor_service as supervisor_module

    supervisor, config, factory = make_supervisor(tmp_path)
    first_model = make_model(config, "first.gguf")
    second_model = make_model(config, "second.gguf")
    model_paths = {"gguf:first": first_model, "gguf:second": second_model}
    monkeypatch.setattr(
        supervisor_module.llamacpp_inventory_service,
        "resolve_model_id",
        lambda model_id: model_paths[model_id],
    )

    first_runtime = await supervisor.start_default_by_model("gguf:first", {"port": 8181})
    second_runtime = await supervisor.start_default_by_model("gguf:second", {"port": 8182})

    assert first_runtime.model_id == "gguf:first"
    assert second_runtime.model_id == "gguf:second"
    assert second_runtime.port == 8182
    assert factory.calls == {"default": 2}


def test_manager_attaches_supervisor_and_uses_it_for_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import LLM_Inference_Manager as manager_module

    config = make_config(tmp_path)
    manager_config = type(
        "ManagerConfig",
        (),
        {
            "app_config": {},
            "ollama": None,
            "huggingface": None,
            "llamafile": None,
            "llamacpp": config,
        },
    )()
    handler_cleanup_called = False
    supervisor_cleanup_called = False

    class FakeHandler:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def _cleanup_managed_server_sync(self) -> None:
            nonlocal handler_cleanup_called
            handler_cleanup_called = True

    class FakeSupervisor:
        def cleanup_sync(self) -> None:
            nonlocal supervisor_cleanup_called
            supervisor_cleanup_called = True

    monkeypatch.setattr(manager_module, "LlamaCppHandler", FakeHandler)
    monkeypatch.setattr(manager_module.LlamaCppSupervisor, "from_manager", lambda _manager: FakeSupervisor())

    manager = manager_module.LLMInferenceManager(manager_config)
    manager.cleanup_on_exit()

    assert manager.llamacpp_supervisor is not None
    assert supervisor_cleanup_called is True
    assert handler_cleanup_called is True
