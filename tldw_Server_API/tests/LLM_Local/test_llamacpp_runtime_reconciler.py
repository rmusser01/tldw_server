from __future__ import annotations

import asyncio
from configparser import ConfigParser
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamaCppConfig
from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_store import JsonLlamaCppProfileStore
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import (
    LlamaCppPortPolicy,
    LlamaCppProfile,
    LlamaCppRuntime,
    LlamaCppRuntimeState,
)
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_reconciler import LlamaCppRuntimeReconciler
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


def _llamacpp_parser(default_models_dir: Path) -> ConfigParser:
    parser = ConfigParser()
    parser.add_section("LlamaCpp")
    parser["LlamaCpp"] = {
        "enabled": "true",
        "models_dir": str(default_models_dir),
        "allowed_paths": "",
        "registered_model_paths": "",
        "imported_asset_folders": "",
    }
    return parser


@pytest.fixture(autouse=True)
def configure_inventory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service

    monkeypatch.setattr(
        llamacpp_inventory_service,
        "load_comprehensive_config",
        lambda: _llamacpp_parser(tmp_path / "models"),
    )


def profile(
    profile_id: str,
    *,
    model_path: Path,
    port: int = 8181,
    enabled: bool = True,
    autostart: bool = False,
    restart_policy: dict[str, object] | None = None,
) -> LlamaCppProfile:
    return LlamaCppProfile(
        profile_id=profile_id,
        name=f"Profile {profile_id}",
        enabled=enabled,
        model_path=str(model_path),
        host="127.0.0.1",
        port=port,
        port_policy=LlamaCppPortPolicy.EXPLICIT,
        server_args={"ctx_size": 4096},
        autostart=autostart,
        restart_policy=dict(restart_policy or {}),
    )


class FakeRunner:
    def __init__(self, profile_id: str, calls: dict[str, int]):
        self.profile_id = profile_id
        self.calls = calls
        self.runtime = LlamaCppRuntime(profile_id=profile_id, state=LlamaCppRuntimeState.DEFINED)
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
        self.runtime = self.runtime.model_copy(update={"state": LlamaCppRuntimeState.STOPPED, "pid": None})


class FailingRunner(FakeRunner):
    async def start(self, model_path: Path, profile: LlamaCppProfile) -> LlamaCppRuntime:
        self.calls[self.profile_id] = self.calls.get(self.profile_id, 0) + 1
        await asyncio.sleep(0)
        self.runtime = LlamaCppRuntime(
            profile_id=self.profile_id,
            state=LlamaCppRuntimeState.FAILED,
            host=profile.host,
            port=profile.port,
            model_path=str(model_path),
            exit_code=42,
            last_error="boom",
            message="boom",
        )
        raise ServerError("boom")


class FakeRunnerFactory:
    def __init__(self, runner_cls: type[FakeRunner] = FakeRunner):
        self.runner_cls = runner_cls
        self.calls: dict[str, int] = {}
        self.runners: dict[str, FakeRunner] = {}

    def __call__(self, _config: LlamaCppConfig, profile_id: str) -> FakeRunner:
        runner = self.runner_cls(profile_id, self.calls)
        self.runners[profile_id] = runner
        return runner


def make_supervisor(
    tmp_path: Path,
    *,
    runner_cls: type[FakeRunner] = FakeRunner,
) -> tuple[LlamaCppSupervisor, LlamaCppConfig, FakeRunnerFactory]:
    config = make_config(tmp_path)
    store = JsonLlamaCppProfileStore(tmp_path / "profiles.json")
    factory = FakeRunnerFactory(runner_cls=runner_cls)
    supervisor = LlamaCppSupervisor(config=config, store=store, runner_factory=factory)
    return supervisor, config, factory


@pytest.mark.asyncio
async def test_reconciler_autostarts_only_enabled_autostart_profiles(tmp_path: Path) -> None:
    supervisor, config, factory = make_supervisor(tmp_path)
    first_model = make_model(config, "auto.gguf")
    second_model = make_model(config, "manual.gguf")
    disabled_model = make_model(config, "disabled.gguf")
    supervisor.store.upsert(profile("auto", model_path=first_model, autostart=True, port=8181))
    supervisor.store.upsert(profile("manual", model_path=second_model, autostart=False, port=8182))
    supervisor.store.upsert(profile("disabled", model_path=disabled_model, enabled=False, autostart=True, port=8183))

    runtimes = await LlamaCppRuntimeReconciler(supervisor).reconcile_startup()

    assert [runtime.profile_id for runtime in runtimes] == ["auto"]
    assert runtimes[0].state == LlamaCppRuntimeState.RUNNING
    assert factory.calls == {"auto": 1}


@pytest.mark.asyncio
async def test_reconciler_records_failed_start_metadata_and_respects_max_restarts(tmp_path: Path) -> None:
    supervisor, config, factory = make_supervisor(tmp_path, runner_cls=FailingRunner)
    model_path = make_model(config, "failing.gguf")
    supervisor.store.upsert(
        profile(
            "failing",
            model_path=model_path,
            autostart=True,
            restart_policy={"max_restarts": 1},
        )
    )
    reconciler = LlamaCppRuntimeReconciler(supervisor)

    first = await reconciler.reconcile_once()
    second = await reconciler.reconcile_once()

    stored = supervisor.store.get("failing")
    assert first[0].state == LlamaCppRuntimeState.FAILED
    assert first[0].last_error == "boom"
    assert second[0].state == LlamaCppRuntimeState.FAILED
    assert factory.calls == {"failing": 1}
    assert stored is not None
    assert stored.last_runtime_failure["state"] == "failed"
    assert stored.last_runtime_failure["last_error"] == "boom"
    assert stored.last_runtime_failure["exit_code"] == 42
    assert stored.last_runtime_failure["restart_count"] == 1
    reloaded = LlamaCppSupervisor(
        config=config,
        store=JsonLlamaCppProfileStore(tmp_path / "profiles.json"),
        runner_factory=FakeRunnerFactory(),
    )
    reloaded_runtime = reloaded.get_runtime("failing")
    assert reloaded_runtime.state == LlamaCppRuntimeState.FAILED
    assert reloaded_runtime.last_error == "boom"
    assert reloaded_runtime.restart_count == 1


@pytest.mark.asyncio
async def test_reconciler_skips_paused_profile_until_resume(tmp_path: Path) -> None:
    supervisor, config, factory = make_supervisor(tmp_path)
    model_path = make_model(config, "paused.gguf")
    supervisor.store.upsert(profile("paused", model_path=model_path, autostart=True, port=8181))
    await supervisor.pause_profile("paused")
    reconciler = LlamaCppRuntimeReconciler(supervisor)

    runtimes = await reconciler.reconcile_once()
    resumed = await supervisor.resume_profile("paused")

    assert runtimes == []
    assert resumed.state == LlamaCppRuntimeState.RUNNING
    assert factory.calls == {"paused": 1}


@pytest.mark.asyncio
async def test_reconciler_shutdown_stops_owned_runners_without_creating_new_ones(tmp_path: Path) -> None:
    supervisor, config, factory = make_supervisor(tmp_path)
    running_model = make_model(config, "running.gguf")
    stored_model = make_model(config, "stored.gguf")
    supervisor.store.upsert(profile("running", model_path=running_model, autostart=True, port=8181))
    supervisor.store.upsert(profile("stored", model_path=stored_model, autostart=True, port=8182))
    await supervisor.start_profile("running")

    await LlamaCppRuntimeReconciler(supervisor).shutdown()

    assert factory.runners["running"].stop_calls == 1
    assert "stored" not in factory.runners
