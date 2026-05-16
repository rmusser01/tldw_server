from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamaCppConfig
from tldw_Server_API.app.core.Local_LLM.llamacpp_process_runner import LlamaCppProcessRunner
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import (
    LlamaCppPortPolicy,
    LlamaCppProfile,
    LlamaCppRuntimeState,
)


class FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode: int | None = None
        self.stderr = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def make_config(tmp_path: Path, *, log_output_file: Path | None = None, port_autoselect: bool = True) -> LlamaCppConfig:
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
        port_autoselect=port_autoselect,
        port_probe_max=3,
        allowed_paths=[models_dir],
        readiness_timeout=0.1,
        stderr_read_timeout=0.1,
        log_output_file=log_output_file,
    )


def make_model(config: LlamaCppConfig, name: str = "model.gguf") -> Path:
    model_path = config.models_dir / name
    model_path.write_text("not really gguf", encoding="utf-8")
    return model_path


def profile(
    profile_id: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8181,
    port_policy: LlamaCppPortPolicy = LlamaCppPortPolicy.EXPLICIT,
    server_args: dict[str, Any] | None = None,
) -> LlamaCppProfile:
    return LlamaCppProfile(
        profile_id=profile_id,
        name=f"Profile {profile_id}",
        model_id=f"gguf:{profile_id}",
        model_path=f"/models/{profile_id}.gguf",
        host=host,
        port=port,
        port_policy=port_policy,
        server_args=server_args or {"ctx_size": 4096, "n_gpu_layers": 1, "threads": 4},
    )


def test_runner_reports_defined_before_first_start(tmp_path: Path):
    config = make_config(tmp_path)
    runner = LlamaCppProcessRunner(config, profile_id="defined")

    runtime = runner.status()

    assert runtime.state == LlamaCppRuntimeState.DEFINED
    assert runtime.pid is None


@pytest.mark.asyncio
async def test_runner_starts_two_profiles_on_distinct_ports_without_stopping_each_other(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_process_runner as runner_module

    processes = [FakeProcess(1001), FakeProcess(1002)]
    commands: list[list[str]] = []

    async def fake_create_subprocess_exec(*command: str, **_kwargs: Any) -> FakeProcess:
        commands.append(list(command))
        return processes[len(commands) - 1]

    async def fake_ready(*_args: Any, **_kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(runner_module, "wait_for_http_ready", fake_ready)
    monkeypatch.setattr(runner_module.platform, "system", lambda: "Windows")

    config = make_config(tmp_path)
    model_path = make_model(config)
    first = LlamaCppProcessRunner(config, profile_id="one")
    second = LlamaCppProcessRunner(config, profile_id="two")

    first_runtime = await first.start(model_path, profile=profile("one", port=8181))
    second_runtime = await second.start(model_path, profile=profile("two", port=8182))

    assert first_runtime.state == LlamaCppRuntimeState.RUNNING
    assert first_runtime.port == 8181
    assert second_runtime.port == 8182
    assert commands[0][commands[0].index("--port") + 1] == "8181"
    assert commands[1][commands[1].index("--port") + 1] == "8182"

    stopped = await first.stop()

    assert stopped.state == LlamaCppRuntimeState.STOPPED
    assert processes[0].terminated is True
    assert processes[1].terminated is False
    assert second.status().state == LlamaCppRuntimeState.RUNNING


@pytest.mark.asyncio
async def test_runner_autoselects_port_when_profile_policy_requests_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_process_runner as runner_module

    commands: list[list[str]] = []

    async def fake_create_subprocess_exec(*command: str, **_kwargs: Any) -> FakeProcess:
        commands.append(list(command))
        return FakeProcess(1003)

    async def fake_ready(*_args: Any, **_kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(runner_module, "wait_for_http_ready", fake_ready)
    monkeypatch.setattr(runner_module.platform, "system", lambda: "Windows")

    config = make_config(tmp_path, port_autoselect=False)
    model_path = make_model(config)
    runner = LlamaCppProcessRunner(config, profile_id="auto")
    monkeypatch.setattr(runner, "_is_port_free", lambda _host, port: port == 8183)

    runtime = await runner.start(model_path, profile=profile("auto", port=8181, port_policy=LlamaCppPortPolicy.AUTOSELECT))

    assert runtime.port == 8183
    assert commands[0][commands[0].index("--port") + 1] == "8183"


@pytest.mark.asyncio
async def test_runner_rejects_model_path_outside_allowed_paths(tmp_path: Path):
    config = make_config(tmp_path)
    outside_model = tmp_path / "outside" / "model.gguf"
    outside_model.parent.mkdir()
    outside_model.write_text("not really gguf", encoding="utf-8")
    runner = LlamaCppProcessRunner(config, profile_id="outside")

    with pytest.raises(ServerError, match="allowed directories"):
        await runner.start(outside_model, profile=profile("outside"))


@pytest.mark.asyncio
async def test_runner_tails_only_owned_log_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_process_runner as runner_module

    async def fake_create_subprocess_exec(*_command: str, **_kwargs: Any) -> FakeProcess:
        return FakeProcess(1004)

    async def fake_ready(*_args: Any, **_kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(runner_module, "wait_for_http_ready", fake_ready)
    monkeypatch.setattr(runner_module.platform, "system", lambda: "Windows")

    log_path = tmp_path / "llamacpp.log"
    config = make_config(tmp_path, log_output_file=log_path)
    model_path = make_model(config)
    runner = LlamaCppProcessRunner(config, profile_id="logs")

    await runner.start(model_path, profile=profile("logs"))
    log_path.write_text("one\napi_key=secret-value\ntwo\n", encoding="utf-8")

    result = runner.tail_logs(2)

    assert result["lines"] == ["api_key=[REDACTED]", "two"]
    assert result["truncated"] is True

    await runner.stop()
