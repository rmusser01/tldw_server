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
    def __init__(self, pid: int, *, stdout: Any = None, stderr: Any = None):
        self.pid = pid
        self.returncode: int | None = None
        self.stdout = stdout
        self.stderr = stderr
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


class FakeStream:
    def __init__(self, chunks: list[bytes]):
        self.chunks = list(chunks)
        self.read_count = 0
        self.read_sizes: list[int] = []

    async def read(self, size: int) -> bytes:
        self.read_count += 1
        self.read_sizes.append(size)
        if self.chunks:
            return self.chunks.pop(0)
        return b""


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
    assert runtime.restart_count == 0
    assert runtime.warnings == []


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
    monkeypatch.setattr(first, "_is_port_free", lambda _host, _port: True)
    monkeypatch.setattr(second, "_is_port_free", lambda _host, _port: True)

    first_runtime = await first.start(model_path, profile=profile("one", port=8181))
    second_runtime = await second.start(model_path, profile=profile("two", port=8182))

    assert first_runtime.state == LlamaCppRuntimeState.RUNNING
    assert first_runtime.port == 8181
    assert first_runtime.endpoint == "http://127.0.0.1:8181"
    assert first_runtime.last_health_at is not None
    assert first_runtime.log_tail_available is False
    assert first_runtime.resolved_args == first_runtime.command
    assert second_runtime.port == 8182
    assert commands[0][commands[0].index("--port") + 1] == "8181"
    assert commands[1][commands[1].index("--port") + 1] == "8182"

    stopped = await first.stop()

    assert stopped.state == LlamaCppRuntimeState.STOPPED
    assert processes[0].terminated is True
    assert processes[1].terminated is False
    assert second.status().state == LlamaCppRuntimeState.RUNNING


@pytest.mark.asyncio
async def test_runner_rejects_occupied_explicit_port_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_process_runner as runner_module

    spawned = False

    async def fake_create_subprocess_exec(*_command: str, **_kwargs: Any) -> FakeProcess:
        nonlocal spawned
        spawned = True
        return FakeProcess(1005)

    monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    config = make_config(tmp_path)
    model_path = make_model(config)
    runner = LlamaCppProcessRunner(config, profile_id="occupied")
    monkeypatch.setattr(runner, "_is_port_free", lambda _host, _port: False)

    with pytest.raises(ServerError, match="port 8181 is not available"):
        await runner.start(model_path, profile=profile("occupied", port=8181))

    assert spawned is False


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
async def test_runner_retains_failed_runtime_details_after_start_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_process_runner as runner_module

    async def fake_create_subprocess_exec(*_command: str, **_kwargs: Any) -> FakeProcess:
        return FakeProcess(1006)

    async def fake_not_ready(*_args: Any, **_kwargs: Any) -> bool:
        return False

    monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(runner_module, "wait_for_http_ready", fake_not_ready)
    monkeypatch.setattr(runner_module.platform, "system", lambda: "Windows")

    config = make_config(tmp_path)
    model_path = make_model(config)
    runner = LlamaCppProcessRunner(config, profile_id="failed")
    monkeypatch.setattr(runner, "_is_port_free", lambda _host, _port: True)

    with pytest.raises(ServerError, match="failed to start"):
        await runner.start(model_path, profile=profile("failed", port=8181))

    runtime = runner.status()

    assert runtime.state == LlamaCppRuntimeState.FAILED
    assert runtime.port == 8181
    assert runtime.endpoint == "http://127.0.0.1:8181"
    assert runtime.model_path == str(model_path.resolve())
    assert runtime.command
    assert runtime.resolved_args == runtime.command
    assert runtime.last_error == "Llama.cpp server failed to start or become ready."


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
@pytest.mark.parametrize(
    ("server_args", "match"),
    [
        ({"model_draft": "OUTSIDE"}, "model_draft"),
        ({"lora_scaled": ["OUTSIDE", 0.5]}, "LoRA path"),
    ],
)
async def test_runner_rejects_path_bearing_args_outside_allowed_paths(
    tmp_path: Path,
    server_args: dict[str, Any],
    match: str,
):
    config = make_config(tmp_path)
    outside_model = tmp_path / "outside" / "model.gguf"
    outside_model.parent.mkdir()
    outside_model.write_text("not really gguf", encoding="utf-8")
    resolved_args = {
        key: [str(outside_model) if item == "OUTSIDE" else item for item in value]
        if isinstance(value, list)
        else str(outside_model)
        if value == "OUTSIDE"
        else value
        for key, value in server_args.items()
    }
    runner = LlamaCppProcessRunner(config, profile_id="arg-path")
    runner._is_port_free = lambda _host, _port: True

    with pytest.raises(ServerError, match=match):
        await runner.start(make_model(config), profile=profile("arg-path", server_args=resolved_args))


@pytest.mark.asyncio
async def test_runner_accepts_existing_handler_server_arg_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_process_runner as runner_module

    commands: list[list[str]] = []

    async def fake_create_subprocess_exec(*command: str, **_kwargs: Any) -> FakeProcess:
        commands.append(list(command))
        return FakeProcess(1007)

    async def fake_ready(*_args: Any, **_kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(runner_module, "wait_for_http_ready", fake_ready)
    monkeypatch.setattr(runner_module.platform, "system", lambda: "Windows")

    config = make_config(tmp_path)
    runner = LlamaCppProcessRunner(config, profile_id="aliases")
    monkeypatch.setattr(runner, "_is_port_free", lambda _host, _port: True)
    existing_aliases = {
        "rope_scaling": "yarn",
        "typical": 0.95,
        "dry_multiplier": 1.0,
        "dry_base": 1.75,
        "dry_allowed_length": 2,
        "cnv": True,
        "no_cnv": True,
        "in_prefix_bos": True,
        "r": "User:",
        "j": "{\"type\":\"object\"}",
    }

    await runner.start(make_model(config), profile=profile("aliases", server_args=existing_aliases))

    command = commands[0]
    assert "--rope-scaling" in command
    assert "--typical" in command
    assert "--dry-multiplier" in command
    assert "--conversation" in command
    assert "--no-conversation" in command
    assert "--in-prefix-bos" in command
    assert "-j" in command


@pytest.mark.asyncio
async def test_runner_filters_blank_optional_args_like_existing_handler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_process_runner as runner_module

    commands: list[list[str]] = []

    async def fake_create_subprocess_exec(*command: str, **_kwargs: Any) -> FakeProcess:
        commands.append(list(command))
        return FakeProcess(1009)

    async def fake_ready(*_args: Any, **_kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(runner_module, "wait_for_http_ready", fake_ready)
    monkeypatch.setattr(runner_module.platform, "system", lambda: "Windows")

    config = make_config(tmp_path)
    runner = LlamaCppProcessRunner(config, profile_id="blank-args")
    monkeypatch.setattr(runner, "_is_port_free", lambda _host, _port: True)

    await runner.start(
        make_model(config),
        profile=profile(
            "blank-args",
            server_args={
                "ctx_size": "",
                "n_gpu_layers": None,
                "threads": "",
                "log_file": "",
            },
        ),
    )

    command = commands[0]
    assert command[command.index("-c") + 1] == "2048"
    assert command[command.index("-ngl") + 1] == "0"
    assert "-t" not in command
    assert "--log-file" not in command


@pytest.mark.asyncio
async def test_runner_drains_pipe_streams_when_no_log_file_is_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from tldw_Server_API.app.core.Local_LLM import llamacpp_process_runner as runner_module

    stdout = FakeStream([b"x" * 2048])
    stderr = FakeStream([b"y" * 2048])

    async def fake_create_subprocess_exec(*_command: str, **_kwargs: Any) -> FakeProcess:
        return FakeProcess(1008, stdout=stdout, stderr=stderr)

    async def fake_ready(*_args: Any, **_kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(runner_module, "wait_for_http_ready", fake_ready)
    monkeypatch.setattr(runner_module.platform, "system", lambda: "Windows")

    config = make_config(tmp_path)
    runner = LlamaCppProcessRunner(config, profile_id="drain")
    monkeypatch.setattr(runner, "_is_port_free", lambda _host, _port: True)

    await runner.start(make_model(config), profile=profile("drain"))
    await asyncio.sleep(0)

    assert stdout.read_count >= 2
    assert stderr.read_count >= 2
    assert stdout.read_sizes
    assert stderr.read_sizes
    assert max(stdout.read_sizes) <= 1024
    assert max(stderr.read_sizes) <= 1024

    await runner.stop()


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
    monkeypatch.setattr(runner, "_is_port_free", lambda _host, _port: True)

    await runner.start(model_path, profile=profile("logs"))
    log_path.write_text("one\napi_key=secret-value\ntwo\n", encoding="utf-8")

    result = runner.tail_logs(2)

    assert result["lines"] == ["api_key=[REDACTED]", "two"]
    assert result["truncated"] is True

    await runner.stop()
