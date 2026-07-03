from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSProviderInitializationError,
    TTSValidationError,
)


class _FakeProcess:
    def __init__(self, *, stderr_text: str = "") -> None:
        self.returncode = None
        self.terminate_called = False
        self.kill_called = False
        self.stderr_text = stderr_text

    async def wait(self) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.terminate_called = True
        self.returncode = 0

    def kill(self) -> None:
        self.kill_called = True
        self.returncode = -9


class _ReadyClient:
    def __init__(self, *, base_url: str, **_kwargs) -> None:
        self.base_url = base_url
        self.health_calls = 0

    async def health(self) -> dict[str, str]:
        self.health_calls += 1
        return {"status": "ok"}

    async def close(self) -> None:
        return None


class _NeverReadyClient:
    def __init__(self, *, base_url: str, **_kwargs) -> None:
        self.base_url = base_url

    async def health(self) -> dict[str, str]:
        raise RuntimeError(
            "raw stderr: token=secret C:/Users/GDesktop-1/Working/tldw/models/audio_cpp/server.json"
        )

    async def close(self) -> None:
        return None


def _workspace_test_dir(name: str) -> Path:
    root = Path.cwd() / "models" / "audio_cpp" / "test_artifacts" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _provider_config(test_root: Path, *, host: str = "127.0.0.1") -> dict[str, object]:
    return {
        "base_url": "http://127.0.0.1:8080",
        "model": "audio-cpp/pocket-tts",
        "model_path": "models/audio_cpp/pocket-tts",
        "binary_path": str(test_root / "bin" / "audiocpp_server"),
        "timeout": 300,
        "extra_params": {
            "managed": True,
            "allow_remote_base_url": False,
            "server": {
                "host": host,
                "port": 8080,
                "autoselect_port": True,
                "port_probe_max": 3,
                "startup_timeout_seconds": 0.05,
                "healthcheck_interval_seconds": 0.01,
                "startup_backoff_seconds": 5,
                "idle_shutdown_seconds": 1,
                "terminate_timeout_seconds": 0.1,
                "server_config_path": "models/audio_cpp/server.json",
                "models_root": "models/audio_cpp",
                "shared_scratch_dir": "models/audio_cpp/runtime/scratch",
                "lazy_load": True,
                "device": 0,
                "threads": 1,
                "model": {
                    "id": "pocket-tts",
                    "family": "pocket_tts",
                    "path": "models/audio_cpp/pocket-tts",
                    "task": "tts",
                    "mode": "offline",
                },
            },
        },
    }


@pytest.mark.unit
def test_sidecar_rejects_non_loopback_host():
    from tldw_Server_API.app.core.TTS.adapters.audio_cpp_sidecar_supervisor import (
        AudioCppSidecarSupervisor,
    )

    test_root = _workspace_test_dir("sidecar_rejects_non_loopback")

    with pytest.raises(TTSValidationError, match="loopback"):
        AudioCppSidecarSupervisor(
            _provider_config(test_root, host="0.0.0.0"),
            repo_root=test_root,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sidecar_autoselects_port_renders_config_and_uses_fixed_command(monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import audio_cpp_sidecar_supervisor as supervisor_module
    from tldw_Server_API.app.core.TTS.adapters.audio_cpp_sidecar_supervisor import AudioCppSidecarSupervisor

    probed_ports: list[int] = []
    spawned: list[tuple[tuple[object, ...], dict[str, object]]] = []
    test_root = _workspace_test_dir("sidecar_autoselect")

    def _fake_is_port_free(host: str, port: int) -> bool:
        assert host == "127.0.0.1"
        probed_ports.append(port)
        return port == 8082

    async def _fake_spawn(*args, **kwargs):
        spawned.append((args, kwargs))
        return _FakeProcess()

    monkeypatch.setattr(supervisor_module, "is_port_free", _fake_is_port_free, raising=True)
    monkeypatch.setattr(supervisor_module.asyncio, "create_subprocess_exec", _fake_spawn, raising=True)
    monkeypatch.setattr(supervisor_module, "AudioCppClient", _ReadyClient, raising=True)
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")

    supervisor = AudioCppSidecarSupervisor(_provider_config(test_root), repo_root=test_root)

    base_url = await supervisor.ensure_started()

    assert base_url == "http://127.0.0.1:8082"
    assert supervisor.port == 8082
    assert probed_ports == [8080, 8081, 8082]
    assert len(spawned) == 1

    command, kwargs = spawned[0]
    assert command == (str(test_root / "bin" / "audiocpp_server"), "--config", str(supervisor.server_config_path))
    assert kwargs["cwd"] == str(test_root)
    assert "HF_TOKEN" not in kwargs["env"]
    assert "OPENAI_API_KEY" not in kwargs["env"]

    server_config = json.loads(supervisor.server_config_path.read_text(encoding="utf-8"))
    assert server_config["host"] == "127.0.0.1"
    assert server_config["port"] == 8082
    assert server_config["models"][0]["id"] == "pocket-tts"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_startup_timeout_terminates_process_records_backoff_and_sanitizes_error(monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import audio_cpp_sidecar_supervisor as supervisor_module
    from tldw_Server_API.app.core.TTS.adapters.audio_cpp_sidecar_supervisor import AudioCppSidecarSupervisor

    process = _FakeProcess(stderr_text="token=secret full local path")
    test_root = _workspace_test_dir("sidecar_timeout")

    async def _fake_spawn(*args, **kwargs):  # noqa: ARG001
        return process

    monkeypatch.setattr(supervisor_module, "is_port_free", lambda host, port: True, raising=True)
    monkeypatch.setattr(supervisor_module.asyncio, "create_subprocess_exec", _fake_spawn, raising=True)
    monkeypatch.setattr(supervisor_module, "AudioCppClient", _NeverReadyClient, raising=True)

    supervisor = AudioCppSidecarSupervisor(_provider_config(test_root), repo_root=test_root)

    with pytest.raises(TTSProviderInitializationError) as excinfo:
        await supervisor.ensure_started()

    assert process.terminate_called is True
    assert supervisor.last_failure_at is not None
    message = str(excinfo.value)
    assert "token=secret" not in message
    assert str(supervisor.server_config_path) not in message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_if_idle_stops_live_process_once():
    from tldw_Server_API.app.core.TTS.adapters.audio_cpp_sidecar_supervisor import (
        AudioCppSidecarSupervisor,
    )

    test_root = _workspace_test_dir("sidecar_idle_shutdown")
    supervisor = AudioCppSidecarSupervisor(_provider_config(test_root), repo_root=test_root)
    process = _FakeProcess()
    supervisor._process = process
    supervisor._base_url = "http://127.0.0.1:8080"
    supervisor._last_activity_at = 0

    first = await supervisor.shutdown_if_idle()
    second = await supervisor.shutdown_if_idle()

    assert first is True
    assert second is False
    assert process.terminate_called is True
