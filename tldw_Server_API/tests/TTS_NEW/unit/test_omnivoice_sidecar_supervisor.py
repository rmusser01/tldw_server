from __future__ import annotations

import asyncio
import os

import httpx
import pytest


class _FakeProcess:
    def __init__(self):
        self.pid = 4321
        self.returncode = None
        self.terminate_called = False
        self.kill_called = False

    async def wait(self):
        self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminate_called = True
        self.returncode = 0

    def kill(self):
        self.kill_called = True
        self.returncode = -9


class _ExitedProcess(_FakeProcess):
    def __init__(self, returncode: int = 1):
        super().__init__()
        self.returncode = returncode


@pytest.mark.unit
@pytest.mark.asyncio
async def test_supervisor_generates_fresh_token_and_includes_it_on_internal_requests(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import omnivoice_sidecar_supervisor as supervisor_module
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import X_TLDW_SIDECAR_TOKEN_HEADER
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_supervisor import OmniVoiceSidecarSupervisor

    recorded = {"trust_env": None, "headers": []}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            recorded["trust_env"] = kwargs.get("trust_env")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, *, headers=None, timeout=None):  # noqa: ARG002
            recorded["headers"].append((url, dict(headers or {})))
            return httpx.Response(200, json={"status": "ok", "ready": True})

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ARG001
        return _FakeProcess()

    def _fake_create_sidecar_async_client(*, timeout: float):
        assert timeout == pytest.approx(0.25)  # nosec B101
        return _FakeClient(trust_env=False)

    monkeypatch.setattr(supervisor_module, "is_port_free", lambda host, port: True, raising=True)
    monkeypatch.setattr(supervisor_module.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)
    monkeypatch.setattr(supervisor_module, "create_sidecar_async_client", _fake_create_sidecar_async_client, raising=True)

    supervisor = OmniVoiceSidecarSupervisor(
        provider_config={"extra_params": {"host": "127.0.0.1", "port": 8039}},
        repo_root=tmp_path,
    )
    other_supervisor = OmniVoiceSidecarSupervisor(
        provider_config={"extra_params": {"host": "127.0.0.1", "port": 8040}},
        repo_root=tmp_path,
    )

    assert supervisor.sidecar_token  # nosec B101
    assert supervisor.sidecar_token != other_supervisor.sidecar_token  # nosec B101

    await supervisor.ensure_started()

    assert recorded["trust_env"] is False  # nosec B101
    assert recorded["headers"]  # nosec B101
    request_url, headers = recorded["headers"][0]
    assert request_url.endswith("/health")  # nosec B101
    assert headers[X_TLDW_SIDECAR_TOKEN_HEADER] == supervisor.sidecar_token  # nosec B101


@pytest.mark.unit
@pytest.mark.asyncio
async def test_supervisor_retries_bounded_port_collisions_before_succeeding(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import omnivoice_sidecar_supervisor as supervisor_module
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_supervisor import OmniVoiceSidecarSupervisor

    probed_ports: list[int] = []
    spawn_ports: list[int] = []

    def _fake_is_port_free(host: str, port: int) -> bool:  # noqa: ARG001
        probed_ports.append(port)
        return port >= 8041

    class _FakeClient:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, *, headers=None, timeout=None):  # noqa: ARG002
            return httpx.Response(200, json={"status": "ok", "ready": True})

    async def _fake_create_subprocess_exec(*args, **kwargs):
        spawn_ports.append(int(kwargs["env"]["OMNIVOICE_SIDECAR_PORT"]))
        return _FakeProcess()

    def _fake_create_sidecar_async_client(*, timeout: float):  # noqa: ARG001
        return _FakeClient()

    monkeypatch.setattr(supervisor_module, "is_port_free", _fake_is_port_free, raising=True)
    monkeypatch.setattr(supervisor_module.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)
    monkeypatch.setattr(supervisor_module, "create_sidecar_async_client", _fake_create_sidecar_async_client, raising=True)

    supervisor = OmniVoiceSidecarSupervisor(
        provider_config={
            "extra_params": {
                "host": "127.0.0.1",
                "port": 8039,
                "autoselect_port": True,
                "port_probe_max": 3,
            }
        },
        repo_root=tmp_path,
    )

    await supervisor.ensure_started()

    assert probed_ports == [8039, 8040, 8041]  # nosec B101
    assert spawn_ports == [8041]  # nosec B101
    assert supervisor.port == 8041  # nosec B101


@pytest.mark.unit
@pytest.mark.asyncio
async def test_supervisor_refuses_spawn_after_close_requested(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_supervisor import OmniVoiceSidecarSupervisor

    supervisor = OmniVoiceSidecarSupervisor(
        provider_config={"extra_params": {"host": "127.0.0.1", "port": 8039}},
        repo_root=tmp_path,
    )
    supervisor.mark_closing()

    with pytest.raises(RuntimeError, match="closing"):
        await supervisor.ensure_started()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_supervisor_readiness_polling_fails_cleanly_when_sidecar_never_reaches_health(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import omnivoice_sidecar_supervisor as supervisor_module
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_supervisor import OmniVoiceSidecarSupervisor

    class _NeverReadyClient:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, *, headers=None, timeout=None):  # noqa: ARG002
            raise httpx.ConnectError("not ready")

        async def post(self, url, *, headers=None):  # noqa: ARG002
            return httpx.Response(500)

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ARG001
        return _FakeProcess()

    monkeypatch.setattr(supervisor_module, "is_port_free", lambda host, port: True, raising=True)
    monkeypatch.setattr(supervisor_module.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)
    monkeypatch.setattr(
        supervisor_module,
        "create_sidecar_async_client",
        lambda *, timeout: _NeverReadyClient(),
        raising=True,
    )

    supervisor = OmniVoiceSidecarSupervisor(
        provider_config={
            "extra_params": {
                "host": "127.0.0.1",
                "port": 8039,
                "healthcheck_timeout_seconds": 0.02,
                "healthcheck_interval_seconds": 0.01,
            }
        },
        repo_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="health") as excinfo:
        await supervisor.ensure_started()

    root_cause = excinfo.value
    while root_cause.__cause__ is not None:
        root_cause = root_cause.__cause__
    assert isinstance(root_cause, httpx.ConnectError)  # nosec B101
    assert supervisor.last_failure_at is not None  # nosec B101


@pytest.mark.unit
def test_supervisor_rejects_non_loopback_host(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_supervisor import OmniVoiceSidecarSupervisor

    with pytest.raises(ValueError, match="loopback"):
        OmniVoiceSidecarSupervisor(
            provider_config={"extra_params": {"host": "192.168.1.40", "port": 8039}},
            repo_root=tmp_path,
        )


@pytest.mark.unit
def test_supervisor_preserves_explicit_zero_config_values(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_supervisor import OmniVoiceSidecarSupervisor

    supervisor = OmniVoiceSidecarSupervisor(
        provider_config={
            "extra_params": {
                "host": "127.0.0.1",
                "port_probe_max": 0,
                "startup_backoff_seconds": 0,
                "idle_shutdown_seconds": 0,
            }
        },
        repo_root=tmp_path,
    )

    assert supervisor._port_probe_max == 0  # nosec B101
    assert supervisor._startup_backoff_seconds == 0  # nosec B101
    assert supervisor._idle_shutdown_seconds == 0  # nosec B101


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_if_idle_returns_false_without_live_process(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_supervisor import OmniVoiceSidecarSupervisor

    supervisor = OmniVoiceSidecarSupervisor(
        provider_config={"extra_params": {"host": "127.0.0.1", "idle_shutdown_seconds": 1}},
        repo_root=tmp_path,
    )
    supervisor._last_activity_at = 0

    stopped = await supervisor.shutdown_if_idle()

    assert stopped is False  # nosec B101


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_if_idle_clears_idle_state_after_stopping_live_process(tmp_path):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_supervisor import OmniVoiceSidecarSupervisor

    supervisor = OmniVoiceSidecarSupervisor(
        provider_config={"extra_params": {"host": "127.0.0.1", "idle_shutdown_seconds": 1}},
        repo_root=tmp_path,
    )
    supervisor._process = _FakeProcess()
    supervisor._last_activity_at = 0

    first = await supervisor.shutdown_if_idle()
    second = await supervisor.shutdown_if_idle()

    assert first is True  # nosec B101
    assert second is False  # nosec B101
    assert supervisor._last_activity_at is None  # nosec B101


@pytest.mark.unit
@pytest.mark.asyncio
async def test_supervisor_readiness_fails_fast_when_child_exits_during_startup(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import omnivoice_sidecar_supervisor as supervisor_module
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_supervisor import OmniVoiceSidecarSupervisor

    call_count = 0

    class _NeverReadyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, *, headers=None):  # noqa: ARG002
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("not ready")

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ARG001
        return _ExitedProcess(returncode=3)

    monkeypatch.setattr(supervisor_module, "is_port_free", lambda host, port: True, raising=True)
    monkeypatch.setattr(supervisor_module.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)
    monkeypatch.setattr(
        supervisor_module,
        "create_sidecar_async_client",
        lambda *, timeout: _NeverReadyClient(),
        raising=True,
    )

    supervisor = OmniVoiceSidecarSupervisor(
        provider_config={
            "extra_params": {
                "host": "127.0.0.1",
                "port": 8039,
                "healthcheck_timeout_seconds": 10,
                "healthcheck_interval_seconds": 0.01,
            }
        },
        repo_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="health"):
        await supervisor.ensure_started()

    assert call_count <= 1  # nosec B101


@pytest.mark.unit
@pytest.mark.asyncio
async def test_supervisor_spawn_sets_pythonpath_and_rotates_token_on_restart(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import omnivoice_sidecar_supervisor as supervisor_module
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_supervisor import OmniVoiceSidecarSupervisor

    spawn_envs: list[dict[str, str]] = []

    class _ReadyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, *, headers=None, timeout=None):  # noqa: ARG002
            return httpx.Response(200, json={"status": "ok", "ready": True})

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ARG001
        spawn_envs.append(dict(kwargs["env"]))
        return _FakeProcess()

    monkeypatch.setattr(supervisor_module, "is_port_free", lambda host, port: True, raising=True)
    monkeypatch.setattr(supervisor_module.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)
    monkeypatch.setattr(
        supervisor_module,
        "create_sidecar_async_client",
        lambda *, timeout: _ReadyClient(),
        raising=True,
    )

    supervisor = OmniVoiceSidecarSupervisor(
        provider_config={"extra_params": {"host": "127.0.0.1", "port": 8039}},
        repo_root=tmp_path,
    )

    await supervisor.ensure_started()
    first_token = supervisor.sidecar_token
    first_process = supervisor._process
    assert first_process is not None  # nosec B101

    first_process.returncode = 1
    await supervisor.ensure_started()

    assert len(spawn_envs) == 2  # nosec B101
    assert spawn_envs[0]["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path)  # nosec B101
    assert spawn_envs[0]["OMNIVOICE_SIDECAR_TOKEN"] != spawn_envs[1]["OMNIVOICE_SIDECAR_TOKEN"]  # nosec B101
    assert supervisor.sidecar_token != first_token  # nosec B101


@pytest.mark.unit
@pytest.mark.asyncio
async def test_supervisor_restarts_existing_process_after_idle_timeout(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import omnivoice_sidecar_supervisor as supervisor_module
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_supervisor import OmniVoiceSidecarSupervisor

    spawned_processes: list[_FakeProcess] = []

    class _ReadyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, *, headers=None, timeout=None):  # noqa: ARG002
            return httpx.Response(200, json={"status": "ok", "ready": True})

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ARG001
        process = _FakeProcess()
        spawned_processes.append(process)
        return process

    monkeypatch.setattr(supervisor_module, "is_port_free", lambda host, port: True, raising=True)
    monkeypatch.setattr(supervisor_module.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)
    monkeypatch.setattr(
        supervisor_module,
        "create_sidecar_async_client",
        lambda *, timeout: _ReadyClient(),
        raising=True,
    )

    supervisor = OmniVoiceSidecarSupervisor(
        provider_config={"extra_params": {"host": "127.0.0.1", "port": 8039, "idle_shutdown_seconds": 1}},
        repo_root=tmp_path,
    )

    await supervisor.ensure_started()
    supervisor._last_activity_at = 0

    await supervisor.ensure_started()

    assert len(spawned_processes) == 2  # nosec B101
    assert spawned_processes[0].terminate_called is True  # nosec B101
