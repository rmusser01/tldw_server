from __future__ import annotations

import asyncio

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

        async def get(self, url, *, headers=None):
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

        async def get(self, url, *, headers=None):  # noqa: ARG002
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

        async def get(self, url, *, headers=None):  # noqa: ARG002
            raise httpx.ConnectError("not ready")

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

    with pytest.raises(RuntimeError, match="health"):
        await supervisor.ensure_started()

    assert supervisor.last_failure_at is not None  # nosec B101
