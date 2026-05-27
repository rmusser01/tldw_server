from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.config import ACPSandboxConfig
from tldw_Server_API.app.core.Agent_Client_Protocol.config import ACPRunnerConfig
from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import ACPRunnerClient
from tldw_Server_API.app.core.Agent_Client_Protocol.sandbox_runner_client import (
    ACPSandboxRunnerManager,
    SandboxSessionHandle,
    _is_self_referential_agent_command,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPMessage
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPResponseError
from tldw_Server_API.app.core.Sandbox.models import RuntimeType
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult
from tldw_Server_API.app.core.Sandbox.runners.lima_runner import LimaRunner
from tldw_Server_API.app.services.acp_runtime_policy_service import (
    ACPRuntimePolicySnapshot,
)


@pytest.mark.unit
def test_acp_sandbox_config_default_network_policy_is_deny_all() -> None:
    cfg = ACPSandboxConfig()
    if cfg.network_policy != "deny_all":
        pytest.fail(f"Expected ACP sandbox default network policy deny_all, got {cfg.network_policy!r}")


@pytest.mark.unit
def test_load_acp_sandbox_config_defaults_network_policy_to_deny_all(monkeypatch) -> None:
    import tldw_Server_API.app.core.Agent_Client_Protocol.config as acp_config

    monkeypatch.delenv("ACP_SANDBOX_NETWORK_POLICY", raising=False)
    monkeypatch.setattr(acp_config, "get_config_section", lambda name: {})
    cfg = acp_config.load_acp_sandbox_config()
    if cfg.network_policy != "deny_all":
        pytest.fail(f"Expected loader fallback network policy deny_all, got {cfg.network_policy!r}")


@pytest.mark.unit
def test_acp_sandbox_config_default_runtime_hardening_flags() -> None:
    cfg = ACPSandboxConfig()
    if cfg.run_as_root is not False:
        pytest.fail(f"Expected ACP sandbox default run_as_root=False, got {cfg.run_as_root!r}")
    if cfg.read_only_root is not True:
        pytest.fail(f"Expected ACP sandbox default read_only_root=True, got {cfg.read_only_root!r}")
    if cfg.ssh_container_port != 2222:
        pytest.fail(f"Expected ACP sandbox default ssh_container_port=2222, got {cfg.ssh_container_port!r}")


@pytest.mark.unit
def test_load_acp_sandbox_config_defaults_runtime_hardening_flags(monkeypatch) -> None:
    import tldw_Server_API.app.core.Agent_Client_Protocol.config as acp_config

    monkeypatch.delenv("ACP_SANDBOX_RUN_AS_ROOT", raising=False)
    monkeypatch.delenv("ACP_SANDBOX_READ_ONLY_ROOT", raising=False)
    monkeypatch.delenv("ACP_SSH_CONTAINER_PORT", raising=False)
    monkeypatch.setattr(acp_config, "get_config_section", lambda name: {})
    cfg = acp_config.load_acp_sandbox_config()
    if cfg.run_as_root is not False:
        pytest.fail(f"Expected loader fallback run_as_root=False, got {cfg.run_as_root!r}")
    if cfg.read_only_root is not True:
        pytest.fail(f"Expected loader fallback read_only_root=True, got {cfg.read_only_root!r}")
    if cfg.ssh_container_port != 2222:
        pytest.fail(f"Expected loader fallback ssh_container_port=2222, got {cfg.ssh_container_port!r}")


@pytest.mark.unit
def test_is_self_referential_agent_command_detects_runner_binary() -> None:
    cases = [
        ("tldw-agent-acp", True),
        ("/usr/local/bin/tldw-agent-acp", True),
        ("claude", False),
        ("/usr/local/bin/codex", False),
    ]
    for command, expected in cases:
        actual = _is_self_referential_agent_command(command)
        if actual != expected:
            pytest.fail(
                f"_is_self_referential_agent_command({command!r}) returned {actual}, expected {expected}"
            )


@pytest.mark.unit
def test_permission_policy_tier_resolution_matches_standard_runner(monkeypatch) -> None:
    from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import ACPRunnerClient
    import tldw_Server_API.app.services.admin_acp_sessions_service as store_src

    class _PolicyStore:
        def resolve_permission_tier(self, tool_name: str) -> str | None:
            if tool_name == "fs.write":
                return "individual"
            return None

    mock_config = MagicMock()
    mock_config.command = "echo"
    mock_config.args = []
    mock_config.env = {}
    mock_config.cwd = None
    mock_config.startup_timeout_sec = 10

    monkeypatch.setattr(store_src, "_store", _PolicyStore(), raising=False)

    runner = ACPRunnerClient(mock_config)
    sandbox = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            agent_command="/usr/local/bin/codex",
        )
    )

    assert runner._determine_permission_tier("fs.write") == "individual"
    assert sandbox._determine_permission_tier("fs.write") == "individual"
    assert runner._determine_permission_tier("git.status") == "auto"
    assert sandbox._determine_permission_tier("git.status") == "auto"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_session_rejects_self_referential_agent_command() -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            agent_command="/usr/local/bin/tldw-agent-acp",
        )
    )

    with pytest.raises(ACPResponseError, match="cannot be tldw-agent-acp"):
        await manager.create_session(cwd="/workspace")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_session_requires_authenticated_user_id() -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            agent_command="/usr/local/bin/codex",
        )
    )

    with pytest.raises(ACPResponseError, match="require an authenticated user_id"):
        await manager.create_session(cwd="/workspace")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_standard_runner_create_session_sends_session_env() -> None:
    class _CallResult:
        def __init__(self, result: dict[str, object]) -> None:
            self.result = result

    class _FakeClient:
        is_running = True

        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        async def call(self, method: str, payload: dict[str, object]):
            self.calls.append((method, payload))
            return _CallResult({"sessionId": "session-env"})

    runner = ACPRunnerClient(ACPRunnerConfig(command="echo"))
    fake_client = _FakeClient()
    runner._client = fake_client

    session_id = await runner.create_session(
        "/repo",
        session_env={"WORKSPACE_TOKEN": "abc"},
    )

    assert session_id == "session-env"
    assert fake_client.calls == [
        ("session/new", {"cwd": "/repo", "mcpServers": [], "env": {"WORKSPACE_TOKEN": "abc"}})
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_standard_runner_create_session_sends_explicit_empty_mcp_servers() -> None:
    class _CallResult:
        def __init__(self, result: dict[str, object]) -> None:
            self.result = result

    class _FakeClient:
        is_running = True

        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        async def call(self, method: str, payload: dict[str, object]):
            self.calls.append((method, payload))
            return _CallResult({"sessionId": "session-empty-mcp"})

    runner = ACPRunnerClient(ACPRunnerConfig(command="echo"))
    fake_client = _FakeClient()
    runner._client = fake_client

    session_id = await runner.create_session(
        "/repo",
        mcp_servers=[],
        agent_type="hermes",
    )

    assert session_id == "session-empty-mcp"
    assert fake_client.calls == [
        ("session/new", {"cwd": "/repo", "mcpServers": [], "agentType": "hermes"})
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_standard_runner_create_session_retries_without_agent_type_for_native_acp() -> None:
    class _CallResult:
        def __init__(self, result: dict[str, object]) -> None:
            self.result = result

    class _FakeClient:
        is_running = True

        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        async def call(self, method: str, payload: dict[str, object]):
            self.calls.append((method, payload))
            if payload.get("agentType") == "hermes":
                raise ACPResponseError("Invalid params")
            return _CallResult({"sessionId": "session-native-hermes"})

    runner = ACPRunnerClient(ACPRunnerConfig(command="echo"))
    fake_client = _FakeClient()
    runner._client = fake_client

    session_id = await runner.create_session("/repo", agent_type="hermes")

    assert session_id == "session-native-hermes"
    assert fake_client.calls == [
        ("session/new", {"cwd": "/repo", "mcpServers": [], "agentType": "hermes"}),
        ("session/new", {"cwd": "/repo", "mcpServers": []}),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_session_rejects_unavailable_vz_macos_runtime(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            runtime="vz_macos",
            network_policy="deny_all",
            agent_command="/usr/local/bin/codex",
        )
    )
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_MACOS_AVAILABLE", "0")

    with pytest.raises(ACPResponseError, match="vz_macos"):
        await manager.create_session(cwd="/workspace", user_id=7)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_session_rejects_seatbelt_without_standard_opt_in(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            runtime="seatbelt",
            network_policy="deny_all",
            agent_command="/usr/local/bin/codex",
        )
    )
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")
    monkeypatch.setenv("TLDW_SANDBOX_SEATBELT_AVAILABLE", "1")
    monkeypatch.delenv("TLDW_SANDBOX_SEATBELT_STANDARD_ENABLED", raising=False)

    with pytest.raises(ACPResponseError, match="seatbelt_standard_disabled"):
        await manager.create_session(cwd="/workspace", user_id=7)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sandbox_create_session_merges_config_and_session_env(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            agent_command="/usr/local/bin/codex",
            agent_env={"CONFIG_ONLY": "yes", "WORKSPACE_TOKEN": "config"},
            ssh_enabled=False,
        )
    )
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")
    monkeypatch.setattr(manager, "_validate_runtime_requirements", lambda _runtime: None)

    import tldw_Server_API.app.core.Agent_Client_Protocol.sandbox_runner_client as src

    class _Obj:
        def __init__(self, obj_id: str) -> None:
            self.id = obj_id

    capture: dict[str, object] = {}

    class _FakeSandboxService:
        def create_session(self, **kwargs):
            capture["session_spec"] = kwargs.get("spec")
            return _Obj("sandbox-session-env")

        def start_run_scaffold(self, **kwargs):
            capture["run_spec"] = kwargs.get("spec")
            return _Obj("run-env")

        def cancel_run(self, _run_id: str) -> None:
            return None

        def destroy_session(self, _session_id: str) -> None:
            return None

    class _FakeHub:
        def subscribe_with_buffer(self, _run_id: str) -> asyncio.Queue:
            return asyncio.Queue()

        def push_stdin(self, _run_id: str, _data: bytes) -> None:
            return None

    class _CallResult:
        def __init__(self, result: dict[str, object]) -> None:
            self.result = result

    class _FakeStreamClient:
        def __init__(self, send_bytes) -> None:
            self._send_bytes = send_bytes
            self.calls: list[tuple[str, dict[str, object]]] = []

        def set_notification_handler(self, _handler) -> None:
            return None

        def set_request_handler(self, _handler) -> None:
            return None

        async def start(self) -> None:
            return None

        async def call(self, method: str, payload: dict[str, object]):
            self.calls.append((method, payload))
            if method == "initialize":
                return _CallResult({"agentCapabilities": {}})
            if method == "session/new":
                return _CallResult({"sessionId": "acp-session-env"})
            if method == "_tldw/session/close":
                return _CallResult({})
            raise AssertionError(f"unexpected method: {method}")

        async def close(self) -> None:
            return None

    class _Store:
        def put_acp_session_control(self, **_kwargs) -> None:
            return None

        def delete_acp_session_control(self, _session_id: str) -> bool:
            return True

    monkeypatch.setattr(src.sandbox_ep, "_service", _FakeSandboxService(), raising=True)
    monkeypatch.setattr(src, "get_hub", lambda: _FakeHub())
    monkeypatch.setattr(src, "ACPStreamClient", _FakeStreamClient)
    monkeypatch.setattr(manager, "_get_sandbox_store", lambda: _Store())

    session_id = await manager.create_session(
        cwd="/workspace",
        user_id=7,
        session_env={"WORKSPACE_TOKEN": "abc"},
    )

    assert session_id == "acp-session-env"
    run_spec = capture["run_spec"]
    agent_env = json.loads(run_spec.env["ACP_AGENT_ENV_JSON"])
    assert agent_env == {"CONFIG_ONLY": "yes", "WORKSPACE_TOKEN": "abc"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_session_persists_stream_backed_control_for_vz_linux(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            runtime="vz_linux",
            network_policy="deny_all",
            agent_command="/usr/local/bin/codex",
            ssh_enabled=False,
        )
    )
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", "1")

    import tldw_Server_API.app.core.Agent_Client_Protocol.sandbox_runner_client as src

    class _Obj:
        def __init__(self, obj_id: str) -> None:
            self.id = obj_id

    capture: dict[str, object] = {}
    persisted: dict[str, object] = {}

    class _FakeSandboxService:
        def __init__(self) -> None:
            self.cancelled: list[str] = []
            self.destroyed: list[str] = []

        def create_session(self, **kwargs):
            capture["session_spec"] = kwargs.get("spec")
            return _Obj("sandbox-session-vz")

        def start_run_scaffold(self, **kwargs):
            capture["run_spec"] = kwargs.get("spec")
            return _Obj("run-vz")

        def cancel_run(self, run_id: str) -> None:
            self.cancelled.append(run_id)

        def destroy_session(self, session_id: str) -> None:
            self.destroyed.append(session_id)

    class _FakeHub:
        def subscribe_with_buffer(self, run_id: str) -> asyncio.Queue:
            return asyncio.Queue()

        def push_stdin(self, run_id: str, data: bytes) -> None:
            return None

    class _CallResult:
        def __init__(self, result: dict[str, object]) -> None:
            self.result = result

    class _FakeStreamClient:
        def __init__(self, send_bytes) -> None:
            self._send_bytes = send_bytes

        def set_notification_handler(self, handler) -> None:
            return None

        def set_request_handler(self, handler) -> None:
            return None

        async def start(self) -> None:
            return None

        async def call(self, method: str, payload: dict[str, object]):
            if method == "initialize":
                return _CallResult({"agentCapabilities": {}})
            if method == "session/new":
                return _CallResult({"sessionId": "acp-session-vz"})
            if method == "_tldw/session/close":
                return _CallResult({})
            raise AssertionError(f"unexpected method: {method}")

        async def close(self) -> None:
            return None

    class _Store:
        def put_acp_session_control(self, **kwargs) -> None:
            persisted.update(kwargs)

        def delete_acp_session_control(self, session_id: str) -> bool:
            return True

    fake_service = _FakeSandboxService()
    monkeypatch.setattr(src.sandbox_ep, "_service", fake_service, raising=True)
    monkeypatch.setattr(src, "get_hub", lambda: _FakeHub())
    monkeypatch.setattr(src, "ACPStreamClient", _FakeStreamClient)
    monkeypatch.setattr(manager, "_get_sandbox_store", lambda: _Store())

    session_id = await manager.create_session(cwd="/workspace", user_id=7)

    assert session_id == "acp-session-vz"
    assert getattr(capture.get("session_spec"), "runtime", None) == RuntimeType.vz_linux
    assert persisted["sandbox_session_id"] == "sandbox-session-vz"
    assert persisted["run_id"] == "run-vz"

    await manager.close_session(session_id)


@pytest.mark.unit
def test_validate_runtime_requirements_uses_single_lima_preflight(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            runtime="lima",
            network_policy="deny_all",
            agent_command="/usr/local/bin/codex",
        )
    )
    calls = {"count": 0}

    def _fake_preflight(self, network_policy: str | None = None):
        del self, network_policy
        calls["count"] += 1
        return RuntimePreflightResult(
            runtime=RuntimeType.lima,
            available=True,
            reasons=[],
            host={"os": "darwin"},
            enforcement_ready={"deny_all": True, "allowlist": True},
        )

    monkeypatch.setattr(LimaRunner, "preflight", _fake_preflight)

    manager._validate_runtime_requirements(RuntimeType.lima)

    assert calls["count"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_session_propagates_non_root_read_only_and_internal_ssh_port(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            agent_command="/usr/local/bin/codex",
            network_policy="deny_all",
            run_as_root=False,
            read_only_root=True,
            ssh_enabled=True,
            ssh_container_port=2222,
        )
    )
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")

    import tldw_Server_API.app.core.Agent_Client_Protocol.sandbox_runner_client as src

    class _Obj:
        def __init__(self, obj_id: str) -> None:
            self.id = obj_id

    capture: dict[str, object] = {}

    class _FakeSandboxService:
        def __init__(self) -> None:
            self.cancelled: list[str] = []
            self.destroyed: list[str] = []

        def create_session(self, **kwargs):
            capture["session_spec"] = kwargs.get("spec")
            return _Obj("sandbox-session-hardened")

        def start_run_scaffold(self, **kwargs):
            capture["run_spec"] = kwargs.get("spec")
            return _Obj("run-hardened")

        def cancel_run(self, run_id: str) -> None:
            self.cancelled.append(run_id)

        def destroy_session(self, session_id: str) -> None:
            self.destroyed.append(session_id)

    class _FakeHub:
        def subscribe_with_buffer(self, run_id: str) -> asyncio.Queue:
            return asyncio.Queue()

        def push_stdin(self, run_id: str, data: bytes) -> None:
            return None

    class _CallResult:
        def __init__(self, result: dict[str, object]) -> None:
            self.result = result

    class _FakeStreamClient:
        def __init__(self, send_bytes) -> None:
            self._send_bytes = send_bytes

        def set_notification_handler(self, handler) -> None:
            return None

        def set_request_handler(self, handler) -> None:
            return None

        async def start(self) -> None:
            return None

        async def call(self, method: str, payload: dict[str, object]):
            if method == "initialize":
                return _CallResult({"agentCapabilities": {}})
            if method == "session/new":
                return _CallResult({"sessionId": "acp-session-hardened"})
            if method == "_tldw/session/close":
                return _CallResult({})
            raise AssertionError(f"unexpected method: {method}")

        async def close(self) -> None:
            return None

    class _Store:
        def put_acp_session_control(self, **kwargs) -> None:
            return None

        def delete_acp_session_control(self, session_id: str) -> bool:
            return True

    fake_service = _FakeSandboxService()
    monkeypatch.setattr(src.sandbox_ep, "_service", fake_service, raising=True)
    monkeypatch.setattr(src, "get_hub", lambda: _FakeHub())
    monkeypatch.setattr(src, "ACPStreamClient", _FakeStreamClient)
    monkeypatch.setattr(manager, "_get_sandbox_store", lambda: _Store())
    monkeypatch.setattr(manager, "_generate_ssh_keypair", lambda: ("PRIVATE", "ssh-ed25519 AAAATEST"))

    async def _fake_allocate_ssh_port() -> int:
        return 4567

    monkeypatch.setattr(manager, "_allocate_ssh_port", _fake_allocate_ssh_port)

    session_id = await manager.create_session(cwd="/workspace", user_id=55)
    if session_id != "acp-session-hardened":
        pytest.fail(f"Expected acp-session-hardened, got {session_id!r}")

    session_spec = capture.get("session_spec")
    run_spec = capture.get("run_spec")
    if session_spec is None:
        pytest.fail("Expected sandbox session spec capture")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sandbox_permission_request_denies_tool_from_runtime_snapshot(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            agent_command="/usr/local/bin/codex",
        )
    )

    async def _fake_snapshot(
        session_id: str,
        *,
        force_refresh: bool = False,
    ) -> ACPRuntimePolicySnapshot | None:
        del session_id, force_refresh
        return ACPRuntimePolicySnapshot(
            session_id="sandbox-session",
            user_id=55,
            policy_snapshot_version="resolved-v1",
            policy_snapshot_fingerprint="sandbox-deny",
            policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
            policy_summary={},
            policy_provenance_summary={"source_kinds": ["profile"]},
            resolved_policy_document={"denied_tools": ["exec.run"]},
            approval_summary={},
            context_summary={},
            execution_config={},
        )

    monkeypatch.setattr(manager, "_get_runtime_policy_snapshot", _fake_snapshot, raising=False)

    async def _fake_check_permission_governance(*args, **kwargs):
        del args, kwargs
        return None

    monkeypatch.setattr(manager, "check_permission_governance", _fake_check_permission_governance, raising=False)

    response = await manager._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="perm-sandbox-deny",
            method="session/request_permission",
            params={
                "sessionId": "sandbox-session",
                "tool": {"name": "exec.run", "input": {"command": "whoami"}},
            },
        )
    )

    assert response.result == {
        "outcome": {
            "outcome": "denied",
            "deny_reason": "tool_denied_by_policy",
            "policy_snapshot_fingerprint": "sandbox-deny",
            "provenance_summary": {"source_kinds": ["profile"]},
        }
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sandbox_permission_request_refreshes_runtime_snapshot_when_policy_changes(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            agent_command="/usr/local/bin/codex",
        )
    )
    session_id = "sandbox-refresh-policy"
    policy_state = {"allowed_tools": ["web.search"], "fingerprint": "sandbox-allow"}
    call_count = {"build_snapshot": 0}

    async def _fake_check_permission_governance(*args, **kwargs):
        del args, kwargs
        return None

    manager.check_permission_governance = _fake_check_permission_governance  # type: ignore[assignment]

    class _Store:
        def __init__(self) -> None:
            self.record = type(
                "_Record",
                (),
                {
                    "session_id": session_id,
                    "user_id": 9,
                    "policy_snapshot_fingerprint": None,
                    "policy_snapshot_version": None,
                    "policy_snapshot_refreshed_at": None,
                    "policy_summary": None,
                    "policy_provenance_summary": None,
                    "policy_refresh_error": None,
                },
            )()

        async def get_session(self, _session_id: str):
            return self.record

        async def update_policy_snapshot_state(self, _session_id: str, **kwargs):
            for key, value in kwargs.items():
                setattr(self.record, key, value)
            return self.record

    class _RuntimePolicyService:
        async def build_snapshot(self, **kwargs):
            del kwargs
            call_count["build_snapshot"] += 1
            return ACPRuntimePolicySnapshot(
                session_id=session_id,
                user_id=9,
                policy_snapshot_version="resolved-v1",
                policy_snapshot_fingerprint=policy_state["fingerprint"],
                policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
                policy_summary={"allowed_tool_count": len(policy_state["allowed_tools"])},
                policy_provenance_summary={"source_kinds": ["capability_mapping"]},
                resolved_policy_document={"allowed_tools": list(policy_state["allowed_tools"])},
                approval_summary={"mode": "allow"},
                context_summary={},
                execution_config={},
            )

        async def persist_snapshot(self, *, session_store, snapshot):
            return await session_store.update_policy_snapshot_state(
                snapshot.session_id,
                policy_snapshot_version=snapshot.policy_snapshot_version,
                policy_snapshot_fingerprint=snapshot.policy_snapshot_fingerprint,
                policy_snapshot_refreshed_at=snapshot.policy_snapshot_refreshed_at,
                policy_summary=snapshot.policy_summary,
                policy_provenance_summary=snapshot.policy_provenance_summary,
                policy_refresh_error=snapshot.refresh_error,
            )

    store = _Store()

    async def _get_store():
        return store

    manager._runtime_policy_service = _RuntimePolicyService()  # type: ignore[assignment]
    manager._get_acp_session_store = _get_store  # type: ignore[assignment]
    manager._should_force_runtime_policy_refresh = lambda *, tier: True  # type: ignore[assignment]

    allowed_response = await manager._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="sandbox-perm-allow",
            method="session/request_permission",
            params={
                "sessionId": session_id,
                "tool": {"name": "web.search", "input": {"query": "opa"}},
            },
        )
    )
    assert allowed_response.result == {"outcome": {"outcome": "approved"}}

    policy_state["allowed_tools"] = ["fs.read"]
    policy_state["fingerprint"] = "sandbox-updated"

    denied_response = await manager._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="sandbox-perm-deny",
            method="session/request_permission",
            params={
                "sessionId": session_id,
                "tool": {"name": "web.search", "input": {"query": "opa"}},
            },
        )
    )
    assert denied_response.result == {
        "outcome": {
            "outcome": "denied",
            "deny_reason": "tool_not_allowed_by_policy",
            "policy_snapshot_fingerprint": "sandbox-updated",
            "provenance_summary": {"source_kinds": ["capability_mapping"]},
        }
    }
    assert call_count["build_snapshot"] == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_low_risk_permission_reuses_cached_runtime_snapshot(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            agent_command="/usr/local/bin/codex",
        )
    )
    session_id = "sandbox-refresh-cache"
    policy_state = {"allowed_tools": ["web.search"], "fingerprint": "sandbox-cache"}
    call_count = {"build_snapshot": 0}

    async def _fake_check_permission_governance(*args, **kwargs):
        del args, kwargs
        return None

    manager.check_permission_governance = _fake_check_permission_governance  # type: ignore[assignment]

    class _Store:
        def __init__(self) -> None:
            self.record = type(
                "_Record",
                (),
                {
                    "session_id": session_id,
                    "user_id": 9,
                    "policy_snapshot_fingerprint": None,
                    "policy_snapshot_version": None,
                    "policy_snapshot_refreshed_at": None,
                    "policy_summary": None,
                    "policy_provenance_summary": None,
                    "policy_refresh_error": None,
                },
            )()

        async def get_session(self, _session_id: str):
            return self.record

        async def update_policy_snapshot_state(self, _session_id: str, **kwargs):
            for key, value in kwargs.items():
                setattr(self.record, key, value)
            return self.record

    class _RuntimePolicyService:
        async def build_snapshot(self, **kwargs):
            del kwargs
            call_count["build_snapshot"] += 1
            return ACPRuntimePolicySnapshot(
                session_id=session_id,
                user_id=9,
                policy_snapshot_version="resolved-v1",
                policy_snapshot_fingerprint=policy_state["fingerprint"],
                policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
                policy_summary={"allowed_tool_count": len(policy_state["allowed_tools"])},
                policy_provenance_summary={"source_kinds": ["capability_mapping"]},
                resolved_policy_document={"allowed_tools": list(policy_state["allowed_tools"])},
                approval_summary={"mode": "allow"},
                context_summary={},
                execution_config={},
            )

        async def persist_snapshot(self, *, session_store, snapshot):
            return await session_store.update_policy_snapshot_state(
                snapshot.session_id,
                policy_snapshot_version=snapshot.policy_snapshot_version,
                policy_snapshot_fingerprint=snapshot.policy_snapshot_fingerprint,
                policy_snapshot_refreshed_at=snapshot.policy_snapshot_refreshed_at,
                policy_summary=snapshot.policy_summary,
                policy_provenance_summary=snapshot.policy_provenance_summary,
                policy_refresh_error=snapshot.refresh_error,
            )

    store = _Store()

    async def _get_store():
        return store

    manager._runtime_policy_service = _RuntimePolicyService()  # type: ignore[assignment]
    manager._get_acp_session_store = _get_store  # type: ignore[assignment]

    first = await manager._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="sandbox-perm-cache-1",
            method="session/request_permission",
            params={
                "sessionId": session_id,
                "tool": {"name": "web.search", "input": {"query": "opa"}},
            },
        )
    )
    assert first.result == {"outcome": {"outcome": "approved"}}
    assert call_count["build_snapshot"] == 1

    policy_state["allowed_tools"] = ["fs.read"]
    policy_state["fingerprint"] = "sandbox-cache-updated"

    second = await manager._handle_request(
        ACPMessage(
            jsonrpc="2.0",
            id="sandbox-perm-cache-2",
            method="session/request_permission",
            params={
                "sessionId": session_id,
                "tool": {"name": "web.search", "input": {"query": "opa"}},
            },
        )
    )
    assert second.result == {"outcome": {"outcome": "approved"}}
    assert call_count["build_snapshot"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_session_rolls_back_when_control_record_persist_fails(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            agent_command="/usr/local/bin/codex",
            ssh_enabled=False,
        )
    )
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")

    import tldw_Server_API.app.core.Agent_Client_Protocol.sandbox_runner_client as src

    class _Obj:
        def __init__(self, obj_id: str) -> None:
            self.id = obj_id

    class _FakeSandboxService:
        def __init__(self) -> None:
            self.cancelled: list[str] = []
            self.destroyed: list[str] = []

        def create_session(self, **kwargs):
            return _Obj("sandbox-session-atomic")

        def start_run_scaffold(self, **kwargs):
            return _Obj("run-atomic")

        def cancel_run(self, run_id: str) -> None:
            self.cancelled.append(run_id)

        def destroy_session(self, session_id: str) -> None:
            self.destroyed.append(session_id)

    class _FakeHub:
        def subscribe_with_buffer(self, run_id: str) -> asyncio.Queue:
            return asyncio.Queue()

        def push_stdin(self, run_id: str, data: bytes) -> None:
            return None

    class _CallResult:
        def __init__(self, result: dict[str, object]) -> None:
            self.result = result

    class _FakeStreamClient:
        def __init__(self, send_bytes) -> None:
            self._send_bytes = send_bytes

        def set_notification_handler(self, handler) -> None:
            return None

        def set_request_handler(self, handler) -> None:
            return None

        async def start(self) -> None:
            return None

        async def call(self, method: str, payload: dict[str, object]):
            if method == "initialize":
                return _CallResult({"agentCapabilities": {}})
            if method == "session/new":
                return _CallResult({"sessionId": "acp-session-atomic"})
            raise AssertionError(f"unexpected method: {method}")

        async def close(self) -> None:
            return None

    class _BrokenStore:
        def put_acp_session_control(self, **kwargs) -> None:
            raise RuntimeError("store_write_failed")

    fake_service = _FakeSandboxService()
    monkeypatch.setattr(src.sandbox_ep, "_service", fake_service, raising=True)
    monkeypatch.setattr(src, "get_hub", lambda: _FakeHub())
    monkeypatch.setattr(src, "ACPStreamClient", _FakeStreamClient)
    monkeypatch.setattr(manager, "_get_sandbox_store", lambda: _BrokenStore())

    with pytest.raises(ACPResponseError, match="persist ACP sandbox session control metadata"):
        await manager.create_session(cwd="/workspace", user_id=88)

    await asyncio.sleep(0)
    if fake_service.cancelled != ["run-atomic"]:
        pytest.fail(f"Expected run rollback cancellation, got: {fake_service.cancelled!r}")
    if fake_service.destroyed != ["sandbox-session-atomic"]:
        pytest.fail(f"Expected sandbox session rollback destroy, got: {fake_service.destroyed!r}")
    if "acp-session-atomic" in manager._sessions:
        pytest.fail("Create-session rollback left stale in-memory ACP session handle")
    if "acp-session-atomic" in manager._updates:
        pytest.fail("Create-session rollback left stale ACP update queue state")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_control_record_fallback_supports_access_and_metadata(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            agent_command="/usr/local/bin/codex",
        )
    )
    control_row = {
        "id": "acp-session-1",
        "user_id": "42",
        "sandbox_session_id": "sandbox-session-1",
        "run_id": "run-1",
        "ssh_host": "127.0.0.1",
        "ssh_port": 4022,
        "ssh_user": "acp",
        "ssh_private_key": "PRIVATE",
        "persona_id": "persona-1",
        "workspace_id": "workspace-1",
        "workspace_group_id": "wg-1",
        "scope_snapshot_id": "scope-1",
    }
    monkeypatch.setattr(manager, "_load_control_record", lambda session_id: dict(control_row))

    assert await manager.verify_session_access("acp-session-1", 42) is True
    assert await manager.verify_session_access("acp-session-1", 7) is False

    metadata = await manager.get_session_metadata("acp-session-1", user_id=42)
    assert metadata is not None
    assert metadata["sandbox_session_id"] == "sandbox-session-1"
    assert metadata["sandbox_run_id"] == "run-1"
    assert metadata["ssh_user"] == "acp"
    assert metadata["persona_id"] == "persona-1"

    ssh_info = await manager.get_ssh_info("acp-session-1", user_id=42)
    assert ssh_info is not None
    assert ssh_info[0] == "127.0.0.1"
    assert ssh_info[1] == 4022
    assert ssh_info[2] == "acp"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_session_rehydrates_and_caches_from_control_record(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            agent_command="/usr/local/bin/codex",
        )
    )

    class _DummyClient:
        async def close(self) -> None:
            return None

    task = asyncio.create_task(asyncio.sleep(60))
    handle = SandboxSessionHandle(
        session_id="acp-session-2",
        user_id=99,
        sandbox_session_id="sandbox-session-2",
        run_id="run-2",
        client=_DummyClient(),  # type: ignore[arg-type]
        reader_task=task,
    )

    control_row = {
        "id": "acp-session-2",
        "user_id": "99",
        "sandbox_session_id": "sandbox-session-2",
        "run_id": "run-2",
    }
    monkeypatch.setattr(manager, "_load_control_record", lambda session_id: dict(control_row))
    attach_calls = {"count": 0}

    async def _fake_attach(control):
        attach_calls["count"] += 1
        return handle

    monkeypatch.setattr(manager, "_attach_to_existing_session", _fake_attach)
    monkeypatch.setattr(manager, "_store_control_record", lambda sess: None)

    try:
        restored = await manager._get_session("acp-session-2", required=True)
        assert restored is handle
        restored_again = await manager._get_session("acp-session-2", required=True)
        assert restored_again is handle
        assert attach_calls["count"] == 1
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.unit
@pytest.mark.asyncio
async def test_close_session_falls_back_to_persisted_control(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            agent_command="/usr/local/bin/codex",
        )
    )

    control_row = {
        "id": "acp-session-3",
        "user_id": "101",
        "sandbox_session_id": "sandbox-session-3",
        "run_id": "run-3",
        "ssh_port": 4044,
    }
    monkeypatch.setattr(manager, "_get_session", lambda session_id, required=False: asyncio.sleep(0, result=None))
    monkeypatch.setattr(manager, "_load_control_record", lambda session_id: dict(control_row))
    deleted = {"value": False}

    def _fake_delete(session_id: str) -> bool:
        deleted["value"] = True
        return True

    monkeypatch.setattr(manager, "_delete_control_record", _fake_delete)

    released_ports: list[int] = []

    async def _fake_release(port: int) -> None:
        released_ports.append(port)

    monkeypatch.setattr(manager, "_release_ssh_port", _fake_release)

    import tldw_Server_API.app.core.Agent_Client_Protocol.sandbox_runner_client as src

    class _FakeSandboxService:
        def __init__(self) -> None:
            self.cancelled: list[str] = []
            self.destroyed: list[str] = []

        def cancel_run(self, run_id: str) -> None:
            self.cancelled.append(run_id)

        def destroy_session(self, session_id: str) -> None:
            self.destroyed.append(session_id)

    fake_service = _FakeSandboxService()
    monkeypatch.setattr(src.sandbox_ep, "_service", fake_service, raising=True)

    await manager.close_session("acp-session-3")
    assert fake_service.cancelled == ["run-3"]
    assert fake_service.destroyed == ["sandbox-session-3"]
    assert released_ports == [4044]
    assert deleted["value"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_close_session_clears_runtime_policy_cache_and_refresh_task(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(
        ACPSandboxConfig(
            enabled=True,
            agent_command="/usr/local/bin/codex",
        )
    )
    session_id = "acp-session-runtime-cache"

    monkeypatch.setattr(manager, "_get_session", lambda sid, required=False: asyncio.sleep(0, result=None))
    monkeypatch.setattr(manager, "_load_control_record", lambda sid: {"id": session_id, "user_id": "1"})
    monkeypatch.setattr(manager, "_delete_control_record", lambda sid: True)

    manager._runtime_policy_snapshots[session_id] = ACPRuntimePolicySnapshot(
        session_id=session_id,
        user_id=1,
        policy_snapshot_version="resolved-v1",
        policy_snapshot_fingerprint="snapshot-cache",
        policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
        policy_summary={},
        policy_provenance_summary={},
        resolved_policy_document={"allowed_tools": ["web.search"]},
        approval_summary={},
        context_summary={},
        execution_config={},
    )

    async def _pending_refresh() -> ACPRuntimePolicySnapshot | None:
        await asyncio.sleep(10)
        return None

    refresh_task = asyncio.create_task(_pending_refresh())
    manager._runtime_policy_refresh_tasks[session_id] = refresh_task

    await manager.close_session(session_id)
    await asyncio.sleep(0)

    assert session_id not in manager._runtime_policy_snapshots
    assert session_id not in manager._runtime_policy_refresh_tasks
    assert refresh_task.cancelled() is True
