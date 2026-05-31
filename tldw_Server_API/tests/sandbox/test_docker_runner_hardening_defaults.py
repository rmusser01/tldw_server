from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sandbox.models import RunSpec, RuntimeType
from tldw_Server_API.app.core.Sandbox.runners.docker_runner import DockerRunner


class _StopAfterCreate(Exception):
    pass


def _capture_docker_create_command(monkeypatch, spec: RunSpec) -> list[str]:
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_AVAILABLE", "1")
    monkeypatch.delenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", raising=False)
    recorded_cmds: list[list[str]] = []

    def _fake_check_output(cmd, text: bool = False, timeout: int | None = None) -> str:
        cmd_list = list(cmd)
        recorded_cmds.append(cmd_list)
        if cmd_list[:3] == ["docker", "version", "--format"]:
            return "24.0.0"
        if cmd_list[:2] == ["docker", "create"]:
            raise _StopAfterCreate()
        raise AssertionError(f"Unexpected check_output call before docker create: {cmd_list!r}")

    monkeypatch.setattr("subprocess.check_output", _fake_check_output)
    runner = DockerRunner()
    with pytest.raises(_StopAfterCreate):
        runner.start_run(run_id="rid-hardening-1234", spec=spec)
    create_cmd = next((cmd for cmd in recorded_cmds if cmd[:2] == ["docker", "create"]), [])
    if not create_cmd:
        pytest.fail(f"docker create command not captured: {recorded_cmds!r}")
    return create_cmd


@pytest.mark.unit
def test_docker_runner_defaults_to_non_root_uid_gid_and_read_only_rootfs(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_DOCKER_DEFAULT_UID", raising=False)
    monkeypatch.delenv("SANDBOX_DOCKER_DEFAULT_GID", raising=False)
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        command=["python", "-c", "print('ok')"],
        timeout_sec=5,
        network_policy="deny_all",
        run_as_root=False,
        read_only_root=True,
    )
    create_cmd = _capture_docker_create_command(monkeypatch, spec)
    if "--read-only" not in create_cmd:
        pytest.fail(f"Expected --read-only in docker create cmd: {create_cmd!r}")
    if "--user" not in create_cmd:
        pytest.fail(f"Expected --user in docker create cmd: {create_cmd!r}")
    user_idx = create_cmd.index("--user")
    if create_cmd[user_idx + 1] != "1000:1000":
        pytest.fail(f"Expected --user 1000:1000, got {create_cmd[user_idx + 1]!r}")


@pytest.mark.unit
def test_docker_runner_bind_mounts_staged_workspace_for_inline_files(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_DOCKER_BIND_WORKSPACE", raising=False)
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        command=["python", "/workspace/hello.py"],
        timeout_sec=5,
        network_policy="deny_all",
        run_as_root=False,
        read_only_root=True,
        files_inline=[("hello.py", b"print('ok')\n")],
    )

    create_cmd = _capture_docker_create_command(monkeypatch, spec)

    staged_mounts = [
        create_cmd[idx + 1]
        for idx, token in enumerate(create_cmd[:-1])
        if token == "--mount" and "dst=/tldw-staged-workspace" in create_cmd[idx + 1]
    ]
    workspace_tmpfs = [
        create_cmd[idx + 1]
        for idx, token in enumerate(create_cmd[:-1])
        if token == "--tmpfs" and create_cmd[idx + 1].startswith("/workspace:")
    ]
    if not staged_mounts:
        pytest.fail(f"Expected read-only staged input mount, got: {create_cmd!r}")
    if "readonly" not in staged_mounts[0]:
        pytest.fail(f"Expected staged input mount to be readonly, got: {staged_mounts[0]!r}")
    if not workspace_tmpfs:
        pytest.fail(f"Expected hardened /workspace tmpfs to remain, got: {create_cmd!r}")
    shell_cmd = create_cmd[-1]
    if "/tldw-staged-workspace/." not in shell_cmd:
        pytest.fail(f"Expected shell prelude to copy staged inputs, got: {shell_cmd!r}")


@pytest.mark.unit
def test_docker_runner_uses_configured_non_root_uid_gid(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_DOCKER_DEFAULT_UID", "2001")
    monkeypatch.setenv("SANDBOX_DOCKER_DEFAULT_GID", "3002")
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        command=["python", "-c", "print('ok')"],
        timeout_sec=5,
        network_policy="deny_all",
        run_as_root=False,
        read_only_root=True,
    )
    create_cmd = _capture_docker_create_command(monkeypatch, spec)
    if "--user" not in create_cmd:
        pytest.fail(f"Expected --user in docker create cmd: {create_cmd!r}")
    user_idx = create_cmd.index("--user")
    if create_cmd[user_idx + 1] != "2001:3002":
        pytest.fail(f"Expected --user 2001:3002, got {create_cmd[user_idx + 1]!r}")


@pytest.mark.unit
def test_docker_runner_adds_ssh_caps_for_acp_internal_ssh_port(monkeypatch) -> None:
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        command=["/usr/local/bin/tldw-acp-entrypoint"],
        timeout_sec=5,
        network_policy="deny_all",
        run_as_root=False,
        read_only_root=True,
        env={"ACP_SSH_PORT": "2222"},
        port_mappings=[{"host_ip": "127.0.0.1", "host_port": 4567, "container_port": 2222}],
    )
    create_cmd = _capture_docker_create_command(monkeypatch, spec)
    if "-p" not in create_cmd:
        pytest.fail(f"Expected -p port mapping in docker create cmd: {create_cmd!r}")
    if "127.0.0.1:4567:2222" not in create_cmd:
        pytest.fail(f"Expected ACP SSH mapping 127.0.0.1:4567:2222, got: {create_cmd!r}")
    required_caps = {"SYS_CHROOT", "SETUID", "SETGID"}
    present_caps: set[str] = set()
    for idx, token in enumerate(create_cmd):
        if token.startswith("--cap") and token.endswith("add") and idx + 1 < len(create_cmd):
            present_caps.add(create_cmd[idx + 1])
    if not required_caps.issubset(present_caps):
        pytest.fail(f"Expected SSH cap-add set {required_caps!r}, got {present_caps!r}")
