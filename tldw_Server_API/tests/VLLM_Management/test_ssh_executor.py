from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.VLLM_Management.executors.ssh import SSHVLLMExecutor
from tldw_Server_API.app.core.VLLM_Management.models import VLLMInstanceRecord
from tldw_Server_API.app.core.VLLM_Management.service import ShellSSHRunner


class RecordingSSHRunner:
    def __init__(self) -> None:
        self.last_command: list[str] | None = None
        self.last_kwargs: dict[str, object] | None = None

    def run(self, command: list[str], **kwargs: object) -> dict[str, object]:
        self.last_command = command
        self.last_kwargs = kwargs
        return {"remote_pid": 9988}


def fake_ssh_instance() -> VLLMInstanceRecord:
    return VLLMInstanceRecord(
        instance_id="ssh-a",
        name="ssh-a",
        execution_mode="ssh",
        transport_config={
            "host": "gpu-a.internal",
            "port": 2222,
            "user": "mlops",
            "auth": {"secret_ref": "ssh-key-vllm-prod"},
            "launcher_path": "/usr/local/bin/tldw-vllm-launcher",
        },
        launch_spec={"model": "meta-llama/Llama-3.1-8B-Instruct", "port": 8002},
        routing_policy={},
        declared_capabilities={"chat": True},
        desired_state="running",
        observed_state="stopped",
        created_at="2026-03-10T00:00:00+00:00",
        updated_at="2026-03-10T00:00:00+00:00",
    )


def test_ssh_executor_uses_launcher_contract_not_shell_backgrounding():
    runner = RecordingSSHRunner()
    executor = SSHVLLMExecutor(ssh_runner=runner)

    result = executor.start(fake_ssh_instance())

    assert runner.last_command is not None
    assert "nohup" not in runner.last_command
    assert "&" not in runner.last_command
    assert runner.last_command[:2] == ["/usr/local/bin/tldw-vllm-launcher", "start"]
    assert result.handle["remote_pid"] == 9988
    assert runner.last_kwargs == {
        "host": "gpu-a.internal",
        "port": 2222,
        "user": "mlops",
        "auth": {"secret_ref": "ssh-key-vllm-prod"},
    }


def test_ssh_executor_accepts_username_alias():
    runner = RecordingSSHRunner()
    executor = SSHVLLMExecutor(ssh_runner=runner)
    instance = VLLMInstanceRecord(
        instance_id="ssh-b",
        name="ssh-b",
        execution_mode="ssh",
        transport_config={
            "host": "gpu-b.internal",
            "port": 2200,
            "username": "ubuntu",
            "auth": {"secret_ref": "ssh-key-vllm-prod"},
            "launcher_path": "/usr/local/bin/tldw-vllm-launcher",
        },
        launch_spec={"model": "meta-llama/Llama-3.1-8B-Instruct", "port": 8002},
        routing_policy={},
        declared_capabilities={"chat": True},
        desired_state="running",
        observed_state="stopped",
        created_at="2026-03-10T00:00:00+00:00",
        updated_at="2026-03-10T00:00:00+00:00",
    )

    executor.start(instance)

    assert runner.last_kwargs == {
        "host": "gpu-b.internal",
        "port": 2200,
        "user": "ubuntu",
        "auth": {"secret_ref": "ssh-key-vllm-prod"},
    }


def test_shell_ssh_runner_resolves_identity_file_from_secret_ref(monkeypatch):
    calls: dict[str, object] = {}

    def fake_run(argv, **kwargs):  # noqa: ANN001, ANN003
        calls["argv"] = list(argv)
        calls["kwargs"] = dict(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("VLLM_SSH_KEY_PATH", "/tmp/id_ed25519")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.VLLM_Management.service.subprocess.run",
        fake_run,
    )

    runner = ShellSSHRunner(connect_timeout_seconds=15)
    runner.run(
        ["/usr/local/bin/tldw-vllm-launcher", "start"],
        host="gpu-c.internal",
        port=2222,
        user="mlops",
        auth={"secret_ref": "VLLM_SSH_KEY_PATH"},
    )

    assert calls["argv"] == [
        "ssh",
        "-p",
        "2222",
        "-o",
        "ConnectTimeout=15",
        "-i",
        "/tmp/id_ed25519",
        "mlops@gpu-c.internal",
        "/usr/local/bin/tldw-vllm-launcher start",
    ]


def test_shell_ssh_runner_rejects_ssh_option_injection_host():
    runner = ShellSSHRunner()

    with pytest.raises(ValueError, match="Invalid SSH host"):
        runner.run(
            ["/usr/local/bin/tldw-vllm-launcher", "start"],
            host="-oProxyCommand=sh",
            port=22,
            user="mlops",
            auth=None,
        )


def test_shell_ssh_runner_redacts_identity_path_from_failure_detail(monkeypatch):
    def fake_run(argv, **kwargs):  # noqa: ANN001, ANN003
        return SimpleNamespace(
            returncode=255,
            stdout="",
            stderr="Load key \"/tmp/id_ed25519\": invalid format",
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.VLLM_Management.service.subprocess.run",
        fake_run,
    )

    runner = ShellSSHRunner()
    with pytest.raises(RuntimeError) as exc_info:
        runner.run(
            ["/usr/local/bin/tldw-vllm-launcher", "start"],
            host="gpu-d.internal",
            port=22,
            user="mlops",
            auth={"identity_file": "/tmp/id_ed25519"},
        )

    assert "/tmp/id_ed25519" not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)
