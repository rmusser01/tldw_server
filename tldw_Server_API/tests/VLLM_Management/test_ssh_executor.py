from tldw_Server_API.app.core.VLLM_Management.executors.ssh import SSHVLLMExecutor
from tldw_Server_API.app.core.VLLM_Management.models import VLLMInstanceRecord


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
