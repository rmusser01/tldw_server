from pathlib import Path

from tldw_Server_API.app.core.VLLM_Management.executors.local import LocalVLLMExecutor
from tldw_Server_API.app.core.VLLM_Management.models import VLLMInstanceRecord


class FakeTask:
    def __init__(self, *, pid: int = 4321, pgid: int = 8765, stdout_path: Path, stderr_path: Path):
        self.pid = pid
        self.pgid = pgid
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path


def fake_local_instance() -> VLLMInstanceRecord:
    return VLLMInstanceRecord(
        instance_id="local-a",
        name="local-a",
        execution_mode="local",
        transport_config={"workdir": "/tmp/vllm", "log_dir": "/tmp/vllm-logs"},
        launch_spec={"model": "meta-llama/Llama-3.1-8B-Instruct", "port": 8002},
        routing_policy={},
        declared_capabilities={"chat": True},
        desired_state="running",
        observed_state="stopped",
        created_at="2026-03-10T00:00:00+00:00",
        updated_at="2026-03-10T00:00:00+00:00",
    )


def test_local_executor_starts_with_structured_argv_and_tracks_logs(tmp_path):
    calls: dict[str, object] = {}

    def fake_process_starter(cmd, workdir, log_dir):
        calls["cmd"] = cmd
        calls["workdir"] = workdir
        calls["log_dir"] = log_dir
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir_path / "stdout.log"
        stderr_path = log_dir_path / "stderr.log"
        stdout_path.touch()
        stderr_path.touch()
        return FakeTask(stdout_path=stdout_path, stderr_path=stderr_path)

    executor = LocalVLLMExecutor(process_starter=fake_process_starter, log_root=tmp_path)

    result = executor.start(fake_local_instance())

    assert calls["cmd"][:2] == ["vllm", "serve"]
    assert result.handle["pid"] == 4321
    assert result.handle["pgid"] == 8765
    assert result.log_handle["stdout_path"].endswith("stdout.log")
    assert result.log_handle["stderr_path"].endswith("stderr.log")


def test_local_executor_stop_uses_process_metadata_from_handle():
    calls: dict[str, object] = {}

    def fake_process_terminator(task, grace_ms=5000):
        calls["pid"] = task.pid
        calls["pgid"] = task.pgid
        calls["grace_ms"] = grace_ms
        return True, False

    executor = LocalVLLMExecutor(process_terminator=fake_process_terminator)
    handle = {
        "pid": 4321,
        "pgid": 8765,
        "stdout_path": "/tmp/stdout.log",
        "stderr_path": "/tmp/stderr.log",
        "command": ["vllm", "serve", "meta-llama/Llama-3.1-8B-Instruct"],
        "workdir": "/tmp/vllm",
    }

    result = executor.stop(fake_local_instance(), handle)

    assert calls == {"pid": 4321, "pgid": 8765, "grace_ms": 5000}
    assert result.status == "stopped"
    assert result.forced is False
