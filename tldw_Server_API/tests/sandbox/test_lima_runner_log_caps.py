from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import tldw_Server_API.app.core.Sandbox.runners.lima_runner as lima_module
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType
from tldw_Server_API.app.core.Sandbox.runners.lima_runner import LimaRunner


def test_lima_tail_reads_newline_free_output_in_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_path = tmp_path / "run.log"
    count = 3 * 1024 * 1024
    log_path.write_bytes(b"a" * count)
    state = {"stop": False}
    chunks: list[int] = []

    def publish(run_id: str, data: bytes, cap: int) -> None:
        del run_id, cap
        chunks.append(len(data))
        if sum(chunks) >= count:
            state["stop"] = True

    monkeypatch.setattr(lima_module, "get_hub", lambda: SimpleNamespace(publish_stdout=publish))
    LimaRunner._tail_log("long-lima-log", str(log_path), state)

    assert sum(chunks) == count
    assert max(chunks) <= 64 * 1024


def test_lima_entry_script_bounds_stored_output_and_preserves_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SANDBOX_MAX_LOG_BYTES", "5")
    LimaRunner._write_entry_script(
        str(tmp_path), ["/bin/sh", "-c", "head -c 1048576 /dev/zero; head -c 1048576 /dev/zero >&2; exit 7"]
    )
    entry = tmp_path / "entry.sh"
    script = entry.read_text().replace("/workspace", str(tmp_path))
    script = script.replace("date +%s%3N 2>/dev/null || date +%s", "date +%s")
    entry.write_text(script)

    result = subprocess.run(["/bin/sh", str(entry)], capture_output=True, timeout=10)

    assert result.returncode == 7
    # The extra byte records truncation without retaining further guest output.
    assert (tmp_path / "run.log").stat().st_size == 6


def test_lima_real_run_replays_logs_with_configured_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SANDBOX_MAX_LOG_BYTES", "5")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_VERSION", "0.0-test")

    run_dir = tmp_path / "lima-run"
    run_dir.mkdir()
    workspace = run_dir / "workspace"

    monkeypatch.setattr(lima_module.tempfile, "mkdtemp", lambda prefix: str(run_dir))

    def _fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        if args[:2] == ["limactl", "shell"]:
            (workspace / "run.log").write_bytes(b"abcdef")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lima_module.subprocess, "run", _fake_run)

    status = LimaRunner()._run_real(
        "run-lima-log-cap",
        RunSpec(
            session_id=None,
            runtime=RuntimeType.lima,
            base_image="ubuntu-24.04",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
            timeout_sec=5,
        ),
    )

    assert status.phase == RunPhase.completed
    assert status.resource_usage["log_bytes"] == 5
    assert status.resource_usage["log_limit_bytes"] == 5
    assert status.resource_usage["log_truncated"] == 1
