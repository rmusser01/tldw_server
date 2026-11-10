from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Sandbox.policy import SandboxPolicyConfig
from tldw_Server_API.app.core.Sandbox.runners import resource_limits as resource_limits_module
from tldw_Server_API.app.core.Sandbox.runners.resource_limits import collect_runner_artifacts


def test_process_logs_drain_both_streams_in_bounded_chunks() -> None:
    """A silent stdout cannot block a large newline-free stderr pipe."""
    count = 3 * 1024 * 1024
    process = subprocess.Popen(
        [sys.executable, "-c", f"import os; os.write(2, b'e' * {count}); os.write(1, b'o' * {count})"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    sizes: dict[str, list[int]] = {"stdout": [], "stderr": []}
    try:
        total = resource_limits_module.drain_process_output(
            process,
            lambda data: sizes["stdout"].append(len(data)),
            lambda data: sizes["stderr"].append(len(data)),
        )
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()

    assert total == 2 * count
    assert sum(sizes["stdout"]) == count
    assert sum(sizes["stderr"]) == count
    assert max(sizes["stdout"] + sizes["stderr"]) <= 64 * 1024


def test_process_logs_publish_short_output_before_child_exit() -> None:
    """An interactive child must receive its log-driven reply before EOF."""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, select, sys; os.write(1, b'ready'); "
            "sys.exit(0 if select.select([0], [], [], 2)[0] else 2)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def reply(data: bytes) -> None:
        assert data == b"ready"
        assert process.stdin is not None
        try:
            process.stdin.write(b"x")
            process.stdin.flush()
        except BrokenPipeError:
            pass

    try:
        resource_limits_module.drain_process_output(process, reply, lambda data: None)
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        if process.stdin is not None:
            process.stdin.close()


def test_collect_runner_artifacts_excludes_internal_files_before_quota(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Internal runner exclusions must not consume user-visible artifact quota."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".internal").write_bytes(b"aaaa")
    (workspace / "visible.txt").write_bytes(b"bbbb")

    def _policy_from_settings(cls: type[SandboxPolicyConfig]) -> SandboxPolicyConfig:
        return cls(max_artifact_file_bytes=100, max_artifact_total_bytes=4)

    monkeypatch.setattr(SandboxPolicyConfig, "from_settings", classmethod(_policy_from_settings))

    result = collect_runner_artifacts(
        str(workspace),
        ["*"],
        exclude_hidden=True,
    )

    assert result.artifacts == {"visible.txt": b"bbbb"}
    assert result.counters["artifact_files_collected"] == 1
    assert result.counters["artifact_files_excluded"] == 1
    assert result.counters["artifact_files_skipped"] == 0


def test_collect_runner_artifacts_preserves_counter_schema_on_collection_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Collection failures should still report the shared artifact counter contract."""

    def _policy_from_settings(cls: type[SandboxPolicyConfig]) -> SandboxPolicyConfig:
        return cls(max_artifact_file_bytes=5, max_artifact_total_bytes=8)

    def _raise_collection_error(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise OSError("cannot read workspace")

    monkeypatch.setattr(SandboxPolicyConfig, "from_settings", classmethod(_policy_from_settings))
    monkeypatch.setattr(resource_limits_module, "collect_limited_artifacts", _raise_collection_error)

    result = collect_runner_artifacts(str(tmp_path), ["*"])

    assert result.artifacts == {}
    assert result.counters["artifact_limit_file_bytes"] == 5
    assert result.counters["artifact_limit_total_bytes"] == 8
    assert result.counters["artifact_files_collected"] == 0
    assert result.counters["artifact_bytes_collected"] == 0
    assert result.counters["artifact_files_skipped"] == 1
    assert result.counters["artifact_skip_read_error"] == 1
