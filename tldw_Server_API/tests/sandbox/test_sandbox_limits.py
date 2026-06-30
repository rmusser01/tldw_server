from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tldw_Server_API.app.core.Sandbox.limits import cap_output_streams, collect_limited_artifacts


def test_cap_output_streams_preserves_stderr_when_both_streams_are_large() -> None:
    result = cap_output_streams(b"o" * 100, b"e" * 100, max_output_bytes=10)

    assert len(result.stdout) + len(result.stderr) <= 10
    assert result.stdout
    assert result.stderr
    assert result.counters["stdout_truncated"] == 1
    assert result.counters["stderr_truncated"] == 1
    assert result.counters["stdout_bytes_original"] == 100
    assert result.counters["stderr_bytes_original"] == 100


def test_cap_output_streams_reuses_unused_stream_budget() -> None:
    result = cap_output_streams(b"oo", b"e" * 100, max_output_bytes=10)

    assert result.stdout == b"oo"
    assert result.stderr == b"e" * 8
    assert len(result.stdout) + len(result.stderr) == 10


def test_collect_limited_artifacts_skips_file_and_total_limit_excesses(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a-small.txt").write_bytes(b"1234")
    (workspace / "b-too-large.txt").write_bytes(b"123456")
    (workspace / "c-would-total.txt").write_bytes(b"12345")
    (workspace / "ignored.log").write_bytes(b"ignored")
    (workspace / "d-link.txt").symlink_to(workspace / "a-small.txt")

    result = collect_limited_artifacts(
        workspace,
        ["*.txt"],
        max_file_bytes=5,
        max_total_bytes=8,
    )

    assert result.artifacts == {"a-small.txt": b"1234"}
    assert result.counters["artifact_files_collected"] == 1
    assert result.counters["artifact_files_skipped"] >= 2
    assert result.counters["artifact_skip_file_limit"] == 1
    assert result.counters["artifact_skip_total_limit"] == 1
    assert result.counters["artifact_bytes_collected"] == 4


def test_collect_limited_artifacts_bounds_read_after_stale_stat(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = workspace / "growing.txt"
    artifact.write_bytes(b"123456")
    original_stat = Path.stat

    def _stale_stat(self: Path, *args, **kwargs):
        if self == artifact and kwargs.get("follow_symlinks", True):
            return SimpleNamespace(st_size=4)
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _stale_stat)

    result = collect_limited_artifacts(
        workspace,
        ["*.txt"],
        max_file_bytes=5,
        max_total_bytes=10,
    )

    assert result.artifacts == {}
    assert result.counters["artifact_files_collected"] == 0
    assert result.counters["artifact_files_skipped"] == 1
    assert result.counters["artifact_skip_file_limit"] == 1
    assert result.counters["artifact_bytes_collected"] == 0


def test_collect_limited_artifacts_keeps_matches_inside_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    nested = workspace / "nested"
    nested.mkdir()
    (nested / "inside.txt").write_bytes(b"ok")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    (nested / "outside-link.txt").symlink_to(outside)

    result = collect_limited_artifacts(
        workspace,
        ["nested/*.txt"],
        max_file_bytes=10,
        max_total_bytes=10,
    )

    assert result.artifacts == {"nested/inside.txt": b"ok"}
    assert result.counters["artifact_files_collected"] == 1
    assert result.counters["artifact_files_skipped"] == 1
    assert result.counters["artifact_skip_symlink"] == 1


def test_collect_limited_artifacts_rejects_symlink_workspace_root(tmp_path) -> None:
    real_workspace = tmp_path / "real-workspace"
    real_workspace.mkdir()
    (real_workspace / "artifact.txt").write_bytes(b"secret")
    workspace_link = tmp_path / "workspace-link"
    workspace_link.symlink_to(real_workspace)

    result = collect_limited_artifacts(
        workspace_link,
        ["*.txt"],
        max_file_bytes=10,
        max_total_bytes=10,
    )

    assert result.artifacts == {}
    assert result.counters["artifact_files_collected"] == 0
    assert result.counters["artifact_files_skipped"] == 1
    assert result.counters["artifact_skip_symlink"] == 1
