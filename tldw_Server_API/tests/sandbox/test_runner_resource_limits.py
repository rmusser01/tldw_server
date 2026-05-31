from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.Sandbox.policy import SandboxPolicyConfig
from tldw_Server_API.app.core.Sandbox.runners import resource_limits as resource_limits_module
from tldw_Server_API.app.core.Sandbox.runners.resource_limits import collect_runner_artifacts


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
