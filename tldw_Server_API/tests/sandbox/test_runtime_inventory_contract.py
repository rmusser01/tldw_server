from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sandbox.models import RuntimeType
from tldw_Server_API.app.core.Sandbox.service import SandboxService


def test_feature_discovery_covers_all_core_runtime_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")

    discovery = SandboxService().feature_discovery()

    discovered = {str(item.get("name")) for item in discovery}
    assert discovered == {runtime.value for runtime in RuntimeType}


def test_worktree_discovery_reports_host_local_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    monkeypatch.setenv("TLDW_SANDBOX_WORKTREE_AVAILABLE", "1")

    discovery = SandboxService().feature_discovery()
    worktree = next(item for item in discovery if item["name"] == "worktree")

    assert worktree["supported_trust_levels"] == ["trusted", "standard"]
    assert worktree["strict_deny_all_supported"] is False
    assert worktree["strict_allowlist_supported"] is False
    assert worktree["egress_allowlist_supported"] is False
    assert worktree["interactive_supported"] is False
    assert "not VM-grade" in str(worktree["notes"])
