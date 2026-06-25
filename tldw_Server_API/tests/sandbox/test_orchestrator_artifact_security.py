from __future__ import annotations

import os
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Sandbox.orchestrator import SandboxOrchestrator


class _FakeArtifactStore:
    def __init__(self, owner: str = "artifact-user") -> None:
        self.owner = owner
        self.adjustments: list[int] = []

    def get_run_owner(self, run_id: str) -> str:
        del run_id
        return self.owner

    def try_reserve_user_artifact_bytes(self, owner: str, size: int, cap_user: int) -> bool:
        del owner, size, cap_user
        return True

    def increment_user_artifact_bytes(self, owner: str, delta: int) -> None:
        del owner
        self.adjustments.append(delta)


def _orchestrator_with_artifact_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SandboxOrchestrator:
    monkeypatch.setenv("SANDBOX_SHARED_ARTIFACTS_DIR", str(tmp_path / "artifact-root"))
    orch = SandboxOrchestrator()
    orch._store = _FakeArtifactStore()  # type: ignore[assignment]
    return orch


@pytest.mark.unit
def test_store_artifacts_rejects_symlink_parent_escape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    orch = _orchestrator_with_artifact_root(monkeypatch, tmp_path)
    run_id = "run-artifact-link-parent"
    art_dir = orch._artifact_dir("artifact-user", run_id)
    art_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-artifacts"
    outside.mkdir()
    try:
        os.symlink(str(outside), str(art_dir / "escape"))
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    orch.store_artifacts(run_id, {"escape/leaked.txt": b"secret"})

    assert not (outside / "leaked.txt").exists()
    assert orch.get_artifact(run_id, "escape/leaked.txt") is None


@pytest.mark.unit
def test_store_artifacts_rejects_symlink_artifact_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    orch = _orchestrator_with_artifact_root(monkeypatch, tmp_path)
    run_id = "run-artifact-root-link"
    art_dir = orch._artifact_dir("artifact-user", run_id)
    art_dir.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-artifact-root"
    outside.mkdir()
    try:
        os.symlink(str(outside), str(art_dir))
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    orch.store_artifacts(run_id, {"leaked.txt": b"secret"})

    assert not (outside / "leaked.txt").exists()
    assert orch.get_artifact(run_id, "leaked.txt") is None


@pytest.mark.unit
def test_get_artifact_path_rejects_symlink_file_escape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    orch = _orchestrator_with_artifact_root(monkeypatch, tmp_path)
    run_id = "run-artifact-link-file"
    art_dir = orch._artifact_dir("artifact-user", run_id)
    art_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-secret.txt"
    outside.write_bytes(b"outside")
    try:
        os.symlink(str(outside), str(art_dir / "leak.txt"))
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    assert orch.get_artifact(run_id, "leak.txt") is None
    assert orch.get_artifact_path(run_id, "leak.txt") is None
