"""Regression tests for Sandbox artifact path confinement."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest

import tldw_Server_API.app.core.Sandbox.orchestrator as orchestrator_module
from tldw_Server_API.app.core.Sandbox.orchestrator import SandboxOrchestrator


class _FakeArtifactStore:
    """Minimal artifact accounting store for orchestrator artifact tests."""

    def __init__(self, owner: str = "artifact-user") -> None:
        self.owner = owner
        self.adjustments: list[int] = []

    def get_run_owner(self, run_id: str) -> str:
        """Return the configured fake owner for any run."""
        del run_id
        return self.owner

    def try_reserve_user_artifact_bytes(self, owner: str, size: int, cap_user: int) -> bool:
        """Accept every artifact reservation in tests."""
        del owner, size, cap_user
        return True

    def increment_user_artifact_bytes(self, owner: str, delta: int) -> None:
        """Record artifact-byte adjustments made during rollback."""
        del owner
        self.adjustments.append(delta)


def _orchestrator_with_artifact_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SandboxOrchestrator:
    """Return an orchestrator using a temp artifact root and fake store."""
    monkeypatch.setenv("SANDBOX_SHARED_ARTIFACTS_DIR", str(tmp_path / "artifact-root"))
    orch = SandboxOrchestrator()
    orch._store = _FakeArtifactStore()  # type: ignore[assignment]
    return orch


@pytest.mark.unit
def test_store_artifacts_rejects_symlink_parent_escape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Writing through a symlinked artifact parent must not reach outside storage."""
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
    """A symlinked artifact root must be rejected before any artifact write."""
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
def test_store_artifacts_rejects_symlink_run_ancestor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A symlinked run ancestor must not become the trusted artifact root."""
    orch = _orchestrator_with_artifact_root(monkeypatch, tmp_path)
    run_id = "run-artifact-ancestor-link"
    art_dir = orch._artifact_dir("artifact-user", run_id)
    art_dir.parent.parent.mkdir(parents=True, exist_ok=True)
    outside_run = tmp_path / "outside-run-ancestor"
    outside_run.mkdir()
    symlink_run = art_dir.parent
    try:
        os.symlink(str(outside_run), str(symlink_run))
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    orch.store_artifacts(run_id, {"leaked.txt": b"secret"})

    assert not (outside_run / "artifacts" / "leaked.txt").exists()
    assert orch.get_artifact(run_id, "leaked.txt") is None
    assert orch.get_artifact_path(run_id, "leaked.txt") is None


@pytest.mark.unit
def test_get_artifact_path_rejects_symlink_file_escape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Artifact reads and path resolution must reject symlinked files."""
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


@pytest.mark.unit
def test_store_artifacts_resists_parent_swap_during_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Replacing a checked parent with a symlink during open must not escape."""
    orch = _orchestrator_with_artifact_root(monkeypatch, tmp_path)
    run_id = "run-artifact-parent-race"
    art_dir = orch._artifact_dir("artifact-user", run_id)
    race_dir = art_dir / "race"
    race_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-race-target"
    outside.mkdir()
    original_open = orchestrator_module.os.open
    swapped = False

    probe = tmp_path / "symlink-probe"
    try:
        os.symlink(str(outside), str(probe))
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    finally:
        with contextlib.suppress(OSError):
            probe.unlink()

    def _racing_open(path: str | bytes | os.PathLike[str], flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        nonlocal swapped
        path_text = os.fspath(path)
        should_swap = (
            not swapped
            and (
                path_text.endswith(f"{os.sep}race{os.sep}leaked.txt")
                or (path_text == "leaked.txt" and dir_fd is not None)
            )
        )
        if should_swap:
            race_dir.rmdir()
            os.symlink(str(outside), str(race_dir))
            swapped = True
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(orchestrator_module.os, "open", _racing_open)

    orch.store_artifacts(run_id, {"race/leaked.txt": b"secret"})

    assert swapped
    assert not (outside / "leaked.txt").exists()
