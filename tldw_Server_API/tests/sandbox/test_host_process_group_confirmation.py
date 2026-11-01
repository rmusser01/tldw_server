from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType
from tldw_Server_API.app.core.Sandbox.runners import seatbelt_runner, worktree_runner
from tldw_Server_API.app.core.Sandbox.runners.seatbelt_runner import SeatbeltRunner
from tldw_Server_API.app.core.Sandbox.runners.worktree_runner import WorktreeRunner


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
@pytest.mark.parametrize("runner", [SeatbeltRunner, WorktreeRunner])
def test_cancel_kills_descendant_after_group_leader_exits(
    runner: type[SeatbeltRunner] | type[WorktreeRunner],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    marker = tmp_path / "child-ready"
    child_script = (
        "import os,signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    )
    leader_script = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", leader_script, child_script, str(marker)],
        start_new_session=True,
    )
    monkeypatch.setattr(runner, "_cancel_grace_seconds", classmethod(lambda cls: 1))
    run_id = f"group-descendant-{runner.__name__}"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner._register_active_run(run_id, process, str(run_dir))
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), "child did not install its TERM handler"
        child_pid = int(marker.read_text())

        confirmed = runner.cancel_run(run_id)
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            assert confirmed is True
            assert run_id not in runner._active_proc
            assert not run_dir.exists()
        else:
            # Linux init may leave an orphan zombie visible to killpg(0).
            stat = Path(f"/proc/{child_pid}/stat").read_text()
            assert stat.rsplit(")", 1)[1].split()[0] == "Z"
            assert confirmed is False
            assert run_id in runner._active_proc
            assert run_dir.exists()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        runner._clear_active_run(run_id)
        runner._consume_cancelled(run_id)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
@pytest.mark.parametrize("runner", [SeatbeltRunner, WorktreeRunner])
def test_unconfirmed_cancel_keeps_owner_and_does_not_mark_cancelled(
    runner: type[SeatbeltRunner] | type[WorktreeRunner],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    run_id = f"group-uncertain-{runner.__name__}"
    runner._register_active_run(run_id, process, str(tmp_path))
    monkeypatch.setattr(runner, "_cancel_grace_seconds", classmethod(lambda cls: 0))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "killpg", lambda *args: (_ for _ in ()).throw(PermissionError("denied")))
            assert runner.cancel_run(run_id) is False

        assert runner._consume_cancelled(run_id) is False
        assert runner._active_proc[run_id] is process
        assert tmp_path.exists()
    finally:
        process.kill()
        process.wait(timeout=5)
        runner._clear_active_run(run_id)
        runner._consume_cancelled(run_id)


@pytest.mark.parametrize("runtime", [RuntimeType.seatbelt, RuntimeType.worktree])
def test_live_group_cannot_finish_or_remove_workspace(
    runtime: RuntimeType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = seatbelt_runner if runtime == RuntimeType.seatbelt else worktree_runner
    runner = SeatbeltRunner if runtime == RuntimeType.seatbelt else WorktreeRunner
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    process = SimpleNamespace(pid=2**30, returncode=0, wait=lambda **kwargs: 0)
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **kwargs: str(run_dir))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runner, "_terminate_process_group", classmethod(lambda cls, proc: False))
    monkeypatch.setattr(os, "killpg", lambda *args: None)
    if runtime == RuntimeType.worktree:
        monkeypatch.setattr(WorktreeRunner, "_is_git_repo", staticmethod(lambda path: True))
        monkeypatch.setattr(WorktreeRunner, "create_worktree", staticmethod(lambda *args, **kwargs: str(worktree)))
        monkeypatch.setattr(WorktreeRunner, "destroy_worktree", staticmethod(lambda *args: shutil.rmtree(worktree)))
    backend_ids: list[str] = []
    spec = RunSpec(None, runtime, None, ["/bin/true"], on_backend_started=backend_ids.append)
    run_id = f"group-return-uncertain-{runtime.value}"
    runner._mark_cancelled(run_id)
    try:
        if runtime == RuntimeType.seatbelt:
            status = SeatbeltRunner()._run_real(run_id, spec)
        else:
            status = WorktreeRunner(allowed_repo_dirs=[str(tmp_path)]).start_run(run_id, spec, str(repo))

        assert status.phase == RunPhase.running
        assert status.finished_at is None
        assert status.message == "backend_termination_unconfirmed"
        assert runner._active_proc[run_id] is process
        assert run_dir.exists()
        assert worktree.exists()
        assert backend_ids == [str(2**30)]
        monkeypatch.setattr(runner, "_terminate_process_group", classmethod(lambda cls, proc: True))
        assert runner.cancel_run(run_id) is True
        assert not run_dir.exists()
        if runtime == RuntimeType.worktree:
            assert not worktree.exists()
    finally:
        runner._clear_active_run(run_id)
        runner._consume_cancelled(run_id)
