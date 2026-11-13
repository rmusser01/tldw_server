"""Containment must pass independently before any hostile regex is executed."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
        if not handle:
            if ctypes.get_last_error() == 87:  # ERROR_INVALID_PARAMETER: no PID
                return False
            raise RuntimeError("cannot inspect descendant liveness")
        try:
            result = kernel.WaitForSingleObject(handle, 0)
            if result not in (0, 258):
                raise RuntimeError("cannot inspect descendant liveness")
            return result == 258  # WAIT_TIMEOUT
        finally:
            kernel.CloseHandle(handle)
    if sys.platform.startswith("linux"):
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split(")", 1)[1].split()[0] == "Z":
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _assert_dead(pid_file):
    pids = [int(value) for value in pid_file.read_text().split()]
    end = time.monotonic() + 2
    while any(_alive(pid) for pid in pids) and time.monotonic() < end:
        time.sleep(0.02)
    assert not any(_alive(pid) for pid in pids), pids


def _probe(pid_file, hard_exit=False):
    return [
        sys.executable,
        "-I",
        "-S",
        "-c",
        (
            "import os,subprocess,sys,time;"
            "child=subprocess.Popen([sys.executable,'-I','-S','-c','import time;time.sleep(60)'],close_fds=True);"
            f"open({str(pid_file)!r},'w').write(str(os.getpid())+' '+str(child.pid));"
            + ("os._exit(7)" if hard_exit else "time.sleep(60)")
        ),
    ]


def test_outer_harness_timeout_kills_probe_and_grandchild(tmp_path):
    from tldw_Server_API.tests.helpers.moderation_process_harness import run_isolated_probe

    record = tmp_path / "pids"
    start = time.monotonic()
    result = run_isolated_probe(_probe(record), timeout_s=0.5)
    assert result.returncode == 124
    assert time.monotonic() - start < 3.5
    _assert_dead(record)


def test_probe_hard_exit_and_invoker_disconnect_still_clean_tree(tmp_path):
    from tldw_Server_API.tests.helpers.moderation_process_harness import run_isolated_probe

    record = tmp_path / "exit-pids"
    result = run_isolated_probe(_probe(record, hard_exit=True))
    assert result.returncode == 7
    _assert_dead(record)

    record = tmp_path / "disconnect-pids"
    helper = Path(__file__).resolve().parents[1] / "helpers" / "moderation_process_harness.py"
    code = (
        "import runpy,threading,time,os;"
        f"h=runpy.run_path({str(helper)!r});"
        f"threading.Thread(target=lambda:h['run_isolated_probe']({_probe(record)!r})).start();"
        f"p=__import__('pathlib').Path({str(record)!r});"
        "end=time.monotonic()+3;"
        "\nwhile not p.exists() and time.monotonic()<end: time.sleep(.01)\n"
        "os._exit(0)"
    )
    invoker = subprocess.Popen([sys.executable, "-I", "-S", "-c", code], close_fds=True)
    try:
        assert invoker.wait(timeout=5) == 0
        assert record.exists()
        _assert_dead(record)
    finally:
        if invoker.poll() is None:
            invoker.kill()
            invoker.wait(timeout=2)


def test_probe_output_is_bounded_and_nonzero_exit_preserved():
    from tldw_Server_API.tests.helpers.moderation_process_harness import run_isolated_probe

    result = run_isolated_probe([sys.executable, "-I", "-S", "-c", "raise SystemExit(9)"])
    assert result.returncode == 9
    result = run_isolated_probe([sys.executable, "-I", "-S", "-c", "import os;os.write(1,b'x'*100000)"])
    assert result.returncode == 125
    assert len(result.stdout) <= 65536


def test_unconfirmed_controller_is_not_killed_or_replaced(monkeypatch):
    import io
    import runpy
    import threading

    from tldw_Server_API.tests.helpers import moderation_process_harness as harness

    run = runpy.run_path(harness.__file__)["run_isolated_probe"]

    released = threading.Event()

    class Controller:
        def __init__(self):
            self.stdin, self.stdout = io.BytesIO(), io.BytesIO()
            self.returncode = None
            self.killed = False

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("controller", timeout)
            released.wait(3)
            self.returncode = 126
            return 126

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

    controller = Controller()
    monkeypatch.setattr(harness.subprocess, "Popen", lambda *a, **k: controller)
    try:
        with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
            run([sys.executable], timeout_s=0.01)
        assert not controller.killed
        with pytest.raises(RuntimeError, match="capacity"):
            run([sys.executable], timeout_s=0.01)
    finally:
        released.set()
        time.sleep(0.05)


@pytest.mark.parametrize("reply", [b'{"containment":false,"returncode":126,"stdout":"","stderr":""}\n', b""])
def test_dead_unconfirmed_controller_does_not_reopen_admission(monkeypatch, reply):
    import io
    import runpy

    from tldw_Server_API.tests.helpers import moderation_process_harness as harness

    run = runpy.run_path(harness.__file__)["run_isolated_probe"]

    class FailedController:
        def __init__(self):
            self.stdin, self.stdout = io.BytesIO(), io.BytesIO(reply)
            self.returncode = 126

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(harness.subprocess, "Popen", lambda *a, **k: FailedController())
    with pytest.raises(RuntimeError, match="containment"):
        run([sys.executable], timeout_s=0.01)
    with pytest.raises(RuntimeError, match="capacity"):
        run([sys.executable], timeout_s=0.01)


@pytest.mark.parametrize("failure", ["false", "raise"])
def test_failed_cleanup_retains_live_owner_until_native_confirmation(tmp_path, failure):
    from tldw_Server_API.tests.helpers.moderation_cleanup_probe import validate_cleanup_recovery

    if os.name != "posix":
        pytest.skip("POSIX process-group ownership evidence")
    validate_cleanup_recovery(tmp_path, failure=failure)


def test_raising_cleanup_owner_survives_invoker_hard_exit(tmp_path):
    from tldw_Server_API.tests.helpers.moderation_cleanup_probe import validate_cleanup_recovery

    if os.name != "posix":
        pytest.skip("POSIX process-group ownership evidence")
    validate_cleanup_recovery(tmp_path, failure="raise", disconnect=True)


@pytest.fixture
def posix_cleanup_boundary(monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.tests.helpers import moderation_probe_controller as controller

    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, delay):
            self.now += delay

    clock = Clock()
    tree = object.__new__(controller.PosixTree)
    tree.process = SimpleNamespace(returncode=7, poll=lambda: 7)
    tree.pgid = 12345
    monkeypatch.setattr(controller, "time", clock)
    monkeypatch.setattr(controller, "sys", SimpleNamespace(platform="darwin"))
    return controller, tree, clock


def test_signal_zero_transient_eperm_retries_until_native_absence(posix_cleanup_boundary, monkeypatch):
    from types import SimpleNamespace

    controller, tree, clock = posix_cleanup_boundary
    signals = []

    def killpg(pgid, sig):
        assert pgid == 12345
        signals.append(sig)
        if sig == 0:
            if clock.now == 0:
                raise PermissionError(1, "uncertain liveness")
            raise ProcessLookupError(3, "absent")

    monkeypatch.setattr(controller, "os", SimpleNamespace(killpg=killpg))
    assert controller._settle_tree(tree, [], 2.0) is True
    assert 0 < clock.now < 2.0
    assert signals == [controller.signal.SIGKILL, 0, 0]


def test_signal_zero_persistent_eperm_uses_budget_and_retains_tree(posix_cleanup_boundary, monkeypatch):
    from types import SimpleNamespace

    controller, tree, clock = posix_cleanup_boundary
    owner = tree.process
    absent = False

    def killpg(pgid, sig):
        assert pgid == 12345
        if absent:
            raise ProcessLookupError(3, "absent")
        if sig == 0:
            raise PermissionError(1, "uncertain liveness")

    monkeypatch.setattr(controller, "os", SimpleNamespace(killpg=killpg))
    assert controller._settle_tree(tree, [], 2.0) is False
    assert 2.0 <= clock.now < 2.01
    assert tree.process is owner
    absent = True
    assert controller._settle_tree(tree, [], clock.now + 0.1) is True
    assert tree.process is owner


def test_sigkill_transient_eperm_retries_until_group_is_absent(posix_cleanup_boundary, monkeypatch):
    from types import SimpleNamespace

    controller, tree, clock = posix_cleanup_boundary

    signals = []

    def killpg(pgid, sig):
        assert pgid == 12345
        signals.append(sig)
        if clock.now == 0 and sig == controller.signal.SIGKILL:
            raise PermissionError(1, "transient denial")
        if clock.now > 0 and sig == 0:
            raise ProcessLookupError(3, "absent")

    monkeypatch.setattr(controller, "os", SimpleNamespace(killpg=killpg))
    assert tree.cleanup(2.0) is True
    assert signals == [controller.signal.SIGKILL, 0, controller.signal.SIGKILL, 0]


def test_sigkill_persistent_eperm_does_not_release_live_group(posix_cleanup_boundary, monkeypatch):
    from types import SimpleNamespace

    controller, tree, clock = posix_cleanup_boundary

    def killpg(pgid, sig):
        assert pgid == 12345
        if sig == controller.signal.SIGKILL:
            raise PermissionError(1, "persistent denial")

    monkeypatch.setattr(controller, "os", SimpleNamespace(killpg=killpg))
    assert tree.cleanup(0.1) is False
    assert clock.now >= 0.1


def test_signal_zero_other_permission_errors_are_not_retried(posix_cleanup_boundary, monkeypatch):
    from types import SimpleNamespace

    controller, tree, clock = posix_cleanup_boundary

    def killpg(pgid, sig):
        assert pgid == 12345
        if sig == 0:
            raise PermissionError(13, "other denial")

    monkeypatch.setattr(controller, "os", SimpleNamespace(killpg=killpg))
    with pytest.raises(PermissionError) as error:
        tree.cleanup(2.0)
    assert error.value.errno == 13
    assert clock.now == 0


@pytest.mark.parametrize("reader_finishes", [True, False])
def test_native_absence_still_requires_reader_completion(posix_cleanup_boundary, monkeypatch, reader_finishes):
    from types import SimpleNamespace

    controller, tree, clock = posix_cleanup_boundary

    def killpg(pgid, sig):
        assert pgid == 12345
        raise ProcessLookupError(3, "absent")

    class Reader:
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            assert timeout == 2.0
            clock.sleep(0.02 if reader_finishes else timeout)
            self.alive = not reader_finishes

    monkeypatch.setattr(controller, "os", SimpleNamespace(killpg=killpg))
    assert controller._settle_tree(tree, [Reader()], 2.0) is reader_finishes
