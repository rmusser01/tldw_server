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
    import threading

    from tldw_Server_API.tests.helpers import moderation_process_harness as harness

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
            harness.run_isolated_probe([sys.executable], timeout_s=0.01)
        assert not controller.killed
        with pytest.raises(RuntimeError, match="capacity"):
            harness.run_isolated_probe([sys.executable], timeout_s=0.01)
    finally:
        released.set()
        time.sleep(0.05)
