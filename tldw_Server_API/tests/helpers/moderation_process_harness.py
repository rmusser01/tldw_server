"""Stdlib-only caller for a separately owned, deadline-enforcing controller."""

from __future__ import annotations

import base64
import json
import math
import subprocess
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

_admission = threading.BoundedSemaphore(1)
_pending_controller: subprocess.Popen | None = None


def run_isolated_probe(
    argv: Sequence[str],
    *,
    timeout_s: float = 10.0,
) -> subprocess.CompletedProcess[bytes]:
    """Run a probe only inside independently established process-tree containment.

    The invoker alone owns the controller's stdin writer. EOF on invoker loss
    triggers cleanup even if the probe hard-exits or inherits open output pipes.
    No timeout supplied by pytest is part of this containment guarantee.
    """
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 10:
        raise ValueError("invalid probe deadline")
    if not argv or any(type(arg) is not str for arg in argv) or sum(map(len, argv)) > 65536:
        raise ValueError("invalid probe argv")
    global _pending_controller
    if not _admission.acquire(blocking=False):
        raise RuntimeError("probe harness capacity_exhausted")
    try:
        controller = Path(__file__).with_name("moderation_probe_controller.py").resolve()
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", str(controller), str(timeout_s), *argv],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=sys.platform != "win32",
            shell=False,
            bufsize=0,
        )
    except BaseException:
        _admission.release()
        raise
    _pending_controller = process
    captured = bytearray()
    lock = threading.Lock()
    finished = {"reader": False, "caller": False}
    read_failed = threading.Event()
    available = threading.Event()
    refused = threading.Event()
    confirmed = threading.Event()
    final_result = []

    def finish(owner: str) -> None:
        global _pending_controller
        with lock:
            finished[owner] = True
            if all(finished.values()):
                _pending_controller = None
                _admission.release()

    def read() -> None:
        pending = bytearray()

        def status(line: bytes) -> None:
            try:
                result = json.loads(line)
                if (
                    type(result) is not dict
                    or set(result) != {"containment", "returncode", "stdout", "stderr"}
                    or type(result["containment"]) is not bool
                    or type(result["returncode"]) is not int
                    or type(result["stdout"]) is not str
                    or type(result["stderr"]) is not str
                    or final_result
                ):
                    raise ValueError
                if result["containment"]:
                    stdout = base64.b64decode(result["stdout"], validate=True)
                    stderr = base64.b64decode(result["stderr"], validate=True)
                    if max(len(stdout), len(stderr)) > 65536:
                        raise ValueError
                    final_result.append(subprocess.CompletedProcess(list(argv), result["returncode"], stdout, stderr))
                    confirmed.set()
                else:
                    if refused.is_set() or result["stdout"] or result["stderr"]:
                        raise ValueError
                    refused.set()
            except (ValueError, TypeError, UnicodeError):
                read_failed.set()
                confirmed.clear()
            available.set()

        try:
            while len(captured) <= 200000:
                part = process.stdout.read(min(16384, 200001 - len(captured)))
                if not part:
                    break
                captured.extend(part)
                pending.extend(part)
                while b"\n" in pending:
                    line, _, remainder = pending.partition(b"\n")
                    pending[:] = remainder
                    status(bytes(line))
            if pending or len(captured) > 200000:
                read_failed.set()
                confirmed.clear()
                available.set()
        except (OSError, ValueError):
            read_failed.set()
            confirmed.clear()
            available.set()
            process.stdin.close()
        finally:
            process.stdout.close()
            # This one bounded reader remains the owner even if the invoker's
            # wait gives up. Never kill a controller with an unconfirmed tree.
            process.wait()
            # A native exit is NOT evidence that descendants were cleaned.
            # Our controller stays alive/reaps after false or raising cleanup;
            # a broken controller without a final ACK permanently quarantines.
            if confirmed.is_set() and not read_failed.is_set():
                finish("reader")
            available.set()

    reader = threading.Thread(target=read, name="probe-controller-result", daemon=True)
    try:
        try:
            reader.start()
        except (RuntimeError, OSError):
            process.stdin.close()
            process.stdout.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # No more controllers can be admitted; ownership is retained.
                pass
            raise RuntimeError("probe controller reader unavailable") from None
        if not available.wait(timeout_s + 2):
            raise RuntimeError("probe containment unconfirmed") from None
        if refused.is_set():
            raise RuntimeError("probe containment cleanup unconfirmed")
        if not confirmed.is_set():
            raise RuntimeError("probe containment unconfirmed")
        reader.join(0.1)
        if reader.is_alive() or read_failed.is_set() or process.returncode != 0 or len(captured) > 200000:
            raise RuntimeError("probe containment controller failed")
        return final_result[0]
    finally:
        process.stdin.close()
        finish("caller")
