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
        )
    except BaseException:
        _admission.release()
        raise
    _pending_controller = process
    captured = bytearray()
    lock = threading.Lock()
    finished = {"reader": False, "caller": False}
    read_failed = threading.Event()

    def finish(owner: str) -> None:
        global _pending_controller
        with lock:
            finished[owner] = True
            if all(finished.values()):
                _pending_controller = None
                _admission.release()

    def read() -> None:
        try:
            while len(captured) <= 200000:
                part = process.stdout.read(min(65536, 200001 - len(captured)))
                if not part:
                    break
                captured.extend(part)
        except (OSError, ValueError):
            read_failed.set()
            process.stdin.close()
        finally:
            process.stdout.close()
            # This one bounded reader remains the owner even if the invoker's
            # wait gives up. Never kill a controller with an unconfirmed tree.
            process.wait()
            finish("reader")

    reader = threading.Thread(target=read, name="probe-controller-result", daemon=True)
    try:
        try:
            reader.start()
        except (RuntimeError, OSError):
            process.stdin.close()
            process.stdout.close()
            try:
                process.wait(timeout=2)
                finish("reader")
            except subprocess.TimeoutExpired:
                # No more controllers can be admitted; ownership is retained.
                pass
            raise RuntimeError("probe controller reader unavailable") from None
        try:
            process.wait(timeout=timeout_s + 2)
        except subprocess.TimeoutExpired:
            raise RuntimeError("probe containment unconfirmed") from None
        reader.join(0.1)
        if reader.is_alive() or read_failed.is_set() or process.returncode != 0 or len(captured) > 200000:
            raise RuntimeError("probe containment controller failed")
        result = json.loads(captured)
        if result["containment"] is not True:
            raise RuntimeError("probe containment unavailable")
        return subprocess.CompletedProcess(
            list(argv),
            result["returncode"],
            base64.b64decode(result["stdout"]),
            base64.b64decode(result["stderr"]),
        )
    finally:
        process.stdin.close()
        finish("caller")
