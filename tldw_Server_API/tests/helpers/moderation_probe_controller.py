"""Independent stdlib-only process-tree owner. Never imports pytest or the app."""

from __future__ import annotations

import base64
import errno
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


class PosixTree:
    """A session/process group remains owned after its leader hard-exits."""

    def __init__(self, argv: list[str]):
        if sys.platform.startswith("linux"):
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            # Adopt/reap orphan grandchildren as well as killing their group.
            if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
                raise RuntimeError("subreaper unavailable")
        self.process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
            shell=False,
            bufsize=0,
        )
        self.pgid = self.process.pid

    def cleanup(self, deadline: float) -> bool:
        signalled = False
        while time.monotonic() < deadline:
            self.process.poll()
            if not signalled:
                try:
                    os.killpg(self.pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError as exc:
                    if exc.errno != errno.EPERM:
                        raise
                    # A group in native exit can briefly deny signalling.
                    # Retry within the deadline; only absence confirms cleanup.
                else:
                    signalled = True
            if self.process.returncode is not None and sys.platform.startswith("linux"):
                try:
                    while os.waitpid(-self.pgid, os.WNOHANG)[0]:
                        pass
                except ChildProcessError:
                    pass
            try:
                os.killpg(self.pgid, 0)
            except ProcessLookupError:
                return self.process.poll() is not None
            except PermissionError as exc:
                if exc.errno != errno.EPERM:
                    raise
                # A just-killed group can be uninspectable; only absence confirms cleanup.
            time.sleep(0.01)
        return False


def _settle_tree(tree, readers, deadline: float) -> bool:
    """Never let a cleanup exception discard the only native tree owner."""
    try:
        if tree.cleanup(deadline) is not True:
            return False
        for reader in readers:
            if reader.is_alive():
                reader.join(max(0, deadline - time.monotonic()))
        return all(not reader.is_alive() for reader in readers)
    except BaseException:  # noqa: BLE001 - retain the live owner and retry; never disclose cleanup errors.
        return False


def _emit_result(contained: bool, code: int, buffers: list[bytearray]) -> None:
    try:
        frame = (
            json.dumps(
                {
                    "containment": contained,
                    "returncode": code,
                    "stdout": base64.b64encode(buffers[0]).decode("ascii") if contained else "",
                    "stderr": base64.b64encode(buffers[1]).decode("ascii") if contained else "",
                }
            ).encode("ascii")
            + b"\n"
        )
        view = memoryview(frame)
        while view:
            written = os.write(sys.stdout.fileno(), view[:65536])
            if not written:
                return
            view = view[written:]
    except (OSError, ValueError):
        # The invoker may have exited; cleanup ownership is independent of it.
        pass


def main() -> int:
    timeout = min(10.0, float(sys.argv[1]))
    deadline = time.monotonic() + timeout
    lost, overflow = threading.Event(), threading.Event()

    def control() -> None:
        # No buffered stdio lock may remain held at interpreter shutdown.
        os.read(sys.stdin.fileno(), 1)
        lost.set()

    threading.Thread(target=control, name="invoker-control", daemon=True).start()
    tree = None
    buffers = [bytearray(), bytearray()]
    readers = []
    contained = False
    code = 126
    try:
        if os.name == "nt":
            # Load only this sibling stdlib helper under isolated mode.
            import runpy

            namespace = runpy.run_path(str(Path(__file__).with_name("moderation_windows_job.py")))
            tree = namespace["WindowsTree"](sys.argv[2:])
        elif os.name == "posix":
            tree = PosixTree(sys.argv[2:])
        else:
            return 126

        def read(stream, buffer) -> None:
            try:
                while True:
                    part = stream.read(min(16384, 65537 - len(buffer)))
                    if not part:
                        return
                    buffer.extend(part)
                    if len(buffer) > 65536:
                        del buffer[65536:]
                        overflow.set()
                        return
            finally:
                stream.close()

        for stream, buffer in zip((tree.process.stdout, tree.process.stderr), buffers, strict=True):
            thread = threading.Thread(target=read, args=(stream, buffer), daemon=True)
            readers.append(thread)
            thread.start()
        while True:
            if lost.is_set() or overflow.is_set():
                code = 125
                break
            if tree.process.poll() is not None:
                code = tree.process.returncode
                break
            if time.monotonic() >= deadline:
                code = 124
                break
            lost.wait(0.01)
    except (OSError, RuntimeError, ValueError, ImportError):
        # Fixed diagnostics only, even for launch/setup failure.
        code = 126
    finally:
        if tree is not None:
            contained = _settle_tree(tree, readers, time.monotonic() + 2)
            if not contained:
                # Report failure once, but remain an actual independent owner.
                # One quarantined loop, no replacement process/helper per retry.
                code = 126
                _emit_result(False, code, buffers)
                while not contained:
                    time.sleep(0.05)
                    contained = _settle_tree(tree, readers, time.monotonic() + 0.1)
            if overflow.is_set():
                code = 125
        else:
            # Constructors must clean partial launches before raising. No tree
            # was published and setup failed; signal that distinct process exit.
            contained = True
        # Send final confirmation even after control EOF: a surviving result
        # reader needs it to retire quarantine. A gone invoker gets no output.
        _emit_result(contained, code, buffers)
    return 0 if tree is not None else 126


if __name__ == "__main__":
    raise SystemExit(main())
