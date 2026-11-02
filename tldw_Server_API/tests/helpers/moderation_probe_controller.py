"""Independent stdlib-only process-tree owner. Never imports pytest or the app."""

from __future__ import annotations

import base64
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
        self.process.poll()
        try:
            os.killpg(self.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        while time.monotonic() < deadline:
            self.process.poll()
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
            time.sleep(0.01)
        return False


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
            thread.start()
            readers.append(thread)
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
            end = time.monotonic() + 2
            contained = tree.cleanup(end)
            for reader in readers:
                reader.join(max(0, end - time.monotonic()))
            contained = contained and all(not reader.is_alive() for reader in readers)
            if overflow.is_set():
                code = 125
        if not lost.is_set():
            sys.stdout.buffer.write(
                json.dumps(
                    {
                        "containment": contained,
                        "returncode": code,
                        "stdout": base64.b64encode(buffers[0]).decode("ascii"),
                        "stderr": base64.b64encode(buffers[1]).decode("ascii"),
                    }
                ).encode("ascii")
            )
            sys.stdout.buffer.flush()
    return 0 if contained else 126


if __name__ == "__main__":
    raise SystemExit(main())
