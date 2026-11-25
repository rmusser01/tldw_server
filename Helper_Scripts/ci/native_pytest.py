"""Run unchanged native pytest arguments with nonterminating thread diagnostics."""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest


class _PhaseDiagnostics:
    def __init__(self, output_fd: int) -> None:
        self._output_fd = output_fd

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Retain completed slow phases without reading test identities or values."""
        if report.when not in {"setup", "call", "teardown"} or report.duration < 1.0:
            return
        line = f"[native-pytest] phase-complete phase={report.when} seconds={report.duration:.3f}\n"
        try:
            os.write(self._output_fd, line.encode("ascii"))
        except OSError:
            pass  # A failed diagnostic sink must not replace pytest's result.


def _dump_thread_stacks(output_fd: int) -> None:
    """Emit frame locations without source lines, locals or exception values."""
    frames = sys._current_frames()
    try:
        lines = ["[native-pytest] thread-dump"]
        for thread_id, frame in frames.items():
            lines.append(f"Thread {thread_id} (most recent call first):")
            for _ in range(100):
                if frame is None:
                    break
                code = frame.f_code
                lines.append(f'  File "{code.co_filename}", line {frame.f_lineno}, in {code.co_name}')
                frame = frame.f_back
        os.write(output_fd, ("\n".join(lines) + "\n").encode("utf-8", errors="backslashreplace"))
    finally:
        frames.clear()


def _watch_threads(output_fd: int, interval: float) -> None:
    """Observe hangs without changing any timeout, cancellation or exit policy."""
    while True:
        time.sleep(interval)
        _dump_thread_stacks(output_fd)


def main(args: list[str] | None = None, *, traceback_interval: float = 120.0) -> int:
    """Preserve pytest's result while observing test, cleanup and interpreter hangs."""
    # Duplicate before pytest redirects stderr; retain diagnostics during captured tests.
    output_fd = os.dup(sys.stderr.fileno())
    threading.Thread(
        target=_watch_threads, args=(output_fd, traceback_interval),
        name="native-pytest-diagnostics", daemon=True,
    ).start()
    print("[native-pytest] pytest-start", flush=True)
    result = int(pytest.main(args, plugins=[_PhaseDiagnostics(output_fd)]))
    print(f"[native-pytest] pytest-return exit={result}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
