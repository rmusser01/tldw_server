#!/usr/bin/env python3
"""Boot the server in a deployment-shaped (scrubbed) environment and probe it.

The test conftests force-set dozens of env vars at import time, so an in-pytest
run never sees the no-override world a real single-user deployment runs in. This
smoke boots ``tldw_Server_API.app.main:app`` as a subprocess with ONLY a pinned
minimal env (empty otherwise), waits for ``/health`` to return 200, and exits
nonzero if startup or the auth default drifts (the #2590/e88c96500f class).

Run:
    python Helper_Scripts/ci/minimal_env_smoke.py
    python Helper_Scripts/ci/minimal_env_smoke.py --port 8123 --timeout 60
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# The pinned minimal env a single-user deployment actually needs, and nothing
# else — no WORKFLOWS_EGRESS_*, no TEST_MODE, no MINIMAL_TEST_APP convenience.
MINIMAL_ENV = {
    "AUTH_MODE": "single_user",
    # >=16 chars, non-placeholder (settings validation)
    "SINGLE_USER_API_KEY": "smoke-minimal-env-key-0123456789",
    # keep the boot fast and hermetic without leaning on TEST_MODE
    "ULTRA_MINIMAL_APP": "1",
    "AUTO_DOWNLOAD_MODELS": "false",
    "DISABLE_AUTHNZ_SCHEDULER": "1",
    "WORKFLOWS_SCHEDULER_ENABLED": "false",
    "MPLBACKEND": "Agg",
}


def _child_env(repo_root: Path, port: int) -> dict[str, str]:
    """A scrubbed environment: MINIMAL_ENV plus only what the OS needs to run."""
    env: dict[str, str] = {"PYTHONPATH": str(repo_root), "PORT": str(port)}
    for passthrough in ("PATH", "HOME", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR", "VIRTUAL_ENV"):
        value = os.environ.get(passthrough)
        if value:
            env[passthrough] = value
    env.update(MINIMAL_ENV)
    return env


def _probe(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=3) as resp:  # noqa: S310 - localhost only
        return resp.status, resp.read().decode("utf-8", "replace")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8137)
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds to wait for /health")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    health_url = f"http://127.0.0.1:{args.port}/health"

    cmd = [
        sys.executable, "-m", "uvicorn",
        "tldw_Server_API.app.main:app",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--log-level", "warning",
    ]
    print(f"[minimal-env-smoke] booting: {' '.join(cmd)}")
    # Child output goes to a temp FILE, not subprocess.PIPE: the app logs
    # heavily at startup and an undrained pipe would fill (~64KB) and deadlock
    # the child before it binds the port.
    log_file = Path(tempfile.mkstemp(prefix="minimal-env-smoke-", suffix=".log")[1])

    def _tail_log(n: int = 25) -> str:
        try:
            return "\n".join(log_file.read_text("utf-8", "replace").splitlines()[-n:])
        except OSError:
            return "<no log captured>"

    with log_file.open("w", encoding="utf-8") as sink:
        proc = subprocess.Popen(  # noqa: S603 - fixed argv
            cmd, cwd=str(repo_root), env=_child_env(repo_root, args.port),
            stdout=sink, stderr=subprocess.STDOUT, text=True,
        )
        try:
            deadline = time.monotonic() + args.timeout
            last_err = "no response"
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    print(
                        f"[minimal-env-smoke] FAIL — server exited early (code {proc.returncode}):\n{_tail_log()}",
                        file=sys.stderr,
                    )
                    return 1
                try:
                    status, body = _probe(health_url)
                    if status == 200 and "healthy" in body.lower():
                        print(f"[minimal-env-smoke] OK — /health returned 200 in a scrubbed environment: {body[:120]}")
                        return 0
                    last_err = f"status={status} body={body[:120]}"
                except (urllib.error.URLError, ConnectionError, OSError) as exc:
                    last_err = repr(exc)
                time.sleep(1.0)
            print(
                f"[minimal-env-smoke] FAIL — /health not healthy within {args.timeout}s "
                f"(last: {last_err}):\n{_tail_log()}",
                file=sys.stderr,
            )
            return 1
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            log_file.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
