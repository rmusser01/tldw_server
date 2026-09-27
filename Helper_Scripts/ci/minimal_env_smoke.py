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
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from loguru import logger

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
    """Build the scrubbed child environment: MINIMAL_ENV plus only the OS
    variables (PATH/HOME/...) the interpreter needs to run at all."""
    env: dict[str, str] = {"PYTHONPATH": str(repo_root), "PORT": str(port)}
    for passthrough in ("PATH", "HOME", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR", "VIRTUAL_ENV"):
        value = os.environ.get(passthrough)
        if value:
            env[passthrough] = value
    env.update(MINIMAL_ENV)
    return env


def _probe(url: str) -> tuple[int, str]:
    """GET *url* and return ``(status_code, body)``.

    A non-2xx response (e.g. a 500 during a broken startup) raises
    ``HTTPError``; catch it so the status and body still reach diagnostics
    instead of being flattened into a generic connection error.
    """
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:  # noqa: S310 - localhost only
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace")


def _is_public_liveness_response(status: int, body: str) -> bool:
    """Return whether a response matches the exact public liveness contract."""
    if status != 200:
        return False
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return False
    return payload == {"status": "ok"}


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
    logger.info("[minimal-env-smoke] booting: {}", " ".join(cmd))
    # Child output goes to a temp FILE, not subprocess.PIPE: the app logs
    # heavily at startup and an undrained pipe would fill (~64KB) and deadlock
    # the child before it binds the port. Close the mkstemp fd immediately
    # (we reopen the path by name) to avoid leaking a descriptor.
    fd, log_path = tempfile.mkstemp(prefix="minimal-env-smoke-", suffix=".log")
    os.close(fd)
    log_file = Path(log_path)

    def _tail_log(n: int = 25) -> str:
        """Return the last *n* lines of the child's captured output."""
        try:
            return "\n".join(log_file.read_text("utf-8", "replace").splitlines()[-n:])
        except OSError:
            return "<no log captured>"

    proc: subprocess.Popen[str] | None = None
    try:
        # The `with` closes the sink before the finally unlinks it — required on
        # Windows, where unlinking an open file raises PermissionError.
        with log_file.open("w", encoding="utf-8") as sink:
            proc = subprocess.Popen(  # noqa: S603 - fixed argv
                cmd, cwd=str(repo_root), env=_child_env(repo_root, args.port),
                stdout=sink, stderr=subprocess.STDOUT, text=True,
            )
            deadline = time.monotonic() + args.timeout
            last_err = "no response"
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    logger.bind(event="minimal_env_smoke", port=args.port).error(
                        "[minimal-env-smoke] FAIL — server exited early (code {}):\n{}",
                        proc.returncode,
                        _tail_log(),
                    )
                    return 1
                try:
                    status, body = _probe(health_url)
                    if _is_public_liveness_response(status, body):
                        logger.info(
                            "[minimal-env-smoke] OK — /health returned 200 in a scrubbed environment: {}",
                            body[:120],
                        )
                        return 0
                    last_err = f"status={status} body={body[:120]}"
                except (urllib.error.URLError, ConnectionError, OSError) as exc:
                    last_err = repr(exc)
                time.sleep(1.0)
            logger.bind(event="minimal_env_smoke", port=args.port).error(
                "[minimal-env-smoke] FAIL — /health did not return canonical public liveness within {}s (last: {}):\n{}",
                args.timeout,
                last_err,
                _tail_log(),
            )
            return 1
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        log_file.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
