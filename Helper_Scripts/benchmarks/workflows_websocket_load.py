#!/usr/bin/env python3
"""Measure real workflow WebSockets on disposable tracked-source/data copies.

No dependency or authentication overrides. The production ASGI app runs on
loopback TCP with lifespan workers disabled. JWT, routing, framing and SQLite
remain real. Event latency includes the append/commit, measured from append start.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import secrets
import shutil
import socket
import sqlite3
import statistics

# fixed local interpreter and git inventory only.
import subprocess  # nosec B404
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_BYTES = 512
INTERVAL_S = 0.137


def _positive(value: str) -> int:
    """Reject empty workloads before creating a server or database."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative(value: str) -> int:
    """Allow empty history but reject negative event counts."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _summary(values: list[float]) -> dict:
    """Retain samples and nearest-rank tails, including failed/incomplete counts elsewhere."""
    ordered = sorted(values)
    return {
        "count": len(values),
        "p50": statistics.median(ordered) if ordered else None,
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1] if ordered else None,
        "p99": ordered[math.ceil(len(ordered) * 0.99) - 1] if ordered else None,
        "max": max(ordered) if ordered else None,
        "samples": values,
    }


async def _serve(runtime: Path) -> None:
    """Run the unmodified production app and sample its own event loop every 10ms."""
    expected_parent = int(os.environ["WORKFLOWS_WS_LOAD_PARENT_PID"])
    if os.getppid() != expected_parent:
        return

    import uvicorn
    from anyio.to_thread import current_default_thread_limiter
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    from tldw_Server_API.app import main as application

    server_errors = Counter()
    server_info = {}

    def record_error(record):
        if record.levelno >= logging.ERROR:
            exception_type = record.exc_info[0] if record.exc_info else None
            server_errors[exception_type.__name__ if exception_type else "ERROR"] += 1
        return True

    error_logger = logging.getLogger("uvicorn.error")
    error_logger.addFilter(record_error)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(
        uvicorn.Config(
            application.app,
            lifespan="off",
            access_log=False,
            log_level="error",
            loop="asyncio",
            ws="websockets",
            timeout_graceful_shutdown=5,
        )
    )
    samples = []

    async def monitor():
        while not server.started:
            if os.getppid() != expected_parent:
                server.should_exit = True
                return
            await asyncio.sleep(0.01)
        server_info.update(
            test_mode=bool(application._TEST_MODE or application._runtime_test_mode_active()),
            explicit_pytest_runtime=bool(application._EXPLICIT_PYTEST_RUNTIME),
            threadpool_tokens=current_default_thread_limiter().total_tokens,
        )
        (runtime / "server.json").write_text(json.dumps(server_info))
        (runtime / "port.json").write_text(json.dumps(listener.getsockname()[1]))
        while True:
            target = time.monotonic_ns() + 10_000_000
            await asyncio.sleep(0.01)
            now = time.monotonic_ns()
            samples.append((now, max(0, now - target) / 1_000_000))
            if (runtime / "stop").exists() or os.getppid() != expected_parent:
                server.should_exit = True
                return

    task = asyncio.create_task(monitor())
    try:
        await server.serve(sockets=[listener])
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        listener.close()
        error_logger.removeFilter(record_error)
        server_info["errors"] = dict(server_errors)
        (runtime / "server.json").write_text(json.dumps(server_info))
        (runtime / "loop.json").write_text(json.dumps(samples))


async def _trial(db, url: str, token: str, clients: int, backlog: int, rounds: int, trial: int) -> dict:
    """Validate ordered history/live events while all subscribers share one ASGI loop."""
    from websockets.asyncio.client import connect

    run_ids = [uuid4().hex for _ in range(clients)]
    for run_id in run_ids:
        db.create_run(
            run_id=run_id,
            tenant_id="benchmark",
            user_id="1",
            inputs={},
            workflow_id=None,
            definition_version=1,
            definition_snapshot={"name": "ws-load", "version": 1, "steps": []},
        )
        for sequence in range(backlog):
            db.append_event(
                "benchmark", run_id, "history", {"run": run_id, "index": sequence, "body": "x" * PAYLOAD_BYTES}
            )
    ready = [asyncio.Event() for _ in run_ids]
    handshakes, snapshots, history_times, event_times = [], [], [], []
    errors = Counter()
    received = mismatches = 0
    started_ns = time.monotonic_ns()

    async def subscriber(index, run_id):
        nonlocal received, mismatches
        started = time.monotonic_ns()
        try:
            async with connect(
                f"{url}/api/v1/workflows/ws?run_id={run_id}",
                additional_headers={"Authorization": f"Bearer {token}"},
                open_timeout=15,
                close_timeout=3,
                max_size=2**20,
                proxy=None,
            ) as ws:
                handshakes.append((time.monotonic_ns() - started) / 1_000_000)
                snapshot = json.loads(await asyncio.wait_for(ws.recv(), 15))
                if snapshot.get("type") != "snapshot" or snapshot.get("run", {}).get("run_id") != run_id:
                    mismatches += 1
                    return
                snapshots.append((time.monotonic_ns() - started) / 1_000_000)
                history_seen = live_seen = 0
                if not backlog:
                    ready[index].set()
                while history_seen < backlog or live_seen < rounds:
                    frame = json.loads(await asyncio.wait_for(ws.recv(), 15))
                    if frame.get("type") == "heartbeat":
                        continue
                    payload = frame.get("payload", {})
                    expected = history_seen + live_seen + 1
                    if frame.get("event_seq") != expected or payload.get("run") != run_id:
                        mismatches += 1
                        return
                    if history_seen < backlog:
                        if frame.get("event_type") != "history" or payload.get("index") != history_seen:
                            mismatches += 1
                            return
                        history_seen += 1
                        if history_seen == backlog:
                            history_times.append((time.monotonic_ns() - started) / 1_000_000)
                            ready[index].set()
                    else:
                        if frame.get("event_type") != "live" or payload.get("index") != live_seen:
                            mismatches += 1
                            return
                        event_times.append((time.monotonic_ns() - payload["append_started_ns"]) / 1_000_000)
                        live_seen += 1
                        received += 1
        except Exception as exc:  # noqa: BLE001 - report every load-client failure by safe class only.
            errors[type(exc).__name__] += 1
        finally:
            ready[index].set()

    tasks = [asyncio.create_task(subscriber(i, run_id)) for i, run_id in enumerate(run_ids)]
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ready)), 60)
        for index in range(rounds):
            for run_id in run_ids:

                def append(run_id=run_id, index=index):
                    db.append_event(
                        "benchmark",
                        run_id,
                        "live",
                        {
                            "run": run_id,
                            "index": index,
                            "append_started_ns": time.monotonic_ns(),
                            "body": "x" * PAYLOAD_BYTES,
                        },
                    )

                await asyncio.to_thread(append)
            await asyncio.sleep(INTERVAL_S)
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    ended_ns = time.monotonic_ns()
    # Allow the existing one-second poll to observe closed peers before the next trial.
    await asyncio.sleep(1.1)
    return {
        "clients": clients,
        "backlog": backlog,
        "trial": trial,
        "start_ns": started_ns,
        "end_ns": ended_ns,
        "expected": clients * rounds,
        "received": received,
        "mismatches": mismatches,
        "errors": dict(errors),
        "handshake_ms": _summary(handshakes),
        "snapshot_ms": _summary(snapshots),
        "history_ms": _summary(history_times),
        "event_ms": _summary(event_times),
        "valid": received == clients * rounds and not errors and not mismatches,
    }


async def _measure(db, url: str, args) -> tuple[list[dict], dict]:
    """Verify network-level authorization refusals, then measure warm-server trials."""
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidStatus

    from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager

    manager = get_jwt_manager()
    token = manager.create_access_token(subject="1")
    foreign = manager.create_access_token(subject="2")
    run_id = uuid4().hex
    db.create_run(
        run_id=run_id,
        tenant_id="benchmark",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "auth-control", "version": 1, "steps": []},
    )
    authorization = {}
    for label, target, credential in [
        ("missing", run_id, None),
        ("foreign_owner", run_id, foreign),
        ("missing_run", uuid4().hex, token),
    ]:
        headers = {"Authorization": f"Bearer {credential}"} if credential else {}
        try:
            async with connect(
                f"{url}/api/v1/workflows/ws?run_id={target}", additional_headers=headers, open_timeout=15, proxy=None
            ):
                authorization[label] = 101
        except InvalidStatus as exc:
            authorization[label] = exc.response.status_code
    trials = []
    for backlog in args.backlog:
        for clients in args.clients:
            for trial in range(args.trials):
                trials.append(await _trial(db, url, token, clients, backlog, args.rounds, trial))
    return trials, authorization


def benchmark(args) -> dict:
    """Copy tracked working source and sandbox all app imports and runtime data."""
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    with tempfile.TemporaryDirectory(prefix="workflows-ws-load-") as directory:
        runtime = Path(directory).resolve()
        project = runtime / "project"
        git_path = shutil.which("git")
        if git_path is None:
            raise RuntimeError("Git is required to inventory isolated source")
        # Resolved local git binary; arguments are a fixed read-only source inventory.
        inventory = subprocess.run(  # nosec B603
            [
                git_path,
                "ls-files",
                "-z",
                "tldw_Server_API",
                "Helper_Scripts",
                "apps/mcp-unified/src",
                "packages/tldw_profile_core/src",
                "pyproject.toml",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            timeout=30,
        )
        files = inventory.stdout.decode().split("\0")
        for relative in filter(None, files):
            source = ROOT / relative
            if source.is_file() and not source.is_symlink():
                target = project / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        script = project / "Helper_Scripts/benchmarks/workflows_websocket_load.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(__file__, script)
        (runtime / "benchmark.env").touch()
        os.environ.pop("PYTEST_CURRENT_TEST", None)
        os.environ.update(
            {
                "USER_DB_BASE_DIR": str(runtime / "users"),
                "DATABASE_URL": f"sqlite:///{runtime}/users.db",
                "JOBS_DB_PATH": str(runtime / "jobs.db"),
                "TLDW_DB_BACKUP_PATH": str(runtime / "backups"),
                "AUTH_MODE": "single_user",
                "TEST_MODE": "false",
                "TESTING": "false",
                "TLDW_TEST_MODE": "false",
                "TLDW_ENV_FILE": str(runtime / "benchmark.env"),
                "TLDW_ENV_FILE_EXCLUSIVE": "true",
                "DISABLE_HEAVY_STARTUP": "1",
                "MCP_JWT_SECRET": secrets.token_urlsafe(48),
                "MCP_API_KEY_SALT": secrets.token_urlsafe(32),
                "JWT_SECRET_KEY": secrets.token_urlsafe(48),
                "SINGLE_USER_API_KEY": secrets.token_urlsafe(32),
                "WORKFLOWS_SQLITE_POOL_SIZE": "0",
                "TLDW_CONTENT_DB_BACKEND": "sqlite",
                "TLDW_DB_BACKEND": "sqlite",
                "TLDW_CONTENT_SQLITE_PATH": str(runtime / "media.db"),
                "LOG_LEVEL": "ERROR",
                "OTEL_SDK_DISABLED": "true",
            }
        )
        source_paths = [
            str(project),
            str(project / "apps/mcp-unified/src"),
            str(project / "packages/tldw_profile_core/src"),
        ]
        sys.path[0:0] = source_paths
        os.environ["PYTHONPATH"] = os.pathsep.join(source_paths + [os.environ.get("PYTHONPATH", "")])
        from tldw_Server_API.app.core.DB_Management.DB_Manager import (
            create_workflows_database,
            single_user_workflows_path,
        )

        if not Path(single_user_workflows_path).resolve().is_relative_to(runtime):
            raise RuntimeError("Workflow database is not isolated")
        db = create_workflows_database()
        with (runtime / "server.log").open("w") as log:
            # fixed interpreter and copied local benchmark server.
            process = subprocess.Popen(  # nosec B603
                [sys.executable, str(script), "--serve", str(runtime)],
                cwd=runtime,
                stdout=log,
                stderr=log,
                env={**os.environ, "WORKFLOWS_WS_LOAD_PARENT_PID": str(os.getpid())},
            )
            try:
                deadline = time.monotonic() + 60
                port_file = runtime / "port.json"
                while not port_file.exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("Benchmark server did not start")
                    time.sleep(0.05)
                port = json.loads(port_file.read_text())
                trials, authorization = asyncio.run(_measure(db, f"ws://127.0.0.1:{port}", args))
            finally:
                db.close()
                (runtime / "stop").touch()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        server_info = json.loads((runtime / "server.json").read_text())
        loop_file = runtime / "loop.json"
        loop = json.loads(loop_file.read_text()) if loop_file.exists() else []
        for trial in trials:
            started, ended = trial.pop("start_ns"), trial.pop("end_ns")
            trial["loop_lag_ms"] = _summary([lag for stamp, lag in loop if started <= stamp <= ended])
        return {
            "schema_version": 1,
            "transport": "loopback TCP / production ASGI app",
            "runtime": {
                "python": platform.python_version(),
                "sqlite": sqlite3.sqlite_version,
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "packages": {
                    name: importlib.metadata.version(name) for name in ("fastapi", "starlette", "uvicorn", "websockets")
                },
            },
            "source_sha256": {
                p: hashlib.sha256((project / p).read_bytes()).hexdigest()
                for p in [
                    "tldw_Server_API/app/api/v1/endpoints/workflows.py",
                    "tldw_Server_API/app/core/DB_Management/Workflows_DB.py",
                    "Helper_Scripts/benchmarks/workflows_websocket_load.py",
                ]
            },
            "method": {
                "rounds": args.rounds,
                "interval_s": INTERVAL_S,
                "payload_bytes": PAYLOAD_BYTES,
                "loop_probe_ms": 10,
                "lifespan": "off",
                "test_mode": server_info["test_mode"],
                "event_latency": "append start through SQLite commit, poll, framing and client receive",
                "source": "tracked working files copied; no dependency/auth overrides",
            },
            "server": server_info,
            "authorization": authorization,
            "server_exit": process.returncode,
            "trials": trials,
            "valid": all(trial["valid"] and trial["loop_lag_ms"]["count"] > 0 for trial in trials)
            and process.returncode == 0
            and all(code == 403 for code in authorization.values())
            and not server_info["errors"]
            and not server_info["test_mode"]
            and not server_info["explicit_pytest_runtime"],
        }


def main() -> int:
    """Write complete trial/correctness evidence; fail for any invalid measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=_positive, nargs="+", default=[1, 16, 64])
    parser.add_argument("--backlog", type=_nonnegative, nargs="+", default=[0, 500])
    parser.add_argument("--rounds", type=_positive, default=30)
    parser.add_argument("--trials", type=_positive, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--serve", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        sys.path.insert(0, str(ROOT))
        asyncio.run(_serve(args.serve))
        return 0
    report = benchmark(args)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return int(not report["valid"])


if __name__ == "__main__":
    raise SystemExit(main())
