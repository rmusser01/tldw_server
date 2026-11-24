"""Exercise fresh production startup, setup and notes inside the test container."""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import os
import secrets
import shutil
import signal

# The runner manages a fixed local test child.
import subprocess  # nosec B404
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from tldw_Server_API.tests.helpers.offline_network import Attempts, block_live_network

from .outbound_isolation import (
    PREFLIGHT_ENV,
    IsolationError,
    assert_isolated_runtime,
    load_prepared_image,
    run_isolated,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path("/outbound-report/result.json")
HOST = "127.0.0.1"
PORT = 18081
ORIGIN = f"http://{HOST}:{PORT}"
COVERAGE = ["production-asgi-lifespan", "local-setup", "local-notes", "python-dns-and-connect"]
SOURCE_TREES = (
    "Helper_Scripts/ci",
    "tldw_Server_API",
    "apps/mcp-unified/src",
    "packages/tldw_profile_core/src",
)


class SmokeError(RuntimeError):
    """A stable, redacted failure reason."""


@dataclass(frozen=True)
class BackendFixture:
    config_path: Path
    key: str
    origin: str = ORIGIN


def write_redacted_child_evidence(report: Path, attempts: Attempts) -> None:
    violations = [
        {"component": "backend", "reason": f"unexpected_{kind}_attempt"}
        for kind in sorted({kind for kind, _ in attempts} & {"dns", "connect"})
    ]
    staged = report.with_suffix(".tmp")
    staged.write_text(json.dumps({"shutdown": True, "violations": violations}), encoding="utf-8")
    staged.replace(report)


def check_child_result(exit_code: int, report: Path) -> list[dict[str, str]]:
    if exit_code:
        raise SmokeError("backend_child_failed")
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SmokeError("backend_missing_shutdown_evidence") from None
    if not isinstance(payload, dict) or set(payload) != {"shutdown", "violations"}:
        raise SmokeError("backend_missing_shutdown_evidence")
    if payload["shutdown"] is not True or not isinstance(payload["violations"], list):
        raise SmokeError("backend_missing_shutdown_evidence")
    if payload["violations"]:
        raise SmokeError("backend_unexpected_network")
    return []


async def serve_until_stop(stop: Path) -> None:
    import uvicorn

    from tldw_Server_API.app import main
    from tldw_Server_API.scripts.run_server_guarded_mcp import GuardedSlidesWebSocketProtocol

    if not Path(main.__file__).resolve().is_relative_to(Path.cwd()):
        raise SmokeError("backend_import_outside_fixture")
    shutdown_completed = False

    async def application(scope, receive, send) -> None:
        if scope["type"] != "lifespan":
            await main.app(scope, receive, send)
            return

        async def observe(message) -> None:
            nonlocal shutdown_completed
            await send(message)
            if message["type"] == "lifespan.shutdown.complete":
                shutdown_completed = True

        await main.app(scope, receive, observe)

    server = uvicorn.Server(uvicorn.Config(
        application, host=HOST, port=PORT, workers=1, lifespan="on", proxy_headers=False,
        ws=GuardedSlidesWebSocketProtocol, ws_per_message_deflate=False,
        loop="asyncio", access_log=False, timeout_graceful_shutdown=12,
    ))

    async def monitor_stop() -> None:
        while not stop.exists():
            await asyncio.sleep(0.05)
        try:
            deferred = getattr(main.app.state, "bg_tasks", {}).get("deferred_startup")
            if deferred is not None:
                await asyncio.wait_for(asyncio.shield(deferred), timeout=20)
        finally:
            server.should_exit = True

    monitor = asyncio.create_task(monitor_stop())
    try:
        await server.serve()
        if not server.started or not stop.is_file() or not shutdown_completed:
            raise SmokeError("backend_lifespan_failed")
        await monitor
    finally:
        if not monitor.done():
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)


def run_child(stop: Path, report: Path) -> None:
    cleanup = ExitStack()
    attempts = cleanup.enter_context(block_live_network(fixture_endpoints=frozenset({(HOST, PORT)})))
    finished = False

    # Register before importing the app, so its exit handlers run before evidence is written.
    @atexit.register
    def record_completion() -> None:
        cleanup.close()
        if finished:
            write_redacted_child_evidence(report, attempts)

    asyncio.run(serve_until_stop(stop))
    finished = True


def _prepare_fixture(root: Path, temp: Path) -> tuple[Path, BackendFixture, dict[str, str]]:
    # The image is already a projection. Copy only Python sources and app data to a fresh writable tree.
    project = temp / "project"
    ignore = shutil.ignore_patterns("__pycache__", ".env", ".env.local", "*.db", "*.db-*", "*.sqlite*")
    for relative in SOURCE_TREES:
        shutil.copytree(root / relative, project / relative, ignore=ignore)
    for filename in ("pyproject.toml", "README.md"):
        shutil.copyfile(root / filename, project / filename)
    state = temp / "state"
    state.mkdir()
    home = state / "home"
    home.mkdir()
    config = project / "tldw_Server_API/Config_Files/config.txt"
    env_file = config.with_name(".env")
    env_file.touch()
    fixture = BackendFixture(config, secrets.token_urlsafe(32))
    source_paths = (project, project / "apps/mcp-unified/src", project / "packages/tldw_profile_core/src")
    child_env = {
        "PATH": os.defpath + ":/usr/local/bin",
        "PYTHONPATH": os.pathsep.join(map(str, source_paths)),
        "PYTHONDONTWRITEBYTECODE": "1",
        PREFLIGHT_ENV: os.environ.get(PREFLIGHT_ENV, ""),
        "HOME": str(home),
        "TMPDIR": str(state),
        "XDG_CACHE_HOME": str(state / "cache"),
        "XDG_CONFIG_HOME": str(state / "settings"),
        "XDG_DATA_HOME": str(state / "data"),
        "TLDW_CONFIG_FILE": str(config),
        "TLDW_ENV_FILE": str(env_file),
        "TLDW_ENV_FILE_EXCLUSIVE": "true",
        "USER_DB_BASE_DIR": str(state / "users"),
        "DATABASE_URL": f"sqlite:///{state}/auth.db",
        "JOBS_DB_PATH": str(state / "jobs.db"),
        "TLDW_CONTENT_SQLITE_PATH": str(state / "content.db"),
        "TLDW_DB_BACKUP_PATH": str(state / "backups"),
        "SINGLE_USER_API_KEY": fixture.key,
        "JWT_SECRET_KEY": secrets.token_urlsafe(48),
        "MCP_JWT_SECRET": secrets.token_urlsafe(48),
        "MCP_API_KEY_SALT": secrets.token_urlsafe(32),
    }
    return project, fixture, child_env


def _wait_ready(child: subprocess.Popen, fixture: BackendFixture) -> None:
    import httpx

    deadline = time.monotonic() + 85
    with httpx.Client(base_url=fixture.origin, trust_env=False, timeout=2) as client:
        while child.poll() is None and time.monotonic() < deadline:
            try:
                response = client.get("/internal/ready")
                if response.status_code == 200:
                    if response.json() != {"status": "ready"}:
                        raise SmokeError("backend_readiness_contract_failed")
                    return
            except httpx.TransportError:
                pass
            time.sleep(0.15)
    raise SmokeError("backend_not_ready")


@contextmanager
def backend_fixture(root: Path, temp: Path, *, trace_prefix: Path | None = None) -> Iterator[BackendFixture]:
    """Provide the isolated fresh API and require a successful guarded shutdown."""
    project, fixture, env = _prepare_fixture(root, temp)
    stop, evidence = temp / "stop", temp / "backend-evidence.json"
    argv = [sys.executable, "-m", "Helper_Scripts.ci.outbound_backend_smoke",
            "--child", str(stop), str(evidence)]
    if trace_prefix is not None:
        argv = ["/usr/bin/strace", "-ff", "-e", "trace=connect,sendto", "-s", "0", "-o", str(trace_prefix), *argv]
    with (temp / "backend.log").open("wb") as log:
        # Only fixed local commands and private fixture paths are passed.
        child = subprocess.Popen(  # nosec B603
            argv, cwd=project, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(child, fixture)
            yield fixture
        finally:
            stop.touch()
            try:
                exit_code = child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
                raise SmokeError("backend_stop_timeout") from None
            check_child_result(exit_code, evidence)


def _exercise_backend(fixture: BackendFixture) -> None:
    import httpx

    def expect(response, code, failure):
        if response.status_code != code:
            raise SmokeError(failure)
        return response

    with httpx.Client(base_url=fixture.origin, trust_env=False, timeout=10, follow_redirects=False) as client:
        root = client.get("/")
        if root.status_code not in {302, 303, 307, 308} or root.headers.get("location") != "/setup":
            raise SmokeError("backend_setup_redirect_failed")
        setup = expect(client.get("/api/v1/setup/status"), 200, "backend_setup_status_failed").json()
        if setup.get("needs_setup") is not True or setup.get("setup_completed") is not False:
            raise SmokeError("backend_setup_defaults_failed")
        if Path(setup["config_path"]).resolve() != fixture.config_path:
            raise SmokeError("backend_config_outside_fixture")

        defaults = {
            "allow_local_file_ingest": False, "chunking_profile": "balanced",
            "metadata_mode": "automatic", "allowed_local_roots": [],
        }
        expect(client.post("/api/v1/setup/first-run/ingest-defaults", json=defaults),
               200, "backend_setup_write_failed")
        state = expect(client.get("/api/v1/setup/first-run/state"), 200, "backend_setup_read_failed").json()
        if state.get("status") != "in_progress" or "ingest_defaults" not in state.get("acknowledged_steps", []):
            raise SmokeError("backend_setup_state_failed")
        expected_step = {key: value for key, value in defaults.items() if key != "allowed_local_roots"}
        if state.get("step_data", {}).get("ingest_defaults") != expected_step | {"acknowledged": True}:
            raise SmokeError("backend_setup_persistence_failed")

        payload = {"title": "Outbound fixture note", "content": "Local fixture note content."}
        headers = {"X-API-KEY": fixture.key}
        note = expect(client.post("/api/v1/notes/", json=payload, headers=headers),
                      201, "backend_note_write_failed").json()
        from uuid import UUID

        note_id = str(UUID(note["id"]))
        stored = expect(client.get(f"/api/v1/notes/{note_id}", headers=headers),
                        200, "backend_note_read_failed").json()
        if stored.get("id") != note_id or any(stored.get(key) != value for key, value in payload.items()):
            raise SmokeError("backend_note_persistence_failed")


def run_backend_smoke(root: Path, report: Path) -> int:
    """Write a terminal record safe to copy out of the isolated container."""
    violations = []
    try:
        with tempfile.TemporaryDirectory(prefix="outbound-api-") as work:
            with backend_fixture(root, Path(work)) as fixture:
                _exercise_backend(fixture)
    except SmokeError as exc:
        violations.append({"component": "backend", "reason": str(exc)})
    # The exception may contain a raw request or an input path.
    except Exception:  # noqa: BLE001
        violations.append({"component": "backend", "reason": "backend_execution_failed"})
    report.write_text(json.dumps({"scenarios": [{
        "scenario": "backend-default", "status": "fail" if violations else "pass",
        "coverage": [] if violations else COVERAGE, "violations": violations,
    }]}), encoding="utf-8")
    return int(bool(violations))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--child", nargs=2, type=Path, metavar=("STOP", "EVIDENCE"))
    group.add_argument("--inside-container", action="store_true")
    group.add_argument("--report-dir", type=Path)
    args = parser.parse_args(argv)
    if args.child or args.inside_container:
        try:
            assert_isolated_runtime()
        except IsolationError:
            print("Backend fixture requires the isolated container preflight.", file=sys.stderr)
            return 1
    if args.child:
        run_child(*args.child)
        return 0
    if args.inside_container:
        return run_backend_smoke(ROOT, REPORT)

    try:
        image = load_prepared_image(ROOT, "backend", args.report_dir)
        with tempfile.TemporaryDirectory(prefix="backend-smoke-", dir=args.report_dir) as output:
            result = run_isolated(image, ["-m", __spec__.name, "--inside-container"],
                                  Path(output), timeout=160)
            evidence = Path(output) / "result.json"
            if evidence.is_file():
                shutil.copyfile(evidence, args.report_dir / "backend-default.json")
            return result
    except IsolationError:
        print("Backend isolation failed. Prepare the backend image and check that Docker is running.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
