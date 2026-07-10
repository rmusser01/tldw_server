#!/usr/bin/env python3
"""Run full-account Chatbooks browser export and clean-destination restore UAT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess  # nosec B404
import sys
import time
import zipfile
from dataclasses import dataclass
from http.client import HTTPConnection
from pathlib import Path
from typing import Any, Literal, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_TEXT = str(PROJECT_ROOT)
if PROJECT_ROOT_TEXT in sys.path:
    sys.path.remove(PROJECT_ROOT_TEXT)
sys.path.insert(0, PROJECT_ROOT_TEXT)
FIXTURE_SCRIPT = PROJECT_ROOT / "Helper_Scripts" / "Testing-related" / "chatbooks_full_account_uat_fixture.py"
WEBUI_SPEC = "e2e/workflows/tier-2-features/chatbooks-full-account-roundtrip.spec.ts"
EXTENSION_SPEC = "tests/e2e/chatbooks-export-download.spec.ts"


class BrowserUatError(RuntimeError):
    """Raised when a browser UAT phase cannot prove the acceptance contract."""


@dataclass(frozen=True)
class BrowserUatConfig:
    """Stable paths and ports for one browser surface round trip."""

    surface: str
    root: Path
    api_port: int
    web_port: int | None = None
    timeout_seconds: float = 240.0

    def __post_init__(self) -> None:
        if self.surface not in {"webui", "extension"}:
            raise ValueError("surface must be webui or extension")
        if not 1 <= int(self.api_port) <= 65535:
            raise ValueError("api_port must be between 1 and 65535")
        if self.surface == "webui" and self.web_port is None:
            object.__setattr__(self, "web_port", 18269)
        if self.web_port is not None and not 1 <= int(self.web_port) <= 65535:
            raise ValueError("web_port must be between 1 and 65535")
        object.__setattr__(self, "root", self.root.expanduser().resolve())

    @property
    def source_root(self) -> Path:
        return self.root / "source"

    @property
    def destination_root(self) -> Path:
        return self.root / "destination"

    @property
    def fixture_archive(self) -> Path:
        return self.source_root / "full-account.chatbook"

    @property
    def downloaded_archive(self) -> Path:
        return self.root / "browser-downloads" / f"{self.surface}-full-account.chatbook"

    @property
    def api_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"


@dataclass(frozen=True)
class UatPlanStep:
    """One externally visible phase in the browser UAT sequence."""

    step_id: str
    phase: str | None = None
    archive_path: Path | None = None


def build_uat_plan(config: BrowserUatConfig) -> list[UatPlanStep]:
    """Return the required source-export/destination-import phase order."""
    return [
        UatPlanStep("prepare-source", phase="source"),
        UatPlanStep("start-source-api", phase="source"),
        UatPlanStep(
            "browser-export",
            phase="export",
            archive_path=config.downloaded_archive,
        ),
        UatPlanStep(
            "inspect-browser-archive",
            phase="source",
            archive_path=config.downloaded_archive,
        ),
        UatPlanStep("stop-source-api", phase="source"),
        UatPlanStep("reset-destination", phase="destination"),
        UatPlanStep("start-destination-api", phase="destination"),
        UatPlanStep(
            "browser-import",
            phase="import",
            archive_path=config.downloaded_archive,
        ),
        UatPlanStep("stop-destination-api", phase="destination"),
        UatPlanStep("verify-destination", phase="destination"),
    ]


class BrowserUatRuntime(Protocol):
    """Runtime boundary used by real execution and deterministic unit tests."""

    def prepare_source(self, config: BrowserUatConfig) -> dict[str, Any]: ...

    def start_api(self, config: BrowserUatConfig, phase: str) -> None: ...

    def run_browser(
        self,
        config: BrowserUatConfig,
        phase: str,
        archive_path: Path,
    ) -> None: ...

    def inspect_archive(self, config: BrowserUatConfig) -> dict[str, Any]: ...

    def stop_api(self, phase: str) -> None: ...

    def reset_destination(self, config: BrowserUatConfig) -> dict[str, Any]: ...

    def verify_destination(self, config: BrowserUatConfig) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_expected(config: BrowserUatConfig) -> dict[str, Any]:
    expected_path = config.root / "expected.json"
    if not expected_path.is_file():
        raise BrowserUatError(f"Fixture expected state is missing: {expected_path}")
    try:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BrowserUatError("Fixture expected state is not valid JSON") from exc
    if not isinstance(expected, dict) or expected.get("schema_version") != "1.0":
        raise BrowserUatError("Fixture expected state has an unsupported schema")
    return expected


def _inventory_entry(
    inventory: list[Any],
    payload_path: str,
    label: str,
) -> dict[str, Any]:
    for entry in inventory:
        if isinstance(entry, dict) and entry.get("path") == payload_path:
            return entry
    raise BrowserUatError(f"Browser archive is missing verified {label} inventory")


def _verify_inventory_payload(
    archive: zipfile.ZipFile,
    inventory: list[Any],
    payload_path: str,
    label: str,
) -> str:
    entry = _inventory_entry(inventory, payload_path, label)
    integrity = entry.get("integrity")
    if not isinstance(integrity, dict):
        raise BrowserUatError(f"Browser archive {label} inventory is not verified")
    try:
        payload = archive.read(payload_path)
    except KeyError as exc:
        raise BrowserUatError(f"Browser archive is missing the {label} payload") from exc
    digest = hashlib.sha256(payload).hexdigest()
    if (
        integrity.get("status") != "verified"
        or integrity.get("algorithm") != "sha256"
        or integrity.get("value") != f"sha256:{digest}"
        or entry.get("size_bytes") != len(payload)
    ):
        raise BrowserUatError(f"Browser archive {label} inventory is not verified")
    return digest


def _verify_no_sensitive_archive_leaks(
    archive: zipfile.ZipFile,
    config: BrowserUatConfig,
) -> None:
    password_hash = hashlib.sha256(b"chatbooks-uat-disabled-login:chatbooks-backup-source").hexdigest()
    forbidden_values = (
        password_hash.encode("utf-8"),
        str(config.source_root).encode("utf-8"),
    )
    for member in archive.infolist():
        if member.is_dir():
            continue
        payload = archive.read(member)
        if any(value in payload for value in forbidden_values):
            raise BrowserUatError("Browser archive contains sensitive data or a raw source storage path")


def inspect_browser_archive(config: BrowserUatConfig) -> dict[str, Any]:
    """Prove the actual browser download is a complete integrity-bearing archive."""
    archive_path = config.downloaded_archive.resolve()
    download_root = config.downloaded_archive.parent.resolve()
    fixture_path = config.fixture_archive.resolve()
    if not archive_path.is_relative_to(download_root):
        raise BrowserUatError("Browser archive escaped the dedicated download directory")
    if archive_path == fixture_path:
        raise BrowserUatError("Browser download path is the fixture archive")
    if not fixture_path.is_file():
        raise BrowserUatError("Source fixture archive is missing")
    if not archive_path.is_file():
        raise BrowserUatError("Browser did not write the downloaded archive")

    archive_sha256 = _sha256_file(archive_path)
    if archive_sha256 == _sha256_file(fixture_path):
        raise BrowserUatError("Browser download is a substituted copy of the fixture archive")

    expected = _load_expected(config)
    media_path = str(expected.get("media", {}).get("archive_path") or "")
    if not media_path:
        raise BrowserUatError("Fixture expected state does not identify bundled media")

    try:
        with zipfile.ZipFile(archive_path) as archive:
            try:
                manifest = json.loads(archive.read("manifest.json"))
            except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise BrowserUatError("Browser archive has no valid manifest.json") from exc
            if not isinstance(manifest, dict) or manifest.get("version") != "1.1.0":
                raise BrowserUatError("Browser backup must use Chatbook format 1.1.0")
            summary = manifest.get("account_inventory_summary")
            if not isinstance(summary, dict) or summary.get("post_write_verification") is not True:
                raise BrowserUatError("Browser archive does not report post-write verification")
            inventory = manifest.get("file_inventory")
            if not isinstance(inventory, list):
                raise BrowserUatError("Browser archive has no verified file inventory")
            _verify_no_sensitive_archive_leaks(archive, config)

            profile_sha = _verify_inventory_payload(
                archive,
                inventory,
                "json/account_profile.json",
                "account profile",
            )
            settings_sha = _verify_inventory_payload(
                archive,
                inventory,
                "json/account_settings.json",
                "account settings",
            )
            media_sha = _verify_inventory_payload(
                archive,
                inventory,
                media_path,
                "bundled media",
            )
    except zipfile.BadZipFile as exc:
        raise BrowserUatError("Browser download is not a valid ZIP archive") from exc

    expected_media_sha = str(expected.get("media", {}).get("artifact_sha256") or "")
    if media_sha != expected_media_sha:
        raise BrowserUatError("Browser archive bundled media bytes do not match the source")
    return {
        "path": str(archive_path),
        "sha256": archive_sha256,
        "format_version": "1.1.0",
        "post_write_verification": True,
        "sensitive_data_scan": True,
        "verified_payloads": {
            "account_profile": profile_sha,
            "account_settings": settings_sha,
            "bundled_media": media_sha,
        },
    }


def validate_phase_scope(phase: str, scope: dict[str, Any]) -> None:
    """Reject an API lifecycle that is attached to the wrong account stores."""
    categories = scope.get("categories")
    if not isinstance(categories, list):
        raise BrowserUatError(f"{phase} API scope did not return account categories")
    counts = {
        str(row.get("category")): int(row.get("count") or 0)
        for row in categories
        if isinstance(row, dict) and row.get("category")
    }
    if phase == "source":
        minimums = {
            "account_settings": 1,
            "characters": 1,
            "media_records": 1,
            "media_stored_artifacts": 1,
            "embeddings": 2,
        }
        if any(counts.get(category, 0) < minimum for category, minimum in minimums.items()):
            raise BrowserUatError("source API scope does not expose the seeded full account")
        return
    if phase == "destination":
        empty_categories = ("media_records", "media_stored_artifacts", "embeddings")
        if any(counts.get(category, 0) != 0 for category in empty_categories):
            raise BrowserUatError("destination API scope is not clean")
        return
    raise BrowserUatError(f"Unsupported API scope phase: {phase}")


def _assert_destination_matches_expected(
    config: BrowserUatConfig,
    destination: dict[str, Any],
) -> None:
    expected = _load_expected(config)
    destination_media = destination.get("media")
    destination_embeddings = destination.get("embeddings")
    if not isinstance(destination_media, dict) or not isinstance(destination_embeddings, dict):
        raise BrowserUatError("Destination verification did not return media and embeddings")
    for key in ("artifact_sha256", "vector_sha256"):
        if destination_media.get(key) != expected.get("media", {}).get(key):
            raise BrowserUatError(f"Destination {key} does not match the source account")
    if destination_embeddings.get("collection_name") != expected.get("embeddings", {}).get(
        "collection_name"
    ) or destination_embeddings.get("collection_ids") != expected.get("embeddings", {}).get("collection_ids"):
        raise BrowserUatError("Destination embedding identifiers do not match the source account")


def run_browser_uat(
    config: BrowserUatConfig,
    *,
    runtime: BrowserUatRuntime | None = None,
) -> dict[str, Any]:
    """Execute source export, clean destination import, and direct-store verification."""
    active_runtime = runtime or RealBrowserUatRuntime()
    archive_evidence: dict[str, Any] = {}
    destination: dict[str, Any] = {}
    source_started = False
    destination_started = False
    try:
        prepared = active_runtime.prepare_source(config)
        active_runtime.start_api(config, "source")
        source_started = True
        active_runtime.run_browser(config, "export", config.downloaded_archive)
        archive_evidence = active_runtime.inspect_archive(config)
        active_runtime.stop_api("source")
        source_started = False

        reset = active_runtime.reset_destination(config)
        if prepared.get("source_user_id") == reset.get("destination_user_id"):
            raise BrowserUatError("Source and destination accounts must be distinct")
        active_runtime.start_api(config, "destination")
        destination_started = True
        active_runtime.run_browser(config, "import", config.downloaded_archive)
        active_runtime.stop_api("destination")
        destination_started = False
        destination = active_runtime.verify_destination(config)
        _assert_destination_matches_expected(config, destination)
        return {
            "surface": config.surface,
            "source_root": str(config.source_root),
            "destination_root": str(config.destination_root),
            "source_user_id": prepared.get("source_user_id"),
            "destination_user_id": reset.get("destination_user_id"),
            "downloaded_archive_path": str(config.downloaded_archive),
            "archive": archive_evidence,
            "destination": destination,
        }
    finally:
        if destination_started:
            active_runtime.stop_api("destination")
        if source_started:
            active_runtime.stop_api("source")
        active_runtime.close()


def _decode_json_output(output: str, label: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    result: dict[str, Any] | None = None
    result_end = -1
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            candidate, consumed = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        candidate_end = index + consumed
        if isinstance(candidate, dict) and candidate_end > result_end:
            result = candidate
            result_end = candidate_end
    if result is None:
        raise BrowserUatError(f"{label} did not return a JSON object")
    return result


def _phase_environment(config: BrowserUatConfig, phase: Literal["source", "destination"]) -> dict[str, str]:
    phase_root = config.source_root if phase == "source" else config.destination_root
    allowed_origins = [f"http://localhost:{config.web_port}"] if config.web_port else []
    return {
        "AUTH_MODE": "multi_user",
        "PROFILE": "multi-user-sqlite",
        "DATABASE_URL": f"sqlite:///{phase_root / 'users.db'}",
        "USER_DB_BASE_DIR": str(phase_root / "user_databases"),
        "JOBS_DB_PATH": str(phase_root / "jobs.db"),
        "JWT_SECRET_KEY": hashlib.sha256(b"chatbooks-full-account-uat-jwt").hexdigest(),
        "CHATBOOKS_JOBS_BACKEND": "core",
        "TLDW_JOBS_BACKEND": "core",
        "CHATBOOKS_CORE_WORKER_ENABLED": "true",
        "ALLOWED_ORIGINS": json.dumps(allowed_origins),
        "CORS_ALLOW_CREDENTIALS": "true",
    }


class RealBrowserUatRuntime:
    """Subprocess-backed implementation used by the command-line release gate."""

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._log_handles: dict[str, Any] = {}
        self._tokens: dict[str, str] = {}

    def _run_fixture(self, command: str, config: BrowserUatConfig) -> dict[str, Any]:
        # The fixture command uses fixed argv and never invokes a shell.
        completed = subprocess.run(  # nosec B603
            [sys.executable, str(FIXTURE_SCRIPT), command, "--root", str(config.root)],
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-4000:]
            raise BrowserUatError(f"Fixture {command} failed: {detail}")
        return _decode_json_output(completed.stdout, f"Fixture {command}")

    def _issue_token(
        self,
        config: BrowserUatConfig,
        phase: Literal["source", "destination"],
        user_id: int,
        username: str,
    ) -> str:
        previous = {key: os.environ.get(key) for key in _phase_environment(config, phase)}
        os.environ.update(_phase_environment(config, phase))
        try:
            from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
            from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings

            reset_settings()
            return JWTService(get_settings()).create_access_token(
                user_id=user_id,
                username=username,
                role="user",
            )
        finally:
            try:
                from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

                reset_settings()
            except (ImportError, RuntimeError):
                pass
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def prepare_source(self, config: BrowserUatConfig) -> dict[str, Any]:
        config.root.mkdir(parents=True, exist_ok=True)
        if config.downloaded_archive.exists():
            config.downloaded_archive.unlink()
        prepared = self._run_fixture("prepare", config)
        user_id = int(prepared["source_user_id"])
        self._tokens["source"] = self._issue_token(
            config,
            "source",
            user_id,
            "chatbooks-backup-source",
        )
        return prepared

    def start_api(self, config: BrowserUatConfig, phase: str) -> None:
        if phase not in {"source", "destination"}:
            raise BrowserUatError(f"Unsupported API phase: {phase}")
        if phase in self._processes:
            raise BrowserUatError(f"API process is already running for {phase}")
        logs_dir = config.root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_handle = (logs_dir / f"{phase}-api.log").open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update(_phase_environment(config, phase))
        # The API command uses fixed argv and never invokes a shell.
        process = subprocess.Popen(  # nosec B603
            [
                sys.executable,
                "-m",
                "uvicorn",
                "tldw_Server_API.app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(config.api_port),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        self._processes[phase] = process
        self._log_handles[phase] = log_handle
        self._wait_for_health(config, phase, process)
        self._verify_api_scope(config, phase)

    def _wait_for_health(
        self,
        config: BrowserUatConfig,
        phase: str,
        process: subprocess.Popen[str],
    ) -> None:
        deadline = time.monotonic() + config.timeout_seconds
        last_error = "health check did not complete"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise BrowserUatError(f"{phase} API exited before health check passed; see logs/{phase}-api.log")
            connection: HTTPConnection | None = None
            try:
                connection = HTTPConnection("127.0.0.1", config.api_port, timeout=1.0)
                connection.request("GET", "/api/v1/health")
                response = connection.getresponse()
                response.read()
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
            except OSError as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
            finally:
                if connection is not None:
                    connection.close()
            time.sleep(0.2)
        raise BrowserUatError(f"{phase} API health check failed: {last_error}")

    def _verify_api_scope(self, config: BrowserUatConfig, phase: str) -> None:
        token = self._tokens.get(phase)
        if not token:
            raise BrowserUatError(f"No API scope token is available for {phase}")
        connection = HTTPConnection("127.0.0.1", config.api_port, timeout=10)
        try:
            connection.request(
                "GET",
                "/api/v1/chatbooks/export/scope",
                headers={"Authorization": f"Bearer {token}"},
            )
            response = connection.getresponse()
            payload = response.read()
        finally:
            connection.close()
        if response.status != 200:
            raise BrowserUatError(f"{phase} API scope preflight returned HTTP {response.status}")
        try:
            scope = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BrowserUatError(f"{phase} API scope preflight returned invalid JSON") from exc
        if not isinstance(scope, dict):
            raise BrowserUatError(f"{phase} API scope preflight returned invalid data")
        validate_phase_scope(phase, scope)

    def _browser_command(self, config: BrowserUatConfig) -> tuple[Path, list[str]]:
        if shutil.which("bunx") is None:
            raise BrowserUatError("bunx is required to run browser UAT")
        if config.surface == "webui":
            return (
                PROJECT_ROOT / "apps" / "tldw-frontend",
                [
                    "bunx",
                    "playwright",
                    "test",
                    WEBUI_SPEC,
                    "--project=tier-2",
                    "--reporter=line",
                ],
            )
        return (
            PROJECT_ROOT / "apps" / "extension",
            [
                "bunx",
                "playwright",
                "test",
                EXTENSION_SPEC,
                "--project=chromium-extension",
                "--reporter=line",
            ],
        )

    def run_browser(
        self,
        config: BrowserUatConfig,
        phase: str,
        archive_path: Path,
    ) -> None:
        if phase not in {"export", "import"}:
            raise BrowserUatError(f"Unsupported browser phase: {phase}")
        token_phase = "source" if phase == "export" else "destination"
        token = self._tokens.get(token_phase)
        if not token:
            raise BrowserUatError(f"No browser token is available for {token_phase}")
        cwd, command = self._browser_command(config)
        env = os.environ.copy()
        env.update(
            {
                "TLDW_CHATBOOK_UAT_PHASE": phase,
                "TLDW_CHATBOOK_UAT_ARCHIVE_PATH": str(archive_path),
                "TLDW_CHATBOOK_UAT_ACCESS_TOKEN": token,
                "TLDW_CHATBOOK_UAT_API_URL": config.api_url,
                "TLDW_E2E_SERVER_URL": config.api_url,
                "TLDW_SERVER_URL": config.api_url,
                "NEXT_PUBLIC_API_URL": config.api_url,
                "CI": "1",
            }
        )
        if config.surface == "webui":
            web_url = f"http://localhost:{config.web_port}"
            env.update(
                {
                    "TLDW_WEB_URL": web_url,
                    "TLDW_WEB_CMD": f"bun run dev -- -p {config.web_port}",
                    "TLDW_WEB_AUTOSTART": "true",
                }
            )
        else:
            env.update(
                {
                    "TLDW_E2E_EXTENSION_HEADLESS": os.environ.get(
                        "TLDW_E2E_EXTENSION_HEADLESS",
                        "1",
                    ),
                    "TLDW_E2E_EXTENSION_TARGET_WAIT_MS": "30000",
                }
            )

        logs_dir = config.root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        browser_log = logs_dir / f"{config.surface}-{phase}.log"
        try:
            # The Playwright command uses fixed argv and never invokes a shell.
            completed = subprocess.run(  # nosec B603
                command,
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=config.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            browser_log.write_text(f"{stdout}\n{stderr}", encoding="utf-8")
            raise BrowserUatError(
                f"{config.surface} browser {phase} timed out after "
                f"{config.timeout_seconds} seconds; see {browser_log}"
            ) from exc
        browser_log.write_text(
            completed.stdout + "\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise BrowserUatError(f"{config.surface} browser {phase} failed; see {browser_log}")

    def inspect_archive(self, config: BrowserUatConfig) -> dict[str, Any]:
        return inspect_browser_archive(config)

    def stop_api(self, phase: str) -> None:
        process = self._processes.pop(phase, None)
        log_handle = self._log_handles.pop(phase, None)
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
        finally:
            if log_handle is not None:
                log_handle.close()

    def reset_destination(self, config: BrowserUatConfig) -> dict[str, Any]:
        reset = self._run_fixture("reset-destination", config)
        user_id = int(reset["destination_user_id"])
        self._tokens["destination"] = self._issue_token(
            config,
            "destination",
            user_id,
            "chatbooks-backup-destination",
        )
        return reset

    def verify_destination(self, config: BrowserUatConfig) -> dict[str, Any]:
        return self._run_fixture("verify", config)

    def close(self) -> None:
        for phase in list(self._processes):
            self.stop_api(phase)
        self._tokens.clear()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "plan"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--surface", choices=("webui", "extension"), required=True)
        subparser.add_argument("--root", type=Path, required=True)
        subparser.add_argument("--api-port", type=int, required=True)
        subparser.add_argument("--web-port", type=int)
        subparser.add_argument("--timeout-seconds", type=float, default=240.0)
    return parser


def _config_from_args(args: argparse.Namespace) -> BrowserUatConfig:
    return BrowserUatConfig(
        surface=args.surface,
        root=args.root,
        api_port=args.api_port,
        web_port=args.web_port,
        timeout_seconds=args.timeout_seconds,
    )


def main() -> int:
    args = _build_parser().parse_args()
    config = _config_from_args(args)
    if args.command == "plan":
        print(
            json.dumps(
                [
                    {
                        "step_id": step.step_id,
                        "phase": step.phase,
                        "archive_path": str(step.archive_path) if step.archive_path else None,
                    }
                    for step in build_uat_plan(config)
                ],
                indent=2,
            )
        )
        return 0
    try:
        result = run_browser_uat(config)
    except BrowserUatError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
