import contextlib
import json
import os
import socket
import subprocess
import shlex
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import pytest

# Robust import of server_lifecycle regardless of PYTHONPATH layout in CI.
try:
    from tldw_Server_API.tests.scripts import server_lifecycle
except ModuleNotFoundError:  # pragma: no cover - environment specific
    import sys
    import importlib
    here = Path(__file__).resolve()
    repo_root = None
    for cand in [here.parent, *here.parents]:
        if (cand / "tldw_Server_API" / "tests" / "scripts" / "server_lifecycle.py").exists():
            repo_root = cand
            break
    if repo_root is None:
        try:
            repo_root = here.parents[4]
        except Exception:
            repo_root = here.parent.parent
    if repo_root and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    pkg_dir = repo_root / "tldw_Server_API" if repo_root else None
    if pkg_dir and str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    for key in list(sys.modules.keys()):
        if key == "tldw_Server_API" or key.startswith("tldw_Server_API."):
            sys.modules.pop(key, None)
    try:
        server_lifecycle = importlib.import_module("tldw_Server_API.tests.scripts.server_lifecycle")
    except ModuleNotFoundError:
        import importlib.util as importlib_util

        if not pkg_dir:
            raise
        module_path = pkg_dir / "tests" / "scripts" / "server_lifecycle.py"
        if not module_path.exists():
            raise
        spec = importlib_util.spec_from_file_location(
            "tldw_Server_API.tests.scripts.server_lifecycle", str(module_path)
        )
        assert spec and spec.loader
        module = importlib_util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # type: ignore[assignment]
        server_lifecycle = module


def _find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _set_env(overrides: Dict[str, str]) -> Dict[str, Optional[str]]:
    original = {key: os.environ.get(key) for key in overrides}
    for key, value in overrides.items():
        os.environ[key] = value
    return original


def _restore_env(original: Dict[str, Optional[str]]) -> None:
    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _print_server_log(label: str) -> None:
    log_path = Path(f"server-{label}.log")
    if log_path.exists():
        print(f"===== server log ({label}) =====")
        try:
            print(log_path.read_text(encoding="utf-8"))
        except Exception:
            _ = None


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "apps" / "tldw-frontend").exists() and (cand / "tldw_Server_API").exists():
            return cand
    return here.parents[4]


def _fetch_frontend(base_url: str) -> Dict[str, Optional[str]]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported frontend URL scheme: {parsed.scheme or '<empty>'}")
    try:
        with urllib.request.urlopen(base_url, timeout=1) as response:
            body = response.read(64 * 1024).decode(
                response.headers.get_content_charset() or "utf-8",
                errors="ignore",
            )
            return {"status": str(response.status), "body": body}
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="ignore")
        return {"status": str(error.code), "body": body}
    except Exception:
        return {"status": None, "body": None}


def _is_frontend_response(body: Optional[str]) -> bool:
    if not body:
        return False
    return "__NEXT_DATA__" in body or 'id="__next"' in body


def _read_log_tail(log_path: Optional[Path], max_bytes: int = 4000) -> str:
    if not log_path:
        return ""
    try:
        data = log_path.read_bytes()
    except Exception:
        return ""
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.decode("utf-8", errors="ignore")


def _wait_for_frontend(
    base_url: str,
    timeout: float = 60.0,
    proc: Optional[subprocess.Popen] = None,
    log_path: Optional[Path] = None,
) -> None:
    deadline = time.monotonic() + timeout
    last_status: Optional[str] = None
    while time.monotonic() < deadline:
        if proc and proc.poll() is not None:
            log_tail = _read_log_tail(log_path)
            hint = ""
            if "Unable to acquire lock" in log_tail or "another instance of next dev" in log_tail:
                hint = (
                    "Next dev appears to be running already. "
                    "Stop it or set TLDW_FRONTEND_URL to the running instance."
                )
            elif "EADDRINUSE" in log_tail or "address already in use" in log_tail:
                hint = "Frontend port is already in use; set TLDW_FRONTEND_URL or free the port."
            message = "Frontend process exited before becoming ready."
            if hint:
                message = f"{message} {hint}"
            if log_tail:
                message = f"{message}\nLog tail:\n{log_tail}"
            raise RuntimeError(message)
        response = _fetch_frontend(base_url)
        if _is_frontend_response(response["body"]):
            return
        last_status = response["status"]
        time.sleep(0.5)
    raise RuntimeError(f"Frontend did not start within {timeout} seconds (status={last_status}).")


@contextlib.contextmanager
def _ensure_frontend_running(repo_root: Path, base_url: str, server_url: str, api_key: str):
    response = _fetch_frontend(base_url)
    if _is_frontend_response(response["body"]):
        yield None
        return
    if response["status"] is not None:
        pytest.skip(
            "Frontend port is already in use by a different service. "
            "Set TLDW_FRONTEND_URL to the correct host/port."
        )

    frontend_dir = repo_root / "apps" / "tldw-frontend"
    if not frontend_dir.exists():
        pytest.skip("apps/tldw-frontend directory not found; cannot auto-start frontend.")
    if not (frontend_dir / "package.json").exists():
        pytest.skip(
            "apps/tldw-frontend/package.json not found; cannot auto-start frontend. "
            "Set TLDW_FRONTEND_URL to a running instance or restore the frontend package.json."
        )

    parsed = urlparse(base_url)
    port = parsed.port or 8080
    env = os.environ.copy()
    env.setdefault("NEXT_PUBLIC_API_URL", server_url)
    env.setdefault("NEXT_PUBLIC_API_VERSION", "v1")
    env.setdefault("NEXT_PUBLIC_X_API_KEY", api_key)
    env.setdefault("NEXT_TELEMETRY_DISABLED", "1")

    log_dir = repo_root / "tmp"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "frontend-webui-dev.log"
    log_file = log_path.open("w", encoding="utf-8")

    command = os.environ.get("TLDW_FRONTEND_CMD")
    if command:
        cmd = shlex.split(command)
    else:
        cmd = None
        for candidate in ("npm", "pnpm", "yarn", "bun"):
            if shutil.which(candidate):
                if candidate == "bun":
                    cmd = ["bun", "run", "dev", "--", "-p", str(port)]
                else:
                    cmd = [candidate, "run", "dev", "--", "-p", str(port)]
                break
        if cmd is None:
            log_file.close()
            pytest.skip("npm/pnpm/yarn/bun is not available; start the frontend manually.")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(frontend_dir),
            env=env,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    except FileNotFoundError:
        log_file.close()
        pytest.skip("Frontend command not found; start the frontend manually.")
    try:
        _wait_for_frontend(base_url, timeout=120.0, proc=proc, log_path=log_path)
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        log_file.close()


def _seed_config_script(server_url: str, api_key: str) -> str:
    config = {
        "serverUrl": server_url,
        "authMode": "single-user",
        "apiKey": api_key,
    }
    config_json = json.dumps(config)
    return (
        "(() => {"
        f"const cfg = {config_json};"
        "localStorage.setItem('tldwConfig', JSON.stringify(cfg));"
        "localStorage.setItem('__tldw_first_run_complete', 'true');"
        "})();"
    )


@pytest.fixture(scope="session")
def server_url() -> str:
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    label = os.environ.get("SERVER_LABEL", "frontend-webui")

    overrides = {
        "SERVER_LABEL": label,
        "SERVER_PORT": str(port),
        "E2E_TEST_BASE_URL": base_url,
        "AUTH_MODE": "single_user",
        "SINGLE_USER_API_KEY": os.getenv("SINGLE_USER_API_KEY", "sk-test-1234567890-VALID"),
        "CSRF_ENABLED": "true",
        "TEST_MODE": os.getenv("TEST_MODE", "true"),
        "EPHEMERAL_CLEANUP_ENABLED": "false",
        "CLAIMS_REBUILD_ENABLED": "false",
    }

    original_env = _set_env(overrides)

    try:
        server_lifecycle.start_server()
        server_lifecycle.health_check()
    except Exception:
        _print_server_log(label)
        _restore_env(original_env)
        raise

    try:
        yield base_url
    finally:
        try:
            server_lifecycle.stop_server()
        finally:
            _restore_env(original_env)


@pytest.fixture(scope="session")
def single_user_api_key(server_url: str) -> str:
    return os.environ["SINGLE_USER_API_KEY"]


@pytest.fixture(scope="session")
def frontend_url(server_url: str, single_user_api_key: str) -> str:
    base_url = os.environ.get("TLDW_FRONTEND_URL")
    if base_url:
        base_url = base_url.rstrip("/")
    else:
        port = _find_free_port()
        base_url = f"http://127.0.0.1:{port}"

    repo_root = _find_repo_root()
    with _ensure_frontend_running(repo_root, base_url, server_url, single_user_api_key):
        yield base_url


@pytest.fixture(scope="session")
def browser():
    from playwright.sync_api import sync_playwright

    headless_env = os.environ.get("PLAYWRIGHT_HEADLESS", "1").lower()
    headless = headless_env not in {"0", "false", "no"}

    with sync_playwright() as playwright:
        exec_path = Path(playwright.chromium.executable_path)
        if exec_path and not exec_path.exists():
            pytest.skip(
                "Playwright Chromium binaries are not installed. "
                "Run `python -m playwright install chromium`."
            )
        browser = playwright.chromium.launch(headless=headless)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def page(frontend_url: str, browser):
    context = browser.new_context(base_url=frontend_url)
    try:
        page = context.new_page()
        page.set_default_timeout(60_000)
        yield page
    finally:
        context.close()


@pytest.fixture
def configured_page(frontend_url: str, browser, server_url: str, single_user_api_key: str):
    context = browser.new_context(base_url=frontend_url)
    context.add_init_script(_seed_config_script(server_url, single_user_api_key))
    try:
        page = context.new_page()
        page.set_default_timeout(90_000)
        yield page
    finally:
        context.close()


@pytest.fixture(scope="session")
def sample_text_path() -> str:
    repo_root = _find_repo_root()
    asset_path = repo_root / "tldw_Server_API" / "tests" / "assets" / "e2e_sample.txt"
    if not asset_path.exists():
        pytest.skip("Sample asset not found: tldw_Server_API/tests/assets/e2e_sample.txt")
    return str(asset_path)
