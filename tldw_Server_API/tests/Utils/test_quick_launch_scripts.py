"""Contract tests for no-Docker quick-launch scripts."""

import ast
import json
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_API_KEY = "quick-launch-test-key-1234567890"
WEBUI_ENV_NAMES = (
    "NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE",
    "TLDW_INTERNAL_API_ORIGIN",
    "NEXT_PUBLIC_API_URL",
    "AUTH_MODE",
    "SINGLE_USER_API_KEY",
    "SINGLE_USER_SESSION_COOKIE_NAME",
    "TLDW_WEBUI_EXPOSE_RUNTIME_AUTH",
)
FRONTEND_ENV_FILES = (".env.production.local", ".env.local", ".env.production", ".env")
# Stands in for `bun`: logs each call with the env the WebUI would see; builds succeed
# (writing .next/BUILD_ID) unless FAKE_BUILD_FAILS is set.
FAKE_BUN_SOURCE = """
import json, os, pathlib, sys
args = sys.argv[1:]
names = os.environ["FAKE_WEBUI_ENV_NAMES"].split(",")
with open(os.environ["FAKE_BUN_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps({"args": args, "env": {n: os.environ.get(n) for n in names}}) + "\\n")
if args[:1] == ["run"] and args[1].startswith("build:"):
    if os.environ.get("FAKE_BUILD_FAILS"):
        sys.exit(1)
    pathlib.Path(".next").mkdir(exist_ok=True)
    pathlib.Path(".next/BUILD_ID").write_text("fake-build", encoding="utf-8")
"""
# Stands in for the venv python: `-m uvicorn` logs its cookie env and serves /health
# (FAKE_API_MODE=crash exits at once, =unhealthy answers 503), the setup wizard is a
# no-op, and anything else (the launcher's dotenv/health one-liners) runs for real.
FAKE_PYTHON_SOURCE = """
import json, os, sys
args = sys.argv[1:]
if args[:2] == ["-m", "uvicorn"]:
    with open(os.environ["FAKE_API_LOG"], "a", encoding="utf-8") as log:
        log.write(json.dumps({n: os.environ.get(n) for n in ("SESSION_COOKIE_SECURE", "CSRF_ENABLED")}) + "\\n")
    mode = os.environ.get("FAKE_API_MODE", "")
    if mode == "crash":
        sys.exit(3)
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(503 if mode == "unhealthy" else 200)
            self.end_headers()
        def log_message(self, *_):
            pass
    port = int(args[args.index("--port") + 1])
    HTTPServer((args[args.index("--host") + 1], port), Health).serve_forever()
elif args[:2] == ["-m", "tldw_Server_API.cli.wizard.cli"]:
    sys.exit(0)
else:
    os.execv(sys.executable, [sys.executable, *args])
"""

def _require(condition: bool, message: str) -> None:
    """Fail with a descriptive assertion message when a contract is broken."""
    if not condition:
        pytest.fail(message)


def _read(path: str) -> str:
    """Return a repository-relative script's text for contract assertions."""
    script_path = REPO_ROOT / path
    if not script_path.exists():
        pytest.fail(f"{path} should exist")
    return script_path.read_text(encoding="utf-8")


def _shell_function_body(source: str, function_name: str) -> str:
    """Return a shell function body from quick-launch.sh source text."""
    marker = f"{function_name}() {{"
    _require(marker in source, f"quick-launch.sh should define {function_name}")
    body_with_suffix = source.split(marker, 1)[1]
    _require("\n}\n" in body_with_suffix, f"quick-launch.sh should close {function_name}")
    return body_with_suffix.split("\n}\n", 1)[0]


def test_launcher_contract_tests_include_docstrings() -> None:
    """Launcher contract tests should keep module and function docstrings."""
    source = _read("tldw_Server_API/tests/Utils/test_quick_launch_scripts.py")
    module = ast.parse(source)
    missing = []

    if ast.get_docstring(module) is None:
        missing.append("<module>")

    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node) is None:
                missing.append(f"{node.name}:{node.lineno}")

    _require(not missing, f"launcher contract tests should document: {', '.join(missing)}")


@pytest.mark.parametrize(
    "path",
    [
        "quick-launch.sh",
        "quick-launch.command",
        "quick-launch.ps1",
    ],
)
def test_quick_launch_scripts_exist(path: str) -> None:
    """Each supported platform should have a repo-root quick-launch shortcut."""
    _require((REPO_ROOT / path).is_file(), f"{path} should exist at the repository root")


@pytest.mark.parametrize(
    "path",
    [
        "quick-launch.sh",
        "quick-launch.ps1",
    ],
)
def test_quick_launch_scripts_wrap_existing_local_single_contract(path: str) -> None:
    """Launchers should reuse the existing local-single setup/start contract."""
    text = _read(path)

    for expected in (
        ".venv",
        "pip",
        "install",
        "tldw_Server_API.cli.wizard.cli",
        "--profile",
        "local-single",
        "uvicorn",
        "tldw_Server_API.app.main:app",
        "127.0.0.1",
        "8000",
    ):
        _require(expected in text, f"{path} should include {expected}")


def test_macos_command_delegates_to_shell_launcher() -> None:
    """The Finder-friendly macOS shortcut should reuse the shell launcher."""
    text = _read("quick-launch.command")

    _require("quick-launch.sh" in text, "quick-launch.command should delegate to quick-launch.sh")
    _require("exec" in text, "quick-launch.command should replace itself with the shell launcher")


def test_shell_launcher_supports_git_bash_venv_layout() -> None:
    """Shell launcher should work when a venv exposes Windows Scripts/python."""
    text = _read("quick-launch.sh")

    _require("resolve_venv_python" in text, "quick-launch.sh should resolve venv Python dynamically")
    _require("$VENV_DIR/Scripts/python" in text, "quick-launch.sh should support Windows venv layout")
    _require("$VENV_DIR/bin/python" in text, "quick-launch.sh should support POSIX venv layout")


@pytest.mark.parametrize(
    ("path", "mode_contract"),
    [
        ("quick-launch.sh", 'mode="${1:-all}"'),
        ("quick-launch.ps1", '[ValidateSet("api", "webui", "all", "help")]'),
    ],
)
def test_root_quick_launch_scripts_expose_api_webui_and_all_modes(
    path: str, mode_contract: str
) -> None:
    """Root quick-launch scripts should be the canonical API/WebUI mode launchers."""
    text = _read(path)

    _require(mode_contract in text, f"{path} should default to all-mode launch")
    for expected in ("api", "webui", "all", "run_api", "run_webui", "run_all"):
        _require(expected in text.lower(), f"{path} should include {expected}")


@pytest.mark.parametrize(
    "path",
    [
        "quick-launch.sh",
        "quick-launch.ps1",
    ],
)
def test_root_quick_launch_scripts_start_current_webui(path: str) -> None:
    """Root quick-launch scripts should launch the current Next.js WebUI."""
    text = _read(path)

    for expected in (
        "apps/tldw-frontend",
        "bun",
        "run",
        "dev",
        "TLDW_WEBUI_PORT",
        "NEXT_PUBLIC_API_URL",
        "8080",
    ):
        _require(expected in text, f"{path} should include {expected}")


def test_root_quick_launch_scripts_avoid_zero_bind_host_in_webui_url() -> None:
    """WebUI API defaults should not point browser clients at 0.0.0.0."""
    shell_text = _read("quick-launch.sh")
    powershell_text = _read("quick-launch.ps1")
    zero_bind_host = ".".join(("0", "0", "0", "0"))

    for expected in ("api_url_host", zero_bind_host, "127.0.0.1"):
        _require(expected in shell_text, f"quick-launch.sh should include {expected}")

    for expected in ("Resolve-QuickLaunchApiUrl", zero_bind_host, "127.0.0.1"):
        _require(expected in powershell_text, f"quick-launch.ps1 should include {expected}")


def test_shell_launcher_backgrounds_uvicorn_directly_for_cleanup() -> None:
    """Shell all-mode cleanup should target the real Uvicorn process PID."""
    text = _read("quick-launch.sh")
    start_api_block = _shell_function_body(text, "start_api_background")

    _require(
        'TLDW_ENV_FILE="$ENV_FILE" "$VENV_PYTHON" -m uvicorn' in start_api_block,
        "quick-launch.sh should start Uvicorn directly in all mode",
    )
    _require(
        '--port "$PORT" &' in start_api_block,
        "quick-launch.sh should background the Uvicorn command itself",
    )
    _require(
        'api_pid="$!"' in start_api_block,
        "quick-launch.sh should capture the backgrounded Uvicorn PID",
    )
    _require("(\n" not in start_api_block, "quick-launch.sh should not wrap Uvicorn in a subshell")


def test_powershell_launcher_validates_api_start_delay() -> None:
    """PowerShell launcher should ignore invalid TLDW_API_START_DELAY (max API wait) values."""
    text = _read("quick-launch.ps1")

    for expected in (
        "Resolve-QuickLaunchApiStartDelay",
        "TLDW_API_START_DELAY",
        "-match '^\\d+$'",
        "using 120",
    ):
        _require(expected in text, f"quick-launch.ps1 should include {expected}")


def test_powershell_launcher_uses_current_host_for_child_api_window() -> None:
    """PowerShell all mode should spawn the same edition as the parent process."""
    text = _read("quick-launch.ps1")

    for expected in (
        "$PSVersionTable.PSEdition",
        '"pwsh"',
        "$PsExe",
        'Start-Process -FilePath $PsExe',
    ):
        _require(expected in text, f"quick-launch.ps1 should include {expected}")


@pytest.mark.parametrize(
    "path",
    [
        "Helper_Scripts/Installer_Scripts/MacOS_Run_tldw.sh",
        "Helper_Scripts/Installer_Scripts/Linux_Run_tldw.sh",
    ],
)
def test_unix_installer_run_scripts_delegate_to_root_quick_launch(path: str) -> None:
    """Installer shortcuts should wrap the canonical root shell launcher."""
    text = _read(path)

    _require("quick-launch.sh" in text, f"{path} should delegate to quick-launch.sh")
    _require("TLDW_VENV_DIR" in text, f"{path} should set legacy installer venv defaults")
    _require("TLDW_SKIP_INSTALL" in text, f"{path} should avoid duplicate installer pip work")

    for duplicated_fragment in (
        "python3 -m uvicorn",
        "bun run dev",
        "apps/tldw-frontend",
    ):
        _require(
            duplicated_fragment not in text,
            f"{path} should not duplicate {duplicated_fragment}",
        )


def test_windows_installer_run_script_delegates_to_root_quick_launch() -> None:
    """Windows installer shortcut should wrap the canonical PowerShell launcher."""
    text = _read("Helper_Scripts/Installer_Scripts/Windows_Run_tldw.bat")

    _require("quick-launch.ps1" in text, "Windows wrapper should delegate to quick-launch.ps1")
    _require("TLDW_VENV_DIR" in text, "Windows wrapper should set legacy installer venv defaults")
    _require("TLDW_SKIP_INSTALL" in text, "Windows wrapper should avoid duplicate installer pip work")

    for duplicated_fragment in (
        "python -m uvicorn",
        "bun run dev",
        "apps\\tldw-frontend",
    ):
        _require(
            duplicated_fragment not in text,
            f"Windows wrapper should not duplicate {duplicated_fragment}",
        )


def test_launcher_tests_do_not_use_brittle_exit_count_contract() -> None:
    """Launcher coverage should avoid exact global exit-code string counts."""
    legacy_test_path = REPO_ROOT / "Helper_Scripts/Installer_Scripts/Tests/test_run_tldw_launchers.py"
    text = _read("tldw_Server_API/tests/Utils/test_quick_launch_scripts.py")
    count_call = "source" ".count"
    exit_propagation = "exit /b " "!errorlevel!"

    _require(not legacy_test_path.exists(), "legacy launcher test module should not remain")
    _require(count_call not in text, "launcher tests should not count global source strings")
    _require(
        exit_propagation not in text,
        "launcher tests should not assert an exact exit propagation count",
    )


@pytest.mark.parametrize(
    "path",
    [
        "quick-launch.sh",
        "quick-launch.ps1",
    ],
)
def test_quick_launch_scripts_skip_reinstall_after_initial_setup(path: str) -> None:
    """Quick launch should not force networked pip install on every run."""
    text = _read(path)

    _require(".initialized" in text, f"{path} should record completed local dependency setup")
    _require("TLDW_FORCE_INSTALL" in text, f"{path} should allow explicit reinstall/update")


def test_powershell_launcher_validates_env_port() -> None:
    """PowerShell launcher should validate TLDW_PORT before casting to int."""
    text = _read("quick-launch.ps1")

    _require("Resolve-QuickLaunchPort" in text, "quick-launch.ps1 should use a port parsing helper")
    _require("-match" in text and "\\d+" in text, "quick-launch.ps1 should validate numeric ports")
    _require("TLDW_PORT" in text, "quick-launch.ps1 should keep env override support")


@pytest.mark.parametrize(
    "path",
    [
        "quick-launch.sh",
        "quick-launch.command",
        "quick-launch.ps1",
    ],
)
def test_quick_launch_scripts_do_not_use_docker_make_or_legacy_gradio(path: str) -> None:
    """No-Docker launchers should not route through Docker, Make, or old UI entrypoints."""
    text = _read(path)

    forbidden_fragments = (
        "docker compose",
        "make quickstart",
        "make install-local",
        "summarize.py",
        "-gui",
    )
    for fragment in forbidden_fragments:
        _require(fragment not in text, f"{path} should not include {fragment}")


@pytest.mark.parametrize(
    "path",
    [
        "quick-launch.sh",
        "quick-launch.command",
        "quick-launch.ps1",
    ],
)
def test_quick_launch_scripts_do_not_print_api_keys_by_default(path: str) -> None:
    """Default launch output should avoid echoing full API keys."""
    text = _read(path)

    forbidden_fragments = (
        "grep '^SINGLE_USER_API_KEY='",
        "Your API key",
        "Your API Key",
        "cut -d= -f2-",
    )
    for fragment in forbidden_fragments:
        _require(fragment not in text, f"{path} should not include {fragment}")


def _free_port() -> int:
    """Return a TCP port that is free on 127.0.0.1 right now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class ShellLauncherSandbox:
    """A copy of quick-launch.sh with a fake bun, venv python and env file around it."""

    def __init__(self, root: Path) -> None:
        """Lay out the launcher, a stub WebUI dir, a fake venv and a fake bun."""
        self.root = root
        self.bun_log = root / "bun-calls.jsonl"
        self.api_log = root / "api-env.jsonl"
        self.webui_dir = root / "apps" / "tldw-frontend"
        self.key_file = self.webui_dir / ".next" / "quick-launch-build-key"
        self.env_file = root / "test.env"
        self.port = _free_port()
        shutil.copy2(REPO_ROOT / "quick-launch.sh", root / "quick-launch.sh")
        self.webui_dir.mkdir(parents=True)
        self.write_env_file(f'AUTH_MODE=single_user\nSINGLE_USER_API_KEY="{TEST_API_KEY}"\n')
        bin_dir = root / "fake-bin"
        venv_bin = root / "venv" / "bin"
        bin_dir.mkdir()
        venv_bin.mkdir(parents=True)
        for name, source in (("fake_bun.py", FAKE_BUN_SOURCE), ("fake_python.py", FAKE_PYTHON_SOURCE)):
            (root / name).write_text(source, encoding="utf-8")
        self._write_wrapper(bin_dir / "bun", root / "fake_bun.py")
        self._write_wrapper(venv_bin / "python", root / "fake_python.py")
        self.base_env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "HOME": str(root),
            "GIT_CEILING_DIRECTORIES": str(root.parent),
            "TLDW_PYTHON": sys.executable,
            "TLDW_SKIP_INSTALL": "1",
            "TLDW_VENV_DIR": str(root / "venv"),
            "TLDW_ENV_FILE": str(self.env_file),
            "TLDW_API_PORT": str(self.port),
            "TLDW_WEBUI_PORT": "8097",
            "FAKE_BUN_LOG": str(self.bun_log),
            "FAKE_API_LOG": str(self.api_log),
            "FAKE_WEBUI_ENV_NAMES": ",".join(WEBUI_ENV_NAMES),
        }

    @property
    def origin(self) -> str:
        """Return the loopback API origin the launcher should derive."""
        return f"http://127.0.0.1:{self.port}"

    @staticmethod
    def _write_wrapper(path: Path, script: Path) -> None:
        """Write an executable shell wrapper that runs `script` with this Python."""
        command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
        path.write_text(f'#!/bin/sh\nexec {command} "$@"\n', encoding="utf-8")
        path.chmod(0o755)

    def write_env_file(self, content: str) -> None:
        """Replace the backend env file the launcher reads."""
        self.env_file.write_text(content, encoding="utf-8")

    def run(self, mode: str = "webui", **extra_env: str) -> tuple[int, str]:
        """Run the launcher in `mode`; return its exit code and combined output."""
        for log in (self.bun_log, self.api_log):
            log.unlink(missing_ok=True)
        result = subprocess.run(  # nosec B603 - fixed local test script
            ["bash", str(self.root / "quick-launch.sh"), mode],
            cwd=self.root,
            env={**self.base_env, **extra_env},
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        return result.returncode, result.stdout + result.stderr

    def launch(self, mode: str = "webui", **extra_env: str) -> list[dict]:
        """Run the launcher, require success, and return the bun calls it made."""
        code, output = self.run(mode, **extra_env)
        _require(code == 0, f"launcher failed ({code}): {output}")
        return self.bun_calls()

    def bun_calls(self) -> list[dict]:
        """Return the recorded bun calls of the last run."""
        return self._jsonl(self.bun_log)

    def api_envs(self) -> list[dict]:
        """Return the cookie env each fake API start saw in the last run."""
        return self._jsonl(self.api_log)

    @staticmethod
    def _jsonl(path: Path) -> list[dict]:
        """Parse a JSON-lines log that may not exist."""
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def git(self, *args: str) -> None:
        """Run git in the sandbox so the launcher sees a source revision."""
        subprocess.run(  # nosec B603 B607 - fixed git arguments in a temp dir
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
            cwd=self.root,
            env=self.base_env,
            capture_output=True,
            check=True,
        )

    def commit_file(self, relative: str, content: str) -> None:
        """Commit one file so `HEAD:apps` reflects (or ignores) the change."""
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.git("add", relative)
        self.git("commit", "-q", "-m", relative)


@pytest.fixture
def shell_sandbox(tmp_path: Path) -> ShellLauncherSandbox:
    """Provide an isolated quick-launch.sh sandbox (POSIX shells only)."""
    if shutil.which("bash") is None or sys.platform.startswith("win"):
        pytest.skip("quick-launch.sh behaviour tests need a POSIX bash")
    return ShellLauncherSandbox(tmp_path)


def _call_args(calls: list[dict]) -> list[list[str]]:
    """Return only the argument lists of recorded bun calls."""
    return [call["args"] for call in calls]


def _start_args(host: str) -> list[str]:
    """Return the expected `bun run start` arguments for a WebUI bind host."""
    return ["run", "start", "--", "-p", "8097", "-H", host]


def test_shell_launcher_builds_once_then_serves_production_webui(shell_sandbox) -> None:
    """Default launch builds the quickstart WebUI once and serves it with next start."""
    code, output = shell_sandbox.run()
    first = shell_sandbox.bun_calls()

    _require(code == 0, f"launcher failed: {output}")
    _require(
        _call_args(first) == [["run", "build:prod"], _start_args("127.0.0.1")],
        f"first launch should build then start the production WebUI: {first}",
    )
    build_env, start_env = first[0]["env"], first[1]["env"]
    _require(
        build_env["TLDW_INTERNAL_API_ORIGIN"] == shell_sandbox.origin,
        "the build should bake the backend origin into the quickstart proxy",
    )
    _require(build_env["SINGLE_USER_API_KEY"] is None, "the API key must stay out of the build")
    _require(
        start_env == {
            "NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE": "quickstart",
            "TLDW_INTERNAL_API_ORIGIN": shell_sandbox.origin,
            "NEXT_PUBLIC_API_URL": None,
            "AUTH_MODE": "single_user",
            "SINGLE_USER_API_KEY": TEST_API_KEY,
            "SINGLE_USER_SESSION_COOKIE_NAME": None,
            "TLDW_WEBUI_EXPOSE_RUNTIME_AUTH": "1",
        },
        f"the WebUI server should get the Docker-quickstart runtime auth env: {start_env}",
    )
    _require("that warning is expected" in output, "the standalone warning should be explained")
    _require(
        "SESSION_COOKIE_SECURE=0" in output,
        "webui mode should say what to set if the sign-in cookie does not stick",
    )

    second = shell_sandbox.launch()
    _require(
        _call_args(second) == [_start_args("127.0.0.1")],
        f"second launch should reuse the build: {second}",
    )


def test_shell_launcher_rebuilds_when_api_origin_changes(shell_sandbox) -> None:
    """A new backend port invalidates the build, since the origin is baked in."""
    shell_sandbox.launch()

    moved = shell_sandbox.launch(TLDW_API_PORT="8028")
    _require(
        _call_args(moved)[0] == ["run", "build:prod"]
        and moved[0]["env"]["TLDW_INTERNAL_API_ORIGIN"] == "http://127.0.0.1:8028",
        f"changing the API port should rebuild against the new origin: {moved}",
    )


def test_shell_launcher_rebuilds_only_when_apps_tree_changes(shell_sandbox) -> None:
    """Backend-only commits reuse the WebUI build; commits under apps/ rebuild it."""
    shell_sandbox.git("init", "-q")
    shell_sandbox.commit_file("apps/tldw-frontend/page.txt", "one")
    shell_sandbox.launch()

    shell_sandbox.commit_file("backend.txt", "server change")
    backend_only = shell_sandbox.launch()
    _require(
        _call_args(backend_only) == [_start_args("127.0.0.1")],
        f"a backend-only commit should not rebuild the WebUI: {backend_only}",
    )

    shell_sandbox.commit_file("apps/tldw-frontend/page.txt", "two")
    frontend = shell_sandbox.launch()
    _require(
        _call_args(frontend)[0] == ["run", "build:prod"],
        f"a commit under apps/ should rebuild the WebUI: {frontend}",
    )


def test_shell_launcher_failed_build_stops_without_starting(shell_sandbox) -> None:
    """A failed build exits non-zero with next steps, starts nothing, leaves no key."""
    shell_sandbox.launch()

    code, output = shell_sandbox.run(TLDW_API_PORT="8028", FAKE_BUILD_FAILS="1")

    _require(code != 0, f"a failed build should fail the launch: {output}")
    _require(
        _call_args(shell_sandbox.bun_calls()) == [["run", "build:prod"]],
        "nothing should start after a failed build",
    )
    _require(not shell_sandbox.key_file.exists(), "a failed build must not leave a build key")
    for hint in ("bun install", "memory", "TLDW_WEBUI_DEV=1"):
        _require(hint in output, f"the failure message should suggest {hint}: {output}")


def test_shell_launcher_dev_server_is_opt_in(shell_sandbox) -> None:
    """TLDW_WEBUI_DEV=1 keeps the old hot-reloading next dev path, without a build."""
    calls = shell_sandbox.launch(TLDW_WEBUI_DEV="1")

    _require(
        _call_args(calls) == [["run", "dev", "--", "-p", "8097"]],
        f"dev opt-in should run next dev only: {calls}",
    )
    env = calls[0]["env"]
    _require(
        env["NEXT_PUBLIC_API_URL"] == shell_sandbox.origin,
        "the dev server should keep pointing the browser at the API directly",
    )
    _require(
        env["SINGLE_USER_API_KEY"] is None and env["TLDW_WEBUI_EXPOSE_RUNTIME_AUTH"] is None,
        f"the dev server gets no key and no sign-in flag: {env}",
    )


def test_shell_launcher_honours_next_public_api_url_override(shell_sandbox) -> None:
    """A user-set NEXT_PUBLIC_API_URL keeps advanced networking in a production build."""
    calls = shell_sandbox.launch(NEXT_PUBLIC_API_URL="http://192.0.2.10:8000")

    _require(
        _call_args(calls) == [["run", "build:dev"], _start_args("0.0.0.0")],
        f"an API URL override should build the advanced profile: {calls}",
    )
    start_env = calls[1]["env"]
    _require(
        start_env["NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE"] == "advanced"
        and start_env["NEXT_PUBLIC_API_URL"] == "http://192.0.2.10:8000",
        f"the browser should call the overridden API URL directly: {start_env}",
    )
    _require(
        start_env["SINGLE_USER_API_KEY"] is None and start_env["TLDW_WEBUI_EXPOSE_RUNTIME_AUTH"] is None,
        f"advanced networking has no runtime auth bridge, so no key or flag: {start_env}",
    )


@pytest.mark.parametrize("env_file", FRONTEND_ENV_FILES)
def test_shell_launcher_reads_api_url_override_from_frontend_env_files(
    shell_sandbox, env_file: str
) -> None:
    """NEXT_PUBLIC_API_URL in the WebUI's .env files is an override too (quickstart refuses it)."""
    (shell_sandbox.webui_dir / env_file).write_text(
        'NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"\n', encoding="utf-8"
    )

    code, output = shell_sandbox.run()
    calls = shell_sandbox.bun_calls()

    _require(code == 0, f"launcher failed: {output}")
    _require(
        _call_args(calls) == [["run", "build:dev"], _start_args("0.0.0.0")]
        and calls[0]["env"]["NEXT_PUBLIC_API_URL"] == "http://127.0.0.1:8000",
        f"{env_file} should switch the build to advanced networking: {calls}",
    )
    _require(env_file in output, f"the launcher should name {env_file}: {output}")


def test_shell_launcher_settings_prefer_shell_then_env_file(shell_sandbox) -> None:
    """Runtime sign-in settings come from the shell first, then the env file."""
    shell_sandbox.write_env_file(
        "AUTH_MODE=multi_user\nSINGLE_USER_API_KEY=file-key-1234567890abcdef\n"
        "TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=0\nSINGLE_USER_SESSION_COOKIE_NAME=file_cookie\n"
    )

    from_file = shell_sandbox.launch()[-1]["env"]
    _require(
        from_file["AUTH_MODE"] == "multi_user"
        and from_file["SINGLE_USER_API_KEY"] == "file-key-1234567890abcdef"
        and from_file["TLDW_WEBUI_EXPOSE_RUNTIME_AUTH"] == "0"
        and from_file["SINGLE_USER_SESSION_COOKIE_NAME"] == "file_cookie",
        f"env file values should reach the WebUI server: {from_file}",
    )

    from_shell = shell_sandbox.launch(
        AUTH_MODE="single_user",
        SINGLE_USER_API_KEY=TEST_API_KEY,
        TLDW_WEBUI_EXPOSE_RUNTIME_AUTH="1",
        SINGLE_USER_SESSION_COOKIE_NAME="shell_cookie",
    )[-1]["env"]
    _require(
        from_shell["AUTH_MODE"] == "single_user"
        and from_shell["SINGLE_USER_API_KEY"] == TEST_API_KEY
        and from_shell["TLDW_WEBUI_EXPOSE_RUNTIME_AUTH"] == "1"
        and from_shell["SINGLE_USER_SESSION_COOKIE_NAME"] == "shell_cookie",
        f"shell values should win over the env file: {from_shell}",
    )


@pytest.mark.parametrize("where", ["shell", "env file"])
def test_shell_launcher_sign_in_opt_out(shell_sandbox, where: str) -> None:
    """TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=0 in the shell or env file turns sign-in off."""
    extra = {}
    if where == "shell":
        extra["TLDW_WEBUI_EXPOSE_RUNTIME_AUTH"] = "0"
    else:
        shell_sandbox.write_env_file(
            f"AUTH_MODE=single_user\nSINGLE_USER_API_KEY={TEST_API_KEY}\nTLDW_WEBUI_EXPOSE_RUNTIME_AUTH=0\n"
        )

    code, output = shell_sandbox.run(**extra)

    _require(code == 0, f"launcher failed: {output}")
    _require(
        shell_sandbox.bun_calls()[-1]["env"]["TLDW_WEBUI_EXPOSE_RUNTIME_AUTH"] == "0",
        f"the opt-out from the {where} should reach the WebUI server",
    )
    _require("Automatic sign-in is off" in output, f"the opt-out should be reported: {output}")


def test_shell_launcher_explains_missing_sign_in_without_venv(shell_sandbox) -> None:
    """Without a venv to read the env file, the launcher says sign-in is unavailable."""
    code, output = shell_sandbox.run(TLDW_VENV_DIR=str(shell_sandbox.root / "missing-venv"))

    _require(code == 0, f"launcher failed: {output}")
    _require(
        "Automatic sign-in is unavailable" in output and "missing-venv" in output,
        f"the launcher should explain why sign-in is unavailable: {output}",
    )
    _require(
        shell_sandbox.bun_calls()[-1]["env"]["SINGLE_USER_API_KEY"] is None,
        "no key should be invented",
    )


def test_shell_launcher_all_mode_relaxes_cookie_security_only_on_loopback(shell_sandbox) -> None:
    """All mode mirrors the Docker overlay on loopback and never relaxes it elsewhere."""
    shell_sandbox.launch("all")
    _require(
        shell_sandbox.api_envs() == [{"SESSION_COOKIE_SECURE": "0", "CSRF_ENABLED": "1"}],
        f"loopback HTTP should get non-Secure cookies with CSRF on: {shell_sandbox.api_envs()}",
    )

    calls = shell_sandbox.launch("all", TLDW_HOST="0.0.0.0")
    _require(
        shell_sandbox.api_envs() == [{"SESSION_COOKIE_SECURE": None, "CSRF_ENABLED": None}],
        f"a non-loopback API host must keep the backend cookie defaults: {shell_sandbox.api_envs()}",
    )
    _require(
        calls[-1]["args"] == _start_args("127.0.0.1")
        and calls[-1]["env"]["TLDW_INTERNAL_API_ORIGIN"] == shell_sandbox.origin,
        f"with TLDW_HOST=0.0.0.0 the WebUI still binds and proxies on 127.0.0.1: {calls}",
    )


@pytest.mark.parametrize("where", ["shell", "env file"])
def test_shell_launcher_all_mode_respects_explicit_cookie_security(shell_sandbox, where: str) -> None:
    """An explicit SESSION_COOKIE_SECURE in the shell or env file is kept."""
    extra = {}
    if where == "shell":
        extra["SESSION_COOKIE_SECURE"] = "1"
    else:
        shell_sandbox.write_env_file(f"AUTH_MODE=single_user\nSINGLE_USER_API_KEY={TEST_API_KEY}\nSESSION_COOKIE_SECURE=1\n")

    shell_sandbox.launch("all", **extra)

    _require(
        shell_sandbox.api_envs()[0]["SESSION_COOKIE_SECURE"] == "1",
        f"the {where} value should win: {shell_sandbox.api_envs()}",
    )


@pytest.mark.parametrize(
    ("api_mode", "message"),
    [("crash", "The API exited"), ("unhealthy", "did not answer")],
)
def test_shell_launcher_all_mode_waits_for_api_health(shell_sandbox, api_mode: str, message: str) -> None:
    """All mode waits for /health and fails clearly, before building, if it never comes."""
    code, output = shell_sandbox.run("all", FAKE_API_MODE=api_mode, TLDW_API_START_DELAY="2")

    _require(code != 0, f"an API that never comes up should fail the launch: {output}")
    _require(message in output, f"the launcher should say '{message}': {output}")
    _require(shell_sandbox.bun_calls() == [], "the WebUI should not build or start")


@pytest.mark.parametrize(
    "path",
    [
        "quick-launch.sh",
        "quick-launch.ps1",
    ],
)
def test_quick_launch_scripts_share_production_webui_contract(path: str) -> None:
    """Both launchers keep the same build, sign-in, cookie and health-wait logic."""
    text = _read(path)

    for expected in (
        "TLDW_WEBUI_DEV",
        "build:prod",
        "build:dev",
        '"start"' if path.endswith(".ps1") else "run start",
        "quick-launch-build-key",
        "BUILD_ID",
        "rev-parse --verify --quiet HEAD:apps",
        "TLDW_INTERNAL_API_ORIGIN",
        "TLDW_WEBUI_EXPOSE_RUNTIME_AUTH",
        "SINGLE_USER_SESSION_COOKIE_NAME",
        "dotenv_values",
        *FRONTEND_ENV_FILES,
        "SESSION_COOKIE_SECURE",
        "CSRF_ENABLED",
        "/health",
        "first launch takes a few minutes",
        "The WebUI build failed",
        "that warning is expected",
        "Automatic sign-in is off",
        "Automatic sign-in is unavailable",
        "programs on this computer can sign in",
        "delete apps/tldw-frontend/.next",
    ):
        _require(expected in text, f"{path} should include {expected}")


def test_powershell_launcher_restores_environment_after_webui() -> None:
    """quick-launch.ps1 restores every $env: value it sets, so later launches start clean.

    Contract check only: there is no pwsh to run a .ps1 sandbox where these tests are written.
    """
    text = _read("quick-launch.ps1")

    for expected in (
        "Save-QuickLaunchEnv",
        "Restore-QuickLaunchEnv",
        "[Environment]::SetEnvironmentVariable",
        "finally",
    ):
        _require(expected in text, f"quick-launch.ps1 should include {expected}")
