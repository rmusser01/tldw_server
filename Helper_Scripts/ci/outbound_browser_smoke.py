"""Run focused production browser probes inside the prepared outbound container."""

from __future__ import annotations

import argparse
import configparser
import http.client
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import subprocess  # nosec B404
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

API_PORT = 18081
WEB_PORT = 18082
EXTENSION_API_PORT = 18083
EXTENSION_PAGE_PORT = 18084
UNAPPROVED_PORT = 18085
COOKIE_NAME = "tldw_single_user_session"
CHROMIUM_CACHE = Path("/opt/outbound-chromium")
SCENARIOS = {
    "shipped-setup": ("shipped setup saves a local setting", "tldw-frontend", API_PORT),
    "production-webui": ("production WebUI renders search results without external assets", "tldw-frontend", WEB_PORT),
    "extension-worker": ("packaged worker uses the configured fixture", "extension", EXTENSION_API_PORT),
    "extension-content-script": ("content script activates on a local page", "extension", EXTENSION_PAGE_PORT),
}
COVERAGE = {
    "shipped-setup": [
        "linux-container",
        "socket-connect-sendto",
        "playwright-context-events",
        "chromium-platform-route-query",
        "shipped-setup-page",
        "python-network-guard",
    ],
    "production-webui": [
        "linux-container",
        "socket-connect-sendto",
        "playwright-context-events",
        "chromium-platform-route-query",
        "webui-browser-and-next",
        "python-network-guard",
    ],
    "extension-worker": [
        "linux-container",
        "socket-connect-sendto",
        "playwright-context-events",
        "chromium-platform-route-query",
        "chrome-mv3-worker",
    ],
    "extension-content-script": [
        "linux-container",
        "socket-connect-sendto",
        "playwright-context-events",
        "chromium-platform-route-query",
        "chrome-content-script",
    ],
}
VIOLATION_CODES = frozenset(
    {
        "backend_scenario_unavailable",
        "browser_isolation_unavailable",
        "browser_observer_incomplete",
        "browser_observer_missing_tripwire",
        "browser_observer_missing_worker",
        "browser_scenario_unavailable",
        "browser_trace_preflight_failed",
        "extension_manifest_policy_failed",
        "next_scenario_unavailable",
        "playwright_evidence_missing",
        "playwright_failed",
        "setup_setting_not_persisted",
        "trace_incomplete",
        "trace_unclassified",
        "unapproved_browser_request",
        "unapproved_socket",
        "unapproved_unix_socket",
    }
)
_FAMILY = re.compile(r"\bsa_family=(AF_[A-Z0-9]+)\b")
_PORT = re.compile(r"\bsin6?_port=htons\((\d+)\)")
_ADDRESS = re.compile(r'(?:inet_addr\(|inet_pton\(AF_INET6, )"([^"]+)"')


class BrowserSmokeError(RuntimeError):
    """A controlled browser fixture or the tracing boundary did not complete."""


def require_extension_host_permission(inventory: Path, manifest: Path) -> str:
    """Check the pristine MV3 host permissions against the reviewed capability."""
    from Helper_Scripts.ci.outbound_inventory import Inventory

    try:
        entry = Inventory.load(inventory).capabilities["extension.github-api"]
        [origin] = entry.origins
        permission = f"{origin}/*"
        value = json.loads(manifest.read_text(encoding="utf-8"))
        matches = (
            entry.kind == "builtin" and entry.component == "extension" and value["host_permissions"] == [permission]
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise BrowserSmokeError("pristine extension permission policy is unavailable") from exc
    if not matches:
        raise BrowserSmokeError("pristine extension permission differs from the reviewed inventory")
    return permission


def check_attempts(
    prefix: Path,
    fixture_ports: set[int],
    *,
    required_port: int | None = None,
    chromium: bool = False,
) -> set[str]:
    """Parse only addressed connect/sendto calls and return stable reason codes."""
    paths = [path for path in prefix.parent.glob(f"{prefix.name}.*") if path.suffix[1:].isdigit()]
    errors: set[str] = set()
    observed: set[int] = set()
    if not paths:
        return {"trace_incomplete"}
    for path in paths:
        finalized = False
        with path.open(encoding="utf-8", errors="replace") as trace:
            for line in trace:
                if line.startswith(("+++ exited ", "+++ killed ")):
                    finalized = True
                if not line.startswith(("connect(", "sendto(")):
                    continue
                family_match = _FAMILY.search(line)
                if family_match is None:
                    # sendto with a null sockaddr uses an already-connected socket.
                    if not (line.startswith("sendto(") and ", NULL, 0)" in line):
                        errors.add("trace_unclassified")
                    continue
                family = family_match.group(1)
                if family in {"AF_NETLINK", "AF_UNSPEC"}:
                    continue
                if family in {"AF_UNIX", "AF_LOCAL"}:
                    errors.add("unapproved_unix_socket")
                    continue
                address_match, port_match = _ADDRESS.search(line), _PORT.search(line)
                if family not in {"AF_INET", "AF_INET6"} or not address_match or not port_match:
                    errors.add("trace_unclassified")
                    continue
                try:
                    address = ipaddress.ip_address(address_match.group(1))
                except ValueError:
                    errors.add("trace_unclassified")
                    continue
                if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
                    address = address.ipv4_mapped
                port = int(port_match.group(1))
                # Chromium uses this failed UDP connect to ask the OS whether IPv6
                # has a route. The dedicated executable wrapper limits attribution to
                # Chromium. A datagram send or successful connect stays a violation.
                if (
                    chromium
                    and family == "AF_INET6"
                    and str(address) == "2001:4860:4860::8888"
                    and port == 443
                    and re.match(r"^connect\(\d+<UDPv6:\[\d+\]>,", line)
                    and re.search(r"= -1 ENETUNREACH\b", line)
                ):
                    continue
                if str(address) == "127.0.0.1" and port in fixture_ports:
                    if re.search(r"= (?:0|[1-9]\d*|-1 EINPROGRESS)\b", line):
                        observed.add(port)
                else:
                    errors.add("unapproved_socket")
        if not finalized:
            errors.add("trace_incomplete")
    if required_port is not None and required_port not in observed:
        errors.add("trace_incomplete")
    return errors


def _passed_test(report: Path, expected_title: str) -> bool:
    def collect_specs(suites: list[dict]) -> Iterator[dict]:
        for suite in suites:
            yield from suite.get("specs", [])
            yield from collect_specs(suite.get("suites", []))

    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("errors"):
            return False
        [spec] = collect_specs(payload["suites"])
        [test] = spec["tests"]
        [result] = test["results"]
        return spec["title"] == expected_title and test["status"] == "expected" and result["status"] == "passed"
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return False


def validate_browser_result(
    report: Path,
    observer: Path,
    expected_title: str,
    *,
    require_worker: bool = False,
    require_tripwire: bool = False,
) -> set[str]:
    """Require positive execution and observer results without decoding raw diagnostics."""
    errors = set() if _passed_test(report, expected_title) else {"playwright_evidence_missing"}
    try:
        evidence = json.loads(observer.read_text(encoding="utf-8"))
        allowed = evidence["allowedRequests"]
        workers = evidence["workerRequests"]
        unexpected = evidence["unexpectedRequests"]
        tripwire = evidence["tripwirePassed"]
    except (KeyError, OSError, TypeError, ValueError):
        return errors | {"browser_observer_incomplete"}
    if any(type(value) is not int or value < 0 for value in (allowed, workers, unexpected)):
        return errors | {"browser_observer_incomplete"}
    if type(tripwire) is not bool or allowed == 0 or workers > allowed:
        errors.add("browser_observer_incomplete")
    if unexpected:
        errors.add("unapproved_browser_request")
    if require_worker and not workers:
        errors.add("browser_observer_missing_worker")
    if require_tripwire and tripwire is not True:
        errors.add("browser_observer_missing_tripwire")
    return errors


def _trace_command(prefix: Path, *argv: str) -> list[str]:
    tracer = shutil.which("strace")
    if not tracer:
        raise BrowserSmokeError("strace is not installed in the browser image")
    return [tracer, "-ff", "--seccomp-bpf", "-e", "trace=connect,sendto", "-s", "0", "-o", str(prefix), "--", *argv]


def _terminate_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def _run_child(
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    *,
    stdout: int = subprocess.DEVNULL,
    timeout: float = 100,
) -> int:
    # The caller supplies fixed tool arguments and fixture paths.
    with subprocess.Popen(  # nosec B603
        argv,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    ) as process:
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_group(process)
            raise BrowserSmokeError("browser fixture subprocess exceeded its deadline") from None


def _calibrate_trace(root: Path, scratch: Path, env: dict[str, str]) -> None:
    prefix = scratch / "preflight-trace"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        code = """import socket, sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=2):
    pass
try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[2])), timeout=2):
        raise RuntimeError("unexpected second listener")
except ConnectionRefusedError:
    pass
"""
        status = _run_child(
            _trace_command(prefix, sys.executable, "-c", code, str(port), str(UNAPPROVED_PORT)),
            root,
            env,
            timeout=10,
        )
    reasons = check_attempts(prefix, {port}, required_port=port)
    if status or reasons != {"unapproved_socket"}:
        raise BrowserSmokeError("browser syscall recorder did not observe the negative control")


def _run_browser_test(root: Path, scratch: Path, env: dict[str, str], scenario: str) -> set[str]:
    title, app, port = SCENARIOS[scenario]
    suite_dir = scratch / scenario
    suite_dir.mkdir()
    report = suite_dir / "playwright.json"
    observer = suite_dir / "observer.json"
    trace = suite_dir / "browser"
    wrapper = suite_dir / "trace-chromium"
    wrapper.write_text(
        '#!/bin/sh\nexec "$OUTBOUND_STRACE" -ff --seccomp-bpf -yy -e trace=connect,sendto -s 0 '
        '-o "$OUTBOUND_CHROMIUM_TRACE" -- "$OUTBOUND_CHROMIUM_BINARY" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    bunx = shutil.which("bunx")
    strace = shutil.which("strace")
    if not bunx or not strace:
        raise BrowserSmokeError("browser test tools are missing from the prepared image")
    with report.open("wb") as output:
        status = _run_child(
            [
                bunx,
                "--no-install",
                "playwright",
                "test",
                "--config",
                "playwright.outbound.config.ts",
                "--workers=1",
                "--retries=0",
                "--reporter=json",
                "--output",
                str(suite_dir / "assets"),
                "--grep",
                f"{re.escape(title)}$",
            ],
            root / "apps" / app,
            {
                **env,
                "OUTBOUND_BROWSER_OBSERVER_FILE": str(observer),
                "OUTBOUND_CHROMIUM_TRACE": str(trace),
                "OUTBOUND_CHROMIUM_WRAPPER": str(wrapper),
                "OUTBOUND_STRACE": str(strace),
            },
            stdout=output.fileno(),
        )
    reasons = check_attempts(trace, {port}, required_port=port, chromium=True)
    reasons.update(
        validate_browser_result(
            report,
            observer,
            title,
            require_worker=scenario == "extension-worker",
            require_tripwire=scenario == "production-webui",
        )
    )
    node = shutil.which("node")
    if not node:
        raise BrowserSmokeError("Node is not installed in the browser image")
    no_skips = root / "apps/tldw-frontend/scripts/assert-playwright-no-skips.mjs"
    validation = subprocess.run(  # nosec B603
        [node, str(no_skips), str(report)],
        cwd=root,
        env=env,
        timeout=10,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if validation.returncode:
        reasons.add("playwright_evidence_missing")
    if status:
        reasons.add("playwright_failed")
    return reasons


def _wait_next(port: int, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 35
    while time.monotonic() < deadline and process.poll() is None:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request("GET", "/login")
            response = connection.getresponse()
            if response.status < 400:
                return
        except (OSError, http.client.HTTPException):
            pass
        finally:
            connection.close()
        time.sleep(0.2)
    raise BrowserSmokeError("standalone Next process did not become ready")


@contextmanager
def _next_server(root: Path, scratch: Path, env: dict[str, str], api_origin: str, key: str) -> Iterator[None]:
    node = shutil.which("node")
    server = root / "apps/tldw-frontend/.next/standalone/apps/tldw-frontend/server.js"
    if not node or not server.is_file():
        raise BrowserSmokeError("standalone Next artifacts are missing")
    pid_path = scratch / "next.pid"
    argv = _trace_command(
        scratch / "next",
        "/bin/sh",
        "-c",
        'printf "%s\\n" "$$" > "$1"; shift; exec "$@"',
        "outbound-next",
        str(pid_path),
        node,
        str(server),
    )
    env = {
        **env,
        "AUTH_MODE": "single_user",
        "HOSTNAME": "127.0.0.1",
        "NEXT_PUBLIC_API_URL": "",
        "NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE": "quickstart",
        "NEXT_TELEMETRY_DISABLED": "1",
        "PORT": str(WEB_PORT),
        "SINGLE_USER_API_KEY": key,
        "SINGLE_USER_SESSION_COOKIE_NAME": COOKIE_NAME,
        "TLDW_INTERNAL_API_ORIGIN": api_origin,
        "TLDW_WEBUI_EXPOSE_RUNTIME_AUTH": "1",
    }
    # The shell command is fixed and receives paths as positional arguments.
    with subprocess.Popen(  # nosec B603
        argv,
        cwd=server.parent,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    ) as tracer:
        try:
            _wait_next(WEB_PORT, tracer)
            yield
        finally:
            try:
                pid = int(pid_path.read_text(encoding="ascii"))
                if tracer.poll() is not None:
                    raise BrowserSmokeError("standalone Next exited before the shutdown request")
                os.kill(pid, signal.SIGTERM)
                if tracer.wait(timeout=15) != 0:
                    raise BrowserSmokeError("standalone Next did not exit cleanly")
            except (OSError, ValueError, subprocess.TimeoutExpired):
                _terminate_group(tracer)
                raise BrowserSmokeError("standalone Next shutdown could not be traced") from None


def _private_env(scratch: Path) -> dict[str, str]:
    home = scratch / "home"
    home.mkdir()
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "TMPDIR": str(scratch),
        "LANG": "C.UTF-8",
        "PLAYWRIGHT_BROWSERS_PATH": str(CHROMIUM_CACHE),
        # Disable browser desktop-bus discovery in the isolated test image.
        "DBUS_SESSION_BUS_ADDRESS": "disabled:",
        "DBUS_SYSTEM_BUS_ADDRESS": "disabled:",
    }


def _write_results(report_dir: Path, failures: dict[str, set[str]]) -> int:
    report = {
        "scenarios": [
            {
                "scenario": scenario,
                "status": "fail" if failures[scenario] else "pass",
                "coverage": [] if failures[scenario] else COVERAGE[scenario],
                "violations": [{"component": scenario, "reason": reason} for reason in sorted(failures[scenario])],
            }
            for scenario in SCENARIOS
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "result.json").write_text(json.dumps(report), encoding="utf-8")
    return int(any(failures.values()))


def run_browser_smoke(root: Path, report_dir: Path) -> int:
    """Run all four prepared browser scenarios and write one redacted report."""
    from Helper_Scripts.ci.outbound_isolation import IsolationError, assert_isolated_runtime

    failures: dict[str, set[str]] = {scenario: set() for scenario in SCENARIOS}
    try:
        assert_isolated_runtime()
        if not CHROMIUM_CACHE.is_dir():
            raise IsolationError("Chromium cache is unavailable")
    except (IsolationError, OSError):
        return _write_results(report_dir, {scenario: {"browser_isolation_unavailable"} for scenario in SCENARIOS})

    from Helper_Scripts.ci.outbound_backend_smoke import LOCAL_PROVIDER_FIXTURE_PORTS, SmokeError, backend_fixture

    with tempfile.TemporaryDirectory(prefix="tldw-browser-") as directory:
        scratch = Path(directory)
        env = _private_env(scratch)
        try:
            _calibrate_trace(root, scratch, env)
        except (OSError, BrowserSmokeError, subprocess.SubprocessError):
            return _write_results(report_dir, {scenario: {"browser_trace_preflight_failed"} for scenario in SCENARIOS})

        def run(scenario: str, extra_env: dict[str, str] | None = None) -> None:
            try:
                failures[scenario].update(_run_browser_test(root, scratch, {**env, **(extra_env or {})}, scenario))
            except (OSError, BrowserSmokeError, subprocess.SubprocessError):
                failures[scenario].add("browser_scenario_unavailable")

        for scenario in ("shipped-setup", "production-webui"):
            backend_trace = scratch / f"{scenario}-backend"
            webui = scenario == "production-webui"
            try:
                with backend_fixture(
                    root,
                    scratch / f"{scenario}-state",
                    trace_prefix=backend_trace,
                    local_provider_fixtures=webui,
                ) as fixture:
                    if scenario == "shipped-setup":
                        run(scenario)
                        config = configparser.ConfigParser(interpolation=None)
                        try:
                            with fixture.config_path.open(encoding="utf-8") as config_file:
                                config.read_file(config_file)
                            if config.getint("Files", "inline_max_bytes") != 262145:
                                failures[scenario].add("setup_setting_not_persisted")
                        except (configparser.Error, OSError, ValueError):
                            failures[scenario].add("setup_setting_not_persisted")
                    else:
                        try:
                            with _next_server(root, scratch, env, fixture.origin, fixture.key):
                                run(
                                    scenario,
                                    {
                                        "OUTBOUND_SESSION_COOKIE": COOKIE_NAME,
                                        "OUTBOUND_SYNTHETIC_API_KEY": fixture.key,
                                    },
                                )
                        except (OSError, BrowserSmokeError, subprocess.SubprocessError):
                            failures[scenario].add("next_scenario_unavailable")
                        failures[scenario].update(check_attempts(scratch / "next", {API_PORT}, required_port=API_PORT))
            except (OSError, SmokeError, subprocess.SubprocessError):
                failures[scenario].add("backend_scenario_unavailable")
            provider_ports = set(LOCAL_PROVIDER_FIXTURE_PORTS) if webui else set()
            failures[scenario].update(check_attempts(backend_trace, provider_ports))

        try:
            permission = require_extension_host_permission(
                root / "Helper_Scripts/ci/outbound_inventory.json",
                root / "apps/extension/build/chrome-mv3/manifest.json",
            )
        except BrowserSmokeError:
            for scenario in ("extension-worker", "extension-content-script"):
                failures[scenario].add("extension_manifest_policy_failed")
        else:
            manifest_env = {"OUTBOUND_MANIFEST_HOST_PERMISSION": permission}
            run("extension-worker", manifest_env)
            run("extension-content-script", manifest_env)
    return _write_results(report_dir, failures)


def main(argv: list[str] | None = None) -> int:
    """Run the browser suite from the prepared container entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside-container", action="store_true", required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("/outbound-report"))
    args = parser.parse_args(argv)
    return run_browser_smoke(Path.cwd(), args.report_dir)


if __name__ == "__main__":
    raise SystemExit(main())
