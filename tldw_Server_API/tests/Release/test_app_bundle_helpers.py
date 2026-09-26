"""Downloaded Docker helpers must preserve trust and instance identity."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
BUNDLE = REPO / "Dockerfiles" / "app-bundle"


@pytest.fixture
def installed_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    directory = tmp_path / "downloaded"
    directory.mkdir()
    for name in ("start.sh", "stop.sh", "status.sh", "compose.yaml"):
        content = (BUNDLE / name).read_text()
        content = content.replace(
            "__CONTROL_IMAGE_DIGEST__", "registry.invalid/tldw/control@sha256:" + "a" * 64
        ).replace("__TRUSTED_KEY_ID__", "test-key")
        output = directory / name
        output.write_text(content)
        output.chmod(0o755)
    (directory / "manifest.json").write_text("{}")
    (directory / "manifest.sig").write_bytes(b"signature")

    binary = tmp_path / "bin"
    binary.mkdir()
    fake_docker = binary / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['FAKE_DOCKER_LOG'], 'a') as log: log.write(json.dumps(args) + '\\n')\n"
        "if args and args[0] == 'info' and os.environ.get('FAKE_DOCKER_FAIL_INFO') == '1': sys.exit(1)\n"
        "if args[:2] == ['compose', 'version'] and os.environ.get('FAKE_DOCKER_FAIL_COMPOSE') == '1': sys.exit(1)\n"
        "if args and args[0] == 'compose' and 'up' in args and os.environ.get('FAKE_DOCKER_FAIL_UP') == '1': sys.exit(1)\n"
        "if args[:2] == ['info', '--format']: print('x86_64')\n"
        "if args[:2] == ['compose', 'version']: print('Docker Compose version v2')\n"
        "if args and args[0] == 'run':\n"
        "  if 'verify' in args and os.environ.get('FAKE_DOCKER_FAIL_VERIFY') == '1': sys.exit(1)\n"
        "  if 'init' in args:\n"
        "    state = pathlib.Path(os.environ['FAKE_STATE_DIR']) / 'instance'\n"
        "    state.mkdir(exist_ok=True)\n"
        "    (state / 'config.env').write_text('TLDW_PROJECT_ID=tldw_test\\nTLDW_PUBLIC_PORT=18080\\n')\n"
        "if args and args[0] == 'compose' and 'ps' in args: print('')\n"
    )
    fake_docker.chmod(0o755)
    fake_lsof = binary / "lsof"
    fake_lsof.write_text(
        "#!/usr/bin/env python3\n" "import os, sys\n" "sys.exit(0 if os.environ.get('FAKE_PORT_BUSY') == '1' else 1)\n"
    )
    fake_lsof.chmod(0o755)
    return directory, binary, tmp_path / "state"


def run_helper(
    name: str,
    installed_bundle: tuple[Path, Path, Path],
    *,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    directory, binary, state = installed_bundle
    log = directory.parent / "docker-calls.jsonl"
    log.unlink(missing_ok=True)
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "TLDW_APP_STATE_DIR": str(state),
        "TLDW_APP_PUBLIC_PORT": "18080",
        "TLDW_APP_NO_BROWSER": "1",
        "FAKE_DOCKER_LOG": str(log),
        "FAKE_STATE_DIR": str(state),
        **(extra_env or {}),
    }
    result = subprocess.run(
        [str(directory / name)],
        cwd=directory.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


def test_first_start_verifies_before_init_and_compose(
    installed_bundle: tuple[Path, Path, Path],
) -> None:
    result, calls = run_helper("start.sh", installed_bundle)

    assert result.returncode == 0, result.stderr
    verify = next(index for index, call in enumerate(calls) if "verify" in call)
    initialize = next(index for index, call in enumerate(calls) if "init" in call)
    pull = next(index for index, call in enumerate(calls) if "pull" in call)
    up = next(index for index, call in enumerate(calls) if "up" in call)
    assert verify < initialize < pull < up
    assert "--network" in calls[verify] and "none" in calls[verify]
    assert "--no-build" in calls[up]
    assert "--wait" in calls[up] and "600" in calls[up]
    assert "tldw_test" in calls[up]
    assert "registry.invalid/tldw/control@sha256:" + "a" * 64 in calls[verify]


def test_repeat_start_reuses_state_and_project(
    installed_bundle: tuple[Path, Path, Path],
) -> None:
    first, _ = run_helper("start.sh", installed_bundle)
    before = (installed_bundle[2] / "instance" / "config.env").read_bytes()
    second, calls = run_helper("start.sh", installed_bundle)

    assert first.returncode == second.returncode == 0
    assert (installed_bundle[2] / "instance" / "config.env").read_bytes() == before
    assert any("verify" in call for call in calls)
    assert any("init" in call for call in calls)
    assert any("tldw_test" in call and "up" in call for call in calls)


def test_bad_signature_stops_before_compose_or_instance_creation(
    installed_bundle: tuple[Path, Path, Path],
) -> None:
    result, calls = run_helper("start.sh", installed_bundle, extra_env={"FAKE_DOCKER_FAIL_VERIFY": "1"})

    assert result.returncode != 0
    assert not (installed_bundle[2] / "instance").exists()
    assert not any("up" in call or "pull" in call for call in calls)


def test_occupied_port_stops_before_compose_up(
    installed_bundle: tuple[Path, Path, Path],
) -> None:
    result, calls = run_helper("start.sh", installed_bundle, extra_env={"FAKE_PORT_BUSY": "1"})

    assert result.returncode != 0
    assert not any("up" in call for call in calls)


def test_failed_start_stops_partial_services_but_keeps_instance(
    installed_bundle: tuple[Path, Path, Path],
) -> None:
    result, calls = run_helper("start.sh", installed_bundle, extra_env={"FAKE_DOCKER_FAIL_UP": "1"})

    assert result.returncode != 0
    assert any("up" in call for call in calls)
    assert any("down" in call for call in calls)
    assert (installed_bundle[2] / "instance" / "config.env").exists()


def test_stop_uses_persisted_identity_from_other_directory(
    installed_bundle: tuple[Path, Path, Path],
) -> None:
    start, _ = run_helper("start.sh", installed_bundle)
    stopped, calls = run_helper("stop.sh", installed_bundle)

    assert start.returncode == stopped.returncode == 0
    assert any("down" in call and "tldw_test" in call for call in calls)
    assert not any("-v" in call or "--volumes" in call for call in calls)


@pytest.mark.parametrize("failure", ["FAKE_DOCKER_FAIL_INFO", "FAKE_DOCKER_FAIL_COMPOSE"])
def test_unavailable_docker_or_compose_stops_before_verification(
    failure: str,
    installed_bundle: tuple[Path, Path, Path],
) -> None:
    result, calls = run_helper("start.sh", installed_bundle, extra_env={failure: "1"})

    assert result.returncode != 0
    assert not any("run" in call or "up" in call for call in calls)


def test_status_reports_persisted_url_from_other_directory(
    installed_bundle: tuple[Path, Path, Path],
) -> None:
    start, _ = run_helper("start.sh", installed_bundle)
    status, calls = run_helper("status.sh", installed_bundle)

    assert start.returncode == status.returncode == 0
    assert "http://127.0.0.1:18080/" in status.stdout
    assert any("ps" in call and "tldw_test" in call for call in calls)


def test_compose_exposes_only_loopback_gateway() -> None:
    compose = yaml.safe_load((BUNDLE / "compose.yaml").read_text())
    services = compose["services"]

    assert set(services) == {"app", "webui", "gateway"}
    assert "ports" not in services["app"]
    assert "ports" not in services["webui"]
    assert services["gateway"]["ports"] == ["127.0.0.1:${TLDW_PUBLIC_PORT}:8080"]
    assert not any("docker.sock" in str(service) for service in services.values())
    assert compose["networks"]["private"] is None or not compose["networks"]["private"].get("internal")


def test_compose_keeps_webui_auth_mode_matched_to_backend() -> None:
    services = yaml.safe_load((BUNDLE / "compose.yaml").read_text())["services"]

    assert services["webui"]["environment"].get("AUTH_MODE") == services["app"]["environment"]["AUTH_MODE"]


def test_compose_shares_instance_session_cookie_name_between_backend_and_webui() -> None:
    services = yaml.safe_load((BUNDLE / "compose.yaml").read_text())["services"]

    assert (
        services["app"]["environment"].get("SINGLE_USER_SESSION_COOKIE_NAME")
        == services["webui"]["environment"]["SINGLE_USER_SESSION_COOKIE_NAME"]
    )


def test_compose_uses_cookies_on_its_http_loopback_gateway() -> None:
    services = yaml.safe_load((BUNDLE / "compose.yaml").read_text())["services"]

    assert services["app"]["environment"].get("SESSION_COOKIE_SECURE") == "0"


def test_compose_persists_enabled_mcp_audit_in_existing_database_volume() -> None:
    app = yaml.safe_load((BUNDLE / "compose.yaml").read_text())["services"]["app"]

    assert app["environment"].get("MCP_AUDIT_LOG_FILE") == "/app/Databases/mcp-audit.log"
    assert "backend_data:/app/Databases" in app["volumes"]
    assert app["environment"].get("MCP_AUDIT_ENABLED", "true") == "true"
    assert "user" not in app


def test_compose_binds_managed_setup_and_cookie_origins_to_persisted_public_port() -> None:
    services = yaml.safe_load((BUNDLE / "compose.yaml").read_text())["services"]
    backend = services["app"]["environment"]
    assert backend.get("TLDW_MANAGED_GATEWAY") == "1"
    assert backend.get("TLDW_GATEWAY_HOP_SECRET") == services["gateway"]["environment"]["TLDW_GATEWAY_HOP_SECRET"]
    assert backend.get("TLDW_MANAGED_PUBLIC_ORIGIN") == "http://127.0.0.1:${TLDW_PUBLIC_PORT:?Public port is required}"
    assert backend.get("ALLOWED_ORIGINS") == backend["TLDW_MANAGED_PUBLIC_ORIGIN"]
    assert "TLDW_SETUP_ALLOW_REMOTE" not in backend
    assert "FORWARDED_ALLOW_IPS" not in backend
