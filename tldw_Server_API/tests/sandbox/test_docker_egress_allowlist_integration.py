from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.config import clear_config_cache
from tldw_Server_API.app.core.Sandbox import network_policy
from tldw_Server_API.app.main import app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.sandbox_real_docker,
    pytest.mark.skipif(
        os.environ.get("TLDW_SANDBOX_DOCKER_EGRESS_IT") != "1",
        reason="Enable TLDW_SANDBOX_DOCKER_EGRESS_IT=1 on a prepared Linux Docker/iptables host",
    ),
]
_IMAGE = "python:3.11-alpine"


@pytest.fixture
def reachable_destinations():
    """Two local HTTP servers, proven reachable from an unrestricted bridge.

    Temporary source/destination-scoped ACCEPT rules also reproduce existing
    terminating host rules; a policy appended after them cannot block traffic.
    The opt-in explicitly permits these temporary host firewall mutations.
    """
    if sys.platform != "linux" or not shutil.which("iptables") or not shutil.which("docker"):
        pytest.skip("requires Linux, Docker, and host iptables privileges")
    try:
        subprocess.check_output(["docker", "info"], timeout=10)
        subprocess.check_output(["docker", "image", "inspect", _IMAGE], timeout=5)
        subprocess.check_output(["iptables", "-S"], timeout=5)
    except (OSError, subprocess.SubprocessError):
        pytest.skip(f"requires running Docker, preloaded {_IMAGE}, and iptables privileges")

    name = "tldw-egress-it-" + uuid.uuid4().hex[:12]
    with contextlib.ExitStack() as cleanup:
        subprocess.check_call(["docker", "network", "create", name], timeout=10)
        cleanup.callback(subprocess.run, ["docker", "network", "rm", name], check=False, timeout=10)
        destinations = []
        for number in range(2):
            server = f"{name}-{number}"
            subprocess.check_call(
                ["docker", "run", "--rm", "-d", "--name", server, "--network", name,
                 _IMAGE, "python", "-m", "http.server", "8080"], timeout=10,
            )
            cleanup.callback(subprocess.run, ["docker", "rm", "-f", server], check=False, timeout=10)
            inspect = subprocess.check_output(["docker", "inspect", server], text=True, timeout=5)
            address = json.loads(inspect)[0]["NetworkSettings"]["Networks"][name]["IPAddress"]
            destinations.append(address)
            for direction in ("-s", "-d"):
                spec = ["FORWARD", direction, address, "-m", "comment", "--comment", name, "-j", "ACCEPT"]
                subprocess.check_call(["iptables", "-I", "FORWARD", "1", *spec[1:]], timeout=5)
                cleanup.callback(subprocess.run, ["iptables", "-D", *spec], check=False, timeout=5)

        # Both servers must respond over another bridge before testing denial.
        control = (
            "import urllib.request; "
            + "; ".join(f"assert urllib.request.urlopen('http://{address}:8080', timeout=2).status == 200" for address in destinations)
        )
        for attempt in range(10):
            result = subprocess.run(
                ["docker", "run", "--rm", "--network", "bridge", _IMAGE, "python", "-c", control],
                capture_output=True, text=True, timeout=15, check=False,
            )
            if result.returncode == 0:
                break
            if attempt == 9:
                pytest.fail(f"controlled destinations are not reachable: {result.stderr}")
            time.sleep(0.2)
        yield destinations


@pytest.mark.parametrize("fallback", [False, True])
def test_docker_egress_allowlist_allows_allowed_and_blocks_others(
    monkeypatch: pytest.MonkeyPatch, reachable_destinations, fallback: bool,
) -> None:
    allowed, blocked = reachable_destinations
    for key, value in {
        "TEST_MODE": "1",
        "SANDBOX_ENABLE_EXECUTION": "true",
        "SANDBOX_BACKGROUND_EXECUTION": "false",
        "TLDW_SANDBOX_DOCKER_FAKE_EXEC": "0",
        "SANDBOX_EGRESS_ENFORCEMENT": "true",
        "SANDBOX_EGRESS_GRANULAR_ENFORCEMENT": "true",
        "SANDBOX_EGRESS_ALLOWLIST": allowed + "/32",
    }.items():
        monkeypatch.setenv(key, value)
    clear_config_cache()
    if fallback:
        real_run = network_policy.subprocess.run

        def force_fallback(args, **kwargs):
            if args[0] == "iptables-restore":
                return SimpleNamespace(returncode=1)
            return real_run(args, **kwargs)

        monkeypatch.setattr(network_policy.subprocess, "run", force_fallback)

    with TestClient(app) as client:
        for address, expected_success in ((allowed, True), (blocked, False)):
            response = client.post(
                "/api/v1/sandbox/runs",
                json={
                    "spec_version": "1.0",
                    "runtime": "docker",
                    "base_image": _IMAGE,
                    "command": ["python", "-c", f"import urllib.request; assert urllib.request.urlopen('http://{address}:8080', timeout=2).status == 200"],
                    "timeout_sec": 30,
                    "network_policy": "allowlist",
                    "trust_level": "trusted",
                },
                timeout=60,
            )
            assert response.status_code == 200, response.text
            status = response.json()
            assert status.get("phase") in {"completed", "failed"}, status
            exit_code = status.get("exit_code")
            assert isinstance(exit_code, int), status
            assert (exit_code == 0) is expected_success, status
    clear_config_cache()
