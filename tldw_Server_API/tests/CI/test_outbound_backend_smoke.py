"""Exact fixture access and shutdown evidence for the outbound backend check."""

import json
import socket
import subprocess  # nosec B404
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tldw_Server_API.tests.helpers.offline_network import block_live_network

ENDPOINTS = frozenset({("127.0.0.1", 18081), ("::1", 18082)})
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(("configured", "expected"), [(None, "1"), ("0", "0")])
def test_app_configures_onnx_telemetry_before_native_imports(configured, expected):
    code = """
import os
import sys

if sys.argv[1] == "unset":
    os.environ.pop("ORT_DISABLE_TELEMETRY", None)
else:
    os.environ["ORT_DISABLE_TELEMETRY"] = sys.argv[1]
import tldw_Server_API.app
assert "onnxruntime" not in sys.modules
print(os.environ.get("ORT_DISABLE_TELEMETRY"))
"""
    child = subprocess.run(  # nosec B603
        [sys.executable, "-B", "-c", code, "unset" if configured is None else configured],
        cwd=ROOT, capture_output=True, text=True, timeout=8, check=True,
    )
    assert child.stdout.strip() == expected


def test_exact_numeric_and_static_names_never_hit_dns(monkeypatch):
    getaddrinfo = socket.getaddrinfo

    def numeric_only(host, port, *args, **kwargs):
        assert host in {"127.0.0.1", "::1"}
        return getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", numeric_only)
    with block_live_network(fixture_endpoints=ENDPOINTS, fixture_names={"fixture.test": "127.0.0.1"}) as attempts:
        assert socket.gethostbyname("fixture.test") == "127.0.0.1"
        assert socket.getaddrinfo("fixture.test", 18081)[0][4] == ("127.0.0.1", 18081)
        assert socket.getaddrinfo("::1", 18082)[0][4][:2] == ("::1", 18082)
        assert socket.getaddrinfo("127.0.0.1", None)[0][4] == ("127.0.0.1", 0)
    assert not attempts


def test_strict_connect_ex_and_create_connection_require_the_registered_port():
    with patch.object(socket.socket, "connect", return_value=None) as dial, \
            patch.object(socket.socket, "connect_ex", return_value=0) as dial_ex:
        with block_live_network(fixture_endpoints=ENDPOINTS, fixture_names={"fixture.test": "127.0.0.1"}) as attempts:
            with socket.socket() as client:
                client.connect(("fixture.test", 18081))
                assert client.connect_ex(("127.0.0.1", 18081)) == 0
                with pytest.raises(OSError):
                    client.connect_ex(("127.0.0.1", 18082))
                with pytest.raises(OSError):
                    client.connect(("192.0.2.4", 18081))
            with pytest.raises(OSError):
                socket.create_connection(("127.0.0.1", 18082), timeout=1)
        assert dial.call_args.args[1] == ("127.0.0.1", 18081)
        assert dial_ex.call_count == 1
    assert attempts == [("connect", "127.0.0.1"), ("connect", "192.0.2.4"), ("connect", "127.0.0.1")]


def test_ambient_and_declared_product_names_allow_no_raw_lookup(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_DB_HOST", "developer-db.test")
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://another-dev-db.test")
    monkeypatch.setattr(socket, "gethostname", lambda: "developer-machine")
    with block_live_network(fixture_endpoints=ENDPOINTS) as attempts:
        for hostname in ("developer-db.test", "another-dev-db.test", "developer-machine", "huggingface.co"):
            with pytest.raises(socket.gaierror):
                socket.getaddrinfo(hostname, None)
        with socket.socket(socket.AF_UNIX) as client, pytest.raises(OSError):
            client.connect(str(tmp_path / "nonfixture.sock"))
    assert attempts == [
        ("dns", "developer-db.test"), ("dns", "another-dev-db.test"), ("dns", "developer-machine"),
        ("dns", "huggingface.co"), ("connect", "<unix>"),
    ]


@pytest.mark.parametrize("endpoints,names", [
    (frozenset({("192.0.2.4", 443)}), {}),
    (frozenset({("127.0.0.1", 0)}), {}),
    (ENDPOINTS, {"other": "127.0.0.2"}),
    (ENDPOINTS, {"192.0.2.4": "127.0.0.1"}),
])
def test_invalid_fixture_configuration_fails(endpoints, names):
    with pytest.raises(ValueError, match="invalid fixture"):
        with block_live_network(fixture_endpoints=endpoints, fixture_names=names):
            pytest.fail("invalid fixture accepted")


@pytest.mark.parametrize("args", [["--inside-container"], ["--child", "stop", "report"]])
def test_internal_backend_entry_points_require_container_preflight(monkeypatch, args):
    from Helper_Scripts.ci import outbound_backend_smoke as smoke

    monkeypatch.delenv("TLDW_OUTBOUND_PREFLIGHT", raising=False)
    monkeypatch.setattr(smoke, "run_backend_smoke", lambda *_: pytest.fail("launched the app"))
    monkeypatch.setattr(smoke, "run_child", lambda *_: pytest.fail("launched the child"))
    assert smoke.main(args) == 1


@pytest.mark.parametrize("case", ["normal", "local-providers", "caught", "shutdown", "atexit", "missing"])
def test_child_terminal_evidence_survives_swallowed_attempts_and_requires_shutdown(tmp_path, case):
    from Helper_Scripts.ci.outbound_backend_smoke import SmokeError, check_child_result

    code = """
import asyncio
import atexit
import socket
import sys
from pathlib import Path
from Helper_Scripts.ci import outbound_backend_smoke as smoke

case = sys.argv[1]
async def serve(stop):
    if case == 'local-providers':
        for port in (5000, 5001, 8000, 8080):
            assert socket.getaddrinfo('127.0.0.1', port)[0][4] == ('127.0.0.1', port)
    if case == 'atexit':
        @atexit.register
        def probe():
            try:
                socket.gethostbyname('sensitive-exit.invalid')
            except OSError:
                pass
    if case == 'caught':
        try:
            socket.getaddrinfo('sensitive-input.invalid', 443)
        except OSError:
            pass
    if case == 'missing':
        raise RuntimeError('simulated termination')
    try:
        await asyncio.sleep(0)
    finally:
        if case == 'shutdown':
            try:
                socket.gethostbyname('sensitive-shutdown.invalid')
            except OSError:
                pass

smoke.serve_until_stop = serve
args = {'local_provider_fixtures': True} if case == 'local-providers' else {}
smoke.run_child(Path(sys.argv[2]), Path(sys.argv[3]), **args)
"""
    report, stop = tmp_path / "result.json", tmp_path / "stop"
    stop.touch()
    child = subprocess.run(  # nosec B603
        [sys.executable, "-B", "-c", code, case, str(stop), str(report)],
        cwd=ROOT, capture_output=True, timeout=8, check=False,
    )
    if case == "missing":
        assert child.returncode != 0
        assert not report.exists()
        with pytest.raises(SmokeError, match="backend_child_failed"):
            check_child_result(child.returncode, report)
    elif case in {"normal", "local-providers"}:
        assert check_child_result(child.returncode, report) == []
    else:
        assert child.returncode == 0
        assert "sensitive-" not in report.read_text()
        assert json.loads(report.read_text())["violations"] == [
            {"component": "backend", "reason": "unexpected_dns_attempt"},
        ]
        with pytest.raises(SmokeError, match="backend_unexpected_network"):
            check_child_result(child.returncode, report)
