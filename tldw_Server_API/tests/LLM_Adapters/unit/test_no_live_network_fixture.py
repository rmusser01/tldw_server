"""The blocker that proves a test makes no live DNS query or connect."""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess  # nosec B404 - a pytest subprocess with a fixed interpreter and arguments.
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.tests.helpers import offline_network
from tldw_Server_API.tests.helpers.offline_network import (
    LIVE_NETWORK_DIRS,
    LIVE_NETWORK_MARKERS,
    block_live_network,
    live_network_allowed,
)

_PROBE_ENV = "TLDW_OFFLINE_BLOCKER_PROBE"
_ELSEWHERE = "tldw_Server_API/tests/Chat/test_elsewhere.py::test_unmarked"


def _item(path, nodeid=_ELSEWHERE, marker=None):
    return SimpleNamespace(nodeid=nodeid, path=path, get_closest_marker=lambda name: name == marker or None)


@pytest.mark.unit
def test_a_non_loopback_lookup_is_blocked_and_recorded():
    with block_live_network() as attempts:
        with pytest.raises(socket.gaierror):
            socket.getaddrinfo("api.openai.com", 443)
        with pytest.raises(socket.gaierror):
            socket.gethostbyname("api.openai.com")
    assert attempts == [("dns", "api.openai.com"), ("dns", "api.openai.com")]
    assert len(attempts.details) == 2
    assert attempts.details[0].startswith("dns api.openai.com in thread MainThread: ")
    assert "test_no_live_network_fixture.py" in attempts.details[0]


@pytest.mark.unit
def test_a_non_loopback_connect_is_blocked_before_any_packet():
    with block_live_network() as attempts:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError):
                sock.connect(("93.184.216.34", 80))
            with pytest.raises(OSError):
                sock.connect_ex(("93.184.216.34", 80))
        finally:
            sock.close()
    assert attempts == [("connect", "93.184.216.34"), ("connect", "93.184.216.34")]


@pytest.mark.unit
def test_loopback_is_allowed_and_not_recorded():
    with block_live_network() as attempts:
        assert socket.getaddrinfo("127.0.0.1", 80)
    assert attempts == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "host", ["127.0.0.2", "::ffff:127.0.0.1", "0.0.0.0", "[::1]", "testserver", socket.gethostname(), None]
)
def test_loopback_is_decided_by_address_not_by_spelling(host, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args, **_kwargs: ["resolved"])
    with block_live_network() as attempts:
        assert socket.getaddrinfo(host, 80) == ["resolved"]
    assert attempts == []


@pytest.mark.unit
def test_a_numeric_host_resolves_on_the_machine_but_its_connect_is_blocked():
    with block_live_network() as attempts:
        assert socket.getaddrinfo("192.168.1.1", 80)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError):
                sock.connect(("192.168.1.1", 80))
        finally:
            sock.close()
    assert attempts == [("connect", "192.168.1.1")]


@pytest.mark.unit
def test_the_configured_test_database_host_is_allowed(monkeypatch):
    monkeypatch.setenv("TEST_DB_HOST", "pg.ci.internal")
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://user:secret@db.ci.internal:5432/tldw_test")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args, **_kwargs: ["resolved"])
    with block_live_network() as attempts:
        assert socket.getaddrinfo("pg.ci.internal", 5432) == ["resolved"]
        assert socket.getaddrinfo("DB.CI.INTERNAL", 5432) == ["resolved"]
        with pytest.raises(socket.gaierror):
            socket.getaddrinfo("other.ci.internal", 5432)
    assert attempts == [("dns", "other.ci.internal")]


@pytest.mark.unit
def test_the_real_functions_are_restored():
    before = (socket.getaddrinfo, socket.gethostbyname, socket.socket.connect, socket.socket.connect_ex)
    with block_live_network():
        assert socket.getaddrinfo is not before[0]
    after = (socket.getaddrinfo, socket.gethostbyname, socket.socket.connect, socket.socket.connect_ex)
    assert after == before


@pytest.mark.unit
@pytest.mark.live_network
def test_a_live_network_marked_test_is_let_through(request):
    """The registered marker (pyproject.toml, --strict-markers) is what the autouse fixture honours."""
    assert live_network_allowed(request.node)


@pytest.mark.unit
@pytest.mark.parametrize("marker", sorted(LIVE_NETWORK_MARKERS))
def test_a_marked_test_opts_out(marker, request):
    assert live_network_allowed(_item(request.node.path, marker=marker))


@pytest.mark.unit
@pytest.mark.parametrize("tree", sorted(LIVE_NETWORK_DIRS))
def test_a_live_server_e2e_tree_opts_out(tree):
    tests_dir = Path(__file__).resolve().parents[2]
    assert live_network_allowed(_item(tests_dir / tree / "test_live.py"))


@pytest.mark.unit
def test_a_baselined_node_id_opts_out(request, monkeypatch):
    monkeypatch.setattr(offline_network, "baseline_node_ids", lambda: frozenset({_ELSEWHERE}))
    assert live_network_allowed(_item(request.node.path))
    assert not live_network_allowed(_item(request.node.path, nodeid=_ELSEWHERE + "[other]"))


@pytest.mark.unit
def test_an_unmarked_test_elsewhere_stays_offline(request):
    assert not live_network_allowed(_item(request.node.path))


@pytest.mark.unit
def test_the_session_blocker_errors_a_test_whose_live_lookup_was_swallowed():
    """tests/conftest.py's autouse blocker reports the attempt at teardown even when the code hid the failure."""
    target = f"{Path(__file__).resolve()}::test_probe_target_swallows_a_live_lookup"
    result = subprocess.run(  # nosec B603 - repo-owned test path, fixed argv, no shell.
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=line", target],
        cwd=Path(__file__).resolve().parents[4],
        env={**os.environ, _PROBE_ENV: "1", "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "live network attempted: [('dns', 'example.com')]" in output, output


@pytest.mark.unit
def test_probe_target_swallows_a_live_lookup():
    """Subprocess target of the test above: the lookup fails, the code swallows it, the test body passes."""
    if not os.environ.get(_PROBE_ENV):
        pytest.skip("runs only as the subprocess target of test_the_session_blocker_errors_a_test_whose_live_lookup_was_swallowed")
    with contextlib.suppress(OSError):
        socket.getaddrinfo("example.com", 80)
