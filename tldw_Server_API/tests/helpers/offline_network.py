"""Block every non-loopback DNS lookup and connect, and record each attempt.

A call whose exception a library swallows still shows up in the returned list, so a test
can assert that no live network was touched. ``tests/conftest.py`` wraps every test in it
(TASK-15486); ``live_network_allowed`` says which tests opt out.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import threading
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

LIVE_NETWORK_MARKERS = frozenset(
    {"live_network", "external_api", "local_llm_service", "provider_inventory", "requires_model", "stt_golden"}
)
LIVE_NETWORK_DIRS = frozenset({"e2e", "server_e2e_tests", "frontend_e2e"})
PUBLIC_ADDRESS = "93.184.216.34"
EGRESS_ALLOWLIST_ENV = "WORKFLOWS_EGRESS_ALLOWLIST"
_LOCAL_NAMES = frozenset({"", "localhost", "testserver"})
_TESTS_DIR = Path(__file__).resolve().parents[1]
# Node ids still allowed to reach the network; tests/lint/test_live_network_baseline_ratchet.py keeps it shrinking.
LIVE_NETWORK_BASELINE = _TESTS_DIR / "live_network_baseline.txt"

# ponytail: wraps the Python-level socket API only. Not seen: `_socket.socket.connect` called on the C
# class, UDP `sendto`, `gethostbyname_ex` / `gethostbyaddr`, and resolvers outside Python (libpq, c-ares,
# grpc). Covered: `socket.connect`/`connect_ex`, `create_connection`, asyncio `open_connection` and
# `sock_connect`, ssl, httpx/anyio/urllib3. Widen when a leak through one of those is found.


class Attempts(list):
    """The ``(kind, host)`` tuples a blocked run recorded, plus ``details`` with thread and stack per attempt."""

    def __init__(self) -> None:
        super().__init__()
        self.details: list[str] = []

    def record(self, kind: str, host: str) -> None:
        self.append((kind, host))
        frames = traceback.extract_stack(limit=12)[:-2]
        where = " <- ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in reversed(frames))
        self.details.append(f"{kind} {host} in thread {threading.current_thread().name}: {where}")


def _host(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("ascii", "replace")
    return str(value).lower().strip("[]")


def _ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return None
    return getattr(ip, "ipv4_mapped", None) or ip


def _allowed_names() -> frozenset[str]:
    """Loopback names, the machine's own hostname and the developer's test PostgreSQL host.

    These names are compared at DNS; a connect to the address they resolve to is still judged by
    address, so a hostname that resolves off loopback is a false positive (blocked), never a leak.
    """
    names = {socket.gethostname(), os.environ.get("TEST_DB_HOST", "")}
    try:
        names.add(urlsplit(os.environ.get("TEST_DATABASE_URL", "")).hostname or "")
    except ValueError:
        pass
    return _LOCAL_NAMES | {name.lower() for name in names if name}


def _is_local(host: str, allowed: frozenset[str]) -> bool:
    if host in allowed:
        return True
    ip = _ip(host)
    return ip is not None and (ip.is_loopback or ip.is_unspecified)


def _strict_address(host: object) -> str | None:
    name = _host(host)
    if "%" in name:
        return None
    address = _ip(name)
    return str(address) if address is not None and address.is_loopback else None


@contextmanager
def _block_except_fixtures(
    endpoints: frozenset[tuple[str, int]], names: Mapping[str, str],
) -> Iterator[Attempts]:
    registered = set()
    for host, port in endpoints:
        numeric = _strict_address(host)
        if numeric is None or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("invalid fixture endpoint")
        registered.add((numeric, port))
    hosts = {host for host, _ in registered}
    aliases = {}
    for alias, host in names.items():
        numeric = _strict_address(host)
        if not alias or _host(alias) != alias or _ip(alias) is not None or numeric not in hosts:
            raise ValueError("invalid fixture name")
        aliases[alias] = numeric

    attempts = Attempts()
    active = True
    getaddrinfo = socket.getaddrinfo
    gethostbyname = socket.gethostbyname
    gethostbyname_ex = socket.gethostbyname_ex
    gethostbyaddr = socket.gethostbyaddr
    create_connection = socket.create_connection

    def fixture_host(value: object) -> str | None:
        return aliases.get(_host(value)) or _strict_address(value)

    def resolve_host(host: object) -> str:
        numeric = fixture_host(host)
        if numeric in hosts:
            return numeric
        attempts.record("dns", _host(host))
        raise socket.gaierror(socket.EAI_NONAME, "non-fixture lookup blocked by the test")

    def resolve(host, port, family=0, type=0, proto=0, flags=0):
        if not active:
            return getaddrinfo(host, port, family, type, proto, flags)
        numeric = resolve_host(host)
        service = int(port) if isinstance(port, str) and port.isascii() and port.isdigit() else port
        if port is not None and (not isinstance(service, int) or (numeric, service) not in registered):
            attempts.record("dns", _host(host))
            raise socket.gaierror(socket.EAI_NONAME, "non-fixture service blocked by the test")
        return getaddrinfo(numeric, service, family, type, proto, flags | socket.AI_NUMERICHOST | socket.AI_NUMERICSERV)

    def resolve_ipv4(host):
        if not active:
            return gethostbyname(host)
        numeric = resolve_host(host)
        if ":" in numeric:
            raise socket.gaierror(socket.EAI_FAMILY, "fixture has no IPv4 address")
        return numeric

    def resolve_ipv4_ex(host):
        return (_host(host), [], [resolve_ipv4(host)]) if active else gethostbyname_ex(host)

    def resolve_reverse(host):
        if not active:
            return gethostbyaddr(host)
        numeric = resolve_host(host)
        return numeric, [], [numeric]

    def fixture_dial(address):
        if isinstance(address, tuple) and len(address) in {2, 4}:
            host, port, *ipv6 = address
            numeric = fixture_host(host)
            if isinstance(port, int) and (numeric, port) in registered and (not ipv6 or ipv6 == [0, 0]):
                # Preserve an IPv4-mapped IPv6 literal for an IPv6 socket.
                wire_host = host if numeric != _host(host) and ":" in _host(host) else numeric
                return (wire_host, port, *ipv6)
            name = _host(host)
        else:
            name = "<unix>"
        attempts.record("connect", name)
        raise OSError("non-fixture connect blocked by the test")

    def guarded_dial(real):
        def connect(sock, address, *args, **kwargs):
            return real(sock, fixture_dial(address) if active else address, *args, **kwargs)
        return connect

    def create(address, *args, **kwargs):
        return create_connection(fixture_dial(address) if active else address, *args, **kwargs)

    with (
        mock.patch.object(socket, "getaddrinfo", resolve),
        mock.patch.object(socket, "gethostbyname", resolve_ipv4),
        mock.patch.object(socket, "gethostbyname_ex", resolve_ipv4_ex),
        mock.patch.object(socket, "gethostbyaddr", resolve_reverse),
        mock.patch.object(socket, "create_connection", create),
        mock.patch.object(socket.socket, "connect", guarded_dial(socket.socket.connect)),
        mock.patch.object(socket.socket, "connect_ex", guarded_dial(socket.socket.connect_ex)),
    ):
        try:
            yield attempts
        finally:
            active = False


@contextmanager
def block_live_network(
    *,
    fixture_endpoints: frozenset[tuple[str, int]] | None = None,
    fixture_names: Mapping[str, str] | None = None,
) -> Iterator[Attempts]:
    if fixture_endpoints is not None:
        with _block_except_fixtures(fixture_endpoints, fixture_names or {}) as attempts:
            yield attempts
        return
    if fixture_names is not None:
        raise ValueError("invalid fixture names without endpoints")
    attempts = Attempts()
    allowed = _allowed_names()
    # monkeypatch is set up before the autouse fixture, so its undo runs after this block exits and can
    # reinstall the wrappers it saw; once closed they pass straight through.
    active = True

    def resolver(real):
        def resolve(host, *args, **kwargs):
            name = _host(host)
            if not active:
                return real(host, *args, **kwargs)
            # A numeric host resolves on the machine; only its connect can leave it.
            if _is_local(name, allowed) or _ip(name) is not None:
                return real(host, *args, **kwargs)
            attempts.record("dns", name)
            raise socket.gaierror(socket.EAI_NONAME, "live DNS lookup blocked by the test")

        return resolve

    def guarded(real):
        def connect(self, address, *args, **kwargs):
            # A str address is an AF_UNIX path; only (host, port) tuples can leave the machine.
            if active and isinstance(address, tuple) and not _is_local(_host(address[0]), allowed):
                attempts.record("connect", _host(address[0]))
                raise OSError("live connect blocked by the test")
            return real(self, address, *args, **kwargs)

        return connect

    with (
        mock.patch.object(socket, "getaddrinfo", resolver(socket.getaddrinfo)),
        mock.patch.object(socket, "gethostbyname", resolver(socket.gethostbyname)),
        mock.patch.object(socket.socket, "connect", guarded(socket.socket.connect)),
        mock.patch.object(socket.socket, "connect_ex", guarded(socket.socket.connect_ex)),
    ):
        try:
            yield attempts
        finally:
            active = False


@lru_cache(maxsize=1)
def baseline_node_ids() -> frozenset[str]:
    """Node ids listed in ``live_network_baseline.txt`` (one per line, ``#`` comments allowed)."""
    if not LIVE_NETWORK_BASELINE.exists():
        return frozenset()
    lines = (line.strip() for line in LIVE_NETWORK_BASELINE.read_text(encoding="utf-8").splitlines())
    return frozenset(line for line in lines if line and not line.startswith("#"))


def live_network_allowed(item) -> bool:
    """A test opts out of the session-wide blocker by marker, by baseline entry, or by living in a live-server e2e tree."""
    if any(item.get_closest_marker(name) for name in LIVE_NETWORK_MARKERS):
        return True
    if item.nodeid in baseline_node_ids():
        return True
    try:
        return Path(item.path).resolve().relative_to(_TESTS_DIR).parts[0] in LIVE_NETWORK_DIRS
    except (ValueError, IndexError):
        return False


def stub_host_resolution(monkeypatch, address: str = PUBLIC_ADDRESS) -> None:
    """Every name host resolves to one public address with no lookup, so the egress policy admits it.

    The policy (``Utils.egress.evaluate_url_policy``) resolves each name host it is asked about;
    callers bind the policy by name, so the seam is the resolver they all share.
    """
    from tldw_Server_API.app.core.Utils import egress

    monkeypatch.setattr(egress, "resolve_host_ips", lambda *_args, **_kwargs: (address,))


def allow_egress_hosts(monkeypatch, *hosts: str) -> None:
    """Add hosts to the egress allowlist when one is set (CI sets one; an empty list admits every public host)."""
    current = [host.strip() for host in os.environ.get(EGRESS_ALLOWLIST_ENV, "").split(",") if host.strip()]
    if current:
        monkeypatch.setenv(EGRESS_ALLOWLIST_ENV, ",".join(dict.fromkeys(current + list(hosts))))
