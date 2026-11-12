"""Block every non-loopback DNS lookup and connect, and record each attempt.

A call whose exception a library swallows still shows up in the returned list, so a test
can assert that no live network was touched.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1", "testserver"})


def _host(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii", "replace")
    return str(value).lower()


@contextmanager
def block_live_network() -> Iterator[list[tuple[str, str]]]:
    attempts: list[tuple[str, str]] = []
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def getaddrinfo(host, *args, **kwargs):
        if _host(host) in _LOOPBACK:
            return real_getaddrinfo(host, *args, **kwargs)
        attempts.append(("dns", _host(host)))
        raise socket.gaierror(socket.EAI_NONAME, "live DNS lookup blocked by the test")

    def gethostbyname(host):
        if _host(host) in _LOOPBACK:
            return real_gethostbyname(host)
        attempts.append(("dns", _host(host)))
        raise socket.gaierror(socket.EAI_NONAME, "live DNS lookup blocked by the test")

    def guarded(real):
        def connect(self, address, *args, **kwargs):
            # A str address is an AF_UNIX path; only (host, port) tuples can leave the machine.
            if isinstance(address, tuple) and _host(address[0]) not in _LOOPBACK:
                attempts.append(("connect", _host(address[0])))
                raise OSError("live connect blocked by the test")
            return real(self, address, *args, **kwargs)

        return connect

    with (
        mock.patch.object(socket, "getaddrinfo", getaddrinfo),
        mock.patch.object(socket, "gethostbyname", gethostbyname),
        mock.patch.object(socket.socket, "connect", guarded(real_connect)),
        mock.patch.object(socket.socket, "connect_ex", guarded(real_connect_ex)),
    ):
        yield attempts
