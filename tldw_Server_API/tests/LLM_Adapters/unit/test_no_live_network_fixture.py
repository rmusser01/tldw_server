"""The blocker that proves a test makes no live DNS query or connect."""

from __future__ import annotations

import socket

import pytest

from tldw_Server_API.tests.helpers.offline_network import block_live_network


@pytest.mark.unit
def test_a_non_loopback_lookup_is_blocked_and_recorded():
    with block_live_network() as attempts:
        with pytest.raises(socket.gaierror):
            socket.getaddrinfo("api.openai.com", 443)
        with pytest.raises(socket.gaierror):
            socket.gethostbyname("api.openai.com")
    assert attempts == [("dns", "api.openai.com"), ("dns", "api.openai.com")]


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
def test_the_real_functions_are_restored():
    before = (socket.getaddrinfo, socket.gethostbyname, socket.socket.connect, socket.socket.connect_ex)
    with block_live_network():
        assert socket.getaddrinfo is not before[0]
    after = (socket.getaddrinfo, socket.gethostbyname, socket.socket.connect, socket.socket.connect_ex)
    assert after == before
