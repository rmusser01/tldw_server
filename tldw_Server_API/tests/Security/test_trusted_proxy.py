import pytest

from tldw_Server_API.app.core.Security.trusted_proxy import (
    is_trusted_proxy_peer,
    resolve_trusted_client_ip,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("peer", "expected"),
    [
        ("203.0.113.7", "203.0.113.7"),
        ("2001:0db8:0:0::7", "2001:db8::7"),
        (None, None),
        ("testclient", None),
        ("127.0.0.1:8000", None),
    ],
)
def test_direct_peer_is_canonical_or_absent(peer, expected):
    assert resolve_trusted_client_ip(peer, ()) == expected


def test_untrusted_peer_cannot_select_forwarded_identity():
    assert resolve_trusted_client_ip(
        "198.51.100.8",
        ("10.0.0.0/8", "not-a-network"),
        forwarded_for_values=("203.0.113.9",),
        single_forwarded_value="192.0.2.4",
    ) == "198.51.100.8"


def test_trusted_peer_predicate_accepts_hosts_and_cidrs():
    entries = ("192.0.2.10", "2001:db8:abcd::/48", "invalid")
    assert is_trusted_proxy_peer("192.0.2.10", entries) is True
    assert is_trusted_proxy_peer("2001:db8:abcd::4", entries) is True
    assert is_trusted_proxy_peer("203.0.113.4", entries) is False


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (("203.0.113.9",), "203.0.113.9"),
        (("198.51.100.99, 203.0.113.9, 10.0.0.2",), "203.0.113.9"),
        (("198.51.100.99", "203.0.113.9, 10.0.0.2"), "203.0.113.9"),
        (("2001:db8:ffff::9, 2001:db8:abcd::2",), "2001:db8:ffff::9"),
        (("10.0.0.2, 10.0.0.3",), "10.0.0.1"),
        (("203.0.113.9,,10.0.0.2",), "10.0.0.1"),
        (("203.0.113.9:443,10.0.0.2",), "10.0.0.1"),
        (("[2001:db8::9],10.0.0.2",), "10.0.0.1"),
        (("for=203.0.113.9,10.0.0.2",), "10.0.0.1"),
        (("fe80::1%eth0,10.0.0.2",), "10.0.0.1"),
    ],
)
def test_xff_is_scanned_from_trusted_edge(values, expected):
    assert resolve_trusted_client_ip(
        "10.0.0.1",
        ("10.0.0.0/8", "2001:db8:abcd::/48"),
        forwarded_for_values=values,
    ) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("203.0.113.9", "203.0.113.9"),
        ("2001:0db8::9", "2001:db8::9"),
        ("203.0.113.9, 10.0.0.2", "10.0.0.1"),
        ("", "10.0.0.1"),
        ("[2001:db8::9]", "10.0.0.1"),
    ],
)
def test_single_forwarded_field_accepts_exactly_one_plain_ip(value, expected):
    assert resolve_trusted_client_ip(
        "10.0.0.1",
        ("10.0.0.0/8",),
        single_forwarded_value=value,
    ) == expected
