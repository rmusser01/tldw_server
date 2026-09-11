from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers

from tldw_Server_API.app.core.AuthNZ.ip_allowlist import (
    is_single_user_ip_allowed,
    resolve_client_ip,
)

pytestmark = pytest.mark.unit


def _settings(allowed):


    return SimpleNamespace(SINGLE_USER_ALLOWED_IPS=allowed)


def _request(peer, raw_headers):
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=Headers(raw=raw_headers))


def _proxy_settings(enabled=True):
    return SimpleNamespace(
        AUTH_TRUST_X_FORWARDED_FOR=enabled,
        AUTH_TRUSTED_PROXY_IPS=["10.0.0.0/8"],
    )


def test_allowlist_empty_allows_any_ip():


    settings = _settings([])
    assert is_single_user_ip_allowed("198.51.100.5", settings) is True
    assert is_single_user_ip_allowed(None, settings) is True


def test_allowlist_denies_missing_client_ip():


    settings = _settings(["203.0.113.10"])
    assert is_single_user_ip_allowed(None, settings) is False


@pytest.mark.parametrize(
    ("allowed", "ip", "expected"),
    [
        (["10.0.0.0/8"], "10.1.2.3", True),
        (["10.0.0.0/8"], "192.168.1.1", False),
        (["192.168.1.5"], "192.168.1.5", True),
        (["192.168.1.5"], "192.168.1.6", False),
        (["2001:db8::/32"], "2001:db8::1", True),
        (["2001:db8::/32"], "2001:db9::1", False),
    ],
)
def test_allowlist_ip_and_cidr_matching(allowed, ip, expected):
    settings = _settings(allowed)
    assert is_single_user_ip_allowed(ip, settings) is expected


def test_allowlist_invalid_entry_is_ignored():


    settings = _settings(["not-an-ip"])
    assert is_single_user_ip_allowed("192.0.2.1", settings) is False


def test_allowlist_invalid_client_ip_rejected():


    settings = _settings(["203.0.113.10"])
    assert is_single_user_ip_allowed("999.999.999.999", settings) is False


def test_resolve_client_ip_combines_repeated_xff_and_ignores_attacker_prefix():
    request = _request(
        "10.0.0.1",
        [
            (b"x-forwarded-for", b"198.51.100.99"),
            (b"x-forwarded-for", b"203.0.113.9, 10.0.0.2"),
        ],
    )
    assert resolve_client_ip(request, _proxy_settings()) == "203.0.113.9"


def test_resolve_client_ip_ignores_forwarding_when_disabled():
    request = _request("10.0.0.1", [(b"x-forwarded-for", b"203.0.113.9")])
    assert resolve_client_ip(request, _proxy_settings(enabled=False)) == "10.0.0.1"


def test_xff_takes_precedence_over_x_real_ip():
    request = _request(
        "10.0.0.1",
        [
            (b"x-forwarded-for", b"203.0.113.9"),
            (b"x-real-ip", b"198.51.100.4"),
        ],
    )
    assert resolve_client_ip(request, _proxy_settings()) == "203.0.113.9"


def test_malformed_xff_does_not_fall_through_to_x_real_ip():
    request = _request(
        "10.0.0.1",
        [
            (b"x-forwarded-for", b"bad, 10.0.0.2"),
            (b"x-real-ip", b"203.0.113.4"),
        ],
    )
    assert resolve_client_ip(request, _proxy_settings()) == "10.0.0.1"


def test_repeated_x_real_ip_is_not_treated_as_a_single_address():
    request = _request(
        "10.0.0.1",
        [
            (b"x-real-ip", b"203.0.113.4"),
            (b"x-real-ip", b"198.51.100.4"),
        ],
    )
    assert resolve_client_ip(request, _proxy_settings()) == "10.0.0.1"


def test_invalid_physical_peer_never_authorizes_forwarded_headers():
    request = _request("testclient", [(b"x-forwarded-for", b"203.0.113.4")])
    assert resolve_client_ip(request, _proxy_settings()) is None
