from types import SimpleNamespace

import pytest
from starlette.requests import Request

from tldw_Server_API.app.core.Resource_Governance.deps import derive_client_ip, derive_entity_key

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]


def _build_request(headers=None, raw_headers=None, client_host="127.0.0.1", client_port=12345):


    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": raw_headers or [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": (client_host, client_port),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def _build_request_with_app(headers=None):
    class _Loader:
        def get_snapshot(self):
            return SimpleNamespace(tenant={"enabled": True, "header": "X-TLDW-Tenant"})

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "app": SimpleNamespace(state=SimpleNamespace(rg_policy_loader=_Loader())),
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_x_forwarded_for_used_when_proxy_trusted(monkeypatch):
    # Trust 10.0.0.0/8 and read X-Forwarded-For
    monkeypatch.setenv("RG_TRUSTED_PROXIES", "10.0.0.0/8")
    monkeypatch.setenv("RG_CLIENT_IP_HEADER", "X-Forwarded-For")

    # Remote peer is a trusted proxy; header contains original client first
    r = _build_request(headers={"X-Forwarded-For": "203.0.113.5, 10.0.0.1"}, client_host="10.1.2.3")

    ent = derive_entity_key(r)
    assert ent == "ip:203.0.113.5"


@pytest.mark.asyncio
async def test_x_forwarded_for_ignored_when_proxy_untrusted(monkeypatch):
    # No trusted proxies configured; header should be ignored
    monkeypatch.delenv("RG_TRUSTED_PROXIES", raising=False)
    monkeypatch.setenv("RG_CLIENT_IP_HEADER", "X-Forwarded-For")

    r = _build_request(headers={"X-Forwarded-For": "198.51.100.7"}, client_host="1.2.3.4")
    ent = derive_entity_key(r)
    assert ent == "ip:1.2.3.4"


@pytest.mark.asyncio
async def test_derive_entity_key_uses_policy_snapshot_tenant_when_config_omitted():
    request = _build_request_with_app(headers={"X-TLDW-Tenant": "acme"})

    ent = derive_entity_key(request)

    assert ent == "tenant:acme"


@pytest.mark.asyncio
async def test_xff_uses_first_untrusted_hop_from_the_right(monkeypatch):
    monkeypatch.setenv("RG_TRUSTED_PROXIES", "10.0.0.0/8")
    monkeypatch.setenv("RG_CLIENT_IP_HEADER", "x-FoRwArDeD-fOr")
    request = _build_request(
        client_host="10.0.0.1",
        raw_headers=[(b"x-forwarded-for", b"198.51.100.99, 203.0.113.9, 10.0.0.2")],
    )
    assert derive_client_ip(request) == "203.0.113.9"


@pytest.mark.asyncio
async def test_custom_header_is_single_ip_only(monkeypatch):
    monkeypatch.setenv("RG_TRUSTED_PROXIES", "10.0.0.0/8")
    monkeypatch.setenv("RG_CLIENT_IP_HEADER", "CF-Connecting-IP")
    request = _build_request(
        client_host="10.0.0.1",
        raw_headers=[(b"cf-connecting-ip", b"203.0.113.9, 10.0.0.2")],
    )
    assert derive_client_ip(request) == "10.0.0.1"


@pytest.mark.asyncio
async def test_repeated_custom_header_is_not_treated_as_a_single_address(monkeypatch):
    monkeypatch.setenv("RG_TRUSTED_PROXIES", "10.0.0.0/8")
    monkeypatch.setenv("RG_CLIENT_IP_HEADER", "CF-Connecting-IP")
    request = _build_request(
        client_host="10.0.0.1",
        raw_headers=[
            (b"cf-connecting-ip", b"203.0.113.9"),
            (b"cf-connecting-ip", b"198.51.100.9"),
        ],
    )
    assert derive_client_ip(request) == "10.0.0.1"


@pytest.mark.asyncio
async def test_invalid_peer_is_unknown_not_loopback(monkeypatch):
    monkeypatch.setenv("RG_TRUSTED_PROXIES", "127.0.0.1")
    monkeypatch.setenv("RG_CLIENT_IP_HEADER", "X-Forwarded-For")
    request = _build_request(
        client_host="testclient",
        raw_headers=[(b"x-forwarded-for", b"203.0.113.9")],
    )
    assert derive_client_ip(request) == "unknown"
