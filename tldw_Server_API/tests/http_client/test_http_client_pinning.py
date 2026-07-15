import hashlib

import pytest


pytestmark = pytest.mark.unit


def _has_httpx():
    try:
        import httpx  # noqa: F401
        return True
    except Exception:
        return False


requires_httpx = pytest.mark.skipif(not _has_httpx(), reason="httpx not installed")


def _install_fake_tls(monkeypatch, der: bytes | None) -> None:
    import socket as _socket
    import ssl as _ssl

    class FakeSSLSocket:
        def getpeercert(self, binary_form=False):
            return der if binary_form else None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

    class FakeSSLContext:
        minimum_version = None

        def wrap_socket(self, sock, server_hostname=None):  # noqa: ARG002
            return FakeSSLSocket()

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

    monkeypatch.setattr(_ssl, "create_default_context", lambda *args, **kwargs: FakeSSLContext())
    monkeypatch.setattr(
        _socket,
        "create_connection",
        lambda addr, timeout=None: FakeSocket(),  # noqa: ARG005
    )


@requires_httpx
def test_tls_pinning_success(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    fake_der = b"fakecert"
    pin = hashlib.sha256(fake_der).hexdigest().lower()
    _install_fake_tls(monkeypatch, fake_der)
    monkeypatch.setattr(hc, "_validate_egress_or_raise", lambda _url: None)

    hc._check_cert_pinning("example.com", 443, {pin}, "1.2")


@requires_httpx
def test_tls_pinning_mismatch(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.exceptions import EgressPolicyError

    _install_fake_tls(monkeypatch, b"anothercert")
    monkeypatch.setattr(hc, "_validate_egress_or_raise", lambda _url: None)

    with pytest.raises(EgressPolicyError) as exc:
        hc._check_cert_pinning("example.com", 443, {"deadbeef"}, "1.2")

    assert exc.value.reason_code == "tls_pin_mismatch"


@requires_httpx
def test_tls_pinning_scoped_validation_uses_original_accepted_ips(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    der = b"scoped-cert"
    pin = hashlib.sha256(der).hexdigest()
    _install_fake_tls(monkeypatch, der)
    captured: dict[str, object] = {}

    def fake_validate(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)

    monkeypatch.setattr(hc, "_validate_egress_or_raise", fake_validate)
    scope = ConfiguredEndpointScope.from_url("https://192.168.1.50:11434")

    hc._check_cert_pinning(
        "192.168.1.50",
        11434,
        {pin},
        "1.2",
        configured_endpoint=scope,
        accepted_resolved_ips=("192.168.1.50",),
    )

    assert captured["url"] == "https://192.168.1.50:11434"
    assert captured["configured_endpoint"] is scope
    assert captured["dns_pin_cache"] == {"192.168.1.50": ("192.168.1.50",)}


@requires_httpx
def test_tls_pinning_scoped_validation_preserves_explicit_https_port_80(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    der = b"scoped-port-cert"
    pin = hashlib.sha256(der).hexdigest()
    _install_fake_tls(monkeypatch, der)
    captured: dict[str, object] = {}

    def fake_validate(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)

    monkeypatch.setattr(hc, "_validate_egress_or_raise", fake_validate)
    scope = ConfiguredEndpointScope.from_url("https://[2001:db8::1]:80")

    hc._check_cert_pinning(
        "2001:db8::1",
        80,
        {pin},
        "1.2",
        configured_endpoint=scope,
        accepted_resolved_ips=("2001:db8::1",),
    )

    assert captured["url"] == "https://[2001:db8::1]:80"
    assert captured["configured_endpoint"] is scope
    assert captured["dns_pin_cache"] == {"2001:db8::1": ("2001:db8::1",)}


@pytest.mark.parametrize("reason_code", ["origin_mismatch", "address_forbidden", "dns_changed"])
def test_tls_pinning_preserves_nested_policy_reason(monkeypatch, reason_code):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.exceptions import EgressPolicyError

    def deny(*_args, **_kwargs):
        raise EgressPolicyError("denied", reason_code=reason_code)

    monkeypatch.setattr(hc, "_validate_egress_or_raise", deny)

    with pytest.raises(EgressPolicyError) as exc:
        hc._check_cert_pinning("example.com", 443, {"pin"}, "1.2")

    assert exc.value.reason_code == reason_code


@pytest.mark.parametrize(
    ("der", "pins", "expected_code"),
    [
        (None, {"pin"}, "tls_pin_missing"),
        (b"certificate", {"wrong"}, "tls_pin_mismatch"),
    ],
)
def test_tls_pinning_assigns_typed_certificate_denials(monkeypatch, der, pins, expected_code):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.exceptions import EgressPolicyError

    monkeypatch.setattr(hc, "_validate_egress_or_raise", lambda *_args, **_kwargs: None)
    _install_fake_tls(monkeypatch, der)

    with pytest.raises(EgressPolicyError) as exc:
        hc._check_cert_pinning("example.com", 443, pins, "1.2")

    assert exc.value.reason_code == expected_code


def test_tls_pinning_assigns_typed_socket_error(monkeypatch):
    import socket as _socket

    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.exceptions import EgressPolicyError

    monkeypatch.setattr(hc, "_validate_egress_or_raise", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        _socket,
        "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("socket failed")),
    )

    with pytest.raises(EgressPolicyError) as exc:
        hc._check_cert_pinning("example.com", 443, {"pin"}, "1.2")

    assert exc.value.reason_code == "tls_pin_error"


@requires_httpx
def test_fetch_enforces_pin_and_preserves_typed_denial(monkeypatch):
    import httpx

    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.exceptions import EgressPolicyError
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    scope = ConfiguredEndpointScope.from_url("https://93.184.216.34:11434")
    calls = {"io": 0}

    def deny_pin(*_args, **kwargs):
        assert kwargs["configured_endpoint"] is scope
        assert kwargs["accepted_resolved_ips"] == ("93.184.216.34",)
        raise EgressPolicyError("mismatch", reason_code="tls_pin_mismatch")

    def handler(request: httpx.Request) -> httpx.Response:
        calls["io"] += 1
        return httpx.Response(200, request=request)

    monkeypatch.setattr(hc, "_check_cert_pinning", deny_pin)
    client = hc.create_client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(EgressPolicyError) as exc:
            hc.fetch(
                method="GET",
                url="https://93.184.216.34:11434/models",
                client=client,
                cert_pinning={"93.184.216.34": {"pin"}},
                configured_endpoint=scope,
            )
    finally:
        client.close()

    assert exc.value.reason_code == "tls_pin_mismatch"
    assert calls["io"] == 0


@requires_httpx
@pytest.mark.asyncio
async def test_afetch_enforces_pin_and_preserves_typed_denial(monkeypatch):
    import httpx

    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.exceptions import EgressPolicyError
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    scope = ConfiguredEndpointScope.from_url("https://93.184.216.34:11434")
    calls = {"io": 0}

    def deny_pin(*_args, **kwargs):
        assert kwargs["configured_endpoint"] is scope
        assert kwargs["accepted_resolved_ips"] == ("93.184.216.34",)
        raise EgressPolicyError("no certificate", reason_code="tls_pin_missing")

    def handler(request: httpx.Request) -> httpx.Response:
        calls["io"] += 1
        return httpx.Response(200, request=request)

    monkeypatch.setattr(hc, "_check_cert_pinning", deny_pin)
    client = hc.create_async_client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(EgressPolicyError) as exc:
            await hc.afetch(
                method="GET",
                url="https://93.184.216.34:11434/models",
                client=client,
                cert_pinning={"93.184.216.34": {"pin"}},
                configured_endpoint=scope,
            )
    finally:
        await client.aclose()

    assert exc.value.reason_code == "tls_pin_missing"
    assert calls["io"] == 0


def test_tls_min_version_mapping():
    import ssl
    from tldw_Server_API.app.core.http_client import _tls_min_version_from_str

    assert _tls_min_version_from_str("1.3") == ssl.TLSVersion.TLSv1_3
    assert _tls_min_version_from_str("1.2") == ssl.TLSVersion.TLSv1_2


@requires_httpx
def test_env_pins_attached_to_client(monkeypatch):
    from tldw_Server_API.app.core.http_client import create_client, _get_client_cert_pins

    monkeypatch.setenv("HTTP_CERT_PINS", "example.com=deadbeef|cafebabe,api.example.com=abcd")
    c = create_client()
    pins = _get_client_cert_pins(c)
    assert pins is not None
    assert "example.com" in pins and "deadbeef" in pins["example.com"]
    assert "api.example.com" in pins and "abcd" in pins["api.example.com"]
