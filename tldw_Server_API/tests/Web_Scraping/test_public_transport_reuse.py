"""Public acquisition through existing HTTPX/curl adapters and real curl I/O."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from tldw_Server_API.app.core import http_client as hc
from tldw_Server_API.app.core.Security import egress
from tldw_Server_API.app.core.Web_Scraping.preflight.adapters import http as adapter
from tldw_Server_API.app.core.Web_Scraping.preflight.probes import ProbeHttpRequest
from tldw_Server_API.app.core.Web_Scraping.runtime.requests import RuntimeRequestContext

URL = "http://public.example/article"
IP = "93.184.216.34"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["excess", "compressed", "success"])
async def test_public_httpx_probe_enforces_existing_body_limit(monkeypatch, kind):
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_models

    monkeypatch.setattr(article_models, "DEFAULT_MAX_ARTICLE_BYTES", 8)
    monkeypatch.setattr(egress, "_resolve_host_ips", lambda *_a, **_k: [IP])
    clients = []
    requests = []
    peers = []

    def send(request):
        peers.append(request.url.host)
        requests.append(request)
        return httpx.Response(
            200,
            headers={"Content-Encoding": "br"} if kind == "compressed" else {},
            stream=httpx.ByteStream(b"0123456789" if kind == "excess" else b"article"),
        )

    def create(**kwargs):
        clients.append(
            httpx.AsyncClient(
                transport=httpx.MockTransport(send), **{k: v for k, v in kwargs.items() if k != "proxies"}
            )
        )
        return clients[-1]

    monkeypatch.setattr(hc, "create_async_client", create)
    with egress.public_url_policy_scope():
        if kind == "success":
            response = await adapter.HttpxProbeTransport().send(ProbeHttpRequest(URL))
            assert response.text == "article"
            await response.aclose()
        else:
            with pytest.raises(hc.NetworkError) as failure:
                await adapter.HttpxProbeTransport().send(ProbeHttpRequest(URL))
            assert str(failure.value) == "NetworkError"
            assert failure.value.__cause__ is None
    assert clients[0].is_closed
    assert requests[0].headers["Accept-Encoding"] == "identity"
    assert peers == [IP]


@pytest.fixture(params=["http", "https"])
def curl_server(monkeypatch, tmp_path, request):
    curl = pytest.importorskip("curl_cffi")
    requests = []
    options = []
    handles = []
    connect_lists = []
    server_names = []
    scheme = request.param
    port = 443 if scheme == "https" else 80
    target_url = f"{scheme}://public.example/article"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, dict(self.headers)))
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/article")
                self.send_header("Set-Cookie", "server-secret=yes; Path=/")
                body = b""
            else:
                self.send_response(200)
                if self.path == "/compressed":
                    self.send_header("Content-Encoding", "br")
                body = b"0123456789" if self.path == "/excess" else b"article"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    if scheme == "https":
        import ssl
        from datetime import datetime, timedelta, timezone

        import certifi
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "public.example")])
        now = datetime.now(timezone.utc)
        certificate = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("public.example")]), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256())
        )
        cert_path = tmp_path / "public-certificate.pem"
        key_path = tmp_path / "public-key.pem"
        cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
        )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        context.set_servername_callback(lambda _socket, name, _context: server_names.append(name))
        server.socket = context.wrap_socket(server.socket, server_side=True)
        monkeypatch.setattr(certifi, "where", lambda: str(cert_path))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    setopt = curl.Curl.setopt

    def record_option(self, option, value):
        options.append((option, value))
        if self not in handles:
            handles.append(self)
        result = setopt(self, option, value)
        if option == curl.CurlOpt.RESOLVE:
            # Only the library boundary maps the asserted public target to this test server.
            from curl_cffi.curl import ffi, lib

            target = lib.curl_slist_append(ffi.NULL, f"public.example:{port}:127.0.0.1:{server.server_port}".encode())
            connect_lists.append(target)
            setopt(self, curl.CurlOpt.CONNECT_TO, target)
        return result

    monkeypatch.setattr(curl.Curl, "setopt", record_option)
    monkeypatch.setattr(egress, "_resolve_host_ips", lambda *_a, **_k: [IP])
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/nonexistent/ambient-ca")
    monkeypatch.setenv("CURL_CA_BUNDLE", "/nonexistent/ambient-ca")
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/ambient-ca")
    try:
        yield curl, requests, options, handles, target_url, server_names
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        from curl_cffi.curl import lib

        for target in connect_lists:
            lib.curl_slist_free_all(target)


@pytest.mark.parametrize("path", ["article", "redirect", "excess", "compressed"])
def test_real_public_curl_is_bounded_pinned_and_credential_free(curl_server, path):
    import certifi

    curl, requests, options, handles, target_url, server_names = curl_server
    with egress.public_url_policy_scope():
        if path in {"excess", "compressed"}:
            with pytest.raises(ValueError, match="max_response_bytes"):
                hc.fetch(
                    target_url.replace("article", path),
                    backend="curl",
                    impersonate="chrome120",
                    headers={"User-Agent": "Canonical public test", "Accept": "text/html"},
                    max_response_bytes=8,
                    timeout=2,
                )
        else:
            response = hc.fetch(
                target_url.replace("article", path),
                backend="curl",
                impersonate="chrome120",
                headers={"User-Agent": "Canonical public test", "Accept": "text/html"},
                max_response_bytes=8,
                timeout=2,
            )
            assert response["text"] == "article"
            assert response["url"] == target_url
    assert requests
    for _, headers in requests:
        assert headers["Host"] == "public.example"
        assert headers["User-Agent"] == "Canonical public test"
        assert "sec-ch-ua" not in {key.lower() for key in headers}
        assert headers["Accept-Encoding"] == "identity"
        assert "Cookie" not in headers
        assert "Authorization" not in headers
    assert (curl.CurlOpt.PROXY, "") in options
    assert (curl.CurlOpt.NETRC, 0) in options
    assert (curl.CurlOpt.RESOLVE, [f"public.example:{443 if target_url.startswith('https') else 80}:{IP}"]) in options
    assert (curl.CurlOpt.TIMEOUT_MS, 2000) in options
    if target_url.startswith("https"):
        assert server_names and all(name == "public.example" for name in server_names)
    assert any(option == curl.CurlOpt.CAINFO and value == certifi.where() for option, value in options)
    assert handles and all(handle._curl is None for handle in handles)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["article", "redirect", "excess", "compressed"])
async def test_real_public_curl_probe_uses_bounded_native_adapter(curl_server, monkeypatch, path):
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_models

    monkeypatch.setattr(article_models, "DEFAULT_MAX_ARTICLE_BYTES", 8)
    guard = SimpleNamespace(decide=AsyncMock(return_value=SimpleNamespace(allowed=True, resolved_ips=(IP,))))
    transport = adapter.CurlCffiProbeTransport(egress_guard=guard, request_context=RuntimeRequestContext())
    target_url = curl_server[4]
    with egress.public_url_policy_scope():
        if path == "redirect":
            from tldw_Server_API.app.core.Web_Scraping.preflight.context import PreflightRuntimeControls

            probe = adapter.GuardedHttpProbe(
                controls=PreflightRuntimeControls(RuntimeRequestContext()), egress_guard=guard, curl_transport=transport
            )
            response = await probe.get(
                ProbeHttpRequest(target_url.replace("article", path), impersonate="chrome120", timeout_s=2)
            )
            assert response.text == "article"
            assert response.url == target_url
        elif path == "article":
            response = await transport.send(ProbeHttpRequest(target_url, impersonate="chrome120", timeout_s=2))
            assert response.text == "article"
            await response.aclose()
        else:
            with pytest.raises(ValueError, match="max_response_bytes"):
                await transport.send(
                    ProbeHttpRequest(target_url.replace("article", path), impersonate="chrome120", timeout_s=2)
                )
    curl, requests, options, handles, target_url, server_names = curl_server
    assert requests[0][1]["Accept-Encoding"] == "identity"
    assert all("Cookie" not in headers for _, headers in requests)
    assert (curl.CurlOpt.PROXY, "") in options
    assert (curl.CurlOpt.NETRC, 0) in options
    assert handles and all(handle._curl is None for handle in handles)


def test_public_httpx_reuses_checked_stream_and_shared_pin_cache(monkeypatch):
    observed = []
    original = hc.stream_response

    @contextmanager
    def checked(**kwargs):
        observed.append(kwargs)
        with original(**kwargs) as response:
            yield response

    def send(request):
        assert request.url.host == IP
        assert request.headers["Host"] == "public.example"
        return httpx.Response(200, stream=httpx.ByteStream(b"article"))

    monkeypatch.setattr(egress, "_resolve_host_ips", lambda *_a, **_k: [IP])
    monkeypatch.setattr(
        hc,
        "_resolve_httpx",
        lambda: SimpleNamespace(Client=lambda **kw: httpx.Client(transport=httpx.MockTransport(send), **kw)),
    )
    monkeypatch.setattr(hc, "stream_response", checked)
    with egress.public_url_policy_scope():
        assert hc.fetch(URL, max_response_bytes=8)["text"] == "article"
    assert len(observed) == 1
    assert observed[0]["client"].is_closed
    assert observed[0]["dns_pin_cache"] == {"public.example": (IP,)}


@pytest.mark.asyncio
async def test_public_external_process_fails_closed_without_transport_attestation():
    from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.external_tools import GuardedExternalToolProbe
    from tldw_Server_API.app.core.Web_Scraping.preflight.context import PreflightRuntimeControls
    from tldw_Server_API.app.core.Web_Scraping.preflight.probes import ProbeUnavailable

    process = AsyncMock(side_effect=AssertionError("Unattested subprocess must not start"))
    probe = GuardedExternalToolProbe(
        controls=PreflightRuntimeControls(RuntimeRequestContext()),
        egress_guard=SimpleNamespace(decide=AsyncMock(return_value=SimpleNamespace(allowed=True, reason="allowed"))),
        which=lambda _: "/controlled/wafw00f",
        process_factory=process,
    )
    with egress.public_url_policy_scope(), pytest.raises(ProbeUnavailable):
        await probe.run_waf(URL, find_all=False, enabled=True)
    process.assert_not_called()


def test_public_httpx_stream_rejects_failed_cookie_clear(monkeypatch):
    class Jar:
        def clear(self):
            raise RuntimeError("Cookie state could not be cleared")

    class Client:
        cookies = Jar()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def stream(self, *_args, **_kwargs):
            raise AssertionError("must not dispatch with retained cookies")

    monkeypatch.setattr(egress, "_resolve_host_ips", lambda *_a, **_k: [IP])
    monkeypatch.setattr(hc, "_resolve_httpx", lambda: SimpleNamespace(Client=lambda **_kw: Client()))
    with egress.public_url_policy_scope(), pytest.raises(RuntimeError, match="Cookie state"):
        hc.fetch(URL, max_response_bytes=8)


def test_public_httpx_factory_cannot_drop_trust_env(monkeypatch):
    calls = []

    def incompatible(**kwargs):
        calls.append(dict(kwargs))
        if "trust_env" in kwargs:
            raise TypeError("unexpected keyword argument 'trust_env'")
        return object()

    with egress.public_url_policy_scope(), pytest.raises(TypeError, match="trust_env"):
        hc._instantiate_client(incompatible, {"trust_env": False})
    assert calls == [{"trust_env": False}]


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://BÜCHER.example/path", ["xn--bcher-kva.example:443:93.184.216.34,[2001:4860:4860::8888]"]),
        ("https://public.example:8443/path", ["public.example:8443:93.184.216.34,[2001:4860:4860::8888]"]),
        ("https://[2001:4860:4860::8888]/path", []),
    ],
)
def test_shared_curl_pins_preserve_logical_idna_port_and_ipv6(url, expected):
    assert hc._curl_resolve_entries(url, (IP, "2001:4860:4860::8888")) == expected
    assert adapter._curl_resolve_entries(url, (IP, "2001:4860:4860::8888")) == expected
