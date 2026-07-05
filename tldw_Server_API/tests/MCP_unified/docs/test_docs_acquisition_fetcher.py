from __future__ import annotations

import gzip
import socket
import ssl
from collections.abc import Iterable

import pytest

from mcp_unified.docs.acquisition import fetcher as fetcher_module
from mcp_unified.docs.acquisition.fetcher import URLFetcher, ValidatedAddressHTTPTransport, _write_request
from mcp_unified.docs.acquisition.models import FetchResponse, NormalizedURL, ResolvedAddress, URLRequest
from mcp_unified.docs.acquisition.policy import SourcePolicy
from mcp_unified.docs.acquisition.resolver import StdlibResolver
from mcp_unified.docs.settings import DocsSettings
from tldw_Server_API.tests.MCP_unified.docs.helpers import FakeResolver, FakeTransport

pytestmark = pytest.mark.unit


class FailingResolver(FakeResolver):
    def resolve(self, host: str, port: int) -> Iterable[ResolvedAddress]:
        self.calls.append((host, port))
        raise OSError("temporary DNS failure")


class ReResolvingTransport(FakeTransport):
    dials_validated_address = False


class FailingTransport(FakeTransport):
    def request(
        self,
        *,
        address: ResolvedAddress,
        request: URLRequest,
        timeout_seconds: float,
    ) -> FetchResponse:
        self.calls.append((address, request, timeout_seconds))
        raise TimeoutError("request timed out")


class DummySocket:
    def __init__(self) -> None:
        self.timeout: float | None = None

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def __enter__(self) -> "DummySocket":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _settings(**overrides: object) -> DocsSettings:
    values: dict[str, object] = {
        "enable_web_acquisition": True,
        "web_source_profile": "online_capable",
        "allow_arbitrary_public_domains": True,
        "max_url_body_bytes": 32,
    }
    values.update(overrides)
    return DocsSettings.from_mapping(values)


def _policy(settings: DocsSettings) -> SourcePolicy:
    return SourcePolicy(
        web_source_profile=settings.web_source_profile,
        preapproved_domains=settings.preapproved_domains,
        allowed_url_prefixes=settings.allowed_url_prefixes,
        denied_domains=settings.denied_domains,
        allow_arbitrary_public_domains=settings.allow_arbitrary_public_domains,
    )


def _fetcher(
    *,
    resolver: FakeResolver,
    transport: FakeTransport,
    settings: DocsSettings | None = None,
) -> URLFetcher:
    actual_settings = settings or _settings()
    return URLFetcher(
        settings=actual_settings,
        policy=_policy(actual_settings),
        resolver=resolver,
        transport=transport,
    )


def test_fetcher_uses_validated_address_and_identity_encoding() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/html"}, body_chunks=[b"<h1>Ok</h1>"])]
    )

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/docs")

    assert result.status == "fetched"  # nosec B101
    assert transport.calls[0][0].ip == "93.184.216.34"  # nosec B101
    assert transport.calls[0][1].headers["accept-encoding"] == "identity"  # nosec B101
    assert result.body == b"<h1>Ok</h1>"  # nosec B101


def test_validated_transport_sets_tls12_minimum_for_https(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_socket = DummySocket()
    wrapped_socket = DummySocket()

    class DummySSLContext:
        def __init__(self) -> None:
            self.minimum_version: ssl.TLSVersion | None = None
            self.wrap_call: tuple[DummySocket, str] | None = None

        def wrap_socket(self, socket_to_wrap: DummySocket, *, server_hostname: str) -> DummySocket:
            self.wrap_call = (socket_to_wrap, server_hostname)
            return wrapped_socket

    context = DummySSLContext()

    def fake_create_connection(address: tuple[str, int], timeout: float) -> DummySocket:
        assert address == ("93.184.216.34", 443)  # nosec B101
        assert timeout == 5.0  # nosec B101
        return raw_socket

    monkeypatch.setattr(fetcher_module.socket, "create_connection", fake_create_connection)
    monkeypatch.setattr(fetcher_module.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(fetcher_module, "_write_request", lambda stream, request: None)
    monkeypatch.setattr(
        fetcher_module,
        "_read_response",
        lambda stream, max_body_bytes: (200, {"content-type": "text/plain"}, [b"ok"]),
    )

    response = ValidatedAddressHTTPTransport().request(
        address=ResolvedAddress(host="example.com", ip="93.184.216.34", port=443),
        request=URLRequest(
            normalized_url=NormalizedURL(
                scheme="https",
                host="example.com",
                port=None,
                path="/docs",
                decoded_path="/docs",
                redacted_url="https://example.com/docs",
            ),
            max_body_bytes=32,
            target="/docs",
        ),
        timeout_seconds=5.0,
    )

    assert context.minimum_version is ssl.TLSVersion.TLSv1_2  # nosec B101
    assert context.wrap_call == (raw_socket, "example.com")  # nosec B101
    assert response.status_code == 200  # nosec B101
    assert response.body_chunks == [b"ok"]  # nosec B101


def test_fetcher_preserves_query_in_request_target_without_result_leakage() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"ok"])]
    )

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/search?q=secret")

    assert transport.calls[0][1].target == "/search?q=secret"  # nosec B101
    assert result.final_url == "https://example.com/search"  # nosec B101
    assert result.canonical_url == "https://example.com/search?q=secret"  # nosec B101


def test_fetcher_denies_non_success_http_status_before_ingestion() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [FetchResponse(status_code=404, headers={"content-type": "text/html"}, body_chunks=[b"<h1>Not found</h1>"])]
    )

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/missing")

    assert result.status == "denied"  # nosec B101
    assert result.reason == "http_status_error"  # nosec B101
    assert result.status_code == 404  # nosec B101
    assert result.body == b""  # nosec B101


def test_fetcher_denies_private_ip_before_transport() -> None:
    resolver = FakeResolver({"internal.example": ["127.0.0.1"]})
    transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/html"}, body_chunks=[b"never"])]
    )

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://internal.example/docs")

    assert result.status == "denied"  # nosec B101
    assert result.reason == "egress_private_address_denied"  # nosec B101
    assert transport.calls == []  # nosec B101


def test_stdlib_resolver_populates_private_address_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_getaddrinfo(host: str, port: int, type: int):
        return [(socket.AF_INET, type, 0, "", ("127.0.0.1", port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    addresses = list(StdlibResolver().resolve("localhost", 443))

    assert len(addresses) == 1  # nosec B101
    assert addresses[0].ip == "127.0.0.1"  # nosec B101
    assert addresses[0].is_private is True  # nosec B101


def test_fetcher_returns_failed_when_resolver_fails() -> None:
    resolver = FailingResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/html"}, body_chunks=[b"never"])]
    )

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/docs")

    assert result.status == "failed"  # nosec B101
    assert result.reason == "dns_resolution_failed"  # nosec B101
    assert transport.calls == []  # nosec B101


def test_fetcher_denies_transport_that_cannot_prove_validated_address_binding() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = ReResolvingTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/html"}, body_chunks=[b"never"])]
    )

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/docs")

    assert result.status == "denied"  # nosec B101
    assert result.reason == "dns_rebinding_risk"  # nosec B101
    assert transport.calls == []  # nosec B101


def test_fetcher_returns_failed_when_transport_fails() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FailingTransport([])

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/docs")

    assert result.status == "failed"  # nosec B101
    assert result.reason == "fetch_error"  # nosec B101


def test_fetcher_returns_approval_required_without_resolver_or_transport() -> None:
    settings = _settings(web_source_profile="local_first", allow_arbitrary_public_domains=False)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/html"}, body_chunks=[b"never"])]
    )

    result = _fetcher(resolver=resolver, transport=transport, settings=settings).fetch("https://example.com/docs")

    assert result.status == "approval_required"  # nosec B101
    assert result.reason == "source_approval_required"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_fetcher_respect_robots_fails_closed_without_robots_client() -> None:
    settings = _settings(respect_robots=True)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/html"}, body_chunks=[b"<h1>Ok</h1>"])]
    )

    result = _fetcher(resolver=resolver, transport=transport, settings=settings).fetch("https://example.com/docs")

    assert result.status == "denied"  # nosec B101
    assert result.reason == "robots_unavailable"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_fetcher_revalidates_redirect_target_and_denies_private_redirect() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"], "internal.example": ["10.0.0.5"]})
    transport = FakeTransport(
        [FetchResponse(status_code=302, headers={"location": "https://internal.example/secret"}, body_chunks=[])]
    )

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/docs")

    assert result.status == "denied"  # nosec B101
    assert result.reason == "egress_private_address_denied"  # nosec B101
    assert len(transport.calls) == 1  # nosec B101


def test_fetcher_denies_redirect_target_that_requires_new_approval() -> None:
    settings = _settings(
        web_source_profile="local_first",
        preapproved_domains=("example.com",),
        allow_arbitrary_public_domains=False,
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"], "other.example": ["93.184.216.35"]})
    transport = FakeTransport(
        [FetchResponse(status_code=302, headers={"location": "https://other.example/docs"}, body_chunks=[])]
    )

    result = _fetcher(resolver=resolver, transport=transport, settings=settings).fetch("https://example.com/start")

    assert result.status == "denied"  # nosec B101
    assert result.reason == "redirect_policy_denied"  # nosec B101
    assert len(transport.calls) == 1  # nosec B101


def test_fetcher_enforces_redirect_limit() -> None:
    settings = _settings(max_url_redirects=1)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(status_code=302, headers={"location": "https://example.com/one"}, body_chunks=[]),
            FetchResponse(status_code=302, headers={"location": "https://example.com/two"}, body_chunks=[]),
        ]
    )

    result = _fetcher(resolver=resolver, transport=transport, settings=settings).fetch("https://example.com/start")

    assert result.status == "denied"  # nosec B101
    assert result.reason == "redirect_limit_exceeded"  # nosec B101


def test_fetcher_denies_content_type_before_body_is_returned() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(
                status_code=200, headers={"content-type": "application/octet-stream"}, body_chunks=[b"not read"]
            )
        ]
    )

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/archive.bin")

    assert result.status == "denied"  # nosec B101
    assert result.reason == "content_type_denied"  # nosec B101
    assert result.body == b""  # nosec B101


def test_fetcher_denies_missing_content_type_when_allowlist_is_configured() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(status_code=200, headers={}, body_chunks=[b"ambiguous"])])

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/no-content-type")

    assert result.status == "denied"  # nosec B101
    assert result.reason == "content_type_denied"  # nosec B101
    assert result.body == b""  # nosec B101


def test_fetcher_decodes_chunked_transfer_body() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(
                status_code=200,
                headers={"content-type": "text/plain", "transfer-encoding": "chunked"},
                body_chunks=[b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"],
            )
        ]
    )

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/chunked.txt")

    assert result.status == "fetched"  # nosec B101
    assert result.body == b"hello world"  # nosec B101


def test_fetcher_enforces_transferred_body_size_limit() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"a" * 33])]
    )

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/large.txt")

    assert result.status == "denied"  # nosec B101
    assert result.reason == "content_too_large"  # nosec B101


def test_fetcher_enforces_decoded_body_size_limit_for_gzip() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    compressed = gzip.compress(b"a" * 64)
    transport = FakeTransport(
        [
            FetchResponse(
                status_code=200,
                headers={"content-type": "text/plain", "content-encoding": "gzip"},
                body_chunks=[compressed],
            )
        ]
    )

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/compressed.txt")

    assert result.status == "denied"  # nosec B101
    assert result.reason == "content_too_large"  # nosec B101


def test_fetcher_reports_unsupported_content_encoding_separately() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(
                status_code=200,
                headers={"content-type": "text/plain", "content-encoding": "br"},
                body_chunks=[b"compressed"],
            )
        ]
    )

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/brotli.txt")

    assert result.status == "denied"  # nosec B101
    assert result.reason == "content_encoding_unsupported"  # nosec B101


def test_write_request_rejects_header_crlf_injection() -> None:
    class FakeStream:
        def sendall(self, payload: bytes) -> None:
            raise AssertionError(f"unexpected payload: {payload!r}")

    request = URLRequest(
        normalized_url=NormalizedURL(
            scheme="https",
            host="example.com",
            port=None,
            path="/docs",
            decoded_path="/docs",
            canonical_url="https://example.com/docs",
            redacted_url="https://example.com/docs",
        ),
        headers={"host": "example.com", "x-test": "ok\r\nInjected: yes"},
    )

    with pytest.raises(ValueError, match="header value"):
        _write_request(FakeStream(), request)  # type: ignore[arg-type]
