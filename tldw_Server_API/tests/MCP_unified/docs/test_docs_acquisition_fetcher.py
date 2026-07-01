from __future__ import annotations

import gzip
from collections.abc import Iterable

from mcp_unified.docs.acquisition.fetcher import URLFetcher
from mcp_unified.docs.acquisition.models import FetchResponse, ResolvedAddress, URLRequest
from mcp_unified.docs.acquisition.policy import SourcePolicy
from mcp_unified.docs.settings import DocsSettings


class FakeResolver:
    def __init__(self, addresses: dict[str, list[str]]) -> None:
        self.addresses = addresses
        self.calls: list[tuple[str, int]] = []

    def resolve(self, host: str, port: int) -> Iterable[ResolvedAddress]:
        self.calls.append((host, port))
        return [ResolvedAddress(host=host, ip=ip, port=port) for ip in self.addresses[host]]


class FailingResolver(FakeResolver):
    def resolve(self, host: str, port: int) -> Iterable[ResolvedAddress]:
        self.calls.append((host, port))
        raise OSError("temporary DNS failure")


class FakeTransport:
    dials_validated_address = True

    def __init__(self, responses: list[FetchResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[ResolvedAddress, URLRequest, float]] = []

    def request(
        self,
        *,
        address: ResolvedAddress,
        request: URLRequest,
        timeout_seconds: float,
    ) -> FetchResponse:
        self.calls.append((address, request, timeout_seconds))
        return self.responses.pop(0)


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


def test_fetcher_preserves_query_in_request_target_without_result_leakage() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"ok"])]
    )

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/search?q=secret")

    assert transport.calls[0][1].target == "/search?q=secret"  # nosec B101
    assert result.final_url == "https://example.com/search"  # nosec B101


def test_fetcher_denies_private_ip_before_transport() -> None:
    resolver = FakeResolver({"internal.example": ["127.0.0.1"]})
    transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/html"}, body_chunks=[b"never"])]
    )

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://internal.example/docs")

    assert result.status == "denied"  # nosec B101
    assert result.reason == "egress_private_address_denied"  # nosec B101
    assert transport.calls == []  # nosec B101


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


def test_fetcher_respect_robots_fails_closed_before_resolver_or_transport() -> None:
    settings = _settings(respect_robots=True)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [FetchResponse(status_code=200, headers={"content-type": "text/html"}, body_chunks=[b"never"])]
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
