from __future__ import annotations

import socket
import ssl
import zlib
from collections.abc import Iterable, Mapping
from urllib.parse import urljoin, urlsplit

from ..settings import DocsSettings
from .models import FetchResponse, FetchResult, NormalizedURL, RedirectHop, ResolvedAddress, URLRequest
from .policy import SourcePolicy, URLPolicyError, normalize_url
from .resolver import StdlibResolver, is_unsafe_egress_ip

_DEFAULT_PORTS = {"http": 80, "https": 443}
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_MAX_STATUS_LINE_BYTES = 8192
_MAX_HEADER_BYTES = 65536


class URLFetcher:
    def __init__(
        self,
        *,
        settings: DocsSettings,
        policy: SourcePolicy,
        resolver: object | None = None,
        transport: object | None = None,
    ) -> None:
        self.settings = settings
        self.policy = policy
        self.resolver = resolver or StdlibResolver()
        self.transport = transport or ValidatedAddressHTTPTransport()

    def fetch(self, raw_url: str) -> FetchResult:
        current_url = raw_url
        redirects: list[RedirectHop] = []
        while True:
            decision = self.policy.evaluate(current_url)
            if decision.status != "allowed":
                return FetchResult(
                    status=decision.status,
                    reason=decision.reason,
                    final_url=decision.redacted_url,
                    redirects=tuple(redirects),
                    safe_argument_hash=decision.safe_argument_hash,
                )
            if decision.normalized_url is None:
                return FetchResult(status="denied", reason="malformed_url", redirects=tuple(redirects))
            try:
                normalized = normalize_url(current_url)
            except URLPolicyError:
                return FetchResult(status="denied", reason="malformed_url", redirects=tuple(redirects))
            if self.settings.respect_robots:
                return FetchResult(
                    status="denied",
                    reason="robots_unavailable",
                    final_url=normalized.redacted_url,
                    redirects=tuple(redirects),
                    safe_argument_hash=decision.safe_argument_hash,
                )
            if not bool(getattr(self.transport, "dials_validated_address", False)):
                return FetchResult(
                    status="denied",
                    reason="dns_rebinding_risk",
                    final_url=normalized.redacted_url,
                    redirects=tuple(redirects),
                )

            port = _request_port(normalized)
            try:
                addresses = list(self.resolver.resolve(normalized.host, port))
            except OSError:
                return FetchResult(
                    status="failed",
                    reason="dns_resolution_failed",
                    final_url=normalized.redacted_url,
                    redirects=tuple(redirects),
                )
            if not addresses:
                return FetchResult(
                    status="failed",
                    reason="dns_resolution_failed",
                    final_url=normalized.redacted_url,
                    redirects=tuple(redirects),
                )
            unsafe_address = _first_unsafe_address(addresses)
            if unsafe_address is not None:
                return FetchResult(
                    status="denied",
                    reason="egress_private_address_denied",
                    final_url=normalized.redacted_url,
                    redirects=tuple(redirects),
                )

            request = URLRequest(
                normalized_url=normalized,
                headers=_request_headers(settings=self.settings, normalized=normalized, port=port),
                max_body_bytes=self.settings.max_url_body_bytes,
                target=_request_target(current_url=current_url, normalized=normalized),
            )
            try:
                response = self.transport.request(
                    address=addresses[0],
                    request=request,
                    timeout_seconds=self.settings.url_request_timeout_seconds,
                )
            except (OSError, ValueError):
                return FetchResult(
                    status="failed",
                    reason="fetch_error",
                    final_url=normalized.redacted_url,
                    redirects=tuple(redirects),
                )
            headers = _normalize_headers(response.headers)

            if response.status_code in _REDIRECT_STATUSES:
                if len(redirects) >= self.settings.max_url_redirects:
                    return FetchResult(
                        status="denied",
                        reason="redirect_limit_exceeded",
                        final_url=normalized.redacted_url,
                        status_code=response.status_code,
                        headers=headers,
                        redirects=tuple(redirects),
                    )
                location = headers.get("location")
                if not location:
                    return FetchResult(
                        status="failed",
                        reason="redirect_location_missing",
                        final_url=normalized.redacted_url,
                        status_code=response.status_code,
                        headers=headers,
                        redirects=tuple(redirects),
                    )
                next_url = urljoin(normalized.canonical_url, location)
                redirect_decision = self.policy.evaluate(next_url)
                if redirect_decision.status != "allowed" or redirect_decision.normalized_url is None:
                    return FetchResult(
                        status="denied",
                        reason="redirect_policy_denied",
                        final_url=normalized.redacted_url,
                        status_code=response.status_code,
                        headers=headers,
                        redirects=tuple(redirects),
                        safe_argument_hash=redirect_decision.safe_argument_hash,
                    )
                redirects.append(
                    RedirectHop(
                        from_url=normalized.redacted_url,
                        to_url=redirect_decision.normalized_url.redacted_url,
                        status_code=response.status_code,
                    )
                )
                current_url = next_url
                continue

            if response.status_code < 200 or response.status_code >= 300:
                return FetchResult(
                    status="denied",
                    reason="http_status_error",
                    final_url=normalized.redacted_url,
                    canonical_url=normalized.canonical_url,
                    status_code=response.status_code,
                    headers=headers,
                    redirects=tuple(redirects),
                )

            if not _content_type_allowed(headers, self.settings.allowed_content_types):
                return FetchResult(
                    status="denied",
                    reason="content_type_denied",
                    final_url=normalized.redacted_url,
                    canonical_url=normalized.canonical_url,
                    status_code=response.status_code,
                    headers=headers,
                    redirects=tuple(redirects),
                )

            body = _join_limited(response.body_chunks, self.settings.max_url_body_bytes)
            if body is None:
                return FetchResult(
                    status="denied",
                    reason="content_too_large",
                    final_url=normalized.redacted_url,
                    status_code=response.status_code,
                    headers=headers,
                    redirects=tuple(redirects),
                )
            body, transfer_reason = _decode_transfer_limited(
                body,
                headers.get("transfer-encoding"),
                self.settings.max_url_body_bytes,
            )
            if body is None:
                return FetchResult(
                    status="denied",
                    reason=transfer_reason or "transfer_encoding_unsupported",
                    final_url=normalized.redacted_url,
                    status_code=response.status_code,
                    headers=headers,
                    redirects=tuple(redirects),
                )
            decoded_body, decode_reason = _decode_limited(
                body,
                headers.get("content-encoding"),
                self.settings.max_url_body_bytes,
            )
            if decoded_body is None:
                return FetchResult(
                    status="denied",
                    reason=decode_reason or "content_too_large",
                    final_url=normalized.redacted_url,
                    status_code=response.status_code,
                    headers=headers,
                    redirects=tuple(redirects),
                )
            return FetchResult(
                status="fetched",
                reason="ok",
                final_url=normalized.redacted_url,
                canonical_url=normalized.canonical_url,
                status_code=response.status_code,
                headers=headers,
                body=decoded_body,
                redirects=tuple(redirects),
            )


class ValidatedAddressHTTPTransport:
    dials_validated_address = True

    def request(
        self,
        *,
        address: ResolvedAddress,
        request: URLRequest,
        timeout_seconds: float,
    ) -> FetchResponse:
        with socket.create_connection((address.ip, address.port), timeout=timeout_seconds) as raw_socket:
            raw_socket.settimeout(timeout_seconds)
            stream = raw_socket
            if request.normalized_url.scheme == "https":
                context = ssl.create_default_context()
                context.minimum_version = ssl.TLSVersion.TLSv1_2
                stream = context.wrap_socket(raw_socket, server_hostname=request.normalized_url.host)
            with stream:
                _write_request(stream, request)
                status_code, headers, body_chunks = _read_response(
                    stream,
                    max_body_bytes=request.max_body_bytes,
                )
        return FetchResponse(status_code=status_code, headers=headers, body_chunks=body_chunks)


def _request_port(normalized: NormalizedURL) -> int:
    return normalized.port or _DEFAULT_PORTS[normalized.scheme]


def _request_headers(*, settings: DocsSettings, normalized: NormalizedURL, port: int) -> dict[str, str]:
    return {
        "host": _host_header(normalized, port),
        "user-agent": settings.url_user_agent,
        "accept": ", ".join(settings.allowed_content_types),
        "accept-encoding": "identity",
        "connection": "close",
    }


def _request_target(*, current_url: str, normalized: NormalizedURL) -> str:
    query = urlsplit(current_url).query
    path = normalized.path or "/"
    return f"{path}?{query}" if query else path


def _host_header(normalized: NormalizedURL, port: int) -> str:
    host = normalized.host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    default_port = _DEFAULT_PORTS[normalized.scheme]
    return host if port == default_port else f"{host}:{port}"


def _first_unsafe_address(addresses: Iterable[ResolvedAddress]) -> ResolvedAddress | None:
    for address in addresses:
        if is_unsafe_egress_ip(address.ip):
            return address
    return None


def _normalize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value).strip() for key, value in headers.items()}


def _content_type_allowed(headers: Mapping[str, str], allowed_content_types: tuple[str, ...]) -> bool:
    if not allowed_content_types:
        return True
    content_type = headers.get("content-type")
    if not content_type:
        return False
    media_type = content_type.split(";", 1)[0].strip().lower()
    allowed = {item.split(";", 1)[0].strip().lower() for item in allowed_content_types}
    return media_type in allowed


def _join_limited(chunks: Iterable[bytes], max_bytes: int) -> bytes | None:
    total = 0
    body = bytearray()
    for chunk in chunks:
        total += len(chunk)
        if total > max_bytes:
            return None
        body.extend(chunk)
    return bytes(body)


def _decode_transfer_limited(
    body: bytes,
    transfer_encoding: str | None,
    max_bytes: int,
) -> tuple[bytes | None, str | None]:
    encodings = [item.strip().lower() for item in (transfer_encoding or "").split(",") if item.strip()]
    if not encodings or encodings == ["identity"]:
        return body, None
    if encodings == ["chunked"]:
        return _decode_chunked_limited(body, max_bytes)
    return None, "transfer_encoding_unsupported"


def _decode_chunked_limited(body: bytes, max_bytes: int) -> tuple[bytes | None, str | None]:
    decoded = bytearray()
    position = 0
    while True:
        line_end = body.find(b"\r\n", position)
        if line_end < 0:
            return None, "transfer_encoding_unsupported"
        size_text = body[position:line_end].split(b";", 1)[0].strip()
        try:
            chunk_size = int(size_text, 16)
        except ValueError:
            return None, "transfer_encoding_unsupported"
        position = line_end + 2
        if chunk_size == 0:
            return bytes(decoded), None
        chunk_end = position + chunk_size
        if chunk_end > len(body):
            return None, "transfer_encoding_unsupported"
        if len(decoded) + chunk_size > max_bytes:
            return None, "content_too_large"
        decoded.extend(body[position:chunk_end])
        position = chunk_end
        if body[position : position + 2] != b"\r\n":
            return None, "transfer_encoding_unsupported"
        position += 2


def _decode_limited(body: bytes, content_encoding: str | None, max_bytes: int) -> tuple[bytes | None, str | None]:
    encoding = (content_encoding or "identity").split(",", 1)[0].strip().lower()
    try:
        if encoding in {"", "identity"}:
            decoded = body
        elif encoding == "gzip":
            decoded = _zlib_decode_limited(body, 16 + zlib.MAX_WBITS, max_bytes)
        elif encoding == "deflate":
            decoded = _zlib_decode_limited(body, zlib.MAX_WBITS, max_bytes)
        else:
            return None, "content_encoding_unsupported"
    except (OSError, EOFError, zlib.error):
        return None, "content_encoding_unsupported"
    if decoded is None:
        return None, "content_too_large"
    if len(decoded) > max_bytes:
        return None, "content_too_large"
    return decoded, None


def _zlib_decode_limited(body: bytes, window_bits: int, max_bytes: int) -> bytes | None:
    decompressor = zlib.decompressobj(window_bits)
    decoded = bytearray()
    for offset in range(0, len(body), 8192):
        remaining = max_bytes + 1 - len(decoded)
        if remaining <= 0:
            return None
        chunk = decompressor.decompress(body[offset : offset + 8192], remaining)
        decoded.extend(chunk)
        if len(decoded) > max_bytes or decompressor.unconsumed_tail:
            return None
    remaining = max_bytes + 1 - len(decoded)
    if remaining <= 0:
        return None
    decoded.extend(decompressor.flush(remaining))
    if len(decoded) > max_bytes:
        return None
    return bytes(decoded)


def _write_request(stream: socket.socket | ssl.SSLSocket, request: URLRequest) -> None:
    target = request.target or request.normalized_url.path or "/"
    if "\r" in target or "\n" in target:
        raise ValueError("request target contains a newline")
    header_lines = [f"GET {target} HTTP/1.1"]
    for key, value in request.headers.items():
        _validate_header(key, value)
        header_lines.append(f"{key}: {value}")
    payload = "\r\n".join(header_lines) + "\r\n\r\n"
    stream.sendall(payload.encode("iso-8859-1"))


def _validate_header(key: str, value: str) -> None:
    key_text = str(key)
    value_text = str(value)
    if not key_text or any(ch in key_text for ch in ":\r\n"):
        raise ValueError("invalid HTTP header name")
    if any(ord(ch) < 33 or ord(ch) > 126 for ch in key_text):
        raise ValueError("invalid HTTP header name")
    if "\r" in value_text or "\n" in value_text:
        raise ValueError("invalid HTTP header value")


def _read_response(
    stream: socket.socket | ssl.SSLSocket,
    *,
    max_body_bytes: int | None,
) -> tuple[int, dict[str, str], list[bytes]]:
    header_buffer = bytearray()
    while b"\r\n\r\n" not in header_buffer:
        chunk = stream.recv(4096)
        if not chunk:
            break
        header_buffer.extend(chunk)
        if len(header_buffer) > _MAX_HEADER_BYTES:
            raise ValueError("response headers too large")
    header_bytes, separator, remainder = bytes(header_buffer).partition(b"\r\n\r\n")
    if not separator:
        raise ValueError("malformed HTTP response")
    status_code, headers = _parse_headers(header_bytes)
    body_chunks = _read_body_chunks(stream, remainder, max_body_bytes=max_body_bytes)
    return status_code, headers, body_chunks


def _parse_headers(header_bytes: bytes) -> tuple[int, dict[str, str]]:
    lines = header_bytes.split(b"\r\n")
    if not lines or len(lines[0]) > _MAX_STATUS_LINE_BYTES:
        raise ValueError("malformed HTTP status line")
    status_parts = lines[0].decode("iso-8859-1").split()
    if len(status_parts) < 2:
        raise ValueError("malformed HTTP status line")
    status_code = int(status_parts[1])
    headers: dict[str, str] = {}
    for raw_line in lines[1:]:
        if b":" not in raw_line:
            continue
        key, value = raw_line.split(b":", 1)
        headers[key.decode("iso-8859-1").strip().lower()] = value.decode("iso-8859-1").strip()
    return status_code, headers


def _read_body_chunks(
    stream: socket.socket | ssl.SSLSocket,
    initial: bytes,
    *,
    max_body_bytes: int | None,
) -> list[bytes]:
    body_chunks: list[bytes] = []
    total = 0
    cap = max_body_bytes + 1 if max_body_bytes is not None else None
    if initial:
        body_chunks.append(initial)
        total += len(initial)
    while cap is None or total < cap:
        chunk = stream.recv(8192)
        if not chunk:
            break
        body_chunks.append(chunk)
        total += len(chunk)
    return body_chunks


__all__ = ["URLFetcher", "ValidatedAddressHTTPTransport"]
