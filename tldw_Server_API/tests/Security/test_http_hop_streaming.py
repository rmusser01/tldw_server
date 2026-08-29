from __future__ import annotations

import asyncio
import gzip
import inspect
import logging
import zlib
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any

import httpcore
import pytest

from tldw_Server_API.app.core.Security import http_hop
from tldw_Server_API.tests.Security.test_http_hop_transport import (
    BlockingReadStream,
    DeterministicClock,
    RecordingBackend,
    RecordingStream,
)

pytestmark = pytest.mark.unit


class ReadRecordingStream(RecordingStream):
    """Record the raw read ceilings requested by the guarded stream."""

    def __init__(self, *, response: tuple[bytes, ...]) -> None:
        super().__init__(server_addr=("8.8.8.8", 80), response=response)
        self.read_sizes: list[int] = []

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        self.read_sizes.append(max_bytes)
        return await super().read(max_bytes, timeout=timeout)


class ErrorReadStream(RecordingStream):
    """Raise a supplied transport error without leaking its text."""

    def __init__(self, error: Exception) -> None:
        super().__init__(server_addr=("8.8.8.8", 80))
        self.error = error

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        del max_bytes, timeout
        raise self.error


class BodyReadCanaryError(Exception):
    """Unique evidence that status-only transport tried to read a body."""


class HeaderThenBodyCanaryStream(RecordingStream):
    """Return one complete header block and fail on every later read."""

    def __init__(self, head: bytes) -> None:
        super().__init__(server_addr=("8.8.8.8", 80), response=(head,))
        self.read_calls = 0

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        self.read_calls += 1
        if self.read_calls > 1:
            raise BodyReadCanaryError("status-only body iterator canary")
        return await super().read(max_bytes, timeout=timeout)


def _limits(**overrides: object) -> http_hop.HTTPHopLimits:
    return replace(http_hop.HTTPHopLimits(), **overrides)


def _request(**overrides: object) -> http_hop.NormalizedHTTPHopRequest:
    values: dict[str, object] = {
        "scheme": "http",
        "host": "api.example.com",
        "port": 80,
        "method": "GET",
        "target": "/works?q=bounded",
    }
    values.update(overrides)
    return http_hop.NormalizedHTTPHopRequest(**values)  # type: ignore[arg-type]


async def _execute(
    response: tuple[bytes, ...],
    *,
    request: http_hop.NormalizedHTTPHopRequest | None = None,
    stream: RecordingStream | None = None,
) -> tuple[http_hop.HTTPHopResponse, RecordingStream]:
    active_stream = stream or RecordingStream(
        server_addr=("8.8.8.8", 80),
        response=response,
    )
    result = await http_hop._execute_http_hop(
        request or _request(),
        resolved_ips=("8.8.8.8",),
        network_backend=RecordingBackend(active_stream),
    )
    return result, active_stream


async def _execute_error(
    response: tuple[bytes, ...],
    *,
    request: http_hop.NormalizedHTTPHopRequest | None = None,
    stream: RecordingStream | None = None,
) -> tuple[http_hop.HTTPHopError, RecordingStream]:
    active_stream = stream or RecordingStream(server_addr=("8.8.8.8", 80), response=response)
    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute(response, request=request, stream=active_stream)
    assert exc.value.__context__ is None
    assert len(str(exc.value)) <= 96
    assert active_stream.closed is True
    return exc.value, active_stream


async def _execute_status(
    response: tuple[bytes, ...],
    *,
    request: http_hop.NormalizedHTTPHopRequest | None = None,
    stream: RecordingStream | None = None,
    clock: DeterministicClock | None = None,
) -> tuple[http_hop.StatusOnlyHTTPHopResponse, RecordingStream]:
    active_stream = stream or RecordingStream(
        server_addr=("8.8.8.8", 80),
        response=response,
    )

    async def resolver(_host: str, _port: int, _timeout_seconds: float) -> Sequence[str]:
        return ("8.8.8.8",)

    result = await http_hop._request_http_hop_status(
        request or _request(),
        resolver=resolver,
        network_backend=RecordingBackend(active_stream),
        clock=clock or DeterministicClock(),
    )
    return result, active_stream


def _head(*headers: tuple[bytes, bytes], reason: bytes = b"OK") -> bytes:
    return (
        b"HTTP/1.1 200 "
        + reason
        + b"\r\n"
        + b"".join(name + b": " + value + b"\r\n" for name, value in headers)
        + b"\r\n"
    )


def _status_head(status: int, *headers: tuple[bytes, bytes]) -> bytes:
    return (
        b"HTTP/1.1 "
        + str(status).encode("ascii")
        + b" Status\r\n"
        + b"".join(name + b": " + value + b"\r\n" for name, value in headers)
        + b"\r\n"
    )


def _encoded_response(encoding: bytes, encoded: bytes) -> tuple[bytes, ...]:
    head = _head(
        (b"Content-Encoding", encoding),
        (b"Content-Length", str(len(encoded)).encode("ascii")),
        (b"Connection", b"close"),
    )
    return (head, *(bytes((value,)) for value in encoded))


def _raw_deflate(value: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(value) + compressor.flush()


def _header_values(request_bytes: bytes, name: bytes) -> list[bytes]:
    prefix = name.lower() + b":"
    return [line.split(b":", 1)[1].strip() for line in request_bytes.split(b"\r\n") if line.lower().startswith(prefix)]


def test_public_request_function_accepts_exactly_one_request_argument() -> None:
    signature = inspect.signature(http_hop.request_http_hop)

    assert tuple(signature.parameters) == ("request",)
    assert signature.parameters["request"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert "request_http_hop" in http_hop.__all__


@pytest.mark.parametrize(
    "headers",
    [
        ((b"Content-Length", b"999999999"),),
        ((b"Transfer-Encoding", b"chunked"),),
        ((b"Content-Encoding", b"gzip"), (b"Content-Length", b"999999999")),
    ],
    ids=("large-fixed", "chunked", "compressed"),
)
async def test_status_only_closes_without_reading_any_response_body(
    headers: tuple[tuple[bytes, bytes], ...],
) -> None:
    stream = HeaderThenBodyCanaryStream(
        _status_head(200, *headers, (b"Connection", b"close"))
    )

    response, active_stream = await _execute_status((), stream=stream)

    assert response.status_code == 200
    assert response.retry_after_seconds is None
    assert stream.read_calls == 1
    assert active_stream.closed is True


async def test_status_only_never_calls_bounded_body_or_header_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def body_canary(*_args: object, **_kwargs: object) -> bytes:
        raise BodyReadCanaryError("bounded body helper called")

    def headers_canary(*_args: object, **_kwargs: object) -> object:
        raise BodyReadCanaryError("ordinary headers exposed")

    monkeypatch.setattr(http_hop, "_read_decoded_body", body_canary)
    monkeypatch.setattr(http_hop, "_response_headers", headers_canary)
    raw = _status_head(
        429,
        (b"Retry-After", b"60"),
        (b"Content-Length", b"500000"),
        (b"Connection", b"close"),
    )

    response, stream = await _execute_status((raw,))

    assert response.retry_after_seconds == 60
    assert stream.closed is True


@pytest.mark.parametrize(
    ("status", "headers", "expected"),
    [
        (429, ((b"Retry-After", b"0"),), 1),
        (429, ((b"Retry-After", b"60"),), 60),
        (503, ((b"Retry-After", b"999999999999"),), 1_800),
        (503, ((b"Retry-After", b"Sun, 23 Aug 2026 00:02:00 GMT"),), 120),
        (503, ((b"Retry-After", b"Sat, 22 Aug 2026 23:59:00 GMT"),), 1),
        (500, ((b"Retry-After", b"300"),), None),
        (429, ((b"Retry-After", b"60"), (b"Retry-After", b"120")), None),
        (429, ((b"Retry-After", b"12x"),), None),
        (429, ((b"Retry-After", b"\xff"),), None),
        (503, ((b"Retry-After", b"Sun, 23 Aug 2026 00:02:00"),), None),
    ],
    ids=(
        "zero-clamps-up",
        "delta",
        "huge-clamps-down",
        "http-date",
        "past-date",
        "wrong-status",
        "duplicate",
        "malformed",
        "non-ascii",
        "naive-date",
    ),
)
async def test_status_only_retry_after_is_strict_bounded_and_status_scoped(
    status: int,
    headers: tuple[tuple[bytes, bytes], ...],
    expected: int | None,
) -> None:
    raw = _status_head(status, *headers, (b"Content-Length", b"0"))

    response, stream = await _execute_status((raw,))

    assert response.retry_after_seconds == expected
    assert stream.closed is True


async def test_status_only_header_limit_failure_closes_before_return() -> None:
    raw = _status_head(
        200,
        (b"X-First", b"one"),
        (b"X-Second", b"two"),
    )
    stream = RecordingStream(server_addr=("8.8.8.8", 80), response=(raw,))

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute_status(
            (),
            request=_request(limits=_limits(max_response_headers=1)),
            stream=stream,
        )

    assert exc.value.code == "response_headers_too_large"
    assert stream.closed is True


async def test_status_only_total_timeout_closes_blocked_header_read() -> None:
    stream = BlockingReadStream(server_addr=("8.8.8.8", 80))

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute_status(
            (),
            request=_request(limits=_limits(total_timeout_seconds=0.01)),
            stream=stream,
        )

    assert exc.value.code == "total_timeout"
    assert stream.closed is True


async def test_status_only_nonfinite_start_clock_fails_before_network_io() -> None:
    stream = RecordingStream(
        server_addr=("8.8.8.8", 80),
        response=(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",),
    )
    backend = RecordingBackend(stream)

    async def resolver(_host: str, _port: int, _timeout_seconds: float) -> Sequence[str]:
        return ("8.8.8.8",)

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await http_hop._request_http_hop_status(
            _request(),
            resolver=resolver,
            network_backend=backend,
            clock=DeterministicClock(monotonic_values=(float("nan"),)),
        )

    assert exc.value.code == "transport_error"
    assert backend.connect_calls == []


async def test_status_only_invalid_utc_clock_fails_closed_and_closes() -> None:
    raw = _status_head(
        503,
        (b"Retry-After", b"Sun, 23 Aug 2026 00:02:00 GMT"),
        (b"Content-Length", b"0"),
    )
    stream = RecordingStream(server_addr=("8.8.8.8", 80), response=(raw,))

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute_status(
            (),
            stream=stream,
            clock=DeterministicClock(utc_now=datetime(2026, 8, 23)),
        )

    assert exc.value.code == "transport_error"
    assert stream.closed is True


async def test_counts_all_informational_and_final_header_bytes_exactly() -> None:
    informational_100 = b"HTTP/1.1 100 Continue\r\nX-First: one\r\n\r\n"
    informational_103 = b"HTTP/1.1 103 Early Hints\r\nLink: </paper>; rel=preload\r\n\r\n"
    final = _head(
        (b"Content-Length", b"0"),
        (b"X-Final", b"yes"),
        reason=b"Deliberately Long Reason",
    )
    raw_headers = informational_100 + informational_103 + final
    request = _request(
        limits=_limits(
            max_response_header_bytes=len(raw_headers),
            max_response_headers=4,
        )
    )

    response, _ = await _execute(tuple(bytes((value,)) for value in raw_headers), request=request)

    assert response.status_code == 200
    assert response.body == b""
    assert response.response_header_bytes == len(raw_headers)
    assert response.wire_bytes == 0


async def test_rejects_aggregate_informational_header_count_overflow() -> None:
    raw = (
        b"HTTP/1.1 100 Continue\r\nX-First: one\r\n\r\n"
        b"HTTP/1.1 103 Early Hints\r\nLink: </paper>\r\n\r\n" + _head((b"Content-Length", b"0"), (b"X-Final", b"yes"))
    )
    request = _request(limits=_limits(max_response_headers=3))

    error, _ = await _execute_error((raw,), request=request)

    assert error.code == "response_headers_too_large"


async def test_rejects_aggregate_header_byte_overflow_at_exact_boundary() -> None:
    informational = b"HTTP/1.1 103 Early Hints\r\nLink: </paper>\r\n\r\n"
    final = _head((b"Content-Length", b"0"), reason=b"All Bytes Count")
    request = _request(limits=_limits(max_response_header_bytes=len(informational + final) - 1))

    error, _ = await _execute_error((informational, final), request=request)

    assert error.code == "response_headers_too_large"


async def test_header_overflow_is_stopped_before_httpcore_parses_it() -> None:
    stream = ReadRecordingStream(response=(b"not-http " + b"sensitive-upstream-detail" * 8,))
    request = _request(limits=_limits(max_response_header_bytes=32))

    error, _ = await _execute_error((), request=request, stream=stream)

    assert error.code == "response_headers_too_large"
    assert stream.read_sizes
    assert max(stream.read_sizes) <= 33
    assert "sensitive-upstream-detail" not in repr(error)


@pytest.mark.parametrize(
    "headers",
    [
        ((b"Content-Length", b""),),
        ((b"Content-Length", b"-1"),),
        ((b"Content-Length", b"two"),),
        ((b"Content-Length", b"2, 2"),),
        ((b"Content-Length", b"2"), (b"Content-Length", b"2")),
        ((b"Content-Length", b"2"), (b"Content-Length", b"3")),
    ],
    ids=("empty", "negative", "nonnumeric", "comma-list", "duplicate", "conflicting"),
)
async def test_rejects_malformed_or_duplicate_content_length(
    headers: tuple[tuple[bytes, bytes], ...],
) -> None:
    error, _ = await _execute_error((_head(*headers),))

    assert error.code == "protocol_error"


async def test_content_length_preflight_rejects_before_reading_body() -> None:
    head = _head((b"Content-Length", b"5"), (b"Connection", b"close"))
    stream = RecordingStream(
        server_addr=("8.8.8.8", 80),
        response=(head, b"abcde"),
    )
    request = _request(limits=_limits(max_wire_bytes=4))

    error, _ = await _execute_error((), request=request, stream=stream)

    assert error.code == "response_too_large"
    assert stream.response == [b"abcde"]


async def test_pathological_content_length_fails_as_bounded_protocol_error() -> None:
    head = _head((b"Content-Length", b"9" * 5000), (b"Connection", b"close"))
    request = _request(limits=_limits(max_wire_bytes=4))

    error, _ = await _execute_error((head,), request=request)

    assert error.code == "protocol_error"


async def test_rejects_transfer_encoding_with_content_length() -> None:
    head = _head(
        (b"Transfer-Encoding", b"chunked"),
        (b"Content-Length", b"0"),
        (b"Connection", b"close"),
    )

    error, _ = await _execute_error((head, b"0\r\n\r\n"))

    assert error.code == "protocol_error"


async def test_rejects_transfer_encoding_on_http_1_0_response() -> None:
    raw = b"HTTP/1.0 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n" b"2\r\nok\r\n0\r\n\r\n"

    error, _ = await _execute_error((raw,))

    assert error.code == "protocol_error"


@pytest.mark.parametrize(
    "framing_header",
    [(b"Content-Length", b"0"), (b"Transfer-Encoding", b"chunked")],
    ids=("content-length", "transfer-encoding"),
)
async def test_rejects_framing_headers_on_204_response(
    framing_header: tuple[bytes, bytes],
) -> None:
    raw = (
        b"HTTP/1.1 204 No Content\r\n"
        + framing_header[0]
        + b": "
        + framing_header[1]
        + b"\r\nConnection: close\r\n\r\n"
    )

    error, _ = await _execute_error((raw,))

    assert error.code == "protocol_error"


async def test_underreported_content_length_cannot_bypass_raw_wire_limit() -> None:
    request = _request(limits=_limits(max_wire_bytes=2))
    raw = _head((b"Content-Length", b"2"), (b"Connection", b"close")) + b"abc"

    error, _ = await _execute_error((raw,), request=request)

    assert error.code == "response_too_large"


async def test_content_length_wire_count_is_exact() -> None:
    head = _head((b"Content-Length", b"3"), (b"Connection", b"close"))
    request = _request(limits=_limits(max_wire_bytes=3))

    response, _ = await _execute((head + b"abc",), request=request)

    assert response.body == b"abc"
    assert response.response_header_bytes == len(head)
    assert response.wire_bytes == 3


async def test_chunked_wire_count_includes_framing_and_trailers() -> None:
    head = _head((b"Transfer-Encoding", b"chunked"), (b"Connection", b"close"))
    wire_body = b"3\r\nabc\r\n0\r\nX-Proof: yes\r\n\r\n"
    request = _request(limits=_limits(max_wire_bytes=len(wire_body)))

    response, _ = await _execute((head, wire_body), request=request)

    assert response.body == b"abc"
    assert response.wire_bytes == len(wire_body)


async def test_chunked_framing_or_trailers_can_overflow_raw_wire_limit() -> None:
    head = _head((b"Transfer-Encoding", b"chunked"), (b"Connection", b"close"))
    wire_body = b"1\r\na\r\n0\r\nX-Proof: yes\r\n\r\n"
    request = _request(limits=_limits(max_wire_bytes=len(wire_body) - 1))

    error, _ = await _execute_error((head, wire_body), request=request)

    assert error.code == "response_too_large"


async def test_eof_delimited_response_obeys_raw_wire_limit() -> None:
    head = _head((b"Connection", b"close"))
    request = _request(limits=_limits(max_wire_bytes=4))

    error, _ = await _execute_error((head, b"abcde"), request=request)

    assert error.code == "response_too_large"


async def test_wire_guard_never_requests_more_than_remaining_plus_one() -> None:
    head = _head((b"Connection", b"close"))
    stream = ReadRecordingStream(response=(head, b"abcde"))
    request = _request(limits=_limits(max_wire_bytes=4))

    error, _ = await _execute_error((), request=request, stream=stream)

    assert error.code == "response_too_large"
    assert stream.read_sizes[-1] <= 5


async def test_identity_body_has_independent_decompressed_limit() -> None:
    head = _head((b"Content-Length", b"5"), (b"Connection", b"close"))
    request = _request(
        limits=_limits(
            max_wire_bytes=5,
            max_decompressed_bytes=4,
            max_parser_input_bytes=4,
        )
    )

    error, _ = await _execute_error((head, b"abcde"), request=request)

    assert error.code == "decompressed_response_too_large"


async def test_identity_body_has_independent_parser_input_limit() -> None:
    head = _head((b"Content-Length", b"5"), (b"Connection", b"close"))
    request = _request(
        limits=_limits(
            max_wire_bytes=5,
            max_decompressed_bytes=16,
            max_parser_input_bytes=4,
        )
    )

    error, _ = await _execute_error((head, b"abcde"), request=request)

    assert error.code == "parser_input_too_large"


@pytest.mark.parametrize(
    ("encoding", "encoded"),
    [
        (b"gzip", gzip.compress(b"adversarial gzip chunks", mtime=0)),
        (b"deflate", zlib.compress(b"adversarial zlib chunks")),
    ],
    ids=("gzip", "zlib-deflate"),
)
async def test_decodes_supported_encoding_across_single_byte_chunks(
    encoding: bytes,
    encoded: bytes,
) -> None:
    expected = b"adversarial gzip chunks" if encoding == b"gzip" else b"adversarial zlib chunks"
    request = _request(limits=_limits(max_wire_bytes=len(encoded)))

    response, stream = await _execute(_encoded_response(encoding, encoded), request=request)

    assert response.body == expected
    assert response.wire_bytes == len(encoded)
    assert _header_values(b"".join(stream.writes), b"accept-encoding") == [b"gzip, deflate"]


@pytest.mark.parametrize(
    ("encoding", "encoded", "expected_code"),
    [
        (b"br", b"unsupported", "unsupported_content_encoding"),
        (b"gzip, deflate", gzip.compress(b"stacked", mtime=0), "unsupported_content_encoding"),
        (b"gzip", gzip.compress(b"truncated", mtime=0)[:-5], "invalid_content_encoding"),
        (
            b"gzip",
            gzip.compress(b"first", mtime=0) + gzip.compress(b"second", mtime=0),
            "invalid_content_encoding",
        ),
        (b"gzip", gzip.compress(b"payload", mtime=0) + b"trailing-data", "invalid_content_encoding"),
        (b"deflate", _raw_deflate(b"raw deflate is not accepted"), "invalid_content_encoding"),
    ],
    ids=("unknown", "stacked", "truncated", "concatenated", "trailing-data", "raw-deflate"),
)
async def test_rejects_unsupported_or_invalid_content_encoding(
    encoding: bytes,
    encoded: bytes,
    expected_code: str,
) -> None:
    request = _request(limits=_limits(max_wire_bytes=max(1, len(encoded))))

    error, _ = await _execute_error(_encoded_response(encoding, encoded), request=request)

    assert error.code == expected_code


async def test_rejects_stacked_content_encoding_split_across_headers() -> None:
    encoded = gzip.compress(b"stacked", mtime=0)
    head = _head(
        (b"Content-Encoding", b"gzip"),
        (b"Content-Encoding", b"deflate"),
        (b"Content-Length", str(len(encoded)).encode("ascii")),
        (b"Connection", b"close"),
    )

    error, _ = await _execute_error((head, encoded))

    assert error.code == "unsupported_content_encoding"


async def test_gzip_bomb_is_stopped_at_bounded_output() -> None:
    encoded = gzip.compress(b"x" * (1024 * 1024), mtime=0)
    request = _request(
        limits=_limits(
            max_wire_bytes=len(encoded),
            max_decompressed_bytes=64,
            max_parser_input_bytes=64,
        )
    )

    error, _ = await _execute_error(_encoded_response(b"gzip", encoded), request=request)

    assert error.code == "decompressed_response_too_large"


class _EmptyInputDrainDecoder:
    eof = False
    unused_data = b""
    unconsumed_tail = b""

    def __init__(self) -> None:
        self.decompress_limits: list[int] = []
        self.inputs: list[bytes] = []

    def decompress(self, data: bytes, max_length: int = 0) -> bytes:
        self.decompress_limits.append(max_length)
        self.inputs.append(data)
        if len(self.inputs) == 1:
            assert data == b"encoded"
            return b"body"
        assert data == b""
        self.eof = True
        return b"tail"


class _UnconsumedTailDecoder:
    eof = False
    unused_data = b""
    unconsumed_tail = b""

    def __init__(self) -> None:
        self.inputs: list[bytes] = []

    def decompress(self, data: bytes, max_length: int = 0) -> bytes:
        assert max_length > 0
        self.inputs.append(data)
        if len(self.inputs) == 1:
            self.unconsumed_tail = b"tail"
            return b"a"
        assert data == b"tail"
        self.unconsumed_tail = b""
        self.eof = True
        return b"b"


class _OverflowingDrainDecoder(_EmptyInputDrainDecoder):
    def decompress(self, data: bytes, max_length: int = 0) -> bytes:
        self.decompress_limits.append(max_length)
        self.inputs.append(data)
        if len(self.inputs) == 1:
            return b"body"
        return b"x" * max_length


@pytest.mark.parametrize(
    ("decoder_type", "expected"),
    [
        (_EmptyInputDrainDecoder, b"bodytail"),
        (_UnconsumedTailDecoder, b"ab"),
    ],
    ids=("empty-input-drain", "unconsumed-tail"),
)
async def test_decoder_handles_bounded_empty_input_drain_and_unconsumed_tail(
    monkeypatch: pytest.MonkeyPatch,
    decoder_type: type[Any],
    expected: bytes,
) -> None:
    decoder = decoder_type()
    monkeypatch.setattr(http_hop.zlib, "decompressobj", lambda _wbits: decoder)
    request = _request(
        limits=_limits(
            max_wire_bytes=7,
            max_decompressed_bytes=16,
            max_parser_input_bytes=16,
        )
    )

    response, _ = await _execute((b"".join(_encoded_response(b"gzip", b"encoded")),), request=request)

    assert response.body == expected
    if isinstance(decoder, _EmptyInputDrainDecoder):
        assert decoder.decompress_limits
        assert decoder.inputs == [b"encoded", b""]
        assert all(0 < value <= 17 for value in decoder.decompress_limits)
    else:
        assert decoder.inputs == [b"encoded", b"tail"]


async def test_decoder_rejects_output_beyond_remaining_empty_input_drain_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = _OverflowingDrainDecoder()
    monkeypatch.setattr(http_hop.zlib, "decompressobj", lambda _wbits: decoder)
    request = _request(
        limits=_limits(
            max_wire_bytes=7,
            max_decompressed_bytes=7,
            max_parser_input_bytes=7,
        )
    )

    error, _ = await _execute_error((b"".join(_encoded_response(b"gzip", b"encoded")),), request=request)

    assert error.code == "decompressed_response_too_large"
    assert decoder.decompress_limits == [8, 4]
    assert decoder.inputs == [b"encoded", b""]


async def test_protocol_and_decode_errors_are_sanitized_and_close_stream() -> None:
    protocol_stream = RecordingStream(
        server_addr=("8.8.8.8", 80),
        response=(b"HTTP/1.1 SECRET-UPSTREAM\r\n\r\n",),
    )
    protocol_error, _ = await _execute_error((), stream=protocol_stream)

    decode_stream = RecordingStream(
        server_addr=("8.8.8.8", 80),
        response=_encoded_response(b"gzip", b"token=SUPER-SECRET"),
    )
    decode_error, _ = await _execute_error((), stream=decode_stream)

    assert protocol_error.code == "protocol_error"
    assert decode_error.code == "invalid_content_encoding"
    for error in (protocol_error, decode_error):
        assert "SECRET" not in str(error)
        assert "SECRET" not in repr(error)


@pytest.mark.parametrize("status_code", [600, 999])
async def test_rejects_status_codes_outside_http_range(status_code: int) -> None:
    raw = f"HTTP/1.1 {status_code} Invalid\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode("ascii")

    error, _ = await _execute_error((raw,))

    assert error.code == "protocol_error"


async def test_accepts_empty_205_reset_content() -> None:
    raw = b"HTTP/1.1 205 Reset Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"

    response, _ = await _execute((raw,))

    assert response.status_code == 205
    assert response.body == b""
    assert response.wire_bytes == 0


async def test_accepts_empty_chunked_205_reset_content() -> None:
    raw = b"HTTP/1.1 205 Reset Content\r\n" b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n" b"0\r\n\r\n"

    response, _ = await _execute((raw,))

    assert response.status_code == 205
    assert response.body == b""
    assert response.wire_bytes == 5


async def test_chunked_205_rejects_bytes_after_empty_terminator() -> None:
    raw = b"HTTP/1.1 205 Reset Content\r\n" b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n" b"0\r\n\r\nEVIL"

    error, _ = await _execute_error((raw,))

    assert error.code == "protocol_error"


async def test_rejects_nonempty_205_reset_content() -> None:
    raw = b"HTTP/1.1 205 Reset Content\r\nContent-Length: 1\r\nConnection: close\r\n\r\nx"

    error, _ = await _execute_error((raw,))

    assert error.code == "protocol_error"


async def test_rejects_nonempty_chunked_205_reset_content() -> None:
    raw = (
        b"HTTP/1.1 205 Reset Content\r\n"
        b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        b"1\r\nx\r\n0\r\n\r\n"
    )

    error, _ = await _execute_error((raw,))

    assert error.code == "protocol_error"


async def test_head_205_rejects_bytes_after_header_terminator() -> None:
    raw = (
        b"HTTP/1.1 205 Reset Content\r\n"
        b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        b"arbitrary trailing bytes"
    )

    error, _ = await _execute_error((raw,), request=_request(method="HEAD"))

    assert error.code == "protocol_error"


async def test_app_logging_suppresses_httpcore_wire_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as app_main

    messages: list[str] = []
    httpcore_logger = logging.getLogger("httpcore")
    http11_logger = logging.getLogger("httpcore.http11")
    monkeypatch.setattr(httpcore_logger, "level", logging.DEBUG)
    monkeypatch.setattr(http11_logger, "level", logging.DEBUG)

    app_main._redirect_external_loggers()

    assert (httpcore_logger.level, http11_logger.level) == (logging.INFO, logging.INFO)

    def capture(_level: int, message: object, _args: object, *args: object, **kwargs: object) -> None:
        del args, kwargs
        messages.append(str(message))

    monkeypatch.setattr(http11_logger, "_log", capture)
    success = _head(
        (b"Set-Cookie", b"token=SUPER-SECRET"),
        (b"Content-Length", b"0"),
        (b"Connection", b"close"),
    )
    response, _ = await _execute((success,))
    head = _head((b"Transfer-Encoding", b"chunked"), (b"Connection", b"close"))
    error, _ = await _execute_error((head, b"SUPER-SECRET\r\n"))

    assert response.status_code == 200
    assert error.code == "protocol_error"
    assert http11_logger.isEnabledFor(logging.DEBUG) is False
    assert "SUPER-SECRET" not in "\n".join(messages)


async def test_read_timeout_is_sanitized_and_closes_stream() -> None:
    stream = ErrorReadStream(httpcore.ReadTimeout("secret idle-read detail"))

    error, _ = await _execute_error((), stream=stream)

    assert error.code == "read_timeout"
    assert error.retryable is True
    assert "secret idle-read detail" not in repr(error)


async def test_total_timeout_covers_resolution_before_connect() -> None:
    resolver_started = asyncio.Event()
    resolver_cancelled = asyncio.Event()

    async def resolver(_host: str, _port: int, _timeout: float) -> tuple[str, ...]:
        resolver_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            resolver_cancelled.set()
        return ("8.8.8.8",)

    stream = RecordingStream(server_addr=("8.8.8.8", 80))
    backend = RecordingBackend(stream)
    request = _request(
        limits=_limits(
            dns_timeout_seconds=1.0,
            total_timeout_seconds=0.02,
        )
    )

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await asyncio.wait_for(
            http_hop._request_http_hop(
                request,
                resolver=resolver,
                network_backend=backend,
            ),
            timeout=1.0,
        )

    assert exc.value.code == "total_timeout"
    assert resolver_started.is_set()
    assert resolver_cancelled.is_set()
    assert backend.connect_calls == []


async def test_total_timeout_covers_body_and_cleanup() -> None:
    async def resolver(_host: str, _port: int, _timeout: float) -> tuple[str, ...]:
        return ("8.8.8.8",)

    stream = BlockingReadStream(server_addr=("8.8.8.8", 80))
    backend = RecordingBackend(stream)
    request = _request(
        limits=_limits(
            read_timeout_seconds=1.0,
            total_timeout_seconds=0.02,
        )
    )

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await asyncio.wait_for(
            http_hop._request_http_hop(
                request,
                resolver=resolver,
                network_backend=backend,
            ),
            timeout=1.0,
        )

    assert exc.value.code == "total_timeout"
    assert stream.read_started.is_set()
    assert stream.closed is True
    assert len(backend.connect_calls) == 1
