from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from typing import Any

import pytest

from tldw_Server_API.app.core.Security import standalone_html_request_guard as guard_module
from tldw_Server_API.app.core.Security.standalone_html_request_guard import (
    GENERATION_MAX_REQUEST_BYTES,
    RAW_HTML_MAX_REQUEST_BYTES,
    ShallowStandaloneFieldScanner,
    StandaloneHtmlRequestGuardMiddleware,
    match_standalone_request_route,
)

GENERATION_PATH = "/api/v1/slides/generations"
SAVE_PATH = "/api/v1/slides/presentations/deck-1/html-source"
DRAFT_PATH = "/api/v1/slides/presentations/deck-1/draft-attachment"


def _scope(
    path: str,
    *,
    method: str = "POST",
    headers: Iterable[tuple[bytes, bytes]] = (),
) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": list(headers),
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


async def _invoke(
    path: str,
    events: list[dict[str, Any]],
    *,
    method: str = "POST",
    headers: Iterable[tuple[bytes, bytes]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    delivered: list[dict[str, Any]] = []
    sent: list[dict[str, Any]] = []
    pending = list(events)

    async def receive() -> dict[str, Any]:
        if pending:
            return pending.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def app(_scope: dict[str, Any], guarded_receive, guarded_send) -> None:
        while True:
            event = await guarded_receive()
            delivered.append(event)
            if event["type"] != "http.request" or not event.get("more_body", False):
                break
        body = b"".join(event.get("body", b"") for event in delivered)
        response = json.dumps({"size": len(body)}, separators=(",", ":")).encode()
        await guarded_send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await guarded_send({"type": "http.response.body", "body": response})

    middleware = StandaloneHtmlRequestGuardMiddleware(app)
    await middleware(_scope(path, method=method, headers=headers), receive, send)
    return delivered, sent, pending


def _request_events(*chunks: bytes) -> list[dict[str, Any]]:
    return [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    ]


def _status_and_body(sent: list[dict[str, Any]]) -> tuple[int, bytes]:
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    return start["status"], body


def test_route_matcher_exposes_fixed_task_11_contracts_without_business_routes() -> None:
    generation = match_standalone_request_route("POST", GENERATION_PATH)
    save = match_standalone_request_route("PUT", SAVE_PATH)
    draft = match_standalone_request_route("POST", DRAFT_PATH)

    assert generation is not None and generation.max_bytes == 4 * 1024 * 1024
    assert generation.strict_json is True
    assert save is not None and save.max_bytes == 1_048_576
    assert draft is not None and draft.max_bytes == 1_048_576
    assert match_standalone_request_route("GET", GENERATION_PATH) is None


def test_source_response_route_is_marked_sensitive_for_body_capture_integrations() -> None:
    observed: dict[str, Any] = {}

    async def app(scope: dict[str, Any], _receive, _send) -> None:
        observed.update(scope)

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message: dict[str, Any]) -> None:
        return None

    middleware = StandaloneHtmlRequestGuardMiddleware(app)
    asyncio.run(
        middleware(
            _scope("/api/v1/slides/presentations/deck-1", method="GET"),
            receive,
            send,
        )
    )

    assert observed["standalone_html_sensitive"] is True


@pytest.mark.parametrize("path,method", [(GENERATION_PATH, "POST"), (SAVE_PATH, "PUT"), (DRAFT_PATH, "POST")])
@pytest.mark.parametrize("encoding", [None, b"identity", b" Identity "])
def test_fixed_routes_accept_absent_or_single_identity_content_encoding(
    path: str,
    method: str,
    encoding: bytes | None,
) -> None:
    body = b"{}" if path == GENERATION_PATH else b"draft"
    headers = [] if encoding is None else [(b"content-encoding", encoding)]
    delivered, sent, _ = asyncio.run(_invoke(path, _request_events(body), method=method, headers=headers))

    assert _status_and_body(sent)[0] == 200
    assert b"".join(event.get("body", b"") for event in delivered) == body


@pytest.mark.parametrize(
    "headers",
    [
        [(b"content-encoding", b"gzip")],
        [(b"content-encoding", b"identity,gzip")],
        [(b"content-encoding", b"identity"), (b"content-encoding", b"identity")],
    ],
)
def test_fixed_routes_reject_compressed_multiple_or_conflicting_content_encoding(
    headers: list[tuple[bytes, bytes]],
) -> None:
    _, sent, _ = asyncio.run(_invoke(GENERATION_PATH, _request_events(b"{}"), headers=headers))

    status, response_body = _status_and_body(sent)
    assert status == 415
    assert json.loads(response_body) == {"detail": "standalone_html_unsupported_encoding"}


@pytest.mark.parametrize(
    "headers",
    [
        [(b"content-length", b"2"), (b"content-length", b"2")],
        [(b"content-length", b"2,2")],
        [(b"content-length", b"-1")],
        [(b"content-length", b"2.0")],
        [(b"content-length", b"2"), (b"transfer-encoding", b"chunked")],
    ],
)
def test_fixed_routes_reject_ambiguous_or_invalid_declared_length(
    headers: list[tuple[bytes, bytes]],
) -> None:
    _, sent, _ = asyncio.run(_invoke(GENERATION_PATH, _request_events(b"{}"), headers=headers))

    assert _status_and_body(sent) == (400, b'{"detail":"invalid_content_length"}')


@pytest.mark.parametrize(
    "path,method,limit",
    [
        (GENERATION_PATH, "POST", GENERATION_MAX_REQUEST_BYTES),
        (SAVE_PATH, "PUT", RAW_HTML_MAX_REQUEST_BYTES),
        (DRAFT_PATH, "POST", RAW_HTML_MAX_REQUEST_BYTES),
    ],
)
def test_fixed_routes_reject_declared_max_plus_one_before_downstream_reads(
    path: str,
    method: str,
    limit: int,
) -> None:
    delivered, sent, pending = asyncio.run(
        _invoke(
            path,
            _request_events(b"never delivered"),
            method=method,
            headers=[(b"content-length", str(limit + 1).encode())],
        )
    )

    assert _status_and_body(sent)[0] == 413
    assert delivered == []
    assert pending == []


@pytest.mark.parametrize("headers", [[], [(b"transfer-encoding", b"chunked")]])
def test_missing_or_chunked_length_is_counted_and_allowed_within_limit(
    headers: list[tuple[bytes, bytes]],
) -> None:
    events = _request_events(b"{", b"}")
    delivered, sent, _ = asyncio.run(_invoke(GENERATION_PATH, events, headers=headers))

    assert _status_and_body(sent)[0] == 200
    assert b"".join(event.get("body", b"") for event in delivered) == b"{}"


def test_dishonest_declared_length_is_rejected_after_bounded_receive() -> None:
    delivered, sent, _ = asyncio.run(
        _invoke(GENERATION_PATH, _request_events(b"{}"), headers=[(b"content-length", b"1")])
    )

    assert _status_and_body(sent) == (400, b'{"detail":"invalid_content_length"}')
    assert delivered == []


def test_fixed_route_stops_buffering_at_first_over_limit_chunk_and_drains_remaining_events() -> None:
    sentinel = b"SECRET-SOURCE-MUST-NOT-BUFFER"
    events = _request_events(b"a" * RAW_HTML_MAX_REQUEST_BYTES, b"x", sentinel)
    delivered, sent, pending = asyncio.run(_invoke(DRAFT_PATH, events))

    assert _status_and_body(sent) == (413, b'{"detail":"standalone_html_storage_limit"}')
    assert delivered == []
    assert pending == []
    assert sentinel not in b"".join(message.get("body", b"") for message in sent)


def test_generation_preflight_accepts_utf8_code_points_and_escapes_split_across_events() -> None:
    body = '{"source":{"prompt":"snowman ☃ and \\u2603"}}'.encode()
    chunks = [body[index : index + 1] for index in range(len(body))]

    delivered, sent, _ = asyncio.run(_invoke(GENERATION_PATH, _request_events(*chunks)))

    assert _status_and_body(sent)[0] == 200
    assert b"".join(event.get("body", b"") for event in delivered) == body


@pytest.mark.parametrize(
    "body",
    [
        b'{"source":{},"source":{}}',
        b'{"number":NaN}',
        b'{"number":Infinity}',
        b'{"source":{"prompt":"\\ud800"}}',
        b'{"source":',
        b'{"source" 1}',
        b"\xff",
    ],
)
def test_strict_generation_json_rejects_duplicate_nonfinite_surrogate_or_malformed_input(body: bytes) -> None:
    _, sent, _ = asyncio.run(_invoke(GENERATION_PATH, _request_events(body)))

    assert _status_and_body(sent) == (422, b'{"detail":"standalone_html_request_invalid"}')


@pytest.mark.parametrize(
    "body",
    [
        b"[" * 9 + b"0" + b"]" * 9,
        b"[" + b"[]," * 512 + b"[]]",
        b"[" + b"0," * 2048 + b"0]",
        b'"' + b"a" * (3 * 1024 * 1024 + 1) + b'"',
    ],
)
def test_generation_json_structure_budgets_use_exact_redacted_error(body: bytes) -> None:
    _, sent, _ = asyncio.run(_invoke(GENERATION_PATH, _request_events(body)))

    assert _status_and_body(sent) == (422, b'{"detail":"json_structure_too_complex"}')


def test_generation_json_token_budget_uses_exact_redacted_error() -> None:
    body = b"[" + b"0," * 2048 + b"0]"
    _, sent, _ = asyncio.run(_invoke(GENERATION_PATH, _request_events(body)))

    assert _status_and_body(sent) == (422, b'{"detail":"json_structure_too_complex"}')


def test_fixed_route_error_never_contains_source_or_parser_excerpt() -> None:
    sentinel = "PRIVATE_SOURCE_8f2e9e"
    body = json.dumps({"source": {"prompt": sentinel}, "number": float("nan")}).encode()

    _, sent, _ = asyncio.run(_invoke(GENERATION_PATH, _request_events(body)))

    status, response_body = _status_and_body(sent)
    assert status == 422
    assert len(response_body) < 128
    assert sentinel.encode() not in response_body


@pytest.mark.parametrize(
    "mode,payload",
    [
        ("rest", b'{"html\\u005fdocument":"SECRET"}'),
        ("rest", b'{"content_kind":"standalone_html"}'),
        ("mcp", b'{"params":{"arguments":{"html_document":"SECRET"}}}'),
        ("mcp", b'{"arguments":{"updates":{"content_kind":"standalone_html"}}}'),
        ("mcp", b'{"arguments":{"patch":{"html\\u005fdocument":"SECRET"}}}'),
    ],
)
def test_shallow_scanner_rejects_only_schema_path_fields_with_escaped_key_decoding(
    mode: str,
    payload: bytes,
) -> None:
    scanner = ShallowStandaloneFieldScanner(mode=mode)

    with pytest.raises(Exception) as rejected:
        for byte in payload:
            scanner.feed(bytes([byte]))

    assert getattr(rejected.value, "status_code", None) == 409
    assert getattr(rejected.value, "code", None) == "standalone_html_creation_requires_generation"
    assert scanner.retained_bytes <= 64


@pytest.mark.parametrize(
    "mode,payload",
    [
        ("rest", b'{"title":"html_document","nested":{"html_document":"legal"}}'),
        ("rest", b'{"content_kind":"structured_slides"}'),
        ("mcp", b'{"params":{"html_document":"legal outside arguments"}}'),
        ("mcp", b'{"params":{"arguments":{"title":"content_kind: standalone_html"}}}'),
        ("mcp", b'{"arguments":{"deep":{"html_document":"legal"}}}'),
    ],
)
def test_shallow_scanner_distinguishes_keys_strings_and_exact_paths(mode: str, payload: bytes) -> None:
    scanner = ShallowStandaloneFieldScanner(mode=mode)

    for byte in payload:
        scanner.feed(bytes([byte]))
    scanner.finish()

    assert scanner.retained_bytes <= 64


def test_generic_rest_guard_replays_allowed_payload_events_byte_for_byte() -> None:
    path = "/api/v1/slides/presentations"
    events = _request_events(b'{"title":"deck",', b'"slides":[]}')

    delivered, sent, _ = asyncio.run(_invoke(path, events))

    assert _status_and_body(sent)[0] == 200
    assert delivered == events


def test_generic_guard_subdivides_protocol_events_before_scanning(monkeypatch) -> None:
    feed_sizes: list[int] = []
    real_scanner = guard_module.ShallowStandaloneFieldScanner

    class RecordingScanner(real_scanner):
        def feed(self, chunk: bytes) -> None:
            feed_sizes.append(len(chunk))
            super().feed(chunk)

    monkeypatch.setattr(guard_module, "ShallowStandaloneFieldScanner", RecordingScanner)
    payload = b'{"title":"' + (b"a" * (2 * 64 * 1024)) + b'","slides":[]}'

    delivered, sent, _ = asyncio.run(_invoke("/api/v1/slides/presentations", _request_events(payload)))

    assert _status_and_body(sent)[0] == 200
    assert b"".join(event.get("body", b"") for event in delivered) == payload
    assert feed_sizes
    assert max(feed_sizes) <= 64 * 1024


def test_generic_guard_rejects_before_forbidden_value_chunk_is_replayed_and_drains() -> None:
    path = "/api/v1/slides/presentations"
    prefix = b'{"title":"deck","html_document":'
    secret = b'"PRIVATE_SOURCE_0c12"}'
    events = _request_events(prefix, secret)

    delivered, sent, pending = asyncio.run(_invoke(path, events))

    assert _status_and_body(sent) == (409, b'{"detail":"standalone_html_creation_requires_generation"}')
    assert delivered == []
    assert pending == []
    assert secret not in b"".join(message.get("body", b"") for message in sent)


def test_generic_guard_preserves_receive_error_and_has_no_spool_to_clean() -> None:
    path = "/api/v1/slides/presentations"
    scope = _scope(path)
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        raise asyncio.CancelledError

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def app(_scope: dict[str, Any], guarded_receive, _send) -> None:
        await guarded_receive()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(StandaloneHtmlRequestGuardMiddleware(app)(scope, receive, send))
    assert sent == []
