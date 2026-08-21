"""Receive-time admission and redaction for standalone HTML transports."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

GENERATION_MAX_REQUEST_BYTES = 4 * 1024 * 1024
RAW_HTML_MAX_REQUEST_BYTES = 1_048_576
GENERATION_JSON_MAX_DEPTH = 8
GENERATION_JSON_MAX_TOKENS = 4_096
GENERATION_JSON_MAX_CONTAINERS = 512
GENERATION_JSON_MAX_MEMBERS_AND_ITEMS = 2_048
GENERATION_JSON_MAX_STRING_BYTES = 3 * 1024 * 1024
SCANNER_CHUNK_BYTES = 64 * 1024

_GENERATION_PATH = "/api/v1/slides/generations"
_RAW_SOURCE_PATH = re.compile(r"^/api/v1/slides/presentations/[^/]+/(?P<operation>html-source|draft-attachment)$")
_GENERIC_PRESENTATION_PATH = re.compile(r"^/api/v1/slides/presentations(?:/[^/]+)?$")
_MCP_HTTP_PATHS = frozenset(
    {
        "/api/v1/mcp/request",
        "/api/v1/mcp/request/batch",
        "/api/v1/mcp/tools/execute",
    }
)
_SOURCE_RESPONSE_PATHS = (
    re.compile(r"^/api/v1/slides/presentations/[^/]+$"),
    re.compile(r"^/api/v1/slides/presentations/[^/]+/html-source$"),
    re.compile(r"^/api/v1/slides/presentations/[^/]+/restore$"),
    re.compile(r"^/api/v1/slides/presentations/[^/]+/versions/[^/]+$"),
    re.compile(r"^/api/v1/slides/presentations/[^/]+/versions/[^/]+/restore$"),
    re.compile(r"^/api/v1/slides/presentations/[^/]+/export$"),
)
_TRACKED_SCANNER_DEPTH = 6
_MAX_SCANNER_TOKEN_BYTES = 128


def _feed_scanner(scanner: ShallowStandaloneFieldScanner, body: bytes) -> None:
    for offset in range(0, len(body), SCANNER_CHUNK_BYTES):
        scanner.feed(body[offset : offset + SCANNER_CHUNK_BYTES])


@dataclass(frozen=True, slots=True)
class StandaloneRequestRoute:
    """One stable receive contract consumed by current and future routes."""

    mode: Literal["fixed_json", "fixed_raw", "generic_rest", "generic_mcp"]
    max_bytes: int | None = None
    strict_json: bool = False
    too_large_code: str = "standalone_html_storage_limit"


_GENERATION_ROUTE = StandaloneRequestRoute(
    mode="fixed_json",
    max_bytes=GENERATION_MAX_REQUEST_BYTES,
    strict_json=True,
    too_large_code="standalone_html_storage_limit",
)
_RAW_ROUTE = StandaloneRequestRoute(
    mode="fixed_raw",
    max_bytes=RAW_HTML_MAX_REQUEST_BYTES,
    strict_json=False,
)
_GENERIC_REST_ROUTE = StandaloneRequestRoute(mode="generic_rest")
_GENERIC_MCP_ROUTE = StandaloneRequestRoute(mode="generic_mcp")


def match_standalone_request_route(method: str, path: str) -> StandaloneRequestRoute | None:
    """Return the receive contract for one fixed or compatibility route."""
    normalized_method = method.upper()
    if normalized_method == "POST" and path == _GENERATION_PATH:
        return _GENERATION_ROUTE
    raw_match = _RAW_SOURCE_PATH.fullmatch(path)
    if raw_match is not None:
        operation = raw_match.group("operation")
        if (operation == "html-source" and normalized_method == "PUT") or (
            operation == "draft-attachment" and normalized_method == "POST"
        ):
            return _RAW_ROUTE
    if _GENERIC_PRESENTATION_PATH.fullmatch(path) is not None:
        if (path.endswith("/presentations") and normalized_method == "POST") or (
            not path.endswith("/presentations") and normalized_method in {"PUT", "PATCH"}
        ):
            return _GENERIC_REST_ROUTE
    if normalized_method == "POST" and path in _MCP_HTTP_PATHS:
        return _GENERIC_MCP_ROUTE
    return None


def is_standalone_sensitive_route(method: str, path: str) -> bool:
    """Return whether validation/serialization errors require source redaction."""
    if match_standalone_request_route(method, path) is not None:
        return True
    return any(pattern.fullmatch(path) is not None for pattern in _SOURCE_RESPONSE_PATHS)


class StandaloneHtmlAdmissionError(Exception):
    """Bounded public rejection that contains only a stable code."""

    __slots__ = ("code", "status_code")

    def __init__(self, status_code: int, code: str) -> None:
        self.status_code = status_code
        self.code = code
        super().__init__(code)


def _reject(status_code: int, code: str) -> NoReturn:
    raise StandaloneHtmlAdmissionError(status_code, code) from None


@dataclass(slots=True)
class _BudgetFrame:
    kind: Literal["object", "array"]
    expects_item: bool = True


class _GenerationJsonPreflight:
    """Incrementally enforce fixed JSON allocation budgets."""

    __slots__ = (
        "_containers",
        "_escaped",
        "_frames",
        "_in_primitive",
        "_in_string",
        "_members_and_items",
        "_root_value_seen",
        "_string_bytes",
        "_tokens",
    )

    def __init__(self) -> None:
        self._containers = 0
        self._escaped = False
        self._frames: list[_BudgetFrame] = []
        self._in_primitive = False
        self._in_string = False
        self._members_and_items = 0
        self._root_value_seen = False
        self._string_bytes = 0
        self._tokens = 0

    def _token(self) -> None:
        self._tokens += 1
        if self._tokens > GENERATION_JSON_MAX_TOKENS:
            _reject(422, "json_structure_too_complex")

    def _member_or_item(self) -> None:
        self._members_and_items += 1
        if self._members_and_items > GENERATION_JSON_MAX_MEMBERS_AND_ITEMS:
            _reject(422, "json_structure_too_complex")

    def _start_value(self) -> None:
        if self._frames and self._frames[-1].kind == "array":
            frame = self._frames[-1]
            if frame.expects_item:
                self._member_or_item()
                frame.expects_item = False
        elif not self._frames:
            self._root_value_seen = True

    def _open_container(self, kind: Literal["object", "array"]) -> None:
        self._start_value()
        self._token()
        self._containers += 1
        if self._containers > GENERATION_JSON_MAX_CONTAINERS:
            _reject(422, "json_structure_too_complex")
        self._frames.append(_BudgetFrame(kind))
        if len(self._frames) > GENERATION_JSON_MAX_DEPTH:
            _reject(422, "json_structure_too_complex")

    def feed(self, chunk: bytes) -> None:
        index = 0
        while index < len(chunk):
            byte = chunk[index]
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                    self._string_bytes += 1
                elif byte == 0x5C:
                    self._escaped = True
                    self._string_bytes += 1
                elif byte == 0x22:
                    self._in_string = False
                else:
                    self._string_bytes += 1
                if self._string_bytes > GENERATION_JSON_MAX_STRING_BYTES:
                    _reject(422, "json_structure_too_complex")
                index += 1
                continue
            if self._in_primitive:
                if byte not in b' \t\r\n{}[],:"':
                    index += 1
                    continue
                self._in_primitive = False
                continue
            if byte in b" \t\r\n":
                index += 1
            elif byte == 0x22:
                self._start_value()
                self._token()
                self._in_string = True
                self._string_bytes = 0
                index += 1
            elif byte == 0x7B:
                self._open_container("object")
                index += 1
            elif byte == 0x5B:
                self._open_container("array")
                index += 1
            elif byte in (0x7D, 0x5D):
                self._token()
                expected = "object" if byte == 0x7D else "array"
                if not self._frames or self._frames[-1].kind != expected:
                    _reject(422, "standalone_html_request_invalid")
                self._frames.pop()
                index += 1
            elif byte == 0x3A:
                self._token()
                self._member_or_item()
                index += 1
            elif byte == 0x2C:
                self._token()
                if self._frames and self._frames[-1].kind == "array":
                    self._frames[-1].expects_item = True
                index += 1
            else:
                self._start_value()
                self._token()
                self._in_primitive = True
                index += 1

    def finish(self) -> None:
        if self._in_string or self._escaped or self._frames or not self._root_value_seen:
            _reject(422, "standalone_html_request_invalid")


def _strict_json_preflight(raw: bytes) -> None:
    def reject_constant(_value: str) -> None:
        _reject(422, "standalone_html_request_invalid")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _reject(422, "standalone_html_request_invalid")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except StandaloneHtmlAdmissionError:
        raise
    except (RecursionError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        _reject(422, "standalone_html_request_invalid")

    pending = [decoded]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
                _reject(422, "standalone_html_request_invalid")
        elif isinstance(value, float) and not math.isfinite(value):
            _reject(422, "standalone_html_request_invalid")
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


@dataclass(slots=True)
class _ScanFrame:
    kind: Literal["object", "array"]
    path: tuple[str, ...] | None
    state: str
    current_key: str | None = None


class ShallowStandaloneFieldScanner:
    """Lexically reject standalone fields while retaining constant state."""

    __slots__ = (
        "_capture_overflow",
        "_depth",
        "_escaped",
        "_frames",
        "_in_primitive",
        "_in_string",
        "_mode",
        "_string_raw",
        "_string_role",
    )

    def __init__(self, *, mode: Literal["rest", "mcp"]) -> None:
        self._capture_overflow = False
        self._depth = 0
        self._escaped = False
        self._frames: list[_ScanFrame] = []
        self._in_primitive = False
        self._in_string = False
        self._mode = mode
        self._string_raw = bytearray()
        self._string_role: Literal["key", "value", "ignore"] = "ignore"

    @property
    def retained_bytes(self) -> int:
        """Return scanner-owned variable bytes for constant-memory assertions."""
        return len(self._string_raw)

    def _tracked_parent(self) -> _ScanFrame | None:
        if self._depth <= _TRACKED_SCANNER_DEPTH and self._frames:
            return self._frames[-1]
        return None

    def _target_path(self, path: tuple[str, ...] | None) -> bool:
        if path is None:
            return False
        if self._mode == "rest":
            return path == ()
        return path in {
            ("arguments",),
            ("arguments", "updates"),
            ("arguments", "patch"),
            ("params", "arguments"),
            ("params", "arguments", "updates"),
            ("params", "arguments", "patch"),
        }

    def _before_value(self) -> tuple[tuple[str, ...] | None, str | None]:
        parent = self._tracked_parent()
        if parent is None:
            return None, None
        if parent.kind == "object" and parent.state == "value":
            path, key = parent.path, parent.current_key
            parent.state = "comma_or_end"
            parent.current_key = None
            return path, key
        if parent.kind == "array" and parent.state == "value_or_end":
            parent.state = "comma_or_end"
        return parent.path, None

    def _child_path(
        self,
        parent_path: tuple[str, ...] | None,
        key: str | None,
        *,
        parent: _ScanFrame | None,
    ) -> tuple[str, ...] | None:
        if self._depth == 0:
            return ()
        if self._mode == "mcp" and self._depth == 1 and parent is not None and parent.kind == "array":
            return ()
        if parent_path is None or key is None:
            return None
        candidate = parent_path + (key,)
        prefixes = {
            ("arguments",),
            ("arguments", "updates"),
            ("arguments", "patch"),
            ("params",),
            ("params", "arguments"),
            ("params", "arguments", "updates"),
            ("params", "arguments", "patch"),
        }
        return candidate if self._mode == "mcp" and candidate in prefixes else None

    def _open(self, kind: Literal["object", "array"]) -> None:
        parent = self._tracked_parent()
        parent_path, key = self._before_value()
        path = self._child_path(parent_path, key, parent=parent)
        self._depth += 1
        if self._depth <= _TRACKED_SCANNER_DEPTH:
            self._frames.append(
                _ScanFrame(
                    kind=kind,
                    path=path,
                    state="key_or_end" if kind == "object" else "value_or_end",
                )
            )

    def _close(self) -> None:
        if self._depth <= _TRACKED_SCANNER_DEPTH and self._frames:
            self._frames.pop()
        if self._depth > 0:
            self._depth -= 1

    def _start_string(self) -> None:
        self._in_string = True
        self._escaped = False
        self._capture_overflow = False
        self._string_raw.clear()
        parent = self._tracked_parent()
        if parent is not None and parent.kind == "object" and parent.state == "key_or_end":
            self._string_role = "key"
            return
        path, key = self._before_value()
        self._string_role = "value" if key == "content_kind" and self._target_path(path) else "ignore"

    def _decoded_string(self) -> str | None:
        if self._capture_overflow:
            return None
        try:
            value = json.loads(b'"' + bytes(self._string_raw) + b'"')
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            return None
        return value if isinstance(value, str) else None

    def _finish_string(self) -> None:
        value = self._decoded_string()
        if self._string_role == "key":
            parent = self._tracked_parent()
            if parent is not None and parent.kind == "object" and parent.state == "key_or_end":
                parent.current_key = value
                parent.state = "colon"
        elif self._string_role == "value" and value == "standalone_html":
            _reject(409, "standalone_html_creation_requires_generation")
        self._string_raw.clear()
        self._capture_overflow = False
        self._string_role = "ignore"

    def feed(self, chunk: bytes) -> None:
        index = 0
        while index < len(chunk):
            byte = chunk[index]
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                    if not self._capture_overflow:
                        self._string_raw.append(byte)
                elif byte == 0x5C:
                    self._escaped = True
                    if not self._capture_overflow:
                        self._string_raw.append(byte)
                elif byte == 0x22:
                    self._in_string = False
                    self._finish_string()
                    index += 1
                    continue
                elif not self._capture_overflow:
                    self._string_raw.append(byte)
                if len(self._string_raw) > _MAX_SCANNER_TOKEN_BYTES:
                    self._capture_overflow = True
                    self._string_raw.clear()
                index += 1
                continue
            if self._in_primitive:
                if byte not in b" \t\r\n,]}":
                    index += 1
                    continue
                self._in_primitive = False
                continue
            if byte in b" \t\r\n":
                index += 1
            elif byte == 0x22:
                self._start_string()
                index += 1
            elif byte == 0x7B:
                self._open("object")
                index += 1
            elif byte == 0x5B:
                self._open("array")
                index += 1
            elif byte in (0x7D, 0x5D):
                self._close()
                index += 1
            elif byte == 0x3A:
                parent = self._tracked_parent()
                if parent is not None and parent.kind == "object" and parent.state == "colon":
                    if parent.current_key == "html_document" and self._target_path(parent.path):
                        _reject(409, "standalone_html_creation_requires_generation")
                    parent.state = "value"
                index += 1
            elif byte == 0x2C:
                parent = self._tracked_parent()
                if parent is not None:
                    parent.state = "key_or_end" if parent.kind == "object" else "value_or_end"
                index += 1
            else:
                self._before_value()
                self._in_primitive = True
                index += 1

    def finish(self) -> None:
        """Discard bounded lexical state; ordinary JSON parsing owns validity."""
        self._string_raw.clear()
        self._capture_overflow = False


def _raw_header_values(scope: Mapping[str, Any], name: bytes) -> list[bytes]:
    lowered = name.lower()
    return [value for key, value in scope.get("headers", ()) if key.lower() == lowered]


def _validate_content_encoding(scope: Mapping[str, Any]) -> None:
    values = _raw_header_values(scope, b"content-encoding")
    if not values:
        return
    if len(values) != 1 or b"," in values[0]:
        _reject(415, "standalone_html_unsupported_encoding")
    try:
        encoding = values[0].decode("ascii").strip().casefold()
    except UnicodeDecodeError:
        encoding = ""
    if encoding != "identity":
        _reject(415, "standalone_html_unsupported_encoding")


def _declared_content_length(scope: Mapping[str, Any], maximum: int) -> int | None:
    values = _raw_header_values(scope, b"content-length")
    transfer_encoding = _raw_header_values(scope, b"transfer-encoding")
    if not values:
        return None
    if len(values) != 1 or b"," in values[0] or transfer_encoding:
        _reject(400, "invalid_content_length")
    raw = values[0]
    if not raw or not raw.isdigit():
        _reject(400, "invalid_content_length")
    significant = raw.lstrip(b"0") or b"0"
    bound = str(maximum).encode("ascii")
    if len(significant) > len(bound) or (len(significant) == len(bound) and significant > bound):
        return maximum + 1
    return int(significant)


async def _drain_receive(receive: Callable[[], Awaitable[dict[str, Any]]]) -> None:
    while True:
        message = await receive()
        if message.get("type") != "http.request" or not message.get("more_body", False):
            return


async def _send_rejection(
    send: Callable[[dict[str, Any]], Awaitable[None]],
    error: StandaloneHtmlAdmissionError,
) -> None:
    body = json.dumps({"detail": error.code}, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": error.status_code,
            "headers": [
                (b"content-length", str(len(body)).encode("ascii")),
                (b"content-type", b"application/json"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class StandaloneHtmlRequestGuardMiddleware:
    """Apply route-aware receive guards before framework body parsing."""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "")
        path = scope.get("path", "")
        if is_standalone_sensitive_route(method, path):
            scope["standalone_html_sensitive"] = True
        route = match_standalone_request_route(method, path)
        if route is None:
            await self.app(scope, receive, send)
            return
        scope["standalone_html_receive_guard"] = True
        if route.mode.startswith("fixed_"):
            await self._fixed(scope, receive, send, route)
            return
        await self._generic(scope, receive, send, route)

    async def _fixed(self, scope, receive, send, route: StandaloneRequestRoute) -> None:
        maximum = route.max_bytes or 0
        try:
            _validate_content_encoding(scope)
            declared = _declared_content_length(scope, maximum)
            if declared is not None and declared > maximum:
                _reject(413, route.too_large_code)
        except StandaloneHtmlAdmissionError as error:
            await _drain_receive(receive)
            await _send_rejection(send, error)
            return

        buffered = bytearray()
        preflight = _GenerationJsonPreflight() if route.strict_json else None
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                return
            chunk = message.get("body", b"")
            if not isinstance(chunk, bytes):
                chunk = bytes(chunk)
            try:
                if len(buffered) + len(chunk) > maximum:
                    _reject(413, route.too_large_code)
                if preflight is not None:
                    preflight.feed(chunk)
                buffered.extend(chunk)
            except StandaloneHtmlAdmissionError as error:
                if message.get("more_body", False):
                    await _drain_receive(receive)
                await _send_rejection(send, error)
                return
            if not message.get("more_body", False):
                break

        try:
            if declared is not None and declared != len(buffered):
                _reject(400, "invalid_content_length")
            if preflight is not None:
                preflight.finish()
                _strict_json_preflight(bytes(buffered))
        except StandaloneHtmlAdmissionError as error:
            await _send_rejection(send, error)
            return

        delivered = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(buffered), "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _generic(self, scope, receive, send, route: StandaloneRequestRoute) -> None:
        mode: Literal["rest", "mcp"] = "mcp" if route.mode == "generic_mcp" else "rest"
        scanner = ShallowStandaloneFieldScanner(mode=mode)
        more_body = False

        async def guarded_receive() -> dict[str, Any]:
            nonlocal more_body
            message = await receive()
            if message.get("type") == "http.request":
                more_body = bool(message.get("more_body", False))
                _feed_scanner(scanner, message.get("body", b""))
                if not more_body:
                    scanner.finish()
            return message

        try:
            await self.app(scope, guarded_receive, send)
        except StandaloneHtmlAdmissionError as error:
            if more_body:
                await _drain_receive(receive)
            await _send_rejection(send, error)


def install_shallow_request_receive_guard(
    request: Request,
    *,
    mode: Literal["rest", "mcp"],
) -> None:
    """Install the same scanner for apps that do not use the main middleware."""
    if request.scope.get("standalone_html_receive_guard"):
        return
    scanner = ShallowStandaloneFieldScanner(mode=mode)
    original_receive = request._receive  # noqa: SLF001 - Starlette receive boundary.

    async def guarded_receive() -> dict[str, Any]:
        message = await original_receive()
        if message.get("type") == "http.request":
            request.scope["standalone_html_guard_more_body"] = bool(message.get("more_body", False))
            _feed_scanner(scanner, message.get("body", b""))
            if not message.get("more_body", False):
                scanner.finish()
        return message

    request._receive = guarded_receive  # noqa: SLF001 - install before Request.body().
    request.scope["standalone_html_receive_guard"] = True
    request.scope["standalone_html_guard_original_receive"] = original_receive


async def drain_guarded_request(request: Request) -> None:
    """Drain the protocol receive callable after a scanner rejection."""
    receive = request.scope.get("standalone_html_guard_original_receive")
    if callable(receive) and request.scope.get("standalone_html_guard_more_body"):
        await _drain_receive(receive)


_KNOWN_LOCATION_COMPONENTS = frozenset(
    {
        "arguments",
        "audience",
        "body",
        "content_kind",
        "conversation_id",
        "delivery_style",
        "generation_config_revision",
        "generation_mode",
        "html_document",
        "html_options",
        "kind",
        "media_id",
        "method",
        "name",
        "note_ids",
        "params",
        "patch",
        "presentation_type",
        "prompt",
        "query",
        "slide_count",
        "source",
        "top_k",
        "updates",
        "visual_direction",
    }
)
_VALIDATION_CODE_MAP = {
    "extra_forbidden": ("unknown_field", "Unknown field."),
    "missing": ("field_required", "Required field is missing."),
    "literal_error": ("invalid_value", "Value is not allowed."),
    "string_too_long": ("invalid_string", "String is too long."),
    "string_too_short": ("invalid_string", "String is too short."),
}


def _sanitized_location(location: Sequence[Any]) -> list[str | int]:
    sanitized: list[str | int] = []
    for component in location[:4]:
        if isinstance(component, str):
            sanitized.append(component if component in _KNOWN_LOCATION_COMPONENTS else "unknown_field")
        elif isinstance(component, int) and not isinstance(component, bool) and 0 <= component <= 100:
            sanitized.append(component)
        else:
            sanitized.append("unknown_index")
    return sanitized


def standalone_request_validation_response(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    """Build an allowlisted, bounded validation response without raw inputs."""
    del request
    public_errors: list[dict[str, Any]] = []
    for item in error.errors()[:20]:
        code, message = _VALIDATION_CODE_MAP.get(
            str(item.get("type", "")),
            ("invalid_request", "Request value is invalid."),
        )
        public_errors.append(
            {
                "code": code,
                "location": _sanitized_location(item.get("loc", ())),
                "message": message,
            }
        )
    return JSONResponse(
        {"detail": "standalone_html_request_invalid", "errors": public_errors},
        status_code=422,
    )


def standalone_response_invalid_response(_error: Exception | None = None) -> JSONResponse:
    """Return the fixed source-free serialization/response-validation failure."""
    return JSONResponse({"detail": "standalone_html_response_invalid"}, status_code=500)


__all__ = [
    "GENERATION_MAX_REQUEST_BYTES",
    "RAW_HTML_MAX_REQUEST_BYTES",
    "SCANNER_CHUNK_BYTES",
    "ShallowStandaloneFieldScanner",
    "StandaloneHtmlAdmissionError",
    "StandaloneHtmlRequestGuardMiddleware",
    "StandaloneRequestRoute",
    "drain_guarded_request",
    "install_shallow_request_receive_guard",
    "is_standalone_sensitive_route",
    "match_standalone_request_route",
    "standalone_request_validation_response",
    "standalone_response_invalid_response",
]
