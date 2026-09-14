"""One bounded, allowlisted provider call for standalone HTML generation."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

import httpx
from anyio import fail_after

from tldw_Server_API.app.core.Slides.standalone_html_config import (
    CLOSED_ADAPTER_CATALOG,
    MAX_DOCUMENT_BYTES,
    MAX_OUTPUT_TOKENS,
    MAX_PROVIDER_RESPONSE_BYTES,
    PROMPT_MAX_BYTES,
    ResolvedExecutionTarget,
    SlidesStandaloneHtmlConfig,
)

_PROVIDER_JSON_MAX_DEPTH = 64
_PROVIDER_JSON_MAX_TOKENS = 200_000
_PROVIDER_JSON_MAX_CONTAINERS = 25_000
_PROVIDER_JSON_MAX_MEMBERS_AND_ITEMS = 100_000
_PROVIDER_JSON_MAX_STRING_BYTES = 7 * 1024 * 1024
_MAX_PROVIDER_CREDENTIAL_BYTES = 4_096
_ANTHROPIC_VERSION = "2023-06-01"
_FENCE = re.compile(
    r"\A```(?P<label>[A-Za-z]*)[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t]*\Z",
    re.IGNORECASE,
)
_CATALOG_BY_ID = {adapter.adapter_id: adapter for adapter in CLOSED_ADAPTER_CATALOG}
_AsyncClient = httpx.AsyncClient


class StandaloneHtmlProviderError(RuntimeError):
    """Bounded provider failure that never includes source, response, or secrets."""

    __slots__ = ("code", "status_code")

    def __init__(self, code: str, *, status_code: int | None = None) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(code)


def _fail(code: str, *, status_code: int | None = None) -> NoReturn:
    raise StandaloneHtmlProviderError(code, status_code=status_code) from None


@dataclass(slots=True)
class _JsonFrame:
    kind: str
    expects_item: bool = True


class _ProviderJsonPreflight:
    """Incrementally enforce allocation budgets before JSON materialization."""

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
        self._frames: list[_JsonFrame] = []
        self._in_primitive = False
        self._in_string = False
        self._members_and_items = 0
        self._root_value_seen = False
        self._string_bytes = 0
        self._tokens = 0

    def _token(self) -> None:
        self._tokens += 1
        if self._tokens > _PROVIDER_JSON_MAX_TOKENS:
            _fail("standalone_html_provider_response_invalid")

    def _member_or_item(self) -> None:
        self._members_and_items += 1
        if self._members_and_items > _PROVIDER_JSON_MAX_MEMBERS_AND_ITEMS:
            _fail("standalone_html_provider_response_invalid")

    def _start_value(self) -> None:
        if self._frames and self._frames[-1].kind == "array":
            frame = self._frames[-1]
            if frame.expects_item:
                self._member_or_item()
                frame.expects_item = False
        elif not self._frames:
            self._root_value_seen = True

    def _open_container(self, kind: str) -> None:
        self._start_value()
        self._token()
        self._containers += 1
        if self._containers > _PROVIDER_JSON_MAX_CONTAINERS:
            _fail("standalone_html_provider_response_invalid")
        self._frames.append(_JsonFrame(kind))
        if len(self._frames) > _PROVIDER_JSON_MAX_DEPTH:
            _fail("standalone_html_provider_response_invalid")

    def feed(self, chunk: bytes) -> None:
        index = 0
        while index < len(chunk):
            byte = chunk[index]
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                    self._string_bytes += 1
                elif byte == 0x5C:  # backslash
                    self._escaped = True
                    self._string_bytes += 1
                elif byte == 0x22:  # quote
                    self._in_string = False
                else:
                    self._string_bytes += 1
                if self._string_bytes > _PROVIDER_JSON_MAX_STRING_BYTES:
                    _fail("standalone_html_provider_response_invalid")
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
                continue
            if byte == 0x22:  # quote
                self._start_value()
                self._token()
                self._in_string = True
                self._string_bytes = 0
                index += 1
                continue
            if byte == 0x7B:  # {
                self._open_container("object")
                index += 1
                continue
            if byte == 0x5B:  # [
                self._open_container("array")
                index += 1
                continue
            if byte in (0x7D, 0x5D):  # } or ]
                self._token()
                expected = "object" if byte == 0x7D else "array"
                if not self._frames or self._frames[-1].kind != expected:
                    _fail("standalone_html_provider_response_invalid")
                self._frames.pop()
                index += 1
                continue
            if byte == 0x3A:  # colon
                self._token()
                self._member_or_item()
                index += 1
                continue
            if byte == 0x2C:  # comma
                self._token()
                if self._frames and self._frames[-1].kind == "array":
                    self._frames[-1].expects_item = True
                index += 1
                continue

            self._start_value()
            self._token()
            self._in_primitive = True
            index += 1

    def finish(self) -> None:
        if self._in_string or self._escaped or self._frames or not self._root_value_seen:
            _fail("standalone_html_provider_response_invalid")


def _contains_lone_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _valid_text(value: object, *, nonblank: bool = False) -> bool:
    if not isinstance(value, str) or _contains_lone_surrogate(value) or "\x00" in value:
        return False
    return not nonblank or bool(value.strip())


def _valid_provider_credential(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= _MAX_PROVIDER_CREDENTIAL_BYTES
        and all(0x21 <= ord(character) <= 0x7E for character in value)
    )


def _strict_json_loads(raw: bytes) -> object:
    def reject_constant(_value: str) -> None:
        _fail("standalone_html_provider_response_invalid")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("standalone_html_provider_response_invalid")
            result[key] = value
        return result

    decoded: object = None
    decode_failed = False
    try:
        text = raw.decode("utf-8", errors="strict")
        decoded = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except StandaloneHtmlProviderError:
        raise
    except (RecursionError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        decode_failed = True
    if decode_failed:
        _fail("standalone_html_provider_response_invalid")

    pending = [decoded]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            if _contains_lone_surrogate(value):
                _fail("standalone_html_provider_response_invalid")
        elif isinstance(value, float):
            if not math.isfinite(value):
                _fail("standalone_html_provider_response_invalid")
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return decoded


def _raw_header_values(headers: httpx.Headers, name: bytes) -> list[bytes]:
    lowered = name.lower()
    return [value for key, value in headers.raw if key.lower() == lowered]


def _validate_content_encoding(headers: httpx.Headers) -> None:
    values = _raw_header_values(headers, b"content-encoding")
    if not values:
        return
    if len(values) != 1 or b"," in values[0]:
        _fail("standalone_html_provider_response_invalid")
    encoding: str | None = None
    try:
        encoding = values[0].decode("ascii").strip().casefold()
    except UnicodeDecodeError:
        pass
    if encoding != "identity":
        _fail("standalone_html_provider_response_invalid")


def _declared_content_length(headers: httpx.Headers) -> int | None:
    values = _raw_header_values(headers, b"content-length")
    if not values:
        return None
    if len(values) != 1 or b"," in values[0]:
        _fail("standalone_html_provider_response_invalid")
    raw = values[0]
    if not raw or not raw.isdigit():
        _fail("standalone_html_provider_response_invalid")
    if _raw_header_values(headers, b"transfer-encoding"):
        _fail("standalone_html_provider_response_invalid")
    significant = raw.lstrip(b"0") or b"0"
    maximum = str(MAX_PROVIDER_RESPONSE_BYTES).encode("ascii")
    if len(significant) > len(maximum) or (len(significant) == len(maximum) and significant > maximum):
        return MAX_PROVIDER_RESPONSE_BYTES + 1
    return int(significant)


def _verify_current_target(
    stored_target: ResolvedExecutionTarget,
    current: object,
) -> SlidesStandaloneHtmlConfig:
    if not isinstance(current, SlidesStandaloneHtmlConfig):
        _fail("standalone_html_endpoint_not_allowed")
    if not current.feature_enabled or not current.egress_enabled:
        _fail("standalone_html_egress_disabled")
    adapter = _CATALOG_BY_ID.get(stored_target.adapter_id)
    if (
        adapter is None
        or adapter.provider != stored_target.provider
        or adapter.endpoint_identity != stored_target.endpoint_identity
    ):
        _fail("standalone_html_endpoint_not_allowed")
    if stored_target in current.allowed_targets:
        return current
    if any(
        candidate.provider == stored_target.provider
        and candidate.adapter_id == stored_target.adapter_id
        and candidate.endpoint_identity == stored_target.endpoint_identity
        for candidate in current.allowed_targets
    ):
        _fail("standalone_html_model_not_allowed")
    _fail("standalone_html_endpoint_not_allowed")


def _load_current_config(
    loader: Callable[[], SlidesStandaloneHtmlConfig],
) -> SlidesStandaloneHtmlConfig:
    current: object = None
    loader_failed = False
    try:
        current = loader()
    except Exception:  # noqa: BLE001 - redact every loader failure at this boundary.
        loader_failed = True
    if loader_failed or not isinstance(current, SlidesStandaloneHtmlConfig):
        _fail("standalone_html_endpoint_not_allowed")
    return current


def _validated_runtime_limits(config: SlidesStandaloneHtmlConfig) -> tuple[int, int]:
    response_limit = config.output_limits.max_provider_response_bytes
    document_limit = config.output_limits.max_document_bytes
    if type(response_limit) is not int or response_limit <= 0 or type(document_limit) is not int or document_limit <= 0:
        _fail("standalone_html_provider_request_invalid")

    limits = config.provider_limits
    if (
        not isinstance(limits.max_output_tokens, int)
        or isinstance(limits.max_output_tokens, bool)
        or not 1 <= limits.max_output_tokens <= MAX_OUTPUT_TOKENS
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
            for value in (
                limits.connect_timeout_seconds,
                limits.read_timeout_seconds,
                limits.overall_timeout_seconds,
            )
        )
    ):
        _fail("standalone_html_provider_request_invalid")
    return (
        min(response_limit, MAX_PROVIDER_RESPONSE_BYTES),
        min(document_limit, MAX_DOCUMENT_BYTES),
    )


def _provider_timeout(config: SlidesStandaloneHtmlConfig) -> httpx.Timeout:
    limits = config.provider_limits
    return httpx.Timeout(
        connect=limits.connect_timeout_seconds,
        read=limits.read_timeout_seconds,
        write=limits.connect_timeout_seconds,
        pool=limits.connect_timeout_seconds,
    )


def _build_request(
    target: ResolvedExecutionTarget,
    *,
    system_prompt: str,
    user_content: str,
    provider_api_key: str | None,
    max_output_tokens: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "Content-Type": "application/json",
    }
    if target.provider == "anthropic":
        if not _valid_provider_credential(provider_api_key):
            _fail("standalone_html_provider_credentials_unavailable")
        headers["x-api-key"] = provider_api_key
        headers["anthropic-version"] = _ANTHROPIC_VERSION
        return headers, {
            "model": target.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": max_output_tokens,
            "stream": False,
        }
    if target.provider == "openai":
        if not _valid_provider_credential(provider_api_key):
            _fail("standalone_html_provider_credentials_unavailable")
        headers["Authorization"] = f"Bearer {provider_api_key}"
    elif target.provider not in {"llama.cpp", "ollama"}:
        _fail("standalone_html_endpoint_not_allowed")

    token_field = (
        "max_completion_tokens"
        if target.provider == "openai" and target.model.casefold().startswith("gpt-5")
        else "max_tokens"
    )
    return headers, {
        "model": target.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        token_field: max_output_tokens,
        "stream": False,
    }


def _extract_text(decoded: object, target: ResolvedExecutionTarget) -> str:
    if not isinstance(decoded, dict):
        _fail("standalone_html_provider_response_invalid")
    if target.provider == "anthropic":
        content = decoded.get("content")
        if not isinstance(content, list) or len(content) != 1:
            _fail("standalone_html_provider_response_invalid")
        block = content[0]
        if (
            not isinstance(block, dict)
            or set(block) != {"type", "text"}
            or block.get("type") != "text"
            or not isinstance(block.get("text"), str)
        ):
            _fail("standalone_html_provider_response_invalid")
        return block["text"]

    choices = decoded.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        _fail("standalone_html_provider_response_invalid")
    choice = choices[0]
    if not isinstance(choice, dict):
        _fail("standalone_html_provider_response_invalid")
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        _fail("standalone_html_provider_response_invalid")
    return message["content"]


def _document_bytes(text: str, *, max_document_bytes: int) -> bytes:
    if _contains_lone_surrogate(text):
        _fail("standalone_html_provider_response_invalid")
    candidate = text.strip()
    if "```" in candidate:
        match = _FENCE.fullmatch(candidate)
        if match is None or match.group("label").casefold() not in {"", "html"}:
            _fail("standalone_html_provider_response_invalid")
        candidate = match.group("body").strip()
        if "```" in candidate:
            _fail("standalone_html_provider_response_invalid")
    lowered = candidate.casefold()
    if not lowered.startswith("<!doctype html>") or not lowered.endswith("</html>"):
        _fail("standalone_html_provider_response_invalid")
    document = candidate.encode("utf-8", errors="strict")
    if len(document) > max_document_bytes:
        _fail("standalone_html_provider_response_too_large")
    return document


async def _request_once(
    *,
    target: ResolvedExecutionTarget,
    system_prompt: str,
    user_content: str,
    provider_api_key: str | None,
    initial_config: SlidesStandaloneHtmlConfig,
    current_config_loader: Callable[[], SlidesStandaloneHtmlConfig],
) -> bytes:
    async with _AsyncClient(
        timeout=_provider_timeout(initial_config),
        trust_env=False,
        follow_redirects=False,
    ) as client:
        current = _load_current_config(current_config_loader)
        response_limit, document_limit = _validated_runtime_limits(current)
        limits = current.provider_limits
        with fail_after(limits.overall_timeout_seconds):
            headers, payload = _build_request(
                target,
                system_prompt=system_prompt,
                user_content=user_content,
                provider_api_key=provider_api_key,
                max_output_tokens=limits.max_output_tokens,
            )
            request_timeout = _provider_timeout(current)
            current = _verify_current_target(target, current)
            async with client.stream(
                "POST",
                target.endpoint_identity,
                headers=headers,
                json=payload,
                timeout=request_timeout,
            ) as response:
                return await _consume_response(
                    response,
                    target=target,
                    response_limit=response_limit,
                    document_limit=document_limit,
                )


async def _consume_response(
    response: httpx.Response,
    *,
    target: ResolvedExecutionTarget,
    response_limit: int,
    document_limit: int,
) -> bytes:
    _validate_content_encoding(response.headers)
    declared_length = _declared_content_length(response.headers)
    if declared_length is not None and declared_length > response_limit:
        _fail("standalone_html_provider_response_too_large")

    success = 200 <= response.status_code < 300
    body = bytearray() if success else None
    preflight = _ProviderJsonPreflight() if success else None
    received = 0
    async for chunk in response.aiter_raw():
        received += len(chunk)
        if received > response_limit:
            _fail("standalone_html_provider_response_too_large")
        if preflight is not None:
            preflight.feed(chunk)
            body.extend(chunk)

    if declared_length is not None and received != declared_length:
        _fail("standalone_html_provider_response_invalid")
    if not success:
        _fail(
            "standalone_html_provider_http_error",
            status_code=response.status_code,
        )
    preflight.finish()
    decoded = _strict_json_loads(bytes(body))
    text = _extract_text(decoded, target)
    return _document_bytes(text, max_document_bytes=document_limit)


async def generate_standalone_html(
    *,
    stored_target: ResolvedExecutionTarget,
    system_prompt: str,
    user_content: str,
    provider_api_key: str | None,
    current_config_loader: Callable[[], SlidesStandaloneHtmlConfig],
) -> bytes:
    """Make exactly one isolated completion call and return bounded HTML bytes."""

    if not isinstance(stored_target, ResolvedExecutionTarget):
        _fail("standalone_html_endpoint_not_allowed")
    if (
        not _valid_text(system_prompt, nonblank=True)
        or len(system_prompt.encode("utf-8")) > PROMPT_MAX_BYTES
        or not _valid_text(user_content, nonblank=True)
        or not callable(current_config_loader)
    ):
        _fail("standalone_html_provider_request_invalid")
    current = _load_current_config(current_config_loader)
    current = _verify_current_target(stored_target, current)
    _validated_runtime_limits(current)

    try:
        return await _request_once(
            target=stored_target,
            system_prompt=system_prompt,
            user_content=user_content,
            provider_api_key=provider_api_key,
            initial_config=current,
            current_config_loader=current_config_loader,
        )
    except asyncio.CancelledError:
        raise
    except StandaloneHtmlProviderError:
        raise
    except (TimeoutError, httpx.TimeoutException):
        failure_code = "standalone_html_provider_timeout"
    except (httpx.HTTPError, OSError):
        failure_code = "standalone_html_provider_unavailable"
    _fail(failure_code)


__all__ = ["StandaloneHtmlProviderError", "generate_standalone_html"]
