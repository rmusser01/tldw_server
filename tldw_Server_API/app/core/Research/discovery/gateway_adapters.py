"""Bounded, gateway-only adapters for the discovery V2 foundation.

Stdlib ElementTree supplies tree types only; XML bytes use DefusedXMLParser.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
import time
import xml.etree.ElementTree as ElementTree  # nosec B405
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import DefusedXMLParser

from tldw_Server_API.app.core.exceptions import (
    _ParseDeadlineExceeded,
    _ParseLimitExceeded,
    _PayloadInvalid,
)

from .contracts import (
    MAX_PAGINATION_CURSOR,
    DeferredNumericCSVQueryBinding,
    DiscoveryOutcomeIdentity,
    DispatchAllowance,
    DispatchIntent,
    OperationKind,
    PlannedDispatchGroup,
    PlannedLogicalAttempt,
    QueryPair,
    RouteLimits,
)
from .executor import (
    BoundDispatch,
    DiscoveryAdapter,
    DiscoveryAdapterError,
    DiscoveryAdapterResult,
    DiscoveryCandidate,
    NumericCSVBindingValues,
    NumericCursor,
)
from .gateway import DiscoveryGatewayResponse
from .identity import (
    build_fingerprint,
    canonicalize_url,
    has_unsafe_url_material,
    normalize_doi,
)

MonotonicClock = Callable[[], int | float]


@dataclass(frozen=True, slots=True)
class _ParsingProfile:
    """Immutable parser ceilings independent from route-policy digests."""

    max_input_bytes: int
    max_records: int
    max_depth: int
    max_nodes: int
    max_string_chars: int
    max_numeric_token_chars: int
    parse_deadline_ms: int


_FOUNDATION_PROFILE = _ParsingProfile(
    max_input_bytes=2_097_152,
    max_records=100,
    max_depth=16,
    max_nodes=50_000,
    max_string_chars=65_536,
    max_numeric_token_chars=32,
    parse_deadline_ms=500,
)
_ADAPTER_IDS = (
    "semantic_scholar_v2",
    "crossref_v2",
    "zenodo_v2",
    "figshare_v2",
    "osf_v2",
    "arxiv_v2",
    "pubmed_v2",
)
_PARSING_PROFILES = MappingProxyType(
    {
        **{(adapter_id, "foundation-v2"): _FOUNDATION_PROFILE for adapter_id in _ADAPTER_IDS},
        ("pubmed_v2", "pubmed-v2-ncbi-identity"): _FOUNDATION_PROFILE,
    }
)
_CLOCK_CHECK_INTERVAL = 256
_XML_CHUNK_BYTES = 8_192
_URL_PATH_DECODE_PASSES = 4
_MAX_XML_ATTRIBUTES_PER_ELEMENT = 16
_MAX_ARXIV_FIELDS_PER_ENTRY = 512
_MISSING = object()
_MIME_TOKEN_CHARACTERS = frozenset("!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyz")
_ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
_OPEN_SEARCH_NAMESPACE = "http://a9.com/-/spec/opensearch/1.1/"
_ARXIV_NAMESPACE = "http://arxiv.org/schemas/atom"
_ATOM_FEED = f"{{{_ATOM_NAMESPACE}}}feed"
_ATOM_ENTRY = f"{{{_ATOM_NAMESPACE}}}entry"
_ATOM_ID = f"{{{_ATOM_NAMESPACE}}}id"
_ATOM_TITLE = f"{{{_ATOM_NAMESPACE}}}title"
_ATOM_SUMMARY = f"{{{_ATOM_NAMESPACE}}}summary"
_ATOM_AUTHOR = f"{{{_ATOM_NAMESPACE}}}author"
_ATOM_NAME = f"{{{_ATOM_NAMESPACE}}}name"
_ATOM_LINK = f"{{{_ATOM_NAMESPACE}}}link"
_ARXIV_DOI = f"{{{_ARXIV_NAMESPACE}}}doi"
_OPEN_SEARCH_TOTAL = f"{{{_OPEN_SEARCH_NAMESPACE}}}totalResults"
_OPEN_SEARCH_START = f"{{{_OPEN_SEARCH_NAMESPACE}}}startIndex"
_OPEN_SEARCH_ITEMS = f"{{{_OPEN_SEARCH_NAMESPACE}}}itemsPerPage"
_ARXIV_ID_RE = re.compile(
    r"(?:\d{4}\.\d{4,5}|[a-z][a-z0-9-]*(?:\.[a-z][a-z0-9-]*)*/\d{7})(?:v[1-9]\d*)?\Z",
    re.IGNORECASE | re.ASCII,
)
_ARXIV_VERSION_RE = re.compile(r"v[1-9]\d*\Z", re.IGNORECASE | re.ASCII)
_XML_ENCODING_RE = re.compile(r"\bencoding\s*=\s*(['\"])([^'\"]+)\1", re.IGNORECASE)
_PUBMED_ID_RE = re.compile(r"[1-9][0-9]{0,15}\Z", re.ASCII)
_PMCID_RE = re.compile(r"PMC[1-9][0-9]{0,15}\Z", re.ASCII)
_PUBMED_BINDING_ID = "pubmed_esearch_ids"
_PUBMED_ROUTE_ID = "pubmed_ncbi_eutils_pubmed_direct"
_PUBMED_BACKEND_ID = "ncbi_eutils_pubmed"
_PUBMED_IDENTITY_ADAPTER_VERSION = "pubmed-v2-ncbi-identity"
_PUBMED_IDENTITY_POLICY_DIGEST = "742b8aca76878ca06ab43ae17130627b5daaebea0a3c3ae25786521a9f159d22"
_NCBI_JSON_VERSION = "0.3"
_NCBI_RATE_COUNT_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z", re.ASCII)
_MAX_PUBMED_AUTHORS_PER_RECORD = 1_024
_MAX_PUBMED_ARTICLE_IDS_PER_RECORD = 64

_TrustedNCBIInputs = tuple[
    PlannedDispatchGroup,
    _ParsingProfile,
    int,
    int,
    int,
    DeferredNumericCSVQueryBinding,
]


class _TrustedNCBIInputsCallback(Protocol):
    def __call__(self, group: object) -> _TrustedNCBIInputs: ...


class _NCBIESearchIDsCallback(Protocol):
    def __call__(
        self,
        payload: Any,
        *,
        profile: _ParsingProfile,
        guard: _ParseGuard,
        retstart: int,
        retmax: int,
        binding: DeferredNumericCSVQueryBinding,
    ) -> tuple[tuple[str, int], ...]: ...


class _NCBISummaryRecordsCallback(Protocol):
    def __call__(
        self,
        payload: Any,
        *,
        expected_ids: tuple[str, ...],
        guard: _ParseGuard,
    ) -> tuple[dict[str, Any], ...]: ...


class _ParseGuard:
    """Cooperative structural and wall-clock parser guard."""

    def __init__(self, profile: _ParsingProfile, clock: MonotonicClock) -> None:
        self.profile = profile
        self.clock = clock
        self.nodes = 0
        self.started = self._read_clock()

    def _read_clock(self) -> float:
        try:
            value = self.clock()
        except Exception as error:
            raise _ParseDeadlineExceeded from error
        if type(value) not in {int, float} or not math.isfinite(value):
            raise _ParseDeadlineExceeded
        return float(value)

    def checkpoint(self) -> None:
        now = self._read_clock()
        elapsed_ms = (now - self.started) * 1000
        if elapsed_ms < 0 or elapsed_ms > self.profile.parse_deadline_ms:
            raise _ParseDeadlineExceeded

    def visit_node(self) -> None:
        self.nodes += 1
        if self.nodes > self.profile.max_nodes:
            raise _ParseLimitExceeded
        if self.nodes % _CLOCK_CHECK_INTERVAL == 0:
            self.checkpoint()


def _raise_adapter_error(error: Exception) -> None:
    if isinstance(error, _ParseDeadlineExceeded):
        raise DiscoveryAdapterError("provider_parse_deadline_exceeded") from None
    if isinstance(error, _ParseLimitExceeded):
        raise DiscoveryAdapterError("provider_parse_limit_exceeded") from None
    raise DiscoveryAdapterError("provider_payload_invalid") from None


def _strict_json(
    response: DiscoveryGatewayResponse,
    *,
    profile: _ParsingProfile,
    max_input_bytes: int,
    clock: MonotonicClock,
) -> tuple[Any, _ParseGuard]:
    """Decode one exact JSON body and enforce all profile ceilings."""
    if type(response.body) is not bytes:
        raise DiscoveryAdapterError("provider_payload_invalid")
    if len(response.body) > max_input_bytes:
        raise DiscoveryAdapterError("provider_parse_limit_exceeded")

    guard: _ParseGuard | None = None
    try:
        guard = _ParseGuard(profile, clock)
        guard.checkpoint()
        if response.body.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")):
            raise _PayloadInvalid
        text = response.body.decode("utf-8", errors="strict")
        if text.startswith("\ufeff"):
            raise _PayloadInvalid

        def parse_integer(token: str) -> int:
            if len(token) > profile.max_numeric_token_chars:
                raise _ParseLimitExceeded
            try:
                return int(token)
            except ValueError as error:
                raise _PayloadInvalid from error

        def parse_float(token: str) -> float:
            if len(token) > profile.max_numeric_token_chars:
                raise _ParseLimitExceeded
            try:
                value = float(token)
            except ValueError as error:
                raise _PayloadInvalid from error
            if not math.isfinite(value):
                raise _PayloadInvalid
            return value

        def reject_constant(_token: str) -> None:
            raise _PayloadInvalid

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise _PayloadInvalid
                result[key] = value
            return result

        payload = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_int=parse_integer,
            parse_float=parse_float,
            parse_constant=reject_constant,
        )
        guard.checkpoint()
        _walk_json(payload, depth=1, guard=guard)
        guard.checkpoint()
        return payload, guard
    except (_PayloadInvalid, _ParseLimitExceeded, _ParseDeadlineExceeded) as error:
        _raise_adapter_error(error)
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError, TypeError):
        raise DiscoveryAdapterError("provider_payload_invalid") from None
    raise AssertionError("unreachable")


def _walk_json(value: Any, *, depth: int, guard: _ParseGuard) -> None:
    guard.visit_node()
    if depth > guard.profile.max_depth:
        raise _ParseLimitExceeded
    if type(value) is str:
        _check_string(value, guard.profile)
        return
    if type(value) is dict:
        for key, item in value.items():
            guard.visit_node()
            _check_string(key, guard.profile)
            _walk_json(item, depth=depth + 1, guard=guard)
        return
    if type(value) is list:
        for item in value:
            _walk_json(item, depth=depth + 1, guard=guard)


class _BoundedXMLTarget:
    """Tree builder that applies structural ceilings while XML is parsed."""

    def __init__(self, guard: _ParseGuard, *, max_name_chars: int) -> None:
        if type(max_name_chars) is not int or max_name_chars <= 0:
            raise _PayloadInvalid
        self.guard = guard
        self.max_name_chars = max_name_chars
        self.builder = ElementTree.TreeBuilder()
        self.depth = 0
        self.text_lengths: list[int] = []
        self.pending_namespace_count = 0
        self.name_chars = 0

    def _visit_name(self, value: str) -> None:
        _check_string(value, self.guard.profile)
        self.name_chars += len(value)
        if self.name_chars > self.max_name_chars:
            raise _ParseLimitExceeded

    def start_ns(self, prefix: str, uri: str) -> None:
        if type(prefix) is not str or type(uri) is not str:
            raise _PayloadInvalid
        if self.pending_namespace_count >= _MAX_XML_ATTRIBUTES_PER_ELEMENT:
            raise _ParseLimitExceeded
        self.guard.visit_node()
        self._visit_name(prefix)
        self._visit_name(uri)
        self.pending_namespace_count += 1

    def end_ns(self, prefix: str) -> None:
        if type(prefix) is not str:
            raise _PayloadInvalid
        _check_string(prefix, self.guard.profile)

    def start(self, tag: str, attributes: dict[str, str]) -> ElementTree.Element:
        if type(tag) is not str or type(attributes) is not dict:
            raise _PayloadInvalid
        self.depth += 1
        if self.depth > self.guard.profile.max_depth:
            raise _ParseLimitExceeded
        self.guard.visit_node()
        self._visit_name(tag)
        if len(attributes) + self.pending_namespace_count > _MAX_XML_ATTRIBUTES_PER_ELEMENT:
            raise _ParseLimitExceeded
        for name, value in attributes.items():
            if type(name) is not str or type(value) is not str:
                raise _PayloadInvalid
            self.guard.visit_node()
            self._visit_name(name)
            _check_string(value, self.guard.profile)
        self.pending_namespace_count = 0
        self.text_lengths.append(0)
        return self.builder.start(tag, attributes)

    def data(self, value: str) -> None:
        if type(value) is not str:
            raise _PayloadInvalid
        _check_string(value, self.guard.profile)
        if self.text_lengths:
            self.text_lengths[-1] += len(value)
            if self.text_lengths[-1] > self.guard.profile.max_string_chars:
                raise _ParseLimitExceeded
        elif value.strip():
            raise _PayloadInvalid
        self.builder.data(value)

    def end(self, tag: str) -> ElementTree.Element:
        if type(tag) is not str or self.depth <= 0 or not self.text_lengths:
            raise _PayloadInvalid
        _check_string(tag, self.guard.profile)
        element = self.builder.end(tag)
        self.text_lengths.pop()
        self.depth -= 1
        return element

    def close(self) -> ElementTree.Element:
        if self.depth != 0 or self.text_lengths or self.pending_namespace_count:
            raise _PayloadInvalid
        root = self.builder.close()
        if type(root) is not ElementTree.Element:
            raise _PayloadInvalid
        return root


def _strict_atom(
    response: DiscoveryGatewayResponse,
    *,
    profile: _ParsingProfile,
    max_input_bytes: int,
    clock: MonotonicClock,
) -> tuple[ElementTree.Element, _ParseGuard]:
    """Parse one UTF-8 Atom body with entity and structural defenses."""
    if type(response.body) is not bytes:
        raise DiscoveryAdapterError("provider_payload_invalid")
    if len(response.body) > max_input_bytes:
        raise DiscoveryAdapterError("provider_parse_limit_exceeded")

    try:
        guard = _ParseGuard(profile, clock)
        guard.checkpoint()
        text = response.body.decode("utf-8", errors="strict")
        if text.startswith("\ufeff") or "\x00" in text:
            raise _PayloadInvalid
        if text.startswith("<?xml"):
            declaration_end = text.find("?>")
            if declaration_end < 0:
                raise _PayloadInvalid
            encoding = _XML_ENCODING_RE.search(text[: declaration_end + 2])
            if encoding is not None and encoding.group(2).casefold() not in {"utf-8", "utf8"}:
                raise _PayloadInvalid

        target = _BoundedXMLTarget(guard, max_name_chars=max_input_bytes)
        parser = DefusedXMLParser(
            encoding="utf-8",
            target=target,
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )
        for offset in range(0, len(response.body), _XML_CHUNK_BYTES):
            guard.checkpoint()
            parser.feed(response.body[offset : offset + _XML_CHUNK_BYTES])
        root = parser.close()
        guard.checkpoint()
        return root, guard
    except (_PayloadInvalid, _ParseLimitExceeded, _ParseDeadlineExceeded) as error:
        _raise_adapter_error(error)
    except (DefusedXmlException, ElementTree.ParseError, UnicodeError, RecursionError, ValueError, TypeError):
        raise DiscoveryAdapterError("provider_payload_invalid") from None
    raise AssertionError("unreachable")


def _check_string(value: str, profile: _ParsingProfile) -> None:
    if len(value) > profile.max_string_chars:
        raise _ParseLimitExceeded
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise _PayloadInvalid


def _response_content_type(response: DiscoveryGatewayResponse) -> str | None:
    if type(response.headers) is not tuple:
        return None
    values: list[str] = []
    for pair in response.headers:
        if type(pair) is not tuple or len(pair) != 2:
            return None
        name, value = pair
        if type(name) is not str or type(value) is not str:
            return None
        if name.lower() == "content-type":
            values.append(value)
    return values[0] if len(values) == 1 else None


def _is_json_content_type(value: str | None) -> bool:
    if type(value) is not str or "," in value:
        return False
    parts = value.split(";")
    media_type = parts[0].strip().lower()
    if media_type == "application/json":
        valid_media_type = True
    elif media_type.startswith("application/") and media_type.endswith("+json"):
        vendor_token = media_type[len("application/") : -len("+json")]
        valid_media_type = (
            bool(vendor_token)
            and "*" not in vendor_token
            and all(character in _MIME_TOKEN_CHARACTERS for character in vendor_token)
        )
    else:
        valid_media_type = False
    if not valid_media_type:
        return False
    return all(_valid_mime_parameter(parameter) for parameter in parts[1:])


def _is_atom_content_type(value: str | None) -> bool:
    if type(value) is not str or "," in value:
        return False
    parts = value.split(";")
    if parts[0].strip().lower() != "application/atom+xml":
        return False
    return all(_valid_mime_parameter(parameter) for parameter in parts[1:])


def _valid_mime_parameter(raw_parameter: str) -> bool:
    parameter = raw_parameter.strip()
    name, separator, raw_value = parameter.partition("=")
    name = name.strip().lower()
    raw_value = raw_value.strip()
    if (
        not separator
        or not name
        or "*" in name
        or any(character not in _MIME_TOKEN_CHARACTERS for character in name)
        or not raw_value
    ):
        return False
    if raw_value.startswith('"') or raw_value.endswith('"'):
        if len(raw_value) < 2 or not (raw_value.startswith('"') and raw_value.endswith('"')):
            return False
        escaped = False
        for character in raw_value[1:-1]:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"' or ord(character) < 32 or ord(character) == 127:
                return False
        return not escaped
    return all(character.lower() in _MIME_TOKEN_CHARACTERS for character in raw_value)


def _checked_response(
    response: object,
    *,
    content_type_matches: Callable[[str | None], bool] = _is_json_content_type,
) -> DiscoveryGatewayResponse:
    """Validate status and MIME without inspecting rejected bodies."""
    if type(response) is not DiscoveryGatewayResponse or type(response.status_code) is not int:
        raise DiscoveryAdapterError("provider_response_rejected")
    if response.status_code == 429:
        retry_after = response.retry_after
        if type(retry_after) is str:
            try:
                return_error = DiscoveryAdapterError(
                    "provider_rate_limited",
                    retry_after=retry_after,
                )
            except (TypeError, ValueError):
                return_error = DiscoveryAdapterError("provider_rate_limited")
        else:
            return_error = DiscoveryAdapterError("provider_rate_limited")
        raise return_error
    if response.status_code != 200:
        raise DiscoveryAdapterError("provider_response_rejected")
    if not content_type_matches(_response_content_type(response)):
        raise DiscoveryAdapterError("provider_response_rejected")
    return response


def _require_dict(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise _PayloadInvalid
    return value


def _require_list(value: Any) -> list[Any]:
    if type(value) is not list:
        raise _PayloadInvalid
    return value


def _required_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key, _MISSING)
    if type(value) is not str or not value.strip():
        raise _PayloadInvalid
    return value


def _optional_text(record: dict[str, Any], key: str) -> str | None:
    value = record.get(key, _MISSING)
    if value is _MISSING or value is None:
        return None
    if type(value) is not str:
        raise _PayloadInvalid
    return value


def _nonnegative_integer(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise _PayloadInvalid
    return value


def _positive_integer(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise _PayloadInvalid
    return value


def _bounded_cursor(value: Any, *, greater_than: int | None = None) -> int:
    if type(value) is not int or not 0 <= value <= MAX_PAGINATION_CURSOR:
        raise _PayloadInvalid
    if greater_than is not None and value <= greater_than:
        raise _PayloadInvalid
    return value


def _safe_url(value: Any) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise _PayloadInvalid
    try:
        if "\\" in value or any(ord(character) < 32 or ord(character) == 127 for character in value):
            return None
        if has_unsafe_url_material(value):
            return None
        parsed = urlsplit(value)
        hostname = parsed.hostname
        if parsed.scheme.lower() not in {"http", "https"} or not hostname:
            return None
        if (
            "%" in parsed.netloc
            or not parsed.netloc.isascii()
            or any(ord(character) <= 32 or ord(character) == 127 for character in parsed.netloc)
            or parsed.netloc.endswith(":")
        ):
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        port = parsed.port
        if port == 0:
            return None
        if _decoded_path_is_unsafe(parsed):
            return None
    except ValueError:
        return None
    normalized_host = hostname.rstrip(".").lower()
    if (
        not _valid_ascii_dns_hostname(normalized_host)
        or normalized_host == "localhost"
        or normalized_host.endswith(".localhost")
    ):
        return None
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        address = None
    if address is None and _looks_ambiguous_numeric_host(normalized_host):
        return None
    if address is not None and (address.version == 6 or not address.is_global):
        return None
    return canonicalize_url(value)


def _decoded_path_is_unsafe(parsed: Any) -> bool:
    path = parsed.path
    for _ in range(_URL_PATH_DECODE_PASSES):
        if "\\" in path or any(ord(character) < 32 or ord(character) == 127 for character in path):
            return True
        candidate = parsed._replace(path=path, query="", fragment="").geturl()
        if has_unsafe_url_material(candidate):
            return True
        decoded = unquote(path)
        if decoded == path:
            return False
        path = decoded
    return True


def _looks_ambiguous_numeric_host(hostname: str) -> bool:
    labels = hostname.split(".")
    return bool(labels) and all(_looks_like_numeric_host_label(label) for label in labels)


def _looks_like_numeric_host_label(label: str) -> bool:
    if label.isdecimal():
        return True
    return label.startswith("0x") and len(label) > 2 and all(character in "0123456789abcdef" for character in label[2:])


def _valid_ascii_dns_hostname(hostname: str) -> bool:
    if not hostname or not hostname.isascii() or len(hostname) > 253:
        return False
    for label in hostname.split("."):
        if not label or len(label) > 63 or label.startswith("-") or label.endswith("-"):
            return False
        if not all(character.isascii() and (character.isalnum() or character == "-") for character in label):
            return False
        try:
            if label.encode("idna").decode("ascii").lower() != label:
                return False
        except UnicodeError:
            return False
    return True


def _normalized_doi(value: Any) -> str | None:
    if value is None or value is _MISSING:
        return None
    if type(value) is not str:
        raise _PayloadInvalid
    return normalize_doi(value)


def _base_record(
    *,
    title: str,
    authors: tuple[str, ...],
    abstract: str | None,
    snippet: str | None,
    doi: str | None,
    pmid: str | None,
    pmcid: str | None,
    arxiv_id: str | None,
    url: str | None,
    pdf_url: str | None,
    provider: str,
    provider_ids: dict[str, str],
) -> dict[str, Any]:
    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "snippet": snippet,
        "doi": doi,
        "pmid": pmid,
        "pmcid": pmcid,
        "arxiv_id": arxiv_id,
        "url": url,
        "pdf_url": pdf_url,
        "provider": provider,
        "provider_ids": provider_ids,
    }


def _canonical_decimal_text(
    value: Any,
    profile: _ParsingProfile,
    *,
    positive: bool = False,
    maximum: int | None = None,
) -> int:
    """Parse one canonical ASCII decimal string within explicit bounds."""
    if (
        type(value) is not str
        or not value
        or len(value) > profile.max_numeric_token_chars
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise _PayloadInvalid
    parsed = int(value)
    if str(parsed) != value or (positive and parsed <= 0) or (maximum is not None and parsed > maximum):
        raise _PayloadInvalid
    return parsed


def _pubmed_id(value: Any, max_chars: int) -> tuple[str, int]:
    """Return one canonical PMID string and its safe numeric binding value."""
    if type(value) is not str or len(value) > max_chars or _PUBMED_ID_RE.fullmatch(value) is None:
        raise _PayloadInvalid
    return value, int(value)


def _ncbi_json_root(payload: Any, expected_type: str) -> dict[str, Any]:
    """Validate one versioned NCBI JSON envelope without exposing provider detail."""
    root = _require_dict(payload)
    if "error" in root:
        if root.get("error") == "API rate limit exceeded":
            raise DiscoveryAdapterError("provider_rate_limited")
        raise DiscoveryAdapterError("provider_response_rejected")
    header = _require_dict(root.get("header", _MISSING))
    if header.get("type") != expected_type or header.get("version") != _NCBI_JSON_VERSION:
        raise _PayloadInvalid
    return root


def _validate_ncbi_message_list(record: dict[str, Any], key: str) -> None:
    """Validate one optional NCBI message-list object without exposing text."""
    value = record.get(key, _MISSING)
    if value is _MISSING:
        return
    messages = _require_dict(value)
    for raw_values in messages.values():
        values = _require_list(raw_values)
        if any(type(item) is not str for item in values):
            raise _PayloadInvalid


def _pubmed_article_ids(
    raw: Any,
    expected_pmid: str,
    guard: _ParseGuard,
) -> tuple[str | None, str | None]:
    """Extract canonical DOI and PMCID values from one PubMed DocSum."""
    article_ids = _require_list(raw)
    if len(article_ids) > _MAX_PUBMED_ARTICLE_IDS_PER_RECORD:
        raise _ParseLimitExceeded
    recognized: dict[str, str] = {}
    for raw_identifier in _guarded_items(article_ids, guard):
        identifier = _require_dict(raw_identifier)
        id_type = _required_text(identifier, "idtype").casefold()
        value = identifier.get("value", _MISSING)
        if value is _MISSING:
            value = identifier.get("id", _MISSING)
        if type(value) is not str or not value:
            raise _PayloadInvalid
        if id_type not in {"pubmed", "doi", "pmc"}:
            continue
        if id_type in recognized:
            raise _PayloadInvalid
        recognized[id_type] = value

    pubmed_value = recognized.get("pubmed")
    if pubmed_value is None or _pubmed_id(pubmed_value, 16)[0] != expected_pmid:
        raise _PayloadInvalid

    raw_doi = recognized.get("doi")
    doi = None if raw_doi is None else normalize_doi(raw_doi)
    if raw_doi is not None and doi is None:
        raise _PayloadInvalid

    pmcid = recognized.get("pmc")
    if pmcid is not None and _PMCID_RE.fullmatch(pmcid) is None:
        raise _PayloadInvalid
    return doi, pmcid


def _pubmed_record(raw: Any, expected_pmid: str, guard: _ParseGuard) -> dict[str, Any]:
    """Normalize one exact PubMed ESummary DocSum to the V2 record shape."""
    record = _require_dict(raw)
    if "error" in record:
        raise DiscoveryAdapterError("provider_response_rejected")
    uid = _required_text(record, "uid")
    if _pubmed_id(uid, 16)[0] != expected_pmid:
        raise _PayloadInvalid
    title = _required_text(record, "title")
    authors_raw = _require_list(record.get("authors", []))
    if len(authors_raw) > _MAX_PUBMED_AUTHORS_PER_RECORD:
        raise _ParseLimitExceeded
    authors = tuple(_required_text(_require_dict(author), "name") for author in _guarded_items(authors_raw, guard))
    doi, pmcid = _pubmed_article_ids(record.get("articleids", _MISSING), expected_pmid, guard)
    provider_ids = {"pubmed_id": expected_pmid, "pmid": expected_pmid}
    if doi is not None:
        provider_ids["doi"] = doi
    if pmcid is not None:
        provider_ids["pmcid"] = pmcid
    return _base_record(
        title=title,
        authors=authors,
        abstract=None,
        snippet=None,
        doi=doi,
        pmid=expected_pmid,
        pmcid=pmcid,
        arxiv_id=None,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{expected_pmid}/",
        pdf_url=(None if pmcid is None else f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"),
        provider="pubmed",
        provider_ids=provider_ids,
    )


def _direct_xml_children(element: ElementTree.Element, tag: str) -> tuple[ElementTree.Element, ...]:
    if type(element) is not ElementTree.Element:
        raise _PayloadInvalid
    return tuple(child for child in element if type(child) is ElementTree.Element and child.tag == tag)


def _single_xml_child(
    element: ElementTree.Element,
    tag: str,
    *,
    required: bool,
) -> ElementTree.Element | None:
    children = _direct_xml_children(element, tag)
    if len(children) > 1 or (required and not children):
        raise _PayloadInvalid
    return children[0] if children else None


def _xml_scalar(element: ElementTree.Element, *, required: bool) -> str | None:
    if type(element) is not ElementTree.Element or element.attrib or len(element):
        raise _PayloadInvalid
    text = element.text
    if text is None:
        value = ""
    elif type(text) is str:
        value = " ".join(text.split())
    else:
        raise _PayloadInvalid
    if required and not value:
        raise _PayloadInvalid
    return value or None


def _xml_integer(element: ElementTree.Element, profile: _ParsingProfile) -> int:
    value = _xml_scalar(element, required=True)
    if type(value) is not str:
        raise _PayloadInvalid
    if len(value) > profile.max_numeric_token_chars:
        raise _ParseLimitExceeded
    if not value.isascii() or not value.isdecimal():
        raise _PayloadInvalid
    return _bounded_cursor(int(value))


def _arxiv_identifier_from_url(value: str, *, path_kind: str) -> str | None:
    if (
        type(value) is not str
        or path_kind not in {"abs", "pdf"}
        or "%" in value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or parsed.netloc.lower() != "arxiv.org"
            or parsed.hostname is None
            or parsed.hostname.lower() != "arxiv.org"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.query
            or parsed.fragment
        ):
            return None
    except ValueError:
        return None
    prefix = f"/{path_kind}/"
    if not parsed.path.startswith(prefix):
        return None
    identifier = parsed.path[len(prefix) :]
    if path_kind == "pdf" and identifier.lower().endswith(".pdf"):
        identifier = identifier[:-4]
    if not identifier or not identifier.isascii() or _ARXIV_ID_RE.fullmatch(identifier) is None:
        return None
    return identifier


def _arxiv_pdf_url(entry: ElementTree.Element, arxiv_id: str) -> str | None:
    for link in _direct_xml_children(entry, _ATOM_LINK):
        attributes = link.attrib
        if (
            attributes.get("rel") != "related"
            or attributes.get("title") != "pdf"
            or attributes.get("type", "").lower() != "application/pdf"
        ):
            continue
        href = attributes.get("href")
        pdf_id = _arxiv_identifier_from_url(href, path_kind="pdf") if type(href) is str else None
        if pdf_id is None:
            continue
        if _ARXIV_VERSION_RE.search(arxiv_id) is None:
            identifiers_match = _ARXIV_VERSION_RE.sub("", pdf_id).casefold() == arxiv_id.casefold()
        else:
            identifiers_match = pdf_id.casefold() == arxiv_id.casefold()
        if not identifiers_match:
            continue
        return f"https://arxiv.org/pdf/{pdf_id}"
    return None


def _check_arxiv_entry_fields(entry: ElementTree.Element, guard: _ParseGuard) -> None:
    fields = len(entry.attrib)
    if fields > _MAX_ARXIV_FIELDS_PER_ENTRY:
        raise _ParseLimitExceeded
    stack = list(entry)
    while stack:
        element = stack.pop()
        fields += 1 + len(element.attrib)
        if fields > _MAX_ARXIV_FIELDS_PER_ENTRY:
            raise _ParseLimitExceeded
        if fields % _CLOCK_CHECK_INTERVAL == 0:
            guard.checkpoint()
        stack.extend(element)


def _arxiv_record(raw: Any, guard: _ParseGuard) -> dict[str, Any]:
    if type(raw) is not ElementTree.Element or raw.tag != _ATOM_ENTRY:
        raise _PayloadInvalid
    _check_arxiv_entry_fields(raw, guard)

    id_element = _single_xml_child(raw, _ATOM_ID, required=True)
    title_element = _single_xml_child(raw, _ATOM_TITLE, required=True)
    summary_element = _single_xml_child(raw, _ATOM_SUMMARY, required=False)
    if id_element is None or title_element is None:
        raise _PayloadInvalid
    entry_id_url = _xml_scalar(id_element, required=True)
    title = _xml_scalar(title_element, required=True)
    abstract = None if summary_element is None else _xml_scalar(summary_element, required=False)
    if type(entry_id_url) is not str or type(title) is not str:
        raise _PayloadInvalid
    arxiv_id = _arxiv_identifier_from_url(entry_id_url, path_kind="abs")
    if arxiv_id is None:
        raise _PayloadInvalid

    author_elements = _direct_xml_children(raw, _ATOM_AUTHOR)
    authors: list[str] = []
    for author in _guarded_items(list(author_elements), guard):
        name_element = _single_xml_child(author, _ATOM_NAME, required=True)
        if name_element is None:
            raise _PayloadInvalid
        name = _xml_scalar(name_element, required=True)
        if type(name) is not str:
            raise _PayloadInvalid
        authors.append(name)

    doi_element = _single_xml_child(raw, _ARXIV_DOI, required=False)
    if doi_element is None:
        doi = None
    else:
        doi_text = _xml_scalar(doi_element, required=True)
        doi = _normalized_doi(doi_text)
    provider_ids = {"arxiv_id": arxiv_id}
    if doi is not None:
        provider_ids["doi"] = doi
    return _base_record(
        title=title,
        authors=tuple(authors),
        abstract=abstract,
        snippet=abstract,
        doi=doi,
        pmid=None,
        pmcid=None,
        arxiv_id=arxiv_id,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=_arxiv_pdf_url(raw, arxiv_id),
        provider="arxiv",
        provider_ids=provider_ids,
    )


def _guarded_items(values: list[Any], guard: _ParseGuard) -> Iterator[Any]:
    for index, value in enumerate(values):
        if index % _CLOCK_CHECK_INTERVAL == 0:
            guard.checkpoint()
        yield value


def _semantic_scholar_record(raw: Any, guard: _ParseGuard) -> dict[str, Any]:
    record = _require_dict(raw)
    paper_id = _required_text(record, "paperId")
    title = _required_text(record, "title")
    authors_raw = _require_list(record.get("authors", []))
    authors = tuple(_required_text(_require_dict(author), "name") for author in _guarded_items(authors_raw, guard))
    abstract = _optional_text(record, "abstract")

    tldr = record.get("tldr", _MISSING)
    if tldr is _MISSING or tldr is None:
        snippet = abstract
    else:
        tldr_record = _require_dict(tldr)
        snippet = _optional_text(tldr_record, "text") or abstract

    external_ids = _require_dict(record.get("externalIds", {}))
    doi = _normalized_doi(external_ids.get("DOI", _MISSING))
    pmid = _optional_text(external_ids, "PubMed")
    pmcid = _optional_text(external_ids, "PubMedCentral")
    arxiv_id = _optional_text(external_ids, "ArXiv")
    provider_ids = {"semantic_scholar_id": paper_id}
    for key, value in (
        ("doi", doi),
        ("pmid", pmid),
        ("pmcid", pmcid),
        ("arxiv_id", arxiv_id),
    ):
        if value is not None:
            provider_ids[key] = value

    open_access = record.get("openAccessPdf", _MISSING)
    if open_access is _MISSING or open_access is None:
        pdf_url = None
    else:
        pdf_url = _safe_url(_require_dict(open_access).get("url"))
    return _base_record(
        title=title,
        authors=authors,
        abstract=abstract,
        snippet=snippet,
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        arxiv_id=arxiv_id,
        url=_safe_url(record.get("url")),
        pdf_url=pdf_url,
        provider="semantic_scholar",
        provider_ids=provider_ids,
    )


def _crossref_record(raw: Any, guard: _ParseGuard) -> dict[str, Any]:
    record = _require_dict(raw)
    raw_doi = _required_text(record, "DOI")
    doi = normalize_doi(raw_doi)
    if doi is None:
        raise _PayloadInvalid
    titles = _require_list(record.get("title", _MISSING))
    if not titles or any(type(title) is not str for title in _guarded_items(titles, guard)) or not titles[0].strip():
        raise _PayloadInvalid
    title = titles[0]
    authors_raw = _require_list(record.get("author", []))
    authors: list[str] = []
    for raw_author in _guarded_items(authors_raw, guard):
        author = _require_dict(raw_author)
        given = _optional_text(author, "given")
        family = _optional_text(author, "family")
        name = " ".join(part for part in (given, family) if part)
        if name:
            authors.append(name)
    abstract = _optional_text(record, "abstract")
    links = _require_list(record.get("link", []))
    pdf_url = None
    for raw_link in _guarded_items(links, guard):
        link = _require_dict(raw_link)
        content_type = _optional_text(link, "content-type")
        link_url = link.get("URL", _MISSING)
        if link_url is not _MISSING and link_url is not None and type(link_url) is not str:
            raise _PayloadInvalid
        if content_type and content_type.lower() == "application/pdf":
            pdf_url = _safe_url(None if link_url is _MISSING else link_url)
            if pdf_url is not None:
                break
    return _base_record(
        title=title,
        authors=tuple(authors),
        abstract=abstract,
        snippet=abstract,
        doi=doi,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        url=_safe_url(record.get("URL")),
        pdf_url=pdf_url,
        provider="crossref",
        provider_ids={"crossref_id": doi, "doi": doi},
    )


def _zenodo_record(raw: Any, guard: _ParseGuard) -> dict[str, Any]:
    record = _require_dict(raw)
    record_id = _positive_integer(record.get("id", _MISSING))
    metadata = _require_dict(record.get("metadata", _MISSING))
    title = _required_text(metadata, "title")
    abstract = _optional_text(metadata, "description")
    record_doi = _normalized_doi(record.get("doi", _MISSING))
    metadata_doi = _normalized_doi(metadata.get("doi", _MISSING))
    if record_doi is not None and metadata_doi is not None and record_doi != metadata_doi:
        raise _PayloadInvalid
    doi = record_doi or metadata_doi
    creators = _require_list(metadata.get("creators", []))
    authors = tuple(_required_text(_require_dict(creator), "name") for creator in _guarded_items(creators, guard))
    links = _require_dict(record.get("links", _MISSING))
    files = _require_list(record.get("files", []))
    pdf_url = None
    for raw_file in _guarded_items(files, guard):
        file_record = _require_dict(raw_file)
        key = _optional_text(file_record, "key")
        if key is not None and key.lower().endswith(".pdf"):
            file_links = _require_dict(file_record.get("links", _MISSING))
            pdf_url = _safe_url(file_links.get("self"))
            if pdf_url is not None:
                break
    provider_ids = {"zenodo_id": str(record_id)}
    if doi is not None:
        provider_ids["doi"] = doi
    return _base_record(
        title=title,
        authors=authors,
        abstract=abstract,
        snippet=abstract,
        doi=doi,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        url=_safe_url(links.get("self_html")),
        pdf_url=pdf_url,
        provider="zenodo",
        provider_ids=provider_ids,
    )


def _figshare_record(raw: Any, _guard: _ParseGuard) -> dict[str, Any]:
    record = _require_dict(raw)
    record_id = _positive_integer(record.get("id", _MISSING))
    title = _required_text(record, "title")
    doi = _normalized_doi(record.get("doi", _MISSING))
    provider_ids = {"figshare_id": str(record_id)}
    if doi is not None:
        provider_ids["doi"] = doi
    return _base_record(
        title=title,
        authors=(),
        abstract=None,
        snippet=None,
        doi=doi,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        url=_safe_url(record.get("url_public_html")),
        pdf_url=None,
        provider="figshare",
        provider_ids=provider_ids,
    )


def _osf_record(raw: Any, _guard: _ParseGuard) -> dict[str, Any]:
    record = _require_dict(raw)
    if record.get("type") != "preprints" or type(record.get("type")) is not str:
        raise _PayloadInvalid
    record_id = _required_text(record, "id")
    attributes = _require_dict(record.get("attributes", _MISSING))
    links = _require_dict(record.get("links", _MISSING))
    title = _required_text(attributes, "title")
    abstract = _optional_text(attributes, "description")
    doi = _normalized_doi(attributes.get("doi", _MISSING))
    provider_ids = {"osf_id": record_id}
    if doi is not None:
        provider_ids["doi"] = doi
    return _base_record(
        title=title,
        authors=(),
        abstract=abstract,
        snippet=abstract,
        doi=doi,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        url=_safe_url(links.get("html")),
        pdf_url=None,
        provider="osf",
        provider_ids=provider_ids,
    )


_NORMALIZERS = MappingProxyType(
    {
        "semantic_scholar_v2": _semantic_scholar_record,
        "crossref_v2": _crossref_record,
        "zenodo_v2": _zenodo_record,
        "figshare_v2": _figshare_record,
        "osf_v2": _osf_record,
        "arxiv_v2": _arxiv_record,
    }
)


def _validate_page_cardinality(records: list[Any], page_size: int) -> None:
    if len(records) > page_size:
        raise _PayloadInvalid


def _semantic_scholar_page(payload: Any, current: int, _seen: int, page_size: int) -> tuple[list[Any], int | None]:
    root = _require_dict(payload)
    records = _require_list(root.get("data", _MISSING))
    total = _nonnegative_integer(root.get("total", _MISSING))
    offset = _nonnegative_integer(root.get("offset", _MISSING))
    if offset > MAX_PAGINATION_CURSOR or offset != current or offset + len(records) > total:
        raise _PayloadInvalid
    next_value = root.get("next", _MISSING)
    if next_value is _MISSING or next_value is None:
        return records, None
    next_offset = _bounded_cursor(next_value, greater_than=current)
    if next_offset > total:
        raise _PayloadInvalid
    return records, next_offset


def _crossref_page(payload: Any, current: int, _seen: int, page_size: int) -> tuple[list[Any], int | None]:
    root = _require_dict(payload)
    if root.get("status") != "ok" or root.get("message-type") != "work-list" or root.get("message-version") != "1.0.0":
        raise _PayloadInvalid
    message = _require_dict(root.get("message", _MISSING))
    total = _nonnegative_integer(message.get("total-results", _MISSING))
    records = _require_list(message.get("items", _MISSING))
    next_value = current + len(records)
    if next_value > total:
        raise _PayloadInvalid
    if next_value < total:
        if not records:
            raise _PayloadInvalid
        return records, _bounded_cursor(next_value, greater_than=current)
    return records, None


def _zenodo_page(payload: Any, current: int, seen: int, page_size: int) -> tuple[list[Any], int | None]:
    root = _require_dict(payload)
    hits = _require_dict(root.get("hits", _MISSING))
    records = _require_list(hits.get("hits", _MISSING))
    total = _nonnegative_integer(hits.get("total", _MISSING))
    consumed = seen + len(records)
    if consumed > total:
        raise _PayloadInvalid
    if consumed < total:
        if not records:
            raise _PayloadInvalid
        return records, _bounded_cursor(current + 1, greater_than=current)
    return records, None


def _figshare_page(payload: Any, current: int, _seen: int, page_size: int) -> tuple[list[Any], int | None]:
    records = _require_list(payload)
    if records and len(records) == page_size:
        return records, _bounded_cursor(current + 1, greater_than=current)
    return records, None


def _osf_page(payload: Any, current: int, _seen: int, page_size: int) -> tuple[list[Any], int | None]:
    root = _require_dict(payload)
    records = _require_list(root.get("data", _MISSING))
    links = _require_dict(root.get("links", _MISSING))
    next_value = links.get("next", _MISSING)
    if next_value is _MISSING:
        raise _PayloadInvalid
    if next_value is None:
        return records, None
    if type(next_value) is not str:
        raise _PayloadInvalid
    return records, _bounded_cursor(current + 1, greater_than=current)


def _arxiv_page(
    payload: Any,
    current: int,
    _seen: int,
    page_size: int,
    profile: _ParsingProfile,
    max_records: int,
) -> tuple[list[Any], int | None]:
    if type(payload) is not ElementTree.Element or payload.tag != _ATOM_FEED:
        raise _PayloadInvalid
    total_element = _single_xml_child(payload, _OPEN_SEARCH_TOTAL, required=True)
    start_element = _single_xml_child(payload, _OPEN_SEARCH_START, required=True)
    items_element = _single_xml_child(payload, _OPEN_SEARCH_ITEMS, required=True)
    if total_element is None or start_element is None or items_element is None:
        raise _PayloadInvalid
    total = _xml_integer(total_element, profile)
    start = _xml_integer(start_element, profile)
    items = _xml_integer(items_element, profile)
    records = list(_direct_xml_children(payload, _ATOM_ENTRY))
    if len(records) > max_records:
        raise _ParseLimitExceeded
    if start != current or items > page_size or len(records) > items or start > total or start + len(records) > total:
        raise _PayloadInvalid
    next_value = start + len(records)
    if next_value < total:
        if not records:
            raise _PayloadInvalid
        return records, _bounded_cursor(next_value, greater_than=current)
    return records, None


_PAGE_READERS = MappingProxyType(
    {
        "semantic_scholar_v2": _semantic_scholar_page,
        "crossref_v2": _crossref_page,
        "zenodo_v2": _zenodo_page,
        "figshare_v2": _figshare_page,
        "osf_v2": _osf_page,
    }
)


def _query_integer(group: PlannedDispatchGroup, name: str) -> int:
    values = tuple(pair.value for pair in group.intents[0].query_pairs if pair.name == name)
    if len(values) != 1 or type(values[0]) is not str or not values[0].isascii() or not values[0].isdecimal():
        raise _PayloadInvalid
    if len(values[0]) > len(str(MAX_PAGINATION_CURSOR)):
        raise _PayloadInvalid
    return _bounded_cursor(int(values[0]))


def _body_integer(group: PlannedDispatchGroup, name: str) -> int:
    values = tuple(pair.value for pair in group.intents[0].json_body_pairs if pair.name == name)
    if len(values) != 1:
        raise _PayloadInvalid
    return _bounded_cursor(values[0])


def _initial_page_and_size(group: PlannedDispatchGroup) -> tuple[int, int]:
    adapter_id = group.adapter_id
    if adapter_id == "semantic_scholar_v2":
        return _query_integer(group, "offset"), _query_integer(group, "limit")
    if adapter_id == "crossref_v2":
        return _query_integer(group, "offset"), _query_integer(group, "rows")
    if adapter_id == "zenodo_v2":
        return _query_integer(group, "page"), _query_integer(group, "size")
    if adapter_id == "figshare_v2":
        return _body_integer(group, "page"), _body_integer(group, "page_size")
    if adapter_id == "osf_v2":
        return _query_integer(group, "page"), _query_integer(group, "page[size]")
    if adapter_id == "arxiv_v2":
        return _query_integer(group, "start"), _query_integer(group, "max_results")
    raise _PayloadInvalid


def _is_sealed_identity_pubmed_group(group: PlannedDispatchGroup) -> bool:
    """Validate all non-secret identity-overlay trust material before dispatch."""
    if (
        type(group.route_id) is not str
        or group.route_id != _PUBMED_ROUTE_ID
        or type(group.backend_id) is not str
        or group.backend_id != _PUBMED_BACKEND_ID
        or type(group.adapter_id) is not str
        or group.adapter_id != "pubmed_v2"
        or type(group.adapter_version) is not str
        or group.adapter_version != _PUBMED_IDENTITY_ADAPTER_VERSION
        or type(group.policy_digest) is not str
        or group.policy_digest != _PUBMED_IDENTITY_POLICY_DIGEST
        or type(group.normalized_query) is not str
        or not group.normalized_query
        or not _is_sealed_identity_pubmed_filters(group.filters)
        or type(group.fallback_order) is not int
        or group.fallback_order != 0
        or type(group.limits) is not RouteLimits
        or not _is_exact_identity_pubmed_limits(group.limits)
        or type(group.allowance) is not DispatchAllowance
        or not _is_exact_identity_pubmed_allowance(group.allowance)
        or type(group.logical_attempts) is not tuple
        or len(group.logical_attempts) != 1
        or type(group.intents) is not tuple
        or len(group.intents) != 2
    ):
        return False
    logical = group.logical_attempts[0]
    if (
        type(logical) is not PlannedLogicalAttempt
        or type(logical.logical_attempt_id) is not str
        or not logical.logical_attempt_id
        or type(logical.catalog_source_id) is not str
        or logical.catalog_source_id != "pubmed"
        or type(logical.selection_reason) is not str
        or logical.selection_reason != "explicit"
        or logical.source_predicate is not None
    ):
        return False
    search, summary = group.intents
    if type(search) is not DispatchIntent or type(summary) is not DispatchIntent:
        return False
    for intent in (search, summary):
        if (
            type(intent.route_id) is not str
            or intent.route_id != group.route_id
            or type(intent.policy_digest) is not str
            or intent.policy_digest != group.policy_digest
            or type(intent.method) is not str
            or type(intent.path) is not str
            or type(intent.limits) is not RouteLimits
            or not _is_exact_identity_pubmed_limits(intent.limits)
            or intent.limits != group.limits
            or type(intent.query_pairs) is not tuple
            or any(type(pair) is not QueryPair for pair in intent.query_pairs)
            or type(intent.json_body_pairs) is not tuple
            or intent.json_body_pairs != ()
            or type(intent.query_bindings) is not tuple
        ):
            return False
    if len(search.query_pairs) < 2:
        return False
    search_term = search.query_pairs[1]
    return (
        type(search_term.name) is str
        and search_term.name == "term"
        and type(search_term.value) is str
        and search_term.value == group.normalized_query
    )


def _is_exact_identity_pubmed_limits(limits: RouteLimits) -> bool:
    values = (
        limits.max_pages,
        limits.max_redirects,
        limits.max_retries,
        limits.timeout_ms,
        limits.max_response_bytes,
        limits.max_results,
        limits.max_request_body_bytes,
    )
    return all(type(value) is int for value in values) and values == (
        1,
        0,
        0,
        20_000,
        2_097_152,
        100,
        16_384,
    )


def _is_exact_identity_pubmed_allowance(allowance: DispatchAllowance) -> bool:
    values = (
        allowance.physical_dispatches,
        allowance.pages,
        allowance.redirects,
        allowance.retries,
    )
    return all(type(value) is int for value in values) and values == (2, 1, 0, 0)


def _is_sealed_identity_pubmed_filters(filters: object) -> bool:
    """Preserve legacy filter metadata without permitting identity overrides."""
    return type(filters) is tuple and all(
        type(pair) is QueryPair
        and type(pair.name) is str
        and pair.name not in {"tool", "email"}
        and type(pair.value) is str
        for pair in filters
    )


def _trusted_pubmed_inputs(
    group: object,
) -> _TrustedNCBIInputs:
    """Validate the exact two-intent PubMed adapter contract before dispatch."""
    if type(group) is not PlannedDispatchGroup or group.adapter_id != "pubmed_v2":
        raise DiscoveryAdapterError("provider_payload_invalid")
    profile = _PARSING_PROFILES.get((group.adapter_id, group.adapter_version))
    if profile is None or type(group.intents) is not tuple or len(group.intents) != 2:
        raise DiscoveryAdapterError("provider_payload_invalid")
    if group.adapter_version == _PUBMED_IDENTITY_ADAPTER_VERSION and not _is_sealed_identity_pubmed_group(group):
        raise DiscoveryAdapterError("provider_payload_invalid")
    search, summary = group.intents
    limits = group.limits
    if (
        type(search) is not DispatchIntent
        or type(summary) is not DispatchIntent
        or type(limits) is not RouteLimits
        or type(search.query_pairs) is not tuple
        or type(summary.query_pairs) is not tuple
        or type(search.json_body_pairs) is not tuple
        or type(summary.json_body_pairs) is not tuple
        or type(search.query_bindings) is not tuple
        or type(summary.query_bindings) is not tuple
        or any(type(pair) is not QueryPair for pair in (*search.query_pairs, *summary.query_pairs))
        or any(
            type(pair.name) is not str or type(pair.value) is not str
            for pair in (*search.query_pairs, *summary.query_pairs)
        )
        or len(summary.query_bindings) != 1
        or type(summary.query_bindings[0]) is not DeferredNumericCSVQueryBinding
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    if (
        search.operation_kind is not OperationKind.SEARCH
        or summary.operation_kind is not OperationKind.CONDITIONAL_SUMMARY
        or search.method != "GET"
        or summary.method != "GET"
        or search.path != "/entrez/eutils/esearch.fcgi"
        or summary.path != "/entrez/eutils/esummary.fcgi"
        or search.json_body_pairs
        or search.query_bindings
        or summary.json_body_pairs
        or type(limits.max_response_bytes) is not int
        or limits.max_response_bytes <= 0
        or type(limits.max_results) is not int
        or limits.max_results <= 0
        or type(limits.max_pages) is not int
        or limits.max_pages != 1
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    search_pairs = tuple((pair.name, pair.value) for pair in search.query_pairs)
    summary_pairs = tuple((pair.name, pair.value) for pair in summary.query_pairs)
    if (
        len(search_pairs) not in {6, 8}
        or len(summary_pairs) not in {2, 4}
        or search_pairs[0] != ("db", "pubmed")
        or search_pairs[1][0] != "term"
        or type(search_pairs[1][1]) is not str
        or not search_pairs[1][1]
        or search_pairs[2][0] != "retstart"
        or search_pairs[3][0] != "retmax"
        or len(summary.query_bindings) != 1
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    base_search_pairs = (
        ("db", "pubmed"),
        search_pairs[1],
        search_pairs[2],
        search_pairs[3],
        ("retmode", "json"),
        ("sort", "relevance"),
    )
    base_summary_pairs = (("db", "pubmed"), ("retmode", "json"))
    identity_pairs = (("tool", "tldw_server"), ("email", "contact@tldwproject.com"))
    if group.adapter_version == "foundation-v2":
        shape_valid = search_pairs == base_search_pairs and summary_pairs == base_summary_pairs
    elif group.adapter_version == _PUBMED_IDENTITY_ADAPTER_VERSION:
        shape_valid = (
            search_pairs == base_search_pairs + identity_pairs and summary_pairs == base_summary_pairs + identity_pairs
        )
    else:
        shape_valid = False
    if not shape_valid:
        raise DiscoveryAdapterError("provider_payload_invalid")
    binding = summary.query_bindings[0]
    if (
        type(binding) is not DeferredNumericCSVQueryBinding
        or binding.binding_id != _PUBMED_BINDING_ID
        or binding.query_name != "id"
        or binding.max_item_chars != 16
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    try:
        retstart = _canonical_decimal_text(search_pairs[2][1], profile, maximum=MAX_PAGINATION_CURSOR)
        retmax = _canonical_decimal_text(
            search_pairs[3][1],
            profile,
            positive=True,
            maximum=min(profile.max_records, limits.max_results),
        )
        if retstart != 0 or binding.max_items != retmax:
            raise _PayloadInvalid
    except _PayloadInvalid as error:
        _raise_adapter_error(error)
    return (
        group,
        profile,
        min(profile.max_input_bytes, limits.max_response_bytes),
        retstart,
        retmax,
        binding,
    )


def _pubmed_esearch_ids(
    payload: Any,
    *,
    profile: _ParsingProfile,
    guard: _ParseGuard,
    retstart: int,
    retmax: int,
    binding: DeferredNumericCSVQueryBinding,
) -> tuple[tuple[str, int], ...]:
    """Validate one ESearch page and return canonical PMID bindings."""
    root = _ncbi_json_root(payload, "esearch")
    result = _require_dict(root.get("esearchresult", _MISSING))
    if "ERROR" in result:
        raise DiscoveryAdapterError("provider_response_rejected")
    count = _canonical_decimal_text(result.get("count", _MISSING), profile)
    returned = _canonical_decimal_text(
        result.get("retmax", _MISSING),
        profile,
        maximum=retmax,
    )
    returned_start = _canonical_decimal_text(
        result.get("retstart", _MISSING),
        profile,
        maximum=MAX_PAGINATION_CURSOR,
    )
    raw_ids = _require_list(result.get("idlist", _MISSING))
    _validate_ncbi_message_list(result, "errorlist")
    _validate_ncbi_message_list(result, "warninglist")
    if (
        returned_start != retstart
        or returned != len(raw_ids)
        or returned_start + returned > count
        or (count > 0 and returned == 0)
        or len(raw_ids) > binding.max_items
    ):
        raise _PayloadInvalid
    ids = tuple(_pubmed_id(value, binding.max_item_chars) for value in _guarded_items(raw_ids, guard))
    if len({value for value, _number in ids}) != len(ids):
        raise _PayloadInvalid
    return ids


def _pubmed_summary_records(
    payload: Any,
    *,
    expected_ids: tuple[str, ...],
    guard: _ParseGuard,
) -> tuple[dict[str, Any], ...]:
    """Validate one complete ESummary response in ESearch order."""
    root = _ncbi_json_root(payload, "esummary")
    result = _require_dict(root.get("result", _MISSING))
    raw_uids = _require_list(result.get("uids", _MISSING))
    if (
        len(raw_uids) != len(expected_ids)
        or any(type(uid) is not str for uid in raw_uids)
        or set(raw_uids) != set(expected_ids)
        or set(result) != {"uids", *expected_ids}
    ):
        raise _PayloadInvalid
    records = []
    for expected_id in expected_ids:
        guard.checkpoint()
        records.append(_pubmed_record(result.get(expected_id, _MISSING), expected_id, guard))
    return tuple(records)


def _validate_identity_ncbi_error_envelope(
    payload: Any,
    profile: _ParsingProfile,
) -> None:
    """Classify only the documented JSON rate envelope for identity routes."""
    root = _require_dict(payload)
    if "error" not in root:
        return
    if (
        set(root) == {"error", "count"}
        and root["error"] == "API rate limit exceeded"
        and type(root["count"]) is str
        and len(root["count"]) <= profile.max_numeric_token_chars
        and _NCBI_RATE_COUNT_RE.fullmatch(root["count"]) is not None
    ):
        raise DiscoveryAdapterError("provider_rate_limited")
    raise _PayloadInvalid


async def _execute_ncbi_esearch_summary(
    group: object,
    dispatch: BoundDispatch,
    clock: MonotonicClock,
    *,
    trusted_inputs: _TrustedNCBIInputsCallback,
    parse_esearch_ids: _NCBIESearchIDsCallback,
    parse_summary_records: _NCBISummaryRecordsCallback,
    strict_rate_envelope: bool,
) -> DiscoveryAdapterResult:
    """Execute one sealed ESearch and conditional ESummary pair."""
    try:
        trusted_group, profile, max_input_bytes, retstart, retmax, binding = trusted_inputs(group)
    except DiscoveryAdapterError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        raise DiscoveryAdapterError("provider_payload_invalid") from None
    search, summary = trusted_group.intents
    search_response = _checked_response(await dispatch(search))
    search_payload, search_guard = _strict_json(
        search_response,
        profile=profile,
        max_input_bytes=max_input_bytes,
        clock=clock,
    )
    try:
        if strict_rate_envelope:
            _validate_identity_ncbi_error_envelope(search_payload, profile)
        ids = parse_esearch_ids(
            search_payload,
            profile=profile,
            guard=search_guard,
            retstart=retstart,
            retmax=retmax,
            binding=binding,
        )
        search_guard.checkpoint()
    except (_PayloadInvalid, _ParseLimitExceeded, _ParseDeadlineExceeded) as error:
        _raise_adapter_error(error)
    except DiscoveryAdapterError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        raise DiscoveryAdapterError("provider_payload_invalid") from None
    if not ids:
        return DiscoveryAdapterResult(candidates=())

    expected_ids = tuple(value for value, _number in ids)
    binding_values = NumericCSVBindingValues(
        binding.binding_id,
        tuple(number for _value, number in ids),
    )
    summary_response = _checked_response(await dispatch(summary, bindings=(binding_values,)))
    summary_payload, summary_guard = _strict_json(
        summary_response,
        profile=profile,
        max_input_bytes=max_input_bytes,
        clock=clock,
    )
    try:
        if strict_rate_envelope:
            _validate_identity_ncbi_error_envelope(summary_payload, profile)
        normalized_records = parse_summary_records(
            summary_payload,
            expected_ids=expected_ids,
            guard=summary_guard,
        )
        candidates: list[DiscoveryCandidate] = []
        records_by_id: dict[str, dict[str, Any]] = {}
        for normalized in normalized_records:
            summary_guard.checkpoint()
            fingerprint = build_fingerprint(normalized)
            candidate_id = DiscoveryOutcomeIdentity.from_fingerprint(fingerprint).document_id
            existing = records_by_id.get(candidate_id)
            if existing is not None:
                if existing != normalized:
                    raise _PayloadInvalid
                continue
            records_by_id[candidate_id] = normalized
            candidates.append(DiscoveryCandidate(candidate_id, normalized))
        summary_guard.checkpoint()
    except (_PayloadInvalid, _ParseLimitExceeded, _ParseDeadlineExceeded) as error:
        _raise_adapter_error(error)
    except DiscoveryAdapterError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        raise DiscoveryAdapterError("provider_payload_invalid") from None
    return DiscoveryAdapterResult(tuple(candidates))


async def _execute_pubmed_adapter(
    group: object,
    dispatch: BoundDispatch,
    clock: MonotonicClock,
) -> DiscoveryAdapterResult:
    """Execute the exact foundation or identity-bearing PubMed contract."""
    strict_rate_envelope = (
        type(group) is PlannedDispatchGroup and group.adapter_version == _PUBMED_IDENTITY_ADAPTER_VERSION
    )
    return await _execute_ncbi_esearch_summary(
        group,
        dispatch,
        clock,
        trusted_inputs=_trusted_pubmed_inputs,
        parse_esearch_ids=_pubmed_esearch_ids,
        parse_summary_records=_pubmed_summary_records,
        strict_rate_envelope=strict_rate_envelope,
    )


def _trusted_adapter_inputs(
    adapter_id: str,
    group: object,
) -> tuple[PlannedDispatchGroup, _ParsingProfile, int, int, int, int]:
    if type(group) is not PlannedDispatchGroup:
        raise DiscoveryAdapterError("provider_payload_invalid")
    if group.adapter_id != adapter_id or type(group.adapter_version) is not str:
        raise DiscoveryAdapterError("provider_payload_invalid")
    profile = _PARSING_PROFILES.get((group.adapter_id, group.adapter_version))
    if profile is None or type(group.intents) is not tuple or len(group.intents) != 1:
        raise DiscoveryAdapterError("provider_payload_invalid")
    limits = group.limits
    if (
        type(limits.max_response_bytes) is not int
        or limits.max_response_bytes <= 0
        or type(limits.max_results) is not int
        or limits.max_results <= 0
        or type(limits.max_pages) is not int
        or limits.max_pages <= 0
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    try:
        initial_page, page_size = _initial_page_and_size(group)
        if page_size <= 0:
            raise _PayloadInvalid
    except _PayloadInvalid as error:
        _raise_adapter_error(error)
    return (
        group,
        profile,
        min(profile.max_input_bytes, limits.max_response_bytes),
        min(profile.max_records, limits.max_results),
        initial_page,
        page_size,
    )


async def _execute_adapter(
    adapter_id: str,
    group: object,
    dispatch: BoundDispatch,
    clock: MonotonicClock,
) -> DiscoveryAdapterResult:
    trusted_group, profile, max_input_bytes, max_records, current_page, page_size = _trusted_adapter_inputs(
        adapter_id,
        group,
    )
    intent = trusted_group.intents[0]
    cursor: NumericCursor | None = None
    seen_records = 0
    candidates: list[DiscoveryCandidate] = []
    records_by_id: dict[str, dict[str, Any]] = {}

    for page_index in range(trusted_group.limits.max_pages):
        response = await dispatch(intent, cursor=cursor)
        if adapter_id == "arxiv_v2":
            response = _checked_response(response, content_type_matches=_is_atom_content_type)
            payload, guard = _strict_atom(
                response,
                profile=profile,
                max_input_bytes=max_input_bytes,
                clock=clock,
            )
        else:
            response = _checked_response(response)
            payload, guard = _strict_json(
                response,
                profile=profile,
                max_input_bytes=max_input_bytes,
                clock=clock,
            )
        try:
            if adapter_id == "arxiv_v2":
                raw_records, next_page = _arxiv_page(
                    payload,
                    current_page,
                    seen_records,
                    page_size,
                    profile,
                    max_records - seen_records,
                )
            else:
                raw_records, next_page = _PAGE_READERS[adapter_id](
                    payload,
                    current_page,
                    seen_records,
                    page_size,
                )
            if len(raw_records) > max_records - seen_records:
                raise _ParseLimitExceeded
            _validate_page_cardinality(raw_records, page_size)
            normalized_page: list[tuple[str, dict[str, Any]]] = []
            for raw_record in raw_records:
                guard.checkpoint()
                normalized = _NORMALIZERS[adapter_id](raw_record, guard)
                fingerprint = build_fingerprint(normalized)
                candidate_id = DiscoveryOutcomeIdentity.from_fingerprint(fingerprint).document_id
                normalized_page.append((candidate_id, normalized))

            for candidate_id, normalized in normalized_page:
                existing = records_by_id.get(candidate_id)
                if existing is not None:
                    if existing != normalized:
                        raise _PayloadInvalid
                    continue
                records_by_id[candidate_id] = normalized
                candidates.append(DiscoveryCandidate(candidate_id, normalized))
            seen_records += len(raw_records)
            guard.checkpoint()
        except (_PayloadInvalid, _ParseLimitExceeded, _ParseDeadlineExceeded) as error:
            _raise_adapter_error(error)
        except (KeyError, TypeError, ValueError, OverflowError):
            raise DiscoveryAdapterError("provider_payload_invalid") from None

        if seen_records >= max_records or next_page is None or page_index + 1 >= trusted_group.limits.max_pages:
            break
        current_page = next_page
        cursor = NumericCursor(next_page)

    return DiscoveryAdapterResult(tuple(candidates))


def _adapter(adapter_id: str, clock: MonotonicClock) -> DiscoveryAdapter:
    async def execute(group: PlannedDispatchGroup, dispatch: BoundDispatch) -> DiscoveryAdapterResult:
        if adapter_id == "pubmed_v2":
            return await _execute_pubmed_adapter(group, dispatch, clock)
        return await _execute_adapter(adapter_id, group, dispatch, clock)

    return execute


def foundation_gateway_adapters(
    *,
    monotonic_clock: MonotonicClock = time.monotonic,
) -> Mapping[str, DiscoveryAdapter]:
    """Return the exact credentialless foundation adapter set."""
    return MappingProxyType({adapter_id: _adapter(adapter_id, monotonic_clock) for adapter_id in _ADAPTER_IDS})
