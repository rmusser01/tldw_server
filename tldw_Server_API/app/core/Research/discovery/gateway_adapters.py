"""Bounded, gateway-only JSON adapters for the discovery V2 foundation."""

from __future__ import annotations

import ipaddress
import json
import math
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlsplit

from .contracts import MAX_PAGINATION_CURSOR, DiscoveryOutcomeIdentity, PlannedDispatchGroup
from .executor import (
    BoundDispatch,
    DiscoveryAdapter,
    DiscoveryAdapterError,
    DiscoveryAdapterResult,
    DiscoveryCandidate,
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
)
_PARSING_PROFILES = MappingProxyType(
    {(adapter_id, "foundation-v2"): _FOUNDATION_PROFILE for adapter_id in _ADAPTER_IDS}
)
_CLOCK_CHECK_INTERVAL = 256
_URL_PATH_DECODE_PASSES = 4
_MISSING = object()
_MIME_TOKEN_CHARACTERS = frozenset("!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyz")


class _PayloadInvalid(Exception):
    pass


class _ParseLimitExceeded(Exception):
    pass


class _ParseDeadlineExceeded(Exception):
    pass


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


def _checked_response(response: object) -> DiscoveryGatewayResponse:
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
    if not _is_json_content_type(_response_content_type(response)):
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
    raise _PayloadInvalid


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
        response = _checked_response(await dispatch(intent, cursor=cursor))
        payload, guard = _strict_json(
            response,
            profile=profile,
            max_input_bytes=max_input_bytes,
            clock=clock,
        )
        try:
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

        if next_page is None or page_index + 1 >= trusted_group.limits.max_pages:
            break
        current_page = next_page
        cursor = NumericCursor(next_page)

    return DiscoveryAdapterResult(tuple(candidates))


def _adapter(adapter_id: str, clock: MonotonicClock) -> DiscoveryAdapter:
    async def execute(group: PlannedDispatchGroup, dispatch: BoundDispatch) -> DiscoveryAdapterResult:
        return await _execute_adapter(adapter_id, group, dispatch, clock)

    return execute


def foundation_gateway_adapters(
    *,
    monotonic_clock: MonotonicClock = time.monotonic,
) -> Mapping[str, DiscoveryAdapter]:
    """Return the exact five credentialless foundation JSON adapters."""
    return MappingProxyType({adapter_id: _adapter(adapter_id, monotonic_clock) for adapter_id in _ADAPTER_IDS})
