"""Offline contract tests for the bounded gateway-only arXiv Atom adapter."""

from __future__ import annotations

import asyncio
import http.client
import importlib
import socket
import urllib.request
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from xml.sax.saxutils import escape, quoteattr

import pytest

from tldw_Server_API.app.core.Research.discovery import executor as executor_module
from tldw_Server_API.app.core.Research.discovery.contracts import (
    MAX_PAGINATION_CURSOR,
    BudgetCeilings,
    DiscoveryOutcomeIdentity,
    ExecutionMode,
)
from tldw_Server_API.app.core.Research.discovery.executor import (
    DiscoveryAdapterResult,
    LogicalOutcomeState,
    NumericCursor,
    PhysicalDispatchState,
    execute_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.gateway import (
    DiscoveryGatewayResponse,
    DiscoveryGatewayTrace,
)
from tldw_Server_API.app.core.Research.discovery.identity import build_fingerprint
from tldw_Server_API.app.core.Research.discovery.planner import (
    PlanningRequest,
    compile_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.registry import (
    DiscoveryRegistry,
    foundation_readiness,
    foundation_registry,
)
from tldw_Server_API.app.core.Security.http_hop import HTTPHopLimits

pytestmark = pytest.mark.unit

_ADAPTER_ID = "arxiv_v2"
_ADAPTER_MODULE = "tldw_Server_API.app.core.Research.discovery.gateway_adapters"
_ATOM = "http://www.w3.org/2005/Atom"
_OPEN_SEARCH = "http://a9.com/-/spec/opensearch/1.1/"
_ARXIV = "http://arxiv.org/schemas/atom"
_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "research_discovery_gateway_adapters"
_NORMALIZED_KEYS = {
    "title",
    "authors",
    "abstract",
    "snippet",
    "doi",
    "pmid",
    "pmcid",
    "arxiv_id",
    "url",
    "pdf_url",
    "provider",
    "provider_ids",
}


def _module():
    return importlib.import_module(_ADAPTER_MODULE)


def _fixture(kind: str) -> bytes:
    return (_FIXTURE_ROOT / f"arxiv_{kind}.xml").read_bytes()


def _registry_with_pages(
    max_pages: int,
    *,
    max_response_bytes: int | None = None,
) -> DiscoveryRegistry:
    base = foundation_registry()
    route_id = base.get_source("arxiv").route_references[0].route_id
    routes = []
    for route in base.routes:
        if route.route_id != route_id:
            routes.append(route)
            continue
        limits = replace(
            route.policy.limits,
            max_pages=max_pages,
            max_response_bytes=(
                route.policy.limits.max_response_bytes if max_response_bytes is None else max_response_bytes
            ),
        )
        routes.append(
            replace(
                route,
                max_physical_dispatches=max_pages,
                policy=replace(route.policy, limits=limits, policy_digest=""),
            )
        )
    return DiscoveryRegistry(
        catalog_version=base.catalog_version,
        registry_version=base.registry_version,
        sources=base.sources,
        routes=tuple(routes),
        backends=base.backends,
    )


def _plan_for(
    *,
    max_pages: int = 1,
    result_limit: int = 1,
    max_response_bytes: int | None = None,
):
    registry = _registry_with_pages(
        max_pages,
        max_response_bytes=max_response_bytes,
    )
    plan = compile_discovery_plan(
        PlanningRequest(
            source_ids=("arxiv",),
            query="  BOUNDED   Discovery  ",
            filters=(),
            result_limit=result_limit,
        ),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(
            max_route_attempts=1,
            max_physical_dispatches=max_pages,
            max_pages_per_route=max_pages,
            max_redirects=0,
            max_retries=0,
            max_wall_time_ms=20_000 * max_pages,
            max_results=result_limit,
        ),
    )
    return registry, plan


def _response(
    route,
    intent,
    body: Any,
    *,
    status_code: Any = 200,
    content_type: str | None = "application/atom+xml; charset=utf-8",
    retry_after: Any = None,
    headers: tuple[tuple[Any, Any], ...] | None = None,
) -> DiscoveryGatewayResponse:
    origin = route.policy.origin
    if headers is None:
        headers = () if content_type is None else (("content-type", content_type),)
    body_length = len(body) if hasattr(body, "__len__") else 0
    return DiscoveryGatewayResponse(
        status_code=status_code,
        headers=headers,
        body=body,
        trace=DiscoveryGatewayTrace(
            route_id=route.route_id,
            policy_digest=route.policy.policy_digest,
            scheme=origin.scheme,
            requested_host=origin.host,
            tls_server_name=origin.host,
            port=origin.port,
            method=intent.method,
            path=intent.path,
            query_keys=tuple(pair.name for pair in intent.query_pairs),
            timeout_ms=intent.limits.timeout_ms,
            max_response_bytes=intent.limits.max_response_bytes,
            http_limits=HTTPHopLimits(),
            status_code=status_code,
            resolved_ips=("93.184.216.34",),
            connected_ip="93.184.216.34",
            response_header_bytes=64,
            wire_bytes=body_length,
            decoded_bytes=body_length,
            elapsed_ms=1,
        ),
        redirect_location=None,
        retry_after=retry_after,
    )


class _RecordingDispatch:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[object, object, object]] = []

    async def __call__(self, intent, *, cursor=None, bindings=()):
        self.calls.append((intent, cursor, bindings))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _CountingClock:
    def __init__(self, *, step: float = 0.0) -> None:
        self.value = 0.0
        self.step = step
        self.calls = 0

    def __call__(self) -> float:
        current = self.value
        self.calls += 1
        self.value += self.step
        return current


async def _invoke(
    bodies: list[Any],
    *,
    max_pages: int = 1,
    result_limit: int = 1,
    max_response_bytes: int | None = None,
    statuses: list[object] | None = None,
    content_types: list[str | None] | None = None,
    retry_afters: list[object] | None = None,
    headers: list[tuple[tuple[Any, Any], ...] | None] | None = None,
    monotonic_clock=None,
):
    registry, plan = _plan_for(
        max_pages=max_pages,
        result_limit=result_limit,
        max_response_bytes=max_response_bytes,
    )
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    intent = group.intents[0]
    statuses = statuses or [200] * len(bodies)
    content_types = content_types or ["application/atom+xml; charset=utf-8"] * len(bodies)
    retry_afters = retry_afters or [None] * len(bodies)
    headers = headers or [None] * len(bodies)
    responses = [
        _response(
            route,
            intent,
            body,
            status_code=status,
            content_type=content_type,
            retry_after=retry_after,
            headers=response_headers,
        )
        for body, status, content_type, retry_after, response_headers in zip(
            bodies,
            statuses,
            content_types,
            retry_afters,
            headers,
        )
    ]
    dispatch = _RecordingDispatch(responses)
    clock = _CountingClock() if monotonic_clock is None else monotonic_clock
    adapter = _module().foundation_gateway_adapters(monotonic_clock=clock)[_ADAPTER_ID]
    result = await adapter(group, dispatch)
    return result, dispatch, group


async def _execute(
    responses: list[object],
    *,
    max_pages: int = 2,
    result_limit: int = 2,
    monotonic_clock=None,
    before_gateway_response=None,
):
    registry, plan = _plan_for(max_pages=max_pages, result_limit=result_limit)
    queued = list(responses)
    gateway_calls = []

    async def gateway(route, intent, *, is_policy_active):
        gateway_calls.append((route, intent))
        if before_gateway_response is not None:
            before_gateway_response(len(gateway_calls))
        response = queued.pop(0)
        if isinstance(response, BaseException):
            raise response
        body, status_code, content_type, retry_after = response
        return _response(
            route,
            intent,
            body,
            status_code=status_code,
            content_type=content_type,
            retry_after=retry_after,
        )

    clock = _CountingClock() if monotonic_clock is None else monotonic_clock
    adapter = _module().foundation_gateway_adapters(monotonic_clock=clock)[_ADAPTER_ID]
    dispatch_ids = iter(f"arxiv-dispatch-{index}" for index in range(1, max_pages + 1))
    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={_ADAPTER_ID: adapter},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: next(dispatch_ids),
    )
    return result, gateway_calls


def _assert_typed_error(error: BaseException, code: str) -> None:
    error_type = executor_module.DiscoveryAdapterError
    assert type(error) is error_type
    assert error.code == code
    assert str(error) == code


def _entry(
    arxiv_id: str,
    *,
    title: str = "Bounded arXiv Record",
    summary: str = "A bounded abstract.",
    authors: tuple[str, ...] = ("Ada Researcher",),
    doi: str | None = None,
    id_url: str | None = None,
    pdf_href: str | None = None,
    extra: str = "",
) -> str:
    id_url = id_url or f"https://arxiv.org/abs/{arxiv_id}"
    author_xml = "".join(f"<author><name>{escape(author)}</name></author>" for author in authors)
    doi_xml = "" if doi is None else f"<arxiv:doi>{escape(doi)}</arxiv:doi>"
    pdf_href = pdf_href if pdf_href is not None else f"https://arxiv.org/pdf/{arxiv_id}"
    pdf_xml = (
        ""
        if pdf_href == ""
        else f'<link href={quoteattr(pdf_href)} rel="related" title="pdf" type="application/pdf" />'
    )
    return (
        "<entry>"
        f"<id>{escape(id_url)}</id>"
        f"<title>{escape(title)}</title>"
        f"<summary>{escape(summary)}</summary>"
        f"{author_xml}{doi_xml}{pdf_xml}{extra}"
        "</entry>"
    )


def _feed(
    entries: list[str],
    *,
    total: object | None = None,
    start: object = 0,
    items: object | None = None,
    feed_link: str = "https://export.arxiv.org/api/query?start=0",
    extra: str = "",
) -> bytes:
    total = len(entries) if total is None else total
    items = max(1, len(entries)) if items is None else items
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<feed xmlns="{_ATOM}" xmlns:opensearch="{_OPEN_SEARCH}" xmlns:arxiv="{_ARXIV}">'
        f'<link href={quoteattr(feed_link)} rel="self" type="application/atom+xml" />'
        f"<opensearch:totalResults>{escape(str(total))}</opensearch:totalResults>"
        f"<opensearch:startIndex>{escape(str(start))}</opensearch:startIndex>"
        f"<opensearch:itemsPerPage>{escape(str(items))}</opensearch:itemsPerPage>"
        f"{''.join(entries)}{extra}</feed>"
    ).encode()


def _normalized(candidate) -> dict[str, Any]:
    record = dict(candidate.record)
    record["provider_ids"] = dict(record["provider_ids"])
    return record


def test_arxiv_profile_and_factory_registration_are_exact_and_immutable() -> None:
    module = _module()
    profiles = module._PARSING_PROFILES

    assert type(profiles) is MappingProxyType
    assert ("arxiv_v2", "foundation-v2") in profiles
    profile = profiles[("arxiv_v2", "foundation-v2")]
    assert (
        profile.max_input_bytes,
        profile.max_records,
        profile.max_depth,
        profile.max_nodes,
        profile.max_string_chars,
        profile.max_numeric_token_chars,
        profile.parse_deadline_ms,
    ) == (2_097_152, 100, 16, 50_000, 65_536, 32, 500)
    with pytest.raises(FrozenInstanceError):
        profile.max_records = 101

    adapters = module.foundation_gateway_adapters()
    assert tuple(adapters)[-1] == _ADAPTER_ID
    assert callable(adapters[_ADAPTER_ID])
    assert module._MAX_XML_ATTRIBUTES_PER_ELEMENT == 16
    assert module._MAX_ARXIV_FIELDS_PER_ENTRY == 512


def test_planner_freezes_exact_gateway_request_without_hidden_filter_or_sort_semantics() -> None:
    registry, plan = _plan_for(result_limit=7)
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    intent = group.intents[0]

    assert group.adapter_id == _ADAPTER_ID
    assert group.adapter_version == "foundation-v2"
    assert route.policy.origin.host == "export.arxiv.org"
    assert intent.method == "GET"
    assert intent.path == "/api/query"
    assert tuple((pair.name, pair.value) for pair in intent.query_pairs) == (
        ("search_query", "all:bounded discovery"),
        ("start", "0"),
        ("max_results", "7"),
    )
    assert intent.json_body_pairs == ()


@pytest.mark.asyncio
async def test_sanitized_success_fixture_normalizes_exact_twelve_field_record() -> None:
    result, dispatch, _group = await _invoke([_fixture("success")])

    assert type(result) is DiscoveryAdapterResult
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    record = _normalized(candidate)
    expected = {
        "title": "Shared Discovery Record",
        "authors": ("Ada Researcher", "Grace Scientist"),
        "abstract": "A sanitized abstract for arXiv discovery adapter testing.",
        "snippet": "A sanitized abstract for arXiv discovery adapter testing.",
        "doi": "10.5555/shared.discovery.2026",
        "pmid": None,
        "pmcid": None,
        "arxiv_id": "2601.01234v2",
        "url": "https://arxiv.org/abs/2601.01234v2",
        "pdf_url": "https://arxiv.org/pdf/2601.01234v2",
        "provider": "arxiv",
        "provider_ids": {
            "arxiv_id": "2601.01234v2",
            "doi": "10.5555/shared.discovery.2026",
        },
    }
    assert record == expected
    assert set(record) == _NORMALIZED_KEYS
    assert "published_date" not in record
    assert "published_at" not in record
    assert candidate.candidate_id == DiscoveryOutcomeIdentity.from_fingerprint(build_fingerprint(expected)).document_id
    assert len(dispatch.calls) == 1
    assert dispatch.calls[0][1] is None


@pytest.mark.asyncio
async def test_sanitized_empty_fixture_returns_exact_empty_result() -> None:
    result, dispatch, _group = await _invoke([_fixture("empty")])

    assert result == DiscoveryAdapterResult(candidates=())
    assert len(dispatch.calls) == 1


@pytest.mark.asyncio
async def test_empty_page_preserves_requested_page_capacity() -> None:
    body = _feed([], total=0, start=0, items=1)

    result, _dispatch, _group = await _invoke([body], result_limit=1)

    assert result == DiscoveryAdapterResult(candidates=())


@pytest.mark.asyncio
async def test_terminal_empty_page_can_report_zero_items() -> None:
    body = _feed([], total=0, start=0, items=0)

    result, _dispatch, _group = await _invoke([body], result_limit=1)

    assert result == DiscoveryAdapterResult(candidates=())


@pytest.mark.asyncio
async def test_short_final_page_can_report_requested_page_capacity() -> None:
    body = _feed(
        [_entry("2601.00001")],
        total=1,
        start=0,
        items=2,
    )

    result, _dispatch, _group = await _invoke([body], result_limit=2)

    assert len(result.candidates) == 1


@pytest.mark.parametrize(
    "content_type",
    (
        "application/atom+xml",
        "Application/Atom+XML; Charset=UTF-8",
        'application/atom+xml; charset="utf-8"',
    ),
)
@pytest.mark.asyncio
async def test_exact_atom_mime_with_valid_parameters_is_accepted(content_type: str) -> None:
    result, _dispatch, _group = await _invoke(
        [_fixture("empty")],
        content_types=[content_type],
    )

    assert result == DiscoveryAdapterResult(candidates=())


@pytest.mark.parametrize(
    "content_type",
    (
        None,
        "application/json",
        "application/xml",
        "text/xml",
        "text/html",
        "application/atom+xml, application/xml",
        "application/atom+xml; charset",
        "application/atom+xml; charset=",
        'application/atom+xml; charset="utf-8',
        "application/*+xml",
    ),
)
@pytest.mark.asyncio
async def test_missing_generic_or_malformed_mime_is_rejected(content_type: str | None) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke([_fixture("empty")], content_types=[content_type])

    _assert_typed_error(caught.value, "provider_response_rejected")


@pytest.mark.asyncio
async def test_duplicate_content_type_headers_are_rejected() -> None:
    with pytest.raises(Exception) as caught:
        await _invoke(
            [_fixture("empty")],
            headers=[
                (
                    ("content-type", "application/atom+xml"),
                    ("Content-Type", "application/atom+xml"),
                )
            ],
        )

    _assert_typed_error(caught.value, "provider_response_rejected")


@pytest.mark.parametrize("status_code", (201, 204, 400, 503, True, 200.0, "200", 429.0))
@pytest.mark.asyncio
async def test_non_200_status_rejects_without_parsing_body_or_clock(status_code: object) -> None:
    def forbidden_clock() -> float:
        raise AssertionError("parse clock must not run")

    with pytest.raises(Exception) as caught:
        await _invoke(
            [b"<!ENTITY fixture-secret>"],
            statuses=[status_code],
            content_types=[None],
            retry_afters=["120"],
            monotonic_clock=forbidden_clock,
        )

    _assert_typed_error(caught.value, "provider_response_rejected")
    assert caught.value.retry_after is None
    assert "fixture-secret" not in repr(caught.value)


@pytest.mark.asyncio
async def test_429_preserves_only_validated_retry_after_and_never_parses_body() -> None:
    def forbidden_clock() -> float:
        raise AssertionError("parse clock must not run")

    with pytest.raises(Exception) as caught:
        await _invoke(
            [b"fixture-secret"],
            statuses=[429],
            content_types=[None],
            retry_afters=["120"],
            monotonic_clock=forbidden_clock,
        )

    _assert_typed_error(caught.value, "provider_rate_limited")
    assert caught.value.retry_after == "120"
    assert "fixture-secret" not in repr(caught.value)


@pytest.mark.parametrize(
    "body",
    (
        b"",
        b"<feed",
        b"<feed></feed><feed></feed>",
        b"not xml",
        b"\xff\xfe<\x00f\x00e\x00e\x00d\x00/\x00>\x00",
        b'<?xml version="1.0" encoding="iso-8859-1"?><feed />',
    ),
)
@pytest.mark.asyncio
async def test_malformed_non_utf8_or_mismatched_encoding_payload_is_rejected(body: bytes) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("encoding", ("utf-16le", "utf-16be", "utf-32le", "utf-32be"))
@pytest.mark.asyncio
async def test_bomless_non_utf8_atom_payload_is_rejected(encoding: str) -> None:
    text = _feed([_entry("2601.00001")]).decode("utf-8")
    body = text[text.index("<feed") :].encode(encoding)

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize(
    "body",
    (
        (
            b'<?xml version="1.0" encoding="utf-8"?>'
            b'<!DOCTYPE feed [<!ENTITY boom "fixture-secret">]>'
            b'<feed xmlns="http://www.w3.org/2005/Atom">&boom;</feed>'
        ),
        (
            b'<?xml version="1.0" encoding="utf-8"?>'
            b'<!DOCTYPE feed [<!ENTITY ext SYSTEM "file:///private/fixture-secret">]>'
            b'<feed xmlns="http://www.w3.org/2005/Atom">&ext;</feed>'
        ),
    ),
)
@pytest.mark.asyncio
async def test_dtd_internal_or_external_entity_input_is_rejected_safely(body: bytes) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")
    assert "fixture-secret" not in repr(caught.value)


@pytest.mark.asyncio
async def test_builtin_xml_entities_remain_valid() -> None:
    body = _feed(
        [
            _entry(
                "2601.00001v1",
                title="Research & Development",
                summary="Safe <bounded> content & results.",
            )
        ]
    )

    result, _dispatch, _group = await _invoke([body])

    record = _normalized(result.candidates[0])
    assert record["title"] == "Research & Development"
    assert record["abstract"] == "Safe <bounded> content & results."


@pytest.mark.parametrize(
    "body",
    (
        b'<feed xmlns="urn:not-atom" />',
        b'<feed xmlns="http://www.w3.org/2005/Atom" />',
        (
            b'<feed xmlns="http://www.w3.org/2005/Atom" '
            b'xmlns:opensearch="urn:not-opensearch">'
            b"<opensearch:totalResults>0</opensearch:totalResults>"
            b"<opensearch:startIndex>0</opensearch:startIndex>"
            b"<opensearch:itemsPerPage>0</opensearch:itemsPerPage>"
            b"</feed>"
        ),
    ),
)
@pytest.mark.asyncio
async def test_wrong_root_namespace_or_missing_exact_envelope_is_rejected(body: bytes) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("field", ("totalResults", "startIndex", "itemsPerPage"))
@pytest.mark.parametrize("duplicate", (False, True))
@pytest.mark.asyncio
async def test_each_pagination_field_is_required_exactly_once(field: str, duplicate: bool) -> None:
    body = _feed([])
    value = 1 if field == "itemsPerPage" else 0
    needle = f"<opensearch:{field}>{value}</opensearch:{field}>".encode("ascii")
    assert needle in body
    body = body.replace(needle, needle + needle if duplicate else b"", 1)

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("value", ("-1", "+1", "1.0", "1e2", "１２", "true", ""))
@pytest.mark.asyncio
async def test_pagination_fields_require_ascii_nonnegative_decimal(value: str) -> None:
    body = _feed([], total=value)

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_excessive_numeric_token_is_a_parse_limit_failure() -> None:
    body = _feed([], total="1" * 33)

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_cursor_above_global_bound_is_rejected() -> None:
    body = _feed([], total=MAX_PAGINATION_CURSOR + 1)

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize(
    ("total", "start", "items", "entry_count"),
    (
        (1, 0, 0, 1),
        (0, 0, 1, 1),
        (2, 0, 0, 0),
    ),
)
@pytest.mark.asyncio
async def test_feed_pagination_envelope_must_match_cursor_and_raw_records(
    total: int,
    start: int,
    items: int,
    entry_count: int,
) -> None:
    entries = [_entry(f"2601.{index + 1:05d}") for index in range(entry_count)]
    body = _feed(entries, total=total, start=start, items=items)

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_page_cardinality_cannot_exceed_requested_max_results() -> None:
    body = _feed(
        [_entry("2601.00001"), _entry("2601.00002")],
        total=2,
        start=0,
        items=2,
    )

    with pytest.raises(Exception) as caught:
        await _invoke([body], result_limit=1)

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("field", ("id", "title"))
@pytest.mark.parametrize("duplicate", (False, True))
@pytest.mark.asyncio
async def test_consumed_entry_scalar_fields_are_required_exactly_once(
    field: str,
    duplicate: bool,
) -> None:
    body = _feed([_entry("2601.00001")])
    open_tag = f"<{field}>".encode("ascii")
    close_tag = f"</{field}>".encode("ascii")
    start = body.index(open_tag)
    end = body.index(close_tag, start) + len(close_tag)
    original = body[start:end]
    body = body[:start] + (original + original if duplicate else b"") + body[end:]

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("field", ("id", "title"))
@pytest.mark.asyncio
async def test_required_entry_scalars_reject_blank_text(field: str) -> None:
    body = _feed([_entry("2601.00001")])
    open_tag = f"<{field}>".encode("ascii")
    close_tag = f"</{field}>".encode("ascii")
    start = body.index(open_tag) + len(open_tag)
    end = body.index(close_tag, start)
    body = body[:start] + b" \n\t " + body[end:]

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("blank", (False, True))
@pytest.mark.asyncio
async def test_optional_summary_may_be_missing_or_blank(blank: bool) -> None:
    body = _feed([_entry("2601.00001")])
    open_tag = b"<summary>"
    close_tag = b"</summary>"
    start = body.index(open_tag)
    end = body.index(close_tag, start) + len(close_tag)
    replacement = b"<summary> \n </summary>" if blank else b""
    body = body[:start] + replacement + body[end:]

    result, _dispatch, _group = await _invoke([body])

    record = _normalized(result.candidates[0])
    assert record["abstract"] is None
    assert record["snippet"] is None


@pytest.mark.asyncio
async def test_optional_summary_rejects_duplicates() -> None:
    body = _feed([_entry("2601.00001", extra="<summary>duplicate</summary>")])

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_entry_without_authors_normalizes_to_empty_tuple() -> None:
    result, _dispatch, _group = await _invoke([_feed([_entry("2601.00001", authors=())])])

    assert result.candidates[0].record["authors"] == ()


@pytest.mark.asyncio
async def test_optional_doi_is_allowed_once_and_rejects_duplicates() -> None:
    body = _feed(
        [
            _entry(
                "2601.00001",
                doi="10.5555/example",
                extra="<arxiv:doi>10.5555/other</arxiv:doi>",
            )
        ]
    )

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize(
    "author_xml",
    (
        "<author />",
        "<author><name /></author>",
        "<author><name> </name></author>",
        "<author><name>Ada</name><name>Grace</name></author>",
        "<author><name><b>Ada</b></name></author>",
    ),
)
@pytest.mark.asyncio
async def test_malformed_author_name_structure_is_rejected(author_xml: str) -> None:
    entry = _entry("2601.00001", authors=()).replace(
        '<link href="https://arxiv.org/pdf/2601.00001"',
        author_xml + '<link href="https://arxiv.org/pdf/2601.00001"',
        1,
    )

    with pytest.raises(Exception) as caught:
        await _invoke([_feed([entry])])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("field", ("title", "summary"))
@pytest.mark.asyncio
async def test_nested_markup_in_scalar_field_is_rejected(field: str) -> None:
    entry = _entry("2601.00001")
    entry = entry.replace(
        f"<{field}>",
        f"<{field}><b>",
        1,
    ).replace(f"</{field}>", f"</b></{field}>", 1)

    with pytest.raises(Exception) as caught:
        await _invoke([_feed([entry])])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize(
    "arxiv_id",
    (
        "2601.01234",
        "2601.01234v2",
        "0704.0001v12",
        "hep-th/9901001v3",
        "math.GT/0309136v1",
        "cond-mat/0207270",
    ),
)
@pytest.mark.asyncio
async def test_modern_and_legacy_versioned_arxiv_ids_are_preserved(arxiv_id: str) -> None:
    result, _dispatch, _group = await _invoke([_feed([_entry(arxiv_id)])])

    record = _normalized(result.candidates[0])
    assert record["arxiv_id"] == arxiv_id
    assert record["provider_ids"]["arxiv_id"] == arxiv_id
    assert record["url"] == f"https://arxiv.org/abs/{arxiv_id}"
    assert record["pdf_url"] == f"https://arxiv.org/pdf/{arxiv_id}"


@pytest.mark.parametrize(
    "id_url",
    (
        "https://attacker.example/abs/2601.00001",
        "https://arxiv.org/pdf/2601.00001",
        "https://arxiv.org/abs/%32%36%30%31.00001",
        "https://arxiv.org/abs/2601.00001?token=fixture-secret",
        "https://arxiv.org/abs/2601.00001#fixture-secret",
        "https://user:pass@arxiv.org/abs/2601.00001",
        "https://arxiv.org:444/abs/2601.00001",
        "https://arxiv.org/abs/../api/query",
        "https://arxiv.org/abs/2601.1",
        "https://arxiv.org/abs/２６０１.０１２３４",
        "https://arxiv.org/abs/K/1234567",
        "https://arxiv.org/abs/İ/1234567",
        "https://arxiv.org/abs/ſ/1234567",
        "arxiv:2601.00001",
    ),
)
@pytest.mark.asyncio
async def test_entry_id_url_rejects_foreign_encoded_or_malformed_forms(id_url: str) -> None:
    body = _feed([_entry("2601.00001", id_url=id_url)])

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_payload_invalid")
    assert "fixture-secret" not in repr(caught.value)


@pytest.mark.asyncio
async def test_pdf_is_derived_only_from_first_exact_matching_safe_declaration() -> None:
    unsafe_first = (
        '<link href="https://attacker.example/private.pdf?token=fixture-secret" '
        'rel="related" title="pdf" type="application/pdf" />'
        '<link href="https://arxiv.org/pdf/2601.00001v2" '
        'rel="related" title="pdf" type="application/pdf" />'
    )
    entry = _entry(
        "2601.00001v2",
        pdf_href="",
        extra=unsafe_first,
    )

    result, _dispatch, _group = await _invoke([_feed([entry])])

    record = _normalized(result.candidates[0])
    assert record["url"] == "https://arxiv.org/abs/2601.00001v2"
    assert record["pdf_url"] is None
    assert "fixture-secret" not in repr(record)


@pytest.mark.asyncio
async def test_absent_pdf_declaration_leaves_pdf_url_none() -> None:
    result, _dispatch, _group = await _invoke([_feed([_entry("2601.00001", pdf_href="")])])

    assert result.candidates[0].record["pdf_url"] is None


@pytest.mark.asyncio
async def test_pdf_version_must_match_versioned_entry_id() -> None:
    entry = _entry(
        "2601.00001v2",
        pdf_href="https://arxiv.org/pdf/2601.00001v1",
    )

    result, _dispatch, _group = await _invoke([_feed([entry])])

    record = _normalized(result.candidates[0])
    assert record["arxiv_id"] == "2601.00001v2"
    assert record["pdf_url"] is None


@pytest.mark.asyncio
async def test_unversioned_entry_can_use_matching_versioned_pdf() -> None:
    entry = _entry(
        "hep-ex/0307015",
        pdf_href="https://arxiv.org/pdf/hep-ex/0307015v1",
    )

    result, _dispatch, _group = await _invoke([_feed([entry])])

    record = _normalized(result.candidates[0])
    assert record["arxiv_id"] == "hep-ex/0307015"
    assert record["pdf_url"] == "https://arxiv.org/pdf/hep-ex/0307015v1"


@pytest.mark.asyncio
async def test_unknown_fields_are_counted_but_not_retained() -> None:
    entry = _entry(
        "2601.00001",
        extra='<unknown token="fixture-secret"><nested>ignored</nested></unknown>',
    )
    body = _feed([entry], extra="<unknown-feed>fixture-secret</unknown-feed>")

    result, _dispatch, _group = await _invoke([body])

    record = _normalized(result.candidates[0])
    assert set(record) == _NORMALIZED_KEYS
    assert "fixture-secret" not in repr(record)


@pytest.mark.asyncio
async def test_body_type_and_route_byte_ceiling_fail_before_xml_parse() -> None:
    def forbidden_clock() -> float:
        raise AssertionError("parse clock must not run")

    with pytest.raises(Exception) as wrong_type:
        await _invoke([bytearray(_fixture("empty"))], monotonic_clock=forbidden_clock)
    _assert_typed_error(wrong_type.value, "provider_payload_invalid")

    body = _fixture("empty")
    with pytest.raises(Exception) as too_large:
        await _invoke(
            [body],
            max_response_bytes=len(body) - 1,
            monotonic_clock=forbidden_clock,
        )
    _assert_typed_error(too_large.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_xml_depth_ceiling_is_enforced_during_parse() -> None:
    nested = "<x>" * 16 + "bounded" + "</x>" * 16
    body = _feed([], extra=nested)

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_xml_structural_node_ceiling_is_enforced() -> None:
    body = _feed([], extra="<x />" * 50_000)

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_xml_attribute_ceiling_is_enforced() -> None:
    attributes = " ".join(f'a{index}="x"' for index in range(17))
    body = _feed([], extra=f"<x {attributes} />")

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_xml_namespace_declarations_count_toward_attribute_ceiling() -> None:
    declarations = "".join(f' xmlns:n{index}="urn:n{index}"' for index in range(17))
    body = _feed([_entry("2601.00001")]).replace(
        f'xmlns:arxiv="{_ARXIV}">'.encode(),
        f'xmlns:arxiv="{_ARXIV}"{declarations}>'.encode(),
        1,
    )

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_cumulative_expanded_xml_name_material_is_bounded() -> None:
    namespace_uri = "urn:" + ("x" * 65_000)
    children = "".join(f"<long:ignored{index} />" for index in range(40))
    body = _feed([], extra=children).replace(
        f'xmlns:arxiv="{_ARXIV}">'.encode(),
        f'xmlns:arxiv="{_ARXIV}" xmlns:long={quoteattr(namespace_uri)}>'.encode(),
        1,
    )
    assert len(body) < 100_000

    with pytest.raises(Exception) as caught:
        await _invoke([body])

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_entry_field_ceiling_is_enforced() -> None:
    entry = _entry("2601.00001", extra="<unknown />" * 513)

    with pytest.raises(Exception) as caught:
        await _invoke([_feed([entry])])

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.parametrize(("author_count", "should_pass"), ((252, True), (253, False)))
@pytest.mark.asyncio
async def test_entry_field_ceiling_bounds_large_author_lists(
    author_count: int,
    should_pass: bool,
) -> None:
    authors = tuple(f"Researcher {index}" for index in range(author_count))
    invocation = _invoke([_feed([_entry("2601.00001", authors=authors)])])

    if should_pass:
        result, _dispatch, _group = await invocation
        assert len(result.candidates[0].record["authors"]) == 252
    else:
        with pytest.raises(Exception) as caught:
            await invocation
        _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_entry_root_attributes_count_toward_field_ceiling() -> None:
    authors = tuple(f"Researcher {index}" for index in range(252))
    entry = _entry("2601.00001", authors=authors).replace(
        "<entry>",
        '<entry rogue="x">',
        1,
    )

    with pytest.raises(Exception) as caught:
        await _invoke([_feed([entry])])

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_text_field_ceiling_is_enforced() -> None:
    entry = _entry("2601.00001", title="x" * 65_537)

    with pytest.raises(Exception) as caught:
        await _invoke([_feed([entry])])

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_record_ceiling_is_enforced_before_normalization() -> None:
    entries = [_entry(f"2601.{index:05d}") for index in range(1, 102)]

    with pytest.raises(Exception) as caught:
        await _invoke(
            [_feed(entries, total=101, items=101)],
            result_limit=100,
        )

    _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_aggregate_record_ceiling_across_pages_fails_atomically() -> None:
    first_entries = [_entry(f"2601.{index:05d}") for index in range(1, 61)]
    second_entries = [_entry(f"2602.{index:05d}") for index in range(1, 61)]
    first = _feed(first_entries, total=120, start=0, items=60)
    second = _feed(second_entries, total=120, start=60, items=60)

    result, gateway_calls = await _execute(
        [
            (first, 200, "application/atom+xml", None),
            (second, 200, "application/atom+xml", None),
        ],
        result_limit=100,
    )

    assert len(gateway_calls) == 2
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "provider_parse_limit_exceeded"


@pytest.mark.asyncio
async def test_parse_deadline_is_cooperatively_enforced() -> None:
    clock = _CountingClock(step=0.2)

    with pytest.raises(Exception) as caught:
        await _invoke([_fixture("success")], monotonic_clock=clock)

    _assert_typed_error(caught.value, "provider_parse_deadline_exceeded")
    assert clock.calls >= 4


@pytest.mark.asyncio
async def test_pagination_uses_only_numeric_cursor_and_ignores_absolute_feed_links() -> None:
    first = _feed(
        [_entry("2601.00001")],
        total=2,
        start=0,
        items=1,
        feed_link="https://attacker.example/private?token=fixture-secret",
        extra='<link href="https://attacker.example/next?start=999" rel="next" />',
    )
    second = _feed(
        [_entry("2601.00002")],
        total=2,
        start=1,
        items=1,
        feed_link="https://attacker.example/private?token=fixture-secret",
    )

    result, dispatch, _group = await _invoke(
        [first, second],
        max_pages=2,
        result_limit=2,
    )

    assert len(result.candidates) == 2
    assert [call[1] for call in dispatch.calls] == [None, NumericCursor(1)]
    assert "fixture-secret" not in repr(result)


@pytest.mark.asyncio
async def test_raw_record_ceiling_stops_before_an_extra_page_dispatch() -> None:
    entries = [_entry(f"2601.{index + 1:05d}") for index in range(100)]
    first = _feed(
        entries,
        total=101,
        start=0,
        items=100,
    )

    result, dispatch, _group = await _invoke(
        [first],
        max_pages=2,
        result_limit=100,
    )

    assert len(result.candidates) == 100
    assert len(dispatch.calls) == 1


@pytest.mark.asyncio
async def test_executor_reconstructs_second_request_start_and_accounts_both_hops() -> None:
    first = _feed([_entry("2601.00001")], total=2, start=0, items=1)
    second = _feed([_entry("2601.00002")], total=2, start=1, items=1)

    result, gateway_calls = await _execute(
        [
            (first, 200, "application/atom+xml", None),
            (second, 200, "application/atom+xml", None),
        ]
    )

    starts = [
        next(pair.value for pair in intent.query_pairs if pair.name == "start") for _route, intent in gateway_calls
    ]
    assert starts == ["0", "1"]
    assert len(result.candidates) == 2
    assert result.logical_outcomes[0].state is LogicalOutcomeState.SUCCEEDED
    assert tuple(record.state for record in result.usage.physical_records) == (
        PhysicalDispatchState.SUCCEEDED,
        PhysicalDispatchState.SUCCEEDED,
    )
    assert result.usage.pages == 2
    assert result.usage.accounting.created == 2
    assert result.usage.accounting.debited == 2


@pytest.mark.asyncio
async def test_later_page_cursor_mismatch_fails_atomically() -> None:
    first = _feed([_entry("2601.00001")], total=2, start=0, items=1)
    mismatched = _feed([_entry("2601.00002")], total=2, start=0, items=1)

    result, gateway_calls = await _execute(
        [
            (first, 200, "application/atom+xml", None),
            (mismatched, 200, "application/atom+xml", None),
        ]
    )

    assert len(gateway_calls) == 2
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "provider_payload_invalid"


@pytest.mark.asyncio
async def test_identical_cross_page_record_deduplicates() -> None:
    entry = _entry("2601.00001")
    first = _feed([entry], total=2, start=0, items=1)
    second = _feed([entry], total=2, start=1, items=1)

    result, _dispatch, _group = await _invoke(
        [first, second],
        max_pages=2,
        result_limit=2,
    )

    assert len(result.candidates) == 1


@pytest.mark.asyncio
async def test_conflicting_same_identity_across_pages_fails_atomically() -> None:
    first = _feed(
        [_entry("2601.00001v1", title="First version")],
        total=2,
        start=0,
        items=1,
    )
    second = _feed(
        [_entry("2601.00001v2", title="Second version")],
        total=2,
        start=1,
        items=1,
    )

    result, _gateway_calls = await _execute(
        [
            (first, 200, "application/atom+xml", None),
            (second, 200, "application/atom+xml", None),
        ]
    )

    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "provider_payload_invalid"


@pytest.mark.asyncio
async def test_official_atom_error_entry_is_sanitized_as_invalid_provider_payload() -> None:
    error_entry = _entry(
        "2601.00001",
        title="Error",
        summary="fixture-secret at /private/arxiv.key",
        id_url="https://arxiv.org/api/errors#incorrect-query",
    )

    with pytest.raises(Exception) as caught:
        await _invoke([_feed([error_entry])])

    _assert_typed_error(caught.value, "provider_payload_invalid")
    assert "fixture-secret" not in repr(caught.value)
    assert "/private/arxiv.key" not in repr(caught.value)


@pytest.mark.asyncio
async def test_later_page_malformed_xml_commits_zero_candidates() -> None:
    first = _feed([_entry("2601.00001")], total=2, start=0, items=1)

    result, gateway_calls = await _execute(
        [
            (first, 200, "application/atom+xml", None),
            (b"<feed", 200, "application/atom+xml", None),
        ]
    )

    assert len(gateway_calls) == 2
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "provider_payload_invalid"


@pytest.mark.asyncio
async def test_later_page_parse_deadline_commits_zero_candidates() -> None:
    first = _feed([_entry("2601.00001")], total=2, start=0, items=1)
    second = _feed([_entry("2601.00002")], total=2, start=1, items=1)
    clock = _CountingClock()

    def expire_before_second_response(call_count: int) -> None:
        if call_count == 2:
            clock.step = 1.0

    result, gateway_calls = await _execute(
        [
            (first, 200, "application/atom+xml", None),
            (second, 200, "application/atom+xml", None),
        ],
        monotonic_clock=clock,
        before_gateway_response=expire_before_second_response,
    )

    assert len(gateway_calls) == 2
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.TIMED_OUT
    assert result.logical_outcomes[0].code == "provider_parse_deadline_exceeded"


@pytest.mark.asyncio
async def test_later_page_rate_limit_commits_zero_and_preserves_safe_retry_after() -> None:
    first = _feed([_entry("2601.00001")], total=2, start=0, items=1)

    result, gateway_calls = await _execute(
        [
            (first, 200, "application/atom+xml", None),
            (b"fixture-secret", 429, "text/plain", "120"),
        ]
    )

    assert len(gateway_calls) == 2
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "provider_rate_limited"
    assert result.logical_outcomes[0].retry_after == "120"
    assert "fixture-secret" not in repr(result)


@pytest.mark.asyncio
async def test_dispatch_cancellation_propagates_without_partial_result() -> None:
    registry, plan = _plan_for()
    group = plan.dispatch_groups[0]
    cancelled = asyncio.CancelledError("fixture-cancelled")
    dispatch = _RecordingDispatch([cancelled])
    adapter = _module().foundation_gateway_adapters()[_ADAPTER_ID]

    with pytest.raises(asyncio.CancelledError) as caught:
        await adapter(group, dispatch)

    assert caught.value is cancelled


@pytest.mark.asyncio
async def test_later_page_cancellation_propagates_without_partial_result() -> None:
    first = _feed([_entry("2601.00001")], total=2, start=0, items=1)
    cancelled = asyncio.CancelledError("fixture-page-two-cancelled")

    with pytest.raises(asyncio.CancelledError) as caught:
        await _execute(
            [
                (first, 200, "application/atom+xml", None),
                cancelled,
            ]
        )

    assert caught.value is cancelled


@pytest.mark.asyncio
async def test_existing_gateway_timeout_classification_propagates_unchanged() -> None:
    registry, plan = _plan_for()
    group = plan.dispatch_groups[0]
    timed_out = executor_module.DiscoveryExecutionError("aggregate_deadline_exceeded")
    dispatch = _RecordingDispatch([timed_out])
    adapter = _module().foundation_gateway_adapters()[_ADAPTER_ID]

    with pytest.raises(executor_module.DiscoveryExecutionError) as caught:
        await adapter(group, dispatch)

    assert caught.value is timed_out


@pytest.mark.asyncio
async def test_unknown_adapter_version_rejects_before_dispatch() -> None:
    _registry, plan = _plan_for()
    group = replace(plan.dispatch_groups[0], adapter_version="unknown-v2")
    dispatch = _RecordingDispatch([])
    adapter = _module().foundation_gateway_adapters()[_ADAPTER_ID]

    with pytest.raises(Exception) as caught:
        await adapter(group, dispatch)

    _assert_typed_error(caught.value, "provider_payload_invalid")
    assert dispatch.calls == []


@pytest.mark.asyncio
async def test_runtime_egress_legacy_and_result_link_tripwires_receive_zero_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("arXiv adapter attempted direct egress or legacy execution")

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Security import http_hop
    from tldw_Server_API.app.core.Third_Party import Arxiv

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(http.client, "HTTPConnection", forbidden)
    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(http_client, "fetch", forbidden)
    monkeypatch.setattr(http_client, "fetch_json", forbidden)
    monkeypatch.setattr(http_hop, "request_http_hop", forbidden)
    monkeypatch.setattr(Arxiv, "search_arxiv_custom_api", forbidden)
    monkeypatch.setattr(Arxiv, "fetch_arxiv_xml", forbidden)
    monkeypatch.setattr(Arxiv, "parse_arxiv_feed", forbidden)

    result, dispatch, _group = await _invoke([_fixture("success")])

    assert len(result.candidates) == 1
    assert result.candidates[0].record["url"] == "https://arxiv.org/abs/2601.01234v2"
    assert result.candidates[0].record["pdf_url"] == "https://arxiv.org/pdf/2601.01234v2"
    assert len(dispatch.calls) == 1
