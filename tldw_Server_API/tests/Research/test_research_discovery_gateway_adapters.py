"""Offline contract tests for the five gateway-only discovery adapters."""

from __future__ import annotations

import ast
import asyncio
import http.client
import importlib
import json
import socket
import urllib.request
import xml.etree.ElementTree as ElementTree
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

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
    DiscoveryExecutionError,
    LogicalOutcomeState,
    NumericCursor,
    PhysicalDispatchState,
    execute_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.gateway import (
    DiscoveryGatewayResponse,
    DiscoveryGatewayTrace,
)
from tldw_Server_API.app.core.Research.discovery.identity import (
    build_fingerprint,
    canonicalize_url,
    has_unsafe_url_material,
)
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

_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "research_discovery_gateway_adapters"
_ADAPTER_MODULE = "tldw_Server_API.app.core.Research.discovery.gateway_adapters"
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
_PROVIDERS = (
    ("semantic_scholar", "semantic_scholar_v2", "semantic_scholar_id"),
    ("crossref", "crossref_v2", "crossref_id"),
    ("zenodo", "zenodo_v2", "zenodo_id"),
    ("figshare", "figshare_v2", "figshare_id"),
    ("osf", "osf_v2", "osf_id"),
)
_PROVIDER_NAMES = tuple(case[0] for case in _PROVIDERS)
_ADAPTER_IDS = tuple(case[1] for case in _PROVIDERS)
_EXPECTED_PROVIDER_ID_KEYS = {
    "semantic_scholar": {
        "semantic_scholar_id",
        "doi",
        "pmid",
        "pmcid",
        "arxiv_id",
    },
    "crossref": {"crossref_id", "doi"},
    "zenodo": {"zenodo_id", "doi"},
    "figshare": {"figshare_id", "doi"},
    "osf": {"osf_id", "doi"},
}
_EXPECTED_SUCCESS_RECORD_KEYS = {
    "semantic_scholar": {
        "paperId",
        "title",
        "authors",
        "abstract",
        "tldr",
        "externalIds",
        "url",
        "openAccessPdf",
    },
    "crossref": {"DOI", "title", "author", "abstract", "URL", "link"},
    "zenodo": {"id", "doi", "links", "files", "metadata"},
    "figshare": {
        "id",
        "title",
        "doi",
        "handle",
        "url",
        "url_public_api",
        "url_public_html",
        "url_private_api",
        "url_private_html",
        "thumb",
        "defined_type",
        "defined_type_name",
        "resource_doi",
        "resource_title",
        "created_date",
        "modified_date",
        "published_date",
        "group_id",
        "timeline",
        "project_id",
    },
    "osf": {"type", "id", "attributes", "links"},
}
_EXPECTED_NORMALIZED_RECORDS = {
    "semantic_scholar": {
        "title": "Shared Discovery Record",
        "authors": ("Ada Researcher",),
        "abstract": "A sanitized abstract for discovery adapter testing.",
        "snippet": "A sanitized summary for discovery adapter testing.",
        "doi": "10.5555/shared.discovery.2026",
        "pmid": "12345678",
        "pmcid": "PMC1234567",
        "arxiv_id": "2601.01234",
        "url": "https://www.semanticscholar.org/paper/S2-PAPER-001",
        "pdf_url": "https://example.org/research/shared-discovery.pdf",
        "provider": "semantic_scholar",
        "provider_ids": {
            "semantic_scholar_id": "S2-PAPER-001",
            "doi": "10.5555/shared.discovery.2026",
            "pmid": "12345678",
            "pmcid": "PMC1234567",
            "arxiv_id": "2601.01234",
        },
    },
    "crossref": {
        "title": "Shared Discovery Record",
        "authors": ("Ada Researcher",),
        "abstract": "<jats:p>A sanitized abstract for discovery adapter testing.</jats:p>",
        "snippet": "<jats:p>A sanitized abstract for discovery adapter testing.</jats:p>",
        "doi": "10.5555/shared.discovery.2026",
        "pmid": None,
        "pmcid": None,
        "arxiv_id": None,
        "url": "https://doi.org/10.5555/shared.discovery.2026",
        "pdf_url": "https://example.org/research/shared-discovery.pdf",
        "provider": "crossref",
        "provider_ids": {
            "crossref_id": "10.5555/shared.discovery.2026",
            "doi": "10.5555/shared.discovery.2026",
        },
    },
    "zenodo": {
        "title": "Shared Discovery Record",
        "authors": ("Researcher, Ada",),
        "abstract": "A sanitized description for discovery adapter testing.",
        "snippet": "A sanitized description for discovery adapter testing.",
        "doi": "10.5555/shared.discovery.2026",
        "pmid": None,
        "pmcid": None,
        "arxiv_id": None,
        "url": "https://zenodo.org/records/1001",
        "pdf_url": "https://zenodo.org/api/records/1001/files/shared-discovery.pdf/content",
        "provider": "zenodo",
        "provider_ids": {
            "zenodo_id": "1001",
            "doi": "10.5555/shared.discovery.2026",
        },
    },
    "figshare": {
        "title": "Shared Discovery Record",
        "authors": (),
        "abstract": None,
        "snippet": None,
        "doi": "10.5555/shared.discovery.2026",
        "pmid": None,
        "pmcid": None,
        "arxiv_id": None,
        "url": "https://figshare.com/articles/dataset/Shared_Discovery_Record/2001",
        "pdf_url": None,
        "provider": "figshare",
        "provider_ids": {
            "figshare_id": "2001",
            "doi": "10.5555/shared.discovery.2026",
        },
    },
    "osf": {
        "title": "Shared Discovery Record",
        "authors": (),
        "abstract": "A sanitized description for discovery adapter testing.",
        "snippet": "A sanitized description for discovery adapter testing.",
        "doi": "10.5555/shared.discovery.2026",
        "pmid": None,
        "pmcid": None,
        "arxiv_id": None,
        "url": "https://osf.io/preprints/osf/osf001/",
        "pdf_url": None,
        "provider": "osf",
        "provider_ids": {
            "osf_id": "osf001",
            "doi": "10.5555/shared.discovery.2026",
        },
    },
}
_PROVIDER_URL_FIELDS = (
    ("semantic_scholar", "url"),
    ("semantic_scholar", "pdf_url"),
    ("crossref", "url"),
    ("crossref", "pdf_url"),
    ("zenodo", "url"),
    ("zenodo", "pdf_url"),
    ("figshare", "url"),
    ("osf", "url"),
)


def _gateway_adapters_module():
    return importlib.import_module(_ADAPTER_MODULE)


def _adapter_error_type():
    return executor_module.DiscoveryAdapterError


def _fixture_bytes(provider: str, kind: str) -> bytes:
    return (_FIXTURE_ROOT / f"{provider}_{kind}.json").read_bytes()


def _fixture_payload(provider: str, kind: str = "success") -> Any:
    return json.loads(_fixture_bytes(provider, kind))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _registry_with_pages(
    provider: str,
    max_pages: int,
    *,
    max_response_bytes: int | None = None,
) -> DiscoveryRegistry:
    base = foundation_registry()
    route_id = base.get_source(provider).route_references[0].route_id
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
        policy = replace(route.policy, limits=limits, policy_digest="")
        routes.append(
            replace(
                route,
                max_physical_dispatches=max_pages,
                policy=policy,
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
    provider: str,
    *,
    max_pages: int = 1,
    result_limit: int = 1,
    max_response_bytes: int | None = None,
):
    registry = _registry_with_pages(
        provider,
        max_pages,
        max_response_bytes=max_response_bytes,
    )
    plan = compile_discovery_plan(
        PlanningRequest(
            source_ids=(provider,),
            query="bounded discovery",
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


def _pubmed_group():
    registry = foundation_registry()
    plan = compile_discovery_plan(
        PlanningRequest(("pubmed",), "bounded discovery", (), 1),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=BudgetCeilings(1, 2, 1, 0, 0, 40_000, 1),
    )
    return plan.dispatch_groups[0]


def _response(
    route,
    intent,
    body: bytes,
    *,
    status_code: Any = 200,
    content_type: str | None = "application/json",
    retry_after: Any = None,
    headers: tuple[tuple[Any, Any], ...] | None = None,
    claimed_decoded_bytes: int | None = None,
) -> DiscoveryGatewayResponse:
    origin = route.policy.origin
    if headers is None:
        headers = () if content_type is None else (("content-type", content_type),)
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
            wire_bytes=len(body),
            decoded_bytes=len(body) if claimed_decoded_bytes is None else claimed_decoded_bytes,
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


def _adapter_for(provider: str, *, monotonic_clock=None):
    module = _gateway_adapters_module()
    clock = _CountingClock() if monotonic_clock is None else monotonic_clock
    adapters = module.foundation_gateway_adapters(monotonic_clock=clock)
    adapter_id = next(case[1] for case in _PROVIDERS if case[0] == provider)
    return adapters[adapter_id]


async def _invoke(
    provider: str,
    bodies: list[bytes],
    *,
    max_pages: int = 1,
    result_limit: int = 1,
    statuses: list[object] | None = None,
    content_types: list[str | None] | None = None,
    retry_afters: list[str | None] | None = None,
    monotonic_clock=None,
):
    registry, plan = _plan_for(
        provider,
        max_pages=max_pages,
        result_limit=result_limit,
    )
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    intent = group.intents[0]
    statuses = statuses or [200] * len(bodies)
    content_types = content_types or ["application/json"] * len(bodies)
    retry_afters = retry_afters or [None] * len(bodies)
    responses = [
        _response(
            route,
            intent,
            body,
            status_code=status,
            content_type=content_type,
            retry_after=retry_after,
        )
        for body, status, content_type, retry_after in zip(
            bodies,
            statuses,
            content_types,
            retry_afters,
        )
    ]
    dispatch = _RecordingDispatch(responses)
    result = await _adapter_for(provider, monotonic_clock=monotonic_clock)(group, dispatch)
    return result, dispatch, group


async def _invoke_response(
    provider: str,
    response: DiscoveryGatewayResponse,
    *,
    max_pages: int = 1,
    result_limit: int = 1,
    monotonic_clock=None,
):
    _registry, plan = _plan_for(
        provider,
        max_pages=max_pages,
        result_limit=result_limit,
    )
    group = plan.dispatch_groups[0]
    dispatch = _RecordingDispatch([response])
    result = await _adapter_for(provider, monotonic_clock=monotonic_clock)(group, dispatch)
    return result, dispatch, group


async def _execute_adapter_plan(
    provider: str,
    responses: list[object],
    *,
    max_pages: int = 2,
    result_limit: int = 2,
    monotonic_clock=None,
    before_gateway_response=None,
):
    registry, plan = _plan_for(
        provider,
        max_pages=max_pages,
        result_limit=result_limit,
    )
    group = plan.dispatch_groups[0]
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
    adapters = _gateway_adapters_module().foundation_gateway_adapters(monotonic_clock=clock)
    dispatch_ids = iter(f"dispatch-{index}" for index in range(1, max_pages + 1))
    result = await execute_discovery_plan(
        plan,
        registry=registry,
        adapters={group.adapter_id: adapters[group.adapter_id]},
        gateway=gateway,
        policy_is_active=lambda _route_id, _digest: True,
        dispatch_id_factory=lambda: next(dispatch_ids),
    )
    return result, gateway_calls


def _assert_typed_error(error: BaseException, code: str) -> None:
    error_type = _adapter_error_type()
    assert type(error) is error_type
    assert error.code == code
    assert str(error) == code


@pytest.mark.asyncio
async def test_ncbi_trusted_input_callback_attribute_error_normalizes_but_adapter_error_propagates() -> None:
    module = _gateway_adapters_module()

    def attribute_failure(_group: object):
        raise AttributeError("synthetic structural callback failure")

    async def forbidden_dispatch(*_args, **_kwargs):
        raise AssertionError("trusted-input failure must precede dispatch")

    with pytest.raises(_adapter_error_type()) as caught:
        await module._execute_ncbi_esearch_summary(
            object(),
            forbidden_dispatch,
            _CountingClock(),
            trusted_inputs=attribute_failure,
            parse_esearch_ids=lambda *_args, **_kwargs: (),
            parse_summary_records=lambda *_args, **_kwargs: (),
            strict_rate_envelope=False,
        )

    _assert_typed_error(caught.value, "provider_payload_invalid")
    expected = _adapter_error_type()("provider_response_rejected")

    def typed_failure(_group: object):
        raise expected

    with pytest.raises(_adapter_error_type()) as propagated:
        await module._execute_ncbi_esearch_summary(
            object(),
            forbidden_dispatch,
            _CountingClock(),
            trusted_inputs=typed_failure,
            parse_esearch_ids=lambda *_args, **_kwargs: (),
            parse_summary_records=lambda *_args, **_kwargs: (),
            strict_rate_envelope=False,
        )

    assert propagated.value is expected

    group = _pubmed_group()
    route = foundation_registry().get_route(group.route_id)
    dispatch = _RecordingDispatch([_response(route, group.intents[0], b"{}")])

    def indexed_parser(*_args, **_kwargs):
        raise IndexError("synthetic indexed parser failure")

    with pytest.raises(_adapter_error_type()) as indexed:
        await module._execute_ncbi_esearch_summary(
            group,
            dispatch,
            _CountingClock(),
            trusted_inputs=module._trusted_pubmed_inputs,
            parse_esearch_ids=indexed_parser,
            parse_summary_records=lambda *_args, **_kwargs: (),
            strict_rate_envelope=False,
        )

    _assert_typed_error(indexed.value, "provider_payload_invalid")


@pytest.mark.parametrize(
    "mutation", ("intent", "query_container", "query_member", "binding_container", "binding_member", "limits")
)
def test_pubmed_trusted_inputs_reject_malformed_containers_before_dereference(mutation: str) -> None:
    module = _gateway_adapters_module()
    group = _pubmed_group()
    search, summary = group.intents
    if mutation == "intent":
        object.__setattr__(group, "intents", (object(), summary))
    elif mutation == "query_container":
        object.__setattr__(search, "query_pairs", [])
    elif mutation == "query_member":
        object.__setattr__(search.query_pairs[0], "name", object())
    elif mutation == "binding_container":
        object.__setattr__(summary, "query_bindings", [])
    elif mutation == "binding_member":
        object.__setattr__(summary.query_bindings[0], "binding_id", object())
    else:
        object.__setattr__(group, "limits", object())

    with pytest.raises(_adapter_error_type()) as caught:
        module._trusted_pubmed_inputs(group)

    _assert_typed_error(caught.value, "provider_payload_invalid")


def _assert_two_completed_physical_hops(result) -> None:
    usage = result.usage
    assert tuple(record.state for record in usage.physical_records) == (
        PhysicalDispatchState.SUCCEEDED,
        PhysicalDispatchState.SUCCEEDED,
    )
    assert usage.pages == 2
    assert usage.accounting.created == 2
    assert usage.accounting.debited == 2
    assert usage.accounting.released == 0
    assert usage.accounting.outstanding == 0


def _semantic_record(index: int, *, doi: str | None = None) -> dict[str, Any]:
    return {
        "paperId": f"S2-{index}",
        "title": f"Record {index}",
        "authors": [{"name": "Ada Lovelace"}],
        "abstract": "Inert abstract.",
        "tldr": {"text": "Inert snippet."},
        "externalIds": {} if doi is None else {"DOI": doi},
        "url": f"https://www.semanticscholar.org/paper/S2-{index}",
        "openAccessPdf": None,
    }


def _semantic_page(
    records: list[dict[str, Any]],
    *,
    total: int | None = None,
    offset: int = 0,
    next_cursor: object = None,
) -> bytes:
    payload: dict[str, Any] = {
        "data": records,
        "total": len(records) if total is None else total,
        "offset": offset,
    }
    if next_cursor is not None:
        payload["next"] = next_cursor
    return _json_bytes(payload)


def _normalized_record(candidate) -> dict[str, Any]:
    record = dict(candidate.record)
    record["provider_ids"] = dict(record["provider_ids"])
    return record


def _set_result_url(provider: str, payload: Any, field: str, value: str) -> None:
    if provider == "semantic_scholar":
        record = payload["data"][0]
        if field == "url":
            record["url"] = value
        else:
            record["openAccessPdf"] = {"url": value}
        return
    if provider == "crossref":
        record = payload["message"]["items"][0]
        if field == "url":
            record["URL"] = value
        else:
            pdf_link = next(link for link in record["link"] if link["content-type"] == "application/pdf")
            pdf_link["URL"] = value
        return
    if provider == "zenodo":
        record = payload["hits"]["hits"][0]
        if field == "url":
            record["links"]["self_html"] = value
        else:
            pdf_file = next(file for file in record["files"] if file["key"].endswith(".pdf"))
            pdf_file["links"]["self"] = value
        return
    if provider == "figshare":
        assert field == "url"
        record = payload[0]
        record["url_public_html"] = value
        record["url_public_api"] = value
        record["url"] = value
        return
    assert provider == "osf" and field == "url"
    payload["data"][0]["links"]["html"] = value


def test_parse_profiles_are_private_exact_immutable_and_version_keyed() -> None:
    module = _gateway_adapters_module()
    profiles = module._PARSING_PROFILES

    assert type(profiles) is MappingProxyType
    expected_adapter_ids = _ADAPTER_IDS + ("arxiv_v2", "pubmed_v2")
    assert set(profiles) == {
        *((adapter_id, "foundation-v2") for adapter_id in expected_adapter_ids),
        ("pubmed_v2", "pubmed-v2-ncbi-identity"),
    }
    assert (
        profiles[("pubmed_v2", "pubmed-v2-ncbi-identity")]
        is profiles[("pubmed_v2", "foundation-v2")]
        is module._FOUNDATION_PROFILE
    )
    expected = (2_097_152, 100, 16, 50_000, 65_536, 32, 500)
    for profile in profiles.values():
        assert (
            profile.max_input_bytes,
            profile.max_records,
            profile.max_depth,
            profile.max_nodes,
            profile.max_string_chars,
            profile.max_numeric_token_chars,
            profile.parse_deadline_ms,
        ) == expected
        with pytest.raises(FrozenInstanceError):
            profile.max_records = 101
    with pytest.raises(TypeError):
        profiles[("extra_v2", "foundation-v2")] = next(iter(profiles.values()))


def test_factory_exposes_exact_seven_gateway_adapter_ids() -> None:
    adapters = _gateway_adapters_module().foundation_gateway_adapters()
    assert tuple(adapters) == _ADAPTER_IDS + ("arxiv_v2", "pubmed_v2")
    assert all(callable(adapter) for adapter in adapters.values())


@pytest.mark.parametrize("provider", _PROVIDER_NAMES)
def test_sanitized_fixture_envelopes_and_consumed_record_keys_are_frozen(provider: str) -> None:
    success = _fixture_payload(provider)
    empty = _fixture_payload(provider, "empty")

    if provider == "semantic_scholar":
        record = success["data"][0]
        assert {"total": success["total"], "offset": success["offset"]} == {
            "total": 1,
            "offset": 0,
        }
        assert empty == {"data": [], "total": 0, "offset": 0}
    elif provider == "crossref":
        assert {
            "status": success["status"],
            "message-type": success["message-type"],
            "message-version": success["message-version"],
            "total-results": success["message"]["total-results"],
        } == {
            "status": "ok",
            "message-type": "work-list",
            "message-version": "1.0.0",
            "total-results": 1,
        }
        assert empty["message"]["items"] == []
        assert empty["message"]["total-results"] == 0
        record = success["message"]["items"][0]
    elif provider == "zenodo":
        assert success["hits"]["total"] == 1
        assert empty == {"hits": {"hits": [], "total": 0}}
        record = success["hits"]["hits"][0]
        assert record["links"]["self_html"] == "https://zenodo.org/records/1001"
        assert not record["files"][0]["key"].endswith(".pdf")
        assert record["files"][1]["key"].endswith(".pdf")
    elif provider == "figshare":
        assert empty == []
        record = success[0]
        assert record["url_public_html"].startswith("https://figshare.com/")
    else:
        assert empty == {"data": [], "links": {"next": None}}
        record = success["data"][0]

    assert set(record) == _EXPECTED_SUCCESS_RECORD_KEYS[provider]


@pytest.mark.parametrize("provider", _PROVIDER_NAMES)
@pytest.mark.asyncio
async def test_success_fixtures_use_exact_envelopes_and_normalized_keyset(provider: str) -> None:
    result, dispatch, _group = await _invoke(provider, [_fixture_bytes(provider, "success")])

    assert type(result) is DiscoveryAdapterResult
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    record = _normalized_record(candidate)
    assert record == _EXPECTED_NORMALIZED_RECORDS[provider]
    assert set(record) == _NORMALIZED_KEYS
    expected_identity = DiscoveryOutcomeIdentity.from_fingerprint(
        build_fingerprint(_EXPECTED_NORMALIZED_RECORDS[provider])
    )
    assert candidate.candidate_id == expected_identity.document_id
    assert len(dispatch.calls) == 1
    assert dispatch.calls[0][1] is None


@pytest.mark.parametrize(("provider", "_adapter_id", "provider_id_key"), _PROVIDERS)
@pytest.mark.asyncio
async def test_success_fixtures_pin_exact_provider_id_keys(
    provider: str,
    _adapter_id: str,
    provider_id_key: str,
) -> None:
    result, _dispatch, _group = await _invoke(provider, [_fixture_bytes(provider, "success")])
    provider_ids = dict(result.candidates[0].record["provider_ids"])

    assert provider_id_key in provider_ids
    assert "id" not in provider_ids
    assert set(provider_ids) == _EXPECTED_PROVIDER_ID_KEYS[provider]


@pytest.mark.parametrize("provider", _PROVIDER_NAMES)
@pytest.mark.asyncio
async def test_valid_empty_fixtures_return_exact_empty_result(provider: str) -> None:
    result, dispatch, _group = await _invoke(provider, [_fixture_bytes(provider, "empty")])

    assert result == DiscoveryAdapterResult(candidates=())
    assert len(dispatch.calls) == 1


@pytest.mark.asyncio
async def test_same_doi_spelling_converges_across_all_provider_fixtures() -> None:
    variants = {
        "semantic_scholar": "DOI: 10.5555/SHARED.DISCOVERY.2026",
        "crossref": "10.5555/SHARED.DISCOVERY.2026",
        "zenodo": "https://doi.org/10.5555/shared.discovery.2026",
        "figshare": "doi:10.5555/shared.discovery.2026",
        "osf": "10.5555/SHARED.DISCOVERY.2026.",
    }
    candidate_ids = set()
    for provider in _PROVIDER_NAMES:
        payload = _fixture_payload(provider)
        doi = variants[provider]
        if provider == "semantic_scholar":
            payload["data"][0]["externalIds"]["DOI"] = doi
        elif provider == "crossref":
            payload["message"]["items"][0]["DOI"] = doi
        elif provider == "zenodo":
            payload["hits"]["hits"][0]["doi"] = doi
            payload["hits"]["hits"][0]["metadata"]["doi"] = doi
        elif provider == "figshare":
            payload[0]["doi"] = doi
        else:
            payload["data"][0]["attributes"]["doi"] = doi
        result, _dispatch, _group = await _invoke(provider, [_json_bytes(payload)])
        candidate = result.candidates[0]
        normalized = _normalized_record(candidate)
        assert normalized["doi"] == "10.5555/shared.discovery.2026"
        assert normalized["provider_ids"]["doi"] == "10.5555/shared.discovery.2026"
        candidate_ids.add(candidate.candidate_id)

    assert len(candidate_ids) == 1


@pytest.mark.parametrize(
    ("provider", "foreign_provider"),
    tuple(
        (provider, foreign_provider)
        for provider in _PROVIDER_NAMES
        for foreign_provider in _PROVIDER_NAMES
        if provider != foreign_provider
    ),
)
@pytest.mark.asyncio
async def test_cross_feed_provider_envelopes_are_rejected(
    provider: str,
    foreign_provider: str,
) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke(provider, [_fixture_bytes(foreign_provider, "success")])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("field", ("status", "message-type", "message-version"))
@pytest.mark.parametrize("remove", (False, True))
@pytest.mark.asyncio
async def test_crossref_requires_exact_success_envelope_metadata(
    field: str,
    remove: bool,
) -> None:
    payload = _fixture_payload("crossref")
    if remove:
        payload.pop(field)
    else:
        payload[field] = "wrong"

    with pytest.raises(Exception) as caught:
        await _invoke("crossref", [_json_bytes(payload)])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("remove", (False, True))
@pytest.mark.asyncio
async def test_osf_requires_preprint_resource_type(remove: bool) -> None:
    payload = _fixture_payload("osf")
    record = payload["data"][0]
    if remove:
        record.pop("type")
    else:
        record["type"] = "nodes"

    with pytest.raises(Exception) as caught:
        await _invoke("osf", [_json_bytes(payload)])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("provider", _PROVIDER_NAMES)
@pytest.mark.asyncio
async def test_unknown_fields_are_counted_but_not_retained(provider: str) -> None:
    payload = _fixture_payload(provider)
    if provider == "semantic_scholar":
        payload["unknown"] = {"api_key": "fixture-secret", "nested": [1, 2, 3]}
        payload["data"][0]["unknown"] = "ignored"
    elif provider == "crossref":
        payload["unknown"] = "ignored"
        payload["message"]["items"][0]["unknown"] = "fixture-secret"
    elif provider == "zenodo":
        payload["unknown"] = ["ignored"]
        payload["hits"]["hits"][0]["unknown"] = "fixture-secret"
    elif provider == "figshare":
        payload[0]["unknown"] = {"token": "fixture-secret"}
    else:
        payload["unknown"] = {"token": "fixture-secret"}
        payload["data"][0]["unknown"] = "ignored"

    result, _dispatch, _group = await _invoke(provider, [_json_bytes(payload)])
    record = dict(result.candidates[0].record)
    assert set(record) == _NORMALIZED_KEYS
    assert "fixture-secret" not in repr(record)


@pytest.mark.parametrize(
    "field",
    ("adapter_id", "adapter_version"),
)
@pytest.mark.asyncio
async def test_unknown_adapter_or_version_rejects_before_dispatch(field: str) -> None:
    _registry, plan = _plan_for("semantic_scholar")
    group = plan.dispatch_groups[0]
    hostile_group = replace(group, **{field: "unknown_v2"})
    dispatch = _RecordingDispatch([])

    with pytest.raises(Exception) as caught:
        await _adapter_for("semantic_scholar")(hostile_group, dispatch)

    _assert_typed_error(caught.value, "provider_payload_invalid")
    assert dispatch.calls == []


@pytest.mark.parametrize("provider", _PROVIDER_NAMES)
@pytest.mark.parametrize("status_code", (201, 204, 400, 503, True, 200.0, "200", 429.0))
@pytest.mark.asyncio
async def test_non_200_status_short_circuits_before_content_type_body_or_clock(
    provider: str,
    status_code: object,
) -> None:
    def forbidden_clock() -> float:
        raise AssertionError("parse clock must not run for rejected status")

    with pytest.raises(Exception) as caught:
        await _invoke(
            provider,
            [b"not-json fixture-secret"],
            statuses=[status_code],
            content_types=[None],
            retry_afters=["120"],
            monotonic_clock=forbidden_clock,
        )

    _assert_typed_error(caught.value, "provider_response_rejected")
    assert caught.value.retry_after is None
    assert "fixture-secret" not in repr(caught.value)


@pytest.mark.parametrize(
    "retry_after",
    ("0", "001", "120", "Wed, 21 Oct 2015 07:28:00 GMT"),
)
@pytest.mark.parametrize("provider", _PROVIDER_NAMES)
@pytest.mark.asyncio
async def test_429_never_parses_body_and_preserves_only_valid_retry_after(
    provider: str,
    retry_after: str,
) -> None:
    def forbidden_clock() -> float:
        raise AssertionError("parse clock must not run for 429")

    with pytest.raises(Exception) as caught:
        await _invoke(
            provider,
            [b'{"error":"fixture-secret",'],
            statuses=[429],
            content_types=[None],
            retry_afters=[retry_after],
            monotonic_clock=forbidden_clock,
        )

    _assert_typed_error(caught.value, "provider_rate_limited")
    assert caught.value.retry_after == retry_after
    assert "fixture-secret" not in repr(caught.value)


@pytest.mark.parametrize(
    "retry_after",
    (
        "-1",
        "+1",
        "1.5",
        "tomorrow",
        " Wed, 21 Oct 2015 07:28:00 GMT",
        "Wed, 21 Oct 2015 07:28:00 PST",
        "Wed, 21 Oct 2015 07:28:00 GMT\nsecret",
        "Sunday, 06-Nov-94 08:49:37 GMT",
        "Sun Nov  6 08:49:37 1994",
        "wed, 21 Oct 2015 07:28:00 GMT",
        "Wed, 21 Oct 2015 07:28:00 GMT ",
        "Wed, 31 Feb 2015 07:28:00 GMT",
    ),
)
@pytest.mark.asyncio
async def test_429_drops_invalid_visible_retry_after_text(retry_after: str) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke(
            "semantic_scholar",
            [b"fixture-secret"],
            statuses=[429],
            content_types=[None],
            retry_afters=[retry_after],
        )

    _assert_typed_error(caught.value, "provider_rate_limited")
    assert caught.value.retry_after is None
    assert retry_after not in repr(caught.value)


@pytest.mark.parametrize(
    ("provider", "content_type"),
    (
        ("semantic_scholar", "application/json"),
        ("figshare", "Application/JSON; Charset=UTF-8"),
        ("crossref", "application/vnd.crossref-api-message+json; version=1.0"),
        ("zenodo", "application/vnd.inveniordm.v1+json"),
        ("osf", "application/vnd.api+json; charset=utf-8"),
    ),
)
@pytest.mark.asyncio
async def test_strict_json_content_types_accept_json_and_valid_vendor_suffixes(
    provider: str,
    content_type: str,
) -> None:
    result, _dispatch, _group = await _invoke(
        provider,
        [_fixture_bytes(provider, "empty")],
        content_types=[content_type],
    )
    assert result == DiscoveryAdapterResult(candidates=())


@pytest.mark.parametrize(
    "content_type",
    (
        None,
        "",
        "text/json",
        "text/plain",
        "application/jsonp",
        "application/+json",
        "application/ +json",
        "application/foo bar+json",
        "application/foo/+json",
        "application/*+json",
        "application/json;",
        "application/json; charset",
        "application/json; =x",
        "application/vnd.example+xml",
        "application/json, text/plain",
    ),
)
@pytest.mark.asyncio
async def test_missing_or_wrong_content_type_rejects_before_json_parse(
    content_type: str | None,
) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke(
            "semantic_scholar",
            [b"not-json fixture-secret"],
            content_types=[content_type],
        )

    _assert_typed_error(caught.value, "provider_response_rejected")
    assert "fixture-secret" not in repr(caught.value)


@pytest.mark.asyncio
async def test_duplicate_content_type_metadata_fails_closed() -> None:
    registry, plan = _plan_for("semantic_scholar")
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    response = _response(
        route,
        group.intents[0],
        _fixture_bytes("semantic_scholar", "empty"),
        headers=(("content-type", "application/json"), ("Content-Type", "application/json")),
    )

    with pytest.raises(Exception) as caught:
        await _invoke_response("semantic_scholar", response)

    _assert_typed_error(caught.value, "provider_response_rejected")


@pytest.mark.parametrize(
    ("body", "expected_code"),
    (
        (b"", "provider_payload_invalid"),
        (b" \t\r\n", "provider_payload_invalid"),
        (b"{", "provider_payload_invalid"),
        (b'{"data":[]} trailing', "provider_payload_invalid"),
        (b'[{"data":[]}]', "provider_payload_invalid"),
        (b'{"data":[],"data":[]}', "provider_payload_invalid"),
        (
            b'{"data":[{"paperId":"one","paperId":"two"}]}',
            "provider_payload_invalid",
        ),
        (b'{"data":[],"d\\u0061ta":[]}', "provider_payload_invalid"),
        (b'{"data":[],"unknown":NaN}', "provider_payload_invalid"),
        (b'{"data":[],"unknown":Infinity}', "provider_payload_invalid"),
        (b'{"data":[],"unknown":-Infinity}', "provider_payload_invalid"),
        (b'{"data":[],"unknown":1e309}', "provider_payload_invalid"),
        (b'{"data":[],"unknown":"\\ud800"}', "provider_payload_invalid"),
        (b"\xef\xbb\xbf" + b'{"data":[]}', "provider_payload_invalid"),
        (b"\xff\xfe" + b'{"data":[]}', "provider_payload_invalid"),
        (b"\xfe\xff" + b'{"data":[]}', "provider_payload_invalid"),
        (b'{"data":[]}\xff', "provider_payload_invalid"),
        pytest.param(
            b'{"data":[],"unknown":' + (b"9" * 10_000) + b"}",
            "provider_parse_limit_exceeded",
            id="oversized-integer-token",
        ),
        pytest.param(
            b'{"data":[],"unknown":1e' + (b"9" * 10_000) + b"}",
            "provider_parse_limit_exceeded",
            id="oversized-exponent-token",
        ),
    ),
)
@pytest.mark.asyncio
async def test_raw_decoder_attacks_fail_with_typed_sanitized_errors(
    body: bytes,
    expected_code: str,
) -> None:
    with pytest.raises(Exception) as caught:
        await _invoke("semantic_scholar", [body])

    _assert_typed_error(caught.value, expected_code)
    assert repr(caught.value) == f"DiscoveryAdapterError('{expected_code}')"


@pytest.mark.parametrize(
    ("token", "should_pass"),
    (
        ("9" * 32, True),
        ("9" * 33, False),
        ("-0." + ("1" * 29), True),
        ("-0." + ("1" * 30), False),
        ("0." + ("1" * 28) + "e1", True),
        ("0." + ("1" * 29) + "e1", False),
        ("0." + ("1" * 27) + "e+1", True),
        ("0." + ("1" * 28) + "e+1", False),
    ),
)
@pytest.mark.asyncio
async def test_numeric_token_limit_counts_sign_decimal_and_exponent(
    token: str,
    should_pass: bool,
) -> None:
    body = b'{"data":[],"total":0,"offset":0,"unknown":' + token.encode("ascii") + b"}"
    if should_pass:
        result, _dispatch, _group = await _invoke("semantic_scholar", [body])
        assert result == DiscoveryAdapterResult(candidates=())
    else:
        with pytest.raises(Exception) as caught:
            await _invoke("semantic_scholar", [body])
        _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.parametrize(("extra_byte", "should_pass"), ((False, True), (True, False)))
@pytest.mark.asyncio
async def test_input_byte_limit_uses_actual_body_length_not_trace_claim(
    extra_byte: bool,
    should_pass: bool,
) -> None:
    registry, plan = _plan_for("semantic_scholar")
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    base = _fixture_bytes("semantic_scholar", "empty")
    body = base + (b" " * (2_097_152 - len(base))) + (b" " if extra_byte else b"")
    response = _response(
        route,
        group.intents[0],
        body,
        claimed_decoded_bytes=1,
    )

    if should_pass:
        result, _dispatch, _group = await _invoke_response("semantic_scholar", response)
        assert result == DiscoveryAdapterResult(candidates=())
    else:
        with pytest.raises(Exception) as caught:
            await _invoke_response("semantic_scholar", response)
        _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.parametrize(("extra_byte", "should_pass"), ((False, True), (True, False)))
@pytest.mark.asyncio
async def test_input_byte_limit_clamps_to_stricter_route_limit(
    extra_byte: bool,
    should_pass: bool,
) -> None:
    route_limit = 128
    registry, plan = _plan_for(
        "semantic_scholar",
        max_response_bytes=route_limit,
    )
    group = plan.dispatch_groups[0]
    route = registry.get_route(group.route_id)
    base = _fixture_bytes("semantic_scholar", "empty")
    body = base + (b" " * (route_limit - len(base))) + (b" " if extra_byte else b"")
    dispatch = _RecordingDispatch([_response(route, group.intents[0], body)])

    assert group.limits.max_response_bytes == route_limit < 2_097_152
    if should_pass:
        result = await _adapter_for("semantic_scholar")(group, dispatch)
        assert result == DiscoveryAdapterResult(candidates=())
    else:
        with pytest.raises(Exception) as caught:
            await _adapter_for("semantic_scholar")(group, dispatch)
        _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


def _payload_with_deep_unknown(max_depth: int) -> bytes:
    value: object = "leaf"
    # Root is depth 1; the outer unknown value is depth 2.
    for _ in range(max_depth - 2):
        value = [value]
    return _json_bytes({"data": [], "total": 0, "offset": 0, "unknown": value})


@pytest.mark.parametrize(("depth", "should_pass"), ((16, True), (17, False)))
@pytest.mark.asyncio
async def test_depth_limit_counts_root_as_one_and_all_unknown_values(
    depth: int,
    should_pass: bool,
) -> None:
    body = _payload_with_deep_unknown(depth)
    if should_pass:
        result, _dispatch, _group = await _invoke("semantic_scholar", [body])
        assert result == DiscoveryAdapterResult(candidates=())
    else:
        with pytest.raises(Exception) as caught:
            await _invoke("semantic_scholar", [body])
        _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.parametrize(("list_items", "should_pass"), ((49_991, True), (49_992, False)))
@pytest.mark.asyncio
async def test_node_limit_counts_root_mapping_keys_values_and_list_elements(
    list_items: int,
    should_pass: bool,
) -> None:
    # root + four key/value pairs + N list elements = 9 + N
    body = _json_bytes({"data": [], "total": 0, "offset": 0, "unknown": [0] * list_items})
    if should_pass:
        result, _dispatch, _group = await _invoke("semantic_scholar", [body])
        assert result == DiscoveryAdapterResult(candidates=())
    else:
        with pytest.raises(Exception) as caught:
            await _invoke("semantic_scholar", [body])
        _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.parametrize(
    ("location", "chars", "should_pass"),
    (
        ("key", 65_536, True),
        ("key", 65_537, False),
        ("value", 65_536, True),
        ("value", 65_537, False),
    ),
)
@pytest.mark.asyncio
async def test_string_limit_is_per_decoded_key_or_value_in_unicode_codepoints(
    location: str,
    chars: int,
    should_pass: bool,
) -> None:
    text = "é" * chars
    payload = (
        {"data": [], "total": 0, "offset": 0, text: 1}
        if location == "key"
        else {"data": [], "total": 0, "offset": 0, "unknown": text}
    )
    body = _json_bytes(payload)
    if should_pass:
        result, _dispatch, _group = await _invoke("semantic_scholar", [body])
        assert result == DiscoveryAdapterResult(candidates=())
    else:
        with pytest.raises(Exception) as caught:
            await _invoke("semantic_scholar", [body])
        _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_string_limit_is_not_an_aggregate_character_budget() -> None:
    text = "x" * 65_536
    body = _json_bytes(
        {
            "data": [],
            "total": 0,
            "offset": 0,
            "unknown_one": text,
            "unknown_two": text,
        }
    )
    result, _dispatch, _group = await _invoke("semantic_scholar", [body])
    assert result == DiscoveryAdapterResult(candidates=())


@pytest.mark.parametrize("location", ("key", "value"))
@pytest.mark.parametrize(("chars", "should_pass"), ((65_536, True), (65_537, False)))
@pytest.mark.asyncio
async def test_string_limit_counts_decoded_escapes_not_raw_token_characters(
    location: str,
    chars: int,
    should_pass: bool,
) -> None:
    escaped = b"\\u0061" * chars
    if location == "key":
        body = b'{"data":[],"total":0,"offset":0,"' + escaped + b'":1}'
    else:
        body = b'{"data":[],"total":0,"offset":0,"unknown":"' + escaped + b'"}'

    if should_pass:
        result, _dispatch, _group = await _invoke("semantic_scholar", [body])
        assert result == DiscoveryAdapterResult(candidates=())
    else:
        with pytest.raises(Exception) as caught:
            await _invoke("semantic_scholar", [body])
        _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.parametrize(("record_count", "should_pass"), ((100, True), (101, False)))
@pytest.mark.asyncio
async def test_profile_record_boundary_is_exact_and_never_truncates(
    record_count: int,
    should_pass: bool,
) -> None:
    body = _semantic_page([_semantic_record(index) for index in range(record_count)])
    if should_pass:
        result, _dispatch, _group = await _invoke(
            "semantic_scholar",
            [body],
            result_limit=100,
        )
        assert len(result.candidates) == 100
    else:
        with pytest.raises(Exception) as caught:
            await _invoke("semantic_scholar", [body], result_limit=100)
        _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


def _zenodo_records(count: int) -> list[dict[str, Any]]:
    base = _fixture_payload("zenodo")["hits"]["hits"][0]
    records = []
    for index in range(count):
        records.append(
            {
                **base,
                "id": 10_000 + index,
                "doi": None,
                "metadata": {
                    **base["metadata"],
                    "title": f"Zenodo record {index}",
                    "doi": None,
                },
            }
        )
    return records


@pytest.mark.parametrize(("record_count", "should_pass"), ((25, True), (26, False)))
@pytest.mark.asyncio
async def test_record_limit_clamps_to_stricter_zenodo_route_max_results(
    record_count: int,
    should_pass: bool,
) -> None:
    body = _json_bytes({"hits": {"hits": _zenodo_records(record_count), "total": record_count}})
    if should_pass:
        result, _dispatch, _group = await _invoke("zenodo", [body], result_limit=25)
        assert len(result.candidates) == 25
    else:
        with pytest.raises(Exception) as caught:
            await _invoke("zenodo", [body], result_limit=25)
        _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.parametrize(("second_page_count", "should_pass"), ((40, True), (41, False)))
@pytest.mark.asyncio
async def test_record_cap_is_aggregate_across_pages_and_atomic(
    second_page_count: int,
    should_pass: bool,
) -> None:
    first = _semantic_page(
        [_semantic_record(index) for index in range(60)],
        total=60 + second_page_count,
        next_cursor=60,
    )
    second = _semantic_page(
        [_semantic_record(60 + index) for index in range(second_page_count)],
        total=60 + second_page_count,
        offset=60,
    )
    if should_pass:
        result, dispatch, _group = await _invoke(
            "semantic_scholar",
            [first, second],
            max_pages=2,
            result_limit=100,
        )
        assert len(result.candidates) == 100
        assert [call[1] for call in dispatch.calls] == [None, NumericCursor(60)]
    else:
        with pytest.raises(Exception) as caught:
            await _invoke(
                "semantic_scholar",
                [first, second],
                max_pages=2,
                result_limit=100,
            )
        _assert_typed_error(caught.value, "provider_parse_limit_exceeded")


@pytest.mark.asyncio
async def test_malformed_later_record_rejects_the_whole_page_atomically() -> None:
    malformed = _semantic_record(2)
    malformed["paperId"] = True
    body = _semantic_page([_semantic_record(1), malformed])

    with pytest.raises(Exception) as caught:
        await _invoke("semantic_scholar", [body], result_limit=2)

    _assert_typed_error(caught.value, "provider_payload_invalid")


def _set_stable_id(provider: str, payload: Any, value: object, *, remove: bool) -> bytes:
    if provider == "semantic_scholar":
        record = payload["data"][0]
        key = "paperId"
    elif provider == "crossref":
        record = payload["message"]["items"][0]
        key = "DOI"
    elif provider == "zenodo":
        record = payload["hits"]["hits"][0]
        key = "id"
    elif provider == "figshare":
        record = payload[0]
        key = "id"
    else:
        record = payload["data"][0]
        key = "id"
    if remove:
        record.pop(key, None)
    else:
        record[key] = value
    return _json_bytes(payload)


@pytest.mark.parametrize("provider", _PROVIDER_NAMES)
@pytest.mark.parametrize(
    ("value", "remove"),
    ((None, False), (True, False), ([], False), ("", False), (None, True)),
)
@pytest.mark.asyncio
async def test_missing_or_malformed_stable_provider_ids_are_rejected_even_with_doi(
    provider: str,
    value: object,
    remove: bool,
) -> None:
    body = _set_stable_id(provider, _fixture_payload(provider), value, remove=remove)
    with pytest.raises(Exception) as caught:
        await _invoke(provider, [body])
    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("provider", ("zenodo", "figshare"))
@pytest.mark.parametrize("value", (0, -1, 1.5, "1001"))
@pytest.mark.asyncio
async def test_numeric_provider_ids_require_exact_positive_integer_live_shape(
    provider: str,
    value: object,
) -> None:
    body = _set_stable_id(provider, _fixture_payload(provider), value, remove=False)
    with pytest.raises(Exception) as caught:
        await _invoke(provider, [body])
    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("provider", ("semantic_scholar", "crossref", "osf"))
@pytest.mark.parametrize("value", (1, 1.5, {}))
@pytest.mark.asyncio
async def test_string_provider_ids_reject_numeric_and_container_spoofs(
    provider: str,
    value: object,
) -> None:
    body = _set_stable_id(provider, _fixture_payload(provider), value, remove=False)
    with pytest.raises(Exception) as caught:
        await _invoke(provider, [body])
    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("provider", ("semantic_scholar", "crossref", "zenodo"))
@pytest.mark.parametrize("value", (True, -1, 1.5, "1", [], None))
@pytest.mark.asyncio
async def test_totals_reject_bool_negative_float_string_container_and_null(
    provider: str,
    value: object,
) -> None:
    payload = _fixture_payload(provider)
    if provider == "semantic_scholar":
        payload["total"] = value
    elif provider == "crossref":
        payload["message"]["total-results"] = value
    else:
        payload["hits"]["total"] = value

    with pytest.raises(Exception) as caught:
        await _invoke(provider, [_json_bytes(payload)])
    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("value", (True, -1, 1.5, "0", [], None))
@pytest.mark.asyncio
async def test_semantic_scholar_offset_rejects_malformed_integer_shapes(value: object) -> None:
    payload = _fixture_payload("semantic_scholar")
    payload["offset"] = value
    with pytest.raises(Exception) as caught:
        await _invoke("semantic_scholar", [_json_bytes(payload)])
    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("missing_field", ("total", "offset"))
@pytest.mark.asyncio
async def test_semantic_scholar_requires_official_envelope_fields(missing_field: str) -> None:
    payload = _fixture_payload("semantic_scholar", "empty")
    payload.pop(missing_field)

    with pytest.raises(Exception) as caught:
        await _invoke("semantic_scholar", [_json_bytes(payload)])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("provider", ("semantic_scholar", "crossref", "zenodo"))
@pytest.mark.asyncio
async def test_declared_total_cannot_be_less_than_consumed_records(provider: str) -> None:
    payload = _fixture_payload(provider)
    if provider == "semantic_scholar":
        payload["total"] = 0
    elif provider == "crossref":
        payload["message"]["total-results"] = 0
    else:
        payload["hits"]["total"] = 0

    with pytest.raises(Exception) as caught:
        await _invoke(provider, [_json_bytes(payload)])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize(
    "next_cursor",
    (True, -1, 0, 1.5, "1", [], {}, MAX_PAGINATION_CURSOR + 1),
)
@pytest.mark.asyncio
async def test_semantic_scholar_rejects_nonprogressing_or_malformed_next_cursor(
    next_cursor: object,
) -> None:
    body = _semantic_page(
        [_semantic_record(1)],
        total=2,
        offset=0,
        next_cursor=next_cursor,
    )
    with pytest.raises(Exception) as caught:
        await _invoke(
            "semantic_scholar",
            [body],
            max_pages=2,
            result_limit=2,
        )
    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_semantic_scholar_next_offset_cannot_exceed_declared_total() -> None:
    body = _semantic_page(
        [_semantic_record(1)],
        total=2,
        offset=0,
        next_cursor=3,
    )

    with pytest.raises(Exception) as caught:
        await _invoke(
            "semantic_scholar",
            [body],
            max_pages=2,
            result_limit=2,
        )

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize(
    ("provider", "mutator"),
    (
        ("semantic_scholar", lambda payload: payload.update(data={})),
        ("crossref", lambda payload: payload.update(message=[])),
        ("zenodo", lambda payload: payload.update(hits=[])),
        ("figshare", lambda payload: payload.append("not-a-record")),
        ("osf", lambda payload: payload.update(data={})),
    ),
)
@pytest.mark.asyncio
async def test_root_and_envelope_schema_drift_is_rejected_atomically(provider: str, mutator) -> None:
    payload = _fixture_payload(provider)
    mutator(payload)
    with pytest.raises(Exception) as caught:
        await _invoke(provider, [_json_bytes(payload)])
    _assert_typed_error(caught.value, "provider_payload_invalid")


def _set_payload_path(payload: Any, path: tuple[str | int, ...], value: object) -> None:
    target = payload
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("provider", "path", "wrong_shape"),
    (
        pytest.param("semantic_scholar", ("data", 0, "authors"), {}, id="s2-authors"),
        pytest.param("semantic_scholar", ("data", 0, "externalIds"), [], id="s2-external-ids"),
        pytest.param("semantic_scholar", ("data", 0, "tldr"), [], id="s2-tldr"),
        pytest.param("semantic_scholar", ("data", 0, "openAccessPdf"), [], id="s2-open-access-pdf"),
        pytest.param("crossref", ("message", "items", 0, "title"), {}, id="crossref-title"),
        pytest.param("crossref", ("message", "items", 0, "author"), {}, id="crossref-author"),
        pytest.param("crossref", ("message", "items", 0, "link"), {}, id="crossref-link"),
        pytest.param("zenodo", ("hits", "hits", 0, "metadata"), [], id="zenodo-metadata"),
        pytest.param("zenodo", ("hits", "hits", 0, "files"), {}, id="zenodo-files"),
        pytest.param("zenodo", ("hits", "hits", 0, "links"), [], id="zenodo-links"),
        pytest.param("osf", ("data", 0, "attributes"), [], id="osf-attributes"),
        pytest.param("osf", ("data", 0, "links"), [], id="osf-links"),
        pytest.param("figshare", (0, "title"), True, id="figshare-title"),
        pytest.param("figshare", (0, "doi"), [], id="figshare-doi"),
        pytest.param("figshare", (0, "url_public_html"), {}, id="figshare-url"),
    ),
)
@pytest.mark.asyncio
async def test_wrong_shaped_consumed_fields_are_rejected(
    provider: str,
    path: tuple[str | int, ...],
    wrong_shape: object,
) -> None:
    payload = _fixture_payload(provider)
    _set_payload_path(payload, path, wrong_shape)

    with pytest.raises(Exception) as caught:
        await _invoke(provider, [_json_bytes(payload)])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize("provider", _PROVIDER_NAMES)
@pytest.mark.asyncio
async def test_provider_page_cannot_exceed_requested_page_size(provider: str) -> None:
    payload = _fixture_payload(provider)
    if provider == "semantic_scholar":
        payload["data"].append(_semantic_record(2, doi="10.5555/overfull.2"))
        payload["total"] = 2
    elif provider == "crossref":
        duplicate = json.loads(json.dumps(payload["message"]["items"][0]))
        duplicate["DOI"] = "10.5555/overfull.2"
        payload["message"]["items"].append(duplicate)
        payload["message"]["total-results"] = 2
    elif provider == "zenodo":
        duplicate = json.loads(json.dumps(payload["hits"]["hits"][0]))
        duplicate["id"] = 1002
        duplicate["doi"] = "10.5555/overfull.2"
        duplicate["metadata"]["doi"] = "10.5555/overfull.2"
        payload["hits"]["hits"].append(duplicate)
        payload["hits"]["total"] = 2
    elif provider == "figshare":
        duplicate = json.loads(json.dumps(payload[0]))
        duplicate["id"] = 2002
        duplicate["doi"] = "10.5555/overfull.2"
        payload.append(duplicate)
    else:
        duplicate = json.loads(json.dumps(payload["data"][0]))
        duplicate["id"] = "osf002"
        duplicate["attributes"]["doi"] = "10.5555/overfull.2"
        payload["data"].append(duplicate)

    with pytest.raises(Exception) as caught:
        await _invoke(provider, [_json_bytes(payload)], result_limit=1)

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_crossref_rejects_blank_consumed_title() -> None:
    payload = _fixture_payload("crossref")
    payload["message"]["items"][0]["title"] = ["   "]

    with pytest.raises(Exception) as caught:
        await _invoke("crossref", [_json_bytes(payload)])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.parametrize(
    ("provider", "expected_cursor"),
    (
        ("semantic_scholar", 1),
        ("crossref", 1),
        ("zenodo", 2),
        ("figshare", 2),
        ("osf", 2),
    ),
)
@pytest.mark.asyncio
async def test_provider_specific_pagination_derives_only_local_numeric_cursor(
    provider: str,
    expected_cursor: int,
) -> None:
    first = _fixture_payload(provider)
    second = _fixture_bytes(provider, "empty")
    if provider == "semantic_scholar":
        first["total"] = 2
        first["next"] = 1
        second_payload = json.loads(json.dumps(first))
        second_payload["offset"] = 1
        second_payload.pop("next")
        second = _json_bytes(second_payload)
    elif provider == "crossref":
        first["message"]["total-results"] = 2
        second_payload = _fixture_payload(provider)
        second_payload["message"]["total-results"] = 2
        second = _json_bytes(second_payload)
    elif provider == "zenodo":
        first["hits"]["total"] = 2
        second_payload = _fixture_payload(provider)
        second_payload["hits"]["total"] = 2
        second = _json_bytes(second_payload)
    elif provider == "osf":
        first["links"]["next"] = "file://127.0.0.1/private?token=fixture-secret"

    result, dispatch, _group = await _invoke(
        provider,
        [_json_bytes(first), second],
        max_pages=2,
        result_limit=1,
    )

    assert len(result.candidates) == 1
    assert [call[1] for call in dispatch.calls] == [None, NumericCursor(expected_cursor)]
    assert all(type(call[1]) in {type(None), NumericCursor} for call in dispatch.calls)
    assert "fixture-secret" not in repr(result)


@pytest.mark.asyncio
async def test_semantic_scholar_rejects_nonempty_page_with_mismatched_offset() -> None:
    first = _semantic_page([_semantic_record(1)], total=2, next_cursor=1)
    second = _semantic_page([_semantic_record(2)], total=2, offset=0)

    with pytest.raises(Exception) as caught:
        await _invoke(
            "semantic_scholar",
            [first, second],
            max_pages=2,
            result_limit=2,
        )

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_zenodo_absolute_next_link_is_ignored_and_total_alone_drives_pagination() -> None:
    payload = _fixture_payload("zenodo")
    payload["links"] = {"next": "http://169.254.169.254/latest/meta-data"}
    payload["hits"]["total"] = 1
    result, dispatch, _group = await _invoke("zenodo", [_json_bytes(payload)])

    assert len(result.candidates) == 1
    assert len(dispatch.calls) == 1
    assert "169.254.169.254" not in repr(result)


@pytest.mark.asyncio
async def test_last_allowed_page_never_dispatches_advertised_n_plus_one() -> None:
    first = _semantic_page([_semantic_record(1)], total=3, next_cursor=1)
    second = _semantic_page([_semantic_record(2)], total=3, offset=1, next_cursor=2)

    result, dispatch, _group = await _invoke(
        "semantic_scholar",
        [first, second],
        max_pages=2,
        result_limit=2,
    )

    assert len(result.candidates) == 2
    assert [call[1] for call in dispatch.calls] == [None, NumericCursor(1)]


@pytest.mark.asyncio
async def test_figshare_paginates_only_when_page_is_full() -> None:
    first = _fixture_payload("figshare")
    result, dispatch, _group = await _invoke(
        "figshare",
        [_json_bytes(first)],
        max_pages=2,
        result_limit=2,
    )

    assert len(result.candidates) == 1
    assert len(dispatch.calls) == 1


@pytest.mark.parametrize(("provider", "field"), _PROVIDER_URL_FIELDS)
@pytest.mark.asyncio
async def test_unsafe_result_urls_are_suppressed_not_cleaned_or_retained(
    provider: str,
    field: str,
) -> None:
    unsafe_urls = (
        "file:///private/paper.pdf",
        "javascript:alert(1)",
        "https://example.org/paper?token=fixture-secret",
        "https://example.org/paper#private",
        "https://user:password@example.org/paper",
        "https://example.org/download/token=fixture-secret",
        "http://localhost/paper",
        "http://127.0.0.1/paper",
        "http://[::1]/paper",
        "http://10.0.0.1/paper",
        "http://169.254.169.254/latest/meta-data",
        "http://192.168.1.1/paper",
        "http://127.1/paper",
        "http://2130706433/paper",
        "http://0177.0.0.1/paper",
        "http://0x7f000001/paper",
        "http://0x7f.1/paper",
        "http://0x7f.0.0.1/paper",
        "http://127.0x0.0.1/paper",
        "http://0177.0x0.1/paper",
        "http://%31%32%37.0.0.1/paper",
        "http://0/paper",
        "http://127.0.0.1\\example.com/paper",
        "http://127。0。0。1/paper",
        "http://127．0．0．1/paper",
        "http://127｡0｡0｡1/paper",
        "http://１２７.０.０.１/paper",
        "http://０x７f.０.０.１/paper",
        "http://exa mple.com/paper",
        "http://example.com:/paper",
        "http://example.com:0/paper",
        "http://example..com/paper",
        "http://-example.com/paper",
        "http://example-.com/paper",
        "http://xn--/paper",
        "https://exa\nmple.org/paper",
        "https://example.org/pa%0d%0aper",
        "https://example.org/download/%2574oken=fixture-secret",
        "https://example.org/download/%253Btoken=fixture-secret",
        "https://example.org/download%252Ftoken=fixture-secret",
        "https://example.org/%253Ftoken%253Dfixture-secret",
        "https://example.org/%25%37%34oken%253Dfixture-secret",
    )
    for unsafe_url in unsafe_urls:
        payload = _fixture_payload(provider)
        _set_result_url(provider, payload, field, unsafe_url)
        result, dispatch, _group = await _invoke(provider, [_json_bytes(payload)])
        normalized = dict(result.candidates[0].record)
        assert normalized[field] is None
        assert unsafe_url not in repr(normalized)
        assert len(dispatch.calls) == 1


@pytest.mark.parametrize(("provider", "field"), _PROVIDER_URL_FIELDS)
@pytest.mark.asyncio
async def test_safe_result_urls_use_existing_canonicalizer(provider: str, field: str) -> None:
    raw_url = "HTTPS://EXAMPLE.ORG:443/research/%7Epaper"
    payload = _fixture_payload(provider)
    _set_result_url(provider, payload, field, raw_url)

    assert has_unsafe_url_material(raw_url) is False
    result, _dispatch, _group = await _invoke(provider, [_json_bytes(payload)])
    assert result.candidates[0].record[field] == canonicalize_url(raw_url)


@pytest.mark.parametrize(
    "raw_url",
    (
        "https://8.8.8.8/research/paper",
        "https://123.example.org/research/paper",
        "https://xn--bcher-kva.example/research/paper",
    ),
)
@pytest.mark.asyncio
async def test_safe_ascii_dns_and_global_ipv4_result_urls_remain_supported(raw_url: str) -> None:
    payload = _fixture_payload("semantic_scholar")
    _set_result_url("semantic_scholar", payload, "url", raw_url)

    result, _dispatch, _group = await _invoke(
        "semantic_scholar",
        [_json_bytes(payload)],
    )

    assert result.candidates[0].record["url"] == canonicalize_url(raw_url)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_url",
    ("http://[::1", "https://[2606:4700:4700::1111]/paper"),
)
async def test_malformed_or_ipv6_literal_result_url_is_safely_suppressed(raw_url: str) -> None:
    payload = _fixture_payload("semantic_scholar")
    _set_result_url("semantic_scholar", payload, "url", raw_url)

    result, dispatch, _group = await _invoke("semantic_scholar", [_json_bytes(payload)])

    assert result.candidates[0].record["url"] is None
    assert raw_url not in repr(result)
    assert len(dispatch.calls) == 1


@pytest.mark.parametrize(
    "first_href",
    (
        "https://arxiv.org/pdf/not-an-arxiv-id",
        "https://arxiv.org/pdf/9999.99999",
    ),
)
def test_arxiv_pdf_scanner_skips_unusable_candidate_and_uses_later_match(first_href: str) -> None:
    entry = ElementTree.fromstring(
        f"""
        <entry xmlns="http://www.w3.org/2005/Atom">
          <link rel="related" title="pdf" type="application/pdf" href="{first_href}" />
          <link rel="related" title="pdf" type="application/pdf" href="https://arxiv.org/pdf/2601.01234" />
        </entry>
        """
    )

    assert _gateway_adapters_module()._arxiv_pdf_url(entry, "2601.01234") == ("https://arxiv.org/pdf/2601.01234")


@pytest.mark.asyncio
async def test_crossref_skips_unsafe_pdf_link_and_uses_later_safe_candidate() -> None:
    payload = _fixture_payload("crossref")
    payload["message"]["items"][0]["link"] = [
        {
            "URL": "https://example.org/paper.pdf?token=fixture-secret",
            "content-type": "application/pdf",
        },
        {
            "URL": "https://example.org/second.pdf",
            "content-type": "application/pdf",
        },
    ]

    result, dispatch, _group = await _invoke("crossref", [_json_bytes(payload)])

    assert result.candidates[0].record["pdf_url"] == "https://example.org/second.pdf"
    assert "fixture-secret" not in repr(result)
    assert len(dispatch.calls) == 1


@pytest.mark.asyncio
async def test_zenodo_skips_unsafe_pdf_file_and_uses_later_safe_candidate() -> None:
    payload = _fixture_payload("zenodo")
    payload["hits"]["hits"][0]["files"].insert(
        0,
        {
            "key": "unsafe.pdf",
            "links": {"self": "https://example.org/unsafe.pdf?token=fixture-secret"},
        },
    )

    result, dispatch, _group = await _invoke("zenodo", [_json_bytes(payload)])

    assert result.candidates[0].record["pdf_url"] == (
        "https://zenodo.org/api/records/1001/files/shared-discovery.pdf/content"
    )
    assert "fixture-secret" not in repr(result)
    assert len(dispatch.calls) == 1


@pytest.mark.parametrize("tldr_shape", ("missing", "null"))
@pytest.mark.asyncio
async def test_semantic_scholar_tldr_missing_or_null_falls_back_to_abstract(
    tldr_shape: str,
) -> None:
    payload = _fixture_payload("semantic_scholar")
    record = payload["data"][0]
    if tldr_shape == "missing":
        record.pop("tldr")
    else:
        record["tldr"] = None

    result, _dispatch, _group = await _invoke("semantic_scholar", [_json_bytes(payload)])
    normalized = result.candidates[0].record
    assert normalized["abstract"] == "A sanitized abstract for discovery adapter testing."
    assert normalized["snippet"] == "A sanitized abstract for discovery adapter testing."


@pytest.mark.asyncio
async def test_repeated_identical_candidates_across_pages_dedupe_deterministically() -> None:
    record = _semantic_record(1, doi="10.5555/repeated")
    first = _semantic_page([record], total=2, next_cursor=1)
    second = _semantic_page([record], total=2, offset=1)

    result, dispatch, _group = await _invoke(
        "semantic_scholar",
        [first, second],
        max_pages=2,
        result_limit=2,
    )

    assert len(result.candidates) == 1
    assert len(dispatch.calls) == 2


@pytest.mark.asyncio
async def test_conflicting_record_for_same_candidate_id_fails_atomically() -> None:
    first_record = _semantic_record(1, doi="10.5555/conflict")
    conflicting = {**first_record, "title": "Conflicting title"}
    first = _semantic_page([first_record], total=2, next_cursor=1)
    second = _semantic_page([conflicting], total=2, offset=1)

    with pytest.raises(Exception) as caught:
        await _invoke(
            "semantic_scholar",
            [first, second],
            max_pages=2,
            result_limit=2,
        )

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_same_raw_provider_id_remains_scoped_by_provider_without_scholarly_id() -> None:
    semantic = _fixture_payload("semantic_scholar")
    semantic_record = semantic["data"][0]
    semantic_record["paperId"] = "shared-provider-id"
    semantic_record["externalIds"] = {}
    osf = _fixture_payload("osf")
    osf_record = osf["data"][0]
    osf_record["id"] = "shared-provider-id"
    osf_record["attributes"].pop("doi", None)

    semantic_result, _dispatch, _group = await _invoke(
        "semantic_scholar",
        [_json_bytes(semantic)],
    )
    osf_result, _dispatch, _group = await _invoke("osf", [_json_bytes(osf)])

    assert semantic_result.candidates[0].candidate_id != osf_result.candidates[0].candidate_id
    assert build_fingerprint(dict(semantic_result.candidates[0].record)).startswith("provider:semantic_scholar:")
    assert build_fingerprint(dict(osf_result.candidates[0].record)).startswith("provider:osf:")


@pytest.mark.asyncio
async def test_conflicting_zenodo_doi_locations_fail_atomically() -> None:
    payload = _fixture_payload("zenodo")
    payload["hits"]["hits"][0]["doi"] = "10.5555/top-level"
    payload["hits"]["hits"][0]["metadata"]["doi"] = "10.5555/metadata"

    with pytest.raises(Exception) as caught:
        await _invoke("zenodo", [_json_bytes(payload)])

    _assert_typed_error(caught.value, "provider_payload_invalid")


@pytest.mark.asyncio
async def test_parse_clock_is_checked_before_after_decode_and_during_records() -> None:
    clock = _CountingClock()
    body = _semantic_page([_semantic_record(index) for index in range(100)])
    result, _dispatch, _group = await _invoke(
        "semantic_scholar",
        [body],
        result_limit=100,
        monotonic_clock=clock,
    )

    assert len(result.candidates) == 100
    assert clock.calls >= 5


@pytest.mark.asyncio
async def test_parse_deadline_can_expire_cooperatively_after_decode() -> None:
    clock = _CountingClock(step=0.3)
    with pytest.raises(Exception) as caught:
        await _invoke(
            "semantic_scholar",
            [_fixture_bytes("semantic_scholar", "success")],
            monotonic_clock=clock,
        )

    _assert_typed_error(caught.value, "provider_parse_deadline_exceeded")
    assert clock.calls >= 3


@pytest.mark.asyncio
async def test_parse_deadline_is_checked_during_wide_unknown_tree_traversal() -> None:
    clock = _CountingClock(step=0.01)
    body = _json_bytes({"data": [], "total": 0, "offset": 0, "unknown": list(range(20_000))})
    with pytest.raises(Exception) as caught:
        await _invoke("semantic_scholar", [body], monotonic_clock=clock)

    _assert_typed_error(caught.value, "provider_parse_deadline_exceeded")
    assert clock.calls >= 51


@pytest.mark.asyncio
async def test_parse_deadline_is_checked_during_wide_consumed_arrays() -> None:
    clock = _CountingClock(step=0.008)
    payload = _fixture_payload("crossref")
    payload["message"]["items"][0]["author"] = [{} for _ in range(10_000)]

    with pytest.raises(Exception) as caught:
        await _invoke("crossref", [_json_bytes(payload)], monotonic_clock=clock)

    _assert_typed_error(caught.value, "provider_parse_deadline_exceeded")
    assert clock.calls >= 63


@pytest.mark.asyncio
async def test_adapter_propagates_cancellation_without_typed_rewrite() -> None:
    _registry, plan = _plan_for("semantic_scholar")
    group = plan.dispatch_groups[0]
    cancelled = asyncio.CancelledError("fixture-cancelled")
    dispatch = _RecordingDispatch([cancelled])

    with pytest.raises(asyncio.CancelledError) as caught:
        await _adapter_for("semantic_scholar")(group, dispatch)

    assert caught.value is cancelled


@pytest.mark.asyncio
async def test_adapter_propagates_existing_gateway_timeout_classification() -> None:
    _registry, plan = _plan_for("semantic_scholar")
    group = plan.dispatch_groups[0]
    timed_out = DiscoveryExecutionError("gateway_timed_out")
    dispatch = _RecordingDispatch([timed_out])

    with pytest.raises(DiscoveryExecutionError) as caught:
        await _adapter_for("semantic_scholar")(group, dispatch)

    assert caught.value is timed_out


@pytest.mark.parametrize(
    ("second_body", "expected_code"),
    (
        (b"{", "provider_payload_invalid"),
        pytest.param(
            b'{"data":[],"unknown":' + (b"9" * 10_000) + b"}",
            "provider_parse_limit_exceeded",
            id="oversized-token",
        ),
    ),
)
@pytest.mark.asyncio
async def test_later_page_parse_failure_commits_zero_page_one_candidates(
    second_body: bytes,
    expected_code: str,
) -> None:
    first = _semantic_page(
        [_semantic_record(1, doi="10.5555/atomic")],
        total=2,
        next_cursor=1,
    )
    result, gateway_calls = await _execute_adapter_plan(
        "semantic_scholar",
        [
            (first, 200, "application/json", None),
            (second_body, 200, "application/json", None),
        ],
    )

    assert len(gateway_calls) == 2
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == expected_code
    _assert_two_completed_physical_hops(result)


@pytest.mark.asyncio
async def test_later_page_parse_deadline_is_logical_timeout_and_commits_zero() -> None:
    first = _semantic_page(
        [_semantic_record(1, doi="10.5555/deadline")],
        total=2,
        next_cursor=1,
    )
    second = _semantic_page(
        [_semantic_record(2, doi="10.5555/deadline-two")],
        total=2,
        offset=1,
    )
    clock = _CountingClock()

    def advance_clock_on_page_two(call_number: int) -> None:
        if call_number == 2:
            clock.step = 0.3

    result, gateway_calls = await _execute_adapter_plan(
        "semantic_scholar",
        [
            (first, 200, "application/json", None),
            (second, 200, "application/json", None),
        ],
        monotonic_clock=clock,
        before_gateway_response=advance_clock_on_page_two,
    )

    assert len(gateway_calls) == 2
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.TIMED_OUT
    assert result.logical_outcomes[0].code == "provider_parse_deadline_exceeded"
    _assert_two_completed_physical_hops(result)


@pytest.mark.asyncio
async def test_later_page_429_commits_zero_and_preserves_safe_retry_after() -> None:
    first = _semantic_page(
        [_semantic_record(1, doi="10.5555/rate-limited")],
        total=2,
        next_cursor=1,
    )
    result, gateway_calls = await _execute_adapter_plan(
        "semantic_scholar",
        [
            (first, 200, "application/json", None),
            (b'{"error":"fixture-secret"}', 429, "text/plain", "120"),
        ],
    )

    assert len(gateway_calls) == 2
    assert result.candidates == ()
    assert result.logical_outcomes[0].state is LogicalOutcomeState.FAILED
    assert result.logical_outcomes[0].code == "provider_rate_limited"
    assert result.logical_outcomes[0].retry_after == "120"
    assert "fixture-secret" not in repr(result)
    _assert_two_completed_physical_hops(result)


@pytest.mark.asyncio
async def test_later_page_cancellation_propagates_without_partial_result() -> None:
    first = _semantic_page(
        [_semantic_record(1, doi="10.5555/cancelled")],
        total=2,
        next_cursor=1,
    )
    cancelled = asyncio.CancelledError("fixture-page-two-cancelled")

    with pytest.raises(asyncio.CancelledError) as caught:
        await _execute_adapter_plan(
            "semantic_scholar",
            [
                (first, 200, "application/json", None),
                cancelled,
            ],
        )

    assert caught.value is cancelled


def test_gateway_adapter_ast_has_no_transport_legacy_wrapper_or_sleep_seam() -> None:
    module_path = Path(__file__).parents[2] / "app" / "core" / "Research" / "discovery" / "gateway_adapters.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    imported_modules = set()
    imported_names = set()
    imported_from = set()
    called_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            imported_modules.add(module_name)
            imported_names.update(alias.name for alias in node.names)
            imported_from.update((module_name, alias.name) for alias in node.names)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)

    banned_prefixes = (
        "httpx",
        "requests",
        "aiohttp",
        "http.client",
        "socket",
        "urllib3",
        "urllib.request",
        "tldw_Server_API.app.core.http_client",
        "tldw_Server_API.app.core.Security.http_hop",
        "tldw_Server_API.app.core.Third_Party",
    )
    assert not {module for module in imported_modules if module.startswith(banned_prefixes)}
    assert ("urllib", "request") not in imported_from
    assert not imported_names.intersection({"afetch", "fetch", "fetch_json", "http_hop", "request_http_hop", "urlopen"})
    assert not called_names.intersection(
        {
            "HTTPConnection",
            "HTTPSConnection",
            "PoolManager",
            "afetch",
            "create_connection",
            "fetch",
            "fetch_json",
            "request_http_hop",
            "sleep",
            "socket",
            "urlopen",
        }
    )


@pytest.mark.parametrize("provider", _PROVIDER_NAMES)
@pytest.mark.asyncio
async def test_runtime_egress_tripwires_and_inert_result_urls_receive_zero_requests(
    provider: str,
    monkeypatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("adapter attempted direct egress")

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Security import http_hop
    from tldw_Server_API.app.core.Third_Party import Semantic_Scholar

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(http.client, "HTTPConnection", forbidden)
    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(http_client, "fetch", forbidden)
    monkeypatch.setattr(http_client, "fetch_json", forbidden)
    monkeypatch.setattr(http_hop, "request_http_hop", forbidden)
    monkeypatch.setattr(Semantic_Scholar, "search_papers_semantic_scholar", forbidden)

    result, dispatch, _group = await _invoke(
        provider,
        [_fixture_bytes(provider, "success")],
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].record["url"]
    if provider in {"semantic_scholar", "crossref", "zenodo"}:
        assert result.candidates[0].record["pdf_url"]
    else:
        assert result.candidates[0].record["pdf_url"] is None
    assert len(dispatch.calls) == 1
