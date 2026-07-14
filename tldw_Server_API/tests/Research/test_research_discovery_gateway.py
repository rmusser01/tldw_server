from __future__ import annotations

import asyncio
import inspect
from dataclasses import asdict, replace

import pytest

from tldw_Server_API.app.core.Research.discovery.contracts import (
    AccessRoute,
    CredentialRequirement,
    DeferredNumericCSVQueryBinding,
    DispatchIntent,
    ExactOrigin,
    JSONBodyPair,
    OperationKind,
    QueryMode,
    QueryPair,
    RouteKind,
    RouteLimits,
    RoutePolicy,
    SourceConstraint,
    canonical_policy_digest,
)
from tldw_Server_API.app.core.Research.discovery.gateway import (
    DiscoveryGatewayError,
    DiscoveryGatewayResponse,
    dispatch_once,
    reconstruct_redirect_intent,
)
from tldw_Server_API.app.core.Research.discovery.registry import foundation_registry
from tldw_Server_API.app.core.Security.http_hop import (
    HTTPHopError,
    HTTPHopLimits,
    HTTPHopResponse,
    NormalizedHTTPHopRequest,
    request_http_hop,
)


def _route_and_intent() -> tuple[AccessRoute, DispatchIntent]:
    limits = RouteLimits(
        max_pages=1,
        max_redirects=1,
        max_retries=0,
        timeout_ms=250,
        max_response_bytes=64,
        max_results=5,
    )
    policy = RoutePolicy(
        policy_version="discovery-route-policy-v1",
        origin=ExactOrigin(scheme="https", host="api.example.test", port=443),
        methods=("GET",),
        paths=("/search",),
        allowed_query_keys=("q", "page"),
        limits=limits,
        pagination_query_key="page",
    )
    route = AccessRoute(
        route_id="example.search",
        backend_id="example",
        adapter_id="example.search",
        route_kind=RouteKind.DIRECT,
        query_modes=(QueryMode.GENERAL_FREE_TEXT,),
        source_constraint=SourceConstraint.NATIVE_CORPUS,
        attribution_basis="native corpus",
        credential_requirement=CredentialRequirement.NONE,
        fallback_order=0,
        max_physical_dispatches=2,
        adapter_version="v1",
        policy=policy,
    )
    intent = DispatchIntent(
        route_id=route.route_id,
        policy_digest=policy.policy_digest,
        operation_kind=OperationKind.SEARCH,
        method="GET",
        path="/search",
        query_pairs=(
            QueryPair(name="q", value="quantum mechanics"),
            QueryPair(name="page", value="1"),
        ),
        limits=limits,
    )
    return route, intent


def _post_route_and_intent() -> tuple[AccessRoute, DispatchIntent]:
    route, intent = _route_and_intent()
    limits = replace(intent.limits, max_request_body_bytes=128)
    policy = replace(
        route.policy,
        methods=("POST",),
        limits=limits,
        allowed_json_body_keys=("search_for", "order_direction"),
        policy_digest="",
    )
    return replace(route, policy=policy), replace(
        intent,
        policy_digest=policy.policy_digest,
        method="POST",
        limits=limits,
        json_body_pairs=(
            JSONBodyPair("order_direction", "desc"),
            JSONBodyPair("search_for", "quantum mechanics"),
        ),
    )


def _hop_response(
    *,
    status_code: int = 200,
    headers: tuple[tuple[str, str], ...] = (("content-type", "application/json"),),
    body: bytes = b'{"ok":true}',
    resolved_ips: tuple[str, ...] = ("93.184.216.34",),
    connected_ip: str = "93.184.216.34",
    response_header_bytes: int = 512,
    wire_bytes: int | None = None,
) -> HTTPHopResponse:
    return HTTPHopResponse(
        status_code=status_code,
        headers=headers,
        body=body,
        resolved_ips=resolved_ips,
        connected_ip=connected_ip,
        response_header_bytes=response_header_bytes,
        wire_bytes=len(body) if wire_bytes is None else wire_bytes,
    )


@pytest.mark.asyncio
async def test_dispatch_once_emits_one_credential_free_normalized_hop() -> None:
    route, intent = _route_and_intent()
    calls: list[NormalizedHTTPHopRequest] = []

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        calls.append(request)
        return _hop_response()

    response = await dispatch_once(
        route,
        intent,
        is_policy_active=lambda route_id, digest: (route_id == route.route_id and digest == route.policy.policy_digest),
        one_hop=one_hop,
    )

    assert calls == [
        NormalizedHTTPHopRequest(
            scheme="https",
            host="api.example.test",
            port=443,
            method="GET",
            target="/search?q=quantum%20mechanics&page=1",
            headers=(),
            body=b"",
            limits=HTTPHopLimits(
                dns_timeout_seconds=0.25,
                connect_timeout_seconds=0.25,
                read_timeout_seconds=0.25,
                write_timeout_seconds=0.25,
                total_timeout_seconds=0.25,
                max_wire_bytes=64,
                max_decompressed_bytes=64,
                max_parser_input_bytes=64,
                max_request_body_bytes=16 * 1024,
            ),
        )
    ]
    assert isinstance(response, DiscoveryGatewayResponse)
    assert response.status_code == 200
    assert response.body == b'{"ok":true}'


@pytest.mark.asyncio
async def test_post_json_body_is_minified_sorted_bounded_and_explicit() -> None:
    route, intent = _post_route_and_intent()
    calls: list[NormalizedHTTPHopRequest] = []

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        calls.append(request)
        return _hop_response()

    response = await dispatch_once(
        route,
        intent,
        is_policy_active=lambda _route_id, _digest: True,
        one_hop=one_hop,
    )

    assert len(calls) == 1
    assert calls[0].method == "POST"
    assert calls[0].target == "/search?q=quantum%20mechanics&page=1"
    assert calls[0].headers == (("content-type", "application/json"),)
    assert calls[0].body == b'{"order_direction":"desc","search_for":"quantum mechanics"}'
    assert calls[0].limits.max_request_body_bytes == 128
    assert response.trace.query_keys == ("q", "page")
    assert "quantum mechanics" not in repr(asdict(response.trace))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ("unresolved-binding", "oversized", "unicode", "undeclared-key", "get-with-body"),
)
async def test_invalid_json_request_shape_rejects_before_hop(case: str) -> None:
    route, intent = _post_route_and_intent()
    if case == "unresolved-binding":
        policy = replace(route.policy, allowed_query_keys=("q", "page", "id"), policy_digest="")
        route = replace(route, policy=policy)
        intent = replace(
            intent,
            policy_digest=policy.policy_digest,
            query_bindings=(DeferredNumericCSVQueryBinding("result_ids", "id", 5, 16),),
        )
    elif case == "oversized":
        limits = replace(intent.limits, max_request_body_bytes=16)
        policy = replace(route.policy, limits=limits, policy_digest="")
        route = replace(route, policy=policy)
        intent = replace(intent, policy_digest=policy.policy_digest, limits=limits)
    elif case == "unicode":
        intent = replace(intent, json_body_pairs=(JSONBodyPair("search_for", "\ud800secret-body"),))
    elif case == "undeclared-key":
        intent = replace(intent, json_body_pairs=(JSONBodyPair("token", "secret-body"),))
    else:
        policy = replace(route.policy, methods=("GET", "POST"), policy_digest="")
        route = replace(route, policy=policy)
        intent = replace(intent, policy_digest=policy.policy_digest, method="GET")
    calls = 0

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        nonlocal calls
        calls += 1
        return _hop_response()

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "request_rejected"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "secret-body" not in repr(caught.value)
    assert calls == 0


@pytest.mark.asyncio
async def test_policy_callback_body_mutation_rejects_before_hop() -> None:
    route, intent = _post_route_and_intent()
    calls = 0

    def mutate_body(_route_id: str, _digest: str) -> bool:
        object.__setattr__(intent, "json_body_pairs", (JSONBodyPair("search_for", "mutated-secret"),))
        return True

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        nonlocal calls
        calls += 1
        return _hop_response()

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(route, intent, is_policy_active=mutate_body, one_hop=one_hop)

    assert caught.value.code == "request_rejected"
    assert calls == 0


@pytest.mark.asyncio
async def test_hostile_json_scalar_subclass_rejects_before_hop() -> None:
    class HostileString(str):
        pass

    route, intent = _post_route_and_intent()
    intent = replace(
        intent,
        json_body_pairs=(JSONBodyPair("search_for", HostileString("secret-body")),),
    )
    calls = 0

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        nonlocal calls
        calls += 1
        return _hop_response()

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "request_rejected"
    assert "secret-body" not in repr(caught.value)
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("channel", "field", "value"),
    (
        ("query", "value", "secret\x00query"),
        ("json", "value", "secret\x00body"),
        ("query", "name", "bad?query"),
        ("json", "name", "bad?body"),
    ),
)
async def test_mutated_request_pairs_are_revalidated_before_hop(
    channel: str,
    field: str,
    value: str,
) -> None:
    route, intent = _post_route_and_intent() if channel == "json" else _route_and_intent()
    pair = intent.json_body_pairs[-1] if channel == "json" else intent.query_pairs[0]
    object.__setattr__(pair, field, value)
    if field == "name":
        policy_field = "allowed_json_body_keys" if channel == "json" else "allowed_query_keys"
        policy_values = (value, "order_direction") if channel == "json" else (value, "page")
        object.__setattr__(route.policy, policy_field, policy_values)
        digest = canonical_policy_digest(route.policy)
        object.__setattr__(route.policy, "policy_digest", digest)
        object.__setattr__(intent, "policy_digest", digest)
    calls = 0

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        nonlocal calls
        calls += 1
        return _hop_response()

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "request_rejected"
    assert "secret" not in repr(caught.value)
    assert calls == 0


@pytest.mark.asyncio
async def test_unresolved_pubmed_summary_binding_rejects_before_hop() -> None:
    route = foundation_registry().get_route("pubmed_ncbi_eutils_pubmed_direct")
    intent = DispatchIntent(
        route_id=route.route_id,
        policy_digest=route.policy.policy_digest,
        operation_kind=OperationKind.CONDITIONAL_SUMMARY,
        method="GET",
        path="/entrez/eutils/esummary.fcgi",
        query_pairs=(QueryPair("db", "pubmed"), QueryPair("retmode", "json")),
        limits=route.policy.limits,
        query_bindings=(
            DeferredNumericCSVQueryBinding(
                binding_id="pubmed_esearch_ids",
                query_name="id",
                max_items=25,
                max_item_chars=16,
            ),
        ),
    )
    calls = 0

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        nonlocal calls
        calls += 1
        return _hop_response()

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "request_rejected"
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mismatch",
    ("route_id", "policy_digest", "method", "path", "query_key", "limits", "inactive"),
)
async def test_policy_mismatches_reject_before_the_hop(mismatch: str) -> None:
    route, intent = _route_and_intent()
    if mismatch == "route_id":
        intent = replace(intent, route_id="other.search")
    elif mismatch == "policy_digest":
        object.__setattr__(route.policy, "policy_digest", "0" * 64)
        intent = replace(intent, policy_digest="0" * 64)
    elif mismatch == "method":
        intent = replace(intent, method="POST")
    elif mismatch == "path":
        intent = replace(intent, path="/other")
    elif mismatch == "query_key":
        intent = replace(intent, query_pairs=(QueryPair(name="token", value="secret"),))
    elif mismatch == "limits":
        intent = replace(intent, limits=replace(intent.limits, max_results=6))

    calls = 0

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        nonlocal calls
        calls += 1
        return _hop_response()

    with pytest.raises(DiscoveryGatewayError):
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: mismatch != "inactive",
            one_hop=one_hop,
        )

    assert calls == 0


@pytest.mark.asyncio
async def test_redirect_is_returned_as_typed_data_and_not_followed() -> None:
    route, intent = _route_and_intent()
    calls = 0

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        nonlocal calls
        calls += 1
        return _hop_response(
            status_code=302,
            headers=(("location", "https://api.example.test/search?page=2"),),
            body=b"",
        )

    response = await dispatch_once(
        route,
        intent,
        is_policy_active=lambda _route_id, _digest: True,
        one_hop=one_hop,
    )

    assert isinstance(response, DiscoveryGatewayResponse)
    assert response.status_code == 302
    assert response.headers == ()
    assert response.redirect_location == "https://api.example.test/search?page=2"
    assert response.retry_after is None
    assert calls == 1


@pytest.mark.asyncio
async def test_http_hop_error_maps_to_stable_bounded_safe_error() -> None:
    route, intent = _route_and_intent()
    unsafe = HTTPHopError("total_timeout", retryable=True)
    unsafe.args = ("secret?query=quantum mechanics body=/private/file",)

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        raise unsafe

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    error = caught.value
    assert error.code == "hop_failed"
    assert error.retryable is True
    assert str(error) == "Discovery gateway hop failed"
    assert len(str(error)) <= 64
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.asyncio
async def test_cancellation_propagates() -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )


@pytest.mark.asyncio
async def test_response_headers_and_trace_contain_only_safe_bounded_evidence() -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        return _hop_response(
            headers=(
                ("content-type", "application/json"),
                ("set-cookie", "session=secret"),
                ("authorization", "Bearer secret"),
                ("x-unsafe", "quantum mechanics /private/file"),
            ),
            body=b'{"secret":"body-only"}',
        )

    response = await dispatch_once(
        route,
        intent,
        is_policy_active=lambda _route_id, _digest: True,
        one_hop=one_hop,
    )

    assert response.headers == (("content-type", "application/json"),)
    assert response.trace.route_id == route.route_id
    assert response.trace.policy_digest == route.policy.policy_digest
    assert response.trace.requested_host == "api.example.test"
    assert response.trace.tls_server_name == "api.example.test"
    assert response.trace.query_keys == ("q", "page")
    assert response.trace.resolved_ips == ("93.184.216.34",)
    assert response.trace.connected_ip == "93.184.216.34"
    assert response.trace.response_header_bytes == 512
    assert response.trace.wire_bytes == len(response.body)
    assert response.trace.decoded_bytes == len(response.body)
    assert response.trace.max_response_bytes == 64
    assert response.trace.http_limits.total_timeout_seconds == 0.25
    assert response.trace.http_limits.max_parser_input_bytes == 64
    trace_text = repr(asdict(response.trace))
    for unsafe_text in ("secret", "quantum mechanics", "body-only", "/private/file"):
        assert unsafe_text not in trace_text


@pytest.mark.asyncio
async def test_injected_hop_cannot_return_body_over_route_limit() -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        return _hop_response(body=b"x" * 65)

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "invalid_hop_response"
    assert str(caught.value) == "Discovery gateway hop response rejected"


@pytest.mark.asyncio
async def test_malformed_hop_headers_map_to_safe_invalid_response() -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        return _hop_response(headers=(("content-type",),))

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "invalid_hop_response"
    assert str(caught.value) == "Discovery gateway hop response rejected"


def test_dispatch_once_defaults_to_the_public_task_12971_seam() -> None:
    parameter = inspect.signature(dispatch_once).parameters["one_hop"]

    assert parameter.default is request_http_hop


@pytest.mark.asyncio
async def test_query_pairs_preserve_order_and_use_strict_percent_encoding() -> None:
    route, intent = _route_and_intent()
    intent = replace(
        intent,
        query_pairs=(
            QueryPair(name="page", value="2"),
            QueryPair(name="q", value="a/b+c &?"),
        ),
    )
    calls: list[NormalizedHTTPHopRequest] = []

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        calls.append(request)
        return _hop_response()

    await dispatch_once(
        route,
        intent,
        is_policy_active=lambda _route_id, _digest: True,
        one_hop=one_hop,
    )

    assert calls[0].target == "/search?page=2&q=a%2Fb%2Bc%20%26%3F"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query_pairs",
    (
        (QueryPair(name="q", value="first"), QueryPair(name="q", value="second")),
        (QueryPair(name="q", value="\ud800secret-query"),),
        (QueryPair(name="q", value="x" * 9_000),),
    ),
    ids=("duplicate-key", "lone-surrogate", "oversized-target"),
)
async def test_invalid_query_shape_is_sanitized_before_hop(
    query_pairs: tuple[QueryPair, ...],
) -> None:
    route, intent = _route_and_intent()
    intent = replace(intent, query_pairs=query_pairs)
    calls = 0

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        nonlocal calls
        calls += 1
        return _hop_response()

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "request_rejected"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "secret-query" not in repr(caught.value)
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("tampering", ("scheme", "host", "port", "credential", "route-id-type"))
async def test_origin_and_credential_tampering_reject_before_hop(tampering: str) -> None:
    route, intent = _route_and_intent()
    if tampering == "scheme":
        object.__setattr__(route.policy.origin, "scheme", "http")
    elif tampering == "host":
        object.__setattr__(route.policy.origin, "host", "attacker.example")
    elif tampering == "port":
        object.__setattr__(route.policy.origin, "port", 8443)
    elif tampering == "credential":
        object.__setattr__(route, "credential_requirement", CredentialRequirement.API_KEY)
    else:
        invalid_route_id = object()
        object.__setattr__(route, "route_id", invalid_route_id)
        object.__setattr__(intent, "route_id", invalid_route_id)
    calls = 0

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        nonlocal calls
        calls += 1
        return _hop_response()

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "request_rejected"
    assert calls == 0


@pytest.mark.asyncio
async def test_policy_callback_mutation_rejects_before_hop() -> None:
    route, intent = _route_and_intent()
    calls = 0

    def mutate_policy(_route_id: str, _digest: str) -> bool:
        object.__setattr__(intent, "path", "/other")
        return True

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        nonlocal calls
        calls += 1
        return _hop_response()

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=mutate_policy,
            one_hop=one_hop,
        )

    assert caught.value.code == "request_rejected"
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation_phase", ("initial", "policy-callback"))
async def test_hostile_scalar_mutation_is_safely_rejected_before_hop(mutation_phase: str) -> None:
    route, intent = _route_and_intent()
    calls = 0

    class HostileString(str):
        def isascii(self) -> bool:
            raise RuntimeError("provider-secret")

    def mutate_host() -> None:
        object.__setattr__(route.policy.origin, "host", HostileString("api.example.test"))

    if mutation_phase == "initial":
        mutate_host()

    def policy_check(_route_id: str, _digest: str) -> bool:
        if mutation_phase == "policy-callback":
            mutate_host()
        return True

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        nonlocal calls
        calls += 1
        return _hop_response()

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=policy_check,
            one_hop=one_hop,
        )

    assert caught.value.code == "request_rejected"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "provider-secret" not in repr(caught.value)
    assert calls == 0


@pytest.mark.asyncio
async def test_trace_uses_pre_dispatch_snapshot_when_hop_mutates_inputs() -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        object.__setattr__(intent, "path", "/mutated-during-hop")
        object.__setattr__(route.policy.origin, "host", "mutated.example")
        return _hop_response()

    response = await dispatch_once(
        route,
        intent,
        is_policy_active=lambda _route_id, _digest: True,
        one_hop=one_hop,
    )

    assert response.trace.path == "/search"
    assert response.trace.requested_host == "api.example.test"


@pytest.mark.asyncio
async def test_hop_cannot_expand_the_trusted_response_limits() -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        object.__setattr__(request.limits, "max_wire_bytes", 1_024)
        object.__setattr__(request.limits, "max_decompressed_bytes", 1_024)
        object.__setattr__(request.limits, "max_parser_input_bytes", 1_024)
        return _hop_response(body=b"x" * 65)

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "invalid_hop_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resolved_ips", "connected_ip"),
    (
        (("10.0.0.1",), "10.0.0.1"),
        (("224.0.0.1",), "224.0.0.1"),
        (("93.184.216.34", "93.184.216.34"), "93.184.216.34"),
        (("2001:0DB8::1",), "2001:0DB8::1"),
        (("::ffff:8.8.8.8",), "::ffff:8.8.8.8"),
        (("64:ff9b::808:808",), "64:ff9b::808:808"),
        (("2001::1",), "2001::1"),
        (("2002:0808:0808::1",), "2002:0808:0808::1"),
        (("93.184.216.34",), "8.8.8.8"),
    ),
    ids=(
        "private",
        "multicast",
        "duplicate",
        "noncanonical",
        "ipv4-mapped",
        "nat64",
        "teredo",
        "6to4",
        "connected-not-resolved",
    ),
)
async def test_impossible_ip_evidence_is_rejected(
    resolved_ips: tuple[str, ...],
    connected_ip: str,
) -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        return _hop_response(resolved_ips=resolved_ips, connected_ip=connected_ip)

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "invalid_hop_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    (
        _hop_response(status_code=101),
        _hop_response(headers=tuple((f"x-{index}", "v") for index in range(129))),
        _hop_response(headers=(("content-type", "\ud800unsafe"),)),
        _hop_response(headers=(("content-type", "application/json"),), response_header_bytes=0),
        _hop_response(headers=(("content-type", "application/json"),), response_header_bytes=1),
        _hop_response(headers=(("x-large", "x" * 70_000),), response_header_bytes=1),
    ),
    ids=(
        "interim-status",
        "header-count",
        "header-unicode",
        "zero-header-counter",
        "undersized-header-counter",
        "unbounded-header-value",
    ),
)
async def test_fabricated_hop_response_is_safely_rejected(response: HTTPHopResponse) -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        return response

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "invalid_hop_response"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "status_code", "body", "wire_bytes"),
    (
        ("GET", 200, b"x", 0),
        ("HEAD", 200, b"x", 1),
        ("HEAD", 200, b"", 1),
        ("GET", 204, b"x", 1),
        ("GET", 204, b"", 1),
        ("GET", 205, b"x", 1),
        ("GET", 205, b"", 1),
        ("GET", 205, b"", 5),
        ("GET", 304, b"x", 1),
        ("GET", 304, b"", 1),
    ),
    ids=(
        "body-without-wire",
        "head-body",
        "head-wire",
        "204-body",
        "204-wire",
        "205-body",
        "205-short-wire",
        "205-five-wire-without-chunked",
        "304-body",
        "304-wire",
    ),
)
async def test_impossible_body_and_wire_evidence_is_rejected(
    method: str,
    status_code: int,
    body: bytes,
    wire_bytes: int,
) -> None:
    route, intent = _route_and_intent()
    if method == "HEAD":
        policy = replace(route.policy, methods=("HEAD",), policy_digest="")
        route = replace(route, policy=policy)
        intent = replace(intent, method="HEAD", policy_digest=policy.policy_digest)

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        return _hop_response(
            status_code=status_code,
            body=body,
            wire_bytes=wire_bytes,
        )

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "invalid_hop_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wire_bytes", "headers"),
    (
        (0, ()),
        (5, (("transfer-encoding", "chunked"),)),
    ),
    ids=("zero-wire", "empty-chunk-terminator"),
)
async def test_valid_empty_205_wire_evidence_is_accepted(
    wire_bytes: int,
    headers: tuple[tuple[str, str], ...],
) -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        return _hop_response(
            status_code=205,
            headers=headers,
            body=b"",
            wire_bytes=wire_bytes,
        )

    response = await dispatch_once(
        route,
        intent,
        is_policy_active=lambda _route_id, _digest: True,
        one_hop=one_hop,
    )

    assert response.status_code == 205
    assert response.body == b""
    assert response.trace.wire_bytes == wire_bytes


@pytest.mark.asyncio
async def test_legitimate_latin1_unknown_header_and_informational_byte_count_are_accepted() -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        return _hop_response(
            headers=(("content-type", "application/json"), ("x-note", "caf\u00e9")),
            response_header_bytes=1_024,
        )

    response = await dispatch_once(
        route,
        intent,
        is_policy_active=lambda _route_id, _digest: True,
        one_hop=one_hop,
    )

    assert response.headers == (("content-type", "application/json"),)
    assert response.trace.response_header_bytes == 1_024


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    (
        (("location", "/next"), ("Location", "/next")),
        (("location", ""),),
        (("location", "/next\nunsafe"),),
        (("location", "x" * 8_193),),
        (("retry-after", "1"), ("Retry-After", "1")),
        (("retry-after", "\ud800"),),
        (("retry-after", "x" * 129),),
    ),
    ids=(
        "duplicate-location",
        "empty-location",
        "location-control",
        "location-too-long",
        "duplicate-retry-after",
        "retry-after-unicode",
        "retry-after-too-long",
    ),
)
async def test_invalid_redirect_and_retry_metadata_rejects_response(
    headers: tuple[tuple[str, str], ...],
) -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        return _hop_response(status_code=302, headers=headers, body=b"")

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "invalid_hop_response"


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_after", ("120", "Wed, 21 Oct 2015 07:28:00 GMT"))
async def test_retry_after_is_returned_as_sanitized_unparsed_typed_data(retry_after: str) -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        return _hop_response(
            status_code=503,
            headers=(("retry-after", f"  {retry_after}  "),),
            body=b"",
        )

    response = await dispatch_once(
        route,
        intent,
        is_policy_active=lambda _route_id, _digest: True,
        one_hop=one_hop,
    )

    assert response.retry_after == retry_after
    assert response.redirect_location is None
    assert retry_after not in repr(asdict(response.trace))


@pytest.mark.asyncio
async def test_missing_redirect_location_remains_typed_none() -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        return _hop_response(status_code=302, headers=(), body=b"")

    response = await dispatch_once(
        route,
        intent,
        is_policy_active=lambda _route_id, _digest: True,
        one_hop=one_hop,
    )

    assert response.redirect_location is None
    assert response.retry_after is None


@pytest.mark.asyncio
@pytest.mark.parametrize("retryable", ("provider-secret", object()))
async def test_non_boolean_hop_retryability_is_not_copied(retryable: object) -> None:
    route, intent = _route_and_intent()
    unsafe = HTTPHopError("transport_error", retryable=True)
    unsafe.retryable = retryable

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        raise unsafe

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.retryable is False
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "provider-secret" not in repr(caught.value)


@pytest.mark.asyncio
async def test_raising_retryable_accessor_maps_to_nonretryable_safe_error() -> None:
    route, intent = _route_and_intent()

    class HostileHTTPHopError(HTTPHopError):
        def __getattribute__(self, name: str) -> object:
            if name == "retryable":
                raise RuntimeError("provider-secret")
            return super().__getattribute__(name)

    unsafe = HostileHTTPHopError("transport_error", retryable=True)

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        raise unsafe

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "hop_failed"
    assert caught.value.retryable is False
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    ("location", "expected_pairs"),
    (
        (
            "/search?page=1&q=quantum%20mechanics",
            (QueryPair("page", "1"), QueryPair("q", "quantum mechanics")),
        ),
        (
            "https://api.example.test:443/search?q=quantum%20mechanics&page=1",
            (QueryPair("q", "quantum mechanics"), QueryPair("page", "1")),
        ),
        (
            "?%71=quantum+mechanics&page=%31",
            (QueryPair("q", "quantum mechanics"), QueryPair("page", "1")),
        ),
    ),
)
def test_redirect_intent_reconstruction_accepts_only_bound_same_origin_shape(
    location: str,
    expected_pairs: tuple[QueryPair, ...],
) -> None:
    route, intent = _route_and_intent()

    redirected = reconstruct_redirect_intent(route, intent, location)

    assert redirected is not None
    assert redirected.route_id == intent.route_id
    assert redirected.policy_digest == intent.policy_digest
    assert redirected.operation_kind is intent.operation_kind
    assert redirected.method == "GET"
    assert redirected.path == "/search"
    assert redirected.query_pairs == expected_pairs
    assert redirected.json_body_pairs == ()
    assert redirected.query_bindings == ()
    assert redirected.limits == intent.limits


def test_redirect_intent_reconstruction_preserves_a_planned_blank_value() -> None:
    route, intent = _route_and_intent()
    intent = replace(
        intent,
        query_pairs=(QueryPair("q", ""), QueryPair("page", "1")),
    )

    redirected = reconstruct_redirect_intent(route, intent, "/search?page=1&q=")

    assert redirected is not None
    assert redirected.query_pairs == (QueryPair("page", "1"), QueryPair("q", ""))


@pytest.mark.parametrize(
    "location",
    (
        "https://attacker.example/search?q=quantum%20mechanics&page=1",
        "http://api.example.test/search?q=quantum%20mechanics&page=1",
        "https://api.example.test:444/search?q=quantum%20mechanics&page=1",
        "https://api.example.test:0/search?q=quantum%20mechanics&page=1",
        "https://api.example.test:/search?q=quantum%20mechanics&page=1",
        "https://user@api.example.test/search?q=quantum%20mechanics&page=1",
        "//attacker.example/search?q=quantum%20mechanics&page=1",
        "/other?q=quantum%20mechanics&page=1",
        "/search?q=quantum%20mechanics&q=quantum%20mechanics&page=1",
        "/search?token=secret",
        "/search?q=%ZZ",
        "/search?q=quantum%20mechanics&page=1#fragment",
        "/search?q=quantum%20mechanics&page=1#",
        "/search?#",
        "/search?missing-equals",
        "/search?q=" + "x" * 9_000,
    ),
)
def test_redirect_intent_reconstruction_rejects_ambiguous_or_unbound_locations(location: str) -> None:
    route, intent = _route_and_intent()

    assert reconstruct_redirect_intent(route, intent, location) is None


@pytest.mark.parametrize(
    "location",
    (
        "/search?q=changed&page=1",
        "/search?q=quantum%20mechanics&page=2",
        "/search?page=1",
        "/search?q=&page=1",
        "/search?q=quantum%20mechanics&page=1&sort=relevance",
        "/summary?q=quantum%20mechanics&page=1",
    ),
)
def test_redirect_intent_reconstruction_rejects_semantic_request_mutation(location: str) -> None:
    route, intent = _route_and_intent()
    policy = replace(
        route.policy,
        paths=("/search", "/summary"),
        allowed_query_keys=("q", "page", "sort"),
        policy_digest="",
    )
    route = replace(route, policy=policy, max_physical_dispatches=3)
    intent = replace(intent, policy_digest=policy.policy_digest)

    assert reconstruct_redirect_intent(route, intent, location) is None


def test_redirect_intent_reconstruction_rejects_body_and_unresolved_binding_state() -> None:
    post_route, post_intent = _post_route_and_intent()
    route, intent = _route_and_intent()
    binding_policy = replace(route.policy, allowed_query_keys=("q", "page", "id"), policy_digest="")
    route = replace(route, policy=binding_policy)
    unresolved = replace(
        intent,
        policy_digest=binding_policy.policy_digest,
        query_bindings=(DeferredNumericCSVQueryBinding("result_ids", "id", 5, 16),),
    )

    assert reconstruct_redirect_intent(post_route, post_intent, "/search?page=2") is None
    assert reconstruct_redirect_intent(route, unresolved, "/search?page=2") is None
