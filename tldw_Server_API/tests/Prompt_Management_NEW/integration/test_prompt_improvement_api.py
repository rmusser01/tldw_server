"""Contract tests for provider-neutral prompt improvement.

The provider adapter is the only mocked boundary. Request parsing, target
isolation, prompt-improvement policy, route registration, and public error
mapping execute through the real FastAPI router.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    AuthContext,
    AuthPrincipal,
    get_auth_principal,
    get_request_user,
)
from tldw_Server_API.app.api.v1.API_Deps.llm_routing_deps import (
    get_request_routing_decision_store,
)
from tldw_Server_API.app.api.v1.endpoints import prompts
from tldw_Server_API.app.core.AuthNZ.llm_budget_guard import enforce_llm_budget
from tldw_Server_API.app.core.LLM_Calls.routing import InMemoryRoutingDecisionStore
from tldw_Server_API.app.core.Prompt_Management.prompt_improvement import (
    MAX_DRAFT_CHARS,
    MAX_FINDING_TEXT_CHARS,
    MAX_PROTECTED_TOKEN_CHARS,
    MAX_PROTECTED_TOKEN_KIND_CHARS,
    MAX_PROTECTED_TOKEN_OCCURRENCES,
    MAX_PROTECTED_TOKEN_TOTAL_CHARS,
    MAX_PROTECTED_TOKENS,
    META_PROMPT_VERSION,
    PROMPT_IMPROVEMENT_LIMITS,
)

pytestmark = pytest.mark.integration

_OPERATION_ID = "5d3e0a2c-fc12-4bc8-87c7-c6de71ff42d9"
_TARGET_DRAFT = "Be helpful to {{audience}}."


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operation_id": _OPERATION_ID,
        "target": "system",
        "text": _TARGET_DRAFT,
        "model_selection": {
            "selected_model": "test-model",
            "provider_hint": "openai",
        },
        "protected_tokens": [
            {
                "kind": "template_variable",
                "value": "{{audience}}",
                "occurrences": 1,
            }
        ],
    }
    payload.update(overrides)
    return payload


def _dispatch_module():
    return importlib.import_module(
        "tldw_Server_API.app.core.Prompt_Management.prompt_improvement_dispatch"
    )


def _prompt_route(app: FastAPI):
    for route in app.routes:
        if getattr(route, "path", None) == "/api/v1/prompts/improve":
            return route
    raise AssertionError("POST /api/v1/prompts/improve is not registered")


async def _allow_dependency() -> None:
    return None


def _test_user() -> SimpleNamespace:
    return SimpleNamespace(
        id=7,
        id_int=7,
        username="ordinary-chat-user",
        roles=["user"],
        permissions=[],
        is_admin=False,
    )


def _configured_providers() -> dict[str, Any]:
    return {"providers": [], "default_provider": "openai"}


@contextmanager
def _isolated_prompt_client(
    *,
    bypass_post_validation_gates: bool = True,
) -> Iterator[tuple[TestClient, FastAPI, Any]]:
    """Build an app whose prompt-improvement gates are explicitly controlled."""

    app = FastAPI()
    app.include_router(prompts.router, prefix="/api/v1/prompts")
    app.state.routing_decision_store = InMemoryRoutingDecisionStore()
    route = _prompt_route(app)

    for dependency in route.dependant.dependencies:
        call = dependency.call
        if call is get_request_user:
            app.dependency_overrides[call] = _test_user
        elif call is get_request_routing_decision_store:
            app.dependency_overrides[call] = lambda: app.state.routing_decision_store
        else:
            app.dependency_overrides[call] = _allow_dependency

    async def allow_post_validation_gates(**_kwargs) -> None:
        return None

    original_gate_runner = prompts._run_prompt_improvement_post_validation_gates
    if bypass_post_validation_gates:
        prompts._run_prompt_improvement_post_validation_gates = allow_post_validation_gates
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, app, route
    finally:
        if bypass_post_validation_gates:
            prompts._run_prompt_improvement_post_validation_gates = original_gate_runner


def _dependency_call(route: Any, predicate) -> Any:
    for dependency in route.dependant.dependencies:
        if predicate(dependency.call):
            return dependency.call
    raise AssertionError("required route dependency was not registered")


def _success_dispatch_result(
    *,
    provider: str = "openai",
    model: str = "test-model",
    candidate: str = "Be consistently helpful to {{audience}}.",
):
    dispatch = _dispatch_module()
    return dispatch.PromptImprovementDispatchResult(
        text=json.dumps(
            {
                "status": "improved",
                "improved_text": candidate,
                "findings": [
                    {
                        "category": "clarity",
                        "issue": "The behavior was vague.",
                        "change": "Made the behavior explicit.",
                    }
                ],
                "target": "system",
            }
        ),
        provider=provider,
        model=model,
        display_name=model,
    )


def test_improve_endpoint_returns_canonical_response_and_isolates_target(monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_dispatch(**kwargs):
        captured.update(kwargs)
        return _success_dispatch_result()

    monkeypatch.setattr(prompts, "dispatch_prompt_improvement", fake_dispatch)

    with _isolated_prompt_client() as (client, _app, _route):
        response = client.post("/api/v1/prompts/improve", json=_payload())

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json() == {
        "schema_version": 1,
        "operation_id": _OPERATION_ID,
        "status": "improved",
        "improved_text": "Be consistently helpful to {{audience}}.",
        "findings": [
            {
                "category": "clarity",
                "issue": "The behavior was vague.",
                "change": "Made the behavior explicit.",
            }
        ],
        "review_required": False,
        "warnings": [],
        "resolved_model": {
            "provider": "openai",
            "model": "test-model",
            "display_name": "test-model",
        },
        "meta_prompt_version": META_PROMPT_VERSION,
    }
    messages = captured["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert json.loads(messages[1]["content"]) == {
        "target": "system",
        "draft": _TARGET_DRAFT,
    }
    assert captured["selected_model"] == "test-model"
    assert captured["provider_hint"] == "openai"
    serialized = json.dumps(messages)
    for forbidden in (
        "COUNTERPART_CONTEXT",
        "CHAT_HISTORY_CONTEXT",
        "ATTACHMENT_CONTEXT",
        "RAG_CONTEXT",
        "TOOL_CONTEXT",
        "CHARACTER_CONTEXT",
        "SAVED_PROMPT_CONTEXT",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("selected_model", "provider_hint", "resolved_provider", "resolved_model"),
    [
        ("test-model", "openai", "openai", "test-model"),
        ("auto", None, "openrouter", "resolved/auto-model"),
    ],
)
def test_improve_endpoint_reports_concrete_and_actual_auto_route(
    monkeypatch,
    selected_model,
    provider_hint,
    resolved_provider,
    resolved_model,
):
    async def fake_dispatch(**kwargs):
        assert kwargs["selected_model"] == selected_model
        assert kwargs["provider_hint"] == provider_hint
        return _success_dispatch_result(
            provider=resolved_provider,
            model=resolved_model,
        )

    monkeypatch.setattr(prompts, "dispatch_prompt_improvement", fake_dispatch)
    payload = _payload(
        model_selection={
            "selected_model": selected_model,
            **({"provider_hint": provider_hint} if provider_hint is not None else {}),
        }
    )

    with _isolated_prompt_client() as (client, _app, _route):
        response = client.post("/api/v1/prompts/improve", json=payload)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["resolved_model"] == {
        "provider": resolved_provider,
        "model": resolved_model,
        "display_name": resolved_model,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        {"unknown_top_level": "must-fail"},
        {"model_selection": {"selected_model": "test-model", "unknown": True}},
        {
            "protected_tokens": [
                {
                    "kind": "template_variable",
                    "value": "{{audience}}",
                    "occurrences": 1,
                    "unknown": True,
                }
            ]
        },
        {"target": "assistant"},
        {"operation_id": "not-a-uuid"},
        {"text": ""},
        {"model_selection": {"selected_model": "x" * 501}},
        {
            "model_selection": {
                "selected_model": "test-model",
                "provider_hint": "p" * 101,
            }
        },
        {
            "protected_tokens": [
                {
                    "kind": "k" * (MAX_PROTECTED_TOKEN_KIND_CHARS + 1),
                    "value": "{{audience}}",
                    "occurrences": 1,
                }
            ]
        },
        {
            "protected_tokens": [
                {
                    "kind": "template_variable",
                    "value": "v" * (MAX_PROTECTED_TOKEN_CHARS + 1),
                    "occurrences": 1,
                }
            ]
        },
        {
            "protected_tokens": [
                {
                    "kind": "template_variable",
                    "value": "{{audience}}",
                    "occurrences": MAX_PROTECTED_TOKEN_OCCURRENCES + 1,
                }
            ]
        },
        {
            "protected_tokens": [
                {
                    "kind": "template_variable",
                    "value": "{{audience}}",
                    "occurrences": 1,
                }
            ]
            * (MAX_PROTECTED_TOKENS + 1)
        },
    ],
)
def test_request_validation_is_bounded_forbidden_and_sanitized(monkeypatch, mutation):
    dispatched = False

    async def fake_dispatch(**_kwargs):
        nonlocal dispatched
        dispatched = True
        return _success_dispatch_result()

    monkeypatch.setattr(prompts, "dispatch_prompt_improvement", fake_dispatch)
    payload = _payload(**mutation)

    with _isolated_prompt_client() as (client, _app, _route):
        response = client.post("/api/v1/prompts/improve", json=payload)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["code"] == "invalid_input"
    assert response.json()["retryable"] is False
    assert response.json()["request_id"]
    assert _TARGET_DRAFT not in response.text
    assert "must-fail" not in response.text
    assert dispatched is False


def test_oversized_draft_uses_stable_error_without_echo(monkeypatch):
    oversized = "PRIVATE_DRAFT_SENTINEL" + ("x" * MAX_DRAFT_CHARS)
    monkeypatch.setattr(
        prompts,
        "dispatch_prompt_improvement",
        AsyncMock(side_effect=AssertionError("provider must not be called")),
    )

    with _isolated_prompt_client() as (client, _app, _route):
        response = client.post(
            "/api/v1/prompts/improve",
            json=_payload(text=oversized, protected_tokens=[]),
        )

    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert response.json()["code"] == "draft_too_large"
    assert "PRIVATE_DRAFT_SENTINEL" not in response.text


@pytest.mark.parametrize(
    ("raw_body", "expected_code"),
    [
        ('{"text":"PRIVATE_MALFORMED_DRAFT",', "invalid_input"),
        ('{"padding":"' + ("x" * 64_001) + '"}', "draft_too_large"),
    ],
)
def test_raw_json_parsing_is_bounded_stable_and_never_echoes_input(
    monkeypatch,
    raw_body,
    expected_code,
):
    monkeypatch.setattr(
        prompts,
        "dispatch_prompt_improvement",
        AsyncMock(side_effect=AssertionError("provider must not be called")),
    )

    with _isolated_prompt_client() as (client, _app, _route):
        response = client.post(
            "/api/v1/prompts/improve",
            content=raw_body,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code in {
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_413_CONTENT_TOO_LARGE,
    }
    assert response.json()["code"] == expected_code
    assert "PRIVATE_MALFORMED_DRAFT" not in response.text
    assert "xxxxx" not in response.text


def _streaming_request(
    chunks: list[bytes],
    *,
    content_type: str | None = "application/json",
    content_length: str | None = None,
) -> tuple[Request, list[int]]:
    headers: list[tuple[bytes, bytes]] = []
    if content_type is not None:
        headers.append((b"content-type", content_type.encode("ascii")))
    if content_length is not None:
        headers.append((b"content-length", content_length.encode("ascii")))
    receive_calls: list[int] = []
    events = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    ]

    async def receive() -> dict[str, Any]:
        receive_calls.append(len(receive_calls))
        if events:
            return events.pop(0)
        return {"type": "http.disconnect"}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/prompts/improve",
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
            "scheme": "http",
        },
        receive,
    )
    return request, receive_calls


def _json_response_body(response: Any) -> dict[str, Any]:
    return json.loads(bytes(response.body))


@pytest.mark.asyncio
async def test_bounded_reader_aborts_chunked_body_before_consuming_tail():
    limit = PROMPT_IMPROVEMENT_LIMITS.max_request_bytes
    request, receive_calls = _streaming_request(
        [b"{" + (b"x" * (limit // 2)), b"x" * (limit // 2 + 1), b"PRIVATE_TAIL"],
    )

    response = await prompts._read_prompt_improvement_payload(
        request,
        request_id="req-chunked",
    )

    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert _json_response_body(response)["code"] == "draft_too_large"
    assert len(receive_calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", [None, "text/plain", "application/xml"])
async def test_bounded_reader_rejects_missing_or_unsupported_json_media_type(content_type):
    request, receive_calls = _streaming_request(
        [json.dumps(_payload()).encode("utf-8")],
        content_type=content_type,
    )

    response = await prompts._read_prompt_improvement_payload(
        request,
        request_id="req-media-type",
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert _json_response_body(response) == {
        "code": "invalid_input",
        "message": "The prompt improvement request is invalid.",
        "retryable": False,
        "request_id": "req-media-type",
    }
    assert receive_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("content_length", ["64001", "-1", "not-an-integer", "1,2"])
async def test_bounded_reader_preflights_content_length_without_reading(content_length):
    request, receive_calls = _streaming_request(
        [b"PRIVATE_BODY_MUST_NOT_BE_READ"],
        content_length=content_length,
    )

    response = await prompts._read_prompt_improvement_payload(
        request,
        request_id="req-content-length",
    )

    expected_code = "draft_too_large" if content_length == "64001" else "invalid_input"
    assert _json_response_body(response)["code"] == expected_code
    assert receive_calls == []


@pytest.mark.asyncio
async def test_bounded_reader_maps_deep_json_and_disconnect_to_stable_invalid_input():
    deep_request, _ = _streaming_request(
        [("[" * 1_500 + "0" + "]" * 1_500).encode("utf-8")]
    )
    deep_response = await prompts._read_prompt_improvement_payload(
        deep_request,
        request_id="req-deep",
    )

    disconnect_request, _ = _streaming_request([])
    disconnect_response = await prompts._read_prompt_improvement_payload(
        disconnect_request,
        request_id="req-disconnect",
    )

    assert _json_response_body(deep_response)["code"] == "invalid_input"
    assert _json_response_body(disconnect_response)["code"] == "invalid_input"


@contextmanager
def _ordered_gate_client(monkeypatch, events: list[str]):
    app = FastAPI()
    app.include_router(prompts.router, prefix="/api/v1/prompts")
    app.state.routing_decision_store = InMemoryRoutingDecisionStore()
    route = _prompt_route(app)

    async def allow() -> None:
        return None

    async def current_user() -> SimpleNamespace:
        return _test_user()

    async def principal(request: Request) -> AuthPrincipal:
        value = AuthPrincipal(
            kind="user",
            user_id=7,
            api_key_id=None,
            subject="ordinary-chat-user",
            token_type="jwt",
            jti="test-jti",
            roles=["user"],
            permissions=[],
            is_admin=False,
            org_ids=[],
            team_ids=[],
        )
        request.state.user_id = 7
        request.state.auth = AuthContext(principal=value)
        return value

    async def record(name: str) -> None:
        events.append(name)

    async def quota(*_args, **_kwargs) -> None:
        await record("quota")

    async def rbac(*_args, **_kwargs) -> None:
        await record("rbac")

    async def budget(*_args, **_kwargs) -> None:
        await record("budget")

    async def billing(*_args, **_kwargs) -> None:
        await record("billing")

    async def get_pool() -> object:
        return object()

    async def quota_dependency() -> None:
        await quota()

    async def rbac_dependency() -> None:
        await rbac()

    async def budget_dependency() -> None:
        await budget()

    async def billing_dependency() -> None:
        await billing()

    monkeypatch.setattr(prompts, "consume_deferred_token_quota", quota, raising=False)
    monkeypatch.setattr(prompts, "enforce_rbac_rate_limit", rbac, raising=False)
    monkeypatch.setattr(prompts, "enforce_llm_budget", budget)
    monkeypatch.setattr(prompts, "get_db_pool", get_pool)
    monkeypatch.setattr(
        prompts,
        "_PROMPT_IMPROVEMENT_BILLING_CHECK",
        billing,
        raising=False,
    )

    for dependency in route.dependant.dependencies:
        call = dependency.call
        if call is get_request_user:
            app.dependency_overrides[call] = current_user
        elif call is get_request_routing_decision_store:
            app.dependency_overrides[call] = lambda: app.state.routing_decision_store
        elif call is get_auth_principal:
            app.dependency_overrides[call] = principal
        elif getattr(call, "_tldw_endpoint_id", None) == "prompts.improve":
            if getattr(call, "_tldw_defer_count", False):
                app.dependency_overrides[call] = allow
            else:
                app.dependency_overrides[call] = quota_dependency
        elif getattr(call, "_tldw_rate_limit_resource", None) == "prompts.improve":
            app.dependency_overrides[call] = rbac_dependency
        elif call is enforce_llm_budget:
            app.dependency_overrides[call] = budget_dependency
        elif getattr(call, "__name__", "") == "_check_limit":
            app.dependency_overrides[call] = billing_dependency
        else:
            app.dependency_overrides[call] = allow

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"content": '{"text":"PRIVATE_MALFORMED",', "headers": {"Content-Type": "application/json"}},
        {
            "content": '{"padding":"' + ("x" * 64_001) + '"}',
            "headers": {"Content-Type": "application/json"},
        },
        {
            "json": _payload(
                protected_tokens=[
                    {"kind": "literal", "value": "missing", "occurrences": 1}
                ]
            )
        },
    ],
)
def test_invalid_requests_do_not_run_operation_gates_or_dispatch(
    monkeypatch,
    request_kwargs,
):
    events: list[str] = []

    async def dispatch(**_kwargs):
        events.append("dispatch")
        return _success_dispatch_result()

    monkeypatch.setattr(prompts, "dispatch_prompt_improvement", dispatch)
    with _ordered_gate_client(monkeypatch, events) as client:
        response = client.post("/api/v1/prompts/improve", **request_kwargs)

    assert response.status_code in {400, 413}
    assert events == []


def test_valid_request_runs_operation_gates_in_order_before_dispatch(monkeypatch):
    events: list[str] = []

    async def dispatch(**_kwargs):
        events.append("dispatch")
        return _success_dispatch_result()

    monkeypatch.setattr(prompts, "dispatch_prompt_improvement", dispatch)
    with _ordered_gate_client(monkeypatch, events) as client:
        response = client.post("/api/v1/prompts/improve", json=_payload())

    assert response.status_code == status.HTTP_200_OK, response.text
    assert events == ["quota", "rbac", "budget", "billing", "dispatch"]


@pytest.mark.parametrize(
    "protected_tokens",
    [
        [{"kind": "template_variable", "value": "{{missing}}", "occurrences": 1}],
        [{"kind": "template_variable", "value": "{{audience}}", "occurrences": 2}],
        [
            {
                "kind": "literal",
                "value": str(index).zfill(MAX_PROTECTED_TOKEN_CHARS),
                "occurrences": 1,
            }
            for index in range((MAX_PROTECTED_TOKEN_TOTAL_CHARS // MAX_PROTECTED_TOKEN_CHARS) + 1)
        ],
    ],
)
def test_protected_token_occurrence_presence_and_total_limits_precede_dispatch(
    monkeypatch,
    protected_tokens,
):
    dispatch_spy = AsyncMock(side_effect=AssertionError("provider must not be called"))
    monkeypatch.setattr(prompts, "dispatch_prompt_improvement", dispatch_spy)
    text = _TARGET_DRAFT
    if len(protected_tokens) > 2:
        text = " ".join(token["value"] for token in protected_tokens)

    with _isolated_prompt_client() as (client, _app, _route):
        response = client.post(
            "/api/v1/prompts/improve",
            json=_payload(text=text, protected_tokens=protected_tokens),
        )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["code"] == "invalid_input"
    dispatch_spy.assert_not_awaited()


def test_public_response_contract_bounds_all_provider_authored_fields():
    schemas = importlib.import_module(
        "tldw_Server_API.app.api.v1.schemas.prompt_schemas"
    )
    valid = {
        "operation_id": _OPERATION_ID,
        "status": "improved",
        "improved_text": "candidate",
        "findings": [{"category": "clarity", "issue": "issue", "change": "change"}],
        "review_required": False,
        "warnings": [],
        "resolved_model": {
            "provider": "openai",
            "model": "test-model",
            "display_name": "Test model",
        },
        "meta_prompt_version": META_PROMPT_VERSION,
    }
    response_model = schemas.PromptImproveResponse.model_validate(valid)
    assert response_model.schema_version == 1

    invalid_payloads = [
        {**valid, "unknown": True},
        {**valid, "improved_text": "x" * (PROMPT_IMPROVEMENT_LIMITS.max_candidate_chars + 1)},
        {
            **valid,
            "findings": [
                {"category": "clarity", "issue": "issue", "change": "change"}
            ]
            * (PROMPT_IMPROVEMENT_LIMITS.max_findings + 1),
        },
        {
            **valid,
            "findings": [
                {
                    "category": "clarity",
                    "issue": "x" * (MAX_FINDING_TEXT_CHARS + 1),
                    "change": "change",
                }
            ],
        },
        {
            **valid,
            "resolved_model": {
                "provider": "p" * (PROMPT_IMPROVEMENT_LIMITS.max_provider_chars + 1),
                "model": "m",
                "display_name": "m",
            },
        },
        {
            **valid,
            "resolved_model": {
                "provider": "p",
                "model": "m" * (PROMPT_IMPROVEMENT_LIMITS.max_model_chars + 1),
                "display_name": "m",
            },
        },
        {
            **valid,
            "warnings": ["w" * (PROMPT_IMPROVEMENT_LIMITS.max_warning_chars + 1)],
        },
        {
            **valid,
            "warnings": ["warning"] * (PROMPT_IMPROVEMENT_LIMITS.max_warnings + 1),
        },
    ]
    for invalid in invalid_payloads:
        with pytest.raises(ValidationError):
            schemas.PromptImproveResponse.model_validate(invalid)


@pytest.mark.parametrize(
    ("gate", "status_code"),
    [
        ("auth", status.HTTP_401_UNAUTHORIZED),
        ("scope", status.HTTP_403_FORBIDDEN),
    ],
)
def test_improve_endpoint_enforces_chat_equivalent_dependencies(
    monkeypatch,
    gate,
    status_code,
):
    monkeypatch.setattr(prompts, "dispatch_prompt_improvement", AsyncMock())

    with _isolated_prompt_client() as (client, app, route):
        if gate == "auth":
            dependency = _dependency_call(route, lambda call: call is get_auth_principal)
        elif gate == "scope":
            dependency = _dependency_call(
                route,
                lambda call: getattr(call, "_tldw_endpoint_id", None) == "prompts.improve",
            )
        else:
            raise AssertionError(f"unexpected pre-body gate: {gate}")

        async def deny_dependency() -> None:
            raise HTTPException(status_code=status_code, detail="gate denied")

        app.dependency_overrides[dependency] = deny_dependency
        response = client.post("/api/v1/prompts/improve", json=_payload())

    assert response.status_code == status_code


@pytest.mark.parametrize(
    ("gate", "status_code"),
    [
        ("rbac", status.HTTP_429_TOO_MANY_REQUESTS),
        ("budget", status.HTTP_402_PAYMENT_REQUIRED),
        ("billing", status.HTTP_402_PAYMENT_REQUIRED),
    ],
)
def test_improve_endpoint_enforces_post_validation_operation_gates(
    monkeypatch,
    gate,
    status_code,
):
    async def allow(*_args, **_kwargs) -> None:
        return None

    async def deny(*_args, **_kwargs) -> None:
        raise HTTPException(status_code=status_code, detail="gate denied")

    async def get_pool() -> object:
        return object()

    monkeypatch.setattr(prompts, "get_db_pool", get_pool)
    monkeypatch.setattr(
        prompts,
        "consume_deferred_token_quota",
        allow,
    )
    monkeypatch.setattr(
        prompts,
        "enforce_rbac_rate_limit",
        deny if gate == "rbac" else allow,
    )
    monkeypatch.setattr(
        prompts,
        "enforce_llm_budget",
        deny if gate == "budget" else allow,
    )
    monkeypatch.setattr(
        prompts,
        "_PROMPT_IMPROVEMENT_BILLING_CHECK",
        deny if gate == "billing" else allow,
    )
    monkeypatch.setattr(prompts, "dispatch_prompt_improvement", AsyncMock())

    with _isolated_prompt_client(bypass_post_validation_gates=False) as (
        client,
        _app,
        _route,
    ):
        response = client.post("/api/v1/prompts/improve", json=_payload())

    assert response.status_code == status_code


def test_improve_dependency_contract_is_ordinary_chat_access_not_prompt_admin():
    with _isolated_prompt_client() as (_client, _app, route):
        calls = [dependency.call for dependency in route.dependant.dependencies]

    assert prompts.verify_prompts_user not in calls
    assert get_auth_principal in calls
    assert any(
        getattr(call, "_tldw_endpoint_id", None) == "prompts.improve"
        and getattr(call, "_tldw_count_as", None) == "call"
        and getattr(call, "_tldw_defer_count", False)
        for call in calls
    )
    assert enforce_llm_budget not in calls


@pytest.mark.parametrize(
    ("error_code", "expected_status", "retryable"),
    [
        ("missing_model", status.HTTP_400_BAD_REQUEST, False),
        ("unsupported_model", status.HTTP_400_BAD_REQUEST, False),
        ("provider_not_configured", status.HTTP_503_SERVICE_UNAVAILABLE, False),
        ("provider_rate_limited", status.HTTP_429_TOO_MANY_REQUESTS, True),
        ("provider_timeout", status.HTTP_504_GATEWAY_TIMEOUT, True),
        ("provider_unavailable", status.HTTP_503_SERVICE_UNAVAILABLE, True),
        ("model_refusal", status.HTTP_422_UNPROCESSABLE_ENTITY, False),
        ("invalid_model_output", status.HTTP_502_BAD_GATEWAY, False),
        ("preservation_failed", status.HTTP_422_UNPROCESSABLE_ENTITY, False),
        ("internal_error", status.HTTP_500_INTERNAL_SERVER_ERROR, False),
    ],
)
def test_stable_error_mapping_never_exposes_internal_exception_text(
    monkeypatch,
    error_code,
    expected_status,
    retryable,
):
    dispatch = _dispatch_module()
    sensitive = "SECRET_PROVIDER_BODY PRIVATE_DRAFT_SENTINEL sk-secret"
    if error_code in {"invalid_model_output", "preservation_failed"}:
        raw = "" if error_code == "invalid_model_output" else "x" * 32_001

        async def fake_dispatch(**_kwargs):
            return dispatch.PromptImprovementDispatchResult(
                text=raw,
                provider="openai",
                model="test-model",
                display_name="test-model",
            )

    else:
        async def fake_dispatch(**_kwargs):
            raise dispatch.PromptImprovementDispatchError(
                error_code,
                internal_detail=sensitive,
                retryable=retryable,
                retry_after_seconds=17 if error_code == "provider_rate_limited" else None,
            )

    monkeypatch.setattr(prompts, "dispatch_prompt_improvement", fake_dispatch)

    with _isolated_prompt_client() as (client, _app, _route):
        response = client.post("/api/v1/prompts/improve", json=_payload())

    assert response.status_code == expected_status, response.text
    body = response.json()
    assert body["code"] == error_code
    assert body["retryable"] is retryable
    assert body["request_id"]
    assert sensitive not in response.text
    assert _TARGET_DRAFT not in response.text
    if error_code == "provider_rate_limited":
        assert body["retry_after_seconds"] == 17
        assert response.headers["Retry-After"] == "17"
    else:
        assert "retry_after_seconds" not in body


def test_capabilities_enable_track_a_with_centralized_limits_and_keep_recipe_disabled():
    with _isolated_prompt_client() as (client, _app, _route):
        response = client.get("/api/v1/prompts/capabilities")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json() == {
        "prompt_improvement_v1": {
            "supported": True,
            "limits": {
                "max_draft_chars": PROMPT_IMPROVEMENT_LIMITS.max_draft_chars,
                "max_request_bytes": PROMPT_IMPROVEMENT_LIMITS.max_request_bytes,
                "max_candidate_chars": PROMPT_IMPROVEMENT_LIMITS.max_candidate_chars,
                "max_raw_output_chars": PROMPT_IMPROVEMENT_LIMITS.max_raw_output_chars,
                "max_findings": PROMPT_IMPROVEMENT_LIMITS.max_findings,
                "max_finding_text_chars": PROMPT_IMPROVEMENT_LIMITS.max_finding_text_chars,
                "max_provider_chars": PROMPT_IMPROVEMENT_LIMITS.max_provider_chars,
                "max_model_chars": PROMPT_IMPROVEMENT_LIMITS.max_model_chars,
                "max_meta_prompt_version_chars": (
                    PROMPT_IMPROVEMENT_LIMITS.max_meta_prompt_version_chars
                ),
                "max_warning_chars": PROMPT_IMPROVEMENT_LIMITS.max_warning_chars,
                "max_warnings": PROMPT_IMPROVEMENT_LIMITS.max_warnings,
                "max_protected_tokens": PROMPT_IMPROVEMENT_LIMITS.max_protected_tokens,
                "max_protected_token_kind_chars": (
                    PROMPT_IMPROVEMENT_LIMITS.max_protected_token_kind_chars
                ),
                "max_protected_token_chars": PROMPT_IMPROVEMENT_LIMITS.max_protected_token_chars,
                "max_protected_token_occurrences": (
                    PROMPT_IMPROVEMENT_LIMITS.max_protected_token_occurrences
                ),
                "max_protected_token_total_chars": (
                    PROMPT_IMPROVEMENT_LIMITS.max_protected_token_total_chars
                ),
            },
        },
        "single_text_recipe_v2": {"supported": False},
    }


def test_prompt_capabilities_catalog_scope_is_standard_for_ordinary_roles():
    from tldw_Server_API.app.core.AuthNZ.privilege_catalog import load_catalog

    scope = next(
        entry for entry in load_catalog().scopes if entry.id == "prompts.capabilities"
    )

    assert scope.rate_limit_class == "standard"
    assert scope.default_roles == ["admin", "analyst", "developer", "user"]


def test_prompt_capabilities_endpoint_requires_real_authentication():
    app = FastAPI()
    app.include_router(prompts.router, prefix="/api/v1/prompts")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/prompts/capabilities")

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_prompt_improvement_endpoint_requires_real_authentication():
    app = FastAPI()
    app.include_router(prompts.router, prefix="/api/v1/prompts")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/prompts/improve", json=_payload())

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


class _EmptyCursor:
    async def fetchone(self):
        return None


class _EmptyConnection:
    async def execute(self, *_args, **_kwargs):
        return _EmptyCursor()


class _EmptyPool:
    pool = None

    @asynccontextmanager
    async def acquire(self):
        yield _EmptyConnection()


def test_prompt_capabilities_catalog_rate_limit_returns_429(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.privilege_catalog import PrivilegeCatalog

    catalog = PrivilegeCatalog.model_validate(
        {
            "version": "capabilities-rate-test-1",
            "updated_at": "2026-08-02T00:00:00Z",
            "scopes": [
                {
                    "id": "prompts.capabilities",
                    "description": "Discover prompt capabilities.",
                    "resource_tags": ["prompts"],
                    "sensitivity_tier": "low",
                    "rate_limit_class": "standard",
                    "default_roles": ["user"],
                    "feature_flag_id": None,
                    "ownership_predicates": [],
                    "doc_url": None,
                }
            ],
            "feature_flags": [],
            "rate_limit_classes": [
                {
                    "id": "standard",
                    "requests_per_min": 1,
                    "burst": 1,
                    "notes": "Focused integration limit",
                }
            ],
            "ownership_predicates": [],
        }
    )
    monkeypatch.setattr(auth_deps, "load_catalog", lambda: catalog, raising=False)
    auth_deps._AUTH_DEPS_FALLBACK_RATE_WINDOWS.clear()
    pool = _EmptyPool()

    app = FastAPI()
    app.include_router(prompts.router, prefix="/api/v1/prompts")

    async def principal(request: Request) -> AuthPrincipal:
        value = AuthPrincipal(
            kind="user",
            user_id=11,
            username="user-11",
            subject="user:11",
            roles=["user"],
        )
        request.state.user_id = value.user_id
        request.state.auth = AuthContext(principal=value)
        return value

    async def get_pool():
        return pool

    app.dependency_overrides[get_auth_principal] = principal
    app.dependency_overrides[auth_deps.get_db_pool] = get_pool

    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.get("/api/v1/prompts/capabilities")
        limited = client.get("/api/v1/prompts/capabilities")

    assert first.status_code == status.HTTP_200_OK, first.text
    assert limited.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert limited.json()["detail"] == (
        "Rate limit exceeded for resource: prompts.capabilities"
    )


def test_prompt_improvement_catalog_rate_limit_is_per_user(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.privilege_catalog import PrivilegeCatalog

    catalog = PrivilegeCatalog.model_validate(
        {
            "version": "rate-test-1",
            "updated_at": "2026-08-01T00:00:00Z",
            "scopes": [
                {
                    "id": "prompts.improve",
                    "description": "Improve a prompt.",
                    "resource_tags": ["prompts"],
                    "sensitivity_tier": "high",
                    "rate_limit_class": "elevated",
                    "default_roles": ["user"],
                    "feature_flag_id": None,
                    "ownership_predicates": [],
                    "doc_url": None,
                }
            ],
            "feature_flags": [],
            "rate_limit_classes": [
                {
                    "id": "elevated",
                    "requests_per_min": 2,
                    "burst": 2,
                    "notes": "Focused integration limit",
                }
            ],
            "ownership_predicates": [],
        }
    )
    monkeypatch.setattr(auth_deps, "load_catalog", lambda: catalog, raising=False)
    auth_deps._AUTH_DEPS_FALLBACK_RATE_WINDOWS.clear()
    pool = _EmptyPool()

    app = FastAPI()
    app.include_router(prompts.router, prefix="/api/v1/prompts")
    app.state.routing_decision_store = InMemoryRoutingDecisionStore()
    route = _prompt_route(app)

    async def principal(request: Request) -> AuthPrincipal:
        user_id = int(request.headers["X-Test-User"])
        value = AuthPrincipal(
            kind="user",
            user_id=user_id,
            username=f"user-{user_id}",
            subject=f"user:{user_id}",
            roles=["user"],
        )
        request.state.user_id = user_id
        request.state.auth = AuthContext(principal=value)
        return value

    async def current_user(request: Request) -> SimpleNamespace:
        user_id = int(request.headers["X-Test-User"])
        return SimpleNamespace(id=user_id, id_int=user_id, username=f"user-{user_id}")

    async def allow(*_args, **_kwargs) -> None:
        return None

    async def allow_dependency() -> None:
        return None

    async def get_pool():
        return pool

    async def dispatch(**_kwargs):
        return _success_dispatch_result()

    monkeypatch.setattr(prompts, "get_db_pool", get_pool, raising=False)
    monkeypatch.setattr(prompts, "consume_deferred_token_quota", allow, raising=False)
    monkeypatch.setattr(prompts, "enforce_llm_budget", allow)
    monkeypatch.setattr(
        prompts,
        "_PROMPT_IMPROVEMENT_BILLING_CHECK",
        allow,
        raising=False,
    )
    monkeypatch.setattr(prompts, "dispatch_prompt_improvement", dispatch)
    app.dependency_overrides[auth_deps.get_db_pool] = get_pool

    for dependency in route.dependant.dependencies:
        call = dependency.call
        if call is get_request_user:
            app.dependency_overrides[call] = current_user
        elif call is get_request_routing_decision_store:
            app.dependency_overrides[call] = lambda: app.state.routing_decision_store
        elif call is get_auth_principal:
            app.dependency_overrides[call] = principal
        elif getattr(call, "_tldw_endpoint_id", None) == "prompts.improve" or call is enforce_llm_budget or getattr(call, "__name__", "") == "_check_limit":
            app.dependency_overrides[call] = allow_dependency

    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post(
            "/api/v1/prompts/improve",
            headers={"X-Test-User": "11"},
            json=_payload(),
        )
        second = client.post(
            "/api/v1/prompts/improve",
            headers={"X-Test-User": "11"},
            json=_payload(),
        )
        limited = client.post(
            "/api/v1/prompts/improve",
            headers={"X-Test-User": "11"},
            json=_payload(),
        )
        isolated = client.post(
            "/api/v1/prompts/improve",
            headers={"X-Test-User": "12"},
            json=_payload(),
        )

    assert [first.status_code, second.status_code] == [200, 200]
    assert limited.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert isolated.status_code == status.HTTP_200_OK


def test_prompt_improvement_openapi_uses_canonical_forbidden_contracts():
    app = FastAPI()
    app.include_router(prompts.router, prefix="/api/v1/prompts")
    document = app.openapi()
    operation = document["paths"]["/api/v1/prompts/improve"]["post"]
    schema = operation["requestBody"]["content"]["application/json"]["schema"]
    if "$ref" in schema:
        schema = document["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
    assert schema.get("additionalProperties") is False
    assert set(schema["properties"]) == {
        "operation_id",
        "target",
        "text",
        "model_selection",
        "protected_tokens",
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert operation["responses"]["400"]["content"]["application/json"]["schema"]


@pytest.mark.asyncio
async def test_dispatch_concrete_route_uses_byok_once_without_tools_or_retry(monkeypatch):
    dispatch = _dispatch_module()
    captured: dict[str, Any] = {}
    touched = 0

    async def fake_resolve_route(request_data, **kwargs):
        captured["route_request"] = request_data
        captured["route_kwargs"] = kwargs
        return SimpleNamespace(provider="openai", model="gpt-test", was_auto=False)

    class FakeByok:
        api_key = "byok-secret"
        app_config = {"provider": {"base_url": "https://example.invalid"}}

        async def touch_last_used(self):
            nonlocal touched
            touched += 1

    async def fake_resolve_byok(provider, **kwargs):
        captured["byok_provider"] = provider
        captured["byok_kwargs"] = kwargs
        return FakeByok()

    async def fake_provider_call(**kwargs):
        captured.setdefault("provider_calls", []).append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"status":"no_change","improved_text":null,"findings":[]}',
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    monkeypatch.setattr(dispatch, "resolve_chat_route", fake_resolve_route)
    monkeypatch.setattr(dispatch, "resolve_byok_credentials", fake_resolve_byok)
    monkeypatch.setattr(dispatch, "perform_chat_api_call_async", fake_provider_call)
    monkeypatch.setattr(dispatch, "is_model_known_for_provider", lambda *_args: True)
    messages = [
        {"role": "system", "content": "server meta prompt"},
        {"role": "user", "content": '{"target":"system","draft":"private"}'},
    ]

    result = await dispatch.dispatch_prompt_improvement(
        request=SimpleNamespace(state=SimpleNamespace()),
        current_user=_test_user(),
        routing_decision_store=InMemoryRoutingDecisionStore(),
        selected_model="gpt-test",
        provider_hint="openai",
        messages=messages,
        request_id="req-123",
        configured_providers_getter=_configured_providers,
    )

    assert result.text == '{"status":"no_change","improved_text":null,"findings":[]}'
    assert (result.provider, result.model) == ("openai", "gpt-test")
    assert captured["route_request"].model == "gpt-test"
    assert captured["route_request"].api_provider == "openai"
    assert captured["route_request"].stream is False
    assert captured["route_request"].tools is None
    assert captured["route_kwargs"]["surface"] == "prompt_improvement"
    assert captured["route_kwargs"]["endpoint"] == "POST:/api/v1/prompts/improve"
    assert captured["route_kwargs"]["latest_user_turn"] == ""
    assert captured["byok_provider"] == "openai"
    assert captured["byok_kwargs"]["user_id"] == 7
    assert len(captured["provider_calls"]) == 1
    call = captured["provider_calls"][0]
    assert call["messages_payload"] == messages
    assert call["api_key"] == "byok-secret"
    assert call["app_config"] == FakeByok.app_config
    assert call["streaming"] is False
    assert call["tools"] is None
    assert 1 <= call["max_tokens"] <= 2048
    policy = call["call_policy"]
    assert policy.max_transport_attempts == 1
    assert policy.allow_streaming is False
    assert policy.allow_tools is False
    assert policy.candidate_count == 1
    assert policy.temperature == 0.2
    assert policy.top_p == 0.95
    assert policy.privacy_safe_errors is True
    assert touched == 1


@pytest.mark.asyncio
async def test_dispatch_auto_returns_actual_resolved_route(monkeypatch):
    dispatch = _dispatch_module()

    async def fake_resolve_route(request_data, **_kwargs):
        assert request_data.model == "auto"
        assert request_data.api_provider is None
        return SimpleNamespace(
            provider="openrouter",
            model="anthropic/claude-test",
            was_auto=True,
        )

    byok = SimpleNamespace(
        api_key="router-byok",
        app_config=None,
        touch_last_used=AsyncMock(),
    )
    provider_call = AsyncMock(
        return_value={
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": " second"},
            ]
        }
    )
    monkeypatch.setattr(dispatch, "resolve_chat_route", fake_resolve_route)
    monkeypatch.setattr(dispatch, "resolve_byok_credentials", AsyncMock(return_value=byok))
    monkeypatch.setattr(dispatch, "perform_chat_api_call_async", provider_call)
    monkeypatch.setattr(dispatch, "is_model_known_for_provider", lambda *_args: True)

    result = await dispatch.dispatch_prompt_improvement(
        request=SimpleNamespace(state=SimpleNamespace()),
        current_user=_test_user(),
        routing_decision_store=InMemoryRoutingDecisionStore(),
        selected_model="auto",
        provider_hint=None,
        messages=[{"role": "system", "content": "meta"}],
        request_id="req-auto",
        configured_providers_getter=lambda: {
            "providers": [],
            "default_provider": "openrouter",
        },
    )

    assert result.text == "first second"
    assert result.provider == "openrouter"
    assert result.model == "anthropic/claude-test"
    byok.touch_last_used.assert_awaited_once_with()
    provider_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_rejects_missing_and_unknown_models_before_provider(monkeypatch):
    dispatch = _dispatch_module()
    provider_call = AsyncMock()
    monkeypatch.setattr(dispatch, "perform_chat_api_call_async", provider_call)

    with pytest.raises(dispatch.PromptImprovementDispatchError) as missing:
        await dispatch.dispatch_prompt_improvement(
            request=SimpleNamespace(state=SimpleNamespace()),
            current_user=_test_user(),
            routing_decision_store=InMemoryRoutingDecisionStore(),
            selected_model="   ",
            provider_hint="openai",
            messages=[{"role": "system", "content": "meta"}],
            request_id="req-missing",
            configured_providers_getter=_configured_providers,
        )
    assert missing.value.code == "missing_model"

    async def fake_resolve_route(*_args, **_kwargs):
        return SimpleNamespace(provider="openai", model="retired-model", was_auto=False)

    monkeypatch.setattr(dispatch, "resolve_chat_route", fake_resolve_route)
    monkeypatch.setattr(dispatch, "is_model_known_for_provider", lambda *_args: False)
    with pytest.raises(dispatch.PromptImprovementDispatchError) as unknown:
        await dispatch.dispatch_prompt_improvement(
            request=SimpleNamespace(state=SimpleNamespace()),
            current_user=_test_user(),
            routing_decision_store=InMemoryRoutingDecisionStore(),
            selected_model="retired-model",
            provider_hint="openai",
            messages=[{"role": "system", "content": "meta"}],
            request_id="req-unknown",
            configured_providers_getter=_configured_providers,
        )
    assert unknown.value.code == "unsupported_model"
    provider_call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_exception", "expected_code", "retryable"),
    [
        (TimeoutError("PRIVATE_DRAFT provider timed out"), "provider_timeout", True),
        (ConnectionError("SECRET_PROVIDER_BODY"), "provider_unavailable", True),
    ],
)
async def test_dispatch_sanitizes_timeout_and_upstream_failures(
    monkeypatch,
    provider_exception,
    expected_code,
    retryable,
):
    dispatch = _dispatch_module()

    async def fake_resolve_route(*_args, **_kwargs):
        return SimpleNamespace(provider="openai", model="gpt-test", was_auto=False)

    byok = SimpleNamespace(api_key="secret", app_config=None, touch_last_used=AsyncMock())
    monkeypatch.setattr(dispatch, "resolve_chat_route", fake_resolve_route)
    monkeypatch.setattr(dispatch, "is_model_known_for_provider", lambda *_args: True)
    monkeypatch.setattr(dispatch, "resolve_byok_credentials", AsyncMock(return_value=byok))
    monkeypatch.setattr(
        dispatch,
        "perform_chat_api_call_async",
        AsyncMock(side_effect=provider_exception),
    )

    with pytest.raises(dispatch.PromptImprovementDispatchError) as captured:
        await dispatch.dispatch_prompt_improvement(
            request=SimpleNamespace(state=SimpleNamespace()),
            current_user=_test_user(),
            routing_decision_store=InMemoryRoutingDecisionStore(),
            selected_model="gpt-test",
            provider_hint="openai",
            messages=[{"role": "system", "content": "meta"}],
            request_id="req-error",
            configured_providers_getter=_configured_providers,
        )

    assert captured.value.code == expected_code
    assert captured.value.retryable is retryable
    assert "PRIVATE_DRAFT" not in str(captured.value)
    assert "SECRET_PROVIDER_BODY" not in str(captured.value)
    byok.touch_last_used.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_dispatch_maps_rate_limit_retry_after_and_provider_refusal(monkeypatch):
    dispatch = _dispatch_module()

    async def fake_resolve_route(*_args, **_kwargs):
        return SimpleNamespace(provider="openai", model="gpt-test", was_auto=False)

    byok = SimpleNamespace(api_key="secret", app_config=None, touch_last_used=AsyncMock())
    monkeypatch.setattr(dispatch, "resolve_chat_route", fake_resolve_route)
    monkeypatch.setattr(dispatch, "is_model_known_for_provider", lambda *_args: True)
    monkeypatch.setattr(dispatch, "resolve_byok_credentials", AsyncMock(return_value=byok))

    class ProviderRateLimit(Exception):
        status_code = 429
        retry_after = 23

    monkeypatch.setattr(
        dispatch,
        "perform_chat_api_call_async",
        AsyncMock(side_effect=ProviderRateLimit("raw private body")),
    )
    with pytest.raises(dispatch.PromptImprovementDispatchError) as limited:
        await dispatch.dispatch_prompt_improvement(
            request=SimpleNamespace(state=SimpleNamespace()),
            current_user=_test_user(),
            routing_decision_store=InMemoryRoutingDecisionStore(),
            selected_model="gpt-test",
            provider_hint="openai",
            messages=[{"role": "system", "content": "meta"}],
            request_id="req-rate",
            configured_providers_getter=_configured_providers,
        )
    assert limited.value.code == "provider_rate_limited"
    assert limited.value.retry_after_seconds == 23

    byok.touch_last_used.reset_mock()
    monkeypatch.setattr(
        dispatch,
        "perform_chat_api_call_async",
        AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "content_filter",
                    }
                ]
            }
        ),
    )
    with pytest.raises(dispatch.PromptImprovementDispatchError) as refused:
        await dispatch.dispatch_prompt_improvement(
            request=SimpleNamespace(state=SimpleNamespace()),
            current_user=_test_user(),
            routing_decision_store=InMemoryRoutingDecisionStore(),
            selected_model="gpt-test",
            provider_hint="openai",
            messages=[{"role": "system", "content": "meta"}],
            request_id="req-refusal",
            configured_providers_getter=_configured_providers,
        )
    assert refused.value.code == "model_refusal"
    byok.touch_last_used.assert_awaited_once_with()
