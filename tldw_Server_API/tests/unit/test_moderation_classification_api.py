"""Real routing/permission and receipt boundaries for explicit classification."""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import moderation
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.DB_Management.Moderation_Review_DB import ModerationReviewStore
from tldw_Server_API.app.core.Moderation import classification
from tldw_Server_API.app.core.Moderation.review_service import ModerationReviewService
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope
from tldw_Server_API.app.core.Slides.generation_disconnect import SlidesGenerationDisconnectMiddleware
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context


@pytest_asyncio.fixture
async def api_case(tmp_path, monkeypatch):
    actor = AuthPrincipal(  # nosec B106 -- 'access' is the token-kind enum, not a credential.
        kind="user",
        user_id=41,
        api_key_id=None,
        subject="user:41",
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[SYSTEM_CONFIGURE],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )
    service = ModerationReviewService(ModerationReviewStore(db_path=str(tmp_path / "review.db")))
    observed = {"resolved": 0, "used": 0, "closed": 0, "calls": [], "context": []}
    credentials = SimpleNamespace(
        trusted_endpoint=SimpleNamespace(
            base_url="http://127.0.0.1:8010/v1", scope=ConfiguredEndpointScope.from_url("http://127.0.0.1:8010/v1")
        ),
        api_key=None,
        app_config={},
        credentials_resolved=True,
    )

    class Runtime:
        def __init__(self, **kwargs):
            observed["owner"] = kwargs["user_id"]

        async def resolve(self, provider, *, model):
            observed["resolved"] += 1
            return credentials

        async def mark_used(self, handle):
            assert handle is credentials  # nosec B101 -- Pytest checks classification, authority or resource retirement.
            observed["used"] += 1

        async def close(self):
            observed["closed"] += 1

    async def provider(**kwargs):
        observed["calls"].append(kwargs)
        observed["context"].append(current_llm_call_context())
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "label": "flagged",
                                "reason": "The fictional access code is disclosed.",
                                "evidence": "ROWAN-731",
                            }
                        )
                    },
                }
            ]
        }

    monkeypatch.setattr(classification, "perform_chat_api_call_async", provider)
    monkeypatch.setattr(classification, "audited_transport", lambda p: p == "llama.cpp")
    monkeypatch.setattr(moderation, "audited_transport", lambda p: p == "llama.cpp", raising=False)
    monkeypatch.setattr(moderation, "get_moderation_review_service", lambda: service)
    # These names are supplied by the new boundary; raising=False establishes a real 404 red first.
    monkeypatch.setattr(moderation, "ProviderCredentialRuntime", Runtime, raising=False)
    monkeypatch.setattr(moderation, "load_server_config_snapshot", lambda: {}, raising=False)
    app = FastAPI()
    app.add_middleware(SlidesGenerationDisconnectMiddleware)
    app.include_router(moderation.router, prefix="/api/v1")

    async def principal(request: Request):
        request.state.auth = AuthContext(principal=actor, ip=None, user_agent=None, request_id=None)
        return actor

    async def user():
        return SimpleNamespace(id=41, id_int=41)

    app.dependency_overrides[auth_deps.get_auth_principal] = principal
    app.dependency_overrides[auth_deps.get_request_user] = user
    payload = {
        "request_id": str(uuid4()),
        "provider": "llama.cpp",
        "model": "existing-model",
        "criterion": "Flag unpublished access codes.",
        "text": "The access code is ROWAN-731.",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(client=client, actor=actor, service=service, payload=payload, observed=observed, app=app)


@pytest.mark.asyncio
async def test_real_route_saves_and_replays_with_trusted_usage_and_runtime_retirement(api_case):
    case = api_case
    first = await case.client.post("/api/v1/moderation/classify", json=case.payload)
    replay = await case.client.post("/api/v1/moderation/classify", json=case.payload)
    assert first.status_code == 200 and replay.json() == first.json()  # nosec B101 -- Pytest checks classification, authority or resource retirement.
    assert (case.observed["resolved"], case.observed["used"], case.observed["closed"], len(case.observed["calls"])) == (  # nosec B101 -- Pytest checks classification, authority or resource retirement.
        1,
        1,
        1,
        1,
    )
    context = case.observed["context"][0]
    assert (context.user_id, context.operation, context.endpoint, context.exempt, context.records_itself) == (  # nosec B101 -- Pytest checks classification, authority or resource retirement.
        41,
        "moderation.classify",
        "/api/v1/moderation/classify",
        False,
        False,
    )


@pytest.mark.asyncio
async def test_changed_receipt_conflicts_without_new_credentials_or_inference(api_case):
    case = api_case
    first = await case.client.post("/api/v1/moderation/classify", json=case.payload)
    changed = await case.client.post("/api/v1/moderation/classify", json={**case.payload, "text": "Changed"})
    assert (first.status_code, changed.status_code, case.observed["resolved"], len(case.observed["calls"])) == (  # nosec B101 -- Pytest checks classification, authority or resource retirement.
        200,
        409,
        1,
        1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("roles", [[], ["reviewer"]])
async def test_nonadmin_is_refused_before_credentials_or_provider(api_case, roles):
    case = api_case
    object.__setattr__(case.actor, "roles", roles)
    object.__setattr__(case.actor, "is_admin", False)
    response = await case.client.post("/api/v1/moderation/classify", json=case.payload)
    assert (response.status_code, case.observed["resolved"], case.service.list_items()["items"]) == (403, 0, [])  # nosec B101 -- Pytest checks classification, authority or resource retirement.


@pytest.mark.asyncio
async def test_unsupported_transport_refuses_without_credential_resolution(api_case):
    case = api_case
    response = await case.client.post("/api/v1/moderation/classify", json={**case.payload, "provider": "openai"})
    assert (response.status_code, case.observed["resolved"]) == (422, 0)  # nosec B101 -- Pytest checks classification, authority or resource retirement.


@pytest.mark.asyncio
async def test_malformed_output_is_explicit_error_and_closes_used_runtime(api_case, monkeypatch):
    async def provider(**_):
        return {"choices": [{"finish_reason": "length", "message": {"content": "secret raw output"}}]}

    monkeypatch.setattr(classification, "perform_chat_api_call_async", provider)
    case = api_case
    response = await case.client.post("/api/v1/moderation/classify", json=case.payload)
    assert (  # nosec B101 -- Pytest checks classification, authority or resource retirement.
        response.status_code,
        case.observed["used"],
        case.observed["closed"],
        case.service.list_items()["items"],
    ) == (502, 1, 1, [])
    assert "secret raw output" not in response.text  # nosec B101 -- Pytest checks classification, authority or resource retirement.


def test_model_classification_uses_chat_governance_without_changing_deterministic_rules():
    from pathlib import Path

    import yaml

    from tldw_Server_API.app.core.Resource_Governance.policy_resolver import PolicyResolver

    root = Path(__file__).resolve().parents[3]
    configuration = yaml.safe_load((root / "tldw_Server_API/Config_Files/resource_governor_policies.yaml").read_text())
    resolver = PolicyResolver(configuration["route_map"], [])
    assert (  # nosec B101 -- Pytest checks classification, authority or resource retirement.
        resolver.resolve("/api/v1/moderation/classify", "POST"),
        resolver.resolve("/api/v1/moderation/test", "POST"),
    ) == ("chat.default", "core.default")


@pytest.mark.asyncio
async def test_classification_ingress_denies_before_credentials_and_provider(api_case, tmp_path, monkeypatch):
    from pathlib import Path

    import yaml

    from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
    from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware
    from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader

    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "tldw_Server_API/Config_Files/resource_governor_policies.yaml").read_text())
    config["policies"]["chat.default"]["requests"] = {"rpm": 1, "burst": 1}
    config["policies"]["chat.default"]["scopes"] = ["ip"]
    path = tmp_path / "policies.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setenv("RG_POLICY_PATH", str(path))
    loader = PolicyLoader(path)
    await loader.load_once()
    case = api_case
    case.app.state.rg_policy_loader = loader
    case.app.state.rg_governor = MemoryResourceGovernor(policy_loader=loader)
    case.app.add_middleware(RGSimpleMiddleware)
    first = await case.client.post("/api/v1/moderation/classify", json=case.payload)
    denied = await case.client.post("/api/v1/moderation/classify", json={**case.payload, "request_id": str(uuid4())})
    assert (first.status_code, denied.status_code, case.observed["resolved"], len(case.observed["calls"])) == (  # nosec B101 -- Pytest checks classification, authority or resource retirement.
        200,
        429,
        1,
        1,
    )
    assert denied.headers.get("Retry-After") and len(case.service.list_items()["items"]) == 1  # nosec B101 -- Pytest checks classification, authority or resource retirement.
