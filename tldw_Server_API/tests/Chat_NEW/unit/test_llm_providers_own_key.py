"""The provider listings for a user with their own provider key (TASK-15583).

A provider the server has no key for is listed on the requesting user's own stored key: usable, with the models
that key lists. Each listing is built from the requesting user's key only; an anonymous or keyless request gets
the server's listing. The admin override (owner's decision, 2026-11-21, TASK-15582) hides a disabled provider and
a denied model for everyone, and a model off ``allowed_models`` only on the server's key.

Only core modules are patched (none under ``app/api/v1/endpoints``): the requesting user comes from a dependency
override, and the vendor's model list from a fake ``_http_fetch`` that answers by the request's key.
"""

from __future__ import annotations

import json
from configparser import ConfigParser
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import llm_providers
from tldw_Server_API.app.core.Provider_Credentials.byok_config import PROVIDER_APP_CONFIG_KEYS
from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError, ResolvedByokCredentials
from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
    LLMProviderOverride,
    set_llm_provider_overrides_cache_for_tests,
)

pytestmark = pytest.mark.unit

_ROUTES = ("/api/v1/llm/providers", "/api/v1/llm/models/metadata", "/api/v1/llm/providers/openai", "/api/v1/llm/models")


def _listing_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stored: dict[tuple[int, str], str],
    server_key: str | None = None,
    stored_base_url: str | None = None,
    stored_model: str | None = None,
) -> tuple[TestClient, SimpleNamespace]:
    """The four listing routes over a config with an [API] section, offline; returns the client and the requester.

    ``stored`` maps (user id, provider) to that user's own key at the vendor; with ``stored_base_url`` each stored
    key is resolved for that endpoint instead, and with ``stored_model`` it carries that default model, as a BYOK
    resolution keeps the server config's ``model``. ``server_key`` is the server's OpenAI key (none by default).
    """
    parser = ConfigParser()
    parser.read_dict({"API": {"default_api": "openai", **({"openai_api_key": server_key} if server_key else {})}})
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "MOCK_OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: parser)
    monkeypatch.setattr("tldw_Server_API.app.core.config_llm_providers.get_api_keys", lambda: {})
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Provider_Credentials.byok_runtime.resolve_static_server_fallback",
        lambda _provider: SimpleNamespace(app_config={}),
    )
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.provider_manager._provider_manager", None)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.tokenizer_resolver.resolve_tokenizer_metadata", lambda *_a, **_k: {}
    )
    monkeypatch.setattr("tldw_Server_API.app.core.Image_Generation.listing.list_image_models_for_catalog", lambda: [])

    async def serves(user_id: Any, provider: str) -> bool:
        return (user_id, provider) in stored

    async def resolve(provider: str, **kwargs: Any) -> ResolvedByokCredentials:
        key = stored.get((kwargs.get("user_id"), provider))
        if key is None:
            raise ByokResolutionError("missing_provider_credentials", provider)
        fields = {"api_base_url": stored_base_url, "model": stored_model}
        section = {name: value for name, value in fields.items() if value}
        return ResolvedByokCredentials(
            provider=provider, api_key=key, app_config={PROVIDER_APP_CONFIG_KEYS[provider]: section},
            credential_fields={}, source="user", allowlisted=True,
        )

    monkeypatch.setattr("tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps.users_own_key_serves", serves)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Provider_Credentials.provider_credential_runtime.resolve_byok_credentials", resolve
    )
    requester = SimpleNamespace(user=None)
    app = FastAPI()
    app.include_router(llm_providers.router, prefix="/api/v1")
    app.dependency_overrides[llm_providers.listing_user_id] = lambda: requester.user
    app.state.llm_manager = SimpleNamespace(llamacpp_supervisor=None)
    return TestClient(app), requester


def _vendor(monkeypatch: pytest.MonkeyPatch, answers: dict[str, list[str] | int]) -> list[str]:
    """The vendor's model list for each key (model IDs, or an HTTP status); returns each request's credential."""
    asked: list[str] = []

    def fetch(**request: Any) -> httpx.Response:
        credential = request["headers"]["Authorization"]
        asked.append(credential)
        answer = answers[credential.removeprefix("Bearer ")]
        if isinstance(answer, int):
            return httpx.Response(answer)
        return httpx.Response(200, json={"data": [{"id": model} for model in answer]})

    monkeypatch.setattr("tldw_Server_API.app.core.LLM_Calls.provider_model_inventory._http_fetch", fetch)
    return asked


def _openai_models(route: str, body: Any) -> list[str]:
    """The OpenAI models one listing route returns."""
    if route.endswith("/models/metadata"):
        return [model["name"] for model in body["models"] if model["provider"] == "openai"]
    if route.endswith("/llm/models"):
        return [model.removeprefix("openai/") for model in body if model.startswith("openai/")]
    return (body if route.endswith("/openai") else _openai_in(body))["models"]


def _openai_in(body: dict[str, Any]) -> dict[str, Any]:
    """The OpenAI entry of a ``/llm/providers`` body."""
    return next(p for p in body["providers"] if p["name"] == "openai")


def _openai_entry(client: TestClient) -> dict[str, Any]:
    return _openai_in(client.get(_ROUTES[0]).json())


def _readiness(entry: dict[str, Any]) -> tuple[Any, Any, Any]:
    return entry["is_configured"], entry["provider_enabled"], entry["readiness_reason_code"]


def test_each_user_lists_a_keyless_provider_on_their_own_key_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """TASK-15583 (AC #1, #3, #5): the server has no OpenAI key; users 7 and 9 stored their own, user 8 did not. Every
    listing route gives each user the models their own key lists, never another user's; user 8's listing is the
    anonymous one; no key appears; each key's model list is fetched once (cached by its digest)."""
    client, requester = _listing_app(monkeypatch, stored={(7, "openai"): "sk-user-7", (9, "openai"): "sk-user-9"})
    asked = _vendor(monkeypatch, {"sk-user-7": ["gpt-7"], "sk-user-9": ["gpt-9"]})

    listings: dict[Any, dict[str, Any]] = {}
    for user in (7, 8, 9, None):
        requester.user = user
        listings[user] = {route: client.get(route).json() for route in _ROUTES}
    requester.user = 7
    again = _openai_entry(client)

    models = {
        user: [_openai_models(route, body) for route, body in by_route.items()] for user, by_route in listings.items()
    }
    assert (models, again["models"], asked) == (
        {7: [["gpt-7"]] * 4, 8: [[]] * 4, 9: [["gpt-9"]] * 4, None: [[]] * 4},
        ["gpt-7"],
        ["Bearer sk-user-7", "Bearer sk-user-9"],
    )
    entries = {user: _openai_in(listings[user][_ROUTES[0]]) for user in (7, 8)}
    assert (
        _readiness(entries[7]),
        _readiness(entries[8]),
        listings[8] == listings[None],
        "sk-user-" in json.dumps(listings),
    ) == ((True, True, None), (False, False, "provider_not_configured"), True, False)


@pytest.mark.parametrize(
    ("override", "expected", "fetched"),
    [
        (
            {"allowed_models": ["gpt-allowed"], "denied_models": ["gpt-denied"]},
            {
                7: (["gpt-allowed", "gpt-off-list", "gpt-7-only"], True, True, "gpt-allowed"),
                8: (["gpt-allowed"], True, True, "gpt-allowed"),
            },
            ["Bearer sk-user-7", "Bearer sk-server"],
        ),
        ({"is_enabled": False}, {7: ([], False, False, None), 8: ([], False, False, None)}, []),
    ],
    ids=["allowlist-and-denylist", "provider-disabled"],
)
def test_the_admin_override_holds_an_own_key_only_to_what_every_key_refuses(
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, Any],
    expected: dict[int, tuple[list[str], bool, bool, str | None]],
    fetched: list[str],
) -> None:
    """TASK-15583 (AC #2 as the owner's 2026-11-21 decision amends it; Review Focus 3): the server has a key and user 7
    has their own. A disabled provider and a denied model are unselectable for both on all four routes, and a disabled
    provider gets no model-list request on either key; a model off allowed_models is listed only for user 7, whose own
    key serves the provider and wins over the server's, as in chat. Each user's entry also gives ``provider_enabled``,
    ``enabled`` and ``default_model``."""
    client, requester = _listing_app(monkeypatch, server_key="sk-server", stored={(7, "openai"): "sk-user-7"})
    asked = _vendor(monkeypatch, {
        "sk-server": ["gpt-allowed", "gpt-off-list", "gpt-denied"],
        "sk-user-7": ["gpt-allowed", "gpt-off-list", "gpt-denied", "gpt-7-only"],
    })
    set_llm_provider_overrides_cache_for_tests({"openai": LLMProviderOverride(provider="openai", **override)})

    listed = {}
    for user in (7, 8):
        requester.user = user
        entry = _openai_entry(client)
        routes = [_openai_models(route, client.get(route).json()) for route in _ROUTES]
        listed[user] = (routes, entry["provider_enabled"], entry["enabled"], entry["default_model"])

    assert (listed, asked) == (
        {user: ([models] * 4, *rest) for user, (models, *rest) in expected.items()},
        fetched,
    )


def test_a_denied_config_default_keeps_the_provider_on_an_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review round 1, I1: the admin denies the model the server's config makes OpenAI's default. A model denial is
    not a provider disable, so user 7's own key still lists OpenAI on every route, without that model, with another
    default. (Resolved with the policy check, the runtime would refuse the handle for that default.)"""
    client, requester = _listing_app(monkeypatch, stored={(7, "openai"): "sk-user-7"}, stored_model="gpt-default")
    _vendor(monkeypatch, {"sk-user-7": ["gpt-default", "gpt-other"]})
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", denied_models=["gpt-default"])}
    )
    requester.user = 7

    entry = _openai_entry(client)
    routes = [_openai_models(route, client.get(route).json()) for route in _ROUTES]

    assert (routes, _readiness(entry), entry["default_model"]) == ([["gpt-other"]] * 4, (True, True, None), "gpt-other")


def test_a_key_stored_for_an_operators_endpoint_does_not_list_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """TASK-15583 (Review Focus 1): a user key the server sends off the vendor's host is not the user's own spend
    (ADR-083, user_unmetered), so it does not list a provider the server has no key for, and nothing is fetched."""
    client, requester = _listing_app(
        monkeypatch, stored={(7, "openai"): "sk-user-7"}, stored_base_url="https://models.operator.example/v1"
    )
    asked = _vendor(monkeypatch, {"sk-user-7": ["gpt-7"]})
    requester.user = 7

    entry = _openai_entry(client)

    assert (_readiness(entry), entry["models"], asked) == ((False, False, "provider_not_configured"), [], [])


def test_a_rejected_own_key_lists_the_provider_unusable(monkeypatch: pytest.MonkeyPatch) -> None:
    """TASK-15583 (Review Focus 2): when the vendor refuses the user's key, the provider is listed on that key as
    unusable (auth_failed), with no models and nothing else to fall back to."""
    client, requester = _listing_app(monkeypatch, stored={(7, "openai"): "sk-user-7"})
    _vendor(monkeypatch, {"sk-user-7": 401})
    requester.user = 7

    entry = _openai_entry(client)

    assert (_readiness(entry), entry["models"]) == ((True, False, "auth_failed"), [])


def test_bad_credentials_on_a_route_get_the_public_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review round 1, M7 (Review Focus 4): through the real dependency, a bad X-API-KEY gets 200 and the anonymous
    body, not a 401; neither is marked private, and both vary by credential (M5)."""
    client, _requester = _listing_app(monkeypatch, stored={})
    client.app.dependency_overrides.clear()

    anonymous = client.get(_ROUTES[0])
    bad = client.get(_ROUTES[0], headers={"X-API-KEY": "not-a-key"})

    assert (
        anonymous.status_code,
        bad.status_code,
        bad.json() == anonymous.json(),
        bad.headers.get("cache-control"),
        anonymous.headers.get("vary"),
    ) == (200, 200, True, None, "Authorization, X-API-KEY, Cookie")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("principal", "expected"),
    [
        (SimpleNamespace(user_id=7), (7, "private, no-store")),
        (SimpleNamespace(user_id=None), (None, None)),
        (401, (None, None)),
        (403, (None, None)),
        (503, (None, None)),
    ],
    ids=["user", "service-principal", "401", "403", "503"],
)
async def test_the_listing_names_its_user_and_never_refuses(
    monkeypatch: pytest.MonkeyPatch, principal: Any, expected: tuple[int | None, str | None]
) -> None:
    """TASK-15583 (Review Focus 4; review round 1, M5): the listings stay public. No credentials, bad ones, or any
    refusal of the auth layer give no user (the server's listing), never an error; a principal without a user is no
    user either. A user's listing is marked private, and every listing varies by credential."""

    async def get_auth_principal(_request: Request) -> Any:
        if isinstance(principal, int):
            raise HTTPException(status_code=principal)
        return principal

    monkeypatch.setattr("tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_auth_principal", get_auth_principal)
    response = Response()

    user_id = await llm_providers.listing_user_id(Request({"type": "http", "headers": []}), response)

    assert (user_id, response.headers.get("cache-control"), response.headers.get("vary")) == (
        *expected,
        "Authorization, X-API-KEY, Cookie",
    )


def test_a_disabled_provider_gets_no_own_key_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review fix (M2): the admin disabled OpenAI, so the listing never touches user 7's stored OpenAI key (a lookup
    can refresh an OAuth token at the vendor); the provider is listed unusable for them, as for everyone."""
    client, requester = _listing_app(monkeypatch, stored={(7, "openai"): "sk-user-7"})
    asked = _vendor(monkeypatch, {"sk-user-7": ["gpt-7"]})
    set_llm_provider_overrides_cache_for_tests({"openai": LLMProviderOverride(provider="openai", is_enabled=False)})
    looked_up: list[str] = []

    async def serves(_user_id: Any, provider: str) -> bool:
        looked_up.append(provider)
        return provider == "openai"

    monkeypatch.setattr("tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps.users_own_key_serves", serves)
    requester.user = 7

    entry = _openai_entry(client)

    assert ("openai" in looked_up, entry["models"], entry["provider_enabled"], asked) == (False, [], False, [])


@pytest.mark.parametrize(
    ("groq_override", "expected"),
    [
        ({"denied_models": ["groq-denied"]}, (["groq-allowed"], True, True)),
        ({"allowed_models": ["groq-allowed"]}, (["groq-allowed"], True, True)),
        ({"is_enabled": False}, ([], False, False)),
    ],
    ids=["denied-model", "allowlist", "provider-disabled"],
)
def test_a_users_own_key_leaves_the_override_on_their_server_key_entries(
    monkeypatch: pytest.MonkeyPatch,
    groq_override: dict[str, Any],
    expected: tuple[list[str], bool, bool],
) -> None:
    """TASK-15618 (#3554's final review, M1): user 7 lists OpenAI on their own key, and Groq runs on the server's key.
    In that mixed listing the Groq entry is still held to the admin override, as for a user with no own key: a denied
    model and a model off allowed_models are not listed, and a disabled provider is listed unusable."""
    client, requester = _listing_app(monkeypatch, stored={(7, "openai"): "sk-user-7"})
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config_llm_providers.get_api_keys", lambda: {"groq": "sk-server-groq"}
    )
    _vendor(monkeypatch, {"sk-user-7": ["gpt-7"], "sk-server-groq": ["groq-allowed", "groq-denied"]})
    set_llm_provider_overrides_cache_for_tests({"groq": LLMProviderOverride(provider="groq", **groq_override)})
    requester.user = 7

    body = client.get(_ROUTES[0]).json()
    groq = next(p for p in body["providers"] if p["name"] == "groq")

    assert (_openai_in(body)["models"], groq["models"], groq.get("enabled"), groq["provider_enabled"]) == (
        ["gpt-7"],
        *expected,
    )
