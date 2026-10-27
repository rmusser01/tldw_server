"""Dotted Anthropic names (claude-sonnet-4.5) stay usable as aliases of the dashed API IDs (TASK-13880)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.LLM_Calls.anthropic_model_ids import canonical_anthropic_model_id
from tldw_Server_API.app.core.Usage.pricing_catalog import reset_pricing_catalog

pytestmark = pytest.mark.unit

_DOTTED = ["claude-sonnet-4.5", "claude-haiku-4.5", "claude-opus-4.1", "claude-opus-4.5", "Claude-Sonnet-4.5"]


@pytest.fixture(autouse=True)
def _fresh_catalogs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()
    chat_service.invalidate_model_alias_caches()
    yield
    chat_service.invalidate_model_alias_caches()


_LISTED = ("claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1", "claude-opus-4-5")


@pytest.fixture
def anthropic_inventory(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Anthropic's live list holds only dashed IDs; an alias lookup of any other name is a 404 (TASK-13460)."""
    from tldw_Server_API.app.core.LLM_Calls import provider_model_inventory
    from tldw_Server_API.app.core.LLM_Calls.provider_readiness import ModelDiscoveryResult

    lookups: list[str] = []

    def lookup(**kwargs: Any) -> Any:
        lookups.append(kwargs["url"].rsplit("/", 1)[-1])
        return SimpleNamespace(status_code=404, json=lambda: {}, close=lambda: None)

    monkeypatch.setattr(chat_service, "resolve_provider_models_base_url", lambda *_a, **_k: "https://api.anthropic.com/v1")
    monkeypatch.setattr(chat_service, "discover_provider_models", lambda *_a, **_k: ModelDiscoveryResult("ready", _LISTED))
    monkeypatch.setattr(provider_model_inventory, "_http_fetch", lookup)
    return lookups


def _known_with_credentials(name: str) -> bool | None:
    return chat_service.is_model_known_for_provider(
        "anthropic", name, api_key="sk-dotted", app_config={}, credentials_resolved=True,
    )


@pytest.mark.parametrize("name", [n for n in _DOTTED if n == n.lower()])
def test_a_saved_dotted_name_stays_known_for_anthropic(anthropic_inventory: list[str], name: str) -> None:
    assert _known_with_credentials(name) is True
    assert anthropic_inventory == []  # matched as the listed dashed ID, no alias lookup


@pytest.mark.parametrize("name", _DOTTED)
def test_without_credentials_a_dotted_name_is_deferred_not_rejected(name: str) -> None:
    assert chat_service.is_model_known_for_provider("anthropic", name) is None


def test_an_unknown_dotted_name_is_still_unknown(anthropic_inventory: list[str]) -> None:
    assert _known_with_credentials("claude-nonesuch-9.9") is False
    assert _known_with_credentials("claude-sonnet-9.9") is False
    # Anthropic is asked for the dashed spelling of a Claude family name, and other names as written.
    assert anthropic_inventory == ["claude-nonesuch-9.9", "claude-sonnet-9-9"]


@pytest.mark.parametrize("name", _DOTTED)
def test_a_dotted_name_with_no_provider_is_inferred_like_its_dashed_id(name: str) -> None:
    """Neither spelling switches a commercial default to another vendor from the static catalog (TASK-13460)."""
    dotted, dotted_debug = chat_service.infer_provider_from_model_catalog(provider="openai", model=name, provider_explicit=False)
    dashed, _ = chat_service.infer_provider_from_model_catalog(
        provider="openai", model=canonical_anthropic_model_id(name), provider_explicit=False,
    )

    assert dotted == dashed == "openai"
    assert dotted_debug["reason"] == "commercial_inventory_requires_credentials"


def test_strict_model_selection_does_not_reject_a_dotted_anthropic_name(
    monkeypatch: pytest.MonkeyPatch, anthropic_inventory: list[str],
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

    monkeypatch.setenv("CHAT_ENFORCE_STRICT_MODEL_SELECTION", "1")
    credentials: dict[str, Any] = {"api_key": "sk-dotted", "app_config": {}, "credentials_resolved": True}

    assert chat_endpoint._should_enforce_strict_model_selection() is True
    assert chat_endpoint._validate_explicit_model_availability("anthropic", "claude-sonnet-4.5", **credentials) is None
    # The gate still rejects a name that is neither an ID nor a dotted alias of one.
    rejected = chat_endpoint._validate_explicit_model_availability("anthropic", "claude-nonesuch-9.9", **credentials)
    assert rejected is not None and rejected["error_code"] == "model_not_available"


def test_canonical_anthropic_model_id_dashes_only_dotted_claude_versions() -> None:
    from tldw_Server_API.app.core.LLM_Calls.anthropic_model_ids import canonical_anthropic_model_id as canon

    assert canon("claude-sonnet-4.5") == "claude-sonnet-4-5"
    assert canon("Claude-Opus-4.1") == "Claude-Opus-4-1"
    assert canon("claude-haiku-4.5") == "claude-haiku-4-5"
    for unchanged in ("claude-sonnet-4-5-20250929", "claude-opus-5", "claude-2.1", "gpt-4.1", "", "claude-3-5-haiku-20241022"):
        assert canon(unchanged) == unchanged


def test_the_adapter_sends_the_dashed_api_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers import anthropic_adapter as anth_mod

    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_ANTHROPIC", "1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
    sent: list[dict[str, Any]] = []

    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "type": "message", "id": "m", "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

    class _Client:
        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_a: Any) -> None:
            return None

        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> _Response:
            sent.append(json)
            return _Response()

    monkeypatch.setattr(anth_mod, "http_client_factory", lambda *a, **k: _Client())
    anth_mod.AnthropicAdapter().chat(
        {"messages": [{"role": "user", "content": "hi"}], "model": "claude-sonnet-4.5", "api_key": "k", "max_tokens": 32}
    )

    assert [payload["model"] for payload in sent] == ["claude-sonnet-4-5"]


def test_the_messages_native_payload_sends_the_dashed_api_id_to_anthropic_only() -> None:
    from types import SimpleNamespace

    from tldw_Server_API.app.api.v1.endpoints import messages as messages_endpoint

    request = SimpleNamespace(model_dump=lambda exclude_none=True: {"model": "x", "messages": []})

    assert messages_endpoint._prepare_native_payload(request, model="claude-sonnet-4.5", provider="anthropic")["model"] == "claude-sonnet-4-5"
    assert messages_endpoint._prepare_native_payload(request, model="claude-sonnet-4.5", provider="llama.cpp")["model"] == "claude-sonnet-4.5"


@pytest.mark.parametrize(
    ("listed", "saved", "quality_rank"),
    [
        ("claude-sonnet-4-5", "claude-sonnet-4.5", 20),
        ("claude-haiku-4-5", "claude-haiku-4.5", 55),
        ("claude-opus-4-1", "claude-opus-4.1", 10),
    ],
)
def test_the_listed_dashed_names_keep_their_capability_metadata_and_router_rank(listed, saved, quality_rank) -> None:
    """Renaming the picker entries to API IDs must not drop tool and vision support or the router rank (review I-A)."""
    from tldw_Server_API.app.api.v1.endpoints.llm_providers import get_model_metadata

    shown = get_model_metadata("anthropic", listed)
    stored = get_model_metadata("anthropic", saved)

    assert shown["tool_support"] is True
    assert shown["vision_support"] is True
    assert shown["quality_rank"] == quality_rank
    for key in ("tool_support", "vision_support", "quality_rank", "latency_rank", "cost_rank", "context_window"):
        assert stored[key] == shown[key], key


@pytest.mark.parametrize(
    ("allowed", "requested", "ok"),
    [
        (["claude-sonnet-4.5"], "claude-sonnet-4-5", True),
        (["claude-sonnet-4-5"], "claude-sonnet-4.5", True),
        (["claude-sonnet-4-5"], "claude-opus-4-1", False),
    ],
)
def test_a_provider_override_allowlist_treats_dotted_and_dashed_anthropic_names_alike(allowed, requested, ok) -> None:
    from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import LLMProviderOverride, ProviderOverrideCallSnapshot

    snapshot = ProviderOverrideCallSnapshot("anthropic", LLMProviderOverride(provider="anthropic", allowed_models=allowed))

    assert (snapshot.policy_error(requested) is None) is ok


def test_the_allowlist_check_never_raises_or_widens_on_non_string_values() -> None:
    """The virtual-key middleware skips enforcement on a TypeError, so a malformed body or entry must not raise."""
    from tldw_Server_API.app.core.LLM_Calls.anthropic_model_ids import allowlist_has_model

    assert allowlist_has_model(["claude-sonnet-4.5"], "claude-sonnet-4-5") is True
    assert allowlist_has_model(["claude-sonnet-4-5", 7], 123) is False
    assert allowlist_has_model([7, "gpt-4o"], {"model": "claude-sonnet-4-5"}) is False


def test_a_provider_override_allowlist_is_still_exact_for_other_providers() -> None:
    from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import LLMProviderOverride, ProviderOverrideCallSnapshot

    snapshot = ProviderOverrideCallSnapshot("openai", LLMProviderOverride(provider="openai", allowed_models=["claude-sonnet-4.5"]))

    assert snapshot.policy_error("claude-sonnet-4-5") is not None


def test_a_dotted_override_allowlist_still_lists_the_dashed_anthropic_models() -> None:
    """An allowlist saved from the old picker must not empty the picker or the auto-router's candidates (review I1)."""
    from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
        LLMProviderOverride,
        apply_llm_provider_overrides_to_listing,
    )

    names = ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-5"]
    payload = {"providers": [{"name": "anthropic", "models": list(names), "models_info": [{"name": n} for n in names]}]}
    overrides = {
        "anthropic": LLMProviderOverride(provider="anthropic", allowed_models=["claude-sonnet-4.5", "claude-haiku-4.5"])
    }

    (entry,) = apply_llm_provider_overrides_to_listing(payload, overrides=overrides)["providers"]

    assert entry["models"] == ["claude-sonnet-4-5", "claude-haiku-4-5"]
    assert [info["name"] for info in entry["models_info"]] == ["claude-sonnet-4-5", "claude-haiku-4-5"]


def test_an_override_allowlist_listing_is_still_exact_for_other_providers() -> None:
    from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
        LLMProviderOverride,
        apply_llm_provider_overrides_to_listing,
    )

    payload = {"providers": [{"name": "openai", "models": ["claude-sonnet-4-5", "gpt-4.1"]}]}
    overrides = {"openai": LLMProviderOverride(provider="openai", allowed_models=["claude-sonnet-4.5", "gpt-4.1"])}

    (entry,) = apply_llm_provider_overrides_to_listing(payload, overrides=overrides)["providers"]

    assert entry["models"] == ["gpt-4.1"]


@pytest.mark.parametrize(
    ("allowed", "model"),
    [
        ({"claude-sonnet-4-5": 1}, {"model": "claude-sonnet-4-5"}),
        ({"claude-sonnet-4-5": 1}, ["claude-sonnet-4-5"]),
        (5, "claude-sonnet-4-5"),
    ],
)
def test_the_allowlist_check_returns_false_instead_of_raising_on_unhashable_or_odd_shapes(allowed, model) -> None:
    """A TypeError in the virtual-key middleware skips enforcement, so the check must return False (review M4)."""
    from tldw_Server_API.app.core.LLM_Calls.anthropic_model_ids import allowlist_has_model

    assert allowlist_has_model(allowed, model) is False
