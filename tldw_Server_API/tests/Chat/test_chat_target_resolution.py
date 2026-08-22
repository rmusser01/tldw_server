from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import sharing
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError


class _Registry:
    def __init__(
        self,
        *providers: str,
        unavailable: set[str] | None = None,
        adapter_error: Exception | None = None,
    ) -> None:
        self._providers = set(providers)
        self._unavailable = unavailable or set()
        self._adapter_error = adapter_error

    def list_providers(self) -> list[str]:
        return sorted(self._providers)

    def resolve_provider_name(self, name: str | None) -> str:
        aliases = {"llamacpp": "llama.cpp", "llama": "llama.cpp"}
        normalized = str(name or "").strip().lower()
        return aliases.get(normalized, normalized)

    def get_adapter(self, name: str):
        if self._adapter_error is not None:
            raise self._adapter_error
        provider = self.resolve_provider_name(name)
        if provider not in self._providers or provider in self._unavailable:
            return None
        return object()


@pytest.fixture
def target_module(monkeypatch):
    from tldw_Server_API.app.core.Chat import chat_target_resolution as target

    monkeypatch.setattr(
        target._adapter_registry,
        "get_registry",
        lambda: _Registry("openai", "anthropic", "llama.cpp", "local-llm"),
    )
    monkeypatch.setattr(target, "validate_provider_override", lambda provider, model: None)
    monkeypatch.setattr(target, "get_default_provider", lambda: "openai")
    monkeypatch.setattr(
        target,
        "get_default_model_for_provider",
        lambda provider: {
            "openai": "gpt-default",
            "anthropic": "claude-default",
            "llama.cpp": "local-default",
        }.get(provider),
    )
    return target


@pytest.mark.unit
def test_resolve_chat_target_accepts_explicit_provider_and_model(target_module) -> None:
    resolved = target_module.resolve_chat_target(
        requested_provider=" OPENAI ", requested_model=" gpt-explicit "
    )

    assert resolved == target_module.ResolvedChatTarget(
        provider="openai", model="gpt-explicit"
    )


@pytest.mark.unit
def test_resolve_chat_target_accepts_provider_qualified_model(target_module) -> None:
    resolved = target_module.resolve_chat_target(
        requested_provider=None,
        requested_model="anthropic/claude-special",
    )

    assert resolved == target_module.ResolvedChatTarget(
        provider="anthropic", model="claude-special"
    )


@pytest.mark.unit
def test_resolve_chat_target_uses_server_provider_and_model_defaults(target_module) -> None:
    resolved = target_module.resolve_chat_target(
        requested_provider=None,
        requested_model=None,
    )

    assert resolved == target_module.ResolvedChatTarget(
        provider="openai", model="gpt-default"
    )


@pytest.mark.unit
def test_resolve_chat_target_uses_registry_aliases(target_module) -> None:
    resolved = target_module.resolve_chat_target(
        requested_provider="llamacpp",
        requested_model="local-model",
    )

    assert resolved.provider == "llama.cpp"
    assert resolved.model == "local-model"


@pytest.mark.unit
def test_resolve_chat_target_enforces_provider_override_policy(
    target_module, monkeypatch
) -> None:
    monkeypatch.setattr(
        target_module,
        "validate_provider_override",
        lambda provider, model: {
            "error_code": "model_not_allowed",
            "message": "sensitive admin text",
        },
    )

    with pytest.raises(ChatConfigurationError) as exc_info:
        target_module.resolve_chat_target(
            requested_provider="openai",
            requested_model="forbidden-model",
        )

    assert exc_info.value.error_code == "provider_configuration_invalid"
    assert "sensitive admin text" not in str(exc_info.value)


@pytest.mark.unit
def test_resolve_chat_target_rejects_unknown_adapter(target_module) -> None:
    with pytest.raises(ChatConfigurationError):
        target_module.resolve_chat_target(
            requested_provider="unknown-provider",
            requested_model="model",
        )


@pytest.mark.unit
def test_resolve_chat_target_rejects_registered_but_disabled_adapter(
    target_module,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        target_module._adapter_registry,
        "get_registry",
        lambda: _Registry("openai", unavailable={"openai"}),
    )

    with pytest.raises(ChatConfigurationError):
        target_module.resolve_chat_target(
            requested_provider="openai",
            requested_model="gpt-model",
        )


@pytest.mark.unit
def test_resolve_chat_target_rejects_adapter_load_failure(
    target_module,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        target_module._adapter_registry,
        "get_registry",
        lambda: _Registry("openai", adapter_error=RuntimeError("private import error")),
    )

    with pytest.raises(ChatConfigurationError) as exc_info:
        target_module.resolve_chat_target(
            requested_provider="openai",
            requested_model="gpt-model",
        )

    assert "private import error" not in str(exc_info.value)


@pytest.mark.unit
def test_request_provider_identity_uses_qualified_model_alias_and_default(
    target_module,
) -> None:
    assert (
        target_module.resolve_chat_provider_identity(
            requested_provider=None,
            requested_model="anthropic/claude-special",
        )
        == "anthropic"
    )
    assert (
        target_module.resolve_chat_provider_identity(
            requested_provider="llamacpp",
            requested_model="local-model",
        )
        == "llama.cpp"
    )
    assert (
        target_module.resolve_chat_provider_identity(
            requested_provider=None,
            requested_model=None,
        )
        == "openai"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("catalog_providers", "expected_provider"),
    [
        (("anthropic",), "anthropic"),
        (("anthropic", "openrouter"), "openai"),
        ((), "openai"),
    ],
)
def test_request_provider_identity_uses_only_unique_catalog_inference(
    target_module,
    monkeypatch,
    catalog_providers: tuple[str, ...],
    expected_provider: str,
) -> None:
    from tldw_Server_API.app.core.Chat import chat_service

    monkeypatch.setattr(chat_service, "_load_models_with_case_cached", lambda provider: [])
    monkeypatch.setattr(
        chat_service,
        "_provider_has_model_cached",
        lambda provider, model: False,
    )
    monkeypatch.setattr(
        chat_service,
        "_find_catalog_providers_for_model_cached",
        lambda model: catalog_providers,
    )

    identity = target_module.resolve_chat_provider_identity(
        requested_provider=None,
        requested_model="claude-special",
    )
    target = target_module.resolve_chat_target(
        requested_provider=None,
        requested_model="claude-special",
    )

    assert identity == target.provider == expected_provider


@pytest.mark.unit
def test_request_provider_identity_never_infers_over_explicit_provider(
    target_module,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.Chat import chat_service

    monkeypatch.setattr(chat_service, "_load_models_with_case_cached", lambda provider: [])
    monkeypatch.setattr(
        chat_service,
        "_provider_has_model_cached",
        lambda provider, model: False,
    )
    monkeypatch.setattr(
        chat_service,
        "_find_catalog_providers_for_model_cached",
        lambda model: ("anthropic",),
    )

    identity = target_module.resolve_chat_provider_identity(
        requested_provider="openai",
        requested_model="claude-special",
    )
    target = target_module.resolve_chat_target(
        requested_provider="openai",
        requested_model="claude-special",
    )

    assert identity == target.provider == "openai"


@pytest.mark.unit
def test_resolve_chat_target_rejects_missing_configured_model(
    target_module, monkeypatch
) -> None:
    monkeypatch.setattr(
        target_module, "get_default_model_for_provider", lambda provider: None
    )

    with pytest.raises(ChatConfigurationError):
        target_module.resolve_chat_target(
            requested_provider=None,
            requested_model=None,
        )


@pytest.mark.unit
def test_explicit_provider_never_falls_back_to_qualified_model_provider(
    target_module,
) -> None:
    resolved = target_module.resolve_chat_target(
        requested_provider="openai",
        requested_model="anthropic/claude-special",
    )

    assert resolved.provider == "openai"
    assert resolved.model == "claude-special"


@dataclass
class _Credentials:
    api_key: str | None
    app_config: dict[str, str] | None = None
    source: str = "server_default"


def _share_context(*, scope_type: str = "team") -> SimpleNamespace:
    return SimpleNamespace(
        recipient_user_id=41,
        share_scope_type=scope_type,
        share_scope_id=73,
    )


@pytest.mark.asyncio
async def test_bootstrap_generation_default_uses_same_target_and_exact_team_scope(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.Chat.chat_target_resolution import ResolvedChatTarget

    captured: dict[str, object] = {}

    async def _resolve_credentials(provider: str, **kwargs):
        captured["provider"] = provider
        captured.update(kwargs)
        return _Credentials(
            api_key="secret",
            app_config={"base_url": "https://must-not-leak.example"},
            source="team",
        )

    monkeypatch.setattr(
        sharing,
        "resolve_chat_target",
        lambda **kwargs: ResolvedChatTarget("openai", "gpt-default"),
        raising=False,
    )
    monkeypatch.setattr(
        sharing, "resolve_byok_credentials", _resolve_credentials, raising=False
    )
    monkeypatch.setattr(
        sharing, "provider_requires_api_key", lambda provider: True, raising=False
    )

    payload = await sharing._resolve_recipient_generation_default(_share_context())

    assert payload == {
        "provider": "openai",
        "model": "gpt-default",
        "ready": True,
        "reason_code": None,
    }
    assert captured == {
        "provider": "openai",
        "user_id": 41,
        "request": None,
        "team_ids": [73],
        "org_ids": [],
        "trusted_base_url_override": False,
    }


@pytest.mark.asyncio
async def test_bootstrap_generation_default_uses_exact_org_scope(monkeypatch) -> None:
    from tldw_Server_API.app.core.Chat.chat_target_resolution import ResolvedChatTarget

    captured: dict[str, object] = {}

    async def _resolve_credentials(provider: str, **kwargs):
        captured.update(kwargs)
        return _Credentials(api_key="secret", source="org")

    monkeypatch.setattr(
        sharing,
        "resolve_chat_target",
        lambda **kwargs: ResolvedChatTarget("openai", "gpt-default"),
        raising=False,
    )
    monkeypatch.setattr(
        sharing, "resolve_byok_credentials", _resolve_credentials, raising=False
    )
    monkeypatch.setattr(
        sharing, "provider_requires_api_key", lambda provider: True, raising=False
    )

    await sharing._resolve_recipient_generation_default(
        _share_context(scope_type="org")
    )

    assert captured["team_ids"] == []
    assert captured["org_ids"] == [73]
    assert captured["request"] is None


@pytest.mark.asyncio
async def test_bootstrap_allows_local_provider_without_api_key(monkeypatch) -> None:
    from tldw_Server_API.app.core.Chat.chat_target_resolution import ResolvedChatTarget

    monkeypatch.setattr(
        sharing,
        "resolve_chat_target",
        lambda **kwargs: ResolvedChatTarget("llama.cpp", "local-model"),
        raising=False,
    )
    monkeypatch.setattr(
        sharing,
        "resolve_byok_credentials",
        lambda *args, **kwargs: pytest.fail("async stub must be installed"),
        raising=False,
    )

    async def _local_credentials(*args, **kwargs):
        return _Credentials(api_key=None, source="server_default")

    monkeypatch.setattr(
        sharing, "resolve_byok_credentials", _local_credentials, raising=False
    )
    monkeypatch.setattr(
        sharing, "provider_requires_api_key", lambda provider: False, raising=False
    )

    payload = await sharing._resolve_recipient_generation_default(_share_context())

    assert payload["ready"] is True
    assert payload["provider"] == "llama.cpp"


@pytest.mark.asyncio
async def test_bootstrap_fails_closed_when_required_credentials_are_absent(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.Chat.chat_target_resolution import ResolvedChatTarget

    async def _missing_credentials(*args, **kwargs):
        return _Credentials(api_key=None, app_config={"credential_source": "hidden"})

    monkeypatch.setattr(
        sharing,
        "resolve_chat_target",
        lambda **kwargs: ResolvedChatTarget("openai", "gpt-default"),
        raising=False,
    )
    monkeypatch.setattr(
        sharing, "resolve_byok_credentials", _missing_credentials, raising=False
    )
    monkeypatch.setattr(
        sharing, "provider_requires_api_key", lambda provider: True, raising=False
    )

    payload = await sharing._resolve_recipient_generation_default(_share_context())

    assert payload == {
        "provider": None,
        "model": None,
        "ready": False,
        "reason_code": "no_provider_configured",
    }
    assert "credential" not in repr(payload).lower()
    assert "base_url" not in repr(payload).lower()


@pytest.mark.asyncio
async def test_bootstrap_fails_closed_without_configured_target(monkeypatch) -> None:
    def _unavailable(**kwargs):
        raise ChatConfigurationError(message="private adapter diagnostics")

    monkeypatch.setattr(sharing, "resolve_chat_target", _unavailable, raising=False)

    payload = await sharing._resolve_recipient_generation_default(_share_context())

    assert payload == {
        "provider": None,
        "model": None,
        "ready": False,
        "reason_code": "no_provider_configured",
    }
    assert "diagnostics" not in repr(payload)


@pytest.mark.asyncio
async def test_bootstrap_reports_stable_unready_for_disabled_adapter(
    target_module,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        target_module._adapter_registry,
        "get_registry",
        lambda: _Registry("openai", unavailable={"openai"}),
    )
    monkeypatch.setattr(sharing, "resolve_chat_target", target_module.resolve_chat_target)

    payload = await sharing._resolve_recipient_generation_default(_share_context())

    assert payload == {
        "provider": None,
        "model": None,
        "ready": False,
        "reason_code": "no_provider_configured",
    }
