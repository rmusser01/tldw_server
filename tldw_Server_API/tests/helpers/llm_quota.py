"""Test helpers for the LLM quota checks (LLM usage accounting spec §4a, §4b)."""

import asyncio
from typing import Any

import pytest


def refuse_llm_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every LLM quota check refuse: the user's monthly LLM token limit is 0 and nothing else is limited."""

    async def _user_quota(_uid: object, key: str) -> int | None:
        """Only the LLM token key is set, at 0."""
        return 0 if key == "limits.llm_tokens_per_month" else None

    async def _no_tokens(_uid: object) -> float:
        """No tokens used this month."""
        return 0.0

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.user_quota", _user_quota)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.llm_tokens_this_month", _no_tokens)


def refuse_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """``refuse_llm_work``, and switch on the check at the call (spec §4b) for a test without the application lifespan.

    The call-site check reads the quotas switch itself, so it is turned on for that
    check only, and cached limits are dropped. Called from a coroutine, it also makes
    the running loop the accounting loop, which the lifespan would otherwise set.
    """
    from tldw_Server_API.app.core.Usage import llm_accounting, quota_resolver

    refuse_llm_work(monkeypatch)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.llm_quota_gate.usage_quotas_enabled", lambda: True)
    quota_resolver.invalidate_all()
    try:
        monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    except RuntimeError:
        return


def own_keys_for_every_provider(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The user has their own key for every provider; returns each provider a boundary asks about.

    For a boundary whose LLM calls run on the server's key: it must refuse an over-limit user
    anyway, without asking (TASK-13956).
    """
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps

    # ponytail: patches the module attribute, so code that imported users_own_key_serves by name at module
    # level still gets the real lookup (False in tests): such a regression passes. Patch that module too then.
    asked: list[str] = []

    async def _own_key(_uid: object, provider: str) -> bool:
        """Every provider runs on the user's own key."""
        asked.append(provider)
        return True

    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    return asked


def users_own_key(monkeypatch: pytest.MonkeyPatch, provider: str = "openai", api_key: str = "sk-users-own-key") -> list[str]:
    """The user stored their own key for ``provider``; returns each provider a boundary asks about.

    The quota's lookup answers True for ``provider`` and False for every other, and the credential
    runtime resolves that key for any user id (source ``user``), with the server's model for it and no base
    URL: a base URL another test left in the environment (``MOCK_OPENAI_BASE_URL``) would make it
    ``user_unmetered``. Every other provider resolves as before.
    """
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.app.core.Provider_Credentials import provider_credential_runtime as runtime_module
    from tldw_Server_API.app.core.Provider_Credentials.byok_config import PROVIDER_APP_CONFIG_KEYS
    from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ResolvedByokCredentials

    asked: list[str] = []
    real_resolver = runtime_module.resolve_byok_credentials

    async def _own_key(_uid: object, name: str) -> bool:
        """Only ``provider`` runs on the user's own key."""
        asked.append(name)
        return name == provider

    async def _resolve(name: str, **kwargs: Any) -> Any:
        """The user's key for ``provider``; the real resolution for everything else."""
        if name != provider or kwargs.get("user_id") is None:
            return await real_resolver(name, **kwargs)
        section = PROVIDER_APP_CONFIG_KEYS[name]
        model = ((kwargs.get("server_config_snapshot") or {}).get(section) or {}).get("model")
        return ResolvedByokCredentials(
            provider=name, api_key=api_key, app_config={section: {"model": model}}, credential_fields={},
            source="user", allowlisted=True,
        )

    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    monkeypatch.setattr(runtime_module, "resolve_byok_credentials", _resolve)
    return asked


def assert_llm_quota_402(response: Any) -> None:
    """The response is the public LLM quota refusal: 402, limit_exceeded, Retry-After."""
    assert response.status_code == 402, response.text
    detail = response.json()["detail"]
    assert (detail["error"], detail["category"], detail["limit"]) == ("limit_exceeded", "llm_tokens_month", 0)
    assert int(response.headers["Retry-After"]) >= 1
