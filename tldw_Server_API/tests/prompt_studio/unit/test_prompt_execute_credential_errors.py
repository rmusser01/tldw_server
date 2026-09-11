from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_prompts as prompts_endpoint,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_schemas import (
    ExecutePromptSimpleRequest,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    ProviderOverridePolicyError,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    prompt_executor as executor_module,
)

pytestmark = pytest.mark.unit


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/prompt-studio/prompts/execute",
            "headers": [],
        }
    )


async def _allow_project(*_args, **_kwargs) -> bool:
    return True


def _patch_prompt_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prompts_endpoint,
        "authoritative_prompt_project",
        lambda *_args, **_kwargs: (1, 7),
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "require_project_access",
        _allow_project,
    )
    monkeypatch.setattr(executor_module, "PromptExecutor", lambda _db: object())


def test_execute_prompt_request_normalizes_every_registered_provider_alias() -> None:
    """The simple endpoint accepts the same provider identities as Chat."""
    failures: list[tuple[str, str, str | None]] = []
    for canonical in ChatProviderRegistry.DEFAULT_ADAPTERS:
        names = (canonical, *ChatProviderRegistry.DEFAULT_ALIASES.get(canonical, ()))
        for name in names:
            normalized = ExecutePromptSimpleRequest(
                prompt_id=1,
                provider=f"  {name}  ",
                model="  model-name  ",
            )
            if normalized.provider != canonical or normalized.model != "model-name":
                failures.append((canonical, name, normalized.provider))

    assert failures == []


def test_execute_prompt_request_keeps_omitted_provider_and_model_compatible() -> None:
    omitted = ExecutePromptSimpleRequest(prompt_id=1)
    explicit_null = ExecutePromptSimpleRequest(
        prompt_id=1,
        provider=None,
        model=None,
    )

    assert (omitted.provider, omitted.model) == ("openai", None)
    assert (explicit_null.provider, explicit_null.model) == (None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", "   "),
        ("provider", "p" * 101),
        ("model", "   "),
        ("model", "m" * 101),
    ],
)
def test_execute_prompt_request_rejects_blank_or_unbounded_provider_fields(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValidationError):
        ExecutePromptSimpleRequest.model_validate(
            {
                "prompt_id": 1,
                "provider": "openai",
                "model": "model-name",
                field: value,
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "sentinel"),
    [
        ("provider", "unsupported-provider-sensitive-sentinel"),
        ("model", "model-sensitive-sentinel-" + "x" * 100),
    ],
)
async def test_execute_prompt_invalid_provider_request_stops_before_adapters(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    sentinel: str,
) -> None:
    """Defensive endpoint validation precedes runtime and executor construction."""
    monkeypatch.setattr(
        prompts_endpoint,
        "authoritative_prompt_project",
        lambda *_args, **_kwargs: (1, 7),
    )
    monkeypatch.setattr(prompts_endpoint, "require_project_access", _allow_project)

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("adapter boundary was crossed")

    monkeypatch.setattr(prompts_endpoint, "derive_trusted_credential_scope", _must_not_run)
    monkeypatch.setattr(prompts_endpoint, "ProviderCredentialRuntime", _must_not_run)
    monkeypatch.setattr(executor_module, "PromptExecutor", _must_not_run)

    values = {
        "prompt_id": 1,
        "inputs": {"task": "test"},
        "provider": "openai",
        "model": "model-name",
    }
    values[field] = sentinel
    payload = (
        ExecutePromptSimpleRequest(**values)
        if field == "provider"
        else ExecutePromptSimpleRequest.model_construct(**values)
    )

    with pytest.raises(HTTPException) as exc_info:
        await prompts_endpoint.execute_prompt_simple(
            payload,
            _request(),
            db=object(),
            user_context={"user_id": "7"},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error_code": "provider_request_invalid",
        "message": "The selected provider or model is invalid.",
    }
    assert sentinel not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
async def test_execute_prompt_passes_normalized_alias_and_model_to_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_prompt_lookup(monkeypatch)
    calls: dict[str, object] = {}
    handle = SimpleNamespace(
        provider="custom-openai-api-99",
        api_key="configured-key",
        app_config={},
        credentials_resolved=True,
    )

    class _Runtime:
        async def resolve(self, provider: str, *, model: str | None = None):
            calls["resolve"] = (provider, model)
            return handle

        async def close(self) -> None:
            return None

    class _Executor:
        def execute(self, prompt_id: int, **kwargs):
            calls["execute"] = (prompt_id, kwargs["provider"], kwargs["model"])
            calls["provider_credentials"] = kwargs.get("provider_credentials")
            return {"success": True, "output": "ok"}

    monkeypatch.setattr(
        prompts_endpoint,
        "derive_trusted_credential_scope",
        lambda *_args, **_kwargs: (7, [], [], False),
    )
    monkeypatch.setattr(prompts_endpoint, "ProviderCredentialRuntime", lambda **_kwargs: _Runtime())
    monkeypatch.setattr(executor_module, "PromptExecutor", lambda _db: _Executor())

    result = await prompts_endpoint.execute_prompt_simple(
        ExecutePromptSimpleRequest(
            prompt_id=1,
            provider="  custom_openai_api_99  ",
            model="  model-name  ",
        ),
        _request(),
        db=object(),
        user_context={"user_id": "7"},
    )

    assert result["output"] == "ok"
    assert calls["resolve"] == ("custom-openai-api-99", "model-name")
    assert calls["execute"] == (1, "custom-openai-api-99", "model-name")
    assert calls["provider_credentials"] is handle


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "expected_status"),
    [
        ("credential_scope_revoked", 403),
        ("provider_disabled", 403),
        ("model_not_allowed", 403),
        ("invalid_provider_credentials", 503),
        ("credential_store_unavailable", 503),
    ],
)
async def test_execute_prompt_preserves_typed_credential_errors(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    expected_status: int,
) -> None:
    """The simple endpoint must match Prompt Studio's structured error contract."""
    _patch_prompt_lookup(monkeypatch)
    runtime = SimpleNamespace()

    async def _resolve(*_args, **_kwargs):
        if code in {"provider_disabled", "model_not_allowed"}:
            raise ProviderOverridePolicyError(code, "openai")
        raise ByokResolutionError(code, "openai")

    async def _close() -> None:
        return None

    runtime.resolve = _resolve
    runtime.close = _close
    monkeypatch.setattr(
        prompts_endpoint,
        "derive_trusted_credential_scope",
        lambda *_args, **_kwargs: (7, [], [], False),
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "ProviderCredentialRuntime",
        lambda **_kwargs: runtime,
    )

    with pytest.raises(HTTPException) as exc_info:
        await prompts_endpoint.execute_prompt_simple(
            ExecutePromptSimpleRequest(
                prompt_id=1,
                inputs={"task": "test"},
                provider="openai",
                model="gpt-test",
            ),
            _request(),
            db=object(),
            user_context={"user_id": "7"},
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == {
        "error_code": code,
        "message": "Provider credentials are unavailable.",
    }


@pytest.mark.asyncio
async def test_execute_prompt_preserves_revoked_scope_derivation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scope revocation before runtime construction remains a typed 403."""
    _patch_prompt_lookup(monkeypatch)

    def _revoked(*_args, **_kwargs):
        raise ByokResolutionError("credential_scope_revoked", "openai")

    monkeypatch.setattr(
        prompts_endpoint,
        "derive_trusted_credential_scope",
        _revoked,
    )

    with pytest.raises(HTTPException) as exc_info:
        await prompts_endpoint.execute_prompt_simple(
            ExecutePromptSimpleRequest(
                prompt_id=1,
                inputs={"task": "test"},
                provider="openai",
                model="gpt-test",
            ),
            _request(),
            db=object(),
            user_context={"user_id": "7"},
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error_code"] == "credential_scope_revoked"


@pytest.mark.asyncio
async def test_execute_prompt_runtime_construction_failure_is_structured_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime bootstrap failures use the same public credential envelope."""
    _patch_prompt_lookup(monkeypatch)
    sentinel = "credential-store-path-and-secret"

    monkeypatch.setattr(
        prompts_endpoint,
        "derive_trusted_credential_scope",
        lambda *_args, **_kwargs: (7, [], [], False),
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "ProviderCredentialRuntime",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(sentinel)),
    )

    with pytest.raises(HTTPException) as exc_info:
        await prompts_endpoint.execute_prompt_simple(
            ExecutePromptSimpleRequest(
                prompt_id=1,
                inputs={"task": "test"},
                provider="openai",
                model="gpt-test",
            ),
            _request(),
            db=object(),
            user_context={"user_id": "7"},
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {
        "error_code": "credential_store_unavailable",
        "message": "Provider credentials are unavailable.",
    }
    assert sentinel not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
async def test_execute_prompt_missing_configured_model_uses_fixed_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_prompt_lookup(monkeypatch)

    class _Runtime:
        async def resolve(self, *_args, **_kwargs):
            return SimpleNamespace(
                api_key="configured-key",
                app_config={},
                credentials_resolved=True,
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        prompts_endpoint,
        "derive_trusted_credential_scope",
        lambda *_args, **_kwargs: (7, [], [], False),
    )
    monkeypatch.setattr(prompts_endpoint, "ProviderCredentialRuntime", lambda **_kwargs: _Runtime())

    with pytest.raises(HTTPException) as exc_info:
        await prompts_endpoint.execute_prompt_simple(
            ExecutePromptSimpleRequest(prompt_id=1, provider="openai"),
            _request(),
            db=object(),
            user_context={"user_id": "7"},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error_code": "provider_configuration_invalid",
        "message": "The selected provider configuration is invalid.",
    }


@pytest.mark.asyncio
async def test_execute_prompt_missing_credentials_uses_fixed_nonreflecting_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_prompt_lookup(monkeypatch)
    sentinel_model = "model-sensitive-sentinel"

    class _Runtime:
        async def resolve(self, *_args, **_kwargs):
            return SimpleNamespace(
                api_key=None,
                app_config={},
                credentials_resolved=True,
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        prompts_endpoint,
        "derive_trusted_credential_scope",
        lambda *_args, **_kwargs: (7, [], [], False),
    )
    monkeypatch.setattr(prompts_endpoint, "ProviderCredentialRuntime", lambda **_kwargs: _Runtime())

    with pytest.raises(HTTPException) as exc_info:
        await prompts_endpoint.execute_prompt_simple(
            ExecutePromptSimpleRequest(
                prompt_id=1,
                provider="openai",
                model=sentinel_model,
            ),
            _request(),
            db=object(),
            user_context={"user_id": "7"},
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {
        "error_code": "missing_provider_credentials",
        "message": "The selected provider credentials are not configured.",
    }
    assert sentinel_model not in str(exc_info.value)
