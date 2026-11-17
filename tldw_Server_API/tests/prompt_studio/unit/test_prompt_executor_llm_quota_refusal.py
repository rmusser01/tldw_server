"""Prompt Studio's executor lets an LLM quota refusal through instead of reporting a provider error (TASK-13638)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_prompts as prompts_endpoint
from tldw_Server_API.app.api.v1.schemas.prompt_studio_schemas import ExecutePromptSimpleRequest
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import prompt_executor as executor_module
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor

pytestmark = pytest.mark.unit


def _refusal() -> LLMQuotaExceededError:
    return LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )


class _RefusingAdapter:
    def chat(self, _request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        raise _refusal()


class _PromptDb:
    client_id = "quota-refusal"

    def get_prompt(self, prompt_id: int) -> dict[str, Any]:
        return {"id": prompt_id, "deleted": False, "system_prompt": "Answer.", "user_prompt": "Q: {question}"}


@pytest.mark.asyncio
async def test_call_llm_lets_the_refusal_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optimizers' own calls go through _call_llm; a refusal is not rewritten as a retryable provider error."""

    async def _run_sync_call(function: Any, **_kwargs: Any) -> Any:
        return function()

    async def _on_success() -> None:
        raise AssertionError("a refused call is not a provider success")

    monkeypatch.setattr(
        executor_module, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _name: _RefusingAdapter())
    )
    monkeypatch.setattr(executor_module, "await_bounded_sync_call", _run_sync_call)

    with pytest.raises(LLMQuotaExceededError):
        await PromptExecutor(_PromptDb())._call_llm(
            provider="openai",
            model="m",
            prompt="p",
            api_key_override="k",
            app_config={},
            credentials_resolved=True,
            on_provider_success=_on_success,
        )


@pytest.mark.asyncio
async def test_prompt_execute_endpoint_answers_the_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /prompt-studio/prompts/execute answers 402, not 502 "Upstream provider request failed"."""

    async def _refused_call(_self: PromptExecutor, **_kwargs: Any) -> dict[str, Any]:
        raise _refusal()

    async def _allow_project(*_args: Any, **_kwargs: Any) -> bool:
        return True

    class _Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None) -> Any:
            return SimpleNamespace(api_key="k", app_config={}, credentials_resolved=True)

        async def close(self) -> None:
            return None

    monkeypatch.setattr(PromptExecutor, "_call_llm", _refused_call)
    monkeypatch.setattr(prompts_endpoint, "authoritative_prompt_project", lambda *_args, **_kwargs: (1, 7))
    monkeypatch.setattr(prompts_endpoint, "require_project_access", _allow_project)
    monkeypatch.setattr(
        prompts_endpoint, "derive_trusted_credential_scope", lambda *_args, **_kwargs: (7, [], [], False)
    )
    monkeypatch.setattr(prompts_endpoint, "ProviderCredentialRuntime", lambda **_kwargs: _Runtime())
    request = Request({"type": "http", "method": "POST", "path": "/api/v1/prompt-studio/prompts/execute", "headers": []})

    with pytest.raises(LLMQuotaExceededError) as exc_info:
        await prompts_endpoint.execute_prompt_simple(
            ExecutePromptSimpleRequest(prompt_id=1, inputs={"question": "q"}, provider="openai", model="m"),
            request,
            db=_PromptDb(),
            user_context={"user_id": "7"},
        )

    assert exc_info.value.detail["error"] == "limit_exceeded"
