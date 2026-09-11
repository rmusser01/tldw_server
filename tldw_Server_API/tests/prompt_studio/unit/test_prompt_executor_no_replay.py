from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    prompt_executor as executor_module,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import (
    PromptExecutor,
)


@pytest.mark.asyncio
async def test_prompt_executor_does_not_replay_rate_limited_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-dispatch rate-limit error is ambiguous and remains single-attempt."""
    dispatches = {"n": 0}

    class _Adapter:
        def chat(self, _request: dict[str, Any]) -> dict[str, Any]:
            dispatches["n"] += 1
            raise RuntimeError("429 rate limit from private provider")

    class _Registry:
        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

    async def _run_sync_call(function, **_kwargs):
        return function()

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(executor_module, "get_registry", _Registry)
    monkeypatch.setattr(executor_module, "await_bounded_sync_call", _run_sync_call)
    monkeypatch.setattr(executor_module.asyncio, "sleep", _no_sleep)

    executor = PromptExecutor(SimpleNamespace(client_id="unit-test"))
    with pytest.raises(RuntimeError) as exc_info:
        await executor._call_llm(
            provider="openai",
            model="safe-model",
            prompt="safe prompt",
            api_key_override="safe-key",
            app_config={},
            credentials_resolved=True,
        )

    assert (
        dispatches["n"],
        str(exc_info.value),
        exc_info.value.__cause__,
        exc_info.value.__context__,
    ) == (1, "Provider returned an error response", None, None)
