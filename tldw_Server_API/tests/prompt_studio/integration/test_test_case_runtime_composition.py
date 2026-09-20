"""End-to-end composition coverage for Prompt Studio test-case execution."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Any

import pytest
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_test_cases as test_case_endpoint,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_test import (
    RunTestCasesSimpleRequest,
)
from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime as runtime_module
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    set_llm_provider_overrides_cache_for_tests,
)
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    test_runner as test_runner_module,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_endpoint_real_manager_runner_cancellation_keeps_runtime_order(
    isolated_db: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drain the first real runner dispatch without starting the second case."""

    api_key = "prompt-composition-key"
    api_base_url = "https://prompt-composition.invalid/v1"
    model = "gpt-prompt-composition"
    monkeypatch.setenv("BYOK_ENABLED", "false")
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    monkeypatch.setenv("OPENAI_API_BASE_URL", api_base_url)
    reset_settings()
    set_llm_provider_overrides_cache_for_tests(None, healthy=True)

    events: list[str] = []
    runtime_instances: list[runtime_module.ProviderCredentialRuntime] = []
    adapter_requests: list[dict[str, Any]] = []
    adapter_started = asyncio.Event()
    adapter_release = threading.Event()
    loop = asyncio.get_running_loop()

    production_runtime = runtime_module.ProviderCredentialRuntime

    class _ObservedProductionRuntime(production_runtime):
        """Record lifecycle calls while retaining production runtime behavior."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            runtime_instances.append(self)

        async def mark_used(self, handle: Any) -> bool:
            events.append("mark:start")
            result = await super().mark_used(handle)
            events.append("mark:done")
            return result

        async def close(self) -> None:
            events.append("close:start")
            await super().close()
            events.append("close:done")

    class _BlockingAdapter:
        def chat(
            self,
            request: dict[str, Any],
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            adapter_requests.append(dict(request))
            dispatch_number = len(adapter_requests)
            events.append(f"dispatch:{dispatch_number}:start")
            loop.call_soon_threadsafe(adapter_started.set)
            if dispatch_number == 1 and not adapter_release.wait(timeout=5):
                raise TimeoutError("test adapter was not released")
            events.append(f"dispatch:{dispatch_number}:done")
            return {
                "choices": [{"message": {"content": "composed response"}}],
                "usage": {"total_tokens": 2},
            }

    adapter = _BlockingAdapter()
    monkeypatch.setattr(
        test_case_endpoint,
        "ProviderCredentialRuntime",
        _ObservedProductionRuntime,
        raising=True,
    )
    monkeypatch.setattr(
        test_runner_module,
        "get_adapter_or_raise",
        lambda provider: adapter if provider == "openai" else None,
        raising=True,
    )

    project = isolated_db.create_project("Composition", user_id="7")
    prompt = isolated_db.create_prompt(
        int(project["id"]),
        "Composed prompt",
        user_prompt="Answer {question}",
    )
    test_cases = [
        isolated_db.create_test_case(
            int(project["id"]),
            name,
            inputs={"question": question},
            expected_outputs={"response": "composed response"},
        )
        for name, question in (("first", "one"), ("second", "two"))
    ]
    payload = RunTestCasesSimpleRequest.model_validate(
        {
            "project_id": int(project["id"]),
            "prompt_id": int(prompt["id"]),
            "test_case_ids": [int(row["id"]) for row in test_cases],
            "provider": "openai",
            "model": model,
        }
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/prompt-studio/test-cases/run",
            "headers": [],
        }
    )

    task = asyncio.create_task(
        test_case_endpoint.run_test_cases_simple(
            payload=payload,
            request=request,
            db=isolated_db,
            user_context={"user_id": "7", "is_admin": True},
        )
    )
    try:
        await asyncio.wait_for(adapter_started.wait(), timeout=2)
        task.cancel()
        await asyncio.sleep(0)

        assert len(runtime_instances) == 1
        assert "close:start" not in events

        adapter_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)

        assert len(adapter_requests) == 1
        dispatched = adapter_requests[0]
        assert dispatched["api_key"] == api_key
        assert dispatched["model"] == model
        assert dispatched["credentials_resolved"] is True
        assert dispatched["app_config"]["openai_api"]["api_base_url"] == api_base_url
        assert events.index("dispatch:1:done") < events.index("mark:start")
        assert events.index("mark:done") < events.index("close:start")
        assert events[-1] == "close:done"
        assert runtime_instances[0]._closed is True
    finally:
        adapter_release.set()
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await task
        set_llm_provider_overrides_cache_for_tests(None, healthy=True)
        reset_settings()
