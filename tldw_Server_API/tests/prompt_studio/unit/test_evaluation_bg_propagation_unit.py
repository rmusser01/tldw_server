import asyncio
import contextlib
import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from loguru import logger
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps import prompt_studio_deps as deps
from tldw_Server_API.app.api.v1.schemas.prompt_studio_schemas import (
    EvaluationCreate,
    ExecutePromptSimpleRequest,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
)
from tldw_Server_API.app.main import app


def _install_owned_worker_cancellation_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> asyncio.Event:
    """Expose when cancellation has entered the owned-worker drain path."""
    from tldw_Server_API.app.core.Chat import bounded_daemon

    drain_entered = asyncio.Event()
    original = bounded_daemon._drain_owned_task

    async def _acknowledging_drain(task: asyncio.Future[Any]) -> tuple[bool, Any]:
        drain_entered.set()
        return await original(task)

    monkeypatch.setattr(bounded_daemon, "_drain_owned_task", _acknowledging_drain)
    return drain_entered


async def _wait_for_thread_event(
    event: threading.Event,
    *,
    timeout: float = 1.0,
) -> None:
    """Wait for a thread event without consuming the default executor under test."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set():
        if loop.time() >= deadline:
            raise AssertionError("Timed out waiting for thread event")
        await asyncio.sleep(0.001)


class _DirectPromptDb:
    client_id = "direct-prompt-endpoint"

    @staticmethod
    def get_prompt(prompt_id: int) -> dict[str, Any]:
        return {
            "id": prompt_id,
            "project_id": 1,
            "deleted": False,
            "user_prompt": f"request-{prompt_id}",
        }


async def _invoke_direct_prompt_endpoint(
    prompt_endpoint: Any,
    *,
    prompt_id: int,
    user_id: int,
    db: Any | None = None,
    provider: str = "openai",
    omit_model: bool = False,
) -> dict[str, Any]:
    """Call the endpoint across its current and request-scoped signatures."""
    request = SimpleNamespace(scope_user_id=user_id)
    available = inspect.signature(prompt_endpoint.execute_prompt_simple).parameters
    payload: dict[str, Any] = {
        "prompt_id": prompt_id,
        "inputs": {},
        "provider": provider,
    }
    if not omit_model:
        payload["model"] = f"model-{prompt_id}"
    kwargs = {
        "payload": ExecutePromptSimpleRequest.model_validate(payload),
        "db": db or _DirectPromptDb(),
        "request": request,
        "user_context": {
            "user_id": str(user_id),
            "client_id": f"client-{user_id}",
            "is_admin": True,
        },
        "current_user": SimpleNamespace(id_int=user_id),
    }
    return await prompt_endpoint.execute_prompt_simple(
        **{key: value for key, value in kwargs.items() if key in available}
    )


def _install_direct_prompt_runtime_seams(
    monkeypatch: pytest.MonkeyPatch,
    prompt_endpoint: Any,
    executor_module: Any,
    runtime_type: type,
    pool: Any,
) -> None:
    """Install the request-runtime and shared-pool seams expected by the endpoint."""
    monkeypatch.setattr(
        prompt_endpoint,
        "derive_trusted_credential_scope",
        lambda request, _user: (request.scope_user_id, [21], [31], True),
        raising=False,
    )
    monkeypatch.setattr(
        prompt_endpoint,
        "ProviderCredentialRuntime",
        runtime_type,
        raising=False,
    )
    monkeypatch.setattr(
        prompt_endpoint,
        "capture_provider_override_call_snapshot",
        lambda _provider: None,
        raising=False,
    )
    monkeypatch.setattr(
        prompt_endpoint,
        "provider_requires_api_key",
        lambda _provider: False,
        raising=False,
    )
    monkeypatch.setattr(
        prompt_endpoint,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    monkeypatch.setattr(
        executor_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    monkeypatch.setattr(
        executor_module,
        "is_runtime_issued_provider_call_credentials",
        lambda *_args, **_kwargs: True,
        raising=False,
    )


@pytest.mark.asyncio
async def test_direct_prompt_endpoint_uses_scoped_runtime_and_shared_adapter_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct prompt execution must dispatch only its request-captured credentials."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_prompts as prompt_endpoint,
    )
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
        prompt_executor as executor_module,
    )

    lifecycle: list[Any] = []
    adapter_requests: list[dict[str, Any]] = []
    entered = threading.Event()
    release = threading.Event()
    pool = BoundedDaemonPool(capacity=1)
    handle = SimpleNamespace(
        provider="openai",
        api_key="scoped-key-7",
        app_config={"openai_api": {"model": "scoped-model-7"}},
        credentials_resolved=True,
    )

    class _Runtime:
        def __init__(self, **kwargs: Any) -> None:
            lifecycle.append(("init", kwargs))

        async def resolve(self, provider: str, *, model: str | None = None):
            lifecycle.append(("resolve", provider, model))
            return handle

        async def mark_used(self, selected: Any) -> None:
            assert selected is handle
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("close")

    class _Adapter:
        @staticmethod
        def chat(
            request: dict[str, Any],
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            adapter_requests.append(dict(request))
            entered.set()
            assert release.wait(timeout=2.0)
            return {"choices": [{"message": {"content": "ok"}}]}

    class _Registry:
        @staticmethod
        def is_local_provider_name(_provider: str) -> bool:
            return False

        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

    _install_direct_prompt_runtime_seams(
        monkeypatch,
        prompt_endpoint,
        executor_module,
        _Runtime,
        pool,
    )
    monkeypatch.setattr(executor_module, "get_registry", lambda: _Registry())
    monkeypatch.setattr(
        executor_module,
        "ensure_app_config",
        lambda *_args, **_kwargs: {
            "openai_api": {"model": "static-model", "api_key": "static-key"}
        },
    )
    monkeypatch.setattr(
        executor_module,
        "resolve_provider_api_key_from_config",
        lambda *_args, **_kwargs: "static-key",
    )

    task = asyncio.create_task(
        _invoke_direct_prompt_endpoint(
            prompt_endpoint,
            prompt_id=1,
            user_id=7,
        )
    )
    try:
        await _wait_for_thread_event(entered)
        active_during_dispatch = pool.active_count
        release.set()
        result = await asyncio.wait_for(task, timeout=2.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert active_during_dispatch == 1
    assert pool.active_count == 0
    assert len(adapter_requests) == 1
    adapter_request = adapter_requests[0]
    assert adapter_request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is handle
    assert {
        key: value
        for key, value in adapter_request.items()
        if key != PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY
    } == {
        "messages": [{"role": "user", "content": "request-1"}],
        "system_message": None,
        "model": "model-1",
        "api_key": "scoped-key-7",
        "temperature": 0.7,
        "max_tokens": 1000,
        "app_config": {"openai_api": {"model": "scoped-model-7"}},
        "credentials_resolved": True,
    }
    assert lifecycle[0][0] == "init"
    assert lifecycle[0][1]["user_id"] == 7
    assert lifecycle[0][1]["team_ids"] == [21]
    assert lifecycle[0][1]["org_ids"] == [31]
    assert lifecycle[0][1]["trusted_base_url_override"] is True
    assert lifecycle[1:] == [("resolve", "openai", "model-1"), "mark", "close"]
    assert result["output"] == "ok"


@pytest.mark.asyncio
async def test_direct_prompt_endpoint_retries_explicit_false_provider_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_prompts as prompt_endpoint,
    )
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
        prompt_executor as executor_module,
    )

    lifecycle: list[str] = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="retry-key",
        app_config={},
        credentials_resolved=True,
    )

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, selected: Any) -> bool:
            assert selected is handle
            lifecycle.append("mark")
            return len(lifecycle) > 1

        async def close(self) -> None:
            lifecycle.append("close")

    class _Executor:
        def __init__(self, _db: Any) -> None:
            return None

        async def execute(self, _prompt_id: int, **kwargs: Any) -> dict[str, Any]:
            await kwargs["on_provider_success"]()
            return {"success": True, "output": "ok"}

    monkeypatch.setattr(prompt_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(executor_module, "PromptExecutor", _Executor)
    monkeypatch.setattr(
        prompt_endpoint,
        "derive_trusted_credential_scope",
        lambda request, _user: (request.scope_user_id, [], [], False),
    )
    monkeypatch.setattr(
        prompt_endpoint,
        "provider_requires_api_key",
        lambda _provider: False,
    )

    result = await _invoke_direct_prompt_endpoint(
        prompt_endpoint,
        prompt_id=1,
        user_id=7,
    )

    assert result["output"] == "ok"
    assert lifecycle == ["mark", "mark", "close"]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_direct_prompt_mark_retry_is_request_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_prompts as prompt_endpoint,
    )
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
        prompt_executor as executor_module,
    )

    first_marks = {owner: asyncio.Event() for owner in (1, 2)}
    attempts = dict.fromkeys(first_marks, 0)
    marked_handles: list[tuple[int, str, int]] = []
    closed: list[int] = []

    class _Runtime:
        def __init__(self, *, user_id: int, **_kwargs: Any) -> None:
            self.user_id = user_id
            self.handle = SimpleNamespace(
                provider="openai",
                api_key=f"key-{user_id}",
                app_config={},
                credentials_resolved=True,
            )

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return self.handle

        async def mark_used(self, selected: Any) -> bool:
            assert selected is self.handle
            attempts[self.user_id] += 1
            attempt = attempts[self.user_id]
            marked_handles.append((self.user_id, selected.api_key, attempt))
            if attempt == 1:
                first_marks[self.user_id].set()
                peer = 2 if self.user_id == 1 else 1
                await asyncio.wait_for(first_marks[peer].wait(), timeout=2)
                return self.user_id != 1
            return True

        async def close(self) -> None:
            closed.append(self.user_id)

    class _Executor:
        def __init__(self, _db: Any) -> None:
            return None

        async def execute(self, prompt_id: int, **kwargs: Any) -> dict[str, Any]:
            await kwargs["on_provider_success"]()
            return {"success": True, "output": f"ok-{prompt_id}"}

    monkeypatch.setattr(prompt_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(executor_module, "PromptExecutor", _Executor)
    monkeypatch.setattr(
        prompt_endpoint,
        "derive_trusted_credential_scope",
        lambda request, _user: (request.scope_user_id, [], [], False),
    )
    monkeypatch.setattr(
        prompt_endpoint,
        "provider_requires_api_key",
        lambda _provider: False,
    )

    results = await asyncio.wait_for(
        asyncio.gather(
            _invoke_direct_prompt_endpoint(prompt_endpoint, prompt_id=1, user_id=1),
            _invoke_direct_prompt_endpoint(prompt_endpoint, prompt_id=2, user_id=2),
        ),
        timeout=2,
    )

    assert [result["output"] for result in results] == ["ok-1", "ok-2"]
    assert attempts == {1: 2, 2: 1}
    assert sorted(marked_handles) == [
        (1, "key-1", 1),
        (1, "key-1", 2),
        (2, "key-2", 1),
    ]
    assert sorted(closed) == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_response",
    [
        pytest.param(None, id="none"),
        pytest.param("  ", id="whitespace"),
        pytest.param(42, id="scalar"),
        pytest.param(b"provider-bytes", id="bytes"),
        pytest.param(
            SimpleNamespace(debug="opaque-provider-secret"),
            id="arbitrary-object",
        ),
        pytest.param([], id="empty-list"),
        pytest.param((), id="empty-tuple"),
        pytest.param({}, id="empty-dict"),
    ],
)
async def test_direct_prompt_endpoint_rejects_nonsemantic_adapter_200_before_mark(
    monkeypatch: pytest.MonkeyPatch,
    provider_response: Any,
) -> None:
    """Malformed HTTP-200 adapter payloads become 502 and remain unmarked."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_prompts as prompt_endpoint,
    )
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
        prompt_executor as executor_module,
    )

    lifecycle: list[str] = []
    pool = BoundedDaemonPool(capacity=1)
    handle = SimpleNamespace(
        provider="openai",
        api_key="invalid-response-key",
        app_config={},
        credentials_resolved=True,
    )

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, _handle: Any) -> None:
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("close")

    class _Adapter:
        @staticmethod
        def chat(
            _request: dict[str, Any],
            timeout: float | None = None,
        ) -> Any:
            del timeout
            return provider_response

    class _Registry:
        @staticmethod
        def is_local_provider_name(_provider: str) -> bool:
            return False

        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

    _install_direct_prompt_runtime_seams(
        monkeypatch,
        prompt_endpoint,
        executor_module,
        _Runtime,
        pool,
    )
    monkeypatch.setattr(executor_module, "get_registry", lambda: _Registry())
    monkeypatch.setattr(executor_module, "ensure_app_config", lambda *_a, **_k: {})
    monkeypatch.setattr(
        executor_module,
        "resolve_provider_api_key_from_config",
        lambda *_a, **_k: "static-key",
    )

    with pytest.raises(HTTPException) as exc_info:
        await _invoke_direct_prompt_endpoint(
            prompt_endpoint,
            prompt_id=1,
            user_id=7,
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Upstream provider request failed."
    assert lifecycle == ["close"]
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_direct_prompt_derives_omitted_non_openai_model_from_runtime_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_prompts as prompt_endpoint,
    )
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
        prompt_executor as executor_module,
    )

    configured_model = "claude-3-5-haiku"
    resolved_models: list[str | None] = []
    dispatched: dict[str, Any] = {}
    handle = SimpleNamespace(
        provider="anthropic",
        api_key="anthropic-key",
        app_config={"anthropic_api": {"model": configured_model}},
        credentials_resolved=True,
    )

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(self, _provider: str, *, model: str | None = None):
            resolved_models.append(model)
            return handle

        async def mark_used(self, selected: Any) -> bool:
            assert selected is handle
            return True

        async def close(self) -> None:
            return None

    class _Executor:
        def __init__(self, _db: Any) -> None:
            return None

        async def execute(self, _prompt_id: int, **kwargs: Any) -> dict[str, Any]:
            dispatched.update(kwargs)
            await kwargs["on_provider_success"]()
            return {"success": True, "output": "ok"}

    monkeypatch.setattr(prompt_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(executor_module, "PromptExecutor", _Executor)
    monkeypatch.setattr(
        prompt_endpoint,
        "derive_trusted_credential_scope",
        lambda *_args: (7, [], [], False),
    )
    monkeypatch.setattr(
        prompt_endpoint,
        "provider_requires_api_key",
        lambda _provider: True,
    )

    await _invoke_direct_prompt_endpoint(
        prompt_endpoint,
        prompt_id=1,
        user_id=7,
        provider="anthropic",
        omit_model=True,
    )

    assert resolved_models == [None]
    assert dispatched["provider"] == "anthropic"
    assert dispatched["model"] == configured_model


@pytest.mark.asyncio
async def test_direct_prompt_checks_authoritative_project_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_prompts as prompt_endpoint,
    )
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
        prompt_executor as executor_module,
    )

    runtime_inits: list[bool] = []

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            runtime_inits.append(True)

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return SimpleNamespace(
                provider="openai",
                api_key="key",
                app_config={},
                credentials_resolved=True,
            )

        async def mark_used(self, _selected: Any) -> bool:
            return True

        async def close(self) -> None:
            return None

    class _Executor:
        def __init__(self, _db: Any) -> None:
            return None

        async def execute(self, _prompt_id: int, **kwargs: Any) -> dict[str, Any]:
            await kwargs["on_provider_success"]()
            return {"success": True, "output": "ok"}

    async def _deny_project(*_args: Any, **_kwargs: Any) -> None:
        raise HTTPException(status_code=403, detail="project denied")

    monkeypatch.setattr(prompt_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(executor_module, "PromptExecutor", _Executor)
    monkeypatch.setattr(
        prompt_endpoint,
        "derive_trusted_credential_scope",
        lambda *_args: (7, [], [], False),
    )
    monkeypatch.setattr(
        prompt_endpoint,
        "provider_requires_api_key",
        lambda _provider: False,
    )
    monkeypatch.setattr(
        prompt_endpoint,
        "require_project_access",
        _deny_project,
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _invoke_direct_prompt_endpoint(
            prompt_endpoint,
            prompt_id=1,
            user_id=7,
        )

    assert exc_info.value.status_code == 403
    assert runtime_inits == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("late_outcome", "expected_marks"),
    [("valid", 1), ("mixed-error", 0)],
)
async def test_direct_prompt_endpoint_cancellation_drains_before_mark_and_close(
    monkeypatch: pytest.MonkeyPatch,
    late_outcome: str,
    expected_marks: int,
) -> None:
    """Caller cancellation cannot abandon the credentialed adapter worker."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_prompts as prompt_endpoint,
    )
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
        prompt_executor as executor_module,
    )

    lifecycle: list[str] = []
    entered = threading.Event()
    release = threading.Event()
    pool = BoundedDaemonPool(capacity=1)
    sentinel = "direct-prompt-cancel-secret-/private/provider.json"
    handle = SimpleNamespace(
        provider="openai",
        api_key="cancel-key",
        app_config={},
        credentials_resolved=True,
    )

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            lifecycle.append("runtime-init")

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            lifecycle.append("resolve")
            return handle

        async def mark_used(self, selected: Any) -> None:
            assert selected is handle
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("close")

    class _Adapter:
        @staticmethod
        def chat(
            _request: dict[str, Any],
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            lifecycle.append("adapter-start")
            entered.set()
            assert release.wait(timeout=2.0)
            lifecycle.append("adapter-exit")
            if late_outcome == "valid":
                return {"choices": [{"message": {"content": "late valid"}}]}
            return {
                "choices": [
                    {"message": {"content": "apparently valid"}},
                    {"error": {"message": sentinel}},
                ]
            }

    class _Registry:
        @staticmethod
        def is_local_provider_name(_provider: str) -> bool:
            return False

        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

    _install_direct_prompt_runtime_seams(
        monkeypatch,
        prompt_endpoint,
        executor_module,
        _Runtime,
        pool,
    )
    monkeypatch.setattr(executor_module, "get_registry", lambda: _Registry())
    monkeypatch.setattr(executor_module, "ensure_app_config", lambda *_a, **_k: {})
    monkeypatch.setattr(
        executor_module,
        "resolve_provider_api_key_from_config",
        lambda *_a, **_k: "static-key",
    )
    drain_entered = _install_owned_worker_cancellation_ack(monkeypatch)

    task = asyncio.create_task(
        _invoke_direct_prompt_endpoint(
            prompt_endpoint,
            prompt_id=1,
            user_id=7,
        )
    )
    try:
        await _wait_for_thread_event(entered)
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        pending_until_worker_exit = not task.done()
        active_until_worker_exit = pool.active_count
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert pending_until_worker_exit is True
    assert active_until_worker_exit == 1
    assert pool.active_count == 0
    assert lifecycle.count("mark") == expected_marks
    assert lifecycle[-1] == "close"
    assert lifecycle.index("adapter-exit") < lifecycle.index("close")
    if expected_marks:
        assert lifecycle.index("adapter-exit") < lifecycle.index("mark")
        assert lifecycle.index("mark") < lifecycle.index("close")
    assert sentinel not in repr(lifecycle)


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_direct_prompt_endpoint_concurrent_scopes_and_marks_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent direct prompts cannot exchange credentials or usage marks."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_prompts as prompt_endpoint,
    )
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
        prompt_executor as executor_module,
    )

    pool = BoundedDaemonPool(capacity=2)
    entered = {name: threading.Event() for name in ("request-1", "request-2")}
    release = {name: threading.Event() for name in entered}
    adapter_requests: dict[str, dict[str, Any]] = {}
    lifecycle: list[tuple[str, int] | tuple[str, str]] = []
    sentinel = "direct-prompt-concurrent-secret-/private/provider.json"

    class _Runtime:
        def __init__(self, *, user_id: int, **_kwargs: Any) -> None:
            self.user_id = user_id
            lifecycle.append(("init", user_id))

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            key = f"scoped-key-{self.user_id}"
            lifecycle.append(("resolve", key))
            return SimpleNamespace(
                provider="openai",
                api_key=key,
                app_config={"openai_api": {"owner": self.user_id}},
                credentials_resolved=True,
            )

        async def mark_used(self, handle: Any) -> None:
            lifecycle.append(("mark", handle.api_key))

        async def close(self) -> None:
            lifecycle.append(("close", self.user_id))

    class _Adapter:
        @staticmethod
        def chat(
            request: dict[str, Any],
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            name = request["messages"][-1]["content"]
            adapter_requests[name] = dict(request)
            entered[name].set()
            assert release[name].wait(timeout=2.0)
            if name == "request-1":
                return {"choices": [{"message": {"content": "valid one"}}]}
            return {
                "choices": [
                    {"message": {"content": "apparently valid"}},
                    {"error": {"message": sentinel}},
                ]
            }

    class _Registry:
        @staticmethod
        def is_local_provider_name(_provider: str) -> bool:
            return False

        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

    _install_direct_prompt_runtime_seams(
        monkeypatch,
        prompt_endpoint,
        executor_module,
        _Runtime,
        pool,
    )
    monkeypatch.setattr(executor_module, "get_registry", lambda: _Registry())
    monkeypatch.setattr(
        executor_module,
        "ensure_app_config",
        lambda *_args, **_kwargs: {
            "openai_api": {"owner": "static", "api_key": "static-key"}
        },
    )
    monkeypatch.setattr(
        executor_module,
        "resolve_provider_api_key_from_config",
        lambda *_args, **_kwargs: "static-key",
    )

    first = asyncio.create_task(
        _invoke_direct_prompt_endpoint(prompt_endpoint, prompt_id=1, user_id=1)
    )
    second = asyncio.create_task(
        _invoke_direct_prompt_endpoint(prompt_endpoint, prompt_id=2, user_id=2)
    )
    try:
        await asyncio.gather(
            *(_wait_for_thread_event(event) for event in entered.values())
        )
        active_during_dispatch = pool.active_count
        release["request-2"].set()
        await asyncio.sleep(0.01)
        release["request-1"].set()
        results = await asyncio.wait_for(
            asyncio.gather(first, second, return_exceptions=True),
            timeout=2.0,
        )
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(first, second, return_exceptions=True)

    assert active_during_dispatch == 2
    assert pool.active_count == 0
    assert adapter_requests["request-1"]["api_key"] == "scoped-key-1"
    assert adapter_requests["request-1"]["app_config"] == {
        "openai_api": {"owner": 1}
    }
    assert adapter_requests["request-2"]["api_key"] == "scoped-key-2"
    assert adapter_requests["request-2"]["app_config"] == {
        "openai_api": {"owner": 2}
    }
    assert sorted(item for item in lifecycle if item[0] == "mark") == [
        ("mark", "scoped-key-1")
    ]
    assert sorted(item for item in lifecycle if item[0] == "close") == [
        ("close", 1),
        ("close", 2),
    ]
    assert isinstance(results[0], dict) and results[0]["output"] == "valid one"
    assert sentinel not in repr(results) + repr(lifecycle)


def test_evaluation_create_preserves_provider_in_current_and_legacy_configs() -> None:
    """Both accepted evaluation config shapes must retain provider identity."""
    current = EvaluationCreate(
        project_id=1,
        prompt_id=1,
        config={"provider": "anthropic", "model_name": "claude-test"},
    )
    legacy = EvaluationCreate(
        project_id=1,
        prompt_id=1,
        model_configs=[
            {
                "api_name": "oai",
                "model": "gpt-test",
                "api_key": "client-supplied-key-must-be-ignored",
            }
        ],
    )

    assert current.config is not None
    assert current.config.provider == "anthropic"
    assert legacy.model_configs is not None
    assert legacy.model_configs[0].api_name == "openai"
    assert legacy.model_configs[0].model == "gpt-test"
    assert "api_key" not in legacy.model_configs[0].model_dump()


def test_evaluation_create_rejects_explicit_empty_provider() -> None:
    """An explicit malformed provider must not silently fall back to OpenAI."""
    with pytest.raises(ValidationError):
        EvaluationCreate(
            project_id=1,
            prompt_id=1,
            config={"provider": "   ", "model_name": "model-a"},
        )


@pytest.mark.parametrize(
    ("field", "alias", "canonical"),
    [
        ("provider", "aws-bedrock", "bedrock"),
        ("api_name", "oai", "openai"),
    ],
)
def test_evaluation_create_canonicalizes_registered_provider_aliases(
    field: str,
    alias: str,
    canonical: str,
) -> None:
    evaluation = EvaluationCreate(
        project_id=1,
        prompt_id=1,
        config={field: alias, "model_name": "model-a"},
    )

    assert evaluation.config is not None
    assert getattr(evaluation.config, field) == canonical


def test_evaluation_create_rejects_unsupported_provider() -> None:
    with pytest.raises(ValidationError, match="Unsupported LLM provider"):
        EvaluationCreate(
            project_id=1,
            prompt_id=1,
            config={"provider": "not-a-provider", "model_name": "model-a"},
        )


@pytest.mark.parametrize("model_field", ["model_name", "model"])
def test_evaluation_create_rejects_explicit_blank_model(model_field: str) -> None:
    with pytest.raises(ValidationError, match="Model must not be empty"):
        EvaluationCreate(
            project_id=1,
            prompt_id=1,
            config={"provider": "openai", model_field: "   "},
        )


def test_evaluation_create_accepts_matching_canonical_alias_fields() -> None:
    evaluation = EvaluationCreate(
        project_id=1,
        prompt_id=1,
        config={
            "provider": "oai",
            "api_name": "openai",
            "model_name": "model-a",
            "model": "model-a",
        },
    )

    assert evaluation.config is not None
    assert evaluation.config.provider == "openai"
    assert evaluation.config.api_name == "openai"
    assert evaluation.config.model_name == "model-a"
    assert evaluation.config.model == "model-a"


@pytest.mark.parametrize(
    "config",
    [
        {"provider": "openai", "api_name": "anthropic", "model": "model-a"},
        {"provider": "openai", "model_name": "model-a", "model": "model-b"},
    ],
    ids=["provider-alias-conflict", "model-alias-conflict"],
)
def test_evaluation_create_rejects_conflicting_alias_fields(
    config: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="conflict"):
        EvaluationCreate(project_id=1, prompt_id=1, config=config)


def test_evaluation_create_accepts_semantically_identical_config_shapes() -> None:
    evaluation = EvaluationCreate(
        project_id=1,
        prompt_id=1,
        config={
            "provider": "oai",
            "model_name": "model-a",
            "timeout_seconds": 17,
        },
        model_configs=[
            {
                "api_name": "openai",
                "model": "model-a",
                "timeout_seconds": 17,
            }
        ],
    )

    assert evaluation.config is not None
    assert evaluation.model_configs is not None


@pytest.mark.parametrize(
    ("field", "default"),
    [("temperature", 0.7), ("max_tokens", 1000)],
)
def test_evaluation_create_accepts_explicit_and_implicit_execution_defaults(
    field: str,
    default: float | int,
) -> None:
    evaluation = EvaluationCreate(
        project_id=1,
        prompt_id=1,
        config={"provider": "openai", "model_name": "model-a", field: default},
        model_configs=[{"api_name": "openai", "model": "model-a"}],
    )

    assert evaluation.config is not None
    assert evaluation.model_configs is not None


@pytest.mark.parametrize(
    ("field", "value"),
    [("temperature", 0.8), ("max_tokens", 1001)],
)
def test_evaluation_create_rejects_real_execution_default_conflicts(
    field: str,
    value: float | int,
) -> None:
    with pytest.raises(ValidationError, match="config and model_configs conflict"):
        EvaluationCreate(
            project_id=1,
            prompt_id=1,
            config={"provider": "openai", "model_name": "model-a", field: value},
            model_configs=[{"api_name": "openai", "model": "model-a"}],
        )


def test_evaluation_create_rejects_contradictory_config_shapes() -> None:
    with pytest.raises(ValidationError, match="config and model_configs conflict"):
        EvaluationCreate(
            project_id=1,
            prompt_id=1,
            config={"provider": "openai", "model_name": "model-a"},
            model_configs=[{"api_name": "openai", "model": "model-b"}],
        )


def test_evaluation_create_requires_model_for_explicit_non_openai_provider() -> None:
    with pytest.raises(ValidationError, match="Model is required"):
        EvaluationCreate(
            project_id=1,
            prompt_id=1,
            config={"provider": "anthropic"},
        )


@pytest.mark.asyncio
async def test_prompt_studio_test_runner_drains_cancelled_sync_adapter(monkeypatch) -> None:
    """A cancelled runner must not abandon an adapter thread behind its caller."""
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner

    started = asyncio.Event()
    release = threading.Event()
    provider_finished = asyncio.Event()
    events: list[str] = []
    loop = asyncio.get_running_loop()
    drain_entered = _install_owned_worker_cancellation_ack(monkeypatch)

    class _RunnerDb:
        client_id = "test-client"

        def get_prompt(self, _prompt_id: int) -> dict[str, Any]:
            return {
                "id": 1,
                "project_id": 1,
                "deleted": False,
                "user_prompt": "Hello {name}",
            }

        def get_test_case(self, _test_case_id: int) -> dict[str, Any]:
            return {
                "id": 1,
                "project_id": 1,
                "inputs": {"name": "world"},
                "expected_outputs": {},
            }

    runner = TestRunner(_RunnerDb())

    def _blocking_adapter(**_kwargs: Any) -> dict[str, Any]:
        loop.call_soon_threadsafe(started.set)
        release.wait()
        loop.call_soon_threadsafe(provider_finished.set)
        events.append("adapter_done")
        return {"choices": [{"message": {"content": "ok"}}]}

    async def _mark_success() -> None:
        events.append("mark")

    monkeypatch.setattr(runner, "_call_adapter", _blocking_adapter)
    task = asyncio.create_task(
        runner.run_test_case(
            prompt_id=1,
            test_case_id=1,
            persist_run=False,
            on_provider_success=_mark_success,
        )
    )

    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    try:
        await asyncio.wait_for(drain_entered.wait(), timeout=2)
        assert task.done() is False
        assert provider_finished.is_set() is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        await asyncio.wait_for(provider_finished.wait(), timeout=2)
        assert events == ["adapter_done", "mark"]
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await task


class _StubCursor:
    def __init__(self):
        self.lastrowid = 1

    def execute(self, *args, **kwargs):

        # Do nothing; simulate successful INSERT
        self.lastrowid = 1


class _StubConn:
    def cursor(self):
        return _StubCursor()

    def commit(self):

        pass


class _StubDB:
    @staticmethod
    def get_prompt(prompt_id: int) -> dict[str, Any]:
        return {"id": prompt_id, "project_id": 1, "deleted": False}

    def get_connection(self):
        return _StubConn()


@pytest.fixture
def override_ps_deps(monkeypatch):
    async def _override_db():
        return _StubDB()

    async def _override_user():
        return {
            "user_id": "u",
            "client_id": "test-client",
            "is_authenticated": True,
            "is_admin": True,
            "permissions": ["all"],
        }

    app.dependency_overrides[deps.get_prompt_studio_db] = _override_db
    app.dependency_overrides[deps.get_prompt_studio_user] = _override_user
    try:
        yield
    finally:
        app.dependency_overrides.pop(deps.get_prompt_studio_db, None)
        app.dependency_overrides.pop(deps.get_prompt_studio_user, None)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "project_id": 1,
            "prompt_id": 1,
            "config": {
                "provider": "openai",
                "api_name": "anthropic",
                "model": "model-a",
            },
        },
        {
            "project_id": 1,
            "prompt_id": 1,
            "config": {
                "provider": "openai",
                "model_name": "model-a",
                "model": "model-b",
            },
        },
        {
            "project_id": 1,
            "prompt_id": 1,
            "config": {"provider": "openai", "model_name": "model-a"},
            "model_configs": [{"api_name": "openai", "model": "model-b"}],
        },
        {
            "project_id": 1,
            "prompt_id": 1,
            "config": {"provider": "anthropic"},
        },
    ],
    ids=[
        "provider-alias-conflict",
        "model-alias-conflict",
        "config-shape-conflict",
        "non-openai-model-required",
    ],
)
def test_evaluation_endpoint_rejects_ambiguous_provider_model_config_with_422(
    payload: dict[str, Any],
    override_ps_deps: None,
) -> None:
    del override_ps_deps

    response = TestClient(app).post(
        "/api/v1/prompt-studio/evaluations",
        json=payload,
        headers={"X-API-KEY": "test-key"},
    )

    assert response.status_code == 422, response.text


def test_evaluation_async_add_task_receives_request_id(monkeypatch, override_ps_deps):


    # Force scheduling branch (not inline) by removing PyTest env hint and disabling TEST_MODE
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("TEST_MODE", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    # Capture add_task call arguments
    captured = {"func": None, "args": None, "kwargs": None}

    from fastapi.background import BackgroundTasks as _BT

    def fake_add_task(self, func, *args, **kwargs):  # noqa: D401
        captured["func"] = func
        captured["args"] = args
        captured["kwargs"] = kwargs
        # Do not schedule to avoid executing background work in tests
        return None

    monkeypatch.setattr(_BT, "add_task", fake_add_task, raising=True)

    # Also patch the runner to a noop (defensive)
    import tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_evaluations as eval_ep

    async def noop_run(*a, **kw):
        return None

    monkeypatch.setattr(eval_ep, "run_evaluation_async", noop_run, raising=True)

    class _IsolatedRuntime:
        """Keep this scheduling-only test away from process-global BYOK state."""

        async def resolve(self, _provider: str, *, model: str | None = None):
            return SimpleNamespace(
                api_key="test-openai-key",
                app_config={"openai_api": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _credentials: object) -> None:
            return None

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        eval_ep,
        "ProviderCredentialRuntime",
        lambda **_kwargs: _IsolatedRuntime(),
        raising=True,
    )

    client = TestClient(app)
    r = client.post(
        "/api/v1/prompt-studio/evaluations",
        json={
            "project_id": 1,
            "prompt_id": 1,
            "name": "Async Eval",
            "test_case_ids": [],
            "config": {"model_name": "gpt-4o-mini"},
            "run_async": True,
        },
        headers={
            "X-Request-ID": "req-eval-xyz",
            "X-API-KEY": "test-key",
        },
    )
    assert r.status_code == 200, r.text
    # Ensure add_task was called and received propagated identifiers
    assert captured["func"] is not None
    # First positional args: (evaluation_id, db)
    assert isinstance(captured["args"], tuple) and len(captured["args"]) >= 2
    assert captured["kwargs"].get("request_id") == "req-eval-xyz"
    # traceparent may be empty if not provided; verify kwarg exists
    assert "traceparent" in captured["kwargs"]


class _SnapshotCursor:
    def __init__(self) -> None:
        self.lastrowid = 17
        self._row: tuple[Any, ...] | None = None

    def execute(self, query: str, *_args: Any) -> None:
        if "SELECT id, project_id, prompt_id, test_case_ids, model_configs" in query:
            self._row = (
                17,
                1,
                1,
                "[1]",
                '[{"model_name":"model-a"}]',
            )

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row


class _SnapshotConnection:
    def __init__(self) -> None:
        self.cursor_instance = _SnapshotCursor()

    def cursor(self) -> _SnapshotCursor:
        return self.cursor_instance

    def commit(self) -> None:
        return None


class _SnapshotDb:
    client_id = "test-client"

    def __init__(self) -> None:
        self.connection = _SnapshotConnection()

    def get_connection(self) -> _SnapshotConnection:
        return self.connection

    @staticmethod
    def get_project(project_id: int) -> dict[str, Any]:
        return {"id": project_id, "user_id": "1"}

    def get_prompt(self, _prompt_id: int) -> dict[str, Any]:
        return {
            "id": 1,
            "project_id": 1,
            "deleted": False,
            "user_prompt": "Hello {name}",
        }

    def get_test_case(self, _test_case_id: int) -> dict[str, Any]:
        return {
            "id": 1,
            "project_id": 1,
            "inputs": {"name": "world"},
            "expected_outputs": {"response": "ok"},
        }

    def get_test_cases_by_ids(self, _test_case_ids: list[int]) -> list[dict[str, Any]]:
        return [{"id": 1, "inputs": {}, "expected_outputs": {}}]

    def get_evaluation(self, _evaluation_id: int) -> dict[str, Any]:
        return {"id": 17, "uuid": "00000000-0000-0000-0000-000000000017"}

    def create_evaluation(self, **_kwargs: Any) -> dict[str, Any]:
        return {"id": 17, "uuid": "00000000-0000-0000-0000-000000000017"}

    def update_evaluation(self, _evaluation_id: int, _updates: dict[str, Any]) -> None:
        return None


class _CapturedBackgroundTasks:
    def __init__(self) -> None:
        self.task: tuple[Any, tuple[Any, ...], dict[str, Any]] | None = None

    def add_task(self, func, *args: Any, **kwargs: Any) -> None:
        self.task = (func, args, kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "background"])
async def test_evaluation_endpoint_retries_explicit_false_provider_mark(
    monkeypatch: pytest.MonkeyPatch,
    run_async: bool,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_evaluations as eval_endpoint,
    )

    lifecycle: list[str] = []
    handle = SimpleNamespace(
        api_key="evaluation-retry-key",
        app_config={"openai_api": {"model": "model-a"}},
        credentials_resolved=True,
    )

    class _Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, selected: Any) -> bool:
            assert selected is handle
            lifecycle.append("mark")
            return lifecycle.count("mark") > 1

        async def close(self) -> None:
            lifecycle.append("close")

    class _Manager:
        def __init__(self, _db: Any) -> None:
            return None

        async def run_evaluation_async(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["provider_credentials"] is handle
            await kwargs["on_provider_success"]()
            return {
                "id": 17,
                "uuid": "00000000-0000-0000-0000-000000000017",
                "prompt_id": 1,
                "status": "completed",
                "metrics": {},
            }

        async def run_evaluation_with_existing_record(
            self,
            **kwargs: Any,
        ) -> dict[str, Any]:
            assert kwargs["provider_credentials"] is handle
            await kwargs["on_provider_success"]()
            return {"metrics": {"total_tests": 1, "pass_rate": 1.0}}

    runtime = _Runtime()
    monkeypatch.setattr(
        eval_endpoint,
        "ProviderCredentialRuntime",
        lambda **_kwargs: runtime,
    )
    monkeypatch.setattr(eval_endpoint, "EvaluationManager", _Manager)
    monkeypatch.setattr(
        eval_endpoint,
        "derive_trusted_credential_scope",
        lambda *_args: (1, [], [], False),
    )
    monkeypatch.setattr(
        eval_endpoint,
        "provider_requires_api_key",
        lambda _provider: False,
    )
    monkeypatch.setattr(eval_endpoint, "_is_prompt_studio_test_mode", lambda: False)
    monkeypatch.setattr(eval_endpoint, "is_test_mode", lambda: False)
    monkeypatch.setattr(eval_endpoint, "ensure_request_id", lambda _request: "req-1")
    monkeypatch.setattr(eval_endpoint, "ensure_traceparent", lambda _request: "")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    background_tasks = _CapturedBackgroundTasks()
    await eval_endpoint.create_evaluation(
        evaluation=EvaluationCreate(
            project_id=1,
            prompt_id=1,
            test_case_ids=[1],
            config={"provider": "openai", "model_name": "model-a"},
            run_async=run_async,
        ),
        background_tasks=background_tasks,  # type: ignore[arg-type]
        request=object(),  # type: ignore[arg-type]
        db=_SnapshotDb(),  # type: ignore[arg-type]
        user_context={"user_id": "1", "client_id": "test-client"},
    )

    if run_async:
        assert background_tasks.task is not None
        func, args, kwargs = background_tasks.task
        await func(*args, **kwargs)

    assert lifecycle == ["mark", "mark", "close"]


@pytest.mark.asyncio
async def test_evaluation_binds_prompt_and_cases_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_evaluations as eval_endpoint,
    )

    class _CrossProjectDb(_SnapshotDb):
        @staticmethod
        def get_project(project_id: int) -> dict[str, Any]:
            return {"id": project_id, "user_id": "1"}

        def get_prompt(self, _prompt_id: int) -> dict[str, Any]:
            return {"id": 1, "project_id": 2, "deleted": False}

        def get_test_case(self, _test_case_id: int) -> dict[str, Any]:
            return {"id": 1, "project_id": 3, "deleted": False}

        def get_test_cases_by_ids(
            self,
            _test_case_ids: list[int],
        ) -> list[dict[str, Any]]:
            return [{"id": 1, "project_id": 3, "deleted": False}]

    runtime_inits: list[bool] = []
    handle = SimpleNamespace(
        api_key="evaluation-key",
        app_config={"openai_api": {"model": "model-a"}},
        credentials_resolved=True,
    )

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            runtime_inits.append(True)

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, _handle: Any) -> bool:
            return True

        async def close(self) -> None:
            return None

    class _Manager:
        def __init__(self, _db: Any) -> None:
            return None

        async def run_evaluation_async(self, **kwargs: Any) -> dict[str, Any]:
            await kwargs["on_provider_success"]()
            return {
                "id": 17,
                "uuid": "00000000-0000-0000-0000-000000000017",
                "prompt_id": 1,
                "status": "completed",
                "metrics": {},
            }

    monkeypatch.setattr(eval_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(eval_endpoint, "EvaluationManager", _Manager)
    monkeypatch.setattr(
        eval_endpoint,
        "derive_trusted_credential_scope",
        lambda *_args: (1, [], [], False),
    )
    monkeypatch.setattr(
        eval_endpoint,
        "provider_requires_api_key",
        lambda _provider: False,
    )

    with pytest.raises(HTTPException):
        await eval_endpoint.create_evaluation(
            evaluation=EvaluationCreate(
                project_id=1,
                prompt_id=1,
                test_case_ids=[1],
                config={"model_name": "model-a"},
            ),
            background_tasks=_CapturedBackgroundTasks(),  # type: ignore[arg-type]
            request=object(),  # type: ignore[arg-type]
            db=_CrossProjectDb(),  # type: ignore[arg-type]
            user_context={
                "user_id": "1",
                "client_id": "audit-client",
                "is_admin": False,
            },
        )

    assert runtime_inits == []


class _TrackingPromptRuntime:
    def __init__(
        self,
        events: list[str],
        handle: object,
        runtime_closed: asyncio.Event | None = None,
    ) -> None:
        self.events = events
        self.handle = handle
        self.closed = False
        self.runtime_closed = runtime_closed

    async def resolve(self, _provider: str, *, model: str | None = None):
        _ = model
        return self.handle

    async def mark_used(self, handle: object) -> None:
        assert handle is self.handle
        self.events.append("mark")

    async def close(self) -> None:
        self.closed = True
        self.events.append("close")
        if self.runtime_closed is not None:
            self.runtime_closed.set()


@pytest.mark.asyncio
async def test_prompt_studio_sync_cancellation_stops_before_next_provider_call(monkeypatch) -> None:
    """Sync cancellation drains one adapter call without continuing the batch."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_evaluations as eval_endpoint,
    )
    from tldw_Server_API.app.core.Chat.bounded_daemon import await_owned_worker

    events: list[str] = []
    dispatches: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()
    provider_finished = asyncio.Event()
    runtime_closed = asyncio.Event()
    drain_entered = _install_owned_worker_cancellation_ack(monkeypatch)
    handle = type(
        "Handle",
        (),
        {
            "api_key": "prompt-key",
            "app_config": {"openai_api": {"model": "model-a"}},
            "credentials_resolved": True,
        },
    )()
    runtime = _TrackingPromptRuntime(events, handle, runtime_closed)

    class _Manager:
        def __init__(self, _db: object) -> None:
            return None

        async def run_evaluation_async(self, **kwargs: Any) -> dict[str, Any]:
            async def _provider_call() -> dict[str, Any]:
                started.set()
                await release.wait()
                provider_finished.set()
                dispatches.append("first")
                events.append("provider_done")
                return {
                    "id": 17,
                    "uuid": "00000000-0000-0000-0000-000000000017",
                    "prompt_id": 1,
                    "status": "completed",
                    "metrics": {},
                }

            on_provider_success = kwargs.get("on_provider_success")
            result = await await_owned_worker(
                _provider_call(),
                on_cancel_success=on_provider_success,
            )
            if on_provider_success is not None:
                await on_provider_success()
            dispatches.append("second")
            return result

    monkeypatch.setattr(eval_endpoint, "ProviderCredentialRuntime", lambda **_kwargs: runtime)
    monkeypatch.setattr(eval_endpoint, "EvaluationManager", _Manager)
    monkeypatch.setattr(eval_endpoint, "derive_trusted_credential_scope", lambda *_args: (1, [], [], False))
    monkeypatch.setattr(eval_endpoint, "provider_requires_api_key", lambda _provider: True)
    monkeypatch.setattr(eval_endpoint, "_is_prompt_studio_test_mode", lambda: False)

    task = asyncio.create_task(
        eval_endpoint.create_evaluation(
            evaluation=EvaluationCreate(
                project_id=1,
                prompt_id=1,
                test_case_ids=[1],
                config={"model_name": "model-a"},
            ),
            background_tasks=_CapturedBackgroundTasks(),  # type: ignore[arg-type]
            request=object(),  # type: ignore[arg-type]
            db=_SnapshotDb(),  # type: ignore[arg-type]
            user_context={"user_id": "1", "client_id": "test-client"},
        )
    )

    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    try:
        await asyncio.wait_for(drain_entered.wait(), timeout=2)
        assert provider_finished.is_set() is False
        assert runtime_closed.is_set() is False
        assert runtime.closed is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        await asyncio.wait_for(provider_finished.wait(), timeout=2)
        await asyncio.wait_for(runtime_closed.wait(), timeout=2)
        assert dispatches == ["first"]
        assert events == ["provider_done", "mark", "close"]
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await task


@pytest.mark.asyncio
async def test_prompt_studio_background_cancellation_stops_before_next_provider_call(monkeypatch) -> None:
    """Background cancellation drains one call without continuing the batch."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_evaluations as eval_endpoint,
    )
    from tldw_Server_API.app.core.Chat.bounded_daemon import await_owned_worker

    events: list[str] = []
    dispatches: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()
    provider_finished = asyncio.Event()
    runtime_closed = asyncio.Event()
    drain_entered = _install_owned_worker_cancellation_ack(monkeypatch)
    handle = type(
        "Handle",
        (),
        {
            "api_key": "prompt-key",
            "app_config": {"openai_api": {"model": "model-a"}},
            "credentials_resolved": True,
        },
    )()
    runtime = _TrackingPromptRuntime(events, handle, runtime_closed)

    class _Manager:
        def __init__(self, _db: object) -> None:
            return None

        async def run_evaluation_with_existing_record(self, **kwargs: Any) -> dict[str, Any]:
            async def _provider_call() -> dict[str, Any]:
                started.set()
                await release.wait()
                provider_finished.set()
                dispatches.append("first")
                events.append("provider_done")
                return {"metrics": {"total_tests": 1, "pass_rate": 1.0}}

            on_provider_success = kwargs.get("on_provider_success")
            result = await await_owned_worker(
                _provider_call(),
                on_cancel_success=on_provider_success,
            )
            if on_provider_success is not None:
                await on_provider_success()
            dispatches.append("second")
            return result

    monkeypatch.setattr(eval_endpoint, "EvaluationManager", _Manager)
    monkeypatch.setattr(eval_endpoint, "provider_requires_api_key", lambda _provider: True)
    monkeypatch.setattr(eval_endpoint, "_is_prompt_studio_test_mode", lambda: False)

    task = asyncio.create_task(
        eval_endpoint.run_evaluation_async(
            17,
            _SnapshotDb(),  # type: ignore[arg-type]
            user_id=1,
            provider="openai",
            model="model-a",
            credential_runtime=runtime,  # type: ignore[arg-type]
            provider_credentials=handle,  # type: ignore[arg-type]
        )
    )

    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    try:
        await asyncio.wait_for(drain_entered.wait(), timeout=2)
        assert provider_finished.is_set() is False
        assert runtime_closed.is_set() is False
        assert runtime.closed is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        await asyncio.wait_for(provider_finished.wait(), timeout=2)
        await asyncio.wait_for(runtime_closed.wait(), timeout=2)
        assert dispatches == ["first"]
        assert events == ["provider_done", "mark", "close"]
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await task


@pytest.mark.asyncio
async def test_prompt_studio_background_cancel_after_running_is_persisted_and_closes_runtime(
    monkeypatch: pytest.MonkeyPatch,
    isolated_db: Any,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_evaluations as eval_endpoint,
    )

    events: list[str] = []
    handle = SimpleNamespace(
        api_key="prompt-key",
        app_config={"openai_api": {"model": "model-a"}},
        credentials_resolved=True,
    )
    runtime = _TrackingPromptRuntime(events, handle)

    class _CancelledManager:
        def __init__(self, _db: object) -> None:
            return None

        async def run_evaluation_with_existing_record(
            self,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            raise asyncio.CancelledError()

    monkeypatch.setattr(eval_endpoint, "EvaluationManager", _CancelledManager)
    project = isolated_db.create_project(
        "Cancelled background evaluation",
        user_id="owner-1",
    )
    prompt = isolated_db.create_prompt(
        int(project["id"]),
        "Cancellation target",
        user_prompt="Hello",
    )
    evaluation = isolated_db.create_evaluation(
        project_id=int(project["id"]),
        prompt_id=int(prompt["id"]),
        model_configs={"provider": "openai", "model_name": "model-a"},
        test_case_ids=[],
        status="pending",
    )

    with pytest.raises(asyncio.CancelledError):
        await eval_endpoint.run_evaluation_async(
            int(evaluation["id"]),
            isolated_db,
            user_id=1,
            provider="openai",
            model="model-a",
            credential_runtime=runtime,  # type: ignore[arg-type]
            provider_credentials=handle,  # type: ignore[arg-type]
        )

    persisted = isolated_db.get_evaluation(int(evaluation["id"]))
    assert persisted is not None
    assert persisted["status"] == "cancelled"
    assert events == ["close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "background"])
@pytest.mark.parametrize(
    ("runtime_auth_source", "expected_status"),
    [("aws_default_chain", None), (None, 503)],
    ids=["default-chain", "explicit-absent"],
)
async def test_prompt_studio_bedrock_auth_contract(
    monkeypatch: pytest.MonkeyPatch,
    run_async: bool,
    runtime_auth_source: str | None,
    expected_status: int | None,
) -> None:
    """Sync and deferred paths must distinguish Bedrock default auth from absence."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_evaluations as eval_endpoint,
    )

    provider_config: dict[str, Any] = {"model": "bedrock-model"}
    if runtime_auth_source is not None:
        provider_config["_runtime_auth_source"] = runtime_auth_source
    handle = type(
        "Handle",
        (),
        {
            "api_key": None,
            "app_config": {"bedrock_api": provider_config},
            "credentials_resolved": True,
        },
    )()
    lifecycle: list[str] = []
    runtime = _TrackingPromptRuntime(lifecycle, handle)

    class _Manager:
        def __init__(self, _db: object) -> None:
            return None

        async def run_evaluation_async(self, **kwargs: Any) -> dict[str, Any]:
            on_provider_success = kwargs.get("on_provider_success")
            if on_provider_success is not None:
                await on_provider_success()
            return {
                "id": 17,
                "uuid": "00000000-0000-0000-0000-000000000017",
                "prompt_id": 1,
                "status": "completed",
                "metrics": {},
            }

        async def run_evaluation_with_existing_record(self, **kwargs: Any) -> dict[str, Any]:
            on_provider_success = kwargs.get("on_provider_success")
            if on_provider_success is not None:
                await on_provider_success()
            return {"metrics": {"total_tests": 1, "pass_rate": 1.0}}

    monkeypatch.setattr(eval_endpoint, "ProviderCredentialRuntime", lambda **_kwargs: runtime)
    monkeypatch.setattr(eval_endpoint, "EvaluationManager", _Manager)
    monkeypatch.setattr(eval_endpoint, "derive_trusted_credential_scope", lambda *_args: (1, [], [], False))
    monkeypatch.setattr(eval_endpoint, "provider_requires_api_key", lambda _provider: True)
    monkeypatch.setattr(eval_endpoint, "_is_prompt_studio_test_mode", lambda: False)
    monkeypatch.setattr(eval_endpoint, "is_test_mode", lambda: False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    evaluation = EvaluationCreate.model_construct(
        project_id=1,
        prompt_id=1,
        test_case_ids=[1],
        model_configs=[{"provider": "bedrock", "model": "bedrock-model"}],
        run_async=run_async,
        metrics=None,
        config=None,
        name=None,
        description=None,
    )
    background_tasks = _CapturedBackgroundTasks()

    if expected_status is not None:
        with pytest.raises(eval_endpoint.HTTPException) as exc_info:
            await eval_endpoint.create_evaluation(
                evaluation=evaluation,
                background_tasks=background_tasks,  # type: ignore[arg-type]
                request=object(),  # type: ignore[arg-type]
                db=_SnapshotDb(),  # type: ignore[arg-type]
                user_context={"user_id": "1", "client_id": "test-client"},
            )
        assert exc_info.value.status_code == expected_status
        assert lifecycle == ["close"]
        return

    await eval_endpoint.create_evaluation(
        evaluation=evaluation,
        background_tasks=background_tasks,  # type: ignore[arg-type]
        request=object(),  # type: ignore[arg-type]
        db=_SnapshotDb(),  # type: ignore[arg-type]
        user_context={"user_id": "1", "client_id": "test-client"},
    )
    if run_async:
        assert background_tasks.task is not None
        func, args, kwargs = background_tasks.task
        await func(*args, **kwargs)
    assert lifecycle == ["mark", "close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "background"])
@pytest.mark.parametrize(
    ("captured_key", "captured_config"),
    [
        (
            "prompt-key-a",
            {"anthropic_api": {"model": "model-a", "api_key": "config-key-a"}},
        ),
        (None, None),
    ],
    ids=["a-to-b", "absent-config-to-b"],
)
async def test_prompt_studio_keeps_request_snapshot_at_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
    run_async: bool,
    captured_key: str | None,
    captured_config: dict[str, Any] | None,
) -> None:
    """Sync and deferred evaluations must dispatch the request-captured snapshot."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_evaluations as eval_endpoint,
    )
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
        evaluation_manager,
        test_runner,
    )

    adapter_requests: list[tuple[dict[str, Any], float | None]] = []
    adapter_providers: list[str] = []
    lifecycle: list[object] = []

    class FakeRuntime:
        def __init__(self, **kwargs: Any) -> None:
            lifecycle.append(("init", kwargs))

        async def resolve(self, provider: str, *, model: str | None = None):
            lifecycle.append(("resolve", provider, model))
            return type(
                "Handle",
                (),
                {
                    "api_key": captured_key,
                    "app_config": captured_config,
                    "credentials_resolved": True,
                },
            )()

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("close")

    async def forbidden_low_level_resolver(*_args: Any, **_kwargs: Any):
        raise AssertionError("Prompt Studio bypassed ProviderCredentialRuntime")

    class RecordingAdapter:
        def chat(self, request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
            adapter_requests.append((dict(request), timeout))
            return {"choices": [{"message": {"content": "ok"}}]}

    async def run_single_test(self, *, model_config: dict[str, Any], **kwargs: Any):
        params = model_config["parameters"]
        await asyncio.to_thread(
            self._call_adapter,
            provider=model_config["provider"],
            model=model_config["model"],
            messages_payload=[{"role": "user", "content": "test"}],
            system_message=None,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            app_config=model_config["app_config"],
            api_key_override=model_config["api_key"],
            credentials_resolved=model_config["credentials_resolved"],
            timeout_seconds=model_config["timeout_seconds"],
        )
        on_provider_success = kwargs.get("on_provider_success")
        if on_provider_success is not None:
            await on_provider_success()
        return {"id": None, "scores": {"aggregate_score": 1.0}, "actual": {"response": "ok"}}

    monkeypatch.setattr(eval_endpoint, "ProviderCredentialRuntime", FakeRuntime)
    monkeypatch.setattr(
        eval_endpoint,
        "derive_trusted_credential_scope",
        lambda _request, _user: (1, [2], [3], True),
    )
    monkeypatch.setattr(
        eval_endpoint,
        "resolve_byok_credentials",
        forbidden_low_level_resolver,
        raising=False,
    )
    monkeypatch.setattr(
        eval_endpoint,
        "resolve_provider_api_key",
        lambda *_a, **_k: ("prompt-key-b", {}),
        raising=False,
    )
    monkeypatch.setattr(eval_endpoint, "provider_requires_api_key", lambda _provider: False)
    monkeypatch.setattr(eval_endpoint, "_is_prompt_studio_test_mode", lambda: False)
    monkeypatch.setattr(eval_endpoint, "is_test_mode", lambda: False)
    monkeypatch.setattr(eval_endpoint, "ensure_request_id", lambda _request: "req-1")
    monkeypatch.setattr(eval_endpoint, "ensure_traceparent", lambda _request: "")
    monkeypatch.setattr(evaluation_manager, "is_test_mode", lambda: False)
    monkeypatch.setattr(evaluation_manager, "resolve_provider_api_key_from_config", lambda *_a: "prompt-key-b")
    monkeypatch.setattr(
        evaluation_manager,
        "ensure_app_config",
        lambda config: config
        if config is not None
        else {
            "anthropic_api": {
                "model": "model-b",
                "api_key": "prompt-key-b",
            }
        },
    )
    def get_recording_adapter(provider: str) -> RecordingAdapter:
        adapter_providers.append(provider)
        return RecordingAdapter()

    monkeypatch.setattr(evaluation_manager, "get_adapter_or_raise", get_recording_adapter)
    monkeypatch.setattr(test_runner.TestRunner, "run_single_test", run_single_test)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    background_tasks = _CapturedBackgroundTasks()
    await eval_endpoint.create_evaluation(
        evaluation=EvaluationCreate(
            project_id=1,
            prompt_id=1,
            test_case_ids=[1],
            model_configs=[
                {
                    "api_name": "anthropic",
                    "model": "model-a",
                    "timeout_seconds": 37,
                    "api_key": "client-supplied-key-must-not-dispatch",
                }
            ],
            run_async=run_async,
        ),
        background_tasks=background_tasks,  # type: ignore[arg-type]
        request=object(),  # type: ignore[arg-type]
        db=_SnapshotDb(),  # type: ignore[arg-type]
        user_context={"user_id": "1", "client_id": "test-client"},
    )

    if run_async:
        assert background_tasks.task is not None
        func, args, kwargs = background_tasks.task
        await func(*args, **kwargs)

    assert adapter_requests
    assert all(request["api_key"] == captured_key for request, _timeout in adapter_requests)
    assert all(
        request["app_config"] == (captured_config or {})
        for request, _timeout in adapter_requests
    )
    assert all(
        request["credentials_resolved"] is True
        for request, _timeout in adapter_requests
    )
    assert [timeout for _request, timeout in adapter_requests] == [37]
    assert adapter_providers == ["anthropic"]
    init_kwargs = lifecycle[0][1]
    assert init_kwargs["user_id"] == 1
    assert init_kwargs["team_ids"] == [2]
    assert init_kwargs["org_ids"] == [3]
    assert init_kwargs["trusted_base_url_override"] is True
    assert lifecycle[1:] == [
        ("resolve", "anthropic", "model-a"),
        "mark_used",
        "close",
    ]


@pytest.mark.asyncio
async def test_prompt_studio_concurrent_adapter_snapshots_remain_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent evaluations must not exchange captured keys or app configs."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_evaluations as eval_endpoint,
    )
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
        evaluation_manager,
        test_runner,
    )

    loop = asyncio.get_running_loop()
    release = threading.Event()
    both_entered = asyncio.Event()
    entered_lock = threading.Lock()
    adapter_requests: list[tuple[str, str | None, dict[str, Any], float | None]] = []
    lifecycle: list[tuple[str, str]] = []
    snapshots = {
        "model-a": (
            "key-a",
            {"anthropic_api": {"model": "model-a", "base_url": "https://a.invalid"}},
            11,
        ),
        "model-b": ("key-b", None, 29),
    }

    class FakeRuntime:
        def __init__(self, **_kwargs: Any) -> None:
            self.model = ""
            self.handle: object | None = None

        async def resolve(self, provider: str, *, model: str | None = None):
            assert provider == "anthropic"
            assert model in snapshots
            self.model = str(model)
            key, app_config, _timeout = snapshots[self.model]
            self.handle = type(
                "Handle",
                (),
                {
                    "api_key": key,
                    "app_config": app_config,
                    "credentials_resolved": True,
                },
            )()
            lifecycle.append(("resolve", self.model))
            return self.handle

        async def mark_used(self, handle: object) -> None:
            assert handle is self.handle
            lifecycle.append(("mark", self.model))

        async def close(self) -> None:
            lifecycle.append(("close", self.model))

    class RecordingAdapter:
        def chat(self, request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
            with entered_lock:
                adapter_requests.append(
                    (
                        request["model"],
                        request["api_key"],
                        request["app_config"],
                        timeout,
                    )
                )
                if len(adapter_requests) == 2:
                    loop.call_soon_threadsafe(both_entered.set)
            release.wait()
            return {"choices": [{"message": {"content": request["model"]}}]}

    async def run_single_test(self, *, model_config: dict[str, Any], **kwargs: Any):
        params = model_config["parameters"]
        await asyncio.to_thread(
            self._call_adapter,
            provider=model_config["provider"],
            model=model_config["model"],
            messages_payload=[{"role": "user", "content": "test"}],
            system_message=None,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            app_config=model_config["app_config"],
            api_key_override=model_config["api_key"],
            credentials_resolved=model_config["credentials_resolved"],
            timeout_seconds=model_config["timeout_seconds"],
        )
        on_provider_success = kwargs.get("on_provider_success")
        if on_provider_success is not None:
            await on_provider_success()
        return {
            "id": None,
            "scores": {"aggregate_score": 1.0},
            "actual": {"response": model_config["model"]},
        }

    monkeypatch.setattr(eval_endpoint, "ProviderCredentialRuntime", FakeRuntime)
    monkeypatch.setattr(
        eval_endpoint,
        "derive_trusted_credential_scope",
        lambda *_args: (1, [], [], False),
    )
    monkeypatch.setattr(eval_endpoint, "provider_requires_api_key", lambda _provider: True)
    monkeypatch.setattr(eval_endpoint, "_is_prompt_studio_test_mode", lambda: False)
    monkeypatch.setattr(evaluation_manager, "is_test_mode", lambda: False)
    monkeypatch.setattr(
        evaluation_manager,
        "ensure_app_config",
        lambda config: config
        if config is not None
        else {
            "anthropic_api": {
                "model": "live-model-b",
                "base_url": "https://live-b.invalid",
            }
        },
    )
    monkeypatch.setattr(
        evaluation_manager,
        "get_adapter_or_raise",
        lambda provider: RecordingAdapter() if provider == "anthropic" else None,
    )
    monkeypatch.setattr(test_runner.TestRunner, "run_single_test", run_single_test)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    tasks = [
        asyncio.create_task(
            eval_endpoint.create_evaluation(
                evaluation=EvaluationCreate(
                    project_id=1,
                    prompt_id=1,
                    test_case_ids=[1],
                    config={
                        "provider": "anthropic",
                        "model_name": model,
                        "timeout_seconds": snapshots[model][2],
                    },
                ),
                background_tasks=_CapturedBackgroundTasks(),  # type: ignore[arg-type]
                request=object(),  # type: ignore[arg-type]
                db=_SnapshotDb(),  # type: ignore[arg-type]
                user_context={"user_id": "1", "client_id": "test-client"},
            )
        )
        for model in snapshots
    ]
    try:
        await asyncio.wait_for(both_entered.wait(), timeout=2)
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.gather(*tasks)

    assert sorted(adapter_requests) == sorted(
        (model, key, app_config or {}, timeout)
        for model, (key, app_config, timeout) in snapshots.items()
    )
    assert sorted(lifecycle) == sorted(
        (event, model)
        for model in snapshots
        for event in ("resolve", "mark", "close")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "background"])
async def test_prompt_studio_timeout_drains_adapter_before_runtime_close_without_mark(
    monkeypatch: pytest.MonkeyPatch,
    run_async: bool,
) -> None:
    """A configured adapter timeout remains owned and is mapped as one failed case."""
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_evaluations as eval_endpoint,
    )
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
        evaluation_manager,
        test_runner,
    )

    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    events: list[str] = []
    adapter_timeouts: list[float | None] = []
    case_results: list[dict[str, Any]] = []
    sentinel = "prompt-studio-timeout-secret-sentinel"
    handle = type(
        "Handle",
        (),
        {
            "api_key": "snapshot-key",
            "app_config": {"anthropic_api": {"model": "model-a"}},
            "credentials_resolved": True,
        },
    )()
    runtime = _TrackingPromptRuntime(events, handle)

    class _TimeoutAdapter:
        def chat(
            self,
            _request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            adapter_timeouts.append(timeout)
            loop.call_soon_threadsafe(started.set)
            release.wait()
            events.append("adapter_done")
            raise TimeoutError(sentinel)

    async def _run_single_test(
        self,
        *,
        prompt_id: int,
        test_case_id: int,
        model_config: dict[str, Any],
        on_provider_success=None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        params = model_config["parameters"]
        result = await self.run_test_case(
            prompt_id=prompt_id,
            test_case_id=test_case_id,
            model=model_config["model"],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            provider=model_config["provider"],
            app_config=model_config["app_config"],
            api_key_override=model_config["api_key"],
            credentials_resolved=model_config["credentials_resolved"],
            timeout_seconds=model_config["timeout_seconds"],
            persist_run=False,
            on_provider_success=on_provider_success,
        )
        case_results.append(result)
        return {
            **result,
            "id": None,
            "success": False,
            "scores": {"aggregate_score": 0.0},
        }

    monkeypatch.setattr(eval_endpoint, "ProviderCredentialRuntime", lambda **_kwargs: runtime)
    monkeypatch.setattr(
        eval_endpoint,
        "derive_trusted_credential_scope",
        lambda *_args: (1, [], [], False),
    )
    monkeypatch.setattr(eval_endpoint, "provider_requires_api_key", lambda _provider: False)
    monkeypatch.setattr(eval_endpoint, "_is_prompt_studio_test_mode", lambda: False)
    monkeypatch.setattr(eval_endpoint, "is_test_mode", lambda: False)
    monkeypatch.setattr(eval_endpoint, "ensure_request_id", lambda _request: "req-timeout")
    monkeypatch.setattr(eval_endpoint, "ensure_traceparent", lambda _request: "")
    monkeypatch.setattr(evaluation_manager, "is_test_mode", lambda: False)
    monkeypatch.setattr(evaluation_manager, "get_adapter_or_raise", lambda _provider: _TimeoutAdapter())
    monkeypatch.setattr(test_runner.TestRunner, "run_single_test", _run_single_test)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    background_tasks = _CapturedBackgroundTasks()
    create_kwargs = {
        "evaluation": EvaluationCreate(
            project_id=1,
            prompt_id=1,
            test_case_ids=[1],
            config={
                "provider": "anthropic",
                "model_name": "model-a",
                "timeout_seconds": 23,
            },
            run_async=run_async,
        ),
        "background_tasks": background_tasks,
        "request": object(),
        "db": _SnapshotDb(),
        "user_context": {"user_id": "1", "client_id": "test-client"},
    }

    operation: asyncio.Task[Any] | None = None
    try:
        if run_async:
            response = await eval_endpoint.create_evaluation(**create_kwargs)
            assert response.status == "running"
            assert background_tasks.task is not None
            func, args, kwargs = background_tasks.task
            operation = asyncio.create_task(func(*args, **kwargs))
        else:
            operation = asyncio.create_task(eval_endpoint.create_evaluation(**create_kwargs))

        await asyncio.wait_for(started.wait(), timeout=2)
        assert runtime.closed is False
        assert events == []

        release.set()
        result = await asyncio.wait_for(operation, timeout=2)
        if not run_async:
            assert result.status == "completed"
    finally:
        release.set()
        if operation is not None and not operation.done():
            operation.cancel()
        if operation is not None:
            with contextlib.suppress(BaseException):
                await operation

    assert adapter_timeouts == [23]
    assert events == ["adapter_done", "close"]
    assert case_results[0]["actual"]["error_code"] == "provider_unavailable"
    assert sentinel not in json.dumps(case_results)


class _RunnerResultDb:
    client_id = "test-client"

    def __init__(self, *, fail_persistence: bool = False) -> None:
        self.fail_persistence = fail_persistence
        self.persisted: list[dict[str, Any]] = []

    def get_prompt(self, _prompt_id: int) -> dict[str, Any]:
        return {
            "id": 1,
            "project_id": 1,
            "deleted": False,
            "user_prompt": "Hello {name}",
        }

    def get_test_case(self, _test_case_id: int) -> dict[str, Any]:
        return {
            "id": 1,
            "project_id": 1,
            "inputs": {"name": "world"},
            "expected_outputs": {"response": "ok"},
        }

    def create_test_run(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail_persistence:
            raise RuntimeError("post-provider persistence failed")
        self.persisted.append(kwargs)
        return {
            "id": 31,
            "test_case_id": kwargs["test_case_id"],
            "prompt_id": kwargs["prompt_id"],
            "inputs": kwargs["inputs"],
            "expected_outputs": kwargs["expected_outputs"],
            "outputs": kwargs["outputs"],
            "model_name": kwargs["model_name"],
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("late_outcome", "expected_marks"),
    [
        ("valid_raw_text", 1),
        ("error_prefix", 0),
        ("canonical_raw_code", 0),
        ("sse_error_envelope", 0),
        ("serialized_error_envelope", 0),
    ],
)
async def test_prompt_studio_runner_starts_direct_and_drains_cancel_before_mark_and_close(
    monkeypatch: pytest.MonkeyPatch,
    late_outcome: str,
    expected_marks: int,
) -> None:
    """Credentialed work bypasses the default executor and retains its runtime."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import test_runner as runner_module

    lifecycle: list[str] = []

    class _ReleaseTrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            lifecycle.append("capacity-release")
            super()._release_capacity()

    class _Runtime:
        async def mark_used(self) -> None:
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("close")

    loop = asyncio.get_running_loop()
    previous_executor = getattr(loop, "_default_executor", None)
    default_executor = ThreadPoolExecutor(max_workers=1)
    default_entered = threading.Event()
    default_release = threading.Event()
    adapter_entered = threading.Event()
    adapter_release = threading.Event()
    adapter_timeouts: list[float | None] = []
    adapter_starts = 0
    sentinel = "prompt-late-error-secret-/srv/provider"
    pool = _ReleaseTrackingPool(capacity=1)
    runner = runner_module.TestRunner(_RunnerResultDb())
    runtime = _Runtime()
    task: asyncio.Task[dict[str, Any]] | None = None

    def _block_default_executor() -> None:
        default_entered.set()
        assert default_release.wait(timeout=2.0)

    def _provider_success(**kwargs: Any) -> str:
        nonlocal adapter_starts
        adapter_starts += 1
        adapter_timeouts.append(kwargs.get("timeout_seconds"))
        lifecycle.append("adapter-start")
        adapter_entered.set()
        assert adapter_release.wait(timeout=2.0)
        lifecycle.append("adapter-exit")
        if late_outcome == "valid_raw_text":
            return "late valid result"
        if late_outcome == "error_prefix":
            return f"Error: {sentinel}"
        if late_outcome == "canonical_raw_code":
            return "provider_unavailable"
        serialized_error = json.dumps(
            {
                "error": {
                    "code": "provider_unavailable",
                    "message": sentinel,
                }
            },
            separators=(",", ":"),
        )
        if late_outcome == "sse_error_envelope":
            return f"data: {serialized_error}\n\n"
        return serialized_error

    def _extract_response(response: Any) -> str:
        lifecycle.append("semantic-extract")
        return runner_module.TestRunner._extract_response_text(response)

    async def _endpoint_scope() -> dict[str, Any]:
        try:
            return await runner.run_test_case(
                prompt_id=1,
                test_case_id=1,
                api_key_override="prompt-late-secret",
                credentials_resolved=True,
                timeout_seconds=0.25,
                persist_run=False,
                on_provider_success=runtime.mark_used,
            )
        finally:
            await runtime.close()

    monkeypatch.setattr(runner_module, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(runner, "_call_adapter", _provider_success)
    monkeypatch.setattr(runner, "_extract_response_text", _extract_response)
    drain_entered = _install_owned_worker_cancellation_ack(monkeypatch)
    loop.set_default_executor(default_executor)
    default_blocker = loop.run_in_executor(None, _block_default_executor)
    try:
        await _wait_for_thread_event(default_entered)
        task = asyncio.create_task(_endpoint_scope())
        await _wait_for_thread_event(adapter_entered)

        assert default_release.is_set() is False
        assert pool.active_count == 1
        task.cancel()
        await asyncio.wait_for(drain_entered.wait(), timeout=1.0)
        assert task.done() is False
        assert lifecycle == ["adapter-start"]

        adapter_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

        expected_lifecycle = [
            "adapter-start",
            "adapter-exit",
            "capacity-release",
            "semantic-extract",
        ]
        expected_lifecycle.extend(["mark"] * expected_marks)
        expected_lifecycle.append("close")
        assert lifecycle == expected_lifecycle
        assert pool.active_count == 0
    finally:
        adapter_release.set()
        default_release.set()
        await asyncio.gather(default_blocker, return_exceptions=True)
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        loop.set_default_executor(previous_executor or ThreadPoolExecutor())
        default_executor.shutdown(wait=True, cancel_futures=True)

    await asyncio.sleep(0)
    assert adapter_starts == 1
    assert adapter_timeouts == [0.25]


@pytest.mark.asyncio
async def test_prompt_studio_runner_pool_exhaustion_rejects_before_dispatch_without_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capacity exhaustion is bounded, sanitized, and never queues a late call."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import test_runner as runner_module

    secret = "prompt-rejected-secret-sentinel"
    pool = BoundedDaemonPool(capacity=1)
    holder_entered = threading.Event()
    holder_release = threading.Event()
    adapter_started = threading.Event()
    marks: list[str] = []
    logs: list[str] = []
    runner = runner_module.TestRunner(_RunnerResultDb())

    def _hold_capacity() -> None:
        holder_entered.set()
        assert holder_release.wait(timeout=2.0)

    def _forbidden_adapter(**_kwargs: Any) -> dict[str, Any]:
        adapter_started.set()
        return {"choices": [{"message": {"content": "must not run"}}]}

    async def _mark_success() -> None:
        marks.append("mark")

    holder = pool.start(
        _hold_capacity,
        name="prompt-test-capacity-holder",
        exhaustion_message="test capacity exhausted",
    )
    monkeypatch.setattr(runner_module, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(runner, "_call_adapter", _forbidden_adapter)
    sink_id = logger.add(logs.append, format="{message}")
    try:
        await _wait_for_thread_event(holder_entered)
        result = await runner.run_test_case(
            prompt_id=1,
            test_case_id=1,
            api_key_override=secret,
            credentials_resolved=True,
            persist_run=False,
            on_provider_success=_mark_success,
        )

        assert result["actual"]["error_code"] == "provider_unavailable"
        assert adapter_started.is_set() is False
        assert marks == []
        assert pool.active_count == 1
        assert secret not in json.dumps(result)
        assert secret not in "".join(logs)
    finally:
        logger.remove(sink_id)
        holder_release.set()
        holder.join(timeout=1.0)

    await asyncio.sleep(0)
    assert adapter_started.is_set() is False
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_prompt_studio_runner_sanitizes_provider_failure_before_log_and_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner

    sentinel = "prompt-studio-provider-secret-sentinel"
    logs: list[str] = []
    db = _RunnerResultDb()
    runner = TestRunner(db)

    def _provider_failure(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(runner, "_call_adapter", _provider_failure)
    sink_id = logger.add(logs.append, format="{message}")
    try:
        result = await runner.run_test_case(prompt_id=1, test_case_id=1)
    finally:
        logger.remove(sink_id)

    assert sentinel not in "".join(logs)
    assert sentinel not in json.dumps(db.persisted)
    assert sentinel not in json.dumps(result)
    assert result["actual"]["error_code"] == "provider_unavailable"


@pytest.mark.asyncio
async def test_prompt_studio_runner_does_not_mark_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner

    marks: list[str] = []
    runner = TestRunner(_RunnerResultDb())

    def _provider_failure(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("provider failed")

    async def _mark_success() -> None:
        marks.append("mark")

    monkeypatch.setattr(runner, "_call_adapter", _provider_failure)
    result = await runner.run_test_case(
        prompt_id=1,
        test_case_id=1,
        persist_run=False,
        on_provider_success=_mark_success,
    )

    assert "error" in result["actual"]
    assert marks == []


@pytest.mark.asyncio
async def test_prompt_studio_runner_rejects_in_band_provider_error_before_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner

    sentinel = "prompt-studio-in-band-secret-sentinel"
    logs: list[str] = []
    marks: list[str] = []
    db = _RunnerResultDb()
    runner = TestRunner(db)

    def _provider_error(**_kwargs: Any) -> dict[str, Any]:
        return {"error": {"message": sentinel}}

    async def _mark_success() -> None:
        marks.append("mark")

    monkeypatch.setattr(runner, "_call_adapter", _provider_error)
    sink_id = logger.add(logs.append, format="{message}")
    try:
        result = await runner.run_test_case(
            prompt_id=1,
            test_case_id=1,
            on_provider_success=_mark_success,
        )
    finally:
        logger.remove(sink_id)

    assert marks == []
    assert result["actual"]["error_code"] == "provider_unavailable"
    assert sentinel not in "".join(logs)
    assert sentinel not in json.dumps(db.persisted)
    assert sentinel not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_response",
    [None, {}, [], "", "   "],
    ids=["none", "empty-object", "empty-list", "empty-string", "whitespace"],
)
async def test_prompt_studio_runner_rejects_empty_or_malformed_response_before_mark(
    monkeypatch: pytest.MonkeyPatch,
    provider_response: Any,
) -> None:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner

    marks: list[str] = []
    runner = TestRunner(_RunnerResultDb())

    def _provider_response(**_kwargs: Any) -> Any:
        return provider_response

    async def _mark_success() -> None:
        marks.append("mark")

    monkeypatch.setattr(runner, "_call_adapter", _provider_response)
    result = await runner.run_test_case(
        prompt_id=1,
        test_case_id=1,
        persist_run=False,
        on_provider_success=_mark_success,
    )

    assert marks == []
    assert result["actual"]["error_code"] == "provider_unavailable"


@pytest.mark.parametrize(
    "error_form",
    ["raw-prefix", "canonical", "serialized", "sse"],
)
def test_prompt_studio_runner_rejects_list_wrapped_provider_errors_without_detail(
    error_form: str,
) -> None:
    """A list wrapper cannot bypass the bounded provider-error contract."""
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import (
        TestRunner,
    )

    sentinel = "prompt-list-error-secret-/private/provider.json"
    serialized = json.dumps(
        {"error": {"code": "provider_unavailable", "message": sentinel}}
    )
    payload = {
        "raw-prefix": f"Error: {sentinel}",
        "canonical": "provider_unavailable",
        "serialized": serialized,
        "sse": f"data: {serialized}\n\n",
    }[error_form]

    with pytest.raises(RuntimeError) as exc_info:
        TestRunner._extract_response_text([payload])

    assert str(exc_info.value) == "Provider returned an error response"
    assert sentinel not in str(exc_info.value)


def test_prompt_studio_runner_preserves_valid_list_wrapped_text() -> None:
    """List-returning adapters remain compatible for real assistant text."""
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import (
        TestRunner,
    )

    assert TestRunner._extract_response_text(["valid assistant text"]) == (
        "valid assistant text"
    )


def _wrap_prompt_studio_response(wrapper: str, payload: str) -> dict[str, Any]:
    if wrapper == "message":
        return {"choices": [{"message": {"content": payload}}]}
    if wrapper == "delta":
        return {"choices": [{"delta": {"content": payload}}]}
    if wrapper == "top-level":
        return {"content": payload}
    if wrapper == "list-text":
        return {
            "choices": [
                {
                    "message": {
                        "content": [{"type": "text", "text": payload}],
                    }
                }
            ]
        }
    raise AssertionError(f"Unknown response wrapper: {wrapper}")


@pytest.mark.parametrize(
    "wrapper",
    ["message", "delta", "top-level", "list-text"],
)
@pytest.mark.parametrize(
    "error_form",
    ["raw-prefix", "canonical", "serialized", "sse"],
)
def test_prompt_studio_runner_rejects_provider_errors_after_nested_extraction(
    wrapper: str,
    error_form: str,
) -> None:
    """Nested response text must re-enter the bounded provider-error guard."""
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import (
        TestRunner,
    )

    sentinel = "prompt-studio-nested-secret-/private/provider.json"
    serialized = json.dumps(
        {"error": {"code": "provider_unavailable", "message": sentinel}}
    )
    payload = {
        "raw-prefix": f"Error: {sentinel}",
        "canonical": "provider_unavailable",
        "serialized": serialized,
        "sse": f"data: {serialized}\n\n",
    }[error_form]

    with pytest.raises(RuntimeError) as exc_info:
        TestRunner._extract_response_text(
            _wrap_prompt_studio_response(wrapper, payload)
        )

    assert str(exc_info.value) == "Provider returned an error response"
    assert sentinel not in str(exc_info.value)


@pytest.mark.parametrize(
    "wrapper",
    ["message", "delta", "top-level", "list-text"],
)
def test_prompt_studio_runner_preserves_noncanonical_assistant_error_json(
    wrapper: str,
) -> None:
    """Assistant-authored noncanonical JSON remains ordinary response text."""
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import (
        TestRunner,
    )

    content = json.dumps(
        {"error": {"code": "fictional_story_error", "message": "plot device"}}
    )

    assert TestRunner._extract_response_text(
        _wrap_prompt_studio_response(wrapper, content)
    ) == content


def _mixed_prompt_studio_response(case: str, sentinel: str) -> dict[str, Any]:
    serialized = json.dumps(
        {"error": {"code": "provider_unavailable", "message": sentinel}}
    )
    valid_choice = {"message": {"content": "valid assistant text"}}
    if case == "later-choice":
        return {
            "choices": [
                valid_choice,
                {"error": {"message": sentinel}},
            ]
        }
    if case == "message-error-sibling":
        return {
            "choices": [
                {
                    "message": {
                        "content": "valid assistant text",
                        "error": {"message": sentinel},
                    }
                }
            ]
        }
    if case == "message-error-block":
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "valid assistant text"},
                            {"type": "error", "error": {"message": sentinel}},
                        ]
                    }
                }
            ]
        }
    if case == "message-later-error-text":
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "valid assistant text"},
                            {"type": "text", "text": f"data: {serialized}\n\n"},
                        ]
                    }
                }
            ]
        }
    if case == "delta-later-error-text":
        return {
            "choices": [
                {
                    "delta": {
                        "content": [
                            {"type": "text", "text": "valid assistant text"},
                            {"type": "text", "text": f"data: {serialized}\n\n"},
                        ]
                    }
                }
            ]
        }
    raise AssertionError(f"Unknown mixed response case: {case}")


@pytest.mark.parametrize(
    "case",
    [
        "later-choice",
        "message-error-sibling",
        "message-error-block",
        "message-later-error-text",
        "delta-later-error-text",
    ],
)
def test_prompt_studio_runner_rejects_mixed_success_and_provider_error(
    case: str,
) -> None:
    """Earlier valid text cannot hide a sibling or later provider error."""
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import (
        TestRunner,
    )

    sentinel = "prompt-studio-mixed-secret-/private/provider.json"

    with pytest.raises(RuntimeError) as exc_info:
        TestRunner._extract_response_text(
            _mixed_prompt_studio_response(case, sentinel)
        )

    assert str(exc_info.value) == "Provider returned an error response"
    assert sentinel not in str(exc_info.value)


def test_prompt_studio_runner_preserves_valid_multipart_noncanonical_json() -> None:
    """Multiple valid text blocks may include assistant-authored error JSON."""
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import (
        TestRunner,
    )

    noncanonical = json.dumps(
        {"error": {"code": "fictional_story_error", "message": "plot device"}}
    )
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "valid assistant text"},
                        {"type": "text", "text": noncanonical},
                    ]
                }
            }
        ]
    }

    assert TestRunner._extract_response_text(response) == (
        "valid assistant text" + noncanonical
    )


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_prompt_studio_concurrent_mixed_error_is_request_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later error cannot mark or contaminate a concurrent valid result."""
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import (
        TestRunner,
    )

    loop = asyncio.get_running_loop()
    entered = {key: asyncio.Event() for key in ("valid-key", "error-key")}
    release = {key: threading.Event() for key in entered}
    marks: list[str] = []
    sentinel = "prompt-studio-concurrent-mixed-secret-/private/provider.json"
    valid_db = _RunnerResultDb()
    error_db = _RunnerResultDb()
    valid_runner = TestRunner(valid_db)
    error_runner = TestRunner(error_db)

    def _adapter(**kwargs: Any) -> dict[str, Any]:
        key = kwargs["api_key_override"]
        loop.call_soon_threadsafe(entered[key].set)
        assert release[key].wait(timeout=2.0)
        if key == "valid-key":
            return {"choices": [{"message": {"content": "valid request output"}}]}
        return _mixed_prompt_studio_response("later-choice", sentinel)

    async def _mark_valid() -> None:
        marks.append("valid-key")

    async def _mark_error() -> None:
        marks.append("error-key")

    monkeypatch.setattr(valid_runner, "_call_adapter", _adapter)
    monkeypatch.setattr(error_runner, "_call_adapter", _adapter)
    valid_task = asyncio.create_task(
        valid_runner.run_test_case(
            prompt_id=1,
            test_case_id=1,
            api_key_override="valid-key",
            on_provider_success=_mark_valid,
        )
    )
    error_task = asyncio.create_task(
        error_runner.run_test_case(
            prompt_id=1,
            test_case_id=1,
            api_key_override="error-key",
            on_provider_success=_mark_error,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release["error-key"].set()
        error_result = await asyncio.wait_for(error_task, timeout=1.0)
        assert marks == []
        release["valid-key"].set()
        valid_result = await asyncio.wait_for(valid_task, timeout=1.0)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(valid_task, error_task, return_exceptions=True)

    assert marks == ["valid-key"]
    assert valid_result["actual"] == {"response": "valid request output"}
    assert error_result["actual"]["error_code"] == "provider_unavailable"
    assert sentinel not in json.dumps(error_result)


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_prompt_studio_concurrent_nested_results_remain_request_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nested error cannot mark or contaminate a concurrent valid result."""
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import (
        TestRunner,
    )

    loop = asyncio.get_running_loop()
    entered = {key: asyncio.Event() for key in ("valid-key", "error-key")}
    release = {key: threading.Event() for key in entered}
    marks: list[str] = []
    sentinel = "prompt-concurrent-nested-secret-/private/provider.json"
    valid_db = _RunnerResultDb()
    error_db = _RunnerResultDb()
    valid_runner = TestRunner(valid_db)
    error_runner = TestRunner(error_db)

    def _adapter(**kwargs: Any) -> dict[str, Any]:
        key = kwargs["api_key_override"]
        loop.call_soon_threadsafe(entered[key].set)
        assert release[key].wait(timeout=2.0)
        if key == "valid-key":
            return {"choices": [{"message": {"content": "valid request output"}}]}
        serialized = json.dumps(
            {"error": {"code": "provider_unavailable", "message": sentinel}}
        )
        return {
            "choices": [{"delta": {"content": f"data: {serialized}\n\n"}}]
        }

    async def _mark_valid() -> None:
        marks.append("valid-key")

    async def _mark_error() -> None:
        marks.append("error-key")

    monkeypatch.setattr(valid_runner, "_call_adapter", _adapter)
    monkeypatch.setattr(error_runner, "_call_adapter", _adapter)
    valid_task = asyncio.create_task(
        valid_runner.run_test_case(
            prompt_id=1,
            test_case_id=1,
            api_key_override="valid-key",
            on_provider_success=_mark_valid,
        )
    )
    error_task = asyncio.create_task(
        error_runner.run_test_case(
            prompt_id=1,
            test_case_id=1,
            api_key_override="error-key",
            on_provider_success=_mark_error,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release["error-key"].set()
        error_result = await asyncio.wait_for(error_task, timeout=1.0)
        assert marks == []
        release["valid-key"].set()
        valid_result = await asyncio.wait_for(valid_task, timeout=1.0)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(valid_task, error_task, return_exceptions=True)

    assert marks == ["valid-key"]
    assert valid_result["actual"] == {"response": "valid request output"}
    assert error_result["actual"]["error_code"] == "provider_unavailable"
    rendered = json.dumps(
        {
            "valid_result": valid_result,
            "error_result": error_result,
            "valid_persisted": valid_db.persisted,
            "error_persisted": error_db.persisted,
        }
    )
    assert sentinel not in rendered
    assert "valid request output" not in json.dumps(error_result)


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_prompt_studio_concurrent_list_results_remain_request_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrapped error cannot mark or contaminate a concurrent valid request."""
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import (
        TestRunner,
    )

    loop = asyncio.get_running_loop()
    entered = {key: asyncio.Event() for key in ("valid-key", "error-key")}
    release = {key: threading.Event() for key in entered}
    marks: list[str] = []
    sentinel = "prompt-concurrent-list-secret-/private/provider.json"
    valid_db = _RunnerResultDb()
    error_db = _RunnerResultDb()
    valid_runner = TestRunner(valid_db)
    error_runner = TestRunner(error_db)

    def _adapter(**kwargs: Any) -> list[str]:
        key = kwargs["api_key_override"]
        loop.call_soon_threadsafe(entered[key].set)
        assert release[key].wait(timeout=2.0)
        if key == "valid-key":
            return ["valid request output"]
        return [
            "data: "
            + json.dumps(
                {
                    "error": {
                        "code": "provider_unavailable",
                        "message": sentinel,
                    }
                }
            )
            + "\n\n"
        ]

    async def _mark_valid() -> None:
        marks.append("valid-key")

    async def _mark_error() -> None:
        marks.append("error-key")

    monkeypatch.setattr(valid_runner, "_call_adapter", _adapter)
    monkeypatch.setattr(error_runner, "_call_adapter", _adapter)
    valid_task = asyncio.create_task(
        valid_runner.run_test_case(
            prompt_id=1,
            test_case_id=1,
            api_key_override="valid-key",
            on_provider_success=_mark_valid,
        )
    )
    error_task = asyncio.create_task(
        error_runner.run_test_case(
            prompt_id=1,
            test_case_id=1,
            api_key_override="error-key",
            on_provider_success=_mark_error,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release["error-key"].set()
        error_result = await asyncio.wait_for(error_task, timeout=1.0)
        assert marks == []
        release["valid-key"].set()
        valid_result = await asyncio.wait_for(valid_task, timeout=1.0)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(valid_task, error_task, return_exceptions=True)

    assert marks == ["valid-key"]
    assert valid_result["actual"] == {"response": "valid request output"}
    assert error_result["actual"]["error_code"] == "provider_unavailable"
    rendered = json.dumps(
        {
            "valid_result": valid_result,
            "error_result": error_result,
            "valid_persisted": valid_db.persisted,
            "error_persisted": error_db.persisted,
        }
    )
    assert sentinel not in rendered
    assert "valid request output" in json.dumps(valid_db.persisted)
    assert "valid request output" not in json.dumps(error_db.persisted)


@pytest.mark.asyncio
async def test_prompt_studio_cancelled_in_band_error_does_not_mark_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner

    started = asyncio.Event()
    release = threading.Event()
    events: list[str] = []
    loop = asyncio.get_running_loop()
    runner = TestRunner(_RunnerResultDb())

    def _provider_error(**_kwargs: Any) -> dict[str, Any]:
        loop.call_soon_threadsafe(started.set)
        release.wait(timeout=5)
        events.append("adapter_done")
        return {"error": {"message": "raw-provider-error"}}

    async def _mark_success() -> None:
        events.append("mark")

    monkeypatch.setattr(runner, "_call_adapter", _provider_error)
    task = asyncio.create_task(
        runner.run_test_case(
            prompt_id=1,
            test_case_id=1,
            persist_run=False,
            on_provider_success=_mark_success,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)

    assert events == ["adapter_done"]


def test_prompt_studio_manager_adapter_rejects_in_band_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import evaluation_manager

    sentinel = "prompt-studio-manager-in-band-secret-sentinel"

    class _Adapter:
        def chat(self, _request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
            del timeout
            return {"error": {"message": sentinel}}

    monkeypatch.setattr(evaluation_manager, "is_test_mode", lambda: False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(evaluation_manager, "get_adapter_or_raise", lambda _provider: _Adapter())

    with pytest.raises(RuntimeError) as exc_info:
        evaluation_manager.EvaluationManager._call_adapter_text(
            provider="openai",
            messages_payload=[{"role": "user", "content": "hello"}],
            temperature=0.1,
            max_tokens=32,
            api_key="key-a",
            model="model-a",
            app_config={},
            credentials_resolved=True,
        )

    assert sentinel not in str(exc_info.value)


@pytest.mark.asyncio
async def test_prompt_studio_runner_marks_adapter_success_before_persistence_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner

    events: list[str] = []
    runner = TestRunner(_RunnerResultDb(fail_persistence=True))

    def _provider_success(**_kwargs: Any) -> dict[str, Any]:
        events.append("adapter_success")
        return {"choices": [{"message": {"content": "ok"}}]}

    async def _mark_success() -> None:
        events.append("mark")

    monkeypatch.setattr(runner, "_call_adapter", _provider_success)
    with pytest.raises(RuntimeError, match="post-provider persistence failed"):
        await runner.run_single_test(
            prompt_id=1,
            test_case_id=1,
            model_config={
                "provider": "openai",
                "model": "model-a",
                "parameters": {"temperature": 0.1, "max_tokens": 32},
            },
            on_provider_success=_mark_success,
        )

    assert events == ["adapter_success", "mark"]


@pytest.mark.asyncio
async def test_prompt_studio_zero_case_evaluation_does_not_mark_credentials() -> None:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.evaluation_manager import (
        EvaluationManager,
    )

    marks: list[str] = []

    async def _mark_success() -> None:
        marks.append("mark")

    result = await EvaluationManager(_SnapshotDb()).run_evaluation_with_existing_record(
        evaluation_id=17,
        prompt_id=1,
        test_case_ids=[],
        on_provider_success=_mark_success,
    )

    assert result["metrics"]["total_tests"] == 0
    assert marks == []


@pytest.mark.asyncio
async def test_prompt_studio_manager_sanitizes_runner_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import evaluation_manager

    sentinel = "prompt-studio-manager-secret-sentinel"
    logs: list[str] = []

    async def _runner_failure(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        evaluation_manager.TestRunner,
        "run_single_test",
        _runner_failure,
    )
    sink_id = logger.add(logs.append, format="{message}")
    try:
        result = await evaluation_manager.EvaluationManager(
            _SnapshotDb()
        ).run_evaluation_with_existing_record(
            evaluation_id=17,
            prompt_id=1,
            test_case_ids=[1],
        )
    finally:
        logger.remove(sink_id)

    assert sentinel not in "".join(logs)
    assert sentinel not in json.dumps(result)
    assert result["results"][0]["actual"]["error_code"] == "provider_unavailable"


@pytest.mark.asyncio
async def test_prompt_studio_background_sanitizes_failure_log_and_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_evaluations as eval_endpoint,
    )

    sentinel = "prompt-studio-background-secret-sentinel"
    logs: list[str] = []
    failed_messages: list[str] = []
    events: list[str] = []
    handle = type(
        "Handle",
        (),
        {
            "api_key": "prompt-key",
            "app_config": {},
            "credentials_resolved": True,
        },
    )()
    runtime = _TrackingPromptRuntime(events, handle)

    class _RecordingCursor(_SnapshotCursor):
        def execute(self, query: str, *args: Any) -> None:
            if "SET status = 'failed'" in query:
                failed_messages.append(str(args[0][0]))
            super().execute(query, *args)

    class _RecordingConnection(_SnapshotConnection):
        def __init__(self) -> None:
            self.cursor_instance = _RecordingCursor()

    class _FailingManager:
        def __init__(self, _db: object) -> None:
            return None

        async def run_evaluation_with_existing_record(self, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(sentinel)

    db = _SnapshotDb()
    db.connection = _RecordingConnection()
    monkeypatch.setattr(eval_endpoint, "EvaluationManager", _FailingManager)
    sink_id = logger.add(logs.append, format="{message}")
    try:
        await eval_endpoint.run_evaluation_async(
            17,
            db,  # type: ignore[arg-type]
            user_id=1,
            provider="openai",
            model="model-a",
            credential_runtime=runtime,  # type: ignore[arg-type]
            provider_credentials=handle,  # type: ignore[arg-type]
        )
    finally:
        logger.remove(sink_id)

    assert sentinel not in "".join(logs)
    assert sentinel not in json.dumps(failed_messages)
    assert failed_messages == ["The chat service provider is currently unavailable."]
    assert events == ["close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["connection", "cursor"])
async def test_prompt_studio_background_closes_runtime_when_db_setup_fails(
    failure_stage: str,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_evaluations as eval_endpoint,
    )

    events: list[str] = []
    handle = object()
    runtime = _TrackingPromptRuntime(events, handle)

    class _CursorFailureConnection:
        def cursor(self) -> object:
            raise RuntimeError("cursor setup failed")

    class _DbSetupFailure:
        def get_connection(self) -> object:
            if failure_stage == "connection":
                raise RuntimeError("connection setup failed")
            return _CursorFailureConnection()

    await eval_endpoint.run_evaluation_async(
        17,
        _DbSetupFailure(),  # type: ignore[arg-type]
        credential_runtime=runtime,  # type: ignore[arg-type]
        provider_credentials=handle,  # type: ignore[arg-type]
    )

    assert events == ["close"]
