"""Request-scoped provider runtime regressions for ``/test-cases/run``."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_test_cases as test_case_endpoint,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_test import (
    RunTestCasesSimpleRequest,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    configured_provider_model_from_snapshot,
)
from tldw_Server_API.app.core.Chat.bounded_daemon import await_owned_worker
from tldw_Server_API.app.core.Chat.Chat_Deps import SanitizedProviderStreamError
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import test_runner
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_case_manager import (
    TestCaseManager,
)

pytestmark = pytest.mark.unit

_ANTHROPIC_MODEL = "claude-3-5-haiku"
_ANTHROPIC_APP_CONFIG = {
    "anthropic_api": {"model": _ANTHROPIC_MODEL}
}


class _EndpointDb:
    client_id = "test-case-runtime"

    @staticmethod
    def get_prompt(prompt_id: int) -> dict[str, Any]:
        return {
            "id": prompt_id,
            "project_id": 71,
            "deleted": False,
            "user_prompt": "Question: {question}",
        }

    @staticmethod
    def get_test_case(test_case_id: int) -> dict[str, Any]:
        return {
            "id": test_case_id,
            "project_id": 71,
            "deleted": False,
        }

    @staticmethod
    def get_test_cases_by_ids(test_case_ids: list[int]) -> list[dict[str, Any]]:
        return [
            {"id": test_case_id, "project_id": 71, "deleted": False}
            for test_case_id in test_case_ids
        ]


class _Handle:
    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None,
        app_config: dict[str, Any],
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.app_config = app_config
        self.credentials_resolved = True


class _Runtime:
    def __init__(
        self,
        events: list[Any],
        outcome: _Handle
        | BaseException
        | Callable[[dict[str, Any], str, str | None], _Handle | BaseException],
        mark_results: list[bool | None] | None = None,
        **scope: Any,
    ) -> None:
        self.events = events
        self.scope = scope
        self.outcome = outcome
        self.handle: _Handle | None = None
        self.mark_results = list(mark_results or [])
        self.mark_count = 0
        self.close_count = 0
        events.append(("init", scope))

    async def resolve(self, provider: str, *, model: str | None = None) -> _Handle:
        self.events.append(("resolve", provider, model))
        outcome = (
            self.outcome(self.scope, provider, model)
            if callable(self.outcome)
            else self.outcome
        )
        if isinstance(outcome, BaseException):
            raise outcome
        self.handle = outcome
        return outcome

    async def mark_used(self, handle: object) -> bool | None:
        assert self.handle is not None and handle is self.handle
        self.mark_count += 1
        self.events.append("mark")
        if self.mark_results:
            return self.mark_results.pop(0)
        return None

    async def close(self) -> None:
        self.close_count += 1
        self.events.append("close")


def _request(user_id: int) -> Any:
    return SimpleNamespace(state=SimpleNamespace(), scope_user_id=user_id)


async def _invoke_endpoint(
    *,
    db: Any | None = None,
    provider: str = "anthropic",
    model: str | None = _ANTHROPIC_MODEL,
    prompt_id: int = 12,
    test_case_ids: list[int] | None = None,
    user_id: int = 7,
) -> dict[str, Any]:
    available = inspect.signature(test_case_endpoint.run_test_cases_simple).parameters
    request_payload: dict[str, Any] = {
        "project_id": 71,
        "prompt_id": prompt_id,
        "test_case_ids": test_case_ids or [3],
        "provider": provider,
    }
    if model is not None:
        request_payload["model"] = model
    kwargs = {
        "payload": RunTestCasesSimpleRequest.model_validate(request_payload),
        "request": _request(user_id),
        "db": db or _EndpointDb(),
        "user_context": {
            "user_id": str(user_id),
            "client_id": f"test-client-{user_id}",
        },
    }
    return await test_case_endpoint.run_test_cases_simple(
        **{name: value for name, value in kwargs.items() if name in available}
    )


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    events: list[Any],
    *,
    outcome: _Handle
    | BaseException
    | Callable[[dict[str, Any], str, str | None], _Handle | BaseException]
    | None = None,
    scope_resolver: Callable[
        [Any, Any], tuple[int, list[int], list[int], bool]
    ]
    | None = None,
    provider_requires_key: Callable[[str], bool] | None = None,
    mark_results: list[bool | None] | None = None,
) -> list[_Runtime]:
    runtimes: list[_Runtime] = []
    selected_outcome = outcome or _Handle(
        provider="anthropic",
        api_key="request-scoped-key",
        app_config=_ANTHROPIC_APP_CONFIG,
    )

    def _factory(**scope: Any) -> _Runtime:
        runtime = _Runtime(
            events,
            selected_outcome,
            mark_results=mark_results,
            **scope,
        )
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(
        test_case_endpoint,
        "ProviderCredentialRuntime",
        _factory,
        raising=False,
    )
    monkeypatch.setattr(
        test_case_endpoint,
        "derive_trusted_credential_scope",
        scope_resolver or (lambda _request, _user: (7, [8], [9], True)),
        raising=False,
    )
    monkeypatch.setattr(
        test_case_endpoint,
        "provider_requires_api_key",
        provider_requires_key
        or (lambda provider: provider not in {"ollama", "llama.cpp"}),
        raising=False,
    )
    monkeypatch.setattr(
        test_runner,
        "is_runtime_issued_provider_call_credentials",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    return runtimes


def _install_owned_worker_cancellation_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> asyncio.Event:
    from tldw_Server_API.app.core.Chat import bounded_daemon

    entered = asyncio.Event()
    original = bounded_daemon._drain_owned_task

    async def _ack(task: asyncio.Future[Any]) -> tuple[bool, Any]:
        entered.set()
        return await original(task)

    monkeypatch.setattr(bounded_daemon, "_drain_owned_task", _ack, raising=True)
    return entered


@pytest.mark.asyncio
async def test_run_endpoint_passes_one_request_snapshot_and_marks_before_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []
    runtimes = _install_runtime(monkeypatch, events)
    captured: dict[str, Any] = {}

    class _Manager:
        def __init__(self, _db: Any) -> None:
            return None

        async def run_batch_tests(self, **kwargs: Any) -> list[dict[str, Any]]:
            captured.update(kwargs)
            await kwargs["on_provider_success"]()
            return [{"test_case_id": 3, "actual": {"response": "ok"}}]

    monkeypatch.setattr(test_case_endpoint, "TestCaseManager", _Manager, raising=True)
    monkeypatch.setattr(
        test_case_endpoint,
        "require_project_access",
        lambda *_args, **_kwargs: asyncio.sleep(0),
        raising=True,
    )

    result = await _invoke_endpoint()

    assert result == {
        "results": [{"test_case_id": 3, "actual": {"response": "ok"}}]
    }
    assert len(runtimes) == 1
    assert runtimes[0].scope["user_id"] == 7
    assert runtimes[0].scope["team_ids"] == [8]
    assert runtimes[0].scope["org_ids"] == [9]
    assert runtimes[0].scope["trusted_base_url_override"] is True
    assert captured["provider"] == "anthropic"
    assert captured["model"] == _ANTHROPIC_MODEL
    assert captured["api_key_override"] == "request-scoped-key"
    assert captured["app_config"] == _ANTHROPIC_APP_CONFIG
    assert captured["credentials_resolved"] is True
    assert captured["provider_credentials"] is runtimes[0].handle
    assert captured["strict_provider_errors"] is True
    assert events == [
        ("init", runtimes[0].scope),
        ("resolve", "anthropic", _ANTHROPIC_MODEL),
        "mark",
        "close",
    ]


@pytest.mark.asyncio
async def test_run_endpoint_retries_explicit_false_provider_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []
    runtimes = _install_runtime(
        monkeypatch,
        events,
        mark_results=[False, True],
    )

    class _Manager:
        def __init__(self, _db: Any) -> None:
            return None

        async def run_batch_tests(self, **kwargs: Any) -> list[dict[str, Any]]:
            await kwargs["on_provider_success"]()
            return [{"test_case_id": 3, "actual": {"response": "ok"}}]

    monkeypatch.setattr(test_case_endpoint, "TestCaseManager", _Manager, raising=True)
    monkeypatch.setattr(
        test_case_endpoint,
        "require_project_access",
        lambda *_args, **_kwargs: asyncio.sleep(0),
        raising=True,
    )

    result = await _invoke_endpoint()

    assert result["results"][0]["actual"] == {"response": "ok"}
    assert len(runtimes) == 1
    assert runtimes[0].mark_count == 2
    assert events[-3:] == ["mark", "mark", "close"]


@pytest.mark.asyncio
async def test_run_endpoint_derives_omitted_non_openai_model_from_runtime_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []
    handle = _Handle(
        provider="anthropic",
        api_key="request-scoped-key",
        app_config=_ANTHROPIC_APP_CONFIG,
    )
    policy_models: list[str | None] = []
    captured: dict[str, Any] = {}

    def _runtime_outcome(
        _scope: dict[str, Any],
        provider: str,
        requested_model: str | None,
    ) -> _Handle:
        policy_models.append(
            requested_model
            or configured_provider_model_from_snapshot(provider, handle.app_config)
        )
        return handle

    runtimes = _install_runtime(
        monkeypatch,
        events,
        outcome=_runtime_outcome,
    )

    class _Manager:
        def __init__(self, _db: Any) -> None:
            return None

        async def run_batch_tests(self, **kwargs: Any) -> list[dict[str, Any]]:
            captured.update(kwargs)
            await kwargs["on_provider_success"]()
            return [{"test_case_id": 3, "actual": {"response": "ok"}}]

    monkeypatch.setattr(test_case_endpoint, "TestCaseManager", _Manager, raising=True)
    monkeypatch.setattr(
        test_case_endpoint,
        "require_project_access",
        lambda *_args, **_kwargs: asyncio.sleep(0),
        raising=True,
    )

    result = await _invoke_endpoint(model=None)

    assert result["results"][0]["actual"] == {"response": "ok"}
    assert policy_models == [_ANTHROPIC_MODEL]
    assert captured["model"] == _ANTHROPIC_MODEL
    assert captured["model"] != "gpt-3.5-turbo"
    assert events == [
        ("init", runtimes[0].scope),
        ("resolve", "anthropic", None),
        "mark",
        "close",
    ]


@pytest.mark.asyncio
async def test_run_endpoint_rejects_cross_project_ids_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CrossProjectDb:
        client_id = "cross-project-test-case-runtime"

        @staticmethod
        def get_prompt(prompt_id: int) -> dict[str, Any]:
            return {"id": prompt_id, "project_id": 72, "deleted": False}

        @staticmethod
        def get_test_case(test_case_id: int) -> dict[str, Any]:
            return {"id": test_case_id, "project_id": 73, "deleted": False}

        @staticmethod
        def get_test_cases_by_ids(test_case_ids: list[int]) -> list[dict[str, Any]]:
            return [
                {"id": test_case_id, "project_id": 73, "deleted": False}
                for test_case_id in test_case_ids
            ]

    events: list[Any] = []
    runtimes = _install_runtime(monkeypatch, events)

    class _Manager:
        def __init__(self, _db: Any) -> None:
            return None

        async def run_batch_tests(self, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(test_case_endpoint, "TestCaseManager", _Manager, raising=True)
    monkeypatch.setattr(
        test_case_endpoint,
        "require_project_access",
        lambda *_args, **_kwargs: asyncio.sleep(0),
        raising=True,
    )

    with pytest.raises(HTTPException):
        await _invoke_endpoint(db=_CrossProjectDb())

    assert runtimes == []


_ENDPOINT_RUNTIME_FAILURES = [
    pytest.param(
        ByokResolutionError("credential_scope_revoked", "anthropic"),
        403,
        "credential_scope_revoked",
        id="revoked-scope",
    ),
    pytest.param(
        ByokResolutionError("credential_store_unavailable", "anthropic"),
        503,
        "credential_store_unavailable",
        id="credential-store-unavailable",
    ),
    pytest.param(
        _Handle(
            provider="anthropic",
            api_key=None,
            app_config={"anthropic_api": {"model": _ANTHROPIC_MODEL}},
        ),
        503,
        "missing_provider_credentials",
        id="missing-credentials",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_outcome", "expected_status", "expected_code"),
    _ENDPOINT_RUNTIME_FAILURES,
)
async def test_run_endpoint_credential_failures_stop_before_adapter_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    runtime_outcome: _Handle | BaseException,
    expected_status: int,
    expected_code: str,
) -> None:
    events: list[Any] = []
    runtimes = _install_runtime(
        monkeypatch,
        events,
        outcome=runtime_outcome,
    )
    manager_calls = 0

    class _ForbiddenManager:
        def __init__(self, _db: Any) -> None:
            return None

        async def run_batch_tests(self, **_kwargs: Any) -> list[dict[str, Any]]:
            nonlocal manager_calls
            manager_calls += 1
            raise AssertionError("credential failure reached the adapter manager")

    monkeypatch.setattr(
        test_case_endpoint,
        "TestCaseManager",
        _ForbiddenManager,
        raising=True,
    )
    monkeypatch.setattr(
        test_case_endpoint,
        "require_project_access",
        lambda *_args, **_kwargs: asyncio.sleep(0),
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _invoke_endpoint()

    assert exc_info.value.status_code == expected_status
    assert isinstance(exc_info.value.detail, dict)
    assert exc_info.value.detail["error_code"] == expected_code
    assert manager_calls == 0
    assert len(runtimes) == 1
    assert runtimes[0].mark_count == 0
    assert runtimes[0].close_count == 1
    assert events[-1] == "close"


@pytest.mark.asyncio
async def test_run_endpoint_allows_keyless_local_provider_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []
    local_handle = _Handle(
        provider="ollama",
        api_key=None,
        app_config={"ollama_api": {"model": "llama3.2"}},
    )
    runtimes = _install_runtime(
        monkeypatch,
        events,
        outcome=local_handle,
    )
    captured: dict[str, Any] = {}

    class _Manager:
        def __init__(self, _db: Any) -> None:
            return None

        async def run_batch_tests(self, **kwargs: Any) -> list[dict[str, Any]]:
            captured.update(kwargs)
            await kwargs["on_provider_success"]()
            return [{"test_case_id": 3, "actual": {"response": "local"}}]

    monkeypatch.setattr(test_case_endpoint, "TestCaseManager", _Manager, raising=True)
    monkeypatch.setattr(
        test_case_endpoint,
        "require_project_access",
        lambda *_args, **_kwargs: asyncio.sleep(0),
        raising=True,
    )

    result = await _invoke_endpoint(provider="ollama", model="llama3.2")

    assert result["results"][0]["actual"] == {"response": "local"}
    assert captured["provider"] == "ollama"
    assert captured["api_key_override"] is None
    assert captured["app_config"] == local_handle.app_config
    assert captured["credentials_resolved"] is True
    assert len(runtimes) == 1
    assert runtimes[0].mark_count == 1
    assert runtimes[0].close_count == 1


@pytest.mark.asyncio
async def test_concurrent_run_endpoints_keep_owner_provider_snapshots_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []

    def _scope(request: Any, _user: Any) -> tuple[int, list[int], list[int], bool]:
        owner = int(request.scope_user_id)
        return owner, [owner * 10 + 1], [owner * 10 + 2], False

    def _outcome(
        scope: dict[str, Any],
        provider: str,
        model: str | None,
    ) -> _Handle:
        owner = int(scope["user_id"])
        section = "anthropic_api" if provider == "anthropic" else "openai_api"
        marker = f"owner-{owner}:{provider}:{model}"
        return _Handle(
            provider=provider,
            api_key=f"key:{marker}",
            app_config={section: {"model": model, "runtime_marker": marker}},
        )

    runtimes = _install_runtime(
        monkeypatch,
        events,
        outcome=_outcome,
        scope_resolver=_scope,
    )
    arrivals = 0
    arrivals_lock = asyncio.Lock()
    ready = asyncio.Event()
    calls: list[dict[str, Any]] = []

    class _Manager:
        def __init__(self, _db: Any) -> None:
            return None

        async def run_batch_tests(self, **kwargs: Any) -> list[dict[str, Any]]:
            nonlocal arrivals
            calls.append(dict(kwargs))
            async with arrivals_lock:
                arrivals += 1
                if arrivals == 3:
                    ready.set()
            await asyncio.wait_for(ready.wait(), timeout=2)
            await kwargs["on_provider_success"]()
            return [{"test_case_id": kwargs["test_case_ids"][0], "actual": {"response": "ok"}}]

    monkeypatch.setattr(test_case_endpoint, "TestCaseManager", _Manager, raising=True)
    monkeypatch.setattr(
        test_case_endpoint,
        "require_project_access",
        lambda *_args, **_kwargs: asyncio.sleep(0),
        raising=True,
    )
    requests = [
        {
            "user_id": 7,
            "provider": "anthropic",
            "model": "claude-owner7-a",
            "prompt_id": 12,
            "test_case_ids": [101],
        },
        {
            "user_id": 7,
            "provider": "openai",
            "model": "gpt-owner7-b",
            "prompt_id": 13,
            "test_case_ids": [102],
        },
        {
            "user_id": 8,
            "provider": "anthropic",
            "model": "claude-owner8-c",
            "prompt_id": 14,
            "test_case_ids": [103],
        },
    ]

    results = await asyncio.gather(
        *(_invoke_endpoint(**request) for request in requests)
    )

    assert len(results) == 3
    assert len(calls) == 3
    assert len(runtimes) == 3
    by_test_case = {call["test_case_ids"][0]: call for call in calls}
    for request in requests:
        call = by_test_case[request["test_case_ids"][0]]
        marker = (
            f"owner-{request['user_id']}:{request['provider']}:{request['model']}"
        )
        section = (
            "anthropic_api"
            if request["provider"] == "anthropic"
            else "openai_api"
        )
        assert call["prompt_id"] == request["prompt_id"]
        assert call["provider"] == request["provider"]
        assert call["model"] == request["model"]
        assert call["api_key_override"] == f"key:{marker}"
        assert call["app_config"] == {
            section: {
                "model": request["model"],
                "runtime_marker": marker,
            }
        }
        assert call["credentials_resolved"] is True
        assert call["strict_provider_errors"] is True

    assert all(runtime.mark_count == 1 for runtime in runtimes)
    assert all(runtime.close_count == 1 for runtime in runtimes)
    assert sorted(
        (runtime.scope["user_id"], tuple(runtime.scope["team_ids"]), tuple(runtime.scope["org_ids"]))
        for runtime in runtimes
    ) == [
        (7, (71,), (72,)),
        (7, (71,), (72,)),
        (8, (81,), (82,)),
    ]


@pytest.mark.asyncio
async def test_test_case_manager_strict_mode_does_not_convert_failure_to_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe_error = SanitizedProviderStreamError(
        code="provider_unavailable",
        message="The chat service provider is currently unavailable.",
        status_code=502,
    )

    async def _fail(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise safe_error

    monkeypatch.setattr(test_runner.TestRunner, "run_multiple_tests", _fail, raising=True)
    manager = TestCaseManager(_EndpointDb())

    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await manager.run_batch_tests(
            prompt_id=12,
            test_case_ids=[3],
            model="gpt-4o-mini",
            provider="openai",
            api_key_override="request-scoped-key",
            app_config={"openai_api": {"model": "gpt-4o-mini"}},
            credentials_resolved=True,
            strict_provider_errors=True,
        )

    assert exc_info.value is safe_error


@pytest.mark.asyncio
async def test_run_endpoint_propagates_safe_failure_instead_of_empty_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []
    runtimes = _install_runtime(monkeypatch, events)
    safe_error = SanitizedProviderStreamError(
        code="provider_unavailable",
        message="The chat service provider is currently unavailable.",
        status_code=502,
    )

    async def _fail(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise safe_error

    monkeypatch.setattr(test_runner.TestRunner, "run_multiple_tests", _fail, raising=True)
    monkeypatch.setattr(
        test_case_endpoint,
        "require_project_access",
        lambda *_args, **_kwargs: asyncio.sleep(0),
        raising=True,
    )

    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await _invoke_endpoint()

    assert exc_info.value is safe_error
    assert len(runtimes) == 1
    assert runtimes[0].close_count == 1
    assert events[-1] == "close"
    assert "mark" not in events


@pytest.mark.asyncio
async def test_run_endpoint_cancellation_stops_after_inflight_call_before_next_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []
    runtimes = _install_runtime(monkeypatch, events)
    started = asyncio.Event()
    release = asyncio.Event()
    provider_done = asyncio.Event()
    dispatches: list[str] = []
    drain_entered = _install_owned_worker_cancellation_ack(monkeypatch)

    class _Manager:
        def __init__(self, _db: Any) -> None:
            return None

        async def run_batch_tests(self, **kwargs: Any) -> list[dict[str, Any]]:
            async def _first_provider_call() -> str:
                started.set()
                await release.wait()
                provider_done.set()
                dispatches.append("first")
                events.append("provider_done")
                return "ok"

            await await_owned_worker(
                _first_provider_call(),
                on_cancel_success=kwargs["on_provider_success"],
            )
            await kwargs["on_provider_success"]()
            dispatches.append("second")
            return [{"test_case_id": 3, "actual": {"response": "ok"}}]

    monkeypatch.setattr(test_case_endpoint, "TestCaseManager", _Manager, raising=True)
    monkeypatch.setattr(
        test_case_endpoint,
        "require_project_access",
        lambda *_args, **_kwargs: asyncio.sleep(0),
        raising=True,
    )

    task = asyncio.create_task(_invoke_endpoint())
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    try:
        await asyncio.wait_for(drain_entered.wait(), timeout=2)
        assert len(runtimes) == 1
        assert provider_done.is_set() is False
        assert runtimes[0].close_count == 0

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

        assert provider_done.is_set() is True
        assert dispatches == ["first"]
        assert events[-3:] == ["provider_done", "mark", "close"]
        assert runtimes[0].close_count == 1
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await task
