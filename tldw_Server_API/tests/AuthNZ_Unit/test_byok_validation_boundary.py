from __future__ import annotations

import asyncio
import copy
import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import shared_keys_scoped as scoped_routes
from tldw_Server_API.app.api.v1.endpoints import user_keys as user_routes
from tldw_Server_API.app.api.v1.schemas.user_keys import (
    SharedProviderKeyUpsertRequest,
    UserProviderKeyUpsertRequest,
)
from tldw_Server_API.app.core.AuthNZ import byok_testing
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
    is_runtime_issued_provider_call_credentials,
)
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.bounded_daemon import (
    BoundedDaemonPool,
    DaemonCapacityError,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
    SanitizedProviderStreamError,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    bind_provider_call_credentials,
)
from tldw_Server_API.app.services import admin_byok_service as shared_service
from tldw_Server_API.tests.provider_credential_test_helpers import (
    resolved_request_fields_async,
)

pytestmark = pytest.mark.unit


class _ObservedPool(BoundedDaemonPool):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.released = threading.Event()

    def _release_capacity(self) -> None:
        super()._release_capacity()
        self.released.set()


class _ObservedAdmission:
    """Record one health-admission token's cross-thread acquire/release balance."""

    def __init__(self, capacity: int) -> None:
        self._semaphore = threading.BoundedSemaphore(capacity)
        self._lock = threading.Lock()
        self.acquire_calls = 0
        self.successful_acquires = 0
        self.release_calls = 0

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        with self._lock:
            self.acquire_calls += 1
        acquired = self._semaphore.acquire(blocking=blocking, timeout=timeout)
        if acquired:
            with self._lock:
                self.successful_acquires += 1
        return acquired

    def release(self) -> None:
        self._semaphore.release()
        with self._lock:
            self.release_calls += 1


def _install_provider_capacity_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    neutral: str | None,
    legacy: str | None,
) -> None:
    for name, value in (
        ("PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY", neutral),
        ("ADMIN_BYOK_VALIDATION_PER_PROVIDER_CONCURRENCY", legacy),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER",
        {},
        raising=False,
    )


def _measure_admission_capacity(admission: Any) -> int:
    acquired = 0
    while admission.acquire(blocking=False):
        acquired += 1
    for _ in range(acquired):
        admission.release()
    return acquired


class _BlockingHealthCall:
    """Block every admitted health call and expose aggregate concurrency."""

    def __init__(self, *, expected_bound: int) -> None:
        self.expected_bound = expected_bound
        self.release = threading.Event()
        self.at_bound = threading.Event()
        self.above_bound = threading.Event()
        self.first_entered = threading.Event()
        self._lock = threading.Lock()
        self.call_count = 0
        self.active_count = 0
        self.max_active = 0

    def __call__(self) -> dict[str, Any]:
        with self._lock:
            self.call_count += 1
            self.active_count += 1
            self.max_active = max(self.max_active, self.active_count)
            self.first_entered.set()
            if self.active_count == self.expected_bound:
                self.at_bound.set()
            if self.active_count > self.expected_bound:
                self.above_bound.set()
        try:
            if not self.release.wait(timeout=5.0):
                raise AssertionError("Timed out waiting to release health call")
            return {"choices": [{"message": {"content": "ok"}}]}
        finally:
            with self._lock:
                self.active_count -= 1


class _BlockingAdapter:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._lock = threading.Lock()
        self.entered = asyncio.Event()
        self.drained = asyncio.Event()
        self.release = threading.Event()
        self.call_count = 0
        self.active_count = 0
        self.timeouts: list[float | None] = []

    def chat(self, _request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            self.call_count += 1
            self.active_count += 1
            self.timeouts.append(timeout)
        self._loop.call_soon_threadsafe(self.entered.set)
        self.release.wait(5)
        with self._lock:
            self.active_count -= 1
            drained = self.active_count == 0
        if drained:
            self._loop.call_soon_threadsafe(self.drained.set)
        return {"choices": [{"message": {"content": "ok"}}]}


class _FailingAdapter:
    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel
        self.call_count = 0

    def chat(self, _request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        self.call_count += 1
        raise ChatProviderError(
            message=f"hostile upstream body {self.sentinel}",
            status_code=502,
            provider="openai",
            details={"endpoint": f"https://provider.invalid/{self.sentinel}"},
        )


class _AuthenticationFailingAdapter:
    def __init__(self, sentinel: str, status_code: int) -> None:
        self.sentinel = sentinel
        self.status_code = status_code
        self.call_count = 0

    def chat(self, _request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        self.call_count += 1
        raise ChatAuthenticationError(
            message=f"hostile authentication body {self.sentinel}",
            provider="openai",
            status_code=self.status_code,
        )


class _ExceptionAdapter:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def chat(self, _request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        raise self.error


class _RecordingAdapter:
    async_chat_is_native = False

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.call_count = 0

    def chat(
        self,
        _request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.call_count += 1
        self.entered.set()
        return {"choices": [{"message": {"content": "ok"}}]}


class _StaticValidationResultAdapter:
    """Return one exact adapter result so the validation boundary owns semantics."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.call_count = 0

    def chat(
        self,
        _request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        del timeout
        self.call_count += 1
        return copy.deepcopy(self.result)


class _CredentialBoundaryAdapter:
    """Exercise the real credential binder at the validation adapter boundary."""

    def __init__(self, observed: list[dict[str, Any]]) -> None:
        self.observed = observed

    def chat(
        self,
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del timeout
        tampered = {
            **request,
            "api_key": "attacker-loose-key",
            "app_config": {"openai_api": {"tenant": "attacker"}},
        }
        bound, credentials = bind_provider_call_credentials(
            "openai",
            tampered,
            consume=True,
        )
        assert credentials is not None
        self.observed.append(bound)
        return {"choices": [{"message": {"content": "ok"}}]}


_VALIDATION_RESULT_SECRET = "sk-validation-result-secret-/private/provider.json"


def _validation_failure_result(case: str) -> Any:
    """Build one fresh invalid or in-band-error provider result."""
    canonical_error = {
        "error": {
            "code": "provider_authentication_failed",
            "message": _VALIDATION_RESULT_SECRET,
        }
    }
    valid_completion = {"choices": [{"message": {"content": "pong"}}]}
    cases: dict[str, Any] = {
        "none": None,
        "empty_string": "",
        "whitespace_string": "   ",
        "empty_dict": {},
        "empty_list": [],
        "malformed_choices": {"choices": []},
        "integer_scalar": 7,
        "boolean_scalar": False,
        "raw_error": f"Error: rejected {_VALIDATION_RESULT_SECRET}",
        "canonical_error": canonical_error,
        "serialized_error": json.dumps(canonical_error),
        "sse_error": f"data: {json.dumps(canonical_error)}\n\n",
        "mixed_error": {
            "choices": [
                valid_completion["choices"][0],
                canonical_error,
            ]
        },
        "nested_error": {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "pong"},
                            canonical_error,
                        ]
                    }
                }
            ]
        },
        "later_error": [valid_completion, canonical_error],
    }
    return copy.deepcopy(cases[case])


def _validation_success_result(case: str) -> Any:
    """Build one fresh semantically meaningful provider result."""
    cases: dict[str, Any] = {
        "raw_text": "pong",
        "openai_text": {"choices": [{"message": {"content": "pong"}}]},
        "openai_content_blocks": {
            "choices": [
                {
                    "message": {
                        "content": [{"type": "text", "text": "pong"}]
                    }
                }
            ]
        },
        "openai_tool_call": {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_validation",
                                "type": "function",
                                "function": {"name": "ping", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        },
        "legacy_function_call": {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "function_call": {"name": "ping", "arguments": "{}"},
                    }
                }
            ]
        },
    }
    return copy.deepcopy(cases[case])


class _OpenAIResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "ok"}}]}


class _OpenAIClient:
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self._calls = calls

    def __enter__(self) -> _OpenAIClient:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> _OpenAIResponse:
        self._calls.append({"url": url, "headers": headers, "json": json})
        return _OpenAIResponse()


def _principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=7, roles=["admin"], is_admin=True)


async def _allow_scope(*_args: Any, **_kwargs: Any) -> None:
    return None


def _install_boundary(monkeypatch: pytest.MonkeyPatch, adapter: Any, pool: _ObservedPool) -> None:
    registry = SimpleNamespace(get_adapter=lambda _provider: adapter)
    monkeypatch.setattr(byok_testing, "_is_test_mode", lambda: False)
    monkeypatch.setattr(byok_testing, "get_registry", lambda: registry)
    monkeypatch.setattr(byok_testing, "build_app_config_for_provider", lambda *_args: {})
    monkeypatch.setattr(
        byok_testing,
        "resolve_default_model_for_provider",
        lambda *_args, **_kwargs: "test-model",
    )
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        byok_testing,
        "PROVIDER_CREDENTIAL_VALIDATION_TIMEOUT_SECONDS",
        30.0,
        raising=False,
    )
    monkeypatch.setattr(user_routes, "_require_byok_enabled", lambda: None)
    monkeypatch.setattr(user_routes, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(user_routes, "is_trusted_base_url_request", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(user_routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(shared_service, "require_byok_enabled", lambda: None)
    monkeypatch.setattr(shared_service, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(shared_service, "normalize_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(scoped_routes, "_require_byok_enabled", lambda: None)
    monkeypatch.setattr(scoped_routes, "_require_org_manager", _allow_scope)
    monkeypatch.setattr(scoped_routes, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(
        scoped_routes,
        "is_trusted_base_url_request",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(scoped_routes, "validate_credential_fields", lambda *_args, **_kwargs: {})


def _install_legacy_boundary(
    monkeypatch: pytest.MonkeyPatch,
    call: Any,
    pool: _ObservedPool,
) -> None:
    registry = SimpleNamespace(get_adapter=lambda _provider: None)
    monkeypatch.setattr(byok_testing, "_is_test_mode", lambda: False)
    monkeypatch.setattr(byok_testing, "get_registry", lambda: registry)
    monkeypatch.setattr(byok_testing, "list_registered_providers", lambda: ["openai"])
    monkeypatch.setattr(byok_testing, "chat_api_call", call)
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        byok_testing,
        "PROVIDER_CREDENTIAL_VALIDATION_TIMEOUT_SECONDS",
        30.0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", ["adapter", "legacy"])
async def test_provider_validation_dispatch_binds_authentic_candidate_credentials(
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
) -> None:
    """Validation dispatch authenticates one capability before trusting loose fields."""

    observed: list[dict[str, Any]] = []
    pool = _ObservedPool(1)
    if branch == "adapter":
        _install_boundary(monkeypatch, _CredentialBoundaryAdapter(observed), pool)
    else:
        def call_legacy(**kwargs: Any) -> dict[str, Any]:
            credentials = kwargs.get(PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY)
            assert is_runtime_issued_provider_call_credentials(
                credentials,
                provider="openai",
            )
            tampered = {
                **kwargs,
                "api_key": "attacker-loose-key",
                "app_config": {"openai_api": {"tenant": "attacker"}},
            }
            bound, resolved = bind_provider_call_credentials(
                "openai",
                tampered,
                consume=True,
            )
            assert resolved is credentials
            observed.append(bound)
            return {"choices": [{"message": {"content": "ok"}}]}

        _install_legacy_boundary(monkeypatch, call_legacy, pool)

    model = await byok_testing.test_provider_credentials(
        provider="openai",
        api_key="sk-candidate-runtime",
        app_config={
            "openai_api": {
                "model": "candidate-model",
                "tenant": "runtime-tenant",
            }
        },
        model="candidate-model",
    )

    assert model == "candidate-model"
    assert len(observed) == 1
    assert observed[0]["api_key"] == "sk-candidate-runtime"
    assert observed[0]["app_config"]["openai_api"]["tenant"] == "runtime-tenant"
    assert PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY not in observed[0]


@pytest.mark.asyncio
async def test_provider_validation_alias_dispatches_through_canonical_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider alias must bind the canonical adapter credential capability."""
    observed: list[dict[str, Any]] = []
    pool = _ObservedPool(1)
    _install_boundary(monkeypatch, _CredentialBoundaryAdapter(observed), pool)

    model = await byok_testing.test_provider_credentials(
        provider="oai",
        api_key="sk-alias-candidate-runtime",
        app_config={"openai_api": {"model": "alias-model"}},
        model="alias-model",
    )

    assert model == "alias-model"
    assert len(observed) == 1
    assert observed[0]["api_key"] == "sk-alias-candidate-runtime"
    assert PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY not in observed[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", ["adapter", "legacy"])
async def test_provider_validation_dispatch_rejects_forged_candidate_capability(
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
) -> None:
    """Neither validation branch may fall back to a boolean trust marker."""

    forged = ProviderCallCredentials(
        provider="openai",
        api_key="forged-key",
        app_config={"openai_api": {"model": "candidate-model"}},
        auth_source="api_key",
        runtime_generation=0,
        runtime_identity=object(),
        credential_identity=object(),
    )

    class ForgedRuntime:
        async def resolve(
            self,
            provider: str,
            *,
            model: str | None = None,
        ) -> ProviderCallCredentials:
            del provider, model
            return forged

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        byok_testing,
        "ProviderCredentialRuntime",
        lambda **_kwargs: ForgedRuntime(),
        raising=False,
    )
    pool = _ObservedPool(1)
    if branch == "adapter":
        _install_boundary(monkeypatch, _CredentialBoundaryAdapter([]), pool)
    else:
        def call_legacy(**kwargs: Any) -> dict[str, Any]:
            bind_provider_call_credentials(
                "openai",
                kwargs,
                consume=True,
            )
            return {"choices": [{"message": {"content": "ok"}}]}

        _install_legacy_boundary(monkeypatch, call_legacy, pool)

    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await byok_testing.test_provider_credentials(
            provider="openai",
            api_key="sk-candidate-runtime",
            app_config={"openai_api": {"model": "candidate-model"}},
            model="candidate-model",
        )

    assert exc_info.value.code == "provider_configuration_invalid"


def _install_shared_admin_chat_boundary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    admin_adapter: Any,
    chat_adapter: Any,
    pool: BoundedDaemonPool,
) -> None:
    """Install real Admin and Chat adapter boundaries over one process pool."""
    admin_registry = SimpleNamespace(get_adapter=lambda _provider: admin_adapter)
    chat_registry = SimpleNamespace(get_adapter=lambda _provider: chat_adapter)
    monkeypatch.setattr(byok_testing, "_is_test_mode", lambda: False)
    monkeypatch.setattr(byok_testing, "get_registry", lambda: admin_registry)
    monkeypatch.setattr(byok_testing, "build_app_config_for_provider", lambda *_args: {})
    monkeypatch.setattr(
        byok_testing,
        "resolve_default_model_for_provider",
        lambda *_args, **_kwargs: "test-model",
    )
    monkeypatch.setattr(
        byok_testing,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: chat_registry)
    monkeypatch.setattr(chat_service, "SYNC_ADAPTER_CALL_POOL", pool)


async def _run_shared_admin_validation(
    *,
    api_key: str,
    timeout_seconds: float = 1.0,
) -> str | None:
    return await byok_testing.test_provider_credentials(
        provider="openai",
        api_key=api_key,
        app_config={"openai_api": {"model": "test-model"}},
        model="test-model",
        timeout_seconds=timeout_seconds,
    )


async def _run_legacy_validation(
    *,
    api_key: str = "sk-legacy-validation",
    app_config: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> str | None:
    return await byok_testing.test_provider_credentials(
        provider="openai",
        api_key=api_key,
        app_config=app_config or {"openai_api": {"model": "legacy-model"}},
        model="legacy-model",
        timeout_seconds=timeout_seconds,
    )


def _install_real_openai_legacy_boundary(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, Any]],
) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter

    _install_legacy_boundary(
        monkeypatch,
        byok_testing.chat_api_call,
        _ObservedPool(1),
    )
    monkeypatch.setattr(
        openai_adapter,
        "http_client_factory",
        lambda **_kwargs: _OpenAIClient(calls),
    )
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_OPENAI", "1")


@pytest.mark.asyncio
async def test_legacy_validation_authoritative_missing_key_ignores_late_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    _install_real_openai_legacy_boundary(monkeypatch, calls)
    monkeypatch.setenv("OPENAI_API_KEY", "late-environment-key-b")

    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await _run_legacy_validation(
            api_key="",
            app_config={"openai_api": {"model": "legacy-model"}},
        )

    assert exc_info.value.status_code == 500
    assert str(exc_info.value) == "The selected provider configuration is invalid."
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("app_config", "expected_url"),
    [
        (
            {
                "openai_api": {
                    "model": "legacy-model",
                    "api_base_url": "https://snapshot-a.example/v1",
                }
            },
            "https://snapshot-a.example/v1/chat/completions",
        ),
        (
            {"openai_api": {"model": "legacy-model"}},
            "https://api.openai.com/v1/chat/completions",
        ),
    ],
    ids=("captured-endpoint-a", "captured-endpoint-absence"),
)
async def test_legacy_validation_endpoint_snapshot_ignores_late_environment_rotation(
    monkeypatch: pytest.MonkeyPatch,
    app_config: dict[str, Any],
    expected_url: str,
) -> None:
    calls: list[dict[str, Any]] = []
    _install_real_openai_legacy_boundary(monkeypatch, calls)
    for name in (
        "OPENAI_API_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "MOCK_OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_BASE_URL", "https://late-b.example/v1")

    model = await _run_legacy_validation(
        api_key="snapshot-key-a",
        app_config=app_config,
    )

    assert model == "legacy-model"
    assert len(calls) == 1
    assert calls[0]["url"] == expected_url
    assert calls[0]["headers"]["Authorization"] == "Bearer snapshot-key-a"


async def _run_user_validation() -> Any:
    return await user_routes.upsert_user_provider_key(
        UserProviderKeyUpsertRequest(provider="openai", api_key="sk-user-validation"),
        request=SimpleNamespace(),
        principal=_principal(),
    )


async def _run_shared_validation() -> Any:
    return await shared_service.upsert_shared_key(
        _principal(),
        SharedProviderKeyUpsertRequest(
            scope_type="org",
            scope_id=42,
            provider="openai",
            api_key="sk-shared-validation",
        ),
    )


async def _run_scoped_validation() -> Any:
    return await scoped_routes.upsert_org_shared_key(
        org_id=42,
        payload=UserProviderKeyUpsertRequest(
            provider="openai",
            api_key="sk-scoped-validation",
        ),
        request=SimpleNamespace(),
        principal=_principal(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["user", "shared"])
async def test_cancelled_validation_retains_capacity_until_secret_worker_exits(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    loop = asyncio.get_running_loop()
    adapter = _BlockingAdapter(loop)
    pool = _ObservedPool(1)
    _install_boundary(monkeypatch, adapter, pool)
    validation = _run_user_validation if family == "user" else _run_shared_validation

    first = asyncio.create_task(validation())
    try:
        await asyncio.wait_for(adapter.entered.wait(), timeout=1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        assert pool.active_count == 1
        assert adapter.timeouts == [30.0]

        with pytest.raises(HTTPException) as exc_info:
            await asyncio.wait_for(validation(), timeout=1)
        assert exc_info.value.status_code == 502
        assert exc_info.value.detail == "The chat service provider is currently unavailable."
        assert adapter.call_count == 1
    finally:
        adapter.release.set()
        await asyncio.wait_for(adapter.drained.wait(), timeout=1)
        if pool.active_count:
            released = await asyncio.to_thread(pool.released.wait, 1)
            assert released is True

    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["user", "shared", "admin"])
async def test_validation_adapter_error_is_sanitized_at_every_http_family(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    sentinel = f"sk-{family}-adapter-secret-/private/{family}-provider-body.json"
    adapter = _FailingAdapter(sentinel)
    pool = _ObservedPool(1)
    _install_boundary(monkeypatch, adapter, pool)
    validation = {
        "user": _run_user_validation,
        "shared": _run_scoped_validation,
        "admin": _run_shared_validation,
    }[family]

    with pytest.raises(HTTPException) as exc_info:
        await validation()

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "The chat service provider is currently unavailable."
    exception_graph = (
        repr(exc_info.value),
        repr(exc_info.value.__cause__),
        repr(exc_info.value.__context__),
    )
    assert sentinel not in "".join(exception_graph)
    assert adapter.call_count == 1
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["user", "scoped", "admin"])
@pytest.mark.parametrize("provider_status", [401, 403])
async def test_validation_authentication_error_becomes_upstream_502_at_every_http_family(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    provider_status: int,
) -> None:
    sentinel = (
        f"sk-{family}-auth-{provider_status}-secret-"
        f"/private/{family}-auth-provider-body.json"
    )
    adapter = _AuthenticationFailingAdapter(sentinel, provider_status)
    pool = _ObservedPool(1)
    _install_boundary(monkeypatch, adapter, pool)
    validation = {
        "user": _run_user_validation,
        "scoped": _run_scoped_validation,
        "admin": _run_shared_validation,
    }[family]

    with pytest.raises(HTTPException) as exc_info:
        await validation()

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == (
        "The selected provider credentials could not be authenticated."
    )
    exception_graph = (
        repr(exc_info.value),
        repr(exc_info.value.__cause__),
        repr(exc_info.value.__context__),
    )
    assert sentinel not in "".join(exception_graph)
    assert adapter.call_count == 1
    assert pool.active_count == 0


@pytest.mark.parametrize("provider_status", [401, 403])
def test_provider_validation_public_error_rewrites_pre_sanitized_auth_to_502(
    provider_status: int,
) -> None:
    sentinel = f"pre-sanitized-auth-{provider_status}-secret"

    error = byok_testing.provider_validation_public_error(
        SanitizedProviderStreamError(
            code="provider_authentication_failed",
            message=sentinel,
            status_code=provider_status,
        )
    )

    assert error.code == "provider_authentication_failed"
    assert error.status_code == 502
    assert error.message == (
        "The selected provider credentials could not be authenticated."
    )
    assert sentinel not in repr(error)


@pytest.mark.concurrent
@pytest.mark.asyncio
async def test_concurrent_auth_failures_remain_detached_and_request_local_across_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinels = {
        "sk-user-validation": "user-auth-secret-/private/user-provider.json",
        "sk-scoped-validation": "scoped-auth-secret-/private/scoped-provider.json",
        "sk-shared-validation": "admin-auth-secret-/private/admin-provider.json",
    }
    statuses = {
        "sk-user-validation": 401,
        "sk-scoped-validation": 403,
        "sk-shared-validation": 401,
    }
    arrived: list[str] = []
    lock = threading.Lock()
    all_entered = threading.Event()
    release = threading.Event()

    class _ConcurrentAuthenticationAdapter:
        def chat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            api_key = request["api_key"]
            with lock:
                arrived.append(api_key)
                if len(arrived) == len(sentinels):
                    all_entered.set()
            if not release.wait(2):
                raise TimeoutError("concurrent authentication test was not released")
            raise ChatAuthenticationError(
                message=f"hostile authentication body {sentinels[api_key]}",
                provider="openai",
                status_code=statuses[api_key],
            )

    pool = _ObservedPool(3)
    _install_boundary(monkeypatch, _ConcurrentAuthenticationAdapter(), pool)
    monkeypatch.setenv(
        "PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY",
        "3",
    )
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER",
        {},
        raising=False,
    )
    tasks = [
        asyncio.create_task(validation())
        for validation in (
            _run_user_validation,
            _run_scoped_validation,
            _run_shared_validation,
        )
    ]
    try:
        assert await asyncio.to_thread(all_entered.wait, 1)
    finally:
        release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert set(arrived) == set(sentinels)
    assert pool.active_count == 0
    for result in results:
        assert isinstance(result, HTTPException)
        assert result.status_code == 502
        assert result.detail == (
            "The selected provider credentials could not be authenticated."
        )
        assert result.__cause__ is None
        assert result.__context__ is None
        assert all(sentinel not in repr(result) for sentinel in sentinels.values())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "expected_status", "expected_detail"),
    [
        (
            ChatBadRequestError,
            400,
            "The selected provider configuration is invalid.",
        ),
        (
            ChatRateLimitError,
            429,
            "The chat service provider is currently unavailable.",
        ),
        (
            ChatConfigurationError,
            500,
            "The selected provider configuration is invalid.",
        ),
    ],
)
async def test_validation_recognized_errors_preserve_safe_status_contract(
    monkeypatch: pytest.MonkeyPatch,
    error_type: (
        type[ChatBadRequestError]
        | type[ChatRateLimitError]
        | type[ChatConfigurationError]
    ),
    expected_status: int,
    expected_detail: str,
) -> None:
    sentinel = "sk-recognized-error-secret-/private/validation-provider-body.json"
    adapter = _ExceptionAdapter(
        error_type(
            message=f"hostile upstream body {sentinel}",
            provider="openai",
        )
    )
    pool = _ObservedPool(1)
    _install_boundary(monkeypatch, adapter, pool)

    with pytest.raises(HTTPException) as exc_info:
        await _run_user_validation()

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail
    exception_graph = (
        repr(exc_info.value),
        repr(exc_info.value.__cause__),
        repr(exc_info.value.__context__),
    )
    assert sentinel not in "".join(exception_graph)
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_validation_provider_timeout_preserves_safe_504_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "sk-timeout-error-secret-/private/validation-timeout-body.json"
    adapter = _ExceptionAdapter(
        ChatProviderError(
            message=f"hostile upstream body {sentinel}",
            status_code=504,
            provider="openai",
        )
    )
    pool = _ObservedPool(1)
    _install_boundary(monkeypatch, adapter, pool)

    with pytest.raises(HTTPException) as exc_info:
        await _run_user_validation()

    assert exc_info.value.status_code == 504
    assert exc_info.value.detail == "The chat service provider is currently unavailable."
    exception_graph = (
        repr(exc_info.value),
        repr(exc_info.value.__cause__),
        repr(exc_info.value.__context__),
    )
    assert sentinel not in "".join(exception_graph)
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_default_validation_deadline_reaches_ollama_http_timeout_without_mutating_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers import local_adapters

    source_config = {
        "ollama_api": {
            "api_url": "http://127.0.0.1:11434/v1",
            "model": "qwen-validation",
            "api_timeout": 300,
        }
    }
    observed_http_timeouts: list[float] = []

    def fake_http_call(**kwargs: Any) -> dict[str, Any]:
        observed_http_timeouts.append(kwargs["timeout"])
        return {"choices": [{"message": {"content": "ok"}}]}

    adapter = local_adapters.OllamaAdapter()
    registry = SimpleNamespace(get_adapter=lambda _provider: adapter)
    pool = _ObservedPool(1)
    monkeypatch.setattr(byok_testing, "_is_test_mode", lambda: False)
    monkeypatch.setattr(byok_testing, "get_registry", lambda: registry)
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        byok_testing,
        "PROVIDER_CREDENTIAL_VALIDATION_TIMEOUT_SECONDS",
        0.2,
    )
    monkeypatch.setattr(
        local_adapters,
        "_chat_with_openai_compatible_local_server",
        fake_http_call,
    )

    model = await byok_testing.test_provider_credentials(
        provider="ollama",
        api_key=None,
        app_config=source_config,
    )

    assert model == "qwen-validation"
    assert observed_http_timeouts == [0.2]
    assert source_config["ollama_api"]["api_timeout"] == 300
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", ["adapter", "legacy"])
async def test_validation_value_error_is_one_detached_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
) -> None:
    sentinel = f"sk-{branch}-value-error-/private/{branch}-config.json"
    pool = _ObservedPool(1)
    if branch == "adapter":
        _install_boundary(monkeypatch, _ExceptionAdapter(ValueError(sentinel)), pool)

        async def validate() -> str | None:
            return await byok_testing.test_provider_credentials(
                provider="openai",
                api_key="sk-adapter-validation",
                app_config={"openai_api": {"model": "adapter-model"}},
                model="adapter-model",
            )

    else:

        def fail_legacy(**_kwargs: Any) -> None:
            raise ValueError(sentinel)

        _install_legacy_boundary(monkeypatch, fail_legacy, pool)
        validate = _run_legacy_validation

    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await validate()

    assert exc_info.value.code == "provider_configuration_invalid"
    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "The selected provider configuration is invalid."
    exception_graph = (
        repr(exc_info.value),
        repr(exc_info.value.__cause__),
        repr(exc_info.value.__context__),
    )
    assert sentinel not in "".join(exception_graph)
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_legacy_validation_error_is_sanitized_and_detached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "sk-legacy-upstream-secret-/private/legacy-provider-body.json"

    def fail_legacy(**_kwargs: Any) -> None:
        raise ChatProviderError(
            message=f"hostile legacy response {sentinel}",
            status_code=502,
            provider="openai",
            details={"endpoint": f"https://provider.invalid/{sentinel}"},
        )

    pool = _ObservedPool(1)
    _install_legacy_boundary(monkeypatch, fail_legacy, pool)

    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await _run_legacy_validation()

    assert exc_info.value.code == "provider_unavailable"
    assert exc_info.value.status_code == 502
    assert exc_info.value.message == "The chat service provider is currently unavailable."
    exception_graph = (
        repr(exc_info.value),
        repr(exc_info.value.__cause__),
        repr(exc_info.value.__context__),
    )
    assert sentinel not in "".join(exception_graph)
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_cancelled_legacy_validation_retains_capacity_until_worker_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    call_count = 0
    call_lock = threading.Lock()

    def blocking_legacy(**_kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        with call_lock:
            call_count += 1
        entered.set()
        release.wait(5)
        return {"choices": [{"message": {"content": "ok"}}]}

    pool = _ObservedPool(1)
    _install_legacy_boundary(monkeypatch, blocking_legacy, pool)
    first = asyncio.create_task(_run_legacy_validation())
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        assert pool.active_count == 1
        with pytest.raises(SanitizedProviderStreamError) as exc_info:
            await _run_legacy_validation()
        assert exc_info.value.status_code == 502
        assert call_count == 1
    finally:
        release.set()
        if pool.active_count:
            assert await asyncio.to_thread(pool.released.wait, 1)

    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_legacy_validation_timeout_keeps_capacity_until_worker_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    call_count = 0

    def blocking_legacy(**_kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        entered.set()
        release.wait(5)
        return {"choices": [{"message": {"content": "ok"}}]}

    pool = _ObservedPool(1)
    _install_legacy_boundary(monkeypatch, blocking_legacy, pool)
    try:
        with pytest.raises(SanitizedProviderStreamError) as exc_info:
            await _run_legacy_validation(timeout_seconds=0.05)
        assert exc_info.value.status_code == 502
        assert await asyncio.to_thread(entered.wait, 1)
        assert pool.active_count == 1

        with pytest.raises(SanitizedProviderStreamError):
            await _run_legacy_validation(timeout_seconds=0.05)
        assert call_count == 1
    finally:
        release.set()
        if pool.active_count:
            assert await asyncio.to_thread(pool.released.wait, 1)

    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_concurrent_legacy_validations_keep_credential_config_snapshots_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    all_entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    snapshots: list[tuple[str | None, dict[str, Any]]] = []

    def gated_legacy(**kwargs: Any) -> dict[str, Any]:
        with lock:
            snapshots.append(
                (
                    kwargs.get("api_key"),
                    copy.deepcopy(kwargs["app_config"]),
                )
            )
            if len(snapshots) == 2:
                all_entered.set()
        release.wait(5)
        return {"choices": [{"message": {"content": "ok"}}]}

    pool = _ObservedPool(2)
    _install_legacy_boundary(monkeypatch, gated_legacy, pool)
    config_a = {"openai_api": {"model": "legacy-model", "tenant": "a"}}
    config_b = {"openai_api": {"model": "legacy-model", "tenant": "b"}}
    tasks = [
        asyncio.create_task(
            _run_legacy_validation(api_key="sk-snapshot-a", app_config=config_a)
        ),
        asyncio.create_task(
            _run_legacy_validation(api_key="sk-snapshot-b", app_config=config_b)
        ),
    ]
    try:
        assert await asyncio.to_thread(all_entered.wait, 1)
        assert {
            (api_key, config["openai_api"]["tenant"])
            for api_key, config in snapshots
        } == {
            ("sk-snapshot-a", "a"),
            ("sk-snapshot-b", "b"),
        }
    finally:
        release.set()
        await asyncio.gather(*tasks)

    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_shared_admin_health_pool_rejects_chat_before_secret_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One blocking Admin check must consume Chat's process-wide sync capacity."""
    loop = asyncio.get_running_loop()
    admin_adapter = _BlockingAdapter(loop)
    chat_adapter = _RecordingAdapter()
    pool = BoundedDaemonPool(1)
    sentinel = "sk-chat-capacity-secret-/private/chat-provider.json"
    _install_shared_admin_chat_boundary(
        monkeypatch,
        admin_adapter=admin_adapter,
        chat_adapter=chat_adapter,
        pool=pool,
    )
    first = asyncio.create_task(
        _run_shared_admin_validation(api_key="sk-admin-blocking")
    )
    try:
        await asyncio.wait_for(admin_adapter.entered.wait(), timeout=1.0)
        resolved_fields = await resolved_request_fields_async(
            "openai",
            api_key=sentinel,
            app_config={"openai_api": {"model": "gpt-4o"}},
            model="gpt-4o",
        )

        with pytest.raises(SanitizedProviderStreamError) as exc_info:
            await chat_service.perform_chat_api_call_async(
                api_endpoint="openai",
                messages_payload=[],
                model="gpt-4o",
                streaming=False,
                **resolved_fields,
            )

        assert exc_info.value.code == "provider_unavailable"
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert sentinel not in repr(exc_info.value)
        assert chat_adapter.call_count == 0
        assert not chat_adapter.entered.is_set()
        assert pool.active_count == 1
    finally:
        admin_adapter.release.set()
        await asyncio.gather(first, return_exceptions=True)

    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_shared_admin_health_pool_rejects_admin_before_secret_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One blocking Chat call must consume Admin's process-wide sync capacity."""
    loop = asyncio.get_running_loop()
    chat_adapter = _BlockingAdapter(loop)
    admin_adapter = _RecordingAdapter()
    pool = BoundedDaemonPool(1)
    sentinel = "sk-admin-capacity-secret-/private/admin-provider.json"
    _install_shared_admin_chat_boundary(
        monkeypatch,
        admin_adapter=admin_adapter,
        chat_adapter=chat_adapter,
        pool=pool,
    )
    resolved_fields = await resolved_request_fields_async(
        "openai",
        api_key="sk-chat-blocking",
        app_config={"openai_api": {"model": "gpt-4o"}},
        model="gpt-4o",
    )
    first = asyncio.create_task(
        chat_service.perform_chat_api_call_async(
            api_endpoint="openai",
            messages_payload=[],
            model="gpt-4o",
            streaming=False,
            **resolved_fields,
        )
    )
    try:
        await asyncio.wait_for(chat_adapter.entered.wait(), timeout=1.0)

        with pytest.raises(SanitizedProviderStreamError) as exc_info:
            await _run_shared_admin_validation(api_key=sentinel)

        assert exc_info.value.code == "provider_unavailable"
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert sentinel not in repr(exc_info.value)
        assert admin_adapter.call_count == 0
        assert not admin_adapter.entered.is_set()
        assert pool.active_count == 1
    finally:
        chat_adapter.release.set()
        await asyncio.gather(first, return_exceptions=True)

    assert pool.active_count == 0
    assert await _run_shared_admin_validation(api_key="sk-admin-recovery") == "test-model"
    assert admin_adapter.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("abandonment", ["timeout", "cancellation"])
async def test_shared_admin_health_pool_retains_capacity_until_actual_exit_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    abandonment: str,
) -> None:
    """Timed-out or cancelled Admin work cannot free capacity before real exit."""
    loop = asyncio.get_running_loop()
    admin_adapter = _BlockingAdapter(loop)
    pool = BoundedDaemonPool(1)
    _install_shared_admin_chat_boundary(
        monkeypatch,
        admin_adapter=admin_adapter,
        chat_adapter=_RecordingAdapter(),
        pool=pool,
    )
    first_secret = f"sk-admin-{abandonment}-secret-/private/provider.json"
    first = asyncio.create_task(
        _run_shared_admin_validation(
            api_key=first_secret,
            timeout_seconds=0.03 if abandonment == "timeout" else 1.0,
        )
    )
    try:
        await asyncio.wait_for(admin_adapter.entered.wait(), timeout=1.0)
        if abandonment == "timeout":
            with pytest.raises(SanitizedProviderStreamError) as first_error:
                await first
            assert first_error.value.code == "provider_unavailable"
            assert first_secret not in repr(first_error.value)
        else:
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

        assert pool.active_count == 1
        second_secret = "sk-admin-capacity-secret-/private/second-provider.json"
        with pytest.raises(SanitizedProviderStreamError) as capacity_error:
            await _run_shared_admin_validation(api_key=second_secret)
        assert capacity_error.value.code == "provider_unavailable"
        assert capacity_error.value.__cause__ is None
        assert capacity_error.value.__context__ is None
        assert second_secret not in repr(capacity_error.value)
        assert admin_adapter.call_count == 1
    finally:
        admin_adapter.release.set()
        await asyncio.gather(first, return_exceptions=True)

    await asyncio.wait_for(admin_adapter.drained.wait(), timeout=1.0)
    for _ in range(1000):
        if pool.active_count == 0:
            break
        await asyncio.sleep(0.001)
    assert pool.active_count == 0
    assert await _run_shared_admin_validation(api_key="sk-admin-recovery") == "test-model"
    assert admin_adapter.call_count == 2


def test_provider_health_capacity_prefers_neutral_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_provider_capacity_environment(
        monkeypatch,
        neutral="3",
        legacy="7",
    )

    admission = byok_testing._provider_health_admission("openai")

    assert _measure_admission_capacity(admission) == 3


def test_provider_health_capacity_uses_legacy_environment_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_provider_capacity_environment(
        monkeypatch,
        neutral=None,
        legacy="5",
    )

    admission = byok_testing._provider_health_admission("openai")

    assert _measure_admission_capacity(admission) == 5


@pytest.mark.parametrize(
    ("neutral", "legacy", "expected"),
    [
        (None, None, 2),
        ("not-an-integer", None, 2),
        ("0", None, 2),
        ("not-an-integer", "5", 5),
        ("0", "5", 5),
    ],
    ids=[
        "missing-default",
        "invalid-default",
        "zero-default",
        "invalid-neutral-legacy",
        "zero-neutral-legacy",
    ],
)
def test_provider_health_capacity_defaults_or_falls_back_safely(
    monkeypatch: pytest.MonkeyPatch,
    neutral: str | None,
    legacy: str | None,
    expected: int,
) -> None:
    _install_provider_capacity_environment(
        monkeypatch,
        neutral=neutral,
        legacy=legacy,
    )

    admission = byok_testing._provider_health_admission("openai")

    assert _measure_admission_capacity(admission) == expected


@pytest.mark.parametrize(
    ("neutral", "legacy"),
    [
        ("999", "4"),
        (None, "999"),
    ],
    ids=["neutral", "legacy"],
)
def test_provider_health_capacity_clamps_oversized_values_to_eight(
    monkeypatch: pytest.MonkeyPatch,
    neutral: str | None,
    legacy: str | None,
) -> None:
    _install_provider_capacity_environment(
        monkeypatch,
        neutral=neutral,
        legacy=legacy,
    )

    admission = byok_testing._provider_health_admission("openai")

    assert _measure_admission_capacity(admission) == 8


def test_provider_health_capacity_builds_one_canonical_semaphore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_provider_capacity_environment(
        monkeypatch,
        neutral="4",
        legacy="7",
    )

    alias_admission = byok_testing._provider_health_admission("oai")
    canonical_admission = byok_testing._provider_health_admission("openai")

    assert alias_admission is canonical_admission
    assert _measure_admission_capacity(alias_admission) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("abandonment", ["timeout", "cancellation"])
async def test_provider_specific_health_admission_tracks_real_worker_lifetime(
    monkeypatch: pytest.MonkeyPatch,
    abandonment: str,
) -> None:
    """Aliases share a real-lifetime bound without blocking another provider."""
    loop = asyncio.get_running_loop()
    openai_adapter = _BlockingAdapter(loop)
    anthropic_adapter = _RecordingAdapter()
    pool = BoundedDaemonPool(4)
    adapters = {
        "oai": openai_adapter,
        "openai": openai_adapter,
        "anthropic": anthropic_adapter,
    }
    monkeypatch.setenv(
        "PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY",
        "1",
    )
    monkeypatch.setattr(byok_testing, "_is_test_mode", lambda: False)
    monkeypatch.setattr(
        byok_testing,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda provider: adapters.get(provider)),
    )
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        threading.BoundedSemaphore(3),
        raising=False,
    )
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER",
        {},
        raising=False,
    )

    async def validate(provider: str, timeout_seconds: float) -> str | None:
        return await byok_testing.test_provider_credentials(
            provider=provider,
            api_key=f"sk-{provider}-validation",
            app_config={
                "openai_api": {"model": "openai-validation-model"},
                "anthropic_api": {"model": "anthropic-validation-model"},
            },
            model=f"{provider}-validation-model",
            timeout_seconds=timeout_seconds,
        )

    first = asyncio.create_task(
        validate("oai", 0.03 if abandonment == "timeout" else 1.0)
    )
    same_provider: asyncio.Task[str | None] | None = None
    other_provider: asyncio.Task[str | None] | None = None
    try:
        await asyncio.wait_for(openai_adapter.entered.wait(), timeout=1.0)
        if abandonment == "timeout":
            with pytest.raises(SanitizedProviderStreamError) as first_error:
                await first
            assert first_error.value.code == "provider_unavailable"
        else:
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

        assert openai_adapter.active_count == 1
        same_provider = asyncio.create_task(validate("openai", 0.03))
        other_provider = asyncio.create_task(validate("anthropic", 0.2))

        assert await other_provider == "anthropic-validation-model"
        with pytest.raises(SanitizedProviderStreamError) as capacity_error:
            await same_provider
        assert capacity_error.value.code == "provider_unavailable"
        assert openai_adapter.call_count == 1
        assert anthropic_adapter.call_count == 1
    finally:
        openai_adapter.release.set()
        await asyncio.gather(
            first,
            *(task for task in (same_provider, other_provider) if task is not None),
            return_exceptions=True,
        )

    await asyncio.wait_for(openai_adapter.drained.wait(), timeout=1.0)
    for _ in range(1000):
        if pool.active_count == 0:
            break
        await asyncio.sleep(0.001)
    assert pool.active_count == 0
    assert await validate("openai", 0.2) == "openai-validation-model"
    assert openai_adapter.call_count == 2


@pytest.mark.asyncio
async def test_provider_specific_admission_waits_before_global_admission_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = BoundedDaemonPool(2)
    provider_admission = _ObservedAdmission(1)
    global_admission = _ObservedAdmission(1)
    assert provider_admission.acquire(blocking=False)
    provider_token_held = True
    dispatched = 0
    monkeypatch.setenv(
        "PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY",
        "1",
    )
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        global_admission,
        raising=False,
    )
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER",
        {"openai": provider_admission},
        raising=False,
    )

    def provider_call() -> dict[str, Any]:
        nonlocal dispatched
        dispatched += 1
        return {"choices": [{"message": {"content": "ok"}}]}

    try:
        with pytest.raises(DaemonCapacityError):
            await byok_testing._run_bounded_health_call(
                provider_call,
                provider="openai",
                timeout_seconds=0.03,
            )
        assert dispatched == 0
        assert global_admission.successful_acquires == 0
        assert pool.active_count == 0

        provider_admission.release()
        provider_token_held = False
        result = await byok_testing._run_bounded_health_call(
            provider_call,
            provider="openai",
            timeout_seconds=0.2,
        )
    finally:
        if provider_token_held:
            provider_admission.release()

    assert result["choices"][0]["message"]["content"] == "ok"
    assert dispatched == 1
    assert provider_admission.successful_acquires == 2
    assert provider_admission.release_calls == 2
    assert global_admission.successful_acquires == 1
    assert global_admission.release_calls == 1


@pytest.mark.asyncio
async def test_cancellation_during_global_admission_releases_provider_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = BoundedDaemonPool(2)
    provider_admission = _ObservedAdmission(1)
    global_admission = _ObservedAdmission(1)
    assert global_admission.acquire(blocking=False)
    dispatched = 0
    monkeypatch.setenv(
        "PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY",
        "1",
    )
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        global_admission,
        raising=False,
    )
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER",
        {"openai": provider_admission},
        raising=False,
    )

    def provider_call() -> dict[str, Any]:
        nonlocal dispatched
        dispatched += 1
        return {"choices": [{"message": {"content": "unexpected"}}]}

    task = asyncio.create_task(
        byok_testing._run_bounded_health_call(
            provider_call,
            provider="openai",
            timeout_seconds=1.0,
        )
    )
    try:
        for _ in range(1000):
            if (
                provider_admission.successful_acquires == 1
                and global_admission.acquire_calls > 1
            ):
                break
            await asyncio.sleep(0.001)
        assert provider_admission.successful_acquires == 1
        assert global_admission.acquire_calls > 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        global_admission.release()

    assert dispatched == 0
    assert pool.active_count == 0
    assert provider_admission.release_calls == 1
    assert global_admission.successful_acquires == 1
    assert global_admission.release_calls == 1


@pytest.mark.asyncio
async def test_pool_start_rejection_releases_provider_and_global_admission_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = BoundedDaemonPool(1)
    provider_admission = _ObservedAdmission(1)
    global_admission = _ObservedAdmission(1)
    occupant_entered = threading.Event()
    occupant_release = threading.Event()
    monkeypatch.setenv(
        "PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY",
        "1",
    )
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        global_admission,
        raising=False,
    )
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER",
        {"openai": provider_admission},
        raising=False,
    )

    def occupy_shared_pool() -> None:
        occupant_entered.set()
        occupant_release.wait(timeout=5.0)

    pool.start(occupy_shared_pool, name="ordinary-chat-occupant")
    assert await asyncio.to_thread(occupant_entered.wait, 1.0)
    try:
        with pytest.raises(DaemonCapacityError):
            await byok_testing._run_bounded_health_call(
                lambda: {"choices": []},
                provider="openai",
                timeout_seconds=0.2,
            )
        assert provider_admission.successful_acquires == 1
        assert provider_admission.release_calls == 1
        assert global_admission.successful_acquires == 1
        assert global_admission.release_calls == 1
    finally:
        occupant_release.set()
        while pool.active_count:
            await asyncio.sleep(0.001)

    result = await byok_testing._run_bounded_health_call(
        lambda: {"choices": [{"message": {"content": "recovered"}}]},
        provider="openai",
        timeout_seconds=0.2,
    )

    assert result["choices"][0]["message"]["content"] == "recovered"
    assert provider_admission.successful_acquires == 2
    assert provider_admission.release_calls == 2
    assert global_admission.successful_acquires == 2
    assert global_admission.release_calls == 2
    assert pool.active_count == 0


@pytest.mark.parametrize(
    ("shared_capacity", "expected_health_admission"),
    [
        (1, 0),
        (2, 1),
        (8, 2),
        (31, 7),
        (32, 8),
        (128, 8),
    ],
)
def test_provider_health_admission_is_derived_from_shared_pool_capacity(
    shared_capacity: int,
    expected_health_admission: int,
) -> None:
    assert (
        byok_testing._provider_health_admission_capacity(shared_capacity)
        == expected_health_admission
    )


@pytest.mark.asyncio
async def test_provider_health_admission_wait_is_bounded_before_dispatch_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = BoundedDaemonPool(2)
    admission = threading.BoundedSemaphore(1)
    assert admission.acquire(blocking=False)
    admission_held = True
    dispatched = 0
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        admission,
        raising=False,
    )

    def provider_call() -> dict[str, Any]:
        nonlocal dispatched
        dispatched += 1
        return {"choices": [{"message": {"content": "ok"}}]}

    try:
        with pytest.raises(DaemonCapacityError):
            await byok_testing._run_bounded_health_call(
                provider_call,
                provider="openai",
                timeout_seconds=0.03,
            )
        assert dispatched == 0
        assert pool.active_count == 0

        admission.release()
        admission_held = False
        result = await byok_testing._run_bounded_health_call(
            provider_call,
            provider="openai",
            timeout_seconds=0.2,
        )
    finally:
        if admission_held:
            admission.release()

    assert result["choices"][0]["message"]["content"] == "ok"
    assert dispatched == 1
    assert pool.active_count == 0
    assert admission.acquire(blocking=False)
    admission.release()


@pytest.mark.asyncio
async def test_provider_health_pool_start_rejection_releases_admission_once_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = BoundedDaemonPool(1)
    admission = _ObservedAdmission(1)
    occupant_entered = threading.Event()
    occupant_release = threading.Event()
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        admission,
        raising=False,
    )

    def occupy_shared_pool() -> None:
        occupant_entered.set()
        occupant_release.wait(timeout=5.0)

    pool.start(occupy_shared_pool, name="ordinary-chat-occupant")
    assert await asyncio.to_thread(occupant_entered.wait, 1.0)
    try:
        with pytest.raises(DaemonCapacityError):
            await byok_testing._run_bounded_health_call(
                lambda: {"choices": []},
                provider="openai",
                timeout_seconds=0.2,
            )
        assert admission.successful_acquires == 1
        assert admission.release_calls == 1
    finally:
        occupant_release.set()
        while pool.active_count:
            await asyncio.sleep(0.001)

    result = await byok_testing._run_bounded_health_call(
        lambda: {"choices": [{"message": {"content": "recovered"}}]},
        provider="openai",
        timeout_seconds=0.2,
    )

    assert result["choices"][0]["message"]["content"] == "recovered"
    assert admission.successful_acquires == 2
    assert admission.release_calls == 2
    assert pool.active_count == 0


def test_provider_health_admission_is_process_wide_across_event_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_bound = 2
    pool = BoundedDaemonPool(4)
    admission = threading.BoundedSemaphore(expected_bound)
    blocking_call = _BlockingHealthCall(expected_bound=expected_bound)
    start_barrier = threading.Barrier(5)
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        admission,
        raising=False,
    )

    def run_health_call() -> None:
        try:
            start_barrier.wait(timeout=1.0)
            results.append(
                asyncio.run(
                    byok_testing._run_bounded_health_call(
                        blocking_call,
                        provider="openai",
                        timeout_seconds=2.0,
                    )
                )
            )
        except BaseException as exc:  # noqa: BLE001 - surface thread failures in test
            errors.append(exc)

    threads = [
        threading.Thread(target=run_health_call, name=f"health-loop-{index}")
        for index in range(4)
    ]
    for thread in threads:
        thread.start()

    try:
        start_barrier.wait(timeout=1.0)
        assert blocking_call.at_bound.wait(timeout=1.0)
        assert not blocking_call.above_bound.wait(timeout=0.1)
    finally:
        blocking_call.release.set()
        for thread in threads:
            thread.join(timeout=3.0)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(results) == 4
    assert blocking_call.call_count == 4
    assert blocking_call.max_active == expected_bound
    assert blocking_call.active_count == 0
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_provider_health_capacity_one_fails_before_adapter_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = BoundedDaemonPool(1)
    dispatched = False
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        None,
        raising=False,
    )

    def provider_call() -> dict[str, Any]:
        nonlocal dispatched
        dispatched = True
        return {"choices": []}

    with pytest.raises(DaemonCapacityError):
        await byok_testing._run_bounded_health_call(
            provider_call,
            provider="openai",
            timeout_seconds=0.2,
        )

    assert dispatched is False
    assert pool.active_count == 0


def _install_validation_result_boundary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    branch: str,
    result: Any,
    pool: _ObservedPool,
) -> _StaticValidationResultAdapter | None:
    """Install one real adapter or legacy result beneath credential validation."""
    adapter: _StaticValidationResultAdapter | None = None
    if branch == "adapter":
        adapter = _StaticValidationResultAdapter(result)
        _install_boundary(monkeypatch, adapter, pool)
    else:
        def legacy_call(**_kwargs: Any) -> Any:
            return copy.deepcopy(result)

        _install_legacy_boundary(monkeypatch, legacy_call, pool)

    monkeypatch.setenv(
        "PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY",
        "2",
    )
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        threading.BoundedSemaphore(2),
        raising=False,
    )
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER",
        {},
        raising=False,
    )
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", ["adapter", "legacy"])
@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("none", "provider_unavailable"),
        ("empty_string", "provider_unavailable"),
        ("whitespace_string", "provider_unavailable"),
        ("empty_dict", "provider_unavailable"),
        ("empty_list", "provider_unavailable"),
        ("malformed_choices", "provider_unavailable"),
        ("integer_scalar", "provider_unavailable"),
        ("boolean_scalar", "provider_unavailable"),
        ("raw_error", "provider_unavailable"),
        ("canonical_error", "provider_authentication_failed"),
        ("serialized_error", "provider_authentication_failed"),
        ("sse_error", "provider_authentication_failed"),
        ("mixed_error", "provider_authentication_failed"),
        ("nested_error", "provider_authentication_failed"),
        ("later_error", "provider_authentication_failed"),
    ],
)
async def test_validation_result_semantics_fail_closed_at_dispatch_boundary(
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
    case: str,
    expected_code: str,
) -> None:
    """Only a meaningful, error-free provider result can validate credentials."""
    pool = _ObservedPool(2)
    adapter = _install_validation_result_boundary(
        monkeypatch,
        branch=branch,
        result=_validation_failure_result(case),
        pool=pool,
    )

    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await byok_testing.test_provider_credentials(
            provider="openai",
            api_key="sk-candidate-under-test",
            app_config={"openai_api": {"model": "semantic-model"}},
            model="semantic-model",
            timeout_seconds=1.0,
        )

    assert exc_info.value.code == expected_code
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert _VALIDATION_RESULT_SECRET not in repr(exc_info.value)
    assert pool.active_count == 0
    if adapter is not None:
        assert adapter.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", ["adapter", "legacy"])
@pytest.mark.parametrize(
    "case",
    [
        "raw_text",
        "openai_text",
        "openai_content_blocks",
        "openai_tool_call",
        "legacy_function_call",
    ],
)
async def test_validation_result_semantics_accept_meaningful_controls(
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
    case: str,
) -> None:
    pool = _ObservedPool(2)
    adapter = _install_validation_result_boundary(
        monkeypatch,
        branch=branch,
        result=_validation_success_result(case),
        pool=pool,
    )

    model = await byok_testing.test_provider_credentials(
        provider="openai",
        api_key="sk-candidate-under-test",
        app_config={"openai_api": {"model": "semantic-model"}},
        model="semantic-model",
        timeout_seconds=1.0,
    )

    assert model == "semantic-model"
    assert pool.active_count == 0
    if adapter is not None:
        assert adapter.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", ["adapter", "legacy"])
async def test_concurrent_validation_results_do_not_cross_success_and_error(
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
) -> None:
    """A valid concurrent result cannot turn an in-band auth error into success."""
    pool = _ObservedPool(2)
    barrier = threading.Barrier(2)
    calls: list[str] = []
    calls_lock = threading.Lock()

    def result_for(api_key: str) -> Any:
        with calls_lock:
            calls.append(api_key)
        barrier.wait(timeout=2.0)
        if api_key == "sk-valid-concurrent":
            return _validation_success_result("openai_text")
        return _validation_failure_result("canonical_error")

    if branch == "adapter":
        class _ConcurrentAdapter:
            def chat(
                self,
                request: dict[str, Any],
                *,
                timeout: float | None = None,
            ) -> Any:
                del timeout
                return result_for(str(request["api_key"]))

        _install_boundary(monkeypatch, _ConcurrentAdapter(), pool)
    else:
        def legacy_call(**kwargs: Any) -> Any:
            return result_for(str(kwargs["api_key"]))

        _install_legacy_boundary(monkeypatch, legacy_call, pool)

    monkeypatch.setenv(
        "PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY",
        "2",
    )
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        threading.BoundedSemaphore(2),
        raising=False,
    )
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER",
        {},
        raising=False,
    )

    async def validate(api_key: str) -> str | None:
        return await byok_testing.test_provider_credentials(
            provider="openai",
            api_key=api_key,
            app_config={"openai_api": {"model": "semantic-model"}},
            model="semantic-model",
            timeout_seconds=1.0,
        )

    valid_result, invalid_result = await asyncio.gather(
        validate("sk-valid-concurrent"),
        validate("sk-invalid-concurrent"),
        return_exceptions=True,
    )

    assert valid_result == "semantic-model"
    assert isinstance(invalid_result, SanitizedProviderStreamError)
    assert invalid_result.code == "provider_authentication_failed"
    assert invalid_result.__cause__ is None
    assert invalid_result.__context__ is None
    assert _VALIDATION_RESULT_SECRET not in repr(invalid_result)
    assert set(calls) == {"sk-valid-concurrent", "sk-invalid-concurrent"}


@pytest.mark.asyncio
async def test_legacy_validation_enforces_egress_in_bounded_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy dispatch cannot bypass the unauthenticated validation guard."""
    from tldw_Server_API.app.core.Security import egress as egress_policy

    pool = _ObservedPool(2)
    dispatched = False
    policy_threads: list[str] = []

    def legacy_call(**_kwargs: Any) -> dict[str, Any]:
        nonlocal dispatched
        dispatched = True
        return _validation_success_result("openai_text")

    def deny_policy(_url: str, **_kwargs: Any) -> SimpleNamespace:
        policy_threads.append(threading.current_thread().name)
        return SimpleNamespace(allowed=False, reason="denied", resolved_ips=())

    _install_legacy_boundary(monkeypatch, legacy_call, pool)
    monkeypatch.setattr(egress_policy, "evaluate_url_policy", deny_policy)
    monkeypatch.setenv(
        "PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY",
        "2",
    )
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER",
        {},
        raising=False,
    )

    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await byok_testing.test_provider_credentials(
            provider="openai",
            api_key="sk-legacy-egress-candidate",
            app_config={"openai_api": {"model": "legacy-egress-model"}},
            model="legacy-egress-model",
            timeout_seconds=1.0,
            enforce_egress_policy=True,
            authoritative_endpoint="http://127.0.0.1:8080/private",
        )

    assert exc_info.value.code == "provider_configuration_invalid"
    assert dispatched is False
    assert any(
        name.startswith("admin-provider-health-") for name in policy_threads
    )
    assert pool.active_count == 0
