import asyncio
import importlib
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified as eval_unified
from tldw_Server_API.app.core import config as config_mod
from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


@contextmanager
def _isolated_provider_override_snapshot():
    """Install one healthy empty provider snapshot and restore it on exit."""
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})
    try:
        yield
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})


@pytest.fixture(autouse=True)
def _isolate_route_and_provider_snapshots():
    """Keep process-global route and provider snapshots deterministic per test."""
    config_mod._route_toggle_policy.cache_clear()
    try:
        with _isolated_provider_override_snapshot():
            yield
    finally:
        config_mod._route_toggle_policy.cache_clear()


def _build_eval_only_app(monkeypatch) -> FastAPI:
    app = FastAPI()
    app.include_router(eval_unified.router, prefix="/api/v1")

    async def _verify_api_key_override():
        return "user_1"

    async def _get_user_override():
        return User(id=1, username="tester", email=None, is_active=True)

    async def _rate_limit_dep_override():
        return None

    app.dependency_overrides[eval_unified.verify_api_key] = _verify_api_key_override
    app.dependency_overrides[eval_unified.get_eval_request_user] = _get_user_override
    app.dependency_overrides[eval_unified.check_evaluation_rate_limit] = _rate_limit_dep_override
    return app


def _reload_main_app(
    monkeypatch,
    *,
    minimal: bool,
    routes_enable: str | None = "evaluations",
    routes_disable: str | None = "research",
):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1" if minimal else "0")

    # Ensure route toggles are consistent for reload-time gating.
    if routes_enable is None or not str(routes_enable).strip():
        monkeypatch.delenv("ROUTES_ENABLE", raising=False)
    else:
        monkeypatch.setenv("ROUTES_ENABLE", str(routes_enable))

    if routes_disable is None or not str(routes_disable).strip():
        monkeypatch.delenv("ROUTES_DISABLE", raising=False)
    else:
        monkeypatch.setenv("ROUTES_DISABLE", str(routes_disable))

    config_mod._route_toggle_policy.cache_clear()

    from tldw_Server_API.app import main as app_main

    return importlib.reload(app_main).app


def _route_method_count(app: FastAPI, path: str, method: str) -> int:
    method_upper = method.upper()
    count = 0
    for route in app.routes:
        route_path = getattr(route, "path", None)
        route_methods = getattr(route, "methods", set()) or set()
        if route_path == path and method_upper in route_methods:
            count += 1
    return count


def test_main_mounts_evaluations_routes_in_minimal_startup(monkeypatch):
    app = _reload_main_app(monkeypatch, minimal=True)
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/api/v1/evaluations/geval" in paths
    assert "/api/v1/evaluations/rate-limits" in paths
    assert "/api/v1/evaluations/embeddings/abtest" in paths


def test_main_mounts_evaluations_routes_in_full_startup(monkeypatch):
    app = _reload_main_app(monkeypatch, minimal=False)
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/api/v1/evaluations/geval" in paths
    assert "/api/v1/evaluations/rag" in paths
    assert "/api/v1/evaluations/embeddings/abtest" in paths


def test_main_omits_evaluations_routes_in_minimal_startup_when_disabled(monkeypatch):
    app = _reload_main_app(
        monkeypatch,
        minimal=True,
        routes_enable=None,
        routes_disable="research,evaluations",
    )
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/api/v1/evaluations/geval" not in paths
    assert "/api/v1/evaluations/rag" not in paths
    assert "/api/v1/evaluations/embeddings/abtest" not in paths


def test_main_registers_abtest_post_route_once_in_minimal_startup(monkeypatch):
    app = _reload_main_app(monkeypatch, minimal=True)
    count = _route_method_count(app, "/api/v1/evaluations/embeddings/abtest", "POST")
    assert count == 1


def test_main_registers_abtest_post_route_once_in_full_startup(monkeypatch):
    app = _reload_main_app(monkeypatch, minimal=False)
    count = _route_method_count(app, "/api/v1/evaluations/embeddings/abtest", "POST")
    assert count == 1


def test_main_has_no_duplicate_method_path_pairs_in_full_startup(monkeypatch):
    app = _reload_main_app(monkeypatch, minimal=False)
    seen: set[tuple[str, str]] = set()
    duplicates: list[tuple[str, str]] = []
    allowed_methods = {"GET", "POST", "PUT", "PATCH", "DELETE"}
    path_prefix = "/api/v1/evaluations/"

    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if not path or not str(path).startswith(path_prefix):
            continue
        for method in methods:
            if method not in allowed_methods:
                continue
            key = (method, path)
            if key in seen:
                duplicates.append(key)
            else:
                seen.add(key)

    assert not duplicates


def test_geval_capacity_exhaustion_returns_detached_503_and_closes_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """The real G-Eval route must fail closed without retaining provider state."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Evaluations import ms_g_eval
    from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as service_module

    secret = "geval-route-rejected-secret-sentinel"
    app = _build_eval_only_app(monkeypatch)
    captured: dict[str, object] = {}
    lifecycle: list[str] = []
    logs: list[str] = []
    holder_entered = threading.Event()
    holder_release = threading.Event()
    adapter_started = threading.Event()
    pool = BoundedDaemonPool(capacity=1)
    credential_handle = eval_unified.ProviderCallCredentials(
        provider="openai",
        api_key=secret,
        app_config={"openai_api": {"model": "gpt-eval-test"}},
        auth_source="config",
        runtime_generation=0,
        runtime_identity=object(),
        credential_identity=object(),
    )

    class _Runtime:
        async def resolve(self, provider: str, *, model: str | None = None):
            assert provider == "openai"
            assert model == "gpt-eval-test"
            lifecycle.append("resolve")
            return credential_handle

        async def mark_used(self, handle: object) -> None:
            assert handle is credential_handle
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("close")

    class _AllowLimiter:
        async def check_rate_limit(self, *_args, **_kwargs):
            return True, {}

    class _WebhookManager:
        async def send_webhook(self, **_kwargs) -> None:
            return None

    def _hold_capacity() -> None:
        holder_entered.set()
        assert holder_release.wait(timeout=3.0)

    def _forbidden_geval(**_kwargs):
        adapter_started.set()
        return {"metrics": {}, "average_score": 0.0, "assessment": "must not run"}

    async def _forbidden_store(**_kwargs):
        lifecycle.append("store")
        return "eval-1"

    @app.exception_handler(HTTPException)
    async def _capture_http_exception(_request, exc: HTTPException):
        captured["exception"] = exc
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    runtime = _Runtime()
    service = service_module.UnifiedEvaluationService(
        db_path=str(tmp_path / "geval-route-capacity.db"),
        enable_webhooks=False,
    )
    monkeypatch.setattr(service, "_store_evaluation_result", _forbidden_store)
    monkeypatch.setattr(service_module, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(ms_g_eval, "run_geval", _forbidden_geval)
    monkeypatch.setattr(eval_unified, "_build_eval_credential_runtime", lambda **_kwargs: runtime)
    monkeypatch.setattr(eval_unified, "_is_eval_test_mode", lambda: True)
    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", lambda _uid: _AllowLimiter())
    monkeypatch.setattr(eval_unified, "_get_webhook_manager_for_user", lambda _uid: _WebhookManager())
    monkeypatch.setattr(
        eval_unified,
        "get_unified_evaluation_service_for_user",
        lambda _uid: service,
    )

    holder = pool.start(
        _hold_capacity,
        name="geval-route-capacity-holder",
        exhaustion_message="test capacity exhausted",
    )
    sink_id = logger.add(logs.append, format="{message}")
    try:
        assert holder_entered.wait(timeout=1.0)
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/evaluations/geval",
                json={
                    "source_text": "source text long enough",
                    "summary": "summary text long enough",
                    "metrics": ["coherence"],
                    "api_name": "openai",
                    "model": "gpt-eval-test",
                },
            )
    finally:
        logger.remove(sink_id)
        holder_release.set()
        holder.join(timeout=1.0)

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "error_code": "provider_capacity_exhausted",
            "message": "The evaluation provider is temporarily busy.",
        }
    }
    assert lifecycle == ["resolve", "close"]
    assert adapter_started.is_set() is False
    assert pool.active_count == 0
    assert secret not in response.text
    assert secret not in "".join(logs)
    public_error = captured["exception"]
    assert isinstance(public_error, HTTPException)
    assert public_error.__cause__ is None
    assert public_error.__context__ is None


@pytest.mark.parametrize(
    ("route", "request_payload"),
    [
        (
            "rag",
            {
                "query": "What is RAG?",
                "retrieved_contexts": ["RAG retrieves context."],
                "generated_response": "RAG retrieves context.",
                "metrics": ["relevance"],
                "api_name": "openai",
                "model": "gpt-eval-test",
            },
        ),
        pytest.param(
            "rag",
            {
                "query": "What is semantic similarity?",
                "retrieved_contexts": ["Similarity compares meaning."],
                "generated_response": "An orange cat sleeps on a window sill.",
                "ground_truth": "Enterprise networks rotate asymmetric keys.",
                "metrics": ["answer_similarity"],
                "api_name": "openai",
                "model": "gpt-eval-test",
            },
            id="rag-answer-similarity-capacity",
        ),
        pytest.param(
            "rag",
            {
                "query": "What is context precision?",
                "retrieved_contexts": ["Precision measures retrieved relevance."],
                "generated_response": "It measures retrieved relevance.",
                "metrics": ["context_precision"],
                "api_name": "openai",
                "model": "gpt-eval-test",
            },
            id="rag-context-precision-capacity",
        ),
        pytest.param(
            "rag",
            {
                "query": "What is context relevance?",
                "retrieved_contexts": ["Relevance compares context with the query."],
                "generated_response": "It compares context with the query.",
                "metrics": ["context_relevance"],
                "api_name": "openai",
                "model": "gpt-eval-test",
            },
            id="rag-context-relevance-capacity",
        ),
        (
            "response-quality",
            {
                "prompt": "What is RAG?",
                "response": "RAG retrieves context.",
                "api_name": "openai",
                "model": "gpt-eval-test",
            },
        ),
    ],
)
def test_sync_evaluation_capacity_exhaustion_is_detached_503(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    route: str,
    request_payload: dict[str, object],
) -> None:
    """RAG and quality routes fail closed before saturated provider dispatch."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Evaluations import (
        rag_evaluator,
        response_quality_evaluator,
    )
    from tldw_Server_API.app.core.Evaluations import (
        unified_evaluation_service as service_module,
    )

    secret = f"{route}-capacity-secret-sentinel"
    app = _build_eval_only_app(monkeypatch)
    captured: dict[str, object] = {}
    lifecycle: list[str] = []
    logs: list[str] = []
    holder_entered = threading.Event()
    holder_release = threading.Event()
    adapter_started = threading.Event()
    pool = BoundedDaemonPool(1)
    credential_handle = eval_unified.ProviderCallCredentials(
        provider="openai",
        api_key=secret,
        app_config={"openai_api": {"model": "gpt-eval-test"}},
        auth_source="config",
        runtime_generation=0,
        runtime_identity=object(),
        credential_identity=object(),
    )

    class Runtime:
        async def resolve(self, provider: str, *, model: str | None = None):
            assert (provider, model) == ("openai", "gpt-eval-test")
            lifecycle.append("resolve")
            return credential_handle

        async def mark_used(self, handle: object) -> None:
            assert handle is credential_handle
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("close")

    class AllowLimiter:
        async def check_rate_limit(self, *_args, **_kwargs):
            return True, {}

    class WebhookManager:
        async def send_webhook(self, **_kwargs) -> None:
            return None

    def hold_capacity() -> None:
        holder_entered.set()
        holder_release.wait(timeout=3.0)

    def forbidden_analyze(*_args, **_kwargs) -> str:
        adapter_started.set()
        return "4"

    async def store_result(**_kwargs) -> str:
        lifecycle.append("store")
        return "eval-1"

    @app.exception_handler(HTTPException)
    async def capture_http_exception(_request, exc: HTTPException):
        captured["exception"] = exc
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    runtime = Runtime()
    service = service_module.UnifiedEvaluationService(
        db_path=str(tmp_path / f"{route}-capacity.db"),
        enable_webhooks=False,
    )
    monkeypatch.setattr(service, "_store_evaluation_result", store_result)
    monkeypatch.setattr(rag_evaluator, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(response_quality_evaluator, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(rag_evaluator, "analyze", forbidden_analyze)
    monkeypatch.setattr(response_quality_evaluator, "analyze", forbidden_analyze)
    monkeypatch.setattr(eval_unified, "_build_eval_credential_runtime", lambda **_kwargs: runtime)
    monkeypatch.setattr(eval_unified, "_is_eval_test_mode", lambda: True)
    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", lambda _uid: AllowLimiter())
    monkeypatch.setattr(eval_unified, "_get_webhook_manager_for_user", lambda _uid: WebhookManager())
    monkeypatch.setattr(
        eval_unified,
        "get_unified_evaluation_service_for_user",
        lambda _uid: service,
    )

    holder = pool.start(
        hold_capacity,
        name=f"{route}-capacity-holder",
        exhaustion_message="test capacity exhausted",
    )
    sink_id = logger.add(logs.append, format="{message}")
    try:
        assert holder_entered.wait(timeout=1.0)
        with TestClient(app) as client:
            response = client.post(
                f"/api/v1/evaluations/{route}",
                json=request_payload,
            )
    finally:
        logger.remove(sink_id)
        holder_release.set()
        holder.join(timeout=1.0)

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "error_code": "provider_capacity_exhausted",
            "message": "The evaluation provider is temporarily busy.",
        }
    }
    assert lifecycle == ["resolve", "close"]
    assert adapter_started.is_set() is False
    assert pool.active_count == 0
    assert secret not in response.text
    assert secret not in "".join(logs)
    public_error = captured["exception"]
    assert isinstance(public_error, HTTPException)
    assert public_error.__cause__ is None
    assert public_error.__context__ is None


def test_response_quality_capacity_failure_drains_admitted_siblings_before_runtime_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A rejected quality sibling must not orphan already-admitted provider work."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import (
        BoundedDaemonPool,
        DaemonCapacityError,
    )
    from tldw_Server_API.app.core.Evaluations import response_quality_evaluator
    from tldw_Server_API.app.core.Evaluations import (
        unified_evaluation_service as service_module,
    )
    from tldw_Server_API.app.core.Evaluations.circuit_breaker import llm_circuit_breaker

    secret = "response-quality-sibling-capacity-secret-sentinel"
    app = _build_eval_only_app(monkeypatch)
    captured: dict[str, object] = {}
    response_box: dict[str, object] = {}
    lifecycle: list[str] = []
    close_active_counts: list[int] = []
    logs: list[str] = []
    adapter_state = {"calls": 0, "sibling_starts": 0}
    adapter_lock = threading.Lock()
    admitted_siblings = threading.Event()
    capacity_rejected = threading.Event()
    sibling_release = threading.Event()
    runtime_closed = threading.Event()
    request_done = threading.Event()

    class TrackingPool(BoundedDaemonPool):
        def _acquire_capacity(self, exhaustion_message: str) -> None:
            try:
                super()._acquire_capacity(exhaustion_message)
            except DaemonCapacityError:
                lifecycle.append("capacity-rejected")
                capacity_rejected.set()
                raise

        def _release_capacity(self) -> None:
            super()._release_capacity()
            lifecycle.append("capacity-released")

    pool = TrackingPool(capacity=2)
    credential_handle = eval_unified.ProviderCallCredentials(
        provider="openai",
        api_key=secret,
        app_config={"openai_api": {"model": "gpt-eval-test"}},
        auth_source="config",
        runtime_generation=0,
        runtime_identity=object(),
        credential_identity=object(),
    )

    class Runtime:
        async def resolve(self, provider: str, *, model: str | None = None):
            assert (provider, model) == ("openai", "gpt-eval-test")
            lifecycle.append("resolve")
            return credential_handle

        async def mark_used(self, handle: object) -> None:
            assert handle is credential_handle
            lifecycle.append("mark")

        async def close(self) -> None:
            close_active_counts.append(pool.active_count)
            lifecycle.append("close")
            runtime_closed.set()

    class AllowLimiter:
        async def check_rate_limit(self, *_args, **_kwargs):
            return True, {}

    class WebhookManager:
        async def send_webhook(self, **_kwargs) -> None:
            return None

    def blocking_analyze(*args, **_kwargs) -> str:
        assert args[3] == secret
        with adapter_lock:
            adapter_state["calls"] += 1
            call_number = adapter_state["calls"]
            if call_number > 1:
                adapter_state["sibling_starts"] += 1
                if adapter_state["sibling_starts"] == 2:
                    admitted_siblings.set()

        if call_number == 1:
            lifecycle.append("provider-probe")
            return "4"

        lifecycle.append(f"provider-sibling-{call_number}-started")
        assert sibling_release.wait(timeout=3.0)
        lifecycle.append(f"provider-sibling-{call_number}-exited")
        return "4"

    async def forbidden_store(**_kwargs) -> str:
        lifecycle.append("store")
        return "eval-1"

    @app.exception_handler(HTTPException)
    async def capture_http_exception(_request, exc: HTTPException):
        captured["exception"] = exc
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    runtime = Runtime()
    service = service_module.UnifiedEvaluationService(
        db_path=str(tmp_path / "response-quality-sibling-capacity.db"),
        enable_webhooks=False,
    )
    monkeypatch.setattr(service, "_store_evaluation_result", forbidden_store)
    monkeypatch.setattr(response_quality_evaluator, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(response_quality_evaluator, "analyze", blocking_analyze)
    monkeypatch.setattr(eval_unified, "_build_eval_credential_runtime", lambda **_kwargs: runtime)
    monkeypatch.setattr(eval_unified, "_is_eval_test_mode", lambda: True)
    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", lambda _uid: AllowLimiter())
    monkeypatch.setattr(eval_unified, "_get_webhook_manager_for_user", lambda _uid: WebhookManager())
    monkeypatch.setattr(
        eval_unified,
        "get_unified_evaluation_service_for_user",
        lambda _uid: service,
    )

    def post_request(client: TestClient) -> None:
        try:
            response_box["response"] = client.post(
                "/api/v1/evaluations/response-quality",
                json={
                    "prompt": "Explain bounded provider work.",
                    "response": "Bounded work has a fixed admission capacity.",
                    "api_name": "openai",
                    "model": "gpt-eval-test",
                },
            )
        except Exception as exc:  # noqa: BLE001 - surface thread failures to the test
            response_box["error"] = exc
        finally:
            request_done.set()

    llm_circuit_breaker.reset_all()
    sink_id = logger.add(logs.append, format="{message}")
    request_thread: threading.Thread | None = None
    try:
        with TestClient(app) as client:
            request_thread = threading.Thread(
                target=post_request,
                args=(client,),
                name="response-quality-capacity-request",
            )
            request_thread.start()
            siblings_observed = admitted_siblings.wait(timeout=1.0)
            rejection_observed = capacity_rejected.wait(timeout=1.0)
            request_finished_before_release = request_done.wait(timeout=0.25)
            runtime_closed_before_release = runtime_closed.is_set()
            sibling_release.set()
            request_thread.join(timeout=3.0)
    finally:
        sibling_release.set()
        if request_thread is not None:
            request_thread.join(timeout=1.0)
        logger.remove(sink_id)
        llm_circuit_breaker.reset_all()

    assert siblings_observed is True
    assert rejection_observed is True
    assert request_finished_before_release is False
    assert runtime_closed_before_release is False
    assert request_thread is not None
    assert request_thread.is_alive() is False
    assert "error" not in response_box
    response = response_box["response"]
    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "error_code": "provider_capacity_exhausted",
            "message": "The evaluation provider is temporarily busy.",
        }
    }
    assert adapter_state == {"calls": 3, "sibling_starts": 2}
    assert "mark" not in lifecycle
    assert "store" not in lifecycle
    assert close_active_counts == [0]
    assert pool.active_count == 0
    close_index = lifecycle.index("close")
    assert sum(
        item.startswith("provider-sibling-") and item.endswith("-exited")
        for item in lifecycle[:close_index]
    ) == 2
    assert lifecycle[:close_index].count("capacity-released") == 3
    assert secret not in response.text
    assert secret not in "".join(logs)
    public_error = captured["exception"]
    assert isinstance(public_error, HTTPException)
    assert public_error.__cause__ is None
    assert public_error.__context__ is None


def test_propositions_preserves_http_429_when_rate_limited(monkeypatch):
    app = _build_eval_only_app(monkeypatch)

    class _DenyLimiter:
        async def check_rate_limit(self, *_args, **_kwargs):
            return False, {"error": "rate limit exceeded", "retry_after": 7}

    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", lambda _uid: _DenyLimiter())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/propositions",
            json={
                "extracted": ["Claim A"],
                "reference": ["Claim A"],
                "method": "semantic",
                "threshold": 0.7,
            },
        )

    assert response.status_code == 429
    assert response.headers.get("retry-after") == "7"


def test_history_preserves_http_403_for_non_admin_cross_user_request(monkeypatch):
    app = _build_eval_only_app(monkeypatch)

    class _DummyService:
        async def get_evaluation_history(self, **_kwargs):
            return []

        async def count_evaluations(self, **_kwargs):
            return 0

    async def _principal_override():
        return SimpleNamespace(is_admin=False, roles=[], permissions=[])

    monkeypatch.setattr(
        eval_unified,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _DummyService(),
    )
    app.dependency_overrides[eval_unified.get_auth_principal] = _principal_override

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/history",
            json={"user_id": "user_2", "limit": 10, "offset": 0},
        )

    assert response.status_code == 403
    assert "Admin privileges required" in response.json()["detail"]


def test_history_includes_canonical_offset_pagination(monkeypatch):
    app = _build_eval_only_app(monkeypatch)

    class _DummyService:
        async def get_evaluation_history(self, **_kwargs):
            return [
                {
                    "id": "eval_1",
                    "created_at": "2026-01-01T00:00:00",
                    "evaluation_type": "g_eval",
                }
            ]

        async def count_evaluations(self, **_kwargs):
            return 3

    monkeypatch.setattr(
        eval_unified,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _DummyService(),
    )
    app.dependency_overrides[eval_unified.get_auth_principal] = lambda: SimpleNamespace(
        is_admin=False, roles=[], permissions=[]
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/history",
            json={"limit": 1, "offset": 1},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_count"] == 3
    assert body["pagination"] == {
        "mode": "offset",
        "limit": 1,
        "offset": 1,
        "total": 3,
        "has_more": True,
        "next_offset": 2,
    }
    assert body["has_more"] is True
    assert body["next_offset"] == 2


def test_propositions_uses_stable_user_id_instead_of_auth_context_token(monkeypatch):
    app = _build_eval_only_app(monkeypatch)

    async def _verify_api_key_override():
        return "super-secret-api-key"

    app.dependency_overrides[eval_unified.verify_api_key] = _verify_api_key_override

    captured = {}

    class _AllowLimiter:
        async def check_rate_limit(self, *_args, **_kwargs):
            return True, {}

    class _DummyService:
        async def evaluate_propositions(self, *, extracted, reference, method, threshold, user_id):
            captured["user_id"] = user_id
            return {
                "evaluation_id": "eval_prop_1",
                "results": {
                    "metrics": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                    "counts": {"matched": 1, "total_extracted": 1, "total_reference": 1},
                    "details": {},
                },
                "evaluation_time": 0.01,
            }

    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", lambda _uid: _AllowLimiter())
    monkeypatch.setattr(
        eval_unified,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _DummyService(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/propositions",
            json={
                "extracted": ["Claim A"],
                "reference": ["Claim A"],
                "method": "semantic",
                "threshold": 0.7,
            },
        )

    assert response.status_code == 200
    assert captured["user_id"] == "1"


def test_ocr_uses_stable_user_id_instead_of_auth_context_token(monkeypatch):
    app = _build_eval_only_app(monkeypatch)

    async def _verify_api_key_override():
        return "super-secret-api-key"

    app.dependency_overrides[eval_unified.verify_api_key] = _verify_api_key_override

    captured = {}

    class _AllowLimiter:
        async def check_rate_limit(self, *_args, **_kwargs):
            return True, {}

    class _DummyService:
        async def evaluate_ocr(self, *, items, ocr_options=None, metrics=None, thresholds=None, user_id):
            captured["user_id"] = user_id
            return {
                "evaluation_id": "eval_ocr_1",
                "results": {"summary": {}},
                "evaluation_time": 0.01,
            }

    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", lambda _uid: _AllowLimiter())
    monkeypatch.setattr(
        eval_unified,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _DummyService(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/ocr",
            json={
                "items": [
                    {
                        "id": "doc-1",
                        "extracted_text": "hello world",
                        "ground_truth_text": "hello world",
                    }
                ]
            },
        )

    assert response.status_code == 200
    assert captured["user_id"] == "1"


def test_batch_geval_uses_stable_and_webhook_user_identity(monkeypatch):
    app = _build_eval_only_app(monkeypatch)

    async def _verify_api_key_override():
        return "super-secret-api-key"

    app.dependency_overrides[eval_unified.verify_api_key] = _verify_api_key_override

    captured = {}

    class _AllowLimiter:
        async def check_rate_limit(self, *_args, **_kwargs):
            return True, {}

    credential_handle = eval_unified.ProviderCallCredentials(
        provider="openai",
        api_key="provider-secret",
        app_config={"openai_api": {"model": "gpt-eval-test"}},
        auth_source="config",
        runtime_generation=0,
        runtime_identity=object(),
        credential_identity=object(),
    )

    class _DummyCredentialRuntime:
        async def resolve(self, provider, *, model=None):
            captured["resolved_provider"] = provider
            captured["requested_model"] = model
            return credential_handle

        async def mark_used(self, handle):
            captured["marked_handle"] = handle

        async def close(self):
            captured["runtime_closed"] = True

    class _DummyService:
        async def evaluate_geval(
            self,
            *,
            source_text,
            summary,
            metrics,
            api_name,
            api_key,
            model,
            user_id,
            webhook_user_id=None,
            app_config=None,
            credentials_resolved=False,
            provider_credentials=None,
        ):
            captured["user_id"] = user_id
            captured["webhook_user_id"] = webhook_user_id
            captured["model"] = model
            captured["app_config"] = app_config
            captured["credentials_resolved"] = credentials_resolved
            captured["provider_credentials"] = provider_credentials
            return {
                "evaluation_id": "eval_batch_1",
                "results": {"metrics": {"coherence": 0.9}},
            }

    monkeypatch.setattr(eval_unified, "_is_eval_test_mode", lambda: True)
    monkeypatch.setattr(
        eval_unified,
        "_build_eval_credential_runtime",
        lambda **_kwargs: _DummyCredentialRuntime(),
    )
    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", lambda _uid: _AllowLimiter())
    monkeypatch.setattr(
        eval_unified,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _DummyService(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/batch",
            json={
                "evaluation_type": "geval",
                "parallel_workers": 1,
                "items": [
                    {
                        "source_text": "source",
                        "summary": "summary",
                        "metrics": ["coherence"],
                    }
                ],
            },
        )

    assert response.status_code == 200, response.text
    assert captured["user_id"] == "1"
    assert captured["webhook_user_id"] == "user_1"
    assert captured["model"] == "gpt-eval-test"
    assert captured["app_config"] == {"openai_api": {"model": "gpt-eval-test"}}
    assert captured["credentials_resolved"] is True
    assert captured["provider_credentials"] is credential_handle
    assert captured["resolved_provider"] == "openai"
    assert captured["requested_model"] is None
    assert captured["marked_handle"] is credential_handle
    assert captured["runtime_closed"] is True


def test_batch_client_teardown_poison_isolated_before_next_credential_resolution(
    monkeypatch,
):
    app = _build_eval_only_app(monkeypatch)
    refresh_started = threading.Event()
    credential_handle = eval_unified.ProviderCallCredentials(
        provider="openai",
        api_key="provider-secret",
        app_config={"openai_api": {"model": "gpt-eval-test"}},
        auth_source="config",
        runtime_generation=0,
        runtime_identity=object(),
        credential_identity=object(),
    )

    async def _blocking_get_db_pool():
        refresh_started.set()
        await asyncio.Event().wait()

    class _AllowLimiter:
        async def check_rate_limit(self, *_args, **_kwargs):
            return True, {}

    class _CredentialRuntime:
        async def resolve(self, provider, *, model=None):
            _ = model
            llm_provider_overrides.capture_provider_override_call_snapshot(provider)
            for _ in range(100):
                if refresh_started.is_set():
                    break
                await asyncio.sleep(0)
            assert refresh_started.is_set()
            return credential_handle

        async def mark_used(self, _handle):
            return None

        async def close(self):
            return None

    class _DummyService:
        async def evaluate_geval(self, **_kwargs):
            return {
                "evaluation_id": "eval_batch_refresh",
                "results": {"metrics": {"coherence": 0.9}},
            }

    monkeypatch.setattr(llm_provider_overrides, "get_db_pool", _blocking_get_db_pool)
    monkeypatch.setattr(eval_unified, "_is_eval_test_mode", lambda: True)
    monkeypatch.setattr(
        eval_unified,
        "_build_eval_credential_runtime",
        lambda **_kwargs: _CredentialRuntime(),
    )
    monkeypatch.setattr(
        eval_unified,
        "get_user_rate_limiter_for_user",
        lambda _uid: _AllowLimiter(),
    )
    monkeypatch.setattr(
        eval_unified,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _DummyService(),
    )

    with _isolated_provider_override_snapshot():
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
            {},
            ttl_enabled=True,
        )
        with llm_provider_overrides._OVERRIDE_LOCK:
            llm_provider_overrides._OVERRIDE_CACHE_REFRESHED_AT -= (
                llm_provider_overrides._OVERRIDE_REFRESH_INTERVAL_SECONDS + 0.1
            )

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/evaluations/batch",
                json={
                    "evaluation_type": "geval",
                    "parallel_workers": 1,
                    "items": [
                        {
                            "source_text": "source",
                            "summary": "summary",
                            "metrics": ["coherence"],
                        }
                    ],
                },
            )

        assert response.status_code == 200, response.text
        assert refresh_started.is_set()
        with pytest.raises(ByokResolutionError) as exc_info:
            llm_provider_overrides.capture_provider_override_call_snapshot("openai")
        assert exc_info.value.code == "credential_store_unavailable"

    snapshot = llm_provider_overrides.capture_provider_override_call_snapshot("openai")

    assert snapshot.provider == "openai"
