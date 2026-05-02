import importlib
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified as eval_unified
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


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

    from tldw_Server_API.app.core import config as config_mod

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

    class _DummyService:
        async def evaluate_geval(
            self,
            *,
            source_text,
            summary,
            metrics,
            api_name,
            api_key,
            user_id,
            webhook_user_id=None,
        ):
            captured["user_id"] = user_id
            captured["webhook_user_id"] = webhook_user_id
            return {
                "evaluation_id": "eval_batch_1",
                "results": {"metrics": {"coherence": 0.9}},
            }

    monkeypatch.setattr(eval_unified, "_is_eval_test_mode", lambda: True)
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

    assert response.status_code == 200
    assert captured["user_id"] == "1"
    assert captured["webhook_user_id"] == "user_1"
