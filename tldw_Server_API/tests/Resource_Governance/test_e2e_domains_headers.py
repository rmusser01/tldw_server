from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware

pytestmark = pytest.mark.rate_limit


@pytest.fixture(params=["memory", "redis"], ids=["rg-memory", "rg-redis"])
def rg_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    backend = str(request.param)
    if backend == "redis":
        # RGSimpleMiddleware lazily builds a RedisResourceGovernor (connecting to
        # the real Redis at 127.0.0.1:6379 by default) the first time a request
        # needs one. An ambient Redis left running locally -- with stale `rg*`
        # keys from earlier runs -- can start these tests already rate-limited,
        # and a real shared Redis also means other xdist workers' tests can
        # interleave with these. Make the "redis" backend hermetic by injecting
        # the in-memory stub, same as test_governor_safety_net.py's `_gov`.
        from tldw_Server_API.app.core.Infrastructure.redis_factory import InMemoryAsyncRedis
        from tldw_Server_API.app.core.Resource_Governance import governor_redis as _governor_redis_mod

        _RealRedisGovernor = _governor_redis_mod.RedisResourceGovernor

        class _InMemoryRedisGovernor(_RealRedisGovernor):
            """Redis test double: a real `RedisResourceGovernor` wired to an
            in-process stub client instead of a real Redis connection.

            Subclassing (rather than monkeypatching an instance after the
            fact) is necessary because `RGSimpleMiddleware` constructs the
            governor itself, lazily, on first request; this class is what
            gets built in its place once patched onto `governor_redis.
            RedisResourceGovernor` below, so the real governor logic runs
            unchanged against `InMemoryAsyncRedis` instead of a live server.
            """

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._client = InMemoryAsyncRedis()

        monkeypatch.setattr(_governor_redis_mod, "RedisResourceGovernor", _InMemoryRedisGovernor)
    return backend


def _write_policy(
    *,
    tmp_path,
    policy_name: str,
    path_glob: str,
    scopes: list[str] | None = None,
    policy_suffix: str | None = None,
) -> str:
    if scopes is None:
        scopes = ["ip"]
    scopes_yaml = ", ".join(scopes)
    policy_id = f"{policy_name}.small"
    if policy_suffix:
        policy_id = f"{policy_id}.{policy_suffix}"
    policy = (
        "schema_version: 1\n"
        "policies:\n"
        f"  {policy_id}:\n"
        "    requests: { rpm: 1 }\n"
        f"    scopes: [{scopes_yaml}]\n"
        "route_map:\n"
        "  by_path:\n"
        f"    {path_glob}: {policy_id}\n"
    )
    p = tmp_path / f"rg_{policy_name}.yaml"
    p.write_text(policy, encoding="utf-8")
    return str(p)


def _assert_deny_headers(resp) -> None:


    assert resp.headers.get("Retry-After") is not None
    assert resp.headers.get("X-RateLimit-Limit") == "1"
    assert resp.headers.get("X-RateLimit-Remaining") == "0"
    reset = resp.headers.get("X-RateLimit-Reset")
    assert reset is not None and int(reset) >= 1


def test_e2e_rag_deny_headers_retry_after(monkeypatch, tmp_path, rg_backend):


    monkeypatch.setenv("RG_ENABLED", "true")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")
    monkeypatch.setenv(
        "RG_POLICY_PATH",
        _write_policy(
            tmp_path=tmp_path,
            policy_name="rag",
            path_glob="/api/v1/rag/*",
            policy_suffix=f"{rg_backend}.{tmp_path.name.replace('-', '_')}",
        ),
    )

    from tldw_Server_API.app.api.v1.endpoints.rag_health import router as rag_health_router

    app = FastAPI()
    app.add_middleware(RGSimpleMiddleware)
    app.include_router(rag_health_router)

    with TestClient(app) as client:
        r1 = client.get("/api/v1/rag/health/live")
        assert r1.status_code == 200, r1.text

        r2 = client.get("/api/v1/rag/health/live")
        assert r2.status_code == 429, r2.text
        _assert_deny_headers(r2)


def test_e2e_media_deny_headers_retry_after(monkeypatch, tmp_path, rg_backend):


    monkeypatch.setenv("RG_ENABLED", "true")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")
    monkeypatch.setenv(
        "RG_POLICY_PATH",
        _write_policy(
            tmp_path=tmp_path,
            policy_name="media",
            path_glob="/api/v1/media/*",
            policy_suffix=f"{rg_backend}.{tmp_path.name.replace('-', '_')}",
        ),
    )

    from tldw_Server_API.app.api.v1.endpoints.media.transcription_models import (
        router as transcription_models_router,
    )

    app = FastAPI()
    app.add_middleware(RGSimpleMiddleware)
    app.include_router(transcription_models_router, prefix="/api/v1/media")

    with TestClient(app) as client:
        r1 = client.get("/api/v1/media/transcription-models")
        assert r1.status_code == 200, r1.text

        r2 = client.get("/api/v1/media/transcription-models")
        assert r2.status_code == 429, r2.text
        _assert_deny_headers(r2)


def test_e2e_research_deny_headers_retry_after(monkeypatch, tmp_path, rg_backend):


    monkeypatch.setenv("RG_ENABLED", "true")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")
    monkeypatch.setenv(
        "RG_POLICY_PATH",
        _write_policy(
            tmp_path=tmp_path,
            policy_name="research",
            path_glob="/api/v1/research/*",
            policy_suffix=f"{rg_backend}.{tmp_path.name.replace('-', '_')}",
        ),
    )

    import tldw_Server_API.app.api.v1.endpoints.research as research_ep

    def _stub_generate_and_search(query, search_params):

        _ = search_params
        return {"web_search_results_dict": {"results": [], "query": query}, "sub_query_dict": {}}

    monkeypatch.setattr(research_ep, "generate_and_search", _stub_generate_and_search, raising=False)

    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

    async def _stub_user():
        return User(id=1, username="rg-test-user", is_active=True)

    async def _stub_db(_current_user=None):  # noqa: ARG001
        return object()

    app = FastAPI()
    app.add_middleware(RGSimpleMiddleware)
    app.include_router(research_ep.router, prefix="/api/v1/research")
    app.dependency_overrides[get_request_user] = _stub_user
    app.dependency_overrides[get_media_db_for_user] = _stub_db

    with TestClient(app) as client:
        r1 = client.post("/api/v1/research/websearch", json={"query": "hello"})
        assert r1.status_code == 200, r1.text

        r2 = client.post("/api/v1/research/websearch", json={"query": "hello"})
        assert r2.status_code == 429, r2.text
        _assert_deny_headers(r2)


def test_e2e_prompt_studio_deny_headers_retry_after(monkeypatch, tmp_path, auth_headers, rg_backend):


    monkeypatch.setenv("RG_ENABLED", "true")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")
    monkeypatch.setenv(
        "RG_POLICY_PATH",
        _write_policy(
            tmp_path=tmp_path,
            policy_name="prompt_studio",
            path_glob="/api/v1/prompt-studio/*",
            scopes=["ip", "api_key"],
            policy_suffix=f"{rg_backend}.{tmp_path.name.replace('-', '_')}",
        ),
    )

    from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import get_prompt_studio_db
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_status import router as prompt_studio_status_router

    class _StubPromptStudioDB:
        def get_job_stats(self):
            return {"queue_depth": 0, "processing": 0, "by_status": {"queued": 0, "processing": 0}, "by_type": {}}

        def get_lease_stats(self, warn_seconds: int):  # noqa: ARG002
            return {"active": 0, "expiring_soon": 0, "stale_processing": 0}

        def count_jobs(self, *, status: str, job_type: str):  # noqa: ARG002
            return 0

    async def _stub_db():
        return _StubPromptStudioDB()

    app = FastAPI()
    app.add_middleware(RGSimpleMiddleware)
    app.include_router(prompt_studio_status_router)
    app.dependency_overrides[get_prompt_studio_db] = _stub_db

    with TestClient(app) as client:
        r1 = client.get("/api/v1/prompt-studio/status", headers=auth_headers)
        assert r1.status_code == 200, r1.text
        assert r1.json().get("success") is True

        r2 = client.get("/api/v1/prompt-studio/status", headers=auth_headers)
        assert r2.status_code == 429, r2.text
        _assert_deny_headers(r2)
