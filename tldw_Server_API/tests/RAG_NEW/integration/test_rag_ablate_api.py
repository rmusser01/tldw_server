import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_endpoint
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import ProviderCredentialRuntime
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import UnifiedSearchResult
from tldw_Server_API.app.main import app as fastapi_app

pytestmark = pytest.mark.integration

_SENTINEL = "rag-ablate-runtime-secret"


class _CountingRuntime(ProviderCredentialRuntime):
    __slots__ = ("close_calls",)

    def __init__(self) -> None:
        self.close_calls = 0

        async def resolver(provider, **_kwargs):
            return ResolvedByokCredentials(
                provider=provider,
                api_key=_SENTINEL,
                app_config={},
                credential_fields={},
                source="user",
                allowlisted=True,
                status=ByokResolutionStatus.RESOLVED,
                auth_source="api_key",
            )

        def reject_server_fallback(_provider):
            raise AssertionError("authenticated ablation must not use server credentials")

        super().__init__(
            user_id=1,
            team_ids=[],
            org_ids=[],
            trusted_base_url_override=False,
            fallback_resolver=reject_server_fallback,
            resolver=resolver,
        )

    async def close(self):
        self.close_calls += 1
        await super().close()


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/rag/ablate",
            "headers": [],
            "query_string": b"",
        }
    )


def _install_direct_ablate_fakes(monkeypatch, runtime):
    monkeypatch.setattr(rag_endpoint, "_build_credential_runtime", lambda *_args: runtime)
    monkeypatch.setattr(rag_endpoint, "_resolve_kanban_db_path", lambda _user: "kanban.db")
    monkeypatch.setattr(rag_endpoint, "rag_result_from_unified_search_result", lambda result: result)
    monkeypatch.setattr(rag_endpoint, "rag_result_to_response", lambda result: result)


@pytest.mark.asyncio
async def test_authenticated_ablate_passes_same_real_runtime_to_all_four_calls(monkeypatch):
    runtime = _CountingRuntime()
    seen = []
    _install_direct_ablate_fakes(monkeypatch, runtime)

    async def provider_boundary(**kwargs):
        received = kwargs["credential_runtime"]
        seen.append(received)
        handle = await received.resolve("openai")
        assert handle.api_key == _SENTINEL  # nosec B101
        return UnifiedSearchResult(documents=[], query="ablate")

    monkeypatch.setattr(rag_endpoint, "unified_rag_pipeline", provider_boundary)
    monkeypatch.setattr(rag_endpoint, "agentic_rag_pipeline", provider_boundary)

    response = await rag_endpoint.rag_ablate(
        request_raw=_request(),
        request=rag_endpoint.AblationRequest(query="ablate", with_answer=True),
        current_user=User(id=1, username="tester", email=None, is_active=True),
        media_db=None,
        chacha_db=None,
    )

    assert seen == [runtime, runtime, runtime, runtime]  # nosec B101
    assert runtime.close_calls == 1  # nosec B101
    assert _SENTINEL not in repr(response)  # nosec B101


@pytest.mark.asyncio
async def test_authenticated_ablate_maps_typed_failure_and_closes(monkeypatch):
    runtime = _CountingRuntime()
    _install_direct_ablate_fakes(monkeypatch, runtime)

    async def fail_closed(**_kwargs):
        raise ByokResolutionError("credential_store_unavailable", "openai")

    monkeypatch.setattr(rag_endpoint, "unified_rag_pipeline", fail_closed)

    with pytest.raises(HTTPException) as exc_info:
        await rag_endpoint.rag_ablate(
            request_raw=_request(),
            request=rag_endpoint.AblationRequest(query="ablate", with_answer=True),
            current_user=User(id=1, username="tester", email=None, is_active=True),
            media_db=None,
            chacha_db=None,
        )

    assert exc_info.value.status_code == 503  # nosec B101
    assert exc_info.value.detail["error_code"] == "credential_store_unavailable"  # nosec B101
    assert runtime.close_calls == 1  # nosec B101


@pytest.mark.asyncio
async def test_authenticated_ablate_propagates_cancellation_and_closes(monkeypatch):
    runtime = _CountingRuntime()
    _install_direct_ablate_fakes(monkeypatch, runtime)

    async def cancel(**_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(rag_endpoint, "unified_rag_pipeline", cancel)

    with pytest.raises(asyncio.CancelledError):
        await rag_endpoint.rag_ablate(
            request_raw=_request(),
            request=rag_endpoint.AblationRequest(query="ablate", with_answer=True),
            current_user=User(id=1, username="tester", email=None, is_active=True),
            media_db=None,
            chacha_db=None,
        )

    assert runtime.close_calls == 1  # nosec B101


@pytest.fixture(autouse=True)
def _test_mode(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")


@pytest.fixture()
def client_with_overrides(monkeypatch, auth_headers):
    async def override_user():
        return User(id=1, username="tester", email=None, is_active=True)

    async def _noop():
        return None

    # Disable RBAC enforcement to avoid DB access
    import tldw_Server_API.app.api.v1.API_Deps.auth_deps as auth_deps
    async def _no_rbac(*args, **kwargs):  # noqa: ARG001
        return None
    monkeypatch.setattr(auth_deps, "enforce_rbac_rate_limit", _no_rbac)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[check_rate_limit] = _noop
    # Avoid DB initialization by overriding DB deps to return None
    try:
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user as _get_chacha_db
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user as _get_media_db
        async def _none_media_db():
            return None
        async def _none_chacha_db():
            return None
        fastapi_app.dependency_overrides[_get_media_db] = _none_media_db
        fastapi_app.dependency_overrides[_get_chacha_db] = _none_chacha_db
    except Exception:
        _ = None

    try:
        # Prefer disabling lifespan to avoid DB/services startup in CI; skip if not supported
        import inspect as _inspect
        if 'lifespan' in _inspect.signature(TestClient.__init__).parameters:
            with TestClient(fastapi_app, headers=auth_headers, raise_server_exceptions=False, lifespan='off') as client:
                yield client
        else:
            with TestClient(fastapi_app, headers=auth_headers, raise_server_exceptions=False) as client:
                yield client
    finally:
        fastapi_app.dependency_overrides.clear()


def test_rag_ablate_smoke(client_with_overrides, monkeypatch):


    client = client_with_overrides

    # Patch retrievers for both unified_pipeline and agentic_chunker to return a simple doc
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

    class FakeRetriever:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            pass
        async def retrieve(self, *args, **kwargs):  # noqa: ARG002
            return [
                Document(
                    id="m1",
                    content="Residual connections enable gradient flow and stabilize deep networks.",
                    metadata={"title": "ResNet", "source": "media_db", "ingestion_date": "2024-01-01"},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]

    import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac
    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up
    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(up, "MultiDatabaseRetriever", FakeRetriever)

    # Run ablations
    resp = client.post(
        "/api/v1/rag/ablate",
        json={
            "query": "What do residual connections do?",
            "top_k": 5,
            "search_mode": "fts",
            "with_answer": False,
            "reranking_strategy": "none"
        },
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert isinstance(out.get("summary"), list) and len(out["summary"]) == 4
    runs = out.get("runs", [])
    assert len(runs) == 4
    labels = [r.get("label") for r in out["summary"]]
    assert set(labels) == {"baseline", "+rerank", "agentic", "agentic_strict"}

    # Verify agentic runs advertise strategy in metadata
    agentic_runs = [r for r in runs if r.get("label") in ("agentic", "agentic_strict")]
    for r in agentic_runs:
        md = r["result"].get("metadata", {})
        assert md.get("strategy") == "agentic"


def test_rag_ablate_capabilities_smoke(auth_headers):


     # Quick smoke to ensure capabilities advertises new agentic knobs
    with TestClient(fastapi_app, headers=auth_headers) as client:
        resp = client.get("/api/v1/rag/capabilities")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        agentic = data.get("features", {}).get("agentic_chunking", {})
        params = set(agentic.get("parameters", []))
        assert {"agentic_adaptive_budgets", "agentic_coverage_target", "agentic_min_corroborating_docs"}.issubset(params)
