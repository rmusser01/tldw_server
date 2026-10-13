"""Credential-runtime coverage for authenticated RAG convenience endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from loguru import logger
from starlette.requests import Request

import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_endpoint
import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as unified_pipeline
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import UnifiedSearchResult
from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

pytestmark = pytest.mark.integration

_SENTINEL = "rag-convenience-runtime-secret"


class _CountingRuntime(ProviderCredentialRuntime):
    __slots__ = ("close_calls",)

    def __init__(self) -> None:
        self.close_calls = 0

        async def resolver(provider: str, **_kwargs: Any) -> ResolvedByokCredentials:
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

        def reject_server_fallback(_provider: str) -> None:
            raise AssertionError("authenticated convenience endpoints must not use server credentials")

        super().__init__(
            user_id=42,
            team_ids=[],
            org_ids=[],
            trusted_base_url_override=False,
            fallback_resolver=reject_server_fallback,
            resolver=resolver,
        )

    async def close(self) -> None:
        self.close_calls += 1
        await super().close()


def _request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": [],
            "query_string": b"",
        }
    )


def _user() -> Any:
    return SimpleNamespace(id=42, id_int=42, username="rag-user")


def _db(path: str) -> Any:
    return SimpleNamespace(db_path=path)


def _install_endpoint_fakes(
    monkeypatch: pytest.MonkeyPatch,
    runtime: _CountingRuntime,
) -> None:
    monkeypatch.setattr(rag_endpoint, "_build_credential_runtime", lambda *_args: runtime)
    monkeypatch.setattr(rag_endpoint, "_resolve_kanban_db_path", lambda _user: "kanban.db")

    async def no_usage_log(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(rag_endpoint, "_log_rag_queries_for_org", no_usage_log)
    monkeypatch.setattr(rag_endpoint, "rag_result_from_unified_search_result", lambda result: result)
    monkeypatch.setattr(rag_endpoint, "rag_result_to_response", lambda result: {"query": result.query})


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_name", ["simple", "advanced"])
async def test_authenticated_convenience_endpoint_passes_one_real_runtime_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_name: str,
) -> None:
    runtime = _CountingRuntime()
    captured: list[ProviderCredentialRuntime] = []
    logs: list[str] = []
    _install_endpoint_fakes(monkeypatch, runtime)

    async def provider_boundary(*_args: Any, **kwargs: Any) -> Any:
        received = kwargs["credential_runtime"]
        captured.append(received)
        handle = await received.resolve("openai")
        assert handle.api_key == _SENTINEL  # nosec B101
        if endpoint_name == "simple":
            return []
        return UnifiedSearchResult(documents=[], query="credential runtime")

    monkeypatch.setattr(rag_endpoint, f"{endpoint_name}_search", provider_boundary)

    sink_id = logger.add(logs.append, format="{message}")
    try:
        if endpoint_name == "simple":
            response = await rag_endpoint.simple_search_endpoint(
                request=_request("/api/v1/rag/simple"),
                query="credential runtime",
                current_user=_user(),
                media_db=_db("media.db"),
                chacha_db=_db("notes.db"),
            )
        else:
            response = await rag_endpoint.advanced_search_endpoint(
                request=_request("/api/v1/rag/advanced"),
                query="credential runtime",
                current_user=_user(),
                media_db=_db("media.db"),
                chacha_db=_db("notes.db"),
            )
    finally:
        logger.remove(sink_id)

    assert captured == [runtime]  # nosec B101
    assert runtime.close_calls == 1  # nosec B101
    assert _SENTINEL not in repr(response)  # nosec B101
    rendered_logs = "".join(logs)
    assert "credential runtime" not in rendered_logs  # nosec B101
    assert "query_len=18" in rendered_logs  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_name", ["simple", "advanced"])
async def test_convenience_endpoint_maps_typed_provider_failure_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_name: str,
) -> None:
    runtime = _CountingRuntime()
    _install_endpoint_fakes(monkeypatch, runtime)

    async def fail_closed(*_args: Any, **_kwargs: Any) -> Any:
        raise ByokResolutionError("invalid_provider_credentials", "openai")

    monkeypatch.setattr(rag_endpoint, f"{endpoint_name}_search", fail_closed)

    with pytest.raises(HTTPException) as exc_info:
        if endpoint_name == "simple":
            await rag_endpoint.simple_search_endpoint(
                request=_request("/api/v1/rag/simple"),
                query="credential runtime",
                current_user=_user(),
                media_db=_db("media.db"),
                chacha_db=_db("notes.db"),
            )
        else:
            await rag_endpoint.advanced_search_endpoint(
                request=_request("/api/v1/rag/advanced"),
                query="credential runtime",
                current_user=_user(),
                media_db=_db("media.db"),
                chacha_db=_db("notes.db"),
            )

    assert exc_info.value.status_code == 503  # nosec B101
    assert exc_info.value.detail["error_code"] == "invalid_provider_credentials"  # nosec B101
    assert runtime.close_calls == 1  # nosec B101


@pytest.mark.asyncio
async def test_advanced_search_with_answer_refuses_an_over_quota_user(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _CountingRuntime()
    ran: list[bool] = []
    _install_endpoint_fakes(monkeypatch, runtime)

    async def recorder(*_args: Any, **_kwargs: Any) -> Any:
        ran.append(True)
        return UnifiedSearchResult(documents=[], query="q")

    monkeypatch.setattr(rag_endpoint, "advanced_search", recorder)
    refuse_llm_work(monkeypatch)

    with pytest.raises(LLMQuotaExceededError):
        await rag_endpoint.advanced_search_endpoint(
            request=_request("/api/v1/rag/advanced"),
            query="q",
            current_user=_user(),
            media_db=_db("media.db"),
            chacha_db=_db("notes.db"),
        )

    assert ran == []  # nosec B101
    assert runtime.close_calls == 0  # nosec B101


@pytest.mark.asyncio
async def test_advanced_search_without_answer_is_not_refused_by_the_llm_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _CountingRuntime()
    ran: list[bool] = []
    _install_endpoint_fakes(monkeypatch, runtime)

    async def recorder(*_args: Any, **_kwargs: Any) -> Any:
        ran.append(True)
        return UnifiedSearchResult(documents=[], query="q")

    monkeypatch.setattr(rag_endpoint, "advanced_search", recorder)
    refuse_llm_work(monkeypatch)

    await rag_endpoint.advanced_search_endpoint(
        request=_request("/api/v1/rag/advanced"),
        query="q",
        with_answer=False,
        current_user=_user(),
        media_db=_db("media.db"),
        chacha_db=_db("notes.db"),
    )

    assert ran == [True]  # nosec B101


@pytest.mark.asyncio
async def test_simple_search_does_not_generate_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def recorder(**kwargs: Any) -> UnifiedSearchResult:
        seen.update(kwargs)
        return UnifiedSearchResult(documents=[], query="q")

    monkeypatch.setattr(unified_pipeline, "unified_rag_pipeline", recorder)

    await unified_pipeline.simple_search("q")

    assert seen["enable_generation"] is False  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_name", ["simple", "advanced"])
async def test_convenience_endpoint_passes_an_llm_quota_refusal_through(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_name: str,
) -> None:
    """A refusal raised inside the search keeps its 402; it is not turned into a 500 (spec §4b)."""
    runtime = _CountingRuntime()
    _install_endpoint_fakes(monkeypatch, runtime)

    async def refused(*_args: Any, **_kwargs: Any) -> Any:
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    monkeypatch.setattr(rag_endpoint, f"{endpoint_name}_search", refused)
    endpoint = rag_endpoint.simple_search_endpoint if endpoint_name == "simple" else rag_endpoint.advanced_search_endpoint

    kwargs = {} if endpoint_name == "simple" else {"with_answer": False}
    with pytest.raises(LLMQuotaExceededError):
        await endpoint(
            request=_request(f"/api/v1/rag/{endpoint_name}"),
            query="q",
            current_user=_user(),
            media_db=_db("media.db"),
            chacha_db=_db("notes.db"),
            **kwargs,
        )

    assert runtime.close_calls == 1  # nosec B101
