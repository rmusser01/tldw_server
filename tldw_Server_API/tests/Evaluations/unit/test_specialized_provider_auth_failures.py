"""Fail-closed provider-auth regressions for specialized evaluations."""

from __future__ import annotations

import asyncio
import inspect
from collections import Counter
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.api.v1.endpoints.evaluations import (
    evaluations_unified as eval_endpoint,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
from tldw_Server_API.app.core.Evaluations import (
    ms_g_eval,
    rag_evaluator,
    response_quality_evaluator,
    unified_evaluation_service,
)
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import (
    UnifiedEvaluationService,
)
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import (
    SummaryProviderError,
)


async def _run_analyze_inline(function, *args: Any, **kwargs: Any):
    """Execute a sync or async evaluator adapter boundary inline."""

    result = function(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def test_geval_terminal_auth_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejected credentials must cause one adapter request, not ten retries."""

    calls = 0

    def reject_credentials(**_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        raise ChatAuthenticationError(
            provider="openai",
            message="hostile upstream body with sk-private-value",
        )

    monkeypatch.setattr(ms_g_eval, "_call_adapter_text", reject_credentials)
    monkeypatch.setattr(
        ms_g_eval,
        "wait_random_exponential",
        lambda **_kwargs: (lambda _retry_state: 0.0),
    )

    with pytest.raises(ChatAuthenticationError) as exc_info:
        ms_g_eval.geval_summarization(
            "judge this summary",
            5,
            "openai",
            "rejected-key",
            model="model-a",
            app_config={"openai_api": {"model": "model-a"}},
            credentials_resolved=True,
        )

    assert calls == 1
    assert exc_info.value.__cause__ is None
    assert "sk-private-value" not in repr(exc_info.value)
    assert "sk-private-value" not in repr(exc_info.value.__context__)


def test_geval_auth_failure_propagates_without_zero_score_persistence_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider rejection is an error, never a successful zero-score result."""

    calls = 0

    def reject_metric(*_args: Any, **_kwargs: Any) -> float:
        nonlocal calls
        calls += 1
        raise ChatAuthenticationError(provider="openai")

    monkeypatch.setattr(ms_g_eval, "geval_summarization", reject_metric)

    with pytest.raises(ChatAuthenticationError):
        ms_g_eval.run_geval(
            "source",
            "summary",
            api_key="rejected-key",
            api_name="openai",
            model="model-a",
            app_config={"openai_api": {"model": "model-a"}},
            credentials_resolved=True,
        )

    assert calls == 1


def test_geval_accepts_registered_bedrock_default_chain_auth() -> None:
    """G-Eval validation follows the registry and resolved Bedrock auth contract."""

    ms_g_eval.validate_inputs(
        "source",
        "summary",
        "bedrock",
        None,
        app_config={
            "bedrock_api": {
                "model": "anthropic.claude-test",
                "_runtime_auth_source": "aws_default_chain",
            }
        },
        credentials_resolved=True,
    )


@pytest.mark.asyncio
async def test_response_quality_auth_failure_uses_one_adapter_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first rejected metric must stop all remaining quality metrics."""

    calls = 0

    def reject_credentials(api_name: str, *_args: Any, **_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        raise SummaryProviderError(code="authentication", provider=api_name)

    async def call_inline(_provider: str, function, *args: Any, **kwargs: Any):
        return await _run_analyze_inline(function, *args, **kwargs)

    monkeypatch.setattr(response_quality_evaluator, "analyze", reject_credentials)
    monkeypatch.setattr(
        response_quality_evaluator.llm_circuit_breaker,
        "call_with_breaker",
        call_inline,
    )

    with pytest.raises(SummaryProviderError) as exc_info:
        await response_quality_evaluator.ResponseQualityEvaluator().evaluate(
            prompt="prompt",
            response="response",
            api_name="openai",
            api_key="rejected-key",
            model="model-a",
            app_config={"openai_api": {"model": "model-a"}},
            credentials_resolved=True,
        )

    assert calls == 1
    assert exc_info.value.code == "authentication"


@pytest.mark.asyncio
async def test_rag_auth_failure_uses_one_adapter_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected first RAG metric must prevent partial-result persistence."""

    calls = 0

    def reject_credentials(api_name: str, *_args: Any, **_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        raise SummaryProviderError(code="authentication", provider=api_name)

    async def call_inline(_provider: str, function, *args: Any, **kwargs: Any):
        return await _run_analyze_inline(function, *args, **kwargs)

    monkeypatch.setattr(rag_evaluator, "analyze", reject_credentials)
    monkeypatch.setattr(
        rag_evaluator.llm_circuit_breaker,
        "call_with_breaker",
        call_inline,
    )

    evaluator = rag_evaluator.RAGEvaluator(
        embedding_provider=None,
        embedding_model=None,
        api_key="rejected-key",
        app_config={"openai_api": {"model": "model-a"}},
        credentials_resolved=True,
    )
    with pytest.raises(SummaryProviderError) as exc_info:
        await evaluator.evaluate(
            query="query",
            contexts=["context"],
            response="response",
            metrics=["relevance", "faithfulness"],
            api_name="openai",
            model="model-a",
        )

    assert calls == 1
    assert exc_info.value.code == "authentication"


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation_kind", ["geval", "rag", "response_quality"])
async def test_concurrent_specialized_auth_failures_remain_request_scoped(
    monkeypatch: pytest.MonkeyPatch,
    evaluation_kind: str,
) -> None:
    """Concurrent rejected providers must not cross-contaminate error identity."""

    calls: Counter[str] = Counter()

    if evaluation_kind == "geval":
        def reject_geval(
            _prompt: str,
            _max_score: float,
            api_name: str,
            *_args: Any,
            **_kwargs: Any,
        ) -> float:
            calls[api_name] += 1
            raise ChatAuthenticationError(provider=api_name)

        monkeypatch.setattr(ms_g_eval, "geval_summarization", reject_geval)

        async def invoke(provider: str) -> BaseException:
            try:
                await asyncio.to_thread(
                    ms_g_eval.run_geval,
                    "source",
                    "summary",
                    "rejected-key",
                    provider,
                    False,
                    None,
                    "model-a",
                    {f"{provider}_api": {"model": "model-a"}},
                    True,
                )
            except (ChatAuthenticationError, SummaryProviderError) as exc:
                return exc
            raise AssertionError("rejected provider returned a successful evaluation")
    else:
        module = (
            rag_evaluator
            if evaluation_kind == "rag"
            else response_quality_evaluator
        )

        def reject_summary(api_name: str, *_args: Any, **_kwargs: Any) -> str:
            calls[api_name] += 1
            raise SummaryProviderError(code="authentication", provider=api_name)

        async def call_inline(_provider: str, function, *args: Any, **kwargs: Any):
            await asyncio.sleep(0)
            return await _run_analyze_inline(function, *args, **kwargs)

        monkeypatch.setattr(module, "analyze", reject_summary)
        monkeypatch.setattr(
            module.llm_circuit_breaker,
            "call_with_breaker",
            call_inline,
        )

        async def invoke(provider: str) -> BaseException:
            try:
                if evaluation_kind == "rag":
                    evaluator = rag_evaluator.RAGEvaluator(
                        embedding_provider=None,
                        embedding_model=None,
                        api_key="rejected-key",
                        app_config={f"{provider}_api": {"model": "model-a"}},
                        credentials_resolved=True,
                    )
                    await evaluator.evaluate(
                        query="query",
                        contexts=["context"],
                        response="response",
                        metrics=["relevance", "faithfulness"],
                        api_name=provider,
                        model="model-a",
                    )
                else:
                    await response_quality_evaluator.ResponseQualityEvaluator().evaluate(
                        prompt="prompt",
                        response="response",
                        api_name=provider,
                        api_key="rejected-key",
                        model="model-a",
                        app_config={f"{provider}_api": {"model": "model-a"}},
                        credentials_resolved=True,
                    )
            except (ChatAuthenticationError, SummaryProviderError) as exc:
                return exc
            raise AssertionError("rejected provider returned a successful evaluation")

    errors = await asyncio.gather(invoke("openai"), invoke("anthropic"))

    assert calls == Counter({"openai": 1, "anthropic": 1})
    assert [getattr(error, "provider", None) for error in errors] == [
        "openai",
        "anthropic",
    ]


class _AllowingLimiter:
    async def check_rate_limit(self, *_args: Any, **_kwargs: Any):
        return True, {"retry_after": 0}


class _NoopWebhookManager:
    async def send_webhook(self, **_kwargs: Any) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("evaluation_kind", "request_factory", "endpoint_name"),
    [
        (
            "geval",
            lambda provider: eval_endpoint.GEvalRequest(
                source_text="source text long enough",
                summary="summary text long enough",
                api_name=provider,
                model="model-a",
            ),
            "evaluate_geval",
        ),
        (
            "rag",
            lambda provider: eval_endpoint.RAGEvaluationRequest(
                query="query",
                retrieved_contexts=["context"],
                generated_response="response",
                api_name=provider,
                model="model-a",
            ),
            "evaluate_rag",
        ),
        (
            "response_quality",
            lambda provider: eval_endpoint.ResponseQualityRequest(
                prompt="prompt",
                response="response",
                api_name=provider,
                model="model-a",
            ),
            "evaluate_response_quality",
        ),
    ],
)
async def test_specialized_endpoint_maps_auth_to_502_without_marking_used(
    monkeypatch: pytest.MonkeyPatch,
    evaluation_kind: str,
    request_factory,
    endpoint_name: str,
) -> None:
    """Public endpoints expose a bounded 502 and never mark rejected keys used."""

    events: list[str] = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="rejected-key",
        app_config={"openai_api": {"model": "model-a"}},
        credentials_resolved=True,
    )

    class Runtime:
        async def mark_used(self, _handle: Any) -> None:
            events.append("mark")

        async def close(self) -> None:
            events.append("close")

    runtime = Runtime()

    async def resolve(request, *_args: Any, **_kwargs: Any):
        provider = request.api_name
        handle.provider = provider
        return provider, "model-a", handle, runtime

    class RejectingService:
        async def evaluate_geval(self, **_kwargs: Any):
            raise SummaryProviderError(code="authentication", provider="openai")

        async def evaluate_rag(self, **_kwargs: Any):
            raise SummaryProviderError(code="authentication", provider="openai")

        async def evaluate_response_quality(self, **_kwargs: Any):
            raise SummaryProviderError(code="authentication", provider="openai")

    monkeypatch.setattr(eval_endpoint, "_resolve_and_validate_eval_provider", resolve)
    monkeypatch.setattr(
        eval_endpoint,
        "get_user_rate_limiter_for_user",
        lambda _user_id: _AllowingLimiter(),
    )
    monkeypatch.setattr(
        eval_endpoint,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: RejectingService(),
    )
    monkeypatch.setattr(
        eval_endpoint,
        "_get_webhook_manager_for_user",
        lambda _user_id: _NoopWebhookManager(),
    )
    monkeypatch.setattr(eval_endpoint, "_is_eval_test_mode", lambda: True)

    endpoint = getattr(eval_endpoint, endpoint_name)
    with pytest.raises(eval_endpoint.HTTPException) as exc_info:
        await endpoint(
            request=request_factory("openai"),
            http_request=SimpleNamespace(),
            response=None,
            user_id="user-1",
            current_user=SimpleNamespace(id=1, id_str="user-1"),
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == {
        "error_code": "provider_authentication_failed",
        "message": "The selected provider credentials could not be authenticated.",
    }
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert events == ["close"]
    assert evaluation_kind in {"geval", "rag", "response_quality"}


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation_kind", ["geval", "rag", "response_quality"])
async def test_specialized_service_never_persists_provider_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    evaluation_kind: str,
) -> None:
    """Typed provider rejections must escape before the persistence boundary."""

    service = UnifiedEvaluationService(
        db_path=str(tmp_path / f"{evaluation_kind}.db"),
        enable_webhooks=False,
    )
    forbidden_store = AsyncMock(side_effect=AssertionError("auth failure persisted"))
    monkeypatch.setattr(service, "_store_evaluation_result", forbidden_store)

    if evaluation_kind == "geval":
        def reject_geval(**_kwargs: Any):
            raise ChatAuthenticationError(provider="openai")

        monkeypatch.setattr(ms_g_eval, "run_geval", reject_geval)
        evaluation = service.evaluate_geval(
            source_text="source",
            summary="summary",
            api_name="openai",
            api_key="rejected-key",
            model="model-a",
            app_config={"openai_api": {"model": "model-a"}},
            credentials_resolved=True,
        )
        expected_error = ChatAuthenticationError
    elif evaluation_kind == "rag":
        class RejectingRAGEvaluator:
            def __init__(self, **_kwargs: Any) -> None:
                return None

            async def evaluate(self, **_kwargs: Any):
                raise SummaryProviderError(code="authentication", provider="openai")

        monkeypatch.setattr(
            unified_evaluation_service,
            "RAGEvaluator",
            RejectingRAGEvaluator,
        )
        evaluation = service.evaluate_rag(
            query="query",
            contexts=["context"],
            response="response",
            metrics=["relevance"],
            api_name="openai",
            api_key="rejected-key",
            model="model-a",
            app_config={"openai_api": {"model": "model-a"}},
            credentials_resolved=True,
        )
        expected_error = SummaryProviderError
    else:
        class RejectingQualityEvaluator:
            async def evaluate(self, **_kwargs: Any):
                raise SummaryProviderError(code="authentication", provider="openai")

        monkeypatch.setattr(
            service,
            "get_quality_evaluator",
            lambda: RejectingQualityEvaluator(),
        )
        evaluation = service.evaluate_response_quality(
            prompt="prompt",
            response="response",
            api_name="openai",
            api_key="rejected-key",
            model="model-a",
            app_config={"openai_api": {"model": "model-a"}},
            credentials_resolved=True,
        )
        expected_error = SummaryProviderError

    with pytest.raises(expected_error):
        await evaluation

    forbidden_store.assert_not_awaited()
