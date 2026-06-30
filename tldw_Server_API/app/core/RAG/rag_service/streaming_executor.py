from __future__ import annotations

import asyncio
import os
import types
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.RAG.rag_service.agentic_chunker import (
    agentic_rag_pipeline,
)
from tldw_Server_API.app.core.RAG.rag_service.agentic_execution import (
    build_agentic_execution_context,
)
from tldw_Server_API.app.core.RAG.rag_service.generation import generate_streaming_response
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    normalize_documents_for_generation,
    unified_rag_pipeline,
)

RAGStreamEvent = dict[str, Any]
PipelineCallable = Callable[..., Awaitable[Any]]
GenerationCallable = Callable[..., Awaitable[Any]]
_PUBLIC_STREAM_ERROR_MESSAGE = "Search failed due to an internal error."

_EXTRA_CONTROL_KEYS = {
    "build_agentic_execution_context",
    "generate_streaming_response",
    "request_defaults",
    "sync_retriever_overrides",
}


def _pipeline_context(extra_context: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in extra_context.items()
        if key not in _EXTRA_CONTROL_KEYS
    }


def _value(
    payload: dict[str, Any],
    request_defaults: dict[str, Any],
    key: str,
    fallback: Any = None,
) -> Any:
    value = payload.get(key)
    if value is not None:
        return value
    return request_defaults.get(key, fallback)


def _to_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _metadata_for_doc(doc: Any) -> dict[str, Any]:
    if isinstance(doc, dict):
        metadata = doc.get("metadata") or {}
    else:
        metadata = getattr(doc, "metadata", {}) or {}
    return metadata if isinstance(metadata, dict) else {}


def _id_for_doc(doc: Any) -> Any:
    if isinstance(doc, dict):
        return doc.get("id")
    return getattr(doc, "id", None)


def _score_for_doc(doc: Any) -> float:
    metadata = _metadata_for_doc(doc)
    raw_score = doc.get("score") if isinstance(doc, dict) else getattr(doc, "score", None)
    if raw_score is None:
        raw_score = metadata.get("score", 0.0)
    try:
        return float(raw_score or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _documents_from_result(result: Any) -> list[Any]:
    if isinstance(result, dict):
        documents = result.get("documents") or result.get("sources") or []
    else:
        documents = getattr(result, "documents", []) or []
    return list(documents)


def _normalize_research_event(event: Any) -> dict[str, Any]:
    event_type_raw = getattr(event, "event_type", "research_update")
    event_type = str(event_type_raw) if event_type_raw is not None else "research_update"
    if not event_type.startswith("research_"):
        event_type = f"research_{event_type}"
    data = getattr(event, "data", {})
    if not isinstance(data, dict):
        data = {"value": data}
    return {"type": event_type, "data": data}


async def _retrieve_standard_documents(
    *,
    resolved_request: ResolvedRAGRequest,
    retrieval_plan: RetrievalPlan,
    standard_pipeline: PipelineCallable,
    pipeline_kwargs: dict[str, Any],
    payload: dict[str, Any],
    progress_queue: asyncio.Queue[Any] | None = None,
) -> list[Any]:
    kwargs = dict(pipeline_kwargs)
    kwargs.setdefault("query", resolved_request.query)
    kwargs.setdefault("top_k", retrieval_plan.top_k)
    kwargs.setdefault("search_mode", retrieval_plan.search_mode)
    kwargs.setdefault("resolved_request", resolved_request)
    kwargs.setdefault("retrieval_plan", retrieval_plan)
    kwargs["enable_generation"] = False

    if progress_queue is not None and bool(payload.get("enable_research_progress", False)):
        async def _stream_research_progress(event: Any) -> None:
            await progress_queue.put(_normalize_research_event(event))

        kwargs["research_progress_callback"] = _stream_research_progress
        kwargs["enable_research_progress"] = True

    retrieval_result = await standard_pipeline(**kwargs)
    return normalize_documents_for_generation(_documents_from_result(retrieval_result))


async def _retrieve_with_progress_events(
    *,
    resolved_request: ResolvedRAGRequest,
    retrieval_plan: RetrievalPlan,
    standard_pipeline: PipelineCallable,
    pipeline_kwargs: dict[str, Any],
    payload: dict[str, Any],
) -> AsyncIterator[RAGStreamEvent | list[Any]]:
    progress_queue: asyncio.Queue[Any] = asyncio.Queue()
    done_marker = object()
    error_marker = object()

    async def _runner() -> None:
        try:
            docs = await _retrieve_standard_documents(
                resolved_request=resolved_request,
                retrieval_plan=retrieval_plan,
                standard_pipeline=standard_pipeline,
                pipeline_kwargs=pipeline_kwargs,
                payload=payload,
                progress_queue=progress_queue,
            )
            await progress_queue.put(docs)
        except Exception as exc:  # noqa: BLE001 - propagate after queued progress drains
            await progress_queue.put((error_marker, exc))
        finally:
            await progress_queue.put(done_marker)

    retrieval_task = asyncio.create_task(_runner())
    docs: list[Any] = []
    pending_error: Exception | None = None
    try:
        while True:
            queued = await progress_queue.get()
            if queued is done_marker:
                break
            if isinstance(queued, tuple) and len(queued) == 2 and queued[0] is error_marker:
                pending_error = queued[1]
                continue
            if isinstance(queued, list):
                docs = queued
                continue
            yield queued
    finally:
        if not retrieval_task.done():
            retrieval_task.cancel()
            try:
                await retrieval_task
            except asyncio.CancelledError:
                pass
    await retrieval_task
    if pending_error is not None:
        raise pending_error
    yield docs


async def _prefetch_documents(
    *,
    resolved_request: ResolvedRAGRequest,
    retrieval_plan: RetrievalPlan,
    standard_pipeline: PipelineCallable,
    pipeline_kwargs: dict[str, Any],
    payload: dict[str, Any],
) -> AsyncIterator[RAGStreamEvent | list[Any]]:
    if bool(payload.get("enable_research_progress", False)):
        async for item in _retrieve_with_progress_events(
            resolved_request=resolved_request,
            retrieval_plan=retrieval_plan,
            standard_pipeline=standard_pipeline,
            pipeline_kwargs=pipeline_kwargs,
            payload=payload,
        ):
            yield item
        return

    docs = await _retrieve_standard_documents(
        resolved_request=resolved_request,
        retrieval_plan=retrieval_plan,
        standard_pipeline=standard_pipeline,
        pipeline_kwargs=pipeline_kwargs,
        payload=payload,
    )
    yield docs


async def _run_agentic_prefetch(
    *,
    resolved_request: ResolvedRAGRequest,
    retrieval_plan: RetrievalPlan,
    agentic_pipeline: PipelineCallable,
    pipeline_kwargs: dict[str, Any],
    payload: dict[str, Any],
    request_defaults: dict[str, Any],
    context_builder: Callable[..., tuple[dict[str, Any], Any]],
) -> tuple[list[Any], list[RAGStreamEvent]]:
    agentic_payload, agentic_cfg = context_builder(
        resolved_request=resolved_request,
        retrieval_plan=retrieval_plan,
        payload_override=payload,
    )
    result = await agentic_pipeline(
        query=resolved_request.query,
        sources=list(retrieval_plan.sources),
        media_db=pipeline_kwargs.get("media_db"),
        chacha_db=pipeline_kwargs.get("chacha_db"),
        media_db_path=pipeline_kwargs.get("media_db_path"),
        notes_db_path=pipeline_kwargs.get("notes_db_path"),
        character_db_path=pipeline_kwargs.get("character_db_path"),
        kanban_db_path=pipeline_kwargs.get("kanban_db_path"),
        search_mode=retrieval_plan.search_mode,
        fts_level=_value(agentic_payload, request_defaults, "fts_level"),
        hybrid_alpha=_value(agentic_payload, request_defaults, "hybrid_alpha"),
        top_k=retrieval_plan.top_k,
        min_score=retrieval_plan.min_score,
        index_namespace=retrieval_plan.index_namespace,
        agentic=agentic_cfg,
        enable_generation=False,
        enable_citations=False,
        include_chunk_citations=False,
        debug_mode=bool(_value(agentic_payload, request_defaults, "debug_mode", False)),
        explain_only=bool(_value(agentic_payload, request_defaults, "explain_only", False)),
        resolved_request=resolved_request,
        retrieval_plan=retrieval_plan,
    )

    metadata = getattr(result, "metadata", {}) if result is not None else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    events: list[RAGStreamEvent] = []
    plan = metadata.get("agentic_metrics", {})
    events.append({"type": "plan", "plan": plan})
    provenance = metadata.get("provenance")
    if provenance:
        events.append({"type": "spans", "count": len(provenance), "provenance": provenance[:50]})
    docs = normalize_documents_for_generation(_documents_from_result(result))
    return docs, events


def _context_events(
    *,
    docs: list[Any],
    payload: dict[str, Any],
    request_defaults: dict[str, Any],
) -> list[RAGStreamEvent]:
    top_k_requested = _value(payload, request_defaults, "top_k", 10)
    top_k_limit = min(10, _to_int(top_k_requested, 10))

    top_contexts = []
    for doc in (docs or [])[:top_k_limit]:
        metadata = _metadata_for_doc(doc)
        top_contexts.append(
            {
                "id": _id_for_doc(doc),
                "title": metadata.get("title"),
                "score": _score_for_doc(doc),
                "url": metadata.get("url"),
                "source": metadata.get("source"),
            }
        )

    scores = [_score_for_doc(doc) for doc in (docs or [])]
    topicality = 0.0
    if scores:
        score_min, score_max = min(scores), max(scores)
        topicality = sum(
            (score - score_min) / (score_max - score_min) if score_max > score_min else 1.0
            for score in scores
        ) / len(scores)

    why = {
        "topicality": round(float(topicality), 4),
        "diversity": None,
        "freshness": None,
    }
    rationale = {
        "plan": [
            "Gather top-k contexts",
            f"Rerank using strategy={payload.get('reranking_strategy', 'flashrank')}",
            "Ground claims from sources",
            "Synthesize final answer",
        ]
    }
    return [
        {"type": "contexts", "contexts": top_contexts, "why": why},
        {"type": "reasoning", **rationale},
    ]


def _generation_config(
    *,
    payload: dict[str, Any],
    request_defaults: dict[str, Any],
) -> dict[str, Any]:
    try:
        from tldw_Server_API.app.core.config import load_and_log_configs

        cfg = load_and_log_configs() or {}
    except Exception as config_error:  # noqa: BLE001 - config load is best-effort in streaming path
        logger.opt(exception=config_error).warning(
            "RAG streaming config load failed; using request/env defaults"
        )
        cfg = {}

    request_provider = _value(payload, request_defaults, "generation_provider")
    env_provider = os.getenv("RAG_DEFAULT_LLM_PROVIDER")
    provider_value = request_provider if isinstance(request_provider, str) else (
        env_provider if env_provider is not None else cfg.get("RAG_DEFAULT_LLM_PROVIDER")
    )
    provider = (
        provider_value.strip()
        if isinstance(provider_value, str) and provider_value.strip()
        else "openai"
    )

    request_model = _value(payload, request_defaults, "generation_model")
    env_model = os.getenv("RAG_DEFAULT_LLM_MODEL")
    model_value = request_model if isinstance(request_model, str) and request_model else (
        env_model if env_model is not None else cfg.get("RAG_DEFAULT_LLM_MODEL")
    )
    model = (
        model_value.strip()
        if isinstance(model_value, str) and model_value.strip()
        else "gpt-4o-mini"
    )

    max_tokens = _to_int(_value(payload, request_defaults, "max_generation_tokens", 500), 500)
    config = {
        "streaming": True,
        "provider": provider,
        "model": model,
        "max_tokens": max_tokens,
    }
    prompt_template = _value(payload, request_defaults, "generation_prompt")
    if isinstance(prompt_template, str) and prompt_template:
        config["prompt_template"] = prompt_template
    return config


async def _stream_generation_events(
    *,
    resolved_request: ResolvedRAGRequest,
    docs: list[Any],
    payload: dict[str, Any],
    request_defaults: dict[str, Any],
    generation_streamer: GenerationCallable,
) -> AsyncIterator[RAGStreamEvent]:
    context = types.SimpleNamespace()
    context.documents = docs
    context.query = resolved_request.query
    context.config = {
        "generation": _generation_config(payload=payload, request_defaults=request_defaults)
    }
    context.metadata = {}

    await generation_streamer(
        context,
        enable_claims=bool(_value(payload, request_defaults, "enable_claims", False)),
        claims_top_k=_value(payload, request_defaults, "claims_top_k"),
        claims_max=_value(payload, request_defaults, "claims_max"),
        claims_concurrency=_value(payload, request_defaults, "claims_concurrency"),
    )

    last_overlay = None
    async for chunk in context.stream_generator:
        yield {"type": "delta", "text": chunk}
        overlay = context.metadata.get("claims_overlay")
        if overlay and overlay != last_overlay:
            yield {"type": "claims_overlay", **overlay}
            last_overlay = overlay

    final_overlay = context.metadata.get("claims_overlay")
    if final_overlay:
        yield {"type": "final_claims", **final_overlay}


async def stream_rag_events(
    *,
    resolved_request: ResolvedRAGRequest,
    retrieval_plan: RetrievalPlan,
    standard_pipeline: PipelineCallable = unified_rag_pipeline,
    agentic_pipeline: PipelineCallable = agentic_rag_pipeline,
    extra_context: dict[str, Any] | None = None,
) -> AsyncIterator[RAGStreamEvent]:
    context = dict(extra_context or {})
    payload = dict(resolved_request.payload or {})
    request_defaults = dict(context.get("request_defaults") or {})
    pipeline_kwargs = _pipeline_context(context)

    generation_streamer: GenerationCallable = context.get(
        "generate_streaming_response",
        generate_streaming_response,
    )
    context_builder = context.get(
        "build_agentic_execution_context",
        build_agentic_execution_context,
    )
    sync_retriever_overrides = context.get("sync_retriever_overrides")

    try:
        if callable(sync_retriever_overrides):
            sync_retriever_overrides()

        docs: list[Any] = []
        if str(resolved_request.strategy).strip().lower() == "agentic":
            try:
                docs, agentic_events = await _run_agentic_prefetch(
                    resolved_request=resolved_request,
                    retrieval_plan=retrieval_plan,
                    agentic_pipeline=agentic_pipeline,
                    pipeline_kwargs=pipeline_kwargs,
                    payload=payload,
                    request_defaults=request_defaults,
                    context_builder=context_builder,
                )
                for event in agentic_events:
                    yield event
            except asyncio.CancelledError:
                raise
            except Exception as agentic_error:  # noqa: BLE001 - agentic streaming prefetch is best-effort
                logger.debug(
                    "Agentic streaming prefetch failed; continuing with empty contexts",
                    exc_info=agentic_error,
                )
                docs = []
        else:
            try:
                async for item in _prefetch_documents(
                    resolved_request=resolved_request,
                    retrieval_plan=retrieval_plan,
                    standard_pipeline=standard_pipeline,
                    pipeline_kwargs=pipeline_kwargs,
                    payload=payload,
                ):
                    if isinstance(item, list):
                        docs = item
                    else:
                        yield item
            except asyncio.CancelledError:
                raise
            except Exception as prefetch_error:  # noqa: BLE001 - retrieval prefetch is best-effort for streaming
                logger.debug(
                    "RAG streaming standard prefetch failed; continuing with empty contexts",
                    exc_info=prefetch_error,
                )
                docs = []

        for event in _context_events(
            docs=docs,
            payload=payload,
            request_defaults=request_defaults,
        ):
            yield event

        async for event in _stream_generation_events(
            resolved_request=resolved_request,
            docs=docs,
            payload=payload,
            request_defaults=request_defaults,
            generation_streamer=generation_streamer,
        ):
            yield event
    except Exception as exc:  # noqa: BLE001 - streaming should surface error payload instead of crashing
        logger.exception("RAG streaming failed")
        yield {
            "type": "error",
            "message": _PUBLIC_STREAM_ERROR_MESSAGE,
        }
