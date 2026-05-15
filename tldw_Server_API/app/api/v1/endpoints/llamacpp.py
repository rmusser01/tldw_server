# llamacpp.py
# Description: This file contains the API endpoints for managing Llama.cpp server operations in tldw_Server_API.
#
# Imports
import inspect
from typing import Any, Optional

#
# Thid-party Libraries
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_request_user, RequireRole, User
from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import (
    LlamaCppConfigResponse,
    LlamaCppConfigUpdateRequest,
    LlamaCppValidationRequest,
    LlamaCppValidationResponse,
)

from tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler import LlamaCppHandler

#
# Local Imports
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import InferenceError, ModelNotFoundError, ServerError
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Manager import LLMInferenceManager
from tldw_Server_API.app.core.Local_LLM import llamacpp_config_service

#
########################################################################################################################
#
# Functions:

router = APIRouter()

#     LlamaCppConfig: Defines paths and default arguments for llama.cpp/server.
#
#     LlamaCpp_Handler:
#
#         Manages a single llama.cpp/server process (_active_server_process).
#
#         start_server(): This is your model swap function. If an existing server is running (managed by this handler), it calls stop_server() first, then starts a new server with the new model_filename and server_args.
#
#         stop_server(): Terminates the managed server process, handling process groups for robust cleanup.
#
#         inference(): Sends requests to the Llama.cpp server's OpenAI-compatible API (e.g., /v1/chat/completions).
#
#         list_models(): Scans the models_dir for .gguf files.
#
#         get_server_status(): Reports the current state of the managed server.
#
#         _cleanup_managed_server_sync(): Ensures server is stopped on application exit.
#
#         Optional logging of llama.cpp/server output to a file.
#
#     LLM_Inference_Manager Updates:
#
#         Initializes and provides access to LlamaCppHandler.
#
#         Delegates relevant calls (start_server, stop_server, run_inference, list_local_models) to the LlamaCppHandler.
#
#     API Endpoints: Provide HTTP interfaces to list models, start/swap the server with a specific model, stop it, get status, and run inference.

# Assuming 'llm_manager' is available, e.g., initialized in main.py and stored
# on app.state.llm_manager. Tests may still patch the module-level llm_manager
# for compatibility, so we fall back to the global when state is missing.


def _resolve_llm_manager(request: Request) -> LLMInferenceManager:
    mgr = getattr(request.app.state, "llm_manager", None)
    if mgr is None:
        mgr = globals().get("llm_manager")
    if mgr is None:
        raise HTTPException(status_code=503, detail="LLM manager not initialized.")
    return mgr  # type: ignore[return-value]


def _llamacpp_unavailable(detail: Optional[str] = None) -> HTTPException:
    base = "Managed llama.cpp backend is not configured."
    guidance = "Enable [LlamaCpp] enabled=true in Config_Files/config.txt and restart the server."
    safe_detail = "backend unavailable" if detail else None
    message = f"{base} ({safe_detail}) {guidance}" if safe_detail else f"{base} {guidance}"
    return HTTPException(status_code=503, detail=message)


def _resolve_llamacpp_target(llm_manager: LLMInferenceManager, required: tuple[str, ...]):
    """
    Return an object (handler or manager) that supports the required llama.cpp methods.
    Falls back to the manager for compatibility with tests that monkeypatch llm_manager directly.
    """
    handler = getattr(llm_manager, "llamacpp", None)
    candidates = [handler, llm_manager]
    for cand in candidates:
        if cand and all(hasattr(cand, name) for name in required):
            return cand
    raise _llamacpp_unavailable()


def _log_sanitized_manager_error(llm_manager: LLMInferenceManager, message: str) -> None:
    """Log a sanitized manager fallback without attaching exception details."""
    llm_manager.logger.error(message)


# --- Llama.cpp Specific Endpoints ---
@router.get(
    "/llamacpp/config",
    summary="Get llama.cpp Admin Config State",
    response_model=LlamaCppConfigResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def get_llamacpp_config_endpoint(llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager)):
    return llamacpp_config_service.get_config_state(llm_manager)


@router.put(
    "/llamacpp/config",
    summary="Update llama.cpp Admin Config",
    response_model=LlamaCppConfigResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def update_llamacpp_config_endpoint(
    payload: LlamaCppConfigUpdateRequest,
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
):
    return llamacpp_config_service.update_config_state(payload, llm_manager)


@router.post(
    "/llamacpp/validate",
    summary="Validate llama.cpp Binary",
    response_model=LlamaCppValidationResponse,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def validate_llamacpp_binary_endpoint(payload: LlamaCppValidationRequest):
    return llamacpp_config_service.validate_binary(payload.binary_path, payload.timeout_seconds)


@router.post(
    "/llamacpp/start_server",
    summary="Start or Swap Llama.cpp Server Model",
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def start_llamacpp_server_endpoint(
    model_filename: str = Body(
        ..., embed=True, description="Filename of the GGUF model to load (e.g., 'mistral-7b-v0.1.Q4_K_M.gguf')"
    ),
    server_args: Optional[dict[str, Any]] = Body(
        {}, embed=True, description="Optional Llama.cpp server arguments (e.g., port, n_gpu_layers)"
    ),
    llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager),
):
    """
    Starts the Llama.cpp server with the specified model.
    If a server is already running, it will be stopped and restarted with the new model (model swap).
    """
    try:
        target = _resolve_llamacpp_target(llm_manager, ("start_server",))
        # Prefer handler.start_server if available, else manager.start_server
        if isinstance(target, LlamaCppHandler):
            result = await target.start_server(model_filename=model_filename, server_args=server_args)
        else:
            result = await target.start_server(backend="llamacpp", model_name=model_filename, server_args=server_args)
        if isinstance(result, dict):
            result.setdefault("status", "started")
            result.setdefault("backend", "llamacpp")
        return result
    except HTTPException:
        raise
    except InferenceError as e:
        raise _llamacpp_unavailable(str(e)) from e
    except (ModelNotFoundError, ServerError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error starting Llama.cpp server")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


@router.post(
    "/llamacpp/stop_server",
    summary="Stop Llama.cpp Server",
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def stop_llamacpp_server_endpoint(llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager)):
    try:
        target = _resolve_llamacpp_target(llm_manager, ("stop_server",))
        if isinstance(target, LlamaCppHandler):
            result = await target.stop_server()
        else:
            result = await target.stop_server(backend="llamacpp")
        return {"status": "stopped", "message": result, "backend": "llamacpp"}
    except HTTPException:
        raise
    except InferenceError as e:
        raise _llamacpp_unavailable(str(e)) from e
    except ServerError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error stopping Llama.cpp server")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


@router.get(
    "/llamacpp/status",
    summary="Get Llama.cpp Server Status",
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def get_llamacpp_status_endpoint(llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager)):
    try:
        target = _resolve_llamacpp_target(llm_manager, ("get_server_status",))
        if isinstance(target, LlamaCppHandler):
            status = await target.get_server_status()
        else:
            status = await target.get_server_status(backend="llamacpp")  # Via manager
        if isinstance(status, dict):
            status.setdefault("backend", "llamacpp")
        return status
    except HTTPException:
        raise
    except InferenceError as e:
        raise _llamacpp_unavailable(str(e)) from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error getting Llama.cpp server status")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


@router.get(
    "/llamacpp/metrics",
    summary="Get Llama.cpp Metrics",
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def get_llamacpp_metrics_endpoint(llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager)):
    try:
        handler = _resolve_llamacpp_target(llm_manager, ("get_metrics",))
        metrics = handler.get_metrics()
        if inspect.isawaitable(metrics):
            metrics = await metrics
        if isinstance(metrics, dict):
            metrics.setdefault("backend", "llamacpp")
        return metrics
    except HTTPException:
        raise
    except InferenceError as e:
        raise _llamacpp_unavailable(str(e)) from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error getting Llama.cpp metrics")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


@router.get("/llamafile/metrics", summary="Get Llamafile Metrics")
async def get_llamafile_metrics_endpoint(llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager)):
    try:
        if not getattr(llm_manager, "llamafile", None):
            raise HTTPException(status_code=400, detail="Llamafile backend is not enabled or configured.")
        handler = llm_manager.llamafile
        if hasattr(handler, "get_metrics"):
            return handler.get_metrics()  # type: ignore[attr-defined]
        return {"message": "metrics not available"}
    except HTTPException:
        raise
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error getting Llamafile metrics")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


@router.get(
    "/llamacpp/models",
    summary="List available Llama.cpp models",
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def list_llamacpp_models_endpoint(llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager)):
    try:
        handler = getattr(llm_manager, "llamacpp", None)
        if handler is not None and hasattr(handler, "list_models"):
            models = await handler.list_models()
        elif hasattr(llm_manager, "list_local_models"):
            models = await llm_manager.list_local_models(backend="llamacpp")
        else:
            raise _llamacpp_unavailable()
        return {"available_models": models, "backend": "llamacpp"}
    except HTTPException:
        raise
    except InferenceError as e:
        raise _llamacpp_unavailable(str(e)) from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error listing Llama.cpp models")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


from tldw_Server_API.app.api.v1.schemas.llamacpp_schemas import LlamaCppInferenceRequest


@router.post("/llamacpp/inference", summary="Run inference with Llama.cpp")
async def run_llamacpp_inference_endpoint(
    payload: LlamaCppInferenceRequest, llm_manager: LLMInferenceManager = Depends(_resolve_llm_manager)
):
    """
    Runs inference using the currently loaded Llama.cpp model.
    Payload should be OpenAI compatible (e.g., include 'messages' list).
    Example: {"messages": [{"role": "user", "content": "Hello!"}], "temperature": 0.7}
    """
    try:
        handler = getattr(llm_manager, "llamacpp", None)
        # Prefer handler methods when available; fallback to manager for compatibility with tests
        if handler and hasattr(handler, "get_server_status") and hasattr(handler, "inference"):
            status = await handler.get_server_status()
            current_model = status.get("model", "unknown_active_model")
            result = await handler.inference(
                prompt=None,  # Assuming payload contains 'messages'
                messages=payload.messages,
                **payload.to_kwargs(),
            )
            # Align response model naming with manager-style return
            result.setdefault("model", current_model)
        elif hasattr(llm_manager, "get_server_status") and hasattr(llm_manager, "run_inference"):
            status = await llm_manager.get_server_status(backend="llamacpp")
            current_model = status.get("model", "unknown_active_model")
            result = await llm_manager.run_inference(
                backend="llamacpp",
                model_name_or_path=current_model,  # Contextual
                prompt=None,  # Assuming payload contains 'messages'
                **payload.to_kwargs(),  # Pass validated payload as kwargs (extras allowed)
            )
        else:
            raise _llamacpp_unavailable()
        if isinstance(result, dict):
            result.setdefault("backend", "llamacpp")
        return result
    except HTTPException:
        raise
    except InferenceError as e:
        raise _llamacpp_unavailable(str(e)) from e
    except ServerError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _log_sanitized_manager_error(llm_manager, "Unexpected error during Llama.cpp inference")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


# --- Llama.cpp Reranker (GGUF embeddings) ---


class LlamaCppRerankItem(BaseModel):
    id: Optional[str] = Field(default=None, description="Optional identifier for the passage")
    text: str = Field(..., min_length=1, description="Passage text to score")


class LlamaCppRerankRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query to rank against passages")
    passages: list[LlamaCppRerankItem] = Field(..., min_length=1, description="Candidate passages to rerank")
    top_k: Optional[int] = Field(
        default=None, ge=1, le=100, description="Top-K results to return (defaults to len(passages))"
    )
    # Optional overrides for llama.cpp and model selection
    model: Optional[str] = Field(default=None, description="GGUF model path (overrides config)")
    binary: Optional[str] = Field(default=None, description="llama-embedding binary name or path")
    ngl: Optional[int] = Field(default=None, ge=0, description="n_gpu_layers (-ngl)")
    separator: Optional[str] = Field(default=None, description="Text separator between query and passages")
    output_format: Optional[str] = Field(default=None, description="Embedding output format (e.g., json+)")
    pooling: Optional[str] = Field(default=None, description="Embedding pooling method (e.g., last)")
    normalize: Optional[int] = Field(default=None, description="Embedding normalize flag (-1, 0, 1)")
    max_doc_chars: Optional[int] = Field(default=None, ge=0, description="Max chars per passage (truncation)")
    # OpenAPI example
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What do llamas eat?",
                    "passages": [
                        {"id": "a", "text": "Llamas eat bananas"},
                        {"id": "b", "text": "Llamas in pyjamas"},
                        {"id": "c", "text": "A bowl of fruit salad"},
                    ],
                    "top_k": 2,
                    "model": "/models/Qwen3-Embedding-0.6B_f16.gguf",
                    "ngl": 99,
                    "separator": "<#sep#>",
                    "output_format": "json+",
                    "pooling": "last",
                    "normalize": -1,
                }
            ]
        }
    )


class LlamaCppRerankResult(BaseModel):
    id: Optional[str] = Field(default=None)
    index: int = Field(..., description="Index of the passage in input list")
    score: float = Field(..., ge=0.0, le=1.0)
    text: Optional[str] = Field(default=None, description="Original passage text (truncated)")


class LlamaCppRerankResponse(BaseModel):
    results: list[LlamaCppRerankResult]


@router.post(
    "/llamacpp/reranking",
    summary="Rerank passages with llama.cpp embeddings (GGUF)",
    response_model=LlamaCppRerankResponse,
    dependencies=[Depends(check_rate_limit)],
)
@router.post(
    "/llamacpp/rerank",
    summary="Rerank passages with llama.cpp embeddings (GGUF)",
    response_model=LlamaCppRerankResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def llamacpp_reranker_endpoint(payload: LlamaCppRerankRequest, current_user: User = Depends(get_request_user)):
    """
    Rerank passages using the llama.cpp embeddings binary (llama-embedding) with a GGUF embedding model
    such as Qwen3-Embedding-0.6B. Scores are cosine(query, passage) normalized to [0,1].
    """
    try:
        # Lazy imports to avoid extra startup cost
        from tldw_Server_API.app.core.RAG.rag_service.advanced_reranking import (
            RerankingConfig,
            RerankingStrategy,
            create_reranker,
        )
        from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to initialize reranking") from e

    # Build documents from passages
    documents: list[Document] = []
    for i, item in enumerate(payload.passages):
        documents.append(
            Document(
                id=item.id or str(i),
                content=item.text,
                metadata={"source": "llamacpp_reranker"},
                source=DataSource.MEDIA_DB,
                score=0.0,
            )
        )

    # Config and overrides
    cfg = RerankingConfig(
        strategy=RerankingStrategy.LLAMA_CPP,
        top_k=min(payload.top_k or len(documents), len(documents)) if documents else 0,
        model_name=payload.model,
    )
    if payload.binary is not None:
        cfg.llama_binary = payload.binary
    if payload.ngl is not None:
        cfg.llama_ngl = payload.ngl
    if payload.separator is not None:
        cfg.llama_embd_separator = payload.separator
    if payload.output_format is not None:
        cfg.llama_embd_output_format = payload.output_format
    if payload.pooling is not None:
        cfg.llama_pooling = payload.pooling
    if payload.normalize is not None:
        cfg.llama_normalize = payload.normalize
    if payload.max_doc_chars is not None:
        cfg.llama_max_doc_chars = payload.max_doc_chars

    reranker = create_reranker(RerankingStrategy.LLAMA_CPP, cfg)

    # Execute reranking
    try:
        # Support both async and sync reranker implementations
        rerank_fn = getattr(reranker, "rerank", None)
        if rerank_fn is None:
            raise RuntimeError("Invalid reranker: missing rerank() method")
        scored = rerank_fn(payload.query, documents)
        if hasattr(scored, "__await__"):
            scored = await scored
        # Enforce top_k even if underlying reranker returns more
        if isinstance(scored, list) and cfg.top_k:
            scored = scored[: cfg.top_k]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to rerank passages") from e

    # Convert results
    # Map back to original order indices
    id_to_index = {(p.id or str(i)): i for i, p in enumerate(payload.passages)}
    results: list[LlamaCppRerankResult] = []
    for sd in scored:
        pid = getattr(sd.document, "id", None)
        idx = id_to_index.get(pid, 0)
        results.append(
            LlamaCppRerankResult(
                id=pid,
                index=idx,
                score=float(getattr(sd, "rerank_score", 0.0)),
                text=getattr(sd.document, "content", None),
            )
        )

    return LlamaCppRerankResponse(results=results)


# Public aliases matching common reranker API shapes
public_router = APIRouter()


class PublicRerankRequest(BaseModel):
    model: Optional[str] = Field(
        default=None, description="Reranker model id/path (GGUF for llama.cpp or HF id for transformers)"
    )
    query: str = Field(..., min_length=1)
    documents: list[str] = Field(..., min_length=1, description="Documents (plain text) to rank")
    top_n: Optional[int] = Field(default=None, ge=1, le=100)
    backend: Optional[str] = Field(default="auto", description="Reranker backend: auto|llamacpp|transformers")
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model": "/models/Qwen3-Embedding-0.6B_f16.gguf",
                    "query": "What is panda?",
                    "top_n": 3,
                    "documents": [
                        "hi",
                        "it is a bear",
                        "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear ...",
                    ],
                }
            ]
        }
    )


class PublicRerankResponse(BaseModel):
    results: list[LlamaCppRerankResult]


async def _run_public_rerank(
    query: str, docs: list[str], model_override: Optional[str], top_k: Optional[int], backend: str
) -> list[LlamaCppRerankResult]:
    from tldw_Server_API.app.core.RAG.rag_service.advanced_reranking import (
        RerankingConfig,
        RerankingStrategy,
        create_reranker,
    )
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

    documents: list[Document] = [
        Document(id=str(i), content=txt, metadata={"source": "public_reranking"}, source=DataSource.MEDIA_DB)
        for i, txt in enumerate(docs)
    ]

    # Select backend
    strategy = RerankingStrategy.FLASHRANK
    model_name = model_override
    b = (backend or "auto").lower()
    is_gguf = bool(model_override and model_override.strip().lower().endswith(".gguf"))
    looks_hf_id = bool(model_override and "/" in model_override and not is_gguf)
    if b == "llamacpp" or is_gguf:
        strategy = RerankingStrategy.LLAMA_CPP
    elif b == "transformers" or looks_hf_id:
        strategy = RerankingStrategy.CROSS_ENCODER
    else:
        # Auto fallback: prefer transformers if it looks like HF id, else llama if gguf
        strategy = (
            RerankingStrategy.LLAMA_CPP
            if is_gguf
            else (RerankingStrategy.CROSS_ENCODER if looks_hf_id else RerankingStrategy.FLASHRANK)
        )

    cfg = RerankingConfig(
        strategy=strategy,
        top_k=min(top_k or len(documents), len(documents)) if documents else 0,
        model_name=model_name,
    )
    reranker = create_reranker(strategy, cfg)
    # Support both async and sync reranker implementations
    rerank_fn = getattr(reranker, "rerank", None)
    if rerank_fn is None:
        raise HTTPException(status_code=500, detail="Invalid reranker: missing rerank() method")
    scored = rerank_fn(query, documents)
    if hasattr(scored, "__await__"):
        scored = await scored
    # Enforce top_k even if underlying reranker returns more
    if isinstance(scored, list) and cfg.top_k:
        scored = scored[: cfg.top_k]
    out: list[LlamaCppRerankResult] = []
    for sd in scored:
        idx = int(getattr(sd.document, "id", "0")) if str(getattr(sd.document, "id", "0")).isdigit() else 0
        out.append(
            LlamaCppRerankResult(
                id=getattr(sd.document, "id", None),
                index=idx,
                score=float(getattr(sd, "rerank_score", 0.0)),
                text=getattr(sd.document, "content", None),
            )
        )
    return out


@public_router.post(
    "/v1/reranking",
    summary="Rerank documents according to a query",
    response_model=PublicRerankResponse,
    dependencies=[Depends(check_rate_limit)],
)
@public_router.post(
    "/v1/rerank",
    summary="Rerank documents according to a query",
    response_model=PublicRerankResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def public_reranking_endpoint(payload: PublicRerankRequest, current_user: User = Depends(get_request_user)):
    try:
        results = await _run_public_rerank(
            query=payload.query,
            docs=payload.documents,
            model_override=payload.model,
            top_k=payload.top_n,
            backend=(payload.backend or "auto"),
        )
        return PublicRerankResponse(results=results)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to rerank documents") from e


#
# End of llamacpp.py
##########################################################################################################################
