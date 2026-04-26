# eval_runner.py - Async evaluation runner for OpenAI-compatible API
"""
Orchestrates evaluation runs asynchronously.

Handles:
- Async task execution
- Progress tracking
- Result aggregation
- Integration with existing evaluation backends
"""

import asyncio
import os
import statistics
import time
from contextlib import suppress
from datetime import datetime
from typing import Any, Callable, Optional

from loguru import logger

try:
    import httpx
except ImportError:
    httpx = None

try:
    import aiohttp
except ImportError:
    aiohttp = None

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.Chat.chat_helpers import extract_response_content
from tldw_Server_API.app.core.Chunking import chunk_for_embedding
from tldw_Server_API.app.core.Chunking.utils.proposition_eval import evaluate_propositions as eval_propositions
from tldw_Server_API.app.core.DB_Management.DB_Manager import (
    create_evaluations_database as _create_evals_db,
)

# Import existing evaluation modules
from tldw_Server_API.app.core.Evaluations.ms_g_eval import run_geval
from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
from tldw_Server_API.app.core.Evaluations.response_quality_evaluator import ResponseQualityEvaluator
from tldw_Server_API.app.core.http_client import RetryPolicy, afetch
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    ensure_app_config,
    get_adapter_or_raise,
    normalize_provider,
    resolve_provider_api_key_from_config,
    resolve_provider_model,
    split_system_message,
)
from tldw_Server_API.app.core.Evaluations.run_state import (
    can_transition_run_status,
    normalize_run_status,
)
from tldw_Server_API.app.core.RAG.rag_custom_metrics import get_custom_metrics
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.app.core.RAG.rag_service.vector_stores import (
    create_from_settings_for_user,
)

_HTTPX_ERROR = getattr(httpx, "HTTPError", None) if httpx is not None else None
_AIOHTTP_ERROR = getattr(aiohttp, "ClientError", None) if aiohttp is not None else None
_OPTIONAL_HTTP_EXCEPTIONS = tuple(
    exc for exc in (_HTTPX_ERROR, _AIOHTTP_ERROR) if exc is not None
)

_EVAL_RUNNER_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    ChatConfigurationError,
) + _OPTIONAL_HTTP_EXCEPTIONS

_EVAL_EMBEDDINGS_IMPORT_EXCEPTIONS = (ImportError, OSError, RuntimeError)

# Safe import of embeddings backend to avoid heavy deps at app import time
try:
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
        create_embeddings_batch,
        get_embedding_config,
    )
    _EVAL_EMBEDDINGS_AVAILABLE = True
except _EVAL_EMBEDDINGS_IMPORT_EXCEPTIONS:
    _EVAL_EMBEDDINGS_AVAILABLE = False
    def get_embedding_config():  # type: ignore[misc]
        return {"embedding_config": {"default_model_id": ""}}
    def create_embeddings_batch(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Embeddings backend unavailable; install required dependencies")


def _call_adapter_text(
    *,
    api_endpoint: str,
    messages_payload: list[dict[str, Any]],
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    system_message: Optional[str] = None,
    response_format: Optional[dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
    user: Optional[str] = None,
    app_config: Optional[dict[str, Any]] = None,
    timeout: Optional[float] = None,
    **extra_kwargs: Any,
) -> str:
    provider = normalize_provider(api_endpoint)
    if not provider:
        raise ChatConfigurationError(provider=api_endpoint, message="LLM provider is required.")
    cfg = ensure_app_config(app_config)
    resolved_model = model or resolve_provider_model(provider, cfg)
    if not resolved_model:
        raise ChatConfigurationError(provider=provider, message="Model is required for provider.")
    system_msg = system_message
    cleaned_messages = messages_payload or []
    if not system_msg:
        system_msg, cleaned_messages = split_system_message(cleaned_messages)
    request: dict[str, Any] = {
        "messages": cleaned_messages,
        "system_message": system_msg,
        "model": resolved_model,
        "api_key": api_key or resolve_provider_api_key_from_config(provider, cfg),
        "temperature": temperature,
        "response_format": response_format,
        "max_tokens": max_tokens,
        "user": user,
        "app_config": cfg,
    }
    request.update(extra_kwargs)
    response = get_adapter_or_raise(provider).chat(request, timeout=timeout)
    return extract_response_content(response) or str(response)



class EvaluationRunner:
    """Manages asynchronous evaluation execution"""

    def __init__(self, db_path: str, max_concurrent_evals: int = 10, eval_timeout: int = 60):
        """
        Initialize evaluation runner with concurrency controls.

        Args:
            db_path: Path to evaluations database
            max_concurrent_evals: Maximum number of concurrent evaluations
            eval_timeout: Timeout in seconds for each evaluation
        """
        # Use backend-aware factory so Postgres content backend is honored
        self.db = _create_evals_db(db_path=db_path)
        # Lazy initialization - create evaluators only when needed
        self._rag_evaluator = None
        self._quality_evaluator = None
        self.running_tasks = {}  # Track running evaluations

        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent_evals)
        self.eval_timeout = eval_timeout

    def _resolve_eval_provider_model_api_key(
        self,
        eval_spec: dict[str, Any],
        eval_config: dict[str, Any],
    ) -> tuple[str, Optional[str], Optional[str]]:
        """Resolve provider, model override, and api_key for evaluator calls."""
        run_config = eval_config.get("config") if isinstance(eval_config, dict) else None
        if hasattr(run_config, "model_dump"):
            run_config = run_config.model_dump()
        elif hasattr(run_config, "dict"):
            run_config = run_config.dict()
        if not isinstance(run_config, dict):
            run_config = {}

        api_name = (
            eval_spec.get("api_name")
            or eval_spec.get("provider")
            or run_config.get("api_name")
            or run_config.get("provider")
            or "openai"
        )
        api_name = str(api_name).strip() or "openai"
        api_key = (
            eval_spec.get("api_key")
            or eval_spec.get("apiKey")
            or run_config.get("api_key")
        )
        model_override = eval_spec.get("model") or run_config.get("model")

        # Backward compatibility: evaluator_model may refer to provider OR model
        evaluator_model = eval_spec.get("evaluator_model")
        if evaluator_model:
            normalized_provider = normalize_provider(str(evaluator_model))
            if normalized_provider:
                api_name = normalized_provider
            elif not model_override:
                model_override = str(evaluator_model)

        # Allow model strings like "provider:model" or "provider/model" to override provider
        if isinstance(model_override, str) and model_override:
            for sep in (":", "/"):
                if sep in model_override:
                    prefix, rest = model_override.split(sep, 1)
                    normalized_provider = normalize_provider(prefix)
                    if normalized_provider and rest:
                        api_name = normalized_provider
                        model_override = rest
                    break

        return api_name, model_override, api_key

    @staticmethod
    def _normalize_metrics(metrics: Any, default: Optional[list[str]] = None) -> list[str]:
        """Normalize metric configuration to a clean list of metric names."""
        if metrics is None:
            return list(default or [])
        if isinstance(metrics, str):
            metric_name = metrics.strip()
            return [metric_name] if metric_name else []
        if isinstance(metrics, (list, tuple, set)):
            normalized_metrics: list[str] = []
            for metric in metrics:
                if metric is None:
                    continue
                metric_name = str(metric).strip()
                if metric_name:
                    normalized_metrics.append(metric_name)
            return normalized_metrics
        return list(default or [])

    @property
    def rag_evaluator(self) -> RAGEvaluator:
        """Get or create RAG evaluator instance (lazy initialization)."""
        if self._rag_evaluator is None:
            self._rag_evaluator = RAGEvaluator()
        return self._rag_evaluator

    @property
    def quality_evaluator(self) -> ResponseQualityEvaluator:
        """Get or create quality evaluator instance (lazy initialization)."""
        if self._quality_evaluator is None:
            self._quality_evaluator = ResponseQualityEvaluator()
        return self._quality_evaluator

    async def run_evaluation(
        self,
        run_id: str,
        eval_id: str,
        eval_config: dict[str, Any],
        background: bool = True
    ):
        """
        Run an evaluation asynchronously.

        Args:
            run_id: The run ID
            eval_id: The evaluation ID
            eval_config: Configuration including eval_type, samples, etc.
            background: If True, run in background task
        """
        if background:
            # Start background task
            task = asyncio.create_task(self._execute_evaluation(run_id, eval_id, eval_config))
            self.running_tasks[run_id] = task
            return {"status": "started", "run_id": run_id}
        else:
            # Run synchronously (for testing)
            return await self._execute_evaluation(run_id, eval_id, eval_config)

    async def _execute_evaluation(
        self,
        run_id: str,
        eval_id: str,
        eval_config: dict[str, Any]
    ):
        """Execute the evaluation"""
        try:
            # Update status to running
            self.db.update_run_status(run_id, "running")
            start_time = time.time()

            # Get evaluation details
            evaluation = self.db.get_evaluation(eval_id)
            if not evaluation:
                raise ValueError(f"Evaluation {eval_id} not found")

            eval_type = evaluation["eval_type"]
            eval_spec = evaluation["eval_spec"]

            # Get samples
            samples = await self._get_samples(evaluation, eval_config)
            total_samples = len(samples)

            # Initialize progress
            progress = {
                "total_samples": total_samples,
                "completed_samples": 0,
                "failed_samples": 0,
                "current_batch": 0
            }
            self.db.update_run_progress(run_id, progress)

            # Special-case: rag_pipeline orchestrates across configs and samples
            sub_type = None
            try:
                sub_type = eval_spec.get("sub_type")
            except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS:
                sub_type = None

            if eval_type == "model_graded" and sub_type == "rag_pipeline":
                progress = {
                    "total_samples": total_samples,
                    "completed_samples": 0,
                    "failed_samples": 0,
                    "current_batch": 0
                }
                self.db.update_run_progress(run_id, progress)

                # Execute rag_pipeline evaluation end-to-end
                results, usage = await self._execute_rag_pipeline_run(
                    run_id=run_id,
                    samples=samples,
                    eval_spec=eval_spec,
                    eval_config=eval_config,
                    run_user_id=evaluation.get("created_by"),
                )

                # Store results and webhook
                self.db.store_run_results(run_id, results, usage)
                webhook_url = eval_config.get("webhook_url")
                if webhook_url:
                    await self._send_webhook(webhook_url, run_id, eval_id, "completed", results)

                logger.info(f"Evaluation run {run_id} (rag_pipeline) completed in {time.time() - start_time:.2f}s")
                return results

            # Get evaluation function for standard types
            eval_fn = self._get_evaluation_function(eval_type, eval_spec)

            # Process samples in batches
            run_config = eval_config.get("config") if isinstance(eval_config, dict) else None
            if hasattr(run_config, "model_dump"):
                run_config = run_config.model_dump()
            elif hasattr(run_config, "dict"):
                run_config = run_config.dict()
            if not isinstance(run_config, dict):
                run_config = {}

            batch_size = run_config.get("batch_size", 10)
            max_workers = run_config.get("max_workers", 4)
            timeout_seconds = run_config.get("timeout_seconds", self.eval_timeout)
            try:
                batch_size = int(batch_size)
            except (TypeError, ValueError):
                batch_size = 10
            if batch_size < 1:
                batch_size = 1
            try:
                max_workers = int(max_workers)
            except (TypeError, ValueError):
                max_workers = 4
            if max_workers < 1:
                max_workers = 1
            try:
                timeout_seconds = float(timeout_seconds)
            except (TypeError, ValueError):
                timeout_seconds = float(self.eval_timeout)

            sample_results = []

            for i in range(0, total_samples, batch_size):
                batch = samples[i:i + batch_size]
                progress["current_batch"] = i // batch_size + 1

                # Process batch
                batch_results = await self._process_batch(
                    batch,
                    eval_fn,
                    eval_spec,
                    eval_config,
                    max_workers,
                    start_index=i,
                    timeout_seconds=timeout_seconds,
                )

                # Update progress
                for result in batch_results:
                    if result.get("error"):
                        progress["failed_samples"] += 1
                    else:
                        progress["completed_samples"] += 1

                    sample_results.append(result)

                self.db.update_run_progress(run_id, progress)

            # Calculate aggregate results (include failed samples)
            eval_metrics = self._normalize_metrics(eval_spec.get("metrics"), default=[])
            aggregate = self._calculate_aggregate_results(
                sample_results,
                eval_metrics,
                eval_spec.get("threshold", 0.7)
            )

            # Calculate token usage
            usage = self._calculate_usage(sample_results)

            # Prepare final results
            duration = time.time() - start_time
            results = {
                "aggregate": aggregate,
                "by_metric": self._calculate_metric_stats(sample_results, eval_metrics),
                "sample_results": sample_results,
                "failed_samples": [r for r in sample_results if r.get("error")]
            }

            # Store results
            self.db.store_run_results(run_id, results, usage)

            # Send webhook if configured
            webhook_url = eval_config.get("webhook_url")
            if webhook_url:
                await self._send_webhook(webhook_url, run_id, eval_id, "completed", results)

            logger.info(f"Evaluation run {run_id} completed in {duration:.2f}s")
            return results

        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Evaluation run {run_id} failed: {e}")
            self.db.update_run_status(run_id, "failed", error_message=str(e))

            # Send failure webhook
            webhook_url = eval_config.get("webhook_url")
            if webhook_url:
                await self._send_webhook(webhook_url, run_id, eval_id, "failed", {"error": str(e)})

            raise
        finally:
            # Clean up running task
            if run_id in self.running_tasks:
                del self.running_tasks[run_id]

    # ============= RAG Pipeline Orchestration =============

    def _expand_values(self, value):
        """Normalize a value or list of values to a list."""
        if value is None:
            return [None]
        if isinstance(value, list):
            return value
        return [value]

    def _cartesian_product(self, dicts_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Cartesian product of parameter dictionaries.

        Each dict may contain fields that are single values or lists.
        Returns a list of flat dicts for each combination.
        """
        if not dicts_list:
            return [{}]
        from itertools import product
        # Merge multiple dicts into a single dict of key->list-of-values
        merged: dict[str, list[Any]] = {}
        for d in dicts_list:
            for k, v in (d or {}).items():
                vals = self._expand_values(v)
                merged.setdefault(k, [])
                # If multiple sweep blocks set the same key, append and deduplicate later
                merged[k].extend(vals)
        # Deduplicate per key preserving order
        for k, vals in merged.items():
            seen = set()
            deduped = []
            for x in vals:
                key = str(x)
                if key not in seen:
                    seen.add(key)
                    deduped.append(x)
            merged[k] = deduped
        # Build product
        keys = list(merged.keys())
        values_lists = [merged[k] for k in keys]
        combos = []
        for combo in product(*values_lists):
            combos.append(dict(zip(keys, combo)))
        return combos

    def _build_config_grid(self, eval_spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Build configuration combinations from rag_pipeline spec."""
        rp = (eval_spec or {}).get("rag_pipeline", {}) or {}
        chunking = rp.get("chunking") or {}
        retrievers = rp.get("retrievers") or [{}]
        rerankers = rp.get("rerankers") or [{}]
        rag = rp.get("rag") or {}

        # Normalize blocks to list of dicts
        if isinstance(retrievers, dict):
            retrievers = [retrievers]
        if isinstance(rerankers, dict):
            rerankers = [rerankers]

        # Expand each block
        chunking_combos = self._cartesian_product([chunking]) if chunking else [{}]
        retriever_combos = []
        for r in retrievers:
            retriever_combos.extend(self._cartesian_product([r]))
        reranker_combos = []
        for rr in rerankers:
            reranker_combos.extend(self._cartesian_product([rr]))
        rag_combos = self._cartesian_product([rag]) if rag else [{}]

        # Cartesian across blocks
        from itertools import product
        grid = []
        for ck, rt, rr, rg in product(chunking_combos, retriever_combos, reranker_combos, rag_combos):
            cfg = {
                "chunking": ck,
                "retriever": rt,
                "reranker": rr,
                "rag": rg,
            }
            grid.append(cfg)

        # Apply search strategy/max_trials
        strategy = rp.get("search_strategy", "grid")
        max_trials = rp.get("max_trials")
        if max_trials and len(grid) > max_trials:
            if strategy == "random":
                import random
                # Deterministic sampling keeps experimental comparisons reproducible.
                random.seed(42)  # nosec B311
                grid = random.sample(grid, max_trials)  # nosec B311
            else:
                grid = grid[:max_trials]
        return grid

    async def _execute_rag_pipeline_run(
        self,
        run_id: str,
        samples: list[dict[str, Any]],
        eval_spec: dict[str, Any],
        eval_config: dict[str, Any],
        run_user_id: Optional[str] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run rag_pipeline across a config grid and dataset, aggregating a leaderboard.

        Returns (results_dict, usage_dict)
        """
        rp = (eval_spec or {}).get("rag_pipeline", {}) or {}
        rp.get("metrics") or {}
        config_grid = self._build_config_grid(eval_spec)
        eval_provider, eval_model, _ = self._resolve_eval_provider_model_api_key(eval_spec, eval_config)

        # Progress accounting
        total_work = max(1, len(config_grid) * max(1, len(samples)))
        completed = 0
        failed = 0

        per_config_results = []
        best = None
        leaderboard = []
        built_collections: list[str] = []

        # Metrics helpers (lazy-load only if enabled)
        enable_custom_metrics = True
        try:
            enable_custom_metrics = bool(rp.get("custom_metrics", True))
        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS:
            enable_custom_metrics = True

        custom_metrics = None
        if enable_custom_metrics:
            try:
                custom_metrics = get_custom_metrics()
            except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Custom metrics initialization skipped: {e}")
                custom_metrics = None

        # Resolve custom metrics timeout (seconds)
        metrics_timeout: float = 2.0
        try:
            if rp.get("custom_metrics_timeout_seconds") is not None:
                metrics_timeout = float(rp.get("custom_metrics_timeout_seconds"))
            else:
                env_val = os.getenv("TLDW_CUSTOM_METRICS_TIMEOUT_SECONDS")
                if env_val:
                    metrics_timeout = float(env_val)
        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS:
            metrics_timeout = 2.0

        # Resolve the user scope used by vector-store operations.
        resolved_user_id = self._resolve_rag_pipeline_user_id(run_user_id, eval_config)

        # Optional ephemeral indexing base namespace
        base_namespace = rp.get("index_namespace")

        # Initialize progress for total work units
        with suppress(_EVAL_RUNNER_NONCRITICAL_EXCEPTIONS):
            self.db.update_run_progress(
                run_id,
                {
                    "total_samples": total_work,
                    "completed_samples": 0,
                    "failed_samples": 0,
                },
            )

        # Iterate configs
        for idx, cfg in enumerate(config_grid):
            cfg_id = f"cfg_{idx+1:03d}"
            per_sample = []
            # Aggregate accumulators
            agg_scores = []
            agg_latency = []

            # Unpack blocks
            ck = cfg.get("chunking", {}) or {}
            rt = cfg.get("retriever", {}) or {}
            rr = cfg.get("reranker", {}) or {}
            rg = cfg.get("rag", {}) or {}

            # Map to unified_rag_pipeline args
            upr_args_common = {
                # Retrieval
                "search_mode": rt.get("search_mode") or "hybrid",
                "hybrid_alpha": rt.get("hybrid_alpha") if rt.get("hybrid_alpha") is not None else 0.7,
                "top_k": rt.get("top_k") or 10,
                "min_score": rt.get("min_score") or 0.0,
                "keyword_filter": rt.get("keyword_filter"),
                # Reranking
                "enable_reranking": True,
                "reranking_strategy": rr.get("strategy") or "flashrank",
                "rerank_top_k": rr.get("top_k") or rt.get("top_k") or 10,
                # Generation
                "enable_generation": True,
                "generation_model": (rg.get("model") or [None])[0] if isinstance(rg.get("model"), list) else rg.get("model"),
                "generation_prompt": (rg.get("prompt_template") or [None])[0] if isinstance(rg.get("prompt_template"), list) else rg.get("prompt_template"),
                "max_generation_tokens": (rg.get("max_tokens") or [None])[0] if isinstance(rg.get("max_tokens"), list) else (rg.get("max_tokens") or 500),
                "user_id": resolved_user_id,
            }

            # Additional retrieval knobs (safe pass-through)
            if rt.get("fts_level") in ("media", "chunk"):
                upr_args_common["fts_level"] = rt.get("fts_level")

            # Advanced pass-through from an optional 'advanced' block
            adv = rp.get("advanced") or {}
            if isinstance(adv, dict):
                allowed_adv_keys = {
                    # Search/expansion/cache
                    "expand_query","expansion_strategies","spell_check","enable_cache","cache_threshold","adaptive_cache",
                    # Security/filters
                    "enable_security_filter","detect_pii","redact_pii","sensitivity_level","content_filter",
                    # Tables/VLM
                    "enable_table_processing","table_method","enable_vlm_late_chunking","vlm_backend","vlm_detect_tables_only","vlm_max_pages","vlm_late_chunk_top_k_docs",
                    # Context/chunking
                    "enable_enhanced_chunking","chunk_type_filter","enable_parent_expansion","parent_context_size","include_parent_document","sibling_window","include_sibling_chunks",
                    # Advanced retrieval
                    "enable_multi_vector_passages","mv_span_chars","mv_stride","mv_max_spans","mv_flatten_to_spans","enable_numeric_table_boost",
                    # Reranking extras
                    "reranking_model","rerank_min_relevance_prob","rerank_sentinel_margin",
                    # Citations/generation guardrails
                    "enable_citations","citation_style","include_page_numbers","enable_chunk_citations","strict_extractive","require_hard_citations","enable_numeric_fidelity","numeric_fidelity_behavior",
                    # Generation extras
                    "enable_abstention","abstention_behavior","enable_multi_turn_synthesis","synthesis_time_budget_sec","synthesis_draft_tokens","synthesis_refine_tokens",
                    # Post-verification/adaptive
                    "enable_post_verification","adaptive_max_retries","adaptive_unsupported_threshold","adaptive_max_claims","adaptive_time_budget_sec","low_confidence_behavior","adaptive_advanced_rewrites","adaptive_rerun_on_low_confidence","adaptive_rerun_include_generation","adaptive_rerun_bypass_cache","adaptive_rerun_time_budget_sec","adaptive_rerun_doc_budget",
                    # Observability/perf
                    "enable_monitoring","enable_observability","trace_id","enable_performance_analysis","timeout_seconds",
                    # Namespace and UX
                    "index_namespace","highlight_results","highlight_query_terms","track_cost","debug_mode",
                    # Claims/factuality
                    "enable_claims","claim_extractor","claim_verifier","claims_top_k","claims_conf_threshold","claims_max","nli_model","claims_concurrency",
                    # Filtering/date/media type
                    "enable_date_filter","date_range","filter_media_types",
                }
                for k, v in adv.items():
                    if k in allowed_adv_keys:
                        upr_args_common[k] = v

            # Enhanced chunking toggles (note: retrieval-time placeholder in unified pipeline)
            if ck:
                upr_args_common.update({
                    "enable_enhanced_chunking": True,
                    "include_sibling_chunks": (ck.get("include_siblings") or [False])[0] if isinstance(ck.get("include_siblings"), list) else ck.get("include_siblings") or False,
                })

            # Ephemeral indexing: build collection if dataset provides corpus and index_namespace is set
            collection_name = None
            chunk_index_stats = None
            try:
                if base_namespace:
                    collection_name = f"{base_namespace}_{cfg_id}"
                    chunk_index_stats = await self._build_ephemeral_index(
                        collection_name=collection_name,
                        samples=samples,
                        chunking_cfg=ck,
                        user_id=resolved_user_id,
                    )
                    upr_args_common["index_namespace"] = collection_name
                    built_collections.append(collection_name)
                    # Register ephemeral collection with TTL
                    try:
                        ttl = int(rp.get("ephemeral_ttl_seconds") or 86400)
                        self.db.register_ephemeral_collection(collection_name, ttl_seconds=ttl, run_id=run_id, namespace=base_namespace)
                    except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as re:
                        logger.warning(f"Failed to register ephemeral collection {collection_name}: {re}")
            except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Ephemeral indexing skipped for {cfg_id}: {e}")

            # Evaluate each sample
            for s_idx, sample in enumerate(samples):
                # Extract query and ground truth
                inp = sample.get("input") or {}
                exp = sample.get("expected") or {}
                query = inp.get("question") or inp.get("query") or inp.get("prompt") or inp.get("text") or str(inp)
                ground_truth = exp.get("answer") if isinstance(exp, dict) else (exp if isinstance(exp, str) else None)

                try:
                    # Call unified RAG pipeline
                    upr_result = await unified_rag_pipeline(
                        query=query,
                        **upr_args_common
                    )
                except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
                    logger.error(f"unified_rag_pipeline failed for {cfg_id} sample {s_idx}: {e}")
                    per_sample.append({"sample_index": s_idx, "error": str(e)})
                    completed += 1
                    failed += 1
                    self.db.update_run_progress(
                        run_id,
                        {
                            "completed_samples": completed,
                            "total_samples": total_work,
                            "failed_samples": failed,
                        },
                    )
                    continue

                # Extract contexts and response from pipeline result
                documents = []
                generated_answer = None
                timings = {}
                try:
                    if isinstance(upr_result, dict):
                        documents = upr_result.get("documents", [])
                        generated_answer = upr_result.get("generated_answer")
                        timings = upr_result.get("timings", {}) or {}
                    else:
                        # Dataclass UnifiedSearchResult
                        documents = getattr(upr_result, "documents", [])
                        generated_answer = getattr(upr_result, "generated_answer", None)
                        timings = getattr(upr_result, "timings", {}) or {}
                except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS:
                    pass

                # Convert documents to string contexts
                ctx_texts: list[str] = []
                try:
                    for d in (documents or [])[: upr_args_common.get("top_k", 10)]:
                        # Document may be dataclass with .content or dict
                        content = getattr(d, "content", None)
                        if content is None and isinstance(d, dict):
                            content = d.get("content")
                        if isinstance(content, str):
                            ctx_texts.append(content)
                except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS:
                    pass

                # Fallback: ensure we have a response
                response = generated_answer or ""

                # Compute RAG evaluation metrics (faithfulness, relevance, answer_similarity)
                try:
                    rag_metrics = await self.rag_evaluator.evaluate(
                        query=query,
                        contexts=ctx_texts,
                        response=response,
                        ground_truth=ground_truth,
                        metrics=self._normalize_metrics(
                            eval_spec.get("metrics"),
                            default=["relevance", "faithfulness", "answer_similarity"],
                        ),
                        api_name=eval_provider,
                        model=eval_model,
                    )
                    # Extract normalized metric scores
                    scores = {}
                    def _extract_score(value: Any) -> Optional[float]:
                        raw_val = value
                        if isinstance(value, dict):
                            raw_val = value.get("score")
                            if raw_val is None:
                                raw_val = value.get("raw_score")
                        else:
                            raw_val = getattr(value, "score", value)
                        try:
                            return float(raw_val)
                        except (TypeError, ValueError):
                            return None

                    for mname, mdata in rag_metrics.get("metrics", {}).items():
                        score_val = _extract_score(mdata)
                        if score_val is not None:
                            scores[mname] = score_val
                    overall = float(rag_metrics.get("overall_score", 0.0))
                except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
                    logger.error(f"RAG metrics failed for {cfg_id} sample {s_idx}: {e}")
                    scores = {}
                    overall = 0.0

                # Optional: retrieval diversity/coverage (gated + timeboxed)
                if enable_custom_metrics and custom_metrics is not None:
                    try:
                        cov = await asyncio.wait_for(
                            custom_metrics.evaluate_retrieval_coverage(query, ctx_texts),
                            timeout=metrics_timeout,
                        )
                    except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
                        cov = None
                        logger.debug(f"Coverage metric skipped: {e}")
                    try:
                        div = await asyncio.wait_for(
                            custom_metrics.evaluate_retrieval_diversity(ctx_texts),
                            timeout=metrics_timeout,
                        )
                    except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
                        div = None
                        logger.debug(f"Diversity metric skipped: {e}")

                    try:
                        if cov is not None:
                            cov_score = _extract_score(cov)
                            if cov_score is not None:
                                scores["retrieval_coverage"] = cov_score
                        if div is not None:
                            div_score = _extract_score(div)
                            if div_score is not None:
                                scores["retrieval_diversity"] = div_score
                    except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS:
                        pass

                # Optional: retrieval nDCG/MRR when relevant IDs provided
                try:
                    expected_rel = None
                    if isinstance(exp, dict):
                        expected_rel = exp.get("relevant_ids") or exp.get("relevant_doc_ids")
                    if expected_rel and isinstance(expected_rel, list):
                        retrieved_ids = []
                        for d in (documents or []):
                            did = getattr(d, "id", None) if not isinstance(d, dict) else d.get("id")
                            if did is not None:
                                retrieved_ids.append(str(did))
                        mrr, ndcg = self._compute_mrr_ndcg(retrieved_ids, [str(x) for x in expected_rel])
                        scores["mrr"] = mrr
                        scores["ndcg"] = ndcg
                except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"MRR/nDCG computation skipped: {e}")

                # Aggregate per-sample record
                latency = 0.0
                with suppress(_EVAL_RUNNER_NONCRITICAL_EXCEPTIONS):
                    latency = float(timings.get("total", 0.0)) * 1000.0
                rec = {
                    "sample_index": s_idx,
                    "scores": scores,
                    "overall": overall,
                    "latency_ms": latency,
                }
                if chunk_index_stats:
                    rec["chunk_index_stats"] = chunk_index_stats
                per_sample.append(rec)
                if overall is not None:
                    agg_scores.append(overall)
                if latency:
                    agg_latency.append(latency)

                completed += 1
                # Lightweight progress update
                self.db.update_run_progress(
                    run_id,
                    {
                        "completed_samples": completed,
                        "total_samples": total_work,
                        "failed_samples": failed,
                    },
                )

            # Aggregate per-config
            mean_overall = statistics.mean(agg_scores) if agg_scores else 0.0
            mean_latency = statistics.mean(agg_latency) if agg_latency else 0.0
            # Compute aggregated retrieval stats across samples if present
            def _mean_score(key: str, _per_sample=per_sample) -> float:
                vals = []
                for r in _per_sample:
                    v = r.get("scores", {}).get(key)
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                return statistics.mean(vals) if vals else 0.0

            retrieval_cov_mean = _mean_score("retrieval_coverage")
            retrieval_div_mean = _mean_score("retrieval_diversity")
            mrr_mean = _mean_score("mrr")
            ndcg_mean = _mean_score("ndcg")

            # Include chunk stats if available
            chunk_stats = None
            for r in per_sample:
                if r.get("chunk_index_stats"):
                    chunk_stats = r["chunk_index_stats"]
                    break
            cohesion_mean = (chunk_stats or {}).get("cohesion_mean", 0.0)
            separation_mean = (chunk_stats or {}).get("separation_mean", 0.0)

            # Aggregation weights for leaderboard score
            weights = rp.get("aggregation_weights") or {"rag_overall": 1.0}
            config_score = (
                (weights.get("rag_overall", 1.0) * mean_overall)
                + (weights.get("retrieval_diversity", 0.0) * retrieval_div_mean)
                + (weights.get("retrieval_coverage", 0.0) * retrieval_cov_mean)
                + (weights.get("chunk_cohesion", 0.0) * cohesion_mean)
                + (weights.get("chunk_separation", 0.0) * separation_mean)
                + (weights.get("mrr", 0.0) * mrr_mean)
                + (weights.get("ndcg", 0.0) * ndcg_mean)
            )

            config_summary = {
                "config_id": cfg_id,
                "config": cfg,
                "aggregate": {
                    "overall": mean_overall,
                    "latency_ms": mean_latency,
                    "retrieval_coverage": retrieval_cov_mean,
                    "retrieval_diversity": retrieval_div_mean,
                    "mrr": mrr_mean,
                    "ndcg": ndcg_mean,
                    "chunk_cohesion": cohesion_mean,
                    "chunk_separation": separation_mean,
                    "config_score": config_score,
                },
                "per_sample": per_sample,
            }
            per_config_results.append(config_summary)
            leaderboard.append({
                "config_id": cfg_id,
                "overall": mean_overall,
                "latency_ms": mean_latency,
                "config_score": config_score,
                "config": cfg,
            })

            # Track best (by config_score then overall)
            if best is None:
                best = config_summary
            else:
                prev_score = best["aggregate"].get("config_score", best["aggregate"].get("overall", 0.0))
                curr_score = config_summary["aggregate"].get("config_score", mean_overall)
                if curr_score > prev_score:
                    best = config_summary

        # Sort leaderboard
        leaderboard.sort(key=lambda x: (-x.get("config_score", x.get("overall", 0.0)), x["latency_ms"]))

        # Build final results structure
        results = {
            "leaderboard": leaderboard,
            "by_config": per_config_results,
            "best_config": best,
            "config_count": len(config_grid),
            "sample_count": len(samples),
            "notes": [
                "Chunking sweeps currently apply at retrieval-time only; indexing-time chunking comparison is planned."
            ]
        }

        usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

        # Optional cleanup of ephemeral collections
        try:
            if rp.get("cleanup_collections") and built_collections:
                from tldw_Server_API.app.core.config import settings as app_settings
                adapter = create_from_settings_for_user(app_settings, resolved_user_id)
                await adapter.initialize()
                for cname in built_collections:
                    try:
                        await adapter.delete_collection(cname)
                    except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as ce:
                        logger.warning(f"Failed to delete collection {cname}: {ce}")
        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Ephemeral collection cleanup skipped: {e}")

        return results, usage

    def _compute_mrr_ndcg(self, retrieved_ids: list[str], relevant_ids: list[str]) -> tuple[float, float]:
        """Compute MRR and nDCG@K for binary relevance.

        Args:
            retrieved_ids: Ranked list of retrieved document IDs
            relevant_ids: Set/list of relevant IDs
        Returns:
            (mrr, ndcg)
        """
        rel_set = set(relevant_ids)
        # MRR
        mrr = 0.0
        for i, rid in enumerate(retrieved_ids, start=1):
            if rid in rel_set:
                mrr = 1.0 / i
                break
        # nDCG
        import math
        gains = [1.0 if rid in rel_set else 0.0 for rid in retrieved_ids]
        dcg = 0.0
        for i, g in enumerate(gains, start=1):
            if g:
                dcg += 1.0 / math.log2(i + 1)
        # Ideal DCG with same number of relevant items
        R = min(len(rel_set), len(retrieved_ids))
        idcg = sum([1.0 / math.log2(i + 1) for i in range(1, R + 1)]) if R > 0 else 0.0
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        return mrr, ndcg

    async def _build_ephemeral_index(
        self,
        collection_name: str,
        samples: list[dict[str, Any]],
        chunking_cfg: dict[str, Any],
        user_id: str,
    ) -> dict[str, Any]:
        """Build an ephemeral index (collection) for a run-config if dataset provides a corpus.

        Supports dataset samples with input.corpus or input.documents as a list of strings or
        list of objects {id, text}.
        """
        # Gather corpus texts
        corpus: list[dict[str, str]] = []
        for s in samples:
            inp = s.get("input") or {}
            docs = inp.get("corpus") or inp.get("documents") or []
            if not isinstance(docs, list):
                continue
            for _i, d in enumerate(docs):
                if isinstance(d, str):
                    corpus.append({"id": f"doc_{len(corpus)+1}", "text": d})
                elif isinstance(d, dict):
                    tid = str(d.get("id") or f"doc_{len(corpus)+1}")
                    txt = d.get("text") or d.get("content")
                    if isinstance(txt, str) and txt.strip():
                        corpus.append({"id": tid, "text": txt})

        if not corpus:
            raise ValueError("No corpus provided in dataset input (expect input.corpus/documents)")

        # Create adapter
        from tldw_Server_API.app.core.config import settings as app_settings
        adapter = create_from_settings_for_user(app_settings, user_id)
        await adapter.initialize()
        await adapter.create_collection(collection_name)

        # Chunk and embed
        total_chunks = 0
        adj_sims = []
        sep_sims = []
        adj_ngram_jaccard = []
        entropy_vals = []
        type_counts = {"text": 0, "code": 0, "table": 0, "header": 0, "list": 0}
        from sklearn.metrics.pairwise import cosine_similarity

        for doc in corpus:
            chunks = chunk_for_embedding(doc["text"], file_name=doc["id"],
                                         method=(chunking_cfg.get("method") or ["sentences"])[0] if isinstance(chunking_cfg.get("method"), list) else (chunking_cfg.get("method") or "sentences"),
                                         max_size=(chunking_cfg.get("chunk_size") or [512])[0] if isinstance(chunking_cfg.get("chunk_size"), list) else (chunking_cfg.get("chunk_size") or 512),
                                         overlap=(chunking_cfg.get("overlap") or [64])[0] if isinstance(chunking_cfg.get("overlap"), list) else (chunking_cfg.get("overlap") or 64))
            if not chunks:
                continue
            ids = [f"{doc['id']}_ch_{i}" for i in range(len(chunks))]
            texts = [c["text"] for c in chunks]
            docs_text = [c.get("text_for_embedding") or c["text"] for c in chunks]
            metas = []
            for i, c in enumerate(chunks):
                m = c.get("metadata", {}).copy()
                m.update({"media_id": doc["id"], "chunk_index": i})
                metas.append(m)
            # Embeddings
            user_app_config = get_embedding_config()
            vectors = await asyncio.get_event_loop().run_in_executor(
                None,
                create_embeddings_batch,
                docs_text,
                user_app_config,
                None,
            )
            # Convert numpy arrays to lists if needed
            vecs = []
            for v in vectors:
                if hasattr(v, 'tolist'):
                    vecs.append(v.tolist())
                else:
                    vecs.append(v)
            # Upsert
            await adapter.upsert_vectors(collection_name, ids, vecs, texts, metas)

            # Cohesion/separation metrics using embeddings for this doc
            try:
                if len(vecs) >= 2:
                    # Adjacent similarities
                    for i in range(len(vecs) - 1):
                        s = cosine_similarity([vecs[i]], [vecs[i+1]])[0][0]
                        adj_sims.append(float(s))
                        # Adjacent n-gram Jaccard (bigrams)
                        def bigrams(t: str):
                            toks = [w for w in t.lower().split() if w]
                            return set(zip(toks, toks[1:])) if len(toks) > 1 else set()
                        b1 = bigrams(texts[i])
                        b2 = bigrams(texts[i+1])
                        if b1 or b2:
                            j = len(b1 & b2) / max(1, len(b1 | b2))
                            adj_ngram_jaccard.append(float(j))
                if len(vecs) >= 3:
                    # Non-adjacent separation: compare i and i+2
                    for i in range(len(vecs) - 2):
                        s = cosine_similarity([vecs[i]], [vecs[i+2]])[0][0]
                        sep_sims.append(float(s))
            except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS:
                pass

            # Token entropy per chunk
            import math
            for t in texts:
                toks = [w for w in t.lower().split() if w]
                if not toks:
                    continue
                from collections import Counter
                counts = Counter(toks)
                total = sum(counts.values())
                probs = [c/total for c in counts.values()]
                H = -sum(p*math.log(p+1e-12, 2) for p in probs)
                # Normalize by log2(V)
                V = max(1, len(counts))
                Hn = H / math.log2(V+1e-12)
                entropy_vals.append(float(Hn))

            # Heuristic type detection counts
            for t in texts:
                tt = t.strip()
                ctype = "text"
                if "```" in tt or any(sym in tt for sym in ['{', '}', ';']) and tt.count('\n') > 0:
                    ctype = "code"
                elif tt.startswith('#'):
                    ctype = "header"
                elif tt.startswith(('- ', '* ', '1.')):
                    ctype = "list"
                elif tt.count('|') >= 3 and tt.count('\n') >= 1:
                    ctype = "table"
                type_counts[ctype] = type_counts.get(ctype, 0) + 1
            total_chunks += len(vecs)

        # Summarize chunk stats
        import statistics
        # Ratios for types
        type_ratios = {}
        if total_chunks > 0:
            for k, v in type_counts.items():
                type_ratios[k] = v / total_chunks
        chunk_stats = {
            "total_chunks": total_chunks,
            "cohesion_mean": statistics.mean(adj_sims) if adj_sims else 0.0,
            "cohesion_count": len(adj_sims),
            "separation_mean": statistics.mean(sep_sims) if sep_sims else 0.0,
            "separation_count": len(sep_sims),
            "ngram_overlap_mean": statistics.mean(adj_ngram_jaccard) if adj_ngram_jaccard else 0.0,
            "token_entropy_mean": statistics.mean(entropy_vals) if entropy_vals else 0.0,
            "type_ratios": type_ratios,
        }
        return chunk_stats

    @staticmethod
    def _resolve_rag_pipeline_user_id(
        run_user_id: Optional[str],
        eval_config: Optional[dict[str, Any]] = None,
    ) -> str:
        """Resolve a stable user identifier for rag-pipeline vector-store calls."""
        candidates: list[Optional[str]] = [run_user_id]

        if isinstance(eval_config, dict):
            candidates.append(eval_config.get("created_by"))
            candidates.append(eval_config.get("user_id"))
            run_cfg = eval_config.get("config")
            if isinstance(run_cfg, dict):
                candidates.append(run_cfg.get("user_id"))

        for value in candidates:
            if value is None:
                continue
            normalized = str(value).strip()
            if normalized:
                return normalized

        try:
            from tldw_Server_API.app.core.config import settings as app_settings

            fallback = app_settings.get("SINGLE_USER_FIXED_ID", "1")
            return str(fallback)
        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS:
            return "1"

    async def _get_samples(
        self,
        evaluation: dict[str, Any],
        eval_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get samples for evaluation"""
        # Check for dataset override
        override = eval_config.get("dataset_override")
        if override:
            try:
                if hasattr(override, "model_dump"):
                    override = override.model_dump()
                elif hasattr(override, "dict"):
                    override = override.dict()
            except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS:
                pass
            if isinstance(override, dict) and "samples" in override:
                return override["samples"]
            if isinstance(override, list):
                return override
            if isinstance(override, dict):
                return [override]

        # Get from evaluation's dataset
        dataset_id = evaluation.get("dataset_id")
        if dataset_id:
            dataset = self.db.get_dataset(dataset_id)
            if dataset:
                return dataset["samples"]

        # Check if evaluation has inline dataset
        if evaluation.get("dataset"):
            return evaluation["dataset"]

        raise ValueError("No samples found for evaluation")

    def _get_evaluation_function(
        self,
        eval_type: str,
        eval_spec: dict[str, Any]
    ) -> Callable:
        """Get the appropriate evaluation function"""
        if eval_type == "model_graded":
            sub_type = eval_spec.get("sub_type")
            # Default to summarization if sub_type is None or empty
            if not sub_type:
                sub_type = "summarization"

            if sub_type == "summarization":
                return self._eval_summarization
            elif sub_type == "rag":
                return self._eval_rag
            elif sub_type == "response_quality":
                return self._eval_response_quality
            else:
                raise ValueError(f"Unknown model_graded sub_type: {sub_type}")

        elif eval_type == "exact_match":
            return self._eval_exact_match

        elif eval_type == "includes":
            return self._eval_includes

        elif eval_type == "fuzzy_match":
            return self._eval_fuzzy_match

        elif eval_type == "proposition_extraction":
            return self._eval_propositions

        elif eval_type == "label_choice":
            return self._eval_label_choice

        elif eval_type == "nli_factcheck":
            return self._eval_nli_factcheck

        else:
            raise ValueError(f"Unknown evaluation type: {eval_type}")

    async def _process_batch(
        self,
        batch: list[dict[str, Any]],
        eval_fn: Callable,
        eval_spec: dict[str, Any],
        eval_config: dict[str, Any],
        max_workers: int,
        start_index: int = 0,
        timeout_seconds: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Process a batch of samples with proper concurrency control"""
        run_timeout = timeout_seconds if timeout_seconds is not None else self.eval_timeout
        try:
            run_timeout = float(run_timeout)
        except (TypeError, ValueError):
            run_timeout = float(self.eval_timeout)
        if run_timeout <= 0:
            run_timeout = float(self.eval_timeout)

        try:
            max_workers = int(max_workers)
        except (TypeError, ValueError):
            max_workers = 1
        if max_workers < 1:
            max_workers = 1

        run_semaphore = asyncio.Semaphore(max_workers)

        async def eval_with_timeout_and_semaphore(sample, sample_id):
            """Evaluate a single sample with timeout and semaphore control"""
            async with self.semaphore, run_semaphore:
                try:
                    # Apply timeout to individual evaluation
                    result = await asyncio.wait_for(
                        eval_fn(
                            sample=sample,
                            eval_spec=eval_spec,
                            config=eval_config,
                            sample_id=sample_id
                        ),
                        timeout=run_timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    logger.error(f"Evaluation timeout for {sample_id}")
                    return {"sample_id": sample_id, "error": f"Timeout after {run_timeout}s"}
                except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
                    logger.error(f"Evaluation failed for {sample_id}: {e}")
                    return {"sample_id": sample_id, "error": str(e)}

        # Create tasks with proper error handling
        tasks = []
        sample_ids: list[str] = []
        for i, sample in enumerate(batch):
            global_index = start_index + i
            dataset_sample_id = sample.get("id") or sample.get("sample_id")
            sample_id = f"sample_{global_index:06d}"
            if dataset_sample_id:
                sample_id = f"{sample_id}_{dataset_sample_id}"
            sample_ids.append(sample_id)
            task = eval_with_timeout_and_semaphore(sample, sample_id)
            tasks.append(task)

        # Process all tasks concurrently (semaphore will limit actual concurrency)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any unexpected exceptions from gather
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "sample_id": sample_ids[i] if i < len(sample_ids) else f"sample_{start_index + i:06d}",
                    "error": str(result)
                })
            else:
                if isinstance(result, dict):
                    # Preserve original dataset sample id when available
                    ds_id = batch[i].get("id") or batch[i].get("sample_id")
                    if ds_id and "dataset_sample_id" not in result:
                        result["dataset_sample_id"] = ds_id
                processed_results.append(result)

        return processed_results

    # ============= Evaluation Functions =============

    async def _eval_summarization(
        self,
        sample: dict[str, Any],
        eval_spec: dict[str, Any],
        config: dict[str, Any],
        sample_id: str
    ) -> dict[str, Any]:
        """Evaluate summarization using G-Eval"""
        try:
            # Extract required fields
            source_text = sample["input"].get("source_text", "")
            summary = sample["input"].get("summary", "")

            # Resolve provider, model, and API key (if provided)
            api_name, model_override, api_key = self._resolve_eval_provider_model_api_key(eval_spec, config)

            # Run G-Eval with controlled thread pool usage
            loop = asyncio.get_event_loop()
            from functools import partial
            call_geval = partial(
                run_geval,
                transcript=source_text,
                summary=summary,
                api_key=api_key,
                api_name=api_name,
                model=model_override,
                save=False,
            )
            result = await loop.run_in_executor(None, call_geval)

            # Parse results
            scores: dict[str, float] = {}
            metrics_list = self._normalize_metrics(
                eval_spec.get("metrics"),
                default=["fluency", "consistency", "relevance", "coherence"],
            )

            if isinstance(result, dict):
                metric_blob = result.get("metrics") or result.get("scores") or {}
                for metric in metrics_list:
                    raw_val = metric_blob.get(metric)
                    if raw_val is None:
                        continue
                    if isinstance(raw_val, dict):
                        raw_val = raw_val.get("score") or raw_val.get("raw_score")
                    try:
                        raw = float(raw_val)
                    except (TypeError, ValueError):
                        continue
                    max_score = 3.0 if metric == "fluency" else 5.0
                    scores[metric] = raw / max_score if raw >= 1.0 else raw
            else:
                # Legacy string output fallback
                import re
                for metric in metrics_list:
                    pattern = f"{metric}.*?([0-9.]+)"
                    match = re.search(pattern, str(result), re.IGNORECASE)
                    if match:
                        max_score = 3.0 if metric == "fluency" else 5.0
                        scores[metric] = float(match.group(1)) / max_score  # Normalize to 0-1

            # Calculate pass/fail
            avg_score = statistics.mean(scores.values()) if scores else 0
            passed = self._evaluate_passed(scores, avg_score, eval_spec, default_threshold=0.7)

            return {
                "sample_id": sample_id,
                "scores": scores,
                "passed": passed,
                "avg_score": avg_score
            }

        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Summarization eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}

    async def _eval_rag(
        self,
        sample: dict[str, Any],
        eval_spec: dict[str, Any],
        config: dict[str, Any],
        sample_id: str
    ) -> dict[str, Any]:
        """Evaluate RAG system"""
        try:
            # Extract fields
            query = sample["input"].get("query", "")
            contexts = sample["input"].get("contexts", [])
            response = sample["input"].get("response", "")
            expected = sample.get("expected")
            if isinstance(expected, dict):
                ground_truth = expected.get("answer", "")
            elif isinstance(expected, str):
                ground_truth = expected
            else:
                ground_truth = ""

            # Run RAG evaluation
            api_name, model_override, _ = self._resolve_eval_provider_model_api_key(eval_spec, config)
            result = await self.rag_evaluator.evaluate(
                query=query,
                contexts=contexts,
                response=response,
                ground_truth=ground_truth,
                metrics=self._normalize_metrics(
                    eval_spec.get("metrics"),
                    default=["relevance", "faithfulness"],
                ),
                api_name=api_name,
                model=model_override,
            )

            # Extract scores
            scores = {}
            for metric_name, metric_data in result.get("metrics", {}).items():
                score_val = metric_data
                if isinstance(metric_data, dict):
                    score_val = metric_data.get("score", metric_data.get("raw_score"))
                else:
                    score_val = getattr(metric_data, "score", metric_data)
                try:
                    scores[metric_name] = float(score_val)
                except (TypeError, ValueError):
                    scores[metric_name] = 0.0

            # Calculate pass/fail
            avg_score = statistics.mean(scores.values()) if scores else 0
            passed = self._evaluate_passed(scores, avg_score, eval_spec, default_threshold=0.7)

            return {
                "sample_id": sample_id,
                "scores": scores,
                "passed": passed,
                "avg_score": avg_score
            }

        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"RAG eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}

    async def _eval_response_quality(
        self,
        sample: dict[str, Any],
        eval_spec: dict[str, Any],
        config: dict[str, Any],
        sample_id: str
    ) -> dict[str, Any]:
        """Evaluate response quality"""
        try:
            # Extract fields
            prompt = sample["input"].get("prompt", "")
            response = sample["input"].get("response", "")
            expected_format = sample["input"].get("expected_format")

            # Run quality evaluation
            api_name, model_override, _ = self._resolve_eval_provider_model_api_key(eval_spec, config)
            result = await self.quality_evaluator.evaluate(
                prompt=prompt,
                response=response,
                expected_format=expected_format,
                custom_criteria=eval_spec.get("custom_criteria"),
                api_name=api_name,
                model=model_override,
            )

            # Extract scores
            scores = {}
            for metric_name, metric_data in result.get("metrics", {}).items():
                score_val = metric_data
                if isinstance(metric_data, dict):
                    score_val = metric_data.get("score", metric_data.get("raw_score"))
                else:
                    score_val = getattr(metric_data, "score", metric_data)
                try:
                    scores[metric_name] = float(score_val)
                except (TypeError, ValueError):
                    scores[metric_name] = 0.0

            # Add overall quality
            scores["overall_quality"] = result.get("overall_quality", 0)

            # Calculate pass/fail
            avg_score = scores.get("overall_quality", 0)
            passed = self._evaluate_passed(scores, avg_score, eval_spec, default_threshold=0.7)

            return {
                "sample_id": sample_id,
                "scores": scores,
                "passed": passed,
                "avg_score": avg_score,
                "format_compliance": result.get("format_compliance", True)
            }

        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Quality eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}

    async def _eval_exact_match(
        self,
        sample: dict[str, Any],
        eval_spec: dict[str, Any],
        config: dict[str, Any],
        sample_id: str
    ) -> dict[str, Any]:
        """Evaluate exact match"""
        try:
            output = sample["input"].get("output", "")
            expected_field = sample.get("expected")
            if isinstance(expected_field, dict):
                expected = (
                    expected_field.get("output")
                    or expected_field.get("answer")
                    or expected_field.get("expected")
                )
            else:
                expected = expected_field
            expected = expected if expected is not None else ""

            # Normalize strings
            output = str(output).strip().lower()
            expected = str(expected).strip().lower()

            passed = output == expected
            score = 1.0 if passed else 0.0

            return {
                "sample_id": sample_id,
                "scores": {"exact_match": score},
                "passed": passed,
                "avg_score": score
            }

        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Exact match eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}

    async def _eval_includes(
        self,
        sample: dict[str, Any],
        eval_spec: dict[str, Any],
        config: dict[str, Any],
        sample_id: str
    ) -> dict[str, Any]:
        """Evaluate if output includes expected content"""
        try:
            output = str(sample["input"].get("output", ""))
            expected_field = sample.get("expected")
            if isinstance(expected_field, dict):
                expected_items = expected_field.get("includes")
                if expected_items is None:
                    expected_items = expected_field.get("output") or expected_field.get("answer") or []
            else:
                expected_items = expected_field or []

            if isinstance(expected_items, str) or not isinstance(expected_items, list):
                expected_items = [expected_items]

            # Check each expected item
            found_count = 0
            for item in expected_items:
                if str(item).lower() in output.lower():
                    found_count += 1

            score = found_count / len(expected_items) if expected_items else 0
            threshold = self._resolve_metric_threshold(eval_spec, "includes", default=0.7)
            passed = score >= threshold

            return {
                "sample_id": sample_id,
                "scores": {"includes": score},
                "passed": passed,
                "avg_score": score,
                "found": found_count,
                "total": len(expected_items)
            }

        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Includes eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}

    async def _eval_fuzzy_match(
        self,
        sample: dict[str, Any],
        eval_spec: dict[str, Any],
        config: dict[str, Any],
        sample_id: str
    ) -> dict[str, Any]:
        """Evaluate fuzzy string matching"""
        try:
            from difflib import SequenceMatcher

            output = str(sample["input"].get("output", ""))
            expected_field = sample.get("expected")
            if isinstance(expected_field, dict):
                expected = (
                    expected_field.get("output")
                    or expected_field.get("answer")
                    or expected_field.get("expected")
                )
            else:
                expected = expected_field
            expected = str(expected or "")

            # Calculate similarity
            similarity = SequenceMatcher(None, output, expected).ratio()
            threshold = self._resolve_metric_threshold(eval_spec, "fuzzy_match", default=0.7)
            passed = similarity >= threshold

            return {
                "sample_id": sample_id,
                "scores": {"fuzzy_match": similarity},
                "passed": passed,
                "avg_score": similarity
            }

        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Fuzzy match eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}

    async def _eval_propositions(
        self,
        sample: dict[str, Any],
        eval_spec: dict[str, Any],
        config: dict[str, Any],
        sample_id: str
    ) -> dict[str, Any]:
        """Evaluate proposition extraction using precision/recall/F1 per sample."""
        try:
            extracted = sample.get("input", {}).get("extracted", []) or []
            reference = sample.get("input", {}).get("reference", []) or []
            method = eval_spec.get("method", "semantic")
            threshold = float(eval_spec.get("threshold", 0.7))
            f1_threshold = float(eval_spec.get("threshold_f1", threshold))

            result = eval_propositions(extracted=extracted, reference=reference, method=method, threshold=threshold)

            scores = {
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "claim_density_per_100_tokens": result.claim_density_per_100_tokens,
                "avg_prop_len_tokens": result.avg_prop_len_tokens,
                "dedup_rate": result.dedup_rate,
            }

            passed = result.f1 >= f1_threshold
            avg_score = result.f1

            return {
                "sample_id": sample_id,
                "scores": scores,
                "passed": passed,
                "avg_score": avg_score,
                "counts": {
                    "matched": result.matched,
                    "total_extracted": result.total_extracted,
                    "total_reference": result.total_reference,
                }
            }
        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Proposition eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}

    async def _eval_label_choice(
        self,
        sample: dict[str, Any],
        eval_spec: dict[str, Any],
        config: dict[str, Any],
        sample_id: str
    ) -> dict[str, Any]:
        """Evaluate single-label classification over a fixed set of allowed labels.

        Expected sample structure:
          sample = {
            "input": {
              "question"|"prompt": str,
              "context": Optional[str],
              "allowed_labels"|"choices": Optional[List[str]],
              "prediction": Optional[str]  # if generate_predictions is False
            },
            "expected": {"label": str} or a plain string label
          }

        eval_spec may include: allowed_labels, label_mapping, structured_output, generate_predictions,
        api_name, temperature, prompt_template.
        """
        try:
            # Resolve fields
            inp = sample.get("input", {}) or {}
            expected_field = sample.get("expected")
            if isinstance(expected_field, dict):
                gold = expected_field.get("label") or expected_field.get("answer") or expected_field.get("expected")
            else:
                gold = expected_field

            question = inp.get("question") or inp.get("prompt") or ""
            context = inp.get("context") or None

            # Allowed labels
            allowed = (
                inp.get("allowed_labels")
                or inp.get("choices")
                or eval_spec.get("allowed_labels")
                or []
            )
            allowed = [str(x).strip() for x in allowed if x is not None]
            if not allowed:
                raise ValueError("label_choice requires 'allowed_labels' (in sample or eval_spec)")

            # Normalization
            mapping = eval_spec.get("label_mapping") or {}
            canon = {k.strip().upper(): v.strip().upper() for k, v in mapping.items()} if mapping else {}
            allowed_up = [a.upper() for a in allowed]

            def norm_label(x: Optional[str]) -> Optional[str]:
                if x is None:
                    return None
                s = str(x).strip()
                up = s.upper()
                if up in canon:
                    up = canon[up]
                # If the canonical is not in allowed, leave as-is; otherwise ensure allowed variant
                return up

            # Try to get prediction without generation if provided
            pred = inp.get("prediction")

            # Generate prediction if needed
            if pred is None and eval_spec.get("generate_predictions", True):
                api_name, model_override, api_key = self._resolve_eval_provider_model_api_key(eval_spec, config)
                temperature = float(eval_spec.get("temperature", 0.0))
                structured = bool(eval_spec.get("structured_output", False))
                allowed_str = ", ".join(allowed)

                # Build messages and system prompt
                if eval_spec.get("prompt_template"):
                    user_prompt = eval_spec["prompt_template"].format(
                        question=question,
                        context=context or "",
                        allowed_labels=allowed_str
                    )
                    system_prompt = None
                else:
                    if structured:
                        system_prompt = "You return strict JSON only with no commentary."
                        user_prompt = (
                            f"Allowed labels: [{allowed_str}]\n"
                            + (f"Context:\n{context}\n\n" if context else "")
                            + f"Question:\n{question}\n\n"
                            + "Respond with exactly: {\"label\": \"<one of the allowed labels>\"}"
                        )
                    else:
                        system_prompt = "You must reply with exactly one label token, nothing else."
                        user_prompt = (
                            f"Allowed labels: {allowed_str}. Respond with only one of these labels.\n\n"
                            + (f"Context (optional):\n{context}\n\n" if context else "")
                            + f"Question:\n{question}\n\nAnswer (one token):"
                        )

                messages = [{"role": "user", "content": user_prompt}]
                response_format = {"type": "json_object"} if structured else None

                try:
                    resp = await asyncio.to_thread(
                        _call_adapter_text,
                        api_endpoint=api_name,
                        messages_payload=messages,
                        temperature=temperature,
                        system_message=system_prompt,
                        response_format=response_format,
                        max_tokens=16,
                        api_key=api_key,
                        model=model_override,
                    )
                except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as ce:
                    logger.error(f"Chat call failed for label_choice {sample_id}: {ce}")
                    resp = ""

                # Extract text content from provider response
                def _extract_content(r: Any) -> str:
                    if isinstance(r, str):
                        return r
                    if isinstance(r, dict):
                        try:
                            choices = r.get("choices")
                            if choices:
                                msg = choices[0].get("message", {})
                                content = msg.get("content")
                                if isinstance(content, list):
                                    return "".join(
                                        part.get("text", "") if isinstance(part, dict) else str(part)
                                        for part in content
                                    )
                                return content if isinstance(content, str) else str(content)
                            if "text" in r:
                                return str(r["text"])  # some providers
                        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS:
                            pass
                    return str(r)

                pred = _extract_content(resp)

            # Parse prediction
            parsed: Optional[str] = None
            if isinstance(pred, (dict, list)):
                if isinstance(pred, dict):
                    parsed = pred.get("label")
            else:
                txt = str(pred).strip()
                if eval_spec.get("structured_output", False):
                    import json as _json
                    try:
                        obj = _json.loads(txt)
                        if isinstance(obj, dict) and "label" in obj:
                            parsed = obj["label"]
                    except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS:
                        parsed = None
                if parsed is None:
                    up = txt.upper()
                    for lab in allowed_up:
                        if lab in up or up.strip() == lab:
                            parsed = lab
                            break

            pred_norm = norm_label(parsed) if parsed is not None else None
            gold_norm = norm_label(gold)

            if pred_norm is None:
                # Unable to parse; count as incorrect
                return {
                    "sample_id": sample_id,
                    "scores": {"accuracy": 0.0},
                    "passed": False,
                    "avg_score": 0.0,
                    "details": {
                        "prediction_raw": pred,
                        "prediction": None,
                        "gold": gold_norm,
                        "allowed_labels": allowed,
                    }
                }

            correct = (gold_norm is not None) and (pred_norm == gold_norm)
            return {
                "sample_id": sample_id,
                "scores": {"accuracy": 1.0 if correct else 0.0},
                "passed": bool(correct),
                "avg_score": 1.0 if correct else 0.0,
                "details": {
                    "prediction": pred_norm,
                    "gold": gold_norm,
                    "allowed_labels": allowed,
                }
            }

        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"label_choice eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}

    async def _eval_nli_factcheck(
        self,
        sample: dict[str, Any],
        eval_spec: dict[str, Any],
        config: dict[str, Any],
        sample_id: str
    ) -> dict[str, Any]:
        """Evaluate factual claims by NLI-style labeling (SUPPORTED/REFUTED/NEI or ENTAILMENT/CONTRADICTION/NEUTRAL).

        Expected sample structure:
          sample = {
            "input": {
              "claim"|"hypothesis": str,
              "evidence"|"premise"|"context": Union[str, List[str]],
              "prediction": Optional[str]
            },
            "expected": {"label": str} or a plain string label
          }

        eval_spec: allowed_labels (default to tri-label), label_mapping, structured_output, generate_predictions,
        api_name/temperature/prompt_template optionally supported.
        """
        try:
            inp = sample.get("input", {}) or {}
            expected_field = sample.get("expected")
            if isinstance(expected_field, dict):
                gold = expected_field.get("label") or expected_field.get("answer") or expected_field.get("expected")
            else:
                gold = expected_field

            claim = inp.get("claim") or inp.get("hypothesis") or ""
            ev = inp.get("evidence") or inp.get("premise") or inp.get("context") or ""
            evidence = "\n".join([str(x) for x in ev]) if isinstance(ev, list) else str(ev)

            # Default allowed labels if not provided
            default_allowed = ["SUPPORTED", "REFUTED", "NEI"]
            allowed = (
                inp.get("allowed_labels")
                or eval_spec.get("allowed_labels")
                or default_allowed
            )
            allowed = [str(x).strip() for x in allowed if x is not None]
            allowed_up = [a.upper() for a in allowed]

            # Normalization map including common aliases
            base_alias = {
                "TRUE": "SUPPORTED",
                "ENTAILMENT": "SUPPORTED",
                "FALSE": "REFUTED",
                "CONTRADICTION": "REFUTED",
                "NEUTRAL": "NEI",
                "NOT_ENTAILED": "NEI",
                "NOT_ENTAILMENT": "NEI",
            }
            mapping = eval_spec.get("label_mapping") or {}
            canon = {**{k.upper(): v.upper() for k, v in base_alias.items()}, **{k.upper(): v.upper() for k, v in mapping.items()}}

            def norm_label(x: Optional[str]) -> Optional[str]:
                if x is None:
                    return None
                s = str(x).strip().upper()
                s = canon.get(s, s)
                return s

            pred = inp.get("prediction")

            if pred is None and eval_spec.get("generate_predictions", True):
                api_name, model_override, api_key = self._resolve_eval_provider_model_api_key(eval_spec, config)
                temperature = float(eval_spec.get("temperature", 0.0))
                structured = bool(eval_spec.get("structured_output", False))

                allowed_str = ", ".join(allowed)
                if eval_spec.get("prompt_template"):
                    user_prompt = eval_spec["prompt_template"].format(
                        claim=claim,
                        evidence=evidence,
                        allowed_labels=allowed_str
                    )
                    system_prompt = None
                else:
                    if structured:
                        system_prompt = "You return strict JSON only with no commentary."
                        user_prompt = (
                            f"Allowed labels: [{allowed_str}]\n"
                            f"Evidence:\n{evidence}\n\nClaim:\n{claim}\n\n"
                            "Respond with exactly: {\"label\": \"<one of the allowed labels>\"}"
                        )
                    else:
                        system_prompt = "You must reply with exactly one label token, nothing else."
                        user_prompt = (
                            f"Allowed labels: {allowed_str}. Respond with only one of these labels.\n\n"
                            f"Evidence:\n{evidence}\n\nClaim:\n{claim}\n\nAnswer (one token):"
                        )

                messages = [{"role": "user", "content": user_prompt}]
                response_format = {"type": "json_object"} if structured else None

                try:
                    resp = await asyncio.to_thread(
                        _call_adapter_text,
                        api_endpoint=api_name,
                        messages_payload=messages,
                        temperature=temperature,
                        system_message=system_prompt,
                        response_format=response_format,
                        max_tokens=16,
                        api_key=api_key,
                        model=model_override,
                    )
                except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as ce:
                    logger.error(f"Chat call failed for nli_factcheck {sample_id}: {ce}")
                    resp = ""

                def _extract_content(r: Any) -> str:
                    if isinstance(r, str):
                        return r
                    if isinstance(r, dict):
                        try:
                            choices = r.get("choices")
                            if choices:
                                msg = choices[0].get("message", {})
                                content = msg.get("content")
                                if isinstance(content, list):
                                    return "".join(
                                        part.get("text", "") if isinstance(part, dict) else str(part)
                                        for part in content
                                    )
                                return content if isinstance(content, str) else str(content)
                            if "text" in r:
                                return str(r["text"])  # other providers
                        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS:
                            pass
                    return str(r)

                pred = _extract_content(resp)

            # Parse
            parsed: Optional[str] = None
            if isinstance(pred, (dict, list)):
                if isinstance(pred, dict):
                    parsed = pred.get("label")
            else:
                txt = str(pred).strip()
                if eval_spec.get("structured_output", False):
                    import json as _json
                    try:
                        obj = _json.loads(txt)
                        if isinstance(obj, dict) and "label" in obj:
                            parsed = obj["label"]
                    except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS:
                        parsed = None
                if parsed is None:
                    up = txt.upper()
                    for lab in allowed_up:
                        if lab in up or up.strip() == lab:
                            parsed = lab
                            break

            pred_norm = norm_label(parsed) if parsed is not None else None
            gold_norm = norm_label(gold)

            if pred_norm is None:
                return {
                    "sample_id": sample_id,
                    "scores": {"accuracy": 0.0},
                    "passed": False,
                    "avg_score": 0.0,
                    "details": {
                        "prediction_raw": pred,
                        "prediction": None,
                        "gold": gold_norm,
                        "allowed_labels": allowed,
                    }
                }

            correct = (gold_norm is not None) and (pred_norm == gold_norm)
            return {
                "sample_id": sample_id,
                "scores": {"accuracy": 1.0 if correct else 0.0},
                "passed": bool(correct),
                "avg_score": 1.0 if correct else 0.0,
                "details": {
                    "prediction": pred_norm,
                    "gold": gold_norm,
                    "allowed_labels": allowed,
                }
            }

        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"nli_factcheck eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}

    # ============= Helper Methods =============

    @staticmethod
    def _coerce_threshold_value(raw: Any) -> Optional[float]:
        """Convert threshold values to float when possible."""
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _extract_threshold_config(
        self,
        eval_spec: dict[str, Any]
    ) -> tuple[Optional[float], dict[str, float]]:
        """Extract global and per-metric thresholds from eval_spec.

        Supports legacy `threshold` and new `thresholds` mappings.
        """
        global_threshold = self._coerce_threshold_value(eval_spec.get("threshold"))
        per_metric: dict[str, float] = {}

        thresholds = eval_spec.get("thresholds")
        if isinstance(thresholds, dict):
            if global_threshold is None:
                for key in ("pass", "overall", "default", "threshold"):
                    if key in thresholds:
                        global_threshold = self._coerce_threshold_value(thresholds.get(key))
                        if global_threshold is not None:
                            break
            for key, value in thresholds.items():
                if key in {"pass", "overall", "default", "threshold", "excellent"}:
                    continue
                metric_threshold = self._coerce_threshold_value(value)
                if metric_threshold is not None:
                    per_metric[key] = metric_threshold

        return global_threshold, per_metric

    def _resolve_metric_threshold(
        self,
        eval_spec: dict[str, Any],
        metric_name: Optional[str],
        *,
        default: float = 0.7
    ) -> float:
        """Resolve a threshold for a single metric."""
        global_threshold, per_metric = self._extract_threshold_config(eval_spec)
        if metric_name and metric_name in per_metric:
            return per_metric[metric_name]
        if global_threshold is not None:
            return global_threshold
        return default

    def _evaluate_passed(
        self,
        scores: dict[str, float],
        avg_score: float,
        eval_spec: dict[str, Any],
        *,
        default_threshold: float = 0.7
    ) -> bool:
        """Determine pass/fail from scores and eval_spec thresholds.

        If per-metric thresholds reference metrics missing from scores, log a warning
        and treat the per-metric check as failed.
        """
        global_threshold, per_metric = self._extract_threshold_config(eval_spec)
        if per_metric:
            missing_metrics = [metric for metric in per_metric if metric not in scores]
            if missing_metrics:
                logger.warning(
                    "Per-metric thresholds reference missing scores; failing per-metric check. "
                    f"missing={sorted(missing_metrics)} available={sorted(scores.keys())}"
                )
                per_metric_passed = False
            else:
                per_metric_passed = all(scores[metric] >= threshold for metric, threshold in per_metric.items())
            if global_threshold is not None:
                return per_metric_passed and avg_score >= global_threshold
            return per_metric_passed

        threshold = global_threshold if global_threshold is not None else default_threshold
        return avg_score >= threshold

    def _calculate_aggregate_results(
        self,
        results: list[dict[str, Any]],
        metrics: Optional[list[str]],
        threshold: float
    ) -> dict[str, Any]:
        """Calculate aggregate statistics"""
        _ = metrics
        _ = threshold
        empty_pass_rate = 0.0
        if not results:
            return {
                "mean_score": 0,
                "std_dev": 0,
                "min_score": 0,
                "max_score": 0,
                "pass_rate": empty_pass_rate,
                "total_samples": 0,
                "failed_samples": 0
            }

        # Get all scores
        all_scores = []
        passed_count = 0
        failed_samples = 0

        for result in results:
            score_val = result.get("avg_score")
            if score_val is None:
                failed_samples += 1
                score_val = 0.0
            try:
                score_val = float(score_val)
            except (TypeError, ValueError):
                failed_samples += 1
                score_val = 0.0
            all_scores.append(score_val)
            if result.get("passed", False):
                passed_count += 1

        if not all_scores:
            return {
                "mean_score": 0,
                "std_dev": 0,
                "min_score": 0,
                "max_score": 0,
                "pass_rate": empty_pass_rate,
                "total_samples": len(results),
                "failed_samples": len(results)
            }

        return {
            "mean_score": statistics.mean(all_scores),
            "std_dev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
            "min_score": min(all_scores),
            "max_score": max(all_scores),
            "pass_rate": passed_count / len(results),
            "total_samples": len(results),
            "failed_samples": failed_samples
        }

    def _calculate_metric_stats(
        self,
        results: list[dict[str, Any]],
        metrics: Optional[list[str]]
    ) -> dict[str, dict[str, float]]:
        """Calculate per-metric statistics"""
        metric_names = self._normalize_metrics(metrics, default=[])
        if not metric_names:
            inferred_metrics: set[str] = set()
            for result in results:
                scores_blob = result.get("scores")
                if not isinstance(scores_blob, dict):
                    continue
                for metric_name in scores_blob:
                    if isinstance(metric_name, str) and metric_name:
                        inferred_metrics.add(metric_name)
            metric_names = sorted(inferred_metrics)

        metric_scores = {metric: [] for metric in metric_names}

        for result in results:
            scores_blob = result.get("scores")
            if not isinstance(scores_blob, dict):
                continue
            for metric, score in scores_blob.items():
                if metric not in metric_scores:
                    continue
                try:
                    metric_scores[metric].append(float(score))
                except (TypeError, ValueError):
                    continue

        metric_stats = {}
        for metric, scores in metric_scores.items():
            if scores:
                metric_stats[metric] = {
                    "mean": statistics.mean(scores),
                    "std": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "min": min(scores),
                    "max": max(scores)
                }
            else:
                metric_stats[metric] = {"mean": 0, "std": 0, "min": 0, "max": 0}

        return metric_stats

    def _calculate_usage(self, results: list[dict[str, Any]]) -> dict[str, int]:
        """Calculate token usage"""
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0

        for result in results:
            if "usage" in result:
                total_tokens += result["usage"].get("total_tokens", 0)
                prompt_tokens += result["usage"].get("prompt_tokens", 0)
                completion_tokens += result["usage"].get("completion_tokens", 0)

        return {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }

    async def _send_webhook(
        self,
        webhook_url: str,
        run_id: str,
        eval_id: str,
        status: str,
        summary: dict[str, Any]
    ):
        """Send webhook notification"""
        try:
            payload = {
                "event": f"run.{status}",
                "run_id": run_id,
                "eval_id": eval_id,
                "status": status,
                "completed_at": int(datetime.utcnow().timestamp()),
                "results_url": f"/api/v1/runs/{run_id}/results",
                "summary": summary,
            }
            resp = await afetch(method="POST", url=webhook_url, json=payload, timeout=10, retry=RetryPolicy(attempts=1))
            if resp.status_code >= 400:
                try:
                    resp.raise_for_status()
                except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS:
                    raise
            logger.info(f"Webhook sent to {webhook_url} for run {run_id}")

        except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to send webhook to {webhook_url}: {e}")

    def cancel_run(self, run_id: str) -> bool:
        """Cancel a running evaluation"""
        task = self.running_tasks.get(run_id)
        if task is None:
            return False

        run = self.db.get_run(run_id)
        current_status = normalize_run_status(run.get("status") if run else None)

        task.cancel()

        if not can_transition_run_status(current_status, "cancelled"):
            self.running_tasks.pop(run_id, None)
            return False

        self.db.update_run_status(run_id, "cancelled")
        self.running_tasks.pop(run_id, None)
        return True

    def get_run_status(self, run_id: str) -> Optional[str]:
        """Get the status of a run"""
        run = self.db.get_run(run_id)
        if run:
            return normalize_run_status(run.get("status"))
        return None

    async def shutdown(self) -> None:
        """Cancel any running tasks and clear internal bookkeeping."""
        for task in list(self.running_tasks.values()):
            if task.done():
                continue
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self.running_tasks.clear()

    # Alias for compatibility with tests
    async def run_evaluation_async(self, run_id: str, eval_config: dict[str, Any]):
        """Alias for run_evaluation to match test expectations"""
        eval_id = eval_config.get("eval_id")
        if not eval_id:
            raise ValueError("eval_id required in eval_config")
        return await self.run_evaluation(run_id, eval_id, eval_config, background=True)
