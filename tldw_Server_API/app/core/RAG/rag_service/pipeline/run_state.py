"""``PipelineState``: the shared mutable state of one ``unified_rag_pipeline`` call (TASK-13950.24).

Each slot is one local of the old function: the 35 parameters the body reassigns (copied from the
frozen ``RagRunConfig``) and ``result``. A slot that is not set reads like an unbound local, so a
read that fails today fails the same way: ``UnboundLocalError``, which the pipeline's ``except``
ladder does not catch, rather than an ``AttributeError``, which it would.
"""

from typing import Any

from .run_config import RagRunConfig
from dataclasses import replace
from typing import Literal
from typing import Optional
from typing import cast
import asyncio
import contextlib
import os
import time
from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

# (group, name) of every parameter the pipeline body assigns; frozen by test_rag_pipeline_types.py.
MUTATED_PARAMS: tuple[tuple[str, str], ...] = (
    ("verification", "adaptive_time_budget_sec"),
    ("scope", "date_range"),
    ("cache", "enable_cache"),
    ("scope", "enable_date_filter"),
    ("research", "enable_discussion_search"),
    ("query_prep", "enable_hyde"),
    ("retrieval", "enable_multi_vector_passages"),
    ("verification", "enable_numeric_fidelity"),
    ("processing", "enable_parent_expansion"),
    ("rerank", "enable_reranking"),
    ("self_correct", "enable_web_fallback"),
    ("query_prep", "expand_query"),
    ("retrieval", "fts_level"),
    ("generation", "generation_prompt"),
    ("retrieval", "hybrid_alpha"),
    ("query_prep", "hyde_model"),
    ("query_prep", "hyde_provider"),
    ("processing", "include_parent_document"),
    ("verification", "low_confidence_behavior"),
    ("runtime", "media_db_path"),
    ("retrieval", "mv_max_spans"),
    ("retrieval", "mv_span_chars"),
    ("retrieval", "mv_stride"),
    ("verification", "nli_model"),
    ("verification", "numeric_fidelity_behavior"),
    ("processing", "parent_context_size"),
    ("processing", "parent_max_tokens"),
    ("request", "query"),
    ("citations", "require_hard_citations"),
    ("rerank", "reranking_strategy"),
    ("runtime", "resolved_request"),
    ("runtime", "retrieval_plan"),
    ("runtime", "sql_retriever"),
    ("generation", "strict_extractive"),
    ("retrieval", "top_k"),
)


class EarlyReturn(Exception):
    """Internal control-flow sentinel for intentional short-circuit paths."""


class PipelineState:
    """One call's mutable state; ``rcfg`` is the frozen configuration it started from."""

    __slots__: tuple[str, ...] = (
        "rcfg",
        "adaptive_time_budget_sec",
        "date_range",
        "enable_cache",
        "enable_date_filter",
        "enable_discussion_search",
        "enable_hyde",
        "enable_multi_vector_passages",
        "enable_numeric_fidelity",
        "enable_parent_expansion",
        "enable_reranking",
        "enable_web_fallback",
        "expand_query",
        "fts_level",
        "generation_prompt",
        "hybrid_alpha",
        "hyde_model",
        "hyde_provider",
        "include_parent_document",
        "low_confidence_behavior",
        "media_db_path",
        "mv_max_spans",
        "mv_span_chars",
        "mv_stride",
        "nli_model",
        "numeric_fidelity_behavior",
        "parent_context_size",
        "parent_max_tokens",
        "query",
        "require_hard_citations",
        "reranking_strategy",
        "resolved_request",
        "retrieval_plan",
        "sql_retriever",
        "strict_extractive",
        "top_k",
        "result",
        # locals the hoisted helpers share with the function body (PR D2)
        "authorization_cache_scope",
        "cache_instance",
        "cache_max_size",
        "cache_setup_attempted",
        "chat_dictionaries_db_path",
        "cumulative_filtered_artifact_counts",
        "optional_provider_failure_code",
        "prompts_db_path",
        "retrieval_index_namespace",
        "retrieval_min_score",
        "retrieval_query",
        "retrieval_search_mode",
        "retrieval_sources",
        "retrieval_top_k",
        "runtime_user_id",
        "security_excluded_counts",
        "source_failures",
        "source_status_retriever",
        "workspace_id",
        "world_books_db_path",
        # locals that cross a planned stage boundary (PR D3), and the five published closures
        "_analyze",
        "_classification",
        "_execute_retrieval_variant",
        "_mark_claims_used",
        "_otel_span",
        "_otel_span_gen",
        "_otel_span_rk",
        "_retrieve_for_claim",
        "_run_claims_engine",
        "_skip_local_retrieval",
        "_skip_retrieval_reason",
        "_skip_retrieval_stack",
        "analysis",
        "analysis_complexity_val",
        "analysis_domain",
        "analysis_intent_val",
        "authorization_cache_error",
        "base_retrieval_executed",
        "cache_retrieval_snapshot",
        "claims_engine_type",
        "claims_handle",
        "claims_payload",
        "claims_state",
        "config",
        "context",
        "context_docs",
        "documents",
        "effective_enable_generation",
        "engine",
        "expanded_queries",
        "factuality_payload",
        "gated_generation",
        "generation_start",
        "generator",
        "hyde_vector",
        "include_rerank_snapshots",
        "job_budget",
        "job_context",
        "llm_client",
        "low_confidence",
        "metrics",
        "query_expansion_corpus",
        "rerank_config",
        "rerank_debug_limit",
        "rerank_start",
        "reranker",
        "reranker_credential_handle",
        "resolved_data_sources",
        "retrieval_cache_eligible",
        "retrieval_scalar_overrides_applied",
        "retrieval_start",
        "retriever",
        "selected_strategy",
        "skip_claims_due_to_groundedness",
        "standard_evidence_coordinated",
        "start_time",
        "vres",
    )

    def __init__(self, rcfg: RagRunConfig) -> None:
        self.rcfg = rcfg
        for group, name in MUTATED_PARAMS:
            setattr(self, name, getattr(getattr(rcfg, group), name))

    def __getattr__(self, name: str) -> Any:
        """Only reached for an unset slot or an unknown name."""
        if name in PipelineState.__slots__:
            raise UnboundLocalError(f"cannot access local variable '{name}' where it is not associated with a value")
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def release(self) -> None:
        """Drop every slot once the run is over.

        The published closures capture ``self`` and are stored on it, a reference cycle that would
        keep the retriever, documents and credential handles alive until a cyclic GC. Nothing runs
        after ``run_rag_pipeline`` returns (every task is awaited), so clearing is safe.
        """
        for name in PipelineState.__slots__:
            with contextlib.suppress(AttributeError):
                delattr(self, name)

    def resolve_effective_enable_generation(self) -> bool:  # an explicit False is retrieval only (TASK-15482)
        return bool(self.rcfg.generation.enable_generation) and bool((self.resolved_request.payload or {}).get("enable_generation", True))

    def planned_index_namespace(self, plan: up.RetrievalPlan) -> Optional[str]:
        if plan.index_namespace is not None:
            return plan.index_namespace
        normalized_sources = {getattr(source, "value", str(source)).strip() for source in plan.sources}
        if "media_db" in normalized_sources:
            return plan.collection_names.get("media_db")
        return None

    def refresh_retrieval_snapshot_from_current_knobs(self,
        *,
        prefer_current_scalars: bool = False,
    ) -> None:

        snapshot_query = self.query
        snapshot_sources = self.rcfg.scope.sources
        snapshot_search_mode = self.rcfg.retrieval.search_mode
        snapshot_top_k = self.top_k
        snapshot_min_score = self.rcfg.retrieval.min_score
        snapshot_index_namespace = self.rcfg.retrieval.index_namespace

        if self.retrieval_plan is not None:
            planned_query = str(self.retrieval_plan.query or "").strip()
            if planned_query:
                snapshot_query = planned_query
            planned_sources = [
                str(getattr(source, "value", source)).strip()
                for source in self.retrieval_plan.sources
                if str(getattr(source, "value", source)).strip()
            ]
            if planned_sources:
                snapshot_sources = planned_sources
            planned_search_mode = str(self.retrieval_plan.search_mode or "").strip().lower()
            if planned_search_mode in {"fts", "vector", "hybrid"}:
                snapshot_search_mode = cast(
                    Literal["fts", "vector", "hybrid"],
                    planned_search_mode,
                )
            if not prefer_current_scalars:
                with contextlib.suppress(TypeError, ValueError):
                    snapshot_top_k = max(1, int(self.retrieval_plan.top_k))
                with contextlib.suppress(TypeError, ValueError):
                    snapshot_min_score = float(self.retrieval_plan.min_score)
            snapshot_index_namespace = self.planned_index_namespace(self.retrieval_plan)

        self.retrieval_query = snapshot_query
        self.retrieval_sources = snapshot_sources
        self.retrieval_search_mode = snapshot_search_mode
        with contextlib.suppress(TypeError, ValueError):
            self.retrieval_top_k = max(1, int(snapshot_top_k))
        with contextlib.suppress(TypeError, ValueError):
            self.retrieval_min_score = float(snapshot_min_score)
        self.retrieval_index_namespace = snapshot_index_namespace

        if self.retrieval_plan is not None:
            self.retrieval_plan = replace(
                self.retrieval_plan,
                query=str(self.retrieval_query),
                sources=tuple(str(source) for source in (self.retrieval_sources or [])),
                search_mode=str(self.retrieval_search_mode),
                top_k=int(self.retrieval_top_k),
                min_score=float(self.retrieval_min_score),
                index_namespace=self.retrieval_index_namespace,
            )

    def retrieval_plan_for(self, query_text: str) -> Optional[up.RetrievalPlan]:
        if self.retrieval_plan is None:
            return None
        if query_text == self.retrieval_plan.query:
            return self.retrieval_plan
        try:
            return replace(self.retrieval_plan, query=query_text)
        except TypeError:
            return self.retrieval_plan

    def sync_effective_query(self, new_query: str) -> None:
        self.query = str(new_query)
        self.retrieval_query = self.query
        self.result.query = self.query
        result_payload = getattr(self.result, "payload", None)
        if isinstance(result_payload, dict):
            result_payload["query"] = self.query
        self.resolved_request.query = self.query
        self.resolved_request.payload["query"] = self.query
        if self.retrieval_plan is not None:
            self.retrieval_plan = replace(self.retrieval_plan, query=self.query)

    def latch_optional_provider_failure(self,
        component: str,
        exc: BaseException,
    ) -> str:
        """Latch the first optional retrieval provider failure for this request."""
        if self.optional_provider_failure_code is None:
            self.optional_provider_failure_code = up._bounded_provider_failure_code(exc)
        self.result.metadata.setdefault("retrieval_coverage", {})[component] = {
            "coverage": "degraded",
            "failure_code": self.optional_provider_failure_code,
        }
        return self.optional_provider_failure_code

    def latched_optional_provider_failure(self, component: str) -> str | None:
        """Return and record the request's latched optional provider failure."""
        if self.rcfg.runtime.credential_runtime is None or self.optional_provider_failure_code is None:
            return None
        self.result.metadata.setdefault("retrieval_coverage", {})[component] = {
            "coverage": "degraded",
            "failure_code": self.optional_provider_failure_code,
        }
        return self.optional_provider_failure_code

    def ensure_profile_resolution_metadata(self) -> dict[str, Any]:
        profile_resolution = self.result.metadata.get("profile_resolution")
        if not isinstance(profile_resolution, dict):
            profile_resolution = {}
        profile_resolution.setdefault("requested_profile", self.rcfg.request.rag_profile)
        profile_resolution.setdefault("applied_profile", self.rcfg.request.rag_profile)
        profile_resolution.setdefault("effective_overrides_count", 0)
        if not isinstance(profile_resolution.get("degraded_features"), list):
            profile_resolution["degraded_features"] = []
        self.result.metadata["profile_resolution"] = profile_resolution
        return profile_resolution

    def record_profile_degradation(self,
        *,
        component: str,
        from_value: str,
        to_value: str,
        reason: str,
    ) -> None:
        if not self.rcfg.observability.include_retrieval_diagnostics:
            return
        profile_resolution = self.ensure_profile_resolution_metadata()
        degraded_features = profile_resolution.get("degraded_features")
        if not isinstance(degraded_features, list):
            degraded_features = []
            profile_resolution["degraded_features"] = degraded_features
        degraded_features.append(
            {
                "component": component,
                "from": from_value,
                "to": to_value,
                "reason": reason,
            }
        )

    def build_cache_namespace(self) -> str:
        def _cache_path_value(value: Any) -> Optional[str]:
            if isinstance(value, (str, os.PathLike)):
                return os.fspath(value)
            return None

        authorization_user_id = (
            self.authorization_cache_scope.get("user_id")
            if self.authorization_cache_scope is not None
            else None
        )
        owner_scope: dict[str, str | dict[str, str | None]]
        if authorization_user_id is not None:
            owner_scope = {"kind": "user", "id": str(authorization_user_id)}
        elif self.runtime_user_id is not None:
            owner_scope = {
                "kind": "user",
                "id": str(self.runtime_user_id),
            }
        elif self.rcfg.runtime.credential_runtime is None and self.rcfg.scope.user_id is not None:
            owner_scope = {"kind": "user", "id": str(self.rcfg.scope.user_id)}
        else:
            owner_paths = {
                "media": _cache_path_value(self.media_db_path)
                or _cache_path_value(getattr(self.rcfg.runtime.media_db, "db_path", None)),
                "notes": _cache_path_value(self.rcfg.runtime.notes_db_path),
                "characters": _cache_path_value(self.rcfg.runtime.character_db_path),
                "kanban": _cache_path_value(self.rcfg.runtime.kanban_db_path),
                "prompts": _cache_path_value(self.prompts_db_path),
                "world_books": _cache_path_value(self.world_books_db_path),
                "dictionaries": _cache_path_value(self.chat_dictionaries_db_path),
            }
            owner_scope = {
                "kind": "database" if any(owner_paths.values()) else "server",
                "paths": owner_paths,
            }

        collection_scope = (
            {
                str(key): str(value)
                for key, value in sorted(self.retrieval_plan.collection_names.items())
            }
            if self.retrieval_plan is not None
            else {}
        )
        normalized_search_mode = str(self.retrieval_search_mode or "").strip().lower()
        effective_date_range = None
        if self.enable_date_filter and isinstance(self.date_range, dict):
            effective_date_range = {
                "start": str(self.date_range.get("start") or ""),
                "end": str(self.date_range.get("end") or ""),
            }
        retrieval_scope = {
            "sources": sorted(
                up._normalize_pipeline_sources(
                    list(self.retrieval_sources) if self.retrieval_sources is not None else None
                )
            ),
            "search_mode": normalized_search_mode,
            "use_fts": normalized_search_mode in {"fts", "hybrid"},
            "use_vector": normalized_search_mode in {"vector", "hybrid"},
            "top_k": int(self.retrieval_top_k),
            "min_score": float(self.retrieval_min_score),
            "fts_level": str(self.fts_level),
            "date_range": effective_date_range,
            "late_chunking": {
                "enabled": bool(self.rcfg.processing.enable_text_late_chunking),
                "method": self.rcfg.processing.chunk_method,
                "size": self.rcfg.processing.chunk_size,
                "overlap": self.rcfg.processing.chunk_overlap,
                "language": self.rcfg.processing.chunk_language,
            },
            "index_namespace": str(self.retrieval_index_namespace or ""),
            "collection_names": collection_scope,
        }
        namespace_parts = [
            f"owner:{up._cache_identity_digest(owner_scope)}",
            f"workspace:{up._cache_identity_digest({'id': 'global' if self.workspace_id is None else str(self.workspace_id)})}",
            f"retrieval:{up._cache_identity_digest(retrieval_scope)}",
        ]
        if self.authorization_cache_scope is not None:
            namespace_parts.insert(
                1,
                f"authorization:{up._cache_identity_digest(self.authorization_cache_scope)}",
            )
        return "|".join(namespace_parts)

    def get_cache_instance(self):
        if self.cache_setup_attempted:
            return self.cache_instance
        self.cache_setup_attempted = True
        cache_cls = None
        if self.rcfg.cache.adaptive_cache and up.AdaptiveCache:
            cache_cls = up.AdaptiveCache
        elif up.SemanticCache:
            cache_cls = up.SemanticCache

        if cache_cls:
            try:
                cache_namespace = self.build_cache_namespace()
                if up.get_shared_cache:
                    self.cache_instance = up.get_shared_cache(
                        cache_cls=cache_cls,
                        similarity_threshold=self.rcfg.cache.cache_threshold,
                        ttl=self.rcfg.cache.cache_ttl,
                        max_size=self.cache_max_size,
                        namespace=cache_namespace,
                    )
                else:
                    self.cache_instance = cache_cls(
                        similarity_threshold=self.rcfg.cache.cache_threshold,
                        ttl=self.rcfg.cache.cache_ttl,
                        namespace=cache_namespace,
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                self.cache_instance = None
                up.logger.warning(
                    "Semantic cache disabled after scoped setup failure "
                    "(error_type={})",
                    type(exc).__name__,
                )
        # Register with the RAGCache facade so health endpoints see real stats
        if self.cache_instance is not None:
            try:
                from ..advanced_cache import register_semantic_cache

                register_semantic_cache(self.cache_instance)
            except (ImportError, TypeError):
                pass
        return self.cache_instance

    async def with_timeout(self, coro, timeout: Optional[float]):
        if timeout and timeout > 0:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro

    async def resilient_call(self,
        component: str,
        func,
        *args,
        bypass_provider_failures: bool = False,
        **kwargs,
    ):
        """Apply circuit breaker, retries, and timeout around async operations when enabled."""
        breaker = None
        if self.rcfg.resilience.enable_resilience and self.rcfg.resilience.circuit_breaker and up.get_coordinator and up.CircuitBreakerConfig:
            try:
                coord = up.get_coordinator()
                if component not in coord.circuit_breakers:
                    coord.register_circuit_breaker(component, up.CircuitBreakerConfig())
                breaker = coord.circuit_breakers[component]
            except (AttributeError, KeyError, TypeError):
                breaker = None

        async def _invoke():
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            except up._RAG_PROVIDER_FAILURES as exc:
                if bypass_provider_failures:
                    raise up._ProviderFailureBypass(exc) from None
                raise

        async def _attempt():
            if breaker is not None:
                return await breaker.call(_invoke)
            return await _invoke()

        if self.rcfg.resilience.enable_resilience and (self.rcfg.resilience.retry_attempts or 0) > 1 and up.RetryPolicy and up.RetryConfig:
            policy = up.RetryPolicy(up.RetryConfig(max_attempts=int(self.rcfg.resilience.retry_attempts or 1)))
            call_coro = policy.execute(_attempt)
        else:
            call_coro = _attempt()

        try:
            return await self.with_timeout(call_coro, self.rcfg.resilience.timeout_seconds)
        except up._ProviderFailureBypass as bypass:
            raise bypass.failure from None

    def apply_generation_gate(self,
        reason: str,
        *,
        coverage: Optional[float] = None,
        unsupported_ratio: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """Record a gating event in metadata for downstream observability."""
        gate = self.result.metadata.setdefault("generation_gate", {})
        gate.update(
            {
                "reason": reason,
                "at": time.time(),
            }
        )
        if coverage is not None:
            gate["coverage"] = coverage
        if unsupported_ratio is not None:
            gate["unsupported_ratio"] = unsupported_ratio
        if threshold is not None:
            gate["threshold"] = threshold

    def build_multi_retriever(self, db_paths: dict[str, str]):
        base_kwargs: dict[str, Any] = {"user_id": self.rcfg.scope.user_id or "0", "media_db": self.rcfg.runtime.media_db}
        if self.rcfg.runtime.credential_runtime is not None:
            base_kwargs["credential_runtime"] = self.rcfg.runtime.credential_runtime
        if self.rcfg.runtime.chacha_db is not None:
            base_kwargs["chacha_db"] = self.rcfg.runtime.chacha_db
        prompts_db = self.rcfg.runtime.prompts_db
        if prompts_db is not None:
            base_kwargs["prompts_db"] = prompts_db
        if self.sql_retriever is not None:
            base_kwargs["sql_retriever"] = self.sql_retriever

        optional_keys = ("chacha_db", "prompts_db", "sql_retriever")
        variants = [dict(base_kwargs)]
        for key in optional_keys:
            if key in base_kwargs:
                variants.append({k: v for k, v in base_kwargs.items() if k != key})
        variants.append({k: v for k, v in base_kwargs.items() if k not in set(optional_keys)})
        seen: set[tuple[str, ...]] = set()
        for variant in variants:
            variant_key = tuple(sorted(variant.keys()))
            if variant_key in seen:
                continue
            seen.add(variant_key)
            try:
                return up.MultiDatabaseRetriever(db_paths, **variant)
            except TypeError:
                continue
        if self.rcfg.runtime.credential_runtime is not None:
            raise TypeError("MultiDatabaseRetriever does not accept credential runtime")
        try:
            return up.MultiDatabaseRetriever(db_paths, user_id=self.rcfg.scope.user_id or "0")
        except TypeError:
            return up.MultiDatabaseRetriever(db_paths)

    def build_pipeline_db_paths(self) -> dict[str, str]:
        """Return the full source-to-database map for every retrieval pass."""
        db_paths: dict[str, str] = {}
        if self.media_db_path:
            db_paths["media_db"] = str(self.media_db_path)
        if self.rcfg.runtime.notes_db_path:
            db_paths["notes_db"] = str(self.rcfg.runtime.notes_db_path)
        if self.prompts_db_path:
            db_paths["prompts_db"] = str(self.prompts_db_path)
        if self.rcfg.runtime.character_db_path:
            db_paths["character_cards_db"] = str(self.rcfg.runtime.character_db_path)
            db_paths["world_books_db"] = str(self.world_books_db_path or self.rcfg.runtime.character_db_path)
            db_paths["chat_dictionaries_db"] = str(self.chat_dictionaries_db_path or self.rcfg.runtime.character_db_path)
        else:
            if self.world_books_db_path:
                db_paths["world_books_db"] = str(self.world_books_db_path)
            if self.chat_dictionaries_db_path:
                db_paths["chat_dictionaries_db"] = str(self.chat_dictionaries_db_path)
        if self.rcfg.runtime.kanban_db_path:
            db_paths["kanban_db"] = str(self.rcfg.runtime.kanban_db_path)
        return db_paths

    def apply_workspace_filtering_to_result(self) -> None:
        """Filter the current result documents and refresh source diagnostics."""
        filtered_documents, filtered_artifact_counts = up._filter_workspace_artifacts(
            list(self.result.documents or []),
            workspace_id=self.workspace_id,
        )
        for source, count in filtered_artifact_counts.items():
            self.cumulative_filtered_artifact_counts[source] = self.cumulative_filtered_artifact_counts.get(source, 0) + count
        self.result.documents = filtered_documents
        if self.rcfg.observability.include_retrieval_diagnostics and self.source_status_retriever is not None:
            self.result.metadata["source_status"] = up._build_source_status(
                up._normalize_pipeline_sources(list(self.retrieval_sources)),
                retriever=self.source_status_retriever,
                documents=filtered_documents,
                filtered_counts=dict(self.cumulative_filtered_artifact_counts),
                source_failures=self.source_failures,
                security_excluded_counts=self.security_excluded_counts,
            )
        if self.workspace_id is not None:
            self.result.metadata["workspace_id"] = self.workspace_id
        self.result.metadata["documents_retrieved"] = len(filtered_documents)

    def decorate_calibration_metadata(self) -> None:
        """
            Attach learned-fusion specific fields (if present) to reranking calibration.

            This keeps Two-Tier calibration behavior intact while making the new
            enable_learned_fusion / calibrator_version flags observable.
            """
        if not isinstance(self.result.metadata, dict):
            return
        cal = self.result.metadata.get("reranking_calibration")
        if not isinstance(cal, dict):
            return
        # Fused score is the calibrated probability of the top document
        if "fused_score" not in cal and "top_doc_prob" in cal:
            with contextlib.suppress(TypeError, ValueError):
                cal["fused_score"] = float(cal.get("top_doc_prob") or 0.0)
        # Mark whether learned fusion was explicitly requested
        if self.rcfg.rerank.enable_learned_fusion:
            cal["enabled"] = True
        # Version tag is purely informational for now
        if self.rcfg.rerank.calibrator_version:
            cal.setdefault("version", self.rcfg.rerank.calibrator_version)
        self.result.metadata["reranking_calibration"] = cal
