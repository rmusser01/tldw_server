"""rerank code moved from unified_pipeline.py (TASK-13950.24)."""

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


async def select_reranker(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Pick the strategy, resolve the LLM reranker credentials and build the reranking config."""
    strategy_map = {
        "flashrank": up.RerankingStrategy.FLASHRANK,
        "cross_encoder": up.RerankingStrategy.CROSS_ENCODER,
        "hybrid": up.RerankingStrategy.HYBRID,
        "llama_cpp": up.RerankingStrategy.LLAMA_CPP,
        "diversity": up.RerankingStrategy.DIVERSITY,
        "llm_scoring": up.RerankingStrategy.LLM_SCORING,
        "two_tier": up.RerankingStrategy.TWO_TIER,
    }

    # Determine LLM reranker provider/model from config when requested
    state.selected_strategy = strategy_map[state.reranking_strategy]
    state.llm_client = None
    state.reranker_credential_handle = None
    if state.selected_strategy in {
        up.RerankingStrategy.LLM_SCORING,
        up.RerankingStrategy.TWO_TIER,
    }:
        try:
            import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
            from tldw_Server_API.app.core.config import load_and_log_configs

            llm_reranker_config = load_and_log_configs()
            if not isinstance(llm_reranker_config, dict):
                llm_reranker_config = {}
            prov = (llm_reranker_config.get("RAG_LLM_RERANKER_PROVIDER") or "").strip()
            model = (llm_reranker_config.get("RAG_LLM_RERANKER_MODEL") or "").strip()
            if not model:
                # No model set -> fallback to FlashRank
                if state.selected_strategy == up.RerankingStrategy.LLM_SCORING:
                    state.selected_strategy = up.RerankingStrategy.FLASHRANK
            else:
                if rcfg.runtime.credential_runtime is not None:
                    try:
                        state.reranker_credential_handle = await rcfg.runtime.credential_runtime.resolve(
                            prov or "openai",
                            model=model,
                        )
                    except up.ByokResolutionError as exc:
                        state.result.metadata["reranking"] = {
                            "degraded": True,
                            "failure_code": exc.code,
                            "verification_available": False,
                        }
                        state.result.metadata.pop(
                            "reranking_calibration",
                            None,
                        )

                class _LLMClient:
                    def __init__(self, provider: str, model_name: str, handle: Any):
                        self.provider = provider or 'openai'
                        self.model_name = model_name
                        self.handle = handle
                        self.credentials_resolved = handle is not None
                        self.used = False

                    def analyze(self, prompt_text: str):
                        # Use analyze with prompt as custom_prompt_arg
                        response = sgl.analyze(
                            api_name=self.provider,
                            input_data=prompt_text,
                            custom_prompt_arg=None,
                            api_key=(
                                self.handle.api_key
                                if self.handle is not None
                                else None
                            ),
                            system_message=None,
                            temp=None,
                            model_override=self.model_name,
                            app_config=(
                                self.handle.app_config
                                if self.handle is not None
                                else None
                            ),
                            credentials_resolved=self.handle is not None,
                            provider_credentials=self.handle,
                            raise_on_error=self.handle is not None,
                        )
                        response = up._consume_bound_sgl_response(
                            response,
                            self.provider,
                            on_content=lambda: setattr(self, "used", True),
                            fail_closed=self.handle is not None,
                        )
                        self.used = True
                        return response

                if rcfg.runtime.credential_runtime is None or state.reranker_credential_handle is not None:
                    state.llm_client = _LLMClient(
                        prov,
                        model,
                        state.reranker_credential_handle,
                    )
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            if state.selected_strategy == up.RerankingStrategy.LLM_SCORING:
                state.selected_strategy = up.RerankingStrategy.FLASHRANK

    # Determine model for reranker when applicable
    model_name_for_reranker = None
    if state.selected_strategy == up.RerankingStrategy.LLAMA_CPP:
        try:
            from tldw_Server_API.app.core.config import load_and_log_configs

            reranker_model_config = load_and_log_configs()
            if not isinstance(reranker_model_config, dict):
                reranker_model_config = {}
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            reranker_model_config = {}
        # Precedence: explicit param -> env/config
        model_name_for_reranker = rcfg.rerank.reranking_model or reranker_model_config.get(
            "RAG_LLAMA_RERANKER_MODEL"
        )
    elif state.selected_strategy == up.RerankingStrategy.CROSS_ENCODER:
        try:
            from tldw_Server_API.app.core.config import load_and_log_configs

            reranker_model_config = load_and_log_configs()
            if not isinstance(reranker_model_config, dict):
                reranker_model_config = {}
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            reranker_model_config = {}
        model_name_for_reranker = rcfg.rerank.reranking_model or reranker_model_config.get(
            "RAG_TRANSFORMERS_RERANKER_MODEL"
        )

    state.rerank_config = up.RerankingConfig(
        strategy=state.selected_strategy,
        top_k=rcfg.rerank.rerank_top_k or state.top_k,
        model_name=model_name_for_reranker,
        # Request-level gating overrides (TwoTier)
        min_relevance_prob=rcfg.rerank.rerank_min_relevance_prob,
        sentinel_margin=rcfg.rerank.rerank_sentinel_margin,
    )
    state.include_rerank_snapshots = rcfg.observability.debug_mode and up._resolve_include_rerank_debug_documents(
        rcfg.observability.include_rerank_debug_documents
    )
    state.rerank_debug_limit = max(1, int(rcfg.rerank.rerank_top_k or state.top_k or 1))


async def apply_rerank(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Run the reranker (two-tier degrades to hybrid on failure) and record snapshots and timings."""
    try:
        if state.include_rerank_snapshots and isinstance(state.result.metadata, dict):
            state.result.metadata["pre_rerank_documents"] = up._serialize_rerank_debug_documents(
                state.result.documents,
                include_content=True,
                limit=state.rerank_debug_limit,
            )
        state.reranker = up.create_reranker(state.selected_strategy, state.rerank_config, llm_client=state.llm_client)
        try:
            reranked = await state.resilient_call(
                "reranking",
                state.reranker.rerank,
                state.query,
                state.result.documents,
            )
        finally:
            if (
                rcfg.runtime.credential_runtime is not None
                and state.reranker_credential_handle is not None
                and state.llm_client is not None
                and state.llm_client.used
            ):
                await rcfg.runtime.credential_runtime.mark_used(
                    state.reranker_credential_handle
                )
    except Exception as rerank_exc:  # noqa: BLE001 - moved from unified_pipeline.py (TASK-13950.24)
        if state.reranking_strategy != "two_tier":
            raise
        up.logger.warning(
            "Two-tier reranker failed at runtime; degrading to hybrid strategy: {}",
            rerank_exc,
        )
        state.record_profile_degradation(
            component="reranking_strategy",
            from_value="two_tier",
            to_value="hybrid",
            reason="runtime_failure",
        )
        state.reranking_strategy = "hybrid"
        state.selected_strategy = up.RerankingStrategy.HYBRID
        state.rerank_config = up.RerankingConfig(
            strategy=state.selected_strategy,
            top_k=rcfg.rerank.rerank_top_k or state.top_k,
            model_name=None,
            min_relevance_prob=rcfg.rerank.rerank_min_relevance_prob,
            sentinel_margin=rcfg.rerank.rerank_sentinel_margin,
        )
        state.reranker = up.create_reranker(state.selected_strategy, state.rerank_config, llm_client=state.llm_client)
        reranked = await state.resilient_call("reranking", state.reranker.rerank, state.query, state.result.documents)
    if reranked and hasattr(reranked[0], "document"):
        state.result.documents = [sd.document for sd in reranked[: (rcfg.rerank.rerank_top_k or state.top_k)]]
    else:
        state.result.documents = reranked[: (rcfg.rerank.rerank_top_k or state.top_k)]
    if state.include_rerank_snapshots and isinstance(state.result.metadata, dict):
        state.result.metadata["reranked_documents"] = up._serialize_rerank_debug_documents(
            state.result.documents,
            include_content=True,
            limit=state.rerank_debug_limit,
        )

    state.result.timings["reranking"] = time.time() - state.rerank_start
    try:
        from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram

        observe_histogram(
            "rag_reranking_duration_seconds",
            state.result.timings["reranking"],
            labels={"strategy": state.reranking_strategy},
        )
        # Also record as a generic phase without difficulty
        observe_histogram(
            "rag_phase_duration_seconds",
            state.result.timings["reranking"],
            labels={"phase": "reranking", "difficulty": "na"},
        )
        if state._otel_span_rk is not None:
            with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                state._otel_span_rk.set_attribute("rag.doc_count", int(len(state.result.documents or [])))
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    if state.metrics:
        state.metrics.reranking_time = state.result.timings["reranking"]


def calibrate_rerank(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Record the reranker's calibration metadata or compute a learned-fusion probability."""
    # If reranker exposes calibration metadata (e.g., TwoTier), record it
    try:
        if hasattr(state.reranker, 'last_metadata') and isinstance(state.reranker.last_metadata, dict):
            if state.reranker.last_metadata.get("degraded"):
                state.result.metadata.pop("reranking_calibration", None)
                reranking_metadata = state.result.metadata.setdefault(
                    "reranking",
                    {},
                )
                reranking_metadata.update(
                    {
                        "degraded": True,
                        "failure_code": reranking_metadata.get(
                            "failure_code"
                        )
                        or state.reranker.last_metadata.get(
                            "failure_code",
                            "provider_unavailable",
                        ),
                        "verification_available": False,
                    }
                )
            else:
                state.result.metadata.setdefault("reranking_calibration", {})
                state.result.metadata["reranking_calibration"].update(state.reranker.last_metadata)
                # Attach learned-fusion specific decoration when applicable
                state.decorate_calibration_metadata()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    # If learned fusion is requested but no calibrator metadata exists,
    # compute a simple fused probability from the top document score.
    # A credential-degraded reranker has no trustworthy verification
    # signal, so it must not synthesize calibration or gate generation.
    if rcfg.rerank.enable_learned_fusion:
        try:
            reranking_status = state.result.metadata.get("reranking", {})
            verification_unavailable = (
                isinstance(reranking_status, dict)
                and reranking_status.get("verification_available") is False
            )
            if (
                isinstance(state.result.metadata, dict)
                and "reranking_calibration" not in state.result.metadata
                and not verification_unavailable
            ):
                top_doc = state.result.documents[0] if state.result.documents else None
                if top_doc is not None:
                    import math as _math_lf
                    import os as _os_lf

                    # Use shared env weights to stay consistent with Two-Tier,
                    # but only CE-style weight is applied since we only have
                    # a single rerank score available here.
                    try:
                        bias = float(_os_lf.getenv("RAG_RERANK_CALIB_BIAS", "-1.5"))
                    except (TypeError, ValueError):
                        bias = -1.5
                    try:
                        w_ce = float(_os_lf.getenv("RAG_RERANK_CALIB_W_CE", "2.5"))
                    except (TypeError, ValueError):
                        w_ce = 2.5
                    try:
                        raw = float(getattr(top_doc, "score", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        raw = 0.0
                    logit = bias + (w_ce * raw)
                    try:
                        fused_prob = 1.0 / (1.0 + _math_lf.exp(-logit))
                    except (TypeError, ValueError):
                        fused_prob = 0.5
                    # Threshold from env (same as Two-Tier gating default)
                    try:
                        thr = float(_os_lf.getenv("RAG_MIN_RELEVANCE_PROB", "0.35"))
                    except (TypeError, ValueError):
                        thr = 0.35
                    gated_flag = fused_prob < thr
                    state.result.metadata["reranking_calibration"] = {
                        "strategy": str(state.reranking_strategy),
                        "top_doc_score": raw,
                        "fused_score": fused_prob,
                        "threshold": thr,
                        "gated": gated_flag,
                    }
                    # Ensure learned-fusion metadata decoration is applied
                    state.decorate_calibration_metadata()
        except (AttributeError, TypeError, ValueError):
            pass


async def run_rerank(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Reranking under the rag.rerank span."""
    if state.enable_reranking and state.result.documents and state.reranking_strategy != "none":
        state.rerank_start = time.time()
        if state.reranking_strategy == "two_tier" and not (up.create_reranker and up.RerankingStrategy and up.RerankingConfig):
            state.record_profile_degradation(
                component="reranking_strategy",
                from_value="two_tier",
                to_value="hybrid",
                reason="unavailable_dependency",
            )
            state.reranking_strategy = "hybrid"
        try:
            # --- OTEL: reranking span ---
            _otel_cm_rk = None
            state._otel_span_rk = None
            if rcfg.observability.enable_observability and up.get_telemetry_manager:
                try:
                    _tm = up.get_telemetry_manager()
                    _tr = _tm.get_tracer("tldw.rag")
                    _attrs = {
                        "rag.phase": "rerank",
                        "rag.strategy": str(state.reranking_strategy),
                        "rag.top_k": int((rcfg.rerank.rerank_top_k or state.top_k) or 0),
                    }
                    _otel_cm_rk = _tr.start_as_current_span("rag.rerank")
                    state._otel_span_rk = _otel_cm_rk.__enter__()
                    for _k, _v in _attrs.items():
                        with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                            state._otel_span_rk.set_attribute(_k, _v)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    _otel_cm_rk = None
                    state._otel_span_rk = None
            if up.create_reranker and up.RerankingStrategy and up.RerankingConfig:
                await select_reranker(rcfg, state)
                await apply_rerank(rcfg, state)

                calibrate_rerank(rcfg, state)

            else:
                state.result.errors.append("Reranking module not available")
                up.logger.warning("Reranking requested but module not available")
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Reranking failed: {str(e)}")
            up.logger.error(f"Reranking error: {e}")
            # Sample payload exemplar on reranking failure
            try:
                from ..payload_exemplars import maybe_record_exemplar

                maybe_record_exemplar(
                    query=state.query,
                    documents=state.result.documents or [],
                    answer=state.result.generated_answer or "",
                    reason="reranking_error",
                    user_id=rcfg.scope.user_id,
                    namespace=rcfg.retrieval.index_namespace,
                )
            except (ImportError, RuntimeError, TypeError, ValueError):
                pass
        finally:
            if _otel_cm_rk is not None:
                with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                    _otel_cm_rk.__exit__(None, None, None)
