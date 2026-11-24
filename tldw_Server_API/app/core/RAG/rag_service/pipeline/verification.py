"""verification code moved from unified_pipeline.py (TASK-13950.24)."""

import asyncio
import time
from typing import TYPE_CHECKING

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


async def check_citations_and_groundedness(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Hard and quote-level citations, the fast groundedness check and the claims-skip flag."""
    # ========== HARD CITATIONS (per-sentence) ==========
    # Build per-sentence citation map using claims (if available) or heuristic fallback
    try:
        if state.result.generated_answer:
            hc = None
            # Prefer claims payload if present
            state.claims_payload = state.result.metadata.get("claims") if isinstance(state.result.metadata, dict) else None
            if up.build_hard_citations:
                hc = up.build_hard_citations(
                    state.result.generated_answer, state.result.documents or [], claims_payload=state.claims_payload
                )
            if isinstance(hc, dict):
                state.result.metadata["hard_citations"] = hc
                # If hard-citation coverage is incomplete and strict mode is requested, apply behavior
                if bool(state.require_hard_citations):
                    cov = float(hc.get("coverage") or 0.0)
                    if cov < 1.0:
                        state.apply_generation_gate("missing_hard_citations", coverage=cov)
                        try:
                            from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter

                            increment_counter("rag_missing_hard_citations_total", 1)
                        except (ImportError, RuntimeError, TypeError, ValueError):
                            pass
                        # Honor low_confidence_behavior
                        if state.low_confidence_behavior == "ask":
                            note = "\n\n[Note] Some statements lack supporting citations. Please clarify or provide sources."
                            state.result.generated_answer = (state.result.generated_answer or "") + note
                        elif state.low_confidence_behavior == "decline":
                            state.result.generated_answer = (
                                "Insufficient evidence: missing citations for some statements."
                            )
                    # Gauge for coverage (report once per answer)
                    try:
                        from tldw_Server_API.app.core.Metrics.metrics_manager import set_gauge

                        set_gauge("rag_hard_citation_coverage", cov, labels={"strategy": "standard"})
                    except (ImportError, RuntimeError, TypeError, ValueError):
                        pass
    except (AttributeError, RuntimeError, TypeError, ValueError) as e:
        state.result.errors.append(f"Hard citations mapping failed: {str(e)}")

    # ========== QUOTE-LEVEL CITATIONS ==========
    try:
        if state.result.generated_answer and up.build_quote_citations:
            qc = up.build_quote_citations(state.result.generated_answer, state.result.documents or [])
            if isinstance(qc, dict):
                state.result.metadata["quote_citations"] = qc
    except (AttributeError, RuntimeError, TypeError, ValueError) as e:
        state.result.errors.append(f"Quote citations mapping failed: {str(e)}")

    # ========== FAST GROUNDEDNESS CHECK (Self-Correcting RAG Stage 5) ==========
    # Lightweight check before expensive claims extraction
    fast_grounded = True  # default assumption
    fast_groundedness_confidence = 0.0
    try:
        if rcfg.self_correct.enable_fast_hallucination_check and state.result.generated_answer and up.check_fast_groundedness:
            with up.otel_span("rag.fast_groundedness_check"):
                fg_result, fg_meta = await up.check_fast_groundedness(
                    query=state.query,
                    answer=state.result.generated_answer,
                    documents=state.result.documents or [],
                    provider=rcfg.self_correct.fast_hallucination_provider,
                    model=rcfg.self_correct.fast_hallucination_model,
                    timeout_sec=rcfg.self_correct.fast_hallucination_timeout_sec,
                    credential_runtime=rcfg.runtime.credential_runtime,
                )
                fast_grounded = fg_result.is_grounded
                fast_groundedness_confidence = fg_result.confidence
                state.result.metadata["fast_groundedness"] = fg_meta

                # If highly confident grounded, can optionally skip full claims
                if fast_grounded and fast_groundedness_confidence >= 0.9:
                    state.result.metadata["fast_groundedness"]["skip_full_claims"] = True
    except (
        AttributeError,
        ConnectionError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        asyncio.TimeoutError,
    ) as e:
        state.result.errors.append(f"Fast groundedness check failed: {str(e)}")

    # ========== CLAIMS & FACTUALITY ==========
    # Optionally skip if fast groundedness check passed with high confidence
    state.skip_claims_due_to_groundedness = (
        rcfg.self_correct.enable_fast_hallucination_check
        and fast_grounded
        and fast_groundedness_confidence >= 0.9
        and state.result.metadata.get("fast_groundedness", {}).get("skip_full_claims", False)
    )


async def run_numeric_fidelity(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Verify the numeric tokens of the answer against the sources."""
    # ========== NUMERIC FIDELITY (verify numeric tokens) ==========
    try:
        if state.result.generated_answer and up.check_numeric_fidelity:
            nf = up.check_numeric_fidelity(state.result.generated_answer, state.result.documents or [])
            if nf:
                state.result.metadata.setdefault("numeric_fidelity", {})
                state.result.metadata["numeric_fidelity"].update(
                    {
                        "present": sorted(nf.present),
                        "missing": sorted(nf.missing),
                        "source_numbers": sorted(nf.union_source_numbers)[:100],
                    }
                )
                if nf.missing:
                    try:
                        from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter

                        increment_counter("rag_numeric_mismatches_total", len(nf.missing))
                    except (ImportError, RuntimeError, TypeError, ValueError):
                        pass
                    # Optional corrective action
                    if state.enable_numeric_fidelity and state.numeric_fidelity_behavior in {"retry", "ask", "decline"}:
                        if state.numeric_fidelity_behavior == "retry":
                            # Best-effort: targeted retrieval on missing numbers (bounded)
                            try:
                                if (
                                    up.MultiDatabaseRetriever
                                    and up.RetrievalConfig
                                    and state.media_db_path
                                    and up._scope_includes_data_source(state.resolved_data_sources, up.DataSource.MEDIA_DB)
                                ):
                                    mdr = state.build_multi_retriever({"media_db": state.media_db_path})
                                    conf = up.RetrievalConfig(
                                        max_results=min(10, state.top_k),
                                        min_score=rcfg.retrieval.min_score,
                                        use_fts=True,
                                        use_vector=True,
                                        include_metadata=True,
                                        fts_level=state.fts_level,
                                        enable_text_late_chunking=rcfg.processing.enable_text_late_chunking,
                                        chunk_method=rcfg.processing.chunk_method,
                                        chunk_size=rcfg.processing.chunk_size,
                                        chunk_overlap=rcfg.processing.chunk_overlap,
                                        chunk_language=rcfg.processing.chunk_language,
                                    )
                                    numeric_added: list[up.Document] = []
                                    for tok in list(nf.missing)[:3]:
                                        latched_code = state.latched_optional_provider_failure(
                                            "numeric_fidelity"
                                        )
                                        if latched_code is not None:
                                            state.result.metadata.setdefault("numeric_fidelity", {}).update(
                                                embedding_coverage="degraded",
                                                failure_code=latched_code,
                                            )
                                            break
                                        try:
                                            numeric_added.extend(await mdr.retrieve(query=f"{state.query} {tok}", sources=[up.DataSource.MEDIA_DB], config=conf, index_namespace=rcfg.retrieval.index_namespace))
                                        except up._RAG_PROVIDER_FAILURES as exc:
                                            if rcfg.runtime.credential_runtime is None:
                                                raise
                                            failure_code = state.latch_optional_provider_failure(
                                                "numeric_fidelity",
                                                exc,
                                            )
                                            state.result.metadata.setdefault("numeric_fidelity", {}).update(
                                                embedding_coverage="degraded",
                                                failure_code=failure_code,
                                            )
                                            break
                                        except (
                                            AttributeError,
                                            ConnectionError,
                                            OSError,
                                            RuntimeError,
                                            TypeError,
                                            ValueError,
                                            asyncio.TimeoutError,
                                        ):
                                            continue
                                    if numeric_added:
                                        # Merge with existing docs and optionally re-rerank in place
                                        by_id_numeric: dict[str, up.Document] = {
                                            getattr(d, "id", ""): d for d in (state.result.documents or [])
                                        }
                                        for d in numeric_added:
                                            cur = by_id_numeric.get(getattr(d, "id", ""))
                                            if cur is None or float(getattr(d, "score", 0.0)) > float(
                                                getattr(cur, "score", 0.0)
                                            ):
                                                by_id_numeric[getattr(d, "id", "")] = d
                                        state.result.documents = sorted(
                                            by_id_numeric.values(),
                                            key=lambda x: getattr(x, "score", 0.0),
                                            reverse=True,
                                        )[: max(state.top_k, 10)]
                                        state.result.metadata.setdefault("numeric_fidelity", {})
                                        state.result.metadata["numeric_fidelity"]["retry_docs_added"] = len(numeric_added)
                                        # Attempt quick regeneration if generator is available
                                        if up.AnswerGenerator:
                                            try:
                                                state.generator = up.AnswerGenerator(
                                                    model=rcfg.generation.generation_model,
                                                    provider=rcfg.generation.generation_provider,
                                                    credential_runtime=rcfg.runtime.credential_runtime,
                                                )
                                                state.context = "\n\n".join(
                                                    [
                                                        getattr(d, "content", str(d))
                                                        for d in (state.result.documents[:5] if state.result.documents else [])
                                                    ]
                                                )
                                                regen = await state.generator.generate(
                                                    query=state.query,
                                                    context=state.context,
                                                    prompt_template=state.generation_prompt,
                                                    max_tokens=rcfg.generation.max_generation_tokens,
                                                )
                                                if isinstance(regen, dict) and regen.get("answer"):
                                                    state.result.generated_answer = regen.get("answer")
                                                elif isinstance(regen, str):
                                                    state.result.generated_answer = regen
                                            except (
                                                AttributeError,
                                                ConnectionError,
                                                OSError,
                                                RuntimeError,
                                                TypeError,
                                                ValueError,
                                                asyncio.TimeoutError,
                                            ):
                                                pass
                            except (
                                AttributeError,
                                ConnectionError,
                                OSError,
                                RuntimeError,
                                TypeError,
                                ValueError,
                                asyncio.TimeoutError,
                            ):
                                pass
                        elif state.numeric_fidelity_behavior == "ask":
                            note = "\n\n[Note] Some numeric values could not be verified against sources. Please clarify or provide references."
                            state.result.generated_answer = (state.result.generated_answer or "") + note
                        elif state.numeric_fidelity_behavior == "decline":
                            state.result.generated_answer = (
                                "Insufficient evidence to verify numeric claims in the current context."
                            )
    except (AttributeError, RuntimeError, TypeError, ValueError) as e:
        state.result.errors.append(f"Numeric fidelity check failed: {str(e)}")


async def verify_answer(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Run the post-generation verifier and record its outcome."""
    verifier = up.PostGenerationVerifier(
        max_retries=rcfg.verification.adaptive_max_retries,
        unsupported_threshold=rcfg.verification.adaptive_unsupported_threshold,
        max_claims=rcfg.verification.adaptive_max_claims,
        time_budget_sec=state.adaptive_time_budget_sec,
        use_advanced_rewrites=rcfg.verification.adaptive_advanced_rewrites,
        credential_runtime=rcfg.runtime.credential_runtime,
    )
    state.vres = await verifier.verify_and_maybe_fix(
        query=state.query,
        answer=state.result.generated_answer,
        base_documents=state.result.documents or [],
        media_db_path=state.media_db_path,
        notes_db_path=rcfg.runtime.notes_db_path,
        character_db_path=rcfg.runtime.character_db_path,
        user_id=rcfg.scope.user_id,
        generation_model=rcfg.generation.generation_model,
        generation_provider=rcfg.generation.generation_provider,
        existing_claims=state.claims_payload,
        existing_summary=state.factuality_payload,
        search_mode=rcfg.retrieval.search_mode,
        hybrid_alpha=state.hybrid_alpha,
        top_k=state.top_k,
    )
    # Attach verification metadata
    state.result.metadata.setdefault("post_verification", {})
    state.result.metadata["post_verification"].update({
        "unsupported_ratio": state.vres.unsupported_ratio,
        "total_claims": state.vres.total_claims,
        "unsupported_count": state.vres.unsupported_count,
        "fixed": state.vres.fixed,
        "reason": state.vres.reason,
        "verification_available": getattr(
            state.vres,
            "verification_available",
            True,
        ),
    })
    verification_failure_code = getattr(state.vres, "failure_code", None)
    if verification_failure_code:
        state.result.metadata["post_verification"]["failure_code"] = verification_failure_code
    if getattr(state.vres, "embedding_coverage", None) == "degraded":
        state.result.metadata["post_verification"]["embedding_coverage"] = "degraded"
    embedding_failure_code = getattr(state.vres, "embedding_failure_code", None)
    if embedding_failure_code in {
        "invalid_provider_credentials",
        "missing_provider_credentials",
        "provider_configuration_invalid",
        "credential_store_unavailable",
        "credential_scope_revoked",
        "provider_unavailable",
    }:
        state.result.metadata["post_verification"]["embedding_failure_code"] = (
            embedding_failure_code
        )
    # Gauge for NLI unsupported ratio
    try:
        from tldw_Server_API.app.core.Metrics.metrics_manager import set_gauge

        set_gauge(
            "rag_nli_unsupported_ratio",
            float(state.vres.unsupported_ratio or 0.0),
            labels={"strategy": "standard"},
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    # Optionally override final answer on successful repair
    if state.vres.fixed and state.vres.new_answer:
        state.result.generated_answer = state.vres.new_answer
    # Behavior toggles on low confidence and not fixed
    state.low_confidence = (state.vres.unsupported_ratio > rcfg.verification.adaptive_unsupported_threshold) and (not state.vres.fixed)
    if state.low_confidence:
        state.apply_generation_gate(
            "nli_low_confidence",
            unsupported_ratio=state.vres.unsupported_ratio,
            threshold=rcfg.verification.adaptive_unsupported_threshold,
        )
        try:
            from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter

            increment_counter("rag_nli_low_confidence_total", 1)
        except (ImportError, RuntimeError, TypeError, ValueError):
            pass
        if state.low_confidence_behavior == "ask":
            note = "\n\n[Note] Evidence is insufficient; please clarify or provide more context."
            state.result.generated_answer = (state.result.generated_answer or "") + note
        elif state.low_confidence_behavior == "decline":
            state.result.generated_answer = "Insufficient evidence found to answer confidently."
    # Sample payload exemplar on failure for debugging (redacted)
    try:
        if state.low_confidence:
            from ..payload_exemplars import maybe_record_exemplar

            maybe_record_exemplar(
                query=state.query,
                documents=state.result.documents or [],
                answer=state.result.generated_answer or "",
                reason="post_verification_low_confidence",
                user_id=rcfg.scope.user_id,
                namespace=rcfg.retrieval.index_namespace,
            )
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass


async def adaptive_rerun(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Re-run the pipeline with adapted settings and keep the better answer."""
    if state.low_confidence and rcfg.verification.adaptive_rerun_on_low_confidence and not rcfg.verification._adaptive_rerun:
        rerun_start = time.time()
        # Prefer to broaden recall on rerun
        rerun_expand = state.expand_query if state.expand_query else True
        # Build rerun with post-verification off and a guard to prevent nesting
        new_result = await up.unified_rag_pipeline(
            query=state.query,
            sources=rcfg.scope.sources,
            media_db_path=state.media_db_path,
            notes_db_path=rcfg.runtime.notes_db_path,
            character_db_path=rcfg.runtime.character_db_path,
            kanban_db_path=rcfg.runtime.kanban_db_path,
            # Prefer adapters to avoid raw SQL in prod
            media_db=rcfg.runtime.media_db,
            chacha_db=rcfg.runtime.chacha_db,
            # Use same retrieval/reranking settings but broaden expansion
            search_mode=rcfg.retrieval.search_mode,
            fts_level=state.fts_level,
            hybrid_alpha=state.hybrid_alpha,
            top_k=state.top_k,
            min_score=rcfg.retrieval.min_score,
            expand_query=rerun_expand,
            expansion_strategies=rcfg.query_prep.expansion_strategies,
            spell_check=rcfg.query_prep.spell_check,
            enable_cache=False if rcfg.verification.adaptive_rerun_bypass_cache else state.enable_cache,
            cache_threshold=rcfg.cache.cache_threshold,
            adaptive_cache=rcfg.cache.adaptive_cache,
            keyword_filter=rcfg.scope.keyword_filter,
            include_media_ids=rcfg.scope.include_media_ids,
            include_note_ids=rcfg.scope.include_note_ids,
            enable_security_filter=rcfg.safety.enable_security_filter,
            detect_pii=rcfg.safety.detect_pii,
            redact_pii=rcfg.safety.redact_pii,
            sensitivity_level=rcfg.safety.sensitivity_level,
            content_filter=rcfg.safety.content_filter,
            enable_table_processing=rcfg.processing.enable_table_processing,
            table_method=rcfg.processing.table_method,
            enable_vlm_late_chunking=rcfg.processing.enable_vlm_late_chunking,
            vlm_backend=rcfg.processing.vlm_backend,
            vlm_detect_tables_only=rcfg.processing.vlm_detect_tables_only,
            vlm_max_pages=rcfg.processing.vlm_max_pages,
            vlm_late_chunk_top_k_docs=rcfg.processing.vlm_late_chunk_top_k_docs,
            enable_enhanced_chunking=rcfg.processing.enable_enhanced_chunking,
            chunk_type_filter=rcfg.processing.chunk_type_filter,
            enable_parent_expansion=state.enable_parent_expansion,
            parent_context_size=state.parent_context_size,
            include_sibling_chunks=rcfg.processing.include_sibling_chunks,
            sibling_window=rcfg.processing.sibling_window,
            include_parent_document=state.include_parent_document,
            parent_max_tokens=state.parent_max_tokens,
            enable_reranking=state.enable_reranking,
            reranking_strategy=state.reranking_strategy,
            rerank_top_k=rcfg.rerank.rerank_top_k,
            reranking_model=rcfg.rerank.reranking_model,
            rerank_min_relevance_prob=rcfg.rerank.rerank_min_relevance_prob,
            rerank_sentinel_margin=rcfg.rerank.rerank_sentinel_margin,
            enable_citations=rcfg.citations.enable_citations,
            citation_style=rcfg.citations.citation_style,
            include_page_numbers=rcfg.citations.include_page_numbers,
            enable_chunk_citations=rcfg.citations.enable_chunk_citations,
            enable_generation=bool(rcfg.verification.adaptive_rerun_include_generation),
            generation_model=rcfg.generation.generation_model,
            generation_provider=rcfg.generation.generation_provider,
            generation_prompt=state.generation_prompt,
            max_generation_tokens=rcfg.generation.max_generation_tokens,
            # Disable post-verification in rerun to avoid loops
            enable_post_verification=False,
            # Guard: mark this as an adaptive rerun
            _adaptive_rerun=True,
            # Preserve guardrails & claims defaults
            enable_injection_filter=rcfg.safety.enable_injection_filter,
            injection_filter_strength=rcfg.safety.injection_filter_strength,
            require_hard_citations=state.require_hard_citations,
            enable_numeric_fidelity=state.enable_numeric_fidelity,
            numeric_fidelity_behavior=state.numeric_fidelity_behavior,
            enable_claims=False,  # skip claims during rerun to save time
            index_namespace=rcfg.retrieval.index_namespace,
            user_id=rcfg.scope.user_id,
            session_id=rcfg.scope.session_id,
            enable_monitoring=rcfg.observability.enable_monitoring,
            enable_observability=rcfg.observability.enable_observability,
            trace_id=rcfg.observability.trace_id,
            credential_runtime=rcfg.runtime.credential_runtime,
            enable_performance_analysis=rcfg.observability.enable_performance_analysis,
            timeout_seconds=rcfg.resilience.timeout_seconds,
            highlight_results=rcfg.observability.highlight_results,
            highlight_query_terms=rcfg.observability.highlight_query_terms,
            track_cost=rcfg.observability.track_cost,
            debug_mode=rcfg.observability.debug_mode,
            include_rerank_debug_documents=rcfg.observability.include_rerank_debug_documents,
        )
        # Quick verify the new answer without repairs to compare factuality
        new_ratio = None
        if up.PostGenerationVerifier and (new_result.generated_answer or "").strip():
            v2 = await up.PostGenerationVerifier(
                max_retries=0,
                max_claims=min(10, rcfg.verification.adaptive_max_claims),
                credential_runtime=rcfg.runtime.credential_runtime,
            ).verify_and_maybe_fix(
                query=state.query,
                answer=new_result.generated_answer,
                base_documents=(
                    new_result.documents[: int(rcfg.verification.adaptive_rerun_doc_budget)]
                    if (rcfg.verification.adaptive_rerun_doc_budget and isinstance(rcfg.verification.adaptive_rerun_doc_budget, int))
                    else (new_result.documents or [])
                ),
                media_db_path=state.media_db_path,
                notes_db_path=rcfg.runtime.notes_db_path,
                character_db_path=rcfg.runtime.character_db_path,
                user_id=rcfg.scope.user_id,
                generation_model=rcfg.generation.generation_model,
                generation_provider=rcfg.generation.generation_provider,
                search_mode=rcfg.retrieval.search_mode,
                hybrid_alpha=state.hybrid_alpha,
                top_k=state.top_k,
            )
            new_ratio = v2.unsupported_ratio
        # Adoption decision with guardrails regression checks
        adopt = new_ratio is not None and new_ratio < state.vres.unsupported_ratio
        try:
            # Numeric fidelity regression check
            old_nf_missing = None
            new_nf_missing = None
            if up.check_numeric_fidelity and (state.result.generated_answer or "").strip():
                old_nf = up.check_numeric_fidelity(state.result.generated_answer, state.result.documents or [])
                if old_nf:
                    old_nf_missing = len(old_nf.missing)
            else:
                # fallback to existing metadata if available
                try:
                    old_nf_missing = (
                        len((state.result.metadata.get("numeric_fidelity") or {}).get("missing", []))
                        if isinstance(state.result.metadata, dict)
                        else None
                    )
                except (AttributeError, TypeError, ValueError):
                    old_nf_missing = None
            if up.check_numeric_fidelity and (new_result.generated_answer or "").strip():
                new_nf = up.check_numeric_fidelity(new_result.generated_answer, new_result.documents or [])
                if new_nf:
                    new_nf_missing = len(new_nf.missing)
            # Hard citation coverage regression check
            old_cov = None
            new_cov = None
            try:
                cov_raw = (
                    (state.result.metadata.get("hard_citations") or {}).get("coverage")
                    if isinstance(state.result.metadata, dict)
                    else None
                )
                old_cov = float(cov_raw) if cov_raw is not None else None
            except (TypeError, ValueError):
                old_cov = None
            if up.build_hard_citations and (new_result.generated_answer or "").strip():
                hc2 = up.build_hard_citations(
                    new_result.generated_answer, new_result.documents or [], claims_payload=None
                )
                if isinstance(hc2, dict):
                    new_cov = float(hc2.get("coverage") or 0.0)

            # If both NF counts present, block adoption on regression
            if adopt and (old_nf_missing is not None and new_nf_missing is not None):
                if new_nf_missing > old_nf_missing:
                    adopt = False
            # If both coverage present, block adoption on regression
            if adopt and (old_cov is not None and new_cov is not None):
                if new_cov < old_cov:
                    adopt = False
        except (AttributeError, TypeError, ValueError):
            # On checker failure, keep original adoption decision
            pass
        dur = time.time() - rerun_start
        state.result.metadata.setdefault("adaptive_rerun", {})
        state.result.metadata["adaptive_rerun"].update(
            {
                "performed": True,
                "duration": round(dur, 6),
                "old_ratio": state.vres.unsupported_ratio,
                "new_ratio": new_ratio,
                "adopted": bool(adopt),
                "bypass_cache": bool(rcfg.verification.adaptive_rerun_bypass_cache),
                "old_nf_missing": old_nf_missing if "old_nf_missing" in locals() else None,
                "new_nf_missing": new_nf_missing if "new_nf_missing" in locals() else None,
                "old_hard_citation_coverage": old_cov if "old_cov" in locals() else None,
                "new_hard_citation_coverage": new_cov if "new_cov" in locals() else None,
            }
        )
        # Metrics for rerun
        try:
            from tldw_Server_API.app.core.Metrics.metrics_manager import (
                increment_counter,
                observe_histogram,
            )

            increment_counter("rag_adaptive_rerun_performed_total", 1)
            if adopt:
                increment_counter("rag_adaptive_rerun_adopted_total", 1)
            observe_histogram(
                "rag_adaptive_rerun_duration_seconds",
                dur,
                labels={"adopted": "true" if adopt else "false"},
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        # Budget check and metric
        try:
            if rcfg.verification.adaptive_rerun_time_budget_sec is not None and dur > float(
                rcfg.verification.adaptive_rerun_time_budget_sec
            ):
                from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter

                increment_counter(
                    "rag_phase_budget_exhausted_total", 1, labels={"phase": "adaptive_rerun"}
                )
                state.result.metadata["adaptive_rerun"]["budget_exhausted"] = True
        except (ImportError, RuntimeError, TypeError, ValueError):
            pass
        if adopt:
            # Replace documents, citations and answer with rerun outputs
            state.result.documents = new_result.documents
            state.result.citations = new_result.citations
            state.result.metadata.update(dict((new_result.metadata or {}).items()))
            state.result.generated_answer = new_result.generated_answer


async def run_post_verification(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Post-generation verification with an optional adaptive rerun."""
    # ========== POST-GENERATION VERIFICATION (ADAPTIVE) ==========
    # May run even if enable_claims was False; uses existing results if available.
    try:
        # Allow env defaults if parameters not explicitly set
        if state.adaptive_time_budget_sec is None:
            try:
                import os as _os
                state.adaptive_time_budget_sec = float(_os.getenv("RAG_ADAPTIVE_TIME_BUDGET_SEC", "0")) or None
            except (TypeError, ValueError):
                state.adaptive_time_budget_sec = None
        if rcfg.verification.enable_post_verification and state.result.generated_answer and up.PostGenerationVerifier:
            await verify_answer(rcfg, state)

            # Optional: bounded full pipeline rerun to seek a better answer
            try:
                await adaptive_rerun(rcfg, state)
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as _er:
                if isinstance(_er, up._RAG_PROVIDER_FAILURES):
                    raise
                state.result.errors.append(f"Adaptive rerun failed: {str(_er)}")
                up.logger.debug(f"Adaptive rerun error: {_er}")
    except (
        AttributeError,
        ConnectionError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        asyncio.TimeoutError,
    ) as e:
        # Non-fatal: log and continue
        state.result.errors.append(f"Post-verification failed: {str(e)}")
        up.logger.warning(f"Post-verification error: {e}")
