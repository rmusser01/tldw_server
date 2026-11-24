"""self_correct code moved from unified_pipeline.py (TASK-13950.24)."""

import asyncio
import time
from typing import TYPE_CHECKING

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


async def grade_documents(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Self-correcting RAG stage 1: grade documents for relevance."""
    if rcfg.self_correct.enable_document_grading and state.result.documents and up.DocumentGrader:
        grading_start = time.time()
        try:
            grading_config = up.GradingConfig(
                provider=rcfg.self_correct.grading_provider or "openai",
                model=rcfg.self_correct.grading_model,
                batch_size=rcfg.self_correct.grading_batch_size,
                timeout_seconds=rcfg.self_correct.grading_timeout_sec,
                fallback_to_score=rcfg.self_correct.grading_fallback_to_score,
                fallback_min_score=rcfg.self_correct.grading_fallback_min_score,
            )
            grader = up.DocumentGrader(
                config=grading_config,
                credential_runtime=rcfg.runtime.credential_runtime,
            )

            filtered_docs, grading_metadata = await grader.filter_relevant(
                query=state.query,
                documents=state.result.documents,
                threshold=rcfg.self_correct.grading_threshold,
                provider=rcfg.self_correct.grading_provider,
                model=rcfg.self_correct.grading_model,
            )

            # Store grading metadata
            state.result.metadata["document_grading"] = {
                "enabled": True,
                "threshold": rcfg.self_correct.grading_threshold,
                "total_graded": grading_metadata.get("total_graded", 0),
                "relevant_count": grading_metadata.get("relevant_count", 0),
                "filtered_count": grading_metadata.get("filtered_count", 0),
                "removed_count": grading_metadata.get("removed_count", 0),
                "avg_relevance": grading_metadata.get("avg_relevance", 0.0),
                "grading_latency_ms": grading_metadata.get("total_latency_ms", 0),
            }
            if grading_metadata.get("verification_available") is False:
                state.result.metadata["document_grading"].update(
                    {
                        "failure_code": grading_metadata.get(
                            "failure_code",
                            "provider_unavailable",
                        ),
                        "verification_available": False,
                    }
                )

            # Check if we should trigger query rewriting loop (Stage 2)
            avg_relevance = grading_metadata.get("avg_relevance", 0.0)
            if rcfg.self_correct.enable_query_rewriting_loop and avg_relevance < rcfg.self_correct.rewrite_relevance_threshold:
                state.result.metadata["document_grading"]["low_relevance_detected"] = True
                state.result.metadata["document_grading"]["will_rewrite"] = True

            # Update documents with filtered set
            if filtered_docs:
                state.result.documents = filtered_docs
            else:
                # If no documents passed grading, keep original for fallback
                state.result.metadata["document_grading"]["fallback_to_original"] = True
                up.logger.warning(f"No documents passed grading threshold {rcfg.self_correct.grading_threshold}, keeping originals")

            state.result.timings["document_grading"] = time.time() - grading_start

            # --- OTEL span for document grading ---
            if rcfg.observability.enable_observability and up.get_telemetry_manager:
                try:
                    _tm = up.get_telemetry_manager()
                    _tr = _tm.get_tracer("tldw.rag")
                    with _tr.start_as_current_span("rag.document_grading") as _span:
                        _span.set_attribute("rag.phase", "document_grading")
                        _span.set_attribute("rag.grading_threshold", rcfg.self_correct.grading_threshold)
                        _span.set_attribute("rag.docs_graded", grading_metadata.get("total_graded", 0))
                        _span.set_attribute("rag.docs_passed", grading_metadata.get("filtered_count", 0))
                        _span.set_attribute("rag.avg_relevance", avg_relevance)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass

        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Document grading failed: {e}")
            up.logger.warning(f"Document grading error: {e}")
            state.result.metadata["document_grading"] = {"enabled": True, "error": str(e)}


async def run_rewrite_loop(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Self-correcting RAG stage 2: rewrite the query and retry retrieval on low relevance."""
    if (
        rcfg.self_correct.enable_query_rewriting_loop
        and up.QueryRewriter
        and state.result.metadata.get("document_grading", {}).get("low_relevance_detected")
    ):
        rewrite_loop_start = time.time()
        rewrite_attempts = []
        best_relevance = state.result.metadata.get("document_grading", {}).get("avg_relevance", 0.0)
        best_docs = state.result.documents
        current_query = state.query

        try:
            rewriter = up.QueryRewriter()

            for attempt in range(rcfg.self_correct.max_rewrite_attempts):
                # Rewrite the query using improve_for_retrieval strategy
                rewrites = rewriter.rewrite_query(
                    current_query,
                    strategies=["improve_for_retrieval"],
                    failed_docs=state.result.documents,
                    failure_reason=f"avg_relevance_{best_relevance:.2f}",
                )

                if not rewrites:
                    up.logger.debug(f"No rewrites generated on attempt {attempt + 1}")
                    break

                # Use the best rewrite (highest confidence)
                best_rewrite = max(rewrites, key=lambda r: r.confidence)
                rewritten_query = best_rewrite.rewritten_query

                rewrite_attempts.append(
                    {
                        "attempt": attempt + 1,
                        "original_query": current_query,
                        "rewritten_query": rewritten_query,
                        "rewrite_type": best_rewrite.rewrite_type,
                        "confidence": best_rewrite.confidence,
                        "explanation": best_rewrite.explanation,
                    }
                )

                up.logger.debug(f"Query rewrite attempt {attempt + 1}: '{rewritten_query}'")

                # Re-run retrieval with rewritten query
                if up.MultiDatabaseRetriever and up.RetrievalConfig:
                    latched_code = state.latched_optional_provider_failure("query_rewrite")
                    if latched_code is not None:
                        rewrite_attempts[-1].update(
                            retrieval_coverage="degraded",
                            failure_code=latched_code,
                        )
                        break
                    try:
                        db_paths = state.build_pipeline_db_paths()

                        state.retriever = state.build_multi_retriever(db_paths)
                        retrieval_config = up.RetrievalConfig(
                            max_results=state.retrieval_top_k,
                            min_score=state.retrieval_min_score,
                            use_fts=(state.retrieval_search_mode in ["fts", "hybrid"]),
                            use_vector=(state.retrieval_search_mode in ["vector", "hybrid"]),
                            include_metadata=True,
                            fts_level=state.fts_level,
                            enable_text_late_chunking=rcfg.processing.enable_text_late_chunking,
                            chunk_method=rcfg.processing.chunk_method,
                            chunk_size=rcfg.processing.chunk_size,
                            chunk_overlap=rcfg.processing.chunk_overlap,
                            chunk_language=rcfg.processing.chunk_language,
                        )
                        data_sources = list(state.resolved_data_sources)

                        new_docs = await state.retriever.retrieve(
                            query=rewritten_query,
                            sources=data_sources,
                            config=retrieval_config,
                            source_failures=state.source_failures,
                            index_namespace=state.retrieval_index_namespace,
                            retrieval_plan=state.retrieval_plan_for(rewritten_query),
                        )

                        if new_docs:
                            # Re-grade the new documents
                            if rcfg.self_correct.enable_document_grading and up.DocumentGrader:
                                grading_config = up.GradingConfig(
                                    provider=rcfg.self_correct.grading_provider or "openai",
                                    model=rcfg.self_correct.grading_model,
                                    batch_size=rcfg.self_correct.grading_batch_size,
                                    timeout_seconds=rcfg.self_correct.grading_timeout_sec,
                                    fallback_to_score=rcfg.self_correct.grading_fallback_to_score,
                                    fallback_min_score=rcfg.self_correct.grading_fallback_min_score,
                                )
                                grader = up.DocumentGrader(
                                    config=grading_config,
                                    credential_runtime=rcfg.runtime.credential_runtime,
                                )
                                _, new_grading_metadata = await grader.filter_relevant(
                                    query=rewritten_query,
                                    documents=new_docs,
                                    threshold=rcfg.self_correct.grading_threshold,
                                )

                                new_avg_relevance = new_grading_metadata.get("avg_relevance", 0.0)
                                rewrite_attempts[-1]["new_avg_relevance"] = new_avg_relevance

                                # Check if we've improved
                                if new_avg_relevance > best_relevance:
                                    best_relevance = new_avg_relevance
                                    best_docs = new_docs
                                    current_query = rewritten_query

                                # Check if we've exceeded threshold
                                if new_avg_relevance >= rcfg.self_correct.rewrite_relevance_threshold:
                                    up.logger.info(
                                        f"Query rewrite succeeded after {attempt + 1} attempts, relevance: {new_avg_relevance:.2f}"
                                    )
                                    rewrite_attempts[-1]["success"] = True
                                    break
                            else:
                                # No grading, just use new docs
                                best_docs = new_docs
                                current_query = rewritten_query
                                break

                    except up._RAG_PROVIDER_FAILURES as ret_err:
                        if rcfg.runtime.credential_runtime is None:
                            raise
                        failure_code = state.latch_optional_provider_failure(
                            "query_rewrite",
                            ret_err,
                        )
                        rewrite_attempts[-1].update(
                            retrieval_coverage="degraded",
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
                    ) as ret_err:
                        up.logger.warning(f"Retrieval with rewritten query failed: {ret_err}")
                        rewrite_attempts[-1]["error"] = str(ret_err)

            # Update result with best documents found
            if best_docs:
                state.result.documents = best_docs

            # Store query rewriting loop metadata
            state.result.metadata["query_rewrite_loop"] = {
                "enabled": True,
                "attempts": len(rewrite_attempts),
                "max_attempts": rcfg.self_correct.max_rewrite_attempts,
                "initial_relevance": state.result.metadata.get("document_grading", {}).get("avg_relevance", 0.0),
                "final_relevance": best_relevance,
                "threshold": rcfg.self_correct.rewrite_relevance_threshold,
                "improved": best_relevance > state.result.metadata.get("document_grading", {}).get("avg_relevance", 0.0),
                "rewrite_attempts": rewrite_attempts,
            }

            state.result.timings["query_rewrite_loop"] = time.time() - rewrite_loop_start

        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Query rewriting loop failed: {e}")
            up.logger.warning(f"Query rewriting loop error: {e}")
            state.result.metadata["query_rewrite_loop"] = {"enabled": True, "error": str(e)}
