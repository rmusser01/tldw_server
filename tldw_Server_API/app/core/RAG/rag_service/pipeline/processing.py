"""processing code moved from unified_pipeline.py (TASK-13950.24)."""

import asyncio
import inspect
import time
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


async def process_tables_and_vlm(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Table processing and VLM late chunking of the retrieved documents."""
    if rcfg.processing.enable_table_processing and state.result.documents:
        table_start = time.time()
        try:
            if up.TableProcessor:
                processor = up.TableProcessor()
                processed_docs = []

                for doc in state.result.documents:
                    process_document = getattr(processor, "process_document", None)
                    process_document_tables = getattr(processor, "process_document_tables", None)

                    if callable(process_document):
                        processed = process_document(doc.content, method=rcfg.processing.table_method)
                        processed = await processed if inspect.isawaitable(processed) else processed
                        doc.content = processed
                    elif callable(process_document_tables):
                        processed = process_document_tables(doc.content, serialize_method=rcfg.processing.table_method)
                        processed = await processed if inspect.isawaitable(processed) else processed
                        if isinstance(processed, tuple):
                            processed_text, table_metadata = processed
                            doc.content = processed_text
                            if table_metadata:
                                merged_metadata = dict(doc.metadata or {})
                                merged_metadata["table_metadata"] = table_metadata
                                doc.metadata = merged_metadata
                        else:
                            doc.content = processed
                    processed_docs.append(doc)

                state.result.documents = processed_docs
                state.result.timings["table_processing"] = time.time() - table_start

        except ImportError:
            state.result.errors.append("Table processing module not available")
            up.logger.warning("Table processing requested but module not available")

    # ========== VLM LATE CHUNKING (Optional) ==========
    if rcfg.processing.enable_vlm_late_chunking and state.result.documents:
        vlm_start = time.time()
        try:
            try:
                from tldw_Server_API.app.core.Ingestion_Media_Processing.VLM.registry import (
                    get_backend as _get_vlm_backend,
                )
            except ImportError:

                def _unavailable_vlm_backend(name: str | None = None) -> None:
                    return None

                _get_vlm_backend = _unavailable_vlm_backend

            # Pick backend
            backend = _get_vlm_backend(rcfg.processing.vlm_backend if rcfg.processing.vlm_backend not in (None, "auto") else None)
            if backend is None:
                state.result.errors.append("VLM requested but no backend available")
            else:
                # Operate on top-k documents to bound cost
                # Allow media_db and notes_db sources when a local PDF path is present
                allowed_sources = {"media_db", "notes_db"}
                selected_docs = [
                    d
                    for d in state.result.documents
                    if (d.metadata or {}).get("source") in allowed_sources and (d.metadata or {}).get("url")
                ]
                selected_docs = selected_docs[: max(1, int(rcfg.processing.vlm_late_chunk_top_k_docs or 1))]

                vlm_added: list[up.Document] = []
                for doc in selected_docs:
                    url = (doc.metadata or {}).get("url")
                    page_limit = rcfg.processing.vlm_max_pages
                    if not url:
                        continue
                    # Resolve PDF path: strictly require local file path (no remote URLs)
                    pdf_path = None
                    cleanup_tmp = False  # noqa: F841 - dead store moved from unified_pipeline.py (TASK-13950.24)
                    try:
                        from pathlib import Path

                        pdf_path_obj = Path(str(url))
                        if pdf_path_obj.exists() and pdf_path_obj.suffix.lower() == ".pdf":
                            pdf_path = str(pdf_path_obj)
                        else:
                            # Unsupported: not a local PDF path
                            continue
                    except (OSError, TypeError, ValueError):
                        continue

                    # Use document-level VLM when available
                    try:
                        detections = []
                        if hasattr(backend, "process_pdf"):
                            res = backend.process_pdf(pdf_path, max_pages=page_limit)
                            by_page: list[dict[str, Any]] = []
                            if isinstance(getattr(res, "extra", None), dict):
                                by_page = res.extra.get("by_page") or []
                            if by_page:
                                for entry in by_page:
                                    page_no = entry.get("page")
                                    for d in entry.get("detections") or []:
                                        label = str(d.get("label"))
                                        if rcfg.processing.vlm_detect_tables_only and label.lower() != "table":
                                            continue
                                        detections.append(
                                            {
                                                "label": label,
                                                "score": float(d.get("score", 0.0)),
                                                "bbox": d.get("bbox") or [0.0, 0.0, 0.0, 0.0],
                                                "page": page_no,
                                            }
                                        )
                        else:
                            # Per-page image mode
                            try:
                                import pymupdf

                                with pymupdf.open(pdf_path) as _doc:
                                    total_pages = len(_doc)
                                    max_pages = min(page_limit or total_pages, total_pages)
                                    for i, page in enumerate(_doc, start=1):
                                        if i > max_pages:
                                            break
                                        pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0), alpha=False)
                                        img_bytes = pix.tobytes("png")
                                        res = backend.process_image(
                                            img_bytes, context={"page": i, "pdf_path": pdf_path}
                                        )
                                        for det in getattr(res, "detections", []) or []:
                                            label = str(getattr(det, "label", ""))
                                            if rcfg.processing.vlm_detect_tables_only and label.lower() != "table":
                                                continue
                                            detections.append(
                                                {
                                                    "label": label,
                                                    "score": float(getattr(det, "score", 0.0)),
                                                    "bbox": list(getattr(det, "bbox", [0.0, 0.0, 0.0, 0.0])),
                                                    "page": i,
                                                }
                                            )
                            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                                continue

                        # Convert detections into lightweight Documents for reranking/search
                        for idx, d in enumerate(detections[:100]):  # bound new docs per source
                            label = str(d.get("label", "vlm"))
                            score = d.get("score", 0.0)
                            bbox = d.get("bbox")
                            page_no = d.get("page")
                            chunk_text = f"Detected {label} ({score:.2f}) on page {page_no} at {bbox}"
                            vlm_added.append(
                                up.Document(
                                    id=f"vlm:{doc.id}:{idx}",
                                    content=chunk_text,
                                    source=doc.source,
                                    metadata={
                                        **(doc.metadata or {}),
                                        "chunk_type": ("table" if str(label).lower() == "table" else "vlm"),
                                        "page": page_no,
                                        "bbox": bbox,
                                        "derived_from": doc.id,
                                    },
                                    score=float(getattr(doc, "score", 0.0)),
                                )
                            )
                    finally:
                        # No temp cleanup needed; remote URLs are not supported
                        pass
                if vlm_added:
                    # Extend document list for downstream processing/reranking
                    state.result.documents.extend(vlm_added)
            state.result.timings["vlm_late_chunking"] = time.time() - vlm_start
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"VLM late-chunking failed: {e}")
            up.logger.warning(f"VLM late-chunking failed: {e}")


async def run_evidence_accumulation(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Progressive evidence accumulation over the retrieved documents."""
    if rcfg.retrieval.enable_evidence_accumulation and state.result.documents and up.EvidenceAccumulator:
        accumulation_start = time.time()
        try:
            accumulator = up.EvidenceAccumulator(
                max_rounds=rcfg.retrieval.accumulation_max_rounds,
                enable_gap_assessment=True,
                **(
                    {"credential_runtime": rcfg.runtime.credential_runtime}
                    if rcfg.runtime.credential_runtime is not None
                    else {}
                ),
            )

            # Create retrieval function for additional rounds
            async def _additional_retrieval(gap_query: str, exclude_ids: set):
                if state.latched_optional_provider_failure("evidence_accumulation") is not None:
                    return []
                if not (up.MultiDatabaseRetriever and up.RetrievalConfig):
                    return []
                try:
                    # Reuse the same retriever setup
                    db_paths = state.build_pipeline_db_paths()

                    retriever = state.build_multi_retriever(db_paths)
                    config = up.RetrievalConfig(
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
                    if not data_sources:
                        return []
                    new_docs = await retriever.retrieve(
                        query=gap_query,
                        sources=data_sources,
                        config=config,
                        source_failures=state.source_failures,
                        index_namespace=state.retrieval_index_namespace,
                        retrieval_plan=state.retrieval_plan_for(gap_query),
                        allowed_media_ids=rcfg.scope.include_media_ids,
                        allowed_note_ids=rcfg.scope.include_note_ids,
                    )
                    # Filter out already-seen documents
                    return [d for d in new_docs if d.id not in exclude_ids]
                except up._RAG_PROVIDER_FAILURES as exc:
                    if rcfg.runtime.credential_runtime is None:
                        raise
                    state.latch_optional_provider_failure("evidence_accumulation", exc)
                    return []
                except (
                    AttributeError,
                    ConnectionError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    asyncio.TimeoutError,
                ) as e:
                    up.logger.warning(f"Additional retrieval failed: {e}")
                    return []

            accumulation_result = await accumulator.accumulate(
                query=state.query,
                initial_results=state.result.documents,
                retrieval_fn=_additional_retrieval,
                time_budget_sec=rcfg.retrieval.accumulation_time_budget_sec,
            )

            # Update documents with accumulated results
            state.result.documents = accumulation_result.documents
            state.result.metadata["evidence_accumulation"] = {
                "total_rounds": accumulation_result.total_rounds,
                "is_sufficient": accumulation_result.is_sufficient,
                "sufficiency_reason": accumulation_result.sufficiency_reason,
                "initial_docs": accumulation_result.metadata.get("initial_docs", 0),
                "final_docs": len(accumulation_result.documents),
                "docs_added": accumulation_result.metadata.get("docs_added", 0),
                **(
                    {
                        key: accumulation_result.metadata[key]
                        for key in ("verification_available", "failure_code")
                        if key in accumulation_result.metadata
                    }
                ),
            }
            state.result.timings["evidence_accumulation"] = time.time() - accumulation_start

        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Evidence accumulation failed: {e}")
            up.logger.warning(f"Evidence accumulation error: {e}")
