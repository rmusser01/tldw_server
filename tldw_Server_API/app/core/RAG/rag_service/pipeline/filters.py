"""filters code moved from unified_pipeline.py (TASK-13950.24)."""

import asyncio
import inspect
import time
from collections import Counter
from typing import TYPE_CHECKING

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


def apply_content_filters(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Instruction-injection filtering, the chunk-type filter and content policy sanitation."""
    if rcfg.safety.enable_injection_filter and state.result.documents:
        inj_start = time.time()
        try:
            if up.downweight_injection_docs:
                summary = up.downweight_injection_docs(
                    state.result.documents, strength=float(rcfg.safety.injection_filter_strength or 0.5)
                )
                state.result.metadata.setdefault("injection_filter", {})
                state.result.metadata["injection_filter"].update(
                    {
                        "affected": int(summary.get("affected", 0)),
                        "total": int(summary.get("total", len(state.result.documents))),
                        "strength": float(rcfg.safety.injection_filter_strength or 0.5),
                    }
                )
                # Optional metric
                try:
                    from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter

                    if int(summary.get("affected", 0)) > 0:
                        increment_counter(
                            "rag_injection_chunks_downweighted_total", int(summary.get("affected", 0))
                        )
                except (ImportError, RuntimeError, TypeError, ValueError):
                    pass
            else:
                state.result.errors.append("Injection filter module not available")
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Injection filtering failed: {str(e)}")
        finally:
            state.result.timings["injection_filter"] = time.time() - inj_start

    # ========== OPTIONAL CHUNK TYPE FILTER (metadata-based) ==========
    if rcfg.processing.chunk_type_filter and state.result.documents:
        try:
            allowed: set[str] = set()
            for t in rcfg.processing.chunk_type_filter:
                norm = up._normalize_chunk_type_value(t)
                if norm:
                    allowed.add(norm)
            before_count = len(state.result.documents)
            filtered_docs = []
            for d in state.result.documents:
                # Whole text records (notes, chats, cards) have no chunk
                # annotation. They remain text, including in the UI default
                # filter; an explicitly typed chunk still keeps its type.
                raw_type = (d.metadata or {}).get("chunk_type")
                doc_type = up._normalize_chunk_type_value(raw_type if raw_type is not None else "text")
                if doc_type and doc_type in allowed:
                    filtered_docs.append(d)
            state.result.documents = filtered_docs
            state.result.metadata["chunk_type_filter_before"] = before_count
            state.result.metadata["chunk_type_filter_after"] = len(state.result.documents)
        except (AttributeError, TypeError, ValueError):
            pass

    # ========== CONTENT POLICY FILTERS & SANITATION ==========
    if state.result.documents:
        try:
            # OCR gating
            if rcfg.safety.ocr_confidence_threshold is not None:
                try:
                    from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter

                    dropped = up.gate_docs_by_ocr_confidence(state.result.documents, float(rcfg.safety.ocr_confidence_threshold))
                    if dropped > 0:
                        increment_counter("rag_ocr_dropped_docs_total", dropped)
                except (ImportError, RuntimeError, TypeError, ValueError):
                    pass
            # HTML sanitation
            if rcfg.safety.enable_html_sanitizer:
                sanitized = 0
                for d in state.result.documents or []:
                    try:
                        before = d.content or ""
                        after = up.sanitize_html_allowlist(before, rcfg.safety.html_allowed_tags, rcfg.safety.html_allowed_attrs)
                        if after != before:
                            d.content = after
                            sanitized += 1
                    except (AttributeError, TypeError, ValueError):
                        continue
                try:
                    if sanitized > 0:
                        from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter

                        increment_counter("rag_sanitized_docs_total", sanitized)
                except (ImportError, RuntimeError, TypeError, ValueError):
                    pass
            # Content policy (PII/PHI)
            if rcfg.safety.enable_content_policy_filter:
                summary = up.apply_content_policy(
                    state.result.documents,
                    policy_types=(rcfg.safety.content_policy_types or ["pii"]),
                    mode=str(rcfg.safety.content_policy_mode or "redact"),
                )
                state.result.metadata.setdefault("content_policy", {})
                state.result.metadata["content_policy"].update(
                    {
                        "enabled": True,
                        "types": rcfg.safety.content_policy_types or ["pii"],
                        "mode": rcfg.safety.content_policy_mode,
                        "affected": int(summary.get("affected", 0)),
                        "dropped": int(summary.get("dropped", 0)),
                    }
                )
                try:
                    from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter

                    if int(summary.get("affected", 0)) > 0:
                        increment_counter(
                            "rag_policy_filtered_chunks_total",
                            int(summary.get("affected", 0)),
                            labels={"mode": str(rcfg.safety.content_policy_mode or "redact")},
                        )
                except (ImportError, RuntimeError, TypeError, ValueError):
                    pass
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Non-fatal: continue
            pass


async def apply_security_filter(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Security filtering by sensitivity level, recording its own failures."""
    if rcfg.safety.enable_security_filter and state.result.documents:
        security_start = time.time()
        incoming_source_counts = Counter(up._document_canonical_source(doc) for doc in state.result.documents)
        try:
            if up.SecurityFilter and up.SensitivityLevel:
                security_filter = up.SecurityFilter()
                sensitivity_map = {
                    "public": up.SensitivityLevel.PUBLIC,
                    "internal": up.SensitivityLevel.INTERNAL,
                    "confidential": up.SensitivityLevel.CONFIDENTIAL,
                    "restricted": up.SensitivityLevel.RESTRICTED,
                }
                sensitivity_value = sensitivity_map.get(rcfg.safety.sensitivity_level)
                if sensitivity_value is None:
                    valid_levels = ", ".join(sensitivity_map.keys())
                    raise ValueError(
                        f"Invalid sensitivity_level '{rcfg.safety.sensitivity_level}'. Expected one of: {valid_levels}"
                    )

                # Detect PII if requested
                if rcfg.safety.detect_pii:
                    detect_pii_batch = getattr(security_filter, "detect_pii_batch", None)
                    if callable(detect_pii_batch):
                        pii_report = detect_pii_batch([doc.content for doc in state.result.documents])
                        if inspect.isawaitable(pii_report):
                            pii_report = await pii_report
                    else:
                        detector = getattr(security_filter, "pii_detector", None)
                        detect_single = getattr(detector, "detect_pii", None) if detector is not None else None
                        if callable(detect_single):
                            pii_report = []
                            for doc in state.result.documents:
                                matches = detect_single(doc.content)
                                if inspect.isawaitable(matches):
                                    matches = await matches
                                pii_report.append(
                                    [
                                        match.to_dict() if hasattr(match, "to_dict") else match
                                        for match in (matches or [])
                                    ]
                                )
                        else:
                            pii_report = []
                    state.result.security_report = {"pii_detected": pii_report}

                sensitivity_filtered_docs = None
                filter_by_sensitivity = getattr(security_filter, "filter_by_sensitivity", None)
                if callable(filter_by_sensitivity):
                    sensitivity_filtered_docs = filter_by_sensitivity(
                        state.result.documents,
                        max_level=sensitivity_value,
                    )
                    if inspect.isawaitable(sensitivity_filtered_docs):
                        sensitivity_filtered_docs = await sensitivity_filtered_docs
                    sensitivity_filtered_docs = up._coerce_security_filter_sequence(
                        sensitivity_filtered_docs,
                        description="document sequence",
                    )

                    if rcfg.safety.redact_pii:
                        redact_pii_fn = getattr(security_filter, "redact_pii", None)
                        if callable(redact_pii_fn):
                            for doc in sensitivity_filtered_docs:
                                redacted = redact_pii_fn(doc.content)
                                doc.content = await redacted if inspect.isawaitable(redacted) else redacted
                else:
                    filter_documents = getattr(security_filter, "filter_documents", None)
                    if callable(filter_documents):
                        serialized_docs = [
                            {
                                "id": doc.id,
                                "content": doc.content,
                                "metadata": dict(doc.metadata or {}),
                                "_document_ref": doc,
                            }
                            for doc in state.result.documents
                        ]
                        filtered_payloads = filter_documents(
                            serialized_docs,
                            user_id=up._resolve_security_user_id(rcfg.scope.user_id, rcfg.scope.feedback_user_id),
                            max_sensitivity=sensitivity_value,
                            mask_pii=bool(rcfg.safety.redact_pii),
                        )
                        if inspect.isawaitable(filtered_payloads):
                            filtered_payloads = await filtered_payloads

                        sensitivity_filtered_docs = []
                        for payload in up._coerce_security_filter_sequence(
                            filtered_payloads,
                            description="payload sequence",
                        ):
                            if not isinstance(payload, dict):
                                up.logger.warning(
                                    "Security filter payload entry had unsupported type {}; skipping entry",
                                    type(payload).__name__,
                                )
                                continue
                            doc_ref = payload.get("_document_ref")
                            if doc_ref is None:
                                continue
                            doc_ref.content = payload.get("content", doc_ref.content)
                            merged_metadata = dict(doc_ref.metadata or {})
                            payload_metadata = payload.get("metadata")
                            if isinstance(payload_metadata, dict):
                                merged_metadata.update(payload_metadata)
                            if "pii_masked" in payload:
                                merged_metadata["pii_masked"] = payload["pii_masked"]
                            if "pii_types" in payload:
                                merged_metadata["pii_types"] = payload["pii_types"]
                            doc_ref.metadata = merged_metadata
                            sensitivity_filtered_docs.append(doc_ref)

                if sensitivity_filtered_docs is not None:
                    state.security_excluded_counts.update(
                        incoming_source_counts
                        - Counter(up._document_canonical_source(doc) for doc in sensitivity_filtered_docs)
                    )
                    # Expose only aggregate outcomes, never excluded evidence.
                    state.result.metadata["security_filter"] = {
                        "excluded_count": max(0, len(state.result.documents) - len(sensitivity_filtered_docs)),
                        "retained_count": len(sensitivity_filtered_docs),
                    }
                    state.result.documents = sensitivity_filtered_docs
                state.result.timings["security_filter"] = time.time() - security_start

        except ImportError:
            state.result.errors.append("Security filter module not available")
            up.logger.warning("Security filter requested but module not available")
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            state.result.errors.append(f"Security filter failed: {str(e)}")
            up.logger.error(f"Security filter error: {e}")
