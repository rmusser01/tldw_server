"""Coordinate retrieved evidence with post-retrieval derived artifacts."""

from __future__ import annotations

from typing import Any, Sequence

from .evidence_models import DerivedEvidence, RetrievedEvidence
from .request_resolution import ResolvedRAGRequest
from .retrieval_plan import RetrievalPlan


class PostRetrievalCoordinator:
    """Own the boundary between retrieved evidence and derived evidence."""

    def derive_evidence(
        self,
        resolved_request: ResolvedRAGRequest,
        retrieved_evidence: RetrievedEvidence,
        *,
        enable_citations: bool,
        enable_verification: bool,
        derived_documents: Sequence[Any] | None = None,
        derived_from_document_ids: Sequence[str] | None = None,
    ) -> DerivedEvidence:
        del resolved_request  # Reserved for later request-aware coordination stages.

        metadata = dict(retrieved_evidence.metadata)
        documents = list(retrieved_evidence.documents)
        documents.extend(list(derived_documents or []))

        lineage = tuple(
            str(document_id).strip()
            for document_id in (derived_from_document_ids or [])
            if str(document_id).strip()
        )

        return DerivedEvidence(
            retrieved=retrieved_evidence,
            documents=documents,
            metadata=metadata,
            citations=list(metadata.get("chunk_citations", []) or []) if enable_citations else [],
            verification_report=metadata.get("verification_report") if enable_verification else None,
            derived_from_document_ids=lineage,
        )


def _metadata_from_coordinated_evidence(coordinated: DerivedEvidence) -> dict[str, Any]:
    updated_metadata = dict(coordinated.metadata)
    if coordinated.citations:
        updated_metadata["chunk_citations"] = list(coordinated.citations)
    if coordinated.verification_report is not None:
        updated_metadata["verification_report"] = coordinated.verification_report
    if coordinated.derived_from_document_ids:
        updated_metadata["derived_from_document_ids"] = list(
            coordinated.derived_from_document_ids
        )
    return updated_metadata


def coordinate_standard_result_evidence(
    result: Any,
    resolved_request: ResolvedRAGRequest,
    *,
    retrieval_plan: RetrievalPlan | None = None,
    coordinator: PostRetrievalCoordinator | None = None,
) -> Any:
    """Coordinate standard-path evidence without changing API result shape."""
    if isinstance(result, dict):
        result_metadata = dict(result.get("metadata") or {})
        result_documents = list(result.get("documents") or result.get("sources") or [])
    else:
        result_metadata = dict(getattr(result, "metadata", None) or {})
        result_documents = list(getattr(result, "documents", None) or [])

    if retrieval_plan is not None:
        result_metadata["retrieval_plan"] = {
            "query": retrieval_plan.query,
            "sources": list(retrieval_plan.sources),
            "search_mode": retrieval_plan.search_mode,
            "top_k": retrieval_plan.top_k,
            "index_namespace": retrieval_plan.index_namespace,
        }

    coordinator_instance = coordinator or PostRetrievalCoordinator()
    coordinated = coordinator_instance.derive_evidence(
        resolved_request,
        RetrievedEvidence(
            documents=result_documents,
            metadata=result_metadata,
        ),
        enable_citations=bool(result_metadata.get("chunk_citations")),
        enable_verification=bool(result_metadata.get("verification_report")),
        derived_documents=None,
        derived_from_document_ids=None,
    )
    if isinstance(result, dict):
        result["documents"] = list(coordinated.documents)
        result["metadata"] = _metadata_from_coordinated_evidence(coordinated)
        return result

    result.documents = list(coordinated.documents)
    result.metadata = _metadata_from_coordinated_evidence(coordinated)
    return result
