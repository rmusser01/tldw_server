"""Coordinate retrieved evidence with post-retrieval derived artifacts."""

from __future__ import annotations

from typing import Any, Sequence

from .evidence_models import DerivedEvidence, RetrievedEvidence


class PostRetrievalCoordinator:
    """Own the boundary between retrieved evidence and derived evidence."""

    def derive_evidence(
        self,
        resolved_request: Any,
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


def coordinate_standard_result_evidence(
    result: Any,
    resolved_request: Any,
    *,
    coordinator: PostRetrievalCoordinator | None = None,
) -> Any:
    """Coordinate standard-path evidence without changing API result shape."""
    result_metadata = dict(getattr(result, "metadata", None) or {})
    coordinator_instance = coordinator or PostRetrievalCoordinator()
    coordinated = coordinator_instance.derive_evidence(
        resolved_request,
        RetrievedEvidence(
            documents=list(getattr(result, "documents", None) or []),
            metadata=result_metadata,
        ),
        enable_citations=bool(result_metadata.get("chunk_citations")),
        enable_verification=bool(result_metadata.get("verification_report")),
        derived_documents=None,
        derived_from_document_ids=None,
    )
    result.documents = list(coordinated.documents)
    updated_metadata = dict(coordinated.metadata)
    if coordinated.citations:
        updated_metadata["chunk_citations"] = list(coordinated.citations)
    if coordinated.verification_report is not None:
        updated_metadata["verification_report"] = coordinated.verification_report
    if coordinated.derived_from_document_ids:
        updated_metadata["derived_from_document_ids"] = list(
            coordinated.derived_from_document_ids
        )
    result.metadata = updated_metadata
    return result
