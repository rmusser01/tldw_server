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
