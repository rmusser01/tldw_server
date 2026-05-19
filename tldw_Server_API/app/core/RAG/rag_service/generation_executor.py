from __future__ import annotations

from typing import Any, Awaitable, Callable

from .request_resolution import ResolvedRAGRequest
from .result_model import RAGResult
from .retrieval_plan import RetrievalPlan


async def execute_generation_phase(
    *,
    resolved_request: ResolvedRAGRequest,
    retrieval_plan: RetrievalPlan,
    derived_evidence: Any,
    generate_answer_fn: Callable[..., Awaitable[Any]],
    generation_context: str,
) -> RAGResult:
    del retrieval_plan  # Reserved for future plan-aware generation behavior.
    generation_query = resolved_request.query
    source_documents = derived_evidence.documents
    generation_prompt = (resolved_request.payload or {}).get("generation_prompt")

    answer_payload = await generate_answer_fn(
        query=generation_query,
        context=generation_context,
        generation_prompt=generation_prompt,
        max_generation_tokens=(resolved_request.payload or {}).get("max_generation_tokens"),
    )
    if isinstance(answer_payload, dict):
        generated_answer = answer_payload.get("answer")
        answer_metadata = dict(answer_payload.get("metadata") or {})
        for key, value in answer_payload.items():
            if key not in {"answer", "metadata"}:
                answer_metadata[key] = value
    else:
        generated_answer = str(answer_payload)
        answer_metadata = {}

    metadata = dict(derived_evidence.metadata)
    metadata.update(answer_metadata)

    return RAGResult(
        query=generation_query,
        documents=list(source_documents),
        generated_answer=generated_answer,
        metadata=metadata,
        chunk_citations=list(derived_evidence.citations),
        verification_report=derived_evidence.verification_report,
    )
