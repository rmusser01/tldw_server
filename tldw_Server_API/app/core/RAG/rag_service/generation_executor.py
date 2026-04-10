from __future__ import annotations

from typing import Any, Awaitable, Callable

from .result_model import RAGResult


async def execute_generation_phase(
    *,
    resolved_request: Any,
    derived_evidence: Any,
    generate_answer_fn: Callable[..., Awaitable[Any]],
) -> RAGResult:
    answer_payload = await generate_answer_fn(
        query=resolved_request.query,
        documents=list(derived_evidence.documents),
        generation_prompt=(resolved_request.payload or {}).get("generation_prompt"),
        max_generation_tokens=(resolved_request.payload or {}).get("max_generation_tokens"),
    )
    if isinstance(answer_payload, dict):
        generated_answer = answer_payload.get("answer")
        answer_metadata = dict(answer_payload.get("metadata") or {})
    else:
        generated_answer = str(answer_payload)
        answer_metadata = {}

    metadata = dict(derived_evidence.metadata)
    metadata.update(answer_metadata)

    return RAGResult(
        query=resolved_request.query,
        documents=list(derived_evidence.documents),
        generated_answer=generated_answer,
        metadata=metadata,
        chunk_citations=list(derived_evidence.citations),
        verification_report=derived_evidence.verification_report,
    )
