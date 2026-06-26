from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from tldw_Server_API.app.core.RAG.rag_service.evidence_models import RetrievedEvidence
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_executor import execute_retrieval_phase
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RPG.errors import RPGValidationError
from tldw_Server_API.app.core.RPG.rules.content_packs import RuleLookupCitation, RuleLookupItem
from tldw_Server_API.app.core.RPG.rules.refs import (
    RulesPackRef,
    RulesPackSourceValidation,
    RulesPackSourceValidator,
)

MAX_RULE_SNIPPET_CHARS = 1500

RetrievalExecutor = Callable[..., Awaitable[RetrievedEvidence]]


@dataclass(frozen=True, slots=True)
class RulesRetrievalResult:
    items: list[RuleLookupItem]
    ready_media_ids: list[int]
    skipped_refs: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class RulesRetriever(Protocol):
    async def retrieve(
        self,
        *,
        owner_user_id: int,
        query: str,
        refs: list[RulesPackRef],
        max_results: int,
    ) -> RulesRetrievalResult:
        ...


class RulesRetrievalAdapter:
    def __init__(
        self,
        *,
        source_validator: RulesPackSourceValidator,
        rag_retriever: Any,
        retrieval_executor: RetrievalExecutor = execute_retrieval_phase,
    ) -> None:
        self._source_validator = source_validator
        self._rag_retriever = rag_retriever
        self._retrieval_executor = retrieval_executor

    async def retrieve(
        self,
        *,
        owner_user_id: int,
        query: str,
        refs: list[RulesPackRef],
        max_results: int,
    ) -> RulesRetrievalResult:
        enabled_refs = [ref for ref in refs if ref.enabled]
        skipped_refs = [{"ref_id": ref.ref_id, "reason": "disabled"} for ref in refs if not ref.enabled]
        ready_media_ids: list[int] = []
        seen_media_ids: set[int] = set()

        for ref in enabled_refs:
            validation = await self._validate_ref(owner_user_id=owner_user_id, ref=ref)
            if not validation.readable:
                raise RPGValidationError("rules_pack_source_unreadable")
            ref_ready_ids = [media_id for media_id in validation.ready_media_ids if isinstance(media_id, int)]
            if not ref_ready_ids:
                skipped_refs.append({"ref_id": ref.ref_id, "reason": "no_ready_media"})
                continue
            for media_id in ref_ready_ids:
                if media_id <= 0 or media_id in seen_media_ids:
                    continue
                seen_media_ids.add(media_id)
                ready_media_ids.append(media_id)

        base_diagnostics = {
            "enabled_rules_pack_count": len(enabled_refs),
            "ready_media_item_count": len(ready_media_ids),
            "skipped_refs": skipped_refs,
            "broad_fallback_used": False,
        }
        if not ready_media_ids:
            return RulesRetrievalResult(
                items=[],
                ready_media_ids=[],
                skipped_refs=skipped_refs,
                diagnostics={**base_diagnostics, "retrieval_result_count": 0, "no_ready_sources": True},
            )

        max_results = max(1, int(max_results))
        resolved_request = ResolvedRAGRequest(
            query=query,
            strategy="standard",
            payload={"sources": ["media_db"], "search_mode": "hybrid", "top_k": max_results},
            index_namespace=None,
            rag_profile=None,
            user_id=str(owner_user_id),
            feedback_user_id=str(owner_user_id),
        )
        retrieval_plan = RetrievalPlan(
            query=query,
            sources=("media_db",),
            search_mode="hybrid",
            top_k=max_results,
            min_score=0.0,
            index_namespace=None,
            collection_names={"media_db": f"user_{owner_user_id}_media_embeddings"},
        )
        evidence = await self._retrieval_executor(
            resolved_request=resolved_request,
            retrieval_plan=retrieval_plan,
            retriever=self._rag_retriever,
            retrieval_config=None,
            allowed_media_ids=ready_media_ids,
            allowed_note_ids=None,
        )
        ready_media_id_set = set(ready_media_ids)
        scoped_documents = [
            document
            for document in list(evidence.documents)
            if _document_media_id(document) in ready_media_id_set
        ]
        items = [_document_to_lookup_item(document) for document in scoped_documents[:max_results]]
        return RulesRetrievalResult(
            items=items,
            ready_media_ids=ready_media_ids,
            skipped_refs=skipped_refs,
            diagnostics={**base_diagnostics, "retrieval_result_count": len(items), "no_ready_sources": False},
        )

    async def _validate_ref(self, *, owner_user_id: int, ref: RulesPackRef) -> RulesPackSourceValidation:
        if ref.source_type == "media_item":
            return await self._source_validator.validate_media_item(owner_user_id, ref.source_id)
        if ref.source_type == "media_collection":
            return await self._source_validator.validate_media_collection(owner_user_id, ref.source_id)
        raise RPGValidationError("invalid_rules_pack_ref_source_type")


def _document_to_lookup_item(document: Any) -> RuleLookupItem:
    metadata = _metadata(document)
    media_id = _document_media_id(document)
    chunk_index = _int_value(metadata.get("chunk_index") or getattr(document, "chunk_index", None))
    snippet_id = str(metadata.get("snippet_id") or metadata.get("chunk_id") or "").strip()
    if not snippet_id:
        snippet_id = _stable_snippet_id(media_id=media_id, chunk_index=chunk_index, document_id=getattr(document, "id", None))

    text = str(getattr(document, "content", "") or "")[:MAX_RULE_SNIPPET_CHARS]
    return RuleLookupItem(
        origin="user_provided",
        text=text,
        citation=RuleLookupCitation(
            source_type="media_item",
            source_id=media_id,
            source_title=_source_title(document=document, metadata=metadata),
            source_url=_optional_text(metadata.get("source_url") or metadata.get("url")),
            license=_optional_text(metadata.get("license")),
            license_url=_optional_text(metadata.get("license_url")),
            attribution=_optional_text(metadata.get("attribution") or metadata.get("author")),
            trust_level="user_provided",
            content_hash=str(metadata.get("content_hash") or metadata.get("hash") or ""),
            snippet_id=snippet_id,
        ),
        score=float(getattr(document, "score", 0.0) or 0.0),
    )


def _document_media_id(document: Any) -> int | None:
    metadata = _metadata(document)
    return _int_value(
        metadata.get("media_id")
        or metadata.get("media_item_id")
        or metadata.get("source_id")
        or getattr(document, "source_document_id", None)
        or getattr(document, "id", None)
    )


def _metadata(document: Any) -> dict[str, Any]:
    raw_metadata = getattr(document, "metadata", None)
    if isinstance(raw_metadata, dict):
        return raw_metadata
    return {}


def _source_title(*, document: Any, metadata: dict[str, Any]) -> str:
    for value in (
        metadata.get("title"),
        metadata.get("source_title"),
        metadata.get("document_title"),
        getattr(document, "source_document_id", None),
        getattr(document, "id", None),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return "User-provided rules"


def _stable_snippet_id(*, media_id: int | None, chunk_index: int | None, document_id: Any) -> str:
    if media_id is not None and chunk_index is not None:
        return f"media:{media_id}:chunk:{chunk_index}"
    if media_id is not None:
        return f"media:{media_id}:chunk:unknown"
    return f"document:{str(document_id or 'unknown')}"


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
