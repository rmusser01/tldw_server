from __future__ import annotations

from typing import Any, Literal

from loguru import logger

from tldw_Server_API.app.core.RPG.errors import RPGError, RPGValidationError
from tldw_Server_API.app.core.RPG.rules.content_packs import (
    BUILT_IN_CITATIONS_BY_ADAPTER,
    RuleLookupResult,
    bundled_citation_lookup_item,
)
from tldw_Server_API.app.core.RPG.rules.refs import rules_pack_ref_from_dict
from tldw_Server_API.app.core.RPG.rules.retrieval import RulesRetriever


class RulesLookupService:
    def __init__(
        self,
        *,
        retriever: RulesRetriever | None = None,
        answer_generator: Any | None = None,
    ) -> None:
        self._retriever = retriever
        self._answer_generator = answer_generator

    async def lookup(
        self,
        *,
        owner_user_id: int,
        adapter_key: str,
        query: str,
        linked_rules_pack_refs: list[dict[str, Any]],
        mode: Literal["lookup", "answer"] = "lookup",
        answer_options: Any | None = None,
    ) -> RuleLookupResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise RPGValidationError("rules_query_required")
        if mode not in {"lookup", "answer"}:
            raise RPGValidationError("invalid_rules_lookup_mode")

        refs = [rules_pack_ref_from_dict(ref) for ref in linked_rules_pack_refs]
        enabled_count = sum(1 for ref in refs if ref.enabled)
        citations = BUILT_IN_CITATIONS_BY_ADAPTER.get(adapter_key, ())
        bundled_items = [bundled_citation_lookup_item(citation) for citation in citations]
        diagnostics: dict[str, Any] = {
            "bundled_policy": "citations_only" if citations else "no_match",
            "result_mode": "retrieval_with_citation_index"
            if self._retriever is not None and refs
            else "citation_index",
            "linked_rules_pack_count": len(refs),
            "enabled_rules_pack_count": enabled_count,
            "ready_media_item_count": 0,
            "retrieval_result_count": 0,
            "bundled_citation_count": len(bundled_items),
            "skipped_refs": [],
            "broad_fallback_used": False,
        }
        retrieval_items = []
        answer_status = "not_requested" if mode == "lookup" else "no_evidence"

        if self._retriever is not None and refs:
            try:
                retrieval = await self._retriever.retrieve(
                    owner_user_id=owner_user_id,
                    query=normalized_query,
                    refs=refs,
                    max_results=8,
                )
            except RPGError:
                raise
            except Exception as exc:
                logger.warning("RPG rules retrieval failed: {}", type(exc).__name__)
                diagnostics["retrieval_error"] = "retrieval_failed"
                if mode == "answer":
                    answer_status = "retrieval_error"
            else:
                retrieval_items = list(retrieval.items)
                if mode == "answer" and retrieval_items:
                    answer_status = "not_generated"
                diagnostics.update(retrieval.diagnostics)
                diagnostics["ready_media_item_count"] = len(retrieval.ready_media_ids)
                diagnostics["retrieval_result_count"] = len(retrieval_items)
                diagnostics["skipped_refs"] = list(retrieval.skipped_refs)

        return RuleLookupResult(
            query=normalized_query,
            mode=mode,
            results=[*retrieval_items, *bundled_items],
            answer=None,
            answer_status=answer_status,
            answer_citation_ids=[],
            diagnostics=diagnostics,
        )
