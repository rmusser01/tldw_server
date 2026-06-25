from __future__ import annotations

from tldw_Server_API.app.core.RPG.errors import RPGValidationError
from tldw_Server_API.app.core.RPG.rules.content_packs import (
    BUILT_IN_CITATIONS_BY_ADAPTER,
    RuleLookupItem,
    RuleLookupResult,
)


class RulesLookupService:
    def lookup(
        self,
        adapter_key: str,
        query: str,
        linked_rules_pack_refs: list[dict[str, object]],
    ) -> RuleLookupResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise RPGValidationError("rules_query_required")

        citations = BUILT_IN_CITATIONS_BY_ADAPTER.get(adapter_key, ())
        return RuleLookupResult(
            query=normalized_query,
            results=[RuleLookupItem(text="", citation=citation, score=0.0) for citation in citations],
            diagnostics={
                "bundled_policy": "citations_only" if citations else "no_match",
                "result_mode": "citation_index",
                "linked_rules_pack_count": len(linked_rules_pack_refs),
            },
        )
