from __future__ import annotations

from typing import Any

from ..models import AccessScope
from ..store.sqlite import DocsCatalogStore


class DocsAliasResolver:
    def __init__(self, store: DocsCatalogStore) -> None:
        self.store = store

    def resolve(self, *, scope: AccessScope, name: str) -> dict[str, Any]:
        query = name.strip()
        matches = self.store.resolve_name(scope=scope, name=query)
        return {"query": query, "matches": matches, "ambiguous": len(matches) > 1}

    def resolve_library_id(self, *, scope: AccessScope, library_name: str) -> dict[str, Any]:
        result = self.resolve(scope=scope, name=library_name)
        package_like = [match for match in result["matches"] if match.get("target_type") == "collection"]
        return {
            "query": library_name,
            "matches": [{**match, "canonical_tool": "docs.resolve"} for match in package_like],
            "canonical_tool": "docs.resolve",
        }
