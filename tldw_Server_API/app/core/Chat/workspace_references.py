"""Bounded recorded Workspace citations shared by message creation and import."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from pydantic import AnyHttpUrl

from tldw_Server_API.app.core.Sync_Contracts.notes_provenance_contract import validate_notes_provenance_payload


def validate_workspace_references(raw: Any) -> dict[str, Any]:
    """Validate recorded references without granting access to their Media pointers."""
    if not isinstance(raw, dict) or set(raw) != {"origin", "sources"}:
        raise ValueError("Workspace references require only origin and sources")
    provenance = validate_notes_provenance_payload(raw)
    sources = provenance.get("sources")
    if provenance["origin"] != "knowledge_qa" or not sources or len(sources) > 64:
        raise ValueError("Workspace references require 1..64 Knowledge sources")
    for source in sources:
        if source["mediaId"] is None or not source["title"].strip() or not source["excerpt"].strip():
            raise ValueError("Workspace references require a Media ID, title and excerpt")
        url = source.get("url")
        if url:
            AnyHttpUrl(url)
    encoded = quote(json.dumps(provenance, ensure_ascii=False, separators=(",", ":")), safe="~()*!.'-")
    if len(encoded) > 262_144:
        raise ValueError("Workspace references exceed the encoded size limit")
    return provenance
