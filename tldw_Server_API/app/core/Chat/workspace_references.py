"""Bounded recorded Workspace citations shared by message creation and import."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from pydantic import AnyHttpUrl

from tldw_Server_API.app.core.Sync_Contracts.citation_support_types import validate_claim_verification
from tldw_Server_API.app.core.Sync_Contracts.notes_provenance_contract import validate_notes_provenance_payload

WORKSPACE_ASSISTANT_METADATA_KEYS = frozenset({
    "workspace_rag_sources_v1", "claim_verification", "claim_verification_imported",
})


def validate_workspace_assistant_metadata(raw: Any) -> dict[str, Any]:
    """Keep bounded historical assessments and pointers without granting authority."""
    if not isinstance(raw, dict) or not raw or set(raw) - WORKSPACE_ASSISTANT_METADATA_KEYS:
        raise ValueError("Unsupported workspace assistant metadata")
    result = dict(raw)
    if "workspace_rag_sources_v1" in raw:
        result["workspace_rag_sources_v1"] = validate_workspace_references(raw["workspace_rag_sources_v1"])
    if "claim_verification" in raw:
        result["claim_verification"] = validate_claim_verification(raw["claim_verification"])
    if "claim_verification_imported" in raw and (
        raw["claim_verification_imported"] is not True or "claim_verification" not in raw
    ):
        raise ValueError("Imported checks require a saved report and a true historical marker")
    return result


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
