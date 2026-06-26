from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from tldw_Server_API.app.core.RPG.constants import (
    RPG_ADAPTER_DND5E_SRD,
    RPG_ADAPTER_FATE,
    RPG_ADAPTER_PF2E,
)
from tldw_Server_API.app.core.RPG.models import RuleCitation


@dataclass(frozen=True, slots=True)
class RuleLookupCitation:
    source_type: str
    source_id: int | None
    source_title: str
    source_url: str | None
    license: str | None
    license_url: str | None
    attribution: str | None
    trust_level: str
    content_hash: str
    snippet_id: str
    adapter_key: str | None = None
    source_version: str | None = None
    content_pack_version: str | None = None


@dataclass(frozen=True, slots=True)
class RuleLookupItem:
    origin: Literal["user_provided", "bundled_citation"]
    text: str
    citation: RuleLookupCitation
    score: float


@dataclass(frozen=True, slots=True)
class RuleLookupResult:
    query: str
    mode: Literal["lookup", "answer"]
    results: list[RuleLookupItem]
    answer: str | None
    answer_status: str
    answer_citation_ids: list[str]
    diagnostics: dict[str, Any]


def bundled_citation_lookup_item(citation: RuleCitation) -> RuleLookupItem:
    return RuleLookupItem(
        origin="bundled_citation",
        text="",
        citation=RuleLookupCitation(
            source_type="bundled_rules_citation",
            source_id=None,
            source_title=citation.source_title,
            source_url=citation.source_url,
            license=citation.license,
            license_url=citation.license_url,
            attribution=citation.attribution,
            trust_level=citation.trust_level,
            content_hash=citation.content_hash,
            snippet_id=citation.snippet_id,
            adapter_key=citation.adapter_key,
            source_version=citation.source_version,
            content_pack_version=citation.content_pack_version,
        ),
        score=0.0,
    )


DND5E_SRD_CITATIONS = (
    RuleCitation(
        adapter_key=RPG_ADAPTER_DND5E_SRD,
        source_title="Systems Reference Document 5.1",
        source_url="https://dnd.wizards.com/resources/systems-reference-document",
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution="D&D 5e SRD rules reference",
        trust_level="reference",
        content_hash="citation-only",
        snippet_id="dnd5e-srd-citation-index",
        source_version="SRD 5.1",
        content_pack_version="1.0.0",
    ),
)

FATE_CITATIONS = (
    RuleCitation(
        adapter_key=RPG_ADAPTER_FATE,
        source_title="Fate SRD",
        source_url="https://fate-srd.com/",
        license="CC-BY-3.0",
        license_url="https://creativecommons.org/licenses/by/3.0/",
        attribution="Fate Core System rules reference",
        trust_level="reference",
        content_hash="citation-only",
        snippet_id="fate-srd-citation-index",
        source_version="Fate Core",
        content_pack_version="1.0.0",
    ),
)

PF2E_CITATIONS = (
    RuleCitation(
        adapter_key=RPG_ADAPTER_PF2E,
        source_title="Archives of Nethys Pathfinder 2e",
        source_url="https://2e.aonprd.com/",
        license="ORC and Paizo Community Use references",
        license_url="https://downloads.paizo.com/ORC_License_FINAL.pdf",
        attribution="Pathfinder Second Edition rules reference",
        trust_level="reference",
        content_hash="citation-only",
        snippet_id="pf2e-citation-index",
        source_version="Pathfinder 2e Remaster",
        content_pack_version="1.0.0",
    ),
)

BUILT_IN_CITATIONS_BY_ADAPTER: dict[str, tuple[RuleCitation, ...]] = {
    RPG_ADAPTER_DND5E_SRD: DND5E_SRD_CITATIONS,
    RPG_ADAPTER_FATE: FATE_CITATIONS,
    RPG_ADAPTER_PF2E: PF2E_CITATIONS,
}
