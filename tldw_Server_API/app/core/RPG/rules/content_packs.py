from __future__ import annotations

from dataclasses import dataclass

from tldw_Server_API.app.core.RPG.constants import (
    RPG_ADAPTER_DND5E_SRD,
    RPG_ADAPTER_FATE,
    RPG_ADAPTER_PF2E,
)
from tldw_Server_API.app.core.RPG.models import RuleCitation


@dataclass(frozen=True, slots=True)
class RuleLookupItem:
    text: str
    citation: RuleCitation
    score: float


@dataclass(frozen=True, slots=True)
class RuleLookupResult:
    query: str
    results: list[RuleLookupItem]
    diagnostics: dict[str, object]


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
