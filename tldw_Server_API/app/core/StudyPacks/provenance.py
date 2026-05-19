"""Citation persistence and deep-dive helpers for study-pack flashcards."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlencode

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

_SUPPORTED_SOURCE_TYPES = {"note", "media", "message"}


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_locator(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        locator = value.strip()
        return locator or None
    if isinstance(value, Mapping):
        payload = {str(key): item for key, item in value.items() if item not in (None, "", [], {})}
        return json.dumps(payload, sort_keys=True) if payload else None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        payload = [item for item in value if item not in (None, "", [], {})]
        return json.dumps(payload, sort_keys=True) if payload else None
    locator = str(value).strip()
    return locator or None


def _parse_locator(locator: Any) -> dict[str, Any] | str | None:
    normalized = _normalize_locator(locator)
    if normalized is None:
        return None
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError:
        return normalized
    if isinstance(parsed, Mapping):
        return {str(key): value for key, value in parsed.items() if value not in (None, "", [], {})}
    if isinstance(parsed, list):
        return normalized
    return normalized


def _normalize_citation_row(raw_citation: Mapping[str, Any], *, fallback_ordinal: int = 0) -> dict[str, Any] | None:
    source_type = _normalize_text(raw_citation.get("source_type"))
    source_id = _normalize_text(raw_citation.get("source_id"))
    if source_type is None or source_id is None:
        return None
    source_type = source_type.lower()
    if source_type not in _SUPPORTED_SOURCE_TYPES:
        return None
    ordinal = _coerce_ordinal(raw_citation.get("ordinal", None), fallback_ordinal=fallback_ordinal)

    citation = dict(raw_citation)
    citation["source_type"] = source_type
    citation["source_id"] = source_id
    citation["citation_text"] = _normalize_text(raw_citation.get("citation_text"))
    citation["locator"] = _normalize_locator(raw_citation.get("locator"))
    citation["ordinal"] = ordinal
    return citation


def _coerce_ordinal(value: Any, *, fallback_ordinal: int) -> int:
    if value in (None, ""):
        ordinal = fallback_ordinal
    else:
        try:
            ordinal = int(value)
        except (TypeError, ValueError):
            ordinal = fallback_ordinal
    if ordinal < 0:
        return fallback_ordinal
    return ordinal


def _route_kind_rank(route_kind: Any) -> int:
    return {
        "exact_locator": 0,
        "workspace_route": 1,
        "citation_only": 2,
    }.get(str(route_kind or "citation_only"), 2)


def _citation_route_rank(citation: Mapping[str, Any]) -> int:
    return _route_kind_rank(_build_deep_dive_target(citation).get("route_kind"))


def _primary_selection_key(citation: Mapping[str, Any], index: int) -> tuple[int, int, int]:
    ordinal = _coerce_ordinal(citation.get("ordinal", None), fallback_ordinal=index)
    return (
        _citation_route_rank(citation),
        ordinal,
        index,
    )


def _deep_dive_selection_key(citation: Mapping[str, Any], index: int) -> tuple[int, int, int]:
    ordinal = _coerce_ordinal(citation.get("ordinal", None), fallback_ordinal=index)
    target = _build_deep_dive_target(citation)
    return (
        _route_kind_rank(target.get("route_kind")),
        ordinal,
        index,
    )


def _route_with_query(base_route: str, params: Mapping[str, Any]) -> str:
    normalized_params = {
        str(key): value
        for key, value in params.items()
        if value not in (None, "", [], {})
    }
    if not normalized_params:
        return base_route
    return f"{base_route}?{urlencode(normalized_params, doseq=True)}"


def _build_note_route(source_id: str, locator: dict[str, Any] | str | None) -> tuple[str, str]:
    base_route = f"/notes/{source_id}"
    if locator is None:
        return "workspace_route", base_route
    if isinstance(locator, Mapping):
        if locator:
            return "exact_locator", _route_with_query(base_route, locator)
        return "workspace_route", base_route
    return "exact_locator", _route_with_query(base_route, {"locator": locator})


def _build_media_route(source_id: str, locator: dict[str, Any] | str | None) -> tuple[str, str]:
    base_route = f"/media/{source_id}"
    if locator is None:
        return "workspace_route", base_route
    if isinstance(locator, Mapping):
        if locator:
            return "exact_locator", _route_with_query(base_route, locator)
        return "workspace_route", base_route
    return "exact_locator", _route_with_query(base_route, {"locator": locator})


def _build_message_route(source_id: str, locator: dict[str, Any] | str | None) -> dict[str, Any]:
    if not isinstance(locator, Mapping):
        return {
            "route_kind": "citation_only",
            "route": None,
            "available": False,
            "fallback_reason": "message_conversation_id_required",
        }

    conversation_id = _normalize_text(locator.get("conversation_id"))
    if conversation_id is None:
        return {
            "route_kind": "citation_only",
            "route": None,
            "available": False,
            "fallback_reason": "message_conversation_id_required",
        }

    base_route = f"/conversations/{conversation_id}"
    query_params = {"message_id": source_id}
    extra_locator = {
        str(key): value
        for key, value in locator.items()
        if key not in {"conversation_id", "message_id"} and value not in (None, "", [], {})
    }
    if extra_locator:
        query_params.update(extra_locator)
        route_kind = "exact_locator"
    else:
        route_kind = "workspace_route"
    return {
        "route_kind": route_kind,
        "route": _route_with_query(base_route, query_params),
        "available": True,
        "fallback_reason": None,
    }


def _build_deep_dive_target(citation: Mapping[str, Any]) -> dict[str, Any]:
    source_type = str(citation["source_type"])
    source_id = str(citation["source_id"])
    citation_ordinal = _coerce_ordinal(citation.get("ordinal", None), fallback_ordinal=0)
    locator = _parse_locator(citation.get("locator"))

    if source_type == "note":
        route_kind, route = _build_note_route(source_id, locator)
        return {
            "source_type": source_type,
            "source_id": source_id,
            "citation_ordinal": citation_ordinal,
            "route_kind": route_kind,
            "route": route,
            "available": True,
            "fallback_reason": None,
        }
    if source_type == "media":
        route_kind, route = _build_media_route(source_id, locator)
        return {
            "source_type": source_type,
            "source_id": source_id,
            "citation_ordinal": citation_ordinal,
            "route_kind": route_kind,
            "route": route,
            "available": True,
            "fallback_reason": None,
        }
    if source_type == "message":
        message_route = _build_message_route(source_id, locator)
        return {
            "source_type": source_type,
            "source_id": source_id,
            "citation_ordinal": citation_ordinal,
            **message_route,
        }
    return {
        "source_type": source_type,
        "source_id": source_id,
        "citation_ordinal": citation_ordinal,
        "route_kind": "citation_only",
        "route": None,
        "available": False,
        "fallback_reason": "unsupported_source_type",
    }


def normalize_citations_for_persistence(citations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize raw citation payloads and assign a deterministic ordinal zero primary."""
    normalized: list[dict[str, Any]] = []
    for index, citation in enumerate(citations):
        row = _normalize_citation_row(citation, fallback_ordinal=index)
        if row is not None:
            normalized.append(row)
    if not normalized:
        return []

    primary_index = min(range(len(normalized)), key=lambda idx: _primary_selection_key(normalized[idx], idx))
    primary = dict(normalized[primary_index])
    primary["ordinal"] = 0

    ordered_rows = [primary]
    for index, citation in enumerate(normalized):
        if index == primary_index:
            continue
        ordered_rows.append(dict(citation))

    for ordinal, citation in enumerate(ordered_rows):
        citation["ordinal"] = ordinal
    return ordered_rows


def select_primary_citation(citations: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Return the citation marked primary, falling back deterministically when needed."""
    normalized: list[dict[str, Any]] = []
    for index, citation in enumerate(citations):
        row = _normalize_citation_row(citation, fallback_ordinal=index)
        if row is not None:
            normalized.append(row)
    if not normalized:
        return None

    selected = min(enumerate(normalized), key=lambda item: _primary_selection_key(item[1], item[0]))[1]
    return dict(selected)


def resolve_deep_dive_target(citations: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Resolve the best citation row to use for a source deep dive."""
    normalized: list[dict[str, Any]] = []
    for index, citation in enumerate(citations):
        row = _normalize_citation_row(citation, fallback_ordinal=index)
        if row is not None:
            normalized.append(row)
    if not normalized:
        return None

    selected = min(enumerate(normalized), key=lambda item: _deep_dive_selection_key(item[1], item[0]))[1]
    return _build_deep_dive_target(selected)


class FlashcardProvenanceStore:
    """Owns flashcard citation persistence, legacy mirroring, and deep-dive lookup."""

    def __init__(self, db: CharactersRAGDB):
        self.db = db

    def list_citations(self, flashcard_uuid: str) -> list[dict[str, Any]]:
        """Return persisted citations in API-friendly form."""
        rows = self.db.list_flashcard_citations(flashcard_uuid)
        citations: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            normalized = _normalize_citation_row(row, fallback_ordinal=index)
            if normalized is None:
                continue
            normalized["id"] = int(row.get("id", 0) or 0)
            normalized["flashcard_uuid"] = str(row.get("flashcard_uuid") or flashcard_uuid)
            normalized["created_at"] = row.get("created_at")
            normalized["last_modified"] = row.get("last_modified")
            normalized["deleted"] = bool(row.get("deleted"))
            normalized["client_id"] = row.get("client_id")
            normalized["version"] = int(row.get("version", 1) or 1)
            citations.append(normalized)
        return citations

    def get_study_pack_summary(self, flashcard_uuid: str) -> dict[str, Any] | None:
        """Return the first active study-pack summary containing the flashcard."""
        return self.db.get_study_pack_for_flashcard(flashcard_uuid)

    def read_flashcard_provenance(self, flashcard_uuid: str) -> dict[str, Any]:
        """Read persisted citations and derive primary/deep-dive selections."""
        citations = self.list_citations(flashcard_uuid)
        return {
            "citations": citations,
            "primary_citation": select_primary_citation(citations),
            "deep_dive_target": resolve_deep_dive_target(citations),
        }

    def persist_flashcard_citations(
        self,
        flashcard_uuid: str,
        citations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Normalize and persist flashcard citations, then mirror the primary citation."""
        prepared = normalize_citations_for_persistence(citations)
        primary = select_primary_citation(prepared)
        inserted = self.db.replace_flashcard_citations_and_source_reference_summary(
            flashcard_uuid,
            prepared,
            source_ref_type=str(primary["source_type"]) if primary is not None else None,
            source_ref_id=str(primary["source_id"]) if primary is not None else None,
        )

        provenance = self.read_flashcard_provenance(flashcard_uuid)
        provenance["inserted_count"] = inserted
        return provenance
