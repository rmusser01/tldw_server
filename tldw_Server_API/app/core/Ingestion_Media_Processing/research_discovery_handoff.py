"""Validate Research Discovery references submitted through Media ingestion."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from tldw_Server_API.app.core.exceptions import (
    ResearchDiscoveryBadRequestError,
    ResearchDiscoveryValidationError,
)


MAX_DISCOVERY_SELECTIONS = 5
_SELECTION_KEYS = frozenset({"result_id", "candidate_id"})


def is_research_discovery_handoff(form_data: Any) -> bool:
    """Return whether either discovery reference field was supplied."""
    return (
        getattr(form_data, "research_discovery_id", None) is not None
        or getattr(form_data, "research_discovery_selections", None) is not None
    )


def parse_research_discovery_selections(raw: str) -> tuple[tuple[str, str], ...]:
    """Parse the bounded selector-only JSON payload."""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise ResearchDiscoveryValidationError("research_discovery_selections_malformed") from None
    if not isinstance(payload, list) or not payload or len(payload) > MAX_DISCOVERY_SELECTIONS:
        detail = (
            "research_discovery_selection_limit_exceeded"
            if isinstance(payload, list) and len(payload) > MAX_DISCOVERY_SELECTIONS
            else "research_discovery_selections_malformed"
        )
        raise ResearchDiscoveryValidationError(detail)

    selections: list[tuple[str, str]] = []
    for item in payload:
        if not isinstance(item, dict) or set(item) != _SELECTION_KEYS:
            raise ResearchDiscoveryValidationError("research_discovery_selections_malformed")
        result_id = item.get("result_id")
        candidate_id = item.get("candidate_id")
        if not isinstance(result_id, str) or not isinstance(candidate_id, str):
            raise ResearchDiscoveryValidationError("research_discovery_selections_malformed")
        pair = (result_id.strip(), candidate_id.strip())
        if not all(pair):
            raise ResearchDiscoveryValidationError("research_discovery_selections_malformed")
        selections.append(pair)

    if len(set(selections)) != len(selections):
        raise ResearchDiscoveryValidationError("research_discovery_duplicate_selection")
    return tuple(selections)


def validate_research_discovery_handoff(
    *,
    form_data: Any,
    files: Sequence[Any] | None,
) -> tuple[tuple[str, str], ...]:
    """Validate discovery mode and return normalized selection pairs."""
    discovery_id = getattr(form_data, "research_discovery_id", None)
    selections_json = getattr(form_data, "research_discovery_selections", None)
    if not isinstance(discovery_id, str) or not discovery_id.strip():
        raise ResearchDiscoveryValidationError("research_discovery_fields_must_be_paired")
    if not isinstance(selections_json, str) or not selections_json.strip():
        raise ResearchDiscoveryValidationError("research_discovery_fields_must_be_paired")
    if getattr(form_data, "media_type", None) != "pdf":
        raise ResearchDiscoveryValidationError("research_discovery_media_type_must_be_pdf")
    if getattr(form_data, "urls", None) or files:
        raise ResearchDiscoveryBadRequestError("research_discovery_conflicting_input_sources")
    if getattr(form_data, "use_cookies", False) or getattr(form_data, "cookies", None):
        raise ResearchDiscoveryBadRequestError("research_discovery_credentials_not_allowed")
    return parse_research_discovery_selections(selections_json)
