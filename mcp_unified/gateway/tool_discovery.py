"""Profile-scoped MCP gateway tool discovery and ranking helpers."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import log
from typing import Any

from mcp_unified.profiles.models import MCPProfile
from mcp_unified.profiles.resolution import build_effective_policy_result

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_DEFAULT_CATEGORY = "uncategorized"
_INSTALLED = "installed"
_RECOMMENDED_UNAVAILABLE = "recommended_unavailable"
_RANKING_METADATA: dict[str, Any] = {
    "semantic_search": False,
    "scoring": "bm25_standard_library",
    "order": [
        "profile_grants",
        "installation_status",
        "category_filter",
        "bm25",
        "category_priority",
        "tool_id",
    ],
}


@dataclass(frozen=True, slots=True)
class _ToolEntry:
    """Normalized internal representation for one visible discovery item."""

    tool_id: str
    tool_name: str | None
    display_name: str
    description: str
    category: str
    capabilities: tuple[str, ...]
    installation_status: str
    source: str
    text: str
    backend_tool: dict[str, Any] | None = None
    activation: str | None = None
    unavailable_reason: str | None = None
    metadata: dict[str, Any] | None = None


def list_profile_tools(profile: MCPProfile, backend_tools: Any) -> dict[str, Any]:
    """Return all tools visible to a profile with deterministic category metadata."""

    entries = _visible_entries(profile, backend_tools)
    ordered = _sort_entries(profile, entries, scores={})
    return {
        "profile_id": profile.id,
        "tools": [_entry_payload(entry, score=0.0) for entry in ordered],
        "categories": _category_payload(profile, ordered),
        "progressive_disclosure": _progressive_disclosure(profile),
        "ranking": _ranking_metadata(),
    }


def search_profile_tools(
    profile: MCPProfile,
    backend_tools: Any,
    *,
    query: str = "",
    category: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search profile-visible tools using filter-first, standard-library BM25 ranking."""

    if limit <= 0:
        return []

    entries = _visible_entries(profile, backend_tools)
    filtered = _filter_by_category(entries, category)
    scores = _bm25_scores(filtered, query)
    ordered = _sort_entries(profile, filtered, scores=scores)
    return [
        _entry_payload(entry, score=scores.get(entry.tool_id, 0.0))
        for entry in ordered[:limit]
    ]


def describe_profile_tool(
    profile: MCPProfile,
    backend_tools: Any,
    tool_id: str,
) -> dict[str, Any] | None:
    """Return a profile-visible tool description by id, or None when out of scope."""

    normalized_tool_id = _clean_text(tool_id)
    if normalized_tool_id is None:
        return None
    for entry in _visible_entries(profile, backend_tools):
        if entry.tool_id == normalized_tool_id:
            return _entry_payload(entry, score=0.0)
    return None


def resolve_profile_tool_call(
    profile: MCPProfile,
    backend_tools: Any,
    tool_id: str,
) -> dict[str, Any]:
    """Resolve a profile-visible bridge tool id to a callable backend tool or denial."""

    normalized_tool_id = _clean_text(tool_id)
    if normalized_tool_id is None:
        return {
            "status": "not_found",
            "reason_code": "tool_not_found",
            "tool_id": tool_id,
        }

    for entry in _visible_entries(profile, backend_tools):
        if entry.tool_id != normalized_tool_id:
            continue
        if entry.installation_status == _INSTALLED and entry.backend_tool is not None:
            return {
                "status": "resolved",
                "tool_id": entry.tool_id,
                "tool_name": entry.tool_name or entry.tool_id,
                "tool": entry.backend_tool,
            }
        return {
            "status": "unavailable",
            "reason_code": "tool_not_enabled",
            "tool_id": entry.tool_id,
            "installation_status": entry.installation_status,
            "activation": entry.activation,
            "unavailable_reason": entry.unavailable_reason,
        }

    return {
        "status": "not_found",
        "reason_code": "tool_not_found",
        "tool_id": normalized_tool_id,
    }


def _visible_entries(profile: MCPProfile, backend_tools: Any) -> list[_ToolEntry]:
    """Return installed and recommendation-only tools visible for a profile."""

    if not getattr(profile, "enabled", False):
        return []
    if build_effective_policy_result(profile).status != "resolved":
        return []

    installed: list[_ToolEntry] = []
    seen_installed: set[str] = set()
    for tool in _backend_tool_sequence(backend_tools):
        entry = _installed_entry(profile, tool)
        if entry is None or entry.tool_id in seen_installed:
            continue
        installed.append(entry)
        seen_installed.add(entry.tool_id)

    entries = list(installed)
    seen_recommendations = set(seen_installed)
    for recommendation in _recommended_tool_sequence(profile):
        entry = _recommended_entry(profile, recommendation)
        if entry is None or entry.tool_id in seen_recommendations:
            continue
        entries.append(entry)
        seen_recommendations.add(entry.tool_id)
    return entries


def _backend_tool_sequence(backend_tools: Any) -> Sequence[Any]:
    """Return a safe sequence of backend tool descriptors."""

    return backend_tools if isinstance(backend_tools, list) else ()


def _installed_entry(profile: MCPProfile, tool: Any) -> _ToolEntry | None:
    """Normalize one installed backend descriptor when profile policy grants it."""

    if not isinstance(tool, dict) or not _installed_tool_allowed(profile, tool):
        return None

    tool_id = _tool_name(tool)
    if tool_id is None:
        return None

    metadata = tool.get("metadata") if isinstance(tool.get("metadata"), dict) else {}
    category = _first_text(metadata, "category") or _first_text(tool, "category")
    category = category or _DEFAULT_CATEGORY
    display_name = (
        _first_text(metadata, "display_name", "displayName", "title", "name")
        or _first_text(tool, "display_name", "displayName", "title")
        or tool_id
    )
    description = (
        _first_text(tool, "description")
        or _first_text(metadata, "description")
        or ""
    )
    capabilities = tuple(_tool_capabilities(tool))
    activation = _first_text(metadata, "activation") or _first_text(tool, "activation")
    unavailable_reason = (
        _first_text(metadata, "unavailable_reason", "reason", "reason_code")
        or _first_text(tool, "unavailable_reason", "reason", "reason_code")
    )
    text = _search_text(
        tool_id,
        display_name,
        category,
        description,
        *capabilities,
        activation,
        unavailable_reason,
    )
    return _ToolEntry(
        tool_id=tool_id,
        tool_name=tool_id,
        display_name=display_name,
        description=description,
        category=category,
        capabilities=capabilities,
        installation_status=_INSTALLED,
        source="backend",
        text=text,
        backend_tool=tool,
        activation=activation,
        unavailable_reason=unavailable_reason,
        metadata=deepcopy(metadata),
    )


def _recommended_entry(
    profile: MCPProfile,
    recommendation: Any,
) -> _ToolEntry | None:
    """Normalize one profile recommendation as visible but not callable."""

    if not isinstance(recommendation, dict):
        return None

    tool_id = _first_text(recommendation, "id", "tool_id", "name")
    if tool_id is None:
        return None

    metadata = (
        recommendation.get("metadata")
        if isinstance(recommendation.get("metadata"), dict)
        else {}
    )
    category = (
        _first_text(recommendation, "category")
        or _first_text(metadata, "category")
        or _DEFAULT_CATEGORY
    )
    display_name = (
        _first_text(recommendation, "display_name", "displayName", "title", "name")
        or _first_text(metadata, "display_name", "displayName", "title", "name")
        or tool_id
    )
    description = (
        _first_text(recommendation, "description")
        or _first_text(metadata, "description")
        or ""
    )
    capabilities = tuple(_recommendation_capabilities(recommendation))
    activation = _first_text(recommendation, "activation") or _first_text(
        metadata,
        "activation",
    )
    unavailable_reason = (
        _first_text(recommendation, "unavailable_reason", "reason", "reason_code")
        or _first_text(metadata, "unavailable_reason", "reason", "reason_code")
        or activation
    )
    text = _search_text(
        tool_id,
        display_name,
        category,
        description,
        *capabilities,
        activation,
        unavailable_reason,
    )
    return _ToolEntry(
        tool_id=tool_id,
        tool_name=None,
        display_name=display_name,
        description=description,
        category=category,
        capabilities=capabilities,
        installation_status=_RECOMMENDED_UNAVAILABLE,
        source="profile_recommendation",
        text=text,
        activation=activation,
        unavailable_reason=unavailable_reason,
        metadata=deepcopy(recommendation),
    )


def _installed_tool_allowed(profile: MCPProfile, tool: dict[str, Any]) -> bool:
    """Return whether effective profile policy grants an installed backend tool."""

    tool_name = _tool_name(tool)
    if tool_name is None:
        return False

    name_result = build_effective_policy_result(profile, tool_name=tool_name)
    if name_result.status == "resolved":
        return True
    if name_result.reason_code != "tool_not_allowed":
        return False

    for capability in _tool_capabilities(tool):
        capability_result = build_effective_policy_result(
            profile,
            tool_name=tool_name,
            capability=capability,
        )
        if capability_result.status == "resolved":
            return True
    return False


def _tool_name(tool: dict[str, Any]) -> str | None:
    """Return a valid backend tool name."""

    return _first_text(tool, "name")


def _tool_capabilities(tool: dict[str, Any]) -> list[str]:
    """Return recognized capability labels from a backend tool descriptor."""

    metadata = tool.get("metadata")
    capability_values: list[Any] = []
    if isinstance(metadata, dict):
        capability_values.extend(_as_sequence(metadata.get("capabilities")))
        capability_values.extend(_as_sequence(metadata.get("capability")))
    capability_values.extend(_as_sequence(tool.get("capabilities")))
    return _unique_texts(capability_values)


def _recommendation_capabilities(recommendation: dict[str, Any]) -> list[str]:
    """Return recognized capability labels from a recommendation descriptor."""

    metadata = recommendation.get("metadata")
    capability_values: list[Any] = []
    capability_values.extend(_as_sequence(recommendation.get("capabilities")))
    capability_values.extend(_as_sequence(recommendation.get("capability")))
    if isinstance(metadata, dict):
        capability_values.extend(_as_sequence(metadata.get("capabilities")))
        capability_values.extend(_as_sequence(metadata.get("capability")))
    return _unique_texts(capability_values)


def _recommended_tool_sequence(profile: MCPProfile) -> Sequence[Any]:
    """Return recommendation entries from profile metadata."""

    tooling = _tooling_metadata(profile)
    recommended_tools = tooling.get("recommended_tools")
    return recommended_tools if isinstance(recommended_tools, list) else ()


def _tooling_metadata(profile: MCPProfile) -> dict[str, Any]:
    """Return profile tooling metadata when present."""

    metadata = profile.metadata if isinstance(profile.metadata, dict) else {}
    tooling = metadata.get("tooling")
    return tooling if isinstance(tooling, dict) else {}


def _progressive_disclosure(profile: MCPProfile) -> dict[str, Any]:
    """Return normalized progressive-disclosure metadata."""

    progressive = _tooling_metadata(profile).get("progressive_disclosure")
    if not isinstance(progressive, dict):
        progressive = {}
    return {
        "direct_categories": _unique_texts(
            _as_sequence(progressive.get("direct_categories")),
        ),
        "deferred_categories": _unique_texts(
            _as_sequence(progressive.get("deferred_categories")),
        ),
        "max_direct_tools": progressive.get("max_direct_tools", 20),
    }


def _category_priority(profile: MCPProfile) -> dict[str, int]:
    """Return category priority from profile progressive-disclosure order."""

    progressive = _progressive_disclosure(profile)
    ordered_categories = [
        *progressive["direct_categories"],
        *progressive["deferred_categories"],
    ]
    return {
        _normalize_category(category): index
        for index, category in enumerate(ordered_categories)
    }


def _filter_by_category(
    entries: list[_ToolEntry],
    category: str | None,
) -> list[_ToolEntry]:
    """Apply an optional category filter before scoring."""

    normalized = _normalize_category(category)
    if normalized is None:
        return entries
    return [
        entry
        for entry in entries
        if _normalize_category(entry.category) == normalized
    ]


def _sort_entries(
    profile: MCPProfile,
    entries: list[_ToolEntry],
    *,
    scores: dict[str, float],
) -> list[_ToolEntry]:
    """Sort entries by installation state, BM25 score, category priority, and id."""

    category_priorities = _category_priority(profile)
    fallback_priority = len(category_priorities)
    return sorted(
        entries,
        key=lambda entry: (
            0 if entry.installation_status == _INSTALLED else 1,
            -scores.get(entry.tool_id, 0.0),
            category_priorities.get(
                _normalize_category(entry.category) or "",
                fallback_priority,
            ),
            _normalize_sort_text(entry.category),
            _normalize_sort_text(entry.tool_id),
        ),
    )


def _category_payload(
    profile: MCPProfile,
    entries: list[_ToolEntry],
) -> list[dict[str, Any]]:
    """Return deterministic category counts for the visible catalog."""

    category_priorities = _category_priority(profile)
    fallback_priority = len(category_priorities)
    categories: dict[str, Counter[str]] = {}
    for entry in entries:
        category = _normalize_category(entry.category) or _DEFAULT_CATEGORY
        categories.setdefault(category, Counter())
        categories[category][entry.installation_status] += 1

    return [
        {
            "category": category,
            "count": sum(counts.values()),
            "installed_count": counts.get(_INSTALLED, 0),
            "recommended_unavailable_count": counts.get(
                _RECOMMENDED_UNAVAILABLE,
                0,
            ),
        }
        for category, counts in sorted(
            categories.items(),
            key=lambda item: (
                category_priorities.get(
                    _normalize_category(item[0]) or "",
                    fallback_priority,
                ),
                _normalize_sort_text(item[0]),
            ),
        )
    ]


def _entry_payload(entry: _ToolEntry, *, score: float) -> dict[str, Any]:
    """Return a JSON-safe public payload for one discovery entry."""

    payload: dict[str, Any] = {
        "tool_id": entry.tool_id,
        "display_name": entry.display_name,
        "description": entry.description,
        "category": entry.category,
        "capabilities": list(entry.capabilities),
        "installation_status": entry.installation_status,
        "source": entry.source,
        "ranking": _ranking_metadata(score),
    }
    if entry.tool_name is not None:
        payload["tool_name"] = entry.tool_name
    if entry.activation is not None:
        payload["activation"] = entry.activation
    if entry.unavailable_reason is not None:
        payload["unavailable_reason"] = entry.unavailable_reason
    if entry.metadata:
        payload["metadata"] = deepcopy(entry.metadata)
    return payload


def _ranking_metadata(score: float | None = None) -> dict[str, Any]:
    """Return ranking metadata that documents the non-semantic scorer."""

    metadata = deepcopy(_RANKING_METADATA)
    if score is not None:
        metadata["bm25_score"] = round(score, 6)
    return metadata


def _bm25_scores(entries: list[_ToolEntry], query: str) -> dict[str, float]:
    """Return BM25 scores for entries using only standard-library primitives."""

    query_terms = _tokenize(query)
    if not query_terms or not entries:
        return {entry.tool_id: 0.0 for entry in entries}

    documents = [_tokenize(entry.text) for entry in entries]
    doc_count = len(documents)
    average_length = sum(len(document) for document in documents) / doc_count
    average_length = average_length or 1.0

    document_frequencies: Counter[str] = Counter()
    for document in documents:
        document_frequencies.update(set(document))

    k1 = 1.5
    b = 0.75
    scores: dict[str, float] = {}
    for entry, document in zip(entries, documents):
        frequencies = Counter(document)
        document_length = len(document) or 1
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            idf = log(
                1
                + (doc_count - document_frequencies[term] + 0.5)
                / (document_frequencies[term] + 0.5),
            )
            denominator = frequency + k1 * (
                1 - b + b * document_length / average_length
            )
            score += idf * (frequency * (k1 + 1)) / denominator
        scores[entry.tool_id] = score
    return scores


def _search_text(*values: Any) -> str:
    """Join text-bearing fields used by the BM25 scorer."""

    return " ".join(value for value in _unique_texts(values))


def _tokenize(value: Any) -> list[str]:
    """Tokenize search text into lowercase alphanumeric terms."""

    if not isinstance(value, str):
        return []
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(value)]


def _first_text(mapping: dict[str, Any], *keys: str) -> str | None:
    """Return the first non-empty string value for the supplied keys."""

    for key in keys:
        value = mapping.get(key)
        cleaned = _clean_text(value)
        if cleaned is not None:
            return cleaned
    return None


def _clean_text(value: Any) -> str | None:
    """Return stripped text for non-empty strings."""

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _unique_texts(values: Sequence[Any]) -> list[str]:
    """Return stripped string values without duplicates, preserving order."""

    texts: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value)
        if cleaned is None or cleaned in seen:
            continue
        texts.append(cleaned)
        seen.add(cleaned)
    return texts


def _as_sequence(value: Any) -> Sequence[Any]:
    """Normalize scalar-or-sequence metadata into a sequence."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return value
    return ()


def _normalize_category(category: str | None) -> str | None:
    """Normalize category labels for filtering and priority lookup."""

    cleaned = _clean_text(category)
    return cleaned.lower() if cleaned is not None else None


def _normalize_sort_text(value: str) -> str:
    """Normalize text for stable lexical tie-breaks."""

    return value.casefold()


__all__ = [
    "describe_profile_tool",
    "list_profile_tools",
    "resolve_profile_tool_call",
    "search_profile_tools",
]
