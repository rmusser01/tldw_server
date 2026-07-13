"""Resolve persisted Research Discovery selections for Media ingestion."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
from tldw_Server_API.app.core.exceptions import ResearchDiscoveryValidationError

from .identity import build_fingerprint, has_unsafe_url_material, normalize_doi, safe_provider_metadata
from .models import DiscoveryOACandidate, is_phase2a_media_handoff_candidate
from .oa import build_candidate_id

MAX_DISCOVERY_SELECTIONS = 5
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class ResolvedDiscoverySelection:
    """Server-owned discovery metadata needed by the Media pipeline."""

    result_id: str
    candidate_id: str
    fingerprint: str
    candidate_type: str
    url: str
    canonical_url: str | None
    title: str
    authors: tuple[str, ...]
    identifiers: dict[str, str]
    source_id: str
    provider: str
    access_status: str | None
    license_hint: str | None
    safe_metadata: dict[str, Any]


def resolve_discovery_selections(
    *,
    owner_user_id: str,
    discovery_id: str,
    selections: Sequence[tuple[str, str]],
    snapshot_db: ResearchSessionsDB | None = None,
) -> tuple[ResolvedDiscoverySelection, ...]:
    """Resolve ordered result/candidate pairs from an owner-scoped snapshot."""
    normalized_selections = _validate_selections(selections)
    db = snapshot_db or ResearchSessionsDB(DatabasePaths.get_research_sessions_db_path(owner_user_id))
    snapshot = db.get_discovery_snapshot(discovery_id, owner_user_id=owner_user_id)
    if snapshot is None:
        raise ResearchDiscoveryValidationError("research_discovery_snapshot_unavailable")

    results = snapshot.response_json.get("results")
    if not isinstance(results, list):
        raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")
    result_index = _index_results(results)

    resolved: list[ResolvedDiscoverySelection] = []
    for result_id, candidate_id in normalized_selections:
        indexed = result_index.get(result_id)
        if indexed is None or candidate_id not in indexed[1]:
            raise ResearchDiscoveryValidationError("research_discovery_selection_not_found")
        result, candidates = indexed
        resolved.append(_resolve_selection(result, candidates[candidate_id]))
    return tuple(resolved)


def _validate_selections(selections: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    if isinstance(selections, (str, bytes)):
        raise ResearchDiscoveryValidationError("research_discovery_selection_malformed")
    normalized: list[tuple[str, str]] = []
    for selection in selections:
        if isinstance(selection, (str, bytes)) or not isinstance(selection, Sequence) or len(selection) != 2:
            raise ResearchDiscoveryValidationError("research_discovery_selection_malformed")
        result_id, candidate_id = selection
        if not isinstance(result_id, str) or not isinstance(candidate_id, str):
            raise ResearchDiscoveryValidationError("research_discovery_selection_malformed")
        pair = (result_id.strip(), candidate_id.strip())
        if not all(pair):
            raise ResearchDiscoveryValidationError("research_discovery_selection_malformed")
        normalized.append(pair)

    if not normalized:
        raise ResearchDiscoveryValidationError("research_discovery_selections_required")
    if len(normalized) > MAX_DISCOVERY_SELECTIONS:
        raise ResearchDiscoveryValidationError("research_discovery_selection_limit_exceeded")
    if len(set(normalized)) != len(normalized):
        raise ResearchDiscoveryValidationError("research_discovery_duplicate_selection")
    return tuple(normalized)


def _index_results(
    results: list[Any],
) -> dict[str, tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]]:
    indexed: dict[str, tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]] = {}
    for result in results:
        if not isinstance(result, Mapping):
            raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")
        result_id = _required_text(result.get("result_id"))
        candidates_raw = result.get("oa_candidates")
        if result_id is None or not isinstance(candidates_raw, list) or result_id in indexed:
            raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")

        candidates: dict[str, Mapping[str, Any]] = {}
        for candidate in candidates_raw:
            if not isinstance(candidate, Mapping):
                raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")
            candidate_id = _required_text(candidate.get("candidate_id"))
            if candidate_id is None or candidate_id in candidates:
                raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")
            candidates[candidate_id] = candidate
        indexed[result_id] = (result, candidates)
    return indexed


def _resolve_selection(
    result: Mapping[str, Any],
    candidate_raw: Mapping[str, Any],
) -> ResolvedDiscoverySelection:
    candidate = _parse_candidate(candidate_raw)
    if not is_phase2a_media_handoff_candidate(candidate) or not _is_safe_http_url(candidate.safe_url):
        raise ResearchDiscoveryValidationError("research_discovery_candidate_not_ingestable")

    result_id = _required_text(result.get("result_id"))
    fingerprint = _required_text(result.get("fingerprint"))
    title = _required_text(result.get("title"))
    source_id = _required_text(result.get("primary_source_id"))
    primary_provider = _required_text(result.get("primary_provider"))
    authors = _authors(result.get("authors"))
    if None in (result_id, fingerprint, title, source_id, primary_provider) or authors is None:
        raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")
    identity_record = dict(result)
    identity_record["source_id"] = source_id
    identity_record["provider"] = primary_provider
    if build_fingerprint(identity_record) != fingerprint:
        raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")
    expected_candidate_id = build_candidate_id(
        result_fingerprint=fingerprint,
        candidate_type=candidate.candidate_type,
        provider=candidate.provider,
        safe_url=candidate.safe_url,
        resolver_reference=candidate.resolver_reference,
    )
    if candidate.candidate_id != expected_candidate_id:
        raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")

    canonical_url = _optional_http_url(result.get("canonical_url"))
    safe_metadata_raw = result.get("safe_metadata")
    if safe_metadata_raw is None:
        safe_metadata_raw = {}
    if not isinstance(safe_metadata_raw, Mapping):
        raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")

    return ResolvedDiscoverySelection(
        result_id=result_id,
        candidate_id=candidate.candidate_id,
        fingerprint=fingerprint,
        candidate_type=candidate.candidate_type,
        url=candidate.safe_url or "",
        canonical_url=canonical_url,
        title=title,
        authors=authors,
        identifiers=_identifiers(result),
        source_id=source_id,
        provider=candidate.provider,
        access_status=candidate.access_status,
        license_hint=candidate.license_hint,
        safe_metadata=safe_provider_metadata(dict(safe_metadata_raw)),
    )


def _parse_candidate(raw: Mapping[str, Any]) -> DiscoveryOACandidate:
    candidate_id = _required_text(raw.get("candidate_id"))
    candidate_type = _required_text(raw.get("candidate_type"))
    safe_url = _optional_text(raw.get("safe_url"))
    provider = _required_text(raw.get("provider"))
    url_redacted = raw.get("url_redacted")
    requires_reresolution = raw.get("requires_reresolution")
    if (
        None in (candidate_id, candidate_type, provider)
        or not isinstance(url_redacted, bool)
        or not isinstance(requires_reresolution, bool)
    ):
        raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")

    return DiscoveryOACandidate(
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        safe_url=safe_url,
        resolver_reference=_optional_text(raw.get("resolver_reference")),
        url_redacted=url_redacted,
        requires_reresolution=requires_reresolution,
        provider=provider,
        access_status=_optional_text(raw.get("access_status")),
        license_hint=_optional_text(raw.get("license_hint")),
        content_type_hint=_optional_text(raw.get("content_type_hint")),
        rank=_integer(raw.get("rank")),
        confidence=_number(raw.get("confidence")),
        warnings=_strings(raw.get("warnings")),
    )


def _identifiers(result: Mapping[str, Any]) -> dict[str, str]:
    provider_ids_raw = result.get("provider_ids")
    if provider_ids_raw is None:
        provider_ids_raw = {}
    if not isinstance(provider_ids_raw, Mapping):
        raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")
    safe_provider_ids = safe_provider_metadata(dict(provider_ids_raw))

    identifiers: dict[str, str] = {}
    doi = normalize_doi(result.get("doi") or safe_provider_ids.get("doi"))
    if doi:
        identifiers["doi"] = doi
    pmid = _optional_identifier_text(result.get("pmid") or safe_provider_ids.get("pmid"))
    if pmid:
        identifiers["pmid"] = pmid.lower()
    pmcid = _normalize_pmcid(result.get("pmcid") or safe_provider_ids.get("pmcid"))
    if pmcid:
        identifiers["pmcid"] = pmcid
    arxiv_id = _normalize_arxiv_id(result.get("arxiv_id") or safe_provider_ids.get("arxiv_id"))
    if arxiv_id:
        identifiers["arxiv_id"] = arxiv_id

    for key, value in safe_provider_ids.items():
        key_text = _required_text(key)
        value_text = _identifier_text(value)
        if key_text and value_text and key_text.lower() not in {"doi", "pmid", "pmcid", "arxiv_id"}:
            identifiers.setdefault(key_text.lower(), value_text)
    return dict(sorted(identifiers.items()))


def _required_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")
    return value.strip() or None


def _optional_identifier_text(value: Any) -> str | None:
    if value is None:
        return None
    text = _identifier_text(value)
    if text is not None or isinstance(value, str):
        return text
    raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")


def _identifier_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    return None


def _authors(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    authors = tuple(_required_text(author) for author in value)
    if any(author is None for author in authors):
        return None
    return tuple(author for author in authors if author is not None)


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")
    return value


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")
    return float(value)


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")
    strings = tuple(_required_text(item) for item in value)
    if any(item is None for item in strings):
        raise ResearchDiscoveryValidationError("research_discovery_snapshot_malformed")
    return tuple(item for item in strings if item is not None)


def _is_safe_http_url(value: str | None) -> bool:
    if not value or has_unsafe_url_material(value):
        return False
    parsed = urlsplit(value)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)


def _optional_http_url(value: Any) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    return text if _is_safe_http_url(text) else None


def _normalize_pmcid(value: Any) -> str | None:
    text = _optional_identifier_text(value)
    if text is None:
        return None
    text = text.upper()
    return f"PMC{text}" if text.isdigit() else text


def _normalize_arxiv_id(value: Any) -> str | None:
    text = _optional_identifier_text(value)
    if text is None:
        return None
    text = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^arxiv:\s*", "", text, flags=re.IGNORECASE)
    return _ARXIV_VERSION_RE.sub("", text).strip().lower() or None
