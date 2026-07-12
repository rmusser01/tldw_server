"""Validate Research Discovery references submitted through Media ingestion."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from starlette import status
from starlette.responses import JSONResponse

from tldw_Server_API.app.core.DB_Management.media_db.dedupe_urls import normalize_media_dedupe_url
from tldw_Server_API.app.core.Ingestion_Media_Processing.persistence import add_media_persist
from tldw_Server_API.app.core.Research.discovery.selection import (
    ResolvedDiscoverySelection,
    resolve_discovery_selections,
)
from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata
from tldw_Server_API.app.core.exceptions import (
    ResearchDiscoveryBadRequestError,
    ResearchDiscoveryValidationError,
)


MAX_DISCOVERY_SELECTIONS = 5
_SELECTION_KEYS = frozenset({"result_id", "candidate_id"})
_STANDARD_IDENTIFIER_KEYS = frozenset({"doi", "pmid", "pmcid", "arxiv_id", "s2_paper_id"})


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
    if getattr(form_data, "overwrite_existing", False):
        raise ResearchDiscoveryBadRequestError("research_discovery_overwrite_not_allowed")
    return parse_research_discovery_selections(selections_json)


async def add_research_discovery_pdfs(
    *,
    background_tasks: Any,
    form_data: Any,
    files: Sequence[Any] | None,
    db: Any,
    current_user: Any,
    usage_log: Any,
    request: Any,
) -> JSONResponse:
    """Resolve selected PDFs and process new items through existing Media persistence."""
    selections = validate_research_discovery_handoff(form_data=form_data, files=files)
    resolved = resolve_discovery_selections(
        owner_user_id=str(current_user.id),
        discovery_id=form_data.research_discovery_id.strip(),
        selections=selections,
    )
    _reject_duplicate_candidate_urls(resolved)

    existing_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    blocked_pairs: set[tuple[str, str]] = set()
    pending: list[ResolvedDiscoverySelection] = []
    for item in resolved:
        pair = (item.result_id, item.candidate_id)
        if _access_is_restricted(item.access_status):
            blocked_pairs.add(pair)
            continue
        existing = _find_existing_media(db, item)
        if existing is None:
            pending.append(item)
        else:
            existing_by_pair[pair] = existing

    persisted_by_url: dict[str, dict[str, Any]] = {}
    if pending:
        urls = [item.url for item in pending]
        trusted_metadata = {item.url: _trusted_metadata(item) for item in pending}
        persistence_form = form_data.model_copy(
            update={
                "urls": urls,
                "use_cookies": False,
                "cookies": None,
                "overwrite_existing": False,
                "title": None,
                "author": None,
            }
        )
        persistence_response = await add_media_persist(
            background_tasks=background_tasks,
            form_data=persistence_form,
            files=None,
            db=db,
            current_user=current_user,
            usage_log=usage_log,
            response=None,
            request=request,
            max_download_bytes=50 * 1024 * 1024,
            allowed_download_content_types={"application/pdf"},
            trusted_source_metadata_by_url=trusted_metadata,
        )
        for result in _response_results(persistence_response):
            input_ref = result.get("input_ref")
            if isinstance(input_ref, str):
                persisted_by_url[_normalized_url(input_ref)] = result

    results: list[dict[str, Any]] = []
    for item in resolved:
        pair = (item.result_id, item.candidate_id)
        existing = existing_by_pair.get(pair)
        if pair in blocked_pairs:
            result = _policy_blocked_result(item)
        elif existing is not None:
            result = _duplicate_result(item, existing)
        else:
            result = dict(
                persisted_by_url.get(
                    _normalized_url(item.url),
                    {
                        "status": "Error",
                        "input_ref": item.url,
                        "media_type": "pdf",
                        "error": "Media persistence returned no result for selection.",
                    },
                )
            )
            result["outcome"] = _outcome_for_persisted_result(result)
            if result.get("error"):
                result["error"] = _safe_outcome_error(result["outcome"])
        result["result_id"] = item.result_id
        result["candidate_id"] = item.candidate_id
        results.append(result)

    all_succeeded = all(result.get("outcome") in {"created", "duplicate_existing"} for result in results)
    return JSONResponse(
        status_code=status.HTTP_200_OK if all_succeeded else status.HTTP_207_MULTI_STATUS,
        content={"results": results},
    )


def _normalized_url(url: str) -> str:
    return str(normalize_media_dedupe_url(url) or url).strip()


def _reject_duplicate_candidate_urls(resolved: Sequence[ResolvedDiscoverySelection]) -> None:
    normalized = [_normalized_url(item.url) for item in resolved]
    if len(set(normalized)) != len(normalized):
        raise ResearchDiscoveryValidationError("research_discovery_duplicate_candidate_url")


def _find_existing_media(db: Any, item: ResolvedDiscoverySelection) -> dict[str, Any] | None:
    checked_urls: set[str] = set()
    for url in (item.url, item.canonical_url):
        if not url:
            continue
        normalized = _normalized_url(url)
        if normalized in checked_urls:
            continue
        checked_urls.add(normalized)
        existing = db.get_media_by_url(url)
        if existing:
            return existing

    for key, value in item.identifiers.items():
        lookup_value = _media_identifier_value(key, value)
        rows, _total = db.search_by_safe_metadata(
            filters=[{"field": key, "op": "eq", "value": lookup_value}],
            match_all=False,
            page=1,
            per_page=1,
            group_by_media=True,
        )
        if rows:
            search_row = rows[0]
            media_id = search_row.get("media_id")
            if media_id is not None:
                return db.get_media_by_id(media_id) or search_row
            return search_row
    return None


def _media_identifier_value(key: str, value: str) -> str:
    if key not in _STANDARD_IDENTIFIER_KEYS:
        return value
    try:
        return str(normalize_safe_metadata({key: value}).get(key) or value)
    except ValueError:
        return value


def _access_is_restricted(access_status: str | None) -> bool:
    normalized = str(access_status or "").strip().lower()
    return normalized in {"closed", "denied", "paywalled", "restricted"}


def _trusted_metadata(item: ResolvedDiscoverySelection) -> dict[str, Any]:
    metadata = dict(item.safe_metadata)
    metadata.update(
        {
            "title": item.title,
            "author": ", ".join(item.authors),
            "url": item.canonical_url or item.url,
            "pdf_url": item.url,
            "source": f"{item.source_id}:{item.provider}",
        }
    )
    if item.license_hint:
        metadata["license"] = item.license_hint
    provider_ids: dict[str, str] = {}
    for key, value in item.identifiers.items():
        if key in _STANDARD_IDENTIFIER_KEYS:
            metadata[key] = value
        else:
            provider_ids[key] = value
    if provider_ids:
        metadata["provider_ids"] = provider_ids
    return metadata


def _response_results(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, JSONResponse):
        payload = json.loads(response.body)
    elif isinstance(response, dict):
        payload = response
    else:
        payload = {}
    results = payload.get("results") if isinstance(payload, dict) else None
    return [dict(result) for result in results if isinstance(result, dict)] if isinstance(results, list) else []


def _duplicate_result(item: ResolvedDiscoverySelection, existing: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "Success",
        "outcome": "duplicate_existing",
        "input_ref": item.url,
        "processing_source": None,
        "media_type": "pdf",
        "metadata": _trusted_metadata(item),
        "db_id": existing.get("id") or existing.get("media_id"),
        "media_uuid": existing.get("media_uuid") or existing.get("uuid"),
        "message": "Matching media already exists.",
    }


def _policy_blocked_result(item: ResolvedDiscoverySelection) -> dict[str, Any]:
    return {
        "status": "Error",
        "outcome": "policy_blocked",
        "input_ref": item.url,
        "processing_source": None,
        "media_type": "pdf",
        "metadata": _trusted_metadata(item),
        "error": "Discovery candidate access policy does not permit ingestion.",
        "db_id": None,
        "media_uuid": None,
    }


def _outcome_for_persisted_result(result: dict[str, Any]) -> str:
    status_text = str(result.get("status") or "").lower()
    error_text = " ".join(str(result.get(key) or "") for key in ("error", "message", "db_message")).lower()
    if "already exists" in error_text:
        return "duplicate_existing"
    if status_text in {"success", "warning"}:
        return "created"
    if "timeout" in error_text or "timed out" in error_text:
        return "timeout"
    if any(marker in error_text for marker in ("ssrf", "egress", "policy", "blocked")):
        return "policy_blocked"
    if any(marker in error_text for marker in ("unsupported", "content-type", "extension")):
        return "unsupported"
    return "failed"


def _safe_outcome_error(outcome: str) -> str:
    return {
        "policy_blocked": "PDF download was blocked by Media security policy.",
        "timeout": "PDF download timed out.",
        "unsupported": "Downloaded content is not a supported PDF.",
    }.get(outcome, "PDF ingestion failed.")
