"""Identity normalization and dedupe helpers for research discovery results."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from .catalog import default_source_catalog
from .models import DiscoveryProvenance, DiscoveryResult


_SENSITIVE_METADATA_KEY_ALIASES = {
    "access_token",
    "accesstoken",
    "api_key",
    "apikey",
    "canonical_url",
    "canonicalurl",
    "download",
    "download_url",
    "downloadurl",
    "downloads",
    "file",
    "file_url",
    "fileurl",
    "full_text_url",
    "fulltexturl",
    "files",
    "header",
    "headers",
    "landing_page_url",
    "landingpageurl",
    "link",
    "links",
    "oaurl",
    "oa_url",
    "pdf",
    "pdf_url",
    "pdfurl",
    "raw_file",
    "raw_files",
    "rawfile",
    "rawfiles",
    "raw_urls",
    "rawurls",
    "source_url",
    "sourceurl",
    "token",
    "url",
    "url_for_pdf",
    "urlforpdf",
    "urls",
}
_SENSITIVE_METADATA_KEY_PARTS = (
    "authorization",
    "credential",
    "secret",
    "signature",
)
_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


def normalize_doi(value: Any) -> str | None:
    """Normalize DOI strings and DOI URLs to a lowercase DOI value."""
    text = _coerce_string(value)
    if text is None:
        return None

    text = unquote(text)
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^doi:\s*", "", text, flags=re.IGNORECASE)
    text = text.split("?", 1)[0].split("#", 1)[0].strip()
    match = _DOI_RE.search(text)
    if match:
        text = match.group(0)
    text = text.rstrip(".,;").lower()
    return text if text.startswith("10.") and "/" in text else None


def canonicalize_url(value: Any) -> str | None:
    """Return a deterministic, query-free URL representation for identity use."""
    text = _coerce_string(value)
    if text is None:
        return None

    parsed = urlsplit(text)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None

    hostname = parsed.hostname.lower() if parsed.hostname else ""
    if not hostname:
        return None

    try:
        port = parsed.port
    except ValueError:
        return None
    netloc = hostname
    if port and not (
        (parsed.scheme.lower() == "http" and port == 80)
        or (parsed.scheme.lower() == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"

    path = quote(unquote(parsed.path or ""), safe="/:@!$&'()*+,;=-._~")
    return urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))


def build_fingerprint(raw: dict[str, Any]) -> str:
    """Build a stable dedupe fingerprint from strongest to weakest identifiers."""
    provider_ids = _provider_ids(raw)

    doi = normalize_doi(raw.get("doi") or provider_ids.get("doi"))
    if doi:
        return f"doi:{doi}"

    pmid = _normalize_identifier(raw.get("pmid") or provider_ids.get("pmid"))
    if pmid:
        return f"pmid:{pmid}"

    pmcid = _normalize_pmcid(raw.get("pmcid") or provider_ids.get("pmcid"))
    if pmcid:
        return f"pmcid:{pmcid}"

    arxiv_id = _normalize_arxiv_id(
        raw.get("arxiv_id") or raw.get("arxiv") or provider_ids.get("arxiv_id")
    )
    if arxiv_id:
        return f"arxiv:{arxiv_id}"

    provider_fingerprint = _provider_id_fingerprint(raw, provider_ids)
    if provider_fingerprint:
        return provider_fingerprint

    canonical_url = _record_url(raw)
    if canonical_url:
        return f"url:{canonical_url}"

    title = _normalize_text(raw.get("title"))
    if title:
        hints = "|".join(
            item
            for item in (
                title,
                _author_hint(raw.get("authors")),
                _date_hint(raw),
            )
            if item
        )
        return f"title:{_digest(hints, length=24)}"

    return f"record:{_digest(safe_provider_metadata(raw), length=24)}"


def stable_result_id(fingerprint: str, primary_source_id: str, primary_provider: str) -> str:
    """Build an opaque stable result id from non-secret identity material."""
    material = {
        "fingerprint": fingerprint,
        "primary_provider": primary_provider,
        "primary_source_id": primary_source_id,
    }
    return f"discovery_result:{_digest(material, length=24)}"


def safe_provider_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Return provider metadata after removing URL, auth, and credential fields."""
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        if _is_sensitive_metadata_key(key):
            continue
        safe_value = _json_safe_value(value)
        if safe_value is not None:
            cleaned[str(key)] = safe_value
    return cleaned


def normalize_and_merge_records(
    records: list[dict[str, Any]],
    catalog_version: str,
) -> list[DiscoveryResult]:
    """Normalize raw provider records, dedupe by fingerprint, and rank results."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        fingerprint = build_fingerprint(record)
        groups.setdefault(fingerprint, []).append(record)

    results = [
        _build_discovery_result(fingerprint, group, catalog_version)
        for fingerprint, group in groups.items()
    ]
    return sorted(results, key=_result_sort_key)


def _build_discovery_result(
    fingerprint: str,
    records: list[dict[str, Any]],
    catalog_version: str,
) -> DiscoveryResult:
    ranked_records = sorted(records, key=lambda record: _record_rank_key(record, fingerprint))
    primary = ranked_records[0]
    primary_source_id = _source_id(primary)
    primary_provider = _provider(primary)
    provider_ids = _merge_provider_ids(ranked_records)
    provenance = tuple(_build_provenance(record) for record in ranked_records)
    doi = _first_normalized_doi(ranked_records, provider_ids)
    oa_candidates = _build_group_oa_candidates(fingerprint, ranked_records, doi, provider_ids)
    recommended_candidate_id = oa_candidates[0].candidate_id if oa_candidates else None
    warnings = _merge_warnings(ranked_records)
    identifier_strength = _identifier_strength(primary)
    source_priority = _source_priority(primary)
    title = _first_nonempty(ranked_records, "title") or ""

    return DiscoveryResult(
        result_id=stable_result_id(fingerprint, primary_source_id, primary_provider),
        fingerprint=fingerprint,
        primary_source_id=primary_source_id,
        primary_provider=primary_provider,
        discovery_mode=_coerce_string(primary.get("discovery_mode")) or "api",
        title=title,
        authors=_first_authors(ranked_records),
        abstract=_first_nonempty(ranked_records, "abstract"),
        doi=doi,
        pmid=_first_identifier(ranked_records, provider_ids, "pmid"),
        pmcid=_first_pmcid(ranked_records, provider_ids),
        arxiv_id=_first_arxiv_id(ranked_records, provider_ids),
        provider_ids=provider_ids,
        canonical_url=_first_url(ranked_records),
        published_at=_first_nonempty(ranked_records, "published_at")
        or _first_nonempty(ranked_records, "published")
        or _first_nonempty(ranked_records, "date"),
        updated_at=_first_nonempty(ranked_records, "updated_at")
        or _first_nonempty(ranked_records, "updated"),
        source_category=_coerce_string(primary.get("source_category") or primary.get("category")),
        oa_candidates=oa_candidates,
        recommended_candidate_id=recommended_candidate_id,
        ingest_eligible=any(candidate.safe_url for candidate in oa_candidates),
        dedupe_confidence=_dedupe_confidence(fingerprint, identifier_strength),
        ranking_signals={
            "fingerprint": fingerprint,
            "has_oa_candidate": bool(oa_candidates),
            "identifier_strength": identifier_strength,
            "primary_source_priority": source_priority,
            "title_length": len(title),
        },
        warnings=warnings,
        merged_provenance=provenance,
        safe_metadata=_merge_safe_metadata(ranked_records),
        adapter_version=_coerce_string(primary.get("adapter_version")) or "",
        catalog_version=catalog_version,
    )


def _build_provenance(record: dict[str, Any]) -> DiscoveryProvenance:
    return DiscoveryProvenance(
        source_id=_source_id(record),
        provider=_provider(record),
        discovery_mode=_coerce_string(record.get("discovery_mode")) or "api",
        provider_ids=_provider_ids(record),
        url=_record_url(record),
        source_rank=_source_rank(record),
        status=_coerce_string(record.get("status")) or "ok",
        warnings=_warnings(record),
        safe_metadata=safe_provider_metadata(record),
        adapter_version=_coerce_string(record.get("adapter_version")) or "",
    )


def _build_group_oa_candidates(
    fingerprint: str,
    records: list[dict[str, Any]],
    doi: str | None,
    merged_provider_ids: dict[str, str],
) -> tuple[Any, ...]:
    from .oa import build_oa_candidates

    by_id: dict[str, Any] = {}
    for record in records:
        raw_urls = _raw_oa_urls(record)
        if not raw_urls:
            continue
        record_provider_ids = dict(merged_provider_ids)
        record_provider_ids.update(_provider_ids(record))
        for candidate in build_oa_candidates(
            result_fingerprint=fingerprint,
            source_id=_source_id(record),
            provider=_provider(record),
            doi=doi,
            provider_ids=record_provider_ids,
            raw_urls=raw_urls,
        ):
            by_id.setdefault(candidate.candidate_id, candidate)
    return tuple(sorted(by_id.values(), key=lambda candidate: (candidate.rank, candidate.candidate_id)))


def _raw_oa_urls(record: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in ("pdf_url", "download_url", "oa_url", "full_text_url"):
        value = _coerce_string(record.get(key))
        if value:
            urls.append(value)
    raw_urls = record.get("raw_urls") or record.get("urls")
    if isinstance(raw_urls, str):
        urls.append(raw_urls)
    elif isinstance(raw_urls, Iterable) and not isinstance(raw_urls, Mapping):
        for item in raw_urls:
            value = _coerce_string(item)
            if value:
                urls.append(value)
    return _dedupe_strings(urls)


def _result_sort_key(result: DiscoveryResult) -> tuple[int, int, int, int, str]:
    signals = result.ranking_signals
    return (
        int(signals.get("primary_source_priority") or 10_000),
        -int(signals.get("identifier_strength") or 0),
        -int(signals.get("title_length") or 0),
        -int(bool(signals.get("has_oa_candidate"))),
        result.fingerprint,
    )


def _record_rank_key(record: dict[str, Any], fingerprint: str) -> tuple[int, int, int, int, str, str]:
    title = _normalize_text(record.get("title"))
    return (
        _source_priority(record),
        -_identifier_strength(record),
        -len(title),
        -int(_has_oa(record)),
        fingerprint,
        _source_id(record),
    )


def _identifier_strength(record: dict[str, Any]) -> int:
    provider_ids = _provider_ids(record)
    if normalize_doi(record.get("doi") or provider_ids.get("doi")):
        return 6
    if _normalize_identifier(record.get("pmid") or provider_ids.get("pmid")):
        return 5
    if _normalize_pmcid(record.get("pmcid") or provider_ids.get("pmcid")):
        return 5
    if _normalize_arxiv_id(record.get("arxiv_id") or provider_ids.get("arxiv_id")):
        return 4
    if provider_ids:
        return 3
    if _record_url(record):
        return 2
    if _normalize_text(record.get("title")):
        return 1
    return 0


def _source_priority(record: dict[str, Any]) -> int:
    source_rank = _source_rank(record)
    if source_rank is not None:
        return source_rank

    source_id = _source_id(record)
    try:
        return default_source_catalog(max_selected_sources=100).get_source(source_id).priority
    except KeyError:
        return 10_000


def _source_rank(record: dict[str, Any]) -> int | None:
    for key in ("source_rank", "source_priority", "rank"):
        value = record.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _dedupe_confidence(fingerprint: str, identifier_strength: int) -> float:
    if fingerprint.startswith("doi:"):
        return 1.0
    if fingerprint.startswith(("pmid:", "pmcid:")):
        return 0.95
    if fingerprint.startswith("arxiv:"):
        return 0.9
    if identifier_strength >= 3:
        return 0.8
    if fingerprint.startswith("url:"):
        return 0.7
    if fingerprint.startswith("title:"):
        return 0.55
    return 0.4


def _has_oa(record: dict[str, Any]) -> bool:
    if record.get("is_oa") is True or record.get("open_access") is True:
        return True
    return bool(_raw_oa_urls(record))


def _merge_safe_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for record in records:
        for key, value in safe_provider_metadata(record).items():
            if key not in merged or merged[key] in (None, "", [], {}):
                merged[key] = value
    return merged


def _merge_provider_ids(records: list[dict[str, Any]]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for record in records:
        for key, value in _provider_ids(record).items():
            merged.setdefault(key, value)
    return dict(sorted(merged.items()))


def _provider_ids(record: dict[str, Any]) -> dict[str, str]:
    provider_ids: dict[str, str] = {}
    raw_provider_ids = record.get("provider_ids")
    if isinstance(raw_provider_ids, Mapping):
        for key, value in raw_provider_ids.items():
            key_text = _coerce_string(key)
            value_text = _coerce_string(value)
            if key_text and value_text and not _is_sensitive_metadata_key(key_text):
                provider_ids[key_text.lower()] = value_text

    identifier_keys = (
        "doi",
        "pmid",
        "pmcid",
        "arxiv_id",
        "s2_paper_id",
        "openalex_id",
        "crossref_id",
        "pubmed_id",
        "id",
    )
    for key in identifier_keys:
        value = _coerce_string(record.get(key))
        if value:
            provider_ids.setdefault(key, value)
    return dict(sorted(provider_ids.items()))


def _provider_id_fingerprint(raw: dict[str, Any], provider_ids: dict[str, str]) -> str | None:
    filtered = {
        key: value
        for key, value in provider_ids.items()
        if key not in {"doi", "pmid", "pmcid", "arxiv_id"}
    }
    if not filtered:
        return None
    source_scope = _provider(raw) or _source_id(raw)
    key, value = sorted(filtered.items())[0]
    return f"provider:{source_scope}:{key}:{_digest(_normalize_text(value), length=16)}"


def _record_url(record: dict[str, Any]) -> str | None:
    for key in ("canonical_url", "url", "landing_page_url", "source_url"):
        url = canonicalize_url(record.get(key))
        if url:
            return url
    return None


def _first_url(records: list[dict[str, Any]]) -> str | None:
    for record in records:
        url = _record_url(record)
        if url:
            return url
    return None


def _first_normalized_doi(records: list[dict[str, Any]], provider_ids: dict[str, str]) -> str | None:
    doi = normalize_doi(provider_ids.get("doi"))
    if doi:
        return doi
    for record in records:
        doi = normalize_doi(record.get("doi"))
        if doi:
            return doi
    return None


def _first_identifier(
    records: list[dict[str, Any]],
    provider_ids: dict[str, str],
    key: str,
) -> str | None:
    value = _normalize_identifier(provider_ids.get(key))
    if value:
        return value
    for record in records:
        value = _normalize_identifier(record.get(key))
        if value:
            return value
    return None


def _first_pmcid(records: list[dict[str, Any]], provider_ids: dict[str, str]) -> str | None:
    value = _normalize_pmcid(provider_ids.get("pmcid"))
    if value:
        return value
    for record in records:
        value = _normalize_pmcid(record.get("pmcid"))
        if value:
            return value
    return None


def _first_arxiv_id(records: list[dict[str, Any]], provider_ids: dict[str, str]) -> str | None:
    value = _normalize_arxiv_id(provider_ids.get("arxiv_id"))
    if value:
        return value
    for record in records:
        value = _normalize_arxiv_id(record.get("arxiv_id") or record.get("arxiv"))
        if value:
            return value
    return None


def _first_nonempty(records: list[dict[str, Any]], key: str) -> str | None:
    for record in records:
        value = _coerce_string(record.get(key))
        if value:
            return value
    return None


def _first_authors(records: list[dict[str, Any]]) -> tuple[str, ...]:
    for record in records:
        authors = _authors(record.get("authors"))
        if authors:
            return authors
    return ()


def _authors(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        authors: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                name = (
                    _coerce_string(item.get("name"))
                    or " ".join(
                        part
                        for part in (
                            _coerce_string(item.get("given")),
                            _coerce_string(item.get("family")),
                        )
                        if part
                    )
                )
            else:
                name = _coerce_string(item)
            if name:
                authors.append(name)
        return tuple(authors)
    return ()


def _author_hint(value: Any) -> str | None:
    authors = _authors(value)
    if not authors:
        return None
    return _normalize_text(authors[0])


def _date_hint(raw: dict[str, Any]) -> str | None:
    for key in ("published_at", "published", "date", "year"):
        value = _coerce_string(raw.get(key))
        if value:
            return value[:10].lower()
    return None


def _normalize_identifier(value: Any) -> str | None:
    text = _coerce_string(value)
    if text is None:
        return None
    return text.strip().lower()


def _normalize_pmcid(value: Any) -> str | None:
    text = _coerce_string(value)
    if text is None:
        return None
    text = text.strip().upper()
    if text.isdigit():
        text = f"PMC{text}"
    return text or None


def _normalize_arxiv_id(value: Any) -> str | None:
    text = _coerce_string(value)
    if text is None:
        return None
    text = text.strip()
    text = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^arxiv:\s*", "", text, flags=re.IGNORECASE)
    text = _ARXIV_VERSION_RE.sub("", text).strip().lower()
    return text or None


def _normalize_text(value: Any) -> str:
    text = _coerce_string(value)
    if text is None:
        return ""
    return " ".join(text.lower().split())


def _source_id(record: dict[str, Any]) -> str:
    return _coerce_string(record.get("source_id")) or _provider(record) or "unknown"


def _provider(record: dict[str, Any]) -> str:
    return _coerce_string(record.get("provider")) or _coerce_string(record.get("source_id")) or "unknown"


def _warnings(record: dict[str, Any]) -> tuple[str, ...]:
    raw_warnings = record.get("warnings")
    if isinstance(raw_warnings, str):
        return (raw_warnings,)
    if isinstance(raw_warnings, Iterable):
        return tuple(
            warning for item in raw_warnings if (warning := _coerce_string(item)) is not None
        )
    return ()


def _merge_warnings(records: list[dict[str, Any]]) -> tuple[str, ...]:
    warnings: list[str] = []
    for record in records:
        warnings.extend(_warnings(record))
    return tuple(_dedupe_strings(warnings))


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _is_sensitive_metadata_key(key: Any) -> bool:
    key_variants = _metadata_key_variants(key)
    return bool(key_variants & _SENSITIVE_METADATA_KEY_ALIASES) or any(
        part in variant
        for variant in key_variants
        for part in _SENSITIVE_METADATA_KEY_PARTS
    )


def _metadata_key_variants(key: Any) -> set[str]:
    key_text = str(key).strip().lower()
    separator_normalized = re.sub(r"[^a-z0-9]+", "_", key_text).strip("_")
    compact = re.sub(r"[^a-z0-9]+", "", key_text)
    return {variant for variant in (key_text, separator_normalized, compact) if variant}


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, str):
        return None if _is_unsafe_url_like_value(value) else value
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_metadata_key(key):
                continue
            safe_item = _json_safe_value(item)
            if safe_item is not None:
                cleaned[str(key)] = safe_item
        return cleaned
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        cleaned_items = []
        for item in value:
            safe_item = _json_safe_value(item)
            if safe_item is not None:
                cleaned_items.append(safe_item)
        return cleaned_items
    return str(value)


def _is_unsafe_url_like_value(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    parsed = urlsplit(text)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return False
    return bool(parsed.query or parsed.fragment or parsed.username or parsed.password)


def _coerce_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            return None
    text = str(value).strip()
    return text or None


def _digest(value: Any, *, length: int) -> str:
    if isinstance(value, str):
        material = value
    else:
        material = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:length]
