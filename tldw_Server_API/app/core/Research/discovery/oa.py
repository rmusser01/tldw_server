"""Open-access candidate sanitization for research discovery."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from tldw_Server_API.app.core.Third_Party import Unpaywall

from .identity import has_unsafe_url_material, has_unsafe_url_path_material, normalize_doi
from .models import DiscoveryOACandidate


_SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "expires",
    "signature",
    "sig",
    "token",
    "x-amz-algorithm",
    "x-amz-credential",
    "x-amz-date",
    "x-amz-expires",
    "x-amz-signature",
    "x-goog-algorithm",
    "x-goog-credential",
    "x-goog-signature",
}


def sanitize_candidate_url(raw_url: str | None) -> tuple[str | None, bool]:
    """Remove signed/token-bearing URL material from a candidate URL."""
    if raw_url is None:
        return None, False

    raw_url = raw_url.strip()
    if not raw_url:
        return None, False

    parsed = urlsplit(raw_url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None, False
    if has_unsafe_url_path_material(raw_url):
        return None, True

    try:
        hostname = parsed.hostname.lower() if parsed.hostname else ""
        port = parsed.port
    except ValueError:
        return None, False
    if not hostname:
        return None, False
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = f"{hostname}:{port}" if port else hostname
    url_redacted = bool(parsed.query or parsed.fragment or parsed.username or parsed.password)
    safe_url = urlunsplit(
        (
            parsed.scheme.lower(),
            netloc,
            parsed.path,
            "",
            "",
        )
    )
    return safe_url, url_redacted


def build_resolver_reference(
    source_id: str,
    provider: str,
    doi: str | None,
    provider_ids: dict[str, str],
    candidate_type: str,
) -> str:
    """Build a non-secret reference that can be used to re-resolve a candidate."""
    normalized_doi = normalize_doi(doi or provider_ids.get("doi"))
    if normalized_doi:
        reference = {
            "candidate_type": candidate_type,
            "doi": normalized_doi,
            "provider": provider,
            "source_id": source_id,
        }
    else:
        reference = {
            "candidate_type": candidate_type,
            "provider": provider,
            "provider_ids": _safe_provider_id_digest(provider_ids),
            "source_id": source_id,
        }
    return f"resolver:{_digest(reference, length=24)}"


def build_candidate_id(
    result_fingerprint: str,
    candidate_type: str,
    provider: str,
    safe_url: str | None,
    resolver_reference: str | None,
) -> str:
    """Build an opaque candidate id from sanitized identity material only."""
    material = {
        "candidate_type": candidate_type,
        "provider": provider,
        "resolver_reference": resolver_reference,
        "result_fingerprint": result_fingerprint,
        "safe_url": safe_url,
    }
    return f"oa_candidate:{_digest(material, length=24)}"


def build_oa_candidates(
    *,
    result_fingerprint: str,
    source_id: str,
    provider: str,
    doi: str | None,
    raw_urls: Sequence[str | None],
    provider_ids: dict[str, str] | None = None,
    candidate_type: str = "pdf",
    access_status: str | None = "open",
    license_hint: str | None = None,
    content_type_hint: str | None = "application/pdf",
) -> list[DiscoveryOACandidate]:
    """Build sanitized OA candidates from provider-supplied raw URLs."""
    provider_ids = provider_ids or {}
    resolver_reference = build_resolver_reference(
        source_id=source_id,
        provider=provider,
        doi=doi,
        provider_ids=provider_ids,
        candidate_type=candidate_type,
    )
    candidates: list[DiscoveryOACandidate] = []
    seen_ids: set[str] = set()

    for raw_url in raw_urls:
        safe_url, url_redacted = sanitize_candidate_url(raw_url)
        if safe_url is None:
            continue
        candidate_id = build_candidate_id(
            result_fingerprint=result_fingerprint,
            candidate_type=candidate_type,
            provider=provider,
            safe_url=safe_url,
            resolver_reference=resolver_reference,
        )
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        candidates.append(
            DiscoveryOACandidate(
                candidate_id=candidate_id,
                candidate_type=candidate_type,
                safe_url=safe_url,
                resolver_reference=resolver_reference,
                url_redacted=url_redacted,
                requires_reresolution=url_redacted,
                provider=provider,
                access_status=access_status,
                license_hint=license_hint,
                content_type_hint=content_type_hint,
                rank=len(candidates) + 1,
                confidence=0.75 if url_redacted else 0.9,
                warnings=("url_redacted",) if url_redacted else (),
            )
        )

    return candidates


class ResearchOAResolver:
    """Resolve and sanitize OA candidates from provider URLs and Unpaywall."""

    def __init__(
        self,
        resolve_oa_pdf_fn: Callable[[str], tuple[str | None, str | None]] = Unpaywall.resolve_oa_pdf,
    ) -> None:
        self._resolve_oa_pdf_fn = resolve_oa_pdf_fn

    def resolve_for_result(
        self,
        *,
        result_fingerprint: str,
        source_id: str,
        provider: str,
        doi: str | None,
        provider_ids: dict[str, str],
        raw_urls: Sequence[str | None],
    ) -> list[DiscoveryOACandidate]:
        """Return sanitized provider candidates plus best-effort Unpaywall results."""
        candidates = build_oa_candidates(
            result_fingerprint=result_fingerprint,
            source_id=source_id,
            provider=provider,
            doi=doi,
            provider_ids=provider_ids,
            raw_urls=raw_urls,
        )

        normalized_doi = normalize_doi(doi or provider_ids.get("doi"))
        if normalized_doi is None:
            return candidates

        pdf_url: str | None = None
        warning: str | None = None
        try:
            pdf_url, warning = self._resolve_oa_pdf_fn(normalized_doi)
        except Exception:
            warning = "Unpaywall request failed."

        if pdf_url:
            candidates.extend(
                _rank_after(
                    build_oa_candidates(
                        result_fingerprint=result_fingerprint,
                        source_id="unpaywall",
                        provider="unpaywall",
                        doi=normalized_doi,
                        provider_ids={"doi": normalized_doi},
                        raw_urls=[pdf_url],
                    ),
                    start_rank=len(candidates),
                )
            )

        if warning:
            if candidates:
                candidates[0] = replace(
                    candidates[0],
                    warnings=(*candidates[0].warnings, _safe_warning(warning)),
                )
            else:
                candidates.append(
                    _warning_candidate(
                        result_fingerprint=result_fingerprint,
                        source_id="unpaywall",
                        provider="unpaywall",
                        doi=normalized_doi,
                        provider_ids={"doi": normalized_doi},
                        warning=warning,
                    )
                )

        return _dedupe_candidates(candidates)


def _warning_candidate(
    *,
    result_fingerprint: str,
    source_id: str,
    provider: str,
    doi: str | None,
    provider_ids: dict[str, str],
    warning: str,
) -> DiscoveryOACandidate:
    resolver_reference = build_resolver_reference(
        source_id=source_id,
        provider=provider,
        doi=doi,
        provider_ids=provider_ids,
        candidate_type="pdf",
    )
    return DiscoveryOACandidate(
        candidate_id=build_candidate_id(
            result_fingerprint=result_fingerprint,
            candidate_type="pdf",
            provider=provider,
            safe_url=None,
            resolver_reference=resolver_reference,
        ),
        candidate_type="pdf",
        safe_url=None,
        resolver_reference=resolver_reference,
        url_redacted=False,
        requires_reresolution=True,
        provider=provider,
        access_status=None,
        license_hint=None,
        content_type_hint="application/pdf",
        rank=1,
        confidence=0.2,
        warnings=(_safe_warning(warning),),
    )


def _rank_after(
    candidates: Iterable[DiscoveryOACandidate],
    *,
    start_rank: int,
) -> list[DiscoveryOACandidate]:
    ranked: list[DiscoveryOACandidate] = []
    for offset, candidate in enumerate(candidates, start=1):
        ranked.append(replace(candidate, rank=start_rank + offset))
    return ranked


def _dedupe_candidates(candidates: Sequence[DiscoveryOACandidate]) -> list[DiscoveryOACandidate]:
    deduped: dict[str, DiscoveryOACandidate] = {}
    for candidate in candidates:
        deduped.setdefault(candidate.candidate_id, candidate)
    return sorted(deduped.values(), key=lambda item: (item.rank, item.candidate_id))


def _safe_provider_id_digest(provider_ids: Mapping[str, str]) -> str:
    safe_ids = {
        key: value
        for key, value in provider_ids.items()
        if key.lower() not in _SENSITIVE_QUERY_KEYS
        and all(part not in key.lower() for part in ("authorization", "credential", "secret", "signature"))
        and not has_unsafe_url_material(value)
    }
    return _digest(safe_ids, length=16)


def _safe_warning(warning: str) -> str:
    warning = str(warning or "").strip()
    if not warning:
        return "OA resolver warning."
    return warning.split("?", 1)[0].split("#", 1)[0]


def _digest(value: Any, *, length: int) -> str:
    material = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:length]
