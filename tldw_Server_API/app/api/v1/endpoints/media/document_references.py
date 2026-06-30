# Document References Endpoint
# Extract and enrich bibliography/references from documents
#
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit, User

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.document_references import (
    DocumentReferencesResponse,
    ReferenceEntry,
)
from tldw_Server_API.app.api.v1.utils.cache import (
    cache_response,
    get_cache_client,
    get_cached_response,
)
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    get_latest_transcription,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.document_workspace_repository import (
    DocumentWorkspaceRepository,
)

router = APIRouter(tags=["Document Workspace"])


# Maximum number of references to enrich (to avoid long response times)
MAX_ENRICHMENT_REFS = 5
# Maximum number of parsed references returned in a single response.
MAX_PARSED_REFERENCES = 100
DEFAULT_REFERENCES_PAGE_LIMIT = 50
MAX_REFERENCES_PAGE_LIMIT = 200
MAX_PARSED_REFERENCES_CAP = 5000
# Delay between external API calls (to avoid rate limiting)
SEMANTIC_SCHOLAR_DELAY = 0.35
CROSSREF_DELAY = 0.1
ARXIV_DELAY = 0.1
# Cache TTL for external enrichment lookups (seconds)
EXTERNAL_ENRICHMENT_CACHE_TTL = 3600
EXTERNAL_ENRICHMENT_COOLDOWN = 300
REFERENCE_ENRICH_EXCEPTIONS = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    asyncio.TimeoutError,
)

# Reference section detection patterns
REFERENCES_PARSER_VERSION = "11"

REFERENCE_HEADING_CORE_PATTERN = re.compile(
    r"(?i)^(?:\d+|[ivxlc]+)?(?:\.\d+)?\s*"
    r"(?:references?|bibliography|works\s+cited|literature\s+cited|cited\s+references?)"
    r"(?:\s*(?:and|&)\s*(?:notes?|citations?))?\s*:?\s*$"
)

REFERENCE_SECTION_END_CORE_PATTERN = re.compile(
    r"(?i)^(?:[A-Z](?:\.\d+)*|\d+(?:\.\d+)*)?\s*"
    r"(?:appendix|acknowledg(?:e)?ments?|supplement(?:ary|al)?|"
    r"data\s+availability|code\s+availability)\b.*$"
)

REFERENCE_TAIL_NUMBERED_PATTERN = re.compile(r"^\s*(?:\[+\d{1,6}\]|\d+[\.\)])\s+")

# DOI/arXiv extraction patterns
DOI_PATTERN = r"(?:https?://(?:dx\.)?doi\.org/)?10\.\d{4,}/[^\s\]\)>\"']+"
ARXIV_PATTERN = r"(?:arXiv[:\s]*)?(\d{4}\.\d{4,5})(?:v\d+)?"
ARXIV_OLD_PATTERN = r"(?:arXiv[:\s]*)?([a-z-]+(?:\.[A-Z]{2})?/\d{7})"
URL_PATTERN = r"https?://[^\s\]\)>\"']+"

# Year extraction pattern
# Include 18xx so classic works (e.g., Darwin-era citations) are not discarded.
YEAR_PATTERN = r"\b(18\d{2}|19\d{2}|20[0-3]\d)\b"


def _get_db_scope(db: Any) -> str:
    """Return a stable scope identifier for the active MediaDatabase."""
    return getattr(db, "db_path_str", None) or str(getattr(db, "db_path", ""))


def _build_references_cache_key(
    media_id: int,
    *,
    enrich: bool,
    user_id: str,
    db_scope: str,
    offset: int,
    limit: int,
    parse_cap: int | None,
    search_query: str | None,
    reference_index: int | None = None,
) -> str:
    scope_str = f"user:{user_id}:db:{db_scope}"
    enrich_flag = "enrich" if enrich else "basic"
    index_flag = f":idx:{reference_index}" if reference_index is not None else ""
    page_flag = f":offset:{offset}:limit:{limit}:cap:{parse_cap if parse_cap is not None else 'all'}"
    normalized_query = re.sub(r"\s+", " ", (search_query or "").strip().lower())
    query_flag = (
        f":q:{hashlib.md5(normalized_query.encode('utf-8'), usedforsecurity=False).hexdigest()[:12]}"
        if normalized_query
        else ":q:none"
    )
    return (
        f"cache:/api/v1/media/{media_id}/references:"
        f"{scope_str}:{enrich_flag}{index_flag}{page_flag}{query_flag}:v{REFERENCES_PARSER_VERSION}"
    )


def _parse_year_from_date(date_str: str | None) -> int | None:
    if not date_str:
        return None
    match = re.search(r"\b(19\d{2}|20[0-3]\d)\b", date_str)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _is_rate_limited(err: str | None) -> bool:
    if not err:
        return False
    lowered = err.lower()
    return any(token in lowered for token in ("429", "too many requests", "rate limit", "throttl"))


def _make_external_cache_key(provider: str, lookup: str) -> str:
    normalized = re.sub(r"\s+", " ", lookup).strip().lower()
    digest = hashlib.md5(normalized.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"cache:/references/enrich:{provider}:{digest}"


def _get_cached_external(key: str) -> tuple[dict[str, Any] | None, str | None] | None:
    cached = get_cached_response(key)
    if cached is None:
        return None
    _etag, payload = cached
    if isinstance(payload, dict):
        return payload.get("data"), payload.get("err")
    return None


def _set_cached_external(key: str, data: dict[str, Any] | None, err: str | None) -> None:
    cache = get_cache_client()
    if cache is None:
        return
    payload = {"data": data, "err": err}
    serialized = json.dumps(payload, default=str, separators=(",", ":"))
    cache.setex(key, EXTERNAL_ENRICHMENT_CACHE_TTL, f"external|{serialized}")


def _get_provider_cooldown_key(provider: str) -> str:
    return f"cache:/references/enrich:cooldown:{provider}"


def _is_provider_cooldown(provider: str) -> bool:
    cache = get_cache_client()
    if cache is None:
        return False
    key = _get_provider_cooldown_key(provider)
    try:
        value = cache.get(key)
        if not value:
            return False
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        until_ts = float(value)
        return time.time() < until_ts
    except REFERENCE_ENRICH_EXCEPTIONS:
        return False


def _set_provider_cooldown(provider: str) -> None:
    cache = get_cache_client()
    if cache is None:
        return
    key = _get_provider_cooldown_key(provider)
    try:
        until_ts = time.time() + EXTERNAL_ENRICHMENT_COOLDOWN
        cache.setex(key, EXTERNAL_ENRICHMENT_COOLDOWN, str(until_ts))
    except REFERENCE_ENRICH_EXCEPTIONS:
        return


def _hash_reference_content(content: str) -> str:
    """Build a stable hash for parsed-reference cache keys."""
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


def _find_reference_section(content: str) -> str | None:
    """Find and extract the references section from document content."""

    def _line_offset(line_idx: int, lines: list[str]) -> int:
        # Keep offsets deterministic without relying on regex line iterators.
        return sum(len(line) + 1 for line in lines[:line_idx])

    def _normalize_heading_line(line: str) -> str:
        line = line.strip()
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = line.replace("**", "").replace("__", "")
        line = line.strip("*_`#:- ")
        line = re.sub(r"\s+", " ", line)
        return line.strip()

    def _is_heading_candidate(line: str) -> bool:
        normalized = _normalize_heading_line(line)
        if not normalized:
            return False
        if len(normalized.split()) > 8:
            return False
        return bool(REFERENCE_HEADING_CORE_PATTERN.match(normalized))

    lines = content.split("\n")
    if not lines:
        return None

    candidate_line_indexes = [idx for idx, line in enumerate(lines) if _is_heading_candidate(line)]
    if not candidate_line_indexes:
        # Fallback: some extractions lose the "References" heading but keep a dense
        # numbered bibliography in the tail (e.g. "[41] ...", "[42] ...").
        if len(lines) < 8:
            return None
        tail_start_idx = max(min(int(len(lines) * 0.65), len(lines) - 1400), 0)
        tail_lines = lines[tail_start_idx:]
        numbered_hits: list[tuple[int, int]] = []
        for idx, line in enumerate(tail_lines):
            match = re.match(r"^\s*\[+(\d{1,6})\]", line)
            if match:
                try:
                    numbered_hits.append((idx, int(match.group(1))))
                except ValueError:
                    continue
                continue
            if REFERENCE_TAIL_NUMBERED_PATTERN.match(line):
                numbered_hits.append((idx, -1))
        if len(numbered_hits) < 6:
            return None

        unique_numbered = {num for _, num in numbered_hits if num >= 0}
        if len(unique_numbered) >= 4:
            min_num = min(unique_numbered)
            max_num = max(unique_numbered)
            span_ok = (max_num - min_num) >= 4
        else:
            span_ok = False
        if not span_ok and len(numbered_hits) < 8:
            return None

        start_line_idx = tail_start_idx + max(numbered_hits[0][0] - 2, 0)
        refs = "\n".join(lines[start_line_idx:]).strip()
        return refs or None

    # Use the last heading to avoid TOC/front-matter mentions.
    heading_line_idx = candidate_line_indexes[-1]
    start_line_idx = heading_line_idx + 1

    # Skip immediate blank lines after heading.
    while start_line_idx < len(lines) and not lines[start_line_idx].strip():
        start_line_idx += 1
    if start_line_idx >= len(lines):
        return None

    def _is_section_end_heading(line: str) -> bool:
        normalized = _normalize_heading_line(line)
        if not normalized:
            return False
        if REFERENCE_SECTION_END_CORE_PATTERN.match(normalized):
            return True
        if REFERENCE_HEADING_CORE_PATTERN.match(normalized):
            return False
        if re.search(YEAR_PATTERN, normalized):
            return False
        # Do not use a generic "title-case line" fallback here. Reference lines
        # frequently wrap into title-case continuation lines without years (for
        # example long paper titles), and the fallback causes premature truncation.
        return False

    end_line_idx = len(lines)
    for idx in range(start_line_idx + 1, len(lines)):
        if _is_section_end_heading(lines[idx]):
            end_line_idx = idx
            break

    start = _line_offset(start_line_idx, lines)
    end = _line_offset(end_line_idx, lines)
    refs = content[start:end].strip()
    return refs or None


def _split_references_with_meta(
    refs_text: str,
    *,
    max_references: int | None = MAX_PARSED_REFERENCES,
) -> tuple[list[str], int, bool]:
    """Split references section into references and return truncation metadata."""
    references: list[str] = []

    # Fix common PDF hyphenation across line breaks and normalize newlines.
    refs_text = re.sub(r"(\w)-\n(\w)", r"\1\2", refs_text)
    refs_text = refs_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not refs_text:
        return [], 0, False

    # Remove known non-reference noise lines common in PDF text extraction.
    cleaned_lines: list[str] = []
    for raw_line in refs_text.split("\n"):
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        if re.match(r"(?i)^(?:appendix|figure|table|algorithm)\b", line):
            cleaned_lines.append("")
            continue
        # Standalone page numbers.
        if re.fullmatch(r"\d{1,4}", line):
            cleaned_lines.append("")
            continue
        # News/prose lines with inline links can appear near section boundaries;
        # treat them as noise instead of candidate references.
        if (
            re.match(r"(?i)^(?:today|yesterday|tomorrow|according\s+to)\b", line)
            and re.search(YEAR_PATTERN, line)
            and re.search(URL_PATTERN, line, re.IGNORECASE)
        ):
            continue
        # Footnote/link markers like: 5 [http...](http...)
        if re.fullmatch(r"\d+\s+`?\[https?://[^\]]+\]\(https?://[^)]+\)`?", line):
            continue
        # Broken markdown-link fragments occasionally emitted by PDF extraction.
        # Example: "Learn-](https://openreview.net/...)"
        if re.fullmatch(r"[^\s\[\]]{2,120}\]\((?:https?|mailto):[^)]+\)", line, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(raw_line)
    refs_text = "\n".join(cleaned_lines).strip()
    if not refs_text:
        return [], 0, False

    def _strip_leading_bracket_label(line: str) -> tuple[str, str] | None:
        stripped = line.lstrip()
        if not stripped.startswith("["):
            return None
        depth = 0
        for idx, char in enumerate(stripped):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    remainder = stripped[idx + 1 :]
                    # Treat only "[label] text" as a label form; "[text](url)" is markdown-link form.
                    if remainder[:1].isspace():
                        return stripped[1:idx].strip(), remainder.strip()
                    return None
        return None

    def _has_explicit_reference_label(line: str) -> bool:
        stripped = line.lstrip()
        if not stripped:
            return False
        if re.match(r"^(?:\[+\d+\]|\d+[\.\)])\s+", stripped):
            return True
        bracket_label = _strip_leading_bracket_label(stripped)
        if bracket_label:
            label = bracket_label[0]
            if re.match(r"^\d{1,6}$", label):
                return True
            if re.search(r"[A-Za-z]", label):
                return True
        return False

    def _extract_markdown_link_text(line: str) -> str | None:
        match = re.match(r"^\s*\[([^\]]+)\]\((?:https?|mailto):[^)]+\)", line)
        if match:
            return match.group(1).strip()
        # Some PDFs produce nested markdown-like forms: [[astro-ph.CO].](url)
        match = re.match(r"^\s*\[\[([^\]]+)\]\.\]\((?:https?|mailto):[^)]+\)", line)
        if match:
            return match.group(1).strip()
        match = re.match(r"^\s*\[\[([^\]]+)\]\]\((?:https?|mailto):[^)]+\)", line)
        if match:
            return match.group(1).strip()
        return None

    def _is_markdown_url_fragment_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        # Canonical markdown link line.
        if re.match(r"^\[[^\]]+\]\((?:https?|mailto):[^)]+\)\s*$", stripped, flags=re.IGNORECASE):
            return True
        # Broken markdown fragment form: "Label](https://...)"
        return bool(re.match(r"^[^\s\[\]]{2,120}\]\((?:https?|mailto):[^)]+\)\s*$", stripped, flags=re.IGNORECASE))

    def _looks_like_prose_sentence_start(text: str) -> bool:
        candidate = text.strip()
        if not candidate:
            return False
        return bool(
            re.match(
                r"(?i)^(?:today|yesterday|tomorrow|this|that|these|those|here|there|"
                r"we|our|their|its|it|he|she|they|according\s+to|during|while|meanwhile)\b",
                candidate,
            )
        )

    def _is_probable_authorish_start(text: str) -> bool:
        candidate = text.strip()
        if not candidate:
            return False
        first_char = candidate[0]
        if not first_char.isalpha() or not first_char.isupper():
            return False
        if _looks_like_prose_sentence_start(candidate):
            return False
        if re.match(r"^\d{4}\b", candidate):
            return False
        if re.match(
            r"(?i)^(?:rev\.|phys\.|journal|global|ecology|proceedings|trends|communications|advances|classical|science|nature)\b",
            candidate,
        ):
            return False
        if re.match(
            r"(?i)^(?:in|proceedings|journal|vol(?:ume)?|pages?|pp\.|chapter|section)\b",
            candidate,
        ):
            return False
        # Unicode-aware surname pattern (supports diacritics like Gaztanaga).
        surname_pattern = r"[^\W\d_][\w'’.\-]*"
        # Surname + initials list: "Abazajian K. N., ..." / "Smith, J. ..."
        if re.match(rf"^{surname_pattern}(?:\s+[A-Z]\.){{1,3}},", candidate):
            return True
        if re.match(rf"^{surname_pattern},\s*[^\W\d_]", candidate):
            return True
        # "Surname et al." style
        return bool(re.match(rf"^{surname_pattern}(?:\s+et\s+al\.)\b", candidate))

    def _looks_like_new_reference(line: str) -> bool:
        if _is_markdown_url_fragment_line(line) and _extract_markdown_link_text(line) is None:
            return False
        if re.match(r"^\s*(?:\[+\d+\]|\d+[\.\)])\s+", line):
            return True

        bracket_label = _strip_leading_bracket_label(line)
        if bracket_label:
            label, rest = bracket_label
            if re.match(r"^\d{4}\b", label):
                return False
            return _is_probable_authorish_start(rest)

        markdown_label = _extract_markdown_link_text(line)
        if markdown_label:
            if re.match(r"^\d{4}\b", markdown_label):
                return False
            return _is_probable_authorish_start(markdown_label)

        return _is_probable_authorish_start(line)

    def _looks_like_fragment_line(line: str) -> bool:
        if _is_markdown_url_fragment_line(line) and _extract_markdown_link_text(line) is None:
            return True
        markdown_label = _extract_markdown_link_text(line)
        if not markdown_label:
            return False
        if _is_probable_authorish_start(markdown_label):
            return False
        if re.match(r"^\d{1,4}(?:[\s,.;:/-]|$)", markdown_label):
            return True
        return bool(
            re.match(
                r"(?i)^(?:phys\.|journal|classical|mon\.|not\.|conference|proceedings|vol\.|pages?)\b",
                markdown_label,
            )
        )

    def _looks_like_reference(text: str) -> bool:
        if not text or len(text) < 30:
            return False
        if _is_arxiv_category_fragment(text):
            return False
        lowered = text.lower().strip()
        is_numbered = bool(re.match(r"^\s*(?:\[+\d+\]|\d+[\.\)])\s+", text))
        is_labeled = _strip_leading_bracket_label(text) is not None
        markdown_label = _extract_markdown_link_text(text)

        if lowered.startswith(
            (
                "appendix",
                "acknowledgment",
                "acknowledgement",
                "copyright",
                "author contributions",
                "supplementary material",
            )
        ):
            return False
        if re.match(r"(?i)^\s*(?:figure|table|algorithm)\b", text):
            return False

        has_doi = bool(re.search(DOI_PATTERN, text, re.IGNORECASE))
        has_arxiv = bool(re.search(ARXIV_PATTERN, text, re.IGNORECASE)) or bool(
            re.search(ARXIV_OLD_PATTERN, text, re.IGNORECASE)
        )
        has_year = bool(re.search(YEAR_PATTERN, text))
        has_url = bool(re.search(URL_PATTERN, text, re.IGNORECASE))
        is_markdown_fragment = _is_markdown_url_fragment_line(text)

        if is_markdown_fragment and markdown_label is None and not (has_doi or has_arxiv):
            return False
        if not (is_numbered or is_labeled) and _looks_like_prose_sentence_start(text):
            return False

        if not (has_doi or has_arxiv or has_year):
            return False
        if (is_numbered or is_labeled) and has_year:
            return True
        if _is_probable_authorish_start(text) and has_year:
            return True
        if markdown_label and _is_probable_authorish_start(markdown_label) and has_year:
            return True
        if has_doi or has_arxiv:
            return True
        # Link-heavy astronomy/physics bibliographies often encode references as markdown links.
        return bool(has_year and has_url and len(text) > 45)

    def _looks_like_continuation_reference(text: str) -> bool:
        candidate = text.strip()
        if not candidate:
            return False
        if _is_arxiv_category_fragment(candidate):
            return True
        markdown_label = _extract_markdown_link_text(candidate)
        has_year = bool(re.search(YEAR_PATTERN, candidate))
        has_doi = bool(re.search(DOI_PATTERN, candidate, re.IGNORECASE))
        has_arxiv = bool(
            re.search(ARXIV_PATTERN, candidate, re.IGNORECASE) or re.search(ARXIV_OLD_PATTERN, candidate, re.IGNORECASE)
        )
        has_url = bool(re.search(URL_PATTERN, candidate, re.IGNORECASE))

        if markdown_label:
            label = markdown_label.strip()
            if _is_probable_authorish_start(label):
                return False
            if re.match(
                r"(?i)^(?:prints?,\s*p\.|phys\.|astropart|classical|journal|not\.\s*r|mon\.\s*not|a&a|apj|mnr|vol\.|pp\.|pages?)",
                label,
            ):
                return True
            return bool(not has_year and (has_doi or has_arxiv or has_url))

        # Non-markdown fragment: "prints, p. arXiv:..." or similar.
        return bool(re.match(r"(?i)^prints?,\s*p\.\s*arxiv:", candidate))

    # Try explicit list formats first: [1], [TAG], 1., 1)
    list_pattern = r"(?m)^\s*(?:\[+\d+\]|\[[^\n]{1,180}\]|\d+[\.\)])\s+"
    if re.search(list_pattern, refs_text):
        parts = re.split(list_pattern, refs_text)
        references = [p.strip().replace("\n", " ") for p in parts if p.strip() and len(p.strip()) > 20]
    else:
        # Fallback: paragraph blocks
        parts = re.split(r"\n\s*\n", refs_text)
        references = [p.strip().replace("\n", " ") for p in parts if p.strip() and len(p.strip()) > 20]

    # Line model pass: handles compact/link-heavy bibliographies (e.g. markdown link lines).
    lines = [ln.rstrip() for ln in refs_text.split("\n")]
    modeled_refs: list[str] = []
    current_ref = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current_ref and len(current_ref) > 20:
                modeled_refs.append(current_ref.strip())
            current_ref = ""
            continue
        if _is_markdown_url_fragment_line(line) and _extract_markdown_link_text(line) is None:
            # Broken link-label fragments are always noise; don't create standalone refs.
            continue
        markdown_label = _extract_markdown_link_text(line)
        # Treat markdown-like fragment lines as continuation when the current
        # line is likely an author prefix that has not reached a year yet.
        # Otherwise skip them to avoid fragment-only entries becoming new refs.
        if _looks_like_fragment_line(line) and current_ref:
            if (
                not re.search(YEAR_PATTERN, current_ref)
                and current_ref.rstrip().endswith(",")
                and _is_probable_authorish_start(current_ref)
            ):
                current_ref = f"{current_ref} {line}".strip()
            continue
        # Some PDF extractions split one author list across two lines where the
        # second line is a markdown-link entry containing the year.
        if (
            current_ref
            and markdown_label
            and not re.search(YEAR_PATTERN, current_ref)
            and current_ref.count(",") >= 3
            and (
                current_ref.rstrip().endswith(",")
                or re.search(r"[^\W\d_][^\W\d_'\-]*\s*$", current_ref.rstrip()) is not None
            )
            and not current_ref.rstrip().endswith(".")
            and _is_probable_authorish_start(current_ref)
            and _is_probable_authorish_start(markdown_label)
        ):
            current_ref = f"{current_ref} {line}".strip()
            continue
        line_has_explicit_label = _has_explicit_reference_label(line)
        if current_ref and _has_explicit_reference_label(current_ref) and not line_has_explicit_label:
            current_ref = f"{current_ref} {line}".strip()
            continue
        if current_ref and _looks_like_new_reference(line):
            if len(current_ref) > 20:
                modeled_refs.append(current_ref.strip())
            current_ref = line
        else:
            current_ref = (current_ref + " " + line).strip() if current_ref else line
    if current_ref and len(current_ref) > 20:
        modeled_refs.append(current_ref.strip())

    if len(modeled_refs) > len(references):
        references = modeled_refs
    elif len(modeled_refs) >= 3 and (len(references) - len(modeled_refs)) <= 1:
        # Prefer line model only when it is close to structured split count.
        # This avoids collapsing multiple valid entries in compact one-line formats.
        references = modeled_refs

    # Merge fragment-only reference entries that are continuation tails.
    merged_references: list[str] = []
    for ref in references:
        if merged_references and _looks_like_continuation_reference(ref):
            merged_references[-1] = f"{merged_references[-1]} {ref}".strip()
            continue
        merged_references.append(ref)
    references = merged_references

    # Filter out lines that do not look like references, but keep a minimum set.
    filtered = [ref for ref in references if _looks_like_reference(ref)]
    removed = [ref for ref in references if ref not in filtered]
    removed_all_obvious_noise = bool(removed) and all(
        _looks_like_prose_sentence_start(ref) or _is_markdown_url_fragment_line(ref) for ref in removed
    )
    if len(filtered) >= 3 or (filtered and removed_all_obvious_noise):
        references = filtered

    total_detected = len(references)
    if max_references is None:
        return references, total_detected, False
    effective_cap = max(1, int(max_references))
    truncated = total_detected > effective_cap
    return references[:effective_cap], total_detected, truncated


def _split_references(refs_text: str) -> list[str]:
    """Compatibility wrapper returning only references."""
    references, _total_detected, _truncated = _split_references_with_meta(refs_text)
    return references


def _extract_doi(text: str) -> str | None:
    """Extract DOI from reference text."""
    match = re.search(DOI_PATTERN, text, re.IGNORECASE)
    if match:
        doi = match.group(0)
        # Clean up the DOI
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
        # Remove trailing punctuation
        doi = doi.rstrip(".,;:)")
        return doi
    return None


def _extract_arxiv_id(text: str) -> str | None:
    """Extract arXiv ID from reference text."""
    # New format: YYMM.NNNNN
    match = re.search(ARXIV_PATTERN, text, re.IGNORECASE)
    if match:
        return match.group(1)
    # Old format: category/NNNNNNN
    match = re.search(ARXIV_OLD_PATTERN, text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _extract_url(text: str) -> str | None:
    """Extract URL from reference text."""
    # First try to find DOI URL
    doi = _extract_doi(text)
    if doi:
        return f"https://doi.org/{doi}"
    # Look for arXiv URL
    arxiv_id = _extract_arxiv_id(text)
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    # Look for any URL
    match = re.search(URL_PATTERN, text)
    if match:
        return match.group(0).rstrip(".,;:)")
    return None


def _extract_year(text: str) -> int | None:
    """Extract publication year from reference text."""
    years = re.findall(YEAR_PATTERN, text)
    if years:
        # Prefer years in parentheses (common citation format)
        for match in re.finditer(r"\((\d{4})\)", text):
            year = int(match.group(1))
            if 1900 <= year <= 2030:
                return year
        # Fall back to first year found
        return int(years[0])
    return None


def _extract_year_from_arxiv_id(arxiv_id: str | None) -> int | None:
    """Infer a year from modern arXiv IDs in YYMM.xxxxx form."""
    if not arxiv_id:
        return None
    match = re.match(r"^(\d{2})(\d{2})\.\d{4,5}$", arxiv_id)
    if not match:
        return None
    year = 2000 + int(match.group(1))
    if 2000 <= year <= 2035:
        return year
    return None


def _is_arxiv_category_fragment(text: str) -> bool:
    """Detect category-only fragments like '[[astro-ph.CO].]'."""
    candidate = (text or "").strip()
    if not candidate or len(candidate) > 100:
        return False
    candidate = re.sub(
        r"\((?:https?|mailto):[^)]+\)\s*$",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    candidate = candidate.strip("[](){} .")
    if not candidate:
        return False
    return bool(
        re.fullmatch(r"[a-z]+(?:-[a-z]+)+(?:\.[A-Z]{2})?", candidate)
        or re.fullmatch(r"[a-z]+(?:-[a-z]+)*\.[A-Z]{2}", candidate)
    )


def _normalize_reference_display_text(raw_text: str) -> str:
    """Clean markdown/link artifacts for user-facing reference text."""
    text = (raw_text or "").strip()
    if not text:
        return ""
    had_broken_link_prefix = bool(
        re.match(
            r"^\s*[^\s\[\]]{2,120}\]\((?:https?|mailto):[^)]+\)",
            text,
            flags=re.IGNORECASE,
        )
    )

    # Standard markdown link form: "[label](url)" -> "label"
    text = re.sub(
        r"\[([^\]]+)\]\((?:https?|mailto):[^)]+\)",
        lambda m: m.group(1).strip(),
        text,
        flags=re.IGNORECASE,
    )
    # Broken markdown form occasionally produced by PDF extraction:
    # "Learn-](https://...)" -> "Learn-"
    text = re.sub(
        r"\b([^\s\[\]]{2,120})\]\((?:https?|mailto):[^)]+\)",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    # Drop synthetic leading label tokens left after stripping broken links.
    if had_broken_link_prefix:
        text = re.sub(
            r"^[A-Za-z][A-Za-z0-9\-]{1,40}-\s+(?=[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3},)",
            "",
            text,
        )
    # Drop bare "(url)" tails left after extraction artifacts.
    text = re.sub(r"\((?:https?|mailto):[^)]+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip("`")


def _parse_reference_basic(raw_text: str) -> ReferenceEntry:
    """Parse a reference string into structured fields using heuristics."""
    display_text = _normalize_reference_display_text(raw_text) or raw_text
    doi = _extract_doi(raw_text)
    arxiv_id = _extract_arxiv_id(raw_text)
    url = _extract_url(raw_text)
    year = _extract_year(display_text)
    if year is None:
        year = _extract_year_from_arxiv_id(arxiv_id)

    # Try to extract title (often in quotes or after authors)
    title = None
    # Look for quoted title
    title_match = re.search(r'"([^"]{10,200})"', display_text)
    if title_match:
        title = title_match.group(1).strip()
    elif not title_match:
        # Look for title after authors (pattern: Authors, Year. Title. ...)
        title_match = re.search(r"^\s*[^.]+\.\s*(\d{4})[.\s]+([^.]{10,150})\.", display_text)
        if title_match:
            title = title_match.group(2).strip()

    # Try to extract authors (usually at the start)
    authors = None
    # Pattern: LastName, F., LastName, F., ... (Year)
    author_match = re.match(r"^((?:[A-Z][a-z]+(?:,\s*[A-Z]\.?)?(?:,?\s+(?:and\s+|&\s*)?)?)+)[\.,]", display_text)
    if author_match:
        authors = author_match.group(1).strip().rstrip(",")
    else:
        # Pattern: F. LastName, F. LastName, ...
        author_match = re.match(r"^((?:[A-Z]\.?\s*[A-Z][a-z]+(?:,?\s+(?:and\s+|&\s*)?)?)+)[\.,]", display_text)
        if author_match:
            authors = author_match.group(1).strip().rstrip(",")

    # Try to extract venue (journal/conference)
    venue = None
    # Look for common venue patterns: "In Proceedings of", "Journal of", etc.
    venue_match = re.search(
        r"(?:In\s+)?(?:Proceedings\s+of\s+)?(?:the\s+)?([A-Z][^,\.]{5,60}(?:Conference|Journal|Symposium|Workshop|Review|Letters|Transactions))",
        display_text,
        re.IGNORECASE,
    )
    if venue_match:
        venue = venue_match.group(1).strip()

    return ReferenceEntry(
        raw_text=display_text[:1000],  # Limit raw text length
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
    )


async def _enrich_with_semantic_scholar(references: list[ReferenceEntry]) -> tuple[list[ReferenceEntry], bool]:
    """Enrich references with Semantic Scholar data.

    Limits enrichment to MAX_ENRICHMENT_REFS to avoid long response times.
    Adds delays between API calls to respect Semantic Scholar rate limits.
    """
    try:
        from tldw_Server_API.app.core.Third_Party.Semantic_Scholar import (
            get_paper_details_semantic_scholar,
            search_papers_semantic_scholar,
        )
    except ImportError:
        logger.warning("Semantic Scholar module not available for enrichment")
        return references, False

    # Limit enrichment to first N references to avoid timeout
    refs_to_enrich = references[:MAX_ENRICHMENT_REFS]
    unenriched_remainder = references[MAX_ENRICHMENT_REFS:]

    enriched = []
    enrichment_performed = False
    api_call_count = 0
    rate_limited = False

    for idx, ref in enumerate(refs_to_enrich):
        enriched_ref = ref.model_copy()
        if rate_limited:
            enriched.append(enriched_ref)
            continue

        # Add delay between API calls (skip first)
        if api_call_count > 0:
            await asyncio.sleep(SEMANTIC_SCHOLAR_DELAY)

        # Try to look up by DOI first
        if ref.doi:
            try:
                cache_key = _make_external_cache_key("semantic_scholar", f"doi:{ref.doi}")
                cached = _get_cached_external(cache_key)
                if cached is not None:
                    paper_data, err = cached
                else:
                    api_call_count += 1
                    paper_data, err = await asyncio.to_thread(
                        get_paper_details_semantic_scholar,
                        f"DOI:{ref.doi}",
                    )
                    _set_cached_external(cache_key, paper_data, err)
                if paper_data and not err:
                    enriched_ref = _apply_semantic_scholar_data(enriched_ref, paper_data)
                    enrichment_performed = True
                    enriched.append(enriched_ref)
                    continue
                if _is_rate_limited(err):
                    rate_limited = True
                    _set_provider_cooldown("semantic_scholar")
                    enriched.append(enriched_ref)
                    enriched.extend(refs_to_enrich[idx + 1 :])
                    break
            except REFERENCE_ENRICH_EXCEPTIONS:
                logger.debug("Semantic Scholar DOI lookup failed")

        # Try to look up by arXiv ID
        if ref.arxiv_id:
            if api_call_count > 0:
                await asyncio.sleep(SEMANTIC_SCHOLAR_DELAY)
            try:
                cache_key = _make_external_cache_key("semantic_scholar", f"arxiv:{ref.arxiv_id}")
                cached = _get_cached_external(cache_key)
                if cached is not None:
                    paper_data, err = cached
                else:
                    api_call_count += 1
                    paper_data, err = await asyncio.to_thread(
                        get_paper_details_semantic_scholar,
                        f"ARXIV:{ref.arxiv_id}",
                    )
                    _set_cached_external(cache_key, paper_data, err)
                if paper_data and not err:
                    enriched_ref = _apply_semantic_scholar_data(enriched_ref, paper_data)
                    enrichment_performed = True
                    enriched.append(enriched_ref)
                    continue
                if _is_rate_limited(err):
                    rate_limited = True
                    _set_provider_cooldown("semantic_scholar")
                    enriched.append(enriched_ref)
                    enriched.extend(refs_to_enrich[idx + 1 :])
                    break
            except REFERENCE_ENRICH_EXCEPTIONS:
                logger.debug("Semantic Scholar arXiv lookup failed")

        # Try title search as fallback
        if ref.title:
            if api_call_count > 0:
                await asyncio.sleep(SEMANTIC_SCHOLAR_DELAY)
            try:
                cache_key = _make_external_cache_key("semantic_scholar_search", ref.title)
                cached = _get_cached_external(cache_key)
                if cached is not None:
                    search_result, err = cached
                else:
                    api_call_count += 1
                    search_result, err = await asyncio.to_thread(
                        search_papers_semantic_scholar,
                        ref.title,
                        limit=1,
                    )
                    _set_cached_external(cache_key, search_result, err)
                if search_result and not err:
                    papers = search_result.get("data", [])
                    if papers:
                        # Check if title matches reasonably
                        paper = papers[0]
                        paper_title = (paper.get("title") or "").lower()
                        ref_title = ref.title.lower()
                        # Simple similarity check
                        if ref_title[:30] in paper_title or paper_title[:30] in ref_title:
                            enriched_ref = _apply_semantic_scholar_data(enriched_ref, paper)
                            enrichment_performed = True
                if _is_rate_limited(err):
                    rate_limited = True
                    _set_provider_cooldown("semantic_scholar")
                    enriched.append(enriched_ref)
                    enriched.extend(refs_to_enrich[idx + 1 :])
                    break
            except REFERENCE_ENRICH_EXCEPTIONS:
                logger.debug("Semantic Scholar title search failed")

        enriched.append(enriched_ref)

    # Append any unenriched references beyond the limit
    enriched.extend(unenriched_remainder)

    return enriched, enrichment_performed


def _apply_semantic_scholar_data(ref: ReferenceEntry, paper: dict[str, Any]) -> ReferenceEntry:
    """Apply Semantic Scholar data to a reference entry."""
    # Update basic fields if not already set
    if not ref.title and paper.get("title"):
        ref.title = paper["title"]

    if not ref.year and paper.get("year"):
        ref.year = paper["year"]

    if not ref.venue and paper.get("venue"):
        ref.venue = paper["venue"]

    if not ref.authors and paper.get("authors"):
        authors = paper.get("authors", [])
        if authors:
            ref.authors = ", ".join([a.get("name", "") for a in authors])

    # Extract DOI from externalIds
    if not ref.doi and paper.get("externalIds"):
        external_ids = paper.get("externalIds", {})
        if external_ids.get("DOI"):
            ref.doi = external_ids["DOI"]
        if not ref.arxiv_id and external_ids.get("ArXiv"):
            ref.arxiv_id = external_ids["ArXiv"]

    # Add enriched fields
    ref.citation_count = paper.get("citationCount")
    ref.semantic_scholar_id = paper.get("paperId")

    if paper.get("openAccessPdf"):
        pdf_info = paper["openAccessPdf"]
        if isinstance(pdf_info, dict) and pdf_info.get("url"):
            ref.open_access_pdf = pdf_info["url"]

    # Update URL if not set
    if not ref.url and paper.get("url"):
        ref.url = paper["url"]

    return ref


def _needs_external_enrichment(ref: ReferenceEntry) -> bool:
    """Return True if any core metadata fields are missing."""
    return any(
        [
            not ref.title,
            not ref.authors,
            not ref.year,
            not ref.venue,
            not ref.url,
            not ref.open_access_pdf,
        ]
    )


def _enrichment_fingerprint(ref: ReferenceEntry) -> tuple[Any, ...]:
    """Return comparable reference fields to detect enrichment changes."""
    return (
        ref.title,
        ref.authors,
        ref.year,
        ref.venue,
        ref.doi,
        ref.arxiv_id,
        ref.url,
        ref.citation_count,
        ref.semantic_scholar_id,
        ref.open_access_pdf,
    )


def _apply_crossref_data(ref: ReferenceEntry, item: dict[str, Any]) -> ReferenceEntry:
    """Apply Crossref data to a reference entry."""
    if not ref.title and item.get("title"):
        ref.title = item["title"]
    if not ref.authors and item.get("authors"):
        ref.authors = item["authors"]
    if not ref.venue and item.get("journal"):
        ref.venue = item["journal"]
    if not ref.year and item.get("pub_date"):
        ref.year = _parse_year_from_date(item.get("pub_date"))
    if not ref.doi and item.get("doi"):
        ref.doi = item["doi"]
    if not ref.url and item.get("url"):
        ref.url = item["url"]
    if not ref.open_access_pdf and item.get("pdf_url"):
        ref.open_access_pdf = item["pdf_url"]
    return ref


def _apply_arxiv_data(ref: ReferenceEntry, item: dict[str, Any]) -> ReferenceEntry:
    """Apply arXiv data to a reference entry."""
    if not ref.title and item.get("title"):
        ref.title = item["title"]
    if not ref.authors and item.get("authors"):
        ref.authors = item["authors"]
    if not ref.year and item.get("published_date"):
        ref.year = _parse_year_from_date(item.get("published_date"))
    if not ref.arxiv_id and item.get("id"):
        ref.arxiv_id = item["id"]
    if not ref.url and item.get("id"):
        ref.url = f"https://arxiv.org/abs/{item['id']}"
    if not ref.open_access_pdf and item.get("pdf_url"):
        ref.open_access_pdf = item["pdf_url"]
    return ref


async def _enrich_with_crossref(references: list[ReferenceEntry]) -> tuple[list[ReferenceEntry], bool]:
    """Enrich references with Crossref metadata (DOI lookups)."""
    try:
        from tldw_Server_API.app.core.Third_Party.Crossref import get_crossref_by_doi
    except ImportError:
        logger.warning("Crossref module not available for enrichment")
        return references, False

    refs_to_enrich = references[:MAX_ENRICHMENT_REFS]
    unenriched_remainder = references[MAX_ENRICHMENT_REFS:]

    enriched: list[ReferenceEntry] = []
    enrichment_performed = False
    api_call_count = 0
    rate_limited = False

    for idx, ref in enumerate(refs_to_enrich):
        enriched_ref = ref.model_copy()
        if rate_limited:
            enriched.append(enriched_ref)
            continue
        if ref.doi and _needs_external_enrichment(enriched_ref):
            try:
                if api_call_count > 0:
                    await asyncio.sleep(CROSSREF_DELAY)
                cache_key = _make_external_cache_key("crossref", f"doi:{ref.doi}")
                cached = _get_cached_external(cache_key)
                if cached is not None:
                    item, err = cached
                else:
                    api_call_count += 1
                    item, err = await asyncio.to_thread(get_crossref_by_doi, ref.doi)
                    _set_cached_external(cache_key, item, err)
                if item and not err:
                    enriched_ref = _apply_crossref_data(enriched_ref, item)
                    enrichment_performed = True
                if _is_rate_limited(err):
                    rate_limited = True
                    _set_provider_cooldown("crossref")
                    enriched.append(enriched_ref)
                    enriched.extend(refs_to_enrich[idx + 1 :])
                    break
            except REFERENCE_ENRICH_EXCEPTIONS:
                logger.debug("Crossref lookup failed")
        enriched.append(enriched_ref)

    enriched.extend(unenriched_remainder)
    return enriched, enrichment_performed


async def _enrich_with_arxiv(references: list[ReferenceEntry]) -> tuple[list[ReferenceEntry], bool]:
    """Enrich references with arXiv metadata (ID lookups)."""
    try:
        from tldw_Server_API.app.core.Third_Party.Arxiv import get_arxiv_by_id
    except ImportError:
        logger.warning("arXiv module not available for enrichment")
        return references, False

    refs_to_enrich = references[:MAX_ENRICHMENT_REFS]
    unenriched_remainder = references[MAX_ENRICHMENT_REFS:]

    enriched: list[ReferenceEntry] = []
    enrichment_performed = False
    api_call_count = 0
    rate_limited = False

    for idx, ref in enumerate(refs_to_enrich):
        enriched_ref = ref.model_copy()
        if rate_limited:
            enriched.append(enriched_ref)
            continue
        if ref.arxiv_id and _needs_external_enrichment(enriched_ref):
            try:
                if api_call_count > 0:
                    await asyncio.sleep(ARXIV_DELAY)
                cache_key = _make_external_cache_key("arxiv", f"id:{ref.arxiv_id}")
                cached = _get_cached_external(cache_key)
                if cached is not None:
                    item, err = cached
                else:
                    api_call_count += 1
                    item, err = await asyncio.to_thread(get_arxiv_by_id, ref.arxiv_id)
                    _set_cached_external(cache_key, item, err)
                if item and not err:
                    enriched_ref = _apply_arxiv_data(enriched_ref, item)
                    enrichment_performed = True
                if _is_rate_limited(err):
                    rate_limited = True
                    _set_provider_cooldown("arxiv")
                    enriched.append(enriched_ref)
                    enriched.extend(refs_to_enrich[idx + 1 :])
                    break
            except REFERENCE_ENRICH_EXCEPTIONS:
                logger.debug("arXiv lookup failed")
        enriched.append(enriched_ref)

    enriched.extend(unenriched_remainder)
    return enriched, enrichment_performed


@router.get(
    "/{media_id:int}/references",
    status_code=status.HTTP_200_OK,
    summary="Get Document References",
    response_model=DocumentReferencesResponse,
    dependencies=[Depends(rbac_rate_limit("media.references"))],
    responses={
        200: {"description": "References retrieved (may be empty if document has no references section)"},
        404: {"description": "Media item not found"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Server error (database or extraction failure)"},
    },
)
async def get_document_references(
    media_id: int = Path(..., description="The ID of the media item"),
    enrich: bool = Query(
        False,
        description="Enrich references with external API data (citation counts, PDFs)",
    ),
    reference_index: int | None = Query(
        None,
        description="When provided, only enrich the reference at this index (0-based).",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Pagination offset in the reference list after parse-cap is applied.",
    ),
    limit: int = Query(
        DEFAULT_REFERENCES_PAGE_LIMIT,
        ge=1,
        le=MAX_REFERENCES_PAGE_LIMIT,
        description="Maximum references to return in this page.",
    ),
    parse_cap: int | None = Query(
        None,
        ge=1,
        le=MAX_PARSED_REFERENCES_CAP,
        description=("Optional cap on parsed references before pagination. " "Unset means no parser cap."),
    ),
    search: str | None = Query(
        None,
        description=(
            "Optional case-insensitive substring filter applied to the full " "reference list before pagination."
        ),
    ),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> DocumentReferencesResponse:
    """
    Extract and return references/bibliography from a document.

    This endpoint parses the document content to find a references section,
    extracts individual references, and optionally enriches them with
    external API data (citation counts, open access PDFs).

    ## Response Pattern (Graceful Degradation)

    Returns HTTP 200 with `has_references=false` when:
    - Document has no identifiable references section
    - Document is empty or has no content

    HTTP errors are reserved for actual failures:
    - **404**: Media ID does not exist
    - **429**: Rate limit exceeded
    - **500**: Database error or extraction crash

    ## Enrichment

    When `enrich=true`, the endpoint attempts to look up each
    reference using:
    - Semantic Scholar (DOI/arXiv ID lookup, title search fallback)
    - Crossref (DOI metadata)
    - arXiv (ID metadata)

    Enrichment adds:
    - Citation counts
    - Open access PDF links
    - Semantic Scholar paper IDs
    - Missing metadata (title, authors, year)

    Enrichment is best-effort and gracefully degrades if external APIs fail.
    Enrichment is limited to the first 5 references on the returned page to avoid
    long response times.
    """
    user_id = str(getattr(current_user, "id", "anonymous"))
    db_scope = _get_db_scope(db)
    cache_key = _build_references_cache_key(
        media_id,
        enrich=enrich,
        user_id=user_id,
        db_scope=db_scope,
        offset=offset,
        limit=limit,
        parse_cap=parse_cap,
        search_query=search,
        reference_index=reference_index,
    )
    cached = get_cached_response(cache_key)
    if cached is not None:
        _etag, payload = cached
        if "pagination" not in payload:
            payload = dict(payload)
            cached_refs = payload.get("references") or []
            cached_count = int(payload.get("returned_count") or len(cached_refs))
            cached_limit = int(payload.get("limit") or cached_count or 1)
            raw_total_available = payload.get("total_available")
            payload["pagination"] = build_offset_pagination_meta(
                total=int(raw_total_available if raw_total_available is not None else cached_count or 0),
                limit=max(1, cached_limit),
                offset=int(payload.get("offset") or 0),
                count=cached_count,
            )
        logger.debug("Returning cached references for media_id={}", media_id)
        return DocumentReferencesResponse(**payload)

    logger.debug(
        "Extracting references for media_id={}, user_id={}, enrich={}",
        media_id,
        getattr(current_user, "id", "?"),
        enrich,
    )

    # 1. Verify media item exists
    try:
        media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    except Exception as e:
        logger.error("Database error fetching media item")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while fetching media item",
        ) from e

    if not media:
        logger.warning(
            "Media not found for references extraction: {} (user: {})",
            media_id,
            getattr(current_user, "id", "?"),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found or is inactive/trashed",
        )

    # 2. Get document content
    content = str(media.get("content") or "")
    if not content.strip():
        content = get_latest_transcription(db, media_id) or ""

    # Normalize escaped and platform-specific newlines
    if "\\n" in content and "\n" not in content:
        content = content.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    content = content.strip()
    if not content:
        logger.debug("No content available for media_id={}", media_id)
        response = DocumentReferencesResponse(
            media_id=media_id,
            has_references=False,
            references=[],
            enrichment_source=None,
            enriched_count=0,
            enrichment_limited=False,
            total_detected=0,
            truncated=False,
            offset=offset,
            limit=limit,
            returned_count=0,
            total_available=0,
            has_more=False,
            next_offset=None,
        )
        cache_response(cache_key, response.model_dump(), media_id=media_id)
        return response

    # 3. Find references section
    refs_section = _find_reference_section(content)
    if not refs_section:
        logger.debug("No references section found in media_id={}", media_id)
        response = DocumentReferencesResponse(
            media_id=media_id,
            has_references=False,
            references=[],
            enrichment_source=None,
            enriched_count=0,
            enrichment_limited=False,
            total_detected=0,
            truncated=False,
            offset=offset,
            limit=limit,
            returned_count=0,
            total_available=0,
            has_more=False,
            next_offset=None,
        )
        cache_response(cache_key, response.model_dump(), media_id=media_id)
        return response

    # 4. Parse individual references (with DB-backed parsed-reference cache)
    repo = DocumentWorkspaceRepository.from_media_db(db)
    refs_hash = _hash_reference_content(refs_section)
    try:
        cached_parsed = repo.get_parsed_references_cache(
            media_id=media_id,
            user_id=user_id,
            parser_version=REFERENCES_PARSER_VERSION,
            content_hash=refs_hash,
        )
    except Exception:
        logger.debug("Failed loading parsed references cache")
        cached_parsed = None
    if cached_parsed is not None:
        raw_refs_all, total_detected = cached_parsed
    else:
        raw_refs_all, total_detected, _was_truncated = _split_references_with_meta(
            refs_section,
            max_references=None,
        )
        if raw_refs_all:
            try:
                repo.upsert_parsed_references_cache(
                    media_id=media_id,
                    user_id=user_id,
                    parser_version=REFERENCES_PARSER_VERSION,
                    content_hash=refs_hash,
                    references=raw_refs_all,
                    total_detected=total_detected,
                    updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
            except Exception:
                logger.debug("Failed saving parsed references cache")

    if not raw_refs_all:
        logger.debug("No individual references parsed from media_id={}", media_id)
        response = DocumentReferencesResponse(
            media_id=media_id,
            has_references=False,
            references=[],
            enrichment_source=None,
            enriched_count=0,
            enrichment_limited=False,
            total_detected=0,
            truncated=False,
            offset=offset,
            limit=limit,
            returned_count=0,
            total_available=0,
            has_more=False,
            next_offset=None,
        )
        cache_response(cache_key, response.model_dump(), media_id=media_id)
        return response

    # 5. Apply parse cap, filter, and paginate raw references
    if parse_cap is None:
        capped_refs = raw_refs_all
    else:
        capped_refs = raw_refs_all[:parse_cap]

    truncated = len(raw_refs_all) > len(capped_refs)
    normalized_search = re.sub(r"\s+", " ", (search or "").strip().lower())
    if normalized_search:
        filtered_refs = [
            ref for ref in capped_refs if normalized_search in _normalize_reference_display_text(ref).lower()
        ]
    else:
        filtered_refs = capped_refs

    total_available = len(filtered_refs)
    page_start = min(offset, total_available)
    page_end = min(page_start + limit, total_available)
    page_raw_refs = filtered_refs[page_start:page_end]

    if not page_raw_refs:
        response = DocumentReferencesResponse(
            media_id=media_id,
            has_references=total_available > 0,
            references=[],
            enrichment_source=None,
            enriched_count=0,
            enrichment_limited=False,
            total_detected=total_detected,
            truncated=truncated,
            offset=offset,
            limit=limit,
            returned_count=0,
            total_available=total_available,
            has_more=False,
            next_offset=None,
            pagination=build_offset_pagination_meta(
                total=total_available,
                limit=limit,
                offset=offset,
                count=0,
            ),
        )
        cache_response(cache_key, response.model_dump(), media_id=media_id)
        return response

    # 6. Parse each reference in current page
    references = [_parse_reference_basic(ref) for ref in page_raw_refs]
    logger.debug(
        "Parsed page references media_id={} total_detected={} total_available={} page_size={}",
        media_id,
        total_detected,
        total_available,
        len(references),
    )

    # 7. Enrich with external APIs if requested
    enrichment_sources: set[str] = set()
    before_enrichment = [_enrichment_fingerprint(ref) for ref in references]
    enrichment_limited = bool(enrich and reference_index is None and len(references) > MAX_ENRICHMENT_REFS)
    if enrich and references:
        try:
            if reference_index is not None:
                if reference_index < 0 or reference_index >= total_available:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="reference_index out of range",
                    )
                target_raw = filtered_refs[reference_index]
                target_ref = _parse_reference_basic(target_raw)
                enriched_refs = [target_ref]
                if not _is_provider_cooldown("semantic_scholar"):
                    enriched_refs, enriched = await _enrich_with_semantic_scholar(enriched_refs)
                    if enriched:
                        enrichment_sources.add("semantic_scholar")
                if not _is_provider_cooldown("crossref"):
                    enriched_refs, enriched = await _enrich_with_crossref(enriched_refs)
                    if enriched:
                        enrichment_sources.add("crossref")
                if not _is_provider_cooldown("arxiv"):
                    enriched_refs, enriched = await _enrich_with_arxiv(enriched_refs)
                    if enriched:
                        enrichment_sources.add("arxiv")
                if page_start <= reference_index < page_end:
                    references[reference_index - page_start] = enriched_refs[0]
            else:
                if not _is_provider_cooldown("semantic_scholar"):
                    references, enriched = await _enrich_with_semantic_scholar(references)
                    if enriched:
                        enrichment_sources.add("semantic_scholar")
                        logger.debug(
                            "Enriched references with Semantic Scholar for media_id={}",
                            media_id,
                        )
                if not _is_provider_cooldown("crossref"):
                    references, enriched = await _enrich_with_crossref(references)
                    if enriched:
                        enrichment_sources.add("crossref")
                        logger.debug(
                            "Enriched references with Crossref for media_id={}",
                            media_id,
                        )
                if not _is_provider_cooldown("arxiv"):
                    references, enriched = await _enrich_with_arxiv(references)
                    if enriched:
                        enrichment_sources.add("arxiv")
                        logger.debug(
                            "Enriched references with arXiv for media_id={}",
                            media_id,
                        )
        except REFERENCE_ENRICH_EXCEPTIONS:
            logger.warning("Failed to enrich references")
            # Continue without enrichment

    enriched_count = sum(
        1
        for idx, ref in enumerate(references)
        if idx < len(before_enrichment) and _enrichment_fingerprint(ref) != before_enrichment[idx]
    )
    enrichment_source = ",".join(sorted(enrichment_sources)) if enrichment_sources else None
    returned_count = len(references)
    has_more = page_end < total_available
    response = DocumentReferencesResponse(
        media_id=media_id,
        has_references=total_available > 0,
        references=references,
        enrichment_source=enrichment_source,
        enriched_count=enriched_count,
        enrichment_limited=enrichment_limited,
        total_detected=total_detected,
        truncated=truncated,
        offset=offset,
        limit=limit,
        returned_count=returned_count,
        total_available=total_available,
        has_more=has_more,
        next_offset=page_end if has_more else None,
        pagination=build_offset_pagination_meta(
            total=total_available,
            limit=limit,
            offset=offset,
            count=returned_count,
        ),
    )
    cache_response(cache_key, response.model_dump(), media_id=media_id)
    return response


__all__ = ["router"]
