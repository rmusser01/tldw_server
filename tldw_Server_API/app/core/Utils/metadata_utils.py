import re
from collections.abc import Mapping
from typing import Any, Optional

_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)
_PMID_RE = re.compile(r"^\d{1,9}$")
_PMCID_RE = re.compile(r"^(?:PMC)?(\d+)$", re.IGNORECASE)
_ARXIV_RE = re.compile(r"^[a-z\-]+(\/\d{7})$|^(\d{4}\.\d{4,5})(v\d+)?$", re.IGNORECASE)


def _norm_str(v: Any) -> Optional[str]:
    """Normalize optional metadata values to non-empty strings."""
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def normalize_safe_metadata(sm: dict[str, Any]) -> dict[str, Any]:
    """Normalize and lightly validate a safe metadata dict.

    - Validates DOI format if provided
    - Normalizes PMID to digits-only string
    - Normalizes PMCID to digits-only string (strips 'PMC' prefix)
    - Normalizes arXiv ID (trims, preserves case where needed)
    - Leaves unknown keys as-is (stringified primitives)

    Raises ValueError on badly malformed DOI/PMID/PMCID.
    """
    out: dict[str, Any] = {}

    # Copy through simple fields
    for key, val in sm.items():
        if key is None:
            continue
        k = str(key).strip()
        if not k:
            continue
        out[k] = val

    # DOI
    _raw_doi = out.get("doi") if "doi" in out else out.get("DOI")
    doi = _norm_str(_raw_doi)
    if _raw_doi is not None and doi is None:
        raise ValueError(f"Invalid DOI format: {_raw_doi}")
    if doi is not None:
        if not _DOI_RE.match(doi):
            raise ValueError(f"Invalid DOI format: {doi}")
        out["doi"] = doi

    # PMID
    pmid = _norm_str(out.get("pmid") or out.get("PMID"))
    if pmid is not None:
        pmid_digits = re.sub(r"\D+", "", pmid)
        if not _PMID_RE.match(pmid_digits):
            raise ValueError(f"Invalid PMID format: {pmid}")
        out["pmid"] = pmid_digits

    # PMCID
    pmcid = _norm_str(out.get("pmcid") or out.get("PMCID"))
    if pmcid is not None:
        m = _PMCID_RE.match(pmcid)
        if not m:
            raise ValueError(f"Invalid PMCID format: {pmcid}")
        out["pmcid"] = m.group(1)

    # arXiv ID
    arx = _norm_str(out.get("arxiv_id") or out.get("arXiv") or out.get("ArXiv"))
    if arx is not None:
        # Basic normalization: trim whitespace
        arx_norm = arx.replace(" ", "")
        # Optional lightweight validation
        if not _ARXIV_RE.match(arx_norm):
            # Not fatal; keep as provided but trimmed
            arx_norm = arx_norm
        out["arxiv_id"] = arx_norm

    # S2 paper id
    s2 = _norm_str(out.get("s2_paper_id") or out.get("paperId"))
    if s2 is not None:
        out["s2_paper_id"] = s2

    return out


def merge_missing_safe_metadata(
    existing: Mapping[str, Any] | None,
    incoming: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Preserve existing metadata values and fill only missing normalized fields."""
    normalized_existing = normalize_safe_metadata(dict(existing or {}))
    normalized_incoming = normalize_safe_metadata(dict(incoming or {}))
    merged = dict(normalized_existing)

    for key, value in normalized_incoming.items():
        if value in (None, ""):
            continue
        if merged.get(key) in (None, ""):
            merged[key] = value

    return merged
