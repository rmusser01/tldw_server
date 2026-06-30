from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import regex as regex_engine


_MAX_REGEX_PATTERN_LENGTH = 1000
_REGEX_TIMEOUT_SECONDS = 0.05
_REGEX_TIMEOUT_ERROR = getattr(regex_engine, "TimeoutError", TimeoutError)
_NESTED_REGEX_QUANTIFIER_RE = re.compile(
    r"\((?:[^()\\]|\\.)*(?:\*|\+|\{\d+(?:,\d*)?\})(?:[^()\\]|\\.)*\)\s*(?:\*|\+|\{\d+(?:,\d*)?\})"
)


def _to_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if isinstance(x, (str, int, float))]
    return [str(val)]


def _normalize_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        fl = payload.get("filters")
        if isinstance(fl, list):
            return [f for f in fl if isinstance(f, dict)]
        return []
    if isinstance(payload, list):
        return [f for f in payload if isinstance(f, dict)]
    return []


def normalize_filters(payload: Any) -> list[dict[str, Any]]:
    """Return a normalized list of filters sorted by priority desc then index."""
    raw = _normalize_payload(payload)
    out: list[dict[str, Any]] = []
    for f in raw:
        t = str(f.get("type") or "").lower()
        a = str(f.get("action") or "").lower()
        if t not in {"keyword", "author", "date_range", "regex", "all"}:
            continue
        if a not in {"include", "exclude", "flag"}:
            continue
        val = f.get("value") if isinstance(f.get("value"), dict) else {}
        try:
            pr = int(f.get("priority")) if f.get("priority") is not None else 0
        except (TypeError, ValueError):
            pr = 0
        active = f.get("is_active")
        if active is None:
            active = True
        out.append({
            "type": t,
            "action": a,
            "value": val,
            "priority": pr,
            "is_active": bool(active),
            "id": f.get("id"),
        })
    # Sort by priority desc, stable for input order
    out.sort(key=lambda x: int(x.get("priority") or 0), reverse=True)
    return out


def _get_text_fields(candidate: dict[str, Any], names: Sequence[str]) -> list[str]:
    vals: list[str] = []
    for n in names:
        v = candidate.get(n)
        if v is None:
            continue
        try:
            s = str(v)
        except (TypeError, ValueError):
            continue
        if s:
            vals.append(s)
    return vals


def _match_keyword(value: dict[str, Any], candidate: dict[str, Any]) -> bool:
    keywords = [k.lower() for k in _to_list(value.get("keywords")) if str(k).strip()]
    if not keywords:
        return False
    fields = value.get("fields")
    field_names: list[str]
    if isinstance(fields, list) and fields:
        field_names = [str(x) for x in fields if str(x).strip()]
    else:
        field = value.get("field")
        if isinstance(field, str) and field.strip():
            field_names = [field.strip()]
        else:
            field_names = ["title", "summary", "content", "author"]
    haystack = "\n".join(_get_text_fields(candidate, field_names)).lower()
    if not haystack:
        return False
    mode = str(value.get("match") or "any").lower()
    if mode == "all":
        return all(k in haystack for k in keywords)
    return any(k in haystack for k in keywords)


def _match_author(value: dict[str, Any], candidate: dict[str, Any]) -> bool:
    names = [k.lower() for k in _to_list(value.get("names")) if str(k).strip()]
    if not names:
        return False
    author = str(candidate.get("author") or "").lower()
    if not author:
        return False
    mode = str(value.get("match") or "any").lower()
    if mode == "all":
        return all(n in author for n in names)
    return any(n in author for n in names)


def _looks_like_nested_quantifier(pattern: str) -> bool:
    return bool(_NESTED_REGEX_QUANTIFIER_RE.search(pattern))


def _compile_regex(pattern: str, flags: str | None) -> Any | None:
    if not pattern:
        return None
    if len(pattern) > _MAX_REGEX_PATTERN_LENGTH:
        return None
    if _looks_like_nested_quantifier(pattern):
        return None
    f = 0
    flag_text = flags if isinstance(flags, str) else None
    if flag_text is None:
        flag_text = "i"
    if isinstance(flag_text, str):
        flag_text = flag_text.lower()
        if "i" in flag_text:
            f |= regex_engine.IGNORECASE
        if "m" in flag_text:
            f |= regex_engine.MULTILINE
        if "s" in flag_text:
            f |= regex_engine.DOTALL
    try:
        return regex_engine.compile(pattern, f)
    except (regex_engine.error, TypeError, ValueError):
        return None


def _safe_regex_search(rx: Any, text: str) -> bool:
    try:
        return bool(rx.search(text, timeout=_REGEX_TIMEOUT_SECONDS))
    except (_REGEX_TIMEOUT_ERROR, TimeoutError, regex_engine.error, TypeError, ValueError):
        return False


def _match_regex(value: dict[str, Any], candidate: dict[str, Any]) -> bool:
    pattern = value.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return False
    flags = value.get("flags")
    rx = _compile_regex(pattern, flags)
    if rx is None:
        return False
    fields = value.get("fields")
    if isinstance(fields, list) and fields:
        for name in fields:
            try:
                hay = str(candidate.get(name) or "")
            except (TypeError, ValueError):
                continue
            if _safe_regex_search(rx, hay):
                return True
        return False
    field = value.get("field")
    if isinstance(field, str) and field:
        hay = str(candidate.get(field) or "")
        return _safe_regex_search(rx, hay)
    # Search across common fields
    for name in ("title", "summary", "content", "author"):
        hay = str(candidate.get(name) or "")
        if _safe_regex_search(rx, hay):
            return True
    return False


def _parse_iso(dt: str) -> datetime | None:
    text = (dt or "").strip()
    if not text:
        return None
    # Try fromisoformat with Z support
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        d = datetime.fromisoformat(text)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError):
        pass
    # Fallback email.utils
    try:
        from email.utils import parsedate_to_datetime

        d = parsedate_to_datetime(dt)
        if d is None:
            return None
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except (ImportError, TypeError, ValueError):
        return None


def _match_date_range(value: dict[str, Any], candidate: dict[str, Any]) -> bool:
    max_age_days = value.get("max_age_days")
    since_raw = value.get("since")
    until_raw = value.get("until")
    if max_age_days is None and not since_raw and not until_raw:
        return False
    if max_age_days is not None and not isinstance(max_age_days, int):
        try:
            max_age_days = int(max_age_days)
        except (TypeError, ValueError):
            return False
    pub = candidate.get("published_at")
    if not pub:
        return False
    dt = _parse_iso(str(pub))
    if not dt:
        return False
    try:
        now = datetime.now(timezone.utc)
        if max_age_days is not None:
            delta = now - dt
            if delta > timedelta(days=max_age_days):
                return False
        if since_raw:
            since_dt = _parse_iso(str(since_raw))
            if since_dt is None or dt < since_dt:
                return False
        if until_raw:
            until_dt = _parse_iso(str(until_raw))
            if until_dt is None or dt > until_dt:
                return False
        return True
    except (OverflowError, TypeError, ValueError):
        return False


def _match_all(_: dict[str, Any], __: dict[str, Any]) -> bool:
    return True


def evaluate_filters(filters: list[dict[str, Any]], candidate: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Return (decision, meta) where decision in {include, exclude, flag, None}.

    Applies active filters in priority order, short-circuiting on first match.
    Meta includes a stable key for tallying.
    """
    for idx, f in enumerate(filters):
        if not f.get("is_active", True):
            continue
        t = f.get("type")
        val = f.get("value") or {}
        matched = False
        if t == "keyword":
            matched = _match_keyword(val, candidate)
        elif t == "author":
            matched = _match_author(val, candidate)
        elif t == "regex":
            matched = _match_regex(val, candidate)
        elif t == "date_range":
            matched = _match_date_range(val, candidate)
        elif t == "all":
            matched = _match_all(val, candidate)
        if matched:
            fid = f.get("id")
            key = f"id:{fid}" if fid is not None else f"idx:{idx}"
            return f.get("action"), {"key": key, "id": fid, "type": t}
    return None, {}
