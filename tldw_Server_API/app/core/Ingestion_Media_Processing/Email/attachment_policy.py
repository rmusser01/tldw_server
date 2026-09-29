"""Minimal MIME selection for the existing nested EML extraction path."""
from __future__ import annotations

import re

DEFAULT_ATTACHMENT_MIME_ALLOWLIST = ("message/rfc822",)
_MIME_PATTERN = re.compile(r"(?:[a-z0-9!#$&^_.+-]+/(?:[a-z0-9!#$&^_.+-]+|\*)|\*/\*)")


def normalize_attachment_mime_patterns(value: list[str] | str | None) -> list[str] | None:
    """Normalize repeated/comma-separated MIME rules; preserve omitted/empty values."""
    if value is None:
        return None
    entries = [value] if isinstance(value, str) else value
    if not isinstance(entries, (list, tuple)) or any(not isinstance(item, str) for item in entries):
        raise ValueError("Attachment MIME rules must be strings")
    patterns = []
    for item in entries:
        for raw in item.split(","):
            rule = raw.strip().lower()
            if not rule or len(rule) > 128 or not _MIME_PATTERN.fullmatch(rule):
                raise ValueError("Attachment MIME rules require type/subtype, type/* or */*")
            if rule not in patterns:
                patterns.append(rule)
            if len(patterns) > 64:
                raise ValueError("At most 64 attachment MIME rules are supported")
    return patterns


def attachment_mime_skip_reason(
    content_type: str,
    filename: str | None,
    allowlist: list[str] | None,
    denylist: list[str] | None,
) -> str | None:
    """Select only supported nested EMLs, with denies preceding compatibility fallback."""
    declared_type = content_type.lower()
    eml_named = bool(filename and filename.lower().endswith(".eml"))
    candidates = [declared_type]
    if eml_named and "message/rfc822" not in candidates:
        candidates.append("message/rfc822")

    def matches(rule: str, candidate: str) -> bool:
        return rule in (candidate, "*/*", candidate.split("/", 1)[0] + "/*")

    if any(matches(rule, candidate) for rule in (denylist or []) for candidate in candidates):
        return "mime_denied"
    # Filename inference is a legacy default; explicit allowlists select declared MIME.
    allowed_candidates = candidates if allowlist is None else [declared_type]
    rules = DEFAULT_ATTACHMENT_MIME_ALLOWLIST if allowlist is None else allowlist
    if not any(matches(rule, candidate) for rule in rules for candidate in allowed_candidates):
        return "mime_not_allowed"
    if declared_type != "message/rfc822" and not eml_named:
        return "unsupported_mime"
    return None
