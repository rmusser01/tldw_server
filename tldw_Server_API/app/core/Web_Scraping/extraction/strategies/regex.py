"""Trusted static regex extraction strategy."""

from __future__ import annotations

import ipaddress
import os
import re
from typing import Any

from bs4 import BeautifulSoup

_MAX_REGEX_TOTAL_MATCHES = 200
_MAX_REGEX_MATCHES_PER_LABEL = {"number": 50}
_PII_LABELS = {"email", "phone", "credit_card"}
_TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}

# This is a fixed, reviewed catalog. Configured and generated patterns use
# Web_Scraping.safe_regex instead of this trusted stdlib-re path.
_REGEX_CATALOG: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")),
    ("phone", re.compile(r"\b\+?\d[\d\s().-]{7,}\d\b")),
    ("url", re.compile(r"\bhttps?://[^\s<>\"]+")),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("ipv6", re.compile(r"\b(?:[A-Fa-f0-9]{0,4}:){2,7}[A-Fa-f0-9]{0,4}\b")),
    (
        "uuid",
        re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"),
    ),
    ("currency", re.compile(r"[$€£¥]\s?\d+(?:,\d{3})*(?:\.\d{2})?")),
    ("percentage", re.compile(r"\b\d+(?:\.\d+)?%")),
    ("number", re.compile(r"\b\d+(?:\.\d+)?\b")),
    ("datetime", re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b")),
    ("postal_us", re.compile(r"\b\d{5}(?:-\d{4})?\b")),
    ("postal_uk", re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b", re.IGNORECASE)),
    ("hex_color", re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")),
    ("social_handle", re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,30}\b")),
    ("mac", re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.IGNORECASE)),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
]


def _regex_pii_mask_enabled() -> bool:
    return os.getenv("REGEX_PII_MASK", "").strip().lower() in _TRUTHY_VALUES


def _mask_pii_value(label: str, value: str) -> str:
    if label == "email":
        if "@" not in value:
            return "***"
        local, domain = value.split("@", 1)
        masked_local = "*" * len(local) if len(local) <= 2 else f"{local[0]}***{local[-1]}"
        return f"{masked_local}@{domain}"
    if label in {"phone", "credit_card"}:
        digits = re.sub(r"\D", "", value)
        if len(digits) <= 4:
            return "*" * len(digits)
        return f"{'*' * (len(digits) - 4)}{digits[-4:]}"
    return value


def _luhn_check(number: str) -> bool:
    digits = [int(digit) for digit in number if digit.isdigit()]
    if len(digits) < 12 or len(digits) > 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def extract_regex_entities(
    html_text: str,
    url: str,
    *,
    mask_pii: bool | None = None,
) -> dict[str, Any]:
    """Extract fixed catalog entities from HTML without evaluating untrusted regex."""
    result: dict[str, Any] = {
        "url": url,
        "title": "N/A",
        "author": "N/A",
        "content": "",
        "date": "N/A",
        "extraction_successful": False,
        "regex_matches": [],
    }
    if not html_text:
        return result

    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None
    if title:
        result["title"] = title
    text = soup.get_text(" ", strip=True)
    result["content"] = text
    if not text:
        return result

    if mask_pii is None:
        mask_pii = _regex_pii_mask_enabled()

    matches: list[dict[str, Any]] = []
    seen_spans: set[tuple[str, int, int]] = set()
    occupied: list[tuple[int, int]] = []
    total_count = 0
    per_label_counts: dict[str, int] = {}

    for label, pattern in _REGEX_CATALOG:
        per_label_limit = _MAX_REGEX_MATCHES_PER_LABEL.get(label, _MAX_REGEX_TOTAL_MATCHES)
        count = per_label_counts.get(label, 0)
        if count >= per_label_limit:
            continue
        for match in pattern.finditer(text):
            if total_count >= _MAX_REGEX_TOTAL_MATCHES or count >= per_label_limit:
                break
            start, end = match.span()
            if any(start < span_end and end > span_start for span_start, span_end in occupied):
                if label == "number":
                    continue
            value = match.group(0)
            if label == "social_handle" and "." in value:
                continue
            if label in {"ipv4", "ipv6"}:
                try:
                    ipaddress.ip_address(value)
                except ValueError:
                    continue
            if label == "credit_card" and not _luhn_check(value):
                continue
            if (label, start, end) in seen_spans:
                continue
            seen_spans.add((label, start, end))
            occupied.append((start, end))
            if mask_pii and label in _PII_LABELS:
                value = _mask_pii_value(label, value)
            matches.append({"url": url, "label": label, "value": value, "span": [start, end]})
            count += 1
            total_count += 1
        per_label_counts[label] = count
        if total_count >= _MAX_REGEX_TOTAL_MATCHES:
            break

    result["regex_matches"] = matches
    result["extraction_successful"] = bool(matches)
    return result
