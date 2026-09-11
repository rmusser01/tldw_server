from __future__ import annotations

import re

ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

GENERIC_PHRASES = {
    "generic waf",
    "generic detection",
    "a waf or some sort of security solution",
    "a waf",
    "waf",
    "some sort of security solution",
    "security solution",
    "used",
}


def clean_text(text: str | None) -> str:
    """Remove ANSI color codes and normalise line endings."""
    if not text:
        return ""
    txt = ANSI_RE.sub("", text)
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    return txt


def parse_wafw00f_output(stdout: str, stderr: str = "") -> list[tuple[str, str | None]]:
    """
    Parse wafw00f output and return a list of detected WAFs.
    Each item is a tuple of ``(waf_name, manufacturer)``.
    """
    text = clean_text((stdout or "") + "\n" + (stderr or ""))
    results = []

    narrative_re = re.compile(
        r"(?:is|behind|protected by)\s+" r"([A-Z][\w\s-]*)(?:\s+\(in\s+)?(?:\s*WAF)?" r"(?:\s*\(([^\)]+)\))?",
        re.IGNORECASE,
    )

    for match in narrative_re.finditer(text):
        name = match.group(1).strip() if match.group(1) else None
        manufacturer = match.group(2).strip() if match.group(2) else None
        if name:
            results.append((name, manufacturer))

    filtered_results = [
        (name, manufacturer) for name, manufacturer in results if name.lower().strip() not in GENERIC_PHRASES
    ]

    if not filtered_results:
        generic_re = re.compile(
            r"generic detection|behind a waf|security solution|protected by",
            re.IGNORECASE,
        )
        if generic_re.search(text):
            return [("Generic WAF", None)]

    seen = set()
    deduped: list[tuple[str, str | None]] = []
    for name, manufacturer in filtered_results:
        key = (name.lower(), (manufacturer or "").lower())
        if key not in seen:
            seen.add(key)
            deduped.append((name, manufacturer))

    return deduped
