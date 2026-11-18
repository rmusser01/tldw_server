"""Derive cited chapter representation without a model judgment or stored count."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

# Keep the Claims literal bracket-run grammar without loading its engine or DBs.
_CITATIONS = re.compile(r"\[[^\]\n]+\](?:[ \t]*(?:\r?\n[ \t]*)?(?:[,;][ \t]*(?:\r?\n[ \t]*)?)?\[[^\]\n]+\])*")
_BRACKETS = re.compile(r"\[([^\]\n]+)\]")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+|$)(.*)$")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_LIST = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_REFERENCE = re.compile(r"^ {0,3}\[([^\]\n]+)\]:")
_INLINE_CODE = re.compile(r"(`+).*?\1")
_INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\([^\)\n]*\)")
_INVENTORIES = frozenset({
    "references", "sources", "citations", "evidence", "bibliography", "works cited",
    "source inventory", "citation inventory", "chapter coverage", "evidence ids", "citation ids",
})


def _label(text: str) -> str:
    """Compare displayed labels while ignoring list/heading markup and citations."""
    text = _LIST.sub("", _BRACKETS.sub("", text))
    text = re.sub(r"^\s*[*_]*(?:Source section|Chapter|Section)(?:\s+\d+)?[*_]*\s*:\s*[*_]*\s*", "", text, flags=re.IGNORECASE)
    text = _LIST.sub("", text)
    return " ".join(re.sub(r"[^\w\s]|_", " ", text.casefold()).split())


def _evidence_owners(sections: list[Any], evidence_by_id: Mapping[str, Any] | None) -> dict[str, int]:
    """An ID belongs to its registered completed section, never its numeric prefix."""
    owners: dict[str, int] = {}
    ambiguous = set()
    for index, section in enumerate(sections):
        if not isinstance(section, Mapping) or section.get("status") != "completed":
            continue
        batches = section.get("batches")
        for batch in batches if isinstance(batches, list) else []:
            evidence = batch.get("evidence") if isinstance(batch, Mapping) else None
            for item in evidence if isinstance(evidence, list) else []:
                citation_id = item.get("id") if isinstance(item, Mapping) else None
                if not isinstance(citation_id, str) or not citation_id or (evidence_by_id is not None and citation_id not in evidence_by_id):
                    continue
                if citation_id in owners and owners[citation_id] != index:
                    ambiguous.add(citation_id)
                owners[citation_id] = index
    return {citation_id: index for citation_id, index in owners.items() if citation_id not in ambiguous}


def _prose_blocks(text: str, titles: set[str]) -> Iterator[str]:
    """Exclude obvious Markdown metadata; uncertain coverage is never inferred."""
    lines = re.sub(r"<!--[\s\S]*?(?:-->|$)", "", text).splitlines()
    fence: tuple[str, int, int] | None = None
    inventory_level: int | None = None
    current = []
    for index, line in enumerate(lines):
        quote_prefix = re.match(r"^(?:[ \t]{0,3}>[ \t]?)+", line)
        prefix = quote_prefix[0] if quote_prefix else ""
        quote_depth = prefix.count(">")
        unquoted = line[len(prefix):]
        if fence and (quote_depth < fence[1] or (fence[2] and unquoted.strip() and not unquoted.startswith(" " * fence[2]))):
            fence = None
        if fence:
            closing = _FENCE.match(unquoted[fence[2]:])
            if closing and closing[1][0] == fence[0][0] and len(closing[1]) >= len(fence[0]) and not closing[2].strip():
                fence = None
            continue
        list_prefix = _LIST.match(unquoted)
        list_indent = len(list_prefix[0]) if list_prefix else 0
        fence_match = _FENCE.match(unquoted[list_indent:])
        if fence_match:
            fence = (fence_match[1], quote_depth, list_indent)
            line = ""
        heading = _HEADING.match(line)
        following = _SETEXT.match(lines[index + 1]) if index + 1 < len(lines) else None
        if heading or (line.strip() and following):
            level = len(heading[1]) if heading else 1 if following[1][0] == "=" else 2
            title = _label(heading[2] if heading else line)
            if inventory_level is not None and level <= inventory_level:
                inventory_level = None
            if title in _INVENTORIES:
                inventory_level = level
            line = ""
        if inventory_level is not None or line.startswith(("    ", "\t")) or _REFERENCE.match(line) or _SETEXT.match(line):
            line = ""
        line = _INLINE_LINK.sub("", _INLINE_CODE.sub("", line))
        if _label(line) in titles | _INVENTORIES:
            line = ""
        if (not line.strip() or _LIST.match(line)) and current:
            yield "\n".join(current)
            current = []
        if line.strip():
            current.append(line)
    if current:
        yield "\n".join(current)


def overview_coverage(
    text: str, sections: list[Any], *, evidence_by_id: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Count completed chapters with cited prose, separately from factual support.

    Production callers may restrict saved IDs to their source-validated evidence.
    Offline callers use saved bindings and do not establish fresh source authority.
    """
    sections = sections if isinstance(sections, list) else []
    text = text if isinstance(text, str) else ""
    completed = {index: str(section.get("title") or f"Section {index + 1}")
                 for index, section in enumerate(sections)
                 if isinstance(section, Mapping) and section.get("status") == "completed"}
    owners = _evidence_owners(sections, evidence_by_id)
    references = {match[1].strip() for line in text.splitlines() if (match := _REFERENCE.match(line))}
    titles = {_label(title) for title in completed.values()}
    covered: set[int] = set()
    for block in _prose_blocks(text, titles):
        cursor = 0
        for match in _CITATIONS.finditer(block):
            prose = block[cursor:match.start()]
            ids = [token.strip().removeprefix("Source ") for group in _BRACKETS.findall(match[0]) for token in group.split(",")]
            if (any(character.isalpha() for character in prose) and _label(prose) not in titles | _INVENTORIES
                    and ids and all(citation_id in owners and citation_id not in references for citation_id in ids)):
                covered.update(owners[citation_id] for citation_id in ids)
            cursor = match.end()
    missing = [{"index": index, "title": title} for index, title in completed.items() if index not in covered]
    return {"selected_sections": len(sections), "completed_sections": len(completed), "cited_sections": len(covered),
            "missing_sections": missing, "complete": bool(completed) and not missing}
