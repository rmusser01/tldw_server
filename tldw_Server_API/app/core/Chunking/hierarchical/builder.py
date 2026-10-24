"""Tree mutation for hierarchical sections and leaf blocks."""

from __future__ import annotations

import re
from typing import Any

from .leaves import build_leaf_block
from .models import HierarchyTextViews, LeafChunkingContext, ResolvedHierarchyOptions


def _extract_header_title(s: str) -> str:
    """Strip the existing ATX prefix while retaining other heading syntax."""
    if s.lstrip().startswith("#"):
        return re.sub(r"^\s*#{1,6}\s+", "", s).strip()
    return s.strip()


def build_hierarchy_tree(
    context: LeafChunkingContext,
    texts: HierarchyTextViews,
    spans: list[tuple[int, int, str]],
    options: ResolvedHierarchyOptions,
) -> dict[str, Any]:
    """Build section bounds and append fresh leaf blocks for source spans."""
    text = texts.original
    root = {"kind": "root", "level": 0, "title": None, "start_offset": 0, "end_offset": len(text), "children": []}
    current_section: dict[str, Any] | None = None
    section_stack: list[dict[str, Any]] = []
    preface_section: dict[str, Any] | None = None

    def _add_block(parent: dict[str, Any], start: int, end: int, kind: str) -> None:
        block = build_leaf_block(context, texts, (start, end, kind), options)
        if block is not None:
            parent.setdefault("children", []).append(block)

    def _close_section(section: dict[str, Any] | None, end: int) -> None:
        if section is not None and section.get("end_offset") is None:
            section["end_offset"] = end

    def _ensure_preface_section(start: int) -> dict[str, Any]:
        nonlocal preface_section
        if preface_section is None:
            preface_section = {
                "kind": "section",
                "level": 1,
                "title": None,
                "start_offset": start,
                "end_offset": None,
                "children": [],
            }
            root["children"].append(preface_section)
        elif preface_section.get("start_offset") is None:
            preface_section["start_offset"] = start
        return preface_section

    for bstart, bend, bkind in spans:
        header_segment = text[bstart:bend]

        # New section on Markdown header
        if bkind == "header_atx":
            # Close previous sections (including any preface) before starting a new one
            _close_section(preface_section, bstart)
            level_match = re.match(r"^\s*(#{1,6})\s", header_segment)
            level = len(level_match.group(1)) if level_match else 1
            while section_stack and section_stack[-1].get("level", 0) >= level:
                top = section_stack.pop()
                _close_section(top, bstart)
            parent_section = section_stack[-1] if section_stack else root
            current_section = {
                "kind": "section",
                "level": level,
                "title": _extract_header_title(header_segment),
                "start_offset": bstart,
                "end_offset": None,
                "children": [],
                "source_kind": "header_atx",
            }
            parent_section.setdefault("children", []).append(current_section)
            section_stack.append(current_section)
            # Record the header itself as a block so offsets include the title text
            _add_block(current_section, bstart, bend, bkind)
        elif bkind == "bold_subsection":
            # Bold-only lines promoted to subsections under the nearest
            # non-bold section (typically the current chapter/agency).
            # We deliberately do not reset the higher-level chapter stack
            # so these remain nested inside their parent chapter.
            # Find the nearest ancestor section that was not itself created
            # from a bold-only subsection.
            target_parent: dict[str, Any] | None = None
            for sec in reversed(section_stack):
                if sec.get("kind") == "section" and sec.get("source_kind") != "bold_subsection":
                    target_parent = sec
                    break
            if target_parent is None:
                target_parent = _ensure_preface_section(bstart)

            # Close any previously-open bold subsections under this parent
            # so that new bold headings become siblings rather than nested.
            while section_stack:
                top = section_stack[-1]
                if top is target_parent or top.get("source_kind") != "bold_subsection":
                    break
                section_stack.pop()
                _close_section(top, bstart)

            parent_level = int(target_parent.get("level", 1) or 1)
            level = min(parent_level + 1, 6)
            current_section = {
                "kind": "section",
                "level": level,
                "title": _extract_header_title(header_segment),
                "start_offset": bstart,
                "end_offset": None,
                "children": [],
                "source_kind": "bold_subsection",
            }
            target_parent.setdefault("children", []).append(current_section)
            section_stack.append(current_section)
            # Record the bold heading itself as a block inside the subsection
            _add_block(current_section, bstart, bend, bkind)
        elif bkind != "blank":
            current_section = section_stack[-1] if section_stack else None
            target_parent = current_section if current_section is not None else _ensure_preface_section(bstart)
            _add_block(target_parent, bstart, bend, bkind)

    # Close tail
    while section_stack:
        _close_section(section_stack.pop(), len(text))
    _close_section(preface_section, len(text))

    return root
