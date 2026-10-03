"""Low-level text joins and grouping for hierarchical chunks."""

from typing import Any

from ..error_policy import (
    CHUNKER_NONCRITICAL_EXCEPTIONS as _CHUNKER_NONCRITICAL_EXCEPTIONS,
)

LANGUAGES_NO_SPACE = {"zh", "zh-cn", "zh-tw", "ja", "th"}


def merge_texts(
    parts: list[tuple[str, dict[str, Any]]],
    *,
    method: Any,
    default_sep: str = " ",
    kind_hint: str | None = None,
) -> str:
    """Join text parts while guaranteeing at least minimal whitespace between them."""
    if not parts:
        return ""
    combined = parts[0][0]
    prev_md = parts[0][1]

    for text_part, md in parts[1:]:
        language = None
        if isinstance(md, dict):
            language = md.get("language")
        if not language and isinstance(prev_md, dict):
            language = prev_md.get("language")

        kind = kind_hint
        if kind is None and isinstance(md, dict):
            kind = md.get("paragraph_kind")

        sep = default_sep
        if kind in {"list_unordered", "list_ordered", "table_md", "code_fence"}:
            sep = "\n"
        elif kind in {"header_atx", "hr"} or method == "structure_aware":
            sep = "\n\n"

        if (
            language
            and language.lower() in LANGUAGES_NO_SPACE
            and kind not in {"header_atx", "list_unordered", "list_ordered", "table_md", "code_fence", "hr"}
        ):
            sep = ""

        need_sep = False
        if combined and text_part:
            last_char = combined[-1]
            first_char = text_part[0]
            if (
                not last_char.isspace()
                and not first_char.isspace()
                or sep.startswith("\n")
                and not combined.endswith(sep)
                or sep == ""
                and not last_char.isspace()
            ):
                need_sep = True

            if need_sep:
                if sep:
                    eff_sep = sep
                    if sep.startswith("\n") and combined.endswith("\n"):
                        trimmed = sep[1:]
                        eff_sep = trimmed if trimmed else "\n"
                    header_like = kind == "header_atx" or kind_hint == "header_atx"
                    if (
                        eff_sep.endswith("\n")
                        and header_like
                        and (not language or language.lower() not in LANGUAGES_NO_SPACE)
                    ):
                        combined = combined.rstrip("\n")
                        combined += " "
                        combined += "\n\n"
                    else:
                        combined += eff_sep
                else:
                    # Empty separator (no-space languages) should not inject whitespace.
                    pass

        combined += text_part
        prev_md = md
    return combined


def group_items_by_elements(
    items: list[dict[str, Any]],
    *,
    method: Any,
    max_elements: int | None,
    overlap: int,
) -> list[dict[str, Any]]:
    """Group element windows while preserving overlap and offset semantics."""
    if max_elements is None or max_elements <= 0:
        return items
    if overlap < 0:
        overlap = 0
    step = max(1, max_elements - overlap)
    grouped: list[dict[str, Any]] = []
    i = 0
    n = len(items)
    while i < n:
        group = items[i : i + max_elements]
        if not group:
            break
        final_window = len(group) < max_elements
        emit = True
        if final_window and overlap > 0 and i > 0:
            try:
                if n <= i + overlap:
                    emit = False
            except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                emit = True
        if not emit:
            break
        # Concatenate texts preserving original content
        parts: list[tuple[str, dict[str, Any]]] = []
        starts: list[int] = []
        ends: list[int] = []
        for it in group:
            t = it.get("text") if isinstance(it, dict) else str(it)
            md = it.get("metadata") if isinstance(it, dict) else {}
            md_dict = dict(md) if isinstance(md, dict) else {}
            parts.append((t, md_dict))
            try:
                s = int(md.get("start_offset")) if md and md.get("start_offset") is not None else None
            except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                s = None
            try:
                e = int(md.get("end_offset")) if md and md.get("end_offset") is not None else None
            except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                e = None
            if s is not None:
                starts.append(s)
            if e is not None:
                ends.append(e)
        language_hint = None
        for _, md_part in parts:
            if md_part.get("language"):
                language_hint = md_part.get("language")
                break
        default_sep = "\n\n" if method == "structure_aware" else " "
        if language_hint and str(language_hint).lower() in LANGUAGES_NO_SPACE and method != "structure_aware":
            default_sep = ""
        agg_text = merge_texts(parts, method=method, default_sep=default_sep)
        start_off = min(starts) if starts else 0
        end_off = max(ends) if ends else start_off + len(agg_text)
        grouped.append(
            {
                "type": "text",
                "text": agg_text,
                "metadata": {
                    "method": method,
                    "start_offset": start_off,
                    "end_offset": end_off,
                    "grouped_elements": len(group),
                },
            }
        )
        if final_window:
            break
        i += step
    return grouped


def group_section_by_kind_weight(
    items: list[dict[str, Any]],
    *,
    method: Any,
    max_weight: int | None,
    overlap: int,
    weights: dict[str, Any],
) -> list[dict[str, Any]]:
    """Group contiguous items by paragraph_kind using weight budget per group.

    Does not cross kind boundaries; code_fence blocks tend to be heavier by default.
    """
    if max_weight is None or max_weight <= 0:
        return items
    if overlap < 0:
        overlap = 0
    # Clamp overlap to max_weight - 1 (no negative step)
    overlap = min(overlap, max(0, max_weight - 1))
    out_groups: list[dict[str, Any]] = []
    i = 0
    n = len(items)
    while i < n:
        # Start new group at i and keep same kind
        first = items[i]
        kind = (first.get("metadata") or {}).get("paragraph_kind") if isinstance(first, dict) else None
        budget = max_weight
        j = i
        parts: list[tuple[str, dict[str, Any]]] = []
        starts: list[int] = []
        ends: list[int] = []
        count = 0
        while j < n:
            it = items[j]
            md = it.get("metadata") if isinstance(it, dict) else {}
            ikind = md.get("paragraph_kind") if isinstance(md, dict) else None
            if ikind != kind:
                break
            w = int(weights.get(str(ikind), 1)) if isinstance(weights, dict) else 1
            if w <= 0:
                w = 1
            if w > budget and count > 0:
                break
            # Take this item
            t = it.get("text") if isinstance(it, dict) else str(it)
            md_dict = dict(md) if isinstance(md, dict) else {}
            parts.append((t, md_dict))
            try:
                s = int(md.get("start_offset")) if md and md.get("start_offset") is not None else None
            except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                s = None
            try:
                e = int(md.get("end_offset")) if md and md.get("end_offset") is not None else None
            except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                e = None
            if s is not None:
                starts.append(s)
            if e is not None:
                ends.append(e)
            budget -= w
            j += 1
            count += 1
            if budget <= 0:
                break
        if count == 0:
            # Fallback to consume one item to make progress
            j = i + 1
            it = items[i]
            t = it.get("text") if isinstance(it, dict) else str(it)
            md = it.get("metadata") if isinstance(it, dict) else {}
            md_dict = dict(md) if isinstance(md, dict) else {}
            parts = [(t, md_dict)]
            starts = [int(md.get("start_offset"))] if md and md.get("start_offset") is not None else []
            ends = [int(md.get("end_offset"))] if md and md.get("end_offset") is not None else []
            count = 1
        sep_hint = " "
        if kind in {"list_unordered", "list_ordered", "table_md", "code_fence"}:
            sep_hint = "\n"
        elif kind in {"header_atx", "hr"} or method == "structure_aware":
            sep_hint = "\n\n"
        agg_text = merge_texts(parts, method=method, default_sep=sep_hint, kind_hint=kind)
        start_off = min(starts) if starts else 0
        end_off = max(ends) if ends else start_off + len(agg_text)
        out_groups.append(
            {
                "type": "text",
                "text": agg_text,
                "metadata": {
                    "method": method,
                    "start_offset": start_off,
                    "end_offset": end_off,
                    "grouped_elements": count,
                    "group_kind": kind,
                },
            }
        )
        # Overlap semantics: step by max(1, count - overlap)
        step = max(1, count - overlap)
        i += step
    return out_groups
