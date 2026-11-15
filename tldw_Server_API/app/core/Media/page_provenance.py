"""Exact PDF locations in the final normalized, stored document text."""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


def page_map_for_content(metadata: Any, content: str) -> list[dict[str, Any]]:
    """Accept locations only for the exact text from which they were extracted."""
    if not isinstance(metadata, Mapping) or not isinstance(content, str):
        return []
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if metadata.get("page_map_content_hash") != digest:
        return []
    return validate_page_map(metadata.get("page_map"), len(content))


def validate_page_map(page_map: Any, content_length: int) -> list[dict[str, Any]]:
    """Validate range structure; persisted maps must use page_map_for_content."""
    if not isinstance(page_map, list) or not page_map:
        return []
    validated = []
    previous_end = 0
    for number, entry in enumerate(page_map, 1):
        if not isinstance(entry, dict):
            return []
        page, start, end = (entry.get(key) for key in ("page_number", "start_char", "end_char"))
        if (
            any(type(value) is not int for value in (page, start, end))
            or page != number
            or (number == 1 and start != 0)
            or not previous_end <= start <= end <= content_length
        ):
            return []
        clean = {"page_number": page, "start_char": start, "end_char": end}
        label = entry.get("printed_page_label")
        if isinstance(label, str) and label.strip():
            clean["printed_page_label"] = label.strip()
        validated.append(clean)
        previous_end = end
    return validated if previous_end == content_length else []


def page_spans_for_range(page_map: Any, start_char: int, end_char: int) -> list[dict[str, Any]]:
    """Clip known physical page spans to a stored-text excerpt's half-open range."""
    if type(start_char) is not int or type(end_char) is not int or not 0 <= start_char < end_char:
        return []
    if not isinstance(page_map, list):
        return []
    spans = []
    for entry in page_map:
        if not isinstance(entry, dict):
            continue
        start, end, number = (entry.get(key) for key in ("start_char", "end_char", "page_number"))
        if any(type(value) is not int for value in (start, end, number)) or number < 1:
            continue
        if start < end_char and end > start_char and start < end:
            span = {"page_number": number, "start_char": max(start, start_char), "end_char": min(end, end_char)}
            if isinstance(entry.get("printed_page_label"), str):
                span["printed_page_label"] = entry["printed_page_label"]
            spans.append(span)
    return spans


def resolve_navigation_span(
    node: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    page_map: Any,
    content_length: int,
) -> tuple[int, int] | None:
    """Resolve exact node bounds, keeping child headings inside their parent."""
    page_map = validate_page_map(page_map, content_length)

    def start_of(item: Mapping[str, Any]) -> int | None:
        target = item.get("target_start")
        if isinstance(target, float) and target.is_integer():
            target = int(target)
        if item.get("target_type") == "char_range":
            return target if type(target) is int and 0 <= target < content_length else None
        if item.get("target_type") != "page" or type(target) is not int or target < 1:
            return None
        if item.get("source") == "pdf_outline":
            matches = [entry for entry in page_map if entry["page_number"] == target]
        elif item.get("source") == "generated_toc":
            matches = [entry for entry in page_map if entry.get("printed_page_label") == str(target)]
        else:
            return None
        return matches[0]["start_char"] if len(matches) == 1 else None

    start = start_of(node)
    if start is None:
        return None
    level = node.get("level", 1)
    position = next((index for index, candidate in enumerate(nodes) if candidate.get("id") == node.get("id")), None)
    if position is None:
        return None
    following = next((candidate for candidate in nodes[position + 1 :] if candidate.get("level", 1) <= level), None)
    next_start = start_of(following) if following is not None else content_length
    if next_start is None or next_start <= start:
        return None
    if node.get("target_type") == "page":
        end = next_start
    else:
        end = node.get("target_end")
        if isinstance(end, float) and end.is_integer():
            end = int(end)
        if type(end) is not int or not start < end <= content_length:
            return None
        # A heading's original leaf span may end at its first child heading.
        children = [
            candidate
            for candidate in nodes
            if candidate.get("level", 1) > level
            and (candidate_start := start_of(candidate)) is not None
            and start < candidate_start < next_start
        ]
        if children:
            end = next_start
    return (start, end) if start < end <= content_length else None
