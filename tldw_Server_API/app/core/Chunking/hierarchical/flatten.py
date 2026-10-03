"""Traversal and output assembly for public-validated hierarchy trees."""

from collections.abc import Callable
from typing import Any

from .grouping import group_items_by_elements, group_section_by_kind_weight, merge_texts


def flatten_tree(
    tree: dict[str, Any],
    normalize_chunk_type: Callable[[Any], str | None],
) -> list[dict[str, Any]]:
    """Flatten a dictionary tree using only the supplied type normalizer."""
    root = tree.get("root") or {"children": tree.get("blocks", [])}
    method = tree.get("method")
    # Elements-per-chunk semantics for structure_aware grouping
    sa_max = tree.get("max_size") if isinstance(tree.get("max_size"), int) else None
    sa_ovl = tree.get("overlap") if isinstance(tree.get("overlap"), int) else 0
    out: list[dict[str, Any]] = []

    def _append_with_titles(items: list[dict[str, Any]], titles: list[str]):
        for ch in items:
            txt = ch.get("text") if isinstance(ch, dict) else str(ch)
            md = dict(ch.get("metadata") or {}) if isinstance(ch, dict) else {}
            md["ancestry_titles"] = titles
            if titles:
                md["section_path"] = " > ".join(titles)
            raw_chunk_type = md.get("chunk_type")
            if raw_chunk_type is None or raw_chunk_type == "":
                raw_chunk_type = md.get("paragraph_kind")
            normalized_chunk_type = normalize_chunk_type(raw_chunk_type)
            if normalized_chunk_type:
                md["chunk_type"] = normalized_chunk_type
            out.append({"text": txt, "metadata": md})

    def _gather_section_items(section_node: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        header_buffer: list[dict[str, Any]] = []

        def _flush_header_buffer(target_item: dict[str, Any]) -> dict[str, Any]:
            """Merge buffered headers into the provided item without increasing element count."""
            nonlocal header_buffer
            if not header_buffer:
                return target_item
            parts: list[tuple[str, dict[str, Any]]] = []
            starts: list[int] = []
            ends: list[int] = []
            for h in header_buffer:
                txt = h.get("text") if isinstance(h, dict) else str(h)
                md_h = h.get("metadata") if isinstance(h, dict) else {}
                s = md_h.get("start_offset")
                e = md_h.get("end_offset")
                parts.append((txt, dict(md_h) if isinstance(md_h, dict) else {}))
                if isinstance(s, int):
                    starts.append(s)
                if isinstance(e, int):
                    ends.append(e)
            header_buffer = []

            t_txt = target_item.get("text") if isinstance(target_item, dict) else str(target_item)
            md_target = dict(target_item.get("metadata") or {}) if isinstance(target_item, dict) else {}
            s_t = md_target.get("start_offset")
            e_t = md_target.get("end_offset")
            if isinstance(s_t, int):
                starts.append(s_t)
            if isinstance(e_t, int):
                ends.append(e_t)
            parts.append((t_txt, md_target))
            merged_start = min(starts) if starts else s_t
            merged_end = max(ends) if ends else e_t

            merged_item = {
                "type": target_item.get("type", "text"),
                "text": merge_texts(parts, method=method, default_sep="\n\n", kind_hint="header_atx"),
                "metadata": md_target,
            }
            if merged_start is not None:
                merged_item["metadata"]["start_offset"] = merged_start
            if merged_end is not None:
                merged_item["metadata"]["end_offset"] = merged_end
            # Preserve paragraph kind (defaulting to target item) and flag header inclusion
            pk = md_target.get("paragraph_kind")
            if pk is not None:
                merged_item["metadata"]["paragraph_kind"] = pk
            merged_item["metadata"]["has_section_header"] = True
            return merged_item

        for child in section_node.get("children") or []:
            if not isinstance(child, dict):
                continue
            for ch in child.get("chunks") or []:
                if not isinstance(ch, dict):
                    continue
                md = ch.get("metadata") or {}
                paragraph_kind = md.get("paragraph_kind")
                if paragraph_kind == "header_atx":
                    header_buffer.append(ch)
                    continue
                if header_buffer:
                    merged = _flush_header_buffer(ch)
                    items.append(merged)
                else:
                    items.append(ch)

        if header_buffer:
            # Section with header but no following content: keep header as-is
            items.extend(header_buffer)

        return items

    def walk(node: dict[str, Any], titles: list[str]):
        kind = node.get("kind")
        if kind == "section":
            title = str(node.get("title") or "").strip()
            titles = titles + ([title] if title else [])
        # For structure_aware, group elements per section using max_size/overlap
        if kind == "section" and method == "structure_aware" and isinstance(sa_max, int) and sa_max > 0:
            section_items = _gather_section_items(node)
            # Optional grouping configuration carried in tree
            grouping_cfg = tree.get("grouping") if isinstance(tree.get("grouping"), dict) else {}
            by_kind = bool(grouping_cfg.get("by_kind", False))
            weights = (
                grouping_cfg.get("element_weights")
                if isinstance(grouping_cfg.get("element_weights"), dict)
                else {
                    "paragraph": 1,
                    "list_unordered": 1,
                    "list_ordered": 1,
                    "table_md": 2,
                    "code_fence": 3,
                }
            )
            if by_kind:
                grouped_items = group_section_by_kind_weight(
                    section_items,
                    method=method,
                    max_weight=sa_max,
                    overlap=sa_ovl if isinstance(sa_ovl, int) else 0,
                    weights=weights,
                )
            else:
                grouped_items = group_items_by_elements(
                    section_items,
                    method=method,
                    max_elements=sa_max,
                    overlap=sa_ovl if isinstance(sa_ovl, int) else 0,
                )
            _append_with_titles(grouped_items, titles)
            for child in node.get("children") or []:
                if isinstance(child, dict) and child.get("kind") == "section":
                    walk(child, titles)
            # Do not descend into children again to avoid duplicating content
            return
        # Default behavior: emit this node's chunks as-is
        for ch in node.get("chunks") or []:
            _append_with_titles([ch], titles)
        for child in node.get("children") or []:
            if isinstance(child, dict):
                walk(child, titles)

    walk(root, [])
    # Normalize chunk_index/total
    for i, item in enumerate(out):
        md = item.setdefault("metadata", {})
        md.setdefault("chunk_index", i + 1)
        md.setdefault("total_chunks", len(out))
    return out
