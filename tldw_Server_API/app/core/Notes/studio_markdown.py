"""Deterministic Markdown rendering and normalization helpers for Notes Studio documents."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from tldw_Server_API.app.core.Sync.v2.notes_moodboard_studio_contract import (
    StudioSectionsV1,
)

NOTE_STUDIO_RENDER_VERSION = 1
_CUE_SECTION_TITLES = {"cue", "cues", "key questions", "questions"}
_SUMMARY_SECTION_TITLES = {"summary", "summaries"}
_DEFAULT_SECTION_TITLES = {
    "cue": "Key Questions",
    "notes": "Notes",
    "summary": "Summary",
}


def stable_content_hash(content: str) -> str:
    """Return a stable sha256 hash with a Notes Studio prefix."""
    normalized = str(content or "").replace("\r\n", "\n").replace("\r", "\n")
    return f"sha256:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def render_studio_markdown(payload: Mapping[str, Any]) -> str:
    """Render canonical Studio payload data into deterministic Markdown."""
    meta = payload.get("meta") if isinstance(payload, Mapping) else {}
    sections = payload.get("sections") if isinstance(payload, Mapping) else []

    title = str((meta or {}).get("title") or "Untitled Study Notes").strip() or "Untitled Study Notes"
    markdown_parts: list[str] = [f"# {title}"]

    for section in _iter_sections(sections):
        section_title = str(section.get("title") or "Section").strip() or "Section"
        kind = str(section.get("kind") or "").strip().lower()
        block_lines: list[str] = [f"## {section_title}", ""]
        if kind == "cue":
            items = section.get("items") or []
            bullet_items = [str(item).strip() for item in items if str(item).strip()]
            if bullet_items:
                block_lines.extend(f"- {item}" for item in bullet_items)
            else:
                block_lines.append("- Review the source excerpt.")
            markdown_parts.append("\n".join(block_lines).strip())
            continue

        content = str(section.get("content") or "").strip()
        if content:
            block_lines.append(content)
        else:
            block_lines.append("Content pending.")
        markdown_parts.append("\n".join(block_lines).strip())

    return "\n\n".join(markdown_parts).strip()


def canonical_studio_sections(sections: Any) -> list[dict[str, Any]]:
    """Validate and dump exact Task 1 Studio sections without semantic repair."""
    parsed = StudioSectionsV1.model_validate({"sections": sections})
    return parsed.model_dump(mode="json")["sections"]


def build_derived_studio_payload(
    payload: Mapping[str, Any],
    *,
    template_type: str,
    handwriting_mode: str,
    render_version: int = NOTE_STUDIO_RENDER_VERSION,
    fallback_title: str = "Untitled Study Notes",
    source_note_id: str,
) -> dict[str, Any]:
    """Build a derive payload around exact contract-validated section state."""
    incoming_meta = payload.get("meta")
    raw_title = incoming_meta.get("title") if isinstance(incoming_meta, Mapping) else None
    title = raw_title.strip() if isinstance(raw_title, str) else ""
    if not title:
        title = str(fallback_title).strip() or "Untitled Study Notes"
    return {
        "meta": {
            "title": title,
            "source_note_id": source_note_id,
        },
        "layout": {
            "template_type": str(template_type).strip() or "lined",
            "handwriting_mode": str(handwriting_mode).strip() or "accented",
            "render_version": int(render_version),
        },
        "sections": canonical_studio_sections(payload.get("sections")),
    }


def normalize_studio_payload(
    payload: Mapping[str, Any] | None,
    *,
    template_type: str,
    handwriting_mode: str,
    render_version: int = NOTE_STUDIO_RENDER_VERSION,
    fallback_title: str = "Untitled Study Notes",
    source_note_id: str | None = None,
    existing_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a generated payload to the Stage 2 canonical Notes Studio shape."""
    incoming_payload = payload if isinstance(payload, Mapping) else {}
    existing = existing_payload if isinstance(existing_payload, Mapping) else {}
    incoming_meta = incoming_payload.get("meta") if isinstance(incoming_payload.get("meta"), Mapping) else {}
    existing_meta = existing.get("meta") if isinstance(existing.get("meta"), Mapping) else {}

    normalized_title = (
        str(incoming_meta.get("title") or existing_meta.get("title") or fallback_title).strip() or fallback_title
    )
    normalized_source_note_id = source_note_id
    if normalized_source_note_id is None:
        normalized_source_note_id = str(
            incoming_meta.get("source_note_id") or existing_meta.get("source_note_id") or ""
        ).strip() or None

    meta: dict[str, Any] = {}
    if isinstance(existing_meta, Mapping):
        meta.update(dict(existing_meta))
    if isinstance(incoming_meta, Mapping):
        meta.update(dict(incoming_meta))
    meta["title"] = normalized_title
    meta["source_note_id"] = normalized_source_note_id

    normalized_sections = _try_canonical_studio_sections(
        incoming_payload.get("sections")
    )
    if normalized_sections is None:
        normalized_sections = _normalize_sections(
            incoming_payload.get("sections"),
            existing_sections=existing.get("sections"),
        )
        if not normalized_sections:
            normalized_sections = _try_canonical_studio_sections(
                existing.get("sections")
            )
            if normalized_sections is None:
                normalized_sections = _normalize_sections(
                    existing.get("sections"), existing_sections=None
                )

    return {
        "meta": meta,
        "layout": {
            "template_type": str(template_type).strip() or "lined",
            "handwriting_mode": str(handwriting_mode).strip() or "accented",
            "render_version": int(render_version),
        },
        "sections": normalized_sections,
    }


def studio_payload_from_markdown(
    markdown: str,
    *,
    template_type: str,
    handwriting_mode: str,
    render_version: int = NOTE_STUDIO_RENDER_VERSION,
    fallback_title: str = "Untitled Study Notes",
    source_note_id: str | None = None,
    existing_payload: Mapping[str, Any] | None = None,
    preserve_existing_sections_when_empty: bool = True,
) -> dict[str, Any]:
    """Rebuild a canonical Studio payload from structured Studio Markdown."""
    title, section_blocks = _parse_studio_markdown(markdown)
    normalized_existing_payload = existing_payload
    if not preserve_existing_sections_when_empty and not section_blocks:
        normalized_existing_payload = None
    payload = {
        "meta": {"title": title or fallback_title},
        "sections": _build_sections_from_blocks(
            section_blocks,
            existing_sections=(
                (normalized_existing_payload or {}).get("sections")
                if isinstance(normalized_existing_payload, Mapping)
                else None
            ),
        ),
    }
    return normalize_studio_payload(
        payload,
        template_type=template_type,
        handwriting_mode=handwriting_mode,
        render_version=render_version,
        fallback_title=title or fallback_title,
        source_note_id=source_note_id,
        existing_payload=normalized_existing_payload,
    )


def _iter_sections(sections: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(sections, list):
        return []
    normalized_sections: list[dict[str, Any]] = []
    for section in sections:
        if isinstance(section, Mapping):
            normalized_sections.append(dict(section))
    return normalized_sections


def _try_canonical_studio_sections(
    sections: Any,
) -> list[dict[str, Any]] | None:
    """Return canonical Studio sections, or ``None`` for invalid input."""
    try:
        return canonical_studio_sections(sections)
    except (TypeError, ValueError):
        return None


def _normalize_sections(sections: Any, *, existing_sections: Any = None) -> list[dict[str, Any]]:
    section_blocks: list[tuple[str, list[str]]] = []
    for section in _iter_sections(sections):
        kind = _normalize_section_kind(
            section.get("kind"),
            str(section.get("title") or ""),
            len(section_blocks),
        )
        title = str(section.get("title") or "").strip() or _DEFAULT_SECTION_TITLES.get(kind, "Section")
        if kind == "cue":
            items = [str(item).strip() for item in section.get("items") or [] if str(item).strip()]
            section_blocks.append((title, [f"- {item}" for item in items]))
            continue
        content = str(section.get("content") or "").strip()
        section_blocks.append((title, content.split("\n") if content else []))

    return _build_sections_from_blocks(section_blocks, existing_sections=existing_sections)


def _parse_studio_markdown(markdown: str) -> tuple[str | None, list[tuple[str, list[str]]]]:
    normalized = str(markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    title: str | None = None
    preamble_lines: list[str] = []
    section_blocks: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for raw_line in normalized.split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()
        if current_title is None and title is None and stripped.startswith("# "):
            title = stripped[2:].strip() or None
            continue
        if stripped.startswith("## "):
            if current_title is not None:
                section_blocks.append((current_title, _trim_blank_lines(current_lines)))
            current_title = stripped[3:].strip() or "Section"
            current_lines = []
            continue
        if current_title is None:
            preamble_lines.append(line)
        else:
            current_lines.append(line)

    if current_title is not None:
        section_blocks.append((current_title, _trim_blank_lines(current_lines)))

    if not section_blocks:
        trimmed_preamble = _trim_blank_lines(preamble_lines)
        if trimmed_preamble:
            section_blocks.append((_DEFAULT_SECTION_TITLES["notes"], trimmed_preamble))

    return title, section_blocks


def _trim_blank_lines(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _build_sections_from_blocks(
    section_blocks: list[tuple[str, list[str]]],
    *,
    existing_sections: Any,
) -> list[dict[str, Any]]:
    existing_ids = _existing_ids_by_kind(existing_sections)
    kind_counts: dict[str, int] = {}
    normalized_sections: list[dict[str, Any]] = []

    for position, (title, lines) in enumerate(section_blocks):
        kind = _infer_section_kind(title, position)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        index = kind_counts[kind] - 1
        title_text = str(title or "").strip() or _DEFAULT_SECTION_TITLES.get(kind, "Section")
        section_id = _resolve_section_id(kind=kind, index=index, existing_ids=existing_ids)
        if kind == "cue":
            normalized_sections.append(
                {
                    "id": section_id,
                    "kind": kind,
                    "title": title_text,
                    "items": _parse_bullet_items(lines),
                }
            )
            continue
        normalized_sections.append(
            {
                "id": section_id,
                "kind": kind,
                "title": title_text,
                "content": "\n".join(_trim_blank_lines(lines)).strip(),
            }
        )

    return normalized_sections


def _existing_ids_by_kind(existing_sections: Any) -> dict[str, list[str]]:
    ids_by_kind: dict[str, list[str]] = {}
    for index, section in enumerate(_iter_sections(existing_sections)):
        kind = _normalize_section_kind(section.get("kind"), str(section.get("title") or ""), index)
        section_id = str(section.get("id") or "").strip()
        if not section_id:
            continue
        ids_by_kind.setdefault(kind, []).append(section_id)
    return ids_by_kind


def _resolve_section_id(*, kind: str, index: int, existing_ids: dict[str, list[str]]) -> str:
    known_ids = existing_ids.get(kind) or []
    if index < len(known_ids):
        return known_ids[index]
    return f"{kind}-{index + 1}"


def _infer_section_kind(title: str, position: int) -> str:
    normalized_title = str(title or "").strip().lower()
    if normalized_title in _CUE_SECTION_TITLES:
        return "cue"
    if normalized_title in _SUMMARY_SECTION_TITLES:
        return "summary"
    if position == 0 and "question" in normalized_title:
        return "cue"
    if "summary" in normalized_title:
        return "summary"
    return "notes"


def _normalize_section_kind(kind: Any, title: str, position: int) -> str:
    normalized_kind = str(kind or "").strip().lower()
    if normalized_kind in {"cue", "notes", "summary"}:
        return normalized_kind
    return _infer_section_kind(title, position)


def _parse_bullet_items(lines: list[str]) -> list[str]:
    items: list[str] = []
    current_item: str | None = None
    for line in _trim_blank_lines(lines):
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            if current_item is not None:
                items.append(current_item.strip())
            current_item = stripped[2:].strip()
            continue
        if current_item is not None and stripped:
            current_item = f"{current_item} {stripped}".strip()
            continue
        if stripped:
            items.append(stripped)
    if current_item is not None:
        items.append(current_item.strip())
    return items
