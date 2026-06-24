"""Parse subtitle files into plain text with cue boundaries preserved."""

from __future__ import annotations

import re

_ASS_DIALOGUE_RE = re.compile(r"^(dialogue|comment):", re.IGNORECASE)
_ASS_TAG_RE = re.compile(r"\{[^}]*\}")
_SUBTITLE_TIMESTAMP_RE = r"(?:\d{1,2}:)?\d{2}:\d{2}[,.]\d{1,3}"
_SRT_VTT_TIMING_RE = re.compile(
    rf"^{_SUBTITLE_TIMESTAMP_RE}\s+-->\s+{_SUBTITLE_TIMESTAMP_RE}(?:\s+.*)?$"
)


def normalize_subtitle_source(text: str, input_type: str) -> str:
    """Normalize subtitle source text while preserving cue boundaries."""
    kind = (input_type or "").lower()
    if kind in {"srt", "vtt"}:
        return _parse_srt_vtt(text, kind=kind)
    if kind == "ass":
        return _parse_ass(text)
    return text


def _parse_srt_vtt(text: str, *, kind: str) -> str:
    cues: list[str] = []
    current: list[str] = []
    lines = text.splitlines()
    skip_block = False

    def _peek_next_non_empty_line(start: int) -> str | None:
        if start >= len(lines):
            return None
        candidate = lines[start].strip().lstrip("\ufeff")
        if not candidate:
            return None
        return candidate

    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip().lstrip("\ufeff")
        if skip_block:
            if not stripped:
                skip_block = False
            continue
        if not stripped:
            if current:
                cue_text = "\n".join(current).strip()
                if cue_text:
                    cues.append(cue_text)
                current = []
            continue
        lowered = stripped.lower()
        if kind == "vtt" and lowered.startswith("webvtt"):
            continue
        if kind == "vtt" and lowered.startswith(("note", "style", "region")) and not current:
            # Skip VTT blocks entirely when not in a cue block.
            skip_block = True
            continue
        if _SRT_VTT_TIMING_RE.match(stripped):
            # Timing line, ignore.
            continue
        if not current:
            next_line = _peek_next_non_empty_line(idx + 1)
            if next_line and _SRT_VTT_TIMING_RE.match(next_line.strip()):
                if kind == "vtt":
                    # Cue identifier line for VTT.
                    continue
                if stripped.isdigit():
                    # Cue index line for SRT.
                    continue
        current.append(stripped)
    if current:
        cue_text = "\n".join(current).strip()
        if cue_text:
            cues.append(cue_text)
    return "\n".join(cues).strip()


def _parse_ass(text: str) -> str:
    cues: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip().lstrip("\ufeff")
        if not stripped:
            continue
        if not _ASS_DIALOGUE_RE.match(stripped):
            continue
        parts = stripped.split(",", 9)
        if len(parts) < 10:
            continue
        payload = parts[9]
        payload = payload.replace("\\N", "\n").replace("\\n", "\n")
        payload = _ASS_TAG_RE.sub("", payload).strip()
        if payload:
            cues.append(payload)
    return "\n".join(cues).strip()
