"""Generate subtitle files from alignment payloads."""

from __future__ import annotations

import html
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.audiobook_schemas import (
    AlignmentPayload,
    AlignmentWord,
    SubtitleFormat,
    SubtitleMode,
    SubtitleVariant,
)
from tldw_Server_API.app.core.config import get_config_value
from tldw_Server_API.app.core.Utils.common import parse_boolean

_ASS_OVERRIDE_RE = re.compile(r"\{[^}]*\}")
_ASS_INLINE_CONTROL_RE = re.compile(r"\\[Nnh]")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class SubtitleCue:
    index: int
    start_ms: int
    end_ms: int
    text: str


def generate_subtitles(
    alignment: AlignmentPayload,
    *,
    format: SubtitleFormat,
    mode: SubtitleMode,
    variant: SubtitleVariant,
    source_text: str | None = None,
    enable_spacy: bool | None = None,
    words_per_cue: int | None = None,
    max_chars: int | None = None,
    max_lines: int | None = None,
    max_duration_ms: int | None = None,
) -> str:
    """Generate subtitle content for the requested format."""
    words = alignment.words
    if not words:
        raise ValueError("alignment words must not be empty")

    effective_words_per_cue = words_per_cue or 1
    if effective_words_per_cue <= 0:
        raise ValueError("words_per_cue must be positive")

    effective_max_chars = _resolve_max_chars(max_chars, variant)
    if mode == "sentence":
        cues = _chunk_sentences(words, source_text=source_text, enable_spacy=enable_spacy)
        duration_limit = max_duration_ms if max_duration_ms is not None else 6000
        if _should_fallback_sentence(cues, effective_max_chars, duration_limit):
            cues = _chunk_words(words, effective_words_per_cue)
    else:
        cues = _build_cues(words, mode, effective_words_per_cue, source_text=source_text)
    line_sep = "\\N" if format == "ass" else "\n"

    rendered: list[SubtitleCue] = []
    for idx, cue_words in enumerate(cues, start=1):
        start_ms = cue_words[0].start_ms
        end_ms = cue_words[-1].end_ms
        if end_ms < start_ms:
            raise ValueError("alignment end_ms must be >= start_ms")
        if mode == "highlight":
            word = cue_words[0]
            safe_word = _sanitize_word_for_format(word.word, format)
            if format == "vtt":
                text = f"<c.hl>{safe_word}</c>"
            elif format == "ass":
                duration_cs = max(1, (end_ms - start_ms) // 10)
                text = f"{{\\k{duration_cs}}}{safe_word}"
            else:
                text = safe_word
        else:
            text = _render_text(cue_words, effective_max_chars, max_lines, line_sep, format)
        if mode == "word_count":
            start_ms, end_ms = _clamp_cue_duration(start_ms, end_ms)
        rendered.append(SubtitleCue(index=idx, start_ms=start_ms, end_ms=end_ms, text=text))

    if format == "srt":
        return _format_srt(rendered)
    if format == "vtt":
        return _format_vtt(rendered, variant)
    if format == "ass":
        return _format_ass(rendered, variant)
    raise ValueError(f"unsupported subtitle format: {format}")


def _resolve_max_chars(max_chars: int | None, variant: SubtitleVariant) -> int | None:
    if max_chars is not None:
        return max_chars
    if variant == "narrow":
        return 28
    if variant in {"wide", "centered"}:
        return 42
    return None


def _build_cues(
    words: Sequence[AlignmentWord],
    mode: SubtitleMode,
    words_per_cue: int,
    *,
    source_text: str | None,
) -> list[list[AlignmentWord]]:
    if mode == "highlight":
        return [[word] for word in words]
    if mode == "word_count":
        return _chunk_words(words, words_per_cue)
    if mode == "sentence":
        return _chunk_sentences(words, source_text=None, enable_spacy=None)
    return _chunk_lines(words, source_text=source_text)


def _chunk_words(words: Sequence[AlignmentWord], words_per_cue: int) -> list[list[AlignmentWord]]:
    cues: list[list[AlignmentWord]] = []
    for i in range(0, len(words), words_per_cue):
        cues.append(list(words[i : i + words_per_cue]))
    return cues


def _chunk_sentences(
    words: Sequence[AlignmentWord],
    *,
    source_text: str | None,
    enable_spacy: bool | None,
) -> list[list[AlignmentWord]]:
    if source_text and _should_use_spacy(enable_spacy):
        spacy_cues = _chunk_sentences_spacy(words, source_text)
        if spacy_cues:
            return spacy_cues
    cues: list[list[AlignmentWord]] = []
    current: list[AlignmentWord] = []
    for word in words:
        current.append(word)
        if word.word.strip().endswith((".", "!", "?")):
            cues.append(current)
            current = []
    if current:
        cues.append(current)
    return cues


def _chunk_sentences_spacy(
    words: Sequence[AlignmentWord],
    source_text: str,
) -> list[list[AlignmentWord]] | None:
    nlp = _load_spacy_model()
    if nlp is None:
        return None
    try:
        doc = nlp(source_text)
    except Exception as exc:
        logger.warning("spaCy sentence parsing failed: {}", exc)
        return None
    sentences = list(getattr(doc, "sents", []) or [])
    if not sentences:
        return None

    cues: list[list[AlignmentWord]] = []
    current: list[AlignmentWord] = []
    sent_idx = 0
    current_end = sentences[0].end_char
    for word in words:
        if word.char_start is not None:
            while sent_idx < len(sentences) and word.char_start >= current_end:
                if current:
                    cues.append(current)
                    current = []
                sent_idx += 1
                if sent_idx < len(sentences):
                    current_end = sentences[sent_idx].end_char
        current.append(word)
    if current:
        cues.append(current)
    return cues if cues else None


def _should_use_spacy(enable_spacy: bool | None) -> bool:
    if enable_spacy is not None:
        return enable_spacy
    env_value = os.getenv("AUDIOBOOK_ENABLE_SPACY")
    if env_value is not None:
        return parse_boolean(env_value, default=False)
    cfg_value = get_config_value("Audiobooks", "enable_spacy_sentence_splitting")
    return parse_boolean(cfg_value, default=False)


@lru_cache(maxsize=1)
def _load_spacy_model():
    try:
        import spacy  # type: ignore
    except Exception as exc:
        logger.warning("spaCy unavailable: {}", exc)
        return None

    model_name = os.getenv("AUDIOBOOK_SPACY_MODEL") or get_config_value(
        "Audiobooks", "spacy_model", "en_core_web_sm"
    )
    try:
        return spacy.load(model_name)
    except Exception as exc:
        logger.warning("Failed to load spaCy model '{}': {}", model_name, exc)
        return None


def _chunk_lines(
    words: Sequence[AlignmentWord],
    *,
    source_text: str | None,
) -> list[list[AlignmentWord]]:
    if source_text and all(word.char_start is not None for word in words):
        return _chunk_lines_from_source(words, source_text)
    cues: list[list[AlignmentWord]] = []
    current: list[AlignmentWord] = []
    for word in words:
        current.append(word)
        if "\n" in word.word:
            cues.append(current)
            current = []
    if current:
        cues.append(current)
    return cues


def _chunk_lines_from_source(
    words: Sequence[AlignmentWord],
    source_text: str,
) -> list[list[AlignmentWord]]:
    cues: list[list[AlignmentWord]] = []
    current: list[AlignmentWord] = []
    text_len = len(source_text)
    for idx, word in enumerate(words):
        current.append(word)
        if idx + 1 >= len(words):
            break
        next_word = words[idx + 1]
        if word.char_start is None or next_word.char_start is None:
            continue
        start = word.char_end if word.char_end is not None else word.char_start
        end = next_word.char_start
        if start < 0 or end < 0:
            continue
        if start > text_len:
            continue
        if end > text_len:
            end = text_len
        if end < start:
            continue
        if "\n" in source_text[start:end]:
            cues.append(current)
            current = []
    if current:
        cues.append(current)
    return cues


def _should_fallback_sentence(
    cues: Sequence[Sequence[AlignmentWord]],
    max_chars: int | None,
    max_duration_ms: int,
) -> bool:
    for cue in cues:
        if not cue:
            continue
        duration = cue[-1].end_ms - cue[0].start_ms
        if duration > max_duration_ms:
            return True
        if max_chars is not None:
            text = " ".join(word.word for word in cue).replace("\n", " ").strip()
            if len(text) > max_chars:
                return True
    return False


def _clamp_cue_duration(start_ms: int, end_ms: int) -> tuple[int, int]:
    min_duration = 800
    max_duration = 6000
    duration = end_ms - start_ms
    if duration < min_duration:
        end_ms = start_ms + min_duration
    elif duration > max_duration:
        end_ms = start_ms + max_duration
    return start_ms, end_ms


def _render_text(
    words: Sequence[AlignmentWord],
    max_chars: int | None,
    max_lines: int | None,
    line_sep: str,
    format: SubtitleFormat,
) -> str:
    """Render sanitized alignment words into wrapped subtitle cue text."""
    tokens = [_sanitize_word_for_format(word.word, format) for word in words]
    if max_lines is not None and max_lines <= 0:
        raise ValueError("max_lines must be positive")
    if max_chars is not None and max_chars <= 0:
        raise ValueError("max_chars must be positive")

    raw_text = " ".join(tokens).replace("\n", " ").strip() if line_sep == "\n" else " ".join(tokens).strip()
    if max_chars is None and max_lines is None:
        return raw_text

    if max_chars is None:
        if max_lines is None:
            return raw_text
        # split lines only on explicit newlines for line mode cues
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()] if line_sep == "\n" else [raw_text]
        if max_lines is not None and len(lines) > max_lines:
            merged = " ".join(lines[max_lines - 1 :])
            lines = lines[: max_lines - 1] + [merged]
        return line_sep.join(lines).strip()
    lines: list[str] = []
    current: list[str] = []
    for token in tokens:
        candidate = " ".join(current + [token]) if current else token
        if len(candidate) <= max_chars or not current:
            current.append(token)
            continue
        lines.append(" ".join(current))
        current = [token]
    if current:
        lines.append(" ".join(current))
    if max_lines is not None and len(lines) > max_lines:
        merged = " ".join(lines[max_lines - 1 :])
        lines = lines[: max_lines - 1] + [merged]
    rendered = line_sep.join(lines).replace("\n\n", "\n").strip() if line_sep == "\n" else line_sep.join(lines).strip()
    return rendered


def _sanitize_word_for_format(text: str, format: SubtitleFormat) -> str:
    """Sanitize a subtitle token for the target subtitle container format."""
    clean = _normalize_word_text(text)
    if format == "vtt":
        return html.escape(clean, quote=False)
    if format == "ass":
        clean = _ASS_OVERRIDE_RE.sub("", clean)
        clean = _ASS_INLINE_CONTROL_RE.sub(" ", clean)
        clean = clean.replace("\\", "")
        clean = clean.replace("{", "(").replace("}", ")")
        return " ".join(clean.split())
    return clean.replace("-->", "-- >")


def _normalize_word_text(text: str) -> str:
    """Collapse control characters and line separators inside subtitle word text."""
    clean = str(text)
    clean = clean.replace("\r\n", "\n").replace("\r", "\n")
    clean = clean.replace("\u2028", "\n").replace("\u2029", "\n")
    clean = _CONTROL_CHARS_RE.sub(" ", clean)
    clean = clean.replace("\n", " ")
    return " ".join(clean.split())


def _format_srt(cues: Iterable[SubtitleCue]) -> str:
    blocks: list[str] = []
    for cue in cues:
        start = _format_srt_time(cue.start_ms)
        end = _format_srt_time(cue.end_ms)
        blocks.append(f"{cue.index}\n{start} --> {end}\n{cue.text}")
    return "\n\n".join(blocks).strip()


def _format_vtt(cues: Iterable[SubtitleCue], variant: SubtitleVariant) -> str:
    blocks: list[str] = ["WEBVTT"]
    settings = " align:center" if variant == "centered" else ""
    for cue in cues:
        start = _format_vtt_time(cue.start_ms)
        end = _format_vtt_time(cue.end_ms)
        blocks.append(f"{start} --> {end}{settings}\n{cue.text}")
    return "\n\n".join(blocks).strip()


def _format_ass(cues: Iterable[SubtitleCue], variant: SubtitleVariant) -> str:
    alignment = "2" if variant == "centered" else "1"
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,"
        "Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Default,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        f"0,0,0,0,100,100,0,0,1,2,0,{alignment},10,10,10,1\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"
    )
    lines: list[str] = [header]
    for cue in cues:
        start = _format_ass_time(cue.start_ms)
        end = _format_ass_time(cue.end_ms)
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{cue.text}")
    return "\n".join(lines).strip()


def _format_srt_time(ms: int) -> str:
    hours, minutes, seconds, millis = _split_time(ms)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _format_vtt_time(ms: int) -> str:
    hours, minutes, seconds, millis = _split_time(ms)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _format_ass_time(ms: int) -> str:
    hours, minutes, seconds, millis = _split_time(ms)
    centis = millis // 10
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centis:02d}"


def _split_time(ms: int) -> tuple[int, int, int, int]:
    total_seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds, millis
