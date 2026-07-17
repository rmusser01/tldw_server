"""LLM-backed slide generation helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.Chat.chat_helpers import extract_response_content
from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call
from tldw_Server_API.app.core.Slides.visual_style_generation import (
    apply_visual_block_fallback,
    build_visual_style_generation_prompt,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Utils.tokenizer import count_tokens


class SlidesGenerationError(Exception):
    """Base exception for slide generation failures."""


class SlidesSourceTooLargeError(SlidesGenerationError):
    """Raised when input exceeds size limits and chunking is disabled."""

    def __init__(
        self,
        message: str,
        *,
        max_source_tokens: int | None = None,
        max_source_chars: int | None = None,
        actual_tokens: int | None = None,
        actual_chars: int | None = None,
    ) -> None:
        super().__init__(message)
        self.max_source_tokens = max_source_tokens
        self.max_source_chars = max_source_chars
        self.actual_tokens = actual_tokens
        self.actual_chars = actual_chars


class SlidesGenerationInputError(SlidesGenerationError):
    """Raised for invalid inputs."""


class SlidesGenerationOutputError(SlidesGenerationError):
    """Raised when LLM output cannot be parsed."""


_SYSTEM_PROMPT = (
    "You are creating presentation slides. Output valid JSON:\n"
    "{\n"
    '  "title": "Presentation Title",\n'
    '  "slides": [\n'
    '    {"order": 0, "layout": "title", "title": "...", "content": "..."},\n'
    '    {"order": 1, "layout": "content", "title": "...", "content": "- Bullet 1\\n- Bullet 2"},\n'
    "    ...\n"
    "  ]\n"
    "}\n\n"
    "Guidelines:\n"
    "- Use only facts explicitly supported by the source material.\n"
    "- Do not invent recommendations, next steps, objectives, methods, trial status, comparisons, "
    "or conclusions unless the source explicitly states them.\n"
    "- Speaker notes must stay source-grounded and must not add presenter chatter or unsupported claims.\n"
    "- Short sources may produce 2-5 slides; do not stretch the deck by adding unsourced content.\n"
    "- Omit conclusion or summary slides for short sources unless the source states conclusions.\n"
    "- 5-12 slides typical length for longer source material\n"
    "- Title slide first; use a final source-facts slide only when enough supported facts remain\n"
    "- Use markdown formatting (bullets, bold, code)\n"
    "- 3-6 bullet points per content slide\n"
    "- Add speaker_notes for details that do not fit slides\n"
)

_SUMMARY_SYSTEM_PROMPT = "Summarize the following content for slide generation."
DEFAULT_MAX_SOURCE_TOKENS = 50_000
DEFAULT_MAX_SOURCE_CHARS = 200_000
MAX_SOURCE_CHUNKS = 20
MAX_CHUNK_SIZE_TOKENS = 4_000
_SLIDE_PLACEHOLDER_TEXTS = {
    "invalid",
    "slide goes here",
    "slides go here",
    "content goes here",
    "todo",
    "tbd",
}


def _is_placeholder_slide_text(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    return not normalized or normalized in _SLIDE_PLACEHOLDER_TEXTS


def _has_usable_slide_content(slide: dict[str, Any]) -> bool:
    layout = str(slide.get("layout") or "").strip().lower()
    title_ok = not _is_placeholder_slide_text(slide.get("title"))
    content_ok = not _is_placeholder_slide_text(slide.get("content"))
    notes_ok = not _is_placeholder_slide_text(slide.get("speaker_notes"))
    if layout == "title":
        return title_ok or content_ok or notes_ok
    return content_ok or notes_ok


def _positive_int_setting(name: str, default: int) -> int:
    raw_value = settings.get(name, default)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning("slides generation: invalid integer setting {}; using default {}", name, default)
        return default
    if value <= 0:
        logger.warning("slides generation: non-positive integer setting {}; using default {}", name, default)
        return default
    return value


def _normalize_provider(provider: str | None) -> str:
    value = (provider or "").strip()
    return value.lower() if value else ""


def _extract_json_block(text: str) -> str:
    if not text:
        return text
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _chunk_by_words(text: str, chunk_size: int) -> list[str]:
    if chunk_size <= 0:
        return [text]
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def _chunk_by_chars(text: str, chunk_size: int) -> list[str]:
    if chunk_size <= 0:
        return [text]
    chunks: list[str] = []
    for idx in range(0, len(text), chunk_size):
        chunks.append(text[idx : idx + chunk_size])
    return chunks


def _extract_test_mode_points(source_text: str, limit: int = 6) -> list[str]:
    """Return deterministic bullet candidates from the source text for test-mode slides."""
    points: list[str] = []
    for raw_line in source_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        candidate = line.lstrip("-*• ").strip()
        if candidate:
            points.append(candidate)
        if len(points) >= limit:
            break
    if points:
        return points

    compact = re.sub(r"\s+", " ", source_text).strip()
    if not compact:
        return ["No source details were provided."]

    sentences = [
        chunk.strip()
        for chunk in re.split(r"(?<=[.!?])\s+", compact)
        if chunk.strip()
    ]
    return sentences[:limit] or [compact[:160]]


class SlidesGenerator:
    def __init__(
        self,
        *,
        llm_call: Callable[..., Any] | None = None,
    ) -> None:
        self._llm_call = llm_call or perform_chat_api_call

    def generate_from_text(
        self,
        *,
        source_text: str,
        title_hint: str | None,
        provider: str,
        model: str | None,
        api_key: str | None,
        temperature: float | None,
        max_tokens: int | None,
        max_source_tokens: int | None,
        max_source_chars: int | None,
        enable_chunking: bool,
        chunk_size_tokens: int | None,
        summary_tokens: int | None,
        visual_style_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not source_text or not source_text.strip():
            raise SlidesGenerationInputError("source_text_required")

        normalized_provider = _normalize_provider(provider)
        if not normalized_provider:
            raise SlidesGenerationInputError("provider_required")

        source_text = source_text.strip()
        actual_tokens = count_tokens(source_text)
        actual_chars = len(source_text)
        effective_max_source_tokens = (
            max_source_tokens
            if max_source_tokens is not None
            else _positive_int_setting("SLIDES_MAX_SOURCE_TOKENS", DEFAULT_MAX_SOURCE_TOKENS)
        )
        effective_max_source_chars = (
            max_source_chars
            if max_source_chars is not None
            else _positive_int_setting("SLIDES_MAX_SOURCE_CHARS", DEFAULT_MAX_SOURCE_CHARS)
        )

        limit_exceeded = False
        if actual_tokens > effective_max_source_tokens:
            limit_exceeded = True
        if actual_chars > effective_max_source_chars:
            limit_exceeded = True

        if limit_exceeded and not enable_chunking:
            raise SlidesSourceTooLargeError(
                "input_too_large",
                max_source_tokens=effective_max_source_tokens,
                max_source_chars=effective_max_source_chars,
                actual_tokens=actual_tokens,
                actual_chars=actual_chars,
            )

        if is_test_mode():
            return self._build_test_mode_payload(
                source_text=source_text,
                title_hint=title_hint,
            )

        prepared_text = source_text
        if enable_chunking:
            prepared_text = self._chunk_and_summarize(
                source_text=source_text,
                provider=normalized_provider,
                model=model,
                api_key=api_key,
                temperature=temperature,
                chunk_size_tokens=chunk_size_tokens,
                summary_tokens=summary_tokens,
                actual_tokens=actual_tokens,
                actual_chars=actual_chars,
            )

        user_prompt = "Source material:\n" + prepared_text
        if title_hint:
            user_prompt = f"Title hint: {title_hint}\n\n" + user_prompt

        system_prompt = _SYSTEM_PROMPT
        style_prompt = build_visual_style_generation_prompt(visual_style_snapshot)
        if style_prompt:
            system_prompt = _SYSTEM_PROMPT + "\n\nStyle guidance:\n" + style_prompt

        llm_response = self._call_llm(
            provider=normalized_provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        payload = self._parse_json(llm_response)
        normalized = self._normalize_payload(payload, title_hint)
        return normalized

    def _build_test_mode_payload(
        self,
        *,
        source_text: str,
        title_hint: str | None,
    ) -> dict[str, Any]:
        """Build a deterministic slide deck payload without calling an external model."""
        title = (title_hint or "Test Mode Presentation").strip() or "Test Mode Presentation"
        points = _extract_test_mode_points(source_text)
        overview_points = points[:3] or ["Source content captured for slides generation."]
        takeaway_points = points[3:6] or points[:2] or [
            "Review the source content for more detail."
        ]

        return {
            "title": title,
            "slides": [
                {
                    "order": 0,
                    "layout": "title",
                    "title": title,
                    "content": "",
                    "speaker_notes": None,
                    "metadata": {},
                },
                {
                    "order": 1,
                    "layout": "content",
                    "title": "Overview",
                    "content": "\n".join(f"- {point}" for point in overview_points),
                    "speaker_notes": "Deterministic slide generated in test mode.",
                    "metadata": {},
                },
                {
                    "order": 2,
                    "layout": "content",
                    "title": "Key Takeaways",
                    "content": "\n".join(f"- {point}" for point in takeaway_points),
                    "speaker_notes": "Use the live provider flow outside test mode for final content.",
                    "metadata": {},
                },
            ],
        }

    def _chunk_and_summarize(
        self,
        *,
        source_text: str,
        provider: str,
        model: str | None,
        api_key: str | None,
        temperature: float | None,
        chunk_size_tokens: int | None,
        summary_tokens: int | None,
        actual_tokens: int | None = None,
        actual_chars: int | None = None,
    ) -> str:
        """Reduce oversized source text into a prompt-sized summary before slide generation."""
        if chunk_size_tokens is None:
            chunk_size = 1000
        else:
            try:
                chunk_size = int(chunk_size_tokens)
            except (TypeError, ValueError) as exc:
                raise SlidesGenerationInputError("chunk_size_invalid") from exc
        if chunk_size < 1:
            raise SlidesGenerationInputError("chunk_size_invalid")
        if chunk_size > MAX_CHUNK_SIZE_TOKENS:
            raise SlidesGenerationInputError("chunk_size_too_large")
        mode = str(settings.get("TOKEN_ESTIMATOR_MODE") or "whitespace").lower()
        if mode == "char_approx":
            try:
                chars_per_token = max(1, int(settings.get("TOKEN_CHAR_APPROX_DIVISOR", 4)))
            except (TypeError, ValueError):
                logger.warning(
                    "slides generation: invalid integer setting {}; using default {}",
                    "TOKEN_CHAR_APPROX_DIVISOR",
                    4,
                )
                chars_per_token = 4
            size_chars = chunk_size * chars_per_token
            chunks = _chunk_by_chars(source_text, size_chars)
        else:
            chunks = _chunk_by_words(source_text, chunk_size)

        if not chunks:
            return ""
        max_chunks = _positive_int_setting("SLIDES_MAX_SOURCE_CHUNKS", MAX_SOURCE_CHUNKS)
        if len(chunks) > max_chunks:
            raise SlidesSourceTooLargeError(
                "input_too_large",
                actual_tokens=actual_tokens,
                actual_chars=actual_chars,
            )

        summary_target = summary_tokens or 200
        summaries: list[str] = []
        for chunk in chunks:
            prompt = "Summarize this content for slide generation:\n\n" + chunk
            summary = self._call_llm(
                provider=provider,
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=summary_target,
                system_prompt=_SUMMARY_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
            summaries.append(summary.strip())
        return "\n\n".join(summaries).strip()

    def _call_llm(
        self,
        *,
        provider: str,
        model: str | None,
        api_key: str | None,
        temperature: float | None,
        max_tokens: int | None,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self._llm_call(
                api_provider=provider,
                model=model,
                api_key=api_key,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.error("slides generation LLM call failed: {}", exc)
            raise SlidesGenerationError("llm_call_failed") from exc

        content = extract_response_content(response)
        if not content:
            raise SlidesGenerationOutputError("empty_llm_response")
        return content

    def _parse_json(self, raw_text: str) -> dict[str, Any]:
        candidate = _extract_json_block(raw_text)
        try:
            parsed = json.loads(candidate)
        except Exception as exc:
            raise SlidesGenerationOutputError("invalid_json_output") from exc
        if not isinstance(parsed, dict):
            raise SlidesGenerationOutputError("json_output_not_object")
        return parsed

    def _normalize_payload(self, payload: dict[str, Any], title_hint: str | None) -> dict[str, Any]:
        slides = payload.get("slides")
        if not isinstance(slides, list) or not slides:
            raise SlidesGenerationOutputError("slides_missing")
        normalized_slides: list[dict[str, Any]] = []
        for idx, slide in enumerate(slides):
            if not isinstance(slide, dict):
                raise SlidesGenerationOutputError("slide_entry_invalid")
            slide.setdefault("order", idx)
            slide.setdefault("layout", "content")
            slide.setdefault("title", None)
            slide.setdefault("content", "")
            slide.setdefault("speaker_notes", None)
            slide.setdefault("metadata", {})
            slide = apply_visual_block_fallback(slide)
            if not _has_usable_slide_content(slide):
                raise SlidesGenerationOutputError("slide_content_missing")
            normalized_slides.append(slide)
        title = payload.get("title")
        if not isinstance(title, str) or not title.strip():
            title = title_hint or "Untitled Presentation"
        return {"title": title.strip(), "slides": normalized_slides}
