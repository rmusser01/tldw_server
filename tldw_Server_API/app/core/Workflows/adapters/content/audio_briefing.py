"""Audio briefing script composition adapter.

Composes a multi-voice audio narration script from article summaries via LLM,
with section markers and voice assignments for downstream multi-voice TTS.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string
from tldw_Server_API.app.core.Workflows.adapters._common import extract_openai_content
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.content._config import AudioBriefingComposeConfig

_BRIEFING_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    UnicodeError,
    ValueError,
)

_DEFAULT_VOICE_MAP: dict[str, str] = {
    "HOST": "af_bella",
    "REPORTER": "am_adam",
    "ANALYST": "bf_emma",
}

_VOICE_MARKER_RE = re.compile(r'^\[([A-Z_]+)\]:\s*', re.MULTILINE)
_REASONING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>[\s\S]*?</(?:think|thinking|reasoning)>\s*",
    flags=re.IGNORECASE,
)
_REASONING_TAG_RE = re.compile(r"</?(?:think|thinking|reasoning)>", flags=re.IGNORECASE)


def _normalize_output_language(value: Any) -> str:
    """Normalize output language hint to a compact non-empty token."""
    if value is None:
        return "en"
    lang = str(value).strip()
    if not lang:
        return "en"
    return lang


def _build_language_rule(output_language: str) -> str:
    normalized = output_language.lower().replace("_", "-").strip()
    if normalized in {"en", "en-us", "en-gb", "english"}:
        return "Reply in English only."
    return f"Reply only in {output_language}. Do not switch languages."


def _strip_reasoning_blocks(text: str) -> str:
    """Strip hidden reasoning tags/blocks that should never be spoken."""
    stripped = _REASONING_BLOCK_RE.sub("", text or "")
    stripped = _REASONING_TAG_RE.sub("", stripped)
    return stripped


def _normalize_item(item: Any) -> dict[str, str]:
    if isinstance(item, dict):
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or item.get("snippet") or "").strip()
        url = str(item.get("url") or item.get("source_url") or "").strip()
        return {"title": title, "summary": summary, "url": url}
    title = str(getattr(item, "title", "") or "").strip()
    summary = str(
        getattr(item, "summary", "") or getattr(item, "snippet", "") or ""
    ).strip()
    url = str(getattr(item, "url", "") or getattr(item, "source_url", "") or "").strip()
    return {"title": title, "summary": summary, "url": url}


def _build_persona_summary_system_prompt(output_language: str, persona_id: str | None) -> str:
    persona_hint = (persona_id or "").strip()
    persona_instruction = (
        f"Adopt this persona style while staying factual: {persona_hint}."
        if persona_hint
        else "Use a neutral professional briefing tone."
    )
    return (
        "You rewrite article summaries for short spoken audio news briefings.\n"
        f"{persona_instruction}\n"
        f"{_build_language_rule(output_language)}\n"
        "Rules:\n"
        "- Return only the rewritten summary text.\n"
        "- Maximum 2 to 3 short spoken sentences.\n"
        "- No markdown, no URLs, no bullet points, no labels, no emojis.\n"
        "- Preserve factual meaning and avoid adding claims."
    )


async def _persona_pre_summarize_items(
    items: list[dict[str, str]],
    *,
    output_language: str,
    provider: str | None,
    model: str | None,
    persona_id: str | None,
) -> list[dict[str, str]]:
    if not items:
        return []
    if not provider:
        return items

    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

    system_prompt = _build_persona_summary_system_prompt(output_language, persona_id)
    rewritten_items: list[dict[str, str]] = []

    for item in items:
        normalized = _normalize_item(item)
        title = normalized.get("title") or "Untitled"
        source_summary = normalized.get("summary") or title
        source_url = normalized.get("url") or ""
        if not source_summary:
            rewritten_items.append(normalized)
            continue

        user_prompt = (
            f"Title: {title}\n"
            f"Summary: {source_summary}\n"
            f"URL: {source_url}\n\n"
            "Rewrite this as a concise spoken summary."
        )
        try:
            response = await perform_chat_api_call_async(
                messages=[{"role": "user", "content": user_prompt}],
                api_provider=provider,
                model=model,
                system_message=system_prompt,
                max_tokens=240,
                temperature=0.35,
            )
            rewritten = _strip_reasoning_blocks(extract_openai_content(response) or "").strip()
            if rewritten:
                normalized["summary"] = rewritten
        except _BRIEFING_NONCRITICAL_EXCEPTIONS:
            logger.warning("Persona pre-summarization failed", exc_info=True)
        rewritten_items.append(normalized)

    return rewritten_items


def _build_system_prompt(target_words: int, multi_voice: bool, output_language: str) -> str:
    """Build the system prompt for LLM script composition."""
    voice_instructions = ""
    if multi_voice:
        voice_instructions = """
Use voice markers to indicate speaker changes:
- [HOST]: for transitions, greetings, and wrap-ups
- [REPORTER]: for article details and reporting
- [ANALYST]: for analysis and expert commentary (optional)

Every line of spoken text MUST start with a voice marker like [HOST]: or [REPORTER]:."""
    else:
        voice_instructions = "\nWrite as a single narrator. Do not use any voice markers or speaker labels."

    return f"""You are a professional audio news briefing scriptwriter. Write a spoken-word
news briefing script that sounds natural when read aloud by text-to-speech.

Target length: approximately {target_words} words.
{voice_instructions}

Rules:
- Write for the ear, not the eye. Use short, clear sentences.
- NO markdown formatting (no headers, bold, italic, links, code blocks).
- NO URLs in the script.
- {_build_language_rule(output_language)}
- Do NOT use emoji, decorative symbols, or ornamental punctuation.
- Do NOT include side notes, production notes, or counters like "(200 chars)".
- Do NOT include section labels, signatures, or prose headers.
- Expand abbreviations on first use (e.g., "AI, or Artificial Intelligence").
- Use [pause] between major topic transitions.
- Start with a greeting that includes the current date context.
- End with a brief wrap-up and sign-off.
- Use natural spoken transitions between stories (e.g., "Moving on to...", "In other news...").
- Avoid jargon unless you explain it immediately."""


def _parse_sections(script: str) -> list[dict[str, str]]:
    """Parse a multi-voice script into sections by voice marker."""
    sections: list[dict[str, str]] = []
    parts = _VOICE_MARKER_RE.split(script)

    # parts[0] is text before the first marker (usually empty or preamble)
    # then alternating: marker_name, text, marker_name, text, ...
    if parts[0].strip():
        sections.append({"voice": "HOST", "text": parts[0].strip()})

    for i in range(1, len(parts) - 1, 2):
        voice = parts[i].strip()
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if text:
            sections.append({"voice": voice, "text": text})

    return sections


def _resolve_voice_assignments(
    sections: list[dict[str, str]],
    voice_map: dict[str, str] | None,
) -> dict[str, str]:
    """Build voice marker -> Kokoro voice ID mapping."""
    assignments = dict(_DEFAULT_VOICE_MAP)
    if voice_map:
        assignments.update(voice_map)

    # Ensure all voices used in sections have assignments
    for section in sections:
        voice = section["voice"]
        if voice not in assignments:
            assignments[voice] = _DEFAULT_VOICE_MAP.get("HOST", "af_heart")

    return assignments


@registry.register(
    "audio_briefing_compose",
    category="content",
    description="Compose multi-voice audio briefing script from article summaries",
    parallelizable=True,
    config_model=AudioBriefingComposeConfig,
    tags=["content", "audio", "briefing"],
)
async def run_audio_briefing_compose_adapter(
    config: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Compose a multi-voice audio briefing script from article summaries.

    Config:
      - items: list[dict] - Article summaries [{title, summary, url}]
      - target_audio_minutes: int - Target duration (default 10)
      - provider: str - LLM provider
      - model: str - LLM model
      - multi_voice: bool - Enable multi-voice markers (default True)
      - voice_map: dict - Override voice assignments
    Output:
      - text: str - Full script text
      - script: str - Alias for text
      - sections: list[dict] - Parsed sections for multi-voice TTS
      - voice_assignments: dict - Voice marker -> Kokoro voice ID
      - word_count: int
      - estimated_minutes: float
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    items = config.get("items") or []

    if not items:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            items = prev.get("items") or prev.get("results") or []

    if not items:
        return {"text": "", "script": "", "sections": [], "error": "missing_items"}

    output_language_cfg = config.get("output_language", "en")
    if isinstance(output_language_cfg, str):
        output_language_cfg = apply_template_to_string(output_language_cfg, context) or output_language_cfg
    output_language = _normalize_output_language(output_language_cfg)

    normalized_items = [_normalize_item(entry) for entry in items]
    normalized_items = [entry for entry in normalized_items if entry.get("title") or entry.get("summary")]
    if not normalized_items:
        return {"text": "", "script": "", "sections": [], "error": "missing_items"}

    persona_summarize = bool(config.get("persona_summarize", False))
    persona_provider_cfg = config.get("persona_provider") or config.get("provider")
    persona_model_cfg = config.get("persona_model") or config.get("model")
    persona_id_cfg = config.get("persona_id")
    if persona_summarize:
        normalized_items = await _persona_pre_summarize_items(
            normalized_items,
            output_language=output_language,
            provider=persona_provider_cfg,
            model=persona_model_cfg,
            persona_id=str(persona_id_cfg).strip() if persona_id_cfg is not None else None,
        )

    items = normalized_items

    multi_voice = config.get("multi_voice", True)
    target_minutes = config.get("target_audio_minutes", 10)
    target_words = target_minutes * 150
    voice_map_cfg = config.get("voice_map")
    if isinstance(voice_map_cfg, str):
        voice_map_cfg = apply_template_to_string(voice_map_cfg, context) or voice_map_cfg

    system_prompt = config.get("system_prompt_override") or _build_system_prompt(
        target_words, multi_voice, output_language
    )

    # Build items text for LLM
    items_text_parts: list[str] = []
    for i, item in enumerate(items[:30]):
        title = item.get("title", f"Story {i + 1}")
        summary = item.get("summary", "")
        items_text_parts.append(f"{i + 1}. {title}: {summary}")

    items_block = "\n".join(items_text_parts)

    prompt = f"""Write a spoken-word news briefing script covering these stories.
Target approximately {target_words} words ({target_minutes} minutes of audio).

Stories to cover:
{items_block}

Write the complete script now."""

    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        max_tokens = config.get("max_tokens") or max(target_words * 2, 2000)

        messages = [{"role": "user", "content": prompt}]
        response = await perform_chat_api_call_async(
            messages=messages,
            api_provider=config.get("provider"),
            model=config.get("model"),
            system_message=system_prompt,
            max_tokens=max_tokens,
            temperature=config.get("temperature", 0.5),
        )

        full_script = _strip_reasoning_blocks(extract_openai_content(response) or "").strip()

        if not full_script:
            return {"text": "", "script": "", "sections": [], "error": "empty_llm_response"}

        # Parse sections for multi-voice
        if multi_voice:
            sections = _parse_sections(full_script)
        else:
            sections = [{"voice": "HOST", "text": full_script}]

        # If no sections were parsed (LLM didn't use markers), wrap as single HOST section
        if not sections:
            sections = [{"voice": "HOST", "text": full_script}]

        voice_assignments = _resolve_voice_assignments(
            sections,
            voice_map_cfg if isinstance(voice_map_cfg, dict) else None,
        )

        word_count = len(full_script.split())
        estimated_minutes = round(word_count / 150, 1)

        return {
            "text": full_script,
            "script": full_script,
            "sections": sections,
            "word_count": word_count,
            "estimated_minutes": estimated_minutes,
            "voice_assignments": voice_assignments,
        }

    except _BRIEFING_NONCRITICAL_EXCEPTIONS:
        logger.exception("Audio briefing compose error")
        return {"text": "", "script": "", "sections": [], "error": "audio_briefing_compose_error"}
