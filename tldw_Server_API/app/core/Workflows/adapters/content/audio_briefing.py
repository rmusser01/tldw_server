"""Audio briefing script composition adapter.

Composes a multi-voice audio narration script from article summaries via LLM,
with section markers and voice assignments for downstream multi-voice TTS.
"""

from __future__ import annotations

import re
import uuid
import json
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string
from tldw_Server_API.app.core.Workflows.adapters._common import (
    extract_openai_content,
    resolve_artifacts_dir,
)
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

_VOICE_MARKER_RE = re.compile(r"^\[([A-Z_]+)\]:\s*", re.MULTILINE)
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
    summary = str(getattr(item, "summary", "") or getattr(item, "snippet", "") or "").strip()
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


def _normalize_voice_marker(value: Any) -> str:
    """Normalize an audio-cast speaker id/label into a script marker."""
    marker = "".join(char.upper() if char.isalnum() else "_" for char in str(value or "").strip()).strip("_")
    return marker or "HOST"


def _coerce_audio_cast_speakers(value: Any) -> list[dict[str, str]]:
    """Coerce a structured audio_cast object into prompt-ready speaker records."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, dict):
        return []
    speakers = value.get("speakers")
    if not isinstance(speakers, list):
        return []

    normalized: list[dict[str, str]] = []
    for speaker in speakers[:4]:
        if not isinstance(speaker, dict):
            continue
        raw_marker = speaker.get("id") or speaker.get("label")
        marker = _normalize_voice_marker(raw_marker)
        label = str(speaker.get("label") or marker.replace("_", " ").title()).strip()
        role = str(speaker.get("role") or "").strip()
        voice = str(speaker.get("voice") or "").strip()
        persona = str(speaker.get("persona") or "").strip()
        normalized.append(
            {
                "marker": marker,
                "label": label,
                "role": role,
                "voice": voice,
                "persona": persona,
            }
        )
    return normalized


def _voice_map_from_audio_cast(speakers: list[dict[str, str]]) -> dict[str, str]:
    return {
        speaker["marker"]: speaker["voice"] for speaker in speakers if speaker.get("marker") and speaker.get("voice")
    }


def _build_system_prompt(
    target_words: int,
    multi_voice: bool,
    output_language: str,
    audio_cast_speakers: list[dict[str, str]] | None = None,
) -> str:
    """Build the system prompt for LLM script composition."""
    voice_instructions = ""
    if multi_voice:
        if audio_cast_speakers:
            speaker_lines = []
            for speaker in audio_cast_speakers:
                descriptor = speaker["label"]
                if speaker.get("role"):
                    descriptor = f"{descriptor}, {speaker['role']}"
                if speaker.get("persona"):
                    descriptor = f"{descriptor}, persona: {speaker['persona']}"
                speaker_lines.append(f"- [{speaker['marker']}]: {descriptor}")
            voice_instructions = (
                "\nUse only these voice markers to indicate speaker changes:\n"
                + "\n".join(speaker_lines)
                + "\n\nEvery line of spoken text MUST start with one of these exact voice markers."
            )
        else:
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


def _register_script_artifact(
    *,
    context: dict[str, Any],
    script: str,
    sections: list[dict[str, str]],
    voice_assignments: dict[str, str],
    output_language: str,
    word_count: int,
    estimated_minutes: float,
) -> dict[str, str] | None:
    """Persist the generated briefing script as a workflow artifact."""
    add_artifact = context.get("add_artifact")
    if not callable(add_artifact):
        return None

    try:
        import time as _time

        step_run_id = str(context.get("step_run_id") or f"audio_script_{int(_time.time() * 1000)}")
        art_dir = resolve_artifacts_dir(step_run_id)
        art_dir.mkdir(parents=True, exist_ok=True)
        script_path = art_dir / "briefing_script.md"
        script_path.write_text(script, encoding="utf-8")
        artifact_id = f"audio_script_{uuid.uuid4()}"
        add_artifact(
            type="audio_script",
            uri=f"file://{script_path}",
            size_bytes=script_path.stat().st_size,
            mime_type="text/markdown",
            metadata={
                "script_artifact": True,
                "title": "Briefing script",
                "sections_count": len(sections),
                "voice_assignments": voice_assignments,
                "output_language": output_language,
                "word_count": word_count,
                "estimated_minutes": estimated_minutes,
            },
            artifact_id=artifact_id,
        )
        return {
            "artifact_id": artifact_id,
            "uri": f"file://{script_path}",
        }
    except _BRIEFING_NONCRITICAL_EXCEPTIONS:
        logger.warning("Audio briefing script artifact registration failed", exc_info=True)
        return None


@registry.register(
    "audio_briefing_compose",
    category="content",
    description="Compose multi-voice audio briefing script from article summaries",
    parallelizable=True,
    config_model=AudioBriefingComposeConfig,
    tags=["content", "audio", "briefing"],
)
async def run_audio_briefing_compose_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
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
    audio_cast_cfg = config.get("audio_cast")
    if isinstance(audio_cast_cfg, str):
        audio_cast_cfg = apply_template_to_string(audio_cast_cfg, context) or audio_cast_cfg
    audio_cast_speakers = _coerce_audio_cast_speakers(audio_cast_cfg)
    voice_map_cfg = config.get("voice_map")
    if isinstance(voice_map_cfg, str):
        voice_map_cfg = apply_template_to_string(voice_map_cfg, context) or voice_map_cfg
    cast_voice_map = _voice_map_from_audio_cast(audio_cast_speakers)
    if isinstance(voice_map_cfg, dict):
        voice_map_cfg = {**cast_voice_map, **voice_map_cfg}
    elif cast_voice_map:
        voice_map_cfg = cast_voice_map

    system_prompt = config.get("system_prompt_override") or _build_system_prompt(
        target_words, multi_voice, output_language, audio_cast_speakers
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
        script_artifact = _register_script_artifact(
            context=context,
            script=full_script,
            sections=sections,
            voice_assignments=voice_assignments,
            output_language=output_language,
            word_count=word_count,
            estimated_minutes=estimated_minutes,
        )

        result: dict[str, Any] = {
            "text": full_script,
            "script": full_script,
            "sections": sections,
            "word_count": word_count,
            "estimated_minutes": estimated_minutes,
            "voice_assignments": voice_assignments,
        }
        if script_artifact:
            result["script_artifact_id"] = script_artifact["artifact_id"]
            result["script_artifact_uri"] = script_artifact["uri"]
        return result

    except _BRIEFING_NONCRITICAL_EXCEPTIONS:
        logger.exception("Audio briefing compose error")
        return {"text": "", "script": "", "sections": [], "error": "audio_briefing_compose_error"}
