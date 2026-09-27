"""
note_title.py
Heuristic and optional LLM-backed title generation for Notes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from tldw_Server_API.app.core.config import settings as core_settings
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Prompt_Management.service_prompts import ResolvedServicePrompt

# Public strategy type shared with the Notes API.
TitleStrategy = Literal["heuristic", "llm", "llm_fallback"]


@dataclass
class TitleGenOptions:
    strategy: TitleStrategy = "heuristic"
    max_len: int = 250
    language: str | None = None


def _normalize_text_for_title(text: str) -> str:
    """Lightweight cleanup to extract a reasonable title string from free text.

    - Trim whitespace
    - Remove leading Markdown syntax (#, *, -, >) for the first line
    - Strip code fence markers and inline backticks
    - Collapse multiple spaces
    - Prefer first non-empty line; if line is long, use first sentence-like chunk
    """
    if not text:
        return ""

    # Normalize line endings and strip outer whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    # Remove code blocks (fences) and inline backticks
    # Remove fenced code blocks completely as they often aren't descriptive titles
    fenced_code_pattern = re.compile(r"```.*?```", re.DOTALL)
    text_wo_fences = re.sub(fenced_code_pattern, "", text)
    text_wo_backticks = text_wo_fences.replace("`", "")

    # Split into lines and find first non-empty
    for raw_line in text_wo_backticks.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        # Strip common markdown list/items/heading markers (headings, blockquotes, lists, ordered lists)
        line = re.sub(r"^(?:#+|>+|[-*+]|\d+\.)\s*", "", line)
        # Strip markdown links: [text](url) -> text
        line = re.sub(r"\[(?P<text>[^\]]+)\]\([^\)]+\)", r"\g<text>", line)
        # Strip image syntax: ![alt](url) -> alt
        line = re.sub(r"!\[(?P<alt>[^\]]*)\]\([^\)]+\)", r"\g<alt>", line)
        # Collapse inner whitespace
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            # If line is too long, try to cut to first sentence-ish delimiter
            # Prefer ., !, ? or em-dash ; fall back to hard cut later
            sentence_cut = re.split(r"(?<=[.!?])\s+|\s+—\s+|\s+-\s+", line, maxsplit=1)
            candidate = sentence_cut[0].strip() if sentence_cut else line
            return candidate

    return ""


def _truncate_title(title: str, max_len: int) -> str:
    title = title.strip()
    if len(title) <= max_len:
        return title
    # Prefer cutting at a word boundary
    truncated = title[: max_len + 1]
    # Backtrack to last whitespace to avoid mid-word cuts
    ws_idx = truncated.rfind(" ")
    if ws_idx > 0:
        truncated = truncated[:ws_idx]
    truncated = truncated.rstrip(" -—:;,.…")
    return truncated


def _fallback_timestamp_title(language: str | None) -> str:
    # Keep language hint implicit; do not attempt translation locally
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return f"Note from {ts}"


def generate_note_title_heuristic(content: str, max_len: int = 250, *, language: str | None = None) -> str:
    """Heuristic-only title generator.

    Best-effort extraction of a concise, descriptive title from content.
    Falls back to a timestamp-based title when content is empty or non-informative.
    """
    if content is None:
        content = ""
    base = _normalize_text_for_title(content)
    if not base:
        return _truncate_title(_fallback_timestamp_title(language), max_len)
    # Remove trailing punctuation often not ideal in titles
    base = base.rstrip(" .:;,-—…")
    if not base:
        base = _fallback_timestamp_title(language)
    return _truncate_title(base, max_len)


def generate_note_title(
    content: str,
    *,
    options: TitleGenOptions | None = None,
    service_prompt: ResolvedServicePrompt | None = None,
) -> str:
    """Generate a title, using a resolved prompt for enabled LLM strategies."""
    if options is None:
        options = TitleGenOptions()
    # LLM strategies remain feature-gated and retain the heuristic fallback.
    try_llm = bool(core_settings.get("NOTES_TITLE_LLM_ENABLED", False))
    if try_llm and options.strategy in ("llm", "llm_fallback"):
        if service_prompt is None:
            raise RuntimeError("Service Prompt resolution is required for LLM note titles.")
        llm_title = _try_generate_title_llm(
            content,
            options,
            system_message=service_prompt.parts["system"],
            title_instruction=service_prompt.parts["title_instruction"],
        )
        if llm_title:
            return _truncate_title(llm_title, options.max_len)
        # Fallback to heuristic on failure
    # Default and fallback path.
    return generate_note_title_heuristic(content, max_len=options.max_len, language=options.language)


def _try_generate_title_llm(
    content: str,
    options: TitleGenOptions,
    *,
    system_message: str,
    title_instruction: str,
) -> str | None:
    """Best-effort LLM title generation. Returns None on failure.

    Keeps synchronous control path; relies on local/adapter-backed sync helpers.
    """
    registry = get_registry()
    adapter = registry.get_adapter("openai")
    if adapter is None:
        return None

    try:
        content_snippet = (content or "").strip()
        if len(content_snippet) > 2000:
            content_snippet = content_snippet[:2000]
        user_msg = (
            f"{title_instruction} no longer than {options.max_len} characters for the following note.\n"
            f"Return only the title with no quotes or extra text.\n\n{content_snippet}"
        )
        messages = [{"role": "user", "content": user_msg}]
        # Avoid streaming for simplicity; small token budget
        result = adapter.chat(
            {
                "messages": messages,
                "system_message": system_message,
                "model": None,
                "temperature": 0.2,
                "max_tokens": 128,
            }
        )
        # Adapter may return dict or provider-specific object.
        # Normalize common dict-like responses.
        if isinstance(result, dict):
            try:
                title = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            except Exception:
                title = ""
        else:
            # Fallback to string-like result if provider returns simple content
            title = str(result).strip()
        title = (title or "").strip().strip('"')
        # Basic sanitization
        title = re.sub(r"\s+", " ", title)
        if title:
            return title
    except Exception:
        # Silent failure; caller falls back to heuristic
        return None
    return None
