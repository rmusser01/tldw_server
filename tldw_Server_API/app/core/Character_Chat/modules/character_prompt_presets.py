"""
Character prompt preset helpers.

Resolves per-character prompt preset metadata and formats
system prompts accordingly.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Character_Chat.modules.character_utils import (
    DEFAULT_CHARACTER_NAME,
    DEFAULT_USER_NAME,
    replace_placeholders,
)

DEFAULT_PROMPT_PRESET = "default"
ST_DEFAULT_PROMPT_PRESET = "st_default"

_VALID_PROMPT_PRESETS = {DEFAULT_PROMPT_PRESET, ST_DEFAULT_PROMPT_PRESET}


def _safe_replace(value: Any, char_name: str, user_name: str) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    if not text:
        return ""
    try:
        return replace_placeholders(text, char_name, user_name)
    except Exception as exc:
        logger.debug("Preset placeholder replacement failed: {}", exc)
        return text


def resolve_character_prompt_preset(character: dict[str, Any]) -> str:
    extensions: Any = character.get("extensions") if character else None
    if isinstance(extensions, str):
        try:
            extensions = json.loads(extensions)
        except json.JSONDecodeError:
            extensions = {}

    preset_value: Any = None
    if isinstance(extensions, dict):
        tldw_ext = extensions.get("tldw")
        if isinstance(tldw_ext, dict):
            preset_value = tldw_ext.get("prompt_preset") or tldw_ext.get("promptPreset")
        if not preset_value:
            preset_value = extensions.get("prompt_preset") or extensions.get("promptPreset")

    if isinstance(preset_value, str):
        preset = preset_value.strip()
        if preset in _VALID_PROMPT_PRESETS:
            return preset
        if preset:
            logger.debug("Unknown character prompt preset '{}'; using default.", preset)

    return DEFAULT_PROMPT_PRESET


def build_character_system_prompt(
    character: dict[str, Any],
    char_name: str | None,
    user_name: str | None,
    preset: str | None = None,
) -> str:
    resolved_char = char_name or DEFAULT_CHARACTER_NAME
    resolved_user = user_name or DEFAULT_USER_NAME
    resolved_preset = preset or resolve_character_prompt_preset(character or {})

    if resolved_preset == ST_DEFAULT_PROMPT_PRESET:
        sections: list[str] = []
        sections.append(f"You are {resolved_char}.")

        system_prompt = _safe_replace(character.get("system_prompt"), resolved_char, resolved_user)
        if system_prompt:
            sections.append(system_prompt)

        description = _safe_replace(character.get("description"), resolved_char, resolved_user)
        if description:
            sections.append(f"Description:\n{description}")

        personality = _safe_replace(character.get("personality"), resolved_char, resolved_user)
        if personality:
            sections.append(f"Personality:\n{personality}")

        scenario = _safe_replace(character.get("scenario"), resolved_char, resolved_user)
        if scenario:
            sections.append(f"Scenario:\n{scenario}")

        message_example = _safe_replace(character.get("message_example"), resolved_char, resolved_user)
        if message_example:
            sections.append(f"Example dialogue:\n{message_example}")

        post_history = _safe_replace(character.get("post_history_instructions"), resolved_char, resolved_user)
        if post_history:
            sections.append(f"Post-history instructions:\n{post_history}")

        return "\n\n".join([s for s in sections if s]).strip()

    parts = [
        f"You are {resolved_char}.",
        _safe_replace(character.get("description"), resolved_char, resolved_user),
        _safe_replace(character.get("personality"), resolved_char, resolved_user),
        _safe_replace(character.get("scenario"), resolved_char, resolved_user),
        _safe_replace(character.get("system_prompt"), resolved_char, resolved_user),
    ]
    return "\n".join([p for p in parts if p]).strip()


# ========================================================================
# Template tokens for custom presets
# ========================================================================

PRESET_TEMPLATE_TOKENS: dict[str, str] = {
    "{{char}}": "Character name",
    "{{user}}": "User/player name",
    "{{description}}": "Character description field",
    "{{personality}}": "Character personality field",
    "{{scenario}}": "Character scenario field",
    "{{system_prompt}}": "Character system prompt field",
    "{{message_example}}": "Character example messages",
    "{{post_history}}": "Post-history instructions",
}

# Maps token placeholders to character dict field names
_TOKEN_FIELD_MAP: dict[str, str] = {
    "{{description}}": "description",
    "{{personality}}": "personality",
    "{{scenario}}": "scenario",
    "{{system_prompt}}": "system_prompt",
    "{{message_example}}": "message_example",
    "{{post_history}}": "post_history_instructions",
}


def build_custom_system_prompt(
    character: dict[str, Any],
    char_name: str | None,
    user_name: str | None,
    section_order: list[str],
    section_templates: dict[str, str],
) -> str:
    """Build a system prompt from a user-defined preset with custom section ordering."""
    resolved_char = char_name or DEFAULT_CHARACTER_NAME
    resolved_user = user_name or DEFAULT_USER_NAME

    sections: list[str] = []
    for section_key in section_order:
        template = section_templates.get(section_key, "")
        if not template:
            continue

        # Replace {{char}} and {{user}} first
        rendered = template.replace("{{char}}", resolved_char).replace("{{user}}", resolved_user)

        # Replace field tokens with character data
        for token, field_name in _TOKEN_FIELD_MAP.items():
            if token in rendered:
                value = _safe_replace(character.get(field_name), resolved_char, resolved_user)
                rendered = rendered.replace(token, value)

        rendered = rendered.strip()
        if rendered:
            sections.append(rendered)

    return "\n\n".join(sections).strip()


def build_saved_builtin_system_prompt(
    character: dict[str, Any],
    char_name: str,
    user_name: str,
    preset: dict[str, Any],
) -> str:
    """Render frozen builtin templates without treating inserted field text as a template."""
    values = {
        token: _safe_replace(character.get(field), char_name, user_name) for token, field in _TOKEN_FIELD_MAP.items()
    }
    values.update({"{{char}}": char_name, "{{user}}": user_name})
    pattern = re.compile("|".join(re.escape(token) for token in values))
    sections = []
    for key in preset["section_order"]:
        template = preset["section_templates"].get(key, "")
        field_tokens = [token for token in _TOKEN_FIELD_MAP if token in template]
        if field_tokens and not any(values[token] for token in field_tokens):
            continue
        rendered = pattern.sub(lambda match: values[match.group()], template)
        if rendered:
            sections.append(rendered)
    separator = "\n" if preset["preset_id"] == DEFAULT_PROMPT_PRESET else "\n\n"
    return separator.join(sections).strip()


def get_builtin_presets() -> list[dict[str, Any]]:
    """Return metadata for the built-in presets."""
    return [
        {
            "preset_id": DEFAULT_PROMPT_PRESET,
            "name": "Default",
            "builtin": True,
            "section_order": ["identity", "description", "personality", "scenario", "system_prompt"],
            "section_templates": {
                "identity": "You are {{char}}.",
                "description": "{{description}}",
                "personality": "{{personality}}",
                "scenario": "{{scenario}}",
                "system_prompt": "{{system_prompt}}",
            },
        },
        {
            "preset_id": ST_DEFAULT_PROMPT_PRESET,
            "name": "SillyTavern Default",
            "builtin": True,
            "section_order": [
                "identity", "system_prompt", "description",
                "personality", "scenario", "message_example", "post_history",
            ],
            "section_templates": {
                "identity": "You are {{char}}.",
                "system_prompt": "{{system_prompt}}",
                "description": "Description:\n{{description}}",
                "personality": "Personality:\n{{personality}}",
                "scenario": "Scenario:\n{{scenario}}",
                "message_example": "Example dialogue:\n{{message_example}}",
                "post_history": "Post-history instructions:\n{{post_history}}",
            },
        },
    ]
