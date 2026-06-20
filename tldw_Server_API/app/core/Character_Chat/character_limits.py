"""Character chat guardrails that are not rate limits."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, status
from loguru import logger

from tldw_Server_API.app.core.config import settings


@dataclass(frozen=True)
class CharacterLimits:
    max_characters: int
    max_import_size_mb: int
    max_chats_per_user: int
    max_messages_per_chat: int
    max_messages_per_chat_soft: int

    @classmethod
    def from_settings(cls, *, overrides: dict[str, Any] | None = None) -> CharacterLimits:
        def _env_int(name: str, configured_value: Any, fallback: int) -> int:
            raw = os.getenv(name)
            if raw is not None:
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    logger.warning("Invalid environment override for {}: {!r}. Using defaults.", name, raw)
            if configured_value is not None:
                try:
                    return int(configured_value)
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid configured value for {}: {!r}. Using fallback {}.",
                        name,
                        configured_value,
                        fallback,
                    )
            return fallback

        overrides = overrides or {}
        max_characters = _env_int(
            "MAX_CHARACTERS_PER_USER",
            overrides.get("MAX_CHARACTERS_PER_USER", settings.get("MAX_CHARACTERS_PER_USER", 1000)),
            1000,
        )
        max_import_size_mb = _env_int(
            "MAX_CHARACTER_IMPORT_SIZE_MB",
            overrides.get("MAX_CHARACTER_IMPORT_SIZE_MB", settings.get("MAX_CHARACTER_IMPORT_SIZE_MB", 10)),
            10,
        )
        max_chats_per_user = _env_int(
            "MAX_CHATS_PER_USER",
            overrides.get("MAX_CHATS_PER_USER", settings.get("MAX_CHATS_PER_USER", 100000)),
            100000,
        )
        max_messages_per_chat = _env_int(
            "MAX_MESSAGES_PER_CHAT",
            overrides.get("MAX_MESSAGES_PER_CHAT", settings.get("MAX_MESSAGES_PER_CHAT", 1000)),
            1000,
        )
        max_messages_per_chat_soft = _env_int(
            "MAX_MESSAGES_PER_CHAT_SOFT",
            overrides.get("MAX_MESSAGES_PER_CHAT_SOFT", settings.get("MAX_MESSAGES_PER_CHAT_SOFT")),
            max_messages_per_chat,
        )
        return cls(
            max_characters=max_characters,
            max_import_size_mb=max_import_size_mb,
            max_chats_per_user=max_chats_per_user,
            max_messages_per_chat=max_messages_per_chat,
            max_messages_per_chat_soft=max_messages_per_chat_soft,
        )


_limits: CharacterLimits | None = None


def get_character_limits() -> CharacterLimits:
    global _limits
    if _limits is None:
        _limits = CharacterLimits.from_settings()
    return _limits


def check_character_limit(user_id: int, current_count: int, limits: CharacterLimits | None = None) -> bool:
    limits = limits or get_character_limits()
    if current_count >= limits.max_characters:
        logger.warning(
            "Character limit exceeded for user {}: {}/{}",
            user_id,
            current_count,
            limits.max_characters,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Character limit exceeded. Maximum {limits.max_characters} characters allowed.",
        )
    return True


def check_import_size(file_size_bytes: int, limits: CharacterLimits | None = None) -> bool:
    limits = limits or get_character_limits()
    max_bytes = int(limits.max_import_size_mb) * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the limit of {limits.max_import_size_mb} MB",
        )
    return True


def check_chat_limit(user_id: int, current_chat_count: int, limits: CharacterLimits | None = None) -> bool:
    limits = limits or get_character_limits()
    if current_chat_count >= limits.max_chats_per_user:
        logger.warning(
            "Chat limit exceeded for user {}: {}/{}",
            user_id,
            current_chat_count,
            limits.max_chats_per_user,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Chat limit exceeded. Maximum {limits.max_chats_per_user} concurrent chats allowed."
            ),
        )
    return True


def check_message_limit(chat_id: str, projected_message_count: int, limits: CharacterLimits | None = None) -> bool:
    limits = limits or get_character_limits()
    if projected_message_count > limits.max_messages_per_chat:
        logger.warning(
            "Message limit exceeded for chat {}: {}/{}",
            chat_id,
            projected_message_count,
            limits.max_messages_per_chat,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Message limit exceeded. Maximum {limits.max_messages_per_chat} messages per chat.",
        )
    return True


def check_soft_message_limit(chat_id: str, current_message_count: int, limits: CharacterLimits | None = None) -> bool:
    """Enforce a soft cap for non-persisted completion requests."""
    limits = limits or get_character_limits()
    if limits.max_messages_per_chat_soft <= 0:
        return True
    if current_message_count >= limits.max_messages_per_chat_soft:
        logger.warning(
            "Soft message limit exceeded for chat {}: {}/{}",
            chat_id,
            current_message_count,
            limits.max_messages_per_chat_soft,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Soft message limit exceeded for non-persisted completions. "
                f"Maximum {limits.max_messages_per_chat_soft} messages per chat."
            ),
        )
    return True
