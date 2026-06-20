# character_rate_limiter.py
"""
Rate limiting for character operations.

**Phase 2 Deprecation Notice**:
Resource Governor is the single enforcement path. Non-rate-limit guardrails
live in character_limits.py and are wrapped here for compatibility.
When RG is unavailable or disabled, the shim fails open with a deprecation
warning. This shim will be removed in a future release.
"""

from __future__ import annotations

import asyncio
import os
import time
import warnings
from typing import Any

from fastapi import HTTPException, status
from loguru import logger

from tldw_Server_API.app.core.Character_Chat.character_limits import (
    CharacterLimits,
    get_character_limits,
)
from tldw_Server_API.app.core.Character_Chat.character_limits import (
    check_character_limit as _check_character_limit,
)
from tldw_Server_API.app.core.Character_Chat.character_limits import (
    check_chat_limit as _check_chat_limit,
)
from tldw_Server_API.app.core.Character_Chat.character_limits import (
    check_import_size as _check_import_size,
)
from tldw_Server_API.app.core.Character_Chat.character_limits import (
    check_message_limit as _check_message_limit,
)
from tldw_Server_API.app.core.Character_Chat.character_limits import (
    check_soft_message_limit as _check_soft_message_limit,
)
from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode, is_truthy

# Optional Resource Governor integration (gated by global RG_ENABLED/config)
RG_IMPORT_EXCEPTIONS = (ImportError, AttributeError)
RG_RUNTIME_EXCEPTIONS = (
    asyncio.TimeoutError,
    RuntimeError,
    OSError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
)

try:  # pragma: no cover - RG is optional
    from tldw_Server_API.app.core.config import rg_enabled  # type: ignore
    from tldw_Server_API.app.core.Resource_Governance import (  # type: ignore
        MemoryResourceGovernor,
        RedisResourceGovernor,
        RGRequest,
    )
    from tldw_Server_API.app.core.Resource_Governance.policy_loader import (  # type: ignore
        PolicyLoader,
        PolicyReloadConfig,
        default_policy_loader,
    )
except RG_IMPORT_EXCEPTIONS:  # pragma: no cover - safe fallback when RG not installed
    MemoryResourceGovernor = None  # type: ignore
    RedisResourceGovernor = None  # type: ignore
    RGRequest = None  # type: ignore
    PolicyLoader = None  # type: ignore
    PolicyReloadConfig = None  # type: ignore
    default_policy_loader = None  # type: ignore
    rg_enabled = None  # type: ignore

_CHARACTER_DEPRECATION_WARNED = False


def _emit_character_legacy_deprecation(context: str) -> None:
    global _CHARACTER_DEPRECATION_WARNED
    if _CHARACTER_DEPRECATION_WARNED:
        return
    _CHARACTER_DEPRECATION_WARNED = True
    msg = (
        "Character Chat legacy rate limiter is deprecated (Phase 2). "
        f"Context: {context}. Enable RG_ENABLED=true for enforcement. "
        "This shim will be removed in a future release."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    logger.warning(msg)


class CharacterRateLimiter:
    """
    Rate limiter for character operations backed by Resource Governor.

    Limits are per-user and apply to character operations and chat flows.
    """

    def __init__(
        self,
        redis_client: object | None = None,
        max_operations: int = 100,
        window_seconds: int = 3600,
        max_characters: int = 1000,
        max_import_size_mb: int = 10,
        max_chats_per_user: int = 100,
        max_messages_per_chat: int = 1000,
        max_messages_per_chat_soft: int | None = None,
        max_chat_completions_per_minute: int = 20,
        max_message_sends_per_minute: int = 60,
        enabled: bool = True,
    ):
        self.redis = redis_client
        self.max_operations = max_operations
        self.window_seconds = window_seconds
        self.max_chat_completions_per_minute = max_chat_completions_per_minute
        self.max_message_sends_per_minute = max_message_sends_per_minute
        self.enabled = bool(enabled)
        self.policy_id = os.getenv("RG_CHARACTER_CHAT_POLICY_ID", "character_chat.default")

        if max_messages_per_chat_soft is None:
            max_messages_per_chat_soft = max_messages_per_chat
        self._limits = CharacterLimits(
            max_characters=int(max_characters),
            max_import_size_mb=int(max_import_size_mb),
            max_chats_per_user=int(max_chats_per_user),
            max_messages_per_chat=int(max_messages_per_chat),
            max_messages_per_chat_soft=int(max_messages_per_chat_soft),
        )

        logger.info(
            "CharacterRateLimiter initialized (RG only): enabled={} policy_id={}",
            self.enabled,
            self.policy_id,
        )

    async def check_rate_limit(self, user_id: int, operation: str = "character_op") -> tuple[bool, int]:
        """
        Check if user has exceeded rate limit using ResourceGovernor.

        Returns (allowed, remaining). Remaining is best-effort and defaults to 0
        because RG does not expose per-user remaining counts here.
        """
        if not self.enabled:
            return True, 0

        if not _rg_character_enabled():
            _emit_character_legacy_deprecation("rg_disabled")
            return True, 0

        if not _rg_character_enforce_requests():
            return True, 0

        rg_decision = await _maybe_enforce_with_rg_character(
            user_id=user_id,
            operation=operation,
            policy_id=self.policy_id,
        )
        if rg_decision is None:
            _log_rg_character_fallback("rg_decision_unavailable")
            _emit_character_legacy_deprecation("rg_decision_unavailable")
            return True, 0

        if not rg_decision.get("allowed", False):
            retry_after = int(rg_decision.get("retry_after") or 60)
            logger.warning(
                "Character rate limit exceeded by ResourceGovernor for user {}: retry_after={}s",
                user_id,
                retry_after,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    "Rate limit exceeded. "
                    f"Policy {rg_decision.get('policy_id', self.policy_id)} denied request."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        return True, 0

    async def check_chat_completion_rate(self, user_id: int) -> tuple[bool, int]:
        return await self.check_rate_limit(user_id=user_id, operation="chat_completion")

    async def check_message_send_rate(self, user_id: int) -> tuple[bool, int]:
        return await self.check_rate_limit(user_id=user_id, operation="message_send")

    async def check_rate_limit_by_type(self, user_id: int, operation_type: str) -> tuple[bool, int]:
        return await self.check_rate_limit(user_id=user_id, operation=operation_type)

    async def get_usage_stats(self, user_id: int) -> dict[str, Any]:
        """Return best-effort RG usage info for character requests."""
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "policy_id": self.policy_id,
            "rate_limit_source": "resource_governor" if _rg_character_enabled() else "disabled",
        }
        if not _rg_character_enabled():
            return payload

        gov = await _get_character_rg_governor()
        if gov is None:
            return payload
        try:
            peek = await gov.peek_with_policy(
                entity=f"user:{user_id}",
                categories=["requests"],
                policy_id=self.policy_id,
            )
            payload["requests"] = peek.get("requests")
        except RG_RUNTIME_EXCEPTIONS:
            pass
        return payload

    async def check_character_limit(self, user_id: int, current_count: int) -> bool:
        return _check_character_limit(user_id, current_count, self._limits)

    def check_import_size(self, file_size_bytes: int) -> bool:
        return _check_import_size(file_size_bytes, self._limits)

    async def check_chat_limit(self, user_id: int, current_chat_count: int) -> bool:
        return _check_chat_limit(user_id, current_chat_count, self._limits)

    async def check_message_limit(self, chat_id: str, projected_message_count: int) -> bool:
        return _check_message_limit(chat_id, projected_message_count, self._limits)

    async def check_soft_message_limit(self, chat_id: str, current_message_count: int) -> bool:
        return _check_soft_message_limit(chat_id, current_message_count, self._limits)


# --- Resource Governor plumbing (optional) ---
_rg_char_governor = None
_rg_char_loader = None
_rg_char_lock = asyncio.Lock()
_rg_char_init_error: str | None = None
_rg_char_init_error_logged = False
_rg_char_fallback_logged = False


def _rg_char_context() -> dict[str, str]:
    policy_path = os.getenv(
        "RG_POLICY_PATH",
        "tldw_Server_API/Config_Files/resource_governor_policies.yaml",
    )
    try:
        policy_path_resolved = os.path.abspath(policy_path)
    except (OSError, ValueError, TypeError):
        policy_path_resolved = policy_path
    return {
        "backend": os.getenv("RG_BACKEND", "memory"),
        "policy_path": policy_path,
        "policy_path_resolved": policy_path_resolved,
        "policy_store": os.getenv("RG_POLICY_STORE", ""),
        "policy_reload_enabled": os.getenv("RG_POLICY_RELOAD_ENABLED", ""),
        "policy_reload_interval": os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", ""),
        "cwd": os.getcwd(),
    }


def _log_rg_character_init_failure(exc: Exception) -> None:
    global _rg_char_init_error, _rg_char_init_error_logged
    _rg_char_init_error = repr(exc)
    if _rg_char_init_error_logged:
        return
    _rg_char_init_error_logged = True
    ctx = _rg_char_context()
    logger.exception(
        "Character Chat ResourceGovernor init failed; enforcement disabled. "
        "backend={backend} policy_path={policy_path} policy_path_resolved={policy_path_resolved} "
        "policy_store={policy_store} reload_enabled={policy_reload_enabled} "
        "reload_interval={policy_reload_interval} cwd={cwd}",
        **ctx,
    )


def _log_rg_character_fallback(reason: str) -> None:
    global _rg_char_fallback_logged
    if _rg_char_fallback_logged:
        return
    _rg_char_fallback_logged = True
    ctx = _rg_char_context()
    logger.error(
        "Character Chat ResourceGovernor unavailable; skipping character rate limits. "
        "reason={} init_error={} backend={backend} policy_path={policy_path} "
        "policy_path_resolved={policy_path_resolved} policy_store={policy_store} "
        "reload_enabled={policy_reload_enabled} reload_interval={policy_reload_interval} cwd={cwd}",
        reason,
        _rg_char_init_error,
        **ctx,
    )


def _rg_character_enabled() -> bool:
    if rg_enabled is not None:
        try:
            return bool(rg_enabled(True))  # type: ignore[func-returns-value]
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return False
    return False


def _rg_character_enforce_requests() -> bool:
    """
    Control whether CharacterRateLimiter should enforce RG request limits.

    Default is disabled to avoid double-enforcement when RG middleware already
    governs ingress routes.
    """
    val = os.getenv("RG_CHARACTER_CHAT_ENFORCE_REQUESTS", "0").strip().lower()
    return is_truthy(val)


async def _get_character_rg_governor():
    global _rg_char_governor, _rg_char_loader
    if not _rg_character_enabled():
        return None
    if RGRequest is None or PolicyLoader is None:
        _log_rg_character_fallback("rg_components_unavailable")
        return None
    if _rg_char_governor is not None:
        return _rg_char_governor
    async with _rg_char_lock:
        if _rg_char_governor is not None:
            return _rg_char_governor
        try:
            loader = (
                default_policy_loader()
                if default_policy_loader
                else PolicyLoader(
                    os.getenv(
                        "RG_POLICY_PATH",
                        "tldw_Server_API/Config_Files/resource_governor_policies.yaml",
                    ),
                    PolicyReloadConfig(
                        enabled=True,
                        interval_sec=int(os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", "10") or "10"),
                    ),
                )
            )
            await loader.load_once()
            _rg_char_loader = loader
            backend = os.getenv("RG_BACKEND", "memory").lower()
            if backend == "redis" and RedisResourceGovernor is not None:
                gov = RedisResourceGovernor(policy_loader=loader)  # type: ignore[call-arg]
            else:
                gov = MemoryResourceGovernor(policy_loader=loader)  # type: ignore[call-arg]
            _rg_char_governor = gov
            return gov
        except RG_RUNTIME_EXCEPTIONS as exc:  # pragma: no cover - optional path
            _log_rg_character_init_failure(exc)
            return None


async def _maybe_enforce_with_rg_character(
    *,
    user_id: int,
    operation: str,
    policy_id: str,
) -> dict[str, object] | None:
    """
    Optionally enforce Character Chat operations via ResourceGovernor.

    Returns a decision dict when RG is used, or None when RG is
    unavailable or disabled.
    """
    gov = await _get_character_rg_governor()
    if gov is None:
        return None
    op_id = f"character-{user_id}-{operation}-{time.time_ns()}"
    try:
        decision, handle = await gov.reserve(
            RGRequest(
                entity=f"user:{user_id}",
                categories={"requests": {"units": 1}},
                tags={
                    "policy_id": policy_id,
                    "module": "character_chat",
                    "operation": operation,
                },
            ),
            op_id=op_id,
        )
        if decision.allowed:
            if handle:
                try:
                    await gov.commit(handle, None, op_id=op_id)
                except (RuntimeError, ValueError, TypeError, AttributeError):
                    logger.debug("Character Chat RG commit failed", exc_info=True)
            return {"allowed": True, "retry_after": None, "policy_id": policy_id}
        return {
            "allowed": False,
            "retry_after": decision.retry_after or 1,
            "policy_id": policy_id,
        }
    except RG_RUNTIME_EXCEPTIONS as exc:
        logger.debug("Character Chat RG reserve failed: {}", exc)
        return None


# Global instance (initialized in dependencies)
_rate_limiter: CharacterRateLimiter | None = None


def get_character_rate_limiter() -> CharacterRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter

    # Honor TEST_MODE by returning permissive limits for guardrails.
    test_mode = is_test_mode()

    # Security check: warn if TEST_MODE is enabled in what looks like production
    if test_mode:
        env_name = os.getenv("ENVIRONMENT", os.getenv("ENV", "")).lower()
        prod_flag = env_flag_enabled("tldw_production")
        if env_name in ("production", "prod", "live") or prod_flag:
            logger.critical(
                "TEST_MODE is enabled in a production environment! "
                "This disables rate limiting and is a security risk. "
                "Unset TEST_MODE environment variable immediately."
            )

    def _env_int_or_test_default(name: str, test_default: int) -> int:
        raw = os.getenv(name)
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
        return test_default

    if test_mode:
        if _rate_limiter is None or not getattr(_rate_limiter, "_is_test_mode", False):
            _rate_limiter = CharacterRateLimiter(
                redis_client=None,
                max_operations=_env_int_or_test_default("CHARACTER_RATE_LIMIT_OPS", 1_000_000_000),
                window_seconds=3600,
                max_characters=_env_int_or_test_default("MAX_CHARACTERS_PER_USER", 1_000_000_000),
                max_import_size_mb=_env_int_or_test_default("MAX_CHARACTER_IMPORT_SIZE_MB", 1_000),
                max_chats_per_user=_env_int_or_test_default("MAX_CHATS_PER_USER", 1_000_000_000),
                max_messages_per_chat=_env_int_or_test_default("MAX_MESSAGES_PER_CHAT", 1_000_000_000),
                max_messages_per_chat_soft=_env_int_or_test_default("MAX_MESSAGES_PER_CHAT_SOFT", 1_000_000_000),
                max_chat_completions_per_minute=_env_int_or_test_default("MAX_CHAT_COMPLETIONS_PER_MINUTE", 1_000_000_000),
                max_message_sends_per_minute=_env_int_or_test_default("MAX_MESSAGE_SENDS_PER_MINUTE", 1_000_000_000),
                enabled=True,
            )
            _rate_limiter._is_test_mode = True
        return _rate_limiter

    if _rate_limiter is None:
        from tldw_Server_API.app.core.config import settings

        def _env_bool(name: str, configured_value: Any, fallback: bool) -> bool:
            raw = os.getenv(name)
            if raw is not None:
                s = str(raw).strip().lower()
                if is_truthy(s):
                    return True
                if s in {"0", "false", "no", "off"}:
                    return False
            if configured_value is not None:
                try:
                    return bool(configured_value)
                except (ValueError, TypeError, AttributeError) as exc:
                    logger.debug("Character rate limit enabled flag coercion failed: {}", exc)
            return fallback

        limits = get_character_limits()
        single_user_mode = bool(settings.get("SINGLE_USER_MODE", False))
        configured_enabled = settings.get("CHARACTER_RATE_LIMIT_ENABLED", None)
        default_enabled = not single_user_mode
        enabled_flag = _env_bool("CHARACTER_RATE_LIMIT_ENABLED", configured_enabled, default_enabled)

        _rate_limiter = CharacterRateLimiter(
            redis_client=None,
            max_operations=int(settings.get("CHARACTER_RATE_LIMIT_OPS", 100) or 100),
            window_seconds=int(settings.get("CHARACTER_RATE_LIMIT_WINDOW", 3600) or 3600),
            max_characters=limits.max_characters,
            max_import_size_mb=limits.max_import_size_mb,
            max_chats_per_user=limits.max_chats_per_user,
            max_messages_per_chat=limits.max_messages_per_chat,
            max_messages_per_chat_soft=limits.max_messages_per_chat_soft,
            max_chat_completions_per_minute=int(settings.get("MAX_CHAT_COMPLETIONS_PER_MINUTE", 20) or 20),
            max_message_sends_per_minute=int(settings.get("MAX_MESSAGE_SENDS_PER_MINUTE", 60) or 60),
            enabled=enabled_flag,
        )

    return _rate_limiter
