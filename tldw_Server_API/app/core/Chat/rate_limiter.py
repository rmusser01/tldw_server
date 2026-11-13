# rate_limiter.py
# Description: Advanced rate limiting with per-conversation and per-user limits
#
# Phase 2 Deprecation Notice
# ─────────────────────────────────────────────────────────────
# This module is a **Phase 2 legacy shim**. Primary rate-limit enforcement is
# handled by the Resource Governor (``RGSimpleMiddleware`` + per-module
# ``_maybe_enforce_with_rg_chat``). When RG is unavailable or disabled, the
# shim fails open (no counters, no enforcement) with a deprecation warning.
# This shim will be removed in a future release.
#
# ``TokenBucket`` is still imported by ``command_router.py`` and is preserved.
# ─────────────────────────────────────────────────────────────
#
# Imports
import asyncio
import os
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.testing import is_test_mode

_RATE_LIMITER_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ConnectionError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

_LEGACY_TEST_LIMIT_KEYS: tuple[str, ...] = (
    "TEST_CHAT_PER_USER_RPM",
    "TEST_CHAT_PER_CONVERSATION_RPM",
    "TEST_CHAT_GLOBAL_RPM",
    "TEST_CHAT_TOKENS_PER_MINUTE",
    "TEST_CHAT_BURST_MULTIPLIER",
)

#######################################################################################################################
#
# Types:

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    global_rpm: int = 60  # Global requests per minute
    per_user_rpm: int = 20  # Per-user requests per minute
    per_conversation_rpm: int = 10  # Per-conversation requests per minute
    per_user_tokens_per_minute: int = 10000  # Token limit per user
    burst_multiplier: float = 1.5  # Allow burst up to 1.5x normal rate

@dataclass
class UsageStats:
    """Usage statistics for tracking."""
    request_count: int = 0
    token_count: int = 0
    last_request_time: Optional[float] = None
    conversation_request_counts: dict[str, int] = None

    def __post_init__(self):
        if self.conversation_request_counts is None:
            self.conversation_request_counts = {}

#######################################################################################################################
#
# Classes:

class TokenBucket:
    """
    Token bucket algorithm for rate limiting with burst support.
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were available, False otherwise
        """
        async with self._lock:
            # Refill bucket
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            # Try to consume
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_for_tokens(self, tokens: int = 1, timeout: float = 60) -> bool:
        """
        Wait for tokens to become available.

        Args:
            tokens: Number of tokens needed
            timeout: Maximum wait time

        Returns:
            True if tokens obtained, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if await self.consume(tokens):
                return True

            # Calculate wait time
            needed = tokens - self.tokens
            wait_time = min(needed / self.refill_rate, 1.0)
            await asyncio.sleep(wait_time)

        return False

    async def refund(self, tokens: int = 1) -> None:
        """
        Safely return tokens to the bucket using the internal lock.

        Args:
            tokens: Number of tokens to return (clamped to capacity)
        """
        if tokens <= 0:
            return
        async with self._lock:
            self.tokens = min(self.capacity, self.tokens + tokens)


_CHAT_DEPRECATION_WARNED = False


def _emit_chat_legacy_deprecation(context: str) -> None:
    global _CHAT_DEPRECATION_WARNED
    if _CHAT_DEPRECATION_WARNED:
        return
    _CHAT_DEPRECATION_WARNED = True
    msg = (
        "Chat legacy rate limiter is deprecated (Phase 2). "
        f"Context: {context}. Enable RG_ENABLED=true for enforcement. "
        "This shim will be removed in a future release."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    logger.warning(msg)


class ConversationRateLimiter:
    """
    Rate limiter with per-conversation, per-user, and global limits.

    **Phase 2 Deprecation Notice**:
    When ResourceGovernor is enabled, chat ingress is governed via RG and this
    shim delegates entirely. When RG is disabled or unavailable, the shim fails
    open (no counters, no enforcement) with a deprecation warning. This shim
    will be removed in a future release.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._state_lock = asyncio.Lock()
        # Lightweight in-memory counters used only for deterministic TEST_MODE
        # legacy enforcement when RG is explicitly disabled.
        self._global_requests: list[float] = []
        self._user_requests: dict[str, list[float]] = {}
        self._conversation_requests: dict[str, list[float]] = {}
        self._user_tokens: dict[str, list[tuple[float, int]]] = {}

    def _effective_limit(self, base_limit: int) -> int:
        try:
            burst = float(self.config.burst_multiplier)
        except _RATE_LIMITER_NONCRITICAL_EXCEPTIONS:
            burst = 1.0
        return max(1, int(base_limit * max(1.0, burst)))

    @staticmethod
    def _prune_timestamps(samples: list[float], now_ts: float, window_seconds: float = 60.0) -> None:
        cutoff = now_ts - window_seconds
        while samples and samples[0] < cutoff:
            samples.pop(0)

    @staticmethod
    def _prune_token_samples(samples: list[tuple[float, int]], now_ts: float, window_seconds: float = 60.0) -> None:
        cutoff = now_ts - window_seconds
        while samples and samples[0][0] < cutoff:
            samples.pop(0)

    async def _check_legacy_test_limits(
        self,
        *,
        user_id: str,
        conversation_id: Optional[str],
        estimated_tokens: int,
    ) -> tuple[bool, Optional[str]]:
        now_ts = time.time()
        user_key = str(user_id)
        conversation_key = str(conversation_id) if conversation_id else None
        token_units = max(0, int(estimated_tokens or 0))

        async with self._state_lock:
            self._prune_timestamps(self._global_requests, now_ts)
            user_samples = self._user_requests.setdefault(user_key, [])
            self._prune_timestamps(user_samples, now_ts)

            conversation_samples: list[float] | None = None
            if conversation_key is not None:
                conversation_samples = self._conversation_requests.setdefault(conversation_key, [])
                self._prune_timestamps(conversation_samples, now_ts)

            token_samples = self._user_tokens.setdefault(user_key, [])
            self._prune_token_samples(token_samples, now_ts)

            if len(self._global_requests) >= self._effective_limit(self.config.global_rpm):
                return False, "Global rate limit exceeded"

            if len(user_samples) >= self._effective_limit(self.config.per_user_rpm):
                return False, "User rate limit exceeded"

            if (
                conversation_samples is not None
                and len(conversation_samples) >= self._effective_limit(self.config.per_conversation_rpm)
            ):
                return False, "Conversation rate limit exceeded"

            if token_units > 0:
                used_tokens = sum(amount for _, amount in token_samples)
                if used_tokens + token_units > self._effective_limit(self.config.per_user_tokens_per_minute):
                    return False, "Token rate limit exceeded"

            # Reserve the capacity on successful checks.
            self._global_requests.append(now_ts)
            user_samples.append(now_ts)
            if conversation_samples is not None:
                conversation_samples.append(now_ts)
            if token_units > 0:
                token_samples.append((now_ts, token_units))

        return True, None

    async def check_rate_limit(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        estimated_tokens: int = 0
    ) -> tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits.

        Args:
            user_id: User identifier
            conversation_id: Optional conversation identifier
            estimated_tokens: Estimated token count for the request

        Returns:
            Tuple of (allowed, error_message)
        """
        if not _rg_chat_primary_enabled():
            if is_test_mode() and any(os.getenv(name) is not None for name in _LEGACY_TEST_LIMIT_KEYS):
                return await self._check_legacy_test_limits(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    estimated_tokens=estimated_tokens,
                )
            # RG is disabled - fail open with deprecation warning.
            _emit_chat_legacy_deprecation("rg_disabled")
            return True, None

        # RG is enabled - try RG enforcement first
        try:
            rg_decision = await _maybe_enforce_with_rg_chat(  # type: ignore[name-defined]
                user_id=user_id,
                conversation_id=conversation_id,
                estimated_tokens=estimated_tokens,
            )
        except NameError:
            rg_decision = None
        except _RATE_LIMITER_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - defensive
            logger.debug("Chat RG enforcement failed: {}", exc)
            rg_decision = None

        if rg_decision is None:
            # RG unavailable: fail open with deprecation warning.
            _log_rg_chat_fallback("rg_decision_unavailable")
            _emit_chat_legacy_deprecation("rg_decision_unavailable")
            return True, None

        if not rg_decision.get("allowed", False):
            policy_id = rg_decision.get("policy_id", "chat.default")
            retry_after = rg_decision.get("retry_after")
            base_msg = f"Rate limit exceeded (ResourceGovernor policy={policy_id})"
            if isinstance(retry_after, (int, float)) and retry_after >= 0:
                return False, f"{base_msg}; retry_after={int(retry_after)}s"
            return False, base_msg

        return True, None

    async def wait_for_capacity(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        estimated_tokens: int = 0,
        timeout: float = 60
    ) -> tuple[bool, Optional[str]]:
        """
        Wait for rate limit capacity to become available.

        Args:
            user_id: User identifier
            conversation_id: Optional conversation identifier
            estimated_tokens: Estimated token count
            timeout: Maximum wait time

        Returns:
            Tuple of (allowed, error_message)
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            allowed, error = await self.check_rate_limit(
                user_id, conversation_id, estimated_tokens
            )

            if allowed:
                return True, None

            # Wait a bit before retrying
            await asyncio.sleep(0.5)

        return False, "Timeout waiting for rate limit capacity"

    async def get_usage_stats(self, user_id: str) -> dict[str, Any]:
        """
        Get usage statistics for a user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with usage statistics
        """
        return {
            "rate_limit_source": "resource_governor",
            "config": {
                "per_user_rpm": self.config.per_user_rpm,
                "per_conversation_rpm": self.config.per_conversation_rpm,
                "per_user_tokens_per_minute": self.config.per_user_tokens_per_minute,
            },
        }

# Global rate limiter instance
_rate_limiter: Optional[ConversationRateLimiter] = None
_rate_limiter_init_lock = threading.Lock()

def get_rate_limiter() -> Optional[ConversationRateLimiter]:
    """Get the global rate limiter instance."""
    return _rate_limiter

def initialize_rate_limiter(config: Optional[RateLimitConfig] = None) -> ConversationRateLimiter:
    """
    Initialize the global rate limiter.

    This function is thread-safe and uses locking to prevent race conditions
    during concurrent initialization.

    Args:
        config: Rate limit configuration (uses defaults if None)

    Returns:
        The initialized rate limiter
    """
    global _rate_limiter
    with _rate_limiter_init_lock:
        config = config or RateLimitConfig()
        # Test-mode overrides via environment for deterministic integration testing
        try:
            if is_test_mode():
                per_user = os.getenv("TEST_CHAT_PER_USER_RPM")
                per_conv = os.getenv("TEST_CHAT_PER_CONVERSATION_RPM")
                global_rpm = os.getenv("TEST_CHAT_GLOBAL_RPM")
                tokens_per_min = os.getenv("TEST_CHAT_TOKENS_PER_MINUTE")
                # Deterministic tests: disable burst by default in TEST_MODE unless explicitly overridden
                burst_mult = os.getenv("TEST_CHAT_BURST_MULTIPLIER")
                if per_user:
                    config.per_user_rpm = max(1, int(per_user))
                if per_conv:
                    config.per_conversation_rpm = max(1, int(per_conv))
                if global_rpm:
                    config.global_rpm = max(1, int(global_rpm))
                if tokens_per_min:
                    config.per_user_tokens_per_minute = max(1, int(tokens_per_min))
                # Default burst to 1.0 in TEST_MODE to make 429s deterministic on the N+1th call
                config.burst_multiplier = float(burst_mult) if burst_mult is not None else 1.0
        except (TypeError, ValueError):
            # Ignore env parse errors and use defaults
            pass
        _rate_limiter = ConversationRateLimiter(config)
        return _rate_limiter


# --- Resource Governor plumbing (optional) ---------------------------------
_rg_chat_governor = None
_rg_chat_loader = None
_rg_chat_lock = asyncio.Lock()
_rg_chat_init_error: Optional[str] = None
_rg_chat_init_error_logged = False
_rg_chat_fallback_logged = False


def _rg_chat_context() -> dict[str, str]:
    policy_path = os.getenv(
        "RG_POLICY_PATH",
        "tldw_Server_API/Config_Files/resource_governor_policies.yaml",
    )
    try:
        policy_path_resolved = os.path.abspath(policy_path)
    except (OSError, TypeError, ValueError):
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


def _log_rg_chat_init_failure(exc: Exception) -> None:
    global _rg_chat_init_error, _rg_chat_init_error_logged
    _rg_chat_init_error = repr(exc)
    if _rg_chat_init_error_logged:
        return
    _rg_chat_init_error_logged = True
    ctx = _rg_chat_context()
    logger.exception(
        "Chat ResourceGovernor init failed; compatibility shim remains diagnostics-only. "
        "backend={backend} policy_path={policy_path} policy_path_resolved={policy_path_resolved} "
        "policy_store={policy_store} reload_enabled={policy_reload_enabled} "
        "reload_interval={policy_reload_interval} cwd={cwd}",
        **ctx,
    )


def _log_rg_chat_fallback(reason: str) -> None:
    global _rg_chat_fallback_logged
    if _rg_chat_fallback_logged:
        return
    _rg_chat_fallback_logged = True
    ctx = _rg_chat_context()
    logger.error(
        "Chat ResourceGovernor unavailable; using diagnostics-only compatibility shim (no enforcement). "
        "reason={} init_error={} backend={backend} policy_path={policy_path} "
        "policy_path_resolved={policy_path_resolved} policy_store={policy_store} "
        "reload_enabled={policy_reload_enabled} reload_interval={policy_reload_interval} cwd={cwd}",
        reason,
        _rg_chat_init_error,
        **ctx,
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
except ImportError:  # pragma: no cover - safe fallback when RG not installed
    MemoryResourceGovernor = None  # type: ignore
    RedisResourceGovernor = None  # type: ignore
    RGRequest = None  # type: ignore
    PolicyLoader = None  # type: ignore
    PolicyReloadConfig = None  # type: ignore
    default_policy_loader = None  # type: ignore
    rg_enabled = None  # type: ignore


def _rg_chat_enabled() -> bool:
    """Return True when RG should gate chat requests."""
    if rg_enabled is not None:
        try:
            return bool(rg_enabled(True))  # type: ignore[func-returns-value]
        except _RATE_LIMITER_NONCRITICAL_EXCEPTIONS:
            return False
    return False


def _rg_chat_primary_enabled() -> bool:
    """
    Return True when ResourceGovernor should be treated as the primary
    source of truth for chat rate limiting decisions.
    """
    return _rg_chat_enabled()


async def _get_chat_rg_governor():
    """Lazily initialize a ResourceGovernor instance for Chat."""
    global _rg_chat_governor, _rg_chat_loader
    if not _rg_chat_enabled():
        return None
    if RGRequest is None or PolicyLoader is None:
        _log_rg_chat_fallback("rg_components_unavailable")
        return None
    if _rg_chat_governor is not None:
        return _rg_chat_governor
    async with _rg_chat_lock:
        if _rg_chat_governor is not None:
            return _rg_chat_governor
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
                        interval_sec=int(
                            os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", "10") or "10"
                        ),
                    ),
                )
            )
            await loader.load_once()
            _rg_chat_loader = loader
            backend = os.getenv("RG_BACKEND", "memory").lower()
            if backend == "redis" and RedisResourceGovernor is not None:
                gov = RedisResourceGovernor(policy_loader=loader)  # type: ignore[call-arg]
            else:
                gov = MemoryResourceGovernor(policy_loader=loader)  # type: ignore[call-arg]
            _rg_chat_governor = gov
            return gov
        except _RATE_LIMITER_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - optional path
            _log_rg_chat_init_failure(exc)
            return None


async def _maybe_enforce_with_rg_chat(
    *,
    user_id: str,
    conversation_id: Optional[str],
    estimated_tokens: int,
) -> Optional[dict[str, object]]:
    """
    Optionally enforce Chat limits via ResourceGovernor.

    Requests are enforced at ingress via `RGSimpleMiddleware`. This helper is
    used for token accounting and MUST NOT reserve
    `requests` to avoid double-enforcement on RG-governed routes.

    Returns a decision dict when RG is used, or None when RG is unavailable or
    disabled.
    """
    gov = await _get_chat_rg_governor()
    if gov is None:
        return None
    policy_id = os.getenv("RG_CHAT_POLICY_ID", "chat.default")
    op_id = f"chat-{user_id}-{conversation_id or 'none'}-{time.time_ns()}"
    try:
        try:
            tokens_units = int(estimated_tokens or 0)
        except (TypeError, ValueError):
            tokens_units = 0
        tokens_units = max(0, tokens_units)

        # Only enforce token budgets here; request-rate limiting happens at ingress.
        categories: dict[str, dict[str, int]] = {}
        if tokens_units > 0:
            categories["tokens"] = {"units": tokens_units}
        else:
            # No token units to enforce; allow and bypass compatibility shim.
            return {"allowed": True, "retry_after": None, "policy_id": policy_id}

        decision, handle = await gov.reserve(
            RGRequest(
                entity=f"user:{user_id}",
                categories=categories,
                tags={
                    "policy_id": policy_id,
                    "module": "chat",
                    "endpoint": "/api/v1/chat/completions",
                },
            ),
            op_id=op_id,
        )
        if decision.allowed:
            if handle:
                try:
                    # Treat reserve as consumption; commit with the same units
                    # to keep semantics simple for now.
                    actuals: dict[str, int] = {}
                    if tokens_units > 0:
                        actuals["tokens"] = tokens_units
                    await gov.commit(handle, actuals=actuals, op_id=op_id)
                except _RATE_LIMITER_NONCRITICAL_EXCEPTIONS:
                    logger.debug("Chat RG commit failed", exc_info=True)
            return {"allowed": True, "retry_after": None, "policy_id": policy_id}
        return {
            "allowed": False,
            "retry_after": decision.retry_after or 1,
            "policy_id": policy_id,
        }
    except _RATE_LIMITER_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Chat RG reserve failed; using diagnostics-only shim path: {}", exc)
        return None
