"""
prompt_quality.py
Heuristic + optional LLM-backed prompt quality scoring for Prompt Studio.

Provides a deterministic heuristic score in [0..10] and an optional
cheap LLM fallback (configurable) with caching to reduce token usage.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
)


class PromptQualityScorer:
    """Prompt quality scoring with optional LLM fallback and caching."""

    def __init__(self, executor: Any | None = None, scorer_model: str | None = None, cache_ttl_sec: int = 600):
        self._executor = executor
        self._scorer_model = scorer_model
        self._cache_ttl = cache_ttl_sec
        self._cache: dict[str, tuple[float, float]] = {}  # key -> (score, expires_at)
        self._on_tokens = None  # Optional callback(int)

    def set_executor(self, executor: Any) -> None:
        self._executor = executor

    def set_model(self, model_name: str | None) -> None:
        self._scorer_model = model_name

    def set_token_callback(self, cb) -> None:
        self._on_tokens = cb

    def _heuristic(self, *, system_text: str, user_text: str) -> float:
        score = 5.0
        sys_len = len(system_text.strip())
        usr_len = len(user_text.strip())
        if 60 <= sys_len <= 800:
            score += 2.0
        elif sys_len < 20 or sys_len > 1200:
            score -= 1.0
        if 20 <= usr_len <= 2000:
            score += 1.0
        elif usr_len < 5:
            score -= 1.0
        action_terms = [
            "provide", "return", "output", "ensure", "follow", "validate",
            "step-by-step", "json", "format", "constraints",
        ]
        hits = sum(1 for t in action_terms if t in system_text.lower())
        score += min(2.0, hits * 0.4)
        ambiguity = ["maybe", "perhaps", "guess", "try to", "sort of"]
        amb_hits = sum(1 for t in ambiguity if t in system_text.lower())
        score -= min(1.5, amb_hits * 0.5)
        vars_user = set(re.findall(r"\{(\w+)\}", user_text))
        vars_sys = set(re.findall(r"\{(\w+)\}", system_text))
        if vars_user:
            overlap = len(vars_user & vars_sys)
            score += min(1.5, overlap * 0.5)
        return float(max(0.0, min(10.0, score)))

    def _cache_key(
        self,
        system_text: str,
        user_text: str,
        scorer_model: str | None = None,
        cache_scope: str | None = None,
    ) -> str:
        h = hashlib.sha256()
        h.update(system_text.encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(user_text.encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update((scorer_model or self._scorer_model or "").encode("utf-8"))
        h.update(b"\0")
        h.update((cache_scope or "").encode("utf-8", errors="ignore"))
        return h.hexdigest()

    async def score_prompt_async(
        self,
        *,
        system_text: str,
        user_text: str,
        model_config: dict[str, Any] | None = None,
        scorer_model: str | None = None,
        provider_credentials: ProviderCallCredentials | None = None,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
        strict_provider_errors: bool = False,
        cache_scope: str | None = None,
        use_cache: bool = True,
    ) -> float:
        """Return a quality score in [0..10] with optional LLM assist.

        Combines heuristic with LLM score (if configured) using a simple blend.
        Uses a TTL cache to reduce token usage.
        """
        base = self._heuristic(system_text=system_text, user_text=user_text)
        selected_model = scorer_model or self._scorer_model
        if not (self._executor and selected_model):
            return base

        key = self._cache_key(
            system_text,
            user_text,
            selected_model,
            cache_scope,
        )
        now = time.time()
        hit = self._cache.get(key) if use_cache else None
        if hit and hit[1] > now:
            return hit[0]

        prompt = (
            "Rate the clarity and effectiveness of this system prompt (0-10).\n"
            "Only return a number.\n\nSystem prompt:\n" + system_text[:1500] + "\n\n"
            "User prompt (context):\n" + (user_text[:1500] if user_text else "") + "\n\nScore:"
        )
        try:
            selected_config = model_config or {}
            parameters = dict(selected_config.get("parameters") or {})
            parameters.update({"temperature": 0.0, "max_tokens": 5})
            res = await self._executor._call_llm(
                provider=str(selected_config.get("provider") or "openai"),
                model=selected_model,
                prompt=prompt,
                parameters=parameters,
                api_key_override=selected_config.get("api_key"),
                app_config=selected_config.get("app_config"),
                credentials_resolved=(
                    selected_config.get("credentials_resolved") is True
                ),
                provider_credentials=provider_credentials,
                timeout_seconds=parameters.get("timeout_seconds"),
                on_provider_success=on_provider_success,
            )
            text = (res or {}).get("content", "").strip()
            if self._on_tokens:
                with contextlib.suppress(Exception):
                    self._on_tokens(int((res or {}).get("tokens", 0) or 0))
            m = re.search(r"\d+(?:\.\d+)?", text)
            llm_score = float(m.group(0)) if m else base
            final = float(max(0.0, min(10.0, 0.6 * base + 0.4 * llm_score)))
            if use_cache:
                self._cache[key] = (final, now + self._cache_ttl)
            return final
        except Exception:
            if strict_provider_errors:
                raise
            return base

    @staticmethod
    def score_to_bin(score: float, bin_size: float) -> int:
        bs = max(1e-6, float(bin_size))
        # Clamp score to [0,10] then bin consistently
        s = max(0.0, min(10.0, float(score)))
        return int(s // bs)
