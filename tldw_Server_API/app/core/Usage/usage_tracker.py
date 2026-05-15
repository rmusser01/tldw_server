"""
usage_tracker.py

Async helpers to log per-request LLM usage and compute costs.
Integrates with AuthNZ DatabasePool for both SQLite and Postgres.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from datetime import date, datetime, timezone
from sqlite3 import Error as SQLiteError
from typing import Any, Mapping

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.ip_allowlist import resolve_client_ip
from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.Metrics import increment_counter

from .llm_usage_normalizer import normalize_llm_usage
from .pricing_catalog import get_pricing_catalog

try:  # pragma: no cover - ledger optional during upgrades/tests
    from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import (  # type: ignore
        LedgerEntry,
        ResourceDailyLedger,
    )
except ImportError:  # pragma: no cover - safe fallback
    LedgerEntry = None  # type: ignore
    ResourceDailyLedger = None  # type: ignore

_USAGE_NONCRITICAL_EXCEPTIONS = (
    OSError,
    RuntimeError,
    SQLiteError,
    TimeoutError,
    TypeError,
    ValueError,
)

_tokens_daily_ledger: ResourceDailyLedger | None = None  # type: ignore[name-defined]
_tokens_daily_ledger_lock = asyncio.Lock()
_tokens_legacy_backfill_done: set[str] = set()


async def _get_tokens_daily_ledger() -> ResourceDailyLedger | None:
    global _tokens_daily_ledger
    if ResourceDailyLedger is None or LedgerEntry is None:
        return None
    if _tokens_daily_ledger is not None:
        return _tokens_daily_ledger
    async with _tokens_daily_ledger_lock:
        if _tokens_daily_ledger is not None:
            return _tokens_daily_ledger
        try:
            ledger = ResourceDailyLedger()  # type: ignore[call-arg]
            await ledger.initialize()
            _tokens_daily_ledger = ledger
            return ledger
        except _USAGE_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - defensive
            logger.debug("LLM usage ResourceDailyLedger init failed; tokens/day caps disabled")
            _tokens_daily_ledger = None
            return None


async def backfill_legacy_tokens_to_ledger(
    *,
    entity_scope: str,
    entity_value: str,
    day_utc: str | None = None,
) -> None:
    """
    Best-effort migration helper: mirror today's tokens usage from ``llm_usage_log``
    into the shared ResourceDailyLedger once per process/entity/day.

    This preserves in-progress daily token caps when upgrading from versions
    that only wrote to ``llm_usage_log`` (and not the ledger).

    This function is fail-open and never raises.
    """
    if ResourceDailyLedger is None or LedgerEntry is None:
        return

    scope = str(entity_scope or "").strip()
    value = str(entity_value or "").strip()
    if scope not in {"user", "api_key"} or not value:
        return

    # Only numeric api_key ids map to llm_usage_log.key_id.
    if scope == "api_key":
        try:
            int(value)
        except (TypeError, ValueError):
            return

    day = day_utc or datetime.now(timezone.utc).date().isoformat()
    key = f"{scope}:{value}:{day}"
    if key in _tokens_legacy_backfill_done:
        return

    try:
        ledger = await _get_tokens_daily_ledger()
        if ledger is None:
            _tokens_legacy_backfill_done.add(key)
            return

        used = await ledger.total_for_day(
            entity_scope=scope,
            entity_value=value,
            category="tokens",
            day_utc=day,
        )
        used_int = int(used or 0)

        try:
            day_val = date.fromisoformat(day)
        except (TypeError, ValueError):
            day_val = datetime.now(timezone.utc).date()

        pool: DatabasePool = await get_db_pool()
        repo = AuthnzUsageRepo(pool)
        legacy_total = 0
        if scope == "user":
            legacy_total = int((await repo.summarize_user_day(user_id=int(value), day=day_val)).get("tokens") or 0)
        else:
            legacy_total = int((await repo.summarize_key_day(key_id=int(value), day=day_val)).get("tokens") or 0)

        delta = int(legacy_total) - int(used_int)
        if delta > 0:
            entry = LedgerEntry(  # type: ignore[call-arg]
                entity_scope=scope,
                entity_value=value,
                category="tokens",
                units=int(delta),
                op_id=f"tokens-legacy:{scope}:{value}:{day}",
                occurred_at=datetime.now(timezone.utc),
            )
            await ledger.add(entry)
    except _USAGE_NONCRITICAL_EXCEPTIONS:
        return
    finally:
        _tokens_legacy_backfill_done.add(key)


def _enabled() -> bool:
    try:
        settings = get_settings()
        val = getattr(settings, "LLM_USAGE_ENABLED", True)
        # Allow env to override
        env_val = os.getenv("LLM_USAGE_ENABLED")
        if env_val is not None:
            return str(env_val).strip().lower() in {"true", "1", "yes", "y", "on"}
        return bool(val)
    except _USAGE_NONCRITICAL_EXCEPTIONS:
        return True


def _derive_request_remote_ip(request: Any, settings: Any) -> str | None:
    try:
        resolved = resolve_client_ip(request, settings)
        if resolved:
            text = str(resolved).strip()
            return text or None
    except Exception:
        resolved = None
    try:
        peer = getattr(getattr(request, "client", None), "host", None)
        if peer:
            text = str(peer).strip()
            return text or None
    except Exception:
        peer = None
    return None


def _derive_request_user_agent(request: Any) -> str | None:
    try:
        headers = getattr(request, "headers", None)
        if headers is None:
            return None
        ua = headers.get("user-agent") or headers.get("User-Agent")
        if ua is None:
            return None
        text = str(ua).strip()
        return text or None
    except Exception:
        return None


def _apply_pii_settings_to_meta(
    *,
    remote_ip: str | None,
    user_agent: str | None,
    settings: Any,
) -> tuple[str | None, str | None]:
    if bool(getattr(settings, "USAGE_LOG_DISABLE_META", False)):
        return None, None
    if not bool(getattr(settings, "PII_REDACT_LOGS", False)):
        return remote_ip, user_agent

    ip_out: str | None
    salt = getattr(settings, "API_KEY_PEPPER", None) or getattr(settings, "JWT_SECRET_KEY", None)
    if remote_ip:
        if salt:
            try:
                digest = hmac.new(
                    str(salt).encode("utf-8"),
                    str(remote_ip).encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                ip_out = f"hash:{digest[:16]}"
            except Exception:
                ip_out = "redacted"
        else:
            ip_out = "redacted"
    else:
        ip_out = None
    # Preserve usage_log behavior: suppress user agent when PII redaction is enabled.
    return ip_out, ""


def compute_costs(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cache_read_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
    billable_input_tokens: int | None = None,
) -> tuple[float, float, float, bool]:
    """
    Compute (prompt_cost, completion_cost, total_cost, estimated)
    given provider, model and token counts.
    """
    catalog = get_pricing_catalog()
    rates, est = catalog.get_rate_details(provider, model)
    in_per_1k = rates["prompt"]
    out_per_1k = rates["completion"]
    prompt_total = max(0, int(prompt_tokens or 0))
    completion_total = max(0, int(completion_tokens or 0))
    cache_read = min(max(0, int(cache_read_input_tokens or 0)), prompt_total)
    cache_write = min(max(0, int(cache_write_input_tokens or 0)), max(0, prompt_total - cache_read))

    if "cache_read" not in rates and "cache_write" not in rates:
        prompt_cost = (prompt_total / 1000.0) * in_per_1k
    else:
        if billable_input_tokens is None:
            normal_input = max(0, prompt_total - cache_read - cache_write)
        else:
            normal_input = min(max(0, int(billable_input_tokens or 0)), prompt_total)
        cache_read_rate = rates.get("cache_read", in_per_1k)
        cache_write_rate = rates.get("cache_write", in_per_1k)
        prompt_cost = (
            (normal_input / 1000.0) * in_per_1k
            + (cache_read / 1000.0) * cache_read_rate
            + (cache_write / 1000.0) * cache_write_rate
        )
    completion_cost = (completion_total / 1000.0) * out_per_1k
    total_cost = prompt_cost + completion_cost
    return prompt_cost, completion_cost, total_cost, est


async def log_llm_usage(
    *,
    user_id: int | None,
    key_id: int | None,
    endpoint: str,
    operation: str,
    provider: str,
    model: str,
    status: int,
    latency_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int | None = None,
    currency: str = "USD",
    request_id: str | None = None,
    estimated: bool | None = None,
    request: Any | None = None,
    remote_ip: str | None = None,
    user_agent: str | None = None,
    token_name: str | None = None,
    conversation_id: str | None = None,
    usage_metadata: Mapping[str, Any] | None = None,
    choice_count: int | None = None,
    estimate_source: str | None = None,
    prompt_fingerprint: str | None = None,
    prompt_fingerprint_version: str | None = None,
    world_book_fingerprint: str | None = None,
) -> None:
    """
    Insert a single llm_usage_log row. Computes costs if needed.

    This function is best-effort and should never raise; errors are logged.
    """
    if not _enabled():
        return

    try:
        pt = int(prompt_tokens or 0)
        ct = int(completion_tokens or 0)
        tt = int(total_tokens) if total_tokens is not None else pt + ct
        resolved_estimate_source = estimate_source
        if resolved_estimate_source is None and usage_metadata is None:
            resolved_estimate_source = "missing_usage" if estimated else "provider_usage"
        # Normalize provider cache/cost metadata without changing the legacy
        # prompt/completion/total token columns used by existing callers.
        normalized_usage = normalize_llm_usage(
            provider=provider,
            usage=usage_metadata,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=tt,
            choice_count=choice_count,
            estimate_source=resolved_estimate_source,
        )
        raw_usage_metadata_json = None
        if usage_metadata is not None:
            raw_usage_metadata_json = json.dumps(
                dict(normalized_usage.raw_usage_metadata),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )

        p_cost, c_cost, t_cost, est_flag = compute_costs(
            provider,
            model,
            pt,
            ct,
            cache_read_input_tokens=normalized_usage.cache_read_input_tokens,
            cache_write_input_tokens=normalized_usage.cache_write_input_tokens,
            billable_input_tokens=normalized_usage.billable_input_tokens,
        )
        if estimated is None:
            estimated = est_flag

        settings = get_settings()
        effective_remote_ip = remote_ip
        effective_user_agent = user_agent
        if request is not None:
            if effective_remote_ip is None:
                effective_remote_ip = _derive_request_remote_ip(request, settings)
            if effective_user_agent is None:
                effective_user_agent = _derive_request_user_agent(request)
        effective_remote_ip, effective_user_agent = _apply_pii_settings_to_meta(
            remote_ip=effective_remote_ip,
            user_agent=effective_user_agent,
            settings=settings,
        )

        # Record cost and tokens in Prometheus metrics (best-effort)
        try:
            increment_counter(
                "llm_cost_dollars",
                float(t_cost),
                labels={"provider": str(provider or "unknown"), "model": str(model or "unknown")},
            )
            # Per-user and per-operation breakdowns
            if user_id is not None:
                increment_counter(
                    "llm_cost_dollars_by_user",
                    float(t_cost),
                    labels={
                        "provider": str(provider or "unknown"),
                        "model": str(model or "unknown"),
                        "user_id": str(user_id),
                    },
                )
            if operation:
                increment_counter(
                    "llm_cost_dollars_by_operation",
                    float(t_cost),
                    labels={
                        "provider": str(provider or "unknown"),
                        "model": str(model or "unknown"),
                        "operation": str(operation or ""),
                    },
                )
            if pt:
                increment_counter(
                    "llm_tokens_used_total",
                    float(pt),
                    labels={"provider": str(provider or "unknown"), "model": str(model or "unknown"), "type": "prompt"},
                )
                if user_id is not None:
                    increment_counter(
                        "llm_tokens_used_total_by_user",
                        float(pt),
                        labels={
                            "provider": str(provider or "unknown"),
                            "model": str(model or "unknown"),
                            "type": "prompt",
                            "user_id": str(user_id),
                        },
                    )
                if operation:
                    increment_counter(
                        "llm_tokens_used_total_by_operation",
                        float(pt),
                        labels={
                            "provider": str(provider or "unknown"),
                            "model": str(model or "unknown"),
                            "type": "prompt",
                            "operation": str(operation or ""),
                        },
                    )
            if ct:
                increment_counter(
                    "llm_tokens_used_total",
                    float(ct),
                    labels={"provider": str(provider or "unknown"), "model": str(model or "unknown"), "type": "completion"},
                )
                if user_id is not None:
                    increment_counter(
                        "llm_tokens_used_total_by_user",
                        float(ct),
                        labels={
                            "provider": str(provider or "unknown"),
                            "model": str(model or "unknown"),
                            "type": "completion",
                            "user_id": str(user_id),
                        },
                    )
                if operation:
                    increment_counter(
                        "llm_tokens_used_total_by_operation",
                        float(ct),
                        labels={
                            "provider": str(provider or "unknown"),
                            "model": str(model or "unknown"),
                            "type": "completion",
                            "operation": str(operation or ""),
                        },
                    )
        except _USAGE_NONCRITICAL_EXCEPTIONS:
            # Metrics must never impact request flow
            pass

        db_pool: DatabasePool = await get_db_pool()
        repo = AuthnzUsageRepo(db_pool)
        effective_token_name = token_name
        try:
            if not effective_token_name and key_id is not None:
                effective_token_name = await repo.get_api_key_name(key_id=int(key_id))
        except Exception:
            effective_token_name = token_name
        await repo.insert_llm_usage_log(
            user_id=user_id,
            key_id=key_id,
            endpoint=endpoint,
            operation=operation,
            provider=provider,
            model=model,
            status=int(status),
            latency_ms=int(latency_ms),
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=tt,
            prompt_cost_usd=float(p_cost),
            completion_cost_usd=float(c_cost),
            total_cost_usd=float(t_cost),
            currency=currency,
            estimated=bool(estimated),
            request_id=request_id,
            remote_ip=effective_remote_ip,
            user_agent=effective_user_agent,
            token_name=effective_token_name,
            conversation_id=(str(conversation_id).strip() if conversation_id is not None else None),
            cached_input_tokens=normalized_usage.cached_input_tokens,
            cache_write_input_tokens=normalized_usage.cache_write_input_tokens,
            cache_read_input_tokens=normalized_usage.cache_read_input_tokens,
            billable_input_tokens=normalized_usage.billable_input_tokens,
            reasoning_tokens=normalized_usage.reasoning_tokens,
            choice_count=normalized_usage.choice_count,
            estimate_source=normalized_usage.estimate_source,
            prompt_fingerprint=prompt_fingerprint,
            prompt_fingerprint_version=prompt_fingerprint_version,
            world_book_fingerprint=world_book_fingerprint,
            raw_usage_metadata_json=raw_usage_metadata_json,
        )

        # Shadow-write daily token usage into the shared ResourceDailyLedger so
        # ResourceGovernor can enforce tokens-per-day caps cross-module.
        try:
            if tt > 0:
                ledger = await _get_tokens_daily_ledger()
                if ledger is not None and LedgerEntry is not None:
                    entity_scope = None
                    entity_value = None
                    try:
                        if user_id is not None:
                            entity_scope = "user"
                            entity_value = str(int(user_id))
                        elif key_id is not None:
                            entity_scope = "api_key"
                            entity_value = str(int(key_id))
                    except (TypeError, ValueError):
                        entity_scope = None
                        entity_value = None

                    if entity_scope and entity_value:
                        rid = str(request_id or "").strip()
                        if rid:
                            op_id = f"llm:{rid}:{operation}:{provider}:{model}:{pt}:{ct}:{tt}"
                        else:
                            op_id = f"llm:{operation}:{provider}:{model}:{int(time.time())}:{pt}:{ct}:{tt}"
                        entry = LedgerEntry(  # type: ignore[call-arg]
                            entity_scope=entity_scope,
                            entity_value=entity_value,
                            category="tokens",
                            units=int(tt),
                            op_id=str(op_id),
                            occurred_at=datetime.now(timezone.utc),
                        )
                        await ledger.add(entry)
        except _USAGE_NONCRITICAL_EXCEPTIONS:
            # Ledger writes must never affect request flow
            pass
    except _USAGE_NONCRITICAL_EXCEPTIONS:
        # Never break request processing due to logging errors
        logger.debug("LLM usage logging skipped/failed")
