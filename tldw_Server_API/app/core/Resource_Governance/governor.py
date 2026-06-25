from __future__ import annotations

"""
Core Resource Governor (memory backend) with idempotency and metrics.

Implements a minimal in-process governor suitable for unit/integration tests
and single-node development. It provides:
  - Token bucket / sliding window for requests/tokens categories
  - Concurrency leases with TTL for streams/jobs categories
  - Idempotent reserve/commit/refund via op_id
  - Monotonic time source injection for deterministic tests
  - Basic policy resolution (policy_id → rules) and strictest-wins across
    global + entity scope

This module does not wire HTTP middleware; that integration happens in the
API layer. Minutes/day ledger durability is left to a separate DAL; this
memory governor implements only in-memory counting for the 'minutes' category.
"""

import contextlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

from .daily_caps import check_daily_cap, consume_daily_cap
from .metrics_rg import _labels, ensure_rg_metrics_registered, rg_metrics_entity_label_enabled
from .tenant import hash_entity

try:
    # Metrics are optional during early startup
    from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
except (ImportError, ModuleNotFoundError):  # pragma: no cover - metrics optional
    get_metrics_registry = None  # type: ignore


TimeSource = Callable[[], float]


@dataclass(frozen=True)
class RGRequest:
    entity: str  # format: "scope:value" (e.g., "user:123")
    categories: dict[str, dict[str, int]]  # e.g., {"requests": {"units": 1}}
    tags: dict[str, str] = field(default_factory=dict)  # endpoint, service, policy_id, etc.


@dataclass
class RGDecision:
    allowed: bool
    retry_after: int | None
    details: dict[str, Any]


class ResourceGovernor:
    async def check(self, req: RGRequest) -> RGDecision:  # pragma: no cover - interface
        raise NotImplementedError

    async def reserve(self, req: RGRequest, op_id: str | None = None) -> tuple[RGDecision, str | None]:  # pragma: no cover - interface
        raise NotImplementedError

    async def commit(self, handle_id: str, actuals: dict[str, int] | None = None, op_id: str | None = None) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def refund(self, handle_id: str, deltas: dict[str, int] | None = None, op_id: str | None = None) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def renew(self, handle_id: str, ttl_s: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def release(self, handle_id: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def peek(self, entity: str, categories: list[str]) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    async def query(self, entity: str, category: str) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    async def reset(self, entity: str, category: str | None = None) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def capabilities(self) -> dict[str, Any]:  # pragma: no cover - interface
        """Return backend capability diagnostics for debugging.

        Implementations should include at least:
          - backend: str
          - real_redis: bool (if applicable)
          - tokens_lua_loaded / multi_lua_loaded: bool (if applicable)
        """
        return {"backend": "unknown"}


# --- Token bucket primitives ---


@dataclass
class _Bucket:
    capacity: float
    refill_per_sec: float
    tokens: float
    last_refill: float

    def refill(self, now: float) -> None:
        if now <= self.last_refill:
            return
        dt = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + dt * self.refill_per_sec)
        self.last_refill = now

    def available(self, now: float) -> float:
        self.refill(now)
        return self.tokens

    def consume(self, units: float, now: float) -> bool:
        self.refill(now)
        if self.tokens >= units:
            self.tokens -= units
            return True
        return False

    def retry_after(self, units: float, now: float) -> int:
        self.refill(now)
        if self.tokens >= units:
            return 0
        deficit = units - self.tokens
        if self.refill_per_sec <= 0:
            return 3600  # effectively unbounded wait
        sec = int((deficit / self.refill_per_sec) + 0.999)
        return max(1, sec)


# --- Concurrency leases ---


@dataclass
class _Lease:
    lease_id: str
    expires_at: float


# --- Reservation handle ---


@dataclass
class _ReservationHandle:
    handle_id: str
    entity: str
    policy_id: str
    categories: dict[str, int]  # reserved units by category
    created_at: float
    expires_at: float
    state: str = "reserved"  # reserved|finalized


class MemoryResourceGovernor(ResourceGovernor):
    """
    In-memory Resource Governor implementation.

    Policy format example:
    {
      "chat.default": {
        "requests": {"rpm": 120, "burst": 2.0},
        "tokens": {"per_min": 60000, "burst": 1.5},
        "streams": {"max_concurrent": 2, "ttl_sec": 90},
        "scopes": ["global", "user"]
      }
    }
    """

    def __init__(
        self,
        *,
        policies: dict[str, dict[str, Any]] | None = None,
        policy_loader: Any | None = None,
        time_source: TimeSource = time.monotonic,
        backend_label: str = "memory",
        default_handle_ttl: int = 120,
    ) -> None:
        self._policies = policies or {}
        self._policy_loader = policy_loader
        self._time = time_source
        self._backend_label = backend_label
        self._default_handle_ttl = max(5, int(default_handle_ttl))
        self._op_ttl = max(60, int(default_handle_ttl))

        # Keyed by (policy_id, category, scope, entity_value)
        self._buckets: dict[tuple[str, str, str, str], _Bucket] = {}
        # Concurrency: (policy_id, category, scope, entity_value) → {lease_id: _Lease}
        self._leases: dict[tuple[str, str, str, str], dict[str, _Lease]] = {}
        # Handles and idempotency
        self._handles: dict[str, _ReservationHandle] = {}
        self._ops: dict[str, dict[str, Any]] = {}  # op_id → {type, handle_id}

        # Metrics
        ensure_rg_metrics_registered()

    # --- Policy helpers ---
    def _get_policy(self, policy_id: str) -> dict[str, Any]:
        if self._policy_loader is not None:
            try:
                pol = self._policy_loader.get_policy(policy_id)  # type: ignore[attr-defined]
                if pol:
                    return pol
            except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                logger.debug(f"Policy loader failed; falling back to static policies: {e}")
        return self._policies.get(policy_id, {})

    @staticmethod
    def _parse_entity(entity: str) -> tuple[str, str]:
        # entity of the form "scope:value" → (scope, value)
        if ":" in entity:
            s, v = entity.split(":", 1)
            return s.strip() or "entity", v.strip()
        return "entity", entity

    # --- Buckets ---
    def _bucket_key(self, policy_id: str, category: str, scope: str, entity_value: str) -> tuple[str, str, str, str]:
        return (policy_id, category, scope, entity_value)

    def _get_bucket(self, policy_id: str, category: str, scope: str, entity_value: str, *, capacity: float, refill_per_sec: float) -> _Bucket:
        k = self._bucket_key(policy_id, category, scope, entity_value)
        b = self._buckets.get(k)
        if b is None:
            b = _Bucket(capacity=float(capacity), refill_per_sec=float(refill_per_sec), tokens=float(capacity), last_refill=self._time())
            self._buckets[k] = b
        return b

    # --- Leases ---
    def _get_lease_map(self, policy_id: str, category: str, scope: str, entity_value: str) -> dict[str, _Lease]:
        k = self._bucket_key(policy_id, category, scope, entity_value)
        m = self._leases.get(k)
        if m is None:
            m = {}
            self._leases[k] = m
        return m

    def _purge_expired_leases(self, m: dict[str, _Lease], now: float) -> None:
        expired = [lid for lid, l in m.items() if l.expires_at <= now]
        for lid in expired:
            del m[lid]

    def _purge_expired_handles(self, now: float) -> None:
        expired = [hid for hid, h in self._handles.items() if h.expires_at <= now]
        for hid in expired:
            with contextlib.suppress(KeyError):
                del self._handles[hid]

    def _purge_expired_ops(self, now: float) -> None:
        ttl = self._op_ttl
        expired: list[str] = []
        for op_id, rec in self._ops.items():
            try:
                created_at = rec.get("created_at")
                if created_at is None:
                    continue
                if (now - float(created_at)) > float(ttl):
                    expired.append(op_id)
            except (OverflowError, TypeError, ValueError):
                continue
        for op_id in expired:
            with contextlib.suppress(KeyError):
                del self._ops[op_id]

    @staticmethod
    def _op_key(phase: str, op_id: str) -> str:
        return f"{phase}:{op_id}"

    # --- Core evaluation ---
    def _category_limits(self, policy: dict[str, Any], category: str) -> dict[str, Any]:
        return dict(policy.get(category, {}))

    def _scopes(self, policy: dict[str, Any]) -> list[str]:
        s = policy.get("scopes")
        if isinstance(s, list) and s:
            return [str(x) for x in s]
        return ["global", "entity"]

    def _compute_headroom_requests_tokens(
        self,
        *,
        policy_id: str,
        policy: dict[str, Any],
        category: str,
        entity_scope: str,
        entity_value: str,
        units: int,
        now: float,
    ) -> tuple[bool, int, dict[str, Any]]:
        cfg = self._category_limits(policy, category)
        # Interpret RPM / per_min and burst
        if category == "requests":
            rpm = float(cfg.get("rpm") or 0)
            burst = float(cfg.get("burst") or 1.0)
            refill_per_sec = rpm / 60.0
            capacity = rpm * max(1.0, burst)
            effective_limit = int(rpm)
        else:  # tokens
            per_min = float(cfg.get("per_min") or 0)
            burst = float(cfg.get("burst") or 1.0)
            refill_per_sec = per_min / 60.0
            capacity = per_min * max(1.0, burst)
            effective_limit = int(per_min)

        if refill_per_sec <= 0 or capacity <= 0:
            # Missing/zero config disables this category. For tokens, treat as
            # unbounded unless a durable daily_cap denies later.
            if category == "tokens":
                return True, 0, {
                    "limit": 0,
                    "burst": float(burst),
                    "remaining": 10**9,
                    "retry_after": None,
                    "unbounded": True,
                }
            # Requests without config are denied by default.
            return False, 60, {"limit": 0, "used": 0, "remaining": 0}

        # Evaluate strictest across scopes: global + entity scope
        scopes = self._scopes(policy)
        scope_keys: list[tuple[str, str]] = []
        if "global" in scopes:
            scope_keys.append(("global", "*"))
        if entity_scope in scopes or "entity" in scopes:
            scope_keys.append((entity_scope, entity_value))

        remainings = []
        retry_after_candidates = []
        for sc, ev in scope_keys:
            b = self._get_bucket(policy_id, category, sc, ev, capacity=capacity, refill_per_sec=refill_per_sec)
            avail = b.available(now)
            remaining = max(0, int(avail))
            remainings.append(remaining)
            if avail >= units:
                retry_after_candidates.append(0)
            else:
                retry_after_candidates.append(b.retry_after(units, now))

        effective_remaining = min(remainings) if remainings else 0
        allowed = effective_remaining >= units
        retry_after = max(retry_after_candidates) if retry_after_candidates else None
        details = {
            "limit": int(effective_limit),
            "burst": float(burst),
            "remaining": int(effective_remaining),
            "retry_after": int(retry_after or 0) if retry_after is not None else None,
        }
        return allowed, int(retry_after or 0) if retry_after is not None else 0, details

    def _acquire_concurrency(
        self,
        *,
        policy_id: str,
        policy: dict[str, Any],
        category: str,
        entity_scope: str,
        entity_value: str,
        units: int,
        now: float,
    ) -> tuple[bool, int, dict[str, Any]]:
        cfg = self._category_limits(policy, category)
        limit = int(cfg.get("max_concurrent") or 0)
        ttl_sec = int(cfg.get("ttl_sec") or 60)
        if limit <= 0:
            return False, 1, {"limit": 0, "remaining": 0}

        scopes = self._scopes(policy)
        scope_keys: list[tuple[str, str]] = []
        if "global" in scopes:
            scope_keys.append(("global", "*"))
        if entity_scope in scopes or "entity" in scopes:
            scope_keys.append((entity_scope, entity_value))

        remainings = []
        retry_after_candidates = []
        for sc, ev in scope_keys:
            m = self._get_lease_map(policy_id, category, sc, ev)
            self._purge_expired_leases(m, now)
            active = len(m)
            remaining = max(0, limit - active)
            remainings.append(remaining)
            retry_after_candidates.append(ttl_sec if remaining <= 0 else 0)

        effective_remaining = min(remainings) if remainings else 0
        allowed = effective_remaining >= units
        retry_after = max(retry_after_candidates) if retry_after_candidates else None
        details = {"limit": int(limit), "remaining": int(effective_remaining), "ttl_sec": ttl_sec, "retry_after": retry_after}
        return allowed, int(retry_after or 0) if retry_after is not None else 0, details

    async def _consume_daily_caps_for_reserve(
        self,
        *,
        req: RGRequest,
        policy_id: str,
        policy: dict[str, Any],
        entity_scope: str,
        entity_value: str,
        reserve_op_id: str,
        decision: RGDecision,
    ) -> RGDecision | None:
        try:
            categories = dict((decision.details or {}).get("categories") or {})
        except (AttributeError, TypeError, ValueError):
            categories = {}

        retry_after = int(decision.retry_after or 0)
        for category, cfg in req.categories.items():
            try:
                units = int((cfg or {}).get("units") or 0)
                daily_cap = int((policy.get(category) or {}).get("daily_cap") or 0)
            except (TypeError, ValueError):
                continue
            if units <= 0 or daily_cap <= 0:
                continue
            allowed, daily_ra, daily_details = await consume_daily_cap(
                entity_scope=entity_scope,
                entity_value=entity_value,
                category=category,
                daily_cap=daily_cap,
                units=units,
                op_id=f"{policy_id}:{reserve_op_id}:{category}",
            )
            retry_after = max(retry_after, int(daily_ra or 0))
            current = dict(categories.get(category) or {})
            current.update(daily_details or {})
            current["retry_after"] = max(int(current.get("retry_after") or 0), int(daily_ra or 0))
            if not allowed:
                current["allowed"] = False
                categories[category] = current
                return RGDecision(
                    allowed=False,
                    retry_after=(retry_after or None),
                    details={"policy_id": policy_id, "categories": categories},
                )
            current.setdefault("allowed", True)
            categories[category] = current
        return None

    # --- Public API ---
    async def check(self, req: RGRequest) -> RGDecision:
        now = self._time()
        policy_id = req.tags.get("policy_id") or "default"
        pol = self._get_policy(policy_id)
        entity_scope, entity_value = self._parse_entity(req.entity)
        backend = self._backend_label

        overall_allowed = True
        per_category: dict[str, Any] = {}
        retry_after_overall = 0

        for category, cfg in req.categories.items():
            units = int(cfg.get("units") or 0)
            if category in ("requests", "tokens"):
                allowed, retry_after, details = self._compute_headroom_requests_tokens(
                    policy_id=policy_id,
                    policy=pol,
                    category=category,
                    entity_scope=entity_scope,
                    entity_value=entity_value,
                    units=units,
                    now=now,
                )
            elif category in ("streams", "jobs"):
                allowed, retry_after, details = self._acquire_concurrency(
                    policy_id=policy_id,
                    policy=pol,
                    category=category,
                    entity_scope=entity_scope,
                    entity_value=entity_value,
                    units=units,
                    now=now,
                )
            else:
                # Minutes and other ledgers are not enforced in memory; allow by default here
                allowed, retry_after, details = True, 0, {"remaining": 10**9}

            # Optional durable daily caps (v1.1) backed by ResourceDailyLedger.
            try:
                cat_cfg = self._category_limits(pol, category)
                daily_cap = int(cat_cfg.get("daily_cap") or 0)
            except (AttributeError, TypeError, ValueError):
                daily_cap = 0
            if daily_cap > 0:
                daily_allowed, daily_ra, daily_details = await check_daily_cap(
                    entity_scope=entity_scope,
                    entity_value=entity_value,
                    category=category,
                    daily_cap=daily_cap,
                    units=units,
                )
                if not daily_allowed:
                    allowed = False
                retry_after = max(int(retry_after or 0), int(daily_ra or 0))
                with contextlib.suppress(AttributeError, TypeError, ValueError):
                    details.update(daily_details or {})
                # Provide limit/remaining for daily-only categories
                try:
                    if not int(details.get("limit") or 0):
                        details["limit"] = int(daily_cap)
                except (TypeError, ValueError):
                    details["limit"] = int(daily_cap)
                try:
                    if details.get("remaining") is None:
                        details["remaining"] = int((daily_details or {}).get("daily_remaining") or 0)
                except (AttributeError, TypeError, ValueError):
                    pass
                with contextlib.suppress(TypeError, ValueError):
                    details["retry_after"] = int(retry_after or 0)

            per_category[category] = {"allowed": bool(allowed), **details}
            overall_allowed = overall_allowed and allowed
            retry_after_overall = max(retry_after_overall, int(details.get("retry_after") or 0))

            # Metrics per category (decision)
            if get_metrics_registry:
                get_metrics_registry().increment(
                    "rg_decisions_total",
                    1,
                    _labels(category=category, scope=entity_scope, backend=backend, result=("allow" if allowed else "deny"), policy_id=policy_id),
                )
                if not allowed:
                    get_metrics_registry().increment(
                        "rg_denials_total",
                        1,
                        _labels(category=category, scope=entity_scope, reason="insufficient_capacity", policy_id=policy_id),
                    )
                # Optional by-entity metrics (hashed)
                try:
                    if rg_metrics_entity_label_enabled():
                        ent_h = hash_entity(req.entity)
                        get_metrics_registry().increment(
                            "rg_decisions_by_entity_total",
                            1,
                            {"category": category, "scope": entity_scope, "backend": backend, "result": ("allow" if allowed else "deny"), "policy_id": policy_id, "entity": ent_h},
                        )
                        if not allowed:
                            get_metrics_registry().increment(
                                "rg_denials_by_entity_total",
                                1,
                                {"category": category, "scope": entity_scope, "reason": "insufficient_capacity", "policy_id": policy_id, "entity": ent_h},
                            )
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass

        return RGDecision(allowed=overall_allowed, retry_after=(retry_after_overall or None), details={"policy_id": policy_id, "categories": per_category})

    async def reserve(self, req: RGRequest, op_id: str | None = None) -> tuple[RGDecision, str | None]:
        now_purge = self._time()
        self._purge_expired_handles(now_purge)
        self._purge_expired_ops(now_purge)
        # Idempotency: return previous outcome for same op_id
        reserve_key = self._op_key("reserve", op_id) if op_id else None
        if reserve_key and reserve_key in self._ops:
            rec = self._ops[reserve_key]
            hid = rec.get("handle_id")
            return rec.get("decision"), hid  # type: ignore[return-value]

        dec = await self.check(req)
        if not dec.allowed:
            if reserve_key:
                self._ops[reserve_key] = {"type": "reserve", "decision": dec, "handle_id": None, "created_at": now_purge}
            return dec, None

        # Consume from buckets / acquire leases
        now = self._time()
        policy_id = dec.details.get("policy_id") or req.tags.get("policy_id") or "default"
        pol = self._get_policy(policy_id)
        entity_scope, entity_value = self._parse_entity(req.entity)
        handle_id = str(uuid.uuid4())
        ttl = self._default_handle_ttl
        h = _ReservationHandle(handle_id=handle_id, entity=req.entity, policy_id=policy_id, categories={}, created_at=now, expires_at=now + ttl)

        daily_denial = await self._consume_daily_caps_for_reserve(
            req=req,
            policy_id=policy_id,
            policy=pol,
            entity_scope=entity_scope,
            entity_value=entity_value,
            reserve_op_id=op_id or handle_id,
            decision=dec,
        )
        if daily_denial is not None:
            if reserve_key:
                self._ops[reserve_key] = {"type": "reserve", "decision": daily_denial, "handle_id": None, "created_at": now}
            return daily_denial, None

        for category, cfg in req.categories.items():
            units = int(cfg.get("units") or 0)
            h.categories[category] = units
            if category in ("requests", "tokens"):
                # consume from both global and entity buckets when applicable
                cl_allowed, _ra, _det = self._compute_headroom_requests_tokens(
                    policy_id=policy_id,
                    policy=pol,
                    category=category,
                    entity_scope=entity_scope,
                    entity_value=entity_value,
                    units=units,
                    now=now,
                )
                if not cl_allowed:
                    logger.warning("reserve inconsistency: allowed in check but deny on consume; ignoring for memory backend")
                cfg = self._category_limits(pol, category)
                if category == "requests":
                    rpm = float(cfg.get("rpm") or 0)
                    burst = float(cfg.get("burst") or 1.0)
                    refill_per_sec = rpm / 60.0
                    capacity = rpm * max(1.0, burst)
                else:
                    per_min = float(cfg.get("per_min") or 0)
                    burst = float(cfg.get("burst") or 1.0)
                    refill_per_sec = per_min / 60.0
                    capacity = per_min * max(1.0, burst)
                # global
                if "global" in self._scopes(pol):
                    b = self._get_bucket(policy_id, category, "global", "*", capacity=capacity, refill_per_sec=refill_per_sec)
                    _ = b.consume(units, now)
                # entity
                b = self._get_bucket(policy_id, category, entity_scope, entity_value, capacity=capacity, refill_per_sec=refill_per_sec)
                _ = b.consume(units, now)

            elif category in ("streams", "jobs"):
                cfgc = self._category_limits(pol, category)
                limit = int(cfgc.get("max_concurrent") or 0)
                ttl_sec = int(cfgc.get("ttl_sec") or 60)
                # Acquire for global and entity scopes (when configured)
                scopes = self._scopes(pol)
                for sc, ev in (("global", "*"), (entity_scope, entity_value)):
                    if sc not in scopes and not (sc == entity_scope and "entity" in scopes):
                        continue
                    m = self._get_lease_map(policy_id, category, sc, ev)
                    self._purge_expired_leases(m, now)
                    if units <= 0:
                        continue
                    if (len(m) + units) > limit:
                        logger.debug("lease contention on reserve: scope={} ev={}", sc, ev)
                        continue
                    for i in range(units):
                        lid = f"{handle_id}:{sc}:{ev}:{i}"
                        m[lid] = _Lease(lease_id=lid, expires_at=now + ttl_sec)
                    # Gauge update (best-effort)
                    if get_metrics_registry:
                        get_metrics_registry().set_gauge(
                            "rg_concurrency_active",
                            float(len(m)),
                            {"category": category, "scope": sc, "policy_id": policy_id},
                        )
            else:
                # minutes / others: no-op in memory (allow)
                pass

        self._handles[handle_id] = h
        if reserve_key:
            self._ops[reserve_key] = {"type": "reserve", "decision": dec, "handle_id": handle_id, "created_at": now}
        return dec, handle_id

    async def commit(self, handle_id: str, actuals: dict[str, int] | None = None, op_id: str | None = None) -> None:
        now_purge = self._time()
        self._purge_expired_handles(now_purge)
        self._purge_expired_ops(now_purge)
        # Idempotent per op_id
        commit_key = self._op_key("commit", op_id) if op_id else None
        if commit_key and commit_key in self._ops:
            rec = self._ops[commit_key]
            if rec.get("type") == "commit" and rec.get("handle_id") == handle_id:
                return
        h = self._handles.get(handle_id)
        if not h:
            return
        now = self._time()
        entity_scope, entity_value = self._parse_entity(h.entity)
        pol = self._get_policy(h.policy_id)

        actuals = actuals or {}
        for category, reserved in list(h.categories.items()):
            actual = int(actuals.get(category, reserved))
            actual = max(0, min(actual, reserved))
            refund_units = reserved - actual
            if refund_units > 0:
                # return difference to buckets
                if category in ("requests", "tokens"):
                    cfg = self._category_limits(pol, category)
                    if category == "requests":
                        rpm = float(cfg.get("rpm") or 0)
                        burst = float(cfg.get("burst") or 1.0)
                        refill_per_sec = rpm / 60.0
                        capacity = rpm * max(1.0, burst)
                    else:
                        per_min = float(cfg.get("per_min") or 0)
                        burst = float(cfg.get("burst") or 1.0)
                        refill_per_sec = per_min / 60.0
                        capacity = per_min * max(1.0, burst)
                    # global
                    if "global" in self._scopes(pol):
                        b = self._get_bucket(h.policy_id, category, "global", "*", capacity=capacity, refill_per_sec=refill_per_sec)
                        b.refill(now)
                        b.tokens = min(b.capacity, b.tokens + refund_units)
                    # entity
                    b = self._get_bucket(h.policy_id, category, entity_scope, entity_value, capacity=capacity, refill_per_sec=refill_per_sec)
                    b.refill(now)
                    b.tokens = min(b.capacity, b.tokens + refund_units)
                    if get_metrics_registry:
                        get_metrics_registry().increment(
                            "rg_refunds_total",
                            1,
                            _labels(category=category, scope=entity_scope, reason="commit_diff", policy_id=h.policy_id),
                        )
                        try:
                            if rg_metrics_entity_label_enabled():
                                ent_h = hash_entity(h.entity)
                                get_metrics_registry().increment(
                                    "rg_refunds_by_entity_total",
                                    1,
                                    {"category": category, "scope": entity_scope, "reason": "commit_diff", "policy_id": h.policy_id, "entity": ent_h},
                                )
                        except (AttributeError, RuntimeError, TypeError, ValueError):
                            pass
                # concurrency: nothing to refund here

        # Release any concurrency leases
        for category in list(h.categories.keys()):
            if category in ("streams", "jobs"):
                scopes = self._scopes(pol)
                for sc, ev in (("global", "*"), (entity_scope, entity_value)):
                    if sc not in scopes and not (sc == entity_scope and "entity" in scopes):
                        continue
                    m = self._get_lease_map(h.policy_id, category, sc, ev)
                    self._purge_expired_leases(m, now)
                    # Remove leases for this handle
                    to_del = [lid for lid in list(m.keys()) if lid.startswith(f"{handle_id}:")]
                    for lid in to_del:
                        del m[lid]
                    if get_metrics_registry:
                        get_metrics_registry().set_gauge(
                            "rg_concurrency_active",
                            float(len(m)),
                            {"category": category, "scope": sc, "policy_id": h.policy_id},
                        )

        h.state = "finalized"
        self._handles.pop(handle_id, None)
        if commit_key:
            self._ops[commit_key] = {"type": "commit", "handle_id": handle_id, "created_at": now}

    async def refund(self, handle_id: str, deltas: dict[str, int] | None = None, op_id: str | None = None) -> None:
        now_purge = self._time()
        self._purge_expired_handles(now_purge)
        self._purge_expired_ops(now_purge)
        # Idempotent per op_id
        refund_key = self._op_key("refund", op_id) if op_id else None
        if refund_key and refund_key in self._ops:
            rec = self._ops[refund_key]
            if rec.get("type") == "refund" and rec.get("handle_id") == handle_id:
                return
        h = self._handles.get(handle_id)
        if not h:
            return
        now = self._time()
        entity_scope, entity_value = self._parse_entity(h.entity)
        pol = self._get_policy(h.policy_id)
        deltas = deltas or {}

        for category, reserved in list(h.categories.items()):
            refund_units = int(deltas.get(category, reserved))
            refund_units = max(0, min(refund_units, reserved))
            if refund_units <= 0:
                continue
            if category in ("requests", "tokens"):
                cfg = self._category_limits(pol, category)
                if category == "requests":
                    rpm = float(cfg.get("rpm") or 0)
                    burst = float(cfg.get("burst") or 1.0)
                    refill_per_sec = rpm / 60.0
                    capacity = rpm * max(1.0, burst)
                else:
                    per_min = float(cfg.get("per_min") or 0)
                    burst = float(cfg.get("burst") or 1.0)
                    refill_per_sec = per_min / 60.0
                    capacity = per_min * max(1.0, burst)
                # global
                if "global" in self._scopes(pol):
                    b = self._get_bucket(h.policy_id, category, "global", "*", capacity=capacity, refill_per_sec=refill_per_sec)
                    b.refill(now)
                    b.tokens = min(b.capacity, b.tokens + refund_units)
                # entity
                b = self._get_bucket(h.policy_id, category, entity_scope, entity_value, capacity=capacity, refill_per_sec=refill_per_sec)
                b.refill(now)
                b.tokens = min(b.capacity, b.tokens + refund_units)
                if get_metrics_registry:
                    get_metrics_registry().increment(
                        "rg_refunds_total",
                        1,
                        _labels(category=category, scope=entity_scope, reason="explicit_refund", policy_id=h.policy_id),
                    )
                    try:
                        if rg_metrics_entity_label_enabled():
                            ent_h = hash_entity(h.entity)
                            get_metrics_registry().increment(
                                "rg_refunds_by_entity_total",
                                1,
                                {"category": category, "scope": entity_scope, "reason": "explicit_refund", "policy_id": h.policy_id, "entity": ent_h},
                            )
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass

        if refund_key:
            self._ops[refund_key] = {"type": "refund", "handle_id": handle_id, "created_at": now}

    async def renew(self, handle_id: str, ttl_s: int) -> None:
        h = self._handles.get(handle_id)
        if not h:
            return
        now = self._time()
        h.expires_at = now + max(1, int(ttl_s))
        entity_scope, entity_value = self._parse_entity(h.entity)
        pol = self._get_policy(h.policy_id)
        for category in list(h.categories.keys()):
            if category in ("streams", "jobs"):
                scopes = self._scopes(pol)
                for sc, ev in (("global", "*"), (entity_scope, entity_value)):
                    if sc not in scopes and not (sc == entity_scope and "entity" in scopes):
                        continue
                    m = self._get_lease_map(h.policy_id, category, sc, ev)
                    # Renew leases for this handle
                    for lid, lease in list(m.items()):
                        if lid.startswith(f"{handle_id}:"):
                            lease.expires_at = now + max(1, int(ttl_s))

    async def release(self, handle_id: str) -> None:
        # Alias to commit with zero actuals for all categories
        await self.commit(handle_id, actuals={})

    async def peek(self, entity: str, categories: list[str]) -> dict[str, Any]:
        now = self._time()
        result: dict[str, Any] = {}
        # Peeks without policy context assume a synthetic policy_id 'default'
        policy_id = "default"
        entity_scope, entity_value = self._parse_entity(entity)
        for category in categories:
            # We report remaining based on current bucket tokens if present
            remainings = []
            for sc, ev in (("global", "*"), (entity_scope, entity_value)):
                b = self._buckets.get(self._bucket_key(policy_id, category, sc, ev))
                if b:
                    remainings.append(int(b.available(now)))
            result[category] = {"remaining": (min(remainings) if remainings else None), "reset": 0}
        return result

    async def peek_with_policy(self, entity: str, categories: list[str], policy_id: str) -> dict[str, Any]:
        now = self._time()
        entity_scope, entity_value = self._parse_entity(entity)
        pol = self._get_policy(policy_id)
        out: dict[str, Any] = {}
        for category in categories:
            if category in ("requests", "tokens"):
                cfg = self._category_limits(pol, category)
                if category == "requests":
                    rpm = float(cfg.get("rpm") or 0)
                    burst = float(cfg.get("burst") or 1.0)
                    refill_per_sec = rpm / 60.0
                    capacity = rpm * max(1.0, burst)
                else:
                    per_min = float(cfg.get("per_min") or 0)
                    burst = float(cfg.get("burst") or 1.0)
                    refill_per_sec = per_min / 60.0
                    capacity = per_min * max(1.0, burst)
                scopes = self._scopes(pol)
                remainings = []
                for sc, ev in (("global", "*"), (entity_scope, entity_value)):
                    if sc not in scopes and not (sc == entity_scope and "entity" in scopes):
                        continue
                    b = self._get_bucket(policy_id, category, sc, ev, capacity=capacity, refill_per_sec=refill_per_sec)
                    remainings.append(int(b.available(now)))
                out[category] = {"remaining": (min(remainings) if remainings else None), "reset": 0}
            else:
                out[category] = {"remaining": None, "reset": 0}
        return out

    async def query(self, entity: str, category: str) -> dict[str, Any]:
        now = self._time()
        policy_id = "default"
        entity_scope, entity_value = self._parse_entity(entity)
        b_global = self._buckets.get(self._bucket_key(policy_id, category, "global", "*"))
        b_entity = self._buckets.get(self._bucket_key(policy_id, category, entity_scope, entity_value))
        return {
            "global": {"available": int(b_global.available(now))} if b_global else None,
            "entity": {"available": int(b_entity.available(now))} if b_entity else None,
        }

    async def reset(self, entity: str, category: str | None = None) -> None:
        entity_scope, entity_value = self._parse_entity(entity)
        keys = list(self._buckets.keys())
        for (pol, cat, sc, ev) in keys:
            if category and cat != category:
                continue
            if sc == entity_scope and ev == entity_value:
                with contextlib.suppress(KeyError):
                    del self._buckets[(pol, cat, sc, ev)]

    async def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self._backend_label,
            "real_redis": False,
            "tokens_lua_loaded": False,
            "multi_lua_loaded": False,
            "last_used_tokens_lua": False,
            "last_used_multi_lua": False,
        }
