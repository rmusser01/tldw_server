"""Policy evaluation shared by the memory and Redis Resource Governor backends.

Both backends decide the same three things for every reservation:
- which policy config applies, with a safe fallback when the ID is unknown;
- which buckets the request charges;
- how many token units a single request may reserve.

Keeping those decisions here is what keeps the two backends in agreement. The
safety-net rule is that no configuration can produce a permanent 429.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from loguru import logger

DEFAULT_POLICY_ID = "default"

# Used only when the loaded policies lack "default", for example on a DB policy
# store seeded before the safety-net defaults. Mirrors the shipped YAML.
BUILTIN_DEFAULT_POLICY: dict[str, Any] = {
    "requests": {"rpm": 600, "burst": 2.0},
    "scopes": ["user", "api_key", "ip"],
    "fail_mode": "fallback_memory",
}

_warned_unknown: set[str] = set()
_warned_lookup_errors: set[tuple[str, str]] = set()


def log_lookup_failure(policy_id: str, exc: BaseException) -> None:
    """Log a policy store lookup failure at ERROR, once per (policy_id, exception type)."""
    key = (policy_id, type(exc).__name__)
    if key not in _warned_lookup_errors:
        _warned_lookup_errors.add(key)
        logger.error("Resource Governor policy store lookup for {!r} failed: {!r}; treating it as undefined", policy_id, exc)


def _lookup(get_policy: Callable[[str], Mapping[str, Any] | None], policy_id: str) -> dict[str, Any]:
    try:
        pol = get_policy(policy_id)
        return dict(pol) if pol else {}
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        log_lookup_failure(policy_id, exc)
        return {}


def _has_requests(policy: Mapping[str, Any]) -> bool:
    try:
        return float((policy.get("requests") or {}).get("rpm") or 0) > 0
    except (AttributeError, TypeError, ValueError):
        return False


def effective_policy(get_policy: Callable[[str], Mapping[str, Any] | None], policy_id: str) -> dict[str, Any]:
    """Return the config to enforce for ``policy_id``.

    If the ID is unknown, use ``default``; if that is missing too, use the built-in
    default. A policy without a usable ``requests`` block inherits ``default``'s,
    because denying every request is never the intent of an omission.
    """
    policy = _lookup(get_policy, policy_id)
    if not policy:
        if policy_id not in _warned_unknown:
            _warned_unknown.add(policy_id)
            logger.error("Resource Governor policy {!r} is not defined; using {!r}", policy_id, DEFAULT_POLICY_ID)
        policy = _lookup(get_policy, DEFAULT_POLICY_ID) or dict(BUILTIN_DEFAULT_POLICY)
    if not _has_requests(policy):
        fallback = _lookup(get_policy, DEFAULT_POLICY_ID)
        policy["requests"] = dict((fallback if _has_requests(fallback) else BUILTIN_DEFAULT_POLICY)["requests"])
    # A bucket that can't hold one request denies forever: raise burst so capacity is 1.
    requests = policy["requests"]
    rpm = float(requests["rpm"])
    if rpm * max(1.0, float(requests.get("burst") or 1.0)) < 1:
        policy["requests"] = {**requests, "burst": 1 / rpm}
    return policy


def scope_pairs(policy: Mapping[str, Any], entity_scope: str, entity_value: str) -> list[tuple[str, str]]:
    """Return the (scope, value) buckets a request charges.

    A policy's ``scopes`` decide whether a server-wide bucket exists. They never
    remove the caller's own bucket: a request whose entity kind the policy does not
    list is charged a per-entity bucket instead of being denied (the ADR-044 bug
    class).
    """
    raw = policy.get("scopes")
    scopes = [str(s) for s in raw] if isinstance(raw, list) and raw else ["global", "entity"]
    pairs: list[tuple[str, str]] = [("global", "*")] if "global" in scopes else []
    pairs.append((entity_scope, entity_value))
    return pairs


def clamp_token_units(
    policy: Mapping[str, Any],
    categories: Mapping[str, Mapping[str, int]],
    *,
    capacity_includes_burst: bool,
) -> dict[str, dict[str, int]]:
    """Cap a token reservation at the bucket's capacity, so it can never be denied forever.

    The memory backend's capacity is ``per_min * burst``. The Redis backend's
    sliding window holds ``per_min``.
    """
    out = {k: dict(v) for k, v in categories.items()}
    tokens = out.get("tokens")
    if not tokens:
        return out
    try:
        cfg = policy.get("tokens") or {}
        per_min = float(cfg.get("per_min") or 0)
        burst = max(1.0, float(cfg.get("burst") or 1.0)) if capacity_includes_burst else 1.0
    except (AttributeError, TypeError, ValueError):
        return out
    capacity = int(per_min * burst)
    if capacity > 0 and int(tokens.get("units") or 0) > capacity:
        tokens["units"] = capacity
    return out
