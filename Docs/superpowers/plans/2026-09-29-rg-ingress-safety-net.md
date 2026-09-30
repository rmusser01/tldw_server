# Resource Governor Ingress Safety Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the ingress Resource Governor from rate-limiting self-hosters in normal use, make every `/api/` route governed by exactly one resolvable policy (including `route_map.by_tag`), and make governance switch off in one place.

**Architecture:** One shared module (`policy_eval.py`) decides policy config, bucket scopes and token clamping for both governor backends. One resolver (`policy_resolver.py`) maps a request to a policy: `by_path`, then tags from a served-route index, then `default`. The middleware, both audits and a CI lint all use it. The middleware charges the authenticated principal rather than the proxy IP. The work ships as three PRs: relief first, coverage second, the switch and docs third.

**Tech Stack:** Python 3.12, FastAPI 0.141.1 / Starlette 1.x, pytest (+ pytest-asyncio), loguru, PyYAML, the Backlog CLI (`backlog`), `gh`.

**Spec:** `Docs/Design/2026-09-29-rg-ingress-safety-net-design.md`

## Global Constraints

- All PRs target `dev`. Never `main`.
- Never use `git stash` (the stash is shared across worktrees). Never pass `--no-verify`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- PR bodies end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)` and include the owner's waiver line: `> **Waived by the repository owner (@rmusser01) on 2026-09-22**, by explicit instruction in the session that produced this PR.`
- The Backlog CLI is the ledger. Never hand-edit task files, and use `--append-notes` rather than `--notes`.
- Loguru only (`from loguru import logger`). No new dependencies.
- New tests carry `pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]` (both are registered markers).
- The default policy: `requests: {rpm: 600, burst: 2.0}`, `scopes: [user, api_key, ip]`.
- `global` is kept only on `authnz.forgot_password`, `authnz.resend_verification` and `authnz.magic_link.request`.
- Run Python with the venv interpreter: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest …`. There is no bare `python` on PATH.

## Review Focus

These are inputs the spec implies but no task's main tests exercise. Each one has a pinning test in the task named at the end of its line.

1. **A cookie that isn't a session cookie** (analytics, CSRF, theme), with no auth headers, must be charged to the IP bucket without a 401 response from the middleware. Task 9, `test_non_session_cookie_charges_ip_and_reaches_route`.
2. **`HEAD` or `OPTIONS` to a tag-governed path** must not raise and must resolve to a policy. Tag matching filters by method and falls back to `default`. Task 7, `test_head_request_falls_back_to_default`.
3. **A `by_path` pattern with a mid-path `*`** (`/api/v1/media/*/reprocess`) must keep its current semantics after the glob helper moves modules. Task 7, `test_mid_path_glob_matches_one_or_more_segments`.
4. **The policy file changes while requests are in flight.** The resolver cache must never serve a policy from the old route map after a reload. Task 7, `test_resolver_cache_is_rebuilt_when_snapshot_changes`.
5. **Redis backend with a stub client after a policy change.** A raised limit applies without a restart. Task 3, `test_raised_limit_applies_without_restart[redis]`.

---

## PR A — Relief: safety-net defaults and the permanent-429 fixes

Branch: `fix/rg-safety-net-relief` from `origin/dev`.

### Task 0: Ledger

**Files:** none (Backlog CLI only).

- [ ] **Step 1: File the parent task**

```bash
backlog task create "RG ingress safety net (spec 1 of 2)" --priority high --labels resource-governance,backend \
  -d "Implements Docs/Design/2026-09-29-rg-ingress-safety-net-design.md in three PRs (relief, coverage, switch and docs). Plan: Docs/superpowers/plans/2026-09-29-rg-ingress-safety-net.md. TASK-13395 closes with PR B." \
  --ac "PR A merged: safety-net defaults and permanent-429 fixes in both backends" \
  --ac "PR B merged: resolver, route index, principal identity, audits, route-map lints, WebUI replay; TASK-13395 closed" \
  --ac "PR C merged: single RG switch, config hygiene, ADR-056, docs"
```

Record the new task ID. Every later ledger step refers to it as `<PARENT>`.

- [ ] **Step 2: Create the branch**

```bash
git fetch -q origin dev && git checkout -q -b fix/rg-safety-net-relief origin/dev
```

### Task 1: Shared policy evaluation module

**Files:**
- Create: `tldw_Server_API/app/core/Resource_Governance/policy_eval.py`
- Test: `tldw_Server_API/tests/Resource_Governance/test_policy_eval.py`

**Interfaces:**
- Produces:
  - `DEFAULT_POLICY_ID: str = "default"`
  - `BUILTIN_DEFAULT_POLICY: dict[str, Any]`
  - `effective_policy(get_policy: Callable[[str], Mapping | None], policy_id: str) -> dict[str, Any]`
  - `scope_pairs(policy: Mapping, entity_scope: str, entity_value: str) -> list[tuple[str, str]]`
  - `clamp_token_units(policy: Mapping, categories: Mapping[str, Mapping[str, int]], *, capacity_includes_burst: bool) -> dict[str, dict[str, int]]`

- [ ] **Step 1: Write the failing tests**

```python
"""Shared policy evaluation: the decisions both governor backends must agree on."""

import pytest

from tldw_Server_API.app.core.Resource_Governance import policy_eval
from tldw_Server_API.app.core.Resource_Governance.policy_eval import (
    BUILTIN_DEFAULT_POLICY,
    clamp_token_units,
    effective_policy,
    scope_pairs,
)

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]


def _getter(policies):
    return lambda pid: policies.get(pid)


def test_known_policy_is_returned_unchanged():
    pol = {"requests": {"rpm": 5}, "scopes": ["user"]}
    assert effective_policy(_getter({"p": pol}), "p") == pol


def test_unknown_policy_falls_back_to_default():
    default = {"requests": {"rpm": 7}, "scopes": ["user"]}
    assert effective_policy(_getter({"default": default}), "typo.policy") == default


def test_missing_default_falls_back_to_builtin():
    assert effective_policy(_getter({}), "anything") == BUILTIN_DEFAULT_POLICY


def test_policy_without_requests_inherits_default_requests():
    default = {"requests": {"rpm": 9, "burst": 2.0}}
    pol = {"tokens": {"per_min": 100}, "scopes": ["user"]}
    out = effective_policy(_getter({"p": pol, "default": default}), "p")
    assert out["requests"] == {"rpm": 9, "burst": 2.0}
    assert out["tokens"] == {"per_min": 100}


def test_getter_errors_are_treated_as_unknown():
    def boom(_pid):
        raise RuntimeError("store down")

    assert effective_policy(boom, "p") == BUILTIN_DEFAULT_POLICY


def test_unknown_policy_is_logged_once(monkeypatch):
    from loguru import logger

    monkeypatch.setattr(policy_eval, "_warned_unknown", set())
    seen = []
    sink = logger.add(lambda m: seen.append(str(m)), level="ERROR")
    try:
        getter = _getter({"default": {"requests": {"rpm": 1}}})
        effective_policy(getter, "typo")
        effective_policy(getter, "typo")
    finally:
        logger.remove(sink)
    assert len([m for m in seen if "'typo'" in m]) == 1


def test_scope_pairs_always_include_the_entity_bucket():
    assert scope_pairs({"scopes": ["user", "api_key"]}, "ip", "1.2.3.4") == [("ip", "1.2.3.4")]


def test_scope_pairs_add_global_only_when_listed():
    assert scope_pairs({"scopes": ["global", "user"]}, "user", "1") == [("global", "*"), ("user", "1")]
    assert scope_pairs({"scopes": ["user"]}, "user", "1") == [("user", "1")]


def test_scope_pairs_default_scopes_are_global_plus_entity():
    assert scope_pairs({}, "user", "1") == [("global", "*"), ("user", "1")]


def test_clamp_caps_oversized_token_reservation_at_capacity():
    pol = {"tokens": {"per_min": 100, "burst": 1.5}}
    cats = {"tokens": {"units": 1000}, "requests": {"units": 1}}
    assert clamp_token_units(pol, cats, capacity_includes_burst=True)["tokens"]["units"] == 150
    assert clamp_token_units(pol, cats, capacity_includes_burst=False)["tokens"]["units"] == 100


def test_clamp_leaves_fitting_and_unbounded_reservations_alone():
    cats = {"tokens": {"units": 50}}
    assert clamp_token_units({"tokens": {"per_min": 100}}, cats, capacity_includes_burst=True) == cats
    assert clamp_token_units({}, {"tokens": {"units": 10**9}}, capacity_includes_burst=True) == {"tokens": {"units": 10**9}}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_policy_eval.py`
Expected: FAIL at collection with `ModuleNotFoundError: ... policy_eval`.

- [ ] **Step 3: Implement**

```python
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


def _lookup(get_policy: Callable[[str], Mapping[str, Any] | None], policy_id: str) -> dict[str, Any]:
    try:
        pol = get_policy(policy_id)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return {}
    return dict(pol) if pol else {}


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_policy_eval.py`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/policy_eval.py tldw_Server_API/tests/Resource_Governance/test_policy_eval.py
git commit -m "feat(rg): shared policy evaluation with safe fallbacks for both governor backends

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 2: Memory governor adopts the shared rules, resizes buckets on reload, and evicts idle buckets

**Files:**
- Modify: `tldw_Server_API/app/core/Resource_Governance/governor.py`
  - `_Bucket` (lines ~94-128)
  - `MemoryResourceGovernor.__init__` (~171-195)
  - `_get_policy` (~198-206)
  - `_get_bucket` (~222-228)
  - `_compute_headroom_requests_tokens` (~282-342)
  - `_acquire_concurrency` (~351-388)
  - `check` (~441)
  - `reserve` (~548-656)
  - the `commit` lease release (~722-735)
  - `renew` (~812-829)
  - `peek_with_policy` (~850-878)
- Test: `tldw_Server_API/tests/Resource_Governance/test_governor_safety_net.py`

**Interfaces:**
- Consumes: `effective_policy`, `scope_pairs` and `clamp_token_units` from Task 1.
- Produces:
  - `_Bucket.last_used: float`
  - `MemoryResourceGovernor._maybe_evict_idle(now: float) -> None`
  - module constants `_EVICT_IDLE_SEC = 600.0`, `_EVICT_INTERVAL_SEC = 60.0`, `_EVICT_BATCH = 5000`

- [ ] **Step 1: Write the failing tests** (memory backend now; Task 3 adds Redis to the same parametrization)

```python
"""Safety-net governor behaviour: no configuration can produce a permanent 429.

These tests run against every backend listed in BACKENDS.
"""

import itertools

import pytest

from tldw_Server_API.app.core.Resource_Governance import MemoryResourceGovernor, RGRequest

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit, pytest.mark.asyncio]

BACKENDS = ["memory"]  # Task 3 appends "redis"
_ns = itertools.count()


class FakeTime:
    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class _Loader:
    def __init__(self, policies):
        self.policies = policies

    def get_policy(self, pid):
        return self.policies.get(pid)


def _gov(backend, policies, clock):
    loader = _Loader(policies)
    if backend == "memory":
        return MemoryResourceGovernor(policy_loader=loader, time_source=clock), loader
    from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor

    return RedisResourceGovernor(policy_loader=loader, time_source=clock, ns=f"rg_t_safety_{next(_ns)}"), loader


def _req(entity, policy_id, **cats):
    return RGRequest(entity=entity, categories=cats or {"requests": {"units": 1}}, tags={"policy_id": policy_id})


async def _admits(gov, req, n):
    out = []
    for i in range(n):
        dec, _h = await gov.reserve(req, op_id=f"{req.entity}-{req.tags['policy_id']}-{id(req)}-{i}")
        out.append(dec.allowed)
    return out


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


async def test_unknown_policy_uses_default_limits(backend):
    clock = FakeTime()
    gov, _ = _gov(backend, {"default": {"requests": {"rpm": 2, "burst": 1.0}, "scopes": ["user"]}}, clock)
    assert await _admits(gov, _req("user:1", "no.such.policy"), 3) == [True, True, False]
    clock.advance(61)
    assert await _admits(gov, _req("user:1", "no.such.policy"), 1) == [True]


async def test_missing_default_uses_builtin(backend):
    gov, _ = _gov(backend, {}, FakeTime())
    assert all(await _admits(gov, _req("user:1", "anything"), 50))


async def test_scope_mismatch_charges_per_entity_bucket(backend):
    gov, _ = _gov(backend, {"p": {"requests": {"rpm": 2, "burst": 1.0}, "scopes": ["user", "api_key"]}}, FakeTime())
    assert await _admits(gov, _req("ip:1.2.3.4", "p"), 3) == [True, True, False]


async def test_policy_without_requests_inherits_default(backend):
    policies = {
        "default": {"requests": {"rpm": 2, "burst": 1.0}, "scopes": ["user"]},
        "tok": {"tokens": {"per_min": 1000}, "scopes": ["user"]},
    }
    gov, _ = _gov(backend, policies, FakeTime())
    assert await _admits(gov, _req("user:1", "tok"), 3) == [True, True, False]


async def test_missing_concurrency_config_is_unbounded(backend):
    gov, _ = _gov(backend, {"p": {"requests": {"rpm": 100}, "scopes": ["user"]}}, FakeTime())
    req = _req("user:1", "p", jobs={"units": 1})
    assert all(await _admits(gov, req, 5))


async def test_oversized_token_reservation_admitted_on_full_bucket(backend):
    clock = FakeTime()
    gov, _ = _gov(backend, {"p": {"requests": {"rpm": 1000}, "tokens": {"per_min": 100, "burst": 1.0}, "scopes": ["user"]}}, clock)
    big = _req("user:1", "p", tokens={"units": 10_000})
    assert await _admits(gov, big, 1) == [True]
    assert await _admits(gov, _req("user:1", "p", tokens={"units": 1}), 1) == [False]
    clock.advance(61)
    assert await _admits(gov, big, 1) == [True]


async def test_raised_limit_applies_without_restart(backend):
    clock = FakeTime()
    gov, loader = _gov(backend, {"p": {"requests": {"rpm": 2, "burst": 1.0}, "scopes": ["user"]}}, clock)
    assert await _admits(gov, _req("user:1", "p"), 3) == [True, True, False]
    loader.policies["p"] = {"requests": {"rpm": 600, "burst": 1.0}, "scopes": ["user"]}
    clock.advance(1)  # refills 10 units at the new rate
    assert all(await _admits(gov, _req("user:1", "p"), 5))


async def test_memory_evicts_full_idle_buckets_only():
    clock = FakeTime()
    policies = {
        "fast": {"requests": {"rpm": 600, "burst": 1.0}, "scopes": ["user"]},
        "slow": {"requests": {"rpm": 0.01, "burst": 100.0}, "scopes": ["user"]},
    }
    gov, _ = _gov("memory", policies, clock)
    await _admits(gov, _req("user:1", "fast"), 1)
    await _admits(gov, _req("user:1", "slow"), 1)
    clock.advance(601)
    await _admits(gov, _req("user:2", "fast"), 1)  # triggers the sweep
    keys = set(gov._buckets)
    assert ("fast", "requests", "user", "1") not in keys  # refilled and idle: evicted
    assert ("slow", "requests", "user", "1") in keys  # still draining: kept
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_governor_safety_net.py`
Expected: FAIL. Unknown policy, builtin default, scope mismatch, requests inheritance and concurrency deny; oversized tokens are never admitted; the limit does not grow; nothing is evicted.

- [ ] **Step 3: Implement in `governor.py`**

1. Import the helpers:

```python
from .policy_eval import clamp_token_units, effective_policy, scope_pairs
```

2. `_Bucket`: add `last_used: float = 0.0` as the last field, and set `self.last_used = now` at the top of `consume`.

3. Module constants, placed below the imports:

```python
# Idle-bucket eviction. A bucket that has refilled to capacity is indistinguishable
# from a fresh one, so dropping it is lossless.
_EVICT_IDLE_SEC = 600.0
_EVICT_INTERVAL_SEC = 60.0
_EVICT_BATCH = 5000
```

4. `__init__`: add `self._last_evict = self._time()` and `self._evict_cursor = 0`.

5. `_get_policy` becomes a raw lookup plus the shared rules:

```python
    def _lookup_policy(self, policy_id: str) -> dict[str, Any] | None:
        if self._policy_loader is not None:
            try:
                pol = self._policy_loader.get_policy(policy_id)  # type: ignore[attr-defined]
                if pol:
                    return pol
            except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                logger.debug(f"Policy loader failed; falling back to static policies: {e}")
        return self._policies.get(policy_id)

    def _get_policy(self, policy_id: str) -> dict[str, Any]:
        return effective_policy(self._lookup_policy, policy_id)
```

6. `_get_bucket` resizes a live bucket when the configured limit has changed:

```python
    def _get_bucket(self, policy_id: str, category: str, scope: str, entity_value: str, *, capacity: float, refill_per_sec: float) -> _Bucket:
        k = self._bucket_key(policy_id, category, scope, entity_value)
        b = self._buckets.get(k)
        now = self._time()
        if b is None:
            b = _Bucket(capacity=float(capacity), refill_per_sec=float(refill_per_sec), tokens=float(capacity), last_refill=now, last_used=now)
            self._buckets[k] = b
        elif b.capacity != float(capacity) or b.refill_per_sec != float(refill_per_sec):
            # A policy reload changed the limit: apply it now rather than at restart.
            b.refill(now)
            b.capacity = float(capacity)
            b.refill_per_sec = float(refill_per_sec)
            b.tokens = min(b.tokens, b.capacity)
        return b
```

7. Scope selection: replace every hand-built scope list with `scope_pairs`.
   - `_compute_headroom_requests_tokens`: replace the block from `scopes = self._scopes(policy)` through the `scope_keys.append((entity_scope, entity_value))` line with:

     ```python
             scope_keys = scope_pairs(policy, entity_scope, entity_value)
     ```

   - `_acquire_concurrency`: make the same replacement. Also change the `limit <= 0` branch to unbounded:

     ```python
             if limit <= 0:
                 return True, 0, {"limit": 0, "remaining": 10**9, "unbounded": True}
     ```

   - `reserve`, requests/tokens consumption: replace the `# global` and `# entity` bucket blocks with:

     ```python
                     for sc, ev in scope_pairs(pol, entity_scope, entity_value):
                         b = self._get_bucket(policy_id, category, sc, ev, capacity=capacity, refill_per_sec=refill_per_sec)
                         _ = b.consume(units, now)
     ```

   - `reserve` concurrency loop, the `commit` lease-release loop, `renew`, and `peek_with_policy`:
     - replace `for sc, ev in (("global", "*"), (entity_scope, entity_value)):` and the `if sc not in scopes and not (...): continue` guard with `for sc, ev in scope_pairs(pol, entity_scope, entity_value):`;
     - in the reserve concurrency loop, also skip when `limit <= 0` (unbounded): `if limit <= 0: continue`.

   After these edits, `grep -n '"global" in scopes\|in scopes or "entity"\|sc not in scopes' tldw_Server_API/app/core/Resource_Governance/governor.py` must print nothing.

8. Token clamp at the entry of `check` and `reserve`. In `check`, right after `pol = self._get_policy(policy_id)`:

   ```python
           req = dataclasses.replace(req, categories=clamp_token_units(pol, req.categories, capacity_includes_burst=True))
   ```

   In `reserve`, before `dec = await self.check(req)`:

   ```python
           _pol_for_clamp = self._get_policy(req.tags.get("policy_id") or "default")
           req = dataclasses.replace(req, categories=clamp_token_units(_pol_for_clamp, req.categories, capacity_includes_burst=True))
   ```

   Add `import dataclasses` at the top if it is not already imported.

9. Eviction. Call it at the top of `reserve`, right after `self._purge_expired_ops(now_purge)`:

   ```python
           self._maybe_evict_idle(now_purge)
   ```

   and add:

   ```python
       def _maybe_evict_idle(self, now: float) -> None:
           """Drop buckets that have refilled and sat idle; bounded work per call."""
           if now - self._last_evict < _EVICT_INTERVAL_SEC:
               return
           self._last_evict = now
           keys = list(self._buckets.keys())
           if keys:
               start = self._evict_cursor % len(keys)
               batch = keys[start:start + _EVICT_BATCH]
               self._evict_cursor = start + len(batch)
               for k in batch:
                   b = self._buckets.get(k)
                   if b is not None and now - b.last_used >= _EVICT_IDLE_SEC and b.available(now) >= b.capacity:
                       del self._buckets[k]
           for k in [k for k, m in self._leases.items() if not m]:
               del self._leases[k]
   ```

- [ ] **Step 4: Run the new tests and the existing memory suite**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_governor_safety_net.py tldw_Server_API/tests/Resource_Governance/test_governor_memory.py tldw_Server_API/tests/Resource_Governance/test_governor_memory_combined.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py`
Expected: all pass.
- If an existing test asserted that an unknown policy, a scope mismatch or a missing concurrency config is denied, it encoded the old bug. Change its expectation to the safety-net behaviour, and add a one-line comment citing the spec section (§4).
- **Check the one direction `scope_pairs` can tighten.** For concurrency (`streams`, `jobs`), an entity kind a policy did not list used to get no lease, which meant no limit. It now gets a per-entity lease. Confirm each policy with `streams` or `jobs` (`audio.default`, `media.default`) lists every entity kind its callers reserve with (`user`, `api_key`, and `ip` for audio). Run `grep -n "max_concurrent" tldw_Server_API/Config_Files/resource_governor_policies.yaml`, read each such policy's `scopes`, and record the check in the commit message. Requests and tokens only ever get looser.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/governor.py tldw_Server_API/tests/Resource_Governance/
git commit -m "fix(rg): memory governor never denies forever; resizes on reload; evicts idle buckets

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 3: Redis governor adopts the shared rules

**Files:**
- Modify: `tldw_Server_API/app/core/Resource_Governance/governor_redis.py`
  - `_get_policy` (~364-369)
  - `_scope_pairs` (~412-419)
  - the inline scope block (~1024-1028)
  - `check` (~803) and `reserve` (~1498) entry points
  - every `max_concurrent` deny site
- Test: `tldw_Server_API/tests/Resource_Governance/test_governor_safety_net.py` (extend)

**Interfaces:** consumes `effective_policy`, `scope_pairs` and `clamp_token_units` from Task 1.

- [ ] **Step 1: Extend the tests to cover Redis**

In `test_governor_safety_net.py`, change `BACKENDS = ["memory"]` to `BACKENDS = ["memory", "redis"]`. The Redis variant uses the in-process stub client, exactly like `test_governor_redis.py::test_requests_sliding_window_with_stub_redis`; the `_gov` helper already builds it that way.

- [ ] **Step 2: Run to verify the Redis cases fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_governor_safety_net.py -k redis`
Expected: FAIL on the unknown-policy, builtin-default, scope-mismatch, requests-inheritance, concurrency and oversized-token cases. `test_raised_limit_applies_without_restart[redis]` may already pass, because Redis reads limits per call. Keep it as the pin (Review Focus 5).

- [ ] **Step 3: Implement in `governor_redis.py`**

1. Imports:

   ```python
   from .policy_eval import clamp_token_units, effective_policy, scope_pairs
   ```

2. `_get_policy`:

   ```python
       def _lookup_policy(self, policy_id: str) -> dict[str, Any] | None:
           try:
               return self._policy_loader.get_policy(policy_id)
           except _RG_NONCRITICAL_EXCEPTIONS:
               return None

       def _get_policy(self, policy_id: str) -> dict[str, Any]:
           return effective_policy(self._lookup_policy, policy_id)
   ```

3. `_scope_pairs` body: `return scope_pairs(policy, entity_scope, entity_value)`.

4. At ~1024-1028, replace the inline `scopes = self._scopes(pol)` / `if "global" in scopes` / `if entity_scope in scopes or "entity" in scopes` block with `scope_keys = self._scope_pairs(pol, entity_scope, entity_value)`. Keep the variable name the surrounding code uses.

   After this, `grep -n '"global" in scopes\|in scopes or "entity"' tldw_Server_API/app/core/Resource_Governance/governor_redis.py` must print only the lines inside `policy_eval`-backed helpers, which means nothing in this file.

5. Token clamp at the start of `check` and `reserve`, once the policy is known. Use the window limit without burst:

   ```python
           req = dataclasses.replace(req, categories=clamp_token_units(self._get_policy(req.tags.get("policy_id") or "default"), req.categories, capacity_includes_burst=False))
   ```

6. Missing concurrency config is unbounded. Run `grep -n "max_concurrent" tldw_Server_API/app/core/Resource_Governance/governor_redis.py`. At every site where a limit of `0` produces a denial, allow instead, mirroring memory's `{"limit": 0, "remaining": 10**9, "unbounded": True}`. Where a site skips work, skip lease acquisition when `limit <= 0`.

- [ ] **Step 4: Run the Redis suites**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_governor_safety_net.py tldw_Server_API/tests/Resource_Governance/test_governor_redis.py tldw_Server_API/tests/Resource_Governance/test_governor_redis_property.py tldw_Server_API/tests/Resource_Governance/test_rg_metrics_redis_backend.py`
Expected: all pass. Handle an old-bug expectation as in Task 2 Step 4.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/tests/Resource_Governance/test_governor_safety_net.py
git commit -m "fix(rg): Redis governor shares the safety-net rules with the memory backend

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 4: Safety-net policy defaults and the policy-reference consistency test

**Files:**
- Modify: `tldw_Server_API/Config_Files/resource_governor_policies.yaml`
- Modify: `tldw_Server_API/app/services/startup_resource_governor.py` (log undefined references)
- Test: `tldw_Server_API/tests/Resource_Governance/test_policy_yaml_safety_net.py`
- Test: `tldw_Server_API/tests/Resource_Governance/test_policy_reference_consistency.py`

**Interfaces:** produces `log_undefined_policy_references(loader) -> list[str]` in `startup_resource_governor.py`.

- [ ] **Step 1: Write the failing tests**

`test_policy_yaml_safety_net.py`:

```python
"""The shipped policy file is a safety net: per-entity buckets, generous limits."""

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]

YAML = Path(__file__).resolve().parents[2] / "Config_Files" / "resource_governor_policies.yaml"
EMAIL_SENDERS = {"authnz.forgot_password", "authnz.resend_verification", "authnz.magic_link.request"}


def _policies():
    return yaml.safe_load(YAML.read_text(encoding="utf-8"))["policies"]


def test_only_email_senders_keep_a_server_wide_bucket():
    with_global = {pid for pid, pol in _policies().items() if "global" in (pol.get("scopes") or ["global"])}
    assert with_global == EMAIL_SENDERS


def test_default_policy_is_the_safety_net():
    default = _policies()["default"]
    assert default["requests"] == {"rpm": 600, "burst": 2.0}
    assert default["scopes"] == ["user", "api_key", "ip"]


@pytest.mark.parametrize(
    ("policy_id", "rpm", "burst"),
    [
        ("core.default", 600, 2.0),
        ("chat.default", 300, 2.0),
        ("character_chat.default", 300, 2.0),
        ("embeddings.default", 300, 2.0),
        ("workflows.default", 300, 2.0),
        ("watchlists.default", 300, 2.0),
        ("mcp.default", 300, 2.0),
        ("mcp.ingestion", 300, 2.0),
        ("mcp.read", 600, 2.0),
        ("mcp.search", 600, 2.0),
        ("authnz.default", 300, 2.0),
        ("research.default", 120, 2.0),
        ("evals.default", 120, 2.0),
        ("rag.default", 300, 2.0),
    ],
)
def test_interactive_policies_sit_above_normal_use(policy_id, rpm, burst):
    req = _policies()[policy_id]["requests"]
    assert (req["rpm"], req["burst"]) == (rpm, burst)


def test_chat_token_budget_is_a_per_user_runaway_guard():
    chat = _policies()["chat.default"]
    assert chat["tokens"] == {"per_min": 1_000_000, "burst": 1.5}
    assert "global" not in chat["scopes"]
```

`test_policy_reference_consistency.py`:

```python
"""Every policy ID the code or the route map refers to exists in the shipped YAML.

At runtime an unknown ID falls back to ``default``. CI still fails, so a typo is
caught before it silently loosens a strict policy.
"""

import re
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]

ROOT = Path(__file__).resolve().parents[3]
YAML = ROOT / "tldw_Server_API" / "Config_Files" / "resource_governor_policies.yaml"
APP = ROOT / "tldw_Server_API" / "app"
_LITERAL = re.compile(r"""policy_id\s*=\s*["']([a-z_]+(?:\.[a-z_]+)+)["']""")
_ENV_DEFAULT = re.compile(r"""(?:getenv|environ\.get)\(\s*["']RG_[A-Z_]+_POLICY_ID["']\s*,\s*["']([a-z_.]+)["']""")
# Referenced on purpose without a shipped policy; each entry says why.
OPTIONAL = {
    # "authnz.federation.login": "federation is opt-in; endpoints skip RG when undefined",
}


def _defined():
    return set(yaml.safe_load(YAML.read_text(encoding="utf-8"))["policies"])


def _referenced():
    data = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    refs = set(str(v) for v in (data["route_map"].get("by_path") or {}).values())
    refs |= set(str(v) for v in (data["route_map"].get("by_tag") or {}).values())
    for path in APP.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        refs |= set(_LITERAL.findall(text)) | set(_ENV_DEFAULT.findall(text))
    return refs


def test_every_referenced_policy_is_defined():
    missing = sorted(_referenced() - _defined() - set(OPTIONAL))
    assert missing == [], f"undefined policy IDs: {missing}"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_policy_yaml_safety_net.py tldw_Server_API/tests/Resource_Governance/test_policy_reference_consistency.py`
Expected:
- The YAML tests FAIL (the global scopes are there and `default` is missing).
- The consistency test either passes or lists undefined IDs. For each ID it lists, either add the policy to the YAML (if a route or code path needs it governed) or add it to `OPTIONAL` with a one-line reason.

- [ ] **Step 3: Edit the YAML**

- `templates.core` → `requests: { rpm: 600, burst: 2.0 }`, `scopes: [user, api_key, ip]`.
- `templates.mcp_base` → `requests: { rpm: 300, burst: 2.0 }`, `scopes: [user, client, api_key]`.
- Add, directly after `core.default: *core_policy`:

```yaml
  # Safety net for every /api/ route no path or tag maps (see ADR-056).
  default: *core_policy
```

- `health.default` → `scopes: [ip]`.
- `chat.default` → `requests: { rpm: 300, burst: 2.0 }`, `tokens: { per_min: 1000000, burst: 1.5 }`, `scopes: [user, api_key, conversation]`.
- `mcp.read` and `mcp.search` → `requests: { rpm: 600, burst: 2.0 }`.
- `authnz.default` → `requests: { rpm: 300, burst: 2.0 }`, `scopes: [entity]`.
- `authnz.reset_password`, `authnz.mfa.verify`, `authnz.mfa.login` → `scopes: [entity]`, rates unchanged.
- `authnz.forgot_password`, `authnz.resend_verification`, `authnz.magic_link.request` → unchanged (they keep `global`).
- `embeddings.default`, `character_chat.default`, `workflows.default`, `watchlists.default`, `rag.default` → `requests: { rpm: 300, burst: 2.0 }`, keeping each one's other keys and scopes.
- `research.default`, `evals.default` → `requests: { rpm: 120, burst: 2.0 }`, keeping the other keys.

- [ ] **Step 4: Log undefined references at startup**

In `startup_resource_governor.py`, add the function below and call it from `init_resource_governor` right after `_register_policy_snapshot_callback(app, rg_loader)`. It logs and continues; it never raises.

```python
def log_undefined_policy_references(loader: Any) -> list[str]:
    """Log route-map targets that name no policy. They fall back to ``default`` at runtime."""
    try:
        snap = loader.get_snapshot()
        route_map = dict(getattr(snap, "route_map", {}) or {})
        policies = set((getattr(snap, "policies", {}) or {}).keys())
    except _STARTUP_GUARD_EXCEPTIONS:
        return []
    targets = set(str(v) for v in (route_map.get("by_path") or {}).values())
    targets |= set(str(v) for v in (route_map.get("by_tag") or {}).values())
    missing = sorted(targets - policies)
    for pid in missing:
        logger.error("RG route_map names undefined policy {!r}; requests fall back to 'default'", pid)
    return missing
```

Add a unit test to `test_policy_reference_consistency.py`:

```python
def test_startup_logs_undefined_route_map_targets():
    from types import SimpleNamespace

    from tldw_Server_API.app.services.startup_resource_governor import log_undefined_policy_references

    snap = SimpleNamespace(route_map={"by_path": {"/a": "known", "/b": "typo"}}, policies={"known": {}})
    loader = SimpleNamespace(get_snapshot=lambda: snap)
    assert log_undefined_policy_references(loader) == ["typo"]
```

- [ ] **Step 5: Run the tests to verify they pass, then run the RG suite**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/Resource_Governance`
Expected: all pass.
- E2E tests that drained a `global` bucket (for example `test_e2e_*_headers.py`, `test_e2e_tokens_daily_cap.py`) may now need more requests or a per-user expectation. Update them to exercise the per-entity bucket, with a comment citing spec §3.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/Config_Files/resource_governor_policies.yaml tldw_Server_API/app/services/startup_resource_governor.py tldw_Server_API/tests/Resource_Governance/
git commit -m "fix(rg): safety-net defaults (no server-wide buckets, generous per-user limits, default policy)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 5: MCP category fallback and the auth single charge

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/auth/rate_limiter.py` (`_maybe_enforce_with_rg_mcp`, ~227-231)
- Modify: `tldw_Server_API/app/api/v1/endpoints/auth.py` (`_reserve_auth_rg_requests`, ~1115-1127)
- Test: `tldw_Server_API/tests/Resource_Governance/test_mcp_category_fallback.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_rg_single_charge.py`

**Interfaces:** none new.

- [ ] **Step 1: Write the failing tests**

`test_mcp_category_fallback.py`:

```python
import pytest

from tldw_Server_API.app.core.MCP_unified.auth import rate_limiter as mcp_rl
from tldw_Server_API.app.core.Resource_Governance.governor import RGDecision

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit, pytest.mark.asyncio]


class _SpyGov:
    def __init__(self):
        self.policy_ids = []

    async def reserve(self, req, op_id=None):
        self.policy_ids.append(req.tags["policy_id"])
        return RGDecision(allowed=True, retry_after=None, details={}), "h"

    async def commit(self, handle, actuals=None, op_id=None):
        return None


class _Loader:
    def get_policy(self, pid):
        return {"requests": {"rpm": 60}} if pid in {"mcp.default", "mcp.read"} else None


async def _run(monkeypatch, category):
    gov = _SpyGov()

    async def _get():
        return gov

    monkeypatch.setattr(mcp_rl, "_get_mcp_rg_governor", _get)
    monkeypatch.setattr(mcp_rl, "_rg_mcp_loader", _Loader())
    result = await mcp_rl._maybe_enforce_with_rg_mcp(key="user:1", category=category)
    return gov, result


async def test_undefined_category_uses_mcp_default(monkeypatch):
    gov, result = await _run(monkeypatch, "browser")
    assert gov.policy_ids == ["mcp.default"] and result["allowed"]


async def test_defined_category_keeps_its_policy(monkeypatch):
    gov, _ = await _run(monkeypatch, "read")
    assert gov.policy_ids == ["mcp.read"]
```

`test_auth_rg_single_charge.py`:

```python
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import auth as auth_ep
from tldw_Server_API.app.core.Resource_Governance.governor import RGDecision

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit, pytest.mark.asyncio]


class _SpyGov:
    def __init__(self):
        self.entities = []

    async def reserve(self, req, op_id=None):
        self.entities.append(req.entity)
        return RGDecision(allowed=True, retry_after=None, details={}), None


def _request(policy_id=None):
    state = SimpleNamespace(rg_policy_id=policy_id) if policy_id else SimpleNamespace()
    return SimpleNamespace(state=state, url=SimpleNamespace(path="/api/v1/auth/forgot-password"), app=SimpleNamespace(state=SimpleNamespace()), client=SimpleNamespace(host="203.0.113.9"), headers={})


@pytest.fixture
def spy(monkeypatch):
    gov = _SpyGov()

    async def _get(_request):
        return gov

    monkeypatch.setattr(auth_ep, "_get_auth_endpoint_rg_governor", _get)
    monkeypatch.setattr(auth_ep, "_auth_rg_policy_defined", lambda *_a: True)
    monkeypatch.setattr(auth_ep, "_auth_request_client_ip", lambda _request: "203.0.113.9")
    return gov


async def test_ingress_charged_same_policy_skips_ip_reservation(spy):
    allowed, _ = await auth_ep._reserve_auth_rg_requests(_request("authnz.forgot_password"), policy_id="authnz.forgot_password")
    assert allowed and spy.entities == []


async def test_per_email_throttle_still_applies(spy):
    await auth_ep._reserve_auth_rg_requests(_request("authnz.forgot_password"), policy_id="authnz.forgot_password", entity="email:abc")
    assert spy.entities == ["email:abc"]


async def test_no_ingress_charge_reserves_by_ip(spy):
    await auth_ep._reserve_auth_rg_requests(_request(), policy_id="authnz.forgot_password")
    assert len(spy.entities) == 1 and spy.entities[0].startswith("ip:")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_mcp_category_fallback.py tldw_Server_API/tests/AuthNZ_Unit/test_auth_rg_single_charge.py`
Expected:
- `test_undefined_category_uses_mcp_default` FAILS: it sees `mcp.browser`.
- `test_ingress_charged_same_policy_skips_ip_reservation` FAILS: it sees an `ip:` reservation.
- The other tests pass.

- [ ] **Step 3: Implement**

MCP, replacing `policy_id = f"mcp.{category}"`:

```python
    policy_id = f"mcp.{category}"
    try:
        if _rg_mcp_loader is not None and not _rg_mcp_loader.get_policy(policy_id):
            policy_id = "mcp.default"  # categories without their own policy share mcp.default
    except Exception:  # noqa: BLE001 - a lookup failure must not deny the tool call
        policy_id = "mcp.default"
```

Auth, at the top of `_reserve_auth_rg_requests`, after `_ = fail_open`:

```python
    # Ingress already charged this request to this policy's per-IP bucket. The two
    # IP derivations (AuthNZ trusted proxies vs RG_TRUSTED_PROXIES) can disagree,
    # so compare the policy, not the IP. Reservations keyed on something else
    # (email hash, user) still apply. See TASK-13144 for unifying the derivations.
    if entity is None and getattr(getattr(request, "state", None), "rg_policy_id", None) == policy_id:
        return True, None
```

- [ ] **Step 4: Run the tests plus the existing auth-RG and MCP suites**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_mcp_category_fallback.py tldw_Server_API/tests/AuthNZ_Unit/test_auth_rg_single_charge.py tldw_Server_API/tests/Resource_Governance/test_rg_cutover_embeddings_mcp.py tldw_Server_API/tests/Resource_Governance/test_rg_cutover_evals_authnz_character_web.py tldw_Server_API/tests/Resource_Governance/test_auth_route_map_coverage.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/auth/rate_limiter.py tldw_Server_API/app/api/v1/endpoints/auth.py tldw_Server_API/tests/Resource_Governance/test_mcp_category_fallback.py tldw_Server_API/tests/AuthNZ_Unit/test_auth_rg_single_charge.py
git commit -m "fix(rg): MCP categories fall back to mcp.default; auth endpoints are charged once

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 6: Ship PR A

- [ ] **Step 1: Run the broad suites**

Run: `RUN_EVALUATIONS=1 RUN_JOBS=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 8 tldw_Server_API/tests/Resource_Governance tldw_Server_API/tests/AuthNZ_Unit tldw_Server_API/tests/Embeddings tldw_Server_API/tests/lint`
Expected: no failures beyond a dev baseline.
- If something fails, check it against the dev baseline (`git archive origin/dev` into the scratchpad, as in the FastAPI bump) before attributing it.

- [ ] **Step 2: Ledger**

```bash
backlog task edit <PARENT> --check-ac 1 --append-notes "PR A (relief): shared policy_eval; memory + Redis never deny forever; resize on reload; idle eviction; safety-net YAML; MCP fallback; auth single charge."
```

- [ ] **Step 3: Rebase, push and open the PR**

```bash
git fetch -q origin dev && git rebase -q origin/dev && git push -q -u origin fix/rg-safety-net-relief
gh pr create --base dev --head fix/rg-safety-net-relief --title "fix(rg): safety-net defaults and no permanent 429s (spec 1, PR A)" --body-file <body.md>
```

The body summarises Tasks 1-5, links the spec and the plan, and ends with the waiver line and the Claude Code footer.

- [ ] **Step 4: Address every Qodo finding (fix, or decline with a posted rationale), then merge when all 7 required checks pass.** Never mention or suggest auto-merge.

---

## PR B — Coverage: resolver, identity, audits, lints, replay

Branch: `feat/rg-policy-resolver` from `origin/dev` after PR A merges.

### Task 7: Policy resolver and served-route index

**Files:**
- Create: `tldw_Server_API/app/core/Resource_Governance/policy_resolver.py`
- Test: `tldw_Server_API/tests/Resource_Governance/test_policy_resolver.py`

**Interfaces:**
- Consumes:
  - `iter_served_routes` and `ServedRoute` from `tldw_Server_API.app.core.Utils.fastapi_routes`
  - `DEFAULT_POLICY_ID` from `policy_eval`
- Produces:
  - `compile_route_glob(pattern: str) -> re.Pattern[str]`
  - `class PolicyResolver(route_map: Mapping[str, Any], served_routes: Iterable[ServedRoute])` with `.resolve(path: str, method: str) -> str | None`
  - `get_policy_resolver(app: Any) -> PolicyResolver | None`

- [ ] **Step 1: Write the failing tests**

```python
"""Policy resolution: by_path, then the innermost mapped tag, then default for /api/."""

from types import SimpleNamespace

import pytest
from fastapi import APIRouter, FastAPI

from tldw_Server_API.app.core.Resource_Governance.policy_resolver import (
    PolicyResolver,
    compile_route_glob,
    get_policy_resolver,
)
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]


def _app():
    inner = APIRouter(tags=["writing"])

    @inner.get("/docs/{doc_id}")
    def doc(doc_id: str) -> dict:
        return {}

    @inner.get("/docs/search")
    def search() -> dict:
        return {}

    middle = APIRouter(tags=["content"])
    middle.include_router(inner, prefix="/writing")
    app = FastAPI()
    app.include_router(middle, prefix="/api/v1")

    @app.post("/api/v1/auth/login", tags=["authentication"])
    def login() -> dict:
        return {}

    return app


ROUTE_MAP = {
    "by_path": {"/api/v1/auth*": "authnz.default", "/api/v1/media/*/reprocess": "media.default"},
    "by_tag": {"writing": "writing.policy", "content": "content.policy", "authentication": "wrong.policy"},
}


def _resolver(route_map=ROUTE_MAP, app=None):
    return PolicyResolver(route_map, iter_served_routes((app or _app()).routes))


def test_path_beats_tag():
    assert _resolver().resolve("/api/v1/auth/login", "POST") == "authnz.default"


def test_innermost_tag_wins_through_nested_includes():
    assert _resolver().resolve("/api/v1/writing/docs/7", "GET") == "writing.policy"


def test_outer_tag_applies_when_inner_tag_unmapped():
    rm = {"by_path": {}, "by_tag": {"content": "content.policy"}}
    assert _resolver(rm).resolve("/api/v1/writing/docs/7", "GET") == "content.policy"


def test_first_served_match_wins_within_a_prefix_group():
    # /docs/{doc_id} and /docs/search share the 4-segment group; served order decides, as in Starlette.
    rm = {"by_path": {}, "by_tag": {"writing": "writing.policy"}}
    assert _resolver(rm).resolve("/api/v1/writing/docs/search", "GET") == "writing.policy"


def test_method_mismatch_falls_back_to_default():
    assert _resolver().resolve("/api/v1/writing/docs/7", "DELETE") == "default"


def test_head_request_falls_back_to_default():
    assert _resolver().resolve("/api/v1/writing/docs/7", "HEAD") == "default"


def test_unmapped_api_path_resolves_to_default():
    assert _resolver().resolve("/api/v1/nothing/here", "GET") == "default"


def test_non_api_path_is_ungoverned():
    assert _resolver().resolve("/docs", "GET") is None
    assert _resolver().resolve("/static/app.js", "GET") is None


def test_mid_path_glob_matches_one_or_more_segments():
    rx = compile_route_glob("/api/v1/media/*/reprocess")
    assert rx.match("/api/v1/media/42/reprocess")
    assert rx.match("/api/v1/media/a/b/reprocess")
    assert not rx.match("/api/v1/media/42/reprocess/extra")
    assert compile_route_glob("/api/v1/auth*").match("/api/v1/authnz/x")  # trailing * is a prefix


def test_resolver_cache_is_rebuilt_when_snapshot_changes():
    app = _app()
    snap1 = SimpleNamespace(route_map={"by_path": {}, "by_tag": {"writing": "one"}})
    loader = SimpleNamespace(get_snapshot=lambda: loader.snap, snap=snap1)
    app.state.rg_policy_loader = loader
    assert get_policy_resolver(app).resolve("/api/v1/writing/docs/7", "GET") == "one"
    loader.snap = SimpleNamespace(route_map={"by_path": {}, "by_tag": {"writing": "two"}})
    assert get_policy_resolver(app).resolve("/api/v1/writing/docs/7", "GET") == "two"


def test_resolver_is_rebuilt_when_routes_are_added():
    app = _app()
    snap = SimpleNamespace(route_map={"by_path": {}, "by_tag": {"late": "late.policy"}})
    app.state.rg_policy_loader = SimpleNamespace(get_snapshot=lambda: snap)
    assert get_policy_resolver(app).resolve("/api/v1/late", "GET") == "default"
    late = APIRouter(tags=["late"])

    @late.get("/late")
    def late_ep() -> dict:
        return {}

    app.include_router(late, prefix="/api/v1")
    assert get_policy_resolver(app).resolve("/api/v1/late", "GET") == "late.policy"


def test_no_loader_means_no_resolver():
    assert get_policy_resolver(FastAPI()) is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_policy_resolver.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
"""Resolve which Resource Governor policy governs a request.

Order (ADR-056):
1. ``route_map.by_path`` globs, first match.
2. The innermost mapped tag of the served route.
3. ``default`` for any other ``/api/`` path.

Anything else is ungoverned. The middleware, both coverage audits and the CI
route-map lint call this module, so what they report is what is enforced.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from starlette.routing import compile_path

from tldw_Server_API.app.core.Utils.fastapi_routes import ServedRoute, iter_served_routes

from .policy_eval import DEFAULT_POLICY_ID

_PREFIX_DEPTH = 4
_CACHE_SIZE = 4096


def compile_route_glob(pattern: str) -> re.Pattern[str]:
    """Compile a ``by_path`` glob: ``*`` matches anything, anchored unless it ends with ``*``."""
    regex = re.escape(pattern).replace("\\*", ".*")
    if not pattern.endswith("*"):
        regex += "$"
    return re.compile(regex)


@dataclass(frozen=True)
class _IndexedRoute:
    regex: re.Pattern[str]
    methods: frozenset[str]
    tags: tuple[str, ...]


def _static_prefix(path: str) -> tuple[str, ...]:
    out: list[str] = []
    for seg in path.strip("/").split("/"):
        if not seg or "{" in seg or len(out) == _PREFIX_DEPTH:
            break
        out.append(seg)
    return tuple(out)


class PolicyResolver:
    """Resolve (path, method) to a policy ID, or None when ungoverned."""

    def __init__(self, route_map: Mapping[str, Any], served_routes: Iterable[ServedRoute]) -> None:
        self._by_path = [
            (compile_route_glob(str(pattern)), str(policy))
            for pattern, policy in dict(route_map.get("by_path") or {}).items()
        ]
        self._by_tag = {str(tag): str(policy) for tag, policy in dict(route_map.get("by_tag") or {}).items()}
        self._groups: dict[tuple[str, ...], list[_IndexedRoute]] = {}
        for route in served_routes:
            if not route.path or not route.methods:
                continue  # mounts and websockets carry no HTTP methods
            regex, _fmt, _conv = compile_path(route.path)
            self._groups.setdefault(_static_prefix(route.path), []).append(
                _IndexedRoute(regex, frozenset(route.methods), tuple(route.tags))
            )
        self._cache: OrderedDict[tuple[str, str], str | None] = OrderedDict()

    def resolve(self, path: str, method: str) -> str | None:
        key = (method.upper(), path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        result = self._resolve_uncached(path, key[0])
        self._cache[key] = result
        if len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    def _resolve_uncached(self, path: str, method: str) -> str | None:
        for regex, policy in self._by_path:
            if regex.match(path):
                return policy
        tagged = self._resolve_tag(path, method)
        if tagged is not None:
            return tagged
        return DEFAULT_POLICY_ID if path.startswith("/api/") else None

    def _resolve_tag(self, path: str, method: str) -> str | None:
        # ponytail: tries the longest static-prefix group first, then served order within a
        # group (as Starlette does). Strict served order across groups would cost a full scan.
        segments = [s for s in path.strip("/").split("/") if s]
        for depth in range(min(len(segments), _PREFIX_DEPTH), -1, -1):
            for route in self._groups.get(tuple(segments[:depth]), ()):
                if method in route.methods and route.regex.match(path):
                    for tag in reversed(route.tags):
                        if tag in self._by_tag:
                            return self._by_tag[tag]
                    return None
        return None


def _routes_version(app: Any) -> Any:
    # ponytail: O(1) top-level counter (bumped by app.include_router / add_api_route), read per
    # request. _get_routes_version() would be exact for nested routers but walks every route
    # (~2k) on each call. Upgrade path: poll it on a timer if nested routers ever mutate
    # after startup.
    router = getattr(app, "router", None)
    version = getattr(router, "_routes_version", None)  # FastAPI private attribute, pinned
    if isinstance(version, int):
        return (version, len(getattr(app, "routes", None) or []))
    return len(getattr(app, "routes", None) or [])


def get_policy_resolver(app: Any) -> PolicyResolver | None:
    """Return the app's resolver, rebuilt when the route map or the route table changes."""
    state = getattr(app, "state", None)
    loader = getattr(state, "rg_policy_loader", None)
    if loader is None:
        return None
    try:
        snap = loader.get_snapshot()
    except (AttributeError, RuntimeError):
        return None
    version = _routes_version(app)
    cached = getattr(state, "rg_policy_resolver", None)
    if cached is not None and cached[0] is snap and cached[1] == version:
        return cached[2]
    resolver = PolicyResolver(getattr(snap, "route_map", {}) or {}, iter_served_routes(getattr(app, "routes", []) or []))
    state.rg_policy_resolver = (snap, version, resolver)
    return resolver
```

Also add to `tests/Utils/test_fastapi_routes.py` a guard for the private version API the resolver relies on:

```python
def test_router_routes_version_changes_on_include() -> None:
    """policy_resolver rebuilds its index on this private FastAPI counter (read per request)."""
    app = FastAPI()
    before = app.router._routes_version
    extra = APIRouter()

    @extra.get("/x")
    def x() -> None:
        return None

    app.include_router(extra)
    assert isinstance(app.router._routes_version, int) and app.router._routes_version != before
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_policy_resolver.py tldw_Server_API/tests/Utils/test_fastapi_routes.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/policy_resolver.py tldw_Server_API/tests/Resource_Governance/test_policy_resolver.py tldw_Server_API/tests/Utils/test_fastapi_routes.py
git commit -m "feat(rg): policy resolver (path, innermost tag, default) over a served-route index

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 8: Middleware resolves policies through the resolver (tag enforcement)

**Files:**
- Modify: `tldw_Server_API/app/core/Resource_Governance/middleware_simple.py`
  - delete `_compiled_map` (line 45), `_init_route_map` (78-99), and the `_derive_policy_id` body (101-159)
  - delete the `_init_route_map` call in `__call__` (230-232)
- Test: `tldw_Server_API/tests/Resource_Governance/test_middleware_tag_enforcement.py`

**Interfaces:** consumes `get_policy_resolver(app)` from Task 7.

- [ ] **Step 1: Write the failing request-level test** (TASK-13395's acceptance criterion)

```python
"""RGSimpleMiddleware enforces tag-only policies, before routing, through nested includes."""

import pytest
import yaml
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware
from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit, pytest.mark.asyncio]


async def _client(tmp_path, monkeypatch, policies, route_map):
    monkeypatch.delenv("RG_POLICY_PATH", raising=False)
    path = tmp_path / "rg.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "policies": policies, "route_map": route_map}), encoding="utf-8")
    loader = PolicyLoader(path, PolicyReloadConfig(enabled=False))
    await loader.load_once()

    inner = APIRouter(tags=["writing"])

    @inner.get("/docs/{doc_id}")
    def doc(doc_id: str) -> dict:
        return {"ok": True}

    middle = APIRouter()
    middle.include_router(inner, prefix="/writing")
    app = FastAPI()
    app.include_router(middle, prefix="/api/v1")

    @app.get("/api/v1/unmapped")
    def unmapped() -> dict:
        return {"ok": True}

    app.add_middleware(RGSimpleMiddleware)
    app.state.rg_policy_loader = loader
    app.state.rg_governor = MemoryResourceGovernor(policy_loader=loader)
    return TestClient(app)


async def test_tag_only_route_is_governed(tmp_path, monkeypatch):
    client = await _client(
        tmp_path,
        monkeypatch,
        policies={"tagpol": {"requests": {"rpm": 2, "burst": 1.0}, "scopes": ["ip"]}, "default": {"requests": {"rpm": 1000}}},
        route_map={"by_path": {}, "by_tag": {"writing": "tagpol"}},
    )
    codes = [client.get("/api/v1/writing/docs/1").status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert client.get("/api/v1/writing/docs/1").json()["policy_id"] == "tagpol"


async def test_unmapped_api_route_is_governed_by_default(tmp_path, monkeypatch):
    client = await _client(
        tmp_path,
        monkeypatch,
        policies={"default": {"requests": {"rpm": 1, "burst": 1.0}, "scopes": ["ip"]}},
        route_map={"by_path": {}, "by_tag": {}},
    )
    assert [client.get("/api/v1/unmapped").status_code for _ in range(2)] == [200, 429]
    assert client.get("/api/v1/unmapped").json()["policy_id"] == "default"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_middleware_tag_enforcement.py`
Expected: FAIL. Both routes return 200 every time: the tag branch is dead and there is no `default`.

- [ ] **Step 3: Implement**

Replace `_derive_policy_id` with:

```python
    def _derive_policy_id(self, request: Request) -> str | None:
        """Path, then the innermost mapped tag, then default (ADR-056)."""
        from .policy_resolver import get_policy_resolver

        try:
            resolver = get_policy_resolver(request.app)
            return resolver.resolve(request.url.path or "/", request.method) if resolver else None
        except _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("RGSimpleMiddleware: policy resolution failed: {}", exc)
            return None
```

Then:
- delete `self._compiled_map` from `__init__`;
- delete the `_init_route_map` method;
- delete the `with contextlib.suppress(...): self._init_route_map(request)` lines in `__call__`.

Update the module docstring's first sentence to "derives a policy_id via policy_resolver (path, tag, default)".

- [ ] **Step 4: Run the new test and the existing middleware tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_middleware_tag_enforcement.py tldw_Server_API/tests/Resource_Governance/test_middleware_simple.py tldw_Server_API/tests/Resource_Governance/test_middleware_enforcement_extended.py tldw_Server_API/tests/Resource_Governance/test_middleware_tokens_headers.py tldw_Server_API/tests/Resource_Governance/test_realtime_route_policy.py`
Expected: all pass. Tests that relied on the removed chat/audio heuristics now resolve through `by_path` or `default`. If one asserted a heuristic policy ID for a path that `by_path` doesn't cover, update it to the resolver's answer and cite spec §1.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/middleware_simple.py tldw_Server_API/tests/Resource_Governance/
git commit -m "feat(rg): ingress enforces tag policies and a default via the policy resolver

Closes the dead by_tag branch (TASK-13395).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 9: Charge the authenticated principal, not the proxy IP

**Files:**
- Modify: `tldw_Server_API/app/core/Resource_Governance/middleware_simple.py` (the ADR-044 block at ~238-274; the entity derivation at ~305)
- Modify: `tldw_Server_API/app/core/Resource_Governance/deps.py` (`derive_entity_key`: remove the hashed `X-API-KEY` and bearer fallbacks at ~136-152)
- Test: `tldw_Server_API/tests/Resource_Governance/test_middleware_identity.py`
- Update as needed: `tldw_Server_API/tests/Resource_Governance/test_middleware_cookie_owner.py`

**Interfaces:** produces `RGSimpleMiddleware._principal_entity(request) -> str | None`.

- [ ] **Step 1: Write the failing tests**

```python
"""Ingress charges the validated principal; invalid credentials fall back to the IP bucket."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ import auth_principal_resolver, jwt_service
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]


class _Snap:
    route_map = {"by_path": {"/api/v1/*": "p"}, "by_tag": {}}
    tenant = {}
    policies = {}


class _Loader:
    def get_snapshot(self):
        return _Snap()

    def get_policy(self, pid):
        return {"requests": {"rpm": 1, "burst": 1.0}, "scopes": ["user", "api_key", "ip"]}


SESSION = get_settings().SINGLE_USER_SESSION_COOKIE_NAME


class _FakeJwt:
    def decode_access_token(self, token):
        if token != "a.valid.jwt":
            raise ValueError("bad signature")
        return {"sub": "42", "scope": "notes.read"}  # a scoped (virtual-key style) token


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("RG_POLICY_PATH", raising=False)
    calls = []

    async def fake_principal(request):
        calls.append(request.url.path)
        user = request.cookies.get(SESSION) or request.headers.get("X-API-KEY")
        if not user or not user.isdigit():
            raise HTTPException(status_code=401, detail="bad credentials")
        return AuthPrincipal(kind="user", user_id=int(user))

    monkeypatch.setattr(auth_principal_resolver, "get_auth_principal", fake_principal)
    monkeypatch.setattr(jwt_service, "get_jwt_service", lambda: _FakeJwt())
    monkeypatch.setattr(get_settings(), "AUTH_MODE", "multi_user")
    app = FastAPI()

    @app.get("/api/v1/thing")
    def thing() -> dict:
        return {"ok": True}

    app.add_middleware(RGSimpleMiddleware)
    app.state.rg_policy_loader = _Loader()
    app.state.rg_governor = MemoryResourceGovernor(policy_loader=_Loader())
    tc = TestClient(app)
    tc.principal_calls = calls
    return tc


def test_two_cookie_users_behind_one_ip_get_separate_buckets(client):
    assert client.get("/api/v1/thing", cookies={SESSION: "1"}).status_code == 200
    assert client.get("/api/v1/thing", cookies={SESSION: "1"}).status_code == 429
    assert client.get("/api/v1/thing", cookies={SESSION: "2"}).status_code == 200


def test_bearer_jwt_is_keyed_by_verified_subject_without_principal_resolution(client):
    auth = {"Authorization": "Bearer a.valid.jwt"}
    assert client.get("/api/v1/thing", headers=auth).status_code == 200
    assert client.get("/api/v1/thing", headers=auth).status_code == 429  # user:42 bucket
    assert client.principal_calls == []  # scoped-token route checks never run pre-routing


def test_invalid_jwt_charges_ip(client):
    assert client.get("/api/v1/thing", headers={"Authorization": "Bearer forged.jwt.x"}).status_code == 200
    assert client.get("/api/v1/thing").status_code == 429  # same anonymous IP bucket


def test_rotating_fake_tokens_share_the_ip_bucket(client):
    assert client.get("/api/v1/thing", headers={"X-API-KEY": "fake-a"}).status_code == 200
    assert client.get("/api/v1/thing", headers={"X-API-KEY": "fake-b"}).status_code == 429


def test_invalid_credentials_reach_the_route(client):
    # The middleware never answers 401 itself; the route's auth decides.
    assert client.get("/api/v1/thing", headers={"X-API-KEY": "fake"}).status_code == 200


def test_non_session_cookie_charges_ip_and_reaches_route(client):
    assert client.get("/api/v1/thing", cookies={"theme": "dark", "csrf_token": "t"}).status_code == 200
    assert client.principal_calls == []  # not a credential: never resolved
    assert client.get("/api/v1/thing").status_code == 429  # same anonymous IP bucket
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_middleware_identity.py`
Expected:
- The cookie-isolation test FAILS: both users share `ip:testclient`.
- The rotating-token test FAILS: each fake key gets its own hashed bucket.
- `test_non_session_cookie_charges_ip_and_reaches_route` may already pass.

- [ ] **Step 3: Implement**

In `middleware_simple.py`, delete the ADR-044 block (the `if (request.cookies and ...)` statement through its inner `return`) and add this method:

```python
    async def _principal_entity(self, request: Request) -> str | None:
        """Charge the validated principal. Invalid or absent credentials charge the IP.

        - A multi-user bearer JWT is keyed by its signature-verified ``sub``
          (``decode_access_token``: no database access, no revocation check). Full
          principal resolution before routing would run the scoped-token check,
          which needs the matched route. For virtual keys that check always fails
          pre-routing, and it would log a security warning on every request.
          Revocation is still enforced by the route's own auth; a revoked token only
          spends its own user's bucket.
        - API keys, non-JWT bearers and the single-user session cookie go through
          ``get_auth_principal``. It caches its AuthContext on request state, so
          endpoint auth reuses this validation. A failure is not cached; the route
          re-checks and returns its own 401.
        - Other cookies (CSRF, theme, analytics) are not credentials, so they are
          never resolved.
        """
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings

        settings = get_settings()
        auth_header = request.headers.get("Authorization") or ""
        has_session_cookie = bool(request.cookies.get(settings.SINGLE_USER_SESSION_COOKIE_NAME))
        if not (auth_header or request.headers.get("X-API-KEY") is not None or has_session_cookie):
            return None
        token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else ""
        if token and token.count(".") == 2 and settings.AUTH_MODE != "single_user":
            from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service

            try:
                sub = get_jwt_service().decode_access_token(token).get("sub")
            except Exception as exc:  # noqa: BLE001 - identity is best-effort; route auth still decides
                logger.debug("RG ingress JWT identity fell back to IP: {}", type(exc).__name__)
                return None
            return f"user:{sub}" if sub else None
        from tldw_Server_API.app.core.AuthNZ import auth_principal_resolver

        try:
            principal = await auth_principal_resolver.get_auth_principal(request)
        except Exception as exc:  # noqa: BLE001 - identity is best-effort; route auth still decides
            logger.debug("RG ingress identity fell back to IP: {}", type(exc).__name__)
            return None
        if getattr(principal, "user_id", None) is not None:
            return f"user:{principal.user_id}"
        if getattr(principal, "api_key_id", None) is not None:
            return f"api_key:{principal.api_key_id}"
        return None
```

In `__call__`, replace `entity = self._derive_entity(request)` with:

```python
        entity = await self._principal_entity(request) or self._derive_entity(request)
```

In `deps.derive_entity_key`, delete the two blocks commented `# Header-based API key fallback (hashed)` and `# Authorization bearer fallback (hashed as api_key)`. An unvalidated credential must not mint a bucket.

Check the other callers keep working. Run:

```bash
git grep -n "derive_entity_key\|get_entity_key" -- 'tldw_Server_API/app/*.py'
```

Each non-middleware caller runs after auth has populated `request.state.user_id`/`api_key_id`. Verify that by reading each call site, and list them in the commit message.

- [ ] **Step 4: Run the tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_middleware_identity.py tldw_Server_API/tests/Resource_Governance/test_middleware_cookie_owner.py tldw_Server_API/tests/Resource_Governance/test_middleware_trusted_proxy_ip.py tldw_Server_API/tests/Resource_Governance/test_deps_trusted_proxy.py tldw_Server_API/tests/Resource_Governance/test_tenant_hash_entity.py`
Expected: the new tests pass. `test_middleware_cookie_owner.py` cases that expected the middleware itself to return 401 for a revoked cookie now see the route's response instead. Update them to assert that the request reaches the route and is charged to the IP, citing ADR-056 / spec §2. Tests asserting the hashed `api_key:` entity for raw headers encode the bypass; update them to expect the `ip:` entity.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/middleware_simple.py tldw_Server_API/app/core/Resource_Governance/deps.py tldw_Server_API/tests/Resource_Governance/
git commit -m "fix(rg): charge the validated principal; unvalidated credentials no longer mint buckets

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 10: Audits report what the resolver enforces

**Files:**
- Modify: `tldw_Server_API/app/core/Resource_Governance/coverage_audit.py` (`audit_governor_coverage`; remove `_route_is_mapped` and `_glob_matches` once unused)
- Modify: `tldw_Server_API/app/services/startup_resource_governor.py` (`_audit_route_map_coverage`; remove `_route_map_matches` once unused)
- Test: `tldw_Server_API/tests/Resource_Governance/test_coverage_audit.py` (update the stub loader, add cases)

**Interfaces:** consumes `get_policy_resolver`, `DEFAULT_POLICY_ID`.

- [ ] **Step 1: Write the failing tests** (add them to `TestAuditGovernorCoverage`)

First add `def get_policy(self, pid): return {"requests": {"rpm": 1}}` to the test file's `_Loader`. Then add:

```python
    @pytest.mark.unit
    def test_tag_only_included_route_is_protected(self):
        router = APIRouter(tags=["writing"])

        @router.get("/docs")
        def docs() -> list[str]:
            return []

        app = FastAPI()
        app.include_router(router, prefix="/api/v1/writing")
        app.user_middleware.append(_Middleware(_RGSimpleMiddleware))
        app.state.rg_policy_loader = _Loader({"by_path": {}, "by_tag": {"writing": "core.default"}})

        result = audit_governor_coverage(app)

        assert {"method": "GET", "path": "/api/v1/writing/docs"} in result["protected_routes"]

    @pytest.mark.unit
    def test_unmapped_api_route_is_protected_by_default(self):
        app = _MockApp(routes=[_MockRoute("/api/v1/anything", {"GET"})], user_middleware=[_Middleware(_RGSimpleMiddleware)], state=type("State", (), {
            "rg_policy_loader": _Loader({"by_path": {}, "by_tag": {}})
        })())
        assert audit_governor_coverage(app)["protected_count"] == 1

    @pytest.mark.unit
    def test_route_mapped_to_undefined_policy_is_reported(self):
        class _Partial(_Loader):
            def get_policy(self, pid):
                return None

        app = _MockApp(routes=[_MockRoute("/api/v1/x", {"GET"})], user_middleware=[_Middleware(_RGSimpleMiddleware)], state=type("State", (), {
            "rg_policy_loader": _Partial({"by_path": {"/api/v1/x": "typo.policy"}})
        })())
        result = audit_governor_coverage(app)
        assert result["unprotected_routes"][0]["reason"] == "policy_undefined"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_coverage_audit.py`
Expected: the three new tests FAIL.

- [ ] **Step 3: Implement**

In `audit_governor_coverage`, replace the classification loop with:

```python
    from .policy_eval import DEFAULT_POLICY_ID
    from .policy_resolver import get_policy_resolver

    resolver = get_policy_resolver(app)
    loader = getattr(getattr(app, "state", None), "rg_policy_loader", None)

    def _defined(policy_id: str) -> bool:
        if policy_id == DEFAULT_POLICY_ID:
            return True  # a built-in default always backs it
        try:
            return bool(loader.get_policy(policy_id)) if loader is not None else False
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False

    for r in routes:
        policy_id = resolver.resolve(r["path"], r["method"]) if resolver else None
        if any(r["path"].startswith(p) for p in prefixes):
            unprotected.append(_public_route(r, reason="excluded_prefix"))
        elif not middleware_installed:
            unprotected.append(_public_route(r, reason="rg_middleware_missing"))
        elif policy_id is None:
            unprotected.append(_public_route(r, reason="route_unmapped"))
        elif not _defined(policy_id):
            unprotected.append(_public_route(r, reason="policy_undefined"))
        else:
            protected.append(_public_route(r))
```

Delete `_get_route_map`, `_route_is_mapped` and `_glob_matches` if nothing else references them; check with `git grep`.

In `startup_resource_governor._audit_route_map_coverage`, replace the body inside its `try:` (from `snap = rg_loader.get_snapshot()` through the `if missing:` warning) with the code below, and delete `_route_map_matches` once unused.

```python
        from tldw_Server_API.app.core.Resource_Governance.policy_eval import DEFAULT_POLICY_ID
        from tldw_Server_API.app.core.Resource_Governance.policy_resolver import get_policy_resolver
        from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes

        resolver = get_policy_resolver(app)
        if resolver is None:
            return
        missing: list[tuple[str, str]] = []
        seen_paths: set[str] = set()
        for route in iter_served_routes(getattr(app, "routes", [])):
            path = route.path
            if not path.startswith("/api/") or path in seen_paths or not route.methods:
                continue
            seen_paths.add(path)
            policy_id = resolver.resolve(path, sorted(route.methods)[0])
            if policy_id and policy_id != DEFAULT_POLICY_ID and not rg_loader.get_policy(policy_id):
                missing.append((path, policy_id))
        if missing:
            sample = ", ".join(f"{p} (policy={pid})" for p, pid in missing[:10])
            logger.warning(f"RG route_map missing coverage for {len(missing)} routes; sample: {sample}")
```

Audit/resolver agreement is structural: both audits call `resolve`. The check over the real app's served routes is the Task 11 lint, which runs against the fully enabled app.

- [ ] **Step 4: Run the audit tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_coverage_audit.py tldw_Server_API/tests/Resource_Governance/test_coverage_endpoint_limit.py tldw_Server_API/tests/Resource_Governance/test_auth_route_map_coverage.py`
Expected: all pass. Existing tests whose `/api/` routes were "unmapped" are now protected by `default`; update those expectations and cite spec §1.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/coverage_audit.py tldw_Server_API/app/services/startup_resource_governor.py tldw_Server_API/tests/Resource_Governance/test_coverage_audit.py
git commit -m "fix(rg): coverage and startup audits report what the resolver enforces

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 11: Route-map lint

**Files:**
- Modify: `Helper_Scripts/ci/route_auth_ratchet.py` (rename `_load_app` to `load_app`; keep `_load_app = load_app` for the existing caller)
- Create: `Helper_Scripts/ci/rg_route_map_lint.py`
- Create: `Helper_Scripts/ci/rg_route_map_lint_allowlist.txt`
- Modify: `tldw_Server_API/Config_Files/resource_governor_policies.yaml` (fixes the lint finds)
- Test: `tldw_Server_API/tests/lint/test_rg_route_map_lint.py`

**Interfaces:** produces `lint(route_map: Mapping, served: list[ServedRoute], allow: set[str]) -> list[str]` in `rg_route_map_lint.py`.

- [ ] **Step 1: Write the failing tests**

```python
"""route_map entries must be reachable and used; the real app is checked in a subprocess."""

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Helper_Scripts.ci.rg_route_map_lint import lint  # noqa: E402
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes  # noqa: E402

pytestmark = pytest.mark.unit


def _served():
    router = APIRouter(tags=["notes"])

    @router.get("/notes/{nid}")
    def note(nid: str) -> None:
        return None

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return list(iter_served_routes(app.routes))


def test_dead_pattern_is_reported():
    assert lint({"by_path": {"/api/v1/nowhere*": "p"}, "by_tag": {}}, _served(), set()) == ["by_path /api/v1/nowhere* matches no served route"]


def test_shadowed_pattern_is_reported():
    rm = {"by_path": {"/api/v1/*": "a", "/api/v1/notes*": "b"}, "by_tag": {}}
    assert lint(rm, _served(), set()) == ["by_path /api/v1/notes* is shadowed by earlier patterns"]


def test_unused_tag_is_reported():
    assert lint({"by_path": {}, "by_tag": {"ghost": "p", "notes": "q"}}, _served(), set()) == ["by_tag ghost is used by no served route"]


def test_allowlisted_problem_is_suppressed():
    problem = "by_tag ghost is used by no served route"
    assert lint({"by_path": {}, "by_tag": {"ghost": "p"}}, _served(), {problem}) == []


def test_shipped_route_map_is_clean():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "Helper_Scripts" / "ci" / "rg_route_map_lint.py")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/lint/test_rg_route_map_lint.py`
Expected: FAIL at import (the module does not exist).

- [ ] **Step 3: Implement the lint**

`Helper_Scripts/ci/rg_route_map_lint.py`:

```python
"""Fail CI when a Resource Governor route_map entry is dead, shadowed or unused.

Builds the fully enabled app the same way the route-auth ratchet does. The
allowlist holds intentional exceptions, one problem string per line, with a
``#`` comment giving the reason.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = Path(__file__).resolve().parent / "rg_route_map_lint_allowlist.txt"


def lint(route_map: Mapping[str, Any], served: list[Any], allow: set[str]) -> list[str]:
    from tldw_Server_API.app.core.Resource_Governance.policy_resolver import compile_route_glob

    paths = [r.path for r in served if getattr(r, "path", None) and getattr(r, "methods", None)]
    patterns = [(str(p), compile_route_glob(str(p))) for p in (route_map.get("by_path") or {})]
    problems: list[str] = []
    for i, (raw, rx) in enumerate(patterns):
        hits = [p for p in paths if rx.match(p)]
        if not hits:
            problems.append(f"by_path {raw} matches no served route")
        elif all(any(earlier.match(p) for _r, earlier in patterns[:i]) for p in hits):
            problems.append(f"by_path {raw} is shadowed by earlier patterns")
    used_tags = {t for r in served for t in (getattr(r, "tags", None) or ())}
    for tag in route_map.get("by_tag") or {}:
        if str(tag) not in used_tags:
            problems.append(f"by_tag {tag} is used by no served route")
    return [p for p in problems if p not in allow]


def _allowlist() -> set[str]:
    if not ALLOWLIST.exists():
        return set()
    out = set()
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            out.add(entry)
    return out


def main() -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from Helper_Scripts.ci.route_auth_ratchet import load_app
    from tldw_Server_API.app.core.Resource_Governance.policy_loader import default_policy_loader
    from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
    import asyncio

    app = load_app()
    loader = default_policy_loader()
    asyncio.run(loader.load_once())
    problems = lint(loader.get_snapshot().route_map or {}, list(iter_served_routes(app.routes)), _allowlist())
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
```

In `route_auth_ratchet.py`, rename `def _load_app` to `def load_app`, add `_load_app = load_app` below it, and update its internal caller.

- [ ] **Step 4: Run the lint against the shipped YAML and fix what it finds**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/ci/rg_route_map_lint.py`
Expected: a list of problems. Fix each one in the YAML:
- `/api/v1/auth*` shadows `/api/v1/authnz*`. Replace it with two entries, `/api/v1/auth` and `/api/v1/auth/*`, both mapping to `authnz.default`, in the same position.
- `/api/v1/vector-stores*` matches nothing. Change it to `/api/v1/vector_stores*`.
- For each unused tag (the survey found `auth`, `subscriptions-deprecated`, `audio-ws`, `chat-dictionaries`, `chat-documents`, `mcp.ingestion`, `embeddings_server`, `character_chat`, `web_scraping`), delete it from `by_tag`.
- For any remaining problem that is intentional (for example a pattern for a router only mounted behind a flag the lint's app does not enable), add the exact problem string to `rg_route_map_lint_allowlist.txt` with a `#` reason.

Re-run until it exits 0.

- [ ] **Step 5: Run the tests and the ratchet test** (the rename must not break it)

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/lint/test_rg_route_map_lint.py tldw_Server_API/tests/lint/test_route_auth_ratchet.py tldw_Server_API/tests/Resource_Governance/test_policy_reference_consistency.py`
Expected: all pass.

- [ ] **Step 6: Wire the lint into CI**

In `.github/workflows/backend-required.yml`, add the lint test file to the "Enforce tenant isolation ratchets" step's pytest list, next to `test_route_auth_ratchet.py`.

- [ ] **Step 7: Commit**

```bash
git add Helper_Scripts/ci/rg_route_map_lint.py Helper_Scripts/ci/rg_route_map_lint_allowlist.txt Helper_Scripts/ci/route_auth_ratchet.py tldw_Server_API/Config_Files/resource_governor_policies.yaml tldw_Server_API/tests/lint/test_rg_route_map_lint.py .github/workflows/backend-required.yml
git commit -m "ci(rg): route-map lint for dead, shadowed and unused entries; fix what it found

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 12: WebUI replay — normal use never hits a governor 429

**Files:**
- Create: `Helper_Scripts/rg_webui_fixture_from_access_log.py`
- Create: `tldw_Server_API/tests/Resource_Governance/fixtures/webui_session_requests.json`
- Test: `tldw_Server_API/tests/Resource_Governance/test_webui_replay_no_429.py`

**Interfaces:** the fixture format is `[[seconds_from_start: float, method: str, path: str], ...]`, containing only `/api/` paths.

- [ ] **Step 1: Write the converter**

```python
"""Convert a timestamped uvicorn access log into the WebUI replay fixture.

Usage: rg_webui_fixture_from_access_log.py ACCESS_LOG OUT_JSON
Expects lines like: 2026-09-30 10:00:01,234 127.0.0.1:5000 "GET /api/v1/notes/?x=1 HTTP/1.1" 200
"""

import json
import re
import sys
from datetime import datetime

_LINE = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) \S+ "(\w+) (\S+) HTTP/[\d.]+" \d{3}')


def convert(lines):
    rows, t0 = [], None
    for line in lines:
        m = _LINE.match(line)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
        path = m.group(3).split("?", 1)[0]
        if not path.startswith("/api/"):
            continue
        t0 = ts if t0 is None else t0
        rows.append([round(ts - t0, 3), m.group(2), path])
    return rows


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as fh:
        data = convert(fh)
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print(f"{len(data)} requests over {data[-1][0] if data else 0:.1f}s")
```

- [ ] **Step 2: Record the fixture**

The goal is a real WebUI session. Governance is off while recording so nothing is throttled.

1. Write `/private/tmp/.../scratchpad/uvicorn_log.json` (the scratchpad path, not the repo):

```json
{"version": 1, "disable_existing_loggers": false,
 "formatters": {"access": {"()": "uvicorn.logging.AccessFormatter", "fmt": "%(asctime)s %(client_addr)s \"%(request_line)s\" %(status_code)s"}},
 "handlers": {"access": {"formatter": "access", "class": "logging.FileHandler", "filename": "rg_access.log"}},
 "loggers": {"uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": false}}}
```

2. Start the backend from the repo root:

```bash
RG_ENABLED=false /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m uvicorn tldw_Server_API.app.main:app --port 8000 --log-config <scratchpad>/uvicorn_log.json
```

3. In `apps/tldw-frontend`:

```bash
TLDW_SERVER_URL=http://127.0.0.1:8000 bunx playwright test e2e/smoke/all-pages.spec.ts
```

4. Convert:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/rg_webui_fixture_from_access_log.py rg_access.log tldw_Server_API/tests/Resource_Governance/fixtures/webui_session_requests.json
```

   Expected: a line like `1234 requests over 312.4s`.

If the Playwright smoke cannot run in this environment, **stop and report it**. Do not hand-write a fixture.

- [ ] **Step 3: Write the replay test**

```python
"""One person's WebUI session plus an extension stream never hits a governor 429.

The shipped policy YAML resolves by_path and default against a catch-all app.
Tag-only routes therefore resolve to `default` here (same or looser limits than
their tag policy), which the spec's goal 1 tolerates.
"""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware
from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit, pytest.mark.asyncio]

FIXTURE = Path(__file__).parent / "fixtures" / "webui_session_requests.json"
YAML = Path(__file__).resolve().parents[2] / "Config_Files" / "resource_governor_policies.yaml"


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


async def test_webui_session_and_extension_stream_see_no_429(monkeypatch):
    monkeypatch.delenv("RG_POLICY_PATH", raising=False)
    loader = PolicyLoader(YAML, PolicyReloadConfig(enabled=False))
    await loader.load_once()
    clock = Clock()
    app = FastAPI()

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def anything(path: str) -> dict:
        return {}

    app.add_middleware(RGSimpleMiddleware)
    app.state.rg_policy_loader = loader
    app.state.rg_governor = MemoryResourceGovernor(policy_loader=loader, time_source=clock)

    async def one_user(self, request):
        return "user:1"

    monkeypatch.setattr(RGSimpleMiddleware, "_principal_entity", one_user)
    client = TestClient(app)

    session = json.loads(FIXTURE.read_text(encoding="utf-8"))
    duration = session[-1][0] if session else 0.0
    timeline = [(t, m, p) for t, m, p in session]
    timeline += [(t + duration, m, p) for t, m, p in session]  # second pass: steady state
    timeline += [(float(s), "GET", "/api/v1/notes/") for s in range(int(2 * duration) + 1)]  # extension, 1/s
    timeline.sort(key=lambda row: row[0])

    denied = []
    for t, method, path in timeline:
        clock.t = 1000.0 + t
        if client.request(method, path).status_code == 429:
            denied.append((t, method, path))
    assert denied == [], f"{len(denied)} governor 429s, first: {denied[:5]}"
```

- [ ] **Step 4: Run it**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_webui_replay_no_429.py`
Expected: PASS.
- If it fails, read the first denials: the policy ID is in the 429 body. Raise that policy's `rpm`/`burst` in the YAML just enough to pass, and record each change and its reason in the commit message. The spec allows raising the starting numbers.
- Also update `test_policy_yaml_safety_net.py` if a number changes.

- [ ] **Step 5: Commit**

```bash
git add Helper_Scripts/rg_webui_fixture_from_access_log.py tldw_Server_API/tests/Resource_Governance/fixtures/webui_session_requests.json tldw_Server_API/tests/Resource_Governance/test_webui_replay_no_429.py
git commit -m "test(rg): replay a recorded WebUI session plus an extension stream with zero 429s

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 13: Ship PR B and close TASK-13395

- [ ] **Step 1: Broad suites** (as in Task 6 Step 1), plus `tldw_Server_API/tests/lint` and `tldw_Server_API/tests/Utils`.

- [ ] **Step 2: Ledger**

```bash
backlog task edit 13395 --check-ac 1 --check-ac 2 --check-ac 3 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 -s Done \
  --append-notes "Owner decision 2026-09-29: option (a). The resolver enforces by_tag pre-routing through a served-route index (path first, innermost tag, then default). Both audits call the same resolver. The request-level test (test_middleware_tag_enforcement.py) covers a tag-only route through nested includes. Spec: Docs/Design/2026-09-29-rg-ingress-safety-net-design.md."
backlog task edit <PARENT> --check-ac 2 --append-notes "PR B (coverage) merged: resolver, principal identity, audits, route-map lint, WebUI replay."
```

- [ ] **Step 3: Rebase, push, open the PR** (title `feat(rg): enforce tag policies, charge principals, honest audits (spec 1, PR B)`). Handle Qodo and merge as in Task 6.

---

## PR C — One switch, config hygiene, ADR and docs

Branch: `fix/rg-single-switch` from `origin/dev` after PR B merges.

### Task 14: One switch turns governance off everywhere

**Files:**
- Modify: `tldw_Server_API/app/main.py` (~2070: `rg_enabled(False)` → `rg_enabled(True)`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py` (~3945: same change)
- Modify: `tldw_Server_API/app/core/AuthNZ/rg_startup_guard.py` (~79: same change)
- Modify: `tldw_Server_API/app/services/startup_resource_governor.py` (`init_resource_governor`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/auth.py` (`_get_auth_endpoint_rg_governor`)
- Modify: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py` (`_rg_enabled_flag`, ~1978)
- Modify: `tldw_Server_API/app/core/Evaluations/config_validator.py` (~185)
- Test: `tldw_Server_API/tests/Resource_Governance/test_rg_single_switch.py`

**Interfaces:** none new; `config.rg_enabled(True)` becomes the only switch.

- [ ] **Step 1: Write the failing tests**

```python
"""RG_ENABLED=false means no enforcement path touches a governor."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]

APP = Path(__file__).resolve().parents[2] / "app"


def test_no_direct_env_reads_or_divergent_defaults():
    offenders = []
    for path in APP.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.name != "config.py" and re.search(r"""getenv\(\s*["']RG_ENABLED["']""", text):
            offenders.append(f"{path.relative_to(APP)}: reads RG_ENABLED directly")
        if "rg_enabled(False)" in text:
            offenders.append(f"{path.relative_to(APP)}: rg_enabled(False)")
    assert offenders == []


@pytest.mark.asyncio
async def test_disabled_governance_attaches_no_governor(monkeypatch):
    monkeypatch.setenv("RG_ENABLED", "false")
    from tldw_Server_API.app.services.startup_resource_governor import init_resource_governor

    app = SimpleNamespace(state=SimpleNamespace())
    await init_resource_governor(app)
    try:
        assert getattr(app.state, "rg_governor", None) is None
        assert getattr(app.state, "rg_policy_loader", None) is not None  # diag still works
    finally:
        await app.state.rg_policy_loader.shutdown()  # stop the auto-reload task


@pytest.mark.asyncio
async def test_disabled_governance_skips_auth_reservations(monkeypatch):
    monkeypatch.setenv("RG_ENABLED", "false")
    from tldw_Server_API.app.api.v1.endpoints import auth as auth_ep

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()), state=SimpleNamespace())
    assert await auth_ep._get_auth_endpoint_rg_governor(request) is None


def test_diag_lazy_governor_is_not_attached_when_disabled(monkeypatch):
    monkeypatch.setenv("RG_ENABLED", "false")
    from tldw_Server_API.app.api.v1.endpoints import resource_governor as rg_ep

    app = SimpleNamespace(state=SimpleNamespace(rg_policy_loader=SimpleNamespace(get_policy=lambda _pid: None)))
    monkeypatch.setattr(rg_ep, "_get_app", lambda: app)
    assert rg_ep._get_or_init_governor() is not None  # diagnostics still work
    assert getattr(app.state, "rg_governor", None) is None  # but enforcement stays off
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_rg_single_switch.py`
Expected: all four FAIL.

- [ ] **Step 3: Implement**

- `init_resource_governor`: after `app.state.rg_policy_store = _store_mode`, wrap the governor construction:

```python
        from tldw_Server_API.app.core.config import rg_enabled as _rg_enabled

        if not _rg_enabled(True):
            app.state.rg_governor = None
            logger.info("Resource Governor disabled (RG_ENABLED / [ResourceGovernor] enabled); policies loaded for diagnostics only")
        else:
            ...existing backend construction...
```

  Skip `_warn_if_enabled_without_governor(app)` when disabled.
- `_get_auth_endpoint_rg_governor`: first statement `if not rg_enabled(True): return None`, importing `rg_enabled` from `tldw_Server_API.app.core.config`.
- `resource_governor._get_or_init_governor` (the diag endpoints): keep the lazily built governor local when governance is off. Today it attaches it to `app.state`, so one diag call would switch enforcement back on for every call site that reads `app.state.rg_governor` (media ingest, workflows). Guard only the attach:

```python
            if loader is not None:
                gov = MemoryResourceGovernor(policy_loader=loader)
                if rg_enabled(True):
                    app.state.rg_governor = gov  # diagnostics only when disabled
```

- `auth_deps._rg_enabled_flag`: `return bool(rg_enabled(True))`.
- `config_validator`: replace `if not _is_truthy(os.getenv("RG_ENABLED")):` with `if not rg_enabled(True):`, importing from `tldw_Server_API.app.core.config`.
- The three `rg_enabled(False)` sites become `rg_enabled(True)`.

- [ ] **Step 4: Run the tests and the RG, startup and auth suites**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -n 4 tldw_Server_API/tests/Resource_Governance tldw_Server_API/tests/AuthNZ_Unit tldw_Server_API/tests/Services/test_main_router_contract.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app tldw_Server_API/tests/Resource_Governance/test_rg_single_switch.py
git commit -m "fix(rg): one switch; governance off means no governor anywhere

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 15: Config hygiene

**Files:**
- Modify: `tldw_Server_API/Config_Files/resource_governor_policies.yaml` (delete `hot_reload`, `metadata`, `defaults`, `route_map.by_route`)
- Modify: `tldw_Server_API/app/core/Resource_Governance/policy_loader.py` (`load_once`, file branch)
- Test: `tldw_Server_API/tests/Resource_Governance/test_policy_loader_unknown_keys.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
import yaml
from loguru import logger

from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit, pytest.mark.asyncio]


async def test_loader_warns_on_keys_it_ignores(tmp_path):
    path = tmp_path / "rg.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "policies": {}, "templates": {}, "schema_version": 1, "bogus": 1, "route_map": {"by_path": {}, "by_route": {}}}), encoding="utf-8")
    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        await PolicyLoader(path, PolicyReloadConfig(enabled=False)).load_once()
    finally:
        logger.remove(sink)
    text = "\n".join(messages)
    assert "bogus" in text and "by_route" in text
    assert "templates" not in text and "schema_version" not in text
```

- [ ] **Step 2: Run it to verify it fails.** Expected: FAIL (no warning).

- [ ] **Step 3: Implement.** In `load_once`'s file branch, after `data = yaml.safe_load(f) or {}`:

```python
            _consumed = {"version", "policies", "tenant", "route_map", "templates", "schema_version"}
            for key in sorted(set(data) - _consumed):
                logger.warning("RG policy file key {!r} is ignored by the loader", key)
            for key in sorted(set(dict(data.get("route_map") or {})) - {"by_path", "by_tag"}):
                logger.warning("RG route_map key {!r} is ignored by the loader", key)
```

Then delete the four ignored sections from the shipped YAML, so the shipped file produces no warnings. Run `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Resource_Governance` and confirm no test reads those sections.

- [ ] **Step 4: Run the test and the policy loader suites.** Expected: all pass.

- [ ] **Step 5: Commit** (`chore(rg): drop ignored policy sections; loader warns on keys it ignores`).

### Task 16: ADR-056 and docs

**Files:**
- Create: `Docs/ADR/056-resource-governor-safety-net.md`
- Modify: `Docs/ADR/README.md` (index row)
- Modify: `Docs/Operations/Env_Vars.md` (the RG section, ~316-381)
- Modify: `tldw_Server_API/app/core/Resource_Governance/README.md`
- Create: `Docs/Operations/Rate_Limits_Troubleshooting.md`
- Modify: `Docs/Design/2026-09-29-rg-ingress-safety-net-design.md` (Status: Implemented; link PRs)

- [ ] **Step 1: Write ADR-056.** Follow the structure of `Docs/ADR/044-cookie-session-governance-owner-preflight.md` (Status, Context, Decision, Consequences). It records:
  - the safety-net posture: per-entity buckets, and `global` only for the three email-sending request policies;
  - the resolution order: path, then innermost tag, then `default`;
  - the identity rule: the validated principal, with invalid credentials charged to the IP;
  - no permanent 429s: the `default` and built-in fallbacks and scope fallback;
  - the single switch;
  - the DB policy store upgrade step: re-seed with `python -m tldw_Server_API.app.core.Resource_Governance.seed_db_from_yaml`, or edit policies, to pick up the new limits.

  It amends ADR-018 (resolution order, default policy) and ADR-044 (preflight generalised; invalid credentials fall through).

- [ ] **Step 1b: Update the "new endpoint" checklist in `Resource_Governance/README.md`.** A new router is governed by `default` automatically. To give it a dedicated policy, add a router tag mapped in `by_tag` (preferred) or a `by_path` entry. A `by_path` entry for a sensitive route always wins over a tag. The route-map lint (`Helper_Scripts/ci/rg_route_map_lint.py`) fails CI on dead, shadowed or unused entries.

- [ ] **Step 2: Correct `Env_Vars.md`'s RG section to match `config.py`:**
  - `RG_ENABLED`: defaults on, via `config.txt [ResourceGovernor] enabled = true`.
  - `RG_POLICY_STORE`: `file` | `db`, default `file`.
  - `RG_POLICY_RELOAD_INTERVAL_SEC`: default 10.
  - `RG_REDIS_FAIL_MODE`: default `fallback_memory`.
  - Remove `RG_TEST_BYPASS` from the RG README.

- [ ] **Step 3: Write `Rate_Limits_Troubleshooting.md`,** "Why am I getting 429s?". It has a table mapping each response shape to the layer that produced it and the switch that tunes it:

  | Response | Layer | How to tune |
  |---|---|---|
  | `{"error":"rate_limited","policy_id":…}` | RG ingress | Edit the named policy in `resource_governor_policies.yaml` (hot-reloads), or `RG_ENABLED=false` |
  | `… (ResourceGovernor policy=chat.default); retry_after=…` | chat token bucket | `chat.default.tokens` |
  | `Rate limit exceeded for resource: X` | rbac_rate_limit | `privilege_catalog.yaml` rate classes |
  | `Rate limit exceeded for endpoint: /path` | auth_deps fallback | `AUTH_DEPS_FALLBACK_RATE_LIMIT` (never 0) |
  | 402 `limit_exceeded` | billing plan limits (multi-user) | `LIMIT_ENFORCEMENT_ENABLED` (Spec 2 revisits defaults) |
  | 402 `budget_exceeded` | virtual-key budget | `LLM_BUDGET_ENFORCE` |
  | `Media ingestion concurrency limit reached.` | media jobs | `media.default.jobs` |
  | `Transcription quota exceeded (daily minutes)` | audio tier | `AUDIO_TIER_LIMITS_JSON` |
  | `Provider rate limit exceeded` | the upstream LLM or TTS provider | the provider's own plan |

  Add a note that memory-backend buckets are per worker process: effective limits scale with `UVICORN_WORKERS`, and `RG_BACKEND=redis` gives exact limits.

- [ ] **Step 4: Update the ADR index and the spec status.** Commit (`docs(rg): ADR-056 safety net; correct RG env docs; 429 troubleshooting page`).

### Task 17: Ship PR C

- [ ] **Step 1: Broad suites,** as in Task 6.
- [ ] **Step 2: Ledger.**

```bash
backlog task edit <PARENT> --check-ac 3 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 -s Done --append-notes "PR C merged: single switch, config hygiene, ADR-056, docs. Spec 2 (usage-quota posture) is next."
```

- [ ] **Step 3: Rebase, push, open the PR** (title `fix(rg): one switch, config hygiene, ADR-056 (spec 1, PR C)`). Handle Qodo and merge as in Task 6.
