# Backend Performance Remediation — Batch 1: Per-Request Fixed Costs Implementation Plan (2026-10-06)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan stage-by-stage. Steps use checkbox (`- [ ]`) syntax for tracking.

**Backlog task:** TASK-13513 · **Index:** [2026-10-06-perf-remediation-coordination-index.md](2026-10-06-perf-remediation-coordination-index.md)

**Goal:** Remove the fixed per-request overhead taxed on every API call: uncached 210k-iteration PBKDF2 on the event loop, ChaCha dependency probe/ensure spam, per-tool RBAC queries in MCP `tools/list`, and the uncached full config rebuild.

**Architecture:** All four fixes are local caching/lifecycle changes — no API or schema changes. Each stage is independently shippable and ordered by leverage.

**Tech Stack:** stdlib (`asyncio.to_thread`, `hashlib`, `time.monotonic`, `functools`), existing AuthNZ/MCP/config modules.

## Global constraints

- Backend only: never modify `apps/**`.
- Activate venv first: `source .venv/bin/activate` (repo root).
- One commit per stage; message references TASK-13513.
- TDD: write/extend the named test first, watch it fail, implement, re-run.
- Bandit on touched paths: `python -m bandit -r tldw_Server_API/app/core/AuthNZ tldw_Server_API/app/api/v1/API_Deps tldw_Server_API/app/core/MCP_unified/auth tldw_Server_API/app/core/config.py -f json -o /tmp/bandit_perf_b1.json`.
- Security rule for Stage 1: cache **successful verifications only**, short TTL, never cache a failed compare.
- Line numbers below are from the 2026-10-06 review; re-locate by symbol name if drifted.

---

## Stage 1: API-key PBKDF2 verification cache + thread offload

**Goal:** Stop running ~50-150ms of synchronous KDF on the event loop for every API-key request.

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/api_key_manager.py` (`_verify_new_format_key`, ~line 520-535)
- Modify: `tldw_Server_API/app/core/AuthNZ/api_key_crypto.py` (no logic change expected; export what the cache needs)
- Test: `tldw_Server_API/tests/AuthNZ/test_api_key_verify_cache.py` (new)

**Change:**
1. Module-level cache in `api_key_manager.py`:
   - Key: `(key_identifier, stored_hash)` → value: `key_info` dict; entries tagged with `time.monotonic()` timestamp.
   - TTL 30s, max 256 entries (evict oldest on insert; simple dict + FIFO deque is fine — no new deps).
   - Only insert after `verify_kdf_hash(...)` returns True (cache only the successful `_verify_new_format_key` outcome).
   - Hit path still re-reads the repo row (1 cheap indexed SELECT) but skips the KDF; if `stored_hash` differs from the cache key, treat as miss.
2. Wrap the `verify_kdf_hash` call in `await asyncio.to_thread(...)` (pure CPU; `validate_api_key` is async).
3. Invalidate the cache in every writer that rotates/revokes API keys (grep for `key_hash` writes in the API-key manager/repository).

**Success criteria:** second verification of the same key within TTL performs 0 KDF derivations; event loop unblocked; AuthNZ suite green.

**Tests:**
- [ ] `test_second_verify_within_ttl_skips_kdf` — monkeypatch-count `hashlib.pbkdf2_hmac`; two validates → count == 1.
- [ ] `test_failed_verify_never_cached` — wrong key twice → count == 2, result None both times.
- [ ] `test_rotated_hash_invalidates_cache` — change stored hash between validates → no stale success.
- [ ] Existing: `python -m pytest tldw_Server_API/tests/AuthNZ -v -m "not external_api"` green.

**Status:** Not Started

---

## Stage 2: ChaCha dependency per-request overhead

**Goal:** End the per-request default-character task, health-probe thread hop, and INFO log in `get_chacha_db_for_user`.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py` (`get_chacha_db_for_user` ~802-841; `_ensure_default_character_async` ~494-527; `_is_instance_healthy` call site ~633; log line ~828)
- Test: `tldw_Server_API/tests/API_Deps/test_chacha_dep_request_overhead.py` (new or extend existing ChaCha deps tests)

**Change:**
1. `default_character_ensured`: module-level `set[str]` of user ids (or attribute on the cached instance) — after one successful `_ensure_default_character_owned`, skip spawning the task for that user; clear on DB instance rebuild.
2. Health probe TTL: per-instance `_last_healthy_at: float`; skip `probe_chacha_connection` when younger than 60s; on probe failure, drop the instance as today.
3. Demote `logger.info("<<<<< ACTUAL get_chacha_db_for_user CALLED >>>>>")` to `logger.debug`.

**Success criteria:** N sequential `get_chacha_db_for_user` calls for the same warm user perform 0 ensure-tasks and 0 probe queries after the first; behavior on unhealthy instance unchanged.

**Tests:**
- [ ] `test_ensure_default_character_runs_once_per_user` — call dependency 3×, count `_ensure_default_character_owned` invocations (mock) == 1.
- [ ] `test_health_probe_skipped_within_ttl` — 2 calls within TTL → 1 probe (mock counter).
- [ ] `test_unhealthy_instance_still_evicted` — failing probe evicts cache entry (existing behavior preserved).

**Status:** Not Started

---

## Stage 3: MCP `tools/list` batched RBAC resolution

**Goal:** Replace 1-4 sequential RBAC queries per listed tool (~300 tools per `tools/list`) with ≤3 queries per request.

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/auth/authnz_rbac.py` (`has_tool_permission`-family, ~107-172)
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py` (tools/list loop, ~2004)
- Test: `tldw_Server_API/tests/MCP_unified/test_tools_list_rbac_batching.py` (new or extend)

**Change:**
1. New `resolve_permission_context(user_id, pool) -> PermissionContext` dataclass: `is_admin: bool`, `granted: dict[str, bool]` (explicit user overrides incl. wildcards), `role_tool_perms: set[str]` — filled by at most 3 queries (admin check, user overrides, role join), evaluated once per `tools/list` request.
2. `_has_tool_permission` gains an overload taking the pre-resolved context and doing pure in-memory checks — port the existing SQL logic's decision order exactly (admin → explicit grant/deny → wildcard → role-derived).
3. The tools/list loop passes the shared context. Keep the uncached per-call path for other callers, unchanged.
4. Optional (only if trivially safe): per-user 5s TTL on `resolve_permission_context` results — skip if cache invalidation on role change is not provable.

**Success criteria:** `tools/list` with a 300-tool catalog issues ≤3 AuthNZ queries total; authorization decisions identical for admin / granted / denied / wildcard cases.

**Tests:**
- [ ] `test_tools_list_issues_at_most_three_rbac_queries` — mocked pool counts `fetchone/fetchall` calls.
- [ ] Parametrized decision-parity test: for {admin, explicit allow, explicit deny, wildcard allow, role-derived allow, no-permission} × {read tool, write tool}, old path and new path return the same boolean (call both implementations directly).

**Status:** Not Started

---

## Stage 4: Cache `load_and_log_configs`

**Goal:** Stop rebuilding the ~1000-line config dict per chat call / provider-adapter fallback.

**Files:**
- Modify: `tldw_Server_API/app/core/config.py` (`load_and_log_configs`, ~3743; `refresh_config_cache` — locate existing)
- Test: `tldw_Server_API/tests/test_config_caching.py` (new or extend existing config tests)

**Change:**
1. Rename the body to `_build_and_log_configs()`; `load_and_log_configs()` returns a module-level cached snapshot built from the already-cached `load_comprehensive_config()` parser.
2. Invalidate in `refresh_config_cache()` (extend it to clear this snapshot too).
3. **Mutation audit first:** grep all callers of `load_and_log_configs` for writes into the returned dict. If any mutate, return `copy.deepcopy(snapshot)` instead of the shared object and record that decision in the task notes (deepcopy is still far cheaper than the 800+-line rebuild).

**Success criteria:** two calls return the same object (or equal copies) with one build; `refresh_config_cache()` forces a rebuild; no caller observes stale keys after config file edits + refresh.

**Tests:**
- [ ] `test_load_and_log_configs_cached` — call twice, assert `is` same object (or equal), build counter (mock `_build_and_log_configs`) == 1.
- [ ] `test_refresh_config_cache_invalidates` — refresh → next call rebuilds.
- [ ] `test_callers_do_not_mutate_shared_snapshot` — a guard test calling the known hot callers (chat orchestrator + one adapter fallback) with a `MappingProxyType`-wrapped snapshot; if callers do mutate, document the deepcopy branch in this test instead.

**Status:** Not Started

---

## Batch verification

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ tldw_Server_API/tests/API_Deps tldw_Server_API/tests/MCP_unified tldw_Server_API/tests/test_config_caching.py -v -m "not external_api and not local_llm_service"
python -m tldw_Server_API.tests.perf.benchmarks.bench_api_key_verify   # expect ms_per_successful_verify ~0 on the cache-hit path
python -m bandit -r tldw_Server_API/app/core/AuthNZ tldw_Server_API/app/api/v1/API_Deps tldw_Server_API/app/core/MCP_unified/auth tldw_Server_API/app/core/config.py -f json -o /tmp/bandit_perf_b1.json
```

Re-record deltas in `Docs/Reviews/PERF_BASELINE_2026_10.md`, then update TASK-13513 (notes, touched files, verification, final summary, DOD).
