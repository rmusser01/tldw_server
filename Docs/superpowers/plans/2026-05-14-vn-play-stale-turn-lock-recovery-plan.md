# VN Play Stale Turn Lock Recovery Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent VN Play sessions from staying permanently blocked when a process dies after acquiring `active_turn_request_id` but before clearing the lock.

**Architecture:** Keep the lock source of truth in `vn_play_sessions.active_turn_request_id`, but use the existing `vn_play_turn_requests.locked_until` and terminal statuses to distinguish fresh in-flight work from stale abandoned work. Recovery must be explicit, guarded, and non-mutating with respect to scene state.

---

## Stage 1: Repository Recovery Primitive

**Goal:** Add a small repository method that can abandon an expired active turn and clear only the matching session lock.

**Success Criteria:**
- [x] Method inspects the active turn request for a session and owner.
- [x] Method only abandons statuses `pending` and `model_calling`.
- [x] Method only recovers when `locked_until` exists and is expired.
- [x] Method clears `active_turn_request_id` only when it still points at the recovered request.
- [x] Method leaves completed, failed, already-abandoned, missing, and fresh locked requests unchanged.

**Tests:**
- [x] Add focused `VNPlayRepository` tests in `tldw_Server_API/tests/VN_Play/test_vn_play_db.py`.

## Stage 2: Service Guard

**Goal:** Invoke stale-lock recovery before turn and restore-action concurrency checks.

**Success Criteria:**
- [x] `submit_turn(...)` attempts recovery before checking `session.active_turn_request_id`.
- [x] `retry_last_turn(...)` attempts recovery before checking `session.active_turn_request_id`.
- [x] Restore/checkpoint action paths that reject active turns also see recovered session state before rejecting.
- [x] Fresh active turns still produce `turn_in_progress`.

**Tests:**
- [x] Add service-level tests in `tldw_Server_API/tests/VN_Play/test_vn_play_turns.py`.

## Stage 3: Verification and Task Closeout

**Goal:** Prove the hardening behavior and record evidence.

**Verification Commands:**
- [x] `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q` (`82 passed, 8 warnings`)
- [x] `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_action_requests.py tldw_Server_API/tests/VN_Play/test_vn_play_scripted_generation_runtime.py -q` (`31 passed, 8 warnings`)
- [x] `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/DB_Management/VNPlay_DB.py tldw_Server_API/app/core/VN_Play/service.py tldw_Server_API/app/core/VN_Play/constants.py`
- [x] `git diff --check`
- [x] `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/DB_Management/VNPlay_DB.py tldw_Server_API/app/core/VN_Play/service.py tldw_Server_API/app/core/VN_Play/constants.py -f json -o /tmp/bandit_vn_play_stale_turn_lock.json` (`0 results`)

---

## Risk Review

- **False recovery while a model call is still running:** Require an expired `locked_until`; fresh or missing leases continue to block.
- **Scene-state drift:** Recovery only changes request status/error and clears the lock. It must not append events, advance `scene_version`, or rewrite scene state.
- **Race with completing worker:** Guard session update by the exact active request id. If another request owns the lock by the time recovery runs, leave it alone.
- **Terminal-state regression:** Never update completed, failed, parse-failed, cancelled, or already-abandoned turn requests.
