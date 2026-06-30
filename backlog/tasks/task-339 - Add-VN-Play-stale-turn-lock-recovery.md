---
id: TASK-339
title: Add VN Play stale turn lock recovery
status: Done
assignee: []
created_date: '2026-05-14 07:03'
labels:
  - vn-play
  - backend
  - runtime
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-05-14-vn-play-stale-turn-lock-recovery-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden VN Play runtime sessions against permanently stuck active_turn_request_id locks after worker/process interruption. The current backend has turn idempotency, scene-version checks, and per-session active turn markers, but the turn lock does not use the existing lease fields and can block a session indefinitely if the process dies after acquiring the lock. Implement a narrow backend recovery path that abandons expired active turn requests and clears the session lock before accepting a new request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stale active turn locks can be abandoned and cleared without advancing scene state or replaying partial model output.
- [x] #2 Fresh active turns still reject concurrent turns and restore actions with turn_in_progress.
- [x] #3 Completed failed and already-abandoned turn requests are never regressed by stale-lock recovery.
- [x] #4 Focused repository and service tests cover stale active_turn_request_id recovery retry behavior and fresh-lock preservation.
- [x] #5 Verification and Bandit results are recorded before completion.
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Implemented stale active turn lock recovery for VN Play runtime sessions. `try_acquire_turn_lock` now stores turn leases atomically with session lock acquisition, and the service attempts guarded stale-lock recovery before normal turn retry and restore/checkpoint active-turn checks. Recovery only abandons `pending` or `model_calling` turn requests with expired `locked_until`, clears the session lock only when it still references the recovered request, and does not append events or advance scene state.

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q` -> 82 passed, 8 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_action_requests.py tldw_Server_API/tests/VN_Play/test_vn_play_scripted_generation_runtime.py -q` -> 31 passed, 8 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/DB_Management/VNPlay_DB.py tldw_Server_API/app/core/VN_Play/service.py tldw_Server_API/app/core/VN_Play/constants.py` -> passed.
- `git diff --check` -> passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/DB_Management/VNPlay_DB.py tldw_Server_API/app/core/VN_Play/service.py tldw_Server_API/app/core/VN_Play/constants.py -f json -o /tmp/bandit_vn_play_stale_turn_lock.json` -> 0 results.
<!-- SECTION:NOTES:END -->

## Final Summary
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added lease-backed stale active turn lock recovery for VN Play sessions. New repository logic abandons expired active turn requests and clears only the matching session lock, while service guards recover stale locks before submit retry and restore/checkpoint concurrency checks. Fresh active locks still block, terminal turn requests are preserved, and focused DB/service/generation regression checks passed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
