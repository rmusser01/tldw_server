---
id: TASK-237
title: Add scripted VN play runtime
status: Done
assignee: []
created_date: '2026-05-10 06:58'
updated_date: '2026-05-10 07:45'
labels:
  - vn
  - api
  - backend
  - play
  - runtime
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
  - 'https://github.com/rmusser01/tldw_server/issues/1486'
documentation:
  - Docs/superpowers/specs/2026-05-10-vn-platform-api-design.md
  - Docs/superpowers/plans/2026-05-10-vn-platform-api-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 5 of the VN platform API implementation plan. Scope: harden VN Play action request idempotency/recovery, add per-session save slots, extend setup options with published script readiness, and add scripted_story session/runtime endpoints backed by published VN script versions and immutable snapshots.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Action requests reject stale scene versions, replay completed keys, expose active/failed/abandoned states, and preserve existing freeform/story behavior.
- [x] #2 Per-session save slots support create/list/read/patch/delete/restore with checkpoint semantics and idempotency replay/conflict handling.
- [x] #3 Setup options expose backend-owned script readiness, policy warning/acknowledgement requirements, default profiles, and empty states.
- [x] #4 scripted_story sessions pin script version, manifest snapshot, policy snapshot, and generation profile snapshot while repeating authoritative policy evaluation at session creation.
- [x] #5 Script runtime endpoints advance scripts, accept visible choices, expose spoiler-safe state/debug owner state, persist model expansions and seeded random results, and keep replay deterministic.
- [x] #6 Focused VN_Play tests, git diff checks, compileall, and Bandit on touched production Python paths are recorded before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current VN_Play DB/service/API/state/action request patterns and script/version repository contracts. 2. Add failing tests for action request recovery, save slots, setup-options script readiness, and scripted_story runtime flow. 3. Extend VN Play persistence and schemas without regressing existing freeform/story behavior. 4. Implement backend-owned story start, setup-options script support, and script runtime endpoints. 5. Run focused VN_Play and affected VN/Scripts/contract tests plus compileall, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Implemented action request recovery for stale/active/failed/abandoned turn requests and same-key session-action concurrency handling.
- Added per-session save slot CRUD/restore APIs with checkpoint-backed restore and idempotent session-action replay.
- Extended setup-options for scripted_story published script versions, readiness, policy warnings, defaults, and empty states.
- Added scripted_story session creation with pinned script/version/manifest/policy/generation snapshots and authoritative policy acknowledgement checks.
- Added story start backend command, scripted advance/choice/state/debug/regenerate endpoints, spoiler-safe public script state, deterministic random results, persisted generation/regeneration events, scripted /turn rejection, no-progress advance guards, and scripted checkpoint/save-slot cursor restore.
- Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q -> 147 passed, 5 warnings.
- Verification: compileall on touched VN Play/API/DB/schema paths exited 0.
- Verification: git diff --check exited 0.
- Verification: Bandit on touched production Python paths wrote /tmp/bandit_vn_play_task226.json with 0 results and 0 errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the scripted VN play runtime backend slice: action recovery hardening, save slots, scripted_story setup/session pinning, story start, scripted advance/choice/state/debug/regenerate APIs, spoiler-safe public state, deterministic random/generation replay, and scripted restore cursor handling. Verified with the full VN_Play pytest suite, compileall, git diff checks, and Bandit on touched production Python paths.
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
