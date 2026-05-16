---
id: TASK-389
title: Risk Gate 3 prototype runtime jobs and preview lifecycle durability
status: Done
assignee:
  - Codex
created_date: '2026-05-15 19:41'
updated_date: '2026-05-16 00:43'
labels:
  - prototype-workspaces
  - risk-gate
  - backend
  - jobs
  - runtime
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1455'
  - 'https://github.com/rmusser01/tldw_server/issues/1440'
  - 'https://github.com/rmusser01/tldw_server/pull/1729'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md
  - >-
    Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-github-issue-bodies.md
  - Docs/API-related/Prototype_Workspaces_API.md
  - Docs/API-related/Prototype_Workspaces_Contract_Matrix.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Burn down prototype workspace runtime durability risk after Risk Gates 1 and 2 by moving the runtime bootstrap, preview lifecycle, snapshot save, and publish validation/promotion paths toward retry-safe Jobs-backed orchestration where appropriate. This tracks GitHub issue #1455 and should stay scoped to backend/core runtime durability; frontend/product work is limited to documenting the runtime status fields that later UI slices need.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prototype runtime jobs are idempotent and safe to retry for branch bootstrap, preview boot/replacement, snapshot save, and publish validation/promotion paths that this slice owns.
- [x] #2 Cancellation, timeout, cleanup, and retry semantics are documented and covered by focused backend tests.
- [x] #3 Preview handle revocation, renewal, active-handle replacement, and persistent lookup behavior are covered by backend tests.
- [x] #4 Failed publish validation never advances canonical or last-known-good prototype workspace pointers.
- [x] #5 Runtime bootstrap status, preview health, and promotion validation failure fields needed by later frontend/operator surfaces are implemented or explicitly documented for the contract freeze.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-05-15-prototype-workspace-risk-gate-3-runtime-durability.md

Stage 1: Job contract and failure semantics
- Add failing tests for retryable and terminal worker failures.
- Add typed prototype job exception/result helpers where needed.
- Document retry/cancel/timeout semantics.

Stage 2: Idempotent runtime state transitions
- Cover branch bootstrap, preview boot, and snapshot-save retry behavior.
- Add missing request identifiers/payload fields needed for idempotency.
- Preserve monotonic session snapshot state.

Stage 3: Preview lifecycle durability
- Cover persistent lookup after memory cache loss, renewal, replacement rollback, revocation, and inactive actor/session behavior.
- Keep persistent and in-memory preview state aligned.

Stage 4: Publish validation and promotion safety
- Prove failed validation, stale candidates, and post-preview persistence failures never advance canonical pointers.
- Preserve compensation behavior around preview grants and promotion request status.

Stage 5: Verification and PR closeout
- Run focused and full prototype tests, Bandit on touched backend paths, update TASK-389, and open PR linked to #1455.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-05-15: Created isolated worktree `.worktrees/prototype-risk-gate-3-runtime-durability` on branch codex/prototype-risk-gate-3-runtime-durability from origin/dev 2184b2168 after PR #1719 / Risk Gate 2 was verified merged and issue #1454 was closed.

2026-05-15: Baseline verification before implementation: `./.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces -q` passed with 90 passed, 5 warnings in 7.20s from the new worktree.

Implemented Risk Gate 3 runtime durability slice in worktree `.worktrees/prototype-risk-gate-3-runtime-durability`: added worker retry/terminal metadata, stable job result envelopes, retry-safe preview handle reuse for identical scope/snapshot/target/profile, retry-safe snapshot save reuse for explicit session-owned snapshot ids, and Risk Gate 3 runtime job contract docs.

Focused runtime/preview/promotion suite passed: `./.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_runtime_jobs.py tldw_Server_API/tests/PrototypeWorkspaces/test_preview_broker.py tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py -q` -> 30 passed, 5 warnings.

Final verification after runtime profile-version fix: `./.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces -q` -> 97 passed, 5 warnings.

Security/hygiene verification: `./.venv/bin/python -m bandit -r tldw_Server_API/app/core/Prototype_Workspaces/jobs.py tldw_Server_API/app/core/Prototype_Workspaces/jobs_worker.py tldw_Server_API/app/core/Prototype_Workspaces/models.py tldw_Server_API/app/core/Prototype_Workspaces/preview_broker.py tldw_Server_API/app/core/Prototype_Workspaces/service.py -f json -o /tmp/bandit_prototype_risk_gate_3.json` -> 0 findings; `git diff --check` -> clean.

Opened PR #1729: https://github.com/rmusser01/tldw_server/pull/1729

2026-05-15: Addressed PR #1729 review feedback: moved prototype job exceptions to app/core/exceptions.py, made worker programming/runtime-state failures terminal with stable failure codes, persisted failure_code through WorkerSDK.error_code, generated deterministic snapshot ids for Jobs-backed snapshot saves, propagated top-level preview runtime profile versions, made session preview retry reuse tolerant of archived workspaces, prevented metadata from overriding authoritative snapshot ids, removed new raw-SQL assertions from prototype tests, and cleaned machine-specific paths from the task log.

2026-05-15: Review-fix verification: `./.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces tldw_Server_API/tests/Jobs/test_worker_sdk.py -q` -> 110 passed, 5 warnings; `./.venv/bin/python -m pytest tldw_Server_API/tests/Jobs/test_worker_sdk.py -q` -> 7 passed, 5 warnings after WorkerSDK jitter hardening; Bandit touched-scope run to `/tmp/bandit_prototype_risk_gate_3_review_fixes.json` -> 0 findings; `git diff --check` -> clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Opened PR #1729 for Risk Gate 3 prototype runtime durability. The slice adds explicit prototype job retry/result metadata, typed terminal runtime failures, retry-safe preview handle reuse keyed by scope/snapshot/target/runtime profile version, retry-safe duplicate snapshot-save handling for session-owned snapshot ids, focused backend coverage, and Risk Gate 3 runtime contract documentation. Verification recorded: PrototypeWorkspaces tests passed with 97 passed and Bandit reported 0 findings on touched backend runtime modules.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused prototype runtime tests pass
- [x] #8 Bandit runs on touched backend paths
- [x] #9 GitHub issue #1455 is linked from the PR
<!-- DOD:END -->
