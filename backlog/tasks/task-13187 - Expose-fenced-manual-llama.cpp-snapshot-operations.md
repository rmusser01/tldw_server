---
id: TASK-13187
title: Expose fenced manual llama.cpp snapshot operations
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 02:19'
updated_date: '2026-09-05 04:49'
labels: []
dependencies:
  - TASK-13186
documentation:
  - Docs/Design/2026-09-04-llamacpp-manual-slot-snapshots.md
  - Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md
  - Docs/superpowers/plans/2026-09-04-llamacpp-manual-snapshots.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow administrators to save and restore managed runtime caches without duplicate dispatch or stale-process mutation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All six snapshot routes enforce admin access, strict schemas and rate limits; no raw prompts, paths or binaries are exposed.
- [x] #2 Generation and owner fences, durable receipts and expiring signed request tokens prevent duplicate or stale dispatch.
- [x] #3 Timeouts after dispatch quarantine the launch; stop recovery works and Pause/Resume remain manual process actions.
- [x] #4 Targeted API, supervisor, shutdown and crash-injection tests pass with checked egress.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Task2 of Docs/superpowers/plans/2026-09-04-llamacpp-manual-snapshots.md. ADR required yes; ADR043 covers runtime ownership and single-dispatch operations. TDD and targeted security/lifecycle review required.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented supervisor-owned manual snapshot operations and six admin routes under ADR-043: launch generations, reservations, private working paths, strict compatibility, signed request tokens, durable receipts, bounded single-dispatch native calls, unknown-outcome recovery and shutdown ownership. Task and final whole-branch reviews are clean after fixing admission/shutdown synchronization, verified staging cleanup, source-derived slot parsing, non-loopback native exposure, cross-profile cleanup ledger races and optional POSIX locking imports. Final verification: 213 targeted backend tests passed with 6 baseline warnings; Ruff/format/compileall/Bandit/diff checks passed. The failed-termination regression proves the real ownership fence stays held; Windows absence-of-fcntl is simulated, not a real-host test. Production build allowlist remains empty pending TASK13188 live evidence. ADR043 linked; no broader runtime support claimed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reviewed backend snapshot implementation complete. All final review findings addressed at c6593deac1; 213 targeted tests pass. Native persistence requires numeric loopback and supported private storage. Live build support remains intentionally unavailable until TASK13188 evidence.
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
