---
id: TASK-13162
title: Expose fenced manual llama.cpp snapshot operations
status: To Do
assignee: []
created_date: '2026-09-05 02:19'
updated_date: '2026-09-05 02:27'
labels: []
dependencies:
  - TASK-13161
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
- [ ] #1 All six snapshot routes enforce admin access, strict schemas and rate limits; no raw prompts, paths or binaries are exposed.
- [ ] #2 Generation and owner fences, durable receipts and expiring signed request tokens prevent duplicate or stale dispatch.
- [ ] #3 Timeouts after dispatch quarantine the launch; stop recovery works and Pause/Resume remain manual process actions.
- [ ] #4 Targeted API, supervisor, shutdown and crash-injection tests pass with checked egress.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
