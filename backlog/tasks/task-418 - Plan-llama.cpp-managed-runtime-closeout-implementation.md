---
id: TASK-418
title: Plan llama.cpp managed runtime closeout implementation
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-17 04:53'
labels:
  - llamacpp
  - planning
  - webui
  - local-llm
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md
  - >-
    Docs/superpowers/plans/2026-05-17-llamacpp-managed-runtime-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a consolidation implementation plan for the remaining llama.cpp managed runtime roadmap work without starting code implementation. This was tracked standalone because origin/dev contained a duplicate TASK-397 ID collision at the time.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is added under Docs/superpowers/plans and references the approved managed runtime design spec.
- [x] #2 Plan starts with the required writing-plans header and includes exact files, tests, commands, and commit checkpoints.
- [x] #3 Plan decomposes remaining work into supervision reconciliation, validation hardening, V1/API compatibility, provider metadata, Admin UI, and rollout verification stages.
- [x] #4 Plan preserves local import/register first and defers remote downloads/catalogs.
- [x] #5 Plan includes verification requirements including focused pytest, frontend tests where relevant, git diff checks, and Bandit for touched Python code.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Docs/superpowers/plans/2026-05-17-llamacpp-managed-runtime-implementation-plan.md as a consolidation and closeout plan grounded in current origin/dev. The plan references the landed Stage 1, Asset Inventory V2, mmproj/model-family, saved profile editor, and acquisition/import plans; it decomposes remaining runtime work into supervision reconciliation, validation hardening, V1/API compatibility, provider metadata, Admin UI, and rollout verification. Bandit not run because this task changes docs/backlog only. Standalone tracking was intentional because origin/dev had a duplicate TASK-397 ID collision at the time. PR #1815 review follow-up replaced a fake key-like TLDW_E2E_API_KEY assignment with a note to supply the value from local test configuration.

Rollout closeout follow-through completed under TASK-418.15: source docs now cover managed runtime profile behavior, default-profile compatibility, local import/register, mmproj pairing, bounded restart/autostart behavior, and deferred acquisition boundaries. `Docs/Published` files are generated and intentionally left untouched. The new admin E2E smoke plus focused frontend/backend/Bandit/diff verification passed. The Playwright smoke requires `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000` when starting the advanced-mode frontend dev server.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the managed runtime implementation closeout plan with exact files, tests, commands, and commit checkpoints. The plan preserves local import/register first, defers remote downloads/catalogs, avoids rebuilding already-landed runtime/profile/API/UI slices, and no longer includes a fake key-like E2E API key value.
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
