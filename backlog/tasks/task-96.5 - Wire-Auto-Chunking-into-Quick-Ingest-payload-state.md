---
id: TASK-96.5
title: Wire Auto Chunking into Quick Ingest payload state
status: Done
assignee:
  - '@codex'
created_date: '2026-05-06 17:31'
updated_date: '2026-05-06 17:41'
labels:
  - frontend
  - chunking
  - quick-ingest
  - auto-chunking
dependencies:
  - TASK-96.4
documentation:
  - Docs/superpowers/specs/2026-05-06-auto-chunking-design.md
  - Docs/superpowers/plans/2026-05-06-auto-chunking-implementation-plan.md
parent_task_id: TASK-96
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the frontend state and payload plumbing slice from the approved Auto Chunking plan. Quick Ingest should default enabled chunking to Auto, send Auto fields without stale Manual settings, preserve Manual as the advanced escape hatch, and keep web/article Quick Ingest payloads in parity with media payloads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Quick Ingest option types include chunking_mode, auto_chunking_goal, and auto_chunking_use_llm.
- [x] #2 Default Quick Ingest state uses Auto when chunking is enabled.
- [x] #3 Media payload construction sends Auto fields for Auto and Manual fields only for Manual.
- [x] #4 processWebScrape JSON payload sends the same Auto fields and omits stale Manual fields in Auto mode.
- [x] #5 Fallback settings schemas expose the Auto fields.
- [x] #6 Focused frontend service/state tests cover Auto, Manual, stale-field omission, and web payload parity.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Task 4 frontend payload/state slice after completing backend wiring TASK-96.4. Following TDD: add failing quick-ingest payload/state tests first, then update types, defaults, payload builder, background parity, and fallback schema.

Implemented Quick Ingest Auto/Manual payload state plumbing in apps/packages/ui. Added shared quick-ingest chunking payload helper, wired direct/background payload builders, defaulted presets and persisted common options to Auto, updated fallback schemas, and covered Auto/Manual/stale-field/web parity tests.

Verification: bun run test -- src/services/__tests__/quick-ingest-batch.test.ts src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx --maxWorkers=1 --no-file-parallelism passed with 43 tests. bun run verify:openapi passed. git diff --check passed. UI package-wide tsc still has unrelated existing failures; filtered touched-file diagnostics had no matches after the local normalization fix. Bandit not applicable for this TypeScript-only slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wired Quick Ingest state and payload construction for Auto Chunking. Auto mode now sends chunking_mode and auto goal fields without stale manual/template settings, Manual mode preserves advanced/template chunking fields, disabled chunking sends no chunking controls, and web scraping JSON payloads stay in parity. Added focused tests and fallback schema entries.
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
