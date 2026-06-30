---
id: TASK-528.5
title: Harden /knowledge ready search source scope and saved profiles
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 03:05'
labels:
  - webui
  - extension
  - knowledge
  - ux
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-06-07-knowledge-ready-search-source-scope-plan.md
parent_task_id: TASK-528
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden the ready /knowledge search workflow for source categories, exact document/media and note selection, saved profiles, suggestions, keyboard shortcuts, preset naming, web fallback, and answer model/provider selection. Do not introduce flashcard or study-set behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ready state supports clear source category and exact document/note selection with accurate selected counts.
- [x] #2 Saved source/search profiles can be created, restored, deleted, and reached from compact/simple mode or an equivalent drawer.
- [x] #3 Search suggestions and keyboard shortcuts work without trapping keyboard users.
- [x] #4 Preset naming is consistent across toolbar, settings, history, and export.
- [x] #5 Web fallback and answer model/provider controls have clear loading and error states.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-06-07-knowledge-ready-search-source-scope-plan.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented ready-search source-scope hardening. Saved profiles persist and restore exact media/note filters; compact/simple toolbar shows exact selected counts; answer-model menu now exposes provider loading/error recovery while preserving server-default and manual model entry; user-facing preset naming is Fast/Balanced/Deep/Custom while keeping the internal thorough id. Verification: focused Knowledge QA Vitest suite passed (8 files, 93 tests), git diff --check passed, scope grep found no flashcard/deck/spaced repetition/study set terminology in touched Knowledge QA files, and Bandit is not applicable because no Python files were touched. Known skip: route fixture e2e was not rerun because the previously recorded Chromium launch/WXT build blockers remain from TASK-528.4/TASK-306.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Ready /knowledge search controls are hardened for source categories, exact document/note scope, saved profiles, compact-mode counts, preset naming, web fallback capability state, and answer model loading/error recovery. Targeted Knowledge QA unit/behavior verification passed. Route-level E2E remains blocked by the previously documented browser/runtime issue, not by this implementation.
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
