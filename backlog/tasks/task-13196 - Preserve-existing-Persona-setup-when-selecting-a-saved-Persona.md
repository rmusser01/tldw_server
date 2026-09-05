---
id: TASK-13196
title: Preserve existing Persona setup when selecting a saved Persona
status: In Progress
assignee: []
created_date: '2026-09-05 23:15'
updated_date: '2026-09-05 23:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real Migu browser UAT completed setup through a provider-backed live reply, but reloading and selecting the same Persona overwrote completion and forced setup again. Preserve saved progress when choosing an existing Persona.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selecting a completed Persona preserves its completed setup and opens its normal workspace without overwriting setup progress.
- [x] #2 Selecting a Persona with unfinished setup resumes its saved stage; only a new or not-started setup is initialized.
- [x] #3 Targeted regression and real-browser reload/reselection evidence verify preserved voice defaults and setup status.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Routine selection/persistence bug fix preserving the existing setup model and API.
1. Reproduce the unconditional setup reset through the real selection handler.
2. Read the selected Persona before choosing whether to resume or initialize setup; preserve completed and in-progress state.
3. Add route regressions and verify real reload/reselection; run focused lint/type checks and record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Selecting an existing Persona now reads its saved profile before changing setup. Completed setup and later in-progress stages are applied unchanged; failed reads leave the selection and saved state untouched. Completing the initial Persona choice still advances to Voice while retaining an existing run ID and completed steps. Added route regressions for completed, in-progress, and failed-read selections. Validation: 90 targeted sidepanel route tests passed; scoped TypeScript has zero owned diagnostics (27 dependency diagnostics remain); ESLint has zero errors and 78 warnings in the existing hook/test files. Bandit skipped because only TypeScript files changed. ADR not required: routine persistence bug fix. Real-browser reload/reselection evidence remains with root; task stays In Progress.

Real browser verification: completed setup with the DeepSeek reply Migu setup preserved., reloaded /persona, selected saved Migu UAT, and reached the normal Live Session workspace. STT en/tiny.en, TTS tldw/af_heart, auto-resume off and manual commit persisted. Final rebased verification and PR publication remain pending.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
