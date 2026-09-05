---
id: TASK-13196
title: Preserve existing Persona setup when selecting a saved Persona
status: Done
assignee: []
created_date: '2026-09-05 23:15'
updated_date: '2026-09-05 23:27'
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
Saved-Persona selection now reads the target before applying setup. Completed and later in-progress setup resume unchanged; failed reads cause no write or selection switch. Initial Persona/archetype choice advances to Voice while preserving existing run identity and completed steps. Real browser reload/reselection preserved completed setup and en/tiny.en, tldw/af_heart, manual commit and auto-resume off. Post-rebase route/voice regression scope: 198 passed. ESLint: zero errors, existing warnings only; scoped TypeScript: zero owned diagnostics, 27 dependency diagnostics remain. Bandit not applicable to this TypeScript-only task. Evidence and incident lesson updated; ADR not required for this persistence bug fix.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed saved-Persona setup survives reload and reselection. Regression and real browser checks passed; unrelated voice acceptance remains under TASK13195.
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
