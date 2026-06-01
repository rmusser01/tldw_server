---
id: TASK-499
title: Implement Companion Home header and launcher shortcuts
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Companion Home shortcut design.

Spec: Docs/superpowers/specs/2026-06-01-companion-home-header-shortcut-design.md
Plan: Docs/superpowers/plans/2026-06-01-companion-home-header-shortcut-implementation-plan.md

Scope:
- Add a Companion Home header icon button immediately left of the signpost shortcut button, navigating to `/`.
- Add a Companion Home shortcut/listing to the page directory and legacy sheet modal.
- Preserve existing persisted header shortcut selections by forcing Companion Home into coerced selection IDs.
- Keep `/` active state exact so `/chat` does not mark Companion Home current.

Verification:
- Focused Vitest coverage for ChatHeader, HeaderShortcuts, header shortcut item hosting, persona defaults, and UI settings shortcut coercion.
- TypeScript check for the UI package.
- Bandit noted as not applicable to frontend-only touched scope unless Python files become touched.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
