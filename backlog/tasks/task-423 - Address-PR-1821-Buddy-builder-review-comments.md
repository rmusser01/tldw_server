---
id: TASK-423
title: Address PR 1821 Buddy builder review comments
status: Done
labels:
- review
- buddy
- persona
- frontend
- qa
references:
- https://github.com/rmusser01/tldw_server/pull/1821
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve still-actionable CodeRabbit and Gemini review comments on PR #1821 for the guided Buddy builder UX, then verify and update the PR branch. Keep fixes scoped to current review findings and the Buddy builder/frontend task records.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Still-actionable PR #1821 review comments are addressed or explicitly skipped with a current-code reason.
- [x] #2 Focused Buddy builder, PersonaGarden, and BuddyShell regression tests pass.
- [x] #3 Rendered or relevant automated verification is recorded when UI behavior changes.
- [x] #4 Git diff whitespace check passes, and Bandit is run only if Python code is touched or documented as not applicable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-05-17: Live PR review sweep found unresolved Gemini/CodeRabbit comments on Buddy draft activation CTA, duplicate setup/import surfaces, drag movement threshold/accumulation, i18n labels, import extension constants, source picker ARIA state, and tracker status alignment. Verifying each against current code before editing.
2026-05-17: Addressed the still-valid review findings in the guided Buddy builder UI. Removed duplicate setup/import mounts while preserving import/archive actions through the single guided import surface, wired the review CTA to activation controls, lowered/cumulated drag movement detection, localized hardcoded labels, reused archive extension constants, added source picker `aria-pressed`, restored starter tag/license metadata in the guided catalog, and marked the completed plan tracker Done. Bandit was not applicable because this slice only touches frontend TypeScript/JSON and Backlog task files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #1821 Buddy builder review comments and verified with focused component tests plus CDP screenshots of the new-user/default-copy/activation/Codex-import flow. No Python files were touched, so Bandit was not applicable.
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
