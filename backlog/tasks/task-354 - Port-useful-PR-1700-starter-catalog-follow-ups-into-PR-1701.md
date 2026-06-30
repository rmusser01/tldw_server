---
id: TASK-354
title: Port useful PR 1700 starter catalog follow-ups into PR 1701
status: Done
assignee: []
created_date: '2026-05-15 01:59'
updated_date: '2026-05-15 02:01'
labels:
  - persona
  - persona-visual
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1700'
  - 'https://github.com/rmusser01/tldw_server/pull/1701'
  - 'https://github.com/rmusser01/tldw_server/issues/1694'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up on PR #1701 by selectively porting non-duplicative useful pieces from overlapping PR #1700: design-spec documentation for the starter catalog placement and focused regressions that protect fixture manifest isolation and duplicate bundled asset-key validation. Keep PR #1701's separate PersonaVisualStarterCatalogService boundary and /copy route semantics unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec documents the bundled starter catalog using PR #1701's actual list detail and copy routes
- [x] #2 Starter catalog tests prove returned detail manifests are isolated from bundled fixture mutation
- [x] #3 Starter catalog tests prove duplicate bundled asset keys are rejected with a stable starter catalog error
- [x] #4 Focused Persona Visual starter catalog tests pass after the changes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ported the useful non-overlapping PR #1700 pieces into PR #1701: design-spec starter catalog placement using #1701's /copy route semantics, manifest preview isolation regression coverage, and duplicate fixture asset-key regression coverage. Verification: focused starter catalog tests passed (9 passed), focused Persona Visual starter/API slice passed (59 passed), py_compile passed, black --check passed, git diff --check passed, and Bandit with pytest assert check excluded reported 0 findings; the raw Bandit run only reported B101 pytest assert usage in the touched test file.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Ported useful PR #1700 follow-ups into PR #1701 without changing the production service boundary. Added design-spec documentation for the starter catalog routes and service semantics, plus regression tests for manifest preview isolation and duplicate fixture asset-key rejection.
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
