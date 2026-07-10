---
id: TASK-12946
title: Address remaining frontend CodeQL alerts for dev
status: In Progress
labels:
- security
- codeql
- frontend
priority: High
references:
- https://github.com/rmusser01/tldw_server/security/code-scanning
- https://github.com/rmusser01/tldw_server/pull/2696
documentation:
- Docs/superpowers/specs/2026-07-10-remaining-frontend-codeql-alerts-design.md
modified_files:
- Docs/superpowers/specs/2026-07-10-remaining-frontend-codeql-alerts-design.md
- backlog/tasks/task-12946 - Address-remaining-frontend-CodeQL-alerts-for-dev.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the 12 JavaScript/TypeScript CodeQL alerts #2251-#2262 that remain unpatched on origin/dev after merged PR #2696 remediated the other 149 current alerts. Apply minimal root-cause fixes, add focused regression coverage, verify frontend typechecking and CodeQL-relevant behavior, and open a PR against dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All remaining CodeQL alert classes #2251-#2262 are addressed in source on a branch based on origin/dev.
- [ ] #2 Focused regression tests cover unsafe HTML/URLs, OPML-free group filtering, provider inference, and logging behavior.
- [ ] #3 Frontend typechecking and targeted tests pass; skipped checks are documented.
- [ ] #4 A pull request is opened against dev with alert mapping and verification results.
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
