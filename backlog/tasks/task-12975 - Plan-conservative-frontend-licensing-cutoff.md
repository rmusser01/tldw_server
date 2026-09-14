---
id: TASK-12975
title: Plan conservative frontend licensing cutoff
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-20 19:08'
labels:
  - licensing
  - frontend
  - planning
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-07-19-frontend-source-available-licensing-design.md
  - >-
    Docs/superpowers/plans/2026-07-20-conservative-frontend-licensing-cutoff-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the execution-ready pre-counsel implementation plan for the approved Perimeter and Countdown licensing cutoff, covering legal corpus, public history, package/product metadata, OpenAPI metadata, contribution freeze, API image isolation, and protected publishing suspension.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The plan maps every immediate-cutoff requirement to exact files and executable verification commands.
- [x] #2 Tasks are reviewer-sized, ordered, test-driven where behavior changes, and contain no implementation placeholders.
- [x] #3 Protected publication and unlicensed protected/API-contract contributions fail closed until later grants exist.
- [x] #4 Public history, third-party terms, GPL backend implementation, and Apache OpenAPI contract boundaries are preserved.
- [x] #5 Post-counsel custom grants and full protected artifact release hardening are explicitly deferred.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased planning artifacts onto latest origin/dev, re-verified public refs on 2026-07-20, and corrected colliding task IDs before execution. Bandit is not applicable to Markdown-only planning work.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Produced and reviewed a six-task implementation plan for the conservative frontend licensing cutoff, with exact files, tests, fail-closed controls, artifact isolation, historical-boundary records, and deferred counsel-reviewed custom terms.
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
