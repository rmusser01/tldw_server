---
id: TASK-45.44.2.2
title: Use design-system registry labels for Knowledge QA source health status
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-16 04:48'
labels: []
dependencies: []
parent_task_id: TASK-45.44.2
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix unbaselined canonical-state-label guard findings in Knowledge QA source health by routing Ready and Unavailable labels through the design-system state registry helper.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 getSourceHealthStatusLabel uses design-system registry labels for ready and unavailable states.
- [x] #2 Focused sourceHealth tests prove registry labels are used for ready/searchable, ready/not searchable, and unavailable statuses.
- [x] #3 Repo-level design-system product-state guard no longer reports sourceHealth.ts as blocked.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Routed ready/searchable and unavailable source-health labels through design-system registry label constants so canonical labels keep defensive non-empty fallbacks.

Verification: RED sourceHealth registry-label test failed against hardcoded labels; GREEN sourceHealth test passed 4/4; product-state guard unit passed 52/52; verify:design-system-state exited 0 and no longer reports sourceHealth.ts as blocked.

Known verification note: full UI TypeScript check still fails on existing repo-wide type debt outside touched files. Bandit is not applicable to this TypeScript-only frontend slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Knowledge QA source-health canonical labels to use the design-system registry and added focused regression coverage for ready and unavailable status labels.
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
