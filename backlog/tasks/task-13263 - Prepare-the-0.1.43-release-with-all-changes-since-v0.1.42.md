---
id: TASK-13263
title: Prepare the 0.1.43 release with all changes since v0.1.42
status: In Progress
assignee: []
created_date: '2026-09-20 19:56'
updated_date: '2026-09-20 20:12'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare a reviewed release candidate based on v0.1.42 and frozen dev d72b1d2850ea947b6d12cac19f6b95867b68a580. Preserve 0.1.42 release fixes, reconcile released main into dev, inventory every new commit and merged PR, update release metadata and protected source records, and open a draft release PR. Track outstanding 0.1.42 publication verification in TASK-13013.3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The candidate includes v0.1.42 and frozen dev as ancestors, with reviewed conflict resolutions.
- [ ] #2 Changelog and release notes cover all post-0.1.42 changes, with an exhaustive commit inventory.
- [ ] #3 Version metadata, documentation and protected-source records are consistent and verified.
- [ ] #4 A draft PR and release plan record checks, publication state and remaining human decisions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Release plan: Docs/superpowers/plans/2026-09-20-release-0.1.43-plan.md. Frozen delta: 616 commits and 19 first-parent merges after v0.1.42. Five conflicts resolved retaining transport security and dev persistence. Independent static review found no concrete merge regression. Notes regression exposed obsolete authority mock, removed; 26 Notes tests pass and 107 other merge regressions pass. Historical license manifest check updated to pin immutable published bytes.

Strict docs build passes. Sync draft PR2971 opened. Existing plugin isolation test failed before fixing four AuthNZ registrations, then 6 passed; 2559 affected tests collect without error. Prompt/editor regressions:1125 passed. Bandit has no new findings; two pre-existing B105 synthetic-secret fixture findings exactly match HEAD. Source manifest and candidate records will be regenerated after frontend type repairs.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
