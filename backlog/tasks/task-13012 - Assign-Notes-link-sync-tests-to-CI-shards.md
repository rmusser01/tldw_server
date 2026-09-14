---
id: TASK-13012
title: Assign Notes link-sync tests to CI shards
status: Done
assignee: []
created_date: '2026-08-12 01:08'
updated_date: '2026-08-12 01:11'
labels:
  - ci
  - notes
  - tests
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2773'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #2773 rebased onto dev after PR #2782, which added two Notes test files without assigning them to the full-suite shard matrix. Add those tests to the existing ChaChaNotesDB and Services shards so the shard coverage guard passes without ignoring coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Notes link migration test is assigned to every duplicated chacha-content-persona shard matrix.
- [x] #2 The Notes graph projection worker test is assigned to every duplicated platform-services-core shard matrix.
- [x] #3 The shard coverage guard passes with zero newly uncovered tests.
- [x] #4 The affected tests pass locally.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Discovered while rebasing PR #2773 onto dev 414e81a12a; local reproduction reports exactly the two tests introduced by PR #2782.

Verification: each new test path appears in all five duplicated shard matrices. Helper_Scripts/ci/check_shard_coverage.py passes with shards=773, test_files=4251, ignored=4, baseline=130, new_uncovered=0. The two affected test files pass: 29 passed. Bandit is not applicable because only CI YAML and task metadata changed. A broader CI policy run had five unrelated current-dev failures; the branch diff confirms this task changes only shard path entries.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Assigned the Notes link migration v58 test to every chacha-content-persona shard and the Notes graph projection worker test to every platform-services-core shard. This restores shard coverage without adding ignores or weakening the baseline. The shard guard and all 29 affected tests pass.
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
