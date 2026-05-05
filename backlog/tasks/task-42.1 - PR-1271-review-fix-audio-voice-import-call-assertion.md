---
id: TASK-42.1
title: 'PR 1271 review fix: audio voice import call assertion'
status: Done
assignee: []
created_date: '2026-05-05 00:15'
updated_date: '2026-05-05 00:23'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1271'
  - 'https://github.com/rmusser01/tldw_server/pull/1271#discussion_r3185257968'
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
parent_task_id: TASK-42
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the actionable Qodo review thread on PR #1271 by making the new audio/voice router laziness test verify deferred imports without depending on router spec ordering. This is a follow-up to TASK-42 and should keep the production router change untouched unless verification shows otherwise.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The audio/voice laziness test no longer asserts a specific import call order for behavior that is order-insensitive.
- [x] #2 The test still verifies module import deferral before router resolution and exact lazy router attribute resolution counts after resolution.
- [x] #3 Focused router-group contract verification passes after the review fix.
- [x] #4 The PR review thread is replied to or resolved with the verification summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed the review-fix assertion to compare the set of imported modules, preserving the pre-resolution empty import assertion and exact post-resolution attribute access counts.

Verification: focused audio/voice test passed; full router group contracts passed; main router contracts passed; OpenAPI contracts passed; Bandit on test file passed with pytest assert rule B101 skipped; git diff hygiene passed.

Replied to Qodo review thread with fix commit and verification summary, then resolved thread `PRRT_kwDOL1aGf85_gbK5`.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the Qodo PR #1271 maintainability finding by removing order-sensitive import call list equality from the audio/voice laziness test. The test now verifies selected modules were imported by resolution time without depending on spec ordering or duplicate import counts, while retaining exact lazy attribute resolution checks. Replied to and resolved the Qodo review thread after pushing the fix.
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
