---
id: TASK-9923
title: Fix Audiobook core review findings
status: Done
assignee: []
created_date: '2026-06-23 18:31'
updated_date: '2026-06-23 18:31'
labels:
  - audiobooks
  - code-review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address current-code review findings in tldw_Server_API/app/core/Audiobooks: duplicate chapter IDs, non-monotonic alignment anchors, unsafe subtitle cue text, loose tag marker validation, and overly broad subtitle timing-line detection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Duplicate explicit and generated chapter IDs cannot collide silently
- [x] #2 Multiple alignment anchors cannot produce a backwards subtitle timeline
- [x] #3 Generated SRT, VTT, and ASS subtitle text is escaped or sanitized per format
- [x] #4 Speed and timestamp tags reject invalid values with parser warnings
- [x] #5 Subtitle parsing only strips real timing lines
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See IMPLEMENTATION_PLAN_audiobook_core_review_fixes.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation completed inline after red tests confirmed the reviewed failures. Verification: direct core regression script passed; python -m compileall passed for touched runtime/test files; Bandit on touched runtime scope reported 0 findings. Pytest targeted commands were attempted, but the local harness stalled in setup/cleanup on this Python 3.14 environment after confirming red failures and two later passing tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed audiobook core review findings: chapter IDs are de-duplicated with warnings, speed/timestamp tags reject invalid values, alignment anchors cannot move later cues before adjusted prior cues, generated subtitle text is sanitized per SRT/VTT/ASS, and subtitle parsing only strips real timing lines.
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
