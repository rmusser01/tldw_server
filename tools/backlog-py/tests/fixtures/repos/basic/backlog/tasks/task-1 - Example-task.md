---
id: TASK-1
title: Example task
status: In Progress
assignee:
  - codex
created_date: '2026-05-10 10:00'
labels:
  - parser
custom_field: preserve-me
nested_unknown:
  source: fixture
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement a fixture that exercises parser preservation behavior.
This paragraph must remain untouched by a no-op render.
<!-- SECTION:DESCRIPTION:END -->

Unowned body content before acceptance criteria must be preserved.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preserve completed acceptance criteria raw line
- [ ] #2 Preserve incomplete acceptance criteria raw line
- [ ] Plain checklist item without an id
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Keep frontmatter order stable.
- Keep unknown body text stable.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
No final summary yet.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Tests written
- [ ] #2 Verification recorded
<!-- DOD:END -->

Trailing unowned body content must also round trip.
