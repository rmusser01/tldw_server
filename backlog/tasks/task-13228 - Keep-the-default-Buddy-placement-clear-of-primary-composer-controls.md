---
id: TASK-13228
title: Keep the default Buddy placement clear of primary composer controls
status: To Do
created_date: 2026-09-09 05:16
priority: high
references:
- TASK-13227
documentation:
- Docs/Reviews/2026-09-09-buddy-v1-qualification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13227 live WebUI qualification at1090×990 found the initial Buddy position covering Console Send: the drag handle intercepted the pointer until Shift+ArrowUp moved the Buddy. Define a safe default or docking behavior while retaining intentional user positioning and keyboard access.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fresh/default and reset Buddy placement leave Console send, stop, and composer input usable at supported desktop/compact sizes.
- [ ] #2 User-selected placement remains editable by pointer and keyboard and stays in viewport across resizing/navigation.
- [ ] #3 Real pointer hit-testing verifies the primary controls alongside the visible Buddy; the chosen placement policy is documented before implementation.
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
