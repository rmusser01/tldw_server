---
id: TASK-15659
title: Make Quick Ingest batch metadata editable on phones
status: In Progress
labels:
- webui
- ux
dependencies:
- TASK-15658
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the responsive source editor from the full-modal review (#3633) on the WebUI from #3638.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Title, speaker, track, tags and inclusion for each source are labeled and reachable at 320 and 390 CSS pixels without horizontal scrolling.
- [x] #2 The desktop editor keeps its columns and each source retains its values and selection across layout changes and submission.
- [ ] #3 Publish a PR against dev with desktop and phone screenshots and regression coverage.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new architecture decision. The accepted review calls for a responsive presentation of the same source overrides. Draft persistence and API contracts stay the same.
Approved design in #3633: retain desktop comparison columns and use labeled per-source editors with visible inclusion on phones. Verify via Media → Add media and the submitted collection payload.
Implemented one keyed editor per source. The phone disclosure and inclusion label share a 44px header; each expanded field wraps within the modal. The desktop columns reuse those inputs. Red-to-green component regression and 497 passing scoped tests. Chromium at 320, 390 and 1280px verified geometry, inclusion, keyboard focus across reflow and the collection item payload. The two screenshot widths are 390 and 1280 pixels.
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
