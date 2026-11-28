---
id: TASK-15655
title: Review every WebUI Quick Ingest state after the one-screen redesign
status: Done
labels:
- webui
- ux
dependencies:
- TASK-15652
references:
- Docs/Design/2026-11-25-quick-ingest-modal-state-review.md
- Docs/Design/assets/quick-ingest/modal-states-review.pdf
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review all reachable WebUI Quick Ingest screens and conditional states on desktop and phone. Document NN/g findings, proposed interaction changes and how to verify them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A state inventory distinguishes browser observations from source and test evidence.
- [x] #2 The design review prioritizes defects by user impact and includes concrete desktop and mobile remedies.
- [x] #3 The report records verification scenarios and an ADR assessment.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review the branch shipped for PR 3631. Inventory conditional WebUI surfaces, inspect desktop and phone fixtures independently, and document ranked interaction changes with acceptance checks. ADR required: no. The proposed review uses the existing Quick Ingest session, job and navigation boundaries.
Two independent assessments of PR #3631 plus paired desktop and phone interaction checks. The 14-row state inventory identifies browser and source coverage. Verified all five PDF pages, screenshot dimensions, pinned source links and local document links. Bandit found no Python code in the asset scope. No new application code. Live service and assistive-technology outcomes were outside this review.
Published the review, illustrated proposal, and separate full-resolution desktop and phone evidence in https://github.com/rmusser01/tldw_server/pull/3633. The implementation and original screenshots are in https://github.com/rmusser01/tldw_server/pull/3631.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reviewed the entire WebUI Quick Ingest modal and its background/return paths. Recorded five prioritized findings, a 21/40 heuristic baseline, concrete desktop and phone designs and seven acceptance walkthroughs. The browser evidence uses controlled service responses.
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
