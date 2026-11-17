---
id: TASK-13677
title: Qualify public character import with its current accessible label
status: Done
labels:
- characters
- ux
- webui
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The public Import control is named Import character. The new browser fixture used /^Import$/ and never clicked it; Promise.all then reported a chooser timeout. Correct the fixture to the actual accessible name and qualify JSON/PNG through the public chooser. No product trigger defect has been established.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Public Import click opens the installed native file chooser in the real browser.
- [x] #2 JSON and PNG selected through that public chooser reach existing preview and confirmed import without validation bypass.
- [x] #3 Relevant mounted tests, real card journey, source checks and independent review qualify the root repair.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Two live format controls reproduce the same chooser timeout; no product source changed yet. Stop/reassess: inspect installed Upload DOM and native click propagation before a third run. Existing field-level input upload tests are narrower. ADR assessment: existing shared Upload interaction, no new architecture rule.
Reassessment evidence: actual first-failure page snapshot line90 names Import character; the fixture regex excluded it. Exact diagnostic CLI38670 was interrupted after PID/cwd/command validation; no public click was reached and no product source changed. Do not credit the two180sec timeouts or interrupted probe as product defects or passes. Correct existing fixture locator and keep bounded chooser timeout.
Corrected public Import character opens the actual native chooser in both formats. Further fixture issues are distinct: soft-deleting the exported seed preserves its unique name, so reimport collides; rename only that owned seed after exporting and before soft delete. PNG preview currently describes filename/format rather than embedded card metadata, so assert its actual supported preview and canonical imported card after confirmed backend decoding. No product source defect is established by these failures; do not bypass preview or server validation.
The existing public control is Import character; no product chooser defect was confirmed. Correct public filechooser, preview and confirmed JSON/PNG import, greeting, two protected turns and five-row cold reopen all execute. Scene phase failures are separately retained under13679, not credited as a passing whole journey. PR #3229; final bounded source review by two independent reviewers. TypeScript/JSON/Markdown-only unit: Bandit N/A locally; fresh full PR67-production-Python scan has0 introduced findings and0 scan errors vs exact merge-base. Installed extension/physical AT/touch/native-toolbar zoom/participants remain unrun. Overall merge and current Scene stability remain separate gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The existing public control is Import character; no product chooser defect was confirmed. Correct public filechooser, preview and confirmed JSON/PNG import, greeting, two protected turns and five-row cold reopen all execute. Scene phase failures are separately retained under13679, not credited as a passing whole journey.
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
