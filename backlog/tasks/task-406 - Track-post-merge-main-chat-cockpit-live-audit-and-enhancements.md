---
id: TASK-406
title: Track post-merge main chat cockpit live audit and enhancements
status: To Do
labels:
- chat
- webui
- ux
- audit
priority: Medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fresh post-merge tracker for the main WebUI /chat cockpit only. Use live browser/server evidence from origin/dev-derived work to identify enhancement follow-ups after the collapsible sidechannel slice, without drifting into extension sidepanel/sidebar or unrelated pages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Inspect the post-merge main `/chat` cockpit from this branch or a fresh `origin/dev` branch with real-server/browser evidence.
- [ ] Identify enhancement follow-ups for main `/chat` only, covering first-use comprehension, power-user flow, IA, rail controls, composition flow, accessibility, and responsive behavior.
- [ ] Separate quick wins from larger cockpit redesign or interaction-model opportunities.
- [ ] Record screenshots or browser notes where they materially support the findings.
- [ ] Do not include extension sidepanel/sidebar or unrelated WebUI pages.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started after the collapsible sidechannel slice. The sidechannel implementation is tracked separately in TASK-405.
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
