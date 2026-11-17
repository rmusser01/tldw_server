---
id: TASK-13678
title: Derive cold chat save feedback from owned history recovery evidence
status: Done
labels:
- chat
- ux
- recovery
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cold reload loses the process-local write store and maps unknown plus server ID to Saved on server, even when the durable latest turn is unresolved. Default to truthful unconfirmed feedback and project current owner-verified H1 recovery/load evidence into the shared save-status source; preserve active writes and original account/view leases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cold server-linked chat is not called saved solely because an ID exists.
- [x] #2 Current owned journal/load evidence distinguishes confirmed history from retained unresolved turns and read failure; retired owner/view/account evidence cannot promote status.
- [x] #3 Shared header/persistence callers, actual cold journal controls, both profiles/types/build and independent review qualify the repair.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fresh independent review Important at Header69/chat-persistence-status53. ADR assessment: existing ADR-049 owner/load/journal authority and acknowledged feedback; no new persistence authority or server protocol.
Current connected381 controls:380pass1fail; a live write end reconstructed its entry and dropped history evidence, so preserve entry fields in that existing shared update. Independent source follow-up also identifies overlapping same-owner journal reads: an older empty read can overwrite newer unresolved evidence. Add delayed-old-empty/newer-unresolved regression and bind publication/current validators to the latest recovery-read revision, preserving durable journal/H1 contracts.
ID alone no longer claims Saved. Current owned read/write evidence, cold unknown state, failed reads and monotonic recovery-read revision govern shared feedback. Native/WebUI recovery/status382 each and real cold empty-journal unconfirmed cell pass; saved-write evidence retains its distinct outcome. PR #3229; final bounded source review by two independent reviewers. TypeScript/JSON/Markdown-only unit: Bandit N/A locally; fresh full PR67-production-Python scan has0 introduced findings and0 scan errors vs exact merge-base. Installed extension/physical AT/touch/native-toolbar zoom/participants remain unrun. Overall merge and current Scene stability remain separate gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
ID alone no longer claims Saved. Current owned read/write evidence, cold unknown state, failed reads and monotonic recovery-read revision govern shared feedback. Native/WebUI recovery/status382 each and real cold empty-journal unconfirmed cell pass; saved-write evidence retains its distinct outcome.
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
