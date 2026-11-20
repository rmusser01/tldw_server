---
id: TASK-15580
title: Fix Research Workspace chat and Knowledge import warnings
status: In Progress
labels:
- research-workspace
- webui
- chat
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Chat readiness degrades when optional queued execution is disabled. The Knowledge handoff also treats signed-in cookie sessions as anonymous and reports a save failure before reading a handoff.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Queue-off chat is ready when its required services are healthy. Missing required queues and provider degradation still warn.
- [x] #2 The import resolves a verified cookie-session owner. Empty workspaces show no import error, and owned pending imports and real failures remain recoverable.
- [x] #3 Regression tests cover both warnings with the production-shaped inputs.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Reuse the existing configuration and verified session boundaries without a new public API or persistence contract.
Observed red failures for the healthy queue-off cases and valid hosted-cookie identity. Focused verification passed 83 backend and 86 WebUI tests. Bandit found no production findings and Ruff matched the touched-file baseline.
Independent review traced both Knowledge handoff senders and Retry. Added verified-cookie opt-in only for those senders and preserved Retry across a transient session lookup failure. Both cases failed before the fix and the full focused WebUI suite now passes 91 tests.
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
