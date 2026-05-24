---
id: TASK-476
title: Validate Research Workspace live backend WebUI extension CDP matrix
status: In Progress
labels:
- research-workspace
- validation
- e2e
- cdp
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run a live backend + WebUI + extension/CDP validation pass for Research Workspace to detect hidden breakage across exposed workflows. Produce a concrete matrix covering route availability, source capture, ingestion/indexing status, grounded chat, migration, export/resume, extension handoff, MCP/ACP/Sandbox workspace model touchpoints, and failure states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Actual backend and WebUI are started and health checked.
- [ ] #2 Research Workspace is inspected through browser/CDP, not Computer Control.
- [ ] #3 Extension/WebUI handoff flows are checked where the local tooling permits.
- [ ] #4 Validation matrix is recorded with pass/fail/blocked status, evidence, and follow-up issues.
- [ ] #5 Any discovered breakage is triaged into fix-now versus follow-up.
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
