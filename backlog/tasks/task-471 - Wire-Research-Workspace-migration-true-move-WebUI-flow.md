---
id: TASK-471
title: Wire Research Workspace migration true-move WebUI flow
status: In Progress
labels:
- research-workspace
- migration
- webui
references:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
- Docs/Design/Research_Workspace_Legacy_Storage_Inventory.md
- Docs/Design/Research_Workspace_Migration_Protocol_API.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the client-side /research-workspace migration driver that uses the existing legacy inventory gate and backend migration protocol to create sessions, upload chunk receipts, finalize, wait for client_delete_eligible, delete only approved local content payloads, write a tombstone, and send client-delete-ack. No /workspace-playground aliases or redirects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
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
