---
id: TASK-478.18
title: Refresh Research Workspace UAT matrix after migration true-move validation
status: Done
labels:
- research-workspace
- uat
- migration
- docs
priority: Medium
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the maintained Research Workspace live UAT matrix and parent workstream notes with TASK-515/TASK-516 evidence for migration true-move deletion, durable tombstone suppression, and remaining migration recovery scope. Keep remaining MCP/ACP/Sandbox and keyboard-only gaps explicit rather than overclaiming Pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RW-UAT-025 records TASK-515/TASK-516 true-move migration evidence for server delete eligibility, client delete ack, tombstone write, blocked retention, and durable no-repersist behavior.
- [x] #2 RW-UAT-025 remains Partial unless the broader import/export recovery walkthrough is validated live.
- [x] #3 The matrix high-risk remainder list preserves MCP/ACP/Sandbox, keyboard-only, vector completion, and migration recovery gaps without overclaiming completed workflows.
- [x] #4 Parent TASK-478 notes mention post-matrix migration follow-up evidence and the remaining explicit gaps.
- [x] #5 Verification is recorded; Bandit is skipped as docs/backlog-only if no runtime code changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Updated `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md` so RW-UAT-025 now cites TASK-515 and TASK-516 evidence: server delete eligibility after readback verification, `client-delete-ack`, tombstone write, blocked retention, and durable no-repersist behavior.
- Kept RW-UAT-025 as `Partial` because broader import/export recovery and a guided migration-recovery walkthrough are still not fully live-validated.
- Updated RW-UAT-026 to include TASK-478.18 as a matrix maintenance follow-up and removed the stale extension-handoff follow-up copy now that TASK-478.12 is done.
- Updated parent TASK-478 status to `In Progress` and added notes for post-original-plan follow-ups plus explicit remaining gaps.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refreshed the Research Workspace UAT matrix after the migration true-move work. The ledger now records TASK-515/TASK-516 live evidence for eligible delete, ack, tombstone, blocked retention, and durable no-repersist behavior, while preserving the broader migration/import/export recovery row as Partial until a full recovery walkthrough is live-validated.

Verification:
- `git diff --check`: passed.
- Bandit: skipped; this task changed docs and Backlog metadata only.
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
