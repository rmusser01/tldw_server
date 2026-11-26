---
id: TASK-15658
title: 'Certify Research Workspace recovery on current dev for GitHub #2608'
status: Done
labels:
- research-workspace
- uat
- destructive-actions
references:
- https://github.com/rmusser01/tldw_server/issues/2608
- TASK-12877
- TASK-12020.27
- TASK-12020.32
- TASK-12020.33
- TASK-15658.1
- TASK-15658.2
- TASK-15658.3
- Docs/ADR/007-research-workspace-canonical-first-slice-shell.md
- Docs/ADR/008-workspace-split-key-persistence-and-indexeddb-offload.md
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- Docs/Reviews/assets/2026-11-25-research-workspace-recovery-uat/evidence.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the remaining RW-UAT-030 evidence gap left by TASK-12020.27 and TASK-12020.35. Recheck all recovery controls against a fresh local WebUI and FastAPI instance using disposable data. Verify imported chats survive hydration and selected-source removal and Undo preserve both membership and domain-owned resources.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Verify workspace archive, restore and delete flows and capture live evidence.
- [x] #2 Verify artifact, chat, message, note, source removal, transfer and bulk recovery against disposable fixtures.
- [x] #3 Verify imported chat preservation and selected-source batch remove and Undo before updating RW-UAT-030.
- [x] #4 Classify any product or environment failures and link focused follow-up work where needed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. This recheck and label correction use the existing Research Workspace route, persistence and recovery contracts from ADR-007 and ADR-008.

Started from freshly fetched dev 1d8589b7c4. Live Chromium tested the current FastAPI and WebUI on isolated disposable data. The original batch removal hit the workspace render boundary after producing a translation object as the Undo label. TASK-15658.1 fixed that and reproduced the browser failure in a real-catalog unit test. The clean rebuilt WebUI passed imported chat reload, selected batch Remove and Undo, archive/cancel/restore and deletion of a typed disposable copy, artifact recovery, clear chat, Quick Notes, single-source recovery, and source-transfer Undo. All three media records and the canonical note were still readable with their original titles and content.

RW-UAT-030 stays Partial. Attempting single-message Delete with valid persisted local chat/message IDs leaves the reply and IndexedDB row intact and reports history_selection_unavailable. TASK-15658.3 records the exact reproduction. TASK-15658.2 records exporter-local chat settings identity warnings while the imported chat remains visible.

The focused suite passed 213 tests. Shared UI typecheck, scoped ESLint, pre-commit for the label fix, and evidence link and manifest validation passed. The tracked changes have no Python source for Bandit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the live recovery recheck and corrected the Undo-label crash. Published local evidence in the UAT matrix. The row remains Partial for the tracked single-message Delete failure.
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
