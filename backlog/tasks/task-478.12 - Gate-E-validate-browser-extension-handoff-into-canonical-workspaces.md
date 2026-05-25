---
id: TASK-478.12
title: 'Gate E: validate browser extension handoff into canonical workspaces'
status: To Do
labels:
- research-workspace
- uat
- gate-e
- browser-extension
- shared-workspaces
- cdp
priority: High
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Workflow gap: extension validation was deferred during UAT because the extension build issue was being handled separately. The browser-extension handoff is still part of the Research Workspace acceptance surface and must target the canonical Shared Workspaces/MCP/ACP/sandbox-aware workspace model.

User goal: capture a page from the browser extension, choose or create the correct workspace, see ingestion/indexing status in WebUI, and use the captured material in RAG/Studio/agent workflows.

Scope:
- Once extension build is available, validate capture/save/organize/summarize/query handoff using CDP/Playwright, not Computer Control.
- Ensure extension-created sources use canonical workspace identifiers and status APIs from TASK-478.3 and TASK-478.7.
- Validate duplicate capture, failed capture, auth/backend unavailable, and workspace-selection states.
- Add or update extension/WebUI integration tests where feasible.

Acceptance criteria:
- Captured browser content appears in the chosen workspace with visible processing/indexing status.
- Extension errors are surfaced clearly and recoverably.
- Captured sources can be selected and used in WebUI RAG/Studio once queryable.
- Extension handoff does not rely on old `workspace-playground` route names, redirects, or aliases.

Depends on: TASK-478.3, TASK-478.4, TASK-478.7; also depends on the extension build being available.
Parallelization: test-plan preparation can happen early; live validation waits for the extension build.
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
