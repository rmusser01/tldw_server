---
id: TASK-476
title: Validate Research Workspace live backend WebUI extension CDP matrix
status: Done
labels:
- research-workspace
- validation
- e2e
- cdp
modified_files:
- tldw_Server_API/app/api/v1/endpoints/workspaces.py
- tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py
- tldw_Server_API/tests/Workspaces/test_workspaces_api.py
- apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/AddSourceModal.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_VALIDATION_MATRIX_2026_05_24.md
- Docs/Reviews/research-workspace-live-validation-2026-05-24.png
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run a live backend + WebUI + extension/CDP validation pass for Research Workspace to detect hidden breakage across exposed workflows. Produce a concrete matrix covering route availability, source capture, ingestion/indexing status, grounded chat, migration, export/resume, extension handoff, MCP/ACP/Sandbox workspace model touchpoints, and failure states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual backend and WebUI are started and health checked.
- [x] #2 Research Workspace is inspected through browser/CDP, not Computer Control.
- [x] #3 Extension/WebUI handoff flows are checked where the local tooling permits.
- [x] #4 Validation matrix is recorded with pass/fail/blocked status, evidence, and follow-up issues.
- [x] #5 Any discovered breakage is triaged into fix-now versus follow-up.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Current-progress fixes committed in `13d62f065`.
- Live matrix: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_VALIDATION_MATRIX_2026_05_24.md`.
- Screenshot: `Docs/Reviews/research-workspace-live-validation-2026-05-24.png`.
- Verification recorded: backend tests `44 passed`, UI tests `59 passed`, Bandit `0 findings`, live API sweep `38/38 passed`.
- Browser validation used Playwright automation only; Computer Control was not used.
- Extension current-build validation is blocked by WXT production build hang. Existing packaged extension build connected to the backend but failed the current Research Workspace contract and is stale-build evidence only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Live backend + WebUI + browser validation completed for Research Workspace. Current-progress fixes were committed in 13d62f065. Matrix recorded in Docs/Reviews/RESEARCH_WORKSPACE_LIVE_VALIDATION_MATRIX_2026_05_24.md. Results: backend/WebUI/API/browser route and source workflows passed; live API sweep passed 38/38; MCP status and MCP Hub Shared Workspaces responded; ACP workspace CRUD and sandbox routes are not exposed in this live route set; extension current-build real-backend E2E is blocked by WXT production build hang, and a stale packaged extension build connected to the backend but failed the current Research Workspace contract.
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
