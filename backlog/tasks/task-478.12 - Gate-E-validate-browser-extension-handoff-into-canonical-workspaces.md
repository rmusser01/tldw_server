---
id: TASK-478.12
title: 'Gate E: validate browser extension handoff into canonical workspaces'
status: Done
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
documentation:
- Docs/superpowers/plans/2026-05-25-task-478-12-extension-workspace-handoff.md
modified_files:
- Docs/superpowers/plans/2026-05-25-task-478-12-extension-workspace-handoff.md
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- apps/extension/tests/e2e/research-workspace.real-backend.spec.ts
- apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx
- apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx
- apps/packages/ui/src/entries/shared/background-init.ts
- apps/packages/ui/src/entries/shared/__tests__/background-init.test.ts
- apps/packages/ui/src/services/background-proxy.ts
- apps/packages/ui/src/services/__tests__/background-proxy.test.ts
- apps/packages/ui/src/services/tldw/domains/web-clipper.ts
- apps/packages/ui/src/services/__tests__/web-clipper-client.test.ts
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
Live CDP validation isolated the failure boundary to extension runtime/background
messaging. During the failing run, the Web Clipper save request timed out before
the background worker handled it, and the backend never saw
`POST /api/v1/web-clipper/save`.

Implementation:
- Kept `/research-workspace` as the canonical open target for workspace saves.
- Made noncritical background startup warmups fire-and-forget so listener setup is
  not gated by OpenAPI drift/model refresh work.
- Preserved conservative fallback behavior for generic unsafe writes.
- Added a scoped exception so only idempotent Web Clipper saves with a non-empty
  `clip_id` can fall back to the direct API request after extension messaging
  timeout.
- Added live Chrome MV3 coverage for save/open handoff and backend persistence.

Live result:
- Chrome MV3 extension saved a deterministic workspace clip against
  `http://127.0.0.1:18002`.
- The extension opened canonical `#/research-workspace`.
- `GET /api/v1/web-clipper/{clip_id}` showed workspace placement.
- `GET /api/v1/workspaces/{workspace_id}/notes` included the clipped body.
- `GET /api/v1/workspaces/{workspace_id}/sources/status` was reachable.

Known gap: browser clips currently persist as workspace notes/placements, not as
first-class `workspace_sources` entries with ingestion/indexing/RAG source
status. RW-UAT-024 is therefore `Partial`, not `Pass`.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-478.12 validates the live browser-extension handoff into canonical Research Workspace. The extension now opens #/research-workspace for workspace saves, the save path survives runtime messaging timeouts through a scoped idempotent fallback, and regression coverage covers both the UI route target and the live Chrome MV3 backend handoff. RW-UAT-024 is updated to Partial because the handoff works as workspace note placement but not yet as first-class indexed source ingestion.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed for canonical extension save/open handoff; first-class indexed source ingestion remains documented as a follow-up gap
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit not applicable: no Python files changed in TASK-478.12
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
