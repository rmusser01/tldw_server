---
id: TASK-300
title: Design ACP workspace integration decision for issue 1540
status: Done
assignee: []
created_date: '2026-05-12 14:26'
updated_date: '2026-05-12 14:54'
labels:
  - ACP
  - workspace
  - design
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1540'
  - 'https://github.com/rmusser01/tldw_server/issues/1526'
  - 'https://github.com/rmusser01/tldw_server/issues/1532'
documentation:
  - Docs/Design/Workspace_Canonical_Model_Decision_2026_05.md
  - Docs/Design/Workspace_Persistence_Architecture.md
  - Docs/Product/ACP_Agent_Orchestration_PRD.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the ACP workspace integration design gate for #1540 using the existing canonical workspace decision in #1526. Scope is a durable decision/spec plus tracker updates that prevent ACP from creating a parallel workspace model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ACP projects tasks runs diagnostics and reviews are mapped to the canonical workspace model
- [x] #2 Workspace ownership permissions retention and MCP/env flow are documented
- [x] #3 UI/API touchpoints across WorkspacePlayground Agent Tasks and ACP Playground are mapped
- [x] #4 Follow-up implementation slices are split into reviewable units
- [x] #5 Issue #1540 can be updated with evidence and next steps
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Docs/Design/ACP_Workspace_Integration_Decision_2026_05.md to define the canonical workspace to ACP execution workspace bridge for issue #1540.

Linked the decision from the canonical workspace decision doc, ACP PRD, and ACP development documentation.

Verification so far: rg trailing-whitespace check returned no matches; targeted rg confirmed canonical workspace, WorkspacePlayground, Agent Tasks, ACP Playground, agent-orchestration/workspaces, and canonical_workspace_id coverage; git diff --check returned exit 0 for tracked docs. Bandit is not applicable to this docs-only slice.

Opened PR #1614 for the ACP workspace integration design gate and commented on issue #1540 with the remaining implementation order.

PR #1614 review pass: Qodo flagged missing trusted-roots flow and missing explicit verification/testing slice; Gemini flagged possible tldw_Server_API casing ambiguity. Reopening TASK-300 for review-fix documentation updates.

Review fixes applied: clarified repository casing for tldw_Server_API, added trusted-root selection/inheritance/enforcement flow, and made verification/testing an explicit implementation slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Docs/Design/ACP_Workspace_Integration_Decision_2026_05.md, linked it from the canonical workspace decision, ACP PRD, and ACP development docs, opened PR #1614, updated issue #1540 with the backend/UI/history closeout order, and addressed PR review comments by clarifying tldw_Server_API casing, trusted-root flow, and verification/testing slice boundaries. Verification: git diff --check passed; trailing-whitespace rg returned no matches; targeted rg confirmed required bridge/workspace/trusted-root touchpoints. Bandit skipped because this is a docs-only change.
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
