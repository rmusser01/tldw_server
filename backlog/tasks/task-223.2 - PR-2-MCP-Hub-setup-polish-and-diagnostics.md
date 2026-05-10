---
id: TASK-223.2
title: 'PR 2: MCP Hub setup polish and diagnostics'
status: In Progress
assignee:
  - '@Codex'
created_date: '2026-05-10 06:13'
updated_date: '2026-05-10 16:09'
labels:
  - mcp
  - webui
  - ux
  - diagnostics
dependencies:
  - TASK-223.1
parent_task_id: TASK-223
priority: medium
documentation:
  - Docs/superpowers/specs/2026-05-10-mcp-hub-walkthrough-remediation-design.md
  - Docs/superpowers/plans/2026-05-10-mcp-hub-setup-polish-diagnostics-plan.md
references:
  - https://github.com/rmusser01/tldw_server/pull/1531
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the second PR-sized remediation slice from the MCP Hub walkthrough. This phase should make setup states easier to understand after the live-discovery and chat blocker fixes land.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No-auth local stdio servers render a neutral or healthy no-credentials-required state instead of missing-auth warnings.
- [x] #2 Legacy Secret Fallback appears only when the selected managed server actually uses the transitional server-level secret flow.
- [x] #3 Tool Catalog empty and stale states offer clear Add server and Refresh discovery actions with setup, runtime, and permissions distinctions.
- [x] #4 MCP Hub or shared diagnostics expose effective deployment mode, API origin, and health endpoint enough to diagnose quickstart versus advanced split-brain configuration.
- [x] #5 Setup isolation expectations for local walkthrough or E2E runs are documented or verified where practical.
- [x] #6 Focused UI tests and a toy MCP E2E smoke cover the polished setup path.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
PR 2 implementation plan drafted at Docs/superpowers/plans/2026-05-10-mcp-hub-setup-polish-diagnostics-plan.md.

Stages:
1. No-auth and legacy secret setup states in Servers & Credentials.
2. Tool Catalog empty and recovery guidance for setup, runtime, and permissions distinctions.
3. Compact deployment diagnostics in the Setup workflow.
4. Walkthrough isolation documentation and skip-safe toy MCP smoke coverage.
5. Focused UI/E2E/docs verification and PR packaging.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-05-10: Started PR 2 from merged PR #1514 and current origin/dev in .worktrees/mcp-hub-pr2-setup-polish on branch codex/mcp-hub-pr2-setup-polish. Baseline focused MCP Hub UI tests passed after installing frontend dependencies from apps/: ExternalServersTab and ToolCatalogsTab, 14 tests passed. Plan created for a frontend/docs-heavy PR 2 using existing PR 1 refresh contract.
2026-05-10: Implemented no-auth stdio setup states, Tool Catalog recovery guidance, Setup deployment diagnostics, toy MCP Playwright smoke coverage, and the isolated local walkthrough docs.
2026-05-10: Verification: `cd apps/packages/ui && bun run test src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx src/components/Option/MCPHub/__tests__/DeploymentDiagnosticsPanel.test.tsx src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx` passed 4 files / 29 tests. `cd apps/tldw-frontend && bunx playwright test e2e/workflows/tier-2-features/mcp-hub.spec.ts --list` listed 9 tests. `cd apps/tldw-frontend && bunx playwright test e2e/workflows/tier-2-features/mcp-hub.spec.ts --reporter=line` passed 3 and skipped 6 live-server-dependent cases. Initial sandboxed Playwright run could not bind `0.0.0.0:8080`; reran with approved escalation. `git diff --check` passed. Bandit skipped because this slice changed TypeScript, docs, and Playwright only; no Python production files were touched.
2026-05-10: Opened PR #1531 for branch codex/mcp-hub-pr2-setup-polish.
2026-05-10: Addressed PR #1531 review feedback. Removed the committed toy walkthrough API key in favor of per-run generation, tightened Tool Catalog chat-executability guidance to chat-enabled tools only, surfaced external-server inventory load failures as a retryable unknown state instead of a false no-server empty state, and cleaned the Playwright toy MCP temp directory in finally.
2026-05-10: Review-fix verification: `cd apps/packages/ui && bun run test src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx` passed 9 tests. `cd apps/packages/ui && bun run test src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx src/components/Option/MCPHub/__tests__/DeploymentDiagnosticsPanel.test.tsx src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx` passed 4 files / 30 tests. `cd apps/tldw-frontend && bunx playwright test e2e/workflows/tier-2-features/mcp-hub.spec.ts --list` listed 9 tests. `cd apps/tldw-frontend && bunx playwright test e2e/workflows/tier-2-features/mcp-hub.spec.ts --reporter=line` passed 3 and skipped 6 live-server-dependent cases. `git diff --check` passed. Bandit remains skipped because the review fixes touched TypeScript, docs, and Playwright only; no Python production code was touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR 2 setup polish is implemented. The MCP Hub now distinguishes no-auth local stdio servers from missing-credential states, keeps legacy secret fallback scoped to transitional server-level auth, gives Tool Catalog users setup/runtime/access-specific recovery actions, and shows compact Setup diagnostics for deployment mode and API origin. The local toy MCP walkthrough is documented with disposable database/storage paths and the E2E smoke now exercises the UI setup path when a live API can mutate and discover the temporary stdio server.

Known skip: the Playwright toy and API cases intentionally skip when no live backend is available or when the API host cannot execute the test runner's temporary stdio file. That is expected for frontend-only CI and non-shared-filesystem deployments.
<!-- SECTION:FINAL_SUMMARY:END -->
