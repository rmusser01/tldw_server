---
id: TASK-12034
title: Adopt MCP Hub capability recovery state
status: Done
created_date: 2026-06-25 22:51
labels:
- webui
- mcp
- ux
- accessibility
priority: medium
references:
- TASK-418.11
documentation:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-25-webui-stage5-mcp-hub-capability-recovery-plan.md
- apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx
- apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx
- apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx
updated_date: 2026-06-25 22:57
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred WebUI capability/error-state follow-up for the standalone `/mcp-hub` route. Add a top-level MCP Hub readiness probe using existing frontend service APIs, render the shared user-language recovery state when the MCP Hub backend capability is unavailable or blocked, preserve existing workflow navigation when available, and keep raw request details behind diagnostics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Standalone MCP Hub shows a shared RecoveryCallout when the MCP Hub capability probe fails before tab-specific content fails piecemeal.
- [x] #2 Recovery state includes user-language title/message, retry action, and diagnostics for method/path/status/raw message without exposing secrets.
- [x] #3 Existing MCP Hub workflows, status summary, query-state routing, and deployment diagnostics remain available when the capability probe succeeds.
- [x] #4 Focused tests cover available and unavailable capability paths.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused Stage 5 plan document for the MCP Hub capability recovery slice.
2. Write failing MCP Hub tests for a rejected capability probe and a successful probe preserving current workflow behavior.
3. Implement a minimal React Query readiness probe in McpHubPage using the existing MCP Hub service contract.
4. Render the shared RecoveryCallout for probe failures with retry diagnostics, preserving the existing page for loading/success states.
5. Run focused frontend tests, lint checks, diff whitespace checks, and document browser-smoke blockers if the local environment still prevents full Playwright interaction.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the MCP Hub capability recovery slice. Added a top-level React Query probe against the existing `/api/v1/mcp/hub/tool-registry/summary` service, mapped failures through the shared capability-state builder, and render a `RecoveryCallout` with retry and diagnostics before tab-specific MCP Hub content fails piecemeal. Existing workflow/status-summary behavior remains unchanged when the probe succeeds.

Verification:
- `cd apps/tldw-frontend && bun run test:run ../packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx ../packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx` passed: 2 files, 19 tests.
- `apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx` passed with no touched-file findings; ESLint emitted the existing repo-root Next pages-directory warning.
- `git diff --check` passed.
- Bandit not applicable: touched code is TS/TSX plus plan/task markdown only.

Known skip: full browser smoke was not rerun for this narrow unit; the prior local browser/dev-server environment blockers remain outside this task scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a shared MCP Hub capability recovery state for `/mcp-hub`. The page now probes the existing MCP Hub tool-registry summary endpoint, shows user-language recovery with retry and diagnostics when the backend capability is unavailable, and preserves existing workflow navigation and FTUX behavior when the probe succeeds. Focused MCP Hub tests and lint/whitespace checks passed.
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
