---
id: TASK-214
title: Implement MCP Hub workflow shell
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-10 02:54'
updated_date: '2026-05-10 03:46'
labels:
  - ux
  - mcp-hub
  - webui
  - extension
  - implementation
dependencies: []
references:
  - apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx
  - apps/packages/ui/src/components/Option/MCPHub/mcpHubWorkflowConfig.ts
  - apps/packages/ui/src/tutorials/definitions/mcp-hub.ts
  - apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts
documentation:
  - >-
    Docs/superpowers/specs/2026-05-10-mcp-hub-workflow-first-control-panel-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 1 of the approved MCP Hub workflow-first design: replace the flat object-centric tab row with workflow and child-view navigation while preserving existing child components and service contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workflow config maps every current MCP Hub view into exactly one workflow.
- [x] #2 McpHubPage defaults to Setup / Servers & Credentials and supports workflow/view query state.
- [x] #3 Audit drilldown opens the mapped workflow and child view while preserving focus context.
- [x] #4 Shared WebUI/extension route behavior uses the existing router shim path rather than direct window history.
- [x] #5 Focused unit/component and E2E/page-object tests cover workflow navigation, deep links, audit drilldown, and route parity.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for workflow config, default route/query parsing, workflow/view selection, and audit drilldown mapping. 2. Implement workflow config and route-state helpers. 3. Refactor McpHubPage to render workflow navigation plus child-view navigation while reusing existing child components. 4. Update MCP Hub E2E page object and tests from tab locators to workflow/view helpers. 5. Run focused frontend tests, E2E where feasible, and Markdown/task verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
In isolated branch codex/mcp-hub-workflow-shell, completed the unit/component red-green loop for Stage 1 workflow shell. Red run failed on missing workflow config and old tab/default behavior. Green run: bunx vitest run ../packages/ui/src/components/Option/MCPHub/__tests__/McpHubWorkflowConfig.test.ts ../packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx ../packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx passed 18 tests across 3 files; jsdom emitted existing CSS parse warnings.

Added final hardening after review: MCP Hub tutorial now targets persistent workflow buttons for non-default sections, and route-state resolution preserves a valid workflow deep link when view is invalid.

Verification: focused Vitest suite passed 29 tests across MCP Hub config/page/FTUX/route/tutorial coverage; touched-file ESLint exited 0 with the existing Next pages-directory warning; git diff --check passed; Playwright MCP Hub spec passed 3 page/navigation/query tests and skipped 5 backend-dependent API checks via server availability guard. Full bunx tsc --noEmit --pretty false remains blocked by unrelated existing errors in EmbeddingsModelSelectionConfig.tsx, persona-visuals.ts, and lib/api/vnPlay.ts. Bandit skipped because touched implementation files are frontend TypeScript/JSON and Backlog metadata only.

After touching the MCP Hub tutorial locale text, mechanically regenerated apps/packages/ui/src/public/_locales/en/tutorials.json from the nested source locale so the extension locale mirror test passes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the workflow-first MCP Hub shell: workflow navigation, child-view tabs, workflow/view URL state, Setup / Servers & Credentials default, audit drilldown remapping, updated E2E page object/spec coverage, and tutorial targets that remain visible under the new shell. Synced the extension tutorial locale mirror after updating MCP Hub tutorial text. Verified focused UI, route, tutorial, lint, whitespace, locale mirror, and MCP Hub browser E2E checks; documented the unrelated full TypeScript baseline failures and non-Python Bandit skip.
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
