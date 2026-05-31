---
id: TASK-220
title: Address PR 1498 review feedback
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-10 04:52'
updated_date: '2026-05-10 04:58'
labels:
  - mcp-hub
  - webui
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1498'
  - apps/packages/ui/src/components/Option/MCPHub/mcpHubWorkflowConfig.ts
  - apps/tldw-frontend/e2e/utils/page-objects/MCPHubPage.ts
  - apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts
  - apps/packages/ui/src/tutorials/definitions/flashcards.ts
  - >-
    Docs/superpowers/specs/2026-05-10-mcp-hub-workflow-first-control-panel-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address actionable CodeRabbit and Qodo feedback on PR #1498 while keeping the branch scoped to MCP Hub workflow shell review fixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CodeRabbit review comments are verified and addressed or documented.
- [x] #2 Qodo review findings are verified and addressed or documented.
- [x] #3 Focused frontend tests and MCP Hub E2E checks are rerun after fixes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved actionable review feedback on PR #1498. CodeRabbit: removed redundant route fallback branch, reused MCPHubPage.VIEW_KEYS in E2E spec, normalized TASK-211 assignee to @Codex. Qodo: updated Flashcards tutorial transfer i18n keys/fallbacks with a regression test, converted local absolute design-doc links to repo-relative links, derived MCP_HUB_VIEW_KEYS from labels and made workflowForMcpHubView fail loudly for unmapped valid views, and added E2E page-object waits after workflow switches.

Verification: flashcards tutorial regression test failed before the fix and passed after. Focused Vitest suite passed 31 tests across MCP Hub config/page/route/tutorial/locale coverage and flashcards tutorial coverage. Touched-file ESLint exited 0 with the existing Next pages-directory warning. git diff --check passed. MCP Hub Playwright spec passed 3 page/navigation/query tests and skipped 5 backend-dependent API checks via server availability guard. bunx tsc --noEmit --pretty false remains blocked by unrelated existing errors in EmbeddingsModelSelectionConfig.tsx, persona-visuals.ts, and lib/api/vnPlay.ts. Bandit skipped because touched implementation files are frontend TypeScript/Markdown/Backlog metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1498 CodeRabbit and Qodo review feedback with scoped fixes: Flashcards tutorial transfer i18n keys, repo-relative design-doc links, workflow config key drift hardening, E2E workflow/view wait hardening, canonical MCPHubPage.VIEW_KEYS usage, redundant route fallback cleanup, and TASK-211 assignee formatting. Verified focused frontend tests, lint, diff whitespace, and MCP Hub Playwright; documented the unrelated TypeScript baseline blocker and non-Python Bandit skip.
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
