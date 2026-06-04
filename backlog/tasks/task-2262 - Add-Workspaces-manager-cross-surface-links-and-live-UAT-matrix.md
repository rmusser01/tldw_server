---
id: TASK-2262
title: Add Workspaces manager cross-surface links and live UAT matrix
status: Done
labels:
- workspaces
- webui
- uat
references:
- Docs/superpowers/plans/2026-06-04-canonical-workspaces-manager-project-creation.md
- Docs/superpowers/specs/2026-06-04-canonical-workspaces-manager-project-creation-design.md
modified_files:
- apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx
- apps/packages/ui/src/components/Option/MCPHub/SharedWorkspacesTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/__tests__/SharedWorkspacesTab.test.tsx
- apps/packages/ui/src/components/Option/ACPPlayground/ACPWorkspacePanel.tsx
- apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPWorkspacePanel.test.tsx
- apps/tldw-frontend/pages/workspaces.tsx
- apps/tldw-frontend/__tests__/navigation/workspaces-page-wrapper.test.ts
- apps/tldw-frontend/e2e/workflows/workspaces-manager.spec.ts
- Docs/Validation/workspaces-manager-uat-matrix.md
- backlog/tasks/task-2262 - Add-Workspaces-manager-cross-surface-links-and-live-UAT-matrix.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 8 from the canonical Workspaces manager plan: add cross-surface links from Research Workspace, MCP Hub, ACP, and related workspace surfaces to the canonical /workspaces manager where appropriate, then add live UAT matrix/e2e smoke coverage for the manager flow. Preserve canonical Workspace ownership; do not add route aliases or redirects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Research Workspace exposes a clear navigation path back to the canonical `/workspaces` manager without changing route identity or adding aliases.
- [x] #2 MCP Hub Shared Workspaces and ACP workspace surfaces use copy/link guardrails that distinguish specialized ownership from canonical Workspace management.
- [x] #3 Focused WebUI tests cover cross-surface manager links and prevent `/workspace-playground` alias/redirect regression.
- [x] #4 A live UAT matrix or template documents manager flows across create, edit, archive, project root, reconciliation, and cross-surface handoff with honest verification status.
- [x] #5 Focused tests pass; broader guard failures are documented as unrelated baseline issues.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Task 8 from `Docs/superpowers/plans/2026-06-04-canonical-workspaces-manager-project-creation.md`.
- Scope should stay on navigation, copy guardrails, and validation evidence. Do not duplicate MCP/ACP/Sandbox ownership inside the canonical manager, and do not add `/workspace-playground` aliases or redirects.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Added canonical manager links from the Research Workspace header Workspaces dropdown and settings menu, preserving `/research-workspace` as its own route and using `/workspaces` only for management.
- Added MCP Hub Shared Workspaces copy and link guardrails that identify Shared Workspaces as MCP path-trust records rather than canonical Workspace metadata.
- Added ACP Workspace terminal empty/unavailable/toolbar links to `/workspaces` so project-root setup is discoverable without duplicating ACP execution state ownership.
- Added `Docs/Validation/workspaces-manager-uat-matrix.md` and `apps/tldw-frontend/e2e/workflows/workspaces-manager.spec.ts`. The matrix records automated coverage, the initial live skip, the live 404 root cause, and the passing live rerun.
- Added the missing Next.js `pages/workspaces.tsx` wrapper plus `workspaces-page-wrapper.test.ts` after Playwright/CDP showed the package route existed but the live WebUI still returned a 404 for `/workspaces`.
- Verification: `./node_modules/.bin/vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx src/components/Option/MCPHub/__tests__/SharedWorkspacesTab.test.tsx src/components/Option/ACPPlayground/__tests__/ACPWorkspacePanel.test.tsx src/routes/__tests__/option-workspaces.route.test.tsx src/routes/__tests__/route-metadata.coverage.test.ts` passed with 58 tests.
- Verification: `./node_modules/.bin/vitest run __tests__/navigation/workspaces-page-wrapper.test.ts` failed before the wrapper existed, then passed after adding it.
- Verification: `npx playwright test e2e/workflows/workspaces-manager.spec.ts --list` discovered 2 tests.
- Verification: `TLDW_WEB_CMD='bun run dev:webpack -- -p 8080' npx playwright test e2e/workflows/workspaces-manager.spec.ts --reporter=line` ran under Playwright/CDP after dependency environment repair; 2 tests skipped because backend preflight was unavailable on `127.0.0.1:8000`.
- Verification: `TLDW_WEB_CMD='bun run dev:webpack -- -p 8080' TLDW_SERVER_URL=http://127.0.0.1:18001 TLDW_E2E_SERVER_URL=http://127.0.0.1:18001 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY npx playwright test e2e/workflows/workspaces-manager.spec.ts --reporter=line` failed before the Next page wrapper fix because `/workspaces` rendered the WebUI 404, then passed after the fix with `2 passed (34.8s)`.
- Verification: `git diff --check` passed.
- Known blocker: `bun run verify:design-system-state` still fails on existing non-Workspaces canonical-state labels in `FirstChatStep.tsx` and `src/services/acp/readiness.ts`.
- Bandit skipped: frontend TypeScript, E2E, docs, and Backlog-only slice.

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
