# Workspaces Manager UAT Matrix

Date: 2026-06-04

Scope: canonical `/workspaces` manager, Research Workspace reconciliation, Project Workspace root setup, and cross-surface handoffs into Research Workspace, MCP Hub Shared Workspaces, ACP, and Sandbox-owned execution surfaces.

This matrix is an evidence tracker. Do not mark a row verified unless it was exercised against a live backend and WebUI in the named environment.

## Route And Ownership Rules

- Canonical manager route: `/workspaces`
- Research Workspace route: `/research-workspace`
- Deprecated names: `/workspace-playground` must not be registered as an alias or redirect.
- MCP Hub Shared Workspaces own path-trust records and workspace-set policy.
- ACP owns agent/session execution state.
- Sandbox owns runtime execution and durable sandbox volume mechanics.
- Workspaces own product identity, profile, archival state, primary-root binding metadata, and links into the specialized surfaces above.

## Matrix

| ID | Journey | User Goal | Expected Result | Automated Evidence | Live Evidence | Status |
| --- | --- | --- | --- | --- | --- | --- |
| WM-UAT-001 | Open canonical manager | Find the server-backed Workspace directory | `/workspaces` renders without aliases or redirects from `/workspace-playground` | `option-workspaces.route.test.tsx`; `route-metadata.coverage.test.ts`; `workspaces-page-wrapper.test.ts` | 2026-06-04 Playwright/CDP live run passed against `http://127.0.0.1:18001` backend and `http://localhost:8080` WebUI | Verified live |
| WM-UAT-002 | Create Research Workspace | Create a durable server record for research sources and notes | New Research Workspace appears in manager list and opens `/research-workspace?source_workspace_id=<id>` | `WorkspacesManagerPage.test.tsx` | Not run in this slice | Automated only |
| WM-UAT-003 | Create Project Workspace | Create a Workspace intended for files, roots, agents, and tools | New Project Workspace appears with setup-pending root state | `WorkspacesManagerPage.test.tsx`; `WorkspaceProjectRootPanel.test.tsx` | Not run in this slice | Automated only |
| WM-UAT-004 | Edit metadata | Rename without changing canonical id or route | Row updates after `PATCH /workspaces/{id}` and preserves version handling | `WorkspacesManagerPage.test.tsx` | Not run in this slice | Automated only |
| WM-UAT-005 | Archive and unarchive | Remove inactive Workspace from default list without deleting it | Archive hides by default; unarchive restores active visibility | `WorkspacesManagerPage.test.tsx` | Not run in this slice | Automated only |
| WM-UAT-006 | Upgrade to Project Workspace | Attach project semantics to an existing Research Workspace | Profile changes to Project Workspace; root panel shows next root action | `WorkspaceProjectRootPanel.test.tsx` | Not run in this slice | Automated only |
| WM-UAT-007 | Host-local root attach | Bind one primary local root | Root state updates without creating a second primary root | `WorkspaceProjectRootPanel.test.tsx` | Not run in this slice | Automated only |
| WM-UAT-008 | Sandbox-managed root attach | Provision or attach sandbox-managed root through Workspace-owned command | Operation status is visible; Sandbox remains owner of runtime mechanics | Backend operation tests from earlier slices; `WorkspaceProjectRootPanel.test.tsx` | Not run in this slice | Automated only |
| WM-UAT-009 | Local Research Workspace reconciliation | Promote eligible local-only metadata without rewriting local payloads | Local-only entries appear separately; eligible row can create server metadata; markers are retained local metadata | `workspace-local-reconciliation.test.ts`; `WorkspaceReconciliationPanel.test.tsx` | Not run in this slice | Automated only |
| WM-UAT-010 | Research Workspace to manager | Return from `/research-workspace` to canonical management | Existing settings menu has `Manage in Workspaces`; navigation saves local state then opens `/workspaces` | `WorkspaceHeader.test.tsx` | 2026-06-04 Playwright/CDP live run passed against `http://127.0.0.1:18001` backend and `http://localhost:8080` WebUI | Verified live |
| WM-UAT-011 | MCP Hub guardrail | Understand that Shared Workspaces are path-trust records, not the canonical manager | Shared Workspaces tab links to `/workspaces` and explains ownership boundary | `SharedWorkspacesTab.test.tsx` | Not run in this slice | Automated only |
| WM-UAT-012 | ACP guardrail | Understand that ACP terminal/session state depends on Workspace project-root setup | No-session terminal state links to `/workspaces` and explains project-root setup | `ACPWorkspacePanel.test.tsx` | Not run in this slice | Automated only |
| WM-UAT-013 | Live backend smoke | Prove manager and handoffs work against an actual backend/WebUI | `apps/tldw-frontend/e2e/workflows/workspaces-manager.spec.ts` | 2026-06-04 run skipped because backend preflight was unavailable on `127.0.0.1:8000`; 2026-06-04 rerun passed with backend `127.0.0.1:18001` and webpack WebUI `localhost:8080` | Verified live |

## Live Run Notes

Record each live run with:

- Backend URL and auth mode
- Frontend URL and build/branch
- Browser driver: CDP/Playwright
- Provider/model configuration if chat/RAG is exercised
- Screenshots or trace paths
- Rows moved from `Automated only` to `Verified live`
- Failures filed as follow-up Backlog tasks
- Skipped runs, including unavailable backend/frontend/tooling reasons

### 2026-06-04 Live Run

- Backend: `http://127.0.0.1:18001`, `AUTH_MODE=single_user`, `SINGLE_USER_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY`, `CHAT_FORCE_MOCK=1`, `STREAMS_UNIFIED=1`.
- Frontend: `http://localhost:8080`, `TLDW_WEB_CMD='bun run dev:webpack -- -p 8080'`, branch `codex/workspaces-manager-roadmap`.
- Browser driver: Playwright/CDP via `npx playwright test e2e/workflows/workspaces-manager.spec.ts --reporter=line`.
- Initial live run failed because `/workspaces` returned the WebUI 404 page; root cause was a missing Next.js `pages/workspaces.tsx` wrapper despite package route metadata coverage.
- Fix added: `apps/tldw-frontend/pages/workspaces.tsx` plus `apps/tldw-frontend/__tests__/navigation/workspaces-page-wrapper.test.ts`.
- Rerun result: `2 passed (34.8s)`.

## Known Baseline Blockers At Creation

- `bun run verify:design-system-state` fails on existing canonical-state label debt outside Workspaces:
  - `src/components/Option/Onboarding/steps/FirstChatStep.tsx`
  - `src/services/acp/readiness.ts`
- The UI package currently has no local `typecheck` script or `./node_modules/.bin/tsc` binary in this worktree.
