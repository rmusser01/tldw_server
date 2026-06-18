# Workspace Phase 2 Single-User Container Evidence

Date: 2026-06-18
Status: Complete for #1995 branch evidence
Tracking: [#1995](https://github.com/rmusser01/tldw_server/issues/1995), [#1984](https://github.com/rmusser01/tldw_server/issues/1984)

This matrix consolidates the final release evidence for the single-user
Workspace Phase 2 container loop. It extends the narrower
`workspaces-manager-uat-matrix.md` with the full resource set from #1995:
workspace notes, media/sources, artifacts, chats, prompts, workflows,
watchlists, ACP sessions, Sandbox sessions, runtime bindings, active-context
eligibility, and global visibility invariants.

## Current Evidence Snapshot

- Backend baseline before contract repair:
  `python -m pytest tldw_Server_API/tests/Workspaces -q` produced
  `459 passed, 1 failed`. The failure was a stale eligibility expectation:
  `acp_session` is now a supported membership adapter, so unsupported coverage
  must use a still-future type such as `acp_run`.
- Focused contract repair verification:
  `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_eligibility.py -q`
  passed with `16 passed, 6 warnings`.
- Backend consolidated verification:
  `python -m pytest tldw_Server_API/tests/Workspaces -q` passed with
  `461 passed, 8 warnings`.
- Frontend focused contract verification:
  `./node_modules/.bin/vitest run ...workspace focused suite...` passed with
  `10 passed (10 files), 79 passed (79 tests)`. The initial run exposed a
  stale `WorkspaceHeader` expectation that clicked the modal-closing ACP
  footer action before asserting the diagnostics action; the test now verifies
  diagnostics first and then the Agent Tasks handoff.
- Browser evidence:
  `TLDW_WEB_CMD='bun run dev:webpack -- -p 8080' TLDW_SERVER_URL=http://127.0.0.1:18001 TLDW_E2E_SERVER_URL=http://127.0.0.1:18001 TLDW_E2E_API_KEY=tldw-test-api-key-1995 ./node_modules/.bin/playwright test e2e/workflows/workspaces-manager.spec.ts --reporter=line --workers=1`
  passed with `2 passed`. The local API server used
  `AUTH_MODE=single_user`, `SINGLE_USER_API_KEY=tldw-test-api-key-1995`,
  `CHAT_FORCE_MOCK=1`, and `STREAMS_UNIFIED=1`.

## Scope And Deferrals

Covered by this matrix:

- Workspace lifecycle, archive/delete, profile/context, roots, file inventory,
  source status, capabilities, memberships, runtime bindings, and eligibility.
- Resource memberships for `workspace_note`, `media`, `workspace_source`,
  `workspace_artifact`, `chat`, `prompt`, `workflow`, `watchlist`,
  `acp_session`, and `sandbox_session`.
- Visibility and recovery invariants: global browse/search/open/edit remains
  domain-owned; active-context operations require an eligible active workspace.

Deferred from the agreed Phase 2 evidence set:

- Global reusable `note` adapter. Current covered notes are Workspace-owned
  `workspace_note` rows.
- `acp_run`, `project_file`, `study_deck`, `quiz`, and `study_pack` membership
  adapters. These remain reserved/future fail-closed resource types.
- Multi-user sharing/collaboration and long-lived preview hosting.

## Scenario Matrix

| ID | Requirement | Expected Result | Backend Evidence | Frontend/Browser Evidence | Status |
| --- | --- | --- | --- | --- | --- |
| W2E-001 | Create workspace from scratch | New `research` or `project` workspace has stable `workspace_id`, versioning, lifecycle timestamps, and context payload | `test_workspaces_api.py`, `test_workspace_core_context.py`, `test_workspace_core_models.py`; backend suite `461 passed` | `WorkspacesManagerPage.test.tsx`; `/workspaces` smoke passed in `workspaces-manager.spec.ts` | Verified |
| W2E-002 | Import/reconcile existing workspace metadata | Existing local/reconciled workspace can map to canonical server metadata without rewriting local payloads | `test_workspace_migration_api.py`; backend suite `461 passed` | `workspace-local-reconciliation.test.ts`, `WorkspaceReconciliationPanel.test.tsx`; focused Vitest `79 passed` | Verified |
| W2E-003 | Attach existing repo/project root | Project workspace can bind a host-local or Sandbox-managed primary root without duplicate primary roots | `test_workspace_project_roots_db.py`, `test_workspace_root_binding_service.py`, `test_workspace_sandbox_root_provisioning.py`; backend suite `461 passed` | `WorkspaceProjectRootPanel.test.tsx`; focused Vitest `79 passed` | Verified |
| W2E-004 | File inventory and root status | File inventory is metadata-only, Jobs-backed, and reports scan/status/items without becoming source content or trust | `test_workspace_file_inventory_*`; backend suite `461 passed` | Root/status surfaced by focused manager/project-root tests; browser manager smoke passed | Verified |
| W2E-005 | Workspace notes membership | Workspace-owned notes attach/list/update/remove and participate in generic membership summary | `test_workspace_sub_resources_api.py`, `test_workspace_membership_adapters.py`, `test_workspace_memberships_api.py`; backend suite `461 passed` | Research Workspace local note/store browser flow not rerun in this slice | Backend verified; frontend gap noted |
| W2E-006 | Media and workspace sources membership | Global media links remain globally visible; workspace sources own selection/order/readiness inside Research Workspace | `test_workspace_source_status_api.py`, `test_workspace_source_preview_context_api.py`, `test_workspace_membership_adapters.py`, `test_workspace_memberships_api.py`; backend suite `461 passed` | Research Workspace source E2E/parity specs not rerun in this slice | Backend verified; frontend gap noted |
| W2E-007 | Workspace artifacts membership | Workspace artifacts attach/list/update/remove/export with lineage and review-state constraints | `test_workspace_sub_resources_api.py`, `test_workspace_membership_adapters.py`, `test_workspace_memberships_api.py`; backend suite `461 passed` | Research Workspace Studio artifact E2E not rerun in this slice | Backend verified; frontend gap noted |
| W2E-008 | Chat membership | Chat/conversation membership gates active-context use without hiding global chat visibility | `test_workspace_membership_adapters.py`, `test_workspace_memberships_api.py`, `test_workspace_context_membership_summary.py`; backend suite `461 passed` | Research Workspace grounded chat/search E2E not rerun in this slice | Backend verified; frontend gap noted |
| W2E-009 | Prompt membership | Prompt adapter canonicalizes allowed prompt identifiers and exposes safe summaries without prompt bodies | `test_workspace_membership_adapters.py`, `test_workspace_memberships_api.py`; backend suite `461 passed` | Route metadata coverage included in focused Vitest; Prompt Workspace E2E not rerun | Backend verified; frontend partial |
| W2E-010 | Workflow membership | Workflow adapter enforces tenant/owner/admin visibility and safe summaries | `test_workspace_membership_adapters.py`, `test_workspace_memberships_api.py`; backend suite `461 passed` | No dedicated Workflows workspace frontend contract in this slice; evidence is backend/API only | Backend verified; frontend gap noted |
| W2E-011 | Watchlist membership | Watchlist adapter validates current-user watchlists and safe summaries for active/deleted state | `test_workspace_membership_adapters.py`, `test_workspace_memberships_api.py`; backend suite `461 passed` | No dedicated Watchlists workspace frontend contract in this slice; evidence is backend/API only | Backend verified; frontend gap noted |
| W2E-012 | ACP session membership | ACP session adapter validates active Workspace runtime binding descriptors and does not grant execution/path trust | `test_workspace_runtime_bindings.py`, `test_workspace_runtime_bindings_api.py`, `test_workspace_membership_adapters.py`, `test_workspace_memberships_api.py`; backend suite `461 passed` | `ACPWorkspacePanel.test.tsx`, `WorkspaceHeader.test.tsx`, route metadata focused Vitest `79 passed` | Verified |
| W2E-013 | Sandbox session membership | Sandbox session adapter validates active Workspace runtime binding descriptors and leaves admission to Sandbox | `test_workspace_runtime_bindings.py`, `test_workspace_runtime_bindings_api.py`, `test_workspace_membership_adapters.py`, `test_workspace_memberships_api.py`; backend suite `461 passed` | Sandbox-owned visible loop not covered by current frontend matrix | Backend verified; frontend gap noted |
| W2E-014 | Active workspace selection and active-context gates | Visibility operations pass after domain permissions; active operations fail closed for no workspace, missing/archived workspace, unsupported type, unlinked/cross-workspace resource, missing runtime, or permission denial | `test_workspace_eligibility.py`, `test_workspace_eligibility_api.py`; backend suite `461 passed` | Frontend eligibility UX contract pending future/client-specific coverage | Backend verified; frontend gap noted |
| W2E-015 | Global browse/search/open remains visible | Workspace selection does not hard-filter Library, Notes, Artifact, Chat, Prompt, Workflow, or Watchlist owner surfaces | Eligibility visibility tests; membership adapter/API tests confirm association-only model; backend suite `461 passed` | Research Workspace global search and owner-surface UI checks pending rerun/manual evidence | Backend verified; manual/frontend pending |
| W2E-016 | Cross-workspace/unlinked recovery | Cross-workspace resource returns copy/switch recovery; unlinked resource returns link recovery with stable reason codes | `test_workspace_eligibility.py`, `test_workspace_eligibility_api.py`; backend suite `461 passed` | Recovery copy in UI pending client-specific evidence | Backend verified; frontend pending |
| W2E-017 | Missing runtime and archived workspace behavior | Runtime-bound operations require `runtime_state="ready"`; archived workspaces block active operations and membership mutation | `test_workspace_eligibility.py`, `test_workspace_runtime_bindings.py`, `test_workspaces_api.py`; backend suite `461 passed` | Manager/project-root focused Vitest `79 passed`; detailed archive browser flow not rerun | Backend verified; frontend partial |
| W2E-018 | Docs/API/frontend contracts match implementation | Canonical contract lists current supported adapters and public API routes; matrix names current deferrals | This document plus `Workspace_Container_Contract_2026_06.md`; `git diff --check` passed | Focused Vitest `79 passed`; browser manager smoke `2 passed` | Verified |
| W2E-019 | Visible single-user loop | User can open manager, move to Research Workspace, and return to manager in single-user mode | Backend suite `461 passed`; local API server single-user mode passed health/auth-dependent UI calls during browser smoke | `workspaces-manager.spec.ts` passed with `2 passed`; create/select specifics are covered by focused manager tests and backend API tests | Verified for branch smoke |
| W2E-020 | Final epic evidence | Final evidence summary with limitations is posted to #1984 and #1995 | [#1995 evidence comment](https://github.com/rmusser01/tldw_server/issues/1995#issuecomment-4738013294) | [#1984 evidence comment](https://github.com/rmusser01/tldw_server/issues/1984#issuecomment-4738013358) | Verified |

## Verification Commands

- Frontend dependencies: `bun install --frozen-lockfile` from `apps/`.
- Focused frontend contract suites:
  - `apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspacesManagerPage.test.tsx`
  - `apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspaceProjectRootPanel.test.tsx`
  - `apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspaceReconciliationPanel.test.tsx`
  - `apps/packages/ui/src/components/Option/Workspaces/__tests__/workspace-local-reconciliation.test.ts`
  - `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`
  - `apps/packages/ui/src/components/Option/MCPHub/__tests__/SharedWorkspacesTab.test.tsx`
  - `apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPWorkspacePanel.test.tsx`
  - `apps/packages/ui/src/routes/__tests__/option-workspaces.route.test.tsx`
  - `apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts`
  - `apps/tldw-frontend/__tests__/navigation/workspaces-page-wrapper.test.ts`
- Focused browser evidence:
  - `apps/tldw-frontend/e2e/workflows/workspaces-manager.spec.ts`
- `git diff --check`.

## Remaining Explicit Gaps

- `research-workspace.spec.ts` and `research-workspace.parity.spec.ts` were not
  rerun in this branch after the manager smoke passed. Their source/chat/studio
  browser flows remain useful follow-up evidence, but the release-blocking
  backend/API contracts and route/manager handoffs are covered above.
- No production backend code was changed in this branch. Bandit is recorded as
  not applicable for this evidence/test/docs-only change set.
