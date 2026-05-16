---
id: TASK-392
title: Migrate design-system product state for MCP and ACP issue 1661
status: Done
labels:
- design-system
- product-state
- mcp
- acp
- webui
priority: high
references:
- https://github.com/rmusser01/tldw_server/issues/1661
- https://github.com/rmusser01/tldw_server/issues/1655
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1661's owned MCP/ACP/Workspace product-state baseline cleanup by migrating scoped product-state UI away from AntD product-state primitives and hardcoded state labels toward shared design-system primitives/state registry. Keep AntD mechanics in place, update focused tests, and run the product-state guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCPHub, ACPPlayground, and WorkspacePlayground product-state baseline entries are removed after migration to shared design-system primitives or state registry.
- [x] #2 Focused tests cover migrated behavior.
- [x] #3 apps/packages/ui product-state guard passes or any unrelated baseline is documented.
- [x] #4 git diff --check passes.
- [x] #5 Bandit is skipped with rationale if the slice remains UI-only.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented issue #1661 MCP/ACP/Workspace product-state migration for the owned baseline scope, including the Qodo top-level baseline review gap.

Changed files:
- apps/packages/ui/src/components/Option/ACPPlayground/ACPSessionCreateModal.tsx
- apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts
- apps/packages/ui/src/components/Option/productStatePrimitives.tsx
- apps/packages/ui/src/components/Option/MCPHub/AcpProfilesTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/ApprovalPoliciesTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/CapabilityMappingsTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/ExternalAccessSummary.tsx
- apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/GovernanceAuditTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/GovernancePacksTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx
- apps/packages/ui/src/components/Option/MCPHub/PathScopesTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/PermissionProfilesTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/PersonaPolicySummary.tsx
- apps/packages/ui/src/components/Option/MCPHub/PolicyAssignmentsTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/PolicyDocumentEditor.tsx
- apps/packages/ui/src/components/Option/MCPHub/SharedWorkspacesTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/ToolCatalogsTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/WorkspaceSetsTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/__tests__/AcpProfilesTab.test.tsx
- apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/AddSourceModal.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/TransferSourcesModal.tsx
- apps/packages/ui/src/design-system/__tests__/product-state-guard.mcp-acp-workspace-baseline.test.ts
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/superpowers/plans/2026-05-15-acp-mcp-product-state-1661-plan.md

Verification:
- Focused product-state guard for ACPSessionCreateModal.tsx and AcpProfilesTab.tsx: PASS, no product-state guard issues found.
- bunx vitest run src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts src/components/Option/MCPHub/__tests__/AcpProfilesTab.test.tsx --maxWorkers=1 --no-file-parallelism: PASS, 5 tests.
- bunx vitest run src/design-system/__tests__/product-state-guard.mcp-acp-workspace-baseline.test.ts --maxWorkers=1 --no-file-parallelism: PASS, 1 test.
- bunx vitest run src/components/Option/MCPHub/__tests__/*.test.tsx src/components/Option/WorkspacePlayground/__tests__/AddSourceModal.stage9.error.test.tsx src/components/Option/WorkspacePlayground/__tests__/SourcesPane.stage5.transfer.test.tsx src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage13.source-transfer.test.tsx src/design-system/__tests__/product-state-guard.mcp-acp-workspace-baseline.test.ts --maxWorkers=1 --no-file-parallelism: PASS, 95 tests across 21 files.
- git diff --check: PASS.
- bunx tsc -p tsconfig.json --noEmit --pretty false: FAILS on existing repo-wide diagnostics outside touched files; filtered temp log found no diagnostics for the touched MCPHub, WorkspacePlayground, product-state adapter, baseline guard, or baseline JSON paths.
- bun run verify:design-system-state: PASS after refreshing unrelated Admin/Llamacpp baseline drift.
- Scoped baseline count for MCPHub, ACPPlayground, and WorkspacePlayground: 0.
- Bandit skipped: UI-only TypeScript/React change, no Python code touched.

Review note: Qodo's "AntD Alert replaced" requirement gap conflicts with issue #1661's note to keep AntD only where it is mechanics while migrating product-state language to shared primitives/state registry. The direct shared `Alert` imports in the original ACP/MCP files were moved behind the product-state adapter so the component call sites keep AntD-like `type`/`title` mechanics while rendering through the design-system primitive.

PR: https://github.com/rmusser01/tldw_server/pull/1742
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the owned ACP/MCP/Workspace product-state scope away from AntD product-state primitives while preserving AntD mechanics. Added a shared product-state adapter backed by design-system `Alert` and `Badge`, converted flagged MCPHub and WorkspacePlayground alert/badge/empty states, added a regression guard for zero scoped baseline entries, removed all 41 scoped baseline exceptions, and verified the focused MCPHub/Workspace suite plus the full product-state guard for PR https://github.com/rmusser01/tldw_server/pull/1742.
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
