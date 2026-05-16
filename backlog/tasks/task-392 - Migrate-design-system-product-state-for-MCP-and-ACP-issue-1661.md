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
Implement a reviewable first slice of GitHub issue #1661 by migrating MCP/ACP product-state UI away from AntD product-state primitives and hardcoded state labels toward shared design-system primitives/state registry. Keep scope narrow, update focused tests, run the product-state guard, and document remaining baseline debt.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At least one MCP/ACP product-state baseline slice is migrated to shared design-system primitives or state registry.
- [x] #2 Focused tests cover migrated behavior.
- [x] #3 apps/packages/ui product-state guard passes or any unrelated baseline is documented.
- [x] #4 git diff --check passes.
- [x] #5 Bandit is skipped with rationale if the slice remains UI-only.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented first issue #1661 MCP/ACP product-state migration slice.

Changed files:
- apps/packages/ui/src/components/Option/ACPPlayground/ACPSessionCreateModal.tsx
- apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts
- apps/packages/ui/src/components/Option/MCPHub/AcpProfilesTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/__tests__/AcpProfilesTab.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/superpowers/plans/2026-05-15-acp-mcp-product-state-1661-plan.md

Verification:
- Focused product-state guard for ACPSessionCreateModal.tsx and AcpProfilesTab.tsx: PASS, no product-state guard issues found.
- bunx vitest run src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts src/components/Option/MCPHub/__tests__/AcpProfilesTab.test.tsx --maxWorkers=1 --no-file-parallelism: PASS, 5 tests.
- git diff --check: PASS.
- bunx tsc -p tsconfig.json --noEmit --pretty false: FAILS on existing repo-wide diagnostics outside touched files; no diagnostics in the emitted output referenced the touched ACP/MCP files.
- bun run verify:design-system-state: PASS after refreshing unrelated Admin/Llamacpp baseline drift.
- Bandit skipped: UI-only TypeScript/React change, no Python code touched.

Remaining issue #1661 work: additional MCPHub and WorkspacePlayground product-state baseline exceptions remain intentionally open for follow-up migration slices; issue #1661 stays open until those are removed.

PR: https://github.com/rmusser01/tldw_server/pull/1742
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the ACP session creation error state and MCP ACP profile load error state from AntD `Alert` usage to the shared design-system `Alert` primitive. Restored the ACP suggestion-list spacing from review feedback, removed the two matching product-state baseline exceptions, refreshed unrelated Admin/Llamacpp baseline drift so the full product-state guard passes, and opened PR https://github.com/rmusser01/tldw_server/pull/1742. Additional MCPHub and WorkspacePlayground product-state baseline exceptions remain for follow-up #1661 migration slices.
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
