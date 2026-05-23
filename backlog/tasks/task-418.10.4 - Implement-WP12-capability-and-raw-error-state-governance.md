---
id: TASK-418.10.4
title: Implement WP12 capability and raw error state governance
status: Done
labels:
- wp12
- webui
- route-governance
- e2e
priority: High
ordinal: 4
parent_task_id: TASK-418.10
references:
- TASK-418.10
- Docs/superpowers/plans/2026-05-17-webui-route-governance-qa-implementation-plan.md
- https://github.com/rmusser01/tldw_server/pull/1963
- https://github.com/rmusser01/tldw_server/pull/1970
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP12 Task 4 from the WebUI route governance QA plan: add capability-state and raw-error governance for representative WebUI routes, and add smoke allowlist discipline without page-level redesign or backend API changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Capability governance covers the planned representative routes and asserts user-language diagnosis plus recovery actions instead of raw endpoint errors as primary UI.
- [x] #2 Smoke allowlist entries require stable ids, scope, owner, rationale, and expiry discipline.
- [x] #3 Focused Playwright capability governance checks pass, with unrelated baseline failures documented.
- [x] #4 Backlog task records touched files, verification, known skips, and final summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation completed for WP12 Task 4. Added route capability/raw-error governance smoke coverage, smoke allowlist metadata validation, shared collapsed diagnostics, and scoped route recovery-state fixes for model settings, evaluations, MCP hub, skills, TTS/speech, and data tables. Verification so far: focused route-governance Playwright passed on isolated port 18080; focused all-pages allowlist metadata Playwright passed; focused Vitest passed for state primitives, EvaluationRecoveryCallout, and ToolCatalogsTab; Bandit scanned touched frontend paths with 0 findings/0 Python LOC. Full frontend tsc remains blocked by unrelated baseline TypeScript errors in Media read-along, Watchlists, WorkspacePlayground, keyboard shortcuts, persona live control, and admin llama.cpp e2e fixtures.

PR: https://github.com/rmusser01/tldw_server/pull/1970

Touched files:
- apps/tldw-frontend/e2e/smoke/route-capability-state-governance.spec.ts
- apps/tldw-frontend/e2e/smoke/smoke.setup.ts
- apps/tldw-frontend/e2e/smoke/all-pages.spec.ts
- apps/packages/ui/src/components/ui/state/StatePanel.tsx
- apps/packages/ui/src/components/ui/state/__tests__/state-primitives.test.tsx
- apps/packages/ui/src/components/Option/Models/AvailableModelsList.tsx
- apps/packages/ui/src/components/Option/Evaluations/components/EvaluationRecoveryCallout.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx
- apps/packages/ui/src/components/Option/MCPHub/ToolCatalogsTab.tsx
- apps/packages/ui/src/components/Option/Skills/SkillsWorkspace.tsx
- apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/DataTables/DataTablesList.tsx
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented WP12 capability/raw-error governance for representative root routes. Raw endpoint and `Not Found (GET ...)` details are no longer primary UI in the covered states; technical details are disclosed via diagnostics/request-details controls, and affected pages provide retry/setup/settings recovery actions. The smoke allowlist now enforces id/scope/pattern/rationale/owner/expiry metadata and has a targeted all-pages metadata check.
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
