---
id: TASK-12042
title: Adopt Skills manager list-load recovery state
status: Done
created_date: 2026-06-26 06:15
references:
- TASK-420
- TASK-418.10.4
- TASK-12041
documentation:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
modified_files:
- Docs/superpowers/plans/2026-06-26-webui-stage13-skills-list-load-recovery-plan.md
- apps/packages/ui/src/components/Option/Skills/Manager.tsx
- apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx
updated_date: 2026-06-26 06:22
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred WebUI capability/error-state follow-up for the Skills manager list-load failure state. Replace the plain list-load error banner with the shared recovery-state pattern and non-secret endpoint diagnostics while preserving Skills manager actions, empty states, retry behavior, filters, and table behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skills list-load failures render through the shared RecoveryCallout primitive.
- [x] #2 The primary copy stays user-facing while request method/path/status/raw message are available only in diagnostics.
- [x] #3 The retry action still refetches the skills list.
- [x] #4 Beginner empty, filter empty, and successful list states remain unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing Skills manager test proving list-load failure uses the shared recovery callout with diagnostics and retry.
2. Implement the minimal Manager.tsx change using the shared buildCapabilityState and RecoveryCallout helpers.
3. Run the focused Vitest suite, direct ESLint on touched TS/TSX files, and whitespace diff checks.
4. Record verification and finalize the Backlog task before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Skills manager list-load recovery state.

Verification:
- RED: `bun run test:run ../packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx` failed on the new RecoveryCallout expectation while the component still rendered the old Alert. The same run also exposed an existing column menu helper issue.
- Root cause for menu helper: the AntD multiple-selection column menu stays open after choosing an item, so the helper's second trigger click closed it before waiting for the menu.
- GREEN: `bun run test:run ../packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx` passed with 24 tests.
- Direct ESLint: `bun apps/node_modules/.bun/eslint@9.39.2+288993669ddeca06/node_modules/eslint/bin/eslint.js -c apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/Skills/Manager.tsx apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx` exited 0 with only the known Next pages-directory notice.
- `git diff --check` passed.
- Design-state guard: `bun scripts/verify-design-system-product-state.mjs` could not start because this worktree's shared UI install cannot resolve `typescript` from `apps/packages/ui/scripts/design-system-product-state-rules.mjs`.
- Bandit: not applicable; touched scope is TS/TSX and markdown only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced the Skills manager list-load error banner with the shared RecoveryCallout and capability diagnostics so the main message stays user-facing while method/path/status/raw failure details are available under diagnostics. Preserved retry/refetch behavior, tightened existing Skills manager error handler types to remove touched-file lint warnings, and fixed the existing column-menu test helper to match the multiple-selection dropdown's open-state behavior.
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
