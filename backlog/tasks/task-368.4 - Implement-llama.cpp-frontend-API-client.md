---
id: TASK-368.4
title: Implement llama.cpp frontend API client
status: Done
assignee: []
created_date: '2026-05-15 03:44'
updated_date: '2026-05-15 15:01'
labels:
  - implementation
  - frontend
  - llamacpp
dependencies:
  - TASK-368.3
documentation:
  - Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-llamacpp-server-management-webui-implementation-plan.md
parent_task_id: TASK-368
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the frontend API client and type slice from the implementation plan. Add TypeScript admin types and facade client methods for config validation inventory registration start-by-model use-in-chat log tail and hardware snapshot. Do not reshape the page UI in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared TypeScript types exist for the llama.cpp admin facade contracts.
- [x] #2 TldwApiClient and the models audio domain client expose the new llama.cpp admin facade methods consistently.
- [x] #3 Client ownership metadata is updated for the new methods.
- [x] #4 Existing llama.cpp admin page tests still pass before page reshape.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Task 4 frontend API client/types slice after TASK-368.3 was finalized and committed at a10bd4376. Scope is limited to `apps/packages/ui/src/types/llamacpp-admin.ts`, `apps/packages/ui/src/services/tldw/TldwApiClient.ts`, `apps/packages/ui/src/services/tldw/domains/models-audio.ts`, and `apps/packages/ui/src/services/tldw/client-ownership.ts`; no page reshape in this task.

Task 4 implemented in 47f2bbdfa with shared llama.cpp admin TypeScript contracts and mirrored facade methods on both `TldwApiClient.ts` and `domains/models-audio.ts`. Ownership metadata was updated for the transitional overlap. Code-quality follow-up 12b5bcc46 tightened `startLlamacppModel` from `any` to `Record<string, unknown>`.

Verification recorded: `bunx vitest run ../packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx --config vitest.config.ts` from `apps/tldw-frontend` passed with 5 tests. `bunx vitest run ../packages/ui/src/services/__tests__/tldw-api-client.ownership-guard.test.ts --config vitest.config.ts` from `apps/tldw-frontend` passed with 1 test. `git diff --check` passed. Broad package typecheck was attempted with local TypeScript 5.9.3 via `apps/packages/ui/node_modules/.bin/tsc --noEmit -p apps/packages/ui/tsconfig.json --pretty false`; it failed on unrelated baseline errors across existing tests/components/services, not on the new llama.cpp client files. A transient `bunx tsc` run also pulled a newer TypeScript and failed on the repo's existing `baseUrl` deprecation gate before using the local binary. Bandit skipped because this slice touched frontend TypeScript only.

Review status: spec compliance review approved at 47f2bbdfa. Code-quality review approved with one P3 type cleanup, addressed in 12b5bcc46. Residual risk: direct request-shape unit tests for the new client methods were not added; coverage is through endpoint-path review, ownership guard, and the existing page smoke test before the page reshape.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the frontend API contract layer for the llama.cpp admin facade. The UI package now has shared admin types plus mirrored client methods for config, validation, inventory, path registration, start-by-model, explicit chat wiring, log tailing, and hardware snapshot access. The legacy llama.cpp client methods remain intact, ownership metadata is updated, and the focused frontend checks pass. Broad package typecheck remains blocked by existing unrelated baseline errors.
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
