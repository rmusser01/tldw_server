---
id: TASK-368.4
title: Implement llama.cpp frontend API client
status: Done
assignee:
  - Codex
created_date: '2026-05-15 03:44'
updated_date: '2026-05-29 04:38'
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
references:
  - https://github.com/rmusser01/tldw_server/pull/1727
  - https://github.com/rmusser01/tldw_server/pull/1764
  - https://github.com/rmusser01/tldw_server/pull/1836
  - https://github.com/rmusser01/tldw_server/pull/2120
modified_files:
  - apps/packages/ui/src/types/llamacpp-admin.ts
  - apps/packages/ui/src/services/tldw/TldwApiClient.ts
  - apps/packages/ui/src/services/tldw/domains/models-audio.ts
  - apps/packages/ui/src/services/tldw/client-ownership.ts
  - apps/packages/ui/src/services/__tests__/tldw-api-client.models-normalization.test.ts
  - apps/packages/ui/src/services/__tests__/tldw-api-client.ownership-guard.test.ts
  - apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
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

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
This is mostly a stale tracker closeout after PR #2120 merged, with one metadata fix made in this branch. The llama.cpp frontend API client, shared TypeScript types, and admin page test coverage are already present on current `origin/dev`; this branch refreshes `TRANSITIONAL_DOMAIN_OVERLAPS` so the ownership guard reflects the actual client/domain overlap inventory.

Implementation provenance:
- PR #1727 (`726958be39 Improve llama.cpp WebUI server management`) added initial llama.cpp admin TypeScript types, `TldwApiClient` methods, `models-audio` domain methods, and admin page test coverage for config, validation, start, use-in-chat, logs, and hardware.
- PR #1764 (`560c8e17b3 Implement llama.cpp asset inventory v2`) added inventory/register/start-by-model frontend contracts and admin page tests.
- PR #1836 (`8616b27754 Add llama.cpp acquisition workflow UI`) extended the llama.cpp frontend contracts and client/domain method coverage for asset acquisition workflows.

Verified behavior:
- Shared `@/types/llamacpp-admin` contracts cover config, validation, inventory, asset, start, use-in-chat, log tail, hardware, profile, runtime, and acquisition responses.
- `TldwApiClient` and `models-audio` expose matching facade methods for the llama.cpp admin APIs in this task scope.
- `apps/packages/ui/src/services/tldw/client-ownership.ts` now includes the current llama.cpp overlap methods required by the guard.
- Existing llama.cpp admin page tests pass before any page reshape work.

Verification commands:
- Initial root-level `bunx vitest run ...` failed before collecting tests because the fresh worktree did not have the package-local alias/dependency setup.
- `bun install` from `apps/` repaired the frontend dependency symlinks without changing tracked files.
- `bun run test src/services/__tests__/tldw-api-client.models-normalization.test.ts src/services/__tests__/tldw-api-client.ownership-guard.test.ts src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx`
- Result: 3 files passed, 32 tests passed in 10.97s. Node emitted expected localStorage experimental warnings.

Known skips:
- Bandit was not run because this branch changes TypeScript/Backlog metadata only and no Python runtime code.
<!-- SECTION:NOTES:END -->

## Final Summary
<!-- SECTION:SUMMARY:BEGIN -->
Closed `TASK-368.4` against the frontend client/type implementation already merged into `dev`, and refreshed the client ownership overlap metadata so the focused TypeScript guard matches the current facade surface.
<!-- SECTION:SUMMARY:END -->
