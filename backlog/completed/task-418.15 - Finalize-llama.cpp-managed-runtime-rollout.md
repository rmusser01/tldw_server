---
id: TASK-418.15
title: Finalize llama.cpp managed runtime rollout
status: Done
labels:
- llamacpp
- backend
- webui
- docs
- e2e
priority: medium
parent_task_id: TASK-418
documentation:
- Docs/superpowers/plans/2026-05-17-llamacpp-managed-runtime-implementation-plan.md
- Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md
modified_files:
- Docs/API-related/llamacpp_integration_modes.md
- Docs/User_Guides/Integrations_Experiments/Setting_up_a_local_LLM.md
- Docs/superpowers/plans/2026-05-17-llamacpp-managed-runtime-implementation-plan.md
- apps/tldw-frontend/e2e/workflows/llamacpp-runtime-admin.spec.ts
- backlog/completed/task-418.15 - Finalize-llama.cpp-managed-runtime-rollout.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 6 from the llama.cpp managed runtime closeout plan: add final documentation and E2E smoke coverage, run focused backend/frontend/security verification, and update rollout tracking without changing runtime semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 llama.cpp docs describe managed runtime profiles, default-profile compatibility, local import/register, mmproj pairing, autostart/restart limitations, and deferred remote catalogs/downloads.
- [x] #2 E2E smoke coverage exercises the llama.cpp runtime admin page with mocked backend responses for assets, profiles, runtimes, warnings, and stopped-profile Chat wiring controls.
- [x] #3 Focused backend and frontend verification are run and recorded, with any environment-limited E2E blocker documented.
- [x] #4 Bandit and git diff checks are run for the touched scope.
- [x] #5 Parent rollout tracking is updated with final summary and known skips/blockers.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closeout implementation completed for Task 6.

Implemented:
- Added mocked Playwright smoke coverage for `/admin/llamacpp` managed runtime assets, profiles, runtime instances, warnings, and running-only Chat wiring.
- Updated source llama.cpp docs with managed runtime profiles, default-profile compatibility, local register/import semantics, mmproj pairing, bounded autostart/restart behavior, and deferred download/catalog boundaries. `Docs/Published` files are generated and intentionally left untouched.
- Updated the managed runtime implementation plan final Task 6 evidence and checklist.

Verification:
- `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bunx playwright test e2e/workflows/llamacpp-runtime-admin.spec.ts --reporter=line` from `apps/tldw-frontend`: 1 passed.
- `bunx vitest run src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx src/components/Option/Admin/__tests__/LlamacppProfilesPanel.test.tsx src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx` from `apps/packages/ui`: 42 passed.
- Focused backend pytest set for llama.cpp profile store, supervisor, runner, runtime API, provider/logs API, profile capabilities, model metadata, and AuthNZ claims: 163 passed, 5 warnings.
- `python -m bandit -r ... -f json -o /tmp/bandit_llamacpp_runtime_rollout.json`: no findings.
- `git diff --check`: passed.

Known notes:
- The Playwright smoke requires `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000` so the frontend dev server satisfies advanced-mode networking validation.
- Remote downloads/catalogs remain deferred to the acquisition workflow and were not implemented in this runtime closeout.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 6 closeout is complete. The llama.cpp docs now describe backend-owned managed runtime profiles, default-profile compatibility, local register/import behavior, explicit mmproj pairing, bounded autostart/restart behavior, and deferred acquisition boundaries. A mocked Playwright smoke covers the admin runtime page across assets, profiles, runtime states, warnings, and running-only Chat wiring. Focused frontend/backend tests, E2E smoke, Bandit, and diff checks all passed; the only environment note is that the Playwright dev server needs `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000` in advanced deployment mode.
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
