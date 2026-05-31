---
id: TASK-418.10.8
title: Address PR 1986 smoke auth review comments
status: Done
labels:
- wp12
- webui
- route-governance
- review-comments
priority: High
parent_task_id: TASK-418.10
references:
- https://github.com/rmusser01/tldw_server/pull/1986
- TASK-418.10.7
documentation:
- Docs/superpowers/plans/2026-05-17-webui-route-governance-qa-implementation-plan.md
modified_files:
- apps/tldw-frontend/e2e/utils/e2e-auth.ts
- apps/tldw-frontend/__tests__/e2e/e2e-auth.test.ts
- apps/tldw-frontend/e2e/smoke/smoke.setup.ts
- apps/tldw-frontend/e2e/utils/helpers.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address review comments on PR #1986 about the smoke-test API key fallback: centralize the local E2E placeholder, fail fast for remote server URLs without an explicit key, update evidence, and keep the WP12 final governance fix scoped to frontend smoke/governance code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Smoke and shared E2E helpers use one auth fallback resolver.
- [x] #2 Local E2E placeholder fallback is documented as non-secret and limited to local server URLs.
- [x] #3 Non-local E2E server URLs fail fast without an explicit API key.
- [x] #4 Focused coverage and affected smoke gates are run and recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Addressed PR #1986 review feedback by extracting E2E auth fallback handling into a single shared helper. The default E2E key remains only as a documented non-secret local placeholder, and remote/non-local server URLs now fail fast unless `TLDW_API_KEY`, `TLDW_E2E_API_KEY`, or `SINGLE_USER_API_KEY` is explicitly provided.

Updated smoke setup and shared E2E helpers to use the same resolver. Added Vitest coverage for explicit env precedence, local fallback, localhost detection, and remote fail-fast behavior.

Bandit was not run because the touched files are TypeScript/TSX frontend/E2E code only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #1986 review comments addressed. Qodo/Gemini hardcoded-key feedback was handled by centralizing the placeholder in `apps/tldw-frontend/e2e/utils/e2e-auth.ts`, documenting it as a non-secret local E2E fixture value, and guarding against accidental use with non-local server URLs. The duplicated fallback in `smoke.setup.ts` and `helpers.ts` was removed.

Tests run:
- `bunx vitest run __tests__/e2e/e2e-auth.test.ts` => 1 file / 4 tests passed
- `bunx playwright test e2e/smoke/all-pages.spec.ts --reporter=line --grep "Smoke Tests - All Pages.*Home" --workers=1` => 1 passed
- `bun run e2e:smoke:stage4` => 29 passed, 1 skipped
- `bun run e2e:smoke:route-governance` => 18 passed
- `bun run e2e:smoke:all-pages:gate` => 123 passed
- `git diff --check` => passed

Known skips: existing Stage 4 single skipped test.

Deferred backend dependencies: none.
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
