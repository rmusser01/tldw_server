---
id: TASK-418.16
title: Address PR 1854 llama.cpp rollout review comments
status: Done
labels:
- llamacpp
- pr-review
- docs
- e2e
priority: medium
parent_task_id: TASK-418
references:
- https://github.com/rmusser01/tldw_server/pull/1854
- https://github.com/rmusser01/tldw_server/pull/1854#pullrequestreview-4314972815
documentation:
- Docs/superpowers/plans/2026-05-17-llamacpp-managed-runtime-implementation-plan.md
modified_files:
- Docs/API-related/llamacpp_integration_modes.md
- apps/tldw-frontend/e2e/workflows/llamacpp-runtime-admin.spec.ts
- backlog/completed/task-418.16 - Address-PR-1854-llama.cpp-rollout-review-comments.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the still-valid Gemini review comments on PR #1854 for the llama.cpp rollout closeout docs and E2E smoke test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Still-valid PR #1854 review comments are addressed or explicitly skipped with reason.
- [x] #2 Docs/Published feedback is skipped as obsolete because generated files are not in the PR diff.
- [x] #3 Focused verification for changed docs/E2E files is run and recorded.
- [x] #4 Changes are committed and pushed to PR #1854.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review sweep:
- Gemini source-doc readability comment addressed by converting the managed profile attribute sentence into a short bullet list.
- Gemini E2E byte-size comment addressed with `MB` and `GB` constants in the smoke fixture.
- Gemini E2E `use-in-chat` mock parsing comment addressed by returning a 400 mock response when the expected profile ID is missing instead of recording an empty string.
- Gemini `Docs/Published` comment skipped as obsolete because generated published docs are not in the PR diff.
- Qodo flaky chat-wiring assertion comment addressed by waiting for the profile-scoped `use-in-chat` POST response before asserting the recorded profile id.
- CodeRabbit `use-in-chat` mock method-validation comment addressed by returning a 405 mock response for non-POST requests.
- CodeRabbit polling comment addressed by polling `api.useInChatProfileIds` after the profile-scoped POST response completes.

Verification:
- `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bunx playwright test e2e/workflows/llamacpp-runtime-admin.spec.ts --reporter=line` from `apps/tldw-frontend`: 1 passed.
- `git diff --check`: passed.
- `git diff --name-status origin/dev...HEAD`: confirmed no `Docs/Published` files in the branch diff.
- Bandit skipped because this review-fix slice changes docs, E2E TypeScript, and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #1854 review comments are addressed. The source docs now use a scannable attribute list, the E2E fixture uses byte-size constants, the `use-in-chat` mock fails fast on unexpected URLs and non-POST requests, and the smoke waits for the async Chat wiring request before polling the recorded mock state. The generated `Docs/Published` feedback was skipped as obsolete because those files are not in the PR diff.
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
