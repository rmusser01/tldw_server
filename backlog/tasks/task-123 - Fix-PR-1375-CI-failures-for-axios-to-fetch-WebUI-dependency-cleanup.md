---
id: TASK-123
title: 'Fix PR #1375 CI failures for axios-to-fetch WebUI dependency cleanup'
status: In Progress
assignee:
  - Codex
created_date: '2026-05-08 13:29'
updated_date: '2026-05-08 17:25'
labels:
  - ci
  - webui
  - notes
  - admin
  - github-actions
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1375'
  - >-
    https://github.com/rmusser01/tldw_server/actions/runs/25534945958/job/74948786462
  - >-
    https://github.com/rmusser01/tldw_server/actions/runs/25534945965/job/74948785552
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the current PR #1375 GitHub Actions failures after the axios-to-fetch WebUI dependency cleanup. Scope covers the failing ElevenLabs timeout smoke path, Notes backend remediation mock pagination failures, and the follow-on full-suite Admin module timeout that blocks CI completion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ElevenLabs fetch service normalizes timeout-style browser fetch failures so TTS and Speech retry alerts show the timeout body.
- [x] #2 Focused ElevenLabs service or hook tests cover the timeout normalization path.
- [x] #3 Notes backend integration tests set concrete pagination count values instead of passing MagicMock totals to pagination metadata.
- [x] #4 Targeted frontend and backend tests for the changed files pass locally.
- [x] #5 Admin route tests that do not need the full app lifespan avoid TestClient startup/shutdown hangs and pass in the full Admin module run.
- [x] #6 CI Admin steps bypass the global xdist `PYTEST_ADDOPTS` setting and run serially to avoid worker-level lifespan hangs.
- [x] #7 actionlint accepts the repository's self-hosted `vz-linux` runner label.
- [x] #8 PR #1375 branch is pushed after fixes are verified.
- [x] #9 Unresolved PR review comments on the fetch-backed API client are fixed with focused regression coverage.
- [x] #10 The ElevenLabs speech request path uses the shared fetch helper without changing the public speech response contract.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm failure scope from PR #1375 checks and keep non-actionable full-suite queue/cancellation noise separate.
2. Frontend fix: add focused ElevenLabs service coverage for browser fetch timeout-style failures that are not DOM AbortError from our own timer, then normalize those errors to the existing "ElevenLabs request timed out" message used by the UI timeout classifier.
3. Notes test fix: update the two failing mocked integration tests to provide concrete count return values for pagination, matching the endpoint's count contract and avoiding MagicMock totals.
4. Smoke-gate fix: update E2E auth seeding to set the current `assistant_setup_dismissed` first-run overlay flag alongside the legacy first-run flag so TTS/Speech smoke tests reach the audio pages.
5. Verify locally with the focused ElevenLabs Vitest file, related TTS provider/Speech page tests, the two failing Notes pytest tests, focused Playwright smoke repro, git diff whitespace check, and Bandit on touched Python test scope.
6. Investigate the remaining full-suite Admin timeout separately from the original WebUI/Notes failures. Reproduce the hang locally with pytest-timeout and reduce it to route-level Admin tests that enter full TestClient lifespan startup/shutdown despite not needing lifespan behavior.
7. Convert BYOK validation and Admin conflict route tests to `httpx.ASGITransport` + `AsyncClient`, keeping real FastAPI routing/dependency behavior while bypassing the expensive app lifespan path that stalls in the full Admin suite.
8. Re-run focused Admin files, the full Admin module locally, whitespace checks, and Bandit on the newly touched Admin test files.
9. Update CI so the Admin module clears the global xdist `PYTEST_ADDOPTS` while the rest of the heavy suite keeps xdist.
10. Fix the fresh actionlint failure by configuring the self-hosted runner labels used by the VZ Linux host-gated workflow.
11. Stage, commit, push the PR branch, then refresh PR #1375 checks and record results in the Backlog task.
12. Address the current PR review surface: default header merging, per-request auth header preservation, protocol-relative URLs, explicit JSON responseType parsing, error-body parsing before success responseType handling, session header normalization, and ElevenLabs helper consolidation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root-cause evidence: Notes backend remediation failed because count_notes/count_keyword_collections mocks returned MagicMock values, which build_offset_pagination_meta compared against ints. UX Smoke Gate failed on ElevenLabs timeout retry assertions after the axios-to-fetch rewrite; the fetch service only maps our own AbortError timeout and passes route-aborted browser fetch failures through as generic errors.

Verification so far: `bunx vitest run src/services/__tests__/elevenlabs.test.ts` passed 6 tests after the red test initially failed as expected; `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes/test_notes_api_integration.py::test_list_notes tldw_Server_API/tests/Notes/test_notes_api_integration.py::test_list_keyword_collections_with_keywords` passed 2 tests after reproducing both failures; `bunx vitest run src/services/__tests__/elevenlabs.test.ts src/hooks/__tests__/useTtsProviderData.test.tsx src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx` passed 21 tests; focused Playwright `bunx playwright test e2e/smoke/stage7-audio-regression.spec.ts --grep "ElevenLabs timeout" --reporter=line --workers=1` passed 2 tests after fixing E2E seedAuth; `git diff --check` passed; Bandit on the touched Notes test file with `-s B101` produced 0 findings in `/tmp/bandit_task123_notes.json`.

Follow-up full-suite failure evidence: PR #1375 full-suite jobs are cancelled at the 45/60 minute job timeout while running the Admin module. Local diagnostics with `pytest-timeout` reproduced the Admin stall. A focused single BYOK validation API test passes alone, but consecutive tests in `test_admin_byok_validation_api.py` hang entering a second `TestClient(app)` when the tests are wrapped by `pytest.mark.asyncio` despite using no awaits. The timeout stack is in Starlette/AnyIO TestClient portal teardown during `__enter__`, with lingering ChaCha, Evaluations maintenance, and telemetry threads visible. This points to pytest-asyncio/TestClient lifespan interaction in this file rather than an endpoint assertion failure.

Admin timeout fix: converted `test_admin_byok_validation_api.py`, `test_admin_conflicts.py`, and `test_admin_conflicts_edgecases.py` from `TestClient(app)` to `httpx.ASGITransport` + `AsyncClient`. These tests still exercise real FastAPI routes and dependency overrides, but avoid repeatedly entering the full app lifespan path that starts background services and can stall during teardown. Focused verification passed for all three files.

CI xdist follow-up: local Admin with xdist reproduced the CI shape as worker-level crashes rather than assertion failures (`-n auto` failed after 641 passed/15 worker crashes; `-n 4` failed after 648 passed/8 worker crashes). The named crashes are spread across remaining full-lifespan route tests such as bundle ops, org search/STT settings, and backup schedules, while serial Admin completes successfully. The CI workflow now clears `PYTEST_ADDOPTS` only for the Admin module so this module runs serially while the rest of the full suite keeps xdist. Fresh serial verification with pytest-timeout completed with 656 passed tests in 296.81s.

Final verification: focused Admin route files passed with 13 passed tests in 9.86s after the nosec-only Bandit comments; `git diff --check` passed; Bandit on the touched Admin test files with `-s B101` produced 0 findings in `/tmp/bandit_task123_admin.json`.

Fresh PR check follow-up after pushing `b4dd21847`: workflows are now starting instead of staying globally waiting, but `Lint Workflows (actionlint)` failed on the existing self-hosted label `vz-linux` in `.github/workflows/vz-linux-host-gated.yml`. Added `.github/actionlint.yaml` with the custom self-hosted labels and made the actionlint workflow pass that config explicitly. Local actionlint v1.7.12 against the same targeted workflow set passed with the new config.

PR review follow-up scope: live GraphQL review threads on PR #1375 showed unresolved feedback for API default header merging, request-specific auth header clobbering, protocol-relative URL support, explicit `responseType: "json"`, error response parsing before binary/success response parsing, `captureSessionIdFromHeaders()` input normalization, a 401 cleanup/redirect regression test, and routing `generateSpeech()` through the shared ElevenLabs fetch helper.

PR review follow-up verification: the new WebUI API client regression tests first failed on the current implementation for the default headers, auth override, protocol-relative URL, explicit JSON responseType, binary error detail, and malformed 401 cleanup cases. After fixes, `bunx vitest run lib/__tests__/api-client.fetch.test.ts ../packages/ui/src/services/__tests__/elevenlabs.test.ts` passed 18 tests. `bun run lint -- lib/api.ts lib/__tests__/api-client.fetch.test.ts ../packages/ui/src/services/elevenlabs.ts ../packages/ui/src/services/__tests__/elevenlabs.test.ts` exited 0 with existing project warnings; the shared UI paths are outside that command's base path, so `apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/services/elevenlabs.ts apps/packages/ui/src/services/__tests__/elevenlabs.test.ts` was run with the repo-pinned ESLint and exited 0. `git diff --check` passed. Bandit is not applicable to this TypeScript-only follow-up.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stabilized PR #1375 CI by keeping the axios-to-fetch WebUI/Notes fixes and addressing the follow-on full-suite Admin timeout. Converted three route-level Admin test files away from full `TestClient` lifespan use, updated the GitHub Actions Admin module step to clear global xdist settings so Admin runs serially, and configured actionlint for the existing VZ Linux self-hosted runner labels. This keeps parallelism for the rest of the suite while avoiding worker-level app lifespan hangs in Admin.

Addressed the follow-on PR review threads by restoring axios-like default header merging, preserving request-specific auth headers, supporting protocol-relative URLs, forcing explicit JSON parsing, parsing error bodies before success responseType handling, normalizing response headers for session capture, and routing ElevenLabs speech generation through the shared fetch helper.
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
