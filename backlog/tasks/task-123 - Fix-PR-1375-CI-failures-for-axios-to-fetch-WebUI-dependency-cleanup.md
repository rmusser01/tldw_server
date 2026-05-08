---
id: TASK-123
title: 'Fix PR #1375 CI failures for axios-to-fetch WebUI dependency cleanup'
status: In Progress
assignee:
  - Codex
created_date: '2026-05-08 13:29'
updated_date: '2026-05-08 13:41'
labels:
  - ci
  - webui
  - notes
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
Resolve the current PR #1375 GitHub Actions failures after the axios-to-fetch WebUI dependency cleanup. Scope is limited to the failing ElevenLabs timeout smoke path and the Notes backend remediation mock pagination failures observed in CI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ElevenLabs fetch service normalizes timeout-style browser fetch failures so TTS and Speech retry alerts show the timeout body.
- [x] #2 Focused ElevenLabs service or hook tests cover the timeout normalization path.
- [x] #3 Notes backend integration tests set concrete pagination count values instead of passing MagicMock totals to pagination metadata.
- [x] #4 Targeted frontend and backend tests for the changed files pass locally.
- [ ] #5 PR #1375 branch is pushed after fixes are verified.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm failure scope from PR #1375 checks and keep non-actionable full-suite queue/cancellation noise separate.
2. Frontend fix: add focused ElevenLabs service coverage for browser fetch timeout-style failures that are not DOM AbortError from our own timer, then normalize those errors to the existing "ElevenLabs request timed out" message used by the UI timeout classifier.
3. Notes test fix: update the two failing mocked integration tests to provide concrete count return values for pagination, matching the endpoint's count contract and avoiding MagicMock totals.
4. Smoke-gate fix: update E2E auth seeding to set the current `assistant_setup_dismissed` first-run overlay flag alongside the legacy first-run flag so TTS/Speech smoke tests reach the audio pages.
5. Verify locally with the focused ElevenLabs Vitest file, related TTS provider/Speech page tests, the two failing Notes pytest tests, focused Playwright smoke repro, git diff whitespace check, and Bandit on touched Python test scope.
6. Stage, commit, push the PR branch, then refresh PR #1375 checks and record results in the Backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root-cause evidence: Notes backend remediation failed because count_notes/count_keyword_collections mocks returned MagicMock values, which build_offset_pagination_meta compared against ints. UX Smoke Gate failed on ElevenLabs timeout retry assertions after the axios-to-fetch rewrite; the fetch service only maps our own AbortError timeout and passes route-aborted browser fetch failures through as generic errors.

Verification so far: `bunx vitest run src/services/__tests__/elevenlabs.test.ts` passed 6 tests after the red test initially failed as expected; `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notes/test_notes_api_integration.py::test_list_notes tldw_Server_API/tests/Notes/test_notes_api_integration.py::test_list_keyword_collections_with_keywords` passed 2 tests after reproducing both failures; `bunx vitest run src/services/__tests__/elevenlabs.test.ts src/hooks/__tests__/useTtsProviderData.test.tsx src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx` passed 21 tests; focused Playwright `bunx playwright test e2e/smoke/stage7-audio-regression.spec.ts --grep "ElevenLabs timeout" --reporter=line --workers=1` passed 2 tests after fixing E2E seedAuth; `git diff --check` passed; Bandit on the touched Notes test file with `-s B101` produced 0 findings in `/tmp/bandit_task123_notes.json`.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
