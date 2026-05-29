---
id: TASK-542
title: Promote /chat real-server cockpit proof to regression rotation
status: Done
labels:
- chat
- webui
- e2e
- regression
modified_files:
- apps/tldw-frontend/package.json
- apps/tldw-frontend/scripts/assert-playwright-no-skips.mjs
- apps/tldw-frontend/__tests__/frontend-ci-networking-workflows.test.ts
- apps/tldw-frontend/__tests__/assert-playwright-no-skips.test.ts
- .github/workflows/frontend-ux-gates.yml
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wire the existing focused /chat real-server cockpit proof into a repeatable regression command and CI/check path from a clean origin/dev branch. Keep scope limited to /chat regression visibility; do not change chat product behavior unless wiring exposes a concrete issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused /chat real-server cockpit proof has a named package script.
- [x] #2 The package script selects the intended five green-path cockpit tests and fails if Playwright reports skips, zero executed tests, unexpected failures, or flakes.
- [x] #3 Frontend UX Gates run the focused /chat real-server cockpit proof as part of the smoke-gate job.
- [x] #4 A static guard fails if the script or workflow step is removed.
- [x] #5 Verification results and any local live-server limitations are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory existing /chat cockpit E2E spec, package scripts, and CI workflows.
2. Add a failing guard that proves the focused /chat real-server proof is not currently exposed as a first-class regression gate.
3. Add the smallest package/CI/docs wiring to make the focused proof discoverable and runnable.
4. Run focused verification and record results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Inventory found apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts exists, but package.json exposed only generic smoke chat and queued-request chat scripts. Added RED guard in frontend-ci-networking-workflows.test.ts and confirmed it failed because e2e:chat-cockpit:real:focused was absent. Implemented the focused script, no-skips assertion helper, and Frontend UX Gates smoke-gate step. Local live backend proof was attempted with a bundled mock OpenAI server, but the existing 8000 port was owned by another process and the isolated 18001 backend failed health-check without logs in this environment; no local live-browser pass is claimed from that attempt.

PR review-fix pass rebased the branch on latest origin/dev and addressed the review findings without changing /chat product behavior. Replaced the committed static mock OpenAI key literal with a per-run GitHub Actions expression, changed the readiness probe to use OPENAI_API_KEY, and made the retry loop variable visible in a failure message for actionlint/shellcheck. Added focused tests for assert-playwright-no-skips.mjs so all-failure Playwright runs remain distinguishable from empty runs and invalid JSON reports fail cleanly. Updated the package script to run report assertion and copy the report artifact even when Playwright exits nonzero, while preserving the Playwright failure as the final command result.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed TASK-542. Added `e2e:chat-cockpit:real:focused` to the WebUI package, covering the five focused real-server cockpit green-path tests from TASK-535 and wrapping the run with a Playwright JSON no-skips assertion. Wired the Frontend UX Gates smoke job to start a local mock OpenAI provider, start the backend, run the focused /chat cockpit proof, upload the mock-provider log with artifacts, and stop both services. Added static Vitest guards proving the package script, mock-provider setup, no-skips helper, workflow step, dynamic mock token, readiness probe, and failure-path report handling remain present. Verification includes the RED guard before wiring, focused Vitest coverage for workflow wiring and the no-skips helper, `node --check scripts/assert-playwright-no-skips.mjs`, package.json parsing, Playwright discovery selecting exactly five focused chat cockpit tests, and `git diff --check`. Local live-browser execution is not claimed because the existing 8000 server belonged to another process and an isolated backend on 18001 failed health-check without logs in this sandboxed run. Bandit skipped because no Python files were changed.
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
