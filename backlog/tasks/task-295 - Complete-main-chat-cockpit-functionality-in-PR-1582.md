---
id: TASK-295
title: Complete main /chat cockpit functionality in PR 1582
status: Done
assignee: []
created_date: '2026-05-12 05:10'
updated_date: '2026-05-12 15:27'
labels:
  - webui
  - chat
  - frontend
  - pr-1582
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1582'
documentation:
  - Docs/superpowers/specs/2026-05-12-main-chat-cockpit-controls-gap-design.md
  - >-
    Docs/superpowers/plans/2026-05-12-main-chat-cockpit-single-pr-completion-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish the remaining main WebUI /chat cockpit work on the existing draft PR #1582 branch only. Scope is the main /chat Playground surface, not the extension sidebar/sidepanel or unrelated pages. The goal is to preserve all existing chat-page functionality while making the new cockpit rails/status controls operate on the same chat state, with real-server verification against the running backend.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #1582 remains the single vehicle for main /chat cockpit completion; no second PR or sidepanel/sidebar route work is introduced.
- [x] #2 Main /chat cockpit and focus modes preserve existing composer workflows: model selection/settings, character/persona controls, Search & Context, web search, MCP/tools, attachments, prompt/tools menus, advanced controls, send/stop behavior, thread search, and artifacts where currently available.
- [x] #3 Cockpit rails and status strip expose direct controls/status for the highest-value shared chat state without creating rail-local duplicate state: context/session controls, runtime/model/persona controls, degraded/error/streaming state, and independent rail visibility where supported.
- [x] #4 Focused unit/component tests cover new shared-state controls and layout behavior; real-server Playwright coverage exercises the running backend without mocked API data or route interception.
- [x] #5 Known baseline blockers are documented separately from this PR's changed behavior, and the draft PR is pushed with verification evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: Reconcile spec/plan with the single-PR scope for PR #1582.
Stage 2: Add independent cockpit context/runtime rail visibility controls with persisted state and tests.
Stage 3: Add remaining direct rail/status controls only where they map to existing shared /chat state and handlers.
Stage 4: Expand real-server Playwright verification against the running backend without mocked API data or route interception.
Stage 5: Push PR #1582 as draft with focused verification evidence and documented baseline blockers.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Docs/superpowers/plans/2026-05-12-main-chat-cockpit-single-pr-completion-plan.md and updated the cockpit gap spec to reflect that remaining main /chat cockpit work stays inside PR #1582 rather than a second PR.

Implemented independent context/runtime rail visibility controls in the main /chat cockpit shell, persisted separately from focus/cockpit mode.

Added direct context clear controls for active file, knowledge, media, and attached research context state using existing /chat setters/handlers.

Added runtime rail stop-generation and regenerate-last-response actions wired to existing shared chat handlers, with regenerate hidden during active streaming.

Wired degraded server readiness from ServerReadinessGate into the main /chat cockpit runtime rail and status strip so degraded health permits chat immediately with visible subsystem warnings.

Expanded the real-server /chat Playwright spec to verify independent rail visibility, provider-qualified chat payload routing, live degraded health entry, and no route interception.

Focused Vitest cockpit suite passed: 7 files, 34 tests. ServerReadinessGate degraded test passed: 1 file, 4 tests. Real-server Playwright cockpit spec passed: 3 tests against http://127.0.0.1:8000. A one-off real-server screenshot capture returned a 200 chat completion and saved /private/tmp/tldw-chat-cockpit-real-server-20260512.png.

Known baseline: the focused Playground cockpit tests still log non-fatal mocked-server 400s from existing chat-settings fetch behavior. The repo-wide UI typecheck remains blocked by unrelated pre-existing errors outside this touched scope. Bandit was not applicable because this slice touched TypeScript, TSX, Markdown, and Playwright files only.

Reopened for PR #1582 review-follow-up: validate current inline comments, fix any still-valid model/settings issues, rerun focused verification, and keep the PR draft.

PR #1582 review follow-up: verified current review comments against dcba5633a. Previously raised provider-qualified selection, provider usability flags, backend config flag assignment, PEP8 wrapping, and a11y source assertion issues were already fixed. Fixed remaining valid scoped settings issue by making updateSetting always delegate through the active scope path and added a regression test for scoped values that match the global default.

Review-follow-up verification: model.scoped-settings test first failed before the store fix, then passed. Targeted UI review tests passed: modelSelectorUtils, useModelSelector capabilities, Playground cockpit a11y, and model scoped settings (19 tests). Backend model metadata filters passed (6 tests) using the repo virtualenv. Real-server /chat Playwright cockpit spec passed against http://127.0.0.1:8000 with no route interception (3 tests). git diff --check passed. Bandit was not rerun because this follow-up touched only TypeScript test/store code and the task record.

Reopened for CI failure follow-up: Playground Device/A11y/Composer Gates failed before tests during bun install. Root cause investigation found the workflow times out in workspace install while running extension WXT prepare; the same install completes locally when SKIP_WXT_PREPARE=1, and this Playground gate does not need extension preparation.

CI failure follow-up fix: added SKIP_WXT_PREPARE=1 at the UI Playground Quality Gates job level so the Playground-only workflow install does not run extension WXT preparation. Verification: SKIP_WXT_PREPARE=1 bun install --frozen-lockfile from apps completed with no changes; exact gate scripts passed locally: composer 31 tests, device matrix 14 tests, accessibility 19 tests; git diff --check passed. actionlint is not installed in this environment, so workflow validation is via YAML shape review and rerun CI after push.

Extended CI install fix after Onboarding E2E Gate failed with the same root cause: install-time cancellation before tests, with sibling UX Smoke stuck in Install frontend dependencies. Applying SKIP_WXT_PREPARE=1 to non-extension frontend/e2e CI jobs that run bun install from apps.

Extended non-extension frontend CI workflows now set SKIP_WXT_PREPARE=1: ci frontend lint, frontend-required, frontend UX gates, frontend e2e tiers, plus the already-pushed Playground quality gate. Verification: SKIP_WXT_PREPARE=1 bun install --frozen-lockfile from apps completed with no changes; exact Playground gate scripts had already passed locally; git diff --check passed. actionlint is validated by GitHub CI because it is not installed locally.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the main /chat cockpit single-PR slice for draft PR #1582: independent rail visibility, direct shared-state rail actions, degraded-health cockpit warnings, and real-server parity coverage are now in the PR branch. Verification includes focused component tests, readiness-gate tests, real-server Playwright without route mocks, and a real-server screenshot capture.

Review follow-up fixed the remaining active scoped-settings defect and revalidated the PR review-comment surface with focused UI, backend, and real-server /chat coverage.

CI follow-up scoped the Playground gate install away from extension WXT preparation and verified the exact gate scripts locally before pushing.

Extended the install-time WXT prepare skip across non-extension frontend/e2e workflows after the same install-time timeout affected onboarding/UX checks.
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
