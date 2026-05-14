---
id: TASK-295
title: Complete main /chat cockpit functionality in PR 1582
status: In Progress
assignee: []
created_date: '2026-05-12 05:10'
updated_date: '2026-05-14 01:32'
labels:
  - webui
  - chat
  - frontend
  - pr-1582
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1582'
  - 'https://github.com/rmusser01/tldw_server/issues/1646'
documentation:
  - Docs/superpowers/specs/2026-05-12-main-chat-cockpit-controls-gap-design.md
  - >-
    Docs/superpowers/plans/2026-05-12-main-chat-cockpit-first-slice-implementation-plan.md
  - >-
    Docs/superpowers/specs/2026-05-13-main-chat-cockpit-p-series-completion-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish the remaining main WebUI /chat cockpit work on the existing draft PR #1582 branch only. Scope is the main /chat Playground surface, not the extension sidebar/sidepanel or unrelated pages. The goal is to preserve all existing chat-page functionality while making the new cockpit rails/status controls operate on the same chat state, with real-server verification against the running backend.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All merge-critical real-server proof uses the running server with no mocked payloads and no page route interception
- [ ] #2 PR #1582 remains draft and is not marked ready or merged until P0 P1 and P2 are all complete and explicitly approved by the human maintainer
- [ ] #3 P0 workflows from issue #1646 are implemented verified and approved including character persona prompts model chat MCP and real-server state-changing proof
- [ ] #4 P1 workflows from issue #1646 are implemented verified and approved including context session run controls keyboard focus and mobile workflows
- [ ] #5 P2 polish from issue #1646 is completed verified and approved including IA copy degraded states duplication decisions visual QA screenshots and final PR closeout notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 0: Reopen honest tracking. Keep TASK-295 and PR #1582 tied to issue #1646 and mark the PR draft until P0/P1/P2 are explicitly approved.
Stage 1: Complete Character / Persona rail workflows including clear, correct selector tab, change flows, inspect/details, bootstrap/persona-memory behavior, tests, and real-server proof.
Stage 2: Complete Prompt rail workflows including select, clear, inline/custom prompt state, isolation, tests, and real-server proof or real empty-state proof.
Stage 3: Complete Model & Chat rail workflows including scoped provider:model summaries, persist/restore harmless setting, duplicate model-id routing, tests, and real-server proof.
Stage 4: Complete MCP rail workflows including tool choice, direct settings workflow, unavailable/degraded states, tool state counts, tests, and real-server proof.
Stage 5: Complete Context and Session rail workflows including next-reply inventory, clear/remove isolation, session status, session switching, tests, and real-server proof.
Stage 6: Complete Run Controls and Recovery including stop, regenerate, disabled states, recoverable errors, and request-state-machine-safe tests.
Stage 7: Complete Keyboard, Focus, and Mobile workflows including focus restoration, keyboard operation, mobile workflow parity, and mobile Playwright proof.
Stage 8: Complete P2 polish and merge readiness including IA/copy/degraded-state cleanup, composer/rail duplication decisions, final visual QA, refreshed real-server screenshots, PR closeout notes, and human approval gates.
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

CI closeout refresh: PR #1582 is still open and draft on codex/chat-degraded-health. Live checks now show targeted frontend/build/lint/e2e/security/required aggregate gates passing after the SKIP_WXT_PREPARE workflow fixes. Remaining red checks are only the broad Full Suite matrix jobs. Representative Full Suite Ubuntu/Python 3.11 job 75624023692 was cancelled at the one-hour job limit during Run tests (Audio); earlier root-level/Admin steps passed, later modules were skipped due cancellation, and the job's Fail if any module failed step itself succeeded. Review threads are all resolved.

Reopened for new PR #1582 review comments posted after CI closeout: validate the new Qodo and Cubic findings against current code, fix still-valid /chat/model/readiness issues, rerun focused tests, and keep the PR draft.

PR #1582 review follow-up: fixed the new valid Qodo/Cubic comments. ServerReadinessGate now emits readiness state after allowed children mount; /chat tracks degraded readiness separately from degraded check names; provider-qualified compare branches preserve provider:model identity while sending the bare model id to the API; scoped OCR language values persist even when matching global defaults; selectedKnowledge clearing is typed as nullable; the real-server cockpit spec no longer embeds a fallback API key; and cockpit English locale keys now mirror into extension public locale messages.

Verification for latest review follow-up: ServerReadinessGate degraded test passed (5 tests). Focused UI suites passed: Playground cockpit controls, playground locale mirror, model scoped settings, resolve-api-provider, and chat-action-utils RAG/compare helpers (39 tests). Real-server /chat cockpit Playwright passed against http://127.0.0.1:8000 with degraded health from chacha_notes and no route interception (3 tests). git diff --check passed. Package UI tsc remains blocked by unrelated baseline test/type errors; verify:design-system-state remains blocked by unrelated Chatbooks baseline entries after the new Playground state-label findings were removed. Bandit was not applicable because this follow-up touched TypeScript/TSX/JSON/Playwright/Markdown only.

Merge bar changed from first complete slice to fully mature main /chat cockpit. Reopened TASK-295 for additional UI/design completion in the same draft PR #1582, still scoped to main /chat only and not sidepanel/sidebar.

Completed mature main /chat cockpit slice in draft PR #1582. Added actionable context-source inventory, runtime/provider diagnostics, scoped setting summaries, MCP/tool entry points, prioritized status strip actions, controlled mobile cockpit tabs, shared message-count state, and stale i18n count recovery.

Mature cockpit verification: focused cockpit Vitest passed (20 tests); real-server Playwright /chat cockpit passed against http://127.0.0.1:8000 with no route interception (3 tests); screenshot harness returned a 200 chat completion with two rendered messages and status strip text showing 2 messages; git diff --check passed; filtered package-ui tsc output had no changed Playground-file errors; design-state filter showed only the existing allowed PlaygroundForm baseline, while the full command remains blocked by existing Chatbooks/shared-product-state baseline findings. Bandit not applicable because only frontend/docs/e2e files were touched.

Added staged P-series completion spec for issue #1646. Earlier cockpit completion notes are historical only; PR #1582 remains draft and must not be considered merge-ready until P0, P1, and P2 are all fully completed and explicitly approved by the human maintainer.

Corrected TASK-295 after issue #1646: the prior checked acceptance criteria and final summary are superseded. The task is In Progress until the P0 P1 and P2 completion gates are implemented verified and explicitly approved.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
