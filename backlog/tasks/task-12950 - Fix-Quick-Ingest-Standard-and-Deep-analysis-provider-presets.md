---
id: TASK-12950
title: Fix Quick Ingest Standard and Deep analysis provider presets
status: Done
labels:
- bug
- frontend
- quick-ingest
documentation:
- Docs/superpowers/specs/2026-07-12-quick-ingest-preset-provider-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Target dev with a shared WebUI/browser-extension fix so the active Quick Ingest wizard hydrates saved preset configuration, lets users select an analysis provider in the configure step, and processes Standard/Deep without a misleading provider error when configured. Preserve the early missing-provider guard and add focused regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New Quick Ingest sessions use the saved preset configuration for the selected/default preset.
- [x] #2 The active wizard configure step exposes an analysis provider control when analysis is enabled.
- [x] #3 Standard and Deep can proceed when a provider is configured; missing-provider flows stay on a recoverable wizard step without a render loop.
- [x] #4 Focused unit/integration tests cover WebUI/extension shared behavior and pass.
- [x] #5 A pull request targets dev.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation and PR review follow-up are complete on https://github.com/rmusser01/tldw_server/pull/2717.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented saved-preset hydration and immutable per-open snapshots across the shared WebUI/extension Quick Ingest host.

Reducer edits merge into the full session config, preserve first-source overrides, and remove explicitly cleared advanced fields.

Added a session-scoped analysis-provider AutoComplete with configured suggestions, free text, clear behavior, localized help/warnings, accessible focus, and nonfatal discovery failure.

Provider preflight recovers from persisted steps 1-3, re-arms on close/reopen, waits for connection checks, and never starts a request while invalid.

Live open events consume their pending request exactly once. Seeded draft opens merge named-preset rebases with open details and remount the reducer; processing/terminal sessions retain their active snapshot.

Initial review follow-up added real Ant Design keyboard selection plus WebUI and Chrome-extension storage-backed hydration smoke coverage.

PR #2717 review follow-up addressed all four remaining threads: captured preset maps now change only with revision/remount boundaries; provider-catalog failures emit a diagnostic warning while remaining nonfatal; the real AutoComplete is full width; and the redundant preparedSessionId effect dependency was removed.

Fresh verification after PR feedback: focused shared Vitest 6 files/177 tests passed; extension-config QuickIngestButton 23 passed; Web TypeScript passed with --incremental false; extension compile passed; touched ESLint 0 errors (35 existing warning-level findings in large test fixtures); WebUI production build passed with 152 routes and token sync; Chrome extension production build passed with token sync; git diff --check passed.

GitHub Actions had no actionable failure logs during the audit; all first-party checks were queued, while CodeRabbit passed/skipped review and Cubic was external/skipped.

Known WebUI browser blocker from initial delivery remains: the local Next.js dev harness produced an unrelated runtime error portal/backend overlay on /media after three attempts. The WebUI production build and storage-backed Playwright scenario are present for CI/a clean harness.

Bandit is not applicable because the touched follow-up scope contains no Python.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Quick Ingest Standard/Deep provider failures across the shared WebUI/browser-extension UI and addressed every review thread on PR #2717. Saved preset maps are hydrated and captured per open, excluded sessions cannot silently swap reducer preset bases, provider discovery failures are diagnostic but nonfatal, and the provider control is accessible and full width. All focused tests, typechecks, lint, and production builds pass. PR: https://github.com/rmusser01/tldw_server/pull/2717. A human-authored Change summary remains required before merge.
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
