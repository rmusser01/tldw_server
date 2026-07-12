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
Implementation and review fixes are complete. Finalize the temporary plan record, commit, push codex/fix-quick-ingest-preset-provider, open a PR targeting dev, then record the PR URL and final status.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented saved-preset hydration and immutable per-open snapshots across the shared WebUI/extension Quick Ingest host.

Reducer edits merge into the full session config, preserve first-source overrides, and remove explicitly cleared advanced fields.

Added a session-scoped analysis-provider AutoComplete with configured suggestions, free text, clear behavior, localized help/warnings, accessible focus, and nonfatal discovery failure.

Provider preflight now recovers from persisted steps 1-3, re-arms on close/reopen, waits for connection checks, and never starts a request while invalid.

Live open events consume their pending request exactly once. Seeded draft opens merge named-preset rebases with open details and remount the reducer; processing/terminal sessions explicitly retain their active snapshot.

Review follow-up added real Ant Design keyboard selection plus WebUI and Chrome-extension storage-backed hydration smoke coverage.

Verification: focused shared Vitest 6 files/176 tests passed; extension-config QuickIngestButton 22 passed; quick-ingest-open 7 passed; Web storage adapter 10 passed; Web TypeScript passed with --incremental false; extension compile passed; touched ESLint 0 errors (94 existing warning-level findings in large test fixtures/source baseline); WebUI production build passed; Chrome extension build passed; Chrome extension provider hydration smoke passed; missing-provider Chrome smoke passed; git diff --check passed.

Known WebUI browser blocker: the local Next.js dev harness produced an unrelated runtime error portal/backend overlay on /media after three attempts, intercepting Quick Ingest clicks. Per the repository three-attempt rule, no fourth run was made. The WebUI production build and the storage-backed Playwright scenario remain present for CI/a clean harness.

Bandit: not applicable because the touched implementation/test/docs scope contains no Python.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Quick Ingest Standard/Deep provider failures across the shared WebUI/browser-extension UI. Saved preset maps are hydrated before draft creation, captured per open, and merged losslessly through edits. The Configure step now exposes an accessible session-only provider selector and invalid runs recover there without starting requests. Review hardening covers pending-event consumption, seeded draft remount/rebase behavior, processing/terminal snapshot preservation, persisted-step redirects, reopen retries, and connection checks. PR: https://github.com/rmusser01/tldw_server/pull/2717. Human-authored Change summary remains required before merge.
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
