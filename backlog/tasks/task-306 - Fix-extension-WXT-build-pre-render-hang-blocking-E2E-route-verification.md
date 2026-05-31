---
id: TASK-306
title: Fix extension WXT build/pre-render hang blocking E2E route verification
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-12 16:16'
updated_date: '2026-05-12 21:31'
labels:
  - extension
  - e2e
  - build
  - wxt
dependencies:
  - TASK-297.5
references:
  - apps/extension/playwright.config.ts
  - apps/extension/tests/e2e/setup/build-extension.ts
  - apps/extension/scripts/build-with-profile.mjs
  - apps/extension/wxt.config.ts
  - TASK-297.5
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The /knowledge parity slice is blocked from browser-level extension verification because the extension WXT build/pre-render step hangs before writing manifest.json. This appears broader than /knowledge: direct production build, development build, WXT dev server pre-render, options-only filtered build, and background-only filtered build all hang after WXT entrypoint transform/import logging. Investigate and fix the extension build/pre-render path so extension E2E route tests can run again.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `bun run build:chrome:prod` from apps/extension completes and writes a valid build/chrome-mv3 or .output/chrome-mv3 manifest with required options/background assets.
- [x] #2 `bun run dev -- --host 127.0.0.1 --port <port>` reaches a usable WXT dev state or documents why dev pre-render is not expected for E2E.
- [x] #3 A targeted extension Playwright run for the knowledge tutorial route cases reaches the browser phase instead of hanging in global setup.
- [x] #4 Root cause is documented with evidence; if the fix is outside repository code or environment-specific, the task records the required local environment remediation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the WXT hang with the narrowest command (`wxt build --filter-entrypoint background --debug`).
2. Inspect the stuck Node process with OS-level sampling to identify whether it is CPU-bound, blocked on I/O, waiting on file watching, or stuck in a specific package.
3. Compare with WXT config/build helpers and current dependency layout to form one root-cause hypothesis.
4. Only if the root cause is a small repo-local harness/build issue, patch it with focused tests or verification; otherwise document the environment/build blocker and leave implementation for a dedicated follow-up.
5. Verify no unrelated /knowledge changes are introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved the WXT build/pre-render hang by replacing the static copilot content-script re-export with a WXT defineContentScript wrapper that dynamically imports @tldw/ui/entries/copilot-popup.content from main(). Root cause: WXT imports JS entrypoints during metadata discovery before applying filtered entrypoint builds, so the static shared copilot import could hold the build/pre-render process open even for unrelated options/background filters.

Verification on 2026-05-12: bun test tests/unit/copilot-entrypoint-lazy-import.test.ts passed; bun run compile passed; bun run build:chrome:prod completed and wrote .output/chrome-mv3 manifest/assets; bun run dev -- --host 127.0.0.1 --port 17311 reached WXT dev/pre-render and was stopped; targeted Playwright knowledge route run reached browser phase. Bandit skipped because touched implementation is TypeScript/Playwright, not Python.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Changed the extension copilot content-script entrypoint from a static shared UI re-export to a WXT defineContentScript wrapper with a runtime dynamic import. This keeps WXT metadata discovery light enough for production builds, filtered builds, and dev pre-render to complete while preserving runtime behavior when the content script actually starts. Added a unit regression test to prevent reverting to the static re-export pattern.

Verification: bun test tests/unit/copilot-entrypoint-lazy-import.test.ts; bun run compile; bun run build:chrome:prod; bun run dev -- --host 127.0.0.1 --port 17311; targeted Playwright knowledge route run reached browser execution. The remaining /knowledge tutorial assertion failure was tracked and fixed in TASK-297.5 rather than treated as a WXT build blocker.
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
