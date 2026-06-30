---
id: TASK-246
title: Migrate AsciiSpinner loading label to design-system registry
status: Done
assignee: []
created_date: '2026-05-10 21:12'
updated_date: '2026-05-10 21:31'
labels:
  - design-system
  - frontend
  - product-state
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the frontend design-system product-state cleanup by replacing the Common SplashScreen AsciiSpinner hardcoded Loading text with the canonical design-system loading state label while preserving spinner animation, dot padding, progress rendering, and canvas behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 AsciiSpinner builds its loading text from getDesignSystemState('loading').label rather than a hardcoded Loading literal.
- [x] #2 The existing spinner frames, loading dot behavior, progress bar, and render positions remain unchanged.
- [x] #3 Focused coverage proves the rendered loading row uses the design-system loading label.
- [x] #4 The matching AsciiSpinner Loading canonical-state-label baseline exception is removed and the design-system verifier passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused red test for AsciiSpinner.render with mocked CharGrid and mocked design-system loading label. 2. Route the rendered loading row through getDesignSystemState('loading').label without changing dot/progress behavior. 3. Remove the AsciiSpinner Loading baseline exception. 4. Verify focused test, product-state guard test, design-system verifier, diff check, and touched-scope typecheck output.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented AsciiSpinner loading label registry lookup via getDesignSystemState('loading').label, added focused Vitest coverage that first failed against the hardcoded literal and now passes, removed the AsciiSpinner canonical-state-label baseline exception, and refreshed current ChatbooksPlaygroundPage AntD Alert baseline IDs after origin/dev drift blocked the full verifier.

Verification: focused AsciiSpinner Vitest passed; product-state guard Vitest passed; bun run verify:design-system-state passed; git diff --check passed; full UI tsc still exits 2 on existing repo-wide type baseline, with no filtered matches for the touched SplashScreen/AsciiSpinner paths. Bandit not applicable because touched implementation files are TypeScript/JSON/Backlog markdown only.

PR: https://github.com/rmusser01/tldw_server/pull/1553

PR review follow-up: Gemini flagged the render-loop loading label lookup and repeated dot suffix string construction in AsciiSpinner.ts. The label is static for this effect, so the follow-up will cache the registry label and dot suffixes outside render while preserving row 17 output.

PR review follow-up implemented: cached the loading design-system label at module scope and replaced per-render repeat/concatenation with precomputed dot suffixes, preserving the exact row 17 strings. Tightened the focused test so it fails if render calls getDesignSystemState after module initialization.

Review follow-up verification: focused AsciiSpinner Vitest passed after failing red on the render-loop lookup; product-state guard Vitest passed; bun run verify:design-system-state passed; git diff --check passed; filtered UI tsc output for AsciiSpinner/SplashScreen touched paths returned no matches while the full command still exits nonzero on existing repo-wide baseline errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
AsciiSpinner now sources the loading label from the registry once at module initialization and uses precomputed loading dot suffixes in render. The focused regression test covers both the registry label output and absence of per-render registry lookup; the PR review thread was addressed without changing visible spinner behavior.
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
