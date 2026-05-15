---
id: TASK-374
title: Implement Stage 2D Watchlist-first Overview onboarding
status: Done
assignee: []
created_date: '2026-05-15 04:56'
updated_date: '2026-05-15 07:02'
labels:
  - watchlists
  - stage2
  - frontend
dependencies:
  - TASK-373
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reposition the existing Overview Quick Setup after the Stage 2 shell setup wizard. Scope: prevent source-first auto-open when no Watchlist is selected, frame existing quick setup as adding initial collection to the selected Watchlist, preserve pipeline builder scope, update copy/tests, and run constrained viewport CDP smoke if UI behavior changes materially.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Overview no longer bypasses Watchlist-first creation with source-first auto-open behavior.
- [x] #2 Existing selected-Watchlist quick setup still creates scoped source/job payloads.
- [x] #3 User-facing Overview copy frames quick setup as initial collection inside the selected Watchlist.
- [x] #4 Constrained viewport smoke evidence is recorded if the rendered flow changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Stage 2D in worktree codex/watchlists-stage1a. Inspecting Overview quick setup behavior, selected Watchlist scoping, and existing tests before edits.

Implemented Stage 2D Overview repositioning. Verification passed: quick setup helper/copy contracts (7 tests), selected-scope static contract (8 tests), targeted Overview UI checks for no-Watchlist fallback and selected-Watchlist initial collection copy (2 tests). git diff --check passed. CDP/Playwright constrained viewport smoke was attempted against Next dev on ports 3027 (Turbopack) and 3028 (webpack) with mocked Watchlists API routes; both servers stayed at 'Compiling /watchlists' until page.goto timed out and were stopped cleanly. No screenshot was produced. This is recorded as a smoke blocker rather than a product-code failure.

Bandit skipped: touched code is frontend TypeScript/JSON only.

Stage 2E retry cleared the earlier CDP blocker: Playwright/CDP loaded /watchlists from the current worktree, dismissed the first-run gate, exercised desktop source-backed setup and 390x844 topic-only setup, captured screenshots in /tmp, and confirmed source/job payloads carried watchlist_id. Computer Use was not used.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 2D repositioned Overview onboarding behind a selected Watchlist: no-Watchlist state now prompts Watchlist creation instead of source-first auto-open, selected-Watchlist collection setup preserves scoped source/job payloads, and user-facing copy says Add initial collection. Verification covered helper/copy/scope tests plus the Stage 2E CDP smoke retry; Bandit skipped because touched files are frontend TypeScript/JSON only.
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
