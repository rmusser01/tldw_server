---
id: TASK-526
title: Flashcards UX PR 2087 follow-up rebase and review fixes
status: Done
labels:
- flashcards
- webui
- pr-review
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2087
modified_files:
- apps/packages/ui/src/components/Flashcards
- apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts
- apps/tldw-frontend/e2e/utils/page-objects/FlashcardsPage.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fetch latest origin/dev and rebase codex/flashcards-ux-phase1-pr
- [x] #2 Inspect PR #2087 comments, review threads, and current check failures
- [x] #3 Address validated review comments and PR issues in flashcards scope
- [x] #4 Run focused verification for any changed files and push the branch
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Rebased the branch onto origin/dev at 84dac2f07.
- Verified current PR review threads and addressed the active Qodo/CodeRabbit findings: removed the disabled Playwright fixme path by making it an active test, surfaced E2E cleanup failures, matched Scheduler deck queries to workspace visibility handoffs, cleared dirty Scheduler state on forced Scheduler hiding, aligned the workspace header Transfer label, and reconciled completed Backlog checklist state.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Follow-up complete for PR #2087. The branch is rebased on latest fetched origin/dev, actionable review findings are fixed, and verification passed: RED component checks failed for the intended missing behavior before implementation; focused component checks passed (3 files, 39 tests); full Flashcards component suite passed (72 files, 388 tests); targeted Playwright invalid-import test passed; neighboring Transfer Playwright checks passed (3 tests); `git diff --check` passed; no `test.fixme` remains in the touched flashcards scope. Bandit skipped: touched files are TS/TSX/E2E/backlog only.
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
