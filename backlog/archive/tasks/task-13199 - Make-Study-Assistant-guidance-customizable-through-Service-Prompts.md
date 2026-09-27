---
id: TASK-13199
title: Make Study Assistant guidance customizable through Service Prompts
status: In Progress
assignee: []
created_date: '2026-09-06 04:31'
updated_date: '2026-09-06 05:17'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2923'
documentation:
  - Docs/Design/study-assistant-service-prompts.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement approved bounded slice: owner-specific explanation, mnemonic, follow-up and freeform guidance for synchronous flashcard and quiz Study Assistant responses, using existing Service Prompts storage and shared Settings. Keep grounding/context/provider settings and fact-checking unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Four action-specific guidance entries support shared Settings save/reset.
- [x] #2 Flashcard and quiz responses use one authenticated-owner selected-action snapshot, with fixed grounding and context carriers.
- [x] #3 Defaults, provider settings, fact-checking and existing response persistence remain compatible; invalid configuration fails before response writes.
- [x] #4 Regression tests cover both endpoints, owner isolation, save/reset, snapshots and failures; lint, Bandit and independent review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: Record design and baseline. Stage 2: Write failing API/model-boundary and Settings regressions. Stage 3: Minimal registry/core/endpoint/shared-metadata integration. Stage 4: Verify, review and prepare integration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

All four stages complete on codex/study-assistant-service-prompts from dev d308a40871. Baseline: 29 assistant tests passed. RED: new backend suite 12 failed/2 passed because saved guidance was ignored and corrupt overrides did not fail; four new Settings tests failed because entries were absent. Final verification: 120 focused backend tests passed (23 new real HTTP/storage/model-boundary tests, seven existing core tests, 90 registry/API tests); 29 existing assistant tests passed including 22 endpoint/sanitization cases, and all nine quiz assistant cases were rerun after the final fake-signature adjustment. Shared Settings/transport 205 passed; WebUI Settings 81 passed. Independent read-only review found no actionable findings and independently passed all 23 new cases. Bandit on five runtime files: zero findings. Runtime/new-test/registry-test Ruff checks and selected formatting checks pass; compile and diff checks pass. Two legacy integration test modules retain the same 15 pre-existing import lint findings, verified against HEAD; no new lint findings introduced. OpenAPI fingerprint unchanged (2073 paths, 3142 schemas), TypeScript schema generation succeeded. Full repository suite, full frontend typecheck and browser smoke not run. Existing dependencies reused with temporary symlinks, removed after testing. Own temporary plan removed after completion. Implementation ready for integration choice; no PR created.

Published PR #2923 against dev at requester option 2. Implementation commit 28f05f9aaf1423a9bbc2fa785898f161008ad5c1. Fresh pre-publish verification: 120 focused backend cases passed in 44.19 seconds. Worktree preserved for review follow-up; no merge or recurring monitor initiated.

PR #2923 Qodo review posted two verified findings: public guidance docstrings lack complete API contracts, and eleven modified endpoint-test doubles need full annotations and docstrings. Addressing these documentation/type-only changes without changing runtime behavior. CI currently pending; branch is behind dev.

Qodo fixes verified: both public guidance APIs now document arguments, return semantics, owner/worker lifecycle and errors; all eleven modified endpoint test callbacks have full parameter/return annotations and docstrings. Fresh affected suite: 52 passed, 454 deselected. AST audit confirms all eleven callback contracts; compilation, runtime Ruff/format and diff checks pass; Bandit on two touched runtime files reports zero findings. Independent follow-up confirmed no behavior changes and caught one exception-documentation detail, now corrected to reflect database cleanup semantics. Current dev has advanced to 33d7f9f1da; no rebase or base merge performed in this review-fix turn.

Tracking-only retirement before updating PR #2923 from dev: newer Persona work independently claimed active TASK-13199. Preserve this complete Study Assistant record in the archive; a replacement active record will link PR #2923 and this history after updating dev. This does not mark implementation merged.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Made explanation, mnemonic, follow-up and freeform Study Assistant guidance editable through existing Service Prompts settings for flashcards and quizzes. One shared authenticated dependency captures selected-action owner guidance; core resolution closes storage on the same worker and passes an immutable string to generation. Fixed grounding/context/history, provider controls, exact defaults, and structured fact-checking remain unchanged. Reuses existing storage, Settings and response persistence without new API fields or jobs infrastructure. Verified 142 distinct focused backend cases and 286 frontend cases, clean touched-runtime Bandit, unchanged API fingerprint and independent review.
<!-- SECTION:FINAL_SUMMARY:END -->

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
