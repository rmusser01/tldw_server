---
id: TASK-13208
title: Make Study Assistant guidance customizable through Service Prompts
status: In Progress
assignee: []
created_date: '2026-09-06 05:17'
updated_date: '2026-09-06 05:21'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2923'
documentation:
  - Docs/Design/study-assistant-service-prompts.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replacement active record for PR #2923 after dev independently assigned TASK-13199 to Persona work. The original complete Study Assistant history is preserved in backlog/archive/tasks/task-13199 - Make-Study-Assistant-guidance-customizable-through-Service-Prompts.md. Implementation exposes owner-specific explanation, mnemonic, follow-up and freeform guidance using existing Service Prompts for flashcards and quizzes; grounding, context, defaults and fact-checking remain unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Four action-specific prompts support shared Settings save/reset.
- [x] #2 Both response endpoints capture authenticated-owner guidance once and preserve defaults, fixed context and fact-check behavior.
- [x] #3 Review findings resolved and tests, Bandit and API compatibility verified.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation and Qodo fixes complete. Merge latest dev into PR branch, verify the combined tree, publish, then await review/required checks and requester merge authorization.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaces the active Study Assistant TASK-13199 to avoid collision with Persona work merged in dev. Original history preserved under backlog/archive/tasks/task-13199 - Make-Study-Assistant-guidance-customizable-through-Service-Prompts.md. PR #2923 implementation 28f05f9aaf and review fixes 2ff84151da remain unchanged. Both Qodo threads are resolved and the latest posted Qodo assessment marks both findings resolved. Updated branch by merging dev 33d7f9f1da at 18e9ccd3a3 without conflicts or rewritten history. No overlap in Study Assistant runtime or Settings files. Original implementation verification: 142 distinct backend and 286 frontend cases; independent review clear. Post-update registry/API suite: 90 passed; OpenAPI fingerprint unchanged (2073 paths, 3142 schemas), runtime Ruff and compile checks pass, Bandit on five runtime files reports zero findings. Full repository suite, full frontend typecheck and browser smoke were not run; unchanged frontend scope uses prior 286-test verification. Awaiting final post-update assistant suite before push.

Post-update assistant suite completed: 52 passed, 454 deselected, in 82.04 seconds. Together with the 90 registry/API cases, 142 backend tests passed on the merged dev tree. Ready to publish the branch update; required CI will rerun on the new head.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Four owner-scoped Study Assistant guidance entries use the existing Service Prompts editor/storage across flashcard and quiz responses. Fixed context, grounding, provider settings, defaults and fact-checking are preserved. Review issues addressed; branch updated to current dev. Tracking record replaced to avoid an independently assigned Persona task ID while preserving all original history. PR #2923 remains open pending review/checks and merge authorization.
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
