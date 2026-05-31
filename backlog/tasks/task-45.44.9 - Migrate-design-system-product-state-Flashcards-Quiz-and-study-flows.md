---
id: TASK-45.44.9
title: 'Migrate design-system product state: Flashcards, Quiz, and study flows'
status: In Progress
assignee: []
created_date: '2026-05-14 03:19'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1666'
  - >-
    Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - 'https://github.com/rmusser01/tldw_server/pull/2000'
  - 'https://github.com/rmusser01/tldw_server/pull/2002'
  - 'https://github.com/rmusser01/tldw_server/pull/2004'
parent_task_id: TASK-45.44
priority: medium
documentation:
  - >-
    TASK-45.44.9.7 / PR #2000 migrated FlashcardTemplateValueModal's template-load
    error Alert to the design-system Alert primitive. Baseline evidence: total product-state
    exceptions 265 -> 264; Flashcards/Quiz/study-flow exceptions 41 -> 40; FlashcardTemplateValueModal
    target rows 1 -> 0. Verification recorded in TASK-45.44.9.7.
  - >-
    TASK-45.44.9.8 / PR #2002 migrated the Flashcards ExportPanel export-preview Alert
    to the design-system Alert primitive. Baseline evidence: total product-state exceptions
    264 -> 263; Flashcards/Quiz/study-flow exceptions 40 -> 39; ExportPanel target rows
    1 -> 0. Verification recorded in TASK-45.44.9.8.
  - >-
    TASK-45.44.9.9 / PR #2004 migrated ReviewTab onboarding and review-retry Alerts
    to the design-system Alert primitive. Baseline evidence: total product-state exceptions
    263 -> 262; Flashcards/Quiz/study-flow exceptions 39 -> 38; ReviewTab target rows
    1 -> 0. Verification recorded in TASK-45.44.9.9.
  - >-
    TASK-45.44.9.11 / PR #2172 migrated StudySuggestionsPanel loading, empty,
    failed, reused-result, and status feedback from AntD Alert/Empty/Tag to design-system
    Alert, EmptyState, LoadingState, and Badge primitives. Baseline evidence:
    total product-state exceptions 110 -> 107; Flashcards/Quiz/study-flow exceptions
    24 -> 21; StudySuggestionsPanel target rows 3 -> 0. Verification recorded
    in TASK-45.44.9.11.
  - >-
    TASK-45.44.9.12 / PR #2172 addressed review findings on StudySuggestionsPanel:
    pending/no-snapshot status now remains in LoadingState instead of falling through
    to EmptyState, and status badge variant mapping now uses the SuggestionStatus
    union with the unreachable active case removed. Verification recorded in
    TASK-45.44.9.12.
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mirror the linked GitHub product-area migration issue. Closure requires zero current product-state baseline exceptions for the owned path map area and the verification gates recorded in the GitHub issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The linked GitHub issue owns current count and public status.
- [ ] #2 Implementation PR tasks are created under this child when the area is too broad for one PR.
- [ ] #3 Backlog notes record PR links and before/after count evidence.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
