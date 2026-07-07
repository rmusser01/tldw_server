---
id: TASK-12170
title: Implement TASK-12169 generation plan normalization and prompt
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-07 20:27'
labels: []
dependencies: []
references:
  - TASK-12169
  - >-
    Docs/superpowers/specs/2026-07-07-advanced-quiz-generation-controls-design.md
  - >-
    Docs/superpowers/plans/2026-07-07-advanced-quiz-generation-controls-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 for TASK-12169: add private quiz generation plan normalization helpers and prompt plan formatting, with focused TDD tests for planned MCQ, multi-select, matching, true/false, fill-blank, and prompt coverage. Scope is limited to quiz_generator.py and assigned quiz generator tests; do not wire question_plan into generation endpoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Private quiz generation helpers normalize planned MCQ, multi-select, matching, true/false, and fill-blank output without enabling endpoint wiring.
- [x] #2 Planned prompt formatter renders exact requested rows and all five output shapes without the legacy four-option MCQ instruction.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: `python -m pytest tldw_Server_API/tests/Quizzes/test_quiz_generator_question_plan.py tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py -q` failed as expected with missing planned helpers/formatter, `_coerce_options` lacking `expected_count`, and MCQ letter normalization capped at A-D.

GREEN: same focused pytest command passed 15 tests.

Bandit: `python -m bandit -r tldw_Server_API/app/services/quiz_generator.py -f json -o /tmp/bandit_task12170.json` returned 0 findings.

`git diff --check` was clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented private generation-plan helpers for TASK-12169 Task 2: strict planned normalization for MCQ, multi-select, matching, true/false, and fill-blank; exact planned option/pair counts; MCQ letter answers beyond D; and a private planned prompt formatter covering all five question shapes. No endpoint or WebUI wiring was added.

Verification: RED focused pytest failed for missing helpers/formatter before implementation; GREEN focused pytest passed 15 tests after implementation; Bandit on quiz_generator.py reported 0 findings; git diff --check was clean.
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
