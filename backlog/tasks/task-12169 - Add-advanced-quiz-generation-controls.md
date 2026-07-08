---
id: TASK-12169
title: Add advanced quiz generation controls
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-07 21:30'
labels: []
dependencies: []
references:
  - Spec review approved by subagent 019f3dfe-9e1d-7762-8bf9-88a8e354e13f
  - Final spec review approved by subagent 019f3e03-825a-7d80-aa7d-3c5c27de712f
  - Plan review approved by subagent 019f3e20-d73f-70d2-8bf9-1074c1c4c5bc
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the first-pass Advanced Quiz Studio controls for generated quizzes: exact per-type question counts and configurable MCQ option counts, including 5-option MCQs. Reuse the existing quiz generation endpoint, schemas, and WebUI quiz creation flow; do not add visual-question generation in this task.

Acceptance criteria:
- Quiz generation accepts a structured question plan or equivalent fields for per-type counts.
- MCQ generation supports at least 4-option and 5-option questions without truncating to four choices.
- Existing default quiz generation behavior remains backward compatible.
- WebUI exposes the new controls in the quiz creation flow.
- Tests cover 5-option MCQ generation/parsing and mixed per-type counts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-07-advanced-quiz-generation-controls-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented TASK-12169 advanced quiz generation controls.

Touched backend files:
- tldw_Server_API/app/api/v1/schemas/quizzes.py
- tldw_Server_API/app/api/v1/endpoints/quizzes.py
- tldw_Server_API/app/services/quiz_generator.py
- tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py
- tldw_Server_API/tests/Quizzes/test_quiz_generator_question_plan.py
- tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py
- tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py
- tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py

Touched frontend files:
- apps/packages/ui/src/services/quizzes.ts
- apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx
- apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.question-plan.test.tsx

Design and plan:
- Docs/superpowers/specs/2026-07-07-advanced-quiz-generation-controls-design.md
- Docs/superpowers/plans/2026-07-07-advanced-quiz-generation-controls-implementation-plan.md
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented advanced quiz generation controls end to end. Backend now accepts a structured question_plan, validates exact per-type counts and option/pair ranges at both schema and service layers, preserves legacy question_types behavior, renders plan-aware prompts, normalizes 5-option MCQ, multi-select, matching, true/false, and fill-blank outputs, and supports deterministic planned generation in test mode. The WebUI Generate tab now exposes a fixed five-row question mix, derives num_questions from enabled rows, submits question_plan, clamps integer numeric controls to backend ranges, and keeps remediation/legacy request types available elsewhere.

Verification:
- Backend focused suite: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py tldw_Server_API/tests/Quizzes/test_quiz_generator_question_plan.py tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py -q -> 120 passed, 5 warnings.
- Frontend focused suite from apps/tldw-frontend: bunx vitest run ../packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.question-plan.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx -> 2 files passed, 17 tests passed.
- Bandit: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit tldw_Server_API/app/api/v1/schemas/quizzes.py tldw_Server_API/app/api/v1/endpoints/quizzes.py tldw_Server_API/app/services/quiz_generator.py -f json -o /tmp/bandit_task_12169.json -> exit 0, 0 findings.

Known caveats:
- Full frontend TypeScript no-emit check was attempted from apps/tldw-frontend and failed on existing unrelated files outside the touched quiz scope: AudioStudio, ScheduledTasks, Skills Manager, scheduled task services, MCP hub, voice-cloning, and e2e fixtures. No touched quiz files were reported in that output.
- Vitest emits existing AntD Alert message deprecation warnings during GenerateTab tests.
- Two unrelated untracked watchlist template files remain untouched in the worktree.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #2686 review follow-up after rebase onto origin/dev: addressed reviewer comments for planned quiz generation by keeping QuizGenerateRequest validation non-mutating, adding requested docstrings/test markers, accepting planned multi_select letter answers, canonicalizing matching correct_answer keys case-insensitively, and moving legacy prompt hint cleanup to the static template so source evidence is not rewritten. Verification: backend focused quiz suite passed (124 passed, 4 warnings); frontend focused GenerateTab suite passed (17 passed, 2 files); Bandit on touched backend schema/service files exited 0 with 0 findings.
UX/HCI follow-up: addressed the senior UX review findings for the Quiz Generate page. Reworked the page into a two-column workbench with a selected-source tray, sticky generation brief, inline blocking reasons, retry actions for source-load errors, clearer study-material availability messaging, 4/5-option presets for option-based question types, localized difficulty labels, and stronger preview action hierarchy. Verification: bunx prettier --write on the touched GenerateTab/test files; bunx vitest run -c vitest.extension.config.ts ../packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.question-plan.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx passed (20 tests); git diff --check passed. Bandit was run against the touched TSX files and produced no findings, but reported TSX syntax parse errors because it is a Python scanner.
Code review follow-up for PR #2688: addressed two Important findings from the requesting-code-review subagent. Combined quiz+flashcard generation now propagates the active AbortSignal through generateFlashcards, createDeck, and createFlashcard, checks for abort after per-card saves, and avoids setting preview state after cancellation. The study-materials checkbox is now bound directly through a noStyle Form.Item so removing the selected media clears the visible checked state. Added regression coverage for canceling during the flashcard leg, clearing the study-materials toggle after media removal, and flashcard service signal forwarding. Verification: bunx vitest run -c vitest.extension.config.ts ../packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.question-plan.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx ../packages/ui/src/services/__tests__/flashcards.test.ts passed (31 tests); git diff --check passed. Bandit was rerun against the touched TS/TSX files and returned no findings, but the files are non-applicable for Bandit because it reports TS/TSX AST parse errors.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
