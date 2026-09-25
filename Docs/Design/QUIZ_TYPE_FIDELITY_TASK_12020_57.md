# Quiz type fidelity

TASK-12020.57 implements the approved review findings in the existing quiz
generation flow. Legacy `question_types` identifies allowed types, while an
explicit `question_plan` also requires exact per-type counts.

## Stage 1: Reproduce
**Goal**: Capture unwanted, dropped, and count-pruned types before production edits.
**Success Criteria**: Focused regressions fail for the observed defects.
**Tests**: Raw unexpected types, advanced legacy types, prompt subsets, tiny counts.
**Status**: Complete. Focused tests reproduced 11 type-fidelity failures.

## Stage 2: Implement
**Goal**: Validate raw output before normalization and render a consistent prompt.
**Success Criteria**: Requested types are enforced before claims and persistence;
legacy flexible mixes and explicit plan counts retain their contracts.
**Tests**: Focused, property, endpoint/integration and full Quizzes regression.
**Status**: Complete. Raw output is checked before normalization, selected types
remain an allowlist for legacy requests, and the rendered prompt describes only
selected types while preserving locked-profile fields.

## Stage 3: Real application
**Goal**: Run the browser workspace quiz flow against the actual llama.cpp model.
**Success Criteria**: Persisted questions all have requested types, grounded
claims, correct answers, and open in the native quiz page.
**Tests**: Real-backend Chromium spec and saved response/persistence evidence.
**Status**: Complete on the original branch: real llama.cpp Chromium UAT passed
with one executed test, grounded claims, selected persisted types, and native
quiz-page access. The port onto current `dev` has not repeated that success:
four live-model attempts failed closed on malformed/empty output or invalid
source citations before persistence. See
`Docs/Reviews/RESEARCH_WORKSPACE_QUIZ_TYPE_FIDELITY_2026_09_25.md`.
