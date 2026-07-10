# Advanced Quiz Customization PRD

Status: Proposed
Owner: Product / WebUI / API
Created: 2026-07-10
Tracking: TASK-12102

## Summary

tldw_server already has a quiz API, quiz generation from mixed sources, quiz
attempts, results, remediation, flashcard integration, and WebUI/extension quiz
routes. The current generated quiz flow supports fewer formats than the stored
question model and does not expose enough exam-style controls.

This PRD expands quiz generation and authoring into a configurable assessment
system with Best of Five, EMQ, assertion/reasoning, and OSCE-style scenarios,
while preserving source citations and clear grading semantics.

## Problem

Users want practice assessments that match real study and professional exam
formats. Current generated quizzes are useful for basic recall, but they do not
cover many high-value formats used in medicine, professional training, and
advanced study. The product also needs controls for per-type counts, difficulty
mix, citation requirements, and remediation behavior.

## Goals

- Add richer quiz generation controls without breaking existing quizzes.
- Support Best of Five as the first advanced generated format.
- Support EMQ-style grouped questions with shared option banks.
- Support assertion/reasoning questions using concise rationale, not hidden
  chain-of-thought.
- Support OSCE-style scenario practice with rubric/checklist feedback in a
  clearly labeled mode.
- Preserve source citations and transparent answer explanations.
- Keep WebUI and extension quiz routes consistent.

## Non-Goals

- High-stakes certification or clinical decision support.
- Fully automated authoritative grading for OSCE/free-text answers.
- Proctoring, identity verification, or anti-cheat systems.
- Replacing flashcards or study packs.
- Exposing hidden chain-of-thought reasoning.

## Existing Foundation

- Backend schemas: `tldw_Server_API/app/api/v1/schemas/quizzes.py`
- Backend endpoint: `tldw_Server_API/app/api/v1/endpoints/quizzes.py`
- Generator: `tldw_Server_API/app/services/quiz_generator.py`
- Source resolver: `tldw_Server_API/app/services/quiz_source_resolver.py`
- WebUI service: `apps/packages/ui/src/services/quizzes.ts`
- WebUI quiz components: `apps/packages/ui/src/components/Quiz/`
- Extension route: `apps/tldw-frontend/extension/routes/option-quiz.tsx`
- Prior design: `Docs/Plans/2026-03-03-quiz-multi-source-generation-design.md`

Current stored `QuestionType` values include:

- `multiple_choice`
- `multi_select`
- `matching`
- `true_false`
- `fill_blank`

Current generation defaults are narrower:

- `multiple_choice`
- `true_false`
- `fill_blank`

This creates a practical path: expand generation and metadata first, then add
new grouped/rubric formats where the current schema is insufficient.

## User Stories

- As a medical student, I can generate Best of Five questions from lecture notes
  with source-backed explanations.
- As a student preparing for exams, I can generate EMQs with shared option banks
  across several stems.
- As a learner practicing reasoning, I can answer assertion/reason questions and
  get concise feedback on the logical relationship.
- As a clinical trainee, I can practice OSCE-style scenarios with checklist
  feedback that is clearly marked as study guidance.
- As an instructor, I can choose a mix of question types, difficulty, and number
  of questions before generation.

## Product Requirements

### PR-1: Generation Profiles

Quiz generation must expose structured profiles:

- Standard recall
- Mixed assessment
- Best of Five
- EMQ
- Assertion / reasoning
- OSCE scenario

Profiles should control prompt templates, validation rules, default question
counts, allowed answer shapes, and grading policy.

### PR-2: Per-Type Controls

Users should be able to configure:

- Total question count
- Per-type question counts
- Difficulty mix
- Focus topics
- Source bundle
- Citation requirement
- Explanation style
- Practice vs graded mode defaults
- Whether generated quizzes should create flashcards after completion

### PR-3: Best Of Five

Best of Five should be represented as a constrained multiple-choice variant:

- Exactly five answer options
- One best answer
- Plausible distractors
- Source-backed explanation
- Optional tags indicating topic and difficulty

This can ship without a new base `QuestionType` if subtype metadata is added.

### PR-4: EMQ

EMQ requires grouped stems sharing a common option bank. The product should
support:

- Scenario or theme prompt
- Shared option list
- Multiple stems
- One best option per stem
- Per-stem explanation and citations

Implementation may use a new grouped-question envelope instead of forcing EMQs
into independent matching questions. The PRD prefers an explicit grouped model
for long-term clarity.

### PR-5: Assertion / Reasoning

Assertion/reasoning questions should support:

- Assertion statement
- Reason statement
- Answer set that captures whether each statement is true and whether the
  reason explains the assertion
- Concise explanation
- Source citations

The system must not ask the model to reveal hidden chain-of-thought. It should
request a concise rationale and evidence summary.

### PR-6: OSCE Scenario Practice

OSCE mode should be treated as a practice simulation, not normal quiz grading.
It should support:

- Station prompt
- Candidate task
- Patient/context information
- Checklist items
- Marking rubric
- Expected key points
- Feedback summary
- Optional free-text answer or self-assessment

LLM-assisted grading must be labeled as advisory and should preserve rubric
evidence. The MVP may support self-marked checklist mode before automated
free-text scoring.

### PR-7: Source Citations And Explanations

Advanced generated questions must include citations where sources are available.
The UI should show:

- Question source badges
- Explanation citations
- Quote snippets, bounded in length
- Missing-citation warnings when generation could not cite a question

### PR-8: Backward Compatibility

Existing quizzes, attempts, and results must continue to work.

New fields should be additive:

- `question_subtype`
- `generation_profile`
- `question_metadata`
- `group_id` or grouped-question table/envelope
- `rubric_json` for OSCE-style scoring

## UX Requirements

- The Generate tab should show profile presets before advanced controls.
- Best of Five should look like normal MCQ taking with five options.
- EMQ should present shared options once and several stems together.
- Assertion/reasoning should clearly explain the answer scale before the first
  question.
- OSCE should use a scenario/checklist interface rather than pretending to be a
  normal auto-graded MCQ.
- Results should show profile-specific feedback and citations.
- Extension should reuse the same quiz workspace where possible.

## API And Data Model Direction

Prefer an additive contract:

- Keep existing `QuestionType` values.
- Add profile/subtype metadata for BOF and assertion/reasoning.
- Add grouped-question support for EMQ if existing matching semantics become
  too contorted.
- Add OSCE station/rubric models separately from normal questions if rubric
  scoring does not fit `AnswerValue`.

Potential fields:

- `Quiz.generation_profile`
- `Question.question_subtype`
- `Question.metadata_json`
- `Question.group_key`
- `Question.rubric_json`
- `AttemptAnswer.feedback_json`

Potential endpoints:

- Extend `POST /api/v1/quizzes/generate`
- Add `GET /api/v1/quizzes/generation-profiles`
- Add `POST /api/v1/quizzes/{id}/questions/groups` if grouped questions need a
  first-class API
- Add OSCE-specific attempt feedback endpoint only if normal attempt submission
  cannot model the interaction cleanly

## Backend Requirements

- Add profile-aware prompt templates.
- Add strict validators per generated profile.
- Add deterministic answer-shape normalization for BOF, EMQ, and
  assertion/reasoning.
- Add advisory rubric feedback path for OSCE.
- Preserve source bundle metadata and citations.
- Record metrics by profile, source type, validation failure, and generation
  latency.
- Add regression tests for legacy quiz behavior.

## WebUI And Extension Requirements

- Extend generation controls with profile presets and advanced controls.
- Update create/manage flows to display/edit subtype metadata where supported.
- Update take/results flows for BOF, EMQ, assertion/reasoning, and OSCE.
- Keep unavailable formats hidden behind capability flags until backend support
  is present.
- Preserve extension route parity through the shared quiz workspace.

## Safety And Accuracy

- OSCE and free-text evaluation must be marked as study guidance, not medical or
  professional assessment.
- Generated explanations must cite source evidence when source-backed.
- Hidden chain-of-thought must not be requested or displayed.
- Users should be able to inspect and edit generated questions before use.
- Prompt and output validation must fail closed rather than saving malformed
  question structures.

## Success Metrics

- Users can generate a Best of Five quiz with five options per question.
- Users can generate an EMQ set with shared options and multiple stems.
- Users can generate assertion/reasoning questions with correct answer scale and
  concise rationale.
- Users can run an OSCE-style self-marked station with checklist feedback.
- Existing quiz generation and attempts remain backward compatible.
- Source citations are visible in generated advanced question results.

## Rollout Plan

### Phase 1: Profiles And Generation Controls

Add generation profile registry, schema extensions, frontend profile controls,
and compatibility tests.

### Phase 2: Best Of Five MVP

Implement BOF as a constrained MCQ subtype. Add backend validation, generation
prompting, UI labels, and result display.

### Phase 3: EMQ

Add grouped option-bank support and grouped UI for taking/results. Add source
citations per stem.

### Phase 4: Assertion / Reasoning

Add answer scale, generation, validation, taking UI, and concise rationale
display.

### Phase 5: OSCE Scenario Practice

Add scenario/checklist/rubric model and self-marked flow. Add advisory
LLM-feedback only after rubric persistence and review UX are stable.

### Phase 6: Documentation And Evaluation

Document profile behavior, limitations, examples, and testing expectations.
Add eval fixtures for generated output validity.

## Backlog Task Set

- Parent: Implement Advanced Quiz Customization.
- Phase 1: Add quiz generation profiles and controls.
- Phase 2: Implement Best of Five generated quizzes.
- Phase 3: Implement EMQ grouped question support.
- Phase 4: Implement assertion/reasoning questions.
- Phase 5: Implement OSCE scenario practice.
- Phase 6: Add docs, examples, and generated-output validation fixtures.

## Open Questions

- Should BOF be only a subtype of multiple choice, or should it be a new
  `QuestionType`?
- Should EMQ grouped questions require a new table, or can they live in
  metadata while preserving clean authoring?
- Should OSCE automated feedback be deferred until self-marking is proven useful?
- Which profiles should be available in the extension MVP?
