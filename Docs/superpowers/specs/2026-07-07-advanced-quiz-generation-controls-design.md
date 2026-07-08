# Advanced Quiz Generation Controls Design

Task: TASK-12169

## Goal

Add first-pass Advanced Quiz Studio controls to generated quizzes. Users should be able to request exact counts per generated question type and configure MCQ, multi-select, and matching option counts, with 5-option MCQs as a first-class use case.

This design intentionally reuses the existing quiz endpoint, quiz storage, source resolver, grading logic, and WebUI generation flow. It does not add visual questions, image extraction, or a new quiz system.

## Current State

Generated quizzes currently accept `num_questions` and `question_types`. The backend generation service only allows `multiple_choice`, `true_false`, and `fill_blank` in generated output. The prompt and parser hard-code MCQs to four options and truncate longer option lists.

Manual quiz creation, quiz storage, grading, import/export, and the take UI already support all five quiz types: `multiple_choice`, `multi_select`, `matching`, `true_false`, and `fill_blank`. Matching questions store `options` as left-side labels and `correct_answer` as a left-to-right mapping. Multi-select questions store `correct_answer` as sorted option indices.

## API Contract

Add an optional `question_plan` field to `QuizGenerateRequest`.

Example:

```json
{
  "sources": [{ "source_type": "note", "source_id": "abc" }],
  "num_questions": 12,
  "question_plan": [
    { "question_type": "multiple_choice", "count": 5, "option_count": 5 },
    { "question_type": "multi_select", "count": 2, "option_count": 5 },
    { "question_type": "matching", "count": 2, "pair_count": 4 },
    { "question_type": "true_false", "count": 2 },
    { "question_type": "fill_blank", "count": 1 }
  ]
}
```

Each plan row will use a Pydantic model with `extra="forbid"`:

- `question_type`: one of the five quiz question types.
- `count`: `1-100`.
- `option_count`: for `multiple_choice` and `multi_select`, default `4`, range `2-6`.
- `pair_count`: for `matching`, default `4`, range `2-6`.

Validation rules:

- `question_plan` and `question_types` are mutually exclusive.
- `num_questions` must be explicitly present when `question_plan` is present.
- The sum of all plan row counts must equal `num_questions`.
- Duplicate `question_type` rows are rejected in v1.
- Type-specific count fields are rejected on incompatible rows.
- The existing `num_questions <= 100` cap remains the total quiz cap.

Legacy requests without `question_plan` keep the current API behavior: `num_questions` plus optional `question_types`, defaulting to the legacy generated types `multiple_choice`, `true_false`, and `fill_blank`.

## Generation Service

Refactor generation around an internal normalized plan while preserving legacy behavior.

Legacy requests:

- Build the current best-effort generation request from `num_questions` and `question_types`.
- Keep legacy defaults as `multiple_choice`, `true_false`, and `fill_blank`.
- Keep loose distribution and best-effort normalization.

Planned requests:

- Use `question_plan` as the only generation shape.
- Build prompt instructions that state exact required counts and option/pair counts.
- Support all five generated question types.
- Dynamically size `max_tokens` based on `num_questions`, with a sane cap, because the current fixed `2000` tokens may be too low for large planned quizzes.

Expected planned output shapes:

- `multiple_choice`: `options` length exactly `option_count`, `correct_answer` is an integer index.
- `multi_select`: `options` length exactly `option_count`, `correct_answer` is a non-empty, unique integer index array with every index in range. Otherwise-valid unsorted indices may be sorted before storage.
- `matching`: `options` length exactly `pair_count`, containing left-side terms; `correct_answer` is an object mapping every left term to a unique right-side value.
- `true_false`: `correct_answer` is exactly `"true"` or `"false"`.
- `fill_blank`: `question_text` contains `___`, `correct_answer` is non-empty text.

Planned validation is stricter than legacy validation:

- Reject malformed questions for the requested type instead of repairing across types.
- Do not trim extra options or pairs, because trimming can invalidate answers.
- Require valid source citations before persistence.
- Require exact counts by plan row before persistence.
- If the model returns too few valid questions, fail the entire generation before any quiz or question is saved.

Failure message should be stable enough to test by substring, for example:

```text
Generated quiz did not satisfy requested question plan: multiple_choice expected 5, got 4
```

Deterministic test mode must support the same plan path, including `multi_select` and `matching`, so planned endpoint tests do not bypass the new behavior.

## WebUI

Update `GenerateTab` only.

Use a fixed five-row controlled state instead of a dynamic builder. Each row represents one supported type:

- Enabled checkbox.
- Count input.
- Option count input for MCQ and multi-select.
- Pair count input for matching.

Defaults:

- MCQ enabled, count `5`, option count `4`.
- True/false enabled, count `3`.
- Fill-blank enabled, count `2`.
- Multi-select disabled, count `1`, option count `4`.
- Matching disabled, count `1`, pair count `4`.

The default enabled total remains `10`. This intentionally adds fill-blank to the default WebUI mix; the current WebUI default is MCQ plus true/false. The change is acceptable because it aligns the UI with the backend legacy default and the advanced mix goal.

UI behavior:

- Show a read-only calculated total.
- Derive `num_questions` from enabled row counts and submit it with `question_plan`.
- Do not let users edit `num_questions` separately when `question_plan` is used.
- Disable Generate when total is `0` or greater than `100`.
- Disable controls while generation is in flight.
- Render the rows as a compact stacked list on mobile rather than a wide table.
- Keep source selection, difficulty, focus topics, and study-material generation unchanged.

The WebUI should use `question_plan` for new generations. Legacy API compatibility remains for external clients and older UI flows.

## Tests

Backend schema tests:

- Valid `question_plan` is accepted.
- Duplicate types are rejected.
- Unknown extra fields are rejected.
- `question_types` plus `question_plan` is rejected.
- Missing explicit `num_questions` with `question_plan` is rejected.
- Sum mismatch is rejected.
- Option and pair counts outside `2-6` are rejected.

Backend generation tests:

- Normalization keeps 5-option MCQs and does not truncate.
- Planned deterministic test-mode output supports all five types.
- Matching normalization supports the generated shape: `options=["A","B"]`, `correct_answer={"A":"x","B":"y"}`.
- Multi-select validation rejects empty, duplicate, or out-of-range indices and stores otherwise-valid unsorted indices sorted.
- Prompt-template test verifies plan instructions include all five output shapes and requested counts.
- Planned generation failure leaves no quiz behind by checking quiz count before and after.

Backend integration test:

- `/api/v1/quizzes/generate` accepts a planned request containing MCQ, multi-select, matching, true/false, and fill-blank.
- The returned and persisted questions match requested type counts.
- MCQ and multi-select questions have requested option counts.
- Matching questions have requested pair counts.

Frontend tests:

- `GenerateTab` renders the five fixed plan rows.
- Calculated total updates as counts change.
- Generate is disabled when total is `0` or greater than `100`.
- A 5-option MCQ generation submits a payload containing both derived `num_questions` and `question_plan`.
- Payload tests should mock the generate call and assert only the request body, not snapshot the full UI.

## Out Of Scope

- Visual MCQs or visual flashcards generated from source figures/images.
- Multiple rows for the same question type, such as both 4-option and 5-option MCQs in one generation.
- Persisting plan metadata after quiz creation.
- Changing quiz grading, import/export, or take-session storage.
- Redesigning the full quiz page.

## Implementation Notes

- Split the current `DEFAULT_QUESTION_TYPES` meaning into legacy defaults and supported generated types.
- Keep storage unchanged; existing DB schema supports all planned question shapes.
- Use strict planned validation only when `question_plan` is present.
- Keep legacy generated behavior backward compatible for clients that only send `num_questions` and `question_types`.
