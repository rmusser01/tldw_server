# Advanced Quiz Generation Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact per-type quiz generation controls, including 5-option MCQs, multi-select, and matching generation.

**Architecture:** Add a backward-compatible `question_plan` request contract, normalize it into an internal generation plan, and make planned generation strict while leaving legacy generation best-effort. The WebUI will submit a fixed five-row plan from `GenerateTab`; no quiz storage schema changes are needed.

**Tech Stack:** FastAPI, Pydantic, existing quiz generation service, SQLite-backed `CharactersRAGDB`, React, Ant Design, Vitest, pytest.

---

## File Structure

- Modify `tldw_Server_API/app/api/v1/schemas/quizzes.py`
  - Add `QuizQuestionPlanItem`.
  - Add `question_plan` to `QuizGenerateRequest`.
  - Validate mutual exclusion, explicit `num_questions`, sum equality, duplicate types, and type-specific option/pair count rules.

- Modify `tldw_Server_API/app/api/v1/endpoints/quizzes.py`
  - Pass `request.question_plan` to `generate_quiz_from_sources`.

- Modify `tldw_Server_API/app/services/quiz_generator.py`
  - Split legacy defaults from supported generated types.
  - Normalize legacy requests and planned requests into an internal plan.
  - Render plan-aware prompt instructions.
  - Normalize MCQ, multi-select, matching, true/false, and fill-blank output.
  - Enforce exact planned counts before persistence.
  - Add deterministic test-mode support for all five types.

- Modify `apps/packages/ui/src/services/quizzes.ts`
  - Add `QuizQuestionPlanItem` type.
  - Add `question_plan` to `QuizGenerateRequestBase`.

- Modify `apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx`
  - Replace simple generated-type controls with a fixed five-row plan state.
  - Derive `num_questions` from enabled rows and submit `question_plan`.

- Add or modify tests:
  - `tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py`
  - `tldw_Server_API/tests/Quizzes/test_quiz_generator_question_plan.py`
  - `tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py`
  - `tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py`
  - `tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py`
  - `apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.question-plan.test.tsx`

## Task 1: Backend Schema Contract

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/quizzes.py`
- Modify: `tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py`

- [ ] **Step 1: Write failing schema tests**

Add tests covering a valid plan and invalid plan cases:

```python
def test_quiz_generate_request_accepts_question_plan():
    request = QuizGenerateRequest.model_validate(
        {
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "num_questions": 5,
            "question_plan": [
                {"question_type": "multiple_choice", "count": 3, "option_count": 5},
                {"question_type": "matching", "count": 2, "pair_count": 4},
            ],
        }
    )

    assert request.question_plan is not None
    assert request.question_plan[0].option_count == 5
    assert request.question_plan[1].pair_count == 4
```

Also add parameterized invalid cases for:

- `question_types` with `question_plan`
- missing explicit `num_questions`
- sum mismatch
- duplicate `question_type`
- extra fields
- `option_count` on `true_false`
- `pair_count` outside `2-6`

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py -q
```

Expected: FAIL because `question_plan` does not exist yet.

- [ ] **Step 3: Implement schema**

In `quizzes.py`, add a model near `QuizGenerateSource`:

```python
class QuizQuestionPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_type: QuestionType
    count: int = Field(..., ge=1, le=100)
    option_count: Optional[int] = Field(None, ge=2, le=6)
    pair_count: Optional[int] = Field(None, ge=2, le=6)
```

Add `question_plan: Optional[list[QuizQuestionPlanItem]] = Field(None, min_length=1)` to `QuizGenerateRequest`.

In `validate_media_id_or_sources`, add:

- If `question_plan` is present, require `"num_questions"` in `self.model_fields_set`.
- Reject `question_types` when `question_plan` is present.
- Reject duplicate `question_type` rows.
- Reject sum mismatch.
- For MCQ and multi-select, default `option_count` to `4` and reject `pair_count`.
- For matching, default `pair_count` to `4` and reject `option_count`.
- For true/false and fill-blank, reject both count fields.

- [ ] **Step 4: Run schema tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/quizzes.py tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py
git commit -m "feat: add quiz generation question plan schema"
```

## Task 2: Generation Plan Normalization And Prompt

**Files:**
- Modify: `tldw_Server_API/app/services/quiz_generator.py`
- Modify: `tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py`
- Create: `tldw_Server_API/tests/Quizzes/test_quiz_generator_question_plan.py`

- [ ] **Step 1: Write failing helper and prompt tests**

Create tests for:

- 5-option MCQ options are not truncated.
- MCQ answers support letters beyond `D`, such as `E`.
- Multi-select validation rejects empty, duplicate, or out-of-range indices, and sorts otherwise-valid indices.
- Matching validation accepts `options=["CPU", "RAM"]` with `correct_answer={"CPU": "Processor", "RAM": "Memory"}`.
- Planned true/false validation rejects anything other than exact `"true"` or `"false"`.
- Planned fill-blank validation rejects questions without `___` or with an empty answer.
- Prompt output includes plan instructions for all five question types.

Example focused test:

```python
def test_normalize_mc_answer_supports_five_options():
    options = ["A", "B", "C", "D", "E"]

    assert quiz_generator._normalize_mc_answer("E", options) == 4
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_question_plan.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py -q
```

Expected: FAIL because planned helpers do not exist and MCQ answers only allow A-D.

- [ ] **Step 3: Implement minimal helpers**

In `quiz_generator.py`:

- Keep `DEFAULT_QUESTION_TYPES = ["multiple_choice", "true_false", "fill_blank"]`.
- Add `SUPPORTED_GENERATED_QUESTION_TYPES = ["multiple_choice", "multi_select", "matching", "true_false", "fill_blank"]`.
- Add internal plan helpers that accept either legacy `question_types` or schema `question_plan`.
- Keep legacy `_coerce_question_types` limited to `DEFAULT_QUESTION_TYPES`; generated `multi_select` and `matching` stay behind `question_plan` for v1.
- Change `_coerce_options(raw, expected_count=None)` so planned mode requires exact counts and legacy mode keeps the existing best-effort four-option cap.
- Update `_normalize_mc_answer` to accept any single-letter answer inside the current option length.
- Add planned normalizers for MCQ, multi-select, matching, true/false, and fill-blank with strict validation.
- Add a small prompt formatter that renders exact plan rows and output shapes.

Keep these helpers private in `quiz_generator.py`; do not create a new module unless the file becomes unmanageable during implementation.

- [ ] **Step 4: Run helper and prompt tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_question_plan.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/services/quiz_generator.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_question_plan.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py
git commit -m "feat: normalize planned quiz generation output"
```

## Task 3: Planned Generation Service And Endpoint

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/quizzes.py`
- Modify: `tldw_Server_API/app/services/quiz_generator.py`
- Modify: `tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py`
- Modify: `tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py`

- [ ] **Step 1: Write failing service and integration tests**

Add tests for:

- `_build_test_mode_questions` produces all five planned types.
- Planned generation persists exact type counts.
- MCQ and multi-select option counts match request.
- Matching pair counts match request.
- Planned generation failure leaves no quiz behind.
- Planned generation failure includes a stable message substring such as `expected 5, got 4`.

Use `TEST_MODE=1` for deterministic endpoint coverage where possible.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py \
  tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py \
  -k "question_plan or generate" -q
```

Expected: FAIL because endpoint and service do not pass/use `question_plan` yet.

- [ ] **Step 3: Wire endpoint and service**

In `endpoints/quizzes.py`, pass `question_plan=request.question_plan` to `generate_quiz_from_sources`.

In `quiz_generator.py`:

- Add `question_plan` parameter to `generate_quiz_from_sources`.
- Pass it through the legacy media wrapper only when needed; existing `generate_quiz_from_media` can omit it.
- Build planned prompt when `question_plan` is present.
- Use dynamic `max_tokens`, choosing a concrete formula during implementation. Start simple: `max_tokens = min(8000, max(2000, num_questions * 220))`.
- Normalize and bucket generated questions by plan row.
- Fail before `_persist_generated_quiz` if any planned row is short.
- Keep strict source provenance validation unchanged.

- [ ] **Step 4: Run service and endpoint tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py \
  tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py \
  -k "question_plan or generate" -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/quizzes.py \
  tldw_Server_API/app/services/quiz_generator.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py \
  tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py
git commit -m "feat: generate quizzes from exact question plans"
```

## Task 4: WebUI Question Plan Controls

**Files:**
- Modify: `apps/packages/ui/src/services/quizzes.ts`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx`
- Create: `apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.question-plan.test.tsx`

- [ ] **Step 1: Write failing frontend tests**

Add tests that:

- Render the five fixed question-plan rows.
- Assert the default row state: MCQ enabled count `5` option count `4`, true/false enabled count `3`, fill-blank enabled count `2`, multi-select disabled count `1` option count `4`, matching disabled count `1` pair count `4`.
- Update calculated total when row counts change.
- Disable Generate when total is `0` or greater than `100`.
- Disable row controls while generation is in flight.
- Keep row controls usable on narrow/mobile layouts without horizontal table overflow.
- Submit a 5-option MCQ row with `num_questions` and `question_plan`.

Mock the generate mutation/service and assert the request body instead of snapshotting the whole page.

- [ ] **Step 2: Run frontend tests and verify failure**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.question-plan.test.tsx
```

Expected: FAIL because the UI still submits `question_types`.

- [ ] **Step 3: Add frontend types**

In `services/quizzes.ts`, add:

```ts
export type QuizQuestionPlanItem = {
  question_type: QuestionType
  count: number
  option_count?: number
  pair_count?: number
}
```

Add `question_plan?: QuizQuestionPlanItem[]` to `QuizGenerateRequestBase`.

- [ ] **Step 4: Implement fixed-row plan UI**

In `GenerateTab.tsx`:

- Replace `QUESTION_TYPE_OPTIONS` form checkbox usage with fixed controlled rows.
- Keep five rows in local state.
- Remove or replace the old editable `numQuestions` field with a read-only calculated total display when submitting `question_plan`.
- Initialize rows with the spec defaults:
  - MCQ enabled, count `5`, option count `4`
  - True/false enabled, count `3`
  - Fill-blank enabled, count `2`
  - Multi-select disabled, count `1`, option count `4`
  - Matching disabled, count `1`, pair count `4`
- Derive `enabledPlanRows` and `totalQuestions`.
- Submit:

```ts
request: {
  sources: selectedSources,
  num_questions: totalQuestions,
  question_plan: enabledPlanRows.map(toQuizQuestionPlanItem),
  difficulty: values.difficulty,
  focus_topics: focusTopics.length > 0 ? focusTopics : undefined
}
```

- Keep difficulty, source selection, focus topics, and study-material generation unchanged.
- Use Ant Design `InputNumber` min/max values that match backend validation:
  - Count: `1-100`
  - Option/pair count: `2-6`
- Keep Generate disabled when no sources, generation is in flight, or total is outside `1-100`.

- [ ] **Step 5: Run frontend tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.question-plan.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/services/quizzes.ts \
  apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx \
  apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.question-plan.test.tsx
git commit -m "feat: add quiz generation mix controls"
```

## Task 5: Final Verification And Backlog Update

**Files:**
- Modify: `backlog/tasks/task-12169 - Add-advanced-quiz-generation-controls.md`

- [ ] **Step 1: Run focused backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_question_plan.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py \
  tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run focused frontend tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.question-plan.test.tsx
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched backend code**

Run:

```bash
source .venv/bin/activate && python -m bandit \
  tldw_Server_API/app/api/v1/schemas/quizzes.py \
  tldw_Server_API/app/api/v1/endpoints/quizzes.py \
  tldw_Server_API/app/services/quiz_generator.py \
  -f json -o /tmp/bandit_task_12169.json
```

Expected: No new findings in touched code.

- [ ] **Step 4: Update backlog task**

Record:

- Implementation summary.
- Files touched.
- Verification commands and results.
- Any skipped full-suite checks.

- [ ] **Step 5: Commit final task update**

```bash
git add 'backlog/tasks/task-12169 - Add-advanced-quiz-generation-controls.md'
git commit -m "chore: record TASK-12169 verification"
```

## Notes For Implementers

- Do not add database migrations. The existing quiz question schema already supports all required shapes.
- Do not add retry loops around the LLM call in this task. Clear failure is enough for v1.
- Do not add visual-question generation.
- Do not add mixed option-count rows for the same question type.
- Keep all new behavior behind `question_plan`; legacy requests should keep working.
