# Assertion / Reasoning Questions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship source-grounded assertion/reasoning quiz generation and taking with a canonical five-outcome scale, concise rationales, and no hidden chain-of-thought request.

**Architecture:** Keep assertion/reasoning as a constrained `multiple_choice` profile. The generator receives separate assertion and reason fields, stores them as labeled Markdown in `question_text`, replaces model-supplied options with one server-owned canonical scale, normalizes the answer strictly, and persists the existing `assertion_reasoning` tag as subtype metadata. The WebUI detects that tag, keeps the canonical option order fixed, and renders one scale explanation in graded, practice, review, and result views. No database migration, new endpoint, state store, or extension-specific flow is required.

**Tech Stack:** FastAPI/Pydantic, Python quiz generation service, ChaChaNotes quiz persistence, React/TypeScript, Ant Design, Vitest, pytest, Ruff, Bandit.

---

### Task 1: Generate and validate canonical assertion/reasoning questions

**Stage:** 1 - Backend contract and persistence
**Goal:** Make the profile available and guarantee a deterministic answer shape before persistence.
**Success Criteria:** Every generated item has separate source statements, a server-owned five-option scale, one unambiguous answer, a concise rationale, citations, and persisted subtype metadata.
**Tests:** Generator registry, prompt, normalization, test-mode, final validation, and persistence tests.
**Status:** Not Started

**Files:**
- Modify: `tldw_Server_API/app/services/quiz_generator.py`
- Modify: `tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py`
- Modify: `tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py`
- Modify: `tldw_Server_API/tests/Quizzes/test_quiz_generate_endpoint_multi_source.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_quiz_assertion_reasoning_subtype.py`

- [ ] **Step 1: Write failing registry and prompt tests**

Add coverage that `assertion_reasoning` is `available`, defaults and restricts to `multiple_choice`, and contributes an instruction that:

```text
- requests separate assertion and reason fields
- uses the canonical A-E truth/explanation outcomes
- requests only a concise evidence-based rationale
- explicitly says not to provide hidden chain-of-thought
```

Also update the profile endpoint test to assert the available profile is returned.

- [ ] **Step 2: Run the prompt tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generate_endpoint_multi_source.py -q
```

Expected: assertion/reasoning availability and prompt assertions fail because the profile is still planned.

- [ ] **Step 3: Make only the registry and prompt contract GREEN**

Mark `assertion_reasoning` available, retain its existing MCQ-only defaults, add the profile instruction, and extend the common JSON example with separate `assertion` and `reason` fields. Update the endpoint assertion and remove that test module's existing unused `ChatConfigurationError` import so the touched-file Ruff gate is clean. Do not add answer normalization or specialized question construction yet.

- [ ] **Step 4: Re-run registry and prompt tests and verify GREEN**

Run the Step 2 command. Expected: the new registry/prompt tests and existing profile tests pass.

- [ ] **Step 5: Write failing normalization and persistence tests**

Cover these behaviors in `test_quiz_generator_test_mode.py`:

```python
def test_assertion_reasoning_normalization_uses_canonical_scale_and_tag(): ...
def test_assertion_reasoning_accepts_zero_based_index_letter_or_exact_label(): ...
def test_assertion_reasoning_rejects_missing_assertion_reason_or_rationale(): ...
def test_assertion_reasoning_rejects_invalid_or_ambiguous_answer(): ...
def test_assertion_reasoning_rejects_non_mcq_items(): ...
def test_assertion_reasoning_rejects_overlength_statements_or_rationale(): ...
def test_assertion_reasoning_discards_chain_of_thought_fields(): ...
def test_test_mode_builds_source_grounded_assertion_reasoning_questions(): ...
async def test_generate_quiz_persists_assertion_reasoning_tag_and_evidence(...): ...
async def test_generate_quiz_revalidates_assertion_reasoning_before_persistence(...): ...
```

The successful fixture should resemble:

```python
{
    "question_type": "multiple_choice",
    "assertion": "The intervention improves the measured outcome.",
    "reason": "The cited trial reports a statistically significant improvement.",
    "correct_answer": "A",
    "explanation": "Both statements are supported, and the trial result explains the assertion.",
    "source_citations": [{"source_type": "media", "source_id": "42"}],
}
```

Parameterize all five accepted zero-based integers, all five ASCII letters in upper/lower case with surrounding whitespace, and all five canonical labels case-insensitively with surrounding whitespace. Explicitly reject numeric strings (`"0"`-`"4"`), prefixed labels such as `"A."`, booleans, floats, out-of-range integers, and unknown text. Set exact maximum lengths of 2,000 characters each for assertion, reason, and explanation; test 2,000 as accepted and 2,001 as rejected. Include `reasoning_steps` and `chain_of_thought` in a raw fixture and prove neither key appears in normalized or persisted output.

In the new ChaChaNotesDB test, create a tagged question through the real SQLite-backed `CharactersRAGDB`, verify get/list round trips exactly one canonical subtype tag, call `start_attempt`, and assert the public attempt question retains that tag while still omitting `correct_answer` and all hidden-reasoning fields.

- [ ] **Step 6: Run the normalization tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_quiz_assertion_reasoning_subtype.py -q
```

Expected: only the new normalization/subtype tests fail because no canonical validator, construction path, or test-mode branch exists. Registry/prompt tests are already green.

- [ ] **Step 7: Add the minimal generator implementation**

In `quiz_generator.py`, add immutable constants equivalent to:

```python
ASSERTION_REASONING_TAG = "assertion_reasoning"
ASSERTION_REASONING_OPTIONS = (
    "Both the assertion and reason are true, and the reason correctly explains the assertion.",
    "Both the assertion and reason are true, but the reason does not explain the assertion.",
    "The assertion is true, but the reason is false.",
    "The assertion is false, but the reason is true.",
    "Both the assertion and reason are false.",
)
```

Then:

- Extend `_coerce_question_tags` so this profile always receives exactly one normalized `assertion_reasoning` tag while retaining user topic/difficulty tags.
- Add a strict answer normalizer implementing exactly the accepted/rejected forms from Step 5. Do not accept numeric strings or punctuation-prefixed letters, avoiding one-based/zero-based ambiguity.
- For this profile, require nonempty assertion, reason, and explanation fields of at most 2,000 characters each; construct `question_text` as labeled Markdown; replace raw options with `list(ASSERTION_REASONING_OPTIONS)`; clear EMQ group metadata; and reject non-MCQ items instead of skipping them.
- Build normalized payloads from an explicit allowlist so raw `reasoning_steps`, `chain_of_thought`, and any other unknown model fields are discarded before persistence.
- Revalidate the normalized collection immediately before persistence, matching the existing EMQ fail-closed pattern.
- Add deterministic test-mode questions with citation evidence, concise rationale, the canonical scale, and subtype tag.
- Keep standard, Best of Five, and EMQ branches unchanged.

- [ ] **Step 8: Run focused and neighboring backend tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generate_endpoint_multi_source.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_quiz_assertion_reasoning_subtype.py -q
```

Expected: all selected tests pass, including existing Best of Five and EMQ regressions.

- [ ] **Step 9: Commit Task 1**

```bash
git add tldw_Server_API/app/services/quiz_generator.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generate_endpoint_multi_source.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_quiz_assertion_reasoning_subtype.py
git commit -m "feat: generate assertion reasoning questions"
```

### Task 2: Present the canonical scale in every quiz mode

**Stage:** 2 - WebUI interaction and results
**Goal:** Explain the specialized answer model once and preserve stable A-E semantics throughout taking and review.
**Success Criteria:** The profile is selectable; tagged questions show a subtype label; options never shuffle; one scale guide appears in graded, practice, review, and results; answers and evidence remain per-question.
**Tests:** Profile payload, graded submission/results, practice feedback, review evidence, and legacy MCQ/EMQ regressions.
**Status:** Not Started

**Files:**
- Modify: `apps/packages/ui/src/services/quizzes.ts`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/TakeQuizTab.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.start-flow.test.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.study-modes.test.tsx`

- [ ] **Step 1: Write failing WebUI tests**

Add tests proving:

- The fallback profile registry exposes `assertion_reasoning` as available and Generate submits its defaults.
- Graded taking renders exactly one answer-scale guide, uses the original five-option order, and submits the original numeric index.
- Results render exactly one guide and show the concise rationale/citations for each assertion/reasoning answer.
- Practice mode renders exactly one guide, retains fixed option order, and shows rationale/citations after both correct and incorrect assertion/reasoning answers.
- Review mode renders exactly one guide plus per-question rationale/citations.
- A normal MCQ still shuffles, Best of Five still shuffles while submitting the original numeric index, and EMQ still uses its shared bank/select path.

- [ ] **Step 2: Run the focused frontend tests and verify RED**

From `apps/packages/ui`:

```bash
bun run test \
  src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx \
  src/components/Quiz/tabs/__tests__/TakeQuizTab.start-flow.test.tsx \
  src/components/Quiz/tabs/__tests__/TakeQuizTab.study-modes.test.tsx
```

Expected: profile availability, scale guide, fixed ordering, and subtype-label assertions fail.

- [ ] **Step 3: Add the minimal frontend implementation**

- Mark the fallback profile available in `services/quizzes.ts`.
- Normalize question-tag detection through the existing tag convention and identify `assertion_reasoning` without adding a new API field.
- Add an `Assertion / Reasoning` tag in the shared question header.
- Make `getOptionEntriesForQuestion` return original index order for assertion/reasoning questions while leaving existing randomization intact for ordinary MCQ and Best of Five.
- Render one unframed semantic `<section>` before the question list whenever the visible collection contains an assertion/reasoning question. Use the first tagged question's canonical options and visible `A.`-`E.` labels; do not use a nested card or raw colors.
- Reuse the guide in graded, practice, review, and result branches. Keep the standard radio input, answer storage, grading, explanation, and citation components.
- In practice mode only for this subtype, show the concise explanation and citations after either a correct or incorrect answer. Preserve the existing feedback policy for every other question subtype.
- Do not add a route, dependency, extension-only UI, or duplicated attempt state.

- [ ] **Step 4: Run focused and neighboring frontend tests**

From `apps/packages/ui`:

```bash
bun run test \
  src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx \
  src/components/Quiz/tabs/__tests__/TakeQuizTab.start-flow.test.tsx \
  src/components/Quiz/tabs/__tests__/TakeQuizTab.study-modes.test.tsx \
  src/components/Quiz/tabs/__tests__/TakeQuizTab.navigation-guardrails.test.tsx \
  src/components/Quiz/tabs/__tests__/TakeQuizTab.design-system-state.test.tsx \
  src/components/Quiz/tabs/__tests__/TakeQuizTab.submission-retry.test.tsx \
  src/components/Quiz/utils/__tests__/optionShuffle.test.ts
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add apps/packages/ui/src/services/quizzes.ts \
  apps/packages/ui/src/components/Quiz/tabs/TakeQuizTab.tsx \
  apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx \
  apps/packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.start-flow.test.tsx \
  apps/packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.study-modes.test.tsx
git commit -m "feat: present assertion reasoning scale"
```

### Task 3: Verify compatibility and close the phase

**Stage:** 3 - Integration and closeout
**Goal:** Prove the specialized profile does not regress existing quiz generation, attempts, or study modes.
**Success Criteria:** Focused backend/frontend suites, syntax/lint/security checks, independent review, and Backlog acceptance criteria are complete.
**Tests:** Combined generator, endpoint, taking, review, migration-adjacent, and static checks.
**Status:** Not Started

**Files:**
- Modify through Backlog CLI/MCP only: `backlog/tasks/task-12102.3.4 - Implement-assertion-and-reasoning-questions.md`
- Remove after completion: `Docs/superpowers/plans/2026-07-09-assertion-reasoning-questions.md`

- [ ] **Step 1: Run combined backend regression tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generate_endpoint_multi_source.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_quiz_emq_group_persistence.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_quiz_assertion_reasoning_subtype.py -q
```

- [ ] **Step 2: Run combined frontend regression tests**

Run the seven frontend files listed in Task 2 Step 4.

- [ ] **Step 3: Run lint, syntax, security, and diff checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check \
  tldw_Server_API/app/services/quiz_generator.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generate_endpoint_multi_source.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_quiz_assertion_reasoning_subtype.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m compileall -q \
  tldw_Server_API/app/services/quiz_generator.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/services/quiz_generator.py \
  -f json -o /tmp/bandit_assertion_reasoning_quiz.json
git diff --check
```

From `apps/packages/ui`:

```bash
env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false -p tsconfig.json
```

If the package-wide typecheck or design-state guard exposes the documented unrelated baseline, record exact diagnostics and verify no touched path appears.

The PRD's cross-profile generation metrics requirement is intentionally not coupled to this subtype slice. It is tracked separately as `TASK-12102.3.7`, which covers bounded profile/source/outcome metrics, validation failures, latency, and fail-open behavior across every generation profile.

- [ ] **Step 4: Request independent code review**

Review the final diff against all five Backlog acceptance criteria, focusing on answer-index ambiguity, canonical scale stability, no chain-of-thought prompting, legacy shuffle behavior, and all-mode rendering.

- [ ] **Step 5: Close Backlog and commit metadata**

Check all acceptance criteria and DoD items, record exact test counts and known skips/baselines, add the final summary, mark `TASK-12102.3.4` Done, remove this completed plan per repository guidance, and commit only intended metadata.
