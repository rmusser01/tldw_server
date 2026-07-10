# EMQ Grouped Question Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate, persist, take, and review EMQ question groups that share one option bank while grading each stem independently.

**Architecture:** Keep EMQ stems as existing `multiple_choice` questions and add nullable `group_id` and `group_prompt` fields to the question contract. The generator validates complete groups before persistence, and the WebUI uses the group fields to render the shared prompt and option bank once while each stem submits its own ordinary answer.

**Tech Stack:** FastAPI/Pydantic, SQLite/PostgreSQL through `CharactersRAGDB`, pytest, React/TypeScript, Ant Design, Vitest/Testing Library.

**Spec:** `Docs/Product/Advanced_Quiz_Customization_PRD.md` (Phase 3 / PR-4)

---

### Task 1: Persist additive EMQ group metadata

**Stage:** 1 - API and persistence contract  
**Goal:** Round-trip nullable group metadata without changing legacy question behavior.  
**Success Criteria:** New and migrated databases expose `group_id` and `group_prompt`; create/list/update/API schemas preserve them; legacy questions return null values.  
**Tests:** Focused ChaChaNotes DB persistence and quiz schema tests.  
**Status:** In Progress

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/quizzes.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_quiz_emq_group_persistence.py`
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v52.py`
- Modify: `tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py`

- [ ] **Step 1: Write failing persistence and schema tests**

Create two `multiple_choice` questions with the same `group_id`, `group_prompt`, and options, then assert `get_question()` and `list_questions()` return both group fields. Update one question's group fields through `update_question()` and assert the explicit DB allowlist persists the changes. Add Pydantic contract assertions for `QuestionCreate`, `QuestionUpdate`, `QuestionPublicResponse`, and `QuizImportQuestion`, including the 128/2000-character limits.

Also add migration tests that run the SQLite V51-to-V52 method against a minimal V51 `quiz_questions` table and assert both columns plus schema version 52. Follow the existing fake-backend pattern to start the PostgreSQL initializer at version 51 and assert the V52 PostgreSQL script is routed through `_apply_postgres_migration_script`; separately assert that script adds both columns and advances `db_schema_version` to 52.

```python
question_id = db.create_question(
    quiz_id=quiz_id,
    question_type="multiple_choice",
    question_text="Which diagnosis best fits this stem?",
    options=["A", "B", "C", "D", "E"],
    correct_answer=1,
    group_id="emq-1",
    group_prompt="Theme: choose the single best diagnosis for each stem.",
)
question = db.get_question(question_id)
assert question["group_id"] == "emq-1"
assert question["group_prompt"].startswith("Theme:")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/ChaChaNotesDB/test_quiz_emq_group_persistence.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v52.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py -q
```

Expected: failures because the schema and DB methods do not accept or return the group fields.

- [ ] **Step 3: Add the minimal additive DB migration and contract fields**

In `CharactersRAGDB`:

- Bump `_CURRENT_SCHEMA_VERSION` from 51 to 52.
- Add SQLite and PostgreSQL V51-to-V52 migrations with nullable `group_id TEXT` and `group_prompt TEXT` columns.
- Register the migration in the linear migration registry and both SQLite/PostgreSQL initialization paths.
- Make the SQLite migration inspect existing columns before `ALTER TABLE`, so repaired or drifted schemas can advance safely.
- Extend `create_question`, `get_question`, `list_questions`, and `update_question` to round-trip the two fields.

In Pydantic schemas, add optional bounded strings to `QuestionCreate`, `QuestionUpdate`, `QuestionPublicResponse`, and `QuizImportQuestion`:

```python
group_id: Optional[str] = Field(None, max_length=128)
group_prompt: Optional[str] = Field(None, max_length=2000)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/api/v1/schemas/quizzes.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_quiz_emq_group_persistence.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v52.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py
git commit -m "feat: persist emq question groups"
```

### Task 2: Generate and validate complete EMQ groups

**Stage:** 2 - Profile-aware generation  
**Goal:** Make the EMQ profile available and fail closed on malformed groups.  
**Success Criteria:** Deterministic and LLM normalization paths emit grouped MCQ stems with an identical bank, per-stem citations, and no partial persistence of invalid groups.  
**Tests:** Generator profile, normalization, deterministic generation, and persistence tests.  
**Status:** Not Started

**Files:**
- Modify: `tldw_Server_API/app/services/quiz_generator.py`
- Modify: `tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py`
- Modify: `tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py`

- [ ] **Step 1: Write failing profile and group-validation tests**

Cover these behaviors:

- `emq` is available and only allows `multiple_choice`.
- Test-mode generation emits at least two stems with the same group ID, prompt, and option list.
- Every stem keeps its own citation.
- `_normalize_questions(..., generation_profile="emq")` raises `ValueError` when a stem lacks group metadata or explanation, exceeds the group-field length limits, has a missing/invalid answer, a group has fewer than two stems, prompts differ, or banks are empty/different.
- The final question limiter preserves complete groups when the LLM returns more stems than requested, then validates the limited list again.
- `generate_quiz_from_sources` persists the group fields.

- [ ] **Step 2: Run generator tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py -q
```

Expected: failures because EMQ is still planned and normalization has no group contract.

- [ ] **Step 3: Implement the minimal EMQ profile and validation**

- Mark the EMQ profile available, default/allow `multiple_choice`, and add a prompt instruction requesting one shared bank and at least two stems per group.
- Extend the prompt JSON example with optional `group_id` and `group_prompt` fields and EMQ-specific constraints.
- For EMQ, permit up to ten options, normalize `group_id`/`group_prompt` onto each question payload, and enforce the same 128/2000-character limits as the API contract before persistence.
- Add strict EMQ answer normalization that accepts only an in-range numeric index, a valid option letter, or an exact option label. Raise on missing or malformed values instead of inheriting the legacy MCQ fallback to index 0.
- Validate after normalization: every EMQ question is MCQ; every stem has a nonempty explanation; every group has at least two stems, one identical nonempty prompt, and one identical bank containing at least two options.
- Replace plain list slicing with an EMQ-aware limiter that keeps groups atomic. A complete group may make the result exceed `num_questions`; preserving group semantics takes precedence over the requested stem count.
- Validate the final limited question list immediately before provenance validation and persistence. Raise before `_persist_generated_quiz` when any invariant fails.
- Build one deterministic group in test mode and pass group fields into `db.create_question`.

The validation boundary should remain one small helper reused after normalization and after final limiting; do not add a new service or question type.

- [ ] **Step 4: Run generator tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add tldw_Server_API/app/services/quiz_generator.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py
git commit -m "feat: generate validated emq groups"
```

### Task 3: Present shared banks in the WebUI

**Stage:** 3 - Taking and review UI  
**Goal:** Show each EMQ option bank once while preserving independent stem answers.  
**Success Criteria:** The profile is selectable; practice, graded, review, and result views identify the group and show one bank; each stem uses a separate selection control and submits its own numeric option index.  
**Tests:** Generate-tab request and TakeQuizTab grouped rendering/interaction tests.  
**Status:** Not Started

**Files:**
- Modify: `apps/packages/ui/src/services/quizzes.ts`
- Modify: `apps/packages/ui/src/components/Quiz/utils/optionShuffle.ts`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/TakeQuizTab.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/utils/__tests__/optionShuffle.test.ts`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.study-modes.test.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.start-flow.test.tsx`

- [ ] **Step 1: Write failing UI tests**

Add an EMQ profile selection test that expects `generation_profile: "emq"` and `question_types: ["multiple_choice"]`. Add a TakeQuizTab fixture with two grouped stems and assert:

- the group prompt and shared bank render once;
- both stem texts render;
- each stem has its own select control;
- selecting values produces two independent answers;
- the review/result path retains the group presentation.

Add a deterministic-pool test proving that selecting any EMQ stem includes its whole group. The atomic group may make the resulting pool larger than the requested count. Add a TakeQuizTab study-mode regression showing practice/review pooling never presents a partial EMQ group.

- [ ] **Step 2: Run focused Vitest files and verify RED**

From `apps/packages/ui`:

```bash
bun run test \
  src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx \
  src/components/Quiz/tabs/__tests__/TakeQuizTab.start-flow.test.tsx \
  src/components/Quiz/tabs/__tests__/TakeQuizTab.study-modes.test.tsx \
  src/components/Quiz/utils/__tests__/optionShuffle.test.ts
```

Expected: EMQ is absent from the client registry and grouped fields/presentation are unsupported.

- [ ] **Step 3: Add client fields and grouped presentation**

- Add `group_id?: string | null` and `group_prompt?: string | null` to question types and create/update/import contracts.
- Add the available EMQ profile to `QUIZ_GENERATION_PROFILES`.
- Add a grouped deterministic pool helper beside the existing pool helper. Treat each EMQ group as one draw unit, preserve member order, and keep the existing ungrouped helper behavior unchanged.
- Use the grouped helper for practice/review question pools, passing each question's normalized group ID. Groups are atomic even when the displayed count exceeds the configured pool size.
- In `TakeQuizTab`, identify an EMQ question by a non-empty group ID.
- Before only the first visible stem in a group, render the group prompt and an alphabetized option bank from that question's options.
- Use the existing Ant Design `Select` for each EMQ stem, storing the original numeric option index through `updateAnswer`.
- Reuse this presentation in graded, practice, review, and result maps; keep legacy MCQ radio rendering unchanged.

No new route, state store, dependency, or nested card is needed.

- [ ] **Step 4: Run focused Vitest files and verify GREEN**

Run the command from Step 2. Expected: all four selected files pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add apps/packages/ui/src/services/quizzes.ts \
  apps/packages/ui/src/components/Quiz/utils/optionShuffle.ts \
  apps/packages/ui/src/components/Quiz/tabs/TakeQuizTab.tsx \
  apps/packages/ui/src/components/Quiz/utils/__tests__/optionShuffle.test.ts \
  apps/packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.study-modes.test.tsx \
  apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx \
  apps/packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.start-flow.test.tsx
git commit -m "feat: present grouped emq questions"
```

### Task 4: Verify compatibility and close the slice

**Stage:** 4 - Integration and closeout  
**Goal:** Prove EMQ support does not regress existing quiz paths and record the result.  
**Success Criteria:** Focused backend/frontend suites, lint/type/syntax checks, and Bandit pass; Backlog acceptance criteria and DoD are complete.  
**Tests:** Combined quiz suites plus migration initialization coverage.  
**Status:** Not Started

**Files:**
- Modify: `backlog/tasks/task-12102.3.3 - Implement-EMQ-grouped-question-support.md` through Backlog CLI/MCP only
- Modify: this plan's stage statuses during execution; remove this plan after all stages are complete per repository guidance

- [ ] **Step 1: Run the combined backend regression suite**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/ChaChaNotesDB/test_quiz_emq_group_persistence.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v52.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generate_endpoint_multi_source.py -q
```

- [ ] **Step 2: Run the combined frontend regression suite**

From `apps/packages/ui`:

```bash
bun run test \
  src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx \
  src/components/Quiz/tabs/__tests__/TakeQuizTab.start-flow.test.tsx \
  src/components/Quiz/tabs/__tests__/TakeQuizTab.study-modes.test.tsx \
  src/components/Quiz/utils/__tests__/optionShuffle.test.ts
```

- [ ] **Step 3: Run security and diff checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/api/v1/schemas/quizzes.py \
  tldw_Server_API/app/services/quiz_generator.py \
  -f json -o /tmp/bandit_emq_quiz.json
git diff --check
```

Expected: zero new Bandit findings and no whitespace errors.

Run scoped Python lint and syntax checks:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/api/v1/schemas/quizzes.py \
  tldw_Server_API/app/services/quiz_generator.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_quiz_emq_group_persistence.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v52.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generate_schema_contract.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_prompt_template.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m compileall -q \
  tldw_Server_API/app/api/v1/schemas/quizzes.py \
  tldw_Server_API/app/services/quiz_generator.py
```

Run the UI package type-check from `apps/packages/ui`:

```bash
bunx tsc --noEmit --pretty false -p tsconfig.json
```

If a broad check exposes an unrelated baseline failure, record the exact output and separately verify all focused tests and touched-file checks.

- [ ] **Step 4: Review requirements and update Backlog**

Check all five acceptance criteria, record exact verification counts and any known skips, complete the DoD, and mark `TASK-12102.3.3` Done.

- [ ] **Step 5: Commit closeout metadata**

Stage only intended feature and Backlog files. Leave pre-existing `apps/bun.lock` and `apps/packages/ui/node_modules/antd` changes untouched.
