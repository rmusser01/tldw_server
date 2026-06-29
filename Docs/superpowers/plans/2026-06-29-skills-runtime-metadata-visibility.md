# Skills Runtime Metadata Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose read-only Skills runtime declaration metadata in API responses and show it in the Skills manager before users run skills.

**Architecture:** Add one backend derivation helper that turns existing skill metadata into a structured runtime dictionary, then surface that through existing Pydantic response models. On the frontend, add optional types plus fallback derivation so the Skills table and test-run modal work with both new and legacy responses.

**Tech Stack:** FastAPI, Pydantic, existing SkillsService registry data, React, TypeScript, Ant Design, Vitest, pytest.

---

## File Structure

- Create `tldw_Server_API/app/core/Skills/runtime_metadata.py`
  - Pure helper for deriving runtime declaration metadata from `context`, `allowed_tools`, `model`, and `disable_model_invocation`.
- Modify `tldw_Server_API/app/api/v1/schemas/skills_schemas.py`
  - Add `SkillRuntimeMetadata`.
  - Add runtime fields to `SkillSummary` and `SkillResponse` through `SkillBase`.
- Modify `tldw_Server_API/app/api/v1/endpoints/skills.py`
  - Use the helper when building list and detail responses.
- Modify `tldw_Server_API/app/core/Skills/skills_service.py`
  - Include accurate runtime metadata in `available_skills` dictionaries used by `/skills/context`.
- Modify `tldw_Server_API/tests/Skills/integration/test_skills_api.py`
  - Add focused assertions for runtime metadata on list/detail/context payloads.
- Modify `apps/packages/ui/src/types/skill.ts`
  - Add optional runtime metadata and optional list-level raw declaration fields.
- Modify `apps/packages/ui/src/components/Option/Skills/Manager.tsx`
  - Add optional Runtime column and pass selected skill runtime to `SkillPreview`.
- Modify `apps/packages/ui/src/components/Option/Skills/SkillPreview.tsx`
  - Show selected-skill runtime impact before actions.
- Modify `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`
  - Cover optional Runtime column and legacy fallback.
- Modify `apps/packages/ui/src/components/Option/Skills/__tests__/SkillPreview.test.tsx`
  - Cover runtime disclosure text/tags.
- Update `backlog/tasks/task-530.13 - Implement-Skills-runtime-metadata-visibility.md`
  - Record touched files, verification, and final summary.

### Task 1: Backend Runtime Metadata Contract

**Files:**
- Create: `tldw_Server_API/app/core/Skills/runtime_metadata.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/skills_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/skills.py`
- Modify: `tldw_Server_API/app/core/Skills/skills_service.py`
- Test: `tldw_Server_API/tests/Skills/integration/test_skills_api.py`

- [ ] **Step 1: Write failing list/detail/context tests**

Add list/detail tests that create a fork skill with declared tools, model override, and `disable-model-invocation: true`, then assert:

```python
assert body["runtime"] == {
    "execution_mode": "fork",
    "test_run_may_call_model": True,
    "declares_tools": True,
    "declared_tool_count": 2,
    "model_override": "gpt-4o",
    "auto_invocation_enabled": False,
}
```

Also assert list summaries include `allowed_tools` and `model`.

Add a separate `/skills/context` assertion using a context-eligible skill where `disable-model-invocation` is false, because the context payload intentionally excludes skills disabled for model auto-invocation. Assert `available_skills` includes runtime metadata without changing `context_text` format.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Skills/integration/test_skills_api.py -k "runtime_metadata" -q
```

Expected: FAIL because runtime fields are missing.

- [ ] **Step 3: Add pure runtime metadata helper**

Implement:

```python
def build_skill_runtime_metadata(
    *,
    context: str | None,
    allowed_tools: list[str] | None,
    model: str | None,
    disable_model_invocation: bool | None,
) -> dict[str, object]:
    execution_mode = "fork" if context == "fork" else "inline"
    declared_tool_count = len(allowed_tools or [])
    return {
        "execution_mode": execution_mode,
        "test_run_may_call_model": execution_mode == "fork",
        "declares_tools": declared_tool_count > 0,
        "declared_tool_count": declared_tool_count,
        "model_override": model,
        "auto_invocation_enabled": not bool(disable_model_invocation),
    }
```

- [ ] **Step 4: Add schema and endpoint wiring**

Add `SkillRuntimeMetadata` to schemas, attach it to `SkillBase` and `SkillSummary`, and populate it from `_skill_data_to_response` and `_metadata_to_summary`.

- [ ] **Step 5: Add context payload wiring**

Add `allowed_tools`, `model`, and `runtime` to `SkillsService._build_context_payload()` available skill dictionaries, without changing `context_text`.

- [ ] **Step 6: Run backend tests to verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Skills/integration/test_skills_api.py -k "runtime_metadata or list_skills_filters_and_sorts_before_pagination or get_context_payload" -q
```

Expected: PASS.

### Task 2: Frontend Types and Manager Runtime Column

**Files:**
- Modify: `apps/packages/ui/src/types/skill.ts`
- Modify: `apps/packages/ui/src/components/Option/Skills/Manager.tsx`
- Test: `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`

- [ ] **Step 1: Write failing Manager tests**

Add a test that:

- Loads one fork skill with runtime metadata.
- Opens column visibility.
- Enables `Runtime`.
- Asserts `Fork`, `Test may call model`, `2 tools declared`, and `Model override` are visible.

Add a legacy compatibility assertion that a row without `runtime`, `allowed_tools`, or `model` still renders when the Runtime column is enabled.

- [ ] **Step 2: Run Manager tests to verify RED**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot
```

Expected: FAIL because the Runtime column does not exist.

- [ ] **Step 3: Add frontend runtime types**

Add:

```ts
export interface SkillRuntimeMetadata {
  execution_mode: SkillContext
  test_run_may_call_model: boolean
  declares_tools: boolean
  declared_tool_count: number
  model_override: string | null
  auto_invocation_enabled: boolean
}
```

Make `SkillSummary.allowed_tools`, `SkillSummary.model`, and `SkillSummary.runtime` optional so older responses and existing mocks remain valid.

- [ ] **Step 4: Add Manager fallback helper and Runtime column**

Implement a helper that returns backend `runtime` when present and otherwise derives from `context`, `allowed_tools`, `model`, and `disable_model_invocation`.

Add `runtime` to optional column keys and render text-bearing tags for mode, model-call possibility, declared tool count, model override, and auto-invocation off state.

- [ ] **Step 5: Run Manager tests to verify GREEN**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot
```

Expected: PASS.

### Task 3: SkillPreview Runtime Disclosure

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Skills/Manager.tsx`
- Modify: `apps/packages/ui/src/components/Option/Skills/SkillPreview.tsx`
- Test: `apps/packages/ui/src/components/Option/Skills/__tests__/SkillPreview.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`

- [ ] **Step 1: Write failing SkillPreview test**

Add a test that renders `SkillPreview` with fork runtime metadata and asserts the modal shows:

- `Fork`
- `Test may call model`
- `2 tools declared`
- `Auto invocation off`
- Copy explaining that `Render prompt only` does not invoke fork/model/tool execution.

- [ ] **Step 2: Run SkillPreview tests to verify RED**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Option/Skills/__tests__/SkillPreview.test.tsx --reporter=dot
```

Expected: FAIL because `SkillPreview` does not accept or render runtime metadata.

- [ ] **Step 3: Pass runtime from Manager to SkillPreview**

Find the selected row by `previewSkill` name and pass derived runtime metadata to `SkillPreview` when available.

- [ ] **Step 4: Render runtime impact in SkillPreview**

Add a compact disclosure block before action buttons. Keep copy behavior-specific:

- Dry render does not invoke fork/model/tool execution.
- A fork test run may call the configured model.
- Declared tools are declarations, not availability guarantees.

- [ ] **Step 5: Run frontend focused tests to verify GREEN**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx src/components/Option/Skills/__tests__/SkillPreview.test.tsx --reporter=dot
```

Expected: PASS.

### Task 4: Verification, Security, and Tracking

**Files:**
- Modify: `backlog/tasks/task-530.13 - Implement-Skills-runtime-metadata-visibility.md`

- [ ] **Step 1: Run backend focused verification**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Skills/integration/test_skills_api.py -k "runtime_metadata or list_skills_filters_and_sorts_before_pagination or get_context_payload" -q
```

Expected: PASS.

- [ ] **Step 2: Run frontend focused verification**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx src/components/Option/Skills/__tests__/SkillPreview.test.tsx --reporter=dot
```

Expected: PASS.

- [ ] **Step 3: Run diff and security checks**

Run:

```bash
git diff --check
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Skills/runtime_metadata.py tldw_Server_API/app/api/v1/schemas/skills_schemas.py tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/core/Skills/skills_service.py -f json -o /tmp/bandit_skills_runtime_metadata_TASK_530_13.json
```

Expected: no whitespace errors and no new Bandit findings in touched backend code.

- [ ] **Step 4: Update Backlog task**

Record:

- Spec path.
- Plan path.
- Touched files.
- Verification commands and results.
- Known skips or baseline failures, if any.

- [ ] **Step 5: Final self-review**

Review the diff for:

- No policy/enforcement changes.
- No persisted schema changes.
- No misleading permission language.
- No UI dependency on `runtime` being present.
- No changed `context_text` semantics.
