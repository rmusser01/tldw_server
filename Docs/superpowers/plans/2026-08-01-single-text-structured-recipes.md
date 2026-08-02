# Single-Text Structured Prompt Recipes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add target-specific, reusable single-text prompt recipes with ordered blocks, roles, labels, variables, starter defaults, XML-style/Markdown/free-form rendering, exact live preview, immutable built-ins, user-owned save/update, and editable application to the current system or user draft.

**Architecture:** Extend structured prompts as a strict discriminated union: schema v1 remains the existing multi-message definition, while schema v2 is explicitly `definition_kind="single_text_recipe"` and renders one destination string. Python and TypeScript implement the same deterministic renderer against shared JSON fixtures. The editor reuses existing block/variable primitives and writes through the current prompt library, sync, Prompt Studio, import/export, and MCP paths; recipe runtime values stay in ephemeral editor state and are rejected from persistence payloads.

**Tech Stack:** Pydantic v2, existing structured-prompt assembler/validator and prompt databases, FastAPI prompt/Prompt Studio APIs, MCP prompt catalog, pytest/Hypothesis, React 18, TypeScript, existing StructuredPromptEditor primitives, Dexie, TanStack Query, Ant Design, Vitest/Testing Library, Playwright, WXT, Next.js.

**Approved design:** `Docs/superpowers/specs/2026-07-22-chat-prompt-improvement-recipes-design.md`

**Backlog task:** `TASK-12984.2`

**Dependency:** Complete and verify `TASK-12984.1` first. Reuse its `PromptAssistMenu`, target adapters, scoped system-override behavior, user-draft adapter, capability service, and exact Undo implementation.

## Global Constraints

- Schema v1 behavior and stored records remain byte/behavior compatible. A v2 definition must never fall through the v1 multi-message assembler.
- Schema v2 identity is exact: `schema_version=2`, `format="structured"`, `definition_kind="single_text_recipe"`, and `assembly_config.assembly_mode="single_text"`.
- Every v2 block role equals `assembly_config.target_role`; the UI does not expose mixed-role composition.
- A recipe compiles exactly one string. It does not create a message array, multi-message conversation, system+user pair, or chat history.
- Saved recipe definitions contain variable declarations and explicitly authored starter defaults only. Runtime value maps are ephemeral and must be omitted by clients and rejected by all server save/update/import paths.
- Applying a recipe never saves it. Saving never applies it. Opening or editing never changes the target draft.
- Applying to system uses Track A’s current scoped override adapter and preserves selected-template identity. Applying to user changes only the unsent draft. Both use Track A’s exact one-step Undo.
- Built-ins (`Clear task`, `Research and analysis`, `Agent workflow`, `Blank`) are source-controlled, immutable, and cloned into a working copy. Saving a built-in always creates a new user-owned record.
- XML output is described as XML-style sections, not guaranteed document-valid XML. Enforce conservative section keys and exact closing-tag collision checks.
- TypeScript and Python renderers must pass the same fixtures. Do not enable server persistence capability until parity, mixed-version, sync, Prompt Studio, MCP, WebUI, and extension tests pass.
- Old/offline servers may build, preview, and apply built-in/unsaved recipes locally. Saving or syncing v2 remains disabled until `single_text_recipe_v2.supported=true` is verified.
- Update `TASK-12984.2` after each task with touched paths and verification. Run Bandit over all touched Python paths before completion.

## Schema and Renderer Contract

Use these exact discriminants and meanings in Python and TypeScript.

```json
{
  "schema_version": 2,
  "format": "structured",
  "definition_kind": "single_text_recipe",
  "assembly_config": {
    "assembly_mode": "single_text",
    "target_role": "system",
    "render_format": "xml",
    "block_separator": "\n\n"
  },
  "variables": [
    {
      "name": "audience",
      "label": "Audience",
      "required": true,
      "default_value": "",
      "input_type": "text"
    }
  ],
  "blocks": [
    {
      "id": "objective",
      "name": "Objective",
      "section_key": "objective",
      "role": "system",
      "kind": "objective",
      "content": "Explain the task for {{audience}}.",
      "enabled": true,
      "order": 10,
      "is_template": true
    }
  ]
}
```

```python
StructuredPromptDefinition = Annotated[
    MultiMessagePromptDefinitionV1 | SingleTextRecipeDefinitionV2,
    Field(discriminator="schema_version"),
]


class SingleTextRecipeAssemblyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assembly_mode: Literal["single_text"] = "single_text"
    target_role: Literal["system", "user"]
    render_format: Literal["xml", "markdown", "freeform"]
    block_separator: str = Field(default="\n\n", max_length=20)
```

The existing user-draft target is called `user_message` in Track A’s UI/API, but persisted block roles continue to use the existing structured-prompt role value `user`. Conversion occurs only at the target adapter boundary.

## Stage 1: Schema-v2 Core and Cross-Language Fixtures

**Goal:** Define, validate, and deterministically render a single-text recipe while preserving every v1 behavior.

**Success Criteria:** Both languages accept/reject identical definitions and render identical strings/legacy snapshots from shared fixtures.

**Tests:** Python unit/property tests, TypeScript unit tests, shared success/error fixtures, all existing v1 tests.

**Status:** Not Started

### Task 1: Add discriminated Python models and validation

**Files:**

- Modify: `tldw_Server_API/app/core/Prompt_Management/structured_prompts/models.py`
- Modify: `tldw_Server_API/app/core/Prompt_Management/structured_prompts/validator.py`
- Modify: `tldw_Server_API/app/core/Prompt_Management/structured_prompts/__init__.py`
- Create: `tldw_Server_API/tests/Prompt_Management/test_single_text_recipe_validator.py`
- Modify: `tldw_Server_API/tests/Prompt_Management/test_structured_prompt_validator.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/property/test_prompt_properties.py`

- [ ] Add failing tests proving v1 remains valid, v2 requires every discriminant, unknown versions fail, extra fields fail, mixed target roles fail, duplicate IDs/names fail as before, enabled XML blocks require valid unique `section_key`, and configured limits reject oversized definitions.

```python
def test_v2_cannot_fall_through_v1_validation():
    payload = valid_recipe(schema_version=2)
    payload.pop("definition_kind")
    issues = validate_prompt_definition(payload)
    assert issues[0].code == "invalid_definition_kind"
```

- [ ] Add property tests for arbitrary block order, duplicate identifiers, Unicode labels/content, separators, variable names, and future schema numbers.
- [ ] Run tests and confirm RED.

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Prompt_Management/test_single_text_recipe_validator.py \
  tldw_Server_API/tests/Prompt_Management/test_structured_prompt_validator.py \
  tldw_Server_API/tests/Prompt_Management_NEW/property/test_prompt_properties.py -q
```

- [ ] Rename the current `PromptDefinition` implementation to `MultiMessagePromptDefinitionV1` while retaining a compatibility export named `PromptDefinition` only if existing imports require it. Add `SingleTextRecipeBlock`, `SingleTextRecipeAssemblyConfig`, and `SingleTextRecipeDefinitionV2`, then expose one discriminated parser function.
- [ ] Centralize v2 limits in one module-level value used by validator and capability response: block count, variable count, label/key/content length, separator length, and maximum rendered output. Do not scatter literal limits across endpoint/UI code.
- [ ] Extend validation with stable issue codes and paths. Validate conservative XML keys with `^[A-Za-z_][A-Za-z0-9_.-]*$`; do not silently rewrite stored keys.
- [ ] Ensure validation never mutates the payload and never coerces v2 into v1.
- [ ] Run all structured validator/conversion tests to GREEN and commit.

```bash
git add tldw_Server_API/app/core/Prompt_Management/structured_prompts tldw_Server_API/tests/Prompt_Management/test_single_text_recipe_validator.py tldw_Server_API/tests/Prompt_Management/test_structured_prompt_validator.py tldw_Server_API/tests/Prompt_Management_NEW/property/test_prompt_properties.py backlog/tasks/task-12984.2\ -\ Implement-single-text-structured-prompt-recipes.md
git commit -m "feat(prompts): add single-text recipe schema v2 (TASK-12984.2)"
```

### Task 2: Implement deterministic Python and TypeScript renderers against shared fixtures

**Files:**

- Create: `Docs/fixtures/single-text-recipes/render-cases.json`
- Create: `Docs/fixtures/single-text-recipes/error-cases.json`
- Create: `tldw_Server_API/app/core/Prompt_Management/structured_prompts/single_text_renderer.py`
- Modify: `tldw_Server_API/app/core/Prompt_Management/structured_prompts/assembler.py`
- Modify: `tldw_Server_API/app/core/Prompt_Management/structured_prompts/legacy_renderer.py`
- Create: `tldw_Server_API/tests/Prompt_Management/test_single_text_recipe_renderer.py`
- Modify: `apps/packages/ui/src/components/Option/Prompt/structured-prompt-utils.ts`
- Modify: `apps/packages/ui/src/components/Option/Prompt/__tests__/structured-prompt-utils.test.ts`

- [ ] Author shared fixtures first for ordering, disabled blocks, XML tags, closing-tag collision, Markdown headings, free-form separators, required/optional variables, starter defaults, runtime overrides, unknown variables, Unicode, empty content, exact whitespace, duplicate order ties, size limits, and target-role legacy snapshots.
- [ ] Add Python and TypeScript tests that read the same fixture files and compare exact rendered output/error code.

```ts
for (const fixture of renderCases) {
  expect(renderSingleTextRecipe(fixture.definition, fixture.runtimeValues)).toEqual({
    text: fixture.expected_text,
    legacy: fixture.expected_legacy,
  })
}
```

- [ ] Run both suites and confirm RED.

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Prompt_Management/test_single_text_recipe_renderer.py -q
cd apps && bunx vitest run packages/ui/src/components/Option/Prompt/__tests__/structured-prompt-utils.test.ts
```

- [ ] Implement both renderers with the same algorithm: stable sort by `(order, original_index)`, discard disabled blocks, resolve only declared `{{name}}` variables, use runtime value then authored `default_value` then optional empty string, reject missing required values, render block by format, and join with the exact separator.

```python
def render_single_text_recipe(
    definition: SingleTextRecipeDefinitionV2,
    runtime_values: Mapping[str, Any] | None = None,
) -> SingleTextRecipeRenderResult: ...
```

- [ ] XML: emit `<section_key>content</section_key>` and reject the exact matching closing tag after substitution. Markdown: emit `## {block.name}\n\n{content}`. Free-form: emit content only. Preserve authored inner whitespace.
- [ ] Return a legacy snapshot with the compiled string in `system_prompt` when target role is `system`, or `user_prompt` when target role is `user`; the other field is empty.
- [ ] Make `assemble_prompt_definition` explicitly dispatch on the parsed union. V1 returns its current message result. V2 returns a tagged `SingleTextRecipeRenderResult` with `rendered_text` and legacy snapshot; it does not fabricate a multi-message assembly.
- [ ] Run both fixture suites and every existing v1 assembler/conversion test, then commit fixtures and renderers together.

```bash
git add Docs/fixtures/single-text-recipes tldw_Server_API/app/core/Prompt_Management/structured_prompts tldw_Server_API/tests/Prompt_Management/test_single_text_recipe_renderer.py apps/packages/ui/src/components/Option/Prompt/structured-prompt-utils.ts apps/packages/ui/src/components/Option/Prompt/__tests__/structured-prompt-utils.test.ts backlog/tasks/task-12984.2\ -\ Implement-single-text-structured-prompt-recipes.md
git commit -m "feat(prompts): render single-text recipes consistently (TASK-12984.2)"
```

## Stage 2: Persistence, Preview, and Compatibility

**Goal:** Store and exchange v2 recipes through existing systems without changing v1 records or leaking runtime values.

**Success Criteria:** Prompt API, Prompt Studio, local DB, sync, import/export, search, and MCP recognize v2 identity and legacy snapshots; old clients fail safely.

**Tests:** API CRUD/preview, DB and migration tests, sync conflict tests, interop round trips, Prompt Studio, MCP catalog.

**Status:** Not Started

### Task 3: Teach prompt APIs and persistence to validate v2 and reject runtime values

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/prompts.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/prompt_schemas.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Prompts_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/prompts_db_helpers.py` only if serialization/migration requires it.
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompts_structured_api.py`
- Modify: `tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_structured_prompt_search.py`

- [ ] Add failing tests for create/get/update/preview/delete/search of system and user recipes, v1/v2 coexistence, unknown-version rejection without record mutation, mismatched schema version, role mismatch, rendered legacy fields, and direct/embedded runtime-value fields rejected with `invalid_recipe_runtime_values`.

```python
payload = recipe_create_payload()
payload["prompt_definition"]["runtime_values"] = {"audience": "private"}
response = client.post("/api/v1/prompts/", json=payload, headers=auth_headers)
assert response.status_code == 400
assert "runtime" in response.json()["detail"].lower()
```

- [ ] Run the focused tests and confirm RED.
- [ ] Replace `_coerce_structured_definition`’s direct `PromptDefinition.model_validate` with the discriminated parser. Update `_render_definition_legacy_fields`, preview, and storage preparation to dispatch v1/v2 explicitly.
- [ ] Extend `StructuredPromptPreviewResponse` and its TypeScript peer with optional `rendered_text`. For v2 preview, return `rendered_text` plus legacy snapshots and an empty `assembled_messages`; preserve the exact existing v1 response. Apply the same response contract to Prompt Studio preview.
- [ ] Add a recursive persistence guard that rejects keys reserved for ephemeral values (`runtime_values`, `variable_values`, `resolved_values`) at the definition root and other specified runtime-value locations. Do not reject those words when they occur as authored content strings.
- [ ] Persist v2 through the existing JSON definition/schema columns; do not add a database column when identity is already inside the schema. Confirm existing migrations and indexes need no destructive change.
- [ ] Search continues to index/use title and compiled legacy content snapshot. Add a Recipe kind filter only at the application layer; do not attempt schema-aware SQLite indexing in v2.
- [ ] Change `/prompts/capabilities` to advertise v2 limits but keep `single_text_recipe_v2.supported=false` until the final parity gate.
- [ ] Run prompt DB/API/search tests and commit.

```bash
git add tldw_Server_API/app/api/v1/endpoints/prompts.py tldw_Server_API/app/api/v1/schemas/prompt_schemas.py tldw_Server_API/app/core/DB_Management/Prompts_DB.py tldw_Server_API/app/core/DB_Management/prompts_db_helpers.py tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompts_structured_api.py tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py tldw_Server_API/tests/Prompt_Management_NEW/integration/test_structured_prompt_search.py backlog/tasks/task-12984.2\ -\ Implement-single-text-structured-prompt-recipes.md
git commit -m "feat(prompts): persist structured recipes safely (TASK-12984.2)"
```

### Task 4: Preserve recipe identity across Prompt Studio, sync, import/export, and MCP

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_prompts.py`
- Modify: `tldw_Server_API/app/core/Prompt_Management/prompt_studio/prompt_executor.py`
- Modify: `tldw_Server_API/app/core/Prompt_Management/Prompts_Interop.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_catalog.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py`
- Modify: relevant Prompt Studio and prompt interop tests.
- Modify: `apps/packages/ui/src/services/prompt-studio.ts`
- Modify: `apps/packages/ui/src/services/prompt-sync.ts`
- Modify: `apps/packages/ui/src/services/__tests__/prompt-sync.structured-prompts.test.ts`
- Modify: `apps/packages/ui/src/services/__tests__/prompt-sync.auto-sync.test.ts`

- [ ] Add failing compatibility tests for v2 Prompt Studio create/preview/update/version retrieval, local-to-server and server-to-local sync, conflict hashes, keep-both recovery, CSV/Markdown/JSON or supported export/import round trips, and MCP catalog output.
- [ ] Prove runtime values never enter create/update/sync/export payloads and a malicious server payload containing them is rejected/quarantined without overwriting a good local record.
- [ ] Run the focused suites and confirm RED.

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Prompt_Management/test_prompts_interop.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py -q
cd apps && bunx vitest run \
  packages/ui/src/services/__tests__/prompt-sync.structured-prompts.test.ts \
  packages/ui/src/services/__tests__/prompt-sync.auto-sync.test.ts
```

- [ ] Reuse the same discriminated parser and renderer in Prompt Studio. Preview returns `rendered_text`. An execution path that requires multi-message v1 must reject v2 with a stable “single-text recipe must be rendered/applied first” error, not silently reinterpret it.
- [ ] Extend sync types from `Record<string, any>` at validation boundaries to the v1/v2 union. Increment `CURRENT_PROMPT_SYNC_PAYLOAD_VERSION` only if on-wire semantics change, and add backward tests before doing so.
- [ ] Include definition kind/schema version and compiled legacy snapshots in conflict hashing so v1 and v2 records with similar text are not treated as identical.
- [ ] Make MCP catalog formatting preserve recipe identity and expose recipe variables as prompt arguments. At MCP render time, map the compiled single field to exactly one protocol prompt message using the recipe target role; keep existing v1 multi-message and assistant-message behavior unchanged.
- [ ] Run all compatibility tests to GREEN and commit.

```bash
git add tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_prompts.py tldw_Server_API/app/core/Prompt_Management/prompt_studio/prompt_executor.py tldw_Server_API/app/core/Prompt_Management/Prompts_Interop.py tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_catalog.py tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py apps/packages/ui/src/services/prompt-studio.ts apps/packages/ui/src/services/prompt-sync.ts apps/packages/ui/src/services/__tests__/prompt-sync.structured-prompts.test.ts apps/packages/ui/src/services/__tests__/prompt-sync.auto-sync.test.ts backlog/tasks/task-12984.2\ -\ Implement-single-text-structured-prompt-recipes.md
git commit -m "feat(prompts): interoperate recipe schema v2 (TASK-12984.2)"
```

## Stage 3: Shared Recipe Editor and Built-ins

**Goal:** Let users visually build and preview a valid target-specific recipe without mutating the current draft.

**Success Criteria:** Editor operations, variables, render modes, preview, built-in cloning, local offline application, save gating, and accessibility behave predictably.

**Tests:** Pure editor-state tests, renderer tests, component interaction/a11y tests.

**Status:** Not Started

### Task 5: Build typed editor state and immutable starter recipes

**Files:**

- Create: `apps/packages/ui/src/components/Common/PromptAssist/recipes/types.ts`
- Create: `apps/packages/ui/src/components/Common/PromptAssist/recipes/built-in-recipes.ts`
- Create: `apps/packages/ui/src/components/Common/PromptAssist/recipes/recipe-editor-state.ts`
- Create: `apps/packages/ui/src/components/Common/PromptAssist/recipes/__tests__/built-in-recipes.test.ts`
- Create: `apps/packages/ui/src/components/Common/PromptAssist/recipes/__tests__/recipe-editor-state.test.ts`

- [ ] Add failing tests for all four starters in both allowed targets, deep-clone isolation, stable IDs/order, exact starter content, add/remove/rename/reorder/toggle block, section-key proposal only on new block, format switch, variable declaration, runtime value separation, required-value gating, source dirty state, Save as new, and Update availability only for explicitly opened saved recipes.

```ts
const working = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "user_message")
working.definition.blocks[0].content = "changed"
expect(CLEAR_TASK_RECIPE.definition.blocks[0].content).not.toBe("changed")
```

- [ ] Run tests and confirm RED.
- [ ] Define precise union types matching Python and narrow unknown stored JSON through a validator before editor use. Avoid `Record<string, any>` inside recipe code.
- [ ] Author starters as frozen source data:

  - Clear task: objective, context/inputs, constraints, output.
  - Research and analysis: question, evidence/source rules, analysis method, uncertainty, deliverable.
  - Agent workflow: objective, allowed actions/tools, plan/execute loop, stop/confirmation conditions, final report.
  - Blank: no blocks or variables.

- [ ] Keep role/target conversion explicit: `system -> system`; `user_message -> user`. Clone and rewrite all built-in block roles to the chosen target without altering content.
- [ ] Keep `runtimeValues` in editor state beside, never inside, `definition`. Serialization helpers accept only the definition and strip no hidden data because invalid state should be rejected before save.
- [ ] Run tests to GREEN and commit.

```bash
git add apps/packages/ui/src/components/Common/PromptAssist/recipes backlog/tasks/task-12984.2\ -\ Implement-single-text-structured-prompt-recipes.md
git commit -m "feat(ui): add recipe state and starter content (TASK-12984.2)"
```

### Task 6: Build the single-field visual editor by reusing structured primitives

**Files:**

- Create: `apps/packages/ui/src/components/Common/PromptAssist/recipes/SingleFieldRecipeEditor.tsx`
- Create: `apps/packages/ui/src/components/Common/PromptAssist/recipes/RecipeBlockEditor.tsx` only if the existing block panel cannot accept single-role/section-key props cleanly.
- Modify: `apps/packages/ui/src/components/Option/Prompt/Structured/BlockListPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Prompt/Structured/BlockEditorPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Prompt/Structured/VariableEditorPanel.tsx`
- Create: `apps/packages/ui/src/components/Common/PromptAssist/recipes/__tests__/SingleFieldRecipeEditor.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Prompt/Structured/__tests__/StructuredPromptEditor.test.tsx`

- [ ] Add failing tests for starter picker, saved-recipe clone, ordered drag/keyboard movement, block add/remove/rename/toggle, validated section key, target-locked role, format selector, variable declarations/defaults/runtime inputs, exact preview, inline errors, Apply disabled until valid, Apply without Save, Save as new, saved-only Update, offline save disabled, and focus/keyboard/mobile behavior.
- [ ] Run tests and confirm RED.

```bash
cd apps && bunx vitest run \
  packages/ui/src/components/Common/PromptAssist/recipes/__tests__/SingleFieldRecipeEditor.test.tsx \
  packages/ui/src/components/Option/Prompt/Structured/__tests__/StructuredPromptEditor.test.tsx
```

- [ ] Generalize existing block/variable panels through small props (`allowedRoles`, `showRole`, `sectionKey`, `onSectionKeyChange`, `runtimeValues`) rather than copying their full implementation. Keep existing v1 defaults and tests unchanged.
- [ ] Render local preview through the fixture-tested TypeScript renderer on every valid change. Show the exact destination string in a plain-text textarea/code surface; do not render Markdown/XML as HTML.
- [ ] Separate starter default inputs from runtime inputs with explicit labels. Changing runtime values must not change `default_value`; switching recipes clears runtime values unless compatible values are deliberately retained by the state reducer’s tested rule.
- [ ] Applying calls the Track A target adapter with compiled text, captures its exact undo snapshot, closes or returns to the draft view, and never invokes save/update.
- [ ] Save calls receive only the validated definition. Built-in/source definition objects remain frozen and untouched.
- [ ] Run focused tests, existing structured editor tests, lint, and typechecks; commit.

```bash
git add apps/packages/ui/src/components/Common/PromptAssist/recipes apps/packages/ui/src/components/Option/Prompt/Structured backlog/tasks/task-12984.2\ -\ Implement-single-text-structured-prompt-recipes.md
git commit -m "feat(ui): add single-field recipe builder (TASK-12984.2)"
```

## Stage 4: Prompt Library and Chat Integrations

**Goal:** Make recipes discoverable, cloneable, saveable, and applicable from both chat targets without confusing them with normal prompts.

**Success Criteria:** Recipe identity survives local/server workflows; library groups/badges/filtering are correct; both chat surfaces share the same editor.

**Tests:** Prompt workspace tests, Dexie tests, service tests, PromptSelect/composer tests.

**Status:** Not Started

### Task 7: Add recipe identity to local prompt library, save, sync, and search UI

**Files:**

- Modify: `apps/packages/ui/src/db/dexie/types.ts`
- Modify: `apps/packages/ui/src/db/dexie/schema.ts` only if an indexed field/migration is truly needed.
- Modify: `apps/packages/ui/src/db/dexie/helpers.ts`
- Modify: `apps/packages/ui/src/components/Option/Prompt/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/Prompt/PromptDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/Prompt/PromptFullPageEditor.tsx`
- Modify: `apps/packages/ui/src/components/Option/Prompt/PromptStarterCards.tsx`
- Modify: `apps/packages/ui/src/components/Option/Prompt/hooks/usePromptEditor.tsx`
- Modify: relevant tests under `apps/packages/ui/src/components/Option/Prompt/__tests__/`.

- [ ] Add failing tests for Recipe badge/group, target filtering, title/content search, click-to-open-builder rather than insert, clone-on-open, source record unchanged, Save as new local record, Update existing saved recipe, sync pending language, conflict/recovery, old/invalid v2 record quarantine, and ordinary v1/system/quick behavior unchanged.
- [ ] Use schema discriminants to derive recipe identity; add a denormalized `promptKind` field only if existing query/index performance requires it and a migration test justifies it. Prefer no Dexie schema bump for a first release if filtering the loaded prompt set is sufficient.
- [ ] On local save, set `promptFormat="structured"`, `promptSchemaVersion=2`, the validated definition, compiled `content`, `system_prompt` or `user_prompt` legacy snapshot, and existing sync metadata. Never store runtime values.
- [ ] Reuse current auto-sync, pending, conflict, and keep-both flows. Allow local-plus-pending recovery only after a server has positively advertised v2 and a transient save/sync failure occurs. When capability is unsupported or unknown/offline, disable Save/Update entirely while retaining unsaved local editing, preview, and Apply.
- [ ] Update prompt workspace editors to route v2 to `SingleFieldRecipeEditor` and v1 to `StructuredPromptEditor`. Never convert one implicitly just because it is opened.
- [ ] Run prompt workspace/service suites and commit.

```bash
cd apps && bunx vitest run \
  packages/ui/src/components/Option/Prompt/__tests__ \
  packages/ui/src/services/__tests__/prompt-sync.structured-prompts.test.ts
```

```bash
git add apps/packages/ui/src/db/dexie apps/packages/ui/src/components/Option/Prompt apps/packages/ui/src/services/prompt-sync.ts apps/packages/ui/src/services/__tests__/prompt-sync.structured-prompts.test.ts backlog/tasks/task-12984.2\ -\ Implement-single-text-structured-prompt-recipes.md
git commit -m "feat(prompts): add recipes to the prompt library (TASK-12984.2)"
```

### Task 8: Add Build from recipe to system and user draft flows

**Files:**

- Modify: `apps/packages/ui/src/components/Common/PromptAssist/PromptAssistMenu.tsx`
- Create: `apps/packages/ui/src/components/Common/PromptAssist/PromptAssistPanel.tsx` if Track A did not already provide the modal/panel mode switch.
- Modify: `apps/packages/ui/src/components/Common/PromptSelect.tsx`
- Modify: `apps/packages/ui/src/components/Chat/composer/PromptAssistComposerAction.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
- Modify: Track A component/integration tests.
- Modify: shared English locale files and synchronized locale outputs.

- [ ] Add failing integration tests proving Build from recipe is available without a model and with an empty draft; opening preserves the current draft; target filters recipes; preview Apply uses the correct target; selected system template identity survives; Undo is exact; runtime values vanish after close/reopen; unsupported online server disables Save but permits local built-in apply; and offline behavior is clearly labelled.
- [ ] Add the third menu action using the same menu/panel on system and user adapters. Do not add route-specific editor copies.
- [ ] For system editor, switch the modal body to recipe mode and back without changing `editorDraft` until Apply. For the composer, open the shared dialog/sheet while leaving `form.values.message` untouched until Apply.
- [ ] Fetch saved recipes through existing local query/sync state, filter by target role, and deep-clone before edit. Re-fetch/invalidate current prompt query keys after save/update.
- [ ] Save v2 only when server capability is supported and user persistence authorization permits it; present existing local/pending recovery language for transient failures. Applying does not depend on server capability because rendering is local.
- [ ] Run all Track A and Track B shared UI tests, locale sync, lint, and typechecks; commit.

```bash
cd apps
bun run --cwd extension locales:sync
bunx vitest run packages/ui/src/components/Common/PromptAssist packages/ui/src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx packages/ui/src/components/Chat/composer
bun run --cwd extension compile
bun run --cwd tldw-frontend compile
```

```bash
git add apps/packages/ui/src/components/Common/PromptAssist apps/packages/ui/src/components/Common/PromptSelect.tsx apps/packages/ui/src/components/Chat/composer/PromptAssistComposerAction.tsx apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx apps/packages/ui/src/components/Sidepanel/Chat/form.tsx apps/packages/ui/src/assets/locales apps/extension/public/locales apps/tldw-frontend/public/locales backlog/tasks/task-12984.2\ -\ Implement-single-text-structured-prompt-recipes.md
git commit -m "feat(chat): apply recipes to current drafts (TASK-12984.2)"
```

## Stage 5: Parity and Release Gate

**Goal:** Enable recipe persistence only after cross-consumer, mixed-version, accessibility, and both-surface behavior are proven.

**Success Criteria:** All v1 regressions and v2 fixtures pass; WebUI/extension can build/apply/save; old/offline behavior is safe; capability flips to supported.

**Tests:** Full scoped backend/frontend, import/export/sync/MCP, Playwright, builds, lint, Bandit.

**Status:** Not Started

### Task 9: Add WebUI and extension recipe journeys

**Files:**

- Create: `apps/tldw-frontend/e2e/workflows/single-text-recipes.spec.ts`
- Create: `apps/extension/tests/e2e/single-text-recipes.spec.ts`
- Modify: E2E fixtures/mocks under each app as required.

- [ ] WebUI tests: build each starter, edit/reorder/toggle blocks, fill variables, switch formats, verify exact preview, Apply system/user + Undo, Save as new, reopen/clone/update, search/group/badge, runtime-value non-persistence, keyboard/mobile/axe.
- [ ] Extension tests: mirror build/apply/save/reopen/offline/mixed-version flows in sidepanel and pop-out. Assert saved API payload contains the definition/defaults and no runtime-value map.
- [ ] Add old-server tests: no v2 capability hides/disables persistence with explanatory copy while built-in local preview/apply still works; unknown future v3 record is not opened/mutated.
- [ ] Run both E2E specs and current prompt/chat journeys.

```bash
(cd apps/tldw-frontend && bunx playwright test e2e/workflows/single-text-recipes.spec.ts e2e/workflows/journeys/prompts-chat.spec.ts --reporter=line)
(cd apps/extension && bunx playwright test tests/e2e/single-text-recipes.spec.ts tests/e2e/prompts-ux.spec.ts --reporter=line)
```

- [ ] Commit parity coverage.

```bash
git add apps/tldw-frontend/e2e/workflows/single-text-recipes.spec.ts apps/extension/tests/e2e/single-text-recipes.spec.ts backlog/tasks/task-12984.2\ -\ Implement-single-text-structured-prompt-recipes.md
git commit -m "test(prompts): cover recipe parity and compatibility (TASK-12984.2)"
```

### Task 10: Enable capability and finalize Track B

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/prompts.py`
- Modify: capability tests.
- Modify: `backlog/tasks/task-12984.2 - Implement-single-text-structured-prompt-recipes.md` through Backlog MCP/CLI only.
- Delete: `Docs/superpowers/plans/2026-08-01-single-text-structured-recipes.md` only after all tasks complete if following repository plan-cleanup policy.

- [ ] Run the complete v1/v2 structured backend suite, Prompt API/Studio/interop/MCP tests, and properties.

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Prompt_Management/test_structured_prompt_assembler.py \
  tldw_Server_API/tests/Prompt_Management/test_structured_prompt_validator.py \
  tldw_Server_API/tests/Prompt_Management/test_structured_prompt_conversion.py \
  tldw_Server_API/tests/Prompt_Management/test_single_text_recipe_validator.py \
  tldw_Server_API/tests/Prompt_Management/test_single_text_recipe_renderer.py \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompts_structured_api.py \
  tldw_Server_API/tests/Prompt_Management_NEW/property/test_prompt_properties.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py -q
```

- [ ] Run complete scoped frontend, fixture, sync, workspace, shared-assist, lint, typecheck, and E2E suites.

```bash
cd apps
bunx vitest run \
  packages/ui/src/components/Option/Prompt/__tests__/structured-prompt-utils.test.ts \
  packages/ui/src/components/Option/Prompt/Structured \
  packages/ui/src/components/Common/PromptAssist \
  packages/ui/src/services/__tests__/prompt-sync.structured-prompts.test.ts \
  packages/ui/src/components/Option/Prompt/__tests__
bunx eslint packages/ui/src/components/Common/PromptAssist packages/ui/src/components/Option/Prompt/Structured packages/ui/src/components/Option/Prompt/structured-prompt-utils.ts packages/ui/src/services/prompt-sync.ts
bun run --cwd extension compile
bun run --cwd tldw-frontend compile
```

- [ ] Run Bandit on touched Python scope and fix new findings.

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Prompt_Management/structured_prompts \
  tldw_Server_API/app/api/v1/endpoints/prompts.py \
  tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_prompts.py \
  tldw_Server_API/app/core/Prompt_Management/Prompts_Interop.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_catalog.py \
  -f json -o /tmp/bandit_task_12984_2.json
```

- [ ] Review `git diff --check`; scan for runtime values in serialization, `Record<string, any>` inside new recipe code, schema fallthrough, duplicated renderers, TODOs, unsafe HTML, and unbounded preview output.

```bash
git diff --check
rg -n "runtimeValues|runtime_values|variable_values|Record<string, any>|TODO|FIXME|dangerouslySetInnerHTML" \
  tldw_Server_API/app/core/Prompt_Management \
  apps/packages/ui/src/components/Common/PromptAssist/recipes \
  apps/packages/ui/src/services/prompt-sync.ts
```

- [ ] With every gate green, change only `single_text_recipe_v2.supported` to true and re-run capability/mixed-version tests. Keep its advertised limits sourced from the validator constants.
- [ ] Record exact verification, Bandit path, E2E evidence, known skips, touched files, and final summary in `TASK-12984.2`; mark Done only when every acceptance criterion passes.
- [ ] Commit only Track B paths and its Backlog update.

```bash
git add tldw_Server_API/app/api/v1/endpoints/prompts.py tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompts_structured_api.py 'backlog/tasks/task-12984.2 - Implement-single-text-structured-prompt-recipes.md'
git diff --cached --check
git commit -m "feat(prompts): complete single-text recipe workflows (TASK-12984.2)"
```

## Final Self-Review Checklist

- [ ] v1 models, rendering, preview, persistence, Prompt Studio, sync, export, and MCP behavior remain covered and passing.
- [ ] v2 cannot parse or execute through the v1 assembler.
- [ ] Python and TypeScript pass the same exact render/error fixtures.
- [ ] All blocks match the recipe target; target conversion is explicit.
- [ ] XML, Markdown, and free-form rendering follow the approved exact rules.
- [ ] Required/default/runtime variable behavior is distinct and tested.
- [ ] No save/update/sync/import/export payload can persist runtime values.
- [ ] Built-ins are immutable and always become user-owned copies on save.
- [ ] Opening/editing/saving does not apply; applying does not save.
- [ ] System and user Apply use Track A adapters and exact Undo.
- [ ] Prompt library badge/group/filter/open behavior is schema-aware.
- [ ] Old/offline/future-schema paths fail safely without implicit migration.
- [ ] `single_text_recipe_v2.supported` is true only after both surface and cross-consumer gates pass.
- [ ] `TASK-12984.2` contains reproducible verification and no unresolved blocker.
