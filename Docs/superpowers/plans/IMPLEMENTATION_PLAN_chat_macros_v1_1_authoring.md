# Chat Macros v1.1 Authoring and Output Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users create, edit, validate, import, export, and delete custom chat macros through the settings UI, and replace raw settings JSON with a structured output-profile editor.

**Architecture:** Keep `MACRO.yaml` as the canonical user-owned source and reuse the v1 CRUD, validation, settings, and ownership boundaries. Add only two backend contract refinements: require the API resource name to match the definition name, and support bounded custom section headings in output profiles. The frontend owns guided authoring, raw YAML import/export, destructive confirmation, and structured profile editing; the server remains the final validation authority.

**Tech Stack:** FastAPI, Pydantic v2, PyYAML, pytest, React 18, TypeScript, js-yaml, Ant Design confirmation infrastructure, lucide-react, Vitest, Testing Library, Playwright.

**Spec:** `Docs/superpowers/specs/2026-07-03-chat-macros-design.md`

**Backlog:** `TASK-13114`

## ADR Check

- **ADR required:** No new ADR.
- **Governing ADR:** [ADR-003: Jobs Vs Scheduler Default](../../ADR/003-jobs-vs-scheduler-default.md).
- **Reason:** v1.1 extends authoring and profile validation while retaining the v1 per-user YAML storage, database records, and Jobs execution ownership; it does not change a durable architecture rule.

## Global Constraints

- Built-in macro definitions remain immutable; users may disable or clone them.
- User definitions remain under `Databases/user_databases/<user_id>/macros/<macro_name>/MACRO.yaml` and all existing path, size, and symlink protections remain in force.
- `schema_version: 1`, background execution, command naming, permission rejection, and existing runtime caps remain unchanged.
- Import and export cover the canonical YAML definition only. Supporting-file bundles, ACP fork retention, foreground execution, and additional presets are separate work items.
- A user macro's name is immutable after creation. The route name, storage directory, and `name` inside YAML must match.
- Existing settings keys not owned by the output-profile editor must survive every save.
- The server is the final validation authority. Client validation improves feedback but cannot authorize an invalid macro.
- Use the existing `js-yaml`, `lucide-react`, Ant Design modal, and download utilities; add no dependencies.
- Preserve compatibility with profiles containing only `format`, `sections`, and `include_branch_outputs`.
- Keep output profiles bounded to ten sections and section identifiers bounded to 64 characters.
- UI controls must be keyboard accessible, responsive at 390px and 1440px widths, and must not overlap or resize when labels and validation messages change.

---

## File Map

- Modify `tldw_Server_API/app/core/Chat_Macros/output_profiles.py`: normalize, serialize, and render custom section headings.
- Modify `tldw_Server_API/app/core/Chat_Macros/service.py`: enforce API/YAML identity consistency during create and update.
- Modify `tldw_Server_API/tests/Chat_Macros/unit/test_macro_service.py`: test identity enforcement and heading rendering/validation.
- Modify `tldw_Server_API/tests/Chat_Macros/integration/test_chat_macros_api.py`: test public create/update mismatch responses and settings round trips.
- Modify `apps/packages/ui/src/services/chat-macros.ts`: replace loose settings/definition records with additive typed contracts.
- Modify `apps/packages/ui/src/services/__tests__/chat-macros.test.ts`: cover create, read, update, and delete client methods used by authoring.
- Create `apps/packages/ui/src/components/Option/Settings/chat-macro-editor-utils.ts`: YAML parsing, guided-draft conversion, safe import bounds, and default definition generation.
- Create `apps/packages/ui/src/components/Option/Settings/__tests__/chat-macro-editor-utils.test.ts`: pure helper coverage.
- Create `apps/packages/ui/src/components/Option/Settings/ChatMacroEditor.tsx`: guided/source authoring, validation, import/export, save, and delete UI.
- Create `apps/packages/ui/src/components/Option/Settings/__tests__/ChatMacroEditor.test.tsx`: authoring behavior and destructive confirmation tests.
- Create `apps/packages/ui/src/components/Option/Settings/OutputProfileEditor.tsx`: named profile and ordered section editing.
- Create `apps/packages/ui/src/components/Option/Settings/__tests__/OutputProfileEditor.test.tsx`: profile round-trip and validation tests.
- Modify `apps/packages/ui/src/components/Option/Settings/ChatMacrosSettings.tsx`: work-focused manager shell, selection, cloning, tabs, and data refresh.
- Modify `apps/packages/ui/src/components/Option/Settings/__tests__/ChatMacrosSettings.test.tsx`: manager integration, regression, and accessible-state tests.
- Modify `apps/packages/ui/src/assets/locale/en/settings.json`: concise labels, actions, empty states, and error text.
- Modify `tldw_Server_API/app/core/Chat_Macros/README.md`: authoring workflow and profile schema.

---

### Task 1: Harden Backend Authoring and Output-Profile Contracts

**Files:**
- Modify: `tldw_Server_API/app/core/Chat_Macros/output_profiles.py`
- Modify: `tldw_Server_API/app/core/Chat_Macros/service.py`
- Test: `tldw_Server_API/tests/Chat_Macros/unit/test_macro_service.py`
- Test: `tldw_Server_API/tests/Chat_Macros/integration/test_chat_macros_api.py`

**Interfaces:**
- Consumes: `ChatMacrosService.create_macro(name, raw, supporting_files)` and `update_macro(name, raw, supporting_files)` from v1.
- Produces: `MacroOutputProfile.section_titles: dict[str, str]`, accepted settings key `section_titles`, and a stable `MacroValidationError` when route/storage identity differs from YAML identity.

- [x] **Step 1: Write failing identity and heading tests**

Add focused cases equivalent to:

```python
def test_create_macro_rejects_definition_name_mismatch(service: ChatMacrosService) -> None:
    with pytest.raises(MacroValidationError, match="must match"):
        service.create_macro("daily_digest", _macro_yaml(name="other_name"))


def test_update_macro_rejects_definition_rename(service: ChatMacrosService) -> None:
    service.create_macro("daily_digest", _macro_yaml(name="daily_digest"))
    with pytest.raises(MacroValidationError, match="must match"):
        service.update_macro("daily_digest", _macro_yaml(name="renamed"))


def test_output_profile_renders_custom_section_titles() -> None:
    profile = normalize_output_profile(
        "handoff",
        {
            "format": "structured_sections",
            "sections": ["summary", "action_items"],
            "section_titles": {"summary": "Executive brief", "action_items": "Owners and dates"},
        },
    )
    rendered = render_output_profile(profile, {"summary": "S", "action_items": "A"})
    assert "## Executive brief" in rendered
    assert "## Owners and dates" in rendered


def test_output_profile_rejects_titles_for_unknown_sections() -> None:
    with pytest.raises(MacroValidationError, match="unknown section"):
        normalize_output_profile(
            "bad",
            {"sections": ["summary"], "section_titles": {"risks": "Risk register"}},
        )
```

Add API cases asserting create/update mismatches return `400`, no user directory is created or renamed, and `GET/PUT /settings` round-trips `section_titles`.

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Chat_Macros/unit/test_macro_service.py \
  tldw_Server_API/tests/Chat_Macros/integration/test_chat_macros_api.py \
  -q
```

Expected: the identity tests fail because mismatched names are accepted, and heading tests fail because `section_titles` is currently an unknown key.

- [x] **Step 3: Implement the minimal backend contract**

Extend the normalized profile without changing existing keys:

```python
MAX_SECTION_TITLE_LENGTH = 128


@dataclass(slots=True)
class MacroOutputProfile:
    name: str = "default"
    format: str = "structured_sections"
    sections: list[str] = field(default_factory=lambda: list(DEFAULT_PROFILE_SECTIONS))
    section_titles: dict[str, str] = field(default_factory=dict)
    include_branch_outputs: bool = False
```

`normalize_output_profile()` must:

- accept only `format`, `sections`, `section_titles`, and `include_branch_outputs`;
- require `section_titles` to be a mapping;
- require every title key to exist in `sections`;
- require each title to be a non-empty string of at most 128 characters;
- copy the normalized mapping in `merge_output_profile()` and `profile_to_dict()`;
- use `profile.section_titles.get(section) or _title(section)` when rendering structured sections.

In `ChatMacrosService`, validate identity before collision or storage operations:

```python
def _require_matching_name(self, resource_name: str, definition: MacroDefinition) -> None:
    if definition.name != resource_name:
        raise MacroValidationError(
            f"macro definition name '{definition.name}' must match resource name '{resource_name}'"
        )
```

Call it from both `create_macro()` and `update_macro()` immediately after `validate_macro(raw)`.

- [x] **Step 4: Run focused and full Chat Macros tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Chat_Macros/unit/test_macro_service.py \
  tldw_Server_API/tests/Chat_Macros/integration/test_chat_macros_api.py \
  -q
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Chat_Macros \
  -q
```

Expected: all focused tests and all Chat Macros tests pass.

- [x] **Step 5: Run Bandit on the changed backend scope**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/Chat_Macros/output_profiles.py \
  tldw_Server_API/app/core/Chat_Macros/service.py \
  -f json -o /tmp/bandit_chat_macros_v1_1_backend.json
```

Expected: no new findings in changed production code.

- [x] **Step 6: Commit the backend contract**

```bash
git add \
  tldw_Server_API/app/core/Chat_Macros/output_profiles.py \
  tldw_Server_API/app/core/Chat_Macros/service.py \
  tldw_Server_API/tests/Chat_Macros/unit/test_macro_service.py \
  tldw_Server_API/tests/Chat_Macros/integration/test_chat_macros_api.py
git commit -m "feat(chat-macros): harden authoring contracts (TASK-13114)"
```

---

### Task 2: Add Typed Frontend YAML and Profile Helpers

**Files:**
- Modify: `apps/packages/ui/src/services/chat-macros.ts`
- Modify: `apps/packages/ui/src/services/__tests__/chat-macros.test.ts`
- Create: `apps/packages/ui/src/components/Option/Settings/chat-macro-editor-utils.ts`
- Create: `apps/packages/ui/src/components/Option/Settings/__tests__/chat-macro-editor-utils.test.ts`

**Interfaces:**
- Consumes: existing v1 `ChatMacroDetail.raw`, CRUD service calls, and `ChatMacroSettingsResponse.settings`.
- Produces: `ChatMacroDefinition`, `ChatMacroSettings`, `ChatMacroOutputProfile`, `GuidedMacroDraft`, `parseMacroSource(raw)`, `serializeGuidedMacro(draft)`, `createBlankMacroDraft()`, `readMacroImport(file)`, and `outputProfilesToSettings(settings, profiles)`.

- [x] **Step 1: Write failing service and pure-helper tests**

Cover all CRUD methods used by the UI:

```typescript
it("creates, loads, updates, and deletes user macros", async () => {
  await createChatMacro({ name: "handoff", raw: "schema_version: 1" })
  await getChatMacro("handoff")
  await updateChatMacro("handoff", { raw: "schema_version: 1" })
  await deleteChatMacro("handoff")

  expect(mocks.apiSend).toHaveBeenNthCalledWith(1, expect.objectContaining({
    path: "/api/v1/chat/macros", method: "POST"
  }))
  expect(mocks.apiSend).toHaveBeenNthCalledWith(4, {
    path: "/api/v1/chat/macros/handoff", method: "DELETE"
  })
})
```

Add pure-helper cases for:

- a valid guided draft serializing to `schema_version: 1` YAML;
- parse/serialize preserving name, command, branch order, merge prompt, output profile, and caps;
- unsupported step types returning `mode: "source"` without rewriting raw YAML;
- malformed/non-mapping YAML returning a bounded client error;
- import rejecting files over 500,000 bytes before `file.text()` is called;
- output-profile edits preserving `disabled_builtins`, `user_macro_enabled`, and unknown future top-level keys.

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
bunx vitest run \
  src/services/__tests__/chat-macros.test.ts \
  src/components/Option/Settings/__tests__/chat-macro-editor-utils.test.ts
```

Expected: missing types/helpers and assertions for CRUD methods not imported by the existing service test.

- [x] **Step 3: Add additive API types**

Define explicit contracts while keeping request functions unchanged:

```typescript
export interface ChatMacroStep {
  id: string
  type: "prompt" | "branch_prompt" | "merge" | "post_result"
  label?: string | null
  output?: string | null
  consumes?: string[]
  prompt?: string | null
  branch_strategy?: "auto" | "chat_native" | "acp_fork" | null
}

export interface ChatMacroDefinition {
  schema_version: 1
  name: string
  command: string
  description?: string | null
  enabled: boolean
  args: Record<string, unknown>
  context: Record<string, unknown>
  execution: Record<string, unknown>
  steps: ChatMacroStep[]
  output_profile: string
  permissions: { tool_calls: string[]; skills: string[] }
}

export interface ChatMacroOutputProfile {
  format: "structured_sections" | "single_response"
  sections: string[]
  section_titles: Record<string, string>
  include_branch_outputs: boolean
}

export interface ChatMacroSettings extends Record<string, unknown> {
  disabled_builtins: string[]
  user_macro_enabled: Record<string, boolean>
  output_profiles: Record<string, ChatMacroOutputProfile>
}
```

Use these types in `ChatMacroDetail` and `ChatMacroSettingsResponse` without narrowing server compatibility at runtime.

- [x] **Step 4: Implement pure authoring helpers**

Use `load` and `dump` from the existing `js-yaml` dependency. Define a guided draft that represents the supported common topology:

```typescript
export interface GuidedMacroDraft {
  name: string
  command: string
  description: string
  outputProfile: string
  maxBranches: number
  maxConcurrency: number
  timeoutSeconds: number
  branches: Array<{ id: string; label: string; output: string; prompt: string }>
  merge: { id: string; output: string; prompt: string }
}
```

`serializeGuidedMacro()` must emit:

- v1 context defaults;
- background-only execution defaults and bounded numeric values;
- one `branch_prompt` per draft branch;
- one merge step consuming all branch output names;
- one `post_result` step consuming the merge output;
- empty permissions arrays.

`parseMacroSource()` must enter guided mode only when the definition uses this topology. Any valid but more advanced definition stays editable in source mode and retains its exact raw text until the user explicitly changes modes.

`readMacroImport()` must accept `.yaml` and `.yml`, reject files above 500,000 bytes, and return text only; backend validation remains mandatory before save.

`outputProfilesToSettings()` must shallow-copy the original settings and replace only `output_profiles`.

- [x] **Step 5: Run helper and service tests**

Run:

```bash
bunx vitest run \
  src/services/__tests__/chat-macros.test.ts \
  src/components/Option/Settings/__tests__/chat-macro-editor-utils.test.ts
```

Expected: all tests pass with no new warnings beyond the repository baseline.

- [x] **Step 6: Commit the typed helper layer**

```bash
git add \
  apps/packages/ui/src/services/chat-macros.ts \
  apps/packages/ui/src/services/__tests__/chat-macros.test.ts \
  apps/packages/ui/src/components/Option/Settings/chat-macro-editor-utils.ts \
  apps/packages/ui/src/components/Option/Settings/__tests__/chat-macro-editor-utils.test.ts
git commit -m "feat(chat-macros): add authoring helpers (TASK-13114)"
```

---

### Task 3: Build the Macro Authoring Editor

**Files:**
- Create: `apps/packages/ui/src/components/Option/Settings/ChatMacroEditor.tsx`
- Create: `apps/packages/ui/src/components/Option/Settings/__tests__/ChatMacroEditor.test.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/settings.json`

**Interfaces:**
- Consumes: Task 2 helpers plus `createChatMacro`, `getChatMacro`, `updateChatMacro`, `deleteChatMacro`, `validateChatMacro`, and `useConfirmDanger`.
- Produces: `ChatMacroEditor({ selected, outputProfileNames, onSaved, onDeleted, onCloneRequested })`.

- [x] **Step 1: Write failing component tests**

Test user-visible behavior, not internal state:

```typescript
it("creates a guided macro only after server validation succeeds", async () => {
  renderEditor({ selected: null })
  await user.type(screen.getByLabelText("Name"), "handoff")
  await user.type(screen.getByLabelText("Command"), "handoff")
  await user.type(screen.getByLabelText("Branch prompt 1"), "List decisions")
  await user.click(screen.getByRole("button", { name: "Save macro" }))

  await waitFor(() => expect(mocks.validateChatMacro).toHaveBeenCalled())
  expect(mocks.createChatMacro).toHaveBeenCalledWith(expect.objectContaining({ name: "handoff" }))
})
```

Also cover:

- backend validation failure is announced and prevents create/update;
- edit loads raw detail and keeps name read-only;
- switching to source mode exposes raw YAML without losing it;
- importing YAML populates source mode but does not persist automatically;
- exporting creates a `.yaml` Blob containing the exact server raw source;
- deleting requires `useConfirmDanger`, defaults focus to cancel, and refreshes only after success;
- built-ins are read-only and offer clone instead of edit/delete;
- stale detail requests cannot replace a newer selection.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
bunx vitest run src/components/Option/Settings/__tests__/ChatMacroEditor.test.tsx
```

Expected: import failure because `ChatMacroEditor.tsx` does not exist.

- [x] **Step 3: Implement the editor state and save sequence**

Use a request-generation ref or `AbortController`-equivalent stale-response guard for detail loads. The save path must be:

```typescript
const validation = await validateChatMacro(raw)
if (!validation.ok || !validation.data?.valid) {
  setValidationError(validation.data?.error ?? responseError(validation.status, validation.error))
  return
}

const response = isCreate
  ? await createChatMacro({ name: draftName, raw })
  : await updateChatMacro(selected.name, { raw })
```

Never infer success from client parsing alone. Disable repeated submissions while a request is active and retain the draft after any failure.

- [x] **Step 4: Implement the work-focused authoring UI**

Use a stable two-column editor on wide screens and a single column on mobile:

- compact identity fields at the top;
- a segmented `Guided` / `YAML` mode control;
- branch rows with icon-only move/delete buttons and tooltips;
- add-branch button, merge prompt, output profile selection, and bounded numeric controls;
- monospace source textarea with server validation result;
- `Upload`, `Download`, `Copy`, `Save`, and `Delete` actions using lucide icons where available;
- no nested cards, oversized headings, decorative gradients, or explanatory feature copy;
- fixed button and control dimensions so validation messages do not shift the toolbar.

The hidden import input must use `accept=".yaml,.yml,text/yaml,text/plain"`. Export must call the existing `downloadBlob()` utility with the exact raw source and `${name}.yaml`.

- [x] **Step 5: Run component and accessibility checks**

Run:

```bash
bunx vitest run src/components/Option/Settings/__tests__/ChatMacroEditor.test.tsx
```

Expected: all editor tests pass; all controls have accessible names and destructive actions remain cancellable.

- [x] **Step 6: Commit the macro editor**

```bash
git add \
  apps/packages/ui/src/components/Option/Settings/ChatMacroEditor.tsx \
  apps/packages/ui/src/components/Option/Settings/__tests__/ChatMacroEditor.test.tsx \
  apps/packages/ui/src/assets/locale/en/settings.json
git commit -m "feat(chat-macros): add macro authoring editor (TASK-13114)"
```

---

### Task 4: Build the Structured Output-Profile Editor

**Files:**
- Create: `apps/packages/ui/src/components/Option/Settings/OutputProfileEditor.tsx`
- Create: `apps/packages/ui/src/components/Option/Settings/__tests__/OutputProfileEditor.test.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/settings.json`

**Interfaces:**
- Consumes: `ChatMacroSettings`, `ChatMacroOutputProfile`, `outputProfilesToSettings()`, and `updateChatMacroSettings()`.
- Produces: `OutputProfileEditor({ settings, onSaved })` that preserves unrelated settings keys.

- [x] **Step 1: Write failing profile-editor tests**

Cover the contract with behavior assertions:

```typescript
it("saves ordered sections and custom headings without dropping other settings", async () => {
  renderProfileEditor()
  await user.click(screen.getByRole("button", { name: "Add section" }))
  await user.type(screen.getByLabelText("Section key 3"), "risks")
  await user.type(screen.getByLabelText("Section heading 3"), "Risk register")
  await user.click(screen.getByRole("button", { name: "Save profiles" }))

  expect(mocks.updateChatMacroSettings).toHaveBeenCalledWith(expect.objectContaining({
    disabled_builtins: ["example"],
    output_profiles: expect.objectContaining({
      default: expect.objectContaining({
        sections: ["summary", "action_items", "risks"],
        section_titles: { risks: "Risk register" }
      })
    })
  }))
})
```

Also cover:

- switching between `Structured sections` and `Single response`;
- toggling branch-result inclusion;
- moving sections up/down without layout shift;
- creating and deleting named profiles while protecting `default`;
- duplicate/invalid/over-ten section errors preventing save;
- server save failure retaining edits;
- a settings-load failure rendering retry rather than editable defaults.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
bunx vitest run src/components/Option/Settings/__tests__/OutputProfileEditor.test.tsx
```

Expected: import failure because `OutputProfileEditor.tsx` does not exist.

- [x] **Step 3: Implement profile draft validation**

Before persistence, require:

```typescript
const SECTION_KEY = /^[a-z][a-z0-9_]{0,63}$/

if (sections.length > 10) errors.push("A profile can contain at most 10 sections.")
if (new Set(sections).size !== sections.length) errors.push("Section keys must be unique.")
if (sections.some((section) => !SECTION_KEY.test(section))) {
  errors.push("Section keys must use lowercase letters, numbers, and underscores.")
}
```

Require profile names to use the same lowercase command-safe pattern and custom headings to be at most 128 characters. Keep empty headings out of `section_titles` so server-generated titles remain the default.

- [x] **Step 4: Implement profile controls**

Use:

- a profile select plus icon actions for add/delete;
- segmented buttons for response format;
- a checkbox/switch for branch outputs;
- stable rows for key, heading, move up, move down, and remove;
- an explicit save button and inline live-region status;
- no raw settings JSON in the primary interface.

When saving, call `outputProfilesToSettings(originalSettings, drafts)` and send the full preserved settings object. Replace local state with the normalized server response after success.

- [x] **Step 5: Run profile tests**

Run:

```bash
bunx vitest run src/components/Option/Settings/__tests__/OutputProfileEditor.test.tsx
```

Expected: all profile tests pass.

- [x] **Step 6: Commit the output-profile editor**

```bash
git add \
  apps/packages/ui/src/components/Option/Settings/OutputProfileEditor.tsx \
  apps/packages/ui/src/components/Option/Settings/__tests__/OutputProfileEditor.test.tsx \
  apps/packages/ui/src/assets/locale/en/settings.json
git commit -m "feat(chat-macros): add output profile editor (TASK-13114)"
```

---

### Task 5: Integrate the Manager, Verify UX, and Document v1.1

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Settings/ChatMacrosSettings.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/__tests__/ChatMacrosSettings.test.tsx`
- Modify: `tldw_Server_API/app/core/Chat_Macros/README.md`
- Modify: `backlog/tasks/task-13114 - Implement-Chat-Macros-v1.1-authoring-and-output-profiles.md`

**Interfaces:**
- Consumes: `ChatMacroEditor`, `OutputProfileEditor`, existing macro list/toggle/clone calls, and settings list route.
- Produces: the final `/settings/chat-macros` manager with `Macros` and `Output profiles` tabs and unchanged chat/run behavior.

- [x] **Step 1: Rewrite manager tests around the final workflow**

Retain existing toggle and clone regressions, then add:

- `New macro` opens a blank editor;
- selecting a user macro loads it in the editor;
- selecting a built-in exposes disable/clone but not edit/delete;
- save/delete/clone refresh the catalog and preserve selection where possible;
- `Output profiles` renders only after settings load succeeds;
- macro-list and settings failures have independent retry actions;
- narrow viewport markup does not hide primary actions;
- no state update occurs after unmount or stale refresh completion.

- [x] **Step 2: Run the integration test and verify RED**

Run:

```bash
bunx vitest run src/components/Option/Settings/__tests__/ChatMacrosSettings.test.tsx
```

Expected: assertions fail against the v1 table, standalone clone panel, YAML validator, and raw settings textarea.

- [x] **Step 3: Integrate the manager shell**

Replace the v1 page composition with:

- a compact page header and `New macro`, `Import`, and refresh actions;
- `Macros` / `Output profiles` tabs;
- within `Macros`, a scan-friendly list/sidebar and the selected editor in an unframed main pane;
- source, enabled state, and validation status visible without opening secondary dialogs;
- independent loading/error boundaries for catalog, detail, and settings;
- clone action attached to the selected built-in instead of the first catalog row.

Keep list and editor dimensions stable with explicit grid tracks such as `minmax(220px, 300px) minmax(0, 1fr)`, collapsing to one column below the existing large breakpoint.

- [x] **Step 4: Update module documentation**

Document in `README.md`:

- create/edit/delete and built-in clone behavior;
- immutable macro names and API/YAML identity matching;
- YAML-only import/export and the 500,000-byte source bound;
- guided mode's supported topology and source-mode fallback;
- output-profile formats, ordered sections, custom `section_titles`, branch inclusion, and ten-section cap;
- server-side validation, permissions rejection, path safety, and user ownership;
- explicitly deferred supporting-file bundles, ACP fork retention UI, foreground execution, and extra presets.

- [x] **Step 5: Run focused frontend and backend verification**

Run:

```bash
bunx vitest run \
  src/services/__tests__/chat-macros.test.ts \
  src/components/Option/Settings/__tests__/chat-macro-editor-utils.test.ts \
  src/components/Option/Settings/__tests__/ChatMacroEditor.test.tsx \
  src/components/Option/Settings/__tests__/OutputProfileEditor.test.tsx \
  src/components/Option/Settings/__tests__/ChatMacrosSettings.test.tsx \
  src/components/Option/ChatWorkspace/__tests__/MacroStatusCard.test.tsx \
  src/components/Option/ChatWorkspace/__tests__/MacroRunDetailDrawer.test.tsx \
  src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx

/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Chat_Macros \
  tldw_Server_API/tests/Services/test_chat_macros_jobs_worker_startup.py \
  -q
```

Expected: all focused frontend and backend tests pass.

- [x] **Step 6: Run static and security checks**

Run:

```bash
cd apps/packages/ui && bunx tsc --noEmit
git diff --check
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/Chat_Macros \
  tldw_Server_API/app/api/v1/endpoints/chat_macros.py \
  tldw_Server_API/app/api/v1/schemas/chat_macros.py \
  -f json -o /tmp/bandit_chat_macros_v1_1_final.json
```

Expected: no type errors, whitespace errors, or new Bandit findings.

- [x] **Step 7: Run desktop and mobile visual verification**

Start the existing frontend dev server on an unused port and use Playwright against `/settings/chat-macros` at:

- desktop: 1440 x 1000;
- mobile: 390 x 844.

Verify:

- list, tabs, toolbars, editor fields, and validation messages do not overlap;
- all button labels fit and icon buttons expose tooltips/accessibility names;
- guided/source switching and profile switching do not resize the page unexpectedly;
- delete confirmation defaults focus to cancel;
- imported YAML and custom section headings remain visible after save and reload;
- built-in `/wrapup` remains immutable and clonable.

Store screenshots under `/tmp/chat-macros-v1-1-visual-qa/`; do not commit them.

- [x] **Step 8: Update Backlog and commit integration**

Record touched files, exact verification results, known skips, and final summary in `TASK-13114`. Check acceptance criteria and Definition of Done only after evidence exists.

```bash
git add \
  apps/packages/ui/src/components/Option/Settings/ChatMacrosSettings.tsx \
  apps/packages/ui/src/components/Option/Settings/__tests__/ChatMacrosSettings.test.tsx \
  tldw_Server_API/app/core/Chat_Macros/README.md \
  "backlog/tasks/task-13114 - Implement-Chat-Macros-v1.1-authoring-and-output-profiles.md"
git commit -m "feat(chat-macros): complete v1.1 authoring workflow (TASK-13114)"
```

---

## Final Review Checklist

- [x] Every production behavior was preceded by a failing test that failed for the intended missing behavior.
- [x] Resource name, YAML name, storage directory, and catalog identity cannot diverge.
- [x] Existing v1 macros and output profiles remain valid without migration.
- [x] Advanced YAML remains exact until the user edits it; guided mode never silently discards unsupported fields.
- [x] Import does not persist automatically, and export uses the exact canonical source.
- [x] Delete is user-scoped, unavailable for built-ins, and requires accessible confirmation.
- [x] Output-profile saves preserve unrelated settings and server normalization replaces local state.
- [x] Existing run, cancel, retry, status-card, and `/wrapup` tests remain green.
- [x] Frontend tests, backend tests, typecheck, `git diff --check`, Bandit, and responsive visual QA are recorded in Backlog.

## Baseline Recorded Before Implementation

- Backend: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Chat_Macros -q` -> `134 passed, 2 warnings`.
- Frontend component: `ChatMacrosSettings.test.tsx` -> `4 passed`.
- Frontend service baseline could not collect because the isolated worktree dependency links did not resolve `wxt/browser`; this is an environment/setup blocker, not a test assertion failure. Before Task 2, use a complete workspace install or a known-good worktree dependency layout instead of adding further ad hoc links.
