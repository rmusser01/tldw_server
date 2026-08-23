# Conversation Title Service Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users customize the automatic conversation-title prompt through Settings → Workflow prompts while preserving current title feature flags, provider-call invariants, fallback behavior, and account/server scope safety.

**Architecture:** Add one synchronous `chat.title.generation` definition to the existing static Service Prompts registry and compatibility fixture. Resolve and render that definition centrally inside `generateTitle`, using one immutable snapshot bound to the caller's expected request scope, so every existing automatic-title caller receives the behavior without caller-specific branches. Keep enablement in Chat settings and show only a localized explanatory link in Workflow prompts.

**Tech Stack:** FastAPI/Python registry and pytest; React/TypeScript, React Router, i18next, Vitest, and Bun; existing Service Prompts API, renderer, scope lease, and Settings editor.

**Spec:** `Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md`; candidate contract `Docs/Design/service-prompt-inventory.md` row `chat.title.generation`; approved bounded-slice decisions from the 2026-08-23 in-chat design review.

## Global Constraints

- Register exactly one new synchronous definition: `chat.title.generation`.
- Its only editable part is `user_template`, in `template` mode, with exactly one required variable: `query`.
- Preserve byte-equivalent no-override provider content by mapping the legacy `{{query}}` marker to the registered `{query}` field before single-pass rendering.
- Preserve `titleGenEnabled` as disabled by default; disabled title generation performs no Service Prompt request.
- Preserve one human message, `toolChoice: "none"`, `saveToDb: false`, caller-selected model, `removeReasoning`, and caller fallback behavior.
- Ordinary setting, prompt-loading, rendering, and provider failures return the caller fallback without logging authored prompt content; cancellation and request-scope changes fail closed.
- A catalog 404 from an older server uses only the packaged client compatibility default. There is no browser-local migration key for conversation titles.
- Workflow prompts always displays Conversation title; it does not duplicate the Chat-settings toggle.
- Do not add a database migration, endpoint, history UI, arbitrary prompt IDs, asynchronous snapshot persistence, or any broader prompt rollout.
- Follow strict RED → GREEN → REFACTOR for every production behavior change.

---

### Task 1: Register the title definition and golden compatibility contract

**Stage:** 1 of 4 — Registry contract

**Goal:** Make the existing generic Service Prompts registry and API recognize a title template whose rendered default is byte-equivalent to the current provider message.

**Success Criteria:** Catalog metadata is stable, the shared fixture contains the canonical `{query}` template, and the generic GET/PUT/DELETE path saves and resets title overrides.

**Tests:** Backend registry/default fixture tests and one title-specific API save/reset test.

**Status:** Complete

**Files:**
- Modify: `apps/packages/ui/src/utils/__fixtures__/service-prompt-rendering.json`
- Modify: `tldw_Server_API/tests/Prompt_Management/test_service_prompts.py`
- Modify: `tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py`
- Modify: `tldw_Server_API/app/core/Prompt_Management/service_prompts.py`

**Interfaces:**
- Consumes: existing `ServicePromptDefinition`, `ServicePromptPart`, `ServicePromptWorkflow`, validator, and generic API/store operations.
- Produces: registry definition `chat.title.generation` with `default_parts.user_template` and workflow ID `chat.title.generation`.

- [ ] **Step 1: Add the failing registry and provider-byte golden tests**

  Add this literal metadata to `EXPECTED_REGISTRY` and the canonical default to the shared fixture:

  ```python
  "chat.title.generation": {
      "label": "Conversation title",
      "description": "Controls the instruction used to generate automatic conversation titles.",
      "parts": (("user_template", "User template", "template", ("query",)),),
      "workflows": (("chat.title.generation", "Automatic conversation titles"),),
  },
  ```

  ```json
  "chat.title.generation": {
    "user_template": "Here is the query:\n\n--------------\n\n{query}\n\n--------------\n\nCreate a concise, 3-5 word phrase as a title for the previous query. Avoid quotation marks or special formatting. RESPOND ONLY WITH THE TITLE TEXT. ANSWER USING THE SAME LANGUAGE AS THE QUERY.\n\n\nExamples of titles:\n\nStellar Achievement Celebration\nFamily Bonding Activities\n🇫🇷 Voyage à Paris\n🍜 Receta de Ramen Casero\nShakespeare Analyse Literarische\n日本の春祭り体験\nДревнегреческая Философия Обзор\n\nResponse:"
  }
  ```

  Add a shared rendering case with the hand-derived expected provider content so changing `{query}` handling or reparsing inserted braces fails both the Python and TypeScript fixture tests:

  ```json
  {
    "name": "title query remains single-pass",
    "definition_id": "chat.title.generation",
    "part_key": "user_template",
    "authored_text": "Title literal {{query}} from {query}",
    "values": { "query": "Explain {query} literally" },
    "expected": "Title literal {query} from Explain {query} literally"
  }
  ```

- [ ] **Step 2: Run the backend registry tests and verify RED**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    tldw_Server_API/tests/Prompt_Management/test_service_prompts.py
  ```

  Expected: FAIL because `chat.title.generation` is absent from `_DEFINITION_SEQUENCE` and the fixture no longer equals registry defaults.

- [ ] **Step 3: Add a failing generic API save/reset test for the title definition**

  In `test_service_prompts_api.py`, define literal title constants and exercise the real router/store:

  ```python
  TITLE_ID = "chat.title.generation"
  TITLE_PATH = f"/api/v1/service-prompts/{TITLE_ID}"
  TITLE_CUSTOM_PARTS = {"user_template": "Name this request: {query}"}

  def test_title_prompt_can_be_saved_and_reset_through_generic_api(api_context) -> None:
      saved = api_context.client.put(
          TITLE_PATH,
          json={"parts": TITLE_CUSTOM_PARTS, "expected_revision": None},
      )
      assert saved.status_code == 200
      assert saved.json()["effective_parts"] == TITLE_CUSTOM_PARTS

      reset = api_context.client.delete(
          TITLE_PATH,
          params={"expected_revision": saved.json()["revision"]},
      )
      assert reset.status_code == 200
      assert reset.json()["source"] == "packaged"
      assert reset.json()["effective_parts"] == reset.json()["default_parts"]
  ```

- [ ] **Step 4: Run the API test and verify RED**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py \
    -k title_prompt
  ```

  Expected: FAIL with the registered-definition 404.

- [ ] **Step 5: Implement the minimal backend definition**

  Add `_TITLE_GENERATION_USER_TEMPLATE_DEFAULT` with the exact fixture bytes and insert one immutable definition into `_DEFINITION_SEQUENCE`:

  ```python
  ServicePromptDefinition(
      id="chat.title.generation",
      label="Conversation title",
      description="Controls the instruction used to generate automatic conversation titles.",
      parts=(
          ServicePromptPart(
              key="user_template",
              label="User template",
              mode="template",
              required_variables=("query",),
          ),
      ),
      default_parts=MappingProxyType(
          {"user_template": _TITLE_GENERATION_USER_TEMPLATE_DEFAULT}
      ),
      affected_workflows=(
          ServicePromptWorkflow(
              id="chat.title.generation",
              label="Automatic conversation titles",
          ),
      ),
  )
  ```

- [ ] **Step 6: Run the complete focused backend tests and verify GREEN**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    tldw_Server_API/tests/Prompt_Management/test_service_prompts.py \
    tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py
  ```

  Expected: all focused backend tests pass.

- [ ] **Step 7: Commit the registry contract**

  ```bash
  git add \
    apps/packages/ui/src/utils/__fixtures__/service-prompt-rendering.json \
    tldw_Server_API/app/core/Prompt_Management/service_prompts.py \
    tldw_Server_API/tests/Prompt_Management/test_service_prompts.py \
    tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py
  git commit -m "feat(service-prompts): register conversation title template"
  ```

---

### Task 2: Resolve and render one scope-bound title snapshot

**Stage:** 2 of 4 — Runtime consumer

**Goal:** Make `generateTitle` use the registered prompt for supported and older servers without allowing account/server changes to select another scope.

**Success Criteria:** Custom/default templates reach the provider once, disabled mode performs no prompt read, ordinary failures fall back, scope changes rethrow canonically, and every acquired lease is released.

**Tests:** Real TypeScript renderer behavior with network/model boundaries mocked; real snapshot loader behavior with client/auth boundaries mocked.

**Status:** Complete

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/domains/service-prompts.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/__tests__/service-prompts.test.ts`
- Modify: `apps/packages/ui/src/services/tldw-server.ts`
- Modify: `apps/packages/ui/src/services/service-prompts.ts`
- Modify: `apps/packages/ui/src/services/__tests__/service-prompts.test.ts`
- Modify: `apps/packages/ui/src/services/title.ts`
- Modify: `apps/packages/ui/src/services/__tests__/title.service-prompt-scope.test.ts`

**Interfaces:**
- Consumes: `loadServicePromptSnapshot(ids, options)`, `renderServicePromptPart`, `createServicePromptScopeChangedError`, the existing `pageAssistModel`, and `TitleGenerationOptions.requestScope`.
- Produces: `KnownServicePromptId` member `chat.title.generation`; snapshot option `requestScope?: ServicePromptRequestScope`; scope-bound central title generation.

- [ ] **Step 1: Write failing client registry and older-server fallback tests**

  Extend the literal test definitions with:

  ```typescript
  "chat.title.generation": definition("chat.title.generation", [{
    key: "user_template",
    label: "User template",
    mode: "template",
    required_variables: ["query"]
  }])
  ```

  Add an old-server snapshot test that requests only `chat.title.generation`, forces catalog 404, and asserts the returned definition is packaged with `user_template` equal to `fixture.defaults["chat.title.generation"].user_template`. Assert the legacy RAG/web-search storage helpers were not called.

- [ ] **Step 2: Write the failing expected-request-scope test**

  Exercise the real loader with an expected account different from `resolveServicePromptScope`:

  ```typescript
  await expect(loadServicePromptSnapshot(
    ["chat.title.generation"],
    { requestScope: { config: scopeOne.config, userId: 999 } }
  )).rejects.toMatchObject({
    status: 412,
    details: { detail: { code: "request_config_scope_changed" } }
  })
  expect(mocks.listServicePrompts).not.toHaveBeenCalled()
  ```

  Add separate literals for a mismatched server and a mismatched `expectedSingleUserApiKeyScope`; each must fail before catalog/detail requests.

- [ ] **Step 3: Run the snapshot tests and verify RED**

  Run:

  ```bash
  bunx vitest run \
    src/services/__tests__/service-prompts.test.ts \
    src/services/tldw/domains/__tests__/service-prompts.test.ts
  ```

  Expected: type/test failures because the ID, legacy definition/default, and `requestScope` loader option do not exist.

- [ ] **Step 4: Add the minimal client definition, compatibility default, and scope check**

  - Add `"chat.title.generation"` to `KnownServicePromptId`.
  - Add the canonical `{query}` `user_template` to `LEGACY_SERVICE_PROMPT_DEFAULTS`.
  - Add one `LEGACY_RENDER_DEFINITIONS` entry and construct its old-server snapshot directly from the packaged compatibility default, with no migration storage read.
  - Extend `loadServicePromptSnapshot` options with `requestScope?: ServicePromptRequestScope`.
  - Immediately after `resolveServicePromptScope`, before `lease.bind`, catalog, migration, or detail reads, require:

  ```typescript
  const expectedRequestScope = options.requestScope
  if (expectedRequestScope) {
    const expectedMatchesResolved =
      servicePromptTargetsMatch(expectedRequestScope.config, scope.config) &&
      (expectedRequestScope.config.expectedSingleUserApiKeyScope ?? null) ===
        (scope.config.expectedSingleUserApiKeyScope ?? null) &&
      (expectedRequestScope.userId === null
        ? scope.userId === null
        : String(expectedRequestScope.userId) === String(scope.userId))

    if (!expectedMatchesResolved) {
      throw createServicePromptScopeChangedError()
    }
  }
  ```

- [ ] **Step 5: Run the snapshot tests and verify GREEN**

  Run the Step 3 command. Expected: all tests pass.

- [ ] **Step 6: Rewrite title tests first around observable provider content and failure behavior**

  Mock only the snapshot-loading network boundary and model provider. Return a complete frozen snapshot with a real render definition, custom part, request scope, both signals, and `release` spy. Add tests that prove:

  ```typescript
  expect(invokedMessages[0].content).toBe(
    "Custom title for literal {query}: What changed?"
  )
  expect(result).toBe("Scoped title")
  expect(snapshot.release).toHaveBeenCalledOnce()
  ```

  Also add independent tests for:

  - disabled setting → fallback, with no snapshot/model request;
  - setting read failure → fallback without a prompt/model request;
  - custom template and packaged template → hand-derived provider bytes;
  - ordinary snapshot-load, render, and model errors → fallback;
  - fallback logging contains only `Error generating title` and not a secret error/prompt value;
  - expected-scope mismatch and structured 412 → rethrow;
  - snapshot scope invalidation → canonical 412, not a fallback;
  - caller abort → AbortError rethrow;
  - release runs after success, fallback, abort, and scope invalidation.

- [ ] **Step 7: Run title tests and verify RED**

  Run:

  ```bash
  bunx vitest run src/services/__tests__/title.service-prompt-scope.test.ts
  ```

  Expected: FAIL because `generateTitle` still renders the local constant and never loads/releases a Service Prompt snapshot.

- [ ] **Step 8: Implement the minimal central title consumer**

  Keep the setting check at the start of the `try` so disabled mode remains request-free while a setting-storage failure still preserves the completed chat. In the enabled path:

  ```typescript
  let snapshot: ServicePromptSnapshot | null = null
  try {
    const isEnabled = await isTitleGenEnabled()
    throwIfAborted(options.signal)
    if (!isEnabled) return fallBackTitle

    snapshot = await loadServicePromptSnapshot(
      ["chat.title.generation"],
      { signal: options.signal, requestScope: options.requestScope }
    )
    const promptConfig = snapshot.definitions["chat.title.generation"]
    if (!promptConfig) throw new Error("Conversation title prompt is unavailable.")

    const prompt = renderServicePromptPart(
      promptConfig.definition,
      "user_template",
      promptConfig.parts.user_template,
      { query }
    )
    const titleModel = await pageAssistModel({
      model,
      toolChoice: "none",
      saveToDb: false,
      requestScope: snapshot.requestScope
    })
    const title = await titleModel.invoke(
      [new HumanMessage({ content: prompt })],
      { signal: snapshot.scopeSignal }
    )
    return removeReasoning(title.content.toString())
  } catch (error) {
    if (snapshot?.scopeInvalidatedSignal.aborted) {
      throw createServicePromptScopeChangedError()
    }
    if (options.signal?.aborted ||
        (error as { name?: unknown } | null)?.name === "AbortError" ||
        isRequestConfigScopeChangedError(error)) {
      throw error
    }
    console.error("Error generating title")
    return fallBackTitle
  } finally {
    snapshot?.release()
  }
  ```

  Keep `DEFAULT_TITLE_GEN_PROMPT` as a compatibility export backed by `LEGACY_SERVICE_PROMPT_DEFAULTS["chat.title.generation"].user_template`; do not reintroduce sequential `.replace` rendering.

- [ ] **Step 9: Run all runtime tests and verify GREEN**

  Run:

  ```bash
  bunx vitest run \
    src/services/__tests__/title.service-prompt-scope.test.ts \
    src/services/__tests__/service-prompts.test.ts \
    src/services/tldw/domains/__tests__/service-prompts.test.ts \
    src/hooks/chat-helper/__tests__/saveMessageOnSuccess.scope.test.ts \
    src/hooks/chat-helper/__tests__/saveMessageOnError.test.ts \
    src/hooks/chat/__tests__/useChatActions.service-prompts.test.tsx
  ```

  Expected: all tests pass, covering normal persistence, error recovery, scoped success, and Compare title callers through the central function contract.

- [ ] **Step 10: Commit the runtime consumer**

  ```bash
  git add \
    apps/packages/ui/src/services/tldw/domains/service-prompts.ts \
    apps/packages/ui/src/services/tldw/domains/__tests__/service-prompts.test.ts \
    apps/packages/ui/src/services/tldw-server.ts \
    apps/packages/ui/src/services/service-prompts.ts \
    apps/packages/ui/src/services/__tests__/service-prompts.test.ts \
    apps/packages/ui/src/services/title.ts \
    apps/packages/ui/src/services/__tests__/title.service-prompt-scope.test.ts
  git commit -m "feat(service-prompts): customize automatic chat titles"
  ```

---

### Task 3: Expose Conversation title guidance in Workflow prompts

**Stage:** 3 of 4 — Settings experience

**Goal:** Make the definition discoverable regardless of enablement while directing users to the existing Chat setting instead of duplicating its toggle.

**Success Criteria:** Conversation title is localized, selectable, editable, and accompanied by an accessible `/settings/chat` link that appears without reading the title feature flag.

**Tests:** Real Settings component/router behavior plus generated locale mirror verification.

**Status:** Complete

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Settings/ServicePromptsSettings.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/settings.json`
- Generate: `apps/packages/ui/src/public/_locales/en/settings.json`

**Interfaces:**
- Consumes: server catalog item `chat.title.generation`, existing React Router `Link`, `getDefinitionText`, `getWorkflowLabel`, and Settings navigation guards.
- Produces: localized definition/workflow metadata and an editor-only Chat settings guidance link.

- [ ] **Step 1: Add the failing Settings behavior test**

  Add a complete title catalog item and detail fixture. Open it through the real list/editor UI and assert:

  ```typescript
  await openPrompt("Conversation title")
  expect(screen.getByRole("heading", { name: "Conversation title" })).toBeVisible()
  expect(screen.getByLabelText("User template")).toHaveValue(
    expect.stringContaining("{query}")
  )
  expect(screen.getByText("Automatic conversation titles")).toBeVisible()
  expect(screen.getByRole("link", { name: "Open Chat settings" }))
    .toHaveAttribute("href", "/settings/chat")
  ```

  The test setup must not mock or read `isTitleGenEnabled`; visibility comes only from the catalog.

- [ ] **Step 2: Run the Settings test and verify RED**

  Run:

  ```bash
  bunx vitest run \
    src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx
  ```

  Expected: FAIL because known title labels/workflow guidance and the Chat settings link do not exist.

- [ ] **Step 3: Implement the minimal localized Settings entry and guidance**

  Add known fallback entries:

  ```typescript
  "chat.title.generation": {
    key: "chatTitleGeneration",
    label: "Conversation title",
    description: "Controls the instruction used to generate automatic conversation titles."
  }
  ```

  ```typescript
  "chat.title.generation": {
    key: "automaticConversationTitles",
    label: "Automatic conversation titles"
  }
  ```

  Directly below the selected definition description, render only for the title ID:

  ```tsx
  {selectedDefinition.id === "chat.title.generation" ? (
    <p className="mt-2 text-sm text-text-muted">
      {t("servicePrompts.titleGeneration.note", {
        defaultValue: "Automatic title generation is enabled or disabled in Chat settings."
      })}{" "}
      <Link className="font-medium text-primary hover:underline" to="/settings/chat">
        {t("servicePrompts.titleGeneration.openChatSettings", {
          defaultValue: "Open Chat settings"
        })}
      </Link>
    </p>
  ) : null}
  ```

  Add the same English values under `servicePrompts.definitions`, `servicePrompts.workflows`, and `servicePrompts.titleGeneration` in the canonical asset locale.

- [ ] **Step 4: Generate and verify the public locale mirror**

  Run from `apps/extension`:

  ```bash
  bun run locales:sync settings.json
  bun run locales:sync --dry-run settings.json
  ```

  Expected: the first command updates `src/public/_locales/en/settings.json`; the dry run reports no pending write.

- [ ] **Step 5: Run the Settings and locale-sync tests and verify GREEN**

  Run:

  ```bash
  bunx vitest run \
    src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx
  ```

  From `apps/extension`, also run:

  ```bash
  bunx vitest run tests/unit/sync-public-locales.test.ts
  ```

  Expected: all tests pass.

- [ ] **Step 6: Commit the Settings experience**

  ```bash
  git add \
    apps/packages/ui/src/components/Option/Settings/ServicePromptsSettings.tsx \
    apps/packages/ui/src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx \
    apps/packages/ui/src/assets/locale/en/settings.json \
    apps/packages/ui/src/public/_locales/en/settings.json
  git commit -m "feat(settings): expose conversation title workflow prompt"
  ```

---

### Task 4: Verify, review, and finalize TASK-13111

**Stage:** 4 of 4 — Quality gates

**Goal:** Prove the bounded slice is compatible, secure, type-safe, and ready for integration.

**Success Criteria:** Focused and surrounding tests, compilation, lint, Bandit, and diff checks pass; review findings are addressed; Backlog records contain evidence and rationale.

**Tests:** Full changed-scope matrix plus build/security/static gates.

**Status:** In Progress

**Files:**
- Modify through MCP: `backlog/tasks/task-13111 - Expose-automatic-conversation-title-prompt-in-Service-Prompts.md`
- Remove after execution per repository guidance: `Docs/superpowers/plans/2026-08-23-conversation-title-service-prompt.md`

**Interfaces:**
- Consumes: Tasks 1–3 commits and `TASK-13111` acceptance criteria.
- Produces: reviewed commits with recorded verification evidence and no temporary plan file in the final tree.

- [ ] **Step 1: Run the complete focused frontend matrix**

  From `apps/packages/ui`:

  ```bash
  bunx vitest run \
    src/services/__tests__/title.service-prompt-scope.test.ts \
    src/services/__tests__/service-prompts.test.ts \
    src/services/tldw/domains/__tests__/service-prompts.test.ts \
    src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx \
    src/hooks/chat-helper/__tests__/saveMessageOnSuccess.scope.test.ts \
    src/hooks/chat-helper/__tests__/saveMessageOnError.test.ts \
    src/hooks/chat/__tests__/useChatActions.service-prompts.test.tsx
  ```

- [ ] **Step 2: Run backend regression and security gates**

  From repository root:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    tldw_Server_API/tests/Prompt_Management/test_service_prompts.py \
    tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py
  ../../.venv/bin/python -m bandit -r \
    tldw_Server_API/app/core/Prompt_Management/service_prompts.py \
    -f json -o /tmp/bandit_task_13111.json
  ```

  Expected: tests pass and Bandit reports no new findings.

- [ ] **Step 3: Run compile, type, lint, locale, and whitespace gates**

  From `apps/extension`:

  ```bash
  bun run compile
  bun run locales:sync --dry-run settings.json
  ```

  From `apps/tldw-frontend`:

  ```bash
  bun run typecheck
  ```

  From `apps/packages/ui`:

  ```bash
  bunx eslint \
    src/services/title.ts \
    src/services/service-prompts.ts \
    src/services/tldw-server.ts \
    src/services/tldw/domains/service-prompts.ts \
    src/components/Option/Settings/ServicePromptsSettings.tsx \
    src/services/__tests__/title.service-prompt-scope.test.ts \
    src/services/__tests__/service-prompts.test.ts \
    src/services/tldw/domains/__tests__/service-prompts.test.ts \
    src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx
  ```

  From repository root:

  ```bash
  git diff --check
  ```

- [ ] **Step 4: Self-review the exact diff**

  Check these mutations explicitly:

  - deleting the title registry entry fails backend/client registry tests;
  - replacing single-pass rendering with `.replace` fails inserted-brace/metasequence tests;
  - dropping expected-scope comparison allows catalog traffic and fails scope tests;
  - swallowing 412/AbortError fails title and persistence caller tests;
  - omitting `release` fails lifecycle tests;
  - reintroducing error interpolation leaks the sentinel and fails the content-free logging test;
  - hiding the Settings entry behind the feature flag fails the always-visible component test;
  - changing provider options fails model-invariant tests.

- [ ] **Step 5: Request code review and address only verified findings**

  Invoke `superpowers:requesting-code-review`, review against the approved design and `TASK-13111`, reproduce every actionable finding with a failing test, then apply the minimal fix and rerun the affected matrix.

- [ ] **Step 6: Finalize Backlog evidence through MCP**

  Record exact commands/counts, checked acceptance criteria, modified files, known warnings/skips, and a final summary explaining both what changed and why the central scope-bound snapshot approach was chosen.

- [ ] **Step 7: Remove this completed temporary plan and commit final metadata**

  Remove only `Docs/superpowers/plans/2026-08-23-conversation-title-service-prompt.md`, update `TASK-13111` to remove the now-historical documentation link while retaining its implementation-plan text and notes, then commit:

  ```bash
  git add backlog/tasks/task-13111\ -\ Expose-automatic-conversation-title-prompt-in-Service-Prompts.md \
    Docs/superpowers/plans/2026-08-23-conversation-title-service-prompt.md
  git commit -m "chore(backlog): finalize TASK-13111"
  ```

- [ ] **Step 8: Re-run final clean-tree gates**

  Run `git status --short`, `git diff --check HEAD^`, and the focused test commands affected by any review changes. Expected: only intentional committed changes, no test failures, and no whitespace errors.

### Task 4 pre-final-review verification update (2026-08-23)

- Completed the focused frontend matrix (7 files / 244 tests), focused backend matrix (78 tests), Bandit (0 findings across 363 LOC), extension compile, locale dry-run, and `git diff --check`.
- The required frontend typecheck is blocked by pre-existing diagnostics only under `apps/tldw-frontend/scripts/__tests__/skills-certification-*`; no `apps/tldw-frontend` file is in this branch's diff. The required UI-directory ESLint command is also environment-blocked because that package and its parents have no ESLint configuration; a supplemental invocation from the frontend config ignored all nine UI inputs as outside its base path.
- Per SDD preflight, the controller will provide the broad final review. Keep this plan and `TASK-13111` In Progress until that review package is generated; do not remove this plan or the Backlog documentation link yet.
