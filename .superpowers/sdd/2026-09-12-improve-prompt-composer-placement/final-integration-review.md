# Final cumulative integration review — TASK-12984.2 / TASK-12984.3

## Verdict

**NEEDS FIXES. Four Important findings; no Critical finding.**

Reviewed branch `codex/single-text-prompt-recipes`, cumulative range
`9da94ebcb41ba45d279ab707ec92a285d31ba5c7..3cbcd92daf31f7bc044f17b4a748b5fc49edc095`.
The production/test tree remained at that HEAD throughout this review. This
artifact is the only repository change made by the final reviewer.

The prior scoped approvals at `a3e9391f6b` (Track B compensation micro-fix) and
`b77fabba33` (placement test correction) were read and respected. They do not
cover the new interleavings below, the broader shell-wiring regression, or the
capability dependency change relative to this review's earlier cumulative base.
Do not treat the whole branch as approved or the old maintained frontend gate
as green at current HEAD. No rebase or production fix was performed.

## Important 1 — same-owner pull releases an outstanding direct mutation's guard

Evidence:

- `apps/packages/ui/src/services/recipe-persistence-uncertainty.ts:203-209`
  reserves a direct mutation with `directRegistry.reserve(id, ownerId)` and no
  operation token/provisional phase.
- `apps/packages/ui/src/services/tldw/request-core.ts:514-524` installs that
  reservation immediately before fetch.
- `apps/packages/ui/src/services/recipe-persistence-registry.ts:81-94` accepts
  an existing same-owner scoped marker for reconciliation. It cannot tell a
  still-running direct mutation from a completed uncertain operation.
- `apps/packages/ui/src/services/prompt-sync.ts:549-587` acquires and commits
  that reconciliation during pull; registry line 111 deletes the scoped marker.

Reproduced sequence using the real sync -> Prompt Studio -> apiSend ->
request-core -> direct registry chain:

1. Start an owned PUT for linked local recipe `exact-id`; hold fetch unresolved.
   The authority reads `scoped`.
2. Pull server prompt 101 into that local ID with the same owner. Its GET returns
   the old remote row while the PUT is still outstanding.
3. Pull returns `success:true`, publishes `synced`, and clears the PUT's marker.
4. A second push dispatches PUT number 2 before PUT number 1 settles.

The review's fetch substitute accepts both requests; the demonstrated defect is
the second real transport dispatch, independent of how a particular server
ultimately resolves the competing writes. This permits concurrent duplicate
mutation attempts and can resend stale data copied by the intervening pull.
No restart, owner change, malformed response, or storage failure is required.
The extension's provisional receipt blocks this particular pre-response order;
the direct WebUI path does not.

The probe substitutes only browser/config/runtime-key/storage boundaries,
Dexie tables/helpers, and fetch, not the production ownership/sync/transport
chain. Both the persistence reviewer and final reviewer reproduced exit 1:

```text
first PUT outstanding; authority: scoped
pull completed before first PUT: true synced authority: clear
second push before first PUT settled: true PUT count: 2
AssertionError: same-ID pull must not admit another write while the original is outstanding
2 !== 1
```

Exact command: `sh /tmp/recipe-active-write-pull-probe.sh`. The complete probe
is preserved in Appendix A rather than relying only on that temporary file.

The maintained test at `prompt-sync.uncertainty.test.ts:632` starts with an
already-error row and attempts a **new** reservation during the final local
commit. It does not hold an **existing** mutation open before pull starts.

Required correction: distinguish active dispatch ownership from reconcilable
uncertainty, and retain the active operation's guard through transport/local
settlement. Pull must not consume another still-running operation's reservation.
Add this pre-existing-active-write ordering alongside the already-covered
post-reconciliation reservation and compensation orderings, retaining
direct/background parity and exact-owner recovery.

## Important 2 — system recipe Undo survives a subsequent improvement

`apps/packages/ui/src/components/Common/PromptSelect.tsx:476-482` starts Improve
now / Review changes without clearing `recipeUndo`. In contrast, the composer
start handler explicitly clears its recipe Undo. The system footer at
`PromptSelect.tsx:922-925` continues offering the stale recipe Undo when the
later improvement is applied. `undoRecipe` at lines 517-530 changes the draft
and override without invalidating the still-active improvement state.

A real-component probe performed this sequence:

1. Open the system editor with `Original system override`.
2. Apply the Clear task recipe with a synthetic runtime Task value.
3. Improve now, using a mocked valid response that appends `\nRefined.` to the
   captured recipe text.
4. Observe both **Undo recipe** and **Undo improvement**.
5. Activate Undo recipe. The editor and override revert to the pre-recipe
   original, yet Undo improvement remains enabled for a different snapshot.

The two expected invalidation assertions failed in one deterministic test:
**1 failed / 68 unselected**, exit 1. All positive recipe Apply, improvement
Apply, and exact original-restoration assertions preceding/between those
failures passed. The probe used the maintained component harness and real
PromptSelect/builder/controller; only an in-memory test transform was added.
No repository source or test was edited. The temporary transform is
`/tmp/prompt-recipe-final-review.vitest.config.mts`; its test body is preserved
in Appendix B.

This violates the design's one-step Undo lifetime (a newer improvement consumes
the old Undo), creates WebUI/extension system-versus-composer inconsistency, and
lets an obsolete action overwrite the latest applied draft. Required correction:
invalidate recipe Undo when the system editor starts a new improvement, and
keep every programmatic draft replacement consistent with the shared
improvement controller. Cover recipe -> Improve now and recipe -> Review changes
for success, cancel/failure, and exact override/template identity behavior.

## Important 3 — capability rate limiting regressed within the cumulative branch

`tldw_Server_API/app/api/v1/endpoints/prompts.py:743-748` runs the decorator's
`rbac_rate_limit("prompts.capabilities")` before the endpoint parameter's
`get_auth_principal`. The limiter at
`tldw_Server_API/app/api/v1/API_Deps/auth_deps.py:2202-2204` returns early when
`request.state.user_id` is absent. Configured catalog/user/role limits are
therefore silently bypassed on this authenticated endpoint.

Fresh existing capability tests: **1 failed / 6 passed / 60 deselected**.
`test_prompt_capabilities_catalog_rate_limit_returns_429` receives 200 rather
than 429 on request 2. The final reviewer independently ran the existing test
against both in-memory dependency orders:

```text
current-order: FAIL ()
base-equivalent-auth-first-order: PASS (second response 429)
```

The earlier reports correctly found this pre-existing relative to the narrow
Task 10 boolean flip, but incorrectly generalized that baseline to the entire
feature. At cumulative base `9da94ebcb4`, the decorator explicitly lists
`Depends(get_auth_principal)` before the rate dependency. Commit
`5f55f7aa4bdf2cf4cd790b1888c9bf7b8a1a73b8` removes it while adding the principal
parameter for persistence-authorization metadata. `auth_deps.py` is identical
at cumulative base and HEAD (blob
`77710d604520e3de3753172b682963dd23ed9962`), and the existing failing test body
is unchanged in that range.

Required correction: restore authentication-before-rate-limit dependency
ordering while retaining the principal needed by the capability response. Run
the existing capability/auth matrix without excluding this test. This finding
does not claim a bypass of recipe write authorization or tenant ownership.

## Important 4 — placement leaves three maintained shell-wiring tests red

`apps/packages/ui/src/components/Chat/composer/__tests__/PromptAssistComposerAction.shell-wiring.test.ts`
still requires the removed topology:

- Line 57 expects `promptAssistComposer={{` in PlaygroundForm; actual count 0.
- Line 83 requires PromptAssistComposerAction to be defined inline inside the
  old SidepanelComposerControlArea wrapping the pro-mode branch.
- Line 112 expects ComposerToolbar to own the action and line 114 forbids the
  direct PlaygroundForm action that the approved placement intentionally adds.

Fresh seven-file Track A/state/diff/shell run: **3 failed / 65 passed**, with
all three failures in this shell file and six adjacent files passing. These are
stale test contracts, not evidence that the approved new mounting is wrong.
However, this file is explicitly part of Track B's recorded authoritative
42-file gate and was absent from placement's four-file gate, so the cumulative
feature matrix is no longer green.

Required correction: update the shell assertions to the approved actual send
clusters while preserving checks for one adapter instance, current draft and
revision, exact send attempt, model/provider, backend/authorization handoff,
focus, and Quick Chat exclusion. Do not delete or skip the ownership proof.
Rerun the cumulative gate including this file.

## Integration coverage and scope audit

Read AGENTS.md; the full approved design; TASK-12984.2 and TASK-12984.3 (official
Backlog CLI fallback plus task records); both progress ledgers; Track B's
release/final-fix reports; the complete placement task report; prior final
reviews/re-reviews; and the final Track B, ownership-recovery, browser-payload,
capability-release, and placement approvals. Backend and persistence checks ran
as explicitly authorized independent review subtasks; the final reviewer
inspected their code evidence and reran both new isolation probes.

No additional Critical/Important finding was established in schema-v2
discrimination and strict validation, bounded deterministic renderers and
variables, authored snapshots/runtime-value exclusion, v1 compatibility,
legacy-route rejection, JSON interop, Prompt Studio version/preview/execution
boundaries, MCP projection, immutable recipe editing, target-specific local
Apply, expected/actual owner identity, credential-revision authorization,
extension no-ambiguous-fallback, unknown-owner quarantine/Forget, or the reviewed
release-compensation ordering.

Inspected current production placement, not only approval text: one shared
action is supplied to each chosen sidepanel variant; Playground owns the
external Send cluster action; Quick Chat remains excluded. The menu uses a body
portal with paired outside-click/focus/resize/capture-scroll cleanup. Feedback
yields to menu and logical/presented drawer ownership, uses collision-aware
visual-viewport placement, and retains Undo state while intentionally hidden.
There is no new dependency/lockfile change, unsafe HTML rendering, or duplicate
action found in these paths. The transient-overlay code remains bounded to
its composer and a single scheduled animation frame.

Improvement request construction still includes only the chosen adapter draft,
active route, operation ID, and draft-derived protected tokens; counterpart
system/user text, sent conversation history, attachments, and RAG context are
not added by this integration. No server-side idempotency ledger was demanded:
its absence is an explicit design non-goal, unlike finding 1's broken existing
client-side active-operation guard.

## Fresh verification

All commands below ran on the reviewed HEAD before writing this artifact.
Positive results do not invalidate the four failing proofs above.

| Gate | Result |
| --- | --- |
| Backend schema/render/DB/API/interop/Studio/MCP/property matrix | 1,127 passed, 3 warnings, exit 0 |
| Existing capability/auth subset | 1 failed, 6 passed, 60 deselected; finding 3 |
| Persistence/transport matrix | 15 files / 1,577 passed, exit 0 |
| Adjacent runtime-auth/owner matrix | 6 files / 70 passed, exit 0 |
| Composer/menu/system/editor/builder/state matrix | 10 files / 340 passed, exit 0 |
| Track A controller/panel/diff/localization plus shell | 3 failed, 65 passed; finding 4 |
| New system Undo integration probe | 1 failed, 68 unselected; finding 2 |
| Real-chain active-write/pull probe | Repeated exit 1, second PUT observed; finding 1 |
| Extension `bun run compile` | Exit 0 |
| Cumulative and worktree `git diff --check` | Exit 0 |

Exact test commands are preserved in Appendix C. No browser was launched in
this final review: the complete Web/package suites and five final overlay
regressions are prior scoped evidence, not claimed as fresh here. With four
deterministic blockers, rerunning those unchanged browser cases would not
establish approval. The correction gate should rerun them together with the
broader cumulative tests.

Inclusive production Bandit ran over all changed Python production files:
16,092 lines, `errors=[]`, no medium/high findings and no finding on changed
lines. It exits 1 solely for 23 LOW B311 findings in unchanged
PromptStudioDatabase random-jitter calls, matching the recorded baseline;
40 existing suppressed checks remain. Result:
`/tmp/bandit_recipe_final_backend_review.json`. Do not describe the inclusive
scan as zero findings. This Markdown-only review adds no Bandit target.

Known limitations retained: the prior full Web build compiled before its
documented 683 KB / 600 KB app-shell budget failure; unrelated full frontend
typecheck diagnostics were not revalidated or relabeled as passing; live-backend
configured browser coverage and manual screen-reader UAT were not run here.
No new PostgreSQL-specific verification was performed. The capability rate
failure is explicitly **removed from the cumulative baseline bucket** by
finding 3.

## Worktree safety and handoff

The assigned worktree was clean before review and after all probes/test runs.
Only this artifact is to be committed. The main checkout has no tracked
changes and retains exactly its two pre-existing untracked TTS files:

- `Docs/superpowers/specs/2026-09-10-tts-gateway-metadata-preferences-design.md`
- `backlog/tasks/task-13140.1 - Design-TTS-gateway-voice-discovery-pricing-and-preference-history.md`

No production/tests, Backlog records, user files, manifests, or lockfiles were
edited. Temporary diagnostics are confined to `/tmp`; in-memory transforms were
not saved into repository files. No rebase, PR mutation, or external message
was performed. The controller owns task-status corrections and any authorized
follow-up implementation; this review does not assume another fix wave.

## Appendix A — active-write/pull probe

Run from `apps/packages/ui` with `bun --eval` and the following source (the same
source is in `/tmp/recipe-active-write-pull-probe.sh`):

```javascript
import { mock } from "bun:test";
import assert from "node:assert/strict";
const rows = new Map();
const config = { serverUrl: "https://recipe-review.test", authMode: "single-user", authSource: "manual", apiKey: "review-fake-key" };
mock.module("wxt/browser", () => ({ browser: { runtime: {} } }));
mock.module("@/utils/safe-storage", () => ({ createSafeStorage: () => ({}) }));
mock.module("@/services/tldw/direct-browser-config", () => ({ resolveDirectBrowserConfig: async () => config }));
mock.module("@/services/tldw/runtime-auth-override", () => ({ getRuntimeSingleUserApiKeyOverride: () => "" }));
mock.module("@/services/prompt-studio-settings", () => ({ getPromptStudioDefaults: async () => ({ defaultProjectId: 42 }), setPromptStudioDefaults: async () => {} }));
mock.module("@/db/dexie/helpers", () => ({ generateID: () => "new-id" }));
mock.module("@/db/dexie/chat", () => ({ PageAssistDatabase: class {} }));
mock.module("@/db/dexie/schema", () => ({ db: {
  transaction: async (...args) => args.at(-1)(),
  prompts: {
    get: async id => rows.get(id),
    update: async (id, fields) => { if (!rows.has(id)) return 0; const row = {...rows.get(id)}; if (typeof fields === "function") { if (fields(row) === false) return 1; } else Object.assign(row, fields); rows.set(id, row); return 1; },
    add: async row => rows.set(row.id, row),
    where: field => ({equals: value => ({first: async () => [...rows.values()].find(row => row[field] === value)})})
  }
} }));
const { CLEAR_TASK_RECIPE } = await import("./src/components/Common/PromptAssist/recipes/built-in-recipes.ts");
const sync = await import("./src/services/prompt-sync.ts");
const authority = await import("./src/services/recipe-persistence-uncertainty.ts");
const owner = await authority.resolveRecipePersistenceOwnerView();
assert.ok(owner);
const oldServer = { id: 101, project_id: 42, name: "Old remote recipe", version_number: 1, updated_at: "2026-09-10T00:00:00Z", prompt_format: "structured", prompt_schema_version: 2, prompt_definition: CLEAR_TASK_RECIPE.definition };
rows.set("exact-id", { id: "exact-id", title: "New local recipe", name: "New local recipe", syncStatus: "synced", promptFormat: "structured", promptSchemaVersion: 2, structuredPromptDefinition: CLEAR_TASK_RECIPE.definition, studioProjectId: 42, serverId: 101 });
let putCount = 0, releaseFirst, signalStarted;
const started = new Promise(resolve => { signalStarted = resolve; });
const held = new Promise(resolve => { releaseFirst = resolve; });
const response = data => new Response(JSON.stringify({success: true, data}), {status: 200, headers: {"content-type": "application/json"}});
globalThis.fetch = async (url, init) => {
  if (init.method === "PUT") {
    putCount++;
    if (putCount === 1) { signalStarted(); await held; }
    return response({...oldServer, name: "Updated remote recipe", version_number: putCount + 1});
  }
  if (init.method === "GET") return response(oldServer);
  throw new Error("Unexpected network call");
};
const first = sync.pushToStudio("exact-id", 42, { expectedOwnerId: owner.ownerId });
await started;
console.log("first PUT outstanding; authority:", await authority.readRecipePersistenceUncertainty("exact-id", owner.ownerId));
const pull = await sync.pullFromStudio(101, "exact-id");
console.log("pull completed before first PUT:", pull.success, pull.syncStatus, "authority:", await authority.readRecipePersistenceUncertainty("exact-id", owner.ownerId));
const second = await sync.pushToStudio("exact-id", 42, { expectedOwnerId: owner.ownerId });
console.log("second push before first PUT settled:", second.success, "PUT count:", putCount);
releaseFirst();
await first;
assert.equal(putCount, 1, "same-ID pull must not admit another write while the original is outstanding");
```

## Appendix B — system Undo in-memory test

The temporary Vitest config imports the checked-in UI config and uses an
`enforce: "pre"` transform only for
`PromptSelect.system-prompt-modal.test.tsx`, inserting this test immediately
before its first `it("keeps local recipe work usable...` within the existing
describe. Its existing beforeEach/platform mocks remain intact.

```typescript
it('final integration probe invalidates recipe Undo when a later improvement starts', async () => {
  const user = userEvent.setup()
  const original = 'Original system override'
  const view = renderPromptSelect({ systemPrompt: original })
  await openEditor(user, original)
  await user.click(screen.getByRole('button', { name: 'Improve prompt' }))
  await user.click(screen.getByRole('button', { name: /Build from recipe/ }))
  await user.type(await screen.findByLabelText('Current value for Task (not saved)'), 'Synthetic review task')
  const compiled = (screen.getByLabelText('Compiled prompt preview') as HTMLTextAreaElement).value
  await user.click(screen.getByRole('button', { name: 'Apply to system prompt' }))
  expect(screen.getByLabelText('Enter system prompt')).toHaveValue(compiled)
  expect(screen.getByRole('button', { name: 'Undo recipe' })).toBeInTheDocument()
  mocks.improvePrompt.mockImplementation(async request => improvementResponse(request.operation_id, request.text + '\nRefined.'))
  await applyImprovementNow(user)
  expect(screen.getByLabelText('Enter system prompt')).toHaveValue(compiled + '\nRefined.')
  expect.soft(screen.queryByRole('button', { name: 'Undo recipe' })).not.toBeInTheDocument()
  const staleUndo = screen.queryByRole('button', { name: 'Undo recipe' })
  if (staleUndo) {
    await user.click(staleUndo)
    expect(screen.getByLabelText('Enter system prompt')).toHaveValue(original)
    expect(view.props.setSystemPrompt).toHaveBeenLastCalledWith(original)
    expect.soft(screen.queryByRole('button', { name: 'Undo improvement' })).not.toBeInTheDocument()
  }
})
```

Exact invocation from `apps/packages/ui`:

```sh
../../tldw-frontend/node_modules/.bin/vitest run src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx --config /tmp/prompt-recipe-final-review.vitest.config.mts -t 'final integration probe' --maxWorkers=1 --no-file-parallelism --reporter=verbose
```

## Appendix C — exact fresh maintained commands

From repository root (activate the existing project venv before Python):

```sh
source .venv/bin/activate && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Prompt_Management/test_single_text_recipe_validator.py tldw_Server_API/tests/Prompt_Management/test_single_text_recipe_renderer.py tldw_Server_API/tests/Prompt_Management/test_v1_transport_parity.py tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py tldw_Server_API/tests/Prompt_Management/test_prompts_interop.py tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompts_structured_api.py tldw_Server_API/tests/Prompt_Management_NEW/integration/test_recipe_persistence_boundaries.py tldw_Server_API/tests/Prompt_Management_NEW/integration/test_structured_prompt_search.py tldw_Server_API/tests/Prompt_Management_NEW/property/test_prompt_properties.py tldw_Server_API/tests/prompt_studio/test_structured_prompt_execution.py tldw_Server_API/tests/prompt_studio/test_structured_prompt_preview_parity.py tldw_Server_API/tests/prompt_studio/test_structured_prompt_route_boundaries.py tldw_Server_API/tests/prompt_studio/test_structured_prompt_versions.py tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py
source .venv/bin/activate && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompt_improvement_api.py -k capabilities
source .venv/bin/activate && python -m bandit tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_prompts.py tldw_Server_API/app/api/v1/endpoints/prompts.py tldw_Server_API/app/api/v1/schemas/prompt_schemas.py tldw_Server_API/app/api/v1/schemas/prompt_studio_project.py tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py tldw_Server_API/app/core/DB_Management/Prompts_DB.py tldw_Server_API/app/core/DB_Management/prompts_db_helpers.py tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_catalog.py tldw_Server_API/app/core/Prompt_Management/Prompts_Interop.py tldw_Server_API/app/core/Prompt_Management/prompt_studio/prompt_executor.py tldw_Server_API/app/core/Prompt_Management/structured_prompts/__init__.py tldw_Server_API/app/core/Prompt_Management/structured_prompts/assembler.py tldw_Server_API/app/core/Prompt_Management/structured_prompts/legacy_renderer.py tldw_Server_API/app/core/Prompt_Management/structured_prompts/models.py tldw_Server_API/app/core/Prompt_Management/structured_prompts/single_text_renderer.py tldw_Server_API/app/core/Prompt_Management/structured_prompts/validator.py -f json -o /tmp/bandit_recipe_final_backend_review.json
```

Backend random seeds: 2224969893 for the 1,127-test matrix; 86236913 for the
capability subset. Backend warnings were existing pytest/deprecation cleanup
warnings. The auth-order probe ran the existing failing test with
`pytest.MonkeyPatch.context()` twice, inserting
`Depends(tests.get_auth_principal)` at index 0 of the in-memory capability
route's `dependencies` only for the second run, then removing it in `finally`.

From `apps/packages/ui`, using repository-pinned Vitest 4.0.18:

```sh
./node_modules/.bin/vitest run src/services/__tests__/recipe-persistence-owner-contract.test.ts src/services/__tests__/recipe-request-snapshot.test.ts src/services/__tests__/recipe-persistence-registry.test.ts src/services/__tests__/recipe-persistence-authority.test.ts src/services/__tests__/recipe-persistence-owner.contract.test.ts src/services/__tests__/request-core.persistence-scope.test.ts src/services/__tests__/api-send.test.ts src/services/__tests__/prompt-sync.structured-prompts.test.ts src/services/__tests__/prompt-sync.auto-sync.test.ts src/services/__tests__/prompt-sync.uncertainty.test.ts src/services/__tests__/structured-prompt-transport.test.ts src/services/__tests__/prompt-studio.recipe-policy.test.ts src/db/dexie/__tests__/firefox-prompt-write-order.test.ts src/db/dexie/__tests__/prompt-rollback.test.ts src/entries/__tests__/background.recipe-persistence-owner.test.ts --sequence.shuffle --sequence.seed=12984 --reporter=dot
./node_modules/.bin/vitest run src/services/__tests__/recipe-persistence-uncertainty.test.ts src/services/tldw/__tests__/request-core.quickstart.test.ts src/services/tldw/__tests__/request-core.hosted.test.ts src/services/__tests__/tldw-auth.refresh-rotation.test.ts src/entries/__tests__/background.effective-auth.test.ts src/hooks/__tests__/useRecipePersistenceOwner.test.tsx --sequence.shuffle --sequence.seed=12984 --reporter=dot
../../tldw-frontend/node_modules/.bin/vitest run src/components/Common/PromptAssist/__tests__/PromptAssistMenu.test.tsx src/components/Chat/composer/__tests__/PromptAssistComposerAction.test.tsx src/components/Sidepanel/Chat/__tests__/SidepanelComposerControlArea.prompt-assist.test.tsx src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.test.tsx src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.dispatch.test.tsx src/components/Common/PromptAssist/recipes/__tests__/SingleFieldRecipeEditor.test.tsx src/components/Common/PromptAssist/recipes/__tests__/recipe-editor-state.test.ts src/components/Common/PromptAssist/recipes/__tests__/built-in-recipes.test.ts --maxWorkers=1 --no-file-parallelism --reporter=dot
../../tldw-frontend/node_modules/.bin/vitest run src/components/Common/PromptAssist/__tests__/PromptAssistPanel.test.tsx src/components/Common/PromptAssist/__tests__/prompt-assist-state.test.ts src/components/Common/PromptAssist/__tests__/PromptDiff.bounded.test.tsx src/components/Common/PromptAssist/__tests__/PromptReviewSurface.test.tsx src/components/Common/PromptAssist/__tests__/PromptAssistLocalization.test.tsx src/components/Common/PromptAssist/__tests__/usePromptAssist.test.tsx src/components/Chat/composer/__tests__/PromptAssistComposerAction.shell-wiring.test.ts --maxWorkers=1 --no-file-parallelism --reporter=dot
```

Node emitted its existing experimental localStorage warning. No test was
disabled or changed. Extension compile ran `bun run compile` from
`apps/extension`. Hygiene used
`git diff --check 9da94ebcb41ba45d279ab707ec92a285d31ba5c7..3cbcd92daf`,
`git diff --check`, and `git status --short --untracked-files=all`.
