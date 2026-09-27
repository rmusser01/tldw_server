# Task 9 independent review — WebUI and extension recipe journeys

## Verdict

**NEEDS FIXES. Two Important test-contract findings remain in `325e705693..16e31bb372`; no Critical finding was found.**

I reviewed all 15 changed files against parent-plan Task 9, the approved recipe design (including sections 10.1–10.6, accessibility, and browser testing), the completed owner-contract recovery review, the parent progress ledger, and `task-9-report.md`. I traced both new Playwright specs through the production `/chat` adapters, recipe builder, local Dexie persistence, synchronization, direct/background transport, and the packaged extension routes.

## Findings

### Important 1 — The browser “reorder” step has no observable assertion and passes if reordering is broken

Location: `apps/tldw-frontend/e2e/workflows/single-text-recipes.spec.ts:361-367`.

The test disables Context, Constraints, and Output, verifies the Objective-only preview, then re-enables Output, edits it, clicks `Move Output up` once, and expects Objective followed by Output. Moving Output up once swaps it only with the disabled Constraints block, so the rendered order of the two enabled blocks is still Objective then Output. The asserted preview is therefore identical whether the move handler works or is a no-op. The component unit test proves its lower-level reducer/list behavior, but Task 9 specifically exists to prove this behavior through the real `/chat` browser surface; a wiring regression in the browser path would remain green.

Keep at least two enabled blocks and assert a changed exact preview or changed DOM block order before and after the keyboard-operable move. Preserve the exact XML/Markdown/free-form assertions and the toggle coverage.

### Important 2 — The packaged-extension payload check does not verify the required persisted definition/defaults

Locations: `apps/extension/tests/e2e/single-text-recipes.spec.ts:317-326`, `apps/extension/tests/e2e/single-text-recipes.spec.ts:448-479`.

The sidepanel journey authors `Extension saved default`, but `expectRuntimeExcluded` asserts only `prompt_format`, `prompt_schema_version`, and absence of runtime keys/sentinel. It never asserts that `prompt_definition` exists or that the authored variable `default_value`, block content, target role, and assembly format reached the request recorded by the fixture server. A packaged background/transport regression that strips just `default_value` while leaving a valid v2 definition would still satisfy every current assertion: the mock mirrors the reduced valid definition, the test supplies fresh runtime input after selecting the saved row, and update/clone continue without checking the lost default.

Task 9 explicitly requires the extension API payload to contain the definition/defaults and no runtime-value map. Deep-assert the create, update, and clone payloads at the fixture-server boundary, including `prompt_definition.schema_version`, `definition_kind`, target/render configuration, the authored block content where relevant, and the saved default while continuing to reject every runtime-value representation.

## Surface and behavior audit

- The Web spec opens the actual `/chat` route, invokes the production composer `PromptAssistComposerAction`, lazy-loads the real recipe builder, and records real browser requests through the production direct request path. Its system test enters the production `PromptSelect` editor and verifies system-only Apply/Undo without changing the selected template identity.
- The extension spec launches a production-built MV3 package. `openSidepanel("/chat")` exercises the real sidepanel route and background proxy; `/options.html#/chat` exercises the real full-chat extension route. The fixture server observes the actual background-dispatched recipe POST/PUT requests. The formal Quick Chat pop-out remains correctly documented as having no recipe adapter and is not cosmetically relabeled.
- Capability support is enabled only by the E2E fixtures. Production `single_text_recipe_v2.supported` remains `False` in `tldw_Server_API/app/api/v1/endpoints/prompts.py`.
- The all-false capability parser change is truthful: a syntactically valid capabilities document is known and unsupported, while malformed/404/transport failures still use unavailable/unknown handling. The backend contract always supplies the validated improvement limits required by this parser.
- Composer recipe Escape/Back now restores focus to the actual Improve trigger through the shared menu ref; Apply still returns focus through the existing target callback. The focused component and real browser assertions cover the intended distinction.
- The narrow drawer change is limited to `100vw` containment. Both browser suites verify its post-animation bounds and horizontal overflow at 390 px. No Improve-button placement/order/style change was made.
- The replaced `bg-surface1`/`bg-background` classes were invalid for the shared Tailwind configuration; the new `bg-surface`/`bg-bg` and text/surface action states use defined semantic tokens. Fresh two-theme axe runs passed in both the Web and packaged-extension journeys.
- Web coverage otherwise exercises every named starter plus Blank, variable input, block editing/toggling, exact XML/Markdown/free-form previews, user/system Apply with exact Undo, saved-source selection, update, clone, search, recipe facet/badge, runtime-value exclusion, known-v2 behavior, old/unknown/offline modes, keyboard focus, mobile containment, two themes, and exact-v3 non-mutation.
- Extension coverage otherwise exercises sidepanel save/select/update/clone/apply/undo, options-chat starter visibility and keyboard/mobile/two-theme axe behavior, sidepanel offline mixed v2/v3 behavior, and old/unknown server local Apply with zero mutations.
- The modernization of `prompts-ux.spec.ts` follows the current full-page Prompt Workspace editor and accessible More/Duplicate/Edit actions. Its create/tag/search/duplicate/edit/export/import flow passed against the packaged extension. No production Prompt Workspace behavior changed.
- The range contains no dependency/lockfile, backend, global chat-scope, owner-contract, capability-rollout, or unrelated shell/placement change. All 15 changed files were inspected.

## Fresh reviewer verification

All commands were run from the isolated worktree at `16e31bb372` before creating this artifact.

- Focused shared UI/transport gate: **5 files / 169 tests passed**, exit 0.
- Plan-required combined Web gate: **6 passed / 1 skipped**, exit 0. The skipped `prompts-chat` journey transparently requires a configured real backend; all six fixture-backed Task 9 `/chat` journeys ran.
- Plan-required combined packaged-extension gate: **6/6 passed**, exit 0, including sidepanel, `/options.html#/chat`, and the modernized Prompt Workspace journey.
- Extension `bun run compile`: exit 0.
- Repository ESLint 9 with the checked-in frontend config over every changed TypeScript/TSX file: exit 0, **0 errors / 2 pre-existing `no-explicit-any` warnings** in `prompts-api.ts`. A first `bunx eslint` attempt selected incompatible ESLint 10 and failed before linting; it made no worktree change and is not counted as verification.
- `git diff --check 325e705693..16e31bb372`: exit 0.
- Static range scan found no added `single_text_recipe_v2.supported=true`, unsafe HTML, TODO/FIXME, dependency/lockfile, backend, or chat-scope change.
- `git status --short --untracked-files=all` was empty before this ignored review artifact was added.

Bandit is not applicable to this Task 9 range because no Python source changed. The passing broad gates do not close the two missing assertions above: both are cases where the current tests can remain green while the advertised behavior regresses.

## Handoff

Return both findings to the Task 9 implementer under strict RED/GREEN. Re-run the 169-test focused gate, both combined Playwright gates, extension compile, scoped ESLint, and diff/static checks. Keep production capability support false and do not change the deferred Improve-button placement.
