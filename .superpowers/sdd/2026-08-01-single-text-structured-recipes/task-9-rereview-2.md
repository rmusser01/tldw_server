# Task 9 fix-round-2 independent re-review

## Verdict

**APPROVED. No Critical, Important, or Minor findings remain in `ba30c1d95f..2a1b9bde0c`.**

The prior extension payload-contract finding is closed. The fix uses exact deep equality against the complete, ordered, user-retargeted Clear task v2 definition at the create, update, and clone HTTP request boundaries.

## Contract audit

- `expectPersistedClearTaskRecipe` at `apps/extension/tests/e2e/single-text-recipes.spec.ts:317-402` requires exact equality for the whole `prompt_definition`, not partial containment.
- The expected assembly configuration includes `single_text`, `target_role: "user"`, XML rendering, and the exact separator.
- The complete Task variable is asserted, including description, required flag, authored default `Extension saved default`, input type, options, and maximum length.
- All four blocks are required in exact order: Objective, Context / inputs, Constraints, and Output. Every ID, name, section key, user role, kind, exact content, enabled flag, numeric order, and template flag is checked.
- Create at lines 539-542 expects the original built-in Objective `Complete this task:\n\n{{task}}`.
- Update at lines 554-557 and clone at lines 560-563 expect only the intended Objective edit, `Extension update: {{task}}`. Because every other field remains fixed in the same exact expected object, unintended changes elsewhere fail the test.
- Runtime exclusion remains whole-request scoped at lines 398-401: the runtime sentinel and known runtime-value-map key spellings are rejected.
- Parsed outbound request bodies are recorded at lines 128-132 before the fixture constructs any response. `mock.creates()` and `mock.updates()` therefore prove the actual POST/PUT boundary rather than a value satisfied only by fixture echo.

## Mutation proof

I temporarily truncated every parsed outbound recipe definition to `blocks.slice(0, 1)` before the fixture recorded and echoed it. The focused packaged-sidepanel journey failed **0/1** at the exact deep-equality assertion, with Context / inputs, Constraints, and Output reported missing. This proves the previous false-green path is closed.

I reverted the mutation with `apply_patch` and confirmed the extension spec exactly matched `2a1b9bde0c` before running the restored gates. The restored focused journey then passed **1/1**.

## Scope audit

The fix range changes only:

- `apps/extension/tests/e2e/single-text-recipes.spec.ts`;
- the existing Backlog task record.

There are no production, dependency/lockfile, backend, capability, global-scope, layout, or Improve-button placement changes. The Web recipe spec is byte-for-byte untouched in this range, so its previously approved real `/chat` reorder proof remains valid. Production recipe capability support remains false at `tldw_Server_API/app/api/v1/endpoints/prompts.py:759`.

## Fresh verification

All positive commands ran against the restored worktree at `2a1b9bde0c` before this artifact was added.

- Focused packaged-sidepanel recipe journey: **1/1 passed**, exit 0.
- Combined packaged-extension gate: **6/6 passed**, exit 0, covering sidepanel `/chat`, `/options.html#/chat`, compatibility modes, mobile/two-theme accessibility, and the current Prompt Workspace journey.
- Extension `bun run compile`: exit 0.
- Scoped ESLint over the changed extension spec: exit 0, no findings; only the repository config's Pages-directory informational message was printed.
- `git diff --check ba30c1d95f..2a1b9bde0c`: exit 0.
- `git diff --exit-code ba30c1d95f..2a1b9bde0c -- apps/tldw-frontend/e2e/workflows/single-text-recipes.spec.ts`: exit 0.
- Restored worktree status was clean before this ignored artifact was created.

Bandit is not applicable because the fix range contains no Python changes.

## Independent cross-check

A separate reviewer reported no findings and independently confirmed the request-capture boundary, exact full definition/default/config/block coverage, create/update/clone semantics, runtime exclusion, and tests-only scope.
