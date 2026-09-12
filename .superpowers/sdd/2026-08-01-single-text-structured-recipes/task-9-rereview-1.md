# Task 9 fix-round-1 independent re-review

## Verdict

**NEEDS FIXES. One Important test-contract finding remains in `abd6c8b335..0bfed68f8e`; no Critical or Minor finding was found.**

The Web reorder fix is effective. The extension request assertions now check the authored Objective and saved default at the actual fixture-server request boundary, but they still do not deep-assert the full authored v2 definition required by Task 9.

## Finding

### Important — Extension create/update/clone can lose three authored blocks without failing the journey

Locations: `apps/extension/tests/e2e/single-text-recipes.spec.ts:317-354`, with create/update/clone call sites at `apps/extension/tests/e2e/single-text-recipes.spec.ts:496-514`.

`expectPersistedRecipe` uses `toMatchObject`, `objectContaining`, and `arrayContaining` and checks only the Objective block. It does not require the Clear task recipe's Context / inputs, Constraints, or Output blocks, nor exact block count/order or the full authored block metadata/content. It also only partially checks variable metadata. Consequently, the packaged-extension journey can remain green if the create, update, and clone request payloads silently lose three quarters of the authored recipe definition.

I proved this with a reviewer-only mutation: immediately after parsing each HTTP request in the fixture server, I replaced `prompt_definition.blocks` with `blocks.slice(0, 1)` before the request was recorded and echoed. The focused real packaged-sidepanel test still passed **1/1**. I then reverted the mutation and confirmed both changed E2E specs exactly matched `0bfed68f8e`. This is not a response-echo-only concern: the assertions read `mock.creates()` / `mock.updates()` entries captured at lines 128-132 before the server response is produced. The request-boundary check is correctly placed; its expected object is incomplete.

Deep-assert the full, target-retargeted Clear task v2 definition for every create, update, and clone request, including:

- schema, definition kind, and exact assembly configuration;
- the complete Task variable (including saved default and remaining metadata);
- all four blocks in exact order with IDs, names, section keys, `role: "user"`, kinds, exact content, enabled/template flags, and numeric order;
- original Objective content on create and `Extension update: {{task}}` on update and clone;
- continued exclusion of the runtime sentinel and every runtime-value-map representation.

Use exact/deep equality (or an equivalent exact block-array comparison) rather than partial containment, so truncation, omission, reordering, or metadata loss cannot pass.

## Closed prior finding — Web reorder now proves a changed enabled-block order

The Web `/chat` journey now re-enables Constraints and Output, verifies the exact pre-move order Objective → Constraints → Output, invokes `Move Output up` by keyboard, and verifies the exact post-move order Objective → Output → Constraints at `apps/tldw-frontend/e2e/workflows/single-text-recipes.spec.ts:361-372`.

Reviewer-only mutation proof: I replaced the Enter press with a no-op enabled-state assertion. The focused real `/chat` Playwright test failed **0/1** with the expected exact diff: the received preview remained Objective → Constraints → Output instead of Objective → Output → Constraints. After restoring the Enter press, the combined Web gate passed. The original Important reorder finding is closed.

## Create/update/clone semantic audit

- Create correctly expects the built-in Objective after target-role retargeting: `Complete this task:\n\n{{task}}`.
- Update correctly expects the user-authored Objective: `Extension update: {{task}}`.
- Clone is implemented as a second create and correctly expects the updated Objective, preserving the edited copy semantics.
- `mock.creates()` and `mock.updates()` filter parsed request records captured before `serverPrompt(...)` creates the fixture response. The fields currently asserted are therefore genuine outbound-request evidence, not assertions satisfied solely by fixture echo.
- The remaining defect is coverage breadth: the helper proves one block and part of one variable, not the full authored definition/default/config/content contract.

## Scope audit

The fix range changes only the two Task 9 E2E specs and the existing Backlog task record. There are no production, dependency/lockfile, backend, global-scope, capability-rollout, Improve-button placement, or layout changes. Production capability support remains false at `tldw_Server_API/app/api/v1/endpoints/prompts.py:759`.

## Fresh verification

All restored positive commands ran from the isolated worktree at `0bfed68f8e` before this artifact was added.

- Combined Web gate: **6 passed / 1 skipped**, exit 0. The one skip is the transparently backend-dependent Prompt → Chat journey; all six fixture-backed real `/chat` recipe journeys ran.
- Combined packaged-extension gate: **6/6 passed**, exit 0, including the real sidepanel `/chat`, `/options.html#/chat`, and current Prompt Workspace journey.
- Extension `bun run compile`: exit 0.
- Scoped ESLint over both changed E2E specs: exit 0, no lint findings (only the repository config's Pages-directory informational message).
- `git diff --check abd6c8b335..0bfed68f8e`: exit 0.
- The restored worktree was clean before this ignored review artifact was created.
- Web no-op mutation: focused test **failed 0/1 as required**.
- Extension three-block-truncation mutation: focused packaged-sidepanel test **passed 1/1**, demonstrating the remaining assertion hole.

Bandit is not applicable because this fix range contains no Python changes.

## Handoff

Return the remaining finding to the Task 9 implementer under strict RED/GREEN. The production implementation does not need to change. Strengthen only the packaged-extension HTTP-boundary contract, then rerun the focused mutation proof, combined extension gate, compile, scoped lint, and diff hygiene.
