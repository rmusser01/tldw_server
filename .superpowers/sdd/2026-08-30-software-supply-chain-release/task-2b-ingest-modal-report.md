# Task 2B report — ingest modal lifecycle

## Status

**DONE_WITH_CONCERNS — Task 2B's modal boundary is complete.** The terminal Quick Ingest dialog was confirmed open at the shared helper's return boundary. The helper now completes the already-supported wizard close lifecycle before returning, and the focused owning regression is green. Both affected live journeys passed their former post-ingest navigation/click boundaries. The combined two-journey command is intentionally not fully green only because Search → Chat reaches a distinct, retained deterministic-mock-content failure after it reaches chat.

## Root-cause trace and minimal repair

1. Both affected journeys call `ingestAndWaitForReady()` before their next page action.
2. The helper waited for the terminal Results UI and returned a media id without calling the existing `dismissQuickIngest()` lifecycle.
3. `QuickIngestWizardModal` keeps the Ant dialog open while `open && !state.isMinimized`; the route-level composer owns `ingestOpen`.
4. Its supported `onClose` calls `hideQuickIngestSession()` and `setIngestOpen(false)`. `dismissQuickIngest()` already uses that Done/close control (and only the supported minimize confirmation when relevant), then asserts the dialog is hidden.

The repair is deliberately one lifecycle call in `ingestAndWaitForReady()`: after terminal state and media-id resolution, it awaits `dismissQuickIngest(page)` before returning. It does not remove modal DOM, force a click, add a sleep, alter a timeout, or change the component owner.

## Behavior-level TDD evidence

The focused regression in `ingest-evaluate-review.spec.ts` runs a real URL ingest through the Task 2A owned graph, attaches the helper-return modal state, requires the Quick Ingest dialog to be hidden, and only then navigates and clicks the Synthetic Review tab.

### RED

The graph used Redis `62457`, mock provider `18091`, API `62458`, and WebUI `62459`; it used the repository-root virtual environment and the committed `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/local-success.json` fixture. The focused command was:

```bash
# The preceding one-shell graph exports the documented endpoint variables.
bunx playwright test e2e/workflows/journeys/ingest-evaluate-review.spec.ts --project=journeys --workers=1 --reporter=line --grep 'closes terminal ingest'
```

Before the helper change, this test reached Quick Ingest's terminal Results step and failed at `expect(dialog).toBeHidden()` after the existing 30-second assertion window. Playwright resolved the visible `role=dialog` Quick Ingest modal 34 times. The terminal UI showed `0 succeeded, 1 failed` for the live URL extraction and its supported `Close the ingest wizard` / Done control. That terminal extraction outcome is separate from the reproduced lifecycle defect: the overlay remained interactive after the helper return.

### GREEN

The initial post-fix focused command passed: `1 passed (9.2s)`. After the test was accurately renamed from “completed” to “terminal” (no behavior changed), the same owned graph and command passed again: `1 passed (17.6s)`. The helper-return assertion proved the dialog was hidden before the supported evaluation navigation/click.

## Final live journey proof

With the same one-shell owned graph, the final command was:

```bash
bunx playwright test e2e/workflows/journeys/ingest-evaluate-review.spec.ts e2e/workflows/journeys/ingest-search-chat.spec.ts --project=journeys --workers=1 --reporter=line --trace=on
```

Result: `2 passed, 1 failed (30.6s)`.

- The owning terminal-ingest regression passed.
- Ingest → Evaluate → Review passed.
- Ingest → Search → Chat passed URL ingest, search, the formerly intercepted follow-on interactions, and sent the chat request. It failed only afterwards at its unchanged `expect(...).toMatch(/playwright/i)` assertion: the assistant returned the committed mock's default text, `onboarding UAT ready. The mock provider returned a deterministic success response.`

The deterministic-mock trace is conclusive enough to classify, not repair: `local-success.json` has no `Playwright` content pattern, so that prompt deliberately falls through to `chat/default.json`, whose text lacks `playwright`. This is a missing deterministic response contract outside the two modal journeys; its assertion was not weakened and no provider/mock/test-orchestration change was made in Task 2B.

## Fix round 1 — explicit terminal close control

Review identified that the initial helper repair could reach the generic Escape fallback when neither terminal Done nor Ant's close control was visible. Commit `27f937002b test(frontend): require explicit terminal ingest close (TASK-13013.7.1)` closes that gap.

- `DismissQuickIngestOptions.terminal` makes only the terminal helper-return path reject descriptively when neither supported control is visible; existing non-terminal callers retain their Escape fallback and the processing/minimize lifecycle.
- `ingestAndWaitForReady()` invokes `dismissQuickIngest(page, { terminal: true })`.
- `apps/tldw-frontend/e2e/quick-ingest-terminal-close.spec.ts` recorded a RED where the missing-control case resolved through Escape, then GREEN proving Done emits no Escape and missing controls reject without Escape.
- The complete package-visible invocation, RED/GREEN, owned live output, typecheck result, and owner references are in `.superpowers/sdd/2026-08-30-software-supply-chain-release/task-2b-ingest-modal-fix-round-1-evidence.md`.
- The current owned two-journey run reported `2 passed, 1 failed (56.2s)`: the modal regression and Ingest → Evaluate → Review passed; the retained Search → Chat failure is only the later deterministic `/playwright/i` content assertion.

The unchanged React lifecycle owner is `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx:1819,1852`; its Results step receives `onClose` and the Ant modal is open only for `open && !state.isMinimized`. The route owner remains `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx:4420-4428`, whose supported `onClose` calls `hideQuickIngestSession()`, `setIngestOpen(false)`, and resets auto-processing state.

## Additional verification

```bash
cd apps/tldw-frontend && bun run typecheck
git diff --check
```

`git diff --check` exited `0`.

`bun run typecheck` exited nonzero only on pre-existing, unrelated dirty-file failures in `DocumentationPage.tsx`, `scripts/__tests__/skills-certification-profile.test.ts`, and `scripts/__tests__/skills-certification-runner.test.ts`. Neither Task 2B file is named in the compiler output. No Python file changed, so Bandit is not applicable.

## Files and commits

- `apps/tldw-frontend/e2e/utils/journey-helpers.ts`
- `apps/tldw-frontend/e2e/workflows/journeys/ingest-evaluate-review.spec.ts`
- `apps/tldw-frontend/e2e/quick-ingest-terminal-close.spec.ts`
- `.superpowers/sdd/2026-08-30-software-supply-chain-release/task-2b-ingest-modal-report.md`
- `.superpowers/sdd/2026-08-30-software-supply-chain-release/task-2b-ingest-modal-fix-round-1-evidence.md`

The earlier blocked-evidence report was committed as `3dc89fe1fb docs: record blocked ingest modal evidence (TASK-13013.7.1)`. This report and the repaired helper/regression are committed together in the follow-on Task 2B implementation commit.

Task 2B commits: `a3f02cd5cb test(frontend): close terminal ingest modal (TASK-13013.7.1)` and `27f937002b test(frontend): require explicit terminal ingest close (TASK-13013.7.1)`.

## Self-review and concerns

- Preserved unrelated dirty files and all prior Task 2A/Task 2E commits.
- Kept the existing supported UI lifecycle; no broad refactor or component change was made.
- No unrelated repository, workstream, Task 2C, or Task 2D file was touched.
- The external URL extraction reached terminal Error results in the controlled run, while the modal lifecycle behavior is independently proven by the visible modal RED and green close assertion. The retained Search → Chat mock-content failure requires a separate package if deterministic answer semantics are to be repaired.
