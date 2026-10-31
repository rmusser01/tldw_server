# Task 5 independent re-review — round 2

## Verdict

**NEEDS FIXES — the round-1 remaining issue is addressed; one newly discovered Important ambiguity-retry issue remains.** No Critical or Minor findings.

- Reviewed exact range: `f6af7a3daffe843f627330406b52a26b208315f7..4e052d965fbe3a502308f821acb28e7159eb920c`.
- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/single-text-prompt-recipes`.
- Associated Backlog task: `TASK-12984.2.1` (confirmed In Progress using the CLI before artifact edits).
- Contract: Task 5 plan, spec section 10.6, review brief, updated implementer report, and ledger rulings/corrections, particularly the local-only no-project ruling and central prohibition on converting ambiguity to pending.
- Method: fresh verification-before-completion evidence and systematic-debugging for the interleaving probe. No production changes, permanent test changes, dependency changes, or test disabling.

## Findings, ordered by severity

### Important R2-1 — Project-less completion can overwrite another operation's durable ambiguity with retryable pending

**Location:** `apps/packages/ui/src/services/prompt-sync.ts:754–756` (the no-project return is classified transient at line 765; the intervening settings await is line 748).

`autoSyncPrompt` checks the local durable status and authority before awaiting `getPromptStudioDefaults()`. If another same-ID operation dispatches ambiguously while that await is pending, it correctly writes `syncStatus: "error"` and retains the authority marker. When the project-less operation resumes, it unconditionally overwrites that error with `pending` and returns `transient` plus `not_dispatched`. Its own no-dispatch evidence does not establish that the shared local row is free of another ambiguous operation.

The registry still blocks the current process, but the durable fallback has been lost. After the in-memory authority restarts, the retained pending row is admitted for another mutation. This violates the central contract that ambiguity must never become pending/auto-retryable. The round-2 field also makes the builder accept and retain this pending result; it does not add a later uncertainty check in `acceptSyncResult`.

**Origin/scope:** this unconditional status write was already present in the no-project branch at the round-2 base `f6af7a3`; it is a newly discovered residual issue in Task 5 fix round 1, not a defect introduced solely by the one-field round-2 diff. It is in the exact branch and retry contract the controller explicitly requested be re-audited.

**Concrete reproduction:** a temporary test in the existing `prompt-sync.uncertainty.test.ts` fixture used real sync functions and the real direct authority registry, with the fixture's mocked DB/settings/Prompt Studio boundary:

1. Seed valid v2 `exact-id` with no linked/default project, then start owned `autoSyncPrompt` and hold its local-defaults promise after the clear-authority preflight.
2. Run owned `pushToStudio("exact-id", 42)` concurrently. Its mocked mutation boundary marks the real registry scoped and returns a dispatched status-0/no-body result. Assert the local row is now `error`.
3. Resolve the held defaults to `{ defaultProjectId: null }`. The first operation returns `transient`, `pending`, and `not_dispatched`; the registry remains scoped, but the row is now `pending`.
4. Use `vi.resetModules()` and fresh imports of sync/authority to model an authority-process restart while retaining the local DB fixture. Assert the fresh registry is clear. A subsequent owned auto-sync to project 42 invokes the mocked create boundary a second time.

Exact probe, inserted immediately after the maintained lost-connection retry test and then removed:

```ts
it("review probe: project-less completion preserves a concurrent durable ambiguity", async () => {
  seed()
  mocks.rows.get("exact-id").studioProjectId = null
  let finishDefaults!: (value: { defaultProjectId: null }) => void
  mocks.defaults.mockImplementationOnce(
    () => new Promise((resolve) => { finishDefaults = resolve })
  )
  const pending = sync.autoSyncPrompt("exact-id", undefined, {
    expectedOwnerId: ownerB
  })
  await vi.waitFor(() => expect(mocks.defaults).toHaveBeenCalledTimes(1))
  mocks.create.mockImplementation(async () => {
    await registry.markRecipePersistenceScoped("exact-id", ownerB)
    return { ...response(), ok: false, status: 0, data: undefined }
  })
  await sync.pushToStudio("exact-id", 42, { expectedOwnerId: ownerB })
  expect(mocks.rows.get("exact-id").syncStatus).toBe("error")
  finishDefaults({ defaultProjectId: null })
  const result = await pending
  expect(result).toMatchObject({
    failureKind: "transient",
    syncStatus: "pending",
    recipeOwnership: { dispatch: { state: "not_dispatched" } }
  })
  expect(await registry.readRecipePersistenceUncertainty("exact-id", ownerB)).toBe("scoped")
  expect.soft(mocks.rows.get("exact-id").syncStatus).toBe("error")
  vi.resetModules()
  const restartedSync = await import("../prompt-sync")
  const restartedRegistry = await import("../recipe-persistence-uncertainty")
  expect(await restartedRegistry.readRecipePersistenceUncertainty("exact-id", ownerB)).toBe("clear")
  mocks.create.mockImplementation(async () => {
    await restartedRegistry.markRecipePersistenceScoped("exact-id", ownerB)
    return { ...response(), ok: false, status: 0, data: undefined }
  })
  await restartedSync.autoSyncPrompt("exact-id", 42, { expectedOwnerId: ownerB })
  expect(mocks.create).toHaveBeenCalledTimes(1)
})
```

Command from `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run src/services/__tests__/prompt-sync.uncertainty.test.ts --reporter=dot -t 'review probe'
```

Evidence: `/tmp/task5-rereview2-concurrency-restart-probe.log`, exit 1, one failed test with two independent assertion failures: `expected 'pending' to be 'error'` and `expected "vi.fn()" to be called 1 times, but got 2 times`. The 43 other tests were name-filter skips, not disabled tests. The initial non-restart probe independently reproduced the status overwrite at `/tmp/task5-rereview2-concurrency-probe.log`.

This probe proves central sync admits the second mocked mutation after a fresh authority is loaded; it does not claim a real remote duplicate was sent. Preventing the durable-error downgrade is required regardless. Make no-project recovery unable to downgrade a concurrently written durable error, return a lock-aware outcome if uncertainty has appeared, and cover this interleaving. A second non-atomic early read alone would leave another read/write race.

## Addressed: project-less Save/Update integration

The missing-field issue from `task-5-rereview-1.md` is fixed. The added `failureKind: "transient"` matches the existing builder's three-part local-pending acceptance predicate: transient failure, pending status, and proven `not_dispatched`. For a non-overlapping no-project operation, the result is produced before any transport call, so this classification is appropriate and does not itself turn that operation into an ambiguous retry.

Fresh maintained tests exercise the actual builder, sync, Prompt Studio client, `apiSend`, and direct/background wiring with only platform boundaries mocked. All four new cases at `PromptRecipeBuilder.dispatch.test.tsx:311` pass:

| Adapter | Action | Verified outcome |
| --- | --- | --- |
| Direct | Save as new | Exact edited row retained, pending, local-pending notice, clear uncertainty, zero fetches |
| Direct | Update | Edited Markdown definition retained rather than old snapshot, pending, notice, clear uncertainty, zero fetches |
| Background | Save as new | Exact edited row retained, pending, local-pending notice, clear uncertainty, zero fetches |
| Background | Update | Edited Markdown definition retained rather than old snapshot, pending, notice, clear uncertainty, zero fetches |

The missing-field fix should be retained. The outstanding finding concerns another operation's ambiguity becoming pending, not the validity of this failure classification for an isolated proven pre-dispatch result.

## Fresh verification

From `apps/packages/ui`, ran the following with `./node_modules/.bin/vitest run`, `--reporter=dot --sequence.shuffle --sequence.seed=12984`:

```text
src/services/__tests__/prompt-sync.structured-prompts.test.ts
src/services/__tests__/prompt-sync.auto-sync.test.ts
src/services/__tests__/prompt-sync.uncertainty.test.ts
src/db/dexie/__tests__/prompt-rollback.test.ts
src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.test.tsx
src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.dispatch.test.tsx
src/components/Common/PromptAssist/recipes/__tests__/SingleFieldRecipeEditor.test.tsx
src/services/__tests__/prompt-studio.recipe-policy.test.ts
src/components/Option/Prompt/__tests__/prompt-sync.owner-callers.test.tsx
src/components/Option/PromptStudio/__tests__/PromptStudioPlaygroundPage.recipe-policy.test.tsx
src/services/__tests__/request-core.persistence-scope.test.ts
src/services/__tests__/recipe-persistence-authority.test.ts
src/entries/__tests__/background.recipe-persistence-owner.test.ts
src/services/__tests__/recipe-persistence-registry.test.ts
src/services/__tests__/api-send.test.ts
```

- **15 files / 1,274 tests passed**, no skips, exit 0. Log: `/tmp/task5-rereview2-focused.log`.
- Temporary probe removed with `apply_patch`; `git status --short` was empty afterward. Reran the restored uncertainty suite: **43/43 passed**, exit 0. Log: `/tmp/task5-rereview2-restored-uncertainty.log`.
- Repository frontend ESLint binary/config on the two changed TS/TSX files: exit 0, no lint findings. Only the existing Next pages-directory config notice. Log: `/tmp/task5-rereview2-eslint.log`.
- `git diff --check f6af7a3daffe843f627330406b52a26b208315f7..4e052d965fbe3a502308f821acb28e7159eb920c`: exit 0.
- Independently inspected the entire three-file fix diff and surrounding central classification, registry, and builder acceptance logic. No unrelated production change in this round.
- Extension compile/full frontend typecheck were not independently rerun in this review. The implementer reports extension pass and the same 86 baseline frontend diagnostics; those are not presented as reviewer-run checks. Optional baseline drawer failures remain outside this focused run.
- Bandit is not applicable to the TS-only fix/report; no Python security scan is claimed.

## Handoff

Addressed: the sole previously open Important missing-classification issue. Newly discovered/open: Important R2-1 above. All original round-1 findings retain their prior addressed status. The review artifact is the only retained file change from this review and is committed separately as requested; the exact commit is reported to the controller after commit verification.
