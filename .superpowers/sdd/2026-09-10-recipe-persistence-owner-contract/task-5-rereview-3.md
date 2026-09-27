# Task 5 independent re-review — round 3

## Verdict

**NEEDS FIXES — Important R2-1's no-project transition is addressed, but two other concurrent-lock downgrade paths remain.** No Critical or Minor findings.

- Exact reviewed range: `e261797fb38c77adc9ce5058b306d8dc90b49bfb..490c50bf97befce188b14b97b9d106123f6cb01a`.
- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/single-text-prompt-recipes`.
- Reviewed the full five-file diff, Task 5 plan, spec section 10.6, updated implementer report, and ledger rulings, including the round-3 builder retention authorization.
- Backlog association `TASK-12984.2.1` confirmed In Progress through the CLI. No review-stage status change was appropriate; no Backlog file was manually edited.
- Skills used: verification-before-completion; systematic-debugging for the two additional failure hypotheses. No production changes or permanent test changes were made.

## Findings, ordered by severity

### Important R3-1 — The other auto-sync pending fallback still downgrades concurrent durable error

**Location:** `apps/packages/ui/src/services/prompt-sync.ts:797–800`, reached from the pre-dispatch local-read catch in `pushToStudio` at lines 825–835.

The new atomic update protects the no-project branch, but the known-project transient fallback below it still performs an unconditional `syncStatus: "pending"` write. After auto-sync has passed its initial local/authority checks, `pushToStudio` rereads the local record. If that read fails while another same-ID operation has completed ambiguously, the catch returns `transient` plus `not_dispatched`. The fallback then overwrites the other operation's durable `error` with pending. This attempt's no-dispatch evidence is not evidence that the shared row is safe to downgrade.

**Reproduced through both real direct and background transport paths:** pause the second local read, complete an owned manual mutation with malformed 2xx response, confirm durable error, then release the paused read with a transient rejection. Both cases change error to pending. Reset modules, recreate the real background listener where applicable, confirm the new authority is clear, and retry to a known project: the mocked fetch boundary observes **two actual prompt mutation requests**, not one.

Evidence: `/tmp/task5-rereview3-adjacent-pending-probe.log`, exit 1, **2 failed tests**, each with expected-error/received-pending and expected-one/received-two-mutations failures. Both reproduce again in `/tmp/task5-rereview3-additional-probes.log`.

**Origin/scope:** this branch predates round 3; it is newly discovered by the controller-requested audit of other pending/error downgrades in the same touched flow. The callback itself did not introduce it. Apply the same durable-lock invariant to every v2 pending transition, including lock-aware result propagation to callers. Merely checking this request's dispatch flag is insufficient.

### Important R3-2 — Delayed uncertainty preflight returns stale status, bypassing the new builder rollback guard

**Location:** `apps/packages/ui/src/services/prompt-sync.ts:728`, consumed by the new guard at `apps/packages/ui/src/components/Common/PromptAssist/recipes/PromptRecipeBuilder.tsx:233–237`; destructive rollback follows at lines 256 and 306.

`autoSyncPrompt` captures `local` before awaiting the authority read. When the real extension authority read is delayed, another same-ID manual write can finish ambiguously and store `error`. The delayed read correctly returns scoped uncertainty, but sync reports the old local/synced status from its stale snapshot, with validation/not-dispatched metadata. The new builder guard only retains rows when the returned status is exactly error, so it misses this already-unresolved operation and executes ordinary rollback.

**Reproduced through the real background listener and real builder for Save and Update, with no database failure:** hold only the first uncertainty-read message after clicking the write action; complete the overlapping owned manual write; assert the row is error; release that message to the real authority. Save deletes the exact local row. Update restores the old XML definition and `syncStatus: "synced"`. Neither shows the unverified-outcome notice. There is one ambiguous remote mutation; the local durable recovery record is erased by the other caller.

Evidence: `/tmp/task5-rereview3-additional-probes.log`, exit 1, **2 additional failed tests**: Save's row is `undefined`; Update's row is `synced`/XML instead of `error`/Markdown. Each also lacks the uncertainty notice. Together with R3-1 this run has **4 failed tests / 8 failing assertions**, with 43 maintained cases skipped only by the name filter.

**Origin/scope:** the stale-status return predates this fix, but the new status-only rollback guard does not close the full builder invariant authorized for round 3. Treat an existing-uncertainty block as a lock-aware outcome rather than ordinary safe validation rollback; do not fabricate dispatch/owner evidence for the current attempt. Ensure rollback cannot erase another operation's durable/authority lock.

## Confirmed addressed and preserved behavior

- Original R2-1: the no-project callback observes the current row within the write transaction. A concurrent error is not modified and produces validation/error with this attempt still accurately not-dispatched. The four maintained central races cover direct/background and settings/local-write-boundary delays, including a fresh authority and no second mutation.
- Builder handling of that specific returned error: the four maintained direct/background Save/Update overlap tests retain the exact error-locked edited row, show the unverified-outcome notice, disable the write action, and refuse another mutation after a fresh authority is loaded.
- Isolated no-project operations: the four unchanged direct/background Save/Update tests retain edited Markdown rows as pending, show the local-pending notice, keep uncertainty clear, and make zero fetches. These remain passing in both full runs.
- The prior missing-transient-field fix remains correct for isolated proven pre-dispatch pending recovery.

## Dexie callback and return-semantics audit

Inspected the installed **Dexie 4.2.1** source, resolved by the UI package to the existing workspace dependency; no dependency was added.

- `apps/packages/ui/node_modules/dexie/dist/dexie.mjs:1567`: `Table.update` delegates to exact-primary-key `Collection.modify`.
- Line 1917: `_write` uses a locked readwrite transaction. The callback reads/clones the record and queues its mutation within that transaction; the production callback is synchronous and has no external await.
- Line 2226: a callback return value of `false` suppresses the write.
- Line 2274: the resolved number is **the number of matched keys**, not necessarily the number of actual writes. An existing row whose callback returns false still resolves to 1; a missing key resolves to 0 without invoking the callback.
- The production code correctly ignores that count and uses `durableError`, set inside the callback, to select the lock-aware result. The count semantics therefore do not undermine this fix.
- The two test doubles currently return 0 for an existing row whose callback returns false, unlike installed Dexie. This is a fixture-fidelity limitation, not a demonstrated production defect here, because neither the changed production branch nor these regressions uses that count. No live IndexedDB-engine verification is claimed; `fake-indexeddb` is not installed in the UI package. The atomicity conclusion is source-backed.

## Exact temporary probe reproduction

These tests were inserted at the start of the existing `PromptRecipeBuilder.dispatch.test.tsx` describe block, using its existing fixtures, and removed afterward with `apply_patch`:

```ts
it.each(["create", "update"])(
  "review probe: background %s cannot roll back concurrent ambiguity found by delayed authority read",
  async (operation) => {
    await startBackground();
    seed(operation);
    mocks.markerFails = false;
    mocks.defaultProjectId = null;
    const user = userEvent.setup();
    const view = renderBuilder(ownerScope);
    if (operation === "update") await selectSaved(user);
    await user.selectOptions(screen.getByRole("combobox", { name: "Output format" }), "markdown");
    const reached = deferred();
    const release = deferred();
    const deliver = mocks.sendMessage.getMockImplementation()!;
    let pauseNextRead = true;
    mocks.sendMessage.mockImplementation(async (message) => {
      if (pauseNextRead && message.type === "tldw:recipe-uncertainty:read") {
        pauseNextRead = false;
        reached.resolve();
        await release.promise;
      }
      return deliver(message);
    });
    await clickWrite(user, operation);
    await reached.promise;
    await pushToStudio("dispatch-id", 42, { expectedOwnerId: ownerScope });
    expect(mocks.rows.get("dispatch-id").syncStatus).toBe("error");
    release.resolve();
    await screen.findByText(/Could not (?:save|update) the recipe|server outcome could not be verified/i);
    expect.soft(mocks.rows.get("dispatch-id")).toMatchObject({
      id: "dispatch-id",
      syncStatus: "error",
      structuredPromptDefinition: { assembly_config: { render_format: "markdown" } },
    });
    expect.soft(screen.queryByText(/server outcome could not be verified/i)).toBeInTheDocument();
    expect(promptMutations()).toHaveLength(1);
    view.unmount();
  },
);

it.each(["direct", "background"])(
  "review probe: %s known-project pre-dispatch failure preserves concurrent ambiguity",
  async (adapter) => {
    if (adapter === "background") await startBackground();
    seed("update");
    mocks.markerFails = false;
    const reached = deferred();
    const release = deferred();
    mocks.beforeRead
      .mockResolvedValueOnce(undefined)
      .mockImplementationOnce(async () => {
        reached.resolve();
        await release.promise;
        throw new Error("transient local read failure");
      });
    const pending = autoSyncPrompt("dispatch-id", 42, {
      expectedOwnerId: ownerScope,
    });
    await reached.promise;
    await pushToStudio("dispatch-id", 42, { expectedOwnerId: ownerScope });
    expect(mocks.rows.get("dispatch-id").syncStatus).toBe("error");
    release.resolve();
    await pending;
    expect.soft(mocks.rows.get("dispatch-id").syncStatus).toBe("error");
    vi.resetModules();
    if (adapter === "background") await startBackground();
    const restartedSync = await import("@/services/prompt-sync");
    const restartedRegistry = await import("@/services/recipe-persistence-uncertainty");
    expect(await restartedRegistry.readRecipePersistenceUncertainty("dispatch-id", ownerScope)).toBe("clear");
    await restartedSync.autoSyncPrompt("dispatch-id", 42, { expectedOwnerId: ownerScope });
    expect(promptMutations()).toHaveLength(1);
  },
);
```

Run from `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.dispatch.test.tsx -t 'review probe' --reporter=dot
```

Only storage/config/runtime-message delivery/fetch boundaries are controlled. Builder, sync, Prompt Studio client, apiSend, request-core, and the background listener remain real. Network calls end at mocked fetch; no real remote writes occurred.

## Fresh verification and limitations

- Eight requested regressions: `vitest run src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.dispatch.test.tsx -t 'concurrent ambiguity|concurrent durable ambiguity' --reporter=dot` — **8 passed / 35 name-filter skips**, exit 0; `/tmp/task5-rereview3-eight.log`.
- Full focused set, both before probes and after their complete removal: **15 files / 1,282 tests passed**, no skips, exit 0. Logs: `/tmp/task5-rereview3-focused.log` and `/tmp/task5-rereview3-restored-focused.log`.
- The full set is the exact 15-path list recorded in `task-5-rereview-2.md`, with `--reporter=dot --sequence.shuffle --sequence.seed=12984`: the three prompt-sync suites, prompt rollback, builder unit/dispatch, editor, Prompt Studio policy, owner-aware caller hook, real Playground policy, request-core scope, direct authority, background authority, registry, and apiSend.
- Repository frontend ESLint binary/config on all four changed TS/TSX files: exit 0, no findings, only the existing Next pages-directory configuration notice; `/tmp/task5-rereview3-eslint.log`.
- `git diff --check e261797fb38c77adc9ce5058b306d8dc90b49bfb..490c50bf97befce188b14b97b9d106123f6cb01a`: exit 0.
- Worktree was clean after removing temporary tests and before creating this artifact. No test expectation from the implementation was relaxed or disabled.
- Extension compile/full frontend typecheck were not independently rerun by this reviewer. The report's extension pass and 86 unchanged frontend baseline diagnostics remain implementer evidence, not reviewer-run checks. Previously recorded optional drawer failures remain outside this focused run.
- No Python/backend changes: Bandit is not applicable to the TS-only fix and this Markdown review; no successful Python security scan is claimed.

## Handoff

Addressed: original R2-1 and its specifically tested no-project builder consequences. Newly discovered/open: Important R3-1 and R3-2 in neighboring pending/rollback paths expressly included in this re-audit. These are residual shared-lock gaps, not regressions in Dexie's callback mechanism. Only this review artifact is retained and committed, as requested; its verified commit and worktree status are reported separately to the controller.
