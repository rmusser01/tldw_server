# PR3613: synchronize the collapsed-sidebar preference test

Task: TASK-15637. PR: https://github.com/rmusser01/tldw_server/pull/3613.

The published `a0a7f21` head failed shared UI coverage shard 25/32 because the omitted-shortcuts case found the default Media button. Its Chat locator also exists in the default selection, so it does not establish that saved preferences have loaded. The unchanged isolated case and full 40-case module pass locally; that does not dismiss the hosted failure or establish flakiness.

Keep the real `useSetting`, registry, Plasmo hook and existing storage IO mock. Make the stored-selection read explicitly asynchronous in the existing case and synchronize its original absence and scrolling assertions with that read. Do not change production rendering, expected shortcuts, test admission, timeouts or assertions.

ADR assessment: no new ADR is needed. This corrects synchronization with the existing asynchronous settings contract; it changes no durable runtime or storage rule. ADR-059 and ADR-084 govern task editing and identity. Accepted ADR rationale remains unchanged.

## Stage 1: Prove the readiness gap
**Goal**: Establish the same assertion failure with a controlled pending storage read.
**Success Criteria**: The original assertion observes default Media before the saved selection resolves; all setup failures and baseline passes remain distinguished from that result.
**Tests**: Existing omitted-priority-shortcuts case with the existing deferred IO seam; original full module baseline.
**Status**: Complete

## Stage 2: Synchronize the existing test
**Goal**: Await stored preferences before asserting the omitted shortcuts and rail layout.
**Success Criteria**: The controlled case passes with every original absence and scrolling assertion retained; production files remain byte exact.
**Tests**: Same controlled case RED/GREEN; full history-navigation module.
**Status**: Complete

## Stage 3: Verify and review
**Goal**: Check connected sidebar/settings behavior and the exact final patch.
**Success Criteria**: Affected and connected tests pass; scoped static checks introduce no finding; independent actual review is clear. Final-staged review belongs to Stage 4. The test repair authors no Python change; incoming integration scope is checked in Stage 4.
**Tests**: Existing sidebar navigation, shortcut and settings hydration/registry suites; baseline/current static comparison.
**Status**: Complete

## Stage 4: Integrate and land
**Goal**: Publish the reviewed repair to the same unarmed PR and verify its actual merge.
**Success Criteria**: Necessary final dev integration preserves owned work and protected stashes; final-staged independent review is clear; normal cached checks/commit and exact-head lease publication succeed. New publication resets hosted qualification. All current required and admitted executions qualify before normal auto-merge; the actual merge parent and current-dev ancestry are independently verified.
**Tests**: Source/index/reviewed-patch/criteria guards, complete hosted CI and feedback census, exact-head merge verification. Skipped and unrelated native paths remain unverified.
**Status**: In Progress

## Validation record

- The unchanged isolated case passed locally, as did all 40 original module cases. These fast reads do not qualify the hosted failure as fixed or flaky.
- Holding the real test storage boundary reproduced the same default-Media absence assertion: controlled RED, one failed case in 233 ms. Releasing the stored selection and awaiting the original absence assertions produced controlled GREEN in 236 ms. The single-case runs are subsets of the full module, not additional cases.
- The exact hosted callback schedule remains unmeasured. The verified readiness gap is that Chat belongs to both the defaults and stored selection; finding Chat alone does not establish preference hydration.
- Production sidebar, settings hook and registry behavior, all four original assertions, test admission and timeouts remain unchanged. The deferred read is released in `finally` even if an assertion fails.
- Five affected/connected modules passed: 74 distinct cases, comprising the original 40 navigation cases and 34 settings/registry/shortcut/lazy-history cases. The targeted GREEN is a subset. Independent actual-source review is CLEAR.
- Scoped syntax and ESLint pass, baseline 0/current 0/new 0. Connected TypeScript diagnostics are baseline 5/current 5/new 0, with none in the touched file; this is not a full-project typecheck. Existing whole-file formatting drift is present in both versions; no broad reformat or baseline write was made. The test repair authors no Python change.
- Normal integration onto independently observed dev `9c5c771b163201e29d6de729c1cd6d688e2d08b0` completed while the PR was unarmed. Incoming review is CLEAR. All 54 incoming paths are byte exact to dev, with no owned overlap. All three original commits remain range equal and their 75,325-byte patch is unchanged. The rebased committed tree matches the independent clean merge-tree preview; normal autostash restored all five owned working files and preserved the protected individual stashes and backup.
- Bounded combined validation completed three disjoint cohorts: 445 passed, 145 skipped, no failures. The approved PostgreSQL endpoint was closed; 144 Auth-cohort skips and one RAG-cohort skip remain unverified. No container or Docker start was attempted. This is not hosted or whole-product qualification.
- All 41 incoming Python files parse under supported Python 3.12. Incoming source is byte exact to pinned dev: one current static scan reports its inherited Bandit baseline (1,155 findings, zero high/two medium) and Ruff baseline (51 findings), with no authored Python delta or suppression. The four original owned Python baseline/current source pairs remain byte exact, so their existing scoped results were reused without rerun.
- The actual canonical OpenAPI fingerprint and committed path union match on the combined tree (21.357 seconds, read-only checks). The connected TypeScript comparison after integration remains 5/5 diagnostics with zero introduced or touched-file diagnostics. Read-only pre-commit preflight found 15 healthy configured hooks across five install keys, with zero initialization/install. No Git pre-commit hook is installed; the configured stage must run explicitly before the normal commit.
