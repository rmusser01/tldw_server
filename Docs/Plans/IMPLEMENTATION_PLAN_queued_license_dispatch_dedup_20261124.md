# Queued license dispatch deduplication — TASK-15637

Spec: the approved design and regression evidence in TASK-15637. The requester authorized this separate fix PR and waived its human Change summary on 2026-11-24. Continue monitoring #3584; preserve the separately stashed UAT516 closeout.

The queue currently notices a license dispatch only after a runner posts its status. Repeated queue events can therefore replace a queued dispatch through the existing PR concurrency group. Give new dispatches a visible PR/head identity and reuse a matching live dispatch before attempting another. This is dispatch coordination, not proof that the policy passed. Runner starvation and the reason GitHub has not assigned a runner remain unknown.

ADR assessment: no new durable architecture decision. This repairs idempotency within the existing queue. ADR-063 describes that queue but remains Proposed; leave its status and rationale untouched. ADR-059 and ADR-084 govern task editing and unique IDs. Policy authority, permissions, concurrency, required gates, admission and retry limits remain unchanged.

## Stage 1: Bound the failure
**Goal**: Confirm the queue/status gap and establish existing behavior.
**Success Criteria**: Record the observed unassigned cancellations and identify the dispatch/action paths without treating them as executable failures.
**Tests**: Existing queue actions and license-dispatch suites.
**Status**: Complete

## Stage 2: Reproduce and repair
**Goal**: Preserve a live license dispatch for the same PR/head before it posts a status.
**Success Criteria**: Real queue behavior suppresses replacement of matching live requests; wrong identity, provenance and completed runs cannot suppress a needed dispatch. Legacy pr-only manual dispatches remain supported.
**Tests**: RED/GREEN behavior tests, later-page match, failed/incomplete read, workflow metadata and payload contracts.
**Status**: Complete

## Stage 3: Validate and review
**Goal**: Prove the bounded fix without changing policy behavior.
**Success Criteria**: Affected suites, actionlint, syntax, baseline/current Ruff and Bandit, canonical task checks and cached configured hooks pass; independent actual and final-staged reviews are clear.
**Tests**: Queue and license CI tests plus existing related workflow contracts. Retain the immutable audit steps digest and shared trusted-step equality checks.
**Status**: In Progress

Verification: original actions/license baseline 244 passed and one inherited Bash-version skip. New behavior tests reproduced 13 failures before production edits; three metadata contracts also failed before the workflow change. Independent review caught a queued-to-running transition between API reads; its new regression failed before scanning pre-run states first. Final six queue suites: 448 passed, one unchanged local Bash<4 skip (11.13 seconds). No hosted policy or runner-assignment claim follows from these local tests.

Expanded license/admission tests found one exact top-level workflow contract that needed the authorized `run-name` key. Its correction pins that expression exactly and retains every permission, concurrency, job and trusted-step assertion. The corrected suite passed all 14 cases. Across the eleven affected suites, 726 distinct cases passed and five existing Bash<4 cases skipped; the repeated 14-case confirmation is not extra coverage.

Python 3.12 syntax passed for all four touched Python files. Ruff: production 0/0, actions tests 2/2 inherited, other touched tests 0/0. Full Bandit: production 0/0; tests gained twelve B101 diagnostics for intentional pytest assertions, with inherited B404/B105 unchanged. No suppressions, assertion removal or baseline edits. The existing local actionlint source build reports `devel` and passes all workflows; hosted pinned 1.7.12 remains required. Independent actual and final-staged reviews are CLEAR (receipt below). The configured pre-commit stage passed from 15 healthy cached environments with initialization and installation forbidden. The manual local-ci-fast stage and wizard-only Ruff/Black were not executed for these paths. No git pre-commit hook is installed, so the normal commit must not be described as executing that hook.

## Stage 4: Land through required gates
**Goal**: Publish one separate fix PR and independently verify its actual merge.
**Success Criteria**: Normal reviewed commit/publication and current-head required CI, normal merge, expected head is an actual merge parent and the merge is an ancestor of independently read dev. Continue #3584 and then publish the UAT516 closeout separately.
**Tests**: Hosted required gates and independent landing checks; no manually dispatched required gates or protection bypass.
**Status**: In Progress

Final-staged independent review: CLEAR on the 33,962-byte patch SHA-256 `013dc680110c9a5e631403be8dc7d65fd2b02b673fbff7b2d460c1adfd62777c`. Normal commit `f4eb631c58b4193190b5f23805d4afeedb5e5944` has parent `518bda3da3db0403a239bfc5350d7e6ddd781eed` and matches that reviewed patch exactly. Normal push used an exact absent-ref lease to the new owned branch. Published as [PR #3613](https://github.com/rmusser01/tldw_server/pull/3613); hosted CI and actual merge remain pending, with auto-merge not yet enabled. These later tracking receipts are not another publication.

Causal hosted repair, 2026-11-25: backend-required job `138991496298` on run `45489977168`, exact `f4eb631` / pull_request / attempt 1, failed the tenant-isolation ratchet step. Its single new quality offense was an assertion inside the fake GitHub client: `tautology_suspect=4`, baseline 3. The complete two annotations preceded one consumed log (168,394 bytes, SHA-256 `888ef100da669efe729f9b615667dcdce839ba8705bcb75a2154787d2c14671f`). No log refetch or CI retry. Move request validation into the collected behavior test; retain the event/branch checks and original no-replacement verdict, and require that the dispatch lookup does not filter on the target PR SHA. Scoped triage reproduces RED 4 then GREEN 3. The ratchet and baseline remain byte-exact.

The necessary final integration normally rebased the unarmed fix onto independently pinned dev `4fa4d1664eb4d1f6278f50eba560772f13232060`. All 47 incoming files are outside the seven owned paths and match dev exactly. The original reviewed patch is byte-exact across that rebase; its combined tree matches an independent clean merge-tree preview. Normal autostash restored all three owned dirty files byte-exact. The separately reviewed UAT516 stash and original UAT branch remain untouched; armed #3584 remains under normal queue control.

Combined verification after integration: 891 passed, one inherited Bash<4 skip, across 18 suites in 165.59 seconds. This includes the exact four-module tenant-ratchet step, affected queue/license tests, workflow/admission/path-classifier contracts and canonical task checks. The earlier 489-pass affected run overlaps this result. Syntax passes; Ruff remains two inherited findings, zero new. Scoped full Bandit gains one intentional pytest-assert B101 (282 to 283); inherited B404 remains one, with no new non-B101 finding or suppression changes. Existing actionlint source build passes. Independent causal and combined reviews are CLEAR; final-staged review and configured cached hooks precede publication. Current-head hosted qualification and actual merge remain open; all F4 hosted evidence becomes historical after publication.
