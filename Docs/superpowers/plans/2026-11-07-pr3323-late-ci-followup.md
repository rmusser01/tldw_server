# PR #3323 Late CI Follow-up

**Task:** TASK-15480 (original local evidence run ID: TASK-15466)
**Follow-up PR:** https://github.com/rmusser01/tldw_server/pull/3438
**Base:** `4b6b9ec446387b957d000d465c38fe682e6940e3` (`dev`); initial publication base was `dfac3ca7f34a296a2dc3a3fbc340efdd1a0bae5c`.
**Goal:** Validate all four late CI failures and publish only confirmed, unowned repairs.
**ADR check:** No new ADR is required for restoring existing lock configuration and generated-artifact contracts. ADR-059 governs task editing; ADR-085 governs the separately owned AuthNZ fixture repair.

## Constraints

- Preserve production behavior unless a regression proves it incorrect.
- Do not weaken assertions, increase thresholds, add dependencies, or bypass CI.
- Independently validate the overlapping AuthNZ and MCP fixes in PR #3406; link their ownership instead of duplicating pending changes.
- The requester has explicitly waived the human Change summary for this PR and authorized normal auto-merge: "Change summary waived for PR #3438." Required checks and review protections remain enforced; no administrative merge or manual required-gate dispatch is authorized.
- Use the shared project venv and existing database fixtures. Keep external logs and experiments outside the checkout.

## Stage 1: Attribute and Reproduce
**Goal:** Establish a current-dev disposition for each original failing job.
**Success Criteria:** Read original failures, compare current code and pending fixes, reproduce the relevant failure mode.
**Tests:** Logging timeout case and module; WebScraping inventory guard; malformed MCP handle matrix; generated-file cold/warm scoped-storage regression.
**Status:** Complete

- [x] Retrieve original run `41692726213` job logs.
- [x] Identify existing ownership of MCP/AuthNZ repairs in #3406.
- [x] Reproduce current failures and validate pending repairs independently.

## Stage 2: Minimal Repairs
**Goal:** Repair only confirmed unowned failures.
**Success Criteria:** Regression-sensitive logging coverage and exact deterministic scanner output; no unrelated changes or duplicate #3406 patch.
**Tests:** Targeted red/green checks, forced reload-race negative control, generated-artifact parity and second-refresh determinism.
**Status:** Complete

- [x] Diagnose the logging test's settings-generation and reload boundaries before choosing a repair.
- [x] Reconcile scanner output with both generated inventory artifacts after an observed failing guard.
- [x] Record exact reviewed deltas and pending-PR evidence for all four failures.

## Stage 3: Qualify and Publish
**Goal:** Open the follow-up PR with an auditable qualification record.
**Success Criteria:** Affected tests and security/lint/format/task guards verified, independent review complete, normal hooks pass, clean published branch.
**Tests:** Exact affected CI groups where feasible, repository-wide execution attempt, scoped Ruff/Black/compile/Bandit, whitespace/task guards.
**Status:** In Progress (publication complete; explicit waiver recorded; exact-head hosted qualification pending)

- [x] Run affected selections and record every failure, skip and environment limitation without relabeling them green.
- [x] Review source/test scope and baseline-compare any inherited security findings.
- [x] Commit code and canonical task records together; open draft PR against dev with all four dispositions and #3406 dependency.
- [x] Record the requester's explicit waiver and authorization for normal auto-merge on this PR.
- [ ] Before landing, satisfy the exact-current-head protected checks and review gates through normal GitHub auto-merge.

## Initial Evidence

| Shard / job | Original failure | Ownership |
| --- | --- | --- |
| core-server-smoke / 126208300683 | Runtime-timeout assertion observed ~5 seconds instead of <0.5 seconds | This task |
| auth-integration-b-z / 126208312136 | Six cold/warm scoped-storage subprocess failures | Pending #3406; validate membership-context repair |
| platform-mcp-inapp / 126208349817 | 36 malformed-handle cases error because `mocker` fixture is absent | Pending #3406; validate standard-library spies |
| integrations / 126208352624 | Exact WebScraping import inventory differs from current import surface | This task |

Original logs are retained under `/tmp/pr3323-followup-20261107/`; the job IDs and linked PR preserve durable provenance.

Before first publication, a final remote check found that concurrent PR #3436 had allocated TASK-15466 through TASK-15476. The canonical task editor renumbered only this follow-up's task to TASK-15480. Earlier local evidence report paths retain their original 15466 run identifier; their verified source hashes and results are unchanged.

## Repair Evidence

### Logging

The ordinary current-dev module run passed 21 tests. A disposable probe scheduled the existing settings initializer on a writer thread during the shared module's reload, after `_LOG_FILE_SETTINGS_INITIALIZED` was reset but before `_LOG_FILE_LOCK_TIMEOUT = 5.0` executed. Initialization set the timeout to 0.2 and the initialized flag to true; the remainder of the reload then reset only the timeout to five seconds. The unchanged timeout test failed at 5.0022 seconds, reproducing the historical symptom. The initiating thread of the historical hosted run was not captured; this demonstrates a matching, source-supported race rather than recovering that exact execution.

The new held-lock regression also failed before repair: constructing the fallback fixture reconfigured the canonical module and let a contender bypass its already-held lock. The fixture now executes the unchanged production source in a private module namespace instead of reloading the live writer's globals. All 22 module cases pass with the disposable race probe. The original environment timeout (0.2), upper bound (0.5), failed-contender inode/ownership assertions, and release-on-error coverage are preserved. No production file changed.

### Import Inventory

The exact artifact guard failed before regeneration. The canonical generator replaces three line references in each generated artifact: `unified_pipeline.py` imports at 3387/3416 become 3244/3273. Commit `9e7019ccc585acd8e544b11a9afad3045301450f` consolidated imports and shifted those anchors. All other fields across 261 records and nine targets are unchanged. The ten inventory tests, exact scanner byte parity, and second-refresh determinism pass. Neither the scanner nor the guard changed.

The original hosted log names those same three differences, and the source consolidation is an ancestor of the merged #3323 head. This is inherited generated-artifact drift, not a change in the Jobs completion implementation.

### Pending Repairs Owned By PR #3406

Independent validation applied only the two relevant test patches from #3406 head `2c4a9852343eaed5b229c090e20bfca087cea43a` to a disposable checkout of the same dev base. Both resulting test files matched the reviewed PR blobs byte-for-byte. Neither repair is already on dev; no duplicate patch is included here.

- MCP: CI's disabled plugin autoload does not supply `pytest-mock`'s `mocker` fixture. Replacing those spies with standard-library `AsyncMock(wraps=...)` and built-in `monkeypatch` changes 36 setup errors into 36 passes; the containing accounting file passes all 63 tests. An injected forbidden repository effect makes all 36 negative cases fail, retaining guard sensitivity.
- AuthNZ: the six failing `org`/`team`/`user` scenarios call `add_org_member` and `add_team_member` without the required membership-write context. A sanitized diagnostic confirms the keyword-only-argument `TypeError`. Both fixture calls now pass the existing trusted bootstrap context, consistent with ADR-085 and neighboring fixtures. The matrix changes six failures/six passes into 12 passes across real SQLite and PostgreSQL, with no skips. Independent mutations removing either context fail as expected.

Verified PR blobs: MCP `36ff914789b08a07373175c293e8eaed92639559`; AuthNZ `a6ffb2e6ec3ac0bd8599a4c2db7cf0fe07a23467`. Relevant production blobs are identical between dev and the tested PR snapshot. Full commands, patch hashes, and retained logs are in the local durable report `pr3323-sidecar-15466-20261107/REPORT.md` under the chat's visualization artifacts. The completed matrices are qualification evidence; the optional 73-case AuthNZ containing-file run was interrupted on coordination request and produced an event-loop-closed finalizer error. It is not counted as a completed-file pass or complete teardown guarantee.

### Qualification Caveats

The first full core-server-smoke selection completed with 228 passes and one unrelated Hypothesis `too_slow` input-generation health-check failure in `test_expand_preserves_document_count_or_increases` (two valid draws after 25.76 seconds), before any Logging case. A second exact-group run also has 228 passes and the same health-check failure (8.85 seconds). The first reported seed passes on isolated attribution (one test); no deadline or health check was relaxed. Neither exact-group run is claimed green.

Bounded independent profiling with the original plugin set and first seed passed 100 valid examples/117 draws. It measured 4.90 seconds under Hypothesis's lazy `_maybe_draw_constant`, 1.76 seconds in Unicode/charmap initialization, and 0.21 seconds in GC; the slowest GC-adjusted draw was 4.63 seconds. The installed Hypothesis source lazily scans imported modules and parses local source for constants, providing a suite/import-state-sensitive overhead mechanism. Draw timing subtracts GC time, so ordinary GC pauses do not directly explain the health check. The two failed runs were not profiled: this supports a mechanism rather than proving their exact slow subcall, and CPU load was not established. The contextual test and its core source match the dev base. This local verification limitation is reported, not masked by a threshold change, plugin-policy change or suppression.

An initial seeded-attribution command did not load Hypothesis's pytest plugin under `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, so pytest rejected `--hypothesis-seed` before running tests. The corrected attribution explicitly loads `hypothesis.extra.pytestplugin`; this command error is not a test result.

Scoped Ruff, changed-line Black, compilation, whitespace, canonical task normalization and unique task IDs pass. Bandit has the same three inherited B105 dummy-secret findings in existing redaction tests as current dev (B101 assertions excluded for both scans), with no new findings or scan errors. No security baseline or suppression changed.

Final combined changed-scope tests pass 32 cases (22 Logging plus ten inventory guards, 110 warnings). The test-quality/task ratchets pass 33 cases (113 warnings); Docs pass 212 cases (653 warnings). Counts overlap with other selections and are not additive full-repository totals.

A repository-wide fail-fast run collected 88,180 tests and finished execution at 840 passes, 12 collection skips, and the existing `tests/Admin/test_backup_storage_scope.py::test_shared_audit_refuses_stale_per_user_file[backup]` failure (11,648 warnings). The fixture expected shared audit mode but observed per-user mode after full collection. PR #3406 already owns a separate settings-generation repair for that failure, binding the audit/backup consumers to the intended current config. This follow-up does not duplicate that patch or claim repository-wide success.

Independent logging review found no actionable findings and reproduced baseline/repair race timings of 5.0126/0.2156 seconds; it also checked preservation of the live module's settings, queue, worker and sinks. Production, generator, guard, CI policy, dependencies and security baselines remain unchanged.

### Publication Qualification

The complete current integrations selection finished with **3,490 passes, 28 skips, zero failures/errors**, and 22,418 warnings in 871.64 seconds. Its 3,518 collected cases include the formerly failing inventory guard. Skips are 17 optional curl-cffi cases, nine opt-in external API cases and two cases whose management router is not enabled in this environment; none is an unavailable PostgreSQL case. The selection and CI plugin/marker settings are recorded in `/tmp/pr3323-followup-20261107/integrations-green.log` and its XML peer.

Source and related task/plan records were committed normally as `1d0a61a9b3fb9c24876ab2684214df9aa18c5357`, published on `codex/jobs-postmerge-ci-followup`, and initially opened as draft PR #3438 against the then-current verified dev base. The committed-task ratchet and unique-ID tests pass 15 cases against that exact base. The later publication-record commit changes only this plan and the task; repaired source/generated-artifact blobs remain unchanged. At that initial publication no merge or auto-merge was requested, and the old #3323 waiver was not inherited. The subsequent explicit authorization below supersedes that initial landing state.

### Review And Auto-merge Follow-up

The requester stated, verbatim: "Change summary waived for PR #3438." The same request authorizes addressing all validated review comments and failing checks, rebasing if needed, and enabling normal auto-merge. This waiver is specific to #3438; it does not remove any GitHub protected-check requirement.

The initial live audit read the full paginated discussion, submitted reviews, inline comments and thread state. There were no submitted reviews or inline threads, and no further thread page. CodeRabbit's draft-skip message is not a completed review. No failing check was reported; the eight required gates had passed on published `c3ccd5a3298b82d9b4f68deced61192e7795f5e4` while broad CI was still running/queued. Those successes do not qualify a new head.

Live dev advanced to `cb70f0c5d134ac585c70fc30b5a99bf0ced98562` through WebUI header PR #3400. Its 12 changed paths do not overlap this PR. With auto-merge verified off, both PR commits rebased cleanly. `git range-diff` proves both patches unchanged (rebased `b2691654a0` and `4718c29076`), and all repaired test/generated-artifact blobs are identical to the previous published head. No production behavior or tests were altered to accommodate the rebase. Fresh affected verification will cover the rebase and waiver metadata before an explicit SHA-specific leased push.

The merge-queue variable is unset. After the final publication and review/check refresh, mark the PR ready and enable normal merge-commit auto-merge. Do not bypass checks, dispatch required gates manually, cancel unrelated CI, or keep rebasing an armed PR. Task completion remains conditional on actual landing, not merely arming auto-merge.

Fresh local verification after this rebase passes **277 tests, zero failures/errors/skips** (848 warnings): 22 Logging, ten inventory, 212 Docs and 33 test-quality/task ratchet cases. Ruff, scoped Black, compilation, whitespace, canonical task normalization and unique IDs pass. Bandit against the exact new dev base retains the same three inherited B105 synthetic-secret findings with no new findings or analysis errors (B101 excluded in both test scans). Backend/helper/config inputs are identical between the two dev bases; the earlier broad matrices remain historical source-identical evidence, not new-head hosted passes.

The old published head's hosted core-server-smoke check completed successfully during this audit, confirming the previously disclosed local Hypothesis limitation is not a hosted failure on that head. No failing check or actionable review comment was reported before final publication. PR #3406 remains open at its independently validated source head. The rebased head still needs its own hosted protected gates before normal auto-merge can land it.

### November 13 Rebase and CI Disposition

Rebased the three published commits from `983fe26a126a35cbb602f70f50fd73d89a62dfcf` onto `4b6b9ec446387b957d000d465c38fe682e6940e3`. All three conflicts overlapped upstream repairs. Preserve the newer generated inventories from `3451c811d0`, the private fallback module fixture from `1dcba9972c`, and the queue-drain regression from `aa54f65509`. Retain this PR's distinct held-lock isolation regression. No new runtime or workflow behavior is introduced; no new ADR is needed.

The old-head full-suite aggregate in run `42420375003` reflects these 14 shard failures:

| Failure | Rebase disposition |
| --- | --- |
| Eleven RAG, paper-search and embeddings setup failures | All eleven check annotations report HTTP 504 before tests. `0de5c3ad22` on dev uses the pinned official PyPI wheel instead of the failing release-asset download. |
| MCP in-app accounting | Dev has the standard-library spies in `c0d9a0a451`; all 63 accounting tests pass with plugin autoload disabled. |
| Admin PostgreSQL restore interruption | The old run failed its offline admission check before restore. Dev's `aa54f65509` finishes connection reset before pool shutdown, preserving the offline guard. The Admin backup shard passed on #3461 head `7044613f96`, whose tree exactly matches the new dev base. |
| AuthNZ integration timeout | The old stack was in the VN quota-lock subprocess, not the earlier missing membership context. The containing test file and the new isolated database teardown regression pass all seven cases on the rebase with real PostgreSQL and no skips. This does not establish the original timeout cause or certify the full shard. The earlier membership context fix is separately on dev in `756e8b0475`. |

The focused rebase run passed 152 cases (23 Logging, ten inventory, 63 MCP accounting, and 56 CI guards). The final Logging, Docs, task and Python setup action checks passed 254 cases, no failures or skips. Scoped Ruff, Black, compilation and whitespace checks passed. Compared with dev, Bandit retained three inherited B105 test literals and reported no new findings or errors (B101 excluded on both sides). Auto-merge was temporarily disarmed before the rebase and is to be restored after the leased push. Keep the requester's summary waiver and all branch protections. Only the new head's checks qualify it for landing; no manual required workflow dispatch or broad-suite pass is claimed.
