# PR #3323 Late CI Follow-up

**Task:** TASK-15480 (original local evidence run ID: TASK-15466)
**Base:** `dfac3ca7f34a296a2dc3a3fbc340efdd1a0bae5c` (`dev`)
**Goal:** Validate all four late CI failures and publish only confirmed, unowned repairs.
**ADR check:** No new ADR is required for restoring existing lock configuration and generated-artifact contracts. ADR-059 governs task editing; ADR-085 governs the separately owned AuthNZ fixture repair.

## Constraints

- Preserve production behavior unless a regression proves it incorrect.
- Do not weaken assertions, increase thresholds, add dependencies, or bypass CI.
- Independently validate the overlapping AuthNZ and MCP fixes in PR #3406; link their ownership instead of duplicating pending changes.
- The human Change summary waiver for #3323 does not extend to this PR. Publish as draft pending its own summary and hosted qualification.
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
**Status:** In Progress

- [ ] Run affected selections and record every failure, skip and environment limitation without relabeling them green.
- [x] Review source/test scope and baseline-compare any inherited security findings.
- [ ] Commit code and canonical task records together; open draft PR against dev with all four dispositions and #3406 dependency.

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
