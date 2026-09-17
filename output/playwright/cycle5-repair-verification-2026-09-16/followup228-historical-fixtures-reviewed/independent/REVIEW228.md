# UAT228 / TASK13260.169 — independent review

**CLEAR for the frozen two-file test-only unit.** No remaining review finding or source correction requested.

Author manifest SHA256: `667f7507516fc37e0bf4004e71db23a060f25771b35619e4fb75d269f216d3ef` (`.tmp/uat228-repair-20260917/owned-manifest.json`). Both current files equal their snapshots before and after the independent run:

- `test_notes_organization_migration_v55.py`: `b35a71d91b581c50a21cb40de81912a152ecffa8961c4ef53a3510ebee7d41a6`
- `test_notes_organization_migration_v57.py`: `78dcefe108031b1fd2327918e085effa1cd98a4e7792852ccb478dd2e4bba70e`

## Review

The fixtures no longer construct the current schema and then rewrite its version. Their temporary seed-only initializers run the real V4 creation and registered migration steps through 54 or 56, assert the expected version and absence of future storage, and restore the normal initializer before upgrading.

The v54 fixture accounts for the maintained V4 template's later portable-ID additions by checking and removing exactly two column declarations and the two known indexes before database construction. It creates the optional historical folder/membership shape without portable identity, preserving the seeded parent/deleted-child tree and membership. These table columns match the existing backfill contract except for the intentionally absent later identity. This is an explicit fixture reconstruction of the relevant historical shape, not an archived schema dump or a relaxation of production migration guards.

The original active/deleted keyword and collection fixtures, versions/device labels, relationships, stable portable-ID checks, suppression uniqueness, and reopen checks remain. The v57 control additionally executes the exact 56→57 step and compares the complete seeded keyword row before normal current-head reopening. PostgreSQL adapter/SQL assertions elsewhere in the files are unchanged.

Independent AST multiset comparison confirms **11 + 52 original assertions retained**, **16 + 57 current assertions**, and **zero original assertions removed**. The test count remains 42. No production migration or catalog guard changed.

## Causal evidence and limits

The retained pre-task168 baseline receipt has **3 failed / 39 passed**: the pseudo-v54 fixture reaches an existing attachment-registry collision and both pseudo-v56 labels reach existing task-catalog drift. The current task168 source still has the same three failing nodes, with v54 failing earlier at its attempt to drop `sync_id` beneath the new constraint. These paired receipts establish the fixture defect predates the survivor migration. They were inspected and hash-bound, not independently rerun in this review.

The author's two intermediate fixture failures and the separate brace-bearing exception/rollback defect are retained. UAT229 is a distinct production task; this approval neither fixes nor dismisses it.

## Fresh independent checks

After activating the project venv:

```sh
TLDW_UAT_EVIDENCE_LABEL=uat228-sidebar-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v55.py tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v57.py -q --tb=short
```

**42 passed, 0 skipped, 4 warnings, 3.16 seconds.** The official required-PostgreSQL runner was used, but these particular cases exercise SQLite and PostgreSQL adapter controls; this is not a new real-PostgreSQL migration acceptance claim. No native database or application runtime was changed.

Ruff: **0 findings**. Bandit: **0 findings / 0 parse errors**, excluding only B101 for test assertions. Both files compiled. The two source/snapshot hashes stayed fixed. The run began with ChaCha dependency `1e20679f6f7576a02b4e4aed5b12d8bd8eef60d647d7b2da6fc98947f2ae8172`. The separate UAT229 author changed only transaction error-formatting calls afterward: the test log was finalized at 11:04:51.338Z and that file's recorded modification time is 11:06:00.883Z, yielding dependency hash `33f987c9e4c6acc3b1502c9f6e10dc66c6e17929f2258c4d67fa40e87b7fe06b`. This review does not certify the later UAT229 source. Exact command, redacted output, static JSON, independent assertion parity, dependency timing, and input hashes are retained here.

No source/test/task/tracker/git/browser/runtime edits were performed by this reviewer. Stage B and native/full-matrix gates remain separate.
