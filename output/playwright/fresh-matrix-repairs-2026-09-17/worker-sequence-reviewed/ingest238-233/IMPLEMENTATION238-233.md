# UAT238 / UAT233 candidate — blocked only by separately tracked UAT247

Tasks TASK13260.180 and TASK13260.175. Baseline b5b1267d74; root subsequently
committed unrelated units through ea2bc4d04e. This packet owns two production
files and one dedicated test file. Native acceptance remains pending. No runtime,
browser, held matrix database, provider inference, task or git mutations were made.

## Changes and cause

- `services/media_ingest_jobs_worker.py`: the persisted job owner is the only
  authorization identity. Missing/invalid/non-positive owners fail before content
  access; payload `user_id`, org/admin flags and ambient dispatcher authority cannot
  supply privileges. The existing scoped-context helper encloses dispatch and resets
  its token on return/error/cancellation. Generic SQLite DB device labels are unchanged.
- `core/Ingestion_Media_Processing/persistence.py`: copy the caller ContextVars into
  seven existing DB executor submissions (chunk count, AV primary, transcript, visual,
  document primary, attachment and archive child). No pool/RLS/client-label changes.
  The actual restricted-role INSERT previously had empty current_user scope despite
  correct row owner. The candidate records the queued owner with admin=0.
- The document result merge had aliased the processor warning list using dict.update,
  then extended that list with itself. Copy the already installed list instead. This
  preserves producer data, warning order/distinct values, failure status and error.
  No terminal defensive dedup or warning-text normalization was added.

## Causal evidence and fixture corrections

All runs use the official `pg_database_config` fixture and required-PG runner. The
test creates/removes only a restricted LOGIN role inside that disposable database;
normal Media constructors create their schema under that role. Existing FORCE RLS
stays active. No replacement DB setup, RLS bypass or privileged runtime write.

1. `uat238-233-initial-red`: 23 failed / 3 passed. Retained setup mismatch: admin-created
   tables then a grantee cannot refresh normal FTS. Corrected fixture to native-style
   restricted table ownership, not a product fix. SQLite low-level same-file get_by_id
   was also an incorrect foreign-read oracle; public owner-scoped search is the proper
   SQLite boundary. PostgreSQL keeps actual low-level RLS denial too.
2. `uat238-233-causal-red`: 20 failed / 6 passed. Real PG persistence and warning failures.
3. `uat238-insert-scope-red`: 2 PG failed / 2 SQLite passed. Exact INSERT settings:
   owner='', admin='0', rolsuper=false, rolbypassrls=false for queued owners1 and2.
4. `uat238-lifetimes-red` and `uat238-lifetimes-corrected-red`: preserved interim
   controls. Two fault-fixture mistakes were corrected (keyword method/form shape,
   then positional connection argument). Production cancellation already converts a
   persistence interruption to Warning; the control now drives actual Jobs cancellation
   and verifies its existing `{}` result plus scope retirement, rather than expanding
   the cancellation contract. Lost executor scope remains the causal assertion.
5. Final in-memory baseline replay: **38 failed / 10 passed / 0 skips**, 13.05s.
   `replay_baseline.py` loads retained baseline bytes without replacing checkout files.
   It covers the final tests, including all seven submissions and error warning controls.
6. Candidate final focused run (`uat238-233-expanded-green` filename notwithstanding):
   **47 passed / 1 failed / 0 skips**, 9.07s. The single remaining failure is UAT247;
   this is explicitly NOT a wholly green run.
7. Existing eight-file adjacent run: **63 passed / 0 skips**, 68.46s. Worker, cancellation,
   chunk consistency, metadata, safe paths, collection synchronization, ingestion Jobs
   and audio transcript integration. These mocks/fixtures do not claim real-model success.

## UAT247 dependency: preserved failing cross-owner control

`test_sequential_jobs_restore_executor_scope[postgresql]` remains active and unchanged.
After owner1 succeeds, normal Media constructor for owner2 rewinds its own serial using
RLS-filtered MAX(id). Before each actual INSERT media_id_seq is `{last_value:1,is_called:false}`.
The second INSERT has correct owner2/admin0 and returns driver SQLSTATE **23505**, with
the failing constraint classified as the Media primary key. No raw driver payload is
attached or logged by the test; only SQLSTATE and a boolean classification are recorded.

`uat238-reuse-sqlstate-proof`: 1 PG failed / 1 SQLite passed. Earlier `reuse-probe` and
`reuse-sequence-proof` are retained. This differs from prior UAT147's cross-component
table allowlist: only the Media-owned serial is rewound here. Parent associated UAT247
with TASK13260.189; another author owns its sequence implementation/tests. This packet
does not weaken/skip the control, change sequence code, or claim multi-owner acceptance.

## Validation and limits

- Ruff on exact logical production/test filenames: **27 baseline / 27 current**, exact
  file/rule/message multiset, zero additions/removals. The new test has no diagnostics.
  Initial baseline-path lint used the private copy's location and therefore different
  per-file rules; retained as preliminary only. `ruff-logical-*` / comparison supersede it.
- Bandit production: zero findings/errors. Dedicated test: zero findings/errors with
  only assert checks B101 excluded as test assertions. Compile: 3 paths pass. Diff check passes.
- Callback cancellation cannot stop an already running Python executor thread. Its
  copied scope lasts until that operation exits and does not leak to reused threads or
  the caller. Existing Jobs cancellation/result semantics remain intact.
- SQLite per-file legacy client labels remain supported; HTTP-style explicit org/team
  content scope is preserved across the persistence boundary. Job scope is personal-only;
  arbitrary payload memberships/admin flags are never trusted.
- Optional Claims persistence lives in a different module and is not claimed repaired.
  Provider extraction/model calls are controlled; no native acceptance is claimed.
- Runtime GUCs are observed at the real Media INSERT. Error/rollback, foreign/unscoped
  denial, prior-scope restoration, direct child writes, transcript/visual/count callbacks,
  and warning/failure preservation are tested. Original 238 native RLS evidence remains
  the frozen matrix packet, separate from these fixture reproductions.

## Repeat commands

Activate `.venv` first. Required fixture runner (never targets held matrix databases):

```sh
TLDW_UAT_EVIDENCE_LABEL=uat238-review node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/DB_Management/test_media_ingest_worker_content_scope.py -q --tb=short
```

Before the separate UAT247 fix, expect47 pass/1 fail. After its frozen handoff, rerun
the complete48 and retain all source hashes. Exact adjacent and baseline commands are
in `receipts/*-command.json`. Root review, UAT247 integration and native acceptance follow.

## Frozen UAT247 combined replay

The full required-PG worker suite now passes **48 / 0 skips**, 9.64 seconds, against the separate frozen UAT247 helper `20a6d438ee5d4406e6e0e008efcda0347d7007742a93b444a19f8daf9ef05d0b` (manifest `ea50ce2285a2d4870b9b95bc422983ee265df96654f205bf148a7efc0e7ec254`). All three owned source hashes remain unchanged. `with247-replay.json` binds the exact safe command/log and original/retained hashes. Earlier 47/1 and SQLSTATE23505 evidence remains preserved. This is the dependency-resolved automated result; independent review/integration and native acceptance remain parent-owned.
