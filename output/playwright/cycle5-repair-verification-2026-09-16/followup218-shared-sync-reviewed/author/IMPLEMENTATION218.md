# UAT218 / TASK13260.156 — final source review packet

## Result and scope

**Frozen after independent-review correction: 176 tests passed, zero skipped,45.06s** on the exact eight paths in `owned-manifest.json` (five production, one new permanent regression, two approved ordering-fixture updates). Native acceptance remains parent-owned and pending. No native DB, job, browser, service, config, task/tracker or git mutation was performed.

Base is integrated commit `12812ff859`; all five baseline source snapshots were compared byte-for-byte with that commit. `owned.patch` excludes UAT209/210/217 and216. The ChaCha change is exactly the generic sync reader's SQL-NULL handling; AST comparison after removing that method confirms the rest of the class/module is identical. Neither216's trigger method nor209's ownership methods changed.

## Cause and implementation

Real ChaCha-first PostgreSQL initialization creates the existing sync log with `entity_id`, a three-operation constraint, non-null payload, and no scope columns. Media previously indexed/read/wrote `entity_uuid`, required scope columns, and already supported link/unlink plus optional payload. Reproducing just the failing index was insufficient for the real contract.

The approved helper in `core_media.py` compares PostgreSQL's complete canonical check definitions exactly, preserving whitespace inside literals, and validates known table shapes and runs in the existing initializer transaction. It keeps exactly one supported physical identifier, adds only missing nullable scope columns, replaces only the catalog-proven exact known three-operation check with the five-operation check, and relaxes only payload nullability. Required identity/operation columns remain non-null. The metadata guard accepts both actual application scope types: fresh Media uses INTEGER; its existing v8 migration uses BIGINT. No rename, row backfill, dual write, migration/head change, generic SQL translation or arbitrary constraint drop was added.

The exact identifier-index statement selects the existing identifier. The central Media writer uses the selected fixed column with unchanged bound values; the central reader aliases it to the existing `entity_uuid` response key. SQLite SQL remains unchanged. ChaCha's generic reader leaves legitimate SQL NULL as None, preserving JSON parsing for non-null payloads.

The same helper is invoked before base indexes and from existing post-core setup so current-head reopen also validates/sets the runtime capability. A fresh initialization may therefore inspect metadata twice; its second call performs no compatibility DDL. Runtime read/write calls do not introspect metadata. This deliberately avoids a separate cache-validity state machine and preserves revalidation on repeated initialization.

## Permanent controls (26)

- Both real constructor orders on official disposable PG and separate-file SQLite; actual Media create-with-keyword and cascading delete produce correct delete/unlink events, owner and pagination.
- Both-order actual StudyPack worker → service → source/to_thread → storage, with only model output/provider resolution and DB accessor supplies controlled: exact pack/deck/card/membership/citation owner and sync payload identities.
- Pre-Media committed ChaCha note and pack/membership/citation rows preserve every original field, table OID, owner and grants; subsequent trigger writes work.
- Both-reader optional-null payload and malformed-JSON behavior; no blanket exception swallowing.
- Reopen preserves exact rows, scope values, constraints (including OIDs), relation flags/owner/grants and sync policies.
- Restricted NOSUPERUSER/NOBYPASSRLS role exercises owner, foreign, org and team visibility after both-order reopen. Roles and grants are created/removed only inside official disposable fixtures using the established fixture pattern.
- Controlled late constructor failure rolls back compatibility DDL plus new Media structures; committed rows remain exact; normal subsequent initialization succeeds.
- Direct helper within caller transaction does not commit pending work and does not issue repeated ALTER on a compatible schema.
- Missing/ambiguous identifier, extra custom operation check, wrong scope nullability, nullable identifier and nullable operation reject reopen with the existing constructor's DatabaseError wrapping SchemaError; no partial changes.

## Causal evidence, including failed author attempts

| Receipt | Result | Meaning |
| --- | --- | --- |
| original standalone RED in diagnosis packet |1FAIL/0skip|Real reverse initializer fails at missing identifier index.|
| `uat218-contract-red.redacted.log` |9FAIL/4PASS|Initial actual-contract RED: reverse initializer, nullable payload and unknown-shape acceptance; passing Media-first/SQLite controls.|
| `uat218-first-green.redacted.log` |7FAIL/6PASS|First implementation correctly repaired reverse order; guard initially rejected canonical INTEGER scope columns; two assertions expected unwrapped SchemaError. Both corrected, no product scope expansion.|
| `uat218-expanded-green.redacted.log` |1FAIL/21PASS|Negative fixture tried altering policy-dependent scope type; PostgreSQL rejected fixture DDL before product call. Replaced with a valid negative nullability setup, preserving rejection requirement.|
| `uat218-final-contract-green.redacted.log` |22PASS/0skip|All primary controls pass before two additional required-column guards.|
| `uat218-adjacent-first.redacted.log` |2FAIL/148PASS|Two old ordering mocks lacked the new schema collaborator. Parent approved exact call-position fixture updates; existing assertions retained.|
| `uat218-nullability-guard-red.redacted.log` |2FAIL/22deselected|Nullable required identity/operation shapes accepted before narrow guard; constructor-boundary causal proof.|
| `uat218-frozen-baseline-red.redacted.log` |15FAIL/8PASS/1deselected/0skip|First-release24 tests against exact pre218 five-file source via in-memory import loader; helper-specific test deliberately excluded because helper does not exist in baseline. No source swaps.|
| `uat218-frozen-green.redacted.log` |174PASS/0skip|First-release24 + adjacent150 on then-frozen source/test bytes; four reported warnings (the configured pytest output does not expand their details).|

Older `uat218-final-baseline-red` (13FAIL/8PASS/1deselected) remains retained; it preceded the final two nullability controls. All prior failures are retained rather than overwritten. The final baseline late-failure control fails at the earlier missing-index bug, as expected, before its injected late error; it is not counted as proof of old rollback correctness.

## Independent-review correction

The reviewer ran the first release's174 controls successfully, then proved two actual-PG counterexamples: custom operation literals `cre ate` and `create ` were accepted and their constraints silently rewritten. Whole-definition whitespace stripping had erased literal data. `first-release/` preserves that release's exact eight source/test snapshots and manifest; `review-literal-guard-red.log`, `review-literal-probe.py` and both `operation-literal-*-receipt.json` files retain the reviewer proof.

Equivalent permanent normal-reopen controls failed **2/2** before correction (`uat218-permanent-literal-red.redacted.log`). The approved correction is three expression changes plus one explanatory comment: compare complete `pg_get_constraintdef` output directly with the two known canonical definitions. No parser, SQL rewrite or new production path was introduced. Both unknown constraints and their table/data state now remain untouched on rejection. Final source/test manifest SHA: `424ea3879fea2ad3bdcd01ed266b3439e20c5058b87b460707b5b68f5e10f029`.

Final rerun: `uat218-reviewed-final-green.redacted.log`, **176PASS/0skip45.06s** (26 permanent +150 adjacent). All eight hashes were verified after completion. Final independent verdict remains reviewer-owned.

## Commands

Use a new evidence label for independent runs so receipts are not overwritten. The helper reads private connection configuration internally, enforces required PG, uses official fixtures, and redacts credentials before exposing output.

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat218-reviewed-final-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs \
 tldw_Server_API/tests/DB_Management/test_media_after_chacha_initialization_postgres.py \
 tldw_Server_API/tests/DB_Management/test_media_db_core_media_schema_ops.py \
 tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py \
 tldw_Server_API/tests/DB_Management/test_media_db_sync_log_ops.py \
 tldw_Server_API/tests/DB_Management/test_media_db_sync_utils.py \
 tldw_Server_API/tests/DB_Management/test_media_db_media_lifecycle_ops.py \
 tldw_Server_API/tests/DB_Management/test_media_postgres_runtime_validation.py \
 tldw_Server_API/tests/DB_Management/test_study_pack_shared_sync_schema.py \
 tldw_Server_API/tests/Characters/test_chacha_postgres_sync_log_entity_column.py -q --tb=short
```

First-release24-case causal replay (final two review cases have their separate retained causal RED):

```sh
source .venv/bin/activate
PYTHONPATH="$PWD/.tmp/uat218-repair-20260917${PYTHONPATH:+:$PYTHONPATH}" \
 TLDW_UAT_EVIDENCE_LABEL=uat218-frozen-baseline-red \
 node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs -p uat218_baseline \
 tldw_Server_API/tests/DB_Management/test_media_after_chacha_initialization_postgres.py \
 -k 'not helper_does_not_commit' -q --tb=short
```

## Static verification and attribution

`ruff-reviewed-final.json`:0 findings across all eight paths. Production Bandit:0 findings/0 parse errors; tests Bandit:0/0 with B101 excluded for pytest assertions. Initial single B608 finding was the fixed-name read projection; the exact expression has a justified closed-set annotation. No unchecked string reaches an SQL identifier. Catalog constraint names use backend escaping. `verification-reviewed-final.json` records AST parse of8files, no trailing whitespace, stable hashes and ChaCha-rest AST equality. `diff-reviewed-final-check.txt` is clean. No new dependency/install was needed.

`review-snapshot/` preserves exact final source/tests; `baseline/` and `baseline-manifest.json` preserve causal originals. `owned-manifest.json` identifies source hashes; `evidence-manifest.json` hashes retained receipts and report. The approved design is `../uat218-diagnosis-20260917/DESIGN218.md`; implementation plan `PLAN218.md` records stage status.

## Limits

This repair concerns shared PostgreSQL Media/ChaCha initialization and sync storage, not the separate216 trigger defect. No native reverse-order occurrence or real job retry is claimed. The original quarantined job2 was not mutated. SQLite controls use its normal separate database files. Unknown/custom schema shapes deliberately fail rather than being silently rewritten. Existing application policies/authorization remain unchanged; the restricted-role tests establish the scoped sync-policy behavior only, not a whole-application tenant-security claim. The new direct schema helper is tested in a caller transaction; this change does not alter the wider Media transaction/lifetime implementation.
