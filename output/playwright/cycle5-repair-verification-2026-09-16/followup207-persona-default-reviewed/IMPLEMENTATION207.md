# UAT207 / TASK13260.145 — frozen review packet

## Outcome
The shared PostgreSQL default bootstrap now resolves an owner-specific default ID, preserving any existing owned legacy research_assistant row and all SQLite IDs. Profile insertion retains caller transaction ownership and cleans up its own failed insert, so a duplicate becomes ConflictError and does not poison retry. Implementation is frozen and ready for independent review; native acceptance remains pending.

## Cause and attribution
The official actual-router fixture reproduced Alice 2 twice200 then Bob3 twice500 on one PostgreSQL backend, with both SQLite users twice200. Driver tracing captured 23505 / persona_profiles_pkey at Bob's INSERT and 25P02 on ordinary retry SELECT. The profile primary key is global while bootstrap used the same research_assistant ID for every user. Existing deleted/is_active query translation works (18 direct controls passed unchanged). Exact driver SQLSTATE is fixture evidence; the native log retained generic failures only.

Three production paths and one new permanent test are the entire source scope. `owned-manifest.json` records baseline, final and byte-identical snapshot attribution. ChaChaNotes_DB.py, generic query/backend wrappers, schema, auth, frontend, native services and browser were not changed.

## Implementation
- Existing core `ensure_default_persona_profile` checks the owned legacy ID including tombstones. New PostgreSQL users receive research_assistant:<user_id>; SQLite remains research_assistant. All lookups retain user_id. Foreign legacy/scoped IDs cannot be reused or overwritten. Policies seed against the resolved default ID; customized profiles/rules and fallback behavior remain intact.
- The endpoint private helper delegates to that existing core helper, covering both profiles/catalog and the already-existing cold session fallback. `source-caller-inventory.md` documents actual catalog/selection/session/runtime propagation. No runtime fallback constant was broadly rewritten.
- PostgreSQL create uses the existing backend-aware connection wrapper inside psycopg's local transaction/savepoint, preserving typed UniqueConstraintError for the existing classifier. IDLE standalone insertion is committed/rolled back locally. Pre-existing transactions retain pending writes; empty managed/backend/borrowed callers BEGIN their own transaction before the insert savepoint. The captured operation state's active-use scope protects the checkout through the local transaction. Existing depth metadata is unchanged. SQLite execute behavior is unchanged.
- A connection that was already aborted by its caller is not blanket-rolled back by this repair. No generic transaction or exception policy change.

## Verification
| Receipt | Result |
|---|---|
| `uat207-causal-permanent-red` | 22 failed / 12 passed / 0 skipped, before implementation |
| `uat207-core-default-red` | 1 failed / 1 passed / 0 skipped, actual cold core materialization |
| `uat207-first-green` | 36 passed / 0 skipped, 54.37s |
| `uat207-adjacent-green` | 121 passed / 0 skipped, 92.45s |
| `uat207-lifetime-adjacent` | 26 passed / 0 skipped, 46.14s |

All tests use the official required-PG runner and isolated official fixtures; no live DB setup or model inference. The 36 primary controls include actual profiles/catalog/detail for two owners, repeat IDs/policy ownership, retained custom legacy data/rules, tombstones, foreign reservations, same-owner raced profile creators, duplicate ID/name retry, unexpected SQL failure, ten successful/failed raw-pending/managed/backend/borrowed rollback cases, scoped session policies and cold materialization, unchanged list/get/batch flags/owner/pagination. The race assertion covers profile identity convergence and usable creation; it does not redesign concurrent policy replacement.

Adjacent suites: Persona profiles API, sessions, live control, existing PG/SQLite persona-memory filters, explicit/borrowed HTTP transaction controls and operation-scope controls. Total final verification:183 passes across the three commands, zero skips. Existing warnings are retained in logs.

Ruff:5 baseline/5 current, identical diagnostic signatures, no additions; new test0. Bandit:0 findings/0 errors across3 production files and new test (B101 excluded for test assertions). Four files compile; no trailing whitespace; all frozen source/snapshot hashes still match after tests.

Initial diagnostic/permanent harness attempts are retained and explicitly excluded from causal counts: invalid FastAPI override default arguments, invalid delegate loader, omitted public user_id response field and invalid policy rule kind. The corrected RED receipts above are the implementation's causal baseline.

## Independent command
From the repository root:

```sh
source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=uat207-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_persona_default_profile_backends.py -q --tb=short
```

Exact adjacent commands are retained in their `*-command.json` receipts. The runner requires PostgreSQL and redacts its DSN/password automatically. Additional structural secret-pattern scan found no unredacted DSN/JWT/long authorization header patterns; this is not a claim to have inspected native credentials. No credentials/session material or native logs were copied into this packet.

## Remaining gate
Parent owns independent review, integration and genuine native retry after the accepted backend restart. No native pass, task completion, clean full HEAD, full Persona domain audit, raw-SQL RLS isolation, or ordinary-role full bootstrap certification is claimed here. Existing global legacy profile IDs are preserved without migration.
