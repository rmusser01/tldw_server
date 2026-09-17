# UAT207 / TASK13260.145: bounded default persona bootstrap

## Diagnosis (unchanged production)
Actual GET /api/v1/persona/profiles against the official shared PostgreSQL fixture succeeds twice for Alice (2), then returns 500 twice for Bob (3). The same actual router flow against separate SQLite user files succeeds all four times. Driver exception tracing records the first Bob INSERT as SQLSTATE 23505 / persona_profiles_pkey / INERROR, then the retry SELECT as 25P02 / INERROR. The global primary key research_assistant collides across users; list/get queries already bind user_id. Existing SQL translation correctly converts deleted=0 and is_active=1; 18 unchanged direct query controls pass.

The first diagnostic run contained six harness failures (invalid dependency defaults / invalid delegate loader); these were not product RED. After correcting only the private harness, the cold-route run is 1 FAIL / 1 PASS; the complete driver run is 1 FAIL / 19 PASS, zero skips. Native backend logs retain generic errors only, so the exact SQLSTATE belongs to the official fixture reproduction, not a claimed native driver capture.

## Approved bounded direction
1. In persona.py, retain an existing owned legacy research_assistant (including recognizing a tombstone so it is not bypassed). For a new PostgreSQL default use research_assistant:<user_id>, with an owner-bound lookup. SQLite retains research_assistant. No migration, ID rewrite, foreign-profile reuse, or authorization relaxation.
2. Seed existing default rules against the resolved default ID. Preserve custom profiles, custom legacy data/rules and current fallback behavior. Default-ID constants in actual session/runtime paths are missing-value fallbacks; valid sessions/runtime contexts propagate their persisted persona_id. Add a scoped-ID session policy control.
3. At PersonaStateStore.create_persona_profile only, bound PostgreSQL INSERT failure using the installed psycopg transaction/savepoint context. Execute through the existing backend-aware connection wrapper so UniqueConstraintError remains typed for the existing catch. Do not alter generic execute_query/backend error classification.
4. Preserve transaction ownership. The existing db.transaction and backend.transaction track depth but do not create nested savepoints; db.transaction at depth zero can settle an existing implicit transaction. Therefore blanket nesting does not satisfy this contract. Psycopg uses a savepoint when a transaction already exists. Before opening that local context, an empty explicitly managed/borrowed caller must BEGIN its own outer transaction, so the profile operation cannot commit it. Preserve active-use checkout bookkeeping across the local context; never roll back caller work. Prove IDLE, raw BEGIN, implicit pending write, empty db/backend-managed and borrowed transaction cases before accepting this branch.

## Required permanent controls before GREEN
- Actual profiles/catalog cold routes for two users on one PostgreSQL backend; same-owner repeats and policy IDs; SQLite original ID.
- Existing owned legacy default and custom rules/data preserved; foreign legacy not reused; tombstones not resurrected/bypassed; own name collision fallback; foreign scoped ID not reused.
- Same-owner raced creators converge on the same owned row with a usable transaction.
- Duplicate-ID/name failures map to ConflictError; unexpected SQL errors still propagate; failed owned INSERT does not poison retry.
- Successful and failed nested creation preserve prior pending work and caller rollback for empty managed, backend-managed, raw/implicit and explicit borrowed contexts.
- Actual scoped persona ID flows through catalog/detail/session policy lookup. No provider inference.

## Limits / scope
Three approved production paths: app/api/v1/endpoints/persona.py, app/core/Persona/session_materialization.py and app/core/DB_Management/chacha/persona_state_store.py. The endpoint delegates to the core helper so cold session fallback shares the same correction. Existing policy replacement behavior and all other persona domains remain outside this repair unless a causal new finding requires explicit association. No runtime/browser/service/config changes; parent retains native acceptance gate. Parent separately assigned UAT209 to Notes ownership; this default collision is UAT207.

## Implementation evidence

Permanent causal suite: 22 FAIL / 12 PASS, zero skips. Separate core cold fallback: 1 FAIL / 1 PASS, zero skips. First complete GREEN: 36 PASS, zero skips. Initial permanent harness errors (omitted public user_id field / invalid tool kind) are retained separately; valid RED uses the actual response contract and mcp_tool kind. Native psycopg savepoint context remains inside the captured operation_state.use(), with existing depth metadata unchanged. Empty managed/borrowed contexts begin their caller transaction before entering the insert savepoint; raw pending work already nests. All ten successful/failed caller rollback controls pass.
