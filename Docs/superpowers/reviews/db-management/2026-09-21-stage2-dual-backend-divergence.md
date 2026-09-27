# Stage 2 (2026-09-21): Dual-Backend Divergence in the Two Unreviewed God Files

## Scope

`PromptStudioDatabase.py` (7,426 LOC) and `ChaChaNotes_DB.py` (45,292 LOC, 408 commits/12mo) — the two
files the April pass inventoried and skipped. Both implement the same logical database twice, once for
SQLite and once for PostgreSQL, behind a shared façade. This stage asks one question of each pair: **has
the duplicate already drifted, and does the test suite cover both sides?**

The 2026-09-21 cross-user isolation audit established the SQLite/PostgreSQL split is only tested on
SQLite. Everything below is an instance of that with a named consequence, not a hypothetical.

## Code Paths Reviewed

`PromptStudioDatabase.py`
- `PromptStudioRowAdapter (275-317)`, `PromptStudioBackendCursorAdapter (320-375)`,
  `PromptStudioBackendCursorWrapper (378-454)`, `PromptStudioBackendConnectionWrapper (457-483)`,
  `PromptStudioBackendManagedTransaction (486-510)`, `BackendPromptStudioDatabaseBase (517-719)`
- `_BackendPromptStudioDatabase (722-3843)` — PostgreSQL implementation, 3,121 lines
- `_SQLitePromptStudioDatabase (3848-7141)` — SQLite implementation, 3,293 lines
- `PromptStudioDatabase (7144-7426)` — `*args/**kwargs` delegating façade, 282 lines
- `_PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS (58-76)`
- `_BackendPromptStudioDatabase.list_optimization_iterations (2420-2470)` and `(2472-2517)`
- 28 inline retry blocks, all inside `_SQLitePromptStudioDatabase` (full line list in Finding 5)

`ChaChaNotes_DB.py`
- `CharactersRAGDB (730-44806)` — one class, 44,076 lines, 800 statically defined methods
- `_CURRENT_SCHEMA_VERSION = 68 (755)`, `_POSTGRES_SCHEMA_VERSION = 70 (756)`
- SQLite base DDL: `character_cards.name TEXT UNIQUE NOT NULL (1027)`,
  `decks.name TEXT UNIQUE NOT NULL (1791)`
- `_sqlite_linear_migration_steps (8536-8609)` — step table, last entry `(67, "_migrate_from_v67_to_v68")`
- `_migrate_from_v67_to_v68 (17700-17714)` (SQLite), `_migrate_from_v67_to_v68_postgres (17733-17779)`,
  `_migrate_from_v68_to_v69_postgres (17781-17826)`, `_migrate_from_v69_to_v70_postgres (17716-17731)`
- `_initialize_schema_sqlite (20349-21169)`, ladder tail `(21100-21114)`
- `_initialize_schema_postgres (24881-25300)`, ladder tail `(25125-25141)`
- `BackendCursorAdapter (448-483)`, `BackendCursorWrapper (510-586)`,
  `BackendConnectionWrapper (589-642)`, `BackendManagedTransaction (660-726)`
- `_delegate_store_method (44809-44815)`, delegation tail `(44862-45218)`
- 34 `_sqlite`/`_postgres` method pairs enumerated in
  `2026-09-21-stage2-chachanotes-backend-pair-inventory.txt`

Siblings compared against: `Evaluations_DB.py:89-121`, `Workflows_DB.py:266-360`,
`media_db/runtime/rows.py:10-102`.

## Tests Reviewed

| Test file | Protects | Downgrades risk? |
| --- | --- | --- |
| `tests/prompt_studio/test_database.py` | real temp-SQLite CRUD over `PromptStudioDatabase` (34 PromptStudio test files total by import-grep) | Partially. It is the only real-DB PromptStudio test, and it exercises none of the four diverged call surfaces. |
| `tests/prompt_studio/integration/test_api_endpoints.py:369-380` | `list_optimizations` pagination defaults | **No.** Substitutes a stub class defining `list_optimizations(self, *_args, **_kwargs)`. |
| `tests/prompt_studio/unit/test_optimization_endpoint_error_mapping.py:32-37,327-380` | `list_optimizations` error mapping | **No.** Same stub shape. |
| `tests/prompt_studio/integration/test_postgres_tenant_session_reuse.py`, `test_connection_pool_returns.py`, `test_concurrency_multiprocessing.py`, `unit/test_prompt_studio_tenant_session.py`, `unit/test_mcts_cache_tenant_identity.py`, `unit/test_optimization_atomic_transitions.py`, `unit/test_prompt_studio_deps.py`, `tests/DB_Management/test_prompt_studio_sync_log_backends.py`, `tests/DB/integration/test_pg_rls_apply.py` | 9 of the 34 PromptStudio test files touch PostgreSQL | Partially. They cover session/tenant/pool plumbing and one sync-log family — not the 59 paired business methods. |
| `tests/DB_Management/test_deck_owner_name_migration_postgres.py`, `test_character_owner_name_migration_postgres.py` | per-owner name uniqueness | **No.** PostgreSQL-only by filename and by content; no SQLite sibling exists. |
| `tests/DB_Management/test_local_keyword_merge_survivor.py`, `test_local_keyword_survivor_migration.py` | `merged_into_sync_id` on `keywords` (SQLite) **and** `chacha_keywords` (PostgreSQL) | Yes for that column. This is the both-backends pattern the deck/character migrations did not follow. |
| ChaChaNotes overall: 478 test files by import-grep, 148 of which mention PostgreSQL | broad | Reachability, not coverage; and 4 of the 1,086 collected `tests/ChaChaNotesDB` tests do not even import locally (`psycopg`, `hypothesis` absent). |

## Validation Commands

```
$ python3 - <<'PY'   # AST census, PromptStudioDatabase.py
...
CLASS _BackendPromptStudioDatabase: lines 722-3843 (3121 lines)
CLASS _SQLitePromptStudioDatabase: lines 3848-7141 (3293 lines)
CLASS PromptStudioDatabase: lines 7144-7426 (282 lines)

== methods defined in BOTH backend+sqlite classes: 59
== of those, also redeclared on facade PromptStudioDatabase: 43
== LOC in the 59 paired methods: backend=2559 sqlite=3006 total=5565
  total mismatched signatures: 7/59
== duplicate defs WITHIN a class (shadowed) ==
  _BackendPromptStudioDatabase.list_optimization_iterations: [(2420, 2470), (2472, 2517)]
== methods ONLY in backend class == [... 'list_optimizations']
PY
```

Runtime reproduction against the **default** backend (`PromptStudioDatabase(tmp, "test-client")`):

```
$ python3 - <<'PY'
import tempfile, os
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
db = PromptStudioDatabase(os.path.join(tempfile.mkdtemp(),'ps.db'), 'test-client')
...
PY
impl: _SQLitePromptStudioDatabase
list_optimizations RAISED: AttributeError '_SQLitePromptStudioDatabase' object has no attribute 'list_optimizations'
project: {'id': 1, 'name': 'p1'}
get_prompt(include_deleted=True) RAISED: TypeError _SQLitePromptStudioDatabase.get_prompt() got an unexpected keyword argument 'include_deleted'
delete_signature(1, True) RAISED: TypeError _SQLitePromptStudioDatabase.delete_signature() takes 2 positional arguments but 3 were given
create_bulk_test_cases(client_id=) RAISED: TypeError _SQLitePromptStudioDatabase.create_bulk_test_cases() got an unexpected keyword argument 'client_id'
```

```
$ grep -rn '(2 \*\* attempt)' tldw_Server_API/app/core/DB_Management --include='*.py' | cut -d: -f1 | sort | uniq -c
  28 .../PromptStudioDatabase.py
$ python3 -c "..."   # regex scan of _BackendPromptStudioDatabase body (lines 722-3843)
# for max_retries|retry|backoff|serialization|deadlock|40001|time.sleep :
2682         max_retries: int = 3,          <- a column default, not a retry loop
2694/2706    payload, max_retries, client_id
3055/3062/3069  retry_job_record / retry_count column
# => zero retry loops, zero serialization-failure handling in the PostgreSQL implementation
```

```
$ grep -n '_CURRENT_SCHEMA_VERSION\|_POSTGRES_SCHEMA_VERSION' tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py | head -2
755:    _CURRENT_SCHEMA_VERSION = 68  # Schema v68 retains local keyword merge survivors
756:    _POSTGRES_SCHEMA_VERSION = 70
$ python3 - <<'PY'   # _sqlite/_postgres pair census on CharactersRAGDB
34 _sqlite/_postgres method PAIRS on CharactersRAGDB
TOTAL LOC in paired sqlite/postgres methods: 7301  (16.1% of file)
UNPAIRED _sqlite only (36)
UNPAIRED _postgres only (20): [... '_migrate_from_v54_to_v55_postgres' ... '_migrate_from_v68_to_v69_postgres', '_migrate_from_v69_to_v70_postgres' ...]
PY
$ grep -rn 'UNIQUE (client_id, name)' tldw_Server_API --include='*.py'
.../ChaChaNotes_DB.py:17752:            "UNIQUE (client_id, name)",
.../ChaChaNotes_DB.py:17799:            "ALTER TABLE decks ADD CONSTRAINT decks_client_id_name_key UNIQUE (client_id, name)",
$ ls tldw_Server_API/tests/DB_Management | grep -i 'owner_name'
test_character_owner_name_migration_postgres.py
test_deck_owner_name_migration_postgres.py
$ python3 - <<'PY'   # setattr delegation tail
setattr calls: 0          # they are for-loops, not literal setattr lines in the grep window
quoted method names in for-loops: 273
PY
$ grep -c 'setattr(' <tail of ChaChaNotes_DB.py from :44809>   # 273 names across the for-loops at 44862-45218
```

## Findings

### FINDING db-management-1

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py:_BackendPromptStudioDatabase.list_optimizations (2122-2183)
             tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py:_SQLitePromptStudioDatabase (3848-7141) — method ABSENT
             tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py:PromptStudioDatabase.list_optimizations (7378-7379)
             tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_optimization.py:list_optimizations (964-1010), call at :986
canonical:   NONE
destination: n/a — this is a defect, not a consolidation
knowledge:   "the two implementations expose the same method set". The facade at :7378 hard-declares
             list_optimizations and delegates unconditionally; only one implementation has it.
             _SQLitePromptStudioDatabase inherits from PromptsDatabase, which does not define it either
             (verified MRO: ['_SQLitePromptStudioDatabase', 'PromptsDatabase', 'object']).
scenario:    ADR-020 makes SQLite the DEFAULT content-storage mode. On a default deployment,
             GET /api/v1/prompt-studio/projects/{project_id}/optimizations calls db.list_optimizations(...)
             at prompt_studio_optimization.py:986. The facade delegates to _SQLitePromptStudioDatabase,
             which raises AttributeError. AttributeError is a member of _OPTIMIZATION_NONCRITICAL_EXCEPTIONS,
             so the handler at prompt_studio_optimization.py:1019 converts it to HTTP 500 "Failed to list optimizations" — every time,
             for every project, with no distinguishing log line. Reproduced above verbatim.
impact:      a public GET endpoint is unconditionally broken on the default backend and returns a
             generic 500 that reads like a transient fault.
tests:       (import-grep reachability, not measured coverage) tests/prompt_studio/integration/test_api_endpoints.py:369,
             tests/prompt_studio/unit/test_optimization_endpoint_error_mapping.py:32,37,327-380 —
             both define a stub `list_optimizations(self, *_args, **_kwargs)` and therefore pass while the
             real default-backend path raises. tests/prompt_studio/test_database.py drives the real SQLite
             impl but never calls list_optimizations.
effort:      cheap to fix (port the ~60-line method, or make the façade raise NotImplementedError
             explicitly); moderate to prevent recurrence (see Actions item 1).
owner-only:  yes — the confirming endpoint lives under tldw_Server_API/app/api/v1/**
confidence:  confirmed (runtime-reproduced)
```

### FINDING db-management-2

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:CharactersRAGDB._CURRENT_SCHEMA_VERSION (755)
             tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:CharactersRAGDB._POSTGRES_SCHEMA_VERSION (756)
             .../ChaChaNotes_DB.py:_migrate_from_v67_to_v68 (17700-17714)            # SQLite: adds keywords.merged_into_sync_id
             .../ChaChaNotes_DB.py:_migrate_from_v67_to_v68_postgres (17733-17779)   # PG: character_cards UNIQUE(client_id,name)
             .../ChaChaNotes_DB.py:_migrate_from_v68_to_v69_postgres (17781-17826)   # PG: decks UNIQUE(client_id,name)
             .../ChaChaNotes_DB.py:_migrate_from_v69_to_v70_postgres (17716-17731)   # PG: chacha_keywords.merged_into_sync_id
             .../ChaChaNotes_DB.py:_sqlite_linear_migration_steps (8536-8609), last step (67, "_migrate_from_v67_to_v68")
             .../ChaChaNotes_DB.py:_initialize_schema_sqlite ladder tail (21100-21114)
             .../ChaChaNotes_DB.py:_initialize_schema_postgres ladder tail (25125-25141)
             .../ChaChaNotes_DB.py base DDL: character_cards.name TEXT UNIQUE NOT NULL (1027)
             .../ChaChaNotes_DB.py base DDL: decks.name TEXT UNIQUE NOT NULL (1791)
canonical:   NONE
destination: n/a
knowledge:   two things diverged at once. (a) The integer stored in db_schema_version means a DIFFERENT
             schema on each backend: SQLite v68 == "keywords has merged_into_sync_id"; PostgreSQL v68 ==
             "character_cards has UNIQUE(client_id,name)" and nothing about merged_into_sync_id (that
             arrives at PG v70). (b) The per-owner name-uniqueness POLICY introduced by the two PG
             migrations was never applied to SQLite, where both tables keep the original base-DDL
             GLOBAL UNIQUE on name alone.
scenario:    Deck naming. On PostgreSQL after v69, `decks` is UNIQUE(client_id, name): two clients may
             each hold a deck named "Spanish". On SQLite, `decks.name` is UNIQUE globally (:1791) and no
             SQLite migration ever relaxes it (grep for UNIQUE (client_id, name) returns only the two
             PostgreSQL ALTER statements). So (i) a user with two devices accumulates two "Spanish" decks
             in shared PostgreSQL, and the first sync-down into their SQLite ChaChaNotes.db fails on
             `UNIQUE constraint failed: decks.name`; (ii) application code that relies on the database
             to reject a duplicate deck/character name gets that rejection on PostgreSQL and silently
             accepts the duplicate on the default SQLite backend. Same shape for character_cards (:1027).
impact:      a documented schema-version integer is no longer a cross-backend identity, and a uniqueness
             policy is enforced on one backend only. This is the exact failure mode the 2026-09-21
             cross-user isolation audit named — divergence that only shows up off SQLite, or only on it.
tests:       (import-grep reachability, not measured coverage) tests/DB_Management/test_deck_owner_name_migration_postgres.py,
             tests/DB_Management/test_character_owner_name_migration_postgres.py — PostgreSQL only; `ls`
             over tests/DB_Management shows no SQLite sibling. Contrast tests/DB_Management/
             test_local_keyword_merge_survivor.py and test_local_keyword_survivor_migration.py, which DO
             assert the same column on both `keywords` and `chacha_keywords` — the pattern to copy.
effort:      moderate. The SQLite side needs a real table rebuild (SQLite cannot drop a column-level
             UNIQUE), plus a decision on whether SQLite should adopt the policy at all given it is a
             per-user file. The cheap half — making the two version constants explicit about meaning
             different things, or renumbering so they agree — is a small diff.
owner-only:  no
confidence:  confirmed (the constant divergence and the missing SQLite uniqueness migration are both
             grep-verified); probable-risk (the sync-down failure, which depends on the Sync v2 path
             actually writing decks into the SQLite mirror)
```

### FINDING db-management-3

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       PromptStudioDatabase.py:_BackendPromptStudioDatabase.get_prompt (2521-2536)
               vs _SQLitePromptStudioDatabase.get_prompt (5330-5357)          # include_deleted missing
             PromptStudioDatabase.py:_BackendPromptStudioDatabase.delete_signature (1768-1799)
               vs _SQLitePromptStudioDatabase.delete_signature (4697-4757)     # positional vs keyword-only
             PromptStudioDatabase.py:_BackendPromptStudioDatabase.create_bulk_test_cases (3700-3724)
               vs _SQLitePromptStudioDatabase.create_bulk_test_cases (6894-6951)  # client_id missing
             PromptStudioDatabase.py:_BackendPromptStudioDatabase.list_evaluations (1970-2032)
               vs _SQLitePromptStudioDatabase.list_evaluations (4988-5072)     # keyword-only vs positional
             PromptStudioDatabase.py:_BackendPromptStudioDatabase._format_test_case (3436-3437)
               vs _SQLitePromptStudioDatabase._format_test_case (5322-5325)    # (row) vs (cursor, row)
             PromptStudioDatabase.py:_BackendPromptStudioDatabase._row_to_dict (1137-1171)
               vs _SQLitePromptStudioDatabase._row_to_dict (5235-5269)         # row optional vs required
             PromptStudioDatabase.py:_BackendPromptStudioDatabase.__init__ (781-800)
               vs _SQLitePromptStudioDatabase.__init__ (3861-3880)             # tenant_user_id/backend/config absent
             façade passthroughs: PromptStudioDatabase.create_signature..list_optimization_iterations (7212-7426)
canonical:   NONE
destination: n/a
knowledge:   the façade's `*args: Any, **kwargs: Any` passthrough signatures assert that both
             implementations accept the same call. 7 of 59 paired methods do not, and because the
             façade erases the signature, neither mypy nor the IDE can see it.
scenario:    Any caller written against the PostgreSQL surface breaks on the default backend at runtime,
             not at type-check time. Reproduced verbatim above: `db.get_prompt(1, include_deleted=True)`
             -> TypeError; `db.delete_signature(1, True)` -> TypeError; `db.create_bulk_test_cases(pid,
             cases, client_id="cx")` -> TypeError. `_format_test_case` differs in ARITY, so any shared
             helper that calls it polymorphically is wrong on one backend by construction.
impact:      the façade actively converts a compile-time class of error into a production 500. Combined
             with `_PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS (58-76)`, which lists both `AttributeError` and
             `TypeError`, several call sites swallow these into benign-looking defaults instead.
tests:       (import-grep reachability, not measured coverage) 34 PromptStudio test files; none exercise
             any of the four diverged call surfaces. tests/prompt_studio/test_database.py is the only
             real-DB test.
effort:      cheap per method; the durable fix is an ABC or Protocol both classes must satisfy, plus a
             lint test asserting signature equality — see Actions item 1.
owner-only:  no
confidence:  confirmed (runtime-reproduced for 3 of 7; the other 4 are AST-verified signature diffs)
```

### FINDING db-management-4

```
axis:        duplication
class:       divergent-copies
severity:    High
sites:       PromptStudioDatabase.py:_BackendPromptStudioDatabase (722-3843), 3,121 lines
             PromptStudioDatabase.py:_SQLitePromptStudioDatabase (3848-7141), 3,293 lines
             PromptStudioDatabase.py:PromptStudioDatabase (7144-7426), 282-line delegating façade
             59 paired method names, 5,565 LOC (backend 2,559 / sqlite 3,006) — full pair table with
             exact line ranges in 2026-09-21-stage2-promptstudio-pair-inventory.txt
             43 of the 59 are additionally redeclared on the façade, giving three copies of the name
canonical:   NONE
destination: split the file the way core/DB_Management/media_db/ was split. That migration is the
             template: Media_DB_v2.py (121 commits of churn) no longer exists and is now
             media_db/{api.py, errors.py, constants.py, repositories/, runtime/, schema/}. The
             equivalent here is prompt_studio_db/ with one module per aggregate (projects, prompts,
             signatures, evaluations, optimizations, test_cases, job_queue), each owning ONE
             backend-neutral implementation over the existing DatabaseBackend abstraction, with
             backend-specific SQL isolated to a dialect module — not a whole second class.
knowledge:   every business rule in Prompt Studio — soft-delete predicates, pagination shape, sync-event
             emission, idempotency keys, job lease semantics — is written twice with parallel SQL. A
             change to any one of them must be made in two places that are 3,000 lines apart, and
             nothing in the file, the type system, or CI checks that both were changed. Findings 1, 3
             and 5 are all instances of "one of the two was changed".
impact:      change amplification is 2x on the largest business-logic surface in the module, and the
             observed drift rate is already 1 missing method + 7 signature mismatches + a retry policy
             present on one side only. This is the largest single-file redundancy in the repository.
tests:       (import-grep reachability, not measured coverage) 34 test files. 9 touch PostgreSQL, all of
             them plumbing-level (tenant session, pool return, RLS apply, concurrency, sync-log). The 59
             paired business methods have no cross-backend parity test.
effort:      expensive — needs a design doc, an ADR, and a staged plan. See Actions.
owner-only:  no (core only), though a parity lint test under tests/lint/ touches shared CI config
confidence:  confirmed
```

### FINDING db-management-5

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       28 inline retry loops, ALL inside _SQLitePromptStudioDatabase (3848-7141):
               PromptStudioDatabase.py:4165, :4220, :4278, :4313, :4401, :4490, :4528, :4582, :4605,
               :4751, :4830, :4918, :5032, :5066, :5159, :5219, :5507, :5647, :6027, :6159, :6263,
               :6267, :6329, :6452, :6675, :6772, :6886, :6946
             ZERO retry loops in _BackendPromptStudioDatabase (722-3843) — regex-verified
             module-level siblings: transaction_utils.py:db_transaction (31-97), backoff at :71 (delay
               0.1 * 2**n, async, no jitter); Workflows_DB.py:_sqlite_retry_execute (1659-1671, backoff at :1666,
               0.05 * 2**n, max_tries param); Workflows_DB.py:_sqlite_retry_commit (1673-1685, backoff at :1680,
               0.05 * 2**n, cap hardcoded to 4)
canonical:   NONE inside DB_Management
destination: core/DB_Management/retry_policy.py — single responsibility: "classify a storage error as
             retryable for a given backend, and produce the next backoff delay". One decorator/context
             manager parameterised by (backend, predicate, max_attempts, base_delay). Explicitly NOT
             Utils.py and NOT http_client.py.
knowledge:   "which storage errors are transient, and how long to wait". Three separate pieces of that
             knowledge have already drifted inside the one file:
               - :4162 tests `"database is locked" in str(e)` with NO `.lower()`; the other 27 use
                 `str(e).lower()`. One case-sensitive predicate among 28.
               - :6265 caps attempts with a hardcoded `attempt < 4` while the enclosing loop at :6185 is
                 `for attempt in range(5)`; every sibling uses `attempt < max_retries - 1`.
               - 5 of 28 omit the jitter term `* (0.5 + random.random())`: :4751, :6675, :6772, :6886,
                 :6946. The other 23 have it.
scenario:    The PostgreSQL side has no retry at all. Prompt Studio's job queue does read-then-conditional-
             update work (acquire_next_job, update_job_status, renew_job_lease, retry_job_record). On
             SQLite, concurrent workers hitting `database is locked` are retried up to 5 times with
             jittered backoff and the operation succeeds. On PostgreSQL the same contention surfaces as a
             serialization failure / deadlock (SQLSTATE 40001 / 40P01) that propagates straight out of
             _BackendPromptStudioDatabase to the caller. Two workers polling the same queue therefore
             behave completely differently per backend: transparently retried on SQLite, hard-failed on
             PostgreSQL. Separately, the 5 non-jittered SQLite copies produce synchronized retry waves
             across workers — a thundering herd the other 23 copies were specifically written to avoid.
impact:      the module's entire contention policy exists on one of two backends, and has three internal
             inconsistencies. Availability of the Prompt Studio job queue is backend-dependent.
cost-driver: n/a (correctness axis). Secondary efficiency note: the non-jittered copies scale their
             collision probability with worker count.
tests:       (import-grep reachability, not measured coverage) transaction_utils has 1 test file;
             tests/prompt_studio/integration/test_concurrency_multiprocessing.py is the only concurrency
             test and it does not assert retry behaviour per backend. No test asserts the PostgreSQL path
             retries anything.
effort:      moderate. Extracting the shared helper is mechanical; deciding the PostgreSQL retry
             predicate (which SQLSTATEs, and whether a retry is safe mid-transaction) needs design.
owner-only:  no
confidence:  confirmed (the 28 sites, the three internal divergences, and the absence of retry on the
             PostgreSQL side are all directly verified); probable-risk (the PostgreSQL 40001 scenario
             depends on the deployed isolation level)
```

### FINDING db-management-6

```
axis:        encapsulation
class:       n/a
severity:    Medium
sites:       ChaChaNotes_DB.py:CharactersRAGDB (730-44806) — one class, 44,076 lines, 800 statically
               defined methods, 0 shadowed names
             ChaChaNotes_DB.py:_delegate_store_method (44809-44815) and the for-loops at (44862-45218)
               attaching 273 further method names via setattr at import time
             34 _sqlite/_postgres method pairs totalling 7,301 LOC (16.1% of the file); largest:
               _initialize_schema_sqlite (20349-21169, 821) vs _initialize_schema_postgres
               (24881-25300, 420); _ensure_study_pack_schema_sqlite (23140-23272, 133) vs _postgres
               (23274-23871, 598); _verify_notes_moodboard_studio_schema_sqlite (14174-14183, 10) vs
               _postgres (15495-16186, 692). Full table in the sidecar.
             36 _sqlite-only and 20 _postgres-only unpaired methods (sidecar)
             the SQLite migration ladder as ~64 sequential `if target_version >= N and
               current_db_version == N-1:` statements inline in _initialize_schema_sqlite (20466-21114),
               duplicated a second time in the same file at (20651-20660)
canonical:   the already-shipped decomposition template is core/DB_Management/media_db/ (31,846 LOC,
             api.py / errors.py / constants.py / repositories/ / runtime/ / schema/), which replaced
             Media_DB_v2.py. The ChaCha equivalent is ALREADY UNDERWAY: core/DB_Management/chacha/ holds
             21 modules and 31,053 LOC (note_graph_suggestion_store 5,229; persona_state_store 4,313;
             task_store 4,195; note_store 2,779; character_store 2,304; message_store 2,107;
             conversation_store 2,044; ...).
destination: continue the chacha/ extraction. The remaining 45,292 lines are not "CRUD not yet moved" —
             they are the schema/migration layer (7,301 LOC of _sqlite/_postgres pairs plus the two
             ladders) and the backend adapter layer. Those want chacha/schema/ and chacha/migrations/
             modules mirroring media_db/schema/, with one migration step per module rather than an
             800-method class.
knowledge:   two problems, both about where knowledge lives rather than about duplication per se.
             (1) The attachment mechanism. `_delegate_store_method` returns a `*args/**kwargs` closure
             and 273 names are bolted on with setattr after class creation. Those methods are invisible
             to mypy, ruff, and every IDE; their signatures are erased exactly the way the
             PromptStudioDatabase façade erases them, which is the mechanism that let finding 3's seven
             signature mismatches survive. The real surface of CharactersRAGDB is ~1,073 methods, of
             which 273 cannot be statically discovered.
             (2) The migration ladder is a state machine written as scattered conditionals, twice, in
             one method. `_sqlite_linear_migration_steps (8536-8609)` already exists as a proper
             version->callable table and is the right shape; the inline ladder at (20466-21114) is a
             parallel encoding of the same transitions that must be kept in sync by hand.
impact:      408 commits in 12 months land in this file — the highest churn in the repository. Every one
             of them is a merge-conflict candidate against every other, and 273 of its methods are
             outside static analysis. It is the blast-radius centre of the module.
tests:       (import-grep reachability, not measured coverage) 478 test files reference
             ChaChaNotes_DB/CharactersRAGDB; 148 mention PostgreSQL; `pytest --collect-only
             tests/ChaChaNotesDB` reports 1,086 tests collected, 4 errors (psycopg/hypothesis absent
             locally). Coverage breadth is genuinely high — this is a maintainability finding, not an
             untested-code finding, and the breadth makes the refactor cheaper than its size suggests.
effort:      expensive, but incremental and already half-done. Needs a design doc + ADR + staged plan.
owner-only:  no
confidence:  confirmed
```

### FINDING db-management-7

```
axis:        correctness
class:       divergent-copies
severity:    Medium
sites:       ChaChaNotes_DB.py:BackendCursorAdapter.fetchall (458-459)      `return list(self._result.rows)`
             ChaChaNotes_DB.py:BackendCursorAdapter.__iter__ (476-477)      `return iter(self._result.rows)`
             Evaluations_DB.py:_BackendCursorAdapter.fetchall (99-100)      `return list(self._result.rows)`
             media_db/runtime/rows.py:BackendCursorAdapter.fetchall (66-69) `rows = self._result.rows[self._index:]`
             PromptStudioDatabase.py:PromptStudioBackendCursorAdapter.fetchall (357-360)
                                                                            `rows = self._result.rows[self._index:]`
             (context: 5 independent DB-API-shim families over the same backends.base.QueryResult —
              media_db/runtime/rows.py:10,54; PromptStudioDatabase.py:275,320,378,457,486;
              ChaChaNotes_DB.py:448,510,589,660; Evaluations_DB.py:89,124;
              Workflows_DB.py:266,298,345)
canonical:   media_db/runtime/rows.py — already extracted into its own module, already imported by
             media_db/runtime/__init__.py:13 and media_db/runtime/execution_ops.py:25, already tested by
             tests/DB_Management/unit/test_media_db_runtime_session.py. It is the correct copy: its
             fetchall honours the cursor position, which is what PEP 249 specifies ("fetch all
             REMAINING rows").
destination: promote media_db/runtime/rows.py to core/DB_Management/backends/dbapi_compat.py — single
             responsibility: "expose a QueryResult through a PEP 249 cursor/row surface". The four other
             families import it. Do NOT grow Utils.py.
knowledge:   PEP 249 cursor position semantics. Two of the four copies dropped it.
scenario:    `cur.fetchone()` to peek at the first row, then `cur.fetchall()` to drain the rest.
             On media_db and PromptStudio the second call returns rows 2..n. On ChaChaNotes and
             Evaluations it returns rows 1..n — the peeked row is silently duplicated. Same for
             `for row in cur:` after a fetchone on ChaChaNotes. These adapters only wrap DatabaseBackend
             QueryResults, i.e. the PostgreSQL path, so the bug is backend-specific: identical code
             gives different results on SQLite and PostgreSQL.
impact:      no current caller in DB_Management does peek-then-drain (scanned ChaChaNotes_DB.py,
             Evaluations_DB.py and chacha/*.py for `X.fetchone()` followed within 12 lines by
             `X.fetchall()`; the four hits are all re-executed cursors, which reset the adapter). So
             this is latent, not live — but it is a trap laid in the two highest-churn files, and it
             only misfires on the backend with the thinnest test coverage.
tests:       (import-grep reachability, not measured coverage) tests/DB_Management/unit/
             test_media_db_runtime_session.py and tests/DB_Management/test_media_db_api_imports.py:119
             cover the correct copy only. Nothing asserts cursor-position semantics for the other four.
effort:      cheap — one module move plus four import swaps, with the existing media_db test as the
             parity check.
owner-only:  no
confidence:  confirmed (the divergence); probable-risk (the downstream consequence — no live caller today)
```

### FINDING db-management-8

```
axis:        correctness
class:       n/a
severity:    Low
sites:       PromptStudioDatabase.py:_BackendPromptStudioDatabase.list_optimization_iterations (2420-2470)
             PromptStudioDatabase.py:_BackendPromptStudioDatabase.list_optimization_iterations (2472-2517)
canonical:   NONE
destination: n/a
knowledge:   n/a — this is dead code with a suppression on it.
scenario:    Two defs of the same name in the same class body; Python binds the second. The first
             (2420-2470, 51 lines) is unreachable. Both carry `# noqa: F811`, so the linter's
             redefinition warning — which is NOT on the global ruff ignore list and NOT covered by the
             pyproject per-file BLE001 grandfather block — was silenced per line rather than fixed. A
             maintainer editing :2420 to change pagination will observe no behaviour change at all.
impact:      51 lines of code that look live and are not, in a file where the same-name-twice problem
             (findings 1 and 3) is the dominant defect class.
tests:       (import-grep reachability, not measured coverage) `list_optimization_iterations` is
             reachable through the façade at PromptStudioDatabase.py:7425; no test distinguishes the two
             bodies, because only one is reachable.
effort:      cheap — delete one, or reconcile them if they differ meaningfully.
owner-only:  no
confidence:  confirmed
```

### FINDING db-management-9

```
axis:        encapsulation
class:       n/a
severity:    Low
sites:       PromptStudioDatabase.py:PromptStudioDatabase.renew_job_lease (7295-7301)
             PromptStudioDatabase.py:PromptStudioDatabase.record_idempotency (7197-7200)
             PromptStudioDatabase.py:_PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS (58-76)
canonical:   NONE
destination: n/a
knowledge:   "a lease was not renewed" and "we could not tell whether the lease was renewed" are
             different facts. The façade collapses them.
scenario:    `renew_job_lease` wraps the delegated call in `except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
             return False`. That tuple (58-76) contains `AttributeError`, `TypeError`, `RuntimeError`,
             `OSError`, `ConnectionError`, `TimeoutError` and `sqlite3.Error`. A transient DB error or a
             connection drop during renewal is therefore reported to the worker as a clean "lease not
             renewed" rather than as a fault — and, given findings 1 and 3, so is a signature mismatch
             or a missing method. `record_idempotency` does the same with `suppress(...)`, so a failed
             idempotency write looks like a successful one.
impact:      low today because the tuple's breadth is the real problem and the concrete consequence
             (lease expiry -> another worker picks the job up -> duplicate execution) depends on worker
             configuration. Listed because the same exception tuple is what converts findings 1 and 3
             from loud errors into quiet wrong answers at other call sites.
tests:       (import-grep reachability, not measured coverage)
             tests/prompt_studio/unit/test_jobs_worker_lifecycle_hardening.py touches lease lifecycle;
             it does not inject a DB fault during renewal.
effort:      cheap — narrow the tuple at these two call sites to the storage errors that actually mean
             "not renewed", and let the rest propagate.
owner-only:  no
confidence:  confirmed (the swallow); assumption (the duplicate-execution consequence)
```

## Suggested Refactor/Actions

1. **Close the divergence loop before closing the individual defects.** The cheapest durable fix for
   findings 1, 3 and 8 is one test, not three patches: a parity test in `tests/lint/` — modelled on the
   existing AST ratchet `tests/lint/test_endpoint_auth_deps_import_boundary.py` — that AST-parses
   `PromptStudioDatabase.py`, asserts `set(methods(_BackendPromptStudioDatabase)) ==
   set(methods(_SQLitePromptStudioDatabase))` modulo a declared allowlist, asserts signature equality
   for every shared name, and asserts no duplicate defs within a class. Seed the allowlist at today's
   value so it can only shrink. That test fails today on `list_optimizations` and on the 7 signature
   mismatches, which is the point. Same ratchet shape applies to the 34 `_sqlite`/`_postgres` pairs in
   `ChaChaNotes_DB.py`.
2. **Fix finding 1 immediately** (port `list_optimizations` to `_SQLitePromptStudioDatabase`, or have
   the façade raise a typed `NotImplementedError` that maps to 501 rather than a generic 500). Small
   enough not to need a design doc. Owner-only if the endpoint's error mapping is touched.
3. **Finding 2 needs a decision, not a patch.** Write `Docs/Design/2026-09-21-chacha-schema-version-parity-design.md`
   answering: do the two backends share a version namespace or not? If yes, renumber and add the missing
   SQLite migrations. If no, rename the constants and the `db_schema_version` rows so nothing can read
   one as the other. Record as an ADR — it is a decision, and ADR-020 already covers the surrounding
   backend policy without settling this.
4. **Findings 4 and 6 are the two expensive ones and share a template.** Both need
   `Docs/Design/…-design.md` + ADR + Backlog task + `IMPLEMENTATION_PLAN_<slug>.md` with 3–5 stages.
   Both should cite `core/DB_Management/media_db/` explicitly as the shape — same team, same layer,
   already shipped, and `Media_DB_v2.py` (121 commits) no longer exists as a file. For ChaChaNotes the
   first stage is not more CRUD extraction (the `chacha/` package already holds 31,053 LOC of that) but
   the schema/migration layer: move the 34 `_sqlite`/`_postgres` pairs into `chacha/schema/` and the two
   ladders into `chacha/migrations/`, one module per version step, replacing the inline `if` chain with
   the `_sqlite_linear_migration_steps` table that already exists at `:8536`.
5. **Finding 5** — create `core/DB_Management/retry_policy.py` (one responsibility: classify a storage
   error as retryable and produce the next delay). Migrate the 28 PromptStudio sites, the two
   `Workflows_DB` helpers and `transaction_utils`. Decide the PostgreSQL predicate in the same design
   doc as item 4's PromptStudio split, since both touch the job queue.
6. **Finding 7** — promote `media_db/runtime/rows.py` to `backends/dbapi_compat.py` and swap four
   imports. Cheap, no design doc, existing media_db test is the parity check.
7. Propose (do not create) Backlog tasks for items 2–6. The Backlog is the ledger of record and must be
   changed through its MCP/CLI, never by hand.
