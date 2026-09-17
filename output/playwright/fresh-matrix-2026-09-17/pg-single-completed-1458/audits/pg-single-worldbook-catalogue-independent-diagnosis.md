# UAT239: independent world-book catalogue failure audit

Task: TASK13260.181. Frozen PostgreSQL single-user source: `8f8774e6c868b304a96d95ab82e28389c129a78b`.

## Disposition

**Root diagnosis confirmed by independent source and native-evidence inspection.** The fresh Character editor catalogue calls `WorldBookService.list_world_books`, which still uses a SQLite-style connection context manager. PostgreSQL returns `BackendConnectionWrapper`, whose class has no `__enter__`/`__exit__`. The method fails before executing its SELECT; the route maps the resulting DB error to HTTP 500. This is independent of UAT238 Media RLS and not evidence of a world-book SQL/policy failure.

No production/test changes, test execution, browser/runtime/DB operation, credentials/private-log access, or tracker changes were performed. Only this ignored audit is written. Static AST inspection confirms the wrapper protocol absence without importing/executing application modules.

## Native evidence

In `testbot-entry-result.txt` (captured **14:35:01.407 UTC**):

| Request | Response | Result |
|---|---|---|
| GET `/api/v1/characters/world-books?include_disabled=true`, 14:33:37.006 | 14:33:37.055 | 500, `detail: Failed to list world books` |
| Same GET, 14:33:38.091 | 14:33:38.108 | Same 500/detail |

The safe error excerpt records the corresponding local times **07:33:37.028** and **07:33:38.105**: `list_world_books:775` reports **'BackendConnectionWrapper' object does not support the context manager protocol**, followed by the route's DB error mapping and access-log 500s. It also reports successful world-book table initialization immediately before each failure. `postgresExcerpt` is empty; no PostgreSQL query failure is claimed.

## Frozen causal path

- `characters_endpoint.py:1507–1516`: catalogue route constructs `WorldBookService` and invokes `list_world_books(include_disabled=...)` before entry counting or response serialization.
- `world_book_manager.py:746–776`: at line **757**, `with self.db.get_connection() as conn` fails on PostgreSQL. Intended query preserves `deleted=False`, optional `enabled=True`, and `ORDER BY name`; rows become dicts and populate `_book_cache`.
- `ChaChaNotes_DB.py:8044–8050`: `get_connection` returns the raw SQLite connection for SQLite, but a `BackendConnectionWrapper` for PostgreSQL.
- Wrapper class at **589–642** implements cursor/execute/commit/rollback and `__getattr__`, but does not implement context-manager special methods. Implicit `with` protocol lookup does not use that `__getattr__` to delegate to the driver.
- The manager catches and wraps the protocol error as `CharactersRAGDBError`; `characters_endpoint.py:1554–1556` maps it to the observed safe `Failed to list world books` 500. The failure occurs before catalogue SELECT, so empty and populated catalogues share this entry failure.

The response schema already uses datetime fields (`world_book_schemas.py:112–116`); this audit identifies no timestamp-string mismatch or additional proven failure.

## Prior UAT214 / TASK13260.153 boundary

That task's recorded defect was **get_character_world_books**, invoked during Character chat/context loading. Its final notes explicitly scope the repair to two pure reads and leave other legacy CRUD patterns unverified:

1. `get_character_world_books`, now `execute_query(..., read_only=True)` at line 1638.
2. `get_entry_counts_for_world_books`, now the same helper at line 818.

The prior permanent suite exercises those readers and `GET /characters/{character_id}/world-books`, including actual PostgreSQL/SQLite filtering/order and transaction controls. It does **not** exercise the separate `/characters/world-books` catalogue manager method. The original bounded UAT214 native chat acceptance remains valid; it is not whole-world-book-lifecycle acceptance. UAT239 is a newly exercised residual, separately tracked.

## Minimal repair and verification requirements

- Change only this pure catalogue read to the existing `db.execute_query(query, tuple(params), read_only=True)` pattern, retaining SQL/parameters/order, dict conversion, cache population, and safe error mapping. Do not add generic `__enter__/__exit__` to the wrapper or convert this read to an unconditional transaction that could settle caller work.
- The existing read helper (ChaChaNotes_DB.py:8234 onward) owns a new PostgreSQL read transaction only when the connection is IDLE and both application/backend transaction depths are zero. It preserves an already-active caller transaction. Use that contract rather than inventing another connection owner.
- Add causal actual PostgreSQL and SQLite tests for **this catalogue method and actual catalogue route**, both `include_disabled` values, empty/populated data, deleted exclusion, enabled filtering, name ordering, cache behavior, aggregate total/enabled/disabled counts and zero/nonzero entry counts. Keep parent-specific UAT214 controls unchanged.
- Assert standalone PostgreSQL returns to IDLE; preserve explicit/nested and raw/implicit caller-owned pending writes and caller rollback. Include safe route error mapping. Use official required-PostgreSQL fixtures with zero skips; fixture seeding may isolate this reader from unrelated unverified legacy writers and must say so. Do not mock the connection wrapper to make protocol compatibility pass.
- After minimal GREEN/static checks and independent review, replay the real Character editor catalogue natively on reviewed source. This audit does not certify catalogue repair, world-book CRUD, tenant isolation, full lifecycle, or matrix acceptance.

## Exact source/evidence binding

All seven source/history files match the original PostgreSQL-single archive manifest, SHA-256 `f9a6d30e6a8faef5635df40d5ee026e18ebc225d344bf29b62a8bcca7f2b2f4f`. No runtime-generated tree walk was performed.

Source paths relative to `sources/pg-single`:

| Path | SHA-256 |
|---|---|
| `tldw_Server_API/app/core/Character_Chat/world_book_manager.py` | `039e29e72ada4c2a160f070d6262272f3db3d893826b9a97c3618ef1908bf691` |
| `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` | `33f987c9e4c6acc3b1502c9f6e10dc66c6e17929f2258c4d67fa40e87b7fe06b` |
| `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py` | `8b1bdad39df395c6f7752a1323cfbda1dd0d65761bbc75c3782bf78589f2f399` |
| `tldw_Server_API/app/api/v1/schemas/world_book_schemas.py` | `9e5bc9026ecc15418bec011d1975c01d77f37dba175a46ef04e8e433ca9b770f` |
| `tldw_Server_API/tests/DB_Management/test_character_world_book_reads_backends.py` | `355a5dd387a44b05d2e78e5cf594aa3a7081f40fcdad523d58c36b85d0a95b2c` |
| `tldw_Server_API/tests/DB_Management/test_world_book_initialization_backends.py` | `0b37e00942c60e47e9020ce6ff256606d9ef1560b6cc0cff29fe7f144f981efa` |
| `backlog/tasks/task-13260.153 - Load-Character-world-books-with-backend-compatible-read-transactions.md` | `2ec33cfc115505ed4f883f045ff7c995fb11de91e44bafabe07ca0bfe0f434d9` |

Inputs relative to `native/pg-single`:

| Path | SHA-256 |
|---|---|
| `worldbook-scope-diagnosis.md` | `a2922b7c2a5128c464641c55ec049f8350072319e86ecdda949c9aaef27929ed` |
| `worldbook-error-excerpt.json` | `17015edbf7ff9c871c1cf72086b14e88cc588d09662e0741a650f2a8b12e5fdb` |
| `testbot-entry-result.txt` | `9d3b84b0887a47fedb29c969e045c9ba4effe530602cba78de3a574bb30ef873` |
