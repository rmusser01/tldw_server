# UAT218 / TASK13260.156 — coherent Media initialization over ChaCha's sync log

**Design only; production held for parent review.** UAT216's three-trigger repair is independently frozen. This proposal does not modify or merge that unit.

## Established boundary and evidence

The real official PostgreSQL fixture initializes CharactersRAGDB first, then MediaDatabase on the same backend. Media fails in `media_db/schema/features/core_media.py:371` at `CREATE INDEX IF NOT EXISTS idx_sync_log_entity_uuid ON sync_log(entity_uuid)`. The source table legitimately contains ChaCha's `entity_id`. This happens before any StudyPack call. Original complete worker-case bytes and the both-order2FAIL/4PASS receipt are in `../uat216-repair-20260917/expanded-red-test.py` / `both-orders-red.log`.

The standalone permanent regression `tldw_Server_API/tests/DB_Management/test_media_after_chacha_initialization_postgres.py` reproduces **1 failure /0 skips**,3.06s, with its own guaranteed cleanup. Its official runner receipt is `.tmp/fresh-uat-recovery-20260916/uat218-standalone-initializer-red.redacted.log`. No expected failure marker, test skip, hand-built schema, native mutation or initializer bypass is used.

The following *additional incompatibilities are established by source comparison*, not yet claimed as separate observed runtime failures:

| Contract | ChaCha-created log | Media contract / real consumer |
| --- | --- | --- |
| Identifier | `entity_id TEXT NOT NULL` | `entity_uuid TEXT NOT NULL`; index and central writer/reader use that spelling. |
| Scope fields | No org_id/team_id in its base log | Media indexes and RLS policies plus writer/reader require nullable org_id/team_id. |
| Operations | create/update/delete check | Media accepts link/unlink too; `runtime/media_lifecycle_ops.py:91` logs unlink during a real cascading media deletion. |
| Payload | TEXT NOT NULL | Media's `_log_sync_event` explicitly permits no payload and writes SQL NULL. Its reader returns None. |

Source locations: ChaCha base log at `ChaChaNotes_DB.py:1420`; Media canonical table at `media_db/media_database_impl.py:796`, indexes at`:932`; shared writer `runtime/sync_utility_ops.py:72`, explicit read projection `runtime/sync_log_ops.py:18`; scope migration `schema/migration_bodies/postgres_early_schema.py:111`; RLS construction `schema/features/postgres_rls.py:92`. ChaCha's generic log reader near`:44090` unconditionally `json.loads`s payload and catches only JSONDecodeError, so a legitimate shared SQL-NULL payload also needs a narrow null-aware read branch if this contract is made coherent.

## Recommended option A: keep the stored identifier, adapt the two Media boundaries

Keep the physical `sync_log` relation and whichever supported identifier it already owns. Media's public read result still exposes `entity_uuid`; when storage is `entity_id`, project `entity_id AS entity_uuid`. Its one central writer uses the selected physical column. SQLite continues its existing entity_uuid SQL unchanged. This preserves old stored identifiers and all current trigger references; it follows the existing ChaCha fixed-column capability patterns used in216 and character history.

Use one small helper in the existing PostgreSQL core-schema module, not a new compatibility framework. The helper executes within the current bootstrap transaction, runs before indexes on initial Media creation, and is also invoked by the existing post-core setup on reopen. It inspects actual schema once, records a private fixed-column capability on the constructed MediaDatabase instance, and performs DDL only for missing/incompatible **known application-owned** fields. Runtime read/write helpers use that constructor-established closed-set value; they do not introspect on every event or translate arbitrary SQL.

### Exact normalization proposed

1. Accept exactly one existing supported identifier name, entity_id or entity_uuid. If the shape is absent/ambiguous, fail with SchemaError and leave the transaction unchanged. Do not rename, copy, derive, merge or backfill identifier values.
2. Add only missing `org_id BIGINT` and `team_id BIGINT` columns, nullable, using the existing v8 migration's schema style. No defaults/backfill; old rows retain personal/unscoped null fields. Existing scope values are never overwritten.
3. Preserve an existing compatible five-operation check. If catalog inspection identifies the exact old application-owned three-operation check, replace **only that check** with `CHECK (operation IN ('create','update','delete','link','unlink'))` in the same transaction. Match both its operation-column dependency and the known full definition/allowed set, not merely a constraint name or substring. Reject unexpected custom shapes rather than dropping arbitrary constraints. Use backend identifier quoting for the catalog-provided constraint name. Validate the replacement against all existing rows; do not use NOT VALID or disable checks.
4. Drop `NOT NULL` only from payload when present, matching Media's already-declared optional-payload contract. No row values are changed. Identifier, entity, operation, time, client_id and version retain their non-null requirements. ChaCha's generic log reader treats a SQL-null payload as None, preserving non-null JSON decoding/error behavior.
5. At the one logical Media identifier-index statement, choose the existing physical identifier. Other Media indexes are unchanged; scope columns exist before their indexes/policies are installed. Avoid repeated unconditional ALTERs on a compatible reopen. Keep the relation OID, sequences, keys, owners, grants, triggers and existing RLS policies; do not disable RLS or bypass its actor settings. Existing Media startup's normal policy establishment still runs unchanged.

The scope/capability helper belongs at the existing schema boundary. It must not issue its own commit/rollback, acquire another connection, or create a database. Constructor failure rolls back the entire initializer transaction, including normalization and newly created Media structures. The caller controls an already-open transaction exactly as current initialization does.

### Expected bounded production paths

1. `media_db/schema/features/core_media.py`: local compatibility helper +call before indexes +exact identifier-index selection.
2. `media_db/schema/backends/postgres_helpers.py`: reuse that helper in post-core/reopen setup so the instance's read/write capability is always established.
3. `media_db/runtime/sync_utility_ops.py`: PostgreSQL central writer selects the fixed physical column; bound values unchanged.
4. `media_db/runtime/sync_log_ops.py`: PostgreSQL read aliases that column to the existing `entity_uuid` response key; other projection/order/pagination unchanged.
5. `ChaChaNotes_DB.py`: only the generic sync reader's SQL-null payload handling; no216 trigger or209 owner-method changes.

Protocol/type declarations may need a private capability attribute in these same schema modules. Do not add a new global SQL adapter, table alias framework, per-event metadata queries, dual-write protocol, or broad trigger rewrite. If live controls establish more required changes, stop and present them before expanding this list.

## Alternative B: rename to Media's canonical physical identifier

Renaming existing entity_id to entity_uuid looks smaller at the failing index but leaves many installed/literal ChaCha trigger and writer references invalid. Although216 and a few existing readers select columns, many other fixed templates do not. A correct rename would require migrating every affected stored function, raw writer and sync consumer, including already-installed functions and unknown extensions. That is substantially broader and could invalidate preserved historical consumers. Adding a second independently writable column instead requires backfill plus bidirectional coherence rules and ambiguity handling. Neither is needed when a fixed-column projection can retain the existing data. Recommend A.

## Required permanent tests before GREEN

- Real Media-first and ChaCha-first initialization, fresh official PostgreSQL, plus separate-file SQLite controls. Both stores perform normal reads/writes afterward, not just table existence.
- Seed historical sync rows through real ChaCha APIs before Media initialization (including216 pack/membership/citation records). Verify exact IDs, owners, timestamps, versions and JSON payloads survive; continue writing after initialization and reopen.
- Media uses actual `_log_sync_event` through ordinary mutation paths; its public sync reader returns entity_uuid consistently in both storage shapes. Exercise real cascading Media deletion with keyword unlink, not only synthetic SQL INSERTs.
- Optional-null payload round-trips through both readers; non-null JSON remains identical. A malformed JSON control preserves existing error behavior without adding broad exception swallowing.
- Existing Media personal/org/team scope values and RLS definitions/flags survive a reopen; actor/ownership tests use official fixture patterns. No role/admin bypass is added by this repair.
- Current-head reopen performs no unnecessary data/identifier migration and remains idempotent. Reopen a real ChaCha-created legacy log through normal Media setup; do not relabel a current schema or manually install a fake log.
- Inject a controlled failure after compatibility DDL but before Media startup finishes: existing rows/constraints/column shape survive rollback, and the next normal attempt succeeds. Explicit caller-owned pending data must not be committed by the helper.
- Known three-operation check upgrades; five-operation check is retained; unknown/ambiguous identifier or custom-constraint shapes fail without partial writes. These last negative cases may use isolated fixture DDL specifically to verify rejection, never to bypass the real successful initialization path.
- Re-run216's actual worker/membership/citation graph in both orders, existing Media sync-reader/writer/cleanup tests, and relevant PostgreSQL migrations/RLS controls with mandatoryPG and zero skips.

## Gates and limits

First preserve causal failures for the actual boundaries, then implement the approved five-path design, then run required PG/SQLite controls, static checks and independent review. No218 production edit has begun. The index failure is proven; later contract mismatches above are source-grounded requirements whose actual controls still need to be exercised. No claim is made that current native startup used reverse order, nor that this synthetic failure explains job2: job2 is216's separately proven Media-first trigger mismatch.
