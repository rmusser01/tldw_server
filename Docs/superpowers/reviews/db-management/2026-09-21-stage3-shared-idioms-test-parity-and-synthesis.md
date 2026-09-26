# Stage 3 (2026-09-21): Shared Idioms, Pagination, SQLite Policy, Test Parity, Synthesis

## Scope

Cross-cutting idioms re-implemented inside `DB_Management/` where a designated owner already exists in
the package or where the copies have measurably drifted: opaque pagination cursors, "current UTC
timestamp" helpers, the WAL-truncate maintenance block, and SQLite connection policy. Then the
module-level test-parity picture and the synthesis across stages 1–3.

Stages 4 and 5 of the classic arc are deliberately collapsed into this file: the API/schema boundary and
data-source boundary work was already done by the April stage 3 and stage 4 files and re-verified as
addressed in stage 1 of this pass. Three substantive stages beats six padded ones.

## Code Paths Reviewed

- `media_db/runtime/email_search_cursor.py` (whole file, 56 lines):
  `email_cursor_scope (15-18)`, `encode_email_cursor (21-27)`, `decode_email_cursor (30-56)`
- `chacha/shared_workspace_chat_store.py:_encode_cursor (1212-1225)`, `_decode_cursor (1227-1274)`
- `Moderation_Review_DB.py:_cursor_offset (69-73)`, `list_items (379-425)` with `LIMIT ? OFFSET ?`
  at `:419` and `next_cursor` at `:424`, `list_audit (626-673)` with `LIMIT ? OFFSET ?` at `:667` and
  `next_cursor` at `:672`
- Timestamp helpers: `Sync_DB.py:utcnow_iso (1430-1434)`, `Explainer_DB.py:utcnow_iso (100-104)`,
  `ResearchSessionsDB.py:_utc_now (17-18)`, `Moderation_Review_DB.py:_utc_now (28-29)` +
  `_format_utc (36-38)`, `Orchestration_DB.py:_now_iso (179-180)`, `RPG_DB.py:_now (901-903)`,
  `ManuscriptDB.py:_now (293-294)`, `codegraph/repository.py:_utc_now (1144-1147)`,
  `Collections_DB.py:_utcnow_iso (88-89)`, `Workflows_DB.py (252)`
- WAL truncate block: `Personalization_DB.py:_truncate_wal_if_possible (465-481)`,
  `Personalization_DB.py:checkpoint_retention_history (483-511)`,
  `Sync_DB.py:_maintain_personal_context_retention_storage (9321-9360)`,
  `ChaChaNotes_DB.py (8088-8100)` (unguarded 4th copy)
- `sqlite_policy.py` (whole file, 150 lines): `configure_sqlite_connection (32-57)`,
  `run_sqlite_quick_check (63-84)`, `begin_immediate_if_needed (87-92)`,
  `configure_sqlite_connection_async (119-150)`
- `Kanban_DB.py:_configure_connection (350-370)`
- Non-adopting SQLite stores: `Evaluations_DB.py:255`, `RPG_DB.py:213`, `VisualIdentity_DB.py:143`,
  `watchlist_alert_rules_db.py`, `OpenWebUI_DB.py`

## Tests Reviewed

| Test file | Protects | Downgrades risk? |
| --- | --- | --- |
| `tests/DB_Management/test_email_search_cursor.py` | the email keyset cursor codec end to end | Yes for that one codec. Note it does not even import locally in this environment (`hypothesis` absent) — one of the 4 collection errors. |
| `tests/DB_Management/` (8 files by import-grep on `sqlite_policy`) | pragma policy application | Partially — they cover the helper, not the five stores that bypass it. |
| `Moderation_Review_DB`: **1** test file by import-grep | the moderation review store | No — the lowest test-file count of any store in the module, and the OFFSET pagination behaviour under concurrent insert is not among what it covers. |
| `transaction_utils`: **1** test file by import-grep | retry-on-conflict transaction helper | No — does not assert backoff shape. |
| `Personalization_DB`: 44 files, `Sync_DB`: 58 files, `Collections_DB`: 67 files, `Workflows_DB`: 47 files, `Evaluations_DB`: 26 files | broad behavioural coverage | Reachability, not coverage. None asserts the WAL-restore default or a timestamp wire format. |

## Validation Commands

```
$ grep -rn '"=" \* (-len\|urlsafe_b64decode\|urlsafe_b64encode' tldw_Server_API/app/core/DB_Management --include='*.py'
media_db/runtime/email_search_cursor.py:27:    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
media_db/runtime/email_search_cursor.py:35:        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
media_db/runtime/email_search_cursor.py:36:        if base64.urlsafe_b64encode(raw).decode().rstrip("=") != cursor:
chacha/shared_workspace_chat_store.py:1225:        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
chacha/shared_workspace_chat_store.py:1238:            padding = "=" * (-len(cursor) % 4)
chacha/shared_workspace_chat_store.py:1249:        if base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=") != cursor:

$ grep -rnE 'def (_now|_utc_now|_now_iso|_utcnow_iso|utcnow_iso)\b' tldw_Server_API/app/core/DB_Management --include='*.py'
ResearchSessionsDB.py:17:def _utc_now() -> str:
Moderation_Review_DB.py:28:def _utc_now() -> str:
Orchestration_DB.py:179:def _now_iso() -> str:
RPG_DB.py:902:    def _now() -> str:
ManuscriptDB.py:293:    def _now(self) -> str:
codegraph/repository.py:1144:def _utc_now() -> str:
Sync_DB.py:1430:def utcnow_iso() -> str:
Explainer_DB.py:102:    def utcnow_iso() -> str:
Collections_DB.py:88:def _utcnow_iso() -> str:
Workflows_DB.py:252:  (module-level, same body as Collections_DB.py:88)

$ grep -rn 'datetime.utcnow()' tldw_Server_API/app/core/DB_Management --include='*.py' | cut -d: -f1 | sort | uniq -c
   4 Evaluations_DB.py
   3 Workflows_DB.py
   3 Collections_DB.py
   1 Workflows_Scheduler_DB.py
   1 Watchlists_DB.py
   1 Voice_Registry_DB.py
   1 PromptStudioDatabase.py
   1 migrations.py

$ grep -rn 'busy_timeout *= *0' tldw_Server_API/app/core/DB_Management --include='*.py'
Personalization_DB.py:473:            connection.execute("PRAGMA busy_timeout = 0")
Personalization_DB.py:496:                connection.execute("PRAGMA busy_timeout = 0")
Sync_DB.py:9332:            connection.execute("PRAGMA busy_timeout = 0")
# restore defaults on those three:  5000, 5000, 10_000

$ for f in RPG_DB.py VisualIdentity_DB.py Evaluations_DB.py watchlist_alert_rules_db.py OpenWebUI_DB.py Kanban_DB.py; do
    echo "$f busy_timeout=$(grep -c busy_timeout $f) wal=$(grep -c journal_mode $f) policy=$(grep -c sqlite_policy $f)"; done
RPG_DB.py busy_timeout=0 wal=0 policy=0
VisualIdentity_DB.py busy_timeout=0 wal=0 policy=0
Evaluations_DB.py busy_timeout=0 wal=0 policy=0
watchlist_alert_rules_db.py busy_timeout=0 wal=0 policy=0
OpenWebUI_DB.py busy_timeout=0 wal=0 policy=0
Kanban_DB.py busy_timeout=1 wal=1 policy=0

$ grep -rn 'busy_timeout_ms=' tldw_Server_API/app --include='*.py' | grep -o 'busy_timeout_ms=[0-9_]*' | sort | uniq -c
   2 busy_timeout_ms=          # passed a variable
   2 busy_timeout_ms=_         # passed a _-prefixed constant
   3 busy_timeout_ms=1000
   2 busy_timeout_ms=3000
   4 busy_timeout_ms=5000
   4 busy_timeout_ms=10000
   2 busy_timeout_ms=30000

$ ls tldw_Server_API/tests/DB_Management | grep -c '_backends.py$'
24
```

## Findings

### FINDING db-management-10

```
axis:        correctness
class:       divergent-copies
severity:    Medium
sites:       Moderation_Review_DB.py:_cursor_offset (69-73)
             Moderation_Review_DB.py:list_items (379-425), LIMIT ? OFFSET ? at :419,
               next_cursor = str(offset + safe_limit) at :424, ORDER BY created_at DESC, id DESC at :404
             Moderation_Review_DB.py:list_audit (626-673), LIMIT ? OFFSET ? at :667,
               next_cursor = str(offset + safe_limit) at :672, ORDER BY created_at DESC, rowid DESC at :666
canonical:   two correct keyset implementations already exist in this package:
             media_db/runtime/email_search_cursor.py (versioned, scope-bound, canonicality-checked)
             and chacha/shared_workspace_chat_store.py:_encode_cursor/_decode_cursor (1212-1274)
destination: whichever survives the consolidation in finding 11 — this store should call it, not roll a
             third scheme.
knowledge:   "what a pagination cursor is". Two stores in this package answer keyset; this one answers
             row offset, and calls the result `next_cursor` so the API shape is indistinguishable.
scenario:    A moderation queue is by construction a table that grows while it is being paged. A
             reviewer fetches page 1 (offset 0, limit 50, ORDER BY created_at DESC). Three new items are
             flagged. The reviewer fetches page 2 with next_cursor="50": the three new rows now occupy
             positions 1-3, so rows that were at 48-50 have shifted to 51-53 and are shown a second
             time. Conversely when items are resolved and removed from the filtered set, rows shift up
             past the offset boundary and are **never shown** — a flagged item can silently skip review
             entirely. Neither is possible with the keyset cursors the package already has.
impact:      silent skips in a moderation review queue are the worst class of pagination bug: the
             operator has no signal that an item was missed.
cost-driver: secondary efficiency issue on the same lines — `LIMIT ? OFFSET ?` makes SQLite scan and
             discard `offset` rows per page, so paging cost is O(page_number × page_size); and
             list_items additionally runs an unfiltered-by-page `SELECT COUNT(*)` at :410-413 on
             every page. Both scale with table size × pages viewed.
tests:       (import-grep reachability, not measured coverage) Moderation_Review_DB has exactly **1**
             test file — the thinnest in the module. No test pages while inserting.
effort:      cheap once finding 11's shared codec exists; moderate standalone (needs a keyset tuple over
             (created_at, id) and a migration path for in-flight cursors).
owner-only:  no
confidence:  confirmed
```

### FINDING db-management-11

```
axis:        duplication
class:       true-duplication
severity:    Medium
sites:       media_db/runtime/email_search_cursor.py:encode_email_cursor (21-27) / decode_email_cursor (30-56)
             chacha/shared_workspace_chat_store.py:_encode_cursor (1212-1225) / _decode_cursor (1227-1274)
             (incompatible third scheme: Moderation_Review_DB.py:_cursor_offset (69-73) — see finding 10)
             (repo-wide this idiom appears at 22 sites; only the three above are in this module)
canonical:   NONE — neither in-module copy is designated, and they are peers in quality
destination: core/DB_Management/pagination_cursor.py — single responsibility: "encode and decode an
             opaque, versioned, scope-bound keyset position". NOT Utils.py, NOT http_client.py. The
             module must keep the two trust classes separate in its API surface: opaque pagination
             cursors (these three) are tamper-evident-by-canonicality-check only, whereas the
             repo's signed/crypto token sites (core/AuthNZ/api_key_crypto.py, the notes.py signature
             segments) are authenticated. A single "base64 helper" that flattens that distinction would
             be worse than the duplication.
knowledge:   the same five-step recipe, independently written twice: (1) urlsafe base64 with padding
             stripped on encode; (2) charset regex guard before decode; (3) max-length guard;
             (4) re-encode and compare to reject non-canonical encodings; (5) shape validation of the
             decoded JSON payload. Step 4 in particular is a non-obvious hardening step that both authors
             happened to get right — which is precisely the kind of knowledge that will not survive
             being written a third time.
impact:      already drifted on the one thing that matters for evolution: `encode_email_cursor` puts a
             version integer at payload[0] and `decode_email_cursor:42` rejects anything but 1;
             `_encode_cursor` emits a bare 3-tuple with no version. The workspace-chat cursor therefore
             cannot change its payload shape without breaking every cursor a client is holding, while
             the email cursor can. They also differ on max size (4096 chars vs `_MAX_CURSOR_BYTES`) and
             on scope binding (email hashes tenant+query+include_deleted into the token and rejects a
             mismatch at :42; the workspace cursor binds nothing, so a cursor from one query is silently
             accepted by another).
tests:       (import-grep reachability, not measured coverage) tests/DB_Management/test_email_search_cursor.py
             covers the email codec (and errors on import here — `hypothesis` absent). The workspace
             codec is reached through tests/Sharing and tests/Workspaces but has no dedicated codec test.
effort:      moderate — the destination module is small, but migrating the workspace cursor to a
             versioned payload needs a grace period for cursors already in flight.
owner-only:  no
confidence:  confirmed
```

### FINDING db-management-12

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       Sync_DB.py:utcnow_iso (1430-1434)                 -> '...+00:00', full microseconds
             Explainer_DB.py:utcnow_iso (100-104)              -> '...+00:00', SECOND resolution
             ResearchSessionsDB.py:_utc_now (17-18)            -> '...+00:00', full microseconds
             Orchestration_DB.py:_now_iso (179-180)            -> '...+00:00', full microseconds
             RPG_DB.py:_now (901-903)                          -> '...+00:00', full microseconds
             codegraph/repository.py:_utc_now (1144-1147)      -> '...Z', forced microsecond precision
             Moderation_Review_DB.py:_utc_now (28-29) + _format_utc (36-38) -> '...Z'
             ManuscriptDB.py:_now (293-294)                    -> delegates to
               ChaChaNotes_DB.py:_get_current_utc_timestamp_iso (25883-25892) -> '...Z', MILLISECOND precision
               (`isoformat(timespec='milliseconds').replace('+00:00','Z')`)
             Collections_DB.py:_utcnow_iso (88-89) and Workflows_DB.py:252 — byte-identical bodies,
               `datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()`
             plus 15 bare `datetime.utcnow()` sites: Evaluations_DB.py x4 (incl. :1322 with
               `isoformat(sep=" ")` and :1833 with default 'T'), Workflows_DB.py x3,
               Collections_DB.py x3 (incl. :4346, a NAIVE retention cutoff compared as a string against
               the AWARE values _utcnow_iso writes at :3666/:3748), Workflows_Scheduler_DB.py,
               Watchlists_DB.py, Voice_Registry_DB.py, PromptStudioDatabase.py, migrations.py
canonical:   `Sync_DB.py:utcnow_iso (1430-1434)` is the de-facto one — it is module-level, documented
             ("Return an ISO-8601 UTC timestamp for Sync v2 rows"), and used across the largest
             cross-store surface. `api/v1/utils/datetime_utils.py` is NOT the right owner for a
             DB_Management timestamp: it lives above this layer and importing it here would be the
             core -> api inversion the architecture doc forbids.
destination: core/DB_Management/timestamps.py — single responsibility: "the one wire format for UTC
             timestamps stored in TEXT columns by this package", exporting `utcnow_iso()` and
             `parse_stored_utc()`. Move Sync_DB's implementation there and have Sync_DB import it, so
             the canonical copy stops being buried in a 14,904-line file.
knowledge:   the WIRE FORMAT of a stored timestamp. Ten helpers produce FOUR mutually unsortable
             renderings of the same instant: `+00:00` with microseconds, `+00:00` truncated to seconds,
             `Z` with microseconds, `Z` with milliseconds. Two of them (`Sync_DB.utcnow_iso` and
             `Explainer_DB.utcnow_iso`) have the SAME NAME and differ on precision — the same shape as
             the repo-wide anchor bug in the workflows/meetings DLQ services.
scenario:    These are strings in TEXT columns, ordered and range-filtered as strings. `'Z'` is 0x5A and
             `'+'` is 0x2B, so for one instant rendered both ways the `Z` form always sorts AFTER the
             `+00:00` form. Any column that receives both renderings — via a store consolidation, a
             backfill, a Sync v2 merge, or simply a second writer using a different helper — produces
             an `ORDER BY created_at DESC` that interleaves incorrectly and a `WHERE updated_at > ?`
             bound that matches rows it should exclude. Independently, `Explainer_DB.utcnow_iso`'s
             second-truncation means two events 400ms apart record an identical timestamp, so any
             tie-break-free ordering on that column is nondeterministic. And `Collections_DB.py:4346`
             builds a NAIVE retention cutoff and compares it as a string against AWARE stored values:
             correct today only because the naive and aware forms share the same
             `YYYY-MM-DDTHH:MM:SS[.ffffff]` prefix, so the appended offset never changes the ordering —
             an accident, not a design.
impact:      Medium rather than High because no single column is today known to receive two renderings.
             The cost is change amplification plus a standing trap: the package has no answer to "what
             does a stored timestamp look like here", so every new store picks one at random and every
             store consolidation is a potential ordering regression.
tests:       (import-grep reachability, not measured coverage) Sync_DB 58 files, Collections_DB 67,
             Workflows_DB 47, Evaluations_DB 26. None asserts a wire format; the format is only ever
             asserted end-to-end in response-shape tests such as
             tests/Flashcards/test_flashcards_timestamp_contract.py and
             tests/StudyPacks/test_study_pack_response_timestamps.py — which is the API boundary, not
             the storage boundary.
effort:      cheap per call site, moderate in aggregate: changing a format changes stored data, so
             adoption has to be new-writes-only plus a tolerant parser. Start by making
             core/DB_Management/timestamps.py the owner and pointing new code at it.
owner-only:  no
confidence:  confirmed (ten helpers, four formats, the same-name precision split, the naive/aware
             cutoff); probable-risk (the ordering consequence, which needs two renderings in one column)
```

### FINDING db-management-13

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       Personalization_DB.py:_truncate_wal_if_possible (465-481) — restore fallback 5000
             Personalization_DB.py:checkpoint_retention_history (483-511) — restore fallback 5000
             Sync_DB.py:_maintain_personal_context_retention_storage (9321-9360) — restore fallback 10_000
             ChaChaNotes_DB.py (8088-8100) — 4th copy of the checkpoint WITHOUT the busy_timeout guard
canonical:   sqlite_policy.py:configure_sqlite_connection (32-57), whose declared default is
             busy_timeout_ms=5000
destination: sqlite_policy.py — it already owns "what pragmas a SQLite connection gets"; adding
             `truncate_wal(conn) -> bool` there keeps the responsibility single rather than growing it.
             This is the one case in this report where the right destination is an existing module,
             because the knowledge is identical to what that module already owns.
knowledge:   the sequence "read PRAGMA busy_timeout -> set it to 0 so the TRUNCATE checkpoint fails fast
             instead of blocking on readers -> run PRAGMA wal_checkpoint(TRUNCATE) -> restore the prior
             timeout in a finally". Copy-pasted three times.
scenario:    The three copies disagree on what to restore when `PRAGMA busy_timeout` returns no row:
             `timeout = 5000 if prior_timeout is None else int(prior_timeout[0])` in Personalization_DB
             (twice) versus `timeout = 10_000 if prior_timeout is None else ...` in Sync_DB. **Neither
             constant is right, and the disagreement is the finding.** The correct value is the one the
             owning connection factory set, and it is knowable: Personalization_DB.py:_connect (80-99)
             calls `configure_sqlite_connection(conn)` with no override, i.e. 5000 — so
             Personalization_DB's fallback is right by coincidence. Sync_DB's 10_000 matches no factory
             in this package (sqlite_policy's default is 5000; the 10_000 callers are
             media_db/runtime/sqlite_bootstrap.py:44 and backends/sqlite_backend.py:162,329, which are
             not the connection that reaches :9321). So Sync_DB's path can silently *raise* the busy
             timeout of a long-lived connection from 5s to 10s after one retention maintenance run, for
             every subsequent statement on that connection. The right fix is to stop hardcoding either:
             restore from `sqlite_policy`'s declared default, which is the single source of truth for
             what this package's connections are configured with.
             The ChaChaNotes copy at :8088-8100 omits the busy_timeout guard entirely, so its
             `wal_checkpoint(TRUNCATE)` at close can block on active readers for the connection's full
             busy timeout instead of failing fast.
impact:      Medium: a connection's contention behaviour is silently mutated by an unrelated maintenance
             routine, and the fourth copy can stall connection close.
tests:       (import-grep reachability, not measured coverage) Personalization_DB 44 files, Sync_DB 58
             files. No test asserts the restored busy_timeout value; the restore branch is only reached
             when the PRAGMA read returns nothing, which no test constructs.
effort:      cheap — one helper in sqlite_policy.py, four call sites.
owner-only:  no
confidence:  confirmed
```

### FINDING db-management-14

```
axis:        correctness
class:       adoption-gap
severity:    Medium
sites:       canonical, bypassed: sqlite_policy.py:configure_sqlite_connection (32-57)
             full hand-rolled reimplementation: Kanban_DB.py:_configure_connection (350-370)
               — same six pragmas in the same order (foreign_keys, busy_timeout, journal_mode,
                 synchronous, temp_store, cache_size), with busy_timeout=30000 and cache_size=-64000,
                 and no in-memory detection (it takes an `enable_wal` parameter instead of the
                 `_is_in_memory_connection` check at sqlite_policy.py:19-30)
             partial bypass — PRAGMA foreign_keys only, NO busy_timeout, NO WAL:
               Evaluations_DB.py:255, RPG_DB.py:213, VisualIdentity_DB.py:143
             no connection pragmas at all: watchlist_alert_rules_db.py, OpenWebUI_DB.py
             (for contrast, 30 modules DO import sqlite_policy, including 20 inside DB_Management)
canonical:   sqlite_policy.py:configure_sqlite_connection — stated default busy_timeout_ms=5000
destination: n/a — the helper exists and is widely adopted; these five stores simply do not call it.
knowledge:   "what a SQLite connection in this package is configured with". The adopters already spread
             the value across five settings (1000, 3000, 5000, 10000, 30000 across the app), and the
             five stores above add a sixth answer: nothing.
scenario:    SQLite's default busy timeout is 0. A connection opened by `Evaluations_DB`, `RPG_DB`,
             `VisualIdentity_DB`, `watchlist_alert_rules_db` or `OpenWebUI_DB` therefore raises
             `sqlite3.OperationalError: database is locked` on the FIRST lock conflict rather than
             waiting. Two concurrent requests writing to the same user's evaluations DB — an eval run
             persisting results while the CRUD endpoint updates the same row — surface as an immediate
             500, where the same contention on any sqlite_policy adopter waits up to 5 seconds and
             succeeds. Evaluations_DB is a genuine multi-writer store (job results, A/B test rows,
             pipeline presets), which is why it heads this list.
impact:      lock-contention behaviour of a user-scoped store is decided by which file it was written
             in. Also note these are the stores least likely to be exercised concurrently in tests,
             which is why the difference has not surfaced.
tests:       (import-grep reachability, not measured coverage) sqlite_policy has 8 test files covering
             the helper; Evaluations_DB 26 files; RPG_DB, VisualIdentity_DB, watchlist_alert_rules_db
             and OpenWebUI_DB are reached only indirectly. Nothing asserts busy_timeout on a connection
             from any of the five.
effort:      cheap — five call sites, one import each. Kanban_DB's block can be deleted outright in
             favour of `configure_sqlite_connection(conn, busy_timeout_ms=30000, cache_size=-64000,
             use_wal=enable_wal)`.
owner-only:  no
confidence:  confirmed
```

### Module-level test-parity picture (input to the findings above, not a finding itself)

`grep -rl "core\.DB_Management" tldw_Server_API/tests` = 1,270 files. Reachability is high and the
module is not under-tested in aggregate. The gap is specifically **cross-backend parity**, and it is
structural rather than accidental:

- PostgreSQL tests are gated on a `psycopg` import that is absent in a default dev environment
  (`pytest --collect-only tests/DB_Management` = 2,961 collected, 4 errors, two of them
  `No module named 'psycopg'`). So the PostgreSQL half of every dual-backend pair is the half that
  silently does not run locally.
- Where a parity test exists it is written per-feature, not per-pair: the 24 `*_backends.py` files in
  `tests/DB_Management/` each pin one family (flashcard counts, world-book lifecycle, persona memory
  filters, conversation updates, keyword literal search, ...). That is a good pattern — but 24 families
  against 59 PromptStudio paired methods and 34 ChaChaNotes `_sqlite`/`_postgres` pairs leaves most of
  the surface unpaired.
- The two failures found in stage 2 (`list_optimizations` missing on SQLite; the deck/character
  uniqueness policy missing on SQLite) are both *inverted* relative to the usual assumption: here it is
  the **SQLite** side that is behind, while the tests that exist for those features are PostgreSQL-only.
  The takeaway is not "add PostgreSQL tests" but "assert the two surfaces are the same surface", which
  a per-pair AST ratchet does in one file — see stage 2 Actions item 1.

## Synthesis across stages 1-3

1. The April 2026 pass is fully discharged: 8 of 8 findings addressed, including the cache-close item
   the 2026-04-15 rebaseline left open. Nothing from that ledger carries forward.
2. Every High finding in this pass sits in one of the two files the April pass inventoried and then
   skipped, and every one of them is an instance of the same root cause: **two implementations of one
   database behind a signature-erasing façade, with no mechanical check that they agree**. That is one
   root cause, five symptoms (findings 1, 2, 3, 4, 5), and one cheap countermeasure (an AST parity
   ratchet) that would have caught three of them before merge.
3. The Medium findings are the ordinary consequence of a 233k-LOC package with no stated answer to four
   small questions: what a cursor is (10, 11), what a stored timestamp looks like (12), how WAL is
   truncated (13), and how a SQLite connection is configured (14). Three of the four already have a
   correct implementation somewhere in the package; only the cursor question needs a new module.
4. Recommended sequencing: the parity ratchet and `list_optimizations` first (cheap, high value, one
   afternoon); then findings 7, 13, 14 (all cheap, all mechanical); then the schema-version decision
   (finding 2, needs an ADR); and only then the two decompositions (findings 4 and 6), which need
   design docs and should explicitly follow the already-shipped `media_db/` template.

## Suggested Refactor/Actions

1. `core/DB_Management/pagination_cursor.py` — one responsibility: opaque versioned keyset cursors.
   Promote the `email_search_cursor.py` design (it has the version field and the scope binding), migrate
   `chacha/shared_workspace_chat_store.py`, then convert `Moderation_Review_DB`'s two OFFSET queries to
   keyset. Keep signed/authenticated tokens out of this module; say so in its docstring.
2. `core/DB_Management/timestamps.py` — move `Sync_DB.utcnow_iso` there, re-export from `Sync_DB` for
   compatibility, and point new stores at it. Do not rewrite existing stored data; add a tolerant
   `parse_stored_utc()` that accepts all four current renderings.
3. Add `truncate_wal(conn)` to `sqlite_policy.py` and collapse the four WAL-checkpoint copies into it,
   restoring the timeout from the policy default rather than a per-copy literal.
4. Point the five non-adopting stores at `configure_sqlite_connection`; delete
   `Kanban_DB._configure_connection` in favour of a parameterised call.
5. Items 1 and 2 warrant `Docs/Design/…-design.md` because they change stored/wire representations;
   items 3 and 4 do not. None of the four is owner-only.
6. Propose these as Backlog tasks through the Backlog MCP/CLI. Do not hand-edit task files.
