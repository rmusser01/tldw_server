# Native restart diagnosis: distinct UAT181 / TASK13260.118

## Finding

The replacement API is blocked in **Flashcards bootstrap**, not by a retained Notes read transaction. At2026-09-17T00:03:23.382634Z the exact owned content database showed PG PID67531 waiting on `ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS front_search TEXT`, with an ungranted `AccessExclusiveLock` on `flashcards`. `pg_blocking_pids(67531)` identified seven older idle transactions:61816,61648,61817,61649,61670,61669,61668.

At00:05:18.842731Z each of those seven held a granted `AccessShareLock` on `flashcards`; none held locks on `notes` or `note_folders`. Their last visible query classifications were read-only SELECTs from flashcards/decks or buddy_attachments. Last query is not necessarily the statement that opened a transaction; the mixed-table history cannot establish every individual caller from metadata alone. Query values and user data were neither output nor retained.

The parent then cleaned up only owned, previously SIGTERM'd API740. A bounded follow-up at00:07:45.186676Z showed all seven original blocker PIDs gone, no blocking PIDs on replacement sessions, and no locks on flashcards/notes/note_folders. This supports old-process ownership without claiming direct socket identity: lsof showed29 API740 sockets to the owned55475 cluster, but PostgreSQL client ports did not match host ports across the Docker/proxy boundary. Transaction start times and the before/replacement-start receipts also place all seven blockers before replacement API2996.

## Native evidence distinction

- Before/after the actual Notes reads, the exact-content metadata receipts at23:59:06.585024Z and00:00:30.707782Z show no notes/note_folders locks.
- `notes-before-restart.txt` shows native Notes keywords/list/collections HTTP200 at23:59:30Z and note detail/tasks HTTP200 at00:00:05Z.
- Parent SIGTERM receipt: API740 at00:00:36.037Z; replacement API2996 started00:00:47.603Z. Old sessions remained during its graceful drain.
- Replacement log `backend-1789603247602.private.log` lines596/638 reports DB initialization failing while ensuring the PostgreSQL flashcard asset schema at00:01:04.140Z /00:01:09.206Z. The generic API process startup is not proof this per-user content DB bootstrap succeeded. The retained replacement-ready JSON's unauthenticated health observation is401; this audit does not relabel that particular receipt200.
- `notes-after-restart.txt` records Notes keyword/collection/list500 at00:01:46Z, followed by list500 at00:01:51Z. These requests cannot get a fully initialized shared content DB.
- The earlier00:02:07 metadata showing new PID67495's granted Notes locks describes locks held during replacement bootstrap. It does not identify an old Notes SELECT leak. At the decisive inspection PID67495 had rolled back and PID67531 was retrying/waiting on Flashcards DDL.

Target content DB fingerprint for all inspections: `2ba0a35cd8e53f76eea53d63cd0d0459e0aea4128471cc05eef8540832fca22b`.

## Source cause and boundary

`ChaChaNotes_DB._ensure_flashcard_asset_schema_postgres` issues the exact ALTER at line22167; `_initialize_schema_postgres` invokes this helper after schema-version handling at24924. The helper runs even if the column already exists, and PostgreSQL must still acquire the table's exclusive DDL lock. A duplicate same-column ensure exists in `_ensure_postgres_flashcards_tsvector`; native initialization error text identifies the asset helper as the first failing boundary.

`_get_thread_connection` pins PostgreSQL connections across operations. `execute_query` defaults `read_only=False`. Its safe opt-in owns a transaction only if status is IDLE and both ChaCha/backend transaction depths are zero. Most Flashcards/Buddy reads currently omit that flag. `backend.execute` automatically settles reads only when it acquired the connection itself, not for the externally supplied pinned connection. Therefore completed standalone reads can leave transactions and AccessShareLocks alive across requests and process drain.

Do not flip the generic defaults, infer purity from SELECT or cursor.description, or rollback external transactions globally. The backend's write-tag/CTE detector distinguishes writes but cannot prove SELECT purity: locking SELECT, set_config, advisory locks, or volatile functions may intentionally retain effects. DML RETURNING, write CTEs, implicit pending writes, and explicit/nested transactions must retain their owner.

## Approved bounded repair inventory

Flag only these inspected side-effect-free SELECT calls with existing `read_only=True`; no SQL, writes, helper defaults, or transaction-depth logic changes.

| Owner group | Functions / SELECT sites |
| --- | --- |
| Deck reads (6) | list_decks normal/shared branches (2); get_deck; get_deck_by_name; list_deck_shares; get_deck_share |
| Template reads (3) | count_flashcard_templates; list_flashcard_templates; get_flashcard_template |
| Card/queue reads (5) | list_flashcards; count_flashcards; list_flashcard_tag_suggestions; get_flashcards_by_uuids; get_next_review_card selection loop |
| Review history reads (5) | list_flashcard_review_sessions final SELECT; get_flashcard_review_session; mark_flashcard_review_session_completed post-write SELECT; get_flashcard_reviewed_cards; get_latest_flashcard_review |
| Card/detail reads (6) | get_flashcard; get_flashcard_asset; get_flashcard_asset_content; get_keywords_for_flashcard; list_flashcard_citations; get_study_pack_for_flashcard |
| Assistant chain (4) | get_or_create_study_assistant_thread pre/post SELECTs (2); get_study_assistant_thread; list_study_assistant_messages |
| Buddy repository chain (5) | get; list_profiles; assets; attachment; latest_results (read-only CTE) |
| Populated Buddy persona check (1) | PersonaStateStore.get_persona_profile, called by BuddyService._response |

Total35 callsites in three production files: ChaChaNotes_DB.py, Buddy_DB.py, chacha/persona_state_store.py. The three UAT177 analytics reads already opt in and need no change. Review-session maintenance, review/rollup repairs, assistant writes, deck/card/template/asset writes already use deliberate transaction scopes and remain untouched. Export delegates to list_flashcards.

## Proposed regression boundary

Use official required PostgreSQL fixtures, one cached CharactersRAGDB/connection, real Buddy→Manage/Generate→Study→assistant reads, and an independent connection attempting the exact existing-column bootstrap DDL with bounded lock_timeout. Reproduce actual retained locks first, then prove standalone reads leave IDLE and unblock bootstrap. Check each inspected starter (including empty/not-found legitimate results), failures, populated Buddy persona path, and SQLite controls. Preserve implicit caller writes, write CTE/RETURNING, explicit/nested/backend scopes, locking SELECT and session-function effects with independent observer commit/rollback checks. No production edits occurred during this diagnosis.

The repair is bounded to these entry paths. A pre-existing transaction owned by any other caller remains that caller's responsibility; this is not a guarantee that the entire content database or all domains have no idle transactions. No live database session termination, schema/data mutation, process mutation, browser action, or config changes were performed by this auditor. Parent-owned process cleanup is recorded solely as corroborating evidence.

## Durable sanitized snapshots

- blocking-metadata-000323.json: exact waiting statement/relation, waiting lock and seven pg_blocking_pids.
- blocker-locks-000518.json: seven granted Flashcards locks, timestamps, client ports and failed direct host-port mapping.
- after-cleanup-metadata-000745.json: all original blockers absent and zero scoped locks.

These JSON files retain the relevant safe subset of the successful read-only metadata calls shown in the audit tool outputs; they do not contain raw queries with values or runtime logs. Parent separately retained old API740 exit00:07:04 and native Notes Retry success00:08:08; those are corroborating parent-owned runtime observations, not actions taken by this auditor.
