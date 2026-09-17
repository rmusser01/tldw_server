# UAT181 independent design and purity review

Reviewed approved `.tmp/uat181-repair-20260916/DESIGN.md` against product baseline `bb5a0c01f1` (documentation checkpoint `b253ca901c`). This is an independent source/design review, not final frozen-diff approval or native acceptance. Parent requested a separate official-fixture populated-asset probe during review; its result appears below.

## Verdict

The original35 proposed SELECT callsites are side-effect-free and appropriate for the existing explicit `read_only=True` ownership boundary. No locking SELECT, advisory lock, sequence operation, session-setting function, DML CTE or DML RETURNING occurs in those35 statements. SQL/parameters/defaults/transaction-depth behavior should remain unchanged.

**The35-site inventory is insufficient for the promised populated Buddy→Flashcards chain.** Actual Buddy attachment/activity target resolution includes unflagged conversation/workspace reads. The author has acknowledged the gap and is adding actual service-chain RED controls plus a bounded expansion to39 sites/four production files. Final review must confirm the added tests fail on baseline and pass with those four additional callsites; this design review does not pre-certify that implementation.

## Actionable finding: populated Buddy target resolution starts an inherited transaction

`BuddyService.attachment` (`core/Buddy/service.py331`) reads the attachment and Buddy, then calls `resolve_target` (234). For an attached conversation this calls `_conversation`→`get_conversation_by_id` (`chacha/conversation_store.py497`, execute_query507). For a workspace it calls `get_workspace` (`ChaChaNotes_DB.py27291`, separate normal/include_deleted SELECT branches). `BuddyService.activity`→`conversations` also reaches `get_conversations_for_user` (`conversation_store.py878`, execute_query922) for workspace attachments.

All three getters currently omit read_only. With the repository reads fixed, these later getters still start an implicit PostgreSQL transaction. A subsequent flagged list_flashcards on the same pinned connection correctly sees INTRANS and refuses to settle that caller's work, so its relation lock persists. A test limited to BuddyRepository, an empty attachment or a profile without a target cannot expose this.

Required regression: real populated BuddyService attachment and activity for both conversation and workspace targets, real persisted assistant result, then Flashcards list on the same cached DB/connection; assert IDLE/no retained Flashcards lock and that an independent connection can execute the exact existing-column bootstrap ALTER under bounded lock_timeout. Include an unavailable/missing target read, because even an empty SELECT can acquire transaction state/locks. Preserve normal target ownership filtering.

The proposed additions are narrowly justified by these call chains: get_conversation_by_id (one callsite), get_conversations_for_user (one), get_workspace normal/include_deleted (two), hence39 total rather than38. Their queries are plain projections/filter/order/limit SELECTs; the inspected normalization helpers do not access the DB. This finding was sent to parent and author before implementation.

## Original35-site purity inventory

Line numbers below identify the inspected baseline execution sites (they will shift when keyword arguments are inserted).

| Group | Baseline sites | Purity and call-chain assessment |
|---|---|---|
|Deck6|34850,34880,34892,34913,34996,35012|Simple deck/share SELECTs; shared/normal list branches both need coverage. Name/deleted/workspace filters only.|
|Template3|35405,35418,35434|COUNT/list/detail; deleted-value helper is pure; template serialization happens after buffered fetch.|
|Card/queue5|35958,36031,36090,36115,36257|Projection/count/tag aggregation/UUID IN/selection loop; FTS uses built-in to_tsquery, no volatile effects. Visibility/table/order helpers are pure. Selection loop can execute multiple empty queries then get_flashcard; all exits must settle owned reads.|
|Review history5|36465,36592,36622,36717,36737|Final SELECTs/aggregates/joins are pure. See mixed-method ownership limitation below: listing has prior maintenance; mark-completed has prior write.|
|Card/detail6|37169,37235,37245,37505,39042,39104|Projection/blob/keywords/citations/study-pack lookup; provenance assembly delegates to these read methods without hidden DB writes. Populated blob conversion has a separate baseline defect, below.|
|Assistant4|42144,42187,42201,42221|Pre/post thread lookup, thread detail, message list are pure. get_or_create itself can INSERT in an existing deliberate transaction; only its SELECTs should opt in. Need both existing-thread early return and newly-created post-select.|
|Buddy repository5|Buddy_DB.py get/list_profiles/assets/attachment/latest_results|latest_results is a SELECT-only ranked CTE with window function and EXISTS. No update acknowledgement is included. Profile JSON decoding follows buffered fetch.|
|Persona1|persona_state_store.py get_persona_profile1783|Pure SELECT; row conversion decodes JSON/booleans only. Populated optional_persona_id path through BuddyService._response is necessary.|

## Ownership behavior and must-test boundaries

1. **Own only IDLE/zero-depth reads.** execute_query checks raw PostgreSQL status plus both ChaCha tx_depth and backend depth. A read starting IDLE enters the existing backend transaction manager; errors roll it back, success commits it. No global inferred purity or default flip is warranted.
2. **Buffered fetch is compatible with finishing the read before caller fetch.** BackendCursorWrapper.execute calls backend.execute and stores a QueryResult; fetchone/fetchall materialize dictionaries from that buffered result. No server cursor is left to consume after the owned commit. Test actual returned rows/blob types, not just IDLE.
3. **Fresh-connection scope setup is not a hidden leak.** `_apply_postgres_client_scope` commits session-scoped set_config(...,false) during acquisition. The scoped read commit does not clear that session tenant setting. Preserve the existing principal/owner assertions; this review did not perform a cross-tenant native test.
4. **Caller ownership controls must use actual pending work.** For representative opted-in plain getters, cover uncommitted public execute_query DML, DML RETURNING and write CTE, explicit/nested db.transaction, explicit backend.transaction, and raw BEGIN. Check both status and independent-observer visibility before/after caller commit/rollback. Do not assert merely that the own connection can see its own pending write.
5. **Do not relabel whole mixed methods pure.** list_flashcard_review_sessions first invokes abandon_stale_flashcard_review_sessions, which intentionally uses db.transaction; mark_flashcard_review_session_completed first calls execute_query UPDATE(commit=True). get_or_create_study_assistant_thread can INSERT. Those existing boundaries may settle pre-existing work independently of the new SELECT flag, especially commit=True. This patch does not repair their historical public transaction semantics. Scope preservation claims to the opted-in read operation; use baseline comparison if a whole-method caller-transaction control fails. Do not broaden181 silently to rewrite mutation ownership.
6. **Empty/error exits matter.** Check actual not-found/empty results, first/second/third queue selection branches, fresh and existing assistant thread, empty and populated Buddy profile/persona. A deliberately invalid read should return the expected error, leave an owned read IDLE, and allow the next real query/constructor. Pre-existing errored transactions should remain caller-owned; do not automatically rescue them.
7. **Keep non-opted-in side effects untouched.** Existing public SELECT FOR UPDATE and session-function controls must retain their original transaction/lock behavior. No blanket SELECT classifier, read_only default or cursor.description inference.
8. **Use the actual bootstrap boundary.** An independent query succeeds even while another connection holds a harmless compatible read lock; therefore SELECT1 is insufficient proof. The bounded lock_timeout exact ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS front_search TEXT is the relevant discriminator. Each lifecycle test should release only its official isolated fixture connections.

## Separate populated asset failure, confirmed without product changes

Parent specifically authorized a private targeted regression under this review directory. `test_asset_content_probe.py` uses the canonical `pg_database_config` fixture via the official test plugin and the existing mandatory-PG runner; SQLite is a real control. No app database or manual database setup was used.

Actual result: **1 PostgreSQL FAIL / 1 SQLite PASS, zero skips, 2.77s**. add_flashcard_asset persisted bytes, and metadata UUID/byte_size assertions passed. The content getter then failed:

```
test_asset_content_probe.py:34 -> db.get_flashcard_asset_content(asset_uuid)
ChaChaNotes_DB.py:37251 -> blob = row[0]
KeyError: 0
```

BackendCursorWrapper.fetchone returns a dict, with image_data available by name. The same getter is reached by the actual content route (`flashcards.py1047`→metadata getter→content getter), but this DB probe did not send an HTTP or native request and did not validate an uploaded image file. Its repository byte fixture is intentionally limited to the binary retrieval contract. This is an existing populated-row defect, not caused by read_only or a new181 regression. Parent/author were notified for separate tracking before any production edit; no fix is included here.

Evidence: `asset-probe-redacted.log`, `asset-probe-command.json`, `asset-probe-source-before.sha256`, and private test. Mandatory runner invocation:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat181-review-asset-probe node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs .tmp/uat181-review-20260917/test_asset_content_probe.py -q --tb=short
```

## Limits and next gate

No production/test-suite/browser/runtime/task/git changes by this reviewer. Private report/probe artifacts are the only writes. The asset probe used only official isolated PostgreSQL/SQLite fixtures; no live session termination, broad locks inspection or app mutation. Final181 source/test diff,39-site count, owned hashes and final required PostgreSQL controls need review after author freeze. No broader promise that all ChaCha domains are free of transaction leaks follows from this bounded patch.
