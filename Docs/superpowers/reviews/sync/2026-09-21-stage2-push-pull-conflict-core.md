# Stage 2 — push / pull / conflict resolution: cursor and watermark correctness

## Scope

The three operations that define Sync's contract with a client: `SyncV2Service.push`,
`SyncV2Service.pull` (both its legacy adapter-version-1 path and its token-paginated versioned
path), and the conflict-resolution batch. Specifically: idempotency of replayed batches, ordering
of sync-log entries, the interaction between an unresolved materialization conflict and the pull
watermark, clock/TTL handling on the pull token, and the efficiency of the versioned scan.

## Code Paths Reviewed

- `v2/service.py:push (4491-5023)` — 533 lines, 33 `SyncPushRejected(...)` construction sites
- `v2/service.py:push` idempotency probe (4715-4738), personal-context conflict replay (4740-4790)
- `v2/service.py:push` duplicated preflight-conflict handler (4847-4878) and (4937-4968)
- `v2/service.py:pull (5025-5276)` — dispatch at 5072-5075, personal-context legacy branch
  (5117-5223), legacy non-personal-context branch (5235-5276)
- `v2/service.py:_scan_pull_page (10658-10700)` — blocker filter at 10684-10693
- `v2/service.py:_resolve_cursor (9974-9990)`, `_update_cursors (10625-10656)`,
  `_parse_cursor (10557-10566)`
- `v2/service.py:_pull_versioned (10159-10304)` — `safe_raw_envelopes` (10245-10253),
  `boundary` (10254-10261), watermark advance (10263-10271)
- `v2/service.py:_scan_versioned_pull_page (10306-10420)` — `load_candidate` (10338-10360),
  interleave loop (10367-10416)
- `v2/service.py:_encode_pull_token (10432-10476)`, `_decode_pull_token (10478-10555)`,
  `_decode_pull_token_segment (556-565)`, `_pull_token_secret (10422-10430)`
- `v2/service.py:resolve_conflicts_batch (6749-6887)`, `resolve_conflict (6890-7103)`,
  `_is_conflict_resolution_replay (7105-7134)`
- `v2/store.py:get_existing_envelope_for_idempotency (1047-1051)`,
  `list_envelopes_after (1053-1073)`
- `Docs/ADR/034-durable-server-origin-sync-mutation-batches.md` — the "conflict-resolution and
  repair amendment" (lines 98-120), which makes an unresolved projection conflict an ordering
  blocker for later history

## Tests Reviewed

- `tests/Sync/test_sync_v2_service.py:test_versioned_pull_does_not_advance_past_unresolved_conflict
  (7667-7756)` — the decisive one. It asserts, for `adapter_version=2` (versioned path), that an
  envelope sitting behind an unresolved materialization blocker is still delivered on the next
  pull after the blocker clears. There is **no analogous test for `adapter_version=1`**, and the
  legacy path fails that same assertion (see `sync-1`).
- `tests/Sync/test_sync_v2_service.py:test_pull_token_rejects_tampering_oversize_and_negotiated_version_set_change
  (7761-…)` — protects `_decode_pull_token`'s signature, size and version-set checks. Downgrades
  the risk on `sync-13` substantially.
- `tests/Sync/test_sync_v2_service.py:2738, 3382-3385` — `next_cursor` on conflicting pushes;
  `:4953-4954, 5022, 5323, 8216` — cursor-advance loops; `:7577-7584` — repeat pull idempotence;
  `:8453-8460, 8585` — exact `next_cursor` values.
- `tests/Sync/test_sync_v2_personal_context_conflicts.py`,
  `tests/Sync/test_sync_v2_personal_context_recovery_budget.py` (2,588 LOC) — the
  personal-context branches of `pull` and the recovery budget.
- All of the above are SQLite-only. `tests/Sync/test_sync_v2_service.py` does not mention Postgres.

## Validation Commands

```
# Repro for sync-1, written to the scratchpad (NOT added to the repo), mirroring
# test_versioned_pull_does_not_advance_past_unresolved_conflict but with adapter_version=1:
$ python -m pytest <scratchpad>/test_repro_legacy_blocker.py -x -q -s -p no:randomly
FIRST envelopes: []
FIRST next_cursor: 2 has_more: False
blocked seq: 1 later seq: 2
SECOND envelopes: []
SECOND next_cursor: 2
E   AssertionError: REPRO: legacy pull advanced past the blocker and dropped later-v1
E   assert [] == ['later-v1']
1 failed, 5 warnings in 2.89s

# The same scenario on the versioned path is green in the repo's own suite:
$ python -m pytest "tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_versioned_pull_does_not_advance_past_unresolved_conflict" -q -p no:randomly
1 passed

$ sed -n 4491,5024p tldw_Server_API/app/core/Sync/v2/service.py | grep -c "SyncPushRejected("
     33

$ diff <(sed -n 4847,4878p .../service.py | sed 's/^ *//') <(sed -n 4937,4968p .../service.py | sed 's/^ *//')
(no output — 32 lines identical modulo indentation)

$ grep -n "max_pull_page_size" tldw_Server_API/app/core/Sync/v2/service.py | head -2
838:    max_pull_page_size: int = 100
```

## Findings

### FINDING sync-1

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       The wrong copy: v2/service.py:pull (5252-5260) — legacy adapter-version-1 branch,
               `next_sequence = max(e.server_sequence for e in raw_envelopes)` with no blocker
               filter, fed by v2/service.py:_scan_pull_page (10658-10700), whose `visible` filter
               at :10684-10693 drops everything at or after `blocker_cursor` while `raw` keeps it.
             The right copy: v2/service.py:_pull_versioned (10245-10261) — computes
               `safe_raw_envelopes` (excluding `blocker_cursor` and `restore_barrier`) FIRST and
               derives `boundary` from that, then advances watermarks only for envelopes
               `<= boundary` (:10263-10271). Its docstring at :10172 states the rule explicitly:
               "Pull a token-paginated page without advancing past hidden conflicts."
             The partial copy: v2/service.py:pull (5186-5197) — the legacy personal-context
               branch handles `restore_barrier` but not the materialization `blocker_cursor`.
canonical:   v2/service.py:_pull_versioned (10245-10261) is the correct implementation
destination: One private `_advance_pull_watermark(raw, page, *, page_limit, blocker_cursor,
             restore_barrier, since)` used by all three branches — the only place that decides
             how far a pull cursor may move.
knowledge:   "A pull cursor may never advance past an envelope that was withheld rather than
             delivered." Written correctly once, partially once, and not at all once, in one file.
scenario:    REPRODUCED. Dataset `dataset-1`, device `device-1` advertising no
             `supported_adapter_versions` (so `_device_supports_adapter_version` at :567-590
             resolves to version 1 and `versioned_mode` at :5072 is False — the default for any
             client that has not negotiated v2).
               1. Envelope `blocked-v1` is inserted at server_sequence 1, then marked
                  `apply_status="conflict"`, and an unresolved `projection_conflict` conflict row
                  is recorded against it. Per ADR-034 it is now an ordering blocker.
               2. Envelope `later-v1` is inserted at server_sequence 2, `apply_status="applied"` —
                  a perfectly deliverable change on a different object.
               3. The device pulls. `_scan_pull_page` returns raw=[1,2]; `visible` drops both
                  (seq 1 has apply_status "conflict"; seq 2 is `>= blocker_cursor`). The response
                  is `envelopes=[]`, `next_cursor="2"`, **`has_more=False`**.
               4. The conflict is resolved out of band. The device pulls again with cursor="2".
                  Response: `envelopes=[]`.
             `later-v1` is never delivered to that device, and `has_more=False` told the client
             it was fully caught up. The same sequence with `adapter_version=2` delivers
             `later-v1` correctly — that is `test_versioned_pull_does_not_advance_past_unresolved_conflict`.
impact:      High. This is silent, permanent, per-device data loss on the default (non-negotiated)
             client path, in the exact window ADR-034's conflict-as-ordering-blocker amendment
             exists to protect. Nothing surfaces it: the server reports success, the client
             reports "up to date", and the loss is proportional to how much history sits behind
             the blocker. It is recoverable only by an operator resetting that device's cursor.
cost-driver: n/a
tests:       import-grep reachability, not measured coverage — tests/Sync/test_sync_v2_service.py
             has the versioned-path test (:7667-7756) and 165 passing tests over `pull`, but no
             legacy-path equivalent. Adding one is the regression test for this fix and is cheap:
             the existing test is a copy-paste away with `adapter_version=1`.
effort:      cheap — one shared helper, one new test mirroring an existing one. No design doc
             needed; it is a defect fix against an existing ADR, not a decision.
owner-only:  no
confidence:  confirmed (reproduced end to end against the real store on SQLite)
```

### FINDING sync-10

```
axis:        correctness
class:       n/a
severity:    Medium
sites:       v2/service.py:resolve_conflict (6890-7103) —
               unreachable branch at :6925-6935 (`if conflict.domain in
               PERSONAL_CONTEXT_SYNC_DOMAINS and _verified_personal_context_exchange is None:
               self.require_active_exchange(...)`), guarded out by the unconditional
               `raise SyncStoreError(...)` at :6915-6916 for exactly that domain set;
               dead parameter `require_personal_context_conflict` (:6901, used only at :6913) —
               zero callers pass it anywhere in app/ or tests/;
               dead parameter `personal_context_exchange` (:6902) — its only consumer is the
               unreachable branch at :6934, yet resolve_conflicts_batch:6866 still passes it.
             Supporting fact: v2/models.py:PERSONAL_CONTEXT_SYNC_DOMAINS (175-181) contains every
             domain whose name starts with `personal_context.`, so the `startswith` guard at
             :6913 and the membership guard at :6915 can never disagree.
canonical:   NONE
destination: n/a — delete
knowledge:   "Who is allowed to resolve a personal-context conflict, and what proof they must
             present." The file currently states it twice with contradictory answers: once as
             "never here, use the batch path" (:6915) and once as "here, if you bring an
             exchange proof" (:6925). Only the first executes.
scenario:    No live failure today — the raise wins, so behaviour is correct. The failure is the
             next edit. A maintainer relaxing :6915 (for instance to allow a read-only
             `personal_context.scope` resolution) would reasonably conclude from :6925-6935 that
             exchange verification is already wired up for the direct path. It is not: the branch
             only fires when `_verified_personal_context_exchange is None`, and every caller that
             would then reach it is the *unbatched* one, whose `personal_context_exchange`
             argument no production caller sets. The result of that edit would be a
             personal-context conflict resolvable with no active-exchange proof.
impact:      Medium rather than Low because the dead code is a security check. Dead validation
             reads as coverage; it is the shape of thing that turns a one-line relaxation into an
             authorization bypass. The two dead parameters also widen a public method's signature
             for nothing.
cost-driver: n/a
tests:       import-grep reachability, not measured coverage —
             tests/Sync/test_sync_v2_personal_context_conflicts.py and
             tests/Sync/test_sync_v2_personal_context_exchange_gate.py (1,722 LOC) cover the batch
             path that is actually reachable. Nothing covers the dead branch, because nothing can.
effort:      cheap — delete the branch and both parameters, and drop the argument at :6866.
owner-only:  no
confidence:  confirmed (unreachability proved from the domain tuple at v2/models.py:175-181;
             zero callers confirmed by grep across app/ and tests/)
```

### FINDING sync-8

```
axis:        efficiency
class:       n/a
severity:    Medium
sites:       v2/service.py:_scan_versioned_pull_page.load_candidate (10338-10360) — issues
             `self.store.list_envelopes_after(..., limit=1)` once per stream to prime
             (:10362-10364) and then once per consumed envelope inside the interleave loop
             (:10415-10416);
             v2/store.py:list_envelopes_after (1053-1073) forwards straight to SQL.
             Contrast: the legacy path fetches one page in one query —
             v2/service.py:_scan_pull_page (10670-10679), `limit=page_limit + 1`.
canonical:   NONE
destination: n/a — keep the k-way merge, change its fetch granularity
knowledge:   n/a (efficiency)
scenario:    n/a
impact:      Medium. Correct but costly; the interleave itself is necessary (it merges S signed
             streams in server_sequence order under the shared personal-context recovery budget),
             only the per-envelope fetch is not.
cost-driver: One SQL round-trip per envelope returned, plus one per negotiated
             `(domain, adapter_version)` stream to prime the merge. With the default
             `max_pull_page_size=100` (v2/service.py:838) and S streams, a single versioned pull
             page costs `S + up to 101` queries where the legacy path costs 1. It scales linearly
             in page size and in the number of enrolled domains, and every query is a separate
             network round-trip on PostgreSQL. `SYNC_PULL_TOKEN_MAX_STREAMS` bounds S, so the
             dominant term is page size.
tests:       import-grep reachability, not measured coverage —
             tests/Sync/test_sync_v2_service.py (the versioned-pull tests at :7667-7900) assert
             behaviour, not query counts. A query-count assertion would be the regression test.
effort:      moderate — refill each stream's buffer with `limit=page_limit + 1` instead of
             `limit=1` and drain the buffer in the loop, re-querying only when a buffer empties.
             The budget accounting (`budget.consume()` at :10357) has to move with it, which is
             the fiddly part.
owner-only:  no
confidence:  confirmed (the per-envelope `limit=1` and the refill call site are both explicit)
```

### FINDING sync-12

```
axis:        duplication
class:       true-duplication
severity:    Low
sites:       v2/service.py:push (4847-4878) and v2/service.py:push (4937-4968) — 32 lines,
             byte-identical modulo indentation: the same `try: conflicts.append(
             self._store_preflight_conflict(...))` wrapped in the same three
             `except SyncIdempotencyConflictError / SyncMaterializationBusyError /
             PersonalContextStorageEncryptionUnavailableError` handlers building the same three
             `SyncPushRejected(...)` values, followed by the same
             `if stop_on_conflict: stopped_after_conflict = True; continue` tail
             (4879-4881 and 4969-4971).
             Wider context: `push` constructs `SyncPushRejected(...)` at 33 separate sites in one
             533-line method.
canonical:   NONE
destination: One private `_record_preflight_conflict(dataset, envelope, outcome, *, exchange,
             conflicts, rejected) -> bool` on `SyncV2Service`, returning whether a conflict was
             actually recorded — which also fixes the defect below.
knowledge:   "Which failures of `_store_preflight_conflict` are retryable, and what error code
             the client sees for each." Written twice; a fourth exception type from that call
             path is two edits, and missing one silently changes only one of the two push
             branches (adapter preflight vs. post-insert head conflict).
scenario:    A real, small defect that the duplication hides: in BOTH copies the
             `if stop_on_conflict: stopped_after_conflict = True` tail runs even when the `try`
             raised and `conflicts.append(...)` never executed. Concretely — push a batch of 5
             envelopes with `stop_on_conflict=True`; envelope 2 hits an `AdapterConflict`, and
             `_store_preflight_conflict` raises `SyncMaterializationBusyError` (the projection
             lock is held). Envelope 2 is rejected with `sync_projection_busy` (retryable,
             correct), but envelopes 3-5 are then rejected with `stopped_after_conflict` —
             a code that tells the client a *conflict* needs review when in fact nothing
             conflicted and the whole tail is simply retryable. The client's recovery path for
             those two codes is different.
impact:      Low. Misleading error code on the batch tail under transient projection contention;
             the envelopes are not lost and a retry succeeds. Reported because it is concrete and
             because it is exactly the class of drift the duplication invites.
cost-driver: n/a
tests:       import-grep reachability, not measured coverage —
             tests/Sync/test_sync_v2_service.py:5590-5682
             (`test_preflight_conflict_rechecks_materialization_blocker_after_evaluation_race`)
             exercises the preflight-conflict path; no test drives `stop_on_conflict=True`
             together with a raising `_store_preflight_conflict`.
effort:      cheap — one helper, one `return False` guard on the `stop_on_conflict` tail.
owner-only:  no
confidence:  confirmed (the byte-identical block, verified by diff; the `stop_on_conflict` tail
             placement, verified by reading both copies)
```

### FINDING sync-13

```
axis:        duplication
class:       divergent-copies
severity:    Low
sites:       This module's site in repo cluster C1: v2/service.py:_decode_pull_token_segment
             (556-565) — `padding = "=" * (-len(segment) % 4)` then
             `base64.b64decode(..., altchars=b"-_", validate=True)`. (The briefing cited :559;
             the def is at :556.)
             Trust classification: **signed token**, not an opaque pagination cursor. It decodes
             both halves of `<payload>.<hmac-sha256>` at v2/service.py:_decode_pull_token
             (10494-10499), verified with `hmac.compare_digest` at :10505.
             Comparators in the same trust class: core/AuthNZ/api_key_crypto.py:_b64decode
             (118-120) and api/v1/endpoints/notes.py (877-891).
canonical:   NONE exists. Within the signed-token class the Sync copy is the strongest and should
             be the one promoted.
destination: `core/Utils/opaque_tokens.py` owning base64url encode/decode of server-issued
             tokens and nothing else, with TWO explicit entry points so the trust boundary is not
             flattened: `decode_cursor_segment(...)` (bounded, validating, no signature) and
             `decode_signed_segment(...)` (bounded, validating, canonical-form-enforcing, used
             only where an HMAC is verified). Explicitly NOT Utils/Utils.py.
knowledge:   "What a server-issued base64url segment is allowed to contain, and how big it may be
             before we decode it." Currently answered 22 times across the repo with at least three
             different levels of strictness.
scenario:    n/a — no live defect in this copy. Recorded so the consolidation does not regress it.
impact:      Low. The Sync copy is the reference implementation of this idiom in the repo: it is
             the only one that combines a pre-decode encoded-size bound
             (`SYNC_PULL_TOKEN_MAX_ENCODED_BYTES`, :10488-10491), `validate=True`, an explicit
             `altchars`, a post-decode size bound (:10500-10501), a version check (:10510), a
             constant-time signature compare (:10505-10506), and TTL/skew bounds
             (:10518-10530). By contrast core/AuthNZ/api_key_crypto.py:_b64decode (118-120), in
             the same trust class, has neither `validate=True` nor any length bound. The one
             thing the Sync copy lacks is the non-canonical-encoding rejection that
             api/v1/endpoints/notes.py:878-891 performs (re-encode and compare); without it two
             distinct token strings can decode to the same signed bytes. Nothing in Sync keys off
             the token string, so that is a hygiene gap rather than a defect.
cost-driver: n/a
tests:       import-grep reachability, not measured coverage —
             tests/Sync/test_sync_v2_service.py:7761+
             (`test_pull_token_rejects_tampering_oversize_and_negotiated_version_set_change`,
             including the single-character mutation at :7818) protects this decode path well.
effort:      moderate as a repo-wide consolidation (22 sites, two trust classes); cheap as a
             Sync-local change (adding the canonicality check is three lines).
owner-only:  no — but the notes.py and chat.py sites in the wider cluster are under
             `tldw_Server_API/app/api/v1/**` and are owner-only.
confidence:  confirmed (line-verified; the comparison against the other two signed-class copies
             was read directly)
```

## Suggested Refactor/Actions

1. `sync-1` is the one thing in this ledger that should be fixed this week. Extract the watermark
   rule into a single helper, route all three pull branches through it, and add the
   `adapter_version=1` mirror of
   `test_versioned_pull_does_not_advance_past_unresolved_conflict`. Defect fix against ADR-034's
   existing rule — needs a Backlog task, not a design doc or an ADR.
2. `sync-10` — delete the unreachable branch and the two dead parameters in the same PR as any
   other `resolve_conflict` work. Do not "fix" the branch by making it reachable; the batch path
   is the decided design (ADR-034's conflict-resolution amendment).
3. `sync-12` — fold the duplicated handler into one helper and make the `stop_on_conflict` tail
   conditional on a conflict actually having been recorded.
4. `sync-8` — buffer per stream instead of `limit=1`. Worth a query-count assertion in the test
   suite so it cannot regress.
5. `sync-13` — leave the Sync copy alone until the repo-wide C1 consolidation happens, then insist
   the shared module keeps the two trust classes separate and adopts the Sync copy's bounds as the
   signed-class baseline.
