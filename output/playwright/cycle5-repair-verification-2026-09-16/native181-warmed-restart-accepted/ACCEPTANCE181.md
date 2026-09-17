# Independent UAT181 final native acceptance audit

**Recommend closing UAT181 / TASK13260.118 within its recorded scope.** Literal AC3 is now supported: actual Flashcards/deck/Notes reads precede an owned overlapping API replacement; the warmed database has no retained read locks; the replacement initializes and serves actual dependent Notes/bootstrap requests successfully. Combine this native receipt with the previously reviewed AC1/AC2 causal regressions and transaction-ownership controls. No additional native gap remains for this literal criterion.

This is a read-only audit of retained evidence, not a new native run. `audit.py` independently verifies source pairs, chronology, metadata, API results and canonical identities. It passes with **25 hash-bound inputs**, **26 warmed Flashcards responses**, **3 post-health Notes/bootstrap responses**, and **20 separate canonical readback/auth-session requests**. `evidence-window-excerpts.json` contains only the relevant timestamps, routes, statuses and safe filter fields; response bodies, credentials and user content are excluded.

## Acceptance chronology — 2026-09-17 UTC

| Time | Retained observation |
| --- | --- |
| 09:18:46.016–.098 | Alice 2 keywords, collections and Notes list each return HTTP 200 on API 82583. Canonical list contains four notes, all client_id 2. |
| 09:20:52.929–09:21:36.782 | Actual Study then Manage reads return HTTP 200: source-review-plans/due, completed review sessions, decks, cards with due/all filters, analytics summary and review/next. The Study UI shows 8 available cards; Manage shows 8 cards. There are 26 responses across these routes. |
| 09:21:38.683 | Same content database: 20 sessions, all idle, zero public-table locks. This is after the warmed reads and before SIGTERM. |
| 09:22:10.222–.325 | Only owned 82583 is sent SIGTERM. Its listener becomes free while the process remains alive. |
| 09:22:20.617 | Replacement 89545 startup begins. |
| 09:22:36.185 | Replacement health HTTP 200; both old 82583 and new 89545 are alive. |
| 09:22:36.412 | Same content database: 23 sessions, all idle, zero public-table locks. |
| 09:22:37.406–.466 | Actual Alice 2 keywords, Notes list and collections return HTTP 200 after readiness. The list has the same four note IDs, all owned by2; native UI shows “Showing 1-4 of 4”. |
| 09:23:01.745–09:23:02.298 | Fresh API sessions verify the original completed StudyPack job and owner-protected canonical resources, then log out. |
| 09:25:26.324–.677 | After those owner/Study readbacks, the same content database has 38 idle sessions and zero public-table locks; both old 82583 and replacement 89545 are still alive. |

The previous audit deliberately left AC3 open because direct Flashcards/deck reads occurred after the earlier replacement. This replay supplies the missing ordering. Neither an empty pre-warm startup nor job polling substitutes for the actual warmed routes here.

## Source attribution

API 82583 is bound by matching before-start/after-health manifests at revision `7acc8b001a7ca2023b0e62782b481aeb4d8bba01`; replacement 89545 is bound by matching manifests at `598d377df25fd78e20b895c3cc5c303d19223b7c`. Each contains exactly 3,668 backend files. The supplied parent runtime contract is that hot reload is disabled; these are captured startup-source receipts, not an inspection of interpreter memory.

Only two backend hashes differ between the runtimes: `ChaChaNotes_DB.py` (separately reviewed UAT222 selected-owner StudyPack/provenance changes) and `core/StudyPacks/provenance.py` (UAT223 route correction). Both new hashes match their frozen task manifests. Operation owner module, HTTP/maintenance dependency, main registration, enrichment callback and StudyPack worker hashes match across the two startup sets. The worker also matches the reviewed UAT204 canonical-owner caller hash. This replay does not attribute UAT222/223 behavior to a new UAT181 change.

## Previously reviewed ownership checks and new readback

The retained independent HTTP/maintenance review covers 77 required-PG/SQLite tests plus four reviewer counterexamples. The StudyPack adopter review covers 21 lifecycle cases plus eight separately attributed UAT197 count controls; it exercises actual accessor/service/to_thread persistence, failure, cancellation and caller transaction decisions. The canonical-owner follow-on review covers 30 cases with cold/warm factory behavior and retains those lifecycle controls. The tagging/clustering review covers 32 required-PG/SQLite cases through actual thread/pool calls, including spawned clustering, caller isolation and incomplete writes. These are previously executed and reviewed controls; this evidence audit does not claim to rerun them.

The new 20-request result is independently checked rather than trusting its `passed` flag: Alice authenticates as 2, original job 5 is completed with pack 2/deck 10, the pack belongs to 2 and predates this replacement, its three cards and three assistant citation contexts return HTTP 200, primary citations match the cited source records, and repeated job reads are unchanged. Bob authenticates as 3 and receives HTTP 404 for the foreign job, pack and assistant context; his deck-filtered cards response is empty. Both sessions log out successfully. This is canonical readback of an already completed job, **not a new worker execution, regeneration or model-quality test**. It complements the prior worker/callback acceptance without widening it.

## Explicit limits

- Metadata observes all public ordinary/partitioned tables in one hash-identified content database, excluding indexes and catalogs. All three snapshots are read-only verified. They demonstrate no idle-in-transaction sessions or retained table-lock obstruction at these times; they are not continuous monitoring.
- The third metadata snapshot follows the post-health Notes and canonical owner/Study requests. It corroborates that the exercised reads did not retain table locks after successful bootstrap.
- Process receipts prove overlap at health HTTP 200 and again after the canonical readbacks at 09:25:26. They are point-in-time observations and do not record eventual old-process exit.
- Nearby Alice Graph 403 responses at 09:19:20 and 09:20:50 are retained separately in the narrow excerpts. They are expected permission responses under UAT221 and are not Notes/bootstrap 500 failures. Earlier Graph suggestion 503s are outside this acceptance window; no blanket clean-session claim is made.
- Native service role remains privileged. The Alice/Bob results verify application ownership, not raw-SQL RLS isolation.
- No assertion covers every Buddy attachment/activity branch, every non-HTTP/background caller, a whole-database lifecycle guarantee or the full fresh-install matrix. Existing unrelated findings remain separate.

No production/test source, runtime, browser, database, configuration, task, tracker or git state was changed by this audit. Only this private evidence packet was written. No new source lint/Bandit run is needed for this metadata-only audit; prior implementation security receipts remain separately attributed.

## Audit preparation note

The first audit-script run rejected its own warmed-route set because a second-only timestamp ending in `Z` was compared lexically against millisecond timestamps, excluding responses within that exact second. The window boundary was normalized to `.000Z`; the unchanged native inputs then passed all assertions. This is a private audit filtering correction, not a product/native failure or an omitted counterexample.
