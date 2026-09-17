# Independent native acceptance: UAT180, UAT189, UAT190, UAT191, UAT192

## Verdict

**All five remaining native acceptance gates pass.** Recommend completing TASK13260.117 (180), .127 (189), .128 (190), .129 (191), and .130 (192), together with their already retained automated and independent-review evidence. This audit makes no task changes and does not certify a full UAT matrix.

The official CLI criteria snapshots are included. The outstanding native criteria are 180 AC3, 189 AC2, 190 AC2, 191 AC2, and 192 AC3. Existing automated gates were already recorded as satisfied; this pass independently checks the native evidence, not a new automated test run.

## Identity and source boundary

- Disposable D: `f2a61712-8d3e-45df-bfb9-0527f86cda21`, owned by Alice/client_id `2`, deck `1`.
- Original source card: `37b10bd7-edf4-4f35-83c1-1490115d8c55`.
- Native application source attribution: backend `a130b8e5507ff2f1ec6a930a7aa12183a44b5d44`, from `source-before-start.json` captured 05:29:59.276 UTC. Current relevant endpoint and ChaCha bytes match that manifest (hashes in `source-attribution.json`). This is a backend manifest, not an immutable frontend bundle attestation. The parent reports the same running development frontend with only unrelated UAT205 HMR during this window.
- Earlier 02:15 reset evidence is a separate retained run, not retroactively attributed to the later a130 manifest.
- Normal identity events in the later capture show Alice/id2. No authentication headers, credentials, session stores, live database queries, browser actions, services, or source changes were used by this audit.

## Acceptance by finding

| Finding / remaining criterion | Independently verified native evidence | Disposition |
|---|---|---|
| UAT180 / .117 AC3: confirmed reset and Manage delete | Earlier real confirmation dialog and its second Reset scheduling button submit `POST /flashcards/D/reset-scheduling` with expected_version3 at 02:15:22.373. Response200 at 02:15:22.448 gives version4, queue `new`, repetitions0, last_reviewed null. Real reload list200 at 02:15:56.933 preserves the front, image asset reference, back, tag, version4 and reset scheduling. Later editor Delete sends expected_version6 at 06:36:05.242; response200 at .275 is `{deleted:true}`. Real reload and subsequent Manage list200 at 06:37:46.172 return four cards without D. | Pass; both required operations are retained. |
| UAT189 / .127 AC2: final singular completion | Actual Good button click snapshot `alice-d-good-rating.txt` shows **“1 card practiced in this cram session”** with “Cram session complete!”. Actual session3 becomes completed with cards_reviewed1; a later GET/reload preserves it. | Pass; singular native case supplements the retained zero/plural regression controls. |
| UAT190 / .128 AC2: exhausted successful nonempty filtered Cram | Before rating, the original tag query returns exactly D and the UI shows one remaining. After the final rating, the original click snapshot shows completion and “You reached the end of your cram queue.”, with no false “No cards match this cram tag filter.”. Importantly, the post-rating filtered GET still returns D, so completion is not caused by an empty server list. After the tag is actually removed, the old-tag GET returns an empty successful list and the UI correctly shows no-match guidance. | Pass; nonempty exhaustion and a genuinely empty tag scope are distinguished. |
| UAT191 / .129 AC2: actual scheduled Cram preview agrees with one rating | Pre-rating GET200 at 06:26:11.856 opts into scheduler preview and returns D version4/repetitions0, `good: "10 min"`. The actual UI has Update schedule checked and Good “10 min”. Exactly one review POST in the retained window targets D, rating3, Cram/deck1/original tag. Response200 at 06:26:35.084 returns version5/repetitions1, last_reviewed `06:26:35.074` and due `06:36:35.074`: exactly 600 seconds. The toast also says next review in 10 minutes. | Pass within the displayed minute precision, in fact exact to the retained timestamp precision. |
| UAT192 / .130 AC3: existing card tag replacement survives reload with actual membership | Actual **PATCH**, not PUT, at 06:29:34.829 sends replacement tag and expected_version5. Response200 at .878 gives version6 and only `uat192-replaced-20260917`. Full reload list200 at 06:31:04.589 preserves it. Old-tag GET200 at 06:34:29.825 has total0/items[]; new-tag GET200 at 06:35:12.392 has total1 and only D/version6. Old/new visible filter receipts agree. | Pass; membership verification goes beyond the serialized tags field. |

## Evidence interpretation and checks

The CLI JSON was parsed only between `### Result` and `### Ran`. `alice-d-final-events.txt` contains 557 events, captured at **06:37:47.358 UTC**. The final retained Manage snapshot is **06:37:47.179**, not the approximate 06:39 in the handoff. Selected request/response bodies and machine-checked conditions are in `parsed-evidence.json`.

The only captured rating POST targets D. The final original card retains version2/repetitions1 and the original Citrine question and answer. This supports preservation of that observed card; it is not a claim that every catalogue row is byte-identical or that no operation occurred outside the retained window.

The session3 end request succeeds200 with completed/count1. The review response and refreshed D list subsequently advertise Good “1 day”; that is the **post-rating next-state preview**, not the earlier 10-minute preview being validated.

The relevant backend list/count tag paths use `flashcard_keywords` joined to the keyword relation, with the PostgreSQL owner join. Current source hashes match the start manifest. Thus the real old/new filtered results exercise membership, rather than merely echoing `tags_json`. No direct SQL state or raw-SQL isolation claim is made.

I viewed `alice-d-completed-session.png`: it shows a reopened suggestions panel and a completed-session entry. It is deliberately **not** used as proof of the original completion message. That proof comes from the retained actual Good-click accessibility snapshot.

## Limits and retained non-success responses

- The pre-Flashcards history in the aggregate event file contains four Persona profile500 responses and two visual-identity500 responses around 06:16, associated with the separately tracked failures. There are also character404/chat403 responses during the account/navigation boundary around 06:19. This audit does not close those issues or claim a clean console or overall Chat pass.
- The post-delete request for D's assistant returns404 “Flashcard not found” at 06:36:05.394. This is consistent with the confirmed deletion and is retained, not hidden.
- Native observations cover Alice's existing PostgreSQL deck and one disposable card, one Good rating with scheduling enabled, one tag replacement, reset, and delete. Other scheduler ratings, races, rollback, stale versions, loading/errors, plural counts and other backends remain covered by the previously recorded tests; they are not newly certified natively here.
- No fresh provider inference, SSE capture, network fault, browser automation, product test rerun, or database inspection was performed for this audit. Bandit is not applicable to these private evidence-only documents.

## Artifacts and integrity

`input-manifest.json` records exact SHA256 and sizes for 32 allowlisted inputs (including official task snapshots). Original evidence was read and hashed in place without changes. `parsed-evidence.json` preserves selected factual rows, `source-attribution.json` records exact source limits, and `audit-manifest.json` hashes the resulting review packet. The mechanical parse/assertion pass succeeded; this is evidence validation, not an application test count.
