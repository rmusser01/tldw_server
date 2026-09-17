# SQLite single: stale re-rate interval preview diagnosis

## Finding

Confirmed stale UI prediction on frozen revision `8f8774e6c868b304a96d95ab82e28389c129a78b`, run `fresh-final-20260917`. Re-rate restores the pre-review card snapshot, including its old interval previews. The next rating correctly operates on the newer persisted schedule. Three scheduled ratings producing three events is intentional; this is not an undo/history-count defect.

## Native sequence (2026-09-17 UTC)

Card `2cdc609e-7d72-4f74-882b-c134e9be32f6`:

| Event | Time | Evidence |
| --- | --- | --- |
| Initial Easy response, HTTP 200 | 13:15:55.006 | Version 2, interval 4; next Hard 6 / Good 10 / Easy 13 days. |
| Scheduled Cram Good response, HTTP 200 | 13:16:55.122 | Version 3, interval 10, due September 27; next Hard **14 days**, Good **25 days**, Easy **1 mo**. Body read at 13:16:55.158. |
| Scheduler-enabled canonical list, HTTP 200 | 13:16:55.199 and .305 | Same version 3 and updated previews. |
| Re-rate UI capture | 13:16:55.415 | Correct saved-10-days toast; visible Hard **6 days**, Good **10 days**, Easy **13 days**, plus “Rate this card again to update your response”. |
| Re-rate Hard request / response, HTTP 200 | 13:21:59.667 / .678 | Rating 2; version 4, interval **14 days**, due `2026-10-01T13:21:59.673Z`, repetitions 3. |

The 14-day persisted result matches the backend prediction already returned after Good. No generation/model inference is involved in this discrepancy.

## Source trace

Paths below are relative to the frozen root `sources/sqlite-single`.

- `ReviewTab.tsx:537,562–563` captures `activeCard` into `cardForUndo` **before** submitting the rating. After a successful `reviewRun.submit`, lines 625–627 store that old snapshot in `lastReviewedCard`; the returned schedule is used for the toast at 657–670, but not the stored card.
- `handleUndoReview` at 833–864 passes this snapshot through `buildReviewUndoState`; `utils/review-undo.ts:8–17` returns the same object as `overrideCard`. `localOverrideCard` takes precedence over refreshed data in both due (238) and Cram (282–285) paths.
- `ReviewTab.tsx:465–469` renders `activeCard.next_intervals` first, explaining the exact old 6/10/13 labels despite completed fresh list reads.
- `useFlashcardReviewRun.ts:206–245` returns the successful response subject to existing ownership guards. `useFlashcardQueries.ts:968–989` invalidates queries after review; it does not update the component snapshot. The request sends UUID/rating/context, not the obsolete scheduling fields.
- `services/flashcards.ts:547–562` already exposes updated schedule, version and `next_intervals` in `FlashcardReviewResponse`. The backend endpoint at `flashcards.py:2038–2053` reviews the current card; `ChaChaNotes_DB.py:37387–37405` returns persisted scheduling fields and newly calculated next previews.

## Contract and test gap

The retained UAT128 tracker line 23 explicitly accepts six events for five initial ratings plus one re-rate. Preserve that contract, queue identity and counters.

`ReviewTab.rerate.test.tsx:268–310` verifies restored identity and a second mutation, but its response fixture (202–214) omits next previews and asserts no refreshed interval. `ReviewTab.cram-mode.test.tsx:533–546` verifies scheduled re-rate ordering/counts through refetch, but not the changed prediction. Inspection only; no tests run here.

A bounded future repair should retain card content/identity while updating the successful re-rate snapshot from the canonical response. Add actual-render controls for due and scheduled Cram: old 6/10/13 previews → successful Good response 14/25/1 mo → re-rate displays the latter, including refetch and repeated-rating controls. Preserve authority/session guards and failed-review behavior. No new undo API or scheduling-rule change is indicated.

## Evidence boundary

Only this ignored audit was written. No source, tests, task, browser, runtime, configuration or database changes; no new test/inference requests. This diagnoses the captured preview discrepancy, not broader Study acceptance.

## SHA-256 binding

All ten inspected frozen files below match their original archive-manifest entries. Paths in the source table are relative to the frozen root; input paths are relative to the matrix packet.

| Input | SHA-256 |
| --- | --- |
| `copy-preparation/sqlite-single-archive-manifest.json` | `26255fe54e27f7e92d849bbf810a7224c655602e3e2f9514eb6cbcce3c96bba1` |
| `native/sqlite-single/chat-card-scheduled-rerate.txt` | `9eb5a766d9524d0e3b1c34edaf34243ad61a9ecf2c1bb722d720be3675cb513c` |
| `native/sqlite-single/chat-card-review-events-final.txt` | `169529a8a74e421efc6a4f0b7ce8ccb0413e446d75bfe86bb5f94f3c95266b01` |

| Frozen source | SHA-256 |
| --- | --- |
| `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx` | `38da019f993e40d6b892e294fe9418afeb0e327ce94d1ae561ddc95cd49572fe` |
| `apps/packages/ui/src/components/Flashcards/utils/review-undo.ts` | `efdfd40f61f7456b999aae7cee8e922068718cf82413a20ea305566a1475ed47` |
| `apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts` | `8a0f75914c299b4fbd27fa700835330738f2872b6a82a0b3785f114b23259e4b` |
| `apps/packages/ui/src/components/Flashcards/hooks/useFlashcardReviewRun.ts` | `1ac37103dbd2987d6bbbb90169ec76f65b7a3294fd5e859a3117be62a24d1a2d` |
| `apps/packages/ui/src/services/flashcards.ts` | `923f290dce3b692d9d1d5160f43dc0fdd03fb4678cb1c34fffc221f762068aaa` |
| `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx` | `9106beb4a58576b1c1d8d9abdec380ab4e78529c094945250a2b4c8b7acd45b4` |
| `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx` | `d0346c08e1a0bc9bd3bad2e1bb34ec6d2de5ec5aa7d1a22ad7e49d3f8e26d718` |
| `tldw_Server_API/app/api/v1/endpoints/flashcards.py` | `cc3f242d2c115a324db2f2ed180256c32408c1cb53a9515e7b795bea4183b107` |
| `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` | `33f987c9e4c6acc3b1502c9f6e10dc66c6e17929f2258c4d67fa40e87b7fe06b` |
| `output/playwright/cycle5-repair-verification-2026-09-16/native-single/cycle5-repair-native-single-RUNNING_TRACKER.md` | `49907b6966162556518222f71ea7a9b0a3d68101658f2d36650eb32646db20c1` |
