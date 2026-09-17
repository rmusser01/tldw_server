# Independent native audit: UAT137 / UAT168 / UAT178

2026-09-17 UTC; parent TASK13260. Read-only audit of retained artifacts, including direct visual inspection of all three listed PNGs. No browser, runtime, database, inference, task, git or product changes. Only this private report and the separately requested residual-candidates report were written.

## Verdicts

| Finding | Verdict | Accepted native boundary |
|---|---|---|
| UAT168 / TASK13260.105 | **PASS — original populated listing/count failure** | Actual selected-deck GET200 returns numeric count/total1 and the real saved Citrine card; Manage visibly renders that card and source note. |
| UAT178 / TASK13260.115 | **PASS — controlled queue failure and recovery** | Precisely scoped transport abort settles to visible error and Retry, with no false completion; remove the same handler, click Retry, receive actual200 and display the same card without a recorded reload. |
| UAT137 / TASK13260.76 | **PASS — native one-card case** | Genuine queue has one pending card. Visible status uses `1 card remaining`; the accessibility status contains singular `1 card remaining, 0 reviewed, Available now: 1`. |

These verdicts do not close UAT184's Manage plural copy or UAT185/186's later assistant failures. They do not certify scheduled rating, completed sessions, asset/reset/delete flows, or every adjacent branch in UAT168's repair.

## Source and runtime provenance

Parent identifies the authenticated Alice PostgreSQL multi-user cell: the observed same-origin `http://127.0.0.1:18583/api/v1/...` is the frontend proxy to real API18503. All audited wire receipts use that actual same-origin address; this reviewer did not contact either service. The parent reports no product edits during capture and subsequent integration as `a647f5cd89`.

[fixed-runtime-source.json](../uat181-native-20260917/fixed-runtime-source.json) at00:38:21.252Z records API PID14540, the then-HEAD `b253ca901c988c5f4d1e7a15cb24f15c2e5b669c` and four frozen181/182 production file hashes. These hashes match current files. Do not relabel the capture as having run at the later integration commit. The earlier [source-manifest.json](source-manifest.json) is explicitly a **protocol-preparation** manifest, not a runtime receipt; its ReviewTab, useFlashcardQueries, useFlashcardReviewRun and ReviewProgress hashes nevertheless all match the inspected current source. The recorded freeze is parent evidence, not an independent runtime import inspection.

## UAT168: populated server result and original identity

[manage-events.txt](manage-events.txt) records selected-deck requests at00:40:17.383Z and200 responses at00:40:17.422Z for both `limit=500&order_by=created_at` and the visible page's `limit=20&order_by=due_at`, each with `deck_id=1`, `due_status=all`, `include_workspace_items=false`, `offset=0`. Each body contains `count:1`, `total:1`, `has_more:false`, and exactly one item:

- Card UUID `37b10bd7-edf4-4f35-83c1-1490115d8c55`, deck1, `client_id:"2"`, `version:1`, New queue state.
- Front: `When do Citrine study volunteers meet?`
- Back: `Citrine study volunteers meet every Tuesday at 14:00.`
- Tag `Citrine study`; `source_ref_type:"note"`, source ID `b83dca90-fab0-4c6f-8c0f-6f1e93dfffc8`.

The deck GET200 names deck1 `Alice UAT151 Citrine Deck`. [manage-deck-selected.txt](manage-deck-selected.txt) records an actual selection of that visible title and renders the original question, deck and note link. [manage-populated.png](manage-populated.png) visibly confirms this populated list. Its `1 Cards` heading is the separately tracked **UAT184**, not evidence that UAT168's numeric count/list failure remains.

Limits: this establishes the original list/count path with real populated data. It does not independently exercise card update/no-op, deck-version or asset reconciliation branches. Those remain covered by their retained automated/native evidence where applicable.

## UAT178: narrow abort → error → actual Retry200

[cram-fault.js](cram-fault.js) matches only same-origin GET `/api/v1/flashcards` with deck1, `due_status=all`, `order_by=due_at`, `limit=200`, `offset=0`; other methods continue. It calls `route.abort('failed')`, never fulfills a fabricated response or card. [Install receipt](cram-fault-installed.txt):00:40:54.448Z. The actual matching queue also has `include_workspace_items=false`. Ordinary probes with other limits and analytics remain outside the predicate.

[cram-recovery-events.txt](cram-recovery-events.txt) records matching failed requests at00:41:29.501/00:41:30.509Z, ending with `net::ERR_FAILED`. [cram-fault-settled.txt](cram-fault-settled.txt) and [cram-load-error.png](cram-load-error.png) show the selected deck, Cram mode, blank tag filter and actionable **Unable to load cram cards** / **Try again to load cards for the selected deck and tag filter.** / **Retry**. No false Cram completion or empty-filter result appears. The dashboard still shows the real New1/Total1 data, demonstrating that the error is visible alongside available summary data.

[cram-restore.js](cram-restore.js) removes only the exact stored predicate+handler; [restore receipt](cram-fault-restored.txt):00:42:21.768Z. [cram-retry-result.txt](cram-retry-result.txt) records an actual click on `flashcards-review-cram-retry`. The next matching GET starts00:42:22.444Z and returns200 at00:42:22.482Z with count/total1, the same card UUID, source note ID and version1. The error disappears and the real original question/source link become usable. The retained action sequence has no reload between unroute and Retry and preserves the selected deck/Cram/tag scope. The monitor records Flashcards requests and page errors, not a full browser navigation HAR.

Across the60 retained recovery events there are **no non-GET Flashcards requests and no pageerror events**. Thus this failed load/recovery produced no observed rating or session-end mutation. The two expected transport failures are intentional native fault evidence, not an actual server500 replay. Cached-active-card, repeated failed-Retry, loading-duration and caller-progress cases are not all natively reproduced here; the scoped automated controls remain separate.

## UAT137: visible and accessible singular agree

The recovered queue's genuine count/total1 and one actual item establish the count without mocking. [cram-retry-result.txt](cram-retry-result.txt) exposes role=status with `1 card remaining, 0 reviewed, Available now: 1`, and separately exposes the visible `1` / `card remaining` content as aria-hidden descendants. Inspected [ReviewProgress.tsx](../../apps/packages/ui/src/components/Flashcards/components/ReviewProgress.tsx#L76), whose hash matches the preparation manifest, supplies `aria-live="polite"`, `aria-atomic="true"` and the `.sr-only` status string.

[cram-schedule-enabled.txt](cram-schedule-enabled.txt) records a later actual Update schedule toggle, now checked; the same singular status, original question and source link remain. [cram-recovered-one-card.png](cram-recovered-one-card.png) visibly shows `Study queue 1 card remaining`,0 reviewed,Available now1, the same deck and note. Update schedule was OFF during fault/retry and deliberately turned ON afterward; do not claim it stayed ON through recovery or that this toggle submitted a rating.

No `1 cards remaining` appears in these accepted Study captures. This is native singular/live-region content acceptance, not an audible screen-reader test or native0/multiple-card coverage. The latter localization controls remain automated. Manage's distinct `1 Cards` copy remains open as UAT184.

## Reviewed artifact hashes

| File | SHA256 |
|---|---|
| manage-events.txt |45a565ef49c88210a4a3653eee740f73bf19b91019bce0b2392e02f6041932ad|
| manage-deck-selected.txt |4ecba14b7f61c96f74607dc7c3e471a6bc231053f5ef3801a68ad655fc415208|
| manage-populated.png |798d36c30d644e623cda1aab54c6185e6204f6422ab27a745ca6761a3a4dde6a|
| cram-fault.js |04b270efd0140b415e9f7e0aab797ba5e5788529a7a630772a644465d6cd1141|
| cram-fault-installed.txt |77ab477242924a6208c6f8eef6277b5ba6bfaec9ad422d907751bc3ca0a8f284|
| cram-fault-settled.txt |7ad91902d081f840d3ce8bc36ebf871e4ad8d51f9b567bed8cabc446eb139966|
| cram-load-error.png |18690703e4d05b68ab26c199e6505450263f9d12210614826ddcd3c9592e2ead|
| cram-restore.js |a870b85bb70945a90d5eab4c43768fcbb8bb4b2786bf618da6fcd47451845a4b|
| cram-fault-restored.txt |cbbc561d846c1418c292747ff48cb197a4499655abbbb2e1200cd27494778e4a|
| cram-retry-result.txt |9872a3cf76ed799828e66722af41e6e1301b51a3c8391ac9dfb01182f2c1d454|
| cram-schedule-enabled.txt |fe116f42078eb6db2e816d27e6e834bfd1bdfc1cfca9f3832d171e026bc6213c|
| cram-recovery-events.txt |d44d21ff4643fa15d0d8c2176398cebc84fe3cc66adbb43ab52610c246d6adfa|
| cram-recovered-one-card.png |0be7bbcbfd37b54024a05bbfb699f4a08ed2d42cc6b85485ed91b76a7f1e7381|
