# Independent native acceptance audit — UAT211

## Disposition

**PASS for the bounded native subday acceptance gate.** Recommend closing TASK13260.149 with its already recorded causal tests, independent source review and static checks. No new defect was found in the supplied duration-label sequence.

The actual Good preview is **10 min**, the persisted review gap is exactly **600 seconds**, the toast says **10 minutes**, and Manage plus the reopened editor still show **10 min** after a full reload. Unscheduled new cards retain the **—** gap control.

## Actual sequence and persisted evidence

All times are UTC on 2026-09-17. The retained identity events establish Alice/user 2 for this Flashcards window.

| Boundary | Evidence |
| --- | --- |
| Original Citrine control | `alice211-citrine-editor.txt` records an actual click on `flashcard-edit-37b10bd7-edf4-4f35-83c1-1490115d8c55`. The editor displays **Next review gap / 10 min** and Recall runs 1. `alice211-citrine-cancel.txt` records the ordinary Cancel click. |
| Isolated greeting fixture | Only greeting card `082ec63a-124e-4c47-ac23-c11a2deba70b` is patched, with `expected_version: 1` and tag `uat211-greeting`, around **07:51:15**. It becomes version 2, repetition 0. |
| Scheduled Cram | The actual tag-filter action, Update schedule click and checked switch are retained. The filtered real list contains only that greeting card and `next_intervals.good: 10 min`. `alice211-answer.txt` shows the Good button’s **10 min** preview. |
| Actual rating | `alice211-good-rating.txt` records the normal rating-3 button click. Exactly one POST `/flashcards/review` occurs at **07:53:48.318**, for the greeting UUID, rating 3, Cram context and that tag. |
| Rating result | Real **200 at 07:53:48.361** returns greeting version 3, repetitions 1, queue `learning`, `interval_days: 0`, `last_reviewed_at: 07:53:48.347`, `due_at: 08:03:48.347`. The independent timestamp subtraction equals **600 seconds**. |
| Toast | The original click snapshot says **“Saved. Next review in 10 minutes (next review gap: 10 minutes).”** It does not round the learning gap to zero days. |
| Manage and reload | The ordinary Manage click shows Next gap 10 min. `alice211-full-reload.txt` records `page.reload()`. Two real list responses at **07:54:17.985** and **07:54:18.014**, both 200, preserve greeting version 3 / repetition 1 and the exact two rating timestamps. |
| Reopened editor | `alice211-reloaded-editor.txt` records the actual greeting edit-button click after reload and shows **Next review gap / 10 min**, Recall runs 1. The later `alice211-editor-cancel.txt` records normal Cancel. |

The rating response’s new `next_intervals.good: 1 day` describes a future subsequent Good rating. It is distinct from the completed rating’s 600-second interval and does not contradict its 10-minute preview or saved-gap label.

## Preservation and new-card control

The original Citrine card `37b10bd7-edf4-4f35-83c1-1490115d8c55` remains **version 2 / repetition 1** in the pre-action and both post-reload lists. Its existing timestamps remain **00:51:18.308 → 01:01:18.308**. There is no mutation request to that UUID and no rating request for it in the audited window beginning 07:42. This is a claim about the observed card fields and requests, not an unrestricted byte-for-byte catalogue comparison.

The post-reload catalogue contains five cards. Three remain version 1, repetition 0, queue `new`, with null `last_reviewed_at` and `due_at`:

- `bfc3be29-fce9-439e-ad56-26e3fbecc30e`
- `b59f0819-47a7-4284-b78e-7805015430b0`
- `5ad3f8eb-e106-4c73-bc29-1d16431a6a63`

The settled Manage snapshot displays exactly two **Next gap 10 min** labels and three **Next gap —** labels, matching the two learning cards and these three unscheduled cards.

## Exact criteria and limits

TASK13260.149 AC1/AC2 already record persisted-timestamp formatter controls and the duration/localization repair. AC3 requires causal tests, review, applicable checks and native subday acceptance. This receipt supplies that final native clause. The task records root’s independent 96 tests / 4 files, the author’s adjacent controls and compiler/static limitations; this auditor did not rerun or relabel those checks.

Native day-scale formatting, other ratings/locales and the complete provider/tenant matrix are outside this bounded run. The run confirms request/result/display consistency; it is not a new scheduler-source audit or a whole-app clean-console claim. No model inference, direct database inspection or native mutation was performed by the auditor.

Parent handoff attributes this run to backend `47e23` / PID56113 and frontend UAT211 commit `4c233fcaa7`. Browser receipts establish the native sequence; they do not independently attest process-start source bytes. That attribution remains the runtime owner’s responsibility.

## Hash-bound packet

`input-manifest.json` records the exact **24** allowed `alice211-*.txt` inputs and official read-only task snapshot, with original byte hashes and no normalization. The additional final editor-cancel receipt arrived during the audit; the initial discovery-count mismatch is documented in `audit-harness-note.md` as a harness inventory event, not a product failure.

`audit_inputs.py` uses a fixed allowlist and verifies the single greeting rating, its unique tag, the 600-second gap, two full-reload payloads, original-card version/repetition preservation, actual UI actions and all displayed gap controls. All assertions pass. `verified-facts.json` retains reduced card/scheduling evidence; it excludes unrelated cumulative event bodies. No source, task, git, browser or runtime changes were made.
