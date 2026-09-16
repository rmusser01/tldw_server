# UAT128 / TASK13260.68 — Cram queue changes under its cursor

Confirmed with native cycle5 chronology and two temporary actual ReviewTab probes. Product/source/test bytes are unchanged from ab527eb3b4.

## Boundary and cause

useCramQueueQuery asks for all cards ordered by due_at (useFlashcardQueries.ts338 onward). A scheduled review invalidates Flashcards queries (940 onward), so the reviewed card moves later when its due date increases. ReviewTab uses cramQueueIndex into this refreshed list. advanceCramQueue finds the reviewed card in the captured/current order and advances to its index+1; later refresh can replace that indexed card. Re-rate uses an override of the old card, then looks it up in the refreshed list, where it can be last. That sets index equal to length and triggers automatic completion even though intended cards remain. Remaining progress independently subtracts reviewedCount from queue length, exposing the contradiction.

## Reproduction, no production changes

Private Vite transform injects tests into the existing actual mounted ReviewTab.cram-mode fixture in memory. All ordinary source files and test files stay unchanged. Controlled queue query reflects the server's documented due_at reordering after a real component rating action.

1. Queue Alpha/Bravo/Charlie, scheduled Alpha Good. With unchanged query order, Bravo remains active (PASS). Refresh to Bravo/Charlie/Alpha, consistent with Alpha's later due date, silently switches active card to Charlie (FAIL; Bravo skipped). Log cycle5-controller-uat128-probe.log:1PASS/1FAIL.
2. Same scenario, actual Re-rate Alpha then Hard. Unchanged-order control offers Bravo (PASS). Refreshed-order case renders Cram session complete/1 practiced while progress says2remaining/1reviewed (FAIL). Log cycle5-controller-uat128-rerate-probe.log:1PASS/1FAIL.
3. The first rerate probe reached the unrelated post-completion StudySuggestionsPanel without its query context and failed as a harness error. That log is preserved as cycle5-controller-uat128-rerate-harness-failure.log. Corrected probe mocks only that ancillary suggestions component, retaining actual ReviewTab, review-run hook, rate actions, cursor, progress and completion logic. Do not count the harness failure as product proof.

## Repair direction after freeze

Progress must follow card identity across query reorder instead of an index into a changing due-sorted list. Use an existing scoped identity/run pattern where possible. Two small candidates: retain the run's ordered UUIDs while reading current card values, or select pending UUIDs while recording which cards the run has already practiced. Avoid retaining old account/deck/tag data across the existing authority reset. Do not change scheduler algorithms, server review-event semantics, or imply Re-rate rolls back the previous scheduled rating.

Required behaviors: every intended card remains reachable; scheduled rating/refetch and re-rate do not skip or prematurely complete; practice-only performs no scheduling writes; the mixed-mode reset follows the existing run contract; removals/empty queues, manual End, account/deck/tag changes and delayed responses preserve current guards. Progress and completion use the same pending-card truth. Permanent regressions must use changing multi-card query order, unlike the existing single-card stable queue fixture. Independent review and bounded native control required after implementation.

This report proposes no live fix, and the native main Cram session was not mutated again after capture098.
