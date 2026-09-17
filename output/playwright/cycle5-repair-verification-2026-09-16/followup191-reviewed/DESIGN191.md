# UAT191 / TASK13260.129 — scheduled Cram interval previews

Root released the exact three-file production design after reviewing causal RED. Root owns native/card reset, tracking and integration. ReviewTab belongs to the 189/190 author and is not edited here.

## Actual cause

Native Good is rating 3, not keyboard label 4 or the deliberately invalid rating 6 used for the negative response control. The successful review saved last_reviewed_at 02:09:54.135 and due_at 02:19:54.135, a 10-minute SM-2+ learning step. The card-list response has next_intervals and scheduler_type null. Existing ReviewTab prefers supplied next_intervals but falls back to calculateIntervals, an older approximate SM-2 calculation that gives a new card 1 day. The due-review endpoint already uses _attach_scheduler_preview with the actual deck's scheduler/settings; the card-list endpoint does not.

## Small proposed repair (awaiting release)

Add an explicit optional scheduler-preview flag to the existing list endpoint, false by default. Only Cram's existing list request opts in. Reuse _attach_scheduler_preview and cache deck reads within one list response. Keep ordinary Manage/list behavior, pagination, filters and no-mutation semantics. Invalid scheduler settings in the opt-in path use the existing review-next 400 contract; ordinary list remains readable. Do not introduce a separate next-card request, parallel scheduler implementation, rating algorithm change or ReviewTab ownership change.

Expected production scope: endpoint flashcards.py, services/flashcards.ts list parameter type, and useFlashcardQueries.ts Cram request. A default-on enrichment would be fewer lines but would change every list caller and expose Manage to scheduler validation failures; the scoped opt-in avoids that unrelated behavior.

## Stages

1. Causal RED: actual in-process HTTP list/next/review with fixture SQLite and official PostgreSQL where required; compare supported rating preview labels to committed due intervals, include custom SM-2+ and FSRS, normal list/default control and read-only identity/version checks. New component controls mount actual ReviewTab and Cram query at the service boundary, prove rating mapping, schedule-off behavior and server preview precedence.
2. Minimal GREEN only after root release; retain current source snapshots before production edits. Keep all 189/190 changes separate.
3. Focused/adjacent regressions, static checks and independent review; native agreement remains root-owned and uses only disposable card D.

No native browser/service/database action has been performed by this agent. Fixture-backed route tests do not claim authentication or native provider acceptance.

## Author completion

Stages 1–2 complete: causal route12RED/4controls and mounted4RED/2controls, plus opt-in8RED/2empty-page controls, followed by the minimal three-file repair. Stage3 author verification complete:58 backend and39 frontend PASS,0skips; lint/security and compiler comparison retained. Independent review and native acceptance remain pending. See IMPLEMENTATION191.md for frozen hashes, exact commands and limits.
